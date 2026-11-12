"""diagnose() in Earth Engine against the NumPy reference on a synthetic stack (needs Earth Engine).

Six years, six images a year, on the synthetic hill: the sun's zenith has a seasonal cycle and a
drift of -1 degree a year (as orbit drift or a sensor switch gives), and NIR has a greening trend.
Band b of image t is m_t (cos i + c_b(θz)) times the vegetation factor, so the stack model
recovers c_b and diagnose() has an exact NumPy twin.
"""

import numpy as np
import pytest
import synthetic as syn

from topocorr import _reference as ref
from topocorr.artefact import OUTPUT_BANDS

pytestmark = pytest.mark.ee

N = 40
IN = (slice(1, -1), slice(1, -1))
T0 = 2015.0
TIMES = T0 + (np.arange(36) + 0.37) / 6.0  # six images a year
ZENITH = 45 + 12 * np.cos(2 * np.pi * TIMES) - 1.0 * (TIMES - T0)
AZIMUTH = np.full(TIMES.size, 152.0)
BRIGHTNESS = 0.8 + 0.2 * np.sin(np.arange(TIMES.size))  # image-to-image atmosphere
BASE = [0.30, 0.26, 0.06, 0.20, 0.18, 0.22]  # c per band at 45 degrees
GREENING = {"nir": 0.02}  # per year


def _c(b, zenith):
    t = (np.asarray(zenith) - 45.0) / 20.0
    return BASE[b] + 0.04 * t + 0.02 * t * t


def _hill():
    cone = syn.ee_cone(32.0, N)
    return cone.add(syn.ee_plane(10.0, 200.0, N).subtract(2000.0)).rename("elevation")


def _stack(elev):
    import ee

    from topocorr.illumination import cos_i
    from topocorr.sensors import BANDS
    from topocorr.terrain import from_elevation

    ter = from_elevation(elev)
    images = []
    for time, zen, azi, m in zip(TIMES, ZENITH, AZIMUTH, BRIGHTNESS, strict=True):
        ci = cos_i(ter, float(zen), float(azi))
        bands = []
        for b, name in enumerate(BANDS):
            veg = 1 + GREENING.get(name, 0.0) * (time - T0)
            bands.append(ci.add(float(_c(b, zen))).multiply(float(m * veg * 0.3)).rename(name))
        ms = (time - 1970) * 365.25 * 86400 * 1000
        images.append(
            ee.Image.cat(bands).set(
                {
                    "tc_sun_zenith": float(zen),
                    "tc_sun_azimuth": float(azi),
                    "tc_sensor": "L8",
                    "system:time_start": ms,
                }  # fmt: skip
            )
        )
    return ee.ImageCollection.fromImages(images)


@pytest.fixture(scope="module")
def setup():
    import topocorr

    elev = _hill()
    images = _stack(elev)
    model = topocorr.fit(
        images, dem=elev, strata=None, region=syn.ee_grid_region(N), samples_per_stratum=300,
        min_pixels=40, min_images=8, progress=False,
    )  # fmt: skip
    return elev, images, model


def test_diagnose_matches_the_reference(setup):
    import ee

    import topocorr
    from topocorr.masking import mask_bands
    from topocorr.terrain import from_elevation

    elev, images, model = setup
    out = topocorr.diagnose(images, model, dem=elev, min_obs=12)
    ter = from_elevation(elev)
    flags = ee.Image.cat(
        [
            mask_bands(ter, float(z), float(a)).select(["tc_flag"], [f"f{t}"])
            for t, (z, a) in enumerate(zip(ZENITH, AZIMUTH, strict=True))
        ]  # fmt: skip
    )
    d = syn.fetch(out.addBands(ter).addBands(flags), N, margin=1)
    slope, aspect = d["slope"][IN], d["aspect"][IN]
    code = np.where(slope < 25, 11, np.where(slope < 40, 12, 13))
    got = {k: d[k][IN] for k in OUTPUT_BANDS}
    checked = 0
    for i, j in zip(*np.nonzero(np.ones_like(slope, bool)), strict=True):
        usable = np.array([(int(d[f"f{t}"][IN][i, j]) & topocorr.flags.EXCLUDED) == 0
                           for t in range(TIMES.size)])  # fmt: skip
        if usable.sum() < 12:
            assert got["risk_class"][i, j] == 0
            continue
        s_ij, a_ij = np.full(TIMES.size, slope[i, j]), np.full(TIMES.size, aspect[i, j])
        ci = ref.cos_i(ZENITH, AZIMUTH, s_ij, a_ij)
        bands, c = {}, {}
        for b, name in ((3, "nir"), (2, "red")):
            veg = 1 + GREENING.get(name, 0.0) * (TIMES - T0)
            bands[name] = (ci + _c(b, ZENITH)) * BRIGHTNESS * veg * 0.3
            c[name] = model.c(name, int(code[i, j]), ZENITH)
        exp = ref.diagnose_pixel(TIMES, bands, ci, ZENITH, c, usable=usable)
        assert got["observed_trend"][i, j] == pytest.approx(exp["observed_trend"], abs=2e-4)
        assert got["artefact_trend"][i, j] == pytest.approx(exp["artefact_trend"], abs=2e-4)
        assert got["geometry_trend"][i, j] == pytest.approx(exp["geometry_trend"], abs=2e-4)
        assert got["n_obs"][i, j] == usable.sum()
        checked += 1
    assert checked > 0.8 * slope.size
    share = got["artefact_share"]
    lo, hi = got["share_low"], got["share_high"]
    ok = np.isfinite(share) & (np.abs(got["observed_trend"]) > 0.001)
    assert np.all(lo[ok] <= share[ok] + 1e-6) and np.all(share[ok] <= hi[ok] + 1e-6)
    # The drift creates a real artefact on these slopes: several pixels reach medium or high risk.
    assert np.isin(got["risk_class"], (2, 3)).mean() > 0.2


def test_flat_ground_has_no_artefact(setup):
    """Validation test 2 in Earth Engine: F = 1 on flat ground, so no artefact trend."""
    import ee

    import topocorr

    _, _, model = setup
    flat = ee.Image.constant(1500.0).rename("elevation").reproject(syn.ee_projection())
    out = topocorr.diagnose(_stack(flat), model, dem=flat, min_obs=12)
    d = syn.fetch(out, 16, margin=1)
    np.testing.assert_allclose(d["artefact_trend"][IN], 0.0, atol=1e-6)
    # The sun itself still drifts: on flat ground cos i = cos θz, whose trend is the geometry trend.
    cos_z = np.cos(np.radians(ZENITH))
    expected = ref.theil_sen(TIMES, ref.deseasonalise(TIMES, cos_z))
    np.testing.assert_allclose(d["geometry_trend"][IN], expected, atol=1e-5)
    assert np.isin(d["risk_class"][IN], (1, 4)).all()


def test_diagnose_checks_its_arguments(setup):
    import topocorr

    _, images, model = setup
    for kwargs, error, match in [
        ({"index": "savi"}, ValueError, "unknown index"),
        ({"harmonics": 4}, ValueError, "harmonics"),
        ({"min_obs": 5}, ValueError, "min_obs"),
    ]:
        with pytest.raises(error, match=match):
            topocorr.diagnose(images, model, **kwargs)
    with pytest.raises(TypeError, match="StackModel"):
        topocorr.diagnose(images, object())


def test_wide_bracket_never_removes_central_observations(setup):
    """Regression: c - 2 se < 0 once masked the lower bracket and, with it, the central trends."""
    import copy
    import dataclasses

    import topocorr

    elev, images, model = setup
    wide = copy.deepcopy(model)
    for curves in wide.curves.values():
        for code, curve in curves.items():
            cov = tuple(tuple(v * 1e6 for v in row) for row in curve.cov)
            curves[code] = dataclasses.replace(curve, cov=cov)
    d0 = syn.fetch(topocorr.diagnose(images, model, dem=elev, min_obs=12), N, margin=1)
    d1 = syn.fetch(topocorr.diagnose(images, wide, dem=elev, min_obs=12), N, margin=1)
    for band in ("observed_trend", "artefact_trend", "n_obs"):
        np.testing.assert_allclose(d1[band][IN], d0[band][IN], atol=1e-7, equal_nan=True)
    share, risk = d1["artefact_share"][IN], d1["risk_class"][IN]
    assert np.all(risk[~np.isfinite(share)] == 0)  # no class without a defined share
    ok = np.isfinite(share)
    assert np.all(d1["share_low"][IN][ok] <= share[ok] + 1e-6)
    assert np.all(share[ok] <= d1["share_high"][IN][ok] + 1e-6)
