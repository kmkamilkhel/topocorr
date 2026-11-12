"""The stack model in Earth Engine on a synthetic stack with a known c(θz) (needs Earth Engine).

Eight images of the synthetic hill under different suns. Band b of image t is
``m_t (cos i + c_b(θz_t))``: stage 1 must return c_b(θz_t) exactly, stage 2 the curve c_b, and
the corrections have closed forms.
"""

import copy

import numpy as np
import pytest
import synthetic as syn

pytestmark = pytest.mark.ee

N = 48
IN = (slice(1, -1), slice(1, -1))
SUNS = [(26.0, 150.0), (32.0, 152.0), (38.0, 155.0), (44.0, 157.0), (50.0, 159.0),
        (55.0, 160.0), (60.0, 162.0), (64.0, 164.0)]  # fmt: skip
BRIGHTNESS = [0.30, 0.42, 0.25, 0.36, 0.48, 0.28, 0.40, 0.33]
BASE = [0.30, 0.26, 0.34, 0.12, 0.18, 0.22]


def _c(b, zenith):
    t = (np.asarray(zenith) - 45.0) / 20.0
    return BASE[b] + 0.05 * t + 0.03 * t * t


def _hill():
    cone = syn.ee_cone(35.0, N)
    return cone.add(syn.ee_plane(12.0, 200.0, N).subtract(2000.0)).rename("elevation")


@pytest.fixture(scope="module")
def stack_images():
    import ee

    from topocorr.illumination import cos_i
    from topocorr.sensors import BANDS
    from topocorr.terrain import from_elevation

    elev = _hill()
    ter = from_elevation(elev)
    images = []
    for t, ((zenith, azimuth), m) in enumerate(zip(SUNS, BRIGHTNESS, strict=True)):
        ci = cos_i(ter, zenith, azimuth)
        bands = [ci.add(float(_c(b, zenith))).multiply(m).rename(n) for b, n in enumerate(BANDS)]
        images.append(
            ee.Image.cat(bands).set(
                {
                    "tc_sun_zenith": zenith,
                    "tc_sun_azimuth": azimuth,
                    "tc_sensor": "L8",
                    "system:time_start": 1_600_000_000_000 + t * 86_400_000 * 16,
                }
            )
        )
    return elev, ee.ImageCollection.fromImages(images)


@pytest.fixture(scope="module")
def model(stack_images):
    import topocorr

    elev, images = stack_images
    return topocorr.fit(
        images,
        dem=elev,
        strata=None,
        region=syn.ee_grid_region(N),
        samples_per_stratum=300,
        min_pixels=50,
        min_images=5,
        progress=False,
    )


def test_fit_recovers_the_known_curves(model):
    from topocorr.sensors import BANDS

    assert set(model.codes) <= {11, 12, 13} and len(model.codes) >= 2
    assert model.settings["sensor_family"] == "OLI" and len(model.images) == 8
    zenith = np.array([s[0] for s in SUNS])
    for b, band in enumerate(BANDS):
        for code in model.codes:
            fits = model.image_fits(band, code)
            ok = fits["usable"]
            np.testing.assert_allclose(fits["c"][ok], _c(b, zenith[ok]), atol=2e-4)
            # Few gentle slopes on the hill: 5-25 may fall back to all slopes.
            assert model.used[band][code] == (code if ok.sum() >= 5 else 10)
            grid = np.linspace(30.0, 60.0, 7)
            np.testing.assert_allclose(model.c(band, code, grid), _c(b, grid), atol=1e-3)
    assert any(model.used["nir"][code] == code for code in model.codes)


@pytest.mark.parametrize("method", ["scsc-stack", "se-stack"])
def test_stack_corrections_have_their_closed_form(model, stack_images, method):
    """ρ = m (cos i + c): SCS+C gives m (cos s cos θz + c), statistical-empirical m (cos θz + c)."""
    import ee

    import topocorr
    from topocorr.sensors import BANDS
    from topocorr.terrain import from_elevation

    elev, images = stack_images
    t = 6  # zenith 60: some self-shadow and cast shadow
    image = ee.Image(images.toList(8).get(t))
    out = topocorr.correct(image, model, method=method, dem=elev, region=syn.ee_grid_region(N))
    d = syn.fetch(out.addBands(from_elevation(elev).select("slope")), N, margin=1)
    zenith, m = SUNS[t][0], BRIGHTNESS[t]
    flag = d["tc_flag"][IN].astype(int)
    corrected = (flag & topocorr.flags.EXCLUDED) == 0
    corrected &= (flag & (topocorr.flags.NO_FIT | topocorr.flags.NO_MODEL)) == 0
    slope = d["slope"][IN]
    assert corrected.mean() > 0.5
    all_bands = sum(topocorr.flags.BAND_BITS.values())
    np.testing.assert_array_equal(d["tc_corrected"][IN][corrected], all_bands)
    excluded = (flag & topocorr.flags.EXCLUDED) != 0
    np.testing.assert_array_equal(d["tc_corrected"][IN][excluded], 0)
    for b, band in enumerate(BANDS):
        if method == "scsc-stack":
            code = np.where(slope < 25, 11, np.where(slope < 40, 12, 13))
            c = np.vectorize(lambda k, b=band: model.c(b, int(k), zenith))(code)
            expected = m * (np.cos(np.radians(slope)) * np.cos(np.radians(zenith)) + c)
            tol = 2e-3 * m  # the curve, not the image's own c
        else:
            expected = np.full(slope.shape, m * (np.cos(np.radians(zenith)) + _c(b, zenith)))
            tol = 1e-4
        np.testing.assert_allclose(d[band][IN][corrected], expected[corrected], atol=tol)
        # Flagged pixels keep their input value.
        original = m * (d["cos_i"][IN] + _c(b, zenith))
        np.testing.assert_allclose(d[band][IN][~corrected], original[~corrected], rtol=1e-4)
    props = out.toDictionary().getInfo()
    assert props["tc_method"] == method and props["tc_model"] == model.id
    if method == "se-stack":
        m_fit = [v for v in props["tc_m_nir"].values() if v != -1]  # -1: too few pixels
        assert m_fit and all(v == pytest.approx(m, rel=1e-3) for v in m_fit)
    else:
        assert set(props["tc_c_nir"]) == {model.name(c) for c in model.codes}


def test_user_strata_survive_saving(stack_images, tmp_path):
    """User strata are stored as an Earth Engine expression and rebuilt after loading."""
    import ee

    import topocorr
    from topocorr.model import model_strata
    from topocorr.terrain import from_elevation

    elev, images = stack_images
    east, _ = syn.ee_offsets(N)
    halves = east.gte(0).add(1).toInt()  # 1 west half, 2 east half
    model = topocorr.fit(
        images,
        dem=elev,
        strata=halves,
        slope_classes=(5, 60),
        region=syn.ee_grid_region(N),
        samples_per_stratum=300,
        min_pixels=50,
        min_images=5,
        progress=False,
        cache=tmp_path / "fit-cache.json",
    )
    assert model.codes == (11, 21) and model.name(21) == "2/5-60"
    assert model_strata(model) is halves  # in memory: the image itself
    again = topocorr.fit(  # every image comes from the cache: no new requests
        images.filter(ee.Filter.neq("tc_sensor", "none")),
        dem=elev,
        strata=halves,
        slope_classes=(5, 60),
        region=syn.ee_grid_region(N),
        samples_per_stratum=300,
        min_pixels=50,
        min_images=5,
        progress=True,
        cache=tmp_path / "fit-cache.json",
    )
    assert again.id == model.id
    loaded = topocorr.load_model(model.to_json(tmp_path / "model.json"))
    image = ee.Image(images.first())
    out = topocorr.correct(image, loaded, method="scsc-stack", dem=elev)
    d = syn.fetch(out.addBands(from_elevation(elev).select("slope")), N, margin=1)
    flag = d["tc_flag"][IN].astype(int)
    assert ((flag & topocorr.flags.NO_MODEL) == 0).mean() > 0.9  # strata rebuilt from the file


def test_fit_and_correct_check_their_arguments(model, stack_images):
    import ee

    import topocorr
    from topocorr.model import strata_image

    elev, images = stack_images
    named = copy.copy(model)
    named.settings = {**model.settings, "dem": "glo30"}
    with pytest.raises(ValueError, match="DEM"):
        topocorr.correct(images, named, method="scsc-stack", dem="srtm")
    for kwargs, match in [
        ({"degree": 5}, "degree"),
        ({"slope_classes": (30, 10)}, "slope_classes"),
        ({"slope_classes": (-5, 30)}, "between 0 and 90"),
        ({"samples_per_stratum": 10}, "samples_per_stratum"),
        ({"min_images": 1}, "min_images"),
        ({"strata": "landcover"}, "unknown strata"),
        ({"strata": 3}, "strata must be"),
    ]:
        with pytest.raises((ValueError, TypeError), match=match):
            topocorr.fit(images, **kwargs)
    with pytest.raises(ValueError, match="empty"):
        topocorr.fit(images.filter(ee.Filter.eq("tc_sensor", "none")), dem=elev, progress=False)
    with pytest.raises(ValueError, match="no usable pixels"):
        topocorr.fit(
            images.limit(2),
            dem=elev,
            strata=None,
            slope_classes=(80, 85),
            region=syn.ee_grid_region(N),
            progress=False,
        )
    stratum = strata_image("worldcover", ee.Image.constant(30), (5, 25, 40, 60))
    assert stratum.bandNames().getInfo() == ["stratum"]


def test_fit_halves_the_batch_when_earth_engine_is_busy(stack_images, monkeypatch):
    import ee

    import topocorr
    from topocorr import model as model_module

    elev, images = stack_images
    sizes = []
    real = model_module._request

    def busy_once(call, *args, **kwargs):
        sizes.append(1)
        if len(sizes) == 1:
            raise ee.EEException("Too many concurrent aggregations.")
        return real(call, *args, **kwargs)

    monkeypatch.setattr(model_module, "_request", busy_once)
    fitted = topocorr.fit(
        images.limit(4),
        dem=elev,
        strata=None,
        region=syn.ee_grid_region(N),
        samples_per_stratum=200,
        min_pixels=50,
        min_images=2,
        batch_size=4,
        progress=False,
    )
    assert len(fitted.images) == 4 and len(sizes) == 3  # one refused, then two batches of 2
