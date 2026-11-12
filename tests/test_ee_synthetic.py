"""Earth Engine against the NumPy reference on synthetic terrain (needs Earth Engine)."""

import numpy as np
import pytest
import synthetic as syn

from topocorr import _reference as ref
from topocorr.flags import CAST_SHADOW, LOW_ILLUMINATION, NO_FIT, SELF_SHADOW, STEEP

pytestmark = pytest.mark.ee

N = 48
ZEN, AZ = 62.5, 158.0
IN = (slice(1, -1), slice(1, -1))  # drop the one-pixel margin


def _angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _hill(n, steep=False):
    """A cone plus a tilt: many slopes and aspects; ``steep`` reaches above 60 degrees."""
    cone = syn.ee_cone(62.0 if steep else 35.0, n)
    return cone.add(syn.ee_plane(12.0, 200.0, n).subtract(2000.0)).rename("elevation")


@pytest.mark.parametrize("surface", ["plane", "cone", "hill"])
def test_slope_aspect_and_cos_i_match_the_reference(surface):
    from topocorr.illumination import cos_i
    from topocorr.terrain import from_elevation

    elev = {
        "plane": syn.ee_plane(30.0, 135.0, N),
        "cone": syn.ee_cone(30.0, N),
        "hill": _hill(N),
    }[surface]
    ter = from_elevation(elev)
    d = syn.fetch(ter.addBands(cos_i(ter, ZEN, AZ)).addBands(syn.ee_lonlat()), N, margin=1)
    gamma = ref.utm_convergence(d["longitude"], d["latitude"], 43)
    assert -1.45 < gamma.mean() < -1.35  # Kalam lies 2.4 degrees west of zone 43's meridian
    s, a = ref.slope_aspect(d["elevation"], syn.RES, gamma)
    np.testing.assert_allclose(d["slope"][IN], s[IN], atol=1e-3)
    tilted = s[IN] > 1.0
    assert np.abs(_angle_diff(d["aspect"][IN], a[IN]))[tilted].max() < 1e-2
    np.testing.assert_allclose(d["cos_i"][IN], ref.cos_i(ZEN, AZ, s, a)[IN], atol=1e-5)


@pytest.mark.parametrize("facing", [0.0, 90.0, 180.0, 270.0])
def test_aspect_is_measured_from_true_north(facing):
    """Planes built from latitude and longitude face a true direction by construction."""
    from topocorr.terrain import from_elevation

    ll = syn.ee_lonlat()
    lat0, lon0 = 35.49, 72.58
    east = ll.select("longitude").subtract(lon0).multiply(111320 * np.cos(np.radians(lat0)))
    north = ll.select("latitude").subtract(lat0).multiply(110950)
    f = np.radians(facing)
    z = east.multiply(np.sin(f)).add(north.multiply(np.cos(f))).multiply(-np.tan(np.radians(30)))
    d = syn.fetch(from_elevation(z.add(2000.0)), 16, margin=1)
    assert np.abs(_angle_diff(d["aspect"][IN], facing)).max() < 0.02
    np.testing.assert_allclose(d["slope"][IN], 30.0, atol=0.1)


@pytest.mark.parametrize("lon0", [72.58, 75.0, 77.42])
def test_aspect_agrees_with_ee_terrain_aspect(lon0):
    """Planes facing true N/E/S/W west of, on and east of zone 43's central meridian (75° E).

    ``ee.Terrain.aspect`` is an independent measure of true-north aspect: the convergence
    changes sign across the zone, so a rotation in the wrong direction would differ by 2γ.
    """
    import ee

    from topocorr.terrain import from_elevation

    lat0, n = 35.49, 16
    x0 = {72.58: 279990.0, 75.0: 500010.0, 77.42: 720030.0}[lon0]
    y0 = {72.58: 3930000.0, 75.0: 3927330.0, 77.42: 3930000.0}[lon0]
    projection = ee.Projection(syn.CRS, [syn.RES, 0, x0, 0, -syn.RES, y0])
    ll = ee.Image.pixelLonLat().reproject(projection)
    east = ll.select("longitude").subtract(lon0).multiply(111320 * np.cos(np.radians(lat0)))
    north = ll.select("latitude").subtract(lat0).multiply(110950)
    bands = []
    for facing in (0, 90, 180, 270):
        f = np.radians(facing)
        z = east.multiply(np.sin(f)).add(north.multiply(np.cos(f)))
        z = z.multiply(-np.tan(np.radians(30))).add(2000.0).rename("elevation")
        bands += [
            from_elevation(z).select(["aspect"], [f"tc_{facing}"]),
            ee.Terrain.aspect(z).rename(f"ee_{facing}"),
        ]
    grid = syn.ee_grid(n, margin=1)
    grid["affineTransform"].update(translateX=x0 - syn.RES, translateY=y0 + syn.RES)
    data = ee.data.computePixels(
        {"expression": ee.Image.cat(bands), "fileFormat": "NUMPY_NDARRAY", "grid": grid}
    )
    for facing in (0, 90, 180, 270):
        tc = np.asarray(data[f"tc_{facing}"], dtype="float64")[IN]
        ref_ee = np.asarray(data[f"ee_{facing}"], dtype="float64")[IN]
        assert np.abs(_angle_diff(tc, facing)).max() < 0.02
        assert np.abs(_angle_diff(tc, ref_ee)).max() < 0.02


def test_masks_match_the_reference():
    from topocorr.masking import mask_bands
    from topocorr.terrain import from_elevation

    ter = from_elevation(_hill(N, steep=True))
    d = syn.fetch(ter.addBands(mask_bands(ter, ZEN, AZ)), N, margin=1)
    ci, s = d["cos_i"][IN], d["slope"][IN]
    expected = ref.flags(ci, s, cast_shadow=d["cast_shadow"][IN] > 0)
    # Pixels within float32 rounding of a threshold may fall either way.
    near = (np.abs(ci) < 1e-5) | (np.abs(ci - 0.1) < 1e-5) | (np.abs(s - 60.0) < 1e-3)
    got = d["tc_flag"][IN].astype("uint8")
    np.testing.assert_array_equal(got[~near], expected[~near])
    for bit in (SELF_SHADOW, LOW_ILLUMINATION, STEEP):
        assert (got & bit).any(), f"flag {bit} never set: the surface does not test it"
    np.testing.assert_array_equal(d["self_shadow"][IN] > 0, (got & SELF_SHADOW) > 0)
    np.testing.assert_array_equal(d["steep"][IN] > 0, (got & STEEP) > 0)


@pytest.mark.parametrize("sun_from", ["south", "north"])
def test_wall_casts_a_shadow_of_known_length(sun_from):
    """A 300 m wall, sun 60 degrees from zenith: the shadow is 300 tan 60 = 520 m (17 pixels)."""
    from topocorr.masking import mask_bands
    from topocorr.terrain import from_elevation

    n, height, zenith = 64, 300.0, 60.0
    azimuth = 180.0 if sun_from == "south" else 0.0
    ter = from_elevation(syn.ee_wall(height, 45.0, n))
    d = syn.fetch(ter.addBands(mask_bands(ter, zenith, azimuth)), n)
    column = n // 2
    wall = np.flatnonzero(d["elevation"][:, column] > 1000.0 + 1.0)
    shadow = (d["tc_flag"][:, column].astype(int) & (CAST_SHADOW | SELF_SHADOW)) > 0
    north_side, south_side = shadow[: wall.min()], shadow[wall.max() + 1 :]
    shaded, lit = (north_side, south_side) if sun_from == "south" else (south_side, north_side)
    length = int(shaded.sum())
    expected = height * np.tan(np.radians(zenith)) / syn.RES
    assert abs(length - expected) <= 2.0, f"shadow of {length} pixels, expected {expected:.1f}"
    assert not lit.any()


def _synthetic_image(n, m, nb, *, zenith=ZEN, azimuth=AZ):
    """Prepared-image stand-in: band b = m[b] cos i + nb[b] on the hill."""
    import ee

    from topocorr.illumination import cos_i
    from topocorr.sensors import BANDS
    from topocorr.terrain import from_elevation

    elev = _hill(n)
    ci = cos_i(from_elevation(elev), zenith, azimuth)
    bands = [ci.multiply(m[i]).add(nb[i]).rename(b) for i, b in enumerate(BANDS)]
    image = ee.Image.cat(bands).set({"tc_sun_zenith": zenith, "tc_sun_azimuth": azimuth})
    return elev, image


M = [0.10, 0.12, 0.15, 0.40, 0.30, 0.20]
NB = [0.020, 0.024, 0.020, 0.060, 0.045, 0.040]


@pytest.mark.parametrize("method", ["cosine", "c", "scsc"])
def test_corrections_match_the_reference(method):
    import ee

    import topocorr
    from topocorr.sensors import BANDS
    from topocorr.terrain import from_elevation

    elev, image = _synthetic_image(N, M, NB)
    out = topocorr.correct(
        image, method=method, dem=elev, region=syn.ee_grid_region(N), sample_size=2000
    )
    ter = from_elevation(elev)
    d = syn.fetch(out.addBands(ter), N, margin=1)
    props = ee.Dictionary(out.toDictionary()).getInfo()
    s, a = d["slope"][IN], d["aspect"][IN]
    cast = (d["tc_flag"][IN].astype(int) & CAST_SHADOW) > 0
    for i, band in enumerate(BANDS):
        rho = M[i] * d["cos_i"][IN] + NB[i]
        expected, flag, fit = ref.correct(
            rho, method=method, zenith=ZEN, azimuth=AZ, slope=s, aspect=a, cast_shadow=cast
        )
        np.testing.assert_allclose(d[band][IN], expected, rtol=1e-4, atol=1e-6)
        if method != "cosine":
            assert props[f"tc_c_{band}"] == pytest.approx(NB[i] / M[i], rel=1e-4)
            assert props[f"tc_m_{band}"] == pytest.approx(M[i], rel=1e-4)
    near = np.abs(d["cos_i"][IN]) < 1e-5
    np.testing.assert_array_equal(d["tc_flag"][IN][~near].astype("uint8"), flag[~near])
    assert props["tc_method"] == method and props["tc_dem"] == "user image"
    if method != "cosine":
        assert props["tc_fit_ok"] == 1 and props["tc_n_sample"] >= 100


def test_unusable_fit_returns_the_image_unchanged():
    import topocorr
    from topocorr.sensors import BANDS

    m = [-v for v in M]  # reflectance falls with illumination: no usable fit
    elev, image = _synthetic_image(N, m, [0.5] * 6)
    out = topocorr.correct(image, method="scsc", dem=elev, region=syn.ee_grid_region(N))
    d = syn.fetch(out, N)
    assert out.get("tc_fit_ok").getInfo() == 0
    rho = np.array([m[i] * d["cos_i"] + 0.5 for i in range(6)])
    np.testing.assert_allclose(np.array([d[b] for b in BANDS]), rho, rtol=1e-5)
    assert np.all(d["tc_flag"].astype(int) & NO_FIT)


def test_layers_do_not_depend_on_the_requested_scale():
    """Requests at 90 m return the 30 m values at the same centres (regression test)."""
    import ee

    from topocorr.masking import mask_bands
    from topocorr.terrain import terrain

    projection = ee.Projection(syn.CRS, [30, 0, 280000, 0, -30, 3936000])
    ter = terrain("glo30", projection)
    image = ter.addBands(mask_bands(ter, 62.12, 159.6))

    def grab(scale, n):
        grid = {
            "dimensions": {"width": n, "height": n},
            "affineTransform": {
                "scaleX": scale,
                "shearX": 0,
                "translateX": 280000,
                "shearY": 0,
                "scaleY": -scale,
                "translateY": 3936000,
            },
            "crsCode": syn.CRS,
        }
        data = ee.data.computePixels(
            {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid}
        )
        return {k: np.asarray(data[k], dtype="float64") for k in data.dtype.names}

    fine, coarse = grab(30, 60), grab(90, 20)
    for band in ("elevation", "slope", "aspect", "cos_i"):
        np.testing.assert_allclose(coarse[band], fine[band][1::3, 1::3], atol=1e-4, err_msg=band)
    flag_coarse = coarse["tc_flag"].astype(int)
    flag_fine = fine["tc_flag"][1::3, 1::3].astype(int)
    other = ~CAST_SHADOW & 0xFF
    np.testing.assert_array_equal(flag_coarse & other, flag_fine & other)
    # hillShadow works on its own grid: cast-shadow edges may move by a pixel or two.
    assert np.mean((flag_coarse & CAST_SHADOW) == (flag_fine & CAST_SHADOW)) > 0.9
