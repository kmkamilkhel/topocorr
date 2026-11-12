"""Smoke tests on real scenes over the Kalam test tile, upper Swat (needs Earth Engine).

Tile: UTM 43N, E 268-292 km, N 3922-3946 km (24 x 24 km).
"""

import numpy as np
import pytest

import topocorr
from topocorr import _reference as ref
from topocorr import sensors

pytestmark = pytest.mark.ee


@pytest.fixture(scope="module")
def kalam():
    import ee

    return ee.Geometry.Rectangle(
        [268000, 3922000, 292000, 3946000], ee.Projection("EPSG:32643"), False
    )


def test_collections_are_harmonised(kalam):
    import ee

    out = {}
    for sensor, start, end in [
        ("landsat", "2023-06-01", "2023-08-01"),
        ("sentinel2", "2023-06-01", "2023-07-01"),
    ]:
        images = topocorr.collection(sensor, kalam, start, end, max_cloud=30)
        first = ee.Image(images.first())
        out[sensor] = ee.Dictionary(
            {
                "n": images.size(),
                "bands": first.bandNames(),
                "props": first.toDictionary(),
                "nir_median": first.select("nir")
                .reduceRegion(ee.Reducer.median(), kalam, 120)
                .get("nir"),
            }
        )
    info = ee.Dictionary(out).getInfo()
    for d in info.values():
        assert d["n"] > 0
        assert d["bands"] == list(sensors.BANDS)
        p = d["props"]
        name = p["tc_sensor"]
        assert (p["tc_sun_zenith"], p["tc_sun_azimuth"]) == pytest.approx(
            sensors.sun_angles(p, name)
        )
        assert 0.05 < d["nir_median"] < 0.6  # surface reflectance, not digital numbers


def test_real_dem_terrain_matches_the_reference():
    """GLO-30 resampled to a 30 m UTM grid in the forest above Kalam."""
    import synthetic as syn

    from topocorr.terrain import terrain

    n = 40
    d = syn.fetch(terrain("glo30", syn.ee_projection()).addBands(syn.ee_lonlat()), n, margin=1)
    gamma = ref.utm_convergence(d["longitude"], d["latitude"], 43)
    s, a = ref.slope_aspect(d["elevation"], syn.RES, gamma)
    inner = (slice(1, -1), slice(1, -1))
    assert np.nanmax(d["slope"][inner]) > 30  # real mountain terrain
    np.testing.assert_allclose(d["slope"][inner], s[inner], atol=1e-3)
    tilted = s[inner] > 1
    diff = (d["aspect"][inner] - a[inner] + 180) % 360 - 180
    assert np.abs(diff[tilted]).max() < 0.01


def _scene(sensor, scene_id, kalam):
    import ee

    images = topocorr.collection(sensor, kalam, "2023-01-01", "2024-01-01")
    return ee.Image(images.filter(ee.Filter.eq("system:index", scene_id)).first())


def _illumination_check(kalam, scene_id):
    """Fit SCS+C on forest; correlation of forest NIR with cos i before and after."""
    import ee

    scene = _scene("L8", scene_id, kalam)
    forest = ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(10)
    corrected = topocorr.correct(scene, method="scsc", region=kalam)  # tree cover by default
    usable = corrected.select("tc_flag").bitwiseAnd(topocorr.flags.EXCLUDED).eq(0)

    def r(image):
        pair = image.select("cos_i").addBands(image.select("nir")).updateMask(forest.And(usable))
        return pair.reduceRegion(ee.Reducer.pearsonsCorrelation(), kalam, 90).get("correlation")

    self_shadow = corrected.select("tc_flag").bitwiseAnd(topocorr.flags.SELF_SHADOW).neq(0)
    info = ee.Dictionary(
        {
            "r_before": r(scene.addBands(corrected.select("cos_i"))),
            "r_after": r(corrected),
            "props": corrected.toDictionary(),
            "self_shadow_forest": self_shadow.updateMask(forest)
            .reduceRegion(ee.Reducer.mean(), kalam, 90)
            .get("tc_flag"),
        }
    ).getInfo()
    p = info["props"]
    fits = {b: (round(p[f"tc_c_{b}"], 3), p[f"tc_fit_ok_{b}"]) for b in sensors.BANDS}
    print()
    print(
        f"{scene_id} zenith {p['tc_sun_zenith']:.1f}: r(nir, cos i) {info['r_before']:.3f} -> "
        f"{info['r_after']:.3f}; self-shadow in forest {info['self_shadow_forest']:.1%}; "
        f"c and fit per band {fits}"
    )
    return info, p


def test_scsc_in_summer_fits_every_band_and_removes_the_illumination_effect(kalam):
    info, p = _illumination_check(kalam, "LC08_150036_20230606")
    assert p["tc_fit_ok"] == 1 and p["tc_n_sample"] >= 500
    assert all(p[f"tc_c_{b}"] > 0 for b in sensors.BANDS)
    assert info["r_before"] > 0.3
    assert abs(info["r_after"]) < 0.5 * info["r_before"]
    assert info["self_shadow_forest"] < 0.01  # high summer sun


def test_scsc_in_winter_corrects_nir_and_leaves_bands_without_a_usable_fit(kalam):
    """Mid-December: only the NIR fit is usable on this scene; other bands stay unchanged."""
    info, p = _illumination_check(kalam, "LC08_150035_20231215")
    assert p["tc_fit_ok_nir"] == 1 and p["tc_c_nir"] > 0
    assert info["r_before"] > 0.5  # strong illumination effect in winter forest
    assert abs(info["r_after"]) < 0.5 * info["r_before"]
    unfitted = [b for b in sensors.BANDS if p[f"tc_fit_ok_{b}"] == 0]
    assert p["tc_fit_ok"] == (0 if unfitted else 1)
    assert 0.1 < info["self_shadow_forest"] < 0.45  # offline estimate for 22 December: 26 %


def test_all_dems_are_available(kalam):
    import ee

    from topocorr.terrain import DEMS, elevation

    centre = kalam.centroid(1)
    values = ee.Dictionary(
        {
            name: elevation(name).reduceRegion(ee.Reducer.first(), centre, 30).get("elevation")
            for name in DEMS
        }
    ).getInfo()
    for name, value in values.items():
        assert 1500 < value < 5500, name


def test_masks_and_correct_map_over_a_collection(kalam):
    import ee

    from topocorr.masking import MASK_BANDS

    images = topocorr.collection("L9", kalam, "2023-06-01", "2023-07-01").limit(2)
    masks = topocorr.masks(images)
    corrected = topocorr.correct(images, method="cosine")
    single = topocorr.masks(ee.Image(images.first()), dem="srtm")
    info = ee.Dictionary(
        {
            "n_images": images.size(),
            "mask_bands": ee.Image(masks.first()).bandNames(),
            "mask_props": ee.Image(masks.first()).toDictionary(["tc_sensor", "tc_sun_zenith"]),
            "corrected_bands": ee.Image(corrected.first()).bandNames(),
            "method": ee.Image(corrected.first()).get("tc_method"),
            "single_bands": single.bandNames(),
        }
    ).getInfo()
    assert info["n_images"] >= 1
    assert info["mask_bands"] == list(MASK_BANDS) == info["single_bands"]
    assert info["mask_props"]["tc_sensor"] == "L9"
    assert info["corrected_bands"] == [*sensors.BANDS, "cos_i", "tc_flag"]
    assert info["method"] == "cosine"


def test_stack_model_on_a_year_of_landsat_8(kalam):
    """Fit the stack model on 2023 Landsat 8 over the tile and correct with it."""
    import ee

    images = topocorr.collection("L8", kalam, "2023-01-01", "2024-01-01", max_cloud=20)
    model = topocorr.fit(images, region=kalam, samples_per_stratum=300, min_images=5)
    trees = [code for code in model.codes if code // 10 == 10]
    assert trees, "no tree strata on the tile"
    for code in trees:
        curve = model.curve("nir", code)
        assert curve is not None and 0 <= curve(45.0) < 1.0
    assert "trees/25-40" in model.summary()
    corrected = topocorr.correct(images.limit(2), model, method="scsc-stack", region=kalam)
    first = ee.Image(corrected.first())
    assert first.get("tc_model").getInfo() == model.id
    nir = ee.Dictionary(first.get("tc_c_nir")).getInfo()
    assert nir["trees/25-40"] == pytest.approx(
        model.c("nir", "trees/25-40", first.get("tc_sun_zenith").getInfo()), rel=1e-6
    )
