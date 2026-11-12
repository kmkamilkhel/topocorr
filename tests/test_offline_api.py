"""Sensor tables, flags, DEM names and argument checks (offline)."""

import json
from pathlib import Path

import pytest

import topocorr
from topocorr import flags, sensors, terrain

SCENES = json.loads((Path(__file__).parent / "data" / "scene_properties.json").read_text())


# ---- sensors ----


@pytest.mark.parametrize("name", ["L5", "L7", "L8", "L9", "S2"])
def test_band_tables_match_the_live_collections(name):
    """Recorded from one scene of each collection over Kalam (tests/data)."""
    s = sensors.SENSORS[name]
    scene = SCENES[name]
    assert scene["collection"] == s.collection
    assert set(s.native_bands) <= set(scene["bands"])
    masks = {"SCL"} if name == "S2" else {"QA_PIXEL", "QA_RADSAT"}
    assert masks <= set(scene["bands"])


@pytest.mark.parametrize("name", ["L5", "L7", "L8", "L9"])
def test_landsat_sun_angles(name):
    p = SCENES[name]["properties"]
    zenith, azimuth = sensors.sun_angles(p, name)
    assert zenith == pytest.approx(90.0 - p["SUN_ELEVATION"])
    assert azimuth == pytest.approx(p["SUN_AZIMUTH"])


def test_sentinel2_sun_angles():
    p = SCENES["S2"]["properties"]
    assert sensors.sun_angles(p, "S2") == pytest.approx(
        (p["MEAN_SOLAR_ZENITH_ANGLE"], p["MEAN_SOLAR_AZIMUTH_ANGLE"])
    )


def test_harmonised_bands():
    assert sensors.SENSORS["L5"].native_bands == [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B7",
    ]
    assert sensors.SENSORS["L8"].native_bands == [
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
    ]
    assert sensors.SENSORS["S2"].bands["nir"] == "B8A"
    landsat = sensors.SENSORS["L9"]
    assert (landsat.scale, landsat.offset) == (2.75e-05, -0.2)
    assert sensors.SENSORS["S2"].scale == 1e-4


def test_sun_below_horizon_and_azimuth_wrap():
    with pytest.raises(ValueError, match="below the horizon"):
        sensors.sun_angles({"SUN_ELEVATION": -1.0, "SUN_AZIMUTH": 100.0}, "L8")
    assert sensors.sun_angles({"SUN_ELEVATION": 30.0, "SUN_AZIMUTH": 370.0}, "L8")[1] == 10.0


def test_resolve():
    assert sensors.resolve("Landsat") == ("L5", "L7", "L8", "L9")
    assert sensors.resolve("sentinel2") == ("S2",)
    assert sensors.resolve("l7") == ("L7",)
    with pytest.raises(ValueError, match="unknown sensor"):
        sensors.resolve("modis")


def test_masked_classes_keep_terrain_shadow():
    assert 2 not in sensors.S2_SCL_MASKED  # SCL 2 "dark area" includes terrain shadow
    assert set(sensors.LANDSAT_QA_BITS.values()) == set(range(6))


def test_collection_checks_arguments_before_calling_earth_engine():
    with pytest.raises(ValueError, match="unknown sensor"):
        topocorr.collection("aster", None, "2020-01-01", "2021-01-01")
    with pytest.raises(ValueError, match="percentage"):
        topocorr.collection("landsat", None, "2020-01-01", "2021-01-01", max_cloud=150)


# ---- flags ----


def test_flag_bits_are_distinct_powers_of_two():
    bits = list(flags.FLAGS.values())
    assert len(set(bits)) == len(bits)
    assert all(b & (b - 1) == 0 for b in bits)
    assert flags.EXCLUDED == flags.CAST_SHADOW | flags.SELF_SHADOW | flags.STEEP


def test_decode():
    assert flags.decode(1 + 8) == ["cast_shadow", "no_fit"]
    for bad in (-1, 64):
        with pytest.raises(ValueError):
            flags.decode(bad)


# ---- DEMs ----


def test_dem_sources():
    assert terrain.source("GLO30").asset == "COPERNICUS/DEM/GLO30_2024_1"
    assert terrain.source("USGS/SRTMGL1_003").name == "srtm"
    user = terrain.source("projects/p/assets/dtm")
    assert (user.asset, user.collection) == ("projects/p/assets/dtm", False)
    assert terrain.dem_name("aw3d30") == "aw3d30"
    with pytest.raises(ValueError, match="unknown DEM"):
        terrain.source("gmted")
    with pytest.raises(TypeError):
        terrain.source(30)


# ---- argument checks of the public functions ----


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"method": "minnaert"}, ValueError, "unknown method"),
        ({"method": "scsc-stack"}, ValueError, "needs a stack model"),
        ({"method": "se-stack", "model": object()}, TypeError, "StackModel"),
        ({"method": "c", "steep": 95}, ValueError, "steep"),
        ({"method": "c", "min_slope": 90}, ValueError, "min_slope"),
        ({"method": "c", "sample_size": 50, "min_pixels": 100}, ValueError, "sample_size"),
        ({"method": "c", "fit_mask": "forest"}, ValueError, "fit_mask"),
    ],
)
def test_correct_checks_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        topocorr.correct(None, **kwargs)


def test_per_image_methods_take_no_model():
    with pytest.raises(ValueError, match="only by the stack methods"):
        topocorr.correct(None, object(), method="scsc")


def test_masks_checks_steep():
    with pytest.raises(ValueError, match="steep"):
        topocorr.masks(None, steep=0)


def test_busy_earth_engine_requests_are_retried():
    import ee

    from topocorr import model

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ee.EEException("Too many concurrent aggregations.")
        return "done"

    assert model._request(flaky, wait=0) == "done" and len(calls) == 3
    with pytest.raises(ee.EEException, match="concurrent"):
        model._request(lambda: (_ for _ in ()).throw(ee.EEException("Too many concurrent")), 2, 0)

    def broken():
        raise ee.EEException("Image.select: band not found")

    with pytest.raises(ee.EEException, match="not found"):
        model._request(broken, wait=0)
    assert model._is_busy(ee.EEException("Quota exceeded")) and not model._is_busy(ValueError())


def test_band_bits_and_mixed_pairs():
    import numpy as np

    from topocorr.flags import BAND_BITS, mixed

    assert BAND_BITS == {"blue": 1, "green": 2, "red": 4, "nir": 8, "swir1": 16, "swir2": 32}
    values = np.array([0, BAND_BITS["nir"], BAND_BITS["red"], BAND_BITS["nir"] | BAND_BITS["red"]])
    np.testing.assert_array_equal(mixed(values, ["nir", "red"]), [False, True, True, False])
    assert not mixed(63, ["nir", "red", "swir1"])
    with pytest.raises(ValueError, match="unknown band"):
        mixed(0, ["nir", "B8"])


@pytest.mark.parametrize("bands", [("nir", "B4"), ("nir", "nir")])
def test_normalized_difference_checks_bands(bands):
    with pytest.raises(ValueError, match="two different bands"):
        topocorr.normalized_difference(None, *bands)


def test_collection_checks_cloud_buffer():
    with pytest.raises(ValueError, match="cloud_buffer"):
        topocorr.collection("L8", None, "2020-01-01", "2021-01-01", cloud_buffer=-1)
