"""The artefact test in NumPy: validation tests 1 (synthetic truth) and 2 (flat control)."""

import numpy as np
import pytest

from topocorr import _reference as ref

YEARS = 2000 + np.sort(np.random.default_rng(0).uniform(0, 12, 160))  # 12 years, ~13 images a year
SLOPE, ASPECT = 35.0, 330.0  # a steep, mostly north-facing slope: sensitive to the sun's height
C = {"nir": 0.25, "red": 0.05}
FLAT = {"nir": 0.30, "red": 0.04}


def _sun(years, drift):
    """Seasonal zenith plus a drift (degrees per year), as orbit drift or a sensor switch gives."""
    season = 45 + 20 * np.cos(2 * np.pi * (years - 2000))  # winter high, summer low
    return season + drift * (years - years[0]), np.full(years.size, 150.0)


def _series(drift, greening):
    zen, azi = _sun(YEARS, drift)
    ci = ref.cos_i(zen, azi, np.full(YEARS.size, SLOPE), np.full(YEARS.size, ASPECT))
    veg = 1 + greening * (YEARS - YEARS[0])  # multiplies NIR only: NDVI rises
    bands = {
        "nir": FLAT["nir"] * veg * ref.illumination_factor(ci, zen, C["nir"]),
        "red": FLAT["red"] * ref.illumination_factor(ci, zen, C["red"]),
    }
    c = {b: np.full(YEARS.size, v) for b, v in C.items()}
    return bands, ci, zen, c


def _diagnose(drift, greening):
    bands, ci, zen, c = _series(drift, greening)
    return ref.diagnose_pixel(YEARS, bands, ci, zen, c, index_name="ndvi")


def test_pure_illumination_drift_is_all_artefact():
    out = _diagnose(drift=-0.8, greening=0.0)
    assert abs(out["observed_trend"]) > 0.001
    assert out["artefact_share"] == pytest.approx(1.0, abs=0.05)
    assert out["risk_class"] == 3
    assert out["geometry_trend"] > 0  # the sun rises over the years: the slope is lit better


def _flat_ground_trend(greening, index_name="ndvi"):
    """The truth: the index trend of the same vegetation on flat ground (no illumination effect)."""
    veg = 1 + greening * (YEARS - YEARS[0])
    flat = {"nir": FLAT["nir"] * veg, "red": np.full(YEARS.size, FLAT["red"])}
    return ref.theil_sen(YEARS, ref.deseasonalise(YEARS, ref.index(index_name, flat)))


def _true_share(out, greening, index_name="ndvi"):
    """Artefact share by definition: the part of the observed trend not seen on flat ground."""
    obs = out["observed_trend"]
    return (obs - _flat_ground_trend(greening, index_name)) / obs


def test_pure_greening_has_no_drift_artefact():
    """Without drift the share is small; what remains is real: on this steep north-facing slope the
    seasonal illumination damps the visible NDVI trend (share about -0.2, against flat ground)."""
    out = _diagnose(drift=0.0, greening=0.01)
    assert out["observed_trend"] > 0.001
    assert out["artefact_share"] == pytest.approx(_true_share(out, 0.01), abs=0.02)
    assert abs(out["artefact_share"]) < 0.25
    assert out["risk_class"] in (1, 2)


def test_mixture_recovers_the_known_share():
    """Drift and browning push NDVI the same way; the share is the drift's part of the trend.

    When the two oppose each other, the observed trend nearly cancels and the share is unstable
    (|share| > 1 is then possible); the risk class uses |share| for that reason.
    """
    out = _diagnose(drift=-0.8, greening=-0.004)
    truth = _true_share(out, -0.004)
    assert 0.2 < truth < 0.8
    assert out["artefact_share"] == pytest.approx(truth, abs=0.1)


# Landsat 5-like orbit drift: the overpass comes later, so the sun is higher (zenith falls) and
# further towards the afternoon (azimuth grows), as measured on the validation tiles (1995-2006:
# zenith -4 to -7 degrees, azimuth +11 to +16 degrees). Forest c from the tiles' stack models:
# red c above NIR c, so brighter illumination raises NDVI.
FOREST_C = {"nir": 0.3, "red": 0.8}


def _drift_pixel(aspect, greening=0.0, index_name="nirv", zenith_drift=-0.5, azimuth_drift=1.3):
    zen = 45 + 20 * np.cos(2 * np.pi * (YEARS - 2000)) + zenith_drift * (YEARS - YEARS[0])
    azi = 135 + azimuth_drift * (YEARS - YEARS[0])
    ci = ref.cos_i(zen, azi, np.full(YEARS.size, 30.0), np.full(YEARS.size, float(aspect)))
    veg = 1 + greening * (YEARS - YEARS[0])
    bands = {
        "nir": FLAT["nir"] * veg * ref.illumination_factor(ci, zen, FOREST_C["nir"]),
        "red": FLAT["red"] * ref.illumination_factor(ci, zen, FOREST_C["red"]),
    }
    c = {b: np.full(YEARS.size, v) for b, v in FOREST_C.items()}
    return ref.diagnose_pixel(YEARS, bands, ci, zen, c, index_name=index_name)


@pytest.mark.parametrize("index_name", ["nirv", "ndvi"])
def test_orbit_drift_has_an_east_west_signature(index_name):
    """Validation test 1, azimuth drift: false greening on west-, browning on east-facing slopes."""
    west, east = _drift_pixel(270, index_name=index_name), _drift_pixel(90, index_name=index_name)
    north, south = _drift_pixel(0, index_name=index_name), _drift_pixel(180, index_name=index_name)
    assert west["observed_trend"] > 0 > east["observed_trend"]
    for out in (west, east):
        assert out["artefact_share"] == pytest.approx(1.0, abs=0.05)
    # the zenith and azimuth drifts nearly cancel on north- and south-facing slopes
    for out in (north, south):
        assert abs(out["observed_trend"]) < 0.1 * abs(west["observed_trend"])
    if index_name == "nirv":  # NIR-based: the drift alone exceeds the trend threshold
        assert west["risk_class"] == east["risk_class"] == 3


def test_orbit_drift_mixed_with_greening_recovers_the_share():
    out = _drift_pixel(270, greening=0.004)
    truth = _true_share(out, 0.004, "nirv")
    assert 0.5 < truth < 0.9
    assert out["artefact_share"] == pytest.approx(truth, abs=0.1)
    assert out["risk_class"] == 3


def test_flat_ground_has_no_artefact():
    """Validation test 2: on flat ground cos i = cos θz, so F = 1 and nothing is predicted."""
    zen, _ = _sun(YEARS, drift=-0.8)
    ci = np.cos(np.radians(zen))
    bands = {"nir": FLAT["nir"] * (1 + 0.01 * (YEARS - YEARS[0])), "red": np.full(YEARS.size, 0.04)}
    c = {b: np.full(YEARS.size, v) for b, v in C.items()}
    out = ref.diagnose_pixel(YEARS, bands, ci, zen, c)
    assert out["artefact_trend"] == pytest.approx(0.0, abs=1e-12)
    assert out["risk_class"] == 1


def test_no_trend_and_not_assessable_classes():
    assert ref.risk_class(0.9, 0.0001, 0.001) == 4
    assert ref.risk_class(float("nan"), 0.01, 0.001) == 0
    assert [ref.risk_class(s, 0.01, 0.001) for s in (0.1, 0.3, -0.7)] == [1, 2, 3]


def test_deseasonalise_removes_the_annual_cycle():
    y = 0.5 + 0.1 * np.cos(2 * np.pi * YEARS) + 0.03 * np.sin(4 * np.pi * YEARS)
    np.testing.assert_allclose(ref.deseasonalise(YEARS, y), 0.5, atol=1e-10)


@pytest.mark.parametrize("name", ["ndvi", "nbr", "ndmi", "evi", "nirv"])
def test_indices(name):
    b = {"nir": np.array(0.4), "red": np.array(0.05), "blue": np.array(0.03),
         "swir1": np.array(0.2), "swir2": np.array(0.1)}  # fmt: skip
    value = float(ref.index(name, b))
    assert -1 < value < 2.5
    with pytest.raises(ValueError, match="unknown index"):
        ref.index("savi", b)


def test_theil_sen_without_spread_is_nan():
    assert np.isnan(ref.theil_sen(np.ones(4), np.arange(4.0)))
