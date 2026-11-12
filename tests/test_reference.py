"""The formulas on synthetic terrain with a known answer (offline)."""

import numpy as np
import pytest
from synthetic import RES, cone, plane

from topocorr import _reference as ref
from topocorr.flags import CAST_SHADOW, LOW_ILLUMINATION, NO_FIT, SELF_SHADOW, STEEP

INNER = (slice(1, -1), slice(1, -1))


# ---- slope and aspect ----


@pytest.mark.parametrize("slope", [10.0, 30.0, 50.0])
@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_plane_slope_and_aspect(slope, aspect):
    s, a = ref.slope_aspect(plane(slope, aspect), RES)
    np.testing.assert_allclose(s[INNER], slope, atol=1e-9)
    diff = (a[INNER] - aspect + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(diff, 0.0, atol=1e-9)


def test_aspect_convention_east_rising_plane_faces_west():
    z = np.add.outer(np.zeros(5), np.arange(5) * 10.0)  # elevation grows towards the east
    _, a = ref.slope_aspect(z, RES)
    assert a[2, 2] == pytest.approx(270.0)


def test_aspect_convention_north_facing_plane():
    z = np.add.outer(np.arange(5) * 10.0, np.zeros(5))  # elevation grows towards the south
    _, a = ref.slope_aspect(z, RES)
    assert a[2, 2] % 360.0 == pytest.approx(0.0)


def test_edges_are_nan_and_flat_has_zero_slope():
    s, _ = ref.slope_aspect(np.full((4, 4), 500.0), RES)
    assert np.isnan(s[0]).all() and np.isnan(s[:, -1]).all()
    assert (s[INNER] == 0).all()


def test_cone_has_every_aspect():
    _, a = ref.slope_aspect(cone(30.0, n=41), RES)
    sectors = np.unique(((a[INNER] + 22.5) % 360 // 45).astype(int))
    assert sectors.tolist() == list(range(8))


# ---- cos i ----


@pytest.mark.parametrize("zenith", [20.0, 45.0, 62.5, 75.0])
@pytest.mark.parametrize("azimuth", [120.0, 160.0])
def test_cos_i_flat_is_cos_zenith(zenith, azimuth):
    assert ref.cos_i(zenith, azimuth, 0.0, 77.0) == pytest.approx(np.cos(np.radians(zenith)))


@pytest.mark.parametrize("zenith,slope", [(30.0, 10.0), (60.0, 35.0), (70.0, 50.0)])
def test_cos_i_towards_away_and_across_the_sun(zenith, slope):
    az = 150.0
    rad = np.radians
    assert ref.cos_i(zenith, az, slope, az) == pytest.approx(np.cos(rad(zenith - slope)))
    assert ref.cos_i(zenith, az, slope, az + 180) == pytest.approx(np.cos(rad(zenith + slope)))
    across = np.cos(rad(zenith)) * np.cos(rad(slope))
    assert ref.cos_i(zenith, az, slope, az + 90) == pytest.approx(across)
    assert ref.cos_i(zenith, az, slope, az - 90) == pytest.approx(across)


def test_cos_i_sun_at_zenith_is_cos_slope():
    slopes = np.array([0.0, 15.0, 40.0, 70.0])
    np.testing.assert_allclose(
        ref.cos_i(0.0, 123.0, slopes, np.array([0, 90, 200, 300.0])), np.cos(np.radians(slopes))
    )


def test_cos_i_mean_around_a_cone_is_cos_zenith_cos_slope():
    aspects = np.linspace(0.0, 360.0, 3600, endpoint=False)
    mean = ref.cos_i(55.0, 140.0, np.full_like(aspects, 30.0), aspects).mean()
    assert mean == pytest.approx(np.cos(np.radians(55.0)) * np.cos(np.radians(30.0)))


def test_cos_i_is_bounded():
    rng = np.random.default_rng(1)
    ci = ref.cos_i(
        rng.uniform(0, 89.9),
        rng.uniform(0, 360),
        rng.uniform(0, 90, 1000),
        rng.uniform(0, 360, 1000),
    )
    assert np.all(np.abs(ci) <= 1 + 1e-12)


# ---- flags ----


def test_self_shadow_starts_where_the_slope_exceeds_ninety_minus_zenith():
    zenith, az = 60.0, 160.0
    away = az + 180.0
    slopes = np.array([29.0, 29.9, 30.1, 31.0])  # the boundary is at 90 - 60 = 30 degrees
    ci = ref.cos_i(zenith, az, slopes, np.full(4, away))
    f = ref.flags(ci, slopes)
    assert [bool(v & SELF_SHADOW) for v in f] == [False, False, True, True]


def test_low_illumination_band():
    ci = np.array([-0.05, 0.0, 0.05, 0.0999, 0.1, 0.5])
    f = ref.flags(ci, np.zeros(6))
    assert [(v & LOW_ILLUMINATION) != 0 for v in f] == [False, False, True, True, False, False]
    assert not np.any((f & LOW_ILLUMINATION != 0) & (f & SELF_SHADOW != 0))


def test_steep_threshold():
    f = ref.flags(np.full(3, 0.5), np.array([59.9, 60.0, 60.1]), steep=60.0)
    assert [(v & STEEP) != 0 for v in f] == [False, False, True]


def test_cast_shadow_is_not_set_on_self_shadowed_pixels():
    f = ref.flags(np.array([0.4, -0.2]), np.zeros(2), cast_shadow=np.array([True, True]))
    assert f[0] & CAST_SHADOW and not f[1] & CAST_SHADOW and f[1] & SELF_SHADOW


def test_no_fit_flag_on_every_pixel():
    assert np.all(ref.flags(np.full(4, 0.5), np.zeros(4), no_fit=True) & NO_FIT)


# ---- corrections ----

ZEN, AZ = 62.5, 158.0


def _cone_geometry(slope=30.0, n=81):
    s, a = ref.slope_aspect(cone(slope, n), RES)
    s, a = s[INNER], a[INNER]
    return s, a, ref.cos_i(ZEN, AZ, s, a)


def _hill_geometry(n=81):
    """Terrain with many slopes and aspects: a cone plus a tilt."""
    z = cone(35.0, n) + plane(12.0, 200.0, n)
    s, a = ref.slope_aspect(z, RES)
    s, a = s[INNER], a[INNER]
    return s, a, ref.cos_i(ZEN, AZ, s, a)


def _unflagged(slope, ci):
    return ~ref.excluded(ref.flags(ci, slope))


def test_cosine_removes_a_lambertian_illumination_effect():
    s, a, ci = _hill_geometry()
    rho = 0.4 * ci
    out, flag, fit = ref.correct(rho, method="cosine", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    keep = _unflagged(s, ci)
    assert fit is None
    np.testing.assert_allclose(out[keep], 0.4 * np.cos(np.radians(ZEN)), rtol=1e-12)


def test_cosine_overcorrects_when_part_of_the_light_is_diffuse():
    s, a, ci = _hill_geometry()
    rho = 0.3 * ci + 0.05
    out, *_ = ref.correct(rho, method="cosine", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    keep = _unflagged(s, ci) & (s > 5)
    dim = out[keep & (ci < 0.3)].mean()
    bright = out[keep & (ci > 0.7)].mean()
    assert dim > 1.2 * bright  # dim slopes become too bright


@pytest.mark.parametrize("method", ["cosine", "c", "scsc"])
def test_flat_pixels_are_unchanged(method):
    s, a, ci = _hill_geometry()
    rho = 0.3 * ci + 0.06
    flat_s = np.zeros_like(s)
    flat_ci = ref.cos_i(ZEN, AZ, flat_s, a)
    out = {
        "cosine": ref.cosine(rho, flat_ci, ZEN),
        "c": ref.c_correction(rho, flat_ci, ZEN, 0.2),
        "scsc": ref.scsc(rho, flat_ci, ZEN, flat_s, 0.2),
    }[method]
    np.testing.assert_allclose(out, rho, rtol=1e-12)


def test_c_fit_recovers_c_exactly_without_noise():
    s, a, ci = _hill_geometry()
    rho = 0.3 * ci + 0.06
    out, flag, fit = ref.correct(rho, method="c", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    assert fit.c == pytest.approx(0.2, rel=1e-9)
    keep = ~ref.excluded(flag)
    expected = 0.3 * np.cos(np.radians(ZEN)) + 0.06
    np.testing.assert_allclose(out[keep], expected, rtol=1e-9)


def test_c_fit_with_noise_is_within_two_percent_and_flat_in_cos_i():
    s, a, ci = _hill_geometry(n=161)
    rng = np.random.default_rng(42)
    rho = 0.3 * ci + 0.06 + rng.normal(0, 0.003, ci.shape)
    out, flag, fit = ref.correct(rho, method="c", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    assert fit.c == pytest.approx(0.2, rel=0.02)
    keep = ~ref.excluded(flag)
    slope_after = np.polyfit(ci[keep], out[keep], 1)[0]
    assert abs(slope_after) < 0.01  # was 0.3 before the correction


def test_scsc_is_constant_across_aspects_within_a_slope_class():
    s, a, ci = _cone_geometry(30.0)
    rho = 0.3 * ci + 0.06
    out, flag, fit = ref.correct(rho, method="scsc", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    keep = ~ref.excluded(flag) & (np.abs(s - 30.0) < 0.5)
    assert keep.sum() > 500
    assert np.ptp(out[keep]) / out[keep].mean() < 0.01
    expected = 0.3 * np.cos(np.radians(s[keep])) * np.cos(np.radians(ZEN)) + 0.06
    np.testing.assert_allclose(out[keep], expected, rtol=1e-9)


def test_scsc_with_c_zero_is_scs_and_large_c_changes_nothing():
    s, a, ci = _hill_geometry()
    rho = 0.25 * ci + 0.02
    keep = _unflagged(s, ci)
    scs = rho * np.cos(np.radians(s)) * np.cos(np.radians(ZEN)) / ci
    np.testing.assert_allclose(ref.scsc(rho, ci, ZEN, s, 0.0)[keep], scs[keep], rtol=1e-12)
    np.testing.assert_allclose(ref.scsc(rho, ci, ZEN, s, 1e9)[keep], rho[keep], rtol=1e-6)


def test_excluded_pixels_keep_their_value_low_illumination_is_corrected():
    s, a, ci = _hill_geometry()
    rho = 0.3 * ci + 0.06
    out, flag, _ = ref.correct(rho, method="c", zenith=ZEN, azimuth=AZ, slope=s, aspect=a)
    excluded = ref.excluded(flag)
    low = (flag & LOW_ILLUMINATION) != 0
    assert excluded.any() and low.any()
    np.testing.assert_array_equal(out[excluded], rho[excluded])
    assert np.all(out[low] != rho[low])


@pytest.mark.parametrize(
    "rho_of_ci,min_pixels",
    [
        (lambda ci: 0.4 - 0.1 * ci, 100),  # m < 0
        (lambda ci: 0.3 * ci - 0.01, 100),  # c < 0
        (lambda ci: 0.3 * ci + 0.06, 10**9),  # too few pixels
    ],
)
@pytest.mark.parametrize("method", ["c", "scsc"])
def test_unusable_fit_leaves_the_image_unchanged_and_flagged(rho_of_ci, min_pixels, method):
    s, a, ci = _hill_geometry()
    rho = rho_of_ci(ci)
    out, flag, fit = ref.correct(
        rho, method=method, zenith=ZEN, azimuth=AZ, slope=s, aspect=a, min_pixels=min_pixels
    )
    assert not fit.usable(min_pixels)
    np.testing.assert_array_equal(out, rho)
    assert np.all(flag & NO_FIT)


def test_degenerate_fit():
    fit = ref.fit_c(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.5, 0.5]))
    assert np.isnan(fit.c) and not fit.usable(2)


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        ref.correct(np.ones(3), method="minnaert", zenith=30, azimuth=0, slope=0, aspect=0)
