"""The stack model's two stages in NumPy, on simulated stacks with a known c(θz)."""

import json

import numpy as np
import pytest

from topocorr import stack
from topocorr.stack import StackModel, fit_curve, stage1


def _sums(x, y):
    return np.array([x.size, x.sum(), (x * x).sum(), y.sum(), (x * y).sum(), (y * y).sum()])


def _true_c(z, base=0.15, slope=0.04, curvature=0.02):
    t = (np.asarray(z) - 45.0) / 20.0
    return base + slope * t + curvature * t * t


def test_stage1_matches_least_squares_and_its_variance():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 1.0, 400)
    y = 0.3 * x + 0.05 + rng.normal(0, 0.01, x.size)
    fit = stage1(_sums(x, y))
    design = np.column_stack([x, np.ones_like(x)])
    (m, n), res, *_ = np.linalg.lstsq(design, y, rcond=None)
    sigma2 = res[0] / (x.size - 2)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    c = n / m
    grad = np.array([-n / m**2, 1 / m])  # dc/dm, dc/dn
    assert fit["m"] == pytest.approx(m) and fit["n"] == pytest.approx(n)
    assert fit["c"] == pytest.approx(c)
    assert fit["var_c"] == pytest.approx(grad @ cov @ grad, rel=1e-6)
    assert fit["count"] == x.size


def test_stage1_is_undefined_without_pixels_or_spread():
    x = np.full(10, 0.5)
    out = stage1(np.stack([_sums(x, x), np.zeros(6), _sums(x[:2], x[:2])]))
    assert np.isnan(out["c"]).all() and np.isnan(out["m"]).all()


def test_curve_recovers_a_known_c_with_noise_and_outliers():
    rng = np.random.default_rng(2)
    z = rng.uniform(20, 65, 100)
    var = rng.uniform(1e-5, 4e-4, z.size)
    c = _true_c(z) + rng.normal(0, np.sqrt(var)) + rng.normal(0, 0.02, z.size)
    bad = rng.choice(z.size, 10, replace=False)  # 10 % gross outliers (snow, haze)
    c[bad] += rng.choice([-0.3, 0.4], bad.size)
    curve = fit_curve(z, c, var)
    grid = np.linspace(22, 63, 20)
    assert np.abs(curve(grid) - _true_c(grid)).max() < 0.01
    assert curve.degree == 2 and curve.n_images == 100
    assert curve.n_outliers >= 10
    assert 0.012 < curve.tau < 0.03  # the scatter added beyond sampling noise (0.02)
    assert np.all(curve.se(grid) > 0) and curve.se(45.0) < 0.01


def test_curve_degree_follows_the_zenith_range_and_holds_at_the_edges():
    z = np.array([40.0, 42.0, 45.0, 47.0, 50.0])
    curve = fit_curve(z, _true_c(z), np.full(z.size, 1e-4))
    assert curve.degree == 0  # 10 degree range: a constant
    two = fit_curve(np.array([30.0, 60.0]), np.array([0.1, 0.2]), np.full(2, 1e-4), degree=2)
    assert two.degree == 0  # two images: never more parameters than images allow
    line = fit_curve(np.linspace(30, 70, 12), _true_c(np.linspace(30, 70, 12)), np.ones(12))
    assert line(80.0) == pytest.approx(line(70.0)) and line(10.0) == pytest.approx(line(30.0))
    assert stack.Curve.from_dict(line.to_dict()) == line


@pytest.mark.parametrize(
    "z,c,v",
    [([1.0, 2.0], [1.0], [1.0, 1.0]), ([], [], []), ([[1.0]], [[1.0]], [[1.0]])],
)
def test_curve_checks_its_input(z, c, v):
    with pytest.raises(ValueError):
        fit_curve(np.asarray(z), np.asarray(c), np.asarray(v))


def test_robust_helpers_handle_degenerate_input():
    assert stack._mad(np.array([1.0, 2.0]), np.zeros(2)) == 0.0
    assert stack._between_image_variance(np.ones(3), np.ones(3), np.ones(3), 3) == 0.0
    assert stack._between_image_variance(np.zeros(5), np.ones(5), np.ones(5), 1) == 0.0


def test_sensor_families():
    assert stack.sensor_family(["L5", "L7"]) == "TM/ETM+"
    assert stack.sensor_family(["S2"]) == "MSI"
    with pytest.raises(ValueError, match="sensor families"):
        stack.sensor_family(["L7", "L8"])


# ---- a simulated stack: 2 land covers x 3 slope classes, 40 images ------------------------------

BANDS = ("nir", "red")
COVERS = {"1": "trees", "2": "grass"}
EDGES = [5.0, 25.0, 40.0, 60.0]
SETTINGS = {
    "strata": "worldcover",
    "cover_names": COVERS,
    "slope_classes": EDGES,
    "degree": 2,
    "samples_per_stratum": 300,
    "min_pixels": 100,
    "min_images": 10,
    "steep": 60.0,
    "dem": "glo30",
    "seed": 0,
    "sensor_family": "OLI",
}


def _simulated_model(rng):
    """Codes 11-13 (trees), 21-23 (grass), 31 (an empty cover, all images too small).

    Trees 40-60 has too few pixels in most images: it must fall back to trees/all slopes.
    """
    codes = [11, 12, 13, 21, 22, 23, 31]
    zenith = rng.uniform(22, 64, 40)
    images = [
        {"id": f"img{t}", "time": t, "zenith": float(z), "azimuth": 150.0, "sensor": "L8"}
        for t, z in enumerate(zenith)
    ]
    sums = np.zeros((zenith.size, len(codes), len(BANDS), 6))
    for t, z in enumerate(zenith):
        brightness = rng.uniform(0.2, 0.5)
        for k, code in enumerate(codes):
            n = 300
            if code == 13 and t % 5:
                n = 40  # usable in only 8 images, fewer than min_images
            if code == 31:
                n = 20
            x = rng.uniform(0.05, 1.0, n)
            for b in range(len(BANDS)):
                c = _true_c(z, base=0.1 + 0.05 * b + 0.02 * (code // 10))
                y = brightness * (x + c) + rng.normal(0, 0.003, n)
                sums[t, k, b] = _sums(x, y)
    return StackModel.build(BANDS, codes, images, sums, SETTINGS), zenith


def test_model_fits_strata_with_fallback_and_not_assessable():
    model, zenith = _simulated_model(np.random.default_rng(3))
    used = model.used["nir"]
    assert used[11] == 11 and used[23] == 23
    assert used[13] == 10  # falls back to trees over all slopes
    assert used[31] is None  # not assessable
    grid = np.linspace(zenith.min(), zenith.max(), 9)
    for b, band in enumerate(BANDS):
        for code in (11, 12, 13, 21, 22, 23):
            truth = _true_c(grid, base=0.1 + 0.05 * b + 0.02 * (code // 10))
            assert np.abs(model.c(band, code, grid) - truth).max() < 0.01, (band, code)
    assert np.isnan(model.c("nir", 31, 45.0))
    assert np.isnan(model.c("nir", 31, np.array([40.0, 50.0]))).all()
    assert model.curve("nir", "trees/all slopes") is model.curves["nir"][10]
    assert model.curve("nir", "trees/40-60") is model.curves["nir"][10]
    assert model.name(12) == "trees/25-40" and model.code("grass/5-25") == 21


def test_model_queries_and_summary():
    model, _ = _simulated_model(np.random.default_rng(4))
    fits = model.image_fits("red", 12)
    assert fits["usable"].all() and fits["zenith"].size == 40
    pooled = model.image_fits("red", "trees/all slopes")
    assert (pooled["count"] > fits["count"]).all()
    text = model.summary()
    assert "not assessable" in text and "trees/all slopes" in text and model.id in text
    with pytest.raises(KeyError, match="band"):
        model.curve("swir1", 11)
    with pytest.raises(KeyError, match="stratum"):
        model.curve("nir", 99)
    with pytest.raises(KeyError, match="stratum"):
        model.code("forest/5-25")


def test_model_saves_and_loads(tmp_path):
    model, _ = _simulated_model(np.random.default_rng(5))
    path = model.to_json(tmp_path / "model.json")
    loaded = stack.load_model(path)
    assert loaded.id == model.id
    assert loaded.curves == model.curves and loaded.used == model.used
    d = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not a topocorr"):
        StackModel.from_dict({**d, "format": "other"})
    with pytest.raises(ValueError, match="newer"):
        StackModel.from_dict({**d, "version": stack.MODEL_VERSION + 1})


def test_stratum_codes():
    assert stack.stratum_code(10, 2) == 102
    assert stack.split_code(1003) == (100, 3)


def test_sun_sides_and_both_sides():
    assert stack.sun_sides([150.0, 210.0, 60.0, 300.0]) == (2, 2)
    model, _ = _simulated_model(np.random.default_rng(5))
    assert not model.sun_both_sides  # every image has the sun south of the tile (azimuth 150)
    for im in model.images:
        im["azimuth"] = 40.0
    assert not model.sun_both_sides  # southern hemisphere: the sun always north
    model.images[0]["azimuth"] = 120.0  # 1 of 40 images south: below the 5 % share
    assert not model.sun_both_sides
    model.images[1]["azimuth"] = 120.0  # 2 of 40: 5 %
    assert model.sun_both_sides


def test_correct_defaults_to_scsc_stack_and_falls_back_near_the_equator():
    from topocorr.correction import _choose_method

    model, _ = _simulated_model(np.random.default_rng(6))
    assert _choose_method(None, model) == ("scsc-stack", model)
    assert _choose_method("scsc", None) == ("scsc", None)
    with pytest.raises(ValueError, match="needs a stack model"):
        _choose_method(None, None)
    for im in model.images[:10]:
        im["azimuth"] = 60.0  # the sun north of the tile in a quarter of the images
    with pytest.warns(UserWarning, match="per-image SCS"):
        assert _choose_method(None, model) == ("scsc", None)
    with pytest.warns(UserWarning, match="per-image SCS"):
        assert _choose_method("scsc-stack", model) == ("scsc", None)
    assert _choose_method("se-stack", model) == ("se-stack", model)  # per-image m: not affected


def test_diagnose_warns_and_records_when_the_sun_is_on_both_sides():
    from topocorr.artefact import _warn_sun_both_sides

    model, _ = _simulated_model(np.random.default_rng(7))
    assert _warn_sun_both_sides({"TM/ETM+": model}) == 0
    other, _ = _simulated_model(np.random.default_rng(8))
    for im in other.images[:10]:
        im["azimuth"] = 60.0
    with pytest.warns(UserWarning, match="tc_sun_both_sides = 1"):
        assert _warn_sun_both_sides({"TM/ETM+": model, "OLI": other}) == 1
