"""compare(): the local half (NumPy) on synthetic tables with a known illumination response."""

import csv
import json

import numpy as np
import pytest
from test_stack import _simulated_model

import topocorr
from topocorr import _reference as ref
from topocorr.comparison import BANDS, METHODS, SCALE

N_POINTS, N_IMAGES = 300, 60


def _tables(drift=0.0, seed=0):
    """Tree points (strata 11, 12) on 10-40° slopes; reflectance = flat x F with the model's c."""
    rng = np.random.default_rng(seed)
    model, _ = _simulated_model(np.random.default_rng(3))
    years = 2015 + np.sort(rng.uniform(0, 6, N_IMAGES))
    zenith = 43 + 15 * np.cos(2 * np.pi * (years - 2015)) - drift * (years - years[0])
    azimuth = np.full(N_IMAGES, 150.0)
    codes = rng.choice([11, 12], N_POINTS)
    slope = rng.uniform(10, 40, N_POINTS)
    aspect = rng.uniform(0, 360, N_POINTS)
    points = {"pid": np.arange(N_POINTS, dtype=float), "stratum": codes.astype(float),
              "slope": slope, "aspect": aspect, "loss": np.zeros(N_POINTS)}  # fmt: skip
    ms = (years - 1970) * 365.25 * 86400 * 1000
    images = [
        {"img": t, "zenith": float(zenith[t]), "azimuth": float(azimuth[t]), "time": float(ms[t])}
        for t in range(N_IMAGES)
    ]
    ci = ref.cos_i(zenith[:, None], azimuth[:, None], slope[None, :], aspect[None, :])
    flat = {"nir": 0.30, "red": 0.04}
    rows = {k: [] for k in ("pid", "img", *BANDS, "tc_flag")}
    for t in range(N_IMAGES):
        for k in range(N_POINTS):
            if ci[t, k] <= 0.05:
                continue
            rows["pid"].append(k)
            rows["img"].append(t)
            for b in BANDS:
                if b in flat:
                    c = float(model.c(b, int(codes[k]), zenith[t]))
                    value = flat[b] * ref.illumination_factor(ci[t, k], zenith[t], c)
                else:
                    value = 0.1
                rows[b].append(round(value * SCALE + rng.normal(0, 20)))
            rows["tc_flag"].append(0)
    raw = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    return {"images": images, "points": points, "raw": raw, "scale": SCALE, "model": model}


@pytest.fixture(scope="module")
def result():
    rows = topocorr.compare(
        _tables(), methods=["none", "cosine", "c", "scsc", "scsc-stack", "c-stack", "se-stack"]
    )
    return {
        r["method"]: r
        for r in (rows if isinstance(rows, list) else rows.reset_index().to_dict("records"))
    }


def test_the_stack_correction_removes_the_illumination_dependence(result):
    assert result["none"]["illumination_r"] > 0.5
    assert result["none"]["sunlit_shaded_gap"] > 0.2
    # the data follow (cos i + c) / (cos z + c): the C correction takes them back exactly
    for m in ("c", "c-stack", "se-stack"):
        assert abs(result[m]["illumination_r"]) < 0.05
    # SCS+C normalises to cos s cos z + c instead, which leaves a little dependence on slope
    for m in ("scsc", "scsc-stack"):
        assert result[m]["illumination_r"] < 0.5 * result["none"]["illumination_r"]
        assert abs(result[m]["sunlit_shaded_gap"]) < 0.05
    # the cosine method ignores c > 0 and over-corrects
    assert result["cosine"]["illumination_r"] < 0
    for method, row in result.items():
        assert row["points"] == N_POINTS and row["images"] == N_IMAGES
        if method != "cosine":  # the cosine method pushes some values above 1
            assert row["out_of_range"] == 0.0


def _rows(rows):
    return {
        r["method"]: r
        for r in (rows if isinstance(rows, list) else rows.reset_index().to_dict("records"))
    }


def test_predicted_drift_left_follows_each_method():
    methods = ["none", "cosine", "c", "scsc-stack", "c-stack", "se-stack"]
    rows = _rows(topocorr.compare(_tables(drift=1.0), methods=methods))
    none = rows["none"]["predicted_drift_left"]
    assert none > 0.001 and rows["none"]["predicted_drift_high"] > 0.5  # the drift trend itself
    # C with the stack c removes the model's illumination exactly; per-image C nearly
    assert rows["c-stack"]["predicted_drift_left"] < 1e-6
    assert rows["c-stack"]["predicted_drift_high"] == 0.0
    assert rows["c"]["predicted_drift_left"] < 0.05 * none
    # SCS+C keeps (cos s cos z + c) / (cos z + c), which changes with z on slopes
    assert 1e-6 < rows["scsc-stack"]["predicted_drift_left"] < 0.2 * none
    assert rows["se-stack"]["predicted_drift_left"] < 0.2 * none
    assert rows["cosine"]["predicted_drift_left"] > 1e-4  # ignores c > 0: over-corrects


def test_without_drift_nothing_is_predicted():
    for r in _rows(topocorr.compare(_tables(drift=0.0), methods=["none", "scsc-stack"])).values():
        assert r["predicted_drift_left"] < 0.0002 and r["predicted_drift_high"] == 0.0


def test_default_methods_include_c_stack():
    from topocorr.comparison import DEFAULT_METHODS

    assert "c-stack" in DEFAULT_METHODS and set(DEFAULT_METHODS) <= set(METHODS)


def test_tables_round_trip_through_csv(tmp_path):
    t = _tables()
    t["model"].to_json(tmp_path / "m_model.json")
    for name, cols in (("raw", t["raw"]), ("points", t["points"])):
        with (tmp_path / f"m_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            w.writerows(zip(*cols.values(), strict=True))
    manifest = {"files": {"raw": "m_raw.csv", "points": "m_points.csv"}, "model": "m_model.json",
                "settings": {"scale": SCALE}, "images": t["images"]}  # fmt: skip
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    from_files = topocorr.compare(tmp_path / "m.json", methods=["none", "scsc-stack"])
    in_memory = topocorr.compare(t, methods=["none", "scsc-stack"])
    if isinstance(from_files, list):
        assert from_files == in_memory
    else:
        assert from_files.equals(in_memory)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"methods": ["minnaert"]}, "unknown method"),
        ({"index": "savi"}, "unknown index"),
        ({"harmonics": 4}, "harmonics"),
    ],  # fmt: skip
)
def test_compare_checks_its_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        topocorr.compare(_tables(), **kwargs)


def test_compare_needs_a_model():
    t = _tables()
    t.pop("model")
    with pytest.raises(TypeError, match="stack model"):
        topocorr.compare(t)


def test_methods_list():
    assert set(METHODS) == {"none", "cosine", "c", "scsc", "scsc-stack", "c-stack", "se-stack"}
