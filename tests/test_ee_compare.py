"""compare_export() on the synthetic stack of the diagnose tests: the manifest (Earth Engine)."""

import json

import pytest
import synthetic as syn
from test_ee_diagnose import TIMES, setup  # noqa: F401 (fixture)

import topocorr

pytestmark = pytest.mark.ee


def test_export_writes_the_manifest_and_the_model(setup, tmp_path):  # noqa: F811
    elev, images, model = setup
    manifest = tmp_path / "site_compare.json"
    out = topocorr.compare_export(
        images, model, region=syn.ee_grid_region(40), n_points=20, prefix="site_compare",
        manifest=manifest, dem=elev, start=False,
    )  # fmt: skip
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(out))
    assert stored["files"] == {"raw": "site_compare_raw.csv", "points": "site_compare_points.csv"}
    assert stored["tasks"] == {"raw": None, "points": None}
    assert len(stored["images"]) == TIMES.size
    assert [im["img"] for im in stored["images"]] == list(range(TIMES.size))
    loaded = topocorr.load_model(tmp_path / stored["model"])
    assert loaded.id == model.id == stored["model_id"]


def test_export_checks_its_arguments(setup):  # noqa: F811
    elev, images, model = setup
    with pytest.raises(TypeError, match="StackModel"):
        topocorr.compare_export(images, object(), region=syn.ee_grid_region(40))
    with pytest.raises(ValueError, match="n_points"):
        topocorr.compare_export(images, model, region=syn.ee_grid_region(40), n_points=0)


def test_export_starts_both_tasks_and_says_where_the_files_go(setup, tmp_path, monkeypatch, capsys):  # noqa: F811
    import ee

    elev, images, model = setup
    started = []

    def fake_start(task):  # no export is run: record the task and give it an id
        started.append(task.config["description"])
        task.id = f"TASK_{len(started)}"

    monkeypatch.setattr(ee.batch.Task, "start", fake_start)
    manifest = tmp_path / "site_compare.json"
    out = topocorr.compare_export(
        images, model, region=syn.ee_grid_region(40), n_points=20, prefix="site_compare",
        manifest=manifest, dem=elev,
    )  # fmt: skip
    assert started == ["site_compare_raw", "site_compare_points"]
    assert out["tasks"] == {"raw": "TASK_1", "points": "TASK_2"}
    printed = capsys.readouterr().out
    assert "TASK_1" in printed and "site_compare_raw.csv" in printed and str(manifest) in printed
