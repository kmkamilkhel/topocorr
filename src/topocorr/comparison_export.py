"""The export half of :func:`topocorr.compare`: the raw stack at fixed sample points (Earth Engine).

See :mod:`topocorr.comparison` for the two steps and the metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from .comparison import SCALE
from .sensors import BANDS
from .stack import StackModel


def compare_export(
    images,
    model: StackModel,
    *,
    region,
    n_points: int = 500,
    folder: str = "topocorr_exports",
    prefix: str = "topocorr_compare",
    manifest: str | Path = "topocorr_compare.json",
    dem="glo30",
    seed: int = 42,
    start: bool = True,
) -> dict:
    """Export the raw stack at fixed sample points, for :func:`compare`.

    Parameters
    ----------
    images : ee.ImageCollection
        Images from :func:`topocorr.collection` (raw reflectance), the stack the model was fitted
        on.
    model : topocorr.StackModel
        The stack model; its strata, slope classes, steep threshold and hill-shadow search are
        used.
    region : ee.Geometry
        Where the points are drawn.
    n_points : int
        Points per stratum.
    folder, prefix : str
        Google Drive folder and file name prefix of the two tables.
    manifest : str or Path
        Local path of the manifest (JSON); the model is saved next to it as
        ``<stem>_model.json``.
    dem : str or ee.Image
        The DEM the model was fitted with.
    seed : int
        Random seed of the points.
    start : bool
        Start the export tasks (False only builds them).

    Returns
    -------
    dict
        The manifest: task ids, Drive folder, file names, images, settings.

    Examples
    --------
    ``topocorr.compare_export(oli, model, region=aoi, folder="topocorr_exports")``
    """
    import ee

    from .masking import masks
    from .model import HANSEN, model_strata, strata_image
    from .terrain import TERRAIN_SCALE, terrain

    if not isinstance(model, StackModel):
        raise TypeError(f"model must be a topocorr StackModel, not {type(model)}")
    if not 1 <= int(n_points) <= 5000:
        raise ValueError("n_points must be between 1 and 5000")
    images = ee.ImageCollection(images)
    settings = model.settings
    projection = ee.Image(images.first()).select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
    ter = terrain(dem, projection)
    stratum = strata_image(model_strata(model), ter.select("slope"), settings["slope_classes"])
    loss = ee.Image(HANSEN).select("loss").unmask(0).rename("loss")
    drawn = stratum.selfMask().stratifiedSample(
        numPoints=int(n_points), classBand="stratum", region=region, projection=projection,
        seed=int(seed), geometries=True,
    )  # fmt: skip
    listed = drawn.toList(drawn.size())
    points = ee.FeatureCollection(
        ee.List.sequence(0, listed.size().subtract(1)).map(
            lambda i: ee.Feature(ee.Feature(listed.get(i)).geometry()).set("pid", i)
        )
    )
    static = ee.Image.cat(
        [stratum, ter.select(["elevation", "slope", "aspect"]), loss, ee.Image.pixelLonLat()]
    ).sampleRegions(points, ["pid"], projection=projection)

    info = ee.Dictionary(
        {
            "id": images.aggregate_array("system:index"),
            "time": images.aggregate_array("system:time_start"),
            "zenith": images.aggregate_array("tc_sun_zenith"),
            "azimuth": images.aggregate_array("tc_sun_azimuth"),
            "sensor": images.aggregate_array("tc_sensor"),
        }
    ).getInfo()
    n_images = len(info["id"])
    image_list = images.toList(n_images)
    steep = float(settings["steep"])
    search = settings.get("shadow_search")
    mask_args = {"steep": steep} | ({"shadow_search": int(search)} if search else {})

    def sample(i):
        img = ee.Image(image_list.get(i))
        flag = ee.Image(masks(img, dem, **mask_args)).select("tc_flag")
        values = ee.Image.cat([img.select(list(BANDS)).multiply(SCALE).round().toInt16(), flag])
        extra = {"img": i, "zenith": img.get("tc_sun_zenith"), "azimuth": img.get("tc_sun_azimuth")}
        return values.sampleRegions(points, ["pid"], projection=projection).map(
            lambda f: f.set(extra)
        )

    raw = ee.FeatureCollection(ee.List.sequence(0, n_images - 1).map(sample)).flatten()
    raw_cols = ["pid", "img", *BANDS, "tc_flag", "zenith", "azimuth"]
    static_cols = ["pid", "stratum", "elevation", "slope", "aspect", "loss"]
    static_cols += ["longitude", "latitude"]
    tasks = {
        "raw": ee.batch.Export.table.toDrive(
            collection=raw, description=f"{prefix}_raw", folder=folder,
            fileNamePrefix=f"{prefix}_raw", fileFormat="CSV", selectors=raw_cols,
        ),
        "points": ee.batch.Export.table.toDrive(
            collection=static, description=f"{prefix}_points", folder=folder,
            fileNamePrefix=f"{prefix}_points", fileFormat="CSV", selectors=static_cols,
        ),
    }  # fmt: skip
    if start:
        for task in tasks.values():
            task.start()
    manifest = Path(manifest)
    model_file = manifest.with_name(f"{manifest.stem}_model.json")
    model.to_json(model_file)
    out = {
        "folder": folder,
        "files": {"raw": f"{prefix}_raw.csv", "points": f"{prefix}_points.csv"},
        "tasks": {k: (t.id if start else None) for k, t in tasks.items()},
        "model": model_file.name,
        "model_id": model.id,
        "settings": {"n_points": int(n_points), "seed": int(seed), "scale": SCALE},
        "images": [
            {"img": k, "id": info["id"][k], "time": info["time"][k], "zenith": info["zenith"][k],
             "azimuth": info["azimuth"][k], "sensor": info["sensor"][k]}
            for k in range(n_images)
        ],
    }  # fmt: skip
    manifest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if start:
        tasks_txt = f"raw task {out['tasks']['raw']}, points task {out['tasks']['points']}"
        print(f"topocorr.compare_export: {tasks_txt}")
        print(f"  Google Drive folder {folder!r}: {out['files']['raw']}, {out['files']['points']}")
        print(f"  download both next to {manifest}, then topocorr.compare({str(manifest)!r})")
    return out
