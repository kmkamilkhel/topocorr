"""Fit the stack model in Earth Engine: strata, per-image sums and :func:`fit`.

Earth Engine draws a stratified sample of usable pixels from every image and returns, per
stratum and band, six sums (:data:`topocorr.stack.SUMS`). Everything after that runs in NumPy
(:mod:`topocorr.stack`), so a fit costs one small table per batch of images.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import ee
import numpy as np

from . import flags as _flags
from .masking import STEEP, _check_steep, mask_bands
from .sensors import BANDS
from .stack import StackModel, sensor_family, stratum_code
from .terrain import TERRAIN_SCALE, dem_name, terrain

WORLDCOVER = "ESA/WorldCover/v200/2021"
"""ESA WorldCover 10 m, 2021 (v200)."""
WORLDCOVER_CLASSES = {10: "trees", 20: "shrub", 30: "grass", 40: "crop", 60: "bare", 100: "moss"}
"""WorldCover classes modelled by ``strata="worldcover"``. Snow and ice, water, built-up,
wetland and mangrove are not modelled; their pixels are not corrected by the stack methods."""
SLOPE_CLASSES = (5.0, 25.0, 40.0, 60.0)
"""Default slope class edges in degrees: 5-25, 25-40, 40-60. The first edge is the smallest
slope used in the fit; pixels below it take the first class when corrected."""

_SUM_NAMES = ["one", "x", "xx"] + [f"{p}_{b}" for p in ("y", "xy", "yy") for b in BANDS]


def _check_slope_classes(slope_classes: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(e) for e in slope_classes)
    if (
        len(edges) < 2
        or len(edges) > 10
        or any(b <= a for a, b in zip(edges, edges[1:], strict=False))
    ):
        raise ValueError("slope_classes must be 2 to 10 increasing edges in degrees")
    if edges[0] < 0 or edges[-1] > 90:
        raise ValueError("slope_classes must lie between 0 and 90 degrees")
    return edges


def _strata_spec(strata: str | ee.Image | None) -> tuple[str, dict[str, str]]:
    """``(kind, cover names)`` of a strata argument."""
    if strata is None:
        return "none", {"1": "all"}
    if isinstance(strata, str):
        if strata != "worldcover":
            raise ValueError(f"unknown strata {strata!r}; use 'worldcover', None or an ee.Image")
        return "worldcover", {str(k): v for k, v in WORLDCOVER_CLASSES.items()}
    if isinstance(strata, ee.Image):
        return "image", {}
    raise TypeError(f"strata must be 'worldcover', None or an ee.Image, not {type(strata)}")


def strata_image(
    strata: str | ee.Image | None,
    slope: ee.Image,
    slope_classes: Sequence[float] = SLOPE_CLASSES,
) -> ee.Image:
    """Stratum codes ``10 cover + slope class`` as band ``stratum`` (0 = not modelled).

    Parameters
    ----------
    strata : "worldcover", None or ee.Image
        Land cover: ESA WorldCover 2021 classes (:data:`WORLDCOVER_CLASSES`), one cover for all
        pixels (``None``), or an integer image of the user's own classes (values above 0).
    slope : ee.Image
        Slope in degrees.
    slope_classes : sequence of float
        Class edges; class k (1, 2, ...) holds slopes from edge k to edge k + 1. Slopes below the
        first edge join class 1, slopes above the last edge the last class.
    """
    edges = _check_slope_classes(slope_classes)
    kind, _ = _strata_spec(strata)
    if kind == "none":
        cover = ee.Image.constant(1)
    elif kind == "worldcover":
        keys = list(WORLDCOVER_CLASSES)
        cover = ee.Image(WORLDCOVER).select("Map").remap(keys, keys, 0)
    else:
        user = ee.Image(strata).select(0).round()
        cover = user.where(user.lte(0), 0)
    k = ee.Image.constant(1)
    for edge in edges[1:-1]:
        k = k.add(slope.gte(edge))
    code = cover.multiply(10).add(k)
    return code.where(cover.eq(0), 0).toInt().rename("stratum")


def _image_sums(
    image: ee.Image,
    *,
    dem: str | ee.Image,
    strata: str | ee.Image | None,
    slope_classes: tuple[float, ...],
    steep: float,
    samples_per_stratum: int,
    region: ee.Geometry | None,
    seed: int,
) -> ee.Feature:
    """Stage-1 sums of one image: a feature with the image's angles and ``groups`` per stratum."""
    zenith = ee.Number(image.get("tc_sun_zenith"))
    azimuth = ee.Number(image.get("tc_sun_azimuth"))
    projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
    ter = terrain(dem, projection)
    geometry = mask_bands(ter, zenith, azimuth, steep=steep)
    ci = geometry.select("cos_i")
    excluded = geometry.select("tc_flag").bitwiseAnd(_flags.EXCLUDED).neq(0)
    slope = ter.select("slope")
    stratum = strata_image(strata, slope, slope_classes)
    usable = (
        excluded.Not()
        .And(slope.gte(slope_classes[0]))
        .And(slope.lte(slope_classes[-1]))
        .And(stratum.gt(0))
    )
    reflectance = image.select(list(BANDS))
    columns = ee.Image.cat(
        [
            ci.multiply(0).add(1).rename("one"),
            ci.rename("x"),
            ci.multiply(ci).rename("xx"),
            reflectance.rename([f"y_{b}" for b in BANDS]),
            reflectance.multiply(ci).rename([f"xy_{b}" for b in BANDS]),
            reflectance.multiply(reflectance).rename([f"yy_{b}" for b in BANDS]),
        ]
    ).toDouble()
    samples = (
        columns.addBands(stratum)
        .updateMask(usable)
        .stratifiedSample(
            numPoints=samples_per_stratum,
            classBand="stratum",
            region=region if region is not None else image.geometry(),
            scale=TERRAIN_SCALE,
            projection=projection,
            seed=seed,
            dropNulls=True,
            tileScale=4,
            geometries=False,
        )
    )
    reducer = ee.Reducer.sum().repeat(len(_SUM_NAMES)).group(len(_SUM_NAMES), "stratum")
    groups = samples.reduceColumns(reducer, [*_SUM_NAMES, "stratum"]).get("groups")
    return ee.Feature(
        None,
        {
            "id": image.get("system:index"),
            "time": image.get("system:time_start"),
            "zenith": zenith,
            "azimuth": azimuth,
            "sensor": image.get("tc_sensor"),
            "groups": groups,
        },
    )


def _settings_key(settings: dict) -> str:
    return json.dumps({k: v for k, v in settings.items() if k != "cover_names"}, sort_keys=True)


def fit(
    images: ee.ImageCollection,
    *,
    dem: str | ee.Image = "glo30",
    strata: str | ee.Image | None = "worldcover",
    slope_classes: Sequence[float] = SLOPE_CLASSES,
    degree: int = 2,
    samples_per_stratum: int = 500,
    min_pixels: int = 100,
    min_images: int = 10,
    steep: float = STEEP,
    region: ee.Geometry | None = None,
    seed: int = 0,
    batch_size: int = 8,
    cache: str | Path | None = None,
    progress: bool = True,
) -> StackModel:
    """Fit the stack model: c as a smooth function of solar zenith, per band and stratum.

    Stage 1 fits ``ρ = m_t cos i + n_t`` in every image, band and stratum on a stratified sample
    of unflagged pixels with slopes inside ``slope_classes``; stage 2 fits ``c(θz)`` across the
    stack (see :mod:`topocorr.stack`).

    Parameters
    ----------
    images : ee.ImageCollection
        Images from :func:`topocorr.collection`, all of one sensor family (Landsat 5/7,
        Landsat 8/9 or Sentinel-2).
    dem : str or ee.Image
        DEM name, image asset id, or an elevation image in metres. Use the same DEM in
        :func:`topocorr.correct`.
    strata : "worldcover", None or ee.Image
        Land cover part of the strata (see :func:`strata_image`).
    slope_classes : sequence of float
        Slope class edges in degrees (default 5-25, 25-40, 40-60).
    degree : int
        Degree of the polynomial in solar zenith (lowered when the zenith range is narrow).
    samples_per_stratum : int
        Pixels drawn per stratum and image.
    min_pixels : int
        Fewest pixels for an image's stage-1 fit to count.
    min_images : int
        Fewest usable images for a stratum's curve; otherwise its land cover's curve is used.
    steep : float
        Steep-slope threshold in degrees; steeper pixels are not used.
    region : ee.Geometry, optional
        Area sampled; by default each image's footprint.
    seed : int
        Random seed of the samples.
    batch_size : int
        Images per Earth Engine request; halved automatically when Earth Engine reports too
        many concurrent computations (after a few waits and retries).
    cache : str or Path, optional
        JSON file of finished images: an interrupted fit resumes from it.
    progress : bool
        Print progress to standard error.

    Returns
    -------
    topocorr.stack.StackModel
    """
    if int(degree) < 0 or int(degree) > 4:
        raise ValueError("degree must be between 0 and 4")
    edges = _check_slope_classes(slope_classes)
    _check_steep(steep)
    if int(samples_per_stratum) < int(min_pixels) or int(min_pixels) < 3:
        raise ValueError(
            "samples_per_stratum must be at least min_pixels, and min_pixels at least 3"
        )
    if int(min_images) < 2:
        raise ValueError("min_images must be at least 2")
    kind, cover_names = _strata_spec(strata)
    settings = {
        "strata": kind,
        "cover_names": cover_names,
        "slope_classes": list(edges),
        "degree": int(degree),
        "samples_per_stratum": int(samples_per_stratum),
        "min_pixels": int(min_pixels),
        "min_images": int(min_images),
        "steep": float(steep),
        "dem": dem_name(dem),
        "seed": int(seed),
    }
    if kind == "image":
        settings["strata_expression"] = ee.Image(strata).serialize()

    ids = images.aggregate_array("system:index").getInfo()
    if not ids:
        raise ValueError("the collection is empty")
    key = _settings_key(settings)
    done: dict[str, dict] = {}
    cache_path = Path(cache) if cache is not None else None
    if cache_path is not None and cache_path.exists():
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        if stored.get("key") == key:
            done = {k: v for k, v in stored["images"].items() if k in ids}
    todo = [i for i in ids if i not in done]
    image_list = images.toList(len(ids))
    position = {image_id: k for k, image_id in enumerate(ids)}

    def one(image):
        return _image_sums(
            ee.Image(image),
            dem=dem,
            strata=strata,
            slope_classes=edges,
            steep=float(steep),
            samples_per_stratum=int(samples_per_stratum),
            region=region,
            seed=int(seed),
        )

    size = int(batch_size)
    while todo:
        batch = todo[:size]
        chosen = ee.List([image_list.get(position[i]) for i in batch])
        try:
            features = _request(ee.FeatureCollection(chosen.map(one)).getInfo)
        except ee.EEException as error:
            if size > 1 and _is_busy(error):
                size = max(1, size // 2)  # still too many at once: smaller batches
                continue
            raise
        for feature in features["features"]:
            p = feature["properties"]
            done[p["id"]] = p
        todo = todo[len(batch) :]
        if cache_path is not None:
            cache_path.write_text(json.dumps({"key": key, "images": done}), encoding="utf-8")
        if progress:
            print(f"topocorr.fit: {len(ids) - len(todo)}/{len(ids)} images", file=sys.stderr)

    return _assemble([done[i] for i in ids if i in done], settings, strata)


def _is_busy(error: Exception) -> bool:
    """Return True for Earth Engine's "try again later" errors (concurrency and rate limits)."""
    text = str(error).lower()
    return any(k in text for k in ("too many concurrent", "rate limit", "quota", "429"))


def _request(call, attempts: int = 4, wait: float = 5.0):
    """Run an Earth Engine request, waiting and retrying when Earth Engine is busy."""
    for attempt in range(attempts):
        try:
            return call()
        except ee.EEException as error:
            if not _is_busy(error) or attempt == attempts - 1:
                raise
            time.sleep(wait * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _assemble(records: list[dict], settings: dict, strata: object = None) -> StackModel:
    """Build a model from per-image records.

    Records hold ``id``, ``time``, ``zenith``, ``azimuth``, ``sensor`` and ``groups``.
    """
    settings = dict(settings)
    settings["sensor_family"] = sensor_family(r["sensor"] for r in records)
    codes = sorted({int(g["stratum"]) for r in records for g in r.get("groups") or []})
    if not codes:
        raise ValueError("no usable pixels in any image: check region, strata and slope classes")
    sums = np.zeros((len(records), len(codes), len(BANDS), 6))
    index = {code: k for k, code in enumerate(codes)}
    for t, record in enumerate(records):
        for group in record.get("groups") or []:
            values = dict(zip(_SUM_NAMES, group["sum"], strict=True))
            k = index[int(group["stratum"])]
            for b, band in enumerate(BANDS):
                sums[t, k, b] = [
                    values["one"],
                    values["x"],
                    values["xx"],
                    values[f"y_{band}"],
                    values[f"xy_{band}"],
                    values[f"yy_{band}"],
                ]
    images = [{k: r[k] for k in ("id", "time", "zenith", "azimuth", "sensor")} for r in records]
    if settings["strata"] == "image" and not settings.get("cover_names"):
        settings["cover_names"] = {str(c // 10): str(c // 10) for c in codes}
    return StackModel.build(
        BANDS,
        codes,
        images,
        sums,
        settings,
        strata_image=strata if isinstance(strata, ee.Image) else None,
    )


def model_strata(model: StackModel) -> str | ee.Image | None:
    """Return the strata argument a model was fitted with (rebuilt for a saved model)."""
    kind = model.settings["strata"]
    if kind == "none":
        return None
    if kind == "worldcover":
        return "worldcover"
    if model.strata_image is not None:
        return model.strata_image
    return ee.Image(ee.deserializer.fromJSON(model.settings["strata_expression"]))


__all__ = ["fit", "model_strata", "strata_image", "stratum_code"]
