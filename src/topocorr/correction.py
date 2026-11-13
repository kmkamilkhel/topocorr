"""Terrain illumination corrections.

Image by image:

``cosine``  ρ cos θz / cos i
``c``       ρ (cos θz + c) / (cos i + c)
``scsc``    ρ (cos s cos θz + c) / (cos i + c)

With a stack model from :func:`topocorr.fit`, per stratum (land cover x slope class):

``scsc-stack``  SCS+C with ``c = c(θz)`` from the model, the same curve for every image
``c-stack``     C correction with ``c = c(θz)`` from the model: normalised to flat ground, so a
                change of sun angle between years leaves no trend if the model is right
``se-stack``    statistical-empirical (Teillet et al. 1982), ρ − m_t (cos i − cos θz), with the
                image's own slope m_t from the model's first stage (same sample and settings),
                referred to a horizontal surface (see below)

For the C and SCS+C methods, ``c = n / m`` comes from a least-squares fit ``ρ = m cos i + n`` per
band and image, on a random sample of unflagged pixels with slope of at least ``min_slope``.
A band's fit is used only when it has at least ``min_pixels`` pixels, ``m > 0`` and ``c >= 0``
(with c < 0 the correction diverges on dimly lit slopes); otherwise that band is returned
unchanged, ``tc_fit_ok_<band>`` is 0 and every pixel of the image carries the "no fit" flag.

Pixels in cast shadow, in self-shadow or on very steep slopes keep their input value. Pixels
with low illumination are corrected and flagged.

Every method adds the band ``tc_corrected``: bit i (:data:`topocorr.flags.BAND_BITS`) is set where
band ``BANDS[i]`` was corrected. Bands can differ: a band whose fit is unusable (often red in
winter forest) is left unchanged while the others are corrected. **Compute ratio indices (NDVI,
NBR, ...) only from bands corrected together, or from raw bands; never from a mix.**
:func:`normalized_difference` does this check and flags the pixels where it fails.

The statistical-empirical method refers every image to a horizontal surface, ρ − m_t (cos i −
cos θz). Teillet et al. (1982) used the scene mean of cos i instead; that mean changes from image
to image (sun position, cloud mask), which would add a per-image offset to a time series.
"""

from __future__ import annotations

import math
import warnings

import ee

from . import flags as _flags
from .masking import SHADOW_SEARCH, STEEP, _check_shadow_search, _check_steep, mask_bands
from .model import _image_sums, model_strata, strata_image
from .sensors import BANDS
from .stack import Curve, StackModel
from .terrain import TERRAIN_SCALE, dem_name, terrain

METHODS = ("cosine", "c", "scsc")
"""Methods available without a fitted stack model."""
MODEL_METHODS = ("scsc-stack", "c-stack", "se-stack")
"""Methods that need a stack model from :func:`topocorr.fit`."""
DEFAULT_METHOD = "scsc-stack"
"""Method used when ``method`` is not given (a stack model is then needed)."""

WORLDCOVER = "ESA/WorldCover/v200/2021"
"""ESA WorldCover 10 m, 2021 (v200): the land cover used by ``fit_mask="worldcover-trees"``."""
FIT_MASKS = {"worldcover-trees": 10}
"""Named fit masks: WorldCover class kept for the fit."""

_DEG = math.pi / 180.0
_PER_IMAGE_FALLBACK = {"scsc-stack": "scsc", "c-stack": "c"}
_LABELS = {"scsc": "SCS+C", "c": "C"}
"""Per-image method used instead of a curve method when the sun is on both sides of the tile."""


def correct(
    images: ee.Image | ee.ImageCollection,
    model: object | None = None,
    *,
    method: str | None = None,
    dem: str | ee.Image = "glo30",
    min_slope: float = 5.0,
    steep: float = STEEP,
    shadow_search: int = SHADOW_SEARCH,
    sample_size: int = 5000,
    min_pixels: int = 100,
    region: ee.Geometry | None = None,
    fit_mask: str | ee.Image | None = "worldcover-trees",
    seed: int = 0,
) -> ee.Image | ee.ImageCollection:
    """Correct the illumination of each image.

    Parameters
    ----------
    images : ee.Image or ee.ImageCollection
        Images prepared by :func:`topocorr.collection`.
    model : topocorr.StackModel, optional
        A stack model from :func:`topocorr.fit`, needed by ``"scsc-stack"``, ``"c-stack"`` and
        ``"se-stack"`` (not used by the per-image methods). Its strata, slope classes, steep
        threshold, sample size and seed are used; ``min_slope``, ``steep``, ``sample_size``,
        ``min_pixels``, ``fit_mask`` and ``seed`` apply to the per-image methods only.
    method : {"scsc-stack", "c-stack", "se-stack", "scsc", "c", "cosine"}, optional
        Correction method. By default ``"scsc-stack"``, which needs ``model``. When the model's
        images have the sun north of the tile in some images and south of it in others (near the
        equator; :attr:`topocorr.StackModel.sun_both_sides`), ``"scsc-stack"`` warns and uses
        per-image SCS+C (``"scsc"``) instead, and ``"c-stack"`` per-image C (``"c"``): one curve
        c(zenith) cannot follow per-image c there and can bias trends on east- and west-facing
        slopes.
    dem : str or ee.Image
        DEM name, image asset id, or an elevation image in metres; for the stack methods, the
        DEM the model was fitted with.
    min_slope : float
        Smallest slope in degrees of the pixels used to fit c.
    steep : float
        Steep-slope threshold in degrees; steeper pixels are not corrected.
    shadow_search : int
        Hill-shadow search distance in pixels (per-image methods; the stack methods use the
        model's value). See :func:`topocorr.masking.shadow_search`.
    sample_size : int
        Number of pixels drawn for each image's fit (before masked pixels are dropped).
    min_pixels : int
        Fewest usable pixels for a fit.
    region : ee.Geometry, optional
        Area to sample for the fit; by default the image footprint.
    fit_mask : str, ee.Image or None
        Pixels used to fit c. The default ``"worldcover-trees"`` fits tree cover only (ESA
        WorldCover 2021, class 10; the 2021 map is also used for older scenes). An ``ee.Image``
        fits the pixels where it is non-zero; ``None`` fits all pixels. Covers of different
        brightness that sit on different aspects (dark forest on shaded slopes, bright grass on
        sunlit ones) bias a fit over all pixels, often to c < 0. The correction is still applied
        to every pixel. Ignored by ``"cosine"``. The stack model (``topocorr.fit``) fits land
        cover strata instead.
    seed : int
        Random seed of the sample.

    Returns
    -------
    ee.Image or ee.ImageCollection
        The reflectance bands corrected, plus ``cos_i`` and ``tc_flag``; properties
        ``tc_method``, ``tc_dem`` and, for ``"c"`` and ``"scsc"``, ``tc_fit_ok`` (all bands
        fitted), ``tc_n_sample`` and per band ``tc_c_<band>``, ``tc_m_<band>``,
        ``tc_fit_ok_<band>``. The stack methods add ``tc_model`` (the model id) and per band
        ``tc_c_<band>`` (``scsc-stack``, ``c-stack``: c per stratum name) or ``tc_m_<band>``
        (``se-stack``:
        m_t per stratum code; -1 where the fit was not usable).

    Examples
    --------
    ``corrected = topocorr.correct(ls, model)`` (``scsc-stack``), or
    ``topocorr.correct(ls, method="scsc", region=aoi)`` without a model.
    """
    method, model = _choose_method(method, model)
    if method in MODEL_METHODS:
        return _correct_with_model(images, model, method=method, dem=dem, region=region)
    if method not in METHODS:
        choices = ", ".join([*METHODS, *MODEL_METHODS])
        raise ValueError(f"unknown method {method!r}; choose one of {choices}")
    if model is not None:
        raise ValueError("model is used only by the stack methods: " + ", ".join(MODEL_METHODS))
    _check_steep(steep)
    _check_shadow_search(shadow_search)
    if not 0 <= float(min_slope) < 90:
        raise ValueError("min_slope must be between 0 and 90 degrees")
    if int(sample_size) < int(min_pixels) or int(min_pixels) < 2:
        raise ValueError("sample_size must be at least min_pixels, and min_pixels at least 2")
    if isinstance(fit_mask, str) and fit_mask not in FIT_MASKS:
        raise ValueError(
            f"unknown fit_mask {fit_mask!r}; use {', '.join(FIT_MASKS)}, an ee.Image or None"
        )

    def one(image: ee.Image) -> ee.Image:
        return _correct_image(
            ee.Image(image),
            method=method,
            dem=dem,
            min_slope=float(min_slope),
            steep=float(steep),
            shadow_search=int(shadow_search),
            sample_size=int(sample_size),
            min_pixels=int(min_pixels),
            region=region,
            fit_mask=fit_mask,
            seed=int(seed),
        )

    if isinstance(images, ee.ImageCollection):
        return images.map(one)
    return one(images)


def _choose_method(method: str | None, model: object | None) -> tuple[str, object | None]:
    """Return the method to run: the default, or per-image SCS+C when the sun is on both sides."""
    if method is None:
        if model is None:
            raise ValueError(
                f"the default method {DEFAULT_METHOD!r} needs a stack model from topocorr.fit; "
                "without one, choose a per-image method: scsc, c or cosine"
            )
        method = DEFAULT_METHOD
    fallback = _PER_IMAGE_FALLBACK.get(method)
    if fallback and isinstance(model, StackModel) and model.sun_both_sides:
        warnings.warn(
            "the sun is north of the tile in some of the model's images and south of it in others "
            "(near the equator); one c(zenith) curve cannot follow per-image c there, so per-image "
            f"{_LABELS[fallback]} (method {fallback!r}) is used instead of {method!r}",
            UserWarning,
            stacklevel=3,
        )
        return fallback, None
    return method, model


def _correct_image(
    image: ee.Image,
    *,
    method: str,
    dem: str | ee.Image,
    min_slope: float,
    steep: float,
    shadow_search: int,
    sample_size: int,
    min_pixels: int,
    region: ee.Geometry | None,
    fit_mask: str | ee.Image | None,
    seed: int,
) -> ee.Image:
    zenith = ee.Number(image.get("tc_sun_zenith"))
    azimuth = ee.Number(image.get("tc_sun_azimuth"))
    projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
    ter = terrain(dem, projection)
    geometry = mask_bands(ter, zenith, azimuth, steep=steep, shadow_search=shadow_search)
    ci = geometry.select("cos_i")
    flag = geometry.select("tc_flag")
    excluded = flag.bitwiseAnd(_flags.EXCLUDED).neq(0)
    reflectance = image.select(list(BANDS))
    cos_z = zenith.multiply(_DEG).cos()
    props: dict[str, object] = {"tc_method": method, "tc_dem": dem_name(dem)}

    if method == "cosine":
        corrected = reflectance.multiply(ee.Image.constant(cos_z)).divide(ci)
        no_fit = ee.Number(0)
        done = [ee.Image.constant(1)] * len(BANDS)
    else:
        usable = excluded.Not().And(ter.select("slope").gte(min_slope))
        mask = _fit_mask_image(fit_mask)
        if mask is not None:
            usable = usable.And(mask)
        samples = (
            reflectance.addBands(ci)
            .updateMask(usable)
            .sample(
                region=region if region is not None else image.geometry(),
                scale=TERRAIN_SCALE,
                projection=projection,
                numPixels=sample_size,
                seed=seed,
                dropNulls=True,
            )
        )
        count = samples.size()
        enough = count.gte(min_pixels)
        all_ok = enough
        parts = []
        done = []
        for band in BANDS:
            fit = ee.Dictionary(samples.reduceColumns(ee.Reducer.linearFit(), ["cos_i", band]))
            m = ee.Number(ee.Algorithms.If(enough, fit.get("scale"), 0))
            n = ee.Number(ee.Algorithms.If(enough, fit.get("offset"), 0))
            c = ee.Number(ee.Algorithms.If(m.gt(0), n.divide(m), -1))
            band_ok = enough.And(m.gt(0)).And(c.gte(0))
            all_ok = all_ok.And(band_ok)
            props[f"tc_m_{band}"] = m
            props[f"tc_c_{band}"] = c
            props[f"tc_fit_ok_{band}"] = band_ok
            numerator = ee.Image.constant(cos_z)
            if method == "scsc":
                numerator = ter.select("slope").multiply(_DEG).cos().multiply(cos_z)
            original = reflectance.select(band)
            fitted = original.multiply(numerator.add(c)).divide(ci.add(c)).rename(band)
            # A band whose fit is not usable is returned unchanged.
            parts.append(ee.Image(ee.Algorithms.If(band_ok, fitted, original)))
            done.append(ee.Image.constant(band_ok))
        corrected = ee.Image.cat(parts)
        no_fit = ee.Number(1).subtract(all_ok)
        props["tc_fit_ok"] = all_ok
        props["tc_n_sample"] = count

    corrected = corrected.where(excluded, reflectance).updateMask(reflectance.mask())
    flag = flag.add(ee.Image.constant(no_fit).multiply(_flags.NO_FIT)).toUint8()
    return (
        image.addBands(corrected.toFloat(), None, True)
        .addBands(ci)
        .addBands(flag.rename("tc_flag"))
        .addBands(_corrected_bits(done, excluded))
        .set(props)
    )


def _corrected_bits(done: list[ee.Image], excluded: ee.Image) -> ee.Image:
    """Band ``tc_corrected``: bit i where band i was corrected (never on excluded pixels)."""
    bits = ee.Image.constant(0)
    for image, band in zip(done, BANDS, strict=True):
        bits = bits.add(ee.Image(image).neq(0).multiply(_flags.BAND_BITS[band]))
    return bits.where(excluded, 0).toUint8().rename("tc_corrected")


def normalized_difference(
    image: ee.Image | ee.ImageCollection, first: str, second: str, name: str = "nd"
) -> ee.Image | ee.ImageCollection:
    """``(first − second) / (first + second)`` and a flag of pixels that mix corrected and raw.

    For an image from :func:`correct`, adds band ``name`` and band ``<name>_mixed``: 1 where one
    of the two bands was corrected and the other was not (see ``tc_corrected``). Such values are
    not comparable with their neighbours or through time; mask them, or compute the index from
    the raw bands. For an image without ``tc_corrected`` (raw data), ``<name>_mixed`` is 0.

    Examples
    --------
    ``ndvi = topocorr.normalized_difference(corrected, "nir", "red", "ndvi")``
    """
    if first not in BANDS or second not in BANDS or first == second:
        raise ValueError(f"two different bands among {', '.join(BANDS)} are needed")
    bits = _flags.BAND_BITS[first] | _flags.BAND_BITS[second]

    def one(img: ee.Image) -> ee.Image:
        img = ee.Image(img)
        has = img.bandNames().contains("tc_corrected")
        done = ee.Image(ee.Algorithms.If(has, img.select(["tc_corrected"]), ee.Image.constant(0)))
        hit = done.bitwiseAnd(bits)
        mixed = hit.neq(0).And(hit.neq(bits)).toUint8().rename(f"{name}_mixed")
        index = img.normalizedDifference([first, second]).rename(name)
        return img.addBands(index).addBands(mixed)

    if isinstance(image, ee.ImageCollection):
        return image.map(one)
    return one(image)


def _fit_mask_image(fit_mask: str | ee.Image | None) -> ee.Image | None:
    """1 where a pixel may be used for the fit, or None for all pixels."""
    if fit_mask is None:
        return None
    if isinstance(fit_mask, str):
        return ee.Image(WORLDCOVER).select("Map").eq(FIT_MASKS[fit_mask])
    return ee.Image(fit_mask).select(0).neq(0)


def _correct_with_model(
    images: ee.Image | ee.ImageCollection,
    model: object,
    *,
    method: str,
    dem: str | ee.Image,
    region: ee.Geometry | None,
) -> ee.Image | ee.ImageCollection:
    if model is None:
        raise ValueError(f"method {method!r} needs a stack model from topocorr.fit")
    if not isinstance(model, StackModel):
        raise TypeError(f"model must be a topocorr StackModel, not {type(model)}")
    fitted_dem = model.settings.get("dem")
    if fitted_dem != "user image" and fitted_dem != dem_name(dem):
        raise ValueError(f"the model was fitted with DEM {fitted_dem!r}, not {dem_name(dem)!r}")

    def one(image: ee.Image) -> ee.Image:
        return _correct_stack(ee.Image(image), model, method=method, dem=dem, region=region)

    if isinstance(images, ee.ImageCollection):
        return images.map(one)
    return one(images)


def _curve_value(curve: Curve, zenith: ee.Number) -> ee.Number:
    """c(θz) of a stage-2 curve, evaluated in Earth Engine (held at the fitted range's edges)."""
    x = zenith.max(curve.zmin).min(curve.zmax).subtract(curve.center).divide(curve.scale)
    total = ee.Number(curve.coef[0])
    for k, a in enumerate(curve.coef[1:], start=1):
        total = total.add(x.pow(k).multiply(a))
    return total


def _stage1_m(groups: ee.List, band_index: int, min_pixels: int) -> ee.List:
    """Per stratum group, the slope m_t of ``ρ = m cos i + n`` (-1 where it is not usable)."""
    n_bands = len(BANDS)

    def one(group):
        s = ee.List(ee.Dictionary(group).get("sum"))
        n, sx, sxx = ee.Number(s.get(0)), ee.Number(s.get(1)), ee.Number(s.get(2))
        sy = ee.Number(s.get(3 + band_index))
        sxy = ee.Number(s.get(3 + n_bands + band_index))
        d = n.multiply(sxx).subtract(sx.multiply(sx))
        m = n.multiply(sxy).subtract(sx.multiply(sy)).divide(d.max(1e-30))
        ok = n.gte(min_pixels).And(d.gt(0)).And(m.gt(0))
        return ee.Algorithms.If(ok, m, -1)

    return groups.map(one)


def _correct_stack(
    image: ee.Image,
    model: StackModel,
    *,
    method: str,
    dem: str | ee.Image,
    region: ee.Geometry | None,
) -> ee.Image:
    settings = model.settings
    edges = tuple(float(e) for e in settings["slope_classes"])
    zenith = ee.Number(image.get("tc_sun_zenith"))
    azimuth = ee.Number(image.get("tc_sun_azimuth"))
    projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
    ter = terrain(dem, projection)
    search = int(settings.get("shadow_search", SHADOW_SEARCH))
    geometry = mask_bands(
        ter, zenith, azimuth, steep=float(settings["steep"]), shadow_search=search
    )
    ci = geometry.select("cos_i")
    flag = geometry.select("tc_flag")
    excluded = flag.bitwiseAnd(_flags.EXCLUDED).neq(0)
    strata = model_strata(model)
    stratum = strata_image(strata, ter.select("slope"), edges)
    reflectance = image.select(list(BANDS))
    cos_z = zenith.multiply(_DEG).cos()
    props: dict[str, object] = {"tc_method": method, "tc_dem": dem_name(dem), "tc_model": model.id}
    no_fit = ee.Image.constant(0)
    no_model = stratum.eq(0)
    parts = []
    done = []

    if method in ("scsc-stack", "c-stack"):
        if method == "scsc-stack":
            numerator = ter.select("slope").multiply(_DEG).cos().multiply(cos_z)
        else:
            numerator = ee.Image.constant(cos_z)
        for band in BANDS:
            codes = [code for code in model.codes if model.used[band][code] is not None]
            values = [_curve_value(model.curve(band, code), zenith) for code in codes]
            # Stratum -1 never occurs; it keeps the lists non-empty when no stratum has a curve.
            c = stratum.remap([*codes, -1], [*values, -999]).unmask(-999)
            missing = c.eq(-999)
            valid = c.gte(0)
            no_model = no_model.Or(missing)
            no_fit = no_fit.Or(missing.Not().And(valid.Not()))
            original = reflectance.select(band)
            fitted = original.multiply(numerator.add(c)).divide(ci.add(c))
            parts.append(original.where(valid, fitted).rename(band))
            done.append(valid)
            props[f"tc_c_{band}"] = ee.Dictionary(
                {model.name(code): value for code, value in zip(codes, values, strict=True)}
            )
    else:
        groups = ee.List(
            _image_sums(
                image,
                dem=dem,
                strata=strata,
                slope_classes=edges,
                steep=float(settings["steep"]),
                samples_per_stratum=int(settings["samples_per_stratum"]),
                region=region,
                seed=int(settings["seed"]),
                shadow_search=search,
            ).get("groups")
        )
        codes = groups.map(lambda g: ee.Number(ee.Dictionary(g).get("stratum")).toInt())
        keys = codes.map(lambda c: ee.Number(c).format("%d"))
        for b, band in enumerate(BANDS):
            ms = _stage1_m(groups, b, int(settings["min_pixels"]))
            m = ee.Image(
                ee.Algorithms.If(
                    codes.size().gt(0),
                    stratum.remap(codes, ms, -1),
                    ee.Image.constant(-1),
                )
            )
            valid = m.gt(0)
            no_fit = no_fit.Or(stratum.gt(0).And(valid.Not()))
            original = reflectance.select(band)
            fitted = original.subtract(m.multiply(ci.subtract(cos_z)))
            parts.append(original.where(valid, fitted).rename(band))
            done.append(valid)
            props[f"tc_m_{band}"] = ee.Dictionary.fromLists(keys, ms)

    corrected = ee.Image.cat(parts).where(excluded, reflectance)
    corrected = corrected.updateMask(reflectance.mask())
    flag = (
        flag.add(no_fit.multiply(_flags.NO_FIT)).add(no_model.multiply(_flags.NO_MODEL)).toUint8()
    )
    return (
        image.addBands(corrected.toFloat(), None, True)
        .addBands(ci)
        .addBands(flag.rename("tc_flag"))
        .addBands(_corrected_bits(done, excluded))
        .set(props)
    )
