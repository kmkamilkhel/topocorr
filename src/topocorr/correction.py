"""Terrain illumination corrections: cosine, C and SCS+C, fitted image by image.

``cosine``  ρ cos θz / cos i
``c``       ρ (cos θz + c) / (cos i + c)
``scsc``    ρ (cos s cos θz + c) / (cos i + c)

For the C and SCS+C methods, ``c = n / m`` comes from a least-squares fit ``ρ = m cos i + n`` per
band and image, on a random sample of unflagged pixels with slope of at least ``min_slope``.
A band's fit is used only when it has at least ``min_pixels`` pixels, ``m > 0`` and ``c >= 0``
(with c < 0 the correction diverges on dimly lit slopes); otherwise that band is returned
unchanged, ``tc_fit_ok_<band>`` is 0 and every pixel of the image carries the "no fit" flag.

Pixels in cast shadow, in self-shadow or on very steep slopes keep their input value. Pixels
with low illumination are corrected and flagged.
"""

from __future__ import annotations

import math

import ee

from . import flags as _flags
from .masking import STEEP, _check_steep, mask_bands
from .sensors import BANDS
from .terrain import TERRAIN_SCALE, dem_name, terrain

METHODS = ("cosine", "c", "scsc")
"""Methods available without a fitted stack model."""
MODEL_METHODS = ("scsc-stack", "ic")
"""Methods that need a model from ``topocorr.fit`` (not available yet)."""

_DEG = math.pi / 180.0


def correct(
    images: ee.Image | ee.ImageCollection,
    model: object | None = None,
    *,
    method: str,
    dem: str | ee.Image = "glo30",
    min_slope: float = 5.0,
    steep: float = STEEP,
    sample_size: int = 5000,
    min_pixels: int = 100,
    region: ee.Geometry | None = None,
    fit_mask: ee.Image | None = None,
    seed: int = 0,
) -> ee.Image | ee.ImageCollection:
    """Correct the illumination of each image.

    Parameters
    ----------
    images : ee.Image or ee.ImageCollection
        Images prepared by :func:`topocorr.collection`.
    model : optional
        A stack model from ``topocorr.fit`` for the stack-level methods. Not needed (and not yet
        available) for ``"cosine"``, ``"c"`` and ``"scsc"``.
    method : {"cosine", "c", "scsc"}
        Correction method.
    dem : str or ee.Image
        DEM name, image asset id, or an elevation image in metres.
    min_slope : float
        Smallest slope in degrees of the pixels used to fit c.
    steep : float
        Steep-slope threshold in degrees; steeper pixels are not corrected.
    sample_size : int
        Number of pixels drawn for each image's fit (before masked pixels are dropped).
    min_pixels : int
        Fewest usable pixels for a fit.
    region : ee.Geometry, optional
        Area to sample for the fit; by default the image footprint.
    fit_mask : ee.Image, optional
        Only pixels where this image is non-zero are used to fit c, for example one land cover
        (``ee.Image("ESA/WorldCover/v200/2021").eq(10)`` for tree cover). Covers of different
        brightness that sit on different aspects (dark forest on shaded slopes, bright grass on
        sunlit ones) bias a fit over all pixels, often to c < 0. The correction is still applied
        to every pixel.
    seed : int
        Random seed of the sample.

    Returns
    -------
    ee.Image or ee.ImageCollection
        The reflectance bands corrected, plus ``cos_i`` and ``tc_flag``; properties
        ``tc_method``, ``tc_dem`` and, for ``"c"`` and ``"scsc"``, ``tc_fit_ok`` (all bands
        fitted), ``tc_n_sample`` and per band ``tc_c_<band>``, ``tc_m_<band>``,
        ``tc_fit_ok_<band>``.

    Examples
    --------
    ``corrected = topocorr.correct(ls, method="scsc", region=aoi)``
    """
    if method in MODEL_METHODS:
        raise NotImplementedError(f"method {method!r} needs a stack model; not available yet")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {', '.join(METHODS)}")
    if model is not None:
        raise NotImplementedError("stack models are not available yet; leave model=None")
    _check_steep(steep)
    if not 0 <= float(min_slope) < 90:
        raise ValueError("min_slope must be between 0 and 90 degrees")
    if int(sample_size) < int(min_pixels) or int(min_pixels) < 2:
        raise ValueError("sample_size must be at least min_pixels, and min_pixels at least 2")

    def one(image: ee.Image) -> ee.Image:
        return _correct_image(
            ee.Image(image),
            method=method,
            dem=dem,
            min_slope=float(min_slope),
            steep=float(steep),
            sample_size=int(sample_size),
            min_pixels=int(min_pixels),
            region=region,
            fit_mask=fit_mask,
            seed=int(seed),
        )

    if isinstance(images, ee.ImageCollection):
        return images.map(one)
    return one(images)


def _correct_image(
    image: ee.Image,
    *,
    method: str,
    dem: str | ee.Image,
    min_slope: float,
    steep: float,
    sample_size: int,
    min_pixels: int,
    region: ee.Geometry | None,
    fit_mask: ee.Image | None,
    seed: int,
) -> ee.Image:
    zenith = ee.Number(image.get("tc_sun_zenith"))
    azimuth = ee.Number(image.get("tc_sun_azimuth"))
    projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
    ter = terrain(dem, projection)
    geometry = mask_bands(ter, zenith, azimuth, steep=steep)
    ci = geometry.select("cos_i")
    flag = geometry.select("tc_flag")
    excluded = flag.bitwiseAnd(_flags.EXCLUDED).neq(0)
    reflectance = image.select(list(BANDS))
    cos_z = zenith.multiply(_DEG).cos()
    props: dict[str, object] = {"tc_method": method, "tc_dem": dem_name(dem)}

    if method == "cosine":
        corrected = reflectance.multiply(ee.Image.constant(cos_z)).divide(ci)
        no_fit = ee.Number(0)
    else:
        usable = excluded.Not().And(ter.select("slope").gte(min_slope))
        if fit_mask is not None:
            usable = usable.And(ee.Image(fit_mask).select(0).neq(0))
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
        .set(props)
    )
