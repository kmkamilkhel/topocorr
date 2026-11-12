"""Illumination masks: cast shadow, self-shadow, low illumination and steep slopes."""

from __future__ import annotations

import math

import ee

from . import flags as _flags
from .illumination import cos_i as _cos_i
from .sensors import BANDS
from .terrain import TERRAIN_SCALE, elevation, terrain

STEEP = 60.0
"""Default steep-slope threshold in degrees."""
LOW_ILLUMINATION_LIMIT = 0.1
"""Upper limit of cos i for the low-illumination flag (``0 < cos i < 0.1``)."""
SHADOW_SEARCH = 200
"""Default distance in pixels within which Earth Engine's hill shadow looks for terrain that can
cast a shadow (200 pixels = 6 km at 30 m). :func:`shadow_search` sets it from the relief instead.
``hillShadow`` works on its own internal grid, so the edges of cast shadows are uncertain by one or
two pixels and can differ slightly between requests of different extent or scale."""

MASK_BANDS = ("cos_i", "cast_shadow", "self_shadow", "low_illumination", "steep", "tc_flag")
"""Bands returned by :func:`masks`."""


def mask_bands(
    terrain_image: ee.Image,
    zenith: float | ee.Number,
    azimuth: float | ee.Number,
    *,
    steep: float = STEEP,
    shadow_search: int = SHADOW_SEARCH,
) -> ee.Image:
    """``cos_i``, the four masks (0/1) and the combined ``tc_flag`` for one sun position.

    Parameters
    ----------
    terrain_image : ee.Image
        Bands ``elevation``, ``slope`` and ``aspect`` (see :func:`topocorr.terrain.terrain`).
    zenith, azimuth : float or ee.Number
        Sun zenith and azimuth in degrees.
    steep : float
        Slopes above this many degrees are flagged as steep.
    shadow_search : int
        Distance in pixels searched for terrain that casts a shadow (see :func:`shadow_search`).
    """
    ci = _cos_i(terrain_image, zenith, azimuth)
    self_shadow = ci.lte(0)
    low = ci.gt(0).And(ci.lt(LOW_ILLUMINATION_LIMIT))
    steep_mask = terrain_image.select("slope").gt(steep)
    elev = terrain_image.select("elevation")
    # Pinned to the terrain grid, like the slopes (see terrain.from_elevation).
    lit = ee.Terrain.hillShadow(elev, azimuth, zenith, int(shadow_search))
    lit = lit.reproject(elev.projection())
    cast = lit.eq(0).And(self_shadow.Not())
    flag = (
        cast.multiply(_flags.CAST_SHADOW)
        .add(self_shadow.multiply(_flags.SELF_SHADOW))
        .add(steep_mask.multiply(_flags.STEEP))
        .add(low.multiply(_flags.LOW_ILLUMINATION))
    )
    return ee.Image.cat(
        [
            ci,
            cast.rename("cast_shadow").toUint8(),
            self_shadow.rename("self_shadow").toUint8(),
            low.rename("low_illumination").toUint8(),
            steep_mask.rename("steep").toUint8(),
            flag.rename("tc_flag").toUint8(),
        ]
    )


def masks(
    images: ee.Image | ee.ImageCollection,
    dem: str | ee.Image = "glo30",
    *,
    steep: float = STEEP,
    shadow_search: int = SHADOW_SEARCH,
) -> ee.Image | ee.ImageCollection:
    """Illumination and masks for each image, from its sun angles and the DEM.

    Parameters
    ----------
    images : ee.Image or ee.ImageCollection
        Images prepared by :func:`topocorr.collection` (they carry ``tc_sun_zenith`` and
        ``tc_sun_azimuth``).
    dem : str or ee.Image
        DEM name (``"glo30"``, ``"nasadem"``, ``"srtm"``, ``"aw3d30"``), image asset id, or an
        elevation image in metres.
    steep : float
        Steep-slope threshold in degrees.
    shadow_search : int
        Distance in pixels searched for terrain that casts a shadow (see :func:`shadow_search`).

    Returns
    -------
    ee.Image or ee.ImageCollection
        Bands ``cos_i``, ``cast_shadow``, ``self_shadow``, ``low_illumination``, ``steep`` and
        ``tc_flag`` (see :mod:`topocorr.flags`), with the input image's properties.
    """
    _check_steep(steep)
    _check_shadow_search(shadow_search)

    def one(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        zenith = ee.Number(image.get("tc_sun_zenith"))
        azimuth = ee.Number(image.get("tc_sun_azimuth"))
        projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
        out = mask_bands(
            terrain(dem, projection), zenith, azimuth, steep=steep, shadow_search=shadow_search
        )
        return image.addBands(out).select(list(MASK_BANDS))

    if isinstance(images, ee.ImageCollection):
        return images.map(one)
    return one(images)


def _check_steep(steep: float) -> None:
    if not 0 < float(steep) <= 90:
        raise ValueError(f"steep must be a slope in degrees between 0 and 90, not {steep}")


def _check_shadow_search(shadow_search: int) -> None:
    if int(shadow_search) < 1 or int(shadow_search) > 2000:
        raise ValueError("shadow_search must be between 1 and 2000 pixels")


def shadow_search(
    region: ee.Geometry,
    max_zenith: float,
    dem: str | ee.Image = "glo30",
    *,
    scale: float = TERRAIN_SCALE,
) -> int:
    """Hill-shadow search distance in pixels for a region: the longest shadow its terrain can cast.

    A shadow is at most (highest − lowest elevation) × tan(θz) long. The relief is read over the
    region grown by that length (terrain outside the region can shade it), once more with the
    longer distance, and the result is rounded up to whole pixels of ``scale`` metres.

    Parameters
    ----------
    region : ee.Geometry
        The area to be corrected or diagnosed.
    max_zenith : float
        Largest solar zenith angle of the images, in degrees (below 89).
    dem : str or ee.Image
        DEM name, image asset id, or an elevation image in metres.
    scale : float
        Pixel size in metres of the grid the masks are computed on.
    """
    if not 0 <= float(max_zenith) < 89:
        raise ValueError("max_zenith must be between 0 and 89 degrees")
    tan_z = math.tan(math.radians(float(max_zenith)))
    elev = elevation(dem)

    def relief(geometry: ee.Geometry) -> float:
        stats = elev.reduceRegion(ee.Reducer.minMax(), geometry, 90, maxPixels=1e9, bestEffort=True)
        stats = stats.getInfo()
        return float(stats["elevation_max"] - stats["elevation_min"])

    length = relief(region) * tan_z
    length = relief(region.buffer(max(length, 1.0), 100)) * tan_z
    return max(1, math.ceil(length / float(scale)))
