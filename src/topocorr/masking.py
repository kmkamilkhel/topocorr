"""Illumination masks: cast shadow, self-shadow, low illumination and steep slopes."""

from __future__ import annotations

import ee

from . import flags as _flags
from .illumination import cos_i as _cos_i
from .sensors import BANDS
from .terrain import TERRAIN_SCALE, terrain

STEEP = 60.0
"""Default steep-slope threshold in degrees."""
LOW_ILLUMINATION_LIMIT = 0.1
"""Upper limit of cos i for the low-illumination flag (``0 < cos i < 0.1``)."""
SHADOW_SEARCH = 200
"""Neighbourhood in pixels that ``ee.Terrain.hillShadow`` searches for shadowing terrain.

200 pixels (6 km at 30 m) covers a 1500 m ridge at a sun zenith of 75°. ``hillShadow`` works on
its own internal grid, so the edges of cast shadows are uncertain by one or two pixels and can
differ slightly between requests of different extent or scale.
"""

MASK_BANDS = ("cos_i", "cast_shadow", "self_shadow", "low_illumination", "steep", "tc_flag")
"""Bands returned by :func:`masks`."""


def mask_bands(
    terrain_image: ee.Image,
    zenith: float | ee.Number,
    azimuth: float | ee.Number,
    *,
    steep: float = STEEP,
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
    """
    ci = _cos_i(terrain_image, zenith, azimuth)
    self_shadow = ci.lte(0)
    low = ci.gt(0).And(ci.lt(LOW_ILLUMINATION_LIMIT))
    steep_mask = terrain_image.select("slope").gt(steep)
    elev = terrain_image.select("elevation")
    # Pinned to the terrain grid, like the slopes (see terrain.from_elevation).
    lit = ee.Terrain.hillShadow(elev, azimuth, zenith, SHADOW_SEARCH)
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

    Returns
    -------
    ee.Image or ee.ImageCollection
        Bands ``cos_i``, ``cast_shadow``, ``self_shadow``, ``low_illumination``, ``steep`` and
        ``tc_flag`` (see :mod:`topocorr.flags`), with the input image's properties.
    """
    _check_steep(steep)

    def one(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        zenith = ee.Number(image.get("tc_sun_zenith"))
        azimuth = ee.Number(image.get("tc_sun_azimuth"))
        projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
        out = mask_bands(terrain(dem, projection), zenith, azimuth, steep=steep)
        return image.addBands(out).select(list(MASK_BANDS))

    if isinstance(images, ee.ImageCollection):
        return images.map(one)
    return one(images)


def _check_steep(steep: float) -> None:
    if not 0 < float(steep) <= 90:
        raise ValueError(f"steep must be a slope in degrees between 0 and 90, not {steep}")
