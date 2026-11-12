"""DEMs and the terrain layers derived from them.

The DEM is resampled bilinearly to the image's grid, so the terrain matches the pixels it
corrects. Slope and aspect come from the four neighbours of each pixel. Aspect is measured
clockwise from **true** north, like the sun azimuth: the grid aspect of a UTM grid is rotated by
the meridian convergence ``γ = atan(tan(λ − λ0) sin φ)``, which reaches about 1.7° at the edge of
a UTM zone at 35° N. The result agrees with ``ee.Terrain.aspect``, which also measures from true
north. Explicit kernels are used so that the layers match the NumPy reference
(:mod:`topocorr._reference`) exactly and stay pinned to the image's grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ee

TERRAIN_SCALE = 30.0
"""Pixel size in metres at which slope and aspect are computed."""

_RAD = 180.0 / math.pi


@dataclass(frozen=True)
class DemSource:
    """An Earth Engine DEM: asset id, elevation band and whether the asset is a collection."""

    name: str
    asset: str
    band: str
    collection: bool


DEMS: dict[str, DemSource] = {
    "glo30": DemSource("glo30", "COPERNICUS/DEM/GLO30_2024_1", "DEM", True),
    "nasadem": DemSource("nasadem", "NASA/NASADEM_HGT/001", "elevation", False),
    "srtm": DemSource("srtm", "USGS/SRTMGL1_003", "elevation", False),
    "aw3d30": DemSource("aw3d30", "JAXA/ALOS/AW3D30/V4_1", "DSM", True),
}
"""Built-in DEMs. ``glo30`` (Copernicus GLO-30, 2024 release) is the default."""


def source(dem: str | ee.Image) -> DemSource | None:
    """Return the built-in source of a DEM name or asset id (None for an ``ee.Image``).

    A string with a ``/`` that is not a built-in asset is read as the id of a single-band image
    asset with elevation in metres.

    Examples
    --------
    >>> source("glo30").asset
    'COPERNICUS/DEM/GLO30_2024_1'
    >>> source("projects/my-project/assets/lidar_dtm").collection
    False
    """
    if isinstance(dem, ee.Image):
        return None
    if not isinstance(dem, str):
        raise TypeError(f"dem must be a DEM name, an asset id or an ee.Image, not {type(dem)}")
    key = dem.strip()
    if key.lower() in DEMS:
        return DEMS[key.lower()]
    for s in DEMS.values():
        if key == s.asset:
            return s
    if "/" in key:
        return DemSource(key, key, "", False)
    raise ValueError(f"unknown DEM {dem!r}; choose one of {', '.join(DEMS)} or give an asset id")


def dem_name(dem: str | ee.Image) -> str:
    """Name recorded in the ``tc_dem`` property of corrected images."""
    s = source(dem)
    return "user image" if s is None else s.name


def elevation(dem: str | ee.Image = "glo30") -> ee.Image:
    """Elevation in metres as a single band named ``elevation``."""
    s = source(dem)
    if s is None:
        return ee.Image(dem).select([0], ["elevation"])
    if s.collection:
        tiles = ee.ImageCollection(s.asset).select(s.band)
        image = tiles.mosaic().setDefaultProjection(tiles.first().projection())
        return image.rename("elevation")
    image = ee.Image(s.asset)
    return (image.select(s.band) if s.band else image.select([0])).rename("elevation")


def convergence(projection: ee.Projection) -> ee.Image:
    """Meridian convergence γ in degrees: the true azimuth of grid north, per pixel.

    ``γ = atan(tan(λ − λ0) sin φ)`` for a UTM projection (EPSG:326zz or EPSG:327zz) with central
    meridian ``λ0 = 6 zz − 183``. Other projections give a masked image, so nothing is silently
    computed against the wrong north.
    """
    crs = projection.crs()
    lonlat = ee.Image.pixelLonLat().reproject(projection)
    is_utm = ee.List(["EPSG:326", "EPSG:327"]).contains(crs.slice(0, 8))
    lon0 = ee.Number.parse(crs.slice(8)).multiply(6).subtract(183)
    gamma = (
        lonlat.select("longitude")
        .subtract(lon0)
        .divide(_RAD)
        .tan()
        .multiply(lonlat.select("latitude").divide(_RAD).sin())
        .atan()
        .multiply(_RAD)
    )
    masked = ee.Image.constant(0).updateMask(0)
    return ee.Image(ee.Algorithms.If(is_utm, gamma, masked)).rename("convergence")


def from_elevation(elev: ee.Image) -> ee.Image:
    """Bands ``elevation``, ``slope`` and ``aspect`` (degrees) on the elevation image's grid.

    Gradients come from the four neighbours of each pixel, as in :mod:`topocorr._reference`.
    The kernels' orientation is read from the longitude and latitude of the grid, so north-up and
    south-up grids give the same result. Aspect is rotated to true north (see
    :func:`convergence`); on flat ground, where it is undefined, it is 0.
    """
    elev = elev.rename("elevation")
    projection = elev.projection()
    step = projection.nominalScale().multiply(2)
    kx = ee.Kernel.fixed(3, 3, [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], 1, 1, False)
    ky = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, 0, 0], [0, -1, 0]], 1, 1, False)
    lonlat = ee.Image.pixelLonLat().reproject(projection)
    east_sign = lonlat.select("longitude").convolve(kx).gt(0).multiply(2).subtract(1)
    north_sign = lonlat.select("latitude").convolve(ky).gt(0).multiply(2).subtract(1)
    dz_east = elev.convolve(kx).multiply(east_sign).divide(step)
    dz_north = elev.convolve(ky).multiply(north_sign).divide(step)
    slope = dz_east.hypot(dz_north).atan().multiply(_RAD).rename("slope")
    # ee.Image.atan2 takes the vector [x, y] (x first): x = -dz_north, y = -dz_east gives the
    # azimuth of the downslope direction, as numpy.arctan2(-dz_east, -dz_north).
    grid_aspect = dz_north.multiply(-1).atan2(dz_east.multiply(-1)).multiply(_RAD)
    aspect = grid_aspect.add(convergence(projection)).add(720).mod(360)
    aspect = aspect.where(slope.eq(0), 0).rename("aspect")
    # Pin the result to the elevation grid: otherwise Earth Engine evaluates the neighbourhood
    # differences at the scale of each request (at 90 m, slopes come out 1.5 times too steep).
    return ee.Image.cat([elev, slope.toFloat(), aspect.toFloat()]).reproject(projection)


def terrain(dem: str | ee.Image, projection: ee.Projection) -> ee.Image:
    """Terrain layers on a grid: the DEM is resampled bilinearly to ``projection`` first.

    The layers stay on that grid whatever scale they are requested at; a request at another
    scale (for example 10 m Sentinel-2 bands) takes the nearest terrain pixel.

    Parameters
    ----------
    dem : str or ee.Image
        A name from :data:`DEMS`, an image asset id, or an elevation image in metres.
    projection : ee.Projection
        Target grid, normally the image's projection at :data:`TERRAIN_SCALE`.
    """
    elev = elevation(dem).resample("bilinear").reproject(projection)
    return from_elevation(elev)
