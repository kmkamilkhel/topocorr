"""Landsat 5-9 Collection 2 Level 2 and Sentinel-2 L2A: bands, scaling, cloud masks, sun angles.

Every image that topocorr prepares has the harmonised reflectance bands ``blue``, ``green``,
``red``, ``nir``, ``swir1`` and ``swir2`` (surface reflectance, unitless), clouds, cloud shadows,
snow, fill and saturated pixels masked, and three properties: ``tc_sensor``, ``tc_sun_zenith``
and ``tc_sun_azimuth`` (degrees; azimuth clockwise from north). The sun angles are the scene
values from the metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import ee

BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
"""Harmonised reflectance band names."""


@dataclass(frozen=True)
class Sensor:
    """How one sensor's Earth Engine collection maps to topocorr's harmonised form."""

    name: str
    collection: str
    bands: Mapping[str, str]
    scale: float
    offset: float
    azimuth_key: str
    zenith_key: str | None = None
    elevation_key: str | None = None
    cloud_key: str = "CLOUD_COVER"

    @property
    def native_bands(self) -> list[str]:
        """Native band names in the order of :data:`BANDS`."""
        return [self.bands[b] for b in BANDS]


_TM = {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5",
       "swir2": "SR_B7"}  # fmt: skip
_OLI = {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6",
        "swir2": "SR_B7"}  # fmt: skip
# B8A (narrow NIR, 865 nm, 20 m) is closer to OLI's NIR band than the broad 10 m B8.
_MSI = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8A", "swir1": "B11", "swir2": "B12"}


def _landsat(name: str, collection: str, bands: Mapping[str, str]) -> Sensor:
    return Sensor(
        name=name,
        collection=collection,
        bands=bands,
        scale=2.75e-05,
        offset=-0.2,
        azimuth_key="SUN_AZIMUTH",
        elevation_key="SUN_ELEVATION",
    )


SENSORS: dict[str, Sensor] = {
    "L5": _landsat("L5", "LANDSAT/LT05/C02/T1_L2", _TM),
    "L7": _landsat("L7", "LANDSAT/LE07/C02/T1_L2", _TM),
    "L8": _landsat("L8", "LANDSAT/LC08/C02/T1_L2", _OLI),
    "L9": _landsat("L9", "LANDSAT/LC09/C02/T1_L2", _OLI),
    "S2": Sensor(
        name="S2",
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=_MSI,
        scale=1e-4,
        offset=0.0,
        azimuth_key="MEAN_SOLAR_AZIMUTH_ANGLE",
        zenith_key="MEAN_SOLAR_ZENITH_ANGLE",
        cloud_key="CLOUDY_PIXEL_PERCENTAGE",
    ),
}
"""The supported sensors by short name."""

GROUPS = {"landsat": ("L5", "L7", "L8", "L9"), "sentinel2": ("S2",)}
"""Names that select several sensors at once."""

LANDSAT_QA_BITS = {"fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3, "cloud_shadow": 4,
                   "snow": 5}  # fmt: skip
"""QA_PIXEL bits that mask a Landsat pixel."""

S2_SCL_MASKED = {0: "no data", 1: "saturated or defective", 3: "cloud shadow",
                 8: "cloud, medium probability", 9: "cloud, high probability", 10: "thin cirrus",
                 11: "snow or ice"}  # fmt: skip
"""Scene classification (SCL) classes that mask a Sentinel-2 pixel.

Class 2 (dark area, which includes terrain shadow) is kept: removing it would remove the shaded
slopes that topocorr has to assess.
"""


def resolve(sensor: str) -> tuple[str, ...]:
    """Sensor short names for a sensor or group name.

    Examples
    --------
    >>> resolve("landsat")
    ('L5', 'L7', 'L8', 'L9')
    >>> resolve("s2")
    ('S2',)
    """
    key = str(sensor).strip()
    if key.lower() in GROUPS:
        return GROUPS[key.lower()]
    if key.upper() in SENSORS:
        return (key.upper(),)
    choices = ", ".join([*GROUPS, *SENSORS])
    raise ValueError(f"unknown sensor {sensor!r}; choose one of {choices}")


def sun_angles(properties: Mapping[str, object], sensor: str) -> tuple[float, float]:
    """Sun zenith and azimuth in degrees from an image's metadata.

    Parameters
    ----------
    properties : mapping
        Image properties, for example from ``image.toDictionary().getInfo()``.
    sensor : str
        Sensor short name (``"L5"``, ``"L7"``, ``"L8"``, ``"L9"`` or ``"S2"``).

    Examples
    --------
    >>> sun_angles({"SUN_ELEVATION": 28.0, "SUN_AZIMUTH": 146.3}, "L5")
    (62.0, 146.3)
    """
    (name,) = resolve(sensor)
    s = SENSORS[name]
    if s.zenith_key is not None:
        zenith = float(properties[s.zenith_key])  # type: ignore[arg-type]
    else:
        zenith = 90.0 - float(properties[s.elevation_key])  # type: ignore[index, arg-type]
    azimuth = float(properties[s.azimuth_key]) % 360.0  # type: ignore[arg-type]
    if not 0.0 <= zenith < 90.0:
        raise ValueError(f"sun zenith {zenith:.2f}° is outside 0-90°: the sun is below the horizon")
    return zenith, azimuth


def _clear(image: ee.Image, sensor: Sensor) -> ee.Image:
    """1 where the pixel is free of cloud, cloud shadow, snow, fill and saturation."""
    if sensor.name == "S2":
        scl = image.select("SCL")
        return scl.remap(list(S2_SCL_MASKED), [0] * len(S2_SCL_MASKED), 1)
    bits = sum(1 << b for b in LANDSAT_QA_BITS.values())
    qa = image.select("QA_PIXEL").bitwiseAnd(bits).eq(0)
    return qa.And(image.select("QA_RADSAT").eq(0))


def prepare(image: ee.Image, sensor: str) -> ee.Image:
    """Harmonised, scaled and masked reflectance of one image, with the sun angles as properties.

    Parameters
    ----------
    image : ee.Image
        An image of the sensor's Earth Engine collection (see :data:`SENSORS`).
    sensor : str
        Sensor short name.

    Returns
    -------
    ee.Image
        Bands :data:`BANDS`; the image's properties plus ``tc_sensor``, ``tc_sun_zenith`` and
        ``tc_sun_azimuth``.
    """
    (name,) = resolve(sensor)
    s = SENSORS[name]
    reflectance = (
        image.select(s.native_bands, list(BANDS)).multiply(s.scale).add(s.offset).toFloat()
    )
    if s.zenith_key is not None:
        zenith = ee.Number(image.get(s.zenith_key))
    else:
        zenith = ee.Number(90).subtract(ee.Number(image.get(s.elevation_key)))
    azimuth = ee.Number(image.get(s.azimuth_key)).mod(360)
    out = image.addBands(reflectance, None, True).select(list(BANDS))
    return out.updateMask(_clear(image, s)).set(
        {"tc_sensor": name, "tc_sun_zenith": zenith, "tc_sun_azimuth": azimuth}
    )


def collection(
    sensor: str,
    region: ee.Geometry,
    start: str,
    end: str,
    *,
    max_cloud: float | None = None,
) -> ee.ImageCollection:
    """Harmonised Landsat 5-9 and/or Sentinel-2 L2A images over a region, sorted by time.

    Parameters
    ----------
    sensor : str
        ``"landsat"`` (Landsat 5, 7, 8 and 9), ``"sentinel2"``, or one of ``"L5"``, ``"L7"``,
        ``"L8"``, ``"L9"``, ``"S2"``.
    region : ee.Geometry
        Images intersecting this region are selected.
    start, end : str
        Date range, ``"YYYY-MM-DD"``; ``end`` is exclusive.
    max_cloud : float, optional
        Largest scene cloud cover in percent (``CLOUD_COVER`` for Landsat,
        ``CLOUDY_PIXEL_PERCENTAGE`` for Sentinel-2).

    Returns
    -------
    ee.ImageCollection
        Prepared images (see :func:`prepare`).

    Examples
    --------
    ``ls = topocorr.collection("landsat", aoi, "1990-01-01", "2026-01-01", max_cloud=60)``
    """
    names = resolve(sensor)
    if max_cloud is not None and not 0 <= max_cloud <= 100:
        raise ValueError("max_cloud is a percentage between 0 and 100")
    merged = None
    for name in names:
        s = SENSORS[name]
        images = ee.ImageCollection(s.collection).filterBounds(region).filterDate(start, end)
        if max_cloud is not None:
            images = images.filter(ee.Filter.lte(s.cloud_key, max_cloud))
        images = images.map(lambda image, name=name: prepare(image, name))
        merged = images if merged is None else merged.merge(images)
    return merged.sort("system:time_start")
