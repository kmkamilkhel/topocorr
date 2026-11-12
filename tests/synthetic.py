"""Synthetic terrain with a known answer, in NumPy and in Earth Engine.

Grids are north-up, 30 m, in UTM zone 43N (EPSG:32643) near Kalam, aligned to multiples of
30 m so that Earth Engine reads them without resampling.
"""

from __future__ import annotations

import numpy as np

CRS = "EPSG:32643"
RES = 30.0
X0 = 279990.0  # west edge, a multiple of 30 m
Y0 = 3930000.0  # north edge, a multiple of 30 m


def centres(n: int, res: float = RES) -> tuple[np.ndarray, np.ndarray]:
    """East and north offsets (m) of pixel centres from the grid centre, shape (n, n)."""
    offsets = (np.arange(n) - (n - 1) / 2) * res
    east = np.broadcast_to(offsets, (n, n))
    north = np.broadcast_to(-offsets[:, None], (n, n))
    return east, north


def plane(slope: float, aspect: float, n: int = 9, res: float = RES) -> np.ndarray:
    """A plane of constant slope facing ``aspect`` (degrees, clockwise from north)."""
    east, north = centres(n, res)
    t = np.tan(np.radians(slope))
    a = np.radians(aspect)
    return 2000.0 - t * (east * np.sin(a) + north * np.cos(a))


def cone(slope: float, n: int = 81, res: float = RES) -> np.ndarray:
    """A cone with constant slope: every aspect is present."""
    east, north = centres(n, res)
    return 3000.0 - np.tan(np.radians(slope)) * np.hypot(east, north)


# ---- Earth Engine versions (import ee lazily: offline tests do not need it) ----


def ee_projection():
    """North-up 30 m grid with its origin at (X0, Y0)."""
    import ee

    return ee.Projection(CRS, [RES, 0, X0, 0, -RES, Y0])


def ee_grid(n: int, margin: int = 0) -> dict:
    """computePixels grid of n x n pixels starting at (X0, Y0), widened by ``margin`` pixels."""
    size = n + 2 * margin
    return {
        "dimensions": {"width": size, "height": size},
        "affineTransform": {
            "scaleX": RES,
            "shearX": 0,
            "translateX": X0 - margin * RES,
            "shearY": 0,
            "scaleY": -RES,
            "translateY": Y0 + margin * RES,
        },
        "crsCode": CRS,
    }


def ee_grid_region(n: int):
    import ee

    return ee.Geometry.Rectangle([X0, Y0 - n * RES, X0 + n * RES, Y0], ee.Projection(CRS), False)


def ee_offsets(n: int):
    """East and north offsets (m) from the centre of the n x n grid at (X0, Y0)."""
    import ee

    # Coordinates in metres (a 1 m projection), sampled on the 30 m grid.
    xy = ee.Image.pixelCoordinates(ee.Projection(CRS)).reproject(ee_projection())
    cx = X0 + n * RES / 2
    cy = Y0 - n * RES / 2
    return xy.select("x").subtract(cx), xy.select("y").subtract(cy)


def ee_plane(slope: float, aspect: float, n: int):
    east, north = ee_offsets(n)
    t = float(np.tan(np.radians(slope)))
    a = float(np.radians(aspect))
    z = east.multiply(np.sin(a)).add(north.multiply(np.cos(a))).multiply(-t).add(2000.0)
    return z.rename("elevation")


def ee_cone(slope: float, n: int):
    east, north = ee_offsets(n)
    t = float(np.tan(np.radians(slope)))
    return east.hypot(north).multiply(-t).add(3000.0).rename("elevation")


def ee_wall(height: float, half_width: float, n: int, axis: str = "east-west"):
    """Flat ground at 1000 m with a wall of ``height`` along the grid's centre line."""
    east, north = ee_offsets(n)
    across = north if axis == "east-west" else east
    return across.abs().lte(half_width).multiply(height).add(1000.0).rename("elevation")


def ee_lonlat():
    """Longitude and latitude of the grid's pixels."""
    import ee

    return ee.Image.pixelLonLat().reproject(ee_projection())


def fetch(image, n: int, margin: int = 0) -> dict[str, np.ndarray]:
    """Pixel values of an image on the synthetic grid, one array per band (float64)."""
    import ee

    data = ee.data.computePixels(
        {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": ee_grid(n, margin)}
    )
    return {name: np.asarray(data[name], dtype="float64") for name in data.dtype.names}
