"""Local solar illumination: the cosine of the incidence angle.

For slope s, aspect φ, sun zenith θz and sun azimuth φa (degrees; aspect and azimuth clockwise
from north)::

    cos i = cos θz cos s + sin θz sin s cos(φa − φ)

cos i is 1 where the sun shines straight onto the slope, ``cos θz`` on flat ground and 0 or less
where the slope faces away from the sun (self-shadow).
"""

from __future__ import annotations

import math

import ee

_DEG = math.pi / 180.0


def cos_i(terrain: ee.Image, zenith: float | ee.Number, azimuth: float | ee.Number) -> ee.Image:
    """Cosine of the solar incidence angle as a band named ``cos_i``.

    Parameters
    ----------
    terrain : ee.Image
        Bands ``slope`` and ``aspect`` in degrees (see :func:`topocorr.terrain.terrain`).
    zenith, azimuth : float or ee.Number
        Sun zenith and azimuth in degrees.
    """
    z = ee.Number(zenith).multiply(_DEG)
    a = ee.Number(azimuth).multiply(_DEG)
    s = terrain.select("slope").multiply(_DEG)
    p = terrain.select("aspect").multiply(_DEG)
    flat = s.cos().multiply(z.cos())
    tilted = s.sin().multiply(z.sin()).multiply(ee.Image.constant(a).subtract(p).cos())
    return flat.add(tilted).rename("cos_i").toFloat()
