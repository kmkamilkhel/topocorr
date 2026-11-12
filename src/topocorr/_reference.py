"""NumPy versions of topocorr's formulas.

The Earth Engine code in topocorr is tested against these functions on synthetic terrain with a
known answer. They follow the same rules: slope and aspect from the four neighbours of each pixel,
aspect rotated from grid north to true north by the meridian convergence, angles in degrees,
aspect and azimuth clockwise from true north, and flagged pixels left at their input value.

Arrays are indexed ``[row, column]`` with rows running from north to south and columns from west
to east, as in a north-up raster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flags import CAST_SHADOW, LOW_ILLUMINATION, NO_FIT, SELF_SHADOW, STEEP

LOW_ILLUMINATION_LIMIT = 0.1
"""Pixels with ``0 < cos i < 0.1`` are flagged as low illumination (flag only)."""


def utm_convergence(lon: np.ndarray, lat: np.ndarray, zone: int) -> np.ndarray:
    """Meridian convergence of a UTM zone in degrees: the true azimuth of grid north.

    ``γ = atan(tan(λ − λ0) sin φ)`` with ``λ0 = 6 zone − 183``.

    Examples
    --------
    >>> round(float(utm_convergence(72.5828, 35.4829, 43)), 3)  # Kalam, zone 43 (75° E)
    -1.404
    """
    dlon = np.radians(np.asarray(lon, dtype="float64") - (6 * zone - 183))
    return np.degrees(np.arctan(np.tan(dlon) * np.sin(np.radians(lat))))


def slope_aspect(
    elevation: np.ndarray, pixel_size: float, convergence: float | np.ndarray = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Slope and aspect in degrees from the four neighbours of each pixel.

    Parameters
    ----------
    elevation : numpy.ndarray
        Elevation in metres, north-up.
    pixel_size : float
        Pixel size in metres.
    convergence : float or numpy.ndarray
        True azimuth of grid north in degrees (see :func:`utm_convergence`); 0 gives the aspect
        relative to grid north.

    Returns
    -------
    slope, aspect : numpy.ndarray
        Slope (0-90) and aspect (0-360, clockwise from true north; the direction the slope
        faces). Edge pixels are NaN.

    Examples
    --------
    >>> z = np.add.outer(np.zeros(5), np.arange(5) * 30.0)  # rises 1 m per m towards the east
    >>> s, a = slope_aspect(z, 30.0)
    >>> round(float(s[2, 2]), 6), round(float(a[2, 2]), 6)
    (45.0, 270.0)
    """
    z = np.asarray(elevation, dtype="float64")
    dzdx = np.full(z.shape, np.nan)
    dzdn = np.full(z.shape, np.nan)
    dzdx[1:-1, 1:-1] = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * pixel_size)
    dzdn[1:-1, 1:-1] = (z[:-2, 1:-1] - z[2:, 1:-1]) / (2 * pixel_size)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + convergence) % 360.0
    aspect = np.where(slope == 0, 0.0, aspect)
    return slope, aspect


def cos_i(zenith: float, azimuth: float, slope: np.ndarray, aspect: np.ndarray) -> np.ndarray:
    """Cosine of the local solar incidence angle.

    ``cos i = cos θz cos s + sin θz sin s cos(φa − φ)``, all angles in degrees.

    Examples
    --------
    >>> float(cos_i(30.0, 180.0, np.array(30.0), np.array(180.0)))  # slope faces the sun
    1.0
    """
    z, a = np.radians(zenith), np.radians(azimuth)
    s, p = np.radians(slope), np.radians(aspect)
    return np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(a - p)


def flags(
    cos_incidence: np.ndarray,
    slope: np.ndarray,
    *,
    steep: float = 60.0,
    cast_shadow: np.ndarray | None = None,
    no_fit: bool = False,
) -> np.ndarray:
    """Flag bits per pixel (see :mod:`topocorr.flags`).

    Self-shadow: ``cos i <= 0``. Low illumination: ``0 < cos i < 0.1``. Steep: slope above
    ``steep`` degrees. Cast shadow: where ``cast_shadow`` is true and the pixel is not
    self-shadowed.
    """
    ci = np.asarray(cos_incidence, dtype="float64")
    out = np.zeros(ci.shape, dtype="uint8")
    self_shadow = ci <= 0
    out |= np.where(self_shadow, SELF_SHADOW, 0).astype("uint8")
    out |= np.where((ci > 0) & (ci < LOW_ILLUMINATION_LIMIT), LOW_ILLUMINATION, 0).astype("uint8")
    out |= np.where(np.asarray(slope) > steep, STEEP, 0).astype("uint8")
    if cast_shadow is not None:
        cast = np.asarray(cast_shadow, dtype=bool) & ~self_shadow
        out |= np.where(cast, CAST_SHADOW, 0).astype("uint8")
    if no_fit:
        out |= np.uint8(NO_FIT)
    return out


def excluded(flag: np.ndarray) -> np.ndarray:
    """Pixels that are not corrected: cast shadow, self-shadow or steep (low illumination is)."""
    return (np.asarray(flag) & (CAST_SHADOW | SELF_SHADOW | STEEP)) != 0


@dataclass(frozen=True)
class LinearFit:
    """Least-squares fit ``ρ = m cos i + n`` and the coefficient ``c = n / m``."""

    m: float
    n: float
    count: int

    @property
    def c(self) -> float:
        """Coefficient c of the C and SCS+C corrections (NaN when m is 0)."""
        return self.n / self.m if self.m != 0 else float("nan")

    def usable(self, min_pixels: int = 100) -> bool:
        """Return True when the fit can be used: enough pixels, m > 0 and c >= 0."""
        return self.count >= min_pixels and self.m > 0 and self.c >= 0


def fit_c(reflectance: np.ndarray, cos_incidence: np.ndarray) -> LinearFit:
    """Fit ``ρ = m cos i + n`` by least squares over the finite pixels.

    Examples
    --------
    >>> ci = np.linspace(0.2, 1.0, 50)
    >>> f = fit_c(0.3 * ci + 0.06, ci)
    >>> round(f.c, 6)
    0.2
    """
    y = np.asarray(reflectance, dtype="float64").ravel()
    x = np.asarray(cos_incidence, dtype="float64").ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.ptp(x) == 0:
        return LinearFit(float("nan"), float("nan"), int(x.size))
    m, n = np.polyfit(x, y, 1)
    return LinearFit(float(m), float(n), int(x.size))


def cosine(reflectance: np.ndarray, cos_incidence: np.ndarray, zenith: float) -> np.ndarray:
    """Cosine correction ``ρ cos θz / cos i``."""
    return np.asarray(reflectance) * np.cos(np.radians(zenith)) / np.asarray(cos_incidence)


def c_correction(
    reflectance: np.ndarray, cos_incidence: np.ndarray, zenith: float, c: float
) -> np.ndarray:
    """C correction ``ρ (cos θz + c) / (cos i + c)``."""
    cz = np.cos(np.radians(zenith))
    return np.asarray(reflectance) * (cz + c) / (np.asarray(cos_incidence) + c)


def scsc(
    reflectance: np.ndarray,
    cos_incidence: np.ndarray,
    zenith: float,
    slope: np.ndarray,
    c: float,
) -> np.ndarray:
    """SCS+C correction ``ρ (cos s cos θz + c) / (cos i + c)``."""
    cz = np.cos(np.radians(zenith))
    cs = np.cos(np.radians(slope))
    return np.asarray(reflectance) * (cs * cz + c) / (np.asarray(cos_incidence) + c)


def correct(
    reflectance: np.ndarray,
    *,
    method: str,
    zenith: float,
    azimuth: float,
    slope: np.ndarray,
    aspect: np.ndarray,
    min_slope: float = 5.0,
    steep: float = 60.0,
    min_pixels: int = 100,
    cast_shadow: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, LinearFit | None]:
    """Correct one band as topocorr does: fit, correct, keep flagged pixels unchanged.

    Returns the corrected band, the flag bits and the fit (None for the cosine method). When the
    fit is not usable, the band is returned unchanged and every pixel carries the "no fit" flag.
    """
    rho = np.asarray(reflectance, dtype="float64")
    ci = cos_i(zenith, azimuth, slope, aspect)
    flag = flags(ci, slope, steep=steep, cast_shadow=cast_shadow)
    keep = excluded(flag)
    fit = None
    if method == "cosine":
        out = cosine(rho, ci, zenith)
    elif method in ("c", "scsc"):
        use = ~keep & (np.asarray(slope) >= min_slope) & np.isfinite(rho)
        fit = fit_c(rho[use], ci[use])
        if not fit.usable(min_pixels):
            return rho.copy(), flag | np.uint8(NO_FIT), fit
        if method == "c":
            out = c_correction(rho, ci, zenith, fit.c)
        else:
            out = scsc(rho, ci, zenith, slope, fit.c)
    else:
        raise ValueError(f"unknown method {method!r}")
    return np.where(keep, rho, out), flag, fit
