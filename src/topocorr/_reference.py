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


# ---- diagnose: is a trend illumination? ----------------------------------------------------

INDICES = {
    "ndvi": ("nir", "red"),
    "nbr": ("nir", "swir2"),
    "ndmi": ("nir", "swir1"),
    "evi": ("nir", "red", "blue"),
    "nirv": ("nir", "red"),
}
"""Indices :func:`index` computes, with the bands each needs."""

RISK_CLASSES = {0: "not assessable", 1: "low", 2: "medium", 3: "high", 4: "no trend"}
"""``risk_class`` values of :func:`topocorr.diagnose`."""
RISK_LIMITS = (0.2, 0.5)
"""Artefact share limits of the low, medium and high classes (provisional)."""


def illumination_factor(
    cos_incidence: np.ndarray, zenith: float | np.ndarray, c: float | np.ndarray
) -> np.ndarray:
    """Return the reflectance of a slope relative to flat ground: (cos i + c) / (cos θz + c).

    Examples
    --------
    >>> round(float(illumination_factor(np.array(0.5), 60.0, 0.0)), 6)  # cos 60° = 0.5: as flat
    1.0
    """
    cz = np.cos(np.radians(zenith))
    return (np.asarray(cos_incidence, dtype="float64") + c) / (cz + c)


def index(name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a vegetation index from bands (``ndvi``, ``nbr``, ``ndmi``, ``evi``, ``nirv``).

    Examples
    --------
    >>> round(float(index("ndvi", {"nir": np.array(0.4), "red": np.array(0.1)})), 3)
    0.6
    """
    if name not in INDICES:
        raise ValueError(f"unknown index {name!r}; choose one of {', '.join(INDICES)}")
    b = {k: np.asarray(v, dtype="float64") for k, v in bands.items()}
    if name == "evi":
        return 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1)
    first, second = INDICES[name][:2]
    nd = (b[first] - b[second]) / (b[first] + b[second])
    return nd * b["nir"] if name == "nirv" else nd


def harmonic_design(years: np.ndarray, harmonics: int = 2) -> np.ndarray:
    """Columns 1, cos 2πkt, sin 2πkt (k = 1..harmonics) for time in decimal years."""
    t = np.asarray(years, dtype="float64")
    cols = [np.ones_like(t)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(cols)


def deseasonalise(years: np.ndarray, values: np.ndarray, harmonics: int = 2) -> np.ndarray:
    """Remove the seasonal harmonics from a series (the mean level is kept)."""
    y = np.asarray(values, dtype="float64")
    x = harmonic_design(years, harmonics)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - x[:, 1:] @ beta[1:]


def theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Median of all pairwise slopes (Theil–Sen estimator).

    Examples
    --------
    >>> theil_sen(np.arange(5.0), np.array([0.0, 2.0, 4.0, 6.0, 100.0]))  # robust to one outlier
    2.0
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else float("nan")


def risk_class(share: float, observed_trend: float, trend_threshold: float) -> int:
    """Risk class from the artefact share (see :data:`RISK_CLASSES`)."""
    if not (np.isfinite(share) and np.isfinite(observed_trend)):
        return 0
    if abs(observed_trend) < trend_threshold:
        return 4
    s = abs(share)
    return 1 if s < RISK_LIMITS[0] else 2 if s <= RISK_LIMITS[1] else 3


def diagnose_pixel(
    years: np.ndarray,
    bands: dict[str, np.ndarray],
    cos_incidence: np.ndarray,
    zenith: np.ndarray,
    c: dict[str, np.ndarray],
    *,
    index_name: str = "ndvi",
    harmonics: int = 2,
    trend_threshold: float = 0.001,
    usable: np.ndarray | None = None,
) -> dict[str, float]:
    """Observed and illumination-predicted index trends of one pixel, as topocorr.diagnose does.

    ``bands`` are the pixel's reflectance series and ``c[band]`` the model's c per acquisition.
    The flat-surface reflectance is the median of ρ / F over the series; the predicted series is
    the index of F · ρ_flat. Both series are deseasonalised with the same harmonics, and their
    Theil–Sen slopes (per year) give the artefact share = predicted / observed trend.
    Acquisitions where ``usable`` is False (by default: self-shadow, cos i <= 0) are left out,
    as topocorr.diagnose leaves out flagged pixels.
    """
    names = INDICES[index_name]
    keep = np.asarray(cos_incidence) > 0 if usable is None else np.asarray(usable, dtype=bool)
    years = np.asarray(years, dtype="float64")[keep]
    cos_incidence = np.asarray(cos_incidence, dtype="float64")[keep]
    zenith = np.broadcast_to(np.asarray(zenith, dtype="float64"), keep.shape)[keep]
    bands = {b: np.asarray(bands[b], dtype="float64")[keep] for b in names}
    c = {b: np.broadcast_to(np.asarray(c[b], dtype="float64"), keep.shape)[keep] for b in names}
    factor = {b: illumination_factor(cos_incidence, zenith, c[b]) for b in names}
    flat = {b: float(np.median(np.asarray(bands[b]) / factor[b])) for b in names}
    observed = index(index_name, {b: bands[b] for b in names})
    predicted = index(index_name, {b: factor[b] * flat[b] for b in names})
    obs_trend = theil_sen(years, deseasonalise(years, observed, harmonics))
    art_trend = theil_sen(years, deseasonalise(years, predicted, harmonics))
    geo_trend = theil_sen(years, deseasonalise(years, cos_incidence, harmonics))
    share = art_trend / obs_trend if obs_trend != 0 else float("nan")
    return {
        "observed_trend": obs_trend,
        "artefact_trend": art_trend,
        "geometry_trend": geo_trend,
        "artefact_share": share,
        "risk_class": risk_class(share, obs_trend, trend_threshold),
    }
