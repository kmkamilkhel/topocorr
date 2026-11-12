"""The stack model: the coefficient c as a smooth function of solar zenith, shared by a stack.

The model is fitted in two stages, per band and stratum (land cover x slope class):

1. **Per image** ``t``: ``ρ = m_t cos i + n_t`` by least squares on a sample of usable pixels,
   which gives ``c_t = n_t / m_t`` and its sampling variance (delta method). ``m_t`` carries the
   image's brightness (atmosphere, phenology, year) and stays free.
2. **Across the stack**: ``c_t = c(θz_t)``, a polynomial in the solar zenith (degree 2 by
   default), fitted by weighted, robust regression. Weights are ``1 / (var c_t + τ²)``, where τ²
   is the scatter between images beyond sampling noise; Tukey's biweight down-weights outlying
   images. Images with ``m_t <= 0`` or too few pixels are left out; images with ``c_t < 0`` are
   kept, so the curve is not biased upward.

A stratum with fewer than ``min_images`` usable images uses the curve of its land cover over all
slope classes (the per-image sums of its slope classes added together); when that also fails, the
stratum is not assessable and is not corrected.

This module is pure NumPy: Earth Engine supplies the per-image sums (:mod:`topocorr.model`).

Examples
--------
>>> z = np.linspace(20, 65, 30)
>>> c = 0.05 + 0.004 * (z - 20)
>>> curve = fit_curve(z, c, np.full(30, 1e-4), degree=1)
>>> round(curve(40.0), 4), round(curve(80.0), 4)  # held at the edge value beyond 65°
(0.13, 0.23)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUMS = ("n", "sx", "sxx", "sy", "sxy", "syy")
"""Per-image sufficient statistics of ``ρ = m x + n`` with ``x = cos i`` and ``y = ρ``."""

SENSOR_FAMILIES = {"L5": "TM/ETM+", "L7": "TM/ETM+", "L8": "OLI", "L9": "OLI", "S2": "MSI"}
"""Sensors whose bands are close enough to share one stack model."""

CENTER, SCALE = 45.0, 20.0
"""The polynomial's variable is ``(θz − 45) / 20`` (degrees), for a well-conditioned fit."""

TUKEY = 4.685
"""Tuning constant of Tukey's biweight (95 % efficiency for normal errors)."""

MODEL_VERSION = 1
"""Version of the saved model format."""

SUN_SIDE_SHARE = 0.05
"""Share of a stack's images that must have the sun on each side of the tile (north and south)
for :attr:`StackModel.sun_both_sides`."""


# ---- stage 1 ---------------------------------------------------------------------------------


def stage1(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Least-squares fit ``ρ = m cos i + n`` from sufficient statistics (last axis :data:`SUMS`).

    Returns arrays ``m``, ``n``, ``c`` (= n / m), ``var_c`` (delta method) and ``count``; NaN
    where the fit is undefined (fewer than 3 pixels, no spread in cos i or m = 0).

    Examples
    --------
    >>> x = np.linspace(0.2, 1.0, 50); y = 0.3 * x + 0.06
    >>> s = np.array([x.size, x.sum(), (x * x).sum(), y.sum(), (x * y).sum(), (y * y).sum()])
    >>> round(float(stage1(s)["c"]), 6)
    0.2
    """
    s = np.asarray(sums, dtype="float64")
    n, sx, sxx, sy, sxy, syy = (s[..., i] for i in range(6))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = n * sxx - sx * sx
        ok = (n >= 3) & (d > 0)
        m = np.where(ok, (n * sxy - sx * sy) / d, np.nan)
        b = np.where(ok, (sy - m * sx) / n, np.nan)
        sse = syy - 2 * m * sxy - 2 * b * sy + m * m * sxx + 2 * m * b * sx + n * b * b
        s2 = np.maximum(sse, 0.0) / (n - 2)
        var_m = s2 * n / d
        var_b = s2 * sxx / d
        cov_mb = -s2 * sx / d
        ok &= m != 0
        c = np.where(ok, b / m, np.nan)
        var_c = np.where(ok, (var_b - 2 * c * cov_mb + c * c * var_m) / (m * m), np.nan)
    return {"m": m, "n": b, "c": c, "var_c": var_c, "count": n}


def usable_images(fit: Mapping[str, np.ndarray], min_pixels: int) -> np.ndarray:
    """Images whose stage-1 fit enters stage 2: enough pixels, m > 0, finite c and variance."""
    return (
        (fit["count"] >= min_pixels)
        & (fit["m"] > 0)
        & np.isfinite(fit["c"])
        & np.isfinite(fit["var_c"])
        & (fit["var_c"] >= 0)
    )


# ---- stage 2 ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """``c(θz)``: a polynomial in ``(θz − center) / scale``, held constant outside the fitted range.

    Attributes
    ----------
    coef : tuple of float
        Coefficients, constant term first.
    cov : tuple of tuple of float
        Covariance of the coefficients.
    zmin, zmax : float
        Range of solar zenith of the fitted images (degrees).
    n_images : int
        Images that entered the fit.
    n_outliers : int
        Images given zero weight by the robust fit.
    tau : float
        Scatter of c between images beyond sampling noise.
    """

    coef: tuple[float, ...]
    cov: tuple[tuple[float, ...], ...]
    zmin: float
    zmax: float
    n_images: int
    n_outliers: int
    tau: float
    center: float = CENTER
    scale: float = SCALE

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.coef) - 1

    def __call__(self, zenith: float | np.ndarray) -> float | np.ndarray:
        """Return c at a solar zenith (degrees), held at the edge value outside the fitted range."""
        x = (np.clip(zenith, self.zmin, self.zmax) - self.center) / self.scale
        out = np.polynomial.polynomial.polyval(x, self.coef)
        return float(out) if np.ndim(out) == 0 else out

    def se(self, zenith: float | np.ndarray) -> float | np.ndarray:
        """Return the standard error of the fitted curve at a solar zenith."""
        x = np.clip(np.asarray(zenith, dtype="float64"), self.zmin, self.zmax) - self.center
        v = np.vander(np.atleast_1d(x / self.scale), self.degree + 1, increasing=True)
        out = np.sqrt(np.einsum("ij,jk,ik->i", v, np.asarray(self.cov), v))
        return float(out[0]) if np.ndim(zenith) == 0 else out

    def to_dict(self) -> dict:
        """Plain dictionary for JSON."""
        return {
            "coef": list(self.coef),
            "cov": [list(r) for r in self.cov],
            "zmin": self.zmin,
            "zmax": self.zmax,
            "n_images": self.n_images,
            "n_outliers": self.n_outliers,
            "tau": self.tau,
            "center": self.center,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Curve:
        """Inverse of :meth:`to_dict`."""
        return cls(
            coef=tuple(float(v) for v in d["coef"]),
            cov=tuple(tuple(float(v) for v in r) for r in d["cov"]),
            zmin=float(d["zmin"]),
            zmax=float(d["zmax"]),
            n_images=int(d["n_images"]),
            n_outliers=int(d["n_outliers"]),
            tau=float(d["tau"]),
            center=float(d.get("center", CENTER)),
            scale=float(d.get("scale", SCALE)),
        )


def effective_degree(zenith: np.ndarray, degree: int) -> int:
    """Lower the degree when the zenith range is narrow: < 15° → 0, < 30° → at most 1.

    Examples
    --------
    >>> effective_degree(np.array([30.0, 40.0]), 2), effective_degree(np.array([20.0, 45.0]), 2)
    (0, 1)
    """
    span = float(np.ptp(zenith)) if np.size(zenith) else 0.0
    if span < 15:
        return 0
    if span < 30:
        return min(degree, 1)
    return degree


def _mad(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    v = values if weights is None else values[weights > 0]
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _between_image_variance(r: np.ndarray, v: np.ndarray, robust: np.ndarray, p: int) -> float:
    """Paule-Mandel estimate of τ²: ``Σ w r² / (v + τ²) = Σ w − p`` (0 if no excess scatter)."""
    dof = float(robust.sum()) - p
    if dof <= 0:
        return 0.0

    def excess(tau2: float) -> float:
        return float(np.sum(robust * r * r / (v + tau2 + 1e-12))) - dof

    if excess(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, float(np.sum(robust * r * r) / max(dof, 1.0)) + float(v.max())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_curve(
    zenith: Sequence[float] | np.ndarray,
    c: Sequence[float] | np.ndarray,
    var_c: Sequence[float] | np.ndarray,
    degree: int = 2,
    *,
    iterations: int = 50,
) -> Curve:
    """Weighted robust polynomial fit of per-image c on solar zenith (stage 2).

    Weights are ``w_robust / (var_c + τ²)``. τ² (scatter between images beyond sampling noise) is
    re-estimated each iteration (Paule-Mandel, over the images kept); ``w_robust`` is Tukey's
    biweight of the standardised residuals. Iterates until the coefficients settle.
    """
    z = np.asarray(zenith, dtype="float64")
    y = np.asarray(c, dtype="float64")
    v = np.asarray(var_c, dtype="float64")
    if not (z.shape == y.shape == v.shape) or z.ndim != 1:
        raise ValueError("zenith, c and var_c must be 1-D arrays of the same length")
    if z.size == 0:
        raise ValueError("no images to fit")
    deg = effective_degree(z, int(degree))
    if z.size <= deg + 1:
        deg = max(0, z.size - 2)
    x = np.vander((z - CENTER) / SCALE, deg + 1, increasing=True)
    robust = np.ones_like(y)
    tau2 = 0.0
    beta = np.zeros(deg + 1)
    for _ in range(iterations):
        w = robust / (v + tau2 + 1e-12)
        xtw = x.T * w
        beta_new = np.linalg.lstsq(xtw @ x, xtw @ y, rcond=None)[0]
        r = y - x @ beta_new
        tau2 = _between_image_variance(r, v, (robust > 0).astype(float), deg + 1)
        u = r / np.sqrt(v + tau2 + 1e-12)
        s = max(_mad(u, robust), 1e-6)
        t = u / (TUKEY * s)
        robust = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0)
        if np.allclose(beta_new, beta, rtol=0, atol=1e-9):
            beta = beta_new
            break
        beta = beta_new
    w = robust / (v + tau2 + 1e-12)
    xtw = x.T * w
    cov = np.linalg.pinv(xtw @ x)
    used = robust > 0
    return Curve(
        coef=tuple(float(b) for b in beta),
        cov=tuple(tuple(float(e) for e in row) for row in cov),
        zmin=float(z[used].min()) if used.any() else float(z.min()),
        zmax=float(z[used].max()) if used.any() else float(z.max()),
        n_images=int(z.size),
        n_outliers=int((~used).sum()),
        tau=float(np.sqrt(tau2)),
    )


# ---- the model -------------------------------------------------------------------------------


def sun_sides(azimuths: Iterable[float]) -> tuple[int, int]:
    """Count the images with the sun north of the tile (cos azimuth > 0) and south of it.

    Examples
    --------
    >>> sun_sides([150.0, 140.0, 60.0])
    (1, 2)
    """
    az = np.radians(np.asarray(list(azimuths), dtype="float64"))
    north = int((np.cos(az) > 0).sum())
    return north, int(az.size) - north


def sensor_family(sensors: Iterable[str]) -> str:
    """Return the one sensor family of a stack; mixing families raises ``ValueError``.

    Examples
    --------
    >>> sensor_family(["L8", "L9", "L8"])
    'OLI'
    """
    families = {SENSOR_FAMILIES.get(str(s), str(s)) for s in sensors}
    if len(families) != 1:
        raise ValueError(
            f"the stack mixes sensor families {sorted(families)}; fit one family at a time "
            "(Landsat 5/7, Landsat 8/9 or Sentinel-2)"
        )
    return families.pop()


def stratum_code(cover: int, slope_class: int) -> int:
    """Stratum code ``10 cover + k`` (k = 1, 2, ... for the slope classes; 0 = all slopes)."""
    return 10 * int(cover) + int(slope_class)


def split_code(code: int) -> tuple[int, int]:
    """Inverse of :func:`stratum_code`: ``(cover, slope_class)``."""
    return int(code) // 10, int(code) % 10


@dataclass
class StackModel:
    """A fitted stack model (from :func:`topocorr.fit`, or :func:`topocorr.load_model`).

    Attributes
    ----------
    bands : tuple of str
        Bands modelled.
    codes : tuple of int
        Strata found in the stack (see :func:`stratum_code`).
    images : list of dict
        Per image: ``id``, ``time`` (ms), ``zenith``, ``azimuth``, ``sensor``.
    sums : numpy.ndarray
        Stage-1 sums, shape (images, strata, bands, 6) in the order of :data:`SUMS`.
    settings : dict
        How the model was fitted: ``strata``, ``cover_names``, ``slope_classes``, ``degree``,
        ``min_pixels``, ``min_images``, ``samples_per_stratum``, ``steep``, ``dem``, ``seed``,
        ``sensor_family``.
    curves : dict
        ``curves[band][code]`` = :class:`Curve`; own strata and pooled land covers (code
        ``10 cover``).
    used : dict
        ``used[band][code]`` = the code of the curve a stratum uses (its own, its land cover's)
        or None (not assessable).
    """

    bands: tuple[str, ...]
    codes: tuple[int, ...]
    images: list[dict]
    sums: np.ndarray
    settings: dict
    curves: dict[str, dict[int, Curve]] = field(default_factory=dict)
    used: dict[str, dict[int, int | None]] = field(default_factory=dict)
    strata_image: object = field(default=None, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        bands: Sequence[str],
        codes: Sequence[int],
        images: list[dict],
        sums: np.ndarray,
        settings: dict,
        strata_image: object = None,
    ) -> StackModel:
        """Run stage 2 on the stage-1 sums and return the model."""
        model = cls(
            bands=tuple(bands),
            codes=tuple(int(c) for c in codes),
            images=list(images),
            sums=np.asarray(sums, dtype="float64"),
            settings=dict(settings),
            strata_image=strata_image,
        )
        model._fit_curves()
        return model

    # stage 2 over every band and stratum, with the land-cover fallback
    def _fit_curves(self) -> None:
        zenith = np.array([im["zenith"] for im in self.images], dtype="float64")
        min_pixels = int(self.settings["min_pixels"])
        min_images = int(self.settings["min_images"])
        degree = int(self.settings["degree"])
        covers = sorted({split_code(c)[0] for c in self.codes})
        self.curves, self.used = {}, {}
        for b, band in enumerate(self.bands):
            curves: dict[int, Curve] = {}
            used: dict[int, int | None] = {}

            def attempt(s: np.ndarray) -> Curve | None:
                fit = stage1(s)
                ok = usable_images(fit, min_pixels)
                if ok.sum() < min_images:
                    return None
                return fit_curve(zenith[ok], fit["c"][ok], fit["var_c"][ok], degree)

            for k, code in enumerate(self.codes):
                curve = attempt(self.sums[:, k, b, :])
                if curve is not None:
                    curves[code] = curve
            for cover in covers:
                members = [k for k, code in enumerate(self.codes) if split_code(code)[0] == cover]
                pooled = attempt(self.sums[:, members, b, :].sum(axis=1))
                if pooled is not None:
                    curves[stratum_code(cover, 0)] = pooled
            for code in self.codes:
                pooled_code = stratum_code(split_code(code)[0], 0)
                if code in curves:
                    used[code] = code
                elif pooled_code in curves:
                    used[code] = pooled_code
                else:
                    used[code] = None
            self.curves[band], self.used[band] = curves, used

    # ---- queries ----
    def name(self, code: int) -> str:
        """Readable stratum name, for example ``"trees/25-40"`` or ``"trees/all slopes"``."""
        cover, k = split_code(code)
        cover_name = self.settings.get("cover_names", {}).get(str(cover), str(cover))
        if k == 0:
            return f"{cover_name}/all slopes"
        edges = self.settings["slope_classes"]
        return f"{cover_name}/{edges[k - 1]:g}-{edges[k]:g}"

    def code(self, stratum: int | str) -> int:
        """Stratum code from a code or a name."""
        if isinstance(stratum, str):
            for code in [*self.codes, *{stratum_code(split_code(c)[0], 0) for c in self.codes}]:
                if self.name(code) == stratum:
                    return code
            raise KeyError(f"unknown stratum {stratum!r}")
        return int(stratum)

    def curve(self, band: str, stratum: int | str) -> Curve | None:
        """Return the curve a stratum uses (its own or its cover's); None if not assessable."""
        if band not in self.used:
            raise KeyError(f"band {band!r} is not in the model; bands: {', '.join(self.bands)}")
        code = self.code(stratum)
        if code % 10 == 0:
            return self.curves[band].get(code)
        if code not in self.used[band]:
            raise KeyError(f"stratum {stratum!r} is not in the model")
        used = self.used[band][code]
        return None if used is None else self.curves[band][used]

    def c(self, band: str, stratum: int | str, zenith: float | np.ndarray) -> float | np.ndarray:
        """Return c of a band and stratum at a solar zenith (NaN if not assessable)."""
        curve = self.curve(band, stratum)
        if curve is None:
            return float("nan") if np.ndim(zenith) == 0 else np.full(np.shape(zenith), np.nan)
        return curve(zenith)

    def image_fits(self, band: str, stratum: int | str) -> dict[str, np.ndarray]:
        """Stage-1 results per image for one band and stratum, with ``zenith`` and ``usable``."""
        b = self.bands.index(band)
        code = self.code(stratum)
        if code % 10 == 0:
            members = [k for k, c in enumerate(self.codes) if split_code(c)[0] == code // 10]
            s = self.sums[:, members, b, :].sum(axis=1)
        else:
            s = self.sums[:, self.codes.index(code), b, :]
        fit = stage1(s)
        fit["zenith"] = np.array([im["zenith"] for im in self.images], dtype="float64")
        fit["usable"] = usable_images(fit, int(self.settings["min_pixels"]))
        return fit

    @property
    def id(self) -> str:
        """Short identifier of the model (hash of its settings, images and sums)."""
        h = hashlib.sha1(json.dumps(self.settings, sort_keys=True, default=str).encode())
        h.update(json.dumps([im["id"] for im in self.images]).encode())
        h.update(np.ascontiguousarray(self.sums).tobytes())
        return h.hexdigest()[:12]

    def summary(self) -> str:
        """Return a text table: per band and stratum, images, c at 30, 45 and 60°, source."""
        lines = [
            f"Stack model {self.id}: {len(self.images)} images, sensor family "
            f"{self.settings.get('sensor_family', '?')}, degree {self.settings['degree']}",
            f"{'band':6s} {'stratum':20s} {'images':>6s} {'c(30)':>7s} {'c(45)':>7s} "
            f"{'c(60)':>7s}  source",
        ]
        for band in self.bands:
            for code in self.codes:
                used = self.used[band][code]
                if used is None:
                    lines.append(f"{band:6s} {self.name(code):20s} {'-':>6s}  not assessable")
                    continue
                cv = self.curves[band][used]
                src = "own" if used == code else self.name(used)
                vals = " ".join(f"{cv(z):7.3f}" for z in (30.0, 45.0, 60.0))
                lines.append(f"{band:6s} {self.name(code):20s} {cv.n_images:6d} {vals}  {src}")
        return "\n".join(lines)

    @property
    def sun_both_sides(self) -> bool:
        """True when the sun is north of the tile in some images and south of it in others.

        This happens near the equator (in the test tiles, within about 10 degrees of it), where
        the sun passes north and south of the tile in the course of the year. Each side must
        hold at least :data:`SUN_SIDE_SHARE` of the images. Per-image c then differs between
        the two sides, which one curve c(zenith) cannot follow, and ``scsc-stack`` can bias
        trends on east- and west-facing slopes; :func:`topocorr.correct` uses per-image SCS+C
        for such a stack.
        """
        north, south = sun_sides(im["azimuth"] for im in self.images)
        total = north + south
        return total > 0 and min(north, south) >= SUN_SIDE_SHARE * total

    # ---- saving ----
    def to_dict(self) -> dict:
        """Plain dictionary (JSON-ready); the stage-2 curves are rebuilt on loading."""
        return {
            "format": "topocorr stack model",
            "version": MODEL_VERSION,
            "bands": list(self.bands),
            "codes": list(self.codes),
            "images": self.images,
            "sums": self.sums.tolist(),
            "settings": self.settings,
        }

    def to_json(self, path: str | Path) -> Path:
        """Save the model as JSON; :func:`load_model` reads it back."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: Mapping) -> StackModel:
        """Inverse of :meth:`to_dict` (stage 2 is refitted, so curves are identical)."""
        if d.get("format") != "topocorr stack model":
            raise ValueError("not a topocorr stack model")
        if int(d.get("version", 0)) > MODEL_VERSION:
            raise ValueError(f"model format {d['version']} is newer than this topocorr")
        return cls.build(
            d["bands"], d["codes"], list(d["images"]), np.asarray(d["sums"]), dict(d["settings"])
        )


def load_model(path: str | Path) -> StackModel:
    """Read a model saved with :meth:`StackModel.to_json`."""
    return StackModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
