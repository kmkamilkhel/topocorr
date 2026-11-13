"""Compare correction methods on one stack: export the raw data once, compare locally.

Two steps, so that the Earth Engine cost does not grow with the number of methods:

1. :func:`compare_export` draws fixed sample points (``n_points`` per stratum of the stack model,
   seeded) and starts two ``Export.table`` tasks: the raw table (every image sampled at the
   points: the six reflectance bands, solar zenith and azimuth, ``tc_flag``) and the static table
   (per point: stratum, slope, aspect, elevation, recorded forest loss, longitude, latitude). It
   writes a manifest (JSON: the images, the settings, the task ids) and the stack model (JSON).
2. :func:`compare` reads the downloaded tables, applies every correction method with the NumPy
   reference (:mod:`topocorr._reference`), and computes at the forest points:

   * ``illumination_r``: median over images of the correlation between NIR and cos i (0 when the
     correction removed the dependence; negative when it over-corrects);
   * ``sunlit_shaded_gap``: median over images of (mean NIR of the sunlit third − mean NIR of the
     shaded third) / mean NIR, the thirds by cos i over the whole table;
   * ``temporal_sd_nir`` and ``temporal_sd_<index>``: median over points of the root-mean-square
     residual after a linear trend and annual harmonics;
   * ``out_of_range``: share of values below 0 or above 1 after correction;
   * ``predicted_drift_left``: the predicted drift left, from the stack model. Each point's
     illumination-only series (its constant flat reflectance × the model's illumination factor
     F, the drift series of :func:`topocorr.diagnose`) is passed through the method, and the
     trend that remains in the deseasonalised index is taken (Theil-Sen, index units per year);
     the median of its absolute value over the points. For ``"none"`` this is the drift trend
     itself; ``"c-stack"`` leaves none where the model's c is usable (a band with c < 0 is left
     uncorrected);
   * ``predicted_drift_high``: share of points whose predicted drift left reaches the trend
     threshold (0.001 per year).

The predicted drift left assumes that the stack model is right: it shows what a method leaves of
the drift the model predicts, not a drift measured in the data. The data-based checks are
:func:`topocorr.diagnose` (observed trend against the drift trend) and the validation in the
documentation (Landsat 5 NIRv west-east trend difference before and after correction).

The export costs about as much as a point export of the raw stack; :func:`compare` makes no Earth
Engine calls.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from . import _reference as ref
from .flags import EXCLUDED
from .sensors import BANDS
from .stack import StackModel, load_model, split_code

DEFAULT_METHODS = ("none", "cosine", "scsc", "scsc-stack", "c-stack", "se-stack")
"""Methods compared by default (``"none"`` is the uncorrected stack)."""
METHODS = ("none", "cosine", "c", "scsc", "scsc-stack", "c-stack", "se-stack")
"""Methods :func:`compare` can apply."""
SCALE = 10000
"""Reflectance is exported as integers: round(ρ × SCALE)."""

_MS_PER_YEAR = 365.25 * 86400 * 1000.0
TREND_THRESHOLD = 0.001
"""|Trend| (index units per year) below which a point counts as having no trend."""


# ---- step 2: compare (NumPy) ---------------------------------------------------------------------


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return {
        k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
        for k in rows[0]
    }


def load_tables(manifest: str | Path) -> dict:
    """Read the manifest of :func:`compare_export` and the two tables downloaded next to it.

    Returns a dict with ``images`` (list of dict), ``points`` and ``raw`` (columns as arrays),
    ``model`` and ``scale``.
    """
    manifest = Path(manifest)
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    here = manifest.parent
    return {
        "images": meta["images"],
        "points": _read_csv(here / meta["files"]["points"]),
        "raw": _read_csv(here / meta["files"]["raw"]),
        "model": load_model(here / meta["model"]),
        "scale": meta["settings"].get("scale", SCALE),
    }


def _arrays(tables: Mapping) -> dict:
    """Image x point arrays of reflectance (NaN where not clear), flags and cos i."""
    images, pts, raw = tables["images"], tables["points"], tables["raw"]
    order = np.argsort(pts["pid"])
    static = {k: np.asarray(v)[order] for k, v in pts.items()}
    pid = static["pid"].astype(int)
    n_t, n_p = len(images), pid.size
    raw_pid = np.asarray(raw["pid"]).astype(int)
    col = np.minimum(np.searchsorted(pid, raw_pid), n_p - 1)
    row = np.asarray(raw["img"]).astype(int)
    ok = pid[col] == raw_pid
    scale = float(tables.get("scale", SCALE))
    data = {b: np.full((n_t, n_p), np.nan) for b in BANDS}
    for b in BANDS:
        data[b][row[ok], col[ok]] = np.asarray(raw[b])[ok] / scale
    flag = np.zeros((n_t, n_p), dtype=np.int64)
    flag[row[ok], col[ok]] = np.asarray(raw["tc_flag"])[ok].astype(np.int64)
    zen = np.array([im["zenith"] for im in images], dtype=float)
    azi = np.array([im["azimuth"] for im in images], dtype=float)
    years = np.array([im["time"] for im in images], dtype=float) / _MS_PER_YEAR + 1970
    ci = ref.cos_i(zen[:, None], azi[:, None], static["slope"][None, :], static["aspect"][None, :])
    usable = np.isfinite(data["nir"]) & ((flag & EXCLUDED) == 0)
    return {"data": data, "ci": ci, "zen": zen, "years": years, "static": static, "usable": usable}


def _is_tree(model: StackModel, code: int) -> bool:
    cover, _ = split_code(int(code))
    name = str(model.settings.get("cover_names", {}).get(str(cover), cover))
    return name.startswith("trees")


def _stack_c(a: dict, model: StackModel, band: str) -> np.ndarray:
    """Return the stack model's c per image and point (NaN where no curve applies)."""
    codes = a["static"]["stratum"].astype(int)
    c = np.full((a["zen"].size, codes.size), np.nan)
    if band not in model.bands:
        return c
    for code in np.unique(codes):
        if int(code) in model.codes:
            curve = np.asarray(model.c(band, int(code), a["zen"]), dtype=float)
            c[:, codes == code] = curve[:, None]
    return c


def _correct(
    a: dict, model: StackModel, method: str, fit_points, min_pixels: int, inputs: Sequence[dict]
) -> list[dict]:
    """Each of ``inputs`` (bands on the image x point grid) after the method.

    Per-image fits (``c``, ``scsc``, ``se-stack``) come from the data ``a["data"]`` and are
    applied to every input (an input may hold only some of the bands of the first). Excluded
    or uncorrectable values keep their input value.
    """
    data, ci, zen, usable = a["data"], a["ci"], a["zen"], a["usable"]
    codes = a["static"]["stratum"].astype(int)
    cos_s = np.cos(np.radians(a["static"]["slope"]))[None, :]
    cz = np.cos(np.radians(zen))[:, None]
    outs = [{b: v.copy() for b, v in x.items()} for x in inputs]
    if method == "none":
        return outs
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    for b in inputs[0]:
        y = data[b]
        pairs = [(x, out) for x, out in zip(inputs, outs, strict=True) if b in x]
        if method == "cosine":
            for x, out in pairs:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[b] = np.where(usable & (ci > 0), x[b] * cz / ci, x[b])
        elif method in ("c", "scsc"):
            for t in range(y.shape[0]):
                use = usable[t] & fit_points & np.isfinite(y[t])
                fit = ref.fit_c(y[t, use], ci[t, use])
                if not fit.usable(min_pixels):
                    continue
                top = (cz[t] if method == "c" else cos_s[0] * cz[t]) + fit.c
                for x, out in pairs:
                    out[b][t] = np.where(usable[t], x[b][t] * top / (ci[t] + fit.c), x[b][t])
        elif method in ("scsc-stack", "c-stack"):
            c = _stack_c(a, model, b)
            ok = usable & np.isfinite(c) & (c >= 0)
            top = (cos_s * cz if method == "scsc-stack" else cz) + c
            for x, out in pairs:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[b] = np.where(ok, x[b] * top / (ci + c), x[b])
        else:  # se-stack
            for t in range(y.shape[0]):
                for code in np.unique(codes):
                    sel = codes == code
                    use = usable[t] & sel & np.isfinite(y[t])
                    if use.sum() < min_pixels:
                        continue
                    fit = ref.fit_c(y[t, use], ci[t, use])
                    if fit.m > 0:
                        for x, out in pairs:
                            new = x[b][t] - fit.m * (ci[t] - cz[t])
                            out[b][t] = np.where(usable[t] & sel, new, out[b][t])
    return outs


def _model_series(a: dict, model: StackModel, names: Sequence[str]) -> dict:
    """Illumination-only bands: each point's flat reflectance x F (the drift series of diagnose).

    F uses the stack model's c; the flat reflectance is the median over the point's usable
    observations of rho / F.
    """
    out = {}
    for b in names:
        f = ref.illumination_factor(a["ci"], a["zen"][:, None], _stack_c(a, model, b))
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = a["usable"] & np.isfinite(f) & (f > 0) & (a["data"][b] > 0)
            ratio = np.where(ok, a["data"][b] / f, np.nan)
        flat = np.full(ratio.shape[1], np.nan)
        has = ok.any(axis=0)
        flat[has] = np.nanmedian(ratio[:, has], axis=0)
        out[b] = flat[None, :] * f
    return out


def _metrics(a, bands, predicted, forest, index, harmonics, min_obs) -> dict:
    ci, years = a["ci"], a["years"]
    usable = a["usable"] & forest[None, :]
    nir = np.where(usable, bands["nir"], np.nan)
    lo, hi = np.nanpercentile(np.where(usable, ci, np.nan), [100 / 3, 200 / 3])
    rs, gaps = [], []
    for t in range(nir.shape[0]):
        ok = np.isfinite(nir[t])
        if ok.sum() < 10 or np.std(ci[t, ok]) == 0 or np.std(nir[t, ok]) == 0:
            continue
        rs.append(float(np.corrcoef(ci[t, ok], nir[t, ok])[0, 1]))
        sun, shade = ok & (ci[t] >= hi), ok & (ci[t] <= lo)
        m = float(np.mean(nir[t, ok]))
        if sun.any() and shade.any() and m > 0:
            gaps.append((float(np.mean(nir[t, sun])) - float(np.mean(nir[t, shade]))) / m)
    names = ref.INDICES[index]
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = ref.index(index, {b: bands[b] for b in names})
    design = np.column_stack([ref.harmonic_design(years, harmonics), years - years.mean()])
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = ref.index(index, {b: predicted[b] for b in names})
    sd_nir, sd_idx, left = [], [], []
    for k in np.flatnonzero(forest):
        ok = usable[:, k] & np.isfinite(idx[:, k])
        if ok.sum() < min_obs:
            continue
        for series, store in ((bands["nir"][:, k], sd_nir), (idx[:, k], sd_idx)):
            coef = np.linalg.lstsq(design[ok], series[ok], rcond=None)[0]
            store.append(float(np.sqrt(np.mean((series[ok] - design[ok] @ coef) ** 2))))
        use = ok & np.isfinite(pred[:, k])
        for b in names:
            use &= predicted[b][:, k] > 0
        if use.sum() < min_obs:
            continue
        trend = ref.theil_sen(years[use], ref.deseasonalise(years[use], pred[use, k], harmonics))
        if np.isfinite(trend):
            left.append(abs(float(trend)))
    values = np.concatenate([bands[b][usable] for b in BANDS])
    values = values[np.isfinite(values)]
    left = np.asarray(left, dtype=float)

    def med(v):
        return float(np.median(v)) if len(v) else None

    return {
        "illumination_r": med(rs),
        "sunlit_shaded_gap": med(gaps),
        "temporal_sd_nir": med(sd_nir),
        f"temporal_sd_{index}": med(sd_idx),
        "out_of_range": float(np.mean((values < 0) | (values > 1))) if values.size else None,
        "predicted_drift_left": med(left),
        "predicted_drift_high": float(np.mean(left >= TREND_THRESHOLD)) if left.size else None,
    }


def compare(
    tables: str | Path | Mapping,
    model: StackModel | str | Path | None = None,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    index: str = "nirv",
    min_slope: float = 10.0,
    fit_min_slope: float = 5.0,
    min_pixels: int = 100,
    harmonics: int = 2,
    min_obs: int = 20,
    exclude_loss: bool = True,
):
    """Compare correction methods on the tables of :func:`compare_export` (no Earth Engine calls).

    Parameters
    ----------
    tables : str, Path or dict
        The manifest written by :func:`compare_export` (the two CSV files downloaded next to
        it), or the dict returned by :func:`load_tables`.
    model : topocorr.StackModel, str or Path, optional
        The stack model; by default the one saved with the manifest.
    methods : sequence of str
        Any of :data:`METHODS` (``"none"`` is the uncorrected stack).
    index : {"nirv", "ndvi", "evi", "nbr", "ndmi"}
        The index for temporal noise and the drift share. NIRv by default: in NDVI the drift
        artefact is often near the noise level.
    min_slope : float
        Smallest slope (degrees) of the forest points where the metrics are measured.
    fit_min_slope, min_pixels : float, int
        Per-image fits (``"c"``, ``"scsc"``, ``"se-stack"``): smallest slope of the tree-cover
        points used, and fewest points per fit.
    harmonics : int
        Annual harmonics in the temporal-noise fit and the drift test (1-3).
    min_obs : int
        Fewest clear observations of a point.
    exclude_loss : bool
        Leave points with recorded forest loss out of the metrics.

    Returns
    -------
    list of dict, or pandas.DataFrame when pandas is installed
        One row per method: ``method``, ``points``, ``images``, ``illumination_r``,
        ``sunlit_shaded_gap``, ``temporal_sd_nir``, ``temporal_sd_<index>``,
        ``out_of_range``, ``predicted_drift_left``, ``predicted_drift_high`` (the predicted
        drift left assumes that the stack model is right; see the module documentation) and
        ``tc_sun_both_sides``: 1 when the model's images have the sun north of the tile in
        some images and south of it in others (near the equator;
        :attr:`topocorr.StackModel.sun_both_sides`), else 0. ``compare`` then warns: it
        applies every method as named, while :func:`topocorr.correct` would use per-image
        SCS+C instead of ``"scsc-stack"`` and per-image C instead of ``"c-stack"``.

    Examples
    --------
    ``table = topocorr.compare("topocorr_compare.json")``
    """
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; choose from {', '.join(METHODS)}")
    if index not in ref.INDICES:
        raise ValueError(f"unknown index {index!r}; choose one of {', '.join(ref.INDICES)}")
    if not 1 <= int(harmonics) <= 3:
        raise ValueError("harmonics must be 1, 2 or 3")
    t = load_tables(tables) if isinstance(tables, (str, Path)) else dict(tables)
    if model is not None:
        t["model"] = load_model(model) if isinstance(model, (str, Path)) else model
    model = t.get("model")
    if not isinstance(model, StackModel):
        raise TypeError("a stack model is needed: pass model= or a manifest from compare_export")
    sun_both_sides = _warn_sun_both_sides(model)
    a = _arrays(t)
    codes = a["static"]["stratum"].astype(int)
    tree = np.array([_is_tree(model, c) for c in codes])
    slope = a["static"]["slope"]
    forest = tree & (slope >= min_slope)
    if exclude_loss and "loss" in a["static"]:
        forest &= a["static"]["loss"] == 0
    fit_points = tree & (slope >= fit_min_slope)
    series = _model_series(a, model, ref.INDICES[index])
    rows = []
    for method in methods:
        bands, predicted = _correct(
            a, model, method, fit_points, int(min_pixels), [a["data"], series]
        )
        row = {"method": method, "points": int(forest.sum()), "images": int(a["zen"].size)}
        row["tc_sun_both_sides"] = sun_both_sides
        row.update(_metrics(a, bands, predicted, forest, index, int(harmonics), int(min_obs)))
        rows.append(row)
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows).set_index("method")


def _warn_sun_both_sides(model: StackModel) -> int:
    """Warn and return 1 when the model has the sun on both sides of the tile, else 0."""
    if not model.sun_both_sides:
        return 0
    warnings.warn(
        "the sun is north of the tile in some of the model's images and south of it in others "
        "(near the equator); one c(zenith) curve cannot follow per-image c there. compare applies "
        "every method as named, while correct() would use per-image SCS+C instead of 'scsc-stack' "
        "and per-image C instead of 'c-stack'; read the stack methods' rows and the predicted "
        "drift left with care (tc_sun_both_sides = 1)",
        UserWarning,
        stacklevel=3,
    )
    return 1
