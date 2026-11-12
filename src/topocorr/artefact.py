"""Is a trend real or illumination? The artefact test.

For every pixel and acquisition the stack model gives how bright the pixel is relative to a flat
surface under the same sun, ``F_b = (cos i + c_b(θz)) / (cos θz + c_b(θz))`` per band, with c from
the pixel's stratum. All series are deseasonalised with the same harmonics, on the same
observations, and their Theil–Sen slopes (index units per year) give two illumination parts of the
observed trend (``observed_trend``):

* ``drift_trend``, the **drift artefact**: the trend that the changing illumination itself causes,
  the trend of the index of F_b times the pixel's constant flat reflectance (median of ρ / F over
  the series). It can create or flip a trend with no vegetation change (orbit drift, sensor
  switches, a changing season of the clear images);
* ``illumination_trend``, the **total illumination effect**: the observed trend minus the trend
  of the illumination-corrected series ρ / F. It also includes ``scaling_trend`` = total − drift:
  a real trend seen larger or smaller than on flat ground because of the pixel's mean
  illumination.

``drift_share`` and ``illumination_share`` divide these by the observed trend. ``risk_class`` uses
the drift share: 1 low (|share| < 0.2), 2 medium (≤ 0.5), 3 high (> 0.5), 4 no trend (|observed
trend| below ``trend_threshold``), 0 not assessable (too few observations, a band of the index
without a usable curve in the pixel's stratum, or no defined share).

``geometry_trend`` is the Theil–Sen slope of the deseasonalised cos i. The interval ``share_low`` to
``share_high`` brackets the drift share: the drift artefact repeated with c ± :data:`BRACKET_SE`
standard errors of the stage-2 curve, spanning those two results and the central one. The bracket
is wider than ± 2 standard errors because the curve's standard error understates how much c
depends on which pixels were sampled (see :data:`BETWEEN_SAMPLE_RATIO`).
The NumPy twin of the per-pixel computation is :func:`topocorr._reference.diagnose_pixel`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import ee

from . import flags as _flags
from ._reference import INDICES, RISK_LIMITS
from .masking import SHADOW_SEARCH, _check_shadow_search, mask_bands
from .model import model_strata, strata_image
from .sensors import BANDS
from .stack import SENSOR_FAMILIES, Curve, StackModel
from .terrain import TERRAIN_SCALE, terrain

_DEG = math.pi / 180.0
_MS_PER_YEAR = 365.25 * 86400 * 1000.0
OUTPUT_BANDS = (
    "observed_trend",
    "drift_trend",
    "illumination_trend",
    "scaling_trend",
    "geometry_trend",
    "drift_share",
    "share_low",
    "share_high",
    "illumination_share",
    "risk_class",
    "n_obs",
)
"""Bands of the image returned by :func:`diagnose`."""


BETWEEN_SAMPLE_RATIO = 1.5
"""Between-sample standard deviation of c divided by the stage-2 curve's standard error.

Refitting the stack model on 10 random halves of the fixed points (Kalam and Ghandruk, Landsat 5/7
and 8-9, forest strata, solar zenith 30, 45 and 60°) spread c more than the curve's standard error
says (median 1.54 over 288 stratum and zenith cells): the curve error covers the scatter between
images, not the choice of pixels.
"""

BRACKET_SE = 2.0 * math.sqrt(1.0 + BETWEEN_SAMPLE_RATIO**2)
"""Half-width of the share bracket in curve standard errors: 2 standard deviations of c, with the
between-sample spread added to the curve's own error, ``2 sqrt(1 + r²)``."""


def _curve_value(curve: Curve, zenith: ee.Number, k_se: float = 0.0) -> ee.Number:
    """c(θz) + k_se · se(θz), evaluated in Earth Engine (held at the fitted range's edges)."""
    x = zenith.max(curve.zmin).min(curve.zmax).subtract(curve.center).divide(curve.scale)
    powers = [ee.Number(1)] + [x.pow(k) for k in range(1, len(curve.coef))]
    value = ee.Number(0)
    for a, p in zip(curve.coef, powers, strict=True):
        value = value.add(p.multiply(a))
    if k_se:
        var = ee.Number(0)
        for i, pi in enumerate(powers):
            for j, pj in enumerate(powers):
                var = var.add(pi.multiply(pj).multiply(curve.cov[i][j]))
        value = value.add(var.max(0).sqrt().multiply(k_se))
    return value


def _c_image(model: StackModel, band: str, stratum: ee.Image, zenith: ee.Number, k_se: float):
    """C of one band per pixel.

    Masked where the stratum has no usable curve, or where the central c (k_se = 0) is negative.
    The bracket values (k_se != 0) are clamped at 0 instead, so that the bracket never removes an
    observation that the central estimate uses.
    """
    codes = [code for code in model.codes if model.used[band][code] is not None]
    central = [_curve_value(model.curve(band, code), zenith) for code in codes]
    # Stratum -1 never occurs; it keeps the lists non-empty.
    c0 = stratum.remap([*codes, -1], [*central, -999]).unmask(-999)
    valid = c0.gte(0)
    if not k_se:
        return c0.updateMask(valid)
    shifted = [_curve_value(model.curve(band, code), zenith, k_se) for code in codes]
    c = stratum.remap([*codes, -1], [*shifted, 0]).unmask(0).max(0)
    return c.updateMask(valid)


def _index(name: str, b: Mapping[str, ee.Image]) -> ee.Image:
    if name == "evi":
        return (
            b["nir"]
            .subtract(b["red"])
            .multiply(2.5)
            .divide(b["nir"].add(b["red"].multiply(6)).subtract(b["blue"].multiply(7.5)).add(1))
        )
    first, second = INDICES[name][:2]
    nd = b[first].subtract(b[second]).divide(b[first].add(b[second]))
    return nd.multiply(b["nir"]) if name == "nirv" else nd


def _models_by_family(model: StackModel | Mapping[str, StackModel]) -> dict[str, StackModel]:
    if isinstance(model, StackModel):
        return {model.settings.get("sensor_family", "?"): model}
    if (
        isinstance(model, Mapping)
        and model
        and all(isinstance(m, StackModel) for m in model.values())
    ):
        return {m.settings.get("sensor_family", key): m for key, m in model.items()}
    raise TypeError("model must be a StackModel or a dict of StackModels, one per sensor family")


def diagnose(
    images: ee.ImageCollection,
    model: StackModel | Mapping[str, StackModel],
    *,
    index: str = "ndvi",
    dem: str | ee.Image = "glo30",
    harmonics: int = 2,
    min_obs: int = 20,
    trend_threshold: float = 0.001,
    shadow_search: int | None = None,
) -> ee.Image:
    """Map how much of each pixel's index trend comes from illumination (drift and total).

    Parameters
    ----------
    images : ee.ImageCollection
        Images from :func:`topocorr.collection` (raw reflectance).
    model : StackModel or dict of StackModel
        A stack model, or one per sensor family (Landsat 5/7, Landsat 8/9, Sentinel-2) for a
        series that spans families; each image uses its family's model.
    index : {"ndvi", "nbr", "ndmi", "evi", "nirv"}
        The vegetation index tested.
    dem : str or ee.Image
        The DEM the model was fitted with.
    harmonics : int
        Annual harmonics removed before the trends (1-3).
    min_obs : int
        Fewest usable observations of a pixel; fewer gives risk class 0.
    trend_threshold : float
        |observed trend| (index units per year) below which the risk class is 4, "no trend".
    shadow_search : int, optional
        Hill-shadow search distance in pixels; by default the model's value.

    Returns
    -------
    ee.Image
        Bands :data:`OUTPUT_BANDS` on the 30 m grid of the first image; properties
        ``tc_index``, ``tc_harmonics``, ``tc_min_obs`` and ``tc_sun_both_sides`` (1 when a
        model's images have the sun north of the tile in some images and south of it in others,
        :attr:`topocorr.StackModel.sun_both_sides`; ``diagnose`` then warns, as the
        artefact estimate rests on one c(zenith) curve that does not follow per-image c there).
    """
    if index not in INDICES:
        raise ValueError(f"unknown index {index!r}; choose one of {', '.join(INDICES)}")
    if not 1 <= int(harmonics) <= 3:
        raise ValueError("harmonics must be 1, 2 or 3")
    if int(min_obs) < 2 * int(harmonics) + 3:
        raise ValueError("min_obs must exceed the number of harmonic terms by at least 2")
    models = _models_by_family(model)
    sun_both_sides = _warn_sun_both_sides(models)
    first_model = next(iter(models.values()))
    strata = model_strata(first_model)
    edges = tuple(float(e) for e in first_model.settings["slope_classes"])
    steep = float(first_model.settings["steep"])
    search = int(shadow_search or first_model.settings.get("shadow_search", SHADOW_SEARCH))
    _check_shadow_search(search)
    needed = INDICES[index]
    harmonic_names = ["constant"] + [
        f"{f}{k}" for k in range(1, harmonics + 1) for f in ("cos", "sin")
    ]
    series = ["obs", "pred", "pred_lo", "pred_hi", "corr", "cos_i"]

    def geometry_of(image: ee.Image):
        """Terrain, masks, cos i and stratum of one image: computed once, used by every variant."""
        zenith = ee.Number(image.get("tc_sun_zenith"))
        projection = image.select(BANDS[3]).projection().atScale(TERRAIN_SCALE)
        ter = terrain(dem, projection)
        geometry = mask_bands(
            ter, zenith, ee.Number(image.get("tc_sun_azimuth")), steep=steep, shadow_search=search
        )
        ci = geometry.select("cos_i")
        usable = geometry.select("tc_flag").bitwiseAnd(_flags.EXCLUDED).eq(0)
        stratum = strata_image(strata, ter.select("slope"), edges)
        return zenith, ci, usable, stratum

    def factors(image: ee.Image, zenith, ci, stratum, k_se: float):
        cos_z = zenith.multiply(_DEG).cos()
        sensor = ee.String(image.get("tc_sensor"))
        out = {}
        for b in needed:
            per_family = []
            for family, m in models.items():
                members = [s for s, f in SENSOR_FAMILIES.items() if f == family] or [family]
                c = _c_image(m, b, stratum, zenith, k_se)
                per_family.append((ee.List(members).contains(sensor), c))
            c = ee.Image.constant(-999).updateMask(0)
            for is_family, c_family in reversed(per_family):
                c = ee.Image(ee.Algorithms.If(is_family, c_family, c))
            out[b] = ci.add(c).divide(c.add(cos_z))
        return out

    def per_image(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        zenith, ci, usable, stratum = geometry_of(image)
        f = factors(image, zenith, ci, stratum, 0.0)
        f_lo = factors(image, zenith, ci, stratum, -BRACKET_SE)
        f_hi = factors(image, zenith, ci, stratum, BRACKET_SE)
        t = ee.Number(image.get("system:time_start")).divide(_MS_PER_YEAR).add(1970)
        terms = [ee.Image.constant(1).rename("constant")]
        for k in range(1, harmonics + 1):
            w = t.multiply(2 * math.pi * k)
            terms += [
                ee.Image.constant(w.cos()).rename(f"cos{k}"),
                ee.Image.constant(w.sin()).rename(f"sin{k}"),
            ]
        bands = {b: image.select(b) for b in needed}
        flat = []  # rho / F per band, for the central c and the bracket
        for prefix, fac in (("flat_", f), ("flatlo_", f_lo), ("flathi_", f_hi)):
            flat += [bands[b].divide(fac[b]).rename(f"{prefix}{b}") for b in needed]
        corrected = _index(index, {b: bands[b].divide(f[b]) for b in needed}).rename("corr")
        stack = ee.Image.cat(
            [
                ee.Image.constant(t).rename("t").toFloat(),
                *terms,
                _index(index, bands).rename("obs"),
                ci.rename("cos_i"),
                corrected,
                *flat,
                *[f[b].rename(f"f_{b}") for b in needed],
                *[f_lo[b].rename(f"flo_{b}") for b in needed],
                *[f_hi[b].rename(f"fhi_{b}") for b in needed],
            ]
        ).toFloat()
        valid = usable.And(stack.select("obs").mask()).And(stack.select("corr").mask())
        for b in needed:  # every band of the index needs a usable curve
            valid = valid.And(stack.select(f"f_{b}").mask()).And(stack.select(f"flat_{b}").mask())
        return stack.updateMask(valid)

    stacked = images.map(per_image)
    flat_names = [f"{p}{b}" for p in ("flat_", "flatlo_", "flathi_") for b in needed]
    flat_median = stacked.select(flat_names).median()

    def with_prediction(img: ee.Image) -> ee.Image:
        """Index of F times the pixel's constant flat reflectance: the drift artefact series."""
        img = ee.Image(img)
        pred = []
        for name, prefix, flat_prefix in (
            ("pred", "f_", "flat_"),
            ("pred_lo", "flo_", "flatlo_"),
            ("pred_hi", "fhi_", "flathi_"),
        ):
            b_pred = {
                b: img.select(f"{prefix}{b}").multiply(flat_median.select(f"{flat_prefix}{b}"))
                for b in needed
            }
            pred.append(_index(index, b_pred).rename(name))
        return img.addBands(pred)

    full = stacked.map(with_prediction)
    regression = full.select([*harmonic_names, *series]).reduce(
        ee.Reducer.linearRegression(len(harmonic_names), len(series))
    )
    coefficients = regression.select("coefficients")  # (terms x series)

    def deseasonalised(img: ee.Image) -> ee.Image:
        img = ee.Image(img)
        x = img.select(harmonic_names[1:]).toArray().toArray(1).arrayTranspose()  # 1 x (terms-1)
        seasonal = x.matrixMultiply(coefficients.arraySlice(0, 1)).arrayProject([1])
        seasonal = seasonal.arrayFlatten([series])
        values = img.select(series).subtract(seasonal)
        return ee.Image.cat([img.select("t"), values])

    ds = full.map(deseasonalised)
    # One Sen's slope per series, combined into one pass: inputs (t, series) pairs in order.
    reducer = None
    for name in series:
        one = ee.Reducer.sensSlope().setOutputs([f"trend_{name}", f"offset_{name}"])
        reducer = one if reducer is None else reducer.combine(one, None, False)
    paired = ds.map(
        lambda img: ee.Image.cat(
            [ee.Image(img).select(["t", name], [f"t_{name}", f"y_{name}"]) for name in series]
        )
    )
    trend = paired.reduce(reducer).select([f"trend_{n}" for n in series])
    n_obs = full.select("obs").count().rename("n_obs")

    obs_t = trend.select("trend_obs")
    drift_t = trend.select("trend_pred")
    total_t = obs_t.subtract(trend.select("trend_corr"))
    share = drift_t.divide(obs_t).rename("drift_share")
    lo = trend.select("trend_pred_lo").divide(obs_t)
    hi = trend.select("trend_pred_hi").divide(obs_t)
    # The bracket always contains the central value (the share need not be monotone in c).
    share_low = lo.min(hi).min(share).rename("share_low")
    share_high = lo.max(hi).max(share).rename("share_high")
    # Classes are set only where the share is defined; everything else stays 0, not assessable.
    s = share.abs().unmask(-1)
    has_trend = obs_t.abs().unmask(-1)
    risk = (
        ee.Image.constant(0)
        .where(s.gte(0).And(s.lt(RISK_LIMITS[0])), 1)
        .where(s.gte(RISK_LIMITS[0]).And(s.lte(RISK_LIMITS[1])), 2)
        .where(s.gt(RISK_LIMITS[1]), 3)
        .where(has_trend.gte(0).And(has_trend.lt(trend_threshold)), 4)
        .where(n_obs.unmask(0).lt(min_obs), 0)
        .toUint8()
        .rename("risk_class")
    )
    return ee.Image.cat(
        [
            obs_t.rename("observed_trend"),
            drift_t.rename("drift_trend"),
            total_t.rename("illumination_trend"),
            total_t.subtract(drift_t).rename("scaling_trend"),
            trend.select("trend_cos_i").rename("geometry_trend"),
            share,
            share_low,
            share_high,
            total_t.divide(obs_t).rename("illumination_share"),
            risk,
            n_obs.toInt16(),
        ]
    ).set(
        {
            "tc_index": index,
            "tc_harmonics": harmonics,
            "tc_min_obs": min_obs,
            "tc_sun_both_sides": sun_both_sides,
        }
    )


def _warn_sun_both_sides(models: Mapping[str, StackModel]) -> int:
    """Warn and return 1 when any model has the sun on both sides of the tile, else 0."""
    if not any(m.sun_both_sides for m in models.values()):
        return 0
    warnings.warn(
        "the sun is north of the tile in some of the model's images and south of it in others "
        "(near the equator); the artefact estimate rests on one c(zenith) curve, which does not "
        "follow per-image c there, so read the artefact share with care (tc_sun_both_sides = 1)",
        UserWarning,
        stacklevel=3,
    )
    return 1
