# Comparing methods

Which correction works best on your stack? `compare` answers in two steps, so the Earth Engine cost
does not grow with the number of methods:

1. **`compare_export`** (Earth Engine) draws fixed sample points (`n_points` per stratum of the
   stack model, seeded) and starts two `Export.table` tasks to Google Drive: the raw table (every
   image at the points: the six reflectance bands, sun angles, `tc_flag`) and the static table
   (per point: stratum, slope, aspect, elevation, recorded forest loss, coordinates). It writes a
   manifest (JSON) and a copy of the stack model next to it.
2. **`compare`** (local, NumPy) reads the two downloaded CSV files, applies every method and
   computes the measures at the forest points. It makes no Earth Engine calls.

```python
topocorr.compare_export(oli, model, region=aoi, n_points=500, manifest="swat_compare.json")
# ... when both tasks are done, download the two CSV files next to the manifest
table = topocorr.compare("swat_compare.json")  # one row per method
```

For one 24 km Landsat 8-9 tile (269 images, 5,761 points) the export used 0.6-2.4 EECU-hours in
two runs (Earth Engine's use varies between runs of the same export) and wrote a 113 MB CSV; `compare` then took about 30 s.

## Measures

| Column | Meaning | Best |
|---|---|---|
| `illumination_r` | median over images of the correlation between NIR and cos i | 0 (negative: over-corrected) |
| `sunlit_shaded_gap` | median over images of (mean NIR, sunlit third - shaded third) / mean NIR | 0 |
| `temporal_sd_nir`, `temporal_sd_<index>` | median over points of the residual after a linear trend and annual harmonics | low |
| `out_of_range` | share of values below 0 or above 1 after correction | low |
| `predicted_drift_left` | predicted drift left (from the stack model), median over points, index units per year | 0 |
| `predicted_drift_high` | share of points whose predicted drift left reaches 0.001 per year | 0 |

**Predicted drift left (from the stack model).** Each point's illumination-only series (its
constant flat reflectance x F, the drift series of [`diagnose`](drift.md)) is passed through the
method, and the trend that remains in the deseasonalised index is taken. For `none` this is the
drift trend itself. `c-stack` leaves none wherever the model's c is usable; a band whose c is
negative (often red in forest) is left uncorrected and keeps its share of the drift. Per-image
methods leave trends of their own where their per-image c differs from image to image.

This measure **assumes that the stack model is right**: it shows what a method leaves of the drift
the model predicts, not a drift measured in the data. The data-based checks are `diagnose`
(observed trend against drift trend) and the tests summarised on the [Validation](validation.md)
page. A data-based drift sensitivity was tried and dropped: on Landsat
8-9 the sun angle barely changes between years at the same date, so the remaining sun-angle
variation is within the season, where phenology also varies and dominates.

## An example: Kalam

| method | illum. r | sunlit-shaded gap | temporal SD NIRv | predicted drift left, Landsat 5 (NIRv per year) |
|---|---|---|---|---|
| none | 0.72 | 0.66 | 0.0212 | 2.7 x 10⁻⁴ |
| cosine | -0.23 | -0.44 | 0.0347 | 1.2 x 10⁻⁴ (10 % of points ≥ 0.001) |
| C per image | 0.01 | 0.00 | 0.0298 | 6.3 x 10⁻⁵ (1.3 % ≥ 0.001) |
| SCS+C per image | 0.14 | 0.09 | 0.0284 | 7.2 x 10⁻⁵ |
| `scsc-stack` | 0.17 | 0.13 | 0.0239 | 3.0 x 10⁻⁵ |
| `c-stack` | 0.07 | 0.09 | 0.0269 | 0 |
| `se-stack` | 0.06 | 0.03 | 0.0209 | 8.3 x 10⁻⁵ |

Kalam, upper Swat (35.5° N): the first three columns from Landsat 8-9 (2013-2025), the last from
Landsat 5 (1990-2011; 172 images, 5,841 forest points). On Landsat 8-9 there is little drift to
remove: the drift trend is 7 x 10⁻⁶ per year, `scsc-stack` and `c-stack` leave about 1.5 x 10⁻⁵
(bands with c < 0 stay uncorrected), `se-stack` 7.9 x 10⁻⁵, while per-image C and SCS+C leave 1.3-1.7 x 10⁻⁴ with 0.7-2.3 % of points
at 0.001 or more, and cosine 6.7 % of points.
