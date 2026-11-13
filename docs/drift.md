# Drift artefact and `diagnose`

## Orbit drift

Landsat 5 crossed the equator later and later between 1995 and 2007, in steps of up to about an
hour (Roy et al. 2020; Qiu et al. 2021). At the same time of year the sun then stood higher and
further towards the afternoon. West-facing slopes were lit more directly over the years and
east-facing slopes less, so uncorrected series show greening on west-facing and browning on
east-facing slopes. Landsat 7 drifted later in its mission; sensor switches (Landsat 5/7 to 8/9)
change the overpass time again.

## What `diagnose` measures

For each pixel, the stack model gives each acquisition's illumination factor per band,

F = (cos i + c) / (cos θz + c),

the reflectance of the slope relative to flat ground. `diagnose` deseasonalises the series with
annual harmonics and fits Theil-Sen trends:

| Band | Meaning |
|---|---|
| `observed_trend` | trend of the index as observed |
| `drift_trend` | **drift artefact**: the trend of F x the pixel's constant flat reflectance, the trend that the changing illumination alone creates |
| `drift_share` | drift trend / observed trend; `share_low`, `share_high` bracket it for the uncertainty of c |
| `illumination_trend` | **total illumination effect**: observed trend minus the trend of the illumination-corrected series (each band divided by F) |
| `illumination_share` | illumination trend / observed trend |
| `scaling_trend` | total minus drift: a real trend scaled by the pixel's mean illumination |
| `geometry_trend` | trend of the illumination geometry |
| `risk_class` | from the drift share: low, medium, high, no trend (observed trend below 0.001 per year), not assessable |
| `n_obs` | clear observations used |

The drift artefact can create or flip a trend with no vegetation change; the risk class is based on
it. The total illumination effect also contains the scaling of a real trend: on a steep shaded slope
a real greening is seen smaller than on flat ground.

## Which index

The drift artefact is strongest in NIR and NIR-based indices. NDVI cancels part of it: on the test
tiles the relative artefact in NIRv and EVI was 5 to 40 times larger than in NDVI, and in NDVI it
is often near the noise level, so an NDVI "medium" can still be a real artefact. **Use NIRv or EVI
for drift checks** (`index="nirv"`).

## Example

```python
risk = topocorr.diagnose(l5, model_l5, index="nirv")
# a series over two sensor families: one model per family
risk = topocorr.diagnose(l5.merge(oli), {"tm": model_l5, "oli": model_oli}, index="nirv")
```

A per-pixel series of more than about 200 images can run out of memory in Earth Engine; keep a map
to one sensor family or split the region. See the [Gallery](gallery.md) for maps.
