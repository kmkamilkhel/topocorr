# Validation

Validated on synthetic data with known truth and on real Landsat time series at mountain forest
sites on four continents; full results in a forthcoming paper. This page summarises the tests of the
method itself and of its Earth Engine implementation. They were run with the development version.

## Does `diagnose` recover a known drift artefact?

Synthetic series with a known truth on real geometry: the sun angles, terrain and stack-model c of
the Kalam (35.5° N) and Ghandruk (28.4° N) tiles, with trends and noise added, for Landsat 5,
Landsat 5+7 and Landsat 8-9, NDVI and NIRv, with exact c and with c perturbed by its uncertainty.
Criteria were fixed before the run.

| Criterion | Required | Result |
|---|---|---|
| drift share within ±0.1 of the truth | ≥ 90 % of cases | 90.2-100 % in all 24 series |
| class high where the true drift share is ≥ 0.7 | ≥ 90 % | NIRv 99.6-100 %; NDVI below 90 % in 7 of the 8 Landsat 5 and 5+7 series (see below) |
| class medium or high where it is ≥ 0.5 | ≥ 95 % | 99.4-100 % |
| false high in the no-drift control | < 5 % | 0 % |

**Limit: NDVI.** Where the true drift share is at least 0.7, NDVI is classed high in only 80-90 %
of cases at Ghandruk and in 60-80 % at Kalam, where such cases are few (3-5; Landsat 5 and 5+7). These are small NDVI drift trends
near the NDVI noise level: noise inflates the observed trend, so the share comes out near 0.4 and
the class medium. All misses are classed at least medium. **Use NIRv or EVI for drift checks.**
Landsat 8-9 had no high-drift cases, so detection there is untested (share recovery 98.8-100 %).

## Is the uncertainty of c realistic?

The stack curve's standard error understates how much c depends on the sampled pixels: refitting on
independent samples gave a between-sample spread about 1.5 times the standard error (median 1.54
over the stratum-band cells of the Kalam and Ghandruk tiles). The `share_low`-`share_high` bracket
of `diagnose` therefore uses c ± 3.6 standard errors.

## Earth Engine against an independent NumPy path

Four Kalam Landsat 8 scenes (solar zenith 22-63°), 2,000 pixels: topocorr in Earth Engine against
the NumPy reference (`topocorr._reference`) on Microsoft Planetary Computer data (Landsat COGs,
Copernicus GLO-30 tiles).

| Quantity | Agreement |
|---|---|
| Reflectance and clear mask | identical (≤ 3 x 10⁻⁸; mask 100 %) |
| Elevation | median 0.14 m |
| Slope / aspect | median 0.03° / 0.09° |
| cos i | 99th percentile 0.003-0.006 |
| Flags | 99.8-99.95 % |
| SCS+C with the same c and terrain | identical (≤ 3 x 10⁻⁸) |

## Known limits

- NDVI drift checks (above): use NIRv or EVI.
- The drift artefact rests on the stack model; near the equator one c(zenith) curve cannot follow
  per-image c (`sun_both_sides`).
- `compare` reports the predicted drift left from the stack model, not a drift measured in the
  data; see [Comparing methods](compare.md).
- Sentinel-2 is not validated in 0.1.

The results at the individual sites will be added here, with the paper's DOI, once the method paper
is published.
