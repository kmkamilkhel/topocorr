# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Citation: Zenodo DOIs (all versions 10.5281/zenodo.24195254; version 0.1.0
  10.5281/zenodo.24195255) in CITATION.cff, README and the documentation; title as on Zenodo.
- Documented Earth Engine use as measured ranges: `diagnose` map, Landsat 5, 1.7-3.9 EECU-hours;
  `compare_export`, Landsat 8-9, 0.6-2.4 EECU-hours (the same export varies between runs).

## [0.1.0] - 2026-11-13

### Added

- `collection()`: Landsat 5, 7, 8 and 9 Collection 2 Level 2 and Sentinel-2 L2A as one harmonised
  form: bands `blue`, `green`, `red`, `nir`, `swir1`, `swir2` (Sentinel-2 NIR is B8A), surface
  reflectance, clouds, cloud shadow, snow, fill and saturation masked (Sentinel-2 SCL class 2,
  which contains terrain shadow, is kept), and the scene sun angles as `tc_sun_zenith` and
  `tc_sun_azimuth`.
- `masks()`: cos i, cast shadow, self-shadow, low illumination (0 < cos i < 0.1, flag only) and
  steep slopes (above 60°) per image, combined in the bit band `tc_flag` (`topocorr.flags`).
- `correct()` with the cosine, C and SCS+C methods; c is fitted per image and band on a sample of
  unflagged pixels with slope of at least `min_slope` (5°), by default on tree cover only (ESA
  WorldCover 2021; `fit_mask` takes another mask, or `None` for all pixels). Bands without a usable fit (m <= 0 or c < 0) are left unchanged and flagged; pixels
  in shadow or on steep slopes keep their input value.
- DEMs: Copernicus GLO-30 (2024 release, default), NASADEM, SRTM, ALOS AW3D30 or any elevation
  image. Slope and aspect from the four neighbours on the image's 30 m grid; aspect is measured
  from true north (UTM grid aspect rotated by the meridian convergence).
- `fit()`: the stack model. Per image, band and stratum (ESA WorldCover land cover x slope
  class 5-25, 25-40, 40-60°), `ρ = m cos i + n` on a stratified sample; across the stack,
  `c = n / m` as a polynomial in solar zenith (degree 2), fitted by weighted robust regression
  (weights from each image's sampling variance plus the scatter between images, Tukey's
  biweight). Strata with too few usable images use their land cover's curve over all slopes, or
  are reported as not assessable. One sensor family per fit. Requests are batched, retried when
  Earth Engine is busy, and can be cached so an interrupted fit resumes.
- `StackModel` (`summary()`, `c()`, `image_fits()`, `to_json()`) and `load_model()`.
- `correct(..., model, method="scsc-stack")`: SCS+C with c from the stack model; and
  `method="se-stack"`: statistical-empirical correction (Teillet et al. 1982) with each image's
  own slope m from the model's first stage, relative to a horizontal surface.
- Flag `no_model` (32): pixels whose stratum the stack model does not cover; not corrected.
- `strata="worldcover-density"` for `fit()`: WorldCover tree cover split by Hansen tree cover in
  2000 (below 30, 30-50, 50-70, at least 70 %); with it, pixels with any recorded Hansen forest
  loss are left out of the fit (they are still corrected).
- Band `tc_corrected` from every correction method: bit i set where band i was corrected
  (`topocorr.flags.BAND_BITS`, `topocorr.flags.mixed`); and `normalized_difference()`, which adds
  `<name>_mixed` where one band of the pair was corrected and the other was not.
- The statistical-empirical method refers every image to a horizontal surface, not to the scene
  mean of cos i as in Teillet et al. (1982), so no per-image offset enters a time series.
- `diagnose()`: how much of each pixel's index trend (NDVI, NBR, NDMI, EVI, NIRv) the
  illumination history alone predicts. From the stack model, each acquisition's illumination
  factor (cos i + c)/(cos θz + c) per band; observed and predicted series are deseasonalised with
  the same harmonics; Theil-Sen trends give `observed_trend`, `artefact_trend`, `geometry_trend`,
  `artefact_share` with a `share_low`-`share_high` bracket (c ± 2 standard errors of the curve) and
  `risk_class` (low, medium, high, no trend, not assessable). A model per sensor family can be
  given for series that span Landsat 5/7 and 8/9. NumPy twin: `_reference.diagnose_pixel`.
- `collection(..., cloud_buffer=3)`: cloud and cloud shadow are grown by 3 pixels before masking
  (0 keeps the QA masks as they are). Undetected shadow rims on sunlit slopes had raised the fitted
  c in every band but NIR on a Kalam test scene.
- `shadow_search` (pixels) for `masks()`, `correct()`, `fit()` (stored in the model) and
  `diagnose()`, and `masking.shadow_search(region, max_zenith)`: the longest shadow the terrain can
  cast, from the relief (region grown by that length) times tan(largest solar zenith).
- `correct(..., model, method="c-stack")`: the C correction with c from the stack model,
  normalised to flat ground. Unlike SCS+C, which keeps the factor (cos s cos θz + c) / (cos θz + c)
  on slopes, it leaves no trend from a change of sun angle when the model is right.
- `compare_export()` and `compare()`: methods compared on one stack in two steps. `compare_export`
  draws fixed points (`n_points` per stratum of the stack model, seeded) and starts two
  `Export.table` tasks (every image at the points: six bands, sun angles, `tc_flag`; and per point:
  stratum, slope, aspect, elevation, forest loss), with a JSON manifest and the model. `compare`
  reads the downloaded tables and applies every method in NumPy, with no Earth Engine calls:
  `none`, `cosine`, `c`, `scsc`, `scsc-stack`, `c-stack`, `se-stack`. Per method: the median
  correlation of NIR with cos i, the sunlit-shaded gap, temporal noise of NIR and an index (NIRv by
  default), the share of values out of range, and the predicted drift left (from the stack model):
  each point's illumination-only series passed through the method, and the median remaining trend
  with the share of points at 0.001 per year or more. One Landsat 8-9 tile (269 images, 5,761
  points): 0.58 EECU-hours, 113 MB CSV.
- Example notebooks (`examples/`): Kalam (Landsat 5 orbit drift, stack models, NIRv drift map,
  method comparison) and Tapantí (the sun on both sides of the tile near the equator), with their
  Earth Engine cost, progress bars and resumable exports.
- Documentation site (mkdocs-material, GitHub Pages): concepts, validation summary, gallery
  (maps from the development version, labelled), example notebooks and API reference.
- Package scaffold: packaging, tests on Linux, Windows and macOS, citation metadata.

### Changed

- `correct()` uses `method="scsc-stack"` when no method is given (`method` was required); without
  a model it asks for one or for a per-image method.
- `correct(..., method="scsc-stack")` warns and uses per-image SCS+C when the model's images have
  the sun north of the tile in some images and south of it in others (`StackModel.sun_both_sides`,
  each side at least 5 % of the images; near the equator). There per-image c differs between the
  two sides and drifts with Landsat 5's overpass, which one c(zenith) curve cannot follow.
  `stack.sun_sides()` counts the images per side.
- `correct(..., method="c-stack")` falls back to per-image C in the same case.
- `compare()` gives the same warning in that case and records `tc_sun_both_sides` (1, else 0) in its
  table; it applies every method as named.
- `diagnose()` gives the same warning for such a model and records it in the property
  `tc_sun_both_sides` (1, else 0); its numbers are unchanged.
- `diagnose()` total illumination effect: the observed trend minus the trend of the
  illumination-corrected series (each band divided by its illumination factor F), on the same
  observations, in Earth Engine and in `_reference.diagnose_pixel`. The trend of F times one
  constant flat reflectance (now the drift artefact, below) misses that the illumination effect
  scales with a changing reflectance level: on synthetic data with known truth (validation test 1,
  real Kalam and Ghandruk geometry) it recovered the NIRv share against flat ground within 0.1 for
  only 35-43 % of Kalam pixels, against 87-97 % for the difference of trends.
- `diagnose()` reports two illumination parts of the observed trend: the drift artefact
  (`drift_trend`, `drift_share`: the trend of F times the pixel's constant flat reflectance, the
  trend that the changing illumination itself causes) and the total illumination effect
  (`illumination_trend`, `illumination_share`: observed minus illumination-corrected trend), with
  `scaling_trend` = total − drift (a real trend scaled by the pixel's mean illumination). The risk
  class and the share bracket use the drift share. Bands `artefact_trend` and `artefact_share` are
  replaced by these.
- The share bracket (`share_low`, `share_high`) uses c ± `BRACKET_SE` (3.6) curve standard errors
  instead of ± 2: the curve's standard error understates how much c depends on the sampled pixels
  (between-sample SD about 1.5 times the standard error, `BETWEEN_SAMPLE_RATIO`).
- README: use NIRv or EVI for drift checks; in NDVI the drift artefact is near the noise level.
- Optional extra `local` (pystac-client, planetary-computer, rasterio, pyarrow) for reading
  Planetary Computer data next to Earth Engine.

[Unreleased]: https://github.com/kmkamilkhel/topocorr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kmkamilkhel/topocorr/releases/tag/v0.1.0
