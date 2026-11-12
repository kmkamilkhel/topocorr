# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- Package scaffold: packaging, tests on Linux, Windows and macOS, citation metadata.

### Changed

- `correct()` uses `method="scsc-stack"` when no method is given (`method` was required); without
  a model it asks for one or for a per-image method.
- `correct(..., method="scsc-stack")` warns and uses per-image SCS+C when the model's images have
  the sun north of the tile in some images and south of it in others (`StackModel.sun_both_sides`,
  each side at least 5 % of the images; near the equator). There per-image c differs between the
  two sides and drifts with Landsat 5's overpass, which one c(zenith) curve cannot follow; on test
  tiles at 4 and 10 degrees latitude it turned a west-minus-east difference of Landsat 5 NIR trends
  negative. `stack.sun_sides()` counts the images per side.

[Unreleased]: https://github.com/kmkamilkhel/topocorr/commits/main
