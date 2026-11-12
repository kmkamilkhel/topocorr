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
- Package scaffold: packaging, tests on Linux, Windows and macOS, citation metadata.

[Unreleased]: https://github.com/kmkamilkhel/topocorr/commits/main
