# topocorr

[![Tests](https://github.com/kmkamilkhel/topocorr/actions/workflows/tests.yml/badge.svg)](https://github.com/kmkamilkhel/topocorr/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Is a mountain trend real, or is it the sun?** topocorr tests Landsat and Sentinel-2 time series
for illumination artefacts on slopes and corrects them consistently through time, in Google Earth
Engine.

On steep terrain the reflectance of a pixel depends on how directly the sun lights it. That
illumination changes between acquisitions with the season, with orbit drift (Landsat 5 and 7) and
with each sensor switch, so a vegetation index can show greening, browning or a break that comes
from geometry, not from the vegetation. topocorr computes the illumination of every pixel and
acquisition from the DEM and the image's sun angles, and:

- flags cast shadow, self-shadow, low illumination and very steep slopes;
- applies the standard corrections (cosine, C, SCS+C), fitted image by image;
- fits a stack model once per site: the C coefficient as a smooth function of solar zenith, per
  band and stratum (land cover x slope class), so no image gets its own noisy coefficient;
- will estimate how much of an observed trend or break the illumination history alone predicts
  (in development).

```python
import ee
import topocorr

ee.Initialize(project="my-project")
aoi = ee.Geometry.Rectangle([72.44, 35.41, 72.71, 35.64])  # upper Swat, Pakistan
oli = topocorr.collection("L8", aoi, "2013-01-01", "2026-01-01", max_cloud=30)

# Image by image; c is fitted on tree cover (ESA WorldCover 2021) unless fit_mask says otherwise
corrected = topocorr.correct(oli, method="scsc", region=aoi)

# Stack model: c(solar zenith) per band and stratum, fitted once (one sensor family per fit)
model = topocorr.fit(oli, region=aoi, cache="swat_l8_fit.json")  # resumes if interrupted
print(model.summary())
model.to_json("swat_l8_model.json")
stack_corrected = topocorr.correct(oli, model, method="scsc-stack")  # or "se-stack"

flags = topocorr.masks(oli)  # cos_i, cast_shadow, self_shadow, low_illumination, steep
```

Sensors: Landsat 5, 7, 8 and 9 Collection 2 Level 2 and Sentinel-2 L2A. Default DEM: Copernicus GLO-30.

## Status

In development; no release yet. Available: `collection`, `masks`, `fit` (stack model) and
`correct` (cosine, C, SCS+C per image; SCS+C and statistical-empirical with the stack model).
The first release (0.1) will add `diagnose` and `compare`.

## Installation

From GitHub until the first release:

```bash
pip install git+https://github.com/kmkamilkhel/topocorr
```

topocorr needs a Google Earth Engine account and a Cloud project registered for Earth Engine
(`earthengine authenticate`).

## License

MIT. See [LICENSE](LICENSE).
