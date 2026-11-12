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
- estimates how much of an observed index trend the illumination history alone predicts
  (`diagnose`).

```python
import ee
import topocorr

ee.Initialize(project="my-project")
aoi = ee.Geometry.Rectangle([72.44, 35.41, 72.71, 35.64])  # upper Swat, Pakistan
oli = topocorr.collection("L8", aoi, "2013-01-01", "2026-01-01", max_cloud=30)

# Image by image; c is fitted on tree cover (ESA WorldCover 2021) unless fit_mask says otherwise
corrected = topocorr.correct(oli, method="scsc", region=aoi)

# Stack model: c(solar zenith) per band and stratum, fitted once (one sensor family per fit).
# "worldcover-density" splits tree cover by canopy density (Hansen tree cover in 2000).
model = topocorr.fit(oli, strata="worldcover-density", region=aoi, cache="swat_l8_fit.json")
print(model.summary())
model.to_json("swat_l8_model.json")
stack_corrected = topocorr.correct(oli, model)  # scsc-stack (default); or method="se-stack"

flags = topocorr.masks(oli)  # cos_i, cast_shadow, self_shadow, low_illumination, steep

# Ratio indices only from bands corrected together: ndvi_mixed = 1 where nir and red were not
ndvi = topocorr.normalized_difference(stack_corrected, "nir", "red", "ndvi")

# Is the NDVI trend real or illumination? Trends, artefact share and risk class per pixel
risk = topocorr.diagnose(oli, model, index="ndvi")
```

**Ratio indices.** A band whose fit is not usable is left unchanged (in winter forest this is often
red), so a corrected image can hold corrected and uncorrected bands side by side. The band
`tc_corrected` records, per pixel, which bands were corrected. Compute NDVI, NBR and other ratio
indices only from bands corrected together, or from the raw bands; never from a mix.
`topocorr.normalized_difference` flags the pixels where the two bands differ.

**Canopy density.** On mountain slopes canopy density often follows aspect: dense forest on shaded
slopes, open forest on sunlit ones. A fit over all forest then learns composition as well as
illumination. `strata="worldcover-density"` splits tree cover by Hansen tree cover in 2000 (below 30,
30-50, 50-70, at least 70 %) and leaves pixels with recorded forest loss out of the fit. The
density layer is from 2000.

**Orbit drift: east- and west-facing slopes.** Landsat 5 crossed the equator later and later
between 1995 and 2007, in steps (up to about an hour; Roy et al. 2020, Qiu et al. 2021). The sun
then stood higher and further towards the afternoon, so west-facing slopes were lit more
directly over the years and east-facing slopes less. In uncorrected series this shows as greening
on west-facing and browning on east-facing slopes. It is strongest in NIR and NIR-based indices
(NIRv, EVI); NDVI cancels part of it. `diagnose` maps where it is a large part of the trend, and
the stack correction removes it.

**Near the equator.** Where the sun is north of the tile in some images and south of it in others
(`model.sun_both_sides`), one c(zenith) curve cannot follow per-image c, and `scsc-stack` would
bias trends on east- and west-facing slopes. `correct` then warns and uses per-image SCS+C. `diagnose`
warns too and sets the property `tc_sun_both_sides`.

Sensors: Landsat 5, 7, 8 and 9 Collection 2 Level 2 and Sentinel-2 L2A. Default DEM: Copernicus GLO-30.

## Status

In development; no release yet. Available: `collection`, `masks`, `fit` (stack model),
`correct` (cosine, C, SCS+C per image; SCS+C and statistical-empirical with the stack model) and
`diagnose`. The first release (0.1) covers Landsat 5, 7, 8 and 9 and adds `compare`; Sentinel-2
harmonisation and break checks follow in 0.2.

## Installation

From GitHub until the first release:

```bash
pip install git+https://github.com/kmkamilkhel/topocorr
```

topocorr needs a Google Earth Engine account and a Cloud project registered for Earth Engine
(`earthengine authenticate`).

## References

- Qiu, S., Zhu, Z., Shang, R., Crawford, C. J. (2021). Can Landsat 7 preserve its science
  capability with a drifting orbit? *Science of Remote Sensing* 4, 100026.
  https://doi.org/10.1016/j.srs.2021.100026
- Roy, D. P., Li, Z., Zhang, H. K., Huang, H. (2020). A conterminous United States analysis of the
  impact of Landsat 5 orbit drift on the temporal consistency of Landsat 5 Thematic Mapper data.
  *Remote Sensing of Environment* 240, 111701. https://doi.org/10.1016/j.rse.2020.111701

## License

MIT. See [LICENSE](LICENSE).
