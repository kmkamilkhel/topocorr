# Quickstart

```python
import ee
import topocorr

ee.Initialize(project="my-project")
aoi = ee.Geometry.Rectangle([72.44, 35.41, 72.71, 35.64])  # upper Swat, Pakistan

# Harmonised Landsat 8-9 surface reflectance: bands blue, green, red, nir, swir1, swir2,
# clouds, shadow, snow and fill masked, sun angles as tc_sun_zenith and tc_sun_azimuth
oli = topocorr.collection("L8", aoi, "2013-01-01", "2026-01-01", max_cloud=30)

# Illumination and flags per image: cos_i and the bit band tc_flag
flags = topocorr.masks(oli)

# Image by image; c is fitted on tree cover (ESA WorldCover 2021) unless fit_mask says otherwise
corrected = topocorr.correct(oli, method="scsc", region=aoi)

# Stack model: c(solar zenith) per band and stratum, fitted once (one sensor family per fit);
# an interrupted fit continues from its cache
model = topocorr.fit(oli, strata="worldcover-density", region=aoi, cache="swat_l8_fit.json")
print(model.summary())
model.to_json("swat_l8_model.json")

stack_corrected = topocorr.correct(oli, model)  # scsc-stack (default); or "c-stack", "se-stack"

# Ratio indices only from bands corrected together: ndvi_mixed = 1 where nir and red were not
ndvi = topocorr.normalized_difference(stack_corrected, "nir", "red", "ndvi")

# Is the trend real or illumination? Drift artefact, total effect and risk class per pixel
risk = topocorr.diagnose(oli, model, index="nirv")
```

Everything above builds Earth Engine computations; nothing runs until you export or request a
result. Maps are best exported on the Landsat scene grid of the first image:

```python
proj = ee.Image(oli.first()).select("nir").projection().getInfo()
task = ee.batch.Export.image.toDrive(
    image=risk.toFloat(),
    description="swat_diagnose",
    folder="topocorr_exports",
    region=aoi,
    crs=proj["crs"],
    crsTransform=proj["transform"],
    maxPixels=1e9,
)
task.start()
```

To compare the methods on your own stack, see [Comparing methods](compare.md). The
[example notebooks](examples.md) run the full workflow for two sites, with their Earth Engine cost.

## Ratio indices

A band whose fit is not usable is left unchanged (in winter forest this is often red), so a
corrected image can hold corrected and uncorrected bands side by side. The band `tc_corrected`
records, per pixel, which bands were corrected. Compute NDVI, NBR and other ratio indices only from
bands corrected together, or from the raw bands; never from a mix.
`topocorr.normalized_difference` flags the pixels where the two bands differ.
