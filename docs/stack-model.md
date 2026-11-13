# The stack model

A per-image C correction fits c on each image separately. In a time series this adds noise (each
image gets its own c from a few hundred pixels) and fails where an image has too little contrast
in cos i, often in winter. The stack model fits c once per site, as a smooth function of solar
zenith:

1. **Stage 1, per image** (Earth Engine): for each band and stratum, a least-squares fit
   ρ = m cos i + n on a stratified sample (500 pixels per stratum by default), returned as six sums
   per image. Images are processed in batches, retried when Earth Engine is busy, and cached, so an
   interrupted fit continues where it stopped.
2. **Stage 2, across the stack** (NumPy): c_t = n_t / m_t as a polynomial in solar zenith (degree 2),
   fitted by weighted robust regression. Weights combine each image's sampling variance (delta
   method) with the scatter between images (Paule-Mandel), and Tukey's biweight down-weights
   outlying images.

**Strata.** Land cover (ESA WorldCover) x slope class (5-25, 25-40, 40-60°). With
`strata="worldcover-density"`, tree cover is split by Hansen tree cover in 2000 (below 30, 30-50,
50-70, at least 70 %), and pixels with recorded forest loss are left out of the fit. On mountain
slopes canopy density often follows aspect (dense forest on shaded slopes, open forest on sunlit
ones), so a fit over all forest learns composition as well as illumination. Strata with too few
usable images use their land cover's curve over all slopes, or are reported as not assessable.

**One sensor family per fit.** Landsat 5/7 and Landsat 8/9 differ in band passes and calibration.
For a series that spans both, fit one model per family and give `diagnose` a model for each.

```python
model = topocorr.fit(oli, strata="worldcover-density", region=aoi, cache="fit_cache.json")
print(model.summary())  # curves, images used, strata with fallback or not assessable
model.c("nir", code, 45.0)  # c of a stratum at a zenith
model.to_json("model.json")  # topocorr.load_model("model.json")
```

The model also records the sun azimuths of its images. `model.sun_both_sides` is true when the sun
is north of the tile in some images and south of it in others ([near the equator](equator.md)).
