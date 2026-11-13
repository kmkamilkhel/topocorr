# Illumination and corrections

## Local illumination

For a pixel with slope *s* and aspect *a*, lit by the sun at zenith θz and azimuth φ, the cosine of
the local incidence angle is

cos i = cos θz cos s + sin θz sin s cos(φ − a).

topocorr computes slope and aspect from the DEM on each image's 30 m grid (Copernicus GLO-30 by
default; NASADEM, SRTM, ALOS AW3D30 or any elevation image), with aspect measured from true north,
and the sun angles from the scene metadata. `masks` returns cos i with the flags in the bit band
`tc_flag`:

| Flag | Meaning | Corrected? |
|---|---|---|
| cast shadow | the terrain blocks the sun (hill-shadow search) | no, keeps its value |
| self-shadow | cos i ≤ 0: the slope faces away from the sun | no |
| steep | slope above 60° | no |
| low illumination | 0 < cos i < 0.1 | yes, flagged |
| no fit | the band's fit was not usable | no |
| no model | the stack model does not cover the pixel's stratum | no |

## The corrections

| Method | Corrected reflectance | c from |
|---|---|---|
| `cosine` | ρ cos θz / cos i | none (c = 0) |
| `c` | ρ (cos θz + c) / (cos i + c) | a fit per image and band |
| `scsc` | ρ (cos s cos θz + c) / (cos i + c) | a fit per image and band |
| `scsc-stack` (default) | ρ (cos s cos θz + c) / (cos i + c) | the [stack model](stack-model.md), c(θz) |
| `c-stack` | ρ (cos θz + c) / (cos i + c) | the stack model, c(θz) |
| `se-stack` | ρ − m_t (cos i − cos θz) | the image's own slope m_t from the stack model's first stage |

For the per-image methods, c = n / m comes from a least-squares fit ρ = m cos i + n per band and
image, on a sample of unflagged tree-cover pixels (ESA WorldCover 2021) with slope of at least 5°.
A band's fit is used only when it has enough pixels, m > 0 and c ≥ 0; otherwise the band is left
unchanged and flagged "no fit". Fitting over all land covers can bias c: dense forest on shaded
slopes and open land on sunlit ones make brightness follow aspect.

**SCS+C or C?** SCS+C normalises a pixel to a sloped canopy under the sun (cos s cos θz), C to
flat ground (cos θz). For a single image the difference is small. Through time, SCS+C keeps the
factor (cos s cos θz + c) / (cos θz + c), which changes with the sun angle on slopes, so a part of
an orbit-drift trend can stay; `c-stack` removes it when the model is right, at the cost of more
noise on steep slopes. See [Comparing methods](compare.md).

**Statistical-empirical.** `se-stack` refers every image to a horizontal surface,
ρ − m_t (cos i − cos θz). Teillet et al. (1982) used the scene mean of cos i instead; that mean
changes from image to image (sun position, cloud mask), which would add a per-image offset to a
time series.

## Every image is the same stack

Use one sensor family per model and per series (Landsat 5/7 or Landsat 8/9). Ratio indices must
come from bands corrected together: the band `tc_corrected` records which bands were corrected at
each pixel, and `normalized_difference` flags mixed pixels.
