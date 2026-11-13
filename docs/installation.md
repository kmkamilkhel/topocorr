# Installation

```bash
pip install topocorr
```

topocorr needs Python 3.11 or newer, NumPy and the Earth Engine Python API. You need a Google Earth
Engine account and a Cloud project registered for Earth Engine:

```bash
earthengine authenticate
```

```python
import ee

ee.Initialize(project="my-project")
```

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `local` | pystac-client, planetary-computer, rasterio, pyarrow | reading Planetary Computer data next to Earth Engine (the NumPy reference path) |

```bash
pip install "topocorr[local]"
```

`compare` returns a pandas DataFrame when pandas is installed, and a list of dicts otherwise.

The [example notebooks](examples.md) have their own environment file with eofigs for the figures.

## Earth Engine use

Earth Engine's noncommercial tiers meter compute in EECU-hours. topocorr keeps heavy work in batch
exports and does the rest locally. Measured on a 24 km Landsat tile (ranges: several runs; Earth
Engine's use of the same export varies between runs):

| Step | EECU-hours |
|---|---|
| `diagnose` map, Landsat 5 (156-172 images) | 1.7-3.9 |
| `diagnose` map, Landsat 8-9 (269 images) | 6.1 |
| `compare_export`, Landsat 8-9 (5,761 points x 269 images; CSV 113 MB) | 0.6-2.4 |
