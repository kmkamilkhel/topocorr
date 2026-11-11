# topocorr

[![Tests](https://github.com/kmkamilkhel/topocorr/actions/workflows/tests.yml/badge.svg)](https://github.com/kmkamilkhel/topocorr/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Is a mountain trend real, or is it the sun?** topocorr tests Landsat and Sentinel-2 time series
for illumination artefacts on slopes and corrects them consistently through time, in Google Earth
Engine.

On steep terrain the reflectance of a pixel depends on how directly the sun lights it. That
illumination changes between acquisitions with the season, with orbit drift (Landsat 5 and 7) and
with each sensor switch, so a vegetation index can show greening, browning or a break that comes
from geometry, not from the vegetation. topocorr computes the illumination history of every pixel
from the DEM and each image's sun angles. It will:

- flag cast shadow, self-shadow, low illumination and very steep slopes;
- apply the standard corrections (cosine, C, SCS+C);
- estimate how much of an observed trend or break the illumination history alone predicts,
  with a correction fitted once for the whole stack.

Sensors: Landsat 5, 7, 8 and 9 Collection 2 Level 2 and Sentinel-2 L2A. Default DEM: Copernicus GLO-30.

## Status

In development; no release yet. The first release (0.1) will provide `collection`, `masks`,
`correct`, `fit`, `diagnose` and `compare`.

## Installation

From GitHub until the first release:

```bash
pip install git+https://github.com/kmkamilkhel/topocorr
```

topocorr needs a Google Earth Engine account and a Cloud project registered for Earth Engine
(`earthengine authenticate`).

## License

MIT. See [LICENSE](LICENSE).
