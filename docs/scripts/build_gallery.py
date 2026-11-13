"""Build the gallery figures of the documentation (eofigs 0.2.0).

    python docs/scripts/build_gallery.py KALAM_L5_TIF GHANDRUK_L5_TIF

The two GeoTIFFs are `diagnose` exports (NDVI, Landsat 5) made with the development version before
0.1, on each tile's Landsat scene grid; their drift bands are named `artefact_trend` and
`artefact_share` there (`drift_trend` and `drift_share` in 0.1). The sun angles of the Kalam
Landsat 5 scenes are in kalam_l5_sun_angles.csv next to this script (scene metadata).
Figures are written to docs/gallery/images/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import eofigs
import numpy as np
import rioxarray  # noqa: F401  (registers the rasterio engine)
import xarray as xr

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "gallery" / "images"
RISK = {0: "not assessable", 1: "low", 2: "medium", 3: "high", 4: "no trend"}


def cos_i(zenith, azimuth, slope, aspect):
    z, a, s, e = (np.radians(v) for v in (zenith, azimuth, slope, aspect))
    return np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(a - e)


def illumination_figure():
    with (HERE / "kalam_l5_sun_angles.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    years = 1970 + np.array([float(r["time_ms"]) for r in rows]) / (365.25 * 86400e3)
    zen = np.array([float(r["zenith"]) for r in rows])
    azi = np.array([float(r["azimuth"]) for r in rows])
    summer = np.isin(np.floor((years % 1) * 12).astype(int) + 1, (6, 7, 8))
    year = np.floor(years[summer]).astype(int)
    fig, ax = eofigs.new_figure(width="single", height_mm=60)
    for aspect, style, side in ((90.0, "-", "east"), (270.0, "--", "west")):
        ci = cos_i(zen[summer], azi[summer], 30.0, aspect)
        ys = np.unique(year)
        ax.plot(ys, [np.median(ci[year == y]) for y in ys], style, marker="o", ms=2.5,
                label=f"{side}-facing 30° slope")  # fmt: skip
    ax.set_xlabel("Year")
    ax.set_ylabel("Median cos i, June-August")
    ax.legend(fontsize=6, frameon=False)
    eofigs.save_figure(fig, IMAGES / "kalam_illumination_by_year", formats=["png"], dpi=300)
    return {"images": int(summer.sum()), "years": [int(ys.min()), int(ys.max())]}


def drift_map(tif, name, label):
    data = xr.open_dataarray(tif, engine="rasterio")
    bands = list(data.attrs["long_name"])

    def band(b):
        return data.sel(band=bands.index(b) + 1).drop_vars("band")

    fig = eofigs.trend_map(slope=band("artefact_trend") * 10, variable="ndvi", exclude=None,
                           aoi_label=label, width="single")  # fmt: skip
    eofigs.save_figure(fig, IMAGES / f"{name}_drift_trend", formats=["png"], dpi=200)
    risk = band("risk_class").values
    values, counts = np.unique(risk[np.isfinite(risk)], return_counts=True)
    shares = {
        RISK[int(v)]: round(100 * n / counts.sum(), 1) for v, n in zip(values, counts, strict=True)
    }
    return {"caption": fig.eofigs_layout["caption"], "risk_shares": shares}


def main():
    eofigs.set_style(journal="custom", font="Arial", bold=True, labels="A")
    IMAGES.mkdir(parents=True, exist_ok=True)
    info = {"illumination": illumination_figure()}
    info["kalam"] = drift_map(sys.argv[1], "kalam_l5", "Kalam tile (24 km)")
    info["ghandruk"] = drift_map(sys.argv[2], "ghandruk_l5", "Ghandruk tile (24 km)")
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
