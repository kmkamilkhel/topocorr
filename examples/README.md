# Example notebooks

| Notebook | Site | Shows |
|---|---|---|
| [kalam.ipynb](kalam.ipynb) | Kalam, upper Swat (Hindu Kush), 35.5° N | Landsat 5 orbit drift on east- and west-facing slopes; stack models; the NIRv drift map (`diagnose`); the method comparison (`compare_export`, `compare`) |
| [tapanti.ipynb](tapanti.ipynb) | Tapantí, Costa Rica, 9.8° N | The sun on both sides of the tile near the equator: `model.sun_both_sides`, the fallback of `correct` to per-image SCS+C, and the `diagnose` warning |

Each notebook lists its Earth Engine use (EECU-hours, measured on these tiles) in its first cell.
Heavy steps are batch exports to Google Drive, started once and recorded in `tasks.json`; a
re-run skips every finished step, so an interrupted notebook can be run again from the top.
Progress bars show the export tasks and downloads.

## Setup

```bash
conda env create -f examples/environment.yml
conda activate topocorr-examples
earthengine authenticate
```

Set `PROJECT` in the first code cell to your Earth Engine Cloud project. Exports land in the
Google Drive folder `DRIVE_FOLDER`. Download them into the notebook's output folder by hand, or
set `RCLONE_REMOTE` to an rclone remote of that folder (for example `"gdrive:topocorr_exports/"`).

Figures are made with [eofigs](https://kmkamilkhel.github.io/eofigs/) and saved as PNG and PDF in
the output folder.
