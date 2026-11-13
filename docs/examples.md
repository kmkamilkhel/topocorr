# Example notebooks

Two notebooks in the repository's [`examples/`](https://github.com/kmkamilkhel/topocorr/tree/main/examples)
folder run the full workflow on real sites. Each lists its Earth Engine use per step in its first
cell; heavy steps are batch exports started once and recorded, so an interrupted notebook can be run
again from the top without repeating any finished step. Progress bars show the export tasks and
downloads.

| Notebook | Site | Shows |
|---|---|---|
| [kalam.ipynb](https://github.com/kmkamilkhel/topocorr/blob/main/examples/kalam.ipynb) | Kalam, upper Swat (Hindu Kush), 35.5° N | Landsat 5 orbit drift on east- and west-facing slopes; stack models per sensor family; the NIRv drift map (`diagnose`); the method comparison (`compare_export`, `compare`) |
| [tapanti.ipynb](https://github.com/kmkamilkhel/topocorr/blob/main/examples/tapanti.ipynb) | Tapantí, Costa Rica, 9.8° N | The sun on both sides of the tile near the equator: `model.sun_both_sides`, the fallback of `correct` to per-image SCS+C, and the `diagnose` warning |

## Setup

```bash
conda env create -f examples/environment.yml
conda activate topocorr-examples
earthengine authenticate
```

Set `PROJECT` in the first code cell to your Earth Engine Cloud project. Exports land in the Google
Drive folder `DRIVE_FOLDER`; download them into the notebook's output folder, or set
`RCLONE_REMOTE` to an rclone remote of that folder.
