# Contributing

Bug reports, ideas and pull requests are welcome. Please open an issue first for larger changes.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By taking part you agree
to uphold it.

## Development setup

With conda:

```bash
conda env create -f environment-dev.yml
conda activate topocorr-dev
pip install -e . --no-deps
pre-commit install
```

With pip:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

Tests marked `ee` call Google Earth Engine. They run when credentials are available
(`earthengine authenticate`) and the environment variable `EE_PROJECT` names a Cloud project
registered for Earth Engine; otherwise they are skipped. All other tests run offline on
synthetic terrain.

## Pull requests

- Work on a branch and open a pull request against `main`; fill in the pull request template.
- All checks (lint, citation, tests on Linux, Windows and macOS, package build) must pass before a
  pull request is merged.

## Guidelines

- Public functions have type hints and NumPy-style docstrings.
- Every formula has a NumPy reference implementation that the tests compare the Earth Engine version
  against on synthetic terrain with a known answer.
- Angles are in degrees at the interface; azimuth and aspect are measured clockwise from north.
- Commit messages are short, plain English and in the imperative ("Add the self-shadow mask").
