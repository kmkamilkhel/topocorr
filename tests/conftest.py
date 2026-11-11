"""Earth Engine tests run only when EE_PROJECT is set and credentials work; otherwise skipped.

Credentials: the local Earth Engine login (``earthengine authenticate``), or application default
credentials (``GOOGLE_APPLICATION_CREDENTIALS``, as set by Workload Identity Federation in CI).
"""

import os

import pytest


def _earth_engine_unavailable() -> str | None:
    project = os.environ.get("EE_PROJECT")
    if not project:
        return "EE_PROJECT is not set"
    try:
        import ee

        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            import google.auth

            scopes = [
                "https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            credentials, _ = google.auth.default(scopes=scopes)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as error:  # any failure means the tests cannot run here
        return f"Earth Engine is not available: {error}"
    return None


def pytest_collection_modifyitems(config, items):
    ee_items = [item for item in items if item.get_closest_marker("ee")]
    if not ee_items:
        return
    reason = _earth_engine_unavailable()
    if reason:
        for item in ee_items:
            item.add_marker(pytest.mark.skip(reason=reason))
