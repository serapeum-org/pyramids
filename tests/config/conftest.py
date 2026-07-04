"""Fixtures for the pyramids.configure tests.

``pyramids.configure`` / ``configure_lazy_vector`` set process-wide GDAL config
options via :func:`gdal.SetConfigOption` (e.g. ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``,
which stops GDAL from discovering sidecar files like ``.ovr`` / ``.msk`` / ``.aux.xml``).
Left in place, those settings leak into every suite collected after ``tests/config/`` and
silently break sidecar-dependent tests (overviews, mask bands, attribute tables). The autouse
fixture below snapshots the whole GDAL config before each config test and restores it after,
so the tests stay isolated regardless of collection order or which keys they set.
"""

import pytest
from osgeo import gdal


@pytest.fixture(autouse=True)
def _restore_gdal_config():
    """Snapshot and restore all GDAL config options around each config test."""
    before = gdal.GetConfigOptions()
    yield
    after = gdal.GetConfigOptions()
    for key in after:
        if key not in before:
            gdal.SetConfigOption(key, None)
    for key, value in before.items():
        if after.get(key) != value:
            gdal.SetConfigOption(key, value)
