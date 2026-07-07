"""Regression test for issue #628.

``NetCDF.get_variable`` on a gridded variable whose spatial plane is not the
trailing two dimensions (e.g. ``T(time, lat, lev, lon)``) used to log a spurious
GDAL ``CE_Failure`` — ``arrayStartIdx[1] + (count[1]-1) * arrayStep[1] >= <dim>``
— while wrapping the index-space ``AsClassicDataset`` view in a georeferencing
VRT. The array and geotransform were already correct; only the ERROR log was
misleading. ``_georeference_index_subset`` now silences that one known-harmless
error, so this asserts the log is gone while the result is unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

FIXTURE = "tests/data/netcdf/cf__48v__1d17-3d21-4d10__y-asc.nc"
GDAL_LOGGER = "pyramids.base.config.gdal"


class _Capture(logging.Handler):
    """Collect emitted log messages for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestGetVariableNonTrailingPlaneGeoref:
    """`get_variable` on a non-trailing-plane gridded variable (issue #628)."""

    def _read(self):
        """Open the fixture and extract ``T`` with explicit lon/lat dims."""
        nc = NetCDF.read_file(FIXTURE)
        return nc.get_variable("T", x_dim="lon", y_dim="lat")

    def test_no_spurious_gdal_bounds_error_logged(self):
        """`get_variable` logs no GDAL `arrayStartIdx >= dim` CE_Failure.

        Test scenario:
            Attach a handler to the GDAL logger, extract the non-trailing-plane
            variable, and assert no captured message mentions the partial-read
            bounds failure that the georeferencing VRT build used to emit.
        """
        handler = _Capture()
        gdal_logger = logging.getLogger(GDAL_LOGGER)
        gdal_logger.addHandler(handler)
        try:
            var = self._read()
            var.read_array()
        finally:
            gdal_logger.removeHandler(handler)
        offending = [m for m in handler.messages if "arrayStartIdx" in m]
        assert not offending, f"spurious GDAL bounds error logged: {offending}"

    def test_array_is_correct(self):
        """The extracted array keeps its `(bands, lat, lon)` shape and values.

        Test scenario:
            The variable folds its two non-spatial axes (time × lev = 6) onto the
            band axis over the 64×128 lat/lon grid; the values stay finite and in
            the physically plausible temperature range seen in the fixture.
        """
        data = self._read().read_array()
        assert data.shape == (6, 64, 128), f"unexpected shape {data.shape}"
        assert np.isfinite(data).all(), "array contains non-finite values"
        assert 150.0 < float(np.nanmin(data)) < float(np.nanmax(data)) < 350.0, (
            f"values outside plausible range: {float(np.nanmin(data))}..{float(np.nanmax(data))}"
        )

    def test_georeferencing_is_real_degrees_not_index_space(self):
        """The subset adopts the file's real lon/lat geotransform, not index space.

        Test scenario:
            An index-space view would carry cell size 1 and origin 0; the
            coordinate-derived correction instead yields the real degree spacing
            (~2.8125° lon) and EPSG:4326, proving the VRT georeferencing still
            applies after the error is silenced.
        """
        var = self._read()
        gt = var.geotransform
        assert var.epsg == 4326, f"expected EPSG:4326, got {var.epsg}"
        assert gt[1] == pytest.approx(2.8125), f"unexpected lon cell size: {gt[1]}"
        assert gt[5] < 0, f"geotransform should be north-up (negative dy), got {gt[5]}"
        assert gt != (0.0, 1.0, 0.0, 0.0, 0.0, -1.0), "geotransform is still index-space"
