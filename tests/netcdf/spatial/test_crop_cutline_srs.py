"""Regression test for issue #629.

Cropping a NetCDF variable by a polygon cutline used to log a GDAL warning —
``the input vector layer has a SRS, but the source raster dataset does not``
— because the variable's backing raster (the ``AsClassicDataset`` view, or a
``wrap_longitude`` MEM raster) carried no projection string even though the
variable tracks its EPSG. ``Selection._crop_one`` now stamps the known CRS onto
the raster before the affine cutline warp, so the warning is gone and a cutline
in a different CRS would be correctly reprojected rather than mis-clipped.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

FIXTURE = "tests/data/netcdf/cf__7v__1d3-2d3-3d1__y-asc.nc"
GDAL_LOGGER = "pyramids.base.config.gdal"
# Gulf of Mexico + Caribbean, in -180..180 longitudes (matches the wrapped grid).
REGION = [
    (-98, 18), (-93, 30), (-80, 31), (-75, 27), (-63, 22),
    (-60, 16), (-65, 10), (-75, 9), (-84, 8), (-92, 11), (-98, 18),
]


class _Capture(logging.Handler):
    """Collect emitted log messages for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestVariableCutlineCropSRS:
    """`crop` on a NetCDF variable stamps its CRS before the cutline warp (issue #629)."""

    def _crop(self):
        """Wrap ``tos`` to -180..180 and crop it by the Gulf/Caribbean polygon."""
        nc = NetCDF.read_file(FIXTURE)
        tos = nc.get_variable("tos").wrap_longitude()
        aoi = FeatureCollection(gpd.GeoDataFrame(geometry=[Polygon(REGION)], crs=4326))
        return tos.crop(aoi)

    def test_no_missing_raster_srs_warning(self):
        """The cutline crop logs no "source raster dataset does not [have a SRS]" warning.

        Test scenario:
            Capture the GDAL logger across the wrapped-variable cutline crop and
            assert nothing mentions the missing-raster-SRS cutline warning.
        """
        handler = _Capture()
        gdal_logger = logging.getLogger(GDAL_LOGGER)
        gdal_logger.addHandler(handler)
        try:
            self._crop()
        finally:
            gdal_logger.removeHandler(handler)
        offending = [
            m for m in handler.messages if "SRS" in m or "cutline" in m.lower()
        ]
        assert not offending, f"cutline SRS warning logged: {offending}"

    def test_crop_result_is_correct(self):
        """The crop clips to the polygon's extent on the wrapped (-180..180) grid.

        Test scenario:
            The Gulf/Caribbean polygon spans lon -98..-60 and lat 8..31; the
            cropped variable's bounds match that window and it stays EPSG:4326.
        """
        result = self._crop()
        bounds = [round(b, 1) for b in result.total_bounds]
        assert result.epsg == 4326, f"expected EPSG:4326, got {result.epsg}"
        assert bounds == [-98.0, 8.0, -60.0, 30.0], f"unexpected crop bounds: {bounds}"

    def test_result_raster_carries_projection(self):
        """The cropped variable's raster carries a non-empty projection.

        Test scenario:
            After the fix stamps the source CRS, the warp output is a normal
            georeferenced raster — its GDAL projection string is populated rather
            than empty.
        """
        result = self._crop()
        assert result._raster.GetProjection(), "result raster has no projection set"
