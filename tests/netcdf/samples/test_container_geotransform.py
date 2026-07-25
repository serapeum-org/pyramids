"""Container geotransform: the derived X origin must be the WEST edge, even for a descending lon.

The root container derives its geotransform from the lon/lat coordinate arrays. The Y branch already
takes the north edge with ``max(lat[0], lat[-1])``; the X branch used a plain ``lon[0]``, which put the
origin at the EAST edge for a descending-longitude grid (ARC-32).
"""

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def _descending_lon_nc(path: str) -> None:
    """Write a tiny EPSG:4326 netCDF whose lon coordinate descends east->west (centres 29..21)."""
    src = gdal.GetDriverByName("MEM").Create("", 5, 4, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    # Pixel 0 spans [30, 28] -> centre 29; pixel 4 spans [22, 20] -> centre 21. lon = 29,27,25,23,21.
    src.SetGeoTransform((30.0, -2.0, 0.0, 10.0, 0.0, -1.0))
    src.GetRasterBand(1).WriteArray(np.arange(20, dtype=np.float32).reshape(4, 5))
    gdal.Translate(path, src, format="netCDF", creationOptions=["WRITE_BOTTOMUP=NO"])
    src = None  # noqa: F841 - release the GDAL handle before the file is reopened


def test_container_x_origin_is_west_edge_for_descending_lon(tmp_path):
    """The container geotransform x-origin sits at the west edge, not the east edge (ARC-32)."""
    path = str(tmp_path / "lon_desc.nc")
    _descending_lon_nc(path)
    nc = NetCDF.read_file(path)
    try:
        lon = np.asarray(nc.lon, dtype=float)
        assert lon[0] > lon[-1], "fixture lon must descend east->west"
        gt = nc.geotransform
        x_cell = abs(gt[1])
        west_edge = min(lon[0], lon[-1]) - x_cell / 2
        east_bug = lon[0] - x_cell / 2  # the pre-ARC-32 value (origin at the EAST edge)
        assert west_edge != pytest.approx(east_bug), "fixture must separate west/east edges"
        assert gt[0] == pytest.approx(west_edge, abs=1e-4), (
            f"x-origin should be the west edge {west_edge}, got {gt[0]} "
            f"(the pre-ARC-32 east-edge bug would give {east_bug})"
        )
    finally:
        nc.close()
