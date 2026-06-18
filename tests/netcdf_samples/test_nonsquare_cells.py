"""Non-square-cell grids: geotransform and bounds must use the lon/lat cell sizes independently.

`cf__7v` (tos) is a rectilinear grid with a 2° longitude cell and a 1° latitude cell. A single
`cell_size` for both axes would stretch the latitude extent (e.g. to -250°); the geotransform and
bounds must reflect each axis's own spacing.
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

NONSQUARE = "cf__7v__1d3-2d3-3d1.nc"  # tos: 2° lon, 1° lat


def test_geotransform_uses_separate_x_y_cells(sample):
    """The geotransform reports 2° for the X pixel and 1° for the Y pixel (not a single value)."""
    nc = NetCDF.read_file(sample(NONSQUARE))
    try:
        gt = nc.get_variable("tos").geotransform
        assert abs(gt[1]) == pytest.approx(2.0), f"X pixel should be 2°, got {gt[1]}"
        assert abs(gt[5]) == pytest.approx(1.0), f"Y pixel should be 1°, got {gt[5]}"
    finally:
        nc.close()


def test_bounds_not_stretched_by_square_cell(sample):
    """Latitude bounds stay within the real -80..90 range, not the stretched -250 a square cell gave."""
    nc = NetCDF.read_file(sample(NONSQUARE))
    try:
        xmin, ymin, xmax, ymax = nc.get_variable("tos").total_bounds
        assert (round(ymin), round(ymax)) == (-80, 90), f"lat bounds should be -80..90, got {ymin}..{ymax}"
        assert (round(xmin), round(xmax)) == (0, 360), f"lon bounds should be 0..360, got {xmin}..{xmax}"
    finally:
        nc.close()
