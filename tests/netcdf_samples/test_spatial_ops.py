"""Spatial operations: crop, to_crs, reproject_variable, warped_view, resample."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

TOS = "cf__7v__1d3-2d3-3d1.nc"  # tos(time, lat, lon), EPSG:4326


def test_crop_bbox_reduces_extent(sample):
    """``crop(bbox=...)`` shrinks the spatial extent of the variables."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        full_cols = nc.get_variable("tos").shape[-1]
        cropped = nc.crop(bbox=(0, -40, 60, 40))
        assert cropped.get_variable("tos").shape[-1] < full_cols
    finally:
        nc.close()


def test_to_crs_reprojects_variables(sample):
    """``to_crs`` reprojects the container's variables to the target CRS."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        assert nc.get_variable("tos").epsg == 4326
        reprojected = nc.to_crs(3857)
        assert reprojected.get_variable("tos").epsg == 3857
    finally:
        nc.close()


def test_reproject_variable_updates_crs(sample):
    """``reproject_variable`` reprojects one variable in place on a file-backed container (issue #587)."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        result = nc.reproject_variable("tos", 3857)
        assert result.get_variable("tos").epsg == 3857
    finally:
        nc.close()


def test_warped_view_changes_crs(sample):
    """``warped_view`` yields a reprojected view of the container."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        view = nc.get_variable("tos").warped_view(3857)
        assert isinstance(view, NetCDF)
        assert view.epsg == 3857
    finally:
        nc.close()


def test_resample_changes_cell_size(sample):
    """``resample`` changes the cell size of a variable view (regression for issue #588)."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        original = nc.cell_size
        resampled = nc.get_variable("tos").resample(original * 2)
        assert resampled.cell_size == pytest.approx(original * 2)
    finally:
        nc.close()
