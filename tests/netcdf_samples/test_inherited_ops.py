"""Inherited Dataset/RasterBase API exercised through a NetCDF variable view.

NetCDF subclasses Dataset, so it inherits ~119 raster methods. Many only work on a single variable
(a `get_variable()` view), not on the multi-variable container. This module drives every inherited
property and every zero-argument inherited method against a real variable view to catch view-specific
breakage (e.g. #588 resample, #592 color_table/footprint), plus a few argument-taking methods.
"""

import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

# Inherited @property members (from Dataset / AbstractDataset / RasterBase).
INHERITED_PROPERTIES = [
    "access", "band_color", "band_count", "band_names", "band_units", "bbox", "block_size",
    "bounds", "cell_size", "color_table", "columns", "crs", "driver_type", "dtype", "epsg",
    "gcp_count", "gcp_projection", "gcps", "gdal_dtype", "has_gcps", "has_rpcs", "is_cog",
    "numpy_dtype", "offset", "overview_count", "raster", "rows", "rpcs", "scale", "shape",
    "total_bounds", "transform",
]

# Inherited zero-argument methods that should run on a variable view without raising.
# (plot_histogram / plot_vector_field are exercised in the plot module; footprint is xfailed below.)
INHERITED_NOARG_METHODS = [
    "aspect", "block_windows", "cluster2", "wrap_longitude", "count_domain_cells",
    "create_overviews", "extract", "focal_mean", "focal_std", "get_attribute_table",
    "get_block_arrangement", "get_cell_coords", "get_cell_points", "get_cell_polygons",
    "get_histogram", "get_mask", "get_overview", "get_tile", "hillshade", "iter_blocks",
    "mask_flags", "preview", "proximity", "read_masks", "read_overview_array",
    "slope", "stats", "to_bytes", "to_cog_bytes", "to_feature_collection",
    "to_image", "to_xyz", "translate",
]


@pytest.fixture
def tos_view(sample):
    """A single-variable Dataset-like view (tos) on which inherited raster ops are valid."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    try:
        yield nc.get_variable("tos")
    finally:
        nc.close()


@pytest.mark.parametrize("prop", INHERITED_PROPERTIES)
def test_inherited_property_accessible(tos_view, prop):
    """Every inherited property is readable on a NetCDF variable view without raising."""
    getattr(tos_view, prop)


@pytest.mark.parametrize("method", INHERITED_NOARG_METHODS)
def test_inherited_noarg_method_runs(tos_view, method):
    """Every inherited zero-argument method runs on a NetCDF variable view without raising."""
    getattr(tos_view, method)()


@pytest.mark.xfail(reason="footprint's internal mask yields a None band on NetCDF views (#592)", strict=False)
def test_inherited_footprint(tos_view):
    """footprint should produce a coverage polygon for the variable view (currently fails, #592)."""
    assert tos_view.footprint() is not None


def test_recreate_overviews_requires_write(tos_view):
    """recreate_overviews regenerates overviews and so requires a writable dataset."""
    with pytest.raises(ReadOnlyError):
        tos_view.recreate_overviews()


def test_inherited_to_cog_writes_file(tos_view, tmp_path):
    """The inherited COG writer produces an on-disk Cloud-Optimized GeoTIFF from a variable view."""
    out = tmp_path / "tos.tif"
    tos_view.to_cog(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_inherited_to_raster_writes_file(tos_view, tmp_path):
    """The inherited GeoTIFF writer produces an on-disk raster from a variable view."""
    out = tmp_path / "tos_raster.tif"
    tos_view.to_raster(str(out))
    assert out.exists() and out.stat().st_size > 0
