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

# Inherited @property members that must return a non-None value on a real variable view.
INHERITED_PROPERTIES_NONNULL = [
    "access", "band_color", "band_count", "band_names", "band_units", "bbox", "block_size",
    "bounds", "cell_size", "columns", "crs", "dtype", "epsg",
    "gcp_count", "gdal_dtype", "has_gcps", "has_rpcs", "is_cog",
    "numpy_dtype", "offset", "overview_count", "raster", "rows", "scale", "shape",
    "total_bounds", "transform",
]

# Inherited @property members that may legitimately return None (no GCPs/RPCs/colour table set,
# or an in-memory variable view that carries no driver metadata).
INHERITED_PROPERTIES_NULLABLE = [
    "color_table", "driver_type", "gcps", "gcp_projection", "rpcs",
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
    # to_image renders via cleopatra (viz extra); skip it on a no-viz install.
    pytest.param("to_image", marks=pytest.mark.plot), "to_xyz", "translate",
]


@pytest.fixture
def tos_view(sample):
    """A single-variable Dataset-like view (tos) on which inherited raster ops are valid."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    try:
        yield nc.get_variable("tos")
    finally:
        nc.close()


@pytest.mark.parametrize("prop", INHERITED_PROPERTIES_NONNULL)
def test_inherited_property_accessible(tos_view, prop):
    """Every non-nullable inherited property returns a non-None value on a NetCDF variable view."""
    assert getattr(tos_view, prop) is not None


@pytest.mark.parametrize("prop", INHERITED_PROPERTIES_NULLABLE)
def test_inherited_nullable_property_accessible(tos_view, prop):
    """Nullable inherited properties are readable without raising (None is an acceptable return)."""
    getattr(tos_view, prop)  # must not raise; None is a legitimate value


@pytest.mark.parametrize("method", INHERITED_NOARG_METHODS)
def test_inherited_noarg_method_runs(tos_view, method):
    """Every inherited zero-argument method runs on a NetCDF variable view without raising."""
    getattr(tos_view, method)()


def test_inherited_footprint(tos_view):
    """footprint produces a coverage polygon for a NetCDF variable view (#592)."""
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
