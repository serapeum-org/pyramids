"""Inherited Dataset/RasterBase API exercised through a NetCDF variable view.

NetCDF subclasses Dataset, so it inherits ~119 raster methods. Many only work on a single variable
(a `get_variable()` view), not on the multi-variable container. This module drives every inherited
property and every zero-argument inherited method against a real variable view to catch view-specific
breakage (e.g. #588 resample, #592 color_table/footprint), plus a few argument-taking methods.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

# Inherited @property members that must return a non-None value on a real variable view.
INHERITED_PROPERTIES_NONNULL = [
    "access",
    "band_color",
    "band_count",
    "band_names",
    "band_units",
    "bbox",
    "block_size",
    "bounds",
    "cell_size",
    "columns",
    "crs",
    "dtype",
    "epsg",
    "gcp_count",
    "gdal_dtype",
    "has_gcps",
    "has_rpcs",
    "is_cog",
    "numpy_dtype",
    "offset",
    "overview_count",
    "raster",
    "rows",
    "scale",
    "shape",
    "total_bounds",
    "transform",
]

# Inherited @property members that may legitimately return None (no GCPs/RPCs/colour table set,
# or an in-memory variable view that carries no driver metadata).
INHERITED_PROPERTIES_NULLABLE = [
    "color_table",
    "driver_type",
    "gcps",
    "gcp_projection",
    "rpcs",
]

# Inherited zero-argument methods that should run on a variable view without raising.
# (plot_histogram / plot_vector_field are exercised in the plot module; footprint has its own test below.)
# The overview ops (create_overviews / get_overview / read_overview_array) are intentionally excluded:
# they write an external `.ovr` next to the *source file*, so running them on the shared committed
# fixture pollutes tests/data/. They get a dedicated tmp_path test below.
INHERITED_NOARG_METHODS = [
    "aspect",
    "block_windows",
    "to_polygons",
    "wrap_longitude",
    "count_domain_cells",
    "extract",
    "focal_mean",
    "focal_std",
    "get_attribute_table",
    "get_block_arrangement",
    "get_cell_coords",
    "get_cell_points",
    "get_cell_polygons",
    "get_histogram",
    "get_mask",
    "get_tile",
    "hillshade",
    "iter_blocks",
    "mask_flags",
    "preview",
    "proximity",
    "read_masks",
    "slope",
    "stats",
    "to_bytes",
    "to_cog_bytes",
    "to_feature_collection",
    # to_image renders via cleopatra (viz extra); skip it on a no-viz install.
    pytest.param("to_image", marks=pytest.mark.plot),
    "to_xyz",
    "translate",
]


@pytest.fixture
def tos_view(sample):
    """A single-variable Dataset-like view (tos) on which inherited raster ops are valid."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
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


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
def test_inherited_footprint(tos_view):
    """footprint covers only the data cells of a NetCDF variable view, not the nodata fill (#592)."""
    arr = np.asarray(tos_view.read_array(band=0))
    data_cells = int((~np.isclose(arr, tos_view.no_data_value[0], rtol=1e-5)).sum())
    fp = tos_view.footprint(band=0)
    assert fp is not None, "footprint should return a GeoDataFrame"
    gt = tos_view.geotransform
    covered = round(fp.geometry.area.sum() / abs(gt[1] * gt[5]))
    assert covered == data_cells, (
        f"footprint should cover {data_cells} data cells, got {covered}"
    )


def test_recreate_overviews_requires_write(sample, tmp_path):
    """recreate_overviews regenerates existing overviews, so on a read-only dataset it raises.

    Test scenario:
        Build external overviews on a tmp copy, reopen it read-only, then `recreate_overviews` must
        raise `ReadOnlyError` (regenerating needs write access). Self-contained on a tmp copy: it no
        longer depends on another test having built overviews on the shared fixture first, and writes
        no `.ovr` into tests/data.
    """
    work = shutil.copy(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"), tmp_path / "tos.nc")

    builder = NetCDF.read_file(str(work))
    try:
        # build external overviews so they exist on the read-only reopen below
        builder.get_variable("tos").create_overviews()
    finally:
        builder.close()

    nc = NetCDF.read_file(str(work))  # fresh read-only reopen sees the overviews
    try:
        view = nc.get_variable("tos")
        with pytest.raises(ReadOnlyError):
            view.recreate_overviews()
    finally:
        nc.close()


def test_overview_ops_isolated_to_temp_copy(sample, tmp_path):
    """The overview ops run on a tmp_path copy so their external `.ovr` never touches tests/data.

    Test scenario:
        `create_overviews` on a read-only variable view writes an external `<source>.0.ovr` next to
        the *source file*. Copy the fixture into `tmp_path` first, so the sidecar lands there (and is
        auto-cleaned); assert the overview family runs, the sidecar is written beside the copy, and no
        new `.ovr` appears in the committed data dir.
    """
    src = sample("cf__7v__1d3-2d3-3d1__y-asc.nc")
    data_dir = Path(src).parent
    before = set(data_dir.glob("*.ovr"))

    work = shutil.copy(src, tmp_path / "tos.nc")
    nc = NetCDF.read_file(str(work))
    try:
        view = nc.get_variable("tos")
        view.create_overviews()
        assert (tmp_path / "tos.nc.0.ovr").exists(), (
            "external overview should land beside the tmp copy"
        )
        assert view.get_overview() is not None, (
            "get_overview should return a band after building"
        )
        assert view.read_overview_array() is not None, (
            "read_overview_array should return data"
        )
    finally:
        nc.close()

    assert set(data_dir.glob("*.ovr")) == before, (
        "overview ops must not write a sidecar into the committed tests/data tree"
    )


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
