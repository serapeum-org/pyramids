"""Tests for the zero-copy lazy get_group view (ARC-12).

`get_group` returns a `Container` that shares the parent's open GDAL dataset and
reads the sub-group in place, instead of MEM-copying every array. These tests
cover zero-copy sharing, group-scoped `variable_names` / `meta_data`, correct
reads (parity with the group-qualified `get_variable` path), nested chaining,
`close()` reference-count safety, and the pickle round-trip carrying
`_group_path`.

Style: Google-style docstrings, <=120 char lines, no inline imports, single
return statement, descriptive assertion messages.
"""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal

from pyramids.netcdf.netcdf import Container, NetCDF

pytestmark = pytest.mark.core

DISK_GROUPS = "tests/data/netcdf/none__35v__1d35__groups-nc4.nc"


def _build_grouped_mem() -> Container:
    """Return an in-memory container: root elevation + forecast (nested surface) + analysis."""
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
    rg = src.GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, 8)
    dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, 5)
    rg.CreateMDArray("elevation", [dim_y, dim_x], dtype).Write(np.full((5, 8), 100.0))

    forecast = rg.CreateGroup("forecast")
    forecast.CreateMDArray("temperature", [dim_y, dim_x], dtype).Write(np.full((5, 8), 300.0))
    surface = forecast.CreateGroup("surface")
    surface.CreateMDArray("t2m", [dim_y, dim_x], dtype).Write(np.full((5, 8), 288.0))

    analysis = rg.CreateGroup("analysis")
    analysis.CreateMDArray("wind_speed", [dim_y, dim_x], dtype).Write(np.full((5, 8), 5.5))
    return Container(src)


class TestZeroCopyView:
    """The view shares the parent dataset rather than copying arrays."""

    def test_get_group_shares_parent_dataset(self):
        """The returned view's raster IS the parent's dataset (no MEM copy)."""
        nc = _build_grouped_mem()
        view = nc.get_group("forecast")
        assert view._raster is nc._raster, "group view must share the parent's gdal.Dataset"
        assert isinstance(view, Container), "group view must be a Container"
        assert view._group_path == "forecast", "view must record its sub-group path"

    def test_view_variable_names_are_group_scoped(self):
        """The view reports the sub-group's variables, not the root's or siblings'."""
        nc = _build_grouped_mem()
        view = nc.get_group("forecast")
        assert "temperature" in view.variable_names, "view must see its own variable"
        assert "elevation" not in view.variable_names, "view must not see the root variable"
        assert "wind_speed" not in view.variable_names, "view must not see a sibling group"

    def test_view_metadata_is_group_scoped(self):
        """`meta_data` traversal is scoped to the sub-group."""
        nc = _build_grouped_mem()
        view = nc.get_group("analysis")
        # Metadata keys variables by full path (e.g. "analysis/wind_speed"); compare leaf names.
        leaves = {name.split("/")[-1] for name in view.meta_data.variables}
        assert "wind_speed" in leaves, "group metadata must include the group's variable"
        assert "temperature" not in leaves, "group metadata must exclude other groups"
        assert "elevation" not in leaves, "group metadata must exclude the root variable"


class TestViewReads:
    """Reading through the view matches the group-qualified get_variable path."""

    def test_read_matches_group_qualified_path(self):
        """`view.get_variable(name)` reads the same data as `nc.get_variable('grp/name')`."""
        nc = _build_grouped_mem()
        view = nc.get_group("forecast")
        via_view = np.asarray(view.get_variable("temperature").read_array(band=0))
        via_root = np.asarray(nc.get_variable("forecast/temperature").read_array(band=0))
        assert_allclose(via_view, via_root, err_msg="view read must match the qualified path")
        assert_allclose(
            via_view, np.full((5, 8), 300.0), err_msg="view must read the stored values"
        )


class TestNestedChaining:
    """get_group composes paths so it can be chained."""

    def test_chained_equals_slashed_path(self):
        """`get_group('a').get_group('b')` matches `get_group('a/b')`."""
        nc = _build_grouped_mem()
        chained = nc.get_group("forecast").get_group("surface")
        slashed = nc.get_group("forecast/surface")
        assert chained._group_path == "forecast/surface", "chained path must compose"
        assert slashed._group_path == "forecast/surface", "slashed path must match"
        assert "t2m" in chained.variable_names, "nested group's variable must be visible"
        assert_allclose(
            np.asarray(chained.get_variable("t2m").read_array(band=0)),
            np.full((5, 8), 288.0),
            err_msg="nested-group read must return the stored values",
        )

    def test_invalid_group_raises(self):
        """An unknown group name raises ValueError."""
        nc = _build_grouped_mem()
        with pytest.raises(ValueError, match="not found"):
            nc.get_group("nonexistent")


class TestCloseRefcount:
    """Sharing the dataset must not let one side close it out from under the other."""

    def test_closing_view_leaves_parent_usable(self):
        """Closing the view only drops its reference; the parent keeps working."""
        nc = _build_grouped_mem()
        view = nc.get_group("forecast")
        view.close()
        assert "forecast" in nc.group_names, "parent must survive the view's close()"
        assert "elevation" in nc.variable_names, "parent reads must still work"

    def test_view_survives_parent_close(self):
        """The view holds its own dataset reference, so it outlives the parent's close()."""
        nc = _build_grouped_mem()
        view = nc.get_group("forecast")
        nc.close()
        assert "temperature" in view.variable_names, "view must still read after parent close()"


class TestPickleRoundTrip:
    """A group view round-trips back to the same sub-group (on-disk source)."""

    def test_group_view_pickle_preserves_group(self):
        """Unpickling a group view yields a view of the same group with the same variables."""
        nc = NetCDF.read_file(DISK_GROUPS)
        group_name = nc.group_names[0]
        view = nc.get_group(group_name)
        restored = pickle.loads(pickle.dumps(view))
        assert restored._group_path == group_name, "pickle must preserve the group path"
        assert sorted(restored.variable_names) == sorted(
            view.variable_names
        ), "restored view must expose the same group variables"
