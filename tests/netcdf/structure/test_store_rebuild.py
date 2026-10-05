"""Tests for the store-level container rebuild behind `rename_dims` / `drop_dims`.

Covers what `NetCDF._rebuilt_container` and `_DimensionRemap` must keep faithful: sub-groups and
their variables, an ancestor group's coordinate arrays (and therefore the geotransform), group
attributes, and the dimensions the result declares.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal, osr

from pyramids.netcdf.netcdf import Container, NetCDF, _DimensionRemap

pytestmark = pytest.mark.core

CF_FIXTURE = "tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
GROUPS_FIXTURE = "tests/data/netcdf/none__35v__1d35__groups-nc4.nc"


def _wgs84():
    """A WGS84 spatial reference to stamp the synthetic stores' data arrays with."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs


def _nested_container():
    """A container with a root `t2m(time,y,x)` and a sub-group `diagnostics/flag(time,y,x)`.

    Built against GDAL's multidim API because `from_array` cannot create sub-groups, and the
    rebuild has to recreate them rather than silently return the working group alone.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    rg = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_time = rg.CreateDimension("time", gdal.DIM_TYPE_TEMPORAL, "", 2)
    dim_y = rg.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", 2)
    dim_x = rg.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", 2)
    for dim, values in (
        (dim_time, [0.0, 6.0]),
        (dim_y, [1.0, 0.0]),
        (dim_x, [0.0, 1.0]),
    ):
        coord = rg.CreateMDArray(dim.GetName(), [dim], f64)
        coord.Write(np.array(values))
        dim.SetIndexingVariable(coord)
    srs = _wgs84()
    t2m = rg.CreateMDArray("t2m", [dim_time, dim_y, dim_x], f64)
    t2m.Write(np.arange(8.0).reshape(2, 2, 2))
    t2m.SetSpatialRef(srs)
    sub = rg.CreateGroup("diagnostics")
    note = sub.CreateAttribute("note", [], gdal.ExtendedDataType.CreateString())
    note.WriteString("quality flags")
    flag = sub.CreateMDArray("flag", [dim_time, dim_y, dim_x], f64)
    flag.Write(np.ones((2, 2, 2)))
    flag.SetSpatialRef(srs)
    return Container(store)


def _ancestor_coordinate_container():
    """A store whose root holds the spatial axes and whose sub-group `g` holds the data.

    The root declares `y`, `x` and `nrows` with coordinate arrays; `g` declares `recNum` and
    holds `obs(recNum,y,x)` plus `other(nrows,y,x)`. A `get_group("g")` rebuild therefore has to
    carry coordinate arrays that live in an *ancestor* group, or the geotransform and the
    untouched `nrows` stamps are lost.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    rg = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_y = rg.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", 2)
    dim_x = rg.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", 2)
    dim_rows = rg.CreateDimension("nrows", "", "", 3)
    for dim, values in (
        (dim_y, [1.0, 0.0]),
        (dim_x, [0.0, 1.0]),
        (dim_rows, [10.0, 20.0, 30.0]),
    ):
        coord = rg.CreateMDArray(dim.GetName(), [dim], f64)
        coord.Write(np.array(values))
        dim.SetIndexingVariable(coord)
    group = rg.CreateGroup("g")
    dim_rec = group.CreateDimension("recNum", "", "", 2)
    rec_coord = group.CreateMDArray("recNum", [dim_rec], f64)
    rec_coord.Write(np.array([0.0, 1.0]))
    dim_rec.SetIndexingVariable(rec_coord)
    srs = _wgs84()
    obs = group.CreateMDArray("obs", [dim_rec, dim_y, dim_x], f64)
    obs.Write(np.arange(8.0).reshape(2, 2, 2))
    obs.SetSpatialRef(srs)
    other = group.CreateMDArray("other", [dim_rows, dim_y, dim_x], f64)
    other.Write(np.arange(12.0).reshape(3, 2, 2))
    other.SetSpatialRef(srs)
    return Container(store)


class TestRebuildGuards:
    """The rebuild refuses the inputs it cannot honour, and owns no reference to its caller."""

    def test_a_rename_and_a_drop_in_one_call_are_refused(self):
        """`_rebuilt_container(rename=..., drop=...)` raises instead of applying both (N4).

        The docstring has always said the two are not combined; before this guard the body
        happily applied them together.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        with pytest.raises(ValueError, match="either rename= or drop=, not both"):
            nc._rebuilt_container(rename={"time": "tt"}, drop={"plev"})

    def test_a_dropped_dimension_is_never_created(self):
        """`_DimensionRemap` refuses to create a destination twin of a dropped dimension (N5).

        A dimension is born in `_created` and nowhere else, so an array reaching `axes` while
        spanning a dropped axis must fail loudly rather than resurrect the axis.
        """
        source = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        dropped = source.GetRootGroup().CreateDimension("time", "", "", 2)
        destination = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        remap = _DimensionRemap(
            destination.GetRootGroup(), {}, {"time"}, NetCDF._recreate_md_array
        )
        with pytest.raises(ValueError, match="being dropped"):
            remap.axes(
                source.GetRootGroup().CreateMDArray(
                    "t", [dropped], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
                )
            )
        assert destination.GetRootGroup().GetDimensions() == [], (
            "the dropped dimension must not reach the destination"
        )

    def test_the_array_copy_is_injected_not_imported(self):
        """`_DimensionRemap` carries coordinates through the callable it was handed (N3).

        The value object used to call `NetCDF._recreate_md_array` by name although it is defined
        before that class, which made it cyclically dependent on the class it serves.
        """
        calls = []

        def _record(dst_group, var_name, src_mdarray, dst_dims):
            calls.append(var_name)
            return NetCDF._recreate_md_array(dst_group, var_name, src_mdarray, dst_dims)

        source = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        src_rg = source.GetRootGroup()
        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = src_rg.CreateDimension("time", "", "", 2)
        coord = src_rg.CreateMDArray("time", [dim], f64)
        coord.Write(np.array([0.0, 1.0]))
        dim.SetIndexingVariable(coord)
        destination = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        remap = _DimensionRemap(
            destination.GetRootGroup(), {"time": "tt"}, set(), _record
        )
        remap.declare([dim])
        remap.carry_coordinates(src_rg, [dim], ["time"])
        assert calls == ["tt"], f"the injected copy must be the one used, got {calls}"
