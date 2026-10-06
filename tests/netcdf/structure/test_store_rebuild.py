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

from pyramids.netcdf.netcdf import (
    _MAX_GROUP_DEPTH,
    Container,
    NetCDF,
    _DimensionRemap,
)

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


def _group_attributes(nc, group_name):
    """The attributes of one sub-group of `nc`'s store, as a `{name: value}` dict of strings."""
    group = nc._raster.GetRootGroup().OpenGroup(group_name)
    return {
        attr.GetName(): attr.ReadAsString()
        for attr in group.GetAttributes()
        if attr.GetDataType().GetClass() == gdal.GEDTC_STRING
    }


def _variables_off(nc, dim_name):
    """The names of `nc`'s variables that do **not** span a dimension called `dim_name`."""
    survivors = set()
    for name in nc.variable_names:
        parts = name.split("/")
        holder = nc._raster.GetRootGroup()
        for part in parts[:-1]:
            holder = holder.OpenGroup(part)
        axes = {dim.GetName() for dim in holder.OpenMDArray(parts[-1]).GetDimensions()}
        if dim_name not in axes:
            survivors.add(name)
    return survivors


class TestRebuildKeepsSubGroups:
    """A hierarchical container keeps every sub-group and every variable in it (C1)."""

    def test_a_hierarchical_container_rename_keeps_every_variable(self):
        """`rename_dims` on the grouped fixture keeps all 29 variables and all 7 sub-groups.

        The rebuild read the working group's arrays and nothing else, so this file came back
        holding one variable and no sub-groups at all — silent, irreversible loss on a
        documented operation.
        """
        nc = NetCDF.read_file(GROUPS_FIXTURE)
        before_variables = sorted(nc.variable_names)
        before_groups = sorted(nc.group_names)
        assert len(before_variables) == 29, (
            f"precondition: 29 variables, got {len(before_variables)}"
        )
        out = nc.rename_dims({"recNum": "rec"})
        assert sorted(out.variable_names) == before_variables, (
            f"{len(out.variable_names)} of {len(before_variables)} variables survived"
        )
        assert sorted(out.group_names) == before_groups, (
            f"sub-groups lost: {sorted(out.group_names)} != {before_groups}"
        )
        assert "rec" in (out.dimension_names or []), f"got {out.dimension_names}"
        assert "recNum" not in (out.dimension_names or []), "the old dim must be gone"

    def test_a_nested_cube_keeps_its_sub_group_variable(self):
        """A root `t2m` plus a `diagnostics/flag` come back as both, with cells intact."""
        nc = _nested_container()
        assert sorted(nc.variable_names) == ["diagnostics/flag", "t2m"], (
            f"precondition: {sorted(nc.variable_names)}"
        )
        out = nc.rename_dims(time="tt")
        assert sorted(out.variable_names) == ["diagnostics/flag", "t2m"], (
            f"the sub-group variable must survive, got {sorted(out.variable_names)}"
        )
        carried = (
            out._raster.GetRootGroup().OpenGroup("diagnostics").OpenMDArray("flag")
        )
        assert_allclose(carried.ReadAsArray(), np.ones((2, 2, 2)))
        axes = [dim.GetName() for dim in carried.GetDimensions()]
        assert axes[0] == "tt", (
            f"the sub-group array must span the renamed axis, got {axes}"
        )

    def test_a_sub_group_keeps_its_own_attributes(self):
        """A sub-group's attributes are copied along with the group (H2, below the root)."""
        nc = _nested_container()
        before = _group_attributes(nc, "diagnostics")
        assert before == {"note": "quality flags"}, f"precondition: {before}"
        out = nc.rename_dims(time="tt")
        assert _group_attributes(out, "diagnostics") == before, (
            f"sub-group attributes lost: {_group_attributes(out, 'diagnostics')}"
        )


class TestDropDimsTakesTheRebuild:
    """`drop_dims` has one exit, so neither shape escapes the rebuild (H3, M3)."""

    def test_the_no_survivor_case_declares_no_dropped_dimension(self):
        """A container whose only variable spans the dropped dimension keeps nothing of it.

        The empty-survivor case fell back to `copy()` plus a `remove_variable` per variable,
        which is exactly the path that leaves the dimension declared: the result still reported
        `time` in `dimension_names` and still carried the `time` coordinate array into `to_file`.
        """
        nc = _nested_container()
        out = nc.drop_dims("time")
        assert out.variable_names == [], (
            f"both variables span time, got {out.variable_names}"
        )
        assert "time" not in (out.dimension_names or []), (
            f"the dropped dimension must be gone, got {out.dimension_names}"
        )
        arrays = out._raster.GetRootGroup().GetMDArrayNames() or []
        assert "time" not in arrays, f"its coordinate array too, got {sorted(arrays)}"

    def test_a_hierarchical_container_drop_works_and_keeps_the_survivors(self):
        """`drop_dims` on the grouped fixture drops only what spans the named dimension (M3).

        Both exits used to raise on this file shape: the survivor scan died with
        `AttributeError: 'LabeledArray' object has no attribute '_band_dim_names'`, and the
        fallback with `remove_variable() cannot act on '<group>/<var>'`.
        """
        nc = NetCDF.read_file(GROUPS_FIXTURE)
        before = set(nc.variable_names)
        expected = _variables_off(nc, "recNum")
        assert expected, "precondition: at least one variable must survive the drop"
        assert expected < before, (
            f"precondition: some of the {len(before)} variables must span recNum"
        )
        out = nc.drop_dims("recNum")
        assert set(out.variable_names) == expected, (
            "survivors must be exactly the variables off recNum; differing: "
            f"{sorted(set(out.variable_names) ^ expected)}"
        )
        assert "recNum" not in (out.dimension_names or []), (
            f"the dropped dimension must be gone, got {out.dimension_names}"
        )

    def test_a_hierarchical_container_drop_keeps_the_sub_groups(self):
        """A sub-group emptied by the drop is still a sub-group of the result."""
        out = _nested_container().drop_dims("time")
        assert out.group_names == ["diagnostics"], (
            f"the sub-group itself must remain, got {out.group_names}"
        )


class TestRebuildOfAGroupView:
    """A `get_group(...)` rebuild keeps its ancestors' coordinates and its own identity."""

    def test_a_group_view_rename_keeps_the_geotransform(self):
        """The ancestor group's `y`/`x` coordinate arrays are carried, so the grid is kept (H1).

        Without them the geotransform was silently replaced by a GDAL default — origin and
        extent of a 512-row raster — while `epsg` still read 4326, so the result looked sound.
        """
        view = _ancestor_coordinate_container().get_group("g")
        before = view.geotransform
        assert before == (-0.5, 1.0, 0, 1.5, 0, -1.0), f"precondition: {before}"
        out = view.rename_dims({"recNum": "rec"})
        assert out.geotransform == before, (
            f"geotransform must be identical, got {out.geotransform}"
        )
        assert out.epsg == view.epsg, f"epsg must be identical, got {out.epsg}"

    def test_a_group_view_rename_keeps_an_untouched_dimensions_stamps(self):
        """A band dimension the caller never named keeps its coordinate values (H1).

        `nrows` is declared in the ancestor group, so its coordinate array was dropped and the
        variable spanning it came back with `None` for its stamps.
        """
        view = _ancestor_coordinate_container().get_group("g")
        before = view.get_variable("other")._band_dim_values_map["nrows"]
        assert_allclose(np.ravel(before), [10.0, 20.0, 30.0])
        out = view.rename_dims({"recNum": "rec"})
        after = out.get_variable("other")._band_dim_values_map["nrows"]
        assert after is not None, "the untouched dimension lost its stamps"
        assert_allclose(np.ravel(after), np.ravel(before))

    def test_a_group_view_rename_returns_a_group_view(self):
        """The rebuild of a view is the equivalent view, not a promoted root container (L6)."""
        view = _ancestor_coordinate_container().get_group("g")
        out = view.rename_dims({"recNum": "rec"})
        assert out._group_path == "g", f"group identity lost, got {out._group_path!r}"
        assert out._raster.GetRootGroup().GetGroupNames() == ["g"], (
            "the rebuilt store must hold the group, not its contents at the root"
        )
        assert sorted(out.variable_names) == sorted(view.variable_names), (
            f"the view's inventory must be unchanged, got {sorted(out.variable_names)}"
        )


class TestRebuildDeclaresNoOrphanDimension:
    """The result declares only the dimensions its surviving arrays actually span (M1)."""

    def test_a_drop_leaves_no_dimension_without_a_referencing_array(self):
        """After `drop_dims('time')` on the CF fixture, `plev` is gone with its coordinate.

        Only `ua` spanned `plev`, and `ua` spans `time` too, so nothing left references the
        pressure axis — yet it used to be declared, and its coordinate array written, because
        every non-dropped source dimension was declared up front.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        assert "plev" in (nc.dimension_names or []), "precondition: plev is declared"
        out = nc.drop_dims("time")
        declared = set(out.dimension_names or [])
        assert "time" not in declared, (
            f"the dropped dimension must be gone, got {declared}"
        )
        assert "plev" not in declared, (
            f"plev is spanned by nothing that survived, got {sorted(declared)}"
        )
        arrays = set(out._raster.GetRootGroup().GetMDArrayNames() or [])
        assert "plev" not in arrays, (
            f"the orphan's coordinate array too, got {sorted(arrays)}"
        )
        spanned = {
            dim.GetName()
            for name in arrays
            for dim in out._raster.GetRootGroup().OpenMDArray(name).GetDimensions()
        }
        assert declared <= spanned, (
            f"every declared dimension must be spanned: {sorted(declared - spanned)}"
        )

    def test_a_rename_still_keeps_every_referenced_dimension(self):
        """On-demand creation must not lose an axis a survivor does use — the inventory holds."""
        nc = NetCDF.read_file(CF_FIXTURE)
        before = set(nc.dimension_names or [])
        out = nc.rename_dims(time="tt")
        after = set(out.dimension_names or [])
        assert after == (before - {"time"}) | {"tt"}, (
            f"only `time` may change name, got {sorted(after)} from {sorted(before)}"
        )


class TestRebuildKeepsGlobalAttributes:
    """A rename or a drop carries the group's attributes across (H2)."""

    def test_a_container_rename_keeps_every_global_attribute(self):
        """`rename_dims` keeps all 18 global attributes of the CF fixture, not none of them.

        `global_attributes` is a documented public member; before this the rebuild returned a
        store with zero attributes, so whether `Conventions` / `title` / `history` survived
        depended on whether the mapping happened to be a no-op.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        before = nc.global_attributes
        assert len(before) == 18, (
            f"precondition: 18 global attributes, got {len(before)}"
        )
        out = nc.rename_dims(time="tt")
        assert out.global_attributes == before, (
            f"global attributes must be carried, got {out.global_attributes}"
        )

    def test_a_container_drop_keeps_every_global_attribute(self):
        """`drop_dims` keeps the group's attributes too — it takes the same rebuild."""
        nc = NetCDF.read_file(CF_FIXTURE)
        before = nc.global_attributes
        out = nc.drop_dims("time")
        assert out.global_attributes == before, (
            f"global attributes must be carried, got {out.global_attributes}"
        )

    def test_the_global_attributes_reach_the_written_file(self, tmp_path):
        """The carried attributes survive `to_file` + reload, where the loss used to be baked in."""
        nc = NetCDF.read_file(CF_FIXTURE)
        before = nc.global_attributes
        path = tmp_path / "renamed.nc"
        nc.rename_dims(time="tt").to_file(str(path))
        reloaded = NetCDF.read_file(str(path)).global_attributes
        missing = {
            key: before[key] for key in before if reloaded.get(key) != before[key]
        }
        assert missing == {}, f"attributes lost or altered on write: {missing}"


def _string_attributes(nc, var_name):
    """The string-valued attributes of one root-level array of `nc`'s store."""
    array = nc._raster.GetRootGroup().OpenMDArray(var_name)
    return {
        attr.GetName(): attr.ReadAsString()
        for attr in array.GetAttributes()
        if attr.GetDataType().GetClass() == gdal.GEDTC_STRING
    }


class TestRebuildRewritesCfDimensionReferences:
    """A CF attribute naming a renamed dimension is rewritten, not copied verbatim (M4)."""

    def test_cell_methods_follows_the_rename(self):
        """`cell_methods = 'time: mean ...'` becomes `'tt: mean ...'` after `rename_dims`.

        Copied verbatim it named a dimension the result no longer has, so a CF-aware reader
        resolving it against the renamed file found nothing.
        """
        nc = NetCDF.read_file(CF_FIXTURE)
        before = _string_attributes(nc, "tas")
        assert before["cell_methods"] == "time: mean (interval: 1 month)", (
            f"precondition: {before.get('cell_methods')!r}"
        )
        out = nc.rename_dims(time="tt")
        after = _string_attributes(out, "tas")
        assert after["cell_methods"] == "tt: mean (interval: 1 month)", (
            f"cell_methods must follow the rename, got {after['cell_methods']!r}"
        )
        assert after["cell_method"] == "tt: mean", (
            f"cell_method must follow it too, got {after['cell_method']!r}"
        )
        assert "time" not in (out.dimension_names or []), (
            "precondition: the store has no `time` left"
        )

    def test_a_bounds_variable_name_is_not_rewritten(self):
        """`bounds = 'time_bnds'` is a variable name the rebuild keeps, so it stays as it is.

        Only whole-word matches are rewritten, which is what separates the dimension token in
        `cell_methods` from a variable whose name merely starts with it.
        """
        out = NetCDF.read_file(CF_FIXTURE).rename_dims(time="tt")
        root = out._raster.GetRootGroup()
        assert "time_bnds" in (root.GetMDArrayNames() or []), (
            "precondition: the bounds array is kept under its own name"
        )
        assert _string_attributes(out, "tt")["bounds"] == "time_bnds", (
            f"bounds must still resolve, got {_string_attributes(out, 'tt')['bounds']!r}"
        )

    def test_an_exchange_rewrites_each_name_once(self):
        """Exchanging two names rewrites in one pass, so neither substitution undoes the other."""
        out = NetCDF.read_file(CF_FIXTURE).rename_dims(time="plev", plev="time")
        assert _string_attributes(out, "tas")["cell_methods"] == (
            "plev: mean (interval: 1 month)"
        ), f"got {_string_attributes(out, 'tas')['cell_methods']!r}"

    def test_a_drop_leaves_the_attribute_values_alone(self):
        """A drop renames nothing, so no attribute value is rewritten."""
        nc = NetCDF.read_file(CF_FIXTURE)
        before = _string_attributes(nc, "area")
        out = nc.drop_dims("time")
        assert _string_attributes(out, "area") == before, (
            f"a drop must copy verbatim, got {_string_attributes(out, 'area')}"
        )


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
        spanning = source.GetRootGroup().CreateMDArray(
            "t", [dropped], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        with pytest.raises(ValueError, match="being dropped"):
            remap.axes(spanning)
        assert destination.GetRootGroup().GetDimensions() == [], (
            "the dropped dimension must not reach the destination"
        )

    def test_the_array_copy_is_injected_not_imported(self):
        """`_DimensionRemap` carries coordinates through the callable it was handed (N3).

        The value object used to call `NetCDF._recreate_md_array` by name although it is defined
        before that class, which made it cyclically dependent on the class it serves.
        """
        calls = []

        def _record(dst_group, var_name, src_mdarray, dst_dims, dim_rename=None):
            calls.append(var_name)
            return NetCDF._recreate_md_array(
                dst_group, var_name, src_mdarray, dst_dims, dim_rename
            )

        source = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        src_rg = source.GetRootGroup()
        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = src_rg.CreateDimension("time", "", "", 2)
        coord = src_rg.CreateMDArray("time", [dim], f64)
        coord.Write(np.array([0.0, 1.0]))
        dim.SetIndexingVariable(coord)
        data = src_rg.CreateMDArray("v", [dim], f64)
        data.Write(np.array([2.0, 3.0]))
        destination = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
        remap = _DimensionRemap(
            destination.GetRootGroup(), {"time": "tt"}, set(), _record
        )
        remap.bind(src_rg, destination.GetRootGroup())
        remap.recreate(destination.GetRootGroup(), "v", data)
        assert calls == ["tt", "v"], (
            f"the injected copy must be the one used, got {calls}"
        )


def _flat_store(time_size=3, time_values=(0.0, 6.0, 12.0), index_time=True):
    """A minimal root-only store with `t2m(time,y,x)`, optionally without a time index variable.

    `index_time=False` leaves the `time` dimension with no indexing variable while still writing
    the CF same-named 1-D array, which is the shape the coordinate fallback exists for.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    rg = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_time = rg.CreateDimension("time", gdal.DIM_TYPE_TEMPORAL, "", time_size)
    dim_y = rg.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", 2)
    dim_x = rg.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", 2)
    time_coord = rg.CreateMDArray("time", [dim_time], f64)
    time_coord.Write(np.array(list(time_values)))
    if index_time:
        dim_time.SetIndexingVariable(time_coord)
    for dim, values in ((dim_y, [1.0, 0.0]), (dim_x, [0.0, 1.0])):
        coord = rg.CreateMDArray(dim.GetName(), [dim], f64)
        coord.Write(np.array(values))
        dim.SetIndexingVariable(coord)
    data = rg.CreateMDArray("t2m", [dim_time, dim_y, dim_x], f64)
    data.Write(np.arange(float(time_size * 4)).reshape(time_size, 2, 2))
    data.SetSpatialRef(_wgs84())
    return store


class TestRebuildCoordinateFallback:
    """A dimension GDAL reports no indexing variable for is still stamped, by CF convention."""

    def test_a_dimension_without_an_indexing_variable_keeps_its_coordinates(self):
        """The rebuild falls back to the CF same-named 1-D array in the dimension's own group.

        `GetIndexingVariable()` answers `None` for a store that never declared one, so without
        the fallback the renamed axis would come back unstamped.
        """
        store = _flat_store(index_time=False)
        assert store.GetRootGroup().GetDimensions()[0].GetIndexingVariable() is None, (
            "precondition: the time dimension must have no indexing variable"
        )
        out = Container(store).rename_dims(time="t")
        assert "t" in (out.dimension_names or []), f"got {out.dimension_names}"
        assert_allclose(np.asarray(out.coords["t"]), [0.0, 6.0, 12.0])


class TestRebuildCarriesStringArrayAttributes:
    """A multi-value string attribute goes through a different GDAL call than a single string."""

    def test_a_string_array_attribute_is_carried_value_by_value(self):
        """`flag_meanings = ['low', 'medium', 'high']` survives a rename intact.

        A one-element string attribute is written with `WriteString` and a multi-element one with
        `WriteStringArray`; only the first path was exercised before.
        """
        store = _flat_store()
        array = store.GetRootGroup().OpenMDArray("t2m")
        attribute = array.CreateAttribute(
            "flag_meanings", [3], gdal.ExtendedDataType.CreateString()
        )
        attribute.WriteStringArray(["low", "medium", "high"])
        out = Container(store).rename_dims(time="t")
        carried = out._raster.GetRootGroup().OpenMDArray("t2m")
        values = {
            attr.GetName(): attr.ReadAsStringArray()
            for attr in carried.GetAttributes()
            if attr.GetName() == "flag_meanings"
        }
        assert values.get("flag_meanings") == ["low", "medium", "high"], (
            f"the string array must be carried verbatim, got {values}"
        )


class TestRebuildGroupNestingLimit:
    """Nesting past the rebuild's depth limit is reported rather than silently dropped."""

    def test_nesting_deeper_than_the_limit_warns(self):
        """Groups below `_MAX_GROUP_DEPTH` are not rebuilt, and the caller is warned by name.

        The limit stops unbounded recursion; the warning is what keeps the omission from being
        silent data loss of the kind the sub-group fix exists to prevent.
        """
        store = _flat_store()
        group = store.GetRootGroup()
        for level in range(_MAX_GROUP_DEPTH + 2):
            group = group.CreateGroup(f"g{level}")
        container = Container(store)
        with pytest.warns(UserWarning, match="Group nesting deeper than"):
            out = container.rename_dims(time="t")
        assert "t" in (out.dimension_names or []), (
            "the rename itself must still succeed"
        )
