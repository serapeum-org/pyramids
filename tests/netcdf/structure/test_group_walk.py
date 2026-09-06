"""What the sub-group walk filters out, and what it silently stops at.

Both cases are built here rather than taken from a fixture, because no store in
`tests/data/netcdf` exercises either: the grouped fixture's sub-groups declare
no dimensions of their own *and* hold no array named after the root's `recNum`,
and nothing in the suite nests more than one group deep.

The stores are written as real netCDF-4 files and read back through
`NetCDF.read_file`, so what is asserted is what a user sees from
`variable_names`, not the private walk in isolation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.netcdf import _MAX_GROUP_DEPTH

pytestmark = pytest.mark.core

_F64 = None  # built lazily inside the writers; gdal types are not module-safe


def _write_inherited_dimension_store(path: Path) -> None:
    """Write a store whose sub-group holds a coordinate array for a parent dimension.

    netCDF-4 dimensions are visible in every descendant group, so `flight/recNum`
    is the coordinate variable of the **root's** `recNum` dimension -- the exact
    shape a sub-group filter that only reads `rg.GetDimensions()` cannot see.
    """
    driver = gdal.GetDriverByName("netCDF")
    dataset = driver.CreateMultiDimensional(str(path), [], ["FORMAT=NC4"])
    root = dataset.GetRootGroup()
    dim = root.CreateDimension("recNum", "", "", 4)
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    root.CreateMDArray("recNum", [dim], f64).Write(np.arange(4.0))
    sub = root.CreateGroup("flight")
    sub.CreateMDArray("recNum", [dim], f64).Write(np.arange(4.0) * 10)
    sub.CreateMDArray("CO", [dim], f64).Write(np.arange(4.0))
    dataset.Close()


def _write_deeply_nested_store(path: Path, depth: int) -> None:
    """Write a chain of `depth` nested groups, each holding one 1-D variable."""
    driver = gdal.GetDriverByName("netCDF")
    dataset = driver.CreateMultiDimensional(str(path), [], ["FORMAT=NC4"])
    root = dataset.GetRootGroup()
    dim = root.CreateDimension("n", "", "", 3)
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    root.CreateMDArray("root_var", [dim], f64).Write(np.arange(3.0))
    group = root
    for level in range(depth):
        group = group.CreateGroup(f"g{level}")
        group.CreateMDArray(f"v{level}", [dim], f64).Write(np.arange(3.0))
    dataset.Close()


class TestAnInheritedDimensionIsFilteredLikeAnOwnedOne:
    """A sub-group's coordinate axis is not a data variable.

    The walk dropped an array whose name matched one of *its own group's*
    dimensions. A netCDF-4 sub-group inherits its parents' dimensions and
    usually declares none itself, so its coordinate variables matched nothing
    and were enumerated as data.

    That is not cosmetic: `variable_names` is what `to_xarray` splits into
    `data_vars`, what `to_netcdf` writes, and what a container-wide `crop` /
    `to_crs` fans out over. A coordinate axis reported as data goes through all
    three as data.
    """

    def test_the_sub_group_axis_is_not_enumerated(self, tmp_path):
        """`flight/recNum` indexes `flight/CO`; it is not a variable of its own.

        Test scenario:
            The root's own `recNum` was already filtered, because the root
            declares that dimension. The sub-group's copy of the same axis was
            not, so one file reported the same axis twice -- once as a
            dimension and once as data.
        """
        path = tmp_path / "inherited.nc"
        _write_inherited_dimension_store(path)

        names = NetCDF.read_file(str(path)).variable_names

        assert names == ["flight/CO"], f"the inherited axis is enumerated: {names}"

    def test_the_group_view_answers_the_same_way(self, tmp_path):
        """Opening the group must not change what the group contains.

        Test scenario:
            `get_group("flight").variable_names` resolves the sub-group as its
            own working group, where `rg.GetDimensions()` is empty -- so a
            filter keyed on the group's declared dimensions gave the container
            and the view two different answers about one group. Reading the
            array's own dimensions gives them one.
        """
        path = tmp_path / "inherited.nc"
        _write_inherited_dimension_store(path)
        dataset = NetCDF.read_file(str(path))

        view = dataset.get_group("flight")

        assert view.variable_names == ["CO"], (
            f"the view disagrees with the container: {view.variable_names}"
        )

    def test_the_axis_is_reachable_the_way_a_root_axis_is(self, tmp_path):
        """Filtering it out costs the same reachability a root axis costs.

        Test scenario:
            This states the price of the change. A dimension coordinate array
            has never been reachable through `get_variable` -- `get_variable`
            gates on the readable superset, which excludes the root's `lat` on
            every CF fixture -- so a sub-group axis now behaves identically:
            absent from both lists, still readable through `_read_variable`,
            which is what the plot engine and the carry paths use.
        """
        path = tmp_path / "inherited.nc"
        _write_inherited_dimension_store(path)
        dataset = NetCDF.read_file(str(path))

        assert "flight/recNum" not in dataset._readable_variable_names(), (
            "an axis must not be advertised as readable when the root's is not"
        )
        values = dataset._read_variable("flight/recNum")

        assert values is not None, (
            "the axis must stay readable by name, but did not resolve at all"
        )
        assert list(values) == [0.0, 10.0, 20.0, 30.0], (
            f"the axis resolved but read back the wrong values: {list(values)}"
        )


class TestTheDepthCapIsAnnouncedWhenItFires:
    """A store nested past the cap loses variables, so it must say so.

    The cap exists so a malformed or hostile store cannot turn enumeration into
    an unbounded walk, and it stays. What changed is that reaching it is no
    longer silent: every listing, export and conversion built from
    `variable_names` was quietly short of the file's contents.
    """

    def test_a_store_past_the_cap_warns_and_names_what_was_skipped(self, tmp_path):
        """The user is told which group the walk stopped at.

        Test scenario:
            A chain one level past the cap loses the variables beneath it. The
            warning has to be specific enough to act on, so it names the depth
            and the group path that was not descended into.
        """
        depth = _MAX_GROUP_DEPTH + 1
        path = tmp_path / "deep.nc"
        _write_deeply_nested_store(path, depth)
        dataset = NetCDF.read_file(str(path))

        with pytest.warns(UserWarning) as record:
            names = dataset.variable_names

        message = str(record[0].message)
        assert str(_MAX_GROUP_DEPTH) in message, f"the cap is not named: {message}"
        assert f"g{_MAX_GROUP_DEPTH - 1}/" in message, (
            f"the group the walk stopped at is not named: {message}"
        )
        assert len(names) == _MAX_GROUP_DEPTH + 1, (
            f"the cap itself must still hold: {len(names)} names"
        )

    def test_the_warning_points_at_the_call_that_read_the_store(self, tmp_path):
        """A warning nobody can locate is only half a signal.

        Test scenario:
            The warning is raised `_MAX_GROUP_DEPTH` frames inside
            `_mdim_data_variable_names`' own recursion, so `stacklevel=2`
            attributed it to that function calling itself -- a file and line in
            `netcdf.py`. Counting from `depth` instead would only move it to
            `_get_variable_names`, one frame further in. What a user needs is
            the line in *their* file, so the level is computed by walking out
            of the package.
        """
        path = tmp_path / "deep.nc"
        _write_deeply_nested_store(path, _MAX_GROUP_DEPTH + 1)
        dataset = NetCDF.read_file(str(path))

        with pytest.warns(UserWarning) as record:
            _ = dataset.variable_names

        raised_at = Path(record[0].filename)
        assert raised_at == Path(__file__), (
            f"the warning is attributed to {raised_at}, not to the caller"
        )

    def test_a_store_within_the_cap_stays_quiet(self, tmp_path):
        """The cap never fires on a real store, so it must never warn on one.

        Test scenario:
            A warning on every ordinary file would be worse than the silence it
            replaces. A chain exactly at the cap is the boundary case: fully
            enumerated, nothing skipped, nothing said.
        """
        path = tmp_path / "at_cap.nc"
        _write_deeply_nested_store(path, _MAX_GROUP_DEPTH)
        dataset = NetCDF.read_file(str(path))

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            names = dataset.variable_names

        depth_warnings = [w for w in record if "nesting" in str(w.message)]
        assert not depth_warnings, f"warned without losing anything: {depth_warnings}"
        assert len(names) == _MAX_GROUP_DEPTH + 1, (
            f"the whole chain should be enumerated: {len(names)} names"
        )
