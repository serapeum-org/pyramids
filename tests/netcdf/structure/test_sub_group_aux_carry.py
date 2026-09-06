"""A carried sub-group auxiliary must arrive with *its own* dimension coordinate.

`_aux_dimension_coordinate_names` named the indexing variable by
`MDArray.GetName()`, which is the **leaf**. Every name that reaches
`open_mdarray` is resolved relative to the group it is given, so for an
auxiliary living in a sub-group the leaf resolves at the *root* -- against a
different array whenever one of that name exists there, and against nothing
when it does not.

Both outcomes are silent and both are wrong in the same way: the carried array
ends up labelled by an axis that is not its own. The store built here is the
smallest thing that shows it -- a root `n` and a sub-group `n` of the same size
and different values -- because no fixture in `tests/data/netcdf` repeats a
coordinate leaf name across groups (on the suite's grouped fixture the
sub-group arrays are indexed by the root's `UTC_time`, so the leaf happens to
be root-relative and the bug cannot fire).

The stores are written as real netCDF-4 files and read back through
`NetCDF.read_file`, and the assertions are on the arrays a user finds in the
result, not on the private helper alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

ROOT_N = [100.0, 200.0, 300.0]
GROUP_N = [1.0, 2.0, 3.0]
GROUP_AUX = [7.0, 8.0, 9.0]


def _write_colliding_axis_store(path: Path, *, root_axis: bool) -> None:
    """Write a store whose sub-group declares its own `n` axis.

    `g/aux(n)` is indexed by the sub-group's `n`, never the root's. With
    `root_axis` the root declares an `n` of the same size and different values,
    which is what a leaf-name lookup finds instead; without it the leaf
    resolves to nothing at all.

    Args:
        path: Destination netCDF-4 file.
        root_axis: Whether the root also declares an `n` dimension and array.
    """
    driver = gdal.GetDriverByName("netCDF")
    dataset = driver.CreateMultiDimensional(str(path), [], ["FORMAT=NC4"])
    root = dataset.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    dim_lat = root.CreateDimension("lat", "", "", 4)
    dim_lon = root.CreateDimension("lon", "", "", 5)
    lat = root.CreateMDArray("lat", [dim_lat], f64)
    lat.Write(np.array([10.0, 9.0, 8.0, 7.0]))
    lat.SetUnit("degrees_north")
    lon = root.CreateMDArray("lon", [dim_lon], f64)
    lon.Write(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    lon.SetUnit("degrees_east")
    root.CreateMDArray("tas", [dim_lat, dim_lon], f64).Write(
        np.arange(20.0).reshape(4, 5)
    )
    if root_axis:
        dim_root_n = root.CreateDimension("n", "", "", 3)
        root.CreateMDArray("n", [dim_root_n], f64).Write(np.array(ROOT_N))
    group = root.CreateGroup("g")
    dim_group_n = group.CreateDimension("n", "", "", 3)
    group.CreateMDArray("n", [dim_group_n], f64).Write(np.array(GROUP_N))
    group.CreateMDArray("aux", [dim_group_n], f64).Write(np.array(GROUP_AUX))
    dataset.Close()


def _result_array(result: NetCDF, name: str) -> list[float]:
    """Read one array of a fan-out result as a flat list."""
    return list(np.asarray(result._working_group().OpenMDArray(name).ReadAsArray()))


class TestTheCarriedCoordinateIsTheAuxiliarysOwn:
    """The axis carried in beside `g/aux` must be `g/n`, not the root's `n`."""

    def test_the_helper_names_the_coordinate_group_qualified(self, tmp_path):
        """The mechanism, so a regression is diagnosable and not just visible.

        Test scenario:
            `GetName()` answered `'n'` here, and `open_mdarray(root, 'n')`
            resolves at the root. The name has to be relative to the group the
            carry resolves against, which is `'g/n'`.
        """
        path = tmp_path / "collide.nc"
        _write_colliding_axis_store(path, root_axis=True)
        dataset = NetCDF.read_file(str(path))
        rg = dataset._working_group()
        aux = dataset._carryable_aux_names(rg, dataset._spatial_variable_names(rg))

        names = dataset._aux_dimension_coordinate_names(rg, aux, {"y", "x", "tas"})

        assert aux == ["g/aux"], f"the fixture no longer models the case: {aux}"
        assert names == ["g/n"], f"the coordinate was named by its leaf: {names}"

    def test_a_resample_carries_the_sub_groups_values(self, tmp_path):
        """What the user sees: the carried array's axis holds the wrong numbers.

        Test scenario:
            `g/aux = [7, 8, 9]` is indexed by `g/n = [1, 2, 3]`. The result
            carried the *root's* `n = [100, 200, 300]` in beside it, so the
            auxiliary arrived labelled with another variable's axis -- with no
            warning, and nothing in the file to show which one is meant.
        """
        path = tmp_path / "collide.nc"
        _write_colliding_axis_store(path, root_axis=True)
        dataset = NetCDF.read_file(str(path))

        result = dataset.resample(cell_size=2.0)

        assert _result_array(result, "aux") == GROUP_AUX, (
            "the auxiliary itself was not carried"
        )
        assert _result_array(result, "n") == GROUP_N, (
            "the carried axis is the root's, not the one the auxiliary is "
            f"indexed by: {_result_array(result, 'n')}"
        )

    def test_the_coordinate_arrives_when_the_root_has_no_such_name(self, tmp_path):
        """The other half of the same defect, which used to be a silent drop.

        Test scenario:
            With no `n` at the root, the leaf name resolved to nothing.
            `add_variable` skipped an unresolved name, so the coordinate was
            dropped without a word and `g/aux` landed on a bare dimension. The
            drop is the same fault as the swap: the name being looked up was
            never the array's.
        """
        path = tmp_path / "no_root_axis.nc"
        _write_colliding_axis_store(path, root_axis=False)
        dataset = NetCDF.read_file(str(path))

        result = dataset.resample(cell_size=2.0)
        arrays = set(result._working_group().GetMDArrayNames() or [])

        assert "n" in arrays, f"the coordinate was dropped: {sorted(arrays)}"
        assert _result_array(result, "n") == GROUP_N, (
            f"the carried axis holds the wrong values: {_result_array(result, 'n')}"
        )

    def test_the_streamed_arm_reads_the_axis_off_the_dimension(self, tmp_path):
        """The streamed writer never had this bug, and must not acquire it.

        Test scenario:
            `_add_aux_var_spec` reads the indexing variable straight off the
            dimension, so no name is looked up and no wrong array can be
            found. The two arms cannot be compared end to end on this store --
            `resample(path=...)` writes a carried auxiliary under its
            group-qualified name, which netCDF-4 forbids, and raises "Name
            contains illegal characters" on the base commit exactly as it does
            here -- so the spec the writer builds is asserted instead.
        """
        path = tmp_path / "collide.nc"
        _write_colliding_axis_store(path, root_axis=True)
        dataset = NetCDF.read_file(str(path))
        rg = dataset._working_group()
        coords: dict = {}

        dataset._add_aux_var_spec("g/aux", rg, {}, coords, {}, {}, ["g/aux"])

        assert list(coords["n"][0]) == GROUP_N, (
            f"the streamed arm wrote the wrong axis: {coords['n'][0]}"
        )
