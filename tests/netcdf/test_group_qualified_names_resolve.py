"""A name the variable enumeration hands out must be a name the readers accept.

Enumerating a grouped NetCDF-4 store recurses into its sub-groups and names
what it finds by path (`flight_03/CO`). `OpenMDArray` resolves names only
within the group it is called on, so three consumers that took those names and
opened them against the root group broke: the xarray export raised
`Array flight_03/CO does not exist`, `add_variable` raised the same, and
`_variable_dim_names` swallowed it and reported no dimensions -- which
classified every sub-group variable as non-spatial.

`open_mdarray` now walks the path, so one resolver serves every caller.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf._mdim import open_mdarray

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
GROUPED = DATA / "none__35v__1d35__groups-nc4.nc"


class TestOpenMdarrayWalksGroupPaths:
    """The resolver the other three consumers share."""

    def test_every_enumerated_name_opens(self):
        """The invariant: enumerate it, then you can open it.

        Test scenario:
            A name the enumeration emits but no reader can resolve is worse
            than one it never emitted, because each consumer fails differently
            -- one raises, one silently reports nothing.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        rg = dataset._working_group()

        unopenable = [n for n in dataset.variable_names if open_mdarray(rg, n) is None]

        assert unopenable == [], f"enumerated but unopenable: {unopenable}"

    def test_a_bare_name_still_resolves(self):
        """The flat case must be unaffected by the path walk."""
        dataset = NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__y-asc.nc"))
        rg = dataset._working_group()

        assert open_mdarray(rg, "temperature") is not None

    def test_a_missing_name_is_still_None_not_an_error(self):
        """The helper's contract is None, including for a bogus path."""
        dataset = NetCDF.read_file(str(GROUPED))
        rg = dataset._working_group()

        assert open_mdarray(rg, "no_such_group/no_such_var") is None
        assert open_mdarray(rg, "no_such_var") is None

    def test_dimensions_resolve_for_a_group_qualified_name(self):
        """`_variable_dim_names` reported `[]` and mis-classified the variable.

        Test scenario:
            Returning no dimensions made every sub-group variable look
            non-spatial, so a container-wide crop treated them as auxiliaries
            and warned that it could not carry them.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        rg = dataset._working_group()
        qualified = next(n for n in dataset.variable_names if "/" in n)

        assert dataset._variable_dim_names(rg, qualified) != []


class TestTheXarrayExportHandlesAGroupedStore:
    """It raised outright; before that it silently exported one variable."""

    def test_it_exports_the_sub_group_variables(self):
        """More than the root's one variable comes back.

        Test scenario:
            At the base commit the export "succeeded" by enumerating only the
            root group -- one variable out of thirty, with nothing said.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exported = dataset.to_xarray()

        assert len(exported.data_vars) > 5

    def test_it_warns_about_what_it_could_not_represent(self):
        """An xarray Dataset has one size per dimension name; a store need not.

        Test scenario:
            Two sub-groups each declare their own `air_press` at different
            lengths, so they cannot both be exported. Skipping silently would
            repeat the defect; the warning names the variables and points at
            `get_variable`.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset.to_xarray()

        skipped = [str(w.message) for w in caught if "skipped" in str(w.message)]
        assert skipped, "no warning for the variables that could not be exported"
        assert "get_variable" in skipped[0]

    def test_a_flat_store_exports_everything_without_warning(self):
        """The common case gains neither a skip nor a warning."""
        dataset = NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__y-asc.nc"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exported = dataset.to_xarray()

        assert "temperature" in exported.data_vars
        assert not [w for w in caught if "skipped" in str(w.message)]


class TestTheLazyExportResolvesTheSameNames:
    """`to_xarray()`'s two arms must agree about which names exist.

    The eager arm was taught the group walk and the lazy one was not, so
    `to_xarray(chunks=...)` on a grouped store went from returning a Dataset to
    raising `Array flight_03/CO does not exist` -- a regression against the base
    commit, on a call the eager arm answers happily.
    """

    def test_a_chunked_export_of_a_grouped_store_returns_a_dataset(self):
        """The lazy arm must resolve a sub-group name, not raise on it.

        Test scenario:
            `build_lazy_array` opened the enumerated name against the root
            group, where a group-qualified name does not exist.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exported = dataset.to_xarray(chunks="auto")

        assert len(exported.data_vars) > 5

    def test_the_two_arms_export_the_same_variables(self):
        """One method, one answer -- `chunks=` is a memory strategy, not a filter.

        Test scenario:
            Comparing the arms against each other is what catches a name the
            eager path resolves and the lazy path does not.
        """
        dataset = NetCDF.read_file(str(GROUPED))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eager = dataset.to_xarray()
            lazy = dataset.to_xarray(chunks="auto")

        assert sorted(lazy.data_vars) == sorted(eager.data_vars)


class TestTheReadPathsResolveGroupQualifiedNames:
    """The other consumers that still opened an enumerated name against the root."""

    def test_read_variable_returns_the_sub_group_array(self):
        """`_read_variable` swallowed the GDAL error and reported "not found".

        Test scenario:
            The MDArray branch catches `RuntimeError`, so a group-qualified
            name fell through to the indexing-variable fallback and the read
            came back as `None` -- a missing variable, silently.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        qualified = next(n for n in dataset.variable_names if "/" in n)

        values = dataset._read_variable(qualified)

        assert values is not None, f"{qualified} read back as missing"
        assert values.size > 0

    def test_the_streaming_fan_out_reads_a_sub_group_variables_attributes(self):
        """`_add_aux_var_spec` treated the failed open as "nothing to carry".

        Test scenario:
            The streamed write would have produced a file missing the
            variable entirely, with no warning -- the same defect as the lazy
            arm's, only silent.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        rg = dataset._working_group()
        qualified = next(n for n in dataset.variable_names if "/" in n)
        dims: dict = {}
        coords: dict = {}
        var_specs: dict = {}
        aux_data: dict = {}

        dataset._add_aux_var_spec(qualified, rg, dims, coords, var_specs, aux_data)

        assert qualified in var_specs, "the aux variable was skipped"
        assert aux_data[qualified].size > 0


class TestAddVariableFromAGroupedStore:
    """The copy has to land under a name netCDF can actually write.

    `add_variable` copies the source's *readable* names, which for a grouped
    store are group-qualified. Creating root-level arrays under those literal
    names built a container that `to_file` then refused with
    `NetCDF: Name contains illegal characters` -- far from the call that caused
    it, on an object the caller had been told was fine.
    """

    @staticmethod
    def _merged() -> NetCDF:
        """A flat container with a grouped store's variables copied into it."""
        target = NetCDF.read_file(str(DATA / "cf__5v__1d4-3d1__geog__y-desc.nc"))
        target.add_variable(NetCDF.read_file(str(GROUPED)))
        return target

    def test_no_copied_array_carries_a_group_separator(self):
        """netCDF-4 forbids `/` in a variable name, so none may be created.

        Test scenario:
            The destination root group is flat; the source's path is not a
            name there, it is a path that no longer means anything.
        """
        merged = self._merged()

        names = merged._working_group().GetMDArrayNames() or []

        assert [n for n in names if "/" in n] == []

    def test_the_result_can_be_written(self, tmp_path):
        """The failure surfaced at write time, so that is where it is pinned.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            `add_variable` returned successfully and left behind a container
            that could not be serialised at all -- `to_file` wrote nothing.
        """
        merged = self._merged()
        destination = tmp_path / "merged.nc"

        merged.to_file(str(destination))

        assert destination.exists()
        written = NetCDF.read_file(str(destination))
        assert len(written._readable_variable_names()) > 5

    def test_the_source_variables_arrive(self):
        """A copy that dropped everything would satisfy the name check alone.

        Test scenario:
            The leaf names of the grouped store's arrays have to be present,
            alongside the destination's own.
        """
        merged = self._merged()

        names = set(merged._working_group().GetMDArrayNames() or [])

        assert "t2m" in names, "the destination's own variable went missing"
        assert len(names) > 6, f"the grouped source's arrays were not copied: {names}"
