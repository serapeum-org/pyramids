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
