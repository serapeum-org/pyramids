"""`nc.variables` says why a readable name is not one of its keys.

`_LazyVariableDict` enumerates the *data* variables, while the accessor it
wraps reads more than those: on a curvilinear ROMS container
`nc.get_variable("lat_rho")` returns a variable that `nc.variables` does not
carry. That asymmetry is deliberate -- `variables` is the data-variable view,
and `set(nc.variables) == set(nc.variable_names)` is a contract the sample
suite asserts on every fixture.

What was not deliberate is how it read. A bare `KeyError('lat_rho')` on a name
the file plainly contains looks like a missing variable rather than a
mis-chosen accessor, so the refusal now names the one that would have worked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"
CURVILINEAR = DATA / "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
RECTILINEAR = DATA / "cf__5v__1d4-4d1__y-asc.nc"


class TestARefusedReadableNamePointsAtTheAccessor:
    """The message distinguishes "not a data variable" from "not here"."""

    @pytest.mark.parametrize("name", ["lat_rho", "lon_rho", "Cs_r", "h"])
    def test_it_names_get_variable_for_a_readable_non_data_variable(self, name: str):
        """The regression: the four names readable but not enumerated.

        Args:
            name: A name `get_variable` resolves on the ROMS fixture.

        Test scenario:
            `lat_rho` / `lon_rho` are the 2-D curvilinear coordinates, `Cs_r`
            a stretching curve and `h` the bathymetry. CF classification
            leaves all four out of the data variables, so the mapping refuses
            them -- but the file has them, and the message has to say which
            accessor does.
        """
        nc = NetCDF.read_file(str(CURVILINEAR))
        assert nc.get_variable(name) is not None

        with pytest.raises(KeyError, match="is not a data variable") as excinfo:
            nc.variables[name]

        assert f"get_variable({name!r})" in str(excinfo.value)

    def test_a_name_that_is_not_in_the_file_is_refused_plainly(self):
        """A typo must not be dressed up as an accessor problem.

        Test scenario:
            The two cases are told apart by the message. A name that is in
            neither set gets the ordinary `KeyError`, with no advice to reach
            for `get_variable` -- which would not have worked either.
        """
        nc = NetCDF.read_file(str(CURVILINEAR))

        with pytest.raises(KeyError) as excinfo:
            nc.variables["not_a_variable"]

        assert "not a data variable" not in str(excinfo.value)
        assert "not_a_variable" in str(excinfo.value)

    def test_get_still_returns_the_default_rather_than_raising(self):
        """`get` is the "I do not mind" spelling and keeps working that way.

        Test scenario:
            The message is for `[]`, which the caller asked to succeed.
            `get` already expresses tolerance for a miss, so it must stay
            silent for both kinds.
        """
        nc = NetCDF.read_file(str(CURVILINEAR))

        assert nc.variables.get("lat_rho") is None
        assert nc.variables.get("not_a_variable", "fallback") == "fallback"


class TestTheDataVariableViewIsUnchanged:
    """Wording a refusal must not move what the mapping contains."""

    def test_the_keys_still_equal_the_enumeration(self):
        """The contract the sample suite asserts on every fixture.

        Test scenario:
            `variables` is the data-variable view. Had the refusal been fixed
            by widening it instead, this equality -- and the sample test that
            checks it across the whole corpus -- would have broken.
        """
        nc = NetCDF.read_file(str(CURVILINEAR))

        assert set(nc.variables) == set(nc.variable_names) == {"salt", "zeta"}

    def test_every_data_variable_still_loads(self):
        """The lazy load past the new guard, on the fixture that has both.

        Test scenario:
            The guard runs before the cache fill, so an over-eager one would
            refuse the enumerated names too and break every reader.
        """
        nc = NetCDF.read_file(str(CURVILINEAR))

        assert all(nc.variables[name] is not None for name in nc.variable_names)

    def test_a_loaded_variable_is_served_from_the_cache(self):
        """Second access must not re-run the guard's readable-name scan.

        Test scenario:
            `__getitem__` short-circuits on a key already materialised, so
            the same object comes back and `get_variable` is not called
            again.
        """
        nc = NetCDF.read_file(str(RECTILINEAR))
        name = nc.variable_names[0]

        assert nc.variables[name] is nc.variables[name]


class TestInvalidateCachesDropsEveryDerivedValue:
    """The derived geotransform is derived state, so it is dropped too."""

    def test_the_derived_geotransform_is_cleared(self):
        """It was the one memo the invalidator walked past.

        Test scenario:
            `_invalidate_caches` exists to drop what was computed from the
            container's contents. The geotransform derived from `lon` / `lat`
            is exactly that, and it was left behind while the variables cache
            it is computed through was cleared out from under it.
        """
        variable = NetCDF.read_file(str(RECTILINEAR)).get_variable("temperature")
        _ = variable.geotransform
        assert variable._derived_geotransform is not None

        variable._invalidate_caches()

        assert variable._derived_geotransform is None

    def test_the_next_read_rederives_the_same_grid(self):
        """Dropping the memo must not change the answer.

        Test scenario:
            Invalidation is only safe if re-derivation is faithful; a cleared
            cache that came back with a different origin would move every
            subsequent point lookup.
        """
        variable = NetCDF.read_file(str(RECTILINEAR)).get_variable("temperature")
        before = tuple(variable.geotransform)

        variable._invalidate_caches()

        assert tuple(variable.geotransform) == before

    def test_setting_a_global_attribute_leaves_the_geotransform_intact(self):
        """The live caller of the invalidator, end to end.

        Test scenario:
            `set_global_attribute` invalidates, so it now drops the derived
            geotransform as well. The grid it describes did not change, so
            the value the next read reports must not either. Built in memory
            rather than read from disk: the fixtures open read-only, and
            `CreateAttribute` refuses those.
        """
        nc = NetCDF.from_array(
            arr=np.ones((5, 5), dtype=np.float64),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 5.0, 0.0, -1.0)),
            variable_name="v",
        )
        before = tuple(nc.geotransform)

        nc.set_global_attribute("title", "regression")

        assert nc._derived_geotransform is None
        assert tuple(nc.geotransform) == before
