"""Tests for ``NetCDF.set_coords`` and ``NetCDF.reset_coords``.

CF marks a variable as a coordinate of another by naming it in that other's ``coordinates``
attribute, and pyramids already reads that: ``cf.classify_variables`` assigns the
auxiliary-coordinate role from it and ``variable_names`` filters itself by the roles. These
two members are the write side, so a promoted variable leaves ``data_vars`` while staying
readable by name, and a demoted one comes back.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from osgeo import gdal

from pyramids.netcdf import GeoReference, NetCDF
from pyramids.netcdf.netcdf import Container

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
NY, NX = 2, 2


def _geo_ref() -> GeoReference:
    """The grid every cube in this module sits on.

    Returns:
        GeoReference: A 1-degree WGS 84 grid.
    """
    return GeoReference(geo=GEO, epsg=4326)


def _container(**variables: float) -> NetCDF:
    """A container holding one constant-valued variable per keyword.

    Args:
        **variables: `name=fill` for each variable to create.

    Returns:
        NetCDF: The container, with the variables in the order given.
    """
    names = list(variables)
    first = names[0]
    container = NetCDF.from_array(
        np.full((NY, NX), variables[first]), geo_ref=_geo_ref(), variable_name=first
    )
    for name in names[1:]:
        donor = NetCDF.from_array(
            np.full((NY, NX), variables[name]), geo_ref=_geo_ref(), variable_name=name
        ).get_variable(name)
        container.set_variable(name, donor)
    return container


def _grouped_store(flat: NetCDF) -> NetCDF:
    """Copy `flat`'s arrays into a sub-group of a fresh store, for the group-view tests.

    Args:
        flat: A root container whose arrays to copy.

    Returns:
        NetCDF: A container whose `inner` sub-group holds the same arrays.
    """
    source = flat._raster.GetRootGroup()
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    inner = store.GetRootGroup().CreateGroup("inner")
    axes = {
        dim.GetName(): inner.CreateDimension(dim.GetName(), "", "", dim.GetSize())
        for dim in source.GetDimensions()
    }
    for name in source.GetMDArrayNames():
        array = source.OpenMDArray(name)
        copied = inner.CreateMDArray(
            name,
            [axes[d.GetName()] for d in array.GetDimensions()],
            array.GetDataType(),
        )
        copied.Write(array.Read())
    return Container(store)


def _roles(nc: NetCDF) -> dict[str, str]:
    """The CF role of every array in the store.

    Args:
        nc: The container to classify.

    Returns:
        dict[str, str]: Array name to CF role.
    """
    cf = nc.meta_data.cf
    return dict(cf.classifications or {}) if cf is not None else {}


class TestSetCoords:
    """Promoting a data variable to an auxiliary coordinate."""

    def test_a_promoted_variable_leaves_the_data_variables(self):
        """The partition the read side already honours now has a write side.

        Test scenario:
            Promoting `expver` drops it from `variable_names` while `t2m` stays, which
            is xarray's `data_vars` behaviour.
        """
        cube = _container(t2m=1.0, expver=5.0)

        promoted = cube.set_coords("expver")

        assert promoted.variable_names == ["t2m"]

    def test_the_cf_role_changes(self):
        """The store itself says so, not just the Python object.

        Test scenario:
            After promotion `cf.classify_variables` reports `expver` as an auxiliary
            coordinate rather than data.
        """
        cube = _container(t2m=1.0, expver=5.0)

        assert _roles(cube)["expver"] == "data"
        assert _roles(cube.set_coords("expver"))["expver"] == "auxiliary_coordinate"

    def test_the_variable_is_still_readable_with_its_values(self):
        """Only the role changes — nothing is moved, copied or deleted.

        Test scenario:
            `get_variable` reaches the promoted variable and its cells are unchanged.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")

        variable = promoted.get_variable("expver")

        assert variable.band_count == 1
        assert_array_equal(
            np.asarray(variable.read_array(squeeze=True)), np.full((NY, NX), 5.0)
        )

    def test_it_does_not_mutate_the_receiver(self):
        """Non-mutating, like the other structural members.

        Test scenario:
            The container it was called on still lists both variables afterwards.
        """
        cube = _container(t2m=1.0, expver=5.0)

        cube.set_coords("expver")

        assert sorted(cube.variable_names) == ["expver", "t2m"]

    def test_several_names_at_once(self):
        """A sequence promotes each of them.

        Test scenario:
            Two of three variables are promoted and only the third remains data.
        """
        cube = _container(t2m=1.0, expver=5.0, angle=7.0)

        promoted = cube.set_coords(["expver", "angle"])

        assert promoted.variable_names == ["t2m"]

    def test_promoting_twice_is_idempotent(self):
        """A name already referenced is not added again.

        Test scenario:
            Promoting an already-promoted variable leaves the same roles, with no
            duplicate reference.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")

        twice = promoted.set_coords("expver")

        assert twice.variable_names == ["t2m"]
        assert _roles(twice)["expver"] == "auxiliary_coordinate"

    def test_the_promotion_survives_a_round_trip(self, tmp_path):
        """The role lives in the file, so reopening keeps it.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A promoted container written with `to_file` and reopened still reports one
            data variable.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")
        path = tmp_path / "promoted.nc"
        promoted.to_file(str(path))

        reopened = NetCDF.read_file(str(path))

        assert reopened.variable_names == ["t2m"]
        assert _roles(reopened)["expver"] == "auxiliary_coordinate"


class TestSetCoordsRefusals:
    """What cannot be promoted says why."""

    def test_an_unknown_name_is_refused(self):
        """The message lists the names that would work.

        Test scenario:
            A misspelt variable raises, naming the container's variables.
        """
        cube = _container(t2m=1.0, expver=5.0)

        with pytest.raises(ValueError, match="not a variable of this container"):
            cube.set_coords("nope")

    def test_a_dimension_is_refused(self):
        """A dimension's coordinate is its same-named array, which this cannot assign.

        Test scenario:
            Naming the `x` axis raises, explaining that the netCDF driver cannot
            reassign a dimension's indexing variable.
        """
        cube = _container(t2m=1.0, expver=5.0)

        with pytest.raises(ValueError, match="already .* dimension's coordinate"):
            cube.set_coords("x")

    def test_promoting_the_only_variable_is_refused(self):
        """Nothing would be left to reference it, so it would be unreachable as a label.

        Test scenario:
            A one-variable container refuses to promote that variable.
        """
        cube = _container(solo=1.0)

        with pytest.raises(ValueError, match="no data variable is left to reference"):
            cube.set_coords("solo")

    def test_a_single_variable_receiver_is_refused(self):
        """A lone variable has no sibling to carry the reference.

        Test scenario:
            Calling it on a variable rather than its container raises, pointing at the
            container.
        """
        variable = _container(t2m=1.0, expver=5.0).get_variable("expver")

        with pytest.raises(ValueError, match="single variable"):
            variable.set_coords("expver")

    def test_a_group_view_is_refused_rather_than_silently_ignored(self):
        """A reference written inside a group is never matched back, so it must refuse.

        Test scenario:
            A `get_group` view raises, naming the group. Without this the write would
            succeed and the role would not change: CF references are relative names
            (`expver`) while a sub-group's arrays classify as `inner/expver`.
        """
        cube = _container(t2m=1.0, expver=5.0)
        view = _grouped_store(cube).get_group("inner")

        with pytest.raises(ValueError, match="get_group"):
            view.set_coords("expver")

        with pytest.raises(ValueError, match="get_group"):
            view.reset_coords()


class TestResetCoords:
    """Demoting an auxiliary coordinate back to a data variable."""

    def test_a_named_coordinate_comes_back(self):
        """The reference is removed, so the role reverts to data.

        Test scenario:
            A promoted variable demoted by name reappears in `variable_names`.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")

        demoted = promoted.reset_coords("expver")

        assert sorted(demoted.variable_names) == ["expver", "t2m"]
        assert _roles(demoted)["expver"] == "data"

    def test_no_argument_demotes_every_auxiliary_coordinate(self):
        """`names=None` is the whole partition.

        Test scenario:
            Two promoted variables both come back when `reset_coords()` is called with
            no argument.
        """
        promoted = _container(t2m=1.0, expver=5.0, angle=7.0).set_coords(
            ["expver", "angle"]
        )

        demoted = promoted.reset_coords()

        assert sorted(demoted.variable_names) == ["angle", "expver", "t2m"]

    def test_a_promote_demote_round_trip_restores_the_roles(self):
        """The two members are inverses.

        Test scenario:
            Promoting then demoting gives back the original roles exactly.
        """
        cube = _container(t2m=1.0, expver=5.0)
        before = _roles(cube)

        after = _roles(cube.set_coords("expver").reset_coords("expver"))

        assert after == before

    def test_a_receiver_that_never_referenced_it_is_left_alone(self):
        """A data variable added after the promotion carries no reference, and that is fine.

        Test scenario:
            A container is promoted, then gains a second data variable, which therefore
            has no `coordinates` attribute. Demoting must leave that variable's
            attributes untouched rather than writing an empty reference onto it, and
            still bring the coordinate back.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")
        latecomer = NetCDF.from_array(
            np.full((NY, NX), 9.0), geo_ref=_geo_ref(), variable_name="sst"
        ).get_variable("sst")
        promoted.set_variable("sst", latecomer)

        demoted = promoted.reset_coords("expver")

        assert sorted(demoted.variable_names) == ["expver", "sst", "t2m"]
        assert _roles(demoted)["expver"] == "data"

    def test_the_variable_is_never_deleted(self):
        """Demotion changes a role, not the store's contents.

        Test scenario:
            The demoted variable still holds its cells.
        """
        demoted = _container(t2m=1.0, expver=5.0).set_coords("expver").reset_coords()

        assert_array_equal(
            np.asarray(demoted.get_variable("expver").read_array(squeeze=True)),
            np.full((NY, NX), 5.0),
        )

    def test_it_does_not_mutate_the_receiver(self):
        """Non-mutating, as `set_coords` is.

        Test scenario:
            The promoted container still reports one data variable after a demotion was
            taken from it.
        """
        promoted = _container(t2m=1.0, expver=5.0).set_coords("expver")

        promoted.reset_coords()

        assert promoted.variable_names == ["t2m"]

    def test_the_demotion_survives_a_round_trip(self, tmp_path):
        """With nothing left to reference, the attribute is gone from the file.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A demoted container written and reopened reports both variables as data.
        """
        demoted = _container(t2m=1.0, expver=5.0).set_coords("expver").reset_coords()
        path = tmp_path / "demoted.nc"
        demoted.to_file(str(path))

        reopened = NetCDF.read_file(str(path))

        assert sorted(reopened.variable_names) == ["expver", "t2m"]

    def test_nothing_to_demote_is_a_plain_copy(self):
        """A container with no auxiliary coordinates comes back unchanged.

        Test scenario:
            `reset_coords()` on a plain container keeps both variables and does not
            raise.
        """
        cube = _container(t2m=1.0, expver=5.0)

        demoted = cube.reset_coords()

        assert sorted(demoted.variable_names) == ["expver", "t2m"]


class TestResetCoordsRefusals:
    """What cannot be demoted says why."""

    def test_a_data_variable_is_refused(self):
        """Only an auxiliary coordinate can be demoted.

        Test scenario:
            Demoting a plain data variable raises, listing the auxiliary coordinates.
        """
        cube = _container(t2m=1.0, expver=5.0)

        with pytest.raises(ValueError, match="not an auxiliary coordinate"):
            cube.reset_coords("t2m")

    def test_a_dimension_coordinate_is_refused(self):
        """A dimension's coordinate is a name convention, not a removable reference.

        Test scenario:
            Demoting the `y` axis raises, pointing at `rename_variable`.
        """
        cube = _container(t2m=1.0, expver=5.0)

        with pytest.raises(ValueError, match="rename_variable"):
            cube.reset_coords("y")

    def test_a_single_variable_receiver_is_refused(self):
        """As with `set_coords`, the partition belongs to a container.

        Test scenario:
            Calling it on a variable raises, pointing at the container.
        """
        variable = _container(t2m=1.0, expver=5.0).get_variable("t2m")

        with pytest.raises(ValueError, match="single variable"):
            variable.reset_coords()
