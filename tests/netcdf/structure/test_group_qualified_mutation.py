"""`rename_variable` / `remove_variable` and the group-qualified names they meet.

The variable enumeration walks sub-groups, so `"flight_03/CO"` is a name the
container reports and `get_variable` accepts. Both mutators check existence
against that same list, so a qualified name got past the check and then died
two calls later inside GDAL with
``RuntimeError('Array flight_03/CO is not an array of this group')`` -- a raw
driver error where both methods document :class:`ValueError`.

The resolution is to refuse it here, because these methods mutate **one**
group: the container's working group. GDAL will not delete another group's
array from the root, and creating the renamed copy at the root would move the
variable rather than rename it. `get_group(...)` already returns a writable
view of the sub-group where the bare leaf name works, so the refusal names
that path -- and `test_the_refusal_names_a_path_that_works` follows it, so the
advice cannot rot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"

# 29 variables spread over seven per-flight sub-groups, plus one at the root.
GROUPED = DATA / "none__35v__1d35__groups-nc4.nc"

# Declares `lat` / `lon` / `time` dimension coordinates alongside `tos`.
FLAT = DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"


def _qualified_name(dataset: NetCDF) -> str:
    """Return the first group-qualified name the container enumerates."""
    return next(name for name in dataset.variable_names if "/" in name)


class TestAQualifiedNameIsRefusedRatherThanFailingInGdal:
    """The error is the documented one, and it says what to do instead."""

    def test_rename_raises_value_error_naming_the_group_view(self):
        """A `ValueError`, not GDAL's `RuntimeError`, and an actionable one.

        Test scenario:
            The name comes straight out of `variable_names`, so the caller has
            every reason to think it is renameable. What they got was a driver
            message about groups, from a method whose `Raises:` section
            promises `ValueError`. The replacement names the sub-group and the
            call that does work.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        qualified = _qualified_name(dataset)
        group = qualified.rpartition("/")[0]

        with pytest.raises(ValueError) as excinfo:
            dataset.rename_variable(qualified, "renamed_thing")

        message = str(excinfo.value)
        assert group in message, f"the refusal must name the sub-group: {message}"
        assert "get_group" in message, (
            f"the refusal must name the call that works: {message}"
        )

    def test_remove_raises_value_error_naming_the_group_view(self):
        """`remove_variable` had no existence check at all, so GDAL answered.

        Test scenario:
            Same qualified name, same driver error, same documented contract.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        qualified = _qualified_name(dataset)
        group = qualified.rpartition("/")[0]

        with pytest.raises(ValueError) as excinfo:
            dataset.remove_variable(qualified)

        message = str(excinfo.value)
        assert group in message, f"the refusal must name the sub-group: {message}"
        assert "get_group" in message, (
            f"the refusal must name the call that works: {message}"
        )

    def test_the_refusal_names_a_path_that_works(self):
        """The advice in the message is executed here, not just asserted.

        Test scenario:
            A refusal that recommends an impossible workaround is worse than
            no advice. So the suggested spelling -- open the group, use the
            leaf name -- is run, and the rename is checked on the view.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        qualified = _qualified_name(dataset)
        group, _, leaf = qualified.rpartition("/")

        view = dataset.get_group(group)
        view.rename_variable(leaf, "renamed_thing")

        assert "renamed_thing" in view.variable_names, (
            f"the recommended path did not rename: {view.variable_names}"
        )
        assert leaf not in view.variable_names, (
            f"the old name survived the rename: {view.variable_names}"
        )

    def test_rename_refuses_a_qualified_destination_name(self):
        """A `/` in `new_name` would build a container that cannot be written.

        Test scenario:
            `_add_md_array_to_group` creates the renamed array in the working
            group under the literal string it is given, so `"grp/b"` becomes a
            root array whose name netCDF-4 forbids -- the container then fails
            `to_file` with "Name contains illegal characters", long after the
            rename that caused it. That is the exact trap `_free_target_name`
            closes for `add_variable`; this closes it for `rename_variable`.
        """
        dataset = NetCDF.read_file(str(FLAT))

        with pytest.raises(ValueError) as excinfo:
            dataset.rename_variable("tos", "grp/renamed")

        assert "grp/renamed" in str(excinfo.value), (
            f"the refusal must echo the rejected name: {excinfo.value}"
        )
        assert "grp/renamed" not in dataset._working_group().GetMDArrayNames(), (
            "an illegally named array was created anyway"
        )

    def test_a_refused_rename_leaves_the_store_untouched(self):
        """Refusing must not be half-done.

        Test scenario:
            The old failure raised after `_writable_root_group` had already
            copied the store, but before `_replace_raster`, so nothing was
            lost. The check now runs first; this pins that the observable
            store is identical either way.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        before = list(dataset.variable_names)

        with pytest.raises(ValueError):
            dataset.rename_variable(_qualified_name(dataset), "renamed_thing")

        assert dataset.variable_names == before, "the refused rename changed the store"
        assert "renamed_thing" not in dataset._readable_variable_names(), (
            "the refused rename created an array"
        )


class TestRemoveVariableReportsAMissingNameAsValueError:
    """The other raw-GDAL escape from the same method."""

    def test_an_absent_name_raises_value_error(self):
        """`remove_variable('nope')` answered with a driver message.

        Test scenario:
            No existence check ran at all, so GDAL's "is not an array of this
            group" reached the caller for a plain typo. `rename_variable`
            already documented and raised `ValueError` for the same mistake.
        """
        dataset = NetCDF.read_file(str(FLAT))

        with pytest.raises(ValueError) as excinfo:
            dataset.remove_variable("zzz_not_here")

        assert "zzz_not_here" in str(excinfo.value), (
            f"the refusal must echo the name: {excinfo.value}"
        )

    def test_a_dimension_coordinate_can_still_be_removed(self):
        """The check must not be `_readable_variable_names`.

        Test scenario:
            `lat` is a dimension coordinate, so it is in neither
            `variable_names` nor the readable superset -- and removing it
            works today. Gating the method on either list would have turned a
            working call into a `ValueError`, which is why the guard resolves
            the array instead of consulting an enumeration.
        """
        dataset = NetCDF.read_file(str(FLAT))
        assert "lat" not in dataset._readable_variable_names(), (
            "the fixture no longer models a dimension coordinate"
        )

        dataset.remove_variable("lat")

        arrays = dataset._working_group().GetMDArrayNames()
        assert "lat" not in arrays, f"the dimension coordinate survived: {arrays}"
