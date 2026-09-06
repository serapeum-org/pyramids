"""What `_readable_variable_names` costs, and what it promises about a name.

Two separate claims are pinned here.

*Cost.* The readable set is consulted once per `get_variable`, so the fan-out
over a container asks it once per variable. It used to answer by walking the
store twice -- `variable_names` walked once, then the readable loop walked
again for the same list -- which made every one of those checks quadratic in
the store's array count for no reason. One walk now serves both.

*Promise.* Its docstring says it lists "every array name `get_variable` will
accept". That is true, and it was read as promising a `NetCDF` back, which it
never did: a 1-D or non-numeric array comes back as a raw `gdal.MDArray`. Some
of those names are not even in the wider list -- GOES ABI declares
`time_bounds` as a *data* variable -- so `read_array` met them through the
ordinary container route and died with an `AttributeError`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"

# Flat CF store: one data variable, three bounds arrays, no sub-groups -- so a
# walk counter counts top-level calls only, with no recursion mixed in.
FLAT = DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"

# GOES ABI: `time_bounds` / `x_image_bounds` / `y_image_bounds` are 1-D and are
# enumerated as data variables; `band_id` / `band_wavelength` are 1-D and are
# not. All five come back from `get_variable` as raw MDArrays.
GEOS = DATA / "cf__9v__1d7-2d2__geos__y-desc.nc"

GROUPED = DATA / "none__35v__1d35__groups-nc4.nc"


class TestTheStoreIsWalkedOncePerAnswer:
    """The readable set costs one walk, not two."""

    def test_one_walk_serves_both_lists(self, monkeypatch):
        """The enumeration and the store list come out of the same walk.

        Test scenario:
            Timing is too noisy to assert, but the walk count is exact, and it
            is the thing that made the call quadratic: `get_variable` checks
            the readable set once, so a fan-out over N variables paid 2N walks
            where N were needed. The fixture is flat, so the recursion cannot
            inflate the count.
        """
        dataset = NetCDF.read_file(str(FLAT))
        original = NetCDF._mdim_data_variable_names
        walks: list[str] = []

        def counting(rg, prefix="", depth=0):
            walks.append(prefix)
            return original(rg, prefix, depth)

        monkeypatch.setattr(NetCDF, "_mdim_data_variable_names", staticmethod(counting))
        dataset._readable_variable_names()

        assert len(walks) == 1, f"the store was walked {len(walks)} times, not once"

    def test_the_answer_is_the_same_as_two_walks_gave(self):
        """Sharing the walk must not change the answer.

        Test scenario:
            The enumeration is the CF-filtered subset of the walk and the
            readable set is the walk plus that subset's order, so both halves
            are asserted by content: `tos` alone is enumerated, and the three
            bounds arrays are what the wider list adds.
        """
        dataset = NetCDF.read_file(str(FLAT))

        enumerated = dataset.variable_names
        readable = dataset._readable_variable_names()

        assert enumerated == ["tos"], f"enumeration changed: {enumerated}"
        assert set(readable) - set(enumerated) == {
            "lat_bnds",
            "lon_bnds",
            "time_bnds",
        }, f"readable set changed: {readable}"

    def test_an_overridden_variable_names_still_decides_what_is_readable(self):
        """Sharing the walk must not bypass the public property.

        Test scenario:
            `variable_names` is public and overridable, and the plot suite's
            curvilinear fixtures splice `XLONG` / `XLAT` in by overriding it on
            a per-instance subclass -- `get_variable` then has to accept those
            names. Handing the walk to `_get_variable_names` unconditionally
            went round the override, so the spliced names vanished from the
            readable set and every `coords=` lookup refused them. The saving is
            therefore taken only when the property is the inherited one.
        """
        dataset = NetCDF.read_file(str(FLAT))
        spliced = [*dataset.variable_names, "XLONG"]
        dataset.__class__ = type(
            "SplicedNetCDF",
            (type(dataset),),
            {"variable_names": property(lambda _self: spliced)},
        )

        readable = dataset._readable_variable_names()

        assert "XLONG" in readable, f"the override was bypassed: {readable}"
        assert "lat_bnds" in readable, (
            f"the store's own arrays must still be added: {readable}"
        )

    def test_the_answer_is_recomputed_rather_than_remembered(self):
        """Nothing is memoised, so nothing has to be invalidated.

        Test scenario:
            The obvious next optimisation is a cache, and the hard part of one
            is invalidation: `add_variable` / `set_variable` / `rename_variable`
            / `remove_variable` all change the answer, and so does
            `_replace_raster` under every container-wide operation. This asserts
            the property a cache would have to preserve -- a rename is visible
            on the very next call -- so a memo added without invalidation fails
            here rather than silently serving a stale list.
        """
        dataset = NetCDF.read_file(str(FLAT))
        before = dataset._readable_variable_names()
        assert "lat_bnds" in before, "fixture no longer carries the bounds array"

        dataset.rename_variable("lat_bnds", "lat_bounds")

        after = dataset._readable_variable_names()
        assert "lat_bounds" in after, f"the rename is not visible: {after}"
        assert "lat_bnds" not in after, f"the old name is still served: {after}"


class TestAnAdvertisedNameThatIsNotARaster:
    """`get_variable` accepts these names; it does not return a raster for them."""

    @pytest.mark.parametrize(
        "array", ["time_bounds", "x_image_bounds", "y_image_bounds", "band_id"]
    )
    def test_the_name_resolves_to_a_raw_md_array(self, array):
        """The documented return type, asserted rather than assumed.

        Args:
            array: A 1-D array of the GOES fixture.

        Test scenario:
            Three of these four are in `variable_names`, so a caller iterating
            the enumeration meets them. `get_variable` resolves each -- it does
            not raise -- and hands back `gdal.MDArray`, which carries none of
            the `NetCDF` surface.
        """
        dataset = NetCDF.read_file(str(GEOS))

        variable = dataset.get_variable(array)

        assert isinstance(variable, gdal.MDArray), (
            f"{array} came back as {type(variable).__name__}"
        )

    def test_read_array_reads_such_a_variable_instead_of_raising(self):
        """The consequence a user actually hits.

        Test scenario:
            `time_bounds` is in `variable_names`, so `nc.read_array(
            variable="time_bounds")` is an ordinary call on an advertised name.
            It answered with `AttributeError: 'MDArray' object has no attribute
            'read_array'` -- an internal error, from the container's own list.
            It now returns the values, checked against the raw GDAL read.
        """
        dataset = NetCDF.read_file(str(GEOS))
        assert "time_bounds" in dataset.variable_names, "fixture changed"
        expected = dataset._working_group().OpenMDArray("time_bounds").ReadAsArray()

        values = dataset.read_array(variable="time_bounds")

        assert np.array_equal(values, expected), f"{values} != {expected}"

    def test_read_array_reads_a_group_qualified_series(self):
        """The same route, reached from the grouped store.

        Test scenario:
            `flight/air_press` is 1-D, so the group walk enumerates it and the
            same `AttributeError` came back. The name has to be resolved
            through its owning group before it can be read at all.
        """
        dataset = NetCDF.read_file(str(GROUPED))
        qualified = next(n for n in dataset.variable_names if "/" in n)
        group, _, leaf = qualified.rpartition("/")
        expected = (
            dataset._working_group().OpenGroup(group).OpenMDArray(leaf).ReadAsArray()
        )

        values = dataset.read_array(variable=qualified)

        assert np.array_equal(values, expected), f"{values} != {expected}"

    def test_a_raster_only_argument_is_refused_with_an_explanation(self):
        """Silently ignoring `window=` would be worse than refusing it.

        Test scenario:
            A raw array has no band, window or chunk plane. Answering a
            windowed request with the whole array would hand back far more
            data than asked for, so the call is refused with the reason.
        """
        dataset = NetCDF.read_file(str(GEOS))

        with pytest.raises(ValueError) as excinfo:
            dataset.read_array(variable="time_bounds", window=[0, 0, 1, 1])

        message = str(excinfo.value)
        assert "time_bounds" in message, f"the refusal must name the array: {message}"
        assert "window" in message, f"the refusal must name the argument: {message}"
