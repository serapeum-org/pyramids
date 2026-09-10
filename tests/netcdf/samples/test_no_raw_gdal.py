"""No public accessor hands back a raw GDAL object.

`get_variable` used to return the `gdal.MDArray` itself whenever GDAL could not
expose the variable as a raster plane -- a 1-D array, or a string/compound one of
any rank. Such an object has none of pyramids' API, and its `Read()` returns an
undecoded buffer for numeric data, so the only way to the values was
`ReadAsArray()`: raw `osgeo`, in a package that exists to avoid it (#1126).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import LabeledArray, NetCDF

GROUPED = "none__35v__1d35__groups-nc4.nc"


class TestANonRasterVariableComesBackLabelled:
    """The shapes GDAL cannot expose as a raster still answer in pyramids' terms."""

    @pytest.mark.parametrize(
        "file_name, variable, dims, first",
        [
            (GROUPED, "UTC_time", ("recNum",), "2012-03-04 03:54:19"),
            ("cf__48v__1d17-3d21-4d10__y-asc.nc", "hyai", ("ilev",), None),
        ],
    )
    def test_it_is_a_labeled_array_not_an_mdarray(
        self, sample, file_name, variable, dims, first
    ):
        """A 1-D variable answers with values, dims and shape.

        Args:
            sample: Fixture resolving a sample file name to its path.
            file_name: The sample store.
            variable: A 1-D variable in it.
            dims: The dimension names GDAL declares.
            first: The expected first value, when it is worth pinning.

        Test scenario:
            A numeric axis (`hyai`, a hybrid-sigma coefficient) and a string
            column (`UTC_time`) reach the same branch. The string one is the
            sharper case: `ReadAsArray` raises on it, so a wrapper that knew only
            the numeric reader would fail here while passing on `hyai`.
        """
        store = NetCDF.read_file(sample(file_name), read_only=True)
        try:
            variable_object = store.get_variable(variable)

            assert isinstance(variable_object, LabeledArray)
            assert variable_object.dims == dims
            assert variable_object.shape == variable_object.values.shape
            if first is not None:
                assert variable_object.values[0] == first
        finally:
            store.close()

    @pytest.mark.parametrize("route", ["group", "slash"])
    def test_both_routes_into_a_group_agree(self, sample, route):
        """A grouped store leaks through two doors, and both go through one branch.

        Args:
            sample: Fixture resolving a sample file name to its path.
            route: `get_group(...).get_variable(name)`, or `get_variable("group/name")`.

        Test scenario:
            The slash form delegates to the group's `get_variable`, so a guard
            placed only where `read_file` lands would catch neither route.
        """
        store = NetCDF.read_file(sample(GROUPED), read_only=True)
        try:
            group_name = store.group_names[0]
            if route == "group":
                variable_object = store.get_group(group_name).get_variable("air_press")
            else:
                variable_object = store.get_variable(f"{group_name}/air_press")

            assert isinstance(variable_object, LabeledArray)
            assert variable_object.dims == ("recNum",)
            assert variable_object.values.dtype == np.float64
        finally:
            store.close()


class TestNoOsgeoTypeEscapes:
    """The property the branch exists to hold, asserted over every sample file."""

    def test_no_readable_variable_returns_a_gdal_object(self, sample_name, sample):
        """Every readable variable answers as a pyramids type, and none of them raises.

        Args:
            sample_name: The sample file, parametrized over the whole registry.
            sample: Fixture resolving a sample file name to its path.

        Test scenario:
            Asserted as "nothing from `osgeo`" rather than "this one call
            raises", because that branch is the only place such an object can
            escape. Refusals are collected rather than skipped: swallowing them
            would let an implementation that raised for every non-raster
            variable satisfy the sweep, which is the shape the issue rejected.
            The readable set, not `variable_names`, is what `get_variable`
            accepts -- on the GOES fixture it is what brings in `band_id` and
            `band_wavelength`, both 1-D and neither enumerated as a data
            variable.
        """
        store = NetCDF.read_file(sample(sample_name), read_only=True)
        try:
            leaked = []
            refused = []
            for name in store._readable_variable_names():
                try:
                    variable_object = store.get_variable(name)
                except Exception as error:
                    refused.append(f"{name} -> {type(error).__name__}: {error}")
                    continue
                if type(variable_object).__module__.startswith("osgeo"):
                    leaked.append(f"{name} -> {type(variable_object).__name__}")

            assert not leaked, f"{sample_name}: raw GDAL objects returned for {leaked}"
            assert not refused, f"{sample_name}: readable variables refused: {refused}"
        finally:
            store.close()

    @pytest.mark.parametrize(
        "file_name, variable",
        [(GROUPED, "UTC_time"), ("cf__9v__1d7-2d2__geos__y-desc.nc", "band_id")],
    )
    def test_the_sweep_has_something_to_find(self, sample, file_name, variable):
        """The sweep above asserts an absence; this pins that the presence exists.

        Args:
            sample: Fixture resolving a sample file name to its path.
            file_name: A store known to hold a non-raster variable.
            variable: That variable.

        Test scenario:
            `assert not leaked` is vacuously true on a store whose variables are
            all rasters, so the sweep alone cannot show a `LabeledArray` was ever
            produced. `band_id` is the sharper of the two: it is readable but not
            enumerated in `variable_names`, so only the readable set reaches it.
        """
        store = NetCDF.read_file(sample(file_name), read_only=True)
        try:
            assert isinstance(store.get_variable(variable), LabeledArray)
        finally:
            store.close()


class TestAVariableWithNoRecords:
    """A zero-length dimension is a shape GDAL declines to read at all."""

    @pytest.mark.interop
    def test_an_empty_variable_reads_as_an_empty_array(self, tmp_path):
        """An unlimited dimension with nothing written is still an advertised variable.

        Args:
            tmp_path: Destination for the store.

        Test scenario:
            GDAL refuses the read -- `count[0] = 0 is invalid` -- rather than
            handing back nothing, so materialising naively turned a variable the
            store advertises into a bare `RuntimeError`. Before the wrapper it
            returned an unread `MDArray`, so this shape has never actually been
            readable; it now answers with an empty array of the declared type.
            Written through xarray because neither GDAL driver will create one:
            MEM rejects the array, and the netCDF writer emits a file it cannot
            reopen.
        """
        xr = pytest.importorskip("xarray")
        path = str(tmp_path / "empty_dim.nc")
        xr.Dataset(
            {
                "empty_v": ("recNum", np.array([], dtype="float64")),
                "full_v": ("n", np.array([1.0, 2.0, 3.0])),
            }
        ).to_netcdf(path)

        store = NetCDF.read_file(path, read_only=True)
        try:
            assert "empty_v" in store.variable_names, "fixture changed"

            empty = store.get_variable("empty_v")

            assert isinstance(empty, LabeledArray)
            assert empty.shape == (0,)
            assert empty.values.shape == (0,)
            assert empty.values.dtype == np.float64
            assert store.get_variable("full_v").values.tolist() == [1.0, 2.0, 3.0]
        finally:
            store.close()
