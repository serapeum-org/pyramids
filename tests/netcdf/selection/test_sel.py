"""Tests for NetCDF.sel() — time/dimension subsetting.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


def _make_nc():
    """Create a 3D NetCDF with known time coordinates [0, 6, 12, 18, 24].

    Carries a -9999.0 no-data sentinel so preservation tests can pin the
    literal (the data values 0-59 never collide with it).
    """
    arr = np.arange(60, dtype=np.float64).reshape(5, 3, 4)
    geo = (0.0, 1.0, 0, 3.0, 0, -1.0)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        variable_name="temp",
        no_data_value=-9999.0,
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12, 18, 24],
    )
    return nc


class TestSelSingleValue:
    """sel(dim=value) selects one band by exact coordinate value."""

    def test_returns_single_band(self):
        """Selecting one time step should return a 1-band dataset.

        Test scenario:
            sel(time=6) → shape (1, 3, 4), coord [6.0].
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=6)
        assert result.shape == (1, 3, 4), f"Expected (1, 3, 4), got {result.shape}"
        read = result.read_array()
        assert read.shape == (
            3,
            4,
        ), f"single-band read should squeeze to 2D, got {read.shape}"

    def test_data_matches_original_band(self):
        """Selected data should match the corresponding band in the original.

        Test scenario:
            sel(time=12) data == original band index 2.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        expected = var.read_array()[2]
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel data should match original band",
        )

    def test_coord_values_preserved(self):
        """The selected result should carry the matching coordinate value.

        Test scenario:
            sel(time=18) → _band_dim_values == [18.0].
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=18)
        assert result._band_dim_values == [
            18.0
        ], f"Expected [18.0], got {result._band_dim_values}"


class TestSelList:
    """sel(dim=[v1, v2, ...]) selects multiple bands by value."""

    def test_selects_multiple_bands(self):
        """Selecting [0, 12, 24] should return 3 bands.

        Test scenario:
            sel(time=[0, 12, 24]) → shape (3, 3, 4).
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[0, 12, 24])
        assert result.shape == (3, 3, 4), f"Expected (3, 3, 4), got {result.shape}"

    def test_data_matches_original_bands(self):
        """Selected bands should match original bands at those indices.

        Test scenario:
            sel(time=[0, 24]) data == original bands 0 and 4.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[0, 24])
        orig = var.read_array()
        expected = orig[[0, 4]]
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel list data should match original bands",
        )

    def test_coord_values_preserved(self):
        """Selected coordinates should match the requested values.

        Test scenario:
            sel(time=[6, 18]) → _band_dim_values == [6.0, 18.0].
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[6, 18])
        assert result._band_dim_values == [
            6.0,
            18.0,
        ], f"Expected [6.0, 18.0], got {result._band_dim_values}"


class TestSelSlice:
    """sel(dim=slice(start, stop)) selects bands in a range."""

    def test_selects_range(self):
        """slice(6, 18) should select time steps 6, 12, 18.

        Test scenario:
            3 bands where 6 <= coord <= 18.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(6, 18))
        assert result.shape == (3, 3, 4), f"Expected (3, 3, 4), got {result.shape}"
        assert result._band_dim_values == [
            6.0,
            12.0,
            18.0,
        ], f"Expected [6, 12, 18], got {result._band_dim_values}"
        expected = np.stack([var.read_array(band=i) for i in (1, 2, 3)], axis=0)
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="slice(6, 18) data should equal stacked bands 1, 2, 3",
        )

    def test_open_start(self):
        """slice(None, 12) should select from beginning up to 12.

        Test scenario:
            Bands where coord <= 12 → time steps 0, 6, 12.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(None, 12))
        assert result._band_dim_values == [
            0.0,
            6.0,
            12.0,
        ], f"Expected [0, 6, 12], got {result._band_dim_values}"

    def test_open_end(self):
        """slice(18, None) should select from 18 to the end.

        Test scenario:
            Bands where coord >= 18 → time steps 18, 24.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(18, None))
        assert result._band_dim_values == [
            18.0,
            24.0,
        ], f"Expected [18, 24], got {result._band_dim_values}"

    def test_descending_coord_open_slice_selects_all(self):
        """`slice(None, None)` on a descending-coord dim selects every band.

        Test scenario:
            CDS-Beta `latitude` is stored north-to-south, so coords
            descend (e.g. `[44, 43, 42, 41, 40]`). Pre-fix the default
            fill `start=44, stop=40` produced the test `44 <= v <= 40`
            which matched nothing. Bounds are now normalised before
            the match, so the slice is direction-agnostic.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        # Manually flip the band-dim coords to mimic a descending axis;
        # the underlying band data stays in storage order so the test
        # only exercises the slice-bound logic.
        var._band_dim_values = list(reversed(var._band_dim_values))
        var._band_dim_values_map = dict(var._band_dim_values_map)
        var._band_dim_values_map["time"] = list(var._band_dim_values)

        result = var.sel(time=slice(None, None))
        assert sorted(result._band_dim_values) == [0.0, 6.0, 12.0, 18.0, 24.0], (
            f"open-ended slice on descending coords should select every "
            f"band, got {result._band_dim_values}"
        )

    def test_descending_coord_explicit_inverted_bounds(self):
        """`slice(high, low)` on descending coords matches the inclusive range.

        Test scenario:
            With `coords = [24, 18, 12, 6, 0]` (descending), the user
            writes `slice(18, 6)` (high → low) — the natural reading
            order. The match must include 18, 12, 6.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        var._band_dim_values = list(reversed(var._band_dim_values))
        var._band_dim_values_map = dict(var._band_dim_values_map)
        var._band_dim_values_map["time"] = list(var._band_dim_values)

        result = var.sel(time=slice(18, 6))
        assert sorted(result._band_dim_values) == [6.0, 12.0, 18.0], (
            f"slice(18, 6) on descending coords should select [6, 12, 18], "
            f"got {result._band_dim_values}"
        )

    def test_equal_bounds_selects_single_band(self):
        """slice(12, 12) hits the inclusive `<=` boundary on both sides.

        Test scenario:
            On ascending coords [0, 6, 12, 18, 24], `slice(12, 12)`
            should select exactly the band at coord 12. Locks in that
            the bound check is `<=` (not `<`) on both sides.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(12, 12))
        assert result._band_dim_values == [12.0], (
            f"slice(12, 12) should select single band [12], got "
            f"{result._band_dim_values!r}"
        )

    def test_ascending_with_inverted_bounds(self):
        """slice(18, 6) on ASCENDING coords gives the same result as on descending.

        Test scenario:
            User mistakenly writes `slice(high, low)` on ascending
            coords. Bound normalisation makes the slice
            direction-agnostic, so the result is the same set of
            bands as `slice(6, 18)`. Locks in the symmetry between
            ascending and descending paths.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(18, 6))
        assert result._band_dim_values == [6.0, 12.0, 18.0], (
            f"slice(18, 6) on ascending should match slice(6, 18), got "
            f"{result._band_dim_values!r}"
        )

    def test_descending_no_match_still_raises(self):
        """A genuinely empty slice on descending coords still raises ValueError.

        Test scenario:
            On descending coords [24, 18, 12, 6, 0], `slice(100, 200)`
            normalises to `(100, 200)` and matches no coord (all
            below 100). Confirms the existing "No bands match" guard
            still fires after normalisation — the swap doesn't mask
            no-match errors.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        var._band_dim_values = list(reversed(var._band_dim_values))
        var._band_dim_values_map = dict(var._band_dim_values_map)
        var._band_dim_values_map["time"] = list(var._band_dim_values)

        time_slice = slice(100, 200)
        with pytest.raises(ValueError, match="No bands match"):
            var.sel(time=time_slice)


class TestSelErrors:
    """sel() error handling."""

    def test_wrong_dim_name_raises(self):
        """Passing a dimension name that doesn't match should raise.

        Test scenario:
            sel(level=5) on a variable with band_dim_name="time".
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        with pytest.raises(ValueError, match="does not match"):
            var.sel(level=5)

    def test_no_match_raises(self):
        """Requesting a value that doesn't exist should raise.

        Test scenario:
            sel(time=999) when values are [0, 6, 12, 18, 24].
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        with pytest.raises(ValueError, match="No bands match"):
            var.sel(time=999)

    def test_2d_variable_raises(self):
        """sel() on a 2D variable (no band dim) should raise.

        Test scenario:
            A 2D variable has _band_dim_name=None.
        """
        arr = np.ones((5, 8), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="flat",
        )
        var = nc.get_variable("flat")
        with pytest.raises(ValueError, match="no band dimension"):
            var.sel(time=0)

    def test_multiple_kwargs_raises(self):
        """sel() with more than one keyword should raise.

        Test scenario:
            sel(time=0, level=1) → ValueError.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        with pytest.raises(ValueError, match="exactly one"):
            var.sel(time=0, level=1)


class TestSelBoundary:
    """Boundary and edge cases for sel()."""

    def test_select_first_element(self):
        """Selecting the first coordinate value should work.

        Test scenario:
            sel(time=0) → first band of the variable.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=0)
        expected = var.read_array()[0]
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel(first) data mismatch",
        )

    def test_select_last_element(self):
        """Selecting the last coordinate value should work.

        Test scenario:
            sel(time=24) → last band of the variable.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=24)
        expected = var.read_array()[4]
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel(last) data mismatch",
        )

    def test_select_all_elements_via_list(self):
        """Selecting all values as a list returns the full array.

        Test scenario:
            sel(time=[0, 6, 12, 18, 24]) → same shape and data as original.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[0, 6, 12, 24, 18])
        assert result.band_count == 5, f"Expected 5 bands, got {result.band_count}"

    def test_select_all_elements_via_slice(self):
        """slice(None, None) should select everything.

        Test scenario:
            Open start + open end → all bands.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=slice(None, None))
        assert result.band_count == 5, f"Expected 5 bands, got {result.band_count}"

    def test_float_coordinate_matching(self):
        """sel() should match float coordinates exactly.

        Test scenario:
            Create with float coords [0.5, 1.5, 2.5], select 1.5.
        """
        arr = np.arange(36, dtype=np.float64).reshape(3, 3, 4)
        geo = (0.0, 1.0, 0, 3.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
            extra_dim_name="level",
            extra_dim_values=[0.5, 1.5, 2.5],
        )
        var = nc.get_variable("v")
        result = var.sel(level=1.5)
        assert result._band_dim_values == [
            1.5
        ], f"Expected [1.5], got {result._band_dim_values}"
        expected = var.read_array()[1]
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="Float coord sel data mismatch",
        )

    def test_empty_list_raises(self):
        """An empty list selector should raise (no bands match).

        Test scenario:
            sel(time=[]) → ValueError.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        with pytest.raises(ValueError, match="No bands match"):
            var.sel(time=[])

    def test_no_kwargs_raises(self):
        """sel() with zero arguments should raise.

        Test scenario:
            var.sel() → ValueError.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        with pytest.raises(ValueError, match="exactly one"):
            var.sel()

    def test_slice_no_match_raises(self):
        """A slice range outside all coordinates should raise.

        Test scenario:
            sel(time=slice(100, 200)) when max coord is 24.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        time_slice = slice(100, 200)
        with pytest.raises(ValueError, match="No bands match"):
            var.sel(time=time_slice)


class TestSelPreservation:
    """sel() should preserve metadata and spatial properties."""

    def test_geotransform_preserved(self):
        """Geotransform should be identical after sel.

        Test scenario:
            The spatial reference doesn't change with band selection.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        assert (
            result.geotransform == var.geotransform
        ), f"Geotransform changed: {var.geotransform} → {result.geotransform}"

    def test_epsg_preserved(self):
        """EPSG should be identical after sel.

        Test scenario:
            CRS doesn't change with band selection.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        assert result.epsg == var.epsg, f"EPSG changed: {var.epsg} → {result.epsg}"

    def test_band_dim_name_preserved(self):
        """_band_dim_name should carry through to the result.

        Test scenario:
            The dimension name 'time' should appear on the result.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[6, 18])
        assert (
            result._band_dim_name == "time"
        ), f"Expected 'time', got {result._band_dim_name}"

    def test_variable_attrs_preserved(self):
        """_variable_attrs should carry through to the result.

        Test scenario:
            Attributes set on the variable should appear after sel.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        var._variable_attrs = {"units": "K", "long_name": "Temperature"}
        result = var.sel(time=12)
        assert result._variable_attrs == {
            "units": "K",
            "long_name": "Temperature",
        }, f"Attrs not preserved: {result._variable_attrs}"

    def test_nodata_preserved(self):
        """No-data value should carry through to the result.

        Test scenario:
            The no-data sentinel should be the same after sel.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=[0, 6])
        orig_ndv = var.no_data_value[0]
        result_ndv = result.no_data_value[0]
        assert orig_ndv == -9999.0, f"fixture nodata should be -9999.0, got {orig_ndv}"
        assert result_ndv == orig_ndv, f"No-data changed: {orig_ndv} → {result_ndv}"


class TestSelReturnsNetCDF:
    """sel() returns a NetCDF with variable metadata preserved."""

    def test_result_is_netcdf(self):
        """sel() should return a NetCDF, not a plain Dataset.

        Test scenario:
            The result must be a NetCDF so that chaining sel() and
            calling read_array(unpack=True) remain available.
        """
        from pyramids.netcdf import NetCDF

        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_result_is_also_dataset(self):
        """sel() result is still a Dataset (via inheritance)."""
        from pyramids.dataset import Dataset

        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        assert isinstance(
            result, Dataset
        ), f"Expected Dataset, got {type(result).__name__}"


class TestSelRoundTrip:
    """sel() result can be written back via set_variable."""

    def test_sel_then_set_variable(self):
        """sel() result should be writable back to the container.

        Test scenario:
            sel → set_variable → verify variable exists with correct shape.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        subset = var.sel(time=[6, 18])
        nc.set_variable("temp_subset", subset)
        assert (
            "temp_subset" in nc.variable_names
        ), f"Expected 'temp_subset' in {nc.variable_names}"
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("temp_subset")
        assert list(md_arr.GetShape()) == [
            2,
            3,
            4,
        ], f"Expected shape [2, 3, 4], got {list(md_arr.GetShape())}"

    def test_sel_set_variable_data_integrity(self):
        """Data written back after sel should be identical when re-read.

        Test scenario:
            sel → set_variable → get_variable → read_array → compare.
        """
        nc = _make_nc()
        var = nc.get_variable("temp")
        subset = var.sel(time=[6, 12])
        expected = subset.read_array()
        nc.set_variable("temp_sub", subset)
        rg = nc._raster.GetRootGroup()
        stored = rg.OpenMDArray("temp_sub").ReadAsArray()
        assert_array_equal(
            stored,
            expected,
            err_msg="Round-trip data mismatch after sel → set_variable",
        )


class TestSelThreeDimRegression:
    """3-D files keep the pre-feature contract after #311 lifted the
    single-band-dim restriction.

    Locks in: legacy ``_band_dim_name`` / ``_band_dim_values`` populated,
    new ``_band_dim_names`` is a 1-tuple, ``_band_dim_sizes`` is a 1-tuple,
    and ``_band_dim_values_map`` carries the same primary list.
    """

    def test_legacy_fields_populated_for_3d(self):
        """3-D variable still exposes legacy single-dim fields unchanged."""
        nc = _make_nc()
        var = nc.get_variable("temp")
        assert (
            var._band_dim_name == "time"
        ), f"primary dim must be 'time', got {var._band_dim_name!r}"
        assert var._band_dim_values == [
            0,
            6,
            12,
            18,
            24,
        ], f"values mismatch: {var._band_dim_values!r}"

    def test_new_fields_are_singleton_for_3d(self):
        """The new multi-dim fields collapse to one-entry containers on a 3-D file."""
        nc = _make_nc()
        var = nc.get_variable("temp")
        assert var._band_dim_names == (
            "time",
        ), f"_band_dim_names must be a 1-tuple, got {var._band_dim_names!r}"
        assert var._band_dim_sizes == (
            5,
        ), f"_band_dim_sizes must be (5,), got {var._band_dim_sizes!r}"
        assert var._band_dim_values_map == {
            "time": [0, 6, 12, 18, 24]
        }, f"_band_dim_values_map mismatch: {var._band_dim_values_map!r}"

    def test_sel_after_pin_keeps_singleton_layout(self):
        """``sel(time=12)`` on a 3-D var still produces a single-band-dim result."""
        nc = _make_nc()
        var = nc.get_variable("temp")
        result = var.sel(time=12)
        assert result._band_dim_names == (
            "time",
        ), f"singleton layout broken: {result._band_dim_names!r}"
        assert result._band_dim_sizes == (
            1,
        ), f"size after sel must be (1,), got {result._band_dim_sizes!r}"
        assert result._band_dim_values_map == {
            "time": [12]
        }, f"values_map after sel mismatch: {result._band_dim_values_map!r}"
