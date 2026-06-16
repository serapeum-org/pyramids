"""Unit tests for pyramids.netcdf.NetCDF.reduce (dimension reduction)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_GEO = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
_ERA5_T2M = "tests/data/netcdf/era5_cds_beta_t2m_jan2022.nc"
_ERA5_PL = "tests/data/netcdf/era5_cds_beta_t_pressure_levels_jan2022.nc"


def _make_time_nc(arr: np.ndarray, time_values: list) -> NetCDF:
    """Build an in-memory `(time, y, x)` NetCDF with one variable `v`.

    Args:
        arr: 3-D `(time, rows, cols)` array.
        time_values: Coordinate values for the time dimension.

    Returns:
        NetCDF: A MEM container with a single variable named `v`.
    """
    return NetCDF.create_from_array(
        arr,
        geo=_GEO,
        epsg=4326,
        variable_name="v",
        extra_dim_name="time",
        extra_dim_values=time_values,
    )


class TestReduceCollapse:
    """Tests for full-dimension collapse (groupby=None)."""

    @pytest.mark.parametrize(
        "how, np_func",
        [
            ("mean", np.mean),
            ("sum", np.sum),
            ("min", np.min),
            ("max", np.max),
            ("std", np.std),
            ("var", np.var),
        ],
    )
    def test_collapse_matches_numpy(self, how, np_func):
        """Collapsing the time axis matches the equivalent NumPy reduction.

        Args:
            how: Reduction op name passed to `reduce`.
            np_func: The NumPy function to compare against.

        Test scenario:
            A 4x3x5 cube reduced over `time` equals `np_func(arr, axis=0)`.
        """
        arr = np.arange(4 * 3 * 5, dtype="float32").reshape(4, 3, 5)
        result = _make_time_nc(arr, [0, 1, 2, 3]).reduce("time", how)
        out = result.get_variable("v").read_array()
        assert np.allclose(out, np_func(arr, axis=0)), f"{how} collapse mismatch"

    def test_collapse_removes_dimension(self):
        """Collapse removes the reduced dimension from the variable.

        Test scenario:
            After `reduce('time')` the variable has no band dims and is 2-D.
        """
        arr = np.ones((4, 3, 5), dtype="float32")
        var = _make_time_nc(arr, [0, 1, 2, 3]).reduce("time", "mean").get_variable("v")
        assert var._band_dim_names == (), f"time should be gone: {var._band_dim_names}"
        assert var.read_array().shape == (3, 5), "result should be 2-D"


class TestReduceWindowed:
    """Tests for windowed reduction via explicit labels."""

    def test_label_groups_reduce_per_window(self):
        """Equal labels are reduced together, in first-appearance order.

        Test scenario:
            Labels `[0, 0, 1, 1]` produce two windows: sum of the first two
            and last two time slices.
        """
        arr = np.arange(4 * 2 * 2, dtype="float32").reshape(4, 2, 2)
        result = _make_time_nc(arr, [10, 20, 30, 40]).reduce(
            "time", "sum", groupby=[0, 0, 1, 1]
        )
        out = result.get_variable("v").read_array()
        assert out.shape == (2, 2, 2), f"expected 2 windows, got {out.shape}"
        assert np.allclose(out[0], arr[0:2].sum(0)), "window 0 wrong"
        assert np.allclose(out[1], arr[2:4].sum(0)), "window 1 wrong"

    def test_window_coordinate_is_first_member(self):
        """A coarsened dim is labelled with each window's first source coord.

        Test scenario:
            Coords `[10, 20, 30, 40]` grouped `[0, 0, 1, 1]` give `[10, 30]`.
        """
        arr = np.ones((4, 2, 2), dtype="float32")
        result = _make_time_nc(arr, [10, 20, 30, 40]).reduce(
            "time", "mean", groupby=[0, 0, 1, 1]
        )
        coord = result.get_variable("v")._band_dim_values
        assert list(coord) == [10.0, 30.0], f"unexpected window coords: {coord}"

    def test_coverage_mismatch_raises(self):
        """Labels not covering the dimension exactly raise ValueError.

        Test scenario:
            3 labels for a size-4 time axis is rejected.
        """
        arr = np.ones((4, 2, 2), dtype="float32")
        with pytest.raises(ValueError, match="covers"):
            _make_time_nc(arr, [0, 1, 2, 3]).reduce("time", "mean", groupby=[0, 0, 1])


class TestReduceSkipna:
    """Tests for NoData handling under skipna."""

    def test_skipna_ignores_nodata(self):
        """skipna masks the NoData value before averaging.

        Test scenario:
            With nodata in one of two time slices, the skipna mean equals the
            single valid slice; without skipna the nodata corrupts the mean.
        """
        arr = np.array(
            [[[10.0, 10.0]], [[-9999.0, 30.0]]], dtype="float32"
        )  # (time=2, y=1, x=2)
        nc = NetCDF.create_from_array(
            arr,
            geo=_GEO,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="v",
            extra_dim_name="time",
            extra_dim_values=[0, 1],
        )
        skip = nc.reduce("time", "mean", skipna=True).get_variable("v").read_array()
        assert skip[0, 0] == pytest.approx(10.0), "all-but-nodata cell should keep the valid value"
        assert skip[0, 1] == pytest.approx(20.0), "mean of 10 and 30 should be 20"

    def test_no_skipna_uses_raw_values(self):
        """Without skipna the raw values (including sentinels) are reduced.

        Test scenario:
            `skipna=False` sum includes the -9999 sentinel verbatim.
        """
        arr = np.array([[[10.0]], [[-9999.0]]], dtype="float32")
        nc = NetCDF.create_from_array(
            arr,
            geo=_GEO,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="v",
            extra_dim_name="time",
            extra_dim_values=[0, 1],
        )
        out = nc.reduce("time", "sum", skipna=False).get_variable("v").read_array()
        assert out[0, 0] == pytest.approx(-9989.0), "raw sum should include sentinel"

    @pytest.mark.parametrize("how", ["sum", "mean", "std", "var", "min", "max"])
    def test_all_nodata_slice_stays_nodata(self, how):
        """A fully-masked cell reduces to NoData, not a spurious 0.

        Args:
            how: The reducer under test.

        Test scenario:
            A cell that is NoData across every time step must remain NoData after
            `skipna` reduction. `nansum`/`nanstd`/`nanvar` return 0 for an
            all-NaN slice, so without an explicit guard `sum` would leak a 0.
        """
        arr = np.array(
            [[[1.0, -9999.0]], [[2.0, -9999.0]]], dtype="float32"
        )  # (time=2, y=1, x=2); column 1 is all-NoData
        nc = NetCDF.create_from_array(
            arr,
            geo=_GEO,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="v",
            extra_dim_name="time",
            extra_dim_values=[0, 1],
        )
        out = nc.reduce("time", how, skipna=True).get_variable("v").read_array()
        assert (
            out[0, 1] == -9999.0
        ), f"{how}: all-nodata cell should stay nodata, got {out[0, 1]}"
        assert out[0, 0] != -9999.0, f"{how}: valid cell should compute a real value"


class TestReducePassthrough:
    """Tests that variables lacking the reduced dim are preserved."""

    def test_static_variable_preserved(self):
        """A variable without the reduced dim survives unchanged.

        Test scenario:
            A container with a `(time, y, x)` `dynamic` and a `(y, x)`
            `static` variable keeps `static` after reducing `time`.
        """
        dyn = np.arange(4 * 3 * 5, dtype="float32").reshape(4, 3, 5)
        nc = _make_time_nc(dyn, [0, 1, 2, 3])
        static_arr = np.full((3, 5), 7.0, dtype="float32")
        static_ds = Dataset.create_from_array(static_arr, geo=_GEO, epsg=4326)
        nc.set_variable("static", static_ds)

        result = nc.reduce("time", "mean")
        assert "static" in result.variable_names, "static variable was dropped"
        assert np.allclose(
            result.get_variable("static").read_array(), static_arr
        ), "static variable should be unchanged"


class TestReduceErrors:
    """Tests for reduce's validation paths."""

    def test_unknown_how_raises(self):
        """An unknown reduction op raises ValueError naming the valid set.

        Test scenario:
            `how='median'` is rejected.
        """
        arr = np.ones((2, 2, 2), dtype="float32")
        with pytest.raises(ValueError, match="how must be one of"):
            _make_time_nc(arr, [0, 1]).reduce("time", "median")

    def test_unknown_dimension_raises(self):
        """Reducing a non-existent dimension raises ValueError.

        Test scenario:
            `reduce('depth')` on a time-only cube is rejected.
        """
        arr = np.ones((2, 2, 2), dtype="float32")
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            _make_time_nc(arr, [0, 1]).reduce("depth", "mean")

    def test_frequency_without_time_coord_raises(self):
        """A frequency groupby on a dim with no time coordinate raises.

        Test scenario:
            The in-memory cube's `time` dim has no CF units, so a `"1MS"`
            grouping cannot decode it.
        """
        arr = np.ones((2, 2, 2), dtype="float32")
        with pytest.raises(ValueError, match="no decodable time coordinate"):
            _make_time_nc(arr, [0, 1]).reduce("time", "mean", groupby="1MS")


class TestReduceRealFixtures:
    """Tests against real CF NetCDF fixtures."""

    def test_monthly_frequency_on_era5(self):
        """Monthly grouping of a single-month ERA5 file yields one window.

        Test scenario:
            `era5_cds_beta_t2m_jan2022` (12 sub-daily steps in Jan 2022)
            reduced by `"1MS"` over `valid_time` gives a single band.
        """
        nc = NetCDF.read_file(_ERA5_T2M)
        result = nc.reduce("valid_time", "mean", groupby="1MS")
        assert result.get_variable("t2m").band_count == 1, "Jan-only data is one month"

    def test_daily_frequency_on_era5(self):
        """Daily grouping yields one window per distinct calendar day.

        Test scenario:
            The 12 steps span 3 days, so `"1D"` produces 3 bands.
        """
        nc = NetCDF.read_file(_ERA5_T2M)
        result = nc.reduce("valid_time", "mean", groupby="1D")
        assert result.get_variable("t2m").band_count == 3, "data spans three days"

    def test_collapse_pressure_level_on_4d(self):
        """Collapsing a non-time dim on a 4-D file removes only that dim.

        Test scenario:
            `era5_..._pressure_levels` `t` is `(valid_time, pressure_level,
            lat, lon)`; collapsing `pressure_level` leaves `(valid_time,)`.
        """
        nc = NetCDF.read_file(_ERA5_PL)
        var = nc.reduce("pressure_level", "mean").get_variable("t")
        assert "pressure_level" not in var._band_dim_names, "pressure_level not removed"
        assert "valid_time" in var._band_dim_names, "valid_time should be preserved"

    @pytest.mark.parametrize("freq, expected", [("6h", 12), ("12h", 6), ("1D", 3)])
    def test_sub_daily_frequency_grouping(self, freq, expected):
        """Sub-daily frequencies bucket by full timestamp, not truncated day.

        Args:
            freq: pandas offset alias passed to `groupby`.
            expected: Expected number of output windows.

        Test scenario:
            `era5_cds_beta_t2m_jan2022` holds 12 six-hourly steps over three
            days, so `"6h"` yields 12 windows, `"12h"` 6, and `"1D"` 3.
            Before the timestamp-resolution fix every sub-daily frequency
            collapsed to the 3 per-day buckets.
        """
        nc = NetCDF.read_file(_ERA5_T2M)
        result = nc.reduce("valid_time", "mean", groupby=freq)
        assert result.get_variable("t2m").band_count == expected, (
            f"{freq} should yield {expected} windows, got "
            f"{result.get_variable('t2m').band_count}"
        )

    def test_sub_daily_window_values_match_numpy(self):
        """Each 6-hourly window is the per-step value (one member per window).

        Test scenario:
            With 12 distinct six-hourly steps, `"6h"` makes each window a
            single source step, so the reduced band equals that step's raw data.
        """
        nc = NetCDF.read_file(_ERA5_T2M)
        source = nc.reduce("valid_time", "mean", groupby="1MS")  # single month sanity
        assert source.get_variable("t2m").band_count == 1, "Jan-only data is one month"
        per_step = nc.reduce("valid_time", "mean", groupby="6h")
        raw = nc.get_variable("t2m").read_array()
        reduced = per_step.get_variable("t2m").read_array()
        assert np.allclose(
            reduced[0], raw[0], equal_nan=True
        ), "first 6h window != first step"


class TestReduceMultiBandDim:
    """Tests for reducing variables that keep more than one band dimension."""

    @staticmethod
    def _make_5d_two_var_nc() -> tuple[NetCDF, np.ndarray, np.ndarray]:
        """Build a 5-D `(d0, d1, d2, y, x)` container with two variables.

        Returns:
            tuple: `(container, v1_array, v2_array)` where the arrays are the
            full 5-D sources of variables `v1` and `v2`.
        """
        a1 = np.arange(2 * 2 * 2 * 3 * 4, dtype="float64").reshape(2, 2, 2, 3, 4)
        a2 = a1 + 1000.0
        extra = [("d0", [0, 1]), ("d1", [0, 1]), ("d2", [0, 1])]
        nc = NetCDF.create_from_array(
            a1,
            geo=_GEO,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="v1",
            extra_dims=extra,
        )
        v2 = NetCDF.create_from_array(
            a2,
            geo=_GEO,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="v2",
            extra_dims=extra,
        )
        nc.set_variable("v2", v2.get_variable("v2"))
        return nc, a1, a2

    def test_second_variable_with_two_remaining_band_dims(self):
        """Reducing a 5-D container collapses one dim, keeping the other two.

        Test scenario:
            Reducing `d0` (mean) on a two-variable 5-D container leaves each
            variable shaped `(d1, d2, y, x)` with values matching numpy — the
            non-first variable (`v2`) must not be corrupted by the
            single-band-axis `Dataset` store.
        """
        nc, a1, a2 = self._make_5d_two_var_nc()
        reduced = nc.reduce("d0", "mean")

        materialize = reduced._materialize_variable_array
        got_v1 = materialize(reduced.get_variable("v1"))
        got_v2 = materialize(reduced.get_variable("v2"))

        assert got_v1.shape == (2, 2, 3, 4), f"v1 wrong shape: {got_v1.shape}"
        assert got_v2.shape == (2, 2, 3, 4), f"v2 wrong shape: {got_v2.shape}"
        assert np.allclose(got_v1, a1.mean(axis=0)), "v1 values do not match numpy mean"
        assert np.allclose(
            got_v2, a2.mean(axis=0)
        ), "v2 (non-first) corrupted by reduce"

    def test_second_variable_band_dim_names_preserved(self):
        """Both surviving band dimensions are recorded on the reduced variable.

        Test scenario:
            After reducing `d0`, the non-first variable `v2` keeps band dims
            `("d1", "d2")` and drops `d0`.
        """
        nc, _, _ = self._make_5d_two_var_nc()
        var = nc.reduce("d0", "sum").get_variable("v2")
        assert var._band_dim_names == (
            "d1",
            "d2",
        ), f"unexpected band dims: {var._band_dim_names}"
        assert "d0" not in var._band_dim_names, "reduced dim should be gone"
