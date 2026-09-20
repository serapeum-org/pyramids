"""`interpolate_na` — fill the interior gaps along a dimension from the cells around them.

Every expectation is xarray's, computed by xarray in the test: `xds.interpolate_na(dim,
method=..., limit=..., use_coordinate=...)`. The case that proves `use_coordinate` works is
the uneven axis, where measuring by coordinate and measuring by position give different
answers.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.interop

xr = pytest.importorskip("xarray")

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
EVEN = [0.0, 1.0, 2.0, 3.0]
UNEVEN = [0.0, 1.0, 5.0, 6.0]
GAPPED = [1.0, np.nan, np.nan, 7.0]


def _variable(values: list[float], stamps: list[float]) -> NetCDF:
    """A one-cell variable over `time`, its gaps left as NaN.

    Args:
        values: One value per step.
        stamps: The `time` coordinate values.

    Returns:
        NetCDF: The variable.
    """
    return NetCDF.from_array(
        np.array(values, dtype="float64").reshape(len(values), 1, 1),
        geo_ref=GEO,
        variable_name="t",
        no_data_value=None,
        dims=ExtraDimensions(name="time", values=stamps),
    ).get_variable("t")


def _exported(values: list[float], stamps: list[float]) -> xr.DataArray:
    """The same column as xarray sees it.

    Args:
        values: One value per step.
        stamps: The `time` coordinate values.

    Returns:
        xr.DataArray: The column.
    """
    return xr.DataArray(
        np.array(values, dtype="float64"), dims=("time",), coords={"time": stamps}
    )


def _read(result: NetCDF) -> list[float]:
    """A one-cell result's values as a flat list, its gaps as NaN.

    Args:
        result: The variable to read.

    Returns:
        list[float]: One value per step.
    """
    values = np.asarray(result.read_array(), dtype="float64").ravel()
    sentinel = result.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values.tolist()


class TestAgainstXarray:
    """Each case answers what `xds.interpolate_na` answers for the same column."""

    @pytest.mark.parametrize("stamps", [EVEN, UNEVEN])
    def test_the_default_linear_fill(self, stamps):
        """An evenly and an unevenly spaced axis both match.

        Args:
            stamps: The `time` coordinates.
        """
        assert_allclose(
            _read(_variable(GAPPED, stamps).interpolate_na("time")),
            _exported(GAPPED, stamps).interpolate_na("time").values,
            equal_nan=True,
        )

    def test_the_uneven_axis_is_the_case_that_proves_the_coordinate_is_used(self):
        """Measuring by coordinate and by position give different answers, and both match.

        Test scenario:
            On `[0, 1, 5, 6]` the two gaps sit right after the first value and right before
            the last, so the coordinate answer `[1, 2, 6, 7]` is nothing like the positional
            `[1, 3, 5, 7]`. Matching both is what shows the distance is really measured.
        """
        exported = _exported(GAPPED, UNEVEN)
        by_coordinate = _read(_variable(GAPPED, UNEVEN).interpolate_na("time"))
        by_position = _read(
            _variable(GAPPED, UNEVEN).interpolate_na("time", use_coordinate=False)
        )
        assert by_coordinate != by_position
        assert_allclose(
            by_coordinate, exported.interpolate_na("time").values, equal_nan=True
        )
        assert_allclose(
            by_position,
            exported.interpolate_na("time", use_coordinate=False).values,
            equal_nan=True,
        )

    @pytest.mark.parametrize("limit", [1, 2, 3])
    def test_the_limit(self, limit):
        """A run longer than the limit keeps the gaps beyond it, as xarray keeps them.

        Args:
            limit: How many consecutive gaps a run may fill.
        """
        assert_allclose(
            _read(_variable(GAPPED, EVEN).interpolate_na("time", limit=limit)),
            _exported(GAPPED, EVEN).interpolate_na("time", limit=limit).values,
            equal_nan=True,
        )

    def test_the_nearest_method(self):
        """`"nearest"` takes the closer neighbour, and agrees with xarray's."""
        assert_allclose(
            _read(_variable(GAPPED, EVEN).interpolate_na("time", "nearest")),
            _exported(GAPPED, EVEN).interpolate_na("time", method="nearest").values,
            equal_nan=True,
        )

    def test_the_leading_and_trailing_gaps_are_left_alone(self):
        """Only an interior gap has two sides to be placed between."""
        values = [np.nan, 2.0, np.nan, 4.0, np.nan]
        stamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert_allclose(
            _read(_variable(values, stamps).interpolate_na("time")),
            _exported(values, stamps).interpolate_na("time").values,
            equal_nan=True,
        )

    def test_a_column_without_gaps_is_unchanged(self):
        """Nothing is missing, so nothing is interpolated."""
        values = [1.0, 2.0, 3.0, 4.0]
        assert_allclose(_read(_variable(values, EVEN).interpolate_na("time")), values)


class TestTheGrid:
    """Interpolating changes values, never the layout."""

    def test_the_dimension_keeps_its_length_and_stamps(self):
        """A filled gap is still a step, so nothing is dropped."""
        result = _variable(GAPPED, UNEVEN).interpolate_na("time")
        assert result.band_count == 4
        assert result._band_dim_values_map["time"] == UNEVEN

    def test_a_declared_sentinel_marks_what_could_not_be_reached(self):
        """A band declaring a sentinel keeps it, and the unreachable edges hold it."""
        values = np.array([np.nan, 2.0, np.nan, 4.0], dtype="float64")
        variable = NetCDF.from_array(
            np.where(np.isnan(values), -9999.0, values).reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-9999.0,
            dims=ExtraDimensions(name="time", values=EVEN),
        ).get_variable("t")
        result = variable.interpolate_na("time")
        assert result.no_data_value[0] == pytest.approx(-9999.0)
        assert_allclose(_read(result), [np.nan, 2.0, 3.0, 4.0], equal_nan=True)

    def test_a_two_dimensional_grid_is_interpolated_cell_by_cell(self):
        """Every cell of the grid gets its own interpolation along the dimension."""
        values = np.array(
            [[[1.0, 10.0]], [[np.nan, np.nan]], [[3.0, 30.0]]], dtype="float64"
        )
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="t",
            no_data_value=None,
            dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0]),
        ).get_variable("t")
        read = np.asarray(variable.interpolate_na("time").read_array(), dtype="float64")
        assert_allclose(read[1].ravel(), [2.0, 20.0])


class TestRefusals:
    """The arguments are checked before anything is read."""

    def test_an_unknown_method(self):
        """Only the two implemented interpolations are accepted."""
        variable = _variable(GAPPED, EVEN)
        with pytest.raises(ValueError, match="method="):
            variable.interpolate_na("time", "cubic")

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_limit_below_one(self, limit):
        """A limit of zero would fill nothing, which is not what the caller meant.

        Args:
            limit: The refused limit.
        """
        variable = _variable(GAPPED, EVEN)
        with pytest.raises(ValueError, match="at least 1"):
            variable.interpolate_na("time", limit=limit)

    def test_a_fractional_limit(self):
        """A limit counts steps, so it is an integer."""
        variable = _variable(GAPPED, EVEN)
        with pytest.raises(TypeError):
            variable.interpolate_na("time", limit=1.5)

    def test_a_dimension_the_variable_lacks(self):
        """The dimension has to be one of the variable's own."""
        variable = _variable(GAPPED, EVEN)
        with pytest.raises(ValueError, match="does not match any band dimension"):
            variable.interpolate_na("level")

    def test_a_boolean_limit_is_refused(self):
        """`True` is an `int` in Python and would quietly mean a limit of one step."""
        variable = _variable(GAPPED, EVEN)
        with pytest.raises(TypeError, match=r"^interpolate_na\(\) needs an integer"):
            variable.interpolate_na("time", limit=True)

    def test_text_stamps_cannot_be_measured_along(self):
        """A distance between two labels is not defined, so the default is refused.

        Test scenario:
            `use_coordinate=True` is the default and asks for the distance between two
            stamps. Text stamps have none, and falling back to position silently would
            answer a different interpolation than the one that was asked for.
        """
        variable = _variable(GAPPED, EVEN)
        variable._band_dim_values_map["time"] = ["a", "b", "c", "d"]
        with pytest.raises(
            ValueError, match="cannot measure distance along 'time'"
        ) as raised:
            variable.interpolate_na("time")
        assert "use_coordinate=False" in str(raised.value), (
            f"the refusal must name the way through, got: {raised.value}"
        )


class TestTheReceivers:
    """Both receivers answer, and they agree."""

    @staticmethod
    def _container() -> NetCDF:
        """A container holding one gridded variable over `time`.

        Returns:
            NetCDF: The container.
        """
        return NetCDF.from_array(
            np.array(GAPPED, dtype="float64").reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=None,
            dims=ExtraDimensions(name="time", values=UNEVEN),
        )

    def test_a_container_answers_a_container(self):
        """The container receiver interpolates every variable that has the dimension."""
        result = self._container().interpolate_na("time")
        assert "t" in result.variable_names

    def test_both_receivers_agree(self):
        """A variable taken from the result holds what interpolating the variable holds."""
        container = self._container()
        assert_allclose(
            _read(container.interpolate_na("time").get_variable("t")),
            _read(container.get_variable("t").interpolate_na("time")),
            equal_nan=True,
        )
