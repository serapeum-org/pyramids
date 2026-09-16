"""A scalar operator folds one variable with itself, so there is no second layout to compare.

`Analysis._fold` hands the variable to `Analysis._combine` as both operands, through the engine's
`weakref.proxy`. No identity test can tell that proxy from a second operand, so the fold must say
so itself: the layout hook is told there is no other operand, and neither compares coordinates nor
fills labels from one.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

HOURS_2000 = ("hours since 2000-01-01", "standard")
TIMES = [0.0, 6.0, 12.0, 18.0]
LEVELS = [1000.0, 850.0, 500.0]


@pytest.fixture
def dated() -> NetCDF:
    """A `(time=4, level=3)` variable on a 2x2 grid carrying hour units for `time`.

    Returns:
        NetCDF: The variable.
    """
    variable = NetCDF.from_array(
        np.arange(1.0, 4 * 3 * 2 * 2 + 1).reshape(4, 3, 2, 2),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        variable_name="t",
        dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
    ).get_variable("t")
    variable._band_dim_time_attrs = {"time": HOURS_2000}
    return variable


@pytest.fixture
def comparisons(monkeypatch) -> list[tuple]:
    """Record every `_disagreeing_coordinates` and `_fill_band_label` call, still running them.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        list[tuple]: `(helper name, first operand, second operand)` per call.
    """
    calls: list[tuple] = []
    compare = NetCDF._disagreeing_coordinates
    fill = NetCDF._fill_band_label

    def recording_compare(left, right):
        """Record a coordinate comparison, then run it.

        Args:
            left: The left operand.
            right: The right operand.

        Returns:
            list[str]: What the comparison returns.
        """
        calls.append(("_disagreeing_coordinates", left, right))
        return compare(left, right)

    def recording_fill(result, partner, name, partner_units):
        """Record a label fill, then run it.

        Args:
            result: The labelled result.
            partner: The operand filled from.
            name: The band dimension.
            partner_units: The partner's resolved units.
        """
        calls.append(("_fill_band_label", result, partner))
        fill(result, partner, name, partner_units)

    monkeypatch.setattr(
        NetCDF, "_disagreeing_coordinates", staticmethod(recording_compare)
    )
    monkeypatch.setattr(NetCDF, "_fill_band_label", staticmethod(recording_fill))
    return calls


class TestAScalarOperatorComparesNothing:
    """`var * 2`, `2 - var` and `var > 5` label the result without a second layout."""

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda v: v * 2, id="mul"),
            pytest.param(lambda v: 2 - v, id="reflected-sub"),
            pytest.param(lambda v: v > 5, id="gt"),
            pytest.param(lambda v: operator.truediv(v, 4), id="truediv"),
        ],
    )
    def test_no_comparison_and_no_fill(self, dated, comparisons, apply):
        """The fold asks for no coordinate comparison and no label fill.

        Args:
            dated: The labelled variable.
            comparisons: The recorded helper calls.
            apply: The scalar operator.

        Test scenario:
            The fold passed the engine's proxy as the second operand, which is not the variable
            itself, so every scalar operator compared the variable's coordinates with its own and
            looked up its units once per dimension on each side.
        """
        apply(dated)
        assert comparisons == [], [call[0] for call in comparisons]

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda v: v * 2, id="mul"),
            pytest.param(lambda v: 2 - v, id="reflected-sub"),
            pytest.param(lambda v: v > 5, id="gt"),
        ],
    )
    def test_the_labels_and_units_are_kept(self, dated, apply):
        """The result still carries the variable's coordinates and time units.

        Args:
            dated: The labelled variable.
            apply: The scalar operator.
        """
        result = apply(dated)
        assert result._band_dim_values_map == {"time": TIMES, "level": LEVELS}, (
            result._band_dim_values_map
        )
        assert result._band_dim_time_attrs == {"time": HOURS_2000}, (
            result._band_dim_time_attrs
        )

    def test_the_fold_s_values_are_computed(self, dated):
        """`2 - var` holds two minus each cell."""
        expected = 2 - np.asarray(dated.read_array(), dtype=np.float64)
        np.testing.assert_array_equal((2 - dated).read_array(), expected)

    def test_one_object_passed_twice_still_compares_at_once(self, dated, comparisons):
        """`var + var` is two operands that are one object: compared, and agreeing at once."""
        result = dated + dated
        assert [call[0] for call in comparisons] == ["_disagreeing_coordinates"]
        assert comparisons[0][1] is comparisons[0][2], comparisons[0]
        assert result._band_dim_values_map["time"] == TIMES
