"""Which units a band dimension carries as CF time units, and when they survive a change.

Only a `<period> since <origin>` unit decodes stamps into instants. A `level` in `millibar`
has units too, but they are not time units, so they neither decode nor decide whether two
operands' levels agree. And a result that carries its operand's time units in memory has to
keep them through an in-place change, or it stops selecting by date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).resolve().parents[2] / "data" / "netcdf"
ERA5_T2M = DATA / "cf__5v__1d4-3d1__geog__y-desc.nc"
COARDS = DATA / "coards__5v__1d4-4d1__y-desc.nc"
LEVELS = [1000.0, 850.0, 500.0]


def _levelled(units: str) -> NetCDF:
    """An in-memory `(level=3)` variable whose `level` carries `units`.

    Args:
        units: The `units` the level dimension is said to have.

    Returns:
        NetCDF: The variable.
    """
    variable = NetCDF.from_array(
        np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        variable_name="v",
        dims=ExtraDimensions(name="level", values=LEVELS),
    ).get_variable("v")
    variable._band_dim_time_attrs = {"level": (units, "standard")}
    return variable


class TestOnlyTimeUnitsAreCandidates:
    """`_time_attr_candidates` offers a dimension's units only when they are CF time units."""

    def test_a_millibar_level_offers_no_time_units(self):
        """The COARDS store's `level` is in millibar, so it offers nothing; `time` offers its units."""
        variable = NetCDF.read_file(str(COARDS)).get_variable("rhum")
        assert list(variable._time_attr_candidates("level")) == []
        assert list(variable._time_attr_candidates("time")) == [
            ("hours since 1-1-1 00:00:0.0", "standard")
        ]

    def test_resolved_attributes_hold_only_time_dimensions(self):
        """What a derived result would carry names `time` and not `level`."""
        variable = NetCDF.read_file(str(COARDS)).get_variable("rhum")
        assert sorted(variable._resolved_band_dim_time_attrs()) == ["time"]

    def test_levels_in_different_non_time_units_compare_by_value(self):
        """`millibar` against `hPa` over the same numbers is one axis, so the levels are kept.

        Test scenario:
            Neither unit decodes a stamp, so there are no instants to compare; the raw values
            decide, as they do when neither side has units at all.
        """
        left = _levelled("millibar")
        right = _levelled("hPa")
        assert NetCDF._disagreeing_coordinates(left, right) == []
        assert (left + right)._band_dim_values_map["level"] == LEVELS


class TestAnInPlaceChangeKeepsTheCarriedUnits:
    """An in-place change rebuilds the object; the time units it carries come through."""

    def test_an_operator_result_still_selects_by_date_after_an_in_place_fill(self):
        """`(t2m * 1.0).fill(1.0, inplace=True)` still selects the first day by its date label.

        Test scenario:
            The operator result has no store of its own and decodes its stamps with the units
            it carries. The in-place rebuild used to drop them, and the date label stopped
            matching.
        """
        result = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m") * 1.0
        units = dict(result._band_dim_time_attrs)
        result.fill(1.0, inplace=True)
        assert result._band_dim_time_attrs == units, result._band_dim_time_attrs
        day = result.sel(valid_time="2022-01-01")
        assert day.band_count == 4
        assert_array_equal(np.unique(np.asarray(day.read_array())), [1.0])


@pytest.fixture
def outlived():
    """`t2m` taken inside a `with` block, returned after the block has closed its container.

    Returns:
        tuple[NetCDF, np.ndarray]: The variable and its values, read while the container was open.
    """
    with NetCDF.read_file(str(ERA5_T2M)) as nc:
        variable = nc.get_variable("t2m")
        values = np.asarray(variable.read_array(), dtype=np.float64)
    return variable, values


class TestAVariableOutlivesItsContainer:
    """A variable keeps selecting, copying and combining after its container closes.

    Resolving the time units used to read the parent container's metadata on every derivation,
    and a closed container refuses that read.
    """

    @pytest.mark.parametrize(
        ("derive", "expected"),
        [
            pytest.param(
                lambda v: v.isel(valid_time=slice(0, 4)), lambda a: a[:4], id="isel"
            ),
            pytest.param(
                lambda v: v.sel(valid_time=1640995200), lambda a: a[:1], id="sel-number"
            ),
            pytest.param(lambda v: v.copy(), lambda a: a, id="copy"),
            pytest.param(lambda v: v * 2, lambda a: a * 2, id="scalar-operator"),
            pytest.param(lambda v: v - v, lambda a: a - a, id="variable-operator"),
        ],
    )
    def test_a_derivation_after_the_block(self, outlived, derive, expected):
        """The derivation succeeds and reads the values it would have read inside the block.

        Args:
            outlived: The variable and its values.
            derive: The derivation under test.
            expected: The same derivation on the values.
        """
        variable, values = outlived
        result = derive(variable)
        read = np.asarray(result.read_array(), dtype=np.float64).reshape(
            -1, *values.shape[-2:]
        )
        assert_array_equal(read, expected(values).reshape(-1, *values.shape[-2:]))

    def test_a_reprojection_after_the_block(self, outlived):
        """`to_crs` builds its warped copy from the variable alone."""
        variable, values = outlived
        assert variable.to_crs(3035).band_count == values.shape[0]

    def test_a_date_label_after_the_block(self, outlived):
        """The variable took its time units while the container was open, so a date still selects.

        Test scenario:
            The parent is closed, so its metadata cannot be read; the units `get_variable`
            carried over decode the stamps instead.
        """
        variable, _ = outlived
        assert variable._band_dim_time_attrs["valid_time"][0].startswith(
            "seconds since"
        )
        assert variable.sel(valid_time="2022-01-01").band_count == 4

    def test_an_operand_agrees_with_itself_without_looking_up_units(self, monkeypatch):
        """A scalar operator compares a variable with itself, and that asks for no units at all."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")

        def refuse(name: str) -> None:
            """A units lookup that must not happen.

            Args:
                name: The dimension asked for.

            Raises:
                AssertionError: Always.
            """
            raise AssertionError(f"units were looked up for {name!r}")

        monkeypatch.setattr(variable, "_time_attr_candidates", refuse)
        assert NetCDF._disagreeing_coordinates(variable, variable) == []

    def test_candidates_are_not_repeated(self):
        """The units the parent declares and the units the variable carries are yielded once."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        candidates = list(variable._time_attr_candidates("valid_time"))
        assert len(candidates) == 1, candidates
