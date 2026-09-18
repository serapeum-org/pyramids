"""The grid an operation stamps on the result it rebuilds.

A rebuilt container derives its geotransform back from the coordinate values it stores, and one
value carries no spacing, so a spatial axis left one cell long came back as a unit cell at the
axis origin. `_stamped` puts the grid the operation produced on the result instead, and
`get_variable` hands it on to the variables it builds. This module pins that stamp: the helper
itself, the members that must keep it, and the in-place swap that must not drop it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines._along_dim import _stamped

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(10.0, 2.0, 0.0, 60.0, 0.0, -2.0), epsg=4326)
"""A 2-degree grid whose 2x2 extent is `[10, 56, 14, 60]`, so the numbers read off by eye."""

EXTENT_CELL = (10.0, 4.0, 0, 60.0, 0, -4.0)
SPATIAL_BOUNDS = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__7v__1d3-2d3-3d1__y-asc.nc"
)
"""The geotransform of the one cell a full spatial weighting of `GEO` leaves."""

TIMES = [0.0, 6.0, 12.0]
MEMBERS = ("rolling", "diff", "cumsum", "shift", "argmin", "argmax", "idxmin", "idxmax")


def _source() -> NetCDF:
    """A three-step 2x2 container on `GEO`.

    Returns:
        NetCDF: The container, holding `v(time, y, x)`.
    """
    return NetCDF.from_array(
        np.arange(3 * 2 * 2, dtype="float64").reshape(3, 2, 2),
        geo_ref=GEO,
        variable_name="v",
        dims=ExtraDimensions(name="time", values=list(TIMES)),
    )


def _weighted_source() -> NetCDF:
    """A container reduced to one cell, whose grid only the stamp can describe.

    Returns:
        NetCDF: The result of weighting `_source()` over both spatial axes.
    """
    return _source().weighted("area")


def _call(nc: NetCDF, member: str) -> NetCDF:
    """Call one along-dimension member on `nc` with arguments every member accepts.

    Args:
        nc: The container or variable to call it on.
        member: One of `MEMBERS`.

    Returns:
        NetCDF: The result.
    """
    if member == "rolling":
        result = nc.rolling("time", 2)
    elif member == "shift":
        result = nc.shift("time", 1)
    else:
        result = getattr(nc, member)("time")
    return result


class TestStamped:
    """`_stamped` sets every geotransform the result can be asked for, and answers the result."""

    @staticmethod
    def _stamped_container() -> tuple[NetCDF, tuple]:
        """A container carrying a grid it could never have derived.

        Returns:
            tuple: The stamped container and the grid it was stamped with.
        """
        grid = (1.0, 2.0, 0.0, 3.0, 0.0, -4.0)
        return _stamped(_source(), grid), grid

    def test_the_same_object_comes_back(self):
        """The helper stamps in place, so the caller can wrap a rebuild in it."""
        container = _source()
        assert _stamped(container, EXTENT_CELL) is container, "must answer its argument"

    @pytest.mark.parametrize(
        "attribute",
        ["_geotransform", "_derived_geotransform", "_stamped_geotransform"],
    )
    def test_every_geotransform_attribute_is_set(self, attribute):
        """The memoised value, the stored one and the stamp all carry the grid.

        Args:
            attribute: The attribute that must hold the grid.

        Test scenario:
            The three disagreeing would let `geotransform` answer one thing and a later
            `get_variable` another, which is the split this stamp exists to close.
        """
        container, grid = self._stamped_container()
        assert getattr(container, attribute) == grid, (
            f"{attribute} is {getattr(container, attribute)!r}, expected {grid!r}"
        )

    def test_a_list_is_stored_as_a_tuple(self):
        """A geotransform given as a list is normalised, so it compares equal to a read one."""
        container = _stamped(_source(), [1.0, 2.0, 0.0, 3.0, 0.0, -4.0])
        assert container.geotransform == (1.0, 2.0, 0.0, 3.0, 0.0, -4.0), (
            f"stored {container.geotransform!r}"
        )

    def test_the_property_answers_the_stamp(self):
        """`geotransform` reads the memoised value, so the stamp is what callers see."""
        container, grid = self._stamped_container()
        assert container.geotransform == grid, f"read {container.geotransform!r}"


class TestTheStampSurvivesGetVariable:
    """A variable of a stamped container reports the container's grid, not index space."""

    def test_the_variable_agrees_with_the_container(self):
        """Both report the extent the weighting reduced.

        Test scenario:
            A variable of a rebuilt container carries no coordinate arrays of its own, so it
            fell back to the index space of the view: `(0.0, 1.0, 0, 1.0, 0, -1.0)`.
        """
        result = _weighted_source()
        variable = result.get_variable("v")
        assert variable.geotransform == result.geotransform == EXTENT_CELL, (
            f"container {result.geotransform!r}, variable {variable.geotransform!r}"
        )

    def test_the_variable_carries_the_stamp_itself(self):
        """The stamp is handed on, so an operation on the variable can pass it along again."""
        variable = _weighted_source().get_variable("v")
        assert variable._stamped_geotransform == EXTENT_CELL, (
            f"stamp {variable._stamped_geotransform!r}"
        )

    def test_asking_twice_answers_the_same_grid(self):
        """`get_variable` is not a one-shot: a second variable is stamped like the first."""
        result = _weighted_source()
        first = result.get_variable("v").geotransform
        assert result.get_variable("v").geotransform == first, "second read differs"

    def test_an_unstamped_container_is_left_alone(self):
        """A container that never went through an operation keeps deriving its own grid."""
        source = _source()
        assert source._stamped_geotransform is None, source._stamped_geotransform
        assert source.get_variable("v").geotransform == GEO.geo, (
            f"derived {source.get_variable('v').geotransform!r}"
        )


class TestTheStampSurvivesTheAlongDimensionMembers:
    """An operation on a one-cell result keeps the grid the weighting produced."""

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_container_keeps_the_extent(self, member):
        """Every member rebuilds the container and stamps the grid it was handed.

        Args:
            member: The member called on the weighted container.

        Test scenario:
            Without the stamp the rebuild derived a unit cell at the axis origin from the one
            stored coordinate, so a `cumsum` after a `weighted` silently moved the result.
        """
        result = _call(_weighted_source(), member)
        assert result.geotransform == EXTENT_CELL, (
            f"{member}() left the container on {result.geotransform!r}"
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_variable_keeps_the_extent(self, member):
        """The variable taken from the result agrees with the container it came from.

        Args:
            member: The member called on the weighted container.
        """
        result = _call(_weighted_source(), member)
        variable = result.get_variable("v")
        assert variable.geotransform == EXTENT_CELL, (
            f"{member}() left the variable on {variable.geotransform!r}"
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_a_variable_receiver_keeps_the_extent(self, member):
        """Called on the variable itself the member rebuilds a variable, stamped the same way.

        Args:
            member: The member called on the weighted variable.
        """
        result = _call(_weighted_source().get_variable("v"), member)
        assert result.geotransform == EXTENT_CELL, (
            f"{member}() left the variable on {result.geotransform!r}"
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_a_full_grid_is_unchanged_by_the_stamp(self, member):
        """A grid long enough to measure derives exactly the stamp, so nothing moves.

        Args:
            member: The member called on the unreduced container.
        """
        result = _call(_source(), member)
        assert result.geotransform == GEO.geo, (
            f"{member}() moved a full grid to {result.geotransform!r}"
        )


class TestTheStampSurvivesAnInPlaceUpdate:
    """`_update_inplace` rebuilds the wrapper, and the stamp is part of what it carries over."""

    def test_changing_the_no_data_value_keeps_the_grid(self):
        """`change_no_data_value` swaps the raster, and the stamped grid comes with it.

        Test scenario:
            The swap runs `NetCDF.__init__` on the rebuilt raster, which clears the memoised
            geotransform; without the stamp in the preserved snapshot the container fell back to
            deriving one from its single coordinate — `(11.5, 1.0, 0, 58.5, 0, -1.0)`.
        """
        result = _weighted_source()
        result.change_no_data_value(-8888.0)
        assert result.geotransform == EXTENT_CELL, f"read {result.geotransform!r}"

    def test_a_variable_taken_after_the_swap_is_stamped(self):
        """The stamp is still there to hand on, so `get_variable` answers the extent too."""
        result = _weighted_source()
        result.change_no_data_value(-8888.0)
        assert result.get_variable("v").geotransform == EXTENT_CELL, (
            f"read {result.get_variable('v').geotransform!r}"
        )

    def test_setting_the_crs_keeps_the_grid(self):
        """`set_crs` is the other in-place swap a weighted result is likely to meet."""
        result = _weighted_source()
        result.set_crs(result.crs)
        assert result.geotransform == EXTENT_CELL, f"read {result.geotransform!r}"


class TestWhatTheStampIsWorth:
    """Without the stamp the same rebuild answers a unit cell at the axis origin."""

    def test_clearing_it_gives_the_grid_back_to_the_coordinates(self):
        """Dropping the stamp restores the derivation it replaces, which is the wrong cell.

        Test scenario:
            This is the behaviour the stamp exists to prevent, pinned so the helper cannot be
            removed as a no-op: the container derives `(11.5, 1.0, ...)` and the variable
            `(0.0, 1.0, ...)` — two answers, neither the extent that was reduced.

            The memoised `cell_size` is put back to the store's own as well, because that is
            the state the stamp replaces: the single-coordinate fallback in
            `_compute_geotransform` reads `cell_size`, so a stamped width alone is enough to
            make the derivation right again.
        """
        result = _weighted_source()
        result._stamped_geotransform = None
        result._geotransform = None
        result._derived_geotransform = None
        result._cell_size = 1.0
        assert result.geotransform != EXTENT_CELL, (
            "the derivation cannot reach the extent"
        )
        assert result.get_variable("v").geotransform != EXTENT_CELL, (
            "an unstamped variable cannot reach the extent either"
        )


class TestTheStampedGridReachesTheCoordinates:
    """The cell the stamp describes is where `lat` / `lon` and `bounds` put it."""

    def test_the_cell_holds_the_source_centre(self):
        """One cell over `[10, 56, 14, 60]` is centred at `(12, 58)`."""
        variable = _weighted_source().get_variable("v")
        assert float(variable.lon[0]) == pytest.approx(12.0), float(variable.lon[0])
        assert float(variable.lat[0]) == pytest.approx(58.0), float(variable.lat[0])

    def test_the_bounds_are_the_source_bounds(self):
        """The one cell covers exactly what it was reduced from."""
        variable = _weighted_source().get_variable("v")
        assert variable.bounds.total_bounds.tolist() == [10.0, 56.0, 14.0, 60.0], (
            variable.bounds.total_bounds.tolist()
        )

    def test_a_member_after_the_weighting_keeps_the_centre(self):
        """A `cumsum` over the remaining dimension leaves the cell where it was."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result: Any = _weighted_source().cumsum("time")
        variable = result.get_variable("v")
        assert float(variable.lon[0]) == pytest.approx(12.0), float(variable.lon[0])
        assert float(variable.lat[0]) == pytest.approx(58.0), float(variable.lat[0])


class TestTheStampedCellSize:
    """`cell_size` is `abs(pixel_width)`, so it has to follow the grid that was stamped."""

    @staticmethod
    def _source() -> NetCDF:
        """A 2x2 cube on 5-degree cells.

        Returns:
            NetCDF: The container, `t(time, y, x)` on `[0, 50, 10, 60]`.
        """
        return NetCDF.from_array(
            np.arange(2 * 2 * 2, dtype="float64").reshape(2, 2, 2),
            geo_ref=GeoReference(geo=(0.0, 5.0, 0.0, 60.0, 0.0, -5.0), epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )

    def test_it_matches_the_stamped_geotransform(self):
        """A weighted result reports the width its own geotransform describes.

        Test scenario:
            `_stamped` set the geotransform and the value `lat` / `lon` derive from, but not the
            memoised `cell_size`, which the rebuilt store had already computed from its
            index-space grid. The result then said `cell_size == 1.0` while its geotransform
            said the cell was 10 degrees wide — two answers for one raster.
        """
        result = self._source().weighted("area")
        variable = result.get_variable("t")
        assert float(variable.cell_size) == pytest.approx(abs(variable.geotransform[1]))
        assert float(result.cell_size) == pytest.approx(abs(result.geotransform[1]))

    def test_the_width_is_the_span_that_was_reduced(self):
        """Two 5-degree columns reduce to one cell 10 degrees wide."""
        variable = self._source().weighted("area").get_variable("t")
        assert float(variable.cell_size) == pytest.approx(10.0)

    def test_a_member_that_keeps_the_grid_keeps_the_width(self):
        """`rolling` touches no spatial axis, so the cell size is the source's."""
        source = self._source()
        result = source.rolling("time", 2)
        assert float(result.get_variable("t").cell_size) == pytest.approx(5.0)


class TestTheStampSurvivesCarryingAnAuxiliary:
    """A real CF store has auxiliary variables to carry, and carrying one rebuilds the wrapper."""

    @staticmethod
    def _bounded() -> NetCDF:
        """A CF store with bounds variables, one of which a spatial weighting carries.

        Returns:
            NetCDF: `tos(time, lat, lon)` beside `time_bnds(time, bnds)`, which survives a
            spatial weighting, and `lat_bnds` / `lon_bnds`, which do not.
        """
        return NetCDF.read_file(str(SPATIAL_BOUNDS))

    @staticmethod
    def _expected(source: NetCDF) -> tuple:
        """The one cell the source's whole extent becomes.

        Args:
            source: The container being weighted.

        Returns:
            tuple: The geotransform of a result whose two spatial axes were reduced.
        """
        variable = source.get_variable("tos")
        geo = list(variable.geotransform)
        geo[1] = geo[1] * variable.columns
        geo[5] = geo[5] * variable.rows
        return tuple(geo)

    def test_the_container_keeps_the_stamped_grid(self):
        """Carrying `time_bnds` must not give the container's grid back to the derivation.

        Test scenario:
            `_carry_auxiliaries` adds the carried variable with `add_variable(copy=False)`, which
            rebuilds the container's wrapper from its raster. The rebuild recomputed the
            geotransform from the single stored coordinate, so the container reported
            `(0.0, 360.0, 0, 185.0, 0, -360.0)` where the stamp had said
            `(0.0, 360.0, 0, 90.0, 0, -170.0)` — an origin 95 degrees out. Every earlier stamp
            test built its source with `from_array`, which has no auxiliary to carry, so the
            path was never exercised.
        """
        source = self._bounded()
        expected = self._expected(source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = source.weighted("area")
        assert result.geotransform == expected, f"read {result.geotransform!r}"

    def test_the_container_and_its_variable_still_agree(self):
        """The split the stamp exists to close stays closed once an auxiliary is carried."""
        source = self._bounded()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = source.weighted("area")
        assert result.geotransform == result.get_variable("tos").geotransform

    def test_the_carried_auxiliary_is_still_there(self):
        """The grid is kept without giving up the variable the rebuild was for."""
        source = self._bounded()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = source.weighted("area")
        assert "time_bnds" in result.variable_names, result.variable_names
