"""Unit tests for ``_CFCoordinateCandidates`` (CF auxiliary-coordinate pairing).

The value object holds the classified x / y coordinate candidates for a data slice
and owns the two-pass ``(x, y)`` pairing. These tests drive it in isolation with
small real numpy arrays and a duck-typed fake ``parent``; the end-to-end plot path
is covered by ``tests/netcdf/plot/test_plot_coords.py``.
"""

from __future__ import annotations

import logging

import numpy as np

from pyramids.netcdf._plot import _CFCoordinateCandidates

_SHAPE = (4, 5)  # (rows, cols)


class _FakeParent:
    """A duck-typed stand-in for a NetCDF container: name -> coord array (or None)."""

    def __init__(self, arrays):
        """Store the name -> array mapping (``None`` values model unreadable vars)."""
        self._arrays = arrays

    @property
    def variable_names(self):
        """Return the declared variable names."""
        return list(self._arrays)

    def _read_variable(self, name):
        """Return the array for ``name`` (``None`` if unreadable)."""
        return self._arrays.get(name)


def _lon1d():
    """A 1-D x (cols-length) coordinate."""
    return np.arange(5.0)


def _lat1d():
    """A 1-D y (rows-length) coordinate."""
    return np.arange(4.0)


def _lat2d():
    """A 2-D coordinate whose values are within latitude bounds."""
    return np.full(_SHAPE, 45.0)


def _lon2d():
    """A 2-D coordinate whose values exceed latitude bounds (reads as longitude)."""
    return np.full(_SHAPE, 200.0)


class TestCFCoordinateCandidates:
    """Tests for ``_CFCoordinateCandidates`` gather + pairing."""

    def test_gather_classifies_x_and_y(self):
        """gather classifies 1-D cols/rows and 2-D coords into the x / y lists.

        Test scenario:
            A 1-D-cols ``lon`` is x-only, a 1-D-rows ``lat`` is y-only, and a 2-D
            ``grid`` matches both axes so it appears in both lists.
        """
        parent = _FakeParent({"lon": _lon1d(), "lat": _lat1d(), "grid": _lat2d()})
        candidates = _CFCoordinateCandidates.gather(
            ["lon", "lat", "grid"], parent, _SHAPE
        )
        x_names = {n for n, _ in candidates._x}
        y_names = {n for n, _ in candidates._y}
        assert x_names == {"lon", "grid"}, f"x candidates wrong: {x_names}"
        assert y_names == {"lat", "grid"}, f"y candidates wrong: {y_names}"

    def test_gather_skips_missing_and_unreadable_vars(self):
        """gather skips names absent from the parent or reading back ``None``.

        Test scenario:
            ``absent`` is not a variable name; ``bad`` reads ``None`` — neither ends
            up in the resolved arrays.
        """
        parent = _FakeParent({"lon": _lon1d(), "bad": None})
        candidates = _CFCoordinateCandidates.gather(
            ["lon", "bad", "absent"], parent, _SHAPE
        )
        assert set(candidates.arrays_by_name()) == {"lon"}, (
            f"only readable, present vars should remain: {set(candidates.arrays_by_name())}"
        )

    def test_gather_applies_squeeze(self):
        """gather squeezes a 3-D leading axis to the 2-D slice shape.

        Test scenario:
            A ``(2, 4, 5)`` coord is reduced to ``(4, 5)`` during gather.
        """
        parent = _FakeParent({"c3": np.zeros((2, 4, 5))})
        candidates = _CFCoordinateCandidates.gather(["c3"], parent, _SHAPE)
        assert candidates.arrays_by_name()["c3"].shape == _SHAPE, (
            "3-D coord must be squeezed"
        )

    def test_best_pair_prefers_lon_lat_names(self):
        """best_pair uses the lon/lat name heuristic before the distinct fallback.

        Test scenario:
            With both a lon/lat named pair and other viable candidates, the named
            pair wins (the distinct fallback is not reached).
        """
        east, lon = np.arange(5.0), np.arange(5.0) + 100
        north, lat = np.arange(4.0), np.arange(4.0) + 100
        parent = _FakeParent({"east": east, "lon": lon, "north": north, "lat": lat})
        candidates = _CFCoordinateCandidates.gather(
            ["east", "lon", "north", "lat"], parent, _SHAPE
        )
        pair = candidates.best_pair()
        assert pair is not None, "a lon/lat pair must be found"
        assert pair[0] is lon and pair[1] is lat, "the lon/lat named pair must win"

    def test_distinct_pair_does_not_collapse_a_single_2d_coord(self):
        """A lone 2-D coord (in both lists) is not paired with itself.

        Test scenario:
            One 2-D coord ``grid`` lands in both x and y lists; the distinct-pair
            guard rejects ``grid``/``grid``, so best_pair returns ``None``.
        """
        parent = _FakeParent({"grid": _lat2d()})
        candidates = _CFCoordinateCandidates.gather(["grid"], parent, _SHAPE)
        assert candidates.best_pair() is None, (
            "a single 2-D coord must not collapse onto both axes"
        )

    def test_distinct_pair_two_2d_disambiguated_by_latitude(self):
        """Two 2-D candidates (non-lon/lat names) are assigned by latitude range.

        Test scenario:
            With ``a`` (longitude-valued) and ``b`` (latitude-valued), the fallback
            assigns the within-latitude array to y.
        """
        lon2d, lat2d = _lon2d(), _lat2d()
        parent = _FakeParent({"a": lon2d, "b": lat2d})
        candidates = _CFCoordinateCandidates.gather(["a", "b"], parent, _SHAPE)
        pair = candidates.best_pair()
        assert pair is not None and pair[0] is lon2d and pair[1] is lat2d, (
            "the within-latitude 2-D array must become y"
        )

    def test_distinct_pair_1d_fallback(self):
        """Distinct 1-D candidates with non-lon/lat names still pair (x, y).

        Test scenario:
            ``east`` (1-D cols) and ``north`` (1-D rows) are paired directly (not a
            2-D disambiguation).
        """
        east, north = _lon1d(), _lat1d()
        parent = _FakeParent({"east": east, "north": north})
        candidates = _CFCoordinateCandidates.gather(["east", "north"], parent, _SHAPE)
        pair = candidates.best_pair()
        assert pair is not None and pair[0] is east and pair[1] is north, (
            "distinct 1-D candidates must pair as (x, y)"
        )

    def test_best_pair_none_when_no_candidates(self):
        """best_pair returns ``None`` when nothing classifies as a coord.

        Test scenario:
            A wrong-shaped array matches neither axis, so there is no pair.
        """
        parent = _FakeParent({"bad": np.zeros((2, 2))})
        candidates = _CFCoordinateCandidates.gather(["bad"], parent, _SHAPE)
        assert candidates.best_pair() is None, "no viable candidates must yield None"

    def test_assign_2d_by_latitude_is_symmetric(self):
        """The 2-D assignment puts the within-latitude array on y regardless of order.

        Test scenario:
            Whether the latitude array is passed as x or y, the result is
            ``(longitude, latitude)`` — the assignment is order-independent.
        """
        lon2d, lat2d = _lon2d(), _lat2d()
        forward = _CFCoordinateCandidates._assign_2d_by_latitude("a", lat2d, "b", lon2d)
        reverse = _CFCoordinateCandidates._assign_2d_by_latitude("a", lon2d, "b", lat2d)
        assert forward[0] is lon2d and forward[1] is lat2d, (
            "lat-first still yields (lon, lat)"
        )
        assert reverse[0] is lon2d and reverse[1] is lat2d, (
            "lon-first still yields (lon, lat)"
        )

    def test_assign_2d_by_latitude_ambiguous_keeps_order(self, caplog):
        """Ambiguous roles keep candidate order and log a DEBUG hint on the _plot logger.

        Test scenario:
            Two within-latitude arrays (ambiguous) return in the given order, and a
            debug record is emitted on the name-preserved ``pyramids.netcdf._plot``
            logger so the diagnostic cannot be silently renamed by a later edit.
        """
        a, b = _lat2d(), _lat2d() + 1.0
        with caplog.at_level(logging.DEBUG, logger="pyramids.netcdf._plot"):
            pair = _CFCoordinateCandidates._assign_2d_by_latitude("a", a, "b", b)
        assert pair[0] is a and pair[1] is b, (
            "ambiguous roles must keep candidate order"
        )
        hints = [r for r in caplog.records if r.name == "pyramids.netcdf._plot"]
        assert hints, (
            "an ambiguous-roles DEBUG record must be emitted on the _plot logger"
        )
        assert "ambiguous" in hints[0].getMessage(), (
            f"unexpected ambiguous-roles log message: {hints[0].getMessage()}"
        )

    def test_arrays_by_name_returns_resolved_arrays(self):
        """arrays_by_name exposes the resolved candidate arrays for the no-match log.

        Test scenario:
            The gathered arrays are returned keyed by name.
        """
        lon = _lon1d()
        candidates = _CFCoordinateCandidates.gather(
            ["lon"], _FakeParent({"lon": lon}), _SHAPE
        )
        assert candidates.arrays_by_name()["lon"] is lon, (
            "resolved array must be exposed by name"
        )
