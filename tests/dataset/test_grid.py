"""Unit tests for the Grid target-grid dataclass."""

from __future__ import annotations

import dataclasses

import pytest

from pyramids.dataset import Grid

pytestmark = pytest.mark.core

_LIKE = object()  # Grid only checks `like is not None`; any sentinel works.
_BOUNDS = (0.0, 0.0, 10.0, 10.0)


class TestGrid:
    """Tests for Grid construction, validation, and the is_empty property."""

    def test_default_is_empty(self):
        """A default Grid holds no fields and reports empty.

        Test scenario:
            Grid() -> every grid field None, is_empty True.
        """
        grid = Grid()
        assert grid.like is None, f"like should default None, got {grid.like!r}"
        assert grid.crs is None, f"crs should default None, got {grid.crs!r}"
        assert grid.resolution is None, f"resolution default: {grid.resolution!r}"
        assert grid.bounds is None, f"bounds should default None, got {grid.bounds!r}"
        assert grid.anchor == "edge", (
            f"anchor should default 'edge', got {grid.anchor!r}"
        )
        assert grid.is_empty is True, "default Grid should be empty"

    def test_like_only_valid_and_not_empty(self):
        """A template Grid carries `like` and is not empty.

        Test scenario:
            Grid(like=<obj>) -> is_empty False, like preserved.
        """
        grid = Grid(like=_LIKE)
        assert grid.like is _LIKE, "like should be stored unchanged"
        assert grid.is_empty is False, "a like Grid is not empty"

    def test_explicit_trio_valid_and_not_empty(self):
        """A complete explicit trio builds a valid, non-empty Grid.

        Test scenario:
            Grid(crs, resolution, bounds) -> fields preserved, is_empty False.
        """
        grid = Grid(crs=32633, resolution=10.0, bounds=_BOUNDS)
        assert grid.crs == 32633, f"crs: {grid.crs}"
        assert grid.resolution == 10.0, f"resolution: {grid.resolution}"
        assert grid.bounds == _BOUNDS, f"bounds: {grid.bounds}"
        assert grid.is_empty is False, "an explicit Grid is not empty"

    def test_crs_accepts_string(self):
        """crs may be a CRS string, not only an EPSG int.

        Test scenario:
            Grid(crs="EPSG:4326", resolution, bounds) constructs.
        """
        grid = Grid(crs="EPSG:4326", resolution=0.5, bounds=_BOUNDS)
        assert grid.crs == "EPSG:4326", f"crs string not preserved: {grid.crs}"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"crs": 4326},
            {"resolution": 10.0},
            {"bounds": _BOUNDS},
        ],
    )
    def test_like_with_any_explicit_member_raises(self, kwargs):
        """like combined with any explicit-grid member is rejected.

        Args:
            kwargs: A single explicit-grid field to pair with `like`.

        Test scenario:
            Grid(like=..., <one of crs/resolution/bounds>) -> ValueError.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            Grid(like=_LIKE, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"crs": 4326},
            {"resolution": 10.0},
            {"bounds": _BOUNDS},
            {"crs": 4326, "resolution": 10.0},
            {"crs": 4326, "bounds": _BOUNDS},
            {"resolution": 10.0, "bounds": _BOUNDS},
        ],
    )
    def test_partial_trio_raises(self, kwargs):
        """Any incomplete crs/resolution/bounds trio is rejected.

        Args:
            kwargs: A strict subset of the explicit-grid trio.

        Test scenario:
            Grid(<1 or 2 of the trio>) -> ValueError about "all be given
            together".
        """
        with pytest.raises(ValueError, match="all be given together"):
            Grid(**kwargs)

    def test_unknown_anchor_with_explicit_raises(self):
        """A non-'edge' anchor on an explicit grid is rejected.

        Test scenario:
            Grid(crs, resolution, bounds, anchor='center') -> ValueError.
        """
        with pytest.raises(ValueError, match="anchor must be"):
            Grid(crs=4326, resolution=10.0, bounds=_BOUNDS, anchor="center")

    def test_unknown_anchor_on_empty_raises(self):
        """The anchor is validated even without an explicit grid.

        Test scenario:
            Grid(anchor='nope') -> ValueError (anchor always checked).
        """
        with pytest.raises(ValueError, match="anchor must be"):
            Grid(anchor="nope")

    def test_is_frozen(self):
        """Grid is immutable; attribute assignment raises.

        Test scenario:
            grid.crs = 4326 -> FrozenInstanceError.
        """
        grid = Grid()
        with pytest.raises(dataclasses.FrozenInstanceError):
            grid.crs = 4326  # type: ignore[misc]

    def test_equality(self):
        """Two Grids with equal fields compare equal.

        Test scenario:
            Structural equality holds for identical explicit grids.
        """
        a = Grid(crs=32633, resolution=10.0, bounds=_BOUNDS)
        b = Grid(crs=32633, resolution=10.0, bounds=_BOUNDS)
        assert a == b, f"equal Grids should compare equal: {a} != {b}"
