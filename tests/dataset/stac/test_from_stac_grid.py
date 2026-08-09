"""Tests for the grid match: from_stac(grid=Grid(like=.../crs=.../...))."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection, Grid
from pyramids.dataset._stac import _resolve_target_grid

pytestmark = pytest.mark.core


@pytest.fixture
def offset_grid_items(tmp_path):
    """Two items on a coarse grid that differs from the target template grid.

    Each is a 2x2 EPSG:4326 raster at cell size 2 (top-left (0, 4)).

    Returns:
        list[dict]: two raw STAC items with a "data" asset.
    """
    items = []
    for i in range(2):
        ds = Dataset.create_from_array(
            np.full((2, 2), float(i + 1), dtype="float32"),
            top_left_corner=(0.0, 4.0),
            cell_size=2.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        p = str(tmp_path / f"g{i}.tif")
        ds.to_file(p)
        items.append(
            {
                "id": f"i{i}",
                "bbox": [0.0, 0.0, 4.0, 4.0],
                "assets": {"data": {"href": p}},
            }
        )
    return items


@pytest.fixture
def template(tmp_path):
    """A fine 4x4 EPSG:4326 grid (cell 1, top-left (0, 4)) to match against."""
    return Dataset.create_from_array(
        np.zeros((4, 4), dtype="float32"),
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999.0,
    )


class TestGrid:
    """Tests for the Grid dataclass and its construction-time validation."""

    def test_empty_is_empty(self):
        """A default Grid reports empty.

        Test scenario:
            Grid() with no fields set -> is_empty True.
        """
        assert Grid().is_empty is True, "default Grid should be empty"

    def test_explicit_not_empty(self):
        """An explicit-grid Grid is not empty.

        Test scenario:
            crs + resolution + bounds -> is_empty False.
        """
        grid = Grid(crs=4326, resolution=1.0, bounds=(0, 0, 4, 4))
        assert grid.is_empty is False, "explicit Grid should not be empty"

    def test_like_with_explicit_raises(self, template):
        """like combined with crs/resolution/bounds raises.

        Test scenario:
            Mutually-exclusive grid specs are rejected at construction.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            Grid(like=template, crs=4326, resolution=1.0, bounds=(0, 0, 4, 4))

    def test_partial_explicit_raises(self):
        """An incomplete crs/resolution/bounds trio raises.

        Test scenario:
            crs + resolution without bounds is rejected at construction.
        """
        with pytest.raises(ValueError, match="all be given together"):
            Grid(crs=4326, resolution=1.0)

    def test_unknown_anchor_raises(self):
        """An unsupported anchor raises.

        Test scenario:
            anchor='center' is not implemented.
        """
        with pytest.raises(ValueError, match="anchor must be"):
            Grid(crs=4326, resolution=1.0, bounds=(0, 0, 4, 4), anchor="center")


class TestResolveTargetGrid:
    """Tests for the _resolve_target_grid helper (Grid -> template Dataset)."""

    def test_none_when_nothing_requested(self):
        """None or an empty Grid -> None (no alignment).

        Test scenario:
            No grid, and an empty Grid, both resolve to None.
        """
        assert _resolve_target_grid(None) is None
        assert _resolve_target_grid(Grid()) is None

    def test_like_returned_directly(self, template):
        """A like Dataset is returned as the template.

        Test scenario:
            Grid(like=...) passes the Dataset through unchanged.
        """
        assert _resolve_target_grid(Grid(like=template)) is template

    def test_explicit_builds_snapped_template(self):
        """An explicit grid builds a template snapped to the resolution.

        Test scenario:
            bounds (0.4, 0.4, 3.6, 3.6) at resolution 1 snap to (0,0,4,4) -> 4x4.
        """
        tpl = _resolve_target_grid(
            Grid(crs=4326, resolution=1.0, bounds=(0.4, 0.4, 3.6, 3.6))
        )
        assert tpl.epsg == 4326, f"epsg: {tpl.epsg}"
        assert (tpl.rows, tpl.columns) == (4, 4), f"shape: {(tpl.rows, tpl.columns)}"


class TestFromStacGridMatch:
    """Tests for from_stac grid match via Grid(like=) / explicit Grid."""

    def test_like_matches_grid(self, offset_grid_items, template):
        """grid=Grid(like=) resamples every timestep onto the template's grid.

        Test scenario:
            Coarse 2x2 items -> a 4x4 cube matching the template.
        """
        coll = DatasetCollection.from_stac(
            offset_grid_items, asset="data", grid=Grid(like=template)
        )
        assert coll.time_length == 2, f"expected 2 timesteps, got {coll.time_length}"
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (
            4,
            4,
        ), f"not aligned to template: {(first.rows, first.columns)}"
        assert first.epsg == 4326, f"epsg: {first.epsg}"

    def test_explicit_grid_matches(self, offset_grid_items):
        """grid=Grid(crs/resolution/bounds) builds and matches an explicit grid.

        Test scenario:
            A 1-degree grid over (0,0,4,4) yields 4x4 timesteps.
        """
        coll = DatasetCollection.from_stac(
            offset_grid_items,
            asset="data",
            grid=Grid(crs=4326, resolution=1.0, bounds=(0.0, 0.0, 4.0, 4.0)),
        )
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (
            4,
            4,
        ), f"shape: {(first.rows, first.columns)}"

    def test_no_grid_match_keeps_native(self, offset_grid_items):
        """Without a grid the cube keeps the native (coarse) grid.

        Test scenario:
            No grid arg -> 2x2 native timesteps.
        """
        coll = DatasetCollection.from_stac(offset_grid_items, asset="data")
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (
            2,
            2,
        ), f"should stay native 2x2: {(first.rows, first.columns)}"


class TestResolveTargetGridMemory:
    """M2: the explicit-grid template is size-guarded against OOM."""

    def test_oversize_grid_raises(self):
        """An absurd grid (degrees/metres mix-up) raises instead of OOM.

        Test scenario:
            A wide lon/lat bounds at a metre-sized resolution would be billions
            of pixels -> clear ValueError pointing at coarser res / like=.
        """
        with pytest.raises(ValueError, match="exceeding the"):
            # ~ (100 deg / 1e-4) ^2 pixels = 1e12, far over the 250M limit.
            _resolve_target_grid(
                Grid(crs=4326, resolution=0.0001, bounds=(0.0, 0.0, 100.0, 100.0))
            )

    def test_just_under_limit_builds(self):
        """A large-but-allowed grid still builds (S2-tile-scale).

        Test scenario:
            A 5000x5000 grid (25M px) is well under the 250M ceiling.
        """
        tpl = _resolve_target_grid(
            Grid(crs=32633, resolution=10.0, bounds=(0.0, 0.0, 50000.0, 50000.0))
        )
        assert (tpl.rows, tpl.columns) == (
            5000,
            5000,
        ), f"shape: {(tpl.rows, tpl.columns)}"
