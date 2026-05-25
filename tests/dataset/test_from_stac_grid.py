"""Tests for PC-2 grid match: from_stac(like=/crs=/resolution=/bounds=)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
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
        items.append({"id": f"i{i}", "bbox": [0.0, 0.0, 4.0, 4.0], "assets": {"data": {"href": p}}})
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


class TestResolveTargetGrid:
    """Tests for the _resolve_target_grid helper."""

    def test_none_when_nothing_requested(self):
        """No grid args -> None (no alignment).

        Test scenario:
            All grid params absent.
        """
        assert _resolve_target_grid(None, None, None, None, "edge") is None

    def test_like_returned_directly(self, template):
        """A like Dataset is returned as the template.

        Test scenario:
            like= passes through unchanged.
        """
        assert _resolve_target_grid(template, None, None, None, "edge") is template

    def test_like_with_explicit_raises(self, template):
        """like= combined with crs/resolution/bounds raises.

        Test scenario:
            Mutually-exclusive grid specs are rejected.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_target_grid(template, 4326, 1.0, (0, 0, 4, 4), "edge")

    def test_partial_explicit_raises(self):
        """An incomplete crs/resolution/bounds trio raises.

        Test scenario:
            crs + resolution without bounds is rejected.
        """
        with pytest.raises(ValueError, match="all be given together"):
            _resolve_target_grid(None, 4326, 1.0, None, "edge")

    def test_explicit_builds_snapped_template(self):
        """An explicit grid builds a template snapped to the resolution.

        Test scenario:
            bounds (0.4, 0.4, 3.6, 3.6) at resolution 1 snap to (0,0,4,4) -> 4x4.
        """
        tpl = _resolve_target_grid(None, 4326, 1.0, (0.4, 0.4, 3.6, 3.6), "edge")
        assert tpl.epsg == 4326, f"epsg: {tpl.epsg}"
        assert (tpl.rows, tpl.columns) == (4, 4), f"shape: {(tpl.rows, tpl.columns)}"

    def test_unknown_anchor_raises(self):
        """An unsupported anchor raises.

        Test scenario:
            anchor='center' is not implemented.
        """
        with pytest.raises(ValueError, match="anchor must be"):
            _resolve_target_grid(None, 4326, 1.0, (0, 0, 4, 4), "center")


class TestFromStacGridMatch:
    """Tests for from_stac grid match via like= / explicit grid."""

    def test_like_matches_grid(self, offset_grid_items, template):
        """like= resamples every timestep onto the template's grid.

        Test scenario:
            Coarse 2x2 items -> a 4x4 cube matching the template.
        """
        coll = DatasetCollection.from_stac(offset_grid_items, asset="data", like=template)
        assert coll.time_length == 2, f"expected 2 timesteps, got {coll.time_length}"
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (4, 4), f"not aligned to template: {(first.rows, first.columns)}"
        assert first.epsg == 4326, f"epsg: {first.epsg}"

    def test_explicit_grid_matches(self, offset_grid_items):
        """crs/resolution/bounds build and match an explicit target grid.

        Test scenario:
            A 1-degree grid over (0,0,4,4) yields 4x4 timesteps.
        """
        coll = DatasetCollection.from_stac(
            offset_grid_items, asset="data", crs=4326, resolution=1.0, bounds=(0.0, 0.0, 4.0, 4.0)
        )
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (4, 4), f"shape: {(first.rows, first.columns)}"

    def test_no_grid_match_keeps_native(self, offset_grid_items):
        """Without like/crs the cube keeps the native (coarse) grid.

        Test scenario:
            No grid args -> 2x2 native timesteps (back-compat).
        """
        coll = DatasetCollection.from_stac(offset_grid_items, asset="data")
        first = coll.datasets[0]
        assert (first.rows, first.columns) == (2, 2), f"should stay native 2x2: {(first.rows, first.columns)}"


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
            _resolve_target_grid(None, 4326, 0.0001, (0.0, 0.0, 100.0, 100.0), "edge")

    def test_just_under_limit_builds(self):
        """A large-but-allowed grid still builds (S2-tile-scale).

        Test scenario:
            A 5000x5000 grid (25M px) is well under the 250M ceiling.
        """
        tpl = _resolve_target_grid(None, 32633, 10.0, (0.0, 0.0, 50000.0, 50000.0), "edge")
        assert (tpl.rows, tpl.columns) == (5000, 5000), f"shape: {(tpl.rows, tpl.columns)}"
