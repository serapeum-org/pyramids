"""Tests for the shared ``resolve_read_window`` bbox/window resolver.

Covers the core that ``Dataset.read_array`` and ``NetCDF._resolve_bbox_to_window``
now share: the ``bbox``/``window`` mutual-exclusivity check and the folding of a
``bbox`` into a one-row ``FeatureCollection`` in the caller-injected CRS.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset.engines._read_window import resolve_read_window
from pyramids.dataset.window import Window
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


class TestResolveReadWindow:
    """``resolve_read_window(window, bbox, *, crs)`` behaviour."""

    def test_no_bbox_returns_window_unchanged(self):
        """With ``bbox=None`` the caller's ``window`` passes straight through.

        Test scenario:
            A pixel ``Window`` and ``bbox=None`` returns the very same object.
        """
        window = Window(1, 2, 3, 4)
        result = resolve_read_window(window, None, crs=4326)
        assert result is window, "window must be returned unchanged when bbox is None"

    def test_no_bbox_no_window_returns_none(self):
        """With both ``bbox`` and ``window`` ``None`` the result is ``None``.

        Test scenario:
            A full read (no window, no bbox) resolves to ``None``.
        """
        result = resolve_read_window(None, None, crs=4326)
        assert result is None, f"expected None for a full read, got {result!r}"

    def test_bbox_and_window_together_raises(self):
        """Passing both ``bbox`` and ``window`` is a ``ValueError``.

        Test scenario:
            The exclusivity guard fires with the shared "not both" message.
        """
        window = Window(0, 0, 2, 2)
        with pytest.raises(ValueError, match="either .*window.* or .*bbox.*not both"):
            resolve_read_window(window, (0, 0, 1, 1), crs=4326)

    def test_bbox_builds_feature_collection_in_crs(self):
        """A ``bbox`` alone folds into a one-row ``FeatureCollection`` covering it.

        Test scenario:
            ``bbox=(0, 0, 2, 3)`` with ``crs=4326`` returns a FeatureCollection
            whose single geometry's bounds equal the bbox in EPSG:4326.
        """
        bbox = (0.0, 0.0, 2.0, 3.0)
        result = resolve_read_window(None, bbox, crs=4326)
        assert isinstance(result, FeatureCollection), (
            f"a bbox must resolve to a FeatureCollection, got {type(result)}"
        )
        assert len(result) == 1, "bbox should yield exactly one feature row"
        np.testing.assert_allclose(
            result.total_bounds,
            np.array(bbox),
            err_msg="the feature's bounds must equal the requested bbox",
        )

    def test_bbox_forwards_injected_crs(self, mocker):
        """The injected ``crs`` is forwarded verbatim to ``from_bbox``.

        Test scenario:
            ``resolve_read_window`` must pass its ``crs`` argument through as
            ``FeatureCollection.from_bbox(bbox, epsg=crs)`` — the mechanism that
            lets IO and NetCDF inject different CRS conventions.
        """
        sentinel = object()
        spy = mocker.patch.object(
            FeatureCollection, "from_bbox", return_value=sentinel
        )
        bbox = (0.0, 0.0, 1.0, 1.0)
        result = resolve_read_window(None, bbox, crs="EPSG:3857")
        assert result is sentinel, "the from_bbox result must be returned as-is"
        spy.assert_called_once_with(bbox, epsg="EPSG:3857")
