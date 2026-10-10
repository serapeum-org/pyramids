"""Self-tests for the shared STAC fixtures, so the fixtures themselves are covered."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestSharedStacFixtures:
    """The shared fixtures produce the shapes the STAC tasks rely on."""

    def test_packed_raster_holds_stored_counts(self, packed_raster_path):
        """The packed raster is written as raw int16 counts, not physical values.

        Args:
            packed_raster_path: The packed int16 raster.

        Test scenario:
            `rescale` tests assert an exact physical result, which only holds if
            the file stores raw counts and the scale lives in the STAC item.
        """
        values = Dataset.read_file(packed_raster_path).read_array()
        assert values.dtype.kind in "iu", f"expected integer counts, got {values.dtype}"
        assert sorted(np.unique(values).tolist()) == [0, 100, 200, 300], (
            f"expected stored counts [0,100,200,300], got {np.unique(values)}"
        )

    def test_packed_item_declares_scale(self, packed_item):
        """The item carries the scale in `raster:bands`, not in the file.

        Args:
            packed_item: Item over the packed raster.

        Test scenario:
            This is the configuration `rescale` has to read — a scale that GDAL
            will not apply on its own.
        """
        band = packed_item["assets"]["data"]["raster:bands"][0]
        assert band["scale"] == 0.01, f"expected scale 0.01, got {band['scale']}"
        assert band["nodata"] == 0, f"expected nodata 0, got {band['nodata']}"

    def test_three_local_items_shape(self, three_local_items):
        """Three items, distinct datetimes, two orbits, readable hrefs.

        Args:
            three_local_items: The three-item fixture.

        Test scenario:
            A property `groupby` must collapse these to two groups, so the orbit
            split and the datetime ordering both matter.
        """
        assert len(three_local_items) == 3, (
            f"expected 3 items, got {len(three_local_items)}"
        )
        orbits = [item["properties"]["orbit"] for item in three_local_items]
        assert orbits == [1, 1, 2], f"expected orbits [1,1,2], got {orbits}"
        stamps = [item["properties"]["datetime"] for item in three_local_items]
        assert len(set(stamps)) == 3, f"datetimes must be distinct, got {stamps}"
        for item in three_local_items:
            href = item["assets"]["data"]["href"]
            assert Dataset.read_file(href).shape[-2:] == (3, 3), f"{href} is not 3x3"
