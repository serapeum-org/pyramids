"""Tests for the documented larger-than-RAM / streaming COG write path (PC-3).

PC-3 is primarily a documentation task (the honest note on to_cog about >RAM and
dask writes). These lock in the two behaviours the note relies on: an on-disk
source COG-encodes via GDAL streaming, and to_file(compute=False) refuses a
MEM-only dataset early.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


class TestStreamingWrite:
    """Tests for the >RAM / deferred write contract."""

    def test_on_disk_source_streams_to_cog(self, tmp_path):
        """A COG can be written from an on-disk source (GDAL streams it).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Write a plain GeoTIFF, reopen it (on-disk backing), then to_cog —
            GDAL streams from the source and the result validates.
        """
        arr = (np.random.default_rng(1).random((600, 600)) * 100).astype("float32")
        plain = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
        plain_path = tmp_path / "plain.tif"
        plain.to_file(plain_path)

        on_disk = Dataset.read_file(str(plain_path))
        out = on_disk.to_cog(tmp_path / "streamed.tif")
        assert (
            Dataset.read_file(str(out)).validate_cog().is_valid
        ), "on-disk source should COG-encode to a valid COG"

    def test_compute_false_refuses_mem_dataset(self, tmp_path):
        """to_file(compute=False) raises early for a MEM-only dataset.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A fresh in-memory Dataset has no on-disk anchor, so a deferred write
            cannot be scheduled — it must fail fast rather than at compute time.
        """
        import pickle

        arr = np.ones((16, 16), dtype="float32")
        mem = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
        with pytest.raises(pickle.PicklingError):
            mem.to_file(tmp_path / "x.tif", compute=False)
