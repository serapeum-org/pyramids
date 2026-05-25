"""Tests for forwarding band tags / colormap / metadata into a COG (PC-2).

Covers stamping dataset metadata, a band-1 colour table, and per-band tags onto
the output — and that doing so never mutates the user's source dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def byte_dataset() -> Dataset:
    """A 32x32 single-band Byte Dataset on EPSG:4326.

    Returns:
        Dataset: An in-memory uint8 dataset.
    """
    arr = (np.arange(32 * 32) % 5).astype("uint8").reshape(32, 32)
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


class TestMetadataForwarding:
    """Tests for to_cog band_tags / colormap / metadata."""

    def test_metadata_round_trips(self, byte_dataset, tmp_path):
        """Dataset-level metadata survives the write.

        Args:
            byte_dataset: Byte fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            metadata={"SOURCE": "unit-test"} is readable on the reopened COG.
        """
        out = byte_dataset.to_cog(tmp_path / "m.tif", metadata={"SOURCE": "unit-test"})
        ds = gdal.Open(str(out))
        got = ds.GetMetadataItem("SOURCE")
        ds = None
        assert got == "unit-test", f"metadata not preserved, got {got!r}"

    def test_colormap_round_trips(self, byte_dataset, tmp_path):
        """A colour table attaches to band 1 and survives the write.

        Args:
            byte_dataset: Byte fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            The reopened COG has a colour table whose entry 1 is red.
        """
        cmap = {0: (0, 0, 0, 255), 1: (255, 0, 0, 255)}
        out = byte_dataset.to_cog(tmp_path / "cm.tif", colormap=cmap)
        ds = gdal.Open(str(out))
        ct = ds.GetRasterBand(1).GetColorTable()
        entry = ct.GetColorEntry(1) if ct is not None else None
        ds = None
        assert ct is not None, "colour table missing on output"
        assert entry[:3] == (255, 0, 0), f"unexpected colour entry {entry}"

    def test_band_tags_round_trip(self, byte_dataset, tmp_path):
        """Per-band tags (0-based) survive the write.

        Args:
            byte_dataset: Byte fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            band_tags={0: {"name": "class"}} appears on band 1 of the COG.
        """
        out = byte_dataset.to_cog(tmp_path / "bt.tif", band_tags={0: {"name": "class"}})
        ds = gdal.Open(str(out))
        got = ds.GetRasterBand(1).GetMetadataItem("name")
        ds = None
        assert got == "class", f"band tag not preserved, got {got!r}"

    def test_source_not_mutated(self, byte_dataset, tmp_path):
        """Stamping metadata does not mutate the user's source dataset.

        Args:
            byte_dataset: Byte fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            After writing with a colormap + metadata, the in-memory source still
            has no colour table and no stamped metadata item.
        """
        byte_dataset.to_cog(
            tmp_path / "x.tif",
            colormap={0: (1, 2, 3, 255)},
            metadata={"STAMP": "yes"},
        )
        src_band = byte_dataset._raster.GetRasterBand(1)
        assert src_band.GetColorTable() is None, "source colour table was mutated"
        assert byte_dataset._raster.GetMetadataItem("STAMP") is None, "source metadata mutated"

    def test_forwarding_produces_valid_cog(self, byte_dataset, tmp_path):
        """A metadata-forwarding write still produces a valid COG.

        Args:
            byte_dataset: Byte fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            colormap + band_tags + metadata together yield a valid COG.
        """
        out = byte_dataset.to_cog(
            tmp_path / "v.tif",
            colormap={0: (0, 0, 0, 255)},
            band_tags={0: {"name": "class"}},
            metadata={"SOURCE": "x"},
        )
        assert Dataset.read_file(str(out)).validate_cog().is_valid, "invalid COG"
