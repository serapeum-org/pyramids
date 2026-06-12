"""Tests for COG.to_cog_bytes / Dataset.to_cog_bytes (in-memory COG encoding).

Covers encoding a Dataset to COG bytes via a ``/vsimem/`` round-trip: the bytes
are a valid TIFF, validate as a COG when written to disk, honour forwarded
``to_cog`` options, and leave no leaked ``/vsimem/`` files behind.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def float_dataset() -> Dataset:
    """A 600x600 Float32 Dataset on EPSG:4326 (large enough for overviews).

    Returns:
        Dataset: An in-memory float32 dataset.
    """
    rng = np.random.default_rng(seed=3)
    arr = (rng.random((600, 600)) * 100.0).astype("float32")
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


def _vsimem_entries() -> list[str]:
    """Return the current top-level /vsimem/ entries (empty list if none).

    Returns:
        list[str]: Names under /vsimem/.
    """
    entries = gdal.ReadDir("/vsimem")
    return list(entries) if entries else []


class TestToCogBytes:
    """Tests for COG.to_cog_bytes."""

    def test_returns_nonempty_tiff_bytes(self, float_dataset):
        """to_cog_bytes returns non-empty bytes with a TIFF byte-order marker.

        Args:
            float_dataset: Fixture float32 Dataset.

        Test scenario:
            The returned blob is non-empty and begins with a TIFF magic marker
            (``II`` little-endian or ``MM`` big-endian).
        """
        blob = float_dataset.to_cog_bytes()
        assert isinstance(blob, bytes), f"expected bytes, got {type(blob)}"
        assert len(blob) > 0, "COG bytes must be non-empty"
        assert blob[:2] in (b"II", b"MM"), f"not a TIFF: leading bytes {blob[:2]!r}"

    def test_bytes_validate_as_cog(self, float_dataset, tmp_path):
        """Bytes written back to disk validate as a COG.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            Persisting the in-memory blob to a file yields a valid COG.
        """
        blob = float_dataset.to_cog_bytes()
        out = tmp_path / "roundtrip.tif"
        out.write_bytes(blob)
        assert (
            Dataset.read_file(str(out)).validate_cog().is_valid
        ), "round-tripped COG bytes must validate"

    def test_forwards_options(self, float_dataset, tmp_path):
        """to_cog_bytes forwards kwargs (e.g. compress) to to_cog.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            Passing compress="ZSTD" produces a COG whose IMAGE_STRUCTURE
            compression is ZSTD.
        """
        blob = float_dataset.to_cog_bytes(compress="ZSTD")
        out = tmp_path / "zstd.tif"
        out.write_bytes(blob)
        ds = gdal.Open(str(out))
        comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
        ds = None
        assert comp == "ZSTD", f"expected ZSTD compression, got {comp}"

    def test_no_vsimem_leak(self, float_dataset):
        """to_cog_bytes unlinks its temporary /vsimem/ file.

        Args:
            float_dataset: Fixture float32 Dataset.

        Test scenario:
            The set of /vsimem/ entries is unchanged after the call (no leaked
            temporary file).
        """
        before = set(_vsimem_entries())
        float_dataset.to_cog_bytes()
        after = set(_vsimem_entries())
        assert after == before, f"leaked /vsimem/ entries: {after - before}"

    def test_equivalent_to_on_disk_write(self, float_dataset, tmp_path):
        """In-memory bytes describe the same COG structure as an on-disk write.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            The blob and a direct to_cog(path) agree on compression, predictor,
            and overview count.
        """
        disk = float_dataset.to_cog(tmp_path / "disk.tif")
        blob = float_dataset.to_cog_bytes()
        mem_path = tmp_path / "mem.tif"
        mem_path.write_bytes(blob)

        info_disk = Dataset.read_file(str(disk)).cog_info()
        info_mem = Dataset.read_file(str(mem_path)).cog_info()
        assert info_mem.compression == info_disk.compression, "compression mismatch"
        assert info_mem.predictor == info_disk.predictor, "predictor mismatch"
        assert (
            info_mem.overview_count == info_disk.overview_count
        ), "overview count mismatch"
