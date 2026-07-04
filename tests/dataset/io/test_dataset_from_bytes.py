"""Tests for :meth:`pyramids.dataset.Dataset.from_bytes`."""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

SINGLE_BAND = "tests/data/acc4000.tif"
MULTI_BAND = "tests/data/geotiff/multi_bands.tif"


@pytest.fixture(scope="module")
def single_band_bytes() -> bytes:
    """Raw bytes of a single-band GeoTIFF fixture.

    Returns:
        bytes: Contents of ``tests/data/acc4000.tif``.
    """
    return Path(SINGLE_BAND).read_bytes()


@pytest.fixture(scope="module")
def multi_band_bytes() -> bytes:
    """Raw bytes of a multi-band GeoTIFF fixture.

    Returns:
        bytes: Contents of ``tests/data/geotiff/multi_bands.tif``.
    """
    return Path(MULTI_BAND).read_bytes()


class TestDatasetFromBytes:
    """Tests for :meth:`Dataset.from_bytes`."""

    def test_round_trip_matches_read_file_single_band(self, single_band_bytes: bytes):
        """A single-band GeoTIFF round-trips identically to ``read_file``.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``Dataset.from_bytes(bytes)`` vs ``Dataset.read_file(path)`` —
            expected: same shape, epsg, cell size, geotransform, no-data,
            band names, and pixel values.
        """
        ref = Dataset.read_file(SINGLE_BAND)
        ds = Dataset.from_bytes(single_band_bytes)
        assert ds.shape == ref.shape, f"shape mismatch: {ds.shape} != {ref.shape}"
        assert ds.epsg == ref.epsg, f"epsg mismatch: {ds.epsg} != {ref.epsg}"
        assert ds.cell_size == ref.cell_size, "cell size mismatch"
        assert ds.geotransform == ref.geotransform, "geotransform mismatch"
        assert ds.no_data_value == ref.no_data_value, "no-data mismatch"
        assert ds.band_names == ref.band_names, "band names mismatch"
        assert np.array_equal(
            ds.read_array(), ref.read_array(), equal_nan=True
        ), "pixel values differ from read_file"

    def test_round_trip_matches_read_file_multi_band(self, multi_band_bytes: bytes):
        """A multi-band GeoTIFF round-trips identically to ``read_file``.

        Args:
            multi_band_bytes: Raw bytes of the multi-band fixture.

        Test scenario:
            ``Dataset.from_bytes`` on multi-band bytes — expected: band count
            and per-band arrays match ``read_file``.
        """
        ref = Dataset.read_file(MULTI_BAND)
        ds = Dataset.from_bytes(multi_band_bytes)
        assert ds.band_count == ref.band_count, "band count mismatch"
        assert np.array_equal(
            ds.read_array(), ref.read_array(), equal_nan=True
        ), "multi-band pixel values differ from read_file"

    def test_returns_dataset_instance(self, single_band_bytes: bytes):
        """The result is a :class:`Dataset` (not a bare ``gdal.Dataset``).

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``type(Dataset.from_bytes(...))`` — expected: ``Dataset``.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds)}"
        assert type(ds) is Dataset, f"expected exactly Dataset, got {type(ds)}"

    def test_backing_path_is_vsimem(self, single_band_bytes: bytes):
        """The dataset is backed by a ``/vsimem/`` path that exists.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            Inspect ``_vsimem_path`` — expected: starts with ``/vsimem/`` and
            ``gdal.VSIStatL`` reports it present while the object is alive.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        assert ds._vsimem_path.startswith(
            "/vsimem/"
        ), f"bad backing path: {ds._vsimem_path!r}"
        assert (
            gdal.VSIStatL(ds._vsimem_path) is not None
        ), "backing /vsimem/ file is missing"

    def test_default_file_name_is_the_vsimem_path(self, single_band_bytes: bytes):
        """Without ``name=``, ``file_name`` is the GDAL description (the ``/vsimem/`` path).

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``Dataset.from_bytes(bytes).file_name`` — expected: equals the
            backing ``/vsimem/`` path.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        assert (
            ds.file_name == ds._vsimem_path
        ), f"unexpected file_name: {ds.file_name!r}"

    def test_name_argument_sets_file_name(self, single_band_bytes: bytes):
        """``name=`` overrides the cosmetic ``file_name``.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``Dataset.from_bytes(bytes, name="scene-A")`` — expected:
            ``file_name == "scene-A"`` while the backing path is unchanged.
        """
        ds = Dataset.from_bytes(single_band_bytes, name="scene-A")
        assert ds.file_name == "scene-A", f"name= not applied: {ds.file_name!r}"
        assert ds._vsimem_path.startswith(
            "/vsimem/"
        ), "backing path should still be /vsimem/"

    def test_read_only_by_default(self, single_band_bytes: bytes):
        """The dataset opens read-only unless asked otherwise.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``Dataset.from_bytes(bytes)`` — expected: ``access == "read_only"``.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        assert ds.access == "read_only", f"expected read_only, got {ds.access!r}"

    def test_read_only_false_gives_write_access(self, single_band_bytes: bytes):
        """``read_only=False`` yields a writable dataset.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``Dataset.from_bytes(bytes, read_only=False)`` — expected:
            ``access == "write"`` and a band write succeeds (``/vsimem/`` is
            always writable at the GDAL level).
        """
        ds = Dataset.from_bytes(single_band_bytes, read_only=False)
        assert ds.access == "write", f"expected write, got {ds.access!r}"
        arr = ds.read_array(band=0)
        ds.raster.GetRasterBand(1).WriteArray(np.zeros_like(arr))
        ds.raster.FlushCache()
        assert np.array_equal(
            ds.read_array(band=0), np.zeros_like(arr)
        ), "write did not take effect"

    def test_vsimem_cleaned_up_on_gc(self, single_band_bytes: bytes):
        """Dropping the last reference removes the ``/vsimem/`` file.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            Capture ``_vsimem_path``, ``del`` the dataset, ``gc.collect()`` —
            expected: ``gdal.VSIStatL`` reports the file gone (``weakref.finalize``
            ran), even though ``close()`` was never called.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        vsi_path = ds._vsimem_path
        assert gdal.VSIStatL(vsi_path) is not None, "precondition: file should exist"
        del ds
        gc.collect()
        assert gdal.VSIStatL(vsi_path) is None, "/vsimem/ file was not cleaned up on GC"

    def test_close_then_gc_still_cleans_up(self, single_band_bytes: bytes):
        """Calling ``close()`` first does not break finalizer cleanup.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``close()`` then drop + GC — expected: no error, and the
            ``/vsimem/`` file is still removed (``silent_unlink`` is idempotent).
        """
        ds = Dataset.from_bytes(single_band_bytes)
        vsi_path = ds._vsimem_path
        ds.close()
        del ds
        gc.collect()
        assert (
            gdal.VSIStatL(vsi_path) is None
        ), "/vsimem/ file lingered after close + GC"

    def test_multiple_instances_independent(self, single_band_bytes: bytes):
        """Two ``from_bytes`` datasets get independent backing files.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            Build two, GC the first — expected: distinct paths, and GC of one
            does not remove the other's file.
        """
        a = Dataset.from_bytes(single_band_bytes)
        b = Dataset.from_bytes(single_band_bytes)
        pa, pb = a._vsimem_path, b._vsimem_path
        assert pa != pb, "two datasets shared a /vsimem/ path"
        del a
        gc.collect()
        assert gdal.VSIStatL(pa) is None, "first dataset's file should be gone"
        assert gdal.VSIStatL(pb) is not None, "second dataset's file must be untouched"

    def test_in_memory_dataset_is_not_picklable(self, single_band_bytes: bytes):
        """A ``from_bytes`` dataset cannot be pickled (no on-disk path).

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``pickle.dumps(Dataset.from_bytes(...))`` — expected: ``TypeError``
            telling the caller to ``.to_file(path)`` first.
        """
        ds = Dataset.from_bytes(single_band_bytes)
        with pytest.raises(TypeError, match=r"to_file") as exc:
            pickle.dumps(ds)
        assert "/vsimem/" in str(exc.value) or "in-memory" in str(
            exc.value
        ), f"unexpected pickle error: {exc.value}"

    def test_to_file_anchors_to_disk(self, single_band_bytes: bytes, tmp_path: Path):
        """``to_file`` writes the in-memory raster to disk and it round-trips.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            ``from_bytes`` → ``to_file`` → ``read_file`` — expected: a real
            file on disk whose pixels equal the original bytes' raster.
        """
        out = tmp_path / "anchored.tif"
        ds = Dataset.from_bytes(single_band_bytes)
        original = ds.read_array()
        ds.to_file(str(out))
        assert out.exists(), "to_file did not create the output file"
        reloaded = Dataset.read_file(str(out))
        assert np.array_equal(
            reloaded.read_array(), original, equal_nan=True
        ), "disk copy differs"

    @pytest.mark.parametrize("bad", ["a string", 42, None, ["bytes"]])
    def test_non_bytes_raises_type_error(self, bad):
        """Non bytes-like input raises ``TypeError``.

        Args:
            bad: An object that is not bytes-like.

        Test scenario:
            ``Dataset.from_bytes(bad)`` — expected: ``TypeError`` mentioning
            ``bytes-like``.
        """
        with pytest.raises(TypeError, match="bytes-like"):
            Dataset.from_bytes(bad)

    @pytest.mark.parametrize(
        "payload", [b"", b"definitely not a tiff"], ids=["empty", "garbage"]
    )
    def test_unopenable_bytes_raise_value_error(self, payload: bytes):
        """Corrupt / truncated bytes raise ``ValueError`` with a ``suffix`` hint.

        Args:
            payload: Bytes that are not a valid raster.

        Test scenario:
            ``Dataset.from_bytes(payload)`` — expected: ``ValueError`` whose
            message mentions ``suffix``.
        """
        with pytest.raises(ValueError, match="suffix"):
            Dataset.from_bytes(payload)

    def test_unopenable_bytes_do_not_leak_vsimem(self):
        """A failed open leaves no orphaned ``/vsimem/`` file.

        Test scenario:
            Snapshot ``/vsimem`` listing, attempt a bad open, snapshot again —
            expected: no new entries.
        """
        before = set(gdal.ReadDir("/vsimem") or [])
        with pytest.raises(ValueError):
            Dataset.from_bytes(b"not a raster")
        after = set(gdal.ReadDir("/vsimem") or [])
        assert after.issubset(before), f"leaked /vsimem/ files: {after - before}"

    def test_works_as_context_manager(self, single_band_bytes: bytes):
        """The dataset honours the ``with`` protocol like any other ``Dataset``.

        Args:
            single_band_bytes: Raw bytes of the single-band fixture.

        Test scenario:
            ``with Dataset.from_bytes(bytes) as ds: ...`` — expected: usable
            inside the block; closed afterwards; ``/vsimem/`` file gone after
            the object is GC'd.
        """
        with Dataset.from_bytes(single_band_bytes) as ds:
            vsi_path = ds._vsimem_path
            assert ds.read_array() is not None, "dataset unusable inside the with-block"
        assert ds.raster is None, "context manager exit should have closed the dataset"
        del ds
        gc.collect()
        assert gdal.VSIStatL(vsi_path) is None, "/vsimem/ file should be cleaned up"
