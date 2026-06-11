"""Tests for `Dataset.to_bytes` and the shared VSI byte read-back helper.

Covers the driver-generic in-memory serializer (`IO.to_bytes` +
`Dataset.to_bytes` facade), its round-trip with `Dataset.from_bytes`, driver
validation, creation options, `/vsimem/` hygiene, and
`pyramids._io.read_vsi_bytes` (also reused by `to_cog_bytes`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from osgeo import gdal

import pyramids.dataset.engines.io as io_engine
from pyramids._io import new_vsimem_path, read_vsi_bytes
from pyramids.dataset import Dataset
from pyramids.dataset.engines.cog import COG

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def ramp_dataset() -> Dataset:
    """A 4x4 float32 ramp on EPSG:4326 with nodata -9999.

    Returns:
        Dataset: In-memory single-band dataset, value == row*4 + col.
    """
    arr = np.arange(16, dtype="float32").reshape(4, 4)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 4), cell_size=1.0, epsg=4326, no_data_value=-9999.0
    )


class TestReadVsiBytes:
    """Tests for the shared read_vsi_bytes helper."""

    def test_reads_back_written_buffer(self):
        """A buffer written to /vsimem/ is read back byte-identical.

        Test scenario:
            FileFromMemBuffer -> read_vsi_bytes round-trips the payload.
        """
        path = new_vsimem_path(".bin")
        gdal.FileFromMemBuffer(path, b"payload-123")
        try:
            assert read_vsi_bytes(path) == b"payload-123", "VSI read-back mismatch"
        finally:
            gdal.Unlink(path)

    def test_missing_path_raises_file_not_found(self):
        """A path that was never written raises FileNotFoundError.

        Test scenario:
            Both the None-return and the UseExceptions RuntimeError shapes of
            GDAL normalise to FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError, match="could not open"):
            read_vsi_bytes("/vsimem/never-written-anywhere.bin")

    def test_empty_file_reads_as_empty_bytes(self):
        """A zero-byte VSI file reads back as b"" instead of crashing.

        Test scenario:
            gdal.VSIFReadL returns None (not b"") for a zero-byte read; the
            helper must normalise that to empty bytes.
        """
        path = new_vsimem_path(".bin")
        gdal.FileFromMemBuffer(path, b"")
        try:
            assert read_vsi_bytes(path) == b"", "empty file should read as b''"
        finally:
            gdal.Unlink(path)

    def test_short_read_raises(self, monkeypatch):
        """A truncated VSIFReadL raises OSError, not a corrupt buffer (L8).

        Test scenario:
            VSIFReadL can return fewer bytes than the file holds without raising;
            the helper must detect the short read and fail loudly rather than
            return a truncated payload.
        """
        path = new_vsimem_path(".bin")
        gdal.FileFromMemBuffer(path, b"payload-bytes")
        real_read = gdal.VSIFReadL
        monkeypatch.setattr(
            gdal,
            "VSIFReadL",
            lambda size, count, handle: (real_read(size, count, handle) or b"")[:3],
        )
        try:
            with pytest.raises(OSError, match="short read"):
                read_vsi_bytes(path)
        finally:
            monkeypatch.undo()
            gdal.Unlink(path)


class TestToBytes:
    """Tests for Dataset.to_bytes."""

    def test_gtiff_round_trip_preserves_everything(self, ramp_dataset):
        """Default GTiff bytes reopen to an identical dataset.

        Test scenario:
            from_bytes(to_bytes(ds)) preserves the array, geotransform, CRS,
            and nodata marker.
        """
        payload = ramp_dataset.to_bytes()
        restored = Dataset.from_bytes(payload)
        np.testing.assert_array_equal(
            restored.read_array(), ramp_dataset.read_array(),
            err_msg="array must survive the bytes round-trip",
        )
        assert restored.geotransform == pytest.approx(ramp_dataset.geotransform), (
            "geotransform changed"
        )
        assert restored.epsg == 4326, f"CRS lost: {restored.epsg}"
        assert restored.no_data_value[0] == pytest.approx(-9999.0), (
            f"nodata lost: {restored.no_data_value}"
        )

    def test_payload_is_a_tiff_file(self, ramp_dataset):
        """The default payload carries TIFF magic bytes.

        Test scenario:
            Little-endian TIFF files start with b"II*\\x00".
        """
        payload = ramp_dataset.to_bytes()
        assert payload[:4] == b"II*\x00", f"not a TIFF payload: {payload[:4]!r}"

    def test_creation_options_compress(self, ramp_dataset):
        """COMPRESS=DEFLATE produces a payload no larger than uncompressed.

        Test scenario:
            Creation options reach the driver; for a constant raster the
            deflate payload is strictly smaller.
        """
        flat = Dataset.create_from_array(
            np.zeros((64, 64), dtype="float32"),
            top_left_corner=(0, 64), cell_size=1.0, epsg=4326,
        )
        compressed = flat.to_bytes(creation_options={"COMPRESS": "DEFLATE"})
        raw = flat.to_bytes()
        assert len(compressed) < len(raw), (
            f"deflate ({len(compressed)}) should beat raw ({len(raw)}) on zeros"
        )

    def test_png_driver(self):
        """A uint8 dataset serializes to a valid PNG payload.

        Test scenario:
            driver="PNG" emits the PNG magic; from_bytes reopens it and the
            pixel values survive.
        """
        ds = Dataset.create_from_array(
            np.full((8, 8), 7, dtype="uint8"),
            top_left_corner=(0, 8), cell_size=1.0, epsg=4326, no_data_value=255,
        )
        payload = ds.to_bytes(driver="PNG")
        assert payload[:4] == b"\x89PNG", f"not a PNG payload: {payload[:4]!r}"
        restored = Dataset.from_bytes(payload, suffix=".png")
        np.testing.assert_array_equal(
            restored.read_array(), ds.read_array(),
            err_msg="PNG round-trip changed pixel values",
        )

    def test_strict_copy_rejects_lossy_driver(self, ramp_dataset):
        """A driver that cannot represent the dtype fails loudly (strict copy).

        Test scenario:
            PNG supports Byte/UInt16 only; serializing a float32 dataset must
            raise (no silent downcast) and leave /vsimem/ clean.
        """
        before = sorted(gdal.ReadDir("/vsimem/") or [])
        with pytest.raises(RuntimeError, match="PNG driver"):
            ramp_dataset.to_bytes(driver="PNG")
        after = sorted(gdal.ReadDir("/vsimem/") or [])
        assert before == after, f"vsimem leaked: {set(after) - set(before)}"

    def test_unknown_driver_raises(self, ramp_dataset):
        """An unknown driver name raises ValueError naming the driver.

        Test scenario:
            GetDriverByName returns None -> clear error, nothing written.
        """
        with pytest.raises(ValueError, match="unknown GDAL driver"):
            ramp_dataset.to_bytes(driver="definitely-not-a-driver")

    def test_driver_without_createcopy_raises(self, ramp_dataset):
        """A driver lacking the CreateCopy capability is rejected up-front.

        Test scenario:
            The MEM driver advertises no DCAP_CREATECOPY -> ValueError before
            anything is written to /vsimem/.
        """
        with pytest.raises(ValueError, match="does not support CreateCopy"):
            ramp_dataset.to_bytes(driver="MEM")

    def test_multi_file_driver_raises_and_cleans_up(self, ramp_dataset):
        """A driver that writes sidecar files is rejected, leaving no leaks.

        Test scenario:
            AAIGrid emits a .prj sidecar next to the .asc for a georeferenced
            raster -> ValueError naming the sibling, and every /vsimem/ entry
            (main file + sidecar) is swept by the cleanup.
        """
        before = sorted(gdal.ReadDir("/vsimem/") or [])
        with pytest.raises(ValueError, match="multi-file output"):
            ramp_dataset.to_bytes(driver="AAIGrid")
        after = sorted(gdal.ReadDir("/vsimem/") or [])
        assert before == after, f"vsimem leaked: {set(after) - set(before)}"

    def test_vsimem_is_clean_after_success(self, ramp_dataset):
        """No /vsimem/ entries leak after a successful serialization.

        Test scenario:
            The /vsimem/ directory listing is unchanged by to_bytes.
        """
        before = sorted(gdal.ReadDir("/vsimem/") or [])
        ramp_dataset.to_bytes()
        after = sorted(gdal.ReadDir("/vsimem/") or [])
        assert before == after, f"vsimem leaked: {set(after) - set(before)}"

    def test_vsimem_is_clean_after_failure(self, ramp_dataset, monkeypatch):
        """No /vsimem/ entries leak when the serialization fails mid-way.

        Test scenario:
            The read-back step is forced to fail after the driver wrote the
            virtual file; the finally-cleanup must still unlink every entry.
        """
        def _boom(path):
            raise FileNotFoundError("forced read-back failure")

        monkeypatch.setattr(io_engine, "read_vsi_bytes", _boom)
        before = sorted(gdal.ReadDir("/vsimem/") or [])
        with pytest.raises(FileNotFoundError, match="forced read-back"):
            ramp_dataset.to_bytes()
        after = sorted(gdal.ReadDir("/vsimem/") or [])
        assert before == after, f"vsimem leaked on failure: {set(after) - set(before)}"

    def test_multi_band_round_trip(self):
        """A 3-band raster survives the bytes round-trip.

        Test scenario:
            Band count and every band's values are preserved.
        """
        arr = np.random.default_rng(7).random((3, 5, 5)).astype("float32")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
        )
        restored = Dataset.from_bytes(ds.to_bytes())
        assert restored.band_count == 3, f"band count lost: {restored.band_count}"
        np.testing.assert_array_equal(
            restored.read_array(), arr, err_msg="multi-band values changed"
        )

    def test_facade_delegates_to_engine(self, ramp_dataset):
        """Dataset.to_bytes and IO.to_bytes return identical payloads.

        Test scenario:
            The facade is a pure delegation of the engine method.
        """
        assert ramp_dataset.to_bytes() == ramp_dataset.io.to_bytes(), (
            "facade and engine outputs differ"
        )

    def test_concurrent_to_bytes_are_independent(self):
        """Parallel to_bytes calls do not cross-contaminate (M2).

        Test scenario:
            Each call serializes into its own unique /vsimem/ subdirectory, so
            eight threads each encoding a distinct constant raster must every one
            round-trip back to its own values — a global prefix scan could have
            unlinked or mis-detected another call's files.
        """
        datasets = [
            Dataset.create_from_array(
                np.full((4, 4), value, dtype="float32"),
                top_left_corner=(0, 4),
                cell_size=1.0,
                epsg=4326,
            )
            for value in range(8)
        ]
        results: dict[int, bytes] = {}

        def encode(i: int) -> None:
            results[i] = datasets[i].to_bytes()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(encode, range(8)))

        for value, payload in results.items():
            restored = Dataset.from_bytes(payload)
            assert np.allclose(restored.read_array(), float(value)), (
                f"payload {value} did not round-trip to its own values"
            )


class TestToCogBytesStillWorks:
    """Regression: to_cog_bytes survives the shared read-back refactor."""

    def test_cog_bytes_round_trip(self, ramp_dataset):
        """to_cog_bytes still emits a reopenable TIFF after the refactor.

        Test scenario:
            The refactored read-back path produces a payload from_bytes can
            open with the original values.
        """
        payload = ramp_dataset.to_cog_bytes()
        assert payload[:4] in (b"II*\x00", b"MM\x00*"), "not a TIFF payload"
        restored = Dataset.from_bytes(payload)
        np.testing.assert_array_equal(
            restored.read_array(), ramp_dataset.read_array(),
            err_msg="COG bytes round-trip changed values",
        )

    def test_to_cog_failure_is_not_masked_by_cleanup(self, ramp_dataset, monkeypatch):
        """An early to_cog failure propagates instead of a cleanup error.

        Test scenario:
            to_cog raises before the /vsimem/ file exists; the finally-cleanup
            must not replace the original exception with the RuntimeError that
            gdal.Unlink raises on a missing path under gdal.UseExceptions().
        """
        def _boom(self, path, **kwargs):
            raise ValueError("forced to_cog failure")

        monkeypatch.setattr(COG, "to_cog", _boom)
        with pytest.raises(ValueError, match="forced to_cog failure"):
            ramp_dataset.to_cog_bytes()
