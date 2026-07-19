"""Unit tests for the in-memory raster helpers added to :mod:`pyramids._io`.

Covers :func:`pyramids._io.new_vsimem_path`, :func:`pyramids._io.silent_unlink`
and :func:`pyramids._io.bytes_to_gdal` — the plumbing behind
``Dataset.from_bytes`` / ``NetCDF.from_bytes``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
from osgeo import gdal

from pyramids._io import bytes_to_gdal, new_vsimem_path, silent_unlink

pytestmark = pytest.mark.core

GEOTIFF_FIXTURE = "tests/data/acc4000.tif"


@pytest.fixture(scope="module")
def geotiff_bytes() -> bytes:
    """Raw bytes of a small single-band GeoTIFF fixture.

    Returns:
        bytes: Contents of ``tests/data/acc4000.tif``.
    """
    return Path(GEOTIFF_FIXTURE).read_bytes()


class TestNewVsimemPath:
    """Tests for :func:`pyramids._io.new_vsimem_path`."""

    def test_default_suffix(self):
        """A path with no explicit suffix ends in ``.tif`` and lives under ``/vsimem/``.

        Test scenario:
            ``new_vsimem_path()`` — expected: starts with ``/vsimem/`` and
            ends with ``.tif``.
        """
        path = new_vsimem_path()
        assert path.startswith("/vsimem/"), f"expected a /vsimem/ path, got {path!r}"
        assert path.endswith(".tif"), f"expected a .tif suffix, got {path!r}"

    @pytest.mark.parametrize("suffix", [".nc", ".asc", ".img", ".vrt"])
    def test_custom_suffix(self, suffix: str):
        """A custom suffix is appended verbatim.

        Args:
            suffix: Extension to request, including the leading dot.

        Test scenario:
            ``new_vsimem_path(suffix)`` — expected: the returned path ends
            with ``suffix``.
        """
        path = new_vsimem_path(suffix)
        assert path.endswith(suffix), f"expected suffix {suffix!r} in {path!r}"

    def test_successive_calls_are_unique(self):
        """Two consecutive calls never collide.

        Test scenario:
            Call the helper many times — expected: all results distinct so
            concurrent conversions cannot clobber each other.
        """
        paths = {new_vsimem_path() for _ in range(256)}
        assert len(paths) == 256, "new_vsimem_path produced a collision"


class TestSilentUnlink:
    """Tests for :func:`pyramids._io.silent_unlink`."""

    def test_removes_existing_vsimem_file(self):
        """An existing ``/vsimem/`` file is removed.

        Test scenario:
            Create a ``/vsimem/`` file, call ``silent_unlink`` — expected:
            ``gdal.VSIStatL`` reports it gone afterwards.
        """
        path = new_vsimem_path()
        gdal.FileFromMemBuffer(path, b"some bytes")
        assert gdal.VSIStatL(path) is not None, "fixture file was not created"
        silent_unlink(path)
        assert gdal.VSIStatL(path) is None, "silent_unlink did not remove the file"

    def test_missing_path_does_not_raise(self):
        """Unlinking a path that does not exist is a no-op, not an error.

        Test scenario:
            ``silent_unlink`` on a never-created path — expected: returns
            ``None`` without raising (this is what makes it safe inside
            ``weakref.finalize``).
        """
        silent_unlink("/vsimem/this-path-never-existed-xyz.tif")


class TestBytesToGdal:
    """Tests for :func:`pyramids._io.bytes_to_gdal`."""

    def test_opens_geotiff_bytes(self, geotiff_bytes: bytes):
        """Valid GeoTIFF bytes open into a usable GDAL dataset.

        Args:
            geotiff_bytes: Raw bytes of the GeoTIFF fixture.

        Test scenario:
            ``bytes_to_gdal(geotiff_bytes)`` — expected: a ``gdal.Dataset``
            with the fixture's raster size, and a ``/vsimem/`` backing path
            that actually exists.
        """
        ref = gdal.Open(GEOTIFF_FIXTURE)
        src, vsi_path = bytes_to_gdal(geotiff_bytes)
        try:
            assert isinstance(src, gdal.Dataset), (
                f"expected gdal.Dataset, got {type(src)}"
            )
            assert src.RasterXSize == ref.RasterXSize, "column count mismatch"
            assert src.RasterYSize == ref.RasterYSize, "row count mismatch"
            assert vsi_path.startswith("/vsimem/"), (
                f"expected /vsimem/ path, got {vsi_path!r}"
            )
            assert gdal.VSIStatL(vsi_path) is not None, (
                "backing /vsimem/ file is missing"
            )
        finally:
            src = None
            silent_unlink(vsi_path)

    def test_returns_distinct_paths(self, geotiff_bytes: bytes):
        """Each call mints its own ``/vsimem/`` path.

        Args:
            geotiff_bytes: Raw bytes of the GeoTIFF fixture.

        Test scenario:
            Two ``bytes_to_gdal`` calls — expected: different backing paths.
        """
        # Keep both dataset handles alive so the /vsimem/ files stay open while we
        # compare paths, then release them before unlinking (handles look "unused"
        # to static analysis but are held for their GDAL resource lifetime).
        s1, p1 = bytes_to_gdal(geotiff_bytes)
        s2, p2 = bytes_to_gdal(geotiff_bytes)
        try:
            assert p1 != p2, "two calls reused the same /vsimem/ path"
        finally:
            s1 = s2 = None
            silent_unlink(p1)
            silent_unlink(p2)

    @pytest.mark.parametrize("bad", ["not bytes", 123, None, ["bytes"], object()])
    def test_non_bytes_input_raises_type_error(self, bad):
        """Non bytes-like input raises ``TypeError`` before touching ``/vsimem/``.

        Args:
            bad: An object that is not ``bytes`` / ``bytearray`` / ``memoryview``.

        Test scenario:
            ``bytes_to_gdal(bad)`` — expected: ``TypeError`` mentioning
            ``bytes-like``.
        """
        with pytest.raises(TypeError, match="bytes-like") as exc:
            bytes_to_gdal(bad)
        assert "bytes-like" in str(exc.value), f"unexpected message: {exc.value}"

    @pytest.mark.parametrize(
        "payload",
        [b"", b"not a raster at all", bytes(range(64))],
        ids=["empty", "garbage-text", "garbage-binary"],
    )
    def test_unopenable_bytes_raise_value_error_and_clean_up(self, payload: bytes):
        """Bytes GDAL cannot parse raise ``ValueError`` and leak no ``/vsimem/`` file.

        Args:
            payload: Bytes that are not a valid raster.

        Test scenario:
            ``bytes_to_gdal(payload)`` — expected: ``ValueError`` mentioning
            ``suffix``, and no orphaned ``/vsimem/`` entry left behind (the
            helper unlinks on failure).
        """
        before = set(gdal.ReadDir("/vsimem") or [])
        with pytest.raises(ValueError, match="suffix"):
            bytes_to_gdal(payload)
        after = set(gdal.ReadDir("/vsimem") or [])
        assert after.issubset(before), (
            f"bytes_to_gdal leaked /vsimem/ files: {after - before}"
        )

    @pytest.mark.parametrize("bytes_like", ["bytes", "bytearray", "memoryview"])
    def test_accepts_all_bytes_like_types(self, geotiff_bytes: bytes, bytes_like: str):
        """``bytearray`` and ``memoryview`` work, not just ``bytes``.

        Args:
            geotiff_bytes: Raw bytes of the GeoTIFF fixture.
            bytes_like: Which bytes-like wrapper to feed in.

        Test scenario:
            ``bytes_to_gdal`` with each bytes-like flavour — expected: opens
            identically.
        """
        wrapped = {
            "bytes": bytes(geotiff_bytes),
            "bytearray": bytearray(geotiff_bytes),
            "memoryview": memoryview(geotiff_bytes),
        }[bytes_like]
        src, vsi_path = bytes_to_gdal(wrapped)
        try:
            assert src is not None and src.RasterCount >= 1, "dataset did not open"
        finally:
            src = None
            silent_unlink(vsi_path)

    def test_caller_owns_cleanup(self, geotiff_bytes: bytes):
        """The helper does not auto-clean on success — the caller must.

        Args:
            geotiff_bytes: Raw bytes of the GeoTIFF fixture.

        Test scenario:
            Open, drop the dataset reference, GC — expected: the ``/vsimem/``
            file is still present (bytes_to_gdal does not register a
            finalizer; ``Dataset.from_bytes`` does that), and an explicit
            ``silent_unlink`` removes it.
        """
        # Hold then drop the dataset reference so the GC below can reclaim it;
        # the test proves the /vsimem/ file survives that (no finalizer). The
        # handle is used for its resource lifetime (S1481 false positive).
        src, vsi_path = bytes_to_gdal(geotiff_bytes)
        src = None
        gc.collect()
        assert gdal.VSIStatL(vsi_path) is not None, (
            "bytes_to_gdal should not auto-clean on success"
        )
        silent_unlink(vsi_path)
        assert gdal.VSIStatL(vsi_path) is None, "explicit cleanup failed"
