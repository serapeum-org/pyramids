"""Regression tests for sub-byte-aligned ``NBITS`` handling in COG writes.

A band whose ``NBITS`` is not 8/16/32/64 (e.g. the 12 a ``SENTINEL2`` read
propagates) used to (a) make ``to_file(driver="COG")`` fail because ``PREDICTOR=2``
is rejected for that width, and (b) silently clip values above the narrow domain
when the width was inherited onto the output. These tests pin the fix: the width
is promoted to the next libtiff-writable value, the predictor is resolved against
that promoted width, and an explicit caller ``NBITS`` still wins (with the
predictor reconciled away so the write does not fail).
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._utils import resolve_cog_predictor
from pyramids.dataset import Dataset
from pyramids.dataset.cog import Compression
from pyramids.dataset.cog.options import (
    _promote_nbits,
    _reconcile_predictor_with_nbits,
)

pytestmark = pytest.mark.core


def _mem_band(gdal_dtype: int = gdal.GDT_UInt16, nbits: int | None = None):
    """Return a MEM dataset and its band 1, optionally tagged with ``NBITS``."""
    ds = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal_dtype)
    if nbits is not None:
        ds.GetRasterBand(1).SetMetadataItem("NBITS", str(nbits), "IMAGE_STRUCTURE")
    return ds, ds.GetRasterBand(1)


class TestResolveCogPredictor:
    """The ``nbits`` argument of :func:`resolve_cog_predictor`."""

    @pytest.mark.parametrize("nbits", [None, 8, 16, 32, 64])
    def test_integer_predictor_2_for_supported_widths(self, nbits):
        """Integer rasters keep ``PREDICTOR=2`` at predictor-safe widths.

        Args:
            nbits: A width libtiff accepts for the horizontal predictor.
        """
        got = resolve_cog_predictor(gdal.GDT_UInt16, nbits)
        assert got == 2, f"expected 2 for nbits={nbits}, got {got}"

    @pytest.mark.parametrize("nbits", [1, 4, 12, 24, 40])
    def test_integer_falls_back_to_no_predictor_for_narrow_widths(self, nbits):
        """A sub-byte-aligned width drops to ``PREDICTOR=1`` (no predictor).

        Args:
            nbits: A width libtiff rejects for ``PREDICTOR=2``.
        """
        got = resolve_cog_predictor(gdal.GDT_UInt16, nbits)
        assert got == 1, f"expected 1 for nbits={nbits}, got {got}"

    @pytest.mark.parametrize("nbits", [None, 12, 16])
    def test_float_always_predictor_3(self, nbits):
        """Float rasters always use ``PREDICTOR=3`` regardless of width.

        Args:
            nbits: Any width; ignored for float.
        """
        got = resolve_cog_predictor(gdal.GDT_Float32, nbits)
        assert got == 3, f"expected 3 for float, got {got}"


class TestPromoteNbits:
    """Width promotion in :func:`_promote_nbits`."""

    @pytest.mark.parametrize(
        "nbits, expected",
        [(1, 8), (4, 8), (12, 16), (24, 32), (40, 64)],
    )
    def test_narrow_widths_promote_to_next_supported(self, nbits, expected):
        """A narrow width rounds up to the next libtiff-writable width.

        Args:
            nbits: The source width.
            expected: The promoted width.
        """
        got = _promote_nbits(nbits)
        assert got == expected, f"promote({nbits}) should be {expected}, got {got}"

    @pytest.mark.parametrize("nbits", [None, 8, 16, 32, 64])
    def test_supported_or_none_needs_no_promotion(self, nbits):
        """An already-supported width (or ``None``) returns ``None``.

        Args:
            nbits: A width needing no change.
        """
        assert _promote_nbits(nbits) is None, f"promote({nbits}) should be None"

    @pytest.mark.parametrize("nbits", [65, 96, 128])
    def test_width_above_max_supported_returns_none(self, nbits):
        """A width beyond 64 has no wider target and returns ``None`` (no crash).

        Args:
            nbits: A width larger than the largest predictor-safe width.
        """
        assert _promote_nbits(nbits) is None, f"promote({nbits}) should be None"


class TestReconcilePredictorWithNbits:
    """Post-merge predictor reconciliation in :func:`_reconcile_predictor_with_nbits`."""

    def test_drops_predictor_for_narrow_nbits(self):
        """A narrow final ``NBITS`` drops the predictor so GDAL accepts the write.

        Test scenario:
            A caller-forced ``NBITS=12`` with ``PREDICTOR=2`` would be rejected;
            the predictor key is removed.
        """
        options = {"NBITS": 12, "PREDICTOR": 2, "COMPRESS": "DEFLATE"}
        _reconcile_predictor_with_nbits(options)
        assert "PREDICTOR" not in options, f"predictor should be dropped: {options}"

    def test_keeps_predictor_for_supported_nbits(self):
        """A predictor-safe final ``NBITS`` leaves the predictor in place."""
        options = {"NBITS": 16, "PREDICTOR": 2}
        _reconcile_predictor_with_nbits(options)
        assert options.get("PREDICTOR") == 2, f"predictor should stay: {options}"

    def test_noop_without_nbits(self):
        """No ``NBITS`` key leaves the options untouched."""
        options = {"PREDICTOR": 2, "COMPRESS": "ZSTD"}
        _reconcile_predictor_with_nbits(options)
        assert options.get("PREDICTOR") == 2, f"predictor should stay: {options}"

    def test_noop_for_non_integer_nbits(self):
        """A non-integer ``NBITS`` value is ignored (predictor untouched).

        Test scenario:
            A malformed ``NBITS`` that cannot be parsed to int must not raise and
            must leave the predictor in place.
        """
        options = {"NBITS": "not-a-number", "PREDICTOR": 2}
        _reconcile_predictor_with_nbits(options)
        assert options.get("PREDICTOR") == 2, f"predictor should stay: {options}"

    def test_noop_when_predictor_already_disabled(self):
        """A narrow ``NBITS`` with an already-disabled predictor is left as-is.

        Test scenario:
            When the caller already set ``PREDICTOR=1`` (no predictor), a narrow
            width needs no further change.
        """
        options = {"NBITS": 12, "PREDICTOR": 1}
        _reconcile_predictor_with_nbits(options)
        assert options.get("PREDICTOR") == 1, f"predictor should stay 1: {options}"


class TestCompressionToOptionsNbits:
    """``Compression._to_options`` reads and promotes the source ``NBITS``."""

    def test_narrow_nbits_promoted_and_predictor_kept(self):
        """A 12-bit source emits ``NBITS=16`` and keeps ``PREDICTOR=2``.

        Test scenario:
            The promoted width is predictor-safe, so the write is both valid and
            free of clipping.
        """
        ds, band = _mem_band(gdal.GDT_UInt16, nbits=12)
        opts = Compression()._to_options(band)
        assert opts["NBITS"] == 16, (
            f"expected promoted NBITS=16, got {opts.get('NBITS')}"
        )
        assert opts["PREDICTOR"] == 2, f"expected PREDICTOR=2, got {opts['PREDICTOR']}"

    def test_supported_nbits_emits_no_nbits_key(self):
        """A 16-bit source needs no promotion and emits no ``NBITS``."""
        ds, band = _mem_band(gdal.GDT_UInt16, nbits=16)
        opts = Compression()._to_options(band)
        assert "NBITS" not in opts, f"NBITS should be absent, got {opts.get('NBITS')}"
        assert opts["PREDICTOR"] == 2, f"expected PREDICTOR=2, got {opts['PREDICTOR']}"

    def test_no_source_nbits_emits_no_nbits_key(self):
        """A band with no ``NBITS`` metadata emits none and keeps ``PREDICTOR=2``."""
        ds, band = _mem_band(gdal.GDT_UInt16, nbits=None)
        opts = Compression()._to_options(band)
        assert "NBITS" not in opts, f"NBITS should be absent, got {opts.get('NBITS')}"
        assert opts["PREDICTOR"] == 2, f"expected PREDICTOR=2, got {opts['PREDICTOR']}"


def _nbits12_source_with_large_value(value: int = 5000) -> Dataset:
    """Build a UInt16 Dataset holding ``value`` (>4095) but tagged ``NBITS=12``.

    Mirrors a *derived* Sentinel-2 band: the values exceed the 12-bit domain
    (e.g. after band math) yet the band still carries the inherited ``NBITS=12``.
    """
    arr = np.full((4, 4), value, dtype=np.uint16)
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0, 4), cell_size=1.0, epsg=4326
    )
    ds.raster.GetRasterBand(1).SetMetadataItem("NBITS", "12", "IMAGE_STRUCTURE")
    return ds


class TestCogWriteNbits:
    """End-to-end ``to_file(driver="COG")`` on a sub-byte-aligned source."""

    def test_write_succeeds_and_value_survives(self, tmp_path):
        """A default COG write succeeds and a >4095 value round-trips.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The 12-bit source is promoted so ``PREDICTOR=2`` is valid and the
            5000 value is not clipped to 4095.
        """
        src = _nbits12_source_with_large_value(5000)
        out = tmp_path / "nb12.tif"
        src.to_file(out, driver="COG")
        reopened = Dataset.read_file(out)
        peak = int(reopened.read_array().max())
        reopened.close()
        src.close()
        assert peak == 5000, f"value clipped to {peak}; expected 5000"

    def test_explicit_narrow_nbits_is_honoured(self, tmp_path):
        """An explicit caller ``NBITS=12`` still writes (predictor reconciled).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Forcing the narrow width would clash with ``PREDICTOR=2``; the write
            site drops the predictor so the caller's width is honoured without a
            hard failure.
        """
        src = _nbits12_source_with_large_value(5000)
        out = tmp_path / "nb12_forced.tif"
        src.to_file(out, driver="COG", creation_options=["NBITS=12"])
        src.close()
        assert out.exists(), "explicit NBITS=12 write should succeed"
