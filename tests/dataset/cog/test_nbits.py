"""Regression tests for sub-byte-aligned ``NBITS`` handling in COG writes.

A band whose ``NBITS`` is not 8/16/32/64 (e.g. the 12 a ``SENTINEL2`` read
propagates) used to (a) make ``to_file(driver="COG")`` fail because ``PREDICTOR=2``
is rejected for that width, and (b) silently clip values above the narrow domain
when the width was inherited onto the output. These tests pin the fix: an
unsigned-integer source is promoted to its dtype's natural width, the predictor is
resolved against that promoted width, float/signed dtypes are left untouched, and
an explicit caller ``NBITS`` still wins (with the predictor reconciled away so the
write does not fail).
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._utils import resolve_cog_predictor
from pyramids.dataset import Dataset
from pyramids.dataset.cog import Compression
from pyramids.base.georeference import GeoReference
from pyramids.dataset.cog.options import (
    _promote_nbits,
    _read_source_nbits,
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


class TestReadSourceNbits:
    """Reading ``NBITS`` from a band in :func:`_read_source_nbits`."""

    def test_reads_integer_nbits(self):
        """A well-formed ``NBITS`` string is returned as an int."""
        _ds, band = _mem_band(gdal.GDT_UInt16, nbits=12)
        assert _read_source_nbits(band) == 12, "should read NBITS=12"

    def test_missing_nbits_is_none(self):
        """A band with no ``NBITS`` metadata returns ``None``."""
        _ds, band = _mem_band(gdal.GDT_UInt16, nbits=None)
        assert _read_source_nbits(band) is None, "no NBITS should read None"

    def test_non_integer_nbits_is_ignored(self):
        """A malformed ``NBITS`` value returns ``None`` rather than raising.

        Test scenario:
            A hand-set non-integer width must not abort the read path.
        """
        ds = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_UInt16)
        ds.GetRasterBand(1).SetMetadataItem("NBITS", "not-a-number", "IMAGE_STRUCTURE")
        assert _read_source_nbits(ds.GetRasterBand(1)) is None, (
            "malformed NBITS -> None"
        )


class TestPromoteNbits:
    """Dtype-aware width promotion in :func:`_promote_nbits`."""

    @pytest.mark.parametrize(
        "nbits, dtype, expected",
        [
            (12, gdal.GDT_UInt16, 16),
            (4, gdal.GDT_UInt16, 16),
            (1, gdal.GDT_Byte, 8),
            (12, gdal.GDT_UInt32, 32),
            (20, gdal.GDT_UInt32, 32),
            (40, gdal.GDT_UInt64, 64),
        ],
    )
    def test_sub_natural_widths_promote_to_dtype_natural(self, nbits, dtype, expected):
        """A sub-natural unsigned width promotes to the dtype's natural width.

        Args:
            nbits: The source width.
            dtype: The source GDAL dtype.
            expected: The natural width promoted to.
        """
        got = _promote_nbits(nbits, dtype)
        assert got == expected, (
            f"promote({nbits},{dtype}) should be {expected}, got {got}"
        )

    @pytest.mark.parametrize(
        "nbits, dtype",
        [(None, gdal.GDT_UInt16), (16, gdal.GDT_UInt16), (32, gdal.GDT_UInt32)],
    )
    def test_natural_or_none_needs_no_promotion(self, nbits, dtype):
        """An already-natural width or ``None`` returns ``None``.

        Args:
            nbits: A width needing no change.
            dtype: The source GDAL dtype.
        """
        assert _promote_nbits(nbits, dtype) is None, f"promote({nbits},{dtype}) -> None"

    @pytest.mark.parametrize(
        "dtype", [gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_Int16]
    )
    def test_non_unsigned_dtypes_are_left_alone(self, dtype):
        """Float and signed-integer dtypes are never promoted (returns ``None``).

        Args:
            dtype: A non-unsigned-integer GDAL dtype.
        """
        assert _promote_nbits(12, dtype) is None, f"promote(12,{dtype}) should be None"


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

    @pytest.mark.parametrize("token", ["NO", "no", "No"])
    def test_noop_for_case_insensitive_disabled_token(self, token):
        """A string ``"NO"`` token (any case) is preserved for a narrow width.

        Args:
            token: A case variant of the disabled-predictor token.
        """
        options = {"NBITS": 12, "PREDICTOR": token}
        _reconcile_predictor_with_nbits(options)
        assert options.get("PREDICTOR") == token, f"predictor should stay: {options}"


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

    def test_uint16_below_one_byte_promotes_to_natural(self):
        """A UInt16 source with ``NBITS<8`` promotes to 16 (not 8) with ``PREDICTOR=2``."""
        ds, band = _mem_band(gdal.GDT_UInt16, nbits=4)
        opts = Compression()._to_options(band)
        assert opts["NBITS"] == 16, (
            f"expected natural NBITS=16, got {opts.get('NBITS')}"
        )
        assert opts["PREDICTOR"] == 2, f"expected PREDICTOR=2, got {opts['PREDICTOR']}"

    def test_uint32_promotes_to_natural_32(self):
        """A UInt32 source with a sub-natural ``NBITS`` promotes to 32, not 16."""
        ds, band = _mem_band(gdal.GDT_UInt32, nbits=12)
        opts = Compression()._to_options(band)
        assert opts["NBITS"] == 32, (
            f"expected natural NBITS=32, got {opts.get('NBITS')}"
        )
        assert opts["PREDICTOR"] == 2, f"expected PREDICTOR=2, got {opts['PREDICTOR']}"

    def test_float_source_emits_no_nbits_and_predictor_3(self):
        """A float source is never promoted: no ``NBITS`` emitted, ``PREDICTOR=3``.

        Test scenario:
            Emitting a promoted ``NBITS`` on a float band would mean half-float and
            corrupt values, so the promotion must skip float dtypes entirely.
        """
        ds, band = _mem_band(gdal.GDT_Float32, nbits=12)
        opts = Compression()._to_options(band)
        assert "NBITS" not in opts, (
            f"float NBITS should not be emitted: {opts.get('NBITS')}"
        )
        assert opts["PREDICTOR"] == 3, (
            f"expected PREDICTOR=3 for float, got {opts['PREDICTOR']}"
        )


def _nbits_source(value, np_dtype, nbits: int, no_data_value) -> Dataset:
    """Build a Dataset of ``np_dtype`` holding ``value`` but tagged with ``nbits``.

    Mirrors a *derived* band whose values exceed its declared narrow domain (e.g.
    Sentinel-2 band math past 4095) while still carrying the inherited ``NBITS``.
    An explicit in-domain ``no_data_value`` keeps the fallback-nodata warning out
    of the suite output.
    """
    arr = np.full((4, 4), value, dtype=np_dtype)
    ds = Dataset.from_array(
             arr,
             no_data_value=no_data_value,
             geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
         )
    ds.raster.GetRasterBand(1).SetMetadataItem("NBITS", str(nbits), "IMAGE_STRUCTURE")
    return ds


class TestCogWriteNbits:
    """End-to-end ``to_file(driver="COG")`` on sub-natural NBITS sources."""

    @pytest.mark.parametrize(
        "value, np_dtype, nbits",
        [
            (5000, np.uint16, 12),  # #1023 Sentinel-2 case (UInt16 9..15)
            (5000, np.uint16, 4),  # UInt16 below one byte (M1)
            (100000, np.uint32, 12),  # UInt32 sub-natural (M1)
        ],
    )
    def test_write_succeeds_and_value_survives(self, value, np_dtype, nbits, tmp_path):
        """A default COG write succeeds and an out-of-domain value round-trips.

        Args:
            value: A value above the declared narrow ``nbits`` domain.
            np_dtype: The source NumPy dtype.
            nbits: The declared sub-natural width.
            tmp_path: pytest temp directory.

        Test scenario:
            The source is promoted to its dtype's natural width so ``PREDICTOR=2``
            is valid and the value is not clipped to the narrow domain.
        """
        src = _nbits_source(value, np_dtype, nbits, no_data_value=0)
        out = tmp_path / "nb.tif"
        src.to_file(out, driver="COG")
        reopened = Dataset.read_file(out)
        peak = int(reopened.read_array().max())
        reopened.close()
        src.close()
        assert peak == value, f"value clipped to {peak}; expected {value}"

    def test_explicit_narrow_nbits_is_honoured(self, tmp_path):
        """An explicit caller ``NBITS=12`` is applied to the output.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Forcing the narrow width would clash with the promoted ``PREDICTOR=2``;
            the write site drops the predictor so the caller's width is honoured
            rather than failing. Uses an in-domain value (3000 <= 4095) so the test
            pins the honoured width, not clipping. The reopened output must report
            ``NBITS=12`` and its in-domain value must round-trip intact.
        """
        src = _nbits_source(3000, np.uint16, 12, no_data_value=0)
        out = tmp_path / "nb12_forced.tif"
        src.to_file(out, driver="COG", creation_options=["NBITS=12"])
        src.close()
        reopened = Dataset.read_file(out)
        band = reopened.raster.GetRasterBand(1)
        applied_nbits = band.GetMetadataItem("NBITS", "IMAGE_STRUCTURE")
        peak = int(reopened.read_array().max())
        reopened.close()
        assert applied_nbits == "12", f"caller NBITS not applied, got {applied_nbits!r}"
        assert peak == 3000, f"in-domain value should round-trip, got {peak}"
