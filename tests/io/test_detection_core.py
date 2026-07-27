"""Tests for the shared format-detection core in `pyramids._resource`.

The magic-byte and extension tables used to be duplicated between `_resource`
and `pyramids.io.sniff`, which let the two drift. These tests pin the single
core: that both public readers resolve through it, that magic bytes beat a lying
extension, and that `read_resource` can fall back to bytes when a name carries no
usable suffix.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pyramids._resource import (
    _EXT_TO_FORMAT,
    read_resource,
    sniff_format,
    sniff_magic,
)

_GEOTIFF = "tests/data/geotiff/era5_land_monthly_averaged.tif"


class TestSharedCore:
    """Tests that both readers resolve through one detection implementation."""

    def test_io_sniff_format_is_the_core_function(self):
        """`pyramids.io.sniff_format` is the very object defined in `_resource`.

        Test scenario:
            Identity — not merely equal behaviour — proves there is a single
            implementation rather than two that can drift.
        """
        from pyramids.io import sniff_format as exported

        assert exported is sniff_format, "pyramids.io must re-export the core sniffer, not reimplement it"

    def test_primary_exts_track_the_shared_table(self):
        """`_PRIMARY_EXTS` is derived from the shared extension table.

        Test scenario:
            Every non-archive extension in the shared table must be treated as a
            primary member, so a newly added format is picked up automatically.
        """
        from pyramids.io.sniff import _PRIMARY_EXTS

        expected = {ext for ext, fmt in _EXT_TO_FORMAT.items() if fmt != "zip"}
        assert set(_PRIMARY_EXTS) == expected, f"_PRIMARY_EXTS drifted from the shared table: {_PRIMARY_EXTS}"


class TestSniffMagic:
    """Tests for `sniff_magic`."""

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (b"PK\x03\x04rest", "zip"),
            (b"II*\x00rest", "tif"),
            (b"MM\x00*rest", "tif"),
            (b"\x89HDF\r\n\x1a\n", "nc"),
            (b"CDF\x01rest", "nc"),
            (b"GRIB0000", "grib"),
            (b"PAR1data", "parquet"),
            (b"SQLite format 3\x00", "gpkg"),
        ],
    )
    def test_recognised_signatures(self, tmp_path: Path, payload: bytes, expected: str):
        """Each known signature maps to its format token.

        Args:
            tmp_path: pytest temporary directory.
            payload: Leading bytes written to the probe file.
            expected: Format token the sniffer should return.

        Test scenario:
            Only the first 16 bytes are read, so a stub file carrying just the
            signature is enough to classify it.
        """
        probe = tmp_path / "probe.bin"
        probe.write_bytes(payload)
        assert sniff_magic(probe) == expected, f"expected {expected} for {payload[:8]!r}"

    def test_unrecognised_payload_returns_none(self, tmp_path: Path):
        """Unknown leading bytes yield `None` so the caller can fall back.

        Test scenario:
            `sniff_magic` reports "I don't know" rather than guessing, leaving the
            extension fallback to `sniff_format`.
        """
        probe = tmp_path / "plain.bin"
        probe.write_bytes(b"just some text")
        assert sniff_magic(probe) is None, "unrecognised bytes should return None"

    def test_missing_file_returns_none(self, tmp_path: Path):
        """A missing path is reported as unknown rather than raising.

        Test scenario:
            Detection must stay non-fatal for callers probing untrusted input.
        """
        assert sniff_magic(tmp_path / "absent.bin") is None, "a missing file should return None"


class TestSniffFormat:
    """Tests for `sniff_format`."""

    def test_magic_beats_a_lying_extension(self, tmp_path: Path):
        """Magic bytes win over a mis-named file.

        Test scenario:
            A zip archive named `.csv` — the portal case this core exists for —
            must still be classified as a zip.
        """
        liar = tmp_path / "resource.csv"
        with zipfile.ZipFile(liar, "w") as archive:
            archive.writestr("inner.txt", "data")
        assert sniff_format(liar) == "zip", "magic bytes should override the extension"

    def test_extension_used_when_bytes_are_inconclusive(self, tmp_path: Path):
        """The extension is the fallback when the bytes say nothing.

        Test scenario:
            A plain-text CSV has no magic signature, so the `.csv` suffix decides.
        """
        table = tmp_path / "table.csv"
        table.write_text("a,b\n1,2\n", encoding="utf-8")
        assert sniff_format(table) == "csv", "a signature-less CSV should fall back to its extension"

    def test_unknown_when_neither_identifies(self, tmp_path: Path):
        """An unrecognised name and payload yield `"unknown"`.

        Test scenario:
            Neither the bytes nor the suffix identify the file.
        """
        blob = tmp_path / "mystery.bin"
        blob.write_bytes(b"nothing recognisable")
        assert sniff_format(blob) == "unknown", "unidentifiable input should be 'unknown'"


class TestReadResourceMagicFallback:
    """Tests for magic-byte resolution inside `read_resource`."""

    def test_extensionless_raster_resolves_via_magic(self, tmp_path: Path):
        """An extension-less download is resolved from its bytes.

        Test scenario:
            A GeoTIFF saved without any suffix carries no name-based hint, so the
            name/fmt lookup fails; the magic-byte fallback must classify it as a
            raster and read it.
        """
        blob = tmp_path / "downloaded_blob"
        shutil.copy(_GEOTIFF, blob)
        result = read_resource(blob)
        assert result.band_count == 9, f"expected the 9-band ERA5 raster, got {result.band_count} bands"

    def test_named_resource_keeps_its_name_based_answer(self, tmp_path: Path):
        """A correctly named resource is unaffected by the new fallback.

        Test scenario:
            The magic sniff runs only when the name is inconclusive, so a plain
            `.csv` still resolves through the tabular branch.
        """
        table = tmp_path / "values.csv"
        table.write_text("a,b\n5,6\n", encoding="utf-8")
        result = read_resource(table)
        assert list(result.columns) == ["a", "b"], f"expected columns ['a', 'b'], got {list(result.columns)}"


class TestLoadResourceTabularCoverage:
    """Tests for the tabular formats `load_resource` gained from the shared reader."""

    def test_tsv_is_read_as_a_frame(self, tmp_path: Path):
        """A `.tsv` resource now returns a DataFrame instead of raw bytes.

        Test scenario:
            Tab-separated data was absent from this module's old dispatch table
            and fell through to the raw-bytes branch.
        """
        from pyramids.io.sniff import load_resource

        table = tmp_path / "values.tsv"
        table.write_text("a\tb\n7\t8\n", encoding="utf-8")
        result = load_resource(table)
        assert isinstance(result, pd.DataFrame), f"expected a DataFrame for .tsv, got {type(result).__name__}"
        assert list(result.columns) == ["a", "b"], f"expected columns ['a', 'b'], got {list(result.columns)}"

    def test_csv_still_reads_as_a_frame(self, tmp_path: Path):
        """The pre-existing CSV behaviour is preserved.

        Test scenario:
            Routing CSV through the shared tabular reader must not change what
            callers already received.
        """
        from pyramids.io.sniff import load_resource

        table = tmp_path / "values.csv"
        table.write_text("a,b\n1,2\n", encoding="utf-8")
        result = load_resource(table)
        assert list(result.columns) == ["a", "b"], f"expected columns ['a', 'b'], got {list(result.columns)}"
