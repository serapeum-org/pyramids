"""ARC-V6: default ``npartitions`` heuristic for ``read_file(backend="dask")``.

Pins the behaviour of :func:`pyramids.feature.collection._resolve_lazy_partitioning`:

* Caller supplies ``npartitions`` → honoured verbatim.
* Caller supplies ``chunksize`` → honoured verbatim.
* Caller supplies neither, local file → ``npartitions = ceil(size / 128MiB)``.
* Caller supplies neither, cloud / VFS path → ``npartitions = 1``
  (no cheap size probe; users must opt-in to more partitions).

Implementation detail: the heuristic is unit-tested directly on the
helper function; the dask-backed read_file integration test at
``tests/feature/test_lazy_read.py`` exercises the end-to-end path
when ``dask-geopandas`` is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyramids.feature.collection import (
    _LAZY_TARGET_BYTES_PER_PARTITION,
    _resolve_lazy_partitioning,
)

pytestmark = pytest.mark.parquet_lazy


class TestUserSuppliedWins:
    """User-supplied ``npartitions`` / ``chunksize`` short-circuit the heuristic."""

    def test_explicit_npartitions(self, tmp_path: Path):
        (tmp_path / "big.geojson").write_bytes(b"x" * 5_000_000)
        result = _resolve_lazy_partitioning(
            str(tmp_path / "big.geojson"),
            npartitions=7,
            chunksize=None,
        )
        assert result == {"npartitions": 7}

    def test_explicit_chunksize(self, tmp_path: Path):
        (tmp_path / "big.geojson").write_bytes(b"x" * 5_000_000)
        result = _resolve_lazy_partitioning(
            str(tmp_path / "big.geojson"),
            npartitions=None,
            chunksize=10_000,
        )
        assert result == {"chunksize": 10_000}

    def test_npartitions_wins_over_chunksize_when_both_supplied(
        self,
        tmp_path: Path,
    ):
        """If both are set, ``npartitions`` takes precedence (dask-geopandas
        rejects both being set — pyramids normalises to npartitions)."""
        (tmp_path / "big.geojson").write_bytes(b"x" * 5_000_000)
        result = _resolve_lazy_partitioning(
            str(tmp_path / "big.geojson"),
            npartitions=4,
            chunksize=10_000,
        )
        assert result == {"npartitions": 4}


class TestLocalHeuristic:
    """Local files get ``npartitions = ceil(size / 128 MiB)``."""

    def test_tiny_file_one_partition(self, tmp_path: Path):
        p = tmp_path / "small.geojson"
        p.write_bytes(b"{}")
        result = _resolve_lazy_partitioning(str(p), None, None)
        assert result == {"npartitions": 1}

    def test_one_mib_still_one_partition(self, tmp_path: Path):
        """A 1 MiB file rounds up to 1 partition under the 128 MiB target."""
        p = tmp_path / "one_mib.geojson"
        p.write_bytes(b"x" * (1 * 1024 * 1024))
        result = _resolve_lazy_partitioning(str(p), None, None)
        assert result == {"npartitions": 1}

    def test_one_gib_gets_eight_partitions(self, tmp_path: Path):
        """1 GiB ÷ 128 MiB target = 8 partitions. Uses sparse-file trick
        to allocate 1 GiB cheaply on systems that support it; falls back
        to a real write on systems that don't."""
        p = tmp_path / "one_gib.geojson"
        gib = 1024 * 1024 * 1024
        with open(p, "wb") as f:
            f.seek(gib - 1)
            f.write(b"\0")
        assert os.path.getsize(p) == gib
        result = _resolve_lazy_partitioning(str(p), None, None)
        assert result == {"npartitions": 8}

    def test_target_constant_is_128_mib(self):
        """Sanity: the exported constant matches the docstring promise."""
        assert _LAZY_TARGET_BYTES_PER_PARTITION == 128 * 1024 * 1024


class TestCloudAndVfsFallback:
    """Remote / VFS paths skip the size probe and fall back to 1 partition."""

    @pytest.mark.parametrize(
        "path",
        [
            "/vsicurl/http://example.invalid/foo.geojson",
            "/vsizip/foo.zip/inner.shp",
            "http://example.invalid/foo.geojson",
            "https://example.invalid/foo.geojson",
            "s3://bucket/foo.parquet",
            "gs://bucket/foo.parquet",
            "az://container/foo.parquet",
        ],
    )
    def test_remote_falls_back_to_one(self, path: str):
        result = _resolve_lazy_partitioning(path, None, None)
        assert result == {"npartitions": 1}


class TestMissingLocalFile:
    """Nonexistent local path → fall back to 1 rather than propagate OSError."""

    def test_missing_file_does_not_raise(self, tmp_path: Path):
        result = _resolve_lazy_partitioning(
            str(tmp_path / "does_not_exist.geojson"),
            None,
            None,
        )
        assert result == {"npartitions": 1}
