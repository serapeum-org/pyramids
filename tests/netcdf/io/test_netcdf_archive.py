"""Tests for :meth:`pyramids.netcdf.NetCDF.read_file` archive kwargs (PY-7).

The kwargs ``vsi=`` and ``file_i=`` are forwarded to
:func:`pyramids._io.read_file`. GDAL's netCDF driver requires Linux
``userfaultfd`` to open a ``.nc`` from any ``/vsi*`` path (archive,
``/vsicurl/``, ``/vsimem/`` via this route), so the end-to-end zip
round-trip lives in ``TestNetCDFReadFileArchiveLinux`` and is gated by
``@pytest.mark.skipif(not sys.platform.startswith("linux"), ...)``. The
forwarding tests in ``TestNetCDFReadFileArchiveForwarding`` run
cross-platform.

CI matrix coverage: ``.github/workflows/tests.yml`` runs
``ubuntu-latest`` on every push / PR across ``py311``, ``py312``,
``py313``. The Linux-only class is exercised on those three shards.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

import pyramids._io as pyramids_io
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

NC_FIXTURE = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"


class TestNetCDFReadFileArchiveForwarding:
    """Cross-platform tests: kwargs are forwarded to ``_io.read_file``."""

    def test_plain_path_no_regression(self):
        """Test plain-path open still returns a multi-variable NetCDF.

        Test scenario:
            Existing single-arg call must keep its pre-PY-7 behaviour.
        """
        nc = NetCDF.read_file(NC_FIXTURE)
        assert sorted(nc.variables) == [
            "Band1",
            "Band2",
            "Band3",
            "Band4",
        ], f"Expected 4 Bands, got {sorted(nc.variables)!r}"

    def test_kwargs_forwarded_to_io_read_file(self, mocker):
        """Test ``vsi=`` and ``file_i=`` reach ``_io.read_file``.

        Test scenario:
            The override must thread both new kwargs through unchanged
            so the archive surface in ``_io.read_file`` (already covered
            by PY-2 Dataset tests) governs the actual behaviour.
        """
        spy = mocker.patch(
            "pyramids.netcdf.netcdf._io.read_file",
            wraps=pyramids_io.read_file,
        )
        NetCDF.read_file(NC_FIXTURE, vsi=None, file_i=0)
        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        assert kwargs.get("vsi") is None, f"vsi forwarded as {kwargs.get('vsi')!r}"
        assert (
            kwargs.get("file_i") == 0
        ), f"file_i forwarded as {kwargs.get('file_i')!r}"

    def test_vsi_value_propagates(self, mocker):
        """Test a non-None ``vsi=`` reaches ``_io.read_file`` verbatim.

        Test scenario:
            Forwarding logic must not coerce or rewrite the ``vsi``
            string — ``_io.read_file`` owns validation.
        """
        sentinel = RuntimeError("intercepted before opening")
        spy = mocker.patch(
            "pyramids.netcdf.netcdf._io.read_file",
            side_effect=sentinel,
        )
        with pytest.raises(RuntimeError, match="intercepted"):
            NetCDF.read_file("anything.zip", vsi="zip", file_i=3)
        forwarded = spy.call_args.kwargs
        assert (
            forwarded.get("vsi") == "zip"
        ), f"Expected vsi='zip', got {forwarded.get('vsi')!r}"
        assert (
            forwarded.get("file_i") == 3
        ), f"Expected file_i=3, got {forwarded.get('file_i')!r}"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="GDAL netCDF driver needs Linux userfaultfd for /vsi* paths",
)
class TestNetCDFReadFileArchiveLinux:
    """End-to-end archive open — Linux only.

    GDAL emits ``RuntimeError: Opening a /vsi file with the netCDF
    driver requires Linux userfaultfd to be available`` on Windows /
    macOS. The class is skipped there.
    """

    @pytest.fixture
    def zipped_nc(self, tmp_path: Path) -> Path:
        """Create a zip containing one ``.nc`` member.

        Args:
            tmp_path: pytest's per-test temp directory.

        Returns:
            Path: Location of the zip on disk.
        """
        zpath = tmp_path / "noah.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.write(NC_FIXTURE, arcname="noah.nc")
        return zpath

    def test_open_zipped_nc_via_vsi_auto(self, zipped_nc: Path):
        """Test ``vsi="auto"`` opens the single member of a ``.zip``.

        Args:
            zipped_nc: Fixture path to a zip holding ``noah.nc``.

        Test scenario:
            Auto-detection from the ``.zip`` extension should find the
            sole member and return a real NetCDF container.
        """
        nc = NetCDF.read_file(zipped_nc, vsi="auto")
        assert sorted(nc.variables) == ["Band1", "Band2", "Band3", "Band4"]

    def test_open_zipped_nc_via_vsi_zip(self, zipped_nc: Path):
        """Test explicit ``vsi="zip"`` works on a ``.zip`` path.

        Args:
            zipped_nc: Fixture path to a zip holding ``noah.nc``.

        Test scenario:
            Explicit kind selection bypasses extension sniffing.
        """
        nc = NetCDF.read_file(zipped_nc, vsi="zip", file_i=0)
        assert sorted(nc.variables) == ["Band1", "Band2", "Band3", "Band4"]
