"""Tests for :meth:`pyramids.netcdf.NetCDF.from_bytes`."""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF, Container

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def netcdf_bytes(noah_nc_path: str) -> bytes:
    """Raw bytes of a small NetCDF fixture.

    Args:
        noah_nc_path: Path to the cf__6v__1d2-2d4__geog__y-asc.nc fixture.

    Returns:
        bytes: Contents of the cf__6v__1d2-2d4__geog__y-asc.nc fixture.
    """
    return Path(noah_nc_path).read_bytes()


class TestNetCDFFromBytes:
    """Tests for :meth:`NetCDF.from_bytes`."""

    def test_returns_netcdf_instance(self, netcdf_bytes: bytes):
        """The result is a :class:`Container` (a :class:`NetCDF`), not a plain Dataset.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            ``NetCDF.from_bytes(bytes)`` opens a file, so it returns the canonical
            container type ``Container`` (API-1, #614), which is a ``NetCDF`` subclass
            — so ``isinstance(nc, NetCDF)`` still holds.
        """
        nc = NetCDF.from_bytes(netcdf_bytes)
        assert isinstance(nc, NetCDF), f"expected NetCDF, got {type(nc)}"
        assert type(nc) is Container, f"expected Container, got {type(nc)}"

    def test_round_trip_matches_read_file(self, netcdf_bytes: bytes, noah_nc_path: str):
        """Opening from bytes matches opening the same file from disk.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.
            noah_nc_path: Path to the cf__6v__1d2-2d4__geog__y-asc.nc fixture.

        Test scenario:
            ``NetCDF.from_bytes(bytes)`` vs ``NetCDF.read_file(path)`` —
            expected: same shape, epsg, and variable list.
        """
        ref = NetCDF.read_file(noah_nc_path)
        nc = NetCDF.from_bytes(netcdf_bytes)
        assert nc.shape == ref.shape, f"shape mismatch: {nc.shape} != {ref.shape}"
        assert nc.epsg == ref.epsg, f"epsg mismatch: {nc.epsg} != {ref.epsg}"
        assert list(nc.variables) == list(ref.variables), "variable list mismatch"

    def test_opened_multi_dimensional_by_default(self, netcdf_bytes: bytes):
        """``open_as_multi_dimensional`` defaults to ``True`` for the NetCDF flavour.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            Inspect the internal multidim flag — expected: ``True``.
        """
        nc = NetCDF.from_bytes(netcdf_bytes)
        assert (
            nc._is_md_array is True
        ), "NetCDF.from_bytes should open in multidim mode by default"

    def test_classic_mode_when_requested(self, netcdf_bytes: bytes):
        """``open_as_multi_dimensional=False`` opens in classic subdataset mode.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            ``NetCDF.from_bytes(bytes, open_as_multi_dimensional=False)`` —
            expected: the internal multidim flag is ``False`` and the object
            is still a usable ``NetCDF``.
        """
        nc = NetCDF.from_bytes(netcdf_bytes, open_as_multi_dimensional=False)
        assert nc._is_md_array is False, "classic mode flag not honoured"
        assert isinstance(nc, NetCDF), "should still be a NetCDF instance"

    def test_backing_path_uses_nc_suffix(self, netcdf_bytes: bytes):
        """The temporary ``/vsimem/`` path ends in ``.nc``.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            Inspect ``_vsimem_path`` — expected: ``/vsimem/...nc`` and the
            file exists while the object is alive.
        """
        nc = NetCDF.from_bytes(netcdf_bytes)
        assert nc._vsimem_path.startswith(
            "/vsimem/"
        ), f"bad backing path: {nc._vsimem_path!r}"
        assert nc._vsimem_path.endswith(
            ".nc"
        ), f"expected .nc suffix: {nc._vsimem_path!r}"
        assert (
            gdal.VSIStatL(nc._vsimem_path) is not None
        ), "backing /vsimem/ file is missing"

    def test_name_argument_sets_file_name(self, netcdf_bytes: bytes):
        """``name=`` overrides the cosmetic ``file_name``.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            ``NetCDF.from_bytes(bytes, name="era5")`` — expected:
            ``file_name == "era5"``.
        """
        nc = NetCDF.from_bytes(netcdf_bytes, name="era5")
        assert nc.file_name == "era5", f"name= not applied: {nc.file_name!r}"

    def test_vsimem_cleaned_up_on_gc(self, netcdf_bytes: bytes):
        """Dropping the last reference removes the ``/vsimem/`` file.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            Capture ``_vsimem_path``, ``del`` + ``gc.collect()`` — expected:
            the file is gone (``weakref.finalize`` ran).
        """
        nc = NetCDF.from_bytes(netcdf_bytes)
        vsi_path = nc._vsimem_path
        assert gdal.VSIStatL(vsi_path) is not None, "precondition: file should exist"
        del nc
        gc.collect()
        assert gdal.VSIStatL(vsi_path) is None, "/vsimem/ file was not cleaned up on GC"

    def test_not_picklable(self, netcdf_bytes: bytes):
        """An in-memory NetCDF cannot be pickled.

        Args:
            netcdf_bytes: Raw bytes of the NetCDF fixture.

        Test scenario:
            ``pickle.dumps(NetCDF.from_bytes(bytes))`` — expected: ``TypeError``
            advising ``.to_file`` first.
        """
        nc = NetCDF.from_bytes(netcdf_bytes)
        with pytest.raises(TypeError, match=r"to_file"):
            pickle.dumps(nc)

    @pytest.mark.parametrize("bad", ["a string", 7, None])
    def test_non_bytes_raises_type_error(self, bad):
        """Non bytes-like input raises ``TypeError``.

        Args:
            bad: An object that is not bytes-like.

        Test scenario:
            ``NetCDF.from_bytes(bad)`` — expected: ``TypeError`` mentioning
            ``bytes-like``.
        """
        with pytest.raises(TypeError, match="bytes-like"):
            NetCDF.from_bytes(bad)

    def test_unopenable_bytes_raise_value_error(self):
        """Bytes that are not a NetCDF raise ``ValueError``.

        Test scenario:
            ``NetCDF.from_bytes(b"nope")`` — expected: ``ValueError`` mentioning
            ``suffix``.
        """
        with pytest.raises(ValueError, match="suffix"):
            NetCDF.from_bytes(b"definitely not a netcdf file")

    def test_unopenable_bytes_do_not_leak_vsimem(self):
        """A failed open leaves no orphaned ``/vsimem/`` file.

        Test scenario:
            Snapshot ``/vsimem`` listing around a bad open — expected: no new
            entries.
        """
        before = set(gdal.ReadDir("/vsimem") or [])
        with pytest.raises(ValueError):
            NetCDF.from_bytes(b"not a netcdf")
        after = set(gdal.ReadDir("/vsimem") or [])
        assert after.issubset(before), f"leaked /vsimem/ files: {after - before}"
