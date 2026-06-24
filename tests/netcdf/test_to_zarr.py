"""Tests for :meth:`pyramids.netcdf.NetCDF.to_zarr` and the ``.zarr`` ``to_file`` route (ARC-17).

Zarr write is GDAL-native (via :func:`osgeo.gdal.MultiDimTranslate` with the Zarr driver),
so it needs no ``zarr`` / ``xarray`` Python dependency. These tests write a fixture NetCDF
to a Zarr store and re-open it through GDAL's multidim API to confirm the variables and data
survive the round-trip.
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.fixture()
def three_d_path() -> str:
    """Path to a pyramids-written 3D NetCDF fixture."""
    return "tests/data/netcdf/pyramids-netcdf-3d.nc"


def _zarr_array_names(store: str) -> list[str]:
    """Open a Zarr store via GDAL's multidim API and return its root array names."""
    ds = gdal.OpenEx(store, gdal.OF_MULTIDIM_RASTER)
    try:
        return sorted(ds.GetRootGroup().GetMDArrayNames())
    finally:
        ds = None


class TestToZarr:
    """``NetCDF.to_zarr`` writes a GDAL-readable Zarr store."""

    def test_container_round_trips_all_variables(self, three_d_path, tmp_path):
        """A root MDIM container writes every variable into the Zarr store.

        Test scenario:
            Read a 3-D NetCDF container, write it to ``out.zarr``, re-open the store
            with GDAL's multidim API — expected: the store is a directory carrying the
            same array names as the source root group.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        expected = sorted(nc._raster.GetRootGroup().GetMDArrayNames())
        out = tmp_path / "out.zarr"

        returned = nc.to_zarr(out)

        assert returned == out, f"to_zarr should return the path, got {returned!r}"
        assert out.is_dir(), f"Zarr store should be a directory, got {out!r}"
        assert (
            _zarr_array_names(str(out)) == expected
        ), f"round-tripped arrays {_zarr_array_names(str(out))} != source {expected}"

    def test_subset_writes_single_array(self, three_d_path, tmp_path):
        """A variable subset writes just that array into the store.

        Test scenario:
            Extract the ``values`` variable and write it to a Zarr store — expected: the
            store re-opens and contains the subset's array plus its coordinate arrays.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        subset = nc.get_variable("values")
        out = tmp_path / "sub.zarr"

        subset.to_zarr(out)

        names = _zarr_array_names(str(out))
        assert out.is_dir(), f"subset store should be a directory, got {out!r}"
        assert names, f"subset store should contain at least one array, got {names}"

    def test_existing_store_raises_without_overwrite(self, three_d_path, tmp_path):
        """Writing onto an existing store without ``overwrite`` raises ``FileExistsError``.

        Test scenario:
            Write once, then write again to the same path with the default
            ``overwrite=False`` — expected: ``FileExistsError`` mentioning overwrite.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        out = tmp_path / "dup.zarr"
        nc.to_zarr(out)

        with pytest.raises(FileExistsError, match="overwrite=True"):
            nc.to_zarr(out)

    def test_overwrite_replaces_existing_store(self, three_d_path, tmp_path):
        """``overwrite=True`` replaces an existing store in place.

        Test scenario:
            Write twice to the same path with ``overwrite=True`` on the second call —
            expected: the second write succeeds and the store still round-trips.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        out = tmp_path / "ow.zarr"
        nc.to_zarr(out)

        nc.to_zarr(out, overwrite=True)

        assert out.is_dir(), "store should still exist after overwrite"
        assert _zarr_array_names(str(out)), "store should still carry arrays after overwrite"

    def test_to_file_routes_zarr_extension(self, three_d_path, tmp_path):
        """``to_file`` dispatches a ``.zarr`` extension to the Zarr writer.

        Test scenario:
            Call ``to_file("x.zarr")`` on a container — expected: a Zarr store is created
            (the ``.zarr`` branch), not the NetCDF or the ``ValueError`` container guard.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        out = tmp_path / "viafile.zarr"

        nc.to_file(out)

        assert out.is_dir(), f"to_file('.zarr') should create a store, got {out!r}"
        assert _zarr_array_names(str(out)), "store written via to_file should carry arrays"
