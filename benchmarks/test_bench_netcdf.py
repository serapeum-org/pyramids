"""Benchmarks for NetCDF multidim reads, selection, and eager-vs-lazy (dask) throughput.

The headline metric the dask effort cares about is **eager vs lazy**: reading a whole variable
into memory directly versus building a chunked dask array and materializing the same whole
variable. Both read the full variable (a clean read-vs-read comparison — the lazy case adds
dask's chunk-assembly overhead, it does not do extra reduction work), and both appear in the
``netcdf-read`` group so a run compares them side by side; selection (``isel`` / ``sel`` /
``subset``) is grouped separately. The memory-bounded streaming property of the lazy path is
asserted separately in ``test_bench_memory.py``.

The fixture builds a synthetic ``(time, lat, lon)`` NetCDF through ``NetCDF.from_xarray``, which
writes via pyramids' own GDAL multidimensional writer — so no ``netCDF4`` / ``h5netcdf`` backend
is required to create representative multidim data.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pyramids.netcdf import NetCDF

TIME = 12
SIZE = 512
VARIABLE = "temperature"


@pytest.fixture(scope="session")
def netcdf_path(tmp_path_factory) -> str:
    """A synthetic ``(time=12, lat=512, lon=512)`` multidim NetCDF written once via GDAL."""
    rng = np.random.default_rng(2)
    data = rng.random((TIME, SIZE, SIZE), dtype=np.float32)
    lat = np.linspace(60.0, 20.0, SIZE)  # north-up (decreasing)
    lon = np.linspace(0.0, 40.0, SIZE)
    ds = xr.Dataset(
        {VARIABLE: (("time", "lat", "lon"), data)},
        coords={"time": np.arange(TIME), "lat": lat, "lon": lon},
    )
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    path = tmp_path_factory.mktemp("bench_nc") / "cube.nc"
    NetCDF.from_xarray(ds, path=str(path))
    return str(path)


@pytest.mark.benchmark(group="netcdf-read")
def test_read_array_eager(benchmark, netcdf_path: str):
    """Read a whole variable eagerly into a NumPy array (reopened each round)."""
    benchmark(lambda: NetCDF.read_file(netcdf_path).read_array(VARIABLE))


@pytest.mark.benchmark(group="netcdf-read")
def test_read_array_lazy(benchmark, netcdf_path: str):
    """Build a chunked dask array and materialize the whole variable — read-vs-read with eager."""

    def _lazy_read():
        arr = NetCDF.read_file(netcdf_path).read_array(VARIABLE, chunks=128)
        return (
            arr.compute()
        )  # materialize the full variable, matching the eager read's work

    benchmark(_lazy_read)


@pytest.mark.benchmark(group="netcdf-select")
def test_isel(benchmark, netcdf_path: str):
    """Index-based selection of a single time step."""
    benchmark(lambda: NetCDF.read_file(netcdf_path).isel(time=5))


@pytest.mark.benchmark(group="netcdf-select")
def test_sel(benchmark, netcdf_path: str):
    """Label-based (nearest) selection of a single time step."""
    benchmark(lambda: NetCDF.read_file(netcdf_path).sel(time=5, method="nearest"))


@pytest.mark.benchmark(group="netcdf-select")
def test_subset(benchmark, netcdf_path: str):
    """Spatial subset of a variable by a geographic bbox."""
    benchmark(
        lambda: NetCDF.read_file(netcdf_path).subset(
            VARIABLE,
            time=0,
            bbox=(5.0, 25.0, 35.0, 45.0),
            crs=4326,
            y_dim="lat",
            x_dim="lon",
        )
    )
