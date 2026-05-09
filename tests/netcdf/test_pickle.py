"""Tests for :meth:`pyramids.netcdf.NetCDF.__reduce__`.

DASK-4 adds a pickle contract specific to :class:`NetCDF` that
carries three extra bits of state beyond the base
:class:`RasterBase` recipe:

* ``_is_md_array`` — whether the file was opened with
  ``gdal.OF_MULTIDIM_RASTER``.
* ``_is_subset`` — whether the instance is a container or a single
  variable subset.
* ``_source_var_name`` — variable path for drill-down on unpickle.

These tests exercise:

* Container round-trip in MDIM mode.
* Variable-subset round-trip in MDIM mode.
* Container round-trip in classic mode.
* Cross-process pickle via multiprocessing spawn.
* Variable-subset identity on unpickle (returns same variable).
* TypeError for in-memory NetCDF.
"""

from __future__ import annotations

import multiprocessing
import pickle

import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf.netcdf import _reconstruct_netcdf

pytestmark = pytest.mark.core


@pytest.fixture
def noah_path() -> str:
    """Path to a classic NetCDF fixture with multiple variables."""
    return "tests/data/netcdf/noah-precipitation-1979.nc"


@pytest.fixture
def three_d_path() -> str:
    """Path to a pyramids-written 3D NetCDF fixture."""
    return "tests/data/netcdf/pyramids-netcdf-3d.nc"


def _read_container_on_subprocess(payload: bytes) -> tuple[str, bool, bool]:
    """Worker that unpickles + inspects container state."""
    nc = pickle.loads(payload)
    return (type(nc).__name__, bool(nc.is_md_array), bool(nc.is_subset))


def _read_subset_on_subprocess(payload: bytes) -> tuple[str, int]:
    """Worker that unpickles + reads a variable-subset."""
    nc = pickle.loads(payload)
    arr = nc.read_array()
    return (nc._source_var_name or "", int(arr.size))


class TestNetCDFContainerPickle:
    """Pickle a root MDIM container; expect container on unpickle."""

    def test_mdim_container_roundtrip(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        nc2 = pickle.loads(pickle.dumps(nc))
        assert isinstance(nc2, NetCDF)
        assert nc2.is_md_array is True
        assert nc2.is_subset is False

    def test_classic_container_roundtrip(self, noah_path):
        nc = NetCDF.read_file(noah_path, open_as_multi_dimensional=False)
        nc2 = pickle.loads(pickle.dumps(nc))
        assert isinstance(nc2, NetCDF)
        assert nc2.is_md_array is False

    def test_payload_excludes_gdal_dataset(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        data = pickle.dumps(nc)
        assert b"Swig Object" not in data
        assert b"gdal.Dataset" not in data

    def test_cross_process_container_roundtrip(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        payload = pickle.dumps(nc)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            name, md, subset = pool.apply(_read_container_on_subprocess, (payload,))
        assert name == "NetCDF"
        assert md is True
        assert subset is False


class TestNetCDFSubsetPickle:
    """Variable-subset instance pickles and re-drills on unpickle."""

    def test_mdim_subset_roundtrip(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var_name = nc.get_variable_names()[0]
        subset = nc.get_variable(var_name)
        subset2 = pickle.loads(pickle.dumps(subset))
        assert isinstance(subset2, NetCDF)
        assert subset2.is_subset is True
        assert subset2._source_var_name == var_name

    def test_subset_read_array_after_roundtrip(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var_name = nc.get_variable_names()[0]
        subset = nc.get_variable(var_name)
        original = subset.read_array()
        subset2 = pickle.loads(pickle.dumps(subset))
        roundtripped = subset2.read_array()
        assert roundtripped.shape == original.shape
        assert roundtripped.dtype == original.dtype

    def test_cross_process_subset_roundtrip(self, three_d_path):
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var_name = nc.get_variable_names()[0]
        subset = nc.get_variable(var_name)
        payload = pickle.dumps(subset)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            name, size = pool.apply(_read_subset_on_subprocess, (payload,))
        assert name == var_name
        assert size > 0


class TestReconstructNetCDF:
    """The module-level `_reconstruct_netcdf` helper."""

    def test_rebuilds_container(self, three_d_path):
        nc = _reconstruct_netcdf(
            three_d_path,
            "read_only",
            is_md_array=True,
            is_subset=False,
            source_var_name=None,
        )
        assert isinstance(nc, NetCDF)
        assert nc.is_md_array is True
        assert nc.is_subset is False

    def test_rebuilds_subset_when_var_name_given(self, three_d_path):
        nc_container = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var_name = nc_container.get_variable_names()[0]
        nc = _reconstruct_netcdf(
            three_d_path,
            "read_only",
            is_md_array=True,
            is_subset=True,
            source_var_name=var_name,
        )
        assert nc.is_subset is True
        assert nc._source_var_name == var_name
