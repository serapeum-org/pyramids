"""Regression tests for #565 — string non-spatial aux vars survive spatial ops.

ERA5's ``expver`` (dtype ``<U4``, dim ``valid_time``) was dropped with a
``UserWarning`` by ``crop`` / ``to_crs`` / ``reduce`` because the GDAL SWIG
bindings cannot ``ReadAsArray`` a string MDArray. The carry path now copies
string variables via the Python-list ``Read()`` / ``Write()`` path, so they are
carried through container spatial ops unchanged.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import shapely
from osgeo import gdal

from pyramids.netcdf import NetCDF

ERA5 = "tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc"
_MASK = gpd.GeoDataFrame(
    geometry=[shapely.geometry.box(-75.0, 4.2, -74.0, 4.8)], crs="EPSG:4326"
)


def _read_expver(nc: NetCDF) -> list:
    return nc._raster.GetRootGroup().OpenMDArray("expver").Read()


class TestCropStringAux:
    """A string aux var is carried through ``crop`` (no warn-and-drop)."""

    def test_crop_carries_string_aux_without_warning(self):
        """crop keeps ``expver`` with its values intact and emits no drop warning."""
        cube = NetCDF.read_file(ERA5)
        source_expver = _read_expver(cube)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked = cube.crop(mask=_MASK, touch=True)
        try:
            assert "expver" in masked.variable_names, "string aux var must survive crop"
            assert _read_expver(masked) == source_expver, "carried values must match"
            dropped = [str(w.message) for w in caught if "could not carry" in str(w.message)]
            assert not dropped, f"expected no drop warning, got: {dropped}"
        finally:
            cube.close()
            masked.close()


class TestAddMdArrayString:
    """`_add_md_array_to_group` copies a string MDArray via the Read/Write path."""

    def test_string_md_array_is_copied(self):
        """A string MDArray round-trips through `_add_md_array_to_group`."""
        values = ["0001", "0001", "0005"]
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("src")
        src_rg = src.GetRootGroup()
        dim = src_rg.CreateDimension("t", "", "", len(values))
        src_md = src_rg.CreateMDArray("expver", [dim], gdal.ExtendedDataType.CreateString())
        src_md.Write(values)

        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("dst")
        dst_rg = dst.GetRootGroup()
        NetCDF._add_md_array_to_group(dst_rg, "expver", src_md)

        copied = dst_rg.OpenMDArray("expver")
        assert copied.GetDataType().GetClass() == gdal.GEDTC_STRING, "dtype preserved"
        assert copied.Read() == values, "string values preserved"

    def test_numeric_md_array_still_copied(self):
        """The numeric path is unchanged: values and NoData carry over."""
        arr = np.array([0, 0, 1, 1], dtype="i4")
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("src")
        src_rg = src.GetRootGroup()
        dim = src_rg.CreateDimension("t", "", "", arr.size)
        src_md = src_rg.CreateMDArray(
            "number", [dim], gdal.ExtendedDataType.Create(gdal.GDT_Int32)
        )
        src_md.Write(arr)

        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("dst")
        dst_rg = dst.GetRootGroup()
        NetCDF._add_md_array_to_group(dst_rg, "number", src_md)

        np.testing.assert_array_equal(dst_rg.OpenMDArray("number").ReadAsArray(), arr)
