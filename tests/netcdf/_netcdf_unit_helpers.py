"""Shared in-memory NetCDF/Dataset builders for the tests/netcdf/test_netcdf_unit_* split.

Extracted verbatim from the original tests/netcdf/test_netcdf_unit.py module-level helpers.
"""

from __future__ import annotations

import numpy as np
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import Container
from tests.netcdf.conftest import make_3d_nc


def _make_3d_nc(
    rows=10,
    cols=12,
    bands=3,
    epsg=4326,
    variable_name="temperature",
    no_data_value=-9999.0,
):
    """Create a 3D in-memory NetCDF for testing.

    Delegates to the shared ``make_3d_nc`` helper in conftest.
    """
    return make_3d_nc(
        rows=rows,
        cols=cols,
        bands=bands,
        epsg=epsg,
        variable_name=variable_name,
        no_data_value=no_data_value,
        arr_type="random",
        seed=42,
    )


def _make_dataset_2d(rows=10, cols=12, no_data=-9999.0):
    """Create a 2D in-memory Dataset for testing.

    Returns:
        Dataset: A plain raster Dataset.
    """
    arr = np.random.RandomState(77).rand(rows, cols).astype(np.float64)
    geo = (0.0, 1.0, 0, float(rows), 0, -1.0)
    return Dataset.create_from_array(
        arr,
        geo=geo,
        epsg=4326,
        no_data_value=no_data,
    )


def _make_dataset_3d(bands=3, rows=10, cols=12, no_data=-9999.0):
    """Create a 3D in-memory Dataset for testing.

    Returns:
        Dataset: A plain raster Dataset with multiple bands.
    """
    arr = np.random.RandomState(77).rand(bands, rows, cols).astype(np.float64)
    geo = (0.0, 1.0, 0, float(rows), 0, -1.0)
    return Dataset.create_from_array(
        arr,
        geo=geo,
        epsg=4326,
        no_data_value=no_data,
    )


def _make_nc_with_time_units(rows=4, cols=5, n_times=3):
    """Create an MDIM NetCDF with a time dimension that has a 'units' attr.

    This is needed to exercise the get_time_variable conversion path.

    Returns:
        NetCDF: An in-memory NetCDF with a time dimension carrying
            a ``units`` attribute of ``"days since 1979-01-01"``.
    """
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("time_test")
    rg = src.GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

    # Create x dimension
    dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, cols)
    x_vals = rg.CreateMDArray("x", [dim_x], dtype)
    x_vals.Write(np.arange(cols, dtype=np.float64) + 0.5)
    dim_x.SetIndexingVariable(x_vals)

    # Create y dimension
    dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, rows)
    y_vals = rg.CreateMDArray("y", [dim_y], dtype)
    y_vals.Write(np.arange(rows, dtype=np.float64)[::-1] + 0.5)
    dim_y.SetIndexingVariable(y_vals)

    # Create time dimension with units attribute
    dim_t = rg.CreateDimension("time", "TEMPORAL", None, n_times)
    t_vals = rg.CreateMDArray("time", [dim_t], dtype)
    t_vals.Write(np.arange(n_times, dtype=np.float64))
    dim_t.SetIndexingVariable(t_vals)

    # Add 'units' attribute to the time variable
    str_dtype = gdal.ExtendedDataType.CreateString()
    units_attr = t_vals.CreateAttribute("units", [], str_dtype)
    units_attr.Write("days since 1979-01-01")

    # Create a data variable
    data_arr = rg.CreateMDArray("temperature", [dim_t, dim_y, dim_x], dtype)
    data_arr.Write(np.random.RandomState(55).rand(n_times, rows, cols))
    data_arr.SetNoDataValueDouble(-9999.0)

    return Container(src)
