from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from pyramids.netcdf.netcdf import NetCDF

# Shared random seed used across NetCDF test modules.
SEED = 42
# Shared geographic geotransform (lon_origin, x_res, 0, lat_origin, 0, -y_res).
GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)


@pytest.fixture(scope="module")
def noah_nc_path() -> str:
    return "tests/data/netcdf/noah-precipitation-1979.nc"


@pytest.fixture(scope="module")
def pyramids_created_nc_3d() -> str:
    return "tests/data/netcdf/pyramids-netcdf-3d.nc"


@pytest.fixture(scope="module")
def two_variable_nc() -> str:
    return "tests/data/netcdf/two_vars_scale_offset.nc"


def make_3d_nc(
    rows: int = 10,
    cols: int = 12,
    bands: int = 3,
    epsg: int = 4326,
    variable_name: str = "temperature",
    no_data_value: float = -9999.0,
    geo: tuple | None = None,
    arr_type: str = "random",
    seed: int = 42,
    extra_dim_name: str | None = None,
    extra_dim_values: Sequence[float] | None = None,
) -> NetCDF:
    """Create a 3D in-memory NetCDF container for testing.

    This is a shared helper used across multiple test modules.
    Callers customise behaviour via keyword arguments.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        bands: Number of bands.
        epsg: EPSG code for the CRS.
        variable_name: Name of the variable.
        no_data_value: No-data sentinel value.
        geo: Geotransform tuple. If ``None``, defaults to
            ``(0.0, 1.0, 0, float(rows), 0, -1.0)``.
        arr_type: ``"random"`` for ``RandomState(seed).rand()``,
            ``"sequential"`` for ``np.arange().reshape()``.
        seed: Random seed (only used when *arr_type* is ``"random"``).
        extra_dim_name: Name of the extra dimension (e.g. ``"time"``).
        extra_dim_values: Coordinate values for the extra dimension.

    Returns:
        NetCDF: An in-memory MDIM container with one variable.
    """
    if geo is None:
        geo = (0.0, 1.0, 0, float(rows), 0, -1.0)

    if arr_type == "random":
        arr = np.random.RandomState(seed).rand(bands, rows, cols).astype(np.float64)
    else:
        arr = np.arange(
            bands * rows * cols,
            dtype=np.float64,
        ).reshape(bands, rows, cols)

    kwargs = {
        "arr": arr,
        "geo": geo,
        "epsg": epsg,
        "no_data_value": no_data_value,
        "variable_name": variable_name,
    }
    if extra_dim_name is not None:
        kwargs["extra_dim_name"] = extra_dim_name
    if extra_dim_values is not None:
        kwargs["extra_dim_values"] = list(extra_dim_values)

    nc = NetCDF.create_from_array(**kwargs)
    return nc


def make_2d_nc(
    rows: int = 10,
    cols: int = 12,
    variable_name: str = "elevation",
) -> NetCDF:
    """Create a 2D in-memory NetCDF container for testing.

    This is a shared helper used across multiple test modules.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        variable_name: Name of the variable.

    Returns:
        NetCDF: An in-memory multidimensional NetCDF container.
    """
    # Arbitrary deterministic fill — only the shape/dtype/structure matter to
    # the consumers; no test asserts on the exact cell values, so the modern
    # default_rng stream (vs the legacy RandomState it replaced) is fine here.
    arr = np.random.default_rng(99).random((rows, cols)).astype(np.float64)
    geo = (0.0, 1.0, 0, float(rows), 0, -1.0)
    return NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        path=None,
        variable_name=variable_name,
    )


def make_plot_3d_nc(n_times: int = 4, rows: int = 5, cols: int = 5) -> NetCDF:
    """Build a 3-D (time, lat, lon) in-memory NetCDF for plot tests.

    Uses ``float32`` data, variable name ``t2m``, and a fixed RNG seed (0)
    so every call with the same arguments returns the same synthetic array.
    Distinct from :func:`make_3d_nc` which uses float64, ``temperature``,
    and a configurable seed.

    Args:
        n_times: Number of time steps.
        rows: Number of latitude rows.
        cols: Number of longitude columns.

    Returns:
        NetCDF: Root MDIM container with a single variable ``t2m``.
    """
    rng = np.random.default_rng(0)
    arr = rng.random((n_times, rows, cols)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(n_times)),
    )
    return nc


class MockDim:
    """Minimal mock of a GDAL MDArray dimension for CF tests.

    Attributes:
        name: Short dimension name (e.g. ``"x"``).
        full_name: Fully qualified name in the form ``"/<name>"``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.full_name = f"/{name}"
