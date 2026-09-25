"""`isel(drop=True)` removes the point-selected axis, matching xarray (#1193).

pyramids and xarray model a scalar `isel` differently — xarray drops the dimension and keeps a
scalar coordinate, pyramids keeps a length-one dimension — but `drop=True` must agree on the
outcome that matters: the selected axis is gone entirely, from neither a dimension nor a
coordinate does `time` survive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.interop

CF_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-4d1__y-asc.nc"
)


def _sides():
    """The pyramids variable and the xarray DataArray for the same CF cube.

    Returns:
        tuple: `(container, xarray_dataarray)` for `temperature`.
    """
    container = NetCDF.read_file(str(CF_PATH))
    return container, container.to_xarray()["temperature"]


def test_drop_true_removes_the_axis_from_both():
    """With `drop=True`, `time` survives in neither pyramids' dims nor xarray's dims/coords."""
    container, da = _sides()
    pyr = container.get_variable("temperature").isel(time=0, drop=True)
    dropped = da.isel(time=0, drop=True)
    assert "time" not in pyr._band_dim_names, pyr._band_dim_names
    assert "time" not in dropped.dims, dropped.dims
    assert "time" not in dropped.coords, list(dropped.coords)


def test_without_drop_both_keep_the_axis():
    """Without `drop`, `time` stays — a length-one dim in pyramids, a scalar coord in xarray."""
    container, da = _sides()
    pyr = container.get_variable("temperature").isel(time=0)
    assert "time" in pyr._band_dim_names, pyr._band_dim_names
    assert "time" in da.isel(time=0).coords, list(da.isel(time=0).coords)
