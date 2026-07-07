"""Build the 4-D test fixture used by tests/netcdf/test_sel_4d.py.

Idempotent — re-running overwrites the file. Run once to regenerate
after intentional changes; do not run on every test invocation.

The encoding ``value = t*1000 + l*100 + y*10 + x`` makes storage order
verifiable: reading any single band reveals the (t, l) combination it
came from, so chained ``sel()`` results can be asserted exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal

OUT = Path(__file__).with_name("cf__5v__1d4-4d1__y-asc.nc")
TIME_VALUES = [0, 6, 12, 18]  # hours
LEVEL_VALUES = [1000, 850, 500]  # hPa
LAT_VALUES = np.linspace(40.0, 44.0, 5)  # degrees north
LON_VALUES = np.linspace(-10.0, -5.0, 6)  # degrees east


def encode(t_idx: int, l_idx: int, y_idx: int, x_idx: int) -> float:
    """Encode (t, l, y, x) into a single float so tests can assert the
    exact (t, l) combination present in any selected band.
    """
    return t_idx * 1000.0 + l_idx * 100.0 + y_idx * 10.0 + x_idx


def _set_string_attr(target, name: str, value: str) -> None:
    """Attach a string-valued attribute to a GDAL MDArray or Group.

    The MDArray API exposes attribute creation via
    `CreateAttribute(name, dimensions, datatype)` followed by
    `Write(value)` — there is no `SetAttributeString` shortcut.
    """
    attr = target.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
    attr.Write(value)


def main() -> None:
    nt, nl, ny, nx = (
        len(TIME_VALUES),
        len(LEVEL_VALUES),
        len(LAT_VALUES),
        len(LON_VALUES),
    )
    data = np.zeros((nt, nl, ny, nx), dtype=np.float64)
    for t in range(nt):
        for l in range(nl):
            for y in range(ny):
                for x in range(nx):
                    data[t, l, y, x] = encode(t, l, y, x)

    if OUT.exists():
        OUT.unlink()
    drv = gdal.GetDriverByName("netCDF")
    ds = drv.CreateMultiDimensional(str(OUT))
    rg = ds.GetRootGroup()

    dt = rg.CreateDimension("time", "TEMPORAL", "FORWARD", nt)
    dl = rg.CreateDimension("pressure_level", "VERTICAL", "DOWN", nl)
    dy = rg.CreateDimension("lat", "HORIZONTAL_Y", "NORTH", ny)
    dx = rg.CreateDimension("lon", "HORIZONTAL_X", "EAST", nx)

    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

    # GDAL's netCDF MDIM writer auto-detects an MDArray as a
    # dimension's indexing variable when the array name matches the
    # dimension name; SetIndexingVariable() is a no-op there. We rely
    # on the name-match contract.
    t_iv = rg.CreateMDArray("time", [dt], f64)
    t_iv.Write(np.asarray(TIME_VALUES, dtype=np.float64))
    _set_string_attr(t_iv, "units", "hours since 2024-01-01")

    l_iv = rg.CreateMDArray("pressure_level", [dl], f64)
    l_iv.Write(np.asarray(LEVEL_VALUES, dtype=np.float64))
    _set_string_attr(l_iv, "units", "hPa")

    y_iv = rg.CreateMDArray("lat", [dy], f64)
    y_iv.Write(LAT_VALUES.astype(np.float64))
    _set_string_attr(y_iv, "units", "degrees_north")

    x_iv = rg.CreateMDArray("lon", [dx], f64)
    x_iv.Write(LON_VALUES.astype(np.float64))
    _set_string_attr(x_iv, "units", "degrees_east")

    var = rg.CreateMDArray("temperature", [dt, dl, dy, dx], f64)
    var.Write(data)
    _set_string_attr(var, "units", "K")

    ds = None  # flush
    print(f"Wrote {OUT} (shape={data.shape})")


if __name__ == "__main__":
    main()
