"""Shrink the CDS-Beta ERA5 pressure-levels fixture for issue #311 e2e tests.

The original CDS retrieval is ~2 MB (28 hourly timesteps × 141×321 grid).
The tests in `tests/netcdf/test_sel_4d.py::TestEra5RealFixture` only need
the 4-D structure with CDS-Beta dim names; the spatial extent and number
of timesteps don't matter for correctness, only for size.

This script slices the original down to (valid_time=4, pressure_level=1,
latitude=8, longitude=10), preserves dim names, indexing variables, the
``t`` variable, and the CF attributes, and writes a fresh ~10 KB
NetCDF in place. Idempotent — re-running on the (already small) output
yields the same result.

Run once after intentional changes; do not run on every test invocation.

The read and write are run in separate subprocesses because GDAL's
Windows file-handle release is unreliable: even after dropping every
Python reference, the original file stays locked within the same
process and can't be unlinked. Spawning fresh interpreters is the
robust way to free the handle.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

PATH = Path(__file__).with_name("era5_cds_beta_t_pressure_levels_jan2022.nc")

NT_OUT = 4    # first 4 timesteps (Jan 1, 2022 from 00:00 to 03:00)
NL_OUT = 1    # the single 500 hPa level the original carries
NY_OUT = 8    # 8-row spatial slice
NX_OUT = 10   # 10-col spatial slice


def _string_attr(target, name: str, value: str) -> None:
    """Attach a string-valued attribute via the MDArray attribute API."""
    attr = target.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
    attr.Write(value)


def _read_string_attr(target, name: str) -> str | None:
    """Return the value of a single-string attribute, or None if absent."""
    result = None
    try:
        attr = target.GetAttribute(name)
    except (RuntimeError, AttributeError):
        attr = None
    if attr is not None:
        try:
            value = attr.Read()
        except RuntimeError:
            value = None
        if value is not None:
            result = str(value)
    return result


def _read_pass(src_path: str, sidecar_path: str) -> None:
    """Open the original, dump slice + metadata to a pickle, and exit."""
    src = gdal.OpenEx(src_path, gdal.OF_MULTIDIM_RASTER)
    src_rg = src.GetRootGroup()

    src_t = src_rg.OpenMDArray("t")
    src_dims = src_t.GetDimensions()
    name_to_dim = {d.GetName(): d for d in src_dims}
    iv_attrs = {}
    iv_values = {}
    for name in ("valid_time", "pressure_level", "latitude", "longitude"):
        iv = name_to_dim[name].GetIndexingVariable()
        try:
            arr = iv.ReadAsArray()
        except RuntimeError:
            arr = None
        iv_values[name] = None if arr is None else arr.tolist()
        iv_attrs[name] = {
            attr_name: _read_string_attr(iv, attr_name)
            for attr_name in (
                "units",
                "calendar",
                "long_name",
                "standard_name",
                "stored_direction",
            )
        }

    var_attrs = {
        attr_name: _read_string_attr(src_t, attr_name)
        for attr_name in ("units", "long_name", "standard_name")
    }

    full = src_t.ReadAsArray()
    sliced = full[:NT_OUT, :NL_OUT, :NY_OUT, :NX_OUT].astype(np.float64)

    payload = {
        "iv_attrs": iv_attrs,
        "iv_values": iv_values,
        "var_attrs": var_attrs,
        "sliced": sliced,
    }
    with open(sidecar_path, "wb") as fh:
        pickle.dump(payload, fh)


def _write_pass(out_path: str, sidecar_path: str) -> None:
    """Read the sidecar pickle and build the small NetCDF at out_path."""
    with open(sidecar_path, "rb") as fh:
        payload = pickle.load(fh)
    iv_attrs = payload["iv_attrs"]
    iv_values = payload["iv_values"]
    var_attrs = payload["var_attrs"]
    sliced = payload["sliced"]

    out = Path(out_path)
    if out.exists():
        out.unlink()
    drv = gdal.GetDriverByName("netCDF")
    ds = drv.CreateMultiDimensional(out_path)
    rg = ds.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

    dt_time = rg.CreateDimension("valid_time", "TEMPORAL", "FORWARD", NT_OUT)
    dt_level = rg.CreateDimension(
        "pressure_level", "VERTICAL", "DOWN", NL_OUT
    )
    dt_lat = rg.CreateDimension("latitude", "HORIZONTAL_Y", "NORTH", NY_OUT)
    dt_lon = rg.CreateDimension("longitude", "HORIZONTAL_X", "EAST", NX_OUT)

    def _write_iv(name: str, dim, length: int):
        iv = rg.CreateMDArray(name, [dim], f64)
        values = iv_values.get(name)
        if values is None:
            data = np.arange(length, dtype=np.float64)
        else:
            data = np.asarray(values[:length], dtype=np.float64)
        iv.Write(data)
        for attr_name, attr_val in iv_attrs.get(name, {}).items():
            if attr_val:
                _string_attr(iv, attr_name, attr_val)

    _write_iv("valid_time", dt_time, NT_OUT)
    _write_iv("pressure_level", dt_level, NL_OUT)
    _write_iv("latitude", dt_lat, NY_OUT)
    _write_iv("longitude", dt_lon, NX_OUT)

    var = rg.CreateMDArray("t", [dt_time, dt_level, dt_lat, dt_lon], f64)
    var.Write(sliced)
    for attr_name, attr_val in var_attrs.items():
        if attr_val:
            _string_attr(var, attr_name, attr_val)

    ds = None


def main() -> None:
    if not PATH.exists():
        raise SystemExit(f"missing source file: {PATH}")

    fd, sidecar_str = tempfile.mkstemp(suffix=".pkl", prefix="era5_311_")
    os.close(fd)
    sidecar = Path(sidecar_str)
    tmp_out = PATH.with_suffix(".nc.tmp")

    # Read pass — extract slice and metadata in a fresh interpreter so
    # the original file's handle is released when the subprocess exits.
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--phase=read",
            f"--src={PATH}",
            f"--sidecar={sidecar}",
        ],
        check=True,
    )

    # Write pass — same isolation strategy keeps the writer's handle
    # on the temp file from leaking back into this process.
    if tmp_out.exists():
        tmp_out.unlink()
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--phase=write",
            f"--out={tmp_out}",
            f"--sidecar={sidecar}",
        ],
        check=True,
    )

    sidecar.unlink(missing_ok=True)
    PATH.unlink()
    tmp_out.replace(PATH)
    size = PATH.stat().st_size
    print(f"wrote {PATH.name}: {size} bytes")


def _phase_main() -> None:
    """Entry-point dispatcher for the two subprocess phases."""
    args = {}
    for raw in sys.argv[1:]:
        if raw.startswith("--") and "=" in raw:
            k, v = raw[2:].split("=", 1)
            args[k] = v
    phase = args.get("phase")
    if phase == "read":
        _read_pass(args["src"], args["sidecar"])
    elif phase == "write":
        _write_pass(args["out"], args["sidecar"])
    else:
        main()


if __name__ == "__main__":
    _phase_main()
