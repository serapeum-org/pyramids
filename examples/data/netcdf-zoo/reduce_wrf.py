"""Reduce the 80-variable WRF example to a structurally-minimal 17-variable subset.

The published ``wrfout_v2_Lambert.nc`` carries 80 variables, but 63 of them are
structural duplicates: they share a ``(rank, dimension-set, dtype, role)`` signature
already covered by another variable, so they exercise no new NetCDF-reader code path.

This keeps exactly one representative per distinct signature — preserving all four ranks,
all three dtypes (``float32`` / ``int32`` / ``char``), every mass/u/v/w/soil stagger, and the
``XLAT`` / ``XLONG`` geolocation pair — and writes the result under its structural name.

Run with the pyramids ``dev`` environment, e.g. ``pixi run -e dev python reduce_wrf.py``.
"""

from pathlib import Path

from pyramids.netcdf import NetCDF

HERE = Path(__file__).resolve().parent
SRC = HERE / "wrfout_v2_Lambert.nc"
OUT = HERE / "none__17v__1d1-2d5-3d6-4d5__stag-str.nc"

KEEP = {
    "Times",
    "P_TOP", "ITIMESTEP", "ZNU", "ZNW", "ZS",
    "T2", "XLAT", "XLONG", "IVGTYP", "MAPFAC_U", "MAPFAC_V",
    "T", "W", "SMOIS", "U", "V",
}


def main() -> None:
    nc = NetCDF.read_file(str(SRC))
    for name in nc.get_variable_names():
        if name not in KEEP:
            nc.remove_variable(name)
    nc.to_file(str(OUT))
    nc.close()
    print(f"wrote {OUT.name} ({len(KEEP)} variables)")


if __name__ == "__main__":
    main()
