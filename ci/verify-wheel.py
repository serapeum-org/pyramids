"""Smoke test for built platform wheels — exercises the vendor bootstrap.

Invoked by `.github/workflows/build-wheels.yml` once per `test-wheels`
matrix cell after the wheel is pip-installed into a clean Python env. The
checks mirror what an end user does on first import:

1. `import pyramids` triggers `pyramids/__init__.py`'s vendor bootstrap
   (sys.path injection, `GDAL_DATA`/`PROJ_DATA`/`GDAL_DRIVER_PATH`
   env vars, Windows `add_dll_directory` + `PATH` prepend).
2. `from osgeo import gdal, ogr, osr` then resolves to the bundled
   `pyramids/_vendor/osgeo` rather than any system osgeo — verified by
   asserting `"_vendor" in osgeo.__file__`.
3. A round-trip `SpatialReference(4326)` + `MEM` raster creation
   exercises the live libgdal/libproj that the wheel ships.

Kept as a standalone file (rather than an inline `python -c "..."`
heredoc in the workflow) so the script reads cleanly and adding a check
doesn't require fighting YAML + shell quoting.
"""

from pathlib import Path

import pyramids
import osgeo
from osgeo import gdal, ogr, osr  # noqa: F401 — ogr import is a smoke test


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


print(f"pyramids {pyramids.__version__}")
print(f"GDAL {gdal.__version__}")

sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)
authority = sr.GetAttrValue("AUTHORITY", 1)
# Use explicit `if … raise` instead of `assert` so `python -O` doesn't
# strip the check. CI doesn't run with -O today but a future runner
# image change shouldn't silently turn this smoke test into a no-op.
if authority != "4326":
    _fail(f"EPSG:4326 authority round-trip failed: got {authority!r}")

ds = gdal.GetDriverByName("MEM").Create("", 10, 10, 1, gdal.GDT_Byte)
ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
ds.SetProjection(sr.ExportToWkt())

# Confirm `osgeo` resolved to pyramids' vendored copy, not any system
# osgeo that might be on sys.path. Check via Path.is_relative_to()
# rather than a brittle `"_vendor" in osgeo.__file__` substring check:
# substring matching would also accept e.g. /home/_vendor_dev_/site-packages/osgeo.
expected_vendor_root = Path(pyramids.__file__).parent / "_vendor"
osgeo_path = Path(osgeo.__file__).resolve()
if not osgeo_path.is_relative_to(expected_vendor_root.resolve()):
    _fail(f"osgeo not from {expected_vendor_root}: resolved to {osgeo_path}")

print("All runtime checks passed.")
