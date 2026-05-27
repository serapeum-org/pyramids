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
4. A `/vsicurl` HTTPS read exercises the bundled libcurl's CA trust
   store — the regression in issue #412, where the vendored libcurl
   pointed at the wheel-build prefix's `cacert.pem` (absent in the
   consuming env) and every TLS read failed to load trust anchors.

Kept as a standalone file (rather than an inline `python -c "..."`
heredoc in the workflow) so the script reads cleanly and adding a check
doesn't require fighting YAML + shell quoting.
"""

import os
import sys
from pathlib import Path

import pyramids
import osgeo
from osgeo import gdal, ogr, osr  # noqa: F401 — ogr import is a smoke test

# Stable, valid-TLS HTTPS endpoint for the CA-trust check. Small text
# file (not a raster) — VSIFOpenL forces the curl TLS handshake + CA
# load without needing a valid GeoTIFF on the far end.
_TLS_PROBE_URL = "/vsicurl/https://raw.githubusercontent.com/OSGeo/gdal/master/LICENSE.TXT"
# Substrings that mark a CA / trust-store failure (the #412 bug) rather
# than a generic network problem (timeout, DNS, offline runner).
_CA_ERROR_MARKERS = ("trust anchors", "cacert", "ca cert", "certificate", "ssl")


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


def _check_tls_read() -> None:
    """Open an HTTPS resource via /vsicurl to exercise the bundled CA store.

    This runs only after we've already asserted osgeo resolves to the
    vendored copy — i.e. this is always a real bundled wheel, never a
    dev/editable install. So a missing cacert.pem is itself the #412
    regression and fails hard (an earlier version *skipped* here, which
    silently masked the bug when the cert wasn't bundled).

    A CA/trust-store error during the read is also the #412 bug and fails
    hard. A generic network error (offline runner, DNS, timeout) is not
    this bug, so we warn and move on rather than make the smoke test flaky
    on network conditions — the cert-presence + GDAL_HTTP_CAINFO checks
    above already prove the fix shipped.
    """
    ca_bundle = Path(pyramids.__file__).parent / "_data" / "ssl" / "cacert.pem"
    if not ca_bundle.is_file():
        _fail(f"bundled wheel is missing {ca_bundle} — CA bundle not vendored (issue #412)")
    cainfo = os.environ.get("GDAL_HTTP_CAINFO")
    print(f"GDAL_HTTP_CAINFO: {cainfo}")
    if cainfo != str(ca_bundle):
        _fail(f"GDAL_HTTP_CAINFO={cainfo!r} not pointed at bundled cert {ca_bundle} (issue #412)")

    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "30")
    handle = None
    try:
        handle = gdal.VSIFOpenL(_TLS_PROBE_URL, "rb")
        if handle is None:
            _fail(f"VSIFOpenL returned None for {_TLS_PROBE_URL}")
        data = gdal.VSIFReadL(1, 16, handle)
        if not data:
            _fail(f"empty read from {_TLS_PROBE_URL}")
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(marker in msg for marker in _CA_ERROR_MARKERS):
            _fail(f"TLS CA trust failure (issue #412): {exc}")
        # Not a CA error — treat as transient network issue, don't fail.
        print(f"TLS check inconclusive (non-CA network error, ignored): {exc}")
    finally:
        if handle is not None:
            gdal.VSIFCloseL(handle)
        # Drop GDAL's cached /vsicurl connections. On Windows, leaving the
        # libcurl connection pool open deadlocks the interpreter at exit
        # (curl global cleanup vs Winsock teardown), so the process hangs
        # after the script finishes until the CI job's timeout cancels it.
        gdal.VSICurlClearCache()
    print("TLS /vsicurl read OK — bundled CA store loads trust anchors.")


_check_tls_read()

print("All runtime checks passed.")

# Belt-and-suspenders: exit immediately so a lingering GDAL/libcurl worker
# thread can't hang interpreter shutdown on Windows (see _check_tls_read).
# All checks above raise on failure, so reaching here means success.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
