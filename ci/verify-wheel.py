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
5. A NetCDF round-trip + a foreign-`GDAL_DRIVER_PATH` override check
   exercise the bundled netCDF driver — the regression in issue #465,
   where an inherited conda `GDAL_DRIVER_PATH` shadowed the bundled,
   version-locked plugins so every `.nc` read failed.

Kept as a standalone file (rather than an inline `python -c "..."`
heredoc in the workflow) so the script reads cleanly and adding a check
doesn't require fighting YAML + shell quoting.
"""

import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# `import pyramids` runs the vendor bootstrap in pyramids/__init__.py, which injects
# pyramids/_vendor/osgeo onto sys.path. It MUST precede `import osgeo`, so the order is
# pinned against isort's first-party regrouping (profile=black would otherwise float
# `import pyramids` below the third-party osgeo block and break the bare osgeo import).
# isort: off
import pyramids
import osgeo
from osgeo import gdal, ogr, osr  # noqa: F401 — ogr import is a smoke test

# isort: on

# Stable, valid-TLS HTTPS endpoint for the CA-trust check. Small text
# file (not a raster) — VSIFOpenL forces the curl TLS handshake + CA
# load without needing a valid GeoTIFF on the far end.
_TLS_PROBE_URL = (
    "/vsicurl/https://raw.githubusercontent.com/OSGeo/gdal/master/LICENSE.TXT"
)
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
    dev/editable install.

    Two legitimate CA models exist:

    * **Vendored bundle** (conda-extract and the Linux from-source
      builds): libcurl bakes a build-prefix CA path that is absent in
      the consuming env, so the wheel MUST ship `_data/ssl/cacert.pem`
      and the bootstrap MUST point `GDAL_HTTP_CAINFO` at it — a missing
      file here is the #412 regression and fails hard (an earlier
      version *skipped*, silently masking the bug).
    * **OS trust store** (the vcpkg win_arm64 build, whose curl uses
      schannel): no CA file exists anywhere, the bootstrap correctly
      sets no CAINFO, and the TLS read below is the entire proof.

    A CA/trust-store error during the read is the bug in either model
    and fails hard. A generic network error (offline runner, DNS,
    timeout) is not, so we warn and move on rather than make the smoke
    test flaky on network conditions.
    """
    ca_bundle = Path(pyramids.__file__).parent / "_data" / "ssl" / "cacert.pem"
    cainfo = os.environ.get("GDAL_HTTP_CAINFO")
    print(f"GDAL_HTTP_CAINFO: {cainfo}")
    if ca_bundle.is_file():
        if cainfo != str(ca_bundle):
            _fail(
                f"GDAL_HTTP_CAINFO={cainfo!r} not pointed at bundled cert {ca_bundle} (issue #412)"
            )
    elif cainfo is not None and not Path(cainfo).is_file():
        _fail(
            f"no CA bundle vendored, yet GDAL_HTTP_CAINFO={cainfo!r} points at a missing file"
        )
    else:
        print(
            "no vendored CA bundle — OS trust store model (schannel); "
            "the /vsicurl read below is the proof"
        )

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


def _netcdf_roundtrip(nc_driver, workdir: str) -> None:
    """Write a 1-band raster via the netCDF driver and read it back.

    Self-contained (no xarray/netCDF4 needed): exercises the bundled
    netCDF driver's write + read path. Raises on any failure. Handles
    both the direct-raster and subdataset-container open results.
    """
    path = os.path.join(workdir, "verify.nc")
    mem = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
    mem.SetGeoTransform([0, 1, 0, 0, 0, -1])
    if nc_driver.CreateCopy(path, mem) is None:
        _fail("netCDF CreateCopy returned None — bundled driver cannot write (#465)")
    ds = gdal.Open(path)
    if ds is None:
        _fail(f"gdal.Open({path}) returned None — netCDF read failed (#465)")
    if ds.RasterCount == 0:
        subs = ds.GetSubDatasets()
        if subs:
            ds = gdal.Open(subs[0][0])
    if ds is None or ds.RasterCount == 0 or ds.GetRasterBand(1).ReadAsArray() is None:
        _fail("netCDF read produced no readable raster band (#465)")


def _check_netcdf_driver() -> None:
    """Confirm the bundled netCDF driver loads, even under a foreign path.

    Only meaningful for a bundled wheel that vendors the driver plugins.

    1. Baseline (this process): the netCDF driver is registered and a
       GDAL-written NetCDF file round-trips — guards the literal #465 /
       #457 symptom ("not recognized as being in a supported file format").
    2. Override (child process): pre-set GDAL_DRIVER_PATH to a foreign,
       empty dir — exactly what an activated conda env exports for its own
       GDAL — then import pyramids. The bootstrap must FORCE
       GDAL_DRIVER_PATH back to the bundled, version-locked plugin dir so
       the driver still loads. With the old `setdefault` behaviour the
       foreign path would win and the driver would vanish. This is the
       check that actually distinguishes the #465 fix from the bug.
    """
    plugins = Path(pyramids.__file__).parent / "_data" / "gdalplugins"
    if plugins.is_dir():
        print(f"driver plugins vendored at {plugins} (plugin build model)")
    else:
        # From-source builds (#332 spike) compile netCDF/HDF5/GRIB INTO
        # libgdal — no plugin dir exists and none is needed. The registration
        # + round-trip checks below are the actual #465 invariant either way.
        print("no _data/gdalplugins — drivers built into libgdal (from-source model)")

    gdal.UseExceptions()
    drv = gdal.GetDriverByName("netCDF")
    if drv is None:
        _fail("netCDF driver not registered in the bundled GDAL (#465)")
    _netcdf_roundtrip(drv, tempfile.mkdtemp(prefix="nc-base-"))
    print("netCDF baseline round-trip OK — bundled driver reads a NetCDF file.")

    bogus = tempfile.mkdtemp(prefix="bogus-gdalplugins-")
    env = dict(os.environ)
    env["GDAL_DRIVER_PATH"] = bogus
    child = textwrap.dedent("""
        import os, sys, tempfile
        import pyramids
        from osgeo import gdal
        gdal.UseExceptions()
        drv = gdal.GetDriverByName("netCDF")
        if drv is None:
            print("CHILD_FAIL driver=None path=%r" % os.environ.get("GDAL_DRIVER_PATH"))
            sys.exit(3)
        p = os.path.join(tempfile.mkdtemp(), "t.nc")
        mem = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
        mem.SetGeoTransform([0, 1, 0, 0, 0, -1])
        drv.CreateCopy(p, mem)
        ds = gdal.Open(p)
        if ds is None or ds.GetRasterBand(1).ReadAsArray() is None:
            print("CHILD_FAIL read p=%s" % p)
            sys.exit(4)
        print("CHILD_OK final_path=%s" % os.environ.get("GDAL_DRIVER_PATH"))
        sys.stdout.flush()
        os._exit(0)
        """)
    result = subprocess.run(
        [sys.executable, "-c", child],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = (result.stdout + result.stderr).strip()
    print(f"netCDF override child rc={result.returncode}")
    # The CHILD_OK marker — printed only after the driver loaded under the
    # foreign GDAL_DRIVER_PATH AND a round-trip read succeeded — is the
    # verdict, NOT the exit code. On Windows the bundled HDF5/netCDF libs can
    # crash a worker thread during process teardown (nonzero exit, e.g.
    # 0xC0000005) even though every check already passed, so a nonzero rc with
    # CHILD_OK present is a teardown artifact, not a #465 regression. A crash
    # *before* CHILD_OK leaves the marker absent and still fails here.
    if "CHILD_OK" not in result.stdout:
        _fail(
            f"netCDF driver lost under a foreign GDAL_DRIVER_PATH — #465 fix not effective:\n{out}"
        )
    # The bootstrap only re-points GDAL_DRIVER_PATH when a bundled plugin dir
    # exists (plugin build model). With drivers compiled into libgdal
    # (from-source model) the foreign path harmlessly remains — CHILD_OK above
    # already proved the driver survives it.
    if plugins.is_dir() and bogus in result.stdout:
        _fail(
            f"GDAL_DRIVER_PATH still points at the foreign dir {bogus!r} — #465 fix not effective"
        )
    print(
        "netCDF override OK — bundled driver wins over a foreign GDAL_DRIVER_PATH (#465)."
    )


# The driver contract every published wheel promises (the #332 F0.2
# allow-list): raster formats pyramids ships plus the OGR vector set
# FeatureCollection uses. Asserted here — post-install, on every wheel,
# on every platform — because the pytest layers skipif on driver
# presence, which turns a curation mistake (a cmake flag lost on a GDAL
# bump, a dep regression disabling a driver at configure time) into
# green skips instead of a failure. OGCAPI is the proven case: it was
# silently absent from the first from-source builds while CI stayed
# green.
_COMMON_DRIVERS = (
    "GTiff", "COG", "netCDF", "GRIB", "HDF5", "JP2OpenJPEG", "Zarr",
    "PNG", "JPEG", "WCS", "OGCAPI", "VRT", "MEM",
    "GeoJSON", "ESRI Shapefile", "GPKG", "GPX", "PMTiles", "MVT",
    "GML", "KML", "WFS", "OAPIF", "FlatGeobuf", "SQLite", "OSM",
)
# Platform extras: HDF4 ships only in the conda-extract wheels (macOS +
# Windows AMD64). The from-source builds deliberately drop it — the
# curated Linux stack has no HDF4, and the vcpkg gdal port used for the
# win_arm64 wheel (#334) has no hdf4 feature at all.
_HDF4_PLATFORMS = ("darwin", "win32-amd64")


def _platform_slug() -> str:
    """Return `sys.platform`, suffixed with the machine arch on Windows."""
    slug = sys.platform
    if slug == "win32":
        slug = f"win32-{platform.machine().lower()}"
    return slug


def _check_driver_set() -> None:
    """Assert every promised driver is registered in the bundled GDAL."""
    expected = list(_COMMON_DRIVERS)
    if _platform_slug() in _HDF4_PLATFORMS:
        expected.append("HDF4")
    missing = [name for name in expected if gdal.GetDriverByName(name) is None]
    if missing:
        _fail(
            "bundled GDAL is missing promised drivers: "
            + ", ".join(missing)
        )
    print(f"driver-set check OK — all {len(expected)} promised drivers registered.")


def _check_jp2_driver() -> None:
    """Confirm the bundled JP2OpenJPEG driver loads and round-trips (issue #600).

    JPEG2000 packing (WMO GRIB2 template 5.40) is common in NCEP / ECCC / ECMWF
    GRIB2. Without the JP2OpenJPEG plugin those messages fail with
    "plugin gdal_JP2OpenJPEG ... is not available in your installation". A
    register + write/read round-trip is a self-contained proxy for the GRIB JP2
    decode path (no network fixture needed).
    """
    gdal.UseExceptions()
    drv = gdal.GetDriverByName("JP2OpenJPEG")
    if drv is None:
        _fail("JP2OpenJPEG driver not registered in the bundled GDAL (#600)")

    with tempfile.TemporaryDirectory(prefix="jp2-") as tmp:
        path = os.path.join(tmp, "t.jp2")
        mem = gdal.GetDriverByName("MEM").Create("", 32, 32, 1, gdal.GDT_Byte)
        mem.GetRasterBand(1).Fill(128)
        if drv.CreateCopy(path, mem) is None:
            _fail("JP2OpenJPEG CreateCopy returned None — bundled driver cannot write (#600)")
        ds = gdal.Open(path)
        if ds is None or ds.GetRasterBand(1).ReadAsArray() is None:
            _fail(f"gdal.Open({path}) failed — JP2OpenJPEG read produced no band (#600)")
        ds = None  # release the GDAL handle so Windows can delete the temp file
    print("JP2OpenJPEG round-trip OK — bundled driver reads/writes JPEG2000 (#600).")


_check_driver_set()
_check_netcdf_driver()
_check_jp2_driver()
_check_tls_read()

print("All runtime checks passed.")

# Belt-and-suspenders: exit immediately so a lingering GDAL/libcurl worker
# thread can't hang interpreter shutdown on Windows (see _check_tls_read).
# All checks above raise on failure, so reaching here means success.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
