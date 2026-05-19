# Troubleshooting

Common installation and runtime issues with pyramids-gis and how to
fix them.

## Installation issues

### `ERROR: ... is not a supported wheel on this platform`

**Symptoms:**
```
ERROR: pyramids_gis-0.13.0-cp312-cp312-manylinux_2_39_x86_64.whl
is not a supported wheel on this platform.
```

**Cause:** Your Linux system has glibc older than 2.39. Our wheels are
tagged `manylinux_2_39` because conda-forge's GDAL is compiled with
GCC 13 and references GLIBCXX_3.4.32 symbols only available in glibc 2.39+
(Ubuntu 24.04+, RHEL 10+, Debian 13+).

**Fix options (pick one):**

1. **Upgrade your distro** to one with glibc 2.39+.

2. **Use conda-forge** instead of pip:
   ```console
   conda install -c conda-forge pyramids
   ```
   conda-forge's GDAL doesn't need that glibc baseline.

3. **Install pyramids-gis from sdist** with system GDAL:
   ```console
   # Install system GDAL first
   sudo apt install libgdal-dev            # Debian/Ubuntu
   sudo dnf install gdal-devel             # Fedora/RHEL
   # Then install pyramids
   pip install --no-binary pyramids-gis pyramids-gis
   ```

**Check your glibc version:**
```console
ldd --version | head -1
# e.g., "ldd (Ubuntu GLIBC 2.39-0ubuntu8.3) 2.39"
```

---

### `gdal-config: command not found` when pip-installing from sdist

**Symptoms:**
```
gdal_config_error: [Errno 2] No such file or directory: 'gdal-config'
Could not find gdal-config. Make sure you have installed the GDAL native
library and development headers.
```

**Cause:** pip fell back to the sdist (source distribution) instead of
a platform wheel. The sdist requires a pre-installed native GDAL. This
can happen if:
- No wheel matches your platform
- You passed `--no-binary` or `--only-binary=:none:`
- Your pip is very old (< 19.0) and doesn't understand the manylinux tag

**Fix:**
```console
# Confirm pip version
pip --version                    # needs >= 19.0; 24+ recommended

# Upgrade pip
pip install --upgrade pip

# Retry install (should pick up the wheel now)
pip install pyramids-gis --force-reinstall
```

If you genuinely don't have a matching wheel, either use conda-forge
or install system GDAL first (see [installation.md](installation.md)).

---

### `Access is denied` / DLL copy fails on Windows pixi install

**Symptoms (during `pixi install`):**
```
Error:   × failed to link freexl-2.0.0-hf297d47_2.conda
  ├─▶ failed to link 'Library/bin/freexl.dll'
  ╰─▶ Access is denied. (os error 5)
```

**Cause:** Windows Defender (or another antivirus) is scanning the DLL
mid-copy and holds a file lock that pixi can't bypass.

**Fix options:**

1. **Add `.pixi/` to Windows Defender's exclusion list:**
   Settings → Privacy & security → Windows Security → Virus & threat
   protection → Manage settings → Add or remove exclusions → Add
   folder → select your project's `.pixi/` dir.

2. **Temporarily disable real-time protection**, run `pixi install`,
   then re-enable. Only do this if you trust the source.

3. **Use a different user profile** (Windows Defender behaves
   per-user).

---

## Runtime issues

### `ImportError: libexpat.so.1: cannot open shared object file`

**Symptoms:**
```python
>>> import pyramids
ImportError: libexpat.so.1: cannot open shared object file: No such
file or directory
```

**Cause:** You're on a minimal Linux image (`python:3.12-slim`,
Alpine, scratch) that doesn't ship `libexpat`. Most full Linux distros
have it installed by default.

**Fix (Debian/Ubuntu slim images):**
```console
apt-get update && apt-get install -y libexpat1
```

**Fix (RPM-based):**
```console
dnf install -y expat
```

**Fix (Alpine — note: wheel isn't supported on Alpine yet; use conda).**

---

### `No module named '_gdal'` when importing osgeo

**Symptoms:**
```python
>>> from osgeo import gdal
ModuleNotFoundError: No module named '_gdal'
```

**Cause:** `from osgeo import gdal` ran **before** `import pyramids`,
so pyramids' vendor bootstrap didn't run. The bootstrap is what adds
the vendored `osgeo` package to `sys.path`.

**Fix:**
```python
import pyramids          # runs the vendor bootstrap
from osgeo import gdal   # now resolves to vendored copy
```

If you absolutely cannot put `import pyramids` first, the vendored
osgeo will still be importable via its full path:

```python
from pyramids._vendor.osgeo import gdal
```

---

### `plugin gdal_netCDF.so is not available` / NetCDF/HDF4/HDF5 drivers missing

**Symptoms:**
```
RuntimeError: plugin gdal_netCDF.so is not available in your installation.
The GDAL_DRIVER_PATH configuration option is not set.
```

**Cause:** The bundled plugins directory isn't being found. This
shouldn't happen with the standard wheel — it usually means something
overrode `GDAL_DRIVER_PATH` after `import pyramids` ran.

**Diagnose:**
```python
import os, pyramids
print("GDAL_DRIVER_PATH:", os.environ.get("GDAL_DRIVER_PATH"))
# Should print .../pyramids/_data/gdalplugins
```

**Fix:** Don't override `GDAL_DRIVER_PATH` to an empty value. If you
legitimately need plugins from multiple sources, concatenate:

```python
import os, pyramids
existing = os.environ.get("GDAL_DRIVER_PATH", "")
os.environ["GDAL_DRIVER_PATH"] = f"/my/extra/plugins:{existing}"
```

---

### `EPSG lookup failed` / `Unable to open EPSG support file`

**Symptoms:**
```
TypeError: in method 'SpatialReference_ImportFromEPSG':
Unable to open EPSG support file gcs.csv
```

**Cause:** `PROJ_DATA` / `GDAL_DATA` environment variables point to a
non-existent directory. Usually because user code overrides them.

**Diagnose:**
```python
import os, pyramids
print("GDAL_DATA:", os.environ.get("GDAL_DATA"))
print("PROJ_DATA:", os.environ.get("PROJ_DATA"))
# Both should point under pyramids/_data/
```

**Fix:** Don't set these env vars manually. pyramids sets them
automatically when imported, using `os.environ.setdefault` so it
respects any pre-existing values.

---

### Conflict with another `osgeo` install

**Symptoms:**
- Unexpected GDAL version reported
- "too-recent versioned symbols" or similar linker errors
- `osgeo.__file__` points to `/usr/local/lib/python3.12/site-packages/osgeo`
  (system install) instead of the vendored path

**Cause:** You have the standalone PyPI `GDAL` package or a system
`osgeo` installed, and it's winning the import race.

**Diagnose:**
```python
import pyramids
import osgeo
print(osgeo.__file__)
# Should contain "_vendor/osgeo"
```

**Fix:**
```console
pip uninstall GDAL           # remove the standalone PyPI package
pip uninstall gdal           # older capitalization
pip install pyramids-gis --force-reinstall
```

If you're in a conda env where osgeo comes from conda's `gdal`, decide:
- **Keep conda's GDAL**: stick with the conda-forge `pyramids` package
- **Use pyramids-gis PyPI wheel**: make a fresh venv without conda

Mixing them is not supported.

---

### On Windows: `ImportError: DLL load failed` for `_gdal.pyd`

**Symptoms:**
```
ImportError: DLL load failed while importing _gdal: The specified module
could not be found.
```

**Cause:** On Windows, even when auditwheel/delvewheel bundles the DLLs,
Python needs to know where to find them. pyramids calls
`os.add_dll_directory()` at import time — but only if pyramids is
imported **before** `osgeo`.

**Fix:**
```python
import pyramids            # adds the bundled DLLs dir
from osgeo import gdal     # now finds them
```

---

### Wheel too large / slow download

Our wheel is ~63 MB (vs rasterio's ~26 MB) because it bundles HDF4,
HDF5, and NetCDF drivers. If you don't need these:

- The conda-forge package is modular and only pulls in HDF4/NetCDF if
  you explicitly install `libgdal-netcdf`, `libgdal-hdf4`.
- We don't currently ship a "lite" wheel without these drivers. File an
  issue if that would be valuable.

---

## Getting more help

### Enable diagnostic output

Set an env var before importing pyramids:

```console
PYRAMIDS_DEBUG_BOOTSTRAP=1 python -c "import pyramids"
```

Output:
```
[pyramids] vendor dir: /.../site-packages/pyramids/_vendor
[pyramids] GDAL_DATA: /.../site-packages/pyramids/_data/gdal_data
[pyramids] PROJ_DATA: /.../site-packages/pyramids/_data/proj_data
```

### File an issue

If the above doesn't solve your problem:

<https://github.com/Serapieum-of-alex/pyramids/issues/new>

Include:

- `pyramids --version` output
- `python -c "import pyramids; from osgeo import gdal; print(pyramids.__version__, gdal.__version__)"`
- `pip show pyramids-gis | head -10`
- Your OS + version (`uname -a` on Linux, `sw_vers` on macOS)
- `ldd --version | head -1` (Linux) or `otool -L` path on macOS
- Full traceback of the error
