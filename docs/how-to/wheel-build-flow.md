# Wheel Build Flow

How `build-wheels.yml` produces platform wheels and how pip clients pick
the right one.

## What gets built per release

`build-wheels.yml` produces **12 platform wheels + 1 sdist** per release:

| Platform | Architecture | Python versions | Wheels |
|----------|--------------|-----------------|--------|
| Linux | x86_64 | 3.11, 3.12, 3.13 | 3 |
| macOS | x86_64 (Intel) | 3.11, 3.12, 3.13 | 3 |
| macOS | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 | 3 |
| Windows | AMD64 (x64) | 3.11, 3.12, 3.13 | 3 |
| (any) | sdist | — | 1 |

**Total: 12 wheels + 1 sdist.**

## Wheel filenames

Each wheel is tagged with its compatibility info:

```
pyramids_gis-0.16.0-cp311-cp311-manylinux_2_39_x86_64.whl     # Linux 3.11
pyramids_gis-0.16.0-cp312-cp312-manylinux_2_39_x86_64.whl     # Linux 3.12
pyramids_gis-0.16.0-cp313-cp313-manylinux_2_39_x86_64.whl     # Linux 3.13
pyramids_gis-0.16.0-cp311-cp311-macosx_11_0_x86_64.whl        # macOS Intel 3.11
pyramids_gis-0.16.0-cp311-cp311-macosx_11_0_arm64.whl         # macOS Apple Silicon 3.11
...
pyramids_gis-0.16.0-cp313-cp313-win_amd64.whl                 # Windows 3.13
pyramids_gis-0.16.0.tar.gz                                    # sdist
```

The compatibility tag (e.g. `cp312-cp312-manylinux_2_39_x86_64`) tells pip
exactly which Python ABI + OS + architecture this wheel was built for.

## Why separate wheels per OS / arch / Python version?

### Per OS / architecture — native libraries differ

Each wheel bundles **the GDAL shared library compiled for that
specific platform**:

- **Linux wheels** contain `libgdal-<hash>.so.38`, `libproj-<hash>.so.25`,
  `libgeos-<hash>.so.3.14`, etc.
- **macOS wheels** contain `libgdal-<hash>.36.dylib`,
  `libproj-<hash>.25.dylib`, etc.
- **Windows wheels** contain `gdal-<hash>.dll`, `proj-<hash>.dll`, etc.

A `.so` won't load on macOS, a `.dylib` won't load on Windows. Each
platform needs its own native library bundle.

### Per Python version — SWIG bindings are ABI-specific

The GDAL Python SWIG bindings (`_gdal.so`, `_ogr.so`, `_osr.so`, etc.)
are compiled per-Python-version. `cp311`'s
`_gdal.cpython-311-x86_64-linux-gnu.so` won't load in cp312 — different
Python C API ABI.

## CI build flow

```
.github/workflows/build-wheels.yml
│
├── build-sdist (1 job, ubuntu-latest)
│   └── python -m build --sdist → pyramids_gis-0.16.0.tar.gz
│
├── build-linux-wheels (1 job, builds 3 wheels in cibuildwheel matrix)
│   └── runs-on: ubuntu-latest
│   └── cibuildwheel:
│       ├── CIBW_BEFORE_ALL  (once per platform):
│       │   bash ci/setup-gdal-from-pixi.sh
│       │   → installs pixi → installs wheel-build env
│       │   → extracts libgdal.so + 150 deps into /usr/local
│       ├── For each of cp311, cp312, cp313:
│       │   ├── CIBW_BEFORE_BUILD (per Python version):
│       │   │   python ci/install-and-vendor-osgeo.py
│       │   │   → pip install GDAL==<X.Y.Z> against libgdal
│       │   │   → copy osgeo/ + osgeo_utils/ → src/pyramids/_vendor/
│       │   │   → copy GDAL_DATA + PROJ_DATA → src/pyramids/_data/
│       │   ├── CIBW_BUILD:
│       │   │   pip wheel . → pyramids_gis-X.Y.Z-cp3NN-cp3NN-linux_x86_64.whl
│       │   └── CIBW_REPAIR_WHEEL_COMMAND:
│       │       auditwheel repair → manylinux_2_39_x86_64.whl
│       │       (bundles libgdal.so + transitive deps, patches RPATH)
│       └── upload-artifact: wheels-linux-x86_64
│
├── build-macos-wheels (2 jobs in matrix: x86_64 + arm64)
│   ├── macos-13 (Intel): builds 3 wheels (cp311/312/313 × x86_64)
│   └── macos-14 (Apple Silicon): builds 3 wheels (cp311/312/313 × arm64)
│   └── Same cibuildwheel pattern as Linux but:
│       ├── ci/setup-gdal-from-pixi.sh handles .dylib paths
│       ├── BUILD_PREFIX = /Users/runner/gdal-prefix (writable on macOS)
│       └── CIBW_REPAIR_WHEEL_COMMAND uses delocate-wheel
│           (macOS equivalent of auditwheel — patches @loader_path)
│   └── upload-artifact: wheels-macos-x86_64 / wheels-macos-arm64
│
└── build-windows-wheels (1 job, builds 3 wheels, windows-2022)
    └── cibuildwheel:
        ├── CIBW_BEFORE_ALL: powershell -File ci/setup-gdal-from-pixi.ps1
        │   → installs pixi → extracts Library/bin DLLs → C:/gdal-prefix
        ├── For each of cp311, cp312, cp313: same vendor + build steps
        └── CIBW_REPAIR_WHEEL_COMMAND uses delvewheel
            (Windows equivalent of auditwheel — bundles DLLs into
             pyramids_gis.libs/ and patches PE import table)
    └── upload-artifact: wheels-windows-AMD64
```

After all build jobs finish, `test-wheels` runs a 12-cell matrix (3
OSes × 3 archs × 3 Python versions, minus combinations that aren't
built) installing each wheel in a clean Python env and running
`pytest -m core`.

## How pip picks the right wheel for users

When a user runs `pip install pyramids-gis`, pip:

1. Asks PyPI for the available files for the package (sdist + all wheels)
2. Filters by the user's platform tags
   - Linux: looks for `manylinux_X_Y_*` ≤ their glibc, or `manylinux2014`,
     etc.
   - macOS: looks for `macosx_X_Y_*` ≤ their OS version
   - Windows: looks for `win_amd64`
3. Filters by their Python version
   - `cp311` for Python 3.11, `cp312` for 3.12, etc.
4. Downloads the single matching wheel (~58 MB)
5. Falls back to the sdist if no wheel matches (which then requires
   system GDAL to be installed for `pip install` to succeed)

Examples:

| User environment | Wheel pip picks |
|------------------|-----------------|
| Ubuntu 24.04 + Python 3.12 | `cp312-cp312-manylinux_2_39_x86_64` |
| M2 Mac + Python 3.13 | `cp313-cp313-macosx_11_0_arm64` |
| Windows 11 + Python 3.11 | `cp311-cp311-win_amd64` |
| Alpine Linux (musl) | no wheel matches → sdist → fails without system GDAL |
| Ubuntu 22.04 (glibc 2.35) | no wheel matches → sdist → fails without system GDAL |

## CI timing

On GitHub-hosted runners (jobs parallel where possible):

| Job | Duration |
|-----|----------|
| `build-sdist` | ~2 min |
| `build-linux-wheels` | ~10 min (3 wheels) |
| `build-macos-wheels` (Intel) | ~12 min (3 wheels) |
| `build-macos-wheels` (arm64) | ~12 min (3 wheels) |
| `build-windows-wheels` | ~15 min (3 wheels) |
| `test-wheels` matrix (12 jobs) | ~5 min (parallel, after builds) |
| `publish` (when enabled) | ~1 min |

**Total wall-clock per release: ~20 min** — all build jobs run in
parallel, then test, then publish.

## What gets uploaded to PyPI

When the `publish:` job runs (currently commented out in
`build-wheels.yml`), `twine upload dist/*` ships everything in `dist/`
after artifacts are downloaded:

- 12 platform wheels
- 1 sdist

PyPI then serves all 13 from `https://pypi.org/project/pyramids-gis/<version>/`
and pip clients auto-select the right one per the rules above.

## Local builds

You can replicate any single OS's wheel build locally if you have
Docker (for Linux) or the host OS (for macOS/Windows):

```bash
# Linux
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64

# Windows (run from a Windows machine)
cibuildwheel --only cp312-win_amd64

# macOS (run from a Mac)
cibuildwheel --only cp312-macosx_arm64
```

cibuildwheel uses Docker for Linux builds even when invoked from
Windows / macOS, so cross-platform local builds are possible for
Linux but not for macOS/Windows (those need the actual host OS).

## File map

| File | Role |
|------|------|
| `.github/workflows/build-wheels.yml` | The full pipeline (build + test + publish) |
| `ci/setup-gdal-from-pixi.sh` | Linux + macOS: install pixi, extract conda-forge binaries |
| `ci/setup-gdal-from-pixi.ps1` | Windows: PowerShell version of the above |
| `ci/install-and-vendor-osgeo.py` | Per-Python: build GDAL SWIG bindings + vendor osgeo + data |
| `pyproject.toml` `[tool.cibuildwheel.*]` | cibuildwheel config per OS |
| `pyproject.toml` `[tool.pixi.feature.wheel-build]` | Minimal pixi env with GDAL native deps |
| `setup.py` | `BinaryDistribution` override to force platform-specific wheel |
| `src/pyramids/__init__.py` | Runtime bootstrap that loads the vendored osgeo |
