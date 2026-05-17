# Wheel Build Flow

How `build-wheels.yml` produces platform wheels and how pip clients pick
the right one.

## What gets built per release

`build-wheels.yml` produces **16 platform wheels + 1 sdist** per release:

| Platform | Architecture                   | Python versions        | Wheels |
|----------|--------------------------------|------------------------|--------|
| Linux    | x86_64                         | 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | arm64 (Apple Silicon)          | 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | x86_64 (Intel, cross-compiled) | 3.11, 3.12, 3.13, 3.14 | 4      |
| Windows  | AMD64 (x64)                    | 3.11, 3.12, 3.13, 3.14 | 4      |
| (any)    | sdist                          | —                      | 1      |

**Total: 16 wheels + 1 sdist.**

The macOS x86_64 wheels are cross-compiled on a `macos-14` (arm64)
runner via Rosetta + ARCHFLAGS — GitHub's `macos-13` (Intel) runner
queue is unusable in practice (jobs sit queued for hours), so we
dropped that runner and use cibuildwheel's cross-compile path
instead.

## Wheel filenames

Each wheel is tagged with its compatibility info:

```
pyramids_gis-0.20.0-cp311-cp311-manylinux_2_39_x86_64.whl     # Linux 3.11
pyramids_gis-0.20.0-cp312-cp312-manylinux_2_39_x86_64.whl     # Linux 3.12
pyramids_gis-0.20.0-cp313-cp313-manylinux_2_39_x86_64.whl     # Linux 3.13
pyramids_gis-0.20.0-cp311-cp311-macosx_11_0_arm64.whl         # macOS arm64 3.11
pyramids_gis-0.20.0-cp311-cp311-macosx_11_0_x86_64.whl        # macOS x86_64 3.11
...
pyramids_gis-0.20.0-cp313-cp313-win_amd64.whl                 # Windows 3.13
pyramids_gis-0.20.0.tar.gz                                    # sdist
```

The compatibility tag (e.g. `cp312-cp312-manylinux_2_39_x86_64`) tells
pip exactly which Python ABI + OS + architecture this wheel was built
for.

## Why separate wheels per OS / arch / Python version?

### Per OS / architecture — native libraries differ

Each wheel bundles **the GDAL shared library compiled for that
specific platform**:

- **Linux wheels** contain `libgdal-<hash>.so.38`, `libproj-<hash>.so.25`,
  `libgeos-<hash>.so.3.14`, etc.
- **macOS wheels** contain `libgdal-<hash>.36.dylib`,
  `libproj-<hash>.25.dylib`, etc.
- **Windows wheels** contain `gdal-<hash>.dll`, `proj-<hash>.dll`, etc.,
  plus the GDAL driver plugin DLLs (`gdal_netCDF.dll`, `gdal_HDF*.dll`)
  and their transitive deps (`netcdf.dll`, `hdf5.dll`, …) bundled by
  `delvewheel --analyze-existing`.

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
│   └── python -m build --sdist → pyramids_gis-X.Y.Z.tar.gz
│
├── build-linux-wheels (1 job, ubuntu-latest, builds 4 wheels)
│   └── cibuildwheel:
│       ├── CIBW_BEFORE_ALL (once):
│       │   bash ci/setup-gdal-from-pixi.sh
│       │   → installs pixi → installs wheel-build env
│       │   → extracts libgdal.so + transitive deps into /usr/local
│       │   → writes ${BUILD_PREFIX}/GDAL_VERSION (read from
│       │     gdal-config — single source of truth = pixi.lock)
│       ├── For each of cp311, cp312, cp313, cp314:
│       │   ├── CIBW_BEFORE_BUILD (per Python version):
│       │   │   python ci/install-and-vendor-osgeo.py
│       │   │   → reads GDAL version from
│       │   │     ${BUILD_PREFIX}/GDAL_VERSION
│       │   │   → pip install GDAL==X.Y.Z against the bundled libgdal
│       │   │   → vendors osgeo/ + osgeo_utils/ → src/pyramids/_vendor/
│       │   │   → vendors GDAL_DATA + PROJ_DATA → src/pyramids/_data/
│       │   ├── CIBW_BUILD:
│       │   │   pip wheel . → pyramids_gis-X.Y.Z-cp3NN-cp3NN-linux_x86_64.whl
│       │   └── CIBW_REPAIR_WHEEL_COMMAND:
│       │       auditwheel repair → manylinux_2_39_x86_64.whl
│       │       (bundles libgdal.so + transitive deps, patches RPATH)
│       └── upload-artifact: wheels-linux-x86_64
│
├── build-macos-wheels (2 jobs in matrix: arm64 + x86_64, both on macos-14)
│   ├── CIBW_BEFORE_ALL: ci/setup-gdal-from-pixi.sh
│   │   → on macOS this script additionally installs symlinks for
│   │     clang/clang++/otool/install_name_tool/codesign/lipo/strip/...
│   │     into /usr/local/bin pointing at the real Xcode toolchain
│   │     binaries. macos-14 runners SIGKILL xcodebuild on every
│   │     /usr/bin/<tool> invocation (the xcrun dispatch), so the
│   │     symlinks let PATH lookups resolve to working binaries
│   │     directly. clang/clang++ are wrapper scripts that ALSO export
│   │     SDKROOT + DEVELOPER_DIR before exec'ing the real binary so
│   │     the toolchain clang can find system headers.
│   ├── For arch == arm64: native build via pixi --frozen
│   ├── For arch == x86_64: cross-compile path
│   │   → ci/setup-gdal-from-pixi.sh delegates to
│   │     ci/setup-gdal-micromamba.sh (pixi can't install for a
│   │     non-host platform; micromamba natively supports
│   │     --platform osx-64). Re-solves the same dep range pin
│   │     declared in [tool.pixi.feature.wheel-build.dependencies].
│   │   → cibuildwheel sets ARCHFLAGS=-arch x86_64 and runs the
│   │     build venv under Rosetta.
│   ├── CIBW_REPAIR_WHEEL_COMMAND uses delocate-wheel
│   │   (macOS equivalent of auditwheel — patches @loader_path)
│   └── upload-artifact: wheels-macos-arm64 / wheels-macos-x86_64
│
└── build-windows-wheels (1 job, windows-2022, builds 4 wheels)
    └── cibuildwheel:
        ├── CIBW_BEFORE_ALL: powershell -File ci/setup-gdal-from-pixi.ps1
        │   → installs pixi → extracts Library/bin DLLs → C:/gdal-prefix
        │   → writes ${BuildPrefix}/GDAL_VERSION (parsed from
        │     conda-meta/gdal-X.Y.Z-*.json — Windows conda-forge gdal
        │     doesn't ship a usable gdal-config)
        ├── For each of cp311, cp312, cp313, cp314: same vendor + build steps
        └── CIBW_REPAIR_WHEEL_COMMAND uses
            `delvewheel repair --analyze-existing`
            → bundles _gdal.pyd's direct deps AND the GDAL plugin
              DLLs' transitive deps (netcdf.dll, hdf5.dll, ...) into
              pyramids_gis.libs/, patches PE import tables.
    └── upload-artifact: wheels-windows-AMD64
```

After all build jobs finish, `test-wheels` runs a 12-cell matrix
(3 OSes × 4 Python versions) installing each wheel in a clean Python
env and running `pytest -m core`. macOS x86_64 testing is skipped —
the wheel is cross-compiled on an arm64 host so we can't install it
on the same runner, and GitHub's macos-13 queue is unusable.

The matrix uses `os` as a real axis (`os: [ubuntu-latest, macos-14,
windows-2022]`) and `include:` adds per-OS properties (`arch`,
`artifact`, `wheel-tag`). An earlier `include`-only shape silently
collapsed all 9 combos to 3 Windows-only jobs, which masked real
test failures.

Wheels are installed with `pip install --no-deps <wheel>` and the
remaining runtime deps (geopandas, numpy, pandas, …) are installed
explicitly. Without `--no-deps`, pip would try to satisfy the
`GDAL >=3.10.0,<4` line in pyramids' `[project.dependencies]` from
PyPI — which has no Windows wheels for GDAL, so it would force a
from-source GDAL compile inside the test runner. The platform wheel
vendors GDAL's Python bindings under `pyramids/_vendor/osgeo/`, so no
PyPI GDAL is needed at runtime.

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
| Intel Mac + Python 3.13 | `cp313-cp313-macosx_11_0_x86_64` |
| Windows 11 + Python 3.11 | `cp311-cp311-win_amd64` |
| Alpine Linux (musl) | no wheel matches → sdist → fails without system GDAL |
| Ubuntu 22.04 (glibc 2.35) | no wheel matches → sdist → fails without system GDAL |

## CI timing

On GitHub-hosted runners (jobs parallel where possible):

| Job | Duration |
|-----|----------|
| `build-sdist` | ~2 min |
| `build-linux-wheels` | ~12 min (4 wheels) |
| `build-macos-wheels` (arm64, native) | ~6 min (4 wheels) |
| `build-macos-wheels` (x86_64, cross-compiled) | ~7 min (4 wheels) |
| `build-windows-wheels` | ~12 min (4 wheels) |
| `test-wheels` matrix (12 jobs) | ~3 min (parallel, after builds) |

**Total wall-clock per release: ~15 min** — all build jobs run in
parallel, then the test matrix.

Each step has an explicit `timeout-minutes` cap (10 for sdist, 30
for each platform build, 20 for the whole test-wheels job, 10 for
its pytest step) plus pytest's own `--timeout=60
--timeout-method=thread` so a hung test fails fast on the runner
that can least afford to babysit (Windows).

## Publishing

Publishing to PyPI lives in `.github/workflows/pypi-release.yml`
(token-based `twine` upload via pixi). `build-wheels.yml` only
**builds + tests** the wheels and uploads them as run artifacts; it
doesn't publish. Wheel artifacts are downloadable from the GitHub
Actions UI for ~90 days, useful for sanity-checking a build locally
before tagging a release.

## Local builds

You can replicate any single OS's wheel build locally if you have
Docker (for Linux) or the host OS (for macOS / Windows):

```bash
# Linux (Docker; can run from any host OS)
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64

# macOS (must run on macOS)
cibuildwheel --only cp312-macosx_arm64
cibuildwheel --only cp312-macosx_x86_64   # cross-compile from arm64

# Windows (must run on Windows)
cibuildwheel --only cp312-win_amd64
```

## File map

| File | Role |
|------|------|
| `.github/workflows/build-wheels.yml` | The full pipeline (build + test) |
| `.github/workflows/pypi-release.yml` | PyPI publish (token-based twine) |
| `ci/setup-gdal-from-pixi.sh` | Linux + macOS native: install pixi, extract conda-forge binaries, install macOS toolchain symlinks |
| `ci/setup-gdal-micromamba.sh` | macOS cross-compile: install micromamba and resolve target-platform env |
| `ci/setup-gdal-from-pixi.ps1` | Windows: PowerShell version of the pixi setup |
| `ci/install-and-vendor-osgeo.py` | Per-Python: build GDAL SWIG bindings + vendor osgeo + data + patch vendored `osgeo/__init__.py` for Windows DLL bootstrap |
| `ci/check-wheel-size.sh` | Enforces the `WHEEL_SIZE_BUDGET_MB` ceiling per built wheel |
| `pyproject.toml` `[tool.cibuildwheel.*]` | cibuildwheel config per OS |
| `pyproject.toml` `[tool.pixi.feature.wheel-build]` | Minimal pixi env with GDAL native deps |
| `setup.py` | `BinaryDistribution` override to force platform-specific wheel |
| `src/pyramids/__init__.py` | Runtime bootstrap that loads the vendored osgeo + prepends `pyramids_gis.libs` to PATH on Windows |

## Pitfalls worth remembering

These are surprises we hit while stabilizing the pipeline (preserved
so the next person doesn't have to rediscover them):

- **cibuildwheel's `environment` option uses REPLACE semantics**, not
  table-merge, across platform overrides. A top-level
  `[tool.cibuildwheel.environment]` is ignored as soon as
  `[tool.cibuildwheel.<platform>.environment]` exists for the same
  platform. Shared env vars must be duplicated per platform.
- **macos-14 runners SIGKILL xcodebuild** regardless of which Xcode is
  selected. `xcrun -f <tool>` and `/usr/bin/<tool>` shims that
  dispatch through xcodebuild always fail. The
  `/usr/local/bin/clang*` wrappers + plain symlinks for the other
  toolchain binaries are the workaround.
- **`os.add_dll_directory` is process-local** on Windows. Multiprocessing
  `spawn` workers don't inherit it. The vendored `osgeo/__init__.py`
  is patched to call `os.add_dll_directory` itself, so spawn workers
  that import osgeo before pyramids still resolve `gdal.dll`.
- **GDAL's native plugin loader uses raw `LoadLibrary`** which doesn't
  honor `os.add_dll_directory` (no `LOAD_LIBRARY_SEARCH_USER_DIRS`
  flag). The runtime bootstrap prepends `pyramids_gis.libs` to `PATH`
  so the GDAL plugin DLLs' transitive deps are findable via the
  legacy DLL search order.
- **numpy 2.x macosx_14_0_arm64 wheel** uses Accelerate ILP64 symbols
  not present on macos-14 runners; pip picks it by default for
  cibuildwheel's framework Python. The arm64 build forces the
  `macosx_11_0_arm64` numpy wheel via `pip download --platform`.
- **pip on Windows can't build GDAL from source** (no compiler, no
  GDAL headers). The wheel-test job installs with `--no-deps` and
  installs the runtime deps separately so pip doesn't try to resolve
  the `GDAL >=3.10.0` line at install time.
