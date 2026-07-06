# Wheel Build Flow

How `bundle-pypi-wheels.yml` produces platform wheels and how pip clients pick
the right one.

Two build models coexist in the pipeline:

- **Linux (glibc + musl): from-source.** The whole native stack (GDAL, PROJ, GEOS,
  HDF5, netCDF, and ~20 codec/support libraries) is compiled inside the
  cibuildwheel container from SHA256-pinned source tarballs
  (`ci/source-build/config.sh`), so the
  wheel tags at the build image's own floor — `manylinux_2_28` / `musllinux_1_2`.
- **macOS / Windows: conda-extract.** Prebuilt conda-forge binaries are extracted
  into a prefix (`ci/setup-gdal-from-pixi.{sh,ps1}`) and bundled by
  delocate / delvewheel, unchanged from the original pipeline.

## What gets built per release

`bundle-pypi-wheels.yml` produces **23 published platform wheels + 1 sdist** per
release, plus 8 unpublished musl canary wheels:

| Platform | Architecture                        | Python versions        | Wheels |
|----------|-------------------------------------|------------------------|--------|
| Linux    | x86_64 (`manylinux_2_28`)           | 3.11, 3.12, 3.13, 3.14 | 4      |
| Linux    | aarch64 (`manylinux_2_28`)          | 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | arm64 (Apple Silicon, `macosx_11_0`)| 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | x86_64 (Intel, cross-compiled)      | 3.11, 3.12, 3.13, 3.14 | 4      |
| Windows  | AMD64 (x64)                         | 3.11, 3.12, 3.13, 3.14 | 4      |
| Windows  | ARM64 (`win_arm64`, vcpkg build)    | 3.12, 3.13, 3.14       | 3      |
| (any)    | sdist                               | —                      | 1      |

**Total published: 23 wheels + 1 sdist.** One **CI canary** family builds and
verifies on every run but is deliberately **not published** because pip could
not resolve pyramids-gis on those platforms yet:

- `build-musl-wheels`: `musllinux_1_2` x86_64 + aarch64 (cp311–cp314) — blocked
  on upstream pyogrio musllinux wheels (#333; geopandas hard-requires pyogrio).

The `win_arm64` wheels (`build-winarm64-wheels`, cp312–cp314; numpy/scipy ship
no cp311 arm64 wheels) are built from source via vcpkg. GDAL comes from the
vcpkg port (currently 3.12.4, trailing the 3.13.1 the other wheels ship);
like Linux, no HDF4. Because shapely and pyogrio publish no win_arm64 wheels
on PyPI, the platform markers in `[project.dependencies]` skip them (and
geopandas) there, and each wheel **vendors the vector stack** — shapely,
geopandas, and pyogrio, built from sdist against the same vcpkg prefix —
under `pyramids/_vendor/` (see `ci/install-and-vendor-osgeo.py`). The
vendoring and the markers both go away once upstream ships win_arm64 wheels.

The macOS x86_64 wheels are cross-compiled on a `macos-14` (arm64)
runner via Rosetta + ARCHFLAGS — GitHub's `macos-13` (Intel) runner
queue is unusable in practice (jobs sit queued for hours), so we
dropped that runner and use cibuildwheel's cross-compile path
instead.

Linux aarch64 wheels build natively on GitHub's
`ubuntu-24.04-arm` runner (no emulation), so build time is comparable
to x86_64 rather than the 8–10× hit you'd get under QEMU.

## Wheel filenames

Each wheel is tagged with its compatibility info:

```
pyramids_gis-0.40.0-cp311-cp311-manylinux_2_28_x86_64.whl     # Linux x86_64 3.11
pyramids_gis-0.40.0-cp312-cp312-manylinux_2_28_x86_64.whl     # Linux x86_64 3.12
pyramids_gis-0.40.0-cp313-cp313-manylinux_2_28_x86_64.whl     # Linux x86_64 3.13
pyramids_gis-0.40.0-cp314-cp314-manylinux_2_28_x86_64.whl     # Linux x86_64 3.14
pyramids_gis-0.40.0-cp311-cp311-manylinux_2_28_aarch64.whl    # Linux aarch64 3.11
pyramids_gis-0.40.0-cp312-cp312-manylinux_2_28_aarch64.whl    # Linux aarch64 3.12
pyramids_gis-0.40.0-cp313-cp313-manylinux_2_28_aarch64.whl    # Linux aarch64 3.13
pyramids_gis-0.40.0-cp314-cp314-manylinux_2_28_aarch64.whl    # Linux aarch64 3.14
pyramids_gis-0.40.0-cp311-cp311-macosx_11_0_arm64.whl         # macOS arm64 3.11
pyramids_gis-0.40.0-cp311-cp311-macosx_11_0_x86_64.whl        # macOS x86_64 3.11
...
pyramids_gis-0.40.0-cp314-cp314-win_amd64.whl                 # Windows 3.14
pyramids_gis-0.40.0.tar.gz                                    # sdist
```

The compatibility tag (e.g. `cp312-cp312-manylinux_2_28_x86_64`) tells
pip exactly which Python ABI + OS + architecture this wheel was built
for.

## Platform coverage status

Last reviewed: 2026-07-04 (after the from-source Linux switch, #332, and the
musl canaries, #333).

### What the wheels cover today

Users on the rows below get a **native, ready-to-import wheel** from
PyPI — no compiler, no system GDAL, no conda required:

| OS / arch | Compatibility tag | Distros / versions that match |
|---|---|---|
| Linux x86_64, glibc ≥ 2.28 | `manylinux_2_28_x86_64` | Ubuntu 20.04+, Debian 11+, RHEL 8+, AL2023, Fedora 38+ |
| Linux aarch64, glibc ≥ 2.28 | `manylinux_2_28_aarch64` | Graviton / Ampere / RPi 4+ on the same floors |
| macOS arm64, ≥ 11.0 | `macosx_11_0_arm64` | M1 / M2 / M3 / M4 Macs on macOS 11+ |
| macOS x86_64, ≥ 11.0 | `macosx_11_0_x86_64` | Intel Macs on macOS 11+ (cross-compiled — see note) |
| Windows AMD64 | `win_amd64` | Windows 10+ on x64 hardware |

All five platform wheels exist for Python **3.11, 3.12, 3.13, and 3.14**.

> **macOS x86_64 caveat**: the wheel is cross-compiled on the
> `macos-14` (arm64) runner because GitHub's `macos-13` (Intel) queue
> sits idle for hours. We can't install + import the cross-compiled
> wheel on the same runner, so the CI test matrix doesn't exercise it.
> If you're on an Intel Mac and the wheel fails to load, please open
> an issue.

> **Feature difference vs macOS/Windows**: the from-source Linux wheel does not
> include the HDF4 driver (macOS/Windows conda-extract wheels do). HDF4 is a
> legacy format with heavy build baggage; see `docs/installation.md`.

### What the wheels DON'T cover

Users on these rows fall back to the **sdist** (which then needs system
GDAL ≥ 3.10 + a C/C++ compiler at install time — usually painful) or
the **conda-forge install path**:

| OS / arch | Why no wheel | Recommended install path | Tracking |
|---|---|---|---|
| Linux glibc < 2.28 (RHEL 7, Ubuntu 18.04, …) | below the manylinux_2_28 image floor | conda-forge | intentional |
| Alpine / musl Linux | built + verified in CI, unpublished (pyogrio has no musl wheels) | conda-forge | #333 |
| Free-threaded CPython (`cp31Nt`) | GDAL SWIG bindings + numpy not ready | use a GIL build | #683 |
| Python 3.10 or earlier | excluded by `requires-python = ">= 3.11"` | upgrade Python, or pin `< 0.20` | intentional |
| Python 3.15+ (future) | not yet released by CPython | conda-forge until wheels ship | #335 |
| PyPy | `skip = ["*pp*"]`; GDAL bindings target CPython | use CPython | intentional |

### Why the Linux floor is 2.28 (history)

The original Linux pipeline extracted conda-forge's GDAL, which is compiled
with GCC 13 — its `libstdc++.so.6` needs `GLIBCXX_3.4.32`, forcing a
`manylinux_2_39` tag (glibc ≥ 2.39: Ubuntu 24.04+, Debian 13+, RHEL 10+ only).
Bundling the GCC-13 C++ runtime to lower the tag was investigated and rejected:
the dual-`libstdc++` ODR collision with pyproj/shapely's native libs segfaults
at runtime (full investigation in #332).

The fix — shipped 2026-07 — is the **from-source model**: compile the entire
stack inside the `manylinux_2_28` image with its own toolchain, so libgdal
links the baseline libstdc++ and bundles none. That is exactly the
same approach rasterio and fiona use; `ci/source-build/config.sh` owns the
recipe. The cost is ownership of ~25 dependency version
pins; the win is covering Ubuntu 20.04/22.04, Debian 11/12, RHEL 8/9, and
Amazon Linux 2023 with a ~30 MB wheel (vs ~47 MB under conda-extract).

### Coverage roadmap (not committed)

| Gap                               | Issue | Status                 | Notes                                                           |
|-----------------------------------|-------|------------------------|-----------------------------------------------------------------|
| Lower glibc floor (< 2.39)        | #332  | **shipped**            | from-source `manylinux_2_28` wheels (this pipeline)             |
| musllinux (Alpine)                | #333  | **built, unpublished** | canaries green in CI; blocked on pyogrio musl wheels            |
| Windows ARM64                     | #334  | **shipped**            | vcpkg build; vector stack vendored (shapely/pyogrio gap)         |
| Python 3.15+                      | #335  | pending upstream       | ships when CPython 3.15 + ecosystem land; one-line `build` bump |
| Free-threaded (`cp313t`/`cp314t`) | #683  | pending upstream       | GDAL SWIG bindings + numpy first; revisit at 3.15               |

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
.github/workflows/bundle-pypi-wheels.yml
│
├── build-sdist (1 job, ubuntu-latest)
│   └── python -m build --sdist → pyramids_gis-X.Y.Z.tar.gz
│
├── build-linux-wheels (2 jobs in matrix: x86_64 + aarch64, builds 4 wheels each)
│   ├── For arch == x86_64 → runs on ubuntu-latest
│   ├── For arch == aarch64 → runs on ubuntu-24.04-arm (native ARM runner; no QEMU)
│   ├── actions/cache restores .srcbuild-cache (gzip'd tar of the compiled
│   │   stack, keyed on config.sh + build-gdal-stack.sh + arch). SKIPPED on
│   │   release (workflow_run) builds — published wheels always cold-build
│   │   from the pinned sources.
│   └── cibuildwheel (manylinux_2_28 image):
│       ├── CIBW_BEFORE_ALL (once per job):
│       │   bash ci/source-build/build-gdal-stack.sh
│       │   → compiles the full stack from SHA256-pinned tarballs
│       │     (ci/source-build/config.sh) into /usr/local
│       │   → collects each dep's LICENSE/COPYING into
│       │     share/pyramids-bundled-licenses/ (gated for completeness)
│       │   → writes ${BUILD_PREFIX}/GDAL_VERSION, stages the CA bundle
│       │   → driver-presence gate (required OGR + raster drivers)
│       ├── For each of cp311, cp312, cp313, cp314:
│       │   ├── CIBW_BEFORE_BUILD (per Python version):
│       │   │   python ci/install-and-vendor-osgeo.py
│       │   │   → pip install GDAL==X.Y.Z against the built libgdal
│       │   │   → vendors osgeo/ + osgeo_utils/ → src/pyramids/_vendor/
│       │   │   → vendors GDAL_DATA + PROJ_DATA → src/pyramids/_data/
│       │   │   → vendors the collected licenses → src/pyramids/_licenses/
│       │   ├── CIBW_BUILD:
│       │   │   pip wheel . → pyramids_gis-X.Y.Z-cp3NN-cp3NN-linux_<arch>.whl
│       │   └── CIBW_REPAIR_WHEEL_COMMAND:
│       │       auditwheel repair --plat manylinux_2_28_<arch>
│       │       (bundles libgdal.so + transitive deps, patches RPATH)
│       └── upload-artifact: wheels-linux-<arch> (one artifact per arch)
│
├── build-musl-wheels (2 canary jobs: x86_64 + aarch64, musllinux_1_2 image)
│   └── same before-all/config.sh flow with musl deltas (apk prereqs,
│       no HAVE_PREAD64 for sqlite); repair --plat musllinux_1_2_<arch>;
│       artifacts named canary-musl-<arch> so the release job can NEVER
│       pick them up (see Publishing).
│
├── build-macos-wheels (2 jobs in matrix: arm64 + x86_64, both on macos-14)
│   ├── CIBW_BEFORE_ALL: ci/setup-gdal-from-pixi.sh (conda-extract)
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
├── build-windows-wheels (1 job, windows-2022, builds 4 wheels)
│   └── cibuildwheel:
│       ├── CIBW_BEFORE_ALL: powershell -File ci/setup-gdal-from-pixi.ps1
│       │   → installs pixi → extracts Library/bin DLLs → C:/gdal-prefix
│       │   → writes ${BuildPrefix}/GDAL_VERSION (parsed from
│       │     conda-meta/gdal-X.Y.Z-*.json — Windows conda-forge gdal
│       │     doesn't ship a usable gdal-config)
│       ├── For each of cp311, cp312, cp313, cp314: same vendor + build steps
│       └── CIBW_REPAIR_WHEEL_COMMAND uses
│           `delvewheel repair --analyze-existing`
│           → bundles _gdal.pyd's direct deps AND the GDAL plugin
│             DLLs' transitive deps (netcdf.dll, hdf5.dll, ...) into
│             pyramids_gis.libs/, patches PE import tables.
│       └── upload-artifact: wheels-windows-AMD64
│
├── verify-debian12 / verify-rocky9 (full hermetic suite on glibc 2.36 / 2.34
│   containers — distros the old 2_39 wheel could never install on) and
│   verify-alpine (full core suite for the musl canary — vector I/O via a
│   canary-built pyogrio musl wheel, since PyPI has none). All three
│   run with --security-opt seccomp=unconfined (the netCDF driver needs
│   userfaultfd for /vsizip reads — see docs/troubleshooting.md).
│
└── release (workflow_run only; needs EVERY build + test + verify job,
    canary verifies included — a red canary blocks the publish)
    → gathers {sdist,wheels-*} artifacts (canary-* names can't match),
      asserts the exact composition (1 sdist + 4 wheels x 5 platforms
      + 3 win_arm64, zero musllinux), attaches everything to the
      GitHub release, and publishes to PyPI.
```

After all build jobs finish, `test-wheels` runs a 16-cell matrix
(4 OSes × 4 Python versions) installing each wheel in a clean Python
env and running `pytest -m core`. The 4 OSes are
`ubuntu-latest` (x86_64), `ubuntu-24.04-arm` (aarch64), `macos-14`
(arm64), and `windows-2022` (AMD64). macOS x86_64 testing is skipped —
the wheel is cross-compiled on an arm64 host so we can't install it
on the same runner, and GitHub's macos-13 queue is unusable.

The matrix uses `os` as a real axis (`os: [ubuntu-latest,
ubuntu-24.04-arm, macos-14, windows-2022]`) and `include:` adds per-OS
properties (`arch`, `artifact`, `wheel-tag`). An earlier
`include`-only shape silently collapsed all combos to Windows-only
jobs, which masked real test failures.

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
4. Downloads the single matching wheel (~30 MB Linux, up to ~60 MB Windows)
5. Falls back to the sdist if no wheel matches (which then requires
   system GDAL to be installed for `pip install` to succeed)

Examples:

| User environment | What pip picks / what happens |
|------------------|-------------------------------|
| Ubuntu 22.04 (x86_64, glibc 2.35) + Python 3.12 | `cp312-cp312-manylinux_2_28_x86_64` wheel |
| Debian 12 (glibc 2.36) + Python 3.13 | `cp313-cp313-manylinux_2_28_x86_64` wheel |
| RHEL 9 / Amazon Linux 2023 (glibc 2.34) | `manylinux_2_28_x86_64` wheel for their Python |
| Graviton (aarch64) + Ubuntu 24.04 + Python 3.13 | `cp313-cp313-manylinux_2_28_aarch64` wheel |
| Raspberry Pi 5 + 64-bit Ubuntu 24.04 + Python 3.12 | `cp312-cp312-manylinux_2_28_aarch64` wheel |
| M2 Mac + Python 3.13 | `cp313-cp313-macosx_11_0_arm64` wheel |
| Intel Mac + Python 3.13 | `cp313-cp313-macosx_11_0_x86_64` wheel (cross-compiled) |
| Windows 11 (x64) + Python 3.11 | `cp311-cp311-win_amd64` wheel |
| RHEL 7 (glibc 2.17) | no wheel matches → sdist fails without system GDAL → use conda-forge |
| Alpine Linux (musl) | no published wheel yet (#333) → use conda-forge |
| Windows 11 ARM64 + Python 3.13 | `cp313-cp313-win_arm64` wheel (native; vendored vector stack) |
| Windows 11 ARM64 + Python 3.11 | no wheel (numpy/scipy ship no cp311 arm64) → use Python 3.12+ |
| Python 3.10 | excluded by `requires-python = ">= 3.11"` — upgrade, or pin `< 0.20` |

## CI timing

On GitHub-hosted runners (jobs parallel where possible):

| Job                                                    | Duration                        |
|--------------------------------------------------------|---------------------------------|
| `build-sdist`                                          | ~2 min                          |
| `build-linux-wheels` (cold: full stack compile)        | ~60 min (4 wheels)              |
| `build-linux-wheels` (warm: stack restored from cache) | ~15 min (4 wheels)              |
| `build-musl-wheels` (canaries, same cold/warm split)   | ~60 / ~15 min                   |
| `build-macos-wheels` (arm64, native)                   | ~6 min (4 wheels)               |
| `build-macos-wheels` (x86_64, cross-compiled)          | ~7 min (4 wheels)               |
| `build-windows-wheels`                                 | ~12 min (4 wheels)              |
| `build-winarm64-wheels` (cold: full vcpkg compile)     | ~75 min (3 wheels)              |
| `build-winarm64-wheels` (warm: vcpkg cache restored)   | ~9 min (3 wheels)               |
| `test-wheels` matrix (16 jobs)                         | ~3 min (parallel, after builds) |
| `verify-debian12` / `verify-rocky9` (full suite)       | ~8 min each                     |
| `verify-alpine` (full core suite, canary pyogrio)      | ~8 min                          |
| `verify-winarm64` (3 cells; full core suite on 3.12)   | ~1–6 min                        |

Release builds are always cold on Linux and Windows ARM64 (both cache
steps are skipped on `workflow_run` so published binaries never come
from a cache), so budget ~80 min wall-clock for a release; branch
iterations with a warm cache complete in ~20 min.

Each step has an explicit `timeout-minutes` cap plus pytest's own
`--timeout=60 --timeout-method=thread` so a hung test fails fast on the
runner that can least afford to babysit (Windows).

## Publishing

Publishing lives in `bundle-pypi-wheels.yml`'s own `release` job: it fires on
`workflow_run` after the `github-release` workflow (commitizen tag) completes,
gathers the `{sdist,wheels-*}` artifacts (the `canary-musl-*` artifacts cannot
match that pattern), asserts the exact release composition before uploading,
attaches everything to the GitHub release, and publishes to PyPI. On `push` /
`workflow_dispatch` runs the job is skipped — those runs only build + test.

`.github/workflows/pypi-release.yml` is an **emergency, sdist-only** manual
fallback for when the wheel pipeline is broken and conda-forge needs a new
tarball; do not use it for normal releases.

## Local builds

You can replicate any single OS's wheel build locally if you have
Docker (for Linux) or the host OS (for macOS / Windows):

```bash
# Linux x86_64 (Docker; can run from any host OS). Cold-compiles the
# whole native stack inside the container (~45-60 min first run;
# .srcbuild-cache/ makes reruns fast).
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64

# musl canary (same stack, Alpine image)
cibuildwheel --only cp312-musllinux_x86_64

# Linux aarch64 — runs natively on an ARM host (e.g. an M-series Mac,
# AWS Graviton dev box, or Raspberry Pi). On an x86 host, this would
# need QEMU emulation (~8–10× slower than native ARM).
cibuildwheel --only cp312-manylinux_aarch64

# macOS (must run on macOS)
cibuildwheel --only cp312-macosx_arm64
cibuildwheel --only cp312-macosx_x86_64   # cross-compile from arm64

# Windows (must run on Windows)
cibuildwheel --only cp312-win_amd64
```

## File map

| File                                               | Role                                                                                   |
|----------------------------------------------------|----------------------------------------------------------------------------------------|
| `.github/workflows/bundle-pypi-wheels.yml`         | The full pipeline (build + test + verify + publish)                                    |
| `.github/workflows/pypi-release.yml`               | Emergency sdist-only PyPI fallback                                                     |
| `ci/source-build/config.sh`                        | Linux: pinned dependency table (version + SHA256 + URL) + per-dep build recipes        |
| `ci/source-build/build-gdal-stack.sh`              | Linux before-all: prereqs, cache, license collection, driver/license gates             |
| `ci/vcpkg.json`                                    | win_arm64: pinned vcpkg manifest (curated gdal feature set, fixed builtin-baseline)    |
| `ci/setup-gdal-from-vcpkg.ps1`                     | win_arm64 before-all: baseline checkout, vcpkg install, Library/ staging, licenses     |
| `ci/setup-gdal-from-pixi.sh`                       | macOS native: pixi install, extract conda-forge binaries, toolchain shims              |
| `ci/setup-gdal-micromamba.sh`                      | macOS cross-compile: install micromamba and resolve target-platform env                |
| `ci/setup-gdal-from-pixi.ps1`                      | Windows: PowerShell version of the pixi setup                                          |
| `ci/install-and-vendor-osgeo.py`                   | Per-Python: build GDAL SWIG bindings + vendor osgeo + data + licenses                  |
| `ci/verify-wheel.py`                               | Post-install smoke test: drivers, JP2 round-trip, TLS /vsicurl                         |
| `ci/check-wheel-size.sh`                           | Enforces the `WHEEL_SIZE_BUDGET_MB` ceiling per built wheel                            |
| `pyproject.toml` `[tool.cibuildwheel.*]`           | cibuildwheel config per OS                                                             |
| `pyproject.toml` `[tool.pixi.feature.wheel-build]` | Pixi env with GDAL native deps (conda-extract model only)                              |
| `setup.py`                                         | `BinaryDistribution` override to force platform-specific wheel                         |
| `src/pyramids/__init__.py`                         | Runtime bootstrap: loads vendored osgeo + prepends `pyramids_gis.libs` to Windows PATH |
| `bundle-pypi-wheels.yml` `env:`                    | `PIXI_VERSION` / `MICROMAMBA_VERSION` pins consumed by `ci/setup-gdal-*`               |

## Toolchain version pinning

`PIXI_VERSION` (currently `0.68.1`) and `MICROMAMBA_VERSION` (currently
`2.6.1`) are pinned in the `env:` block of
`.github/workflows/bundle-pypi-wheels.yml`. The
`ci/setup-gdal-from-pixi.{sh,ps1}` and `ci/setup-gdal-micromamba.sh` scripts
read them and pass the version through to `pixi.sh/install.{sh,ps1}` /
`micro.mamba.pm`, so the installer always pulls the same binary — making
wheel builds reproducible across CI runs.

Only macOS and Windows consume these pins now: their cibuildwheel runs
`before-all` on the host and inherits the env directly. The Linux builds
compile from source inside the container and don't use pixi at all — their
"toolchain pin" is the set of `*_VERSION` + `*_SHA256` variables in
`ci/source-build/config.sh`.

For local development, install the same pixi version with:

```bash
PIXI_VERSION="0.68.1" curl -fsSL https://pixi.sh/install.sh | bash
```

To bump pixi: edit `PIXI_VERSION` in `bundle-pypi-wheels.yml`, push, let CI
re-build wheels with the new version. `pixi lock` produces different
lock-file headers across versions (0.63 emits `pypi-prerelease-mode: ...`,
0.68 prefers the v7 lock format), so coordinating the bump in one place
avoids "why does my lock diff have 35 k of churn?" surprises.

## Pitfalls worth remembering

These are surprises we hit while stabilizing the pipeline (preserved
so the next person doesn't have to rediscover them):

- **cibuildwheel's `environment` option uses REPLACE semantics**, not
  table-merge, across platform overrides. A top-level
  `[tool.cibuildwheel.environment]` is ignored as soon as
  `[tool.cibuildwheel.<platform>.environment]` exists for the same
  platform. Shared env vars must be duplicated per platform.
- **GDAL 3.13's netCDF driver needs Linux userfaultfd** to read `.nc` files
  through non-`/vsimem` VSI paths (`/vsizip/`, `/vsitar/`). Docker's default
  seccomp profile blocks the syscall, so those reads fail in containers unless
  run with `--security-opt seccomp=unconfined` (which the verify jobs do). Same
  behavior under both build models — it's a GDAL property, not a build defect.
- **musl (Alpine) removed the LFS64 aliases** (musl 1.2.4+): sqlite must NOT be
  built with `-DHAVE_PREAD64` there — config.sh applies those defines on glibc
  only. c-blosc's benchmark also fails to link on musl; its build turns
  `BUILD_BENCHMARKS`/`BUILD_TESTS`/`BUILD_FUZZERS` off.
- **actions/cache is a 10 GB pool per repo**: several generations of ~1 GB
  uncompressed stack tars evicted each other; the tars are gzipped and keyed on
  the build scripts' hash + GDAL version + libc flavor + arch.
- **Single-host upstreams time out**: tukaani.org (xz) went dark mid-burn-in;
  xz is fetched from its GitHub release mirror (SHA256-verified either way).
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
