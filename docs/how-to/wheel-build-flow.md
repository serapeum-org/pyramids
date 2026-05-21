# Wheel Build Flow

How `build-wheels.yml` produces platform wheels and how pip clients pick
the right one.

## What gets built per release

`build-wheels.yml` produces **20 platform wheels + 1 sdist** per
release:

| Platform | Architecture                       | Python versions        | Wheels |
|----------|------------------------------------|------------------------|--------|
| Linux    | x86_64 (`manylinux_2_39`)          | 3.11, 3.12, 3.13, 3.14 | 4      |
| Linux    | aarch64 (`manylinux_2_39`)         | 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | arm64 (Apple Silicon, `macosx_11_0`)| 3.11, 3.12, 3.13, 3.14 | 4      |
| macOS    | x86_64 (Intel, cross-compiled)     | 3.11, 3.12, 3.13, 3.14 | 4      |
| Windows  | AMD64 (x64)                        | 3.11, 3.12, 3.13, 3.14 | 4      |
| (any)    | sdist                              | —                      | 1      |

**Total: 20 wheels + 1 sdist.**

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
pyramids_gis-0.20.0-cp311-cp311-manylinux_2_39_x86_64.whl     # Linux x86_64 3.11
pyramids_gis-0.20.0-cp312-cp312-manylinux_2_39_x86_64.whl     # Linux x86_64 3.12
pyramids_gis-0.20.0-cp313-cp313-manylinux_2_39_x86_64.whl     # Linux x86_64 3.13
pyramids_gis-0.20.0-cp314-cp314-manylinux_2_39_x86_64.whl     # Linux x86_64 3.14
pyramids_gis-0.20.0-cp311-cp311-manylinux_2_39_aarch64.whl    # Linux aarch64 3.11
pyramids_gis-0.20.0-cp312-cp312-manylinux_2_39_aarch64.whl    # Linux aarch64 3.12
pyramids_gis-0.20.0-cp313-cp313-manylinux_2_39_aarch64.whl    # Linux aarch64 3.13
pyramids_gis-0.20.0-cp314-cp314-manylinux_2_39_aarch64.whl    # Linux aarch64 3.14
pyramids_gis-0.20.0-cp311-cp311-macosx_11_0_arm64.whl         # macOS arm64 3.11
pyramids_gis-0.20.0-cp311-cp311-macosx_11_0_x86_64.whl        # macOS x86_64 3.11
...
pyramids_gis-0.20.0-cp314-cp314-win_amd64.whl                 # Windows 3.14
pyramids_gis-0.20.0.tar.gz                                    # sdist
```

The compatibility tag (e.g. `cp312-cp312-manylinux_2_39_x86_64`) tells
pip exactly which Python ABI + OS + architecture this wheel was built
for.

## Platform coverage status

Last reviewed: 2026-05-17 (after the M7 aarch64 + L6 py314 landings
and the M8 manylinux floor-lowering attempts).

### What the wheels cover today

Users on the rows below get a **native, ready-to-import wheel** from
PyPI — no compiler, no system GDAL, no conda required:

| OS / arch | Compatibility tag | Distros / versions that match |
|---|---|---|
| Linux x86_64, glibc ≥ 2.39 | `manylinux_2_39_x86_64` | Ubuntu 24.04+, Debian 13+, RHEL 10+, Fedora 39+ |
| Linux aarch64, glibc ≥ 2.39 | `manylinux_2_39_aarch64` | Graviton / Ampere / RPi 5 64-bit on Ubuntu 24.04+, RHEL 10+ |
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

### What the wheels DON'T cover

Users on these rows fall back to the **sdist** (which then needs system
GDAL ≥ 3.10 + a C/C++ compiler at install time — usually painful) or
the **conda-forge install path** (which is glibc-agnostic and just
works):

| OS / arch | Why no wheel | Recommended install path | Tracking |
|---|---|---|---|
| Linux x86_64, glibc 2.28–2.38 | `manylinux_2_39` needs glibc ≥ 2.39 | conda-forge | #332 (wontfix) / #338 |
| Linux aarch64, glibc 2.28–2.38 | same | conda-forge | #332 (wontfix) / #338 |
| Alpine / musl Linux | no `musllinux_*`; needs from-source GDAL | conda-forge | #333 |
| Windows on ARM64 | no `win_arm64`; conda-forge GDAL is x64-only on Windows | AMD64 wheel under x86 emulation | #334 |
| Free-threaded CPython (`cp313t`/`cp314t`) | skipped; no free-threaded conda-forge GDAL | use a GIL build | — |
| Python 3.10 or earlier | excluded by `requires-python = ">= 3.11"` | upgrade Python, or pin `< 0.20` | intentional |
| Python 3.15+ (future) | not yet released by CPython | conda-forge until wheels ship | #335 |
| PyPy | `skip = ["*pp*"]`; GDAL bindings target CPython | use CPython | intentional |

#### When each uncovered platform could get a wheel

With the conda-forge-**extraction** build (no from-source GDAL), a platform is addable only once conda-forge ships
GDAL for it. The blocker per target and when it is expected to clear:

| Target | Blocker (extraction model) | Unblocks |
|---|---|---|
| Linux glibc 2.28-2.38 | conda-forge GDAL = GCC-13 (`GLIBCXX_3.4.32`) | never; only as distros EOL (2027-2032) |
| Alpine / musl | conda-forge has no musl GDAL to extract | no committed timeline (conda-forge has no musl target) |
| Windows ARM64 | no conda-forge `win-arm64` GDAL yet | conda-forge win-arm64 migration reaches GDAL (~months-1yr) |
| Free-threaded (`cp314t`) | no free-threaded conda-forge GDAL build | gdal feedstock enables it (~1-2yr) |
| Python 3.15+ | CPython 3.15 unreleased; no `gdal` py3.15 | ~late 2026, then a one-line `build` bump |

**Clears by just waiting on upstream** (then a config bump): Python 3.15, Windows ARM64, eventually free-threaded.
**Stays open under extraction** (only the from-source build — #332, `wontfix` — would close it): Linux glibc < 2.39
(intrinsic to conda-forge's GCC-13) and musl (no conda-forge target).

Concretely, the glibc gap excludes a large slice of production Linux:

| Distro | glibc | Wheel matches today? |
|---|---|---|
| Ubuntu 22.04 LTS | 2.35 | ❌ — use conda-forge |
| Ubuntu 24.04 LTS | 2.39 | ✓ |
| Debian 12 (bookworm) | 2.36 | ❌ — use conda-forge |
| Debian 13 (trixie) | 2.39 | ✓ |
| RHEL / Rocky / Alma 9 | 2.34 | ❌ — use conda-forge |
| RHEL / Rocky / Alma 10 | 2.39 | ✓ |
| Fedora 38 | 2.37 | ❌ — use conda-forge |
| Amazon Linux 2023 | 2.34 | ❌ — use conda-forge |

### Why the glibc floor is 2.39 (and not 2.28)

conda-forge's GDAL is compiled with **GCC 13**, whose `libstdc++.so.6`
exports `GLIBCXX_3.4.32` (a symbol absent from the system libstdc++ on
every ❌ row above). `auditwheel` correctly refuses to tag the wheel
below `manylinux_2_39` because the SWIG `.so` extensions
(`_gdal.cpython-3XX-…-linux-gnu.so`) reference that symbol directly.

Lowering the floor was attempted and is now **`wontfix`** (see #332 for
the full investigation). The full "Step 2c" recipe — mutate auditwheel's
policy (`symbol_versions` + drop `libstdc++.so.6` / `libgcc_s.so.1` from
`lib_whitelist`) and **bundle** the GCC-13 C++ runtime — does build and
tag a `manylinux_2_28` wheel. But the bundled `libstdc++` **segfaults at
runtime**: a dual-`libstdc++` ODR collision (via `STB_GNU_UNIQUE`) where
`pyproj`/`shapely`'s native libs and our bundled C++ runtime can't
coexist in one process. (Targeting `manylinux2014`/2.17 fails even
earlier — numpy ships no `cp314` wheel below `manylinux_2_28`.)

The **only** technique that works is to **build GDAL + PROJ + GEOS from
source** with the manylinux toolchain (the rasterio/fiona model), so
libgdal links the baseline `libstdc++` and bundles none. That's a ~1-week
build-pipeline rewrite and is **decided against** — hence #332 is
`wontfix`. In the meantime, conda-forge covers every glibc-< 2.39 user
out-of-the-box.

### Coverage roadmap (not committed)

| Gap | Issue | Status | Notes |
|---|---|---|---|
| Lower glibc floor (< 2.39) | #332 | **wontfix** | bundled `libstdc++` segfaults; only from-source GDAL works |
| musllinux (Alpine) | #333 | Won't-do for now | needs from-source GDAL (~1 week); conda-forge covers Alpine |
| Windows ARM64 | #334 | Won't-do for now | blocked on conda-forge `gdal` win-arm64; <2% of Windows |
| Python 3.15+ | #335 | reactive | ships when CPython 3.15 + conda-forge `gdal` are out; one-line `build` bump |
| Free-threaded (`cp313t`/`cp314t`) | — | Won't-do for now | no free-threaded conda-forge GDAL; use a GIL build |

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
├── build-linux-wheels (2 jobs in matrix: x86_64 + aarch64, builds 4 wheels each)
│   ├── For arch == x86_64 → runs on ubuntu-latest
│   ├── For arch == aarch64 → runs on ubuntu-24.04-arm (native ARM runner; no QEMU)
│   └── cibuildwheel (shared logic across both arches):
│       ├── CIBW_BEFORE_ALL (once per job):
│       │   bash ci/setup-gdal-from-pixi.sh
│       │   → installs pixi → installs wheel-build env (linux-64 or
│       │     linux-aarch64 platform pin, matching the runner)
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
│       │   │   pip wheel . → pyramids_gis-X.Y.Z-cp3NN-cp3NN-linux_<arch>.whl
│       │   └── CIBW_REPAIR_WHEEL_COMMAND:
│       │       auditwheel repair → manylinux_2_39_<arch>.whl
│       │       (bundles libgdal.so + transitive deps, patches RPATH)
│       └── upload-artifact: wheels-linux-<arch> (one artifact per arch)
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
4. Downloads the single matching wheel (~58 MB)
5. Falls back to the sdist if no wheel matches (which then requires
   system GDAL to be installed for `pip install` to succeed)

Examples:

| User environment | What pip picks / what happens |
|------------------|-------------------------------|
| Ubuntu 24.04 (x86_64) + Python 3.12 | `cp312-cp312-manylinux_2_39_x86_64` wheel |
| Graviton (aarch64) + Ubuntu 24.04 + Python 3.13 | `cp313-cp313-manylinux_2_39_aarch64` wheel |
| Raspberry Pi 5 + 64-bit Ubuntu 24.04 + Python 3.12 | `cp312-cp312-manylinux_2_39_aarch64` wheel |
| M2 Mac + Python 3.13 | `cp313-cp313-macosx_11_0_arm64` wheel |
| Intel Mac + Python 3.13 | `cp313-cp313-macosx_11_0_x86_64` wheel (cross-compiled) |
| Windows 11 (x64) + Python 3.11 | `cp311-cp311-win_amd64` wheel |
| Ubuntu 22.04 (glibc 2.35) | no wheel matches → sdist fails without system GDAL → use conda-forge |
| RHEL 9 (glibc 2.34) | same as above → use conda-forge |
| Amazon Linux 2023 (glibc 2.34) | same as above → use conda-forge |
| Alpine Linux (musl) | no wheel matches → use conda-forge |
| Windows on ARM64 | no `win_arm64` wheel → run AMD64 wheel under x86 emulation |
| Python 3.10 | excluded by `requires-python = ">= 3.11"` — upgrade, or pin `< 0.20` |

## CI timing

On GitHub-hosted runners (jobs parallel where possible):

| Job | Duration |
|-----|----------|
| `build-sdist` | ~2 min |
| `build-linux-wheels` (x86_64, ubuntu-latest) | ~12 min (4 wheels) |
| `build-linux-wheels` (aarch64, ubuntu-24.04-arm, native ARM) | ~12 min (4 wheels) |
| `build-macos-wheels` (arm64, native) | ~6 min (4 wheels) |
| `build-macos-wheels` (x86_64, cross-compiled) | ~7 min (4 wheels) |
| `build-windows-wheels` | ~12 min (4 wheels) |
| `test-wheels` matrix (16 jobs) | ~3 min (parallel, after builds) |

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
# Linux x86_64 (Docker; can run from any host OS)
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64

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

| File | Role |
|------|------|
| `.github/workflows/build-wheels.yml` | The full pipeline (build + test) |
| `.github/workflows/pypi-release.yml` | PyPI publish (token-based twine) |
| `ci/setup-gdal-from-pixi.sh` | Linux + macOS native: pixi install, extract conda-forge binaries, toolchain shims |
| `ci/setup-gdal-micromamba.sh` | macOS cross-compile: install micromamba and resolve target-platform env |
| `ci/setup-gdal-from-pixi.ps1` | Windows: PowerShell version of the pixi setup |
| `ci/install-and-vendor-osgeo.py` | Per-Python: build GDAL SWIG bindings + vendor osgeo + data + DLL-bootstrap patch |
| `ci/check-wheel-size.sh` | Enforces the `WHEEL_SIZE_BUDGET_MB` ceiling per built wheel |
| `pyproject.toml` `[tool.cibuildwheel.*]` | cibuildwheel config per OS |
| `pyproject.toml` `[tool.pixi.feature.wheel-build]` | Minimal pixi env with GDAL native deps |
| `setup.py` | `BinaryDistribution` override to force platform-specific wheel |
| `src/pyramids/__init__.py` | Runtime bootstrap: loads vendored osgeo + prepends `pyramids_gis.libs` to Windows PATH |
| `build-wheels.yml` `env:` | `PIXI_VERSION` / `MICROMAMBA_VERSION` toolchain pins consumed by the `ci/setup-gdal-*` scripts |

## Toolchain version pinning

`PIXI_VERSION` (currently `0.68.1`) and `MICROMAMBA_VERSION` (currently
`2.6.1`) are pinned in the `env:` block of
`.github/workflows/build-wheels.yml`. The `ci/setup-gdal-from-pixi.{sh,ps1}`
and `ci/setup-gdal-micromamba.sh` scripts read them and pass the version
through to `pixi.sh/install.{sh,ps1}` / `micro.mamba.pm`, so the installer
always pulls the same binary — making wheel builds reproducible across CI
runs.

macOS and Windows cibuildwheel runs `before-all` on the host and inherits
these env vars directly. The Linux build runs `before-all` inside the
manylinux container, which doesn't auto-inherit host env, so they're
forwarded with `CIBW_ENVIRONMENT_PASS_LINUX` (also in the workflow `env:`
block).

For local development, install the same pixi version with:

```bash
PIXI_VERSION="0.68.1" curl -fsSL https://pixi.sh/install.sh | bash
```

To bump pixi: edit `PIXI_VERSION` in `build-wheels.yml`, push, let CI
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
