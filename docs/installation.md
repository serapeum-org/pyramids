# Installation

pyramids-gis ships **self-contained platform wheels** on PyPI that bundle
the GDAL/OGR/PROJ/GEOS native libraries. `pip install pyramids-gis`
works out of the box on Linux, macOS, and Windows — no system GDAL
installation required.

**Package name:** `pyramids-gis`
**Supported Python versions:** 3.11, 3.12, 3.13 (requires `>=3.11,<4`)

## Quick install (recommended for most users)

### With pip (PyPI platform wheels)

```console
pip install pyramids-gis
```

That's it. The wheel includes GDAL 3.13, PROJ, GEOS, HDF5, NetCDF, GRIB,
JPEG2000 (OpenJPEG, for JP2-packed GRIB2), libtiff, and all other native
dependencies (HDF4 additionally ships in the macOS/Windows wheels). No
`gdal-config`, no `apt install libgdal-dev`, no OSGeo4W installer needed.

### Optional extras

- `viz`: `cleopatra[tiles]` (plotting + basemap tiles via mercantile /
  xyzservices / Pillow)
- `lazy`: dask / distributed / zarr / fsspec / flox / kerchunk / h5py
  (powers `Dataset.read_array(chunks=…)`, `DatasetCollection.data`,
  `DatasetCollection.to_zarr`, and `NetCDF.to_kerchunk` /
  `combine_kerchunk` HDF5/NetCDF reference manifests)
- xarray interop (`DatasetCollection.to_netcdf`, `NetCDF.from_xarray` /
  `to_xarray`) is **not** a pyramids extra — pyramids is GDAL-backed, so
  xarray is a peer. `pip install xarray` directly when you want those helpers.
- `parquet`: pyarrow + dask-geopandas + `[lazy]` (eager GeoParquet I/O
  and the lazy `LazyFeatureCollection`)
- `dev`: nbval, pre-commit, pytest, coverage, build, twine, etc.
- `docs`: mkdocs, mkdocs-material, mkdocstrings, mike, etc.

```console
pip install "pyramids-gis[viz]"                  # plotting
pip install "pyramids-gis[lazy]"                 # dask-backed chunked I/O
pip install "pyramids-gis[viz,lazy]"             # combine extras with commas
pip install xarray                               # xarray / NetCDF4 interop (peer dep, not an extra)
```

### With conda-forge

```console
conda install -c conda-forge pyramids
```

conda-forge gets native GDAL via conda itself (not bundled in the
package). Use this if you're already in a conda/mamba environment.

### With pixi

```console
pixi add pyramids-gis
```

#### Linux: the glibc baseline (usually nothing to do)

pyramids-gis ships its Linux wheels tagged **`manylinux_2_28`** (GDAL and
its native stack are compiled from source with the manylinux toolchain).
pixi picks a wheel **at lock time** against a *declared* baseline, and
pixi's default Linux baseline is **glibc 2.28** (it tracks conda-forge's
floor). Because the wheel tag equals the default baseline, the wheel
resolves out of the box — **no `system-requirements` entry is needed**
(verified against pixi 0.65 defaults — newer versions, including the
0.68.1 this repo pins in CI, share the same baseline; rasterio's 2_28
wheels resolve the same way).

Two situations still need a declared baseline in the **consuming
project's** `pyproject.toml` (or `pixi.toml`):

- **You pin one of the older releases that shipped `manylinux_2_39`
  wheels (0.2x–0.39.x)** — those wheels are tagged
  `manylinux_2_39` and exceed the default baseline, so pixi silently
  falls back to the GDAL-less sdist (→ `ModuleNotFoundError: No module
  named 'osgeo'` at runtime). Declare:

  ```toml
  [tool.pixi.system-requirements]
  libc = "2.39"
  ```

- **You run an older pixi** whose default Linux baseline predates the
  2.28 floor — declare `libc = "2.28"` (satisfied by Ubuntu 20.04+,
  Debian 11+, RHEL 8+, Amazon Linux 2023).

It only affects Linux; macOS and Windows are unaffected. If you'd rather
avoid the topic entirely, install pyramids-gis from conda-forge instead
(`pixi add --channel conda-forge pyramids`), which gets native GDAL via
conda and has no manylinux tag to match.

## Verify the install

Open Python and run:

```python
import pyramids
from osgeo import gdal
print(pyramids.__version__)
print(gdal.__version__)          # should print 3.13.x (3.12.x on Windows ARM64)
```

## Platform support matrix

| Platform | Architecture | Wheel tag | Status |
|----------|-------------|-----------|--------|
| Linux (glibc ≥ 2.28) | x86_64 | `manylinux_2_28_x86_64` | ✅ Supported |
| Linux (glibc ≥ 2.28) | aarch64 | `manylinux_2_28_aarch64` | ✅ Supported |
| Linux (glibc < 2.28) | any | — | ❌ Fall back to conda |
| macOS 11+ | x86_64 | `macosx_11_0_x86_64` | ✅ Supported |
| macOS 11+ | arm64 (Apple Silicon) | `macosx_11_0_arm64` | ✅ Supported |
| Windows 10+ | x64 | `win_amd64` | ✅ Supported |
| Windows 11+ | ARM64 | `win_arm64` | ✅ Supported (Python 3.12+; see the ARM64 note below) |
| Alpine (musl) | any | — | ⏸ Built in CI, unpublished — blocked on upstream `pyogrio` musl wheels |

Distros covered by the Linux wheel out of the box:

- Ubuntu 20.04 LTS and newer
- Debian 11 (bullseye) and newer
- RHEL / Rocky / Alma Linux 8 and newer
- Amazon Linux 2023
- Fedora 38 and newer
- Arch Linux (rolling)

If your distro has **glibc < 2.28**, use the conda-forge path instead.

### Coverage gaps — what has no wheel yet, and why

Every remaining gap waits on something upstream of pyramids; none of
them can be closed from this repo alone. Progress is tracked in
[#333](https://github.com/serapeum-org/pyramids/issues/333) (Alpine) and
[#335](https://github.com/serapeum-org/pyramids/issues/335) (Python 3.15).
Until a row clears, the workaround column applies.

| Platform / target | Why no wheel today | Ships when | Workaround |
|---|---|---|---|
| Alpine / musl Linux | built + verified in CI; `pyogrio` has no musl wheels | pyogrio ships them (#333) | conda-forge |
| Python 3.15 | CPython unreleased; ecosystem needs cp315 wheels | after CPython 3.15, ~Oct 2026 (#335) | — |
| Free-threaded (`cp31Nt`) | GDAL SWIG bindings + numpy not ready | revisit at 3.15 (#683) | standard (GIL) build |
| Linux glibc < 2.28 | below the oldest maintained manylinux image | never (intentional) | conda-forge |
| PyPy / Python ≤ 3.10 | GDAL bindings target CPython 3.11+ | never (intentional) | CPython 3.11+ |

One **feature-level** difference inside covered platforms: the Linux
and Windows ARM64 wheels do not include the HDF4 driver (the macOS
wheels — both architectures — and the Windows x64 wheel do). HDF4 is a
legacy format; if you need it there, use conda-forge. This is the only
driver difference — the wheel pipeline asserts the full promised driver
set on every build, so drivers cannot silently disappear from a release.

> **Windows ARM64 note**: the `win_arm64` wheel ships Python 3.12–3.14
> (numpy/scipy publish no cp311 ARM64 wheels), carries GDAL 3.12.4 (the
> pinned vcpkg port version; the other platforms ship 3.13.1), has no
> HDF4 driver (like Linux), and **vendors its vector stack** — shapely,
> geopandas, and pyogrio ship inside the wheel under
> `pyramids/_vendor/` because upstream publishes no ARM64 wheels for
> them. `import geopandas` etc. work normally after `import pyramids`;
> the vendored copies disappear once upstream ships ARM64 wheels.
> On Python 3.11 there is no wheel and pip falls back to the sdist,
> which installs *successfully* but cannot provide GDAL or the vector
> stack (the dependency markers skip shapely/geopandas on Windows
> ARM64) — use Python 3.12+ or conda-forge instead. The same caveat
> applies to any deliberate sdist install on Windows ARM64.

## System dependencies

The wheel bundles nearly everything (expat is linked statically into the
bundled GDAL). The only system dependencies are standard C runtime
libraries that every Linux distro — including `python:*-slim` Docker
images — ships out of the box:

- `libc.so.6`, `libm.so.6`, `libpthread.so.0`, `libdl.so.2` (glibc)
- `libgcc_s.so.1`, `libstdc++.so.6` (GCC runtime)

No system packages need to be installed — `pip install pyramids-gis`
works on a bare `python:3.12-slim` image as-is.

## Editable / development install

For contributing to pyramids-gis, use pixi (which manages GDAL via
conda-forge for development):

```console
git clone https://github.com/Serapieum-of-alex/pyramids.git
cd pyramids
pixi install -e dev
pixi run -e dev pip install -e .
pixi run -e dev main      # runs the main test suite
```

Pixi environments available:

| Environment | Purpose |
|-------------|---------|
| `dev` | Default development env (includes viz + xarray + test tooling) |
| `docs` | Documentation toolchain (mkdocs + plugins) |
| `py311`, `py312`, `py313`, `py314` | Single-Python-version test envs |
| `wheel-build` | Minimal env used by cibuildwheel to obtain native GDAL |

## Install from source (no pixi)

If you're not using pixi and want to install from source, you'll need
the native GDAL library available at configure time (because the sdist
does **not** include it — only the PyPI wheel does).

```console
# 1. Install native GDAL via your system package manager first
# (see Platform-specific: no wheel available below)

# 2. Then install pyramids-gis from source
git clone https://github.com/Serapieum-of-alex/pyramids.git
cd pyramids
pip install .
```

## Platform-specific: no wheel available

If you're on a platform we don't ship a wheel for (e.g., musllinux/Alpine,
glibc < 2.28), pip will try to build pyramids-gis
from the sdist. That requires a pre-installed native GDAL:

### Linux (Debian/Ubuntu)

```console
sudo apt update
sudo apt install gdal-bin libgdal-dev
pip install pyramids-gis
```

### Linux (Fedora/RHEL/Rocky)

```console
sudo dnf install gdal gdal-devel
pip install pyramids-gis
```

### macOS (Homebrew)

```console
brew install gdal
pip install pyramids-gis
```

### Windows without a wheel

Use conda or pixi — installing GDAL natively on Windows is impractical.

## Install directly from GitHub

Latest `main`:

```console
pip install "git+https://github.com/Serapieum-of-alex/pyramids.git"
```

A specific tagged release:

```console
pip install "git+https://github.com/Serapieum-of-alex/pyramids.git@<version>"
```

Note: this installs from the sdist, not a wheel, so the same
pre-installed-native-GDAL caveat applies.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common install and
runtime issues.

## Bundled software and attribution

`pyramids-gis` is licensed under GPLv3. The platform wheels published on
PyPI bundle **GDAL** and its native dependencies — PROJ, GEOS, libtiff,
libgeotiff, NetCDF-C, HDF5, OpenJPEG, libcurl, OpenSSL, libpng,
libjpeg-turbo, libwebp, zlib / libdeflate / zstd / liblzma, and more
(macOS/Windows additionally bundle HDF4 and libxml2 via the conda-forge
build) — each under its own MIT, BSD, LGPL, or Apache license.

The full license text for every bundled library ships inside the wheel
under `pyramids/_licenses/<package>/` and stays bound to that wheel even
when it's redistributed. A human-readable index of what's bundled and
the SPDX identifier for each component lives in
[`THIRD_PARTY_LICENSES.md`](https://github.com/Serapieum-of-alex/pyramids/blob/main/docs/about/THIRD_PARTY_LICENSES.md)
under `docs/about/`.

If you redistribute a `pyramids-gis` platform wheel — directly or as
part of a larger product — the MIT / BSD / LGPL / Apache attribution
notices in `pyramids/_licenses/` must travel with it. The practical
way to satisfy this is to leave that directory untouched inside the
wheel.

The **sdist** does not bundle any third-party binaries (you build your
own GDAL out-of-band), so none of the above applies to sdist installs;
only platform wheels carry the bundled native libraries and the
corresponding attribution obligation.

If you use `pyramids-gis` in academic or publication contexts, please
also cite GDAL itself per
[gdal.org/cite_gdal.html](https://gdal.org/cite_gdal.html). Pyramids
stands on GDAL's shoulders — that project deserves the credit.

## Further reading

- Documentation: <https://serapeum-org.github.io/pyramids/latest>
- Source: <https://github.com/Serapieum-of-alex/pyramids>
- PyPI: <https://pypi.org/project/pyramids-gis/>
- conda-forge: <https://anaconda.org/conda-forge/pyramids>
- Third-party licenses: <https://github.com/Serapieum-of-alex/pyramids/blob/main/docs/about/THIRD_PARTY_LICENSES.md>
- GDAL upstream: <https://gdal.org>
