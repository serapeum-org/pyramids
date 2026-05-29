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

That's it. The wheel includes GDAL 3.12, PROJ, GEOS, HDF4/5, NetCDF,
libtiff, and all other native dependencies. No `gdal-config`, no
`apt install libgdal-dev`, no OSGeo4W installer needed.

### Optional extras

- `viz`: `cleopatra[tiles]` (plotting + basemap tiles via mercantile /
  xyzservices / Pillow)
- `lazy`: dask / distributed / zarr / fsspec / flox (powers
  `Dataset.read_array(chunks=…)`, `DatasetCollection.data`,
  `DatasetCollection.to_zarr`)
- `xarray`: xarray (required for `DatasetCollection.to_netcdf` and
  `NetCDF.from_xarray`)
- `netcdf-lazy`: `[lazy]` + kerchunk + h5py (HDF5/NetCDF chunked reads
  via kerchunk references)
- `parquet`: pyarrow (vector parquet I/O)
- `parquet-lazy`: `[lazy]` + `[parquet]` + dask-geopandas (lazy vector
  reads)
- `dev`: nbval, pre-commit, pytest, coverage, build, twine, etc.
- `docs`: mkdocs, mkdocs-material, mkdocstrings, mike, etc.

```console
pip install "pyramids-gis[viz]"                  # plotting
pip install "pyramids-gis[xarray]"               # xarray / NetCDF4 interop
pip install "pyramids-gis[lazy]"                 # dask-backed chunked I/O
pip install "pyramids-gis[viz,lazy,xarray]"      # combine extras with commas
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

#### Linux: raise the glibc baseline so the wheel resolves

pyramids-gis ships its Linux wheels tagged **`manylinux_2_39`** (the
bundled GDAL is built with conda-forge's GCC 13, which needs
`GLIBCXX_3.4.32`). pixi/uv pick a wheel **at lock time** against a
*declared* baseline, and pixi's default Linux baseline is **glibc 2.17**
(manylinux2014). Because `2.39 > 2.17`, no wheel matches, and pixi either
falls back to the GDAL-less sdist (→ `ModuleNotFoundError: No module
named 'osgeo'` at runtime) or — for a wheel-only release with no sdist —
fails outright with:

```text
because pyramids-gis==X.Y.Z has no wheels with a matching platform tag
(e.g., `manylinux_2_28_x86_64`) ... cannot be used
```

Tell pixi the target actually has glibc ≥ 2.39 by adding this to the
**consuming project's** `pyproject.toml` (or `pixi.toml`):

```toml
[tool.pixi.system-requirements]
libc = "2.39"
```

pixi now advertises glibc 2.39, the `manylinux_2_39` wheel matches, and
the bundled wheel is locked. This declares the env targets **Ubuntu
24.04+ / RHEL 10+** (older Linux won't resolve — use conda-forge there).
It only affects Linux; macOS and Windows are unaffected. This is a
*consumer-side* setting — it can't be exported by pyramids itself, so
each pixi project that depends on the bundled wheel on Linux sets it.

If you'd rather not pin a glibc floor, install pyramids-gis from
conda-forge instead (`pixi add --channel conda-forge pyramids`), which
gets native GDAL via conda and has no manylinux tag to match.

## Verify the install

Open Python and run:

```python
import pyramids
from osgeo import gdal
print(pyramids.__version__)
print(gdal.__version__)          # should print 3.12.x
```

## Platform support matrix

| Platform | Architecture | Wheel tag | Status |
|----------|-------------|-----------|--------|
| Linux (glibc ≥ 2.39) | x86_64 | `manylinux_2_39_x86_64` | ✅ Supported |
| Linux (glibc < 2.39) | x86_64 | — | ❌ Fall back to conda |
| Linux | aarch64 | — | 🔵 Planned |
| macOS 11+ | x86_64 | `macosx_11_0_x86_64` | ✅ Supported |
| macOS 11+ | arm64 (Apple Silicon) | `macosx_11_0_arm64` | ✅ Supported |
| Windows 10+ | x64 | `win_amd64` | ✅ Supported |
| Alpine (musl) | any | — | 🔵 Planned |

Distros covered by the Linux wheel out of the box:

- Ubuntu 24.04 LTS and newer
- Debian 13 (trixie) and newer
- RHEL / Rocky / Alma Linux 10 and newer
- Fedora 39 and newer
- Arch Linux (rolling)

If your distro has **glibc < 2.39**, use the conda-forge path instead.

## System dependencies

The wheel bundles nearly everything. The only system dependencies are
standard C runtime libraries that every Linux distro ships:

- `libc.so.6`, `libm.so.6`, `libpthread.so.0`, `libdl.so.2` (glibc)
- `libexpat.so.1` (XML parsing — on **minimal** Debian/Alpine images this
  may need `apt-get install libexpat1`; full distros have it)
- `libgcc_s.so.1`, `libstdc++.so.6` (GCC runtime)

On Docker `python:3.12-slim`:

```console
apt-get update && apt-get install -y libexpat1
pip install pyramids-gis
```

No other system packages are required.

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

If you're on a platform we don't ship a wheel for (e.g., Linux aarch64,
musllinux/Alpine, glibc < 2.39), pip will try to build pyramids-gis
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
libgeotiff, NetCDF-C, HDF5 / HDF4, libcurl, OpenSSL, libxml2, libpng,
libjpeg-turbo, libwebp, zlib / libdeflate / zstd / lz4 / bzip2 / liblzma,
and on Linux the GCC 13 libstdc++ — each under its own MIT, BSD, LGPL,
or Apache license.

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
