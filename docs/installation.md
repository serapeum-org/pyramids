# Installation

This page explains how to install pyramids-gis and its dependencies using Pixi/conda or pip. The instructions and versions below are aligned with the project’s pyproject.toml.

Package name: pyramids-gis
Supported Python versions: 3.11+ (requires Python >=3.11,<4)

## Dependencies

### Core runtime (PyPI)
- GDAL >=3.10.0,<4
- numpy >=2.0.0
- pandas >=2.0.0
- geopandas >=1.0.0
- Shapely >=2.0.0
- pyproj >=3.7.0
- PyYAML >=6.0.0
- hpc-utils >=0.1.5

### Native GDAL libraries (conda-forge only)
- libgdal-netcdf >=3.10,<4
- libgdal-hdf4 >=3.10,<4

!!! warning "GDAL requires the native C/C++ library"
    GDAL is listed as a PyPI dependency, but the GDAL Python bindings
    **require the native GDAL library to be installed on your system first**.
    If you install via `pip install pyramids-gis` without the native library,
    the installation will fail when pip tries to build the GDAL wheel.
    See the [Installing GDAL for pip users](#installing-gdal-for-pip-users)
    section below for platform-specific instructions.

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

Combine extras with commas, e.g.
`pip install 'pyramids-gis[viz,lazy,xarray]'`.

## Recommended: Pixi/Conda environment
This repository includes a Pixi configuration to create fully-solvable environments with the right GDAL build from conda-forge.

Prerequisites: Install Pixi (https://pixi.sh/) or have conda/mamba with the conda-forge channel available.

### Using Pixi
From the project root:

```console
pixi run main          # runs the main test suite to ensure the env is solvable
pixi shell             # enter the Pixi environment
```

Pixi environments provided:
- default: includes dev + viz extras
- docs: documentation toolchain
- py3XX: pinned Python versions (one per supported minor version)

To install the package in editable mode inside the Pixi environment:

```console
pip install -e .
```

### Using conda/mamba directly
Create and activate an environment (example with Python 3.12):

```console
mamba create -n pyramids -c conda-forge python=3.12 gdal libgdal-netcdf libgdal-hdf4
mamba activate pyramids
```

Then install the package from PyPI (release):

```console
pip install pyramids-gis
```

Optionally include extras (examples):

```console
pip install "pyramids-gis[viz]"        # installs cleopatra
pip install "pyramids-gis[dev]"        # developer tools
pip install "pyramids-gis[docs]"       # docs toolchain
```

## Installing with pip

GDAL is declared as a PyPI dependency. When you run `pip install pyramids-gis`,
pip will automatically try to install the GDAL Python bindings. However, the
GDAL Python package **compiles against the native GDAL C/C++ library**, which
must already be present on your system.

If the native library is missing, pip will fail with errors like
`gdal-config: command not found` or missing header files.

### Installing GDAL for pip users

Install the native GDAL library for your platform **before** running
`pip install pyramids-gis`.

#### Linux (Debian/Ubuntu)

```console
sudo apt update
sudo apt install gdal-bin libgdal-dev
```

Verify the installed version:

```console
gdal-config --version
```

Then install pyramids-gis (pip will build the GDAL Python bindings against
the system library):

```console
pip install pyramids-gis
```

!!! tip
    If `pip install` fails with a version mismatch, pin the GDAL Python
    package to match your system version:
    `pip install GDAL==$(gdal-config --version)`
    then `pip install pyramids-gis`.

#### Linux (Fedora/RHEL)

```console
sudo dnf install gdal gdal-devel
```

#### macOS (Homebrew)

```console
brew install gdal
```

After installation, verify with `gdal-config --version`, then:

```console
pip install pyramids-gis
```

#### macOS (MacPorts)

```console
sudo port install gdal
```

#### Windows

On Windows, installing GDAL natively is more involved. The recommended
approaches in order of preference:

1. **Use conda/pixi** (strongly recommended, see above sections).
2. **OSGeo4W installer**: Download from https://trac.osgeo.org/osgeo4w/ and
   install the GDAL package. Then run pip from within the OSGeo4W shell.
3. **Pre-built wheels**: Christoph Gohlke historically provided pre-built
   Windows wheels. Check if a compatible wheel is available for your Python
   version and install it manually before installing pyramids-gis.

#### Using conda for GDAL only (hybrid approach)

If you prefer pip for everything else but want a reliable GDAL installation,
you can install only GDAL from conda-forge:

```console
conda install -c conda-forge gdal libgdal-netcdf libgdal-hdf4
pip install pyramids-gis
```

This gives you conda-managed native libraries with pip-managed Python packages.

## Install from source
Clone the repository and install:

```console
git clone https://github.com/serapeum-org/pyramids.git
cd pyramids
python -m pip install .
```

Editable (development) install:

```console
git clone https://github.com/serapeum-org/pyramids.git
cd pyramids
pip install -e .[dev]
```

Install directly from GitHub (latest main):

```console
pip install "git+https://github.com/serapeum-org/pyramids.git"
```

Install a specific tagged release from GitHub:

```console
pip install "git+https://github.com/serapeum-org/pyramids.git@<version>"
```

## Quick check
After installation, open Python and run:

```python
import pyramids
print(pyramids.__version__)
```

You can also run the test suite via Pixi:
```console
pixi run -e dev pytest -m "not plot" -q
```

To run a specific test file:
```console
pixi run -e dev pytest tests/netcdf/test_dimensions.py
```

## Notes
- Supported Python versions are 3.11+.
- Prefer conda-forge for GDAL and related libraries.
- Documentation: https://serapeum-org.github.io/pyramids/latest
- Source repository: https://github.com/serapeum-org/pyramids
