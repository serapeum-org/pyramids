[![Documentations](https://img.shields.io/badge/Documentations-blue?logo=github&logoColor=white)](https://serapeum-org.github.io/pyramids/main/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyramids-gis.png)](https://img.shields.io/pypi/pyversions/pyramids-gis)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![GitHub last commit](https://img.shields.io/github/last-commit/serapeum-org/pyramids)
![GitHub Repo stars](https://img.shields.io/github/stars/serapeum-org/pyramids?style=social)
[![codecov](https://codecov.io/gh/serapeum-org/pyramids/graph/badge.svg?token=g0DV4dCa8N)](https://codecov.io/gh/serapeum-org/pyramids)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5e3aa4d0acc843d1a91caf33545ecf03)](https://www.codacy.com/gh/serapeum-org/pyramids/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=serapeum-org/pyramids&amp;utm_campaign=Badge_Grade)

![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/serapeum-org/pyramids/latest?include_prereleases&style=plastic)

[![pages-build-deployment](https://github.com/serapeum-org/pyramids/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/serapeum-org/pyramids/actions/workflows/pages/pages-build-deployment)

Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyramids-green.svg)](https://anaconda.org/conda-forge/pyramids) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Downloads](https://pepy.tech/badge/pyramids-gis)](https://pepy.tech/project/pyramids-gis) [![Downloads](https://pepy.tech/badge/pyramids-gis/month)](https://pepy.tech/project/pyramids-gis)  [![Downloads](https://pepy.tech/badge/pyramids-gis/week)](https://pepy.tech/project/pyramids-gis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/pyramids-gis?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![PyPI version](https://badge.fury.io/py/pyramids-gis.svg)](https://badge.fury.io/py/pyramids-gis) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) |

### conda-forge feedstock
[Conda-forge feedstock](https://github.com/conda-forge/pyramids-feedstock)


pyramids - GIS utility package
=====================================================================
**pyramids** is a GIS utility package built on top of GDAL/OGR for working with raster data (GeoTIFF, NetCDF),
vector data (shapefiles, GeoJSON), and multi-temporal datacubes.

`pyramids-gis` is licensed under GPLv3 (see [LICENSE.md](LICENSE.md)). The
platform wheels published on PyPI bundle [GDAL](https://gdal.org/) and its
native dependencies (PROJ, GEOS, libtiff, NetCDF-C, HDF5, libcurl, …) — each
under its own MIT, BSD, LGPL, or Apache license. The full attribution list and
shipped license texts are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md);
if you use `pyramids-gis` in publications please also cite GDAL itself per
[gdal.org/cite_gdal.html](https://gdal.org/cite_gdal.html).

```mermaid
graph LR
    GeoTIFF & NetCDF & Shapefile & UGRID -->|read| pyramids
    subgraph pyramids
        direction TB
        Dataset
        NetCDF_class[NetCDF]
        UgridDataset
        DatasetCollection
        FeatureCollection
        subgraph Engines["Dataset engines (ds.io · ds.spatial · ds.bands · ds.analysis · ds.cell · ds.vectorize · ds.cog)"]
        end
        subgraph Plotting["Plotting layer — _plot_helpers.render_array · mesh_render · NetCDFPlot · Selectors / ColourOpts / FacetSpec · basemap"]
        end
    end
    Dataset -->|crop · reproject · align| Dataset
    Dataset --- Engines
    FeatureCollection -->|rasterize| Dataset
    UgridDataset -->|interpolate| Dataset
    Dataset -->|vectorize| FeatureCollection
    DatasetCollection -->|lazy temporal stack| Dataset
    NetCDF_class -->|extends| Dataset
    Dataset & NetCDF_class & DatasetCollection & UgridDataset -->|plot| Plotting
    Plotting -->|delegates| cleopatra(["cleopatra<br/>ArrayGlyph · MeshGlyph · tiles"])
```

For detailed architecture diagrams, see
[docs/overview/architecture.md](docs/overview/architecture.md).

Main Features
-------------

- **Dataset** - Read, write, crop, reproject, and align single-band and multi-band rasters (GeoTIFF)
  with full no-data handling and coordinate reference system support. Public API is organized into
  seven engine collaborators (`ds.io`, `ds.spatial`, `ds.bands`, `ds.analysis`, `ds.cell`,
  `ds.vectorize`, `ds.cog`); same-named facade methods on the Dataset itself keep the
  short form working — `ds.crop(mask)` and `ds.spatial.crop(mask)` are equivalent.
- **NetCDF** - Extends Dataset for NetCDF files with time/variable dimensions and CF conventions metadata.
  Optional xarray interoperability. `NetCDF.plot` exposes an xarray-aligned plotting API
  (`variable=` + grouped `Selectors` / `ColourOpts` / `FacetSpec` dataclasses, curvilinear `coords=`,
  `kind=`, `animate=`, lazy `chunks=`).
- **UgridDataset** - Read and visualize UGRID-1.0 unstructured meshes (triangles, quads, mixed).
  Supports mesh-to-raster interpolation and mesh-to-vector export.
- **DatasetCollection** - Manage time-series of co-registered rasters as a lazy temporal stack
  (per-timestep gdal handles open on demand; the full cube is never materialised in RAM)
  with optional dask-backed reductions and groupby.
- **FeatureCollection** - Work with vector data (shapefiles, GeoJSON) through a unified GeoDataFrame and
  OGR DataSource interface, including rasterization and geometry operations.
- **Plotting** - `Dataset` / `NetCDF` / `DatasetCollection` / `UgridDataset` all expose a `plot` method
  backed by [cleopatra](https://github.com/serapeum-org/cleopatra) (the `[viz]` extra), routed through
  a shared `pyramids.dataset._plot_helpers` core. Optional web-tile basemap underlays via
  `pyramids.basemap.add_basemap` (a thin wrapper over `cleopatra.tiles.add_tiles`).
- **Cloud-Optimized GeoTIFF (COG)** - First-class read/write/validate support via `ds.to_cog`,
  `ds.is_cog`, and `ds.validate_cog`.
- **Spatial operations** - Align rasters to a reference grid, reproject between coordinate systems,
  crop to vector boundaries, and convert between raster, NetCDF, and vector formats.

Installing pyramids
===============

Installing `pyramids` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge pyramids
```

It is possible to list all the versions of `pyramids` available on your platform with:

```
conda search pyramids --channel conda-forge
```

### Install from GitHub (development)

To install the latest development version, you can install the library from GitHub:

```
pip install git+https://github.com/serapeum-org/pyramids
```
Note: installing from GitHub uses the sdist and requires a pre-installed
system GDAL. See the full [installation guide](docs/installation.md)
and [troubleshooting](docs/troubleshooting.md) for details.

## pip

To install the latest release from PyPI:

```
pip install pyramids-gis
```


## Optional extras

```
pip install pyramids-gis[viz]      # cleopatra plotting support
pip install pyramids-gis[xarray]   # xarray/NetCDF4 interoperability
```

Quick start
===========

```python
from pyramids.dataset import Dataset

# Open a raster file
src = Dataset.read_file("path/to/raster.tif")
print(src.epsg)        # coordinate reference system EPSG code
print(src.cell_size)   # pixel resolution
print(src.shape)       # (bands, rows, columns)

# Read the raster data as a NumPy array
arr = src.read_array()                  # all bands
band0 = src.read_array(band=0)          # one band

# Spatial ops route through the spatial engine; the facade stays short
reprojected = src.to_crs(to_epsg=3857)  # same as src.spatial.to_crs(...)
```

```python
from pyramids.netcdf import NetCDF
from pyramids import Selectors, ColourOpts, FacetSpec   # grouped plot options

# Open a NetCDF file
nc = NetCDF.read_file("path/to/data.nc")
print(nc.variables)

# xarray-aligned plotting (needs the [viz] extra)
nc.plot("t2m", selectors=Selectors(time="2020-07-01", level=850),
        colour=ColourOpts(cmap="coolwarm", robust=True))
nc.plot("t2m", facet=FacetSpec(col="time", col_wrap=4))   # small multiples
nc.plot("t2m", animate="time", chunks={"time": 1})        # lazy per-frame animation
```

```python
from pyramids.feature import FeatureCollection

# Open a vector file
vector = FeatureCollection.read_file("path/to/shapefile.shp")
print(vector.epsg)            # CRS EPSG code
print(vector.total_bounds)    # (minx, miny, maxx, maxy)
```

```python
from pyramids.dataset import DatasetCollection

# Build a lazy stack of co-registered rasters (no pixels read yet)
cube = DatasetCollection.from_files(["a.tif", "b.tif", "c.tif"])
print(cube.time_length, cube.shape)

# Reductions over the time axis use dask under the hood
mean = cube.mean()                       # nan-aware by default
```

Testing
=======

This project uses [pixi](https://pixi.sh) as the environment and task manager.

```console
# Install dependencies and create dev environment
pixi install -e dev

# Run all tests (excluding plot tests)
pixi run -e dev main

# Run plot tests only
pixi run -e dev plot

# Run a specific test file
pixi run -e dev pytest tests/netcdf/test_dimensions.py -v

# Run a single test by node id
pixi run -e dev pytest tests/netcdf/test_dimensions.py::TestStripBraces::test_with_braces -q
```

Docker
======

A Dockerfile is provided to run pyramids-gis in a controlled environment with the correct GDAL stack
preinstalled via conda-forge. The image uses a multi-stage pixi build for a minimal production container.

Build the image:

```
docker build -t pyramids-gis:latest .
```

Run the container (mount your current folder as /workspace):

```
docker run --rm -it -v ${PWD}:/workspace pyramids-gis:latest bash
```

Inside the container you can verify the package is installed:

```
python -c "import pyramids; print('pyramids', pyramids.__version__)"
```
