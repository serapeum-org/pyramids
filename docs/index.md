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

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyramids-green.svg)](https://anaconda.org/conda-forge/pyramids) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Downloads](https://pepy.tech/badge/pyramids-gis)](https://pepy.tech/project/pyramids-gis) [![Downloads](https://pepy.tech/badge/pyramids-gis/month)](https://pepy.tech/project/pyramids-gis)  [![Downloads](https://pepy.tech/badge/pyramids-gis/week)](https://pepy.tech/project/pyramids-gis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/pyramids-gis?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![PyPI version](https://badge.fury.io/py/pyramids-gis.svg)](https://badge.fury.io/py/pyramids-gis) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

### conda-forge feedstock
[Conda-forge feedstock](https://github.com/conda-forge/pyramids-feedstock)


pyramids - GIS utility package
=====================================================================
**pyramids** is a GIS utility package using gdal, ....

pyramids

![1](./_images/package-work-flow/overall.png)

Main Features
-------------

- GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
  needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
  convert different forms of distributed data (rasters, NetCDF, shapefiles)
- Cloud Optimized GeoTIFF (COG) read, write, and validation, with
  transparent S3 / GCS / Azure / HTTPS support via GDAL's virtual
  filesystem. See the [COG tutorial](tutorials/cog.md).

## Installation

- Conda (conda-forge):
Installing `pyramids` from the `conda-forge` channel can be achieved by:

```bash
conda install -c conda-forge pyramids
```

It is possible to list all the versions of `pyramids` available on your platform with:

```bash
conda search pyramids --channel conda-forge
```

- pip (PyPI):

to install the last release, you can easily use pip

```bash
pip install pyramids-gis
```

- From source (latest):

to install the last development to time, you can install the library from GitHub

```bash
pip install git+https://github.com/serapeum-org/pyramids
```

Quick start
===========

## Minimal example: open a dataset and inspect metadata

```python
from pyramids.dataset import Dataset

# Use your own raster path (GeoTIFF/ASC/NetCDF supported); here we show a relative test file
path = "tests/data/geotiff/dem.tif"  # adjust path as needed

ds = Dataset.read_file(path)
print(ds.width, ds.height, ds.transform)
print(ds.meta)

# Access array data
arr = ds.read()
print(arr.shape, arr.dtype)

# Save a single band to a new GeoTIFF (writes alongside input by default)
out = "./dem_copy.tif"
ds.to_file(out)
print("Saved to", out)
```

## Next steps
- Explore the Tutorials for end-to-end workflows.
- See How it works for architecture and data flow.
- Browse the API Reference for details of classes and functions.

![Dataset diagram](./_images/pyramids-dataset.svg)
