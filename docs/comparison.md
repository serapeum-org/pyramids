---
title: Python raster libraries compared — pyramids, rasterio, rioxarray, GDAL and geopandas
description: An honest comparison of the Python geospatial libraries for reading, writing and processing rasters,
  vectors and datacubes, and where pyramids fits among them.
---

# Reading and writing rasters in Python: which library should you use?

Python's geospatial stack has several good libraries and they solve different problems, despite overlapping
descriptions. This page says what each is for and where `pyramids` fits — which, for plenty of workloads, is not
the right answer.

## The short answer

| You want to… | Use |
|---|---|
| Read and write raster windows, fast, close to the metal | [rasterio](https://pypi.org/project/rasterio/) |
| Put a CRS on labelled N-D arrays, in an xarray pipeline | [rioxarray](https://pypi.org/project/rioxarray/) + [xarray](https://pypi.org/project/xarray/) |
| Work with vector data as a DataFrame | [geopandas](https://pypi.org/project/geopandas/) |
| Reach a GDAL capability no wrapper exposes | [GDAL](https://pypi.org/project/GDAL/) Python bindings |
| Handle rasters, vectors, datacubes and meshes behind one API | **pyramids** |
| Do the common GIS operations without assembling the boilerplate | **pyramids** |

If your pipeline is already xarray-shaped, rioxarray fits it better than this library. If you need raw windowed
I/O throughput, rasterio is closer to GDAL and has less between you and it. Neither is worth replacing for the
sake of one convenience.

## What pyramids is

A high-level GIS utility layer built on GDAL/OGR: reading, writing, cropping, reprojecting, aligning,
mosaicking and rasterizing, with first-class support for Cloud Optimized GeoTIFFs, STAC catalogs, lazy/Dask
computation, and cloud object stores (S3, GCS, Azure, HTTPS).

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("dem.tif")
ds = ds.crop(mask).to_crs(4326).align(reference)
ds.to_file("dem_aligned.tif")
```

## Where it differs

**One API across four data models.** `Dataset` for rasters, `FeatureCollection` for vectors,
`DatasetCollection` for multi-temporal stacks, `NetCDF` for labelled cubes and `UgridDataset` for unstructured
meshes — all in one package with one set of conventions. Elsewhere that is rasterio plus geopandas plus xarray
plus something for the mesh, each with its own idea of what a CRS or a nodata value is. Unstructured-grid
support in particular has few alternatives in this part of the stack.

**Multi-temporal stacks are a first-class object.** `DatasetCollection` treats a time series of rasters as one
thing to crop, align and write, rather than a list you loop over.

**The common operations are one call.** Aligning two rasters onto a shared grid, mosaicking a directory,
rasterizing a vector against a reference — each is a method rather than an assembly of GDAL calls. That is the
whole proposition, and whether it is worth a dependency depends on how often you write that boilerplate.

**Optional features stay optional.** Plotting, Dask, Parquet and STAC are extras, so a bare install pulls no
plotting stack:

```bash
pip install pyramids-gis            # core
pip install "pyramids-gis[lazy]"    # + Dask, Zarr, fsspec, s3fs
pip install "pyramids-gis[viz]"     # + plotting
pip install "pyramids-gis[stac]"    # + STAC catalog access
```

## Where the others are stronger

**rasterio** is the reference for raster I/O in Python — mature, widely deployed, extensively documented, with a
large community and far more Stack Overflow answers behind it. Its windowed-read API gives you more control than
this library exposes. For most people writing raster code, rasterio is the correct default and pyramids is only
worth it if the higher-level operations earn their place.

**rioxarray** is the right answer inside an xarray pipeline. If your data is already `DataArray`s and you want
CRS-aware reprojection and clipping on them, adopting a different object model would be a step backwards.

**geopandas** is the standard for vector work and has a far richer analytical surface than
`FeatureCollection` — joins, overlays, dissolves and the whole pandas API. pyramids depends on geopandas rather
than competing with it; `FeatureCollection` exists to sit beside the raster classes, not to replace a
GeoDataFrame.

**The GDAL Python bindings** expose everything, because they are GDAL. Any wrapper, this one included, covers a
subset. When you need a driver option or an algorithm no wrapper surfaces, go straight to the bindings.

**xarray** remains the right model for labelled N-D data. `NetCDF` here is an xarray-compatible convenience over
GDAL's NetCDF driver, not a replacement for xarray's ecosystem.

## Honest limitations

- A smaller community than rasterio or geopandas: fewer answers when you get stuck, and a smaller pool of
  people who have hit your bug before.
- GPLv3, where rasterio and geopandas are BSD. That matters for some downstream projects.
- The convenience layer means less direct control than calling GDAL yourself.
- Published wheels bundle GDAL and its native dependencies, which makes installation simple but the wheels
  large. See [third-party licenses](about/THIRD_PARTY_LICENSES.md) for the bundled components; if you use
  pyramids in a publication, cite [GDAL](https://gdal.org/en/stable/faq.html#how-do-i-cite-gdal) as well.

## Installing

```bash
pip install pyramids-gis
conda install -c conda-forge pyramids
```

## Where to go next

- [Quickstart](quickstart.md) — the shortest path to a working example
- [Core concepts](concepts.md) — how Dataset, DatasetCollection and FeatureCollection relate
- [How do I…?](examples/index.md) — task-shaped recipes
