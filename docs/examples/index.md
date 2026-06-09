# Examples — "How do I…?"

A task-oriented index of the runnable example notebooks. Find the operation you need, open the
notebook. Every notebook runs offline on the repository's test data.

## Read, write & convert

| I want to… | Notebook |
|------------|----------|
| Open a raster and read values | [Dataset basics](dataset/dataset.ipynb) |
| Convert GeoTIFF ↔ NetCDF (incl. bands ↔ variables) | [GeoTIFF ↔ NetCDF](conversions/geotiff-netcdf.ipynb) |
| Convert between raster formats (ASCII / COG / Zarr) | [Other raster formats](conversions/raster-formats.ipynb) |
| Convert between vector formats (SHP / GPKG / GeoParquet) | [Vector formats](conversions/vector-formats.ipynb) |
| Read only part of a raster (window / tile / bbox) | [Windowed & tiled reads](operations/windowed-reads.ipynb) |

## Transform & align

| I want to… | Notebook |
|------------|----------|
| Reproject, resample, or align to a reference grid | [Reproject, resample & align](operations/reproject-resample-align.ipynb) |
| Crop / mask to a polygon or bounding box | [Crop & mask](operations/crop-mask.ipynb) |
| Mosaic tiles or stack bands into one raster | [Mosaic & merge](operations/mosaic-merge.ipynb) |
| Handle no-data and fill gaps | [No-data & gap filling](operations/nodata-gap-filling.ipynb) |

## Analyse

| I want to… | Notebook |
|------------|----------|
| Compute zonal statistics per polygon | [Zonal statistics](operations/zonal-statistics.ipynb) |
| Sample / extract raster values at points | [Extract at points](operations/extract-at-points.ipynb) |
| Derive slope / aspect / hillshade from a DEM | [Terrain analysis](operations/terrain-analysis.ipynb) |
| Do raster algebra (`apply` / `map_blocks` / `overlay`) | [Raster algebra](operations/raster-algebra.ipynb) |

## Vector & raster–vector

| I want to… | Notebook |
|------------|----------|
| Rasterize a vector / vectorize a raster | [Rasterize ↔ vectorize](operations/rasterize-vectorize.ipynb) |
| Build & reshape vector geometry | [Vector geometry toolkit](operations/vector-geometry.ipynb) |

## Visualize

| I want to… | Notebook |
|------------|----------|
| Render a raster (image / plot / histogram) | [Visualization gallery](operations/visualization.ipynb) |
| Read & write colour tables / band colours | [Color tables & band colors](operations/color-tables.ipynb) |
| Build overviews (image pyramids) | [Overviews](operations/overviews.ipynb) |

## Scale out (lazy / Dask)

| I want to… | Notebook |
|------------|----------|
| Use pyramids with Dask — overview | [Using pyramids with Dask](dask/overview.ipynb) |
| Lazy raster / collection / vector / NetCDF | [Dataset](dask/dataset.ipynb) · [Collection](dask/collection.ipynb) · [Feature](dask/feature.ipynb) · [NetCDF](dask/netcdf.ipynb) |

## Cloud & catalogs

| I want to… | Notebook |
|------------|----------|
| Read Cloud Optimized GeoTIFFs | [COG basics](cog/cog-basics.ipynb) |
| Search & load a STAC catalog | [STAC (local)](stac/stac-local.ipynb) |
| Work with Zarr stores | [Zarr basics](zarr/zarr-basics.ipynb) |
