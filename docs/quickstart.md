# Quickstart — 10 minutes to pyramids

A fast, copy-pasteable tour of the core workflow. If you have not installed pyramids yet, see
[Installation](installation.md) (`pip install pyramids-gis` works out of the box — the wheel bundles GDAL).

Everything below runs offline on the sample data shipped in the repository's `examples/data/` folder, so you can
follow along verbatim. New to the vocabulary (CRS, geotransform, datacube, …)? Skim
[Core concepts & terminology](concepts.md) first.

## Rasters in 60 seconds

Read a GeoTIFF, inspect its georeferencing, pull the pixels into NumPy, then crop, reproject, and write it back.

```python
import numpy as np
from pyramids.dataset import Dataset

# 1. Open — lazily; pixels are read on demand
ds = Dataset.read_file("examples/data/acc4000.tif")

# 2. Inspect the georeferencing (no GDAL boilerplate)
ds.epsg            # -> 32618      (CRS as an EPSG code)
ds.cell_size       # -> 4000.0     (pixel size in CRS units)
ds.bbox            # -> (432968.1, 468007.8, 488968.1, 520007.8)
ds.shape           # -> (1, 13, 14)  (bands, rows, cols)
ds.band_count      # -> 1
ds.no_data_value   # -> (-3.4e+38,)  (per band)

# 3. Read the values as a NumPy array
arr = np.asarray(ds.read_array())        # (13, 14) float32

# 4. Reproject to WGS84 (EPSG:4326)
wgs84 = ds.to_crs(4326)

# 5. Crop to a bounding box (in the dataset's CRS)
inner = ds.crop(bbox=(440000, 475000, 480000, 515000), epsg=ds.epsg)

# 6. Write back out — the driver is inferred from the extension
wgs84.to_file("acc_wgs84.tif")           # GeoTIFF
ds.to_cog("acc.cog.tif")                 # Cloud-Optimized GeoTIFF

# 7. Plot (needs the [viz] extra — cleopatra)
ds.plot(band=0, title="flow accumulation")
```

!!! tip "One object, not a dangling C pointer"
    `Dataset` wraps a `gdal.Dataset` and exposes its georeferencing as plain Python attributes. You never touch
    `osgeo.gdal` directly, and there are no manual `FlushCache()` / `None`-the-handle rituals.

## Vectors

Vector data is a `FeatureCollection` — a thin, CRS-aware wrapper over a GeoPandas `GeoDataFrame`.

```python
from pyramids.feature import FeatureCollection

fc = FeatureCollection.read_file("examples/data/coello_polygons.geojson")
fc.epsg            # -> 32618
fc.column          # -> ['fid', 'column2', 'columns3', 'geometry']
fc.plot()          # geopandas/matplotlib by default; pass basemap=... for a tiled backdrop
```

Round-trip vectors and rasters:

```python
from pyramids.dataset import Dataset

raster = Dataset.from_features(fc, cell_size=200)  # rasterize the polygons onto a grid
polys  = raster.to_feature_collection()            # vectorize a raster back to polygons
```

## NetCDF / CF datacubes

`read_file` on a NetCDF returns a **container** (a `NetCDF` describing the whole file). Pin one variable to get a
`NetCDF` that behaves like a single raster.

```python
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("examples/data/netcdf/pyramids-netcdf-3d.nc")
nc.variable_names          # -> ['values']
values = nc.get_variable("values")   # a Variable — one raster
values.shape               # (time, rows, cols)
values.plot(variable="values")
values.to_file("values.tif")
```

See [Core concepts](concepts.md#container-vs-variable) for the container-vs-variable model, and the
[NetCDF tutorial](tutorials/netcdf-plotting.md) for CF / UGRID / packed data.

## Time-series datacubes

Stack a folder of aligned rasters into a `DatasetCollection`, then persist it as Zarr:

```python
from pyramids.dataset import DatasetCollection

cube = DatasetCollection.from_files(sorted_list_of_geotiffs)
cube.time_length           # number of timesteps
cube.to_zarr("cube.zarr")  # parallel, chunked write
```

## Common one-liners (cheat-sheet)

| I want to… | Call |
|------------|------|
| Open a raster / vector / NetCDF | `Dataset.read_file(p)` · `FeatureCollection.read_file(p)` · `NetCDF.read_file(p)` |
| Read pixels to NumPy | `ds.read_array(band=0)` |
| Reproject to another CRS | `ds.to_crs(4326)` |
| Crop to a bbox / polygon | `ds.crop(bbox=..., epsg=...)` |
| Align to a reference grid | `ds.align(reference_ds)` |
| Mosaic tiles / stack bands | `pyramids.dataset.merge.merge_rasters(...)` / `stack_bands(...)` |
| Write GeoTIFF / COG | `ds.to_file(p)` · `ds.to_cog(p)` |
| Zonal statistics | `ds.zonal_stats(feature_collection)` |
| Rasterize / vectorize | `Dataset.from_features(fc, ...)` · `ds.to_feature_collection()` |
| Read from S3 / GS / Azure | `Dataset.read_file("s3://bucket/key.tif")` |
| Lazy / Dask-backed read | `ds.read_array(chunks="auto")` |
| Read from a web service | `Dataset.from_wcs(...)` · `Dataset.from_wms(...)` · `FeatureCollection.from_wfs(...)` |

## Next steps

- **Learn the model** — [Core concepts & terminology](concepts.md): which class to use, and the GIS vocabulary.
- **By task** — the ["How do I…?" example index](examples/index.md): 49 runnable notebooks, grouped by goal.
- **Deeper tutorials** — [Dataset](tutorials/dataset.md), [FeatureCollection](tutorials/feature.md),
  [DatasetCollection](tutorials/datacube-basics.md), [COG](tutorials/cog.md), [STAC](tutorials/stac.md),
  [Lazy / Dask](tutorials/lazy/lazy-compute.md).
- **Look things up** — the [API Reference](reference/dataset/index.md).
- **Compare** — [pyramids vs rasterio / xarray / rioxarray](comparison-other-gis-packages.md).
