# Dataset

`Dataset` is the core raster class. It wraps a GDAL raster (GeoTIFF, NetCDF, ASCII, COG, …) and
exposes a high-level, Pythonic API for reading, transforming, analysing, and writing geospatial
rasters. `NetCDF` and `DatasetCollection` build on top of it.

## The model

- A `Dataset` has one or more **bands** over a regular grid, georeferenced by a **geotransform**
  (origin + cell size) and a **CRS** (EPSG).
- Reads are **eager by default** (NumPy); pass `chunks=` to read lazily as a Dask array.
- Cells that don't hold data carry a **no-data** sentinel.

## A first look

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("tests/data/acc4000.tif")
ds.shape, ds.band_count, ds.epsg, ds.cell_size   # grid + georeferencing
arr = ds.read_array()                            # (bands, rows, cols) NumPy array
ds.to_file("copy.tif")                           # driver inferred from the extension
```

Useful attributes: `shape`, `rows`, `columns`, `band_count`, `band_names`, `epsg`, `crs`,
`cell_size`, `geotransform`, `bbox`, `bounds`, `no_data_value`, `dtype`, `meta_data`. The full,
always-current list is generated from the source in the [API reference](../reference/dataset/index.md).

## What you can do

The operations are demonstrated as short, runnable notebooks — start from the
[**"How do I…?" index**](../examples/index.md). In brief:

- **Read / write / convert** — [Dataset basics](../examples/dataset/dataset.ipynb),
  [format conversions](../examples/conversions/geotiff-netcdf.ipynb),
  [windowed reads](../examples/operations/windowed-reads.ipynb).
- **Transform** — [reproject / resample / align](../examples/operations/reproject-resample-align.ipynb),
  [crop & mask](../examples/operations/crop-mask.ipynb),
  [mosaic & merge](../examples/operations/mosaic-merge.ipynb),
  [no-data & gap filling](../examples/operations/nodata-gap-filling.ipynb).
- **Analyse** — [zonal statistics](../examples/operations/zonal-statistics.ipynb),
  [extract at points](../examples/operations/extract-at-points.ipynb),
  [terrain analysis](../examples/operations/terrain-analysis.ipynb),
  [raster algebra](../examples/operations/raster-algebra.ipynb).
- **Raster ↔ vector** — [rasterize ↔ vectorize](../examples/operations/rasterize-vectorize.ipynb).
- **Visualize** — [visualization gallery](../examples/operations/visualization.ipynb),
  [color tables](../examples/operations/color-tables.ipynb),
  [overviews](../examples/operations/overviews.ipynb).
- **Scale out** — [using pyramids with Dask](../examples/dask/overview.ipynb).

## Related classes

- [`DatasetCollection`](../examples/dask/collection.ipynb) — a time-stack of co-registered rasters.
- [`NetCDF`](../reference/netcdf/index.md) — multidimensional / CF rasters.
- [`FeatureCollection`](feature.md) — the vector counterpart.
