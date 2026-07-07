# Tutorial: DatasetCollection basics

A `DatasetCollection` is a time-stacked set of co-registered rasters that share a spatial template
(rows, columns, cell size, CRS). This tutorial builds one from a folder of rasters and computes a
time-axis aggregation.

For the full picture of how the class works (the two backing paths, the cost model, and when each
applies), see the [DatasetCollection reference](../reference/dataset/collection.md).

## Build a collection from a folder

```python
from pyramids.dataset import DatasetCollection

# A folder containing several co-registered rasters (use repo test data or your own)
folder = "tests/data/geotiff/rhine"  # adjust as needed

# Order the timesteps by the numeric part of each file name
dc = DatasetCollection.read_multiple_files(
    folder, with_order=True, regex_string=r"\d+", date=False
)
print(dc)
```

All rasters in the folder must share the same shape and georeferencing. Use `date=True` (the
default) with a `fmt` string when the file names carry parseable dates (e.g. `MSWEP_1979.01.01.tif`).

## Reduce over the time axis

The reductions run lazily over a dask graph and return a NumPy array of shape `(bands, rows, cols)`:

```python
mean_arr = dc.mean(skipna=True)   # (bands, rows, cols)
print(mean_arr.shape)

total = dc.sum(skipna=True)
```

To reduce within groups (e.g. monthly means), pass per-timestep labels to `groupby`:

```python
months = [1, 1, 2, 2, 3]          # one label per timestep
monthly_mean = dc.groupby(months).mean(skipna=True)   # {label: ndarray}
```

## Next steps

- [Lazy collections](lazy/lazy-collection.md) — construction (`from_files`, `from_archive`,
  `from_stac`), reductions, `groupby`, and serialization (Zarr / NetCDF / kerchunk).
- [DatasetCollection notebook](../examples/dataset/dataset_collection.ipynb) — runnable, end-to-end.
- [DatasetCollection reference](../reference/dataset/collection.md) — full API.
