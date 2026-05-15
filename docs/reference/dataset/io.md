# I/O Operations

Array reading/writing, file serialization, tiling, and overview operations.

## Open a raster — paths, URLs, archives, and bytes

`Dataset.read_file(path)` accepts plain paths, `/vsi*` paths, and URL
schemes (`http(s)://`, `s3://`, `gs://`, `az://` / `abfs://`,
`file://`) — URLs are transparently rewritten to GDAL's virtual
filesystem so cloud objects open with HTTP range requests, no extra
boilerplate.

For archive members, pass `vsi="zip"` / `"tar"` / `"gzip"` / `"auto"`
plus the optional `file_i=` index. For the bytes-already-in-memory case
(HTTP response bodies, DB blobs, S3 `get_object` payloads), use
`Dataset.from_bytes(data)` (and `NetCDF.from_bytes` for NetCDFs) — the
bytes are written to a temporary `/vsimem/` path and cleaned up on
garbage collection. To merge every member of an archive into one
multi-band Dataset see `Dataset.from_archive`; for one-timestep-per-member
see `DatasetCollection.from_archive`.

```python
from pyramids.dataset import Dataset

# Local path / URL / s3:// / gs:// — same call
ds = Dataset.read_file("https://example.com/scene.tif")

# Specific member from a (remote) zip
ds = Dataset.read_file("scene.zip", vsi="zip", file_i=0)

# Bytes already in memory (no temp file)
ds = Dataset.from_bytes(downloaded_bytes, name="scene-A")
```

See the [Recipes](../../how-to/recipes.md#read-a-raster-held-in-memory-as-bytes)
page for the bytes / archive / cloud-HTTP-retry recipes.

## Windowed reads — `bbox=` / `epsg=`

`read_array(bbox=(W, S, E, N), epsg=…)` reads a geographic-bbox window
in one call. `epsg` defaults to the dataset's own CRS; a bbox in a
foreign CRS is reprojected by the existing pipeline. The legacy 4-int
pixel `window=[off_x, off_y, n_cols, n_rows]` form still works, and the
GeoDataFrame `window=` form remains accepted. `window=` and `bbox=` are
mutually exclusive.

## Lazy reads — `chunks=…`

`Dataset.read_array(chunks=…)` opts in to a lazy `dask.array.Array`
rather than the default eager `numpy.ndarray`. The same switch powers
every per-pixel op (`focal_*`, `slope`, `aspect`, `hillshade`,
`focal_apply`). `chunks=None` (the default) preserves the legacy
numpy path and does not import dask.

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("big.tif")
lazy = ds.read_array(chunks=(1, 1024, 1024))   # dask.array.Array
lazy.mean(axis=(1, 2)).compute()
```

See [Lazy rasters](../../tutorials/lazy/lazy-raster.md) for chunk-size rules,
locks, `Dataset.to_zarr` / `from_zarr`, and parallel Zarr writes.

Install: `pip install 'pyramids-gis[lazy]'`.

::: pyramids.dataset.engines.IO
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
