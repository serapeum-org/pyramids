# Lazy time-series cubes — `DatasetCollection`

`DatasetCollection` is pyramids' time-series primitive: a stack of
aligned rasters — one per timestep — that exposes a single 4-D
`(T, B, R, C)` `dask.array.Array` and a matching set of reductions.
It sits on top of the lazy `Dataset` path: each timestep is opened
on demand through `CachingFileManager`, workers never see a live
`gdal.Dataset` handle, and the whole collection pickles cleanly for
`dask.distributed`.

See [lazy-compute.md](lazy-compute.md) for schedulers and
`configure`. This page covers the collection-specific surface:
construction, the 4-D cube, reductions, and the two serialisation
formats (Zarr + kerchunk).

Installing the extras:

```bash
pip install 'pyramids-gis[lazy]'         # cube + reductions + Zarr
pip install 'pyramids-gis[netcdf-lazy]'  # + to_kerchunk
# from_stac needs no extra — it's duck-typed over any iterable of
# STAC-like items (pystac.Item, raw JSON dicts, etc.). Users who
# build items via pystac-client install pystac themselves.
```

## Constructing a collection

Three entry points:

| Entry point                                         | Source                            |
|-----------------------------------------------------|-----------------------------------|
| `DatasetCollection.from_files(files)`               | A list of local or cloud paths    |
| `DatasetCollection.from_stac(items, asset="B04")`   | STAC items (pystac / raw dicts)   |
| `DatasetCollection.read_multiple_files(folder)`     | A folder, sorted by date regex    |

`from_files` opens only the first file eagerly — enough to derive
the geobox and dtype — and references the rest by path only:

```python
from pathlib import Path
from pyramids.dataset import DatasetCollection

files = sorted(Path("/data/noah").glob("noah_*.tif"))
cube = DatasetCollection.from_files(files)

cube.time_length   # len(files)
cube.rows, cube.columns, cube.shape
```

No per-file opens happen until you access `cube.data`.

`from_stac` takes any iterable of STAC Items — `pystac.Item`
objects from a live pystac-client search, raw JSON dicts from an
HTTP response, or any other duck-typed object with `.assets` and
`.bbox` — and resolves the named asset's `href` per item. pyramids
does not import pystac; pull it yourself (directly or transitively
via `pystac-client`) when you need to construct Items:

```python
import pystac_client

from pyramids.dataset import DatasetCollection

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=(-1.0, 41.0, 0.0, 42.0),
    datetime="2022-06-01/2022-06-30",
    query={"eo:cloud_cover": {"lt": 20}},
)

cube = DatasetCollection.from_stac(
    search.items(),
    asset="red",             # the B04 band alias
    max_items=20,            # cap for quick-look
)
```

Optional kwargs:

- `bbox=(minx, miny, maxx, maxy)` — lon/lat filter; items whose
  `bbox` doesn't intersect are dropped before hrefs are resolved.
- `max_items=N` — cap the item count after bbox filtering.
- `patch_url=callable` — rewrite each href before pyramids opens it.
  Useful for signing Planetary Computer URLs or adding query
  parameters.

## The 4-D cube — `collection.data`

`collection.data` is a lazy `dask.array.Array` of shape
`(T, B, R, C)`:

```python
cube = DatasetCollection.from_files(files)

cube.data.shape      # (n_files, bands, rows, cols)
cube.data.chunks     # ((1, 1, ...), (bands,), (rows,), (cols,))
```

Each per-file read is scheduled as a `dask.delayed` task that opens
the file via `CachingFileManager` and reads its full array. Workers
therefore never serialise a `gdal.Dataset` — only the file path
crosses the pickle boundary, matching the pattern xarray / stackstac
/ odc-stac use for `dask.distributed` safety.

Access the whole graph the same way you'd access any dask array:

```python
# reductions along time
time_mean = cube.data.mean(axis=0).compute()     # (B, R, C)

# slice a time window
sub = cube.data[0:30]

# combine with arithmetic
anomaly = cube.data - cube.data.mean(axis=0, keepdims=True)
```

## Built-in reductions

A small set of common reductions ships on the collection directly so
you don't have to write `.data.nanmean(axis=0).compute()` every
time:

| Method                              | Action                                  |
|-------------------------------------|-----------------------------------------|
| `collection.mean(skipna=True)`      | Element-wise mean across time           |
| `collection.sum(skipna=True)`       | Element-wise sum                        |
| `collection.min(skipna=True)`       | Element-wise min                        |
| `collection.max(skipna=True)`       | Element-wise max                        |
| `collection.std(skipna=True)`       | Element-wise standard deviation         |
| `collection.var(skipna=True)`       | Element-wise variance                   |

Each returns a `numpy.ndarray` of shape `(B, R, C)` after a single
`.compute()`. `skipna=True` (the default) routes to the `nan*` dask
variant.

```python
cube = DatasetCollection.from_files(files)
mean_rain = cube.mean()     # numpy.ndarray, shape (bands, rows, cols)
```

## Grouped reductions — `collection.groupby(labels)`

`groupby` lets you compute per-group reductions — monthly means,
per-cluster averages, seasonal climatologies. Pass a label array the
same length as the collection:

```python
import numpy as np

cube = DatasetCollection.from_files(daily_files)
months = np.array([d.month for d in dates])          # one per file

grouped = cube.groupby(months)
monthly_mean = grouped.mean()       # dict {1: array, 2: array, ...}
monthly_std  = grouped.std()
```

Under the hood, pyramids routes through **flox** (single-pass
grouped reduction — installed with `[lazy]`). When flox isn't
available it falls back to a per-label loop with equivalent
semantics. Both paths return a dict mapping label → numpy array.

## Per-timestep metadata — `RasterMeta`

Every collection caches a picklable `RasterMeta` snapshot of the
template file's geobox + dtype + nodata. Available via
`collection.meta` — always reachable without reopening the template
dataset:

```python
cube = DatasetCollection.from_files(files)

cube.meta.epsg              # 4326
cube.meta.shape             # (bands, rows, cols)
cube.meta.cell_size         # 0.001
cube.meta.transform         # GeoTransform tuple
cube.meta.nodata            # per-band tuple
cube.meta.block_size        # per-band (bw, bh)
```

The snapshot is derived eagerly at construction so:

- Downstream lazy paths (per-file reads, Zarr writes, kerchunk
  manifests) read geo metadata without paying a GDAL-open cost per
  call.
- The collection pickles cleanly even if the template Dataset handle
  is closed or points at a `/vsimem/` file.

## Writing a cube to Zarr — `collection.to_zarr`

Zarr is the only output where pyramids can do truly parallel writes:
each dask chunk in `collection.data` lands in an independent Zarr
chunk file. A rioxarray-compatible attribute schema is written so
downstream `xr.open_zarr(store)` consumers can reconstruct the geobox
without pyramids:

```python
cube = DatasetCollection.from_files(files)
cube.to_zarr("cube.zarr", mode="w")
```

The root group carries `epsg`, `GeoTransform`, `crs_wkt`, `nodata`,
`band_names`, `dtype`, plus a `pyramids_zarr_version` marker and the
source file list.

Cloud writes via fsspec:

```python
cube.to_zarr(
    "s3://bucket/key.zarr",
    storage_options={"anon": False, "key": "...", "secret": "..."},
)
```

Pass `compute=False` to return a `dask.delayed.Delayed` for deferred
execution; useful when the store-write is one step of a larger
graph.

`collection.to_zarr` raises `RuntimeError` on a collection without a
`files` list (e.g. the legacy `create_cube(src, n)` path) — Zarr
writes need a source file per timestep. `ImportError` raised when
the `[lazy]` extra is missing.

## Kerchunk manifest — `collection.to_kerchunk`

For **NetCDF / HDF5**-backed collections, emit a single JSON manifest
that points at every timestep's source file. Downstream consumers
open the entire cube as a lazy Zarr-backed xarray with **zero data
rewrite**:

```python
cube = DatasetCollection.from_files(netcdf_files)
cube.to_kerchunk("cube.kerchunk.json", concat_dim="time")
```

Consume with fsspec + xarray:

```python
import fsspec, xarray as xr

mapper = fsspec.get_mapper("reference://", fo="cube.kerchunk.json")
ds = xr.open_zarr(mapper, consolidated=False)
```

GeoTIFF-backed collections raise `NotImplementedError` on
`to_kerchunk` — the GeoTIFF manifest path needs `kerchunk.tiff` +
`tifffile` which isn't yet wired. Use `to_zarr` for those today.

Kerchunk lives behind `[netcdf-lazy]`.

## When to use which output

| Goal                                             | Reach for         |
|--------------------------------------------------|-------------------|
| Parallel write, rewrite to compact storage       | `to_zarr`         |
| Zero-copy index over existing files              | `to_kerchunk`     |
| Reduce to a single array, no persistence         | `.mean()` / etc.  |
| Emit per-timestep processed artefacts            | `.data` + user loop |

## Interoperating with single-raster operations

The template `Dataset` is always accessible via `collection.base` —
use it for anything that operates on a single raster (plotting,
`zonal_stats` with a reduced slab, geobox inspection):

```python
cube = DatasetCollection.from_files(files)

# per-timestep zonal stats — loop over time axis
stats_per_step = []
for t in range(cube.time_length):
    slab = cube.data[t].compute()        # one timestep as numpy
    # ... build a Dataset around `slab` if needed ...

# or: one zonal pass per reduction result
mean_raster = cube.mean()     # numpy array (B, R, C)
```

## Worked example — Sentinel-2 surface reflectance cube

```python
import pystac_client

from pyramids import configure
from pyramids.dataset import DatasetCollection

configure(cloud_defaults=True, aws={"aws_unsigned": True})

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=(-1.0, 41.0, -0.7, 41.3),
    datetime="2023-06-01/2023-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

cube = DatasetCollection.from_stac(
    search.items(), asset="nir", max_items=30,
)

# Per-pixel summer mean NIR reflectance.
summer_mean = cube.mean()          # numpy (1, H, W)

# Write the cube for reuse.
cube.to_zarr("s2_nir_summer.zarr")
```

A full runnable notebook spanning search → cube → reduction →
write is at `docs/examples/dataset/dask-lazy-datasets.ipynb`.

## Minimal-install behaviour

- `DatasetCollection.from_files` works without the `[lazy]` extra
  — construction is metadata-only.
- `collection.data`, the six reductions, and `to_zarr` raise
  `ImportError` for the `[lazy]` extra when missing.
- `collection.to_kerchunk` raises `ImportError` for
  `[netcdf-lazy]`; also raises `NotImplementedError` on
  GeoTIFF-backed collections regardless of extras.
- `DatasetCollection.from_stac` needs no pyramids extra — the
  implementation is duck-typed and does not import pystac.
