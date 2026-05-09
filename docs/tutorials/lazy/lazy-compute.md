# Lazy computation with Dask

Pyramids ships a Dask-backed path for every subsystem that routinely
handles data too large to load eagerly: rasters, NetCDFs, time-series
collections, and vector tables. This page is the entry point. It
explains **when to go lazy**, **what the common building blocks look
like**, and **how to wire a cluster**. The per-subsystem how-tos drill
into the specifics:

- [Lazy raster I/O](lazy-raster.md) — `Dataset.read_array(chunks=…)`,
  `Dataset.to_zarr`, `focal_*`, `slope`, `aspect`, `hillshade`,
  `zonal_stats`.
- [Lazy NetCDF](lazy-netcdf.md) — `NetCDF.read_array(chunks=…)`,
  `to_kerchunk`, `combine_kerchunk`, `open_mfdataset`, the xarray
  backend entry point.
- [Lazy DatasetCollection](lazy-collection.md) — time-series cubes,
  `from_stac`, `from_files`, `to_zarr`, `to_kerchunk`, `groupby`.
- [Lazy vector reads](lazy-vector.md) — `LazyFeatureCollection`,
  GeoParquet pushdown, `spatial_shuffle`, `sjoin`.

## When to go lazy

The rule of thumb is:

| Situation                                 | Go lazy? |
|-------------------------------------------|----------|
| Single file < 500 MB, single-pass read    | No       |
| Single file > 500 MB and you'll reduce    | Yes      |
| Many files (time-series, STAC, tiles)     | Yes      |
| Cloud-hosted data (`s3://`, `http(s)://`) | Yes      |
| Interactive notebook, bounded memory      | Yes      |
| You need process-parallel writes (Zarr)   | Yes      |

Lazy wins come from two places: **avoiding I/O** that a reduction
makes redundant, and **parallelising I/O** across workers. If neither
applies, the eager path is faster because it skips the graph-building
cost.

## The three building blocks

Every lazy entry point in pyramids produces one of these three
objects:

1. `dask.array.Array` — returned by `Dataset.read_array(chunks=…)`,
   `NetCDF.read_array(chunks=…)`, and `open_mfdataset(...)`. The
   standard N-D lazy-array type; supports numpy-style arithmetic,
   slicing, and reductions.
2. `LazyFeatureCollection` — a `dask_geopandas.GeoDataFrame` subclass
   returned by `FeatureCollection.read_file(..., backend="dask")` and
   `FeatureCollection.read_parquet(..., backend="dask")`. Partitioned
   over rows, shapely ops run per-partition.
3. `DatasetCollection` with file-backed graph — `from_files(...)` and
   `from_stac(...)` stack per-file delayed reads into a 4-D
   `(T, B, R, C)` dask array exposed as `collection.data`.

All three materialise via `.compute()` (return eager twin) or
`.persist()` (keep lazy, cache graph in worker memory).

## Installing the extras

Lazy support lives behind optional extras so the eager install stays
minimal:

| Extra            | Pulls                                            | Enables                                      |
|------------------|--------------------------------------------------|----------------------------------------------|
| `[lazy]`         | `dask`, `zarr`, `fsspec`, `flox`                 | Raster + NetCDF lazy reads, Zarr, groupby    |
| `[netcdf-lazy]`  | `[lazy]` + `kerchunk`, `h5py`                    | `to_kerchunk`, `combine_kerchunk` manifests  |
| `[parquet-lazy]` | `[lazy]` + `pyarrow`, `dask-geopandas`           | `LazyFeatureCollection`, GeoParquet          |
| `[xarray]`       | `xarray`                                         | `NetCDF.to_xarray()` / `.from_xarray()`      |

Install one or many:

```bash
pip install 'pyramids-gis[lazy,netcdf-lazy,parquet-lazy,stac,zonal]'
```

When an extra is missing, the corresponding entry point raises a
branded `ImportError` naming the extra — no silent `None` sentinels,
no confusing `AttributeError` deep in a call chain.

## `pyramids.configure` — GDAL options in one call

The single biggest speedup for cloud COG / NetCDF workloads comes
from setting five GDAL config options that default to values
unsuitable for range-request readers. odc-stac's Pangeo benchmark
measured an 18× speedup — 68 s → 3.75 s — from a single
`configure_rio(...)` call.

Pyramids ships the same preset:

```python
from pyramids import configure

applied = configure(cloud_defaults=True)
# {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
#  'GDAL_HTTP_MAX_RETRY': '10',
#  'GDAL_HTTP_RETRY_DELAY': '0.5',
#  'GDAL_HTTP_MULTIRANGE': 'YES',
#  'VSI_CACHE': 'TRUE'}
```

Pass credentials through the same call:

```python
configure(cloud_defaults=True, aws={"aws_unsigned": True})
# expands to AWS_NO_SIGN_REQUEST=YES

configure(cloud_defaults=True, azure={"storage_connection_string": "..."})
# expands to AZURE_STORAGE_CONNECTION_STRING=...
```

Override a single key:

```python
configure(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="3")
```

On a distributed cluster, pass the `client` so workers replay the
same config:

```python
from dask.distributed import Client, LocalCluster
from pyramids import configure

cluster = LocalCluster(n_workers=4)
client = Client(cluster)

configure(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
# every worker now has the same GDAL env at startup
```

`configure` registers a dask `WorkerPlugin` that re-applies the
config when any new worker joins — so autoscaling clusters stay
consistent without manual wiring.

## `pyramids.configure_lazy_vector` — vector-side defaults

The vector path has two knobs worth centralising:

1. The **dask scheduler**. Shapely holds the GIL, so the default
   `threads` scheduler serialises vector ops to one core. Flip it
   globally:

    ```python
    from pyramids import configure_lazy_vector

    configure_lazy_vector(scheduler="processes")
    ```

2. The **partition size heuristic**. When
   `FeatureCollection.read_file(backend="dask")` is called without
   `npartitions=` or `chunksize=`, pyramids picks a partition count
   that keeps each partition near the target byte budget (128 MiB
   default). Raise it for bigger workers:

    ```python
    configure_lazy_vector(target_bytes_per_partition=256 * 1024 * 1024)
    ```

The `client=` kwarg works the same way as `configure`: a worker
plugin replays the settings on every worker.

## Protocols and dispatch helpers

Library code that accepts both eager and lazy pyramids objects should
not reach for `isinstance(x, Dataset)` directly. Pyramids exposes
structural types — PEP 544 `runtime_checkable` Protocols — that let
you write generic utilities without importing the concrete classes:

```python
from pyramids.base.protocols import SpatialObject, LazySpatialObject

def describe(obj: SpatialObject | LazySpatialObject) -> int | None:
    return obj.epsg  # cheap on both branches
```

- `SpatialObject` — shared by eager `Dataset` and `FeatureCollection`.
  Exposes `epsg`, `total_bounds`, `top_left_corner`, `read_file`,
  `to_file`, `plot`.
- `LazySpatialObject` — shared by `LazyFeatureCollection` (and any
  future lazy-raster twin). Same read-only metadata surface minus
  `top_left_corner`, plus `npartitions`, `compute()`, `persist()`.

Three small helpers live alongside:

```python
from pyramids.base.protocols import ArrayLike, is_lazy, as_numpy
from pyramids.feature import is_lazy_fc, has_lazy_backend
```

- `ArrayLike` — type alias for `numpy.ndarray | dask.array.Array`.
  Use in function signatures that may return either.
- `is_lazy(arr)` — `True` when `arr` is a dask array.
- `as_numpy(arr)` — forces to numpy, no-op if already numpy. Useful
  at subsystem boundaries where the next step is pure numpy.
- `is_lazy_fc(obj)` — `True` for `LazyFeatureCollection`, `False`
  otherwise (including minimal installs without dask-geopandas).
- `has_lazy_backend()` — `True` when the `[parquet-lazy]` extra is
  available.

## Pickle & handle hygiene — `CachingFileManager`

Every lazy read in pyramids goes through one pattern copied verbatim
from xarray: the **`CachingFileManager`**. You won't interact with it
directly most of the time, but its semantics affect what works on a
distributed cluster and what doesn't:

- Per `(opener, path, access, kwargs)` key, at most one `gdal.Dataset`
  handle is kept in a process-local LRU cache.
- The manager pickles to its recipe, not its live handle — dask can
  serialise tasks that carry a manager to remote workers without
  leaking handles.
- On unpickle, the worker reconstructs with an empty cache and opens
  fresh on first `acquire()`.

Consequence: **`Dataset`, `NetCDF`, and `FeatureCollection` instances
are safe to ship to dask workers via `client.submit` or
`client.scatter`.** Concretely:

```python
from pyramids.dataset import Dataset
from dask.distributed import Client

client = Client()
ds = Dataset.read_file("s3://bucket/scene.tif")

# safe — the handle isn't shipped, only the open recipe
future = client.submit(lambda d: d.read_array().shape, ds)
future.result()  # (1, H, W)
```

The accompanying `SerializableLock` token-backed lock guards against
the GDAL block-cache race that used to force eager reads. You never
instantiate one manually — pyramids picks it up via `default_lock()`
when you pass `lock=None`.

## Choosing a scheduler

Pyramids does **not** pick a scheduler for you by default. The right
choice depends on what you're doing:

| Scheduler        | When to use                                               |
|------------------|-----------------------------------------------------------|
| `threads`        | Raster reads (GDAL releases the GIL during I/O + decode). |
| `processes`      | Vector ops (shapely holds the GIL).                       |
| `synchronous`    | Debugging (`pdb`-friendly, no task graph).                |
| `distributed`    | Multi-node, or when you want a dashboard / plugins.       |

Set globally for the session:

```python
import dask

dask.config.set(scheduler="processes")
```

Or spin up a `LocalCluster` and pass the `Client` to `configure` /
`configure_lazy_vector`:

```python
from dask.distributed import Client, LocalCluster
from pyramids import configure, configure_lazy_vector

cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)

configure(cloud_defaults=True, client=client)
configure_lazy_vector(scheduler="processes", client=client)
```

## A typical lazy pipeline

End-to-end, an analyst workflow tends to look like this:

```python
from dask.distributed import Client, LocalCluster

from pyramids import configure, configure_lazy_vector
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.feature import FeatureCollection

cluster = LocalCluster(n_workers=4)
client = Client(cluster)

configure(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
configure_lazy_vector(scheduler="processes", client=client)

# 1. Open raster lazily.
dem = Dataset.read_file("s3://elevation/dem.tif")
slope = dem.slope(chunks=(1024, 1024))   # dask.array.Array, lazy

# 2. Open a stack lazily.
cube = DatasetCollection.from_files(sorted_tifs)   # 4-D dask array
monthly_mean = cube.groupby(months).mean()         # dict[month, ndarray]

# 3. Open a vector file lazily.
zones = FeatureCollection.read_parquet(
    "s3://vectors/zones.parquet", backend="dask",
)
shuffled = zones.spatial_shuffle(by="hilbert").persist()

# 4. Reduce. Single compute materialises the whole DAG.
stats = dem.zonal_stats(shuffled.compute(), stats=("mean",))
```

The four stages each map onto a per-subsystem how-to. Continue to:

- [Lazy raster I/O](lazy-raster.md) for `Dataset` + per-pixel ops.
- [Lazy NetCDF](lazy-netcdf.md) for NetCDFs and kerchunk.
- [Lazy DatasetCollection](lazy-collection.md) for time-series cubes.
- [Lazy vector reads](lazy-vector.md) for `LazyFeatureCollection`.

## Worked examples

The bundled notebooks exercise the full stack against real public
datasets:

- `docs/examples/dataset/dask-lazy-datasets.ipynb` — Sentinel-2 COGs
  on AWS open-data.
- `docs/examples/feature/dask-lazy-features.ipynb` — Overture Maps
  GeoParquet on AWS.
- `docs/examples/netcdf/dask-lazy-netcdf.ipynb` — ERA5 reanalysis.
- `docs/examples/netcdf/ugrid/dask-lazy-ugrid.ipynb` — FVCOM NECOFS
  via OPeNDAP.

They do not run during `mkdocs build` (the cells hit public cloud
endpoints) but render statically on the docs site.

## When lazy is the wrong answer

A handful of patterns look lazy but aren't:

- `.iloc[0]`, `len(lazy_fc)`, `.plot()` — each materialises rows.
  Compute once, reuse the eager twin.
- `backend="dask"` on a 50 MB GeoJSON — the partitioning cost exceeds
  the read cost; use the default `backend="pandas"`.
- `focal_mean(chunks=...)` on a raster that fits in RAM — the map-
  overlap halo bookkeeping beats the parallel win.
- Calling `.compute()` inside a tight loop — build the graph once,
  compute once, not N times.

If you're unsure, benchmark both paths on a representative sample.
The docstrings for every lazy entry point name their optional extra
so the error message on a minimal install tells you exactly what to
install.
