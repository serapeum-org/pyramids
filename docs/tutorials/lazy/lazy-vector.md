# Lazy vector reads — `LazyFeatureCollection`

`FeatureCollection.read_file(path, backend="dask")` and
`FeatureCollection.read_parquet(path, backend="dask")` return a
`LazyFeatureCollection` — a subclass of `dask_geopandas.GeoDataFrame`
that satisfies the `LazySpatialObject` protocol. Every partition-aware
operation inherited from dask-geopandas (`to_crs`, `clip`, `sjoin`,
`spatial_shuffle`) runs lazily; materialise with `.compute()` when you
need eager rows.

## Which reader should I use?

Pick by what you have and what you're trying to do. All four entry
points below start from the same user-facing
`pyramids.feature.FeatureCollection` class so you don't need to learn
a different API for each; the table just tells you which method to
call.

| Situation | Use |
|-----------|-----|
| Small / medium GeoJSON / Shapefile / GeoPackage, fits in RAM | `FeatureCollection.read_file(path)` (eager) |
| Large GeoJSON / Shapefile / GeoPackage you want to *partition* | `FeatureCollection.read_file(path, backend="dask", npartitions=…)` |
| GeoParquet with **pushdown filters** (`filters=[…]`, `columns=[…]`, `split_row_groups=…`) | `FeatureCollection.read_parquet(path, backend="dask", filters=…)` |
| GeoParquet, small enough to load eagerly | `FeatureCollection.read_parquet(path)` |
| Streaming batches of Arrow records for a **custom partitioner** (no pyramids wrapper) | `FeatureCollection.open_arrow(path, batch_size=…)` |

Rules of thumb:

1. **If you have GeoParquet**, prefer `read_parquet`. It pushes
   `filters` / `columns` / row-group splits down to pyarrow — true I/O
   savings. The `backend="dask"` variant parallelises row-group loads.
2. **If you have GeoJSON / Shapefile / GeoPackage**, `read_file` is
   the only option. The dask backend doesn't accept filter kwargs
   (dask-geopandas has no pushdown for those formats) — supplying
   `bbox=` / `mask=` / etc. raises `ValueError`. Filter after
   `.compute()` or pre-slice with `pyogrio` directly.
3. **If you're building your own dask partitioner** or streaming into
   arrow-consuming code (polars, duckdb), `open_arrow` returns a
   `pyarrow.RecordBatchReader` — no pyramids wrapping, no dask graph.

## ⚠ Before you benchmark: set `scheduler="processes"`

Shapely and GEOS hold the Python GIL. Under dask's **default threaded**
scheduler, `map_partitions` on geometry ops runs single-core. If your
benchmark shows the dask backend *slower* than pandas, that is almost
always why.

```python
import dask
dask.config.set(scheduler="processes")   # process-based, no GIL contention

# or spin up an explicit cluster:
from dask.distributed import LocalCluster, Client
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
```

Set one of these **before** issuing the first `.compute()` call. Without
it, the lazy API still works, but the speedup you expect won't appear.

## When to use `backend="dask"`

Rule of thumb:

- **> 500 MB GeoDataFrame**, or **> 10 million rows** — the partitioned
  read wins.
- Smaller than that, the eager path is faster because it avoids the
  partitioning cost.
- Anything that materialises rows (`.iloc`, `.loc`, `len`, `.plot`,
  `.to_file` for OGR drivers) forces a compute. Don't go lazy just to
  immediately materialise.

## Which ops stay lazy vs need `.compute()`

| Runs lazily on `LazyFeatureCollection`                                   | Requires `.compute()` first                   |
|--------------------------------------------------------------------------|-----------------------------------------------|
| `to_crs`, `clip`, `sjoin` (partition-pruned if `spatial_partitions` set) | `.iloc`, `.loc`, `len`                        |
| `spatial_shuffle`                                                        | `.plot` (no lazy plotting path)               |
| `.compute`, `.persist` (compute barriers)                                | `.to_file` (no lazy OGR write path)           |
| Any inherited `dask_geopandas.GeoDataFrame` method                       | pyramids-specific methods (`extract_vertices`, `rasterize_with_col`, `with_coordinates`, `with_centroid`, `center_points`) |

The rule: if dask-geopandas has a lazy implementation, so does
`LazyFeatureCollection`. Everything else — including every pyramids-specific
method that walks geometries one by one — raises `AttributeError` on a
lazy FC; call `.compute()` first.

## Subclass preservation — every return is a pyramids type

Every inherited `dask_geopandas.GeoDataFrame` method is intercepted so
the return value is always a pyramids type, never the backend class:

| Return class after...                                        | You get                  |
|--------------------------------------------------------------|--------------------------|
| `lfc.to_crs(4326)`, `.clip(...)`, `.sjoin(...)`              | `LazyFeatureCollection`  |
| `lfc.copy()`, `.drop_duplicates()`, `.repartition(...)`      | `LazyFeatureCollection`  |
| `lfc.spatial_shuffle(...)`, `.persist()`                     | `LazyFeatureCollection`  |
| `lfc.head(N)`, `.tail(N)`, `.compute()`                      | `FeatureCollection`      |

This means `compute_total_bounds()`, `epsg`, and `is_lazy_fc(x)` stay
available after a reproject or clip — you never have to re-wrap the
result. The eager twin after `.compute()` / `.head()` is a
`FeatureCollection`, so pyramids-specific eager methods
(`rasterize_with_col`, `extract_vertices`, etc.) are available on
that side too.

## `spatial_shuffle` → `sjoin` pruning workflow

The biggest speedup from going lazy comes from partition-pruned
`sjoin` — each partition has a bounding box, and dask drops partition
pairs that can't intersect before dispatching work. This requires the
bounding boxes to actually exist:

```python
from pyramids.feature import FeatureCollection

# 1. Load both frames lazily.
left = FeatureCollection.read_parquet("census_blocks.parquet", backend="dask")
right = FeatureCollection.read_parquet("roads.parquet", backend="dask")

# 2. Shuffle to populate spatial_partitions. This is a one-time cost
#    amortised across subsequent sjoins with the same frame.
left_shuffled = left.spatial_shuffle(by="hilbert")
right_shuffled = right.spatial_shuffle(by="hilbert")

# 3. Join. Dask drops partition pairs whose bboxes don't overlap.
joined = left_shuffled.sjoin(right_shuffled, how="inner", predicate="intersects")

# 4. Materialise.
result = joined.compute()   # a FeatureCollection, not a bare GeoDataFrame
```

`.spatial_shuffle` is partially eager — it computes Hilbert distances for
every row before rebuilding partitions. On 10 GB+ inputs, `.persist()` the
result so subsequent reads don't recompute the shuffle graph.

## `.persist()` for interactive workflows

`.compute()` materialises rows and returns an eager `FeatureCollection` —
you leave the lazy domain. `.persist()` materialises the task graph into
worker memory but keeps the lazy wrapper, so subsequent ops build on the
persisted partitions without recomputing:

```python
from pyramids.feature import FeatureCollection

lazy = FeatureCollection.read_parquet("large.parquet", backend="dask")
shuffled = lazy.spatial_shuffle().persist()   # still lazy, but graph is warm

# subsequent sjoin / clip calls benefit from the persisted shuffle:
hits = shuffled.sjoin(query_fc, predicate="intersects").compute()
```

## Writing a lazy FC back to disk

`LazyFeatureCollection.to_parquet(path)` is the only lazy-native write
path. It writes a partitioned directory of `part.N.parquet` files and
always blocks until every partition is materialised — `compute=False`
is rejected to keep the pyramids "`to_*` always writes" invariant. All
other kwargs (`compression=`, `write_index=`, `storage_options=`) pass
through to `dask_geopandas.GeoDataFrame.to_parquet`.

```python
lfc = FeatureCollection.read_parquet("in.parquet", backend="dask")
lfc.spatial_shuffle().to_parquet("out.parquet")  # partitioned directory
```

`to_file()` is NOT lazy — it raises `NotImplementedError` because
dask-geopandas has no lazy OGR write path. Call `.compute().to_file(path)`
to materialise first.

## `total_bounds` is lazy — and honest about it

`LazyFeatureCollection` inherits `total_bounds` from
`dask_geopandas.GeoDataFrame`. That attribute returns a **dask Scalar**,
not a numpy array of four floats. Indexing it returns another lazy
object:

```python
lfc = FeatureCollection.read_parquet("points.parquet", backend="dask")
lfc.total_bounds           # <dask.Scalar ...> — NOT a tuple
lfc.total_bounds.compute() # → array([xmin, ymin, xmax, ymax])
```

When you want concrete numbers, call `compute_total_bounds()` —
self-documenting one-liner that makes the `O(partitions)` reduction
explicit at the call site:

```python
xmin, ymin, xmax, ymax = lfc.compute_total_bounds()
```

`LazyFeatureCollection` **does not** expose the eager
`top_left_corner` property (it used to, by silently calling
`.compute()` inside the property — that leaked the laziness). For
generic code that needs to accept both eager and lazy FCs, type-check
against `SpatialObject | LazySpatialObject` from
`pyramids.base.protocols` and branch on
`pyramids.feature.is_lazy_fc(x)` before touching bounds.

## Minimal-install guard

On installs without the `[parquet-lazy]` extra (i.e. no `dask-geopandas`),
`from pyramids.feature import LazyFeatureCollection` raises an
`ImportError` with an actionable install hint — good UX for users who
just want to know why their code failed. Library authors writing
dispatch code that must run on both minimal and full installs wrap the
import in a `try/except`:

```python
try:
    from pyramids.feature import LazyFeatureCollection
except ImportError:
    LazyFeatureCollection = None

if LazyFeatureCollection is not None and isinstance(obj, LazyFeatureCollection):
    ...
```

Python 3's `hasattr` only catches `AttributeError`, so it does NOT
return `False` here — the `ImportError` propagates. `try/except` is the
correct guard. Typical user code doesn't need it: if you're working
with lazy frames at all, you already have the extra installed and the
direct import just works.

## Interop with `Dataset.zonal_stats`

The raster-side `Dataset.zonal_stats(fc)` currently expects an eager
`FeatureCollection`. Until a lazy path lands, call `.compute()` first:

```python
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

ds = Dataset.read_file("dem.tif")
polys = FeatureCollection.read_parquet("zones.parquet", backend="dask")

# must materialise for now:
stats = ds.zonal_stats(polys.compute(), stats=("mean", "sum"))
```

A lazy `zonal_stats` that accepts a `LazyFeatureCollection` is tracked as a
follow-up against DASK-25.

## Install

`LazyFeatureCollection` requires the `[parquet-lazy]` extra:

```bash
pip install 'pyramids-gis[parquet-lazy]'
```

That pulls `pyarrow`, `dask`, `dask-geopandas`, `zarr`, and `fsspec`.
On minimal installs, `LazyFeatureCollection` is `None` and the dask
branches of `read_file` / `read_parquet` raise `ImportError` with an
actionable install hint.
