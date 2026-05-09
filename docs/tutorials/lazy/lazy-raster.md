# Lazy rasters — `Dataset`

`Dataset` can behave as either an eager or lazy data source depending
on whether you ask `read_array` for a numpy array or a
`dask.array.Array`. Every per-pixel op — `focal_mean`, `focal_std`,
`slope`, `aspect`, `hillshade`, user-supplied kernels via
`focal_apply` — accepts the same `chunks=` switch and falls through
to `dask.array.map_overlap` when it is set. Zonal stats and COG
writes share the same pattern: zero behaviour change when you don't
opt in, fully parallel when you do.

See [lazy-compute.md](lazy-compute.md) for the common building blocks
(`configure`, schedulers, protocols).

Installing the extra:

```bash
pip install 'pyramids-gis[lazy]'
```

## Reading a lazy array — `Dataset.read_array(chunks=…)`

`Dataset.read_array` is a single entry point for both paths. The
default `chunks=None` preserves the legacy numpy behaviour exactly;
any other value opts in to a lazy dask array:

| `chunks=` value         | Resulting chunk shape                         |
|-------------------------|-----------------------------------------------|
| `None` (default)        | Eager `numpy.ndarray` — zero dask import cost |
| `"auto"`                | dask picks shapes near its byte target        |
| `-1`                    | Single chunk covering the full array          |
| `int`, e.g. `512`       | Same size on every dimension                  |
| `tuple`, e.g. `(1,512,512)` | Per-dimension sizes                       |
| `dict`, e.g. `{0:1,1:512,2:512}` | Per-dimension-index sizes            |

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("tests/data/geotiff/noah-precipitation-1979-corrected.tif")

eager = ds.read_array()                 # numpy.ndarray
lazy = ds.read_array(chunks=(512, 512)) # dask.array.Array

lazy.shape        # (H, W)
lazy.chunks       # ((512, ...), (512, ...))
lazy.compute()    # force materialisation → numpy array
```

When `chunks` is truthy and `dask` isn't installed, the call raises
an `ImportError` that names the `[lazy]` extra — no silent fall-back
to eager.

### Picking a chunk size

Rules of thumb:

- **Match the on-disk tile size** when possible — GDAL advertises
  `GetBlockSize()` and the default `"auto"` respects it. A 256×256
  cloud-optimised GeoTIFF reads best at 256×256 chunks; re-chunking
  downstream wastes the first read.
- **Aim for 50–200 MiB per chunk** on dense float32 data. Too small
  (< 5 MiB) and the graph overhead dominates; too large (> 500 MiB)
  and memory pressure wipes the parallelism win.
- **Leading band axis as `1`** — every band-heavy op reduces band-
  by-band, so `(1, 512, 512)` reads one band per chunk and keeps
  downstream `.sum(axis=0)` graphs small.

### Locks and the `CachingFileManager`

Each lazy chunk read opens the `Dataset` through a process-scoped
`CachingFileManager`. Concurrent reads on the same handle are guarded
by a `SerializableLock` in single-process contexts and a
`dask.distributed.Lock` when a client is running. You don't need to
wire this yourself — `lock=None` does the right thing.

Opt out when you know the backend is safe lock-free (per-thread
handle, independent HTTP ranges):

```python
lazy = ds.read_array(chunks=(1024, 1024), lock=False)
```

## Parallel writes — `Dataset.to_zarr` / `from_zarr`

Zarr is the only raster output format where pyramids can write in
true parallel. Each dask chunk lands in an independent Zarr chunk
file, so a 4-worker write fans out N chunks / 4 per worker with no
coordination. GeoTIFF has no analogue — GDAL serialises its band
writes.

```python
ds = Dataset.read_file("large.tif")

ds.to_zarr("out.zarr", chunks=(1, 1024, 1024), mode="w")
```

The store metadata follows rioxarray's geobox-attribute convention
(`spatial_ref`, `GeoTransform`, `epsg`, `no_data_value`, `band_names`,
`dtype`, `shape`), so a downstream consumer can open the result with
plain rioxarray:

```python
import rioxarray
da = rioxarray.open_rasterio("out.zarr")
```

Round-trip through `Dataset.from_zarr` to stay in the pyramids stack
(returns a `Dataset`, with `chunks=...` keeping the dask backing):

```python
reloaded = Dataset.from_zarr("out.zarr", chunks=(1, 1024, 1024))
```

Pass `compute=False` to defer the write:

```python
delayed = ds.to_zarr("out.zarr", compute=False)
# ... build a larger graph ...
dask.compute(delayed, other_task)
```

`storage_options=` forwards to fsspec so cloud stores work:

```python
ds.to_zarr(
    "s3://bucket/key.zarr",
    chunks=(1, 1024, 1024),
    storage_options={"anon": False, "key": "...", "secret": "..."},
)
```

Install: `[lazy]` extra pulls `zarr` + `fsspec`.

## Neighborhood ops — `focal_*`, `slope`, `aspect`, `hillshade`

Every focal op takes the same `chunks=` switch and resolves to
`dask.array.map_overlap(kernel, depth=radius, boundary="reflect",
trim=True)` when it's set. The kernel is identical to the eager
path (SciPy `ndimage` filters, centered-difference gradients), so the
only thing that changes is the halo bookkeeping.

| Method                                                   | Kernel                                       |
|----------------------------------------------------------|----------------------------------------------|
| `ds.focal_mean(radius)`                                  | `uniform_filter` — box mean                  |
| `ds.focal_std(radius)`                                   | Two-pass stable std (L4)                     |
| `ds.focal_apply(func, radius)`                           | `generic_filter` with user `func`            |
| `ds.slope(units="degrees")` / `"percent"`                | Centered gradient magnitude                  |
| `ds.aspect()`                                            | Centered gradient direction (west-facing 0°) |
| `ds.hillshade(azimuth=315.0, altitude=45.0)`             | Horn's formula                               |

```python
from pyramids.dataset import Dataset

dem = Dataset.read_file("dem.tif")

# eager — same as before
slope_eager = dem.slope()

# lazy — parallelised across chunks
slope_lazy = dem.slope(chunks=(1024, 1024))
slope_lazy.to_hdf5("slope.h5")   # dask IO path
slope_lazy.compute()             # or materialise
```

### Picking the radius / chunk combo

`map_overlap` reads a halo of `depth=radius` around each chunk —
larger radii pull more data. For a 3×3 box filter (`radius=1`) and a
1024×1024 chunk, the halo overhead is 0.4 %; for `radius=50` on a
256×256 chunk, it's ~50 %. Rule of thumb: **keep `chunks >= 10 *
radius` on each axis** to keep halos negligible.

### User kernels via `focal_apply`

```python
import numpy as np
from pyramids.dataset import Dataset

ds = Dataset.read_file("dem.tif")
# max over a 5x5 window, run lazily
big_max = ds.focal_apply(np.max, radius=2, chunks=(512, 512))
```

`func` receives a 1-D flat array of window values and returns one
scalar per window. `generic_filter` is pure-Python and slow — the
`dask` path parallelises its throughput but can't make the kernel
itself faster. For vectorisable kernels, write them as numpy-over-
array and use `dask.array.map_overlap` directly.

## Zonal statistics — `Dataset.zonal_stats`

```python
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

dem = Dataset.read_file("dem.tif")
polys = FeatureCollection.read_file("watersheds.geojson")

df = dem.zonal_stats(polys, stats=("mean", "std", "count"))
# pandas.DataFrame indexed by polys.index, one column per stat
```

Implementation is single-pass: pyramids rasterises every polygon into
one integer label grid, then uses `np.bincount` for sum / mean / count
and a per-label loop for min / max / std / var. CRS is aligned
automatically (H4 — mismatched CRS raises `ValueError` rather than
silently producing wrong numbers).

Area-weighted coverage (where a cell contributes proportionally to
how much of it lies inside the polygon) is **not yet supported** —
the current path is cell-centre-based. An area-weighted
`method="fractional"` implementation is planned; see
`planning/dask/zonal_stats/` for the full plan.

Lazy `LazyFeatureCollection` inputs are not yet supported — call
`.compute()` first. Tracked as a follow-on against DASK-25.

## Interacting with the `dask.array.Array`

Once you have a lazy array, every `dask.array` operation is
available — pyramids does not add a custom wrapper. Common patterns:

```python
lazy = ds.read_array(chunks=(1024, 1024))

# reductions
lazy.mean().compute()
lazy.max(axis=(1, 2)).compute()          # per-band max

# arithmetic — builds graph, no I/O yet
normalised = (lazy - lazy.mean()) / lazy.std()

# pixel-wise where
mask = lazy > 100
masked = lazy.where(mask, other=0)

# rechunk (cheap — just graph rewrite)
rechunked = lazy.rechunk((1, 2048, 2048))

# persist into worker memory (keeps lazy)
lazy_persisted = lazy.persist()
```

`as_numpy(arr)` and `is_lazy(arr)` from `pyramids.base.protocols` are
the two dispatch helpers you'll reach for in mixed-backend code:

```python
from pyramids.base.protocols import is_lazy, as_numpy

result = ds.read_array(chunks="auto")
# ... build graph ...
if is_lazy(result):
    result = as_numpy(result)   # forces compute, no-op for numpy
```

## Minimal-install behaviour

`Dataset.read_array(chunks=...)` with no `dask` installed raises:

```text
ImportError: chunks= requires the optional 'dask' dependency. Install
with: pip install 'pyramids-gis[lazy]'
```

`Dataset.to_zarr` raises the same error naming both `dask` and `zarr`
(the extra pulls both). `Dataset.focal_*`, `slope`, `aspect`,
`hillshade` all work eagerly without the extra; only the `chunks=`
branch gates on it. `Dataset.zonal_stats` works eagerly with no
optional extras.

## Worked example — Sentinel-2 COG on AWS

```python
from pyramids import configure
from pyramids.dataset import Dataset

configure(cloud_defaults=True, aws={"aws_unsigned": True})

url = (
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
    "sentinel-s2-l2a-cogs/35/T/LM/2022/7/"
    "S2A_35TLM_20220716_0_L2A/B08.tif"
)
nir = Dataset.read_file(url)

# Read a 2048×2048 tile near the scene centre as a lazy chunked array.
lazy = nir.read_array(chunks=(512, 512))
print(lazy.shape)                    # (10980, 10980)

# Reduce without loading everything.
peak = lazy.max().compute()
print(f"max DN: {peak}")
```

The full, runnable version is
`docs/examples/dataset/dask-lazy-datasets.ipynb`.
