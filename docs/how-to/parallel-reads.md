# Parallel windowed reads

For I/O-bound workloads — many windows of a large or remote raster — reading windows concurrently is a
quick win. GDAL releases the GIL while it does I/O, so threads genuinely overlap. pyramids exposes this two
ways: the low-level `read_array(threadsafe=True)` primitive, and the `read_windows` convenience that fans a
list of windows across a thread pool for you.

## `read_windows` — the one-liner

```python
from pyramids.dataset import Dataset, Window

ds = Dataset.read_file("big.tif")            # must be path-backed (disk or /vsimem/)
windows = list(ds.block_windows())           # or any list of Window objects
blocks = ds.read_windows(windows, threads=8) # one ndarray per window, in input order
```

`read_windows` submits each window to a `concurrent.futures.ThreadPoolExecutor`, reading every window
through a **per-thread GDAL handle** (`read_array(threadsafe=True)`), and returns the results in the same
order as the input windows. `threads=1` is exactly the sequential path.

## How the threading model works

A single `gdal.Dataset` handle is **not** safe for concurrent calls. pyramids therefore opens one read-only
handle **per thread**, keyed by the dataset's path (the `ThreadLocalFileManager` behind
`read_array(threadsafe=True)`). That is why:

- The dataset must be **reopenable** — path-backed on disk or under `/vsimem/`. A pure in-memory (`MEM`)
  dataset has no path to reopen per thread and is rejected.
- Threads scale with **I/O**, not CPU. For CPU-bound post-processing, move the heavy work into the worker or
  use the Dask path (`read_array(chunks=...)`) instead.

## Sanity-check the speedup yourself

This is illustrative, not a CI assertion — wall-clock numbers depend on the storage and the raster:

```python
import time
from pyramids.dataset import Dataset

ds = Dataset.read_file("big.tif")
windows = list(ds.block_windows())

t0 = time.perf_counter()
_ = [ds.read_array(window=w) for w in windows]          # sequential
seq = time.perf_counter() - t0

t0 = time.perf_counter()
_ = ds.read_windows(windows, threads=8)                  # parallel
par = time.perf_counter() - t0

print(f"sequential {seq:.3f}s  parallel {par:.3f}s  speedup {seq / par:.1f}x")
```

On a large COG over `/vsicurl/` the parallel path is typically several times faster; on a small local file
the thread overhead can make them comparable — measure on your data.

## See also

- [`Dataset.read_windows`](../reference/dataset/io.md) — the API reference.
- `read_array(chunks=...)` — the lazy / Dask path for compute-heavy pipelines.
