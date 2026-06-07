# DatasetCollection

Time-stacked collection of co-registered rasters. Holds N rasters that share
a spatial template (rows, columns, cell size, CRS) and exposes them along a
single time axis for multi-temporal analysis.

![DatasetCollection diagram](../../_images/pyramids-multi-dataset.svg)

## The two paths

The class operates through **two distinct backing paths**, each serving a
different concern. Knowing which path a method routes through is the key
to using the class correctly.

### Path A — per-timestep `gdal.Dataset` handles

A list of lazy `Dataset` instances, one per timestep, populated on first
access. Each `Dataset.read_file(path)` opens a gdal handle without reading
pixels; the cost is one file descriptor + a metadata read per timestep.
Pixel data flows block-by-block through GDAL when methods invoke
`read_array` / `crop` / `to_crs` etc.

| Method                                         | Behavior                                                              |
|------------------------------------------------|-----------------------------------------------------------------------|
| `iloc(i)`, `__getitem__`, `__setitem__`        | Direct indexed access — returns a `Dataset` (`iloc`) or its array.    |
| `head` / `tail` / `first` / `last`             | Subset views — returns a 3D slice (`head`/`tail`) or 2D array.        |
| `values` getter                                | Derived per-call: `np.stack([ds.read_array() for ds in datasets])`.   |
| `values =` setter                              | Rebuilds the list via `Dataset.create_from_array(...)` per slice.     |
| `crop`, `to_crs`, `align`, `apply`, `overlay`  | Loop the handles; produce a new collection wrapping the results.      |
| `to_file`, `to_cog_stack`                      | Loop the handles; write each timestep to disk.                        |
| `plot`                                         | Materialises the cube on demand for `cleopatra.array_glyph`.          |

Path A works for both **file-backed** and **in-memory** collections.
After a mutating op (in-place `crop`, `apply`, `__setitem__`,
`values =`), the collection becomes in-memory; the new `Dataset`
instances live in the GDAL `MEM` driver and Path A keeps working.

### Path B — dask graph over file paths

A list of path strings. The `data` property assembles a
`dask.array.Array` of shape `(time, bands, rows, cols)` from
`[dask.delayed(_read_time_step)(p) for p in self._files]`. Workers reopen
each path on demand via a process-cached `CachingFileManager` — gdal
handles never cross the pickle boundary, only path strings do. This is
what makes the path scale to `dask.distributed` clusters and to cubes
larger than RAM.

| Method                                          | Behavior                                                        |
|-------------------------------------------------|-----------------------------------------------------------------|
| `data` (property)                               | The dask graph itself.                                          |
| `mean / sum / min / max / std / var`            | Time-axis reductions via `_reduce` over `data`.                 |
| `groupby(labels).<reduction>(...)`              | Per-label reductions via `_GroupedCollection`.                  |
| `to_zarr`                                       | Streams the cube to a Zarr store; never holds it all in RAM.    |
| `to_kerchunk`                                   | Pure metadata pass; reads only a few bytes per file.            |

Path B works for **file-backed** collections only. After a mutating op
clears `_files`, the property raises:

```
RuntimeError: DatasetCollection.data requires a file-backed collection.
Use DatasetCollection.from_files(...) to construct one.
```

The transition is explicit; in-memory collections do not silently
fail or return stale results.

## Boundary

The two paths read different attributes (`_datasets` vs `_files`).
They are **not** parallel views of the same store; they cannot drift.
A collection moves from "file-backed + usable from both paths" to
"in-memory + Path A only" the moment a mutating op runs.

| Operation                                | After:        | Path A | Path B            |
|------------------------------------------|---------------|--------|-------------------|
| `read_multiple_files / from_files / from_stac` | file-backed | works  | works             |
| `crop(inplace=False)` etc. (returns new) | in-memory     | works  | raises (explicit) |
| `crop(inplace=True)` etc. (mutates self) | in-memory     | works  | raises (explicit) |
| `values = arr` / `__setitem__`           | in-memory     | works  | raises (explicit) |

## Cost model

|                                  | Path A                              | Path B                                              |
|----------------------------------|-------------------------------------|-----------------------------------------------------|
| Handles at rest                  | N file descriptors (one per Dataset)| 0                                                   |
| Where reads happen               | Synchronously per-method            | Inside dask tasks (parallelisable across workers)   |
| Caching                          | The collection owns the handles     | Process-global LRU `FILE_CACHE` (default 128 paths) |
| Pickle                           | Cache dropped on `__getstate__`     | Path strings serialise cleanly                      |
| Larger-than-RAM cubes            | Block-streaming per timestep        | Block-streaming across the whole cube + workers     |

## Choosing a path

* **Per-timestep transformations** (crop / reproject / align / apply /
  overlay / write each step to disk) — Path A. Each `Dataset.<op>`
  already block-streams through GDAL; the collection is the loop.
* **Time-axis reductions** (mean / sum / std / groupby) and
  **out-of-process writes** (zarr, kerchunk) — Path B. Pickleable
  paths cross dask boundaries; gdal handles cannot.

::: pyramids.dataset.collection.DatasetCollection
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
