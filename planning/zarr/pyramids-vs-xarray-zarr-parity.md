# Zarr functionality parity: `pyramids` vs `xarray`

A feature-parity / gap analysis between the Zarr read-write layer in
**pyramids** (`src/pyramids/dataset/ops/_zarr.py`, `_geobox_zarr.py`,
`collection.py`, `netcdf/_kerchunk_*`, `stac/_loader.py`, `base/remote.py`) and
the Zarr layer in **xarray** (`Dataset.to_zarr` / `xr.open_zarr` /
`open_dataset(engine="zarr")`, plus `DataTree`).

The two projects use Zarr for **different jobs**, so this is not an apples-to-apples
comparison of two interchangeable APIs:

- **xarray** is a general **labeled-array serializer**. Any number of variables,
  arbitrary named dimensions, arbitrary coordinates, hierarchical groups, and a
  full CF encode/decode pipeline. It knows nothing about CRSs or affine
  transforms on its own (that is `rioxarray` / `odc-geo` / the GeoZarr
  extension), and it has no notion of raster overviews.
- **pyramids** is a **georeferenced-raster / datacube serializer** built directly
  on the `zarr` v3 Python API. One `data` array with a fixed axis model
  (`band, y, x` or `time, band, y, x`), a baked-in GeoZarr/CF spatial encoding
  (CRS + 6-element affine), OME-Zarr multiscale pyramids, GDAL interop, and a
  native kerchunk manifest builder.

So "parity" cuts both ways: pyramids does geospatial things xarray's core cannot,
and xarray does array-serialization things pyramids deliberately does not. Both
directions are documented below.

> Scope note: this compares pyramids' **own** GeoZarr reader/writer (path A) to
> xarray core. pyramids also has a second Zarr **read** path through GDAL's
> multidim driver (`NetCDF`/`LabeledDataset`), and a kerchunk facility; those are
> noted where relevant but the head-to-head is pyramids' native writer vs xarray.

---

## 1. Executive summary

| Dimension | pyramids | xarray | Who leads |
|---|---|---|---|
| Georeferencing (CRS + affine) built in | ✅ GeoZarr/CF `spatial_ref`, WKT, EPSG, 6-el affine incl. rotation | ❌ core; needs rioxarray/odc-geo | **pyramids** |
| Multiscale / overview pyramids | ✅ OME-Zarr v0.4 `multiscales`, GDAL overview interop | ❌ core; needs ndpyramid/xarray-multiscale | **pyramids** |
| Native kerchunk / virtual-Zarr builder | ✅ zarr-v3-safe h5py builder | ❌ core (VirtualiZarr/kerchunk external) | **pyramids** |
| GDAL / STAC round-trip | ✅ | ❌ | **pyramids** |
| Arbitrary many variables per store | ❌ one `data` array | ✅ | **xarray** |
| Arbitrary dimensions / coordinates | ❌ fixed `band,y,x` / `time,band,y,x` | ✅ any | **xarray** |
| Hierarchical groups / DataTree | ❌ | ✅ | **xarray** |
| CF scale/offset packing in the store | ❌ materialized to float64 | ✅ `mask_and_scale` | **xarray** |
| Time / calendar coordinate in cube store | ❌ only `time_length` attr | ✅ `decode_times`, cftime | **xarray** |
| Zarr v2 **and** v3 output | ❌ v3 store only (writes v2-compat dim attrs) | ✅ `zarr_format=2|3` | **xarray** |
| Per-variable `encoding` dict (filters, dtype, …) | ⚠️ compressor + chunks + fill only | ✅ full | **xarray** |
| Fully lazy read | ⚠️ cube lazy (dask); `Dataset` materializes to GDAL | ✅ lazy by default | **xarray** |
| Region write / append | ✅ cube: `append_dim`, `region` | ✅ `append_dim`, `region="auto"` | ~parity |
| Consolidated metadata | ✅ always | ✅ `consolidated=` | ~parity |
| Cloud stores (fsspec, requester-pays) | ✅ `storage_options`, requester-pays helpers | ✅ `storage_options` | ~parity |
| Deferred write (`compute=False`) | ✅ `dask.delayed` | ✅ | ~parity |

**One-line takeaway:** pyramids is *ahead* of xarray-core on everything
geospatial (CRS, affine, overviews, GDAL/STAC, kerchunk) and *behind* xarray on
everything general-purpose about array serialization (multi-variable, arbitrary
dims, groups, CF packing/time decode, dual zarr-format output, per-variable
encoding, lazy reads).

---

## 2. API surface, side by side

### pyramids

```python
# single raster
Dataset.to_zarr(store, *, compute=True, mode="w", chunks=None,
                storage_options=None, compressor="auto",
                overview_factors=None, overview_resampling="average")
Dataset.from_zarr(store, *, chunks=None, storage_options=None,
                  level=1, data_name=None)

# 4-D datacube
DatasetCollection.to_zarr(store, *, compute=True, mode="w",
                          storage_options=None, compressor="auto",
                          append_dim=None, region=None)
DatasetCollection.from_zarr(store, *, storage_options=None)

# NetCDF/HDF5 -> virtual Zarr (kerchunk)
NetCDF.to_kerchunk(src, out, *, inline_threshold=500, vlen_encode="embed",
                   backend="native")
NetCDF.combine_kerchunk(srcs, out, *, concat_dims=("time",),
                        identical_dims=("lat","lon"), backend="native")
```
(`src/pyramids/dataset/dataset.py:2147`, `:2252`;
`src/pyramids/dataset/collection.py:1593`, `:2393`;
`src/pyramids/netcdf/netcdf.py:9127`, `:9157`)

### xarray

```python
Dataset.to_zarr(store=None, chunk_store=None, mode=None, synchronizer=None,
                group=None, encoding=None, compute=True, consolidated=None,
                append_dim=None, region=None, safe_chunks=True,
                storage_options=None, zarr_version=None, zarr_format=None,
                write_empty_chunks=None, chunkmanager_store_kwargs=None)

xr.open_zarr(store, group=None, synchronizer=None, chunks="auto",
             decode_cf=True, mask_and_scale=True, decode_times=True,
             concat_characters=True, decode_coords=True, drop_variables=None,
             consolidated=None, overwrite_encoded_chunks=False,
             chunk_store=None, storage_options=None, decode_timedelta=None,
             use_cftime=None, zarr_version=None, zarr_format=None)

xr.open_dataset(store, engine="zarr", ...)      # same via the backend
DataTree.to_zarr(...) / xr.open_datatree(..., engine="zarr")   # hierarchical
```

The shapes rhyme (`store`, `mode`, `compute`, `append_dim`, `region`,
`storage_options`, `consolidated`), but xarray carries a whole extra column of
knobs (`group`, `encoding`, `synchronizer`, `zarr_format`, `safe_chunks`,
`write_empty_chunks`, the full CF decode flags) that pyramids either fixes by
convention or doesn't expose.

---

## 3. Where pyramids leads (xarray-core gaps)

These are things pyramids does out of the box that xarray core cannot do without
third-party extensions.

### 3.1 Georeferencing as a first-class citizen
pyramids writes a **GeoZarr/CF** spatial encoding automatically
(`_geobox_zarr.py:write_geobox`):
- a scalar `spatial_ref` grid-mapping array holding `crs_wkt`, `epsg`, and a
  space-delimited 6-element `GeoTransform` (**rotation and south-up preserved** —
  not just top-left + square cell size, `_zarr.py:940`);
- 1-D pixel-centre `x`/`y` coordinate arrays with CF `axis`/`standard_name`/`units`;
- `grid_mapping="spatial_ref"` on the data array, plus CF `grid_mapping_name` +
  projection parameters for CRSs the CF table recognises;
- both the v2 `_ARRAY_DIMENSIONS` attr **and** the v3-native `dimension_names`
  array property, for cross-reader compatibility.

xarray core has no CRS or affine concept. You get equivalent behavior only via
`rioxarray` (which writes a similar `spatial_ref`/`grid_mapping`) or `odc-geo`.
pyramids also **reads foreign** GeoZarr stores tolerantly (`detect_data_var`
4-rule auto-detect, `read_geobox` transform-from-x/y fallback), so odc-geo /
GDAL / rioxarray stores open without pyramids in the loop.

### 3.2 Multiscale / overview pyramids
`to_zarr(overview_factors=[2,4,8])` writes decimated `data_<factor>` arrays and
an **OGC / OME-Zarr v0.4 `multiscales`** attribute
(`_zarr.py:_write_overview_levels`, `:640`), read back with
`from_zarr(level=4)`. GDAL's Zarr v3 driver picks these up as overviews.
xarray core has nothing here — the ecosystem answer is `ndpyramid` /
`xarray-multiscale`, separate packages.

### 3.3 Native kerchunk / virtual-Zarr builder
`netcdf/_kerchunk_builder.py` emits zarr-v2 reference manifests from NetCDF4/HDF5
via `h5py` **without instantiating a live zarr group** (works around a zarr-v3
`sync()` deadlock, #530), with native concat (`combine_manifests`). xarray core
doesn't build kerchunk references; that's `kerchunk` / `VirtualiZarr` land.

### 3.4 GDAL + STAC round-trip
- `stac/_loader.py:_load_zarr` routes STAC Zarr assets through the shared reader
  (4-D → lazy `DatasetCollection`, else `Dataset`).
- `base/remote.py` maps `application/vnd+zarr` / `.zarr` to a GDAL multidim read
  path and carries requester-pays fsspec helpers.
- The whole store is designed to open in GDAL's Zarr driver (fill_value written
  to the array metadata field GDAL actually reads, not just the CF attribute —
  `_zarr.py:608`).

---

## 4. Where xarray leads (pyramids gaps)

These are the concrete parity gaps if the goal is to make pyramids' Zarr layer as
capable as xarray's as a general serializer. Ordered roughly by impact.

### 4.1 One data variable per store  *(structural)*
pyramids writes a single `data` array; every band is a slice of it. xarray
serializes a whole `Dataset` — arbitrarily many named data variables in one
store (and `DataTree`/groups for nesting). Consequences in pyramids:
- multiple physical variables (e.g. `elevation` + `slope` + a mask) can't live in
  one pyramids store as distinct arrays;
- because all bands share one array, a single `fill_value` can only honestly
  represent a sentinel **all** bands agree on — the well-documented three-spelling
  no-data dance (`_zarr.py:29-33`, `_agreed_sentinel`) exists precisely because of
  this constraint. xarray sidesteps it: each variable has its own `_FillValue`.

**Gap to close for parity:** a multi-array writer (write N named arrays + a shared
geobox), or at least an escape hatch to name the primary array and attach siblings.

### 4.2 Fixed dimensionality  *(structural)*
pyramids' axis model is hard-wired: `(band, y, x)` or `(time, band, y, x)`. xarray
handles any dims/coordinates (e.g. `(member, level, time, lat, lon)`). Anything
that isn't a raster or a time-stacked raster cube has no home in pyramids' Zarr
layer.

### 4.3 CF scale/offset packing not preserved in the store  *(data fidelity)*
The pyramids store has **no `scale_factor`/`add_offset` channel**, so a CF-packed
source is written **materialized to float64** (`-9999` @ scale 0.5 → `-4999.5`),
and reads back declaring no packing (`_zarr.py:_metadata_dict`, `:431-440`).
xarray round-trips packing losslessly via `encoding={"scale_factor":…, "add_offset":…, "dtype":"int16"}`
and decodes it on read with `mask_and_scale=True`. This is a real fidelity/size
gap: a packed int16 dataset round-tripped through pyramids Zarr is ~4× larger and
loses its compact on-disk form.

**Gap to close:** honor `scale_factor`/`add_offset` as encoding on write and
`mask_and_scale` on read (pyramids already knows CF packing — it just doesn't wire
it into the Zarr store).

### 4.4 Cube Zarr drops the time / calendar axis  *(data fidelity)*
`DatasetCollection.to_zarr` writes only a `time_length` **attribute**, not a
`time` coordinate array (`collection.py:1636`). The docs steer you to `to_netcdf`
if you need a real calendar axis. xarray writes a proper CF `time` coordinate
(units, calendar) and decodes it to `datetime64`/`cftime` on read
(`decode_times`, `use_cftime`).

**Gap to close:** emit a CF `time` coordinate array (values + `units` + `calendar`)
in the cube writer; decode it in `from_zarr`.

### 4.5 Zarr v2 output not supported  *(interop)*
pyramids writes through the zarr **v3** store API only (it does add v2-compat
`_ARRAY_DIMENSIONS` dim attrs, and the kerchunk builder emits v2 *reference*
metadata, but you cannot ask `to_zarr` for a v2 **store**). xarray exposes
`zarr_format=2|3` on both read and write. Some consumers are still v2-only.

**Gap to close:** a `zarr_format` / `zarr_version` passthrough on `to_zarr`.

### 4.6 Limited per-array `encoding`  *(control)*
pyramids exposes `compressor` (auto / None / v3 codec list) + `chunks` + the
fill value. xarray takes a full per-variable `encoding` dict: `compressors`,
**`filters`** (e.g. delta/shuffle as a separate pipeline stage), `dtype`,
`_FillValue`, `chunks`, `write_empty_chunks`, `serializer`, etc. pyramids has no
filters channel and no per-band encoding (one array, one codec set).

**Gap to close:** accept an `encoding`-style mapping (at least `filters`, `dtype`,
per-chunk options).

### 4.7 Read of a `Dataset` materializes to NumPy/GDAL  *(laziness)*
`from_zarr` on a single `Dataset` ends in a GDAL-backed, materialized array
(`_read_data_array`: `chunks` gives a *parallel* read but still `.compute()`s to
NumPy because Datasets are GDAL-backed). xarray's `open_zarr` is lazy by default —
nothing loads until you compute, and you keep working in dask/xarray space.
(pyramids' **cube** `from_zarr` *is* lazy dask — `collection.py:1493` — so this gap
is specific to the single-`Dataset` path, which is bounded by the GDAL backing.)

### 4.8 No hierarchical groups / DataTree  *(structure)*
xarray has `group=` and `DataTree.to_zarr`/`open_datatree` for nested groups in
one store. pyramids writes a flat group (base `data` + overview `data_<n>` +
coords). No nested-group model.

### 4.9 Missing operational knobs  *(control)*
xarray exposes `synchronizer` (concurrent-writer locking), `safe_chunks`
(guard against corrupting region writes with mismatched chunks),
`write_empty_chunks` (skip all-fill chunks), and `chunk_store` (separate
metadata/chunk stores). pyramids exposes none of these; region/append safety is
handled internally (e.g. append rolls back a failed resize, `collection.py:1699`)
but isn't user-tunable.

---

## 5. Rough parity (both do it, details differ)

| Feature | pyramids | xarray |
|---|---|---|
| **Append along a dim** | cube `to_zarr(mode="a", append_dim="time")`; requires size-1 time chunks; extends `time_length` + file list, rolls back on failure | `to_zarr(append_dim="time")` |
| **Region write** | cube `to_zarr(mode="a", region={"time": slice(a,b)})`; `mode="a"` alone is rejected | `to_zarr(region=…)` or `region="auto"` (xarray aligns for you) |
| **Consolidated metadata** | always called, with a warning-filter for the v3 "not in spec" `ZarrUserWarning` | `consolidated=True/False/None` |
| **Deferred write** | `compute=False` → `dask.delayed` bundling data + metadata atomically | `compute=False` → dask delayed |
| **Cloud stores** | `storage_options` → `FsspecStore.from_url`; requester-pays helpers in `remote.py`; s3/gs/az/http via fsspec | `storage_options` → fsspec; same schemes |
| **Chunk control on write** | `chunks` 3-tuple `(band,y,x)` triggers rechunk | `encoding={var:{"chunks":…}}` |
| **Compression** | `compressor` auto/None/v3-codec(list) | `encoding={var:{"compressors":…}}` |

Two subtle differences worth noting even where the feature exists on both:
- **`region="auto"`**: xarray can infer the target region from coordinate labels;
  pyramids requires explicit positional slices (`_region_to_slices`).
- **Append safety**: pyramids' append is idempotent under dask recompute and uses a
  synchronous scheduler to avoid nested-compute deadlock (`collection.py:365`); this
  is an internal guarantee, not a user knob like xarray's `safe_chunks`.

---

## 6. Interop reality check (do the stores open in each other?)

- **pyramids store → xarray**: should open with `xr.open_zarr`. The `data` array
  has `_ARRAY_DIMENSIONS`, the CF `grid_mapping`/`spatial_ref` is present, and
  `_FillValue` is written in the CF spelling xarray reads. Caveat: with
  `overview_factors`, xarray sees the `data_<n>` overview arrays as ordinary
  extra variables (it has no multiscale concept). The cube store opens but has
  **no `time` coordinate** (see §4.4).
- **xarray store → pyramids**: pyramids' foreign-store tolerance (`detect_data_var`,
  `read_geobox` with x/y fallback) is built to open rioxarray/odc-geo/GDAL GeoZarr
  stores. A plain xarray store with a non-obvious primary array may need
  `data_name=`. A CF-packed xarray variable would be **materialized** on read into
  a pyramids `Dataset` (no packing channel).
- **GDAL v3 Zarr driver** reads pyramids stores including overviews (that interop
  is an explicit design goal). Note the GDAL-side limitation: GDAL's Zarr driver
  did not support zarr-v3 **string** dtypes before GDAL 3.13 (OSGeo/gdal#13782) —
  pyramids' `LabeledDataset` skips such arrays with a warning.

---

## 7. Suggested parity roadmap (if the goal is to close the gaps)

Ordered by value-to-effort for making pyramids' Zarr layer more xarray-like
without giving up its geospatial strengths:

1. **CF scale/offset packing** (§4.3) — highest fidelity/size win; pyramids already
   models packing, it just needs to write `scale_factor`/`add_offset` as encoding
   and honor them on read. Removes the "packed source silently bloats to float64"
   surprise.
2. **Time coordinate in the cube writer** (§4.4) — write a CF `time` array
   (values + `units` + `calendar`) so cubes stop being second-class vs `to_netcdf`
   and round-trip a real calendar.
3. **`zarr_format` passthrough** (§4.5) — cheap interop win for v2-only consumers.
4. **Per-array `encoding` (esp. `filters`, `dtype`)** (§4.6) — expose the zarr-v3
   codec pipeline pyramids already uses internally.
5. **Multi-array / sibling-variable writes** (§4.1) — larger, structural; would also
   dissolve the shared-`fill_value` constraint. Consider only if multi-variable
   stores are a real use case.
6. **Operational knobs** (§4.9) — `safe_chunks`, `write_empty_chunks` as thin
   passthroughs to the underlying zarr calls.

Items §4.2 (arbitrary dims), §4.7 (lazy single-`Dataset` reads, bounded by GDAL
backing), and §4.8 (DataTree/groups) are deep architectural divergences that
probably shouldn't be "closed" — they're where pyramids' raster/cube model and
xarray's general-array model legitimately part ways.

---

## 8. Reference: file map of pyramids' Zarr code

| Area | File | Key symbols |
|---|---|---|
| Dataset writer/reader | `src/pyramids/dataset/ops/_zarr.py` | `write_dataset_to_zarr:501`, `read_dataset_from_zarr:858`, `_write_overview_levels:640`, `_metadata_dict:333` |
| GeoZarr/CF geobox | `src/pyramids/dataset/ops/_geobox_zarr.py` | `write_geobox:78`, `finalize_zarr_metadata:214`, `read_geobox:597`, `detect_data_var:481`, `normalize_compressors:309` |
| Datacube writer/reader | `src/pyramids/dataset/collection.py` | `to_zarr:1593`, `from_zarr:2393`, `_append_to_zarr:1699`, `_region_to_slices:417` |
| Public forwarders | `src/pyramids/dataset/dataset.py` | `to_zarr:2147`, `from_zarr:2252` |
| Kerchunk (virtual Zarr) | `src/pyramids/netcdf/_kerchunk_builder.py`, `_kerchunk_facade.py` | `build_single_manifest`, `combine_manifests`, `to_kerchunk:221`, `combine_kerchunk:328` |
| STAC Zarr assets | `src/pyramids/stac/_loader.py` | `_load_zarr:401` |
| Cloud/remote | `src/pyramids/base/remote.py` | `s3fs_requester_pays_kwargs:1529`, engine maps |
| User docs | `docs/reference/zarr.md`, `docs/tutorials/lazy/zarr.md`, `docs/tutorials/lazy/kerchunk.md` | — |
