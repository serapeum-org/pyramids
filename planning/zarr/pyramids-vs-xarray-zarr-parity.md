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

## 7. Implementation plans (how to close each gap in pyramids)

Ordered by value-to-effort for making pyramids' Zarr layer more xarray-like
without giving up its geospatial strengths:

Ordered by value-to-effort:

| # | Feature | Gap | Effort | Value | Plan |
|---|---|---|---|---|---|
| 1 | CF scale/offset packing | §4.3 | M | High | §7.1 |
| 2 | Time coordinate in cube writer | §4.4 | S–M | High | §7.2 |
| 3 | `zarr_format` v2/v3 output | §4.5 | M | Med | §7.3 |
| 4 | Per-array `encoding` / `filters` | §4.6 | S–M | Med | §7.4 |
| 5 | Operational knobs (`safe_chunks`, `write_empty_chunks`) | §4.9 | S | Med | §7.5 |
| 6 | `region="auto"` for the cube | §5 note | S | Low | §7.6 |
| 7 | Multi-array / sibling variables | §4.1 | L | Cond. | §7.7 |

Items §4.2 (arbitrary dims), §4.7 (lazy single-`Dataset` reads, bounded by GDAL
backing), and §4.8 (DataTree/groups) are deep architectural divergences that
probably shouldn't be "closed" — they're where pyramids' raster/cube model and
xarray's general-array model legitimately part ways.

The plans below cite **both** codebases: xarray paths are relative to
`xarray/` (v2026.7.0, extracted for this review); pyramids paths are the ones in
§8. A cross-cutting note first, because four of the plans depend on it.

### 7.0 Cross-cutting: the `arr.to_zarr` write call is the main constraint

The base-array write goes through **dask.array's** `arr.to_zarr(store, component="data", overwrite=…, compute=…, **codec_kwargs, **fill_kwargs)`
(`_zarr.py:611`; collection.py:1671-1684, 1734). That call already forwards
`compressors=` and `fill_value=` as `**kwargs` to zarr's array creation, which is
why those two features work today. Features 3–5 need to forward **more** create
kwargs (`zarr_format`, `filters`, `serializer`, `write_empty_chunks`), and dask's
`to_zarr` kwarg-forwarding is version-sensitive across the dask/zarr-v3 boundary.

**Decision point for the implementer:** either (a) confirm each new kwarg
forwards cleanly through `da.Array.to_zarr` on the pinned dask+zarr versions and
keep the current call, or (b) replace the base write with an explicit
`group.create_array(name="data", shape, dtype, chunks, dimension_names, **codec_kwargs)`
followed by `dask.array.store(arr, z, regions=…, lock=…)`. Option (b) is a small
refactor that gives pyramids full control of the create call and makes 7.1/7.3/
7.4/7.5 uniform (one place to thread every create kwarg). **Recommendation: do (b)
first as an enabling refactor**, then layer 7.1–7.5 on top. `_write_overview_levels`
already uses the explicit `root.create_array(...)` form (`_zarr.py:691`), so the
pattern is in-repo.

---

### 7.1 CF scale/offset packing  *(gap §4.3)*

**Goal:** a CF-packed source (int16 + scale/offset) round-trips through Zarr as
packed int16 + `scale_factor`/`add_offset` attrs, instead of being materialised to
float64.

**How xarray does it.** `CFScaleOffsetCoder` (`xarray/coding/variables.py:493`).
Encode (`:508`) leaves data as float and only subtracts offset then divides by
scale — `data -= add_offset; data /= scale_factor` — moving `scale_factor`/
`add_offset` from `encoding` into `attrs` via `pop_to`; the final integer cast is
deferred to `CFMaskCoder`/`NonStringCoder` (which is why scale/offset runs *before*
mask in the coder list, `conventions.py:92`). Decode (`:526`) applies
`data.astype(float); data *= scale_factor; data += add_offset` lazily.
`_choose_float_dtype` (`:442`) picks the decode float width (int32 upcasts to
float64 for precision; offset-only ⇒ always float64). The on-disk array keeps the
packed dtype; the store carries `scale_factor`, `add_offset`, `_FillValue` attrs.

**Where it plugs into pyramids.** The exact seam is the `packed` branch of
`_metadata_dict` (`_zarr.py:431-440`), which today forces `float64` and physical
units. pyramids already owns every primitive needed:
- `Dataset._effective_packing(band)` → `(scale, offset)` (`dataset.py:1478-1535`);
  NetCDF override at `netcdf.py:5953-5981`.
- `apply_unpack(arr, scale, offset)` (`base/_utils.py:1817-1898`) — the shared
  read-side unpack (`arr*scale + offset`).
- `_is_identity_packing` (`base/_utils.py:1588-1621`).
- `Dataset.scale` / `Dataset.offset` setters (`dataset.py:3853-3907` →
  `Bands.scale/offset` `bands.py:1015-1061`, which call `SetScale`/`SetOffset`).
- `read_array(..., unpack=False)` (`engines/io.py:499`) to get **stored counts**.

**Design (mirrors the existing `fill_value` agreement exactly).** One Zarr array
holds every band, so a single `scale_factor`/`add_offset` attribute can only be
honest when **all bands share the same packing** — the same constraint that drove
`_agreed_sentinel` (`_zarr.py:90`). Add an `_agreed_packing(ds)` helper alongside
it:
1. If every band's `_effective_packing` is identical and non-identity → write the
   **stored counts** (`read_array(unpack=False)`), keep the stored dtype, stamp
   `scale_factor`/`add_offset` on the `data` array attrs, and set `_FillValue` /
   `fill_value` to the **stored** sentinel (not `_physical_no_data`).
2. If bands disagree (or mixed packed/unpacked) → keep **today's** behaviour
   (materialise to float64, physical sentinel) and document it. No regression.

**Write path changes:**
- `_build_dask_array` (`_zarr.py:473`) grows an `unpack: bool` parameter and passes
  it to `read_array` (default `True`; `False` on the agreed-packing path).
- `_metadata_dict` (`_zarr.py:333`) emits `scale_factor`/`add_offset` and the
  stored dtype/sentinel on the agreed path.
- `_write_overview_levels` (`_zarr.py:640`) currently *unpacks* GDAL's decimated
  counts (`apply_unpack`, `:680`). On the packed path, keep the counts and carry
  the same `scale_factor`/`add_offset` onto each `data_<factor>` array — CF packing
  is affine, so averaging counts then unpacking equals unpacking then averaging
  (the existing comment at `:673-676` already notes this identity).

**Read path changes:** `read_dataset_from_zarr` (`_zarr.py:858-952`) reads
`scale_factor`/`add_offset` from the array attrs and, after `Dataset.from_array`,
sets `dataset.scale = [...] ; dataset.offset = [...]` so the returned Dataset
carries packing lazily (matching how GeoTIFF/NetCDF reads behave) rather than
eagerly unpacking. `_normalize_no_data` stays as-is (stored sentinel).

**Collection:** same idea in `_unpack_lazy_timestep` (`collection.py:516-572`,
which already branches on `_is_identity_packing`) and `_finalize_collection_metadata`
(`collection.py:280-313`).

**On-disk:** `data` dtype = stored int; attrs `scale_factor`, `add_offset`
(CF spelling xarray/GDAL read), plus existing `no_data_value` (stored sentinel) and
`_FillValue` = stored sentinel. Result: xarray's `mask_and_scale` and GDAL's Zarr
driver both auto-unpack; store is ~4× smaller for int16.

**Tests** (`tests/dataset/ops/` — new `test_zarr_packing.py`): round-trip a packed
int16 `Dataset`; assert on-disk `root["data"].dtype == int16` and the
`scale_factor`/`add_offset` attrs; assert values decode to physical; assert store
size < the float64 path; assert **disagreeing bands** fall back to float64
(regression guard). Add a cube case to `tests/dataset/collection/test_zarr.py`.

**Docs:** `docs/reference/zarr.md` — new "CF packing (scale/offset)" subsection
under the on-disk layout; update the `to_zarr` docstring note that currently says
packed sources are materialised (`_zarr.py:521-527`, dataset.py `to_zarr` docstring).

**Risk:** fill-value semantics flip from physical to stored on the packed path —
must be covered by the sentinel-agreement tests. Overviews must not double-unpack.

---

### 7.2 Time / calendar coordinate in the cube writer  *(gap §4.4)*

**Goal:** `DatasetCollection.to_zarr` writes a real CF `time` coordinate (values +
`units` + `calendar`) so cubes round-trip a calendar and open date-aware in xarray.

**How xarray does it.** `CFDatetimeCoder` (`xarray/coding/times.py:1355`). Encode
(`:1383`) turns datetime64/cftime into a numeric array + `units="{unit} since {ref}"`
+ `calendar` attrs (`encode_cf_datetime`, `:1021`; unit inference `:771`, calendar
inference `:752`). Decode (`:1407`) reverses via `decode_cf_datetime` (`:535`),
falling back to cftime for out-of-range or non-standard calendars.

**Where it plugs into pyramids — the encoder already exists.** pyramids has a full
CF time encoder that the **NetCDF** writer uses and the Zarr writer simply doesn't
call:
- `TimeAxis.resolve(time_coords, length, collection_time)` → `TimeAxis`
  (`_cube_time.py:62-141`); `_encode` (`:143-215`) produces `values` (int64 ns
  since epoch) + `attrs{units, calendar}` using
  `cf_epoch_units` / `CF_EPOCH_CALENDAR="proleptic_gregorian"`
  (`base/_cf_epoch.py:50-128`).
- The NetCDF writer calls exactly this at `_cube_netcdf_writer.py:119` and lays the
  axis into `coords[time_dim] = (axis.values, axis.attrs)` (`:308-310`).
- Reader side already exists too: `pyramids.netcdf.utils.decode_cf_time` /
  `is_cf_time_units` (imported in `netcdf/cf.py:19`).

**Implementation.**
- **Write:** in `DatasetCollection.to_zarr` (`collection.py:1593-1697`) — add a
  `time_coords=None` parameter (mirroring `to_netcdf`), call
  `TimeAxis.resolve(time_coords, self.time_length, self.time)`, and in
  `_finalize_collection_metadata` (`collection.py:280-313`) write `axis.values` as a
  1-D `time` array via `group.create_array`, stamping `axis.attrs` (units/calendar),
  `_ARRAY_DIMENSIONS=["time"]` + native `dimension_names`, and
  `build_coordinate_attrs("time", …)` (`netcdf/cf.py:107-174`, adds `axis="T"`,
  `standard_name`). This mirrors how `write_geobox` writes the `x`/`y` coords
  (`_geobox_zarr.py:143-159`) — reuse that pattern.
- **Append:** `_append_to_zarr` (`collection.py:1699-1754`) must also resize and
  extend the `time` array in lockstep with `data` (it already resizes `data`'s time
  axis at `:1732`).
- **Read:** `from_zarr` (`collection.py:2392-2468`) reads the `time` array + attrs,
  decodes with `decode_cf_time`, and assigns `self._time` (the `time` setter is at
  `collection.py:1114-1138`).
- **Remove** the "does not emit a `time` coordinate" Note (`collection.py:1636-1640`)
  and the docs caveat.

**On-disk:** `time` 1-D int64 array with `units="nanoseconds since 1970-01-01…"`,
`calendar="proleptic_gregorian"`, `axis="T"`, `standard_name="time"`,
`_ARRAY_DIMENSIONS=["time"]`.

**Tests** (`tests/dataset/collection/test_zarr.py`): build a cube with explicit
datetimes, round-trip, assert `col.time` recovered; assert `xr.open_zarr` decodes
the dates; append case extends the time axis.

**Docs:** `docs/reference/zarr.md` cube section (drop the "no time coordinate" note);
`docs/tutorials/lazy/zarr.md`.

**Risk:** low — the encoder/decoder are battle-tested by the NetCDF path. Main care
is append keeping `time` and `data` lengths in sync (roll back both on failure, as
the existing append already does for `data`).

---

### 7.3 Zarr v2 **and** v3 output  *(gap §4.5)*

**Goal:** `to_zarr(..., zarr_format=2|3)` so v2-only consumers can read pyramids
stores. Today the writer is v3-only.

**How xarray does it.** `_handle_zarr_version_or_format` (`xarray/backends/zarr.py:2005`)
reconciles the deprecated `zarr_version` with `zarr_format`; `_zarr_v3()` (`:109`)
detects the **library** major version (≥3) separately from the **format** written.
In the write path (`set_variables`, `:1209`): v3 stores dimension names in the
native `dimension_names` metadata field (`:1342`) while v2 uses the
`_ARRAY_DIMENSIONS` attr (`:1344`); codecs are `compressor` (singular, v2) vs
`compressors`/`filters`/`serializer` (v3); `fill_value` is a valid encoding key only
on v3.

**Where it plugs into pyramids.** There is **no version detection anywhere** today,
and every write site assumes the v3 API. The good news: pyramids already writes the
v2-compat `_ARRAY_DIMENSIONS` attr *alongside* the v3 `dimension_names` kwarg
(`_geobox_zarr.py:119-129`), so a v2 store just needs the kwarg dropped.
- **Add detection** in `_require_zarr()` (`_zarr.py:77`): return `zarr.__version__`
  parts (or reuse `packaging`) so callers can branch.
- **`normalize_compressors`** (`_geobox_zarr.py:309-322`) grows a `zarr_format` arg
  and returns `compressor=`/`filters=` for v2 vs `compressors=`/`filters=`/
  `serializer=` for v3.
- **Every `create_array` call** (`_geobox_zarr.py:119`, `_zarr.py:691`) drops
  `dimension_names=` on v2 (relying on the `_ARRAY_DIMENSIONS` attr already written).
- **The base write** (`_zarr.py:611`) forwards `zarr_format=`; this is the call the
  §7.0 refactor makes reliable (dask's `to_zarr` v2/v3 forwarding is the risk).
- **`open_group`/`consolidate_metadata`/`FsspecStore`** (`_zarr.py:923`, `746`, `968`)
  — pass `zarr_format` where the installed zarr requires it.
- Thread `zarr_format` through `Dataset.to_zarr`/`DatasetCollection.to_zarr`
  signatures (public forwarders `dataset.py:2147`, `collection.py:1593`).

**On-disk:** v2 → `.zarray`/`.zattrs`/`.zgroup`, `_ARRAY_DIMENSIONS`; v3 → `zarr.json`,
native `dimension_names`. (The kerchunk builder already emits v2 *reference*
metadata, `netcdf/_kerchunk_builder.py`, so the v2 attribute conventions are
in-repo for reference.)

**Tests:** parametrize the existing round-trip tests over `zarr_format=[2, 3]`
(`tests/dataset/ops/test_zarr.py`, `tests/dataset/collection/test_zarr.py`);
assert the on-disk marker file (`.zarray` vs `zarr.json`).

**Docs:** `docs/reference/zarr.md` Codec/compression section.

**Risk:** medium — mostly the dask `to_zarr` forwarding (§7.0). Do §7.0 first.

---

### 7.4 Per-array `encoding` / `filters`  *(gap §4.6)*

**Goal:** expose the zarr codec **pipeline** (filters/serializer, explicit dtype)
that pyramids uses internally, not just a single `compressor`.

**How xarray does it.** `extract_zarr_variable_encoding` (`xarray/backends/zarr.py:471`)
keeps a whitelist — `{chunks, shards, compressor, compressors, filters, serializer,
write_empty_chunks, chunk_key_encoding, fill_value(v3)}` (`:496-508`) — and splats it
straight into `zarr.create` (`_create_new_array`, `:1173`). It does **not** translate
v2 filters ↔ v3 codecs; it forwards whatever the user set and validates against the
format's allowed set.

**Where it plugs into pyramids.** Because pyramids has one `data` array, "encoding"
is per-store, not per-variable — simpler than xarray. `normalize_compressors`
(`_geobox_zarr.py:309`) is the single funnel and today only produces `compressors=`.
- Broaden it (or add an `encoding: dict | None` / `filters=` parameter on `to_zarr`)
  to build a `codec_kwargs` dict including `filters=` and, on v3, `serializer=`,
  validated against a small whitelist mirroring xarray's.
- Thread it through the same create/`to_zarr` calls as §7.3 (shared plumbing — **do
  7.3 and 7.4 together**).
- Optional `dtype=` override belongs with §7.1 (it changes the on-disk cast).

**Tests:** `tests/dataset/ops/test_zarr.py` — write with a delta/shuffle filter +
Blosc, assert the codec chain lands in the array metadata and values round-trip.

**Docs:** `docs/reference/zarr.md` Codec/compression section (extend the existing
`compressor=` docs).

**Risk:** low once §7.0/§7.3 plumbing exists.

---

### 7.5 Operational knobs: `safe_chunks`, `write_empty_chunks`  *(gap §4.9)*

**Goal:** user control over the two write-safety knobs xarray exposes.

**How xarray does it.** `write_empty_chunks` is plumbed to `create`/(v3) nested
`config` (`xarray/backends/zarr.py:1191-1197`). `safe_chunks` guards parallel dask
writes where multiple dask chunks map onto one zarr chunk/shard, via
`validate_grid_chunks_alignment` (`xarray/backends/chunks.py:184`); region writes in
`r+` set `allow_partial_chunks=False` to force exact last-chunk alignment.

**Where it plugs into pyramids.**
- `write_empty_chunks`: passthrough on `to_zarr` → the create call (v3: nested in
  `config=`). Trivial once §7.0/§7.3 land.
- `safe_chunks`: most relevant to the cube region/append path (`_append_to_zarr`
  `collection.py:1699`, `_region_to_slices` `:417`). pyramids already forces size-1
  time chunks for append, so alignment is largely controlled; add a user-facing
  `safe_chunks: bool = True` and a lightweight alignment assertion (each incoming
  dask chunk a multiple of the store chunk on the write axis) mirroring xarray's
  interior-chunk check. `safe_chunks=False` skips it.

**Tests:** `tests/dataset/collection/test_zarr.py` — a misaligned region write raises
with `safe_chunks=True` and succeeds (with a corruption caveat) at `False`.

**Docs:** `docs/reference/zarr.md` incremental-writes section.

**Risk:** low.

---

### 7.6 `region="auto"` for the cube  *(parity refinement, §5 note)*

**Goal:** `DatasetCollection.to_zarr(mode="a", region="auto")` resolves the target
time slice from datetime labels, matching xarray.

**How xarray does it.** `_auto_detect_regions` (`xarray/backends/zarr.py:1366`):
reads & CF-decodes the store's existing coordinate, builds a `pd.Index`, does
`index.get_indexer(new_labels)`, then validates the result is all-found and
**contiguous** (`np.diff(idxs) == 1`) before turning it into a `slice`.

**Where it plugs into pyramids.** This **depends on §7.2** (there must be a `time`
coordinate in the store to match against). Once §7.2 lands: in
`DatasetCollection.to_zarr`, when `region="auto"`, read the store's `time` array,
`get_indexer` the collection's own `time` values to positions, assert contiguity,
and hand the resulting `{"time": slice(a, b)}` to the existing
`_region_to_slices` (`collection.py:417`) path. Everything downstream is unchanged.

**Tests:** append a mid-cube slice by dates with `region="auto"`; assert it targets
the right integer slice and rejects non-contiguous/unknown dates.

**Risk:** low; purely additive on top of §7.2.

---

### 7.7 Multi-array / sibling variables  *(gap §4.1 — structural, likely defer)*

**Goal (if pursued):** more than one named data array per store (e.g. `elevation` +
`slope`), each with its own dtype/fill/packing — which also dissolves the
shared-`fill_value` and shared-packing constraints that §7.1 and the existing
sentinel logic work around.

**Reality.** The **read** side is already multi-array-aware: `detect_data_var`
(`_geobox_zarr.py:481`) has foreign-store logic (prefers `data`, then a
`grid_mapping` array, filters CF-referenced/coordinate arrays), and
`read_dataset_from_zarr` accepts `data_name=`. The **write** side hard-codes
`root["data"]` everywhere (`finalize_zarr_metadata` `_geobox_zarr.py:248`,
`write_geobox(data_name="data")`, the collection writer). A real multi-variable
writer would be a new entry point — e.g. `write_datasets_to_zarr(mapping: dict[str,
Dataset], store, *, shared geobox)` — writing N arrays that share one `spatial_ref`
+ `x`/`y`, each carrying its own `_FillValue`/`scale_factor`. `band_names` become
per-array variable names.

**Recommendation:** **defer** unless multi-variable stores are a real requirement —
it's a genuine model divergence (pyramids is a raster/cube library, not a general
Dataset serializer), and §7.1 already removes the most painful symptom (float64
bloat) without it.

---

### 7.8 Suggested sequencing

1. **§7.0 enabling refactor** — explicit `create_array` + `da.store` for the base
   write (unblocks 3/4/5 cleanly).
2. **§7.2 time coordinate** — self-contained, the encoder already exists, unblocks §7.6.
3. **§7.1 packing** — highest fidelity/size win; self-contained.
4. **§7.3 + §7.4 `zarr_format` + encoding/filters** — shared create-kwarg plumbing.
5. **§7.5 knobs** — cheap passthroughs on the same plumbing.
6. **§7.6 `region="auto"`** — small, after §7.2.
7. **§7.7 multi-variable** — only if demanded.

Each of 7.1–7.6 is independently shippable behind its own PR and its own tests; none
changes the default on-disk layout for an unpacked, single-timestep,
v3-default store, so existing stores and readers are unaffected.

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
| Packing primitives | `src/pyramids/base/_utils.py` | `apply_unpack:1817`, `_is_identity_packing:1588`, `write_packing:1719` |
| Packing resolver | `src/pyramids/dataset/dataset.py` | `_effective_packing:1478`, `scale:3853`, `offset:3887` |
| Physical no-data | `src/pyramids/dataset/engines/analysis.py` | `_physical_no_data:1168` |
| CF time encoder | `src/pyramids/dataset/_cube_time.py` | `TimeAxis.resolve:62`, `_encode:143` |
| CF epoch constants | `src/pyramids/base/_cf_epoch.py` | `cf_epoch_units:57`, `CF_EPOCH_CALENDAR:54` |
| NetCDF cube time path | `src/pyramids/netcdf/_cube_netcdf_writer.py`, `netcdf/cf.py` | `write:119`, `build_coordinate_attrs:107` |
| Optional-dep gating | `src/pyramids/base/_utils.py` | `import_zarr:1458`, `require_optional:1218`, `lazy_extra_hint:1375` |
| User docs | `docs/reference/zarr.md`, `docs/tutorials/lazy/zarr.md`, `docs/tutorials/lazy/kerchunk.md` | — |

## 9. Reference: xarray implementation index (for the §7 plans)

Paths relative to `xarray/` (v2026.7.0). xarray's scale/offset, masking and time
encoding are **not** zarr-specific — they run in a generic CF coder pipeline shared
with the netCDF backend; only the second block is zarr-specific.

| Concern | File | Key symbols |
|---|---|---|
| Coder pipeline order | `conventions.py` | `encode_cf_variable:68` (order `:92`), `decode_cf_variable:109` (`:196`) |
| Scale/offset | `coding/variables.py` | `CFScaleOffsetCoder:493` (encode `:508`, decode `:526`), `_choose_float_dtype:442` |
| Mask / fill / unsigned | `coding/variables.py` | `CFMaskCoder:269`, `_check_fill_values:167`, unsigned `:203`/`:239` |
| Time / calendar | `coding/times.py` | `CFDatetimeCoder:1355`, `encode_cf_datetime:1021`, `decode_cf_datetime:535`, `infer_datetime_units:771` |
| zarr encoding extraction | `backends/zarr.py` | `extract_zarr_variable_encoding:471` (valid keys `:496`), `_determine_zarr_chunks:336` |
| zarr fill value | `backends/zarr.py` | `FillValueCoder:118`, fill dispatch `:1243-1264` |
| zarr write core | `backends/zarr.py` | `set_variables:1209`, `_create_new_array:1173`, dims `:411` |
| region / append | `backends/zarr.py` | `_auto_detect_regions:1366`, `_validate_and_autodetect_region:1399` |
| format / open | `backends/zarr.py` | `_get_open_params:1884`, `_handle_zarr_version_or_format:2005`, `_zarr_v3:109` |
| safe_chunks | `backends/chunks.py` | `validate_grid_chunks_alignment:184`, `grid_rechunk:156` |
| top-level entry | `backends/writers.py` | `to_zarr:733`, `get_writable_zarr_store:634`, `_datatree_to_zarr:928` |

---

## 10. Implementation tasks (self-contained specs)

> **Read this first.** Each task below is written to be executed **without
> guessing**: it embeds the *current* pyramids code being changed, the *exact*
> replacement, the on-disk attribute names, the full test code to add, and the
> docstring edits. Line numbers are from the checkout these specs were written
> against (`src/pyramids/dataset/ops/_zarr.py` @ 977 lines,
> `_geobox_zarr.py` @ 671 lines, `collection.py`, etc.) — **re-`grep` the anchor
> strings** shown in each step before editing, because earlier tasks shift line
> numbers. Anchors are quoted verbatim so they're greppable.

### 10.0 Conventions used in every task

- **Repo tooling.** Tests run under `pixi` (see `pyproject.toml`). Typical loop:
  `pixi run -e default pytest tests/dataset/ops/test_zarr.py -q`. Lint/format via
  the repo `pre-commit` (`.pre-commit-config.yaml`): run `pixi run pre-commit run
  --files <changed files>` before committing. Zarr tests are gated
  `@pytest.mark.lazy` and skipped when the `[lazy]` extra (`zarr>=3`, `dask`,
  `fsspec`) is absent — never remove that gate.
- **Docstring style.** This codebase uses Google-style docstrings **with runnable
  doctests** (see any function in `_zarr.py`). New public behaviour should add a
  doctest marked `# doctest: +SKIP` when it needs the `[lazy]` extra, mirroring
  `write_dataset_to_zarr`'s docstring.
- **No-guess rule for attributes.** Every attribute this plan writes uses a name
  already established in the store or the CF convention. The full current
  attribute set is: root group — `pyramids_zarr_version`, `time_length`,
  `pyramids_file_list`, `multiscales`; `data` array — `_ARRAY_DIMENSIONS`,
  `grid_mapping`, `no_data_value`, `_FillValue`, `band_names`, `dtype`, `shape`,
  `epsg`, `GeoTransform`, `nodata` (cube spelling); `spatial_ref` scalar —
  `crs_wkt`, `spatial_ref`, `GeoTransform`, `epsg`, plus CF grid-mapping params.
  Do not invent new names beyond the ones each task specifies.
- **Definition of Done (DoD)** is a checklist. A task ships only when every box is
  literally checkable by running the named command / opening the store and
  asserting the named attribute.

---

### TASK-0 — Enabling refactor: own the `data`-array create call

**Why.** Four later tasks (3, 4, 5, and cleanly 1) need to pass extra kwargs to
the zarr array creation (`zarr_format`, `filters`, `serializer`,
`write_empty_chunks`, an explicit `dtype`). Today the base `data` array is created
*implicitly* by dask inside `arr.to_zarr(...)`, which only reliably forwards
`compressors=`/`fill_value=`. This task moves the create into pyramids so there is
**one** call site that owns every create kwarg.

**Current code — `src/pyramids/dataset/ops/_zarr.py`, `write_dataset_to_zarr`
(anchor: `write_result = arr.to_zarr(`, ~line 611):**

```python
    arr = _build_dask_array(ds, chunks)
    metadata = _metadata_dict(ds)
    resolved_store = _resolve_store(store, storage_options)

    codec_kwargs = normalize_compressors(compressor)
    fill_kwargs = (
        {"fill_value": metadata["_FillValue"]} if "_FillValue" in metadata else {}
    )
    write_result = arr.to_zarr(
        resolved_store,
        component="data",
        overwrite=(mode == "w"),
        compute=compute,
        **codec_kwargs,
        **fill_kwargs,
    )
    if compute:
        _finalize_metadata(resolved_store, metadata)
        ...
```

**Replacement approach.** Introduce a private helper `_create_and_store_data(...)`
in `_zarr.py` that (a) opens/creates the group, (b) `group.create_array("data",
shape=arr.shape, dtype=arr.dtype, chunks=<resolved>, dimension_names=("band","y","x"),
overwrite=(mode=="w"), **codec_kwargs, **fill_kwargs)`, then (c) writes the dask
array into it with `dask.array.store(arr, z, lock=…, compute=compute)`. Return the
same `write_result` semantics (`None` for `compute=True`, a `Delayed`/stored value
for `compute=False`) so the existing `_finalize_after_write` delayed path is
unchanged.

```python
def _create_and_store_data(
    resolved_store: Any,
    arr: Any,
    *,
    name: str,
    mode: str,
    dimension_names: tuple[str, ...],
    codec_kwargs: dict[str, Any],
    fill_kwargs: dict[str, Any],
    compute: bool,
    zarr_format: int | None = None,   # wired in TASK-3; ignored/None here
) -> Any:
    """Create the target array and stream `arr` into it, returning the store task.

    Replaces the implicit array creation dask's `arr.to_zarr` did, so pyramids owns
    every `create_array` kwarg (codecs, fill value, dimension names, and — from
    TASK-3 on — `zarr_format`). Chunks come from the dask array's own block shape so
    the on-disk chunking matches what the caller rechunked to in `_build_dask_array`.
    """
    zarr = _require_zarr()
    import dask.array as da

    root = zarr.open_group(resolved_store, mode=("w" if mode == "w" else "a"))
    create_kwargs: dict[str, Any] = dict(
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=tuple(c[0] for c in arr.chunks),  # first block per axis = uniform chunk
        dimension_names=dimension_names,
        overwrite=(mode == "w"),
        **codec_kwargs,
        **fill_kwargs,
    )
    if zarr_format is not None:
        create_kwargs["zarr_format"] = zarr_format
    z = root.create_array(name, **create_kwargs)
    return da.store(arr, z, lock=True, compute=compute, return_stored=False)
```

Then in `write_dataset_to_zarr` replace the `arr.to_zarr(...)` block with:

```python
    write_result = _create_and_store_data(
        resolved_store, arr,
        name="data", mode=mode,
        dimension_names=("band", "y", "x"),
        codec_kwargs=codec_kwargs, fill_kwargs=fill_kwargs,
        compute=compute,
    )
```

Do the **same** substitution in `DatasetCollection.to_zarr`
(`collection.py`, anchor `write_result = data.to_zarr(`, ~line 1678) with
`dimension_names=("time", "band", "y", "x")`, and in the region/append paths
(`data.to_zarr(existing, region=…)` at ~1671 and ~1734, and `_append_region`
~409) — but those write into an **already-created** array, so they keep using
`da.store(data, existing_array, regions=[slices], compute=…)` instead of
`create_array`. (dask's `Array.to_zarr` into an existing `zarr.Array` is just
`da.store`; switching keeps behaviour identical.)

**Gotcha — chunk uniformity.** `create_array(chunks=…)` needs one chunk tuple per
axis; dask allows a ragged last block. Use `tuple(c[0] for c in arr.chunks)` (first
block per axis) exactly as xarray does (`_determine_zarr_chunks`,
`xarray/backends/zarr.py:357-372`), and ensure `_build_dask_array` produces
uniform interior chunks (it already rechunks to a single tuple when `chunks` is a
3-tuple).

**Gotcha — atomicity for `compute=False`.** The current code bundles the data
write + metadata write into one `dask.delayed(_finalize_after_write)(write_result,
…)`. Keep that: `da.store(..., compute=False)` returns a `Delayed`/list you pass as
`write_result` to `_finalize_after_write` unchanged.

**Scope (files):** `src/pyramids/dataset/ops/_zarr.py`
(`write_dataset_to_zarr:501`, new `_create_and_store_data`),
`src/pyramids/dataset/collection.py` (`to_zarr:1593`, `_append_to_zarr:1699`,
`_append_region:365`, region branch ~1664).

**Dependencies:** none. **Enables:** TASK-1 (clean dtype control), TASK-3, TASK-4,
TASK-5.

**DoD:**
- [ ] `grep -n "arr.to_zarr\|data.to_zarr" src/pyramids/dataset/ops/_zarr.py src/pyramids/dataset/collection.py` returns **no** create-time calls (region/append into an existing array may still use `da.store`).
- [ ] `pixi run pytest tests/dataset/ops/test_zarr.py tests/dataset/collection/test_zarr.py tests/dataset/ops/test_zarr_affine_roundtrip.py tests/dataset/ops/test_zarr_fill_value.py -q` is green with **zero** changes to those test files (pure refactor; behaviour identical).
- [ ] A written store still has `root["data"].chunks` equal to the pre-refactor chunking for the `small_dataset` fixture (add a temporary assert, then remove).
- [ ] `compute=False` still returns a `Delayed` whose `.compute()` finalizes metadata after the data write (existing `TestDeferred`-style tests pass).
- [ ] No public API or docstring change.

---

### TASK-1 — CF scale/offset packing preserved in the store

**Objective.** A CF-packed source (e.g. `int16` with `scale=0.01`) round-trips
through Zarr as **packed `int16` + `scale_factor`/`add_offset` attributes**, not
materialised to `float64`. On read, the returned `Dataset` carries the packing
(lazy), matching how GeoTIFF/NetCDF reads behave.

**Background — how packing is modelled (do not re-derive):**
- `apply_unpack(arr, scale, offset)` (`base/_utils.py:1817`) = `arr*scale+offset`
  as float64; identity `(None|1.0, None|0.0)` is a genuine no-op.
- `_is_identity_packing(scale, offset)` (`base/_utils.py:1588`) — treats `None`,
  `(1.0, 0.0)`, and an unusable scale (0/non-finite) as identity; array-valued
  counts only when **every** element is identity.
- `Dataset._effective_packing(band=0) -> (scale, offset)` (`dataset.py:1478`) —
  the canonical per-band resolver (`GetScale()`/`GetOffset()`, either may be
  `None`).
- `Dataset.scale` / `Dataset.offset` setters (`dataset.py:3853` / `:3887` →
  `Bands.scale/offset` `engines/bands.py:1015-1061`, calling `SetScale`/`SetOffset`
  per band). Getters normalise unset → `1.0` / `0`.
- `Dataset.read_array(..., unpack=True, ...)` (facade `dataset.py`; engine
  `engines/io.py:499`) — `unpack=False` returns **stored counts**.
- `Analysis._physical_no_data(band)` (`engines/analysis.py:1168`) — the sentinel in
  physical units; what the store's `fill_value` uses **today** on the packed path.

**Current code — the seam, `src/pyramids/dataset/ops/_zarr.py`, `_metadata_dict`
(anchor: `packed = any(`, ~line 431):**

```python
    packed = any(
        not _is_identity_packing(*ds._effective_packing(index))
        for index in range(ds.band_count)
    )
    nodata_tuple = (
        tuple(ds.analysis._physical_no_data(index) for index in range(ds.band_count))
        if packed
        else ds.no_data_value
    )
    written_dtype = np.dtype("float64") if packed else np.dtype(ds.numpy_dtype[0])
```

This is the exact branch that materialises. The plan **adds a third state**: *packed
AND all bands agree on the same non-identity `(scale, offset)`* → keep stored dtype,
write counts, stamp `scale_factor`/`add_offset`, and use the **stored** sentinel.

**Step 1 — add an agreement helper next to `_agreed_sentinel` (`_zarr.py:90`).**

```python
def _agreed_packing(ds: Dataset) -> tuple[float, float] | None:
    """The one `(scale, offset)` every band shares, or `None` when they differ.

    One Zarr array holds every band, so a single `scale_factor` / `add_offset`
    attribute can only be honest when all bands pack identically — the same
    constraint `_agreed_sentinel` enforces for `fill_value`. Returns the shared
    non-identity pair, or `None` when the bands disagree, when any band is
    unpacked, or when there are no bands (so the caller keeps materialising).
    """
    if ds.band_count == 0:
        return None
    pairs = [ds._effective_packing(i) for i in range(ds.band_count)]
    if any(_is_identity_packing(s, o) for s, o in pairs):
        return None  # a mix of packed/unpacked bands has no shared packing
    first_s, first_o = pairs[0]
    s0 = 1.0 if first_s is None else float(first_s)
    o0 = 0.0 if first_o is None else float(first_o)
    for s, o in pairs[1:]:
        s_i = 1.0 if s is None else float(s)
        o_i = 0.0 if o is None else float(o)
        if not (math.isclose(s_i, s0) and math.isclose(o_i, o0)):
            return None
    return (s0, o0)
```

(`math` is already imported in `_zarr.py`.)

**Step 2 — branch `_metadata_dict` on the agreed-packing case.** Replace the
`packed`/`nodata_tuple`/`written_dtype` block above with:

```python
    agreed_packing = _agreed_packing(ds)
    packed = any(
        not _is_identity_packing(*ds._effective_packing(index))
        for index in range(ds.band_count)
    )
    if agreed_packing is not None:
        # Store counts + CF scale/offset; sentinel and dtype stay as STORED.
        nodata_tuple = ds.no_data_value          # stored-unit sentinels
        written_dtype = np.dtype(ds.numpy_dtype[0])
    elif packed:
        # Bands disagree (or mixed): fall back to today's materialise-to-float64.
        nodata_tuple = tuple(
            ds.analysis._physical_no_data(index) for index in range(ds.band_count)
        )
        written_dtype = np.dtype("float64")
    else:
        nodata_tuple = ds.no_data_value
        written_dtype = np.dtype(ds.numpy_dtype[0])
```

Then, before `return metadata`, stamp the CF packing attrs when agreed:

```python
    if agreed_packing is not None:
        scale, offset = agreed_packing
        metadata["scale_factor"] = scale
        metadata["add_offset"] = offset
```

`scale_factor`/`add_offset` land on the `data` array via the existing
`finalize_zarr_metadata(... data_attrs=metadata ...)` path (`_finalize_metadata`
already forwards `metadata` as `data_attrs`). **These are the CF spellings xarray's
`mask_and_scale` and GDAL's Zarr driver both read.**

**Step 3 — write stored counts, not physical values.** In `_build_dask_array`
(`_zarr.py:473`) add an `unpack: bool = True` param and pass it through to
`read_array`:

```python
def _build_dask_array(ds: Dataset, chunks: Any, *, unpack: bool = True) -> Any:
    ...
    arr = ds.read_array(chunks=read_chunks, unpack=unpack)
    ...
```

In `write_dataset_to_zarr`, decide `unpack` from the same agreement check so the
array and its metadata agree:

```python
    unpack = _agreed_packing(ds) is None      # store counts only on the agreed path
    arr = _build_dask_array(ds, chunks, unpack=unpack)
    metadata = _metadata_dict(ds)
```

**Step 4 — `fill_value` must be the stored sentinel on the packed path.** The
existing `_metadata_dict` computes `_FillValue` from `no_data_list` via
`_agreed_sentinel` (`_zarr.py:458`). Because Step 2 sets `nodata_tuple =
ds.no_data_value` (stored units) on the agreed path, `_FillValue` and the array
`fill_value` are already the **stored** sentinel — no extra change. Verify the
`_representable(agreed, written_dtype)` guard (`_zarr.py:468`) still holds (stored
sentinel in the stored integer dtype → representable).

**Step 5 — overviews carry the same packing.** In `_write_overview_levels`
(`_zarr.py:640`) the current code **unpacks** GDAL's decimated counts
(`apply_unpack(...)`, anchor `apply_unpack(` ~line 680) and writes physical values.
On the agreed-packing path, **keep the counts** and stamp `scale_factor`/
`add_offset` (from `metadata`) on each `data_<factor>` array. Change:

```python
        levels = [
            np.asarray(
                apply_unpack(                      # <-- REMOVE unpack on agreed path
                    np.asarray(ds.raster.GetRasterBand(b + 1).GetOverview(ov_index).ReadAsArray()),
                    *ds._effective_packing(b),
                )
            )
            for b in range(band_count)
        ]
```

to (guarded):

```python
        agreed = "scale_factor" in metadata and "add_offset" in metadata
        levels = [
            np.asarray(ds.raster.GetRasterBand(b + 1).GetOverview(ov_index).ReadAsArray())
            if agreed
            else np.asarray(
                apply_unpack(
                    np.asarray(ds.raster.GetRasterBand(b + 1).GetOverview(ov_index).ReadAsArray()),
                    *ds._effective_packing(b),
                )
            )
            for b in range(band_count)
        ]
```

and after `za.attrs["_ARRAY_DIMENSIONS"] = ...` add:

```python
        if "scale_factor" in metadata:
            za.attrs["scale_factor"] = metadata["scale_factor"]
            za.attrs["add_offset"] = metadata["add_offset"]
```

The affine-average identity that makes this correct is already documented in the
comment at `_zarr.py:673-676` ("unpacking the averaged counts equals averaging the
physical values"). Also set the level array's `dtype` to the counts' dtype (it
already uses `level_arr.dtype`, so keeping counts keeps the integer dtype).

**Step 6 — read side sets packing on the returned Dataset.** In
`read_dataset_from_zarr` (`_zarr.py:858`), after `dataset =
Dataset.from_array(...)` and `dataset.bands.apply_names(...)` (anchor
`dataset.bands.apply_names(`, ~line 951), add:

```python
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None or offset is not None:
        # Carry packing lazily (like a GeoTIFF/NetCDF read) instead of unpacking now.
        dataset.scale = [float(scale) if scale is not None else 1.0] * dataset.band_count
        dataset.offset = [float(offset) if offset is not None else 0.0] * dataset.band_count
```

`attrs` here is `dict(zarr_array.attrs)` already read at `_zarr.py:927`. The stored
`no_data_value`/`_normalize_no_data` path is unchanged (stored sentinel), which is
correct because the returned Dataset is now packed.

**Step 7 — collection (cube) parity.** The cube reads per-file packing in
`_unpack_lazy_timestep` (`collection.py:516`, already branches on
`_is_identity_packing`). For the cube **writer**, add the same agreed-packing logic
to `_finalize_collection_metadata` (`collection.py:280`) `data_attrs` and to the
`self.data` build (write counts when the cube's `meta` packing agrees). Scope this as
a **follow-up sub-PR** if the single-`Dataset` path lands first; the DoD covers both.

**On-disk result:** `root["data"].dtype == int16`; `data` attrs include
`scale_factor`, `add_offset`, `no_data_value` (stored), `_FillValue` (stored),
`dtype="int16"`. `xr.open_zarr(store)` returns **physical** (unpacked) values;
GDAL's Zarr driver reports the scale/offset.

**Full test file — create `tests/dataset/ops/test_zarr_packing.py`:**

```python
"""CF scale/offset packing round-trips through Zarr (parity gap 4.3 / TASK-1)."""
from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None


def _packed_tif(tmp_path, name="packed.tif", scale=0.01, offset=0.0, nodata=-9999):
    """A saved 1-band int16 raster with uniform packing, reopened from disk."""
    arr = np.array([[100, 200], [300, nodata]], dtype="int16")
    ds = Dataset.from_array(
        arr,
        no_data_value=nodata,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        path=str(tmp_path / name),
    )
    ds.scale = [scale]
    ds.offset = [offset]
    ds.close()
    return Dataset.read_file(str(tmp_path / name))


@pytest.mark.lazy
def test_packed_store_keeps_integer_dtype_and_cf_attrs(tmp_path):
    src = _packed_tif(tmp_path, scale=0.01)
    store = str(tmp_path / "packed.zarr")
    src.to_zarr(store)
    root = zarr.open_group(store, mode="r")
    assert root["data"].dtype == np.dtype("int16")          # NOT float64
    assert root["data"].attrs["scale_factor"] == pytest.approx(0.01)
    assert root["data"].attrs["add_offset"] == pytest.approx(0.0)


@pytest.mark.lazy
def test_packed_roundtrip_returns_packed_dataset(tmp_path):
    src = _packed_tif(tmp_path, scale=0.01)
    store = str(tmp_path / "packed.zarr")
    src.to_zarr(store)
    rt = Dataset.from_zarr(store)
    assert rt.scale[0] == pytest.approx(0.01)
    # physical values match (stored 100 -> 1.0)
    np.testing.assert_allclose(rt.read_array(unpack=True), src.read_array(unpack=True))


@pytest.mark.lazy
def test_disagreeing_bands_fall_back_to_float64(tmp_path):
    arr = np.array([[[100, 200]], [[100, 200]]], dtype="int16")
    ds = Dataset.from_array(
        arr,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        path=str(tmp_path / "mixed.tif"),
    )
    ds.scale = [0.01, 0.5]          # bands DISAGREE
    ds.close()
    ds = Dataset.read_file(str(tmp_path / "mixed.tif"))
    store = str(tmp_path / "mixed.zarr")
    ds.to_zarr(store)
    root = zarr.open_group(store, mode="r")
    assert root["data"].dtype == np.dtype("float64")        # fallback
    assert "scale_factor" not in dict(root["data"].attrs)


@pytest.mark.lazy
def test_packed_store_smaller_than_float64(tmp_path):
    import os
    src = _packed_tif(tmp_path, scale=0.01)
    packed = str(tmp_path / "packed.zarr")
    src.to_zarr(packed)
    # a store forced unpacked (disagreeing bands is one way) would be float64;
    # here we just assert the int16 chunk is <= a float64 chunk of the same shape
    root = zarr.open_group(packed, mode="r")
    assert root["data"].dtype.itemsize == 2


@pytest.mark.lazy
def test_xarray_reads_physical_values(tmp_path):
    xr = pytest.importorskip("xarray")
    src = _packed_tif(tmp_path, scale=0.01)
    store = str(tmp_path / "packed.zarr")
    src.to_zarr(store)
    da = xr.open_zarr(store)["data"]
    # mask_and_scale is on by default: xarray returns unpacked floats
    assert float(da.isel(band=0, y=0, x=0)) == pytest.approx(1.0)
```

Add a cube analogue in `tests/dataset/collection/test_zarr.py` once Step 7 lands.

**Docstring / docs edits:**
- `Dataset.to_zarr` docstring (`dataset.py:2159`) — replace the paragraph starting
  "The values written are the physical ones..." with: packed sources whose bands
  agree are stored packed with `scale_factor`/`add_offset`; only mixed/disagreeing
  packing is materialised to float64.
- `_zarr.py` module docstring (lines 21-33) and `write_dataset_to_zarr` docstring
  (lines 521-527) — same correction.
- `docs/reference/zarr.md` — new "### CF packing (scale/offset)" subsection under
  "On-disk layout"; note the agree/disagree rule and the interop consequence.
- `docs/change-log.md` — entry.

**Dependencies:** TASK-0 recommended (clean dtype control), not strict.
**Effort:** M. **Risk:** fill-value semantics flip (physical→stored) on the agreed
path — covered by `test_packed_store_keeps_integer_dtype_and_cf_attrs` +
`test_disagreeing_bands_fall_back_to_float64`; overviews must not double-unpack
(Step 5).

**DoD:**
- [ ] `_agreed_packing` added beside `_agreed_sentinel`; unit-tested for
      agree / disagree / mixed-unpacked / zero-band cases.
- [ ] Agreed path: `root["data"].dtype` is the source integer dtype and carries
      `scale_factor`/`add_offset`; `_FillValue`/`fill_value` are the **stored** sentinel.
- [ ] Disagreeing/mixed path: unchanged float64 behaviour (regression test green).
- [ ] `from_zarr` returns a Dataset with `.scale`/`.offset` set (no eager unpack);
      physical values match the source.
- [ ] Overview levels on the packed path store counts + the same packing attrs; a
      `Dataset.from_zarr(store, level=2)` returns correct physical values.
- [ ] `xr.open_zarr` returns unpacked values (`test_xarray_reads_physical_values`).
- [ ] Cube packed round-trip works (Step 7) or is tracked as an explicit follow-up
      task with its own DoD.
- [ ] All new tests + the full existing Zarr suite pass; docs + changelog updated.

---

### TASK-2 — CF time / calendar coordinate in the cube writer

**Objective.** `DatasetCollection.to_zarr` writes a real CF `time` coordinate
array (values + `units` + `calendar`) so a cube round-trips its calendar and opens
date-aware in xarray. Today it writes only a `time_length` **attribute**.

**Background — the encoder already exists; do not reimplement it.**
- `TimeAxis.resolve(time_coords, length, collection_time, *, warn_stacklevel=5)`
  → `TimeAxis` (`src/pyramids/dataset/_cube_time.py:62`). Precedence: explicit
  `time_coords` → the collection's own `time` → positional `arange(length)`.
- `TimeAxis` has `.values` (1-D array; datetime64 → int64 ns-since-epoch) and
  `.attrs` (`units` = `cf_epoch_units("nanoseconds")` = `"nanoseconds since
  1970-01-01 00:00:00"`, `calendar` = `"proleptic_gregorian"`; or the positional
  `note`). See `_cube_time.py:143-215`.
- The NetCDF writer already calls it: `CubeNetCDFWriter.write` →
  `TimeAxis.resolve(time_coords, collection.time_length, collection.time)` and lays
  it into `coords[time_dim] = (axis.values, axis.attrs)`
  (`netcdf/_cube_netcdf_writer.py:119, 308-310`).
- CF coord attrs helper: `build_coordinate_attrs(dim_name, is_geographic=True) ->
  dict[str, str]` (`netcdf/cf.py:107`) — for `"time"` it stamps `axis="T"`,
  `standard_name`/`long_name`.
- Reader: `decode_cf_time(values, unit, calendar="standard", context=None) ->
  NDArray` (`netcdf/utils.py:1295`) — decodes int64+units back to `datetime64[ns]`
  (or cftime for non-standard calendars); `is_cf_time_units(units)`
  (`netcdf/utils.py:884`).
- The x/y coordinate-writing pattern to mirror is `write_geobox`'s inner `_put`
  (`_geobox_zarr.py:117-130`): `group.create_array(name, shape, dtype,
  dimension_names, overwrite=True)`, assign values, set `_ARRAY_DIMENSIONS`.

**Current writer — `collection.py`, `to_zarr` (anchor `data = self.data`,
~line 1658) and the `Note:` block (~1636-1640):**

```python
        Note:
            Unlike :meth:`to_netcdf`, this writer does not emit a ``time``
            coordinate — only ``time_length`` as an attribute. A collection's
            :attr:`time` (calendar) axis is therefore not carried into the Zarr
            store; use :meth:`to_netcdf` when the calendar axis must round-trip.
```

**Current finalize — `collection.py`, `_finalize_collection_metadata`
(anchor `def _finalize_collection_metadata(`, ~line 280):** writes root attrs
`pyramids_zarr_version`, `time_length`, `pyramids_file_list` and data attrs, then
`finalize_zarr_metadata(...)` (which consolidates). This is where the `time` array
gets written.

**Step 1 — add `time_coords` to the public signature.** Change
`DatasetCollection.to_zarr` (`collection.py:1593`) signature to add, after
`compressor`:

```python
        time_coords: Sequence[Any] | None = None,
```

and delete the `Note:` block above. (`Sequence` is already imported — it's used by
`to_netcdf`.)

**Step 2 — resolve the axis and thread it to finalize.** In `to_zarr`, right after
`files = self._require_files("to_zarr")` and the `import_zarr(...)` guard, resolve:

```python
        from pyramids.dataset._cube_time import TimeAxis

        time_axis = TimeAxis.resolve(
            time_coords, self.time_length, self.time, warn_stacklevel=3
        )
```

(`warn_stacklevel=3` because the chain is `user -> to_zarr -> resolve -> _encode ->
warn`, shallower than the `to_netcdf` default of 5.)

Pass `time_axis` into the finalize calls. Change the `compute=True` branch
(anchor `_finalize_collection_metadata(resolved_store, self._meta, files)`, ~1686):

```python
            _finalize_collection_metadata(resolved_store, self._meta, files, time_axis)
```

and the deferred branch (anchor `dask.delayed(_finalize_after_write)(`, ~1691):

```python
            result = dask.delayed(_finalize_after_write)(
                write_result, resolved_store, self._meta, files, time_axis
            )
```

**Step 3 — write the `time` array in the finalizer.** Change
`_finalize_collection_metadata` (`collection.py:280`) to take `time_axis` and write
the coordinate array **before** `finalize_zarr_metadata` (so the array exists when
metadata is consolidated). Full replacement:

```python
def _finalize_collection_metadata(resolved_store, meta, files: list, time_axis=None) -> None:
    """Write pyramids geobox + time coordinate on a freshly-written cube Zarr.

    Sets `crs_wkt`, `GeoTransform`, `epsg`, `nodata`, `band_names`, `time_length`
    and a pyramids version marker, writes the CF `time` coordinate array
    (`time_axis.values` + units/calendar), then finalizes the geobox + consolidates.
    """
    import zarr

    from pyramids.netcdf.cf import build_coordinate_attrs

    if time_axis is not None:
        root = zarr.open_group(resolved_store, mode="a")
        values = np.asarray(time_axis.values)
        t = root.create_array(
            "time",
            shape=values.shape,
            dtype=values.dtype,
            dimension_names=("time",),
            overwrite=True,
        )
        t[...] = values
        t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        # CF axis attrs (axis="T", standard_name/long_name), then units/calendar
        t.attrs.update(build_coordinate_attrs("time"))
        t.attrs.update(time_axis.attrs)   # units + calendar (or positional `note`)

    finalize_zarr_metadata(
        resolved_store,
        root_attrs={
            "pyramids_zarr_version": ZARR_SCHEMA_VERSION,
            "time_length": int(len(files)),
            "pyramids_file_list": list(files),
        },
        data_attrs={
            "epsg": int(meta.epsg) if meta.epsg is not None else 0,
            "GeoTransform": " ".join(str(v) for v in meta.geotransform),
            "crs_wkt": meta.crs.to_wkt() if meta.crs is not None else "",
            "nodata": [None if v is None else float(v) for v in meta.nodata],
            "band_names": list(meta.band_names) if meta.band_names else [],
            "dtype": str(meta.dtype),
        },
        epsg=int(meta.epsg) if meta.epsg is not None else None,
        geotransform=tuple(float(v) for v in meta.geotransform),
        crs_wkt=meta.crs.to_wkt() if meta.crs is not None else "",
        rows=int(meta.rows),
        cols=int(meta.columns),
        dims=["time", "band", "y", "x"],
    )
```

Update `_finalize_after_write` (`collection.py:434`) to accept and forward
`time_axis`:

```python
def _finalize_after_write(data_result, resolved_store, meta, files, time_axis=None) -> None:
    del data_result
    _finalize_collection_metadata(resolved_store, meta, files, time_axis)
```

**Step 4 — extend the `time` array on append.** In `_append_to_zarr`
(`collection.py:1699`) the resize/write of `data` is mirrored for `time`. After the
`data` resize+write in the `compute=True` branch (anchor `existing.resize((new_total,`,
~1732) and inside `_append_region` (`collection.py:365`), also resize+write the
`time` sub-array. Concretely, resolve a `TimeAxis` for the appended cube in
`to_zarr`'s append branch (it already has `time_coords` from Step 1) and pass its
`.values` down; in the finalizer, extend `root["time"]`:

```python
        time_arr = root["time"] if "time" in root else None
        if time_arr is not None and appended_time_values is not None:
            time_arr.resize((new_total,))
            time_arr[old_t:new_total] = np.asarray(appended_time_values)
```

Wrap in the same rollback try/except that guards the `data` resize
(`collection.py:1736-1738`) so a failed append rolls back both arrays. Thread
`appended_time_values` through `_append_to_zarr`/`_append_region`/
`_finalize_append_metadata` (add a param; keep the positional-index fallback when
the appended cube has no dates).

**Step 5 — read the `time` array back.** In `from_zarr` (`collection.py:2392`),
after the `template.bands.apply_names(...)` line (anchor
`template.bands.apply_names(`, ~2463), read and decode `time`:

```python
        if "time" in root:
            from pyramids.netcdf.utils import decode_cf_time, is_cf_time_units

            time_arr = root["time"]
            t_attrs = dict(time_arr.attrs)
            raw = np.asarray(time_arr[:])
            units = t_attrs.get("units")
            if is_cf_time_units(units):
                decoded = decode_cf_time(
                    raw, units, t_attrs.get("calendar", "standard"), context="cube Zarr time"
                )
                time_values = list(decoded)
            else:
                time_values = list(raw)   # positional index, not a calendar
        else:
            time_values = None
```

and set it on the returned collection before `return`:

```python
        collection = cls(template, time_length, meta=meta, zarr_store=resolved)
        if time_values is not None:
            collection.time = time_values
        return collection
```

(Replaces the bare `return cls(template, time_length, meta=meta, zarr_store=resolved)`
at `collection.py:2468`. The `time` setter validates length == `time_length`,
`collection.py:1126`.)

**On-disk result:** a `time` 1-D array with `_ARRAY_DIMENSIONS=["time"]`, native
`dimension_names=("time",)`, `units="nanoseconds since 1970-01-01 00:00:00"`,
`calendar="proleptic_gregorian"`, `axis="T"`, `standard_name="time"`. Unchanged for
an undated cube except the array holds the positional index with the `note` attr.

**Full tests — add to `tests/dataset/collection/test_zarr.py`:**

```python
@requires_zarr
def test_cube_zarr_roundtrips_datetime_axis(tmp_path, three_files_ramp):
    import numpy as np
    from pyramids.dataset import DatasetCollection

    col = DatasetCollection.from_files(three_files_ramp)
    dates = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[ns]")
    store = str(tmp_path / "cube_time.zarr")
    col.to_zarr(store, time_coords=dates)

    rt = DatasetCollection.from_zarr(store)
    assert rt.time is not None
    assert [np.datetime64(t, "D") for t in rt.time] == [
        np.datetime64("2020-01-01"), np.datetime64("2020-02-01"), np.datetime64("2020-03-01"),
    ]


@requires_zarr
def test_cube_zarr_time_opens_in_xarray(tmp_path, three_files_ramp):
    import numpy as np
    import pytest
    xr = pytest.importorskip("xarray")
    from pyramids.dataset import DatasetCollection

    col = DatasetCollection.from_files(three_files_ramp)
    dates = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[ns]")
    store = str(tmp_path / "cube_time.zarr")
    col.to_zarr(store, time_coords=dates)

    ds = xr.open_zarr(store)
    assert "time" in ds.coords
    assert str(ds["time"].values[0])[:10] == "2020-01-01"


@requires_zarr
def test_cube_zarr_undated_writes_positional_time(tmp_path, three_files_ramp):
    import zarr
    from pyramids.dataset import DatasetCollection

    col = DatasetCollection.from_files(three_files_ramp)
    store = str(tmp_path / "cube_nodate.zarr")
    col.to_zarr(store)                      # no time_coords, collection undated
    root = zarr.open_group(store, mode="r")
    assert "time" in root
    assert root["time"].attrs.get("note", "").startswith("positional")


@requires_zarr
def test_cube_zarr_append_extends_time(tmp_path, three_files_ramp):
    import numpy as np
    from pyramids.dataset import DatasetCollection

    col = DatasetCollection.from_files(three_files_ramp[:2])
    store = str(tmp_path / "cube_append.zarr")
    col.to_zarr(store, time_coords=np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"))

    col2 = DatasetCollection.from_files(three_files_ramp[2:])
    col2.to_zarr(store, mode="a", append_dim="time",
                 time_coords=np.array(["2020-03-01"], dtype="datetime64[ns]"))

    rt = DatasetCollection.from_zarr(store)
    assert rt.time_length == 3
    assert np.datetime64(rt.time[2], "D") == np.datetime64("2020-03-01")
```

(`requires_zarr` and `three_files_ramp` already exist in that test module — see
`from tests._marks import requires_lazy as requires_zarr`.)

**Docs edits:** `docs/reference/zarr.md` cube section — replace the "does not emit a
time coordinate" note with the new `time` coordinate layout + a `time_coords=`
example; `docs/tutorials/lazy/zarr.md`; `docs/change-log.md`.

**Dependencies:** none. **Enables:** TASK-6. **Effort:** S–M (encoder/decoder
exist). **Risk:** low; main care is append keeping `time` and `data` lengths in
lockstep with shared rollback (Step 4).

**DoD:**
- [ ] `to_zarr` writes a `time` array with `units`/`calendar`/`axis="T"`/
      `standard_name="time"`/`_ARRAY_DIMENSIONS=["time"]`.
- [ ] `to_zarr(time_coords=[...])` accepts explicit datetimes; an undated cube
      writes the positional index with the `note` attr (no crash).
- [ ] `from_zarr` recovers `collection.time` (datetimes for a CF axis, raw for a
      positional one).
- [ ] Append extends `time` in lockstep with `data`; a forced failure rolls back
      both (add a fault-injection test or assert shapes after a simulated error).
- [ ] `xr.open_zarr` shows a decoded `time` coordinate.
- [ ] The `Note:` block is gone from code **and** `docs/reference/zarr.md`.
- [ ] All four new tests + the existing cube suite pass; docs + changelog updated.

---

### TASK-3 — Zarr v2 **and** v3 output (`zarr_format=`)

**Objective.** `Dataset.to_zarr(..., zarr_format=2|3)` and
`DatasetCollection.to_zarr(..., zarr_format=2|3)`. Default (`None`) preserves
today's behaviour (whatever the installed zarr writes by default — v3 on zarr≥3).

**Background — pyramids has NO zarr-version detection today** (confirmed by grep:
the only `..._version` symbol is `ZARR_SCHEMA_VERSION="2"`, a *pyramids layout*
marker, unrelated to the zarr **format**). Every create call assumes the v3 API
(`create_array(..., dimension_names=…, compressors=…)`, `FsspecStore.from_url`).
Crucially, pyramids **already** writes the v2-compat `_ARRAY_DIMENSIONS` attr next
to the v3 `dimension_names` kwarg (`_geobox_zarr.py:127-129`, `write_geobox`), so a
v2 store just needs the `dimension_names=` kwarg **dropped** — the dims are still
recoverable from the attr.

How xarray handles the split (reference, `xarray/backends/zarr.py`): `_zarr_v3()`
(`:109`) detects the library major version; `_handle_zarr_version_or_format`
(`:2005`) reconciles the args; v3 uses `dimension_names` + `compressors`/`filters`/
`serializer`, v2 uses `_ARRAY_DIMENSIONS` attr + `compressor` (singular) +
`filters`.

**Step 1 — version detection.** In `_zarr.py`, extend `_require_zarr` (`:77`) or add
a helper:

```python
def _zarr_major() -> int:
    """Installed zarr-python major version (3 for zarr>=3)."""
    zarr = _require_zarr()
    return int(str(zarr.__version__).split(".", 1)[0])
```

**Step 2 — format-aware codec kwargs.** Change `normalize_compressors`
(`_geobox_zarr.py:309`) to take the target format and emit the right kwarg name.
Current:

```python
def normalize_compressors(compressor: Any) -> dict[str, Any]:
    if compressor == "auto":
        return {}
    if compressor is None:
        return {"compressors": None}
    if isinstance(compressor, (list, tuple)):
        return {"compressors": list(compressor)}
    return {"compressors": [compressor]}
```

Replacement:

```python
def normalize_compressors(compressor: Any, *, zarr_format: int | None = None) -> dict[str, Any]:
    """Map a user ``compressor=`` argument to zarr ``create_array`` kwargs.

    v3 expects an iterable ``compressors=``; v2 expects a single ``compressor=``.
    ``zarr_format=None`` keeps the v3 spelling (the default on zarr>=3).
    """
    key = "compressor" if zarr_format == 2 else "compressors"
    if compressor == "auto":
        return {}
    if compressor is None:
        return {key: None}
    if isinstance(compressor, (list, tuple)):
        codecs = list(compressor)
        return {key: (codecs[0] if zarr_format == 2 else codecs)}
    return {key: (compressor if zarr_format == 2 else [compressor])}
```

**Step 3 — drop `dimension_names=` on v2 in every create call.** There are two:
`write_geobox._put` (`_geobox_zarr.py:119`) and `_write_overview_levels`
(`_zarr.py:691`). Both must become format-aware. Pass `zarr_format` down into
`write_geobox` / `finalize_zarr_metadata` (add the param, default `None`) and guard:

```python
        create_kwargs = dict(shape=values.shape, dtype=values.dtype, overwrite=True)
        if zarr_format != 2:
            create_kwargs["dimension_names"] = tuple(var_dims)
        arr = group.create_array(name, **create_kwargs)
        arr[...] = values
        arr.attrs["_ARRAY_DIMENSIONS"] = list(var_dims)   # ALWAYS written (v2 + v3)
```

The TASK-0 `_create_and_store_data` already threads `zarr_format` into its
`create_array`; make it drop `dimension_names` when `zarr_format == 2` the same way.

**Step 4 — thread `zarr_format` through the public API and finalizers.**
- `Dataset.to_zarr` (`dataset.py:2147`) and `write_dataset_to_zarr` (`_zarr.py:501`)
  gain `zarr_format: int | None = None`; pass to `_create_and_store_data`,
  `normalize_compressors(compressor, zarr_format=zarr_format)`, and
  `_finalize_metadata`/`finalize_zarr_metadata` (which pass it to `write_geobox`).
- `DatasetCollection.to_zarr` (`collection.py:1593`) + the module finalizers
  (`_finalize_collection_metadata`, `_finalize_after_write`) gain the same param.
- `open_group(...)` calls (`_geobox_zarr.py:246`, `collection.py`) and
  `consolidate_metadata` take `zarr_format=` only where the installed zarr requires
  it — on zarr≥3 `open_group(mode="a")` auto-detects an existing store's format, so
  pass `zarr_format` only on the initial create (`_create_and_store_data`), not on
  the reopen-to-finalize. Verify against the pinned zarr version.

**Step 5 — `FsspecStore`.** `_resolve_store` (`_zarr.py:955`) uses
`FsspecStore.from_url` (v3-only). For a v2 target on a cloud URL, fall back to the
zarr-v2 fsspec mapper. Since the default and the common case are v3, gate this: if
`_zarr_major() < 3` or `zarr_format == 2` with a URL + `storage_options`, build the
store via `fsspec.get_mapper(url, **storage_options)` instead. Document that v2
cloud writes need the fsspec mapper path.

**On-disk result:** `zarr_format=2` → `.zgroup`/`.zarray`/`.zattrs` files,
`_ARRAY_DIMENSIONS` attr, single `compressor`. `zarr_format=3` → `zarr.json`,
native `dimension_names`, `compressors` list. `from_zarr` reads both (it already
reads `_ARRAY_DIMENSIONS` and auto-detects via `read_geobox`).

**Tests — parametrize the core round-trips.** In `tests/dataset/ops/test_zarr.py`
add:

```python
@pytest.mark.lazy
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_roundtrip_zarr_format(small_dataset, tmp_path, zarr_format):
    store = str(tmp_path / f"fmt{zarr_format}.zarr")
    small_dataset.to_zarr(store, zarr_format=zarr_format)
    reloaded = Dataset.from_zarr(store)
    np.testing.assert_array_equal(
        np.atleast_3d(small_dataset.read_array()).squeeze(),
        np.atleast_3d(reloaded.read_array()).squeeze(),
    )
    # on-disk marker
    import os
    if zarr_format == 2:
        assert os.path.exists(os.path.join(store, ".zgroup"))
    else:
        assert os.path.exists(os.path.join(store, "zarr.json"))
```

Mirror one case in `tests/dataset/collection/test_zarr.py`.

**Docs:** `docs/reference/zarr.md` "Codec / compression control" section documents
`zarr_format`; note the v2 cloud caveat.

**Dependencies:** TASK-0 (single create call site). **Effort:** M. **Risk:** the
zarr-v2/v3 API differences in `open_group`/`consolidate_metadata`/`FsspecStore` —
verify each against the pinned zarr; keep v3 the untouched default.

**DoD:**
- [ ] `zarr_format=2` yields a v2 store (`.zgroup` present), `zarr_format=3` a v3
      store (`zarr.json` present), `None` unchanged from today.
- [ ] Both formats round-trip through `from_zarr` and open in `xr.open_zarr`.
- [ ] Codec kwarg is correct per format (v2 `compressor`, v3 `compressors`).
- [ ] Parametrized `[2, 3]` tests added for Dataset and cube; existing tests green.
- [ ] Docs + changelog updated.

---

### TASK-4 — Per-array `encoding` / `filters`

**Objective.** Expose the codec **pipeline** (filters + serializer, explicit dtype)
that pyramids uses internally, not just a single `compressor`.

**Background.** pyramids has one `data` array, so "encoding" is **per-store**, far
simpler than xarray's per-variable dict. The single funnel is
`normalize_compressors` (`_geobox_zarr.py:309`), today producing only
`compressors=`. xarray's whitelist (`extract_zarr_variable_encoding`,
`xarray/backends/zarr.py:496`) is the reference set of valid keys: `{chunks, shards,
compressor, compressors, filters, serializer, write_empty_chunks,
chunk_key_encoding, fill_value}`.

**Step 1 — accept an `encoding` mapping.** Add `encoding: dict | None = None` to
`Dataset.to_zarr` / `write_dataset_to_zarr` and `DatasetCollection.to_zarr`. Build a
validated kwargs dict:

```python
_VALID_ENCODING_KEYS_V3 = {"chunks", "compressors", "filters", "serializer", "write_empty_chunks"}
_VALID_ENCODING_KEYS_V2 = {"chunks", "compressor", "filters", "write_empty_chunks"}

def _encoding_kwargs(encoding: dict | None, *, zarr_format: int | None) -> dict[str, Any]:
    """Validate a user `encoding` mapping and return zarr create kwargs.

    Raises ValueError on an unknown key (matching xarray, which refuses unknown
    user-supplied encoding keys rather than silently dropping them).
    """
    if not encoding:
        return {}
    valid = _VALID_ENCODING_KEYS_V2 if zarr_format == 2 else _VALID_ENCODING_KEYS_V3
    unknown = set(encoding) - valid
    if unknown:
        raise ValueError(
            f"unsupported zarr encoding keys {sorted(unknown)}; "
            f"valid keys for zarr_format={zarr_format or 3}: {sorted(valid)}"
        )
    return dict(encoding)
```

**Step 2 — merge with the compressor path.** In `write_dataset_to_zarr`, merge
`_encoding_kwargs(encoding, zarr_format=zarr_format)` into `codec_kwargs` (encoding
wins on conflict; a `filters=` there rides alongside the `compressors=` from
`normalize_compressors`). Pass the merged dict into `_create_and_store_data`
(TASK-0). Because TASK-0 owns the create call, `filters=`/`serializer=` reach
`create_array` directly.

**Step 3 — `dtype` override** belongs with TASK-1 (it changes the on-disk cast); if
requested here, apply it as `arr = arr.astype(encoding["dtype"])` before the store
and record it in `_metadata_dict`'s `dtype`. Keep out of scope unless needed.

**Tests — `tests/dataset/ops/test_zarr.py`:**

```python
@pytest.mark.lazy
def test_encoding_filters_roundtrip(small_dataset, tmp_path):
    from numcodecs import Delta                     # or the zarr-v3 codec equivalent
    store = str(tmp_path / "filters.zarr")
    small_dataset.to_zarr(store, encoding={"filters": [Delta(dtype="f4")]})
    root = zarr.open_group(store, mode="r")
    # filter chain recorded in array metadata
    assert root["data"].metadata is not None
    reloaded = Dataset.from_zarr(store)
    np.testing.assert_array_equal(
        np.atleast_3d(small_dataset.read_array()).squeeze(),
        np.atleast_3d(reloaded.read_array()).squeeze(),
    )


@pytest.mark.lazy
def test_encoding_rejects_unknown_key(small_dataset, tmp_path):
    with pytest.raises(ValueError, match="unsupported zarr encoding"):
        small_dataset.to_zarr(str(tmp_path / "bad.zarr"), encoding={"nope": 1})
```

(Use whichever codec the pinned zarr-v3 exposes; `numcodecs` Delta/Shuffle are the
usual filters. Confirm the exact import against the installed version.)

**Docs:** extend the "Codec / compression control" section with an `encoding=`
example (delta + blosc chain).

**Dependencies:** TASK-0, TASK-3 (shares create plumbing and the format split).
**Effort:** S–M. **Risk:** low.

**DoD:**
- [ ] `to_zarr(encoding={"filters": [...]})` lands the filter chain in the `data`
      array metadata; values round-trip exactly.
- [ ] An unknown encoding key raises `ValueError` (not silently dropped).
- [ ] Works for `zarr_format` 2 and 3 (correct key set per format).
- [ ] Tests added; docs + changelog updated.

---

### TASK-5 — Operational knobs: `write_empty_chunks`, `safe_chunks`

**Objective.** User control over the two write-safety knobs xarray exposes.

**Background.** `write_empty_chunks` (skip / force writing all-fill chunks) is a
plain create kwarg (v3 nests it under `config=`, `xarray/backends/zarr.py:1191`).
`safe_chunks` guards region/append writes where several dask chunks map onto one
zarr chunk (silent corruption risk); xarray validates via
`validate_grid_chunks_alignment` (`xarray/backends/chunks.py:184`). pyramids already
forces **size-1 time chunks** for append (`collection.py:1703-1704` comment), so its
append is aligned by construction; the value here is a user-facing escape + an
explicit check for **region** writes.

**Step 1 — `write_empty_chunks` passthrough.** Add `write_empty_chunks: bool | None
= None` to `to_zarr`. In TASK-0's `_create_and_store_data`, when not `None`:
- v3: `create_kwargs["config"] = {"write_empty_chunks": write_empty_chunks}`
- v2: `create_kwargs["write_empty_chunks"] = write_empty_chunks`
(Confirm the exact spelling against the pinned zarr; xarray relocates it into
`config` for v3.)

**Step 2 — `safe_chunks` for region/append.** Add `safe_chunks: bool = True` to
`DatasetCollection.to_zarr`. Before a region write (`collection.py:1664` branch) and
before `_append_to_zarr`'s region write, add a lightweight check mirroring xarray's
interior-chunk rule: each incoming dask chunk on the write axis must be a whole
multiple of the store array's chunk size on that axis (except a permitted partial
last chunk). Implement as a small module helper:

```python
def _assert_region_chunks_aligned(data, target_chunks, region_slices, *, safe_chunks: bool) -> None:
    """Raise when `data`'s chunks would straddle `target_chunks` in a region write.

    Mirrors xarray's safe_chunks guard: concurrent writers to the same zarr chunk
    race, so each dask chunk must align to the store's chunk grid. `safe_chunks=False`
    skips the check (caller accepts the corruption risk).
    """
    if not safe_chunks:
        return
    for axis, sl in enumerate(region_slices):
        if sl == slice(None):
            continue
        store_chunk = target_chunks[axis]
        for block in data.chunks[axis][:-1]:      # interior blocks must be exact multiples
            if block % store_chunk:
                raise ValueError(
                    f"region write on axis {axis}: dask chunk {block} is not a multiple "
                    f"of the store chunk {store_chunk}; pass safe_chunks=False to override "
                    f"(risk of corrupted data on concurrent writes)."
                )
```

Call it from the region and append branches with the target array's `.chunks`.

**Tests — `tests/dataset/collection/test_zarr.py`:**

```python
@requires_zarr
def test_write_empty_chunks_passthrough(tmp_path, three_files_ramp):
    from pyramids.dataset import DatasetCollection
    col = DatasetCollection.from_files(three_files_ramp)
    store = str(tmp_path / "we.zarr")
    col.to_zarr(store, write_empty_chunks=True)     # must not raise

@requires_zarr
def test_safe_chunks_rejects_misaligned_region(tmp_path, three_files_ramp):
    import pytest
    from pyramids.dataset import DatasetCollection
    # build a store, then attempt a region write whose dask chunks straddle the grid
    ...
    with pytest.raises(ValueError, match="safe_chunks=False"):
        col.to_zarr(store, mode="a", region={"time": slice(0, 2)}, safe_chunks=True)
    col.to_zarr(store, mode="a", region={"time": slice(0, 2)}, safe_chunks=False)  # ok
```

**Docs:** `docs/reference/zarr.md` incremental-writes section.

**Dependencies:** TASK-0 (create kwarg for `write_empty_chunks`). **Effort:** S.
**Risk:** low.

**DoD:**
- [ ] `write_empty_chunks=True/False` reaches the create call for both formats.
- [ ] A misaligned region/append raises with `safe_chunks=True`, proceeds with
      `safe_chunks=False`.
- [ ] Tests added; docs + changelog updated.

---

### TASK-6 — `region="auto"` for the cube

**Objective.** `DatasetCollection.to_zarr(mode="a", region="auto")` resolves the
target `time` slice from the collection's datetime labels, matching xarray.

**Background.** Needs a `time` coordinate in the store — **depends on TASK-2**.
xarray's algorithm (`_auto_detect_regions`, `xarray/backends/zarr.py:1366`): read
the store's existing coord, build a `pd.Index`, `index.get_indexer(new_labels)`,
require every label found and the indices **contiguous** (`np.diff == 1`), then
build a `slice`. pyramids already turns a `{"time": slice}` dict into positional
slices via `_region_to_slices` (`collection.py:417`) — this task only computes that
slice from labels.

**Step 1 — accept the sentinel.** In `DatasetCollection.to_zarr`, change `region:
dict | None` to `region: dict | str | None` and handle `"auto"` in the region
branch (`collection.py:1664`):

```python
        if region is not None:
            import zarr
            existing_group = zarr.open_group(resolved_store, mode="a")
            if region == "auto":
                region = {"time": _auto_time_region(existing_group, self.time)}
            existing = existing_group["data"]
            _assert_region_chunks_aligned(data, existing.chunks,
                                          _region_to_slices(region, data.ndim),
                                          safe_chunks=safe_chunks)   # TASK-5
            return da.store(data, existing,
                            regions=[_region_to_slices(region, data.ndim)],
                            compute=compute)
```

**Step 2 — the resolver.**

```python
def _auto_time_region(group, collection_time) -> slice:
    """Resolve `region='auto'` to a `time` slice from datetime labels.

    Reads the store's `time` coordinate, decodes it, and maps this collection's own
    `time` values onto contiguous integer positions — the xarray `_auto_detect_regions`
    algorithm, restricted to the cube's single `time` axis.
    """
    import pandas as pd
    from pyramids.netcdf.utils import decode_cf_time, is_cf_time_units

    if "time" not in group:
        raise KeyError("region='auto' needs a 'time' coordinate in the store; "
                       "write it with to_zarr(time_coords=...) (TASK-2).")
    if collection_time is None:
        raise ValueError("region='auto' needs this collection to carry a time axis.")
    t = group["time"]
    attrs = dict(t.attrs)
    raw = np.asarray(t[:])
    units = attrs.get("units")
    store_labels = decode_cf_time(raw, units, attrs.get("calendar", "standard")) \
        if is_cf_time_units(units) else raw
    index = pd.Index(np.asarray(store_labels))
    idxs = index.get_indexer(pd.Index(np.asarray(collection_time)))
    if (idxs == -1).any():
        missing = [l for l, i in zip(collection_time, idxs) if i == -1]
        raise KeyError(f"region='auto': times not found in store: {missing}")
    if len(idxs) > 1 and not np.all(np.diff(idxs) == 1):
        raise ValueError(f"region='auto': target times are not contiguous: {idxs.tolist()}")
    return slice(int(idxs[0]), int(idxs[-1]) + 1)
```

**Tests — `tests/dataset/collection/test_zarr.py`:**

```python
@requires_zarr
def test_region_auto_targets_correct_slice(tmp_path, three_files_ramp):
    import numpy as np
    from pyramids.dataset import DatasetCollection
    dates = np.array(["2020-01-01","2020-02-01","2020-03-01"], dtype="datetime64[ns]")
    col = DatasetCollection.from_files(three_files_ramp)
    store = str(tmp_path / "auto.zarr")
    col.to_zarr(store, time_coords=dates)

    fix = DatasetCollection.from_files(three_files_ramp[1:2])
    fix.time = [np.datetime64("2020-02-01")]
    fix.to_zarr(store, mode="a", region="auto")           # must target slice(1, 2)
    # assert only that timestep changed (compare arrays)

@requires_zarr
def test_region_auto_rejects_unknown_and_noncontiguous(tmp_path, three_files_ramp):
    import numpy as np, pytest
    from pyramids.dataset import DatasetCollection
    ...
    with pytest.raises(KeyError):
        bad.to_zarr(store, mode="a", region="auto")       # date not in store
```

**Docs:** `docs/reference/zarr.md` incremental-writes section — a `region="auto"`
example alongside the explicit-slice one.

**Dependencies:** TASK-2 (needs the `time` coord), TASK-5 (`_assert_region_chunks_aligned`,
optional). **Effort:** S. **Risk:** low; purely additive.

**DoD:**
- [ ] `region="auto"` maps datetimes to the correct integer slice and writes there.
- [ ] Unknown or non-contiguous dates raise (KeyError / ValueError, matching xarray).
- [ ] Tests added; docs + changelog updated.

---

### TASK-7 — Multi-array / sibling variables *(structural; scope-gated — build only on real demand)*

**Objective (if pursued).** Let one store hold ≥2 named data arrays (e.g.
`elevation` + `slope`) sharing one geobox, each with its **own** dtype / `_FillValue`
/ `scale_factor` — which also dissolves the single-array shared-sentinel and
shared-packing constraints that TASK-1 and `_agreed_sentinel` work around.

**What already exists (read side is ready).** `detect_data_var`
(`_geobox_zarr.py:481`) already picks a primary array among many using CF roles
(`grid_mapping`, `bounds`, `coordinates`, `cell_measures`), and
`read_dataset_from_zarr` accepts `data_name=` (`_zarr.py:864`). So a multi-array
store **reads** today (one variable at a time).

**What's hard-coded to a single `"data"` array (write side).**
- `finalize_zarr_metadata` opens `root["data"]` explicitly (`_geobox_zarr.py:248`).
- `write_geobox(data_name="data")` (`_geobox_zarr.py:251`).
- `write_dataset_to_zarr` writes `component="data"` / creates `"data"`
  (`_zarr.py:611` pre-TASK-0, or `_create_and_store_data(name="data")` post-TASK-0).
- `DatasetCollection` cube writer/reader all address `root["data"]`.

**Design sketch (new entry point, does not change existing single-array API).**
Add `write_datasets_to_zarr(mapping: dict[str, Dataset], store, *, shared_geobox=True,
zarr_format=None, ...)` in a new `ops/_zarr_multi.py`:
1. Validate all datasets share one geobox (CRS + geotransform + shape) when
   `shared_geobox=True`; else write per-array geoboxes.
2. Create the group; for each `name -> ds`, create an array `name` via the TASK-0
   `_create_and_store_data(name=name, ...)`, each with its **own** `_metadata_dict(ds)`
   (own `_FillValue`, own `scale_factor`/`add_offset` from TASK-1).
3. Write **one** shared `spatial_ref` + `x`/`y` via `write_geobox` (skip the
   `data_name="data"` assumption — pass each array name, or write the grid mapping
   once and set `grid_mapping="spatial_ref"` on every array).
4. `band_names` become per-array; a single-band variable is a 2-D array.
Reading: `Dataset.from_zarr(store, data_name="elevation")` already works; optionally
add `read_datasets_from_zarr(store) -> dict[str, Dataset]` iterating
`group.array_keys()` minus coords/CF-referenced (reuse `_cf_referenced_names`,
`_NON_DATA_ARRAYS`).

**Dependencies:** TASK-1 (per-array packing), TASK-3 (format). **Effort:** L.
**Status:** **Deferred.** Genuine model divergence — pyramids is a raster/cube
library, not a general `Dataset` serializer, and TASK-1 already removes the painful
float64-bloat symptom without it. Build only if multi-variable stores are a real
requirement.

**DoD (if pursued):**
- [ ] A store holds ≥2 named arrays sharing one `spatial_ref`/`x`/`y`; each carries
      its own `_FillValue` (+ `scale_factor`/`add_offset` where packed).
- [ ] `Dataset.from_zarr(store, data_name=name)` reads each back; optional
      `read_datasets_from_zarr` returns the mapping.
- [ ] `xr.open_zarr` sees all variables with correct dims + shared CRS.
- [ ] The existing single-`data` API and its on-disk layout are unchanged.
- [ ] Tests (`tests/dataset/ops/test_zarr_multivar.py`) + docs + changelog.

---

### TASK-8 — Hierarchical groups / DataTree-style nesting *(stretch)*

**Objective (if pursued).** One store holding several cubes/scenes under sub-groups,
analogous to `DataTree.to_zarr` (`xarray/backends/writers.py:928`, each node → one
zarr group by relative path via `ZarrStore.get_child_store`,
`xarray/backends/zarr.py:845`).

**Approach.** A thin wrapper over the TASK-7 multi-array writer that writes each
named dataset/cube into `root.require_group(path)` instead of the root, then
consolidates once at the top level. Read via a `open_datatree`-style walker, or lean
on `xr.open_datatree(engine="zarr")` for interop.

**Dependencies:** TASK-7 (shares the multi-container model). **Effort:** L.
**Status:** **Deferred / stretch** — no current pyramids concept maps to a DataTree;
pursue only on concrete demand.

**DoD (if pursued):**
- [ ] Multiple named datasets/cubes write into sub-groups of one store and round-trip.
- [ ] `xr.open_datatree(store, engine="zarr")` reads the hierarchy with CRS/dims.
- [ ] Tests + docs + changelog.

---

### TASK-9 — Lazy `open_zarr`-style read for a single `Dataset` *(stretch)*

**Objective (if pursued).** An optional lazy read path for a single `Dataset` so
values aren't materialised to GDAL/NumPy on open (gap §4.7).

**Constraint.** pyramids `Dataset`s are **GDAL-backed**, so a truly lazy `Dataset`
isn't achievable without a backing-store rework. The realistic surface is: return
the lazy `dask.array` (already available via `_read_data_array` with `chunks`,
`_zarr.py:786`) or an `xarray.DataArray`, **not** a lazy `Dataset`. The cube reader
is *already* lazy (`collection.py:1493-1496`, `da.from_zarr`), so this only concerns
the single-`Dataset` path.

**Approach.** Add `Dataset.from_zarr(store, ..., lazy=False)`; when `lazy=True`,
return a documented lazy view (dask array or `DataArray`) rather than constructing a
GDAL-backed `Dataset`. Be explicit in the docstring about what "lazy" means here (no
false promise of a lazy `Dataset`).

**Dependencies:** none. **Effort:** L (architectural). **Status:** **Deferred /
stretch.**

**DoD (if pursued):**
- [ ] A documented lazy read returns without materialising; a follow-up compute
      equals the eager read.
- [ ] Docstring states the GDAL-backing caveat plainly.
- [ ] Tests + docs + changelog.

---

### 10.10 Task dependency graph & recommended order

```
TASK-0 (own the create call)
 ├─> TASK-1 (packing) ───────────────> TASK-7 (multi-array) ──> TASK-8 (groups)
 ├─> TASK-3 (zarr_format) ─┬─> TASK-4 (encoding/filters)
 │                         └─> TASK-7
 ├─> TASK-4
 └─> TASK-5 (safe/empty chunks)
TASK-2 (time coord) ───────────────────> TASK-6 (region="auto")
TASK-9 (lazy Dataset read)  [independent, stretch]
```

**Recommended shippable order:** TASK-0 → TASK-2 → TASK-1 → TASK-3 → TASK-4 →
TASK-5 → TASK-6, then reassess TASK-7/8/9 against real demand.

**Invariants every task must preserve (regression fence):**
- The default on-disk layout for an **unpacked, single-timestep, v3-default,
  no-`time_coords`, no-`encoding`** store is byte-compatible with today's output, so
  stores written before these changes keep reading and existing readers are
  unaffected. (TASK-2 adds a `time` array to *cube* stores — that is an additive
  change a CF/GeoZarr reader ignores if it doesn't want it; call it out in the
  changelog.)
- `Dataset.from_zarr` / `DatasetCollection.from_zarr` keep reading every store the
  current code reads, including legacy flat-attr and foreign GeoZarr stores
  (`read_geobox`'s tolerance, `_geobox_zarr.py:597`).
- The `[lazy]`-extra gate (`import_zarr` / `lazy_extra_hint`) stays on every public
  entry point; no hard `import zarr` at module top level.

### 10.11 How to execute one task (checklist for the implementing agent)

1. `git checkout -b feat/zarr-<task-slug>` off the branch this plan lives on.
2. Re-`grep` every anchor string quoted in the task (line numbers drift between
   tasks); confirm the current code matches the "Current code" excerpt before editing.
3. Make the code change exactly as specified; keep each task's change minimal and
   self-contained (one task = one PR).
4. Add the task's test file / test functions verbatim (adjust only import paths /
   fixture names if the repo has moved them); run
   `pixi run pytest <the named test files> -q`.
5. Run the full Zarr suite (`tests/dataset/ops/test_zarr*.py`,
   `tests/dataset/collection/test_zarr.py`, `tests/netcdf/samples/test_zarr.py`) to
   catch regressions.
6. `pixi run pre-commit run --files <changed files>`; fix lint/format/type findings.
7. Update `docs/reference/zarr.md`, the relevant tutorial, and `docs/change-log.md`
   as the task's docs section lists.
8. Tick every DoD box; a task is not done until all are literally checkable.
9. Open the PR; title `feat(zarr): <objective>`; body links this plan section.
