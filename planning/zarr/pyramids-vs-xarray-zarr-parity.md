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
