# Migration guide

This guide helps downstream packages that depend on `pyramids` migrate across breaking changes. It is
organized by **subpackage**, and under each subpackage by the **release that introduced the changes**.

`isinstance(x, NetCDF)` keeps working throughout, so most code needs no change. Everything below either emits a
`DeprecationWarning` (discoverable with the command in the next section) or is a hard behavior change (called
out explicitly).

## Finding what affects you (one command)

Run your own test suite against the new `pyramids` with deprecation warnings turned into errors — this surfaces
every deprecated call site:

```bash
python -W error::DeprecationWarning -m pytest
```

Or, in code under test:

```python
import warnings
warnings.simplefilter("error", DeprecationWarning)
```

This catches the deprecations; the hard behavior changes do **not** warn — search for them manually.

## base

### unreleased

**`abfs://` now means Azure Data Lake Storage Gen2, not Blob.** It mapped to GDAL's `/vsiaz/` (Blob) and now maps
to `/vsiadls/` (Gen2), matching what `abfs` means everywhere else in the Azure and Hadoop ecosystems — it is the
Azure Blob *File System* driver, which is Gen2.

- If you used `abfs://` against a **Gen2** account, this is the fix: you were being routed to the Blob handler and
  now get the right one, with directory semantics.
- If you used `abfs://` against a **flat Blob** account, switch to `az://`, which is unchanged and still names
  Blob. Credentials are the same `AZURE_STORAGE_*` family either way.
- `abfss://` is accepted too, if you prefer to be explicit about TLS. There is no `adls://` scheme: the
  registered Azure Data Lake name is `adl://` (Gen1), which GDAL does not handle.
- The canonical `abfss://<filesystem>@<account>.dfs.core.windows.net/<path>` form is understood: the filesystem
  becomes the container and the account is dropped from the path, because GDAL takes it from configuration. The
  rewrite is deterministic — the same URL always yields the same `/vsi*` path — and if the URL names an account
  that disagrees with the configured one, a `UserWarning` says which account will actually be read. Set
  `AZURE_STORAGE_ACCOUNT` (or `AccountName=` inside `AZURE_STORAGE_CONNECTION_STRING`) to match, or use the bare
  `abfs://<filesystem>/<path>` form.

**`/vsiadls/` is now recognised as a remote, network-backed path.** Previously `is_remote()` and
`is_network_backed()` both returned `False` for a Gen2 path, so it was classified as a local file — opening still
worked, but anything reasoning about the path (credential handling, archive chaining) got the wrong answer.

**Archive chaining now covers every network handler, including for raw `/vsi*` paths.** A raster inside a zip
or tar is rewritten to `/vsizip/<handler>/...` for `/vsiadls/`, `/vsioss/` (Alibaba), `/vsiswift/` (OpenStack),
`/vsihdfs/` and `/vsiwebhdfs/`, not only for S3, Google Cloud, Azure Blob and `/vsicurl/`. Four of those have no
URL scheme at all, so the raw `/vsi*` spelling is the only way to name them — paths already in `/vsi*` form now
go through the chaining too, where before they were returned untouched. If you were chaining by hand, the manual
prefix is no longer needed, and a hand-built `/vsizip//vsioss/...` still works unchanged.

**Credentials are unchanged.** `/vsiadls/` reads the same `AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_ACCESS_KEY` /
`AZURE_STORAGE_SAS_TOKEN` / `AZURE_STORAGE_CONNECTION_STRING` family as `/vsiaz/`, and honours
`AZURE_NO_SIGN_REQUEST` identically (verified: with it set both handlers skip the instance-metadata credential
lookup). `CloudConfig` gains an `azure_no_sign_request=True` flag so anonymous Azure reads have the same knob the
S3 side already had.

### 0.47.0

**`Dataset.epsg` no longer reports EPSG:4326 for a raster that has no CRS.** It previously substituted WGS 84
whenever the projection was empty, so an ungeoreferenced grid claimed a georeference it did not have — and that
claim propagated into `to_file`, `to_crs`, `bounds` and alignment checks. It now returns `None`, matching GDAL,
rasterio and geopandas, none of which substitute a default.

What changed, and what did not:

- **No CRS and no evidence** (a GeoTIFF or ASCII grid with an empty projection) → `epsg` is `None`. Operations
  that genuinely need a CRS raise `CRSError` naming the fix instead of proceeding.
- **A CRS with no EPSG authority** (geostationary, spherical-earth GRIB) → unchanged by this release. Note that
  `epsg` is *not* uniformly `None` here: `NetCDF` reports `None` for a geostationary grid (see #706), while a
  spherical-earth GRIB still resolves through `epsg_from_wkt`'s `4326` default. Read `crs` when you need the
  authoritative answer for such a grid.
- **A CF NetCDF with `degrees_east` / `degrees_north` axes and no `grid_mapping`** → still `4326`. CF leaves the
  datum implicit for these and the whole ecosystem reads them as WGS 84, so this is a reading of the file's
  metadata rather than an assumption. Three things narrow it:
    - Only a *coordinate* variable's units are evidence — one that declares `axis`, is named in a `coordinates`
      attribute (a curvilinear grid's 2-D lat/lon), or carries a conventional coordinate name. A data variable in
      `degrees_east` (a wind direction, a solar angle) is not evidence.
    - A **horizontal** axis carrying a linear unit (`m`, `km`) or plain `degrees`/`rad` vetoes it, so a projected,
      rotated-pole or geostationary grid is not relabelled on the strength of auxiliary lat/lon arrays. A
      **vertical** axis does not veto — its units say nothing about the horizontal frame. Vertical axes are
      recognised from CF's `axis: Z`, falling back to conventional names (`depth`, `lev`, …) only for a file that
      declares no axes at all.
    - The grid's own extent must fit in the lon/lat range. A metre-scale grid is not lat/lon whatever its
      metadata says.

**A CRS with no EPSG code now reports `epsg is None` too.** Previously any projection the EPSG register does
not name — an orthographic or geostationary projection, a rotated pole, a spherical-earth GRIB `GEOGCS` — fell
back to `4326`. That claimed WGS 84 for grids that are not WGS 84: an orthographic frame is not lat/lon at all,
and a spherical datum differs from the WGS 84 ellipsoid by up to ~20 km.

The CRS itself is unaffected — `ds.crs` still returns the WKT, `crs_spec()` falls back to it, and reprojection,
cropping and alignment all keep working. Only the *code* is absent, because there is not one. If your code reads
`ds.epsg` on such a grid, read `ds.crs` (or `crs_spec(ds.epsg, ds.crs)`) instead.

If you relied on the old default, set the CRS explicitly — `dataset.epsg = <code>` in process, or
`gdal_edit.py -a_srs EPSG:<code> <file>` on disk. To find affected code, look for `.epsg` used without a `None`
check, and for the `dataset.epsg or dataset.crs` idiom, which is now `crs_spec(dataset.epsg, dataset.crs)`.

### 0.46.1

**`import pyramids` no longer configures logging.** Previously, importing the package ran `Config()`, which
added a handler to the **root** logger, called `root.setLevel(logging.INFO)`, pinned six third-party loggers to
`WARNING`, and printed `Logging is configured.` to stderr. A library reconfiguring the root logger silently
takes over logging for the whole host application, so that is gone.

| Change | Kind | What you do |
|--------|------|-------------|
| Import no longer installs a console handler | behavior | Configure logging yourself, or ask pyramids for it |
| Root logger untouched (no handler, no `setLevel`) | behavior | Nothing — this is the fix |
| `Config` / `LoggerManager` default `level=INFO` → `None` | behavior | Pass a level to get console output |
| `Logging is configured.` announce moved from INFO to DEBUG | behavior | Nothing |
| Third-party loggers only quieted on request | behavior | Pass `quiet_third_party=True` to restore |

**If you saw pyramids log lines and want them back**, pick one:

```python
# Option A — you own logging (recommended for applications)
import logging

logging.basicConfig(level=logging.INFO)
import pyramids  # pyramids records propagate to your handler
```

```python
# Option B — let pyramids install its coloured console handler
from pyramids.base.config import Config

Config(level="INFO")
```

Option B sets `propagate = False` on the `pyramids` logger, so records are emitted once by pyramids' own
handler rather than twice (once more through your root handler). Do not combine it with Option A expecting
both to print.

Option B is reversible: a bare `Config()` removes the handlers pyramids installed and restores propagation.
That matters for test suites — pytest's `caplog` attaches to the *root* logger, so while `propagate` is
`False` a `caplog` assertion on a `pyramids.*` logger captures nothing and passes vacuously. If a dependency
opts in on your behalf, call `Config()` to hand the namespace back.

**If you relied on the third-party quieting**, pass it explicitly:

```python
Config(level="INFO", quiet_third_party=True)  # pins fiona/rasterio/shapely/matplotlib/urllib3/osgeo to WARNING
```

**Exception types on three dtype helpers changed** to a descriptive `ValueError`, from the incidental error
that leaked out of an empty table lookup. Only affects code catching the old type:

| Call | Before | After |
|------|--------|-------|
| `numpy_to_gdal_dtype(<unmapped dtype>)` | `IndexError` | `ValueError` |
| `gdal_to_numpy_dtype(gdal.GDT_Unknown)` | `AttributeError` | `ValueError` |
| `gdal_to_ogr_dtype(<complex band>)` | `TypeError` | `ValueError` |

## dataset

### unreleased

**`Dataset.no_data_value` can report a numpy scalar where it reported a Python `int`.** Hard change, silent —
the number is the same, only its type differs. It affects one case: an **unsigned** band whose no-data is unset
or `NaN`, where pyramids substitutes the dtype's maximum. That substituted sentinel is now built as a numpy
scalar of the band's dtype instead of a Python `int`, so it agrees in type as well as value with the sentinel
picked when a requested no-data overflows the band — previously the same answer read back as `255` from one path
and `np.uint8(255)` from the other, and the `==` pinning their agreement could not see the difference.

What a caller sees, for a `uint8` band with no usable no-data:

```python
# Before
ds.no_data_value      # (255,)          <- builtin int

# After
ds.no_data_value      # (np.float64(255.0),)
```

`float64`, not `uint8`: GDAL's `SetNoDataValue` takes a C double and refuses a numpy `uint8`, so the value is
round-tripped through a `float64`. Signed and floating bands are unaffected (nothing is substituted for them),
and a band with a concrete no-data already reported a numpy scalar before this release.

Anything comparing with `==`, or reading the value into numpy, needs no change. Change what needs a builtin:

- `json.dumps(ds.no_data_value)` → `json.dumps([float(v) for v in ds.no_data_value])`
- `"%d" % nodata` and `is`-comparisons against small ints
- arithmetic where wrapping matters — a numpy *integer* scalar wraps at the dtype bound
  (`np.uint8(255) + 1 == 0`) instead of promoting. `float(nodata)` first if you are doing arithmetic on it.

**`create_from_array` is now `from_array`, and takes a `GeoReference`.** Hard change, no deprecation alias — the
old name and the old flat keywords are gone. The same rename applies to `UgridDataset.create_from_arrays` ->
`from_arrays` (still plural: it takes three arrays, not one).

The four flat georeferencing keywords are replaced by one `geo_ref` argument, and `driver_type` is gone
entirely — the driver is now derived from the path extension:

```python
# Before
Dataset.create_from_array(arr, geo=GEO, epsg=4326)
Dataset.create_from_array(arr, top_left_corner=(0, 10), cell_size=0.05)
Dataset.create_from_array(arr, driver_type="GTiff", path="out.tif")
Dataset.create(cell_size=0.05, rows=r, columns=c, dtype="float32", bands=1,
               top_left_corner=(0, 10), epsg=4326)
Dataset.create_empty(rows, cols, geo=GEO, epsg=4326, driver_type="MEM")
UgridDataset.create_from_arrays(node_x, node_y, faces)

# After
from pyramids.dataset import Dataset, GeoReference

Dataset.from_array(arr, geo_ref=GeoReference(geo=GEO, epsg=4326))
Dataset.from_array(arr, geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=0.05))
Dataset.from_array(arr, geo_ref=GeoReference(geo=GEO), path="out.tif")  # driver from the extension
Dataset.create(rows=r, columns=c, dtype="float32", bands=1,
               geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=0.05, epsg=4326))
Dataset.create_empty(rows, cols, geo_ref=GeoReference(geo=GEO, epsg=4326))
UgridDataset.from_arrays(node_x, node_y, faces)
```

The module-level `pyramids.netcdf.engines.variables.create_from_array` is renamed `from_array` too. It carries
no underscore but is an engine internal — reach it through `NetCDF.from_array`, its public facade.

`GeoReference` is importable from `pyramids.dataset`, `pyramids.netcdf` and
`pyramids.netcdf.array_options` — all three resolve to the same class.

**Why there is no alias.** The signature and the name were both changing; keeping the old spelling working would
have meant migrating twice. Both failure modes are loud — the old name raises `AttributeError`, a flat keyword
raises `TypeError` — and because `NetCDF.from_array` now declares its real parameters instead of
`*args, **kwargs`, **mypy catches both statically**, which the old facade prevented.

**`driver_type` removed from `create_from_array` and `create_empty`.** `path` alone decides memory-vs-disk, and
the extension names the format, so the parameter could only agree with `path` or contradict it. `path=None`
gives an in-memory raster; `path="out.tif"` a GTiff; `path="out.nc"` a netCDF — the last of which was rejected
on *both* routes before: the default `driver_type="MEM"` by the `.tif` suffix check, and an explicit
`driver_type="netcdf"` by the unconditional `COMPRESS=LZW` the netCDF driver refuses. Removing the parameter
*and* gating that option is what makes it work. Creation `options` still require a `path`, which is the
invariant the old `driver_type="GTiff"` guard was really enforcing.

Formats that GDAL can only write by copy (`.png`, `.jp2`) now raise `FileFormatNotSupportedError` naming the
extension and the driver, instead of failing inside GDAL. An extension the driver catalog does not know still
raises `DriverNotExistError`.

**`merge_rasters` and `stack_bands` now refuse write-by-copy-only destinations.** Both have two internal
write paths and only one of them can produce such a format, so accepting it would make the destination's
legality depend on `method` (or on whether the inputs share a dtype) — the same defect as letting an
unrelated argument pick the format. Both take the stricter answer, which refuses `.png`, `.jpg` / `.jpeg`,
`.jp2` / `.j2k`, `.asc` and `.vrt`. The one this costs is **`.asc`**, which `merge_rasters` could write
before through its z-order (`method="first"` / `"last"`) path. Write a GTiff and convert it.

**`Dataset.to_file` now accepts more extensions, and one of them can lose data.** This method is untouched by
the rename, but it reads the same driver catalog, so correcting the catalog's extension rows widened what it
accepts. `.tiff`, `.png`, `.jpg`, `.jpeg`, `.img`, `.jp2` and `.j2k` previously raised
`DriverNotExistError` and now write a
file. The sharp case is dtype. PNG and JPEG store 8-bit data, so a float32 raster written to `.png` is
converted to `Byte` — values clipped, fractional parts gone. GDAL reports that only as a `RuntimeWarning`, so
pyramids now raises it to a `DtypeNarrowingWarning` naming both dtypes and the driver. JPEG2000 is stricter
still: `.jp2` / `.j2k` accept only their own dtypes and **raise** rather than converting, so a float32 raster
there warns and then fails with `FailedToSaveError` — write those from a `uint8` / `int16` source.

```python
import warnings
from pyramids.errors import DtypeNarrowingWarning

warnings.simplefilter("error", DtypeNarrowingWarning)   # make the conversion fail loudly
```

Note the deliberate asymmetry with the constructors: `to_file("x.png")` writes (PNG supports `CreateCopy`)
while `from_array(path="x.png")` raises `FileFormatNotSupportedError` (PNG does not support `Create`). Whether
a format is reachable depends on how the call writes, not only on the extension.

**You can no longer name a driver whose extension the catalog does not list.** This is the one capability the
`driver_type` removal costs. On `main`, `driver_type="EHdr", path="out.bil"` worked; now `.bil` raises
`DriverNotExistError` and there is no escape hatch. If you need a format the catalog does not carry, write a
GTiff (or build in memory) and convert, or open an issue asking for the extension to be added. Sibling spellings
of a catalogued format now resolve alike: `.tiff` as well as `.tif`, `.nc4` as well as `.nc`, `.jpg` as well as
`.jpeg`, and `.j2k` as well as `.jp2`. Resolving is not the same as being writable by these constructors: all
four JPEG/JPEG2000 spellings resolve and are then refused with `FileFormatNotSupportedError`, because they are
write-by-copy-only formats. (`.jpeg` did not resolve on `main` at all — the catalog carried it with a leading
dot, so the lookup never matched.)

**`Dataset.create` no longer requires a CRS.** `epsg` was a required positional; omitting it raised
`TypeError`. It is now a field of `geo_ref` defaulting to `4326`, so a call that says nothing about the
CRS silently stamps WGS 84 instead of refusing. `from_array` and `create_empty` always behaved this way —
only `create` changes, and it changes from "state the CRS" to "we assume WGS 84". Pass `epsg` explicitly,
or `GeoReference(..., epsg=None)` for a deliberately CRS-less raster.

**`Dataset.create`'s positional order changed, and its CRS handling widened.** `cell_size` is gone from slot 0,
so `rows` and `columns` each shift up one — but `geo_ref` is required and keyword-only, so a ported positional call
fails loudly with `TypeError` rather than silently misreading its arguments. Separately, `create` used to take
`epsg: int` straight through `sr_from_epsg`; it now shares the helper the other constructors use, so
`GeoReference(epsg=None)` yields a raster with no CRS, and non-EPSG CRS strings are accepted.

**`create_empty` is the only constructor where `geo_ref` is optional.** A header-only allocation often does not
care where it sits, so omitting `geo_ref` — or passing one that carries no transform at all, such as
`GeoReference(epsg=3857)` — keeps the identity transform. A *partially* specified reference is not covered by
that convenience: a `top_left_corner` without a `cell_size` (or the reverse) raises, exactly as it does in
`from_array` and `create`, rather than silently discarding the half you supplied.

**`RasterLike` and `RasterBase` changed shape.** `create_from_array` is `from_array` on the
`pyramids.base.protocols.RasterLike` protocol and on the `RasterBase` ABC; `bands_values` is gone, and
`variable_name` moved down to `NetCDF`, where it actually applies. If you subclass `RasterBase`, rename your
override and adopt the new signature. `RasterLike` is `@runtime_checkable`, so this one bites at runtime too: an
`isinstance(obj, RasterLike)` guard on a stale duck type carrying `create_from_array` flips from `True` to
`False` and silently takes the else-branch, rather than raising anywhere you can see it.

**`pyramids.netcdf.array_options.GeoTransform` is gone**, with no re-export. It was a local alias for the plain
6-tuple, and `pyramids.dataset.GeoTransform` is a richer `NamedTuple` of the same shape but a different type, so
re-exporting that under the old name would have silently changed what the name meant. If you imported the alias
for annotation, use `pyramids.base.georeference.GeoTransformTuple` (the structural 6-tuple) or the
`pyramids.dataset.GeoTransform` NamedTuple — whichever you actually meant.

**`NetCDF.from_array` returns a `Container`.** It always did; the annotation said `NetCDF`. `Container` adds no
public API over `NetCDF`, so nothing changes at runtime — the type is just honest now.

**`create_overviews` now refuses a plain VRT whose description is not a path.** It raises `OverviewTargetError`
instead of returning normally. That is a new exception, importable as
`from pyramids.errors import OverviewTargetError`, which subclasses `ValueError`,
so an existing `except ValueError` around the call keeps catching it — catch the new type to tell "this dataset
can never work" apart from "these arguments were wrong". `recreate_overviews` raises it for the same shape, where
it previously reported a misleading `ReadOnlyError` advising a reopen that a handle with no path cannot perform —
or, when the VRT exposed **no** levels at all, returned normally with a `UserWarning` saying to call
`create_overviews()` first. That advice is now refused too, so a pipeline that called `recreate_overviews()`
defensively across a mix of handles and ignored the warning goes from a silent no-op to a raise.
A plain VRT owns no pixel storage, so GDAL can only write its overviews to an external `.ovr`
sidecar named after the dataset description — and when that description is not a path, the sidecar landed as a
file called literally `.ovr` in the process's working directory, attached to nothing, while the levels the handle
was already exposing were dropped. The call reported success and silently did neither thing it promised.

Three descriptions are refused: an empty one, a blank one, and any that begins with `<` once stripped — in
practice an inline VRT XML document passed to `Dataset.read_file(...)`, which previously surfaced as a raw GDAL
`RuntimeError` naming a file whose name was the whole document. No real path starts with `<`.

These calls hand back such a handle, and so start raising:

- `Dataset.get_overview_dataset(...)` — **only** its lazily described form, taken when the parent can be reopened
  by name. When the parent cannot be (a `from_array` raster, for instance), the method materialises a
  `MEM` level instead, and that still builds.
- anything wrapping a `gdal.Translate(..., format="VRT")` or `gdal.BuildVRT("", ...)` result kept in memory.
- `NetCDF.get_variable(...)` when the classic view comes back in **index space** — an irregularly spaced
  coordinate defeats the geotransform guess, and the view is then wrapped in a plain pathless VRT. A view over a
  regular grid is not VRT-wrapped and is unaffected.
- `Dataset.wrap_longitude()` on a **file-backed** source, whose lazy roll is a plain pathless VRT. On an
  in-memory source the roll is a `MEM` raster and is unaffected. This one previously produced the damage above
  verbatim — a stray `.ovr` and no levels — so the refusal is a fix, not a regression.
- `Dataset.to_zarr(..., overview_factors=[...])` on any handle the refusal covers, since it builds the pyramid
  levels through `create_overviews`. The target is checked pre-flight, so the call leaves no store behind, and
  before the `compute` argument, so a call that is also `compute=False` reports the refusal rather than the
  `ValueError`. Without `overview_factors` the refusal does not apply; a description-less VRT then fails where
  it always did, in `to_zarr`'s base-array write.

Write the view out first and build the overviews on the saved raster:

```python
view.to_file("level.tif")
view.close()  # on Windows an open handle keeps the parent file locked
saved = Dataset.read_file("level.tif", read_only=False)
saved.create_overviews(overview_levels=[2, 4])
```

Unaffected **by the `create_overviews` refusal**: **warped** VRTs keep their overviews in RAM and need no
sidecar. These produce one — `Dataset.warped_view(...)`, `Dataset.to_crs(...)` in its warping form,
`Dataset.crop(mask, touch=False)` with a **vector** mask, and the lazy `georeference` / `orthorectify` forms.
(`to_crs(..., maintain_alignment=True)` and `crop` with a *raster* mask are different paths: both return a `MEM`
dataset, exempt as a non-VRT rather than as a warped one.)
Tests cover `warped_view` and `to_crs`; the others are exempt by the same root-element check. A VRT with a
real path (including under `/vsimem/`) names its sidecar after that path, and `MEM` rasters are not VRTs at all.
Warped VRTs are **not** exempt from the `recreate_overviews` refusal — see the next entry.

**`recreate_overviews` now raises `OverviewTargetError` instead of `ReadOnlyError` when a VRT computes the
levels.** GDAL reports every unwritable overview target with the same `CPLE_NoWriteAccess` it uses for a
genuinely read-only dataset, so the old message told callers to reopen with `read_only=False` whenever the write
was refused. For a level a VRT computes rather than stores, that advice cannot help and often cannot even be
followed. Two shapes are affected:

- a **warped** VRT — its levels land on `VRTWarpedRasterBand`s, which are never writable. A warped view taken
  from a *writable* parent fails identically, and a pathless view has no path to reopen. `create_overviews()`
  still builds a warped view's levels; only in-place regeneration is refused.
- a **plain VRT that inherits its levels from the source** it wraps — the common
  `gdal.Translate(..., format="VRT")` or mosaic over an already-overviewed raster. GDAL serves each such level
  from an implicit read-only VRT it builds for the level, not from the handle you opened, so the read-only
  dataset in its message is one `read_only=False` cannot reach — holding the source open writable does not help
  either. Give the VRT its own sidecar with `create_overviews()`, or regenerate on the source raster instead.

The two are told apart by who owns the level: a level owned by a VRT is computed, one owned by a real raster
(the dataset itself for an internal overview, the `.ovr` GTiff for an external one) is stored.

`ReadOnlyError` is now raised only when a reopen is still worth trying — a stored level on a handle that is
itself open read-only. It is **not** a promise that reopening will succeed: a VRT serving an explicit
`<Overview>` owns a real, on-disk-writable `.ovr`, yet GDAL opens VRT sources read-only and refuses however the
parent was opened. What has changed is that a dataset **already open for writing** never gets told to reopen —
that shape now raises `OverviewTargetError` too, since the access mode demonstrably is not the blocker.

### 0.48.0

**`recreate_overviews` now rebuilds a band's levels in one cascading pass.** Each deeper level is decimated from
the level above rather than from the full-resolution band, matching what `create_overviews` (`BuildOverviews`)
has always done — previously `recreate_overviews` rebuilt every level directly from the source. Nothing warns;
the call still succeeds.

Level 0 is unaffected. A level >= 1 keeps its previous values only where the resampling survives being applied
twice:

- `nearest` — unchanged everywhere; GDAL does not cascade it.
- `average`, `rms` — unchanged on a floating-point band with no no-data value. An integer band picks up per-level
  rounding of up to 1 DN, and a no-data gap changes the result regardless of dtype.
- `cubic`, `cubicspline`, `bilinear`, `lanczos`, `gauss`, `mode` — changed on ordinary data. `mode` can move by
  whole classes, because the mode of modes is not the mode.

There is no way to get the old per-level values back: no API rebuilds a deep level directly from the source any
more. If your pipeline depends on them, rebuild the levels you need yourself from
`Dataset.read_array()` and write them out.

### 0.47.0

**Operations that need a CRS now refuse instead of assuming one.** Each raises `CRSError` naming the operation
and how to fix it, where previously the missing CRS was silently filled in with WGS 84:

- `Dataset.to_crs(...)` — reprojection has no source frame to warp from.
- `Dataset.align(...)` — both the receiver and the reference must carry a CRS; two rasters that both report
  `epsg is None` are no longer treated as matching.
- `Dataset.crop(bbox=...)` / `NetCDF.crop(bbox=...)` without an explicit `epsg=`.

**A cube with no CRS no longer writes a fabricated one.** Previously both writers recorded `4326` and a WGS 84
WKT, claiming a projection the data never had. Now:

- `to_zarr` records `epsg: 0` with an empty `crs_wkt`. Readers should treat that pair as "no CRS";
  `geobox_crs()` does.
- `to_netcdf` **omits** both attributes entirely, since a NetCDF has no geobox slot to put a null in.

On read, a NetCDF's `crs_wkt` / `epsg` root attributes are adopted only when written beside the `GeoTransform`
the same writer emits, so a stray attribute in a third-party file no longer defines the CRS.

**`read_part(bbox=...)` and `point(...)` changed their default CRS, for every raster.** `bbox_crs` /
`point_crs` used to default to `4326`, so an unqualified bbox or point was interpreted as lon/lat and
reprojected into the raster's CRS. They now default to `None`, meaning "already in the raster's own
coordinates", and nothing is transformed. On a raster that is not in EPSG:4326 this changes which pixel an
unqualified call reads.

- To keep the old behaviour, pass the CRS explicitly: `ds.read_part(bbox, bbox_crs=4326)`,
  `ds.point(x, y, point_crs=4326)`.
- To read in the raster's own coordinates — usually what you want — leave it out.

Neither method refuses a raster with no CRS, so a windowed read of an ungeoreferenced raster keeps working.
Passing an explicit `bbox_crs` / `point_crs` against a raster that has none raises `CRSError`, rather than being
ignored: there is no frame to transform into.

## cli

### 0.47.0

**`pyramids calc` refuses a first input with no CRS.** The result cannot be georeferenced and pyramids will not
stamp a default; set a CRS on the input first. `pyramids georeference` is unaffected — its GCPs and `--gcp-crs`
replace the georeference wholesale.

## netcdf

### unreleased

**`NetCDF.variable_names` reports the store's declaration order, not alphabetical order.** Hard change, silent —
the *set* of names is unchanged, only the order, so nothing raises and nothing warns. The property used to hand
back the CF classification's own list, which is sorted; it now filters the declared list instead, keeping the
order the file uses.

```python
# Before
nc.variable_names     # ['q', 'z']   <- alphabetical

# After
nc.variable_names     # ['z', 'q']   <- the order the store declares
```

This matters beyond iteration order. `_fan_out_eager` templates a container-wide result from the **first**
spatial variable — taking its geotransform, CRS, no-data and extra dimensions — so a container-wide `to_crs`,
`resample` or `crop` can now template from a different variable than it did before, and the order propagates
into `to_netcdf` and `to_xarray` output.

The declared order is the intended answer: it is what the file says, it matches what `ncdump` and xarray show,
and templating a fan-out from an alphabetically-first variable was arbitrary. If you depended on the sorted
order, sort explicitly: `sorted(nc.variable_names)`.

### 0.37.0

Introduces the `Container` / `Variable` type split plus a wave of API consolidation. Each change below shows
**what changed** and the **before → after** so you can update at a glance.

#### At a glance

`breaking*` = only breaks *exact-type* checks; `isinstance(x, NetCDF)` still holds.

| Change | Kind | What you do |
|--------|------|-------------|
| `read_file` / `get_variable` return `Container` / `Variable` | breaking* | use `isinstance(x, NetCDF)` |
| `subset()` returns a `NetCDF`, not a `Dataset` | breaking* | nothing unless you used `type(x) is Dataset` |
| `CFInfo` is frozen | breaking | `dataclasses.replace(cf, ...)` |
| `LabeledDataset.read_file(engine=...)` validates `engine` | breaking | pass a valid engine name |
| `NetCDF(gdal_dataset)` direct construction | deprecated | use `read_file` / `get_variable` |
| `get_variable_names()` | deprecated | `variable_names` property |
| `NetCDF.plot(colour=...)` | breaking | loose colour kwargs + `axes=CoordinateSpec(...)` |
| `MetaData` / `DimMetaData` | deprecated | `ClassicDimMetadata` / `ClassicDimensionInfo` |
| kerchunk `backend="kerchunk"` | deprecated | `backend="legacy"` |
| `_LabeledArray` / `_apply_unpack` | renamed | `LabeledArray` / `apply_unpack` |

#### Breaking changes (update required)

**1. Opening a store returns `Container`; extracting a variable returns `Variable`.**
`NetCDF` is now a base class with two concrete subclasses. Both still pass `isinstance(x, NetCDF)`, so only
*exact-type* checks break.

```python
# Before — everything was a NetCDF
nc  = NetCDF.read_file("cube.nc")     # NetCDF
var = nc.get_variable("t")            # NetCDF
type(nc) is NetCDF                    # True

# After
nc  = NetCDF.read_file("cube.nc")     # Container
var = nc.get_variable("t")            # Variable
isinstance(nc, NetCDF)                # True   <- use this
type(nc) is NetCDF                    # False  <- this no longer holds
```

**2. `subset()` returns a `NetCDF` (a `Variable`), not a plain `Dataset`.**

```python
# Before
ds = nc.subset("t", time=0)           # Dataset

# After
var = nc.subset("t", time=0)          # Variable (a NetCDF); read_array()/crop()/sel() all still work
```

**3. `CFInfo` is immutable (frozen).** In-place assignment now raises `dataclasses.FrozenInstanceError`.

```python
# Before
cf.<field> = new_value                # mutated in place

# After
import dataclasses
cf = dataclasses.replace(cf, <field>=new_value)
```

**4. `LabeledDataset.read_file(engine=...)` validates the engine.** An unrecognised value now raises
`ValueError` instead of being silently ignored.

```python
# Before
ds = LabeledDataset.read_file(store, engine="zar")    # typo silently ignored

# After — valid: "zarr", "netcdf", "netcdf4", "hdf5", "h5netcdf", or None
ds = LabeledDataset.read_file(store, engine="zarr")
```

#### Deprecations (old still works, warns — update when convenient)

**5. Constructing the base `NetCDF(...)` directly is deprecated.** Use the typed entry points.

```python
# Before
nc = NetCDF(gdal_dataset)

# After
nc  = NetCDF.read_file("cube.nc")     # -> Container
var = nc.get_variable("t")            # -> Variable
```

**6. `get_variable_names()` -> the `variable_names` property.**

```python
# Before
names = nc.get_variable_names()
# After
names = nc.variable_names
```

**7. `NetCDF.plot` mirrors `Dataset.plot` — `colour=` is gone; colour is loose kwargs and coordinates group
into `axes=CoordinateSpec(...)`.** The `colour=` parameter and both the `ColorOpts` dataclass and its British
alias `ColourOpts` have been removed from `pyramids.netcdf`. Pass the colour knobs (`cmap`, `vmin`, `vmax`,
`robust`, `center`, `extend`, `levels`, `norm`) as loose keyword arguments, and group the curvilinear-coordinate
parameters (`coords`, `x_dim`, `y_dim`) into the new frozen `CoordinateSpec` dataclass passed as `axes=`.

```python
# Before
from pyramids.netcdf import ColourOpts
nc.plot("t2m", colour=ColourOpts(cmap="viridis", robust=True), x_dim="rlon", y_dim="rlat")
# After
from pyramids.netcdf import CoordinateSpec
nc.plot("t2m", cmap="viridis", robust=True, axes=CoordinateSpec(x_dim="rlon", y_dim="rlat"))
```

For colour-bar styling and data-style presets, pass the cleopatra bags exactly as `Dataset.plot` does —
`colorbar=ColorBar(...)` (or `colorbar=False` to hide it) and `data_style=DataStyle(...)`.

**8. Classic dimension models renamed.**

```python
# Before
from pyramids.netcdf.dimensions import MetaData, DimMetaData
# After
from pyramids.netcdf.dimensions import ClassicDimMetadata, ClassicDimensionInfo
```

**9. kerchunk `backend="kerchunk"` -> `backend="legacy"`.**

```python
# Before
nc.to_kerchunk("refs.json", backend="kerchunk")
# After
nc.to_kerchunk("refs.json", backend="legacy")   # or omit backend= for the native default
```

#### Renames where the old name still works (no rush)

**10. Promoted internals (underscore aliases kept).**

```python
# Before
from pyramids.netcdf.labeled import _LabeledArray
from pyramids.netcdf._lazy import _apply_unpack
# After
from pyramids.netcdf import LabeledArray
from pyramids.netcdf._lazy import apply_unpack
```

Also: the private modules `_kerchunk.py` -> `_kerchunk_facade.py` and `_kerchunk_native.py` -> `_kerchunk_builder.py`
were renamed. They are internal — use `NetCDF.to_kerchunk` / `NetCDF.combine_kerchunk` rather than importing them.

#### New, opt-in capabilities (not breaking)

- **Typed dispatch:** branch on `isinstance(x, Container)` vs `isinstance(x, Variable)` instead of inspecting
  `is_subset` / `band_count`. Import from `pyramids.netcdf` or `pyramids.netcdf.variable`.
- **Cloud read tuning:** `CloudConfig(vsicurl_tuning=True, curl_cache_size=...)` enables the fast single-file
  `/vsicurl/` read preset.

#### Naming note

The new public types are named `Container` and `Variable`. They read cleanly when namespace-qualified
(`pyramids.netcdf.Container`). If you `from pyramids.netcdf import Variable` in a module that also uses a
labeled-array library, the name collides conceptually with that library's own `Variable` type — prefer the
namespace-qualified form, or alias on import:

```python
from pyramids.netcdf import Variable as NcVariable
```

#### Migration checklist

1. Pin the new `pyramids` version in your dependencies.
2. Run your suite under `-W error::DeprecationWarning` and fix everything it flags (items 5-10 above).
3. Fix the hard-behavior-change items (1-4) — they do not warn: search for `type(x) is NetCDF`, `type()` /
   `Dataset` checks on `subset()` results, `CFInfo` mutation, and unrecognised `LabeledDataset` engines.
4. Done — `isinstance(x, NetCDF)` keeps working, so most code needs no change.
