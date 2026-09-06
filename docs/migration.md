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

**`Dataset.dtype` reports numpy's spelling, so a Byte raster reads `uint8` rather than `byte`.** Hard change,
silent — the property still returns one string per band, but two of the catalog's names moved, two half-precision
types became reachable, and one band type that used to return a value now raises. The names are numpy's wherever
numpy has one, which is what STAC's `raster:bands[].data_type` wants; GDAL's complex-integer and complex-half
types have no numpy name, so those keep the catalog's own spelling rather than all collapsing onto `complex64`.

| Band type | Before | After |
|---|---|---|
| `GDT_Byte` | `['byte']` | `['uint8']` |
| `GDT_CFloat64` | `['complex-float64']` | `['complex128']` |
| `GDT_Float16` | `IndexError` from the catalog lookup | `['float16']` |
| `GDT_CFloat16` | `IndexError` from the catalog lookup | `['complex-float16']` |
| `GDT_Unknown` | the catalog's empty name — a float `NaN`, not a `str` | `ValueError` naming the code |
| everything else (`uint16`, `int32`, `float32`, `complex-int16`, `complex-float32`, …) | unchanged | unchanged |

`byte` and `complex-float64` are the two that break working code: a `ds.dtype[0] == "byte"` branch stops
matching, and anything that feeds `dtype` into STAC, a manifest or a filename emits a different string. Search
for string comparisons against `"byte"` and `"complex-float64"` and update them.

Feeding the value *back* is better off than it was. `numpy_to_gdal_dtype("byte")` returned `14` — `GDT_Int8`,
a different band type — so `numpy_to_gdal_dtype(ds.dtype[0])` on a Byte raster silently produced the wrong
code; `"uint8"` gives `1`, which is right. `"complex-float64"` raised `TypeError` there and `"complex128"`
gives `11`. The two half-precision rows are an addition, not a break — they raised before. `GDT_Unknown` turns
a value into an exception, but the value it returned was a `NaN` that violated the property's own `list[str]`
contract, so nothing could have used it.

**`Dataset.copy()` with no `path` returns a dataset in `"write"` mode, even when the source was read-only.**
Hard change, silent — the copy used to inherit the source's access mode:

```python
ds = Dataset.read_file("raster.tif")     # read_only
clone = ds.copy()

# Before
clone.access          # 'read_only'   <- a write raised ReadOnlyError
# After
clone.access          # 'write'
```

An in-memory copy is a fresh `MEM` raster that nothing else references, so refusing writes on it protected
nothing — the source file is untouched either way — while a `copy()` taken specifically to get a mutable
working copy of a read-only file could not be written to, and the `ReadOnlyError` advised reopening the
*source* with `read_only=False`, which is not what the caller wanted. Copying **to a path** is unchanged in
spirit: a format that supports `Create` gives `"write"`, and a write-by-copy-only format (`.png`, `.jpg` /
`.jpeg`, `.jp2` / `.j2k`, `.asc`) still gives `"read_only"`, because `CreateCopy` hands back a read-only
dataset for those. If you relied on the copy being read-only as a guard, guard it yourself — there is no
parameter to ask for the old mode.

**`Dataset.no_data_value` can report a numpy scalar where it reported a Python `int`.** Hard change, silent —
the number is the same, only its type differs. It affects one case: an unsigned band **wider than 8 bits**
(`uint16`, `uint32`, `uint64`) created with a `NaN` no-data, where pyramids substitutes the dtype's maximum
because `NaN` cannot be stored there. That substituted sentinel is now built as a numpy scalar instead of a
Python `int`, so it agrees in type as well as value with the sentinel picked when a *requested* no-data
overflows the band — previously the same answer read back as `65535` from one path and a numpy scalar from the
other, and the `==` pinning their agreement could not see the difference.

What a caller sees, for a `uint16` band created with `no_data_value=np.nan`:

```python
# Before
ds.no_data_value      # (65535,)                  <- builtin int

# After
ds.no_data_value      # (np.float64(65535.0),)
```

Measured across the dtypes, with `Dataset.create(rows=3, columns=4, dtype=..., bands=1, no_data_value=...)`:

| dtype and no-data | Before | After |
|---|---|---|
| `uint16` + `NaN` | `(65535,)` — `int` | `(np.float64(65535.0),)` |
| `uint32` + `NaN` | `(4294967295,)` — `int` | `(np.float64(4294967295.0),)` |
| `uint64` + `NaN` | `(18446744073709551615,)` — `int` | `(np.uint64(18446744073709551615),)` |
| `uint8` + `NaN` | `(nan,)` | `(nan,)` — unchanged |
| signed / floating + `NaN` | `(nan,)` | `(nan,)` — unchanged |
| any dtype, `no_data_value=None` | `(None,)` | `(None,)` — unchanged |

Two things the shape of that table decides for you:

- **`uint8` is excluded.** A Byte band with a `NaN` no-data reports `(nan,)`, before and after — nothing is
  substituted, because 255 is white in 8-bit imagery and fabricating it as a sentinel would put every white
  pixel out of domain. Do not test this change on a Byte raster: it will not show there.
- **Only `NaN` triggers it.** An *unset* no-data (`no_data_value=None`) reports `(None,)` on both sides. The
  substitution is for a sentinel that cannot be stored, not for the absence of one.

`float64`, not `uint16`: GDAL's `SetNoDataValue` takes a C double and the value is round-tripped through it. The
one exception is `uint64`, whose maximum has no exact `float64`, so it stays a numpy `uint64`. A band with a
concrete no-data already reported a numpy scalar before this release.

Anything comparing with `==`, or reading the value into numpy, needs no change. Change what needs a builtin:

- `json.dumps(ds.no_data_value)` → `json.dumps([float(v) for v in ds.no_data_value])`
- `"%d" % nodata` and `is`-comparisons against small ints
- arithmetic where wrapping matters — only on the `uint64` row, whose value is a numpy *integer* scalar and
  wraps at the dtype bound (`np.uint64(2**64 - 1) + 1 == 0`) instead of promoting. The `uint16` / `uint32` rows
  are `float64` and do not wrap. `float(nodata)` first if you are doing arithmetic on any of them.

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

**`file_i` counts only real files, refuses a negative index, and now works for a tar.** Hard change, silent for
zip and loud for tar. `file_i` is public on `Dataset.read_file` and `NetCDF.read_file`, and it selects a member
of a compressed archive. Three behaviours moved. Measured on a zip and a tar each holding a `sub/` directory
entry, `sub/a.asc` and `b.asc`:

| Call | Before | After |
|---|---|---|
| zip, `file_i=0` | `.../probe.zip/sub/` — the **directory** entry | `.../probe.zip/sub/a.asc` |
| zip, `file_i=1` | `.../probe.zip/sub/a.asc` | `.../probe.zip/b.asc` |
| zip, `file_i=2` | `.../probe.zip/b.asc` | `FileFormatNotSupportedError` naming the count and members |
| zip, `file_i=-1` | `.../probe.zip/b.asc` — Python's from-the-end | `FileFormatNotSupportedError` |
| zip, `file_i=-5` | `IndexError: list index out of range` | `FileFormatNotSupportedError` |
| tar, any `file_i` | `/vsitar/<archive>` — the index was **ignored** | the member at that index |

- **Directory entries no longer consume an index.** A zip written with explicit directory entries shifted every
  member by one and made `file_i=0` name a directory GDAL cannot open. If you compensated by adding one, remove
  the adjustment.
- **A negative index is refused rather than counted from the end.** `file_i=-1` never meant "the last member" —
  it happened to work through Python's list indexing while `-5` raised a bare `IndexError` from the same line.
  Both now raise `FileFormatNotSupportedError`, which names how many members the archive holds and lists them.
  To read the last member, pass its index: `len(members) - 1`.
- **A tar now honours `file_i` at all.** Every index resolved to the bare `/vsitar/<archive>` path before, and
  GDAL picked a member itself. A tar read that relied on that pick can now land on a different file — pass the
  index you want, or open the member path directly.

**`to_zarr` writes the band's sentinel as the array's `fill_value`.** Hard change, silent, and it changes bytes
on disk. The store's array metadata carried `fill_value: 0.0` regardless of the raster's no-data, which was
recorded only in a pyramids-specific `no_data_value` attribute. It now carries the real sentinel, in both the
Zarr `fill_value` slot and a CF `_FillValue` attribute. Written from a `float32` raster whose no-data is
`-9999.0`:

```text
# Before                                    # After
data/zarr.json  fill_value = 0.0            data/zarr.json  fill_value = -9999.0
attrs: no_data_value = [-9999.0]            attrs: _FillValue = -9999.0, no_data_value = [-9999.0]

Dataset.read_file(store).no_data_value      Dataset.read_file(store).no_data_value
  -> (0.0,)                                   -> (-9999.0,)
```

The cell values are untouched — only the declared sentinel moved — so pyramids reading its own store back is
the fix: the raster's no-data survives the round trip instead of being replaced by `0.0`. The cost is borne by
every other reader. Any consumer that honours the Zarr `fill_value` or the CF `_FillValue` — zarr-python and
GDAL do, and so does anything layered on them — reads `-9999.0` from a store written by the new version where
it read `0.0` from one written by the old, and masks accordingly. Stores already on disk are unaffected until
they are rewritten. If a downstream pipeline depended on `0.0` being the fill value, rewrite the store or set
the reader's fill explicitly.

**No-data masking in the analysis engine now takes its tolerance from the band's dtype.** Hard change, silent.
`count_domain_cells`, `apply`, `fill`, `footprint`, `plot_histogram` and `to_image` all ask one question — does this
cell hold the band's no-data sentinel — and each asked it with a *relative* tolerance of its own: `rtol=1e-3` in
`count_domain_cells` and `apply`, `1e-6` in `fill`, `1e-5` in the two renderers. So one band could be counted one way
and drawn another, and every one of them discarded ordinary cells that merely lay near the sentinel. A relative
window scales with the sentinel: `rtol=1e-3` around `-9999` masks everything down to `-10009`, and `rtol=1e-5` around
an `int32` sentinel of `2e9` masks everything within `20 000` of it.

The tolerance is now the band's own — none at all for an integer or boolean band, whose sentinel is stored exactly,
and single precision's `eps` (about `1.2e-7` relative) for a floating band of any width, which is the slack a
sentinel picks up passing through `float32` storage or a driver's decimal text. `numpy.isclose`'s default
`atol=1e-8` is dropped with it, so a sentinel of `0` no longer swallows genuinely small cells.

Two consequences of "the band's own" worth stating, because the obvious reading of each is the wrong one:

- **The float tolerance does not narrow further for a wide band, and does not widen for a narrow one.** It is
  single precision's `eps` for `float16`, `float32` and `float64` alike. Taking each dtype's own `eps` would
  make a `float16` band the loosest of the three — one ULP of half precision is `9.8e-4`, a window of `±9.8`
  around a `-10000` sentinel, which swallows both of that sentinel's neighbouring representable values.
- **An integer sentinel is compared as an exact integer, not through a `float`.** That matters only at the
  64-bit limits, where it is the whole answer: `float(2**63 - 1)` rounds *up* past `int64`'s maximum, so a band
  whose sentinel is its own dtype maximum would be judged unable to hold it and mask nothing at all — with
  `fill` then overwriting the very cells it was told to leave alone. `int64` and `uint64` maxima are exactly
  the sentinels pyramids fabricates for an unsigned band with an unstorable no-data, so this is the ordinary
  path for a 64-bit raster rather than a corner of one.

For a caller, a cell within ~0.1% of the sentinel that used to be no-data to `count_domain_cells` / `apply` is now
data; nothing that was masked before is unmasked *less* accurately, the change only ever keeps cells. On the test
suite's own rasters nothing moves at all — all 473 fixture bands carrying a concrete sentinel (of 519 scanned)
mask identically under the old rule and the new one — so this surfaces only on data that holds values close to its
own sentinel. Ask `pyramids.base._domain.is_no_data(arr, nodata, rtol=...)` directly if you want the old, looser
window.

**Coordinate axis arrays can differ in the last ULP.** `RasterBase.get_x_lon_dimension_array` /
`get_y_lat_dimension_array` moved from an element-wise accumulation to the shared `GeoTransform.x_axis` / `y_axis`,
which multiplies the index by the cell size instead of adding it repeatedly. The new values are the more accurate
ones — `10.35` where the walk produced `10.350000000000001` — but any golden file, doctest or notebook output that
pins the printed coordinate will move. Compare axis arrays with `numpy.allclose`, not with `==` or a repr.

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

## feature

### unreleased

**`pyramids.feature.bbox` no longer carries `Transformer` and `crs_from_user_input`.** Both were incidental
re-exports — names the module imported to implement `transform`, never advertised in its docs or `__all__`. The
reprojection itself moved down to `pyramids.base._bbox` so `pyramids.base` could stop importing `pyramids.feature`
(and, with it, geopandas), and its imports went with it. Nothing that the module documents changed:
`from pyramids.feature.bbox import Bbox, transform` still works and returns the same objects as before.

If you imported either name *through* `pyramids.feature.bbox`, import it from its own home instead:

```python
from pyproj import Transformer
from pyramids.base.crs import crs_from_user_input
```

## cli

### 0.47.0

**`pyramids calc` refuses a first input with no CRS.** The result cannot be georeferenced and pyramids will not
stamp a default; set a CRS on the input first. `pyramids georeference` is unaffected — its GCPs and `--gcp-crs`
replace the georeference wholesale.

## netcdf

### unreleased

**`NetCDF.variable_names` answers a different question, so the set of names changes.** Hard change, silent —
nothing raises and nothing warns, and both the membership and the order can move. The property used to hand
back the CF classification's own list; it now filters the store's declared list, which means it walks sub-groups
and it drops the arrays CF says are not data. `nc.variables` has the same keys, so it moves with it.

Two things change, and they pull in opposite directions:

- **Group-qualified names appear.** A grouped netCDF-4 store's sub-group arrays are enumerated now, named by
  their path with `/` as the separator. On the repository's own `none__35v__1d35__groups-nc4.nc` fixture the
  list goes from **1 name to 29**, 28 of which carry a `/`:

  ```python
  # Before
  nc.variable_names     # ['UTC_time']

  # After
  nc.variable_names     # ['UTC_time',
                        #  'mozaic_flight_2012030403540535_ascent/air_press',
                        #  'mozaic_flight_2012030403540535_ascent/CO', ...]   <- 29 in all
  ```

- **CF's non-data arrays leave.** A bounds array, a 2-D curvilinear coordinate field, an ancillary variable is
  still in the store and still readable, but it is no longer *enumerated* as a data variable. Measured over the
  26 netCDF fixtures in the repository: 18 lists are byte-identical, 8 changed, and **none changed order
  alone**. What left, by fixture: `lat_bnds` / `lon_bnds` / `time_bnds` (two fixtures), `lon_rho` / `lat_rho` /
  `Cs_r` / `h` (a curvilinear ROMS grid, 6 names → 2), `xc` / `yc`, `DQF` / `band_id` / `band_wavelength`,
  `expver`, and UGRID's `face_node_connectivity`.

Nothing became unreachable. Every name that left is still accepted by `get_variable`, and asking `variables`
for one says so rather than raising a bare `KeyError`:

```python
nc.variable_names                 # ['tos']            <- was ['lon_bnds', 'lat_bnds', 'time_bnds', 'tos']
nc.get_variable("lat_bnds")       # still works, shape (170, 2)
nc.variables["lat_bnds"]          # KeyError: "'lat_bnds' is not a data variable, so it is not a key of
                                  #  `variables`; read it with `get_variable('lat_bnds')`."
```

A group-qualified name works the same way, and a group can also be opened on its own:

```python
nc.get_variable("mozaic_flight_2012030403540535_ascent/CO")        # shape (74,)
nc.get_group("mozaic_flight_2012030403540535_ascent").variable_names
# ['air_press', 'CO', 'O3', 'altitude']
```

The order within the list is the store's declaration order rather than the classification's. That matters
beyond iteration: `_fan_out_eager` templates a container-wide result from the **first** spatial variable —
taking its geotransform, CRS, no-data and extra dimensions — so a container-wide `to_crs`, `resample` or `crop`
can template from a different variable than it did before, and the order propagates into `to_netcdf` and
`to_xarray` output.

What to do: if you iterated `variable_names` to convert, export or plot a whole store, re-measure what you get.
A grouped store now hands you every group's arrays under `/`-qualified names — filter on `"/" not in name` if
you want only the root's. A CF store hands you fewer names — name the bounds or ancillary arrays you need
explicitly and read them with `get_variable`. If you depended on the sorted order, sort explicitly:
`sorted(nc.variable_names)`.

**`NetCDF.to_xarray()` on a grouped store exports every group's arrays, under flattened names.** Hard change,
silent. It followed `variable_names`, so where that returned one name the export had one data variable. On the
same grouped fixture:

```python
# Before
list(nc.to_xarray().data_vars)    # ['UTC_time']

# After
list(nc.to_xarray().data_vars)    # ['UTC_time',
                                  #  'mozaic_flight_2012030403540535_ascent_air_press',
                                  #  'mozaic_flight_2012030403540535_ascent_CO', ...]   <- 9 in all
```

Three rules decide those keys, and they are worth knowing before you index the result:

- **The `/` becomes `_`.** An `xr.Dataset` is one flat namespace and netCDF forbids `/` in a variable name, so
  a path is flattened with the group kept as a prefix — `flight_a/CO` and `flight_b/CO` stay apart as
  `flight_a_CO` and `flight_b_CO`. Exporting the group path verbatim produced a Dataset that could not be
  written back at all.
- **The store's own name wins a collision.** Where a flattened sub-group name lands on a name the store really
  holds — a root array literally called `flight_a_CO`, or a dimension coordinate of that name — the root array
  keeps the plain key and the *sub-group* array takes `_2`, `_3`, …, whichever order the two happen to be
  enumerated in. A `UserWarning` lists each rename as `store name -> export name`; the name before the arrow is
  the one `get_variable` takes, and the renamed variable also carries it as a `pyramids_store_name` attribute,
  since the suffixed key alone cannot be turned back into it.
- **A variable whose dimensions clash is skipped, with a warning naming it.** An `xr.Dataset` has one size per
  dimension name, which a grouped store need not honour: the fixture above holds 29 arrays across eight flight
  groups that all declare a `recNum` of their own, so 9 are exported and the other 20 are named in a
  `UserWarning` telling you to read them with `get_variable()` instead.

If you keyed the exported Dataset, or round-tripped it through `to_netcdf`, re-derive the keys rather than
hard-coding them — and catch the warnings if you need to know what was renamed or skipped.

**A reprojected, resampled or cropped container reports its own grid, not the source's.** Hard change, silent.
`to_crs`, `resample` and `crop` fan out over a container and build a new one, and the result's `x` / `y` axis
arrays are what its georeference is read from — pyramids prefers a coordinate pair over the stored transform.
Two things about those axes were wrong, in two different arms, and both are fixed.

**The in-memory result's axes are `float64`.** They took the *data* variable's dtype, so reprojecting an
`int16` raster into EPSG:3857 wrote metre coordinates into an `int16` axis, where every one of them saturated.
`coards__4v__1d3-3d1__y-desc.nc` → `to_crs(3857)`:

```text
# Before                                          # After
x  dtype int16,  [-32768, -32768, -32768, …]      x  dtype float64, [-17788309.8, -17464393.7, …]
y  dtype int16,  [ 32767,  32767,  32767, …]      y  dtype float64, [ 13331117.2,  13007201.1, …]
geotransform (-32768.0, 0.0, 0, 32767.0, 0, -0.0) geotransform (-17950267.9, 323916.1, 0,
             <- a zero pixel size                              13493075.2, 0, -323916.1)
```

**The written file no longer carries the source's `lat` / `lon`.** `to_crs`, `resample` and `crop` with a
`path=` argument stream the result to disk, and that writer copied the source's spatial coordinate arrays into
the output beside the result's own `x` / `y`. Reopening such a file read the lon/lat pair in preference to the
stored transform and reported the *source's* degrees under the result's CRS. Reprojecting
`cf__7v__1d3-2d3-3d1__y-asc.nc` to EPSG:3857 and reading the file back:

```text
# Before                                          # After
epsg          3857                                epsg          3857
geotransform  (0.0, 2.0, 0, 90.0, 0, -1.0)        geotransform  (-20037477.8, 1054539.6, 0,
              <- degrees, under a metre CRS                      242528680.9, 0, -1054817.1)
arrays        […, 'lat', 'lat_bnds', 'lon',       arrays        […, 'lat_bnds', 'lon_bnds',
               'lon_bnds', 'x', 'y']                             'x', 'y']
```

A spatial axis describes the grid it was written for, and every operation that reaches here has changed that
grid, so the result derives its own `x` / `y` and no source axis is carried. Bounds arrays **are** still
carried, verbatim and on a bare dimension — netCDF allows a dimension with no coordinate variable — so a
`lat_bnds` in the output still holds the source's rows and no longer has a `lat` beside it to be read against.
That is deliberate: stale metadata next to an array is the lesser of the two costs, against a file that is
silently georeferenced wrong.

If you read a written file's `lat` / `lon` arrays, read `x` / `y` instead, or take the coordinates from the
container's `geotransform`. If you consumed the carried `lat_bnds` / `lon_bnds`, note that they describe the
*source* grid and always did — they are unchanged, only the axis that used to sit beside them is gone. And if
you compared a reprojected container's `geotransform` against a stored value, re-derive it: on a raster whose
data dtype could not hold the new coordinates, the old number was not merely imprecise but degenerate.

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
