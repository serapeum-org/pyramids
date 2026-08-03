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
- The canonical `abfss://<filesystem>@<account>.dfs.core.windows.net/<path>` form is understood, but GDAL
  takes the storage account from configuration rather than from the URL. If the URL names an account,
  `AZURE_STORAGE_ACCOUNT` must be set to match it — a mismatch, or naming an account with none configured,
  raises instead of silently reading a different account.

**`/vsiadls/` is now recognised as a remote, network-backed path.** Previously `is_remote()` and
`is_network_backed()` both returned `False` for a Gen2 path, so it was classified as a local file — opening still
worked, but anything reasoning about the path (credential handling, archive chaining) got the wrong answer.

**Archive chaining now covers every network handler, including for raw `/vsi*` paths.** A raster inside a zip
or tar is rewritten to `/vsizip/<handler>/...` for `/vsiadls/`, `/vsioss/` (Alibaba), `/vsiswift/` (OpenStack),
`/vsihdfs/` and `/vsiwebhdfs/`, not only for S3, Google Cloud, Azure Blob and `/vsicurl/`. Four of those have no
URL scheme at all, so the raw `/vsi*` spelling is the only way to name them — paths already in `/vsi*` form now
go through the chaining too, where before they were returned untouched. If you were chaining by hand, the manual
prefix is no longer needed, and a hand-built `/vsizip//vsioss/...` still works unchanged.

**Credentials for `abfs://` are the same `AZURE_STORAGE_*` family**, with one exception: GDAL's anonymous-access
switch is per-handler, so a workflow relying on `AZURE_NO_SIGN_REQUEST` against `/vsiaz/` needs the `/vsiadls/`
equivalent once `abfs://` routes there.

**`Dataset.epsg` no longer reports EPSG:4326 for a raster that has no CRS.** It previously substituted WGS 84
whenever the projection was empty, so an ungeoreferenced grid claimed a georeference it did not have — and that
claim propagated into `to_file`, `to_crs`, `bounds` and alignment checks. It now returns `None`, matching GDAL,
rasterio, rioxarray and geopandas, none of which substitute a default.

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

### unreleased

**`pyramids calc` refuses a first input with no CRS.** The result cannot be georeferenced and pyramids will not
stamp a default; set a CRS on the input first. `pyramids georeference` is unaffected — its GCPs and `--gcp-crs`
replace the georeference wholesale.

## netcdf

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
| `ColourOpts` | deprecated | `ColorOpts` |
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

**7. `ColourOpts` -> `ColorOpts`.**

```python
# Before
from pyramids.netcdf import ColourOpts
opts = ColourOpts(cmap="viridis")
# After
from pyramids.netcdf import ColorOpts
opts = ColorOpts(cmap="viridis")
```

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
(`pyramids.netcdf.Container`). If you `from pyramids.netcdf import Variable` in a module that also uses `xarray`,
the name collides conceptually with `xarray.Variable` — prefer the namespace-qualified form, or alias on import:

```python
from pyramids.netcdf import Variable as NcVariable
```

#### Migration checklist

1. Pin the new `pyramids` version in your dependencies.
2. Run your suite under `-W error::DeprecationWarning` and fix everything it flags (items 5-10 above).
3. Fix the hard-behavior-change items (1-4) — they do not warn: search for `type(x) is NetCDF`, `type()` /
   `Dataset` checks on `subset()` results, `CFInfo` mutation, and unrecognised `LabeledDataset` engines.
4. Done — `isinstance(x, NetCDF)` keeps working, so most code needs no change.
