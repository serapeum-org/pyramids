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
