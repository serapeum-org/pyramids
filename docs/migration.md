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

## netcdf

### Unreleased

> Status: the consolidation (PR #612) is merged to `main`; the `Container` / `Variable` type split (PR #625) is
> pending. Both target the next release after `0.36.0`. The exact version number is assigned by commitizen at
> release time, so this section is labelled `Unreleased` until then.

#### Hard behavior changes — must fix (no warning)

| Change | Old behavior | New behavior | Migration |
|--------|--------------|--------------|-----------|
| `CFInfo` is frozen (API-8) | `cf_info.x = ...` worked | mutation raises `dataclasses.FrozenInstanceError` | build a new instance or use `dataclasses.replace(cf_info, x=...)` |
| `LabeledDataset` engine validation (API-11) | an unknown `engine=` was silently accepted | raises `ValueError` unless `engine` is one of `("zarr", "netcdf", "netcdf4", "hdf5", "h5netcdf")` or `None` | pass a valid engine |
| `NetCDF.subset()` return type (API-2) | returned a plain `Dataset` | returns a `NetCDF` (a `Variable` after PR #625) | usually fine (superset API); only matters if you did `type(x) is Dataset` |
| `type(x) is NetCDF` (PR #625, pending) | `True` for opened files | `False` — the object is a `Container` / `Variable` subclass | use `isinstance(x, NetCDF)` |

#### Deprecations — still work, warn now, fix at leisure

| Deprecated | Replacement |
|------------|-------------|
| `nc.get_variable_names()` | the `nc.variable_names` property |
| `ColourOpts` | `ColorOpts` (`ColourOpts` is now a deprecated subclass that still compares equal by value) |
| `MetaData` / `DimMetaData` (in `pyramids.netcdf.dimensions`) | `ClassicDimMetadata` / `ClassicDimensionInfo` |
| `to_kerchunk(..., backend="kerchunk")` / `combine_kerchunk(..., backend="kerchunk")` | `backend="legacy"` |
| `NetCDF(gdal_dataset)` direct construction (PR #625, pending) | `NetCDF.read_file(...)` / `NetCDF.create_from_array(...)` (returns a `Container`); `container.get_variable(...)` (returns a `Variable`) |

These all still function; they emit a `DeprecationWarning` and are on a one-major-version removal path.

#### Renames where the old name still works (no rush)

- `_LabeledArray` -> `LabeledArray`, `_apply_unpack` -> `apply_unpack` (API-9). The underscore names are kept as
  aliases, so old imports work; prefer the public names.
- Private module renames: `_kerchunk.py` -> `_kerchunk_facade.py`, `_kerchunk_native.py` -> `_kerchunk_builder.py`.
  These are underscore-prefixed internals — do not import them directly; use `NetCDF.to_kerchunk` /
  `NetCDF.combine_kerchunk`. If you imported the private modules, update the paths.

#### New, opt-in capabilities (not breaking)

- **Typed dispatch (PR #625):** branch on `isinstance(x, Container)` vs `isinstance(x, Variable)` instead of
  inspecting `is_subset` / `band_count`. Import from `pyramids.netcdf` or `pyramids.netcdf.variable`.
- **Cloud read tuning (ARC-16):** `CloudConfig(vsicurl_tuning=True, curl_cache_size=...)` enables the fast
  single-file `/vsicurl/` read preset.

#### Naming gotcha with the type split (PR #625)

The new public types are named `Container` and `Variable`. They read cleanly when namespace-qualified
(`pyramids.netcdf.Container`). If you `from pyramids.netcdf import Variable` in a module that also uses `xarray`,
the name collides conceptually with `xarray.Variable` — prefer the namespace-qualified form, or alias on import:

```python
from pyramids.netcdf import Variable as NcVariable
```

#### Migration checklist

1. Pin the new `pyramids` version in your dependencies.
2. Run your suite under `-W error::DeprecationWarning` and fix everything it flags (the deprecation + rename
   tables above).
3. Search for the hard-behavior-change items — `type(x) is NetCDF`, `type()` / `Dataset` checks on `subset()`
   results, and any `CFInfo` mutation — and fix those (they do not warn).
4. Done — `isinstance(x, NetCDF)` keeps working, so most code needs no change.
