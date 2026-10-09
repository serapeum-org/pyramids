# The cube as a nested dict

Export a cube's whole structure to plain Python objects and rebuild it from them. It needs a
multidimensional open — reopen with `open_as_multi_dimensional=True` if the cube was opened classically.

Every other export goes to a
*file* or a *foreign object* — `to_file`, `to_zarr`, `to_kerchunk`, `to_dataframe`, `to_xarray`, `to_stac_item`.
This is the one that goes to nothing but `dict`, `list`, `str` and numbers, which is what inspection, diffing two
cubes, JSON transport and asserting on structure in a test all want.

```python
payload = cube.to_dict()           # values included
structure = cube.to_dict(data=False)   # structure only: no read_array call
rebuilt = NetCDF.from_dict(payload)
```

## The schema is a superset of xarray's

`dims`, `coords`, `data_vars` and `attrs` keep their xarray meanings, so a reader that knows only xarray's
schema finds what it expects. Everything GDAL needs rides under one extra `pyramids` key:

| key | what it carries |
|---|---|
| `schema` | the schema version, refused by `from_dict` if it is newer than this reader |
| `epsg` | the CRS, `None` only under `data=False` |
| `geotransform` | the six affine coefficients |
| `spatial_dims` | the `(y, x)` pair every variable in the payload shares |
| `no_data_value` | the sentinel per variable |

`coords` carries the **band** dimensions only — the spatial axes are the `geotransform` rather than coordinate
arrays, so an xarray-only reader of this payload gets a `Dataset` with no spatial coordinates. Under
`data=False` both variables and coordinates report `dims`/`dtype`/`shape`/`attrs` and no `data`.

xarray's schema has **no slot for a CRS or a geotransform** — a `crs` attribute survives there only because it
happens to ride in `attrs`. For a GDAL-backed cube the georeferencing *is* the object, so matching xarray exactly
would have made the round trip silently drop it.

The payload records *pyramids'* dimension layout — band axes first, `(y, x)` last — which is the shape
`read_array` returns. A store's own declared order is not always that: variable `U` of the CAM/CESM fixture in
this repo declares `['time', 'lat', 'lev', 'lon']`, interleaving a spatial axis between two band axes.

## What round-trips, and what does not

Round-trips: dimensions and their order, a band dimension's coordinates **and their CF attributes** (so a time
axis keeps its units and calendar), variable arrays and dtypes, the CRS, the geotransform, the no-data sentinels
and the spatial dimension names.

When variables sit on *different* band axes, `add_variable` leaves the later ones' coordinate map empty upstream
of this module, so those axes arrive already unstamped and `from_dict` warns that it rebuilt them so.

Does **not**, in each case because `from_array` is the only constructor and takes neither:

- **Per-variable attributes.** `to_dict` reports every one it finds — 32 on this repo's
  `cf__5v__1d4-3d1__geog__y-desc.nc` — and the rebuilt variable carries only the `grid_mapping` that
  `from_array` regenerates. There is no per-variable attribute setter to restore them through.
- **Arbitrary global attributes.** Globals reach `from_array` as a `CFAttributes`, a fixed set of CF fields
  rather than a free mapping, so the CF-recognised ones are restored and the rest dropped.
- **Non-gridded variables.** They have no raster plane, so `from_array` cannot rebuild them; `to_dict` drops
  them with a warning naming each. 12 of the 43 variables in the CAM/CESM fixture are of this kind.

A no-data **sentinel** also widens to a float for an integer cube (`-1` becomes `-1.0`); the array dtype itself
round-trips exactly.

Because of the attribute losses above, `to_dict(a) == to_dict(from_dict(to_dict(a)))` is **not** an identity —
the two payloads differ in `attrs` and in each variable's `attrs`, while `dims`, `coords`, `shape`, `dtype` and
every value agree. Diff on those keys rather than on the whole dict.

## JSON

Python's `json` emits `NaN` and `Infinity` and reads them back, so a payload round-trips through
`json.dumps`/`loads` between Python processes. Those tokens are **not** RFC-8259 JSON, so `JSON.parse`,
`serde_json` and strict Python parsers reject them: a gappy raster's payload is Python-JSON, not wire-JSON.
Convert the non-finite cells before sending one somewhere strict.

## API

::: pyramids.netcdf.dict_io
    options:
      members:
        - cube_to_dict
        - cube_from_dict
      show_root_heading: false
      show_source: true
