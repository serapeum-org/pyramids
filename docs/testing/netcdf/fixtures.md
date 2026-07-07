# NetCDF test fixtures

The NetCDF test suite is driven by a curated set of small `.nc` files under `tests/data/netcdf/` (with parallel
copies under `examples/data/netcdf/` for the docs notebooks). Each fixture is named so its structure — and, where it
matters, its CRS and Y-axis orientation — is legible from the filename alone.

## Naming convention

```
<convention>__<Nv>v__<rank-breakdown>__[<crs>]__[<feature>]__y-<asc|desc>.nc
```

| Segment                         | Meaning                                                                                                                                                                                                                    |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `convention`                    | CF-conventions family: `cf`, `coards`, `none` (no `Conventions` attr), `ugrid`                                                                                                                                             |
| `<Nv>v`                         | total variable count — **including coordinate variables**                                                                                                                                                                  |
| `rank-breakdown`                | count of variables at each dimensionality, e.g. `1d3-3d1` = 3 one-D + 1 three-D (sums to `Nv`)                                                                                                                             |
| `crs` *(optional)*              | `geos` (geostationary), `geog` (geographic lat/lon), `proj` (projected, e.g. UTM)                                                                                                                                          |
| `feature` *(optional)*          | structural tags: `curv` (curvilinear), `stag` (staggered), `str` (string vars), `nc4` (netCDF-4), `groups-nc4` (has groups), `scaleoffset` (packed data)                                                                   |
| `y-asc` / `y-desc` *(optional)* | Y-axis direction — `asc` = row 0 is the south edge (data stored south→north), `desc` = row 0 is the north edge. Present only when the file has a genuine 1-D spatial Y axis (omitted for curvilinear / no-Y / mesh files). |

## Fixtures

| File                                           | Conv.       |   Vars | Rank breakdown           | CRS             | Y-axis   | nc4   | What it exercises                                                                                                                |
|------------------------------------------------|-------------|--------|--------------------------|-----------------|----------|-------|----------------------------------------------------------------------------------------------------------------------------------|
| `cf__4v__1d3-3d1__proj__y-desc.nc`             | CF-1.6      |      4 | 3×1D, 1×3D               | projected       | desc     | ✓     | Projected (UTM) grid, north-up; pyramids-written — **projected-descending** orientation case                                     |
| `cf__5v__1d4-3d1__geog__y-desc.nc`             | CF-1.7      |      5 | 4×1D, 1×3D               | geographic      | desc     | ✓     | ERA5 `t2m(time,lat,lon)`; **geographic-descending** orientation case                                                             |
| `cf__5v__1d4-4d1__geog__y-desc.nc`             | CF-1.6      |      5 | 4×1D, 1×4D               | geographic      | desc     | ✓     | ERA5 pressure levels — 4-D `(time,level,lat,lon)`, geographic-descending                                                         |
| `cf__5v__1d4-4d1__y-asc.nc`                    | CF-1.6      |      5 | 4×1D, 1×4D               | geographic      | asc      | ✓     | pyramids-written 4-D cube (round-trip)                                                                                           |
| `cf__6v__1d2-2d4__geog__y-asc.nc`              | CF-1.5      |      6 | 2×1D, 4×2D               | geographic      | asc      |       | NOAH precipitation, four `Band(lat,lon)` vars; **geographic-ascending** orientation case (flipped on read)                       |
| `cf__7v__1d3-2d3-3d1__y-asc.nc`                | CF-1.0      |      7 | 3×1D, 3×2D, 1×3D         | geographic      | asc      |       | Single CF data var `tos(time,lat,lon)` + coords/bounds                                                                           |
| `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`        | CF-1.4      |      8 | 3×1D, 3×2D, 1×3D, 1×4D   | geographic      | —        | ✓     | Curvilinear + staggered grid                                                                                                     |
| `cf__9v__1d7-2d2__geos__y-asc.nc`              | CF-1.7      |      9 | 7×1D, 2×2D               | geostationary   | asc      | ✓     | GOES-16 ABI, projected scan-angle Y, packed `CMI` — **projected-ascending** case and the #705 windowed-read regression fixture   |
| `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`           | CF-1.0      |     12 | 4×1D, 5×2D, 2×3D, 1×4D   | geographic      | asc      |       | Mix of all dimensionalities (CCSM sample): 3-D `pr`/`tas`, 4-D `ua`                                                              |
| `cf__20v__1d3-3d17__y-desc.nc`                 | CF-1.0      |     20 | 3×1D, 17×3D              | geographic      | desc     | ✓     | Many 3-D vars (17 packed surface fields), ERA-40 subset                                                                          |
| `cf__40v__1d28-2d9-3d3__nc4.nc`                | CF-1.6      |     40 | 28×1D, 9×2D, 3×3D        | geographic      | —        | ✓     | Large variable count, netCDF-4, no plain spatial Y                                                                               |
| `cf__48v__1d17-3d21-4d10__y-asc.nc`            | CF-1.0      |     48 | 17×1D, 21×3D, 10×4D      | geographic      | asc      | ✓     | CAM init — 10 `(time,lev,lat,lon)` 4-D + 21 3-D                                                                                  |
| `coards__4v__1d2-2d2__scaleoffset__y-asc.nc`   | COARDS/CF   |      4 | 2×1D, 2×2D               | geographic      | asc      |       | Two vars with `scale_factor`/`add_offset` — packed-data (`unpack=True`) reads                                                    |
| `coards__4v__1d3-3d1__y-desc.nc`               | COARDS      |      4 | 3×1D, 1×3D               | geographic      | desc     | ✓     | COARDS `air(time,lat,lon)`, int16-packed; **geographic-descending** orientation case                                             |
| `coards__5v__1d4-4d1__y-desc.nc`               | COARDS      |      5 | 4×1D, 1×4D               | geographic      | desc     | ✓     | Single 4-D var `rhum(time,level,lat,lon)`, multi-level                                                                           |
| `none__1v__1d1.nc`                             | none        |      1 | 1×1D                     | none            | —        |       | Bare single 1-D variable                                                                                                         |
| `none__4v__1d1-2d2-3d1__curv.nc`               | none        |      4 | 1×1D, 2×2D, 1×3D         | geographic      | —        | ✓     | Curvilinear grid (RASM-like), 2-D coordinates                                                                                    |
| `none__4v__1d3-3d1__geog__y-asc.nc`            | none        |      4 | 3×1D, 1×3D               | geographic      | asc      | ✓     | MSWEP precip, geographic-ascending, **no `Conventions` attr**                                                                    |
| `none__5v__1d2-2d2-3d1__curv.nc`               | none        |      5 | 2×1D, 2×2D, 1×3D         | geographic      | —        | ✓     | Curvilinear grid (ROMS-like)                                                                                                     |
| `none__11v__1d11.nc`                           | none        |     11 | 11×1D                    | none            | —        |       | Multiple 1-D vars — aircraft track time series (no spatial Y axis)                                                               |
| `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`      | none        |     17 | 1×1D, 5×2D, 6×3D, 5×4D   | geographic      | —        | ✓     | Staggered grid + string variables                                                                                                |
| `none__35v__1d35__groups-nc4.nc`               | none        |     35 | 35×1D                    | none            | —        | ✓     | **7 groups**, netCDF-4 — group traversal                                                                                         |
| `none__111v__1d96-2d13-3d2__str.nc`            | (AWIPS)     |    111 | 96×1D, 13×2D, 2×3D       | geographic      | —        |       | Many 1-D station-obs vars + char 2-D fields (no spatial Y axis)                                                                  |
| `ugrid__1v__1d1.nc`                            | none        |      1 | 1×1D                     | none            | —        |       | UGRID unstructured mesh — single var                                                                                             |
| `ugrid__1v__3d1.nc`                            | none        |      1 | 1×3D                     | geographic      | —        | ✓     | UGRID mesh — 3-D var                                                                                                             |
| `ugrid__6v__1d5-2d1.nc`                        | MPAS        |      6 | 5×1D, 1×2D               | geographic      | —        | ✓     | MPAS-convention unstructured mesh                                                                                                |

*"Vars" and the rank breakdown count coordinate variables. "Y-axis" is blank for curvilinear (2-D coordinate) files,
mesh files, and files with no spatial Y axis, where a single ascending/descending direction is not meaningful. The
"CRS" column is the **detected** CRS pyramids resolves (most lat/lon CF/COARDS grids read as `geographic`,
EPSG:4326); the filename `crs` tag is applied only to the files a test cares about (`geos` for #705, `geog`/`proj`
for the orientation 2×2), so a file can be `geographic` here without a `geog` tag in its name.*

## The Y-orientation 2×2

`tests/netcdf/spatial/test_y_orientation.py::TestFastPathOrientationAllCases` verifies the interaction of
**CRS type × Y-direction** — an ascending Y axis (row 0 = south) must be flipped to north-up, a descending one is
kept, and GDAL's classic driver only auto-flips a recognised geographic latitude (not a projected
`projection_y_coordinate`). The parametrized cases use four on-disk fixtures; the projected-descending cell has no
on-disk fixture in the test and is covered by a UTM grid generated at runtime (the `projected_descending_nc`
fixture):

|                | ascending (→ **flip**)                   | descending (→ **keep**)                                                     |
|----------------|------------------------------------------|-----------------------------------------------------------------------------|
| **projected**  | `cf__9v__1d7-2d2__geos__y-asc.nc` (GOES) | UTM grid generated at runtime (`projected_descending_nc`)                   |
| **geographic** | `cf__6v__1d2-2d4__geog__y-asc.nc` (NOAH) | `cf__5v__1d4-3d1__geog__y-desc.nc` (ERA5), `coards__4v__1d3-3d1__y-desc.nc` |

Matching on-disk fixtures that carry the same orientation but are **not** used by this parametrized test:
`cf__4v__1d3-3d1__proj__y-desc.nc` (projected-descending) and `none__4v__1d3-3d1__geog__y-asc.nc`
(MSWEP, geographic-ascending).

## Related tests

- `tests/netcdf/spatial/test_y_orientation.py` — Y-axis orientation across the 2×2 above (flip decision, north-up
  geotransform, byte-identity of the fast classic-driver read vs the multidim view).
- `tests/netcdf/spatial/test_windowed_read_705.py` — the geostationary/chunked windowed-read crash (#705) and the
  fast-path fallback branches (uses `cf__9v__1d7-2d2__geos__y-asc.nc`).
- `tests/netcdf/samples/` — structural coverage (conventions, dimensions, groups, curvilinear, string vars) over the
  full fixture set.

## Provenance

The original source filenames (e.g. `tos_O1_2001-2002.nc`, `air_temperature.nc`, a GOES-16 ABI granule) and their
download/build steps are recorded in `tests/data/netcdf/README.md`.
