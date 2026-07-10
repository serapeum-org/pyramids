# NetCDF sample files — a catalog of real-world NetCDF shapes

A curated set of **real** NetCDF files downloaded from public sources, chosen to cover every common
shape/convention a reader has to handle: single vs many variables; pure-1D / 2D / 3D / 4D and mixed
dimensionality; CF and COARDS conventions; curvilinear (2-D coordinate) grids; staggered grids; hierarchical
groups; and unstructured **UGRID** meshes.

Nothing here is synthetic — each file is a published example used by the wider community (Unidata's netCDF
example suite, the `pydata/xarray-data` tutorial set, and `UXARRAY`'s UGRID mesh fixtures).

> These live under `tests/data/netcdf/` as **committed** test fixtures (small, deflate-compressed;
> ~22 MB total). They are the canonical sample set for the netcdf-subpackage test suite.

## Naming convention

Files are named to describe their **structure**, not their scientific content. The pattern is:

```
<convention>__<Nv>__<dim-histogram>[__<features>].nc
```

Four `__`-delimited fields (a single `-` separates items inside a field):

1. **convention** — one of `cf` · `coards` · `ugrid` · `none`. `none` means no CF/COARDS/UGRID convention is
   declared (the file may still carry another convention such as AWIPS; see the catalog notes).
2. **`<N>v`** — total number of variables. This counts **all arrays** GDAL's multidimensional API enumerates:
   data variables *plus* coordinate variables, bounds, and connectivity arrays. Example: `12v`.
3. **dim-histogram** — how many variables exist at each rank, zeros omitted: `1d4-2d5-3d2-4d1` reads as
   "four 1-D, five 2-D, two 3-D, one 4-D". The counts always sum to `<N>`.
4. **features** *(optional)* — any of: `curv` (curvilinear / 2-D coordinate grid) · `stag` (staggered grid) ·
   `groups` (netCDF-4 hierarchical groups) · `nc4` (netCDF-4 container) · `str` (character/string variables) ·
   `mesh` (unstructured mesh; implied by `ugrid`).

Parse a name back into its fields with `name.split("__")`. Names sort naturally by convention, then by
variable count. The structural signature is unique across this set, so there are no collisions (if two ever
collided, a `-b` / `-c` disambiguator would be appended).

## Sources

- **Unidata netCDF example files** — `https://archive.unidata.ucar.edu/software/netcdf/examples/`
- **xarray tutorial data** (`pydata/xarray-data`) — `https://github.com/pydata/xarray-data`
- **UXARRAY UGRID meshfiles** — `https://github.com/UXARRAY/uxarray/tree/main/test/meshfiles/ugrid`

`ndim` below counts data + coordinate arrays as GDAL's multidimensional API enumerates them (so coordinate
variables, bounds, and connectivity arrays are included). "dim-coords" = number of 1-D variables named like
their own dimension (the classic CF coordinate variables).

## Catalog

| File                                      | Size    | Convention | #vars | ndim histogram         | Notable structure                                                                                       |
|-------------------------------------------|---------|------------|-------|------------------------|---------------------------------------------------------------------------------------------------------|
| `none__1v__1d1.nc`                        | 0.04 MB | (none)     | 1     | 1×1D                   | **Single variable**, trivial (`var1(dim1)`)                                                             |
| `cf__7v__1d3-2d3-3d1__y-asc.nc`                  | 2.9 MB  | CF-1.0     | 7     | 3×1D, 3×2D, 1×3D       | **Single data variable** `tos(time,lat,lon)` + coords/bounds                                            |
| `coards__4v__1d3-3d1__y-desc.nc`                  | 0.2 MB  | COARDS     | 4     | 3×1D, 1×3D             | Single data variable `air(time,lat,lon)`, int16-packed; COARDS (100 steps, deflate)                     |
| `cf__6v__1d2-2d4__geog__y-asc.nc`                 | 4.2 MB  | CF-1.5     | 6     | 2×1D, 4×2D             | NOAH precipitation, four `Band(lat,lon)` vars; latitude stored **south→north** (Y-flip case)            |
| `cf__9v__1d7-2d2__geos__y-desc.nc`                | 0.3 MB  | CF-1.7     | 9     | 7×1D, 2×2D             | GOES-16 ABI, geostationary CRS; `int16`-packed radian scan-angle `y` with **negative** `scale_factor`   |
| `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`             | 2.8 MB  | CF-1.0     | 12    | 4×1D, 5×2D, 2×3D, 1×4D | **Mix of all dims**, CF; `pr`/`tas`(3D), `ua`(4D), `area`/`msk_rgn`(2D)                                 |
| `cf__20v__1d3-3d17__y-desc.nc`                    | 1.7 MB  | CF-1.0     | 20    | 3×1D, 17×3D            | **Many 3-D variables** (17 int16-packed surface fields `(time,lat,lon)`), CF (12 steps)§                |
| `cf__48v__1d17-3d21-4d10__y-asc.nc`              | 3.1 MB  | CF-1.0     | 48    | 17×1D, 21×3D, 10×4D    | **Mix of all** — CAM init; 10 `(time,lev,lat,lon)` 4-D + 21 3-D + coords (6 levels)¶                    |
| `coards__5v__1d4-4d1__y-desc.nc`                  | 0.2 MB  | COARDS     | 5     | 4×1D, 1×4D             | **Single 4-D variable** `rhum(time,level,lat,lon)`, int16-packed; COARDS multi-level (12×4×37×72)‡      |
| `cf__40v__1d28-2d9-3d3__nc4.nc`           | 0.24 MB | CF-1.6     | 40    | 28×1D, 9×2D, 3×3D      | netCDF-4; satellite L2 with averaging kernels; mixed 1D/2D/3D                                           |
| `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`   | 2.8 MB  | CF-1.4     | 8     | 3×1D, 3×2D, 1×3D, 1×4D | **Curvilinear** ROMS ocean grid (rho-point only in this subset); mix incl. 4-D (6 levels)※              |
| `none__4v__1d1-2d2-3d1__curv.nc`          | 1.7 MB  | (none)     | 4     | 1×1D, 2×2D, 1×3D       | **Curvilinear / 2-D coordinates** (`xc(y,x)`, `yc(y,x)`), `Tair(time,y,x)` (6 steps)‖                   |
| `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` | 3.8 MB  | (none)     | 17    | 1×1D, 5×2D, 6×3D, 5×4D | **Reduced WRF subset** — one var per distinct shape; `U/V/W`/`MAPFAC_*` on `*_stag` dims; char `Times`† |
| `none__5v__1d2-2d2-3d1__curv.nc`          | 1.6 MB  | (none)     | 5     | 2×1D, 2×2D, 1×3D       | **Multiple 2-D** image bands (McIDAS satellite); 2-D `lat`/`lon` (deflate, full res)                    |
| `none__111v__1d96-2d13-3d2__str.nc`       | 0.27 MB | (AWIPS)    | 111   | 96×1D, 13×2D, 2×3D     | **Many 1-D variables** — station obs over `recNum`; char 2-D fields                                     |
| `none__11v__1d11.nc`                      | 0.01 MB | (none)     | 11    | 11×1D                  | **Multiple 1-D variables** — aircraft track time series                                                 |
| `none__35v__1d35__groups-nc4.nc`          | 0.09 MB | (none)     | 35    | 35×1D                  | **Hierarchical groups** (netCDF-4); 1-D arrays nested under flight groups                               |
| `ugrid__6v__1d5-2d1.nc`                   | 0.02 MB | UGRID*     | 6     | 5×1D, 1×2D             | **UGRID mesh topology** — `node_lon`/`face_lon`, `face_node_connectivity(n_face,n_max_face_nodes)`      |
| `ugrid__1v__3d1.nc`                       | 0.01 MB | UGRID*     | 1     | 1×3D                   | **UGRID data**, 3-D on an unstructured mesh: `multi_dim_data(time,lev,n_face)`                          |
| `ugrid__1v__1d1.nc`                       | 0.04 MB | UGRID*     | 1     | 1×1D                   | **UGRID/unstructured data** on a cubed-sphere SE grid: `psi(ncol)`                                      |

\* The UGRID files come from UXARRAY's `meshfiles/ugrid/` set. GDAL's multidim view flattens the scalar
`mesh_topology` dummy variable, so the `Conventions` attribute may read empty/`MPAS`; the unstructured nature
is unmistakable from the `n_node`/`n_face`/`ncol` dimensions and the `*_connectivity` arrays.

† `none__17v__…` is a **reduced subset** of the 80-variable `wrfout_v2_Lambert.nc`: the original had 63
redundant variables that all share a shape already covered by another. One representative per distinct
`(rank × dimension-set × dtype × role)` signature was kept (all four ranks, all three dtypes
`float32`/`int32`/`char`, every mass/u/v/w/soil stagger, and the `XLAT`/`XLONG` geolocation pair). The `Time`
dimension is then cropped to 3 steps and the 3-D/4-D arrays deflate-compressed, shrinking the file from 82 MB
to ~3.8 MB with no loss of reader-code-path coverage.

‡ `coards__5v__…` is shrunk from the published `rhum.2003.nc` (365 daily steps × 8 levels × 73×144, ~61 MB)
to a tiny fixture (~0.2 MB): the first 12 time steps and 4 levels, the grid decimated by 2 to 37×72, and
`rhum` stored chunked + deflate-compressed. Nothing structural changed — the four 1-D coordinate axes and the
single `int16`-packed 4-D `rhum` variable (with its `scale_factor`/`add_offset`/`_FillValue`) are intact, only
smaller.

§ `cf__20v__…` is shrunk from the published `ECMWF_ERA-40_subset.nc` (62 × 73×144, ~22 MB) by cropping `time`
to 12 steps and deflate-compressing, to ~1.7 MB. All 17 `int16`-packed 3-D fields, the three coordinate axes,
their packing, and CF-1.0 are intact; only the `time` dimension is shorter.

¶ `cf__48v__…` is shrunk from the published CAM init file (`time=1`, 26 levels, 64×128, ~18 MB) by cropping
the vertical to 6 levels (interface levels `ilev` to 7) and deflate-compressing, to ~3.1 MB. All 48 variables,
the full 1-D/3-D/4-D + `float64`/`int32`/`char` mix, the `lev`/`ilev` pairing, and CF-1.0 are intact; `time`
is length 1 and cannot be cropped.

‖ `none__4v__…` is shrunk from the published `rasm.nc` (`time=36`, 205×275, ~17 MB) by cropping `time` to 6
steps and deflate-compressing, to ~1.7 MB. The curvilinear layout is intact — `Tair` plus the 2-D `xc(y,x)` /
`yc(y,x)` coordinate arrays at full 205×275 resolution; only `time` is shorter.

※ `cf__8v__…` (ROMS) is shrunk from the published `ROMS_example.nc` (already deflate-compressed, ~9.4 MB) by
cropping the vertical `s_rho` to 6 levels, to ~2.8 MB; both time steps and the full 191×371 grid (with its 2-D
`lat_rho`/`lon_rho` curvilinear coords) are kept. Note: despite the `stag` tag, this published subset contains
only rho-point dimensions — there are no `u`/`v` staggered dims (the tag reflects ROMS being an Arakawa-C model
in general).

## Name ↔ source mapping (traceability)

The structural names above replace the original published filenames. This table records the mapping so each
file's provenance is never lost.

| Structural name                              | Original filename                       | Source            |
|----------------------------------------------|-----------------------------------------|-------------------|
| `none__1v__1d1.nc`                           | `testrh.nc`                             | Unidata examples  |
| `none__11v__1d11.nc`                         | `WMI_Lear.nc`                           | Unidata examples  |
| `none__35v__1d35__groups-nc4.nc`             | `test_hgroups.nc`                       | Unidata examples  |
| `none__111v__1d96-2d13-3d2__str.nc`          | `madis-sao.nc`                          | Unidata examples  |
| `none__5v__1d2-2d2-3d1__curv.nc`             | `IMAGE0002.nc`                          | Unidata examples  |
| `none__4v__1d1-2d2-3d1__curv.nc`             | `rasm.nc`                               | xarray-data       |
| `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`    | `wrfout_v2_Lambert.nc` (reduced, 17/80) | Unidata examples  |
| `coards__4v__1d3-3d1__y-desc.nc`                     | `air_temperature.nc`                    | xarray-data       |
| `cf__6v__1d2-2d4__geog__y-asc.nc`                    | CreateCopy of `tests/data/geotiff/noah-precipitation` | this repo (GDAL)  |
| `cf__9v__1d7-2d2__geos__y-desc.nc`                   | `OR_ABI-L2-CMIPM1-M6C13_G16_s20241801200284…` | NOAA GOES-16 (AWS) |
| `coards__5v__1d4-4d1__y-desc.nc`                     | `rhum.2003.nc`                          | Unidata examples  |
| `cf__7v__1d3-2d3-3d1__y-asc.nc`                     | `tos_O1_2001-2002.nc`                   | Unidata examples  |
| `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`                | `sresa1b_ncar_ccsm3-example.nc`         | Unidata examples  |
| `cf__20v__1d3-3d17__y-desc.nc`                       | `ECMWF_ERA-40_subset.nc`                | Unidata examples  |
| `cf__48v__1d17-3d21-4d10__y-asc.nc`                 | `cami_0000-09-01_64x128_L26_c030918.nc` | Unidata examples  |
| `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`      | `ROMS_example.nc`                       | xarray-data       |
| `cf__40v__1d28-2d9-3d3__nc4.nc`              | `OMI-Aura_L2-example.nc`                | Unidata examples  |
| `ugrid__6v__1d5-2d1.nc`                      | `quad-hexagon/grid.nc`                  | UXARRAY meshfiles |
| `ugrid__1v__3d1.nc`                          | `quad-hexagon/multi_dim_data.nc`        | UXARRAY meshfiles |
| `ugrid__1v__1d1.nc`                          | `outCSne30/outCSne30_vortex.nc`         | UXARRAY meshfiles |

## Scenario index — which file to use for what

- **One variable**: `none__1v__1d1.nc` (bare 1-D), `cf__7v__1d3-2d3-3d1__y-asc.nc` (one CF data var `tos` + coords),
  `coards__4v__1d3-3d1__y-desc.nc`, `ugrid__1v__1d1.nc` / `ugrid__1v__3d1.nc` (one var on a mesh).
- **Multiple variables**: `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`, `cf__20v__1d3-3d17__y-desc.nc`, `cf__48v__1d17-3d21-4d10__y-asc.nc`,
  `cf__40v__1d28-2d9-3d3__nc4.nc`, `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`,
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`, `none__111v__1d96-2d13-3d2__str.nc`,
  `none__35v__1d35__groups-nc4.nc`.
- **Multiple 1-D variables**: `none__11v__1d11.nc` (aircraft track), `none__111v__1d96-2d13-3d2__str.nc`
  (station obs), `none__35v__1d35__groups-nc4.nc` (1-D arrays in groups).
- **Multiple 2-D variables**: `none__5v__1d2-2d2-3d1__curv.nc` (image bands), `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`
  (`area`, `msk_rgn`, bounds), `none__111v__1d96-2d13-3d2__str.nc`, `cf__40v__1d28-2d9-3d3__nc4.nc`.
- **Multiple 3-D variables**: `cf__20v__1d3-3d17__y-desc.nc` (17×), `cf__48v__1d17-3d21-4d10__y-asc.nc` (21×),
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (6×, mass + u/v stagger).
- **Multiple 4-D variables**: `cf__48v__1d17-3d21-4d10__y-asc.nc` (10× `(time,lev,lat,lon)`),
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (5×, mass/u/v/w/soil stagger); single-4-D:
  `coards__5v__1d4-4d1__y-desc.nc`, `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`, `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc` (`ua`).
- **Mix of all dimensionalities (1D+2D+3D+4D)**: `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc` (compact, 12 vars),
  `cf__48v__1d17-3d21-4d10__y-asc.nc`, `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`,
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`.
- **CF convention**: `cf__7v__1d3-2d3-3d1__y-asc.nc`, `cf__12v__1d4-2d5-3d2-4d1__y-asc.nc`, `cf__20v__1d3-3d17__y-desc.nc`,
  `cf__48v__1d17-3d21-4d10__y-asc.nc` (CF-1.0), `cf__40v__1d28-2d9-3d3__nc4.nc` (CF-1.6),
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc` (CF-1.4).
- **COARDS convention**: `coards__5v__1d4-4d1__y-desc.nc`, `coards__4v__1d3-3d1__y-desc.nc`.
- **UGRID / unstructured convention**: `ugrid__6v__1d5-2d1.nc` (mesh only), `ugrid__1v__3d1.nc`
  (data on faces), `ugrid__1v__1d1.nc` (data on `ncol`).
- **Curvilinear / 2-D coordinate grids**: `none__4v__1d1-2d2-3d1__curv.nc` (`xc`/`yc`),
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc` (rho/u/v lat-lon), `none__5v__1d2-2d2-3d1__curv.nc`.
- **Staggered grids**: `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (`*_stag` dims),
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc` (Arakawa-C).
- **Hierarchical groups (netCDF-4)**: `none__35v__1d35__groups-nc4.nc`.
- **Character/string variables**: `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (`Times`),
  `none__111v__1d96-2d13-3d2__str.nc` (station-id chars).

## Regenerating

`download.sh` fetches the **full published source files** under their structural names (Unidata files use the
`archive.unidata.ucar.edu` host; the old `www.unidata.ucar.edu/software/netcdf/examples/` path now 404s). It
does **not** reproduce the reduced fixtures: most catalog files were shrunk once (deflate compression plus the
dimension crops noted in the footnotes above) to keep the set small, and that step is intentionally not
scripted — so a fresh `download.sh` run restores the full-size originals. The only name that differs is the
WRF file: the full download is `none__80v__…`, while the catalog's `none__17v__…` is a manually-extracted
17-variable subset. The per-file characterization was produced with GDAL's multidimensional API
(`gdal.OpenEx(path, OF_MULTIDIM_RASTER)`).
