# NetCDF "zoo" — a catalog of real-world NetCDF shapes

A curated set of **real** NetCDF files downloaded from public sources, chosen to cover every common
shape/convention a reader has to handle: single vs many variables; pure-1D / 2D / 3D / 4D and mixed
dimensionality; CF and COARDS conventions; curvilinear (2-D coordinate) grids; staggered grids; hierarchical
groups; and unstructured **UGRID** meshes.

Nothing here is synthetic — each file is a published example used by the wider community (Unidata's netCDF
example suite, the `pydata/xarray-data` tutorial set, and `UXARRAY`'s UGRID mesh fixtures).

> These are **not committed** to git (large binaries); they live under `examples/data/netcdf-zoo/` as a local
> reference set. Total ~150 MB.

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

| File                                            | Size    | Convention | #vars | ndim histogram            | Notable structure                                                                                  |
|-------------------------------------------------|---------|------------|-------|---------------------------|----------------------------------------------------------------------------------------------------|
| `none__1v__1d1.nc`                              | 0.04 MB | (none)     | 1     | 1×1D                      | **Single variable**, trivial (`var1(dim1)`)                                                        |
| `cf__7v__1d3-2d3-3d1.nc`                        | 2.9 MB  | CF-1.0     | 7     | 3×1D, 3×2D, 1×3D          | **Single data variable** `tos(time,lat,lon)` + coords/bounds                                       |
| `coards__4v__1d3-3d1.nc`                        | 7.8 MB  | COARDS     | 4     | 3×1D, 1×3D                | Single data variable `air(time,lat,lon)`; COARDS                                                   |
| `cf__12v__1d4-2d5-3d2-4d1.nc`                   | 2.8 MB  | CF-1.0     | 12    | 4×1D, 5×2D, 2×3D, 1×4D    | **Mix of all dims**, CF; `pr`/`tas`(3D), `ua`(4D), `area`/`msk_rgn`(2D)                            |
| `cf__20v__1d3-3d17.nc`                          | 22 MB   | CF-1.0     | 20    | 3×1D, 17×3D               | **Many 3-D variables** (17 surface fields `(time,lat,lon)`), CF                                    |
| `cf__48v__1d17-3d21-4d10.nc`                    | 18 MB   | CF-1.0     | 48    | 17×1D, 21×3D, 10×4D       | **Mix of all** — CAM init; 10 `(time,lev,lat,lon)` 4-D + 21 3-D + coords                           |
| `coards__5v__1d4-4d1.nc`                        | 17 MB   | COARDS     | 5     | 4×1D, 1×4D                | **Single 4-D variable** `rhum(time,level,lat,lon)`, int16-packed; COARDS multi-level (100 steps)‡  |
| `cf__40v__1d28-2d9-3d3__nc4.nc`                 | 0.24 MB | CF-1.6     | 40    | 28×1D, 9×2D, 3×3D         | netCDF-4; satellite L2 with averaging kernels; mixed 1D/2D/3D                                      |
| `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`         | 9.4 MB  | CF-1.4     | 8     | 3×1D, 3×2D, 1×3D, 1×4D    | **Staggered curvilinear** ocean grid (rho/u/v points); mix incl. 4-D                               |
| `none__4v__1d1-2d2-3d1__curv.nc`                | 17 MB   | (none)     | 4     | 1×1D, 2×2D, 1×3D          | **Curvilinear / 2-D coordinates** (`xc(y,x)`, `yc(y,x)`), `Tair(time,y,x)`                         |
| `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`       | 27 MB   | (none)     | 17    | 1×1D, 5×2D, 6×3D, 5×4D    | **Reduced WRF subset** — one var per distinct shape; `U/V/W`/`MAPFAC_*` on `*_stag` dims; char `Times`†|
| `none__5v__1d2-2d2-3d1__curv.nc`                | 8.6 MB  | (none)     | 5     | 2×1D, 2×2D, 1×3D          | **Multiple 2-D** image bands (McIDAS satellite); 2-D `lat`/`lon`                                   |
| `none__111v__1d96-2d13-3d2__str.nc`             | 0.27 MB | (AWIPS)    | 111   | 96×1D, 13×2D, 2×3D        | **Many 1-D variables** — station obs over `recNum`; char 2-D fields                                |
| `none__11v__1d11.nc`                            | 0.01 MB | (none)     | 11    | 11×1D                     | **Multiple 1-D variables** — aircraft track time series                                            |
| `none__35v__1d35__groups-nc4.nc`                | 0.09 MB | (none)     | 35    | 35×1D                     | **Hierarchical groups** (netCDF-4); 1-D arrays nested under flight groups                          |
| `ugrid__6v__1d5-2d1.nc`                         | 0.02 MB | UGRID*     | 6     | 5×1D, 1×2D                | **UGRID mesh topology** — `node_lon`/`face_lon`, `face_node_connectivity(n_face,n_max_face_nodes)` |
| `ugrid__1v__3d1.nc`                             | 0.01 MB | UGRID*     | 1     | 1×3D                      | **UGRID data**, 3-D on an unstructured mesh: `multi_dim_data(time,lev,n_face)`                     |
| `ugrid__1v__1d1.nc`                             | 0.04 MB | UGRID*     | 1     | 1×1D                      | **UGRID/unstructured data** on a cubed-sphere SE grid: `psi(ncol)`                                 |

\* The UGRID files come from UXARRAY's `meshfiles/ugrid/` set. GDAL's multidim view flattens the scalar
`mesh_topology` dummy variable, so the `Conventions` attribute may read empty/`MPAS`; the unstructured nature
is unmistakable from the `n_node`/`n_face`/`ncol` dimensions and the `*_connectivity` arrays.

† `none__17v__…` is a **reduced subset** of the 80-variable `wrfout_v2_Lambert.nc`: the original had 63
redundant variables that all share a shape already covered by another. One representative per distinct
`(rank × dimension-set × dtype × role)` signature was kept (all four ranks, all three dtypes
`float32`/`int32`/`char`, every mass/u/v/w/soil stagger, and the `XLAT`/`XLONG` geolocation pair), shrinking
the file from 82 MB to 27 MB with no loss of reader-code-path coverage. Reproduce with `reduce_wrf.py`.

‡ `coards__5v__…` was trimmed from the published 365 daily steps (~61 MB) to its first 100 (~17 MB). Nothing
structural changed — the four 1-D coordinate axes and the single `int16`-packed 4-D `rhum` variable (with its
`scale_factor`/`add_offset`/`_FillValue`) are intact; only the `time` dimension is shorter. Reproduce with
`trim_rhum.py`.

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
| `coards__4v__1d3-3d1.nc`                     | `air_temperature.nc`                    | xarray-data       |
| `coards__5v__1d4-4d1.nc`                     | `rhum.2003.nc`                          | Unidata examples  |
| `cf__7v__1d3-2d3-3d1.nc`                     | `tos_O1_2001-2002.nc`                   | Unidata examples  |
| `cf__12v__1d4-2d5-3d2-4d1.nc`                | `sresa1b_ncar_ccsm3-example.nc`         | Unidata examples  |
| `cf__20v__1d3-3d17.nc`                       | `ECMWF_ERA-40_subset.nc`                | Unidata examples  |
| `cf__48v__1d17-3d21-4d10.nc`                 | `cami_0000-09-01_64x128_L26_c030918.nc` | Unidata examples  |
| `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`      | `ROMS_example.nc`                       | xarray-data       |
| `cf__40v__1d28-2d9-3d3__nc4.nc`              | `OMI-Aura_L2-example.nc`                | Unidata examples  |
| `ugrid__6v__1d5-2d1.nc`                      | `quad-hexagon/grid.nc`                  | UXARRAY meshfiles |
| `ugrid__1v__3d1.nc`                          | `quad-hexagon/multi_dim_data.nc`        | UXARRAY meshfiles |
| `ugrid__1v__1d1.nc`                          | `outCSne30/outCSne30_vortex.nc`         | UXARRAY meshfiles |

## Scenario index — which file to use for what

- **One variable**: `none__1v__1d1.nc` (bare 1-D), `cf__7v__1d3-2d3-3d1.nc` (one CF data var `tos` + coords),
  `coards__4v__1d3-3d1.nc`, `ugrid__1v__1d1.nc` / `ugrid__1v__3d1.nc` (one var on a mesh).
- **Multiple variables**: `cf__12v__1d4-2d5-3d2-4d1.nc`, `cf__20v__1d3-3d17.nc`, `cf__48v__1d17-3d21-4d10.nc`,
  `cf__40v__1d28-2d9-3d3__nc4.nc`, `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`,
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`, `none__111v__1d96-2d13-3d2__str.nc`,
  `none__35v__1d35__groups-nc4.nc`.
- **Multiple 1-D variables**: `none__11v__1d11.nc` (aircraft track), `none__111v__1d96-2d13-3d2__str.nc`
  (station obs), `none__35v__1d35__groups-nc4.nc` (1-D arrays in groups).
- **Multiple 2-D variables**: `none__5v__1d2-2d2-3d1__curv.nc` (image bands), `cf__12v__1d4-2d5-3d2-4d1.nc`
  (`area`, `msk_rgn`, bounds), `none__111v__1d96-2d13-3d2__str.nc`, `cf__40v__1d28-2d9-3d3__nc4.nc`.
- **Multiple 3-D variables**: `cf__20v__1d3-3d17.nc` (17×), `cf__48v__1d17-3d21-4d10.nc` (21×),
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (6×, mass + u/v stagger).
- **Multiple 4-D variables**: `cf__48v__1d17-3d21-4d10.nc` (10× `(time,lev,lat,lon)`),
  `none__17v__1d1-2d5-3d6-4d5__stag-str.nc` (5×, mass/u/v/w/soil stagger); single-4-D:
  `coards__5v__1d4-4d1.nc`, `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`, `cf__12v__1d4-2d5-3d2-4d1.nc` (`ua`).
- **Mix of all dimensionalities (1D+2D+3D+4D)**: `cf__12v__1d4-2d5-3d2-4d1.nc` (compact, 12 vars),
  `cf__48v__1d17-3d21-4d10.nc`, `none__17v__1d1-2d5-3d6-4d5__stag-str.nc`,
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc`.
- **CF convention**: `cf__7v__1d3-2d3-3d1.nc`, `cf__12v__1d4-2d5-3d2-4d1.nc`, `cf__20v__1d3-3d17.nc`,
  `cf__48v__1d17-3d21-4d10.nc` (CF-1.0), `cf__40v__1d28-2d9-3d3__nc4.nc` (CF-1.6),
  `cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc` (CF-1.4).
- **COARDS convention**: `coards__5v__1d4-4d1.nc`, `coards__4v__1d3-3d1.nc`.
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

The download list is in `download.sh` (Unidata files use the `archive.unidata.ucar.edu` host; the old
`www.unidata.ucar.edu/software/netcdf/examples/` path now 404s). It downloads each file under its original
published name and then renames it to the structural name (see the mapping table above). The per-file
characterization was produced with GDAL's multidimensional API (`gdal.OpenEx(path, OF_MULTIDIM_RASTER)`).
