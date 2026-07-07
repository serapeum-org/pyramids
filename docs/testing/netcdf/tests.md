# NetCDF test scenarios

A short map of what the NetCDF test suite checks and which fixtures drive it. The fixtures themselves — their
naming convention, variable/rank breakdown, CRS, groups, and Y-axis direction — are catalogued in
[NetCDF fixtures](fixtures.md).

## Y-axis orientation

External NetCDF files store latitude **south→north** (row 0 = the south edge), while GDAL's raster convention is
row 0 = north (negative Y pixel size). pyramids therefore flips a variable on read **iff** its raw geotransform has
a positive Y pixel size (`gt[5] > 0`). The subtlety the tests pin down: GDAL's classic netCDF driver auto-flips a
recognised **geographic latitude** but **not** a projected `projection_y_coordinate` (e.g. GOES), so the fast read
path forces `GDAL_NETCDF_BOTTOMUP` to replay pyramids' own flip decision.

`tests/netcdf/spatial/test_y_orientation.py::TestFastPathOrientationAllCases` verifies the full **2×2 of CRS type ×
Y-direction**, asserting for each case (a) the recorded flip decision, (b) a north-up geotransform, and (c) that the
fast classic-driver read is **byte-identical** to the multidim view:

| | ascending (→ **flip**) | descending (→ **keep**) |
|---|---|---|
| **projected** | `…__geos__y-asc` (GOES) | `…__proj__y-desc` |
| **geographic** | `…__geog__y-asc` (NOAH, MSWEP) | `…__geog__y-desc` (ERA5), `coards…__y-desc` |

The projected-descending cell has no on-disk fixture, so that case builds a UTM grid at runtime.

Supporting orientation tests in the same file:

- `TestExternalFileOrientation` — an external (south-up) file comes back north-up (negative Y pixel size, origin at
  the north edge).
- `TestReadVariableConsistency` — the two read paths (`_read_variable` vs `get_variable().read_array()`) agree.
- `TestPyramidsCreatedNotFlipped` / `TestOneDimNotFlipped` — pyramids-written files are already GDAL-order (not
  re-flipped), and 1-D coordinate arrays are never flipped.
- `TestDiskRoundTripOrientation` — orientation survives save → reload.

## Windowed reads (#705)

`tests/netcdf/spatial/test_windowed_read_705.py` guards the geostationary/chunked windowed-read crash on
GDAL ≥ 3.13 (`arrayStartIdx[...] >= <dim>`):

- `TestWindowedRead705` — a partial-window read *raises* on the raw multidim view, then succeeds and matches the
  full read once the eager materialize swaps in the classic-driver raster; also the `to_crs(4326)` + bbox path.
- `TestFastPathFallbacks` — the fast path is taken for an on-disk variable and declines (falls back to the slow,
  correct copy) for in-memory, grouped, or orientation-unknown variables.

Fixture: `cf__9v__1d7-2d2__geos__y-asc.nc` (GOES-16, chunked, projected scan-angle Y).

## Structural scenarios

Coverage of the axes encoded in the [fixture names](fixtures.md), kept deliberately broad by a smoke test over
every file (`tests/netcdf/samples/test_smoke_all_files.py`):

| Axis | What is checked | Representative fixtures | Tests |
|---|---|---|---|
| **Convention** | CF / COARDS / none / UGRID(MPAS) detection and handling | `cf__*`, `coards__*`, `none__*`, `ugrid__*` | `samples/test_cf.py`, `samples/test_global_attributes.py`, `structure/test_global_attributes.py` |
| **Variables** | variable listing, access, metadata, rename | high-count files (`cf__48v…`, `none__111v…`) | `samples/test_variables_access.py`, `structure/test_add_variable_metadata.py`, `structure/test_rename_variable.py` |
| **Dimensions & coords** | 1-D…4-D ranks, coordinate reads, band-dim tracking | `cf__12v…`, `coards__5v…` (4-D) | `samples/test_dimensions_coords.py`, `structure/test_dimensions.py`, `structure/test_band_dim_view.py` |
| **Groups** | nested-group traversal (netCDF-4) | `none__35v__1d35__groups-nc4.nc` | `structure/test_groups.py`, `samples/test_groups.py` |
| **Curvilinear / staggered** | 2-D coordinate grids, staggered cells, windowed crop | `none__4v…__curv`, `none__5v…__curv`, `cf__8v…__curv-stag` | `samples/test_curvilinear_crop.py` |
| **String variables** | string/char-typed vars in the read path | `none__111v…__str`, `none__17v…__stag-str` | `samples/test_labeled.py`, `unit/test_netcdf_unit_read.py` |
| **Packed data** | `scale_factor` / `add_offset` unpacking | `coards__4v…__scaleoffset`, `cf__9v…__geos` | `samples/test_read_and_open.py` (`unpack=True`) |

## Read-path contract

The MDIM read plumbing that all of the above sits on is unit-tested directly:

- `tests/netcdf/test_netcdf_core.py::TestReadMdArray` — `_read_md_array` returns the classic dataset (and the
  `y_flipped` decision) for 2-D / 3-D variables.
- `tests/netcdf/unit/test_netcdf_unit_read.py` — classic vs MDIM read modes, the Y-flip decision, band-dim tracking,
  1-D string/numeric variables, and the error/fallback branches.
- `tests/netcdf/structure/test_mdim.py` — multidimensional open, group resolution, and dimension enumeration.
