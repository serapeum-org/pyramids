# NetCDF test scenarios

A short map of what the NetCDF test suite checks and which fixtures drive it. The fixtures themselves — their
naming convention, variable/rank breakdown, CRS, groups, and Y-axis direction — are catalogued in
[NetCDF fixtures](fixtures.md).

## Y-axis orientation

External NetCDF files store latitude **south→north** (row 0 = the south edge), while GDAL's raster convention is
row 0 = north (negative Y pixel size). pyramids therefore flips a variable on read **iff** its Y coordinate
**ascends**, read the way GDAL's own classic netCDF driver decides `bBottomUp`: through `GetUnscaled()`, so the
coordinate's `scale_factor` / `add_offset` are applied first.

The rule lives once, in `pyramids.netcdf._mdim`, and **all three read paths share it**: the eager
`get_variable().read_array()`, the chunked `read_array(chunks=...)` (via `build_lazy_array`), and the internal
`_read_variable`. When the rule lived only in the eager path, a chunked read of a geostationary variable came back
`flipud` of the eager read — #705, still live on a public API.

Two subtleties the tests pin down:

- The decision must **not** be read off the multidim view's geotransform. `GDALMDArray::GuessGeoTransform()` builds
  that from the *raw* coordinate values, so a geostationary granule — whose `y` is packed with a **negative**
  `scale_factor` — reports a positive `gt[5]` for an array that is already north-up. Flipping it mirrored the
  raster (#705).
- The decision must **not** key off the CRS. GDAL's classic driver auto-flips a recognised **geographic latitude**
  but not a projected `projection_y_coordinate`; pyramids' rule is CRS-agnostic and handles both.

`tests/netcdf/spatial/test_y_orientation.py::TestOrientationAllCases` verifies the full **CRS type × Y-direction**
matrix, asserting for each case (a) the recorded flip decision, (b) a north-up geotransform, and (c) byte-identity
with a reference array reordered so row 0 sits at the largest scaled Y coordinate:

| | ascending (→ **flip**) | descending (→ **keep**) |
|---|---|---|
| **geostationary** | *(no known producer)* | `…__geos__y-desc` (GOES — raw values ascend) |
| **projected** | runtime UTM grid | runtime UTM grid |
| **geographic** | `…__geog__y-asc` (NOAH, MSWEP) | `…__geog__y-desc` (ERA5), `coards…__y-desc` |

Neither projected cell has an on-disk fixture, so both build a UTM grid at runtime (`WRITE_BOTTOMUP=YES` / `NO`).

The same rule applies to the X axis, mirrored: a longitude stored **east→west** is reversed so `col 0 = west`, on
every read path.
GDAL's classic driver never flips X — it reports a negative `gt[1]` — but a negative pixel width cannot survive
pyramids' `abs()`-based cell size and bbox arithmetic, and the coordinate-derived geotransform used to take
`lon[0]` for the west edge, so such a file came back mirrored west-east under a shifted bbox. No known producer
writes one, so `TestXAxisOrientation` builds it at runtime (`x_descending_nc`) and also asserts the invariant that
every on-disk fixture ascends in X.

Supporting orientation tests in the same file:

- `TestExternalFileOrientation` — an external (south-up) file comes back north-up (negative Y pixel size, origin at
  the north edge).
- `TestReadVariableConsistency` — the two read paths (`_read_variable` vs `get_variable().read_array()`) agree.
- `TestPyramidsCreatedNotFlipped` / `TestOneDimNotFlipped` — pyramids-written files are already GDAL-order (not
  re-flipped), and 1-D coordinate arrays are never flipped.
- `TestDiskRoundTripOrientation` — orientation survives save → reload.

## Windowed reads (#705)

`tests/netcdf/spatial/test_windowed_read_705.py` guards the windowed-read crash on GDAL ≥ 3.13
(`arrayStartIdx[...] >= <dim>`). The crash comes from reading a window through a **reversed**
`MDArray.GetView("[::-1, ...]")`, not from `AsClassicDataset` itself — an unreversed view services windowed reads
fine — so only a genuinely bottom-up file is affected:

- `TestWindowedRead705` — a geostationary variable is never reversed, so its view is window-readable as-is; a
  bottom-up geographic file (NOAH) *raises* until the eager materialize reads the unreversed array and flips it
  with NumPy. Also covers the `to_crs(4326)` + bbox path from the issue.
- `TestMaterializeIntegrity` — materializing is idempotent, preserves the pixels and `unpack=True` values, and
  falls back to a plain copy when the raw view cannot be rebuilt.
- `TestClassicDriverNotUsedForPixels` — the materialize reads the multidim array, never the classic subdataset
  driver, which returns pure fill for some 4-D packed variables (`coards__5v__1d4-4d1__y-desc.nc::rhum`).
- `TestGeostationaryGroundTruth` — `read_array()` equals the classic driver's array, not its `flipud`.

Fixtures: `cf__9v__1d7-2d2__geos__y-desc.nc` (GOES-16, chunked, packed scan-angle Y) and
`cf__6v__1d2-2d4__geog__y-asc.nc` (NOAH, bottom-up).

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
