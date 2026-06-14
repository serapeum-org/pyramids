# Comparison with other GIS packages

How pyramids relates to the established Python geospatial / scientific-array stack. This page compares
pyramids against [rasterio](https://rasterio.readthedocs.io/),
[xarray](https://docs.xarray.dev/), and [rioxarray](https://corteva.github.io/rioxarray/), and is the
place to add further comparisons over time.

## How to read this — a fairness note

A raw feature checklist is **unfair to rasterio, xarray, and rioxarray**, for different reasons:

- **rasterio** is deliberately a *focused raster-core* library (Unix-philosophy). Most things it does not
  do itself are provided, often more maturely, by sibling packages — `fiona`/`geopandas` (vector),
  `rasterstats` (zonal), `rioxarray`/`stackstac` (datacube), `rio-cogeo` (COG), `shapely` (geometry).
- **xarray is not a GIS library at all.** It is the standard for *N-dimensional labelled arrays* and
  *datacubes* (NetCDF / CF / Zarr / Dask). It has no native CRS, geotransform, or raster concept; its
  geospatial powers come from extensions: `rioxarray`, `xrspatial`, `stackstac`/`odc-stac`, `uxarray`,
  `cfgrib`. Comparing it on a GIS checklist understates it — it is the *substrate* datacube tools build on.
- **rioxarray** is the rasterio-on-xarray bridge: it adds CRS, geotransform, reproject / clip / merge,
  and GeoTIFF/COG read-write to xarray via the `.rio` accessor. It is xarray's "missing geospatial
  column" made concrete — the closest single-package peer to pyramids' raster + datacube combo — but it
  inherits xarray's vector / zonal / terrain / STAC blanks (filled by the ecosystem).

pyramids overlaps this stack on the datacube / NetCDF axis, but it is a different kind of tool: pyramids
is **GDAL/GIS-first** (a CRS-aware raster + vector + datacube library), whereas xarray/rioxarray are
**array-first**. So the tables compare **what each single library ships**, not what each *ecosystem* can
do; `→pkg` marks a capability supplied by an ecosystem package.

Legend: ✓ built-in · ✓✓ a strength · ✓✓✓ the de-facto standard · ◐ partial / needs wiring · ✗ not
provided · `→pkg` via an ecosystem package.

**A ✓ means the capability is reachable through that library's *own* public API — a method, function, or
CLI command it ships — not merely that the underlying engine (GDAL/GEOS/PROJ) could do it if you dropped
down to it.** pyramids is a GDAL/OGR wrapper, so nearly all of its raster ✓s are "a GDAL call wrapped in a
pyramids method" — which counts, exactly as rasterio's and rioxarray's GDAL wrappers count, *because there
is a Pythonic entry point*. Where pyramids has **no** wrapper and you would have to call `osgeo.gdal`
yourself, the cell is **not** a ✓ (see GRIB, GCP/RPC below). Every pyramids ✓ in these tables was checked
against a concrete public symbol in `src/pyramids`.

**A ✓ is still not parity in maturity, performance, or edge-case robustness.** rasterio, xarray, and
rioxarray are years more battle-tested and far more widely deployed; several of pyramids' ✓s are
comparatively new. Read the tables as *surface coverage*, with the maturity row as the counterweight.

## pyramids vs rasterio vs xarray vs rioxarray

### Core raster I/O

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| GDAL raster formats (GeoTIFF, …) | ✓ `Dataset` | ✓✓ standard | ◐ `→rioxarray` | ✓ `open_rasterio` |
| Windowed read & write | ✓ `read/write_array(window=)`, boundless, `Window` algebra | ✓✓ `Window` | ◐ Dask chunks | ◐ Dask chunks |
| In-memory raster (bytes) | ✓ `from_bytes`/`to_bytes` | ✓ `MemoryFile` | ◐ `→rioxarray` | ◐ via `MemoryFile` |
| Overviews / pyramids | ✓ `create_overviews`, `read_overview_array` | ✓ `build_overviews` | ✗ `→rio` | ◐ `→rasterio` |
| COG write / validate / inspect | ✓✓ `to_cog` | ◐ `→rio-cogeo` | ✗ `→rioxarray` | ◐ `to_raster(COG)` |
| Decimated reads (preview / tile) | ✓✓ `out_shape` + `preview` | ◐ `out_shape` | ◐ `→rioxarray` | ◐ `overview_level` |
| No-data / masks / colour interp | ✓ `mask_flags`/`read_masks`/`band_color` | ✓ | ◐ `.where` | ✓ `.rio.nodata` |

### CRS, warping & alignment

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Reproject / warp | ✓ `to_crs` | ✓ `warp.reproject` | ✗ `→rioxarray` | ✓✓ `.rio.reproject` |
| Lazy / on-the-fly warp (VRT) | ✓ `warped_view` | ✓✓ `WarpedVRT` | ✗ | ◐ `→rasterio` |
| Resample (spatial) | ✓ `resample` | ✓ resampling enums | ✗ `→rioxarray` | ✓ `reproject(resampling=)` |
| Align / snap to target grid | ✓ `align` | ◐ manual | ◐ `align` (labels) | ✓✓ `reproject_match` |
| CRS / affine transforms | ✓ | ✓✓ `Affine` | ✗ `→rioxarray` | ✓✓ `.rio.crs` |
| GCP / RPC georeferencing | ✓ `set_gcps`/`georeference`/`orthorectify` | ✓✓ `gcps`/`rpcs` | ✗ | ◐ `→rasterio` |

Note: xarray's own `.resample` operates on a labelled dimension (e.g. time), not spatial reprojection —
spatial resample/warp is the `rioxarray` accessor's job.

### Raster analysis

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Crop by bbox / geometry | ✓ `crop` | ✓ `mask.mask` | ◐ `.sel` | ✓ `.rio.clip` |
| Mosaic / merge | ✓ `merge` | ✓ `merge.merge` | ✗ `→stackstac` | ✓ `merge_arrays` |
| Zonal statistics | ✓ `zonal_stats` | ✗ `→rasterstats` | ✗ `→xrspatial` | ✗ `→xrspatial` |
| Terrain (slope / aspect / hillshade) | ✓ `slope`/`aspect`/`hillshade` | ✗ `→gdaldem` | ✗ `→xrspatial` | ✗ `→xrsp.` |
| Proximity / sieve / contour | ✓ `proximity`/`sieve`/`contour` | ◐ `features.sieve` | ✗ `→xrspatial` | ✗ `→scipy` |
| Interpolation / gridding | ✓ `from_points` | ✗ `→scipy` | ◐ `.interp` | ◐ `interpolate_na` |
| Connected-component clustering | ✓ `cluster` | ✗ `→scipy` | ✗ `→scipy` | ✗ `→scipy` |
| Point sampling | ✓ `sample` / `point` | ✓ `sample` | ✓✓ `.sel(method=)` | ✓✓ `.sel(method=)` |
| Band statistics (min/max/mean/std) | ✓ `stats` | ◐ numpy | ✓✓ `.mean` | ✓✓ `.mean` |
| Histogram | ✓ `get_histogram` | ◐ numpy | ✓ `.plot.hist` | ✓ `.plot.hist` |

### Raster ↔ vector

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Rasterize vectors | ✓ `from_features` | ✓ `features.rasterize` | ✗ `→geocube` | ✗ `→geocube` |
| Vectorize / polygonize | ✓ `to_feature_collection` | ✓ `features.shapes` | ✗ `→rasterio` | ✗ `→rasterio` |
| Dataset footprint polygon | ✓ `footprint` | ◐ `mask` + `shapes` | ✗ `→rioxarray` | ◐ `.rio.bounds` |

### Vector data (standalone)

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Vector I/O (read/write) | ✓ `FeatureCollection` | ✗ `→fiona` | ✗ `→geopandas` | ✗ `→geopandas` |
| Geometry operations | ✓ | ✗ `→shapely` | ✗ `→shapely` | ✗ `→shapely` |
| GeoParquet | ✓ | ✗ `→geopandas` | ✗ `→geopandas` | ✗ `→geopandas` |

### Multi-dimensional / datacube / formats

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Time-series datacube | ✓ `DatasetCollection` | ✗ `→rioxarray` | ✓✓✓ the standard | ✓✓ xarray + CRS |
| NetCDF + CF conventions | ✓✓ first-class | ◐ subdatasets only | ✓✓✓ first-class | ✓✓ + CRS |
| UGRID unstructured grids | ✓ | ✗ | ◐ `→uxarray` | ✗ `→uxarray` |
| Zarr | ✓ `to_zarr`/`from_zarr` | ◐ GDAL driver | ✓✓ native | ✓ + CRS |
| GRIB (read) | ◐ via `read_file` (no GRIB-specific API) | ◐ via GDAL | ◐ `→cfgrib` | ◐ `→cfgrib` |

### Cloud, STAC & lazy compute

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Cloud VSI (s3 / gs / az) | ✓ `remote` | ✓ `/vsi*/` | ◐ `→fsspec` | ◐ `→fsspec` |
| STAC search / load / mosaic | ✓✓ `stac/` | ✗ `→pystac-client` | ✗ `→stackstac` | ✗ `→stackstac` |
| Requester-pays / signing | ✓ `Signer` | ◐ `AWSSession` | ✗ `→fsspec` | ◐ rasterio session |
| Lazy / Dask-backed arrays | ✓ Dask `chunks=` | ✗ `→rioxarray` | ✓✓✓ standard | ✓✓✓ standard |
| Concurrent windowed reads | ✓ `read_array(threadsafe=)`, `read_windows` | ✓✓ | ✓ via Dask | ✓ via Dask |

### Tooling & maturity

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| CLI | ✓ `pyramids` (incl. `edit-info`, `calc`, `georeference`, `shapes`, `rasterize`) | ✓ `rio` | ✗ | ✗ |
| Plotting | ◐ `→cleopatra` | ✓ `rasterio.plot` | ✓✓ `xarray.plot` | ✓✓ `xarray.plot` |
| Maturity / adoption / community | younger | ✓✓✓ standard | ✓✓✓ huge (Pangeo) | ✓✓ widely used |
| Stability / docs depth | growing | ✓✓✓ | ✓✓✓ | ✓✓ |

### Honest summary

- **rasterio's real strengths:** maturity, stability, enormous adoption, the most granular windowed-I/O
  API, and a deep, composable raster ecosystem (`rio-cogeo`, `rio-tiler`, `rasterstats`). Its narrow core
  is a *feature*. pyramids now matches it on boundless windowed reads, decimated `out_shape` reads,
  in-memory byte round-trips, per-thread concurrent reads (`read_windows`), lazy on-the-fly warping
  (`warped_view` — its `WarpedVRT` equivalent), per-band mask-flag introspection (`mask_flags`/
  `read_masks`), pixel↔map accessors (`xy`/`rowcol`), and — the last remaining capability gap, now closed —
  **GCP / RPC georeferencing** (`set_gcps`/`georeference`/`set_rpcs`/`orthorectify`). With that, there is no
  longer a *raster capability* rasterio has that pyramids lacks; rasterio's edge is now **maturity,
  ecosystem depth, and ergonomic polish** (the granular `windows` algebra, the `rio` plugin family), not
  missing features. Treat the surface ✓s as real, with the maturity rows as the honest counterweight.
- **xarray's real strengths:** the de-facto standard for N-dimensional labelled arrays and datacubes —
  NetCDF/CF, Zarr, Dask-backed lazy compute, `groupby`/reductions, label-based selection, and faceted
  plotting. Outside its remit (CRS, warping, GIS raster ops) it leans on `rioxarray` / `xrspatial`.
- **rioxarray's real strengths:** turns xarray into a CRS-aware raster engine — `reproject` /
  `reproject_match` / `clip` / `merge` and GeoTIFF/COG I/O on lazy Dask datacubes. It is the closest
  single-package peer to pyramids' raster + datacube combo, but inherits xarray's vector / zonal /
  terrain / STAC blanks (→ ecosystem).
- **pyramids' real strengths:** breadth in **one** GDAL/GIS-first package, each capability behind its
  own Pythonic API — raster **and** vector **and** datacube; NetCDF/CF/UGRID, STAC, terrain / zonal /
  interpolation / clustering, COG tooling, and lazy/Dask, without stitching together several libraries.
- **When to pick which:**
  - **pyramids** — an integrated, batteries-included, CRS-aware GDAL toolkit covering raster, vector,
    and datacubes together.
  - **rasterio (+ ecosystem)** — a mature, minimal, highly-composable raster core; add the pieces you need.
  - **xarray + rioxarray** — your problem is genuinely N-dimensional scientific arrays / large lazy
    datacubes, and you want a CRS-aware raster layer on top.

### Honest conclusion (strict "ships its own API" audit)

Re-auditing every pyramids cell against a concrete public symbol — counting a capability **only** when
pyramids exposes its own method / function / CLI command, never just because GDAL could do it — the
breadth claims hold up better than a "thin wrapper" reputation would suggest:

- Of the ~40 capabilities checked, pyramids now ships a real public API for **every raster capability
  rasterio has**. The previously-flagged gap — **GCP / RPC georeferencing** — is closed
  (`set_gcps`/`gcps`/`georeference`/`set_rpcs`/`rpcs`/`orthorectify`, routed through `gdal.Warp`). The only
  remaining rasterio-side item is **GRIB-specific handling**: pyramids opens GRIB via the generic
  `read_file` (as any GDAL driver does) but exposes no GRIB-*specific* surface (parameter/WMO catalog,
  message inspection) — a format-metadata nicety, not a raster capability.
- A further pass for rasterio-only features turned up only one **ergonomic** difference, not a capability:
  a one-call `geometry_mask` (pyramids goes vector→raster via `from_features`/`rasterize` then reads the
  mask). Everything substantive — windowed/boundless/decimated/threadsafe reads, `warped_view`, masks,
  `xy`/`rowcol`, densified bbox reprojection (`feature.bbox.transform_bounds`), nodata `fill`, GCP/RPC, and
  the raster↔vector CLI verbs (`pyramids shapes`/`rasterize`) — is present.
- Against **xarray / rioxarray** the differences are **not** capability gaps but *kind-of-tool* and
  *maturity* differences: they are array-first and far more battle-tested on huge lazy datacubes;
  pyramids is GDAL/GIS-first and bundles vector + zonal + terrain + STAC that xarray/rioxarray leave to
  the ecosystem.
- The honest caveat is therefore **not** "pyramids lacks the features" — it is "pyramids' features are
  younger and less battle-tested." Surface coverage is broad and genuinely API-backed; maturity,
  performance tuning, and edge-case hardening are where the established libraries still lead.

> Scope reminder: pyramids stays a *generic* GDAL/OGR toolkit — the breadth above is generic primitives
> and format support, not domain logic. See [Scope](SCOPE.md) for the boundary.
