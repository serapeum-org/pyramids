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

**A ✓ means the capability exists in that library's own API — not parity in maturity, performance, or
edge-case robustness.** rasterio, xarray, and rioxarray are years more battle-tested and far more widely
deployed; several of pyramids' ✓s are comparatively new. Read the tables as *surface coverage*, with the
maturity row as the counterweight.

## pyramids vs rasterio vs xarray vs rioxarray

### Core raster I/O

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| GDAL raster formats (GeoTIFF, …) | ✓ `Dataset` | ✓✓ standard | ◐ `→rioxarray` | ✓ `open_rasterio` |
| Windowed / block read & write | ✓ `write_array(window=)` | ✓✓ `Window` | ◐ Dask chunks | ◐ Dask chunks |
| In-memory raster (bytes) | ◐ `from_bytes` | ✓ `MemoryFile` | ◐ `→rioxarray` | ◐ via `MemoryFile` |
| Overviews / pyramids | ✓ `create_overviews` | ✓ `build_overviews` | ✗ `→rio` | ◐ `→rasterio` |
| COG write / validate / inspect | ✓✓ `to_cog` | ◐ `→rio-cogeo` | ✗ `→rioxarray` | ◐ `to_raster(COG)` |
| Decimated reads (preview / tile) | ✓ `preview` | ◐ `out_shape` | ◐ `→rioxarray` | ◐ `overview_level` |
| No-data / masks / colour interp | ✓ | ✓ | ◐ `.where` | ✓ `.rio.nodata` |

### CRS, warping & alignment

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Reproject / warp | ✓ `to_crs` | ✓ `warp.reproject` | ✗ `→rioxarray` | ✓✓ `.rio.reproject` |
| Resample (spatial) | ✓ `resample` | ✓ resampling enums | ✗ `→rioxarray` | ✓ `reproject(resampling=)` |
| Align / snap to target grid | ✓ `align` | ◐ manual | ◐ `align` (labels) | ✓✓ `reproject_match` |
| CRS / affine transforms | ✓ | ✓✓ `Affine` | ✗ `→rioxarray` | ✓✓ `.rio.crs` |

Note: xarray's own `.resample` operates on a labelled dimension (e.g. time), not spatial reprojection —
spatial resample/warp is the `rioxarray` accessor's job.

### Raster analysis

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Crop by bbox / geometry | ✓ `crop` | ✓ `mask.mask` | ◐ `.sel` | ✓ `.rio.clip` |
| Mosaic / merge | ✓ `merge` | ✓ `merge.merge` | ✗ `→stackstac` | ✓ `merge_arrays` |
| Zonal statistics | ✓ `zonal` | ✗ `→rasterstats` | ✗ `→xrspatial` | ✗ `→xrspatial` |
| Terrain (slope / aspect / hillshade) | ✓ built-in | ✗ `→gdaldem` | ✗ `→xrspatial` | ✗ `→xrspatial` |
| Proximity / sieve / contour | ✓ built-in | ◐ `features.sieve` | ✗ `→xrspatial` | ✗ `→scipy` |
| Interpolation / gridding | ✓ `gdal.Grid` | ✗ `→scipy` | ◐ `.interp` | ◐ `interpolate_na` |
| Connected-component clustering | ✓ `cluster` | ✗ `→scipy` | ✗ `→scipy` | ✗ `→scipy` |
| Point sampling | ✓ `sample` / `point` | ✓ `sample` | ✓✓ `.sel(method=)` | ✓✓ `.sel(method=)` |
| Band statistics (min/max/mean/std) | ✓ `stats` | ◐ numpy | ✓✓ `.mean` | ✓✓ `.mean` |
| Histogram | ◐ viz (`plot_histogram`) | ◐ numpy | ✓ `.plot.hist` | ✓ `.plot.hist` |

### Raster ↔ vector

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Rasterize vectors | ✓ `rasterize` | ✓ `features.rasterize` | ✗ `→geocube` | ✗ `→geocube` |
| Vectorize / polygonize | ✓ `to_polygon` | ✓ `features.shapes` | ✗ `→rasterio` | ✗ `→rasterio` |
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
| Zarr | ✓ | ◐ GDAL driver | ✓✓ native | ✓ + CRS |
| GRIB | ✓ (+ WMO glossary) | ◐ via GDAL | ◐ `→cfgrib` | ◐ `→cfgrib` |

### Cloud, STAC & lazy compute

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| Cloud VSI (s3 / gs / az) | ✓ `remote` | ✓ `/vsi*/` | ◐ `→fsspec` | ◐ `→fsspec` |
| STAC search / load / mosaic | ✓✓ `stac/` | ✗ `→pystac-client` | ✗ `→stackstac` | ✗ `→stackstac` |
| Requester-pays / signing | ✓ `Signer` | ◐ `AWSSession` | ✗ `→fsspec` | ◐ rasterio session |
| Lazy / Dask-backed arrays | ✓ Dask `chunks=` | ✗ `→rioxarray` | ✓✓✓ standard | ✓✓✓ standard |
| Concurrent windowed reads | ◐ | ✓✓ | ✓ via Dask | ✓ via Dask |

### Tooling & maturity

| Capability | pyramids | rasterio | xarray | rioxarray |
|---|---|---|---|---|
| CLI | ✓ `pyramids` | ✓ `rio` | ✗ | ✗ |
| Plotting | ◐ `→cleopatra` | ✓ `rasterio.plot` | ✓✓ `xarray.plot` | ✓✓ `xarray.plot` |
| Maturity / adoption / community | younger | ✓✓✓ standard | ✓✓✓ huge (Pangeo) | ✓✓ widely used |
| Stability / docs depth | growing | ✓✓✓ | ✓✓✓ | ✓✓ |

### Honest summary

- **rasterio's real strengths:** maturity, stability, enormous adoption, the most granular windowed-I/O
  API, strong concurrency, and a deep, composable raster ecosystem. Its narrow core is a *feature*.
- **xarray's real strengths:** the de-facto standard for N-dimensional labelled arrays and datacubes —
  NetCDF/CF, Zarr, Dask-backed lazy compute, `groupby`/reductions, label-based selection, and faceted
  plotting. Outside its remit (CRS, warping, GIS raster ops) it leans on `rioxarray` / `xrspatial`.
- **rioxarray's real strengths:** turns xarray into a CRS-aware raster engine — `reproject` /
  `reproject_match` / `clip` / `merge` and GeoTIFF/COG I/O on lazy Dask datacubes. It is the closest
  single-package peer to pyramids' raster + datacube combo, but inherits xarray's vector / zonal /
  terrain / STAC blanks (→ ecosystem).
- **pyramids' real strengths:** breadth in **one** GDAL/GIS-first package — raster **and** vector **and**
  datacube; NetCDF/CF/UGRID, STAC, terrain / zonal / interpolation, COG tooling, and lazy/Dask, without
  stitching together several libraries.
- **When to pick which:**
  - **pyramids** — an integrated, batteries-included, CRS-aware GDAL toolkit covering raster, vector,
    and datacubes together.
  - **rasterio (+ ecosystem)** — a mature, minimal, highly-composable raster core; add the pieces you need.
  - **xarray + rioxarray** — your problem is genuinely N-dimensional scientific arrays / large lazy
    datacubes, and you want a CRS-aware raster layer on top.

> Scope reminder: pyramids stays a *generic* GDAL/OGR toolkit — the breadth above is generic primitives
> and format support, not domain logic. See [Scope](SCOPE.md) for the boundary.
