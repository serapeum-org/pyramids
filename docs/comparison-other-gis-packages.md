# Comparison with other GIS packages

How pyramids relates to the established Python geospatial / scientific-array stack. This page compares
pyramids against [rasterio](https://rasterio.readthedocs.io/) and
[xarray](https://docs.xarray.dev/), and is the place to add further comparisons over time.

## How to read this — a fairness note

A raw feature checklist is **unfair to both rasterio and xarray**, for opposite reasons:

- **rasterio** is deliberately a *focused raster-core* library (Unix-philosophy). Most things it does not
  do itself are provided, often more maturely, by sibling packages — `fiona`/`geopandas` (vector),
  `rasterstats` (zonal), `rioxarray`/`stackstac` (datacube), `rio-cogeo` (COG), `shapely` (geometry).
- **xarray is not a GIS library at all.** It is the standard for *N-dimensional labelled arrays* and
  *datacubes* (NetCDF / CF / Zarr / Dask). It has no native CRS, geotransform, or raster concept; its
  geospatial powers come from extensions: `rioxarray` (raster GIS accessor), `xrspatial` (analysis),
  `stackstac` / `odc-stac` (STAC → xarray), `uxarray` (UGRID), `cfgrib` (GRIB). Comparing it on a GIS
  checklist understates it — it is the *substrate* most geospatial datacube tools build on.

pyramids overlaps xarray on the datacube / NetCDF axis, but the two are different kinds of tool: pyramids
is **GDAL/GIS-first** (a CRS-aware raster + vector + datacube library), xarray is **array-first** and
domain-agnostic. So the tables compare **what each single library ships**, not what each *ecosystem* can
do; `→pkg` marks a capability supplied by an ecosystem package.

Legend: ✓ built-in · ✓✓ a strength · ✓✓✓ the de-facto standard · ◐ partial / needs wiring · ✗ not
provided · `→pkg` via an ecosystem package.

**A ✓ means the capability exists in that library's own API — not parity in maturity, performance, or
edge-case robustness.** rasterio and xarray are years more battle-tested and far more widely deployed;
several of pyramids' ✓s are comparatively new. Read the tables as *surface coverage*, with the
maturity row as the counterweight.

## pyramids vs rasterio vs xarray

### Core raster I/O

| Capability                       | pyramids                               | rasterio            | xarray                 |
|----------------------------------|----------------------------------------|---------------------|------------------------|
| GDAL raster formats (GeoTIFF, …) | ✓ `Dataset`                            | ✓✓ standard         | ◐ `→rioxarray`         |
| Windowed / block read & write    | ✓ `read_part` / `write_array(window=)` | ✓✓ `Window`         | ◐ Dask chunks          |
| In-memory raster (bytes)         | ◐ `from_bytes` / `to_cog_bytes`        | ✓ `MemoryFile`      | ◐ `→rioxarray`         |
| Overviews / pyramids             | ✓ `create_overviews`                   | ✓ `build_overviews` | ✗ `→rio`               |
| COG write / validate / inspect   | ✓✓ `to_cog`                            | ◐ `→rio-cogeo`      | ✗ `→rioxarray`         |
| Decimated reads (preview / tile) | ✓ `preview` / `point`                  | ◐ `out_shape`       | ◐ `→rioxarray`         |
| No-data / masks / colour interp  | ✓                                      | ✓                   | ◐ `.where` (no colour) |

### CRS, warping & alignment

| Capability                  | pyramids     | rasterio           | xarray             |
|-----------------------------|--------------|--------------------|--------------------|
| Reproject / warp            | ✓ `to_crs`   | ✓ `warp.reproject` | ✗ `→rioxarray`     |
| Resample (spatial)          | ✓ `resample` | ✓ resampling enums | ✗ `→rioxarray`     |
| Align / snap to target grid | ✓ `align`    | ◐ manual           | ◐ `align` (labels) |
| CRS / affine transforms     | ✓            | ✓✓ `Affine`        | ✗ `→rioxarray`     |

Note: xarray's own `.resample` operates on a labelled dimension (e.g. time), not spatial reprojection —
spatial resample/warp is the `rioxarray` accessor's job.

### Raster analysis

| Capability                           | pyramids                      | rasterio           | xarray                       |
|--------------------------------------|-------------------------------|--------------------|------------------------------|
| Crop by bbox / geometry              | ✓ `crop`                      | ✓ `mask.mask`      | ◐ `.sel` / `→rioxarray.clip` |
| Mosaic / merge                       | ✓ `merge`                     | ✓ `merge.merge`    | ✗ `→stackstac` / `odc`       |
| Zonal statistics                     | ✓ `zonal`                     | ✗ `→rasterstats`   | ✗ `→xrspatial`               |
| Terrain (slope / aspect / hillshade) | ✓ built-in                    | ✗ `→gdaldem`       | ✗ `→xrspatial`               |
| Proximity / sieve / contour          | ✓ built-in                    | ◐ `features.sieve` | ✗ `→xrspatial`               |
| Interpolation / gridding             | ✓ `gdal.Grid`                 | ✗ `→scipy`         | ◐ `.interp` (regular)        |
| Connected-component clustering       | ✓ `cluster`                   | ✗ `→scipy`         | ✗ `→scipy`                   |
| Point sampling                       | ✓ `sample` / `point`          | ✓ `sample`         | ✓✓ `.sel(method=)`           |
| Band statistics (min/max/mean/std)   | ✓ `stats`                     | ◐ numpy            | ✓✓ `.mean` / `groupby`       |
| Histogram                            | ◐ viz only (`plot_histogram`) | ◐ numpy            | ✓ `.plot.hist`               |

### Raster ↔ vector

| Capability                | pyramids       | rasterio               | xarray                      |
|---------------------------|----------------|------------------------|-----------------------------|
| Rasterize vectors         | ✓ `rasterize`  | ✓ `features.rasterize` | ✗ `→geocube` / `regionmask` |
| Vectorize / polygonize    | ✓ `to_polygon` | ✓ `features.shapes`    | ✗ `→rasterio.features`      |
| Dataset footprint polygon | ✓ `footprint`  | ◐ `mask` + `shapes`    | ✗ `→rioxarray`              |

### Vector data (standalone)

| Capability                       | pyramids              | rasterio                 | xarray         |
|----------------------------------|-----------------------|--------------------------|----------------|
| Vector I/O (shapefile, GeoJSON…) | ✓ `FeatureCollection` | ✗ `→fiona` / `geopandas` | ✗ `→geopandas` |
| Geometry operations              | ✓                     | ✗ `→shapely`             | ✗ `→shapely`   |
| GeoParquet                       | ✓                     | ✗ `→geopandas`           | ✗ `→geopandas` |

### Multi-dimensional / datacube / formats

| Capability               | pyramids              | rasterio                     | xarray           |
|--------------------------|-----------------------|------------------------------|------------------|
| Time-series datacube     | ✓ `DatasetCollection` | ✗ `→rioxarray` / `stackstac` | ✓✓✓ the standard |
| NetCDF + CF conventions  | ✓✓ first-class        | ◐ subdatasets only           | ✓✓✓ first-class  |
| UGRID unstructured grids | ✓                     | ✗                            | ◐ `→uxarray`     |
| Zarr                     | ✓                     | ◐ GDAL driver                | ✓✓ native        |
| GRIB                     | ✓ (+ WMO glossary)    | ◐ via GDAL                   | ◐ `→cfgrib`      |

### Cloud, STAC & lazy compute

| Capability                     | pyramids         | rasterio           | xarray                      |
|--------------------------------|------------------|--------------------|-----------------------------|
| Cloud VSI (s3 / gs / az)       | ✓ `remote`       | ✓ `/vsi*/`         | ◐ `→fsspec`                 |
| STAC search / load / mosaic    | ✓✓ `stac/`       | ✗ `→pystac-client` | ✗ `→stackstac` / `odc-stac` |
| Requester-pays / token signing | ✓ `Signer`       | ◐ `AWSSession`     | ✗ `→fsspec`                 |
| Lazy / Dask-backed arrays      | ✓ Dask `chunks=` | ✗ `→rioxarray`     | ✓✓✓ the standard            |
| Concurrent windowed reads      | ◐                | ✓✓                 | ✓ via Dask                  |

### Tooling & maturity

| Capability                      | pyramids       | rasterio          | xarray            |
|---------------------------------|----------------|-------------------|-------------------|
| CLI                             | ✓ `pyramids`   | ✓ `rio`           | ✗                 |
| Plotting                        | ◐ `→cleopatra` | ✓ `rasterio.plot` | ✓✓ `xarray.plot`  |
| Maturity / adoption / community | younger        | ✓✓✓ standard      | ✓✓✓ huge (Pangeo) |
| Stability / docs depth          | growing        | ✓✓✓               | ✓✓✓               |

### Honest summary

- **pyramids' real strengths:** breadth in **one** GDAL/GIS-first package — raster **and** vector **and**
  datacube; NetCDF/CF/UGRID, STAC, terrain / zonal / interpolation, COG tooling, and lazy/Dask, without
  stitching together several libraries.
- **When to pick which:**
  - **pyramids** — you want an integrated, batteries-included, CRS-aware GDAL toolkit covering raster,
    vector, and datacubes together.

> Scope reminder: pyramids stays a *generic* GDAL/OGR toolkit — the breadth above is generic primitives
> and format support, not domain logic. See [Scope](SCOPE.md) for the boundary.
