# Comparison with other GIS packages

How pyramids relates to the established Python geospatial stack. This page starts with
[rasterio](https://rasterio.readthedocs.io/) — the closest single-library comparison — and is the place
to add comparisons against other packages (rioxarray, fiona/geopandas, stackstac, …) over time.

## How to read this — a fairness note

A raw feature checklist would be **unfair to rasterio**. rasterio is deliberately a *focused raster-core*
library (Unix-philosophy): most things it does not do itself are provided, often more maturely, by its
sibling packages — `fiona`/`geopandas` (vector), `rasterstats` (zonal), `rioxarray`/`xarray`/`stackstac`
(datacube/NetCDF), `rio-cogeo` (COG), `shapely` (geometry), `pystac-client`/`odc-stac` (STAC). pyramids
instead bundles raster + vector + datacube + NetCDF + STAC into **one** package.

So this compares **what each single library ships**, not what each *ecosystem* can do. Many "✗ core" rows
below are filled — capably — by a rasterio companion package, noted as `→pkg`.

Legend: ✓ built-in · ✓✓ a particular strength · ◐ partial / needs manual wiring · ✗ not provided ·
`→pkg` provided by an ecosystem package.

## pyramids vs rasterio

### Core raster I/O

| Capability | pyramids | rasterio |
|---|---|---|
| Read/write GDAL raster formats (GeoTIFF, etc.) | ✓ `Dataset` | ✓ (de-facto standard) |
| Windowed / block-level read & write | ◐ `read_part`, `read_tile`, `read_array(bbox=)` | ✓✓ `Window` API |
| In-memory raster (bytes) | ✓ `from_bytes` / `to_bytes` | ✓ `MemoryFile` |
| Overviews / pyramids | ✓ `create_overviews` | ✓ `build_overviews` |
| COG write / validate / inspect | ✓✓ `to_cog`, `validate`, `info` | ◐ COG driver · `→rio-cogeo` |
| Decimated reads (preview / tile / point) | ✓ `preview`, `read_tile`, `point` | ◐ via `out_shape` |
| No-data, masks, color tables / interpretation | ✓ | ✓ |

### CRS, warping & alignment

| Capability | pyramids | rasterio |
|---|---|---|
| Reproject / warp | ✓ `to_crs`, `reproject` | ✓ `warp.reproject`, `WarpedVRT` |
| Resample | ✓ `resample` | ✓ (resampling enums) |
| Align / snap to a target grid | ✓ `align` | ◐ manual (`calculate_default_transform`) |
| Arbitrary CRS / affine transforms | ✓ | ✓✓ `Affine`, `rasterio.crs` |

### Raster analysis

| Capability | pyramids | rasterio |
|---|---|---|
| Crop by bbox / geometry | ✓ `crop`, `extract` | ✓ `mask.mask` |
| Mosaic / merge | ✓ `merge` | ✓ `merge.merge` |
| Zonal statistics | ✓ built-in `zonal` | ✗ core · `→rasterstats` |
| Terrain (slope / aspect / hillshade) | ✓ built-in | ✗ core · `→gdaldem` / `richdem` |
| Proximity / sieve / contour | ✓ built-in | ◐ `features.sieve` only · rest `→gdal` |
| Interpolation / gridding | ✓ `gdal.Grid` bridge | ✗ core · `→scipy` / `gdal` |
| Connected-component clustering | ✓ `cluster` | ✗ core · `→scipy.ndimage` |
| Point sampling | ✓ `sample`, `point` | ✓ `sample.sample_gen` |
| Band statistics / histogram | ✓ `stats` | ◐ via numpy / band stats |

### Raster ↔ vector

| Capability | pyramids | rasterio |
|---|---|---|
| Rasterize vectors | ✓ `rasterize` | ✓ `features.rasterize` |
| Vectorize / polygonize | ✓ `to_polygon`, `vectorize` | ✓ `features.shapes` |
| Dataset footprint polygon | ✓ `footprint` | ◐ `dataset.mask` + `shapes` |

### Vector data (standalone)

| Capability | pyramids | rasterio |
|---|---|---|
| Vector I/O (shapefile, GeoJSON, …) | ✓ `FeatureCollection` | ✗ out of scope · `→fiona` / `geopandas` |
| Geometry operations | ✓ | ✗ · `→shapely` / `geopandas` |
| GeoParquet | ✓ (STAC + vector) | ✗ · `→geopandas` / `pyarrow` |

### Multi-dimensional / datacube / formats

| Capability | pyramids | rasterio |
|---|---|---|
| Time-series datacube | ✓ `DatasetCollection` | ✗ · `→rioxarray` / `xarray` / `stackstac` |
| NetCDF + CF conventions (time / vars) | ✓✓ first-class | ◐ subdatasets only, no CF/time · `→rioxarray` |
| UGRID unstructured grids | ✓ | ✗ |
| Zarr | ✓ | ◐ GDAL Zarr driver · `→rioxarray` |
| GRIB | ✓ (+ WMO field glossary) | ◐ read via GDAL |

### Cloud, STAC & lazy compute

| Capability | pyramids | rasterio |
|---|---|---|
| Cloud VSI (`s3://` / `gs://` / `az://`) | ✓ `remote`, VSI rewrite | ✓ `Session` / env, `/vsi*/` |
| STAC search / load / VRT-mosaic | ✓✓ built-in `stac/` | ✗ · `→pystac-client` / `stackstac` / `odc-stac` |
| Requester-pays / token signing | ✓ generic `Signer` framework | ◐ `AWSSession` / env |
| Lazy / Dask-backed arrays | ✓ `LazyDataset` / `LazyCollection` / lazy NetCDF | ✗ core · `→rioxarray` + Dask |
| Thread-safe concurrent windowed reads | ◐ | ✓✓ a core strength |

### Tooling & maturity

| Capability | pyramids | rasterio |
|---|---|---|
| CLI | ✓ `pyramids` CLI | ✓ `rio` |
| Plotting | ◐ via `cleopatra` (optional `[viz]`) | ✓ `rasterio.plot` |
| Maturity / adoption / community | younger, evolving API | ✓✓✓ de-facto standard, large community |
| Stability / battle-testing / docs depth | growing | ✓✓✓ very mature |

### Honest summary

- **rasterio's real strengths:** maturity, stability, enormous adoption, the most granular windowed-I/O
  API, strong concurrency, and a deep, composable ecosystem. Its narrow core is a *feature* — you
  assemble exactly what you need from fiona / rasterstats / rioxarray / stackstac / etc.
- **pyramids' real strengths:** breadth in **one** package — raster **and** vector **and** datacube;
  NetCDF/CF/UGRID, STAC, terrain / zonal / interpolation, COG tooling, and lazy/Dask all first-class
  without stitching together five libraries.
- **When to pick which:** reach for pyramids when you want an integrated, batteries-included GDAL-backed
  toolkit; reach for rasterio (+ its ecosystem) when you want a mature, minimal, highly-composable raster
  core and are happy to add the pieces you need.

> Scope reminder: pyramids stays a *generic* GDAL/OGR toolkit. The breadth above is generic primitives and
> format support — not domain logic. See [Scope](SCOPE.md) for the boundary.
