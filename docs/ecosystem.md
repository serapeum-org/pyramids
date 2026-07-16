# Ecosystem & related projects

Where pyramids sits in the Python geospatial / scientific-array stack — the libraries it **builds on**, the
**companion** tools in its own family, and the **peers** you might reach for instead or alongside.

## Built on

pyramids is a high-level, object-oriented layer over these; you rarely call them directly.

| Library | Role in pyramids |
|---------|------------------|
| [GDAL / OGR](https://gdal.org/) | the raster & vector I/O engine (all formats, warping, VSI cloud/archive access) |
| [PROJ](https://proj.org/) · [GEOS](https://libgeos.org/) | projections / CRS transforms · geometry predicates |
| [geopandas](https://geopandas.org/) · [shapely](https://shapely.readthedocs.io/) · [pyproj](https://pyproj4.github.io/pyproj/) | `FeatureCollection` internals — the `GeoDataFrame`, geometry ops, CRS objects |
| [numpy](https://numpy.org/) · [pandas](https://pandas.pydata.org/) | array & tabular data returned by reads |
| [dask](https://www.dask.org/) · [zarr](https://zarr.dev/) | lazy/chunked arrays and the parallel Zarr datacube writer (`[lazy]` extra) |
| [pystac / pystac-client](https://pystac.readthedocs.io/) | STAC catalog search & item loading (`[stac]` extra) |
| [pyarrow](https://arrow.apache.org/docs/python/) | GeoParquet I/O (`[parquet]` extra) |

## Companion tools (same family)

| Project | What it adds |
|---------|--------------|
| [cleopatra](https://github.com/serapeum-org/cleopatra) | **all plotting.** Every `plot*` method builds a cleopatra glyph — array rendering, basemaps, RGB, vector fields. Install via the `[viz]` extra. |
| hpc-utils (`hpc.indexing`) | pixel/index lookup utilities used by point sampling and extraction. |
| [serapeum-org](https://github.com/serapeum-org) | the umbrella org — pyramids, cleopatra, and shared CI/tooling live here. |

## Peers — and when to pick them

pyramids is a **GDAL/GIS-first** raster **+** vector **+** datacube toolkit. The array-first stack overlaps on
the datacube/NetCDF axis. A full, symbol-checked feature matrix lives in
[Comparison with other GIS packages](comparison-other-gis-packages.md); in short:

- [**rasterio**](https://rasterio.readthedocs.io/) (+ `rio-cogeo` / `rasterstats` / `rio-tiler`) — a mature,
  minimal, highly-composable raster core. Pick it when you want the smallest raster dependency and will add the
  pieces yourself.
- [**xarray**](https://docs.xarray.dev/) + [**rioxarray**](https://corteva.github.io/rioxarray/) — the standard
  for N-dimensional labelled arrays / large lazy datacubes, with a CRS-aware raster layer on top. Pick them when
  your problem is genuinely N-dimensional scientific arrays. pyramids can hand off to and from xarray
  (`NetCDF.to_xarray` / `from_xarray`).
- [**geopandas**](https://geopandas.org/) / [**fiona**](https://fiona.readthedocs.io/) — dedicated vector.
  `FeatureCollection` already wraps geopandas, so you can drop down to it anytime.

!!! info "Interop, not lock-in"
    Because pyramids is GDAL-backed and wraps a `GeoDataFrame`, moving data to rasterio, xarray/rioxarray, or
    geopandas is a one-liner (`to_xarray`, the underlying `.geometry`, or a written GeoTIFF/GeoParquet). Use
    pyramids for breadth in one package; drop to a peer for its niche strength.
