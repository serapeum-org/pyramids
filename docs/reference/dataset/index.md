# Dataset Class

## At a glance

```mermaid
flowchart LR
    CR["<b>create / read</b><br/>read_file · create_from_array<br/>from_features · from_band_files<br/>from_zarr · from_bytes"] --> DS(("Dataset"))

    DS --> PR["<b>properties</b><br/>rows · columns · band_count · band_names<br/>epsg · crs · cell_size · geotransform<br/>bbox · bounds · no_data_value · dtype"]
    DS --> AC["<b>access data</b><br/>read_array — window · bbox · chunks<br/>sample · extract · get_tile · read_part"]
    DS --> SP["<b>spatial</b><br/>crop · to_crs · warped_view · resample<br/>align · fill_gaps · wrap_longitude"]
    DS --> AN["<b>analysis</b><br/>stats · zonal_stats · apply · overlay<br/>map_blocks · slope · aspect · hillshade<br/>proximity · cluster"]
    DS --> ND["<b>no-data</b><br/>change_no_data_value · fill · get_mask"]
    DS --> VE["<b>vectorize</b><br/>to_feature_collection · contour · sieve"]
    DS --> VI["<b>visualize</b><br/>plot · plot_histogram · to_image<br/>color_table · create_overviews · preview"]
    DS --> WR["<b>write</b><br/>to_file — .tif · .nc · .asc<br/>to_cog · to_zarr · to_terrain_rgb"]
```

## Architecture — the engine layer

`Dataset` is a thin **facade**: each family of operations lives in its own engine
(`ds.io`, `ds.spatial`, …) and `ds.<method>(...)` forwards to `ds.<engine>.<method>(...)`.
The reference pages below are one per engine.

```mermaid
flowchart TB
    DS(("Dataset<br/>facade"))
    DS -->|ds.io| IO["<b>IO</b> · io.md<br/>read_array · write_array · to_file<br/>to_bytes · get_tile · to_xyz<br/>to_terrain_rgb · create_overviews"]
    DS -->|ds.spatial| SP["<b>Spatial</b> · spatial.md<br/>crop · to_crs · warped_view<br/>resample · align · wrap_longitude"]
    DS -->|ds.analysis| AN["<b>Analysis</b> · analysis.md<br/>stats · extract · sample · overlay<br/>proximity · masks · footprint · plot"]
    DS -->|ds.bands| BA["<b>Bands</b> · band_metadata.md<br/>attribute tables · colours<br/>add_band · change_no_data_value"]
    DS -->|ds.cell| CE["<b>Cell</b> · cell.md<br/>get_cell_coords / _polygons / _points<br/>map ↔ array coordinates"]
    DS -->|ds.georef| GE["<b>Georef</b> · georef.md<br/>GCPs · RPCs · orthorectify<br/>set_gcps · georeference"]
    DS -->|ds.vectorize| VE["<b>Vectorize</b> · vectorize.md<br/>contour · to_feature_collection<br/>cluster · translate"]
    DS -->|ds.cog| CG["<b>COG</b><br/>to_cog · validate_cog · info<br/>read_part · preview · read_tile"]
```

- Detailed class diagram for the `Dataset` class and related components:

```mermaid
classDiagram
    %% configuration class
    class Config {
    }

    %% abstract base class for rasters
    class RasterBase {
        +__init__(src, access)
        +__str__()
        +__repr__()
        +access()
        +raster()
        +raster(value)
        +values()
        +rows()
        +columns()
        +shape()
        +geotransform()
        +top_left_corner()
        +epsg()
        +epsg(value)
        +crs()
        +crs(value)
        +cell_size()
        +no_data_value()
        +no_data_value(value)
        +meta_data()
        +meta_data(value)
        +block_size()
        +block_size(value)
        +file_name()
        +driver_type()
        +read_file(path, read_only)
        +read_array(band, window)
        +_read_block(band, window)
        +plot(band, exclude_value, rgb, surface_reflectance, cutoff, overview, overview_index, percentile, basemap, **kwargs)
    }

    %% concrete raster class
    class Dataset {
        +__init__(src, access)
        +__str__()
        +__repr__()
        +access()
        +raster()
        +raster(value)
        +values()
        +rows()
        +columns()
        +shape()
        +geotransform()
        +epsg()
        +epsg(value)
        +crs()
        +crs(value)
        +cell_size()
        +band_count()
        +band_names()
        +band_names(name_list)
        +band_units()
        +band_units(value)
        +no_data_value()
        +no_data_value(value)
        +meta_data()
        +meta_data(value)
        +block_size()
        +block_size(value)
        +file_name()
        +driver_type()
        +scale()
        +scale(value)
        +offset()
        +offset(value)
        +read_file(path, read_only)
        +create_from_array(arr, top_left_corner, cell_size, epsg)
        +read_array(band, window)
        +_read_block(band, window)
        +_resolve_plot_band(band, rgb)
        +plot(band, exclude_value, rgb, surface_reflectance, cutoff, overview, overview_index, percentile, basemap, rgb_options, **kwargs)
        +to_file(path, driver, band)
        +to_crs(to_epsg, method, maintain_alignment)
        +resample(cell_size, method)
        +align(alignment_src)
        +crop(mask, touch)
        +merge(src, dst, no_data_value, init, n)
        +apply(ufunc)
        +overlay(classes_map, exclude_value)
    }



    %% Driver catalog
    class _utils_Catalog {
    }

    %% NetCDF
    class NetCDF {
    }

    %% error classes
    class _errors_ReadOnlyError
    class _errors_DatasetNotFoundError
    class _errors_NoDataValueError
    class _errors_AlignmentError
    class _errors_DriverNotExistError
    class _errors_FileFormatNotSupportedError
    class _errors_OptionalPackageDoesNotExist
    class _errors_FailedToSaveError
    class _errors_OutOfBoundsError

    %% inheritance relations
    RasterBase <|-- Dataset
    Dataset <|-- NetCDF

    %% composition/usage relations
    RasterBase ..> _utils_Catalog : "uses Catalog constant"
    RasterBase ..> feature_FeatureCollection : "vector ops"
    Dataset ..> feature_FeatureCollection : "vector ops"
    Dataset ..> _errors_ReadOnlyError : "raises"
    Dataset ..> _errors_AlignmentError : "raises"
    Dataset ..> _errors_NoDataValueError : "raises"
    Dataset ..> _errors_FailedToSaveError : "raises"
    Dataset ..> _errors_OutOfBoundsError : "raises"
    NetCDF ..> _errors_OptionalPackageDoesNotExist : "raises"
    Config ..> Dataset : "initialises raster settings"

```


```mermaid
classDiagram

    %% Central dataset class with its main attributes
    class Dataset {
        +raster
        +cell_size
        +values
        +shape
        +rows
        +columns
        +pivot_point
        +geotransform
        +bounds
        +bbox
        +epsg
        +crs
        +lon
        +lat
        +x
        +y
        +band_count
        +band_names
        +variables
        +no_data_value
        +meta_data
        +dtype
        +gdal_dtype
        +numpy_dtype
        +file_name
        +time_stamp
        +driver_type
    }

    %% Group: visualisation functionality
    class Visualization {
        +plot()
        +overview_count()
        +read_overview_array()
        +create_overviews()
        +recreate_overviews()
        +get_overview()
    }
    Dataset --> Visualization : «visualisation»

    %% Group: data access methods
    class AccessData {
        +read_array()
        +get_variables()
        +count_domain_cells()
        +get_band_names()
        +extract()
        +stats()
    }
    Dataset --> AccessData : «data access»

    %% Group: mathematical operations on raster values
    class MathOperations {
        +apply()
        +fill()
        +normalize()
        +cluster()
        +cluster2()
        +get_tile()
        +groupNeighbours()
    }
    Dataset --> MathOperations : «math ops»

    %% Group: spatial operations and reprojection
    class SpatialOperations {
        +to_crs()
        +resample()
        +align()
        +crop()
        +locate_points()
        +overlay()
        +extract()
        +footprint()
    }
    Dataset --> SpatialOperations : «spatial ops»

    %% Group: conversion to other data types
    class Conversion {
        +to_feature_collection()
    }
    Dataset --> Conversion : «conversion»

    %% Group: coordinate system handling
    class OSR {
        +create_sr_from_epsg()
    }
    Dataset --> OSR : «osr»

    %% Group: bounding‐box and bounds calculations
    class BBoxBounds {
        +calculate_bbox()
        +calculate_bounds()
    }
    Dataset --> BBoxBounds : «bbox/bounds»

    %% Group: CRS/EPSG getters
    class CrsEpsg {
        +get_crs()
        +get_epsg()
    }
    Dataset --> CrsEpsg : «crs/epsg»

    %% Group: latitude/longitude getters
    class LatLon {
        +get_lat_lon()
    }
    Dataset --> LatLon : «lat/lon»

    %% Group: band names management
    class BandNames {
        +get_band_names_internal()
        +set_band_names()
    }
    Dataset --> BandNames : «band names»

    %% Group: timestamp handling
    class TimeStamp {
        +get_time_variable()
        +read_variable()
    }
    Dataset --> TimeStamp : «time»

    %% Group: handling of no‐data values
    class NoDataValue {
        +set_no_data_value()
        +set_no_data_value_backend()
        +change_no_data_value_attr()
    }
    Dataset --> NoDataValue : «no data value»

    %% Group: helpers for creating GDAL datasets
    class GdalDataset {
        +create_empty_driver()
        +create_driver_from_scratch()
        +create_mem_gtiff_dataset()
    }
    Dataset --> GdalDataset : «gdal creation»

    %% Group: factory methods for creating Dataset objects
    class CreateObject {
        +from_gdal_dataset()
        +read_file()
        +create_from_array()
        +dataset_like()
        +from_bytes()
        +from_band_files()
        +from_archive()
    }
    Dataset --> CreateObject : «object factory»

```

## Factory methods at a glance

| Method | Use when |
|---|---|
| `read_file(path, vsi=…, file_i=…)` | Open a path, URL, or archive member (zip/tar/gzip). URLs auto-rewrite to `/vsi*`. |
| `from_bytes(data, suffix=".tif")` | The caller already holds the bytes (HTTP body, DB blob, S3 `get_object` payload). Backed by `/vsimem/`. |
| `from_band_files(paths)` | Stack N single-band rasters (one file per band) into one multi-band Dataset — the natural target for the `<asset>.<band>.tif` layout of GEE / Landsat / Sentinel downloads. |
| `from_archive(url_or_path, member_glob=…)` | Merge every matching member of a local or remote archive into one multi-band Dataset (composes `from_band_files` over `gdal.ReadDir`). For one-Dataset-per-member use `DatasetCollection.from_archive`. |
| `create_from_array(arr, …)` | Build a Dataset from a numpy array + geobox. |
| `dataset_like(template, arr)` | Stamp a new Dataset that inherits its grid / CRS from `template`. |

See the [Recipes](../../how-to/recipes.md) page for runnable examples
of each.

::: pyramids.dataset.Dataset
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
