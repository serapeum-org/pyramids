# Architecture Diagrams

Below are core diagrams describing the system at multiple levels.

## C4: System Context

```mermaid
flowchart LR
  user(User) -->|Provides GIS data paths & commands| pyramids{{pyramids package}}
  ext1[(Raster files\nGeoTIFF/ASC/NetCDF)] --> pyramids
  ext2[(Vector files\nShapefile/GeoJSON/GPKG)] --> pyramids
  pyramids --> out1[(Processed rasters\nGeoTIFF/ASC)]
  pyramids --> out2[(Processed vectors\nGeoJSON/GPKG)]
```

## C4: Containers

```mermaid
flowchart TB
  subgraph Runtime Process
    A[Dataset]:::c --> B[_io]
    C[DatasetCollection]:::c --> B
    D[FeatureCollection]:::c --> B
    A --> E[_utils]
    C --> E
    D --> E
  end
  classDef c fill:#eef,stroke:#88f
```

## C4: Components

```mermaid
flowchart LR
  io[_io: read_file, to_ascii, path parsing]
  utils[_utils: geometry/index helpers]
  ds[dataset.Dataset]
  abs[dataset.RasterBase]
  dc[dataset.DatasetCollection]
  fc[feature.FeatureCollection]

  abs --> ds
  ds --> io
  dc --> ds
  fc --> io
  ds --> utils
  fc --> utils
```

## UML Class: Raster Core

```mermaid
classDiagram
  class RasterBase {
    <<abstract>>
    +read_file(path, read_only)
    +to_file(path, band)
  }
  class Dataset {
    +read_file(path, read_only, file_i)
    +to_file(path, band, tile_length)
    +read()
  }
  RasterBase <|-- Dataset
```

## UML Class: Vector Core

```mermaid
classDiagram
  class FeatureCollection {
    +read_file(path)
    +to_file(path, driver)
  }
```

## Sequence: Read Raster from Zip

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  participant IO as _io
  U->>DS: Dataset.read_file("dem.zip!dem.tif")
  DS->>IO: _parse_path(path)
  IO-->>DS: zip path + inner file
  DS->>IO: read_file(...)
  IO-->>DS: array + meta
  DS-->>U: Dataset instance
```

## Sequence: Save Raster to GeoTIFF

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  U->>DS: to_file("out.tif")
  DS-->>U: writes file
```

## Sequence: Build DatasetCollection from Folder

```mermaid
sequenceDiagram
  participant U as User
  participant DC as DatasetCollection
  participant DS as Dataset
  U->>DC: from_folder("./rasters/*.tif")
  DC->>DS: open each file
  DS-->>DC: Dataset objects
  DC-->>U: DatasetCollection
```

## Sequence: Zonal Statistics

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  participant FC as FeatureCollection
  U->>FC: read_file(polygons.gpkg)
  U->>DS: read_file(raster.tif)
  FC->>DS: zonal_stats(raster)
  DS-->>U: table
```

## Sequence: Align and Resample

```mermaid
sequenceDiagram
  participant U as User
  participant DS as Dataset
  U->>DS: align_to(reference)
  DS-->>U: aligned dataset
```

## Dependency Graph (Modules)

```mermaid
flowchart LR
  abstract_dataset --> dataset
  _io --> dataset
  _utils --> dataset
  dataset --> dataset_collection
  _io --> feature
  _utils --> feature
```

- brief class diagram for the `Dataset` class and related components:

```mermaid
classDiagram
    %% configuration class
    class config_Config {
        +__init__(config_file)
        +load_config()
        +initialize_gdal()
        +dynamic_env_variables()
        +set_env_conda()
        +set_env_os()
    }

    %% abstract base class for rasters
    class abstract_dataset_RasterBase {
        +__init__(src, access)
        +values() np.ndarray
        +rows() int
        +columns() int
        +shape() (bands, rows, cols)
        +geotransform()
        +top_left_corner()
        +epsg() int
        +crs()
        +cell_size() int
        +no_data_value
        +meta_data()
    }

    %% concrete raster class
    class dataset_Dataset {
        +__init__(src, access)
        +read_file(path, read_only)
        +create_from_array(array, top_left_corner, cell_size, epsg)
        +read_array(band, window)
        +to_file(path, driver)
        +align(alignment_src, data_src)
        +resample(cell_size, method)
        +crop(window)
        +plot(title, ticks_spacing, cmap, color_scale, vmin, cbar_label)
    }

    %% NetCDF: raster class specialised for NetCDF variables
    class netcdf_NetCDF {
        +__init__(src, access)
        +get_variable_names()
        +get_variables()
        +time_stamp()
        +lat()
        +lon()
    }

    %% DataCube: stack of rasters
    class collection_DatasetCollection {
        +__init__(src, time_length, files)
        +create_cube(src, dataset_length)
        +read_multiple_files(path, with_order, regex_string, date, file_name_data_fmt, start, end, fmt, extension)
        +base() Dataset
        +files() list
        +time_length() int
        +shape() (time, rows, cols)
        +rows() int
        +columns() int
    }

    %% FeatureCollection for vector data
    class feature_FeatureCollection {
        +__init__(gdf)
        +GetXYCoords()
        +GetPointCoords()
        +GetLineCoords()
        +GetPolyCoords()
        +Explode()
        +CombineGeometrics()
        +GCSDistance()
        +ReprojectPoints()
        +AddSpatialReference()
        +WriteShapefile()
    }

    %% Driver catalog
    class _utils_Catalog {
        +__init__(raster_driver)
        +get_driver(driver)
        +get_gdal_name(driver)
        +get_driver_by_extension(extension)
        +get_extension(driver)
        +exists(driver)
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
    abstract_dataset_RasterBase <|-- dataset_Dataset
    dataset_Dataset <|-- netcdf_NetCDF

    %% composition/usage relations
    collection_DatasetCollection --> dataset_Dataset : "base raster"
    abstract_dataset_RasterBase ..> _utils_Catalog : "uses Catalog constant"
    abstract_dataset_RasterBase ..> feature_FeatureCollection : "vector ops"
    dataset_Dataset ..> feature_FeatureCollection : "vector ops"
    feature_FeatureCollection ..> _utils_Catalog : "uses drivers"
    dataset_Dataset ..> _errors_ReadOnlyError : "raises"
    dataset_Dataset ..> _errors_AlignmentError : "raises"
    dataset_Dataset ..> _errors_NoDataValueError : "raises"
    dataset_Dataset ..> _errors_FailedToSaveError : "raises"
    dataset_Dataset ..> _errors_OutOfBoundsError : "raises"
    collection_DatasetCollection ..> _errors_DatasetNotFoundError : "raises"
    feature_FeatureCollection ..> _errors_DriverNotExistError : "raises"
    netcdf_NetCDF ..> _errors_OptionalPackageDoesNotExist : "raises"
    config_Config ..> dataset_Dataset : "initialises raster settings"

```

- Detailed class diagram:

```mermaid
classDiagram
    %% configuration class
    class config_Config {
        +__init__(config_file)
        +load_config()
        +initialize_gdal()
        +set_env_conda()
        +dynamic_env_variables()
        +setup_logging()
        +set_error_handler()
    }

    %% abstract base class for rasters
    class abstract_dataset_RasterBase {
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
        +plot(band, exclude_value, rgb, surface_reflectance, cutoff, overview, overview_index, **kwargs)
    }

    %% concrete raster class
    class dataset_Dataset {
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
        +plot(band, exclude_value, rgb, surface_reflectance, cutoff, overview, overview_index, **kwargs)
        +to_file(path, driver, band)
        +to_crs(to_epsg, method, maintain_alignment)
        +resample(cell_size, method)
        +align(alignment_src)
        +crop(mask, touch)
        +merge(src, dst, no_data_value, init, n)
        +apply(ufunc)
        +overlay(classes_map, exclude_value)
    }

    %% NetCDF: raster class specialised for NetCDF variables
    class netcdf_NetCDF {
        +__init__(src, access)
        +__str__()
        +__repr__()
        +lon()
        +lat()
        +x()
        +y()
        +get_y_lat_dimension_array(pivot_y, cell_size, rows)
        +get_x_lon_dimension_array(pivot_x, cell_size, columns)
        +variables()
        +no_data_value()
        +no_data_value(value)
        +file_name()
        +time_stamp()
        +read_file(path, read_only, open_as_multi_dimensional)
        +_get_time_variable()
        +_get_lat_lon()
        +_read_variable(var)
        +get_variable_names()
        +_read_md_array(variable_name)
        +get_variables(read_only)
        +is_subset()
        +is_md_array()
        +create_main_dimension(group, dim_name, dtype, values)
        +create_from_array(arr, geo, bands_values, epsg, no_data_value, driver_type, path, variable_name)
        +_create_netcdf_from_array(arr, variable_name, cols, rows, bands, bands_values, geo, epsg, no_data_value, driver_type, path)
        +_add_md_array_to_group(dst_group, var_name, src_mdarray)
        +add_variable(dataset, variable_name)
        +remove_variable(variable_name)
    }

    %% DataCube: stack of rasters
    class collection_DatasetCollection {
        +__init__(src, time_length, files)
        +__str__()
        +__repr__()
        +base()
        +files()
        +time_length()
        +rows()
        +columns()
        +shape()
        +create_cube(src, dataset_length)
        +read_multiple_files(path, with_order, regex_string, date, file_name_data_fmt, start, end, fmt, extension)
        +open_datacube(band)
        +values()
        +values(val)
        +__getitem__(key)
        +__setitem__(key, value)
        +__len__()
        +__iter__()
        +head(n)
        +tail(n)
        +first()
        +last()
        +iloc(i)
        +plot(band, exclude_value, **kwargs)
        +to_file(path, driver, band)
        +to_crs(to_epsg, method, maintain_alignment)
        +crop(mask, inplace, touch)
        +align(alignment_src)
        +merge(src, dst, no_data_value, init, n)
        +apply(ufunc)
        +overlay(classes_map, exclude_value)
    }

    %% FeatureCollection for vector data
    class feature_FeatureCollection {
        +__init__(gdf)
        +__str__()
        +feature()
        +epsg()
        +total_bounds()
        +top_left_corner()
        +layers_count()
        +layer_names()
        +column()
        +file_name()
        +dtypes()
        +read_file(path)
        +create_ds(driver, path)
        +_create_driver(driver, path)
        +_copy_driver_to_memory(ds, name)
        +to_file(path, driver)
        +_gdf_to_ds(inplace, gdal_dataset)
        +_ds_to_gdf_with_io(inplace)
        +_ds_to_gdf_in_memory(inplace)
        +GetXYCoords()
        +GetPointCoords()
        +GetLineCoords()
        +GetPolyCoords()
        +Explode()
        +MultiGeomHandler()
        +GetCoords()
        +XY()
        +CreatePolygon()
        +CreatePoint()
        +CombineGeometrics()
        +GCSDistance()
        +ReprojectPoints()
        +ReprojectPoints_2()
        +AddSpatialReference()
        +PolygonCenterPoint()
        +WriteShapefile()
    }

    %% Driver catalog
    class _utils_Catalog {
        +__init__(raster_driver)
        +get_driver(driver)
        +get_gdal_name(driver)
        +get_driver_by_extension(extension)
        +get_extension(driver)
        +exists(driver)
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
    abstract_dataset_RasterBase <|-- dataset_Dataset
    dataset_Dataset <|-- netcdf_NetCDF

    %% composition/usage relations
    collection_DatasetCollection --> dataset_Dataset : "base raster"
    abstract_dataset_RasterBase ..> _utils_Catalog : "uses Catalog constant"
    abstract_dataset_RasterBase ..> feature_FeatureCollection : "vector ops"
    dataset_Dataset ..> feature_FeatureCollection : "vector ops"
    feature_FeatureCollection ..> _utils_Catalog : "uses drivers"
    dataset_Dataset ..> _errors_ReadOnlyError : "raises"
    dataset_Dataset ..> _errors_AlignmentError : "raises"
    dataset_Dataset ..> _errors_NoDataValueError : "raises"
    dataset_Dataset ..> _errors_FailedToSaveError : "raises"
    dataset_Dataset ..> _errors_OutOfBoundsError : "raises"
    collection_DatasetCollection ..> _errors_DatasetNotFoundError : "raises"
    feature_FeatureCollection ..> _errors_DriverNotExistError : "raises"
    netcdf_NetCDF ..> _errors_OptionalPackageDoesNotExist : "raises"
    config_Config ..> dataset_Dataset : "initialises raster settings"

```
