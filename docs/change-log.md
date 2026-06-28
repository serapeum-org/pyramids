# Change log


## 0.38.0 (2026-06-28)

### BREAKING CHANGE

- opening a store now returns a Container and extracting a
variable returns a Variable (both NetCDF subclasses, so isinstance(x,
NetCDF) still holds); type(x) is NetCDF is no longer true for these.
Directly constructing NetCDF(...) is deprecated and emits a
DeprecationWarning — use NetCDF.read_file(...) / NetCDF.create_from_array(...)
(-> Container) and container.get_variable(...) (-> Variable). NetCDF remains
an isinstance-compatible base for one major version.

### Feat

- **base**: RasterLike protocol + numpy.typing array-return typing (ARC-18/ARC-19) (#648)
- **netcdf**: zero-copy lazy get_group view (ARC-12) (#642)
- **netcdf**: read remote NetCDF over OPeNDAP/THREDDS via dods:// URLs (#644)
- **netcdf**: stream create_from_array writes for dask inputs (ARC-11) (#637)
- **feature**: add FeatureCollection.from_wfs OGC Web Feature Service reader (#635)
- **wcs**: add Dataset.from_wcs OGC Web Coverage Service reader (#632)
- **wcs**: add Dataset.from_wcs OGC Web Coverage Service reader
                                                                                                              
  Add a version-normalising OGC WCS reader exposed as the                                                     
  Dataset.from_wcs classmethod (implementation in dataset/_wcs.py),                                           
  backed by GDAL's native WCS driver so the 1.0.0 vs 2.0.x GetCoverage                                        
  dialect fork is handled inside GDAL rather than hand-written.                                               
                                                                                                              
  - Caller supplies one lon/lat bbox plus optional resolution and                                             
    output_crs; the reader issues the version-correct GetCoverage.                                            
  - CRS shim: attach a caller-supplied coverage_crs when the server's                                         
    advertised CRS is absent from PROJ (e.g. SoilGrids EPSG:152160).                                          
  - Client-side bbox reprojection into the coverage's native CRS via                                          
    pyproj, densified with transform_bounds so distorted or large                                             
    windows stay covered.                                                                                     
  - GetCapabilities fetched once per endpoint (LRU-cached); an unknown                                        
    coverage raises ValueError, server errors raise the new WCSError.                                         
  - The windowed read goes through an in-memory dataset, so a                                                 
    non-raster ExceptionReport body can never be written to a .tif.                                           
  - No new dependencies: GDAL handles transport and decode, pyproj                                            
    (core) the CRS transform, and Dataset the warp/IO.                                                        
  - Tests: pure helpers, the CRS shim, the capabilities cache, and a                                          
    protocol-faithful mock server that drives GDAL end-to-end for both                                        
    WCS dialects (asserting each call shape), plus gated live SoilGrids                                       
    tests; 100% line and branch coverage on the new module.                                                   
                                                                                                              
  Closes #626
- **netcdf**: split NetCDF into Container + Variable types (API-1) (#625)

### Fix

- **netcdf**: drop xarray interop tests duplicated in test_xarray_interop (#636)

### Refactor

- **netcdf**: extract engine subpackage from the NetCDF god-object (STR-1) (#627)
- **netcdf**: consolidate duplicated MDIM/CF/UGRID logic behind the stable API (#612)

## 0.36.0 (2026-06-23)

### Feat

- **netcdf**: add exploration example notebooks and supporting grid work (#581)
- **feature**: FeatureCollection GIS gaps — fishnet, IDW interpolation, PMTiles/MVT, GPX, FeatureServer and a native H3 engine(#591)
- **dataset**: add Dataset.to_terrain_rgb (terrain-RGB encoding + XYZ tiles) (#590)

### Fix

- **test**: skip optional-dependency tests on the bare-wheel core CI run (#610)
- **packaging**: bundle the JP2OpenJPEG driver so JP2-packed GRIB2 reads (#601)

## 0.35.0 (2026-06-16)

### Feat

- **feature**: add voronoi and quadtree tessellation ops to FeatureCollection (#562)

### Fix

- clear SonarCloud main quality gate (reliability, security, hotspots) (#577)
- **netcdf**: path-style S3 addressing for anonymous LabeledDataset reads (#569)
- **dataset**: array_to_map_coordinates honours non-square and rotated grids (#568)
- **netcdf**: carry string aux variables through container spatial ops (#567)
- **netcdf**: release the GDAL handle on close() so the file unlocks immediately (#566)
- **netcdf**: release the GDAL handle on close() so the file unlocks immediately
- **netcdf**: build kerchunk manifests natively to kill the zarr-v3 deadlock (#561)

## 0.34.0 (2026-06-14)

### Feat

- **collection**: RGB true-colour animation in DatasetCollection.plot (#557)
- **dataset**: GCP/RPC georeferencing + mask, window, CLI & parallel-read ergonomics (#536)

### Fix

- **netcdf**: resolve S3 bucket region for anonymous LabeledDataset reads (#537)

## 0.33.0 (2026-06-14)

### Feat

- NetCDF GDAL-multidim LabeledDataset rewrite, extras consolidation, wheel/CLI hardening (#520)
- **dataset**: decimated (out_shape) reads in the core read_array (#512)
- **dataset**: boundless windowed reads with fill values (#511)
- **dataset**: thread-safe parallel reads via per-thread GDAL handles (#510)
- **dataset**: add read_array(masked=True) returning a MaskedArray (#503)
- **dataset**: add first-class Window object with block iteration (#504)
- **dataset**: add warped_view - a lazy VRT-backed reprojected view (#508)
- **dataset**: expand resampling methods to the full GDAL set (#501)
- **dataset**: add Dataset.to_bytes for in-memory serialization (#502)
- **cli**: add info/bounds/clip/warp/merge/overview/sample/convert subcommands (#509)
- **dataset**: add xy()/rowcol() aliases and a GeoTransform object (#506)

### Fix

- **netcdf**: carry non-spatial aux variables through container spatial ops (#514)

## 0.32.0 (2026-06-08)

### Feat

- **dataset**: add create_empty / empty_like out-of-core raster allocators (#473)
- **netcdf**: surface time coord/length and window interleaved-layer variables (#468)

### Fix

- **collection**: render DatasetCollection.plot() when rasters have no nodata (#481)

### Refactor

- remove out-of-scope domain surfaces (grids, EO signers, natural_earth/relief) (#476)
- remove out-of-scope domain surfaces; deprecate the rest

### Perf

- **wheels**: trim niche GDAL_DATA and add wheel size/leak guardrails (#475)
- **wheels**: shrink the bundled platform wheel 18-27% and make wheel CI deterministic (#471)

## 0.31.0 (2026-06-03)

### Fix

- **bootstrap**: force GDAL_DRIVER_PATH to the bundled plugins (#465)

## 0.30.0 (2026-06-02)

### Feat

- **netcdf**: add NetCDF.subset for windowed gridded cloud-cube reads (#462)
- **basemap**: add relief() low-res global relief raster for offline backdrops (#461)

## 0.29.0 (2026-06-01)

### Feat

- **netcdf**: add LabeledDataset reader for label-indexed NetCDF/Zarr stores (#455)

## 0.28.0 (2026-05-31)

### Feat

- **plot**: upgrade cleopatra integration to 0.12.0 API and adopt new capabilities (#451)
- **io**: add decompress-aware resource reader (sniff → Dataset / FeatureCollection / DataFrame) (#448)

### Fix

- **netcdf**: georeference geostationary (GOES) reads via the classic driver (#452)

## 0.27.0 (2026-05-28)

### Feat

- **dataset**: accept arbitrary CRS in Dataset.to_crs (#444)

## 0.26.0 (2026-05-28)

### Refactor

- **zarr**: GeoZarr geobox, lazy read, cube round-trip, codecs, append + foreign/STAC reads (#417)

## 0.25.1 (2026-05-27)

### Fix

- **wheels**: vendor curl CA bundle so vendored-GDAL HTTPS reads work  (#414)

## 0.25.0 (2026-05-26)

### Feat

- add grid adapters, unit conversion, and Natural Earth features (#407)

## 0.24.1 (2026-05-26)

### Fix

- **crs**: resolve un-authority-tagged CRSes instead of returning child codes (#404)

## 0.24.0 (2026-05-26)

### Feat

- **cog**: unify write policy and add inspection, reads, profiles, and CLI (#402)

## 0.23.0 (2026-05-25)

### Feat

- **stac**: add GDAL-native STAC subpackage and fix no-data/windowed-read bugs (#363)

## 0.22.0 (2026-05-23)

### Feat

- **dataset**: extend merge_rasters and add DatasetCollection.reduce_time (#360)

## 0.21.0 (2026-05-21)

### Feat

- add cloud-provider readers and GDAL-native raster operations (#339)

## 0.20.0 (2026-05-19)

### Feat

- **wheels**: bundle GDAL into pip-installable platform wheels (#243)

### Fix

- **wheels**: only force platform tag in setup.py when PACKAGE_DATA=1

## 0.19.0 (2026-05-15)

### Feat

- add bytes/archive/band-stack/bbox/NetCDF/HTTP-knob entry points (#322)

## 0.18.0 (2026-05-11)

### Feat

- **plot**: xarray-aligned NetCDF.plot API and unified plotting layer (#316)

## 0.17.0 (2026-05-10)

### Feat

- **netcdf**: support 4-D+ NetCDFs and fix CDS-Beta metadata path (#313)

## 0.16.0 (2026-05-09)

### Fix

- **wheel-test**: unblock core matrix; add branch input to dispatch (#308)

### Refactor

- re-evaluate package architecture; close audit(#289)

## 0.15.0 (2026-04-24)

### Feat

- **dask**: integrate Dask across dataset, netcdf, collection, and feature paths (#253)

### Fix

- wheel-test fails for plot tests (#261)

### Refactor

- **feature**: modernize feature subpackage and expand I/O surface (#252)

## 0.14.0 (2026-04-14)

### Feat

- **cog**: add Cloud Optimized GeoTIFF support (write, validate, cloud I/O, stack export) (#249)

## 0.13.0 (2026-04-12)

### Feat

- **basemap**: add web tile basemap plotting support (#231)

## 0.12.0 (2026-04-11)

### Feat

- delegate UGRID plotting to cleopatra and add dunder methods to FeatureCollection (#230)
- **netcdf**: add UGRID unstructured mesh support (#224)

## 0.11.0 (2026-04-05)

### Feat

- **netcdf**: add CF conventions support and xarray interop (#198)

### Fix

- resolve architectural issues across dataset, netcdf, and feature modules (#197)

## 0.10.0 (2026-04-04)

### Feat

- **netcdf**: add variable operations, metadata refinement, and  rename ArrayInfo to VariableInfo (#172)
- **netcdf**: add variable operations, metadata refinement, and
  rename ArrayInfo to VariableInfo

### Refactor

- **dataset**: resolve 20 design issues from architectural review (#175)
- restructure package layout and decompose Dataset into mixins (#170)

## 0.9.1 (2026-03-29)

### Fix

- **packaging**: add GDAL to PyPI deps and fix missing YAML files in wheel (#162)

### Refactor

- **api**: replace os module with pathlib across all public APIs (#157)

## 0.9.0 (2026-03-27)

### Feat

- **logging**: Centralized logging via `LoggerManager` with colored console and optional file logging; reduced third‑party log noise. (#135)
- **logging**: Centralized logging via `LoggerManager` with colored console and optional file logging; reduced third‑party log noise. (#135)

### Refactor

- **typing**: modernize type annotations and add mypy (#146)
- **netcdf**: extract into subpackage with MDIM round-trip support (#91)

## 0.8.0 (2025-09-06)

### Feat

- **plot**: add to_xyz method and improve plotting integration with Cleopatra
- **docker**: add Docker support with workflow and documentation

### CI

- **docker**: fix docker-release workflow configuration

### Docs

- update badges and improve formatting in README and index files
- fix typo in dev container setup steps
- migrate to MkDocs with Google-style docstrings and diagrams

### Chore

- **deps**: disable changelog autoupdate - Disabled `update_changelog_on_bump` in pyproject settings - Updated styles in pyproject for improved terminal prompts - Simplified release workflow by removing unnecessary debug steps
- **release**: enhance debug visibility and tag handling in workflow
- **release**: ensure tags are fetched for release workflow
- **deps**: update sha256 checksum for `pyramids-gis` to match new version 0.7.3
- **pyproject**: remove `major_version_zero` configuration
- **release**: add Commitizen and automated release workflow
- update dependencies, workflows, and docs
- **dev**: add dev container and docs
- **ci**: update workflow permissions and group publishing dependencies


## 0.7.3 (2025-08-02)


### Dev
* add pixi configuration to the pyproject.toml file.
* relax the limits for the dependencies in the pyproject.toml file.
* gdal is installed from conda-forge channel.
* update the workflow to use the new pixi configuration.

## 0.7.2 (2025-01-12)


### Dev
* replace the setup.py with pyproject.toml
* automatic search for the gdal_plugins_path in the conda environment, and setting the `GDAL_DRIVER_PATH` environment
variable.
* add python 3.12 to CI.



## 0.7.1 (2024-12-07)

* update `cleopatra` package version to 0.5.1 and update the api to use the new version.
* update the miniconda workflow in ci.
* update gdal to 3.10 and update the DataSource to Dataset in the `FeatureCollection.file_name`.
* add `libgdal-netcdf` and `libgdal-hdf4` to the conda dependencies.


## 0.7.0 (2024-06-01)

* install viz, dev, all packages through pip.
* create a separate module for the netcdf files.
* add configuration file and module for setting gdal configurations.

### AbstractDataset
* add `meta_data` property to return the metadata of the dataset.
* add `access` property to indicate the access mode of the dataset.


### Dataset
* add extra parameter `file_i` to the `read_file` method to read a specific file in a compressed file.
* initialize the `GDAL_TIFF_INTERNAL_MASK` configuration to `No`
* the add the `access` parameter to the constructor to set the access mode of the dataset.
* add the `band_units` property to return the units of the bands.
* the `__str__` and the `__repr__` methods return string numpy like data type (instead of the gdal constant) of the
dataset.
* add `meta_data` property setter to set any key:value as a metadata of the dataset.
* add `scale` and `offset` properties to set the scale and offset of the bands.
* add `copy` method to copy the dataset to memory.
* add `get_attribute_table`set_attribute_table` method to get/set the attribute table of a specific band.
* the `plot` method uses the rgb bands defined in the dataset plotting (if exist).
* add `create` method to create a new dataset from scratch.
* add `write_array` method to write an array to an existing dataset.
* add `get_mask` method to get the mask of a dataset band.
* add `band_color` method to get the color assigned to a specific band (RGB).
* add `get_band_by_color` method to get the band index by its color.
* add `get_histogram` method to get/calculate  the histogram of a specific band.
* the `read_array` method takes and extra parameter `window` to lazily read a `window` of the raster, the window is
[xoff, yoff, x-window, y-window], the `window` can also be a geodataframe.
* add `get_block_arrangement` method divide the raster into tiles based on the block size.
* add tiff file writing options (compression/tile/tile_length)
* add `close` method to flush to desk and close a dataset.
* add `add_band` method to add an array as a band to an existing dataset.
* rename `pivot_point` to `top_left_corner` in the `create` method.
* the `to_file` method return a `Dataset` object pointing to the saved dataset rather than the need to re-read the
saved dataset after you save it.

### Datacube
* the `Datacube` is moved to a separate module `datacube`.

### NetCDF
* move all the netcdf related functions to a separate module `netcdf`.

### FeatureCollection
* rename the `pivot_point` to `top_left_corner`

### Deprecated
*Cropping a raster using a polygon is done now directly using gdal.wrap nand the the `_crop_with_polygon_by_rasterizing`
is deprecated.
* rename the interpolation method `nearest neighbour` to `nearest neighbor`.


## 0.6.0 (2024-02-24)

* move the dem module to a separate package "digital-rivers".



## 0.5.6 (2024-01-09)

### Dataset
* create `create_overviews`, `recreate_overview`, `read_overview_array` methods, and `overview_count` attribute to
handle overviews.
* The `plot` method takes an extra parameters `overviews` and `overview_index` to enable plotting overviews instead
of the real values in the bands.



## 0.5.5 (2024-01-04)

### Dataset
* Count domain cells for a specific band.



## 0.5.4 (2023-12-31)

### Dataset
* fix the un-updated array dimension bug in the crop method when the mask is a vector mask and the touch parameter is
True.



## 0.5.3 (2023-12-28)

### Dataset
* Introduce a new parameter touch to the crop method in the Dataset to enable considering the cells that most of the
cell lies inside the mask, not only the cells that lie entirely inside the mask.
* Introduce a new parameter inplace to the crop method in the Dataset to enable replacing the dataset object with the
new cropped dataset.
* Adjust the stats method to take a mask to calculate the stats inside this mask.


## 0.5.2 (2023-12-27)

### Dataset
* add _iloc method to get the gdal band object by index.
* add stats method to calculate the statistics of the raster bands.


## 0.5.1 (2023-11-27)

### Dataset
* revert the convert_longitude method to not use the gdal_wrap method as it is not working with the new version of gdal (newer tan 3.7.1).
* bump up versions.


## 0.5.0 (2023-10-01)

### Dataset
* The dtype attribute is not initialized in the __init__, but initialized when using the dtype property.
* Create band_names setter method.
* add gdal_dtype, numpy_dtype, and dtype attribute to change between the dtype (data type general name), and the corresponding data type in numpy and gdal.
* Create color_table attribute, getter & setter property to set a symbology (assign color to different values in each band).
* The read_array method returns array with the same type as the dtype of the first band in the raster.
* add a setter method to the band_names property.
* The methods (create_driver_from_scratch, get_band_names) is converted to private method.
* The no_data_value check the dtype before setting any value.
* The convert_longitude used the gdal.Wrap method instead of making the calculation step by step.
* The to_polygon is converted to _band_to_polygon private method used in the clusters method.
* The to_geodataframe is converted to to_feature_collection. change the backend of the function to use the crop function is a vector mask is given.
* the to_crs takes an extra parameter "inplace".
* The locate_points is converted to map_to_array_coordinates.
* Create array_to_map_coordinates to translate the array indices into real map coordinates.

### DataCube
* rename the read_separate_files to read_multiple_files, and enable it to use regex strings to filter files in a given directory.
* rename read_dataset to open_datacube.
* rename the data attribute to values

### FeatureCollection
* Add a pivot_point attribute to return the top left corner/first coordinates of the polygon.
* Add a layers_count property to return the number of layers in the file.
* Add a layer_names property to return the layers names.
* Add a column property to return column names.
* Add the file_name property to store the file name.
* Add the dtypes property to retrieve the data types of the columns in the file.
* Rename bounds to total_bounds.
* The _gdf_to_ds can convert the GeoDataFrame to a ogr.DataSource and to a gdal.Dataset.
* The create_point method returns a shapely point object or a GeoDataFrame if an epsg number is given.


## 0.4.2 (2023-04-27)

* fix bug in plotting dataset without specifying the band
* fix bug in passing ot not passing band index in case of multi band rasters
* change the bounds in to_dataset method to total_bounds tp get the bbox of the whole geometries in the gdf
* add convert_longitude method to convert longitude to range between -180 and 180


## 0.4.1 (2023-04-23)

* adjust all spatial operation functions to work with multi-band rasters.
* use gdal exceptions to capture runtime error of not finding the the file.
* add cluster method to dataset class.
* time_stamp attribute returns None if there is no time_stamp.
* restructure the no_data_value related functions.
* plot function can plot rgb image for multi-band rasters.
* to_file detect the driver type from the extension in the path.


## 0.4.0 (2023-04-11)

* Restructure the whole package to two main objects Dataset and FeatureCollection
* Add class for multiple Dataset "DataCube".
* Link both Dataset and FeatureCollection to convert between raster and vector data types.
* Remove rasterio and netcdf from dependencies and depend only on gdal.
* Test read rasters/netcdf from virtual file systems (aws, compressed)
* Add dunder methods for all classes.
* add plotting functionality and cleopatra (plotting package) as an optional package.
* remove loops and replace it with ufunc from numpy.


## 0.3.3 (2023-02-06)

* fix bug in reading the ogr drivers catalog for the vector class
* fix bug in creating rasterLike in the asciiToRaster method


## 0.3.2 (2023-01-29)

* refactor code
* add documentation
* fix creating memory driver with compression in _createDataset


## 0.3.1 (2023-01-25)

* add pyarrow to use parquet data type for saving dataframes and geodataframes
* add H3 indexing package, and add new module indexing with functions to convert geometries to indices back and forth.
* fix bug in calculating pivot point of the netcdf file
* rasterToDataFrame function will create geometries of the cells only based on the add_geometry parameter.


## 0.3.0 (2023-01-23)


* add array module to deal with any array operations.
* add openDataset, getEPSG create SpatialReference, and setNoDataValue utility function, getCellCoords, ...
* add rasterToPolygon, PolygonToRaster, rasterToGeoDataFrame, conversion between ogr DataSource and GeoDataFrame.


## 0.2.11 (2023-01-14)


* add utils module for functions dealing with compressing files, and others utility functions


## 0.2.11 (2022-12-27)


* fix bug in pypi package names in requirements.txt file


## 0.2.10 (2022-12-25)


* lock numpy version to 1.23.5 as conda-forge can not install 1.24.0


## 0.2.9 (2022-12-19)


* Use environment.yaml and requirements.txt instead of pyproject.toml and replace poetry env by conda env


## 0.1.0 (2022-05-24)



* First release on PyPI.
