# `NetCDF` — public API

A one-line map of every public member the `NetCDF` class itself defines — 65 in all: 38 methods, 20 properties,
6 classmethods and 1 staticmethod. For the full signatures, arguments and examples, see the rendered
[NetCDF Class](index.md) reference; this page is the index you scan to find the member you want.

`NetCDF` extends `Dataset`, so it also inherits a further 119 public members it does not redefine — band
handling, the COG surface, and the rest of the raster API. Those live in the
[Dataset reference](../dataset/index.md).

Two object shapes share this class, and several members behave differently across them:

- a **Container** — what `read_file` / `from_bytes` / `from_array` return; `band_count == 0`; describes the file.
- a **Variable** — what `get_variable` / `variables[name]` / `sel` / `subset` return; `band_count >= 1`;
  behaves as a single raster.

---

## Opening, construction and lifetime

| Member             | What it does                                                                     |
|--------------------|----------------------------------------------------------------------------------|
| `read_file()`      | Opens a `.nc` from a path, URL, or archive member; returns a Container.          |
| `from_bytes()`     | Opens a NetCDF held in memory as a byte string.                                  |
| `from_array()`     | Builds a Container from a NumPy array plus a geo-reference.                      |
| `from_xarray()`    | Builds a `NetCDF` from an `xarray.Dataset` (needs the optional xarray peer dep). |
| `open_mfdataset()` | Opens many files and stacks one variable into a single lazy dask array.          |
| `copy()`           | Deep, standalone copy of this dataset, optionally written to `path`.             |
| `close()`          | Releases every GDAL handle this container holds, then closes the base.           |

## Variables, dimensions and groups

| Member                   | What it does                                                              |
|--------------------------|---------------------------------------------------------------------------|
| `variable_names`         | Names of the data variables, excluding dimension coordinate arrays.       |
| `variables`              | Lazy `{name: subset}` mapping of every data variable.                     |
| `get_variable()`         | Extracts one variable as a classic-raster `NetCDF` (a Variable).          |
| `get_variable_names()`   | Deprecated alias for the `variable_names` property.                       |
| `dimension_names`        | Names of all dimensions, in storage order.                                |
| `dimension_sizes`        | `{name: size}` for every dimension, read from the multidimensional group. |
| `get_dimension_values()` | Stored coordinates of any dimension — `level`, `depth`, `member`, `time`. |
| `group_names`            | Names of the sub-groups in the root group.                                |
| `get_group()`            | Opens a netCDF-4 sub-group as its own Container, without copying data.    |
| `is_subset`              | Whether this object is a single-variable subset rather than a Container.  |
| `is_md_array`            | Whether the dataset was opened in multidimensional mode.                  |
| `file_name`              | The file path, with any `NETCDF:"path":var` prefix stripped.              |

## Coordinates and time

| Member                    | What it does                                                             |
|---------------------------|--------------------------------------------------------------------------|
| `lon` / `x`               | Longitude / x coordinate values as a 1-D array.                          |
| `lat` / `y`               | Latitude / y coordinate values as a 1-D array.                           |
| `top_left_corner`         | Coordinates of the raster's top-left corner.                             |
| `geotransform`            | Geotransform derived from the coordinate arrays, then cached.            |
| `epsg`                    | EPSG code, resolved from the variables when asked of a Container.        |
| `time_stamp`              | Time coordinate values parsed from the CF `time` variable.               |
| `get_time_variable()`     | Decodes the time axis to date strings; `time_format` sets the precision. |
| `get_time_values()`       | Raw, undecoded values of the time axis — the time spelling of the above. |
| `create_main_dimension()` | Creates a dimension with its indexing variable (static helper).          |

## Reading and selecting

| Member         | What it does                                                                    |
|----------------|---------------------------------------------------------------------------------|
| `read_array()` | Reads a variable eagerly, or lazily into dask when `chunks` is given.           |
| `subset()`     | Reads a windowed `(variable, time, bbox)` slice without materialising the cube. |
| `sel()`        | Selects bands by coordinate value, date label, or `method="nearest"`.           |
| `reduce()`     | Reduces every variable along a named dimension (`how`, `groupby`, `skipna`).    |

## Spatial operations

| Member          | What it does                                                         |
|-----------------|----------------------------------------------------------------------|
| `crop()`        | Crops by polygon mask, raster mask, or bbox tuple.                   |
| `to_crs()`      | Reprojects the dataset to another CRS.                               |
| `resample()`    | Resamples to a different cell size.                                  |
| `warped_view()` | A lazy, reprojected view of a Variable — no data is read until used. |

## Editing variables in place

| Member                 | What it does                                                                |
|------------------------|-----------------------------------------------------------------------------|
| `set_variable()`       | Writes a classic `Dataset` back into this container as an MDArray variable. |
| `add_variable()`       | Copies MDArray variables in from another `NetCDF`.                          |
| `remove_variable()`    | Deletes a variable from this container.                                     |
| `rename_variable()`    | Renames a variable in this container.                                       |
| `crop_variable()`      | Crops one variable and stores the result back.                              |
| `reproject_variable()` | Reprojects one variable and stores the result back.                         |
| `resample_variable()`  | Resamples one variable and stores the result back.                          |

## Metadata and attributes

| Member | What it does |
|---|---|
| `meta_data` | Structured metadata for this NetCDF (cached). |
| `get_all_metadata()` | The same, re-traversed from GDAL rather than served from cache. |
| `global_attributes` | Global attributes on the root group. |
| `set_global_attribute()` | Sets one global attribute on the root group. |
| `delete_global_attribute()` | Deletes one global attribute from the root group. |
| `scale` | Per-band CF `scale_factor` a read applies, per band. |
| `offset` | Per-band CF `add_offset` a read applies, per band. |

## Writing and interop

| Member | What it does |
|---|---|
| `to_file()` | Saves the dataset to disk. |
| `to_xarray()` | Converts the container to an `xarray.Dataset`, optionally chunked. |
| `to_kerchunk()` | Emits a kerchunk JSON reference manifest for this file. |
| `combine_kerchunk()` | Combines per-file manifests into one cube index. |
| `to_cog()` | Writes a Cloud-Optimized GeoTIFF. Variable only. |
| `to_feature_collection()` | Converts the raster to a vector `FeatureCollection`. Variable only. |
| `write_array()` | Writes an array into the raster bands. Variable only. |

## Analysis and plotting

| Member | What it does |
|---|---|
| `plot()` | Plots a 2-D slice — `selectors`, `facet`, `axes`, `animate`, `chunks`, colour. |
| `stats()` | Per-band summary statistics. Variable only. |
| `slope()` | Slope raster from an elevation variable. Variable only. |
| `hillshade()` | Hillshade raster from an elevation variable. Variable only. |
| `zonal_stats()` | Statistics per zone of a mask or feature set. Variable only. |
| `sample()` | Samples values at points. Variable only. |

---

**"Variable only"** marks the members that are container-guarded: they are `Dataset` operations that need a
single raster, so calling them on a Container raises with a message pointing at `get_variable`.
