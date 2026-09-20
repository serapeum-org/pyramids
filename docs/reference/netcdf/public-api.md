# `NetCDF` — public API

A one-line map of every public member the `NetCDF` class itself defines — 96 in all: 62 methods, 27 properties,
6 classmethods and 1 staticmethod, plus the four mapping dunders (`__getitem__`, `__contains__`, `__iter__`,
`__len__`). `concat` and `merge` are counted among the methods: each is callable on the class
(`NetCDF.concat([a, b], dim)`) and on a cube (`a.concat([b], dim)`, which joins the
receiver first). For the full signatures, arguments and examples, see the rendered
[NetCDF Class](index.md) reference; this page is the index you scan to find the member you want.

`NetCDF` extends `Dataset`, so it also inherits a further 137 public members it does not redefine — band
handling, the COG surface, the missing-data members (`where`, `fillna`, `isnull`, `notnull`, `equals`,
`identical`) and the rest of the raster API. 124 of those are declared in `Dataset`'s own body and the
remaining 13 come from `RasterBase` above it. They live in the
[Dataset reference](../dataset/index.md), the six named above on its
[Analysis page](../dataset/analysis.md#missing-data-and-comparison).

Two object shapes share this class, and several members behave differently across them:

- a **Container** — what `read_file` / `from_bytes` / `from_array` return; `band_count == 0`; describes the file.
- a **Variable** — what `get_variable` / `variables[name]` / `sel` / `subset` return; `band_count >= 1`;
  behaves as a single raster.

---

## Opening, construction and lifetime

| Member          | What it does                                                                     |
|-----------------|----------------------------------------------------------------------------------|
| `read_file()`   | Opens a `.nc` from a path, URL, or archive member; returns a Container.          |
| `from_bytes()`  | Opens a NetCDF held in memory as a byte string.                                  |
| `from_array()`  | Builds a Container from a NumPy array plus a geo-reference.                      |
| `from_xarray()` | Builds a `NetCDF` from an `xarray.Dataset` (needs the optional xarray peer dep). |
| `copy()`        | Deep, standalone copy of this dataset, optionally written to `path`.             |
| `close()`       | Releases every GDAL handle this container holds, then closes the base.           |

## Variables, dimensions and groups

| Member                   | What it does                                                                    |
|--------------------------|---------------------------------------------------------------------------------|
| `variable_names`         | Names of the data variables, excluding dimension coordinate arrays.             |
| `variables`              | Lazy `{name: subset}` mapping of every data variable.                           |
| `get_variable()`         | Extracts one variable as a classic-raster `NetCDF` (a Variable).                |
| `get_variable_names()`   | Deprecated alias for the `variable_names` property.                             |
| `nc[name]`               | The variable called `name`; `KeyError` where `get_variable` gives `ValueError`. |
| `name in nc`             | Whether `name` is one of the data variables.                                    |
| `iter(nc)` / `len(nc)`   | The data-variable names, and how many there are.                                |
| `get()`                  | The variable, or a default when the container has no such name.                 |
| `keys()`                 | The data-variable names, as a fresh list.                                       |
| `values()`               | Every variable, loading each.                                                   |
| `items()`                | `(name, variable)` for every variable, loading each.                            |
| `dimension_names`        | Names of all dimensions, in storage order.                                      |
| `dimension_sizes`        | `{name: size}` for every dimension, read from the multidimensional group.       |
| `get_dimension_values()` | Stored coordinates of any dimension — `level`, `depth`, `member`, `time`.       |
| `group_names`            | Names of the sub-groups in the root group.                                      |
| `get_group()`            | Opens a netCDF-4 sub-group as its own Container, without copying data.          |
| `is_subset`              | Whether this object is a single-variable subset rather than a Container.        |
| `is_md_array`            | Whether the dataset was opened in multidimensional mode.                        |
| `file_name`              | The file path, with any `NETCDF:"path":var` prefix stripped.                    |

## xarray-compatible spellings

Aliases so habits from xarray transfer without renaming anything. Each names its canonical member, and each
docstring states where it diverges from the xarray member it echoes.

| Member      | What it does                                                                           |
|-------------|----------------------------------------------------------------------------------------|
| `data_vars` | `variables` under xarray's name — a mapping, so `nc.data_vars["t2m"]` works.           |
| `dims`      | `{name: length}` — **not** `dimension_names`, a list. `sizes` is the durable spelling. |
| `sizes`     | The same mapping as `dims`; xarray is turning its own `dims` into a set of names.      |
| `attrs`     | `global_attributes` under xarray's name.                                               |
| `coords`    | `{name: stored coordinate}` for every indexed dimension, from `get_dimension_values`.  |

## Cheap introspection

Metadata only: no data variable's array is read, though opening a variable does read its coordinate axes.
A variable with no raster plane is the exception — see each member's docstring.

| Member   | What it does                                                                      |
|----------|-----------------------------------------------------------------------------------|
| `dtypes` | `{name: dtype}` for every data variable, from the band description.               |
| `nbytes` | Total size of the data variables, computed from shape and dtype rather than read. |
| `info()` | Prints an `ncdump -h`-shaped summary to a buffer, or to `sys.stdout`.             |

## Coordinates and time

| Member                    | What it does                                                              |
|---------------------------|---------------------------------------------------------------------------|
| `lon` / `x`               | Longitude / x coordinate values as a 1-D array.                           |
| `lat` / `y`               | Latitude / y coordinate values as a 1-D array.                            |
| `top_left_corner`         | Coordinates of the raster's top-left corner.                              |
| `geotransform`            | Geotransform derived from the coordinate arrays, then cached.             |
| `epsg`                    | EPSG code, resolved from the variables when asked of a Container.         |
| `time_stamp`              | Time coordinate values parsed from the CF `time` variable.                |
| `get_time_variable()`     | Decodes the time axis to date strings; `time_format` sets the precision.  |
| `get_time_values()`       | Raw, undecoded values of the time axis — `get_dimension_values()` for it. |
| `create_main_dimension()` | Creates a dimension with its indexing variable (static helper).           |

## Reading and selecting

| Member             | What it does                                                                         |
|--------------------|--------------------------------------------------------------------------------------|
| `read_array()`     | Reads eagerly, or lazily into dask with `chunks=`. Needs `variable=` on a Container. |
| `subset()`         | Reads a windowed `(variable, time, bbox)` slice without materialising the cube.      |
| `sel()`            | Selects bands by coordinate value, date label, or `method="nearest"`.                |
| `isel()`           | Selects bands by position along one or more band dims; works without coordinates.    |
| `open_mfdataset()` | Stacks one variable across many files into a single lazy dask array.                 |
| `reduce()`         | Reduces a container or a variable along `dim` — `how`, `q`, `groupby`, `skipna`.     |
| `coarsen()`        | Reduces fixed-size windows along `dim` — `window`, `boundary`, `how`.                |
| `rolling()`        | Reduces a moving window along `dim`, keeping its length — `center`, `min_periods`.   |
| `diff()`           | Differences neighbouring steps along `dim` — `n`, `label`.                           |
| `cumsum()`         | Totals the values along `dim`, step by step — `skipna`.                              |
| `shift()`          | Moves the values along `dim`, filling the vacated steps — `periods`, `fill_value`.   |
| `ffill()`          | Carries the last valid value along `dim` into the gaps after it — `limit`.           |
| `bfill()`          | Carries the next valid value along `dim` back into the gaps before it — `limit`.     |
| `dropna()`         | Removes the steps of `dim` whose cells are missing — `how`, `thresh`.                |
| `interpolate_na()` | Fills the interior gaps along `dim` from both sides — `method`, `limit`.             |
| `to_dataframe()`   | The cube as a pandas frame, indexed by its dimensions — `variables`, `dropna`.       |
| `concat()`         | Joins cubes end to end along `dim` (classmethod).                                    |
| `merge()`          | Puts several cubes' variables on one grid (classmethod) — `compat`.                  |
| `argmin()`         | The position along `dim` of the smallest value; `-1` where there is none.            |
| `argmax()`         | The position along `dim` of the largest value; `-1` where there is none.             |
| `idxmin()`         | The coordinate along `dim` of the smallest value; NaN where there is none.           |
| `idxmax()`         | The coordinate along `dim` of the largest value; NaN where there is none.            |
| `weighted()`       | Weighted statistics over the spatial axes or a band dim — `"area"`, `how`.           |

## Spatial operations

| Member          | What it does                                                                        |
|-----------------|-------------------------------------------------------------------------------------|
| `crop()`        | Crops by polygon mask, raster mask, or bbox tuple. `chunks=` is Variable only.      |
| `to_crs()`      | Reprojects the dataset to another CRS.                                              |
| `resample()`    | Resamples to a different cell size.                                                 |
| `warped_view()` | A lazy, reprojected view of a Variable — no data is read until used. Variable only. |

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

| Member                      | What it does                                                    |
|-----------------------------|-----------------------------------------------------------------|
| `meta_data`                 | Structured metadata for this NetCDF (cached).                   |
| `get_all_metadata()`        | The same, re-traversed from GDAL rather than served from cache. |
| `global_attributes`         | Global attributes on the root group.                            |
| `set_global_attribute()`    | Sets one global attribute on the root group.                    |
| `delete_global_attribute()` | Deletes one global attribute from the root group.               |
| `scale`                     | Per-band CF `scale_factor` a read applies, per band.            |
| `offset`                    | Per-band CF `add_offset` a read applies, per band.              |

## Writing and interop

| Member                    | What it does                                                                |
|---------------------------|-----------------------------------------------------------------------------|
| `to_file()`               | Saves the dataset to disk.                                                  |
| `to_xarray()`             | Converts the container to an `xarray.Dataset` — `chunks=`, `decode_times=`. |
| `to_kerchunk()`           | Emits a kerchunk JSON reference manifest for this file.                     |
| `combine_kerchunk()`      | Combines per-file manifests into one cube index.                            |
| `to_cog()`                | Writes a Cloud-Optimized GeoTIFF. Variable only.                            |
| `to_feature_collection()` | Converts the raster to a vector `FeatureCollection`. Variable only.         |
| `write_array()`           | Writes an array into the raster bands. Variable only.                       |

## Analysis and plotting

| Member          | What it does                                                                   |
|-----------------|--------------------------------------------------------------------------------|
| `combine()`     | Combines two rasters cell by cell, keeping the band dimensions. Variable only. |
| `plot()`        | Plots a 2-D slice — `selectors`, `facet`, `axes`, `animate`, `chunks`, colour. |
| `stats()`       | Per-band summary statistics. Variable only.                                    |
| `slope()`       | Slope raster from an elevation variable. Variable only.                        |
| `hillshade()`   | Hillshade raster from an elevation variable. Variable only.                    |
| `zonal_stats()` | Statistics per zone of a mask or feature set. Variable only.                   |
| `sample()`      | Samples values at points. Variable only.                                       |

---

**"Variable only"** marks the members that are container-guarded: they are `Dataset` operations that need a
single raster, so calling them on a Container raises with a message pointing at `get_variable`. Two members
are restricted only in part, noted in their own rows: `read_array()` needs `variable=` on a Container, and
`crop()` accepts `chunks=` only on a Variable.
