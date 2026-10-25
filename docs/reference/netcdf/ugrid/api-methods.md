# UgridDataset — Complete Method Reference

`UgridDataset` exposes 143 public methods and 19 properties. Many are
inherited from the `_UgridFacades` mixin and do not appear directly on
the class definition, making them hard to discover via code search or
LLM introspection. This page lists **every** public method and property
grouped by category.

> **Source files**: methods marked **(F)** live on `_UgridFacades`
> (`_facades.py`); those marked **(D)** live directly on `UgridDataset`
> (`dataset.py`).

---

## Properties

| Property | Source | Description |
|----------|--------|-------------|
| `mesh` | **(D)** | The `Mesh2d` topology object |
| `mesh_name` | **(D)** | Name of the mesh topology variable |
| `data_variable_names` | **(D)** | Names of all data variables |
| `crs` | **(D)** | CRS as `pyproj.CRS`, or `None` |
| `epsg` | **(D)** | EPSG code of the CRS, or `None` |
| `crs_wkt` | **(D)** | CRS as WKT string, or `None` |
| `bounds` | **(D)** | Bounding box `(xmin, ymin, xmax, ymax)` |
| `global_attributes` | **(D)** | File-level NetCDF attributes dict |
| `n_node` | **(D)** | Number of mesh nodes |
| `n_face` | **(D)** | Number of mesh faces |
| `n_edge` | **(D)** | Number of mesh edges |
| `data_vars` | **(D)** | Copy of the `{name: MeshVariable}` mapping |
| `metadata` | **(D)** | Full metadata summary (`UgridMetadata`) |
| `time_values` | **(D)** | Parsed time coordinate values, or `None` |
| `volume_topology` | **(D)** | Ragged vertical layering, or `None` |
| `n_volume` | **(D)** | Total number of 3-D volumes, or `None` |
| `n_layers` | **(D)** | Number of vertical layers, or `None` |
| `n_layer_interfaces` | **(D)** | Number of layer interfaces, or `None` |
| `layer_values` | **(D)** | Layer coordinate values, or `None` |

---

## I/O & Constructors

| Method | Source | Description |
|--------|--------|-------------|
| `open_mfdataset(paths, driver, mesh_name, sort)` | **(D)** | Open many files on one topology, stacked along time |
| `to_dask_dataframe(variables, dropna, npartitions)` | **(D)** | The lazy sibling of `to_dataframe` |
| `read_file(path, mesh_name, driver, lazy)` | **(D)** cls | Open a UGRID/2DM/Selafin/DFSU/XMDF file |
| `from_arrays(node_x, node_y, fnc, ...)` | **(D)** cls | Create from numpy arrays |
| `from_structured(dataset, band_names, ...)` | **(D)** cls | Build a mesh from a raster Dataset |
| `from_geodataframe(gdf, mesh_name)` | **(D)** cls | Build from polygon GeoDataFrame |
| `from_xarray(ds)` | **(D)** cls | Read from xarray.Dataset in UGRID-CF |
| `from_zarr(store, **kwargs)` | **(D)** cls | Read from a zarr store |
| `from_bytes(data, suffix)` | **(D)** cls | Read from in-memory buffer |
| `read_all(path, driver)` | **(D)** cls | Read every topology in a file |
| `mesh_names(path, driver)` | **(D)** static | List topology names in a file |
| `to_file(path, driver, **options)` | **(D)** | Write to file |
| `to_bytes()` | **(D)** | Serialize to in-memory NetCDF bytes |
| `to_dataset(variable_name, cell_size, ...)` | **(D)** | Rasterize to regular-grid `Dataset` |
| `to_geodataframe(variable_name, location)` | **(D)** | Convert to GeoDataFrame |
| `to_feature_collection(variable_name, loc)` | **(D)** | Convert to pyramids FeatureCollection |
| `to_dataframe(...)` | **(D)** | Convert to pandas DataFrame |
| `to_xarray()` | **(F)** | Export as xarray.Dataset (UGRID-CF) |
| `to_zarr(store, **kwargs)` | **(F)** | Write to zarr store via xarray |
| `attach_data(path, driver, lazy)` | **(D)** | Attach variables from a results file |
| `copy()` | **(D)** | Deep copy the dataset |
| `info(buf, drivers)` | **(D)** | Print human-readable summary |

---

## CRS Management

| Method | Source | Description |
|--------|--------|-------------|
| `set_crs(crs, epsg)` | **(D)** | Assign a CRS without reprojecting |
| `to_crs(to_epsg)` | **(D)** | Reproject all node coordinates |

---

## Variable Management

| Method | Source | Description |
|--------|--------|-------------|
| `get_data(variable_name)` | **(D)** | Get a variable by name |
| `with_variable(name, data, location, nodata)` | **(D)** | Return new ds with variable added/replaced |
| `drop_variables(names)` | **(D)** | Return new ds without named variable(s) |
| `rename_variable(old, new)` | **(D)** | Return new ds with variable renamed |
| `set_global_attribute(name, value)` | **(D)** | Return new ds with attribute added |
| `delete_global_attribute(name)` | **(D)** | Return new ds without an attribute |
| `active_mask(variable_name)` | **(D)** | Per-element active mask, or `None` |
| `masked_values(variable_name)` | **(D)** | Values with nodata + active mask → NaN |

---

## Mapping Interface

`UgridDataset` implements the read-only `Mapping` protocol:

| Method | Source | Description |
|--------|--------|-------------|
| `keys()` | **(D)** | Variable names |
| `values()` | **(D)** | Variable objects |
| `items()` | **(D)** | `(name, variable)` pairs |
| `get(key, default)` | **(D)** | Variable by key, or default |
| `ds["depth"]` | **(D)** | `__getitem__` |
| `"depth" in ds` | **(D)** | `__contains__` |
| `len(ds)` | **(D)** | `__len__` — number of variables |
| `for name in ds` | **(D)** | `__iter__` — iterate variable names |

---

## Array / Value Operations

| Method | Source | Description |
|--------|--------|-------------|
| `fillna(value, variable_name)` | **(F)** | Replace missing values |
| `isnull(variable_name)` | **(F)** | Boolean mask — `True` where missing |
| `notnull(variable_name)` | **(F)** | Boolean mask — `True` where valid |
| `clip_values(lower, upper, variable_name)` | **(F)** | Clamp values to `[lower, upper]` |
| `round(decimals, variable_name)` | **(F)** | Round to N decimal places |
| `astype(dtype, variable_name)` | **(F)** | Cast variable to a different dtype |
| `isin(test_values, variable_name)` | **(F)** | Boolean mask for set membership |
| `where(cond, other, variable_name)` | **(F)** | Keep the elements a condition selects, mask the rest |
| `apply(func, variable_name)` | **(F)** | Apply a function elementwise, keeping the shape |
| `convert_units(target, variable_name)` | **(D)** | Convert values to target units (shared affine table) |

---

## Dataset Comparison

| Method | Source | Description |
|--------|--------|-------------|
| `equals(other)` | **(D)** | Value-equal (same non-missing values) |
| `identical(other)` | **(D)** | Strict (dtype, nodata, CRS, topology) |

---

## Domain Metrics

| Method | Source | Description |
|--------|--------|-------------|
| `footprint(variable_name)` | **(F)** | Area actually covered by valid data (not the mesh outline) |
| `domain_area()` | **(F)** | Total area of the mesh domain |
| `count_domain_cells(variable_name)` | **(F)** | Number of active (non-masked) faces |

---

## Spatial Operations

| Method | Source | Description |
|--------|--------|-------------|
| `slope(units, variable_name)` | **(F)** | Per-face slope of a mesh DEM, degrees or radians |
| `aspect(variable_name)` | **(F)** | Per-face aspect, degrees clockwise from north |
| `hillshade(azimuth, altitude, variable_name)` | **(F)** | Shaded relief in 0–255 |
| `crop(mask, touch, bbox, variable_name)` | **(D)** | Crop to polygon or bbox |
| `clip(mask, touch)` | **(D)** | Clip to polygon mask |
| `subset_by_bounds(bbox, ...)` | **(D)** | Subset by bounding box |
| `extract(x, y, variable_name)` | **(F)** | Sample values at (x, y) via STRtree |
| `sample_random(n, variable_name, seed)` | **(F)** | Draw N random spatial samples |
| `sample(variable_name, x, y, method, ...)` | **(D)** | Interpolated point sampling |
| `sample_line(line, variable_name)` | **(F)** | Sample along a line geometry |
| `burn(geometry, value, name, fill, ...)` | **(F)** | Burn vector geometry onto faces |

---

## Graph & Morphology

| Method | Source | Description |
|--------|--------|-------------|
| `cluster(lower, upper, variable_name)` | **(F)** | Label connected faces whose values fall in a range |
| `connected_components()` | **(F)** | Label connected face components |
| `exterior_edges()` | **(F)** | Boundary edge indices |
| `exterior_faces()` | **(F)** | Face indices touching the boundary |
| `bounding_polygon()` | **(F)** | Mesh footprint as dissolved polygon |
| `circumcenters()` | **(F)** | Circumcentre of each face (triangles) |
| `voronoi()` | **(F)** | Voronoi tessellation (mesh dual) |
| `sparse_adjacency(kind)` | **(F)** | Adjacency as a scipy sparse matrix |
| `binary_dilation(variable_name, ...)` | **(F)** | Grow a boolean face mask |
| `binary_erosion(variable_name, ...)` | **(F)** | Shrink a boolean face mask |
| `proximity(target_faces)` | **(F)** | Graph distance from faces to targets |
| `sieve(threshold, variable_name)` | **(F)** | Remove small regions below threshold |

---

## Interpolation & Regridding

| Method | Source | Description |
|--------|--------|-------------|
| `to_location(location, variable_name)` | **(F)** | Remap variable to another mesh facet |
| `fill_na_spatial(variable_name, max_dist)` | **(F)** | Fill gaps from nearest valid element |
| `laplace_interpolate(variable_name, ...)` | **(F)** | Fill gaps via Laplace's equation |
| `rasterize_like(reference, variable_name, ...)` | **(F)** | Interpolate onto an existing raster grid |
| `interp_time(new_times, method, bounds)` | **(F)** | Interpolate temporal variables |
| `regrid(target, method, variable_name)` | **(F)** | Regrid onto another mesh |

---

## Vectorization

| Method | Source | Description |
|--------|--------|-------------|
| `polygonize(variable_name, decimals)` | **(F)** | Dissolve faces of equal value → polygons |
| `contour(levels, variable_name)` | **(F)** | Iso-lines as GeoDataFrame of LineStrings |

---

## Plotting

| Method | Source | Description |
|--------|--------|-------------|
| `plot(variable_name, ax, cmap, title, ...)` | **(D)** | Plot mesh data via cleopatra |
| `plot_outline(ax, **kwargs)` | **(D)** | Plot mesh wireframe |
| `plot_histogram(bins, variable_name, ...)` | **(F)** | Histogram of a variable's values |
| `plot_vector_field(u, v, kind, thin, ...)` | **(D)** | Quiver / barbs field from two components |
| `animate(variable_name, interval, **kwargs)` | **(D)** | Animate a temporal variable over its time axis |
| `to_image(variable_name, cell_size, cmap, method)` | **(D)** | Rasterise and colour-map to a PIL image |

---

## Statistics & Analysis

| Method | Source | Description |
|--------|--------|-------------|
| `get_histogram(bins, value_range, variable_name)` | **(F)** | Histogram counts and bin edges |
| `focal_mean(rings, variable_name)` | **(F)** | Mean over each face's k-ring neighbourhood |
| `focal_std(rings, variable_name)` | **(F)** | Standard deviation over the same neighbourhood |
| `focal_apply(func, rings, variable_name)` | **(F)** | Arbitrary aggregation over the same neighbourhood |
| `summary(variables, metrics, skipna, ddof)` | **(D)** | Per-variable summary table |
| `weighted(variable_name, how, time_index, q)` | **(D)** | Area-weighted statistic |
| `zonal_stats(zones, variable_name, stats, ...)` | **(D)** | Aggregate within polygon zones |

---

## Temporal Operations

!!! warning "Behaviour change: `time_values` now carries decoded dates"

    For a file whose time variable declares CF `units` (`"<interval> since <origin>"`),
    `time_values` and `to_dataframe()["time"]` return **decoded stamps** — `numpy.datetime64`
    for the Gregorian calendar family, `cftime` objects for a model calendar such as
    `360_day`. They previously returned positional integers for those files.

    ```python
    # before
    ds.time_values            # [0, 1]
    # after
    ds.time_values            # [datetime64('2001-05-05T00:00:05'), ...]
    ```

    Code that treated the values as integers (`ds.time_values[0] + 1`, or comparing against
    `0`) needs updating. A file carrying **no** CF time units is unaffected and still reports
    positional indices, so the fallback behaviour is unchanged.

    This lands as a behaviour change within the `0.x` line rather than as a major release:
    returning positional indices for a file that states its stamps was the defect
    [#1381](https://github.com/serapeum-org/pyramids-core/issues/1381) describes, so the decoded
    values are the fix. The note is here because the element type nevertheless changes for
    files that already carried CF units.


| Method | Source | Description |
|--------|--------|-------------|
| `polyfit(deg, variable_name)` | **(D)** | Per-element polynomial fit along time (trend map) |
| `curvefit(func, p0, variable_name)` | **(D)** | Per-element fit of an arbitrary model along time |
| `differentiate(variable_name)` | **(D)** | Derivative along time, honouring the coordinate spacing |
| `integrate(variable_name)` | **(D)** | Definite integral along time (consumes the axis) |
| `cumulative_integrate(variable_name)` | **(D)** | Running integral along time (keeps the axis) |
| `rank(variable_name, pct)` | **(D)** | Rank each element's values along time, ties averaged |
| `coarsen(window, how, variable_name)` | **(D)** | Block-aggregate time by a count of steps |
| `interp_like(other)` | **(D)** | Interpolate onto another dataset's time stamps |
| `convert_calendar(calendar, align_on)` | **(D)** | Restamp the time axis onto another CF calendar |
| `interp_calendar(target)` | **(D)** | Interpolate onto another dataset's stamps across calendars |
| `sel_time(index)` | **(D)** | Select a single time step |
| `sel_time_range(start, stop)` | **(D)** | Select a time range |
| `reduce(how, skipna, q, groupby)` | **(D)** | Reduce time dimension |
| `resample(freq, how, skipna, q)` | **(D)** | Bin time axis and reduce |
| `mean(skipna)` | **(D)** | Mean over time |
| `sum(skipna)` | **(D)** | Sum over time |
| `min(skipna)` | **(D)** | Minimum over time |
| `max(skipna)` | **(D)** | Maximum over time |
| `std(skipna)` | **(D)** | Std-dev over time |
| `var(skipna)` | **(D)** | Variance over time |
| `cumsum()` | **(D)** | Cumulative sum along time |
| `cumprod()` | **(D)** | Cumulative product along time |
| `shift(periods)` | **(D)** | Shift along time, vacated → NaN |
| `ffill(limit, dim)` | **(D)** | Forward-fill gaps |
| `bfill(limit, dim)` | **(D)** | Back-fill gaps |
| `interpolate_na(method, limit, ...)` | **(D)** | Fill time gaps by interpolation |
| `rolling(window, how, center, ...)` | **(D)** | Rolling-window reduction |
| `diff(n)` | **(D)** | Discrete difference along time |
| `head(n)` | **(D)** | Keep first N time steps |
| `tail(n)` | **(D)** | Keep last N time steps |
| `thin(step)` | **(D)** | Keep every step-th time step |
| `squeeze()` | **(D)** | Drop time dim when length 1 |
| `argmin()` | **(D)** | Time index of each element's min |
| `argmax()` | **(D)** | Time index of each element's max |
| `idxmin()` | **(D)** | Time coordinate of each element's min |
| `idxmax()` | **(D)** | Time coordinate of each element's max |
| `dropna(how, thresh, dim)` | **(D)** | Drop time steps without enough valid cells |
| `drop_duplicates()` | **(D)** | Keep first of each distinct time coord |
| `sortby()` | **(D)** | Sort time steps by coordinate value |

---

## Selection & Indexing

| Method | Source | Description |
|--------|--------|-------------|
| `isel(time, element)` | **(D)** | Select by position |
| `sel(time, method, tolerance)` | **(D)** | Select by coordinate value |
| `drop_isel(indices)` | **(D)** | Drop time steps at positions |
| `drop_sel(values)` | **(D)** | Drop time steps by coordinate value |

---

## Layer Operations (3-D Meshes)

| Method | Source | Description |
|--------|--------|-------------|
| `isel_layer(index)` | **(D)** | Select along layer axis by position |
| `sel_layer(value)` | **(D)** | Select along layer axis by value |
| `collapse_layer(layer, location)` | **(D)** | Collapse 3-D → 2-D field |

---

## Joining

| Method | Source | Description |
|--------|--------|-------------|
| `concat(others)` | **(D)** | Concatenate along time axis |
| `merge(others, compat)` | **(D)** | Merge variables from same-topology ds |
