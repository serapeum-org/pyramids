# Plotting

> For a worked, example-driven walkthrough see the
> [**Plotting NetCDF data**](../../tutorials/netcdf-plotting.md) tutorial. This page is the API
> reference (signature table + the auto-generated `Selectors` / `ColourOpts` / `FacetSpec` docs).

`NetCDF.plot` has its own, xarray-aligned plotting surface — it does **not** inherit the
GeoTIFF / Sentinel-imagery semantics of `Dataset.plot`. You pick a *variable*, slice along the
non-spatial dimensions, and pass colour / faceting / coordinate options through small grouped,
frozen dataclasses (`Selectors`, `ColourOpts`, `FacetSpec`) re-exported from both `pyramids` and
`pyramids.netcdf`.

```python
from pyramids import Selectors, ColourOpts, FacetSpec
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5.nc")

# pick a variable, select along non-spatial dims, xarray-style colour kwargs
nc.plot("t2m", selectors=Selectors(time="2020-01-01", level=850),
        colour=ColourOpts(cmap="coolwarm", robust=True))

# curvilinear (WRF) grid -> pcolormesh, faceted over time
nc.plot("T2", coords=(XLONG, XLAT), kind="pcolormesh",
        facet=FacetSpec(col="time", col_wrap=4))

# animate over a dimension with lazy, per-frame reads ([lazy] extra)
nc.plot("t2m", animate="time", chunks={"time": 1})
```

## Signature

`NetCDF.plot(variable=None, *, selectors=None, colour=None, facet=None, coords=None, kind="auto",
animate=None, chunks=None, basemap=None, exclude_value=None, title=None, ax=None, figsize=None,
**kwargs)`

| Parameter   | Type                                | Notes |
|-------------|-------------------------------------|-------|
| `variable`  | `str`, optional                     | Variable to plot; defaults to the dataset's single / active variable. |
| `selectors` | `Selectors`, optional               | Slice along non-spatial dimensions — `time=`, `level=`, `member=`, plus generic `sel=` / `isel=`. |
| `colour`    | `ColourOpts`, optional              | Colour mapping — `cmap`, `vmin`, `vmax`, `robust`, `levels`, `norm`, `center`, `extend`, `add_colorbar`, `cbar_kwargs`. |
| `facet`     | `FacetSpec`, optional               | Small-multiples grid — `col=`, `row=`, `col_wrap=`. |
| `coords`    | `tuple[ndarray, ndarray]`, optional | Curvilinear `(x_2d, y_2d)` coordinates -> `pcolormesh`. Auto-detected from CF `coordinates` / WRF / ROMS / NEMO conventions when omitted. |
| `kind`      | `str`, optional                     | `"auto"`, `"imshow"`, `"pcolormesh"`, `"contour"`, `"contourf"`. |
| `animate`   | `bool` or `str`, optional           | Animate over a dimension (its name, or `True` for the leading non-spatial dimension). |
| `chunks`    | `dict`, optional                    | Dask chunking — switches to a lazy read; only the rendered slice / frame is materialised. Requires the `[lazy]` extra. |
| `basemap`   | `bool` or `str`, optional           | Overlay a web-tile basemap (provider name as a string, e.g. `"CartoDB.Positron"`). Requires the `[viz]` extra. |
| `**kwargs`  |                                     | Forwarded to cleopatra's `ArrayGlyph` for figure / colour-bar styling, `color_scale`, etc. |

The GeoTIFF-only kwargs `band`, `rgb`, `surface_reflectance`, `cutoff`, `percentile`, `overview`,
and `overview_index` are **not** accepted on `NetCDF.plot` — passing any of them raises `TypeError`
with a hint pointing at the replacement above (use `selectors=` to pick a slice, `colour=` for the
colour scale, and so on).

Internally `NetCDF.plot` is a thin facade over `pyramids.netcdf._plot.NetCDFPlot`, which shares the
`pyramids.dataset._plot_helpers.render_array` rendering core with `Dataset.plot` and
`DatasetCollection.plot` (and `mesh_render` with `UgridDataset.plot`). The full rendered method
signature and docstring are on the [NetCDF Class](index.md) reference page.

## Option dataclasses

::: pyramids.netcdf.Selectors
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.netcdf.ColourOpts
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

::: pyramids.netcdf.FacetSpec
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
