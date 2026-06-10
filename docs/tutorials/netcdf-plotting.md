# Plotting NetCDF data

`NetCDF.plot` is a dedicated, **xarray-aligned** plotting surface — it does *not* inherit
`Dataset.plot`'s GeoTIFF / Sentinel-imagery semantics. Instead of `band=<int>` you pick a
*variable*; instead of loose keyword soup you pass small, frozen option dataclasses; and you get
first-class support for selecting along non-spatial dimensions, curvilinear grids, contour/pcolormesh
artists, small-multiples (facets), animation, and lazy (dask-backed) reads.

```python
from pyramids.netcdf import Selectors, ColourOpts, FacetSpec
from pyramids.netcdf import NetCDF

nc = NetCDF.read_file("era5.nc")
nc.plot("t2m")                       # the simplest call
```

> Plotting requires the `[viz]` extra: `pip install 'pyramids-gis[viz]'`
> (which pins `cleopatra[tiles]`). Animation/lazy reads additionally need the `[lazy]` extra.

`NetCDF.plot` returns a [cleopatra `ArrayGlyph`](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/)
(or a `FacetGrid` when faceting) — use `glyph.fig` / `glyph.ax` to drop down to raw matplotlib.

The full signature:

```python
NetCDF.plot(
    variable=None, *,
    selectors=None,   # Selectors(time=, level=, member=, sel=, isel=)
    colour=None,      # ColourOpts(cmap=, vmin=, vmax=, robust=, levels=, norm=, center=, extend=, add_colorbar=, cbar_kwargs=)
    facet=None,       # FacetSpec(col=, row=, col_wrap=)
    coords=None,      # (x_2d, y_2d) curvilinear coordinates -> pcolormesh
    kind="auto",      # "auto" | "imshow" | "pcolormesh" | "contour" | "contourf"
    animate=None,     # True | dimension name
    chunks=None,      # dask chunks -> lazy read
    basemap=None,     # True | provider name
    exclude_value=None,
    title=None, ax=None, figsize=None,
    **kwargs,         # forwarded to cleopatra's ArrayGlyph (color_scale, cbar_label, ...)
)
```

See the [Plotting reference](../reference/netcdf/plot.md) for the parameter table and the
auto-generated `Selectors` / `ColourOpts` / `FacetSpec` docs, and the
[NetCDF Class reference](../reference/netcdf/index.md) for the rendered method docstring.

## 1. Pick a variable

If the file has a single data variable (or one has been selected), `nc.plot()` works with no
arguments. Otherwise name it:

```python
nc.plot("t2m")
nc.plot("tp")            # total precipitation
```

## 2. Select along non-spatial dimensions

A NetCDF variable is often 3-D (`time, lat, lon`) or 4-D (`time, level, lat, lon`). Use
`Selectors` to reduce it to a 2-D slice — `time=` / `level=` / `member=` are convenience aliases for
the common CF dimensions, and `sel=` / `isel=` are the generic label- and integer-based escape
hatches (mirroring `xarray.Dataset.sel` / `.isel`):

```python
from pyramids.netcdf import Selectors

# label-based: a timestamp and a pressure level
nc.plot("t", selectors=Selectors(time="2020-07-01", level=850))

# integer-based: the 0th time step
nc.plot("t2m", selectors=Selectors(isel={"time": 0}))

# an ensemble member plus a generic label selection
nc.plot("t2m", selectors=Selectors(member=3, sel={"forecast_period": "24h"}))
```

If you leave a non-spatial dimension unpinned, the leading slice is rendered (and, for `animate=`,
it becomes the frame axis — see below).

## 3. Colour mapping

Group all colour options under `ColourOpts` — these mirror xarray's plotting kwargs:

```python
from pyramids.netcdf import ColourOpts

# robust limits (2nd / 98th percentile), a diverging cmap centred on zero
nc.plot("t2m_anomaly", colour=ColourOpts(cmap="RdBu_r", robust=True, center=0.0))

# explicit limits + discrete levels + extend arrows on the colorbar
nc.plot("tp", colour=ColourOpts(vmin=0, vmax=50, levels=11, extend="max"))

# pass a custom matplotlib Normalize
import matplotlib.colors as mcolors
nc.plot("chlor_a", colour=ColourOpts(norm=mcolors.LogNorm(vmin=0.01, vmax=10)))

# suppress the colorbar entirely
nc.plot("t2m", colour=ColourOpts(add_colorbar=False))
```

For the legacy `color_scale` / `gamma` / `line_threshold` / `line_scale` / `bounds` / `midpoint`
knobs, pass them through `**kwargs` — they reach cleopatra's `ArrayGlyph` directly. `color_scale`
takes a [`cleopatra.styles.ColorScale`](https://serapeum-org.github.io/cleopatra/latest/) string
alias (`"linear"`, `"power"`, `"sym-lognorm"`, `"boundary-norm"`, `"midpoint"`; case-insensitive):

```python
nc.plot("classes", color_scale="boundary-norm", bounds=[1, 2, 3, 4, 5])
```

## 4. Render kind

By default pyramids picks `imshow` for regular (1-D coordinate) grids. Override with `kind=`:

```python
nc.plot("sst", kind="pcolormesh")     # honours irregular spacing
nc.plot("z500", kind="contour")       # line contours
nc.plot("z500", kind="contourf")      # filled contours
```

Valid kinds: `"auto"`, `"imshow"`, `"pcolormesh"`, `"contour"`, `"contourf"`.

## 5. Curvilinear grids

WRF, ROMS, NEMO and other models store 2-D latitude/longitude arrays rather than 1-D axes. Pass them
explicitly via `coords=(x_2d, y_2d)` (which forces `pcolormesh`):

```python
nc.plot("T2", coords=(XLONG, XLAT))
```

If you don't pass `coords`, pyramids tries to auto-detect them from the variable's CF `coordinates`
attribute and from the common model conventions (`XLAT`/`XLONG` for WRF, `lat_rho`/`lon_rho` for
ROMS, `nav_lat`/`nav_lon` for NEMO). When auto-detection falls back to the bounding-box extent
instead, a `logging.WARNING` is emitted so you know the rendering used the regular-grid path.

## 6. Faceting (small multiples)

Render one panel per coordinate value along a dimension with `FacetSpec` — the panels share a single
colorbar and colour scale:

```python
from pyramids.netcdf import FacetSpec

# one column per month, wrap after 4
nc.plot("t2m", facet=FacetSpec(col="time", col_wrap=4))

# a 2-D grid: time across, level down
nc.plot("t", facet=FacetSpec(col="time", row="level"))
```

Faceting returns a `cleopatra.array_glyph.FacetGrid`; iterate `grid.axes` for per-panel matplotlib
access. A `basemap=` overlay is applied to every visible panel.

## 7. Animation

Animate over a dimension with `animate=` — pass the dimension name, or `True` to animate over the
leading non-spatial dimension:

```python
anim = nc.plot("t2m", animate="time")     # an ArrayGlyph driving a matplotlib FuncAnimation
anim = nc.plot("t2m", animate=True)        # same, when `time` is the leading dim

# combine with selectors to fix the *other* dims first
anim = nc.plot("t", selectors=Selectors(level=500), animate="time")
```

Pair it with `chunks=` (see below) and pyramids streams one frame at a time from disk via a
`data_getter` callback — the whole cube is never held in memory.

## 8. Lazy / dask reads

For multi-GB reanalysis files, pass `chunks=` to switch the read to a dask-backed lazy path; only
the slice (or animation frame) actually rendered is materialised:

```python
nc.plot("t2m", chunks={"time": 1})                          # lazy static plot of the leading slice
nc.plot("t2m", animate="time", chunks={"time": 1})          # lazy per-frame streaming
```

Requires the `[lazy]` extra (`pip install 'pyramids-gis[lazy]'`). See
[Lazy NetCDF](lazy/lazy-netcdf.md) for chunk-size guidance.

## 9. Basemap underlay

Overlay a web-tile basemap with `basemap=True` (OpenStreetMap) or `basemap="<provider>"`:

```python
nc.plot("sst", crs=4326, basemap="CartoDB.Positron")
```

This routes through cleopatra's tile helpers (`cleopatra.tiles.add_tiles`), so it needs the `[viz]`
extra. When faceting, one tile layer is fetched per panel.

## 10. Escape hatch — raw matplotlib

`NetCDF.plot` returns the cleopatra glyph; reach the underlying figure/axes for any customisation
cleopatra doesn't expose:

```python
glyph = nc.plot("t2m", title="2 m temperature")
glyph.ax.set_xlabel("longitude")
glyph.fig.savefig("t2m.png", dpi=200)
```

## What `NetCDF.plot` does *not* accept

The GeoTIFF / Sentinel kwargs from `Dataset.plot` are rejected on `NetCDF.plot` with a `TypeError`
that points at the replacement:

| GeoTIFF kwarg (`Dataset.plot`) | NetCDF equivalent (`NetCDF.plot`) |
|--------------------------------|-----------------------------------|
| `band=<int>`                   | `variable=` + `selectors=`        |
| `rgb=`, `surface_reflectance=`, `cutoff=` | (no NetCDF equivalent — true-colour composites are a raster concept) |
| `percentile=`                  | `colour=ColourOpts(robust=True)`  |
| `overview=`, `overview_index=` | (NetCDF has no GDAL overview pyramids) |

## Under the hood

`NetCDF.plot` is a thin facade over `pyramids.netcdf._plot.NetCDFPlot`, which resolves the variable,
selectors, curvilinear coordinates, facet stack, and animation axis, then hands a 2-D (or stacked)
array to the shared `pyramids.dataset._plot_helpers.render_array` core — the same renderer used by
`Dataset.plot`, `DatasetCollection.plot` (animate mode), and (via `mesh_render`) `UgridDataset.plot`.
That core builds a cleopatra `ArrayGlyph` and dispatches to `plot` / `facet` / `animate`. See the
[architecture diagrams](../overview/diagrams.md#uml-class-plotting-layer).
