# Plot backends

Every pyramids `plot` and `animate` method takes a `backend=` keyword. The default, `"matplotlib"`, draws through
[cleopatra](basemap.md) and returns a cleopatra glyph — the long-standing behaviour, unchanged. Any other backend
name routes the same call to a renderer registered in the `pyramids.plot_backends` entry-point group, and returns
that backend's own object (an interactive map, a 3-D scene, a web map, …).

pyramids declares **no dependency** on any non-matplotlib renderer. A bare install plots through matplotlib exactly
as before; a richer backend becomes available only when you install the package that provides it. In practice that
package is [digitalearth](https://github.com/serapeum-org/Digital-Earth), which ships interactive (HoloViz/Bokeh),
3-D (PyVista) and web (MapLibre/deck.gl) tiers on top of pyramids.

This is the plotting sibling of the [raster driver hook](dataset/writing-a-raster-driver.md): the same
entry-point discovery, cache and first-wins collision rule, and the same one-way dependency — your package imports
pyramids, pyramids knows only the group name.

## Using a backend

```python
ds.plot(backend="interactive")          # an interactive map object
ds.plot(backend="3d", z_exaggeration=2) # a 3-D scene; extra kwargs pass through to the backend
collection.plot(backend="web")          # the collection animation, as a web map
mesh.plot(backend="interactive")        # a UgridDataset mesh, interactively
```

The `backend=` keyword is available on `Dataset.plot` / `Dataset.animate`, `NetCDF.plot`, `FeatureCollection.plot`,
`DatasetCollection.plot` / `DatasetCollection.animate`, and `UgridDataset.plot` / `UgridDataset.animate`.

### What carries over, and what does not

On a non-matplotlib backend the facade forwards the **data-selection** arguments and lets the backend accept or
refuse each by name. The forwarded set is per facade — the data selector plus the arguments that describe *what to
show*, with `None`-valued and default ones dropped:

- **`Dataset.plot`** — `band`, `exclude_value`, `rgb_options`, `basemap`, a bool `colorbar`, `title`, a
  non-default `kind`.
- **`Dataset.animate`** — `bands`, `animation_axis_values`, `exclude_value`, `basemap`, a non-default `interval`,
  and `animate=True`.
- **`NetCDF.plot`** — `variable`, `selectors`, `axes`, `exclude_value`, `basemap`, a bool `colorbar`, `title`, a
  non-default `kind`, `animate`.
- **`FeatureCollection.plot`** — `column`, `basemap`, a bool `colorbar`, `title`, a non-default `kind`.
- **`DatasetCollection.plot` / `.animate`** — a non-default `band`, `exclude_value`, `rgb_options`, `basemap`, a
  bool `colorbar`, `animation_axis_values`, and `animate=True`.
- **`UgridDataset.plot`** — `variable_name`, a non-default `cmap`, `title`, `basemap`, a bool `colorbar`, a
  non-default `kind`.
- **`UgridDataset.animate`** — `variable_name`, a non-default `interval`, and `animate=True`.

Any further keyword you pass is forwarded too. Arguments that only mean something on the matplotlib read/render
path are instead **rejected** with a `ValueError`, so a silently-ignored argument is never mistaken for an applied
one. These are the typed cleopatra render groups and raster-index overlays:

| Parameter | Type | Why it is matplotlib-only |
|---|---|---|
| `color` | `ColorScaling` | matplotlib norm (power / sym-log / boundary / midpoint) |
| `contour` | `Contour` | matplotlib contour lines and labels |
| `cells` | `CellValues` | per-cell matplotlib text annotation |
| `data_style` | `DataStyle` | cleopatra relief / hillshade styling |
| `classify` | `Classify` | cleopatra vector classification (vectors) |
| `points` | `ndarray` / `PointOverlay` | a raster-index `(value, row, col)` overlay |
| `frame_label` | `FrameLabel` | cleopatra animation frame caption styling |
| `colorbar` | `ColorBar` instance | matplotlib colour-bar styling (a plain `bool` is fine) |

plus the per-facade read/render directives that pyramids cannot apply once it hands the object to a backend:
`Dataset.plot`'s `overview` / `overview_index`, `NetCDF.plot`'s `facet` / `chunks`, and a non-default `engine`
(`"cleopatra"`) on `FeatureCollection.plot`. `fig` / `ax` apply to matplotlib only.

## Resolution order

A backend name is resolved to a render callable by, in order:

1. a registered `pyramids.plot_backends` entry point named `backend`;
2. otherwise, if [digitalearth](https://github.com/serapeum-org/Digital-Earth) is installed, its `quickmap`
   (imported only when needed — never on a bare install);
3. otherwise a branded error naming the backend and the extra to install, e.g.
   `pip install 'digitalearth[3d]'`.

So digitalearth — the **reference plot backend** — serves every backend the moment it is installed, with no entry
point required; a third party can also register a backend of its own, and a registered entry point always wins over
the digitalearth fallback. The fallback is kept deliberately as a zero-config convenience: it costs nothing on a bare
install (digitalearth is never imported — presence is probed with `find_spec` first) and lets any digitalearth
version render before it registers an entry point of its own.

## Registering a backend

A backend is a callable registered in the `pyramids.plot_backends` entry-point group:

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.plot_backends"]
globe = "your_package.backends:render_globe"
```

```python
def render_globe(obj, *, backend, **kwargs):
    ...  # render obj, return the backend's own object
```

See **[Writing a plot backend](writing-a-plot-backend.md)** for the full contract — the render call shape, the
forwarded vs rejected arguments, the dispatch order and failure handling. Validate your callable against that
contract in your own tests with the conformance kit (no pytest required):

```python
from pyramids.base.plot_backend_conformance import assert_plot_backend_conforms
from your_package.backends import render_globe

def test_backend_conforms():
    assert_plot_backend_conforms(render_globe)
```
