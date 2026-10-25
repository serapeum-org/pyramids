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

On a non-matplotlib backend the facade forwards only the **backend-agnostic** arguments — the data selector
(`band` / `variable` / `column` / `variable_name`), `basemap`, a boolean `colorbar`, `title`, a non-default `kind`,
and any extra keywords you pass — and lets the backend accept or refuse each by name.

The matplotlib-only styling is **rejected** with a `ValueError` so a silently-ignored style is never mistaken for an
applied one. These are the typed cleopatra render groups and the raster-index overlays:

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

`fig` / `ax` apply to matplotlib only, and `FeatureCollection.plot`'s `engine=` (`"geopandas"` / `"cleopatra"`)
applies to `backend="matplotlib"` only; both are ignored off matplotlib.

## Resolution order

A backend name is resolved to a render callable by, in order:

1. a registered `pyramids.plot_backends` entry point named `backend`;
2. otherwise, if [digitalearth](https://github.com/serapeum-org/Digital-Earth) is installed, its `quickmap`
   (imported only when needed — never on a bare install);
3. otherwise a branded error naming the backend and the extra to install, e.g.
   `pip install 'digitalearth[3d]'`.

So digitalearth serves every backend the moment it is installed, with no entry point required; a third party can
also register a backend of its own, and a registered entry point wins over the digitalearth fallback.

## Registering a backend

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.plot_backends"]
globe = "your_package.backends:render_globe"
```

The entry point loads to a **callable** with this shape:

```python
from typing import Any

def render_globe(obj: Any, *, backend: str, **kwargs: Any) -> Any:
    """Render a pyramids object and return the backend's own object.

    `obj` is the Dataset / NetCDF / FeatureCollection / DatasetCollection / UgridDataset the user called
    `plot` on; `backend` is the selected name (so one callable can serve several names); `kwargs` are the
    backend-agnostic forwarded arguments plus the user's extras. Animation paths pass `animate=True`.
    """
    ...
```

Keep the module import light — defer the heavy rendering backend (PyVista, Bokeh, …) to inside the callable, so
discovery never imports it. A name registered twice keeps the first and warns; an entry point that fails to load,
or loads a non-callable, is skipped with a warning rather than breaking `plot`.
