# Writing a plot backend

pyramids draws through matplotlib (via [cleopatra](basemap.md)). A richer renderer — an interactive
map, a 3-D scene, a web map — is a **plot backend**: a separate distribution that registers a render
callable into the `pyramids.plot_backends` entry-point group. `plot(backend="…")` then routes the
object into that backend with no dependency coupling: pyramids declares no dependency on any
renderer and imports nothing extra on a bare install (it imports a registered backend only when a
call asks for it), so an install with no backends plots through matplotlib exactly as before.

It is the renderer sibling of the [raster driver hook](dataset/writing-a-raster-driver.md), with one
difference of direction: a driver is a **source** (it reads a file into a pyramids object), while a
backend is a **sink** (it takes a pyramids object and returns its own figure). A driver registers a
*class*; a backend registers a plain *callable*. The canonical plot backend is
[digitalearth](https://github.com/serapeum-org/Digital-Earth), which serves the `interactive` /
`3d` / `web` tiers.

## Registering

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.plot_backends"]
globe = "your_package.backends:render_globe"
```

The dependency runs one way: your package imports pyramids and builds on its objects; pyramids knows
only the group name. The entry-point **name** (`globe`) is the value a caller passes as
`plot(backend="globe")`.

## The contract

A backend is a **callable** with this shape:

```python
from typing import Any

def render_globe(obj: Any, *, backend: str, **kwargs: Any) -> Any:
    """Render a pyramids object and return the backend's own object.

    - `obj` is the Dataset / NetCDF / FeatureCollection / DatasetCollection / UgridDataset the user
      called `plot` (or `animate`) on.
    - `backend` is the selected name, so one callable can serve several names (e.g. a single
      function registered under `interactive` and `web` that branches on it).
    - `**kwargs` are the backend-agnostic arguments the facade forwarded (below). An animation call
      forwards `animate=True`.

    Returns the backend's own object (an interactive map, a scene, a figure) — pyramids passes it
    straight back to the caller; it is never coerced to a matplotlib figure.
    """
    ...
```

Keep the module import **light** — defer the heavy drawing stack (PyVista, Bokeh, …) to inside the
callable, so discovery, which imports the entry point, never pulls your renderer into a bare
`import pyramids`.

### What the facade forwards

A non-matplotlib `plot(backend=…)` call forwards only the **data-selection** arguments — the data
selector (`band` / `variable` / `column` / `variable_name`), `basemap`, a bool `colorbar`, `title`,
a non-default `kind`, plus any extra keywords — and drops the ones left at their default. The
matplotlib-only styling (the typed `color` / `contour` / `cells` / `data_style` / `classify` render
groups, the `points` overlay, a `ColorBar` instance, and the read/render directives `overview` /
`chunks` / `facet` / a non-default `engine`) is **rejected with a `ValueError` before dispatch**, so
your callable never receives it. The exact forwarded set per facade is listed in the
[Plot backends reference](plot-backends.md).

The selector values are in the facade's **own index space**: a `band` is a **0-based** raster band
index (band `0` is the first band, not band `1`), a `variable` / `variable_name` is a name, a
`column` is a vector attribute name. Honor them as given.

## How pyramids dispatches

`resolve_plot_backend(name)` resolves a non-matplotlib backend name in this order:

1. a registered `pyramids.plot_backends` entry point named `name` — discovered once and cached;
2. otherwise, if [digitalearth](https://github.com/serapeum-org/Digital-Earth) is installed, its
   `quickmap` (an optional import — pyramids declares no dependency on it, and never imports it on a
   bare install);
3. otherwise a branded `OptionalPackageDoesNotExist` naming the backend and the extra to install.

A **registered entry point always wins** over the digitalearth fallback. The fallback is kept as a
zero-config convenience — it lets any digitalearth version render the moment it is installed, before
it registers an entry point of its own — and costs nothing otherwise: on a bare install digitalearth
is never imported, because presence is probed with `importlib.util.find_spec` first. `"matplotlib"`
is never resolved here; the facades short-circuit the default to cleopatra before calling in.

## Failure handling

Every discovery failure is a **warning, never an exception**: an entry point that raises on load, or
loads a non-callable, is skipped with a `RuntimeWarning`; a name registered twice keeps the first
and warns. One broken third-party backend cannot stop `plot()` — it falls back to the next step or
raises the branded install hint. The default matplotlib path is never affected.

## Checking what is installed

```python
from pyramids.base._plot_backends import plot_backends

plot_backends()   # {name: render callable}, discovered once and cached; {} on a bare install
```

## Conformance kit

Run the reusable kit against your callable in your own test suite — it needs no pytest and checks
the same contract the seam dispatches through (that the object is callable and accepts the
`fn(obj, *, backend, **kwargs)` render call shape):

```python
from pyramids.base.plot_backend_conformance import assert_plot_backend_conforms
from your_package.backends import render_globe

def test_backend_conforms():
    assert_plot_backend_conforms(render_globe)
```

## Compatibility

This hook is additive: a bare pyramids install behaves exactly as it did before it existed — every
`plot()` renders through matplotlib. A backend is reached only when a caller asks for it by name and
its package is installed, so an absent or broken backend never changes a default plot.
