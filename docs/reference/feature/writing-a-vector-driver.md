# Writing a vector driver

pyramids reads vector data through geopandas/pyogrio, which hands off to OGR's auto-detected driver.
Some sources OGR cannot open — or opens **wrongly** — are read by a separate distribution that
registers a driver into the `pyramids.vector_drivers` entry-point group. `FeatureCollection.read_file`
then routes those sources into the plugin with no import coupling: pyramids never imports a plugin,
and a bare install reads through OGR exactly as before.

The motivating case is a look-alike extension. A CityJSON `.json` handed to OGR's GeoJSON driver does
not fail — it opens and returns the wrong geometry. So, unlike the
[raster hook](../dataset/writing-a-raster-driver.md) (which tries GDAL first and consults a driver
only when GDAL *fails*), a claiming vector driver **pre-empts OGR**: a registered driver whose
`can_open` claims a source wins **before** the geopandas/pyogrio path is tried. A cheap suffix/scheme
pre-filter runs first, so a plugin's `can_open` is consulted only for a source whose suffix or scheme
the driver declares — a `.cityjson` driver is never asked about a `.shp`.

## Registering

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.vector_drivers"]
cityjson = "your_package.drivers:CityJSONDriver"
```

The dependency runs one way: your package imports pyramids' vector types and builds them; pyramids
knows only the group name.

## The contract

A driver is a **class** whose claims are class-level, so the registry reads them without
constructing it:

```python
import json
from geopandas import GeoDataFrame

class CityJSONDriver:
    name = "cityjson"                        # unique; the value accepted by driver=
    long_name = "CityJSON"
    api_version = 1                          # must equal VECTOR_DRIVER_API_VERSION
    suffixes = (".cityjson", ".json")        # file suffixes, lowercase + dot-prefixed
    schemes = ()                             # URL schemes, lowercase, no "://"
    capabilities = frozenset({"read"})
    extra = None                             # import name of an optional dep, or None

    @classmethod
    def can_open(cls, source) -> bool:       # cheap; suffix first, sniff only on a match; never raises
        text = str(source)
        if not text.lower().endswith((".cityjson", ".json")):
            return False
        try:
            with open(text, "rb") as handle:
                head = handle.read(512)
        except OSError:
            return False
        return b'"type"' in head and b'"CityJSON"' in head

    def read(self, source, *, layer=None, bbox=None, mask=None, rows=None,
             columns=None, where=None, **kwargs) -> GeoDataFrame:
        ...                                  # a heavy backend, if any, is imported HERE
```

Declare **at least one** of `suffixes` / `schemes` — the scan's pre-filter routes a source to your
`can_open` only when its suffix or scheme matches, so a driver with neither would never be consulted.
Content-sniffing (telling CityJSON from GeoJSON for a shared `.json`) happens **inside** `can_open`,
after the suffix match. The example's own `endswith` check is belt-and-braces — dispatch has already
matched the suffix before calling `can_open`, but keeping the check leaves `can_open` correct if it is
ever called directly. Declare the `"read"` capability, and return a `GeoDataFrame` from `read` —
`read_file` wraps it in a `FeatureCollection` and rejects any other type with a `TypeError` naming
your driver.

Keep the driver module importable without its heavy backend: put that import inside `read`, and name
the backend in `extra` so discovery skips the driver when the backend is not installed (its
`can_open` is then never reached).

### Capabilities

- `read` — required; gates `read`, the method `FeatureCollection.read_file` dispatches into.
- `needs_auth` — declarative; tells `describe_drivers` and your docs that the source needs
  credentials.

## How pyramids dispatches

`FeatureCollection.read_file(source, *, driver=None, backend="pandas", …)` resolves in this order:

1. an explicit `driver="cityjson"` routes straight to that registered driver, bypassing OGR
   (an unregistered name raises `DriverNotExistError`);
2. otherwise, if any driver is installed, the first one whose **declared suffix/scheme matches the
   source** and whose `can_open` then claims it is dispatched — **before** OGR;
3. otherwise the built-in geopandas/pyogrio OGR path reads the source, exactly as on a bare install.

A driver-selected read is **eager**: a plugin driver returns a materialised `GeoDataFrame`, so there
is no lazy path for it. `backend="dask"` is rejected for a driver-selected read with a `ValueError`,
and `npartitions` / `chunksize` do not apply. Discovery is lazy and cached — the entry-point scan
runs **at most once per process**, and a bare install (no plugins) registers nothing, so a normal
`.geojson` / `.shp` / `.gpkg` read falls straight through to OGR.

## Failure handling

One broken third-party package cannot stop a vector file from opening through OGR:

- a driver whose `extra` dependency is not importable is skipped **silently** at discovery — its
  module is loaded, but because the backend import lives inside `read` (see above) the backend itself
  is never imported, and its `can_open` is never reached;
- a driver that raises on import, declares a future `api_version`, or collides on a `name` is skipped
  with a **warning**;
- a `can_open` that raises during the scan is treated as "does not claim" and warned about, so the
  read falls through to OGR.

## Checking what is installed

```python
from pyramids.feature._drivers import describe_drivers, vector_drivers

vector_drivers()        # {name: driver class}, discovered once and cached
describe_drivers()      # [{name, long_name, suffixes, schemes, capabilities}, ...]
```

## Compatibility

`api_version` is bumped only on a breaking change to the driver contract. A driver declaring a
different version is skipped with a warning, so an old plugin never breaks a pyramids upgrade and a
new plugin never half-loads on an old pyramids. This hook is additive: a bare pyramids install
behaves exactly as it did before it existed.
