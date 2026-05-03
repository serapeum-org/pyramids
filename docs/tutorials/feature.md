# FeatureCollection tutorial

`FeatureCollection` is a `geopandas.GeoDataFrame` with pyramids
geospatial utilities on top. Anywhere a `GeoDataFrame` works, a
`FeatureCollection` works — and the added methods cover what pyramids
needs on top: rasterization, cloud / archive I/O, CRS-safe
concatenation, streaming reads, and introspection.

This tutorial walks through the everyday operations with runnable
snippets. Every snippet is standalone.

## Install

```bash
pip install pyramids-gis        # core
pip install 'pyramids-gis[parquet]'  # optional GeoParquet
```

## Construct

### From a file

`read_file` handles Shapefile, GeoJSON, GeoPackage, KML, FlatGeobuf —
anything pyogrio / fiona supports — plus zip/tar/gz archives and
cloud URLs (`s3://`, `gs://`, `az://`, `abfs://`, `http(s)://`).

```python
from pyramids.feature import FeatureCollection

fc = FeatureCollection.read_file("roads.shp")
fc = FeatureCollection.read_file("roads.zip")                   # auto /vsizip/
fc = FeatureCollection.read_file("s3://bucket/roads.shp")       # auto /vsis3/
fc = FeatureCollection.read_file("https://example.com/data.tar.gz")
```

Filter on open:

```python
fc = FeatureCollection.read_file(
    "gpkg-file.gpkg",
    layer="roads",
    bbox=(30.0, 30.0, 31.0, 31.0),
    columns=["osm_id", "highway", "lanes"],
    where="highway = 'motorway'",
)
```

### From Python data

```python
from shapely.geometry import Point
from pyramids.feature import FeatureCollection

# records orient: list of dicts
fc = FeatureCollection.from_records(
    [
        {"id": 1, "geometry": Point(0, 0)},
        {"id": 2, "geometry": Point(1, 1)},
    ],
    crs=4326,
)

# list orient: columnar dict of equal-length lists
fc = FeatureCollection.from_records(
    {"id": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
    orient="list",
    crs=4326,
)
```

### From an existing GeoDataFrame

```python
import geopandas as gpd
from pyramids.feature import FeatureCollection

gdf = gpd.read_file("roads.shp")
fc = FeatureCollection(gdf)                # constructor accepts a GDF
fc = FeatureCollection.from_features(gdf)  # explicit classmethod
```

### GeoParquet

```python
fc = FeatureCollection.read_parquet("roads.parquet")
fc.to_parquet("out.parquet", compression="zstd")
```

If `pyarrow` is missing, pyramids raises a branded `ImportError`
that names the `[parquet]` extra (D-M5):

```text
ImportError: GeoParquet support requires the optional 'pyarrow'
dependency. Install with: pip install 'pyramids-gis[parquet]'
(or `pixi add pyarrow` in a pixi workspace).
```

## Inspect before reading

```python
from pyramids.feature import FeatureCollection

# Which layers does this file have?
FeatureCollection.list_layers("multi_layer.gpkg")
# ['roads', 'buildings', 'parks']

# What's the schema (without loading)?
fc = FeatureCollection.read_file("roads.shp", rows=0)  # empty read is fast
fc.schema
# {'geometry': 'Point', 'properties': {'id': 'int64', 'name': 'object'},
#  'crs': <Projected CRS: EPSG:4326>}
```

`list_layers` is LRU-cached (C15): 128 resolved paths, and call
`FeatureCollection.list_layers_cache_clear()` if you need to invalidate
after an out-of-band write.

## Stream very large files

`iter_features` yields chunks without materialising the whole file:

```python
from pyramids.feature import FeatureCollection

total = 0
for chunk in FeatureCollection.iter_features("huge.gpkg", chunksize=5000):
    total += len(chunk)

# Stream as dicts instead of FeatureCollection chunks:
for record in FeatureCollection.iter_features("huge.gpkg", as_dict=True):
    ...
```

Hint the tile strategy when the file supports it (ARC-34):

```python
for chunk in FeatureCollection.iter_features(
    "huge.parquet", tile_strategy="row_group", chunksize=5000,
):
    ...
```

Strategies:

| `tile_strategy` | Meaning |
|-----------------|---------|
| `"auto"` (default) | Pick per format — row-groups for GeoParquet, rtree for Shapefile, none elsewhere |
| `"rtree"` | Force rtree tiling (requires a spatial index) |
| `"row_group"` | Force parquet row-group tiling |
| `"none"` | Sequential read |

## Transform

### Per-vertex coordinates

`with_coordinates()` returns a new FC with `x` and `y` list-columns
(one entry per vertex of each row's geometry). MultiPolygon and
GeometryCollection rows are exploded first.

```python
from shapely.geometry import Point
from pyramids.feature import FeatureCollection

fc = FeatureCollection.from_records(
    [
        {"id": 1, "geometry": Point(1.0, 2.0)},
        {"id": 2, "geometry": Point(3.0, 4.0)},
    ],
    crs=4326,
)
out = fc.with_coordinates()
print(list(out["x"]))  # [1.0, 3.0]
print(list(out["y"]))  # [2.0, 4.0]
```

### Centroids

`with_centroid()` attaches `avg_x`, `avg_y`, and `center_point`
columns.

```python
from shapely.geometry import Polygon
from pyramids.feature import FeatureCollection
import geopandas as gpd

fc = FeatureCollection(
    gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]),
        ],
        crs="EPSG:4326",
    )
)
out = fc.with_centroid()
for p in out["center_point"]:
    print(p.x, p.y)
```

Empty or all-NaN geometries emit a `GeometryWarning` (a subclass of
`UserWarning`) that callers can filter selectively:

```python
import warnings
from pyramids.base._errors import GeometryWarning

warnings.filterwarnings("ignore", category=GeometryWarning)
```

### Reproject a coordinate array

```python
from pyramids.feature import FeatureCollection

x, y = FeatureCollection.reproject_coordinates(
    [31.0, 31.1], [30.0, 30.1],
    from_crs=4326, to_crs=3857,
)
```

`from_crs` / `to_crs` accept anything `pyproj.Transformer.from_crs`
accepts: EPSG int, authority string, WKT, Proj4, or `pyproj.CRS`.

### Concatenate safely

`FeatureCollection.concat` is CRS-checked — mixing a WGS84 FC with a
Web-Mercator one now raises instead of silently taking `self`'s CRS
(C32):

```python
import geopandas as gpd
from shapely.geometry import Point
from pyramids.feature import FeatureCollection

a = FeatureCollection(
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
)
b = FeatureCollection(
    gpd.GeoDataFrame({"id": [2]}, geometry=[Point(1, 1)], crs="EPSG:4326")
)
combined = a.concat(b)
```

To force-concat across CRSes, reproject first:

```python
combined = a.concat(b.to_crs(a.crs))
```

## Context-manager protocol

`FeatureCollection` implements `__enter__` / `__exit__` / `close()`
(ARC-5). OGR-backed resources — anything held through the private
`_ogr` bridge — are released on exit even if the body raises.

```python
from pyramids.feature import FeatureCollection

with FeatureCollection.read_file("roads.shp") as fc:
    fc.to_file("out.geojson", driver="GeoJSON")
```

## Rasterize

Go from a `FeatureCollection` to a `Dataset` via the raster side (ARC-4):

```python
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

fc = FeatureCollection.read_file("roads.shp")
ds = Dataset.from_features(fc, cell_size=30, column_name="lanes")
ds.to_file("roads_raster.tif")
```

`Dataset.from_features` validates up front (M4):

```python
>>> Dataset.from_features(fc, cell_size=30, column_name=123)
Traceback (most recent call last):
    ...
TypeError: column_name must be str, list[str], or None; got int.
```

## Build geometries directly

Lower-level factory functions on `pyramids.feature.geometry`:

```python
from pyramids.feature.geometry import (
    create_polygon, polygon_wkt, create_points, point_collection,
)

poly = create_polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
poly.area  # 1.0

wkt = polygon_wkt([(0, 0), (1, 0), (1, 1), (0, 1)])
# 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'

pts = create_points([(0.0, 0.0), (1.5, 2.5)])
# [<POINT (0 0)>, <POINT (1.5 2.5)>]

gdf = point_collection([(0, 0), (1, 1)], crs=4326)
```

## Error types

`pyramids.base._errors` groups vector-side failures under `FeatureError`:

| Exception | Parents | Raised from |
|-----------|---------|-------------|
| `FeatureError` | `Exception` | Base — catch-all for `pyramids.feature` failures |
| `InvalidGeometryError` | `FeatureError`, `ValueError` | Empty / malformed / wrong-type geometry (ARC-9, C22) |
| `CRSError` | `FeatureError`, `ValueError` | Missing / ambiguous / unparseable CRS (ARC-7, M1) |
| `VectorDriverError` | `FeatureError`, `RuntimeError` | OGR driver-level failure |
| `GeometryWarning` | `UserWarning` | Emitted (not raised) on degenerate geometry with fallback (L6) |

Legacy `except (ValueError, RuntimeError)` handlers keep working because
of the multi-inheritance.

## See also

- [Reference → FeatureCollection](../reference/feature/index.md)
- [Reference → Geometry helpers](../reference/feature/geometry.md)
- [Reference → CRS helpers](../reference/base/crs.md)
