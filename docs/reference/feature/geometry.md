# Geometry Helpers

Pure functions for building and inspecting shapely geometries. These
sit on `pyramids.feature.geometry` and are re-exposed as
`FeatureCollection` static methods for convenience (e.g.
`FeatureCollection.create_polygon` is a thin delegate to
`pyramids.feature.geometry.create_polygon`).

## Factories

Create shapely geometries from coordinate sequences. ARC-15 split these
by return type so each function has a single, unambiguous output:

| Function | Returns |
|----------|---------|
| `create_polygon(coords)` | `shapely.Polygon` |
| `polygon_wkt(coords)` | WKT `str` |
| `create_points(coords)` | `list[shapely.Point]` |
| `point_collection(coords, crs)` | `geopandas.GeoDataFrame` |

::: pyramids.feature.geometry.create_polygon
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.polygon_wkt
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.create_points
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.point_collection
    options:
        show_root_heading: true
        heading_level: 3

## Coordinate Extraction

Per-geometry coordinate accessors. `get_coords` dispatches by geometry
type; the typed helpers (`get_point_coords`, `get_line_coords`,
`get_poly_coords`, `get_xy_coords`) are lazy — they check `coord_type`
before touching the geometry's attributes (M2), so an invalid
`coord_type` raises before a degenerate geometry can trigger a
`GEOSException`.

::: pyramids.feature.geometry.get_coords
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.get_xy_coords
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.get_point_coords
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.get_line_coords
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.get_poly_coords
    options:
        show_root_heading: true
        heading_level: 3

## Multi-geometry Handling

`explode_gdf` expands `MultiPolygon` (or `GeometryCollection`) rows
into one-geometry-per-row frames; `multi_geom_handler` and
`geometry_collection_coords` return per-part coordinate sequences
without exploding.

::: pyramids.feature.geometry.explode_gdf
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.multi_geom_handler
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.geometry.geometry_collection_coords
    options:
        show_root_heading: true
        heading_level: 3
