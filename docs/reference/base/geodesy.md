# Geodesy Helpers

Ground-measurement helpers in `pyramids.base.geodesy`. Where
[`pyramids.base.crs`](crs.md) answers what a coordinate *means*, these answer how
far apart two coordinates are on the figure of the earth, and how much of a CRS's
own unit a given ground distance occupies at a given place.

The ellipsoid always comes from the CRS's own datum, never from a hard-coded
WGS 84 default — the same discipline
[`Dataset.cell_area`](../dataset/cell.md) uses when it integrates cell areas.

## Why the place matters

There is no single metres-per-degree constant. A degree of longitude is about
111 km at the equator and 0 at the pole, and Web Mercator stretches its own metre
by `1 / cos(lat)`. So `ground_distance_in_crs` takes the location as a required
argument rather than assuming one:

```python
from pyramids.base.geodesy import ground_distance_in_crs

ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 0.0))    # 0.8983 degrees
ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 60.0))   # 1.7917 degrees
```

This is what sizing a scale bar needs: cleopatra's `add_scale_bar` takes a length
in the axes' data units, and converting "100 km on the ground" into those units
is a question about the Earth, not about pixels.

## Coordinates are always geographic degrees

Every function here takes geographic degrees and uses `crs` only to select the
ellipsoid. Projected coordinates are **refused**, not answered, because PROJ
would return `nan` for an out-of-range latitude and that would propagate
silently. Reproject first with
[`reproject_coordinates`](crs.md#pyramids.base.crs.reproject_coordinates).

`at` is always in `(x, y)` order — easting then northing, longitude then
latitude — whatever axis order the CRS itself declares.

## Vector counterparts

`FeatureCollection` wraps these as the geodesic counterparts of the planar
operations it inherits from `GeoDataFrame`, for callers who hold geometries
rather than coordinates:

| Geodesic | Planar twin |
|---|---|
| `FeatureCollection.geodesic_distance` | `.distance` |
| `FeatureCollection.geodesic_length` | `.length` |
| `FeatureCollection.geodesic_area` | `.area` |

See [the FeatureCollection reference](../feature/index.md).

## Functions

::: pyramids.base.geodesy.geodesic_distance
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.geodesy.ground_distance_in_crs
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.geodesy.geodesic_geometry_length
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.geodesy.geodesic_geometry_area
    options:
        show_root_heading: true
        heading_level: 3
