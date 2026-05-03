# CRS & Reprojection Helpers

CRS-handling helpers in `pyramids.base.crs` — the single source of
truth for `osr.SpatialReference` construction, WKT/Proj4 → EPSG
resolution, and coordinate reprojection. The most-commonly-used
ones are also re-exposed as `FeatureCollection` static methods for
ergonomic continuity (e.g.
`FeatureCollection.reproject_coordinates` delegates to
`pyramids.base.crs.reproject_coordinates`).

## Functions

::: pyramids.base.crs.sr_from_epsg
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.crs.sr_from_wkt
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.crs.create_sr_from_proj
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.crs.get_epsg_from_prj
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.crs.epsg_from_wkt
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.base.crs.reproject_coordinates
    options:
        show_root_heading: true
        heading_level: 3
