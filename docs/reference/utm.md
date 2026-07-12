# UTM Zone & EPSG Helpers

UTM helpers in `pyramids.utm` for resolving the UTM zone / EPSG code of a WGS84
point or vector layer — used for per-tile reprojection, local-metre areas of
interest, and STAC point cubes.

These return the **EPSG-correct** zone: the plain 6°-wide longitude bands, with
the hemisphere selecting the `326xx` (north) or `327xx` (south) band. The
Norway/Svalbard zone shifts are the MGRS grid-zone lettering convention, *not* the
UTM CRS definitions, so they are deliberately not applied — EPSG:32631 (0°E–6°E) is
the zone whose area of use covers Bergen at 5°E, while EPSG:32632 (6°E–12°E) does
not. The values here agree with `pyproj.database.query_utm_crs_info`.

## Functions

::: pyramids.utm.utm_zone
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.utm.utm_epsg
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.utm.utm_epsg_for_polygon
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.utm.project_to_utm
    options:
        show_root_heading: true
        heading_level: 3
