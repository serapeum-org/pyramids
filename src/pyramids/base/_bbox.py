"""Bbox reprojection, at the layer every reader can reach.

`transform` is the package's one bbox reprojection: densified edges through
`pyproj.Transformer.transform_bounds`, with the latitudes clamped when the
destination is geographic.

It lives under `base` rather than in `feature.bbox` because the coverage
readers need it, and `base` importing `feature` inverts the layering -- it also
pulled geopandas into any process that touched `base._coverage`. Nothing here
needs shapely or geopandas, only pyproj, so the move costs those callers
nothing. `pyramids.feature.bbox` re-exports it, so the name callers already use
still resolves.
"""

from __future__ import annotations

from pyproj import Transformer

from pyramids.base.crs import crs_from_user_input

Bbox = tuple[float, float, float, float]


def transform(
    bbox: Bbox,
    src_crs: object,
    dst_crs: object,
    densify_pts: int = 21,
) -> Bbox:
    """Reproject a bbox between two CRSes via :func:`pyproj.Transformer.transform_bounds`.

    The four edges are densified to `densify_pts` interior points before
    reprojecting, so curved CRS boundaries are not crudely axis-aligned. When
    the destination CRS is geographic, the returned latitudes are clamped to
    `[-90, 90]` to absorb floating-point overshoot at the poles.

    Args:
        bbox: A `(west, south, east, north)` / `(minx, miny, maxx, maxy)`
            tuple in `src_crs` units.
        src_crs: Source CRS — anything :meth:`pyproj.CRS.from_user_input`
            accepts (EPSG int, `"EPSG:XXXX"`, WKT, or PROJ string).
        dst_crs: Destination CRS, same accepted forms as `src_crs`.
        densify_pts: Number of densification points per edge. Defaults to 21.

    Returns:
        The reprojected `(minx, miny, maxx, maxy)` bbox in `dst_crs` units.

    Raises:
        pyramids.base._errors.CRSError: Either CRS could not be interpreted by
            `pyproj.CRS.from_user_input`.

    Examples:
        - A same-CRS transform is an identity (modulo float noise):
            ```python
            >>> [round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)]
            [-10.0, -5.0, 10.0, 5.0]

            ```
        - Reproject WGS84 degrees to Web Mercator metres:
            ```python
            >>> west, south, east, north = transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)
            >>> round(east)
            1113195

            ```
        - Authority strings are accepted for either CRS:
            ```python
            >>> result = transform((0.0, 0.0, 1.0, 1.0), "EPSG:4326", "EPSG:3857")
            >>> round(result[2]) > 0
            True

            ```

    See Also:
        pyramids.feature.bbox: Re-exports this function under the name
            callers used before it moved down to `base`.
    """
    src = crs_from_user_input(src_crs)
    dst = crs_from_user_input(dst_crs)
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    west, south, east, north = bbox
    minx, miny, maxx, maxy = transformer.transform_bounds(
        west, south, east, north, densify_pts=densify_pts
    )
    if dst.is_geographic:
        miny = max(miny, -90.0)
        maxy = min(maxy, 90.0)
    return (minx, miny, maxx, maxy)
