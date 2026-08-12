"""Shared CRS / bbox / window helpers for the coverage readers.

Both the WCS reader (:mod:`pyramids.dataset._wcs`) and the OGC API – Coverages
reader (:mod:`pyramids.dataset._ogc_coverages`) validate a lon/lat ``bbox``,
normalise a ``resolution``, resolve a coverage's native CRS (applying the
``coverage_crs`` shim when the advertised CRS is absent from PROJ) and project the
``bbox`` into that native CRS. Those steps are **protocol-neutral**, so they live
here once — neither reader reaches into the other's internals — and the CRS
resolver raises the protocol-neutral :class:`~pyramids.base._errors.CoverageError`,
which each reader re-wraps into its own branded error (WCSError / OGCAPIError).
"""

from __future__ import annotations

from math import isfinite

from osgeo import gdal, osr
from pyproj import CRS, Transformer

from pyramids.base._errors import CoverageError
from pyramids.base.crs import crs_from_user_input


def validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Validate a ``(minx, miny, maxx, maxy)`` bbox."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"bbox must have minx < maxx and miny < maxy, got {bbox!r}")
    return minx, miny, maxx, maxy


def resolution_pair(
    resolution: float | tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Normalise `resolution` to an ``(x_res, y_res)`` pair (or ``None``).

    Raises:
        ValueError: any axis of `resolution` is not strictly positive (a zero or
            negative pixel size cannot size a read window).
    """
    result: tuple[float, float] | None = None
    if resolution is not None:
        if isinstance(resolution, (int, float)):
            result = (float(resolution), float(resolution))
        else:
            x_res, y_res = resolution
            result = (float(x_res), float(y_res))
        if result[0] <= 0 or result[1] <= 0:
            raise ValueError(
                f"resolution must be strictly positive on each axis, got {resolution!r}"
            )
    return result


def resolve_native_srs(
    src: gdal.Dataset, coverage_crs: str | None
) -> osr.SpatialReference:
    """Return the coverage's native CRS, applying the ``coverage_crs`` shim.

    GDAL reports no spatial reference when the server's advertised CRS is not in
    the PROJ database. The caller must then supply ``coverage_crs``.

    Raises:
        CoverageError: The dataset has no CRS and no ``coverage_crs`` was given.
        ValueError: ``coverage_crs`` could not be interpreted.
    """
    srs = src.GetSpatialRef()
    if srs is not None:
        result = srs.Clone()
    elif coverage_crs is None:
        raise CoverageError(
            "the coverage has no resolvable spatial reference (the service likely "
            "advertises a CRS absent from the PROJ database). Pass coverage_crs= "
            "with the coverage's CRS, e.g. the proj4 string."
        )
    else:
        result = osr.SpatialReference()
        try:
            # GDAL exceptions are enabled package-wide, so a bad CRS raises here.
            result.SetFromUserInput(coverage_crs)
        except RuntimeError as exc:
            raise ValueError(
                f"coverage_crs could not be interpreted: {coverage_crs!r} ({exc})"
            ) from exc
    return result


def native_projwin(
    bbox: tuple[float, float, float, float],
    crs: str,
    native_srs: osr.SpatialReference,
) -> list[float]:
    """Transform a lon/lat-ordered `bbox` into a native-CRS ``projWin``.

    Returns ``[ulx, uly, lrx, lry]`` in the native CRS, the form
    :func:`gdal.Translate` expects.

    Raises:
        ValueError: the bbox does not project to a finite native-CRS window
            (``pyproj`` returns ``inf``/``nan`` when the bbox falls outside the
            native CRS's area of use).
    """
    native = CRS.from_user_input(native_srs.ExportToWkt())
    transformer = Transformer.from_crs(crs_from_user_input(crs), native, always_xy=True)
    minx, miny, maxx, maxy = bbox
    # Densify the edges (not just the corners) so the native-CRS window still
    # covers the requested area under projection curvature / interruptions (e.g.
    # the Interrupted Goode Homolosine), where the corner hull can bow inward.
    left, bottom, right, top = transformer.transform_bounds(
        minx, miny, maxx, maxy, densify_pts=21
    )
    projwin = [left, top, right, bottom]
    if not all(isfinite(v) for v in projwin):
        raise ValueError(
            f"bbox {bbox!r} does not project to a finite window in the coverage's "
            "native CRS; pass a bbox within the coverage's extent / CRS area of use"
        )
    return projwin


# Pixel-count caps for a windowed coverage read, shared by the WCS / WMTS / OGC
# API readers. DEFAULT_MAX_PX bounds the longer side of a no-resolution "preview"
# read; MAX_PX is the hard ceiling enforced on every read (even with a resolution)
# so a fine resolution over a wide bbox cannot request an unbounded allocation.
DEFAULT_MAX_PX = 1024
MAX_PX = 25000


def read_size(projwin: list[float], res: tuple[float, float] | None) -> tuple[int, int]:
    """Compute the capped ``(width, height)`` pixel size for a windowed read.

    ``projwin`` is ``[ulx, uly, lrx, lry]`` in the native CRS; its span gives the
    window extent in that CRS's units. With a ``res`` the size follows directly
    (``span / res``); without one the longer side is capped at :data:`DEFAULT_MAX_PX`
    and the shorter scaled to preserve the aspect ratio. Every dimension is clamped
    to at least 1 and rejected above the hard :data:`MAX_PX` ceiling, so even a fine
    ``res`` over a wide ``bbox`` cannot request an unbounded read. Callers that read
    at native resolution should pass :func:`native_resolution` as ``res`` to bound
    that read at the ceiling.

    Raises:
        ValueError: `res` has a non-positive axis (a degenerate or fully rotated
            geotransform yields a zero native pixel size), or the requested window
            exceeds :data:`MAX_PX` on either side.
    """
    ulx, uly, lrx, lry = projwin
    span_x = abs(lrx - ulx)
    span_y = abs(uly - lry)
    if res is not None:
        x_res, y_res = res
        if x_res <= 0 or y_res <= 0:
            # native_resolution() can yield a zero axis for a degenerate or fully
            # rotated geotransform (the pixel scale lives in gt[2]/gt[4], not
            # gt[1]/gt[5]); reject it clearly instead of dividing by zero.
            raise ValueError(
                f"resolution must be strictly positive on each axis to size a read, "
                f"got {res!r}; pass an explicit positive resolution"
            )
        width = max(1, round(span_x / x_res))
        height = max(1, round(span_y / y_res))
    elif span_x >= span_y:
        width = DEFAULT_MAX_PX
        height = (
            max(1, round(DEFAULT_MAX_PX * span_y / span_x))
            if span_x
            else DEFAULT_MAX_PX
        )
    else:
        height = DEFAULT_MAX_PX
        width = (
            max(1, round(DEFAULT_MAX_PX * span_x / span_y))
            if span_y
            else DEFAULT_MAX_PX
        )
    if width > MAX_PX or height > MAX_PX:
        raise ValueError(
            f"the requested window is {width}x{height} px (over the {MAX_PX} px "
            "limit); pass a coarser resolution or a smaller bbox to keep the read "
            "bounded"
        )
    return width, height


def native_resolution(src: gdal.Dataset) -> tuple[float, float]:
    """Return the source raster's absolute native ``(x_res, y_res)`` from its geotransform."""
    gt = src.GetGeoTransform()
    return (abs(gt[1]), abs(gt[5]))
