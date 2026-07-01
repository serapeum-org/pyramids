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
    src: "gdal.Dataset", coverage_crs: str | None
) -> "osr.SpatialReference":
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
    native_srs: "osr.SpatialReference",
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
    transformer = Transformer.from_crs(CRS.from_user_input(crs), native, always_xy=True)
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
