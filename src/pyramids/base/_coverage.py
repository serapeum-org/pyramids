"""Shared CRS / bbox / window helpers for the coverage readers.

Both the WCS reader (:mod:`pyramids.dataset._wcs`) and the OGC API – Coverages
reader (:mod:`pyramids.dataset._ogc_coverages`) validate a lon/lat ``bbox``,
normalise a ``resolution``, resolve a coverage's native CRS (applying the
``coverage_crs`` shim when the advertised CRS is absent from PROJ) and project the
``bbox`` into that native CRS. Those steps are **protocol-neutral**, so they live
here once — neither reader reaches into the other's internals — and the CRS
resolver raises the protocol-neutral :class:`~pyramids.base._errors.CoverageError`,
which each reader re-wraps into its own branded error (WCSError / OGCAPIError).

The two GDAL calls those readers wrap around live here too. Every network reader
— WCS, WMS / WMTS (:mod:`pyramids.dataset._wms`) and OGC API – Coverages — opens a
connection string with GDAL and then materialises a window of it into a ``MEM``
dataset, and each has to turn the same two failure shapes into its own branded
error: a ``RuntimeError`` (GDAL raises under ``gdal.UseExceptions()``) and a
``None`` return (a driver that declines the source without raising).
:func:`open_network_dataset` and :func:`translate_to_mem` own that sequence and
that classification; the readers pass in their own exception class and the words
that name the request, so the messages stay branded per protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, cast

from osgeo import gdal, osr

from pyramids.base._bbox import transform as bbox_transform
from pyramids.base._errors import CoverageError, CRSError
from pyramids.base._grid import grid_size
from pyramids.base.crs import sr_from_user_input


def validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Validate a ``(minx, miny, maxx, maxy)`` bbox.

    Args:
        bbox: Four numbers, or anything `float()` accepts for each of them --
            a bbox read out of JSON arrives as strings often enough that
            coercing is worth more than refusing.

    Returns:
        tuple[float, float, float, float]: The bbox as floats.

    Raises:
        ValueError: `bbox` is not four values, one of them is text `float()`
            cannot read, any of them is not finite, or the box is empty or
            inverted on either axis.
        TypeError: One of the four is a value `float()` refuses outright, such
            as `None` or a list. Raised by the coercion rather than by a check
            here -- the message names the type, which is what the caller needs.

    Examples:
        - An ordinary box passes and comes back as floats:
            ```python
            >>> from pyramids.base._coverage import validate_bbox
            >>> validate_bbox(("1", "2", "3", "4"))
            (1.0, 2.0, 3.0, 4.0)

            ```
        - A non-finite corner is refused here rather than reaching a request
          URL as the literal text `nan`:
            ```python
            >>> from pyramids.base._coverage import validate_bbox
            >>> validate_bbox((1.0, 2.0, float("nan"), 4.0))
            Traceback (most recent call last):
            ValueError: bbox must be four finite numbers, got (1.0, 2.0, nan, 4.0)

            ```
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    # Checked before the ordering test, which cannot see them: every comparison
    # against NaN is False, so `minx >= maxx` passes a NaN corner straight
    # through, and an infinite one compares as a legitimately huge box. Both
    # then reach a WCS / WMS request as the literal text `nan` / `inf`, where
    # the failure is the server's and reads as a network problem.
    if not all(isfinite(v) for v in (minx, miny, maxx, maxy)):
        raise ValueError(f"bbox must be four finite numbers, got {bbox!r}")
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
        # Stamped here too, not only on the `coverage_crs` branch below. The
        # clone carries whatever mapping the driver attached -- usually GDAL's
        # authority-compliant default -- so leaving it made the two branches
        # disagree, and the SRS on a WCS/WMS result raster then declared
        # traditional order or authority order depending only on which branch
        # resolved it. `native_projwin`'s always_xy transformer assumes
        # traditional throughout.
        result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    elif coverage_crs is None:
        raise CoverageError(
            "the coverage has no resolvable spatial reference (the service likely "
            "advertises a CRS absent from the PROJ database). Pass coverage_crs= "
            "with the coverage's CRS, e.g. the proj4 string."
        )
    else:
        try:
            # `sr_from_user_input` rather than a bare SetFromUserInput: it
            # stamps traditional axis order, matching the clone branch above
            # and `native_projwin`'s always_xy transformer. Building it raw
            # left a geographic `coverage_crs` in authority-compliant order.
            result = sr_from_user_input(coverage_crs)
        except (RuntimeError, CRSError) as exc:
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
    # `base._bbox.transform` is the package's bbox reprojection: same
    # densification (21 points per edge, so a curved or interrupted projection
    # is not crudely axis-aligned), same always_xy convention, and it resolves a
    # CRS whose code only GDAL's PROJ database carries (#943).
    left, bottom, right, top = bbox_transform(
        cast("tuple[float, float, float, float]", tuple(bbox)),
        crs,
        native_srs.ExportToWkt(),
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
        width, height = grid_size(span_x, span_y, (x_res, y_res), max_px=None)
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


def open_network_dataset(
    connection: str,
    *,
    error: type[Exception],
    subject: str,
    open_options: Sequence[str] | None = None,
) -> gdal.Dataset:
    """Open a GDAL network connection, re-branding both failure shapes as `error`.

    The WCS, WMS / WMTS and OGC API – Coverages readers each hand GDAL a connection
    string — a ``<WCS_GDAL>`` or ``<GDAL_WMS>`` service descriptor, a ``WMTS:`` or
    ``OGCAPI:`` connection — and each has to answer for two different failures: GDAL
    raises ``RuntimeError`` under ``gdal.UseExceptions()``, but a driver that declines
    the source without an error returns ``None`` instead. Handling only the first
    leaks a ``None`` that fails an attribute access one frame later; handling neither
    leaks a raw GDAL message with no idea which coverage or layer it was about.

    Args:
        connection: The GDAL connection string / service descriptor to open.
        error: The reader's branded exception class (``WCSError``, ``WMSError``,
            ``OGCAPIError``, ...), called with a single message argument.
        subject: What is being opened, already worded for the message — e.g.
            ``f"WCS coverage {coverage!r}"`` or ``f"WMTS layer {layer!r}"``. It is the
            only reader-specific text in either message.
        open_options: GDAL open options. ``None`` (the default) opens through
            :func:`osgeo.gdal.Open`; a sequence opens through :func:`osgeo.gdal.OpenEx`
            with ``gdal.OF_RASTER`` and these options, which is how the ``OGCAPI``
            driver is told which API and image format to negotiate.

    Returns:
        gdal.Dataset: The opened dataset, never ``None``.

    Raises:
        error: GDAL raised while opening, or returned no dataset.

    Examples:
        - A connection GDAL can open comes back as a dataset (a ``/vsimem`` GeoTIFF
          stands in here for the network source):
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._coverage import open_network_dataset
            >>> writer = gdal.GetDriverByName("GTiff").Create("/vsimem/doc_cov.tif", 2, 3, 1)
            >>> writer = None
            >>> src = open_network_dataset(
            ...     "/vsimem/doc_cov.tif", error=RuntimeError, subject="coverage 'demo'"
            ... )
            >>> (src.RasterXSize, src.RasterYSize)
            (2, 3)
            >>> src = None
            >>> _ = gdal.Unlink("/vsimem/doc_cov.tif")

            ```
        - A source GDAL refuses raises the reader's own error class, naming the
          subject and keeping GDAL's own message as the tail:
            ```python
            >>> from pyramids.base._coverage import open_network_dataset
            >>> class DemoError(Exception):
            ...     pass
            >>> try:
            ...     open_network_dataset(
            ...         "/vsimem/absent.tif", error=DemoError, subject="coverage 'demo'"
            ...     )
            ... except DemoError as exc:
            ...     print(str(exc).startswith("could not open coverage 'demo': "))
            True

            ```
    """
    try:
        if open_options is None:
            src = gdal.Open(connection)
        else:
            src = gdal.OpenEx(
                connection, gdal.OF_RASTER, open_options=list(open_options)
            )
    except RuntimeError as exc:
        raise error(f"could not open {subject}: {exc}") from exc
    if src is None:
        raise error(f"GDAL returned no dataset for {subject}")
    return src


def translate_to_mem(
    src: gdal.Dataset,
    *,
    error: type[Exception],
    action: str,
    subject: str,
    **options: Any,
) -> gdal.Dataset:
    """Materialise a window of `src` into a ``MEM`` dataset, re-branding failures as `error`.

    Every network reader ends the same way: :func:`osgeo.gdal.Translate` into ``MEM``,
    never straight to the caller's output path. That is what guarantees a service
    answering with an ``<ows:ExceptionReport>`` (or an HTML error page, or a truncated
    body) cannot be written to a ``.tif`` the caller then has to discover is not a
    raster — the failure happens here, before any file exists. The two failure shapes
    are the ones :func:`open_network_dataset` describes: a ``RuntimeError`` and a
    ``None`` return.

    The window itself is `options`: ``projWin`` bounds the area, ``width`` / ``height``
    (or ``xRes`` / ``yRes``) bound the allocation, ``resampleAlg`` picks the kernel.
    ``format="MEM"`` is set here and is not overridable. Bounding the read is the
    **caller's** job: pass the intended size through :func:`read_size` first, which is
    where the :data:`MAX_PX` ceiling is enforced.

    Args:
        src: The opened network dataset to read from.
        error: The reader's branded exception class, called with one message.
        action: What the read is, in the reader's own words — ``"WCS GetCoverage"``,
            ``"WMS GetMap"``, ``"WMTS tile read"``, ``"OGC API coverage read"``. It
            opens both messages.
        subject: The request target as it should read in the message, normally
            ``repr(coverage)`` / ``repr(layer)``.
        **options: Extra :func:`osgeo.gdal.TranslateOptions` keywords describing the
            window.

    Returns:
        gdal.Dataset: The in-memory window, never ``None``.

    Raises:
        error: GDAL raised while translating, or produced no raster.

    Examples:
        - The ``projWin`` window is materialised in memory at the source's own
          resolution:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._coverage import translate_to_mem
            >>> src = gdal.GetDriverByName("MEM").Create("", 8, 8, 1)
            >>> _ = src.SetGeoTransform((0.0, 1.0, 0.0, 8.0, 0.0, -1.0))
            >>> mem = translate_to_mem(
            ...     src,
            ...     error=RuntimeError,
            ...     action="demo read",
            ...     subject="'demo'",
            ...     projWin=[2.0, 6.0, 6.0, 2.0],
            ... )
            >>> (mem.RasterXSize, mem.RasterYSize)
            (4, 4)

            ```
        - A read GDAL cannot satisfy surfaces as the reader's own error, opened by the
          action and naming the subject:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._coverage import translate_to_mem
            >>> class DemoError(Exception):
            ...     pass
            >>> src = gdal.GetDriverByName("MEM").Create("", 8, 8, 1)
            >>> try:
            ...     translate_to_mem(
            ...         src,
            ...         error=DemoError,
            ...         action="demo read",
            ...         subject="'demo'",
            ...         bandList=[5],
            ...     )
            ... except DemoError as exc:
            ...     print(str(exc).startswith("demo read failed for 'demo': "))
            True

            ```
    """
    try:
        mem = gdal.Translate(
            "", src, options=gdal.TranslateOptions(format="MEM", **options)
        )
    except RuntimeError as exc:
        raise error(f"{action} failed for {subject}: {exc}") from exc
    if mem is None:
        raise error(f"{action} returned no raster for {subject}")
    return mem
