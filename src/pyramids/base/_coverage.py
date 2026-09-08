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

The antimeridian split is here for the same reason. A ``bbox`` whose ``minx``
exceeds its ``maxx`` crosses the 180 degree seam, and every reader serves it the
same way: :func:`validate_bbox` with ``allow_antimeridian=True`` stops calling it
inverted, :func:`seam_halves` cuts it into the one or two ``west < east`` boxes to
actually request, and :func:`window_overlaps` drops a half that misses the
coverage before it costs a request. What each reader still owns is the merge,
because that is where the protocols differ -- a WMS divides a pixel width between
the halves, a vector reader de-duplicates features across them.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, cast

from osgeo import gdal, osr

from pyramids.base._bbox import split_antimeridian
from pyramids.base._bbox import transform as bbox_transform
from pyramids.base._errors import CoverageError, CRSError
from pyramids.base._grid import grid_size
from pyramids.base.crs import sr_from_user_input


def validate_bbox(
    bbox: tuple[float, float, float, float],
    *,
    allow_antimeridian: bool = False,
) -> tuple[float, float, float, float]:
    """Validate a ``(minx, miny, maxx, maxy)`` bbox.

    Args:
        bbox: Four numbers, or anything `float()` accepts for each of them --
            a bbox read out of JSON arrives as strings often enough that
            coercing is worth more than refusing.
        allow_antimeridian: Accept ``minx > maxx`` as a box crossing the 180
            degree seam rather than an inverted one. Off by default, because
            for most callers an inverted box is a mistake and silently reading
            it as a wrap would hide it. A reader that can actually serve the
            wrap -- by splitting it with :func:`seam_halves` and merging the
            results, as :meth:`Dataset.crop` does -- passes `True`. The Y axis
            is never wrapped: there is no seam in latitude, so ``miny >= maxy``
            stays an error either way.

    Returns:
        tuple[float, float, float, float]: The bbox as floats.

    Raises:
        ValueError: `bbox` is not four values, one of them is text `float()`
            cannot read, any of them is not finite, or the box is empty or
            inverted on either axis. With `allow_antimeridian`, ``minx > maxx``
            is no longer inverted -- but ``minx == maxx`` still is, since a
            zero-width box is empty whichever way it is read.
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
    wraps = allow_antimeridian and minx > maxx
    if (minx >= maxx and not wraps) or miny >= maxy:
        raise ValueError(f"bbox must have minx < maxx and miny < maxy, got {bbox!r}")
    return minx, miny, maxx, maxy


def _is_lonlat_degrees_from_greenwich(srs: osr.SpatialReference) -> bool:
    """Whether ``180`` is this CRS's antimeridian and ``360`` its seam offset.

    ``IsGeographic()`` alone does not settle either. A geographic CRS may count
    its longitudes from a different prime meridian -- EPSG:4807 (NTF, Paris) puts
    zero about 2.34 degrees east of Greenwich, so its antimeridian is not at 180 --
    and may express them in grads rather than degrees, where the half-turn is 200.
    Splitting such a bbox at 180 would cut it in the wrong place and then check the
    halves against a 360 that is not the width of the world in those units.

    Both are vanishingly rare as an OGC request CRS, which is why the check is a
    refusal rather than a conversion: the caller is far likelier to have transposed
    two corners than to genuinely want a wrap in grads from Paris.

    Args:
        srs: The CRS the bbox is expressed in.

    Returns:
        bool: True when the CRS is geographic, in degrees, and counted from
            Greenwich.

    Examples:
        - Plain lon/lat qualifies, and so does CRS84:
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> from pyramids.base._coverage import _is_lonlat_degrees_from_greenwich
            >>> _is_lonlat_degrees_from_greenwich(sr_from_user_input("EPSG:4326"))
            True

            ```
        - A projected CRS does not, having no meridian to speak of:
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> from pyramids.base._coverage import _is_lonlat_degrees_from_greenwich
            >>> _is_lonlat_degrees_from_greenwich(sr_from_user_input("EPSG:3857"))
            False

            ```
        - Nor does a geographic CRS counted from Paris, whose antimeridian is not
          at 180:
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> from pyramids.base._coverage import _is_lonlat_degrees_from_greenwich
            >>> _is_lonlat_degrees_from_greenwich(sr_from_user_input("EPSG:4807"))
            False

            ```
    """
    if not srs.IsGeographic():
        return False
    # GetAngularUnits reports radians per unit: degrees are pi/180, grads pi/200.
    degrees = abs(srs.GetAngularUnits() - 0.017453292519943295) < 1e-12
    # The offset is the PRIMEM node's second value, in degrees. There is no
    # GetPrimeMeridian on this binding, and a CRS carrying no PRIMEM at all is
    # Greenwich by definition.
    offset = srs.GetAttrValue("PRIMEM", 1)
    try:
        greenwich = offset is None or abs(float(offset)) < 1e-9
    except ValueError:
        greenwich = False
    return bool(degrees and greenwich)


def check_seam_bbox(bbox: tuple[float, float, float, float], crs: str) -> None:
    """Refuse a ``minx > maxx`` bbox this reader cannot read as an antimeridian wrap.

    A no-op for an ordinary box. For a wrapping one it asserts the two things the
    seam split silently assumes, so a transposed or projected-CRS bbox fails with a
    message naming the problem instead of producing a raster stitched at a seam
    that is not there.

    Every raster reader needs this, which is why it lives here rather than beside
    one of them. Without it a wrapping bbox given in a projected CRS is cut at
    ``+/-180`` *metres*: a transposed Web Mercator box around Scandinavia splits
    into two windows over central Europe and comes back with no error at all,
    where before the wrap was accepted it was a clean :class:`ValueError`. The
    corner-range half matters just as much -- a half that overhangs ``180`` yields
    an inverted window that :func:`window_overlaps` then discards, so the caller
    silently receives a fraction of what they asked for.

    Args:
        bbox: The validated ``(minx, miny, maxx, maxy)``, possibly wrapping.
        crs: The CRS ``bbox`` is expressed in (the WMS request CRS).

    Raises:
        ValueError: ``bbox`` wraps but ``crs`` is not geographic — the 180 degree
            seam is a lon/lat feature, and in a projected CRS ``minx > maxx`` is
            just an inverted box. Or it wraps but a corner lies outside
            ``-180 .. 180``, where "west of the seam" and "east of it" stop
            meaning anything.

    Examples:
        - An ordinary box passes in any CRS, projected included, because nothing
          about it needs a seam:
            ```python
            >>> from pyramids.base._coverage import check_seam_bbox
            >>> check_seam_bbox((5.0, 51.0, 6.0, 52.0), "EPSG:3857") is None
            True

            ```
        - A wrapping box in a projected CRS is refused rather than stitched at a
          seam that CRS does not have:
            ```python
            >>> from pyramids.base._coverage import check_seam_bbox
            >>> box = (170.0, -10.0, -170.0, 10.0)
            >>> check_seam_bbox(box, "EPSG:3857")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ValueError: bbox (170.0, ...) has minx > maxx, ...it has no such seam...

            ```
        - So is a wrapping box reaching outside ``-180 .. 180``, where the two
          sides of the seam stop being well defined:
            ```python
            >>> from pyramids.base._coverage import check_seam_bbox
            >>> box = (190.0, -10.0, -170.0, 10.0)
            >>> check_seam_bbox(box, "EPSG:4326")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ValueError: an antimeridian bbox must have both corners within -18...

            ```
    """
    minx, _, maxx, _ = bbox
    if minx > maxx:
        if not _is_lonlat_degrees_from_greenwich(sr_from_user_input(crs)):
            raise ValueError(
                f"bbox {bbox!r} has minx > maxx, which reads as a box crossing the "
                f"180 degree seam - but crs={crs!r} is not a geographic (lon/lat) "
                "CRS in degrees from Greenwich, so it has no such seam. Pass the "
                "bbox in a lon/lat CRS, or give it as minx < maxx."
            )
        if minx > 180.0 or maxx < -180.0:
            raise ValueError(
                "an antimeridian bbox must have both corners within -180..180 "
                f"degrees, got {bbox!r}"
            )


def seam_halves(
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """The one or two ``west < east`` boxes a request should actually ask for.

    A network reader has no grid to measure when it validates a bbox -- it is
    fetching the grid. So unlike :func:`pyramids.dataset.engines.spatial._antimeridian_halves`,
    which reads the seam out of a dataset it already holds, this splits at the
    180 degree meridian, which is where the seam is for the geographic CRS an
    OGC request declares.

    Note:
        This is, and must stay, a pass-through to
        :func:`pyramids.base._bbox.split_antimeridian`. The two names are
        deliberate -- `split_antimeridian` is the geometry primitive, re-exported
        from `pyramids.feature.bbox` and used by the crop engine, while this is the
        network readers' entry to the seam contract that lives beside
        :func:`check_seam_bbox` and :func:`window_overlaps`. Keeping a second name
        is only safe while it holds no logic of its own: any rule about *how* a
        bbox splits belongs in the primitive, so the two cannot drift apart.

    Args:
        bbox: A validated ``(minx, miny, maxx, maxy)``, possibly wrapping.

    Returns:
        list[tuple[float, float, float, float]]: One box when it does not wrap,
            two in west-to-east order when it does.

    Examples:
        - An ordinary box is handed back untouched, so a caller can split
          unconditionally and only branch on the length:
            ```python
            >>> from pyramids.base._coverage import seam_halves
            >>> seam_halves((10.0, -5.0, 20.0, 5.0))
            [(10.0, -5.0, 20.0, 5.0)]

            ```
        - A wrapping box becomes the two halves either side of the seam:
            ```python
            >>> from pyramids.base._coverage import seam_halves
            >>> seam_halves((170.0, -10.0, -170.0, 10.0))
            [(170.0, -10.0, 180.0, 10.0), (-180.0, -10.0, -170.0, 10.0)]

            ```
    """
    return split_antimeridian(bbox)


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

    Both branches return an SRS stamped with ``OAMS_TRADITIONAL_GIS_ORDER``, so a
    caller reading ``.GetSpatialRef()`` off a WCS / WMS / OGC API result raster
    gets lon/lat order whichever branch resolved it.

    Note:
        The ``coverage_crs`` branch resolves through
        :func:`pyramids.base.crs.sr_from_user_input`, which round-trips the CRS
        through pyproj. The resulting WKT keeps its root ``AUTHORITY`` node --
        so ``GetAuthorityCode(None)``, and hence ``Dataset.epsg``, are unchanged
        -- but loses the nested ones on the datum, spheroid, prime meridian and
        unit. A read whose CRS came from an explicit ``coverage_crs`` therefore
        reports a shorter ``.GetProjection()`` string than a raw
        ``SetFromUserInput`` would have produced.

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
        # resolved it.
        #
        # What consumes the stamp is `SetSpatialRef` on the result raster
        # (`dataset/_wcs.py`), which carries the SRS *object* -- and with it the
        # mapping -- onto what the caller gets back. It is not `native_projwin`:
        # that sees the SRS only as `native_srs.ExportToWkt()`, and WKT does not
        # encode the data-axis-to-SRS-axis mapping, so the stamp is invisible to
        # it (a clone with the mapping flipped exports byte-identical WKT).
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
            # stamps traditional axis order, matching the clone branch above,
            # so the SRS this hands to `SetSpatialRef` on the result raster
            # declares lon/lat either way. Building it raw left a geographic
            # `coverage_crs` in authority-compliant order. The cost is the WKT
            # detail named in the docstring's Note: the nested AUTHORITY nodes
            # do not survive the pyproj round-trip.
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


def seam_offset(
    bbox: tuple[float, float, float, float],
    crs: str,
    native_srs: Any,
) -> float:
    """The x distance between the -180 and +180 meridians in the layer's own CRS.

    What every seam-alignment check needs to know the east half really does
    continue where the west one stops: 360 for a geographic layer, the full world
    width (about 40 075 017 m) for a Web-Mercator one. Measured rather than
    assumed, because all three raster readers window the source in *its* CRS, not
    in the request's -- a WMTS layer is cropped in the pyramid's CRS, and a WCS or
    OGC API coverage in the coverage's. Assuming 360 there makes the check compare
    metres against degrees and reject every well-formed pair.

    Args:
        bbox: The request bbox, used only for the latitude band the meridians are
            measured across.
        crs: The CRS ``bbox`` is expressed in.
        native_srs: The layer's native spatial reference.

    Returns:
        float: The seam-to-seam x span in the native CRS's units.

    Examples:
        - A lon/lat layer measures the seam as the 360 degrees it is:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base._coverage import seam_offset
            >>> native = osr.SpatialReference()
            >>> _ = native.ImportFromEPSG(4326)
            >>> seam_offset((170.0, -10.0, -170.0, 10.0), "EPSG:4326", native)
            360.0

            ```
        - A Web Mercator layer measures the same seam in metres, so the check the
          offset feeds is done in the units the halves are actually cropped in:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base._coverage import seam_offset
            >>> native = osr.SpatialReference()
            >>> _ = native.ImportFromEPSG(3857)
            >>> round(seam_offset((170.0, -10.0, -170.0, 10.0), "EPSG:4326", native))
            40075017

            ```
    """
    _, miny, _, maxy = bbox
    world = native_projwin((-180.0, miny, 180.0, maxy), crs, native_srs)
    return world[2] - world[0]


def window_overlaps(projwin: list[float], src: gdal.Dataset) -> bool:
    """Whether a native-CRS ``[ulx, uly, lrx, lry]`` window meets `src`'s own extent.

    The seam readers split an antimeridian ``bbox`` into two halves and fetch each
    one; a half that misses the coverage entirely is skipped rather than requested.
    GDAL does not refuse such a window -- it warns ("Computed source window ...
    falls completely outside source raster extent") and fills the result with
    no-data -- so the point is not to avoid an error but to avoid paying for a
    request whose answer is a block of nothing, and then concatenating that block
    into the stitch as though it were data. This is the network equivalent of the
    overlap test in
    :func:`pyramids.dataset.engines.spatial._crop_seam_halves`, which reads the
    extent off a dataset it already holds.

    Note:
        The extent is bounded by all four corners, so a rotated geotransform
        (``gt[2]`` / ``gt[4]`` non-zero) is measured rather than under-reported --
        two opposite corners are not enough, because a rotated grid reaches beyond
        both of them on one axis. No reader can currently deliver a rotated source
        here, since ``gdal.Translate(projWin=...)`` refuses a rotated geotransform
        outright, so this is a defensive bound rather than a supported path.

    Args:
        projwin: ``[ulx, uly, lrx, lry]`` in `src`'s CRS, as
            :func:`pyramids.base._coverage.native_projwin` returns it.
        src: The opened coverage, read for its geotransform and pixel size.

    Returns:
        bool: True when the window and the source extent share area. Touching
            edges do not count as overlap -- a zero-area intersection has no
            pixels to read.

    Examples:
        - A window inside the raster overlaps, one beyond its east edge does not:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._coverage import window_overlaps
            >>> src = gdal.GetDriverByName("MEM").Create("", 10, 10, 1)
            >>> _ = src.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
            >>> window_overlaps([2.0, 8.0, 4.0, 6.0], src)
            True
            >>> window_overlaps([20.0, 8.0, 30.0, 6.0], src)
            False

            ```
    """
    gt = src.GetGeoTransform()
    # All four corners, not two: on a rotated grid the axis-aligned extent is set
    # by corners the diagonal does not touch, so an origin/far-corner pair
    # under-reports one axis and drops halves that do have data. min/max over the
    # four also makes the bound axis-order agnostic, which covers a south-up or
    # west-positive grid without a separate case.
    corners: list[tuple[float, float]] = [
        (
            float(gt[0] + x * gt[1] + y * gt[2]),
            float(gt[3] + x * gt[4] + y * gt[5]),
        )
        for x, y in (
            (0, 0),
            (src.RasterXSize, 0),
            (0, src.RasterYSize),
            (src.RasterXSize, src.RasterYSize),
        )
    ]
    minx, maxx = min(c[0] for c in corners), max(c[0] for c in corners)
    miny, maxy = min(c[1] for c in corners), max(c[1] for c in corners)
    win_minx, win_maxx = sorted((projwin[0], projwin[2]))
    win_miny, win_maxy = sorted((projwin[3], projwin[1]))
    return win_minx < maxx and win_maxx > minx and win_miny < maxy and win_maxy > miny


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
    ``format="MEM"`` is set here, and passing it is refused rather than silently
    colliding. Bounding the read is the **caller's** job, and the callers do not
    all bound it the same way -- the WCS and OGC coverage reads size through
    :func:`read_size`, while the WMS GetMap path sizes through its own
    ``_output_size`` with no pixel ceiling, because the server has already been
    told the size it should render. Where :func:`read_size` is used, it is
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
    if "format" in options:
        # Refused by name. It used to be a second `format=` on the same call, so
        # a caller passing one got `TranslateOptions() got multiple values for
        # keyword argument 'format'` -- a TypeError naming an internal call,
        # where the docstring promised a refusal.
        raise ValueError(
            "format is fixed to 'MEM' by translate_to_mem; the result is an "
            "in-memory dataset by construction. Write it out afterwards if you "
            "need another format."
        )
    # Built outside the guard: a bad keyword here is the caller's mistake, and
    # re-branding it as a service error would blame the server for it. Only the
    # translate itself is guarded.
    translate_options = gdal.TranslateOptions(format="MEM", **options)
    try:
        mem = gdal.Translate("", src, options=translate_options)
    except RuntimeError as exc:
        raise error(f"{action} failed for {subject}: {exc}") from exc
    if mem is None:
        raise error(f"{action} returned no raster for {subject}")
    return mem
