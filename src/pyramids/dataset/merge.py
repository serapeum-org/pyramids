"""Free-function entry points for merging a list of rasters.

* :func:`merge_rasters` — mosaic rasters that tile the *same area* into one
  raster (spatial merge).
* :func:`stack_bands` — stack N single-band rasters that cover the *same
  grid* into one multi-band raster (band-wise merge). Thin alias for
  :meth:`pyramids.dataset.Dataset.from_band_files`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, osr
from pyproj.exceptions import ProjError

from pyramids.base._utils import DEFAULT_RESAMPLING, resolve_resampling
from pyramids.base.remote import signer_cloud_config
from pyramids.dataset._driver import resolve_output_driver
from pyramids.dataset.dataset import _INHERIT_NO_DATA, Dataset
from pyramids.feature.bbox import normalise_longitude
from pyramids.feature.bbox import transform as bbox_transform

_VRT_METHODS = ("first", "last")
_REDUCE_METHODS = ("min", "max", "sum")
_MERGE_METHODS = _VRT_METHODS + _REDUCE_METHODS

# Rows per strip for the min/max/sum reduction. The union grid is reduced one
# full-width strip at a time so peak memory is O(strip) rather than O(grid).
_MERGE_STRIP_ROWS = 512

# Fraction of a pixel within which a window edge is treated as landing exactly on
# a grid line. Reprojection and float arithmetic put an edge a few ulps past a
# boundary, and snapping outward on that noise costs a spurious row or column.
# Note this is not what keeps the two merge paths consistent: both consume the one
# window `_restrict_grid` resolves, so they agree for any tolerance, zero included.
_GRID_SNAP_TOLERANCE = 1e-6

# Value a strip accumulator starts at, per reduction, so the first real sample wins:
# +inf loses every fmin, -inf loses every fmax, 0 is the additive identity. Cells that
# never receive a sample keep this value and are replaced by the fill at the end.
_REDUCE_IDENTITY = {"min": np.inf, "max": -np.inf, "sum": 0.0}


def _validated_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Coerce `bbox` to four finite floats in ``(west, south, east, north)`` order.

    Validated up front, and in one place, because every downstream consumer fails
    differently and late: a string of the right length is happily unpacked into four
    coordinates, a ``NaN`` reaches the grid arithmetic and surfaces as ``cannot
    convert float NaN to integer``, and an inverted bbox is rejected on one merge
    path while GDAL silently normalises it on the other.

    Args:
        bbox: The caller's window, expected to be four numbers.

    Returns:
        tuple[float, float, float, float]: ``(west, south, east, north)``.

    Raises:
        TypeError: `bbox` is a string or bytes, is not iterable, or holds a
            non-numeric element. Any other iterable of four numbers is accepted,
            `np.ndarray` included.
        ValueError: There are not exactly four values, a coordinate is not finite,
            or the box is inverted or empty in either axis.
    """
    # Accept any non-string iterable rather than testing `isinstance(bbox, Sequence)`:
    # `np.ndarray` does not register as a `Sequence`, and `GeoDataFrame.total_bounds`
    # -- the most natural way a caller here produces a bbox -- returns exactly that.
    # Strings are excluded first because a 4-character one would otherwise unpack into
    # four coordinates.
    if isinstance(bbox, (str, bytes)):
        raise TypeError(
            f"bbox must be four numbers (west, south, east, north), "
            f"got {type(bbox).__name__}: {bbox!r}"
        )
    try:
        values = list(bbox)
    except TypeError as exc:
        raise TypeError(
            f"bbox must be four numbers (west, south, east, north), "
            f"got {type(bbox).__name__}: {bbox!r}"
        ) from exc
    if len(values) != 4:
        raise ValueError(
            f"bbox must have exactly 4 values "
            f"(west, south, east, north), got {len(values)}: {tuple(values)!r}"
        )
    try:
        west, south, east, north = (float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"bbox values must be numbers, got {tuple(values)!r}") from exc
    if not all(np.isfinite(v) for v in (west, south, east, north)):
        raise ValueError(f"bbox values must all be finite, got {tuple(values)!r}")
    if west >= east or south >= north:
        raise ValueError(
            f"bbox must be (west, south, east, north) with west < east "
            f"and south < north, got {tuple(values)!r}. A zero-area or inverted box "
            "selects nothing."
        )
    return west, south, east, north


def _match_longitude_convention(
    west: float, east: float, projection: str, grid_west: float, grid_east: float
) -> tuple[float, float]:
    """Rewrite a lon/lat window into the longitude convention the mosaic uses.

    Global grids derived from climate NetCDF commonly run ``0..360`` while callers
    write bboxes as ``-180..180``. Left alone the two conventions overlap only
    partially, and the clamp in `_restrict_grid` silently reduces the window to that
    partial overlap — the caller gets the eastern sliver of the area they asked for
    and a successful return.

    Args:
        west: Window's western edge, in the mosaic's CRS.
        east: Window's eastern edge, in the mosaic's CRS.
        projection: The mosaic's CRS as WKT, or any form
            :meth:`osr.SpatialReference.SetFromUserInput` accepts.
        grid_west: The mosaic's western edge.
        grid_east: The mosaic's eastern edge.

    Returns:
        tuple[float, float]: ``(west, east)``, rewritten when the conventions differ
        and left untouched otherwise (including for every projected CRS).
    """
    result = (west, east)
    srs = osr.SpatialReference()
    try:
        srs.SetFromUserInput(projection)
        geographic = bool(srs.IsGeographic())
    except (RuntimeError, TypeError):
        geographic = False
    if geographic:
        if grid_east > 180.0 and west < 0.0:
            rewritten = normalise_longitude((west, 0.0, east, 0.0), "0..360")
            result = (rewritten[0], rewritten[2])
        elif grid_west < 0.0 and west > 180.0:
            rewritten = normalise_longitude((west, 0.0, east, 0.0), "-180..180")
            result = (rewritten[0], rewritten[2])
    return result


def _bbox_in_projection(
    bbox: Sequence[float], bbox_crs: int | str | None, projection: str
) -> tuple[float, float, float, float]:
    """Express `bbox` in `projection`'s coordinates.

    Args:
        bbox: ``(west, south, east, north)`` in `bbox_crs`, or already in
            `projection` when `bbox_crs` is ``None``.
        bbox_crs: CRS the bbox is given in — any form
            :meth:`pyproj.CRS.from_user_input` accepts. ``None`` means the bbox is
            already in the target CRS.
        projection: The target CRS as WKT (a grid's ``GetProjection()``).

    Returns:
        tuple[float, float, float, float]: ``(west, south, east, north)`` in the
        target CRS.

    Raises:
        TypeError: `bbox` is not four numbers.
        ValueError: The bbox is inverted, does not project to a finite extent, or
            crosses the antimeridian once reprojected. `transform_bounds` signals
            that last case by returning ``west > east``, which would otherwise be
            read as a box spanning the long way round.
    """
    west, south, east, north = _validated_bbox(bbox)
    if bbox_crs is None:
        return west, south, east, north
    # Delegate to the shared bbox reprojector rather than transforming corners here.
    # Corners are not enough: a reprojected rectangle is a curved quadrilateral whose
    # extreme point generally lies in the *interior* of an edge, so a corner envelope
    # under-covers the requested area -- measured at ~0.035 deg (~4 km) off the north
    # edge for an EPSG:3035 window into a lon/lat mosaic. `feature.bbox.transform`
    # densifies every edge (via `pyproj.Transformer.transform_bounds`), which is also
    # what GDAL does internally for `projWinSRS`, so both merge paths agree.
    try:
        west, south, east, north = bbox_transform(
            (west, south, east, north), bbox_crs, projection
        )
    except (ProjError, ValueError, TypeError) as exc:
        raise ValueError(
            f"bbox {tuple(bbox)!r} could not be reprojected from "
            f"{bbox_crs!r} into the mosaic CRS: {exc}"
        ) from exc
    if not all(np.isfinite(v) for v in (west, south, east, north)):
        raise ValueError(
            f"bbox {tuple(bbox)!r} does not project from {bbox_crs!r} "
            "into the mosaic CRS; it likely falls outside that CRS's area of use."
        )
    # `transform_bounds` signals an antimeridian crossing by returning west > east
    # rather than by widening the envelope. Taken at face value that reads as a box
    # spanning the long way round: sorting the two edges would resolve the ~2 deg
    # window this describes into its ~358 deg complement -- silently the wrong area,
    # and a near-global read from a call whose whole purpose is a bounded one.
    # `_validated_bbox` already refuses an inverted bbox on input; refuse the same
    # shape here, where reprojection is what produced it.
    if west > east:
        raise ValueError(
            f"bbox {tuple(bbox)!r} crosses the antimeridian when "
            f"reprojected from {bbox_crs!r} into the mosaic CRS (it spans "
            f"{west} to {east}). Split it into one window either side of "
            "180 deg and merge them separately."
        )
    return west, south, east, north


def _restrict_grid(
    geotransform: Sequence[float],
    x_size: int,
    y_size: int,
    projection: str,
    bbox: Sequence[float],
    bbox_crs: int | str | None,
) -> tuple[tuple[float, float, float, float, float, float], int, int]:
    """Clip a union grid to `bbox`, snapped outward onto the grid's own pixels.

    Snapping outward (floor the offsets, ceil the far edges) keeps the result a
    strict sub-grid of the union: every output pixel still lines up with a source
    pixel, so the strip reduction stays byte-identical to a whole-grid pass over
    the same area. Rounding inward would drop a partially-covered edge pixel the
    caller asked for.

    Args:
        geotransform: The union grid's GDAL geotransform.
        x_size: Union grid width in pixels.
        y_size: Union grid height in pixels.
        projection: The union grid's CRS as WKT.
        bbox: ``(west, south, east, north)`` window to keep.
        bbox_crs: CRS of `bbox`, or ``None`` when it is already in `projection`.

    Returns:
        tuple: ``(geotransform, x_size, y_size)`` for the clipped grid.

    Raises:
        ValueError: The mosaic is rotated/sheared or has a zero pixel size; the bbox
            selects no whole pixel; or it does not overlap the mosaic at all — a
            silent empty output would look like a successful merge of nothing.
        TypeError: `bbox` is not four numbers.
    """
    west, south, east, north = _bbox_in_projection(bbox, bbox_crs, projection)
    origin_x, pixel_w, row_skew, origin_y, col_skew, pixel_h = (
        float(v) for v in geotransform
    )
    # Defensive: gdal.BuildVRT produces an axis-aligned union grid, so a skewed
    # geotransform should not reach here. Guard it anyway rather than silently
    # mis-georeferencing, and do not imply that dropping the bbox would help — a
    # rotated mosaic is not something merge_rasters handles either way.
    if row_skew or col_skew:
        raise ValueError(
            "cannot resolve a bbox against a rotated or sheared mosaic "
            f"(geotransform skew terms {row_skew!r}, {col_skew!r}); the window would "
            "be applied as if the grid were axis-aligned, mis-georeferencing the "
            "output."
        )
    if not pixel_w or not pixel_h:
        raise ValueError(
            f"mosaic has a zero pixel size ({pixel_w!r}, {pixel_h!r}); "
            "the bbox window cannot be resolved onto its grid."
        )

    # Put the window in the mosaic's longitude convention before measuring offsets
    # against it. A -180..180 window against a 0..360 mosaic otherwise overlaps only
    # partially, and the clamp below would quietly return that sliver as a success.
    grid_edges = sorted((origin_x, origin_x + x_size * pixel_w))
    west, east = _match_longitude_convention(
        west, east, projection, grid_edges[0], grid_edges[1]
    )
    # Rewriting can itself put the window across the seam (a window spanning the prime
    # meridian becomes 350..10 in 0..360). Reject it for the same reason the
    # antimeridian case is rejected: sorting the edges below would silently resolve it
    # into the complement of the requested area.
    if west > east:
        raise ValueError(
            f"bbox {tuple(bbox)!r} crosses the seam of the mosaic's "
            f"longitude convention (it spans {west} to {east} once rewritten to "
            "match). Split it either side of the seam and merge them separately."
        )

    # Column/row offsets of the window's edges on the union grid. Divide by the signed
    # pixel size so a south-up grid (`pixel_h > 0`) maps its edges the same way round;
    # `min`/`max` then order them without assuming a north-up raster.
    cols = sorted(((west - origin_x) / pixel_w, (east - origin_x) / pixel_w))
    rows = sorted(((north - origin_y) / pixel_h, (south - origin_y) / pixel_h))

    # Snap outward, but only past a real boundary: an edge that lands on a pixel line
    # to within float noise must not add a spurious row or column, which would both
    # widen the read and shift the output's origin off the caller's request.
    col_start = int(np.floor(cols[0] + _GRID_SNAP_TOLERANCE))
    col_stop = int(np.ceil(cols[1] - _GRID_SNAP_TOLERANCE))
    row_start = int(np.floor(rows[0] + _GRID_SNAP_TOLERANCE))
    row_stop = int(np.ceil(rows[1] - _GRID_SNAP_TOLERANCE))

    # Check for a collapsed window *before* clamping. The tolerance is applied inward
    # at both edges, so a box that starts on a pixel boundary and spans less than the
    # tolerance snaps to zero width -- even sitting squarely inside the mosaic. That is
    # a different fault from a box that misses the mosaic, and reporting the latter
    # sends the caller to check their extents when the problem is the size of the box.
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError(
            f"bbox {tuple(bbox)!r} selects no whole pixel of the "
            "mosaic; it is degenerately thin in at least one axis. Widen it to at "
            "least one cell."
        )

    col_start, col_stop = max(0, col_start), min(x_size, col_stop)
    row_start, row_stop = max(0, row_start), min(y_size, row_stop)
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError(
            f"bbox {tuple(bbox)!r} does not overlap the mosaic "
            "extent; nothing would be written."
        )

    clipped = (
        origin_x + col_start * pixel_w,
        pixel_w,
        0.0,
        origin_y + row_start * pixel_h,
        0.0,
        pixel_h,
    )
    return clipped, col_stop - col_start, row_stop - row_start


def _source_bounds(
    path: str | Path | gdal.Dataset,
) -> tuple[float, float, float, float]:
    """Return a source raster's ``(west, south, east, north)`` extent.

    Used to skip sources that do not overlap a strip's latitude band during the
    tiled reduction. Accepts a path or an already-open dataset.

    Args:
        path: A source raster path/URL or an open :class:`gdal.Dataset`.

    Returns:
        tuple[float, float, float, float]: The source extent as ``(west, south,
        east, north)``.

    Raises:
        RuntimeError: The path could not be opened.
    """
    if isinstance(path, gdal.Dataset):
        ds, opened = path, False
    else:
        ds, opened = gdal.Open(str(path)), True
    if ds is None:
        raise RuntimeError(f"gdal.Open returned None for merge source {path!r}.")
    gt = ds.GetGeoTransform()
    x_far = gt[0] + gt[1] * ds.RasterXSize
    y_far = gt[3] + gt[5] * ds.RasterYSize
    bounds = (
        min(gt[0], x_far),
        min(gt[3], y_far),
        max(gt[0], x_far),
        max(gt[3], y_far),
    )
    if opened:
        # Close the handle we opened; a caller-supplied gdal.Dataset is theirs to own.
        ds = None
    return bounds


# The signer -> CloudConfig helper now lives in pyramids.base.remote
# (shared with pyramids.stac.load_asset so the rule lives in one place).
# Kept as a module-level name because call sites and tests import
# ``_cloud_config`` from here.
_cloud_config = signer_cloud_config


def merge_rasters(
    src: Sequence[str | Path],
    dst: str | Path,
    no_data_value: float | int | str = "0",
    init: float | int | str = "nan",
    n: float | int | str = "nan",
    method: str = "last",
    dst_crs: int | str | None = None,
    resampling: str = DEFAULT_RESAMPLING,
    signer: Any = None,
    *,
    bbox: Sequence[float] | None = None,
    bbox_crs: int | str | None = None,
) -> None:
    """Merge a group of rasters into one raster, resolving overlaps by ``method``.

    The overlap-resolution ``method`` selects how overlapping pixels are
    combined:

    * ``"last"`` (default) / ``"first"`` — z-order compositing: the last (or
      first) source covering a pixel wins. Implemented cheaply with
      :func:`gdal.BuildVRT` + :func:`gdal.Translate`.
    * ``"min"`` / ``"max"`` / ``"sum"`` — per-pixel reduction across every
      source overlapping that pixel, ignoring no-data. Each source is aligned
      onto the union grid and the bands are stacked and reduced with NaN-aware
      numpy.

    Args:
        src (Sequence[str | Path]):
            Paths to all input rasters.
        dst (str | Path):
            Path to the output raster. Its extension alone selects the output
            driver (`.tif` -> GTiff, `.nc` -> netCDF, …) — the same
            resolution every other write path in the package uses — so one
            `dst` yields the same format for every `method`. `COMPRESS=LZW`
            is a GTiff creation option and is applied only when the extension
            resolves to GTiff; other formats are written with their driver
            defaults. A write-by-copy-only format is refused for every
            `method` -- that is `.png`, `.jpg` / `.jpeg`, `.jp2` / `.j2k` and
            `.asc`, and `.vrt` on top (a VRT writes a reference, not a
            raster). The z-order path could produce several of them, since it
            writes via `gdal.Translate`, and the reduction path could not;
            letting `method` decide what `dst` may be is the same defect as
            letting it decide the format, so both take the stricter answer.
            `.asc` is the one this costs: it was writable before, through the
            z-order path only. Write a GTiff and convert.
        no_data_value (float | int | str):
            Stamped on the output bands as the nodata marker. For the reduction
            methods it also fills pixels with no source coverage.
        init (float | int | str):
            Reported value for pixels with no source coverage in the VRT (z-order
            methods only). Maps to :func:`gdal.BuildVRTOptions` ``VRTNodata``.
        n (float | int | str):
            Source pixels matching this value are ignored — both when building
            the VRT mosaic (z-order) and when reducing (the value is treated as
            source no-data).
        method (str):
            Overlap-resolution rule: one of ``"first"``, ``"last"`` (default),
            ``"min"``, ``"max"``, ``"sum"``.
        dst_crs (int | str | None):
            Target CRS for the mosaic, as an EPSG code (``32632``) or any
            GDAL-parseable CRS string (``"EPSG:32632"``, a WKT, a PROJ string).
            Each source whose CRS differs from the target is reprojected onto it
            (via :func:`gdal.Warp`) **before** compositing, so tiles in different
            CRSs — e.g. a Sentinel-2 AOI straddling two UTM zones — mosaic
            correctly. ``None`` (default) keeps the previous behaviour: sources
            are assumed to share a CRS and are composited as-is, *except* when
            they are found to disagree, in which case they are reprojected onto
            the first source's CRS. Reprojection always happens before the
            ``BuildVRT``/``Warp`` compositing step because that step has no
            reprojection capability and assumes a single shared grid.
        resampling (str):
            Resampling method used when a source is reprojected to ``dst_crs``
            (or to the common CRS on auto-detect). Case-insensitive; any key
            of :data:`pyramids.base._utils.INTERPOLATION_METHODS`: ``"nearest"``
            (alias ``"nearest neighbor"``, the default), ``"bilinear"``,
            ``"cubic"``, ``"cubic_spline"``, ``"lanczos"``, ``"average"``,
            ``"mode"``, ``"max"``, ``"min"``, ``"med"``, ``"q1"``, ``"q3"``,
            ``"sum"``, and ``"rms"``.
            Prefer ``"bilinear"``/``"cubic"`` for continuous data (reflectance,
            DEM) to avoid the blockiness nearest introduces across reprojection.
            Ignored when no source is reprojected.
        signer (Any):
            Optional signer exposing ``sign_href(str) -> str`` and
            ``gdal_env() -> dict[str, str]`` (e.g. a
            :class:`pyramids.stac.signers.Signer`). When given, **both** hooks
            are applied — exactly as :func:`pyramids.stac.load_asset` does:
            ``signer.sign_href`` rewrites every source path first (e.g. grafting
            a SAS token onto a blob URL), then ``signer.gdal_env()`` is installed
            via :class:`~pyramids.base.remote.CloudConfig` for the duration of
            the merge. This means URL-signing signers (Planetary Computer SAS,
            whose credential rides the href and whose ``gdal_env()`` is empty)
            and env-based signers (Requester-Pays, bearer) both authenticate
            without wrapping the call in a ``with CloudConfig(...)`` block.
            ``None`` (default) leaves source hrefs untouched and installs no
            extra config.
        bbox (Sequence[float] | None):
            Optional ``(west, south, east, north)`` window to restrict the merge
            to. ``None`` (default) merges the full extent of every source, which
            is what the function has always done.

            This is not a convenience for cropping afterwards: without it GDAL is
            given no reason to read less, so a mosaic of remote sources pulls the
            **entire** source extent through ``/vsicurl`` even when the caller
            wants a fraction of it. A full Sentinel-2 tile is 10980x10980 px
            (verified against a public Earth Search COG), so an area of interest
            covering a fraction of one tile still costs the whole tile without a
            window. How much that saves depends on the ratio between the source
            footprint and the window.

            The window is resolved once, onto the mosaic's own pixel grid, and
            used by both methods: z-order passes it to :func:`gdal.Translate` as
            ``projWin``, so only the byte ranges the window touches are requested;
            the reduction methods clip the union grid to it. Snapping outward onto
            whole pixels keeps the result a strict sub-grid, and resolving it once
            means both methods return the same grid for the same arguments.

            A bbox must be ordered ``west < east`` and ``south < north``. A window
            crossing the antimeridian (``west > east``) is rejected rather than
            silently reinterpreted as its own complement; split it and merge the
            two halves. :meth:`pyramids.dataset.Dataset.crop` handles the seam
            directly, but a mosaic is composited on one grid, which a seam-crossing
            window would not have.

            For a lon/lat mosaic the window is rewritten into the mosaic's own
            longitude convention first, so a ``-180..180`` bbox reads correctly
            against a ``0..360`` grid (and the reverse). A window that ends up
            spanning that convention's seam is rejected on the same grounds as an
            antimeridian one. A window extending past the mosaic is clipped to it;
            one that misses it entirely raises.
        bbox_crs (int | str | None):
            CRS that ``bbox`` is expressed in — an EPSG code (``4326``), an
            authority string (``"EPSG:4326"``), a WKT, or anything
            :meth:`pyproj.CRS.from_user_input` accepts. ``None`` (default) means
            ``bbox`` is already in the mosaic's own CRS — which, when ``dst_crs``
            is given, is ``dst_crs``, since sources are reprojected before the
            window is applied. Ignored when ``bbox`` is ``None``. Named to match
            :meth:`pyramids.dataset.engines.cog.COG.read_part`.

    Returns:
        None

    Note:
        The z-order methods (``"first"``/``"last"``) preserve each source's data
        type via ``BuildVRT`` + ``Translate``. The reduction methods
        (``"min"``/``"max"``/``"sum"``) align every source onto the union grid
        with ``gdal.Warp`` (nearest resampling — exact for already-aligned tiles)
        and write a single-precision-safe **Float64** output regardless of the
        source dtype, so they may differ in dtype from a z-order merge of the
        same integer inputs.

    Raises:
        TypeError: ``resampling`` is not a string, or ``bbox`` is not four numbers
            (a string, a scalar, or a sequence holding a non-numeric element).
        ValueError: ``method``/``resampling`` is not a supported value,
            ``dst_crs`` cannot be parsed as a CRS, a source carries no CRS, or
            ``bbox`` is malformed (wrong length, non-finite, inverted, zero-area),
            crosses the antimeridian or the mosaic's longitude seam once
            reprojected, selects no whole pixel, does not overlap the mosaic, or
            cannot be projected into its CRS.
        RuntimeError: GDAL failed to open a source, reproject it, or build the
            source mosaic.
        DriverNotExistError: `dst` has no extension, or one the driver catalog
            does not know.
        FileFormatNotSupportedError: `dst`'s extension maps to a
            write-by-copy-only format, for any `method`.

    Examples:
        - Mosaic two tiles, keeping the larger value wherever they overlap:
            ```python
            >>> from pyramids.dataset.merge import merge_rasters
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["tile_a.tif", "tile_b.tif"],
            ...     "mosaic_max.tif",
            ...     no_data_value=-9999.0,
            ...     method="max",
            ... )

            ```
        - Default last-wins compositing (unchanged from the previous behaviour):
            ```python
            >>> merge_rasters(["tile_a.tif", "tile_b.tif"], "mosaic.tif")  # doctest: +SKIP

            ```
        - Mosaic tiles from two UTM zones into a single CRS:
            ```python
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["utm32_tile.tif", "utm33_tile.tif"],
            ...     "mosaic_utm32.tif",
            ...     dst_crs=32632,
            ... )

            ```
        - Mosaic Requester-Pays S3 tiles by passing a signer (no ``with`` block):
            ```python
            >>> from pyramids.stac import AWSRequesterPaysSigner  # doctest: +SKIP
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["s3://bucket/a.tif", "s3://bucket/b.tif"],
            ...     "mosaic.tif",
            ...     signer=AWSRequesterPaysSigner(region="us-west-2"),
            ... )

            ```
    """
    if method not in _MERGE_METHODS:
        raise ValueError(
            f"method must be one of {list(_MERGE_METHODS)}, got {method!r}."
        )
    # Resolve `dst` here, before anything is opened. Both write paths resolve it
    # again where they need the driver name, but a destination the catalog
    # cannot answer for is a pure argument error -- and it used to be reported
    # only after every source had been opened and possibly reprojected, which
    # for /vsicurl/ inputs is network work spent to reach a typo. It also
    # reported at two different points depending on `method`.
    # Resolved here so a destination the catalog cannot answer for fails
    # before any source is opened, and so both write paths below agree. They
    # call it again where they need the name; it is a cached catalog lookup, so
    # the repeat is free and keeps each path readable on its own.
    resolve_output_driver(dst)

    # SMELL: `init` and `n` default to the string `"nan"`, which
    # round-trips through GDAL as float NaN. For integer-typed
    # rasters (e.g. UInt16) GDAL emits a warning per band:
    # `Band data type of <T> cannot represent the specified NoData
    # value of nan`. The defaults are kept for backwards-compat
    # with the previous gdal_merge.main-based signature; callers
    # that hit integer rasters should pass an explicit numeric
    # value instead of relying on the default.
    src_paths = [str(p) for p in src]
    if signer is not None:
        # Apply the signer's href rewrite to every source (e.g. graft a SAS
        # token onto a blob URL) so URL-signing signers authenticate. A no-op
        # for signers that authenticate via gdal_env() only — the base
        # sign_href returns the href unchanged. This mirrors load_asset, which
        # applies BOTH signer hooks (sign_href + gdal_env); applying only the
        # env half here would silently read URL-signed sources unauthenticated.
        src_paths = [signer.sign_href(p) for p in src_paths]

    # All GDAL reads/writes run under the signer's cloud config (a no-op when
    # signer is None) so authenticated remote sources open with the right
    # credentials for the whole merge.
    with _cloud_config(signer, path=src_paths):
        # Put every source on one CRS before compositing. The BuildVRT/Warp
        # mosaic below cannot reproject — it stitches pixel grids assuming a
        # shared CRS — so mismatched sources must be warped first or they would
        # mis-align silently. `_keepalive` holds the in-memory warped VRTs so
        # GDAL does not free them while the mosaic is built.
        sources, _keepalive = _prepare_sources(src_paths, dst_crs, resampling)

        if method in _REDUCE_METHODS:
            _merge_reduce(sources, str(dst), method, no_data_value, n, bbox, bbox_crs)
            return

        # z-order: "last" keeps natural order (last source wins); "first"
        # reverses so the original first source is placed last in the VRT and
        # therefore wins.
        ordered = list(reversed(sources)) if method == "first" else sources
        vrt_opts = gdal.BuildVRTOptions(
            srcNodata=str(n),
            VRTNodata=str(init),
        )
        vrt_ds = gdal.BuildVRT("", ordered, options=vrt_opts)
        if vrt_ds is None:
            raise RuntimeError(
                f"gdal.BuildVRT returned None for sources {src_paths!r}; "
                "check that all paths are readable rasters with consistent "
                "band counts and CRS."
            )
        proj_win = None
        if bbox is not None:
            # Resolve the window here, in the mosaic's own CRS, and hand Translate a
            # `projWin` that already lies on the mosaic's pixel boundaries -- rather
            # than passing the caller's bbox with `projWinSRS` and letting GDAL
            # reproject it. Two independent reprojections of the same window round to
            # opposite sides of a pixel edge, and the two merge paths then return
            # different rasters for identical arguments (measured 3x3 at x=111364.8
            # against 4x3 at x=0.0 for `dst_crs=3857`). One resolver for both paths
            # makes them agree by construction, and `_restrict_grid` also rejects a
            # disjoint window, which GDAL does not: it writes a 1x1 no-data raster at
            # the window's origin, which reads back as a successful merge of nothing.
            clipped, window_x, window_y = _restrict_grid(
                vrt_ds.GetGeoTransform(),
                vrt_ds.RasterXSize,
                vrt_ds.RasterYSize,
                vrt_ds.GetProjection(),
                bbox,
                bbox_crs,
            )
            # (ulx, uly, lrx, lry) — already snapped, so GDAL's own rounding is a
            # no-op and the output grid matches the reduction path's exactly.
            proj_win = [
                clipped[0],
                clipped[3],
                clipped[0] + window_x * clipped[1],
                clipped[3] + window_y * clipped[5],
            ]

        # `projWin` is what stops the read at the window: without it GDAL has no
        # reason to restrict what it pulls through /vsicurl and materialises the
        # whole mosaic extent.
        # Hand gdal.Translate the driver the catalog resolved rather than
        # letting it re-infer from the extension. The two tables disagree: the
        # catalog knows `.nc4` as a netCDF alias, GDAL's netCDF driver
        # advertises only `nc`, so the same `dst` wrote a netCDF through the
        # reduction path and died with "Could not identify an output driver"
        # here -- the format depending on `method` again, which is exactly what
        # resolving once was meant to stop.
        # Strict gate, not `for_copy`: this method has two write paths and the
        # other builds with `Create`. Relaxing only this one would make `.png`
        # legal for method="last" and illegal for method="min" -- the very
        # asymmetry the shared resolution above exists to remove.
        out_driver = resolve_output_driver(dst)
        # LZW is a GTiff creation option; other drivers reject it.
        translate_opts = gdal.TranslateOptions(
            format=out_driver,
            creationOptions=["COMPRESS=LZW"] if out_driver == "GTiff" else [],
            noData=str(no_data_value),
            projWin=proj_win,
        )
        out_ds = gdal.Translate(str(dst), vrt_ds, options=translate_opts)
        if out_ds is None:
            raise RuntimeError(
                f"gdal.Translate returned None writing the mosaic to {str(dst)!r}."
            )
        out_ds.FlushCache()
        out_ds = None
        vrt_ds = None


def _as_srs(crs: int | str) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from an EPSG code or a CRS string.

    Args:
        crs: An EPSG code as an ``int`` (e.g. ``32632``), or any CRS string
            GDAL can parse via ``SetFromUserInput`` — ``"EPSG:32632"``, a WKT
            string, or a PROJ string.

    Returns:
        osr.SpatialReference: The parsed spatial reference.

    Raises:
        ValueError: ``crs`` could not be parsed as a CRS.
    """
    srs = osr.SpatialReference()
    try:
        # GDAL may signal a parse failure either by a non-zero return code or,
        # when exceptions are enabled (pyramids enables them at import), by
        # raising RuntimeError. Treat both as a bad CRS.
        if isinstance(crs, int):
            failed = srs.ImportFromEPSG(crs) != 0
        else:
            failed = srs.SetFromUserInput(str(crs)) != 0
    except RuntimeError as exc:
        raise ValueError(f"Could not parse dst_crs={crs!r} as a CRS.") from exc
    if failed:
        raise ValueError(f"Could not parse dst_crs={crs!r} as a CRS.")
    return srs


def _prepare_sources(
    src_paths: list[str],
    dst_crs: int | str | None,
    resampling: str = DEFAULT_RESAMPLING,
) -> tuple[list, list]:
    """Put every source on a single CRS before the mosaic is composited.

    The compositing step (:func:`gdal.BuildVRT` / :func:`gdal.Warp`) does not
    reproject — it assumes all sources share a CRS. This helper warps any source
    whose CRS differs from the target onto an in-memory warped VRT first, so
    tiles from different CRSs (e.g. neighbouring UTM zones) mosaic correctly.

    Each source is opened exactly once; the open handle is reused both to read
    the source CRS and (when no reproject is needed) as the compositor input, so
    no path is opened twice — relevant under Requester-Pays where each open is
    billable.

    Args:
        src_paths: Source raster paths.
        dst_crs: Target CRS as an EPSG code or CRS string. ``None`` keeps the
            previous behaviour — sources are returned untouched when they all
            share a CRS, and only reprojected (onto the first source's CRS) when
            they are found to disagree.
        resampling: Resampling method used when reprojecting a mismatched-CRS
            source — any key of
            :data:`pyramids.base._utils.INTERPOLATION_METHODS`,
            case-insensitive (e.g. ``"nearest neighbor"`` (default),
            ``"bilinear"``, ``"average"``). Unused when no source is
            reprojected.

    Returns:
        tuple[list, list]: ``(sources, keepalive)``. ``sources`` is the
            per-source input to feed the compositor — open
            :class:`gdal.Dataset` handles (warped VRTs for reprojected sources,
            plain opens otherwise). ``keepalive`` holds the same datasets so the
            caller keeps them referenced (and prevents GDAL from freeing them)
            until the mosaic is built.

    Raises:
        TypeError: ``resampling`` is not a string.
        ValueError: ``dst_crs`` (or ``resampling``) could not be parsed, or a
            source carries no CRS.
        RuntimeError: A source could not be opened, or a reprojecting
            :func:`gdal.Warp` failed.
    """
    resample_alg = resolve_resampling(resampling)

    # Open each source once; read its CRS from that same handle.
    opened: list = []
    source_srs: list[osr.SpatialReference] = []
    for path in src_paths:
        dataset = gdal.Open(path)
        if dataset is None:
            raise RuntimeError(f"gdal.Open returned None for source {path!r}.")
        wkt = dataset.GetProjection()
        if not wkt:
            raise ValueError(
                f"source {path!r} has no CRS; every source must carry a CRS to "
                "be merged/reprojected."
            )
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        opened.append(dataset)
        source_srs.append(srs)

    target_srs = _as_srs(dst_crs) if dst_crs is not None else None
    if target_srs is None:
        # Auto-detect: no reproject when every source already shares a CRS.
        disagree = any(not source_srs[0].IsSame(other) for other in source_srs[1:])
        if not disagree:
            return opened, opened
        target_srs = source_srs[0]

    # At least one source needs reprojecting (or dst_crs forces a target). Feed
    # the compositor open datasets uniformly — gdal.BuildVRT rejects a mix of
    # path strings and dataset objects.
    target_wkt = target_srs.ExportToWkt()
    sources: list = []
    for dataset, srs in zip(opened, source_srs):
        if srs.IsSame(target_srs):
            sources.append(dataset)
            continue
        warped = gdal.Warp(
            "",
            dataset,
            options=gdal.WarpOptions(
                format="VRT", dstSRS=target_wkt, resampleAlg=resample_alg
            ),
        )
        if warped is None:
            raise RuntimeError(
                "gdal.Warp returned None reprojecting a source to the target CRS."
            )
        sources.append(warped)
    return sources, sources


def _source_misses_strip(
    bounds: tuple[float, float, float, float],
    strip_lat: tuple[float, float],
    strip_bounds: Sequence[float],
) -> bool:
    """Return whether a source's extent lies entirely outside this strip.

    Warping such a source would only add all-no-data, leaving the accumulator and
    coverage mask unchanged. Both axes are tested: a strip spans the *windowed*
    grid's width, so on a wide east-west mosaic a latitude-only test still warped
    every source sharing the strip's band, which is the cost a window exists to
    avoid.

    Args:
        bounds: The source's ``(west, south, east, north)`` extent.
        strip_lat: The strip's ``(south, north)`` edges.
        strip_bounds: The strip's ``(west, south, east, north)`` extent.

    Returns:
        bool: `True` when the source cannot contribute to this strip.
    """
    src_west, src_south, src_east, src_north = bounds
    strip_south, strip_north = strip_lat
    strip_west, strip_east = strip_bounds[0], strip_bounds[2]
    outside_lat = src_north <= strip_south or src_south >= strip_north
    outside_lon = src_east <= strip_west or src_west >= strip_east
    return outside_lat or outside_lon


def _warp_onto_strip(
    path: Any,
    strip_bounds: Sequence[float],
    x_size: int,
    ysize: int,
    src_nodata: float | int | str | None,
) -> np.ndarray:
    """Warp one source onto a strip's window and read it as a 3-D float64 cube.

    Args:
        path: Source path or an already-open :class:`gdal.Dataset`.
        strip_bounds: The strip's ``(west, south, east, north)`` window.
        x_size: Strip width in pixels.
        ysize: Strip height in pixels.
        src_nodata: The sources' no-data value, or `None`.

    Returns:
        np.ndarray: The warped strip, always ``(bands, rows, cols)``, no-data as NaN.

    Raises:
        RuntimeError: :func:`gdal.Warp` failed for this source.
    """
    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=strip_bounds,
        width=x_size,
        height=ysize,
        srcNodata=src_nodata,
        dstNodata=float("nan"),
    )
    warped = gdal.Warp("", path, options=warp_opts)
    if warped is None:
        raise RuntimeError(
            f"gdal.Warp returned None warping source {path!r} onto the union grid."
        )
    # np.asarray pins the type: GDAL's ReadAsArray is untyped, so without it the
    # float64 cube is inferred as Any and leaks out of the annotated return.
    array = np.asarray(warped.ReadAsArray()).astype("float64")
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    return array


def _fold_into(
    acc: np.ndarray, array: np.ndarray, valid: np.ndarray, method: str
) -> None:
    """Fold one warped source into the strip accumulator, in place.

    Args:
        acc: The strip accumulator, modified in place.
        array: The warped source strip, no-data as NaN.
        valid: Mask of the finite cells in `array`.
        method: One of ``"min"``, ``"max"``, ``"sum"``.

    Returns:
        None
    """
    if method == "min":
        np.fmin(acc, array, out=acc)  # fmin/fmax ignore NaN
    elif method == "max":
        np.fmax(acc, array, out=acc)
    else:
        np.add(acc, array, out=acc, where=valid)


def _reduce_strip(
    src_paths: list,
    src_bounds: list[tuple[float, float, float, float]],
    strip_bounds: list[float],
    strip_lat: tuple[float, float],
    shape: tuple[int, int, int],
    method: str,
    src_nodata: float | None,
    fill: float,
) -> np.ndarray:
    """Reduce one union-grid strip across every overlapping source.

    Warps each source that overlaps the strip's latitude band onto the strip window
    and folds it into a NaN-aware accumulator, then fills no-coverage cells. Extracted
    from :func:`_merge_reduce` to keep that function's nesting (cognitive complexity)
    low.

    Args:
        src_paths: Source rasters (paths or open datasets).
        src_bounds: Each source's ``(west, south, east, north)`` extent.
        strip_bounds: The strip's ``[west, south, east, north]`` output bounds.
        strip_lat: The strip's ``(south, north)`` latitude band for the overlap prune.
        shape: The strip cube shape ``(band_count, rows, cols)``.
        method: One of ``"min"``, ``"max"``, ``"sum"``.
        src_nodata: Source pixel value to treat as no-data, or ``None``.
        fill: Value written where no source covers a pixel.

    Returns:
        np.ndarray: The reduced strip, shape ``shape``.

    Raises:
        RuntimeError: GDAL failed to warp a source onto the strip.
    """
    _, ysize, x_size = shape
    acc = np.full(shape, _REDUCE_IDENTITY[method], dtype="float64")
    # A boolean "has any valid source" mask suffices: min/max/sum never divide by a
    # count, only test presence below, so a bool cube (1 byte/px) replaces int64.
    covered = np.zeros(shape, dtype=bool)

    for path, bounds in zip(src_paths, src_bounds):
        if _source_misses_strip(bounds, strip_lat, strip_bounds):
            continue
        array = _warp_onto_strip(path, strip_bounds, x_size, ysize, src_nodata)
        valid = ~np.isnan(array)
        covered |= valid
        _fold_into(acc, array, valid, method)
        del array, valid

    # No-coverage cells are still +inf/-inf/0 in acc; replace them with the fill.
    return np.where(covered, acc, fill)


def _merge_reduce(
    src_paths: list,
    dst: str,
    method: str,
    no_data_value: float | int | str,
    n: float | int | str,
    bbox: Sequence[float] | None = None,
    bbox_crs: int | str | None = None,
) -> None:
    """Merge sources by reducing overlapping pixels with min/max/sum.

    The union grid (from a scratch :func:`gdal.BuildVRT`) is reduced one full-width
    row strip at a time: each source is warped onto that strip's window and folded
    into the strip accumulator with a NaN-aware reduction, then the reduced strip is
    written to the output. Peak memory is therefore ``O(strip)`` rather than
    ``O(grid)`` — a very large mosaic is merged without materialising the whole
    output. Sources whose extent does not overlap a strip's latitude band are
    skipped (they would warp to all-no-data). Nearest-neighbour warping onto the
    exact union grid makes the strip reduction byte-identical to a whole-grid pass.
    Pixels with no source coverage are written as ``no_data_value``.

    Args:
        src_paths: Source rasters as path strings or already-open
            :class:`gdal.Dataset` objects (e.g. reprojected warped VRTs from
            :func:`_prepare_sources`).
        dst: Output raster path.
        method: One of ``"min"``, ``"max"``, ``"sum"``.
        no_data_value: Output no-data value and no-coverage fill.
        n: Source pixel value to treat as no-data (``"nan"`` means none).
        bbox: Optional ``(west, south, east, north)`` window. When given, the union
            grid is clipped to it before the output is created, so only the window
            is allocated and read — clipping the strip loop alone would not help,
            because the output is sized from the union before the loop runs.
        bbox_crs: CRS of `bbox`, or ``None`` when it is already in the union grid's
            CRS (which is `dst_crs` when the caller passed one).

    Raises:
        RuntimeError: GDAL failed to build the union mosaic or to warp a source.
    """
    template = gdal.BuildVRT("", src_paths)
    if template is None:
        raise RuntimeError(
            f"gdal.BuildVRT returned None for sources {src_paths!r}; "
            "check that all paths are readable rasters with consistent "
            "band counts and CRS."
        )
    geotransform = template.GetGeoTransform()
    projection = template.GetProjection()
    x_size, y_size = template.RasterXSize, template.RasterYSize
    band_count = template.RasterCount
    template = None

    # Clip the grid itself, not just the strip loop: the output is created at
    # `x_size`/`y_size` below and the loop walks every row of it, so restricting
    # only the loop would still allocate -- and read -- the full union extent.
    if bbox is not None:
        geotransform, x_size, y_size = _restrict_grid(
            geotransform, x_size, y_size, projection, bbox, bbox_crs
        )

    src_nodata = None if str(n).lower() == "nan" else float(n)
    fill = float(no_data_value)
    # Extent of every source, computed once, to skip sources a strip cannot touch.
    src_bounds = [_source_bounds(path) for path in src_paths]

    # Resolve from the extension so that one `dst` does not yield two different
    # formats depending on an unrelated argument: this reduction path hardcoded
    # GTiff while the z-order path below lets gdal.Translate infer, so `.nc`
    # produced a netCDF for method="last" and a GTiff for method="min". LZW is
    # GTiff-specific and is applied only there.
    out_driver = resolve_output_driver(dst)
    out_options = ["COMPRESS=LZW"] if out_driver == "GTiff" else []
    out_ds = gdal.GetDriverByName(out_driver).Create(
        dst, x_size, y_size, band_count, gdal.GDT_Float64, options=out_options
    )
    if out_ds is None:
        raise RuntimeError(
            f"gdal.Create returned None writing the reduced mosaic to {dst!r}; "
            f"check the output path is writable ({gdal.GetLastErrorMsg()!r})."
        )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    for band_index in range(band_count):
        out_ds.GetRasterBand(band_index + 1).SetNoDataValue(fill)

    for yoff in range(0, y_size, _MERGE_STRIP_ROWS):
        ysize = min(_MERGE_STRIP_ROWS, y_size - yoff)
        strip_north = geotransform[3] + geotransform[5] * yoff
        strip_south = geotransform[3] + geotransform[5] * (yoff + ysize)
        strip_bounds = [
            geotransform[0],
            strip_south,
            geotransform[0] + geotransform[1] * x_size,
            strip_north,
        ]
        reduced = _reduce_strip(
            src_paths,
            src_bounds,
            strip_bounds,
            (strip_south, strip_north),
            (band_count, ysize, x_size),
            method,
            src_nodata,
            fill,
        )
        for band_index in range(band_count):
            out_ds.GetRasterBand(band_index + 1).WriteArray(
                reduced[band_index], 0, yoff
            )
        del reduced

    out_ds.FlushCache()
    out_ds = None


def stack_bands(
    files: list[str | Path],
    *,
    band_names: list[str] | None = None,
    align: bool = False,
    no_data_value: Any = _INHERIT_NO_DATA,
    path: str | Path | None = None,
    signer: Any = None,
) -> Dataset:
    """Stack N single-band rasters into one multi-band :class:`Dataset`.

    Free-function alias for :meth:`pyramids.dataset.Dataset.from_band_files`
    — see that method for the full contract, edge cases, and examples.

    Args:
        files: Single-band raster paths/URLs to stack (order = band order).
        band_names: Explicit per-band names; ``None`` derives them from the
            file names.
        align: When ``True``, resample mismatched inputs onto ``files[0]``'s
            grid instead of raising :class:`~pyramids.base._errors.AlignmentError`.
        no_data_value: No-data value for the output bands; omitted means
            "inherit from the source rasters".
        path: Output path, whose extension selects the driver (``.tif`` ->
            GTiff, ``.nc`` -> netCDF, …); ``None`` keeps the result in memory.
            `COMPRESS=LZW` is applied only when the extension resolves to
            GTiff. A write-by-copy-only format such as PNG is refused — see
            :meth:`pyramids.dataset.Dataset.from_band_files` for why both of
            its write paths answer alike.
        signer: Optional signer exposing ``sign_href(str) -> str`` and
            ``gdal_env() -> dict[str, str]`` (e.g. a
            :class:`pyramids.stac.signers.Signer`). When given, **both** hooks
            are applied (as in :func:`pyramids.stac.load_asset`): every input
            href is rewritten through ``signer.sign_href`` first, then
            ``signer.gdal_env()`` is installed via
            :class:`~pyramids.base.remote.CloudConfig` for the duration of the
            stack, so authenticated cloud inputs (URL-signed or env-credentialed)
            read with the right credentials. ``None`` (default) leaves behaviour
            unchanged.

    Returns:
        Dataset: A multi-band dataset, one band per input file.

    Raises:
        DriverNotExistError: `path` has no extension, or one the driver
            catalog does not know.
        FileFormatNotSupportedError: `path`'s extension maps to a
            write-by-copy-only format, whichever write path the inputs take.
    """
    if signer is not None:
        files = [signer.sign_href(str(f)) for f in files]
    with _cloud_config(signer, path=[str(f) for f in files]):
        result = Dataset.from_band_files(
            files,
            band_names=band_names,
            align=align,
            no_data_value=no_data_value,
            path=path,
        )
    return result
