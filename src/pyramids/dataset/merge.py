"""Free-function entry points for merging a list of rasters.

* :func:`merge_rasters` — mosaic rasters that tile the *same area* into one
  raster (spatial merge).
* :func:`stack_bands` — stack N single-band rasters that cover the *same
  grid* into one multi-band raster (band-wise merge). Thin alias for
  :meth:`pyramids.dataset.Dataset.from_band_files`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, osr
from pyproj.exceptions import ProjError

from pyramids.base._coverage import open_network_dataset, run_gdal_op
from pyramids.base._domain import (
    INHERIT_NO_DATA,
    fits_dtype,
    free_no_data,
    inherit_no_data,
    no_data_candidates,
)
from pyramids.base._utils import (
    DEFAULT_RESAMPLING,
    gdal_to_numpy_type,
    resolve_resampling,
)
from pyramids.base.remote import redact_credentials, signer_cloud_config
from pyramids.dataset._driver import resolve_output_driver
from pyramids.dataset.dataset import Dataset
from pyramids.dataset.transform import GeoTransform
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


@dataclass(frozen=True)
class _Source:
    """A merge source paired with the name to report it by when GDAL fails on it.

    ``_prepare_sources`` hands the reduction path open :class:`gdal.Dataset`
    handles, whose ``repr`` is a SWIG proxy address. Naming the source is the
    whole point of #1107, so the label travels with the handle instead of being
    recovered from it.

    Attributes:
        label: Display name, already quoted/positioned by the caller, e.g.
            ``"1/2 'tile.tif'"``.
        handle: The path string or open :class:`gdal.Dataset` GDAL is given.
    """

    label: str
    handle: Any

    @classmethod
    def of(cls, source: Any) -> _Source:
        """Return `source` as a `_Source`, labelling a bare path by its repr."""
        return source if isinstance(source, cls) else cls(f"{source!r}", source)


_READABLE_RASTERS_HINT = (
    "check that all paths are readable rasters with consistent band counts and CRS"
)


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
        RuntimeError: The path could not be opened -- the message names the
            source and chains GDAL's own error, which for a ``/vsicurl/`` or
            ``/vsis3/`` source carries only the HTTP status and no URL.
    """
    if isinstance(path, gdal.Dataset):
        ds, opened = path, False
    else:
        source = str(path)
        ds = open_network_dataset(
            source, error=RuntimeError, subject=f"merge source {source!r}"
        )
        opened = True
    bounds = GeoTransform(*ds.GetGeoTransform()).extent(ds.RasterXSize, ds.RasterYSize)
    if opened:
        # Close the handle we opened; a caller-supplied gdal.Dataset is theirs to own.
        ds = None
    return bounds


# The signer -> CloudConfig helper now lives in pyramids.base.remote
# (shared with pyramids.stac.load_asset so the rule lives in one place).
# Kept as a module-level name because call sites and tests import
# ``_cloud_config`` from here.
_cloud_config = signer_cloud_config


def _is_nan_spelling(value: float | int | str) -> bool:
    """Whether `value` is one of the ways this module's defaults spell NaN.

    `n` and `init` both default to the string `"nan"`, and both read that as "the
    caller said nothing" rather than as a value. Matching on the spelling is what
    makes `"NaN"` and `float("nan")` mean it too.

    Args:
        value: The `n` or `init` a caller passed, or the default.

    Returns:
        bool: `True` when the value is a NaN in any of those spellings.

    Examples:
        - Every spelling of the default answers alike:
            ```python
            >>> from pyramids.dataset.merge import _is_nan_spelling
            >>> [_is_nan_spelling(v) for v in ("nan", "NaN", float("nan"))]
            [True, True, True]

            ```
        - A real value is not one of them:
            ```python
            >>> from pyramids.dataset.merge import _is_nan_spelling
            >>> _is_nan_spelling(0)
            False

            ```
    """
    return str(value).lower() == "nan"


# The float the module's `"nan"` defaults resolve to, named once.
NAN = float("nan")

# GDAL's own words for "this band holds no data to measure", which is the one
# failure `_mosaic_value_range` treats as an answer rather than an error.
_NO_VALID_PIXELS = "no valid pixels found"

_MAX_TILED_SOURCES = 512


def _sources_tile_their_union(bounds: list[tuple[float, float, float, float]]) -> bool:
    """Whether the sources cover every point of the area they span, leaving no gap.

    Answered from the footprints alone, without reading a pixel -- which is the
    whole point, since it decides whether reading is necessary. A mosaic with no
    uncovered pixel has nothing for a marker to mark, and proving a marker unused
    costs a full survey of every source.

    The rectangles are compared on the grid their own edges define: every source
    edge becomes a cut line, which turns "is every point covered" into "is every
    cell of a small boolean grid covered". That is exact for the axis-aligned
    footprints :func:`gdal.BuildVRT` composites, and float noise on a shared edge
    can only answer `False` -- a sliver cell nothing covers -- which costs a
    survey rather than skipping one that was needed.

    Args:
        bounds: Each source's ``(west, south, east, north)`` extent.

    Returns:
        bool: `True` when the footprints leave no gap between them; `False` when
        they do, when there are none, or when there are more than
        `_MAX_TILED_SOURCES` of them -- past which the cut grid is no longer
        small, and the cautious answer is the cheap one.

    Examples:
        - Two tiles sharing an edge cover the strip they span:
            ```python
            >>> from pyramids.dataset.merge import _sources_tile_their_union
            >>> _sources_tile_their_union([(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)])
            True

            ```
        - Two that do not leave the column between them uncovered:
            ```python
            >>> from pyramids.dataset.merge import _sources_tile_their_union
            >>> _sources_tile_their_union([(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 5.0, 2.0)])
            False

            ```
    """
    tiled = False
    if bounds and len(bounds) <= _MAX_TILED_SOURCES:
        xs = np.unique(
            np.asarray([(box[0], box[2]) for box in bounds], dtype="float64")
        )
        ys = np.unique(
            np.asarray([(box[1], box[3]) for box in bounds], dtype="float64")
        )
        cells = np.zeros((ys.size - 1, xs.size - 1), dtype=bool)
        for west, south, east, north in bounds:
            columns = np.searchsorted(xs, (west, east))
            rows = np.searchsorted(ys, (south, north))
            cells[rows[0] : rows[1], columns[0] : columns[1]] = True
        tiled = bool(cells.all())
    return tiled


def _source_nodata(n: float | int | str) -> float | None:
    """The value to treat as source no-data, or `None` to use each source's own.

    The default `n="nan"` does not mean "ignore NaN cells" -- it has to mean "no
    override", because handing GDAL a blanket value *replaces* every source's own
    declaration with it. A mosaic of tiles declaring -9999 and -32768 then
    composited both holes as real measurements, and only the winner's were masked
    afterwards by the inherited marker. Left unset, each source's declared value
    marks that source's own holes, which is what makes the two markers agree.

    The test is on the *spelling*, so `"nan"`, `"NaN"` and `float("nan")` all
    mean "no override"; anything else is coerced to a float, since that is what
    both consumers -- :func:`gdal.BuildVRTOptions`' ``srcNodata`` and
    :func:`gdal.WarpOptions`' ``srcNodata`` -- take.

    Args:
        n: The caller's source-no-data value; `"nan"` (the default) means none.

    Returns:
        float | None: The override, or `None` to leave each source with its own.

    Raises:
        ValueError: `n` is a string that names no number.

    Examples:
        - The default asks for no override, so each source keeps its own:
            ```python
            >>> from pyramids.dataset.merge import _source_nodata
            >>> print(_source_nodata("nan"))
            None

            ```
        - Any other value is coerced to the float GDAL wants:
            ```python
            >>> from pyramids.dataset.merge import _source_nodata
            >>> _source_nodata(-9999)
            -9999.0

            ```
    """
    return None if _is_nan_spelling(n) else float(n)


def _mosaic_value_range(mosaic: gdal.Dataset) -> tuple[float, float] | None:
    """The smallest and largest values the mosaic's cells hold, across every band.

    Asked before a sentinel is chosen, so that a value the data already uses is
    never stamped on it. GDAL walks the mosaic block by block, so this costs a
    read of the sources but never a copy of them in memory -- and it answers for
    the whole mosaic what would otherwise take a full array per candidate.

    The range covers the cells GDAL counts as data: `ComputeRasterMinMax` skips
    any that hold the band's own declared no-data. That is the right answer for
    the caller -- a cell already marked absent is not one a new sentinel could
    collide with -- but it does mean the range is not the raw span of the bytes
    on disk.

    Args:
        mosaic: The composited VRT.

    Returns:
        tuple[float, float] | None: ``(minimum, maximum)`` over every band, or
        `None` when no band holds a value at all -- a mosaic that is entirely
        no-data rules out no sentinel.
    """
    low: float | None = None
    high: float | None = None
    for index in range(mosaic.RasterCount):
        try:
            # `False` asks for the exact range rather than an approximation. The
            # approximation is not merely less precise, it is unusable here: it
            # reads a decimated sample, and a sentinel has to be proven absent
            # from every cell, not from a sample. Measured on a VRT over two
            # tiles carrying overviews -- which a VRT does inherit, so there is
            # no "no overviews anyway" to fall back on -- the approximate range
            # was (0.0, 163.0) where the exact one was (0.0, 255.0), and a
            # sentinel of 255 chosen from it would have masked real cells.
            band_low, band_high = mosaic.GetRasterBand(index + 1).ComputeRasterMinMax(
                False
            )
        except RuntimeError as exc:
            # A band holding nothing but no-data has no minimum, and GDAL says so
            # by raising (exceptions are enabled package-wide). Such a band
            # constrains no candidate, so it is skipped. Every other RuntimeError
            # from this call is a failed read -- a truncated remote object, an
            # expired credential mid-scan -- and skipping those would choose a
            # marker from whatever bands happened to answer, or from none at all.
            if _NO_VALID_PIXELS not in str(exc):
                raise
            continue
        low = band_low if low is None else min(low, band_low)
        high = band_high if high is None else max(high, band_high)
    return None if low is None or high is None else (low, high)


def _unused_marker(
    mosaic: gdal.Dataset, dtype: np.dtype, preferred: Sequence[Any] = ()
) -> Any | None:
    """A sentinel `dtype` can store that the mosaic's own cells do not already use.

    The candidates, and the order they are tried in, come from
    :func:`~pyramids.base._domain.no_data_candidates`: the caller's own
    `preferred` values first, then the package default ``-9999`` where the dtype
    can hold it, then the dtype's own extremes, an unsigned band offering its
    maximum before its `0`. Asking in that order here and in
    :func:`~pyramids.base._domain.free_no_data` is what makes a mosaic's marker
    the same one a single raster would have been given.

    A preferred value is a preference, not an instruction: it is range-tested
    like every other candidate, and passed over when the data uses it. That is
    the whole guarantee -- a sentinel is one the data provably does not contain
    -- and a value taken on trust would reintroduce the defect this exists to
    fix.

    Two questions are asked of the data, cheapest first: whether a candidate
    falls outside the mosaic's own minimum and maximum (one streamed pass, and
    enough in the ordinary case), and only if none does, whether it occurs
    anywhere in the mosaic's cells (a full read).

    Args:
        mosaic: The composited VRT.
        dtype: The dtype the sentinel must be storable in.
        preferred: Values to try before the package's own, most preferred first.
            One the dtype cannot store is dropped rather than refused.

    Returns:
        Any | None: The chosen sentinel, or `None` when the data uses every value
        the dtype could spare. A mosaic that is entirely no-data, whose range is
        therefore unanswerable, takes the first candidate, and `None` when the
        dtype offers none.

    Raises:
        RuntimeError: GDAL failed to read the mosaic while resolving the range.

    Examples:
        - A signed band whose cells stay small takes the package default:
            ```python
            >>> import numpy as np
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _unused_marker
            >>> mosaic = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int16)
            >>> _ = mosaic.GetRasterBand(1).WriteArray(np.array([[1, 2], [3, 4]], "int16"))
            >>> _unused_marker(mosaic, np.dtype("int16"))
            -9999

            ```
        - A preferred value the data does not hold wins, and one it does hold is
          passed over -- which is what keeps a sentinel from masking real cells:
            ```python
            >>> import numpy as np
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _unused_marker
            >>> mosaic = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int16)
            >>> _ = mosaic.GetRasterBand(1).WriteArray(np.array([[0, 2], [3, 4]], "int16"))
            >>> _unused_marker(mosaic, np.dtype("int16"), preferred=(7,))
            7
            >>> _unused_marker(mosaic, np.dtype("int16"), preferred=(0,))
            -9999

            ```
    """
    candidates = no_data_candidates(dtype, preferred)
    bounds = _mosaic_value_range(mosaic)
    if bounds is None:
        chosen = candidates[0] if candidates else None
    else:
        low, high = bounds
        # The range test alone clears a candidate in the ordinary case: a sentinel
        # outside the data's own extremes cannot occur in it, and asking that way
        # costs one streamed pass instead of a full array per candidate.
        chosen = next((value for value in candidates if not low <= value <= high), None)
        if chosen is None:
            # Every candidate lies inside the data's range, so the cheap test
            # cannot clear one and the mosaic has to be read. It takes data
            # spanning the dtype's own extremes to reach here, and `free_no_data`
            # then enumerates a narrow dtype's whole range exactly.
            chosen = free_no_data(dtype, preferred, np.asarray(mosaic.ReadAsArray()))
    return chosen


def _requested_no_data(no_data_value: Any) -> Any:
    """Read the caller's `no_data_value`, refusing one that would vanish silently.

    Everything that is not the sentinel or `None` is handed to
    :func:`gdal.Translate` as ``-a_nodata``, which answers an unparsable value
    with "Nodata value was not set to output band" and writes no marker at all --
    an unmarked mosaic from a typo, which is the defect this module exists to
    close. `"inherit"` is the likeliest such typo, since the signature's default
    renders as exactly that word, so it is accepted as the sentinel it names
    rather than refused.

    Args:
        no_data_value: Whatever the caller passed, including the default sentinel.

    Returns:
        Any: The sentinel, `None`, or the value unchanged.

    Raises:
        ValueError: The value is neither the sentinel, nor `None`, nor a number.

    Examples:
        - The word the default renders as means the default:
            ```python
            >>> from pyramids.base._domain import INHERIT_NO_DATA
            >>> from pyramids.dataset.merge import _requested_no_data
            >>> _requested_no_data("inherit") is INHERIT_NO_DATA
            True

            ```
        - Anything else that is not a number is refused rather than dropped:
            ```python
            >>> from pyramids.dataset.merge import _requested_no_data
            >>> _requested_no_data("nodata")
            Traceback (most recent call last):
                ...
            ValueError: no_data_value='nodata' is not a number...

            ```
    """
    requested = no_data_value
    if isinstance(no_data_value, str) and no_data_value.strip().lower() == "inherit":
        requested = INHERIT_NO_DATA
    elif no_data_value is not INHERIT_NO_DATA and no_data_value is not None:
        try:
            float(no_data_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"no_data_value={no_data_value!r} is not a number. Pass a value "
                "the output band can hold, None for no marker at all, or omit it "
                "(or pass 'inherit') to take the marker from the sources."
            ) from exc
    return requested


def _validated_init(init: float | int | str) -> float | int | str:
    """Refuse an `init` that names no number, before GDAL is asked to composite with it.

    `init` becomes ``VRTNodata`` whenever no settled marker replaces it, and GDAL
    answers an unparsable one by failing the whole mosaic ("Invalid -vrtnodata
    value"). Whether that happens depends on facts the caller cannot see -- the
    output's data type, and whether the sources leave a gap -- so the same
    argument works or fails depending on the data. Refusing it here makes the
    answer the same either way, and names the argument at fault.

    Args:
        init: The caller's uncovered-pixel value.

    Returns:
        float | int | str: The value unchanged.

    Raises:
        ValueError: `init` names no number and is not a spelling of NaN.

    Examples:
        - The default, and any number, pass through unchanged:
            ```python
            >>> from pyramids.dataset.merge import _validated_init
            >>> _validated_init("nan"), _validated_init(-9999)
            ('nan', -9999)

            ```
        - Anything else is refused here rather than by GDAL later:
            ```python
            >>> from pyramids.dataset.merge import _validated_init
            >>> _validated_init("none")
            Traceback (most recent call last):
                ...
            ValueError: init='none' is not a number...

            ```
    """
    if not _is_nan_spelling(init):
        try:
            float(init)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"init={init!r} is not a number. It fills the pixels no source "
                "covers, so it has to be one the output band can hold -- or "
                "'nan', the default, to leave that to the marker."
            ) from exc
    return init


def _explicit_fill(ordered: list, init: float | int | str, marker: Any) -> str | None:
    """The gap fill for a marker the caller chose, or `None` to leave `init` alone.

    A marker only masks the pixels that hold it, and the pixels no source covers
    hold `init` -- so a caller who passes `no_data_value` and leaves `init` at its
    default gets a band declaring their value over gaps holding something else.
    Measured on Int32 tiles, `no_data_value=-1` wrote gaps of `0` under a declared
    `-1`, masking nothing; on float tiles it wrote `NaN` under a declared `-1`,
    which `read_array(masked=True)` also leaves unmasked and `stats()` propagates.
    The reduction methods already fill with the marker, so this is also what makes
    the two write paths answer alike.

    An `init` the caller passed themselves is left alone: they named the value
    their uncovered pixels should hold, and it is not this function's place to
    overrule them.

    Args:
        ordered: The compositor inputs, in z-order; ``ordered[0]`` carries the
            output's band type.
        init: The caller's uncovered-pixel value.
        marker: The no-data value the caller passed.

    Returns:
        str | None: The ``VRTNodata`` to composite with, or `None` to keep `init`
        -- because the caller chose one, or because the band could not store the
        marker anyway.

    Warns:
        UserWarning: `marker` cannot be stored in the output band, so GDAL will
            drop it and the mosaic will carry no marker at all.

    Examples:
        - A marker the caller named takes the gaps while `init` is left alone:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _explicit_fill
            >>> band = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int32)
            >>> _explicit_fill([band], "nan", -1)
            '-1'

            ```
        - An `init` the caller named is left to hold them instead:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _explicit_fill
            >>> band = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int32)
            >>> print(_explicit_fill([band], 7, -1))
            None

            ```
    """
    dtype = np.dtype(gdal_to_numpy_type(ordered[0].GetRasterBand(1).DataType))
    try:
        candidate: float | None = float(marker)
    except (TypeError, ValueError):
        # Not a number this can judge; `gdal.Translate` will report it instead.
        candidate = None
    fill = None
    if candidate is not None and not fits_dtype(candidate, dtype):
        warnings.warn(
            f"no_data_value={marker!r} cannot be stored in a {dtype} band, so "
            "GDAL will drop it and the mosaic will carry no marker at all; pass "
            "a value that data type can hold, or omit no_data_value to have one "
            "chosen.",
            stacklevel=3,
        )
    elif _is_nan_spelling(init):
        fill = str(marker)
    return fill


def _storable_marker(
    ordered: list,
    src_paths: list[str],
    init: float | int | str,
    inherited: float | None,
    bbox: Sequence[float] | None = None,
    bbox_crs: int | str | None = None,
) -> Any | None:
    """Settle what a mosaic declares, and fills its gaps with, when nothing was passed.

    Inheriting nothing is not the same as having nothing to mask: pixels no source
    covers are still written, and leaving them undeclared turns them into ordinary
    data -- on an integer mosaic a literal `0`, indistinguishable from a real
    measurement and pulled straight into
    :meth:`~pyramids.dataset.Dataset.stats`. So a marker is settled on whenever
    there is something for it to mark, in four questions asked in this order:

    1. **What did the sources declare**, when the mosaic's dtype can store it? It
       usually can -- the sources and the mosaic share a dtype. What it cannot
       store is a `NaN` inherited onto an integer band, which is a real state:
       GDAL accepts `SetNoDataValue(nan)` on an integer band and reports it back
       unchanged, and so does
       :attr:`~pyramids.dataset.Dataset.no_data_value`, which never fabricates a
       storable number in its place. Stamping it on the output is what fails --
       :func:`gdal.Translate` answers "Nodata value was not set to output band,
       as it cannot be represented on its data type" and writes the band with no
       marker at all, leaving the same undeclared gaps. That warns and falls
       through.
    2. **Is `init` already `NaN`, on a band that can hold it?** Then the pixels
       are marked by the value they already carry, no real cell can collide with
       it, and nothing has to be read to know that. This is the floating mosaic,
       and the common case.
    3. **Do the sources leave any pixel uncovered at all?** When no source
       declared a marker and their footprints tile the area they span, there is
       nothing for one to mark. Choosing a value anyway would mean surveying every
       source to prove it unused -- reading a whole mosaic to mark nothing -- so
       the answer is no marker. :func:`gdal.BuildVRT` composites axis-aligned
       footprints, so :func:`_sources_tile_their_union` settles this from the
       geotransforms alone.
    4. **Which value can the dtype store that the data does not use?** The survey
       runs only here. `init` is offered to it as the *preferred* candidate rather
       than taken on trust: a numeric `init` the data already holds would
       otherwise become the mosaic's marker and mask that data, and `init=0` over
       an elevation mosaic is #1086 exactly, reached through a different argument.
       Refusing it warns, because the caller's uncovered pixels then hold the
       chosen value instead of the one they asked for.

    Whichever it is, the caller fills the mosaic's gaps with it, so the pixels the
    marker exists to cover really do hold it.

    Args:
        ordered: The compositor inputs, in z-order. ``ordered[0]`` decides the
            dtype: :func:`gdal.BuildVRT` gives the mosaic the first source's band
            type and skips any source that disagrees, so this is the mosaic's own
            type rather than an approximation of it.
        src_paths: The source paths, for the error message on a failed build.
        init: The caller's uncovered-pixel value, already validated as a NaN
            spelling or a number. A NaN spelling expresses no preference here,
            question 2 above having already had its chance at it.
        inherited: What the sources declared, or `None` when none of them did.
        bbox: The caller's window, when they gave one. The survey is clipped to
            it, since it is the only region that will be written -- and it is the
            region a caller passing a window is paying to read.
        bbox_crs: CRS of `bbox`, or `None` when it is already the mosaic's.

    Returns:
        Any | None: The value to declare and fill gaps with, or `None` -- when the
        mosaic has no uncovered pixel to mark, which is silent, or when the data
        leaves no value free, which warns.

    Raises:
        RuntimeError: GDAL failed to build or read the probe mosaic question 4
            measures. Only question 4 reads a pixel; the others answer from the
            arguments and the sources' geotransforms.

    Warns:
        UserWarning: The sources' own value cannot be stored in the mosaic's data
            type; or a numeric `init` was refused because the data uses it; or the
            mosaic's cells use every value that type could spare, so it is written
            with no marker at all.

    Examples:
        - Sources that tile their area leave nothing to mark, so nothing is
          chosen -- and no pixel is read to decide that:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _storable_marker
            >>> def tile(west, north):
            ...     ds = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int32)
            ...     ds.SetGeoTransform((west, 1.0, 0.0, north, 0.0, -1.0))
            ...     return ds
            >>> ordered = [tile(0.0, 2.0), tile(2.0, 2.0)]
            >>> print(_storable_marker(ordered, ["a", "b"], "nan", None))
            None

            ```
        - The same two sources placed diagonally leave a gap, which earns the
          package default -- a value an Int32 band can hold and the data does not
          use:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset.merge import _storable_marker
            >>> def tile(west, north):
            ...     ds = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Int32)
            ...     ds.SetGeoTransform((west, 1.0, 0.0, north, 0.0, -1.0))
            ...     return ds
            >>> ordered = [tile(0.0, 2.0), tile(2.0, 4.0)]
            >>> _storable_marker(ordered, ["a", "b"], "nan", None)
            -9999

            ```

    See Also:
        - :func:`pyramids.base._domain.inherit_no_data`: Produces `inherited`,
          resolving what the sources declare without regard to what the output can
          store.
        - :func:`_sources_tile_their_union`: Answers question 3.
        - :func:`_unused_marker`: Answers question 4.
    """
    # Band 1 decides the dtype, as it does for the inheritance itself, and
    # `gdal.Translate` stamps one marker on every band either way. Metadata only:
    # neither this nor the footprints below reads a pixel.
    dtype = np.dtype(gdal_to_numpy_type(ordered[0].GetRasterBand(1).DataType))
    # `merge_rasters` has already refused an `init` that names no number, so the
    # only two shapes reaching here are a NaN spelling -- no preference, since
    # question 2 above has had its chance at it -- and a number.
    preferred: tuple[Any, ...] = () if _is_nan_spelling(init) else (float(init),)
    # Annotated up front: the branches below answer with an inherited float, a
    # NaN, nothing, or whatever the survey finds, and mypy otherwise pins the
    # variable to the first of those.
    marker: Any | None
    if inherited is not None and fits_dtype(inherited, dtype):
        marker = inherited
    elif inherited is None and _is_nan_spelling(init) and fits_dtype(NAN, dtype):
        marker = NAN
    elif inherited is None and _sources_tile_their_union(
        [_source_bounds(handle) for handle in ordered]
    ):
        marker = None
    else:
        if inherited is not None:
            warnings.warn(
                f"the sources declare a no-data value of {inherited!r}, which a "
                f"{dtype} band cannot store, so GDAL would drop it and leave the "
                "mosaic unmarked; choosing a value that data type can hold "
                "instead.",
                stacklevel=3,
            )
        # Composited bare, with neither `srcNodata` nor `VRTNodata`: this mosaic
        # exists only to be measured, and the default `"nan"` for either is what
        # GDAL answers with "Band data type of <T> cannot represent the specified
        # NoData value of nan" on the very integer bands this branch serves.
        # A bare VRT still takes the sources' own declaration onto its band, and
        # `ComputeRasterMinMax` then skips the cells holding it -- but that costs
        # this branch nothing either way. Reached with `inherited is None`, no
        # source declared anything, so the probe declares nothing and every cell
        # is in view. Reached with `inherited` set, it is by definition a value
        # this dtype cannot store, so no cell holds it and none is skipped.
        window = None
        if bbox is not None:
            # In the mosaic's own CRS, and through the same reprojection the
            # write uses, so the survey measures the raster that will be written
            # rather than a different one.
            window = list(
                _bbox_in_projection(bbox, bbox_crs, ordered[0].GetProjection())
            )
        probe = run_gdal_op(
            partial(
                gdal.BuildVRT,
                "",
                ordered,
                options=gdal.BuildVRTOptions(outputBounds=window),
            ),
            error=RuntimeError,
            action="building the source mosaic",
            subject=f"sources {src_paths!r}",
            hint=_READABLE_RASTERS_HINT,
        )
        marker = _unused_marker(probe, dtype, preferred)
        if marker is None:
            warnings.warn(
                f"the mosaic's {dtype} cells use every value that data type could "
                "spare as a no-data marker, so the pixels no source covers are "
                "written as ordinary data; pass no_data_value= to choose one, "
                "accepting that it masks any real cell holding it.",
                stacklevel=3,
            )
        elif preferred and marker != preferred[0]:
            warnings.warn(
                f"init={init!r} cannot mark the mosaic's uncovered pixels because "
                f"its own cells hold that value; they are filled with {marker!r} "
                "instead, which the data does not use.",
                stacklevel=3,
            )
    return marker


def merge_rasters(
    src: Sequence[str | Path],
    dst: str | Path,
    no_data_value: Any = INHERIT_NO_DATA,
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
        no_data_value (float | int | str | None):
            Stamped on the output bands as the nodata marker. Omitted means
            **inherit from the sources**: the first value they declare wins, and
            a disagreement warns. Passing a value explicitly overrides that --
            but note that any real cell holding it becomes unreadable, which is
            why 0 is a poor choice for an elevation, bathymetry, anomaly or
            difference raster (#1086). Passing ``None`` explicitly asks for no
            marker at all, on either path, and the string ``"inherit"`` -- the
            word the default renders as -- asks for the default. A wrapper that
            forwards an "unset" of its own can import the sentinel itself as
            ``from pyramids.dataset.merge import INHERIT_NO_DATA``.

            **Whichever marker is settled on fills the pixels no source covers
            too**, on both write paths and however it was settled, so the gaps
            really hold what the band declares. `init` keeps those pixels only
            when you name it yourself: on Int32 tiles ``no_data_value=-1`` alone
            writes gaps of ``-1`` under a declared ``-1``, while
            ``no_data_value=-1, init=7`` writes gaps of ``7`` under a declared
            ``-1``, which masks nothing (both measured). The exception is a
            value the output band cannot store — it warns, GDAL drops it, and
            `init` keeps the gaps because there is no storable marker to fill
            them with.

            The value is read from each source's **band 1**, and one marker is
            stamped on every band of the output. It is read from *every* source,
            including one ``BuildVRT`` goes on to drop for a band type that
            disagrees with the first's -- so on such a mosaic a source that
            contributes no pixel can still decide the marker, or raise the
            disagreement warning. That is a symptom of the drop, which GDAL
            reports separately, rather than of the inheritance. A multi-band merge therefore
            keeps band 1's answer throughout: where band 1 declares nothing and
            band 2 declares a marker, nothing is inherited, and where the bands
            declare different markers, band 1's is stamped over band 2's. For a
            GeoTIFF that costs nothing, because the format has no per-band marker
            to lose -- ``TIFFTAG_GDAL_NODATA`` holds one value for the whole
            dataset, and GDAL warns that setting a second "will be used for all
            bands on re-opening". It is observable only for sources that do carry
            one per band, such as a VRT.

            When **no source declares one**, a marker is chosen wherever there
            is something for it to mark: a mosaic generally has pixels no source
            covers, and leaving those undeclared makes them read as real data --
            a literal `0` on an integer mosaic, which then poisons
            :meth:`~pyramids.dataset.Dataset.stats`. The reduction methods
            write Float64 and take ``NaN``, which no real cell can hold,
            whatever the sources' footprints. The z-order methods take the value
            those pixels would otherwise hold, `init` — so ``NaN`` too on a
            floating mosaic, the common case. Where the band cannot store
            ``NaN``, the footprints decide. Sources that **tile their whole
            area** leave no pixel to mark, so nothing is declared and no pixel
            is read to settle it: two contiguous Int32 tiles declaring nothing
            come back declaring nothing (measured). Sources that **leave a gap**
            earn a sentinel the dtype can store and the mosaic's own cells do
            not use -- a numeric `init` first, then ``-9999`` where the dtype
            holds it, then the dtype's own extremes, an unsigned band offering
            its maximum before its `0`, since `0` is the likelier real
            observation. A gapped signed integer mosaic of sources declaring
            nothing therefore comes back declaring ``-9999`` and an unsigned one
            its maximum -- ``65535`` for ``UInt16`` -- with the gaps holding it
            either way. Only a gapped mosaic whose data uses every value its
            dtype could spare is written without a marker, and that warns.

            An inherited value the output band cannot store -- a ``NaN`` from
            integer sources, which GDAL both accepts and reports back -- would
            be dropped by :func:`gdal.Translate` and leave the mosaic unmarked,
            so it warns and gives way to a storable one.
        init (float | int | str):
            Reported value for pixels with no source coverage in the VRT (z-order
            methods only — the reduction methods never read it). Maps to
            :func:`gdal.BuildVRTOptions` ``VRTNodata``.

            It is the fallback, not the last word: whatever marker gets settled
            on is filled into the gaps instead, so `init` is what survives only
            when that marker turns out to be `init` itself. Sources declaring
            ``-9999`` and an ``init=5`` yield gaps holding ``-9999`` (measured),
            and so does an explicit ``no_data_value=-9999``. The default
            ``"nan"`` is therefore what a floating mosaic inheriting nothing
            ends up declaring, and on an integer mosaic — which cannot store it
            — the storable sentinel described under `no_data_value` takes its
            place. `init` has the gaps to itself only when you name it, which is
            the case that needs it to agree with the marker.

            A **numeric** `init` is also the sentinel search's preferred
            candidate, so an integer mosaic inheriting nothing declares `init`
            itself when its own cells do not hold that value. One they do hold
            is passed over -- ``init=0`` over tiles containing real zeros is
            #1086 through a different argument -- and that warns. It must name a
            number, or be a spelling of ``NaN``; anything else is refused up
            front, since whether it would have reached GDAL at all depends on
            the output's data type and on whether the sources leave a gap.
        n (float | int | str):
            Source pixels matching this value are ignored — both when building
            the VRT mosaic (z-order, as ``srcNodata``) and when warping onto the
            union grid (the reduction methods, likewise). It is an **override**,
            applied to every source alike, and it *replaces* each source's own
            declaration rather than adding to it: over tiles declaring ``-9999``
            and ``-32768``, ``n=-9999`` composites the second tile's holes as
            real ``-32768`` measurements (measured). The default ``"nan"``
            therefore means "no override" — the spelling is matched, so
            ``"NaN"`` and ``float("nan")`` mean it too — and each source's own
            declared value marks that source's own holes, which is what lets the
            gaps and the inherited marker agree.
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
        The z-order methods (``"first"``/``"last"``) preserve the data type via
        ``BuildVRT`` + ``Translate`` — specifically that of the first source **in
        z-order**, which for ``method="first"`` is the *last* one you passed,
        since the list is reversed so that it wins. That is the source
        ``BuildVRT`` takes its band type from; one whose band type disagrees is
        skipped entirely, with a GDAL warning naming it, so the output then
        covers the remaining sources' extent rather than the union. The
        reduction methods (``"min"``/``"max"``/``"sum"``) align every source onto
        the union grid with ``gdal.Warp`` (nearest resampling — exact for
        already-aligned tiles) and write a single-precision-safe **Float64**
        output regardless of the source dtype, so they may differ in dtype from a
        z-order merge of the same integer inputs. The warp itself is asked for
        Float64 as well, so that the ``NaN`` marking a strip's uncovered pixels
        survives it: rounded into an integer destination it would come back as a
        real ``0`` (or the type's minimum for Int32) and be folded into the
        result as data.

    Raises:
        TypeError: ``resampling`` is not a string, or ``bbox`` is not four numbers
            (a string, a scalar, or a sequence holding a non-numeric element).
        ValueError: ``method``/``resampling`` is not a supported value,
            ``dst_crs`` cannot be parsed as a CRS, a source carries no CRS,
            ``no_data_value``, ``init`` or ``n`` names no number, or ``bbox`` is
            malformed
            (wrong length, non-finite, inverted, zero-area), crosses the
            antimeridian or the mosaic's longitude seam once reprojected, selects
            no whole pixel, does not overlap the mosaic, or cannot be projected
            into its CRS.
        RuntimeError: GDAL failed to open a source, reproject it, or build the
            source mosaic. When a source is at fault the message names it and
            its position in `src`, and chains GDAL's own error; any credential
            in a signed URL is redacted.
        DriverNotExistError: `dst` has no extension, or one the driver catalog
            does not know.
        FileFormatNotSupportedError: `dst`'s extension maps to a
            write-by-copy-only format, for any `method`.

    Warns:
        UserWarning: The sources declare more than one distinct no-data value
            (the first wins, and all of them are named); or the value they
            declare is one the output's data type cannot store, so a storable
            one is chosen in its place; or a numeric `init` was passed over as
            the marker because the mosaic's own cells hold that value, so the
            gaps hold the chosen sentinel instead; or the mosaic's cells use
            every value its dtype could spare as a marker, so it is written
            without one. Those four arise only while the marker is being
            inherited. Passing `no_data_value` explicitly silences them and
            raises one of its own where the value does not fit the output band:
            GDAL then drops it and the mosaic carries no marker at all.

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
        - Default last-wins compositing. The overlap rule is unchanged, but the
          mosaic's no-data marker is now inherited from the sources rather than
          set to 0 (#1086):
            ```python
            >>> merge_rasters(["tile_a.tif", "tile_b.tif"], "mosaic.tif")  # doctest: +SKIP

            ```
        - Override the inherited marker on a z-order merge of integer tiles.
          The pixels no tile covers are filled with ``-1`` as well, so the
          marker the band declares is the one they hold:
            ```python
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["tile_a.tif", "tile_b.tif"],
            ...     "mosaic.tif",
            ...     no_data_value=-1,
            ... )

            ```
        - Keep a different value in the uncovered pixels by naming `init`. The
          band still declares ``-1``, but the gaps hold ``0`` and stay readable:
            ```python
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["tile_a.tif", "tile_b.tif"],
            ...     "mosaic.tif",
            ...     no_data_value=-1,
            ...     init=0,
            ... )

            ```
        - Ask for no marker at all, so every value in the mosaic stays readable
          — including a real 0 m in an elevation model:
            ```python
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["dem_a.tif", "dem_b.tif"],
            ...     "dem.tif",
            ...     no_data_value=None,
            ... )

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

    # `init` and `n` both default to the string `"nan"`, which round-trips
    # through GDAL as float NaN -- a value no integer band can store. `n` never
    # reaches one as itself: the default means "no override" and is not passed on
    # at all (see `_source_nodata`). `init` gives way to a storable marker
    # whenever one is settled on, which is while the marker is being inherited
    # (see `_storable_marker`) and, now, when the caller named it and left `init`
    # alone (see `_explicit_fill`). It still reaches the band as NaN when the
    # caller names both. The spellings are kept for backwards-compat with the
    # previous gdal_merge.main-based signature.
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
        requested = _requested_no_data(no_data_value)
        _validated_init(init)
        inheriting = requested is INHERIT_NO_DATA
        sources, _keepalive = _prepare_sources(src_paths, dst_crs, resampling)
        # Under its own name rather than rebound onto the parameter, so that
        # `no_data_value` keeps meaning "what the caller asked for" throughout --
        # including the sentinel, which `inheriting` above is the only reading of.
        resolved_no_data = requested
        if inheriting:
            # Read it off the handles _prepare_sources already opened rather
            # than reopening: each open is billable under Requester-Pays.
            resolved_no_data = inherit_no_data(
                [handle.GetRasterBand(1).GetNoDataValue() for handle in sources]
            )

        if method in _REDUCE_METHODS:
            # Pair each handle with its path so a failure on the reduce path
            # names the source, not a SWIG proxy address (#1107).
            labelled = [
                _Source(f"{index + 1}/{len(src_paths)} {path!r}", handle)
                for index, (path, handle) in enumerate(zip(src_paths, sources))
            ]
            if inheriting and resolved_no_data is None:
                # Nothing to inherit. The reduction writes Float64, so NaN is
                # storable and no real cell can hold it -- the same rule the
                # z-order path applies, answered by the dtype it writes.
                resolved_no_data = float("nan")
            _merge_reduce(
                labelled, str(dst), method, resolved_no_data, n, bbox, bbox_crs
            )
            return

        # z-order: "last" keeps natural order (last source wins); "first"
        # reverses so the original first source is placed last in the VRT and
        # therefore wins.
        ordered = list(reversed(sources)) if method == "first" else sources
        vrt_fill: str | None = str(init)
        if inheriting:
            # Whether or not a value was inherited, the mosaic's uncovered pixels
            # have to be accounted for -- and the value has to be one this output
            # band can actually hold.
            resolved_no_data = _storable_marker(
                ordered, src_paths, init, resolved_no_data, bbox, bbox_crs
            )
            if resolved_no_data is not None:
                # Fill with what the output is about to declare. Where `init` is
                # already that value this is a no-op; otherwise the marker would
                # be stamped over gaps holding something else -- NaN on a float
                # mosaic inheriting -9999, or the 0 GDAL substitutes on an
                # integer one -- and would mask nothing.
                vrt_fill = str(resolved_no_data)
        elif resolved_no_data is not None:
            # The caller named the marker, so the gaps are filled with it too
            # unless they also named `init`. Otherwise the band declares one
            # value while its uncovered pixels hold another, which is the same
            # defect on the one path that was left out of it.
            explicit = _explicit_fill(ordered, init, resolved_no_data)
            if explicit is not None:
                vrt_fill = explicit
        if (
            resolved_no_data is None
            and _is_nan_spelling(init)
            and not fits_dtype(
                NAN, np.dtype(gdal_to_numpy_type(ordered[0].GetRasterBand(1).DataType))
            )
        ):
            # Nothing is being marked and the band cannot store the NaN `init`
            # defaults to, so asking GDAL for it buys nothing and costs a warning
            # per band ("Band data type of <T> cannot represent the specified
            # NoData value of nan"). The gaps it would have filled either do not
            # exist -- the sources tile their area -- or already read as 0,
            # because the NaN was refused. A floating band is untouched: there
            # the NaN is storable and the gaps really do come back as NaN.
            vrt_fill = None
        vrt_opts = gdal.BuildVRTOptions(
            srcNodata=_source_nodata(n),
            VRTNodata=vrt_fill,
        )
        vrt_ds = run_gdal_op(
            partial(gdal.BuildVRT, "", ordered, options=vrt_opts),
            error=RuntimeError,
            action="building the source mosaic",
            subject=f"sources {src_paths!r}",
            hint=_READABLE_RASTERS_HINT,
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
        # "none" *removes* the marker rather than omitting the option: the VRT
        # above carries `init` (NaN by default) as VRTNodata, so merely leaving
        # noData unset would stamp NaN -- invalid on an integer band, which is
        # exactly the "sentinel the band cannot store" defect. It is reached
        # only when the caller asked for no marker by passing `None`, or when
        # `_storable_marker` found the data using every value it could have
        # chosen (and said so); inheriting nothing otherwise yields a marker
        # rather than removing one.
        translate_opts = gdal.TranslateOptions(
            format=out_driver,
            creationOptions=["COMPRESS=LZW"] if out_driver == "GTiff" else [],
            projWin=proj_win,
            noData=("none" if resolved_no_data is None else str(resolved_no_data)),
        )
        out_ds = run_gdal_op(
            partial(gdal.Translate, str(dst), vrt_ds, options=translate_opts),
            error=RuntimeError,
            action="writing the mosaic",
            subject=f"destination {str(dst)!r}",
            outcome="produced no output",
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
            :func:`gdal.Warp` failed. Either way the message names the source
            and redacts any credential in it; when GDAL raised (the usual case
            under :func:`gdal.UseExceptions`) it also carries the source's
            position in ``src_paths`` and chains GDAL's own error.
    """
    resample_alg = resolve_resampling(resampling)

    # Open each source once; read its CRS from that same handle.
    opened: list = []
    source_srs: list[osr.SpatialReference] = []
    for index, path in enumerate(src_paths):
        # open_network_dataset (PR #1084) owns both GDAL failure shapes, and
        # redacts. Naming the source is what #1107 asks for: for a /vsicurl/ or
        # /vsis3/ source GDAL's own message is just the HTTP status ("HTTP
        # response code: 403") and names nothing. The index says how far the
        # open got on a big mosaic.
        dataset = open_network_dataset(
            path,
            error=RuntimeError,
            subject=f"source {index + 1}/{len(src_paths)} {path!r}",
        )
        wkt = dataset.GetProjection()
        if not wkt:
            raise ValueError(
                redact_credentials(
                    f"source {path!r} has no CRS; every source must carry a CRS "
                    "to be merged/reprojected."
                )
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
    for index, (path, dataset, srs) in enumerate(zip(src_paths, opened, source_srs)):
        if srs.IsSame(target_srs):
            sources.append(dataset)
            continue
        # Built outside the thunk: a bad option is the caller's mistake, and
        # re-branding it would blame the source for an argument error. `partial`
        # rather than a lambda so nothing closes over the loop variable.
        warp_opts = gdal.WarpOptions(
            format="VRT", dstSRS=target_wkt, resampleAlg=resample_alg
        )
        warped = run_gdal_op(
            partial(gdal.Warp, "", dataset, options=warp_opts),
            error=RuntimeError,
            action="reprojecting to the target CRS",
            subject=f"source {index + 1}/{len(src_paths)} {path!r}",
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
    source: _Source,
    strip_bounds: Sequence[float],
    x_size: int,
    ysize: int,
    src_nodata: float | int | str | None,
) -> np.ndarray:
    """Warp one source onto a strip's window and read it as a 3-D float64 cube.

    The warp is asked for a Float64 destination whatever the source's own type,
    because the strip's contract is that everything the source does not cover
    comes back as ``NaN`` — which only a floating destination can hold. Cells
    the source declares as no-data are turned into ``NaN`` too, so the fold that
    follows has one question to ask of a cell rather than two.

    Args:
        source: The source to warp, paired with the label to report it by.
        strip_bounds: The strip's ``(west, south, east, north)`` window.
        x_size: Strip width in pixels.
        ysize: Strip height in pixels.
        src_nodata: A value to treat as no-data in every source, overriding what
            each declares, or `None` to leave each with its own.

    Returns:
        np.ndarray: The warped strip, always ``(bands, rows, cols)`` even for a
        single-band source, in float64, with no-data and no coverage alike as NaN.

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
        # Warp into the dtype the reduction writes, rather than into the source's
        # and casting after. `dstNodata=nan` is the whole basis of the NaN-aware
        # fold below, and an integer destination cannot hold it: GDAL rounds it
        # into the type instead ("destination nodata value has been rounded to
        # 0, UInt16 being an integer datatype" -- 0 for Byte, UInt16, Int16 and
        # UInt32, and -2147483648 for Int32) and the strip comes back with those
        # as real values where it has no coverage. They then win every `fmin`
        # and are added by every `sum`, so a min/max/sum mosaic of integer tiles
        # that do not tile contiguously collapsed to all-zero -- whatever
        # `no_data_value` said, since `covered` never saw a gap. The cast below
        # is then a no-op rather than a second full-size copy.
        outputType=gdal.GDT_Float64,
    )
    warped = run_gdal_op(
        partial(gdal.Warp, "", source.handle, options=warp_opts),
        error=RuntimeError,
        action="warping onto the union grid",
        subject=f"source {source.label}",
    )
    # np.asarray pins the type: GDAL's ReadAsArray is untyped, so without it the
    # float64 cube is inferred as Any and leaks out of the annotated return.
    array = np.asarray(warped.ReadAsArray()).astype("float64", copy=False)
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
        src_paths: Sources as :class:`_Source` pairs, or bare paths/open
            datasets (labelled by their repr).
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

    for source, bounds in zip(src_paths, src_bounds):
        if _source_misses_strip(bounds, strip_lat, strip_bounds):
            continue
        array = _warp_onto_strip(source, strip_bounds, x_size, ysize, src_nodata)
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
    no_data_value: float | int | str | None,
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
    Pixels with no source coverage are written as `no_data_value`, or as ``NaN``
    when that is `None`. The output is Float64 whatever the sources' dtype, since
    the fold works in NaN-aware floating point throughout.

    Args:
        src_paths: Sources as :class:`_Source` pairs (so a failure names the
            source rather than a SWIG proxy), or bare path strings / already-open
            :class:`gdal.Dataset` objects (e.g. reprojected warped VRTs from
            :func:`_prepare_sources`).
        dst: Output raster path.
        method: One of ``"min"``, ``"max"``, ``"sum"``.
        no_data_value: Output no-data value and no-coverage fill. ``None``
            means the caller asked for no marker at all, so uncovered pixels are
            filled with NaN and the output declares nothing -- the same answer
            the z-order path gives for that request. "Nothing was inherited" is
            resolved to NaN by :func:`merge_rasters` before it calls here, so
            that case arrives as a value and is declared.
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
    sources = [_Source.of(source) for source in src_paths]
    template = run_gdal_op(
        partial(gdal.BuildVRT, "", [source.handle for source in sources]),
        error=RuntimeError,
        action="building the union mosaic",
        subject="sources [" + ", ".join(source.label for source in sources) + "]",
        hint=_READABLE_RASTERS_HINT,
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

    src_nodata = _source_nodata(n)
    # The reduction always needs *some* fill for uncovered pixels, and the
    # output is Float64, so NaN is the neutral choice when the caller asked for
    # no marker -- it cannot collide with real data the way 0 did (#1086). It is
    # then filled but not declared, so `no_data_value=None` means the same thing
    # on this path as on the z-order one.
    fill = float("nan") if no_data_value is None else float(no_data_value)
    # Extent of every source, computed once, to skip sources a strip cannot touch.
    src_bounds = [_source_bounds(source.handle) for source in sources]

    # Resolve from the extension so that one `dst` does not yield two different
    # formats depending on an unrelated argument: this reduction path hardcoded
    # GTiff while the z-order path below lets gdal.Translate infer, so `.nc`
    # produced a netCDF for method="last" and a GTiff for method="min". LZW is
    # GTiff-specific and is applied only there.
    out_driver = resolve_output_driver(dst)
    out_options = ["COMPRESS=LZW"] if out_driver == "GTiff" else []
    # Resolved before the thunk: an unknown driver yields None here, and
    # `.Create` on it would raise AttributeError -- which run_gdal_op does not
    # catch, so the failure would escape un-branded.
    driver = gdal.GetDriverByName(out_driver)
    out_ds = run_gdal_op(
        partial(
            driver.Create,
            dst,
            x_size,
            y_size,
            band_count,
            gdal.GDT_Float64,
            options=out_options,
        ),
        error=RuntimeError,
        action="writing the reduced mosaic",
        subject=f"destination {dst!r}",
        hint="check the output path is writable",
        outcome="produced no output",
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    if no_data_value is not None:
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
            sources,
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
    no_data_value: Any = INHERIT_NO_DATA,
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
        no_data_value: No-data value stamped on the output bands. Omitted means
            **inherit from the source rasters**: the first file that declares
            one wins, a disagreement warns, and if none declares one the output
            declares none either. Pass a value to override that, or ``None`` for
            "declare no sentinel at all". The *inheritance* is the same rule
            :func:`merge_rasters` follows, resolved by the same
            :func:`~pyramids.base._domain.inherit_no_data` -- but what happens
            when nothing is inherited is not, and deliberately so. A stack
            covers one grid, so it has no uncovered pixel and declares nothing;
            a mosaic generally has them, and settles on a marker rather than
            leaving them to read as data.
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

    Warns:
        UserWarning: The inputs declare more than one distinct no-data value,
            while `no_data_value` is being inherited. The first file's wins.

    Examples:
        - Stack a Landsat scene's per-band GeoTIFFs; the band names come from
          the file names:
            ```python
            >>> from pyramids.dataset.merge import stack_bands
            >>> scene = stack_bands(  # doctest: +SKIP
            ...     ["scene.B2.tif", "scene.B3.tif", "scene.B4.tif"]
            ... )
            >>> scene.band_names  # doctest: +SKIP
            ['B2', 'B3', 'B4']

            ```
        - Name the bands and write the stack straight to disk:
            ```python
            >>> from pyramids.dataset.merge import stack_bands
            >>> stack_bands(  # doctest: +SKIP
            ...     ["b2.tif", "b3.tif", "b4.tif"],
            ...     band_names=["blue", "green", "red"],
            ...     path="stack.tif",
            ... )

            ```
        - Resample mismatched inputs onto the first file's grid, and declare no
          sentinel on the result:
            ```python
            >>> from pyramids.dataset.merge import stack_bands
            >>> stack_bands(  # doctest: +SKIP
            ...     ["b2.tif", "coarse_b11.tif"],
            ...     align=True,
            ...     no_data_value=None,
            ... )

            ```

    See Also:
        - :meth:`pyramids.dataset.Dataset.from_band_files`: The method this
          delegates to, with the full contract and runnable examples.
        - :func:`merge_rasters`: Mosaics rasters that tile the same area,
          rather than stacking rasters that share a grid.
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
