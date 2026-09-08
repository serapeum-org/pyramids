"""OGC API – Coverages → :class:`~pyramids.dataset.Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_ogc_coverages`. It
fetches a coverage subset from an OGC API – Coverages service and returns a
single-raster :class:`~pyramids.dataset.Dataset`.

OGC API – Coverages is the **modern REST/JSON successor to WCS**: a landing page
links to ``/collections``, each coverage is a collection exposing
``/collections/{id}/coverage`` (with ``subset`` query subsetting and format
negotiation). The transport here is **GDAL's native ``OGCAPI`` driver** — no
third-party OGC or HTTP client library. The driver discovers the coverage,
negotiates the
GeoTIFF representation and exposes it as a single (often planet-spanning) virtual
raster whose 256×256 tiles are fetched lazily on read.

That virtual raster is **unbounded** — opening it spans the whole coverage, so a
read with no window allocates petabytes. The coverage ``API=COVERAGE`` mode does
**not** honour ``MINX/MINY/MAXX/MAXY`` open options either (they are silently
ignored). The correct subset is therefore done at *read* time: we resolve the
coverage's native CRS, project the requested lon/lat ``bbox`` into it, and call
:func:`gdal.Translate` with that ``projWin`` **and an explicit output size cap**.
The driver then fetches only the tiles (or a coarse overview) intersecting the
window. Both are mandatory: ``projWin`` bounds the area, the size cap bounds the
allocation. Hence ``bbox`` is **required** here, unlike :meth:`from_wcs`.

pyramids adds, on top of the driver, a cached ``/collections`` check so an
unadvertised coverage fails fast with a clear :class:`ValueError`, and so
transport / driver failures surface as
:class:`~pyramids.base._errors.OGCAPIError`.

It also serves an **antimeridian** ``bbox`` -- one whose ``minx > maxx`` -- the
way :meth:`pyramids.dataset.Dataset.crop` does: the window is split at the 180
degree seam (:func:`pyramids.base._coverage.seam_halves`), both halves are read
off the same open coverage, and the two are concatenated along longitude. Such a
``bbox`` used to be refused with ``ValueError: bbox must have minx < maxx and
miny < maxy``.

This is the OGC-API-era sibling of :mod:`pyramids.dataset._wcs` (the WCS reader);
the two share their bbox/CRS/window helpers through the protocol-neutral
:mod:`pyramids.base._coverage`, and share their ``/collections`` discovery with
:mod:`pyramids.feature._oapif` (the OGC API – Features reader) through
:mod:`pyramids.base._ogc_api`.

Scope boundary (see ``docs/SCOPE.md``): this reader takes only generic OGC
inputs. Provider specifics — coverage-name catalogs, agency auth endpoints, the
proj4/WKT string for a coverage whose advertised CRS is absent from PROJ — live
in the downstream consumer (``earthlens``), which calls ``from_ogc_coverages``
and passes ``coverage_crs`` / ``auth`` as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlunsplit

from osgeo import gdal

from pyramids.base._coverage import check_seam_bbox as _check_seam_bbox
from pyramids.base._coverage import native_projwin as _native_projwin
from pyramids.base._coverage import open_network_dataset as _open_network_dataset
from pyramids.base._coverage import read_size as _read_size
from pyramids.base._coverage import resolution_pair as _resolution_pair
from pyramids.base._coverage import resolve_native_srs as _resolve_native_srs
from pyramids.base._coverage import seam_halves as _seam_halves
from pyramids.base._coverage import seam_offset as _seam_offset
from pyramids.base._coverage import translate_to_mem as _translate_to_mem
from pyramids.base._coverage import validate_bbox as _validate_bbox
from pyramids.base._coverage import window_overlaps as _window_overlaps
from pyramids.base._errors import CoverageError, OGCAPIError
from pyramids.base._ogc_api import append_path as _append_path
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config
from pyramids.base._ogc_api import get_collections as _get_collections
from pyramids.base._ogc_api import not_advertised

# Borrowed rather than reimplemented, so a seam read here, a seam read over WCS
# and `Dataset.crop` all stitch the same way: `_stitch_lon_halves` is the crop
# engine's longitude concatenation, which keeps the west half's geotransform so
# longitude continues past the seam rather than jumping back to -180.
from pyramids.dataset.engines.spatial import _stitch_lon_halves

if TYPE_CHECKING:
    from osgeo import osr

    from pyramids.dataset.dataset import Dataset

_OPEN_OPTIONS = ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF", "CACHE=NO"]


def _coverage_connection(endpoint: str, coverage: str) -> str:
    """Build the GDAL ``OGCAPI:`` connection string for one coverage collection.

    The ``/collections/{coverage}`` path segment is inserted **before** any
    existing query string (via the shared
    :func:`pyramids.base._ogc_api.append_path`) so a query-string-auth endpoint
    (e.g. ``https://host/ogc?api_key=…``) keeps its query intact instead of
    producing ``…?api_key=…/collections/{coverage}``. The coverage identifier is
    URL-encoded so a name containing ``/`` or other reserved characters lands as a
    single path segment.
    """
    parts, path = _append_path(endpoint, f"/collections/{quote(coverage, safe='')}")
    base = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
    return f"OGCAPI:{base}"


def _open_coverage(connection: str, coverage: str) -> gdal.Dataset:
    """Open an OGC API – Coverages coverage with GDAL, classifying failures.

    Raises:
        OGCAPIError: The ``OGCAPI`` driver is absent from this GDAL build, or GDAL
            could not open the coverage (service error, bad representation,
            unresolvable CRS, …).
    """
    if gdal.GetDriverByName("OGCAPI") is None:
        raise OGCAPIError(
            "the OGCAPI driver is not available in this GDAL build; OGC API – "
            "Coverages reads require GDAL built with the OGCAPI driver"
        )
    return _open_network_dataset(
        connection,
        error=OGCAPIError,
        subject=f"OGC API coverage {coverage!r}",
        open_options=_OPEN_OPTIONS,
    )


def _translate_window(
    src: gdal.Dataset, projwin: list[float], size: tuple[int, int], coverage: str
) -> gdal.Dataset:
    """Materialise the bounded, size-capped window via :func:`gdal.Translate` → MEM.

    The ``projWin`` bounds the fetched area and the explicit ``width``/``height``
    bound the allocation (without it the unbounded virtual raster would allocate
    petabytes). Translating into an in-memory dataset means a non-raster / error
    body makes ``gdal.Translate`` fail and we raise :class:`OGCAPIError` here,
    before any file is produced.

    Raises:
        OGCAPIError: GDAL could not produce a raster for the requested window.
    """
    width, height = size
    return _translate_to_mem(
        src,
        error=OGCAPIError,
        action="OGC API coverage read",
        subject=repr(coverage),
        projWin=projwin,
        width=width,
        height=height,
    )


def _window_sizes(
    projwins: list[list[float]], res: tuple[float, float] | None
) -> list[tuple[int, int]]:
    """Pixel ``(width, height)`` for each window, on one shared grid when there are two.

    One window is sized exactly as it always was, straight through
    :func:`~pyramids.base._coverage.read_size`. Two windows are the halves of an
    antimeridian read and have to stitch afterwards, which they can only do on a
    common grid: sizing each half on its own with no ``res`` would cap **each** at
    :data:`~pyramids.base._coverage.DEFAULT_MAX_PX` on its longer side, giving the
    two different pixel sizes and different row counts. So the combined span is
    sized once and the resolution read back out of it, then applied to both halves
    -- which also keeps the whole seam-crossing read inside the same pixel budget
    an unwrapped read of the same width would get.

    Args:
        projwins: One or two ``[ulx, uly, lrx, lry]`` windows in the coverage's
            native CRS, in west-to-east order.
        res: The caller's ``(x_res, y_res)``, or ``None`` to size from the cap.

    Returns:
        list[tuple[int, int]]: The ``(width, height)`` for each window, in the
            same order.

    Raises:
        ValueError: A window exceeds the pixel ceiling, or `res` has a
            non-positive axis (both raised by
            :func:`~pyramids.base._coverage.read_size`).

    Examples:
        - One window with an explicit resolution is sized straight from it:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _window_sizes
            >>> _window_sizes([[170.0, 10.0, 175.0, -10.0]], (0.05, 0.05))
            [(100, 400)]

            ```
        - Two halves of a seam read share one resolution, so their widths are in
          proportion to their spans and their heights are identical -- which is
          what lets them be concatenated afterwards:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _window_sizes
            >>> halves = [[170.0, 10.0, 180.0, -10.0], [-180.0, 10.0, -175.0, -10.0]]
            >>> sizes = _window_sizes(halves, None)
            >>> sizes
            [(512, 1024), (256, 1024)]
            >>> sizes[0][1] == sizes[1][1]
            True

            ```
        - The cap is spent on the combined span, not once per half, so the two
          widths add up to what a single unwrapped window of that span would get --
          15 degrees against 20 of latitude, so the tall side takes the 1024 cap and
          the width follows the aspect ratio:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _window_sizes
            >>> halves = [[170.0, 10.0, 180.0, -10.0], [-180.0, 10.0, -175.0, -10.0]]
            >>> sizes = _window_sizes(halves, None)
            >>> sum(width for width, _ in sizes)
            768

            ```
    """
    if len(projwins) < 2:
        return [_read_size(projwin, res) for projwin in projwins]
    grid_res = res
    if grid_res is None:
        span_x = sum(abs(pw[2] - pw[0]) for pw in projwins)
        ulx, uly, _, lry = projwins[0]
        total, height = _read_size([ulx, uly, ulx + span_x, lry], None)
        grid_res = (span_x / total, abs(uly - lry) / height)
    # Round the west half, then take the east as the remainder of the combined
    # width, exactly as `_seam_windows` does for WMS. Rounding each half on its own
    # gives them cell sizes that differ in the sixth decimal, which the stitch
    # cannot represent: it keeps the west geotransform, so the east half's declared
    # edge drifts from where its data actually ends.
    west, east = projwins
    total, height = _read_size(
        [west[0], west[1], west[0] + _span(west) + _span(east), west[3]], grid_res
    )
    west_width = max(1, min(total - 1, round(_span(west) / grid_res[0])))
    return [(west_width, height), (total - west_width, height)]


def _span(projwin: list[float]) -> float:
    """The absolute x extent of a ``[ulx, uly, lrx, lry]`` window.

    Args:
        projwin: A native-CRS window.

    Returns:
        float: ``abs(lrx - ulx)``.

    Examples:
        - The width of an ordinary window:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _span
            >>> _span([170.0, 10.0, 180.0, -10.0])
            10.0

            ```
        - Corner order does not matter, so a window read off a west-positive grid
          measures the same:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _span
            >>> _span([180.0, 10.0, 170.0, -10.0])
            10.0

            ```
    """
    return abs(projwin[2] - projwin[0])


def _align_to_sizes(
    projwins: list[list[float]], sizes: list[tuple[int, int]]
) -> list[list[float]]:
    """Trim each window to the whole number of pixels it will actually be read at.

    :func:`_window_sizes` gives the two halves of a seam read one cell size and
    splits the combined width between them, which means neither half's requested
    span is any longer an exact multiple of that cell size. `gdal.Translate` honours
    the window and the size it is given, so leaving the spans untouched would make
    it resample each half by a fraction of a pixel and hand back two grids that no
    longer share one. Snapping the far edge to ``ulx + width * cell`` is what keeps
    the promise the sizes make.

    A single window is returned untouched: its size came straight from its own
    span, so there is nothing to reconcile.

    Args:
        projwins: One or two ``[ulx, uly, lrx, lry]`` windows.
        sizes: The ``(width, height)`` chosen for each, in the same order.

    Returns:
        list[list[float]]: The windows, with the far x edge snapped for a split
            read.

    Examples:
        - A single window is handed back as it was:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _align_to_sizes
            >>> _align_to_sizes([[170.0, 10.0, 175.0, -10.0]], [(100, 400)])
            [[170.0, 10.0, 175.0, -10.0]]

            ```
        - Two halves are snapped onto the cell size their widths imply, so both
          report the same one:
            ```python
            >>> from pyramids.dataset._ogc_coverages import _align_to_sizes
            >>> halves = [[170.0, 10.0, 180.0, -10.0], [-180.0, 10.0, -175.0, -10.0]]
            >>> snapped = _align_to_sizes(halves, [(512, 1024), (256, 1024)])
            >>> cells = [(w[2] - w[0]) / n for w, (n, _) in zip(snapped, [(512, 0), (256, 0)])]
            >>> cells[0] == cells[1]
            True

            ```
    """
    if len(projwins) < 2:
        return projwins
    west, east = projwins
    cell = _span(west) / sizes[0][0]
    return [
        [west[0], west[1], west[0] + sizes[0][0] * cell, west[3]],
        [east[0], east[1], east[0] + sizes[1][0] * cell, east[3]],
    ]


def _fetch_windows(
    connection: str,
    coverage: str,
    windows: list[tuple[float, float, float, float]],
    coverage_crs: str | None,
    res: tuple[float, float] | None,
    config: dict[str, str],
) -> tuple[list[gdal.Dataset], osr.SpatialReference]:
    """Open the coverage once and materialise every requested window from it.

    `windows` holds one box for an ordinary request and two for one crossing the
    antimeridian. Reading both halves off a single open handle is what lets them
    stitch: they share the coverage's CRS, its lattice and -- via
    :func:`_window_sizes` -- one resolution.

    Args:
        connection: The ``OGCAPI:`` connection string for the coverage.
        coverage: The coverage identifier, used in error messages.
        windows: One or two ``(minx, miny, maxx, maxy)`` CRS84 boxes, each with
            ``minx < maxx``, in west-to-east order.
        coverage_crs: The CRS shim for a coverage the service advertises in a CRS
            absent from PROJ, or ``None``.
        res: The caller's normalised ``(x_res, y_res)``, or ``None``.
        config: GDAL config options (auth / timeout) to install around the read.

    Returns:
        tuple[list[gdal.Dataset], osr.SpatialReference]: The ``MEM`` raster for
            each window that had data, and the coverage's native CRS. With two
            windows a half that misses the coverage entirely is dropped, so the
            list can be shorter than `windows` -- and empty when neither overlaps.

    Raises:
        ValueError: ``coverage_crs`` cannot be interpreted, a window does not
            project to a finite native-CRS extent, or a window exceeds the pixel
            ceiling.
        OGCAPIError: The ``OGCAPI`` driver is unavailable, the coverage could not
            be opened or has no resolvable CRS, or GDAL produced no raster.
    """
    mems: list[gdal.Dataset] = []
    with gdal.config_options(config):
        src = _open_coverage(connection, coverage)
        try:
            # _resolve_native_srs is the shared, protocol-neutral resolver;
            # normalise its CoverageError (CRS-less coverage, no coverage_crs shim)
            # to this reader's OGCAPIError so the documented Raises contract holds
            # and the message names OGC API. A bad coverage_crs raises ValueError,
            # which propagates.
            try:
                native_srs = _resolve_native_srs(src, coverage_crs)
            except CoverageError as exc:
                raise OGCAPIError(
                    f"OGC API coverage {coverage!r} has no resolvable spatial reference; "
                    "the service advertised no usable CRS for the coverage"
                ) from exc
            projwins = [_native_projwin(w, "EPSG:4326", native_srs) for w in windows]
            if len(projwins) > 1:
                # Overlap-filtered before fetching, as _crop_seam_halves does,
                # but only for a split request. A lone window is left to GDAL,
                # which warns and fills a miss with no-data rather than raising;
                # filtering it too would empty the list and leave _window_sizes
                # nothing to size.
                projwins = [pw for pw in projwins if _window_overlaps(pw, src)]
            sizes = _window_sizes(projwins, res)
            projwins = _align_to_sizes(projwins, sizes)
            for projwin, size in zip(projwins, sizes, strict=True):
                mems.append(_translate_window(src, projwin, size, coverage))
        finally:
            # release the opened coverage handle on every path, error or not.
            src = None
    return mems, native_srs


def from_ogc_coverages(
    dataset_cls: type[Dataset],
    endpoint: str,
    *,
    coverage: str,
    bbox: tuple[float, float, float, float],
    output_crs: str | None = None,
    resolution: float | tuple[float, float] | None = None,
    coverage_crs: str | None = None,
    output: str | Path | None = None,
    resample: str = "nearest",
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> Dataset:
    """Fetch an OGC API – Coverages coverage subset and return a :class:`Dataset`.

    This is the private implementation; the public API is the
    :meth:`pyramids.dataset.Dataset.from_ogc_coverages` classmethod, which
    forwards here. See that method for the full parameter documentation.

    ``bbox`` is **required**: the ``OGCAPI`` coverage driver exposes the coverage as
    an unbounded virtual raster, so a windowless read is impossible. The lon/lat
    (CRS84) ``bbox`` is projected into the coverage's native CRS and read with a
    size cap so the fetch stays bounded.

    ``coverage_crs`` is the CRS shim: when the service advertises a CRS the local
    PROJ database cannot resolve, GDAL opens the coverage with no spatial reference
    and the bbox cannot be projected. Passing ``coverage_crs`` (any proj4 / WKT /
    authority string) supplies that CRS explicitly, mirroring :meth:`from_wcs`.

    ``bbox`` may cross the antimeridian. ``minx > maxx`` is read as a box wrapping
    the 180 degree seam (it used to be refused as inverted): the window is split at
    the seam, both halves are read off the same open coverage, and the two are
    concatenated along longitude. The merged raster keeps the **west** half's
    geotransform, so its longitudes continue past the seam -- 170..180 then
    180..190 -- rather than jumping back to -180. A half that misses the coverage
    entirely is dropped before it is requested, so a one-sided overlap returns just
    that half. With no ``resolution`` the pixel cap is applied to the combined span
    once and shared by both halves, so a seam read is budgeted like the unwrapped
    read of the same width rather than twice over.

    Note:
        The stitched result of a seam-crossing read is a plain
        :class:`~pyramids.dataset.Dataset`, not `dataset_cls`, because the merge
        rebuilds the raster through :meth:`Dataset.from_array`. Every
        single-window read still returns `dataset_cls`.

    Raises:
        ValueError: ``bbox`` is malformed, ``coverage`` is not advertised by the
            service, ``coverage_crs`` cannot be interpreted, or a seam-crossing
            ``bbox`` overlaps no part of the coverage / produced two halves that do
            not meet at the seam.
        OGCAPIError: The ``OGCAPI`` driver is unavailable, the service could not be
            reached, or it returned an error / a non-raster body.
    """
    box = _validate_bbox(bbox, allow_antimeridian=True)
    # This reader's bbox is contractually CRS84, so the projected half of the
    # guard cannot fire -- the corner-range half can, and an overhanging half
    # would otherwise be dropped without a word.
    _check_seam_bbox(box, "EPSG:4326")
    res = _resolution_pair(resolution)
    # One window normally, two when the bbox wraps the seam. Splitting
    # unconditionally keeps the ordinary request on exactly the path it had.
    windows = _seam_halves(box)

    collections = _get_collections(endpoint, auth, timeout)
    if collections and coverage not in collections:
        raise not_advertised("coverage", coverage, endpoint, collections)

    mems, native_srs = _fetch_windows(
        _coverage_connection(endpoint, coverage),
        coverage,
        windows,
        coverage_crs,
        res,
        _gdal_http_config(auth, timeout),
    )

    parts: list[Dataset] = []
    try:
        for mem in mems:
            mem.SetSpatialRef(native_srs)
            parts.append(dataset_cls(mem, access="write"))
        if not parts:
            raise ValueError(
                f"bbox {bbox!r} crosses the antimeridian but neither half overlaps "
                f"the extent of coverage {coverage!r}"
            )
        if len(parts) == 1:
            # Hand ownership of the only part to the caller, so the cleanup below
            # does not close the raster being returned.
            ds, parts = parts[0], []
        else:
            # _stitch_lon_halves copies both halves into a new raster, so the parts
            # stay owned here and are closed by the finally. Its first argument is
            # only read for band names; the west half carries the same ones.
            # The seam offset is measured in the coverage's own CRS: the halves are
            # windowed in that CRS, so for a projected coverage they meet at the
            # world width in metres, not at 360.
            ds = _stitch_lon_halves(
                parts[0],
                parts[0],
                parts[1],
                _seam_offset(box, "EPSG:4326", native_srs),
            )
    finally:
        for part in parts:
            part.close()

    if output_crs is not None:
        ds = ds.to_crs(output_crs, method=resample)

    if output is not None:
        ds.to_file(output)
    return ds
