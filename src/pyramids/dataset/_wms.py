"""OGC Web Map Service (WMS) / Web Map Tile Service (WMTS) → :class:`Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_wms` and
:meth:`pyramids.dataset.Dataset.from_wmts`. Both pull an OGC *map* layer into a
single-raster :class:`~pyramids.dataset.Dataset` using **GDAL's native WMS / WMTS
drivers** — no third-party OGC or raster-I/O libraries — matching the WCS /
WFS / OGC API readers.

Unlike WCS (which returns coverage *data values*), a WMS / WMTS layer is a
**rendered map**: the service styles the data server-side and returns an image, so
the result is a georeferenced RGB / RGBA raster (a picture), not scientific pixel
values. Read a WMS/WMTS layer when you want the *imagery* (satellite mosaics,
rendered basemap-as-data, GIBS true-colour); read WCS / OGC API – Coverages when
you want the underlying values.

The two protocols map onto GDAL differently:

* **WMS** (``GetMap``): GDAL opens a ``<GDAL_WMS>`` service descriptor whose
  ``<DataWindow>`` already carries the georeferenced request window (bbox + output
  pixel size) in the requested CRS. The service renders that exact extent, so the
  descriptor *is* the crop — no client-side reprojection is needed.
* **WMTS** (tiled ``GetTile``): GDAL opens the layer as a full georeferenced tile
  pyramid (with overviews) via the ``WMTS:<caps_url>,layer=<id>`` connection
  string. We then crop the requested bbox out of it with :func:`gdal.Translate`
  (reprojecting the bbox into the layer's native CRS with ``pyproj``), exactly as
  the WCS reader windows a coverage.

Both readers accept a bbox that **crosses the antimeridian** (``minx > maxx``,
e.g. ``(170, -10, -170, 10)``), the same wrap :meth:`Dataset.crop` accepts: the
request is split at the 180 degree seam, each half is fetched on its own, and the
two are stitched back into one raster that keeps the west half's geotransform, so
longitude runs on past the seam (``170 .. 180`` then ``180 .. 190``). See
:func:`_seam_windows` for how a WMS request's *pixel* width is divided between
the halves — the part that has no analogue in the other OGC readers.

Scope boundary (see ``docs/SCOPE.md``): these readers take only generic OGC
inputs. Provider specifics — endpoint / layer catalogs, agency auth (NASA GIBS,
EUMETSAT), tile-matrix-set naming — live in the downstream consumer, which calls
``from_wms`` / ``from_wmts`` with a concrete endpoint and layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from osgeo import gdal

from pyramids.base._coverage import MAX_PX
from pyramids.base._coverage import check_seam_bbox as _check_seam_bbox
from pyramids.base._coverage import native_projwin as _native_projwin
from pyramids.base._coverage import native_resolution as _native_resolution
from pyramids.base._coverage import open_network_dataset as _open_network_dataset
from pyramids.base._coverage import read_size as _read_size
from pyramids.base._coverage import resolution_pair as _resolution_pair
from pyramids.base._coverage import resolve_native_srs as _resolve_native_srs_neutral
from pyramids.base._coverage import seam_halves as _seam_halves
from pyramids.base._coverage import seam_offset as _seam_offset
from pyramids.base._coverage import translate_to_mem as _translate_to_mem
from pyramids.base._coverage import validate_bbox as _validate_bbox
from pyramids.base._coverage import window_overlaps as _window_overlaps
from pyramids.base._errors import CoverageError, WMSError
from pyramids.base._grid import grid_size
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config
from pyramids.base._ogc_api import not_advertised
from pyramids.dataset._subdataset import subdatasets_of

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


def _xml_escape(text: str) -> str:
    """Minimal XML escaping for descriptor text nodes."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _layers_value(layers: str | list[str] | tuple[str, ...]) -> str:
    """Normalise ``layers`` to a comma-joined WMS ``<Layers>`` value.

    Raises:
        ValueError: ``layers`` is empty or contains a blank entry — either would
            produce a malformed ``<Layers>`` (e.g. ``<Layers></Layers>`` or a
            leading comma) that the server rejects opaquely.
    """
    items = [layers] if isinstance(layers, str) else list(layers)
    if not items or any(not str(item).strip() for item in items):
        raise ValueError("layers must name at least one non-empty layer")
    return ",".join(items)


def _crs_element_tag(version: str) -> str:
    """Return the WMS CRS element tag for ``version`` (``CRS`` >= 1.3.0, else ``SRS``).

    GDAL's WMS minidriver refuses to initialize a descriptor that uses ``<CRS>``
    for WMS 1.1.1 and below (it requires ``<SRS>``), and vice-versa for 1.3.0+, so
    the tag must track the requested version. An unparseable version is assumed
    modern (``CRS``).
    """
    try:
        parsed = tuple(int(part) for part in version.split("."))
    except ValueError:
        parsed = (1, 3, 0)
    return "CRS" if parsed >= (1, 3, 0) else "SRS"


def _output_size(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int] | None,
    resolution: float | tuple[float, float] | None,
) -> tuple[int, int]:
    """Resolve the WMS output image size in pixels.

    A WMS ``GetMap`` needs explicit width/height, so exactly one of ``size`` or
    ``resolution`` must be given: ``size`` is used verbatim; ``resolution`` (pixel
    size in the bbox CRS units) is divided into the bbox extent.

    The extent is measured through :func:`~pyramids.base._coverage.seam_halves`, so
    an antimeridian bbox (``minx > maxx``) is sized from the two halves it really
    spans rather than from the negative ``maxx - minx`` that reading it as an
    ordinary box would give.

    Args:
        bbox: The validated ``(minx, miny, maxx, maxy)``, possibly wrapping.
        size: The caller's ``(width, height)`` in pixels, or ``None`` to derive it
            from `resolution`.
        resolution: Pixel size in the bbox CRS units -- a scalar for square pixels
            or an ``(x_res, y_res)`` pair -- or ``None`` when `size` is given.

    Returns:
        tuple[int, int]: The ``(width, height)`` to request, each at least 1 pixel.

    Raises:
        ValueError: both or neither of ``size`` / ``resolution`` were given,
            ``size`` is not two positive integers, or ``resolution`` over this
            extent exceeds :data:`~pyramids.base._coverage.MAX_PX` on either axis.

    Examples:
        - A wrapping bbox is sized from the span it actually covers (20 degrees
          across the seam, not the -340 the raw corners subtract to):
            ```python
            >>> from pyramids.dataset._wms import _output_size
            >>> _output_size((170.0, -10.0, -170.0, 10.0), None, 0.05)
            (400, 400)

            ```
    """
    if size is not None and resolution is not None:
        raise ValueError("pass either size=(width, height) or resolution=, not both.")
    if size is not None:
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"size must be two positive integers, got {size!r}")
        result = (width, height)
    else:
        res = _resolution_pair(resolution)
        if res is None:
            raise ValueError(
                "from_wms needs the output size: pass size=(width, height) or "
                "resolution=<pixel size in the bbox CRS units>."
            )
        _, miny, _, maxy = bbox
        span_x = sum(half[2] - half[0] for half in _seam_halves(bbox))
        # A GetMap is an HTTP fetch like any other, so it takes the same ceiling
        # the coverage readers use. It matters more here than it used to: a
        # transposed bbox is now read as a 359 degree wrap, and at a fine
        # resolution that resolves to hundreds of thousands of columns -- two
        # requests no server will serve, and half a gigabyte per half held in
        # memory before the stitch copies both again.
        result = grid_size(span_x, maxy - miny, res, max_px=MAX_PX)
    return result


def _seam_windows(
    bbox: tuple[float, float, float, float], size: tuple[int, int]
) -> list[tuple[tuple[float, float, float, float], tuple[int, int]]]:
    """The one or two ``GetMap`` requests that together render ``bbox`` at ``size``.

    An ordinary bbox is one request, handed back untouched. A wrapping one must be
    two, because ``GetMap`` cannot be asked for ``minx > maxx`` — and unlike the
    other OGC readers a WMS request also carries an explicit **pixel** size, so the
    requested ``width`` has to be divided between them. The rule:

    1. The ground resolution is fixed first, at ``span / width`` over the *whole*
       wrapping span — exactly the resolution a non-wrapping request of the same
       span and width would get. Both halves are then asked for at that one
       resolution, so it is uniform across the seam by construction.
    2. The seam is snapped to the nearest whole pixel boundary:
       ``west_columns = round(west_span / res)``, and the east half takes the
       remaining ``width - west_columns``. The two counts sum to ``width`` by
       definition, so the stitched raster is exactly as wide as was asked for.
    3. Snapping moves the requested window by at most half a pixel, so the halves
       are asked for over the *snapped* boxes (``180 - west_columns * res .. 180``
       and ``-180 .. -180 + east_columns * res``) rather than the raw ones. The
       alternative — keeping the raw corners and letting each half round its own
       width — leaves one pixel straddling the seam, which no single ``GetMap``
       can render, and gives the two halves subtly different resolutions.

    When the snap puts every column on one side (a sliver under half a pixel wide
    on the other), the result is that one request at the full ``width``: the sliver
    was already unrenderable, and dropping it *is* the half-pixel snap rather than
    a second exception to it.

    Args:
        bbox: The validated ``(minx, miny, maxx, maxy)``, possibly wrapping.
        size: The requested output ``(width, height)`` in pixels.

    Returns:
        list[tuple[tuple[float, float, float, float], tuple[int, int]]]: One
            ``(bbox, size)`` request, or two in west-to-east order.

    Raises:
        ValueError: `bbox` wraps but `size` is under two pixels wide, which cannot
            give each side of the seam a pixel of its own.

    Examples:
        - An ordinary bbox is one request, unchanged:
            ```python
            >>> from pyramids.dataset._wms import _seam_windows
            >>> _seam_windows((5.0, 51.0, 6.0, 52.0), (512, 256))
            [((5.0, 51.0, 6.0, 52.0), (512, 256))]

            ```
        - A wrapping bbox becomes two requests whose widths sum to the requested
          400, split in proportion to each half's share of the 20 degree span:
            ```python
            >>> from pyramids.dataset._wms import _seam_windows
            >>> windows = _seam_windows((170.0, -10.0, -170.0, 10.0), (400, 200))
            >>> [px for _, px in windows]
            [(200, 200), (200, 200)]
            >>> [tuple(round(v, 9) for v in box) for box, _ in windows]
            [(170.0, -10.0, 180.0, 10.0), (-180.0, -10.0, -170.0, 10.0)]

            ```
        - An uneven split still sums to the requested width — the halves get 133
          and 267 columns of one shared 0.0375 degree pixel, not 200 each:
            ```python
            >>> from pyramids.dataset._wms import _seam_windows
            >>> windows = _seam_windows((175.0, -10.0, -170.0, 10.0), (400, 200))
            >>> [px[0] for _, px in windows]
            [133, 267]

            ```
    """
    halves = _seam_halves(bbox)
    width, height = size
    if len(halves) == 1:
        result = [(bbox, size)]
    else:
        west_half, east_half = halves
        _, miny, _, maxy = bbox
        if width < 2:
            raise ValueError(
                f"a bbox crossing the antimeridian is rendered as two requests, so "
                f"it needs at least 2 pixels of width; got {width}. Ask for a wider "
                f"size, or a finer resolution."
            )
        res = ((west_half[2] - west_half[0]) + (east_half[2] - east_half[0])) / width
        # Nearest pixel boundary, which is what bounds the window shift at half a
        # pixel -- but rounded half *up* rather than through `round()`, whose
        # banker's rounding sends an exactly-half-a-pixel half to zero and drops
        # it. At (170, -10, -170, 10) and a width of 1 that discarded the entire
        # western half and moved the request 10 degrees east of anything asked
        # for. A half now collapses only when its span is strictly under half a
        # pixel, which is the documented rule.
        west_columns = int((west_half[2] - west_half[0]) / res + 0.5)
        east_columns = width - west_columns
        west_window = (
            (180.0 - west_columns * res, miny, 180.0, maxy),
            (west_columns, height),
        )
        east_window = (
            (-180.0, miny, -180.0 + east_columns * res, maxy),
            (east_columns, height),
        )
        if west_columns == 0:
            result = [east_window]
        elif east_columns == 0:
            result = [west_window]
        else:
            result = [west_window, east_window]
    return result


def _wms_descriptor(
    endpoint: str,
    layers: str,
    crs: str,
    image_format: str,
    version: str,
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    bands: int,
) -> str:
    """Build the GDAL ``<GDAL_WMS>`` service description for a ``GetMap`` window.

    ``<DataWindow>`` carries the request in the service CRS: upper-left / lower-
    right corners from ``bbox`` and the output ``SizeX`` / ``SizeY`` in pixels. GDAL
    handles the WMS 1.3.0 EPSG:4326 lat/lon axis swap internally, so the corners are
    always written x/y (lon/lat). The CRS element is ``<CRS>`` for WMS >= 1.3.0 and
    ``<SRS>`` for 1.1.1 and below, as GDAL's minidriver requires.
    """
    minx, miny, maxx, maxy = bbox
    width, height = size
    crs_tag = _crs_element_tag(version)
    return (
        "<GDAL_WMS>\n"
        '  <Service name="WMS">\n'
        f"    <Version>{_xml_escape(version)}</Version>\n"
        f"    <ServerUrl>{_xml_escape(endpoint)}</ServerUrl>\n"
        f"    <Layers>{_xml_escape(layers)}</Layers>\n"
        f"    <{crs_tag}>{_xml_escape(crs)}</{crs_tag}>\n"
        f"    <ImageFormat>{_xml_escape(image_format)}</ImageFormat>\n"
        "  </Service>\n"
        "  <DataWindow>\n"
        f"    <UpperLeftX>{minx}</UpperLeftX>\n"
        f"    <UpperLeftY>{maxy}</UpperLeftY>\n"
        f"    <LowerRightX>{maxx}</LowerRightX>\n"
        f"    <LowerRightY>{miny}</LowerRightY>\n"
        f"    <SizeX>{width}</SizeX>\n"
        f"    <SizeY>{height}</SizeY>\n"
        "  </DataWindow>\n"
        f"  <BandsCount>{bands}</BandsCount>\n"
        "</GDAL_WMS>\n"
    )


def _wmts_connection(endpoint: str, layer: str, tile_matrix_set: str | None) -> str:
    """Build the GDAL ``WMTS:`` connection string for one layer.

    ``endpoint`` is the WMTS ``GetCapabilities`` URL; GDAL fetches it and exposes
    each layer as ``WMTS:<url>,layer=<id>`` (optionally pinned to a
    ``tilematrixset``).
    """
    conn = f"WMTS:{endpoint},layer={layer}"
    if tile_matrix_set:
        conn += f",tilematrixset={tile_matrix_set}"
    return conn


def _open(connection: str, layer: str, hint: str) -> gdal.Dataset:
    """Open a WMS descriptor / WMTS connection with GDAL, classifying failures.

    Raises:
        WMSError: GDAL could not open the layer (server error, bad descriptor /
            connection, unknown layer, …).
    """
    return _open_network_dataset(
        connection, error=WMSError, subject=f"{hint} layer {layer!r}"
    )


def _available_wmts_layers(endpoint: str) -> list[str]:
    """List the layer ids a WMTS endpoint advertises (best-effort, for hints).

    Returns an empty list when the capabilities cannot be read; this only ever
    enriches an error message, so a failure here must not mask the original one.
    """
    try:
        caps = gdal.Open(f"WMTS:{endpoint}")
    except RuntimeError:
        caps = None
    if caps is None:
        return []
    # Enumerate the caps container's subdatasets through the shared builder (the same
    # surface Dataset.subdatasets exposes), then pull the layer id from each
    # ``WMTS:<url>,layer=<id>`` name. Split on the comma-prefixed ``,layer=`` key, not
    # a bare ``layer=`` that a query-string in the caps URL might also carry.
    layers = [
        sub.name.split(",layer=", 1)[1].split(",", 1)[0]
        for sub in subdatasets_of(caps)
        if ",layer=" in sub.name
    ]
    return sorted(set(layers))


def _translate_window(
    src: gdal.Dataset,
    projwin: list[float],
    layer: str,
    resolution: tuple[float, float] | None,
    resample: str,
) -> gdal.Dataset:
    """Crop the requested ``projWin`` out of a WMTS pyramid into MEM.

    When ``resolution`` is given GDAL reads from the matching overview level; when
    it is ``None`` the finest level is used. Either way the read is bounded by the
    shared pixel ceiling (:data:`~pyramids.base._coverage.MAX_PX`): a read that would
    exceed it — the finest level over a very wide bbox, or a fine resolution — is
    rejected before allocation rather than materialising an unbounded MEM raster.

    Raises:
        ValueError: the requested window exceeds the pixel ceiling.
        WMSError: GDAL could not produce a raster for the requested window.
    """
    # Bound the allocation: read_size is called purely for its ceiling check (it
    # raises ValueError past MAX_PX) and the returned (width, height) is intentionally
    # discarded. A native (resolution=None) read is sized from the source's own
    # resolution.
    _read_size(projwin, resolution or _native_resolution(src))
    window: dict[str, Any] = {"projWin": projwin, "resampleAlg": resample}
    if resolution is not None:
        window["xRes"], window["yRes"] = resolution
    return _translate_to_mem(
        src,
        error=WMSError,
        action="WMTS tile read",
        subject=repr(layer),
        **window,
    )


def _render_wms(src: gdal.Dataset, layers: str) -> gdal.Dataset:
    """Fetch the WMS ``GetMap`` window into MEM, classifying failures as WMSError.

    The ``GetMap`` HTTP request fires during :func:`gdal.Translate` (``gdal.Open``
    only parses the descriptor), so a server error / non-image body raises here —
    this wrapper turns that raw ``RuntimeError`` into the documented
    :class:`WMSError`, mirroring the WCS reader's ``_translate_window``.

    Raises:
        WMSError: GDAL could not render the requested window.
    """
    return _translate_to_mem(
        src, error=WMSError, action="WMS GetMap", subject=repr(layers)
    )


def _check_halves_concatenable(
    west: gdal.Dataset, east: gdal.Dataset, seam_offset: float
) -> None:
    """Assert that two seam-adjacent fetches really do tile one raster.

    Both halves were asked for over the same latitude band, at the same pixel
    size, meeting at the seam — this turns any violation of that into a clear
    error rather than a silently shifted stitch. It mirrors
    :func:`pyramids.dataset.engines.spatial._check_lon_halves_concatenable`, with
    the 360 degree seam offset generalised to ``seam_offset`` so a WMTS layer in a
    projected CRS is checked in its own units.

    Args:
        west: The pre-seam half.
        east: The post-seam half.
        seam_offset: The native-CRS distance from the -180 to the +180 meridian.

    Raises:
        ValueError: The halves differ in rows, band count, pixel size or top edge,
            or they do not meet at the seam.

    Examples:
        - Two halves that meet exactly at 180 pass, and the check returns nothing:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _check_halves_concatenable
            >>> west = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            >>> _ = west.SetGeoTransform((170.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> east = gdal.GetDriverByName("MEM").Create("", 10, 40, 1)
            >>> _ = east.SetGeoTransform((-180.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> _check_halves_concatenable(west, east, 360.0) is None
            True

            ```
        - A gap between them is caught rather than silently stitched, because the
          east half no longer starts where the west one ends:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _check_halves_concatenable
            >>> west = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            >>> _ = west.SetGeoTransform((170.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> east = gdal.GetDriverByName("MEM").Create("", 10, 40, 1)
            >>> _ = east.SetGeoTransform((-179.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> _check_halves_concatenable(west, east, 360.0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ValueError: antimeridian halves are 1.0 apart at the seam (over ha...

            ```
    """
    if west.RasterYSize != east.RasterYSize or west.RasterCount != east.RasterCount:
        raise ValueError(
            "antimeridian halves are not concatenable "
            f"(rows {west.RasterYSize}/{east.RasterYSize}, "
            f"bands {west.RasterCount}/{east.RasterCount})"
        )
    west_gt, east_gt = west.GetGeoTransform(), east.GetGeoTransform()
    cell_x, cell_y = abs(west_gt[1]), abs(west_gt[5])
    if abs(cell_x - abs(east_gt[1])) > 1e-6 * cell_x or abs(
        cell_y - abs(east_gt[5])
    ) > 1e-6 * abs(cell_y):
        raise ValueError(
            "antimeridian halves were rendered at different resolutions "
            f"({west_gt[1]}, {west_gt[5]}) vs ({east_gt[1]}, {east_gt[5]}); "
            "they cannot be stitched into one uniform grid"
        )
    if abs(west_gt[3] - east_gt[3]) > 0.5 * cell_y:
        raise ValueError(
            "antimeridian halves do not share a top edge "
            f"({west_gt[3]} vs {east_gt[3]}), so they are not row-aligned"
        )
    gap = abs((west_gt[0] + west.RasterXSize * west_gt[1]) - (east_gt[0] + seam_offset))
    if gap > 0.5 * cell_x:
        raise ValueError(
            f"antimeridian halves are {gap} apart at the seam (over half a "
            f"{cell_x} pixel), so they do not tile a continuous raster"
        )


def _merge_lon_halves(
    west: gdal.Dataset, east: gdal.Dataset, seam_offset: float
) -> gdal.Dataset:
    """Concatenate two seam-adjacent fetches into one ``MEM`` raster.

    The merged raster keeps the **west** half's geotransform, so longitude runs on
    past the seam (``170 .. 180`` then ``180 .. 190``) instead of jumping back to
    -180 mid-raster — the same convention
    :func:`pyramids.dataset.engines.spatial._stitch_lon_halves` gives a stitched
    :meth:`Dataset.crop`. The pixels are copied as raw bytes, and the west half's
    band descriptions, metadata, units, no-data value, colour interpretation and
    **colour table** are copied with them, so the same layer renders identically
    whether or not the request happened to cross the seam.

    Args:
        west: The pre-seam half.
        east: The post-seam half, placed immediately to its right.
        seam_offset: The native-CRS distance from the -180 to the +180 meridian,
            for the alignment check.

    Returns:
        gdal.Dataset: The stitched raster, ``west.RasterXSize + east.RasterXSize``
            columns wide.

    Raises:
        ValueError: The halves do not tile a continuous raster.

    Examples:
        - The stitch is as wide as both halves together and keeps the west half's
          origin, so longitude runs on past the seam instead of wrapping:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _merge_lon_halves
            >>> west = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            >>> _ = west.SetGeoTransform((170.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> east = gdal.GetDriverByName("MEM").Create("", 10, 40, 1)
            >>> _ = east.SetGeoTransform((-180.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> merged = _merge_lon_halves(west, east, 360.0)
            >>> merged.RasterXSize, merged.RasterYSize
            (30, 40)
            >>> merged.GetGeoTransform()[0]
            170.0

            ```
        - The east edge of the result is therefore past 180, which is what makes it
          one continuous raster rather than two:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _merge_lon_halves
            >>> west = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            >>> _ = west.SetGeoTransform((170.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> east = gdal.GetDriverByName("MEM").Create("", 10, 40, 1)
            >>> _ = east.SetGeoTransform((-180.0, 0.5, 0.0, 10.0, 0.0, -0.5))
            >>> merged = _merge_lon_halves(west, east, 360.0)
            >>> gt = merged.GetGeoTransform()
            >>> gt[0] + merged.RasterXSize * gt[1]
            185.0

            ```
    """
    _check_halves_concatenable(west, east, seam_offset)
    data_type = west.GetRasterBand(1).DataType
    merged = gdal.GetDriverByName("MEM").Create(
        "",
        west.RasterXSize + east.RasterXSize,
        west.RasterYSize,
        west.RasterCount,
        data_type,
    )
    merged.SetGeoTransform(west.GetGeoTransform())
    merged.SetProjection(west.GetProjection())
    merged.SetMetadata(west.GetMetadata())
    for index in range(1, west.RasterCount + 1):
        source, target = west.GetRasterBand(index), merged.GetRasterBand(index)
        target.SetColorInterpretation(source.GetColorInterpretation())
        # The palette in particular: a WMS is the rendered-map reader, so a
        # paletted image/png with bands=1 is an ordinary answer. Copying the
        # interpretation without the table is worse than copying neither -- the
        # band then declares itself GCI_PaletteIndex with nothing to look up.
        color_table = source.GetRasterColorTable()
        if color_table is not None:
            target.SetRasterColorTable(color_table)
        target.SetDescription(source.GetDescription())
        target.SetMetadata(source.GetMetadata())
        unit = source.GetUnitType()
        if unit:
            target.SetUnitType(unit)
        no_data = source.GetNoDataValue()
        if no_data is not None:
            target.SetNoDataValue(no_data)
    for part, x_offset in ((west, 0), (east, west.RasterXSize)):
        merged.WriteRaster(
            x_offset,
            0,
            part.RasterXSize,
            part.RasterYSize,
            part.ReadRaster(buf_type=data_type),
            buf_type=data_type,
        )
    return merged


def _collect_halves(fetch: Any, windows: list[Any], seam_offset: float) -> gdal.Dataset:
    """Fetch each window and stitch the result, closing every part it does not return.

    The ownership dance both readers need: one window is handed straight back (and
    must therefore *not* be closed), two are merged into a third (and must be).
    A failure part-way through closes whatever was already fetched.

    Args:
        fetch: Callable turning one window into a ``gdal.Dataset``.
        windows: The one or two windows to fetch, in west-to-east order.
        seam_offset: The native-CRS distance from the -180 to the +180 meridian.

    Returns:
        gdal.Dataset: The single fetch, or the stitched pair.

    Raises:
        ValueError: `windows` is empty, or two halves were fetched but do not tile
            a continuous raster.

    Examples:
        - One window is handed straight back, still open for the caller to use:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _collect_halves
            >>> def fetch(window):
            ...     part = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            ...     _ = part.SetGeoTransform((window, 0.5, 0.0, 10.0, 0.0, -0.5))
            ...     return part
            >>> only = _collect_halves(fetch, [170.0], 360.0)
            >>> only.RasterXSize
            20

            ```
        - Two windows come back as one stitched raster, twice as wide, on the west
          half's origin:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.dataset._wms import _collect_halves
            >>> def fetch(window):
            ...     part = gdal.GetDriverByName("MEM").Create("", 20, 40, 1)
            ...     _ = part.SetGeoTransform((window, 0.5, 0.0, 10.0, 0.0, -0.5))
            ...     return part
            >>> merged = _collect_halves(fetch, [170.0, -180.0], 360.0)
            >>> merged.RasterXSize
            40
            >>> merged.GetGeoTransform()[0]
            170.0

            ```
    """
    if not windows:
        raise ValueError("_collect_halves needs at least one window, got none")
    parts: list[gdal.Dataset] = []
    try:
        for window in windows:
            parts.append(fetch(window))
        if len(parts) == 1:
            result, parts = parts[0], []  # hand ownership to the caller
        else:
            result = _merge_lon_halves(parts[0], parts[1], seam_offset)
    finally:
        for part in parts:
            part.Close()
    return result


def _reproject_tail(ds: Dataset, output_crs: str | None, resample: str) -> Dataset:
    """Reproject the result into ``output_crs`` when one was requested.

    Only a CRS change happens here — the requested ``resolution`` is already
    applied upstream (WMS bakes it into the descriptor image size; WMTS applies it
    in the windowed :func:`gdal.Translate`), so ``cell_size`` is deliberately not
    passed: doing so would reinterpret the request/native-CRS resolution as an
    ``output_crs`` cell size (a unit mismatch across e.g. degrees → metres) and, on
    the WMTS path, resample the layer a second time.
    """
    if output_crs is not None:
        ds = ds.to_crs(output_crs, method=resample)
    return ds


def from_wms(
    dataset_cls: type[Dataset],
    endpoint: str,
    *,
    layers: str | list[str] | tuple[str, ...],
    bbox: tuple[float, float, float, float],
    crs: str,
    size: tuple[int, int] | None,
    resolution: float | tuple[float, float] | None,
    image_format: str,
    version: str,
    bands: int,
    output_crs: str | None,
    output: str | Path | None,
    resample: str,
    auth: tuple[str, str] | None,
    timeout: float,
) -> Dataset:
    """Render a WMS ``GetMap`` window and return a :class:`Dataset`.

    Private implementation; the public API is
    :meth:`pyramids.dataset.Dataset.from_wms`, which owns the argument defaults and
    forwards resolved values here (so they are not restated on this signature) —
    see that method for the full parameter documentation.

    Changed from the previous behaviour: a bbox crossing the antimeridian
    (``minx > maxx``) used to be refused as inverted, the way every ``minx > maxx``
    bbox was. It is now read as a wrap, matching :meth:`Dataset.crop` — two
    ``GetMap`` requests either side of the 180 degree seam, stitched into one
    raster that keeps the west half's geotransform (so its bbox reads e.g.
    ``170 .. 190``). :func:`_seam_windows` documents how the requested pixel width
    is divided; the stitched raster is exactly that wide, at one uniform
    resolution, with the seam snapped to the nearest pixel boundary (a window shift
    of at most half a pixel). An inverted bbox is therefore no longer caught here
    — only ``miny >= maxy`` still is.

    Raises:
        ValueError: ``bbox`` is malformed, ``layers`` is empty, ``size`` /
            ``resolution`` was not given exactly once, or ``bbox`` wraps but
            ``crs`` is projected (see :func:`~pyramids.base._coverage.check_seam_bbox`).
        WMSError: the server could not be reached or returned a non-raster body.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox, allow_antimeridian=True)
    window = (minx, miny, maxx, maxy)
    layers_value = _layers_value(layers)
    _check_seam_bbox(window, crs)
    windows = _seam_windows(window, _output_size(window, size, resolution))
    config = _gdal_http_config(auth, timeout)

    def fetch(request: Any) -> gdal.Dataset:
        """Render one ``(bbox, size)`` window through its own ``GetMap``."""
        descriptor = _wms_descriptor(
            endpoint,
            layers_value,
            crs,
            image_format,
            version,
            request[0],
            request[1],
            bands,
        )
        src = _open(descriptor, layers_value, "WMS")
        try:
            rendered = _render_wms(src, layers_value)
        finally:
            src = None
        return rendered

    with gdal.config_options(config):
        # 360.0: the seam offset is in degrees because `check_seam_bbox` has
        # already established that a wrapping request CRS is geographic, and a WMS
        # renders in the request CRS itself (no native-CRS detour, unlike WMTS).
        mem = _collect_halves(fetch, windows, 360.0)

    ds = dataset_cls(mem, access="write")
    ds = _reproject_tail(ds, output_crs, resample)
    if output is not None:
        ds.to_file(output)
    return ds


def from_wmts(
    dataset_cls: type[Dataset],
    endpoint: str,
    *,
    layer: str,
    bbox: tuple[float, float, float, float],
    crs: str,
    tile_matrix_set: str | None,
    resolution: float | tuple[float, float] | None,
    layer_crs: str | None,
    output_crs: str | None,
    output: str | Path | None,
    resample: str,
    auth: tuple[str, str] | None,
    timeout: float,
) -> Dataset:
    """Crop a WMTS layer to ``bbox`` and return a :class:`Dataset`.

    Private implementation; the public API is
    :meth:`pyramids.dataset.Dataset.from_wmts`, which forwards here and documents
    the parameters.

    Changed from the previous behaviour: a bbox crossing the antimeridian
    (``minx > maxx``) used to be refused as inverted. It is now read as a wrap,
    matching :meth:`Dataset.crop` — the pyramid is windowed either side of the 180
    degree seam and the two crops are stitched, keeping the west half's
    geotransform. A wrapping read pins both halves to one resolution (the layer's
    own when ``resolution`` is ``None``) so they land on a single grid; a
    non-wrapping read is untouched.

    Raises:
        ValueError: ``bbox`` is malformed, ``layer_crs`` cannot be interpreted, or
            ``bbox`` wraps but ``crs`` is projected (see :func:`~pyramids.base._coverage.check_seam_bbox`).
        WMSError: the server could not be reached, the layer is unknown, or the
            tile read failed.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox, allow_antimeridian=True)
    window = (minx, miny, maxx, maxy)
    _check_seam_bbox(window, crs)
    res = _resolution_pair(resolution)
    halves = _seam_halves(window)
    connection = _wmts_connection(endpoint, layer, tile_matrix_set)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        try:
            src = _open(connection, layer, "WMTS")
        except WMSError as exc:
            available = _available_wmts_layers(endpoint)
            if available and layer not in available:
                raise not_advertised("layer", layer, endpoint, available) from exc
            raise
        try:
            native_srs = _resolve_native_srs(src, layer_crs)
            # A split read pins both halves to one pixel size — left to itself each
            # gdal.Translate would size its half from the source independently, and
            # two grids that disagree cannot be stitched. A single-window read keeps
            # the previous `resolution=None` meaning (finest level) untouched.
            split = len(halves) > 1
            half_res = (res or _native_resolution(src)) if split else res

            def crop_half(half: Any) -> gdal.Dataset:
                """Window one ``west < east`` half out of the pyramid."""
                projwin = _native_projwin(half, crs, native_srs)
                return _translate_window(src, projwin, layer, half_res, resample)

            # The seam offset is only measured when there is a seam to check: the
            # whole-world transform it needs is meaningless (and can be non-finite)
            # for a layer whose CRS does not span both meridians.
            offset = _seam_offset(window, crs, native_srs) if split else 0.0
            if split:
                # Drop a half that misses the pyramid, as the WCS and OGC API
                # Coverages readers do. A regional layer would otherwise return
                # that half as a block of no-data and concatenate it in as though
                # it were data. Only for a split read: a lone window keeps GDAL's
                # own lenient behaviour, unchanged.
                halves = [
                    half
                    for half in halves
                    if _window_overlaps(_native_projwin(half, crs, native_srs), src)
                ]
                if not halves:
                    raise ValueError(
                        f"bbox {bbox!r} crosses the antimeridian but neither half "
                        f"overlaps the extent of layer {layer!r}"
                    )
            mem = _collect_halves(crop_half, halves, offset)
        finally:
            src = None

    mem.SetSpatialRef(native_srs)
    ds = dataset_cls(mem, access="write")
    ds = _reproject_tail(ds, output_crs, resample)
    if output is not None:
        ds.to_file(output)
    return ds


def _resolve_native_srs(src: gdal.Dataset, layer_crs: str | None):
    """Resolve the WMTS layer's native CRS, re-branding CoverageError as WMSError."""
    try:
        return _resolve_native_srs_neutral(src, layer_crs)
    except CoverageError as exc:
        raise WMSError(str(exc)) from exc
