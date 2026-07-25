"""OGC Web Map Service (WMS) / Web Map Tile Service (WMTS) → :class:`Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_wms` and
:meth:`pyramids.dataset.Dataset.from_wmts`. Both pull an OGC *map* layer into a
single-raster :class:`~pyramids.dataset.Dataset` using **GDAL's native WMS / WMTS
drivers** — no ``owslib``, no ``rasterio`` — matching the WCS / WFS / OGC API
readers.

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

Scope boundary (see ``docs/SCOPE.md``): these readers take only generic OGC
inputs. Provider specifics — endpoint / layer catalogs, agency auth (NASA GIBS,
EUMETSAT), tile-matrix-set naming — live in the downstream consumer, which calls
``from_wms`` / ``from_wmts`` with a concrete endpoint and layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from osgeo import gdal

from pyramids.base._coverage import native_projwin as _native_projwin
from pyramids.base._coverage import native_resolution as _native_resolution
from pyramids.base._coverage import read_size as _read_size
from pyramids.base._coverage import resolution_pair as _resolution_pair
from pyramids.base._coverage import resolve_native_srs as _resolve_native_srs_neutral
from pyramids.base._coverage import validate_bbox as _validate_bbox
from pyramids.base._errors import CoverageError, WMSError
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config

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

    Raises:
        ValueError: both or neither of ``size`` / ``resolution`` were given, or
            ``size`` is not two positive integers.
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
        minx, miny, maxx, maxy = bbox
        result = (
            max(1, round((maxx - minx) / res[0])),
            max(1, round((maxy - miny) / res[1])),
        )
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
    try:
        src = gdal.Open(connection)
    except RuntimeError as exc:
        raise WMSError(f"could not open {hint} layer {layer!r}: {exc}") from exc
    if src is None:
        raise WMSError(f"GDAL returned no dataset for {hint} layer {layer!r}")
    return src


def _available_wmts_layers(endpoint: str) -> list[str]:
    """List the layer ids a WMTS endpoint advertises (best-effort, for hints).

    Returns an empty list when the capabilities cannot be read; this only ever
    enriches an error message, so a failure here must not mask the original one.
    """
    try:
        caps = gdal.Open(f"WMTS:{endpoint}")
    except RuntimeError:
        caps = None
    layers: list[str] = []
    if caps is not None:
        for key, value in caps.GetMetadata("SUBDATASETS").items():
            # Split on the comma-prefixed ``,layer=`` GDAL subdataset key, not a
            # bare ``layer=`` that a query-string in the caps URL might also carry.
            if key.endswith("_NAME") and ",layer=" in value:
                layers.append(value.split(",layer=", 1)[1].split(",", 1)[0])
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
    # Bound the allocation: reject a read that would exceed the pixel ceiling. A
    # native (resolution=None) read is sized from the source's own resolution.
    _read_size(projwin, resolution or _native_resolution(src))
    kwargs: dict = {"format": "MEM", "projWin": projwin, "resampleAlg": resample}
    if resolution is not None:
        kwargs["xRes"], kwargs["yRes"] = resolution
    try:
        mem = gdal.Translate("", src, options=gdal.TranslateOptions(**kwargs))
    except RuntimeError as exc:
        raise WMSError(f"WMTS tile read failed for {layer!r}: {exc}") from exc
    if mem is None:
        raise WMSError(f"WMTS tile read returned no raster for {layer!r}")
    return mem


def _render_wms(src: gdal.Dataset, layers: str) -> gdal.Dataset:
    """Fetch the WMS ``GetMap`` window into MEM, classifying failures as WMSError.

    The ``GetMap`` HTTP request fires during :func:`gdal.Translate` (``gdal.Open``
    only parses the descriptor), so a server error / non-image body raises here —
    this wrapper turns that raw ``RuntimeError`` into the documented
    :class:`WMSError`, mirroring the WCS reader's ``_translate_window``.

    Raises:
        WMSError: GDAL could not render the requested window.
    """
    try:
        mem = gdal.Translate("", src, options=gdal.TranslateOptions(format="MEM"))
    except RuntimeError as exc:
        raise WMSError(f"WMS GetMap failed for {layers!r}: {exc}") from exc
    if mem is None:
        raise WMSError(f"WMS GetMap returned no raster for {layers!r}")
    return mem


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

    Raises:
        ValueError: ``bbox`` is malformed, ``layers`` is empty, or ``size`` /
            ``resolution`` was not given exactly once.
        WMSError: the server could not be reached or returned a non-raster body.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox)
    layers_value = _layers_value(layers)
    width, height = _output_size((minx, miny, maxx, maxy), size, resolution)
    descriptor = _wms_descriptor(
        endpoint,
        layers_value,
        crs,
        image_format,
        version,
        (minx, miny, maxx, maxy),
        (width, height),
        bands,
    )
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        src = _open(descriptor, layers_value, "WMS")
        try:
            mem = _render_wms(src, layers_value)
        finally:
            src = None

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

    Raises:
        ValueError: ``bbox`` is malformed, or ``layer_crs`` cannot be interpreted.
        WMSError: the server could not be reached, the layer is unknown, or the
            tile read failed.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox)
    res = _resolution_pair(resolution)
    connection = _wmts_connection(endpoint, layer, tile_matrix_set)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        try:
            src = _open(connection, layer, "WMTS")
        except WMSError as exc:
            available = _available_wmts_layers(endpoint)
            if available and layer not in available:
                raise ValueError(
                    f"layer {layer!r} is not advertised by {endpoint!r}. "
                    f"Available layers: {available[:10]}"
                    + (" …" if len(available) > 10 else "")
                ) from exc
            raise
        try:
            native_srs = _resolve_native_srs(src, layer_crs)
            projwin = _native_projwin((minx, miny, maxx, maxy), crs, native_srs)
            mem = _translate_window(src, projwin, layer, res, resample)
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
