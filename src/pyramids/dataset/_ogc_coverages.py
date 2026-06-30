"""OGC API – Coverages → :class:`~pyramids.dataset.Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_ogc_coverages`. It
fetches a coverage subset from an OGC API – Coverages service and returns a
single-raster :class:`~pyramids.dataset.Dataset`.

OGC API – Coverages is the **modern REST/JSON successor to WCS**: a landing page
links to ``/collections``, each coverage is a collection exposing
``/collections/{id}/coverage`` (with ``subset`` query subsetting and format
negotiation). The transport here is **GDAL's native ``OGCAPI`` driver** — no
``owslib`` / ``requests``. The driver discovers the coverage, negotiates the
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

This is the OGC-API-era sibling of :mod:`pyramids.dataset._wcs` (the WCS reader),
whose bbox/CRS/window helpers it reuses, and it shares its ``/collections``
discovery with :mod:`pyramids.feature._oapif` (the OGC API – Features reader)
through :mod:`pyramids.base._ogc_api`.

Scope boundary (see ``docs/SCOPE.md``): this reader takes only generic OGC
inputs. Provider specifics — coverage-name catalogs, agency auth endpoints,
non-PROJ CRS — live in the downstream consumer (``earthlens``), which calls
``from_ogc_coverages`` and passes ``auth`` as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit, urlunsplit

from osgeo import gdal

from pyramids.base._errors import OGCAPIError, WCSError
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config
from pyramids.base._ogc_api import get_collections as _get_collections
from pyramids.dataset._wcs import (
    _native_projwin,
    _resolution_pair,
    _resolve_native_srs,
    _validate_bbox,
)

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

# Default cap on the longer side of a bbox-only read (no resolution given), in
# pixels. Keeps an otherwise unbounded coverage read to a manageable preview while
# preserving the bbox aspect ratio.
_DEFAULT_MAX_PX = 1024

# Hard ceiling on either side of the read, enforced even when a `resolution` is
# given, so a fine resolution over a wide bbox cannot request an unbounded read.
_MAX_PX = 25000

_OPEN_OPTIONS = ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF", "CACHE=NO"]


def _coverage_connection(endpoint: str, coverage: str) -> str:
    """Build the GDAL ``OGCAPI:`` connection string for one coverage collection.

    The ``/collections/{coverage}`` path segment is inserted **before** any
    existing query string (mirroring :func:`pyramids.base._ogc_api.collections_url`)
    so a query-string-auth endpoint (e.g. ``https://host/ogc?api_key=…``) keeps its
    query intact instead of producing ``…?api_key=…/collections/{coverage}``. The
    coverage identifier is URL-encoded so a name containing ``/`` or other reserved
    characters lands as a single path segment.
    """
    parts = urlsplit(endpoint)
    path = f"{parts.path.rstrip('/')}/collections/{quote(coverage, safe='')}"
    base = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
    return f"OGCAPI:{base}"


def _open_coverage(connection: str, coverage: str) -> "gdal.Dataset":
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
    try:
        src = gdal.OpenEx(connection, gdal.OF_RASTER, open_options=_OPEN_OPTIONS)
    except RuntimeError as exc:
        raise OGCAPIError(f"could not open OGC API coverage {coverage!r}: {exc}") from exc
    if src is None:
        raise OGCAPIError(f"GDAL returned no dataset for OGC API coverage {coverage!r}")
    return src


def _read_size(
    projwin: list[float], res: tuple[float, float] | None
) -> tuple[int, int]:
    """Compute the ``(width, height)`` pixel cap for the windowed read.

    ``projwin`` is ``[ulx, uly, lrx, lry]`` in the native CRS; its span gives the
    window's extent in that CRS's units. With a ``resolution`` the size follows
    directly (``span / res``); without one the longer side is capped at
    :data:`_DEFAULT_MAX_PX` and the shorter scaled to preserve the aspect ratio.
    Either way every dimension is clamped to at least 1 and rejected above the hard
    :data:`_MAX_PX` ceiling, so even a fine ``resolution`` over a wide ``bbox``
    cannot request an unbounded read.

    Raises:
        ValueError: the requested window exceeds :data:`_MAX_PX` on either side.
    """
    ulx, uly, lrx, lry = projwin
    span_x = abs(lrx - ulx)
    span_y = abs(uly - lry)
    if res is not None:
        x_res, y_res = res
        width = max(1, round(span_x / x_res)) if x_res else _DEFAULT_MAX_PX
        height = max(1, round(span_y / y_res)) if y_res else _DEFAULT_MAX_PX
    elif span_x >= span_y:
        width = _DEFAULT_MAX_PX
        height = max(1, round(_DEFAULT_MAX_PX * span_y / span_x)) if span_x else _DEFAULT_MAX_PX
    else:
        height = _DEFAULT_MAX_PX
        width = max(1, round(_DEFAULT_MAX_PX * span_x / span_y)) if span_y else _DEFAULT_MAX_PX
    if width > _MAX_PX or height > _MAX_PX:
        raise ValueError(
            f"the requested window is {width}x{height} px (over the {_MAX_PX} px limit); "
            "pass a coarser resolution or a smaller bbox to keep the read bounded"
        )
    return width, height


def _translate_window(
    src: "gdal.Dataset", projwin: list[float], size: tuple[int, int], coverage: str
) -> "gdal.Dataset":
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
    options = gdal.TranslateOptions(
        format="MEM", projWin=projwin, width=width, height=height
    )
    try:
        mem = gdal.Translate("", src, options=options)
    except RuntimeError as exc:
        raise OGCAPIError(f"OGC API coverage read failed for {coverage!r}: {exc}") from exc
    if mem is None:
        raise OGCAPIError(f"OGC API coverage read returned no raster for {coverage!r}")
    return mem


def from_ogc_coverages(
    dataset_cls: type["Dataset"],
    endpoint: str,
    *,
    coverage: str,
    bbox: tuple[float, float, float, float],
    output_crs: str | None = None,
    resolution: float | tuple[float, float] | None = None,
    output: str | Path | None = None,
    resample: str = "nearest",
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> "Dataset":
    """Fetch an OGC API – Coverages coverage subset and return a :class:`Dataset`.

    This is the private implementation; the public API is the
    :meth:`pyramids.dataset.Dataset.from_ogc_coverages` classmethod, which
    forwards here. See that method for the full parameter documentation.

    ``bbox`` is **required**: the ``OGCAPI`` coverage driver exposes the coverage as
    an unbounded virtual raster, so a windowless read is impossible. The lon/lat
    (CRS84) ``bbox`` is projected into the coverage's native CRS and read with a
    size cap so the fetch stays bounded.

    Raises:
        ValueError: ``bbox`` is malformed, or ``coverage`` is not advertised by the
            service.
        OGCAPIError: The ``OGCAPI`` driver is unavailable, the service could not be
            reached, or it returned an error / a non-raster body.
    """
    box = _validate_bbox(bbox)
    res = _resolution_pair(resolution)

    collections = _get_collections(endpoint, auth, timeout)
    if collections and coverage not in collections:
        raise ValueError(
            f"coverage {coverage!r} is not advertised by {endpoint!r}. "
            f"Available coverages: {sorted(collections)[:10]}"
            + (" …" if len(collections) > 10 else "")
        )

    connection = _coverage_connection(endpoint, coverage)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        src = _open_coverage(connection, coverage)
        try:
            # _resolve_native_srs is shared with the WCS reader; normalise its
            # WCSError (CRS-less coverage) to this reader's OGCAPIError so the
            # documented Raises contract holds and the message names OGC API.
            native_srs = _resolve_native_srs(src, None)
        except WCSError as exc:
            raise OGCAPIError(
                f"OGC API coverage {coverage!r} has no resolvable spatial reference; "
                "the service advertised no usable CRS for the coverage"
            ) from exc
        projwin = _native_projwin(box, "EPSG:4326", native_srs)
        size = _read_size(projwin, res)
        mem = _translate_window(src, projwin, size, coverage)
        src = None

    mem.SetSpatialRef(native_srs)
    ds = dataset_cls(mem, access="write")

    if output_crs is not None:
        ds = ds.to_crs(output_crs, method=resample)

    if output is not None:
        ds.to_file(output)
    return ds
