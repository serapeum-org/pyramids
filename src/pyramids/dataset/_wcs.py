"""OGC Web Coverage Service (WCS) → :class:`~pyramids.dataset.Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_wcs`. It fetches a
coverage subset from an OGC WCS server and returns a single-raster
:class:`~pyramids.dataset.Dataset`.

The transport is **GDAL's native WCS driver** — no ``owslib``, no ``rasterio``.
GDAL performs ``GetCapabilities`` / ``DescribeCoverage``, negotiates the WCS
version, and issues the version-correct ``GetCoverage`` (the ``1.0.0`` ``bbox`` +
``resx/resy`` form versus the ``2.0.x`` ``subsets`` + ``scaling`` form). pyramids
adds the two things the driver does *not* handle on its own:

* **A CRS shim.** Some servers (e.g. ISRIC SoilGrids) advertise a coverage CRS
  under an authority code that the local PROJ database does not know
  (``EPSG:152160`` — a custom Interrupted Goode Homolosine). GDAL then opens the
  coverage with *no* spatial reference and cannot reproject the request window.
  The caller supplies ``coverage_crs`` (a proj4 / WKT / authority string) and we
  attach it ourselves.
* **Client-side bbox reprojection.** The public API is lon/lat; we transform the
  query ``bbox`` into the coverage's native CRS with ``pyproj`` (already a core
  dependency) and hand GDAL a native-CRS window. This is what makes subsetting
  land on the right pixels even when the server only honours its native CRS.

Scope boundary (see ``docs/SCOPE.md``): this reader takes only generic OGC
inputs. Provider specifics — SoilGrids' ``map=/map/<property>.map`` URL scheme,
coverage-name catalogs, the ``EPSG:152160 == IGH`` fact, agency auth endpoints —
live in the downstream consumer (``earthlens``), which calls ``from_wcs`` and
passes ``coverage_crs`` / ``auth`` as needed.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from osgeo import gdal
from pyproj import CRS as _PyprojCRS
from pyproj.exceptions import CRSError as _PyprojCRSError

from pyramids.base._coverage import native_projwin as _native_projwin
from pyramids.base._coverage import resolution_pair as _resolution_pair
from pyramids.base._coverage import resolve_native_srs as _resolve_native_srs_neutral
from pyramids.base._coverage import validate_bbox as _validate_bbox
from pyramids.base._errors import CoverageError, WCSError
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config


def _http_get(url: str, auth: tuple[str, str] | None, timeout: float, what: str) -> bytes:
    """GET ``url`` (optional HTTP Basic auth), returning the raw body.

    Shared by the ``GetCapabilities`` (discovery) and direct ``GetCoverage``
    paths. Raises :class:`WCSError` on any transport-level failure so both call
    sites keep a uniform error contract.
    """
    opener = urllib.request.build_opener()
    if auth is not None:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, url, auth[0], auth[1])
        opener.add_handler(urllib.request.HTTPBasicAuthHandler(mgr))
    try:
        with opener.open(url, timeout=timeout) as resp:
            return resp.read()
    except OSError as exc:
        # urllib.error.URLError / HTTPError both derive from OSError.
        raise WCSError(f"WCS {what} request failed for {url!r}: {exc}") from exc

if TYPE_CHECKING:
    from osgeo import osr

    from pyramids.dataset.dataset import Dataset


def _resolve_native_srs(
    src: "gdal.Dataset", coverage_crs: str | None
) -> "osr.SpatialReference":
    """Resolve the coverage's native CRS, re-branding CoverageError as WCSError.

    Delegates to the shared, protocol-neutral resolver in
    :mod:`pyramids.base._coverage` and re-wraps its :class:`CoverageError`
    (CRS-less coverage, no ``coverage_crs`` shim) into :class:`WCSError`, so
    :meth:`pyramids.dataset.Dataset.from_wcs` keeps its documented error contract.
    A bad ``coverage_crs`` still raises :class:`ValueError` from the shared resolver.
    """
    try:
        return _resolve_native_srs_neutral(src, coverage_crs)
    except CoverageError as exc:
        raise WCSError(str(exc)) from exc


def _localname(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag (`{ns}Name` → `Name`)."""
    return tag.rsplit("}", 1)[-1]


def _capabilities_url(endpoint: str, version: str | None) -> str:
    """Build the ``GetCapabilities`` URL for a WCS endpoint."""
    sep = "&" if "?" in endpoint else "?"
    url = f"{endpoint}{sep}SERVICE=WCS&REQUEST=GetCapabilities"
    if version:
        url += f"&VERSION={version}"
    return url


@lru_cache(maxsize=32)
def _get_capabilities(
    endpoint: str, version: str | None, auth: tuple[str, str] | None, timeout: float
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Fetch and parse ``GetCapabilities`` once per endpoint (LRU-cached).

    Returns the advertised ``(versions, coverage_ids)``. A repeated call with the
    same arguments is served from the cache, so it costs no extra network round
    trip — this is the capabilities cache the design calls for.

    Raises:
        WCSError: The request failed at the transport level, or the server
            answered with an ``<ows:ExceptionReport>`` / non-XML body.
    """
    url = _capabilities_url(endpoint, version)
    payload = _http_get(url, auth, timeout, "GetCapabilities")

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise WCSError(
            f"WCS GetCapabilities returned a non-XML body from {endpoint!r}: {exc}"
        ) from exc

    if _localname(root.tag) in ("ExceptionReport", "ServiceExceptionReport"):
        raise WCSError(f"WCS server returned an exception for {endpoint!r}: {_exception_text(root)}")

    versions = {root.attrib["version"]} if root.attrib.get("version") else set()
    for el in root.iter():
        if _localname(el.tag) == "ServiceTypeVersion" and el.text:
            versions.add(el.text.strip())
    return tuple(sorted(versions)), frozenset(_extract_coverages(root))


def _extract_coverages(root: ET.Element) -> set[str]:
    """Collect coverage identifiers from a ``GetCapabilities`` document.

    WCS 2.0.x / 1.1.x advertise coverages as ``CoverageId`` / ``Identifier``.
    WCS 1.0.0 uses ``<name>`` — but only the ``<name>`` *inside a coverage
    offering* is an identifier; the ``<Service><name>`` is the service title, so
    a blind ``name`` sweep would wrongly admit it. We therefore collect ``name``
    only when its parent is a ``CoverageOfferingBrief`` / ``CoverageOffering``.
    """
    coverages: set[str] = set()
    for el in root.iter():
        if _localname(el.tag) in ("CoverageId", "Identifier") and el.text:
            coverages.add(el.text.strip())
    if coverages:
        return coverages
    for parent in root.iter():
        if _localname(parent.tag) not in ("CoverageOfferingBrief", "CoverageOffering"):
            continue
        for child in parent:
            if _localname(child.tag) == "name" and child.text:
                coverages.add(child.text.strip())
    return coverages


def _exception_text(root: ET.Element) -> str:
    """Extract the human-readable message from an OWS/WCS exception document."""
    for el in root.iter():
        if _localname(el.tag) in ("ExceptionText", "ServiceException") and el.text:
            return el.text.strip()
    return (root.text or "").strip() or "no message provided"


def _service_descriptor(
    endpoint: str,
    coverage: str,
    version: str | None,
    wcs_format: str | None,
    extra_params: dict[str, str] | None,
) -> str:
    """Build the GDAL ``<WCS_GDAL>`` service-description XML for one coverage."""
    lines = [
        "<WCS_GDAL>",
        f"  <ServiceURL>{_xml_escape(endpoint)}</ServiceURL>",
        f"  <CoverageName>{_xml_escape(coverage)}</CoverageName>",
    ]
    if version:
        lines.append(f"  <Version>{_xml_escape(version)}</Version>")
    if wcs_format:
        lines.append(f"  <PreferredFormat>{_xml_escape(wcs_format)}</PreferredFormat>")
    if extra_params:
        extra = "".join(f"&{k}={v}" for k, v in extra_params.items())
        lines.append(f"  <GetCoverageExtra>{_xml_escape(extra)}</GetCoverageExtra>")
    lines.append("</WCS_GDAL>")
    return "\n".join(lines) + "\n"


def _xml_escape(text: str) -> str:
    """Minimal XML escaping for descriptor text nodes."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _open_service(descriptor: str, coverage: str) -> "gdal.Dataset":
    """Open a WCS service descriptor with GDAL, classifying failures.

    Raises:
        WCSError: GDAL could not open the coverage (server error, bad descriptor,
            unresolvable CRS, …).
    """
    try:
        src = gdal.Open(descriptor)
    except RuntimeError as exc:
        raise WCSError(f"could not open WCS coverage {coverage!r}: {exc}") from exc
    if src is None:
        raise WCSError(f"GDAL returned no dataset for WCS coverage {coverage!r}")
    return src


def _default_subset_axes(crs: str) -> tuple[str, str]:
    """Best-effort WCS 2.0 ``SUBSET`` axis labels for ``crs``.

    Geographic CRSs get ``("Long", "Lat")``; everything else ``("X", "Y")``.
    There is no ``DescribeCoverage`` in direct mode to learn the coverage's real
    axis names from, so a server that labels them differently needs an explicit
    ``subset_axes`` override.
    """
    try:
        is_geographic = _PyprojCRS.from_user_input(crs).is_geographic
    except (_PyprojCRSError, ValueError, TypeError):
        is_geographic = str(crs).strip().upper().endswith(":4326")
    return ("Long", "Lat") if is_geographic else ("X", "Y")


def _getcoverage_url(
    endpoint: str,
    coverage: str,
    crs: str,
    bbox: tuple[float, float, float, float],
    version: str | None,
    wcs_format: str | None,
    resolution: float | tuple[float, float] | None,
    subset_axes: tuple[str, str] | None,
    extra_params: dict[str, str] | None,
) -> str:
    """Build a direct KVP ``GetCoverage`` URL (no capabilities / DescribeCoverage).

    Supports WCS ``2.0.x`` (``COVERAGEID`` + ``SUBSET`` + ``SUBSETTINGCRS`` — the
    default when ``version`` is omitted) and ``1.0.0`` (``COVERAGE`` + ``CRS`` +
    ``BBOX`` + ``RESX``/``RESY``). ``bbox`` is ``(minx, miny, maxx, maxy)`` in
    ``crs``; for ``2.0.x`` the first ``subset_axes`` label takes the x (min-x,
    max-x) range and the second the y range, so lon/lat values land on the right
    axis regardless of the CRS's declared axis order.

    Raises:
        ValueError: the WCS version is unsupported for direct mode, or ``1.0.0``
            was requested without a ``resolution`` (needed for the output grid).
    """
    minx, miny, maxx, maxy = bbox
    ver = version or "2.0.0"
    params: list[tuple[str, str]] = [
        ("SERVICE", "WCS"),
        ("VERSION", ver),
        ("REQUEST", "GetCoverage"),
    ]
    if ver.startswith("2."):
        x_axis, y_axis = subset_axes or _default_subset_axes(crs)
        params += [
            ("COVERAGEID", coverage),
            ("SUBSET", f"{x_axis}({minx},{maxx})"),
            ("SUBSET", f"{y_axis}({miny},{maxy})"),
            ("SUBSETTINGCRS", crs),
        ]
    elif ver.startswith("1.0"):
        res = _resolution_pair(resolution)
        if res is None:
            raise ValueError(
                "direct WCS 1.0.0 GetCoverage needs an output grid; pass "
                "resolution=... (mapped to RESX/RESY)."
            )
        params += [
            ("COVERAGE", coverage),
            ("CRS", crs),
            ("BBOX", f"{minx},{miny},{maxx},{maxy}"),
            ("RESX", str(res[0])),
            ("RESY", str(res[1])),
        ]
    else:
        raise ValueError(
            f"direct GetCoverage supports WCS 1.0.0 and 2.0.x; got {ver!r}. "
            "Use discovery mode (direct=False) for other versions."
        )
    if wcs_format:
        params.append(("FORMAT", wcs_format))
    if extra_params:
        params += list(extra_params.items())
    # Keep ',():/' literal so CRS shorthand / URIs (EPSG:4326,
    # http://www.opengis.net/def/crs/…) and SUBSET syntax survive intact — quirky
    # shim servers often string-match these rather than percent-decode.
    query = "&".join(
        f"{key}={urllib.parse.quote(str(val), safe=',():/')}" for key, val in params
    )
    sep = "&" if "?" in endpoint else "?"
    return f"{endpoint}{sep}{query}"


def _open_getcoverage_bytes(payload: bytes, coverage: str) -> "gdal.Dataset":
    """Read a direct ``GetCoverage`` response into an in-memory raster.

    A non-raster body (an ``<ows:ExceptionReport>``) is caught before it can be
    opened as a file the caller sees: an XML exception raises :class:`WCSError`
    with the server message; otherwise the bytes are materialised in ``/vsimem``,
    opened, copied into ``MEM``, and the temp is unlinked.

    Raises:
        WCSError: the body is an exception report / non-raster, or GDAL could not
            read it as a raster.
    """
    if payload.lstrip()[:1] == b"<":
        try:
            root: ET.Element | None = ET.fromstring(payload)
        except ET.ParseError:
            root = None
        if root is not None and _localname(root.tag) in (
            "ExceptionReport",
            "ServiceExceptionReport",
        ):
            raise WCSError(
                f"WCS server returned an exception for {coverage!r}: {_exception_text(root)}"
            )
        raise WCSError(f"WCS GetCoverage returned a non-raster body for {coverage!r}")
    vsipath = f"/vsimem/wcs_getcoverage_{uuid.uuid4().hex}.tif"
    gdal.FileFromMemBuffer(vsipath, payload)
    try:
        src = gdal.Open(vsipath)
        mem = (
            gdal.Translate("", src, options=gdal.TranslateOptions(format="MEM"))
            if src is not None
            else None
        )
    except RuntimeError as exc:
        # GDAL raises (rather than returning None) on a bad file when exceptions
        # are enabled — an HTML error page, truncated body, etc.
        raise WCSError(
            f"WCS GetCoverage returned no raster for {coverage!r}: {exc}"
        ) from exc
    finally:
        gdal.Unlink(vsipath)
    if mem is None:
        raise WCSError(f"WCS GetCoverage returned no raster for {coverage!r}")
    return mem


def _from_wcs_direct(
    dataset_cls: type["Dataset"],
    endpoint: str,
    coverage: str,
    bbox: tuple[float, float, float, float],
    crs: str,
    version: str | None,
    wcs_format: str | None,
    resolution: float | tuple[float, float] | None,
    subset_axes: tuple[str, str] | None,
    coverage_crs: str | None,
    auth: tuple[str, str] | None,
    timeout: float,
    extra_params: dict[str, str] | None,
) -> tuple["Dataset", str | None]:
    """Direct ``GetCoverage`` path: build the KVP request, fetch, wrap.

    Returns ``(ds, native_wkt)`` so the shared finalize step can resample within
    the coverage's own CRS when only a ``resolution`` was requested. ``native_wkt``
    is ``None`` when the returned raster carries no CRS and no ``coverage_crs``
    shim was supplied.
    """
    url = _getcoverage_url(
        endpoint, coverage, crs, bbox, version, wcs_format, resolution,
        subset_axes, extra_params,
    )
    payload = _http_get(url, auth, timeout, "GetCoverage")
    mem = _open_getcoverage_bytes(payload, coverage)
    if not mem.GetSpatialRef() and coverage_crs is not None:
        mem.SetSpatialRef(_resolve_native_srs(mem, coverage_crs))
    ds = dataset_cls(mem, access="write")
    native = mem.GetSpatialRef()
    native_wkt = native.ExportToWkt() if native else None
    return ds, native_wkt


def _finalize(
    ds: "Dataset",
    output_crs: str | None,
    res: tuple[float, float] | None,
    resample: str,
    native_wkt: str | None,
    output: str | Path | None,
) -> "Dataset":
    """Apply the optional reproject/resample + write shared by both WCS paths.

    With ``output_crs`` set, reproject to it; with only ``res`` set, resample
    within the coverage's own CRS (``native_wkt``); otherwise leave the raster
    as fetched. Write to ``output`` last, only after a valid raster exists.
    """
    if output_crs is not None:
        target: str | None = output_crs
    elif res and native_wkt is not None:
        target = native_wkt
    else:
        target = None
    if target is not None:
        ds = ds.to_crs(target, method=resample, cell_size=res)
    if output is not None:
        ds.to_file(output)
    return ds


def from_wcs(
    dataset_cls: type["Dataset"],
    endpoint: str,
    *,
    coverage: str,
    bbox: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    output_crs: str | None = None,
    resolution: float | tuple[float, float] | None = None,
    version: str | None = None,
    coverage_crs: str | None = None,
    wcs_format: str | None = None,
    output: str | Path | None = None,
    resample: str = "nearest",
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
    extra_params: dict[str, str] | None = None,
    direct: bool = False,
    subset_axes: tuple[str, str] | None = None,
) -> "Dataset":
    """Fetch a WCS coverage subset and return a :class:`Dataset`.

    This is the private implementation; the public API is the
    :meth:`pyramids.dataset.Dataset.from_wcs` classmethod, which forwards here.
    See that method for the full parameter documentation.

    With ``direct=False`` (default) the full OGC handshake runs: ``GetCapabilities``
    validates the coverage, then GDAL's WCS driver negotiates ``DescribeCoverage``
    before the windowed ``GetCoverage``. With ``direct=True`` both discovery steps
    are skipped and a KVP ``GetCoverage`` request is issued straight from the
    caller-supplied parameters — for ``GetCoverage``-only "WCS shim" endpoints that
    502/400 on capabilities/describe.

    Raises:
        ValueError: ``bbox`` is malformed, ``coverage`` is not advertised (discovery
            mode), ``coverage_crs`` cannot be interpreted, or (direct mode) the WCS
            version is unsupported / ``1.0.0`` lacks a ``resolution``.
        WCSError: The server could not be reached or returned an error / a
            non-raster body.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox)
    res = _resolution_pair(resolution)
    window = (minx, miny, maxx, maxy)

    if direct:
        ds, native_wkt = _from_wcs_direct(
            dataset_cls, endpoint, coverage, window, crs, version, wcs_format,
            resolution, subset_axes, coverage_crs, auth, timeout, extra_params,
        )
    else:
        _, coverages = _get_capabilities(endpoint, version, auth, timeout)
        if coverages and coverage not in coverages:
            raise ValueError(
                f"coverage {coverage!r} is not advertised by {endpoint!r}. "
                f"Available coverages: {sorted(coverages)[:10]}"
                + (" …" if len(coverages) > 10 else "")
            )
        descriptor = _service_descriptor(endpoint, coverage, version, wcs_format, extra_params)
        config = _gdal_http_config(auth, timeout)
        with gdal.config_options(config):
            src = _open_service(descriptor, coverage)
            native_srs = _resolve_native_srs(src, coverage_crs)
            projwin = _native_projwin(window, crs, native_srs)
            mem = _translate_window(src, projwin, coverage)
            src = None
        mem.SetSpatialRef(native_srs)
        ds = dataset_cls(mem, access="write")
        # WKT round-trips more faithfully than proj4 for exotic / compound CRS.
        native_wkt = native_srs.ExportToWkt()

    return _finalize(ds, output_crs, res, resample, native_wkt, output)


def _translate_window(
    src: "gdal.Dataset", projwin: list[float], coverage: str
) -> "gdal.Dataset":
    """Issue the windowed ``GetCoverage`` via :func:`gdal.Translate` into MEM.

    Translating into an in-memory dataset (never directly to the user's output
    path) guarantees an ``<ows:ExceptionReport>`` body can never be written to a
    ``.tif``: a non-raster response makes ``gdal.Translate`` fail and we raise
    :class:`WCSError` here, before any file is produced.

    Raises:
        WCSError: GDAL could not produce a raster for the requested window.
    """
    options = gdal.TranslateOptions(format="MEM", projWin=projwin)
    try:
        mem = gdal.Translate("", src, options=options)
    except RuntimeError as exc:
        raise WCSError(f"WCS GetCoverage failed for {coverage!r}: {exc}") from exc
    if mem is None:
        raise WCSError(f"WCS GetCoverage returned no raster for {coverage!r}")
    return mem
