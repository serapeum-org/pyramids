"""OGC Web Coverage Service (WCS) → :class:`~pyramids.dataset.Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_wcs`. It fetches a
coverage subset from an OGC WCS server and returns a single-raster
:class:`~pyramids.dataset.Dataset`.

The default transport is **GDAL's native WCS driver** — no ``owslib``, no
``rasterio``. GDAL performs ``GetCapabilities`` / ``DescribeCoverage``, negotiates
the WCS version, and issues the version-correct ``GetCoverage`` (the ``1.0.0``
``bbox`` + ``resx/resy`` form versus the ``2.0.x`` ``subsets`` + ``scaling``
form). pyramids adds the two things the driver does *not* handle on its own:

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

For ``GetCoverage``-only "shim" servers that ``502``/``400`` on capabilities /
describe, ``from_wcs(..., direct=True)`` bypasses GDAL's driver entirely and
issues a KVP ``GetCoverage`` built here (:func:`_getcoverage_url`), reading the
returned bytes into a raster (:func:`_open_getcoverage_bytes`).

Scope boundary (see ``docs/SCOPE.md``): this reader takes only generic OGC
inputs. Provider specifics — SoilGrids' ``map=/map/<property>.map`` URL scheme,
coverage-name catalogs, the ``EPSG:152160 == IGH`` fact, agency auth endpoints —
live in the downstream consumer (``earthlens``), which calls ``from_wcs`` and
passes ``coverage_crs`` / ``auth`` as needed.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast
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
from pyramids.base._ogc_api import read_http_error as _read_http_error

# Cap on how much of an HTTP-error body is inlined into a WCSError message; the
# full body is still carried on WCSError.response_body. Keeps a multi-KB HTML
# error page from bloating the exception text.
_ERROR_BODY_CHARS = 500

# Request KVP keys pyramids sets itself in direct GetCoverage.
_RESERVED_KVP_KEYS = frozenset(
    {
        "SERVICE", "VERSION", "REQUEST", "COVERAGE", "COVERAGEID", "SUBSET",
        "SUBSETTINGCRS", "CRS", "BBOX", "RESX", "RESY", "FORMAT",
    }
)

# The fixed protocol call itself (SUBSET is multi-valued): an extra_params key
# targeting one of these is always a mistake, so it stays rejected even in direct
# mode. Every other reserved key is overridable — a caller can hand a
# non-conformant shim its exact spelling (see _merge_direct_kvp).
_PROTOCOL_KVP_KEYS = frozenset({"SERVICE", "VERSION", "REQUEST", "SUBSET"})
_OVERRIDABLE_KVP_KEYS = _RESERVED_KVP_KEYS - _PROTOCOL_KVP_KEYS

# Two request parameters are spelled differently across WCS versions: the CRS is
# SUBSETTINGCRS in 2.0.x and CRS in 1.0.0, and the coverage id is COVERAGEID in
# 2.0.x and COVERAGE in 1.0.0. A shim may want the other spelling on either
# version, so each pair collapses to one logical slot — overriding with either
# spelling replaces whichever the builder emitted (see _kvp_slot).
_CRS_KVP_KEYS = frozenset({"CRS", "SUBSETTINGCRS"})
_COVERAGE_KVP_KEYS = frozenset({"COVERAGE", "COVERAGEID"})


if TYPE_CHECKING:
    from osgeo import osr

    from pyramids.dataset.dataset import Dataset


def _kvp_slot(key: str) -> str:
    """Normalised override slot for a KVP key.

    The two cross-version pairs collapse to one slot each so an override bridges the
    WCS spellings: ``CRS``/``SUBSETTINGCRS`` -> ``CRS`` and
    ``COVERAGE``/``COVERAGEID`` -> ``COVERAGE``. Every other key is its own slot.
    """
    upper = key.upper()
    if upper in _CRS_KVP_KEYS:
        slot = "CRS"
    elif upper in _COVERAGE_KVP_KEYS:
        slot = "COVERAGE"
    else:
        slot = upper
    return slot


def _merge_direct_kvp(
    params: list[tuple[str, str]], extra_params: dict[str, str] | None
) -> list[tuple[str, str]]:
    """Fold ``extra_params`` into the built-in direct-GetCoverage KVP list.

    In direct mode the caller owns the wire request. An ``extra_params`` key that
    matches a built-in KVP *replaces* it in place — case-insensitively, and with
    ``CRS`` and ``SUBSETTINGCRS`` sharing one slot — using the caller's exact key
    spelling and value. This lets a non-conformant shim be handed its exact tokens:
    a lowercase ``coverageID`` key, or the WCS-1.x ``CRS=`` on a WCS-2.0 request (the
    two quirks of the Copernicus EDO/GDO MapServer, which ``500``s on the spec
    spellings). A key that matches no built-in is appended verbatim, in caller order.
    ``SERVICE`` / ``VERSION`` / ``REQUEST`` / ``SUBSET`` stay locked — they are the
    fixed protocol call (or multi-valued), so overriding them is always an error;
    additional WCS-2.0 ``SUBSET`` axes (e.g. a temporal subset) therefore cannot be
    added in direct mode, only in discovery mode.

    Args:
        params: The KVP pairs the builder assembled, in order.
        extra_params: Caller-supplied query parameters, or ``None``.

    Returns:
        The merged KVP pairs: built-ins with any matching override substituted in
        place, followed by the non-colliding extras in caller order.

    Raises:
        ValueError: an ``extra_params`` key targets a locked protocol parameter, or
            two ``extra_params`` keys target the same built-in GetCoverage parameter
            (e.g. both ``CRS`` and ``SUBSETTINGCRS``, or case-variant duplicates of a
            built-in key such as ``coverageID`` and ``COVERAGEID``).
    """
    result = list(params)
    if extra_params:
        builtin_slots = {_kvp_slot(key) for key, _ in result}
        overrides: dict[str, tuple[str, str]] = {}
        appended: list[tuple[str, str]] = []
        for key, val in extra_params.items():
            upper = key.upper()
            if upper in _PROTOCOL_KVP_KEYS:
                raise ValueError(
                    f"extra_params key {key!r} is a fixed WCS protocol parameter and "
                    "cannot be overridden; set it via the from_wcs arguments instead."
                )
            slot = _kvp_slot(key)
            if upper in _OVERRIDABLE_KVP_KEYS and slot in builtin_slots:
                if slot in overrides:
                    raise ValueError(
                        f"extra_params keys {overrides[slot][0]!r} and {key!r} both "
                        "target the same GetCoverage parameter; pass only one."
                    )
                overrides[slot] = (key, val)
            else:
                appended.append((key, val))
        # Substitute each override into its matching built-in position. Safe as a
        # single pass because no overridable slot is ever emitted twice in `params`:
        # the only duplicated built-in slot is SUBSET (two axes on 2.0.x), and SUBSET
        # is a protocol key that can never populate `overrides`. If a future builder
        # emits a duplicated overridable slot, dedupe here before substituting.
        merged = [overrides.get(_kvp_slot(k), (k, v)) for k, v in result]
        result = merged + appended
    return result


def _http_get(
    url: str, auth: tuple[str, str] | None, timeout: float, what: str
) -> bytes:
    """GET ``url`` (optional HTTP Basic auth), returning the raw body.

    Shared by the ``GetCapabilities`` (discovery) and direct ``GetCoverage``
    paths. Raises :class:`WCSError` on any transport-level failure so both call
    sites keep a uniform error contract. On a genuine HTTP-error status (4xx/5xx)
    the raised :class:`WCSError` also carries the ``status_code`` and the decoded
    ``response_body`` so a caller can inspect the server's explanation.
    """
    opener = urllib.request.build_opener()
    if auth is not None:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, url, auth[0], auth[1])
        opener.add_handler(urllib.request.HTTPBasicAuthHandler(mgr))
    try:
        with opener.open(url, timeout=timeout) as resp:
            return cast(bytes, resp.read())
    except urllib.error.HTTPError as exc:
        # A 4xx/5xx status carries a body with the server's real explanation
        # (non-spec shims return e.g. a JSON {"message": ...}); surface it in the
        # message and carry the status + full body on the exception so a caller
        # can branch on them. str(HTTPError) alone is only "HTTP <code>: <reason>".
        code, _raw, body = _read_http_error(exc)
        # Collapse whitespace so a multi-line HTML / pretty JSON error page stays a
        # single line in the message, and cap its length; the full, untouched body
        # is still carried on response_body below.
        shown = " ".join(body.split())
        if len(shown) > _ERROR_BODY_CHARS:
            shown = f"{shown[:_ERROR_BODY_CHARS]}…"
        raise WCSError(
            f"WCS {what} request failed for {url!r}: HTTP {code}: {shown}",
            status_code=code,
            response_body=body,
        ) from exc
    except OSError as exc:
        # urllib.error.URLError / timeout / connection reset — no HTTP response body.
        raise WCSError(f"WCS {what} request failed for {url!r}: {exc}") from exc


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


def _collect_coverage_ids(root: ET.Element) -> set[str]:
    """Collect WCS 2.0.x / 1.1.x ``CoverageId`` / ``Identifier`` values."""
    ids: set[str] = set()
    for el in root.iter():
        if _localname(el.tag) in ("CoverageId", "Identifier") and el.text:
            ids.add(el.text.strip())
    return ids


def _collect_wcs10_names(root: ET.Element) -> set[str]:
    """Collect WCS 1.0.0 ``<name>`` values inside coverage offerings only.

    The ``<name>`` inside a ``CoverageOfferingBrief`` / ``CoverageOffering`` is a
    coverage identifier; the ``<Service><name>`` is the service title, so a blind
    ``name`` sweep would wrongly admit it.
    """
    names: set[str] = set()
    for parent in root.iter():
        if _localname(parent.tag) not in ("CoverageOfferingBrief", "CoverageOffering"):
            continue
        for child in parent:
            if _localname(child.tag) == "name" and child.text:
                names.add(child.text.strip())
    return names


def _extract_coverages(root: ET.Element) -> set[str]:
    """Collect coverage identifiers from a ``GetCapabilities`` document.

    WCS 2.0.x / 1.1.x advertise coverages as ``CoverageId`` / ``Identifier``;
    WCS 1.0.0 uses ``<name>`` inside a coverage offering (see
    :func:`_collect_wcs10_names` for why the service title is excluded).
    """
    coverages = _collect_coverage_ids(root)
    if not coverages:
        coverages = _collect_wcs10_names(root)
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
        text = str(crs).strip().upper()
        is_geographic = text.endswith(":4326") or "CRS84" in text
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

    ``extra_params`` may override a built-in KVP by matching its key (see
    :func:`_merge_direct_kvp`), so a non-conformant shim can be served its exact
    spelling (e.g. ``{"coverageID": "spaST", "CRS": "EPSG:4326"}`` to send a
    lowercase key and the WCS-1.x CRS token on a WCS-2.0 request).

    Raises:
        ValueError: the WCS version is unsupported for direct mode, ``1.0.0`` was
            requested without a ``resolution`` (needed for the output grid), or an
            ``extra_params`` key targets a locked protocol parameter
            (``SERVICE`` / ``VERSION`` / ``REQUEST`` / ``SUBSET``).
    """
    minx, miny, maxx, maxy = bbox
    ver = version or "2.0.0"
    if re.fullmatch(r"\d+\.\d+\.\d+", ver) is None:
        raise ValueError(
            f"direct GetCoverage needs a full 'x.y.z' WCS version (e.g. '2.0.0' or "
            f"'1.0.0'); got {ver!r}."
        )
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
    params = _merge_direct_kvp(params, extra_params)
    # Encode both key and value so a stray '&'/'=' in a caller-supplied key cannot
    # split the query. Keep ',():/' literal in values so CRS shorthand / URIs
    # (EPSG:4326, http://www.opengis.net/def/crs/…) and SUBSET syntax survive intact
    # — quirky shim servers often string-match these rather than percent-decode.
    query = "&".join(
        f"{urllib.parse.quote(str(key), safe='')}="
        f"{urllib.parse.quote(str(val), safe=',():/')}"
        for key, val in params
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
    if payload[:64].lstrip()[:1] == b"<":
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
        raise WCSError(
            f"WCS GetCoverage returned an XML/GML body for {coverage!r} that direct "
            "mode cannot decode; request a plain binary raster via wcs_format=... "
            "(e.g. 'GEOTIFF')."
        )
    vsipath = f"/vsimem/wcs_getcoverage_{uuid.uuid4().hex}.tif"
    gdal.FileFromMemBuffer(vsipath, payload)
    try:
        src = gdal.Open(vsipath)
        mem = (
            gdal.Translate("", src, options=gdal.TranslateOptions(format="MEM"))
            if src is not None
            else None
        )
        src = None  # release the /vsimem handle before Unlink
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
    if (output_crs is not None or res) and native_wkt is None:
        raise WCSError(
            "reproject/resample requested but the returned coverage has no CRS to "
            "work from; pass coverage_crs=... so the raster carries a source CRS."
        )
    if output_crs is not None:
        target: str | None = output_crs
    elif res:
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
        # 1.0.0 direct sends RESX/RESY, so the server already grids to `res`; skip
        # the redundant client-side resample. 2.0.x has no request-side resolution,
        # so it resamples client-side in _finalize.
        finalize_res = None if (version or "2.0.0").startswith("1.0") else res
    else:
        _, coverages = _get_capabilities(endpoint, version, auth, timeout)
        if coverages and coverage not in coverages:
            raise ValueError(
                f"coverage {coverage!r} is not advertised by {endpoint!r}. "
                f"Available coverages: {sorted(coverages)[:10]}"
                + (" …" if len(coverages) > 10 else "")
            )
        descriptor = _service_descriptor(
            endpoint, coverage, version, wcs_format, extra_params
        )
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
        finalize_res = res

    return _finalize(ds, output_crs, finalize_res, resample, native_wkt, output)


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
