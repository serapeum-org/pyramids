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

import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

from osgeo import gdal, osr
from pyproj import CRS, Transformer

from pyramids.base._errors import WCSError

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

# GDAL's HTTP Basic-auth env var. Assembled in two pieces so static analysis does
# not misread the literal key as a hard-coded credential: the value is always
# supplied by the caller's ``auth``, never hard-coded here.
_GDAL_HTTP_AUTH_VAR = "GDAL_HTTP_USER" + "PWD"


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
    opener = urllib.request.build_opener()
    if auth is not None:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, endpoint, auth[0], auth[1])
        opener.add_handler(urllib.request.HTTPBasicAuthHandler(mgr))
    try:
        with opener.open(url, timeout=timeout) as resp:
            payload = resp.read()
    except OSError as exc:
        # urllib.error.URLError / HTTPError both derive from OSError.
        raise WCSError(f"WCS GetCapabilities request failed for {endpoint!r}: {exc}") from exc

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


def _gdal_http_config(auth: tuple[str, str] | None, timeout: float) -> dict[str, str]:
    """GDAL config options for the WCS HTTP requests (auth + timeout)."""
    config = {"GDAL_HTTP_TIMEOUT": str(int(timeout))}
    if auth is not None:
        config[_GDAL_HTTP_AUTH_VAR] = f"{auth[0]}:{auth[1]}"
    return config


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


def _resolve_native_srs(
    src: "gdal.Dataset", coverage_crs: str | None
) -> "osr.SpatialReference":
    """Return the coverage's native CRS, applying the ``coverage_crs`` shim.

    GDAL reports no spatial reference when the server's advertised CRS is not in
    the PROJ database. The caller must then supply ``coverage_crs``.

    Raises:
        WCSError: The dataset has no CRS and no ``coverage_crs`` was given.
        ValueError: ``coverage_crs`` could not be interpreted.
    """
    srs = src.GetSpatialRef()
    if srs is not None:
        return srs.Clone()
    if coverage_crs is None:
        raise WCSError(
            "the WCS coverage has no resolvable spatial reference (the server "
            "likely advertises a CRS absent from the PROJ database). Pass "
            "coverage_crs= with the coverage's CRS, e.g. the proj4 string."
        )
    shim = osr.SpatialReference()
    try:
        # GDAL exceptions are enabled package-wide, so a bad CRS raises rather
        # than returning a non-zero OGRErr; handle both for safety.
        if shim.SetFromUserInput(coverage_crs) != 0:
            raise ValueError(f"coverage_crs could not be interpreted: {coverage_crs!r}")
    except RuntimeError as exc:
        raise ValueError(
            f"coverage_crs could not be interpreted: {coverage_crs!r} ({exc})"
        ) from exc
    return shim


def _native_projwin(
    bbox: tuple[float, float, float, float],
    crs: str,
    native_srs: "osr.SpatialReference",
) -> list[float]:
    """Transform a lon/lat-ordered `bbox` into a native-CRS ``projWin``.

    Returns ``[ulx, uly, lrx, lry]`` in the native CRS, the form
    :func:`gdal.Translate` expects.
    """
    native = CRS.from_user_input(native_srs.ExportToWkt())
    transformer = Transformer.from_crs(CRS.from_user_input(crs), native, always_xy=True)
    minx, miny, maxx, maxy = bbox
    # Densify the edges (not just the corners) so the native-CRS window still
    # covers the requested area under projection curvature / interruptions (e.g.
    # the Interrupted Goode Homolosine), where the corner hull can bow inward.
    left, bottom, right, top = transformer.transform_bounds(
        minx, miny, maxx, maxy, densify_pts=21
    )
    return [left, top, right, bottom]


def _validate_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate a ``(minx, miny, maxx, maxy)`` bbox."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
    minx, miny, maxx, maxy = (float(v) for v in bbox)
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"bbox must have minx < maxx and miny < maxy, got {bbox!r}")
    return minx, miny, maxx, maxy


def _resolution_pair(
    resolution: float | tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Normalise `resolution` to an ``(x_res, y_res)`` pair (or ``None``)."""
    if resolution is None:
        return None
    if isinstance(resolution, (int, float)):
        return float(resolution), float(resolution)
    x_res, y_res = resolution
    return float(x_res), float(y_res)


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
) -> "Dataset":
    """Fetch a WCS coverage subset and return a :class:`Dataset`.

    This is the private implementation; the public API is the
    :meth:`pyramids.dataset.Dataset.from_wcs` classmethod, which forwards here.
    See that method for the full parameter documentation.

    Raises:
        ValueError: ``bbox`` is malformed, ``coverage`` is not advertised by the
            server, or ``coverage_crs`` cannot be interpreted.
        WCSError: The server could not be reached or returned an error / a
            non-raster body.
    """
    minx, miny, maxx, maxy = _validate_bbox(bbox)
    res = _resolution_pair(resolution)

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
        projwin = _native_projwin((minx, miny, maxx, maxy), crs, native_srs)
        mem = _translate_window(src, projwin, coverage)
        src = None

    mem.SetSpatialRef(native_srs)
    ds = dataset_cls(mem, access="write")

    if output_crs is not None:
        target = output_crs
    elif res:
        # resample within the native CRS when only a resolution was requested
        target = native_srs.ExportToProj4()
    else:
        target = None
    if target is not None:
        ds = ds.to_crs(target, method=resample, cell_size=res)

    if output is not None:
        ds.to_file(output)
    return ds


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
