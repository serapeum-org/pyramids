"""OGC API – Coverages → :class:`~pyramids.dataset.Dataset`.

Implementation behind :meth:`pyramids.dataset.Dataset.from_ogc_coverages`. It
fetches a coverage subset from an OGC API – Coverages service and returns a
single-raster :class:`~pyramids.dataset.Dataset`.

OGC API – Coverages is the **modern REST/JSON successor to WCS**: a landing page
links to ``/collections``, each coverage is a collection exposing
``/collections/{id}/coverage`` (with ``bbox`` / ``subset`` query subsetting and
format negotiation). The transport here is **GDAL's native ``OGCAPI`` driver** —
no ``owslib`` / ``requests``. The driver discovers the coverage, negotiates the
GeoTIFF representation and issues the windowed read; the bytes decode through the
same GDAL raster path that backs :class:`Dataset`. pyramids adds, on top of the
driver, a cached ``/collections`` check so an unadvertised coverage fails fast
with a clear :class:`ValueError`, and so transport / driver failures surface as
:class:`~pyramids.base._errors.OGCAPIError`.

This is the OGC-API-era sibling of :mod:`pyramids.dataset._wcs` (the WCS reader)
and shares its ``/collections`` discovery with :mod:`pyramids.feature._oapif` (the
OGC API – Features reader) through :mod:`pyramids.base._ogc_api`.

Scope boundary (see ``docs/SCOPE.md``): this reader takes only generic OGC
inputs. Provider specifics — coverage-name catalogs, agency auth endpoints,
non-PROJ CRS — live in the downstream consumer (``earthlens``), which calls
``from_ogc_coverages`` and passes ``auth`` as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from osgeo import gdal

from pyramids.base._errors import OGCAPIError
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config
from pyramids.base._ogc_api import get_collections as _get_collections

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


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


def _coverage_connection(endpoint: str, coverage: str) -> str:
    """Build the GDAL ``OGCAPI:`` connection string for one coverage collection."""
    return f"OGCAPI:{endpoint.rstrip('/')}/collections/{coverage}"


def _open_options(bbox: tuple[float, float, float, float] | None) -> list[str]:
    """Assemble the GDAL ``OGCAPI`` open options for a coverage read.

    Always pins the coverage API and a GeoTIFF representation; a ``bbox`` (lon/lat
    / CRS84) becomes the ``MINX/MINY/MAXX/MAXY`` subset window the driver honours.
    """
    opts = ["API=COVERAGE", "IMAGE_FORMAT=GEOTIFF"]
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        opts += [f"MINX={minx}", f"MINY={miny}", f"MAXX={maxx}", f"MAXY={maxy}"]
    return opts


def _open_coverage(connection: str, opts: list[str], coverage: str) -> "gdal.Dataset":
    """Open an OGC API – Coverages coverage with GDAL, classifying failures.

    Raises:
        OGCAPIError: GDAL could not open the coverage (service error, bad
            representation, unresolvable CRS, …).
    """
    try:
        src = gdal.OpenEx(connection, gdal.OF_RASTER, open_options=opts)
    except RuntimeError as exc:
        raise OGCAPIError(f"could not open OGC API coverage {coverage!r}: {exc}") from exc
    if src is None:
        raise OGCAPIError(f"GDAL returned no dataset for OGC API coverage {coverage!r}")
    return src


def from_ogc_coverages(
    dataset_cls: type["Dataset"],
    endpoint: str,
    *,
    coverage: str,
    bbox: tuple[float, float, float, float] | None = None,
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

    Raises:
        ValueError: ``bbox`` is malformed, or ``coverage`` is not advertised by the
            service.
        OGCAPIError: The service could not be reached or returned an error / a
            non-raster body.
    """
    box = _validate_bbox(bbox) if bbox is not None else None
    res = _resolution_pair(resolution)

    collections = _get_collections(endpoint, auth, timeout)
    if collections and coverage not in collections:
        raise ValueError(
            f"coverage {coverage!r} is not advertised by {endpoint!r}. "
            f"Available coverages: {sorted(collections)[:10]}"
            + (" …" if len(collections) > 10 else "")
        )

    connection = _coverage_connection(endpoint, coverage)
    opts = _open_options(box)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        src = _open_coverage(connection, opts, coverage)
        native_wkt = src.GetSpatialRef().ExportToWkt() if src.GetSpatialRef() else None
        ds = dataset_cls(src, access="write")

    if output_crs is not None:
        target = output_crs
    elif res:
        # resample within the native CRS when only a resolution was requested;
        # WKT round-trips more faithfully than proj4 for exotic / compound CRS.
        target = native_wkt
    else:
        target = None
    if target is not None:
        ds = ds.to_crs(target, method=resample, cell_size=res)

    if output is not None:
        ds.to_file(output)
    return ds
