"""Input / constructor engine for :class:`~pyramids.feature.FeatureCollection` (ARC-36).

Module-level implementations of every way to *build* a FeatureCollection, kept out
of the god-class as free functions that take the `FeatureCollection` class (`fc_cls`)
or a collection (`fc`) as their first argument. The `FeatureCollection` methods are
thin facades over these functions; the full docstrings/doctests stay on the facades
(the public API). The symmetric output side lives in :mod:`pyramids.feature._write`.

Covers layer listing (with its LRU cache), the web readers (ArcGIS FeatureServer
pagination, GPX sub-layers), the file readers (vector files, GeoParquet, the
streaming ``iter_features`` / ``open_arrow`` paths) plus the eager/lazy backend
dispatch shared by ``read_file`` and ``read_parquet``, and the in-memory
constructors (``from_features``, ``from_bbox``, ``from_records``). The web readers
call back through the `fc_cls` facades (`fc_cls.read_file`,
`fc_cls._read_featureserver_page`) so existing tests that monkeypatch those class
methods still intercept.

``_LAZY_TARGET_BYTES_PER_PARTITION`` is the tunable knob behind
:func:`_resolve_lazy_partitioning`; :func:`pyramids.configure_lazy_vector` patches it
here (the input engine owns it).
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import geopandas as gpd
import pandas as pd
import pyogrio
import pyproj
from geopandas import GeoDataFrame
from osgeo import gdal
from pyproj.exceptions import CRSError as _PyprojCRSError
from shapely.geometry import box

from pyramids import _io as _pyramids_io
from pyramids.base._errors import FeatureError, VectorTileServerError
from pyramids.base._ogc_api import http_error_detail, http_get_with_retry
from pyramids.base._utils import import_pyarrow
from pyramids.base.crs import _pyproj_crs_via_gdal
from pyramids.base.remote import _ARCHIVE_MARKER_RE, is_remote, to_fsspec_url

if TYPE_CHECKING:
    from pyramids.feature._lazy_collection import LazyFeatureCollection
    from pyramids.feature.collection import FeatureCollection

_DEFAULT_ITER_BATCH_SIZE: int = 1000
_LAZY_TARGET_BYTES_PER_PARTITION: int = 128 * 1024 * 1024


@lru_cache(maxsize=128)
def _list_layers_cached(resolved_path: str) -> tuple[str, ...]:
    """Return a tuple of layer names for a resolved path (memoised)."""
    arr = pyogrio.list_layers(resolved_path)
    return tuple(str(row[0]) for row in arr)


def list_layers(path: str | Path) -> list[str]:
    """List every vector-layer name in `path`, memoised (see FeatureCollection.list_layers)."""
    path_str = str(path)
    if not is_remote(path_str):
        local = Path(path_str)
        if not local.exists():
            raise FileNotFoundError(f"list_layers: no file at {path_str!r}.")
    resolved = str(_pyramids_io._parse_path(path))
    return list(_list_layers_cached(resolved))


def list_layers_cache_clear() -> None:
    """Clear the LRU cache backing :func:`list_layers`."""
    _list_layers_cached.cache_clear()


def read_gpx_layers(
    fc_cls: type[FeatureCollection], path: str | Path
) -> dict[str, FeatureCollection]:
    """Read every non-empty GPX sub-layer into a dict (see FeatureCollection.read_gpx_layers)."""
    result: dict[str, Any] = {}
    for name in fc_cls.list_layers(path):
        fc = fc_cls.read_file(path, layer=name)
        if len(fc) > 0:
            result[name] = fc
    return result


def read_featureserver_page(
    fc_cls: type[FeatureCollection], page_url: str
) -> FeatureCollection:
    """Read one ESRIJSON page from an ArcGIS FeatureServer query URL."""
    return fc_cls.read_file(page_url)


def from_featureserver(
    fc_cls: type[FeatureCollection],
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    max_records: int | None = None,
    page_size: int = 1000,
    max_pages: int = 1000,
) -> FeatureCollection:
    """Read a paged ArcGIS FeatureServer layer (see FeatureCollection.from_featureserver)."""
    if page_size < 1:
        raise ValueError(f"from_featureserver: page_size must be >= 1, got {page_size}")
    if max_records is not None and max_records < 0:
        raise ValueError(
            f"from_featureserver: max_records must be >= 0 or None, got {max_records}"
        )
    base = url.split("?", 1)[0].rstrip("/")
    if not base.lower().endswith("/query"):
        base = f"{base}/query"
    pages, first_crs = collect_featureserver_pages(
        fc_cls, base, where, out_fields, max_records, page_size, max_pages
    )
    # Concatenate in one pass (pd.concat preserves the shared CRS); repeatedly calling .concat()
    # re-sets the CRS and trips a geopandas DeprecationWarning.
    if pages:
        return fc_cls(pd.concat(pages, ignore_index=True))
    return fc_cls(gpd.GeoDataFrame(geometry=[], crs=first_crs))


def collect_featureserver_pages(
    fc_cls: type[FeatureCollection],
    base: str,
    where: str,
    out_fields: str,
    max_records: int | None,
    page_size: int,
    max_pages: int,
) -> tuple[list[FeatureCollection], Any]:
    """Page through a FeatureServer /query endpoint; return (pages, first_crs)."""
    pages: list = []
    first_crs = None
    offset = 0
    fetched = 0
    page_index = 0
    while max_records is None or fetched < max_records:
        if page_index >= max_pages:
            warnings.warn(
                f"from_featureserver: stopped after {max_pages} pages (max_pages). The server may not "
                "honour resultOffset paging; raise max_pages or set max_records if more features are "
                "expected.",
                stacklevel=2,
            )
            break
        this_page = (
            page_size if max_records is None else min(page_size, max_records - fetched)
        )
        query = urlencode(
            {
                "where": where,
                "outFields": out_fields,
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": this_page,
            }
        )
        # Call the facade so tests that monkeypatch FeatureCollection._read_featureserver_page intercept.
        page = fc_cls._read_featureserver_page(f"{base}?{query}")
        if first_crs is None:
            first_crs = page.crs
        count = len(page)
        if count == 0:
            break
        pages.append(page)
        fetched += count
        offset += count
        page_index += 1
        if count < this_page:  # last (short) page
            break
    return pages, first_crs


# ArcGIS vector tiles use the Web Mercator (EPSG:3857) quad-tree by default. The
# origin is the top-left corner and the world spans +/- this many metres each way.
_WEBMERC_ORIGIN = 20037508.342789244
_WEBMERC_WKIDS = frozenset({3857, 102100, 102113, 900913})
_VTS_USER_AGENT = "pyramids-gis VectorTileServer client"


def _vts_base_and_query(url: str) -> tuple[str, str]:
    """Split a VectorTileServer URL into its ``(base_without_query, query_string)``.

    ArcGIS secures services with a ``?token=…`` query on every request, so the query
    must be preserved and re-attached to the metadata and tile URLs rather than
    clobbered by a naive ``?f=json`` concatenation.
    """
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    return base, parts.query


def _vts_request(
    url: str, auth: tuple[str, str] | None, *, accept_json: bool
) -> urllib.request.Request:
    """Build the urllib request for a VectorTileServer fetch, sending Basic auth preemptively."""
    headers = {
        "User-Agent": _VTS_USER_AGENT
    }  # one UA for the metadata and tile requests
    if accept_json:
        headers["Accept"] = "application/json"
    if auth is not None:
        # Preemptive Basic auth (matches from_wfs / from_ogc_features): a service that
        # 403s without a 401 challenge, or blocks the default urllib UA, still authenticates.
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return urllib.request.Request(url, headers=headers)


def fetch_vectortileserver_metadata(
    url: str, auth: tuple[str, str] | None, timeout: float
) -> dict[str, Any]:
    """GET ``<url>?f=json`` and return the parsed ArcGIS VectorTileServer service metadata."""
    base, query = _vts_base_and_query(url)
    merged = f"{query}&f=json" if query else "f=json"
    meta_url = f"{base}?{merged}"
    try:
        payload = http_get_with_retry(
            _vts_request(meta_url, auth, accept_json=True), timeout
        )
    except urllib.error.HTTPError as exc:
        raise VectorTileServerError(
            f"VectorTileServer metadata request failed for {url!r}: HTTP {exc.code} {http_error_detail(exc)}"
        ) from exc
    except OSError as exc:
        raise VectorTileServerError(
            f"VectorTileServer metadata request failed for {url!r}: {exc}"
        ) from exc
    try:
        doc = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise VectorTileServerError(
            f"VectorTileServer metadata returned a non-JSON body from {url!r}: {exc}"
        ) from exc
    if not isinstance(doc, dict) or "tileInfo" not in doc:
        raise VectorTileServerError(
            f"{url!r} does not describe an ArcGIS VectorTileServer "
            "(no 'tileInfo' in the service metadata)."
        )
    return doc


def fetch_vectortileserver_tile(
    tile_url: str, auth: tuple[str, str] | None, timeout: float
) -> bytes | None:
    """GET one ``.pbf`` tile; return its bytes, or ``None`` when the tile is absent (HTTP 404)."""
    try:
        return http_get_with_retry(
            _vts_request(tile_url, auth, accept_json=False), timeout
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # a missing tile just means the covered cell holds no data
        raise VectorTileServerError(
            f"VectorTileServer tile request failed for {tile_url!r}: HTTP {exc.code} {http_error_detail(exc)}"
        ) from exc
    except OSError as exc:
        raise VectorTileServerError(
            f"VectorTileServer tile request failed for {tile_url!r}: {exc}"
        ) from exc


def _resolve_vts_tiling(
    meta: dict[str, Any],
) -> tuple[float, float, int, dict[int, float], str]:
    """Return ``(origin_x, origin_y, tile_size, lods, tile_template)`` from the service metadata.

    ``lods`` maps each level-of-detail integer to its resolution (map units per pixel).

    Raises:
        ValueError: The tiling CRS is not Web Mercator (the only scheme GDAL's MVT
            georeferencing supports).
        VectorTileServerError: The metadata carries no levels of detail.
    """
    tile_info = meta.get("tileInfo") or {}
    spatial_ref = tile_info.get("spatialReference") or {}
    wkid = int(spatial_ref.get("latestWkid") or spatial_ref.get("wkid") or 3857)
    if wkid not in _WEBMERC_WKIDS:
        raise ValueError(
            "from_vectortileserver supports only Web Mercator vector tiles (EPSG:3857); "
            f"the service tiling CRS is wkid={wkid}."
        )
    origin = tile_info.get("origin") or {}
    origin_x = float(origin.get("x", -_WEBMERC_ORIGIN))
    origin_y = float(origin.get("y", _WEBMERC_ORIGIN))
    tile_size = int(tile_info.get("cols") or tile_info.get("rows") or 512)
    lods = {
        int(lod["level"]): float(lod["resolution"])
        for lod in tile_info.get("lods", [])
        if "level" in lod and "resolution" in lod
    }
    if not lods:
        raise VectorTileServerError(
            "VectorTileServer metadata has no tileInfo.lods (levels of detail); cannot compute tiles."
        )
    templates = meta.get("tiles") or ["tile/{z}/{y}/{x}.pbf"]
    return origin_x, origin_y, tile_size, lods, templates[0]


def _vts_bbox_3857(
    bbox: tuple[float, float, float, float] | None, meta: dict[str, Any]
) -> tuple[float, float, float, float]:
    """Resolve the read extent in EPSG:3857 from a lon/lat ``bbox`` or the service full extent."""
    if bbox is not None:
        west, south, east, north = (float(v) for v in bbox)
        if not (west < east and south < north):
            raise ValueError(
                "from_vectortileserver: bbox must be (west, south, east, north) in EPSG:4326 with "
                f"west < east and south < north; got {bbox!r}."
            )
        transformer = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
        # transform_bounds densifies the edges, so the 3857 box spans the whole extent
        # (exact for 4326, and correct for a rotated / non-axis-aligned source CRS below).
        minx, miny, maxx, maxy = transformer.transform_bounds(west, south, east, north)
        return minx, miny, maxx, maxy
    extent = meta.get("fullExtent") or meta.get("initialExtent") or {}
    try:
        xmin, ymin, xmax, ymax = (
            float(extent[key]) for key in ("xmin", "ymin", "xmax", "ymax")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "from_vectortileserver: no bbox given and the service metadata has no usable fullExtent; "
            "pass bbox=(west, south, east, north)."
        ) from exc
    # fullExtent is usually already in the tile-matrix CRS (Web Mercator), but some
    # services report it in the data's own CRS (e.g. 4326 or a projected SR). Honour its
    # spatialReference instead of assuming metres, reprojecting the corners to 3857 when
    # it is not Web Mercator so the tile maths stay in the tile CRS.
    spatial_ref = extent.get("spatialReference") or {}
    wkid = int(spatial_ref.get("latestWkid") or spatial_ref.get("wkid") or 3857)
    if wkid in _WEBMERC_WKIDS:
        return xmin, ymin, xmax, ymax
    try:
        transformer = pyproj.Transformer.from_crs(wkid, 3857, always_xy=True)
    except _PyprojCRSError as exc:
        raise VectorTileServerError(
            f"VectorTileServer fullExtent is in an unrecognised CRS (wkid={wkid}); "
            "pass an explicit bbox=(west, south, east, north) in EPSG:4326."
        ) from exc
    # transform_bounds over all four edges, so a projected (possibly non-axis-aligned)
    # fullExtent is fully covered rather than under-spanned by two corners.
    minx, miny, maxx, maxy = transformer.transform_bounds(xmin, ymin, xmax, ymax)
    return minx, miny, maxx, maxy


def _vts_tile_range(
    bbox_3857: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    tile_span: float,
) -> tuple[int, int, int, int]:
    """Inclusive ``(col_min, col_max, row_min, row_max)``, clamped to the valid grid.

    The grid dimension is derived from ``tile_span`` (the Web Mercator world span divided
    by the tile span in metres), *not* from ``2**level`` — so it stays correct for a
    service that advertises a non-canonical LOD numbering or resolution rather than the
    standard zoom-exponent scheme, using the same ``origin`` / ``tile_span`` values the
    ranges are computed from. Clamping keeps the covering-tile list, the count that drives
    the zoom-pick / ``max_tiles`` cap, and the fetch in agreement, and stops the reader
    requesting out-of-world tiles that would only 404.
    """
    minx, miny, maxx, maxy = bbox_3857
    last = max(0, round((2 * _WEBMERC_ORIGIN) / tile_span) - 1)
    col_min = max(0, int(math.floor((minx - origin_x) / tile_span)))
    col_max = min(last, int(math.floor((maxx - origin_x) / tile_span)))
    # y grows downward from the top-left origin, so the north edge maps to the smallest row.
    row_min = max(0, int(math.floor((origin_y - maxy) / tile_span)))
    row_max = min(last, int(math.floor((origin_y - miny) / tile_span)))
    return col_min, col_max, row_min, row_max


def _vts_tile_count(
    bbox_3857: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    tile_span: float,
) -> int:
    """Number of in-grid tiles covering ``bbox_3857`` at ``tile_span`` (matches the fetched set)."""
    col_min, col_max, row_min, row_max = _vts_tile_range(
        bbox_3857, origin_x, origin_y, tile_span
    )
    return max(0, col_max - col_min + 1) * max(0, row_max - row_min + 1)


def _pick_vts_zoom(
    bbox_3857: tuple[float, float, float, float],
    lods: dict[int, float],
    origin_x: float,
    origin_y: float,
    tile_size: int,
    max_tiles: int,
) -> int:
    """Highest LOD whose covering-tile count fits ``max_tiles`` (else the coarsest LOD)."""
    for level in sorted(lods, reverse=True):
        tile_span = tile_size * lods[level]
        if _vts_tile_count(bbox_3857, origin_x, origin_y, tile_span) <= max_tiles:
            return level
    return min(lods)


def _resolve_vts_level(
    zoom: int | None,
    bbox_3857: tuple[float, float, float, float],
    lods: dict[int, float],
    origin_x: float,
    origin_y: float,
    tile_size: int,
    max_tiles: int,
    url: str,
) -> int:
    """Validate an explicit ``zoom`` against the advertised LODs, or auto-pick one."""
    if zoom is not None:
        if zoom not in lods:
            raise ValueError(
                f"from_vectortileserver: zoom {zoom} is not an advertised LOD of {url!r}; "
                f"available levels: {sorted(lods)}."
            )
        return zoom
    return _pick_vts_zoom(bbox_3857, lods, origin_x, origin_y, tile_size, max_tiles)


def _covering_vts_tiles(
    bbox_3857: tuple[float, float, float, float],
    level: int,
    origin_x: float,
    origin_y: float,
    tile_span: float,
    max_tiles: int,
) -> list[tuple[int, int, int]]:
    """List ``(z, x, y)`` tiles covering ``bbox_3857`` at ``level``; warn + truncate past ``max_tiles``."""
    col_min, col_max, row_min, row_max = _vts_tile_range(
        bbox_3857, origin_x, origin_y, tile_span
    )
    # The range is already clamped to the valid, in-world grid, so every index here is a
    # real tile — the count above (which drove the zoom-pick / cap) matches this list.
    tiles = [
        (level, x, y)
        for x in range(col_min, col_max + 1)
        for y in range(row_min, row_max + 1)
    ]
    if len(tiles) > max_tiles:
        warnings.warn(
            f"from_vectortileserver: bbox at zoom {level} covers {len(tiles)} tiles, over the "
            f"max_tiles={max_tiles} cap; reading the first {max_tiles}. Raise max_tiles or use a "
            "smaller bbox / lower zoom to read the rest.",
            stacklevel=2,
        )
        tiles = tiles[:max_tiles]
    return tiles


def _read_vts_tile_frame(
    tile_bytes: bytes, z: int, x: int, y: int, layer: str | None, work_dir: str
) -> list[GeoDataFrame]:
    """Parse one MVT tile's bytes into EPSG:3857 GeoDataFrames (one per read sub-layer).

    GDAL's MVT driver georeferences a tile to EPSG:3857 when the file path ends in
    ``/{z}/{x}/{y}.pbf`` (verified against GDAL 3.13), so the tile is written under
    that structure and read with geopandas — no manual coordinate maths.
    """
    tile_path = os.path.join(work_dir, str(z), str(x), f"{y}.pbf")
    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    with open(tile_path, "wb") as handle:
        handle.write(tile_bytes)
    available = [row[0] for row in pyogrio.list_layers(tile_path)]
    names = [layer] if layer is not None else available
    frames: list[GeoDataFrame] = []
    for name in names:
        if name not in available:
            continue  # this tile does not carry the requested sub-layer
        gdf = gpd.read_file(tile_path, engine="pyogrio", layer=name)
        if len(gdf) == 0:
            continue
        # ``mvt_id`` is GDAL's per-tile feature id — not stable across tiles, so it is
        # not a cross-tile key. Drop it and tag the source sub-layer instead.
        gdf = gdf.drop(columns=["mvt_id"], errors="ignore")
        gdf["layer"] = name
        frames.append(gdf)
    return frames


def _assemble_vts_frames(
    fc_cls: type[FeatureCollection], frames: list[GeoDataFrame]
) -> FeatureCollection:
    """Concatenate per-tile frames into one EPSG:3857 FeatureCollection, dropping seam duplicates."""
    if not frames:
        return fc_cls(gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"))
    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:3857"
    )
    # MVT clips features at tile boundaries, so a feature spanning tiles is returned as
    # several clipped pieces (kept as distinct rows). Drop only *exact*-duplicate rows —
    # identical attributes and byte-identical geometry. MVT quantises each tile to its own
    # local grid, so near-seam buffer copies are usually not byte-identical and are NOT
    # removed here: this is a cheap exact guard, not full seam reassembly (see the facade).
    merged["_wkb"] = merged.geometry.to_wkb()
    subset = [column for column in merged.columns if column != "geometry"]
    deduped = merged.drop_duplicates(subset=subset).drop(columns="_wkb")
    return fc_cls(deduped.reset_index(drop=True))


def from_vectortileserver(
    fc_cls: type[FeatureCollection],
    url: str,
    *,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    zoom: int | None = None,
    output_crs: str | None = None,
    max_tiles: int = 1000,
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> FeatureCollection:
    """Read an ArcGIS VectorTileServer endpoint (see FeatureCollection.from_vectortileserver)."""
    if max_tiles < 1:
        raise ValueError(
            f"from_vectortileserver: max_tiles must be >= 1, got {max_tiles}."
        )
    meta = fc_cls._fetch_vectortileserver_metadata(url, auth, timeout)
    origin_x, origin_y, tile_size, lods, template = _resolve_vts_tiling(meta)
    bbox_3857 = _vts_bbox_3857(bbox, meta)
    level = _resolve_vts_level(
        zoom, bbox_3857, lods, origin_x, origin_y, tile_size, max_tiles, url
    )
    tile_span = tile_size * lods[level]
    tiles = _covering_vts_tiles(
        bbox_3857, level, origin_x, origin_y, tile_span, max_tiles
    )

    base, query = _vts_base_and_query(url)
    query_suffix = (
        f"?{query}" if query else ""
    )  # carry ?token=… onto every tile request
    frames: list[GeoDataFrame] = []
    with tempfile.TemporaryDirectory(prefix="pyramids_vts_") as work_dir:
        for tz, tx, ty in tiles:
            tile_url = f"{base}/{template.format(z=tz, x=tx, y=ty)}{query_suffix}"
            tile_bytes = fc_cls._fetch_vectortileserver_tile(tile_url, auth, timeout)
            if not tile_bytes:
                continue
            frames.extend(_read_vts_tile_frame(tile_bytes, tz, tx, ty, layer, work_dir))

    result = _assemble_vts_frames(fc_cls, frames)
    if output_crs is not None:
        result = result.to_crs(
            output_crs
        )  # to_crs preserves the FeatureCollection subclass
    return result


def _resolve_lazy_partitioning(
    path: str, npartitions: int | None, chunksize: int | None
) -> dict[str, Any]:
    """Default `npartitions` from file size when neither knob is given (see FeatureCollection.read_file)."""
    kwargs: dict[str, Any] = {}
    if npartitions is not None:
        kwargs["npartitions"] = npartitions
    elif chunksize is not None:
        kwargs["chunksize"] = chunksize
    elif path.startswith(("/vsi", "http://", "https://", "s3://", "gs://", "az://")):
        # Remote / VFS path — no cheap size probe. Fall back to 1.
        kwargs["npartitions"] = 1
    else:
        try:
            size = os.path.getsize(path)
        except OSError:
            kwargs["npartitions"] = 1
        else:
            kwargs["npartitions"] = max(
                1, math.ceil(size / _LAZY_TARGET_BYTES_PER_PARTITION)
            )
    return kwargs


def _require_pyarrow() -> None:
    """Raise a pyramids-branded ImportError if pyarrow is absent."""
    import_pyarrow(
        "GeoParquet support requires the optional 'pyarrow' "
        "dependency. Install with one of:\n"
        "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
        "  - conda-forge: conda install -c conda-forge pyramids-parquet"
    )


def _compact(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `mapping` with the ``None``-valued entries removed (ARC-72)."""
    return {key: value for key, value in mapping.items() if value is not None}


def _import_dask_geopandas():
    """Import and return ``dask_geopandas`` or raise a pyramids-branded ImportError (ARC-72)."""
    try:
        import dask_geopandas
    except ImportError as exc:
        raise ImportError(
            "backend='dask' requires the optional "
            "'dask-geopandas' dependency. Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-parquet"
        ) from exc
    return dask_geopandas


def read_file_dask(
    resolved: str,
    *,
    layer: str | int | None,
    bbox: Any,
    mask: Any,
    rows: slice | int | None,
    columns: list[str] | None,
    where: str | None,
    npartitions: int | None,
    chunksize: int | None,
) -> LazyFeatureCollection:
    """Dask backend for :func:`read_file`: reject unsupported filters, wrap as LazyFC."""
    # dask_geopandas.read_file does NOT forward pyogrio filter kwargs
    # (bbox / mask / rows / columns / where) — silently dropping them was the bug.
    # Raise a clear ValueError instead so users know to pre-filter or call .compute()
    # and filter eagerly.
    unsupported = {
        "bbox": bbox,
        "mask": mask,
        "rows": rows,
        "columns": columns,
        "where": where,
        "layer": layer,
    }
    supplied = [k for k, v in unsupported.items() if v is not None]
    if supplied:
        raise ValueError(
            f"backend='dask' does not support filter kwargs "
            f"{supplied}. dask_geopandas.read_file has no "
            "pushdown story for these. Either omit them and "
            "filter post-load via .clip / .loc / .compute, or "
            "switch to read_parquet(backend='dask', filters=...)"
        )
    dask_geopandas = _import_dask_geopandas()
    partition_kwargs = _resolve_lazy_partitioning(resolved, npartitions, chunksize)
    # Local import breaks the collection <-> _lazy_collection cycle
    # (_lazy_collection imports FeatureCollection from collection).
    from pyramids.feature._lazy_collection import LazyFeatureCollection

    dask_gdf = dask_geopandas.read_file(resolved, **partition_kwargs)
    return LazyFeatureCollection.from_dask_gdf(dask_gdf)


def read_file(
    fc_cls: type[FeatureCollection],
    path: str | Path,
    *,
    layer: str | int | None = None,
    bbox: Any = None,
    mask: Any = None,
    rows: slice | int | None = None,
    columns: list[str] | None = None,
    where: str | None = None,
    backend: str = "pandas",
    npartitions: int | None = None,
    chunksize: int | None = None,
    **kwargs: Any,
) -> FeatureCollection | LazyFeatureCollection:
    """Read a vector file into a FeatureCollection (see FeatureCollection.read_file)."""
    # Only pass kwargs that were actually supplied — passing the unset
    # defaults (None) confuses some geopandas engines (ARC-72).
    passthrough = _compact(
        {
            "layer": layer,
            "bbox": bbox,
            "mask": mask,
            "rows": rows,
            "columns": columns,
            "where": where,
        }
    )
    passthrough.update(kwargs)
    if (
        backend == "pandas"
        and _is_remote_geojson(path)
        and not _gdal_http_options_active()
    ):
        # A remote GeoJSON is staged to a local file and read from there (#1008):
        # streaming a *redirecting* remote GeoJSON over GDAL's /vsicurl/ can segfault
        # the interpreter in a build whose bundled libcurl/OpenSSL differs from the
        # interpreter's. GeoJSON has no spatial index, so this loses no read pushdown.
        # Staging is skipped when a GDAL HTTP option is set so the caller's auth/TLS
        # tuning still reaches GDAL's /vsicurl/ reader (urllib would drop it).
        return _read_remote_geojson_staged(fc_cls, path, passthrough)
    resolved = _pyramids_io._parse_path(path)
    if backend == "dask":
        return read_file_dask(
            resolved,
            layer=layer,
            bbox=bbox,
            mask=mask,
            rows=rows,
            columns=columns,
            where=where,
            npartitions=npartitions,
            chunksize=chunksize,
        )
    if backend != "pandas":
        raise ValueError(f"backend must be 'pandas' or 'dask', got {backend!r}")
    gdf = _read_file_healing_crs(resolved, passthrough)
    return fc_cls(gdf)


_GEOJSON_SUFFIXES = (".geojson",)
"""Extension staged as remote GeoJSON (:func:`_is_remote_geojson`); bare ``.json`` is too broad
(TopoJSON/ESRIJSON/plain JSON) so it keeps the normal ``/vsicurl/`` reader."""

_VSICURL_PREFIXES = ("/vsicurl_streaming/", "/vsicurl/")
"""GDAL virtual-filesystem prefixes for a streamed remote read (:func:`_strip_vsicurl`)."""

_HTTPS_SCHEME = "https://"
"""The only URL scheme staged locally; a plain ``http://`` read carries no TLS to divert."""

_REMOTE_READ_TIMEOUT = 60.0
"""Per-read socket timeout (seconds) for a staged remote GeoJSON download (issue #1008 review M2);
it bounds each blocking read, not the whole transfer."""


def _strip_vsicurl(text: str) -> str:
    """Return the bare URL behind a leading ``/vsicurl/`` or ``/vsicurl_streaming/`` wrapper.

    Only a *leading* prefix is removed, so a chained virtual path such as
    ``/vsizip//vsicurl/…`` is left untouched. GDAL's ``/vsicurl?url=`` query form is not
    decoded here, so a caller who uses that spelling — or reads on the ``dask`` backend —
    is not diverted to local staging and still reaches GDAL's ``/vsicurl/`` reader.

    Args:
        text: A path or URL, optionally wrapped in a ``/vsicurl*`` prefix.

    Returns:
        str: The URL with a leading ``/vsicurl*`` prefix removed, or ``text`` unchanged.
    """
    stripped = text
    for prefix in _VSICURL_PREFIXES:
        if text.startswith(prefix):
            stripped = text[len(prefix) :]
            break
    return stripped


def _is_remote_geojson(path: str | Path) -> bool:
    """True for an ``https://`` GeoJSON URL, bare or ``/vsicurl/``-wrapped (issue #1008).

    Only an ``https://`` GeoJSON qualifies: the segfault is a TLS/OpenSSL clash while
    GDAL's ``/vsicurl/`` follows a redirect, so it is the *TLS* read that must be kept
    away from GDAL. A plain ``http://`` read carries no TLS, and a local path, an
    ``s3://``/``gs://`` object, or any non-GeoJSON remote file returns ``False`` and
    keeps its normal read path.

    Args:
        path: The path or URL handed to :func:`read_file`.

    Returns:
        bool: Whether the path is a remote GeoJSON that should be staged locally.
    """
    url = _strip_vsicurl(str(path))
    without_query = url.split("?", 1)[0].rstrip("/")
    stem = without_query.lower()
    # A GeoJSON *inside* a remote archive (`.zip/inner.geojson`) is excluded so it keeps
    # the /vsizip//vsicurl/ chain `_parse_path` builds. Scan only the URL *path* so an
    # archive-extension host (the `.zip` TLD) is not mistaken for an archive member.
    archive_member = _ARCHIVE_MARKER_RE.search(urlsplit(without_query).path.lower())
    result = (
        url.lower().startswith(_HTTPS_SCHEME)
        and stem.endswith(_GEOJSON_SUFFIXES)
        and not archive_member
    )
    return result


def _read_remote_geojson_staged(
    fc_cls: type[FeatureCollection],
    path: str | Path,
    passthrough: dict[str, Any],
) -> FeatureCollection:
    """Download a remote GeoJSON and read the local copy, avoiding a ``/vsicurl/`` read.

    Streaming a redirecting remote GeoJSON over GDAL's ``/vsicurl/`` can segfault the
    interpreter in a build whose bundled libcurl/OpenSSL differs from the interpreter's
    (issue #1008 — the manylinux wheel, whose vendored OpenSSL 3 clashes with CPython's).
    The bytes are fetched with :mod:`urllib` (Python's own TLS, which follows the
    redirect) and GDAL is handed a plain local file, so GDAL never does the remote read.

    Args:
        fc_cls: The FeatureCollection class to wrap the result in.
        path: The remote GeoJSON path (bare ``https://`` or ``/vsicurl/``-wrapped).
        passthrough: Keyword arguments for :func:`geopandas.read_file`.

    Returns:
        FeatureCollection: The features read from the staged local copy.
    """
    url = _strip_vsicurl(str(path))
    if not url.lower().startswith(_HTTPS_SCHEME):
        # Self-guard the https invariant so the helper cannot become an
        # http/file/ftp fetch if the routing in read_file ever changes.
        raise ValueError(
            f"remote GeoJSON staging requires an https:// URL, got {url!r}"
        )
    request = urllib.request.Request(url, headers={"User-Agent": "pyramids-gis"})
    with tempfile.TemporaryDirectory(prefix="pyramids_geojson_") as work_dir:
        local = os.path.join(work_dir, "remote.geojson")
        try:
            with urllib.request.urlopen(  # nosec B310 - https enforced above
                request, timeout=_REMOTE_READ_TIMEOUT
            ) as response:
                with open(local, "wb") as handle:
                    shutil.copyfileobj(response, handle)
        except urllib.error.URLError as error:
            # Surface a download failure as the module's error type, mirroring the
            # GDAL/pyogrio error the /vsicurl/ path would have raised.
            raise FeatureError(
                f"failed to download remote GeoJSON {url!r}: {error}"
            ) from error
        gdf = _read_file_healing_crs(local, passthrough)
    return fc_cls(gdf)


_GDAL_HTTP_OPTION_KEYS = (
    "GDAL_HTTP_HEADERS",
    "GDAL_HTTP_HEADER_FILE",
    "GDAL_HTTP_AUTH",
    "GDAL_HTTP_BEARER",
    "GDAL_HTTP_USERPWD",
    "GDAL_HTTP_COOKIE",
    "GDAL_HTTP_COOKIEFILE",
    "GDAL_HTTP_PROXY",
    "GDAL_HTTP_PROXYUSERPWD",
    "GDAL_HTTP_UNSAFESSL",
    "GDAL_HTTP_CAPATH",
    "GDAL_HTTP_CACERT",
    "GDAL_HTTP_SSLCERT",
    "GDAL_HTTP_SSLKEY",
    "GDAL_HTTP_SSLKEYPASSWORD",
    "CPL_VSIL_CURL_USERPWD",
)
"""GDAL HTTP config options a ``/vsicurl/`` read honours (:func:`_gdal_http_options_active`)."""


def _gdal_http_options_active() -> bool:
    """True when a GDAL HTTP auth/TLS config option is set (issue #1008 review M1).

    The staging fallback fetches with :mod:`urllib` and cannot see GDAL config options,
    so a caller who supplied credentials or TLS tuning through environment variables or
    :class:`pyramids.base.remote.CloudConfig` must keep the ``/vsicurl/`` path. When any
    such option is set, :func:`read_file` skips staging and lets GDAL do the read so the
    caller's options still apply.

    Any non-empty value counts as set — an explicitly falsy value (e.g.
    ``GDAL_HTTP_UNSAFESSL=NO``) still keeps the ``/vsicurl/`` path, a deliberately
    conservative choice that never strips a caller's HTTP configuration.

    Returns:
        bool: Whether any GDAL HTTP auth/TLS config option is currently set.
    """
    return any(gdal.GetConfigOption(key) for key in _GDAL_HTTP_OPTION_KEYS)


_CRS_HEALING_LOCK = threading.Lock()
"""Serialises the process-wide pyproj patch in :func:`_pyproj_resolving_through_gdal`."""


def _read_file_healing_crs(resolved: Any, passthrough: dict[str, Any]) -> GeoDataFrame:
    """Read a vector file, resolving a CRS the reader's PROJ database cannot look up.

    The reader reports a layer's CRS as an authority string (``"EPSG:10857"``) and
    geopandas resolves it with pyproj, so a file written in a CRS whose code lives in
    GDAL's PROJ database but not pyproj's fails to open at all — before a single
    feature is returned. That is issue #943 arriving through the vector reader instead
    of the raster one.

    Only the *lookup* is missing, never the projection: the same code resolves through
    :func:`crs_from_user_input`. So on that specific failure the geometry is re-read
    with the CRS suppressed and the resolved CRS attached afterwards.

    Args:
        resolved: The path or file-like object to read.
        passthrough: Keyword arguments for :func:`geopandas.read_file`.

    Returns:
        GeoDataFrame: The features, carrying their CRS.
    """
    try:
        gdf = gpd.read_file(resolved, **passthrough)
    except _PyprojCRSError:
        with _pyproj_resolving_through_gdal():
            gdf = gpd.read_file(resolved, **passthrough)
    return gdf


@contextmanager
def _pyproj_resolving_through_gdal() -> Iterator[None]:
    """Let :meth:`pyproj.CRS.from_user_input` fall back to GDAL's PROJ database.

    The obvious repair — read the layer with a lower-level reader and attach the
    resolved CRS afterwards — means rebuilding the GeoDataFrame by hand, and a
    hand-built frame is a second implementation of `read_file` that has to keep pace
    with the real one. The first attempt at exactly that silently dropped `layer=`
    and `rows=`, discarded datetime offsets, left JSON columns as strings and could
    not take the `GeoDataFrame` form of `bbox=` — none of which is *about* CRSes.

    So the read is left entirely to geopandas, and only the single call that fails is
    widened: pyproj keeps its own answer whenever it has one, and falls back to the
    same GDAL rescue used everywhere else when it does not. Everything about the frame
    — layer and row selection, spatial filters, dtypes, timezones, column labels —
    stays whatever `read_file` already produces.

    The patch is process-wide for its duration, so it is serialised and only entered
    after an unpatched read has already failed. It is deliberately narrow: it adds a
    fallback to a call that would otherwise raise, and never changes an answer pyproj
    was able to give.

    Yields:
        None: for the duration of the widened resolution.
    """
    original = pyproj.CRS.from_user_input

    def _healed(value, **kwargs):
        """Resolve `value` as pyproj normally would, falling back to GDAL.

        Args:
            value: Whatever the caller passed to `CRS.from_user_input`.
            **kwargs: Forwarded unchanged.

        Returns:
            pyproj.CRS: The resolved CRS.

        Raises:
            pyproj.exceptions.CRSError: Neither pyproj nor GDAL can read `value`;
                pyproj's own error is re-raised so the type is unchanged.
        """
        try:
            return original(value, **kwargs)
        except _PyprojCRSError:
            # `_pyproj_crs_via_gdal`, not `crs_from_user_input`: the latter routes
            # back through the patched `from_user_input` and would recurse.
            rescued = _pyproj_crs_via_gdal(value)
            if rescued is None:
                raise
            return rescued

    with _CRS_HEALING_LOCK:
        pyproj.CRS.from_user_input = _healed  # type: ignore[method-assign]
        try:
            yield
        finally:
            pyproj.CRS.from_user_input = original  # type: ignore[method-assign]


def _validate_iter_features_args(
    fc_cls: type[FeatureCollection],
    *,
    chunksize: int | None,
    tile_strategy: str,
    where: str | None,
    bbox: tuple[float, float, float, float] | None,
    include_index: bool,
) -> None:
    """Validate :func:`iter_features` arguments (raises before any I/O)."""
    if chunksize is not None and chunksize < 1:
        raise ValueError(f"chunksize must be >= 1 when supplied; got {chunksize}.")
    if tile_strategy not in fc_cls._VALID_TILE_STRATEGIES:
        raise ValueError(
            f"tile_strategy must be one of "
            f"{fc_cls._VALID_TILE_STRATEGIES}; got {tile_strategy!r}."
        )
    # The emitted id / _row_index is the absolute source-file row position, computed as
    # range(start, start + len(chunk)). That only holds when nothing filters at the driver
    # level: a pushed-down `where` or `bbox` makes skip_features count over the filtered set,
    # so the positions would be wrong. Refuse that combination rather than emit wrong ids
    # (ARC-31). The Python-side bbox path (tile_strategy="none") reads full chunks and masks
    # row_indices afterwards, so it stays correct.
    if include_index and (
        where is not None or (bbox is not None and tile_strategy != "none")
    ):
        raise ValueError(
            "iter_features(include_index=True) is incompatible with driver-side filtering "
            "because the emitted id is the absolute source-file row position: pass where=None "
            "and either bbox=None or tile_strategy='none' (Python-side bbox)."
        )


def iter_features(
    fc_cls: type[FeatureCollection],
    path: str | Path,
    *,
    layer: str | int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: str | None = None,
    chunksize: int | None = None,
    tile_strategy: str = "auto",
    include_index: bool = False,
) -> Iterator[dict[str, Any] | FeatureCollection]:
    """Stream features from `path` without materialising the file (see FeatureCollection.iter_features)."""
    # Runs lazily on first iteration (iter_features is a generator), preserving the
    # the "validate on first next()" behaviour.
    _validate_iter_features_args(
        fc_cls,
        chunksize=chunksize,
        tile_strategy=tile_strategy,
        where=where,
        bbox=bbox,
        include_index=include_index,
    )

    resolved = str(_pyramids_io._parse_path(path))

    # pyogrio's read_info is O(1); use it to size the layer so we can
    # iterate in fixed-size batches via skip_features / max_features.
    info_kwargs: dict[str, Any] = {}
    if layer is not None:
        info_kwargs["layer"] = layer
    info = pyogrio.read_info(resolved, **info_kwargs)
    total = int(info["features"])

    if chunksize is None:
        batch_size = _DEFAULT_ITER_BATCH_SIZE
    else:
        batch_size = int(chunksize)

    read_kwargs, python_bbox = build_iter_read_kwargs(layer, where, bbox, tile_strategy)

    for start in range(0, total, batch_size):
        gdf_chunk = gpd.read_file(
            resolved,
            skip_features=start,
            max_features=batch_size,
            **read_kwargs,
        )
        # Absolute row indices captured before any bbox masking, so callers
        # can map yielded features back to their source rows.
        row_indices = (
            list(range(start, start + len(gdf_chunk))) if include_index else None
        )
        if python_bbox is not None and len(gdf_chunk) > 0:
            xmin, ymin, xmax, ymax = python_bbox
            mask = gdf_chunk.intersects(box(xmin, ymin, xmax, ymax))
            if row_indices is not None:
                row_indices = [ri for ri, keep in zip(row_indices, mask) if keep]
            gdf_chunk = gdf_chunk[mask]
        yield from emit_features(
            fc_cls, gdf_chunk, row_indices, chunksize, include_index
        )


def build_iter_read_kwargs(
    layer: str | int | None,
    where: str | None,
    bbox: tuple[float, float, float, float] | None,
    tile_strategy: str,
) -> tuple[dict[str, Any], tuple[float, float, float, float] | None]:
    """Build the pyogrio ``read_file`` kwargs for :func:`iter_features`.

    The engine is pinned to pyogrio (``skip_features`` / ``max_features`` are
    pyogrio-specific; some engines silently ignore them). For every
    ``tile_strategy`` except ``"none"`` the ``bbox`` is pushed down to pyogrio;
    for ``"none"`` it is held back for a post-load Python filter.
    """
    read_kwargs: dict[str, Any] = {"engine": "pyogrio"}
    if layer is not None:
        read_kwargs["layer"] = layer
    if where is not None:
        read_kwargs["where"] = where
    pushdown_bbox = bbox if tile_strategy != "none" else None
    python_bbox = bbox if tile_strategy == "none" else None
    if pushdown_bbox is not None:
        read_kwargs["bbox"] = pushdown_bbox
    return read_kwargs, python_bbox


def emit_features(
    fc_cls: type[FeatureCollection],
    gdf_chunk: Any,
    row_indices: list[int] | None,
    chunksize: int | None,
    include_index: bool,
) -> Iterator[dict[str, Any] | FeatureCollection]:
    """Yield a processed chunk for :func:`iter_features` (per-feature dicts or FC chunks)."""
    if chunksize is None:
        iterator = gdf_chunk.iterfeatures(na="null")
        if include_index and row_indices is not None:
            for ri, feat in zip(row_indices, iterator):
                feat["id"] = ri
                yield feat
        else:
            yield from iterator
    else:
        chunk_fc = fc_cls(gdf_chunk)
        if include_index:
            chunk_fc["_row_index"] = row_indices
        yield chunk_fc


def open_arrow(
    path: str | Path,
    *,
    layer: str | int | None = None,
    columns: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: str | None = None,
    batch_size: int | None = None,
) -> Any:
    """Open a vector file as a streaming pyarrow RecordBatchReader (see FeatureCollection.open_arrow)."""
    try:
        from pyogrio.raw import open_arrow as _pyogrio_open_arrow
    except ImportError as exc:
        raise ImportError(
            "open_arrow requires the optional 'pyogrio' dependency. "
            "Install with one of:\n"
            "  - PyPI:        pip install pyogrio\n"
            "  - conda-forge: conda install -c conda-forge pyogrio"
        ) from exc
    resolved = _pyramids_io._parse_path(path)
    kwargs: dict[str, Any] = {}
    if layer is not None:
        kwargs["layer"] = layer
    if columns is not None:
        kwargs["columns"] = columns
    if bbox is not None:
        kwargs["bbox"] = bbox
    if where is not None:
        kwargs["where"] = where
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return _pyogrio_open_arrow(resolved, **kwargs)


def read_parquet_dask(
    resolved: str,
    *,
    columns: list[str] | None,
    split_row_groups: bool | None,
    filters: list | None,
    blocksize: int | str | None,
    storage_options: dict | None,
    extra_kwargs: dict[str, Any],
) -> LazyFeatureCollection:
    """Dask backend for :func:`read_parquet`: wrap dask_geopandas as a LazyFeatureCollection."""
    # Check deps in order of specificity — the dask-geopandas hint beats the
    # generic pyarrow one. When both are missing, this error names the extra.
    dask_geopandas = _import_dask_geopandas()
    dask_kwargs = _compact(
        {
            "columns": columns,
            "split_row_groups": split_row_groups,
            "filters": filters,
            "blocksize": blocksize,
            "storage_options": storage_options,
        }
    )
    dask_kwargs.update(extra_kwargs)
    # dask_geopandas is installed → assert pyarrow too, so the user gets the
    # pyramids-branded hint (not the upstream message). `[parquet]` pulls both.
    _require_pyarrow()
    # Local import breaks the collection <-> _lazy_collection cycle.
    from pyramids.feature._lazy_collection import LazyFeatureCollection

    dask_gdf = dask_geopandas.read_parquet(resolved, **dask_kwargs)
    return LazyFeatureCollection.from_dask_gdf(dask_gdf)


def read_parquet(
    fc_cls: type[FeatureCollection],
    path: str | Path,
    *,
    columns: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    backend: str = "pandas",
    split_row_groups: bool | None = None,
    filters: list | None = None,
    blocksize: int | str | None = None,
    storage_options: dict | None = None,
    **kwargs: Any,
) -> FeatureCollection | LazyFeatureCollection:
    """Read a GeoParquet file into a FeatureCollection (see FeatureCollection.read_parquet)."""
    # geopandas and dask-geopandas read Parquet through pyarrow + fsspec, which
    # speak s3://, gs:// and az:// natively. Unlike GDAL they do not understand
    # the /vsis3/ form _parse_path produces, and on Windows a leading "/vsis3/"
    # resolves against the drive root, so the read dies with FileNotFoundError.
    # Hand fsspec the URL untouched; local paths still go through _parse_path.
    path_str = str(path)
    resolved = (
        to_fsspec_url(path_str)
        if is_remote(path_str)
        else _pyramids_io._parse_path(path)
    )
    if backend == "dask":
        return read_parquet_dask(
            resolved,
            columns=columns,
            split_row_groups=split_row_groups,
            filters=filters,
            blocksize=blocksize,
            storage_options=storage_options,
            extra_kwargs=kwargs,
        )
    if backend != "pandas":
        raise ValueError(f"backend must be 'pandas' or 'dask', got {backend!r}")
    _require_pyarrow()
    # geopandas 1.x forwards **kwargs into pyarrow.parquet.read_table, which has
    # never accepted the pandas-style `engine=` kwarg; _require_pyarrow() above
    # hard-guarantees the pyarrow backend, so no injection is needed here.
    passthrough: dict[str, Any] = {}
    passthrough.update(kwargs)
    if columns is not None:
        passthrough["columns"] = columns
    if bbox is not None:
        passthrough["bbox"] = bbox
    if storage_options is not None:
        passthrough["storage_options"] = storage_options
    gdf = gpd.read_parquet(resolved, **passthrough)
    return fc_cls(gdf)


def from_features(
    fc_cls: type[FeatureCollection],
    features: Iterable[Any],
    *,
    crs: Any = None,
    columns: list[str] | None = None,
) -> FeatureCollection:
    """Build an FC from feature-shaped inputs (see FeatureCollection.from_features)."""
    # Materialise the iterator so we can detect the empty case before handing off
    # to geopandas: gpd.from_features([]) returns a GeoDataFrame with no geometry
    # column, which breaks every pyramids op that assumes the column exists.
    features_list = list(features)
    if not features_list:
        raise ValueError(
            "from_features requires at least one feature. An empty "
            "iterable would produce a GeoDataFrame with no geometry "
            "column, which breaks downstream pyramids methods."
        )
    gdf = gpd.GeoDataFrame.from_features(features_list, crs=crs, columns=columns)
    return fc_cls(gdf)


def from_bbox(
    fc_cls: type[FeatureCollection],
    bbox: tuple[float, float, float, float] | list[float],
    *,
    epsg: Any,
) -> FeatureCollection:
    """Build a one-row FC from a (west, south, east, north) bbox (see FeatureCollection.from_bbox)."""
    if epsg is None:
        raise ValueError(
            "from_bbox requires an explicit epsg= for the bbox CRS; "
            "a bbox without a CRS is ambiguous"
        )
    try:
        seq = list(bbox)
    except TypeError as exc:
        raise ValueError(
            f"bbox must be a 4-element (west, south, east, north) sequence; got {bbox!r}"
        ) from exc
    if len(seq) != 4:
        raise ValueError(
            f"bbox must have exactly 4 elements (west, south, east, north); got {len(seq)}: {seq!r}"
        )
    try:
        w, s, e, n = (float(v) for v in seq)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"bbox elements must be numbers; got {seq!r}") from exc
    # NaN slips past the ordering checks below (nan >= x is False), so reject it
    # explicitly — e.g. an empty frame's all-NaN total_bounds.
    if any(math.isnan(v) for v in (w, s, e, n)):
        raise ValueError(f"bbox coordinates must not be NaN; got {seq!r}")
    if w >= e:
        raise ValueError(f"bbox must satisfy west < east; got west={w}, east={e}")
    if s >= n:
        raise ValueError(f"bbox must satisfy south < north; got south={s}, north={n}")
    return fc_cls(geometry=[box(w, s, e, n)], crs=epsg)


def from_records(
    fc_cls: type[FeatureCollection],
    records: Any,
    *,
    geometry: str = "geometry",
    crs: Any = None,
    orient: str = "records",
) -> FeatureCollection:
    """Build an FC from dict records or a columnar dict (see FeatureCollection.from_records)."""

    def _empty_fc() -> FeatureCollection:
        # Both empty-input branches build a single-column frame whose column name
        # matches the geometry= kwarg, so GeoDataFrame(..., geometry=…) sets it as
        # the active geometry column and the returned FC has geometry.name == geometry.
        return fc_cls(gpd.GeoDataFrame({geometry: []}, geometry=geometry, crs=crs))

    if orient == "records":
        records_list = list(records)
        if not records_list:
            return _empty_fc()
        df = pd.DataFrame.from_records(records_list)
    elif orient == "list":
        # Columnar dict of equal-length lists. pd.DataFrame accepts this shape
        # natively and raises ValueError on mismatched lengths (propagated as-is).
        if not isinstance(records, dict):
            raise ValueError(
                f"orient='list' expects a dict of column → list; got {type(records).__name__}."
            )
        df = pd.DataFrame(records)
        if len(df) == 0:
            return _empty_fc()
    else:
        raise ValueError(f"orient must be 'records' or 'list'; got {orient!r}.")
    if geometry not in df.columns:
        raise FeatureError(
            f"records missing required geometry column {geometry!r}; "
            f"columns present: {list(df.columns)}"
        )
    return fc_cls(gpd.GeoDataFrame(df, geometry=geometry, crs=crs))
