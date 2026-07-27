"""I/O reader engine for :class:`~pyramids.feature.FeatureCollection` (ARC-36).

Module-level implementations of the collection's readers, kept out of the
god-class as free functions that take the `FeatureCollection` class (`fc_cls`) or a
collection (`fc`) as their first argument. The `FeatureCollection` reader methods
are thin facades over these functions; the full docstrings/doctests stay on the
facades (the public API).

Part 1 covers layer listing (with its LRU cache) and the web readers (ArcGIS
FeatureServer pagination, GPX sub-layers). The web readers call back through the
`fc_cls` facades (`fc_cls.read_file`, `fc_cls._read_featureserver_page`) so existing
tests that monkeypatch those class methods still intercept.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd

from pyramids import _io as _pyramids_io
from pyramids.base.remote import is_remote


@lru_cache(maxsize=128)
def _list_layers_cached(resolved_path: str) -> tuple[str, ...]:
    """Return a tuple of layer names for a resolved path (memoised)."""
    import pyogrio

    arr = pyogrio.list_layers(resolved_path)
    return tuple(str(row[0]) for row in arr)


def list_layers(fc_cls: type, path: str | Path) -> list[str]:
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


def read_gpx_layers(fc_cls: type, path: str | Path) -> dict[str, Any]:
    """Read every non-empty GPX sub-layer into a dict (see FeatureCollection.read_gpx_layers)."""
    result: dict[str, Any] = {}
    for name in fc_cls.list_layers(path):
        fc = fc_cls.read_file(path, layer=name)
        if len(fc) > 0:
            result[name] = fc
    return result


def read_featureserver_page(fc_cls: type, page_url: str) -> Any:
    """Read one ESRIJSON page from an ArcGIS FeatureServer query URL."""
    return fc_cls.read_file(page_url)


def from_featureserver(
    fc_cls: type,
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    max_records: int | None = None,
    page_size: int = 1000,
    max_pages: int = 1000,
) -> Any:
    """Read a paged ArcGIS FeatureServer layer (see FeatureCollection.from_featureserver)."""
    if page_size < 1:
        raise ValueError(f"from_featureserver: page_size must be >= 1, got {page_size}")
    if max_records is not None and max_records < 0:
        raise ValueError(f"from_featureserver: max_records must be >= 0 or None, got {max_records}")
    base = url.split("?", 1)[0].rstrip("/")
    if not base.lower().endswith("/query"):
        base = f"{base}/query"
    pages, first_crs = collect_featureserver_pages(fc_cls, base, where, out_fields, max_records, page_size, max_pages)
    # Concatenate in one pass (pd.concat preserves the shared CRS); repeatedly calling .concat()
    # re-sets the CRS and trips a geopandas DeprecationWarning.
    if pages:
        return fc_cls(pd.concat(pages, ignore_index=True))
    return fc_cls(gpd.GeoDataFrame(geometry=[], crs=first_crs))


def collect_featureserver_pages(
    fc_cls: type, base: str, where: str, out_fields: str, max_records: int | None, page_size: int, max_pages: int
) -> tuple[list, Any]:
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
        this_page = page_size if max_records is None else min(page_size, max_records - fetched)
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
