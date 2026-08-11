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

import math
import os
import threading
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import pyogrio
import pyproj
from geopandas import GeoDataFrame
from pyproj.exceptions import CRSError as _PyprojCRSError
from shapely.geometry import box

from pyramids import _io as _pyramids_io
from pyramids.base._errors import FeatureError
from pyramids.base._utils import import_pyarrow
from pyramids.base.crs import _pyproj_crs_via_gdal
from pyramids.base.remote import is_remote, to_fsspec_url

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
    gdf = _read_file_healing_crs(resolved, passthrough)
    return fc_cls(gdf)


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
