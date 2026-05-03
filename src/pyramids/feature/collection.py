"""FeatureCollection — a GeoDataFrame with pyramids-specific GIS methods.

The `feature` subpackage is organised as:

- :mod:`pyramids.feature.collection` (this module) — the
  :class:`FeatureCollection` class.
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors (`create_polygon`, `create_point`, `get_coords`,
  `explode_gdf`, `multi_geom_handler`, …).
- :mod:`pyramids.feature._ogr` — private OGR bridge.

CRS / EPSG / reprojection helpers (`get_epsg_from_prj`,
`reproject_coordinates`, `create_sr_from_proj`) live in
:mod:`pyramids.base.crs`. The :class:`FeatureCollection` class
exposes the most-commonly-used ones as static-method delegates for
ergonomic continuity.

`FeatureCollection` is a direct subclass of
:class:`geopandas.GeoDataFrame`. Every GeoDataFrame method
is inherited; pyramids adds rasterization, Dataset interop, vertex
extraction, and CRS-helper delegates on top. `ogr.DataSource` is
internal only; see :mod:`pyramids.feature._ogr`.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pyramids.feature._lazy_collection import LazyFeatureCollection

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids import _io as _pyramids_io
from pyramids.base._errors import CRSError, FeatureError, GeometryWarning
from pyramids.base import crs as _crs
from pyramids.base._utils import Catalog, import_pyarrow
from pyramids.base.remote import is_remote
from pyramids.basemap.basemap import add_basemap
from pyramids.feature import geometry as _geom

CATALOG = Catalog(raster_driver=False)

# default per-chunk batch size for `iter_features` when the
# user does not pass `chunksize`. Matches pyogrio's own
# `read_dataframe` default for row-group streaming, so the fast
# path (GeoParquet + row_group tile strategy) does not re-chunk.
_DEFAULT_ITER_BATCH_SIZE: int = 1000


# target bytes per partition when read_file(backend="dask") is
# called with neither npartitions nor chunksize. 128 MiB is
# dask-geopandas' own read_parquet default and a reasonable size for
# shapely-heavy geometry ops on modern cores. Small enough that even a
# 1 GiB shapefile gets 8 partitions; large enough to avoid one-partition-
# per-10-rows pathologies on huge feature counts.
_LAZY_TARGET_BYTES_PER_PARTITION: int = 128 * 1024 * 1024


def _resolve_lazy_partitioning(
    path: str,
    npartitions: int | None,
    chunksize: int | None,
) -> dict[str, Any]:
    """default `npartitions` from file size when not given.

    Called by `read_file(backend="dask")`. If the caller supplies
    either `npartitions` or `chunksize` we honor it verbatim. If
    they supply neither, we stat the resolved path and pick
    `npartitions = max(1, ceil(size / 128 MiB))`.

    On cloud / virtual-FS paths (`/vsi*`, `http(s)://`, `s3://`,
    etc.) `os.stat` can't size the file cheaply — there we fall
    back to `npartitions=1` rather than emit a pre-flight HEAD
    request with ambiguous semantics. Users who want more partitions
    on cloud-hosted files should pass `npartitions=` explicitly.

    Args:
        path: The already-`_to_vsi`-resolved path string.
        npartitions: User-supplied partition count, if any.
        chunksize: User-supplied rows-per-partition, if any.

    Returns:
        dict: kwargs to forward to :func:`dask_geopandas.read_file`.
        Exactly one of `npartitions` / `chunksize` is populated.
    """
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
                1,
                math.ceil(size / _LAZY_TARGET_BYTES_PER_PARTITION),
            )
    return kwargs


def _require_pyarrow() -> None:
    """Raise a pyramids-branded ImportError if pyarrow is absent.

    `geopandas.read_parquet` / `GeoDataFrame.to_parquet` raise a
    generic ImportError that mentions neither `pyramids-gis` nor
    the `[parquet]` extra. This helper surfaces the install
    instruction up front so the Raises docstring is truthful.
    """
    import_pyarrow(
        "GeoParquet support requires the optional 'pyarrow' "
        "dependency. Install with: pip install 'pyramids-gis[parquet]' "
        "(or `pixi add pyarrow` in a pixi workspace)."
    )


# module-level LRU cache backing `FeatureCollection.list_layers`.
# Keyed on the already-resolved `str` path (post `_parse_path`). The
# tuple return type plays nicely with `functools.lru_cache` (lists are
# unhashable and would break LRU internals if returned directly).
@functools.lru_cache(maxsize=128)
def _list_layers_cached(resolved_path: str) -> tuple[str, ...]:
    """Return a tuple of layer names for a resolved path (memoised)."""
    import pyogrio

    arr = pyogrio.list_layers(resolved_path)
    return tuple(str(row[0]) for row in arr)


class FeatureCollection(GeoDataFrame):
    """A :class:`geopandas.GeoDataFrame` with pyramids-specific GIS methods.

    `FeatureCollection` *is a* `GeoDataFrame` — ``isinstance(fc,
    GeoDataFrame)` is `True`` — so every geopandas method is
    available directly. Pyramids adds rasterization, Dataset interop,
    vertex extraction, and CRS helpers on top.

    The OGR/GDAL backend is internal only; see
    :mod:`pyramids.feature._ogr`.
    """

    @property
    def _constructor(self):
        """Return the type pandas uses when constructing new frames."""
        return FeatureCollection

    # merge with GeoDataFrame._metadata instead of replacing it.
    # The parent class lists `_geometry_column_name` (the name of the
    # active geometry column); overriding _metadata with just our own
    # entries drops that attribute on pickle / copy / concat, and the
    # restored object can no longer find its geometry column. Always
    # splat the parent's list first.
    # dedupe via `dict.fromkeys` so that if a future geopandas
    # release adds one of our own names to its own `_metadata` list,
    # the pyramids subclass does not carry a duplicate entry. Python
    # preserves insertion order in dicts since 3.7, so the parent's
    # ordering is preserved.
    _metadata: list[str] = list(
        dict.fromkeys(
            [
                *GeoDataFrame._metadata,
                "_epsg_cache_crs",
                "_epsg_cache_value",
            ]
        )
    )
    """Instance attributes pandas must preserve across copy/slice/pickle.

    Holds:

    * `GeoDataFrame._metadata` (currently `_geometry_column_name`)
      — required for pickle round-trips to remember which column is
      the active geometry column.
    * `_epsg_cache_crs` / `_epsg_cache_value` — the EPSG
      cache.

    The list is wrapped in `list(dict.fromkeys(...))` so that a
    future geopandas release adding one of our own names to its own
    `_metadata` list does not produce a duplicate entry. `dict`
    preserves insertion order since Python 3.7, so the parent's
    ordering is preserved.
    """

    def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Construct a FeatureCollection.

        Accepts anything :class:`geopandas.GeoDataFrame` accepts.
        Rejects `ogr.DataSource` / `gdal.Dataset` with a clear error
        .
        """
        if isinstance(data, (ogr.DataSource, gdal.Dataset)):
            raise TypeError(
                "FeatureCollection no longer accepts ogr.DataSource or "
                "gdal.Dataset objects. OGR is an internal implementation "
                "detail. Use FeatureCollection.read_file(path) to load a "
                "file, or pass a GeoDataFrame."
            )
        super().__init__(data, *args, **kwargs)

    def __enter__(self) -> FeatureCollection:
        """Enter a context-managed block. Returns `self`.

        Returns:
            FeatureCollection: `self` — the exact same instance, so
            `with... as fc:` binds `fc` to this collection.

        Examples:
            - Use as a context manager and access rows inside the block:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1, 2]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs="EPSG:4326",
                ... )
                >>> with FeatureCollection(gdf) as fc:
                ...     n = len(fc)
                >>> n
                2

                ```
            - Exceptions raised inside the block still propagate:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> try:
                ...     with fc:
                ...         raise RuntimeError("boom")
                ... except RuntimeError as err:
                ...     print(err)
                boom

                ```
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit the context-managed block. Calls :meth:`close`.

        Args:
            exc_type: Exception class if the block raised, else `None`.
            exc: Exception instance if the block raised, else `None`.
            tb: Traceback for the raised exception, else `None`.

        Returns:
            bool: Always `False` — exceptions from inside the `with`
            block propagate to the caller rather than being swallowed.

        Examples:
            - The clean-exit path returns `False` so nothing is swallowed:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.__exit__(None, None, None)
                False

                ```
            - A `with` block that finishes normally just releases the FC:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ... )
                >>> with FeatureCollection(gdf) as fc:
                ...     pass
                >>> len(fc)
                1

                ```
        """
        self.close()
        return False

    def close(self) -> None:
        """Release resources held by this FeatureCollection.

        No-op today (the OGR bridge is self-cleaning). Exists so future
        resource-holding features have an idiomatic release point.

        Returns:
            None: This method does not return a value.

        Examples:
            - `close()` is idempotent — calling it repeatedly is safe:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.close()
                >>> fc.close()
                >>> len(fc)
                1

                ```
            - The collection remains usable after `close` (no-op today):
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"v": [7]}, geometry=[Point(2, 3)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.close()
                >>> fc.epsg
                4326

                ```
        """
        return None

    @classmethod
    def from_features(
        cls,
        features: Iterable[Any],
        *,
        crs: Any = None,
        columns: list[str] | None = None,
    ) -> FeatureCollection:
        """Build a FeatureCollection from feature-shaped inputs.

        Delegates to :meth:`geopandas.GeoDataFrame.from_features` and
        wraps the result. Accepts any of the shapes that method
        accepts:

        * a list (or iterator) of GeoJSON feature dicts of the form
          `{"type": "Feature", "geometry": {...}, "properties": {...}}`,
        * any object exposing `__geo_interface__` (shapely
          geometries, fiona records, custom feature classes), or
        * a bare `FeatureCollection` dict (`{"type":
          "FeatureCollection", "features": [...]}`).

        Args:
            features (Iterable):
                Feature dicts of the form
                `{"type": "Feature", "geometry": {...}, "properties": {...}}`,
                or any `__geo_interface__` provider. Also accepts a
                bare `FeatureCollection` dict.
            crs:
                CRS to attach to the result (EPSG int, `"EPSG:4326"`,
                WKT, Proj, or a :class:`pyproj.CRS`). `None` leaves
                the CRS unset.
            columns (list[str] | None):
                Explicit column order for the output. When `None`,
                geopandas infers columns from the first feature.

        Returns:
            FeatureCollection: A new FC backed by the supplied features.

        Raises:
            ValueError: If `features` is empty or exhausted before any
                feature is consumed. An empty GeoDataFrame from
                `from_features` has no `geometry` column, which
                breaks downstream pyramids methods that assume the
                column exists. Fail fast instead.

        Examples:
            - Build from a list of feature dicts:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> feats = [
                ...     {"type": "Feature",
                ...      "geometry": {"type": "Point", "coordinates": [0, 0]},
                ...      "properties": {"name": "a"}},
                ...     {"type": "Feature",
                ...      "geometry": {"type": "Point", "coordinates": [1, 1]},
                ...      "properties": {"name": "b"}},
                ... ]
                >>> fc = FeatureCollection.from_features(feats, crs=4326)
                >>> len(fc)
                2
                >>> fc.epsg
                4326

                ```
        """
        # materialise an iterator so we can detect the empty case
        # before handing off to geopandas. `geopandas.from_features([])`
        # returns a GeoDataFrame with no `geometry` column, which
        # breaks every pyramids op that assumes the column exists.
        features_list = list(features)
        if not features_list:
            raise ValueError(
                "from_features requires at least one feature. An empty "
                "iterable would produce a GeoDataFrame with no geometry "
                "column, which breaks downstream pyramids methods."
            )
        gdf = gpd.GeoDataFrame.from_features(features_list, crs=crs, columns=columns)
        return cls(gdf)

    @classmethod
    def from_records(
        cls,
        records: Any,
        *,
        geometry: str = "geometry",
        crs: Any = None,
        orient: str = "records",
    ) -> FeatureCollection:
        """Build a FeatureCollection from dict records.

        Two input orientations are accepted (C26 added the second):

        * `orient="records"` (default) — an iterable of per-row dicts,
          each of the form `{column: value,..., geometry: <shapely>}`.
          The dict's keys become column names; the key named by
          `geometry` must hold a shapely geometry.
        * `orient="list"` — a single columnar dict mapping each
          column name to a list of values of equal length, for
          example `{"id": [1, 2], "geometry": [pt_a, pt_b]}`.

        Useful for ingesting rows from an API response that doesn't
        emit GeoJSON but already has shapely geoms.

        Args:
            records:
                Per-row iterable of dicts when `orient="records"`, or a
                single columnar dict when `orient="list"`.
            geometry (str):
                Name of the column / key holding the shapely geometry.
                Default `"geometry"`.
            crs:
                CRS to attach (same forms as :meth:`from_features`).
            orient (str):
                `"records"` (default) or `"list"` — matches the
                pandas `from_dict`/`from_records` conventions.

        Returns:
            FeatureCollection: A new FC with one row per record.

        Raises:
            FeatureError: If a record is missing the `geometry`
                column.
            ValueError: If `orient` is not one of the supported
                values.

        Examples:
            - Per-row records with the default geometry key:
                ```python
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> recs = [
                ...     {"id": 1, "geometry": Point(0, 0)},
                ...     {"id": 2, "geometry": Point(1, 1)},
                ... ]
                >>> fc = FeatureCollection.from_records(recs, crs=4326)
                >>> len(fc)
                2
                >>> fc.epsg
                4326

                ```
            - Custom geometry key via the `geometry=` kwarg:
                ```python
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> recs = [
                ...     {"id": 1, "geom": Point(0, 0)},
                ...     {"id": 2, "geom": Point(1, 1)},
                ... ]
                >>> fc = FeatureCollection.from_records(
                ...     recs, geometry="geom", crs=4326,
                ... )
                >>> fc.geometry.name
                'geom'

                ```
            - Columnar dict via `orient="list"`:
                ```python
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> cols = {"id": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]}
                >>> fc = FeatureCollection.from_records(
                ...     cols, orient="list", crs=4326,
                ... )
                >>> list(fc["id"])
                [1, 2]

                ```
        """
        # empty-input branches both build a single-column frame
        # whose column name matches the `geometry=` kwarg, so
        # `GeoDataFrame(..., geometry=…)` sets it as the active
        # geometry column and the returned FC has
        # `geometry.name == geometry`.
        def _empty_fc() -> FeatureCollection:
            return cls(gpd.GeoDataFrame({geometry: []}, geometry=geometry, crs=crs))

        if orient == "records":
            records_list = list(records)
            if not records_list:
                return _empty_fc()
            df = pd.DataFrame.from_records(records_list)
        elif orient == "list":
            # columnar dict of equal-length lists. Straight into
            # `pd.DataFrame` which accepts this shape natively and
            # raises `ValueError` on mismatched lengths (propagated
            # to the caller as-is — the pandas message is already clear).
            if not isinstance(records, dict):
                raise ValueError(
                    f"orient='list' expects a dict of column → list; "
                    f"got {type(records).__name__}."
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
        return cls(gpd.GeoDataFrame(df, geometry=geometry, crs=crs))

    _VALID_TILE_STRATEGIES: tuple[str, ...] = (
        "auto",
        "rtree",
        "row_group",
        "none",
    )

    @classmethod
    def iter_features(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        where: str | None = None,
        chunksize: int | None = None,
        tile_strategy: str = "auto",
        include_index: bool = False,
    ) -> Any:
        """Stream features from `path` without materializing the full file.

        . Two orthogonal knobs:

        * **Chunk shape**. `chunksize=None` yields one GeoJSON-style
          dict per row (fiona idiom). `chunksize=N` yields
          :class:`FeatureCollection` batches of up to N rows each so
          batched pipelines get a DataFrame-shaped payload.
        * **Tile strategy**. Controls whether the `bbox`
          filter is pushed into the format's spatial index (rtree on
          GPKG, row-group statistics on Parquet, …) or applied after
          a full scan. Pass one of:

          - `"auto"` (default) — let pyogrio pick. For a GPKG,
            pyogrio queries the `rtree_<layer>_geom` companion
            table automatically. For a Parquet file, pyogrio /
            pyarrow push the bbox down to the row-group statistics
            and skip non-matching groups. For formats without a
            spatial index (GeoJSON, Shapefile without a `.qix`)
            this falls back to a full scan in the driver.
          - `"rtree"` — same as `"auto"`; kept as an explicit
            name so pipeline code can document intent.
          - `"row_group"` — same as `"auto"`; explicit name for
            the Parquet case.
          - `"none"` — disable index pushdown; read whole chunks
            from the driver and apply the bbox filter in Python.
            Useful when the on-disk spatial index is stale or
            suspected wrong; also exercises the "slow path" in
            tests.

        `bbox` / `where` compose with any tile_strategy. Paths run
        through :func:`pyramids._io._parse_path` so cloud URLs and
        archive paths work the same way as in :meth:`read_file`.

        Args:
            path (str | Path): File path, URL, archive path.
            layer (str | int | None): Layer selector for multi-layer
                formats.
            bbox: `(minx, miny, maxx, maxy)` filter.
            where (str | None): OGR SQL predicate.
            chunksize (int | None): `None` yields dicts, an `int`
                yields `FeatureCollection` chunks.
            tile_strategy (str): One of `"auto"`, `"rtree"`,
                `"row_group"`, `"none"`. Default `"auto"`.
            include_index (bool): When `True`, each yielded dict gets
                an additional `"id"` key whose value is the
                0-based file-row index of that feature. The chunked
                form (`chunksize=N`) attaches the same index as a
                `"_row_index"` column on the yielded FC. The indices
                stay aligned with the on-disk rows even when a
                Python-side bbox filter (`tile_strategy="none"`)
                drops some rows — only the surviving features are
                yielded, and their ids match the positions they had
                in the source file. Defaults to `False` for
                back-compat with the fiona idiom.

        Yields:
            dict | FeatureCollection: Per-feature dicts when
            `chunksize` is `None`; FeatureCollection chunks
            otherwise.

        Raises:
            ValueError: If `chunksize` is given but `< 1`, or if
                `tile_strategy` is not one of the accepted values.

        Examples:
            - Stream features one at a time as GeoJSON-style dicts:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> path = d / "pts.geojson"
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1, 2, 3]},
                ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
                ...     crs="EPSG:4326",
                ... )
                >>> gdf.to_file(path, driver="GeoJSON")
                >>> feats = list(FeatureCollection.iter_features(path))
                >>> len(feats)
                3
                >>> feats[0]["properties"]["id"]
                1

                ```
            - Stream in `chunksize=2` batches as FeatureCollection chunks:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> path = d / "pts.geojson"
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1, 2, 3]},
                ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
                ...     crs="EPSG:4326",
                ... )
                >>> gdf.to_file(path, driver="GeoJSON")
                >>> chunks = list(
                ...     FeatureCollection.iter_features(path, chunksize=2)
                ... )
                >>> [len(c) for c in chunks]
                [2, 1]

                ```
            - Invalid `chunksize` raises `ValueError`:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> gen = FeatureCollection.iter_features("anywhere", chunksize=0)
                >>> next(gen)
                Traceback (most recent call last):
                    ...
                ValueError: chunksize must be >= 1 when supplied; got 0.

                ```
        """
        if chunksize is not None and chunksize < 1:
            raise ValueError(f"chunksize must be >= 1 when supplied; got {chunksize}.")
        if tile_strategy not in cls._VALID_TILE_STRATEGIES:
            raise ValueError(
                f"tile_strategy must be one of "
                f"{cls._VALID_TILE_STRATEGIES}; got {tile_strategy!r}."
            )

        import pyogrio

        resolved = str(_pyramids_io._parse_path(path))

        # Determine how many features are in the layer so we can
        # iterate in fixed-size batches via skip_features / max_features.
        # pyogrio's read_info is O(1) per call.
        info_kwargs: dict[str, Any] = {}
        if layer is not None:
            info_kwargs["layer"] = layer
        info = pyogrio.read_info(resolved, **info_kwargs)
        total = int(info["features"])

        if chunksize is None:
            batch_size = _DEFAULT_ITER_BATCH_SIZE
        else:
            batch_size = int(chunksize)

        # D-M3: pin the engine to pyogrio. `skip_features` /
        # `max_features` are pyogrio-specific (geopandas' fiona
        # engine silently ignores them, which would turn every chunk
        # into a full scan). Pinning the engine makes the contract
        # explicit and fails fast if pyogrio is absent.
        read_kwargs: dict[str, Any] = {"engine": "pyogrio"}
        if layer is not None:
            read_kwargs["layer"] = layer
        if where is not None:
            read_kwargs["where"] = where

        # when tile_strategy is "auto"/"rtree"/"row_group",
        # forward the bbox to pyogrio which transparently uses the
        # format's spatial index. When "none", hold the bbox back
        # and apply it in Python after each chunk loads.
        pushdown_bbox = bbox if tile_strategy != "none" else None
        python_bbox = bbox if tile_strategy == "none" else None
        if pushdown_bbox is not None:
            read_kwargs["bbox"] = pushdown_bbox

        for start in range(0, total, batch_size):
            gdf_chunk = gpd.read_file(
                resolved,
                skip_features=start,
                max_features=batch_size,
                **read_kwargs,
            )
            # remember the absolute row indices before any
            # bbox-based masking so callers can map yielded features
            # back to their source rows even after a Python-side filter
            # has dropped some of them.
            if include_index:
                row_indices = list(range(start, start + len(gdf_chunk)))
            if python_bbox is not None and len(gdf_chunk) > 0:
                xmin, ymin, xmax, ymax = python_bbox
                mask = gdf_chunk.intersects(box(xmin, ymin, xmax, ymax))
                if include_index:
                    row_indices = [ri for ri, keep in zip(row_indices, mask) if keep]
                gdf_chunk = gdf_chunk[mask]
            if chunksize is None:
                iterator = gdf_chunk.iterfeatures(na="null")
                if include_index:
                    for ri, feat in zip(row_indices, iterator):
                        feat["id"] = ri
                        yield feat
                else:
                    for feat in iterator:
                        yield feat
            else:
                chunk_fc = cls(gdf_chunk)
                if include_index:
                    chunk_fc["_row_index"] = row_indices
                yield chunk_fc

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        bbox: tuple[float, float, float, float] | Any = None,
        mask: Any = None,
        rows: slice | int | None = None,
        columns: list[str] | None = None,
        where: str | None = None,
        backend: str = "pandas",
        npartitions: int | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> FeatureCollection | LazyFeatureCollection:
        """Read a vector file into a FeatureCollection.

        path is first routed through
        :func:`pyramids._io._parse_path`, which handles:

        * Cloud-URL rewriting (`s3://`, `gs://`, `az://`,
          `abfs://`, `http(s)://`, `file://` → GDAL `/vsi*/`
          form). verified end-to-end through an HTTP test.
          For AWS / GCS / Azure credentials either set the standard
          environment variables (`AWS_ACCESS_KEY_ID`,
          `AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`,
          `AZURE_STORAGE_CONNECTION_STRING`, …) or scope them via
          :class:`pyramids.base.remote.CloudConfig` as a context
          manager around the `read_file` call.
        * Compressed-archive dispatch for `.zip`, `.tar`, `.tar.gz`,
          `.gz` on **local** paths — the returned path is a
          `/vsizip/`, `/vsitar/` or `/vsigzip/` string that
          :func:`geopandas.read_file` (via GDAL's virtual filesystem)
          can open directly. You can either pass just the archive
          path (first contained file wins) or
          `archive.zip/inner.geojson` to target a specific member.
          Cloud + archive chaining (`http://host/x.zip`) is not
          automatic today — if you need it, stage the archive
          locally first or use `CloudConfig` with an explicit
          `/vsizip//vsicurl/...` path.

        filter kwargs are pushed down to fiona/pyogrio so the
        dataset never fully materializes when only a subset is needed.

        Args:
            path (str | Path):
                File path, URL, archive path, or
                `archive.ext/inner-file` form.
            layer (str | int | None):
                Layer name or index for multi-layer formats
                (GeoPackage, GDB, KML, …). `None` reads the first /
                default layer.
            bbox:
                `(minx, miny, maxx, maxy)` tuple, or a
                `GeoDataFrame` / `GeoSeries` / shapely geometry
                whose total bounds are used. Only features
                intersecting the bbox are loaded.
            mask:
                A shapely geometry (or mapping / GeoSeries /
                GeoDataFrame) whose geometries are used as a mask —
                only features intersecting the mask are loaded. Finer
                than `bbox` (actual geometry intersection, not just
                envelope). Mutually exclusive with `bbox`.
            rows (slice | int | None):
                `int` — read at most N rows. `slice` — read the
                given range of rows. Useful for sampling.
            columns (list[str] | None):
                Restrict loaded attribute columns. Geometry is
                always loaded. `None` loads every column.
            where (str | None):
                OGR SQL `WHERE`-clause predicate pushed down to the
                driver (e.g. `"population > 10000"`). Avoids loading
                non-matching features.
            **kwargs:
                Forwarded to :func:`geopandas.read_file` verbatim for
                engine-specific options (`engine="pyogrio"`,
                `use_arrow=True`, driver-specific creation options).

        Returns:
            FeatureCollection: The (possibly filtered) features
            wrapped as a FeatureCollection.

        Examples:
            - Load a GeoJSON file:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.read_file("tests/data/coello-gauges.geojson")
                >>> len(fc) > 0
                True

                ```
        """
        resolved = _pyramids_io._parse_path(path)
        if backend == "dask":
            # dask_geopandas.read_file does NOT forward pyogrio
            # filter kwargs (bbox / mask / rows / columns / where) —
            # silently dropping them was the bug. Raise a clear
            # ValueError instead so users know to either pre-filter
            # or call .compute() and filter eagerly.
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
            try:
                import dask_geopandas
            except ImportError as exc:
                raise ImportError(
                    "backend='dask' requires the optional "
                    "'dask-geopandas' dependency. Install with: "
                    "pip install 'pyramids-gis[parquet-lazy]'"
                ) from exc
            # default npartitions from file size when neither
            # kwarg was supplied; one-partition fallback defeats the
            # point of going lazy.
            partition_kwargs = _resolve_lazy_partitioning(
                resolved,
                npartitions,
                chunksize,
            )
            # wrap the lazy return as a LazyFeatureCollection so the
            # dask branch stays inside the pyramids type system.
            from pyramids.feature._lazy_collection import LazyFeatureCollection

            dask_gdf = dask_geopandas.read_file(resolved, **partition_kwargs)
            return LazyFeatureCollection.from_dask_gdf(dask_gdf)
        if backend != "pandas":
            raise ValueError(f"backend must be 'pandas' or 'dask', got {backend!r}")
        # Only pass kwargs that were actually supplied — passing the
        # defaults (None) is fine for some geopandas engines but
        # confuses others. Build a clean kwargs dict.
        passthrough: dict[str, Any] = {}
        if layer is not None:
            passthrough["layer"] = layer
        if bbox is not None:
            passthrough["bbox"] = bbox
        if mask is not None:
            passthrough["mask"] = mask
        if rows is not None:
            passthrough["rows"] = rows
        if columns is not None:
            passthrough["columns"] = columns
        if where is not None:
            passthrough["where"] = where
        passthrough.update(kwargs)
        gdf = gpd.read_file(resolved, **passthrough)
        return cls(gdf)

    @property
    def epsg(self) -> int | None:
        """EPSG code of this FeatureCollection's CRS (cached).

        The value is cached per CRS-object identity so repeated access
        on hot paths skips the `pyproj.CRS.to_epsg` call. The cache
        auto-invalidates whenever `self.crs` is replaced.

        identity-miss falls back to equality. If `self.crs` has
        been reassigned to a different CRS object that nevertheless
        compares equal to the cached one (e.g. `fc.crs = pyproj.CRS(
        "EPSG:4326")` on a frame already in EPSG:4326), we adopt the
        new object as the cache key and skip the `.to_epsg()` call.
        Only when the value really differs do we recompute.

        the equality fallback is cheaper than a fresh
        `.to_epsg()` (which re-parses the CRS) but it is not free —
        `pyproj.CRS.__eq__` does a WKT2 string comparison. If a
        future pandas/geopandas release stops returning the same
        `self.crs` object identity across accesses, the fallback
        runs on every `fc.epsg` and adds up on hot loops. Switch
        the cache key to `self.crs.to_wkt()` if a profile ever
        shows this dominating.

        Returns:
            int | None: The integer EPSG code if the CRS is registered
            in the EPSG authority; `None` when the FC has no CRS set
            or when its CRS cannot be mapped to a single EPSG code.

        Examples:
            - Frame built with WGS84 reports EPSG 4326:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.epsg
                4326

                ```
            - A frame without a CRS returns `None`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
                ... )
                >>> fc.epsg is None
                True

                ```
            - Reprojecting to Web Mercator updates the cached code:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc = fc.to_crs(3857)
                >>> fc.epsg
                3857

                ```
        """
        crs = self.crs
        cached_crs = getattr(self, "_epsg_cache_crs", None)
        if cached_crs is crs:
            return getattr(self, "_epsg_cache_value", None)
        # try equality before falling back to a fresh to_epsg() call.
        # pyproj.CRS comparison is cheaper than a full re-parse, and the
        # common "reassign an equivalent CRS" case (e.g. set_crs chain)
        # should stay in the fast path.
        if cached_crs is not None and crs is not None:
            try:
                equivalent = cached_crs == crs
            except (TypeError, ValueError):
                equivalent = False
            if equivalent:
                object.__setattr__(self, "_epsg_cache_crs", crs)
                return getattr(self, "_epsg_cache_value", None)
        if crs is None:
            value: int | None = None
        else:
            code = crs.to_epsg()
            value = int(code) if code is not None else None
        object.__setattr__(self, "_epsg_cache_crs", crs)
        object.__setattr__(self, "_epsg_cache_value", value)
        return value

    @property
    def top_left_corner(self) -> list[Number]:
        """Top-left corner `[xmin, ymax]` of the total bounds.

        Returns:
            list[Number]: Two-element list `[xmin, ymax]` — the
            minimum x-coordinate paired with the maximum y-coordinate
            of the union of all geometry bounds.

        Examples:
            - Two points span a unit square — the top-left is `[0, 1]`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(0, 0), Point(1, 1)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.top_left_corner
                [0.0, 1.0]

                ```
            - Offset points yield the offset top-left corner:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(10, 20), Point(15, 30)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.top_left_corner
                [10.0, 30.0]

                ```
        """
        bounds = self.total_bounds.tolist()
        return [bounds[0], bounds[3]]

    @property
    def column(self) -> list[str]:
        """Deprecated alias for :attr:`columns` returning a `list[str]`.

        Returns:
            list[str]: Column names in their current order, including
            the active geometry column.

        Examples:
            - A frame with an `id` field reports both columns:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.column
                ['id', 'geometry']

                ```
            - Multiple attribute columns appear in insertion order:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"name": ["a"], "pop": [100]},
                ...         geometry=[Point(0, 0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.column
                ['name', 'pop', 'geometry']

                ```
        """
        return self.columns.tolist()

    def __str__(self) -> str:
        """Return a short, pyramids-branded summary of the collection."""
        n = len(self)
        cols = self.columns.tolist()
        epsg = self.epsg
        return f"FeatureCollection({n} features, " f"columns={cols}, epsg={epsg})"

    def __repr__(self) -> str:
        """Return a pyramids-branded repr."""
        return (
            f"FeatureCollection(n_features={len(self)}, "
            f"columns={self.columns.tolist()}, epsg={self.epsg})"
        )

    @property
    def schema(self) -> dict:
        """Fiona-style schema: geometry type + field-type dict.

        Returns a dict shaped like fiona's `schema` attribute so
        callers migrating from `fiona.open(path).schema` can consume
        this without rewriting. The dict has three keys:

        * `"geometry"`: single string (`"Point"`, `"Polygon"`,
          …) when every row has the same geom type, otherwise
          `"Unknown"`.
        * `"properties"`: `{column_name: dtype_string}` for every
          non-geometry column.
        * `"crs"`: the :attr:`crs` as a :class:`pyproj.CRS` object,
          or `None` when the FC has no CRS set. Matches
          fiona's convention — callers migrating from
          `fiona.open(path).schema['crs']` can consume it directly.

        Empty FeatureCollections (`len(self) == 0`) report
        `"Unknown"` for the geometry type.

        Returns:
            dict: Three-key dict with `"geometry"`, `"properties"`,
            and `"crs"`.

        Examples:
            - Homogeneous point collection reports `"Point"`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(0, 0), Point(1, 1)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> schema = fc.schema
                >>> schema["geometry"]
                'Point'
                >>> schema["properties"]
                {'id': 'int64'}
                >>> schema["crs"].to_epsg()
                4326

                ```
            - Mixed geometry types collapse to `"Unknown"`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point, LineString
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(0, 0), LineString([(0, 0), (1, 1)])],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.schema["geometry"]
                'Unknown'

                ```
            - Frames without a CRS return `crs=None`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
                ... )
                >>> fc.schema["crs"] is None
                True

                ```
        """
        geom_types = {g.geom_type for g in self.geometry if g is not None}
        if len(geom_types) == 1:
            (geom_type,) = geom_types
        else:
            geom_type = "Unknown"
        properties = {
            col: str(dt) for col, dt in self.dtypes.items() if col != "geometry"
        }
        return {
            "geometry": geom_type,
            "properties": properties,
            "crs": self.crs,
        }

    @classmethod
    def list_layers(cls, path: str | Path) -> list[str]:
        """List every vector-layer name in `path`.

        Routes through :func:`pyramids._io._parse_path` so the same
        cloud-URL / archive rewriting that :meth:`read_file` uses
        applies here too. Uses :func:`pyogrio.list_layers` under the
        hood (geopandas' default engine).

        results are memoised behind a 128-entry LRU cache keyed on
        the resolved `str` path. Re-calling `list_layers` on the
        same cloud URL or local path in a loop now costs one hash
        lookup instead of one datasource open. Call
        :meth:`list_layers_cache_clear` to invalidate after an
        out-of-band write.

        Args:
            path (str | Path):
                File path, URL, or archive path. Single-layer formats
                like GeoJSON return one name; multi-layer formats
                (GPKG, GDB, KML) return every layer.

        Returns:
            list[str]: Layer names in the order the driver reports them.

        Raises:
            FileNotFoundError: If `path` is a local filesystem path
                that does not exist. Cloud URLs and `/vsi*` paths
                skip this check and defer to the underlying driver
                . Previously all failures surfaced as an opaque
                `VectorDriverError("Failed to open datasource")`.

        Examples:
            - A single-layer GeoJSON returns one name derived from the filename:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> path = d / "pts.geojson"
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ... )
                >>> gdf.to_file(path, driver="GeoJSON")
                >>> FeatureCollection.list_layers(path)
                ['pts']

                ```
            - A missing local path raises `FileNotFoundError`:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> FeatureCollection.list_layers("does/not/exist.geojson")
                Traceback (most recent call last):
                    ...
                FileNotFoundError: list_layers: no file at 'does/not/exist.geojson'.

                ```
        """
        # pre-check local-path existence so the caller sees
        # a `FileNotFoundError` naming the path instead of a generic
        # driver-open failure. Defer to `base.remote.is_remote` as
        # the single source of truth for which schemes are remote —
        # the previous hardcoded prefix tuple would silently treat any
        # future scheme as local and raise a misleading error.
        path_str = str(path)
        if not is_remote(path_str):
            local = Path(path_str)
            if not local.exists():
                raise FileNotFoundError(f"list_layers: no file at {path_str!r}.")

        resolved = str(_pyramids_io._parse_path(path))
        return list(_list_layers_cached(resolved))

    @classmethod
    def list_layers_cache_clear(cls) -> None:
        """Clear the C15 LRU cache backing :meth:`list_layers`.

        Call this after writing a new layer to an existing multi-layer
        file (e.g. a GPKG) if you then want :meth:`list_layers` to see
        the new layer. Otherwise the 128-entry LRU cache is self-
        managing and callers do not need to touch it.

        Returns:
            None: This method does not return a value.

        Examples:
            - Clearing an empty cache is a safe no-op:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> FeatureCollection.list_layers_cache_clear()
                >>> FeatureCollection.list_layers_cache_clear()

                ```
            - After an out-of-band write, clear the cache so the next
              `list_layers` call re-reads the updated file:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> path = d / "pts.geojson"
                >>> gpd.GeoDataFrame(
                ...     {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ... ).to_file(path, driver="GeoJSON")
                >>> _ = FeatureCollection.list_layers(path)
                >>> FeatureCollection.list_layers_cache_clear()
                >>> FeatureCollection.list_layers(path)
                ['pts']

                ```
        """
        _list_layers_cached.cache_clear()

    @classmethod
    def open_arrow(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        columns: list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        where: str | None = None,
        batch_size: int | None = None,
    ) -> Any:
        """Open a vector file as a streaming :class:`pyarrow.RecordBatchReader`.

        Thin wrapper over :func:`pyogrio.raw.open_arrow` that surfaces
        the underlying Arrow RecordBatch iterator. Rows are yielded in
        batches, so callers can iterate through multi-GB datasets
        without materializing the whole table in memory — useful for
        building custom dask partitioners.

        Args:
            path: Vector file path (Shapefile, GPKG, FlatGeobuf,
                GeoJSON, GeoParquet,...). Routed through
                :func:`pyramids._io._parse_path` so cloud URLs work.
            layer: Layer name or index for multi-layer formats.
            columns: Attribute columns to load (`geometry` is
                always included).
            bbox: `(minx, miny, maxx, maxy)` filter.
            where: OGR SQL `WHERE` predicate pushed down to the
                driver.
            batch_size: Requested RecordBatch size in rows. `None`
                uses the driver default.

        Returns:
            pyarrow.RecordBatchReader: A streaming reader. Call
            `.read_all()` to materialise, or iterate for row-batch
            consumption.

        Raises:
            ImportError: If :mod:`pyogrio` is not installed.
        """
        try:
            from pyogrio.raw import open_arrow
        except ImportError as exc:
            raise ImportError(
                "open_arrow requires the optional 'pyogrio' dependency. "
                "Install with: pip install pyogrio"
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
        return open_arrow(resolved, **kwargs)

    @classmethod
    def read_parquet(
        cls,
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
        """Read a GeoParquet file into a FeatureCollection.

        GeoParquet is a cloud-native columnar vector format (OGC-
        adopted December 2024) — faster to scan than GeoJSON, smaller
        than Shapefile, and partitioned in a way that suits distributed
        compute. This method is a thin wrapper around
        :func:`geopandas.read_parquet`; the path is first routed
        through :func:`pyramids._io._parse_path` so cloud URLs
        (`s3://`, `gs://`, `http(s)://`, …) resolve the same way
        they do in :meth:`read_file`.

        Requires the optional :mod:`pyarrow` dependency. Install with
        `pip install pyramids-gis[parquet]` or
        `pixi add pyarrow`.

        Args:
            path (str | Path):
                Local path, cloud URL, or any form
                :func:`pyramids._io._parse_path` accepts.
            columns (list[str] | None):
                Project a subset of columns — Parquet's columnar
                layout makes this a true I/O win, unlike row-oriented
                formats. `geometry` is always loaded. `None`
                loads every column.
            bbox (tuple[float, float, float, float] | None):
                `(minx, miny, maxx, maxy)` spatial filter.
                Forwarded to :func:`geopandas.read_parquet` which uses
                the file's GeoParquet spatial-index metadata when
                present to skip non-matching row groups — a true I/O
                win on large files. `None` (default) loads every
                feature.
            **kwargs:
                Forwarded to :func:`geopandas.read_parquet`
                (`storage_options=` for fsspec, etc.).

        Returns:
            FeatureCollection: The file's features wrapped as a
            FeatureCollection.

        Raises:
            ImportError: If :mod:`pyarrow` is not installed, with a
                pyramids-branded message pointing at the
                `[parquet]` optional-dependency extra (D-M5).

        Examples:
            - Round-trip a small FC through GeoParquet (requires pyarrow):
                ```python
                >>> import tempfile  # doctest: +SKIP
                >>> from pathlib import Path  # doctest: +SKIP
                >>> import geopandas as gpd  # doctest: +SKIP
                >>> from shapely.geometry import Point  # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection  # doctest: +SKIP
                >>> d = Path(tempfile.mkdtemp())  # doctest: +SKIP
                >>> path = d / "pts.parquet"  # doctest: +SKIP
                >>> gpd.GeoDataFrame(
                ...     {"id": [1, 2]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs="EPSG:4326",
                ... ).to_parquet(path)  # doctest: +SKIP
                >>> fc = FeatureCollection.read_parquet(path)  # doctest: +SKIP
                >>> len(fc)  # doctest: +SKIP
                2
                >>> fc.epsg  # doctest: +SKIP
                4326

                ```
            - Project a subset of columns to speed up I/O on wide files:
                ```python
                >>> fc = FeatureCollection.read_parquet(  # doctest: +SKIP
                ...     "s3://bucket/big.parquet",
                ...     columns=["id", "geometry"],
                ... )
                >>> fc.column  # doctest: +SKIP
                ['id', 'geometry']

                ```
            - A missing pyarrow dependency raises a branded `ImportError`:
                ```python
                >>> FeatureCollection.read_parquet("x.parquet")  # doctest: +SKIP
                Traceback (most recent call last):
                    ...
                ImportError: GeoParquet support requires the optional 'pyarrow'...

                ```
        """
        resolved = _pyramids_io._parse_path(path)
        if backend == "dask":
            # check deps in order of specificity — the backend
            # request is the more specific signal, so the
            # dask-geopandas hint beats the generic pyarrow one.
            # When both are missing, the dask-geopandas error names
            # the extra that installs both ([parquet-lazy]).
            try:
                import dask_geopandas
            except ImportError as exc:
                raise ImportError(
                    "backend='dask' requires the optional "
                    "'dask-geopandas' dependency. Install with: "
                    "pip install 'pyramids-gis[parquet-lazy]'"
                ) from exc
            dask_kwargs: dict[str, Any] = {}
            if columns is not None:
                dask_kwargs["columns"] = columns
            if split_row_groups is not None:
                dask_kwargs["split_row_groups"] = split_row_groups
            if filters is not None:
                dask_kwargs["filters"] = filters
            if blocksize is not None:
                dask_kwargs["blocksize"] = blocksize
            if storage_options is not None:
                dask_kwargs["storage_options"] = storage_options
            dask_kwargs.update(kwargs)
            # dask_geopandas is installed → assert pyarrow too, so
            # the user gets the pyramids-branded hint (not the
            # upstream message dask_geopandas would emit when it tries
            # to read). `[parquet-lazy]` pulls both.
            _require_pyarrow()
            # wrap the lazy return as a LazyFeatureCollection so the
            # dask branch stays inside the pyramids type system.
            from pyramids.feature._lazy_collection import LazyFeatureCollection

            dask_gdf = dask_geopandas.read_parquet(resolved, **dask_kwargs)
            return LazyFeatureCollection.from_dask_gdf(dask_gdf)
        if backend != "pandas":
            raise ValueError(f"backend must be 'pandas' or 'dask', got {backend!r}")
        _require_pyarrow()
        # geopandas 1.x forwards **kwargs straight into
        # `pyarrow.parquet.read_table`, which has never accepted the
        # pandas-style `engine=` kwarg. `_require_pyarrow()` above
        # already hard-guarantees the pyarrow backend, so no injection
        # is needed here. If geopandas ever reintroduces a fastparquet
        # path it will be opt-in via a new kwarg, not a silent switch.
        passthrough: dict[str, Any] = {}
        passthrough.update(kwargs)
        if columns is not None:
            passthrough["columns"] = columns
        if bbox is not None:
            passthrough["bbox"] = bbox
        if storage_options is not None:
            passthrough["storage_options"] = storage_options
        gdf = gpd.read_parquet(resolved, **passthrough)
        return cls(gdf)

    def to_parquet(
        self,
        path: str | Path,
        *,
        compression: str = "snappy",
        index: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this FeatureCollection to GeoParquet.

        Thin wrapper around :meth:`geopandas.GeoDataFrame.to_parquet`
        that defaults :param:`compression` to `"snappy"` — the
        format-standard tradeoff between speed and size.

        Requires the optional :mod:`pyarrow` dependency. Install with
        `pip install pyramids-gis[parquet]` or
        `pixi add pyarrow`.

        Args:
            path (str | Path):
                Destination file path.
            compression (str):
                Parquet compression codec — `"snappy"` (default),
                `"gzip"`, `"brotli"`, `"lz4"`, `"zstd"`, or
                `"none"`. `"snappy"` is the GeoParquet-spec
                recommended default.
            index (bool | None):
                Whether to include the pandas index as a column.
                `None` (default) uses geopandas' default behavior:
                preserve a non-default index, drop the default
                `RangeIndex`.
            **kwargs:
                Forwarded to :meth:`geopandas.GeoDataFrame.to_parquet`.

        Raises:
            ImportError: If :mod:`pyarrow` is not installed, with a
                pyramids-branded message pointing at the
                `[parquet]` optional-dependency extra (D-M5).

        Examples:
            - Write a FeatureCollection with the default snappy codec:
                ```python
                >>> import tempfile  # doctest: +SKIP
                >>> from pathlib import Path  # doctest: +SKIP
                >>> import geopandas as gpd  # doctest: +SKIP
                >>> from shapely.geometry import Point  # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection  # doctest: +SKIP
                >>> d = Path(tempfile.mkdtemp())  # doctest: +SKIP
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(0, 0), Point(1, 1)],
                ...         crs="EPSG:4326",
                ...     )
                ... )  # doctest: +SKIP
                >>> path = d / "out.parquet"  # doctest: +SKIP
                >>> fc.to_parquet(path)  # doctest: +SKIP
                >>> path.exists()  # doctest: +SKIP
                True

                ```
            - Pick a different codec (e.g. zstd for better compression):
                ```python
                >>> import tempfile  # doctest: +SKIP
                >>> from pathlib import Path  # doctest: +SKIP
                >>> import geopandas as gpd  # doctest: +SKIP
                >>> from shapely.geometry import Point  # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection  # doctest: +SKIP
                >>> d = Path(tempfile.mkdtemp())  # doctest: +SKIP
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )  # doctest: +SKIP
                >>> fc.to_parquet(d / "out.parquet", compression="zstd")  # doctest: +SKIP

                ```
        """
        _require_pyarrow()
        super().to_parquet(path, compression=compression, index=index, **kwargs)

    def to_file(
        self,
        path: str | Path,
        driver: str = "geojson",
        *,
        layer: str | None = None,
        mode: str = "w",
        **creation_options: Any,
    ) -> None:
        """Write this FeatureCollection to a vector file.

        `layer`, `mode`, and arbitrary driver creation
        options are now first-class kwargs. Previously callers had to
        rely on implicit `**kwargs` forwarding, which hurt
        discoverability.

        Args:
            path (str | Path):
                Destination file path.
            driver (str):
                Driver alias (e.g. `"geojson"`, `"gpkg"`) or
                literal GDAL driver name (`"GeoJSON"`, `"GPKG"`,
                `"ESRI Shapefile"`). Resolved via :class:`Catalog`.
            layer (str | None):
                Layer name for multi-layer drivers (GPKG, GDB, …).
                Writing two layers into the same GPKG is the canonical
                use case. `None` defers to the driver default.
            mode (str):
                `"w"` (default) overwrites; `"a"` appends to an
                existing layer. Append support depends on the driver
                — GPKG and Shapefile accept it, GeoJSON does not.
            **creation_options:
                Driver-specific creation options, forwarded to the
                underlying engine (pyogrio / fiona). Examples:

                * GPKG: `SPATIAL_INDEX="YES"`, `FID="id"`.
                * Shapefile: `ENCODING="UTF-8"`.
                * GeoJSON: `COORDINATE_PRECISION=6`, `RFC7946=YES`.

                Keys are case-preserving and passed verbatim to the
                driver; consult the GDAL driver docs for the full
                list.

                pyogrio (the default geopandas engine on 1.0+)
                raises :class:`ValueError` with the message
                `"unrecognized option '<name>' for driver '<driver>'"`
                when a supplied option is neither in the driver's
                dataset nor its layer creation-option list. This
                surfaces typos (`SPATIAL_INDX` vs `SPATIAL_INDEX`)
                at write-time rather than silently producing a
                different file. Some drivers may still accept options
                that pyogrio does not list — verify against the
                driver's docs when in doubt.

        Raises:
            ValueError: If `mode` isn't `"w"` or `"a"`, or if a
                supplied creation option is not recognised by the
                driver (raised by pyogrio — see the `**creation_options`
                note above).

        Examples:
            - Round-trip a small FC through GeoJSON (the default driver):
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(0, 0), Point(1, 1)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> path = d / "out.geojson"
                >>> fc.to_file(path)
                >>> path.exists()
                True
                >>> FeatureCollection.read_file(path).column
                ['id', 'geometry']

                ```
            - Write to GeoPackage with a named layer:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> path = d / "out.gpkg"
                >>> fc.to_file(path, driver="gpkg", layer="rivers")
                >>> FeatureCollection.list_layers(path)
                ['rivers']

                ```
            - Invalid `mode` raises `ValueError` before touching the file:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
                ...     )
                ... )
                >>> fc.to_file("ignored.geojson", mode="x")
                Traceback (most recent call last):
                    ...
                ValueError: mode must be 'w' (write) or 'a' (append); got 'x'.

                ```
        """
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' (write) or 'a' (append); got {mode!r}.")
        try:
            resolved = CATALOG.get_gdal_name(driver) or driver
        except AttributeError:
            resolved = driver

        # pin the engine to pyogrio to match :meth:`read_file` and
        # :meth:`iter_features`. Callers who want fiona for some reason
        # can override via `engine="fiona"` in creation_options, but
        # the default gets the fast path and the pyogrio-specific
        # unknown-option validation.
        passthrough: dict[str, Any] = {
            "driver": resolved,
            "mode": mode,
            "engine": "pyogrio",
        }
        if layer is not None:
            passthrough["layer"] = layer
        passthrough.update(creation_options)
        super().to_file(path, **passthrough)

    # FeatureCollection.to_dataset was moved to
    # Dataset.from_features(features,...) to break the circular import
    # that used to force a CLAUDE.md-violating inline
    # `from pyramids.dataset import Dataset` inside the method body.
    # Callers should migrate:
    # fc.to_dataset(dataset=ds, column_name="pop")
    # → Dataset.from_features(fc, template=ds, column_name="pop")
    # fc.to_dataset(cell_size=10)
    # → Dataset.from_features(fc, cell_size=10)

    # Static delegates to helper modules kept for backward compatibility
    # (callers may do `FeatureCollection.create_polygon(...)`).

    @staticmethod
    def _create_sr_from_proj(
        prj: str, string_type: str | None = None
    ) -> osr.SpatialReference:
        """Delegate to :func:`pyramids.base.crs.create_sr_from_proj`."""
        return _crs.create_sr_from_proj(prj, string_type)

    @staticmethod
    def get_epsg_from_prj(prj: str) -> int:
        """Return the EPSG code identified by a projection string.

        Thin delegate to :func:`pyramids.base.crs.get_epsg_from_prj` so callers
        that already hold a :class:`FeatureCollection` do not need to import the
        helper module.

        Args:
            prj (str): Projection string (WKT, ESRI WKT, or Proj4).

        Returns:
            int: The resolved EPSG code.

        Raises:
            CRSError: If `prj` is an empty string.

        Examples:
            - Identify EPSG:4326 from its WKT representation:
                ```python
                >>> from osgeo import osr
                >>> from pyramids.feature import FeatureCollection
                >>> ref = osr.SpatialReference()
                >>> _ = ref.ImportFromEPSG(4326)
                >>> FeatureCollection.get_epsg_from_prj(ref.ExportToWkt())
                4326

                ```
            - Empty string raises `CRSError` instead of defaulting:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> FeatureCollection.get_epsg_from_prj("")
                Traceback (most recent call last):
                    ...
                pyramids.base._errors.CRSError:...empty projection string...

                ```
        """
        return _crs.get_epsg_from_prj(prj)

    @staticmethod
    def reproject_coordinates(
        x: list[float],
        y: list[float],
        *,
        from_crs: Any = 4326,
        to_crs: Any = 3857,
        precision: int | None = 6,
    ) -> tuple[list[float], list[float]]:
        """Delegate to :func:`pyramids.base.crs.reproject_coordinates`.

        canonical replacement for the deleted
        `reproject_points` / `reproject_points2` pair. Argument and
        return order is `(x, y)` throughout. `from_crs` / `to_crs`
        accept any form :meth:`pyproj.Transformer.from_crs` understands
        (EPSG int, authority string, WKT, Proj4, :class:`pyproj.CRS`).

        Args:
            x (list[float]): X-coordinates in the source CRS.
            y (list[float]): Y-coordinates in the source CRS.
            from_crs: Source CRS (EPSG int, authority string, WKT, Proj4, or
                :class:`pyproj.CRS`). Default `4326`.
            to_crs: Target CRS, same forms as `from_crs`. Default `3857`.
            precision (int | None): Decimal places for each returned value, or
                `None` to disable rounding. Default `6`.

        Returns:
            tuple[list[float], list[float]]: `(x, y)` in the target CRS.

        Raises:
            ValueError: If `len(x)!= len(y)`.
            CRSError: If either CRS cannot be parsed by pyproj.

        Examples:
            - Reproject a single WGS84 point into Web Mercator:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> x, y = FeatureCollection.reproject_coordinates(
                ...     [31.0], [30.0], from_crs=4326, to_crs=3857
                ... )
                >>> round(x[0])
                3450904
                >>> round(y[0])
                3503550

                ```
            - Round-trip 4326 -> 3857 -> 4326 recovers the original to
              `precision=6` decimals:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> x1, y1 = FeatureCollection.reproject_coordinates(
                ...     [12.5], [55.7], from_crs=4326, to_crs=3857
                ... )
                >>> x2, y2 = FeatureCollection.reproject_coordinates(
                ...     x1, y1, from_crs=3857, to_crs=4326
                ... )
                >>> round(x2[0], 4), round(y2[0], 4)
                (12.5, 55.7)

                ```
            - Mismatched list lengths raise `ValueError`:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> FeatureCollection.reproject_coordinates(
                ...     [1.0, 2.0], [3.0], from_crs=4326, to_crs=3857
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: x and y must have equal length...

                ```
        """
        return _crs.reproject_coordinates(
            x, y, from_crs=from_crs, to_crs=to_crs, precision=precision
        )

    @staticmethod
    def _get_xy_coords(geometry: Any, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_xy_coords`."""
        return _geom.get_xy_coords(geometry, coord_type)

    @staticmethod
    def _get_point_coords(geometry: Point, coord_type: str) -> float | int:
        """Delegate to :func:`pyramids.feature.geometry.get_point_coords`."""
        return _geom.get_point_coords(geometry, coord_type)

    @staticmethod
    def _get_line_coords(geometry: LineString, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_line_coords`."""
        return _geom.get_line_coords(geometry, coord_type)

    @staticmethod
    def _get_poly_coords(geometry: Polygon, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_poly_coords`."""
        return _geom.get_poly_coords(geometry, coord_type)

    @staticmethod
    def _explode_gdf(gdf: GeoDataFrame, geometry: str = "multipolygon") -> GeoDataFrame:
        """Delegate to :func:`pyramids.feature.geometry.explode_gdf`."""
        return _geom.explode_gdf(gdf, geometry)

    @staticmethod
    def _multi_geom_handler(
        multi_geometry: MultiPolygon | MultiPoint | MultiLineString,
        coord_type: str,
        geom_type: str,
    ) -> list:
        """Delegate to :func:`pyramids.feature.geometry.multi_geom_handler`."""
        return _geom.multi_geom_handler(multi_geometry, coord_type, geom_type)

    @staticmethod
    def _geometry_collection(geom: Any, coord_type: str) -> list[Any]:
        """Delegate to :func:`pyramids.feature.geometry.geometry_collection_coords`."""
        return _geom.geometry_collection_coords(geom, coord_type)

    @staticmethod
    def _get_coords(row: Any, geom_col: str, coord_type: str) -> Any:
        """Delegate to :func:`pyramids.feature.geometry.get_coords`."""
        return _geom.get_coords(row, geom_col, coord_type)

    @staticmethod
    def create_polygon(coords: list[tuple[float, float]]) -> Polygon:
        """Create a :class:`shapely.Polygon` from coordinates.

        Delegates to :func:`pyramids.feature.geometry.create_polygon`.
        For the WKT-string form use :meth:`polygon_wkt` instead.

        the `wkt=True` polymorphic kwarg was removed
        outright (no deprecation shim) — a polymorphic return type
        was the whole motivation for splitting in the first
        place.

        Args:
            coords (list[tuple[float, float]]): Ring coordinates. At least 3
                vertices are required.

        Returns:
            Polygon: A shapely `Polygon`.

        Raises:
            InvalidGeometryError: If `coords` has fewer than 3 vertices.

        Examples:
            - Build a unit square and inspect its bounds / area:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> poly = FeatureCollection.create_polygon(
                ...     [(0, 0), (1, 0), (1, 1), (0, 1)]
                ... )
                >>> poly.area
                1.0
                >>> poly.bounds
                (0.0, 0.0, 1.0, 1.0)

                ```
            - A polygon with fewer than 3 vertices raises
              `InvalidGeometryError`:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> FeatureCollection.create_polygon([(0, 0), (1, 1)])
                Traceback (most recent call last):
                    ...
                pyramids.base._errors.InvalidGeometryError: ...at least 3 vertices...

                ```
        """
        return _geom.create_polygon(coords)

    @staticmethod
    def polygon_wkt(coords: list[tuple[float, float]]) -> str:
        """Return the WKT for a polygon built from `coords`.

        Delegates to :func:`pyramids.feature.geometry.polygon_wkt`. This is the
        WKT-string counterpart of :meth:`create_polygon`; the two were split in
        so each entry point has a single return type.

        Args:
            coords (list[tuple[float, float]]): Ring coordinates.

        Returns:
            str: Well-Known Text representation of the polygon.

        Examples:
            - A unit-square ring produces a closed WKT polygon string:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> wkt = FeatureCollection.polygon_wkt(
                ...     [(0, 0), (1, 0), (1, 1), (0, 1)]
                ... )
                >>> wkt.startswith("POLYGON")
                True
                >>> wkt.count("(")
                2

                ```
            - A triangle WKT can be round-tripped back through shapely:
                ```python
                >>> from shapely import wkt as _wkt
                >>> from pyramids.feature import FeatureCollection
                >>> s = FeatureCollection.polygon_wkt(
                ...     [(0, 0), (2, 0), (1, 2)]
                ... )
                >>> _wkt.loads(s).area
                2.0

                ```
        """
        return _geom.polygon_wkt(coords)

    @staticmethod
    def create_points(coords: Iterable[tuple[float,...]]) -> list[Point]:
        """Return a list of shapely Points from `coords`.

        Delegates to :func:`pyramids.feature.geometry.create_points`.
        fixed the return type to always be `list[Point]`; for the
        `FeatureCollection` wrapper form use :meth:`point_collection`.

        Args:
            coords: Iterable of `(x, y)` tuples.

        Returns:
            list[Point]: The constructed shapely Points.

        Examples:
            - Build two points and read back their coordinates:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> pts = FeatureCollection.create_points([(0.0, 0.0), (1.5, 2.5)])
                >>> len(pts)
                2
                >>> pts[1].x, pts[1].y
                (1.5, 2.5)

                ```
            - Works with any iterable of coordinate pairs:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> pts = FeatureCollection.create_points(
                ...     iter([(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)])
                ... )
                >>> [(p.x, p.y) for p in pts]
                [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]

                ```
        """
        return _geom.create_points(coords)

    @staticmethod
    def point_collection(
        coords: Iterable[tuple[float, ...]], crs: Any
    ) -> FeatureCollection:
        """Return a FeatureCollection of points with the given CRS.

        Delegates to :func:`pyramids.feature.geometry.point_collection` and
        wraps the result as a `FeatureCollection`. This is the
        `FeatureCollection` counterpart of :meth:`create_points`.

        Args:
            coords: Iterable of `(x, y)` tuples.
            crs: Any CRS form accepted by :class:`geopandas.GeoDataFrame` (EPSG
                int, WKT, Proj string, or :class:`pyproj.CRS`).

        Returns:
            FeatureCollection: A new `FeatureCollection` with a single
            `geometry` column of shapely `Point` rows.

        Examples:
            - Build a 3-point WGS84 FC and inspect its CRS and geometry:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.point_collection(
                ...     [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], crs="EPSG:4326",
                ... )
                >>> len(fc)
                3
                >>> fc.crs.to_epsg()
                4326
                >>> fc.geometry.iloc[1].x
                1.0

                ```
            - Integer EPSG works too; geometry column names default to
              `"geometry"`:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.point_collection(
                ...     [(10.0, 20.0), (30.0, 40.0)], crs=3857,
                ... )
                >>> fc.geometry.name
                'geometry'
                >>> [(p.x, p.y) for p in fc.geometry]
                [(10.0, 20.0), (30.0, 40.0)]

                ```
        """
        return FeatureCollection(_geom.point_collection(coords, crs=crs))

    def with_coordinates(self) -> FeatureCollection:
        """Return a new FeatureCollection with per-vertex `x` and `y` columns.

        non-mutating replacement for the old `xy()` method
        (which has been deleted). Matches pandas / geopandas
        convention — data-transformation methods return a new object.
        The `with_` prefix follows the stdlib/pandas pattern for
        "return a copy with this change applied" (e.g.
        :meth:`pathlib.Path.with_suffix`).

        Explodes MultiPolygon and GeometryCollection geometries into
        their parts first, then attaches `x` and `y` columns
        containing the coordinate sequences of each row.

        Returns:
            FeatureCollection: A new FeatureCollection (`self` is
            not modified) with the original columns plus `x` and
            `y` per-vertex coordinate lists.

        Examples:
            - A Point FC gets scalar `x` / `y` per row:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(1.0, 2.0), Point(3.0, 4.0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.with_coordinates()
                >>> list(out["x"])
                [1.0, 3.0]
                >>> list(out["y"])
                [2.0, 4.0]

                ```
            - The input FC is not mutated:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0.0, 0.0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> _ = fc.with_coordinates()
                >>> "x" in fc.columns
                False

                ```
        """
        gdf = _geom.explode_gdf(
            gpd.GeoDataFrame(self, copy=True), geometry="multipolygon"
        )
        gdf = _geom.explode_gdf(gdf, geometry="geometrycollection")

        fc = FeatureCollection(gdf)
        fc["x"] = fc.apply(
            _geom.get_coords, geom_col="geometry", coord_type="x", axis=1
        )
        fc["y"] = fc.apply(
            _geom.get_coords, geom_col="geometry", coord_type="y", axis=1
        )
        fc.reset_index(drop=True, inplace=True)
        return fc

    def plot(
        self,
        column: str | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot features, optionally on a web-tile basemap.

        Delegates to :meth:`geopandas.GeoDataFrame.plot` and, when
        `basemap` is truthy, adds an OSM (or named provider) tile
        layer underneath.

        Raises:
            ValueError: If `basemap` is requested but the FC has no CRS.
        """
        ax = super().plot(column=column, **kwargs)

        if basemap:
            if self.epsg is None:
                raise CRSError(
                    "FeatureCollection must have a CRS (epsg) to use basemap."
                )
            source = basemap if isinstance(basemap, str) else None
            add_basemap(ax, crs=self.epsg, source=source)

        return ax

    def concat(self, other: GeoDataFrame) -> FeatureCollection:
        """Concatenate another GeoDataFrame onto this FeatureCollection.

        mirrors :func:`pandas.concat` — returns a new
        `FeatureCollection` and never mutates `self`. No
        `inplace` kwarg (pandas' `pd.concat` has never had one;
        follow the convention).

        Equivalent to `pd.concat([fc, other])` which also works
        directly and returns a `FeatureCollection` via the
        `_constructor` hook.

        a CRS mismatch between `self` and `other` raises
        :class:`pyramids.base._errors.CRSError`. The old behaviour
        silently adopted `self`'s CRS — which corrupted the
        `other` rows' coordinates if the two frames were in
        different CRSes. Callers that want to force-concat across
        CRSes must `other.to_crs(self.crs)` first. An
        unset-on-one-side case (one CRS is `None`) is permitted so
        you can seed a CRS by concatenating a CRS-carrying frame
        onto a freshly-constructed empty FC.

        Args:
            other (GeoDataFrame): The rows to append.

        Returns:
            FeatureCollection: A new FC containing `self`'s rows
            followed by `other`'s rows, with `self`'s CRS and a
            freshly-reset index.

        Raises:
            CRSError: If both frames carry a CRS and the two CRSes
                do not match.

        Examples:
            - Concatenate two single-row FCs on matching CRS:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> a = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> b = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [2]}, geometry=[Point(1, 1)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = a.concat(b)
                >>> len(out)
                2
                >>> list(out["id"])
                [1, 2]
                >>> out.crs.to_epsg()
                4326

                ```
            - CRS mismatch raises `CRSError`:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> a = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1]}, geometry=[Point(0, 0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> b = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [2]}, geometry=[Point(1, 1)],
                ...         crs="EPSG:3857",
                ...     )
                ... )
                >>> a.concat(b)
                Traceback (most recent call last):
                    ...
                pyramids.base._errors.CRSError: concat: CRS mismatch...

                ```
        """
        # validate CRS agreement up front.
        if self.crs is not None and other.crs is not None:
            if self.crs != other.crs:
                raise CRSError(
                    f"concat: CRS mismatch — self.crs = {self.crs!r}, "
                    f"other.crs = {other.crs!r}. Reproject one side "
                    f"— `other.to_crs(self.crs)` OR "
                    f"`self.to_crs(other.crs)` — before "
                    f"concatenating, or strip one CRS with "
                    f".set_crs(None, allow_override=True)."
                )
        combined = gpd.GeoDataFrame(pd.concat([self, other]))
        combined.index = list(range(len(combined)))
        combined.crs = self.crs if self.crs is not None else other.crs
        return FeatureCollection(combined)

    def with_centroid(self) -> FeatureCollection:
        """Return a new FC with per-feature center-point columns attached.

        non-mutating replacement for the old `center_point()`
        method (which has been deleted). The `with_` prefix mirrors
        stdlib / pandas conventions for "return a copy with this
        change applied".

        Computes average x/y per feature (after
        :meth:`with_coordinates`) and attaches three columns:
        `avg_x`, `avg_y` and `center_point` (shapely `Point`).

        feeding a degenerate or empty geometry (for example an
        empty `Point`, or a `Polygon` whose ring has zero area)
        produces `(NaN, NaN)` averages. The method emits a single
        `UserWarning` listing the row indices whose `avg_x` /
        `avg_y` could not be computed so downstream code can guard
        against the NaN centroids instead of silently consuming them.
        The `center_point` value at those rows is an empty
        `shapely.Point` (`Point.is_empty is True`) rather than a
        `(NaN, NaN)` point.

        Returns:
            FeatureCollection: A new FeatureCollection (`self` is
            not modified) with `x`, `y`, `avg_x`, `avg_y`,
            `center_point` columns added.

        Examples:
            - Compute centroids for a 2-polygon FC:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Polygon
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[
                ...             Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                ...             Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]),
                ...         ],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.with_centroid()
                >>> [(p.x, p.y) for p in out["center_point"]]
                [(0.8, 0.8), (4.8, 4.8)]

                ```
            - A Point FC is a no-op for the coordinate lists (each row
              is already a single vertex); the centroid equals the point:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2]},
                ...         geometry=[Point(3.0, 4.0), Point(7.0, 8.0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.with_centroid()
                >>> [(p.x, p.y) for p in out["center_point"]]
                [(3.0, 4.0), (7.0, 8.0)]

                ```
        """
        fc = self.with_coordinates()
        for i, row_i in fc.iterrows():
            fc.loc[i, "avg_x"] = np.mean(row_i["x"])
            fc.loc[i, "avg_y"] = np.mean(row_i["y"])

        # detect rows whose averaged coordinate could not be
        # computed (empty geometry, all-NaN rings, etc.). Emit a single
        # summary warning and substitute an empty Point so the column
        # does not expose a `(NaN, NaN)` Point that would then crash
        # downstream reprojections.
        avg_x = fc["avg_x"].to_numpy()
        avg_y = fc["avg_y"].to_numpy()
        bad_mask = np.isnan(avg_x) | np.isnan(avg_y)
        if bad_mask.any():
            bad_idx = [int(i) for i, is_bad in enumerate(bad_mask) if is_bad]
            warnings.warn(
                f"with_centroid: {len(bad_idx)} row(s) yielded NaN centroids "
                f"(rows {bad_idx}). Their `center_point` is an empty "
                f"shapely.Point. Drop or repair those rows before running "
                f"a method that requires a valid centroid (e.g. reproject, "
                f"distance).",
                GeometryWarning,
                stacklevel=2,
            )

        # single-pass build. The previous implementation built a
        # throwaway `coords_list` (with NaN placeholders for the bad
        # rows), called `create_points` on it, then iterated the
        # result a second time to substitute empty Points for the bad
        # rows. Skip both intermediates — write the final column value
        # directly.
        cleaned: list[Any] = [
            Point() if bad else Point(ax, ay)
            for ax, ay, bad in zip(avg_x.tolist(), avg_y.tolist(), bad_mask.tolist())
        ]
        fc["center_point"] = cleaned
        return fc
