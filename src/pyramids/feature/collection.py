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
from typing import TYPE_CHECKING, Any, Callable, Iterable
from urllib.parse import urlencode

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature._lazy_collection import LazyFeatureCollection

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr
from shapely.geometry import Point, Polygon, box

from pyramids import _io as _pyramids_io
from pyramids.base._errors import (
    CRSError,
    FeatureError,
    GeometryWarning,
    InvalidGeometryError,
)
from pyramids.base._utils import Catalog, import_pyarrow, require_cleopatra
from pyramids.base.remote import is_remote
from pyramids.basemap.basemap import add_basemap
from pyramids.feature import _h3
from pyramids.feature import geometry as _geom
from pyramids.feature import tessellation as _tess
from pyramids.feature._oapif import from_ogc_api_features as _from_ogc_api_features
from pyramids.feature._wfs import from_wfs as _from_wfs

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
        "dependency. Install with one of:\n"
        "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
        "  - conda-forge: conda install -c conda-forge pyramids-parquet"
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
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float] | list[float],
        *,
        epsg: Any,
    ) -> FeatureCollection:
        """Build a one-row FeatureCollection from a geographic bounding box.

        The bbox is the canonical ``(west, south, east, north)`` quadruple in
        the CRS named by ``epsg``. The result is a single-row FC whose only
        geometry is a rectangular Polygon — handy for cropping a raster or
        windowed-reading it without writing out the polygon vertices by hand:

        .. code-block:: python

            mask = FeatureCollection.from_bbox((31.0, 30.0, 31.1, 30.1), epsg=4326)
            cropped = dataset.crop(mask)

        Most callers do not need to build this themselves — :meth:`Dataset.crop`
        and :meth:`Dataset.read_array` (via :meth:`pyramids.dataset.engines.io.IO.read_array`)
        accept the bbox/``epsg`` pair directly and call this helper internally.

        Args:
            bbox: A 4-element ``(west, south, east, north)`` tuple / list of
                numbers. Must satisfy ``west < east`` and ``south < north``.
            epsg: CRS for the bbox coordinates — anything ``geopandas`` accepts
                for ``crs=`` (EPSG int such as ``4326``, ``"EPSG:4326"`` string,
                WKT, Proj, or a :class:`pyproj.CRS`). Required (a bbox without
                a CRS is ambiguous).

        Returns:
            FeatureCollection: A one-row FC carrying the rectangular polygon,
            in the supplied CRS.

        Raises:
            ValueError: ``bbox`` is not a 4-element sequence, or violates
                ``west < east`` / ``south < north``, or ``epsg`` is ``None``.
            TypeError: ``bbox`` elements are not numbers.

        Examples:
            - Build a one-row FC from a bbox and inspect it:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.from_bbox((31.0, 30.0, 31.1, 30.1), epsg=4326)
                >>> len(fc)
                1
                >>> tuple(float(v) for v in fc.total_bounds)
                (31.0, 30.0, 31.1, 30.1)
                >>> fc.crs.to_epsg()
                4326

                ```
            - Use it as a mask to crop a raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.feature import FeatureCollection
                >>> arr = np.arange(100, dtype="int16").reshape(10, 10)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
                ... )
                >>> fc = FeatureCollection.from_bbox((0.1, -0.2, 0.2, -0.1), epsg=4326)
                >>> ds.crop(mask=fc).shape
                (1, 2, 2)

                ```
            - ``epsg=None`` is rejected — a bbox without a CRS is ambiguous:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> try:
                ...     FeatureCollection.from_bbox((0, 0, 1, 1), epsg=None)
                ... except ValueError as exc:
                ...     print("epsg" in str(exc))
                True

                ```

        See Also:
            - :meth:`pyramids.dataset.engines.spatial.Spatial.crop`: accepts
              ``bbox=`` / ``epsg=`` directly and routes through this helper.
            - :meth:`pyramids.dataset.engines.io.IO.read_array`: same.
        """
        if epsg is None:
            raise ValueError(
                "from_bbox requires an explicit epsg= for the bbox CRS; "
                "a bbox without a CRS is ambiguous"
            )
        try:
            seq = list(bbox)
        except TypeError as exc:
            raise ValueError(
                f"bbox must be a 4-element (west, south, east, north) sequence; "
                f"got {bbox!r}"
            ) from exc
        if len(seq) != 4:
            raise ValueError(
                f"bbox must have exactly 4 elements (west, south, east, north); "
                f"got {len(seq)}: {seq!r}"
            )
        try:
            w, s, e, n = (float(v) for v in seq)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bbox elements must be numbers; got {seq!r}") from exc
        if not (w < e):
            raise ValueError(f"bbox must satisfy west < east; got west={w}, east={e}")
        if not (s < n):
            raise ValueError(
                f"bbox must satisfy south < north; got south={s}, north={n}"
            )
        return cls(geometry=[box(w, s, e, n)], crs=epsg)

    @classmethod
    def fishnet(
        cls,
        bounds: tuple[float, float, float, float] | list[float],
        cell_size: float,
        *,
        crs: Any | None = None,
    ) -> FeatureCollection:
        """Build a vector grid of square cell polygons over an arbitrary extent.

        The vector / arbitrary-bbox analogue of :meth:`pyramids.dataset.Dataset.get_cell_polygons` (which is
        raster-aligned). Cells are full ``cell_size`` squares laid row-major from the lower-left corner of
        ``bounds``; the grid has ``ceil(width / cell_size)`` columns and ``ceil(height / cell_size)`` rows, and
        carries integer ``row`` / ``col`` index columns.

        Args:
            bounds: ``(minx, miny, maxx, maxy)`` extent the grid covers, in the units of ``crs``.
            cell_size: Side length of each square cell, in the same units. Must be positive.
            crs: CRS for the grid — anything ``geopandas`` accepts for ``crs=`` — or ``None`` for a CRS-less grid.

        Returns:
            FeatureCollection: One square polygon per cell, with ``row`` and ``col`` columns, in ``crs``.

        Raises:
            ValueError: If ``cell_size`` is not positive, or ``bounds`` is degenerate (``minx >= maxx`` or
                ``miny >= maxy``).

        Examples:
            - A 2x2 grid over a one-degree square:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> grid = FeatureCollection.fishnet((0.0, 0.0, 1.0, 1.0), 0.5, crs="EPSG:4326")
                >>> len(grid)
                4
                >>> sorted(grid.columns)
                ['col', 'geometry', 'row']
                >>> grid.crs.to_epsg()
                4326

                ```

        See Also:
            - :meth:`pyramids.dataset.Dataset.get_cell_polygons`: the raster-aligned grid-cell equivalent.
        """
        polygons, rows, cols = _tess.fishnet_cells(bounds, cell_size)
        return cls(gpd.GeoDataFrame({"row": rows, "col": cols}, geometry=polygons, crs=crs))

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
                    "'dask-geopandas' dependency. Install with one of:\n"
                    "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
                    "  - conda-forge: conda install -c conda-forge pyramids-parquet"
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
    def read_gpx_layers(cls, path: str | Path) -> dict[str, FeatureCollection]:
        """Read every non-empty sub-layer of a GPX file into a dict of FeatureCollections.

        A GPX file exposes up to five sub-layers — ``waypoints``, ``routes``, ``tracks``, ``route_points``,
        ``track_points``. GDAL always advertises all five even when a file has none of a given kind; this reads
        each and returns only the ones that actually contain features, keyed by layer name.

        Args:
            path: Path to a ``.gpx`` file.

        Returns:
            dict[str, FeatureCollection]: One entry per **non-empty** sub-layer, keyed by its GPX layer name.

        Examples:
            - A GPX with a waypoint and a track yields those sub-layers (empty ``routes`` is omitted):
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> from pyramids.feature import FeatureCollection
                >>> gpx = (
                ...     '<?xml version="1.0"?>\\n'
                ...     '<gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">'
                ...     '<wpt lat="1.0" lon="2.0"><name>wp1</name></wpt>'
                ...     '<trk><name>t1</name><trkseg>'
                ...     '<trkpt lat="1.0" lon="2.0"/><trkpt lat="1.1" lon="2.1"/>'
                ...     '</trkseg></trk></gpx>'
                ... )
                >>> p = Path(tempfile.mkdtemp()) / "t.gpx"
                >>> _ = p.write_text(gpx)
                >>> layers = FeatureCollection.read_gpx_layers(p)
                >>> sorted(layers)
                ['track_points', 'tracks', 'waypoints']
                >>> len(layers["waypoints"])
                1

                ```
        """
        result: dict[str, FeatureCollection] = {}
        for name in cls.list_layers(path):
            fc = cls.read_file(path, layer=name)
            if len(fc) > 0:
                result[name] = fc
        return result

    @classmethod
    def _read_featureserver_page(cls, page_url: str) -> FeatureCollection:
        """Read one ESRIJSON page from an ArcGIS FeatureServer query URL.

        Isolated so :meth:`from_featureserver`'s pagination can be unit-tested without a live endpoint.

        Args:
            page_url: A fully-formed ``.../query?...&f=json&resultOffset=...`` URL.

        Returns:
            FeatureCollection: The features in this page (possibly empty).
        """
        return cls.read_file(page_url)

    @classmethod
    def from_featureserver(
        cls,
        url: str,
        *,
        where: str = "1=1",
        out_fields: str = "*",
        max_records: int | None = None,
        page_size: int = 1000,
        max_pages: int = 1000,
    ) -> FeatureCollection:
        """Read an ArcGIS **FeatureServer** layer into a FeatureCollection, following pagination.

        FeatureServer endpoints cap the number of records returned per request (``maxRecordCount``), so reading
        a large layer requires paging through it. This issues ``.../query`` requests with increasing
        ``resultOffset`` until the server stops returning new features (or ``max_records`` is reached) and
        concatenates the pages. Each page is read with GDAL's ESRIJSON driver (generic ArcGIS REST — no
        provider-specific auth).

        ``max_pages`` is a safety cap: a server that does not honour ``resultOffset`` (no pagination support)
        would otherwise return the same first page forever; on hitting the cap a ``UserWarning`` is emitted and
        paging stops.

        Args:
            url: A FeatureServer layer URL (with or without a trailing ``/query``).
            where: SQL ``where`` filter. Defaults to ``"1=1"`` (all features).
            out_fields: Comma-separated attribute fields to fetch, or ``"*"`` for all.
            max_records: Cap on the total number of features read, or ``None`` for all.
            page_size: Records requested per page (``resultRecordCount``). The server may return fewer.
            max_pages: Hard cap on the number of page requests, guarding against a server that ignores
                ``resultOffset``. Defaults to 1000.

        Returns:
            FeatureCollection: All features across the paged responses (empty if the layer has none).

        Examples:
            - Read a public FeatureServer layer (network call — skipped in doctests):
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.from_featureserver(  # doctest: +SKIP
                ...     "https://services.arcgis.com/.../FeatureServer/0", where="STATE='CA'"
                ... )

                ```
        """
        if page_size < 1:
            raise ValueError(f"from_featureserver: page_size must be >= 1, got {page_size}")
        if max_records is not None and max_records < 0:
            raise ValueError(f"from_featureserver: max_records must be >= 0 or None, got {max_records}")
        base = url.split("?", 1)[0].rstrip("/")
        if not base.lower().endswith("/query"):
            base = f"{base}/query"
        pages, first_crs = cls._collect_featureserver_pages(
            base, where, out_fields, max_records, page_size, max_pages
        )
        # Concatenate in one pass (pd.concat preserves the shared CRS) — repeatedly calling .concat()
        # re-sets the CRS and trips a geopandas DeprecationWarning.
        if pages:
            combined = FeatureCollection(pd.concat(pages, ignore_index=True))
        else:
            combined = cls(gpd.GeoDataFrame(geometry=[], crs=first_crs))
        return combined

    @classmethod
    def from_wfs(
        cls,
        endpoint: str,
        *,
        typename: str,
        bbox: tuple[float, float, float, float] | None = None,
        output_crs: str | None = None,
        where: str | None = None,
        max_features: int | None = None,
        version: str | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> FeatureCollection:
        """Read a feature type from an OGC **Web Feature Service** (WFS).

        Fetches a subset of a feature type from a WFS server and returns it as a
        :class:`FeatureCollection`. The transport is GDAL's native OGR ``WFS:``
        driver, so the WFS ``1.x`` vs ``2.0.0`` dialect fork — ``typeName`` versus
        ``typeNames`` — is handled inside GDAL; the caller always supplies a
        single lon/lat ``bbox`` and an optional attribute filter. This is the
        vector sibling of :meth:`pyramids.dataset.Dataset.from_wcs`.

        The ``typename`` is validated against a (cached) ``GetCapabilities`` so an
        unadvertised feature type fails fast with a clear :class:`ValueError`
        rather than an opaque driver error.

        Args:
            endpoint: The WFS service URL (e.g. ``"https://geoserver.example/ows"``).
                Catalog / type-name routing belongs in the calling layer, not here.
            typename: The feature-type identifier as advertised by
                ``GetCapabilities`` (e.g. ``"topp:states"``). A value the server
                does not advertise raises :class:`ValueError`.
            bbox: Optional ``(minx, miny, maxx, maxy)`` spatial filter, interpreted
                in the feature type's **native CRS** (which WFS layers advertise;
                usually ``EPSG:4326``, lon/lat). Only intersecting features are
                returned. ``None`` (default) fetches all features.
            output_crs: Optional CRS to reproject the result into (any form
                :meth:`to_crs` accepts). ``None`` (default) keeps the server's CRS.
            where: Optional OGR/SQL attribute filter (e.g. ``"PERSONS > 1000000"``)
                pushed down to the server / driver.
            max_features: Optional cap on the number of features returned. ``None``
                (default) returns all.
            version: Force a WFS protocol version (``"1.0.0"``, ``"1.1.0"``,
                ``"2.0.0"``). ``None`` (default) lets GDAL negotiate from the
                server's capabilities.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds for the metadata / feature requests
                (whole seconds; a value below 1 is clamped to 1). Defaults to
                ``60.0``.

        Returns:
            FeatureCollection: The fetched features (empty if the filter matches
            none).

        Raises:
            ValueError: ``typename`` or ``version`` is not advertised, ``bbox`` is
                malformed, or ``max_features`` is negative.
            pyramids.errors.WFSError: The server could not be reached or returned
                an error / a non-feature (``<ows:ExceptionReport>``) body, or
                ``output_crs`` was requested but the result carries no CRS.

        Examples:
            Read a bbox subset of a public feature type (network call — skipped in
            doctests):

            ```python
            >>> from pyramids.feature import FeatureCollection
            >>> fc = FeatureCollection.from_wfs(  # doctest: +SKIP
            ...     "https://geoserver.example/ows",
            ...     typename="topp:states",
            ...     bbox=(-104, 35, -94, 41),
            ...     where="PERSONS > 1000000",
            ... )

            ```

        See Also:
            - :meth:`read_file`: read a vector file or URL.
            - :meth:`from_featureserver`: read an Esri ArcGIS FeatureServer layer.
            - :meth:`pyramids.dataset.Dataset.from_wcs`: the raster (WCS) sibling.
        """
        return _from_wfs(
            cls,
            endpoint,
            typename=typename,
            bbox=bbox,
            output_crs=output_crs,
            where=where,
            max_features=max_features,
            version=version,
            auth=auth,
            timeout=timeout,
        )

    @classmethod
    def from_ogc_api_features(
        cls,
        endpoint: str,
        *,
        collection: str,
        bbox: tuple[float, float, float, float] | None = None,
        output_crs: str | None = None,
        where: str | None = None,
        max_features: int | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float = 60.0,
    ) -> FeatureCollection:
        """Read a collection from an **OGC API – Features** service.

        Fetches a subset of a collection from an OGC API – Features service and
        returns it as a :class:`FeatureCollection`. OGC API – Features is the
        modern REST/JSON successor to WFS: a landing page links to ``/collections``
        and each collection exposes ``/collections/{id}/items`` as GeoJSON, paged
        through ``rel="next"`` links. The transport is GDAL's native OGR ``OAPIF``
        driver, so conformance negotiation and paging happen inside GDAL; the
        caller supplies a single lon/lat ``bbox`` and an optional attribute filter.
        This is the OGC-API-era sibling of :meth:`from_wfs`.

        The ``collection`` is validated against a (cached) ``/collections``
        document so an unadvertised collection fails fast with a clear
        :class:`ValueError` rather than an opaque driver error.

        Args:
            endpoint: The OGC API landing-page / base URL (e.g.
                ``"https://demo.pygeoapi.io/master"``). Catalog routing belongs in
                the calling layer, not here.
            collection: The collection identifier as advertised by ``/collections``
                (e.g. ``"lakes"``). A value the service does not advertise raises
                :class:`ValueError`.
            bbox: Optional ``(minx, miny, maxx, maxy)`` spatial filter in **lon/lat
                (CRS84)** — the OGC API – Features default for the ``bbox`` query,
                regardless of the collection's storage CRS. Only intersecting
                features are returned. ``None`` (default) fetches all features.
            output_crs: Optional CRS to reproject the result into (any form
                :meth:`to_crs` accepts). ``None`` (default) keeps the service's CRS.
            where: Optional OGR/SQL attribute filter (e.g. ``"scalerank <= 2"``)
                pushed down to the driver.
            max_features: Optional cap on the number of features returned. ``None``
                (default) returns all, across as many pages as the service serves.
            auth: Optional ``(username, password)`` for Basic-authed services.
            timeout: HTTP timeout in seconds for the metadata / items requests
                (whole seconds; a value below 1 is clamped to 1). Defaults to
                ``60.0``.

        Returns:
            FeatureCollection: The fetched features (empty if the filter matches
            none).

        Raises:
            ValueError: ``collection`` is not advertised, ``bbox`` is malformed, or
                ``max_features`` is negative.
            pyramids.errors.OGCAPIError: The service could not be reached or
                returned an error / a non-feature body, or ``output_crs`` was
                requested but the result carries no CRS.

        Examples:
            Read a bbox subset of a public collection (network call — skipped in
            doctests):

            ```python
            >>> from pyramids.feature import FeatureCollection
            >>> fc = FeatureCollection.from_ogc_api_features(  # doctest: +SKIP
            ...     "https://demo.pygeoapi.io/master",
            ...     collection="lakes",
            ...     bbox=(-104, 35, -94, 41),
            ...     where="scalerank <= 2",
            ... )

            ```

        See Also:
            - :meth:`from_wfs`: the classic WFS sibling.
            - :meth:`from_featureserver`: read an Esri ArcGIS FeatureServer layer.
            - :meth:`read_file`: read a vector file or URL.
        """
        return _from_ogc_api_features(
            cls,
            endpoint,
            collection=collection,
            bbox=bbox,
            output_crs=output_crs,
            where=where,
            max_features=max_features,
            auth=auth,
            timeout=timeout,
        )

    @classmethod
    def _collect_featureserver_pages(
        cls,
        base: str,
        where: str,
        out_fields: str,
        max_records: int | None,
        page_size: int,
        max_pages: int,
    ) -> tuple[list[FeatureCollection], Any]:
        """Page through a FeatureServer ``/query`` endpoint, returning the pages and the first page's CRS.

        Extracted from :meth:`from_featureserver` so each stays within the cognitive-complexity budget. Stops on
        a short / empty page, when ``max_records`` is reached, or when ``max_pages`` is hit (a server that
        ignores ``resultOffset`` would otherwise loop forever).

        Args:
            base: The ``.../query`` endpoint URL.
            where: SQL ``where`` filter.
            out_fields: Comma-separated fields, or ``"*"``.
            max_records: Total-feature cap, or ``None``.
            page_size: Records requested per page.
            max_pages: Hard cap on page requests.

        Returns:
            tuple: ``(pages, first_crs)`` — the non-empty page collections and the CRS of the first page read.
        """
        pages: list[FeatureCollection] = []
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
            page = cls._read_featureserver_page(f"{base}?{query}")
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

        Requires the optional :mod:`pyarrow` dependency. Install with one of:

        - PyPI: ``pip install 'pyramids-gis[parquet]'``
        - conda-forge: ``conda install -c conda-forge pyramids-parquet``

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
            # the extra that installs both ([parquet]).
            try:
                import dask_geopandas
            except ImportError as exc:
                raise ImportError(
                    "backend='dask' requires the optional "
                    "'dask-geopandas' dependency. Install with one of:\n"
                    "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
                    "  - conda-forge: conda install -c conda-forge pyramids-parquet"
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
            # to read). `[parquet]` pulls both.
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

        Requires the optional :mod:`pyarrow` dependency. Install with one of:

        - PyPI: ``pip install 'pyramids-gis[parquet]'``
        - conda-forge: ``conda install -c conda-forge pyramids-parquet``

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

    def _to_vector_tiles(
        self,
        path: str | Path,
        driver: str,
        *,
        min_zoom: int,
        max_zoom: int | None,
        layer_name: str | None,
        **creation_options: Any,
    ) -> Path:
        """Write this collection as a tiled-vector pyramid via ``to_file``, returning the output path.

        Shared backend for :meth:`to_pmtiles` and :meth:`to_mvt` — the two differ only by GDAL driver.

        Args:
            path: Destination (a ``.pmtiles`` file for PMTiles, a tile-root directory for MVT).
            driver: GDAL driver name (``"PMTiles"`` or ``"MVT"``).
            min_zoom: Minimum tile zoom level (``MINZOOM``).
            max_zoom: Maximum tile zoom level (``MAXZOOM``); ``None`` lets the driver choose.
            layer_name: Tile layer name, or ``None`` for the driver default.
            **creation_options: Extra driver creation options forwarded verbatim.

        Returns:
            Path: The written ``path``.
        """
        options = dict(creation_options)
        options["MINZOOM"] = min_zoom
        if max_zoom is not None:
            options["MAXZOOM"] = max_zoom
        self.to_file(path, driver=driver, layer=layer_name, **options)
        return Path(path)

    def to_pmtiles(
        self,
        path: str | Path,
        *,
        min_zoom: int = 0,
        max_zoom: int | None = None,
        layer_name: str | None = None,
        **creation_options: Any,
    ) -> Path:
        """Write this FeatureCollection to a single-file **PMTiles** vector-tile pyramid.

        Thin wrapper over GDAL's PMTiles driver (via :meth:`to_file`) for serving large vector layers to web
        map engines. The output is a single ``.pmtiles`` archive that reopens with :meth:`read_file`.

        Args:
            path: Destination ``.pmtiles`` file path.
            min_zoom: Minimum tile zoom level. Defaults to 0.
            max_zoom: Maximum tile zoom level; ``None`` lets the driver choose from the data.
            layer_name: Name of the tile layer, or ``None`` for the driver default.
            **creation_options: Extra PMTiles creation options forwarded to the driver.

        Returns:
            Path: The written ``.pmtiles`` path.

        Examples:
            - Write a small layer and confirm the archive exists:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2, 3]},
                ...         geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.to_pmtiles(d / "layer.pmtiles", max_zoom=5)
                >>> out.exists()
                True
                >>> out.suffix
                '.pmtiles'

                ```
        """
        return self._to_vector_tiles(
            path, "PMTiles", min_zoom=min_zoom, max_zoom=max_zoom, layer_name=layer_name, **creation_options
        )

    def to_mvt(
        self,
        path: str | Path,
        *,
        min_zoom: int = 0,
        max_zoom: int | None = None,
        layer_name: str | None = None,
        **creation_options: Any,
    ) -> Path:
        """Write this FeatureCollection to a **Mapbox Vector Tiles** (MVT) tile pyramid.

        Thin wrapper over GDAL's MVT driver (via :meth:`to_file`). The output is a tile-root directory of
        ``{z}/{x}/{y}.pbf`` tiles. See :meth:`to_pmtiles` for the single-file PMTiles equivalent.

        Args:
            path: Destination tile-root directory.
            min_zoom: Minimum tile zoom level. Defaults to 0.
            max_zoom: Maximum tile zoom level; ``None`` lets the driver choose from the data.
            layer_name: Name of the tile layer, or ``None`` for the driver default.
            **creation_options: Extra MVT creation options forwarded to the driver.

        Returns:
            Path: The written tile-root directory.

        Examples:
            - Write a small layer and confirm the tile root exists:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> d = Path(tempfile.mkdtemp())
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2, 3]},
                ...         geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.to_mvt(d / "tiles", max_zoom=5)
                >>> out.exists()
                True

                ```
        """
        return self._to_vector_tiles(
            path, "MVT", min_zoom=min_zoom, max_zoom=max_zoom, layer_name=layer_name, **creation_options
        )

    # FeatureCollection.to_dataset was moved to
    # Dataset.from_features(features,...) to break the circular import
    # that used to force a CLAUDE.md-violating inline
    # `from pyramids.dataset import Dataset` inside the method body.
    # Callers should migrate:
    # fc.to_dataset(dataset=ds, column_name="pop")
    # → Dataset.from_features(fc, template=ds, column_name="pop")
    # fc.to_dataset(cell_size=10)
    # → Dataset.from_features(fc, cell_size=10)

    def explode(self, geometry: str = "multipolygon") -> FeatureCollection:
        """Explode multi-geometry rows into per-row single geometries.

        Returns a new ``FeatureCollection`` where every row whose geometry
        type matches ``geometry`` is split so each child geometry becomes
        its own row. The current frame is not mutated.

        Args:
            geometry (str): The geometry type to explode (case-insensitive).
                Defaults to ``"multipolygon"``.

        Returns:
            FeatureCollection: A new collection with the same CRS as
            ``self`` and exploded geometries.

        Examples:
            - Explode a frame mixing one MultiPolygon with a Polygon:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Polygon, MultiPolygon
                >>> from pyramids.feature import FeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {
                ...         "name": ["a", "b"],
                ...         "geometry": [
                ...             MultiPolygon([
                ...                 Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                ...                 Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
                ...             ]),
                ...             Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
                ...         ],
                ...     },
                ...     crs="EPSG:4326",
                ... )
                >>> fc = FeatureCollection(gdf)
                >>> result = fc.explode("multipolygon")
                >>> len(result)
                3
                >>> [g.geom_type for g in result.geometry]
                ['Polygon', 'Polygon', 'Polygon']

                ```
        """
        return FeatureCollection(_geom.explode_gdf(self, geometry=geometry))

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
        engine: str = "geopandas",
        **kwargs: Any,
    ) -> Any:
        """Plot features, optionally on a web-tile basemap.

        Two rendering back-ends are available via ``engine``:

        - ``"geopandas"`` (default): delegate to
          :meth:`geopandas.GeoDataFrame.plot` and return the matplotlib
          ``Axes``. This is the long-standing behaviour and is unchanged.
        - ``"cleopatra"``: render polygons through
          :class:`~cleopatra.polygon_glyph.PolygonGlyph` or points through
          :class:`~cleopatra.scatter_glyph.ScatterGlyph` — sharing the
          colour/colorbar styling of the raster glyph path — and return the
          cleopatra glyph. Requires the ``[viz]`` extra.

        When ``basemap`` is truthy, an OSM (or named provider) tile layer is
        added underneath in either engine.

        Args:
            column: Column whose values drive the colour mapping. ``None``
                renders a single flat colour.
            basemap: ``True`` for OpenStreetMap, or a provider name string.
            engine: ``"geopandas"`` (default) or ``"cleopatra"``.
            **kwargs: Forwarded to the chosen back-end. For ``"cleopatra"``
                they are filtered to the glyph's accepted options via
                ``filter_kwargs``.

        Returns:
            The matplotlib ``Axes`` for ``engine="geopandas"``, or the
            cleopatra glyph (``PolygonGlyph``/``ScatterGlyph``) for
            ``engine="cleopatra"``.

        Raises:
            ValueError: If ``engine`` is not a supported value, or
                ``engine="cleopatra"`` is used with unsupported geometry.
            CRSError: If `basemap` is requested but the FC has no CRS.

        Examples:
            - Default geopandas engine returns a matplotlib ``Axes`` you can
              keep styling (tagged ``+SKIP`` — needs the ``[viz]`` extra):

                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> gdf = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
                >>> fc = FeatureCollection(gdf)
                >>> ax = fc.plot(column="v")  # doctest: +SKIP
                >>> _ = ax.set_title("points")  # doctest: +SKIP
                ```
            - The cleopatra engine returns the glyph, exposing the colorbar:

                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> gdf = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
                >>> fc = FeatureCollection(gdf)
                >>> glyph = fc.plot(column="v", engine="cleopatra")  # doctest: +SKIP
                >>> _ = glyph.cbar.set_label("value")  # doctest: +SKIP
                ```
        """
        if engine == "geopandas":
            result = super().plot(column=column, **kwargs)
            ax = result
        elif engine == "cleopatra":
            result, ax = self._plot_cleopatra(column=column, **kwargs)
        else:
            raise ValueError(
                f"Unsupported engine {engine!r}; " "choose 'geopandas' or 'cleopatra'."
            )

        if basemap:
            if self.epsg is None:
                raise CRSError(
                    "FeatureCollection must have a CRS (epsg) to use basemap."
                )
            source = basemap if isinstance(basemap, str) else None
            add_basemap(ax, crs=self.epsg, source=source)

        return result

    def _plot_cleopatra(self, column: str | None = None, **kwargs: Any):
        """Render via cleopatra ``PolygonGlyph``/``ScatterGlyph``.

        Picks the glyph from the geometry type (points → ``ScatterGlyph``,
        polygons → ``PolygonGlyph``), drives the colour from ``column`` when
        given, and returns ``(glyph, ax)`` so the caller can still overlay a
        basemap on ``ax``.

        Only polygon **exterior** rings are rendered: ``PolygonGlyph`` takes a
        sequence of single vertex rings and has no representation for holes, so
        interior rings are dropped and a polygon with a hole appears filled. A
        :class:`~pyramids.base._errors.GeometryWarning` is emitted when any
        interior ring is present; use ``engine="geopandas"`` to render holes
        correctly.

        Args:
            column: Column whose values colour the features, or ``None``.
            **kwargs: Style options, filtered to the glyph's accepted keys.

        Returns:
            tuple: ``(glyph, ax)`` — the cleopatra glyph and its ``Axes``.

        Raises:
            ValueError: If ``column`` is not a column of this collection, or
                the geometry is neither all single-``Point`` nor all-polygon
                (``MultiPoint`` is not supported).
        """
        require_cleopatra()

        if column is not None and column not in self.columns:
            raise ValueError(
                f"Column {column!r} not found; available columns: "
                f"{list(self.columns)}."
            )
        values = self[column].to_numpy() if column is not None else None
        geom_types = set(self.geom_type.unique())
        if geom_types <= {"Point"}:
            glyph = self._cleopatra_scatter_glyph(values, **kwargs)
        elif geom_types <= {"Polygon", "MultiPolygon"}:
            glyph = self._cleopatra_polygon_glyph(values, **kwargs)
        else:
            raise ValueError(
                "engine='cleopatra' supports single Point or "
                "Polygon/MultiPolygon geometries; got "
                f"{sorted(geom_types)} (MultiPoint is not supported)."
            )
        _fig, ax, _coll = glyph.plot()
        return glyph, ax

    def _cleopatra_scatter_glyph(self, values: Any, **kwargs: Any) -> Any:
        """Build a ``ScatterGlyph`` from this collection's point geometries.

        Args:
            values: Per-point colour values, or ``None`` for a flat colour.
            **kwargs: Style options, filtered to the glyph's accepted keys.

        Returns:
            cleopatra.scatter_glyph.ScatterGlyph: The point glyph.
        """
        require_cleopatra()
        from cleopatra.scatter_glyph import ScatterGlyph

        return ScatterGlyph(
            self.geometry.x.to_numpy(),
            self.geometry.y.to_numpy(),
            values=values,
            **ScatterGlyph.filter_kwargs(kwargs),
        )

    def _cleopatra_polygon_glyph(self, values: Any, **kwargs: Any) -> Any:
        """Build a ``PolygonGlyph`` from polygon exterior rings.

        MultiPolygons are expanded to one ring per part (the row's value is
        repeated for each part). Interior rings (holes) are dropped — see
        :meth:`_plot_cleopatra` — and a
        :class:`~pyramids.base._errors.GeometryWarning` is emitted when any
        are present.

        Args:
            values: Per-feature colour values, or ``None`` for a flat colour.
            **kwargs: Style options, filtered to the glyph's accepted keys.

        Returns:
            cleopatra.polygon_glyph.PolygonGlyph: The polygon glyph.
        """
        require_cleopatra()
        from cleopatra.polygon_glyph import PolygonGlyph

        polygons: list = []
        poly_values: list | None = [] if values is not None else None
        has_holes = False
        for idx, geom in enumerate(self.geometry):
            # A plain Polygon has no ``.geoms``; a MultiPolygon does.
            for part in getattr(geom, "geoms", [geom]):
                polygons.append(np.asarray(part.exterior.coords))
                has_holes = has_holes or bool(part.interiors)
                if poly_values is not None:
                    poly_values.append(values[idx])
        if has_holes:
            warnings.warn(
                "engine='cleopatra' renders only polygon exterior rings; "
                "interior rings (holes) are dropped and will appear "
                "filled. Use engine='geopandas' to render holes.",
                GeometryWarning,
                stacklevel=2,
            )
        return PolygonGlyph(
            polygons,
            values=np.asarray(poly_values) if poly_values is not None else None,
            **PolygonGlyph.filter_kwargs(kwargs),
        )

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

    def _require_point_geometry(self, op: str) -> None:
        """Raise :class:`InvalidGeometryError` unless every geometry is a single ``Point``.

        Args:
            op: Name of the calling operation, used in the error message.

        Raises:
            InvalidGeometryError: If the collection holds any non-``Point`` geometry (or is empty).
        """
        geom_types = sorted(set(self.geom_type.dropna().unique()))
        if geom_types != ["Point"]:
            raise InvalidGeometryError(
                f"{op}: requires all-Point geometries, got {geom_types or 'an empty collection'}"
            )

    def _require_column(self, op: str, column: str | None) -> None:
        """Raise :class:`ValueError` if ``column`` is given but absent from the collection.

        Args:
            op: Name of the calling operation, used in the error message.
            column: Column name to check, or ``None`` to skip the check.

        Raises:
            ValueError: If ``column`` is not ``None`` and not one of this collection's columns.
        """
        if column is not None and column not in self.columns:
            raise ValueError(f"{op}: column {column!r} not found; available columns are {list(self.columns)}")

    def voronoi(
        self,
        *,
        values: str | None = None,
        clip: FeatureCollection | None = None,
    ) -> FeatureCollection:
        """Voronoi (Thiessen) tessellation of a point ``FeatureCollection``.

        Returns one polygon per distinct input point, ordered so cell *i* corresponds to the *i*-th distinct
        point (``shapely.voronoi_polygons(ordered=True)``). Coincident (duplicate) points, and points that
        produce an empty cell after clipping, are skipped. With ``clip`` each cell is intersected with the
        boundary; with ``values`` the named column is copied onto each cell so the result can be rendered as a
        choropleth.

        Args:
            values: Name of a column copied onto each cell (cell *i* ← point *i*), or ``None`` to carry no
                attribute.
            clip: A boundary ``FeatureCollection`` each cell is intersected with (reprojected to this
                collection's CRS), or ``None`` to keep shapely's default bounded cells.

        Returns:
            FeatureCollection: One polygon per surviving cell, in this collection's CRS, carrying ``values``
            when given.

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``, or there are fewer than two distinct
                points with finite coordinates.
            ValueError: If ``values`` names a column that is not in the collection.

        Examples:
            - Tessellate four points and count the cells:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"v": [10, 20, 30, 40]},
                ...         geometry=[Point(0, 0), Point(2, 0), Point(0, 2), Point(2, 2)],
                ...         crs="EPSG:32618",
                ...     )
                ... )
                >>> cells = fc.voronoi(values="v")
                >>> len(cells)
                4
                >>> sorted(cells["v"].tolist())
                [10, 20, 30, 40]

                ```
        """
        self._require_point_geometry("voronoi")
        self._require_column("voronoi", values)
        xs, ys, keep = _tess.point_xy(self.geometry)
        ux, uy, unique = _tess.dedupe_xy(xs, ys)
        if ux.size < 2:
            raise InvalidGeometryError(
                f"voronoi: need at least 2 distinct points with finite coordinates to tessellate, got {ux.size}"
            )
        carried = self[values].to_numpy()[keep][unique] if values is not None else None
        boundary = _tess.resolve_clip(clip, self.crs)
        geometries: list = []
        attributes: list = []
        for i, cell in enumerate(_tess.voronoi_cells(ux, uy)):
            bounded = cell if boundary is None else cell.intersection(boundary)
            for part in _tess.polygon_parts(bounded):
                geometries.append(part)
                if carried is not None:
                    attributes.append(carried[i])
        data = {values: attributes} if values is not None else {}
        result = FeatureCollection(gpd.GeoDataFrame(data, geometry=geometries, crs=self.crs))
        return result

    def quadtree(
        self,
        *,
        column: str | None = None,
        agg: str | Callable = "mean",
        nmax: int = 100,
        nmin: int = 0,
        clip: FeatureCollection | None = None,
    ) -> FeatureCollection:
        """Adaptive quad-tree binning of a point ``FeatureCollection`` into rectangular cells.

        Recursively splits the points' bounding box into quadrants until each cell holds ``<= nmax`` points,
        then attaches a per-cell aggregate of ``column`` (or the point count when ``column`` is ``None``) to
        each cell polygon.

        Args:
            column: Numeric column aggregated per cell, or ``None`` to attach the point count (density).
            agg: Per-cell reducer — one of ``"mean"`` / ``"sum"`` / ``"median"`` / ``"min"`` / ``"max"`` /
                ``"std"`` / ``"count"`` or a callable taking a 1-D array. Ignored when ``column`` is ``None``.
                NaN values in ``column`` propagate to a NaN cell value when every point in a cell is NaN.
            nmax: Maximum points in a cell before it is split (smaller → finer grid).
            nmin: Cells with fewer than this many points are dropped.
            clip: A boundary ``FeatureCollection`` each cell is intersected with (reprojected to this
                collection's CRS), or ``None`` to keep the full rectangular cells.

        Returns:
            FeatureCollection: One polygon per kept cell, in this collection's CRS, with the aggregate in a
            column named ``column`` (or ``"count"`` when ``column`` is ``None``).

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``, or there is no point with finite
                coordinates.
            ValueError: If ``column`` names a column that is not in the collection, if ``nmax`` is less than 1,
                or if ``agg`` is neither a known reducer name nor a callable.

        Examples:
            - Bin four points to one point per cell and read the counts:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"v": [10, 20, 30, 40]},
                ...         geometry=[Point(0, 0), Point(2, 0), Point(0, 2), Point(2, 2)],
                ...         crs="EPSG:32618",
                ...     )
                ... )
                >>> cells = fc.quadtree(nmax=1)
                >>> int(cells["count"].sum())
                4

                ```
        """
        self._require_point_geometry("quadtree")
        self._require_column("quadtree", column)
        if nmax < 1:
            raise ValueError(f"quadtree: nmax must be >= 1, got {nmax}")
        xs, ys, keep = _tess.point_xy(self.geometry)
        if xs.size < 1:
            raise InvalidGeometryError("quadtree: need at least 1 point with finite coordinates, got 0")
        reducer = len if column is None else _tess.resolve_reducer(agg)
        column_values = None if column is None else self[column].to_numpy(dtype=float)[keep]

        def agg_fn(idx: np.ndarray) -> float:
            return float(len(idx)) if column_values is None else float(reducer(column_values[idx]))

        boundary = _tess.resolve_clip(clip, self.crs)
        cells = _tess.quadtree_cells(xs, ys, agg_fn, nmax, nmin)
        geometries: list = []
        values_out: list = []
        for xmin, ymin, xmax, ymax, value in cells:
            rectangle = box(xmin, ymin, xmax, ymax)
            bounded = rectangle if boundary is None else rectangle.intersection(boundary)
            for part in _tess.polygon_parts(bounded):
                geometries.append(part)
                values_out.append(value)
        name = column if column is not None else "count"
        result = FeatureCollection(gpd.GeoDataFrame({name: values_out}, geometry=geometries, crs=self.crs))
        return result

    def interpolate_to_raster(
        self,
        column: str,
        *,
        method: str = "idw",
        cell_size: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        power: float = 2.0,
        n_neighbors: int | None = None,
        nodata: float = -9999.0,
    ) -> "Dataset":
        """Interpolate a point column onto a continuous raster surface (point → grid).

        Reads ``column`` as the z-value at each point geometry and grids it with ``gdal.Grid`` via
        :meth:`pyramids.dataset.Dataset.from_points`. This is distinct from the inherited geopandas
        ``GeoSeries.interpolate`` (which is 1-D interpolation *along* a line). Only inverse-distance weighting
        (``method="idw"``) is available here; kriging needs the optional ``pykrige`` dependency.

        IDW extrapolates across the whole output extent (no convex-hull mask), so ``nodata`` only appears in
        cells ``gdal.Grid`` cannot estimate. Coincident (duplicate) points are not pre-averaged — they are
        handled by the inverse-distance weighting itself.

        Args:
            column: Numeric attribute column interpolated as the z-value at each point.
            method: Interpolation method. Only ``"idw"`` (inverse-distance weighting) is supported.
            cell_size: Output pixel size in the layer's CRS units. Defaults to a grid spanning the layer extent
                (see :meth:`Dataset.from_points`).
            bounds: ``(minx, miny, maxx, maxy)`` output extent; defaults to the points' total bounds.
            power: IDW distance exponent (higher → more local).
            n_neighbors: If given, limit each estimate to the nearest ``n_neighbors`` points (``invdistnn``);
                otherwise use all points (``invdist``).
            nodata: Value written to cells GDAL cannot interpolate.

        Returns:
            Dataset: A single-band raster of the interpolated surface, in the layer's CRS.

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``.
            ValueError: If ``method`` is not ``"idw"``, ``column`` is missing / non-numeric / all-NaN, or there
                are fewer than 3 points.

        Examples:
            - Inverse-distance interpolate four corner readings onto a 1-degree grid:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"rain": [1.0, 2.0, 3.0, 4.0]},
                ...         geometry=[Point(0, 0), Point(3, 0), Point(0, 3), Point(3, 3)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> surface = fc.interpolate_to_raster("rain", cell_size=1.0)
                >>> surface.band_count
                1
                >>> surface.epsg
                4326

                ```

        See Also:
            - :meth:`pyramids.dataset.Dataset.from_points`: the underlying ``gdal.Grid`` interpolation this
              method delegates to (accepts any ``gdal.Grid`` algorithm string).
        """
        self._require_point_geometry("interpolate_to_raster")
        self._require_column("interpolate_to_raster", column)
        if method != "idw":
            raise ValueError(
                f"interpolate_to_raster: method {method!r} is not supported; only 'idw' is available "
                "(kriging needs the optional 'pykrige' dependency)"
            )
        if len(self) < 3:
            raise ValueError(f"interpolate_to_raster: need at least 3 points, got {len(self)}")
        try:
            values = self[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"interpolate_to_raster: column {column!r} must be numeric") from exc
        if np.isnan(values).all():
            raise ValueError(f"interpolate_to_raster: column {column!r} is all-NaN")
        if n_neighbors is not None:
            algorithm = f"invdistnn:power={power}:max_points={n_neighbors}:nodata={nodata}"
        else:
            algorithm = f"invdist:power={power}:smoothing=0.0:nodata={nodata}"
        # local import: pyramids.dataset imports pyramids.feature, so import here to break the cycle.
        from pyramids.dataset import Dataset

        return Dataset.from_points(self, column, algorithm=algorithm, cell_size=cell_size, bbox=bounds)

    def _h3_cells(self, resolution: int, op: str) -> list[str]:
        """Return the H3 cell index of each point at ``resolution`` (helper for to_h3 / h3_bin).

        Args:
            resolution: H3 resolution, 0-15.
            op: Calling operation name, used in error messages.

        Returns:
            list[str]: One H3 cell index (hex string) per point, in row order.

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``.
            ValueError: If ``resolution`` is outside 0-15, or the collection has no CRS.
        """
        self._require_point_geometry(op)
        if not 0 <= resolution <= 15:
            raise ValueError(f"{op}: resolution must be 0-15, got {resolution}")
        if self.crs is None:
            raise ValueError(f"{op}: a CRS is required to convert points to lat/lng for H3 indexing")
        pts = self if self.epsg == 4326 else self.to_crs(4326)
        return [_h3.latlng_to_cell(geom.y, geom.x, resolution) for geom in pts.geometry]

    def to_h3(self, resolution: int) -> FeatureCollection:
        """Attach the H3 cell index of each point as an ``h3`` column.

        Indexes every point geometry into Uber's H3 hexagonal grid at the given resolution (computed in
        EPSG:4326 — points are reprojected for the lookup, but the returned collection keeps its own geometry
        and CRS). Uses pyramids' built-in H3 engine, so no ``h3`` dependency is required.

        Args:
            resolution: H3 resolution, 0 (coarsest) to 15 (finest).

        Returns:
            FeatureCollection: A copy of this collection with an ``h3`` column of cell-index strings.

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``.
            ValueError: If ``resolution`` is outside 0-15, or the collection has no CRS.

        Examples:
            - Index three points at resolution 9:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"id": [1, 2, 3]},
                ...         geometry=[Point(-122.418, 37.775), Point(-122.42, 37.776), Point(0, 0)],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> out = fc.to_h3(9)
                >>> out["h3"].tolist()
                ['89283082803ffff', '8928308280fffff', '89754e64993ffff']

                ```
        """
        cells = self._h3_cells(resolution, "to_h3")
        result = FeatureCollection(self.copy())
        result["h3"] = cells
        return result

    def h3_bin(
        self,
        resolution: int,
        *,
        agg: str | Callable = "count",
        column: str | None = None,
    ) -> FeatureCollection:
        """Aggregate points into H3 hexagon cells, one polygon per occupied cell.

        Groups the points by their H3 cell at ``resolution`` and returns one hexagon (or pentagon) polygon per
        occupied cell carrying an aggregate: the point **count** when ``column`` is ``None``, otherwise ``agg``
        applied to ``column``. The output is in EPSG:4326 (H3 is lat/lng). No ``h3`` dependency is required.

        Args:
            resolution: H3 resolution, 0-15.
            agg: Per-cell reducer applied to ``column`` — one of ``"count"`` / ``"mean"`` / ``"sum"`` /
                ``"median"`` / ``"min"`` / ``"max"`` / ``"std"`` or a callable taking a 1-D array. Ignored when
                ``column`` is ``None`` (point count).
            column: Numeric column aggregated per cell, or ``None`` to count points (density).

        Returns:
            FeatureCollection: One hexagon polygon per occupied cell, in EPSG:4326, with an ``h3`` index column
            and an aggregate column (``"count"`` when ``column`` is ``None``, else named ``column``).

        Raises:
            InvalidGeometryError: If the geometries are not all ``Point``.
            ValueError: If ``resolution`` is outside 0-15, the collection has no CRS, ``column`` is missing /
                non-numeric, or ``agg`` is not a known reducer name or callable.

        Examples:
            - Bin four nearby points into H3 cells at resolution 9 and read the counts:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection(
                ...     gpd.GeoDataFrame(
                ...         {"v": [1.0, 2.0, 3.0, 4.0]},
                ...         geometry=[
                ...             Point(-122.418, 37.775), Point(-122.4181, 37.7751),
                ...             Point(-122.40, 37.78), Point(-122.40, 37.78),
                ...         ],
                ...         crs="EPSG:4326",
                ...     )
                ... )
                >>> cells = fc.h3_bin(9)
                >>> int(cells["count"].sum())
                4
                >>> cells.crs.to_epsg()
                4326

                ```
        """
        self._require_column("h3_bin", column)
        cells = self._h3_cells(resolution, "h3_bin")
        if column is None:
            counts = pd.Series(cells, dtype="object").value_counts()
            items = [(cell, int(n)) for cell, n in counts.items()]
            name = "count"
        else:
            reducer = _tess.resolve_reducer(agg)
            try:
                values = self[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"h3_bin: column {column!r} must be numeric") from exc
            grouped = pd.DataFrame({"_cell": cells, "_v": values}).groupby("_cell")["_v"]
            items = [(cell, float(reducer(grp.to_numpy()))) for cell, grp in grouped]
            name = column
        geometries: list = []
        idx: list = []
        agg_values: list = []
        for cell, value in items:
            boundary = _h3.cell_to_boundary(cell)
            geometries.append(Polygon([(lng, lat) for (lat, lng) in boundary]))
            idx.append(cell)
            agg_values.append(value)
        frame = gpd.GeoDataFrame({"h3": idx, name: agg_values}, geometry=geometries, crs="EPSG:4326")
        return FeatureCollection(frame)
