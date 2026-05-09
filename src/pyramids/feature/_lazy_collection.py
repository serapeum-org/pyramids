"""Lazy vector collection — subclasses :class:`dask_geopandas.GeoDataFrame`.

A LazyFeatureCollection IS a
:class:`dask_geopandas.GeoDataFrame`; it is NOT a
:class:`~pyramids.feature.FeatureCollection` or :class:`geopandas.GeoDataFrame`.
Both classes satisfy the :class:`~pyramids.base.protocols.SpatialObject`
protocol so consumers accept either via `isinstance(x, SpatialObject)`.

Importing this module requires `dask-geopandas` to be installed (the
`[parquet-lazy]` extra). The eager `FeatureCollection` is unaffected:
its readers only import this module from inside the `backend='dask'`
branch, so minimal installs never evaluate this file.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import dask_geopandas
import geopandas

if TYPE_CHECKING:
    from pathlib import Path

    from pyramids.feature.collection import FeatureCollection


def _rebuild_lazy_fc(parent_reduce: tuple) -> LazyFeatureCollection:
    """Reconstruct a :class:`LazyFeatureCollection` from the parent reduce tuple.

    Runs the reconstructor returned by
    :meth:`dask_geopandas.GeoDataFrame.__reduce__`, re-applies any pickled
    state, then re-binds `__class__` so the restored object is a
    LazyFeatureCollection rather than a plain dask_geopandas frame.
    """
    reconstructor = parent_reduce[0]
    args = parent_reduce[1]
    instance = reconstructor(*args)
    if len(parent_reduce) > 2 and parent_reduce[2] is not None:
        state = parent_reduce[2]
        setter = getattr(instance, "__setstate__", None)
        if setter is not None:
            setter(state)
        else:
            instance.__dict__.update(state)
    instance.__class__ = LazyFeatureCollection
    return instance


class LazyFeatureCollection(dask_geopandas.GeoDataFrame):
    """Lazy vector collection backed by a dask_geopandas graph.

    Subclasses :class:`dask_geopandas.GeoDataFrame` — every method of that
    parent is available unchanged (`to_crs`, `clip`, `sjoin`,
    `.npartitions`, `.spatial_partitions`). Overrides: `compute` /
    `persist` return pyramids-native wrappers, `spatial_shuffle`
    re-wraps the shuffled result, `to_file` / `plot` raise with
    actionable messages, and `read_file` is a classmethod delegating
    back to the eager reader so the
    :class:`~pyramids.base.protocols.SpatialObject` protocol is satisfied.

    Construct via `FeatureCollection.read_file(..., backend='dask')` or
    `FeatureCollection.read_parquet(..., backend='dask')`. Direct
    construction is discouraged; use
    :meth:`LazyFeatureCollection.from_dask_gdf` if you must wrap a
    pre-built dask-geopandas frame.

    Note:
        On installs without the `[parquet-lazy]` extra, importing this
        class from :mod:`pyramids.feature` returns `None` (sentinel).
        Consumer code using `isinstance` must guard explicitly:
        `if LazyFeatureCollection is not None and isinstance(x, LazyFeatureCollection):...`.
    """

    def __getattribute__(self, name: str) -> Any:
        """Return a class-swapped view of every inherited method's result.

        dask-geopandas inherits from dask-expr and does not honour pandas'
        classic `_constructor` hook. Methods like `to_crs`, `clip`,
        `sjoin`, `copy`, `drop_duplicates` go through
        `map_partitions` with meta inference and produce a plain
        :class:`dask_geopandas.GeoDataFrame`, silently dropping the
        :class:`LazyFeatureCollection` subclass. This hook wraps every
        public callable so that a :class:`dask_geopandas.GeoDataFrame`
        result is rebranded back to the subclass before it reaches the
        caller. Private names (`_`-prefixed) are passed through
        unchanged to keep dask-expr internals — meta inference, name
        caches, partition probing — on the hot path they expect.
        """
        attr = object.__getattribute__(self, name)
        if name.startswith("_") or not callable(attr):
            return attr
        cls = type(self)

        @functools.wraps(attr)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, dask_geopandas.GeoDataFrame) and not isinstance(
                result,
                cls,
            ):
                result = cls.from_dask_gdf(result)
            elif isinstance(result, geopandas.GeoDataFrame) and not isinstance(
                result,
                geopandas.GeoSeries,
            ):
                from pyramids.feature.collection import FeatureCollection

                result = FeatureCollection(result)
            return result

        return wrapper

    @classmethod
    def from_dask_gdf(cls, dask_gdf: Any) -> LazyFeatureCollection:
        """Wrap an existing :class:`dask_geopandas.GeoDataFrame` as a LazyFC.

        This is the canonical construction path for code that already holds
        a pre-built :class:`dask_geopandas.GeoDataFrame` (for example,
        from :func:`dask_geopandas.from_geopandas` or a third-party reader)
        and wants to use it through the pyramids type system.

        Implementation uses class-swap (`copy()` + `__class__`
        assignment) rather than a four-arg reconstruction, because
        `(dask, _name, _meta, divisions)` is private dask.dataframe API
        that shifted between dask-expr and legacy. Class-swap relies only
        on Python's object model and survives upstream churn. The
        invariant "`LazyFeatureCollection` adds no extra instance state
        beyond :class:`dask_geopandas.GeoDataFrame`" is pinned by
        :func:`tests.feature.test_lazy_feature_collection.test_from_dask_gdf_preserves_state`.

        Args:
            dask_gdf: Pre-built `dask_geopandas.GeoDataFrame` to wrap.

        Returns:
            LazyFeatureCollection: A view over `dask_gdf` whose
            `__class__` has been rebound to LazyFeatureCollection.
            The task graph, partition names, metadata, and divisions
            are unchanged — only the concrete class is swapped.

        Examples:
            - Wrap a fresh dask_geopandas frame:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1, 2]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs="EPSG:4326",
                ... )
                >>> ddf = dg.from_geopandas(gdf, npartitions=1)
                >>> lfc = LazyFeatureCollection.from_dask_gdf(ddf)
                >>> lfc.npartitions
                1
                >>> lfc.epsg
                4326

                ```
        """
        result = dask_gdf.copy()
        result.__class__ = cls
        return result

    # Deprecated alias kept for one release cycle. Old callsites
    # (readers inside collection.py, tests) can transition at their own
    # pace; remove the alias in a follow-up commit.
    _from_dask_gdf = from_dask_gdf

    def __reduce__(self):
        """Preserve the LazyFeatureCollection class across pickle round-trips.

        `from_dask_gdf` uses `result.__class__ = cls` to rebind the
        concrete class. The parent's default `__reduce__` reconstructs
        with the base :class:`dask_geopandas.GeoDataFrame`, silently
        dropping the subclass on unpickle. Wrapping super's reduce tuple
        in a module-level trampoline re-applies the class swap.
        """
        return (_rebuild_lazy_fc, (super().__reduce__(),))

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        npartitions: int | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> LazyFeatureCollection:
        """Construct a LazyFC from a vector file.

        Satisfies the :class:`~pyramids.base.protocols.SpatialObject`
        protocol's `read_file` classmethod requirement. Delegates to
        :meth:`FeatureCollection.read_file` with `backend='dask'`.

        The import of :class:`FeatureCollection` is inline by
        necessity — :mod:`pyramids.feature.collection` imports this module
        from inside its dask branch, so a top-of-file import here would
        form a cycle. Per `CLAUDE.md`, circular-import breaks are an
        accepted exception to the no-inline-imports rule.

        Args:
            path: Path to a vector file GDAL/OGR can open (GeoJSON,
                Shapefile, GeoPackage, FlatGeobuf, …).
            npartitions: Target number of dask partitions. Mutually
                exclusive with `chunksize`.
            chunksize: Target rows per partition. Mutually exclusive
                with `npartitions`.
            **kwargs: Forwarded to :meth:`FeatureCollection.read_file`.
                Filter kwargs (`bbox` / `mask` / `rows` /
                `columns` / `where` / `layer`) are rejected
                because dask-geopandas has no pushdown for them.

        Returns:
            LazyFeatureCollection: A new lazy FC.

        Examples:
            - Read a small GeoJSON as a lazy FC via the classmethod:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1, 2, 3]},
                ...     geometry=[Point(i, i) for i in range(3)],
                ...     crs="EPSG:4326",
                ... )
                >>> d = Path(tempfile.mkdtemp())
                >>> p = d / "pts.geojson"
                >>> gdf.to_file(p, driver="GeoJSON")
                >>> lfc = LazyFeatureCollection.read_file(str(p), npartitions=1)
                >>> lfc.npartitions
                1
                >>> len(lfc.compute())
                3

                ```
        """
        from pyramids.feature.collection import FeatureCollection

        result = FeatureCollection.read_file(
            path,
            backend="dask",
            npartitions=npartitions,
            chunksize=chunksize,
            **kwargs,
        )
        return result

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or `None` when the CRS is unset.

        Cheap — reads the cached pyproj CRS attached to the
        dask-geopandas frame and calls `.to_epsg()`. No graph
        materialisation.

        Examples:
            - Inspect the EPSG of a lazy FC built from a WGS84 GDF:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1]},
                ...     geometry=[Point(0, 0)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=1)
                ... )
                >>> lfc.epsg
                4326

                ```
        """
        crs = self.crs
        result = None if crs is None else crs.to_epsg()
        return result

    def compute_total_bounds(self) -> Any:
        """Materialise :attr:`total_bounds` via an explicit `.compute()` call.

        :attr:`total_bounds` on a
        :class:`dask_geopandas.GeoDataFrame` is a lazy dask Scalar.
        Most consumers who want concrete numbers today wrote
        `lfc.total_bounds.compute()` — this helper gives them a
        single-line, self-documenting equivalent that also makes the
        compute-cost explicit in the method name.

        Returns:
            numpy.ndarray: `[minx, miny, maxx, maxy]` as a
            4-element 1-D array — the exact shape
            :class:`geopandas.GeoDataFrame.total_bounds` returns.

        Examples:
            - Force the reduction and destructure into named values:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1, 2]},
                ...     geometry=[Point(0, 0), Point(10, 20)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=1)
                ... )
                >>> xmin, ymin, xmax, ymax = lfc.compute_total_bounds()
                >>> xmin, ymax
                (0.0, 20.0)

                ```
        """
        return self.total_bounds.compute()

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """No lazy plot path; materialize first.

        Raises:
            NotImplementedError: Always. Call `.compute().plot(...)` to
                materialize before plotting.

        Examples:
            - Calling `.plot` directly raises with an actionable hint:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1]},
                ...     geometry=[Point(0, 0)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=1)
                ... )
                >>> try:
                ...     lfc.plot()
                ... except NotImplementedError as exc:
                ...     "compute().plot" in str(exc)
                True

                ```
        """
        raise NotImplementedError(
            "LazyFeatureCollection.plot is not supported — call "
            ".compute().plot(...) to materialize first."
        )

    def compute(self, **kwargs: Any) -> FeatureCollection:
        """Materialize the graph and return an eager :class:`FeatureCollection`.

        The `from pyramids.feature.collection import FeatureCollection`
        import below is inline by necessity — `collection.py` imports
        this module from inside its dask branch, so a top-of-file import
        here would form a cycle. Per `CLAUDE.md`, circular-import
        breaks are the second accepted exception to the no-inline-imports
        rule.

        Args:
            **kwargs: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.compute` (e.g.
                `scheduler="processes"`).

        Returns:
            FeatureCollection: an eager wrapper around the materialized
            :class:`geopandas.GeoDataFrame`.

        Examples:
            - Compute a two-partition lazy frame back to eager form:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import (
                ...     FeatureCollection, LazyFeatureCollection,
                ... )
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1, 2, 3, 4]},
                ...     geometry=[Point(i, i) for i in range(4)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=2)
                ... )
                >>> eager = lfc.compute()
                >>> len(eager)
                4
                >>> eager.epsg
                4326

                ```
        """
        from pyramids.feature.collection import FeatureCollection

        gdf = super().compute(**kwargs)
        result = FeatureCollection(gdf)
        return result

    def persist(self, **kwargs: Any) -> LazyFeatureCollection:
        """Persist partitions in worker memory; keep the FC lazy.

        Args:
            **kwargs: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.persist`.

        Returns:
            LazyFeatureCollection: A new lazy FC whose graph has been
            materialized into worker memory.

        Examples:
            - Persist and confirm the result is still lazy:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1, 2]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=2)
                ... )
                >>> persisted = lfc.persist()
                >>> isinstance(persisted, LazyFeatureCollection)
                True
                >>> persisted.npartitions
                2

                ```
        """
        persisted = super().persist(**kwargs)
        result = type(self).from_dask_gdf(persisted)
        return result

    def spatial_shuffle(
        self,
        by: str = "hilbert",
        level: int = 16,
        *,
        npartitions: int | None = None,
        divisions: Any = None,
    ) -> LazyFeatureCollection:
        """Spatially sort rows via a space-filling curve.

        Thin override of :meth:`dask_geopandas.GeoDataFrame.spatial_shuffle`
        that rewraps the result as a :class:`LazyFeatureCollection`.
        Populates :attr:`spatial_partitions`, enabling partition-pruned
        :meth:`sjoin` on the result.

        Args:
            by: One of `"hilbert"`, `"morton"`, or `"geohash"`.
            level: Curve resolution; default 16.
            npartitions: Optional output partition count.
            divisions: Optional explicit division boundaries.

        Returns:
            LazyFeatureCollection: A new lazy FC with `spatial_partitions`
            populated.

        Examples:
            - Shuffle a two-partition frame and confirm bboxes are set:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": list(range(8))},
                ...     geometry=[Point(i, i) for i in range(8)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=2)
                ... )
                >>> shuffled = lfc.spatial_shuffle()
                >>> shuffled.spatial_partitions is not None
                True

                ```
        """
        kwargs: dict[str, Any] = {}
        if npartitions is not None:
            kwargs["npartitions"] = npartitions
        if divisions is not None:
            kwargs["divisions"] = divisions
        shuffled = super().spatial_shuffle(by=by, level=level, **kwargs)
        result = type(self).from_dask_gdf(shuffled)
        return result

    def to_file(self, path: Any, *args: Any, **kwargs: Any) -> None:
        """No lazy OGR write path exists; raise with an actionable message.

        Raises:
            NotImplementedError: Always. dask-geopandas has no lazy OGR
                write path for GeoJSON / Shapefile / GeoPackage. Call
                `.compute().to_file(path)` to materialize first, or
                use `to_parquet` which IS lazy.

        Examples:
            - The hint at the raise site points to the supported paths:
                ```python
                >>> import geopandas as gpd
                >>> import dask_geopandas as dg
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import LazyFeatureCollection
                >>> gdf = gpd.GeoDataFrame(
                ...     {"v": [1]},
                ...     geometry=[Point(0, 0)],
                ...     crs="EPSG:4326",
                ... )
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=1)
                ... )
                >>> try:
                ...     lfc.to_file("out.geojson")
                ... except NotImplementedError as exc:
                ...     "compute().to_file" in str(exc)
                True

                ```
        """
        raise NotImplementedError(
            "LazyFeatureCollection.to_file is not supported — "
            "dask-geopandas has no lazy OGR write path. Call "
            ".compute().to_file(path) to materialize first, or "
            "write via read_parquet + fc.to_parquet(path) which IS lazy."
        )

    def to_parquet(self, path: Any, *args: Any, **kwargs: Any) -> None:
        """Write the lazy frame to GeoParquet (the one genuinely-lazy writer).

        explicit wrapper around
        :meth:`dask_geopandas.GeoDataFrame.to_parquet`. The parent
        returns a :class:`dask.delayed.Delayed` (or `None` when the
        call is implicitly computed), which leaks the dask detail into
        user code and breaks the pyramids convention that `to_*`
        writers return `None`. This override normalises the contract:

        * The call blocks until every partition has been written.
        * Return is always `None` — matches
          :meth:`FeatureCollection.to_parquet` and
          :meth:`LazyFeatureCollection.to_file` (the raising stub).
        * All other kwargs (`compression=`, `write_index=`,
          `storage_options=`) forward to the parent.

        Args:
            path: Destination path or directory. Cloud URLs (`s3://`,
                `gs://`, `az://`) work via `storage_options`.
            *args: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.to_parquet`.
            **kwargs: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.to_parquet`.
                `compute=True` is forced; passing `compute=False`
                raises :class:`ValueError` because the pyramids API
                contract is "`to_*` always writes, never defers".

        Raises:
            ValueError: If `compute=False` is supplied. Use the
                parent's API directly if you need a Delayed.

        Examples:
            - Round-trip a lazy FC through GeoParquet:
                ```python
                >>> import tempfile  # doctest: +SKIP
                >>> from pathlib import Path  # doctest: +SKIP
                >>> import geopandas as gpd  # doctest: +SKIP
                >>> import dask_geopandas as dg  # doctest: +SKIP
                >>> from shapely.geometry import Point  # doctest: +SKIP
                >>> from pyramids.feature import LazyFeatureCollection  # doctest: +SKIP
                >>> d = Path(tempfile.mkdtemp())  # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(
                ...     {"id": [1, 2]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs="EPSG:4326",
                ... )  # doctest: +SKIP
                >>> lfc = LazyFeatureCollection.from_dask_gdf(
                ...     dg.from_geopandas(gdf, npartitions=1)
                ... )  # doctest: +SKIP
                >>> lfc.to_parquet(d / "out.parquet")  # doctest: +SKIP

                ```
        """
        if kwargs.get("compute") is False:
            raise ValueError(
                "LazyFeatureCollection.to_parquet always writes — "
                "passing compute=False is not supported. Use "
                "self._data.to_parquet(...) or go through the "
                "dask_geopandas parent API directly if you need a "
                "Delayed."
            )
        kwargs["compute"] = True
        super().to_parquet(path, *args, **kwargs)

    def __repr__(self) -> str:
        """Pyramids-branded repr; avoids the generic Dask DataFrame one."""
        return (
            f"LazyFeatureCollection(npartitions={self.npartitions}, "
            f"columns={list(self.columns)}, crs={self.crs})"
        )
