"""Lazy, dask-backed :class:`~pyramids.netcdf.NetCDF` cube — the lazy twin of an eager cube.

Mirrors the vector-side :class:`~pyramids.feature.LazyFeatureCollection` for the raster cube
(issue #1229). A :class:`LazyNetCDF` holds one :class:`dask.array.Array` per gridded variable —
read lazily through :meth:`NetCDF.read_array` (which builds the array with
:func:`pyramids.netcdf._lazy.build_lazy_array`) — alongside the eager cube it was built from, which
is both the source of its geo / band metadata and the eager twin its lifecycle materialises back to.

A cube cannot subclass a dask type the way ``LazyFeatureCollection`` subclasses
``dask_geopandas.GeoDataFrame`` (a cube is many arrays, not one frame), so this is a wrapper rather
than a class-swap. The dask-lifecycle members live here: :attr:`~LazyNetCDF.chunks` /
:attr:`~LazyNetCDF.chunksizes` read the chunking off the arrays; :meth:`~LazyNetCDF.compute` /
:meth:`~LazyNetCDF.load` return the eager cube; :meth:`~LazyNetCDF.persist` realises the graph in
worker memory while staying lazy; :meth:`~LazyNetCDF.unify_chunks` re-chunks every variable to a
common chunking across shared dimensions.

**Operation boundary.** Any eager operation (an array reduction, a GDAL spatial op, I/O) reached
through attribute access materialises the cube — it warns once and delegates to the eager twin —
because this v1 does not yet compose array-native ops lazily across a chain; the per-operation dask
streaming inside :meth:`NetCDF._materialize_variable_array` already gives most of the memory
benefit for a single reduction. Keeping a chain of array-native ops lazy at the cube level is the
documented next step for issue #1229.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pyramids.base._utils import import_dask

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.netcdf.netcdf import NetCDF

_DASK_MISSING = "The lazy NetCDF cube needs dask; install the `lazy` extra."
_MISSING = (
    object()
)  # sentinel so __getattr__ fetches a forwarded attribute exactly once


class _LazyVar(NamedTuple):
    """One variable's deferred state: its dask array and the band layout describing it.

    Carried by a :class:`LazyNetCDF` that has composed one or more array-native ops (#1237), so
    :meth:`LazyNetCDF.compute` can rebuild the eager variable from the *transformed* array through
    the same `_stack_reduced_variable` path the eager ops use.

    Attributes:
        array: The dask array, laid out `(*band_dim_sizes, rows, cols)`.
        band_names: The result's band dimensions, outermost first.
        values_map: Each band dimension's coordinates, or `None` for one without.
        no_data: The no-data value the result declares, or `None`.
        geotransform: The variable's geotransform (a band op never moves the spatial plane).
        epsg: The CRS spec (`crs_spec(epsg, crs)`) carried onto the rebuilt variable.
        dim_names: The dimension names outermost first, `(*band_names, row_name, col_name)`.
        time_attrs: Resolved CF time units per band dimension, for the stamps that survive.
        source_var: The original eager variable, passed as `source=` to the rebuild so its
            attributes and CRS carry onto the result.
    """

    array: Any
    band_names: list[str]
    values_map: dict[str, Any]
    no_data: Any
    geotransform: tuple[float, float, float, float, float, float]
    epsg: Any
    dim_names: tuple[str, ...]
    time_attrs: dict[str, Any]
    source_var: Any


class LazyNetCDF:
    """A lazy, dask-backed view of a :class:`~pyramids.netcdf.NetCDF` cube.

    Built by :meth:`NetCDF.chunk`. Holds a :class:`dask.array.Array` per gridded variable and the
    eager cube they were read from; the eager cube supplies geo / band metadata and is the twin
    returned by :meth:`compute` / :meth:`load`.

    Attributes:
        variable_names (list[str]): The gridded variables carried as dask arrays.

    Examples:
        - A lazy view reports its chunking and computes back to its eager twin:

          ```python
          >>> from pyramids.netcdf import NetCDF  # doctest: +SKIP
          >>> lazy = NetCDF.read_file("cube.nc").chunk("auto")  # doctest: +SKIP
          >>> lazy.chunks  # doctest: +SKIP
          {'time': (4,), 'y': (256, 256), 'x': (256, 256)}
          >>> eager = lazy.compute()  # doctest: +SKIP

          ```
    """

    def __init__(
        self,
        source: NetCDF,
        arrays: dict[str, Any],
        dim_names: dict[str, tuple[str, ...]],
        records: dict[str, _LazyVar] | None = None,
    ) -> None:
        """Wrap per-variable dask arrays alongside the eager cube they came from.

        Args:
            source: The eager cube this view was built from — the source of its metadata and the
                twin :meth:`compute` returns when no op has transformed the arrays.
            arrays: One dask array per gridded variable, laid out `(*band_sizes, rows, cols)`.
            dim_names: The dimension names per variable, outermost first, i.e.
                `(*band_dim_names, row_name, col_name)` — the keys :attr:`chunks` reports.
            records: Per-variable deferred state, set once an array-native op has composed over the
                cube (#1237). When present the cube is **transformed**: :meth:`compute` rebuilds the
                eager cube from these arrays rather than returning the source. `None` (the default,
                from :meth:`NetCDF.chunk`) is an untransformed view whose arrays are chunked reads of
                the source.
        """
        self._source = source
        self._arrays = dict(arrays)
        self._dim_names = dict(dim_names)
        self._records = records
        self._materialize_warned = False

    @property
    def variable_names(self) -> list[str]:
        """The gridded variables this view carries, in build order."""
        return list(self._arrays)

    @property
    def chunks(self) -> dict[str, tuple[int, ...]]:
        """Per-dimension chunk sizes, keyed by dimension name.

        Merged across variables — two variables that disagree on a shared dimension leave the last
        one's chunking here until :meth:`unify_chunks` reconciles them.
        """
        merged: dict[str, tuple[int, ...]] = {}
        for name, array in self._arrays.items():
            for dim, chunk in zip(self._dim_names[name], array.chunks):
                merged[dim] = tuple(int(size) for size in chunk)
        return merged

    @property
    def chunksizes(self) -> dict[str, tuple[int, ...]]:
        """Alias of :attr:`chunks` (xarray keeps both)."""
        return self.chunks

    def compute(self, **kwargs: Any) -> NetCDF:
        """Materialise the lazy view and return an **independent** eager NetCDF cube.

        The lazy arrays are chunked reads of the source cube, so the result holds the same data; it is
        returned as an independent copy (`compute() is not` the source), matching xarray's `compute()`,
        so mutating the result never touches the original or any other `compute()` result. The
        `**kwargs` are accepted for API symmetry with xarray / dask and ignored — in this v1 the data is
        a view of the already-eager source, so nothing has to be scheduled.

        Returns:
            NetCDF: A fresh eager cube, independent of the source.
        """
        if self._records is not None:
            # A cube transformed by a composed op (#1237): materialise the deferred dask arrays
            # through the eager rebuild rather than returning the (untransformed) source.
            return self._rebuild()
        return self._source.copy()

    def load(self, **kwargs: Any) -> NetCDF:
        """Materialise in place and return the eager cube — the in-place form of :meth:`compute`.

        Matches xarray's `load`, which realises in place and returns the same object: this returns the
        **source cube itself** (not a copy), so it aliases the cube `.chunk()` was called on. Use
        :meth:`compute` when you need an independent result. The `**kwargs` are accepted for symmetry
        and ignored.

        Returns:
            NetCDF: The source cube (the eager twin), realised in place. A transformed cube
            (a composed op) has no untransformed source to alias, so it returns the rebuilt eager
            result, as :meth:`compute` does.
        """
        if self._records is not None:
            return self._rebuild()
        return self._source

    def persist(self, **kwargs: Any):
        """Realise the dask graph in worker memory, staying lazy.

        Returns a new :class:`LazyNetCDF` whose arrays are `dask.persist`-ed — the graph is kept but
        its results are held in memory / across the cluster, so later reads do not recompute it.

        Returns:
            LazyNetCDF: A lazy view over the persisted arrays.
        """
        import_dask(_DASK_MISSING)
        import dask

        names = list(self._arrays)
        persisted = dask.persist(*(self._arrays[name] for name in names), **kwargs)
        arrays = dict(zip(names, persisted))
        return LazyNetCDF(self._source, arrays, self._dim_names)

    def unify_chunks(self):
        """Re-chunk every variable to a common chunking across their shared dimensions.

        A no-op when the view holds fewer than two variables. Two variables chunked differently on a
        shared dimension come out sharing one chunking afterward.

        Returns:
            LazyNetCDF: A lazy view whose variables agree on their shared-dimension chunking.
        """
        import_dask(_DASK_MISSING)
        import dask.array as da

        if len(self._arrays) < 2:
            return self
        names = list(self._arrays)
        pairs: list[Any] = []
        for name in names:
            pairs.extend((self._arrays[name], self._dim_names[name]))
        _, unified = da.core.unify_chunks(*pairs)
        arrays = dict(zip(names, unified))
        return LazyNetCDF(self._source, arrays, self._dim_names, self._records)

    def _current_records(self) -> dict[str, _LazyVar]:
        """The per-variable deferred records, derived from the source for an untransformed view.

        An untransformed cube (`_records is None`, straight from :meth:`NetCDF.chunk`) carries only
        chunked reads; its records are read off the source variables so a first op has a uniform
        record to transform. A transformed cube returns its carried records.

        Returns:
            dict[str, _LazyVar]: One record per gridded variable.
        """
        if self._records is not None:
            return self._records
        from pyramids.base.crs import crs_spec
        from pyramids.netcdf.engines._along_dim import _read_no_data

        source = self._source
        pinned = source._source_var_name
        records: dict[str, _LazyVar] = {}
        for name in self._arrays:
            var = (
                source if pinned is not None else source._require_raster_variable(name)
            )
            records[name] = _LazyVar(
                array=self._arrays[name],
                band_names=list(var._band_dim_names),
                values_map=dict(var._band_dim_values_map),
                no_data=_read_no_data(var),
                geotransform=var.geotransform,
                epsg=crs_spec(var.epsg, var.crs),
                dim_names=self._dim_names[name],
                time_attrs=var._resolved_band_dim_time_attrs(),
                source_var=var,
            )
        return records

    def _rebuild(self) -> NetCDF:
        """Materialise the transformed records into an eager cube through the eager rebuild path.

        One record (a pinned variable, the composition path this version supports) rebuilds through
        :func:`_variable_from_applied`, the same helper the eager ops use, so the result's band
        coordinates, unlabelled-dimension gaps and CF time units match the eager reduce exactly.

        Returns:
            NetCDF: The eager variable the deferred ops computed to.
        """
        from pyramids.base.protocols import as_numpy
        from pyramids.netcdf.engines._along_dim import _Applied, _variable_from_applied

        records = self._current_records()
        (rec,) = records.values()
        applied = _Applied(
            as_numpy(rec.array), rec.band_names, rec.values_map, rec.no_data
        )
        return _variable_from_applied(rec.source_var, applied)

    def reduce(
        self,
        dim: str,
        how: str = "mean",
        *,
        skipna: bool = True,
        q: float | None = None,
    ) -> Any:
        """Reduce a band dimension, composing lazily when the cube is a single variable (#1237).

        Collapses `dim` with `how`, keeping the result a lazy cube: the reduction is a `dask.array`
        step deferred until :meth:`compute`, so a chain of reductions stays lazy and reads the store
        only once per block. Mirrors :meth:`NetCDF.reduce`.

        A single pinned variable composes lazily and returns a :class:`LazyNetCDF`. A multi-variable
        container cannot yet compose at the cube level, so it auto-materialises to the eager cube and
        reduces there, warning once — the v1 lazy/eager boundary (the same shape as the GDAL-op
        boundary). Pin a variable with `get_variable(...)` first to keep a container reduce lazy.

        Args:
            dim: The band dimension to collapse.
            how: The reduction (`"mean"`, `"sum"`, `"max"`, a quantile, …), as :meth:`NetCDF.reduce`.
            skipna: Whether gaps are skipped.
            q: The quantile, for `how="quantile"`.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager reduced cube for a
            multi-variable container (materialised at the boundary).
        """
        if len(self._current_records()) != 1:
            # Container composition is not built yet: materialise at the boundary and reduce
            # eagerly, warning once, rather than silently reading the whole cube without notice.
            self._warn_materialize("reduce")
            return self.compute().reduce(dim, how, skipna=skipna, q=q)
        return self._compose_reduction(
            dim,
            how,
            skipna=skipna,
            q=q,
            group_positions=None,
            resize=None,
            window_mean_coords=False,
        )

    def coarsen(
        self,
        dim: str,
        window: int,
        *,
        how: str = "mean",
        boundary: str = "exact",
        skipna: bool = True,
        q: float | None = None,
    ) -> Any:
        """Block-aggregate a band dimension into windows, composing lazily for a single variable.

        The lazy twin of :meth:`NetCDF.coarsen` — the same grouped reduction, kept as a deferred
        `dask.array` step so it composes with other lazy ops. A multi-variable container
        materialises at the boundary and coarsens eagerly, warning once (as :meth:`reduce` does).

        Args:
            dim: The band dimension to block-aggregate.
            window: The number of steps per window.
            how: The reduction, as :meth:`NetCDF.coarsen`.
            boundary: How a trailing partial window is handled (`"exact"`, `"trim"`, `"pad"`).
            skipna: Whether gaps are skipped.
            q: The quantile, for `how="quantile"`.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager coarsened cube for a
            multi-variable container.
        """
        from pyramids.netcdf.engines.selection import (
            _BOUNDARIES,
            _check_how,
            _check_quantile,
            _check_window,
            _coarsen_windows,
        )
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        length = _check_window(window, caller="coarsen")
        if boundary not in _BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {list(_BOUNDARIES)}, got {boundary!r}."
            )
        records = self._current_records()
        if len(records) != 1:
            self._warn_materialize("coarsen")
            return self.compute().coarsen(
                dim, window, how=how, boundary=boundary, skipna=skipna, q=q
            )
        (rec,) = records.values()
        size = rec.array.shape[rec.band_names.index(dim)]
        resized, positions = _coarsen_windows(dim, size, length, boundary)
        return self._compose_reduction(
            dim,
            how,
            skipna=skipna,
            q=q,
            group_positions=positions,
            resize=resized,
            window_mean_coords=True,
        )

    def rolling(
        self,
        dim: str,
        window: int,
        *,
        how: str = "mean",
        center: bool = False,
        min_periods: int | None = None,
        q: float | None = None,
    ) -> Any:
        """Moving-window reduction along a band dimension, composing lazily for a single variable.

        The lazy twin of :meth:`NetCDF.rolling` — the window reduction stays a deferred
        `dask.array` step. A multi-variable container materialises at the boundary, warning once.

        Args:
            dim: The band dimension to roll along.
            window: The window length in steps.
            how: The reduction, as :meth:`NetCDF.rolling`.
            center: Whether the window is centred rather than trailing.
            min_periods: The minimum valid steps for a non-gap result; defaults to `window`.
            q: The quantile, for `how="quantile"`.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager rolled cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _rolled_array
        from pyramids.netcdf.engines.selection import (
            _check_how,
            _check_min_periods,
            _check_quantile,
            _check_window,
        )
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        length = _check_window(window, caller="rolling")
        needed = _check_min_periods(min_periods, length)
        if len(self._current_records()) != 1:
            self._warn_materialize("rolling")
            return self.compute().rolling(
                dim, window, how=how, center=center, min_periods=min_periods, q=q
            )
        return self._compose_direct(
            _rolled_array,
            dim,
            window=length,
            center=bool(center),
            min_periods=needed,
            how=how,
            q=q,
        )

    def diff(self, dim: str, n: int = 1, *, label: str = "upper") -> Any:
        """Difference neighbouring steps along a band dimension, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.diff`; the difference stays a deferred `dask.array` step.
        A multi-variable container differences eagerly at the boundary, warning once.

        Args:
            dim: The band dimension to difference along.
            n: The order (how many times the difference is taken).
            label: `"upper"` or `"lower"` — which step each difference is labelled with.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager diffed cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _diffed_array
        from pyramids.netcdf.engines.selection import _DIFF_LABELS, _check_order

        order = _check_order(n)
        if label not in _DIFF_LABELS:
            raise ValueError(
                f"label must be one of {list(_DIFF_LABELS)}, got {label!r}."
            )
        if len(self._current_records()) != 1:
            self._warn_materialize("diff")
            return self.compute().diff(dim, n, label=label)
        return self._compose_direct(_diffed_array, dim, n=order, label=label)

    def cumsum(self, dim: str, *, skipna: bool = True) -> Any:
        """Running total along a band dimension, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.cumsum`; the cumulative sum stays a deferred `dask.array`
        step, keeping the dimension's length. A multi-variable container totals eagerly at the
        boundary, warning once.

        Args:
            dim: The band dimension to total along.
            skipna: Whether gaps are skipped.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager totalled cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _CumSum

        if len(self._current_records()) != 1:
            self._warn_materialize("cumsum")
            return self.compute().cumsum(dim, skipna=skipna)
        return self._compose_op(_CumSum(skipna=bool(skipna)), dim)

    def shift(self, dim: str, periods: int = 1, *, fill_value: Any = None) -> Any:
        """Shift values along a band dimension, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.shift`; the shift stays a deferred `dask.array` step,
        keeping the dimension's length. A multi-variable container shifts eagerly at the boundary,
        warning once.

        Args:
            dim: The band dimension to shift along.
            periods: How many steps to move (positive towards the end, negative the other way).
            fill_value: The value for the vacated steps; the variable's no-data (or NaN) by default.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager shifted cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _Shift
        from pyramids.netcdf.engines.selection import _check_fill_value, _check_periods

        steps = _check_periods(periods)
        _check_fill_value(fill_value)
        if len(self._current_records()) != 1:
            self._warn_materialize("shift")
            return self.compute().shift(dim, periods, fill_value=fill_value)
        return self._compose_op(_Shift(periods=steps, fill_value=fill_value), dim)

    def rank(self, dim: str, *, pct: bool = False) -> Any:
        """Rank each pixel's values along a band dimension, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.rank`. `rank` uses `scipy.stats.rankdata`, which needs a
        numpy array, so it cannot stay a pure dask graph; instead the eager kernel runs per spatial
        block (the band axis whole) through `dask.array.map_blocks`, which is lazy and bit-for-bit
        with the eager rank. A multi-variable container ranks eagerly at the boundary, warning once.

        Args:
            dim: The band dimension to rank along.
            pct: Whether to return ranks as a fraction of the valid count.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager ranked cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _Rank

        if len(self._current_records()) != 1:
            self._warn_materialize("rank")
            return self.compute().rank(dim, pct=pct)
        return self._compose_mapblocks(_Rank(pct=pct), dim)

    def argmin(self, dim: str, *, skipna: bool = True) -> Any:
        """Position of the minimum along a band dimension, composing lazily (#1237)."""
        return self._extremum(dim, "min", False, "argmin", skipna)

    def argmax(self, dim: str, *, skipna: bool = True) -> Any:
        """Position of the maximum along a band dimension, composing lazily (#1237)."""
        return self._extremum(dim, "max", False, "argmax", skipna)

    def idxmin(self, dim: str, *, skipna: bool = True) -> Any:
        """Coordinate of the minimum along a band dimension, composing lazily (#1237)."""
        return self._extremum(dim, "min", True, "idxmin", skipna)

    def idxmax(self, dim: str, *, skipna: bool = True) -> Any:
        """Coordinate of the maximum along a band dimension, composing lazily (#1237)."""
        return self._extremum(dim, "max", True, "idxmax", skipna)

    def interpolate_na(
        self,
        dim: str,
        method: str = "linear",
        *,
        limit: int | None = None,
        use_coordinate: bool = True,
    ) -> Any:
        """Fill interior gaps along a band dimension, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.interpolate_na`; the interpolation (a scipy kernel) runs per
        spatial block via map_blocks, keeping the dimension's length. A multi-variable container
        interpolates eagerly at the boundary, warning once.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager result otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _Interpolate
        from pyramids.netcdf.engines.selection import _check_limit

        if len(self._current_records()) != 1:
            self._warn_materialize("interpolate_na")
            return self.compute().interpolate_na(
                dim, method, limit=limit, use_coordinate=use_coordinate
            )
        op = _Interpolate(
            method=method,
            limit=_check_limit(limit, caller="interpolate_na"),
            use_coordinate=bool(use_coordinate),
        )
        return self._compose_mapblocks(op, dim)

    def interp(self, method: str = "linear", **coords: Any) -> Any:
        """Interpolate band dimensions onto new coordinate values, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.interp`; each `dim=targets` interpolation (a scipy kernel)
        runs per spatial block via map_blocks, resizing the dimension to the target length. Several
        `dim=targets` pairs compose in sequence. A multi-variable container interpolates eagerly at
        the boundary, warning once.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager result otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _InterpTo
        from pyramids.netcdf.engines.selection import (
            _INTERP_MIN_POINTS,
            _interp_targets,
            _refuse_spatial_interp,
            _resolve_interp_kind,
        )

        kind = _resolve_interp_kind(method)
        if not coords or len(self._current_records()) != 1:
            self._warn_materialize("interp")
            return self.compute().interp(method, **coords)
        result: Any = self
        for dim, target in coords.items():
            _refuse_spatial_interp(self._source, dim, caller="interp")
            rec = next(iter(result._current_records().values()))
            if dim not in rec.band_names:
                raise ValueError(
                    f"interp(): {dim!r} is not a band dimension of this variable."
                )
            size = rec.array.shape[rec.band_names.index(dim)]
            minimum = _INTERP_MIN_POINTS[kind]
            if size < minimum:
                raise ValueError(
                    f"interp() method {kind!r} needs at least {minimum} source steps along "
                    f"{dim!r}, got {size}."
                )
            targets = _interp_targets(target, dim, caller="interp")
            result = result._compose_mapblocks(
                _InterpTo(target=targets, kind=kind, caller="interp"), dim
            )
        return result

    def pad(
        self, *, mode: str = "constant", constant_values: Any = None, **pad_width: Any
    ) -> Any:
        """Pad band dimensions with a constant fill, composing lazily (#1237).

        The lazy twin of :meth:`NetCDF.pad` for **band** dimensions (extended via map_blocks). A
        spatial-axis pad moves the geotransform (a GDAL op) and a multi-variable container cannot
        compose, so either materialises at the boundary, warning once; several band dimensions in
        one call compose in sequence.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube when every padded dimension is a band dimension of a
            single variable; an eager padded cube otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _Pad
        from pyramids.netcdf.engines.selection import _pad_before_after

        records = self._current_records()
        band_names = next(iter(records.values())).band_names if records else []
        if (
            mode != "constant"
            or not pad_width
            or len(records) != 1
            or any(dim not in band_names for dim in pad_width)
        ):
            self._warn_materialize("pad")
            return self.compute().pad(
                mode=mode, constant_values=constant_values, **pad_width
            )
        result: Any = self
        for dim, width in pad_width.items():
            before, after = _pad_before_after(width, dim)
            result = result._compose_mapblocks(
                _Pad(before=before, after=after, fill_value=constant_values), dim
            )
        return result

    def clip(self, min: Any = None, max: Any = None) -> Any:
        """Bound the values to `[min, max]`, composing lazily (#1237); a gap stays a gap.

        Cell-wise and shape-preserving, so it runs the eager :meth:`NetCDF.clip` per spatial block
        via map_blocks (bit-for-bit; the dtype widening is bound-based, so it is the same for every
        block). A multi-variable container clips eagerly at the boundary, warning once.
        """
        if len(self._current_records()) != 1:
            self._warn_materialize("clip")
            return self.compute().clip(min=min, max=max)
        return self._compose_cellwise("clip", min, max)

    def fillna(self, value: float | int) -> Any:
        """Fill the gaps with `value`, composing lazily (#1237).

        Cell-wise and shape-preserving; runs the eager :meth:`NetCDF.fillna` per block via
        map_blocks. A multi-variable container fills eagerly at the boundary, warning once.
        """
        if len(self._current_records()) != 1:
            self._warn_materialize("fillna")
            return self.compute().fillna(value)
        return self._compose_cellwise("fillna", value)

    def round(self, decimals: int = 0) -> Any:
        """Round the values, composing lazily (#1237); a gap stays a gap.

        Cell-wise and shape-preserving; runs the eager :meth:`NetCDF.round` per block via
        map_blocks. An integer band rounded to tens (`decimals < 0`) may widen based on the data,
        which is not consistent across blocks, so that case materialises at the boundary, as does a
        multi-variable container — both warn once.
        """
        if isinstance(decimals, bool) or not isinstance(decimals, (int, np.integer)):
            raise TypeError(
                f"round() needs an integer number of decimals, got {decimals!r}."
            )
        records = self._current_records()
        integer_widen = (
            len(records) == 1
            and next(iter(records.values())).array.dtype.kind in "iu"
            and int(decimals) < 0
        )
        if len(records) != 1 or integer_widen:
            self._warn_materialize("round")
            return self.compute().round(decimals)
        return self._compose_cellwise("round", decimals)

    def _binary_op(self, other: Any, op_name: str) -> Any:
        """A cell-wise operator: compose lazily for a real scalar, else materialise (#1237).

        `lazy * 2`, `lazy - 273.15`, `lazy >= threshold` and the like run the eager operator per
        block (the result dtype/sentinel is value-based, so it is consistent across blocks). A
        raster/array operand needs grid + band alignment through the combine engine, and a
        multi-variable container cannot compose, so either materialises at the boundary, warning
        once.
        """
        from numbers import Real

        scalar = isinstance(other, Real) and not isinstance(other, bool)
        if not scalar or len(self._current_records()) != 1:
            self._warn_materialize(op_name.strip("_"))
            return getattr(self.compute(), op_name)(other)
        return self._compose_cellwise(op_name, other)

    def _unary_op(self, op_name: str) -> Any:
        """A cell-wise unary operator (`-cube`, `abs(cube)`): compose per block, else materialise."""
        if len(self._current_records()) != 1:
            self._warn_materialize(op_name.strip("_"))
            return getattr(self.compute(), op_name)()
        return self._compose_cellwise(op_name)

    def __add__(self, other: Any) -> Any:
        """Add a real scalar cell by cell, lazily (a raster operand materialises)."""
        return self._binary_op(other, "__add__")

    def __radd__(self, other: Any) -> Any:
        """Right-hand add of a real scalar, lazily."""
        return self._binary_op(other, "__radd__")

    def __sub__(self, other: Any) -> Any:
        """Subtract a real scalar cell by cell, lazily."""
        return self._binary_op(other, "__sub__")

    def __rsub__(self, other: Any) -> Any:
        """Right-hand subtract of a real scalar, lazily."""
        return self._binary_op(other, "__rsub__")

    def __mul__(self, other: Any) -> Any:
        """Multiply by a real scalar cell by cell, lazily."""
        return self._binary_op(other, "__mul__")

    def __rmul__(self, other: Any) -> Any:
        """Right-hand multiply by a real scalar, lazily."""
        return self._binary_op(other, "__rmul__")

    def __truediv__(self, other: Any) -> Any:
        """Divide by a real scalar cell by cell, lazily."""
        return self._binary_op(other, "__truediv__")

    def __pow__(self, other: Any) -> Any:
        """Raise to a real-scalar power cell by cell, lazily."""
        return self._binary_op(other, "__pow__")

    def __ge__(self, other: Any) -> Any:
        """Cell-by-cell `>=` a real scalar, as a Byte mask, lazily."""
        return self._binary_op(other, "__ge__")

    def __gt__(self, other: Any) -> Any:
        """Cell-by-cell `>` a real scalar, lazily."""
        return self._binary_op(other, "__gt__")

    def __le__(self, other: Any) -> Any:
        """Cell-by-cell `<=` a real scalar, lazily."""
        return self._binary_op(other, "__le__")

    def __lt__(self, other: Any) -> Any:
        """Cell-by-cell `<` a real scalar, lazily."""
        return self._binary_op(other, "__lt__")

    def __neg__(self) -> Any:
        """Negate the values cell by cell, lazily."""
        return self._unary_op("__neg__")

    def __abs__(self) -> Any:
        """Absolute value cell by cell, lazily."""
        return self._unary_op("__abs__")

    def squeeze(self, dim: str | None = None) -> Any:
        """Drop length-one band dimensions, composing lazily (#1237).

        Pure band-axis reshape (no data touched), so it stays lazy: the dask array's size-1 band
        axes are squeezed and the band metadata narrowed to match :meth:`NetCDF.squeeze`. A
        multi-variable container squeezes eagerly at the boundary, warning once.
        """
        from pyramids.netcdf.engines.selection import _assert_band_dimension

        if len(self._current_records()) != 1:
            self._warn_materialize("squeeze")
            return self.compute().squeeze(dim)
        name, rec = next(iter(self._current_records().items()))
        names = list(rec.band_names)
        sizes = list(rec.array.shape[: len(names)])
        if dim is not None:
            _assert_band_dimension(self._source, dim, caller="squeeze")
            length = sizes[names.index(dim)]
            if length != 1:
                raise ValueError(
                    f"squeeze() drops a dimension of length one, and {dim!r} has "
                    f"length {length}. Select one step first with isel({dim}=[0])."
                )
            gone = {dim}
        else:
            gone = {n for n, size in zip(names, sizes) if size == 1}
        if not gone:
            return self
        axes = tuple(i for i, n in enumerate(names) if n in gone)
        new_arr = rec.array.squeeze(axis=axes)
        new_names = [n for n in names if n not in gone]
        new_vmap = {n: rec.values_map.get(n) for n in new_names}
        new = rec._replace(
            array=new_arr,
            band_names=new_names,
            values_map=new_vmap,
            dim_names=(*new_names, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(
            self._source, {name: new_arr}, {name: new.dim_names}, {name: new}
        )

    def transpose(self, *dims: Any) -> Any:
        """Reorder the band dimensions (spatial plane stays trailing), composing lazily (#1237).

        Pure band-axis permutation, so it stays lazy: `_transpose_order` resolves and validates the
        order (as :meth:`NetCDF.transpose`), then the dask array's band axes are permuted. A
        multi-variable container transposes eagerly at the boundary, warning once.
        """
        from pyramids.netcdf.engines.selection import _transpose_order

        if len(self._current_records()) != 1:
            self._warn_materialize("transpose")
            return self.compute().transpose(*dims)
        name, rec = next(iter(self._current_records().items()))
        order = _transpose_order(list(rec.band_names), dims)
        perm = [rec.band_names.index(n) for n in order]
        ndim = rec.array.ndim
        new_arr = rec.array.transpose([*perm, ndim - 2, ndim - 1])
        new_vmap = {n: rec.values_map.get(n) for n in order}
        new = rec._replace(
            array=new_arr,
            band_names=list(order),
            values_map=new_vmap,
            dim_names=(*order, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(
            self._source, {name: new_arr}, {name: new.dim_names}, {name: new}
        )

    def isel(self, *, drop: bool = False, **indexers: Any) -> Any:
        """Select bands by position along band dimensions, composing lazily (#1237).

        Pure band-axis indexing, so it stays lazy: each selector is resolved to positions (the same
        `_resolve_positional_indices` :meth:`NetCDF.isel` uses) and the dask array is indexed along
        that axis, the coordinates narrowed to match; `drop=True` squeezes the axes a scalar
        selector collapsed. A multi-variable container selects eagerly at the boundary, warning once.
        """
        from pyramids.netcdf.engines.selection import (
            _assert_band_dimension,
            _resolve_positional_indices,
        )

        if not indexers:
            raise ValueError(
                "isel() requires at least one keyword argument, e.g. isel(time=0)."
            )
        if len(self._current_records()) != 1:
            self._warn_materialize("isel")
            return self.compute().isel(drop=drop, **indexers)
        rec = next(iter(self._current_records().values()))
        resolved: list[tuple[str, list[int]]] = []
        scalar_dims: list[str] = []
        for dim_name, selector in indexers.items():
            _assert_band_dimension(self._source, dim_name, caller="isel")
            axis = rec.band_names.index(dim_name)
            size = int(rec.array.shape[axis])
            resolved.append(
                (dim_name, _resolve_positional_indices(selector, size, dim_name))
            )
            if not isinstance(selector, (slice, list, tuple)):
                scalar_dims.append(dim_name)
        result: Any = self
        for dim_name, dim_indices in resolved:
            result = result._subset_band(dim_name, dim_indices)
        if drop:
            for dim_name in scalar_dims:
                if (
                    dim_name
                    in next(iter(result._current_records().values())).band_names
                ):
                    result = result.squeeze(dim_name)
        return result

    def sel(
        self,
        *,
        method: str | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Select bands by coordinate value along band dimensions, composing lazily (#1237).

        The value twin of :meth:`isel`: each selector is resolved to positions against the source
        coordinates (the same `_resolve_one_dim` :meth:`NetCDF.sel` uses, honouring
        `method="nearest"` / `tolerance`), then the dask array is indexed — staying lazy. A
        multi-variable container selects eagerly at the boundary, warning once.
        """
        from pyramids.netcdf.engines.selection import _resolve_one_dim

        if not kwargs:
            raise ValueError(
                "sel() requires at least one keyword argument, e.g. sel(time=6)."
            )
        if method not in (None, "nearest"):
            raise ValueError(
                f"sel() method must be None (exact) or 'nearest', got {method!r}."
            )
        if tolerance is not None and method != "nearest":
            raise ValueError(
                "sel() tolerance= is only meaningful with method='nearest' — without it "
                "a label either matches exactly or does not match at all, and there is no "
                "distance for a tolerance to bound."
            )
        if len(self._current_records()) != 1:
            self._warn_materialize("sel")
            return self.compute().sel(method=method, tolerance=tolerance, **kwargs)
        resolved = [
            (
                dim_name,
                _resolve_one_dim(self._source, dim_name, selector, method, tolerance),
            )
            for dim_name, selector in kwargs.items()
        ]
        result: Any = self
        for dim_name, dim_indices in resolved:
            result = result._subset_band(dim_name, dim_indices)
        return result

    def _subset_band(self, dim_name: str, dim_indices: list[int]) -> LazyNetCDF:
        """Index the single variable's dask array along one band axis and narrow its coordinates.

        The lazy, dimension-preserving twin of `_subset_along_dim`: a plain dask index along the
        band axis (no flatten, no compute), keeping the dimension and relabelling its coordinates.
        The caller has checked there is exactly one variable.

        Returns:
            LazyNetCDF: The cube with that band axis narrowed to `dim_indices`, still lazy.
        """
        name, rec = next(iter(self._current_records().items()))
        axis = rec.band_names.index(dim_name)
        new_arr = rec.array[(slice(None),) * axis + (list(dim_indices),)]
        coords = rec.values_map.get(dim_name)
        new_vmap = dict(rec.values_map)
        new_vmap[dim_name] = (
            [coords[i] for i in dim_indices] if coords is not None else None
        )
        new = rec._replace(array=new_arr, values_map=new_vmap)
        return LazyNetCDF(
            self._source, {name: new_arr}, {name: new.dim_names}, {name: new}
        )

    def _compose_cellwise(self, method: str, *args: Any, **kwargs: Any) -> LazyNetCDF:
        """Run a cell-wise eager op (`clip`/`fillna`/`round`) per spatial block via map_blocks.

        These ops are element-wise and band-agnostic, and their shared gap/dtype machinery is not
        dask-safe (it mutates in place and calls `.all()`), so each numpy block is wrapped in a
        throwaway eager `Dataset` and run through the real method — bit-for-bit, shape-preserving.
        The output dtype and no-data value (data-independent for these ops) come from a 1x1 spatial
        probe. The caller has checked there is exactly one variable.

        Returns:
            LazyNetCDF: The cube with the cell-wise op applied, still lazy.
        """
        import dask.array as da

        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset

        name, rec = next(iter(self._current_records().items()))
        ndv = rec.no_data

        def _block(block: Any) -> tuple[Any, Any]:
            flat = np.asarray(block)
            shape = flat.shape
            ds = Dataset.from_array(
                flat.reshape(-1, shape[-2], shape[-1]),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, float(shape[-2])), cell_size=1.0, epsg=4326
                ),
                no_data_value=ndv,
            )
            out = getattr(ds, method)(*args, **kwargs)
            values = np.asarray(out.read_array(squeeze=True)).reshape(shape)
            return values, out.no_data_value[0]

        probe, out_ndv = _block(np.asarray(rec.array[..., :1, :1].compute()))
        new_arr = da.map_blocks(
            lambda block: _block(block)[0],
            rec.array,
            dtype=probe.dtype,
            meta=np.empty((0,) * rec.array.ndim, dtype=probe.dtype),
        )
        new = rec._replace(array=new_arr, no_data=out_ndv)
        return LazyNetCDF(
            self._source, {name: new_arr}, {name: new.dim_names}, {name: new}
        )

    def _extremum(
        self, dim: str, extreme: str, coordinate: bool, caller: str, skipna: bool
    ) -> Any:
        """Shared `argmin`/`argmax`/`idxmin`/`idxmax`: collapse `dim` to its extremum, lazily.

        The extremum locators use `np.argmin`/`argmax` (numpy), so they compose via map_blocks like
        `rank`; collapsing `dim` is handled by `_compose_mapblocks`'s `drop_axis`. A multi-variable
        container locates eagerly at the boundary, warning once.

        Returns:
            LazyNetCDF | NetCDF: A lazy cube for a pinned variable; an eager result otherwise.
        """
        from pyramids.netcdf.engines._along_dim import _Extremum

        if len(self._current_records()) != 1:
            self._warn_materialize(caller)
            return getattr(self.compute(), caller)(dim, skipna=skipna)
        op = _Extremum(
            extreme=extreme,
            coordinate=coordinate,
            skipna=bool(skipna),
            caller=caller,
        )
        return self._compose_mapblocks(op, dim)

    def _compose_mapblocks(self, op: Any, dim: str) -> LazyNetCDF:
        """Run an `_AlongDim` op's eager kernel per spatial block via `dask.array.map_blocks`.

        For ops whose kernel is numpy/scipy and does not dispatch on dask (`rank`, `interp`, `pad`,
        the extremum locators), the band axis is rechunked whole so each block holds the full band
        axis, and the op's `apply` runs on each `(whole bands, y-chunk, x-chunk)` numpy block
        through `override`. The result is lazy and bit-for-bit with the eager op. The output band
        layout (names, coords, no-data, and any dimension the op drops or resizes) is read from a
        1x1 spatial probe, which is data-independent for these ops. The caller has checked there is
        exactly one variable.

        Returns:
            LazyNetCDF: The cube with the op applied, still lazy.
        """
        import dask.array as da

        op.start()
        name, rec = next(iter(self._current_records().items()))
        nbd = len(rec.band_names)
        arr = rec.array.rechunk({i: -1 for i in range(nbd)})
        meta_in = (list(rec.band_names), dict(rec.values_map), rec.no_data)

        def _run(block: Any) -> Any:
            return op.apply(self._source, None, dim, override=(block, *meta_in)).values

        probe = op.apply(
            self._source,
            None,
            dim,
            override=(np.asarray(arr[..., :1, :1].compute()), *meta_in),
        )
        out_names = list(probe.band_names)
        dropped = [i for i, band in enumerate(rec.band_names) if band not in out_names]
        band_chunks = tuple((int(size),) for size in probe.values.shape[:-2])
        new_arr = da.map_blocks(
            _run,
            arr,
            dtype=probe.values.dtype,
            drop_axis=dropped or None,
            chunks=(*band_chunks, arr.chunks[-2], arr.chunks[-1]),
            meta=np.empty((0,) * (len(band_chunks) + 2), dtype=probe.values.dtype),
        )
        new = rec._replace(
            array=new_arr,
            band_names=out_names,
            values_map=dict(probe.values_map),
            no_data=probe.no_data,
            dim_names=(*out_names, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(
            self._source, {name: new_arr}, {name: new.dim_names}, {name: new}
        )

    def _compose_op(self, op: Any, dim: str) -> LazyNetCDF:
        """Run an `_AlongDim` op (one supporting `override`) on the single variable's dask array.

        Sets `op.materialize = False` so the op's `apply` returns a deferred dask array, and runs
        it on the cube's current array via `override`, for ops whose kernel keeps instance state
        (so a factored free function would be awkward). The caller has checked one variable.

        Returns:
            LazyNetCDF: The cube with the op applied, still lazy.
        """
        op.materialize = False
        name, rec = next(iter(self._current_records().items()))
        applied = op.apply(
            self._source,
            None,
            dim,
            override=(rec.array, rec.band_names, rec.values_map, rec.no_data),
        )
        new = rec._replace(
            array=applied.values,
            band_names=applied.band_names,
            values_map=applied.values_map,
            no_data=applied.no_data,
            dim_names=(*applied.band_names, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(
            self._source, {name: applied.values}, {name: new.dim_names}, {name: new}
        )

    def _compose_direct(self, kernel: Any, dim: str, **params: Any) -> LazyNetCDF:
        """Run a factored along-dim `kernel` on the single variable's dask array, deferring it.

        The kernel has the uniform signature
        `(nc, arr, band_names, values_map, no_data, dim, *, materialize, **params)` and returns
        `(array, band_names, values_map, no_data)`; called with `materialize=False` it keeps the
        result a `dask.array`, so the op composes with later lazy ops until :meth:`compute` (#1237).
        The caller has checked there is exactly one variable.

        Returns:
            LazyNetCDF: The cube with the op applied, still lazy.
        """
        name, rec = next(iter(self._current_records().items()))
        arr, band_names, values_map, no_data = kernel(
            self._source,
            rec.array,
            rec.band_names,
            rec.values_map,
            rec.no_data,
            dim,
            materialize=False,
            **params,
        )
        new = rec._replace(
            array=arr,
            band_names=band_names,
            values_map=values_map,
            no_data=no_data,
            dim_names=(*band_names, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(self._source, {name: arr}, {name: new.dim_names}, {name: new})

    def _compose_reduction(
        self,
        dim: str,
        how: str,
        *,
        skipna: bool,
        q: float | None,
        group_positions: list | None,
        resize: int | None,
        window_mean_coords: bool,
    ) -> LazyNetCDF:
        """Run a `_Reduction` kernel on the single variable's dask array, deferring it (#1237).

        Shared by :meth:`reduce` and :meth:`coarsen`; the caller has checked there is exactly one
        variable. The reduced array stays a `dask.array` (`materialize=False`), so the step composes
        with later lazy ops until :meth:`compute`.

        Returns:
            LazyNetCDF: The cube with the dimension reduced, still lazy.
        """
        from pyramids.netcdf.engines._along_dim import _reduced_array

        name, rec = next(iter(self._current_records().items()))
        arr, band_names, values_map, no_data = _reduced_array(
            self._source,
            None,
            dim,
            how,
            group_positions=group_positions,
            skipna=skipna,
            q=q,
            resize=resize,
            window_mean_coords=window_mean_coords,
            materialize=False,
            override=(rec.array, rec.band_names, rec.values_map, rec.no_data),
        )
        new = rec._replace(
            array=arr,
            band_names=band_names,
            values_map=values_map,
            no_data=no_data,
            dim_names=(*band_names, *rec.dim_names[-2:]),
        )
        return LazyNetCDF(self._source, {name: arr}, {name: new.dim_names}, {name: new})

    def _warn_materialize(self, name: str) -> None:
        """Emit the one-time boundary warning when an op falls back to the eager cube."""
        if not self._materialize_warned:
            warnings.warn(
                f"{name!r} on a lazy NetCDF cube is not composed lazily yet; it materialises the "
                f"cube and runs eagerly. Call .compute() for an explicit eager cube.",
                UserWarning,
                stacklevel=3,
            )
            self._materialize_warned = True

    def __getattr__(self, name: str) -> Any:
        """Delegate any eager operation to the **source** cube, warning once.

        Reached only for names this wrapper does not define. A leading-underscore name is refused
        outright (it is internal, not an operation). A public name absent from the eager cube also
        raises :class:`AttributeError` **without** warning, so an existence probe (`hasattr`, feature
        detection) neither warns spuriously nor burns the once-flag. The attribute is fetched exactly
        once (a sentinel default on a single `getattr`, not a `hasattr`-then-`getattr`), so a property
        getter on the source is not run twice.

        A public name that exists forwards to :data:`self._source` **directly**, aliasing the source
        (the `load` contract) rather than returning an independent :meth:`compute` copy: a functional
        op returns a new cube and leaves the source intact, but an in-place mutator reached this way
        would touch the source, so call :meth:`compute` first when you need an isolated eager cube.
        This is the v1 lazy/eager boundary (array-native ops do not yet compose lazily); it emits a
        one-time :class:`UserWarning`, and the once-flag is set only after `warn` returns, so a first
        access under warnings-as-error does not silence later boundary warnings.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # A transformed cube materialises its deferred arrays (the rebuilt eager result); an
        # untransformed view aliases its source, as before.
        target = self._rebuild() if self._records is not None else self._source
        attr = getattr(target, name, _MISSING)
        if attr is _MISSING:
            raise AttributeError(name)
        if not self._materialize_warned:
            warnings.warn(
                f"Accessing {name!r} on a lazy NetCDF cube materialises it onto the eager source "
                f"cube (aliasing it, like .load()); call .compute() for an independent eager cube.",
                UserWarning,
                stacklevel=2,
            )
            self._materialize_warned = True
        return attr

    def __repr__(self) -> str:
        """A short, dask-aware summary naming the variables and their chunking."""
        return f"LazyNetCDF(variables={self.variable_names}, chunks={self.chunks})"
