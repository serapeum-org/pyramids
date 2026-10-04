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
