"""Lazy, dask-backed :class:`~pyramids.netcdf.NetCDF` cube — the lazy twin of an eager cube.

Mirrors the vector-side :class:`~pyramids.feature.LazyFeatureCollection` for the raster cube
(issue #1229). A :class:`LazyNetCDF` holds one :class:`dask.array.Array` per gridded variable —
read lazily from the store through :func:`pyramids.netcdf._lazy.build_lazy_array` — alongside the
eager cube it was built from, which is both the source of its geo / band metadata and the eager
twin its lifecycle materialises back to.

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
documented next step (see ``planning/xarray/lazy-cube-design.md`` §4).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pyramids.base._utils import import_dask

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.netcdf.netcdf import NetCDF

_DASK_MISSING = "The lazy NetCDF cube needs dask; install the `lazy` extra."


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
    ) -> None:
        """Wrap per-variable dask arrays alongside the eager cube they came from.

        Args:
            source: The eager cube this view was built from — the source of its metadata and the
                twin :meth:`compute` returns.
            arrays: One dask array per gridded variable, laid out `(*band_sizes, rows, cols)`.
            dim_names: The dimension names per variable, outermost first, i.e.
                `(*band_dim_names, row_name, col_name)` — the keys :attr:`chunks` reports.
        """
        self._source = source
        self._arrays = dict(arrays)
        self._dim_names = dict(dim_names)
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
        return self._source.copy()

    def load(self, **kwargs: Any) -> NetCDF:
        """Materialise in place and return the eager cube — the in-place form of :meth:`compute`.

        Matches xarray's `load`, which realises in place and returns the same object: this returns the
        **source cube itself** (not a copy), so it aliases the cube `.chunk()` was called on. Use
        :meth:`compute` when you need an independent result. The `**kwargs` are accepted for symmetry
        and ignored.

        Returns:
            NetCDF: The source cube (the eager twin), realised in place.
        """
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
        return LazyNetCDF(self._source, arrays, self._dim_names)

    def __getattr__(self, name: str) -> Any:
        """Delegate any eager operation to the materialised cube, warning once.

        Reached only for names this wrapper does not define. A leading-underscore name is refused
        outright (it is internal, not an operation); anything else materialises the cube via
        :meth:`compute`, emits a one-time :class:`UserWarning`, and forwards the attribute to the
        eager twin — the v1 lazy/eager boundary (array-native ops do not yet compose lazily).
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if not self._materialize_warned:
            self._materialize_warned = True
            warnings.warn(
                f"Accessing {name!r} on a lazy NetCDF cube materialises it to an eager cube; "
                f"call .compute() explicitly to make the boundary clear.",
                UserWarning,
                stacklevel=2,
            )
        return getattr(self.compute(), name)

    def __repr__(self) -> str:
        """A short, dask-aware summary naming the variables and their chunking."""
        return f"LazyNetCDF(variables={self.variable_names}, chunks={self.chunks})"
