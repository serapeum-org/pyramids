"""Read-path strategies for ``Dataset.read_array``.

``read_array`` picks exactly one read path from the resolved options: lazy
(``chunks``), decimated (``out_shape``), boundless, thread-safe, or the default
eager read. Each path is a :class:`ReadStrategy` that (a) declares when it
:meth:`matches` a :class:`~pyramids.dataset.engines._read_request.ReadRequest`,
(b) enforces its own per-path preconditions, (c) performs the read against the IO
engine, and (d) names the array backend it produces. The dispatch collapses to:
pick the first matching strategy, read, record the backend.

The per-path preconditions live here (rather than on ``ReadRequest``) because they
depend on the *resolved* ``window`` — the value ``read_array`` computes from
``bbox``/``window`` and a polygon-to-pixel conversion just before dispatch and
passes to :meth:`ReadStrategy.read` alongside the request. The strategy order in
:data:`READ_STRATEGIES` reproduces the original ``if/elif`` ladder exactly, so the
same input still raises the same error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from geopandas.geodataframe import GeoDataFrame

from pyramids.dataset.engines._read_request import ReadRequest
from pyramids.dataset.engines._validate import validate_band_index

if TYPE_CHECKING:
    from pyramids.dataset.engines.io import IO


class ReadStrategy(ABC):
    """One read path: when it applies, how it reads, and what backend it yields.

    Attributes:
        backend: The array backend a successful read on this path produces —
            ``"numpy"`` for eager paths, ``"dask"`` for the lazy path. Recorded on
            the dataset after the read so ``Dataset`` reports the right backend.

    Examples:
        - Selecting the path for a chunked (lazy) request picks the dask backend:
            ```python
            >>> from pyramids.dataset.engines._read_request import ReadRequest
            >>> from pyramids.dataset.engines._read_strategies import READ_STRATEGIES
            >>> req = ReadRequest(
            ...     band=0, chunks=4, lock=None, out_shape=None,
            ...     resampling="nearest", boundless=False, fill_value=None,
            ...     masked=False, scaled=False, threadsafe=False,
            ... )
            >>> next(s for s in READ_STRATEGIES if s.matches(req)).backend
            'dask'

            ```
        - A plain request falls through to the eager (numpy) path:
            ```python
            >>> from pyramids.dataset.engines._read_request import ReadRequest
            >>> from pyramids.dataset.engines._read_strategies import READ_STRATEGIES
            >>> req = ReadRequest(
            ...     band=0, chunks=None, lock=None, out_shape=None,
            ...     resampling="nearest", boundless=False, fill_value=None,
            ...     masked=False, scaled=False, threadsafe=False,
            ... )
            >>> next(s for s in READ_STRATEGIES if s.matches(req)).backend
            'numpy'

            ```
    """

    backend: str

    @abstractmethod
    def matches(self, req: ReadRequest) -> bool:
        """Return whether this strategy handles ``req``'s option combination."""

    @abstractmethod
    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Validate this path's preconditions and read through ``io``.

        Args:
            io: The IO engine bound to the dataset being read.
            req: The validated option bundle selecting this path.
            window: The resolved read window — a pixel ``Window``, a
                ``GeoDataFrame`` (an unconverted geometry window handed to the
                boundless path), a ``[col_off, row_off, cols, rows]`` list, or
                ``None`` for a full read.
        """


class LazyRead(ReadStrategy):
    """Dask-backed lazy read, selected by ``chunks=``."""

    backend = "dask"

    def matches(self, req: ReadRequest) -> bool:
        """Match when a dask ``chunks`` spec was given."""
        return req.chunks is not None

    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Reject eager-only pairings, then build the lazy array."""
        if window is not None:
            raise ValueError(
                "read_array(chunks=..., window=...) is not supported; "
                "read lazily and slice the resulting dask array instead."
            )
        if req.out_shape is not None:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "chunks=; decimate eagerly, or coarsen the dask array."
            )
        if req.masked:
            raise NotImplementedError(
                "read_array(masked=True) is not supported together with "
                "chunks=; read eagerly, or mask the dask array yourself."
            )
        # ``matches`` only selected this path because a chunk spec was given.
        assert req.chunks is not None
        return io._lazy_read_array(
            band=req.band, chunks=req.chunks, lock=req.lock, threadsafe=req.threadsafe
        )


class DecimatedRead(ReadStrategy):
    """Overview-decimated read, selected by ``out_shape=``."""

    backend = "numpy"

    def matches(self, req: ReadRequest) -> bool:
        """Match when a decimated ``out_shape`` was given."""
        return req.out_shape is not None

    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Reject masked decimation, then read at the requested shape."""
        if req.masked:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "masked=True; decimation and masking are not combined yet. "
                "Read decimated without masked, or mask the result yourself."
            )
        # ``matches`` only selected this path because an out_shape was given.
        assert req.out_shape is not None
        return io._decimated_read(req.band, window, req.out_shape, req.resampling)


class BoundlessRead(ReadStrategy):
    """Edge-padded windowed read, selected by ``boundless=True``."""

    backend = "numpy"

    def matches(self, req: ReadRequest) -> bool:
        """Match when boundless padding was requested."""
        return req.boundless

    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Require a pixel window, reject masked boundless, then pad-read."""
        if req.masked:
            raise NotImplementedError(
                "read_array(boundless=True) is not supported together "
                "with masked=True; boundless fills and masking are not "
                "combined yet. Read boundless without masked, or mask the "
                "result yourself."
            )
        if window is None:
            raise ValueError(
                "read_array(boundless=True) requires a window; a full read "
                "cannot extend past the raster."
            )
        if isinstance(window, GeoDataFrame):
            raise ValueError(
                "boundless reads need a pixel window (Window or "
                "[col_off, row_off, cols, rows] list); geometry windows "
                "are clipped by definition."
            )
        return io._boundless_read(req.band, window, req.fill_value)


class ThreadsafeRead(ReadStrategy):
    """Eager read via a private per-thread handle, selected by ``threadsafe=True``."""

    backend = "numpy"

    def matches(self, req: ReadRequest) -> bool:
        """Match when a per-thread isolated read was requested."""
        return req.threadsafe

    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Reject masked thread-safe reads, then read via the per-thread handle."""
        if req.masked:
            raise NotImplementedError(
                "read_array(threadsafe=True) is not supported together "
                "with masked=True; the mask band would be read from the "
                "shared handle, defeating the per-thread isolation. Read "
                "masked without threadsafe, or mask the result yourself."
            )
        return io._threadsafe_eager_read(band=req.band, window=window)


class EagerRead(ReadStrategy):
    """The default eager read on the shared handle; matches everything else."""

    backend = "numpy"

    def matches(self, req: ReadRequest) -> bool:
        """Always match — this is the fallback path."""
        return True

    def read(self, io: IO, req: ReadRequest, window: Any) -> Any:
        """Read all bands or a single band directly, optionally masking the result."""
        band = req.band
        arr: Any
        # The shared handle is used directly here; the captured cloud config is
        # already installed by read_array's @under_gdal_env decorator.
        if band is None and io._ds.band_count > 1:
            if window is None:
                arr = np.ones(
                    (io._ds.band_count, io._ds.rows, io._ds.columns),
                    dtype=io._ds.numpy_dtype[0],
                )
                for i in range(io._ds.band_count):
                    arr[i, :, :] = io._ds._raster.GetRasterBand(i + 1).ReadAsArray()
            else:
                # ``window`` here is a resolved pixel window (``Window`` or a
                # ``[col_off, row_off, cols, rows]`` list — any geometry window was
                # converted to one before dispatch). Stack per-band block reads so
                # ``_read_block`` applies the identical window to every band without
                # re-parsing its dimensions here.
                arr = np.stack(
                    [io._read_block(i, window) for i in range(io._ds.band_count)],
                    axis=0,
                )
        else:
            validate_band_index(band, io._ds.band_count)
            if band is None:
                band = 0
            if window is None:
                arr = io._ds._iloc(band).ReadAsArray()
            else:
                arr = io._read_block(band, window)
        if req.masked:
            arr = io._to_masked(arr, band, window=window)
        return arr


READ_STRATEGIES: tuple[ReadStrategy, ...] = (
    LazyRead(),
    DecimatedRead(),
    BoundlessRead(),
    ThreadsafeRead(),
    EagerRead(),
)
"""The read paths in selection order (first :meth:`~ReadStrategy.matches` wins).

The order reproduces ``read_array``'s original ``if chunks / elif out_shape / elif
boundless / elif threadsafe / else`` ladder, so multi-option inputs still resolve
to the same path and raise the same precondition error.
"""
