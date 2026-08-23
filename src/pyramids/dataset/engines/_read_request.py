"""The read-option request object for ``Dataset.read_array``.

``read_array`` accepts a dozen loosely-coupled options (``chunks``, ``out_shape``,
``boundless``, ``threadsafe``, ``masked``, ``fill_value``, ``resampling`` …) whose
*combinations* are what decide which read path runs and which pairings are
rejected. Grouping them into one frozen value object lets the whole
option-compatibility matrix live in a single ``__post_init__`` instead of being
inlined at the head of ``read_array``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ReadRequest:
    """A validated bundle of ``read_array`` options plus the resolved read target.

    ``__post_init__`` enforces the *option-combination* invariants that hold
    regardless of which read path runs (the pairings that are never legal). It is
    built from the raw options *before* the ``bbox``/``window`` are resolved, so
    these guards keep firing ahead of window resolution exactly as they did inline.
    The per-path preconditions that depend on the resolved ``window`` (e.g.
    ``chunks=`` with a ``window=``) are enforced by each
    :class:`~pyramids.dataset.engines._read_strategies.ReadStrategy`, which receives
    the resolved window separately, so this stays the cross-cutting matrix only.

    Attributes:
        band: The 0-based band index to read, or ``None`` for all bands.
        chunks: Dask chunk spec; non-``None`` selects the lazy path.
        lock: Dask read lock (lazy path only).
        out_shape: ``(rows, cols)`` decimated output shape, or ``None``.
        resampling: Resampling name; only meaningful together with ``out_shape``.
        boundless: Whether to pad reads that extend past the raster edge.
        fill_value: Pad value for boundless reads.
        masked: Whether to wrap the result as a masked array.
        scaled: Whether to apply each band's GDAL scale/offset
            (``raw * scale + offset``) to the read result.
        threadsafe: Whether to read through a private per-thread handle.

    Examples:
        - A valid full eager read of band 0 passes the matrix and keeps its fields:
            ```python
            >>> from pyramids.dataset.engines._read_request import ReadRequest
            >>> req = ReadRequest(
            ...     band=0, chunks=None, lock=None, out_shape=None,
            ...     resampling="nearest", boundless=False, fill_value=None,
            ...     masked=False, scaled=False, threadsafe=False,
            ... )
            >>> req.band
            0

            ```
        - An illegal option pairing (a fill value without a boundless read) is
          rejected at construction:
            ```python
            >>> from pyramids.dataset.engines._read_request import ReadRequest
            >>> ReadRequest(
            ...     band=0, chunks=None, lock=None, out_shape=None,
            ...     resampling="nearest", boundless=False, fill_value=5.0,
            ...     masked=False, scaled=False, threadsafe=False,
            ... )
            Traceback (most recent call last):
            ValueError: read_array(fill_value=...) only applies to boundless reads; pass boundless=True as well.

            ```
    """

    band: int | None
    chunks: int | tuple | dict | str | None
    lock: Any
    out_shape: tuple[int, int] | None
    resampling: str
    boundless: bool
    fill_value: float | None
    masked: bool
    scaled: bool
    threadsafe: bool

    def __post_init__(self) -> None:
        """Reject the option pairings that are never legal (see class docstring)."""
        if self.fill_value is not None and not self.boundless:
            raise ValueError(
                "read_array(fill_value=...) only applies to boundless reads; "
                "pass boundless=True as well."
            )
        if self.boundless and self.chunks is not None:
            raise ValueError(
                "read_array(chunks=..., boundless=True) is not supported; "
                "boundless fills apply to eager windowed reads only."
            )
        if self.boundless and self.out_shape is not None:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "boundless=True; decimated boundless reads are not combined "
                "yet. Read boundless at native resolution and decimate the "
                "result yourself."
            )
        if self.boundless and self.threadsafe:
            raise NotImplementedError(
                "read_array(boundless=True) is not supported together with "
                "threadsafe=True; the boundless read uses the shared handle, "
                "defeating the per-thread isolation. Read boundless without "
                "threadsafe, or pad the result yourself."
            )
        if self.out_shape is not None and self.threadsafe:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "threadsafe=True; the decimated read uses the shared handle, "
                "defeating the per-thread isolation. Read decimated without "
                "threadsafe, or decimate a threadsafe full read yourself."
            )
        if (
            self.out_shape is None
            and isinstance(self.resampling, str)
            and (self.resampling.strip().lower() != "nearest")
        ):
            raise ValueError(
                "read_array(resampling=...) only applies to out_shape reads; "
                "pass out_shape=(rows, cols) as well."
            )
