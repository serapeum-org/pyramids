"""Grouped, validated option dataclasses for :func:`NetCDF.from_array`.

`from_array` used to take eighteen flat parameters. They are organised here into four
cohesive, frozen dataclasses so the call site reads as a handful of named groups instead of a long
positional list:

- :class:`~pyramids.base.georeference.GeoReference` — how the array maps to space. Defined in
  ``base`` (shared with the raster constructors) and re-exported here.
- :class:`ExtraDimensions` — the non-spatial (time / level / …) dimensions of a 3-D+ array.
- :class:`Encoding` — on-disk write options (chunking, compression); only effective when a `path`
  is given.
- :class:`CFAttributes` — CF global attributes (`title`, `institution`, `source`, `history`).

Each is importable from the subpackage, e.g. ``from pyramids.netcdf import GeoReference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# `GeoReference` moved to `base` — it is shared vocabulary between the raster and
# netCDF constructors, not a netCDF-only writer option. Re-exported here so every
# historical `from pyramids.netcdf.array_options import GeoReference` keeps working.
from pyramids.base.georeference import (  # noqa: F401
    GeoReference,
)


@dataclass(frozen=True)
class ExtraDimensions:
    """The non-spatial dimensions of a 3-D+ array.

    Use the single-dimension `name` / `values` pair for a 3-D `(extra, rows, cols)` array, or the
    ordered `dims` list of `(name, values)` pairs for a 4-D+ array. The two forms are mutually
    exclusive.

    Attributes:
        name: Name of the non-spatial dimension for a 3-D array (e.g. `"time"`, `"level"`,
            `"depth"`). Ignored for 2-D arrays. Defaults to `"time"`.
        values: Coordinate values for that dimension; length must equal `arr.shape[0]`. `None`
            uses integer indices `[0, 1, ..., size - 1]`. Mutually exclusive with `dims`.
        dims: Ordered list of `(dim_name, values)` pairs describing every non-spatial dimension of
            a 4-D+ array, in storage order. `len(dims)` must equal `arr.ndim - 2`. Each `values`
            is a list of length `arr.shape[i]` or `None`. Mutually exclusive with `name` /
            `values`.
    """

    name: str = "time"
    values: list | None = None
    dims: list[tuple[str, list | None]] | None = None


@dataclass(frozen=True)
class Encoding:
    """On-disk write options. Only effective when `from_array` is given a `path`.

    Attributes:
        chunk_sizes: Chunk sizes for the data variable, matching the array dimensions (e.g.
            `(1, 256, 256)` for a 3-D array). `None` uses GDAL default chunking.
        compression: Compression algorithm name (`"DEFLATE"`, `"ZSTD"`, …). `None` means no
            compression.
        compression_level: Compression level (e.g. 1-9 for DEFLATE). `None` uses the GDAL default.
    """

    chunk_sizes: tuple | list | None = None
    compression: str | None = None
    compression_level: int | None = None


@dataclass(frozen=True)
class CFAttributes:
    """CF global attributes written to the dataset.

    Attributes:
        title: Short description of the dataset.
        institution: Where the data was produced.
        source: How the data was produced.
        history: Audit trail of processing steps.
    """

    title: str | None = None
    institution: str | None = None
    source: str | None = None
    history: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the provided (non-`None`) attributes as a `{name: value}` mapping."""
        return {
            key: value
            for key, value in (
                ("title", self.title),
                ("institution", self.institution),
                ("source", self.source),
                ("history", self.history),
            )
            if value is not None
        }
