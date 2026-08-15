"""Grouped, validated option dataclasses for :func:`NetCDF.create_from_array`.

`create_from_array` used to take eighteen flat parameters. They are organised here into four
cohesive, frozen dataclasses so the call site reads as a handful of named groups instead of a long
positional list:

- :class:`GeoReference` — how the array maps to space (`geo` / `epsg`, or `top_left_corner` +
  `cell_size`).
- :class:`ExtraDimensions` — the non-spatial (time / level / …) dimensions of a 3-D+ array.
- :class:`Encoding` — on-disk write options (chunking, compression); only effective when a `path`
  is given.
- :class:`CFAttributes` — CF global attributes (`title`, `institution`, `source`, `history`).

Each is importable from the subpackage, e.g. ``from pyramids.netcdf import GeoReference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GeoTransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class GeoReference:
    """How the array is georeferenced in space.

    Provide either an affine `geo` transform directly, or a `top_left_corner` together with a
    `cell_size` from which a north-up `geo` is built. `epsg` is the coordinate reference system.

    Attributes:
        geo: Affine geotransform `(x_min, pixel_size, rotation, y_max, rotation, pixel_size)`.
            Takes precedence over `top_left_corner` / `cell_size` when given.
        epsg: EPSG code for the spatial reference. Defaults to 4326. `None` leaves the CRS
            unset (e.g. when carrying through a source variable that has no CRS).
        top_left_corner: `(x, y)` of the top-left corner, used with `cell_size` to build `geo`
            when `geo` is not supplied.
        cell_size: Pixel size, used with `top_left_corner` to build `geo`.
    """

    geo: GeoTransform | None = None
    epsg: str | int | None = 4326
    top_left_corner: tuple[float, float] | None = None
    cell_size: int | float | None = None

    def resolve_geotransform(self) -> GeoTransform:
        """Return the affine geotransform, building it from corner + cell size when needed.

        Returns:
            The 6-tuple geotransform — `geo` verbatim when provided, otherwise a north-up
            transform derived from `top_left_corner` and `cell_size`.

        Raises:
            ValueError: If neither `geo` nor both `top_left_corner` and `cell_size` are given.
        """
        if self.geo is not None:
            return self.geo
        if self.top_left_corner is not None and self.cell_size is not None:
            return (
                self.top_left_corner[0],
                self.cell_size,
                0,
                self.top_left_corner[1],
                0,
                -self.cell_size,
            )
        raise ValueError(
            "Either 'geo' or both 'top_left_corner' and 'cell_size' must be provided."
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
    """On-disk write options. Only effective when `create_from_array` is given a `path`.

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
