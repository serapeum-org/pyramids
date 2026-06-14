"""First-class pixel-space window for raster reads and writes.

A :class:`Window` names a rectangular block of pixels by column/row offset and
size — **x-first**, matching GDAL's ``ReadAsArray(xoff, yoff, xsize, ysize)``
argument order. It is accepted everywhere pyramids takes a window
(:meth:`Dataset.read_array`, :meth:`Dataset.write_array`,
:meth:`Dataset.iter_blocks`) and replaces the two historical bare-sequence
forms, whose axis orders disagreed (the read list was x-first, the write tuple
y-first).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from osgeo import gdal


@dataclass(frozen=True)
class Window:
    """A rectangular pixel-space window: column/row offset plus size.

    The field order is **x-first** (column before row), matching GDAL's
    ``ReadAsArray(xoff, yoff, xsize, ysize)``.

    Args:
        col_off: Column (x) offset of the window's left edge, in pixels.
        row_off: Row (y) offset of the window's top edge, in pixels.
        cols: Window width in pixels (> 0).
        rows: Window height in pixels (> 0).

    Raises:
        TypeError: Any field is not an integer (GDAL would silently round a
            fractional offset to a neighbouring pixel).
        ValueError: ``cols`` or ``rows`` is not strictly positive.

    Examples:
        - Name a 2x3 block starting at column 4, row 1 and inspect it:
            ```python
            >>> from pyramids.dataset.window import Window
            >>> w = Window(col_off=4, row_off=1, cols=2, rows=3)
            >>> w.shape
            (3, 2)
            >>> w.to_read_args()
            (4, 1, 2, 3)

            ```
        - Windows are immutable value objects:
            ```python
            >>> from pyramids.dataset.window import Window
            >>> Window(0, 0, 2, 2) == Window(0, 0, 2, 2)
            True

            ```
        - A non-positive size is rejected:
            ```python
            >>> from pyramids.dataset.window import Window
            >>> try:
            ...     Window(0, 0, 0, 2)
            ... except ValueError as exc:
            ...     print("strictly positive" in str(exc))
            True

            ```
    """

    col_off: int
    row_off: int
    cols: int
    rows: int

    def __post_init__(self):
        """Validate the window geometry."""
        for name in ("col_off", "row_off", "cols", "rows"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral):
                raise TypeError(
                    f"window fields must be integers, got {name}={value!r}; "
                    f"GDAL silently rounds fractional offsets to a "
                    f"neighbouring pixel."
                )
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError(
                f"window size must be strictly positive, got "
                f"cols={self.cols}, rows={self.rows}."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` — the numpy shape of an array read from this window."""
        return (self.rows, self.cols)

    def to_read_args(self) -> tuple[int, int, int, int]:
        """Return ``(xoff, yoff, xsize, ysize)`` for ``ReadAsArray``/``WriteArray``.

        Returns:
            tuple[int, int, int, int]: GDAL-ordered window arguments.
        """
        return (self.col_off, self.row_off, self.cols, self.rows)

    @classmethod
    def from_bounds(
        cls,
        bbox: tuple[float, float, float, float],
        geotransform: tuple[float, float, float, float, float, float],
    ) -> Window:
        """Build the pixel window covering a map-space bounding box.

        Pixel offsets are floored and the far edge is ceiled, so the window
        always fully covers ``bbox`` (it may extend one pixel beyond a bbox
        edge that does not fall on a pixel boundary).

        Args:
            bbox: ``(min_x, min_y, max_x, max_y)`` in the raster's CRS.
            geotransform: The GDAL 6-tuple of the raster.

        Returns:
            Window: The covering pixel window.

        Raises:
            ValueError: ``bbox`` is inverted (min >= max on either axis), or
                ``geotransform`` is singular (not invertible).

        Examples:
            - A unit-cell grid with origin (0, 4): the bbox (1, 1, 3, 3)
              covers a 2x2 block starting at column 1, row 1:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> gt = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> Window.from_bounds((1.0, 1.0, 3.0, 3.0), gt)
                Window(col_off=1, row_off=1, cols=2, rows=2)

                ```
            - Round-trip through to_bounds returns the same box on aligned
              input:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> gt = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> Window.from_bounds((1.0, 1.0, 3.0, 3.0), gt).to_bounds(gt)
                (1.0, 1.0, 3.0, 3.0)

                ```
        """
        min_x, min_y, max_x, max_y = bbox
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"bbox must be (min_x, min_y, max_x, max_y), got {bbox}.")
        inverse = gdal.InvGeoTransform(geotransform)
        if inverse is None:
            raise ValueError(
                f"geotransform {geotransform} is singular and cannot be inverted."
            )
        # Project all four bbox corners: under a south-up (positive dy) or
        # rotated geotransform the min/max pixel coordinates do not come from
        # the (min_x, max_y) / (max_x, min_y) corners alone.
        corners = [
            gdal.ApplyGeoTransform(inverse, x, y)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
        ]
        cols_px = [corner[0] for corner in corners]
        rows_px = [corner[1] for corner in corners]
        # floor (not int(): it truncates toward zero) so bboxes extending
        # left of / above the raster origin resolve to negative offsets.
        col_off = int(math.floor(min(cols_px)))
        row_off = int(math.floor(min(rows_px)))
        cols = max(1, int(math.ceil(max(cols_px))) - col_off)
        rows = max(1, int(math.ceil(max(rows_px))) - row_off)
        return cls(col_off=col_off, row_off=row_off, cols=cols, rows=rows)

    def to_bounds(
        self,
        geotransform: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Return the map-space ``(min_x, min_y, max_x, max_y)`` of this window.

        Args:
            geotransform: The GDAL 6-tuple of the raster.

        Returns:
            tuple[float, float, float, float]: The window's bounding box.
        """
        left, top = gdal.ApplyGeoTransform(
            list(geotransform), float(self.col_off), float(self.row_off)
        )
        right, bottom = gdal.ApplyGeoTransform(
            list(geotransform),
            float(self.col_off + self.cols),
            float(self.row_off + self.rows),
        )
        return (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

    def intersection(self, other: Window) -> Window | None:
        """Return the overlapping window, or ``None`` when disjoint.

        Args:
            other: The window to intersect with.

        Returns:
            Window | None: The overlap, or ``None`` if the windows do not
                share any pixel.

        Examples:
            - Two overlapping blocks share a 1x1 corner:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window(0, 0, 2, 2).intersection(Window(1, 1, 2, 2))
                Window(col_off=1, row_off=1, cols=1, rows=1)

                ```
            - Disjoint blocks intersect to None:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window(0, 0, 2, 2).intersection(Window(5, 5, 2, 2)) is None
                True

                ```
        """
        col_off = max(self.col_off, other.col_off)
        row_off = max(self.row_off, other.row_off)
        col_end = min(self.col_off + self.cols, other.col_off + other.cols)
        row_end = min(self.row_off + self.rows, other.row_off + other.rows)
        result: Window | None = None
        if col_end > col_off and row_end > row_off:
            result = Window(col_off, row_off, col_end - col_off, row_end - row_off)
        return result

    def union(self, other: Window) -> Window:
        """Return the smallest window containing both windows.

        Args:
            other: The window to merge with.

        Returns:
            Window: The bounding window of the pair.

        Examples:
            - The union of two corner blocks spans the enclosing rectangle:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window(0, 0, 2, 2).union(Window(3, 3, 2, 2))
                Window(col_off=0, row_off=0, cols=5, rows=5)

                ```
        """
        col_off = min(self.col_off, other.col_off)
        row_off = min(self.row_off, other.row_off)
        col_end = max(self.col_off + self.cols, other.col_off + other.cols)
        row_end = max(self.row_off + self.rows, other.row_off + other.rows)
        return Window(col_off, row_off, col_end - col_off, row_end - row_off)

    @classmethod
    def rounded(
        cls,
        col_off: float,
        row_off: float,
        cols: float,
        rows: float,
    ) -> Window:
        """Snap possibly-fractional pixel coordinates to an integer ``Window``.

        Offsets are floored and sizes ceiled, so the resulting window always
        **covers** the fractional region. (A :class:`Window` itself is
        integer-only; this is the constructor for when upstream math produced
        fractional pixel coordinates.)

        Args:
            col_off: Fractional column offset.
            row_off: Fractional row offset.
            cols: Fractional width.
            rows: Fractional height.

        Returns:
            Window: the covering integer window.

        Examples:
            - Floor the offsets, ceil the sizes:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window.rounded(1.4, 2.6, 9.9, 3.1)
                Window(col_off=1, row_off=2, cols=10, rows=4)

                ```
        """
        return cls(
            math.floor(col_off),
            math.floor(row_off),
            math.ceil(cols),
            math.ceil(rows),
        )

    def crop(self, rows: int, cols: int) -> Window | None:
        """Clamp the window to a raster's ``(rows, cols)`` pixel extent.

        Args:
            rows: The raster's height in pixels.
            cols: The raster's width in pixels.

        Returns:
            Window | None: the part of this window inside the extent, or
                ``None`` when the window lies entirely outside it.

        Examples:
            - An oversized window is clamped to the raster:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window(0, 0, 100, 100).crop(rows=8, cols=8)
                Window(col_off=0, row_off=0, cols=8, rows=8)

                ```
            - A window fully outside the extent clamps to None:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> Window(20, 20, 5, 5).crop(rows=8, cols=8) is None
                True

                ```
        """
        return self.intersection(Window(0, 0, cols, rows))

    def todict(self) -> dict[str, int]:
        """Return the window as a ``{col_off, row_off, cols, rows}`` dict.

        Returns:
            dict[str, int]: the four fields, suitable for JSON / round-trip.

        Examples:
            - Serialise and round-trip through :meth:`from_dict`:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> d = Window(4, 1, 2, 3).todict()
                >>> sorted(d.items())
                [('col_off', 4), ('cols', 2), ('row_off', 1), ('rows', 3)]
                >>> Window.from_dict(d)
                Window(col_off=4, row_off=1, cols=2, rows=3)

                ```
        """
        return {
            "col_off": self.col_off,
            "row_off": self.row_off,
            "cols": self.cols,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Window:
        """Build a ``Window`` from a ``{col_off, row_off, cols, rows}`` dict.

        Args:
            data: A mapping with the four window fields.

        Returns:
            Window: the reconstructed window.
        """
        return cls(
            col_off=data["col_off"],
            row_off=data["row_off"],
            cols=data["cols"],
            rows=data["rows"],
        )

    def __iter__(self):
        """Iterate ``(col_off, row_off, cols, rows)`` so ``tuple(window)`` works.

        Yields:
            int: the four fields in GDAL (x-first) order.
        """
        yield from (self.col_off, self.row_off, self.cols, self.rows)

    def transform(
        self,
        geotransform: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """Return the geotransform of *this window* (its own top-left origin).

        The pixel size and rotation terms are unchanged; only the origin (indices
        0 and 3) shift to the window's top-left corner — the geotransform you
        write a cropped sub-raster with.

        Args:
            geotransform: The parent raster's GDAL 6-tuple.

        Returns:
            tuple: the window's geotransform.

        Examples:
            - The origin shifts to the window; the cell size is preserved:
                ```python
                >>> from pyramids.dataset.window import Window
                >>> gt = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> Window(2, 1, 3, 3).transform(gt)
                (2.0, 1.0, 0.0, 3.0, 0.0, -1.0)

                ```
        """
        origin_x, origin_y = gdal.ApplyGeoTransform(
            list(geotransform), float(self.col_off), float(self.row_off)
        )
        return (
            origin_x,
            geotransform[1],
            geotransform[2],
            origin_y,
            geotransform[4],
            geotransform[5],
        )
