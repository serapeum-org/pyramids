"""Affine-style geotransform value object.

GDAL exposes a raster's georeferencing as a bare 6-tuple
``(x_origin, pixel_width, row_rotation, y_origin, column_rotation, pixel_height)``.
:class:`GeoTransform` wraps that tuple in a named, algebra-capable object —
``transform * (col, row)`` maps pixel space to map space the way users of
affine libraries expect — without adding a third-party dependency.
"""

from __future__ import annotations

from typing import NamedTuple

from osgeo import gdal


class GeoTransform(NamedTuple):
    """A GDAL geotransform as a named, algebra-capable value object.

    Field order matches the GDAL 6-tuple, so ``GeoTransform(*ds.geotransform)``
    and ``tuple(transform)`` round-trip losslessly.

    Attributes:
        x_origin: X coordinate of the top-left corner of the top-left pixel.
        pixel_width: Pixel size along x (the cell width).
        row_rotation: Row rotation term (0 for north-up rasters).
        y_origin: Y coordinate of the top-left corner of the top-left pixel.
        column_rotation: Column rotation term (0 for north-up rasters).
        pixel_height: Pixel size along y — **negative** for north-up rasters.

    Examples:
        - Build from a dataset's tuple and apply to a pixel coordinate:
            ```python
            >>> from pyramids.dataset.transform import GeoTransform
            >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
            >>> gt * (0, 0)
            (0.0, 4.0)
            >>> gt * (2, 1)
            (2.0, 3.0)

            ```
        - The inverse maps map space back to pixel space:
            ```python
            >>> from pyramids.dataset.transform import GeoTransform
            >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
            >>> gt.inverse * (2.0, 3.0)
            (2.0, 1.0)

            ```
        - Round-trip with the bare GDAL tuple:
            ```python
            >>> from pyramids.dataset.transform import GeoTransform
            >>> tuple(GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
            (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

            ```
    """

    x_origin: float
    pixel_width: float
    row_rotation: float
    y_origin: float
    column_rotation: float
    pixel_height: float

    def __mul__(self, col_row: tuple[float, float]) -> tuple[float, float]:  # type: ignore[override]
        """Apply the transform to a ``(col, row)`` pixel coordinate.

        Args:
            col_row: ``(column, row)`` pixel coordinate (fractions allowed —
                e.g. ``(col + 0.5, row + 0.5)`` for the cell centre).

        Returns:
            tuple[float, float]: The ``(x, y)`` map coordinate of the pixel's
                reference corner (or the supplied fractional position).
        """
        col, row = col_row
        x = self.x_origin + col * self.pixel_width + row * self.row_rotation
        y = self.y_origin + col * self.column_rotation + row * self.pixel_height
        return (x, y)

    @property
    def inverse(self) -> "GeoTransform":
        """The inverted transform, mapping ``(x, y)`` map space to ``(col, row)``.

        Returns:
            GeoTransform: The inverse, such that ``gt.inverse * (gt * (c, r))``
                returns ``(c, r)``.

        Raises:
            ValueError: The transform is singular and cannot be inverted.
        """
        inverted = gdal.InvGeoTransform(list(self))
        if inverted is None:
            raise ValueError(f"geotransform {tuple(self)} is not invertible.")
        return GeoTransform(*inverted)

    @classmethod
    def from_bounds(
        cls,
        bbox: tuple[float, float, float, float],
        rows: int,
        cols: int,
    ) -> "GeoTransform":
        """Build the north-up transform fitting ``bbox`` to a grid shape.

        Args:
            bbox: ``(min_x, min_y, max_x, max_y)`` map-space bounds.
            rows: Number of rows of the target grid.
            cols: Number of columns of the target grid.

        Returns:
            GeoTransform: North-up transform whose grid exactly spans the box.

        Raises:
            ValueError: ``bbox`` is inverted or ``rows``/``cols`` not positive.

        Examples:
            - A 4x4 grid over the unit-origin box:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform.from_bounds((0.0, 0.0, 4.0, 4.0), rows=4, cols=4)
                GeoTransform(x_origin=0.0, pixel_width=1.0, row_rotation=0.0, \
y_origin=4.0, column_rotation=0.0, pixel_height=-1.0)

                ```
        """
        min_x, min_y, max_x, max_y = bbox
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"bbox must be (min_x, min_y, max_x, max_y), got {bbox}.")
        if rows <= 0 or cols <= 0:
            raise ValueError(f"rows/cols must be positive, got rows={rows}, cols={cols}.")
        return cls(
            x_origin=float(min_x),
            pixel_width=(max_x - min_x) / cols,
            row_rotation=0.0,
            y_origin=float(max_y),
            column_rotation=0.0,
            pixel_height=-(max_y - min_y) / rows,
        )
