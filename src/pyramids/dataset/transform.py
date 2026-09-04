"""Affine-style geotransform value object.

GDAL exposes a raster's georeferencing as a bare 6-tuple
``(x_origin, pixel_width, row_rotation, y_origin, column_rotation, pixel_height)``.
:class:`GeoTransform` wraps that tuple in a named, algebra-capable object —
``transform * (col, row)`` maps pixel space to map space the way users of
affine libraries expect — without adding a third-party dependency.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
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

        Raises:
            TypeError: ``col_row`` is not a two-element ``(col, row)`` pair —
                in particular, ``transform * n`` tuple repetition is not
                supported.
        """
        try:
            col, row = col_row
            if isinstance(col, str) or isinstance(row, str):
                # float() would coerce numeric strings; the contract is numeric
                # (col, row) — reject strings rather than silently accept "2".
                raise TypeError("(col, row) must be numeric, not str")
            col, row = float(col), float(row)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"GeoTransform multiplication expects a (col, row) pair, got {col_row!r}."
            ) from error
        x = self.x_origin + col * self.pixel_width + row * self.row_rotation
        y = self.y_origin + col * self.column_rotation + row * self.pixel_height
        return (x, y)

    def __rmul__(self, other: object) -> tuple[float, float]:  # type: ignore[override]
        """Reject reflected multiplication such as ``2 * transform``.

        Plain tuples implement ``n * t`` as sequence repetition, which for a
        transform would silently build a meaningless 12-element tuple; the
        reflected operator is therefore disabled.

        Args:
            other: The left operand of ``other * transform``.

        Raises:
            TypeError: Always — write ``transform * (col, row)`` instead.
        """
        raise TypeError(
            f"unsupported operand for *: {other!r} * GeoTransform; "
            "write transform * (col, row) instead."
        )

    @property
    def inverse(self) -> GeoTransform:
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

    @property
    def is_axis_aligned(self) -> bool:
        """True when the grid has no rotation, so rows are y and columns are x.

        Both rotation terms are zero on an axis-aligned grid, which is what lets
        a caller divide by the pixel size instead of inverting the full affine.

        Returns:
            bool: True when neither rotation term is set.

        Examples:
            - A plain north-up grid is axis-aligned:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0).is_axis_aligned
                True

                ```
            - Any shear term makes it not:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.5, 4.0, 0.0, -1.0).is_axis_aligned
                False

                ```
        """
        return not self.row_rotation and not self.column_rotation

    def apply(
        self,
        cols: np.typing.ArrayLike,
        rows: np.typing.ArrayLike,
        *,
        center: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map pixel coordinates to map coordinates, element-wise.

        The array form of ``transform * (col, row)``. Broadcasts, so it serves
        both a single pixel and whole coordinate arrays, and does not consume
        its inputs.

        Args:
            cols: Column coordinate(s), fractions allowed.
            rows: Row coordinate(s), fractions allowed.
            center: Offset by half a pixel, giving cell centres rather than
                the top-left corner of each cell.

        Returns:
            tuple[np.ndarray, np.ndarray]: The `(x, y)` map coordinates.

        Examples:
            - The top-left corner, and the centre of the same cell:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> [float(v) for v in gt.apply(0, 0)[0]]
                [0.0]
                >>> [float(v) for v in gt.apply(0, 0, center=True)[0]]
                [0.5]

                ```
            - Several pixels at once:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> xs, ys = gt.apply([0, 1, 2], [0, 0, 0])
                >>> [float(v) for v in xs]
                [0.0, 1.0, 2.0]

                ```
        """
        cols_arr = np.atleast_1d(np.asarray(cols, dtype=float))
        rows_arr = np.atleast_1d(np.asarray(rows, dtype=float))
        shift = 0.5 if center else 0.0
        xs = (
            self.x_origin
            + (cols_arr + shift) * self.pixel_width
            + (rows_arr + shift) * self.row_rotation
        )
        ys = (
            self.y_origin
            + (cols_arr + shift) * self.column_rotation
            + (rows_arr + shift) * self.pixel_height
        )
        return xs, ys

    def invert(
        self, x: np.typing.ArrayLike, y: np.typing.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map map coordinates back to fractional pixel coordinates.

        The array counterpart of :attr:`inverse`, applied element-wise. The
        result is fractional; a caller wanting array indices floors it.

        Args:
            x: Map x coordinate(s).
            y: Map y coordinate(s).

        Returns:
            tuple[np.ndarray, np.ndarray]: The fractional `(col, row)`.

        Raises:
            ValueError: The transform is singular and cannot be inverted.

        Examples:
            - Invert a point back to its pixel:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> cols, rows = gt.invert(2.0, 3.0)
                >>> (float(cols[0]), float(rows[0]))
                (2.0, 1.0)

                ```
            - Round-trips with :meth:`apply`:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.5, 4.0, 0.25, -1.0)
                >>> xs, ys = gt.apply([3], [2])
                >>> [round(float(v), 9) for v in gt.invert(xs, ys)[0]]
                [3.0]

                ```
        """
        inverse = self.inverse
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        cols = (
            inverse.x_origin
            + x_arr * inverse.pixel_width
            + y_arr * inverse.row_rotation
        )
        rows = (
            inverse.y_origin
            + x_arr * inverse.column_rotation
            + y_arr * inverse.pixel_height
        )
        return cols, rows

    def x_axis(self, columns: int) -> np.ndarray:
        """The 1-D array of x coordinates at each column's cell centre.

        Uses the **signed** pixel width, so a west-to-east grid ascends and a
        mirrored one descends. Note this is not the vectorised form of
        `self * (col + 0.5, 0)` on a rotated grid: it ignores `row_rotation`, so
        it describes the axis only when :attr:`is_axis_aligned` holds. On a
        rotated grid there is no single x per column to return.

        Args:
            columns: Number of columns.

        Returns:
            np.ndarray: `columns` x coordinates, at cell centres.

        Examples:
            - Cell centres of a unit grid start half a cell in:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0).x_axis(4).tolist()
                [0.5, 1.5, 2.5, 3.5]

                ```
            - The sign of the pixel width is honoured:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(10.0, -2.0, 0.0, 0.0, 0.0, -1.0).x_axis(3).tolist()
                [9.0, 7.0, 5.0]

                ```
        """
        return self.x_origin + (np.arange(columns, dtype="float64") + 0.5) * (
            self.pixel_width
        )

    def y_axis(self, rows: int) -> np.ndarray:
        """The 1-D array of y coordinates at each row's cell centre.

        Uses the **signed** pixel height, so a north-up grid (negative height)
        descends and a south-up one ascends. As with :meth:`x_axis`, this
        describes the axis only when :attr:`is_axis_aligned` holds.

        Args:
            rows: Number of rows.

        Returns:
            np.ndarray: `rows` y coordinates, at cell centres.

        Examples:
            - A north-up grid descends from the top:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0).y_axis(4).tolist()
                [3.5, 2.5, 1.5, 0.5]

                ```
            - A south-up grid ascends:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 0.0, 0.0, 1.0).y_axis(3).tolist()
                [0.5, 1.5, 2.5]

                ```
        """
        return self.y_origin + (np.arange(rows, dtype="float64") + 0.5) * (
            self.pixel_height
        )

    def extent(self, columns: int, rows: int) -> tuple[float, float, float, float]:
        """The normalised map-space bounding box of a grid this size.

        The inverse of :meth:`from_bounds`, and the answer to "where is this
        raster". All four corners are projected and then reduced, rather than
        the top-left and bottom-right pair alone: on a south-up grid (positive
        `pixel_height`) the pair arrives already swapped, and on a rotated one
        the extreme x and y do not come from that diagonal at all. Taking the
        pair on trust yields an inverted or under-sized box in both cases.

        Args:
            columns: Grid width in pixels.
            rows: Grid height in pixels.

        Returns:
            tuple[float, float, float, float]: `(min_x, min_y, max_x, max_y)`,
                always ordered min-before-max on each axis.

        Examples:
            - A north-up grid, four cells on a side:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0).extent(4, 4)
                (0.0, 0.0, 4.0, 4.0)

                ```
            - A south-up grid covers the same box, not an inverted one:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 0.0, 0.0, 1.0).extent(4, 4)
                (0.0, 0.0, 4.0, 4.0)

                ```
            - Round-trips with `from_bounds` on an axis-aligned grid:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> box = (10.0, 20.0, 30.0, 50.0)
                >>> grid = GeoTransform.from_bounds(box, rows=30, cols=20)
                >>> grid.extent(columns=20, rows=30) == box
                True

                ```
        """
        corners_x, corners_y = self.apply(
            [0, columns, 0, columns],
            [0, 0, rows, rows],
        )
        return (
            float(corners_x.min()),
            float(corners_y.min()),
            float(corners_x.max()),
            float(corners_y.max()),
        )

    def scaled(self, x_factor: float, y_factor: float) -> GeoTransform:
        """The transform for a grid whose cells are larger by the given factors.

        Scales all four resolution and rotation terms, as GDAL's
        `GDALOverviewDataset` does: leaving the rotation terms unscaled shears
        every pixel but the origin on a skewed grid. The origin is fixed, since
        a decimated grid still starts at the same corner.

        The axis pairing is the part that is easy to get wrong: `row_rotation`
        is the y-per-column term and so follows the **row** factor, while
        `column_rotation` is the x-per-row term and follows the **column**
        factor.

        Args:
            x_factor: How much wider each cell becomes along x (columns).
            y_factor: How much taller each cell becomes along y (rows).

        Returns:
            GeoTransform: The scaled transform, origin unchanged.

        Examples:
            - Halve the resolution of a north-up grid:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0).scaled(2, 2)
                GeoTransform(x_origin=0.0, pixel_width=2.0, row_rotation=0.0, y_origin=4.0, column_rotation=0.0, pixel_height=-2.0)

                ```
            - A rotated grid keeps its shear, scaled on the matching axis:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> rotated = GeoTransform(0.0, 1.0, 0.5, 4.0, 0.25, -1.0)
                >>> rotated.scaled(2, 4).row_rotation
                2.0
                >>> rotated.scaled(2, 4).column_rotation
                0.5

                ```
        """
        return GeoTransform(
            self.x_origin,
            self.pixel_width * x_factor,
            self.row_rotation * y_factor,
            self.y_origin,
            self.column_rotation * x_factor,
            self.pixel_height * y_factor,
        )

    def rescaled_to(
        self, from_shape: tuple[int, int], to_shape: tuple[int, int]
    ) -> GeoTransform:
        """The transform for the same extent resampled to a different shape.

        A `to_shape` equal to `from_shape` returns this transform unchanged; a
        decimated shape (fewer, larger cells over the same extent) scales
        through :meth:`scaled` by the row and column decimation factors.

        Args:
            from_shape: The `(rows, columns)` this transform describes.
            to_shape: The `(rows, columns)` of the target grid.

        Returns:
            GeoTransform: The transform for `to_shape` over the same extent.

        Examples:
            - Decimating by two doubles the cell size:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> gt.rescaled_to((4, 4), (2, 2)).pixel_width
                2.0

                ```
            - The same shape is a no-op:
                ```python
                >>> from pyramids.dataset.transform import GeoTransform
                >>> gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
                >>> gt.rescaled_to((4, 4), (4, 4)) == gt
                True

                ```
        """
        from_rows, from_columns = from_shape
        to_rows, to_columns = to_shape
        if (to_rows, to_columns) == (from_rows, from_columns):
            return self
        return self.scaled(from_columns / to_columns, from_rows / to_rows)

    @classmethod
    def from_bounds(
        cls,
        bbox: tuple[float, float, float, float],
        rows: int,
        cols: int,
    ) -> GeoTransform:
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
            raise ValueError(
                f"rows/cols must be positive, got rows={rows}, cols={cols}."
            )
        return cls(
            x_origin=float(min_x),
            pixel_width=(max_x - min_x) / cols,
            row_rotation=0.0,
            y_origin=float(max_y),
            column_rotation=0.0,
            pixel_height=-(max_y - min_y) / rows,
        )
