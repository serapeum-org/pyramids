"""How an array maps to space — the georeferencing value object.

:class:`GeoReference` groups the four fields every raster constructor needs to
place an array on the earth: an affine ``geo`` transform and an ``epsg`` code,
or a ``top_left_corner`` plus ``cell_size`` from which a north-up transform is
built.

It lives in ``base`` rather than in a subpackage because it is shared
vocabulary: :class:`~pyramids.dataset.Dataset` and
:class:`~pyramids.netcdf.NetCDF` constructors both take it, and ``base`` is the
layer both build on. It is re-exported from ``pyramids.dataset``,
``pyramids.netcdf`` and ``pyramids.netcdf.array_options``, so every historical
import path keeps resolving.
"""

from __future__ import annotations

from dataclasses import dataclass

# The structural 6-tuple, not `pyramids.dataset.transform.GeoTransform`. That
# NamedTuple is a richer value object built on top of this shape; annotating
# with the plain tuple keeps both it and a bare `(x, dx, 0, y, 0, -dy)` literal
# assignable, which is what callers actually pass. Importing it here would also
# invert the layering `base` exists to keep straight.
GeoTransformTuple = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class GeoReference:
    """How the array is georeferenced in space.

    Provide either an affine `geo` transform directly, or a `top_left_corner` together with a
    `cell_size` from which a north-up `geo` is built. `epsg` is the coordinate reference system.

    Attributes:
        geo: Affine geotransform `(x_min, pixel_width, row_rotation, y_max, column_rotation,
            pixel_height)`, where `pixel_height` is negative for the usual north-up raster.
            Takes precedence over `top_left_corner` / `cell_size` when given.
        epsg: EPSG code for the spatial reference. Defaults to 4326. `None` leaves the CRS
            unset (e.g. when carrying through a source variable that has no CRS).
        top_left_corner: `(x, y)` of the top-left corner, used with `cell_size` to build `geo`
            when `geo` is not supplied.
        cell_size: Pixel size, used with `top_left_corner` to build `geo`.

    Examples:
        - An affine transform with an explicit CRS:
            ```python
            >>> from pyramids.base.georeference import GeoReference
            >>> ref = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
            >>> ref.resolve_geotransform()
            (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
            >>> ref.epsg
            4326

            ```

        - A corner and a cell size, from which a north-up transform is built:
            ```python
            >>> from pyramids.base.georeference import GeoReference
            >>> GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0).resolve_geotransform()
            (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)

            ```

        - The value object is frozen, so a variant is made with `dataclasses.replace`:
            ```python
            >>> from dataclasses import replace
            >>> from pyramids.base.georeference import GeoReference
            >>> ref = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
            >>> replace(ref, epsg=3857).epsg
            3857
            >>> ref.epsg
            4326

            ```
    """

    geo: GeoTransformTuple | None = None
    epsg: str | int | None = 4326
    top_left_corner: tuple[float, float] | None = None
    cell_size: int | float | None = None

    def resolve_geotransform(self) -> GeoTransformTuple:
        """Return the affine geotransform, building it from corner + cell size when needed.

        Returns:
            The 6-tuple geotransform — `geo` verbatim when provided, otherwise a north-up
            transform derived from `top_left_corner` and `cell_size`. Every term of a derived
            transform is a `float`, including the two zero rotation terms, so the result always
            matches the `GeoTransformTuple` shape rather than mixing `int` and `float`.

        Raises:
            ValueError: If neither `geo` nor both `top_left_corner` and `cell_size` are given —
                including a half-filled pair, where only one of `top_left_corner` / `cell_size`
                is supplied.

        Examples:
            - An explicit `geo` is returned unchanged:
                ```python
                >>> from pyramids.base.georeference import GeoReference
                >>> GeoReference(geo=(10.0, 0.5, 0.0, 50.0, 0.0, -0.5)).resolve_geotransform()
                (10.0, 0.5, 0.0, 50.0, 0.0, -0.5)

                ```

            - A corner plus an integer cell size still yields six floats:
                ```python
                >>> from pyramids.base.georeference import GeoReference
                >>> geo = GeoReference(top_left_corner=(0, 3), cell_size=1).resolve_geotransform()
                >>> geo
                (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
                >>> [type(term).__name__ for term in geo]
                ['float', 'float', 'float', 'float', 'float', 'float']

                ```

            - Half a pair is a mistake, not a request for the origin:
                ```python
                >>> from pyramids.base.georeference import GeoReference
                >>> GeoReference(top_left_corner=(10.0, 50.0)).resolve_geotransform()
                Traceback (most recent call last):
                ValueError: Either 'geo' or both 'top_left_corner' and 'cell_size' must be provided.

                ```

        See Also:
            - :class:`GeoReference`: The value object this method resolves.
        """
        if self.geo is not None:
            geo = self.geo
        elif self.top_left_corner is not None and self.cell_size is not None:
            # Every element is built as a float, including the two rotation
            # terms. Literal `0` ints satisfied the old untyped return but not
            # the `GeoTransformTuple` annotation this type acquired when it
            # moved to `base`, and they leaked into reprs as `(0.0, 1.0, 0, ...)`.
            geo = (
                float(self.top_left_corner[0]),
                float(self.cell_size),
                0.0,
                float(self.top_left_corner[1]),
                0.0,
                -float(self.cell_size),
            )
        else:
            raise ValueError(
                "Either 'geo' or both 'top_left_corner' and 'cell_size' must be "
                "provided."
            )
        return geo


__all__ = ["GeoReference", "GeoTransformTuple"]
