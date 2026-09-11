"""Cell engine.

Owns the Cell family of operations on a Dataset. Accessed as
``ds.cell``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.cell.<method>(...)`` are equivalent.
"""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, locate_values
from pandas import DataFrame
from pyproj import CRS

from pyramids.base.crs import crs_from_user_input, crs_spec, require_crs_spec
from pyramids.dataset.engines._base import _Engine
from pyramids.feature import FeatureCollection, create_points, create_polygon

if TYPE_CHECKING:
    from pyramids.dataset.dataset import (  # noqa: F401  (forward ref in _Engine["Dataset"])
        Dataset,
    )


# A degenerate geotransform -- a zero cell size, or a rotation that collapses
# the parallelogram -- describes cells with no extent. Answering 0.0 would make
# `domain_area` report no ground for a raster full of data.
_NO_EXTENT = (
    "the raster's geotransform gives its cells no area; check the cell size "
    "and rotation terms"
)

# Square metres per unit of area. The names are the ones a caller writes, not
# GDAL's or PROJ's spellings, because this is the surface a user types.
_AREA_UNITS: dict[str, float] = {
    "m2": 1.0,
    "km2": 1e6,
    "ha": 1e4,
}


def _area_scale(unit: str) -> float:
    """Square metres in one `unit`.

    Args:
        unit: One of `m2`, `km2`, `ha`. Matched after `strip().lower()`, so
            `KM2` and `" km2 "` name the same unit as `km2`.

    Returns:
        float: The divisor that turns square metres into `unit`.

    Raises:
        ValueError: `unit` is not one this package converts to, or is not a
            string at all -- `None` and `2` are refused the same way.
    """
    try:
        # Normalised first: `KM2` and `" km2"` are the same request as `km2`,
        # and refusing them buys nothing. `strip`/`lower` are attributes, so a
        # non-string argument still falls through to the refusal below.
        scale = _AREA_UNITS[unit.strip().lower()]
    except (KeyError, AttributeError):
        # `AttributeError` as well as `KeyError`: anything that is not a string
        # -- `None`, `2`, a list -- fails on `strip` before the lookup can miss
        # it, and leaking that would contradict the `ValueError` this method
        # documents. Normalising first is what makes `AttributeError` the way a
        # non-string arrives, rather than the `TypeError` an unhashable key
        # used to raise.
        raise ValueError(
            f"unknown area unit {unit!r}; expected one of "
            f"{', '.join(sorted(_AREA_UNITS))}"
        ) from None
    return scale


class Cell(_Engine["Dataset"]):
    """Cell-geometry operations on a Dataset.

    Owns the real implementations of `get_cell_coords`,
    `get_cell_polygons`, `get_cell_points`,
    `map_to_array_coordinates`, and `array_to_map_coordinates` after
    L-2 PR 2.1. `Dataset` exposes a same-named facade for each method
    that delegates to this collaborator, so `ds.get_cell_coords(...)`
    and `ds.cell.get_cell_coords(...)` are equivalent.
    """

    def get_cell_coords(
        self, location: str = "center", domain_only: bool = False
    ) -> np.typing.NDArray:
        """Get coordinates for the center/corner of cells inside the dataset domain.

        Returns the coordinates of the cell centers inside the domain (only the cells that
        do not have nodata value)

        Args:
            location (str):
                Location of the coordinates. Use `center` for the center of a cell, `corner` for the corner of the
                cell (top-left corner).
            domain_only (bool):
                True to exclude the cells out of the domain. Default is False. The domain is judged
                against band 0's stored values, where `no_data_value` lives, so a CF-packed band's
                gaps are excluded too.

        Returns:
            np.ndarray:
                `(N, 2)` array of `(x, y)` coordinates in row-major cell order: every cell, or only the
                domain cells when `domain_only=True`.

        Raises:
            ValueError: `location` is neither `center` nor `corner`.

        Examples:
            - Create `Dataset` consists of 1 bands, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> coords = dataset.get_cell_coords()
              >>> print(coords)
              [[ 0.025 -0.025]
               [ 0.075 -0.025]
               [ 0.125 -0.025]
               [ 0.025 -0.075]
               [ 0.075 -0.075]
               [ 0.125 -0.075]
               [ 0.025 -0.125]
               [ 0.075 -0.125]
               [ 0.125 -0.125]]

              ```

            - Get the coordinates of the top left corner of cells inside the domain.

              ```python
              >>> coords = dataset.get_cell_coords(location="corner")
              >>> print(coords)
              [[ 0.    0.  ]
               [ 0.05  0.  ]
               [ 0.1   0.  ]
               [ 0.   -0.05]
               [ 0.05 -0.05]
               [ 0.1  -0.05]
               [ 0.   -0.1 ]
               [ 0.05 -0.1 ]
               [ 0.1  -0.1 ]]

              ```
        """
        location = location.lower()
        if location not in ["center", "corner"]:
            raise ValueError(
                "The location parameter can have one of these values: 'center', 'corner', "
                f"but the value: {location} is given."
            )

        if location == "center":
            add_value = 0.5
        else:
            add_value = 0
        (
            x_init,
            cell_size_x,
            xy_span,
            y_init,
            yy_span,
            cell_size_y,
        ) = self._ds.geotransform

        if cell_size_x != cell_size_y:
            if np.abs(cell_size_x) != np.abs(cell_size_y):
                self._ds.logger.warning(
                    f"The given raster does not have a square cells, the cell size is "
                    f"{cell_size_x}*{cell_size_y} "
                )

        no_val = (
            self._ds.no_data_value[0]
            if self._ds.no_data_value[0] is not None
            else np.nan
        )
        # Stored counts: only *which* cells are in the domain matters here, and that
        # is decided against the stored sentinel. A physical read of a packed band
        # never contains it, so every gap came back as a cell.
        arr = self._ds.read_array(band=0, unpack=False)
        if domain_only and no_val not in arr:
            self._ds.logger.warning(
                "The no data value does not exist in the band, so all the cells will be considered, and the "
                "domain_only filter will not be applied."
            )

        mask_values: list[Any] | None = [no_val] if domain_only else None
        indices = get_indices2(arr, mask=mask_values)

        rows = np.asarray([i[0] for i in indices], dtype=float) + add_value
        cols = np.asarray([i[1] for i in indices], dtype=float) + add_value
        # Full affine, not axis-wise scaling. `xy_span` / `yy_span` (geotransform
        # [2] and [4]) are the rotation/skew terms; dropping them places every cell
        # of a rotated raster at the wrong location -- silently, because the result
        # still looks like a well-formed grid. They are zero for a north-up raster,
        # so this is identical to the previous arithmetic in the common case.
        # Vectorised over the index arrays rather than per cell (ARC-57).
        x = x_init + cell_size_x * cols + xy_span * rows
        y = y_init + yy_span * cols + cell_size_y * rows
        coords = np.column_stack((x, y))

        return coords

    def _attach_crs(self, gdf: GeoDataFrame) -> None:
        """Label `gdf` in place with the raster's CRS, or leave it unprojected.

        Uses the resolved CRS *spec* (the EPSG code when it is resolvable downstream,
        else the WKT), routed through `crs_from_user_input` rather than a bare `epsg=`:
        geopandas resolves a code with pyproj, which cannot look up a code only GDAL's
        PROJ database carries (issue #943). Passing the bare `_get_epsg()` code alone was
        `None` for a valid CRS with no EPSG authority (a custom projection or a proj4/WKT
        reprojection), which crashed `set_crs` and, before #893, silently mislabelled such
        cells as EPSG:4326 (issue #979). `crs_spec` is `None` only when the raster truly
        has no CRS -- then the frame is left unprojected.

        Args:
            gdf (GeoDataFrame): The frame to label in place.
        """
        crs = crs_spec(self._ds.epsg, self._ds.crs)
        if crs is not None:
            gdf.set_crs(crs_from_user_input(crs), inplace=True)

    def cell_area(self, unit: str = "m2") -> np.ndarray:
        """The ground area each cell covers, honouring the raster's own CRS.

        `cell_size` answers in the CRS's units, which for a geographic raster
        are degrees -- and a degree of longitude is 111 km at the equator and
        19 km at 80 degrees north. Anything asking "how much ground is this"
        therefore cannot use it, and `get_cell_polygons().area` does not help
        either: those polygons are in the same degrees, so every cell on the
        grid reports the identical area.

        Two cases, decided by the CRS:

        * **projected** -- the cell is a parallelogram in a linear unit, so its
          area is the absolute determinant of the geotransform's linear part,
          `|dx*dy - rx*ry|`, converted from the CRS's own linear unit. Constant
          across the raster, rotation included.
        * **geographic** -- the area is integrated in closed form over the
          CRS's own ellipsoid, not taken on a sphere of an assumed radius and
          not approximated by a geodesic polygon. A cell is bounded by
          parallels, which are not geodesics, so a geodesic routine bows its
          north and south edges poleward and biases every partial domain. Every
          cell in a row shares a latitude band and therefore an area, so the
          whole raster costs one vectorised pass over the row edges.

        Args:
            unit: `m2` (default), `km2` or `ha`. Case and surrounding
                whitespace are ignored, so `KM2` and `" km2 "` also work.

        Returns:
            np.ndarray: Area per cell, shaped like the band, and always a
            **read-only broadcast view** -- over one value per row for a
            geographic raster, over a single value for a projected one, rotated
            or not. Reading and arithmetic behave as usual (`values * areas`);
            assigning into it raises, so call `.copy()` for a writeable array.

            The view is what keeps a global 1-degree grid at 180 stored floats
            rather than 64 800, but only while it stays a view: `nbytes`
            reports the expanded size, and pickling, `np.save` and `reshape`
            all materialise the full array. `domain_area` sums it without
            doing so.

        Raises:
            CRSError: The raster carries neither an EPSG code nor a WKT, so
                nothing says what its coordinates measure. A `ValueError`
                subclass, so an `except ValueError` still catches it.
            ValueError: `unit` is not recognised, or the raster is geographic
                *and* rotated -- a case where cells in one row no longer share
                a latitude band, and which is better solved by warping to a
                north-up grid or a projected CRS than by spending a geodesic
                call on every cell. Also when the CRS is neither geographic nor
                projected, so its axes are not a ground plane -- a geocentric
                or engineering CRS, since pyproj reads `is_geographic` and
                `is_projected` through a compound CRS to its horizontal part;
                when a geographic CRS names no ellipsoid to integrate over, or
                names a prolate one; when a row lies entirely beyond a pole;
                and when the geotransform leaves the cells no extent -- a zero
                cell size, or a rotation that collapses the parallelogram.

        Examples:
            - A projected raster has one area for every cell, straight from the
              geotransform:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
                >>> raster = Dataset.from_array(np.ones((2, 2), "float32"), geo_ref=geo_ref)
                >>> raster.cell_area().tolist()
                [[900.0, 900.0], [900.0, 900.0]]

                ```
            - A geographic raster's cells shrink towards the pole, which is the
              whole reason this method exists:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(top_left_corner=(0.0, 90.0), cell_size=1.0, epsg=4326)
                >>> raster = Dataset.from_array(np.ones((90, 4), "float32"), geo_ref=geo_ref)
                >>> areas = raster.cell_area(unit="km2")
                >>> round(float(areas[89, 0]))     # the row touching the equator
                12308
                >>> round(float(areas[0, 0]))      # the row touching the pole
                109

                ```
            - Rotation is refused rather than answered approximately, because a
              row's cells no longer share a latitude band:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> geo_ref = GeoReference(geo=(0.0, 1.0, 0.2, 90.0, 0.2, -1.0), epsg=4326)
                >>> raster = Dataset.from_array(np.ones((4, 4), "float32"), geo_ref=geo_ref)
                >>> raster.cell_area()  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: a rotated geographic raster ... warp it to a north-up grid ...

                ```

        See Also:
            Analysis.domain_area: The area-weighted sibling of
                `count_domain_cells`, which sums this over a band's valid cells
                without materialising it.
        """
        scale = _area_scale(unit)
        geo = self._ds.geotransform
        # The same route `_attach_crs` takes, for the same reasons: the EPSG
        # code when it resolves and the WKT otherwise (#943), with `None`
        # meaning the raster truly has no CRS rather than an empty string
        # that every downstream constructor rejects opaquely (#979). Resolving
        # from `self._ds.crs` alone would let one raster answer through
        # `get_cell_polygons` and fail here. `require_crs_spec` raises the
        # project's `CRSError`, itself a `ValueError`, naming the fix.
        crs = crs_from_user_input(
            require_crs_spec(self._ds.epsg, self._ds.crs, "compute cell area")
        )
        if crs.is_geographic:
            if bool(geo[2]) or bool(geo[4]):
                raise ValueError(
                    "a rotated geographic raster has cells that do not share a "
                    "latitude band, so its area cannot be resolved per row; "
                    "warp it to a north-up grid or to a projected CRS first"
                )
            per_row = self._parallel_row_areas(crs) / scale
            # Checked before the broadcast, not after. `per_row > 0.0` on the
            # expanded view would allocate one byte per pixel -- 400 MB on a
            # 20000x20000 raster, and a `MemoryError` on the out-of-core grids
            # this method exists to serve -- to test `rows` distinct numbers.
            if not np.all(per_row > 0.0):
                raise ValueError(_NO_EXTENT)
            areas = np.broadcast_to(
                per_row[:, np.newaxis], (self._ds.rows, self._ds.columns)
            )
        elif crs.is_projected:
            # `unit_conversion_factor` is metres per CRS unit, so it squares.
            metres = crs.axis_info[0].unit_conversion_factor
            determinant = abs(geo[1] * geo[5] - geo[2] * geo[4])
            one = determinant * metres * metres / scale
            if one <= 0.0:
                raise ValueError(_NO_EXTENT)
            areas = np.broadcast_to(np.float64(one), (self._ds.rows, self._ds.columns))
        else:
            # Geocentric and engineering CRSs reach here. Their axes are not a
            # ground plane, so a determinant of the geotransform is not an area
            # of anything -- better to say so than to return a number that
            # looks like one. A compound CRS does not reach here: pyproj reads
            # `is_geographic` / `is_projected` through to its horizontal part,
            # so a DEM with a vertical datum takes the branch it should.
            raise ValueError(
                f"the raster's CRS is neither geographic nor projected "
                f"({crs.type_name}), so its cells have no ground area; "
                "reproject it to a projected or geographic CRS first"
            )
        return areas

    def _parallel_row_areas(self, crs: CRS) -> np.ndarray:
        """Square metres of one cell in each row, on the CRS's own ellipsoid.

        A raster cell is bounded by two **parallels** and two meridians, and a
        parallel is not a geodesic anywhere but the equator. Asking a geodesic
        polygon routine for the area therefore bows each north and south edge
        poleward, and the residual does not cancel within the cell: it biases
        every partial domain, by 2.6e-05 per cell at 1 degree and 3.7e-02 at
        30. It hides in a global total, where the bulges of vertically adjacent
        rows telescope away exactly, which is why an ungapped-globe check
        cannot see it at any cell size.

        So the area is integrated in closed form instead: `_zone_areas` gives
        the ellipsoidal area between each pair of parallels, and the answer is
        that times the longitude span. Vectorised over rows, exact at any cell
        size, and it needs no geodesic call at all.

        Args:
            crs: The raster's CRS, already resolved.

        Returns:
            np.ndarray: One area per row, in square metres, in row order.

        Raises:
            ValueError: The CRS's datum names no ellipsoid, so there is no
                figure of the earth to integrate over, or a row of the raster
                lies entirely beyond a pole. `_zone_areas` adds one more
                refusal this propagates: a prolate ellipsoid.
        """
        geod = crs.get_geod()
        if geod is None:
            # A geographic CRS whose datum names no ellipsoid: there is no
            # figure of the earth to integrate over, so there is no area.
            raise ValueError(
                "the raster's geographic CRS declares no ellipsoid, so its "
                "cells have no ground area; reproject it to one that does"
            )
        _, dx, _, top, _, dy = self._ds.geotransform
        # The geotransform speaks the CRS's angular unit, which is degrees for
        # every ordinary geographic CRS but grads for a few (EPSG:4807). The
        # factor converts to radians, so the latitudes have to travel through
        # it too rather than being handed to `deg2rad`.
        to_radians = crs.axis_info[0].unit_conversion_factor
        edges = (top + np.arange(self._ds.rows + 1) * dy) * to_radians
        span = abs(dx) * to_radians
        pole = np.pi / 2
        # A single edge past the pole is ordinary and handled by the clip in
        # `_zone_areas`: a cell-centred global grid (ERA5's, say) puts its
        # first edge half a cell beyond 90, and the half of that cell which
        # exists is exactly what the clipped integral returns. A row with
        # *both* edges outside is a different thing -- it describes ground that
        # is not on the ellipsoid at all, and clipping would silently hand it
        # back as a zero-area row for the caller to sum.
        above = (edges[:-1] > pole) & (edges[1:] > pole)
        below = (edges[:-1] < -pole) & (edges[1:] < -pole)
        off = above | below
        if off.any():
            degrees = np.degrees(edges)
            raise ValueError(
                f"the raster's latitude extent runs off the ellipsoid: it "
                f"spans {degrees[0]:.6g} to {degrees[-1]:.6g} degrees, so "
                f"{int(off.sum())} of its {self._ds.rows} rows lie entirely "
                "beyond a pole; crop or re-georeference it first"
            )
        return self._zone_areas(edges, geod.a, geod.f) * span

    @staticmethod
    def _zone_areas(edges: np.ndarray, a: float, f: float) -> np.ndarray:
        """Area between consecutive parallels, per radian of longitude.

        The ellipsoidal area element is
        `a^2 (1 - e^2) cos(phi) / (1 - e^2 sin^2 phi)^2`. Its antiderivative in
        latitude is

            `a^2 (1 - e^2) [ sin/(2(1 - e^2 sin^2)) + arctanh(e sin)/(2e) ]`,

        so a band's area is that evaluated at two parallels and subtracted.
        Subtracting it *as written* is the problem: the antiderivative is
        4.1e13 at the pole while the band against it at 1e-6 degrees covers
        6.2e-03 m2 per radian of longitude, and the difference of two nearly
        equal doubles keeps none of that. It reaches exactly `0.0` near the
        pole at 1e-7 degrees, which then trips the no-extent guard and refuses
        the raster.

        So the subtraction is done first, in closed form, and never evaluated:

        * `sin(u) - sin(l) = 2 cos((u+l)/2) sin((u-l)/2)`, exact for any gap;
        * the rational term differences to
          `(sin u - sin l)(1 + e^2 sin u sin l) / (2 (1 - e^2 sin^2 u)(1 - e^2 sin^2 l))`;
        * `arctanh x - arctanh y = arctanh((x - y)/(1 - xy))`.

        Every one carries the small quantity through as a factor, so the answer
        keeps full relative precision at any cell size.

        Args:
            edges: Row edge latitudes in **radians**, `rows + 1` of them.
            a: The ellipsoid's semi-major axis, in metres.
            f: Its flattening; `0.0` for a sphere.

        Returns:
            np.ndarray: One area per row, in square metres per radian of
            longitude, in row order and never negative -- `0.0` when a band's
            two edges coincide, which the caller refuses as a cell with no
            extent.

        Raises:
            ValueError: `f` is negative, describing a prolate figure that this
                closed form does not cover.
        """
        # Clipped before the sine, because past the pole the sine turns back
        # down and the integral would answer for the wrong latitude: an edge at
        # 100 degrees reads as 80 and yields a plausible 2.06e9 m2 for ground
        # that does not exist. Nothing here can produce a `nan` -- `|e sin| < 1`
        # always, so `arctanh` stays finite -- the hazard is a wrong number, not
        # an obvious one. `_parallel_row_areas` refuses a row that is entirely
        # outside; this keeps the routine honest for the half-cell that is not.
        latitude = np.clip(edges, -np.pi / 2, np.pi / 2)
        upper, lower = latitude[:-1], latitude[1:]
        sine_difference = (
            2.0 * np.cos((upper + lower) / 2.0) * np.sin((upper - lower) / 2.0)
        )
        eccentricity_squared = f * (2.0 - f)
        if f < 0.0:
            # Guarded separately from the sphere: a prolate figure gives a
            # negative `e^2`, whose `arctanh` form turns into an `arctan` one.
            # No geodetic datum is prolate, so refusing beats quietly handing
            # back the spherical answer, which is what `e^2 <= 0` used to do.
            raise ValueError(
                f"the CRS's ellipsoid is prolate (flattening {f}), which cell "
                "area is not derived for; reproject to an oblate or spherical "
                "datum first"
            )
        if eccentricity_squared <= 0.0:
            # A spherical datum -- every major weather model ships GRIB on one.
            # `<=` rather than `==` because a prolate figure, the only way this
            # goes negative, is already refused above: reaching here with a
            # non-positive `e^2` means it is exactly zero.
            # Both terms above tend to `(sin u - sin l)/2` as `e` vanishes, so
            # the general form's division by `e` is avoided rather than
            # approached.
            areas = a * a * sine_difference
        else:
            sine_upper, sine_lower = np.sin(upper), np.sin(lower)
            eccentricity = np.sqrt(eccentricity_squared)
            product = eccentricity_squared * sine_upper * sine_lower
            rational = (
                sine_difference
                * (1.0 + product)
                / (
                    2.0
                    * (1.0 - eccentricity_squared * sine_upper * sine_upper)
                    * (1.0 - eccentricity_squared * sine_lower * sine_lower)
                )
            )
            inverse = np.arctanh(eccentricity * sine_difference / (1.0 - product)) / (
                2.0 * eccentricity
            )
            areas = a * a * (1.0 - eccentricity_squared) * (rational + inverse)
        return np.asarray(np.abs(areas), dtype="float64")

    def get_cell_polygons(self, domain_only: bool = False) -> GeoDataFrame:
        """Get a polygon shapely geometry for the raster cells.

        Args:
            domain_only (bool):
                True to get the polygons of the cells inside the domain.

        Returns:
            GeoDataFrame:
                With two columns, geometry, and id. The frame is labelled with the
                raster's own CRS -- its EPSG code when that code is resolvable
                downstream, otherwise its WKT (so a custom authority-less projection, or
                a code only GDAL's PROJ database carries, is preserved rather than
                fabricated as EPSG:4326) -- or left unprojected (`crs=None`) when the
                raster has no CRS.

        Examples:
            - Create `Dataset` consists of 1 band, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_polygons()
              >>> print(gdf)  # doctest: +NORMALIZE_WHITESPACE
                                                     geometry  id
              0  POLYGON ((0 0, 0.05 0, 0.05 -0.05, 0 -0.05, 0 0))   0
              1  POLYGON ((0.05 0, 0.1 0, 0.1 -0.05, 0.05 -0.05...   1
              2  POLYGON ((0.1 0, 0.15 0, 0.15 -0.05, 0.1 -0.05...   2
              3  POLYGON ((0 -0.05, 0.05 -0.05, 0.05 -0.1, 0 -0...   3
              4  POLYGON ((0.05 -0.05, 0.1 -0.05, 0.1 -0.1, 0.0...   4
              5  POLYGON ((0.1 -0.05, 0.15 -0.05, 0.15 -0.1, 0....   5
              6  POLYGON ((0 -0.1, 0.05 -0.1, 0.05 -0.15, 0 -0....   6
              7  POLYGON ((0.05 -0.1, 0.1 -0.1, 0.1 -0.15, 0.05...   7
              8  POLYGON ((0.1 -0.1, 0.15 -0.1, 0.15 -0.15, 0.1...   8
              >>> fig, ax = dataset.plot()  # doctest: +SKIP
              >>> gdf.plot(ax=ax, facecolor='none', edgecolor="gray", linewidth=2)  # doctest: +SKIP
              <Axes: >

              ```

        ![get_cell_polygons](./../../_images/dataset/get_cell_polygons.png)
        """
        coords = self.get_cell_coords(location="corner", domain_only=domain_only)
        # Step one column and one row through the affine rather than offsetting both
        # axes by the pixel *width*. Using geotransform[1] for x and y made every
        # cell square: on a raster with 2 x 5 pixels the polygons came back 2 x 2.
        # Expressed as vector steps it also carries the rotation terms, so a skewed
        # raster yields the right parallelogram instead of an axis-aligned box.
        _, col_dx, row_dx, _, col_dy, row_dy = self._ds.geotransform
        x = np.zeros((coords.shape[0], 4))
        y = np.zeros((coords.shape[0], 4))
        x[:, 0] = coords[:, 0]
        y[:, 0] = coords[:, 1]
        x[:, 1] = x[:, 0] + col_dx
        y[:, 1] = y[:, 0] + col_dy
        x[:, 2] = x[:, 0] + col_dx + row_dx
        y[:, 2] = y[:, 0] + col_dy + row_dy
        x[:, 3] = x[:, 0] + row_dx
        y[:, 3] = y[:, 0] + row_dy

        # Stack once into (n_cells, 4, 2) and hand each cell's ring straight to
        # the polygon builder. The previous form transposed the corners through
        # four intermediate lists of tuples and then re-indexed them per cell,
        # allocating 4 * n_cells tuples before a single polygon was built.
        rings = np.stack((x, y), axis=-1)
        polygons = [create_polygon(ring) for ring in rings.tolist()]
        gdf = gpd.GeoDataFrame(geometry=polygons)
        self._attach_crs(gdf)
        gdf["id"] = gdf.index
        return gdf

    def get_cell_points(
        self, location: str = "center", domain_only: bool = False
    ) -> GeoDataFrame:
        """Get a point shapely geometry for the raster cells center point.

        Args:
            location (str):
                Location of the point, ["corner", "center"]. Default is "center".
            domain_only (bool):
                True to get the points of the cells inside the domain only.

        Returns:
            GeoDataFrame:
                With two columns, geometry, and id. The frame is labelled with the
                raster's own CRS -- its EPSG code when that code is resolvable
                downstream, otherwise its WKT (so a custom authority-less projection, or
                a code only GDAL's PROJ database carries, is preserved rather than
                fabricated as EPSG:4326) -- or left unprojected (`crs=None`) when the
                raster has no CRS.

        Examples:
            - Create `Dataset` consists of 1 band, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_points()
              >>> print(gdf)
                             geometry  id
              0  POINT (0.025 -0.025)   0
              1  POINT (0.075 -0.025)   1
              2  POINT (0.125 -0.025)   2
              3  POINT (0.025 -0.075)   3
              4  POINT (0.075 -0.075)   4
              5  POINT (0.125 -0.075)   5
              6  POINT (0.025 -0.125)   6
              7  POINT (0.075 -0.125)   7
              8  POINT (0.125 -0.125)   8
              >>> fig, ax = dataset.plot()  # doctest: +SKIP
              >>> gdf.plot(ax=ax, facecolor='black', linewidth=2)  # doctest: +SKIP
              <Axes: >

              ```

            ![get_cell_points](./../../_images/dataset/get_cell_points.png)

            - Get the coordinates of the top left corner of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_points(location="corner")
              >>> print(gdf)  # doctest: +NORMALIZE_WHITESPACE
                          geometry  id
              0         POINT (0 0)   0
              1      POINT (0.05 0)   1
              2       POINT (0.1 0)   2
              3     POINT (0 -0.05)   3
              4  POINT (0.05 -0.05)   4
              5   POINT (0.1 -0.05)   5
              6      POINT (0 -0.1)   6
              7   POINT (0.05 -0.1)   7
              8    POINT (0.1 -0.1)   8
              >>> fig, ax = dataset.plot()  # doctest: +SKIP
              >>> gdf.plot(ax=ax, facecolor='black', linewidth=4)  # doctest: +SKIP
              <Axes: >

              ```

            ![get_cell_points-corner](./../../_images/dataset/get_cell_points-corner.png)
        """
        coords = self.get_cell_coords(location=location, domain_only=domain_only)
        coords_tuples = list(zip(coords[:, 0], coords[:, 1]))
        points = create_points(coords_tuples)
        gdf = gpd.GeoDataFrame(geometry=points)
        self._attach_crs(gdf)
        gdf["id"] = gdf.index
        return gdf

    def map_to_array_coordinates(
        self,
        points: GeoDataFrame | FeatureCollection | DataFrame,
    ) -> np.typing.NDArray:
        """Convert coordinates of points to array indices.

        - map_to_array_coordinates locates a point with real coordinates (x, y) or (lon, lat) on the array by finding
            the cell indices (row, column) of the nearest cell in the raster.
        - The point coordinate system of the raster has to be projected to be able to calculate the distance.

        Args:
            points (GeoDataFrame | pandas.DataFrame | FeatureCollection):
                - GeoDataFrame: GeoDataFrame with POINT geometry.
                - DataFrame: DataFrame with x, y columns.

        Returns:
            np.ndarray:
                Array with shape (N, 2) containing the row and column indices in the array.

        Examples:
            - Create `Dataset` consisting of 2 bands, 10 rows, 10 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```
            - DataFrame with x, y columns:

              - We can give the function a DataFrame with x, y columns to array the coordinates of the points that are located within the dataset domain.

              ```python
              >>> points = pd.DataFrame({"x": [0.025, 0.175, 0.375], "y": [0.025, 0.225, 0.125]})
              >>> indices = dataset.map_to_array_coordinates(points)
              >>> print(indices)
              [[0 0]
               [0 3]
               [0 7]]

              ```
            - GeoDataFrame with POINT geometry:

              - We can give the function a GeoDataFrame with POINT geometry to array the coordinates of the points that locate within the dataset domain.

              ```python
              >>> from shapely.geometry import Point
              >>> from geopandas import GeoDataFrame
              >>> points = GeoDataFrame({"geometry": [Point(0.025, 0.025), Point(0.175, 0.225), Point(0.375, 0.125)]})
              >>> indices = dataset.map_to_array_coordinates(points)
              >>> print(indices)
              [[0 0]
               [0 3]
               [0 7]]

              ```
        """
        if isinstance(points, FeatureCollection):
            verts = points.with_coordinates()
            points = verts.loc[:, ["x", "y"]].values
        elif isinstance(points, GeoDataFrame):
            verts = FeatureCollection(points).with_coordinates()
            points = verts.loc[:, ["x", "y"]].values
        elif isinstance(points, DataFrame):
            # `any`, not `all`: a frame carrying only "x" is just as unusable as one
            # carrying neither, but `all(... not in ...)` only fires when *both* are
            # missing, so the half-formed case fell through to a KeyError below.
            if any(elem not in points.columns for elem in ["x", "y"]):
                raise ValueError(
                    "If the input is a DataFrame, it should have two columns x, and y"
                )
            points = points.loc[:, ["x", "y"]].values
        else:
            raise TypeError(
                "please check points input it should be GeoDataFrame/DataFrame/FeatureCollection - given"
                f" {type(points)}"
            )

        indices = locate_values(points, self._ds.x, self._ds.y)
        indices = indices[:, [1, 0]]
        return np.asarray(indices)

    def array_to_map_coordinates(
        self,
        rows_index: list[Number] | np.ndarray,
        column_index: list[Number] | np.ndarray,
        center: bool = False,
    ) -> tuple[list[Number], list[Number]]:
        """Convert array indices to map coordinates.

        array_to_map_coordinates converts the array indices (rows, cols) to real coordinates (x, y) or (lon, lat).

        Args:
            rows_index (List[Number] | np.ndarray):
                The row indices of the cells in the raster array. Any finite
                iterable of indices is accepted (list, tuple, ndarray, generator).
            column_index (List[Number] | np.ndarray):
                The column indices of the cells in the raster array. Any finite
                iterable of indices is accepted (list, tuple, ndarray, generator).
            center (bool):
                If True, the coordinates will be the center of the cell. Default is False.

        Returns:
            Tuple[List[Number], List[Number]]:
                A tuple of two equal-length lists, the x coordinates and the y
                coordinates of the cells. The elements are plain Python floats
                regardless of whether the inputs were lists or NumPy arrays.

        Raises:
            ValueError: ``rows_index`` and ``column_index`` have different
                lengths (the indices are paired element-wise).

        Examples:
            - Create `Dataset` consisting of 1 band, 10 rows, 10 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 3, size=(10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Now call the function with two lists of row and column indices:

              ```python
              >>> rows_index = [1, 3, 5]
              >>> column_index = [2, 4, 6]
              >>> coords = dataset.array_to_map_coordinates(rows_index, column_index)
              >>> print(coords) # doctest: +SKIP
              ([0.1, 0.2, 0.3], [-0.05, -0.15, -0.25])

              ```

        Notes:
            The conversion is affine-exact: it applies the full geotransform
            (per-axis pixel sizes and the rotation terms), so it is correct on
            non-square and rotated grids rather than assuming square pixels.

        See Also:
            - :meth:`map_to_array_coordinates`: The inverse direction
              (map coordinates to the nearest array indices).
            - :meth:`pyramids.dataset.abstract_dataset.RasterBase.xy`: The
              scalar/array affine companion used by the affine-style API.
        """
        # Materialise to 1-D float arrays first. Going through ``list`` accepts
        # any finite iterable of indices (lists, tuples, ndarrays, generators)
        # and gives one source of truth for the equal-length check.
        cols = np.asarray(list(column_index), dtype=float)
        rows = np.asarray(list(rows_index), dtype=float)
        if cols.shape != rows.shape:
            raise ValueError(
                "rows_index and column_index must have the same length, got "
                f"{rows.size} and {cols.size}."
            )
        # Apply the full affine geotransform (vectorised over the points) so the
        # y axis uses the pixel height (geotransform[5]) and the rotation terms
        # are honoured, instead of reusing the pixel width for both axes (issue
        # #505). The ``center`` half-pixel shift is a +0.5 offset on (col, row).
        (
            x_origin,
            pixel_width,
            row_rotation,
            y_origin,
            column_rotation,
            pixel_height,
        ) = self._ds.geotransform
        offset = 0.5 if center else 0.0
        cols = cols + offset
        rows = rows + offset
        x_coords = (x_origin + cols * pixel_width + rows * row_rotation).tolist()
        y_coords = (y_origin + cols * column_rotation + rows * pixel_height).tolist()

        return x_coords, y_coords
