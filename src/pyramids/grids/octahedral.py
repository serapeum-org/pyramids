"""Adapt an octahedral reduced-Gaussian field to a regular :class:`~pyramids.dataset.Dataset`.

Octahedral reduced-Gaussian grids (ECMWF ``O`` grids) store values as a single ragged
sequence of points whose latitude/longitude coordinates are known per point. pyramids
already interpolates scattered points to a raster with ``gdal.Grid``
(:func:`pyramids.dataset.ops.interpolate.grid_points`); this module wraps the ragged
points in a :class:`~geopandas.GeoDataFrame` and reuses that path. No new
third-party dependencies.
"""

from __future__ import annotations

import numpy as np
from geopandas import GeoDataFrame, points_from_xy

from pyramids.dataset.dataset import Dataset
from pyramids.dataset.ops.interpolate import grid_points


def from_octahedral(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    *,
    cell_size: float,
    algorithm: str = "nearest",
    epsg: int = 4326,
) -> Dataset:
    """Regrid an octahedral reduced-Gaussian field onto a regular-grid :class:`Dataset`.

    The per-point ``lats``/``lons``/``values`` triples are wrapped in a point
    :class:`~geopandas.GeoDataFrame` and interpolated with ``gdal.Grid`` via
    :func:`~pyramids.dataset.ops.interpolate.grid_points`.

    Args:
        lats: 1-D array of point latitudes.
        lons: 1-D array of point longitudes, same length as ``lats``.
        values: 1-D array of field values, same length as ``lats``.
        cell_size: Output pixel size in the target CRS units.
        algorithm: A ``gdal.Grid`` algorithm string (e.g. ``"nearest"``,
            ``"invdist:power=2.0:smoothing=0.0"``, ``"linear"``).
        epsg: Output EPSG code.

    Returns:
        A single-band :class:`~pyramids.dataset.Dataset` of the interpolated surface.

    Raises:
        ValueError: ``lats``, ``lons`` and ``values`` are not 1-D arrays of equal
            length.

    Examples:
        - Grid four corner observations with nearest-neighbour and inspect the result:
            ```python
            >>> import numpy as np
            >>> from pyramids.grids import from_octahedral
            >>> lats = np.array([0.0, 0.0, 5.0, 5.0])
            >>> lons = np.array([0.0, 5.0, 0.0, 5.0])
            >>> values = np.array([1.0, 2.0, 3.0, 4.0])
            >>> ds = from_octahedral(lats, lons, values, cell_size=1.0, algorithm="nearest")
            >>> (ds.rows, ds.columns, ds.band_count)
            (5, 5, 1)

            ```
        - Arrays of unequal length are rejected:
            ```python
            >>> import numpy as np
            >>> from pyramids.grids import from_octahedral
            >>> try:
            ...     from_octahedral(np.zeros(4), np.zeros(3), np.zeros(4), cell_size=1.0)
            ... except ValueError as exc:
            ...     print("equal length" in str(exc))
            True

            ```

    See Also:
        - :func:`pyramids.grids.from_orca`: regrid curvilinear ``(ny, nx)`` fields.
        - :func:`pyramids.dataset.ops.interpolate.grid_points`: the scattered-point
          interpolation this adapter delegates to.
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if not (lats.size == lons.size == values.size):
        raise ValueError(
            "lats, lons and values must have equal length; got "
            f"{lats.size}, {lons.size}, {values.size}."
        )

    gdf = GeoDataFrame(
        {"z": values},
        geometry=points_from_xy(lons, lats),
        crs=epsg,
    )
    result = grid_points(
        gdf, "z", Dataset, algorithm=algorithm, cell_size=cell_size, epsg=epsg
    )
    return result
