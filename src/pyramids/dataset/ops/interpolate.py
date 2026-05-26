"""Interpolate scattered point samples onto a regular grid via ``gdal.Grid``.

Backs :meth:`pyramids.dataset.Dataset.from_points` — a GDAL-native way to turn
gauge/station observations into a continuous raster. Supports every ``gdal.Grid``
algorithm
(``invdist``, ``invdistnn``, ``nearest``, ``linear``, ``average``, …) via the
algorithm string. No new third-party dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geopandas import GeoDataFrame

    from pyramids.dataset.dataset import Dataset

_DEFAULT_ALGORITHM = "invdist:power=2.0:smoothing=0.0"


def grid_points(
    points: GeoDataFrame,
    value_column: str,
    dataset_cls: type[Dataset],
    *,
    algorithm: str = _DEFAULT_ALGORITHM,
    cell_size: float | None = None,
    width: int | None = None,
    height: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    epsg: Any | None = None,
) -> Dataset:
    """Interpolate a point layer's ``value_column`` onto a grid with ``gdal.Grid``.

    Args:
        points: A point :class:`~pyramids.feature.FeatureCollection` /
            :class:`geopandas.GeoDataFrame` carrying ``value_column``.
        value_column: Numeric attribute column to interpolate (the Z field).
        dataset_cls: The :class:`~pyramids.dataset.Dataset` class to wrap the
            result in (passed by the classmethod so subclasses round-trip).
        algorithm: A ``gdal.Grid`` algorithm string, e.g.
            ``"invdist:power=2.0:smoothing=0.0"``, ``"nearest"``, ``"linear"``,
            ``"average:radius1=0:radius2=0"``.
        cell_size: Output pixel size (in the points' CRS units). Required unless
            both ``width`` and ``height`` are given.
        width: Output width in pixels (overrides ``cell_size`` for the x axis).
        height: Output height in pixels (overrides ``cell_size`` for the y axis).
        bbox: ``(minx, miny, maxx, maxy)`` output extent; defaults to the
            points' total bounds.
        epsg: Output EPSG code; defaults to the points' CRS.

    Returns:
        A single-band :class:`~pyramids.dataset.Dataset` of the interpolated
        surface.

    Raises:
        ValueError: ``value_column`` missing, bounds degenerate, or neither
            ``cell_size`` nor ``width``+``height`` provided.
        FailedToSaveError: ``gdal.Grid`` returned no dataset.
    """
    if value_column not in points.columns:
        raise ValueError(
            f"value_column {value_column!r} is not in the points columns: "
            f"{list(points.columns)}"
        )

    if bbox is not None:
        minx, miny, maxx, maxy = (float(v) for v in bbox)
    else:
        minx, miny, maxx, maxy = (float(v) for v in points.total_bounds)
    if maxx <= minx or maxy <= miny:
        raise ValueError(
            f"degenerate output bounds (minx={minx}, miny={miny}, maxx={maxx}, "
            f"maxy={maxy}); pass a valid bbox or non-collinear points."
        )

    if width is None or height is None:
        if cell_size is None:
            raise ValueError(
                "from_points requires either cell_size or both width and height."
            )
        width = max(1, round((maxx - minx) / cell_size))
        height = max(1, round((maxy - miny) / cell_size))

    output_srs: str | None = None
    if epsg is not None:
        output_srs = f"EPSG:{int(epsg)}"
    elif points.crs is not None:
        output_srs = points.crs.to_wkt()

    options = gdal.GridOptions(
        format="MEM",
        algorithm=algorithm,
        zfield=value_column,
        outputBounds=[minx, maxy, maxx, miny],
        width=int(width),
        height=int(height),
        outputSRS=output_srs,
    )
    with _feature_ogr.as_vsimem_path(points) as src_path:
        result = gdal.Grid("", src_path, options=options)
    if result is None:
        raise FailedToSaveError(
            f"gdal.Grid returned no dataset for algorithm {algorithm!r}."
        )
    return dataset_cls(result)
