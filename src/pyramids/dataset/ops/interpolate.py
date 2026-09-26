"""Interpolate scattered point samples onto a regular grid via ``gdal.Grid``.

Backs :meth:`pyramids.dataset.Dataset.from_points` — a GDAL-native way to turn
gauge/station observations into a continuous raster. Supports every ``gdal.Grid``
algorithm
(``invdist``, ``invdistnn``, ``nearest``, ``linear``, ``average``, …) via the
algorithm string. No new third-party dependencies.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from osgeo import gdal

from pyramids.base._errors import FailedToSaveError
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geopandas import GeoDataFrame

    from pyramids.dataset.dataset import Dataset

_DEFAULT_ALGORITHM = "invdist:power=2.0:smoothing=0.0"


def _grid_from_arrays(
    x: Any,
    y: Any,
    z: Any,
    options: Any,
) -> gdal.Dataset | None:
    """Run ``gdal.Grid`` on raw coordinate arrays, no geometry layer in between.

    ``gdal.Grid`` needs an OGR-readable source, not a NumPy array. Serialising the
    points to GeoJSON through a :class:`~geopandas.GeoDataFrame` — the path
    :func:`grid_points` falls back to for a non-point layer — pays for a shapely
    geometry per point and a verbose text encoding, which is unusable above roughly
    ``10**6`` points. Writing the three arrays to an in-memory CSV read through an
    OGR VRT skips both: the coordinates become the point geometry via the VRT's
    ``PointFromColumns`` encoding and the value rides along as a real field.
    Measured at about four times the throughput of the GeoJSON path on a million
    points, with byte-identical grid output.

    ``%.17g`` formats each float at the 17 significant digits that round-trip an
    IEEE-754 double exactly, so the CSV detour perturbs no coordinate.

    Args:
        x: Point x-coordinates (anything :func:`numpy.asarray` reads as 1-D).
        y: Point y-coordinates, the same length as ``x``.
        z: The value to interpolate at each point, the same length as ``x``.
        options: A :func:`osgeo.gdal.GridOptions` bundle whose ``zfield`` is
            ``"z"`` — the fixed name this helper writes the value column under.

    Returns:
        gdal.Dataset | None: The gridded raster, or ``None`` when ``gdal.Grid``
        declines the request (the caller turns that into an error).
    """
    # The OGR CSV driver names its single layer after the file stem, and an
    # OGRVRTLayer's name= is also the source layer it looks up — so the two must
    # agree or GDAL reports "Failed to find layer". Deriving both from one token
    # keeps them in step and unique across concurrent calls.
    layer = f"grid_{uuid.uuid4().hex}"
    csv_path = f"/vsimem/{layer}.csv"
    vrt_path = f"/vsimem/{layer}.vrt"
    frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z)})
    gdal.FileFromMemBuffer(
        csv_path, frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    )
    vrt = (
        f'<OGRVRTDataSource><OGRVRTLayer name="{layer}">'
        f'<SrcDataSource relativeToVRT="0">{csv_path}</SrcDataSource>'
        "<GeometryType>wkbPoint</GeometryType>"
        '<GeometryField encoding="PointFromColumns" x="x" y="y"/>'
        '<Field name="z" src="z" type="Real"/>'
        "</OGRVRTLayer></OGRVRTDataSource>"
    )
    gdal.FileFromMemBuffer(vrt_path, vrt.encode("utf-8"))
    try:
        return gdal.Grid("", vrt_path, options=options)
    finally:
        gdal.Unlink(csv_path)
        gdal.Unlink(vrt_path)


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

    all_points = len(points) > 0 and bool((points.geom_type == "Point").all())
    if all_points:
        # Fast path: hand the raw coordinate arrays to gdal.Grid via an in-memory
        # CSV + OGR VRT, skipping the per-point shapely geometry and the GeoJSON
        # serialization the fallback below pays for. The CSV always names its
        # columns x/y/z, so `value_column` may be anything (even "x") without a
        # clash, and the options' zfield is "z" to match.
        options = gdal.GridOptions(
            format="MEM",
            algorithm=algorithm,
            zfield="z",
            outputBounds=[minx, maxy, maxx, miny],
            width=int(width),
            height=int(height),
            outputSRS=output_srs,
        )
        result = _grid_from_arrays(
            points.geometry.x.to_numpy(),
            points.geometry.y.to_numpy(),
            np.asarray(points[value_column].to_numpy(), dtype=float),
            options,
        )
    else:
        # A non-point layer (or an empty one) keeps the original GeoJSON round
        # trip, which handles whatever geometry gdal.Grid is handed.
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
