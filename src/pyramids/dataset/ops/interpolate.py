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


def _resolve_size(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    cell_size: float | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Validate the output bounds and settle the output ``(width, height)``.

    Shared by both grid entry points so the degenerate-bounds guard and the
    cell-size-to-pixel arithmetic live in exactly one place.

    Args:
        minx: West edge of the output extent.
        miny: South edge.
        maxx: East edge.
        maxy: North edge.
        cell_size: Output pixel size in the coordinates' own units. Used only when
            ``width`` / ``height`` are not both given.
        width: Explicit output width in pixels, or ``None`` to derive it.
        height: Explicit output height in pixels, or ``None`` to derive it.

    Returns:
        tuple[int, int]: The ``(width, height)`` to request, each at least 1 pixel.

    Raises:
        ValueError: The bounds are empty/degenerate, or neither ``cell_size`` nor
            both of ``width`` / ``height`` were given.
    """
    if maxx <= minx or maxy <= miny:
        raise ValueError(
            f"degenerate output bounds (minx={minx}, miny={miny}, maxx={maxx}, "
            f"maxy={maxy}); pass a valid bbox or non-collinear points."
        )
    if width is None or height is None:
        if cell_size is None:
            raise ValueError(
                "gridding requires either cell_size or both width and height."
            )
        width = max(1, round((maxx - minx) / cell_size))
        height = max(1, round((maxy - miny) / cell_size))
    return int(width), int(height)


def grid_arrays(
    x: Any,
    y: Any,
    z: Any,
    dataset_cls: type[Dataset],
    *,
    algorithm: str = _DEFAULT_ALGORITHM,
    cell_size: float | None = None,
    width: int | None = None,
    height: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    output_srs: str | None = None,
) -> Dataset:
    """Grid raw coordinate arrays with ``gdal.Grid`` — no geometry ever built.

    The array-native core of the scattered-point bridge. :func:`grid_points`
    (behind :meth:`Dataset.from_points` and
    :meth:`~pyramids.feature.FeatureCollection.interpolate_to_raster`) delegates
    here for a point layer after pulling ``x`` / ``y`` / ``z`` off the geometry,
    and :meth:`Dataset.from_point_arrays` calls it directly on arrays that were
    never wrapped in shapely at all. One core, so the sizing, options and grid call
    are defined once.

    Args:
        x: Point x-coordinates (anything :func:`numpy.asarray` reads as 1-D).
        y: Point y-coordinates, the same length as ``x``.
        z: The value to interpolate at each point, the same length as ``x``.
        dataset_cls: The :class:`~pyramids.dataset.Dataset` class to wrap the
            result in (so a subclass round-trips).
        algorithm: A ``gdal.Grid`` algorithm string, e.g.
            ``"invdist:power=2.0:smoothing=0.0"``, ``"nearest"``, ``"linear"``.
        cell_size: Output pixel size in the coordinates' units. Required unless
            both ``width`` and ``height`` are given.
        width: Output width in pixels (with ``height``, overrides ``cell_size``).
        height: Output height in pixels (with ``width``, overrides ``cell_size``).
        bbox: ``(minx, miny, maxx, maxy)`` output extent; defaults to the arrays'
            own min/max.
        output_srs: A resolved SRS string (WKT or ``"EPSG:<n>"``) to stamp on the
            result, or ``None``. ``gdal.Grid`` labels the output with it; it does
            not reproject.

    Returns:
        Dataset: A single-band raster of the interpolated surface.

    Raises:
        ValueError: The arrays are not 1-D of equal length, are empty, the bounds
            are degenerate, or the sizing arguments are insufficient.
        FailedToSaveError: ``gdal.Grid`` returned no dataset.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    zs = np.asarray(z, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or zs.ndim != 1:
        raise ValueError(
            f"grid_arrays expects 1-D arrays; got x.ndim={xs.ndim}, y.ndim="
            f"{ys.ndim}, z.ndim={zs.ndim}."
        )
    if not xs.shape == ys.shape == zs.shape:
        raise ValueError(
            f"grid_arrays: x, y and z must have equal length; got {xs.size}, "
            f"{ys.size} and {zs.size}."
        )
    if xs.size == 0:
        raise ValueError("grid_arrays requires at least one point; got empty arrays.")

    if bbox is not None:
        minx, miny, maxx, maxy = (float(v) for v in bbox)
    else:
        minx, miny = float(xs.min()), float(ys.min())
        maxx, maxy = float(xs.max()), float(ys.max())
    out_w, out_h = _resolve_size(
        minx, miny, maxx, maxy, cell_size=cell_size, width=width, height=height
    )

    options = gdal.GridOptions(
        format="MEM",
        algorithm=algorithm,
        zfield="z",
        outputBounds=[minx, maxy, maxx, miny],
        width=out_w,
        height=out_h,
        outputSRS=output_srs,
    )
    result = _grid_from_arrays(xs, ys, zs, options)
    if result is None:
        raise FailedToSaveError(
            f"gdal.Grid returned no dataset for algorithm {algorithm!r}."
        )
    return dataset_cls(result)


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

    output_srs: str | None = None
    if epsg is not None:
        output_srs = f"EPSG:{int(epsg)}"
    elif points.crs is not None:
        output_srs = points.crs.to_wkt()

    if len(points) > 0 and bool((points.geom_type == "Point").all()):
        # A point layer needs no geometry at all for gridding — pull the raw
        # coordinates off the geometry and hand them to the shared array core,
        # which skips the per-point shapely round trip and the GeoJSON encoding.
        return grid_arrays(
            points.geometry.x.to_numpy(),
            points.geometry.y.to_numpy(),
            points[value_column].to_numpy(),
            dataset_cls,
            algorithm=algorithm,
            cell_size=cell_size,
            width=width,
            height=height,
            bbox=bbox,
            output_srs=output_srs,
        )

    # A non-point layer (or an empty one) keeps the original GeoJSON round trip,
    # which handles whatever geometry gdal.Grid is handed and reads its z from the
    # named attribute column directly.
    if bbox is not None:
        minx, miny, maxx, maxy = (float(v) for v in bbox)
    else:
        minx, miny, maxx, maxy = (float(v) for v in points.total_bounds)
    out_w, out_h = _resolve_size(
        minx, miny, maxx, maxy, cell_size=cell_size, width=width, height=height
    )
    options = gdal.GridOptions(
        format="MEM",
        algorithm=algorithm,
        zfield=value_column,
        outputBounds=[minx, maxy, maxx, miny],
        width=out_w,
        height=out_h,
        outputSRS=output_srs,
    )
    with _feature_ogr.as_vsimem_path(points) as src_path:
        result = gdal.Grid("", src_path, options=options)
    if result is None:
        raise FailedToSaveError(
            f"gdal.Grid returned no dataset for algorithm {algorithm!r}."
        )
    return dataset_cls(result)
