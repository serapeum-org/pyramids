"""Analysis engine for :class:`~pyramids.feature.FeatureCollection` (ARC-36).

Module-level implementations of the collection's analysis operations -- coordinate
and centroid derivation, IDW point->raster interpolation, and H3 indexing/binning.
The `FeatureCollection` methods are thin facades over these functions.

Because a `FeatureCollection` *is-a* `GeoDataFrame` (pandas rebuilds instances on
every slice; only `_metadata` survives `__finalize__`), the engine is a set of free
functions that take the collection as their first argument and build results via
`type(fc)(...)`, never stored instance state (an `ds.cog`-style engine object would
silently desync). This mirrors the existing `from_wfs`->`_wfs` delegation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from pyramids.base._errors import GeometryWarning
from pyramids.feature import _h3
from pyramids.feature import geometry as _geom
from pyramids.feature import tessellation as _tess

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature.collection import FeatureCollection


def with_coordinates(fc: FeatureCollection) -> FeatureCollection:
    """Explode multi-geometries and attach per-vertex ``x`` / ``y`` columns."""
    gdf = _geom.explode_gdf(gpd.GeoDataFrame(fc, copy=True), geometry="multipolygon")
    gdf = _geom.explode_gdf(gdf, geometry="geometrycollection")
    result = type(fc)(gdf)
    result["x"] = result.apply(
        _geom.get_coords, geom_col="geometry", coord_type="x", axis=1
    )
    result["y"] = result.apply(
        _geom.get_coords, geom_col="geometry", coord_type="y", axis=1
    )
    result.reset_index(drop=True, inplace=True)
    return result


def with_centroid(fc: FeatureCollection) -> FeatureCollection:
    """Attach a ``center_point`` column from the mean of each row's coordinates."""
    result = with_coordinates(fc)
    result["avg_x"] = result["x"].map(np.mean)
    result["avg_y"] = result["y"].map(np.mean)
    avg_x = result["avg_x"].to_numpy()
    avg_y = result["avg_y"].to_numpy()
    bad_mask = np.isnan(avg_x) | np.isnan(avg_y)
    if bad_mask.any():
        bad_idx = [int(i) for i, is_bad in enumerate(bad_mask) if is_bad]
        warnings.warn(
            f"with_centroid: {len(bad_idx)} row(s) yielded NaN centroids (rows {bad_idx}). "
            "Their `center_point` is an empty shapely.Point. Drop or repair those rows before "
            "running a method that requires a valid centroid (e.g. reproject, distance).",
            GeometryWarning,
            stacklevel=3,
        )
    cleaned: list[Any] = [
        Point() if bad else Point(ax, ay)
        for ax, ay, bad in zip(avg_x.tolist(), avg_y.tolist(), bad_mask.tolist())
    ]
    result["center_point"] = cleaned
    return result


def interpolate_to_raster(
    fc: FeatureCollection,
    column: str,
    *,
    method: str,
    cell_size: float | None,
    bounds: tuple[float, float, float, float] | None,
    power: float,
    n_neighbors: int | None,
    nodata: float,
) -> Dataset:
    """Interpolate a numeric point column to a single-band raster via IDW (gdal.Grid)."""
    fc._require_point_geometry("interpolate_to_raster")
    fc._require_column("interpolate_to_raster", column)
    if method != "idw":
        raise ValueError(
            f"interpolate_to_raster: method {method!r} is not supported; only 'idw' is available. "
            "Kriging is out of scope for pyramids -- see the geostatista package (the serapeum "
            "geostatistics tier)."
        )
    if len(fc) < 3:
        raise ValueError(
            f"interpolate_to_raster: need at least 3 points, got {len(fc)}"
        )
    try:
        values = fc[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"interpolate_to_raster: column {column!r} must be numeric"
        ) from exc
    if np.isnan(values).all():
        raise ValueError(f"interpolate_to_raster: column {column!r} is all-NaN")
    if n_neighbors is not None:
        algorithm = f"invdistnn:power={power}:max_points={n_neighbors}:nodata={nodata}"
    else:
        algorithm = f"invdist:power={power}:smoothing=0.0:nodata={nodata}"
    # local import: pyramids.dataset imports pyramids.feature, so import here to break the cycle.
    from pyramids.dataset import Dataset

    return Dataset.from_points(
        fc, column, algorithm=algorithm, cell_size=cell_size, bbox=bounds
    )


def h3_cells(fc: FeatureCollection, resolution: int, op: str) -> list[str]:
    """Return the H3 cell index of each point at ``resolution`` (in EPSG:4326)."""
    fc._require_point_geometry(op)
    if not 0 <= resolution <= 15:
        raise ValueError(f"{op}: resolution must be 0-15, got {resolution}")
    if fc.crs is None:
        raise ValueError(
            f"{op}: a CRS is required to convert points to lat/lng for H3 indexing"
        )
    pts = fc if fc.epsg == 4326 else fc.to_crs(4326)
    return [_h3.latlng_to_cell(geom.y, geom.x, resolution) for geom in pts.geometry]


def to_h3(fc: FeatureCollection, resolution: int) -> FeatureCollection:
    """Attach each point's H3 cell index as an ``h3`` column."""
    cells = h3_cells(fc, resolution, "to_h3")
    result = type(fc)(fc.copy())
    result["h3"] = cells
    return result


def h3_bin(
    fc: FeatureCollection, resolution: int, *, agg: Any, column: str | None
) -> FeatureCollection:
    """Aggregate points into H3 hexagon cells; one polygon per occupied cell (EPSG:4326)."""
    fc._require_column("h3_bin", column)
    cells = h3_cells(fc, resolution, "h3_bin")
    if column is None:
        counts = pd.Series(cells, dtype="object").value_counts()
        items: list[tuple[Any, float]] = [(cell, int(n)) for cell, n in counts.items()]
        name = "count"
    else:
        reducer = _tess.resolve_reducer(agg)
        try:
            values = fc[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"h3_bin: column {column!r} must be numeric") from exc
        grouped = pd.DataFrame({"_cell": cells, "_v": values}).groupby("_cell")["_v"]
        items = [(cell, float(reducer(grp.to_numpy()))) for cell, grp in grouped]
        name = column
    geometries: list = []
    idx: list = []
    agg_values: list = []
    for cell, value in items:
        boundary = _h3.cell_to_boundary(cell)
        geometries.append(Polygon([(lng, lat) for (lat, lng) in boundary]))
        idx.append(cell)
        agg_values.append(value)
    frame = gpd.GeoDataFrame(
        {"h3": idx, name: agg_values}, geometry=geometries, crs="EPSG:4326"
    )
    return type(fc)(frame)
