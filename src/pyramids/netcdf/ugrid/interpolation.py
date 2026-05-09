"""Mesh-to-grid interpolation for bridging unstructured and structured data.

Implements the core bridge between UgridDataset (unstructured mesh)
and Dataset (regular grid). Converts mesh data to regular grid arrays
using nearest-neighbor or linear interpolation.

Depends on:
    - mesh.py: Mesh2d
    - scipy.spatial: cKDTree (optional, imported inline)
    - scipy.interpolate: LinearNDInterpolator (optional, imported inline)
"""

from __future__ import annotations

import numpy as np

from pyramids.netcdf.ugrid.mesh import Mesh2d


def mesh_to_grid(
    mesh: Mesh2d,
    data: np.ndarray,
    location: str,
    cell_size: float,
    method: str = "nearest",
    bounds: tuple[float, float, float, float] | None = None,
    nodata: float = -9999.0,
    max_distance: float | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Interpolate mesh data onto a regular grid.

    This is the core bridge function. The result is a 2D numpy array
    and a GDAL-style geotransform that can be wrapped in a pyramids
    Dataset.

    Args:
        mesh: Source mesh topology.
        data: 1D data array (n_faces, n_nodes, or n_edges).
        location: "face", "node", or "edge".
        cell_size: Target grid cell size in coordinate units.
        method: Interpolation method ("nearest" or "linear").
        bounds: (xmin, ymin, xmax, ymax). Defaults to mesh bounds.
        nodata: Fill value for grid cells outside the mesh.
        max_distance: Maximum distance from a mesh element for a
            grid cell to receive a value. Defaults to 2 * cell_size.

    Returns:
        Tuple of (grid_array, geotransform) where grid_array is 2D
        (rows, cols) and geotransform is a 6-element tuple
        (x_origin, cell_size, 0, y_origin, 0, -cell_size).

    Raises:
        ValueError: If location is unknown or edge coordinates are
            unavailable for edge-centered data.
    """
    if bounds is None:
        bounds = mesh.bounds
    xmin, ymin, xmax, ymax = bounds

    if max_distance is None:
        max_distance = 2.0 * cell_size

    cols = int(np.ceil((xmax - xmin) / cell_size))
    rows = int(np.ceil((ymax - ymin) / cell_size))

    x_centers = xmin + (np.arange(cols) + 0.5) * cell_size
    y_centers = ymax - (np.arange(rows) + 0.5) * cell_size
    xx, yy = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    source_points, source_values = _get_source_data(mesh, data, location)

    if method == "nearest":
        grid_values = _interpolate_nearest(
            source_points, source_values, grid_points, nodata, max_distance
        )
    elif method == "linear":
        grid_values = _interpolate_linear(
            source_points, source_values, grid_points, nodata
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    grid_array = grid_values.reshape(rows, cols)
    geotransform = (xmin, cell_size, 0.0, ymax, 0.0, -cell_size)

    result = (grid_array, geotransform)
    return result


def _get_source_data(
    mesh: Mesh2d,
    data: np.ndarray,
    location: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract source point coordinates based on mesh location.

    Args:
        mesh: Source mesh topology.
        data: 1D data array.
        location: "face", "node", or "edge".

    Returns:
        Tuple of (source_points, source_values) where source_points
        is (N, 2) and source_values is (N,).

    Raises:
        ValueError: If location is unknown or edge coords unavailable.
    """
    if location == "face":
        cx, cy = mesh.face_centroids
    elif location == "node":
        cx, cy = mesh.node_x, mesh.node_y
    elif location == "edge":
        if mesh._edge_x is not None and mesh._edge_y is not None:
            cx, cy = mesh._edge_x, mesh._edge_y
        else:
            raise ValueError("Edge coordinates not available for interpolation.")
    else:
        raise ValueError(f"Unknown location: {location}")

    source_points = np.column_stack([cx, cy])
    source_values = np.asarray(data, dtype=np.float64)

    result = (source_points, source_values)
    return result


def _interpolate_nearest(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    nodata: float,
    max_distance: float,
) -> np.ndarray:
    """Nearest-neighbor interpolation with max_distance cutoff.

    Args:
        source_points: (N, 2) source coordinates.
        source_values: (N,) source data values.
        target_points: (M, 2) target grid coordinates.
        nodata: Fill value for out-of-range cells.
        max_distance: Maximum distance for assignment.

    Returns:
        (M,) array of interpolated values.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(source_points)
    distances, indices = tree.query(target_points, k=1)

    result = np.full(len(target_points), nodata, dtype=np.float64)
    valid = distances <= max_distance
    result[valid] = source_values[indices[valid]]

    return result


def _interpolate_linear(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """Linear interpolation via scipy.interpolate.LinearNDInterpolator.

    Args:
        source_points: (N, 2) source coordinates.
        source_values: (N,) source data values.
        target_points: (M, 2) target grid coordinates.
        nodata: Fill value for points outside the convex hull.

    Returns:
        (M,) array of interpolated values.
    """
    from scipy.interpolate import LinearNDInterpolator

    interpolator = LinearNDInterpolator(source_points, source_values, fill_value=nodata)
    result = interpolator(target_points)

    return result
