"""Terrain derivatives on an unstructured mesh: slope, aspect and hillshade (issue #1374).

An unstructured bathymetry / bed-level mesh is a DEM, but the raster terrain family is defined
over a 3x3 pixel window, which has no mesh equivalent. The mesh analogue is the **per-face
plane gradient**: fit a plane through the face's own node elevations and read its slope off
that. For a triangle the fit is exact; for a polygon it is the least-squares plane.

The formulas mirror :mod:`pyramids.dataset.ops._focal` exactly, so the two sides agree on
units and conventions -- slope in degrees by default, aspect in ``[0, 360)`` clockwise from
north naming the direction the slope *faces*, and hillshade clipped to ``0..255``.

One sign detail makes that reuse correct. The raster kernels take ``dz_dy`` from
:func:`numpy.gradient` over **rows**, which run southward on a north-up raster, so their
``dz_dy`` is the *negated* northward derivative. A mesh gradient is computed in map
coordinates, so the northward component is negated before the shared formulas are applied.
"""

from __future__ import annotations

import numpy as np

from pyramids.netcdf.ugrid.mesh import Mesh2d

__all__ = ["aspect", "face_plane_gradient", "hillshade", "slope"]


def face_plane_gradient(
    mesh: Mesh2d, node_values: np.ndarray
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """The per-face surface gradient, fitted from the face's own node values.

    Fits ``z = a*x + b*y + c`` through each face's nodes by least squares and returns
    ``(a, b)`` -- the eastward and **northward** derivatives. Exact for a triangle,
    least-squares for a polygon with more nodes.

    Solved in centroid-relative coordinates, which removes the intercept and leaves a 2x2
    normal-equation system per face that closes in form, so the whole mesh is computed
    vectorised rather than face by face. The padded slots of the face-node connectivity are
    masked out of every sum, so a mixed-element mesh needs no special case.

    Args:
        mesh: The mesh supplying the geometry and the face-node connectivity.
        node_values: One elevation per node, length ``n_node``.

    Returns:
        tuple: ``(dz_dx, dz_dy)``, each of length ``n_face``, in value-units per
        coordinate-unit. A face whose nodes are collinear or fewer than three -- a degenerate
        face, with no plane through it -- yields ``NaN`` in both, rather than raising or
        dividing by zero.

    Raises:
        ValueError: ``node_values`` is not one value per node.
    """
    values = np.asarray(node_values, dtype="float64")
    if values.ndim != 1 or values.shape[0] != mesh.n_node:
        raise ValueError(
            f"terrain: expected one value per node -- a 1-D array of {mesh.n_node}, got shape "
            f"{values.shape}."
        )
    conn = mesh.face_node_connectivity
    table = np.asarray(conn.data)
    valid = (table != conn.fill_value) & (table >= 0)
    # Gather with the fills pointed at node 0; they are masked out of every sum below, so the
    # value they pick up never reaches the result.
    index = np.where(valid, table, 0)
    weight = valid.astype("float64")
    count = weight.sum(axis=1)

    x = np.asarray(mesh.node_x, dtype="float64")[index]
    y = np.asarray(mesh.node_y, dtype="float64")[index]
    z = values[index]

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = (weight * x).sum(axis=1) / count
        y_mean = (weight * y).sum(axis=1) / count
        z_mean = (weight * z).sum(axis=1) / count
    dx = np.where(valid, x - x_mean[:, None], 0.0)
    dy = np.where(valid, y - y_mean[:, None], 0.0)
    dz = np.where(valid, z - z_mean[:, None], 0.0)

    sxx = (dx * dx).sum(axis=1)
    sxy = (dx * dy).sum(axis=1)
    syy = (dy * dy).sum(axis=1)
    sxz = (dx * dz).sum(axis=1)
    syz = (dy * dz).sum(axis=1)

    det = sxx * syy - sxy * sxy
    # A face with fewer than three nodes, or three collinear ones, has a singular system and
    # no plane: NaN says "undefined here" instead of inventing a gradient.
    singular = (det == 0) | (count < 3) | ~np.isfinite(det)
    safe = np.where(singular, 1.0, det)
    dz_dx = np.where(singular, np.nan, (syy * sxz - sxy * syz) / safe)
    dz_dy = np.where(singular, np.nan, (sxx * syz - sxy * sxz) / safe)
    return dz_dx, dz_dy


def _row_wise(dz_dy_north: np.typing.NDArray) -> np.typing.NDArray:
    """The northward derivative as the raster kernels' row-wise one (negated).

    The shared formulas were written against :func:`numpy.gradient` over raster rows, which run
    southward on a north-up grid. Negating here is what lets the mesh reuse them unchanged.

    Args:
        dz_dy_north: The derivative in the northward (increasing y) direction.

    Returns:
        np.ndarray: The same derivative with the raster's row-wise sign convention.
    """
    return -dz_dy_north


def slope(
    mesh: Mesh2d, node_values: np.ndarray, *, units: str = "degrees"
) -> np.typing.NDArray:
    """Per-face slope of a mesh DEM, in degrees (default) or radians.

    Args:
        mesh: The mesh to read.
        node_values: One elevation per node.
        units: ``"degrees"`` (default) or ``"radians"``.

    Returns:
        np.ndarray: Per-face slope magnitude of length ``n_face``; ``NaN`` on a degenerate face.

    Raises:
        ValueError: ``units`` is neither ``"degrees"`` nor ``"radians"``, or ``node_values`` is
            not one value per node.
    """
    if units not in ("degrees", "radians"):
        raise ValueError(f"slope: units must be 'degrees' or 'radians', got {units!r}.")
    dz_dx, dz_dy = face_plane_gradient(mesh, node_values)
    # `hypot` is sign-blind, so `_row_wise` cannot change this result; it is applied for
    # symmetry with the `arctan2` calls, where the sign is load bearing.
    radians = np.arctan(np.hypot(dz_dx, _row_wise(dz_dy)))
    result = np.degrees(radians) if units == "degrees" else radians
    return np.asarray(result, dtype="float64")


def aspect(mesh: Mesh2d, node_values: np.ndarray) -> np.typing.NDArray:
    """Per-face aspect of a mesh DEM, in degrees clockwise from north.

    Names the direction the slope **faces** (downhill), matching the raster convention: a
    surface rising to the east faces west, 270 degrees.

    Args:
        mesh: The mesh to read.
        node_values: One elevation per node.

    Returns:
        np.ndarray: Per-face aspect in ``[0, 360)`` of length ``n_face``; ``NaN`` on a
        degenerate face. A perfectly flat face has no direction to face, and answers ``270.0``
        rather than anything meaningful: the signed zeros make ``arctan2(-0.0, -0.0) == -pi``.
        The raster ``aspect`` answers 270.0 on a flat DEM for the same reason, so the two
        agree — but treat the value on a flat face as an artefact, not a direction.

    Raises:
        ValueError: ``node_values`` is not one value per node.
    """
    dz_dx, dz_dy = face_plane_gradient(mesh, node_values)
    angle = np.degrees(np.arctan2(_row_wise(dz_dy), -dz_dx))
    return np.asarray(np.mod(450.0 - angle, 360.0), dtype="float64")


def hillshade(
    mesh: Mesh2d,
    node_values: np.ndarray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.typing.NDArray:
    """Per-face shaded relief of a mesh DEM, clipped to ``0..255``.

    Args:
        mesh: The mesh to read.
        node_values: One elevation per node.
        azimuth: Sun azimuth in degrees (0 = north, 90 = east). Defaults to 315 (NW), the
            cartographic convention the raster version uses.
        altitude: Sun altitude above the horizon in degrees. Defaults to 45.

    Returns:
        np.ndarray: Per-face intensity in ``[0, 255]`` of length ``n_face``; ``NaN`` on a
        degenerate face. A flat mesh saturates at ``sin(altitude) * 255``, as the raster
        version does.

    Raises:
        ValueError: ``node_values`` is not one value per node.
    """
    dz_dx, dz_dy = face_plane_gradient(mesh, node_values)
    row_wise = _row_wise(dz_dy)
    az_rad = np.deg2rad(360.0 - azimuth + 90.0)
    alt_rad = np.deg2rad(altitude)
    slope_rad = np.arctan(np.hypot(dz_dx, row_wise))
    aspect_rad = np.arctan2(row_wise, -dz_dx)
    shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(
        slope_rad
    ) * np.cos(az_rad - aspect_rad)
    return np.asarray(np.clip(shaded * 255.0, 0.0, 255.0), dtype="float64")
