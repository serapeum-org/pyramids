"""Neighbourhood (focal) statistics over a mesh's face adjacency (issue #1376).

The raster ``focal_*`` family aggregates each cell over a ``(2*radius+1)``-side **pixel
window**, which has no meaning on an unstructured mesh -- faces have neither a row/column
address nor a fixed neighbour count. The concept does carry over, though: aggregate each face
over its **k-ring neighbourhood**, the faces reachable within ``k`` hops of the face adjacency.

Hence the parameter is named ``rings``, not ``radius``: it counts adjacency hops, and calling
it a radius would imply a distance in coordinate units or a pixel window, which it is not.

Two conventions, both chosen to mirror the raster family:

- **The centre face is included** in its own neighbourhood, as the centre pixel is included in
  a raster window.
- **Boundary faces aggregate over whatever neighbours they have.** There is no padding and no
  fabricated value, so a face on the mesh edge is averaged over a smaller neighbourhood rather
  than against invented data.

``nodata`` faces are excluded from every aggregation, and so are inactive ones when an active
(wet/dry) mask is supplied -- a dry neighbour should not drag a wet face's mean down.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import sparse

from pyramids.base._array_ops import nodata_mask
from pyramids.netcdf.ugrid.graph import sparse_adjacency
from pyramids.netcdf.ugrid.mesh import Mesh2d

__all__ = ["focal_apply", "focal_mean", "focal_std", "neighbourhood"]

_DENSE_FRACTION = 0.1
"""Reach fraction of the mesh above which the neighbourhood is no longer usefully sparse."""

_DENSE_MIN_FACES = 200
"""Mesh size below which density does not matter: a fully dense matrix is a few KB of bools."""


def neighbourhood(mesh: Mesh2d, rings: int = 1) -> Any:
    """The boolean k-ring neighbourhood matrix of the mesh faces, centre included.

    Row ``i`` marks every face within ``rings`` adjacency hops of face ``i``, including ``i``
    itself. Built as a boolean power of ``A + I``: each multiplication extends the reach by one
    hop, which is both the definition of a ring and cheaper than a per-face traversal.

    **Cost grows with ``rings``.** Each hop adds the neighbours of everything already reached,
    so as ``rings`` approaches the mesh diameter the matrix tends towards a dense ``n_face**2``.
    A warning is issued once the reach passes a tenth of a mesh of at least 200 faces, which
    is where the sparse representation stops paying for itself; below that size a fully dense
    matrix is a few kilobytes and the density does not matter.

    Pass the result back as ``reach=`` to :func:`focal_mean`, :func:`focal_std` or
    :func:`focal_apply` to build it once for a sweep instead of once per call.

    Args:
        mesh: The mesh to read.
        rings: How many adjacency hops the neighbourhood spans. ``1`` is a face and its
            edge-sharing neighbours.

    Returns:
        scipy.sparse.csr_matrix: An ``(n_face, n_face)`` boolean matrix.

    Raises:
        ValueError: ``rings`` is less than 1.
    """
    if rings < 1:
        raise ValueError(f"focal: rings must be at least 1, got {rings}.")
    adjacency = sparse.csr_matrix(sparse_adjacency(mesh, kind="face"), dtype=bool)
    identity = sparse.identity(mesh.n_face, dtype=bool, format="csr")
    one_hop = (adjacency + identity).astype(bool)
    reach = one_hop
    for _ in range(rings - 1):
        reach = (reach @ one_hop).astype(bool)
    result = sparse.csr_matrix(reach)
    density = result.nnz / max(mesh.n_face**2, 1)
    # Both conditions matter: density alone fires on every small mesh, where a fully dense
    # matrix is a few kilobytes and nobody cares.
    if density > _DENSE_FRACTION and mesh.n_face >= _DENSE_MIN_FACES:
        warnings.warn(
            f"focal: rings={rings} reaches {density:.0%} of a {mesh.n_face}-face mesh, so the "
            f"neighbourhood matrix is no longer sparse ({result.nnz} stored entries) and both "
            f"its memory and the time to build it grow with the square of the mesh. Use fewer "
            f"rings, or a whole-mesh reduction if that is what you meant.",
            UserWarning,
            # `neighbourhood` <- `_resolve_reach` <- `focal_mean` <- the facade <- the caller.
            # Four is the common case (a facade call); a direct kernel call lands one frame
            # early, which is still inside the caller's own file.
            stacklevel=5,
        )
    return result


def _resolve_reach(mesh: Mesh2d, rings: int, reach: Any) -> Any:
    """The neighbourhood matrix to aggregate over, built or validated.

    Validating matters because the three helpers failed differently on a bad ``reach``: two
    raised numpy's ``matmul: dimension mismatch`` and :func:`focal_apply` returned a **wrong
    answer**, silently truncating the matrix to the first ``n_face`` rows.

    ``rings`` and ``reach`` are mutually exclusive rather than one silently winning: a
    ``reach`` built for one ring with ``rings=3`` alongside it is a contradiction, and honouring
    the matrix while ignoring the argument is the kind of thing a caller discovers much later.

    Args:
        mesh: The mesh the values belong to.
        rings: Adjacency hops requested, or ``1`` (the default) when ``reach`` is supplied.
        reach: A matrix from :func:`neighbourhood` to reuse, or ``None`` to build one.

    Returns:
        scipy.sparse.csr_matrix: The neighbourhood matrix, in CSR form.

    Raises:
        ValueError: ``reach`` is not ``(n_face, n_face)``, or a non-default ``rings`` was
            passed alongside it.
    """
    if reach is None:
        return neighbourhood(mesh, rings)
    if rings != 1:
        raise ValueError(
            f"focal: rings={rings} was given together with a precomputed reach matrix, and "
            f"the two can disagree. Pass rings only when building the matrix here, or build "
            f"it yourself with neighbourhood(mesh, rings) and pass reach alone."
        )
    expected = (mesh.n_face, mesh.n_face)
    shape = tuple(getattr(reach, "shape", ()))
    if shape != expected:
        raise ValueError(
            f"focal: the reach matrix is {shape}, but this mesh has {mesh.n_face} faces so it "
            f"must be {expected}. It was probably built from a different mesh."
        )
    # `focal_apply` reads `indices`/`indptr`, which only a CSR matrix exposes; the other two
    # only multiply, so they accepted anything matmul-able. Converting here makes all three
    # take the same thing.
    return sparse.csr_matrix(reach)


def _usable(
    values: np.ndarray, nodata: Any, active: np.ndarray | None, mesh: Mesh2d
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """The float values and the boolean mask of faces that may contribute to an aggregate.

    Args:
        values: Per-face values.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask, or ``None``.
        mesh: The mesh, for the expected length.

    Returns:
        tuple: ``(values_as_float, usable_mask)``.

    Raises:
        ValueError: ``values`` is not one value per face, or ``active`` does not match it.
    """
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1 or array.shape[0] != mesh.n_face:
        raise ValueError(
            f"focal: expected one value per face -- a 1-D array of {mesh.n_face}, got shape "
            f"{array.shape}. Select a single step or layer first, e.g. with isel()."
        )
    usable = ~nodata_mask(array, nodata)
    if active is not None:
        mask = np.asarray(active, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError(
                f"focal: the active mask shape {mask.shape} does not match the values "
                f"{array.shape}; reduce it to one value per face first."
            )
        usable &= mask
    return array, usable


def _sums(
    reach: Any, values: np.typing.NDArray, usable: np.typing.NDArray
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """Per-face neighbourhood count, sum and sum of squares, excluding unusable faces.

    The three moments every vectorised aggregation here is built from: one sparse product
    each, rather than gathering each face's neighbourhood.

    Args:
        reach: The boolean k-ring neighbourhood matrix from :func:`neighbourhood`.
        values: Per-face values, already float.
        usable: Per-face mask of the faces allowed to contribute.

    Returns:
        tuple: ``(count, total, total_sq)``, each of length ``n_face``.
    """
    weights = usable.astype("float64")
    filled = np.where(usable, values, 0.0)
    count = np.asarray(reach @ weights, dtype="float64")
    total = np.asarray(reach @ filled, dtype="float64")
    total_sq = np.asarray(reach @ (filled * filled), dtype="float64")
    return count, total, total_sq


def focal_mean(
    mesh: Mesh2d,
    values: np.ndarray,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
    reach: Any = None,
) -> np.typing.NDArray:
    """The mean of each face's k-ring neighbourhood, centre included.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.
        reach: A matrix from :func:`neighbourhood` to reuse, or ``None`` to build one. Passing
            it lets a sweep over many variables or steps pay for it once. It must be
            ``(n_face, n_face)``, and ``rings`` must be left at its default alongside it, since
            the two can disagree.

    Returns:
        np.ndarray: Per-face mean of length ``n_face``. A face whose neighbourhood holds no
        usable value at all yields ``NaN`` -- there is nothing to average.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
        TypeError: ``func`` returned something other than a scalar.
    """
    array, usable = _usable(values, nodata, active, mesh)
    matrix = _resolve_reach(mesh, rings, reach)
    count, total, _ = _sums(matrix, array, usable)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)
    return np.asarray(result, dtype="float64")


def focal_std(
    mesh: Mesh2d,
    values: np.ndarray,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
    reach: Any = None,
) -> np.typing.NDArray:
    """The population standard deviation of each face's k-ring neighbourhood.

    Computed from the neighbourhood sums rather than by gathering each neighbourhood, so the
    whole mesh is one sparse product per moment.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.
        reach: A matrix from :func:`neighbourhood` to reuse, or ``None`` to build one. Passing
            it lets a sweep over many variables or steps pay for it once. It must be
            ``(n_face, n_face)``, and ``rings`` must be left at its default alongside it, since
            the two can disagree.

    Returns:
        np.ndarray: Per-face standard deviation of length ``n_face``; ``NaN`` where the
        neighbourhood holds no usable value.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
    """
    array, usable = _usable(values, nodata, active, mesh)
    matrix = _resolve_reach(mesh, rings, reach)
    count, total, total_sq = _sums(matrix, array, usable)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(count > 0, count, 1.0)
        mean = total / safe
        variance = total_sq / safe - mean * mean
        # Catmull's identity is one subtraction away from a tiny negative on a constant
        # neighbourhood; clamp rather than hand back a NaN from the square root.
        result = np.where(count > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)
    return np.asarray(result, dtype="float64")


def focal_apply(
    mesh: Mesh2d,
    values: np.ndarray,
    func: Any,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
    reach: Any = None,
) -> np.typing.NDArray:
    """Aggregate each face's k-ring neighbourhood with a user-supplied callable.

    ``func`` receives the 1-D array of usable values in one neighbourhood and returns a scalar,
    so anything numpy-shaped works (``np.ptp``, ``np.median``, a lambda). Unlike
    :func:`focal_mean` and :func:`focal_std`, this cannot be expressed as a sparse product, so
    it visits each face in turn -- noticeably slower on a large mesh, which is the price of an
    arbitrary aggregation.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        func: Callable taking a 1-D array and returning a scalar.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.
        reach: A matrix from :func:`neighbourhood` to reuse, or ``None`` to build one. Passing
            it lets a sweep over many variables or steps pay for it once. It must be
            ``(n_face, n_face)``, and ``rings`` must be left at its default alongside it, since
            the two can disagree.

    Returns:
        np.ndarray: Per-face aggregate of length ``n_face``; ``NaN`` where the neighbourhood
        holds no usable value, so ``func`` is never handed an empty array.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
        TypeError: ``func`` returned something other than a scalar.
    """
    array, usable = _usable(values, nodata, active, mesh)
    matrix = _resolve_reach(mesh, rings, reach)
    # The CSR structure is read directly rather than through `tolil().rows`, which would
    # materialise a Python list per face on top of the matrix it already has.
    indices, indptr = matrix.indices, matrix.indptr
    out = np.full(mesh.n_face, np.nan, dtype="float64")
    for face in range(mesh.n_face):
        # The neighbourhood always contains the centre face (the identity term in
        # `neighbourhood`), so a row is never empty and needs no guard for that.
        members = indices[indptr[face] : indptr[face + 1]]
        picked = members[usable[members]]
        if picked.size == 0:
            continue
        answer = func(array[picked])
        try:
            out[face] = float(answer)
        except (TypeError, ValueError) as error:
            # numpy's own message ("only 0-dimensional arrays can be converted...") never
            # mentions `func` or the contract it broke.
            raise TypeError(
                f"focal_apply: func must return one scalar per neighbourhood, but it returned "
                f"{type(answer).__name__} "
                f"{getattr(answer, 'shape', '') or ''} for face {face}. "
                f"Use a reducing function such as np.median or np.ptp."
            ) from error
    return out
