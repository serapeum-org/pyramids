"""Mesh-native analysis: point extraction, random sampling, contouring.

Module-level functions taking :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`
plus plain arrays — no dataset import, keeping the import graph acyclic.
Same contract as ``fill.py``, ``graph.py``, ``vector_ops.py``.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex


def extract_points(
    mesh: Mesh2d,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    location: str = "face",
) -> np.ndarray:
    """Sample variable values at arbitrary coordinates.

    Args:
        mesh: The mesh topology.
        values: Variable data (``(n_elements,)`` or ``(n_time, n_elements)``).
        x: Query x-coordinates.
        y: Query y-coordinates.
        location: ``"face"`` (point-in-face) or ``"node"`` (nearest node).

    Returns:
        Extracted values — shape ``(n_points,)`` or ``(n_time, n_points)``.
        Points outside the mesh get ``NaN``.

    Raises:
        ValueError: If *location* is not ``"face"`` or ``"node"``.

    Examples:
        - Sample face values at a query point inside the first triangle:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> from pyramids.netcdf.ugrid.analysis import extract_points
            >>> fnc = Connectivity(
            ...     np.array([[0, 1, 2], [1, 3, 2]]),
            ...     -1, "face_node_connectivity", 0,
            ... )
            >>> mesh = Mesh2d(
            ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            ...     face_node_connectivity=fnc,
            ... )
            >>> values = np.array([10.0, 20.0])
            >>> extract_points(mesh, values, np.array([0.3]), np.array([0.3]))
            array([10.])

            ```
        - Points outside the mesh return NaN:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> from pyramids.netcdf.ugrid.analysis import extract_points
            >>> fnc = Connectivity(
            ...     np.array([[0, 1, 2], [1, 3, 2]]),
            ...     -1, "face_node_connectivity", 0,
            ... )
            >>> mesh = Mesh2d(
            ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            ...     face_node_connectivity=fnc,
            ... )
            >>> extract_points(
            ...     mesh, np.array([10.0, 20.0]),
            ...     np.array([5.0]), np.array([5.0]),
            ... )
            array([nan])

            ```

    See Also:
        sample_random: Draw spatially random samples instead of
            querying specific coordinates.

    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    idx = MeshSpatialIndex(mesh)

    if location == "face":
        element_ids = idx.locate_faces(x, y)
        outside = element_ids < 0
        safe_ids = np.where(outside, 0, element_ids)
        result = values[..., safe_ids].astype(float)
        result[..., outside] = np.nan
    elif location == "node":
        element_ids = idx.locate_nearest_node(x, y).ravel()
        result = values[..., element_ids].astype(float)
    else:
        raise ValueError(f"location must be 'face' or 'node', got {location!r}")
    return result


def sample_random(
    mesh: Mesh2d,
    values: np.ndarray,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw *n* spatially random samples from the mesh domain.

    Uses rejection sampling within the mesh bounding box, keeping only
    points that fall inside a mesh face.

    Args:
        mesh: The mesh topology.
        values: Variable data (``(n_elements,)`` or ``(n_time, n_elements)``).
        n: Number of samples to draw.
        seed: RNG seed for reproducibility.

    Returns:
        ``(x, y, sampled_values)`` — arrays of length *n*.

    Raises:
        ValueError: If *n* < 1.
        RuntimeError: If rejection sampling cannot collect *n* points
            after 100 batches (the mesh domain may be degenerate or
            extremely sparse relative to its bounding box).

    Examples:
        - Draw three random samples with a fixed seed:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> from pyramids.netcdf.ugrid.analysis import sample_random
            >>> fnc = Connectivity(
            ...     np.array([[0, 1, 2], [1, 3, 2]]),
            ...     -1, "face_node_connectivity", 0,
            ... )
            >>> mesh = Mesh2d(
            ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            ...     face_node_connectivity=fnc,
            ... )
            >>> values = np.array([10.0, 20.0])
            >>> x, y, sampled = sample_random(mesh, values, 3, seed=42)
            >>> x.shape
            (3,)
            >>> sampled.shape
            (3,)

            ```

    See Also:
        extract_points: Sample at specific coordinates rather than
            random ones.

    """
    if n < 1:
        raise ValueError(f"`n` must be >= 1, got {n}.")
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = mesh.bounds
    idx = MeshSpatialIndex(mesh)

    collected_x: list[float] = []
    collected_y: list[float] = []
    collected_ids: list[int] = []
    max_iters = 100

    iterations = 0
    while len(collected_x) < n:
        iterations += 1
        if iterations > max_iters:
            raise RuntimeError(
                f"sample_random failed to collect {n} samples after "
                f"{max_iters} batches — the mesh domain may be "
                f"degenerate or extremely sparse relative to its "
                f"bounding box."
            )
        batch = max(n * 3, 256)
        cx = rng.uniform(xmin, xmax, batch)
        cy = rng.uniform(ymin, ymax, batch)
        face_ids = idx.locate_faces(cx, cy)
        inside = face_ids >= 0
        cx, cy = cx[inside], cy[inside]
        face_ids = face_ids[inside]
        needed = n - len(collected_x)
        collected_x.extend(cx[:needed].tolist())
        collected_y.extend(cy[:needed].tolist())
        collected_ids.extend(face_ids[:needed].tolist())

    sx = np.array(collected_x[:n])
    sy = np.array(collected_y[:n])
    ids = np.array(collected_ids[:n], dtype=np.intp)
    sampled = values[..., ids].astype(float)
    return sx, sy, sampled


def contour_lines(
    mesh: Mesh2d,
    values: np.ndarray,
    levels: Any = None,
    location: str = "face",
) -> Any:
    """Generate contour lines from mesh data via tricontour.

    Args:
        mesh: The mesh topology.
        values: 1-D variable data (one value per element).
        levels: Contour levels (int for count, array for explicit).
        location: ``"face"`` or ``"node"``.

    Returns:
        GeoDataFrame: Contour lines with a ``level`` column.
    """
    from pyramids.base._utils import require_cleopatra

    require_cleopatra()
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri_arr = mesh.fan_triangles
    triangulation = mtri.Triangulation(mesh.node_x, mesh.node_y, tri_arr)

    if location == "face":
        node_values = _face_values_to_nodes(mesh, values)
    else:
        node_values = np.asarray(values)

    if levels is None:
        levels = 10
    fig, ax = plt.subplots()
    cs = ax.tricontour(triangulation, node_values, levels)

    geometries = []
    level_values = []
    for level_val, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 2:
                geometries.append(LineString(seg))
                level_values.append(float(level_val))
    plt.close(fig)

    result = gpd.GeoDataFrame(
        {"level": level_values}, geometry=geometries, crs=mesh.crs
    )
    return result


def _face_values_to_nodes(mesh: Mesh2d, face_values: np.ndarray) -> np.ndarray:
    """Average face values to nodes for tricontour input.

    Each node gets the mean of its surrounding face values.
    """
    node_sums = np.zeros(mesh.n_node, dtype=float)
    node_counts = np.zeros(mesh.n_node, dtype=float)
    fnc = mesh.face_node_connectivity
    masked = fnc.as_masked()
    valid = ~np.ma.getmaskarray(masked)

    rows, cols = np.nonzero(valid)
    node_ids = masked.data[rows, cols]
    np.add.at(node_sums, node_ids, face_values[rows])
    np.add.at(node_counts, node_ids, 1.0)

    node_counts[node_counts == 0] = 1.0
    return node_sums / node_counts


def plot_histogram(
    values: np.ndarray,
    nodata: Any = None,
    bins: int = 15,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a variable's value distribution as a histogram.

    Delegates to cleopatra's ``HistogramGlyph``.

    Args:
        values: The flat data array.
        nodata: No-data sentinel to exclude.
        bins: Number of histogram bins.
        ax: Existing matplotlib axes to draw into.
        **kwargs: Forwarded to ``HistogramGlyph``.

    Returns:
        ``(fig, ax, hist)`` from ``HistogramGlyph.histogram``.

    Raises:
        ValueError: If all values are NaN / nodata and nothing
            remains to histogram.

    See Also:
        contour_lines: Spatial contour extraction from the same
            mesh variable.
    """
    from pyramids.base._utils import require_cleopatra

    require_cleopatra()
    from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph

    valid = valid_samples(values, nodata)
    glyph = HistogramGlyph(valid, ax=ax, **HistogramGlyph.filter_kwargs(kwargs))
    result = glyph.histogram(bins=bins)
    return result


def valid_samples(values: np.ndarray, nodata: Any = None) -> np.ndarray:
    """The flat array of values that are neither NaN nor the declared no-data sentinel.

    Factored out of :func:`plot_histogram` so the numbers :func:`histogram` reports and the
    bars :func:`plot_histogram` draws are derived from the **same** samples and cannot
    disagree (issue #1375).

    Args:
        values: The data array, of any shape.
        nodata: No-data sentinel to exclude, or ``None``.

    Returns:
        np.ndarray: The 1-D array of valid samples.

    Raises:
        ValueError: Nothing survives the masking, so there is no distribution to describe.
    """
    flat = np.asarray(values).ravel()
    mask = np.ones(flat.shape, dtype=bool)
    if np.issubdtype(flat.dtype, np.floating):
        mask &= ~np.isnan(flat)
    if nodata is not None and not (
        np.issubdtype(flat.dtype, np.floating) and np.isnan(nodata)
    ):
        mask &= flat != nodata
    valid = flat[mask]
    if valid.size == 0:
        raise ValueError("No valid samples to histogram after masking nodata / NaN.")
    return valid


def histogram(
    values: np.ndarray,
    nodata: Any = None,
    bins: int = 256,
    value_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The histogram of a variable's values as data, rather than as a picture.

    The mesh counterpart of the raster ``get_histogram``. :func:`plot_histogram` could already
    draw the distribution but gave no way to obtain it; both now mask through
    :func:`valid_samples`, so the counts here describe exactly the samples the bars show.

    Args:
        values: The data array, of any shape.
        nodata: No-data sentinel to exclude, or ``None``.
        bins: Number of equal-width bins. Defaults to 256, as the raster version does.
        value_range: ``(low, high)`` to bin within, or ``None`` to span the data.

    Returns:
        tuple: ``(counts, edges)`` -- the per-bin counts and the ``bins + 1`` bin edges, in the
        variable's own units.

    Raises:
        ValueError: Nothing survives the no-data masking.
    """
    valid = valid_samples(values, nodata)
    counts, edges = np.histogram(valid, bins=bins, range=value_range)
    return counts, edges
