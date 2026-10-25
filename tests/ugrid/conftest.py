"""Shared fixtures for UGRID tests.

Provides reusable mesh topologies, connectivity arrays, and
test file paths used across multiple test modules.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

try:
    # Imported once at module scope rather than inside the autouse fixture, which would run
    # the import machinery after every test in this package.
    from matplotlib import animation
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is a viz-extra dependency
    animation = None
    plt = None
_MATPLOTLIB_IMPORT = plt is not None

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable


def _two_triangle_mesh() -> Mesh2d:
    """Return the shared 4-node / 2-face triangle mesh."""
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=fnc,
    )


def _square_mesh() -> Mesh2d:
    """Return the shared 5-node / 4-face square mesh."""
    fnc = Connectivity(
        data=np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.intp,
        ),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 0.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
        face_node_connectivity=fnc,
    )


def make_two_triangle_ds(
    values: np.ndarray | None = None,
    var_name: str = "depth",
    nodata=None,
    global_attributes: dict | None = None,
    load_data: bool = True,
) -> UgridDataset:
    """Build a minimal 2-face triangular UgridDataset.

    Parameters ``nodata``, ``global_attributes``, and ``load_data``
    let callers test edge cases without rebuilding the mesh inline.
    """
    mesh = _two_triangle_mesh()
    if load_data:
        if values is None:
            values = np.array([10.0, 20.0])
        kwargs: dict = dict(
            name=var_name,
            location="face",
            mesh_name="mesh2d",
            shape=values.shape,
            _data=values,
        )
    else:
        kwargs = dict(
            name=var_name,
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
    if nodata is not None:
        kwargs["nodata"] = nodata
    var = MeshVariable(**kwargs)
    return UgridDataset(
        mesh=mesh,
        data_variables={var_name: var},
        global_attributes=global_attributes or {},
    )


def make_square_ds(
    values: np.ndarray | None = None,
) -> UgridDataset:
    """Build a 4-face square UgridDataset with a center node."""
    mesh = _square_mesh()
    if values is None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
    var = MeshVariable(
        name="elevation",
        location="face",
        mesh_name="mesh2d",
        shape=values.shape,
        _data=values,
    )
    return UgridDataset(
        mesh=mesh,
        data_variables={"elevation": var},
        global_attributes={},
    )


def make_single_triangle_ds(
    with_data: bool = False,
) -> UgridDataset:
    """Build a minimal 1-face triangular UgridDataset with no CRS."""
    fnc = Connectivity(
        data=np.array([[0, 1, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5]),
        node_y=np.array([0.0, 0.0, 1.0]),
        face_node_connectivity=fnc,
    )
    data_variables: dict[str, MeshVariable] = {}
    if with_data:
        data_variables["depth"] = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(1,),
            _data=np.array([5.0]),
        )
    return UgridDataset(
        mesh=mesh,
        data_variables=data_variables,
        global_attributes={},
    )


@pytest.fixture(scope="session")
def ugrid_convention_nc_path():
    """Path to the UGRID convention NC UGRID test file.

    Returns:
        Path: Absolute path to ugrid.nc.
    """
    return Path("tests/data/netcdf/ugrid/ugrid.nc")


@pytest.fixture
def triangle_mesh():
    """Minimal 2-face triangular mesh.

    Layout::

         2---3
        / \\ / \\
       0---1---4

    Returns:
        Mesh2d: 5 nodes, 2 triangular faces.
    """
    node_x = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    faces = np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp)
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


@pytest.fixture
def mixed_mesh():
    """3-face mixed mesh: 1 quad + 2 triangles.

    Layout::

       3---4---5
       |   |  /
       0---1-2

    Returns:
        Mesh2d: 6 nodes, 3 faces (1 quad, 2 triangles).
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    faces = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, -1],
            [1, 5, 4, -1],
        ],
        dtype=np.intp,
    )
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure, and drop every animation, after each test here.

    The plot-marked tests build glyphs and animations, and matplotlib keeps both alive until
    they are released - which surfaced as `RuntimeWarning: More than 20 figures have been
    opened` from cleopatra, and as `Animation was deleted without rendering anything` landing
    on whichever unrelated test happened to be running when the collector got to it.

    `plt.close("all")` alone fixed only the first: a `FuncAnimation` is held by
    `Animation._all_anims`, so it outlives its figure and warns on collection. Clearing that
    list and collecting here means an animation a test discarded is accounted for inside that
    test.
    """
    yield
    if plt is None:  # pragma: no cover - matplotlib is a viz-extra dependency
        return
    pending = getattr(animation.Animation, "_all_anims", None)
    if pending:
        # Only when a test actually built one. Clearing the registry and collecting is what
        # stops the deletion warning landing on a later test, but doing it unconditionally
        # after every test in the package took the suite from ~35s to ~300s.
        pending.clear()
        gc.collect()
    plt.close("all")
