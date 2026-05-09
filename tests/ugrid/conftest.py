"""Shared fixtures for UGRID tests.

Provides reusable mesh topologies, connectivity arrays, and
test file paths used across multiple test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable


@pytest.fixture(scope="session")
def ugrid_convention_nc_path():
    """Path to the UGRID convention NC UGRID test file.

    Returns:
        Path: Absolute path to ugrid.nc.
    """
    p = Path("tests/data/netcdf/ugrid/ugrid.nc")
    if not p.exists():
        pytest.skip("UGRID convention NC test file not available")
    return p


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
