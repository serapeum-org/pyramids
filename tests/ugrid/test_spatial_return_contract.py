"""Regression tests for STR-3: the ugrid spatial subsetters' return contract.

``clip_mesh`` / ``subset_by_bounds`` / ``_subset_mesh_by_face_indices`` return the rebuilt
``(mesh, data_variables)`` rather than a ``UgridDataset``, so ``ugrid.spatial`` no longer
imports ``ugrid.dataset`` (the import cycle is broken). These pin that contract.
"""

import sys

import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import clip_mesh, subset_by_bounds

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def unit_square_dataset():
    """A 2x2-cell unit-square UGRID dataset with a face variable.

    Returns:
        UgridDataset: 9 nodes, 4 quad faces, one ``temperature`` face variable.
    """
    from pyramids.netcdf.ugrid.dataset import UgridDataset

    return UgridDataset.create_from_arrays(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        ),
        data={"temperature": np.arange(4, dtype=np.float64)},
        data_locations={"temperature": "face"},
    )


class TestSpatialReturnContract:
    """``clip_mesh`` / ``subset_by_bounds`` return ``(mesh, data_variables)`` (STR-3)."""

    def test_clip_mesh_returns_mesh_and_data_vars(self, unit_square_dataset):
        """``clip_mesh`` returns a ``(Mesh2d, dict)`` pair, not a dataset.

        Test scenario:
            Clipping the left column returns the rebuilt mesh and the sliced data-variable
            dict; the mesh has the expected reduced face count and the dict keys match.
        """
        mesh, data_vars = clip_mesh(unit_square_dataset, box(-0.1, -0.1, 1.1, 2.1), touch=False)
        assert isinstance(mesh, Mesh2d), f"first element should be a Mesh2d, got {type(mesh)}"
        assert isinstance(data_vars, dict), f"second element should be a dict, got {type(data_vars)}"
        assert mesh.n_face == 2, f"expected 2 clipped faces, got {mesh.n_face}"
        assert "temperature" in data_vars, "the face variable should survive the clip"

    def test_subset_by_bounds_returns_mesh_and_data_vars(self, unit_square_dataset):
        """``subset_by_bounds`` returns a ``(Mesh2d, dict)`` pair.

        Test scenario:
            Full-cover bounds return all 4 faces as a mesh + data-variable dict pair.
        """
        mesh, data_vars = subset_by_bounds(unit_square_dataset, -1.0, -1.0, 3.0, 3.0)
        assert isinstance(mesh, Mesh2d), f"first element should be a Mesh2d, got {type(mesh)}"
        assert mesh.n_face == 4, f"expected all 4 faces, got {mesh.n_face}"
        assert isinstance(data_vars, dict), "second element should be a dict"


class TestNoImportCycle:
    """``ugrid.spatial`` must not depend on ``ugrid.dataset`` (STR-3)."""

    def test_spatial_module_does_not_import_dataset_at_module_scope(self):
        """The spatial module's source contains no module-scope import of ugrid.dataset.

        Test scenario:
            Read the spatial module source and assert no top-level
            ``from pyramids.netcdf.ugrid.dataset import`` / ``import ... dataset`` statement
            exists outside docstrings — the cycle is broken structurally.
        """
        import pyramids.netcdf.ugrid.spatial as spatial_mod

        source = open(spatial_mod.__file__, encoding="utf-8").read()
        offenders = [
            line
            for line in source.splitlines()
            if line.lstrip().startswith(("import ", "from "))
            and "ugrid.dataset" in line
        ]
        assert not offenders, f"spatial must not import ugrid.dataset; found: {offenders}"
