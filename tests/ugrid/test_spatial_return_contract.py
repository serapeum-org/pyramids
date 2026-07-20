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
        mesh, data_vars = clip_mesh(
            unit_square_dataset, box(-0.1, -0.1, 1.1, 2.1), touch=False
        )
        assert isinstance(mesh, Mesh2d), (
            f"first element should be a Mesh2d, got {type(mesh)}"
        )
        assert isinstance(data_vars, dict), (
            f"second element should be a dict, got {type(data_vars)}"
        )
        assert mesh.n_face == 2, f"expected 2 clipped faces, got {mesh.n_face}"
        assert "temperature" in data_vars, "the face variable should survive the clip"

    def test_subset_by_bounds_returns_mesh_and_data_vars(self, unit_square_dataset):
        """``subset_by_bounds`` returns a ``(Mesh2d, dict)`` pair.

        Test scenario:
            Full-cover bounds return all 4 faces as a mesh + data-variable dict pair.
        """
        mesh, data_vars = subset_by_bounds(unit_square_dataset, -1.0, -1.0, 3.0, 3.0)
        assert isinstance(mesh, Mesh2d), (
            f"first element should be a Mesh2d, got {type(mesh)}"
        )
        assert mesh.n_face == 4, f"expected all 4 faces, got {mesh.n_face}"
        assert isinstance(data_vars, dict), "second element should be a dict"


class TestEdgeRemapWithTopology:
    """Edge filtering + node/edge renumbering on a mesh WITH edge topology (test gap G2)."""

    def _two_triangle_dataset_with_edges(self):
        """Two triangles sharing an edge, with edge_node_connectivity and an edge variable.

        Returns:
            UgridDataset: nodes 0-3, faces ``[[0,1,2],[1,3,2]]``, edges
            ``[[0,1],[1,2],[2,0],[1,3],[3,2]]`` and an ``edge_flux`` edge variable
            ``[10,11,12,13,14]``.
        """
        from pyramids.netcdf.ugrid.connectivity import Connectivity
        from pyramids.netcdf.ugrid.dataset import UgridDataset
        from pyramids.netcdf.ugrid.mesh import Mesh2d
        from pyramids.netcdf.ugrid.models import MeshVariable

        edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.0, 1.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            edge_node_connectivity=Connectivity(
                data=edges,
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )
        edge_var = MeshVariable(
            name="edge_flux",
            location="edge",
            mesh_name="mesh",
            shape=(5,),
            _data=np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        )
        return UgridDataset(
            mesh=mesh, data_variables={"edge_flux": edge_var}, global_attributes={}
        )

    def test_edge_filter_renumber_and_slice(self):
        """Keeping one triangle drops non-incident edges, renumbers nodes, slices the edge var.

        Test scenario:
            Subset to face 0 (nodes 0,1,2). Only edges whose nodes all survive are kept —
            ``[0,1],[1,2],[2,0]`` (indices 0,1,2); ``[1,3]`` and ``[3,2]`` are dropped. The
            rebuilt edge_node_connectivity must reference only renumbered surviving nodes
            (all ``< n_node == 3``), and the ``edge_flux`` edge variable must slice to the
            kept edges ``[10, 11, 12]``. Pins the vectorized edge-remap rewrite (G2).
        """
        from pyramids.netcdf.ugrid.spatial import _subset_mesh_by_face_indices

        ds = self._two_triangle_dataset_with_edges()
        mesh, data_vars = _subset_mesh_by_face_indices(ds, [0])

        assert mesh.n_node == 3, f"expected 3 kept nodes, got {mesh.n_node}"
        assert mesh.edge_node_connectivity is not None, "edge topology must survive"
        enc = mesh.edge_node_connectivity.data
        assert enc.shape[0] == 3, f"expected 3 kept edges, got {enc.shape[0]}"
        assert (enc < mesh.n_node).all(), (
            f"edge nodes must be renumbered < n_node: {enc}"
        )
        np.testing.assert_array_equal(
            enc,
            np.array([[0, 1], [1, 2], [2, 0]]),
            err_msg="edge connectivity misremapped",
        )
        np.testing.assert_array_equal(
            np.asarray(data_vars["edge_flux"].data),
            np.array([10.0, 11.0, 12.0]),
            err_msg="edge variable not sliced to the kept edges",
        )


class TestEdgeVariableClip:
    """Edge-located variables on a mesh without edge topology (review H1)."""

    @pytest.fixture(scope="function")
    def square_with_edge_var(self):
        """A 2x2-cell unit-square dataset with an *edge*-located variable but no edge topology.

        Returns:
            UgridDataset: ``create_from_arrays`` builds only face_node_connectivity, so the
            mesh has no ``edge_node_connectivity`` — the H1 trigger condition.
        """
        from pyramids.netcdf.ugrid.dataset import UgridDataset

        return UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            face_node_connectivity=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
            ),
            data={"edge_flux": np.arange(12, dtype=np.float64)},
            data_locations={"edge_flux": "edge"},
        )

    def test_clip_drops_edge_variable_without_edge_topology(self, square_with_edge_var):
        """Clipping drops an edge variable when the mesh has no edge_node_connectivity (H1).

        Test scenario:
            The mesh built by ``create_from_arrays`` has no ``edge_node_connectivity``, so the
            edges that survive a clip are undeterminable. The edge-located ``edge_flux`` must be
            dropped (with a warning) rather than carried at full length onto the clipped mesh,
            which would leave the dataset internally inconsistent (edge var length != edge count).
        """
        assert square_with_edge_var.mesh.edge_node_connectivity is None, (
            "precondition: no edges"
        )
        with pytest.warns(UserWarning, match="no edge_node_connectivity"):
            mesh, data_vars = clip_mesh(
                square_with_edge_var, box(-0.1, -0.1, 1.1, 2.1), touch=False
            )
        assert mesh.n_face == 2, f"expected 2 clipped faces, got {mesh.n_face}"
        assert "edge_flux" not in data_vars, (
            "edge variable must be dropped (not carried unsliced) when the mesh has no "
            f"edge topology; got data_vars keys {list(data_vars)}"
        )


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
        assert not offenders, (
            f"spatial must not import ugrid.dataset; found: {offenders}"
        )
