"""Connectivity fill normalisation on Mesh2d (issue #1400).

Every member that walks a connectivity table decides "is this row padding?" by comparing
against the stored fill value, which is whatever the file or the driver declared. A fill
outside the index range is unambiguous and is normalised to `-1` on the way in; a fill
inside it is also a legal index, so padding and data cannot be told apart and it is refused.

`Mesh2d` carries five tables addressing three different ranges -- nodes, edges and faces --
so each is checked against its own range rather than blindly against the node count.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.core

NODE_X = np.array([0.0, 1.0, 0.5, 1.5])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])


def _conn(data, fill_value, cf_role):
    """Wrap an index array as a Connectivity with the given fill."""
    return Connectivity(
        data=np.array(data, dtype=np.intp),
        fill_value=fill_value,
        cf_role=cf_role,
        original_start_index=0,
    )


def _faces(fill_value=-1):
    """A 2-face triangle table, optionally padded with `fill_value`."""
    return _conn([[0, 1, 2], [1, 3, 2]], fill_value, "face_node_connectivity")


class TestFaceNodeFill:
    """The table every mesh has, indexing nodes."""

    @pytest.mark.parametrize("fill_value", [0, 2, 3], ids=["first", "middle", "last"])
    def test_a_fill_inside_the_node_range_is_refused(self, fill_value):
        """A fill that is also a node index makes padding unreadable (#1400)."""
        faces = _faces(fill_value)

        with pytest.raises(ValueError, match="valid node index") as caught:
            Mesh2d(NODE_X, NODE_Y, faces)

        message = str(caught.value)
        assert "face_node_connectivity" in message, (
            f"the refusal must name the table, got: {message}"
        )
        assert str(fill_value) in message

    @pytest.mark.parametrize(
        "fill_value", [-99, 4, 999999], ids=["negative", "n", "ugrid-example"]
    )
    def test_an_out_of_range_fill_is_normalised(self, fill_value):
        """An unambiguous fill is rewritten to -1, as xugrid and uxarray do (#1400)."""
        mixed = _conn(
            [[0, 1, 2], [1, 3, fill_value]], fill_value, "face_node_connectivity"
        )
        mesh = Mesh2d(NODE_X, NODE_Y, mixed)

        stored = mesh.face_node_connectivity
        assert stored.fill_value == -1, f"fill not normalised: {stored.fill_value}"
        np.testing.assert_array_equal(
            np.asarray(stored.data), np.array([[0, 1, 2], [1, 3, -1]])
        )

    def test_a_minus_one_fill_passes_through_untouched(self):
        """The case every in-repo reader produces must not be copied (#1400)."""
        faces = _faces()
        mesh = Mesh2d(NODE_X, NODE_Y, faces)
        assert mesh.face_node_connectivity is faces


class TestOtherTablesUseTheirOwnRange:
    """A face_edge table indexes edges and a face_face table indexes faces, not nodes."""

    def test_a_face_edge_fill_is_checked_against_the_edge_count(self):
        """`fill_value=3` is refused on 4 edges even though it is also a node index.

        Test scenario:
            Checking every table against the node count would be wrong in both directions.
            Here the edge count is 4, so a fill of 3 is a legal *edge* index and must be
            refused, and the message has to say "edge" rather than "node".
        """
        edges = _conn([[0, 1], [1, 2], [2, 0], [1, 3]], -1, "edge_node_connectivity")
        face_edges = _conn([[0, 1, 2], [3, 1, -1]], 3, "face_edge_connectivity")

        with pytest.raises(ValueError, match="valid edge index") as caught:
            Mesh2d(NODE_X, NODE_Y, _faces(), edges, face_edges)

        assert "face_edge_connectivity" in str(caught.value)

    def test_a_face_edge_fill_above_the_edge_count_is_normalised(self):
        """A fill beyond the edge count is unambiguous, so it normalises (#1400).

        Test scenario:
            4 nodes but only 2 edges, and a fill of 3. Against the node count 3 would be
            in range and refused; against the edge count it is out of range and normalises.
            This is the case that proves the range is per-table.
        """
        edges = _conn([[0, 1], [1, 2]], -1, "edge_node_connectivity")
        face_edges = _conn([[0, 1], [1, 3]], 3, "face_edge_connectivity")

        mesh = Mesh2d(NODE_X, NODE_Y, _faces(), edges, face_edges)

        stored = mesh.face_edge_connectivity
        assert stored is not None
        assert stored.fill_value == -1
        np.testing.assert_array_equal(
            np.asarray(stored.data), np.array([[0, 1], [1, -1]])
        )

    def test_a_face_face_fill_is_checked_against_the_face_count(self):
        """`face_face_connectivity` indexes faces, so the message says "face"."""
        face_faces = _conn([[1, -1], [0, -1]], 1, "face_face_connectivity")

        with pytest.raises(ValueError, match="valid face index") as caught:
            Mesh2d(NODE_X, NODE_Y, _faces(), None, None, face_faces)

        assert "face_face_connectivity" in str(caught.value)

    def test_an_optional_table_left_as_none_stays_none(self):
        """Normalising must not materialise a table the caller did not pass."""
        mesh = Mesh2d(NODE_X, NODE_Y, _faces())
        assert mesh.edge_node_connectivity is None
        assert mesh.face_edge_connectivity is None
        assert mesh.face_face_connectivity is None
        assert mesh.edge_face_connectivity is None

    def test_geometry_still_works_after_normalisation(self):
        """The derived geometry reads the normalised table correctly (#1400).

        Test scenario:
            The point of normalising rather than refusing is that the mesh stays usable.
            A padded quad/triangle mix declared with `999999` must give the same centroids
            as the identical mesh declared with `-1`.
        """
        exotic = Mesh2d(
            NODE_X,
            NODE_Y,
            _conn([[0, 1, 2], [1, 3, 999999]], 999999, "face_node_connectivity"),
        )
        plain = Mesh2d(
            NODE_X,
            NODE_Y,
            _conn([[0, 1, 2], [1, 3, -1]], -1, "face_node_connectivity"),
        )
        np.testing.assert_allclose(
            np.asarray(exotic.face_centroids), np.asarray(plain.face_centroids)
        )
