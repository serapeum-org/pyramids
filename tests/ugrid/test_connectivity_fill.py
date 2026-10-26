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


class TestDtypeAndFillType:
    """The dtype and the type of the declared fill (round 1 H1, H2, L3)."""

    @pytest.mark.parametrize(
        ("dtype", "fill_value"),
        [(np.uint8, 255), (np.uint16, 65535), (np.int8, 127), (np.uint32, 4294967295)],
        ids=["uint8", "uint16", "int8", "uint32"],
    )
    def test_an_unsigned_table_normalises_instead_of_overflowing(
        self, dtype, fill_value
    ):
        """`-1` has no representation in an unsigned dtype, so the result is signed (H1).

        Test scenario:
            Writing the sentinel into the caller's dtype raised
            `OverflowError: Python integer -1 out of bounds for uint8` before the padding
            could be marked; narrowing back afterwards would have been worse, wrapping -1
            onto 255 so `fill_value` claimed -1 while the padding still read 255. 255 and
            65535 are the canonical fills for an unsigned table, and the UGRID guidance
            ("larger than the number of nodes") pushes writers towards such sentinels.
        """
        table = _conn(
            [[0, 1, 2], [1, 3, fill_value]], fill_value, "face_node_connectivity"
        )
        table = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, fill_value]], dtype=dtype),
            fill_value=fill_value,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )

        mesh = Mesh2d(NODE_X, NODE_Y, table)

        stored = mesh.face_node_connectivity
        assert stored.fill_value == -1
        assert np.issubdtype(np.asarray(stored.data).dtype, np.signedinteger), (
            f"the result must be signed, got {np.asarray(stored.data).dtype}"
        )
        np.testing.assert_array_equal(
            np.asarray(stored.data), np.array([[0, 1, 2], [1, 3, -1]])
        )

    @pytest.mark.parametrize(
        "fill_value",
        [999999.5, 2.5, -1.5, float("nan"), "999999", True],
        ids=["fractional", "small-fraction", "negative-fraction", "nan", "str", "bool"],
    )
    def test_a_non_integer_fill_is_refused(self, fill_value):
        """A fill that is not an integer cannot be compared against the table (H2, L3).

        Test scenario:
            `int(999999.5)` compared against a value that is not in the data, so the padding
            survived while `fill_value` was set to -1 -- the ambiguity this gate removes,
            recreated silently with the audit trail erased. `int(2.5)` also misreported the
            caller's value as "2", and `-1.5` slipped through the `== -1` early return
            entirely. `bool` is refused because `True` is not a node index even though it
            indexes like one.
        """
        table = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 3]], dtype=np.float64),
            fill_value=fill_value,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        with pytest.raises(TypeError, match="must be an integer or None"):
            Mesh2d(NODE_X, NODE_Y, table)


class TestRangeIsDerivedWhenUnknown:
    """`index_count == 0` must not license an unvalidated rewrite (round 1 H3)."""

    def test_a_face_edge_fill_is_refused_with_no_edge_table(self):
        """With no companion table the range comes from the table itself (H3).

        Test scenario:
            `edge_node_connectivity` is optional in UGRID while `face_edge_connectivity` is
            present, so `edge_count == 0` is legal input. The old guard `if index_count and
            0 <= fill < index_count` was dead code there -- `0 <= fill < 0` is unsatisfiable
            -- and the rewrite then ran against a fill nothing had validated, turning
            `[[0, 1], [1, 0]]` into `[[-1, 1], [1, -1]]` and destroying every reference to
            edge 0 with no error.
        """
        face_edges = _conn([[0, 1], [1, 0]], 0, "face_edge_connectivity")

        with pytest.raises(ValueError, match="valid edge index") as caught:
            Mesh2d(NODE_X, NODE_Y, _faces(), None, face_edges)

        message = str(caught.value)
        assert "face_edge_connectivity" in message
        assert "[0, 2)" in message, (
            f"the range must be the one derived from the table, got: {message}"
        )

    def test_an_out_of_range_fill_still_normalises_with_no_edge_table(self):
        """The derived range must not refuse a legitimately out-of-range fill (H3)."""
        face_edges = _conn([[0, 1], [1, 999]], 999, "face_edge_connectivity")

        mesh = Mesh2d(NODE_X, NODE_Y, _faces(), None, face_edges)

        stored = mesh.face_edge_connectivity
        assert stored is not None
        np.testing.assert_array_equal(
            np.asarray(stored.data), np.array([[0, 1], [1, -1]])
        )


class TestRefusalKeysOnOccurrence:
    """An in-range fill that never appears is not ambiguous (round 1 M1)."""

    def test_an_unused_in_range_fill_is_accepted(self):
        """A table with no padding is readable whatever its declared fill says (M1).

        Test scenario:
            The UGRID examples declare `_FillValue = 999999` on every connectivity
            unconditionally, padded or not. Refusing on the declared value alone meant a
            mesh with a million nodes could not be constructed at all over a sentinel it
            never uses. A fully-triangular table with `fill_value=2` has no ambiguity.
        """
        dense = _conn([[0, 1, 3], [1, 3, 0]], 2, "face_node_connectivity")

        mesh = Mesh2d(NODE_X, NODE_Y, dense)

        stored = mesh.face_node_connectivity
        assert stored.fill_value == -1, "the declared fill is still normalised away"
        np.testing.assert_array_equal(
            np.asarray(stored.data), np.array([[0, 1, 3], [1, 3, 0]])
        )

    def test_an_in_range_fill_that_does_occur_is_still_refused(self):
        """The refusal must not have been weakened into never firing (M1)."""
        padded = _conn([[0, 1, 2], [1, 3, 2]], 2, "face_node_connectivity")
        with pytest.raises(ValueError, match="valid node index"):
            Mesh2d(NODE_X, NODE_Y, padded)


class TestOtherNegatives:
    """A stray negative addresses a node from the wrong end (round 1 M2)."""

    def test_a_negative_that_is_not_the_fill_becomes_padding(self):
        """Every negative is normalised, not just the declared fill (M2).

        Test scenario:
            `Network1d._real_edges` already excludes all negatives, with a comment naming the
            hazard: a negative index addresses a node from the wrong end just as silently as
            the conventional fill. `Mesh2d` kept them, so `get_element(1)` returned
            `[1, 3, -5]` and `nodes_per_element()` counted -5 as a real node.
        """
        mixed = _conn(
            [[0, 1, 2], [1, 3, -5], [0, 3, 999]], 999, "face_node_connectivity"
        )

        mesh = Mesh2d(NODE_X, NODE_Y, mixed)

        np.testing.assert_array_equal(
            np.asarray(mesh.face_node_connectivity.data),
            np.array([[0, 1, 2], [1, 3, -1], [0, 3, -1]]),
        )
        assert mesh.face_node_connectivity.nodes_per_element().tolist() == [3, 2, 2]


class TestPassThroughInputs:
    """Inputs with no fill to normalise must survive untouched (round 1 regression guard)."""

    def test_a_bare_array_connectivity_is_left_alone(self):
        """`Network1d` accepts a raw ndarray, which has no `fill_value` at all.

        Test scenario:
            The annotation says `Connectivity`, but `Network1d(edge_node_connectivity=
            np.empty((0, 2)))` is what `tests/ugrid/test_graph_network_round2.py` passes, so
            the attribute lookup has to tolerate its absence. Narrowing that away broke the
            existing suite.
        """
        from pyramids.netcdf.ugrid.network import Network1d

        raw = np.empty((0, 2), dtype=np.intp)
        net = Network1d(np.array([]), np.array([]), raw)
        assert net.edge_node_connectivity is raw

    def test_a_none_fill_is_left_alone(self):
        """`Connectivity(fill_value=None)` declares a table with no sentinels."""
        no_fill = Connectivity(
            data=np.array([[0, 1], [1, 2]], dtype=np.intp),
            fill_value=None,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(NODE_X, NODE_Y, _faces(), no_fill)
        assert mesh.edge_node_connectivity is no_fill
