"""Unit tests for pyramids.netcdf.ugrid.network.Network1d.

Covers construction from arrays, topology/geometry properties (node/edge
counts, bounds, edge lengths and midpoints), the network sort -- over stored
edge directions and over an orientation derived from a named outlet --
topology subsetting, and the GeoDataFrame export (one LineString per edge).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.network import Network1d

pytestmark = pytest.mark.core

#: A straight 4-node chain. Acyclic and orientable, so its traversal order is a property of
#: the topology rather than of the `directed` flag it happens to be built with.
CHAIN = (
    np.array([0.0, 1.0, 2.0, 3.0]),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([[0, 1], [1, 2], [2, 3]]),
)


@pytest.fixture
def y_network():
    """A small Y-shaped river network (two headwaters -> confluence -> outlet).

    Layout (flow downward)::

        0       1       (headwaters)
         \\     /
          \\   /
            2           (confluence)
            |
            3           (outlet)

    Returns:
        Network1d: 4 nodes, 3 directed edges oriented downstream.
    """
    node_x = np.array([0.0, 2.0, 1.0, 1.0])
    node_y = np.array([2.0, 2.0, 1.0, 0.0])
    edges = np.array([[0, 2], [1, 2], [2, 3]], dtype=np.intp)
    return Network1d.from_arrays(node_x, node_y, edges, directed=True, epsg=4326)


class TestConstruction:
    """Construction and basic topology counts."""

    def test_from_arrays_counts(self, y_network):
        """from_arrays yields the expected node and edge counts."""
        assert y_network.n_node == 4, f"expected 4 nodes, got {y_network.n_node}"
        assert y_network.n_edge == 3, f"expected 3 edges, got {y_network.n_edge}"

    def test_directed_flag(self, y_network):
        """The directed flag is carried through from_arrays."""
        assert y_network.directed is True

    def test_crs_from_epsg(self, y_network):
        """An epsg argument builds a usable CRS."""
        assert y_network.crs is not None
        assert y_network.crs.to_epsg() == 4326

    def test_edge_node_connectivity_type(self, y_network):
        """The edge connectivity is a Connectivity wrapper of shape (n_edge, 2)."""
        enc = y_network.edge_node_connectivity
        assert isinstance(enc, Connectivity)
        assert enc.data.shape == (3, 2)

    def test_no_n_face_anywhere(self, y_network):
        """A 1D network does not advertise `n_face` to any probe (issue #1352).

        Test scenario:
            The class used to carry an `n_face` property whose body only raised, which left
            `dir()` and `hasattr(Network1d, "n_face")` advertising a member that could never
            be read while `hasattr(net, "n_face")` denied it. With the property gone every
            probe agrees, and the counts a network does have stay visible.
        """
        assert hasattr(y_network, "n_face") is False, (
            "an instance must not advertise n_face"
        )
        assert hasattr(Network1d, "n_face") is False, (
            "the class must not advertise n_face either -- a property would, because "
            "class access never invokes its getter"
        )
        assert "n_face" not in dir(y_network), "n_face must not be listed by dir()"
        assert "n_face" not in vars(Network1d), "no n_face descriptor on the class"
        assert getattr(y_network, "n_face", "absent") == "absent"
        # The hint lives in `__getattr__`, which Python consults only after normal lookup
        # fails -- so the attribute stays absent to every probe while the guidance survives.
        with pytest.raises(AttributeError, match="has no faces") as caught:
            _ = y_network.n_face
        assert "n_node" in str(caught.value), (
            f"the message must still name the counts a network has, got: {caught.value}"
        )

    def test_the_node_and_edge_counts_are_advertised(self, y_network):
        """The counts a network does have stay visible to the same probe."""
        assert hasattr(y_network, "n_node") is True
        assert hasattr(y_network, "n_edge") is True


class TestGeometry:
    """Derived geometric properties."""

    def test_bounds(self, y_network):
        """bounds spans the full node extent."""
        assert y_network.bounds == (0.0, 0.0, 2.0, 2.0)

    def test_edge_lengths(self, y_network):
        """Edge lengths match the Euclidean node distances."""
        expected = np.array([np.sqrt(2.0), np.sqrt(2.0), 1.0])
        np.testing.assert_array_almost_equal(
            y_network.edge_lengths, expected, err_msg="edge lengths wrong"
        )

    def test_edge_midpoints(self, y_network):
        """Edge midpoints are the mean of each edge's node coordinates."""
        mx, my = y_network.edge_midpoints
        np.testing.assert_array_almost_equal(mx, np.array([0.5, 1.5, 1.0]))
        np.testing.assert_array_almost_equal(my, np.array([1.5, 1.5, 0.5]))

    def test_get_edge_coords(self, y_network):
        """get_edge_coords returns the start and end node coordinates."""
        start, end = y_network.get_edge_coords(2)
        np.testing.assert_array_equal(start, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(end, np.array([1.0, 0.0]))

    def test_kd_trees_build(self, y_network):
        """Node and edge KD-trees build and have the right sizes."""
        assert y_network.node_tree.n == 4
        assert y_network.edge_tree.n == 3


class TestSort:
    """Directed network traversal / topological sort."""

    def test_downstream_order(self, y_network):
        """Downstream sort puts headwaters first and the outlet last."""
        order = y_network.sort("downstream")
        assert set(order.tolist()) == {0, 1, 2, 3}
        # headwaters (0, 1) before the confluence (2), confluence before outlet (3)
        pos = {int(n): i for i, n in enumerate(order)}
        assert pos[0] < pos[2]
        assert pos[1] < pos[2]
        assert pos[2] < pos[3]
        assert order[-1] == 3, "outlet must be last downstream"

    def test_upstream_order(self, y_network):
        """Upstream sort reverses the flow: the outlet comes first."""
        order = y_network.sort("upstream")
        assert order[0] == 3, "outlet must be first upstream"
        pos = {int(n): i for i, n in enumerate(order)}
        assert pos[3] < pos[2]
        assert pos[2] < pos[0]
        assert pos[2] < pos[1]

    def test_sort_undirected_without_outlet_names_the_candidates(self):
        """An undirected network with no outlet is refused, but actionably (issue #1353).

        Test scenario:
            The message used to offer only `directed=True`, which meant rebuilding the object
            from the identical arrays. It now names the degree-1 nodes that are candidate
            outlets, because a network stores no elevation and the choice is the caller's.
        """
        net = Network1d.from_arrays(*CHAIN)
        with pytest.raises(ValueError) as caught:
            net.sort()
        message = str(caught.value)
        assert "undirected" in message
        assert "sort(outlet=" in message, "the message must name the way through"
        assert "Candidate outlets" in message
        assert "0, 3" in message, (
            f"both ends of the chain are degree-1 candidates, got {message!r}"
        )

    @pytest.mark.parametrize(
        ("outlet", "expected"),
        [(3, [0, 1, 2, 3]), (0, [3, 2, 1, 0])],
        ids=["outlet-at-the-end", "outlet-at-the-start"],
    )
    def test_sort_undirected_orients_from_the_outlet(self, outlet, expected):
        """Naming the outlet orients the edges and the sort proceeds (issue #1353)."""
        net = Network1d.from_arrays(*CHAIN)
        order = net.sort(outlet=outlet).tolist()
        assert order == expected, (
            f"every edge should run towards node {outlet}, giving {expected}, got {order}"
        )

    def test_sort_undirected_upstream_reverses_the_inferred_orientation(self):
        """`upstream` reverses the derived edges, as it does the stored ones."""
        net = Network1d.from_arrays(*CHAIN)
        assert net.sort("upstream", outlet=3).tolist() == [3, 2, 1, 0]

    def test_sort_undirected_matches_the_directed_answer(self):
        """The inferred orientation gives the same order the directed network does.

        Test scenario:
            The whole objection in #1353 is that `directed` records how the edges were
            labelled, not whether an order exists. Same nodes, same edges, one built
            directed and one oriented from its outlet -- the answers must agree.
        """
        inferred = Network1d.from_arrays(*CHAIN).sort(outlet=3)
        declared = Network1d.from_arrays(*CHAIN, directed=True).sort()
        assert inferred.tolist() == declared.tolist()

    def test_sort_outlet_refused_on_a_directed_network(self):
        """A directed network already carries its orientation, so outlet= is refused."""
        net = Network1d.from_arrays(*CHAIN, directed=True)
        with pytest.raises(ValueError, match="already directed"):
            net.sort(outlet=3)

    @pytest.mark.parametrize("outlet", [-1, 4, 99], ids=["negative", "one-past", "far"])
    def test_sort_outlet_must_be_a_node_index(self, outlet):
        """An outlet outside the node range is refused by range, not by IndexError."""
        net = Network1d.from_arrays(*CHAIN)
        with pytest.raises(ValueError, match=r"outlet must be a node index"):
            net.sort(outlet=outlet)

    def test_sort_refuses_an_edge_with_no_defensible_direction(self):
        """An odd cycle leaves one edge equidistant from the outlet, so it is refused.

        Test scenario:
            In a triangle, nodes 1 and 2 are both one hop from outlet 0, so nothing decides
            which way the edge between them runs. That is the genuine ambiguity #1353 says
            should still raise -- and the message names the edge and the distance.
        """
        triangle = Network1d.from_arrays(
            np.array([0.0, 1.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
            np.array([[0, 1], [1, 2], [2, 0]]),
        )
        with pytest.raises(ValueError) as caught:
            triangle.sort(outlet=0)
        message = str(caught.value)
        assert "nothing decides which way it runs" in message
        assert "hop(s) from outlet 0" in message

    def test_sort_orients_an_even_cycle(self):
        """A diamond is orientable: every edge spans two different distances.

        Test scenario:
            The guard above must refuse only genuine ambiguity, not every cycle. Both arms of
            a diamond run from the far node to the near one, so the orientation is defined and
            the outlet comes last.
        """
        diamond = Network1d.from_arrays(
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 1.0, -1.0, 0.0]),
            np.array([[0, 1], [0, 2], [1, 3], [2, 3]]),
        )
        order = diamond.sort(outlet=3).tolist()
        assert order[0] == 0, f"the far node drains first, got {order}"
        assert order[-1] == 3, f"the outlet is the sink, got {order}"

    def test_sort_refuses_a_component_the_outlet_cannot_reach(self):
        """An edge in another component has no distance to the outlet, so it is refused."""
        split = Network1d.from_arrays(
            np.array([0.0, 1.0, 5.0, 6.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [2, 3]]),
        )
        with pytest.raises(ValueError, match="cannot reach"):
            split.sort(outlet=0)

    def test_sort_bad_direction(self, y_network):
        """An unknown direction raises."""
        with pytest.raises(ValueError, match="downstream"):
            y_network.sort("sideways")

    def test_sort_detects_cycle(self):
        """A cyclic directed network cannot be sorted."""
        net = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [2, 0]]),
            directed=True,
        )
        with pytest.raises(ValueError, match="cycle"):
            net.sort()

    def test_the_message_says_so_when_no_node_is_a_candidate(self):
        """A network whose every node has two or more edges has no outlet to suggest.

        Test scenario:
            In a ring every node has degree 2, so the degree-1 heuristic finds nothing. The
            message has to say that rather than print an empty candidate list.
        """
        ring = Network1d.from_arrays(
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
        )
        with pytest.raises(ValueError) as caught:
            ring.sort()
        message = str(caught.value)
        assert "No degree-1 node to suggest" in message, (
            f"a ring has no leaf to offer, got:\n{message}"
        )
        assert "two or more edges" in message

    def test_the_candidate_list_is_truncated(self):
        """A hub with many spokes lists the first candidates and marks the rest elided.

        Test scenario:
            Ten leaves would make the message a wall of indices, so only the first eight are
            named and the list ends in an ellipsis.
        """
        spokes = 10
        hub_x = np.concatenate([[0.0], np.arange(1, spokes + 1, dtype="float64")])
        star = Network1d.from_arrays(
            hub_x,
            np.zeros(spokes + 1),
            np.array([[0, spoke] for spoke in range(1, spokes + 1)]),
        )
        with pytest.raises(ValueError) as caught:
            star.sort()
        message = str(caught.value)
        assert "1, 2, 3, 4, 5, 6, 7, 8, ..." in message, (
            f"eight candidates then an ellipsis, got:\n{message}"
        )

    def test_sort_steps_over_a_padded_row_when_undirected(self):
        """A fill row must not reach the degree census or the orientation (H1).

        Test scenario:
            `Connectivity.from_gdal_array` writes `-1` into every slot that held a file's
            `_FillValue`, so a driver-read network is padded. Reading the raw table put `-1`
            into `np.bincount`, which refuses negatives -- so the helper whose only job is to
            build a friendly message died inside numpy, and the caller never saw the
            candidate-outlet guidance at all.
        """
        padded = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [-1, -1]]),
        )
        with pytest.raises(ValueError) as caught:
            padded.sort()
        message = str(caught.value)
        assert "Candidate outlets" in message, (
            f"the padded row must not reach bincount, got:\n{message}"
        )
        assert "0, 2" in message, (
            f"nodes 0 and 2 are the degree-1 ends of the two real edges, got:\n{message}"
        )

    def test_sort_orients_a_padded_network(self):
        """A padded row is skipped rather than refused as an unreachable component (H1).

        Test scenario:
            The fill row used to be read as an edge joining nodes -1 and -1, which the
            reachability guard then reported as a component the outlet cannot reach -- a
            confident, wrong diagnosis naming an edge and nodes that do not exist. Node 3 is
            unattached here, so the two real edges orient towards node 2.
        """
        padded = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [-1, -1]]),
        )
        order = padded.sort(outlet=2).tolist()
        assert order[-1] == 2, f"the outlet is the sink, got {order}"
        assert order.index(0) < order.index(1) < order.index(2), (
            f"the chain must drain 0 -> 1 -> 2, got {order}"
        )

    def test_sort_directed_steps_over_a_padded_row(self):
        """The directed path skips a fill row instead of reporting a cycle (L9).

        Test scenario:
            `adjacency[-1].append(-1)` and `indegree[-1] += 1` corrupted the last node, so
            Kahn's algorithm could not place every node and the chain was reported as cyclic.
        """
        padded = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [2, 3], [-1, -1]]),
            directed=True,
        )
        assert padded.sort().tolist() == [0, 1, 2, 3], (
            "a padded row must not make an acyclic chain look cyclic"
        )

    @pytest.mark.parametrize(
        "edges",
        [
            [[0, 1], [1, 5]],
            [[0, 1], [9, 2]],
        ],
        ids=["out-of-range-target", "out-of-range-source"],
    )
    def test_sort_refuses_an_edge_index_past_the_node_count(self, edges):
        """An index beyond the node count is named, not leaked as IndexError (M4).

        Test scenario:
            The adjacency list is sized `n_node`, so `neighbours[5]` on a 3-node network
            raised `IndexError` -- a type the documented contract does not promise. The
            degree census had the matching defect, reporting node 5 as a candidate outlet of
            a 3-node network because `np.bincount` widens past `minlength`.
        """
        net = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array(edges),
        )
        with pytest.raises(ValueError, match=r"outside the node range"):
            net.sort(outlet=0)
        with pytest.raises(ValueError, match=r"outside the node range"):
            net.sort()

    def test_sort_passes_an_isolated_node_through_as_a_source(self):
        """A node no edge names is admitted, in ascending order among the sources (M6).

        Test scenario:
            The docstring used to claim the outlet becomes "the only sink", which an isolated
            node falsifies -- it has out-degree 0 too. Refusing the network would be the other
            way to make the claim true, but the directed path already admits such nodes (see
            `test_sort_keeps_an_isolated_node`), so both paths now agree and say so.
        """
        with_isolated = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 9.0]),
            np.array([0.0, 0.0, 0.0, 9.0]),
            np.array([[0, 1], [1, 2]]),
        )
        order = with_isolated.sort(outlet=2).tolist()
        assert set(order) == {0, 1, 2, 3}, (
            f"every node must appear exactly once, got {order}"
        )
        assert order[-1] == 2, (
            f"the outlet is still the last of the oriented chain: {order}"
        )
        assert order.index(3) < order.index(1), (
            f"an isolated node ranks among the sources, got {order}"
        )

    def test_sort_on_an_edgeless_network_says_there_is_nothing_to_orient(self):
        """An edgeless network gets its own message, not the all-degree-2 one (M8).

        Test scenario:
            With no edges, `np.bincount` yields all-zero degrees, so the "no degree-1 node"
            branch fired and announced that every node has two or more edges -- the exact
            opposite of the truth. Every node has zero.
        """
        edgeless = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.empty((0, 2), dtype=int),
        )
        with pytest.raises(ValueError) as caught:
            edgeless.sort()
        message = str(caught.value)
        assert "no edges" in message, f"the edgeless case must say so, got:\n{message}"
        assert "two or more edges" not in message, (
            f"the all-degree-2 wording is the opposite of the truth here, got:\n{message}"
        )

    def test_sort_on_an_edgeless_network_returns_every_node(self):
        """An edgeless network is the all-isolated case, so every node passes through (M8)."""
        edgeless = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.empty((0, 2), dtype=int),
        )
        assert edgeless.sort(outlet=0).tolist() == [0, 1, 2], (
            "with no edge to orient, the nodes come back in index order"
        )

    @pytest.mark.parametrize("outlet", [True, False], ids=["True", "False"])
    def test_sort_refuses_a_bool_outlet(self, outlet):
        """A bool is not a node index, even though it indexes like one (M7).

        Test scenario:
            `0 <= int(outlet) < n` accepts `True` and silently orients towards node 1, which
            is a plausible-looking order for a node the caller never named.
        """
        net = Network1d.from_arrays(*CHAIN)
        with pytest.raises(ValueError, match="outlet must be an integer node index"):
            net.sort(outlet=outlet)

    @pytest.mark.parametrize("outlet", [2.9, 0.5, 3.0], ids=["2.9", "0.5", "3.0"])
    def test_sort_refuses_a_float_outlet(self, outlet):
        """A float is refused rather than truncated (M7).

        Test scenario:
            `int(2.9)` is 2, so the network was oriented towards node 2 for a caller who
            asked for something that is not a node index at all. Even `3.0`, which truncates
            exactly, is refused -- the signature says `int`.
        """
        net = Network1d.from_arrays(*CHAIN)
        with pytest.raises(ValueError, match="outlet must be an integer node index"):
            net.sort(outlet=outlet)

    def test_sort_accepts_a_numpy_integer_outlet(self):
        """A numpy integer is a node index and must keep working (M7).

        Test scenario:
            The type guard must not reject the integer types numpy hands back from
            `argmin`/`flatnonzero`, which is how a caller would most naturally pick an outlet.
        """
        net = Network1d.from_arrays(*CHAIN)
        assert net.sort(outlet=np.int64(3)).tolist() == [0, 1, 2, 3]


class TestTopologySubset:
    """Edge subsetting with node renumbering."""

    def test_subset_keeps_edges_and_renumbers(self, y_network):
        """Subsetting to one edge keeps 2 nodes renumbered to 0..1."""
        sub = y_network.topology_subset([2])
        assert sub.n_edge == 1
        assert sub.n_node == 2
        np.testing.assert_array_equal(
            sub.edge_node_connectivity.data, np.array([[0, 1]])
        )
        # nodes 2 and 3 of the parent become 0 and 1 in the subset
        np.testing.assert_array_equal(sub.node_x, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(sub.node_y, np.array([1.0, 0.0]))


class TestToGeoDataFrame:
    """Vector export."""

    def test_edges_to_linestrings(self, y_network):
        """to_geodataframe yields one LineString per edge with the network CRS."""
        gdf = y_network.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3, f"expected 3 edges, got {len(gdf)}"
        assert all(isinstance(g, LineString) for g in gdf.geometry)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326
        # the outlet edge (2 -> 3) runs from (1, 1) to (1, 0)
        np.testing.assert_array_almost_equal(
            np.array(gdf.geometry.iloc[2].coords),
            np.array([[1.0, 1.0], [1.0, 0.0]]),
        )

    def test_nodes_to_points(self, y_network):
        """to_geodataframe(location='node') yields one Point per node."""
        gdf = y_network.to_geodataframe(location="node")
        assert len(gdf) == 4
        assert all(isinstance(g, Point) for g in gdf.geometry)

    def test_bad_location(self, y_network):
        """An unknown location raises."""
        with pytest.raises(ValueError, match="edge.*node"):
            y_network.to_geodataframe(location="face")


class TestPaddedAndIsolatedTopology:
    """The topology shapes the clean Y fixture cannot reach.

    Every test above builds a network from a dense `(n_edge, 2)` table with no fill and no
    unattached node, so three behaviours went unpinned: what a padded row does to a subset,
    whether `sort` keeps a node no edge names, and which branch of `_edge_linestrings` runs.
    """

    @pytest.fixture
    def padded_network(self):
        """A chain of two edges plus a third row that is all fill."""
        return Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [-1, -1]]),
        )

    def test_a_padded_row_stays_a_fill_through_a_subset(self, padded_network):
        """Subsetting across a padded row does not turn it into an edge.

        Test scenario:
            `np.unique` keeps the `-1`, and numpy's negative indexing then addresses the
            *last* node -- so the padded row used to come back as a real edge joining it.
            Taking edge 0 and the padded edge 2 must give one real edge over two nodes, with
            the padded row still padded.
        """
        sub = padded_network.topology_subset([0, 2])

        assert sub.n_node == 2, (
            f"only the two nodes of edge 0 are used, got {sub.n_node}"
        )
        np.testing.assert_array_equal(
            np.asarray(sub.edge_node_connectivity.data),
            np.array([[0, 1], [-1, -1]]),
        )

    def test_the_filled_linestring_path_is_used(self, padded_network):
        """A connectivity carrying a fill takes the per-edge build, not the vectorised one.

        Test scenario:
            `_edge_linestrings` fast-paths a clean `(n_edge, 2)` table through
            `shapely.linestrings` and otherwise builds each edge from its own nodes. Only the
            fast path had ever run. The fallback keeps one row per declared edge, so edge
            variables stay aligned, and the padded row becomes an empty LineString -- it
            declares no edge, so it has no geometry.
        """
        frame = padded_network.to_geodataframe("edge")

        assert len(frame) == padded_network.n_edge == 3, (
            f"one row per declared edge, got {len(frame)}"
        )
        assert [len(geom.coords) for geom in frame.geometry] == [2, 2, 0], (
            f"the two real edges have two vertices and the padded row none, got "
            f"{[len(geom.coords) for geom in frame.geometry]}"
        )
        assert frame.geometry.iloc[2].is_empty, (
            "the padded row's geometry should be empty, not a degenerate line"
        )

    def test_the_vectorised_path_is_still_used_for_a_clean_table(self, y_network):
        """The contrast case, so the two branches are both covered."""
        frame = y_network.to_geodataframe("edge")

        assert [len(geom.coords) for geom in frame.geometry] == [2, 2, 2], (
            f"a clean table gives every edge two vertices, got "
            f"{[len(geom.coords) for geom in frame.geometry]}"
        )

    def test_sort_keeps_a_node_no_edge_names(self):
        """An isolated node appears in the ordering rather than being dropped.

        Test scenario:
            `sort` walks the edges, so a node the edge table never names has no
            predecessors and no successors. It is still a node of the network, and the
            ordering is documented as covering the nodes -- so dropping it would make
            `len(order) != n_node` and silently misalign any node-indexed array built from it.
        """
        isolated = Network1d.from_arrays(
            np.array([0.0, 1.0, 5.0]),
            np.array([0.0, 0.0, 5.0]),
            np.array([[0, 1]]),
            directed=True,
        )

        order = isolated.sort("downstream")

        assert len(order) == isolated.n_node == 3, (
            f"the ordering covers every node, got {len(order)} of {isolated.n_node}"
        )
        assert set(order.tolist()) == {0, 1, 2}, (
            f"the isolated node 2 must be in the ordering, got {order.tolist()}"
        )


class TestAmbiguousFillValue:
    """A fill value inside the node range makes the edge table unreadable (round 2 H1)."""

    @staticmethod
    def _connectivity(fill_value):
        """An edge table for a 4-node chain, padded with `fill_value`."""
        return Connectivity(
            data=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.intp),
            fill_value=fill_value,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )

    @pytest.mark.parametrize("fill_value", [0, 2, 3], ids=["first", "middle", "last"])
    def test_a_fill_inside_the_node_range_is_refused(self, fill_value):
        """Padding cannot be told from data when the fill is a legal index (H1).

        Test scenario:
            `_real_edges` masks any row containing the fill value. For `fill_value=2` the
            rows `[1, 2]` and `[2, 3]` are real edges, yet they look exactly like padding --
            so `sort()` dropped them and returned a silently wrong topological order,
            `[0, 2, 3, 1]` instead of `[0, 1, 2, 3]`, with no exception. A wrong river
            ordering is the worst failure this class has, because every routing step
            downstream consumes it unchecked. The ambiguity is refused at construction
            instead.
        """
        node_x = np.array([0.0, 1.0, 2.0, 3.0])
        node_y = np.array([0.0, 0.0, 0.0, 0.0])
        connectivity = self._connectivity(fill_value)

        with pytest.raises(ValueError, match="fill value") as caught:
            Network1d(node_x, node_y, connectivity)
        message = str(caught.value)
        assert str(fill_value) in message, f"the fill must be named, got: {message}"
        assert "valid node index" in message, (
            f"the message must say why the fill is the problem, got: {message}"
        )
        assert "negative fill value" in message, (
            f"the message must name a fill that works, got: {message}"
        )

    @pytest.mark.parametrize(
        "fill_value", [-1, -99, 4, 999], ids=["minus-one", "negative", "n", "far"]
    )
    def test_a_fill_outside_the_node_range_is_accepted(self, fill_value):
        """A fill that cannot collide with a node index stays usable.

        Test scenario:
            `-1` is what both in-repo constructors write, and any value at or beyond
            `n_node` is equally unambiguous. Neither may be refused, and the directed sort
            must give the same answer for all of them.
        """
        net = Network1d(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            self._connectivity(fill_value),
            directed=True,
        )
        assert net.sort().tolist() == [0, 1, 2, 3], (
            f"fill_value={fill_value} must not change the order"
        )

    def test_from_arrays_is_unaffected(self):
        """`from_arrays` normalises the fill to -1, so it can never trip the guard."""
        net = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [2, 3]]),
            directed=True,
        )
        assert net.edge_node_connectivity.fill_value == -1
        assert net.sort().tolist() == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "fill_value",
        [-99, 4, 999, 999999],
        ids=["negative", "n", "far", "ugrid-example"],
    )
    def test_an_out_of_range_fill_is_normalised_to_minus_one(self, fill_value):
        """An unambiguous fill is remapped on the way in, not carried (#1400).

        Test scenario:
            Every member that walks the table tests rows against the stored fill, so the
            answer depended on a number pyramids did not control. xugrid and uxarray avoid
            that by never carrying a foreign fill -- xugrid normalises to -1 ("This ensures
            we can always index"), uxarray to `np.iinfo(np.intp).min`. The UGRID spec
            prescribes no value and its own examples use `999999` and `-999`, so an
            out-of-range fill is legitimate input that has to be accepted and normalised.
        """
        padded = Connectivity(
            data=np.array([[0, 1], [1, 2], [fill_value, fill_value]], dtype=np.intp),
            fill_value=fill_value,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        net = Network1d(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 0.0]), padded
        )

        assert net.edge_node_connectivity.fill_value == -1, (
            f"the stored fill must be normalised, got "
            f"{net.edge_node_connectivity.fill_value}"
        )
        stored = np.asarray(net.edge_node_connectivity.data)
        assert fill_value not in stored.tolist() or fill_value == -1, (
            f"no occurrence of the original fill may survive in the table, got {stored}"
        )
        np.testing.assert_array_equal(stored[2], np.array([-1, -1]))
        assert net.real_edge_indices.tolist() == [0, 1], (
            "the padded row must still read as padding after normalisation"
        )

    def test_normalisation_leaves_the_real_indices_alone(self):
        """Only the fill is rewritten; genuine node indices are untouched (#1400)."""
        padded = Connectivity(
            data=np.array([[0, 1], [1, 2], [999999, 999999]], dtype=np.intp),
            fill_value=999999,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        net = Network1d(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 0.0]), padded
        )
        np.testing.assert_array_equal(
            np.asarray(net.edge_node_connectivity.data)[:2],
            np.array([[0, 1], [1, 2]]),
        )
        assert net.sort(outlet=2).tolist()[-1] == 2, "the network must still sort"

    def test_a_minus_one_fill_is_left_exactly_as_it_is(self):
        """The common case is not rewritten, so nothing in-repo changes (#1400).

        Test scenario:
            Both in-repo constructors already write -1. Normalisation must be a no-op for
            them -- same object, same array -- so no existing read pays for it.
        """
        original = Connectivity(
            data=np.array([[0, 1], [1, 2], [-1, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        net = Network1d(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 0.0]), original
        )
        assert net.edge_node_connectivity is original, (
            "a -1 fill needs no normalisation, so the Connectivity must pass through"
        )
