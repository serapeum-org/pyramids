"""Value-range ``cluster`` and nodata-aware ``footprint`` (issue #1379).

Two raster operations that looked covered on the mesh side but were not:
``connected_components`` labels by topology alone and is not value-aware, and
``bounding_polygon`` is the mesh outline rather than the extent of the data.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

# A four-face strip, so faces 0 and 3 are NOT adjacent - which is what makes a
# two-pool cluster test meaningful.
_NODE_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def _strip(values, nodata=None):
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values)},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        ds["v"].nodata = nodata
    return ds


class TestCluster:
    """``cluster`` groups connected faces whose values fall in a range."""

    def test_separates_two_pools_that_topology_alone_cannot(self):
        """Regression (#1379): the whole point is being value-aware.

        Test scenario:
            All four faces form one topological component, so `connected_components` answers
            `[0, 0, 0, 0]`. With the middle two out of range, `cluster` must report two
            distinct pools instead.
        """
        ds = _strip([5.0, 0.0, 0.0, 7.0])
        assert ds.connected_components().tolist() == [0, 0, 0, 0]
        labels = ds.cluster(1.0, 10.0)["v"].data
        assert labels.tolist() == [1, 0, 0, 2]

    def test_excluded_faces_do_not_bridge_two_clusters(self):
        """Regression: masking the adjacency left explicit stored zeros, which csgraph still
        treated as edges — so the out-of-range faces silently joined the pools into one."""
        ds = _strip([5.0, 0.0, 0.0, 7.0])
        labels = ds.cluster(1.0, 10.0)["v"].data
        assert len(set(labels[labels > 0].tolist())) == 2, (
            "the two pools must stay separate, not be bridged by the excluded faces"
        )

    def test_all_in_range_is_one_cluster(self):
        assert _strip([1.0, 2.0, 3.0, 4.0]).cluster(0.0, 10.0)["v"].data.tolist() == [
            1,
            1,
            1,
            1,
        ]

    def test_none_in_range_labels_nothing(self):
        assert _strip([1.0, 2.0, 3.0, 4.0]).cluster(100.0, 200.0)[
            "v"
        ].data.tolist() == [0, 0, 0, 0]

    def test_bounds_are_inclusive(self):
        labels = _strip([1.0, 2.0, 3.0, 4.0]).cluster(2.0, 3.0)["v"].data
        assert labels.tolist() == [0, 1, 1, 0]

    def test_nodata_is_never_in_range(self):
        """A missing value cannot be inside a value range."""
        ds = _strip([-999.0, 2.0, 3.0, -999.0], nodata=-999.0)
        labels = ds.cluster(-1000.0, 1000.0)["v"].data
        assert labels.tolist() == [0, 1, 1, 0]

    def test_refuses_a_temporal_variable(self):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"v": np.zeros((3, 4))},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError, match="one value per face"):
            ds.cluster(0.0, 1.0)


class TestFootprint:
    """``footprint`` is the extent of the data, not the extent of the mesh."""

    def test_all_valid_matches_the_domain(self):
        ds = _strip([1.0, 2.0, 3.0, 4.0])
        assert ds.footprint().area == pytest.approx(ds.bounding_polygon().area)

    def test_nodata_shrinks_it_below_the_domain(self):
        """Regression (#1379): `bounding_polygon` describes the mesh, so it could never
        answer where the data actually is."""
        ds = _strip([1.0, -999.0, -999.0, 4.0], nodata=-999.0)
        footprint = ds.footprint()
        assert footprint.area < ds.bounding_polygon().area
        assert footprint.area == pytest.approx(2.0)

    def test_nan_counts_as_absent(self):
        ds = _strip([1.0, np.nan, np.nan, 4.0])
        assert ds.footprint().area < ds.bounding_polygon().area

    def test_no_valid_face_gives_an_empty_geometry(self):
        ds = _strip([-999.0, -999.0, -999.0, -999.0], nodata=-999.0)
        assert ds.footprint().is_empty

    def test_the_active_mask_does_not_shrink_it(self):
        """Settled in #1379: footprint excludes `nodata` only.

        Test scenario:
            A dry element has a real value and is still covered by the data, so excluding it
            would conflate "no data here" with "dry here" — the distinction #1331 preserves.
            Pinned so the behaviour cannot drift.
        """
        ds = _strip([1.0, 2.0, 3.0, 4.0])
        ds["v"].active = np.array([True, False, False, True])
        assert ds.footprint().area == pytest.approx(ds.bounding_polygon().area)

    def test_refuses_a_temporal_variable(self):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"v": np.zeros((3, 4))},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError, match="one value per face"):
            ds.footprint()


class TestClusterLabelsDeclareNoSentinel:
    """A label array must not inherit the source's nodata (review L7)."""

    def test_the_labels_carry_no_nodata(self):
        """`-9999` is representable as an integer label, so `_rewrap`'s existing guard kept
        it - declaring a no-data value no label array can ever hold, while the real "not in
        any cluster" marker is 0. Any `nodata_mask` on the labels would be wrong both ways.
        """
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array(
                [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
            ),
            data={"v": np.array([5.0, -9999.0, -9999.0, 7.0])},
            data_locations={"v": "face"},
        )
        ds["v"].nodata = -9999.0
        labels = ds.cluster(1, 10)["v"]
        assert labels.nodata is None, f"labels declared nodata={labels.nodata}"
        assert np.asarray(labels.data).tolist() == [1, 0, 0, 2]

    def test_the_source_variable_is_untouched(self):
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 1.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
            data={"v": np.array([5.0, 7.0])},
            data_locations={"v": "face"},
        )
        ds["v"].nodata = -9999.0
        ds.cluster(1, 10)
        assert ds["v"].nodata == -9999.0
