"""Unit tests for _group_neighbours (BFS flood-fill) and cluster methods.

Tests cover: basic BFS correctness, 8-connectivity, large rasters that
would have exceeded the old recursion limit, edge cases (single cell,
entire array in bounds, no cells in bounds, boundary cells, diagonal-only
connections).
"""

import numpy as np
import pytest

from pyramids.dataset.dataset import Dataset
from pyramids.dataset._collaborators import Vectorize

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def make_dataset():
    """Factory fixture that creates an in-memory Dataset from a 2-D array.

    Returns:
        callable: Function accepting a numpy array and returning a Dataset.
    """

    def _make(arr: np.ndarray) -> Dataset:
        return Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        )

    return _make


class TestGroupNeighbours:
    """Tests for Vectorize._group_neighbours static method.

    Note on starting-cell behavior: _group_neighbours queues (i, j) but
    does not mark it.  When a discovered neighbor later checks *its* own
    neighbors it will find the starting cell still unmarked, mark it, and
    add it to position/values.  Therefore the starting cell appears in
    the output whenever it has at least one in-bound neighbor.
    """

    @staticmethod
    def _call(array, i, j, lower_bound, upper_bound, count=1):
        """Helper that calls _group_neighbours and returns position, values, cluster."""
        cluster = np.zeros(array.shape, dtype=float)
        position = []
        values = []
        Vectorize._group_neighbours(
            array, i, j, lower_bound, upper_bound, position, values, count, cluster
        )
        return position, values, cluster

    def test_single_cell_no_neighbours(self):
        """Test BFS on a 1x1 array produces no neighbours.

        Test scenario:
            A 1x1 array has no adjacent cells, so position and values should
            remain empty and the cluster array should stay all zeros.
        """
        arr = np.array([[5.0]])
        position, values, cluster = self._call(arr, 0, 0, 0, 10)
        assert position == [], f"Expected no neighbours, got {position}"
        assert values == [], f"Expected no values, got {values}"
        assert (
            cluster[0, 0] == 0
        ), "Starting cell should not be marked by _group_neighbours"

    def test_all_cells_in_bounds_3x3(self):
        """Test BFS floods the entire 3x3 array when all cells are in range.

        Test scenario:
            Starting from the center of a uniform 3x3 array, all 8
            direct neighbours are discovered, plus the starting cell is
            re-discovered by its neighbours (9 total).
        """
        arr = np.ones((3, 3), dtype=float) * 5
        position, values, cluster = self._call(arr, 1, 1, 1, 10)

        assert (
            len(position) == 9
        ), f"Expected 9 cells (8 neighbors + start), got {len(position)}"
        assert all(v == 5.0 for v in values), f"All values should be 5.0, got {values}"
        visited = {(r, c) for r, c in position}
        expected = {(r, c) for r in range(3) for c in range(3)}
        assert visited == expected, f"Expected all 9 cells, got {visited}"

    def test_no_cells_in_bounds(self):
        """Test BFS finds nothing when no neighbour is within bounds.

        Test scenario:
            All cells have value 1 but bounds are [5, 10], so nothing
            should be found.
        """
        arr = np.ones((3, 3), dtype=float)
        position, values, cluster = self._call(arr, 1, 1, 5, 10)

        assert position == [], f"Expected empty position, got {position}"
        assert values == [], f"Expected empty values, got {values}"

    def test_boundary_values_inclusive(self):
        """Test that lower and upper bounds are inclusive.

        Test scenario:
            Cells with exactly the lower_bound and upper_bound values
            should be included in the cluster.
        """
        arr = np.array([[2.0, 5.0], [8.0, 3.0]])
        position, values, cluster = self._call(arr, 0, 0, 2.0, 8.0)

        found_values = set(values)
        assert 5.0 in found_values, "Value within bounds should be found"
        assert 8.0 in found_values, "Value exactly at upper_bound should be found"
        assert 3.0 in found_values, "Value within bounds should be found"
        assert 2.0 in found_values, "Starting cell re-discovered at lower_bound"

    def test_diagonal_only_connection(self):
        """Test that diagonally-connected cells are discovered (8-connectivity).

        Test scenario:
            A checkerboard pattern where in-bound cells only touch
            diagonally. All should form a connected cluster including
            the starting cell re-discovered by its diagonal neighbours.
        """
        arr = np.array(
            [
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 5.0],
            ]
        )
        position, values, cluster = self._call(arr, 1, 1, 4, 6)

        found_positions = {(r, c) for r, c in position}
        expected = {(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)}
        assert (
            found_positions == expected
        ), f"Expected diagonal corners + center {expected}, got {found_positions}"

    def test_l_shaped_region(self):
        """Test BFS correctly follows an L-shaped connected region.

        Test scenario:
            An L-shaped region of in-bound cells; starting from (2, 0)
            which is pre-marked, BFS should find the remaining 4 cells.
        """
        arr = np.array(
            [
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ]
        )
        cluster_arr = np.zeros((3, 3), dtype=float)
        cluster_arr[2, 0] = 1
        position = []
        values = []
        Vectorize._group_neighbours(arr, 2, 0, 4, 6, position, values, 1, cluster_arr)

        found = {(r, c) for r, c in position}
        expected = {(0, 0), (1, 0), (2, 1), (2, 2)}
        assert found == expected, f"Expected L-shape cells {expected}, got {found}"

    def test_does_not_revisit_already_clustered_cells(self):
        """Test BFS skips cells that already have a non-zero cluster value.

        Test scenario:
            Pre-mark some cells in the cluster array; BFS should not
            overwrite them or append them to the results.
        """
        arr = np.ones((3, 3), dtype=float) * 5
        cluster_arr = np.zeros((3, 3), dtype=float)
        cluster_arr[0, 0] = 99
        cluster_arr[0, 1] = 99
        position = []
        values = []
        Vectorize._group_neighbours(arr, 1, 1, 1, 10, position, values, 1, cluster_arr)

        found = {(r, c) for r, c in position}
        assert (0, 0) not in found, "Cell (0,0) already clustered, should be skipped"
        assert (0, 1) not in found, "Cell (0,1) already clustered, should be skipped"
        expected_count = 7
        assert (
            len(found) == expected_count
        ), f"Expected {expected_count} cells (6 unmarked neighbors + start), got {len(found)}"

    def test_corner_start_position(self):
        """Test BFS starting from a corner cell discovers all cells.

        Test scenario:
            Starting from (0, 0) in a 3x3 uniform array, all 9 cells
            should be found (8 reachable + start re-discovered).
        """
        arr = np.ones((3, 3), dtype=float) * 5
        position, values, cluster = self._call(arr, 0, 0, 1, 10)

        assert (
            len(position) == 9
        ), f"All 9 cells should be reachable from corner, got {len(position)}"

    def test_edge_start_position(self):
        """Test BFS starting from an edge (non-corner) cell discovers all cells.

        Test scenario:
            Starting from (0, 1) in a 3x3 uniform array, all 9 cells
            should be found.
        """
        arr = np.ones((3, 3), dtype=float) * 5
        position, values, cluster = self._call(arr, 0, 1, 1, 10)

        assert (
            len(position) == 9
        ), f"All 9 cells should be reachable from edge, got {len(position)}"

    def test_cluster_number_assigned_correctly(self):
        """Test that discovered cells are marked with the correct cluster count.

        Test scenario:
            Pass count=7; all discovered cells should have cluster value 7.
        """
        arr = np.ones((3, 3), dtype=float) * 5
        position, values, cluster = self._call(arr, 1, 1, 1, 10, count=7)

        for r, c in position:
            assert (
                cluster[r, c] == 7
            ), f"Cell ({r},{c}) should have cluster=7, got {cluster[r, c]}"

    def test_values_match_array_contents(self):
        """Test that collected values match the actual array cell values.

        Test scenario:
            Array has distinct values per cell; verify each collected
            value matches its position in the array.
        """
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        position, values, cluster = self._call(arr, 1, 1, 1, 9)

        for (r, c), v in zip(position, values):
            assert v == arr[r, c], f"Value at ({r},{c}) should be {arr[r, c]}, got {v}"

    def test_two_disconnected_regions(self):
        """Test BFS only floods one connected component from the start cell.

        Test scenario:
            Two 2x2 blocks of value 5 separated by a 2-cell-wide gap of
            zeros. 8-connectivity cannot bridge a 2-wide gap, so BFS from
            the top-left block should not reach the bottom-right block.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0],
                [0.0, 0.0, 0.0, 5.0, 5.0],
            ]
        )
        position, values, cluster = self._call(arr, 0, 0, 4, 6)

        found = {(r, c) for r, c in position}
        top_left_block = {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert (
            found == top_left_block
        ), f"BFS should only reach top-left block {top_left_block}, got {found}"

    def test_diagonal_bridges_one_cell_gap(self):
        """Test that 8-connectivity bridges a 1-cell diagonal gap.

        Test scenario:
            Two 2x2 blocks of value 5 with only a diagonal connection
            through (1,1)-(2,2). All 8 cells should form one component.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 5.0],
                [0.0, 0.0, 5.0, 5.0],
            ]
        )
        position, values, cluster = self._call(arr, 0, 0, 4, 6)

        assert (
            len(position) == 8
        ), f"Diagonal bridge should connect both blocks (8 cells), got {len(position)}"

    def test_large_raster_no_recursion_error(self):
        """Test BFS handles a large connected region without hitting recursion limit.

        Test scenario:
            A 200x200 uniform array (40,000 cells) would exceed Python's
            default recursion limit of 1000. The iterative BFS must handle
            it without error.
        """
        arr = np.ones((200, 200), dtype=float) * 5
        position, values, cluster = self._call(arr, 100, 100, 1, 10)

        expected_count = 200 * 200
        assert (
            len(position) == expected_count
        ), f"Expected {expected_count} cells (all including start), got {len(position)}"

    def test_float_bounds_with_float_values(self):
        """Test BFS with non-integer float bounds and array values.

        Test scenario:
            Array has fractional values (0.1, 0.5, 0.9). Bounds [0.15, 0.85]
            should include 0.5 but exclude 0.1 and 0.9.
        """
        arr = np.array([[0.1, 0.5], [0.9, 0.5]])
        position, values, cluster = self._call(arr, 0, 1, 0.15, 0.85)

        found_positions = {(r, c) for r, c in position}
        assert (0, 0) not in found_positions, "0.1 is below lower_bound 0.15"
        assert (1, 0) not in found_positions, "0.9 is above upper_bound 0.85"
        assert (1, 1) in found_positions, "0.5 should be in bounds"

    def test_narrow_corridor(self):
        """Test BFS follows a 1-cell-wide corridor.

        Test scenario:
            A narrow S-shaped path of in-bound cells through an otherwise
            out-of-bound array. BFS should follow the entire corridor.
        """
        arr = np.zeros((5, 5), dtype=float)
        arr[0, 0] = 5.0
        arr[0, 1] = 5.0
        arr[0, 2] = 5.0
        arr[1, 2] = 5.0
        arr[2, 2] = 5.0
        arr[2, 1] = 5.0
        arr[2, 0] = 5.0
        position, values, cluster = self._call(arr, 0, 0, 4, 6)

        assert len(position) == 7, f"Expected 7 cells in corridor, got {len(position)}"

    def test_does_not_modify_input_array(self):
        """Test BFS does not mutate the input data array.

        Test scenario:
            After calling _group_neighbours, the original array should be
            unchanged.
        """
        arr = np.array([[5.0, 5.0], [5.0, 5.0]])
        arr_copy = arr.copy()
        self._call(arr, 0, 0, 1, 10)

        assert np.array_equal(arr, arr_copy), "Input array should not be modified"

    def test_starting_cell_out_of_bounds_value(self):
        """Test BFS when the starting cell value is outside the bounds.

        Test scenario:
            Start from a cell whose value is not in [lower, upper].
            BFS still enqueues the starting cell, but its in-bound
            neighbors will be found. The starting cell itself won't be
            re-marked because its value is out of bounds.
        """
        arr = np.array(
            [
                [0.0, 5.0],
                [5.0, 5.0],
            ]
        )
        position, values, cluster = self._call(arr, 0, 0, 4, 6)

        found = {(r, c) for r, c in position}
        assert (
            0,
            0,
        ) not in found, (
            "Starting cell has value 0 (out of bounds), should not be marked"
        )
        assert len(found) == 3, f"Expected 3 in-bound neighbors, got {len(found)}"


class TestCluster:
    """Tests for Dataset.cluster instance method."""

    @pytest.fixture
    def uniform_dataset(self, make_dataset):
        """Dataset where all cells have the same value.

        Returns:
            Dataset: 4x4 array filled with 5.0.
        """
        arr = np.ones((4, 4), dtype=np.float64) * 5
        return make_dataset(arr)

    @pytest.fixture
    def two_cluster_dataset(self, make_dataset):
        """Dataset with two clearly separated clusters (2-wide gap).

        Returns:
            Dataset: 5x5 array with two disconnected groups of value 5
            separated by a gap too wide for diagonal bridging.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0],
                [0.0, 0.0, 0.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        return make_dataset(arr)

    def test_return_types(self, uniform_dataset):
        """Test that cluster returns the correct types.

        Test scenario:
            Verify the four return values have the expected types:
            ndarray, int, list, list.
        """
        cluster_array, count, position, values = uniform_dataset.cluster(1, 10)

        assert isinstance(
            cluster_array, np.ndarray
        ), f"Expected np.ndarray, got {type(cluster_array)}"
        assert isinstance(count, int), f"Expected int, got {type(count)}"
        assert isinstance(position, list), f"Expected list, got {type(position)}"
        assert isinstance(values, list), f"Expected list, got {type(values)}"

    def test_uniform_array_single_cluster(self, uniform_dataset):
        """Test that a uniform array within bounds produces exactly one cluster.

        Test scenario:
            All 16 cells have value 5 and bounds are [1, 10], so there
            should be exactly 1 cluster containing all cells.
        """
        cluster_array, count, position, values = uniform_dataset.cluster(1, 10)

        assert (
            count == 2
        ), f"Expected count=2 (1 cluster + final increment), got {count}"
        assert len(position) == 16, f"Expected 16 positions, got {len(position)}"
        assert len(values) == 16, f"Expected 16 values, got {len(values)}"
        unique_clusters = set(cluster_array.flatten()) - {0}
        assert unique_clusters == {
            1
        }, f"Expected single cluster label {{1}}, got {unique_clusters}"

    def test_no_cells_in_bounds(self, make_dataset):
        """Test cluster with bounds that exclude all cells.

        Test scenario:
            Array values are all 5 but bounds are [10, 20], so no cells
            should be clustered.
        """
        arr = np.ones((3, 3), dtype=np.float64) * 5
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(10, 20)

        assert count == 1, f"Expected count=1 (no clusters found), got {count}"
        assert position == [], f"Expected empty positions, got {position}"
        assert values == [], f"Expected empty values, got {values}"
        assert np.all(cluster_array == 0), "All cells should be unassigned (0)"

    def test_two_disconnected_clusters(self, two_cluster_dataset):
        """Test correct identification of two separated clusters.

        Test scenario:
            Two 2x2 blocks of value 5 separated by a 2-wide zero gap.
            With bounds [4, 6], both blocks should be separate clusters.
        """
        cluster_array, count, position, values = two_cluster_dataset.cluster(4, 6)

        assert (
            count == 3
        ), f"Expected count=3 (2 clusters + final increment), got {count}"
        assert len(position) == 8, f"Expected 8 clustered cells, got {len(position)}"

        cluster_labels = set(cluster_array.flatten()) - {0}
        assert (
            len(cluster_labels) == 2
        ), f"Expected 2 distinct cluster labels, got {cluster_labels}"

    def test_diagonal_bridge_forms_single_cluster(self, make_dataset):
        """Test that a 1-cell diagonal gap connects two blocks into one cluster.

        Test scenario:
            Two 2x2 blocks with a diagonal connection at (1,1)-(2,2).
            8-connectivity means all 8 cells form a single cluster.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 5.0],
                [0.0, 0.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert count == 2, f"Expected count=2 (1 cluster + increment), got {count}"
        assert (
            len(position) == 8
        ), f"Expected 8 cells in single cluster, got {len(position)}"

    def test_isolated_single_cell(self, make_dataset):
        """Test that a single isolated cell forms its own cluster.

        Test scenario:
            Only the center cell of a 3x3 array is in bounds. It has no
            in-bound neighbours so it should still be detected.
        """
        arr = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert len(position) == 1, f"Expected 1 position, got {len(position)}"
        assert position[0] == [1, 1], f"Expected position [1,1], got {position[0]}"
        assert values == [5.0], f"Expected values [5.0], got {values}"
        assert (
            cluster_array[1, 1] == 1
        ), f"Center cell should have cluster=1, got {cluster_array[1, 1]}"

    def test_boundary_values_exact_match(self, make_dataset):
        """Test cells with values exactly at the bounds are included.

        Test scenario:
            Array has values [2, 5, 8]. Bounds are [2, 8]. All three
            values should be included.
        """
        arr = np.array([[2.0, 5.0, 8.0]], dtype=np.float64)
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(2.0, 8.0)

        assert len(position) == 3, f"Expected 3 cells, got {len(position)}"
        assert set(values) == {
            2.0,
            5.0,
            8.0,
        }, f"Expected {{2, 5, 8}}, got {set(values)}"

    def test_diagonal_connectivity(self, make_dataset):
        """Test that diagonally-connected cells belong to the same cluster.

        Test scenario:
            Checkerboard pattern where in-bound cells only touch
            diagonally. All should form one cluster.
        """
        arr = np.array(
            [
                [5.0, 0.0, 5.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        )
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert (
            count == 2
        ), f"Expected count=2 (1 diagonal cluster + increment), got {count}"
        assert (
            len(position) == 5
        ), f"Expected 5 cells in diagonal cluster, got {len(position)}"

    def test_cluster_array_shape_matches_input(self, make_dataset):
        """Test the returned cluster array has the same shape as the raster.

        Test scenario:
            A 5x7 array should produce a 5x7 cluster array.
        """
        arr = np.ones((5, 7), dtype=np.float64) * 3
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(1, 10)

        assert cluster_array.shape == (
            5,
            7,
        ), f"Expected shape (5, 7), got {cluster_array.shape}"

    def test_multiple_isolated_cells(self, make_dataset):
        """Test that multiple isolated in-bound cells each form their own cluster.

        Test scenario:
            Scatter individual in-bound cells at the corners of a 5x5
            array with no adjacent in-bound neighbours (2-wide gap
            prevents diagonal bridging).
        """
        arr = np.zeros((5, 5), dtype=np.float64)
        arr[0, 0] = 5.0
        arr[0, 4] = 5.0
        arr[4, 0] = 5.0
        arr[4, 4] = 5.0
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert (
            count == 5
        ), f"Expected count=5 (4 isolated clusters + increment), got {count}"
        assert len(position) == 4, f"Expected 4 positions, got {len(position)}"

    def test_large_raster_completes(self, make_dataset):
        """Test cluster on a large raster completes without recursion error.

        Test scenario:
            A 200x200 uniform array with all cells in bounds would exceed
            Python's recursion limit with the old recursive implementation.
            The iterative BFS should handle it.
        """
        arr = np.ones((200, 200), dtype=np.float64) * 5
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(1, 10)

        assert count == 2, f"Expected count=2 (1 cluster + increment), got {count}"
        assert len(position) == 40000, f"Expected 40000 cells, got {len(position)}"
        assert np.all(cluster_array == 1), "All cells should belong to cluster 1"

    def test_mixed_values_partial_clustering(self, make_dataset):
        """Test cluster with a mix of in-bound and out-of-bound values.

        Test scenario:
            Array has values 1-9; bounds [3, 7] should cluster only
            cells with values 3, 4, 5, 6, 7.
        """
        np.random.seed(42)
        arr = np.random.randint(1, 10, size=(6, 6)).astype(np.float64)
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(3, 7)

        for v in values:
            assert 3 <= v <= 7, f"Value {v} is outside bounds [3, 7]"

        in_bound_count = np.sum((arr >= 3) & (arr <= 7))
        assert (
            len(position) == in_bound_count
        ), f"Expected {in_bound_count} cells in bounds, got {len(position)}"

    def test_single_row_array(self, make_dataset):
        """Test cluster on a 1-row array.

        Test scenario:
            A single-row array [5, 5, 0, 5, 5] should produce two
            clusters separated by the zero.
        """
        arr = np.array([[5.0, 5.0, 0.0, 5.0, 5.0]], dtype=np.float64)
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert count == 3, f"Expected count=3 (2 clusters + increment), got {count}"
        assert len(position) == 4, f"Expected 4 cells, got {len(position)}"

    def test_single_column_array(self, make_dataset):
        """Test cluster on a 1-column array.

        Test scenario:
            A single-column array should cluster connected vertical cells.
        """
        arr = np.array([[5.0], [5.0], [0.0], [5.0]], dtype=np.float64)
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert count == 3, f"Expected count=3 (2 clusters + increment), got {count}"
        assert len(position) == 3, f"Expected 3 cells, got {len(position)}"

    def test_cluster_labels_are_sequential(self, make_dataset):
        """Test that cluster labels are assigned sequentially starting from 1.

        Test scenario:
            Three isolated cells should produce cluster labels 1, 2, 3
            in row-major scan order.
        """
        arr = np.zeros((5, 1), dtype=np.float64)
        arr[0, 0] = 5.0
        arr[2, 0] = 5.0
        arr[4, 0] = 5.0
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert (
            cluster_array[0, 0] == 1
        ), f"First cluster should be label 1, got {cluster_array[0, 0]}"
        assert (
            cluster_array[2, 0] == 2
        ), f"Second cluster should be label 2, got {cluster_array[2, 0]}"
        assert (
            cluster_array[4, 0] == 3
        ), f"Third cluster should be label 3, got {cluster_array[4, 0]}"

    def test_out_of_bound_cells_remain_zero(self, make_dataset):
        """Test that cells outside the value bounds stay 0 in the cluster array.

        Test scenario:
            A mix of in-bound and out-of-bound cells; verify out-of-bound
            cells have cluster value 0.
        """
        arr = np.array(
            [
                [5.0, 0.0],
                [0.0, 5.0],
            ],
            dtype=np.float64,
        )
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(4, 6)

        assert (
            cluster_array[0, 1] == 0
        ), f"Out-of-bound cell (0,1) should be 0, got {cluster_array[0, 1]}"
        assert (
            cluster_array[1, 0] == 0
        ), f"Out-of-bound cell (1,0) should be 0, got {cluster_array[1, 0]}"

    def test_position_values_correspondence(self, make_dataset):
        """Test that position and values lists are aligned.

        Test scenario:
            For each (position[i], values[i]) pair, the value should
            match the original array at that position.
        """
        np.random.seed(99)
        arr = np.random.uniform(1, 10, size=(4, 4)).astype(np.float64)
        dataset = make_dataset(arr)
        cluster_array, count, position, values = dataset.cluster(3, 7)

        for (r, c), v in zip(position, values):
            assert (
                v == arr[r, c]
            ), f"Value {v} at position ({r},{c}) doesn't match array value {arr[r, c]}"

    def test_cluster_with_real_dem(self, rhine_dem, clusters):
        """Test cluster against pre-computed golden results from a real DEM.

        Test scenario:
            Cluster values between 0.1 and 20 in the Rhine DEM and
            compare against stored expected results.
        """
        dataset = Dataset(rhine_dem)
        cluster_array, count, position, values = dataset.cluster(0.1, 20)

        assert count == 155, f"Expected 155 clusters, got {count}"
        assert np.array_equal(
            cluster_array, clusters
        ), "Cluster array does not match pre-computed golden result"
        assert len(position) == 2364, f"Expected 2364 positions, got {len(position)}"
        assert len(values) == 2364, f"Expected 2364 values, got {len(values)}"
