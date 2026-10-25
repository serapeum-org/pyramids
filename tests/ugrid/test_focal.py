"""k-ring focal statistics over the mesh face adjacency (issue #1376).

The raster ``focal_*`` family aggregates over a ``(2*radius+1)`` pixel window, which has no
mesh equivalent. The mesh analogue is the k-ring face neighbourhood -- hence ``rings``, not
``radius``. These tests hand-compute every expected value on a four-face strip, where the
neighbourhoods are small enough to check by eye.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.focal import neighbourhood

pytestmark = pytest.mark.core

# A four-face strip: faces 0-1-2-3 in a chain, so the 1-ring neighbourhoods are
# {0,1}, {0,1,2}, {1,2,3}, {2,3} - small enough to verify by hand.
_NODE_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def _strip(values, nodata=None, active=None):
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values, dtype="float64")},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        ds["v"].nodata = nodata
    if active is not None:
        ds["v"].active = np.asarray(active, dtype=bool)
    return ds


class TestNeighbourhood:
    """The k-ring reach matrix the aggregations are built on."""

    def test_one_ring_includes_the_centre(self):
        """The centre face is in its own neighbourhood, as a raster window includes its centre."""
        reach = neighbourhood(_strip([0.0] * 4).mesh, 1).toarray().astype(int)
        assert reach.tolist() == [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]

    def test_two_rings_extends_by_one_hop(self):
        reach = neighbourhood(_strip([0.0] * 4).mesh, 2).toarray().astype(int)
        assert reach[0].tolist() == [1, 1, 1, 0]
        assert reach[1].tolist() == [1, 1, 1, 1]

    def test_rings_below_one_is_refused(self):
        mesh = _strip([0.0] * 4).mesh
        with pytest.raises(ValueError, match="at least 1"):
            neighbourhood(mesh, 0)


class TestFocalMean:
    """Hand-computed means over the strip."""

    def test_one_ring(self):
        """mean{0,10}=5, mean{0,10,20}=10, mean{10,20,30}=20, mean{20,30}=25."""
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_mean().tolist() == pytest.approx(
            [5.0, 10.0, 20.0, 25.0]
        )

    def test_boundary_faces_use_only_what_they_have(self):
        """No padding: face 0 averages over two faces, not a fabricated three."""
        result = _strip([0.0, 10.0, 20.0, 30.0]).focal_mean()
        assert result[0] == pytest.approx(5.0), "padding would have skewed this"

    def test_two_rings(self):
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_mean(2).tolist() == pytest.approx(
            [10.0, 15.0, 15.0, 20.0]
        )

    def test_a_constant_field_is_unchanged(self):
        assert _strip([7.0] * 4).focal_mean().tolist() == pytest.approx([7.0] * 4)

    def test_nodata_neighbours_are_excluded(self):
        """Face 1's neighbourhood is {0,1,2}; with face 0 nodata it averages {10,20}."""
        ds = _strip([-999.0, 10.0, 20.0, 30.0], nodata=-999.0)
        assert ds.focal_mean()[1] == pytest.approx(15.0)

    def test_an_all_nodata_neighbourhood_is_nan(self):
        ds = _strip([-999.0, -999.0, 20.0, 30.0], nodata=-999.0)
        assert np.isnan(ds.focal_mean()[0]), "nothing usable to average"

    def test_inactive_neighbours_are_excluded(self):
        """A dry neighbour should not drag a wet face's mean down."""
        ds = _strip([0.0, 10.0, 20.0, 30.0], active=[False, True, True, True])
        assert ds.focal_mean()[1] == pytest.approx(15.0)

    def test_refuses_a_temporal_variable(self):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"v": np.zeros((3, 4))},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError, match="one value per face"):
            ds.focal_mean()


class TestFocalStd:
    """Population standard deviation over the same neighbourhoods."""

    def test_one_ring(self):
        """std{0,10}=5, std{0,10,20}=8.165 (population, not sample)."""
        result = _strip([0.0, 10.0, 20.0, 30.0]).focal_std()
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(np.std([0.0, 10.0, 20.0]))

    def test_a_constant_neighbourhood_is_exactly_zero(self):
        """The sums identity can land a hair below zero on a constant field; it is clamped,
        so the square root never sees a negative."""
        result = _strip([7.0] * 4).focal_std()
        assert result.tolist() == pytest.approx([0.0] * 4)
        assert not np.isnan(result).any()

    def test_all_nodata_is_nan(self):
        ds = _strip([-999.0, -999.0, -999.0, -999.0], nodata=-999.0)
        assert np.isnan(ds.focal_std()).all()


class TestFocalApply:
    """An arbitrary aggregation over the same neighbourhoods."""

    def test_peak_to_peak(self):
        """ptp{0,10}=10, ptp{0,10,20}=20, ptp{10,20,30}=20, ptp{20,30}=10."""
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_apply(
            np.ptp
        ).tolist() == pytest.approx([10.0, 20.0, 20.0, 10.0])

    def test_median(self):
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_apply(np.median)[
            1
        ] == pytest.approx(10.0)

    def test_matches_focal_mean_when_given_mean(self):
        """The arbitrary path and the vectorised one must agree on the same statistic."""
        ds = _strip([0.0, 10.0, 20.0, 30.0])
        assert ds.focal_apply(np.mean).tolist() == pytest.approx(
            ds.focal_mean().tolist()
        )

    def test_func_never_sees_an_empty_array(self):
        """An all-nodata neighbourhood yields NaN without calling func."""
        ds = _strip([-999.0, -999.0, -999.0, -999.0], nodata=-999.0)

        def explode(_values):
            raise AssertionError("func must not be called on an empty neighbourhood")

        assert np.isnan(ds.focal_apply(explode)).all()


class TestKernelGuards:
    """The kernel's own validation, which the facade pre-empts from above."""

    def test_a_mismatched_active_mask_is_refused(self):
        """The facade always derives the mask from the variable, so only a direct call can
        pass one of the wrong length - and silently broadcasting it would mask the wrong
        faces."""
        from pyramids.netcdf.ugrid.focal import focal_mean

        mesh = _strip([0.0, 10.0, 20.0, 30.0]).mesh
        with pytest.raises(ValueError, match="does not match the values"):
            focal_mean(
                mesh,
                np.array([0.0, 10.0, 20.0, 30.0]),
                active=np.array([True, False]),
            )

    def test_a_wrong_length_value_array_is_refused(self):
        from pyramids.netcdf.ugrid.focal import focal_mean

        mesh = _strip([0.0, 10.0, 20.0, 30.0]).mesh
        with pytest.raises(ValueError, match="one value per face"):
            focal_mean(mesh, np.array([1.0, 2.0]))


class TestNeighbourhoodCost:
    """`rings` has no upper bound, so the cost must at least be visible (review M11)."""

    @staticmethod
    def _strip(faces):
        """A chain of `faces` triangles, where ring k reaches about 2k faces."""
        count = faces + 2
        node_x = np.arange(float(count))
        node_y = np.array([float(index % 2) for index in range(count)])
        connectivity = np.array(
            [[index, index + 1, index + 2] for index in range(faces)]
        )
        return UgridDataset.from_arrays(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=connectivity,
            data={"v": np.zeros(faces)},
            data_locations={"v": "face"},
        ).mesh

    def test_a_small_mesh_never_warns(self):
        """A fully dense 4x4 matrix is 16 bools; warning about it would be noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            neighbourhood(_strip([0.0, 10.0, 20.0, 30.0]).mesh, 3)

    def test_a_large_dense_reach_warns(self):
        """Where it actually matters: the matrix stops being sparse on a real mesh."""
        mesh = self._strip(240)
        with pytest.warns(UserWarning, match="no longer sparse"):
            neighbourhood(mesh, 60)

    def test_a_large_sparse_reach_does_not_warn(self):
        """Size alone is not a reason to warn - a 1-ring reach on a big mesh is fine."""
        mesh = self._strip(240)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            neighbourhood(mesh, 1)

    def test_a_precomputed_reach_is_reused(self):
        """`focal_mean` and `focal_std` each built their own; a sweep can now share one."""
        from pyramids.netcdf.ugrid.focal import focal_mean, focal_std

        ds = _strip([0.0, 10.0, 20.0, 30.0])
        reach = neighbourhood(ds.mesh, 1)
        values = np.array([0.0, 10.0, 20.0, 30.0])
        assert np.allclose(focal_mean(ds.mesh, values, reach=reach), ds.focal_mean()), (
            "the shared matrix must give the same answer"
        )
        assert np.allclose(focal_std(ds.mesh, values, reach=reach), ds.focal_std())


class TestFocalApplyContract:
    """`func` must return a scalar, and the error must say so (review N4)."""

    def test_a_non_scalar_return_names_func_and_the_contract(self):
        """Was numpy's "only 0-dimensional arrays can be converted to Python scalars", which
        mentions neither func nor the documented contract."""
        ds = _strip([0.0, 10.0, 20.0, 30.0])
        with pytest.raises(TypeError, match="func must return one scalar"):
            ds.focal_apply(lambda neighbourhood: neighbourhood)

    def test_the_message_suggests_a_reducing_function(self):
        ds = _strip([0.0, 10.0, 20.0, 30.0])
        with pytest.raises(TypeError, match="np.median|np.ptp"):
            ds.focal_apply(lambda neighbourhood: neighbourhood * 2.0)


class TestReachIsValidated:
    """A supplied `reach` must be checked, not trusted (round 2 M5, M6, L5).

    The three helpers failed differently on a bad matrix: `focal_mean` and `focal_std` raised
    numpy's `matmul: dimension mismatch`, and `focal_apply` returned a WRONG ANSWER - it reads
    `indices`/`indptr` positionally, so an oversized matrix was silently truncated to the
    first `n_face` rows.
    """

    @staticmethod
    def _mesh_and_values():
        ds = _strip([1.0, 3.0, 5.0, 7.0])
        return ds, ds.mesh, np.array([1.0, 3.0, 5.0, 7.0])

    @pytest.mark.parametrize("helper", ["mean", "std", "apply"])
    def test_a_wrong_sized_reach_is_refused_by_every_helper(self, helper):
        """Every focal helper refuses a reach matrix built for a different mesh.

        Args:
        helper: Which of the three focal helpers to call.
        """
        from scipy import sparse

        from pyramids.netcdf.ugrid.focal import focal_apply, focal_mean, focal_std

        _ds, mesh, values = self._mesh_and_values()
        wrong = sparse.identity(9, dtype=bool, format="csr")
        calls = {
            "mean": lambda: focal_mean(mesh, values, reach=wrong),
            "std": lambda: focal_std(mesh, values, reach=wrong),
            "apply": lambda: focal_apply(mesh, values, np.median, reach=wrong),
        }
        with pytest.raises(ValueError, match="reach matrix is"):
            calls[helper]()

    def test_the_message_names_both_shapes(self):
        from scipy import sparse

        from pyramids.netcdf.ugrid.focal import focal_mean

        _ds, mesh, values = self._mesh_and_values()
        wrong = sparse.identity(9, dtype=bool, format="csr")
        with pytest.raises(ValueError) as info:
            focal_mean(mesh, values, reach=wrong)
        message = str(info.value)
        assert "(9, 9)" in message, message
        assert "(4, 4)" in message, message

    def test_rings_alongside_reach_is_refused(self):
        """Honouring the matrix and ignoring `rings` is a contradiction the caller would
        discover much later."""
        from pyramids.netcdf.ugrid.focal import focal_mean

        _ds, mesh, values = self._mesh_and_values()
        reach = neighbourhood(mesh, 1)
        with pytest.raises(ValueError, match="given together with a precomputed reach"):
            focal_mean(mesh, values, rings=3, reach=reach)

    def test_a_non_csr_reach_is_accepted_by_focal_apply(self):
        """`focal_apply` needs CSR while the others only multiply; all three now take the
        same thing."""
        from scipy import sparse

        from pyramids.netcdf.ugrid.focal import focal_apply

        _ds, mesh, values = self._mesh_and_values()
        as_coo = sparse.coo_matrix(neighbourhood(mesh, 1))
        assert np.allclose(
            focal_apply(mesh, values, np.mean, reach=as_coo),
            focal_apply(mesh, values, np.mean),
        )


class TestReachReachableFromTheFacade:
    """The documented reuse must be usable without importing the kernel (round 2 M6)."""

    def test_the_facade_accepts_a_precomputed_reach(self):
        ds = _strip([0.0, 10.0, 20.0, 30.0])
        reach = neighbourhood(ds.mesh, 1)
        assert np.allclose(ds.focal_mean(reach=reach), ds.focal_mean())
        assert np.allclose(ds.focal_std(reach=reach), ds.focal_std())
        assert np.allclose(
            ds.focal_apply(np.median, reach=reach), ds.focal_apply(np.median)
        )

    def test_the_facade_refuses_a_wrong_sized_reach_too(self):
        from scipy import sparse

        ds = _strip([0.0, 10.0, 20.0, 30.0])
        wrong = sparse.identity(9, dtype=bool, format="csr")
        with pytest.raises(ValueError, match="reach matrix is"):
            ds.focal_mean(reach=wrong)
