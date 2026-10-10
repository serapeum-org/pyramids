"""Tests for the UgridDataset along-time family (Phase 3B).

reduce / mean / sum / … / cumsum / shift / ffill / bfill / interpolate_na / rolling on an
in-memory mesh with a temporal face variable. These reuse the shared pure kernels in
``pyramids.base._reductions`` but are exercised through the mesh API.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


def _temporal_mesh(data: np.ndarray) -> UgridDataset:
    """Two-face mesh carrying one temporal variable ``d`` of shape (n_time, 2)."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": data},
        data_locations={"d": "face"},
    )


@pytest.fixture
def mesh() -> UgridDataset:
    return _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


class TestReduce:
    def test_mean_collapses_time(self, mesh):
        out = mesh.reduce("mean")
        assert out["d"].data.tolist() == [3.0, 4.0]
        assert out["d"].has_time is False

    def test_stat_wrappers(self, mesh):
        assert mesh.sum()["d"].data.tolist() == [9.0, 12.0]
        assert mesh.min()["d"].data.tolist() == [1.0, 2.0]
        assert mesh.max()["d"].data.tolist() == [5.0, 6.0]
        np.testing.assert_allclose(mesh.std()["d"].data, np.std([1.0, 3.0, 5.0]))

    def test_no_temporal_variable_raises(self):
        static = _temporal_mesh(np.array([[1.0, 2.0]]))  # single step -> still temporal
        # A genuinely static mesh: build one with a 1-D variable.
        flat = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
        )
        assert static  # fixture sanity
        with pytest.raises(ValueError, match="time dimension"):
            flat.reduce("mean")

    def test_immutable(self, mesh):
        mesh.reduce("mean")
        assert mesh["d"].has_time is True  # original untouched


class TestTransforms:
    def test_cumsum(self, mesh):
        assert mesh.cumsum()["d"].data.tolist() == [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]

    def test_cumprod(self, mesh):
        assert mesh.cumprod()["d"].data.tolist() == [
            [1.0, 2.0],
            [3.0, 8.0],
            [15.0, 48.0],
        ]

    def test_shift_forward_vacates_with_nan(self, mesh):
        out = mesh.shift(1)["d"].data
        assert np.isnan(out[0]).all()
        assert out[1].tolist() == [1.0, 2.0]
        assert out[2].tolist() == [3.0, 4.0]

    def test_shape_preserved(self, mesh):
        for derived in (mesh.cumsum(), mesh.shift(1), mesh.ffill(), mesh.rolling(2)):
            assert derived["d"].data.shape == (3, 2)
            assert derived["d"].has_time is True


class TestFill:
    @pytest.fixture
    def gappy(self) -> UgridDataset:
        return _temporal_mesh(np.array([[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]]))

    def test_ffill(self, gappy):
        assert gappy.ffill()["d"].data.tolist() == [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]]

    def test_bfill(self, gappy):
        assert gappy.bfill()["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0], [5.0, 6.0]]

    def test_interpolate_na_linear(self, gappy):
        assert gappy.interpolate_na()["d"].data.tolist() == [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]

    def test_ffill_limit_leaves_far_gaps(self):
        mesh = _temporal_mesh(
            np.array([[1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan], [4.0, 4.0]])
        )
        out = mesh.ffill(limit=1)["d"].data
        assert out[1].tolist() == [1.0, 1.0]  # within limit
        assert np.isnan(out[2]).all()  # beyond limit

    def test_interpolate_na_rejects_unknown_method_up_front(self):
        # The method is validated before the time check, so a bogus method is rejected as such
        # even on a mesh with no temporal variable (where the time check would otherwise mask it).
        static = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
            data_locations={"d": "face"},
        )
        with pytest.raises(ValueError, match=r"interpolate_na\(\) takes method="):
            static.interpolate_na(method="bogus")

    def test_interpolate_na_cubic_recovers_quadratic(self):
        # NC3: y = x**2 along time with step 2 a gap; a cubic recovers 4.0, where the default
        # linear fill would give 5.0.
        mesh = _temporal_mesh(
            np.array(
                [[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [9.0, 9.0], [16.0, 16.0]]
            )
        )
        out = mesh.interpolate_na(method="cubic")["d"].data
        np.testing.assert_allclose(out[2], [4.0, 4.0])


class TestRolling:
    def test_trailing_mean(self, mesh):
        out = mesh.rolling(2, "mean")["d"].data
        assert out.tolist() == [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]

    def test_min_periods_blanks_short_windows(self, mesh):
        out = mesh.rolling(2, "mean", min_periods=2)["d"].data
        # step 0 has only itself in a trailing window -> fewer than 2 valid -> NaN
        assert np.isnan(out[0]).all()
        assert out[1].tolist() == [2.0, 3.0]


class TestSelection:
    @pytest.fixture
    def four(self) -> UgridDataset:
        return _temporal_mesh(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )

    def test_isel_int_collapses(self, four):
        out = four.isel(1)
        assert out["d"].data.tolist() == [3.0, 4.0]
        assert out["d"].has_time is False

    def test_isel_slice_keeps_time(self, four):
        out = four.isel(slice(1, 3))
        assert out["d"].data.tolist() == [[3.0, 4.0], [5.0, 6.0]]
        assert out["d"].has_time is True

    def test_isel_list(self, four):
        assert four.isel([0, 2])["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_head_tail_thin(self, four):
        assert four.head(2)["d"].data.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert four.tail(2)["d"].data.tolist() == [[5.0, 6.0], [7.0, 8.0]]
        assert four.thin(2)["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_drop_isel(self, four):
        assert four.drop_isel([0, 1])["d"].data.tolist() == [[5.0, 6.0], [7.0, 8.0]]

    def test_diff(self, four):
        assert four.diff()["d"].data.tolist() == [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]

    def test_thin_rejects_zero(self, four):
        with pytest.raises(ValueError, match="thin step"):
            four.thin(0)

    def test_squeeze_single_step(self):
        mesh = _temporal_mesh(np.array([[9.0, 9.0]]))
        out = mesh.squeeze()
        assert out["d"].data.tolist() == [9.0, 9.0]
        assert out["d"].has_time is False

    def test_squeeze_multistep_is_noop(self, four):
        assert four.squeeze()["d"].data.shape == (4, 2)

    def test_time_values_trimmed(self):
        mesh = _temporal_mesh(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        mesh["d"].attributes = {"time_values": [10, 20, 30]}
        assert mesh.isel([0, 2])["d"].attributes["time_values"] == [10, 30]
        assert mesh.diff()["d"].attributes["time_values"] == [20, 30]


class TestConcatMerge:
    def _mesh(self, data: np.ndarray, name: str = "d") -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={name: data},
            data_locations={name: "face"},
        )

    def test_concat_along_time(self):
        a = self._mesh(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = self._mesh(np.array([[5.0, 6.0]]))
        out = a.concat(b)
        assert out["d"].data.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_concat_list_and_time_values(self):
        a = self._mesh(np.array([[1.0, 1.0]]))
        a["d"].attributes = {"time_values": [0]}
        b = self._mesh(np.array([[2.0, 2.0]]))
        b["d"].attributes = {"time_values": [1]}
        out = a.concat([b])
        assert out["d"].attributes["time_values"] == [0, 1]

    def test_concat_variable_mismatch_raises(self):
        a = self._mesh(np.array([[1.0, 2.0]]), "x")
        b = self._mesh(np.array([[3.0, 4.0]]), "y")
        with pytest.raises(ValueError, match="same variables"):
            a.concat(b)

    def test_merge_union(self):
        a = self._mesh(np.array([1.0, 2.0]), "x")
        b = self._mesh(np.array([3.0, 4.0]), "y")
        merged = a.merge(b)
        assert sorted(merged.data_variable_names) == ["x", "y"]

    def test_merge_conflict_raises(self):
        a = self._mesh(np.array([1.0, 2.0]), "x")
        b = self._mesh(np.array([9.0, 9.0]), "x")
        with pytest.raises(ValueError, match="more than one"):
            a.merge(b)

    def test_different_topology_raises(self):
        a = self._mesh(np.array([[1.0, 2.0]]))
        other = UgridDataset.from_arrays(
            node_x=np.array([0.0, 2.0, 2.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([[1.0, 2.0]])},
            data_locations={"d": "face"},
        )
        with pytest.raises(ValueError, match="same mesh topology"):
            a.concat(other)


class TestMappingSurface:
    @pytest.fixture
    def mesh(self) -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
        )

    def test_len_contains_iter(self, mesh):
        assert len(mesh) == 2
        assert "a" in mesh
        assert "z" not in mesh
        assert sorted(iter(mesh)) == ["a", "b"]

    def test_keys_values_items_get(self, mesh):
        assert sorted(mesh.keys()) == ["a", "b"]
        assert len(mesh.values()) == 2
        assert dict(mesh.items())["a"].location == "face"
        assert mesh.get("missing") is None
        assert mesh.get("a").name == "a"

    def test_data_vars_is_a_copy(self, mesh):
        dv = mesh.data_vars
        dv.clear()
        assert len(mesh) == 2  # dataset untouched


class TestVariableManagement:
    @pytest.fixture
    def mesh(self) -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
        )

    def test_with_variable_is_immutable(self, mesh):
        out = mesh.with_variable("e", np.array([5.0, 6.0]))
        assert sorted(out.data_variable_names) == ["d", "e"]
        assert mesh.data_variable_names == ["d"]  # original untouched

    def test_with_variable_temporal(self, mesh):
        out = mesh.with_variable("t", np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert out["t"].has_time is True

    def test_drop_variables(self, mesh):
        out = mesh.with_variable("e", np.array([5.0, 6.0])).drop_variables("d")
        assert out.data_variable_names == ["e"]

    def test_drop_missing_raises(self, mesh):
        with pytest.raises(KeyError, match="not found"):
            mesh.drop_variables("zzz")

    def test_rename_preserves_lazy_and_relabels(self, mesh):
        out = mesh.rename_variable("d", "depth")
        assert out.data_variable_names == ["depth"]
        assert out["depth"].name == "depth"
        assert out["depth"].data.tolist() == [1.0, 2.0]

    def test_rename_conflict_raises(self, mesh):
        two = mesh.with_variable("e", np.array([5.0, 6.0]))
        with pytest.raises(ValueError, match="already exists"):
            two.rename_variable("d", "e")


class TestLabelSelection:
    def _mesh(self, data: np.ndarray, time_values: list | None = None) -> UgridDataset:
        m = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": data},
            data_locations={"d": "face"},
        )
        if time_values is not None:
            m["d"].attributes = {"time_values": time_values}
        return m

    def test_sel_scalar_collapses(self):
        m = self._mesh(np.array([[1.0, 2.0], [3.0, 4.0]]), [10, 20])
        out = m.sel(20)
        assert out["d"].data.tolist() == [3.0, 4.0]
        assert out["d"].has_time is False

    def test_sel_sequence(self):
        m = self._mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [10, 20, 30])
        assert m.sel([10, 30])["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_sel_missing_raises(self):
        m = self._mesh(np.array([[1.0, 2.0]]), [10])
        with pytest.raises(ValueError, match="not found"):
            m.sel(999)

    def test_drop_sel(self):
        m = self._mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [10, 20, 30])
        assert m.drop_sel(20)["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_sortby(self):
        m = self._mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), [10, 30, 20])
        out = m.sortby()
        assert out["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]
        assert out["d"].attributes["time_values"] == [10, 20, 30]

    def test_drop_duplicates(self):
        m = self._mesh(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), [5, 5, 9])
        out = m.drop_duplicates()
        assert out["d"].data.tolist() == [[1.0, 1.0], [3.0, 3.0]]
        assert out["d"].attributes["time_values"] == [5, 9]


class TestDropna:
    def _mesh(self, data: np.ndarray) -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": data},
            data_locations={"d": "face"},
        )

    def test_dropna_any_drops_steps_with_a_gap(self):
        m = self._mesh(np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]]))
        assert m.dropna("any")["d"].data.tolist() == [[1.0, 2.0], [5.0, 6.0]]

    def test_dropna_all_keeps_partial_steps(self):
        m = self._mesh(np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]]))
        out = m.dropna("all")["d"].data
        assert out.shape == (2, 2)  # the all-NaN step is dropped, the partial one kept

    def test_dropna_thresh(self):
        m = self._mesh(np.array([[1.0, 2.0], [np.nan, 4.0]]))
        assert m.dropna(thresh=2)["d"].data.tolist() == [[1.0, 2.0]]


class TestAlongTimeValidation:
    @pytest.fixture
    def mesh(self) -> UgridDataset:
        return _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    def test_reduce_unknown_how_raises(self, mesh):
        with pytest.raises(ValueError, match="how"):
            mesh.reduce("bogus")

    def test_reduce_quantile_requires_q(self, mesh):
        with pytest.raises(ValueError, match="quantile"):
            mesh.reduce("quantile")

    def test_reduce_q_rejected_for_non_quantile(self, mesh):
        with pytest.raises(ValueError, match="q"):
            mesh.reduce("mean", q=0.5)

    def test_reduce_quantile_with_q_works(self, mesh):
        out = mesh.reduce("quantile", q=0.5)
        assert out["d"].data.tolist() == [3.0, 4.0]

    def test_rolling_unknown_how_raises(self, mesh):
        with pytest.raises(ValueError, match="how"):
            mesh.rolling(2, "bogus")

    def test_dropna_unknown_how_raises(self, mesh):
        with pytest.raises(ValueError, match="how"):
            mesh.dropna(how="bogus")


class TestReviewFixes:
    def test_reduce_drops_stale_time_values(self):
        # L2: a collapsed (static) variable must not carry the old time coordinate.
        mesh = _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        mesh["d"].attributes = {"time_values": [10, 20, 30], "units": "m"}
        out = mesh.reduce("mean")
        assert "time_values" not in out["d"].attributes
        assert out["d"].attributes.get("units") == "m"  # other attrs preserved

    def test_reduce_does_not_mutate_source_attributes(self):
        # L2: _static_from must copy, not alias, the source attributes dict.
        mesh = _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0]]))
        mesh["d"].attributes = {"units": "m"}
        out = mesh.reduce("sum")
        out["d"].attributes["units"] = "km"
        assert mesh["d"].attributes["units"] == "m"  # source untouched

    def test_concat_temporal_static_mismatch_raises(self):
        temporal = _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0]]))
        static = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([5.0, 6.0])},
            data_locations={"d": "face"},
        )
        with pytest.raises(ValueError, match="temporal in one dataset but static"):
            temporal.concat(static)


class TestRound2Fixes:
    def _two_temporal(self, len_a: int, len_b: int) -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={
                "a": np.ones((len_a, 2)),
                "b": np.ones((len_b, 2)),
            },
            data_locations={"a": "face", "b": "face"},
        )

    def test_m1_mismatched_time_lengths_raise(self):
        mesh = self._two_temporal(3, 2)
        with pytest.raises(ValueError, match="share one time length"):
            mesh.isel(2)

    def test_m4_rolling_rejects_counting_how(self):
        mesh = _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="rolling how"):
            mesh.rolling(2, "all")

    def test_l2_squeeze_squeezes_a_later_single_step_var(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"multi": np.ones((3, 2)), "single": np.ones((1, 2))},
            data_locations={"multi": "face", "single": "face"},
        )
        out = mesh.squeeze()
        assert out["single"].has_time is False  # squeezed despite not being first
        assert out["multi"].has_time is True

    def test_l3_concat_static_in_self_temporal_in_other_raises(self):
        static = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
            data_locations={"d": "face"},
        )
        temporal = _temporal_mesh(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="static in one dataset but temporal"):
            static.concat(temporal)

    def test_n2_reduce_all_declares_flag_nodata(self):
        mesh = _temporal_mesh(np.array([[1.0, 0.0], [1.0, 1.0]]))
        out = mesh.reduce("any")
        assert out["d"].nodata == 255
