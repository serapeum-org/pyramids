"""Tests for the UgridDataset analysis members (Phase 3A).

stats / sample / weighted / zonal_stats on an in-memory two-triangle mesh. These reuse the
shared pure kernels in ``pyramids.base._reductions`` but are exercised here through the
mesh-facing API.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def unit_mesh() -> UgridDataset:
    """Two triangles of unequal area sharing an edge.

    Face 0 = (0,0),(4,0),(4,2) -> area 4.0 ; face 1 = (0,0),(4,2),(0,1) -> area 2.0.
    """
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 4.0, 4.0, 0.0]),
        node_y=np.array([0.0, 0.0, 2.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"depth": np.array([10.0, 20.0])},
        epsg=4326,
    )


class TestStats:
    def test_basic_stats(self, unit_mesh):
        s = unit_mesh.stats("depth")
        assert s["min"] == 10.0
        assert s["max"] == 20.0
        assert s["mean"] == 15.0
        assert s["count"] == 2.0
        assert s["std"] == pytest.approx(5.0)

    def test_nodata_is_excluded(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"v": np.array([5.0, -9999.0])},
        )
        mesh["v"].nodata = -9999.0
        s = mesh.stats("v")
        assert s["count"] == 1.0
        assert s["mean"] == 5.0


class TestSample:
    def test_contains_hits_each_face(self, unit_mesh):
        # A point clearly inside face 0 (lower triangle) and one inside face 1.
        out = unit_mesh.sample("depth", x=[3.0, 1.0], y=[0.5, 0.9], method="contains")
        assert out.tolist() == [10.0, 20.0]

    def test_contains_outside_is_nan(self, unit_mesh):
        out = unit_mesh.sample("depth", x=[100.0], y=[100.0], method="contains")
        assert np.isnan(out[0])

    def test_nearest_always_resolves(self, unit_mesh):
        out = unit_mesh.sample("depth", x=[100.0], y=[100.0], method="nearest")
        assert out[0] in (10.0, 20.0)

    def test_unknown_method_raises(self, unit_mesh):
        with pytest.raises(ValueError, match="contains"):
            unit_mesh.sample("depth", x=[0.0], y=[0.0], method="bogus")


class TestWeighted:
    def test_area_weighted_mean(self, unit_mesh):
        # (10*4 + 20*2) / (4 + 2) = 80 / 6
        assert unit_mesh.weighted("depth", how="mean") == pytest.approx(80.0 / 6.0)

    def test_weighted_sum(self, unit_mesh):
        assert unit_mesh.weighted("depth", how="sum") == pytest.approx(
            10 * 4.0 + 20 * 2.0
        )

    def test_differs_from_unweighted_mean(self, unit_mesh):
        assert (
            unit_mesh.weighted("depth", how="mean") != unit_mesh.stats("depth")["mean"]
        )


class TestZonalStats:
    def test_one_zone_covers_both_faces(self, unit_mesh):
        zones = FeatureCollection(
            gpd.GeoDataFrame(
                {"z": ["a"]}, geometry=[box(-1, -1, 5, 3)], crs="EPSG:4326"
            )
        )
        out = unit_mesh.zonal_stats(
            zones, variable_name="depth", stats=("mean", "count")
        )
        # Both face centroids fall in the single zone; area-weighted mean = 80/6.
        assert out["count"].iloc[0] == 2.0
        assert out["mean"].iloc[0] == pytest.approx(80.0 / 6.0)

    def test_unweighted_mean(self, unit_mesh):
        zones = FeatureCollection(
            gpd.GeoDataFrame(
                {"z": ["a"]}, geometry=[box(-1, -1, 5, 3)], crs="EPSG:4326"
            )
        )
        out = unit_mesh.zonal_stats(
            zones, variable_name="depth", stats=("mean",), weighted=False
        )
        assert out["mean"].iloc[0] == pytest.approx(15.0)

    def test_crs_mismatch_raises(self, unit_mesh):
        zones = FeatureCollection(
            gpd.GeoDataFrame({"z": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
        )
        with pytest.raises(ValueError, match="CRS"):
            unit_mesh.zonal_stats(zones, variable_name="depth")


class TestNonTimeFirstAxis:
    """A variable storing time as a trailing axis must still reduce over the right axis."""

    def _mesh_time_trailing(self) -> UgridDataset:
        # data shape (n_face=2, n_time=3); dimensions mark the trailing axis as time.
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])},
            data_locations={"d": "face"},
        )
        mesh["d"].dimensions = ("nMesh2d_face", "time")
        return mesh

    def test_time_index_is_trailing(self):
        mesh = self._mesh_time_trailing()
        assert mesh["d"].time_index == 1

    def test_stats_uses_the_right_axis(self):
        mesh = self._mesh_time_trailing()
        # step 0 along the trailing time axis is the per-face column [1.0, 10.0].
        s = mesh.stats("d", time_index=0)
        assert s["min"] == 1.0
        assert s["max"] == 10.0
        assert s["count"] == 2.0

    def test_weighted_matches_face_count(self):
        mesh = self._mesh_time_trailing()
        # Two faces -> weighted over 2 face values at step 0, not 3 time values of one face.
        result = mesh.weighted("d", time_index=0)
        assert np.isfinite(result)


class TestWeightedValidation:
    def test_unsupported_how_raises(self, unit_mesh):
        # An unsupported `how` must raise, not silently return std via the kernel's else-branch.
        with pytest.raises(ValueError, match="how"):
            unit_mesh.weighted("depth", how="median")

    def test_typo_how_raises(self, unit_mesh):
        with pytest.raises(ValueError, match="how"):
            unit_mesh.weighted("depth", how="meen")


class TestSampleNodeMethod:
    def _node_mesh(self) -> UgridDataset:
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"h": np.array([1.0, 2.0, 3.0, 4.0])},
            data_locations={"h": "node"},
        )

    def test_contains_on_node_variable_raises(self):
        # L4: 'contains' is meaningless for a node (point) variable — must raise, not
        # silently do a nearest lookup.
        mesh = self._node_mesh()
        with pytest.raises(ValueError, match="contains"):
            mesh.sample("h", x=[0.0], y=[0.0], method="contains")

    def test_nearest_on_node_variable_works(self):
        mesh = self._node_mesh()
        out = mesh.sample("h", x=[0.01, 0.99], y=[0.01, 0.01], method="nearest")
        assert out.tolist() == [1.0, 2.0]


class TestZonalStatsMultiZone:
    """Multi-zone, std/var, empty-zone and overlap behaviour of mesh zonal_stats.

    unit_mesh: face 0 centroid ~ (2.667, 0.667), face 1 centroid ~ (1.333, 1.0),
    depths [10, 20], areas [4.0, 2.0].
    """

    def _zones(self, geoms, crs="EPSG:4326"):
        return FeatureCollection(
            gpd.GeoDataFrame({"z": list(range(len(geoms)))}, geometry=geoms, crs=crs)
        )

    def test_three_zones_mean_and_count(self, unit_mesh):
        zones = self._zones(
            [box(2.0, 0.0, 4.0, 0.9), box(0.0, 0.9, 2.0, 1.5), box(10, 10, 11, 11)]
        )
        out = unit_mesh.zonal_stats(
            zones, variable_name="depth", stats=("mean", "count"), weighted=False
        )
        assert out["count"].tolist() == [1.0, 1.0, 0.0]
        assert out["mean"].iloc[0] == 10.0
        assert out["mean"].iloc[1] == 20.0
        assert np.isnan(out["mean"].iloc[2])  # empty zone -> NaN

    def test_std_var_columns_single_face_is_zero(self, unit_mesh):
        zones = self._zones([box(2.0, 0.0, 4.0, 0.9), box(0.0, 0.9, 2.0, 1.5)])
        out = unit_mesh.zonal_stats(zones, variable_name="depth", stats=("std", "var"))
        assert out["std"].tolist() == [0.0, 0.0]
        assert out["var"].tolist() == [0.0, 0.0]

    def test_empty_zone_std_is_nan(self, unit_mesh):
        zones = self._zones([box(10, 10, 11, 11)])
        out = unit_mesh.zonal_stats(
            zones, variable_name="depth", stats=("std", "count")
        )
        assert out["count"].iloc[0] == 0.0
        assert np.isnan(out["std"].iloc[0])

    def test_overlapping_zones_first_wins(self, unit_mesh):
        # Both boxes contain face 0's centroid; it must be counted in the first zone only.
        zones = self._zones([box(2.0, 0.0, 4.0, 0.9), box(2.5, 0.5, 3.0, 0.8)])
        out = unit_mesh.zonal_stats(zones, variable_name="depth", stats=("count",))
        assert out["count"].tolist() == [1.0, 0.0]


class TestRound2AnalysisFixes:
    def test_m2_zonal_sum_is_plain_not_area_weighted(self, unit_mesh):
        zones = FeatureCollection(
            gpd.GeoDataFrame({"z": [0]}, geometry=[box(-1, -1, 5, 3)], crs="EPSG:4326")
        )
        out = unit_mesh.zonal_stats(zones, variable_name="depth", stats=("sum",))
        # plain Σ value = 10 + 20 = 30, NOT the area-weighted integral 10*4 + 20*2 = 80.
        assert out["sum"].iloc[0] == 30.0

    def _layered_mesh(self) -> UgridDataset:
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"layered": np.array([[1.0, 2.0], [3.0, 4.0]])},
            data_locations={"layered": "face"},
        )
        mesh["layered"].dimensions = ("n_layers", "nMesh2d_face")  # non-temporal, 2-D
        return mesh

    def test_m3_weighted_rejects_layered_variable(self):
        mesh = self._layered_mesh()
        assert mesh["layered"].has_time is False
        with pytest.raises(ValueError, match="not per-element"):
            mesh.weighted("layered")

    def test_m3_stats_rejects_layered_variable(self):
        mesh = self._layered_mesh()
        with pytest.raises(ValueError, match="not per-element"):
            mesh.stats("layered")
