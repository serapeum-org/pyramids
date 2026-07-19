"""Tests for FeatureCollection tessellation/binning ops (voronoi, quadtree)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from pyramids.base._errors import InvalidGeometryError
from pyramids.feature import FeatureCollection
from pyramids.feature import tessellation as tess


@pytest.fixture()
def point_fc() -> FeatureCollection:
    """Nine points on a 3x3 grid in a projected CRS, with a value column ``v``."""
    coords = [(x, y) for y in (0.0, 2.0, 4.0) for x in (0.0, 2.0, 4.0)]
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"v": list(range(1, 10))},
            geometry=[Point(x, y) for x, y in coords],
            crs="EPSG:32618",
        )
    )


@pytest.fixture()
def boundary_fc() -> FeatureCollection:
    """A square boundary covering the lower-left quadrant of the point grid."""
    square = Point(0.0, 0.0).buffer(2.5).envelope
    return FeatureCollection(gpd.GeoDataFrame(geometry=[square], crs="EPSG:32618"))


class TestVoronoi:
    def test_one_cell_per_point(self, point_fc: FeatureCollection) -> None:
        cells = point_fc.voronoi()
        assert isinstance(cells, FeatureCollection)
        assert len(cells) == len(point_fc)
        assert cells.crs.to_epsg() == 32618
        assert set(cells.geom_type.unique()) == {"Polygon"}

    def test_values_carried_to_containing_cell(
        self, point_fc: FeatureCollection
    ) -> None:
        cells = point_fc.voronoi(values="v")
        assert "v" in cells.columns
        assert sorted(cells["v"].tolist()) == list(range(1, 10))
        # each point falls inside the cell carrying its own value
        for _, point_row in point_fc.iterrows():
            match = cells[cells.contains(point_row.geometry)]
            assert match["v"].tolist() == [point_row["v"]]

    def test_clip_shrinks_cells(
        self, point_fc: FeatureCollection, boundary_fc: FeatureCollection
    ) -> None:
        full = point_fc.voronoi()
        clipped = point_fc.voronoi(clip=boundary_fc)
        assert clipped.area.sum() < full.area.sum()
        boundary = boundary_fc.geometry.union_all()
        assert clipped.geometry.apply(lambda g: g.within(boundary.buffer(1e-6))).all()

    def test_reprojected_clip(
        self, point_fc: FeatureCollection, boundary_fc: FeatureCollection
    ) -> None:
        clip_wgs84 = FeatureCollection(boundary_fc.to_crs(4326))
        clipped = point_fc.voronoi(clip=clip_wgs84)
        assert clipped.crs.to_epsg() == 32618
        assert clipped.area.sum() < point_fc.voronoi().area.sum()

    def test_non_point_raises(self) -> None:
        polys = FeatureCollection(
            gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1.0)], crs="EPSG:32618")
        )
        with pytest.raises(InvalidGeometryError):
            polys.voronoi()

    def test_single_point_raises(self) -> None:
        single = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:32618")
        )
        with pytest.raises(InvalidGeometryError):
            single.voronoi()

    def test_empty_raises(self) -> None:
        empty = FeatureCollection(gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32618"))
        with pytest.raises(InvalidGeometryError):
            empty.voronoi()

    def test_multipoint_rejected(self) -> None:
        mp = FeatureCollection(
            gpd.GeoDataFrame(geometry=[MultiPoint([(0, 0), (1, 1)])], crs="EPSG:32618")
        )
        with pytest.raises(InvalidGeometryError):
            mp.voronoi()

    def test_duplicate_point_skipped(self) -> None:
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1, 2, 3, 4, 5]},
                geometry=[
                    Point(0, 0),
                    Point(0, 0),
                    Point(2, 0),
                    Point(0, 2),
                    Point(2, 2),
                ],
                crs="EPSG:32618",
            )
        )
        cells = fc.voronoi(values="v")
        assert len(cells) == 4, (
            f"duplicate point should yield one empty cell, got {len(cells)} cells"
        )

    def test_collinear_points_do_not_crash(self) -> None:
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                geometry=[Point(0, 0), Point(1, 0), Point(2, 0)], crs="EPSG:32618"
            )
        )
        cells = fc.voronoi()
        assert len(cells) >= 1, (
            "collinear points should still tessellate into at least one cell"
        )
        assert set(cells.geom_type.unique()) == {"Polygon"}

    def test_clip_excluding_all_cells_is_empty(
        self, point_fc: FeatureCollection
    ) -> None:
        far = Point(1000.0, 1000.0).buffer(1.0).envelope
        clip = FeatureCollection(gpd.GeoDataFrame(geometry=[far], crs="EPSG:32618"))
        cells = point_fc.voronoi(clip=clip)
        assert len(cells) == 0, (
            f"a disjoint clip should exclude every cell, got {len(cells)}"
        )
        assert cells.crs.to_epsg() == 32618

    def test_missing_values_column_raises(self, point_fc: FeatureCollection) -> None:
        with pytest.raises(ValueError, match="not found"):
            point_fc.voronoi(values="nope")

    def test_clip_splitting_cell_duplicates_value(self) -> None:
        corners = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [100, 200, 300, 400]},
                geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
                crs="EPSG:32618",
            )
        )
        two_boxes = MultiPolygon([box(1, 1, 2, 2), box(3, 3, 4, 4)])
        clip = FeatureCollection(
            gpd.GeoDataFrame(geometry=[two_boxes], crs="EPSG:32618")
        )
        cells = corners.voronoi(values="v", clip=clip)
        assert len(cells) == 2, (
            "the two clip boxes both fall in one Voronoi cell, splitting it in two"
        )
        assert cells["v"].tolist() == [
            100,
            100,
        ], "both split parts carry the source point's value"


class TestQuadtree:
    def test_count_density(self, point_fc: FeatureCollection) -> None:
        cells = point_fc.quadtree(nmax=1)
        assert isinstance(cells, FeatureCollection)
        assert "count" in cells.columns
        assert int(cells["count"].sum()) == len(point_fc)
        assert cells.crs.to_epsg() == 32618

    def test_nmax_controls_resolution(self, point_fc: FeatureCollection) -> None:
        fine = point_fc.quadtree(nmax=1)
        coarse = point_fc.quadtree(nmax=100)
        assert len(fine) > len(coarse)
        assert len(coarse) == 1

    @pytest.mark.parametrize(
        "agg", ["mean", "sum", "median", "min", "max", "std", "count"]
    )
    def test_named_reducers(self, point_fc: FeatureCollection, agg: str) -> None:
        cells = point_fc.quadtree(column="v", agg=agg, nmax=100)
        assert len(cells) == 1
        expected = {
            "mean": 5.0,
            "sum": 45.0,
            "median": 5.0,
            "min": 1.0,
            "max": 9.0,
            "std": float(np.std(range(1, 10))),
            "count": 9.0,
        }[agg]
        assert cells["v"].iloc[0] == pytest.approx(expected)

    def test_callable_reducer(self, point_fc: FeatureCollection) -> None:
        cells = point_fc.quadtree(column="v", agg=lambda a: float(a.size), nmax=100)
        assert cells["v"].iloc[0] == pytest.approx(9.0)

    def test_nmin_drops_sparse_cells(self, point_fc: FeatureCollection) -> None:
        cells = point_fc.quadtree(nmax=1, nmin=2)
        assert len(cells) == 0

    def test_clip_intersection(
        self, point_fc: FeatureCollection, boundary_fc: FeatureCollection
    ) -> None:
        clipped = point_fc.quadtree(nmax=1, clip=boundary_fc)
        boundary = boundary_fc.geometry.union_all()
        assert clipped.geometry.apply(lambda g: g.within(boundary.buffer(1e-6))).all()

    def test_unknown_agg_raises(self, point_fc: FeatureCollection) -> None:
        with pytest.raises(ValueError):
            point_fc.quadtree(column="v", agg="bogus")

    def test_non_point_raises(self) -> None:
        polys = FeatureCollection(
            gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1.0)], crs="EPSG:32618")
        )
        with pytest.raises(InvalidGeometryError):
            polys.quadtree()

    def test_callable_agg_returning_nan(self, point_fc: FeatureCollection) -> None:
        cells = point_fc.quadtree(column="v", agg=lambda a: float("nan"), nmax=100)
        assert len(cells) == 1
        assert np.isnan(cells["v"].iloc[0]), (
            "a NaN-returning reducer should propagate NaN to the cell"
        )

    def test_clip_excluding_all_cells_is_empty(
        self, point_fc: FeatureCollection
    ) -> None:
        far = Point(1000.0, 1000.0).buffer(1.0).envelope
        clip = FeatureCollection(gpd.GeoDataFrame(geometry=[far], crs="EPSG:32618"))
        cells = point_fc.quadtree(nmax=1, clip=clip)
        assert len(cells) == 0, (
            f"a disjoint clip should exclude every cell, got {len(cells)}"
        )

    def test_single_point_bins_to_one_cell(self) -> None:
        single = FeatureCollection(
            gpd.GeoDataFrame({"v": [7]}, geometry=[Point(5, 5)], crs="EPSG:32618")
        )
        cells = single.quadtree()
        assert len(cells) == 1
        assert int(cells["count"].iloc[0]) == 1

    def test_empty_raises(self) -> None:
        empty = FeatureCollection(gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32618"))
        with pytest.raises(InvalidGeometryError):
            empty.quadtree()

    def test_missing_column_raises(self, point_fc: FeatureCollection) -> None:
        with pytest.raises(ValueError, match="not found"):
            point_fc.quadtree(column="nope")

    @pytest.mark.parametrize("nmax", [0, -1])
    def test_nmax_below_one_raises(
        self, point_fc: FeatureCollection, nmax: int
    ) -> None:
        with pytest.raises(ValueError, match="nmax must be >= 1"):
            point_fc.quadtree(nmax=nmax)


class TestTessellationHelpers:
    def test_point_xy_drops_non_finite(self) -> None:
        series = gpd.GeoSeries([Point(float("inf"), 0), Point(2, 3), Point(4, 5)])
        xs, ys, keep = tess.point_xy(series)
        assert xs.tolist() == pytest.approx([2.0, 4.0])
        assert ys.tolist() == pytest.approx([3.0, 5.0])
        assert keep.tolist() == [1, 2]

    def test_polygon_parts_explodes_multipolygon(self) -> None:
        mp = MultiPolygon(
            [Point(0, 0).buffer(1.0).envelope, Point(10, 10).buffer(1.0).envelope]
        )
        parts = tess.polygon_parts(mp)
        assert len(parts) == 2
        assert all(p.geom_type == "Polygon" for p in parts)

    def test_polygon_parts_filters_non_polygon_from_collection(self) -> None:
        collection = GeometryCollection([Point(0, 0), Point(0, 0).buffer(1.0).envelope])
        parts = tess.polygon_parts(collection)
        assert [p.geom_type for p in parts] == ["Polygon"]

    def test_polygon_parts_flattens_nested_multipolygon(self) -> None:
        nested = GeometryCollection(
            [Point(5, 5), MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])]
        )
        parts = tess.polygon_parts(nested)
        assert [p.geom_type for p in parts] == ["Polygon", "Polygon"]
        assert sorted(p.area for p in parts) == pytest.approx([1.0, 1.0])

    def test_dedupe_xy_keeps_first_occurrence(self) -> None:
        xs = np.array([0.0, 2.0, 0.0, 2.0, 4.0])
        ys = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        ux, uy, keep = tess.dedupe_xy(xs, ys)
        assert keep.tolist() == [0, 1, 4]
        assert ux.tolist() == pytest.approx([0.0, 2.0, 4.0])
        assert uy.tolist() == pytest.approx([0.0, 0.0, 1.0])

    def test_polygon_parts_none_and_empty(self) -> None:
        assert tess.polygon_parts(None) == []
        assert tess.polygon_parts(Polygon()) == []

    def test_resolve_clip_passthrough_shapely(self) -> None:
        geom = Point(0, 0).buffer(1.0)
        assert tess.resolve_clip(geom, "EPSG:4326") is geom

    def test_resolve_clip_none(self) -> None:
        assert tess.resolve_clip(None, "EPSG:4326") is None

    def test_resolve_reducer_callable_passthrough(self) -> None:
        fn = np.nanmax
        assert tess.resolve_reducer(fn) is fn

    def test_quadtree_cells_degenerate_extent(self) -> None:
        xs = np.array([5.0, 5.0])
        ys = np.array([5.0, 5.0])
        cells = tess.quadtree_cells(xs, ys, lambda idx: float(len(idx)), nmax=1, nmin=0)
        assert len(cells) == 1, "coincident points must not recurse forever"
        assert cells[0][4] == pytest.approx(2.0)
