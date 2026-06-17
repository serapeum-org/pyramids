"""Tests for the FeatureCollection gap methods (issue #576 in-scope set).

Covers ``fishnet``, ``interpolate_to_raster`` (IDW), ``to_pmtiles`` / ``to_mvt``, ``read_gpx_layers`` and the
paginated ``from_featureserver`` reader, plus the ``tessellation.fishnet_cells`` helper.
"""

import math
import tempfile
import urllib.parse as urlparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from pyramids.base._errors import InvalidGeometryError
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.feature import tessellation as tess


@pytest.fixture()
def corner_points() -> FeatureCollection:
    """Four corner points of a 3x3 square with a numeric ``rain`` column, EPSG:4326."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"rain": [1.0, 2.0, 3.0, 4.0]},
            geometry=[Point(0, 0), Point(3, 0), Point(0, 3), Point(3, 3)],
            crs="EPSG:4326",
        )
    )


@pytest.fixture()
def small_points() -> FeatureCollection:
    """Three points with an ``id`` column, EPSG:4326."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
    )


class TestFishnetCells:
    def test_unit_square_2x2(self) -> None:
        """A 0.5 cell over the unit square yields a 2x2 row-major grid."""
        polygons, rows, cols = tess.fishnet_cells((0.0, 0.0, 1.0, 1.0), 0.5)
        assert len(polygons) == 4, f"expected 4 cells, got {len(polygons)}"
        assert rows == [0, 0, 1, 1], f"unexpected row indices: {rows}"
        assert cols == [0, 1, 0, 1], f"unexpected col indices: {cols}"
        assert all(p.geom_type == "Polygon" and p.is_valid for p in polygons)

    @pytest.mark.parametrize("cell_size", [0.0, -1.0])
    def test_non_positive_cell_size_raises(self, cell_size: float) -> None:
        """A non-positive cell size is rejected."""
        with pytest.raises(ValueError, match="cell_size must be > 0"):
            tess.fishnet_cells((0.0, 0.0, 1.0, 1.0), cell_size)

    def test_degenerate_bounds_raise(self) -> None:
        """Bounds with minx >= maxx are rejected."""
        with pytest.raises(ValueError, match="minx < maxx"):
            tess.fishnet_cells((1.0, 0.0, 0.0, 1.0), 0.5)


class TestFishnet:
    def test_cell_count_and_columns(self) -> None:
        """fishnet returns ceil(w/cs)*ceil(h/cs) square cells with row/col columns in the given CRS."""
        grid = FeatureCollection.fishnet((0.0, 0.0, 1.0, 1.0), 0.5, crs="EPSG:4326")
        assert len(grid) == 4, f"expected 4 cells, got {len(grid)}"
        assert sorted(grid.columns) == ["col", "geometry", "row"]
        assert grid.crs.to_epsg() == 4326
        assert set(grid.geom_type) == {"Polygon"}
        assert grid.geometry.is_valid.all()

    def test_uneven_extent_count(self) -> None:
        """A non-divisible extent still tiles to ceil(w/cs) * ceil(h/cs) cells."""
        grid = FeatureCollection.fishnet((0, 0, 10, 4), 3.0)
        expected = math.ceil(10 / 3) * math.ceil(4 / 3)
        assert len(grid) == expected, f"expected {expected}, got {len(grid)}"

    def test_cells_do_not_overlap(self) -> None:
        """Adjacent cells share only edges (zero-area intersections)."""
        grid = FeatureCollection.fishnet((0.0, 0.0, 1.0, 1.0), 0.5)
        geoms = list(grid.geometry)
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                assert geoms[i].intersection(geoms[j]).area == pytest.approx(0.0)

    def test_crs_none(self) -> None:
        """crs=None produces a CRS-less grid."""
        grid = FeatureCollection.fishnet((0.0, 0.0, 1.0, 1.0), 0.5)
        assert grid.crs is None


class TestInterpolateToRaster:
    def test_idw_returns_dataset(self, corner_points: FeatureCollection) -> None:
        """IDW interpolation returns a single-band Dataset in the layer CRS honouring cell_size."""
        surface = corner_points.interpolate_to_raster("rain", cell_size=1.0)
        assert isinstance(surface, Dataset)
        assert surface.band_count == 1
        assert surface.epsg == 4326
        assert surface.shape == (1, 3, 3), f"unexpected shape {surface.shape}"

    def test_n_neighbors_branch(self, corner_points: FeatureCollection) -> None:
        """The n_neighbors path (invdistnn) also produces a Dataset of the expected grid."""
        surface = corner_points.interpolate_to_raster("rain", cell_size=1.0, n_neighbors=3)
        assert isinstance(surface, Dataset)
        assert surface.shape == (1, 3, 3)

    def test_bounds_override(self, corner_points: FeatureCollection) -> None:
        """An explicit bounds box drives the output extent."""
        surface = corner_points.interpolate_to_raster("rain", cell_size=1.0, bounds=(0, 0, 2, 2))
        assert surface.shape == (1, 2, 2), f"unexpected shape {surface.shape}"

    def test_unknown_method_raises(self, corner_points: FeatureCollection) -> None:
        """A non-idw method is rejected and points at the optional kriging dependency."""
        with pytest.raises(ValueError, match="pykrige"):
            corner_points.interpolate_to_raster("rain", method="kriging")

    def test_too_few_points_raises(self) -> None:
        """Fewer than three points cannot be interpolated."""
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        )
        with pytest.raises(ValueError, match="at least 3 points"):
            fc.interpolate_to_raster("v")

    def test_missing_column_raises(self, corner_points: FeatureCollection) -> None:
        """An absent column is rejected before gridding."""
        with pytest.raises(ValueError, match="not found"):
            corner_points.interpolate_to_raster("nope")

    def test_non_numeric_column_raises(self) -> None:
        """A non-numeric column is rejected with a clear message."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"s": ["a", "b", "c"]},
                geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ValueError, match="must be numeric"):
            fc.interpolate_to_raster("s")

    def test_all_nan_column_raises(self) -> None:
        """An all-NaN column has nothing to interpolate."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [np.nan, np.nan, np.nan]},
                geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ValueError, match="all-NaN"):
            fc.interpolate_to_raster("v")

    def test_non_point_raises(self) -> None:
        """Non-point geometries are rejected (point-only operation)."""
        polys = FeatureCollection(
            gpd.GeoDataFrame({"v": [1.0]}, geometry=[Point(0, 0).buffer(1.0)], crs="EPSG:4326")
        )
        with pytest.raises(InvalidGeometryError):
            polys.interpolate_to_raster("v")


class TestVectorTileWriters:
    def test_to_pmtiles_writes_and_roundtrips(self, small_points: FeatureCollection) -> None:
        """to_pmtiles writes a .pmtiles archive (returned as Path) that reopens via read_file."""
        out = small_points.to_pmtiles(Path(tempfile.mkdtemp()) / "layer.pmtiles", max_zoom=5)
        assert isinstance(out, Path)
        assert out.exists() and out.suffix == ".pmtiles"
        assert len(FeatureCollection.read_file(out)) >= 1

    def test_to_mvt_writes_tile_root(self, small_points: FeatureCollection) -> None:
        """to_mvt writes a tile-root directory (returned as Path)."""
        out = small_points.to_mvt(Path(tempfile.mkdtemp()) / "tiles", max_zoom=5)
        assert isinstance(out, Path)
        assert out.exists()


class TestReadGpxLayers:
    @pytest.fixture()
    def gpx_path(self) -> Path:
        """A minimal GPX file with one waypoint and a two-point track (no routes)."""
        gpx = (
            '<?xml version="1.0"?>\n'
            '<gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">'
            '<wpt lat="1.0" lon="2.0"><name>wp1</name></wpt>'
            '<trk><name>t1</name><trkseg>'
            '<trkpt lat="1.0" lon="2.0"/><trkpt lat="1.1" lon="2.1"/>'
            "</trkseg></trk></gpx>"
        )
        path = Path(tempfile.mkdtemp()) / "t.gpx"
        path.write_text(gpx)
        return path

    def test_returns_non_empty_layers_only(self, gpx_path: Path) -> None:
        """Only sub-layers with features are returned; empty routes/route_points are omitted."""
        layers = FeatureCollection.read_gpx_layers(gpx_path)
        assert sorted(layers) == ["track_points", "tracks", "waypoints"]
        assert "routes" not in layers
        assert all(isinstance(fc, FeatureCollection) for fc in layers.values())

    def test_layer_contents(self, gpx_path: Path) -> None:
        """The waypoints layer carries the single waypoint."""
        layers = FeatureCollection.read_gpx_layers(gpx_path)
        assert len(layers["waypoints"]) == 1, f"expected 1 waypoint, got {len(layers['waypoints'])}"


def _page_factory(total: int, page_size: int):
    """Return a fake _read_featureserver_page classmethod serving ``total`` rows in ``page_size`` chunks."""

    calls: list[str] = []

    def fake(cls, page_url: str) -> FeatureCollection:
        calls.append(page_url)
        query = dict(urlparse.parse_qsl(page_url.split("?", 1)[1]))
        offset = int(query["resultOffset"])
        n = max(0, min(page_size, total - offset))
        return FeatureCollection(
            gpd.GeoDataFrame(
                {"id": list(range(offset, offset + n))},
                geometry=[Point(i, i) for i in range(n)],
                crs="EPSG:4326",
            )
        )

    return classmethod(fake), calls


class TestFromFeatureserver:
    def test_paginates_until_short_page(self, monkeypatch) -> None:
        """Pagination keeps requesting pages until a short page, accumulating every feature."""
        fake, calls = _page_factory(total=3, page_size=2)
        monkeypatch.setattr(FeatureCollection, "_read_featureserver_page", fake)
        fc = FeatureCollection.from_featureserver("https://x/FeatureServer/0", page_size=2)
        assert len(fc) == 3, f"expected 3 rows, got {len(fc)}"
        assert len(calls) == 2, f"expected 2 page requests, got {len(calls)}"
        assert fc.crs.to_epsg() == 4326

    def test_max_records_caps_fetch(self, monkeypatch) -> None:
        """max_records caps the total fetched and stops paging early."""
        fake, calls = _page_factory(total=10, page_size=2)
        monkeypatch.setattr(FeatureCollection, "_read_featureserver_page", fake)
        fc = FeatureCollection.from_featureserver("https://x/FeatureServer/0", page_size=2, max_records=2)
        assert len(fc) == 2, f"expected 2 rows, got {len(fc)}"
        assert len(calls) == 1, f"expected 1 page request, got {len(calls)}"

    def test_empty_server_returns_empty_collection(self, monkeypatch) -> None:
        """An empty layer yields an empty FeatureCollection, not an error."""
        fake, _ = _page_factory(total=0, page_size=2)
        monkeypatch.setattr(FeatureCollection, "_read_featureserver_page", fake)
        fc = FeatureCollection.from_featureserver("https://x/FeatureServer/0")
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 0

    def test_query_url_construction(self, monkeypatch) -> None:
        """The reader appends /query and the expected query params to a bare layer URL."""
        fake, calls = _page_factory(total=1, page_size=10)
        monkeypatch.setattr(FeatureCollection, "_read_featureserver_page", fake)
        FeatureCollection.from_featureserver("https://x/FeatureServer/0", where="A=1")
        assert "/FeatureServer/0/query?" in calls[0], f"unexpected URL: {calls[0]}"
        query = dict(urlparse.parse_qsl(calls[0].split("?", 1)[1]))
        assert query["where"] == "A=1"
        assert query["f"] == "json"
        assert query["resultOffset"] == "0"
