"""ARC-32: GeoParquet read/write on FeatureCollection.

Covers :meth:`FeatureCollection.read_parquet` and
:meth:`FeatureCollection.to_parquet`:

* round-trip fidelity (feature count, CRS, attributes, geometries)
* ``columns=`` projection (Parquet's columnar layout wins here)
* ``compression`` options
* cloud / archive path handling via ``_io._parse_path``
* actionable ``ImportError`` when pyarrow isn't installed

pyarrow is an optional dependency (``pyramids-gis[parquet]``); every
test in this module is gated on ``pyarrow`` being importable and
skips cleanly otherwise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed (install with `pip install pyramids-gis[parquet]`)",
)


@pytest.fixture
def small_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [0.1, 0.2, 0.3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def polygon_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"area": [1.0, 4.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ],
            crs="EPSG:32636",
        )
    )


class TestRoundTrip:
    """to_parquet → read_parquet preserves the contents."""

    def test_point_round_trip(self, tmp_path: Path, small_fc: FeatureCollection):
        p = tmp_path / "points.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p)
        assert isinstance(rt, FeatureCollection)
        assert len(rt) == len(small_fc)
        assert rt.epsg == small_fc.epsg
        assert list(rt["name"]) == list(small_fc["name"])
        assert list(rt["id"]) == list(small_fc["id"])

    def test_polygon_round_trip(self, tmp_path: Path, polygon_fc: FeatureCollection):
        p = tmp_path / "polys.parquet"
        polygon_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p)
        assert len(rt) == 2
        assert rt.epsg == 32636
        assert list(rt["area"]) == [1.0, 4.0]
        # Geometries compare equal after round-trip (shapely equality).
        for orig, got in zip(polygon_fc.geometry, rt.geometry):
            assert orig.equals(got)

    def test_subclass_preserved(self, tmp_path: Path, small_fc: FeatureCollection):
        """read_parquet returns a FeatureCollection, not a bare GDF."""
        p = tmp_path / "sub.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p)
        assert type(rt).__name__ == "FeatureCollection"


class TestColumnsProjection:
    """columns= reduces the columns loaded — Parquet columnar wins."""

    def test_projects_to_subset(self, tmp_path: Path, small_fc: FeatureCollection):
        p = tmp_path / "proj.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p, columns=["name", "geometry"])
        # geometry always loaded; 'name' projected; 'id' / 'score' dropped.
        assert "name" in rt.columns
        assert "geometry" in rt.columns
        assert "id" not in rt.columns
        assert "score" not in rt.columns

    def test_columns_none_loads_all(self, tmp_path: Path, small_fc: FeatureCollection):
        p = tmp_path / "all.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p)
        assert set(rt.columns) == set(small_fc.columns)


class TestCompression:
    """Compression codec is passed through; file differs by size."""

    def test_snappy_default(self, tmp_path: Path, small_fc: FeatureCollection):
        p = tmp_path / "snappy.parquet"
        small_fc.to_parquet(p)
        assert p.stat().st_size > 0

    def test_gzip_option(self, tmp_path: Path, small_fc: FeatureCollection):
        p = tmp_path / "gzip.parquet"
        small_fc.to_parquet(p, compression="gzip")
        rt = FeatureCollection.read_parquet(p)
        assert len(rt) == len(small_fc)


class TestPathRewrite:
    """ARC-23 integration: _parse_path handles the path before geopandas."""

    def test_path_object(self, tmp_path: Path, small_fc: FeatureCollection):
        """pathlib.Path objects work (not just strings)."""
        p = tmp_path / "plain.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p)
        assert len(rt) == len(small_fc)

    def test_file_url(self, tmp_path: Path, small_fc: FeatureCollection):
        """file:// URLs are rewritten like in read_file."""
        p = tmp_path / "url.parquet"
        small_fc.to_parquet(p)
        rt = FeatureCollection.read_parquet(p.resolve().as_uri())
        assert len(rt) == len(small_fc)


class TestMissingPyarrow:
    """Without pyarrow, a clear ImportError fires up from geopandas."""

    def test_read_without_pyarrow_raises(
        self, tmp_path: Path, small_fc: FeatureCollection, monkeypatch
    ):
        """Simulate pyarrow-missing by patching gpd.read_parquet."""

        def _raise(*a, **kw):
            raise ImportError("pyarrow is required for Parquet support")

        monkeypatch.setattr(gpd, "read_parquet", _raise)
        p = tmp_path / "x.parquet"
        small_fc.to_parquet(p)
        with pytest.raises(ImportError, match="pyarrow"):
            FeatureCollection.read_parquet(p)
