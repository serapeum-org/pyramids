"""Tests for ARC-23: FeatureCollection.read_file handles compressed archives.

``FeatureCollection.read_file`` now routes through
:func:`pyramids._io._parse_path` so that ``.zip``, ``.tar``,
``.tar.gz`` and ``.gz`` paths work transparently via GDAL's virtual
filesystem (``/vsizip/``, ``/vsitar/``, ``/vsigzip/``).
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def points_gdf() -> gpd.GeoDataFrame:
    """Simple two-point GeoDataFrame in EPSG:4326."""
    return gpd.GeoDataFrame(
        {"id": [1, 2], "name": ["a", "b"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )


@pytest.fixture
def geojson_on_disk(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> Path:
    """Write ``points_gdf`` to a plain GeoJSON and yield the path."""
    path = tmp_path / "points.geojson"
    points_gdf.to_file(path, driver="GeoJSON")
    return path


class TestZipRead:
    """ARC-23: reading from a ``.zip`` archive containing vector data."""

    def test_read_zip_implicit_member(
        self, tmp_path: Path, geojson_on_disk: Path, points_gdf
    ):
        """Path ending in ``.zip`` reads the first file inside the archive."""
        zip_path = tmp_path / "points.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(geojson_on_disk, arcname="points.geojson")

        fc = FeatureCollection.read_file(zip_path)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == len(points_gdf)
        assert fc.epsg == 4326

    def test_read_zip_explicit_member(
        self, tmp_path: Path, geojson_on_disk: Path, points_gdf
    ):
        """``archive.zip/inner.geojson`` reads the named member."""
        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(geojson_on_disk, arcname="inner.geojson")
            # A second file, deliberately not the first alphabetically.
            z.writestr("other.geojson", "irrelevant")

        nested = str(zip_path) + "/inner.geojson"
        fc = FeatureCollection.read_file(nested)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == len(points_gdf)


class TestTarRead:
    """ARC-23: reading from a ``.tar`` archive containing vector data."""

    def test_read_tar_explicit_member(
        self, tmp_path: Path, geojson_on_disk: Path, points_gdf
    ):
        """``archive.tar/inner.geojson`` reads the named member.

        Unlike ``.zip``, GDAL's ``/vsitar/`` does not auto-pick the
        first file — callers must name the inner member. The
        ``_parse_path`` helper preserves that structure.
        """
        tar_path = tmp_path / "points.tar"
        with tarfile.open(tar_path, "w") as t:
            t.add(geojson_on_disk, arcname="points.geojson")

        nested = str(tar_path) + "/points.geojson"
        fc = FeatureCollection.read_file(nested)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == len(points_gdf)


class TestGzipRead:
    """ARC-23: reading from a ``.gz``-compressed single file."""

    def test_read_gzipped_geojson(
        self, tmp_path: Path, geojson_on_disk: Path, points_gdf
    ):
        """A single-file gzip wrapper is transparently unpacked via /vsigzip/."""
        gz_path = tmp_path / "points.geojson.gz"
        with open(geojson_on_disk, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        fc = FeatureCollection.read_file(gz_path)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == len(points_gdf)


class TestPlainPassThrough:
    """Plain (non-compressed) paths continue to work untouched."""

    def test_plain_geojson(self, geojson_on_disk: Path, points_gdf):
        """A regular file path is handled by geopandas directly."""
        fc = FeatureCollection.read_file(geojson_on_disk)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == len(points_gdf)
