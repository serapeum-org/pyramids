"""Tests for the decompress-aware resource reader (:mod:`pyramids._resource`)."""
from __future__ import annotations

import gzip
import shutil
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from pyramids import read_resource, sniff_kind
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

RASTER_FILE = "tests/data/acc4000.tif"
VECTOR_FILE = "tests/data/test_vector.geojson"


def _gzip_file(src: str | Path, dst: Path) -> Path:
    """Gzip ``src`` to ``dst`` (single-member gzip)."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dst


def _zip_file(src: str | Path, dst: Path, arcname: str | None = None) -> Path:
    """Zip ``src`` into ``dst`` as a single member."""
    with zipfile.ZipFile(dst, "w") as zf:
        zf.write(src, arcname or Path(src).name)
    return dst


class TestSniffKind:
    """Suffix + ``fmt`` classification, no I/O."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("worldpop.tif", "raster"),
            ("scene.tiff", "raster"),
            ("cube.nc", "raster"),
            ("noah.nc4", "raster"),
            ("kontur.gpkg", "vector"),
            ("admin.shp", "vector"),
            ("aoi.geojson", "vector"),
            ("export.gdb", "vector"),
            ("survey.csv", "tabular"),
            ("survey.tsv", "tabular"),
            ("survey.xlsx", "tabular"),
            ("survey.parquet", "tabular"),
        ],
    )
    def test_suffix(self, name: str, expected: str):
        assert sniff_kind(name) == expected

    def test_gzip_layer_is_peeled(self):
        assert sniff_kind("kontur.gpkg.gz") == "vector"
        assert sniff_kind("chirps.tif.gz") == "raster"
        assert sniff_kind("survey.csv.gz") == "tabular"

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("GeoTIFF", "raster"),
            ("Geopackage", "vector"),
            ("SHP", "vector"),
            ("CSV", "tabular"),
            ("XLSX", "tabular"),
            (".geojson", "vector"),
        ],
    )
    def test_fmt_tiebreaker_on_bare_zip(self, fmt: str, expected: str):
        # a bare `.zip` carries no inner suffix in its name; lean on `fmt`
        assert sniff_kind("download.zip", fmt=fmt) == expected

    def test_suffix_wins_over_fmt(self):
        # an explicit, recognised suffix is authoritative; `fmt` is only a fallback
        assert sniff_kind("admin.shp", fmt="GeoTIFF") == "vector"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="could not determine"):
            sniff_kind("mystery.dat")

    def test_bare_zip_without_fmt_raises(self):
        with pytest.raises(ValueError, match="could not determine"):
            sniff_kind("download.zip")


class TestReadPlain:
    """Reading uncompressed resources of each family."""

    def test_raster(self):
        ds = read_resource(RASTER_FILE)
        assert isinstance(ds, Dataset)

    def test_vector(self):
        fc = read_resource(VECTOR_FILE)
        assert isinstance(fc, FeatureCollection)

    def test_tabular_csv(self, tmp_path: Path):
        csv = tmp_path / "survey.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        df = read_resource(csv)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_tabular_tsv(self, tmp_path: Path):
        tsv = tmp_path / "survey.tsv"
        tsv.write_text("a\tb\n1\t2\n")
        df = read_resource(tsv)
        assert list(df.columns) == ["a", "b"]


class TestReadCompressed:
    """Decompression round-trips via GDAL VSI / pandas."""

    def test_gzipped_raster(self, tmp_path: Path):
        gz = _gzip_file(RASTER_FILE, tmp_path / "acc4000.tif.gz")
        ds = read_resource(gz)
        assert isinstance(ds, Dataset)

    def test_gzipped_vector(self, tmp_path: Path):
        gz = _gzip_file(VECTOR_FILE, tmp_path / "test_vector.geojson.gz")
        fc = read_resource(gz)
        assert isinstance(fc, FeatureCollection)

    def test_zipped_raster_sniffed_from_members(self, tmp_path: Path):
        # bare `.zip`, no `fmt` -> the reader peeks inside and finds the .tif
        zp = _zip_file(RASTER_FILE, tmp_path / "meta_hrsl.zip")
        ds = read_resource(zp)
        assert isinstance(ds, Dataset)

    def test_zipped_raster_with_fmt(self, tmp_path: Path):
        zp = _zip_file(RASTER_FILE, tmp_path / "meta_hrsl.zip")
        ds = read_resource(zp, fmt="GeoTIFF")
        assert isinstance(ds, Dataset)

    def test_gzipped_csv(self, tmp_path: Path):
        plain = tmp_path / "survey.csv"
        plain.write_text("a,b\n1,2\n3,4\n")
        gz = _gzip_file(plain, tmp_path / "survey.csv.gz")
        df = read_resource(gz)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestExplicitOverrides:
    """`kind` override and error surface."""

    def test_kind_override(self):
        ds = read_resource(RASTER_FILE, kind="raster")
        assert isinstance(ds, Dataset)

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="unsupported resource kind"):
            read_resource(RASTER_FILE, kind="bogus")  # type: ignore[arg-type]

    def test_undeterminable_raises(self, tmp_path: Path):
        mystery = tmp_path / "mystery.dat"
        mystery.write_bytes(b"\x00\x01")
        with pytest.raises(ValueError, match="could not determine"):
            read_resource(mystery)


class TestMultiLayerPolicy:
    """Default-first-layer-and-warn policy for multi-layer containers."""

    @pytest.fixture
    def multilayer_gpkg(self, tmp_path: Path) -> Path:
        gdf = gpd.read_file(VECTOR_FILE)
        path = tmp_path / "multi.gpkg"
        gdf.to_file(path, layer="alpha", driver="GPKG")
        gdf.to_file(path, layer="beta", driver="GPKG")
        FeatureCollection.list_layers_cache_clear()
        return path

    def test_warns_and_reads_first(self, multilayer_gpkg: Path):
        with pytest.warns(UserWarning, match="contains 2 layers"):
            fc = read_resource(multilayer_gpkg)
        assert isinstance(fc, FeatureCollection)

    def test_layer_selection_suppresses_warning(self, multilayer_gpkg: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fc = read_resource(multilayer_gpkg, layer="beta")
        assert isinstance(fc, FeatureCollection)

    def test_single_layer_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fc = read_resource(VECTOR_FILE)
        assert isinstance(fc, FeatureCollection)
