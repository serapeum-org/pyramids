"""Tests for the decompress-aware resource reader (:mod:`pyramids._resource`)."""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from pyramids import read_resource, sniff_kind
from pyramids._resource import (
    _archive_members_for_kind,
    _is_archive,
    _read_tabular,
    _select_vector_member,
    _sniff_from_archive,
    _strip_compression,
    _warn_if_multilayer,
)
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


def _shapefile_dir(tmp_path: Path) -> Path:
    """Write ``VECTOR_FILE`` out as a Shapefile (a multi-file format)."""
    out = tmp_path / "shp"
    out.mkdir()
    gpd.read_file(VECTOR_FILE).to_file(out / "aoi.shp")
    return out


def _zip_dir(src_dir: Path, dst: Path) -> Path:
    """Zip every file in ``src_dir`` (flat) into ``dst``."""
    with zipfile.ZipFile(dst, "w") as zf:
        for f in sorted(src_dir.iterdir()):
            zf.write(f, f.name)
    return dst


def _targz_file(src: str | Path, dst: Path, arcname: str | None = None) -> Path:
    """Tar+gzip ``src`` into ``dst`` as a single member."""
    with tarfile.open(dst, "w:gz") as tf:
        tf.add(src, arcname=arcname or Path(src).name)
    return dst


class TestArchiveRegressions:
    """Regressions for H1 (zipped Shapefile) and M1 (`.tar.gz`)."""

    def test_zipped_shapefile_reads_shp_not_sidecar(self, tmp_path: Path):
        # H1: a bare `.zip` of a Shapefile must resolve to the `.shp` member,
        # not the alphabetically-first sidecar (`aoi.cpg` / `aoi.dbf`).
        zp = _zip_dir(_shapefile_dir(tmp_path), tmp_path / "hotosm.zip")
        fc = read_resource(zp)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) > 0

    def test_zipped_shapefile_named_inner_suffix(self, tmp_path: Path):
        # COD-AB `.shp.zip` case: the inner suffix is in the name.
        zp = _zip_dir(_shapefile_dir(tmp_path), tmp_path / "aoi.shp.zip")
        assert sniff_kind(zp) == "vector"
        fc = read_resource(zp)
        assert isinstance(fc, FeatureCollection)

    def test_targz_raster_via_member_resolution(self, tmp_path: Path):
        # M1: `.tar.gz` raster must target the member; sniff peeks inside.
        tgz = _targz_file(RASTER_FILE, tmp_path / "data.tar.gz")
        ds = read_resource(tgz)
        assert isinstance(ds, Dataset)

    def test_targz_raster_with_fmt(self, tmp_path: Path):
        tgz = _targz_file(RASTER_FILE, tmp_path / "data.tar.gz")
        ds = read_resource(tgz, fmt="GeoTIFF")
        assert isinstance(ds, Dataset)

    def test_targz_sniffs_from_members(self, tmp_path: Path):
        tgz = _targz_file(RASTER_FILE, tmp_path / "data.tar.gz")
        # name alone is a container -> name-based sniff cannot classify it
        with pytest.raises(ValueError, match="could not determine"):
            sniff_kind(tgz)
        # but read_resource peeks inside and dispatches correctly
        assert isinstance(read_resource(tgz), Dataset)


class TestUncoveredFormats:
    """M2: formats named in the #447 acceptance criteria that lacked coverage."""

    def test_gpkg_gz_vector(self, tmp_path: Path):
        plain = tmp_path / "kontur.gpkg"
        gpd.read_file(VECTOR_FILE).to_file(plain, driver="GPKG")
        gz = _gzip_file(plain, tmp_path / "kontur.gpkg.gz")
        fc = read_resource(gz)
        assert isinstance(fc, FeatureCollection)

    def test_xlsx_tabular(self, tmp_path: Path):
        pytest.importorskip("openpyxl")
        xlsx = tmp_path / "survey.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(xlsx, index=False)
        df = read_resource(xlsx)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]


class TestLayerOnNonVector:
    """L4: `layer=` is ignored — with a warning — for non-vector resources."""

    def test_layer_on_raster_warns(self):
        with pytest.warns(UserWarning, match="ignored for raster"):
            ds = read_resource(RASTER_FILE, layer="whatever")
        assert isinstance(ds, Dataset)


class TestNameHelpers:
    """Pure, name-based helpers (`_is_archive`, `_strip_compression`)."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("a.zip", True),
            ("a.tar", True),
            ("a.tgz", True),
            ("a.tar.gz", True),
            ("a.gz", False),
            ("a.tif", False),
        ],
    )
    def test_is_archive(self, name: str, expected: bool):
        assert _is_archive(Path(name)) is expected

    @pytest.mark.parametrize(
        "name, inner",
        [
            ("kontur.gpkg.gz", "kontur.gpkg"),
            ("rivers.shp.zip", "rivers.shp"),
            ("noah.tif.tar.gz", "noah.tif"),
            ("download.zip", "download"),
            ("plain.tif", "plain.tif"),
        ],
    )
    def test_strip_compression(self, name: str, inner: str):
        assert _strip_compression(name) == inner


class TestArchivePeeking:
    """`_archive_members_for_kind` / `_sniff_from_archive` best-effort branches."""

    def test_members_for_kind_missing_archive_returns_empty(self, tmp_path: Path):
        assert _archive_members_for_kind(tmp_path / "nope.zip", "raster") == []

    def test_sniff_unlistable_archive_returns_none(self, tmp_path: Path):
        assert _sniff_from_archive(tmp_path / "nope.zip") is None

    def test_sniff_non_archive_returns_none(self):
        assert _sniff_from_archive(Path("plain.tif")) is None

    def test_sniff_tabular_member(self, tmp_path: Path):
        zp = tmp_path / "tab.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2\n")
        assert _sniff_from_archive(zp) == "tabular"

    def test_sniff_unknown_members_returns_none(self, tmp_path: Path):
        zp = tmp_path / "misc.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("readme.xyz", "hello")
        assert _sniff_from_archive(zp) is None


class TestSelectVectorMember:
    """Member / layer resolution for multi-member vector archives."""

    def test_select_by_index(self):
        member, passthrough = _select_vector_member(
            ["a.shp", "b.shp"], 1, Path("x.zip")
        )
        assert member == "b.shp"
        assert passthrough is None

    def test_select_by_name_stem(self):
        member, passthrough = _select_vector_member(
            ["a.shp", "b.shp"], "a", Path("x.zip")
        )
        assert member == "a.shp"
        assert passthrough is None

    def test_none_with_multiple_members_warns(self):
        with pytest.warns(UserWarning, match="2 vector members"):
            member, passthrough = _select_vector_member(
                ["a.shp", "b.shp"], None, Path("x.zip")
            )
        assert member == "a.shp"
        assert passthrough is None

    def test_unmatched_layer_forwarded_as_internal_layer(self):
        # a single-file container (GPKG) with an internal layer name: the layer
        # is not a member stem, so it is forwarded to the reader untouched.
        member, passthrough = _select_vector_member(
            ["city.gpkg"], "roads", Path("x.zip")
        )
        assert member == "city.gpkg"
        assert passthrough == "roads"

    def test_out_of_range_index_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            _select_vector_member(["a.shp", "b.shp"], 5, Path("x.zip"))

    def test_bool_layer_is_not_used_as_index(self):
        # bool subclasses int but must not be treated as a member index
        member, passthrough = _select_vector_member(
            ["a.shp", "b.shp"], True, Path("x.zip")
        )
        assert member == "a.shp"
        assert passthrough is True


class TestLazyImport:
    """`import pyramids` must stay light (M1): no eager dataset/feature/dask."""

    def test_import_pyramids_does_not_pull_heavy_stack(self):
        code = (
            "import sys, pyramids\n"
            "assert 'pyramids.feature' not in sys.modules, 'feature eagerly imported'\n"
            "assert 'pyramids.dataset' not in sys.modules, 'dataset eagerly imported'\n"
            "assert 'dask_geopandas' not in sys.modules, 'dask_geopandas eagerly imported'\n"
            "assert callable(pyramids.read_resource), 'lazy export not callable'\n"
            "assert 'pyramids.feature' in sys.modules, 'access did not load feature'\n"
            "print('LAZY_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "LAZY_OK" in result.stdout, result.stdout

    def test_unknown_attribute_raises(self):
        import pyramids

        with pytest.raises(AttributeError, match="has no attribute"):
            pyramids.this_attribute_does_not_exist

    def test_dir_includes_lazy_exports(self):
        import pyramids

        names = dir(pyramids)
        assert "read_resource" in names
        assert "sniff_kind" in names


class TestHelperRobustness:
    """Defensive branches that must never raise or mis-dispatch."""

    def test_warn_if_multilayer_unreadable_is_silent(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_multilayer(tmp_path / "missing.gpkg")

    def test_kind_raster_on_archive_without_raster_member_raises(self, tmp_path: Path):
        # GDAL rejects the unreadable member with a RuntimeError.
        zp = tmp_path / "blob.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("blob.bin", "\x00\x01\x02")
        with pytest.raises(RuntimeError):
            read_resource(zp, kind="raster")

    def test_kind_vector_on_archive_without_vector_member_raises(self, tmp_path: Path):
        # pyogrio raises DataSourceError (a RuntimeError subclass) for the member.
        zp = tmp_path / "blob.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("blob.bin", "\x00\x01\x02")
        with pytest.raises(RuntimeError):
            read_resource(zp, kind="vector")


class TestReadTabularBranches:
    """`_read_tabular` per-suffix dispatch and engine-missing handling."""

    def test_unsupported_suffix_raises(self, tmp_path: Path):
        f = tmp_path / "weird.dat"
        f.write_text("junk")
        with pytest.raises(ValueError, match="unsupported tabular suffix"):
            _read_tabular(f)

    def test_parquet_round_trip(self, tmp_path: Path):
        pytest.importorskip("pyarrow")
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(pq)
        df = _read_tabular(pq)
        assert list(df["a"]) == [1, 2]

    def test_xlsx_engine_missing_reraises_with_hint(self, tmp_path: Path, monkeypatch):
        # pandas imports the Excel engine lazily; simulate it being absent and
        # assert the wrapper re-raises with install guidance.
        def _raise(*args, **kwargs):
            raise ImportError("Missing optional dependency 'openpyxl'")

        monkeypatch.setattr(pd, "read_excel", _raise)
        f = tmp_path / "survey.xlsx"
        f.write_bytes(b"stub")
        with pytest.raises(ImportError, match="Excel engine"):
            _read_tabular(f)

    def test_parquet_engine_missing_reraises_with_hint(
        self, tmp_path: Path, monkeypatch
    ):
        def _raise(*args, **kwargs):
            raise ImportError("Missing optional dependency 'pyarrow'")

        monkeypatch.setattr(pd, "read_parquet", _raise)
        f = tmp_path / "t.parquet"
        f.write_bytes(b"stub")
        with pytest.raises(ImportError, match="parquet engine"):
            _read_tabular(f)
