"""Unit tests for pyramids.io.sniff (resource format detection + dispatch)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.io.sniff import _load_parquet, load_resource, sniff_format
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

_GEOTIFF = "tests/data/geotiff/era5_land_monthly_averaged.tif"
_NETCDF = "tests/data/netcdf/noah-precipitation-1979.nc"
_GEOJSON = "tests/data/basin.geojson"


@pytest.fixture
def csv_file(tmp_path):
    """Write a tiny CSV file.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        pathlib.Path: Path to a 2-row CSV.
    """
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    return path


@pytest.fixture
def grib_file(tmp_path):
    """Write a 1-band GRIB2 file.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        pathlib.Path: Path to a GRIB2 file.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 6, 4, 1, gdal.GDT_Float32)
    mem.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(np.full((4, 6), 280.0, "float32"))
    path = tmp_path / "x.grib2"
    gdal.GetDriverByName("GRIB").CreateCopy(str(path), mem).FlushCache()
    return path


@pytest.fixture
def parquet_file(tmp_path):
    """Write a tiny non-geo Parquet file.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        pathlib.Path: Path to a plain (tabular) Parquet file.
    """
    path = tmp_path / "table.parquet"
    pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_parquet(path)
    return path


class TestSniffFormat:
    """Tests for sniff_format."""

    def test_tif_by_magic(self):
        """A GeoTIFF is detected via its TIFF magic bytes.

        Test scenario:
            The era5 GeoTIFF fixture sniffs as `tif`.
        """
        assert sniff_format(_GEOTIFF) == "tif"

    def test_netcdf_by_magic(self):
        """A NetCDF is detected via HDF5/CDF magic bytes.

        Test scenario:
            The noah NetCDF fixture sniffs as `nc`.
        """
        assert sniff_format(_NETCDF) == "nc"

    def test_grib_by_magic(self, grib_file):
        """A GRIB file is detected via its `GRIB` magic bytes.

        Args:
            grib_file: Fixture path to a GRIB2 file.

        Test scenario:
            The GRIB2 fixture sniffs as `grib`.
        """
        assert sniff_format(grib_file) == "grib"

    def test_parquet_by_magic(self, tmp_path):
        """A Parquet file is detected via its `PAR1` magic bytes.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Raw `PAR1` magic in a `.bin` file sniffs as `parquet` (no
            pyarrow needed — only the leading bytes are read).
        """
        p = tmp_path / "data.bin"
        p.write_bytes(b"PAR1" + b"\x00" * 24)
        assert sniff_format(p) == "parquet"

    def test_netcdf_classic_cdf_magic(self, tmp_path):
        """Classic NetCDF (`CDF` magic) is detected.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Raw `CDF\\x01` magic sniffs as `nc`.
        """
        p = tmp_path / "classic.bin"
        p.write_bytes(b"CDF\x01" + b"\x00" * 12)
        assert sniff_format(p) == "nc"

    def test_geopackage_by_magic(self, tmp_path):
        """GeoPackage (SQLite) is detected via its magic header.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Raw `SQLite format 3\\x00` magic sniffs as `gpkg`.
        """
        p = tmp_path / "db.bin"
        p.write_bytes(b"SQLite format 3\x00")
        assert sniff_format(p) == "gpkg"

    def test_hdf5_magic(self, tmp_path):
        """HDF5-backed NetCDF (`\\x89HDF` magic) is detected as nc.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Raw HDF5 signature bytes sniff as `nc`.
        """
        p = tmp_path / "h5.bin"
        p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 8)
        assert sniff_format(p) == "nc"

    def test_geojson_by_extension(self):
        """GeoJSON (no binary magic) is detected by extension.

        Test scenario:
            The basin GeoJSON sniffs as `geojson` via its suffix.
        """
        assert sniff_format(_GEOJSON) == "geojson"

    def test_csv_by_extension(self, csv_file):
        """CSV is detected by extension.

        Args:
            csv_file: Fixture path to a CSV file.

        Test scenario:
            A `.csv` file sniffs as `csv`.
        """
        assert sniff_format(csv_file) == "csv"

    def test_zip_by_magic(self, tmp_path):
        """A ZIP is detected via its `PK\\x03\\x04` magic bytes.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            An archive sniffs as `zip`.
        """
        zp = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zp, "w") as archive:
            archive.writestr("inner.txt", "hi")
        assert sniff_format(zp) == "zip"

    def test_magic_beats_extension(self, tmp_path):
        """Magic bytes win over a misleading extension.

        Test scenario:
            A real ZIP named `.csv` still sniffs as `zip`.
        """
        liar = tmp_path / "actually.csv"
        with zipfile.ZipFile(liar, "w") as archive:
            archive.writestr("a.txt", "x")
        assert sniff_format(liar) == "zip", "magic bytes should override the extension"

    def test_missing_file_is_unknown(self):
        """A missing file is reported as unknown.

        Test scenario:
            A non-existent path yields `unknown` (no exception).
        """
        assert sniff_format("does-not-exist.bin") == "unknown"

    def test_unknown_extension(self, tmp_path):
        """An unrecognised extension with no magic is unknown.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A `.bin` file with arbitrary bytes sniffs as `unknown`.
        """
        p = tmp_path / "blob.bin"
        p.write_bytes(b"\x00\x01\x02random")
        assert sniff_format(p) == "unknown"


@pytest.mark.parquet
class TestLoadParquet:
    """Tests for _load_parquet (require the [parquet] extra)."""

    def test_non_geo_parquet_returns_dataframe(self, parquet_file):
        """A plain Parquet file loads as a pandas DataFrame.

        Args:
            parquet_file: Fixture path to a non-geo Parquet file.

        Test scenario:
            No GeoParquet `geo` metadata → DataFrame with the rows.
        """
        df = _load_parquet(Path(parquet_file))
        assert isinstance(
            df, pd.DataFrame
        ), f"Expected DataFrame, got {type(df).__name__}"
        assert list(df["x"]) == [1, 2, 3], f"Unexpected rows: {df}"


class TestLoadResource:
    """Tests for load_resource."""

    def test_loads_geotiff(self):
        """A GeoTIFF loads as a Dataset.

        Test scenario:
            The era5 GeoTIFF returns a 9-band Dataset.
        """
        ds = load_resource(_GEOTIFF)
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds).__name__}"
        assert ds.band_count == 9, f"Expected 9 bands, got {ds.band_count}"

    def test_loads_netcdf(self):
        """A NetCDF loads as a NetCDF container.

        Test scenario:
            The noah NetCDF returns a NetCDF.
        """
        assert isinstance(load_resource(_NETCDF), NetCDF), "Expected a NetCDF"

    def test_loads_geojson(self):
        """A GeoJSON loads as a FeatureCollection.

        Test scenario:
            The basin GeoJSON returns a FeatureCollection.
        """
        assert isinstance(load_resource(_GEOJSON), FeatureCollection), "Expected an FC"

    def test_loads_csv(self, csv_file):
        """A CSV loads as a pandas DataFrame.

        Args:
            csv_file: Fixture path to a CSV file.

        Test scenario:
            The CSV returns a 2x2 DataFrame.
        """
        df = load_resource(csv_file)
        assert isinstance(df, pd.DataFrame) and df.shape == (
            2,
            2,
        ), f"Bad CSV load: {df}"

    def test_loads_grib(self, grib_file):
        """A GRIB file loads as a Dataset.

        Args:
            grib_file: Fixture path to a GRIB2 file.

        Test scenario:
            The GRIB2 returns a 1-band Dataset.
        """
        ds = load_resource(grib_file)
        assert isinstance(ds, Dataset) and ds.band_count == 1, "Bad GRIB load"

    @pytest.mark.parquet
    def test_loads_parquet(self, parquet_file):
        """A non-geo Parquet loads as a DataFrame.

        Args:
            parquet_file: Fixture path to a Parquet file.

        Test scenario:
            The Parquet returns a DataFrame.
        """
        assert isinstance(
            load_resource(parquet_file), pd.DataFrame
        ), "Expected DataFrame"

    def test_zip_single_member_redispatches(self, tmp_path):
        """A ZIP with one primary member re-dispatches to that member.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A zip wrapping the GeoJSON returns a FeatureCollection.
        """
        zp = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zp, "w") as archive:
            archive.write(_GEOJSON, "basin.geojson")
        result = load_resource(zp, extract_to=tmp_path / "ext")
        assert isinstance(
            result, FeatureCollection
        ), f"Expected FC, got {type(result).__name__}"

    def test_zip_shapefile_redispatches(self, tmp_path):
        """A ZIP with a shapefile set re-dispatches to the .shp (the HDX case).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A zip of a `.shp` + sidecars opens as a FeatureCollection.
        """
        fc = FeatureCollection.read_file(_GEOJSON)
        shp_dir = tmp_path / "shp"
        shp_dir.mkdir()
        fc.to_file(str(shp_dir / "basin.shp"))
        zp = tmp_path / "shp.zip"
        with zipfile.ZipFile(zp, "w") as archive:
            for member in shp_dir.iterdir():
                archive.write(member, member.name)
        result = load_resource(zp, extract_to=tmp_path / "ext_shp")
        assert isinstance(
            result, FeatureCollection
        ), f"Expected FC from shapefile zip, got {type(result).__name__}"

    def test_zip_no_primary_returns_dir(self, tmp_path):
        """A ZIP with no recognisable primary returns the extraction dir.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A zip of two unrelated .txt members yields the dir Path.
        """
        zp = tmp_path / "docs.zip"
        with zipfile.ZipFile(zp, "w") as archive:
            archive.writestr("readme.txt", "a")
            archive.writestr("notes.txt", "b")
        dest = tmp_path / "ext"
        result = load_resource(zp, extract_to=dest)
        assert result == dest, f"Expected the extraction dir {dest}, got {result}"

    def test_expected_format_override(self):
        """expected_format skips sniffing and forces the reader.

        Test scenario:
            Forcing `nc` opens the NetCDF without magic detection.
        """
        assert isinstance(
            load_resource(_NETCDF, expected_format="nc"), NetCDF
        ), "expected_format override should open as NetCDF"

    def test_unknown_returns_bytes(self, tmp_path):
        """An unrecognised resource is returned as raw bytes.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A `.bin` blob comes back as the raw bytes.
        """
        p = tmp_path / "blob.bin"
        p.write_bytes(b"\x00\x01raw")
        result = load_resource(p)
        assert (
            result == b"\x00\x01raw"
        ), f"Expected raw bytes, got {type(result).__name__}"
