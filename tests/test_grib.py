"""Unit tests for pyramids.grib (GDAL-backed GRIB reader)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import DriverNotExistError
from pyramids.dataset import Dataset
from pyramids.grib import (
    _parse_grib_seconds,
    _parse_leading_int,
    _require_grib_driver,
    grib_band_metadata,
    open_grib,
)

pytestmark = pytest.mark.core


@pytest.fixture
def grib_path(tmp_path):
    """Write a small 2-band GRIB2 file and return its path.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        pathlib.Path: Path to an 8x6, 2-band GRIB2 on EPSG:4326.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 8, 6, 2, gdal.GDT_Float32)
    mem.SetGeoTransform((0.0, 1.0, 0.0, 6.0, 0.0, -1.0))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(np.full((6, 8), 280.0, "float32"))
    mem.GetRasterBand(2).WriteArray(np.full((6, 8), 101325.0, "float32"))
    path = tmp_path / "sample.grib2"
    dst = gdal.GetDriverByName("GRIB").CreateCopy(str(path), mem)
    dst.FlushCache()
    dst = None
    mem = None
    return path


class TestRequireGribDriver:
    """Tests for _require_grib_driver."""

    def test_passes_when_driver_present(self):
        """Returns silently when the GRIB driver is available.

        Test scenario:
            The libgdal-grib plugin is a core dependency, so the guard is a
            no-op in the test environment.
        """
        assert _require_grib_driver() is None, "Guard should pass when driver present"

    def test_raises_when_driver_absent(self, mocker):
        """Raises DriverNotExistError when GDAL has no GRIB driver.

        Args:
            mocker: pytest-mock fixture.

        Test scenario:
            Simulate a GDAL build without the plugin; the error names it.
        """
        mocker.patch("pyramids.grib.gdal.GetDriverByName", return_value=None)
        with pytest.raises(DriverNotExistError, match="libgdal-grib"):
            _require_grib_driver()


class TestParseGribSeconds:
    """Tests for _parse_grib_seconds."""

    def test_valid_timestamp(self):
        """A ``"<unix> sec UTC"`` string parses to an aware UTC datetime.

        Test scenario:
            Epoch seconds 0 map to 1970-01-01T00:00:00+00:00.
        """
        result = _parse_grib_seconds("0 sec UTC")
        assert result == datetime(1970, 1, 1, tzinfo=timezone.utc), f"Got {result}"

    def test_known_epoch(self):
        """A realistic forecast timestamp parses to the right calendar date.

        Test scenario:
            1700000000 → 2023-11-14 (UTC).
        """
        result = _parse_grib_seconds("1700000000 sec UTC")
        assert result.year == 2023 and result.tzinfo == timezone.utc, f"Got {result}"

    @pytest.mark.parametrize("value", [None, "", "n/a sec UTC", "notanint"])
    def test_unparseable_returns_none(self, value):
        """Missing or non-numeric values return None.

        Args:
            value: A missing / malformed GRIB time string.

        Test scenario:
            None, empty, and non-integer leading tokens all yield None.
        """
        assert _parse_grib_seconds(value) is None, f"Expected None for {value!r}"


class TestParseLeadingInt:
    """Tests for _parse_leading_int."""

    @pytest.mark.parametrize(
        "value, expected",
        [("10800 s", 10800), ("0", 0), ("42", 42)],
    )
    def test_parses_leading_int(self, value, expected):
        """The leading integer token is parsed.

        Args:
            value: GRIB metadata string whose first token is an integer.
            expected: Expected integer.

        Test scenario:
            Values with and without a trailing unit are handled.
        """
        assert _parse_leading_int(value) == expected, f"Bad parse of {value!r}"

    @pytest.mark.parametrize("value", [None, "", "abc"])
    def test_unparseable_returns_none(self, value):
        """Missing or non-numeric values return None.

        Args:
            value: A missing / malformed value.

        Test scenario:
            None, empty, and non-integer tokens yield None.
        """
        assert _parse_leading_int(value) is None, f"Expected None for {value!r}"


class TestOpenGrib:
    """Tests for open_grib."""

    def test_opens_as_dataset(self, grib_path):
        """A GRIB file opens as a multi-band pyramids Dataset.

        Args:
            grib_path: Fixture path to a 2-band GRIB2.

        Test scenario:
            open_grib returns a Dataset on EPSG:4326 with one band per message.
        """
        ds = open_grib(grib_path)
        assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds).__name__}"
        assert ds.band_count == 2, f"Expected 2 bands, got {ds.band_count}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_uses_grib_driver(self, grib_path):
        """The opened dataset is backed by GDAL's GRIB driver.

        Args:
            grib_path: Fixture path to a GRIB2.

        Test scenario:
            The backing gdal.Dataset reports the GRIB driver.
        """
        ds = open_grib(grib_path)
        assert ds.raster.GetDriver().ShortName == "GRIB", "Should use the GRIB driver"

    def test_accepts_str_path(self, grib_path):
        """A string path is accepted as well as a Path.

        Args:
            grib_path: Fixture path to a GRIB2.

        Test scenario:
            Passing ``str(path)`` opens the same dataset.
        """
        ds = open_grib(str(grib_path))
        assert ds.band_count == 2, "String path should open the GRIB"

    def test_raises_without_driver(self, grib_path, mocker):
        """open_grib raises DriverNotExistError when the driver is absent.

        Args:
            grib_path: Fixture path to a GRIB2.
            mocker: pytest-mock fixture.

        Test scenario:
            The driver guard fires before any read attempt.
        """
        mocker.patch("pyramids.grib.gdal.GetDriverByName", return_value=None)
        with pytest.raises(DriverNotExistError, match="GRIB"):
            open_grib(grib_path)


class TestGribBandMetadata:
    """Tests for grib_band_metadata."""

    def test_one_entry_per_band(self, grib_path):
        """Returns one metadata dict per band, in band order.

        Args:
            grib_path: Fixture path to a 2-band GRIB2.

        Test scenario:
            The 2-band file yields band indices [1, 2].
        """
        meta = grib_band_metadata(open_grib(grib_path))
        assert [m["band"] for m in meta] == [1, 2], f"Unexpected band indices: {meta}"

    def test_exposes_expected_keys(self, grib_path):
        """Each entry exposes the documented GRIB fields.

        Args:
            grib_path: Fixture path to a GRIB2.

        Test scenario:
            Every entry carries the full key set.
        """
        entry = grib_band_metadata(open_grib(grib_path))[0]
        expected = {
            "band",
            "element",
            "short_name",
            "comment",
            "unit",
            "discipline",
            "ref_time",
            "valid_time",
            "forecast_seconds",
        }
        assert set(entry) == expected, f"Key mismatch: {set(entry) ^ expected}"

    def test_times_and_horizon_are_typed(self, grib_path):
        """Reference/valid times decode to datetimes and horizon to int.

        Args:
            grib_path: Fixture path to a GRIB2.

        Test scenario:
            GDAL writes GRIB_VALID_TIME / GRIB_FORECAST_SECONDS, so the parsed
            fields are a UTC datetime and an int respectively.
        """
        entry = grib_band_metadata(open_grib(grib_path))[0]
        assert isinstance(
            entry["valid_time"], datetime
        ), "valid_time should be datetime"
        assert entry["valid_time"].tzinfo == timezone.utc, "valid_time should be UTC"
        assert isinstance(
            entry["forecast_seconds"], int
        ), "forecast_seconds should be int"
