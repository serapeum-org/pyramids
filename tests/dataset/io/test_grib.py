"""Unit tests for pyramids.grib (GDAL-backed GRIB reader)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import DriverNotExistError
from pyramids.dataset import Dataset
from pyramids.dataset.cog import cog_info
from pyramids.grib import (
    _parse_grib_seconds,
    _parse_leading_int,
    _require_grib_driver,
    _select_grib_band,
    grib_band_metadata,
    grib_to_cog,
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


@pytest.fixture
def grib_1band_path(tmp_path):
    """Write a single-message 8x6 GRIB2 on EPSG:4326 and return its path.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        pathlib.Path: Path to a 1-band GRIB2.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 8, 6, 1, gdal.GDT_Float32)
    mem.SetGeoTransform((0.0, 1.0, 0.0, 6.0, 0.0, -1.0))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(np.full((6, 8), 280.0, "float32"))
    path = tmp_path / "single.grib2"
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
        """A `"<unix> sec UTC"` string parses to an aware UTC datetime.

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
            Passing `str(path)` opens the same dataset.
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


class TestSelectGribBand:
    """Tests for _select_grib_band (0-based band resolution from GRIB element)."""

    def test_none_variable_single_band(self):
        """variable=None on a single-message file selects band 0."""
        meta = [{"band": 1, "element": "TMP"}]
        assert _select_grib_band(meta, None) == 0, "single-band None should be 0"

    def test_none_variable_multiband_raises(self):
        """variable=None on a multi-message file raises with the element list."""
        meta = [{"band": 1, "element": "TMP"}, {"band": 2, "element": "PRES"}]
        with pytest.raises(ValueError, match="pass variable="):
            _select_grib_band(meta, None)

    def test_matches_element_returns_zero_based(self):
        """A matching element returns its 0-based index (band 2 -> 1)."""
        meta = [{"band": 1, "element": "TMP"}, {"band": 2, "element": "PRES"}]
        assert _select_grib_band(meta, "PRES") == 1, "PRES is band 2 -> index 1"

    def test_match_is_case_insensitive(self):
        """Element matching ignores case."""
        meta = [{"band": 1, "element": "TMP"}]
        assert _select_grib_band(meta, "tmp") == 0, "lowercase should match TMP"

    def test_multiple_matches_warns_and_uses_first(self):
        """Several messages sharing the element warn and select the first."""
        meta = [{"band": 1, "element": "TMP"}, {"band": 2, "element": "TMP"}]
        with pytest.warns(UserWarning, match="using the first"):
            assert _select_grib_band(meta, "TMP") == 0

    def test_unknown_element_raises(self):
        """An absent element raises with the available-elements list."""
        meta = [{"band": 1, "element": "TMP"}]
        with pytest.raises(ValueError, match="available elements"):
            _select_grib_band(meta, "NOPE")

    def test_int_selects_band_number(self):
        """An int selects that 1-based band directly (band 2 -> index 1)."""
        meta = [{"band": 1, "element": "TMP"}, {"band": 2, "element": "TMP"}]
        assert _select_grib_band(meta, 2) == 1, "band number 2 -> index 1"

    def test_int_out_of_range_raises(self):
        """An out-of-range band number raises."""
        meta = [{"band": 1, "element": "TMP"}]
        with pytest.raises(ValueError, match="out of range"):
            _select_grib_band(meta, 5)

    def test_empty_string_raises(self):
        """An empty-string variable raises instead of matching None-element bands."""
        meta = [{"band": 1, "element": None}]
        with pytest.raises(ValueError, match="non-empty"):
            _select_grib_band(meta, "")

    def test_bool_variable_raises(self):
        """A bool is rejected rather than treated as a band number (True == 1)."""
        meta = [{"band": 1, "element": "TMP"}]
        with pytest.raises(ValueError, match="band number, element name, or None"):
            _select_grib_band(meta, True)

    def test_non_str_int_variable_raises(self):
        """A float variable raises a clear ValueError, not a cryptic AttributeError."""
        meta = [{"band": 1, "element": "TMP"}]
        with pytest.raises(ValueError, match="band number, element name, or None"):
            _select_grib_band(meta, 1.5)

    def test_matches_element_with_surrounding_whitespace(self):
        """A stored element with stray whitespace still matches (both stripped)."""
        meta = [{"band": 1, "element": " TMP "}]
        assert _select_grib_band(meta, "tmp") == 0, "padded element should still match"


class TestGribToCog:
    """Tests for grib_to_cog (open_grib -> band select -> to_cog)."""

    def test_single_band_writes_valid_cog(self, grib_1band_path, tmp_path):
        """A single-message GRIB with variable=None writes a valid COG.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(grib_1band_path, output=tmp_path / "single_cog.tif")
        assert out.exists(), "grib_to_cog should write the output file"
        assert cog_info(out).is_cog, "output should pass the COG validator"

    def test_selects_variable_band_preserves_data(self, grib_1band_path, tmp_path):
        """Selecting the sole message by its GRIB element preserves its data in the COG.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2 (values 280.0).
            tmp_path: pytest temp directory.
        """
        with open_grib(grib_1band_path) as src:
            element = grib_band_metadata(src)[0]["element"]
        # `element or None` keeps the test valid on GDAL builds that emit an empty
        # or missing GRIB element (which would otherwise trip the empty-string guard).
        out = grib_to_cog(
            grib_1band_path, output=tmp_path / "var_cog.tif", variable=element or None
        )
        with Dataset.read_file(str(out)) as ds:
            arr = ds.read_array()
        assert np.allclose(arr, 280.0), "selected band values should reach the COG"

    def test_selects_element_by_string_writes_that_bands_data(
        self, grib_path, tmp_path, mocker
    ):
        """A distinct element string selects the right band's data on a multi-band file.

        Args:
            grib_path: Fixture path to a 2-band GRIB2 (band1=280.0, band2=101325.0).
            tmp_path: pytest temp directory.
            mocker: pytest-mock fixture (GDAL writes 'unknown' for synthetic
                elements, so distinct names are injected).
        """
        mocker.patch(
            "pyramids.grib.grib_band_metadata",
            return_value=[
                {"band": 1, "element": "AAA"},
                {"band": 2, "element": "BBB"},
            ],
        )
        out = grib_to_cog(grib_path, output=tmp_path / "bbb_cog.tif", variable="BBB")
        with Dataset.read_file(str(out)) as ds:
            arr = ds.read_array()
        assert np.allclose(arr, 101325.0), "element BBB (band 2) data reaches COG"

    def test_int_band_selection_writes_that_bands_data(self, grib_path, tmp_path):
        """An int band number selects that specific message; its data reaches the COG.

        Args:
            grib_path: Fixture path to a 2-band GRIB2 (band1=280.0, band2=101325.0).
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(grib_path, output=tmp_path / "band2_cog.tif", variable=2)
        with Dataset.read_file(str(out)) as ds:
            arr = ds.read_array()
        assert cog_info(out).is_cog, "output should be a COG"
        assert np.allclose(arr, 101325.0), "band 2 values (101325) should reach COG"

    def test_preserves_grib_band_metadata(self, grib_1band_path, tmp_path):
        """The output COG carries the source GRIB_* band metadata.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(grib_1band_path, output=tmp_path / "meta_cog.tif")
        with Dataset.read_file(str(out)) as ds:
            md = ds.raster.GetRasterBand(1).GetMetadata()
        assert any(k.startswith("GRIB_") for k in md), "GRIB_* metadata should survive"

    def test_shared_element_warns_and_takes_first(self, grib_path, tmp_path, mocker):
        """A str element shared by several messages warns and selects the first band.

        Args:
            grib_path: Fixture path to a 2-band GRIB2 (band1=280.0, band2=101325.0).
            tmp_path: pytest temp directory.
            mocker: pytest-mock fixture (inject a duplicated element deterministically).
        """
        mocker.patch(
            "pyramids.grib.grib_band_metadata",
            return_value=[
                {"band": 1, "element": "TMP"},
                {"band": 2, "element": "TMP"},
            ],
        )
        with pytest.warns(UserWarning, match="using the first"):
            out = grib_to_cog(
                grib_path, output=tmp_path / "warn_cog.tif", variable="TMP"
            )
        with Dataset.read_file(str(out)) as ds:
            arr = ds.read_array()
        assert np.allclose(arr, 280.0), "first matching band (280) should win"

    def test_preserves_native_crs(self, grib_1band_path, tmp_path):
        """Without target_crs the COG keeps the GRIB's native EPSG:4326.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(grib_1band_path, output=tmp_path / "native_cog.tif")
        with Dataset.read_file(str(out)) as ds:
            assert ds.epsg == 4326, "native CRS should survive"

    def test_target_crs_reprojects(self, grib_1band_path, tmp_path):
        """target_crs reprojects the COG to the requested EPSG.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(
            grib_1band_path, output=tmp_path / "reproj_cog.tif", target_crs=3857
        )
        with Dataset.read_file(str(out)) as ds:
            assert ds.epsg == 3857, "should reproject to 3857"
        assert cog_info(out).is_cog, "reprojected output should still be a COG"

    def test_multiband_requires_variable(self, grib_path, tmp_path):
        """A multi-message GRIB without variable= raises.

        Args:
            grib_path: Fixture path to a 2-band GRIB2.
            tmp_path: pytest temp directory.
        """
        with pytest.raises(ValueError, match="pass variable="):
            grib_to_cog(grib_path, output=tmp_path / "x.tif")

    def test_unknown_variable_raises(self, grib_path, tmp_path):
        """Requesting an element no message carries raises.

        Args:
            grib_path: Fixture path to a 2-band GRIB2.
            tmp_path: pytest temp directory.
        """
        with pytest.raises(ValueError, match="No GRIB message with element"):
            grib_to_cog(grib_path, output=tmp_path / "y.tif", variable="NOPE")

    def test_missing_grib_raises(self, tmp_path):
        """A missing GRIB path raises FileNotFoundError (propagated from open_grib)."""
        with pytest.raises(FileNotFoundError):
            grib_to_cog(tmp_path / "nope.grib2", output=tmp_path / "x.tif")

    def test_invalid_cog_profile_raises(self, grib_1band_path, tmp_path):
        """An unrecognised cog_profile raises ValueError (propagated from to_cog).

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        with pytest.raises(ValueError, match="COG profile"):
            grib_to_cog(
                grib_1band_path, output=tmp_path / "bad.tif", cog_profile="bogus"
            )

    def test_non_default_cog_profile_writes_cog(self, grib_1band_path, tmp_path):
        """A non-default cog_profile (lzw) still produces a valid COG.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(
            grib_1band_path, output=tmp_path / "lzw.tif", cog_profile="lzw"
        )
        assert cog_info(out).is_cog, "lzw profile should still be a valid COG"

    def test_string_target_crs_reprojects(self, grib_1band_path, tmp_path):
        """A string target_crs reprojects, matching to_cog's int|str contract.

        Args:
            grib_1band_path: Fixture path to a 1-band GRIB2.
            tmp_path: pytest temp directory.
        """
        out = grib_to_cog(
            grib_1band_path, output=tmp_path / "str_crs.tif", target_crs="EPSG:3857"
        )
        with Dataset.read_file(str(out)) as ds:
            assert ds.epsg == 3857, "string target_crs should reproject to 3857"
