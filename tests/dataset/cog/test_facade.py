"""Unit tests for pyramids.dataset.cog.facade (the write_cog facade)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset.cog import facade
from pyramids.dataset.cog.facade import (
    PYRAMIDS_COG_DEFAULTS,
    _array_to_dataset,
    _coerce_epsg,
    _normalize_to_dataset,
    _resolve_predictor,
    write_cog,
)
from pyramids.dataset.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


def _read_compression(path: str | Path) -> str:
    """Return the GDAL IMAGE_STRUCTURE compression token of a raster.

    Args:
        path: Path to a raster readable by GDAL.

    Returns:
        The compression name (e.g. `"DEFLATE"`), or `""` when absent.
    """
    ds = gdal.Open(str(path))
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or ""
    ds = None
    return comp


def _read_predictor(path: str | Path) -> str:
    """Return the GDAL IMAGE_STRUCTURE PREDICTOR token of a raster.

    Args:
        path: Path to a raster readable by GDAL.

    Returns:
        The predictor token (e.g. `"2"`/`"3"`), or `""` when absent.
    """
    ds = gdal.Open(str(path))
    pred = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE") or ""
    ds = None
    return pred


@pytest.fixture
def float_array() -> np.ndarray:
    """A deterministic 2-D float32 array.

    Returns:
        A `(64, 64)` float32 array seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=1337)
    return (rng.random((64, 64)) * 100.0).astype("float32")


@pytest.fixture
def int_array() -> np.ndarray:
    """A deterministic 2-D int16 array.

    Returns:
        A `(48, 48)` int16 array seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=7)
    return (rng.integers(0, 50, size=(48, 48))).astype("int16")


class TestResolvePredictor:
    """Tests for _resolve_predictor."""

    @pytest.mark.parametrize(
        "gdal_dtype",
        [
            gdal.GDT_Byte,
            gdal.GDT_Int8,
            gdal.GDT_UInt16,
            gdal.GDT_Int16,
            gdal.GDT_UInt32,
            gdal.GDT_Int32,
            gdal.GDT_UInt64,
            gdal.GDT_Int64,
        ],
    )
    def test_integer_types_use_predictor_2(self, gdal_dtype):
        """Integer GDAL types resolve to horizontal differencing (2).

        Args:
            gdal_dtype: A GDAL integer data-type code.

        Test scenario:
            Every integer code maps to PREDICTOR=2 per GDAL/libtiff.
        """
        result = _resolve_predictor(gdal_dtype)
        assert result == 2, f"Expected predictor 2 for int type, got {result}"

    @pytest.mark.parametrize(
        "gdal_dtype",
        [gdal.GDT_Float32, gdal.GDT_Float64],
    )
    def test_float_types_use_predictor_3(self, gdal_dtype):
        """Floating-point GDAL types resolve to the float predictor (3).

        Args:
            gdal_dtype: A GDAL floating-point data-type code.

        Test scenario:
            Float codes map to PREDICTOR=3 per GDAL/libtiff.
        """
        result = _resolve_predictor(gdal_dtype)
        assert result == 3, f"Expected predictor 3 for float type, got {result}"


class TestCoerceEpsg:
    """Tests for _coerce_epsg."""

    def test_int_passthrough(self):
        """An integer EPSG code is returned unchanged.

        Test scenario:
            `_coerce_epsg(4326)` returns `4326` without parsing.
        """
        assert _coerce_epsg(4326) == 4326, "Integer EPSG should pass through"

    def test_authority_string(self):
        """An `EPSG:XXXX` string resolves to its integer code.

        Test scenario:
            `"EPSG:3857"` resolves to `3857`.
        """
        assert _coerce_epsg("EPSG:3857") == 3857, "Authority string should resolve"

    def test_wkt_string(self):
        """A WKT/PROJ string resolves to the matching EPSG code.

        Test scenario:
            The WGS84 PROJ4 string resolves to `4326`.
        """
        assert _coerce_epsg("+proj=longlat +datum=WGS84 +no_defs") == 4326, (
            "WGS84 PROJ string should resolve to 4326"
        )

    def test_unresolvable_raises(self):
        """A CRS with no EPSG code raises ValueError.

        Test scenario:
            A bespoke PROJ string with no authority code raises.
        """
        bespoke = "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84"
        with pytest.raises(ValueError, match="EPSG"):
            _coerce_epsg(bespoke)


class TestArrayToDataset:
    """Tests for _array_to_dataset."""

    def test_builds_dataset_2d(self, float_array):
        """A 2-D array becomes a single-band Dataset with the given geo/CRS.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            Shape, band count, and EPSG are preserved.
        """
        ds = _array_to_dataset(float_array, 4326, COG_GEOTRANSFORM, None)
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_builds_dataset_3d(self):
        """A 3-D array becomes a multi-band Dataset.

        Test scenario:
            A `(3, 16, 16)` array yields a 3-band Dataset.
        """
        arr = np.zeros((3, 16, 16), dtype="float32")
        ds = _array_to_dataset(arr, 4326, COG_GEOTRANSFORM, None)
        assert ds.band_count == 3, f"Expected 3 bands, got {ds.band_count}"

    def test_nodata_applied(self, int_array):
        """A provided nodata value is written to the dataset.

        Args:
            int_array: Fixture providing a 2-D int16 array.

        Test scenario:
            The first band's nodata equals the requested value.
        """
        ds = _array_to_dataset(int_array, 4326, COG_GEOTRANSFORM, -1)
        assert ds.no_data_value[0] == -1, f"nodata not applied: {ds.no_data_value}"

    def test_missing_crs_raises(self, float_array):
        """Omitting `crs` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            `crs=None` is rejected with a helpful message.
        """
        with pytest.raises(ValueError, match="crs.*transform|requires both"):
            _array_to_dataset(float_array, None, COG_GEOTRANSFORM, None)

    def test_missing_transform_raises(self, float_array):
        """Omitting `transform` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            `transform=None` is rejected with a helpful message.
        """
        with pytest.raises(ValueError, match="transform|requires both"):
            _array_to_dataset(float_array, 4326, None, None)


class TestNormalizeToDataset:
    """Tests for _normalize_to_dataset."""

    def test_dataset_passthrough(self, float_array):
        """An existing Dataset is returned as-is.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            The same object identity is preserved.
        """
        ds = Dataset.create_from_array(float_array, geo=COG_GEOTRANSFORM, epsg=4326)
        result = _normalize_to_dataset(ds, None, None, None)
        assert result is ds, "Existing Dataset should pass through unchanged"

    def test_gdal_dataset_wrapped(self, mem_dataset):
        """A raw gdal.Dataset is wrapped into a Dataset.

        Args:
            mem_dataset: Shared fixture providing a gdal.Dataset.

        Test scenario:
            The wrapped result is a pyramids Dataset of matching size.
        """
        result = _normalize_to_dataset(mem_dataset, None, None, None)
        assert isinstance(result, Dataset), "gdal.Dataset should wrap to Dataset"
        assert result.columns == 512, f"Expected 512 cols, got {result.columns}"

    def test_path_read(self, float_array, tmp_path):
        """A path to an existing raster is opened.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            A written GeoTIFF path is read back into a Dataset.
        """
        src = Dataset.create_from_array(float_array, geo=COG_GEOTRANSFORM, epsg=4326)
        p = tmp_path / "src.tif"
        src.to_file(str(p))
        result = _normalize_to_dataset(str(p), None, None, None)
        assert isinstance(result, Dataset), "Path should open into a Dataset"

    def test_ndarray_dispatch(self, float_array):
        """A NumPy array is dispatched to array construction.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            The result is a single-band Dataset.
        """
        result = _normalize_to_dataset(float_array, 4326, COG_GEOTRANSFORM, None)
        assert result.band_count == 1, f"Expected 1 band, got {result.band_count}"

    def test_nodata_set_on_prebuilt(self, mem_dataset):
        """A nodata override is applied to a pre-built dataset.

        Args:
            mem_dataset: Shared fixture providing a gdal.Dataset.

        Test scenario:
            Passing `nodata` updates every band's nodata marker.
        """
        result = _normalize_to_dataset(mem_dataset, None, None, 5.0)
        assert result.no_data_value[0] == pytest.approx(5.0), (
            f"nodata override not applied: {result.no_data_value}"
        )

    def test_unsupported_type_raises(self):
        """An unsupported input type raises TypeError.

        Test scenario:
            Passing an integer (not a raster) is rejected.
        """
        with pytest.raises(TypeError, match="write_cog accepts"):
            _normalize_to_dataset(42, None, None, None)


class TestWriteCog:
    """Tests for write_cog."""

    def test_float_array_writes_valid_cog(self, float_array, tmp_path):
        """A float array is written as a valid COG with a report.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The returned path exists and the report reports validity.
        """
        out = tmp_path / "float.tif"
        path, report = write_cog(float_array, out, crs=4326, transform=COG_GEOTRANSFORM)
        assert path.exists(), "Output COG should exist"
        assert report is not None and report.is_valid, (
            f"Expected a valid COG report, got {report}"
        )

    def test_delegates_to_to_cog_forwarding_options_as_extra(
        self, float_array, tmp_path, monkeypatch
    ):
        """write_cog is a thin delegator: it calls to_cog once, forwarding options.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            After ARC-1 the single write policy lives in COG.to_cog. write_cog
            must normalise the input and call ds.to_cog(output, extra=options)
            exactly once — it no longer pre-applies defaults, resolves the
            predictor, or owns the STATISTICS retry (that moved to the engine,
            ARC-4). The caller's options flow through unchanged as `extra`.
        """
        out = tmp_path / "delegate.tif"
        calls: list[dict] = []

        def fake_to_cog(self, output, *, extra=None):
            calls.append({"output": Path(output), "extra": extra})
            return Path(output)

        monkeypatch.setattr(Dataset, "to_cog", fake_to_cog)
        path, report = write_cog(
            float_array,
            out,
            crs=4326,
            transform=COG_GEOTRANSFORM,
            options={"COMPRESS": "ZSTD", "LEVEL": 18},
            validate=False,
        )
        assert len(calls) == 1, f"write_cog must call to_cog once, got {len(calls)}"
        assert calls[0]["extra"] == {
            "COMPRESS": "ZSTD",
            "LEVEL": 18,
        }, f"options must be forwarded verbatim as extra, got {calls[0]['extra']}"
        assert path == out, f"unexpected output path: {path}"
        assert report is None, "validate=False must yield report=None"

    def test_error_from_to_cog_propagates(self, float_array, tmp_path, monkeypatch):
        """An error raised by to_cog propagates out of write_cog unchanged.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            write_cog adds no error handling of its own; whatever to_cog raises
            surfaces to the caller. (The STATISTICS retry now lives in the
            engine and is covered in test_unified_write_policy.py.)
        """

        def fake_to_cog(self, output, *, extra=None):
            raise RuntimeError("disk full")

        monkeypatch.setattr(Dataset, "to_cog", fake_to_cog)
        with pytest.raises(RuntimeError, match="disk full"):
            write_cog(
                float_array,
                tmp_path / "boom.tif",
                crs=4326,
                transform=COG_GEOTRANSFORM,
                validate=False,
            )

    def test_int_array_writes_valid_cog(self, int_array, tmp_path):
        """An integer array is written as a valid COG.

        Args:
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The int path validates as a COG.
        """
        out = tmp_path / "int.tif"
        _, report = write_cog(int_array, out, crs=4326, transform=COG_GEOTRANSFORM)
        assert report.is_valid, f"Expected valid COG, errors: {report.errors}"

    def test_default_compression_is_deflate(self, float_array, tmp_path):
        """The house default compression (DEFLATE) is applied.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The written file reports DEFLATE compression.
        """
        out = tmp_path / "deflate.tif"
        write_cog(float_array, out, crs=4326, transform=COG_GEOTRANSFORM)
        assert _read_compression(out) == "DEFLATE", (
            "Default compression should be DEFLATE"
        )

    def test_options_override_compression(self, int_array, tmp_path):
        """User `options` override the house defaults.

        Args:
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            `options={'COMPRESS': 'LZW'}` produces an LZW COG.
        """
        out = tmp_path / "lzw.tif"
        write_cog(
            int_array,
            out,
            crs=4326,
            transform=COG_GEOTRANSFORM,
            options={"COMPRESS": "LZW"},
        )
        assert _read_compression(out) == "LZW", "COMPRESS override should win"

    def test_validate_false_skips_report(self, float_array, tmp_path):
        """`validate=False` writes the file and returns `None` report.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The file exists and the second return value is None.
        """
        out = tmp_path / "novalidate.tif"
        path, report = write_cog(
            float_array, out, crs=4326, transform=COG_GEOTRANSFORM, validate=False
        )
        assert path.exists(), "Output should still be written"
        assert report is None, (
            f"Report should be None when validate=False, got {report}"
        )

    def test_predictor_resolved_for_float(self, float_array, tmp_path):
        """A float raster is written with PREDICTOR=3.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            After ARC-1 the predictor is resolved inside COG.to_cog, so the
            assertion is on the written file rather than the call kwargs: a
            float source must carry the floating-point predictor (3).
        """
        out, _ = write_cog(
            float_array, tmp_path / "p.tif", crs=4326, transform=COG_GEOTRANSFORM
        )
        assert _read_predictor(out) == "3", "float source should yield PREDICTOR=3"

    def test_predictor_resolved_for_int(self, int_array, tmp_path):
        """An integer raster is written with PREDICTOR=2.

        Args:
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            An integer source must carry horizontal-differencing predictor (2).
        """
        out, _ = write_cog(
            int_array, tmp_path / "p.tif", crs=4326, transform=COG_GEOTRANSFORM
        )
        assert _read_predictor(out) == "2", "int source should yield PREDICTOR=2"

    def test_explicit_predictor_not_overridden(self, float_array, tmp_path):
        """A caller-supplied PREDICTOR overrides the dtype-aware default.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            `options={'PREDICTOR': 1}` forwards as `extra` and wins over the
            auto-resolved float default, disabling the predictor.
        """
        out, _ = write_cog(
            float_array,
            tmp_path / "p.tif",
            crs=4326,
            transform=COG_GEOTRANSFORM,
            options={"PREDICTOR": 1},
        )
        assert _read_predictor(out) in ("", "1"), "explicit predictor=1 should win"

    def test_from_gdal_dataset(self, mem_dataset, tmp_path):
        """A gdal.Dataset input is accepted and writes a valid COG.

        Args:
            mem_dataset: Shared fixture providing a gdal.Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            The 512x512 source produces a valid 512-tile COG.
        """
        out = tmp_path / "fromgdal.tif"
        _, report = write_cog(mem_dataset, out)
        assert report.is_valid, f"Expected valid COG, errors: {report.errors}"

    def test_from_existing_path(self, float_array, tmp_path):
        """A path input is read and re-encoded into a COG.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            A plain GeoTIFF path is read and rewritten as a valid COG.
            Statistics are disabled for this re-encode: GDAL's COG
            `STATISTICS=YES` pass can raise "no valid pixels found in
            sampling" when re-encoding a disk-read float source on some GDAL
            builds (it succeeds for in-memory sources). The default
            `STATISTICS=YES` path is covered by the in-memory write tests
            above; here we only assert the path → COG re-encode is valid.
        """
        src = Dataset.create_from_array(float_array, geo=COG_GEOTRANSFORM, epsg=4326)
        src_path = tmp_path / "plain.tif"
        src.to_file(str(src_path))
        out = tmp_path / "cog.tif"
        _, report = write_cog(str(src_path), out, options={"STATISTICS": False})
        assert report.is_valid, f"Expected valid COG, errors: {report.errors}"

    def test_missing_crs_for_array_raises(self, float_array, tmp_path):
        """A NumPy array without `crs` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The array branch requires both crs and transform.
        """
        with pytest.raises(ValueError, match="crs.*transform|requires both"):
            write_cog(float_array, tmp_path / "x.tif", transform=COG_GEOTRANSFORM)

    def test_invalid_cog_raises_runtime_error(self, mocker, float_array, tmp_path):
        """A failed validation surfaces as RuntimeError.

        Args:
            mocker: pytest-mock fixture.
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            When the validator reports invalid, write_cog raises and
            includes the validator errors.
        """
        mocker.patch.object(
            facade,
            "_validate_file",
            return_value=SimpleNamespace(is_valid=False, errors=["boom"]),
        )
        with pytest.raises(RuntimeError, match="invalid COG"):
            write_cog(
                float_array, tmp_path / "bad.tif", crs=4326, transform=COG_GEOTRANSFORM
            )

    def test_strict_forwarded_to_validator(self, mocker, float_array, tmp_path):
        """`strict` is forwarded to the validator.

        Args:
            mocker: pytest-mock fixture.
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            `strict=True` reaches `validate(strict=...)`.
        """
        spy = mocker.patch.object(
            facade,
            "_validate_file",
            return_value=SimpleNamespace(is_valid=True, errors=[]),
        )
        write_cog(
            float_array,
            tmp_path / "s.tif",
            crs=4326,
            transform=COG_GEOTRANSFORM,
            strict=True,
        )
        assert spy.call_args.kwargs["strict"] is True, "strict should be forwarded"


class TestPyramidsCogDefaults:
    """Tests for the PYRAMIDS_COG_DEFAULTS constant."""

    def test_predictor_absent(self):
        """PREDICTOR is intentionally not part of the static defaults.

        Test scenario:
            Predictor is resolved per dtype, so it must not be hardcoded.
        """
        assert "PREDICTOR" not in PYRAMIDS_COG_DEFAULTS, (
            "PREDICTOR must be resolved per dtype, not fixed in defaults"
        )

    def test_expected_house_defaults(self):
        """The static house defaults carry the documented values.

        Test scenario:
            DEFLATE, 512px tiles, and IF_SAFER BigTIFF. After ARC-3,
            OVERVIEW_RESAMPLING is no longer a static default — it is resolved
            per-dtype inside COG.to_cog — so it is intentionally absent here.
        """
        assert PYRAMIDS_COG_DEFAULTS["COMPRESS"] == "DEFLATE"
        assert PYRAMIDS_COG_DEFAULTS["BLOCKSIZE"] == 512
        assert PYRAMIDS_COG_DEFAULTS["BIGTIFF"] == "IF_SAFER"
        assert "OVERVIEW_RESAMPLING" not in PYRAMIDS_COG_DEFAULTS, (
            "overview resampling is dtype-resolved in to_cog, not a static default"
        )
