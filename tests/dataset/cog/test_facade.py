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
    _dataarray_to_dataset,
    _normalize_to_dataset,
    _resolve_predictor,
    write_cog,
)
from pyramids.dataset.dataset import Dataset

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


def _read_compression(path: str | Path) -> str:
    """Return the GDAL IMAGE_STRUCTURE compression token of a raster.

    Args:
        path: Path to a raster readable by GDAL.

    Returns:
        The compression name (e.g. ``"DEFLATE"``), or ``""`` when absent.
    """
    ds = gdal.Open(str(path))
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or ""
    ds = None
    return comp


@pytest.fixture
def float_array() -> np.ndarray:
    """A deterministic 2-D float32 array.

    Returns:
        A ``(64, 64)`` float32 array seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=1337)
    return (rng.random((64, 64)) * 100.0).astype("float32")


@pytest.fixture
def int_array() -> np.ndarray:
    """A deterministic 2-D int16 array.

    Returns:
        A ``(48, 48)`` int16 array seeded for reproducibility.
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
            ``_coerce_epsg(4326)`` returns ``4326`` without parsing.
        """
        assert _coerce_epsg(4326) == 4326, "Integer EPSG should pass through"

    def test_authority_string(self):
        """An ``EPSG:XXXX`` string resolves to its integer code.

        Test scenario:
            ``"EPSG:3857"`` resolves to ``3857``.
        """
        assert _coerce_epsg("EPSG:3857") == 3857, "Authority string should resolve"

    def test_wkt_string(self):
        """A WKT/PROJ string resolves to the matching EPSG code.

        Test scenario:
            The WGS84 PROJ4 string resolves to ``4326``.
        """
        assert (
            _coerce_epsg("+proj=longlat +datum=WGS84 +no_defs") == 4326
        ), "WGS84 PROJ string should resolve to 4326"

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
        ds = _array_to_dataset(float_array, 4326, _GEOTRANSFORM, None)
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_builds_dataset_3d(self):
        """A 3-D array becomes a multi-band Dataset.

        Test scenario:
            A ``(3, 16, 16)`` array yields a 3-band Dataset.
        """
        arr = np.zeros((3, 16, 16), dtype="float32")
        ds = _array_to_dataset(arr, 4326, _GEOTRANSFORM, None)
        assert ds.band_count == 3, f"Expected 3 bands, got {ds.band_count}"

    def test_nodata_applied(self, int_array):
        """A provided nodata value is written to the dataset.

        Args:
            int_array: Fixture providing a 2-D int16 array.

        Test scenario:
            The first band's nodata equals the requested value.
        """
        ds = _array_to_dataset(int_array, 4326, _GEOTRANSFORM, -1)
        assert ds.no_data_value[0] == -1, f"nodata not applied: {ds.no_data_value}"

    def test_missing_crs_raises(self, float_array):
        """Omitting ``crs`` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            ``crs=None`` is rejected with a helpful message.
        """
        with pytest.raises(ValueError, match="crs.*transform|requires both"):
            _array_to_dataset(float_array, None, _GEOTRANSFORM, None)

    def test_missing_transform_raises(self, float_array):
        """Omitting ``transform`` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            ``transform=None`` is rejected with a helpful message.
        """
        with pytest.raises(ValueError, match="transform|requires both"):
            _array_to_dataset(float_array, 4326, None, None)


class _Coord:
    """Minimal stand-in for an xarray coordinate (exposes ``.values``)."""

    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeDataArray:
    """Duck-typed DataArray for exercising _dataarray_to_dataset branches."""

    def __init__(self, values, x, y, attrs=None, rio=None):
        self.values = np.asarray(values)
        self._coords = {"longitude": _Coord(x), "latitude": _Coord(y)}
        self.coords = self._coords
        self.attrs = attrs or {}
        if rio is not None:
            self.rio = rio

    def __getitem__(self, key):
        return self._coords[key]


@pytest.mark.xarray
class TestDataArrayToDataset:
    """Tests for _dataarray_to_dataset (xarray paths)."""

    def _make_dataarray(self, *, attrs=None):
        """Build a small real xarray.DataArray with lon/lat coords.

        Args:
            attrs: Optional attribute dict to attach (e.g. a ``crs`` key).

        Returns:
            A ``(4, 5)`` float32 DataArray on a regular lon/lat grid.
        """
        xr = pytest.importorskip("xarray")
        data = np.arange(20, dtype="float32").reshape(4, 5)
        lat = np.array([12.0, 11.0, 10.0, 9.0])
        lon = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
        return xr.DataArray(
            data,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
            attrs=attrs or {},
        )

    def test_explicit_crs(self):
        """An explicit ``crs`` builds a Dataset from coordinate geometry.

        Test scenario:
            A lon/lat DataArray with ``crs=4326`` yields a 4x5 Dataset.
        """
        da = self._make_dataarray()
        ds = _dataarray_to_dataset(da, 4326, None)
        assert (ds.rows, ds.columns) == (4, 5), f"Unexpected shape {ds.shape}"
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_crs_from_attrs(self):
        """CRS is read from ``da.attrs['crs']`` when not passed explicitly.

        Test scenario:
            ``attrs={'crs': 4326}`` is honored.
        """
        da = self._make_dataarray(attrs={"crs": 4326})
        ds = _dataarray_to_dataset(da, None, None)
        assert ds.epsg == 4326, f"CRS from attrs not honored, got {ds.epsg}"

    def test_crs_from_rio_accessor(self):
        """CRS falls back to a ``.rio.crs`` accessor when present.

        Test scenario:
            A duck-typed DataArray exposing ``rio.crs`` is honored even
            though rioxarray is not a dependency.
        """
        fake = _FakeDataArray(
            np.zeros((2, 2), dtype="float32"),
            x=[30.0, 31.0],
            y=[11.0, 10.0],
            rio=SimpleNamespace(crs=4326),
        )
        ds = _dataarray_to_dataset(fake, None, None)
        assert ds.epsg == 4326, f"CRS from rio accessor not honored, got {ds.epsg}"

    def test_missing_coords_raises(self):
        """A DataArray lacking spatial coords raises ValueError.

        Test scenario:
            Coordinates named neither x/y nor lon/lat are rejected.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.zeros((2, 2), dtype="float32"),
            coords={"a": [0, 1], "b": [0, 1]},
            dims=("a", "b"),
        )
        with pytest.raises(ValueError, match="coordinates"):
            _dataarray_to_dataset(da, 4326, None)

    def test_single_cell_axis_raises(self):
        """Spatial coords with fewer than 2 cells raise ValueError.

        Test scenario:
            A 1x1 grid cannot yield a cell size and is rejected.
        """
        fake = _FakeDataArray(
            np.zeros((1, 1), dtype="float32"), x=[30.0], y=[10.0], rio=None
        )
        with pytest.raises(ValueError, match="at least 2 cells"):
            _dataarray_to_dataset(fake, 4326, None)

    def test_missing_crs_raises(self):
        """No explicit/embedded CRS raises ValueError.

        Test scenario:
            A DataArray with no crs attr and no rio accessor is rejected.
        """
        da = self._make_dataarray()
        with pytest.raises(ValueError, match="CRS"):
            _dataarray_to_dataset(da, None, None)


class TestNormalizeToDataset:
    """Tests for _normalize_to_dataset."""

    def test_dataset_passthrough(self, float_array):
        """An existing Dataset is returned as-is.

        Args:
            float_array: Fixture providing a 2-D float32 array.

        Test scenario:
            The same object identity is preserved.
        """
        ds = Dataset.create_from_array(float_array, geo=_GEOTRANSFORM, epsg=4326)
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
        src = Dataset.create_from_array(float_array, geo=_GEOTRANSFORM, epsg=4326)
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
        result = _normalize_to_dataset(float_array, 4326, _GEOTRANSFORM, None)
        assert result.band_count == 1, f"Expected 1 band, got {result.band_count}"

    def test_nodata_set_on_prebuilt(self, mem_dataset):
        """A nodata override is applied to a pre-built dataset.

        Args:
            mem_dataset: Shared fixture providing a gdal.Dataset.

        Test scenario:
            Passing ``nodata`` updates every band's nodata marker.
        """
        result = _normalize_to_dataset(mem_dataset, None, None, 5.0)
        assert (
            result.no_data_value[0] == 5.0
        ), f"nodata override not applied: {result.no_data_value}"

    def test_unsupported_type_raises(self):
        """An unsupported input type raises TypeError.

        Test scenario:
            Passing an integer (not a raster) is rejected.
        """
        with pytest.raises(TypeError, match="write_cog accepts"):
            _normalize_to_dataset(42, None, None, None)

    @pytest.mark.xarray
    def test_dataarray_dispatch(self):
        """A real xarray.DataArray is dispatched by its type name.

        Test scenario:
            An ``xr.DataArray`` routes to ``_dataarray_to_dataset`` and
            yields a Dataset of matching shape.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            np.arange(20, dtype="float32").reshape(4, 5),
            coords={
                "latitude": np.array([12.0, 11.0, 10.0, 9.0]),
                "longitude": np.array([30.0, 31.0, 32.0, 33.0, 34.0]),
            },
            dims=("latitude", "longitude"),
        )
        result = _normalize_to_dataset(da, 4326, None, None)
        assert (result.rows, result.columns) == (
            4,
            5,
        ), f"Unexpected shape {result.shape}"


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
        path, report = write_cog(float_array, out, crs=4326, transform=_GEOTRANSFORM)
        assert path.exists(), "Output COG should exist"
        assert (
            report is not None and report.is_valid
        ), f"Expected a valid COG report, got {report}"

    def test_int_array_writes_valid_cog(self, int_array, tmp_path):
        """An integer array is written as a valid COG.

        Args:
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The int path validates as a COG.
        """
        out = tmp_path / "int.tif"
        path, report = write_cog(int_array, out, crs=4326, transform=_GEOTRANSFORM)
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
        write_cog(float_array, out, crs=4326, transform=_GEOTRANSFORM)
        assert (
            _read_compression(out) == "DEFLATE"
        ), "Default compression should be DEFLATE"

    def test_options_override_compression(self, int_array, tmp_path):
        """User ``options`` override the house defaults.

        Args:
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            ``options={'COMPRESS': 'LZW'}`` produces an LZW COG.
        """
        out = tmp_path / "lzw.tif"
        write_cog(
            int_array,
            out,
            crs=4326,
            transform=_GEOTRANSFORM,
            options={"COMPRESS": "LZW"},
        )
        assert _read_compression(out) == "LZW", "COMPRESS override should win"

    def test_validate_false_skips_report(self, float_array, tmp_path):
        """``validate=False`` writes the file and returns ``None`` report.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The file exists and the second return value is None.
        """
        out = tmp_path / "novalidate.tif"
        path, report = write_cog(
            float_array, out, crs=4326, transform=_GEOTRANSFORM, validate=False
        )
        assert path.exists(), "Output should still be written"
        assert (
            report is None
        ), f"Report should be None when validate=False, got {report}"

    def test_predictor_resolved_for_float(self, mocker, float_array, tmp_path):
        """A float raster forwards PREDICTOR=3 to the writer.

        Args:
            mocker: pytest-mock fixture.
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            ``Dataset.to_cog`` is called with ``predictor=3`` and
            ``blocksize=512`` / ``overview_resampling='average'``.
        """
        spy = mocker.spy(Dataset, "to_cog")
        write_cog(float_array, tmp_path / "p.tif", crs=4326, transform=_GEOTRANSFORM)
        kwargs = spy.call_args.kwargs
        assert (
            kwargs["predictor"] == 3
        ), f"Expected predictor 3, got {kwargs['predictor']}"
        assert kwargs["blocksize"] == 512, "House default blocksize should be 512"
        assert (
            kwargs["overview_resampling"] == "average"
        ), "House default overview resampling should be average"

    def test_predictor_resolved_for_int(self, mocker, int_array, tmp_path):
        """An integer raster forwards PREDICTOR=2 to the writer.

        Args:
            mocker: pytest-mock fixture.
            int_array: Fixture providing a 2-D int16 array.
            tmp_path: pytest temp directory.

        Test scenario:
            ``Dataset.to_cog`` is called with ``predictor=2``.
        """
        spy = mocker.spy(Dataset, "to_cog")
        write_cog(int_array, tmp_path / "p.tif", crs=4326, transform=_GEOTRANSFORM)
        assert (
            spy.call_args.kwargs["predictor"] == 2
        ), "Int raster should use predictor 2"

    def test_explicit_predictor_not_overridden(self, mocker, float_array, tmp_path):
        """A caller-supplied PREDICTOR is not auto-resolved.

        Args:
            mocker: pytest-mock fixture.
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            ``options={'PREDICTOR': 1}`` is forwarded verbatim.
        """
        spy = mocker.spy(Dataset, "to_cog")
        write_cog(
            float_array,
            tmp_path / "p.tif",
            crs=4326,
            transform=_GEOTRANSFORM,
            options={"PREDICTOR": 1},
        )
        assert spy.call_args.kwargs["predictor"] == 1, "Explicit predictor should win"

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
        """A path input is re-encoded into a COG.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            A plain GeoTIFF path is read and rewritten as a valid COG.
        """
        src = Dataset.create_from_array(float_array, geo=_GEOTRANSFORM, epsg=4326)
        src_path = tmp_path / "plain.tif"
        src.to_file(str(src_path))
        out = tmp_path / "cog.tif"
        _, report = write_cog(str(src_path), out)
        assert report.is_valid, f"Expected valid COG, errors: {report.errors}"

    def test_missing_crs_for_array_raises(self, float_array, tmp_path):
        """A NumPy array without ``crs`` raises ValueError.

        Args:
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            The array branch requires both crs and transform.
        """
        with pytest.raises(ValueError, match="crs.*transform|requires both"):
            write_cog(float_array, tmp_path / "x.tif", transform=_GEOTRANSFORM)

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
                float_array, tmp_path / "bad.tif", crs=4326, transform=_GEOTRANSFORM
            )

    def test_strict_forwarded_to_validator(self, mocker, float_array, tmp_path):
        """``strict`` is forwarded to the validator.

        Args:
            mocker: pytest-mock fixture.
            float_array: Fixture providing a 2-D float32 array.
            tmp_path: pytest temp directory.

        Test scenario:
            ``strict=True`` reaches ``validate(strict=...)``.
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
            transform=_GEOTRANSFORM,
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
        assert (
            "PREDICTOR" not in PYRAMIDS_COG_DEFAULTS
        ), "PREDICTOR must be resolved per dtype, not fixed in defaults"

    def test_expected_house_defaults(self):
        """The house defaults carry the documented values.

        Test scenario:
            DEFLATE, 512px tiles, AVERAGE overviews, and IF_SAFER BigTIFF.
        """
        assert PYRAMIDS_COG_DEFAULTS["COMPRESS"] == "DEFLATE"
        assert PYRAMIDS_COG_DEFAULTS["BLOCKSIZE"] == 512
        assert PYRAMIDS_COG_DEFAULTS["OVERVIEW_RESAMPLING"] == "AVERAGE"
        assert PYRAMIDS_COG_DEFAULTS["BIGTIFF"] == "IF_SAFER"
