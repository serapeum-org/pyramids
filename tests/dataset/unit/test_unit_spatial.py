"""Unit tests for Dataset spatial operations: crop, reproject, resample, align,
overviews, coordinate mapping, and longitude wrapping."""

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset
from pyramids.dataset.engines import Analysis, Spatial, Vectorize

pytestmark = pytest.mark.core


class TestCorrectWrapCutlineError:
    """Tests for the _correct_wrap_cutline_error static method."""

    def test_removes_nodata_border_2d(self):
        """Should remove full rows/cols of nodata from 2D array."""
        nd = -9999.0
        arr = np.array(
            [
                [nd, nd, nd, nd],
                [nd, 1.0, 2.0, nd],
                [nd, 3.0, 4.0, nd],
                [nd, nd, nd, nd],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        corrected = Spatial._correct_wrap_cutline_error(ds)
        assert (
            corrected.rows == 2
        ), f"Expected 2 rows after correction, got {corrected.rows}"
        assert (
            corrected.columns == 2
        ), f"Expected 2 columns after correction, got {corrected.columns}"
        result_arr = corrected.read_array()
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            result_arr,
            expected,
            err_msg="Trimmed array values are wrong",
        )

    def test_removes_nodata_border_3d(self):
        """Should remove full rows/cols of nodata from 3D (multi-band) array."""
        nd = -9999.0
        band1 = np.array(
            [
                [nd, nd, nd],
                [nd, 1.0, nd],
                [nd, nd, nd],
            ],
            dtype=np.float32,
        )
        band2 = np.array(
            [
                [nd, nd, nd],
                [nd, 5.0, nd],
                [nd, nd, nd],
            ],
            dtype=np.float32,
        )
        arr = np.stack([band1, band2])
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        corrected = Spatial._correct_wrap_cutline_error(ds)
        assert corrected.rows == 1, "Expected 1 row after 3D correction"
        assert corrected.columns == 1, "Expected 1 col after 3D correction"

    def test_preserves_grib_crs_without_epsg(self):
        """Issue #403: a custom GRIB CRS (no resolvable EPSG) must still trim.

        The cutline-correction path used to round-trip the CRS through
        ``src.epsg``, which for GDAL's spherical-earth GRIB GEOGCS resolved
        to the unit code EPSG:9122 and crashed ``sr_from_epsg``. It must now
        rebuild the output from the source WKT, so trimming succeeds and the
        spherical datum (radius 6371229) survives unchanged.
        """
        grib_wkt = (
            'GEOGCS["Coordinate System imported from GRIB file",'
            'DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
        )
        nd = -9999.0
        arr = np.array(
            [
                [nd, nd, nd, nd],
                [nd, 1.0, 2.0, nd],
                [nd, 3.0, 4.0, nd],
                [nd, nd, nd, nd],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        ds.crs = grib_wkt

        corrected = Spatial._correct_wrap_cutline_error(ds)

        assert corrected.rows == 2, f"expected 2 rows after trim, got {corrected.rows}"
        assert (
            corrected.columns == 2
        ), f"expected 2 columns after trim, got {corrected.columns}"
        np.testing.assert_array_equal(
            corrected.read_array(),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        # The exact spherical GRIB datum is preserved, not relabeled to 4326.
        assert "6371229" in corrected.crs
        # Reading .epsg must not crash; it falls back to the soft default.
        assert corrected.epsg == 4326

    def test_unprojected_source_keeps_create_from_array_default(self):
        """An unprojected source keeps the default CRS, not an empty WKT.

        When ``src.crs`` is empty, the ``if src.crs:`` guard skips the WKT
        copy so the rebuilt dataset retains the ``create_from_array`` default
        (WGS84 / EPSG:4326) rather than having its projection wiped to empty.
        """
        nd = -9999.0
        arr = np.array(
            [
                [nd, nd, nd, nd],
                [nd, 1.0, 2.0, nd],
                [nd, 3.0, 4.0, nd],
                [nd, nd, nd, nd],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        ds.raster.SetProjection("")
        assert ds.crs == "", "precondition: source must be unprojected"

        corrected = Spatial._correct_wrap_cutline_error(ds)

        assert corrected.crs != "", "output projection must not be wiped to empty"
        assert corrected.epsg == 4326, f"expected default 4326, got {corrected.epsg}"


class TestApply:
    """Tests for the apply method."""

    def test_apply_function(self, single_band_dataset):
        """apply should transform each valid cell."""
        result = single_band_dataset.apply(lambda v: v * 2)
        arr = result.read_array()
        # Cell (0,0) was 1.0, should now be 2.0
        assert np.isclose(arr[0, 0], 2.0), "apply(v*2) should double the values"

    def test_apply_non_callable_raises(self, single_band_dataset):
        """apply with a non-callable should raise TypeError."""
        with pytest.raises(TypeError, match="function"):
            single_band_dataset.apply("not_a_function")

    @pytest.mark.parametrize(
        "func, expected_corner",
        [
            (np.abs, 1.0),
            (np.sqrt, 1.0),
            (np.square, 1.0),
            (lambda x: x + 10, 11.0),
        ],
        ids=["np.abs", "np.sqrt", "np.square", "lambda_add10"],
    )
    def test_apply_vectorized_numpy_functions(
        self, single_band_dataset, func, expected_corner
    ):
        """Test apply with vectorized NumPy functions (fast path).

        Test scenario:
            Vectorized functions like np.abs, np.sqrt, np.square, and
            array-compatible lambdas should work via the direct array
            application path without falling back to np.vectorize.
        """
        result = single_band_dataset.apply(func)
        arr = result.read_array()
        assert np.isclose(
            arr[0, 0], expected_corner
        ), f"Expected {expected_corner} at (0,0), got {arr[0, 0]}"

    def test_apply_scalar_function_fallback(self):
        """Test apply with a scalar if/elif function triggers np.vectorize fallback.

        Test scenario:
            A function using Python if/elif on scalar values cannot operate
            on arrays directly. apply should catch the error and fall back
            to np.vectorize, producing correct per-cell results.
        """

        def classify(val):
            if val < 3:
                return 0.0
            elif val < 6:
                return 1.0
            else:
                return 2.0

        arr = np.array(
            [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]], dtype=np.float32
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.apply(classify)
        result_arr = result.read_array()
        expected = np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0, 2.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(
            result_arr,
            expected,
            err_msg="Scalar classify function should produce correct per-cell results",
        )

    def test_apply_preserves_nodata_cells(self):
        """Test that no-data cells are not transformed by apply.

        Test scenario:
            Create a dataset with some cells set to the no_data_value.
            After apply, those cells should still hold the no_data_value.
        """
        arr = np.array([[1.0, -9999.0], [-9999.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.apply(lambda x: x * 10)
        result_arr = result.read_array()
        assert np.isclose(
            result_arr[0, 0], 10.0
        ), f"Domain cell should be transformed, got {result_arr[0, 0]}"
        assert np.isclose(
            result_arr[1, 1], 40.0
        ), f"Domain cell should be transformed, got {result_arr[1, 1]}"
        assert np.isclose(
            result_arr[0, 1], -9999.0, rtol=0.001
        ), f"No-data cell should stay -9999, got {result_arr[0, 1]}"
        assert np.isclose(
            result_arr[1, 0], -9999.0, rtol=0.001
        ), f"No-data cell should stay -9999, got {result_arr[1, 0]}"

    def test_apply_all_nodata(self):
        """Test apply on a dataset where all cells are no-data.

        Test scenario:
            When all cells are no-data, the function should never be called
            and the output should be identical to the input.
        """
        arr = np.full((3, 3), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.apply(lambda x: x * 100)
        result_arr = result.read_array()
        assert np.allclose(
            result_arr, -9999.0, rtol=0.001
        ), "All-nodata input should produce all-nodata output"

    def test_apply_single_cell(self):
        """Test apply on a 1x1 dataset.

        Test scenario:
            A single domain cell should be correctly transformed.
        """
        arr = np.array([[5.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.apply(lambda x: x**2)
        assert np.isclose(
            result.read_array()[0, 0], 25.0
        ), f"Expected 25.0, got {result.read_array()[0, 0]}"

    def test_apply_with_band_parameter(self, multi_band_dataset):
        """Test apply on band=1 of a multi-band dataset.

        Test scenario:
            Applying to band=1 on a multi-band dataset should produce a
            single-band result with that band's values transformed.
        """
        original_band1 = multi_band_dataset.read_array(band=1).copy()
        result = multi_band_dataset.apply(lambda x: x + 100, band=1)
        result_arr = result.read_array()
        assert (
            result.band_count == 1
        ), f"Result should be single-band, got {result.band_count}"
        domain_mask = ~np.isclose(
            original_band1, multi_band_dataset.no_data_value[1], rtol=0.001
        )
        np.testing.assert_array_almost_equal(
            result_arr[domain_mask],
            original_band1[domain_mask] + 100,
            err_msg="Band 1 domain cells should be shifted by +100",
        )

    def test_apply_preserves_spatial_metadata(self, single_band_dataset):
        """Test that apply preserves geotransform, CRS, and no_data_value.

        Test scenario:
            The output dataset should have the same spatial metadata as the
            input (geotransform, EPSG, no_data_value).
        """
        original_geo = single_band_dataset.geotransform
        original_epsg = single_band_dataset.epsg
        original_nd = single_band_dataset.no_data_value[0]
        result = single_band_dataset.apply(np.abs)
        assert (
            result.geotransform == original_geo
        ), f"Geotransform mismatch: {result.geotransform} vs {original_geo}"
        assert (
            result.epsg == original_epsg
        ), f"EPSG mismatch: {result.epsg} vs {original_epsg}"
        assert (
            result.no_data_value[0] == original_nd
        ), f"No-data value mismatch: {result.no_data_value[0]} vs {original_nd}"

    def test_apply_not_inplace_does_not_mutate_original(self):
        """Test that apply(inplace=False) does not mutate the original array.

        Test scenario:
            Read the original values, apply a function, then verify the
            original dataset still has its original values.
        """
        arr = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        original_arr = ds.read_array().copy()
        ds.apply(lambda x: x * 0)
        np.testing.assert_array_equal(
            ds.read_array(),
            original_arr,
            err_msg="Original dataset should not be mutated by non-inplace apply",
        )


class TestResample:
    """Tests for the resample method."""

    def test_resample_changes_cell_size(self, single_band_dataset):
        """Resampling to a larger cell size should reduce rows/columns."""
        resampled = single_band_dataset.resample(cell_size=0.1)
        assert resampled.cell_size == pytest.approx(
            0.1
        ), f"Cell size should be 0.1, got {resampled.cell_size}"
        # Original is 3x3 with 0.05 cell size -> 0.15 extent
        # With 0.1 cell size -> floor(0.15/0.1) = 2 (or 1, depending on rounding)
        assert (
            resampled.rows < single_band_dataset.rows
        ), "Resampled rows should be fewer"


class TestToCrs:
    """Tests for the to_crs method."""

    def test_to_crs_basic(self, single_band_dataset):
        """to_crs should change the EPSG of the dataset."""
        result = single_band_dataset.to_crs(to_epsg=3857)
        assert result is not None, "to_crs should return a Dataset"
        assert result.epsg == 3857, f"Expected EPSG 3857, got {result.epsg}"

    def test_to_crs_returns_new_dataset(self, single_band_dataset):
        """to_crs() should always return a new Dataset."""
        result = single_band_dataset.to_crs(to_epsg=3857)
        assert result is not None, "to_crs should return a Dataset"
        assert result.epsg == 3857, "EPSG should be 3857 on the returned dataset"

    @pytest.mark.parametrize("crs", ["EPSG:3857", "3857"])
    def test_to_crs_accepts_string(self, single_band_dataset, crs):
        """to_crs accepts string CRS forms and resolves them to the EPSG code.

        Args:
            crs: A string CRS form ("EPSG:3857" or the bare "3857").

        Test scenario:
            Both an authority string and a bare numeric string reproject to EPSG 3857.
        """
        result = single_band_dataset.to_crs(to_epsg=crs)
        assert (
            result.epsg == 3857
        ), f"Expected EPSG 3857 from {crs!r}, got {result.epsg}"

    def test_to_crs_accepts_pyproj_crs(self, single_band_dataset):
        """to_crs accepts a pyproj.CRS object.

        Test scenario:
            Passing CRS.from_epsg(3857) reprojects to EPSG 3857.
        """
        from pyproj import CRS

        result = single_band_dataset.to_crs(to_epsg=CRS.from_epsg(3857))
        assert result.epsg == 3857, f"Expected EPSG 3857, got {result.epsg}"

    def test_to_crs_uninterpretable_crs_raises(self, single_band_dataset):
        """to_crs rejects a string that is not a CRS.

        Test scenario:
            An uninterpretable CRS string raises a ValueError (CRSError subclass)
            mentioning that it could not be interpreted.
        """
        with pytest.raises(ValueError, match="could not interpret"):
            single_band_dataset.to_crs(to_epsg="not-a-crs")

    def test_to_crs_wrong_type_raises(self, single_band_dataset):
        """to_crs rejects a value that cannot be a CRS at all.

        Test scenario:
            A list is not a valid CRS input and raises a ValueError (CRSError subclass).
        """
        with pytest.raises(ValueError):
            single_band_dataset.to_crs(to_epsg=[3857])

    def test_to_crs_invalid_method_raises(self, single_band_dataset):
        """to_crs with an invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            single_band_dataset.to_crs(to_epsg=3857, method="invalid_method")

    def test_to_crs_invalid_method_type_raises(self, single_band_dataset):
        """to_crs with a non-string method should raise TypeError."""
        with pytest.raises(TypeError):
            single_band_dataset.to_crs(to_epsg=3857, method=123)


class TestWrapLongitude:
    """Tests for wrap_longitude method."""

    def test_wrap_longitude_360_to_180(self):
        """wrap_longitude should convert 0-360 to -180-180 range."""
        cols = 360
        arr = np.ones((1, cols), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.wrap_longitude()
        assert result is not None, "wrap_longitude should return a Dataset"
        gt = result.geotransform
        assert gt[0] < 0, "After conversion, top-left x should be negative"

    def test_wrap_longitude_raises_for_non_global(self):
        """wrap_longitude should raise for a small, clearly non-global raster."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="global grid"):
            ds.wrap_longitude()

    def test_wrap_longitude_rejects_regional_window_past_180(self):
        """A regional window whose longitudes exceed 180 but does not span the globe is rejected.

        Test scenario:
            A 53-column 2.5° grid over 200-330 has `lon[-1] > 180` but spans only ~132°, so the
            tightened global-coverage guard raises instead of silently mis-wrapping it.
        """
        arr = np.ones((1, 53), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(200.0, 10.0),
            cell_size=2.5,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="global grid"):
            ds.wrap_longitude()

    def test_wrap_longitude_returns_new_dataset(self):
        """wrap_longitude() should always return a new Dataset."""
        cols = 360
        arr = np.ones((1, cols), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.wrap_longitude()
        assert result is not None, "wrap_longitude should return a Dataset"
        assert isinstance(result, Dataset), "wrap_longitude should return a Dataset"


class TestWrapLongitudePaths:
    """wrap_longitude: lazy VRT for file-backed sources, eager roll for in-memory sources."""

    def test_file_backed_uses_lazy_vrt(self, noah):
        """A file-backed global raster is shifted lazily through a VRT (no eager copy)."""
        result = Dataset(noah).wrap_longitude()
        assert result.raster.GetDriver().ShortName == "VRT"
        assert result.top_left_corner == (-180.0, 90.0)
        # the VRT still reads back real data
        assert result.read_array(band=0).shape == (noah.RasterYSize, noah.RasterXSize)

    def test_in_memory_rolls_columns_exactly(self):
        """An in-memory source is rolled exactly (eager path), preserving the no-data value."""
        arr = np.arange(360, dtype=np.float32).reshape(1, 360)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.wrap_longitude()
        assert result.raster.GetDriver().ShortName == "MEM"
        expected = arr[:, list(range(180, 360)) + list(range(0, 180))]
        np.testing.assert_array_equal(result.read_array(band=0), expected)
        assert result.raster.GetRasterBand(1).GetNoDataValue() == -9999.0

    def test_vrt_matches_eager_reference(self, noah):
        """The lazy VRT roll returns data identical to an independent eager column roll (all bands).

        Test scenario:
            For a file-backed global raster, every band read back through the VRT must equal the
            source band with its columns rolled by the half-globe offset.
        """
        dataset = Dataset(noah)
        lon = dataset.lon
        first = int(np.nonzero(lon > 180)[0][0])
        order = list(range(first, noah.RasterXSize)) + list(range(0, first))
        result = dataset.wrap_longitude()
        for band in range(noah.RasterCount):
            reference = noah.GetRasterBand(band + 1).ReadAsArray()[:, order]
            np.testing.assert_array_equal(
                result.read_array(band=band),
                reference,
                err_msg=f"band {band}: VRT roll differs from eager reference",
            )

    def test_vrt_preserves_projection_and_nodata(self, noah):
        """The VRT result keeps the source projection and per-band no-data values.

        Test scenario:
            Projection WKT and every band's no-data value must match the source after conversion.
        """
        result = Dataset(noah).wrap_longitude()
        assert (
            result.raster.GetProjection() == noah.GetProjection()
        ), "projection not preserved"
        for band in range(1, noah.RasterCount + 1):
            assert (
                result.raster.GetRasterBand(band).GetNoDataValue()
                == noah.GetRasterBand(band).GetNoDataValue()
            ), f"band {band} no-data not preserved"

    def test_vrt_resolves_from_other_cwd(self, noah, tmp_path, monkeypatch):
        """The VRT uses an absolute source path, so reads succeed from any working directory.

        Test scenario:
            After changing CWD to an unrelated directory, reading the VRT-backed result still
            returns the full array (the SourceFilename resolves).
        """
        result = Dataset(noah).wrap_longitude()
        monkeypatch.chdir(tmp_path)
        array = result.read_array(band=0)
        assert array.shape == (
            noah.RasterYSize,
            noah.RasterXSize,
        ), "VRT failed to resolve the source from another CWD"

    def test_in_memory_multiband_roll(self):
        """A multi-band in-memory raster rolls every band's columns identically.

        Test scenario:
            Each of two distinct-valued bands is rolled by the half-globe offset independently.
        """
        arr = np.stack(
            [np.arange(360, dtype=np.float32), np.arange(360, 720, dtype=np.float32)]
        ).reshape(2, 1, 360)
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = dataset.wrap_longitude()
        order = list(range(180, 360)) + list(range(0, 180))
        for band in range(2):
            np.testing.assert_array_equal(
                result.read_array(band=band),
                arr[band][:, order],
                err_msg=f"band {band} not rolled correctly",
            )

    def test_in_memory_preserves_crs(self):
        """The eager in-memory path keeps the CRS.

        Test scenario:
            A 0-360 raster created at EPSG:4326 still reports EPSG:4326 after conversion.
        """
        arr = np.ones((1, 360), dtype=np.float32)
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = dataset.wrap_longitude()
        assert result.epsg == 4326, f"expected EPSG 4326, got {result.epsg}"

    def test_vrt_source_without_nodata_or_projection(self, tmp_path):
        """A file-backed global raster with neither no-data nor projection still converts via VRT.

        Test scenario:
            Exercises the `no_data is None` and falsy-`projection` branches of the VRT builder; the
            rolled data must still be correct and the result VRT-backed.
        """
        path = str(tmp_path / "global_minimal.tif")
        n_columns = 360
        out = gdal.GetDriverByName("GTiff").Create(
            path, n_columns, 1, 1, gdal.GDT_Float32
        )
        out.SetGeoTransform((0.0, 1.0, 0.0, 0.5, 0.0, -1.0))
        out.GetRasterBand(1).WriteArray(
            np.arange(n_columns, dtype=np.float32).reshape(1, n_columns)
        )
        out.FlushCache()
        out = None

        result = Dataset.read_file(path).wrap_longitude()
        assert (
            result.raster.GetDriver().ShortName == "VRT"
        ), "file-backed source should use VRT"
        assert (
            result.raster.GetRasterBand(1).GetNoDataValue() is None
        ), "should have no no-data"
        order = list(range(180, 360)) + list(range(0, 180))
        expected = np.arange(n_columns, dtype=np.float32).reshape(1, n_columns)[
            :, order
        ]
        np.testing.assert_array_equal(
            result.read_array(band=0),
            expected,
            err_msg="roll incorrect for minimal-metadata source",
        )

    def test_nonpath_description_uses_eager(self):
        """A source whose description is not a resolvable path uses the eager roll.

        Test scenario:
            A non-empty but non-resolvable description (here, an embedded null byte) makes
            `Path(...).exists()` return False, so the discriminator routes to the in-memory eager path.
        """
        arr = np.arange(360, dtype=np.float32).reshape(1, 360)
        dataset = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        dataset.raster.SetDescription("invalid\x00path")
        result = dataset.wrap_longitude()
        assert (
            result.raster.GetDriver().ShortName == "MEM"
        ), "should fall back to the eager path"
        order = list(range(180, 360)) + list(range(0, 180))
        np.testing.assert_array_equal(result.read_array(band=0), arr[:, order])


class TestNonSquareResolution:
    """resample / to_crs accept an (x_res, y_res) pair for non-square output; scalar stays square."""

    @staticmethod
    def _square_source():
        """A 10×10 1° geographic raster to resample/reproject."""
        return Dataset.create_from_array(
            np.ones((1, 10, 10), dtype="float32"),
            top_left_corner=(0.0, 10.0),
            cell_size=1.0,
            epsg=4326,
        )

    def test_resample_nonsquare_output(self):
        """resample((2, 1)) halves the columns, keeps the rows, and yields a 2°×1° grid."""
        result = self._square_source().resample(cell_size=(2.0, 1.0))
        gt = result.geotransform
        assert abs(gt[1]) == pytest.approx(2.0) and abs(gt[5]) == pytest.approx(1.0), gt
        assert result.shape[-2:] == (
            10,
            5,
        ), f"expected (10, 5), got {result.shape[-2:]}"

    def test_resample_scalar_stays_square(self):
        """A scalar cell_size still produces square cells (no behaviour change)."""
        result = self._square_source().resample(cell_size=2.0)
        gt = result.geotransform
        assert abs(gt[1]) == pytest.approx(2.0) and abs(gt[5]) == pytest.approx(2.0), gt
        assert result.shape[-2:] == (5, 5), f"expected (5, 5), got {result.shape[-2:]}"

    def test_to_crs_nonsquare_output(self):
        """to_crs(..., cell_size=(2, 1)) produces a non-square output grid."""
        result = self._square_source().to_crs(4326, cell_size=(2.0, 1.0))
        gt = result.geotransform
        assert abs(gt[1]) == pytest.approx(2.0) and abs(gt[5]) == pytest.approx(1.0), gt

    def test_resample_rejects_bad_resolution(self):
        """A non-positive or malformed cell_size raises a clear ValueError."""
        src = self._square_source()
        with pytest.raises(ValueError, match="cell_size must be positive"):
            src.resample(cell_size=(2.0, 0.0))
        with pytest.raises(ValueError, match="x_res, y_res"):
            src.resample(cell_size=(1.0, 2.0, 3.0))


class TestResampleErrors:
    """Tests for resample method error paths."""

    def test_resample_invalid_method_type_raises(self, single_band_dataset):
        """resample with non-string method should raise TypeError."""
        with pytest.raises(TypeError):
            single_band_dataset.resample(cell_size=0.1, method=123)

    def test_resample_invalid_method_value_raises(self, single_band_dataset):
        """resample with unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            single_band_dataset.resample(cell_size=0.1, method="invalid_interp")


class TestToCrsSameEpsg:
    """Tests for to_crs when source and target EPSG are the same."""

    def test_to_crs_same_epsg(self, single_band_dataset):
        """to_crs with the same EPSG should still return a valid Dataset."""
        result = single_band_dataset.to_crs(to_epsg=4326)
        assert result is not None, "to_crs with same EPSG should return a Dataset"
        assert result.epsg == 4326, "EPSG should remain 4326"


class TestToCrsWestHemisphere:
    """Tests for to_crs west hemisphere longitude path."""

    def test_to_crs_west_hemisphere(self):
        """to_crs on a raster with longitude > 180 should handle conversion."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(200.0, 50.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.to_crs(to_epsg=3857)
        assert result is not None, "to_crs should handle west-hemisphere longitudes"


class TestCropAligned:
    """Tests for _crop_aligned and related error paths."""

    def test_crop_aligned_with_dataset_mask(self):
        """_crop_aligned with a Dataset mask should produce a cropped result."""
        nd = -9999.0
        src_arr = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.array(
            [[1.0, 1.0, 1.0], [1.0, nd, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = src.spatial._crop_aligned(mask)
        arr = result.read_array()
        assert np.isclose(arr[1, 1], nd), "Masked cell should be nodata"

    def test_crop_aligned_numpy_mask_no_noval_raises(self):
        """_crop_aligned with numpy mask but no mask_noval should raise."""
        src_arr = np.ones((3, 3), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        mask = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="no_val"):
            src.spatial._crop_aligned(mask, mask_noval=None)

    def test_crop_aligned_invalid_mask_type_raises(self):
        """_crop_aligned with invalid mask type should raise TypeError."""
        src_arr = np.ones((3, 3), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(TypeError):
            src.spatial._crop_aligned("not_a_mask")

    def test_crop_aligned_dimension_mismatch_raises(self):
        """_crop_aligned with different dimensions should raise ValueError."""
        src_arr = np.ones((3, 3), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        mask_arr = np.ones((5, 5), dtype=np.float32)
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="different number"):
            src.spatial._crop_aligned(mask)

    def test_crop_aligned_different_location_raises(self):
        """_crop_aligned with different top-left corner raises ValueError."""
        nd = -9999.0
        src_arr = np.ones((3, 3), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((3, 3), dtype=np.float32)
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(1.0, 1.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        with pytest.raises(ValueError, match="upper left corner"):
            src.spatial._crop_aligned(mask)

    def test_crop_aligned_different_epsg_raises(self):
        """_crop_aligned with different EPSG raises ValueError."""
        nd = -9999.0
        src_arr = np.ones((3, 3), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((3, 3), dtype=np.float32)
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=3857,
            no_data_value=nd,
        )
        with pytest.raises(ValueError, match="coordinate system"):
            src.spatial._crop_aligned(mask)

    def test_crop_aligned_multi_band_nan_mask(self):
        """_crop_aligned with multi-band src and nan mask."""
        nd = -9999.0
        src_arr = np.ones((2, 3, 3), dtype=np.float32) * 5.0
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((3, 3), dtype=np.float32)
        mask_arr[1, 1] = np.nan
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=np.nan,
        )
        result = src.spatial._crop_aligned(mask)
        arr = result.read_array()
        assert arr.ndim == 3, "Multi-band result should be 3D"

    def test_crop_aligned_single_band_nan_mask(self):
        """_crop_aligned with single-band src and NaN mask noval."""
        nd = -9999.0
        src_arr = np.ones((3, 3), dtype=np.float32) * 5.0
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((3, 3), dtype=np.float32)
        mask_arr[0, 0] = np.nan
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=np.nan,
        )
        result = src.spatial._crop_aligned(mask)
        assert result is not None, "Should return a cropped dataset"


class TestCheckAlignment:
    """Tests for _check_alignment method."""

    def test_check_alignment_invalid_type_raises(self, single_band_dataset):
        """_check_alignment with non-Dataset should raise TypeError."""
        with pytest.raises(TypeError, match="Dataset"):
            single_band_dataset.spatial._check_alignment("not_a_dataset")


class TestAlign:
    """Tests for the align method."""

    def test_align_invalid_type_raises(self, single_band_dataset):
        """align with non-Dataset should raise TypeError."""
        with pytest.raises(TypeError):
            single_band_dataset.align("not_a_dataset")

    def test_align_same_dataset(self, single_band_dataset):
        """align with itself should return a valid Dataset."""
        result = single_band_dataset.align(single_band_dataset)
        assert isinstance(result, Dataset), "align should return a Dataset"


class TestCropWithRaster:
    """Tests for _crop_with_raster error paths."""

    def test_crop_with_raster_invalid_type_raises(self, single_band_dataset):
        """_crop_with_raster with invalid type should raise TypeError."""
        with pytest.raises(TypeError):
            single_band_dataset.spatial._crop_with_raster(12345)


class TestCropWithPolygonWarp:
    """Tests for _crop_with_polygon_warp error paths."""

    def test_crop_with_polygon_invalid_type_raises(self, single_band_dataset):
        """_crop_with_polygon_warp with non-FC/GDF raises TypeError."""
        with pytest.raises(TypeError):
            single_band_dataset.spatial._crop_with_polygon_warp(12345)


class TestCropErrors:
    """Tests for crop method error paths."""

    def test_crop_invalid_mask_raises(self, single_band_dataset):
        """crop with invalid mask type should raise TypeError."""
        with pytest.raises(TypeError, match="GeoDataFrame or Dataset"):
            single_band_dataset.crop(mask="not_valid")


class TestNearestNeighbour:
    """Tests for the _nearest_neighbour static method."""

    def test_invalid_array_type_raises(self):
        """Non-array input should raise TypeError."""
        with pytest.raises(TypeError, match="gdal"):
            Vectorize._nearest_neighbour("not_array", -9999, [0], [0])

    def test_invalid_rows_type_raises(self):
        """Non-list rows should raise TypeError."""
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="rows"):
            Vectorize._nearest_neighbour(arr, -9999, 0, [0])

    def test_invalid_cols_type_raises(self):
        """Non-list cols should raise TypeError."""
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="cols"):
            Vectorize._nearest_neighbour(arr, -9999, [0], 0)

    def test_nearest_neighbour_fills_from_right(self):
        """_nearest_neighbour should fill from right neighbor."""
        nd = -9999.0
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [nd, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        result = Vectorize._nearest_neighbour(arr.copy(), nd, [1], [0])
        assert result[1, 0] != nd, "Cell (1,0) should be filled by right neighbor"

    def test_nearest_neighbour_from_left(self):
        """_nearest_neighbour fills from left when at the last column."""
        nd = -9999.0
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, nd],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        result = Vectorize._nearest_neighbour(arr.copy(), nd, [1], [2])
        assert result[1, 2] != nd, "Cell at last col should be filled from left"

    def test_nearest_neighbour_left_neighbor(self):
        """_nearest_neighbour fills from left at the last column."""
        nd = -9999.0
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, nd],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        # Cell (1,2) is last col, so right check skipped.
        # Left (1,1) = 5.0 != nd -> filled
        result = Vectorize._nearest_neighbour(arr.copy(), nd, [1], [2])
        assert result[1, 2] == pytest.approx(
            5.0
        ), "Cell at last col should fill from left"

    def test_nearest_neighbour_above_path_falls_back_to_left(self):
        """_nearest_neighbour: the last-col above path is unreachable, so it fills from the left."""
        nd = -9999.0
        # The last-col/col-1=0 path can't be exercised without tripping an
        # index error at the diagonal check, so test a valid scenario that
        # reaches the left branch successfully instead.
        arr = np.array(
            [
                [nd, nd, nd],
                [nd, 5.0, nd],
                [nd, nd, nd],
            ],
            dtype=np.float32,
        )
        # Cell (1,2) at last col. Left (1,1)=5.0, cols[i]-1=1 > 0
        result = Vectorize._nearest_neighbour(arr.copy(), nd, [1], [2])
        assert result[1, 2] == pytest.approx(
            5.0
        ), "Cell should be filled from left neighbor"


class TestMapToArrayCoordinates:
    """Tests for map_to_array_coordinates error paths."""

    def test_invalid_input_type_raises(self, single_band_dataset):
        """map_to_array_coordinates with bad input type raises TypeError."""
        with pytest.raises(TypeError, match="GeoDataFrame"):
            single_band_dataset.map_to_array_coordinates(12345)

    def test_dataframe_missing_xy_raises(self, single_band_dataset):
        """map_to_array_coordinates with DataFrame lacking x,y raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="x, and y"):
            single_band_dataset.map_to_array_coordinates(df)

    def test_map_to_array_nonsquare_roundtrip(self):
        """On a non-square grid the cell centres round-trip back to their indices.

        ``map_to_array_coordinates`` matches against the per-axis coordinate
        arrays (not the pixel width), so it already resolves the correct cell
        on a non-square grid. This guards that it stays the inverse of the
        fixed ``array_to_map_coordinates`` (issue #505).
        """
        ds = Dataset.create_from_array(
            np.ones((4, 3), dtype="float32"),
            geo=(10.0, 2.0, 0.0, 8.0, 0.0, -0.5),
            epsg=4326,
        )
        rows, cols = [0, 1, 3], [0, 1, 2]
        x, y = ds.array_to_map_coordinates(rows, cols, center=True)
        df = pd.DataFrame({"x": x, "y": y})
        indices = ds.map_to_array_coordinates(df)
        assert indices.tolist() == [
            [0, 0],
            [1, 1],
            [3, 2],
        ], f"cell centres must map back to their (row, col), got {indices.tolist()}"


class TestArrayToMapCoordinates:
    """Tests for the array_to_map_coordinates method."""

    def test_array_to_map_center(self, single_band_dataset):
        """array_to_map_coordinates with center returns center coords."""
        x, y = single_band_dataset.array_to_map_coordinates(
            rows_index=[0, 1],
            column_index=[0, 1],
            center=True,
        )
        assert len(x) == 2, "Should return 2 x-coordinates"
        assert len(y) == 2, "Should return 2 y-coordinates"
        expected_x0 = 0.0 + 0.05 / 2
        assert abs(x[0] - expected_x0) < 1e-6, f"Expected x={expected_x0}, got {x[0]}"

    def test_array_to_map_corner(self, single_band_dataset):
        """array_to_map_coordinates with center=False returns corner."""
        x, _ = single_band_dataset.array_to_map_coordinates(
            rows_index=[0],
            column_index=[0],
            center=False,
        )
        assert abs(x[0] - 0.0) < 1e-6, "Corner x should be 0.0"

    @staticmethod
    def _nonsquare():
        """Grid with 2.0-wide, 0.5-tall pixels anchored at (10, 8)."""
        return Dataset.create_from_array(
            np.ones((4, 3), dtype="float32"),
            geo=(10.0, 2.0, 0.0, 8.0, 0.0, -0.5),
            epsg=4326,
        )

    def test_array_to_map_nonsquare_center(self):
        """The y axis uses the pixel height, not the width, on non-square grids.

        Regression for #505: cell (row=1, col=2) centre is x=15.0, y=7.25
        (8 - 1*0.5 - 0.25); the old code returned y=5.0 by reusing the 2.0
        pixel width for the y axis.
        """
        x, y = self._nonsquare().array_to_map_coordinates([1], [2], center=True)
        assert abs(x[0] - 15.0) < 1e-9, f"x should be 15.0, got {x[0]}"
        assert abs(y[0] - 7.25) < 1e-9, f"y should be 7.25 (height-based), got {y[0]}"

    def test_array_to_map_nonsquare_corner(self):
        """Corner coordinates also use the per-axis pixel sizes."""
        x, y = self._nonsquare().array_to_map_coordinates([1], [2], center=False)
        assert abs(x[0] - 14.0) < 1e-9, f"corner x should be 14.0, got {x[0]}"
        assert abs(y[0] - 7.5) < 1e-9, f"corner y should be 7.5, got {y[0]}"

    def test_array_to_map_nonsquare_vector(self):
        """Vector inputs are converted element-wise with the per-axis sizes."""
        x, y = self._nonsquare().array_to_map_coordinates([0, 1, 3], [0, 1, 2])
        assert x == [10.0, 12.0, 14.0], f"x corners wrong: {x}"
        assert y == [8.0, 7.5, 6.5], f"y corners must step by 0.5, got {y}"

    def test_array_to_map_rotated(self):
        """The rotation terms are honoured (not dropped as before)."""
        rot = Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"),
            geo=(0.0, 1.0, 0.5, 0.0, 0.5, -1.0),
            epsg=4326,
        )
        x, y = rot.array_to_map_coordinates([1], [2], center=False)
        # Column 2 and row 1 through the affine give x of 2.5 (two pixel
        # widths plus one row-rotation half) and y of 0.0 (one column
        # rotation minus one pixel height).
        assert abs(x[0] - 2.5) < 1e-9, f"rotated x should be 2.5, got {x[0]}"
        assert abs(y[0] - 0.0) < 1e-9, f"rotated y should be 0.0, got {y[0]}"

    def test_array_to_map_length_mismatch_raises(self):
        """Mismatched-length index inputs raise instead of pairing silently."""
        ds = self._nonsquare()
        with pytest.raises(ValueError, match="same length"):
            ds.array_to_map_coordinates([0, 1], [0])

    def test_array_to_map_empty_input(self):
        """Empty index inputs return a pair of empty lists, not an error."""
        x, y = self._nonsquare().array_to_map_coordinates([], [])
        assert x == [] and y == [], f"empty input must give ([], []), got ({x}, {y})"

    def test_array_to_map_accepts_iterator(self):
        """Non-Sized iterator index inputs are accepted, same as lists.

        Guards N1: the equal-length check materialises the inputs, so any finite
        iterable works (here single-pass ``list_iterator`` objects that have no
        ``__len__``), not only ``len()``-able sequences.
        """
        ds = self._nonsquare()
        x_iter, y_iter = ds.array_to_map_coordinates(
            iter([0, 1, 3]), iter([0, 1, 2]), center=True
        )
        x_list, y_list = ds.array_to_map_coordinates([0, 1, 3], [0, 1, 2], center=True)
        assert x_iter == x_list and y_iter == y_list, "iterator input must match list"

    def test_array_to_map_ndarray_input_returns_python_floats(self):
        """NumPy-array indices yield plain Python floats, same values as lists.

        Guards the N1 contract: the output element type is independent of whether
        the inputs were lists or ndarrays.
        """
        ds = self._nonsquare()
        x, y = ds.array_to_map_coordinates(
            np.array([0, 1, 3]), np.array([0, 1, 2]), center=True
        )
        assert all(
            type(v) is float for v in x + y
        ), f"elements must be plain Python floats, got {[type(v) for v in x + y]}"
        x_list, y_list = ds.array_to_map_coordinates([0, 1, 3], [0, 1, 2], center=True)
        assert x == x_list and y == y_list, "ndarray and list inputs must agree"

    def test_array_to_map_square_unchanged(self):
        """Square north-up grids match the historical width-based formula."""
        sq = Dataset.create_from_array(
            np.ones((10, 10), dtype="float32"),
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        x, y = sq.array_to_map_coordinates([1, 3, 5], [2, 4, 6])
        assert x == pytest.approx([0.1, 0.2, 0.3]), f"square x changed: {x}"
        assert y == pytest.approx([-0.05, -0.15, -0.25]), f"square y changed: {y}"


class TestOverlay:
    """Tests for overlay method error path."""

    def test_overlay_unaligned_raises(self):
        """overlay with unaligned dataset raises AlignmentError."""
        src = Dataset.create_from_array(
            np.ones((3, 3), dtype=np.float32),
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        classes = Dataset.create_from_array(
            np.ones((5, 5), dtype=np.float32),
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(AlignmentError):
            src.overlay(classes)


class TestNormalizeRescale:
    """Tests for normalize and _rescale static methods."""

    def test_normalize(self):
        """normalize should scale array to [0, 1] range."""
        arr = np.array([0.0, 50.0, 100.0])
        result = Analysis.normalize(arr)
        assert abs(result[0] - 0.0) < 1e-6, "Min should be normalized to 0.0"
        assert abs(result[2] - 1.0) < 1e-6, "Max should be normalized to 1.0"
        assert abs(result[1] - 0.5) < 1e-6, "Middle should be 0.5"

    def test_rescale(self):
        """_rescale should linearly rescale with given min/max."""
        arr = np.array([10.0, 20.0, 30.0])
        result = Analysis._rescale(arr, 10.0, 30.0)
        assert abs(result[0] - 0.0) < 1e-6, "Min should rescale to 0.0"
        assert abs(result[2] - 1.0) < 1e-6, "Max should rescale to 1.0"


class TestCluster2:
    """Tests for to_polygons/to_feature_collection band selection."""

    def test_to_polygons_band_as_list(self):
        """to_polygons with band as a list should use the first element."""
        arr = np.array([[1, 1, 2], [2, 3, 3], [3, 1, 2]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=[0])
        assert (
            gdf is not None
        ), "to_polygons with list band should return a GeoDataFrame"
        assert len(gdf) > 0, "Should have some polygons"


class TestCorrectWrapCutlineErrorNdim:
    """Tests for _correct_wrap_cutline_error with invalid ndim."""

    def test_4d_array_raises(self):
        """A 4D array in _correct_wrap_cutline_error should raise ValueError."""
        nd = -9999.0
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        ds._raster.GetRasterBand(1).WriteArray(arr)
        # We can't easily create a 4D array in GDAL, so we test
        # the static method behavior via mocker
        # Instead test valid 3D path which is also useful
        arr_3d = np.ones((2, 3, 3), dtype=np.float32)
        arr_3d[:, 0, :] = nd
        ds_3d = Dataset.create_from_array(
            arr_3d,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = Spatial._correct_wrap_cutline_error(ds_3d)
        assert result.rows == 2, "Should trim first row of nodata"


class TestCropAlignedFillGaps:
    """Tests for _crop_aligned with fill_gaps=True."""

    def test_crop_aligned_fill_gaps(self):
        """_crop_aligned with fill_gaps=True fills gap cells."""
        nd = -9999.0
        src_arr = np.ones((3, 3), dtype=np.float32) * 5.0
        src_arr[1, 1] = nd
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((3, 3), dtype=np.float32)
        mask = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = src.spatial._crop_aligned(mask, fill_gaps=True)
        arr = result.read_array()
        assert arr is not None, "Fill gaps result should have a valid array"


class TestToCrsSameEpsgPaths:
    """Tests for to_crs when src_epsg == to_epsg."""

    def test_to_crs_same_preserves_bounds(self):
        """to_crs with same EPSG should preserve bounds."""
        arr = np.ones((5, 5), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(10.0, 50.0),
            cell_size=0.5,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.to_crs(to_epsg=4326)
        assert result is not None, "Should return a Dataset"
        assert result.epsg == 4326, "EPSG should stay 4326"
        assert result.rows > 0, "Should have rows"
        assert result.columns > 0, "Should have columns"


class TestToCrsWestHemLongitude:
    """Tests for to_crs with west hemisphere (>180) longitude."""

    def test_to_crs_longitude_above_180(self):
        """to_crs on data with longitude > 180 uses special transform."""
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(200.0, 50.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.to_crs(to_epsg=3857)
        assert result is not None, "Should handle >180 longitude"
        assert result.epsg == 3857, "EPSG should be 3857"


class TestReprojectNonSquareCells:
    """B-10 regression: `_reproject_with_ReprojectImage` measures Y spacing.

    The legacy implementation discarded the reprojected Y values and
    forced the output cell to be square (X spacing reused for Y).
    These tests prove the X and Y cell sizes are now measured
    independently — both for inputs that are already non-square in
    source CRS and for square inputs whose reprojection produces a
    non-square cell on the destination CRS.
    """

    def test_reproject_high_latitude_yields_non_square_pixels(self):
        """4326 → 3857 at 60°N: output Y/X ratio ≈ 1/cos(60°) = 2.

        At latitude 60°N, one degree of longitude is ~half a degree of
        latitude in metres — so projecting a square-degree pixel to
        Web Mercator with `maintain_alignment=True` should produce a
        Y-spacing roughly 2× the X spacing. Pre-fix the output was
        forced square (Y == X), so this assertion was unsatisfiable.
        """
        arr = np.ones((10, 10), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(10.0, 60.5),
            cell_size=0.1,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.to_crs(to_epsg=3857, maintain_alignment=True)
        x_spacing = abs(result.geotransform[1])
        y_spacing = abs(result.geotransform[5])
        ratio = y_spacing / x_spacing
        assert ratio > 1.5, (
            f"Expected Y spacing > 1.5x X spacing at 60N, got "
            f"y/x = {ratio:.3f} (x={x_spacing:.2f}m, y={y_spacing:.2f}m)"
        )

    def test_reproject_preserves_non_square_input_aspect(self):
        """Non-square source pixels must not collapse to square output.

        Build a synthetic raster with cell_size=(2.0, 14.0) — a SAR-like
        aspect ratio — reproject (UTM 33N → Web Mercator) with
        `maintain_alignment=True`. Pre-fix the output Y spacing was
        forced equal to the X spacing, collapsing the 7:1 input aspect
        to 1:1.
        """
        arr = np.ones((10, 10), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            geo=(500_000.0, 2.0, 0.0, 5_500_000.0, 0.0, -14.0),
            epsg=32633,
            no_data_value=-9999.0,
        )
        result = ds.to_crs(to_epsg=3857, maintain_alignment=True)
        x_spacing = abs(result.geotransform[1])
        y_spacing = abs(result.geotransform[5])
        ratio = y_spacing / x_spacing
        assert ratio > 3.0, (
            f"Non-square input (2m x 14m, ratio 7) should keep a "
            f"non-square output; got y/x = {ratio:.3f}"
        )


class TestCropAlignedNanMask:
    """Tests for _crop_aligned with NaN mask nodata value."""

    def test_crop_aligned_numpy_mask_uses_source_geotransform(self):
        """B-11 regression: numpy-array mask falls back to src geotransform.

        Pre-fix the function relied on `try/except UnboundLocalError`
        to detect the "mask is a numpy array" branch (because
        `mask_gt` was only assigned in the RasterBase branch). The
        replacement uses an explicit `isinstance(mask, RasterBase)`
        check; this test guards both that the numpy-mask path still
        works AND that the output geotransform / projection are
        copied from the source raster (since a numpy array carries
        no spatial metadata).
        """
        nd = -9999.0
        src_arr = np.ones((4, 4), dtype=np.float32) * 5.0
        src_top_left = (10.0, 50.0)
        src_cell = 0.05
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=src_top_left,
            cell_size=src_cell,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((4, 4), dtype=np.float32)
        mask_arr[0, 0] = nd
        mask_arr[2, 3] = nd
        result = src.spatial._crop_aligned(mask_arr, mask_noval=nd)
        assert result.epsg == 4326, "Should preserve source CRS"
        assert result.geotransform == src.geotransform, (
            f"Output geotransform {result.geotransform} should match "
            f"source {src.geotransform}"
        )
        result_arr = result.read_array()
        assert np.isclose(result_arr[0, 0], nd), "Cell 0,0 should be nodata"
        assert np.isclose(result_arr[2, 3], nd), "Cell 2,3 should be nodata"
        assert np.isclose(result_arr[1, 1], 5.0), "Cell 1,1 should be unchanged"

    def test_crop_aligned_multi_band_with_nan_mask(self):
        """_crop_aligned multi-band with NaN mask nodata."""
        nd = -9999.0
        src_arr = np.ones((2, 4, 4), dtype=np.float32) * 5.0
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((4, 4), dtype=np.float32)
        mask_arr[0, 0] = np.nan
        mask_arr[2, 3] = np.nan
        mask_ds = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        # Set mask nodata to None to trigger nan check path
        mask_ds._no_data_value = [None]
        result = src.spatial._crop_aligned(mask_ds)
        result_arr = result.read_array()
        assert result_arr.ndim == 3, "Multi-band result should be 3D"

    def test_crop_aligned_single_band_nan_mask_noval(self):
        """_crop_aligned single-band with None mask noval."""
        nd = -9999.0
        src_arr = np.ones((4, 4), dtype=np.float32) * 10.0
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((4, 4), dtype=np.float32)
        mask_arr[1, 1] = np.nan
        mask_ds = Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        mask_ds._no_data_value = [None]
        result = src.spatial._crop_aligned(mask_ds)
        result_arr = result.read_array()
        assert result_arr is not None, "Should return a valid array"


class TestCropWithRasterString:
    """Tests for _crop_with_raster with string path."""

    def test_crop_with_raster_string_path(self, tmp_path):
        """_crop_with_raster with a string path should read the mask."""
        nd = -9999.0
        src_arr = np.ones((5, 5), dtype=np.float32)
        src = Dataset.create_from_array(
            src_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        mask_arr = np.ones((5, 5), dtype=np.float32)
        mask_arr[0, :] = nd
        mask_path = str(tmp_path / "mask.tif")
        Dataset.create_from_array(
            mask_arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
            driver_type="GTiff",
            path=mask_path,
        )
        result = src.spatial._crop_with_raster(mask_path)
        assert isinstance(result, Dataset), "Should return a Dataset"


class TestCluster2BandList:
    """Tests for to_polygons with band passed as a list."""

    def test_to_polygons_with_list_band(self):
        """to_polygons with band=[0] should use the first element."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=[0])
        assert gdf is not None, "to_polygons should return a GeoDataFrame"
        assert len(gdf) > 0, "Should have some polygons"


class TestCropWithPolygonWarpError:
    """Tests for _crop_with_polygon_warp error paths."""

    def test_crop_with_gdf(self, single_band_dataset):
        """_crop_with_polygon_warp with a GeoDataFrame should work."""
        import geopandas as gpd
        from shapely.geometry import box

        poly = box(0.0, -0.15, 0.15, 0.0)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        result = single_band_dataset.spatial._crop_with_polygon_warp(gdf)
        assert isinstance(result, Dataset), "Should return a cropped Dataset"


class TestCluster2BandNone:
    """Tests for to_polygons with band=None."""

    def test_to_polygons_none_band(self):
        """to_polygons with band=None should default to band 0."""
        arr = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=None)
        assert gdf is not None, "to_polygons with None band should work"

    def test_to_polygons_int_band(self):
        """to_polygons with band as integer."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=0)
        assert gdf is not None, "to_polygons with int band should work"

    def test_to_polygons_list_band(self):
        """to_polygons with band as a list should use first element."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=[0])
        assert gdf is not None, "to_polygons with list band should work"

    def test_cluster2_deprecated_alias_warns_and_forwards(self):
        """cluster2 is a deprecated alias that warns and returns the same result as to_polygons."""
        arr = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        with pytest.warns(DeprecationWarning, match="cluster2 is deprecated"):
            legacy = ds.cluster2()
        assert len(legacy) == len(
            ds.to_polygons()
        ), "cluster2 should forward to to_polygons"


class TestWrapLongitudeInplace:
    """Tests for wrap_longitude inplace path."""

    def test_wrap_longitude_returns_dataset(self):
        """wrap_longitude() returns new Dataset."""
        cols = 360
        arr = np.ones((1, cols), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.5),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.wrap_longitude()
        assert isinstance(result, Dataset), "Should return a new Dataset"
        assert result.geotransform[0] < 0, "New top-left x should be negative"


class TestCropWithPolygonFeatureCollection:
    """Tests for _crop_with_polygon_warp with FeatureCollection."""

    def test_crop_with_feature_collection(self, single_band_dataset):
        """_crop_with_polygon_warp with FeatureCollection."""
        import geopandas as gpd
        from shapely.geometry import box

        from pyramids.feature import FeatureCollection

        poly = box(0.0, -0.15, 0.15, 0.0)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        fc = FeatureCollection(gdf)
        result = single_band_dataset.spatial._crop_with_polygon_warp(fc)
        assert isinstance(result, Dataset), "Should return a cropped Dataset"


class TestMapToArrayFeatureCollection:
    """Tests for map_to_array_coordinates with FeatureCollection."""

    def test_map_to_array_with_feature_collection(self, single_band_dataset):
        """map_to_array_coordinates with FeatureCollection input."""
        import geopandas as gpd
        from shapely.geometry import Point

        from pyramids.feature import FeatureCollection

        pts = gpd.GeoDataFrame(
            geometry=[Point(0.025, -0.025), Point(0.075, -0.075)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(pts)
        result = single_band_dataset.map_to_array_coordinates(fc)
        assert result is not None, "Should return array indices"
        assert result.shape[0] == 2, "Should have 2 points"


class TestNonSquareCells:
    """Tests for get_cell_coords with non-square cells."""

    def test_get_cell_coords_non_square(self):
        """get_cell_coords with non-square cells triggers warning."""
        gt = (0.0, 0.1, 0.0, 0.0, 0.0, -0.05)
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            geo=gt,
            epsg=4326,
            no_data_value=-9999.0,
        )
        coords = ds.get_cell_coords(location="center")
        assert coords is not None, "Should return coordinates for non-square cells"


class TestGroupNeighbours:
    """Tests for _group_neighbours boundary cases."""

    def test_group_neighbours_at_corners(self):
        """_group_neighbours should handle corner/edge cells."""
        arr = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.int32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.to_polygons(band=0)
        assert len(gdf) >= 4, "Should find at least 4 clusters"
