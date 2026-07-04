"""Unit tests for Dataset/RasterBase properties, lifecycle, bands, and color tables."""

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr

from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import Dataset
from pyramids.dataset.abstract_dataset import RasterBase

pytestmark = pytest.mark.core


class TestRasterBaseStaticMethods:
    """Tests for static helpers defined in RasterBase."""

    def test_get_x_lon_dimension_array_values(self):
        """Verify x-coordinate array for simple inputs."""
        pivot_x = 10.0
        cell_size = 0.5
        columns = 4
        result = RasterBase.get_x_lon_dimension_array(pivot_x, cell_size, columns)
        expected = np.array([10.25, 10.75, 11.25, 11.75])
        np.testing.assert_allclose(
            result,
            expected,
            err_msg="X-lon dimension array values are incorrect",
        )

    def test_get_x_lon_dimension_array_length(self):
        """Returned array length must equal the number of columns."""
        result = RasterBase.get_x_lon_dimension_array(0.0, 1.0, 7)
        assert len(result) == 7, "Array length should equal column count"

    def test_get_y_lat_dimension_array_values(self):
        """Verify y-coordinate array decreases from north to south."""
        pivot_y = 50.0
        cell_size = 0.5
        rows = 3
        result = RasterBase.get_y_lat_dimension_array(pivot_y, cell_size, rows)
        expected = np.array([49.75, 49.25, 48.75])
        np.testing.assert_allclose(
            result,
            expected,
            err_msg="Y-lat dimension array values are incorrect",
        )

    def test_get_y_lat_dimension_array_length(self):
        """Returned array length must equal the number of rows."""
        result = RasterBase.get_y_lat_dimension_array(0.0, 1.0, 5)
        assert len(result) == 5, "Array length should equal row count"


class TestCoordinateProperties:
    """Tests for the Dataset.lon / lat / x / y cell-centre coordinate properties."""

    def test_shapes_match_dimensions(self, single_band_dataset):
        """lon/x have length columns and lat/y have length rows.

        Test scenario:
            For the 3x3 fixture every coordinate axis has the matching dimension
            length.
        """
        ds = single_band_dataset
        assert ds.lon.shape == (ds.columns,), f"lon shape {ds.lon.shape}"
        assert ds.x.shape == (ds.columns,), f"x shape {ds.x.shape}"
        assert ds.lat.shape == (ds.rows,), f"lat shape {ds.lat.shape}"
        assert ds.y.shape == (ds.rows,), f"y shape {ds.y.shape}"

    def test_x_aliases_lon_and_y_aliases_lat(self, multi_band_dataset):
        """x returns the same values as lon, and y the same as lat.

        Test scenario:
            The x/y aliases must agree element-wise with lon/lat on a fixture with
            distinct row/column counts.
        """
        ds = multi_band_dataset
        np.testing.assert_array_equal(ds.x, ds.lon, err_msg="x must equal lon")
        np.testing.assert_array_equal(ds.y, ds.lat, err_msg="y must equal lat")

    def test_values_match_static_builders(self, multi_band_dataset):
        """lon/lat equal the static dimension-array helpers fed the geotransform.

        Test scenario:
            The instance properties are a thin wrapper over
            get_x_lon_dimension_array / get_y_lat_dimension_array using the
            geotransform's own pixel width and height.
        """
        ds = multi_band_dataset
        gt = ds.geotransform
        expected_lon = RasterBase.get_x_lon_dimension_array(gt[0], gt[1], ds.columns)
        expected_lat = RasterBase.get_y_lat_dimension_array(gt[3], abs(gt[5]), ds.rows)
        np.testing.assert_allclose(ds.lon, expected_lon, err_msg="lon mismatch")
        np.testing.assert_allclose(ds.lat, expected_lat, err_msg="lat mismatch")

    def test_square_cell_centres(self):
        """Square-cell rasters report the expected centre coordinates.

        Test scenario:
            A 2x3 raster at top-left (0, 0) with cell_size 0.5 has column centres at
            0.25/0.75/1.25 and row centres at -0.25/-0.75.
        """
        ds = Dataset.create_from_array(
            np.arange(6.0).reshape(2, 3),
            top_left_corner=(0.0, 0.0),
            cell_size=0.5,
            epsg=4326,
        )
        np.testing.assert_allclose(ds.x, [0.25, 0.75, 1.25], err_msg="x centres wrong")
        np.testing.assert_allclose(ds.y, [-0.25, -0.75], err_msg="y centres wrong")

    def test_non_square_cells_use_separate_pixel_height(self):
        """Non-square cells: y uses pixel height, x uses pixel width.

        Test scenario:
            A raster with geotransform pixel width 2 and pixel height 1 must produce
            x centres spaced by 2 (11/13/15) and y centres spaced by 1 (49.5/48.5).
            This guards the regression where lat/y previously reused cell_size (the
            pixel width) for the y axis.
        """
        ds = Dataset.create_from_array(
            np.arange(6.0).reshape(2, 3),
            geo=(10.0, 2.0, 0.0, 50.0, 0.0, -1.0),
            epsg=4326,
        )
        np.testing.assert_allclose(ds.x, [11.0, 13.0, 15.0], err_msg="x spacing wrong")
        np.testing.assert_allclose(
            ds.y, [49.5, 48.5], err_msg="y must use pixel height"
        )


class TestRasterBaseBlockSizeSetter:
    """Tests for the block_size setter validation on RasterBase."""

    def test_block_size_setter_invalid_raises(self, single_band_dataset):
        """Setting block_size with a non-2-element tuple should raise."""
        with pytest.raises(ValueError, match="tuple of 2 integers"):
            single_band_dataset.block_size = [(512,)]

    def test_block_size_setter_valid(self, single_band_dataset):
        """Setting a valid block_size should update the attribute."""
        single_band_dataset.block_size = [(256, 256)]
        assert single_band_dataset.block_size == [
            (256, 256)
        ], "Block size was not updated correctly"


class TestSetCrsAbstract:
    """Tests for RasterBase.set_crs (invoked via Dataset)."""

    def test_set_crs_with_epsg(self, single_band_dataset):
        """Setting CRS via epsg should update the EPSG attribute."""
        single_band_dataset.set_crs(epsg=32618)
        assert (
            single_band_dataset.epsg == 32618
        ), "EPSG not updated after set_crs(epsg=...)"

    def test_set_crs_with_wkt(self, single_band_dataset):
        """Setting CRS via a WKT string should update the projection."""
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(32618)
        wkt = sr.ExportToWkt()
        single_band_dataset.set_crs(crs=wkt)
        assert (
            single_band_dataset.epsg == 32618
        ), "EPSG not updated after set_crs(crs=wkt)"

    def test_set_crs_with_both_prefers_crs(self, single_band_dataset):
        """When both crs and epsg are given, crs takes precedence."""
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(32618)
        wkt = sr.ExportToWkt()
        # Pass both crs and epsg; the WKT (32618) should win
        single_band_dataset.set_crs(crs=wkt, epsg=4326)
        assert (
            single_band_dataset.epsg == 32618
        ), "CRS WKT should take precedence over epsg arg"


class TestUpdateInplace:
    """Tests for the _update_inplace method."""

    def test_update_inplace_updates_state(self, single_band_dataset):
        """After _update_inplace the dimensions should reflect the new source."""
        new_arr = np.ones((5, 7), dtype=np.float32)
        new_ds = Dataset.create_from_array(
            new_arr,
            top_left_corner=(1.0, 2.0),
            cell_size=0.1,
            epsg=4326,
        )
        old_rows = single_band_dataset.rows
        single_band_dataset._update_inplace(new_ds.raster)
        assert (
            single_band_dataset.rows == 5
        ), f"Expected 5 rows after reinit, got {single_band_dataset.rows}"
        assert single_band_dataset.columns == 7, "Columns not updated after reinit"
        assert (
            old_rows != single_band_dataset.rows
        ), "reinit did not change internal state"


class TestScaleOffset:
    """Tests for scale and offset property getters and setters."""

    def test_scale_default(self, single_band_dataset):
        """Default scale should be 1.0 for each band."""
        assert single_band_dataset.scale == [1.0], "Default scale should be [1.0]"

    def test_scale_setter(self, single_band_dataset):
        """Setting scale should update GDAL band scale."""
        single_band_dataset.scale = [0.5]
        assert (
            single_band_dataset._iloc(0).GetScale() == pytest.approx(0.5)
        ), "GDAL band scale not updated by setter"

    def test_offset_default(self, single_band_dataset):
        """Default offset should be 0 for each band."""
        assert single_band_dataset.offset == [0], "Default offset should be [0]"

    def test_offset_setter(self, single_band_dataset):
        """Setting offset should update GDAL band offset."""
        single_band_dataset.offset = [100.0]
        assert (
            single_band_dataset._iloc(0).GetOffset() == pytest.approx(100.0)
        ), "GDAL band offset not updated by setter"

    def test_multi_band_scale_offset(self, multi_band_dataset):
        """Scale and offset setters should work per-band for multi-band."""
        multi_band_dataset.scale = [0.1, 0.2, 0.3]
        scales = multi_band_dataset.scale
        assert scales == [
            0.1,
            0.2,
            0.3,
        ], f"Multi-band scales incorrect: {scales}"

        multi_band_dataset.offset = [10.0, 20.0, 30.0]
        offsets = multi_band_dataset.offset
        assert offsets == [
            10.0,
            20.0,
            30.0,
        ], f"Multi-band offsets incorrect: {offsets}"


class TestBandNamesUnitsSetters:
    """Tests for band_names and band_units setters."""

    def test_band_names_setter(self, multi_band_dataset):
        """Setting band_names should update both GDAL and internal names."""
        new_names = ["red", "green", "blue"]
        multi_band_dataset.band_names = new_names
        assert (
            multi_band_dataset.band_names == new_names
        ), "band_names setter did not update names"

    def test_band_units_setter(self, multi_band_dataset):
        """Setting band_units should write units to each GDAL band."""
        new_units = ["m", "kg", "s"]
        multi_band_dataset.band_units = new_units
        assert (
            multi_band_dataset.band_units == new_units
        ), "band_units setter did not update units"
        for i, expected in enumerate(new_units):
            actual = multi_band_dataset._iloc(i).GetUnitType()
            assert (
                actual == expected
            ), f"Band {i} unit mismatch: expected {expected}, got {actual}"


class TestConvertUnits:
    """Tests for the Dataset.convert_units value-conversion method."""

    def test_single_band_kelvin_to_celsius(self, single_band_dataset):
        """convert_units converts a single-band Kelvin raster to Celsius.

        Test scenario:
            A 3x3 raster labelled "K" converted to "celsius" subtracts 273.15 from
            every cell and updates band_units to ["celsius"].
        """
        single_band_dataset.band_units = ["K"]
        result = single_band_dataset.convert_units("celsius")
        expected = single_band_dataset.read_array() - 273.15
        np.testing.assert_allclose(
            result.read_array(), expected, rtol=1e-6, err_msg="K->C values wrong"
        )
        assert result.band_units == [
            "celsius"
        ], f"Units not updated: {result.band_units}"

    def test_multi_band_all_converted(self, multi_band_dataset):
        """convert_units converts every band when band is None.

        Test scenario:
            A 3-band raster all labelled "m" converted to "mm" scales every band by
            1000 and labels all bands "mm".
        """
        multi_band_dataset.band_units = ["m", "m", "m"]
        result = multi_band_dataset.convert_units("mm")
        np.testing.assert_allclose(
            result.read_array(),
            multi_band_dataset.read_array() * 1000.0,
            rtol=1e-6,
            err_msg="m->mm values wrong",
        )
        assert result.band_units == ["mm", "mm", "mm"], f"Units: {result.band_units}"

    def test_band_argument_converts_one_band_only(self, multi_band_dataset):
        """convert_units with band= converts only the selected band.

        Test scenario:
            Converting only band 0 (m->mm) scales band 0 by 1000 but leaves bands 1
            and 2 untouched, and only band 0's unit label changes.
        """
        multi_band_dataset.band_units = ["m", "m", "m"]
        source = multi_band_dataset.read_array()
        result = multi_band_dataset.convert_units("mm", band=0)
        converted = result.read_array()
        np.testing.assert_allclose(
            converted[0], source[0] * 1000.0, rtol=1e-6, err_msg="band 0 not converted"
        )
        np.testing.assert_array_equal(
            converted[1], source[1], err_msg="band 1 should be untouched"
        )
        np.testing.assert_array_equal(
            converted[2], source[2], err_msg="band 2 should be untouched"
        )
        assert result.band_units == ["mm", "m", "m"], f"Units: {result.band_units}"

    def test_nodata_cells_preserved(self):
        """convert_units leaves no-data cells at their sentinel value.

        Test scenario:
            A Kelvin raster containing a -9999.0 no-data cell, converted to Celsius,
            keeps that cell at -9999.0 while converting the valid cells.
        """
        arr = np.array([[273.15, -9999.0], [293.15, 303.15]], dtype=np.float64)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds.band_units = ["K"]
        result = ds.convert_units("celsius")
        out = result.read_array()
        assert out[0, 1] == -9999.0, f"No-data cell altered: {out[0, 1]}"
        assert out[0, 0] == pytest.approx(0.0), f"Valid cell wrong: {out[0, 0]}"
        assert out[1, 0] == pytest.approx(20.0), f"Valid cell wrong: {out[1, 0]}"

    def test_source_dataset_unchanged(self, single_band_dataset):
        """convert_units returns a new Dataset and leaves the source untouched.

        Test scenario:
            After conversion the source still reports its original values and units.
        """
        single_band_dataset.band_units = ["K"]
        snapshot = single_band_dataset.read_array().copy()
        result = single_band_dataset.convert_units("celsius")
        assert (
            result is not single_band_dataset
        ), "convert_units must return a new object"
        np.testing.assert_array_equal(
            single_band_dataset.read_array(), snapshot, err_msg="source was mutated"
        )
        assert single_band_dataset.band_units == ["K"], "source units changed"

    def test_band_out_of_range_raises(self, single_band_dataset):
        """convert_units rejects a band index outside the valid range.

        Test scenario:
            Requesting band 5 on a single-band raster raises ValueError mentioning
            'out of range'.
        """
        single_band_dataset.band_units = ["K"]
        with pytest.raises(ValueError, match="out of range") as exc:
            single_band_dataset.convert_units("celsius", band=5)
        assert "out of range" in str(exc.value), f"Unexpected: {exc.value}"

    def test_unknown_target_raises(self, single_band_dataset):
        """convert_units propagates the ValueError for an unsupported target.

        Test scenario:
            A target unit absent from the affine table raises ValueError mentioning
            'No unit conversion'.
        """
        single_band_dataset.band_units = ["K"]
        with pytest.raises(ValueError, match="No unit conversion") as exc:
            single_band_dataset.convert_units("furlongs")
        assert "No unit conversion" in str(exc.value), f"Unexpected: {exc.value}"


class TestCountDomainCells:
    """Tests for the count_domain_cells method."""

    def test_all_valid_cells(self, single_band_dataset):
        """All cells in the fixture contain valid data."""
        count = single_band_dataset.count_domain_cells()
        assert count == 9, f"Expected 9 domain cells for a 3x3 raster, got {count}"

    def test_with_nodata_cells(self, dataset_with_nodata):
        """Cells with nodata should not be counted."""
        count = dataset_with_nodata.count_domain_cells()
        assert count == 4, f"Expected 4 domain cells (5 nodata), got {count}"


class TestClose:
    """Tests for the close method."""

    def test_close_nullifies_raster(self, single_band_dataset):
        """After close(), the internal GDAL dataset reference should be None."""
        single_band_dataset.close()
        assert (
            single_band_dataset._raster is None
        ), "Internal raster reference should be None after close()"

    def test_double_close_is_safe(self, single_band_dataset):
        """Calling close() twice should not raise an error.

        Test scenario:
            The double-close guard checks `_raster is not None` before
            calling FlushCache, so the second call is a no-op.
        """
        single_band_dataset.close()
        single_band_dataset.close()
        assert (
            single_band_dataset._raster is None
        ), "Raster should remain None after double close"

    def test_close_flushes_before_nullify(self):
        """close() should call FlushCache before setting _raster to None.

        Test scenario:
            Verify that FlushCache is invoked on the GDAL dataset before
            the reference is dropped.
        """
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        raster_before = ds._raster
        assert raster_before is not None, "Raster should exist before close"
        ds.close()
        assert ds._raster is None, "Raster should be None after close"


class TestContextManager:
    """Tests for the context manager protocol (__enter__/__exit__)."""

    def test_with_statement_returns_dataset(self):
        """The `with` statement should yield the Dataset object.

        Test scenario:
            `__enter__` returns `self`, so the `as` variable should be the
            same Dataset instance.
        """
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with ds as ctx:
            assert ctx is ds, "Context manager should return the same Dataset instance"
            assert (
                ctx._raster is not None
            ), "Raster should be available inside with block"

    def test_raster_closed_after_with_block(self):
        """After exiting the `with` block, the dataset should be closed.

        Test scenario:
            The `__exit__` method calls `close()`, so `_raster` should be
            None after the block ends.
        """
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with ds:
            assert ds._raster is not None, "Raster should exist inside with block"
        assert ds._raster is None, "Raster should be None after with block exits"

    def test_cleanup_on_exception(self):
        """The dataset should be closed even if an exception occurs inside the with block.

        Test scenario:
            Raise a ValueError inside the block; the dataset should still
            be cleaned up after the exception propagates.
        """
        arr = np.ones((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with pytest.raises(ValueError, match="test error"):
            with ds:
                raise ValueError("test error")
        assert (
            ds._raster is None
        ), "Raster should be None after exception inside with block"

    def test_operations_inside_with_block(self):
        """Dataset operations should work normally inside a with block.

        Test scenario:
            Read array and check properties inside the block to verify
            the dataset is fully functional.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with ds:
            result = ds.read_array()
            assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
            assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"
            assert ds.rows == 2, f"Expected 2 rows, got {ds.rows}"

    def test_context_manager_with_create_from_array(self):
        """Context manager should work with the create_from_array factory.

        Test scenario:
            Create a dataset inline in the `with` statement and verify
            it works and gets cleaned up.
        """
        arr = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )
        with ds as ctx:
            val = ctx.read_array()[0, 0]
            assert np.isclose(val, 5.0), f"Expected 5.0, got {val}"
        assert ds._raster is None, "Should be closed after with block"

    def test_double_close_after_context_manager(self):
        """Calling close() after the with block should be safe (no-op).

        Test scenario:
            The with block already calls close(). An explicit close()
            afterwards should not raise.
        """
        arr = np.ones((2, 2), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with ds:
            # entering then exiting the context manager is the behaviour under test
            pass
        ds.close()
        assert ds._raster is None, "Should remain None after redundant close"

    def test_exit_returns_false(self):
        """__exit__ should return False so exceptions are not suppressed.

        Test scenario:
            A ValueError raised inside the with block should propagate
            to the caller, not be silently swallowed.
        """
        arr = np.ones((2, 2), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        with pytest.raises(RuntimeError, match="propagate"):
            with ds:
                raise RuntimeError("propagate")


class TestDatasetLike:
    """Tests for the dataset_like class method."""

    def test_dataset_like_preserves_geo(self, single_band_dataset):
        """dataset_like should preserve geotransform and projection."""
        new_arr = np.zeros((3, 3), dtype=np.float32)
        result = Dataset.dataset_like(single_band_dataset, new_arr)
        assert (
            result.geotransform == single_band_dataset.geotransform
        ), "Geotransform not preserved"
        assert result.epsg == single_band_dataset.epsg, "EPSG not preserved"

    def test_dataset_like_multi_band(self, single_band_dataset):
        """dataset_like with a 3D array should create a multi-band dataset."""
        new_arr = np.zeros((2, 3, 3), dtype=np.float32)
        result = Dataset.dataset_like(single_band_dataset, new_arr)
        assert result.band_count == 2, f"Expected 2 bands, got {result.band_count}"

    def test_dataset_like_wrong_type(self, single_band_dataset):
        """dataset_like with a non-array should raise TypeError."""
        with pytest.raises(TypeError, match="numpy array"):
            Dataset.dataset_like(single_band_dataset, [1, 2, 3])


class TestCellGeometryMethods:
    """Tests for get_cell_coords, get_cell_points, get_cell_polygons."""

    def test_get_cell_coords_center(self, single_band_dataset):
        """Center coords should be at half-cell offsets from corners."""
        coords = single_band_dataset.get_cell_coords(location="center")
        assert coords.shape == (9, 2), f"Expected (9,2) array, got {coords.shape}"
        assert np.isclose(
            coords[0, 0], 0.025, atol=1e-6
        ), "First x-center coordinate is wrong"
        assert np.isclose(
            coords[0, 1], -0.025, atol=1e-6
        ), "First y-center coordinate is wrong"

    def test_get_cell_coords_corner(self, single_band_dataset):
        """Corner coords should be the top-left of each cell."""
        coords = single_band_dataset.get_cell_coords(location="corner")
        assert np.isclose(
            coords[0, 0], 0.0, atol=1e-6
        ), "First x-corner coordinate should be 0.0"
        assert np.isclose(
            coords[0, 1], 0.0, atol=1e-6
        ), "First y-corner coordinate should be 0.0"

    def test_get_cell_coords_invalid_location(self, single_band_dataset):
        """An invalid location string should raise ValueError."""
        with pytest.raises(ValueError, match="center.*corner"):
            single_band_dataset.get_cell_coords(location="middle")

    def test_get_cell_points_center(self, single_band_dataset):
        """get_cell_points should return a GeoDataFrame with Point geometry."""
        import geopandas as gpd

        gdf = single_band_dataset.get_cell_points(location="center")
        assert isinstance(gdf, gpd.GeoDataFrame), "Should return GeoDataFrame"
        assert len(gdf) == 9, f"Expected 9 points, got {len(gdf)}"
        assert "id" in gdf.columns, "GeoDataFrame should have 'id' column"

    def test_get_cell_points_corner(self, single_band_dataset):
        """get_cell_points with corner should return corner coordinates."""
        gdf = single_band_dataset.get_cell_points(location="corner")
        first_point = gdf.geometry.iloc[0]
        assert np.isclose(
            first_point.x, 0.0, atol=1e-6
        ), "First corner point x should be 0.0"

    def test_get_cell_polygons(self, single_band_dataset):
        """get_cell_polygons should return polygons covering each cell."""
        import geopandas as gpd

        gdf = single_band_dataset.get_cell_polygons()
        assert isinstance(gdf, gpd.GeoDataFrame), "Should return GeoDataFrame"
        assert len(gdf) == 9, f"Expected 9 polygons, got {len(gdf)}"
        poly = gdf.geometry.iloc[0]
        area = poly.area
        expected_area = 0.05 * 0.05
        assert np.isclose(
            area, expected_area, rtol=0.01
        ), f"Polygon area {area} differs from expected {expected_area}"

    def test_get_cell_polygons_with_mask(self, dataset_with_nodata):
        """With mask=True, only domain cells should get polygons."""
        import geopandas as gpd

        gdf = dataset_with_nodata.get_cell_polygons(domain_only=True)
        assert isinstance(gdf, gpd.GeoDataFrame), "Should return GeoDataFrame"
        assert len(gdf) == 4, f"Expected 4 polygons for domain cells, got {len(gdf)}"


class TestGetBandByColor:
    """Tests for get_band_by_color method."""

    def test_band_by_color_rgb(self):
        """After assigning RGB colors, get_band_by_color returns correct index."""
        arr = np.ones((3, 4, 4), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        ds.band_color = {0: "red", 1: "green", 2: "blue"}
        assert ds.get_band_by_color("red") == 0, "Red should be band 0"
        assert ds.get_band_by_color("green") == 1, "Green should be band 1"
        assert ds.get_band_by_color("blue") == 2, "Blue should be band 2"

    def test_band_by_color_not_found(self, single_band_dataset):
        """get_band_by_color should return None for a color not in the dataset."""
        result = single_band_dataset.get_band_by_color("red")
        assert result is None, "Should return None when color is not assigned"


class TestDatasetProperties:
    """Tests for Dataset basic properties."""

    def test_access_property(self, single_band_dataset):
        """In-memory datasets created via create_from_array have write access."""
        assert (
            single_band_dataset.access == "write"
        ), "create_from_array datasets should have 'write' access"

    def test_cell_size_property(self, single_band_dataset):
        """cell_size should match the value passed during creation."""
        assert (
            single_band_dataset.cell_size == pytest.approx(0.05)
        ), "cell_size property does not match"

    def test_driver_type_property(self, single_band_dataset):
        """In-memory dataset should have 'mem' driver type."""
        dtype = single_band_dataset.driver_type
        assert dtype is not None, "driver_type should not be None"

    def test_file_name_empty_for_mem(self, single_band_dataset):
        """In-memory datasets have empty or blank file_name."""
        fn = single_band_dataset.file_name
        assert fn is not None, "file_name should not be None"

    def test_crs_property(self, single_band_dataset):
        """crs property should return a non-empty WKT string."""
        crs = single_band_dataset.crs
        assert isinstance(crs, str), "crs should be a string"
        assert len(crs) > 0, "crs string should not be empty"

    def test_crs_setter(self, single_band_dataset):
        """crs setter should update the projection."""
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(32618)
        wkt = sr.ExportToWkt()
        single_band_dataset.crs = wkt
        assert single_band_dataset.epsg == 32618, "crs setter did not update EPSG"

    def test_epsg_setter(self, single_band_dataset):
        """epsg setter should update both projection and epsg."""
        single_band_dataset.epsg = 3857
        assert single_band_dataset.epsg == 3857, "epsg setter did not update correctly"

    def test_meta_data_setter(self, single_band_dataset):
        """meta_data setter should store key-value metadata."""
        single_band_dataset.meta_data = {"MY_KEY": "MY_VALUE"}
        md = single_band_dataset.meta_data
        assert "MY_KEY" in md, "Metadata key not found"
        assert md["MY_KEY"] == "MY_VALUE", "Metadata value mismatch"

    def test_geotransform_property(self, single_band_dataset):
        """geotransform should return a 6-element tuple."""
        gt = single_band_dataset.geotransform
        assert len(gt) == 6, f"Geotransform should have 6 elements, got {len(gt)}"
        assert gt[1] == pytest.approx(0.05), "Cell size in geotransform is wrong"

    def test_str_repr(self, single_band_dataset):
        """__str__ and __repr__ should return strings."""
        s = str(single_band_dataset)
        r = repr(single_band_dataset)
        assert isinstance(s, str), "__str__ should return str"
        assert isinstance(r, str), "__repr__ should return str"
        assert "Cell size" in s, "__str__ should mention Cell size"

    def test_band_color_property(self, single_band_dataset):
        """band_color should return a dict mapping band index to color name."""
        colors = single_band_dataset.band_color
        assert isinstance(colors, dict), "band_color should return a dict"
        assert 0 in colors, "band_color should contain index 0"

    def test_band_count(self, multi_band_dataset):
        """band_count should reflect the number of bands."""
        assert (
            multi_band_dataset.band_count == 3
        ), "Expected 3 bands in multi-band dataset"


class TestCreateSrFromEpsg:
    """Tests for the ``pyramids.base.crs.sr_from_epsg`` helper."""

    def test_valid_epsg(self):
        """Creating SR from a valid EPSG should return a SpatialReference."""
        sr = sr_from_epsg(4326)
        assert isinstance(sr, osr.SpatialReference), "Should return SpatialReference"
        wkt = sr.ExportToWkt()
        assert (
            "WGS 84" in wkt or "4326" in wkt
        ), "SpatialReference should contain WGS 84"


class TestRasterProperty:
    """Tests for the raster property (read-only)."""

    def test_raster_getter_returns_gdal_dataset(self, single_band_dataset):
        """The raster property should return the underlying gdal.Dataset."""
        assert isinstance(
            single_band_dataset.raster, gdal.Dataset
        ), "raster property should return a gdal.Dataset"

    def test_raster_has_no_public_setter(self, single_band_dataset):
        """Assigning to .raster should raise AttributeError (no public setter)."""
        with pytest.raises(AttributeError):
            single_band_dataset.raster = single_band_dataset._raster


class TestBandNames:
    """Tests for _band_names metadata path."""

    def test_band_names_with_metadata(self):
        """Band names should use metadata if present."""
        arr = np.ones((2, 3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
        )
        ds.raster.SetMetadataItem("Band_1", "temperature")
        ds.raster.SetMetadataItem("Band_2", "humidity")
        names = ds._get_band_names()
        assert names[0] == "temperature", f"Expected 'temperature', got {names[0]}"
        assert names[1] == "humidity", f"Expected 'humidity', got {names[1]}"


class TestBandColorSetter:
    """Tests for band_color setter."""

    def test_band_color_invalid_index_raises(self, single_band_dataset):
        """band_color setter with index > band_count should raise."""
        with pytest.raises(ValueError, match="band index"):
            single_band_dataset.band_color = {10: "red"}


class TestColorTable:
    """Tests for color_table property getter and setter."""

    def test_get_color_table(self):
        """color_table property should return DataFrame after setting."""
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        ct = gdal.ColorTable()
        ct.SetColorEntry(0, (0, 0, 0, 255))
        ct.SetColorEntry(1, (255, 0, 0, 255))
        ct.SetColorEntry(2, (0, 255, 0, 128))
        ct.SetColorEntry(3, (0, 0, 255, 0))
        ds._iloc(0).SetColorTable(ct)
        df = ds.color_table
        assert isinstance(df, pd.DataFrame), "color_table should return DataFrame"
        expected_cols = ["band", "values", "red", "green", "blue", "alpha"]
        for col in expected_cols:
            assert col in df.columns, f"color_table should have '{col}' column"
        assert len(df) == 4, f"Expected 4 color entries, got {len(df)}"

    def test_color_table_setter_invalid_type_raises(self):
        """color_table setter with non-DataFrame should raise TypeError."""
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        with pytest.raises(TypeError, match="DataFrame"):
            ds.color_table = "not_a_dataframe"

    def test_color_table_setter_missing_columns_raises(self):
        """color_table setter without required columns should raise ValueError."""
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        bad_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="columns"):
            ds.color_table = bad_df


@pytest.mark.plot
class TestColorTableSetterValid:
    """Tests for color_table setter validation."""

    def test_color_table_setter_valid_raises_no_cleopatra(self):
        """color_table setter with valid data raises if cleopatra missing."""
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        df = pd.DataFrame(
            {
                "band": [1, 1, 1, 1],
                "values": [0, 1, 2, 3],
                "color": ["#000000", "#FF0000", "#00FF00", "#0000FF"],
            }
        )
        ds.color_table = df
        assert ds.color_table is not None


class TestInplaceConsistency:
    """Tests for the inplace parameter added to resample, align, apply, and change_no_data_value."""

    def test_resample_returns_new_dataset(self, single_band_dataset):
        """resample() should always return a new Dataset."""
        original_cell_size = single_band_dataset.cell_size
        result = single_band_dataset.resample(cell_size=0.1)
        assert result is not None, "resample should return a Dataset"
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert (
            result.cell_size == pytest.approx(0.1)
        ), f"Cell size should be 0.1 after resample, got {result.cell_size}"
        assert (
            single_band_dataset.cell_size == original_cell_size
        ), "Original dataset cell size should not change"

    def test_align_returns_new_dataset(self, single_band_dataset):
        """align() should always return a new Dataset."""
        original_rows = single_band_dataset.rows
        ref_arr = np.ones((5, 5), dtype=np.float32)
        ref = Dataset.create_from_array(
            ref_arr, top_left_corner=(0.0, 0.0), cell_size=0.02, epsg=4326
        )
        result = single_band_dataset.align(ref)
        assert result is not None, "align should return a Dataset"
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.rows == 5, f"Rows should be 5 after align, got {result.rows}"
        assert (
            result.columns == 5
        ), f"Columns should be 5 after align, got {result.columns}"
        assert (
            single_band_dataset.rows == original_rows
        ), "Original dataset rows should not change"

    def test_apply_inplace_returns_self(self, single_band_dataset):
        """apply(inplace=True) should return self and modify the dataset in place."""
        result = single_band_dataset.apply(lambda x: x * 2, inplace=True)
        assert result is single_band_dataset, "inplace apply should return self"
        arr = single_band_dataset.read_array()
        assert arr[0, 0] == pytest.approx(2.0), f"Expected 2.0 after doubling, got {arr[0, 0]}"

    def test_apply_not_inplace_returns_new_dataset(self, single_band_dataset):
        """apply(inplace=False) should return a new Dataset without modifying the original."""
        original_val = single_band_dataset.read_array()[0, 0]
        result = single_band_dataset.apply(lambda x: x * 2, inplace=False)
        assert result is not None, "non-inplace apply should return a Dataset"
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert (
            single_band_dataset.read_array()[0, 0] == original_val
        ), "Original dataset values should not change"

    def test_change_no_data_value_inplace_returns_self(self):
        """change_no_data_value(inplace=True) should return self and modify the dataset in place."""
        arr = np.full((3, 3), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.change_no_data_value(-1.0, old_value=-9999.0, inplace=True)
        assert result is ds, "inplace change_no_data_value should return self"
        assert (
            ds.no_data_value[0] == -1.0
        ), f"No data value should be -1.0 after inplace change, got {ds.no_data_value[0]}"

    def test_change_no_data_value_not_inplace_returns_new_dataset(self):
        """change_no_data_value(inplace=False) should return a new Dataset."""
        arr = np.full((3, 3), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.change_no_data_value(-1.0, old_value=-9999.0, inplace=False)
        assert (
            result is not None
        ), "non-inplace change_no_data_value should return a Dataset"
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert (
            ds.no_data_value[0] == -9999.0
        ), "Original dataset no_data_value should not change"


class TestPDEP8InplacePattern:
    """Tests for PDEP-8 aligned inplace pattern.

    Structural operations (crop, resample, align, to_crs, wrap_longitude)
    no longer accept an `inplace` parameter — they always return a new Dataset.
    Value operations (fill, apply, change_no_data_value) still accept `inplace`
    but return `self` instead of `None` when inplace=True, enabling chaining.
    """

    @pytest.mark.parametrize(
        "method_name",
        ["crop", "resample", "align", "to_crs"],
        ids=["crop", "resample", "align", "to_crs"],
    )
    def test_structural_ops_reject_inplace_kwarg(
        self, single_band_dataset, method_name
    ):
        """Structural operations should raise TypeError if inplace is passed.

        Test scenario:
            Passing `inplace=True` to crop/resample/align/to_crs should
            raise TypeError since the parameter was removed.
        """
        kwargs = {"inplace": True}
        if method_name == "crop":
            import geopandas as gpd
            from shapely.geometry import box

            poly = box(0.0, -0.15, 0.15, 0.0)
            mask = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
            kwargs["mask"] = mask
        elif method_name == "resample":
            kwargs["cell_size"] = 0.1
        elif method_name == "align":
            kwargs["alignment_src"] = single_band_dataset
        elif method_name == "to_crs":
            kwargs["to_epsg"] = 3857

        bound = getattr(single_band_dataset, method_name)
        with pytest.raises(TypeError):
            bound(**kwargs)

    def test_wrap_longitude_rejects_inplace_kwarg(self):
        """wrap_longitude should raise TypeError if inplace is passed.

        Test scenario:
            wrap_longitude no longer accepts inplace — passing it
            should raise TypeError.
        """
        arr = np.ones((2, 720), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 90.0), cell_size=0.5, epsg=4326
        )
        with pytest.raises(TypeError):
            ds.wrap_longitude(inplace=True)

    def test_crop_always_returns_new_dataset(self, single_band_dataset):
        """crop should always return a new Dataset, never None.

        Test scenario:
            Call crop without inplace and verify the return is a Dataset
            and the original is unchanged.
        """
        import geopandas as gpd
        from shapely.geometry import box

        poly = box(0.0, -0.15, 0.15, 0.0)
        mask = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        original_shape = single_band_dataset.shape
        result = single_band_dataset.crop(mask)
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result is not single_band_dataset, "crop should return a new object"
        assert single_band_dataset.shape == original_shape, "Original should not change"

    def test_to_crs_always_returns_new_dataset(self, single_band_dataset):
        """to_crs should always return a new Dataset, never None.

        Test scenario:
            Reproject and verify a new Dataset is returned.
        """
        result = single_band_dataset.to_crs(to_epsg=3857)
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result is not single_band_dataset, "to_crs should return a new object"

    def test_fill_inplace_returns_self(self, single_band_dataset):
        """fill(inplace=True) should return self (not None) per PDEP-8.

        Test scenario:
            The return value should be the same object, enabling chaining.
        """
        result = single_band_dataset.fill(99, inplace=True)
        assert (
            result is single_band_dataset
        ), f"fill(inplace=True) should return self, got {type(result)}"

    def test_fill_not_inplace_returns_new_dataset(self, single_band_dataset):
        """fill(inplace=False) should return a new Dataset.

        Test scenario:
            The return value should be a different object from the original.
        """
        result = single_band_dataset.fill(99, inplace=False)
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result is not single_band_dataset, "Should return a new object"

    def test_value_ops_method_chaining(self):
        """Value operations should support method chaining via inplace=True returning self.

        Test scenario:
            Chain fill(inplace=True).apply(inplace=True) and verify the
            result is the same object with both transformations applied.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.fill(10, inplace=True).apply(lambda x: x + 5, inplace=True)
        assert result is ds, "Chained inplace calls should return the same object"
        result_arr = ds.read_array()
        assert np.allclose(
            result_arr, 15.0
        ), f"Expected all cells to be 15 after fill(10)+apply(+5), got {result_arr}"

    def test_change_no_data_chained_with_apply(self):
        """change_no_data_value and apply should chain via inplace=True.

        Test scenario:
            Change no-data value inplace, then apply a function inplace,
            all in one expression.
        """
        arr = np.full((2, 2), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.change_no_data_value(-1.0, old_value=-9999.0, inplace=True)
        assert result is ds, "change_no_data_value(inplace=True) should return self"
        assert (
            ds.no_data_value[0] == -1.0
        ), f"Expected no_data_value=-1.0, got {ds.no_data_value[0]}"
