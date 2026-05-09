"""End-to-end tests for ``NetCDF.create_from_array``.

Every test creates a NetCDF, inspects the MDIM structure, reads data
back through ``get_variable``, and — where applicable — saves to disk,
reloads, and verifies the round-trip.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
SEED = 42


class TestCreateFromArray2D:
    """2-D array (rows, cols) -> single-variable NetCDF with x/y dims only."""

    def test_variable_exists(self):
        """A 2-D array should produce a container with one named variable.

        Test scenario:
            Create from a (10, 20) array, verify the variable name appears
            in ``variable_names``.
        """
        arr = np.random.RandomState(SEED).rand(10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="elevation",
            path=None,
        )
        assert (
            "elevation" in nc.variable_names
        ), f"Expected 'elevation' in {nc.variable_names}"

    def test_dimensions_are_x_y_only(self):
        """A 2-D array should produce exactly two spatial dimensions.

        Test scenario:
            Dimension names should be ``["x", "y"]`` (no extra dim).
        """
        arr = np.random.RandomState(SEED).rand(8, 12).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="dem",
            path=None,
        )
        dims = nc.dimension_names
        assert "x" in dims, f"Expected 'x' in dims, got {dims}"
        assert "y" in dims, f"Expected 'y' in dims, got {dims}"
        assert len(dims) == 2, f"Expected 2 dimensions for 2-D array, got {len(dims)}"

    def test_variable_shape(self):
        """Extracted variable shape should be ``(1, rows, cols)``.

        Test scenario:
            A single 2-D slice becomes 1 band in classic raster view.
        """
        arr = np.random.RandomState(SEED).rand(10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="sst",
            path=None,
        )
        var = nc.get_variable("sst")
        assert var.shape == (1, 10, 20), f"Expected (1, 10, 20), got {var.shape}"

    def test_data_preserved(self):
        """Array values should survive the create -> get_variable round-trip.

        Test scenario:
            Read the variable back and compare to the original array.
        """
        arr = np.random.RandomState(SEED).rand(6, 8).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="precip",
            path=None,
        )
        var = nc.get_variable("precip")
        read_back = var.read_array(band=0)
        assert_allclose(
            read_back,
            arr,
            rtol=1e-6,
            err_msg="2-D data not preserved through round-trip",
        )

    def test_no_data_value_set(self):
        """The no-data value should propagate to the variable's MDArray.

        Test scenario:
            Pass ``no_data_value=-999`` and verify it on the extracted
            variable.
        """
        arr = np.random.RandomState(SEED).rand(5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            no_data_value=-999.0,
            variable_name="ndv_test",
            path=None,
        )
        var = nc.get_variable("ndv_test")
        assert_allclose(
            var.no_data_value[0],
            -999.0,
            rtol=1e-6,
            err_msg=f"Expected nodata=-999.0, got {var.no_data_value[0]}",
        )

    def test_epsg_propagated(self):
        """The EPSG code should be readable on the extracted variable.

        Test scenario:
            Create with ``epsg=32637`` and verify.
        """
        arr = np.random.RandomState(SEED).rand(5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            epsg=32637,
            variable_name="utm_var",
            path=None,
        )
        var = nc.get_variable("utm_var")
        assert var.epsg == 32637, f"Expected EPSG 32637, got {var.epsg}"


class TestCreateFromArray3D:
    """3-D array (extra_dim, rows, cols) -> variable with time/level dim."""

    def test_extra_dim_created(self):
        """The extra dimension should appear in ``dimension_names``.

        Test scenario:
            Create with ``extra_dim_name="time"`` and verify.
        """
        arr = np.random.RandomState(SEED).rand(5, 10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            extra_dim_name="time",
            path=None,
        )
        dims = nc.dimension_names
        assert "time" in dims, f"Expected 'time' in dims, got {dims}"
        assert len(dims) == 3, f"Expected 3 dimensions, got {len(dims)}"

    def test_custom_dim_name(self):
        """Users should be able to set any dimension name.

        Test scenario:
            Use ``extra_dim_name="depth"`` and verify it appears.
        """
        arr = np.random.RandomState(SEED).rand(4, 6, 8).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="salinity",
            extra_dim_name="depth",
            path=None,
        )
        assert (
            "depth" in nc.dimension_names
        ), f"Expected 'depth' in {nc.dimension_names}"

    def test_custom_dim_values(self):
        """Explicit coordinate values should be stored on the dimension.

        Test scenario:
            Pass ``extra_dim_values=[100, 200, 300]`` and read them
            back from the root group.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="pressure",
            extra_dim_name="level",
            extra_dim_values=[100, 200, 300],
            path=None,
        )
        rg = nc._raster.GetRootGroup()
        level_arr = rg.OpenMDArray("level")
        stored = level_arr.ReadAsArray().tolist()
        assert stored == [
            100.0,
            200.0,
            300.0,
        ], f"Expected [100, 200, 300], got {stored}"

    def test_default_dim_values_are_zero_indexed(self):
        """When ``extra_dim_values`` is None, defaults to ``[0, 1, ..., N-1]``.

        Test scenario:
            Omit ``extra_dim_values`` for a 4-slice array and verify
            the stored coordinates.
        """
        arr = np.random.RandomState(SEED).rand(4, 5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="wind",
            extra_dim_name="time",
            path=None,
        )
        rg = nc._raster.GetRootGroup()
        time_arr = rg.OpenMDArray("time")
        stored = time_arr.ReadAsArray().tolist()
        assert stored == [0.0, 1.0, 2.0, 3.0], f"Expected [0, 1, 2, 3], got {stored}"

    def test_variable_band_count_matches_extra_dim(self):
        """Classic view should have one band per extra-dim slice.

        Test scenario:
            A (5, 10, 20) array -> variable with 5 bands.
        """
        arr = np.random.RandomState(SEED).rand(5, 10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temp",
            path=None,
        )
        var = nc.get_variable("temp")
        assert var.band_count == 5, f"Expected 5 bands, got {var.band_count}"

    def test_3d_data_preserved(self):
        """All slices should survive the round-trip.

        Test scenario:
            Create 3-D -> extract variable -> read all bands -> compare.
        """
        arr = np.random.RandomState(SEED).rand(3, 8, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="rain",
            path=None,
        )
        var = nc.get_variable("rain")
        read_back = var.read_array()
        assert_allclose(
            read_back,
            arr,
            rtol=1e-6,
            err_msg="3-D data not preserved through round-trip",
        )


class TestCreateFromArrayGeoParams:
    """``geo`` vs ``top_left_corner`` + ``cell_size`` parameter paths."""

    def test_geo_tuple(self):
        """Creating with an explicit ``geo`` tuple should set the geotransform.

        Test scenario:
            Pass ``geo=(30, 0.5, 0, 35, 0, -0.5)`` and verify on the
            extracted variable.
        """
        arr = np.random.RandomState(SEED).rand(10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="v",
            path=None,
        )
        var = nc.get_variable("v")
        gt = var.geotransform
        assert gt is not None, "Geotransform should not be None"
        assert gt[1] == 0.5, f"Expected pixel size 0.5, got {gt[1]}"

    def test_top_left_corner_and_cell_size(self):
        """``top_left_corner`` + ``cell_size`` should build ``geo`` internally.

        Test scenario:
            Omit ``geo``, provide ``top_left_corner=(30, 35)`` and
            ``cell_size=0.5``. Verify the resulting geotransform.
        """
        arr = np.random.RandomState(SEED).rand(10, 20).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            top_left_corner=(30.0, 35.0),
            cell_size=0.5,
            variable_name="v",
            path=None,
        )
        var = nc.get_variable("v")
        gt = var.geotransform
        assert gt is not None, "Geotransform should not be None"
        assert gt[1] == 0.5, f"Expected pixel size 0.5, got {gt[1]}"

    def test_no_geo_raises(self):
        """Omitting both ``geo`` and ``top_left_corner`` should raise.

        Test scenario:
            Call without any spatial reference -> ``ValueError``.
        """
        arr = np.random.RandomState(SEED).rand(5, 5).astype(np.float64)
        with pytest.raises(ValueError, match="top_left_corner"):
            NetCDF.create_from_array(
                arr=arr,
                variable_name="v",
                path=None,
            )


class TestCreateFromArrayValidation:
    """Input validation via ``DimMetaData``."""

    def test_mismatched_extra_dim_values_raises(self):
        """``extra_dim_values`` length must match ``arr.shape[0]``.

        Test scenario:
            Pass 2 values for a 3-slice array -> ``ValueError``.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        with pytest.raises(ValueError, match="values length.*does not match size"):
            NetCDF.create_from_array(
                arr=arr,
                geo=GEO,
                variable_name="v",
                extra_dim_name="time",
                extra_dim_values=[0, 6],
                path=None,
            )

    def test_empty_dim_name_raises(self):
        """An empty ``extra_dim_name`` should raise.

        Test scenario:
            Pass ``extra_dim_name=""`` for a 3-D array -> ``ValueError``.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        with pytest.raises(ValueError, match="name cannot be empty"):
            NetCDF.create_from_array(
                arr=arr,
                geo=GEO,
                variable_name="v",
                extra_dim_name="",
                path=None,
            )

    def test_extra_dim_values_correct_length_passes(self):
        """Matching length should not raise.

        Test scenario:
            Pass 3 values for a 3-slice array -> no error.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="v",
            extra_dim_name="time",
            extra_dim_values=[0, 6, 12],
            path=None,
        )
        assert (
            "v" in nc.variable_names
        ), f"Variable should exist, got {nc.variable_names}"


class TestCreateFromArrayDefaults:
    """Default parameter behavior."""

    def test_default_variable_name(self):
        """When ``variable_name`` is None, it should default to ``"data"``.

        Test scenario:
            Omit ``variable_name`` and verify.
        """
        arr = np.random.RandomState(SEED).rand(5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            path=None,
        )
        assert "data" in nc.variable_names, f"Expected 'data' in {nc.variable_names}"

    def test_default_extra_dim_name_is_time(self):
        """When ``extra_dim_name`` is not provided, it defaults to ``"time"``.

        Test scenario:
            Create a 3-D array without specifying ``extra_dim_name``.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="v",
            path=None,
        )
        assert "time" in nc.dimension_names, f"Expected 'time' in {nc.dimension_names}"

    def test_default_epsg_is_4326(self):
        """When ``epsg`` is not provided, it defaults to 4326.

        Test scenario:
            Omit ``epsg`` and verify on the extracted variable.
        """
        arr = np.random.RandomState(SEED).rand(5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="v",
            path=None,
        )
        var = nc.get_variable("v")
        assert var.epsg == 4326, f"Expected EPSG 4326, got {var.epsg}"


class TestCreateFromArrayDiskRoundTrip:
    """Save to disk and reload — full end-to-end round-trip."""

    def test_2d_disk_roundtrip(self, tmp_path):
        """2-D array -> save -> reload -> verify variable and data.

        Test scenario:
            Create, save to .nc, reload, compare variable names and
            array values.
        """
        arr = np.random.RandomState(SEED).rand(8, 12).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="elev",
            path=None,
        )
        out = str(tmp_path / "test_2d.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert (
            "elev" in reloaded.variable_names
        ), f"Expected 'elev' in {reloaded.variable_names}"
        var = reloaded.get_variable("elev")
        assert_allclose(
            var.read_array(band=0),
            arr,
            rtol=1e-5,
            err_msg="2-D disk round-trip data mismatch",
        )

    def test_3d_disk_roundtrip(self, tmp_path):
        """3-D array -> save -> reload -> verify dimensions and data.

        Test scenario:
            Create with ``extra_dim_name="level"``, save, reload, verify
            the dimension name and all slices.
        """
        arr = np.random.RandomState(SEED).rand(4, 6, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="temperature",
            extra_dim_name="level",
            extra_dim_values=[1000, 850, 500, 200],
            path=None,
        )
        out = str(tmp_path / "test_3d.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert (
            "temperature" in reloaded.variable_names
        ), f"Expected 'temperature' in {reloaded.variable_names}"
        var = reloaded.get_variable("temperature")
        assert var.band_count == 4, f"Expected 4 bands (levels), got {var.band_count}"
        assert_allclose(
            var.read_array(),
            arr,
            rtol=1e-5,
            err_msg="3-D disk round-trip data mismatch",
        )

    def test_disk_roundtrip_preserves_extra_dim_values(self, tmp_path):
        """Coordinate values for the extra dimension should survive disk I/O.

        Test scenario:
            Create with ``extra_dim_values=[100, 200, 300]``, save, reload,
            read the dimension's indexing variable.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 5).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="p",
            extra_dim_name="level",
            extra_dim_values=[100, 200, 300],
            path=None,
        )
        out = str(tmp_path / "dim_values.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        rg = reloaded._raster.GetRootGroup()
        level_arr = rg.OpenMDArray("level")
        stored = level_arr.ReadAsArray().tolist()
        assert_allclose(
            stored,
            [100.0, 200.0, 300.0],
            rtol=1e-5,
            err_msg=f"Extra dim values not preserved: {stored}",
        )


class TestCreateFromArrayDtypes:
    """Different NumPy dtypes should produce valid NetCDF files."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int16,
            np.int32,
        ],
    )
    def test_dtype_roundtrip(self, dtype):
        """Array dtype should be preserved through create -> read.

        Args:
            dtype: NumPy dtype to test.

        Test scenario:
            Create from a specific dtype, extract variable, verify
            the read-back array has the same dtype.
        """
        arr = np.ones((5, 8), dtype=dtype)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="typed",
            path=None,
        )
        var = nc.get_variable("typed")
        read_back = var.read_array(band=0)
        assert (
            read_back.dtype == dtype
        ), f"Expected dtype {dtype}, got {read_back.dtype}"


class TestCreateFromArraySetVariableRoundTrip:
    """Full workflow: create -> get_variable -> modify -> set_variable -> verify."""

    def test_modify_and_set_back(self):
        """Modified data written via ``set_variable`` should be readable.

        Test scenario:
            Create 3-D NetCDF, extract variable, multiply by 2, write
            back as a new variable, verify stored values.
        """
        arr = np.random.RandomState(SEED).rand(3, 6, 8).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="original",
            extra_dim_name="time",
            extra_dim_values=[0, 6, 12],
            path=None,
        )
        var = nc.get_variable("original")
        modified = var.read_array() * 2.0
        ds = Dataset.create_from_array(
            modified,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        ds._band_dim_name = var._band_dim_name
        ds._band_dim_values = var._band_dim_values
        nc.set_variable("doubled", ds)
        assert (
            "doubled" in nc.variable_names
        ), f"Expected 'doubled' in {nc.variable_names}"
        rg = nc._raster.GetRootGroup()
        stored = rg.OpenMDArray("doubled").ReadAsArray()
        assert_allclose(
            stored,
            arr * 2.0,
            rtol=1e-5,
            err_msg="set_variable data mismatch",
        )

    def test_full_disk_workflow(self, tmp_path):
        """Create -> extract -> modify -> set_variable -> save -> reload -> verify.

        Test scenario:
            The complete end-to-end workflow saving to disk and
            reloading the modified variable.
        """
        arr = np.random.RandomState(SEED).rand(2, 10, 15).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=GEO,
            variable_name="precip",
            extra_dim_name="time",
            extra_dim_values=[0, 12],
            path=None,
        )
        var = nc.get_variable("precip")
        scaled = var.read_array() + 100.0
        ds = Dataset.create_from_array(
            scaled,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        nc.set_variable("precip_offset", ds)
        out = str(tmp_path / "workflow.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert (
            "precip_offset" in reloaded.variable_names
        ), f"Expected 'precip_offset' in {reloaded.variable_names}"
        var_reloaded = reloaded.get_variable("precip_offset")
        assert_allclose(
            var_reloaded.read_array(),
            scaled,
            rtol=1e-5,
            err_msg="Full disk workflow data mismatch",
        )
