"""Selection and subsetting: sel (coordinate-value band selection) and subset (windowed Dataset)."""

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF
from tests.netcdf.samples.conftest import RHUM, TOS

pytestmark = pytest.mark.core


def test_sel_level_pins_band_dimension(sample):
    """``sel(level=...)`` on a 4-D variable pins that level, dropping the band count from 48 to 12."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        var = nc.get_variable("rhum")
        assert var.read_array().shape[0] == 48
        selected = var.sel(level=1000.0)
        assert isinstance(selected, NetCDF)
        assert selected.read_array().shape[0] == 12, "sel should keep the 12 time steps at one level"
    finally:
        nc.close()


def test_sel_unknown_value_raises(sample):
    """Selecting a coordinate value that does not exist raises a clear error."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        with pytest.raises((ValueError, KeyError)):
            nc.get_variable("rhum").sel(level=123456.0)
    finally:
        nc.close()


def test_subset_returns_smaller_dataset(sample):
    """``subset`` with a pinned time and a bbox returns a Dataset covering a smaller extent."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        full_cols = nc.read_array(variable="tos").shape[-1]
        ds = nc.subset("tos", time=0, bbox=(0, -40, 60, 40))
        assert isinstance(ds, Dataset)
        assert ds.shape[-1] < full_cols, "bbox subset should reduce the column extent"
    finally:
        nc.close()


class TestAntimeridianCrop:
    """Crop a NetCDF variable with a geographic west > east (antimeridian) bbox."""

    @staticmethod
    def _global_variable(top_left_x=-180.0):
        """Return (source array, global NetCDF variable) for the given lon origin."""
        arr = np.arange(180 * 360, dtype="float32").reshape(180, 360)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(top_left_x, 1.0, 0.0, 90.0, 0.0, -1.0),
            epsg=4326,
            variable_name="v",
        )
        return arr, nc.get_variable("v")

    def test_strip_values_and_extent(self):
        """A -180..180 variable crop across the dateline stitches a contiguous strip."""
        arr, var = self._global_variable()
        strip = var.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert isinstance(strip, NetCDF), "result stays a NetCDF variable"
        assert strip.shape == (1, 20, 20), "20 lat x 20 lon strip"
        assert strip.bbox == pytest.approx([170.0, -10.0, 190.0, 10.0]), "past seam"
        expected = np.concatenate([arr[80:100, 350:360], arr[80:100, 0:10]], axis=-1)
        got = np.asarray(strip.read_array())
        assert np.array_equal(got, expected), "seam values preserved"

    def test_on_0_360_grid(self):
        """A 0..360 variable crops the same STAC bbox as a contiguous 170..190 strip."""
        arr, var = self._global_variable(top_left_x=0.0)
        strip = var.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        got = np.asarray(strip.read_array())
        assert np.array_equal(got, arr[80:100, 170:190]), "0..360 values preserved"

    def test_normal_bbox_unchanged(self):
        """A west < east bbox still crops normally and returns a NetCDF."""
        _, var = self._global_variable()
        out = var.crop(bbox=(10.0, -10.0, 30.0, 10.0))
        assert isinstance(out, NetCDF), "normal crop stays a NetCDF"
        assert out.shape == (1, 20, 20), "normal crop shape"

    def test_no_overlap_raises(self):
        """An antimeridian bbox disjoint from the variable's longitudes raises."""
        arr = np.arange(180 * 50, dtype="float32").reshape(180, 50)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0.0, 90.0, 0.0, -1.0),
            epsg=4326,
            variable_name="v",
        )  # lon 0..50 only
        var = nc.get_variable("v")
        with pytest.raises(ValueError, match="does not overlap"):
            var.crop(bbox=(170.0, -10.0, -170.0, 10.0))

    def test_container_fans_out_across_variables(self):
        """A root-container antimeridian crop crops every variable into the seam strip."""
        v_arr = np.arange(180 * 360, dtype="float32").reshape(180, 360)
        w_arr = (v_arr * -1.0).astype("float32")
        geo = (-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
        nc = NetCDF.create_from_array(arr=v_arr, geo=geo, epsg=4326, variable_name="v")
        nc.set_variable("w", Dataset.create_from_array(w_arr, geo=geo, epsg=4326))
        cropped = nc.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert isinstance(cropped, NetCDF), "container crop stays a NetCDF container"
        assert sorted(cropped.variable_names) == ["v", "w"], "every variable is kept"
        for name, src in (("v", v_arr), ("w", w_arr)):
            var = cropped.get_variable(name)
            assert var.shape == (1, 20, 20), f"{name} strip shape"
            assert var.bbox == pytest.approx([170.0, -10.0, 190.0, 10.0]), "past seam"
            expected = np.concatenate(
                [src[80:100, 350:360], src[80:100, 0:10]], axis=-1
            )
            assert np.array_equal(np.asarray(var.read_array()), expected), name

    def test_container_on_0_360_grid(self):
        """A 0..360 root-container antimeridian crop windows every variable to 170..190."""
        v_arr = np.arange(180 * 360, dtype="float32").reshape(180, 360)
        w_arr = (v_arr + 1000.0).astype("float32")
        geo = (0.0, 1.0, 0.0, 90.0, 0.0, -1.0)
        nc = NetCDF.create_from_array(arr=v_arr, geo=geo, epsg=4326, variable_name="v")
        nc.set_variable("w", Dataset.create_from_array(w_arr, geo=geo, epsg=4326))
        cropped = nc.crop(bbox=(170.0, -10.0, -170.0, 10.0))
        assert sorted(cropped.variable_names) == ["v", "w"], "every variable is kept"
        for name, src in (("v", v_arr), ("w", w_arr)):
            var = cropped.get_variable(name)
            assert var.shape == (1, 20, 20), f"{name} strip shape"
            got = np.asarray(var.read_array())
            assert np.array_equal(got, src[80:100, 170:190]), f"{name} 0..360 values"

    def test_single_side_overlap_returns_half(self):
        """When only one side of the seam overlaps, that half is returned as-is."""
        arr = np.arange(180 * 10, dtype="float32").reshape(180, 10)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(170.0, 1.0, 0.0, 90.0, 0.0, -1.0),
            epsg=4326,
            variable_name="v",
        )  # lon 170..180 only (west side of the seam)
        strip = nc.get_variable("v").crop(bbox=(175.0, -10.0, -170.0, 10.0))
        assert strip.bbox[0] == pytest.approx(175.0), "west edge kept"
        assert strip.bbox[2] == pytest.approx(180.0), "only the west half (no wrap)"

    def test_chunks_rejected_on_container(self):
        """``chunks`` is unsupported for an antimeridian container crop (eager merge)."""
        arr = np.arange(180 * 360, dtype="float32").reshape(180, 360)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0),
            epsg=4326,
            variable_name="v",
        )
        with pytest.raises(ValueError, match="chunks"):
            nc.crop(bbox=(170.0, -10.0, -170.0, 10.0), chunks="auto")
