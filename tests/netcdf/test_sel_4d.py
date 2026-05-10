"""Tests for sel() on 4-D NetCDFs (multiple band dimensions).

Covers issue #311. Uses two fixtures:
- ``tests/data/netcdf/pyramids-netcdf-4d.nc`` (synthetic, shape
  ``(time=4, pressure_level=3, lat=5, lon=6)``, pixel values encode
  ``t*1000 + l*100 + y*10 + x`` so storage order is verifiable).
- ``tests/data/netcdf/era5_cds_beta_t_pressure_levels_jan2022.nc``
  (real CDS-Beta ERA5 pressure-levels retrieval, shape
  ``(28, 1, 141, 321)``) for end-to-end coverage.

Style: Google-style docstrings, <=120 char lines, descriptive
assertions.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

SYNTH_PATH = "tests/data/netcdf/pyramids-netcdf-4d.nc"
ERA5_PATH = "tests/data/netcdf/era5_cds_beta_t_pressure_levels_jan2022.nc"

# Synthetic-fixture geometry.
NT, NL, NY, NX = 4, 3, 5, 6
TIME_VALUES = [0.0, 6.0, 12.0, 18.0]
LEVEL_VALUES = [1000.0, 850.0, 500.0]
# pyramids/GDAL flips the y-axis on read (north-up convention), so
# `arr[band, 0, 0]` corresponds to (t, l, y=NY-1, x=0) in the source
# encoding `t*1000 + l*100 + y*10 + x`.
TOP_LEFT_Y = NY - 1


def _expect(t_idx: int, l_idx: int, y_idx: int = TOP_LEFT_Y, x_idx: int = 0) -> float:
    """Pixel value the encoder wrote at (t, l, y, x) on the synthetic fixture."""
    return t_idx * 1000.0 + l_idx * 100.0 + y_idx * 10.0 + x_idx


@pytest.fixture(scope="module")
def synth_var():
    """The temperature variable from the synthetic 4-D fixture."""
    nc = NetCDF.read_file(SYNTH_PATH)
    return nc.get_variable("temperature")


@pytest.fixture(scope="module")
def era5_var():
    """The ``t`` variable from the real CDS-Beta ERA5 fixture."""
    nc = NetCDF.read_file(ERA5_PATH)
    return nc.get_variable("t")


class TestFourDimVariableMetadata:
    """Build path must populate the new band-dim fields with every non-spatial dim."""

    def test_both_band_dims_tracked(self, synth_var):
        """``_band_dim_names`` lists both non-spatial dims in storage order."""
        assert synth_var._band_dim_names == ("time", "pressure_level"), (
            f"Expected ('time', 'pressure_level'), got {synth_var._band_dim_names!r}"
        )

    def test_primary_band_dim_legacy_fields(self, synth_var):
        """Legacy ``_band_dim_name`` / ``_band_dim_values`` point at the primary dim."""
        assert synth_var._band_dim_name == "time", (
            f"primary dim must be 'time', got {synth_var._band_dim_name!r}"
        )
        assert synth_var._band_dim_values == TIME_VALUES, (
            f"primary values mismatch: {synth_var._band_dim_values!r}"
        )

    def test_values_map_keyed_by_dim_name(self, synth_var):
        """``_band_dim_values_map`` carries coords per dim, by dim name."""
        assert synth_var._band_dim_values_map["time"] == TIME_VALUES
        assert synth_var._band_dim_values_map["pressure_level"] == LEVEL_VALUES

    def test_sizes_in_storage_order(self, synth_var):
        """``_band_dim_sizes`` matches the dim ordering."""
        assert synth_var._band_dim_sizes == (NT, NL), (
            f"sizes must be ({NT}, {NL}), got {synth_var._band_dim_sizes}"
        )

    def test_md_array_dims_lock_storage_order(self, synth_var):
        """Band dims appear first in storage order; spatial dims follow.

        Pyramids may rewrite the spatial dim names internally (e.g.
        ``subset_lat_*`` when materializing the y-flip), so check the
        positions of the band dims only.
        """
        assert synth_var._md_array_dims[:2] == ["time", "pressure_level"], (
            f"band dims must be first two: {synth_var._md_array_dims}"
        )
        assert len(synth_var._md_array_dims) == 4, (
            f"4-D variable must have 4 dims: {synth_var._md_array_dims}"
        )


class TestSelByPressureLevel:
    """``sel(pressure_level=...)`` on a 4-D file."""

    def test_select_single_level_returns_one_per_time(self, synth_var):
        """Pinning one level keeps every time step → shape (NT, NY, NX)."""
        result = synth_var.sel(pressure_level=500)
        assert result.read_array().shape == (NT, NY, NX), (
            f"Expected ({NT}, {NY}, {NX}), got {result.read_array().shape}"
        )

    def test_select_single_level_pixel_values_match_encoding(self, synth_var):
        """Each band's top-left pixel matches ``encode(t, l=2, y=NY-1, x=0)``."""
        arr = synth_var.sel(pressure_level=500).read_array()
        for t in range(NT):
            assert arr[t, 0, 0] == _expect(t, 2), (
                f"Band {t}: expected {_expect(t, 2)}, got {arr[t, 0, 0]}"
            )

    def test_pressure_level_values_preserved_after_pin(self, synth_var):
        """``_band_dim_values_map`` reflects the pinned level; time dim untouched."""
        result = synth_var.sel(pressure_level=500)
        assert result._band_dim_values_map["pressure_level"] == [500.0]
        assert result._band_dim_values_map["time"] == TIME_VALUES

    def test_select_multiple_levels(self, synth_var):
        """Selecting two levels keeps NT*2 bands."""
        result = synth_var.sel(pressure_level=[1000, 500])
        # Two pinned levels × NT times = NT*2 bands.
        assert result.read_array().shape == (NT * 2, NY, NX), (
            f"Expected ({NT * 2}, {NY}, {NX}), got {result.read_array().shape}"
        )

    def test_sel_updates_band_dim_sizes(self, synth_var):
        """``_band_dim_sizes`` reflects the pinned axis after sel()."""
        result = synth_var.sel(pressure_level=500)
        assert result._band_dim_sizes == (NT, 1), (
            f"Expected ({NT}, 1), got {result._band_dim_sizes}"
        )


class TestSelByTime:
    """``sel(time=...)`` on a 4-D file (the legacy primary dim)."""

    def test_select_single_time_collapses_to_levels(self, synth_var):
        """Pinning one time leaves NL levels → shape (NL, NY, NX)."""
        result = synth_var.sel(time=12)
        assert result.read_array().shape == (NL, NY, NX), (
            f"Expected ({NL}, {NY}, {NX}), got {result.read_array().shape}"
        )

    def test_select_single_time_pixel_values(self, synth_var):
        """Each level-band's top-left matches ``encode(t=2, l, y=NY-1, x=0)``."""
        arr = synth_var.sel(time=12).read_array()
        for l_idx in range(NL):
            assert arr[l_idx, 0, 0] == _expect(2, l_idx), (
                f"Level {l_idx}: expected {_expect(2, l_idx)}, "
                f"got {arr[l_idx, 0, 0]}"
            )


class TestSelChained:
    """Chained sel() pins multiple band dims."""

    def test_pin_time_then_level(self, synth_var):
        """``sel(time=…).sel(pressure_level=…)`` flattens to a single 2-D map."""
        result = synth_var.sel(time=12).sel(pressure_level=500)
        # 1 time × 1 level → squeezed to 2-D.
        assert result.read_array().shape == (NY, NX), (
            f"Expected ({NY}, {NX}), got {result.read_array().shape}"
        )

    def test_pin_level_then_time_same_result(self, synth_var):
        """sel commutes over different dims (assert byte-identical arrays)."""
        a = synth_var.sel(time=12).sel(pressure_level=500).read_array()
        b = synth_var.sel(pressure_level=500).sel(time=12).read_array()
        assert_array_equal(
            a, b, err_msg="sel() must commute over different dims"
        )

    def test_chained_pixel_value(self, synth_var):
        """The single pinned cell is exactly ``encode(2, 2, NY-1, 0)``."""
        arr = synth_var.sel(time=12).sel(pressure_level=500).read_array()
        assert arr[0, 0] == _expect(2, 2), (
            f"Expected {_expect(2, 2)}, got {arr[0, 0]}"
        )


class TestSelErrorMessages:
    """Error contract on 4-D files."""

    def test_unknown_dim_name(self, synth_var):
        """Unknown dim name raises a ValueError mentioning the available dims."""
        with pytest.raises(ValueError, match="any band dimension"):
            synth_var.sel(latitude=42)

    def test_value_not_present(self, synth_var):
        """A missing coordinate value yields a ValueError listing the available ones."""
        with pytest.raises(ValueError, match="No bands match"):
            synth_var.sel(pressure_level=999)

    def test_too_many_kwargs(self, synth_var):
        """``sel()`` rejects calls with more than one keyword argument."""
        with pytest.raises(ValueError, match="exactly one keyword"):
            synth_var.sel(time=12, pressure_level=500)


class TestEra5RealFixture:
    """End-to-end check on a real CDS-Beta ERA5 pressure-levels retrieval.

    The fixture is sliced down to `(valid_time=4, pressure_level=1,
    latitude=8, longitude=10)` (~19 KB) to keep the test data small while
    preserving the CDS-Beta dim names, indexing variables, and CF
    attributes. The ``pressure_level=1`` makes some level-axis tests
    degenerate (it always was — the original CDS retrieval shipped only
    one level), but the storage-order math is still exercised on real
    CDS-derived data.
    """

    ERA5_NT, ERA5_NL, ERA5_NY, ERA5_NX = 4, 1, 8, 10

    def test_band_dim_metadata(self, era5_var):
        """Both dims are tracked from the multidim driver."""
        assert era5_var._band_dim_names == (
            "valid_time",
            "pressure_level",
        ), f"got {era5_var._band_dim_names!r}"
        assert era5_var._band_dim_sizes == (self.ERA5_NT, self.ERA5_NL), (
            f"got {era5_var._band_dim_sizes!r}"
        )
        assert era5_var._band_dim_values_map["pressure_level"] == [500.0]

    def test_sel_pressure_level_passes_through_time_bands(self, era5_var):
        """sel(pressure_level=500) keeps every time step (NL is already 1)."""
        result = era5_var.sel(pressure_level=500)
        expected = (self.ERA5_NT, self.ERA5_NY, self.ERA5_NX)
        assert result.read_array().shape == expected, (
            f"got {result.read_array().shape}"
        )

    def test_sel_valid_time_collapses_to_2d(self, era5_var):
        """A single valid_time pin collapses to 2-D (1 time × 1 level)."""
        first_t = era5_var._band_dim_values_map["valid_time"][0]
        result = era5_var.sel(valid_time=first_t)
        expected = (self.ERA5_NY, self.ERA5_NX)
        assert result.read_array().shape == expected, (
            f"got {result.read_array().shape}"
        )


class TestRootContainer4DSpatialOps:
    """Verify `crop()` / `to_crs()` / `resample()` on a 4-D *root container*
    preserve every band-dim instead of flattening the secondary axis (issue #314).

    The PR-review M2 finding pointed out that the rebuild path through
    `_apply_to_all_variables` previously called `create_from_array` with a
    single `extra_dim_*` API, silently dropping the secondary band-dim. This
    suite locks in the multi-band-dim round-trip.
    """

    def test_create_from_array_with_extra_dims(self):
        """`extra_dims` API materialises every non-spatial dim on a 4-D array."""
        arr = np.arange(2 * 3 * 5 * 6).reshape(2, 3, 5, 6).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            extra_dims=[("time", [10, 20]), ("level", [1000, 850, 500])],
            variable_name="temp",
        )
        var = nc.get_variable("temp")
        assert var._band_dim_names == ("time", "level"), (
            f"expected ('time', 'level'), got {var._band_dim_names!r}"
        )
        assert var._band_dim_sizes == (2, 3), (
            f"expected sizes (2, 3), got {var._band_dim_sizes!r}"
        )
        assert var._band_dim_values_map["time"] == [10.0, 20.0]
        assert var._band_dim_values_map["level"] == [1000.0, 850.0, 500.0]

    def test_create_from_array_extra_dims_mutually_exclusive_with_legacy(self):
        """`extra_dims` and `extra_dim_values` together raise `ValueError`."""
        arr = np.zeros((3, 5, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="mutually exclusive"):
            NetCDF.create_from_array(
                arr=arr,
                geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
                extra_dim_values=[1, 2, 3],
                extra_dims=[("time", [1, 2, 3])],
                variable_name="temp",
            )

    def test_create_from_array_extra_dims_length_validated(self):
        """`extra_dims` length must equal `arr.ndim - 2`."""
        arr = np.zeros((2, 3, 5, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="must have 2 entries"):
            NetCDF.create_from_array(
                arr=arr,
                geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
                extra_dims=[("time", [1, 2])],   # only 1 entry, need 2
                variable_name="temp",
            )

    def test_create_from_array_extra_dims_values_length_validated(self):
        """Each per-dim values list must match `arr.shape[i]`."""
        arr = np.zeros((2, 3, 5, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="does not match arr.shape"):
            NetCDF.create_from_array(
                arr=arr,
                geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
                extra_dims=[("time", [1, 2, 3]), ("level", [1, 2, 3])],
                variable_name="temp",
            )

    def test_crop_root_container_preserves_both_band_dims(self):
        """`nc.crop(mask=...)` on the bundled CDS-Beta 4-D fixture keeps both dims."""
        import geopandas as gpd
        from shapely.geometry import box

        nc = NetCDF.read_file(ERA5_PATH)
        var = nc.get_variable("t")
        mask = gpd.GeoDataFrame(
            geometry=[box(-49.5, 63.5, -48.5, 64.5)], crs=f"EPSG:{var.epsg}"
        )

        result = nc.crop(mask=mask)
        inner = result.get_variable("t")

        assert inner._band_dim_names == ("valid_time", "pressure_level"), (
            f"crop on root container dropped a band-dim: "
            f"got {inner._band_dim_names!r}"
        )
        assert inner._band_dim_sizes == (4, 1), (
            f"sizes mismatch after crop: {inner._band_dim_sizes!r}"
        )
        assert inner._band_dim_values_map["pressure_level"] == [500.0]
        # sel() across either axis still works on the cropped container.
        sub = inner.sel(pressure_level=500)
        assert sub.read_array().shape[0] == 4, (
            f"pin level should leave 4 time bands, got "
            f"{sub.read_array().shape}"
        )

    def test_crop_root_container_synthetic_4d_round_trip(self):
        """Synthetic `(4, 3)` cube survives a no-op-style crop on the root.

        Test scenario:
            Crop with a mask that covers the full extent. The output
            shape and `_band_dim_*` fields must match the input.
        """
        import geopandas as gpd
        from shapely.geometry import box

        nc = NetCDF.read_file(SYNTH_PATH)
        var = nc.get_variable("temperature")
        # Mask spanning the whole extent — crop is essentially a copy
        # via the multi-variable rebuild path.
        xmin, ymin, xmax, ymax = (-10.5, 39.5, -4.5, 44.5)
        mask = gpd.GeoDataFrame(
            geometry=[box(xmin, ymin, xmax, ymax)], crs=f"EPSG:{var.epsg}"
        )

        result = nc.crop(mask=mask)
        inner = result.get_variable("temperature")

        assert inner._band_dim_names == ("time", "pressure_level"), (
            f"got {inner._band_dim_names!r}"
        )
        assert inner._band_dim_sizes == (NT, NL), (
            f"sizes mismatch: {inner._band_dim_sizes!r}"
        )
        # Values must round-trip
        assert inner._band_dim_values_map["time"] == TIME_VALUES
        assert inner._band_dim_values_map["pressure_level"] == LEVEL_VALUES
