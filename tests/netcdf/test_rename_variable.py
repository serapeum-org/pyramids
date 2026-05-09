"""Tests for NetCDF.rename_variable.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

SEED = 42
GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)


def _make_nc(var_name="temperature"):
    """Create an in-memory NetCDF with one 3D variable."""
    arr = np.random.RandomState(SEED).rand(3, 5, 8).astype(np.float64)
    return NetCDF.create_from_array(
        arr=arr,
        geo=GEO,
        variable_name=var_name,
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12],
    )


def _make_multi_nc():
    """Create an in-memory NetCDF with two variables."""
    from pyramids.dataset import Dataset

    nc = _make_nc("temp")
    arr2 = np.random.RandomState(99).rand(3, 5, 8).astype(np.float64)
    ds2 = Dataset.create_from_array(
        arr2,
        geo=GEO,
        epsg=4326,
        no_data_value=-9999.0,
    )
    nc.set_variable("pressure", ds2)
    return nc


class TestRenameVariableHappyPath:
    """Normal rename operations."""

    def test_old_name_removed(self):
        """After rename, old name should not be in variable_names.

        Test scenario:
            Rename 'temperature' → 'temp'. 'temperature' should
            disappear from variable_names.
        """
        nc = _make_nc()
        nc.rename_variable("temperature", "temp")
        assert (
            "temperature" not in nc.variable_names
        ), f"Old name should be gone, got {nc.variable_names}"

    def test_new_name_present(self):
        """After rename, new name should be in variable_names.

        Test scenario:
            Rename 'temperature' → 'temp'. 'temp' should appear.
        """
        nc = _make_nc()
        nc.rename_variable("temperature", "temp")
        assert (
            "temp" in nc.variable_names
        ), f"New name should be present, got {nc.variable_names}"

    def test_data_preserved(self):
        """Data should be identical after rename.

        Test scenario:
            Read data before rename, rename, read again — arrays
            should match.
        """
        nc = _make_nc()
        orig_data = nc.get_variable("temperature").read_array()
        nc.rename_variable("temperature", "new_temp")
        renamed_data = nc.get_variable("new_temp").read_array()
        assert_allclose(
            renamed_data,
            orig_data,
            err_msg="Data should be preserved after rename",
        )

    def test_shape_preserved(self):
        """Variable shape should not change after rename.

        Test scenario:
            (3, 5, 8) before → (3, 5, 8) after.
        """
        nc = _make_nc()
        orig_shape = nc.get_variable("temperature").shape
        nc.rename_variable("temperature", "t")
        new_shape = nc.get_variable("t").shape
        assert new_shape == orig_shape, f"Shape changed: {orig_shape} → {new_shape}"

    def test_other_variables_unaffected(self):
        """Renaming one variable should not affect others.

        Test scenario:
            Container with 'temp' and 'pressure'. Rename 'temp' →
            'air_temp'. 'pressure' should still be accessible.
        """
        nc = _make_multi_nc()
        nc.rename_variable("temp", "air_temp")
        assert (
            "pressure" in nc.variable_names
        ), f"Other variable should be unaffected: {nc.variable_names}"
        assert (
            "air_temp" in nc.variable_names
        ), f"Renamed variable should exist: {nc.variable_names}"

    def test_variable_count_unchanged(self):
        """Number of variables should not change after rename.

        Test scenario:
            2 variables before, 2 variables after.
        """
        nc = _make_multi_nc()
        count_before = len(nc.variable_names)
        nc.rename_variable("temp", "air_temp")
        count_after = len(nc.variable_names)
        assert (
            count_after == count_before
        ), f"Variable count changed: {count_before} → {count_after}"


class TestRenameVariableErrors:
    """Error cases for rename_variable."""

    def test_old_name_not_found_raises(self):
        """Renaming a non-existent variable should raise ValueError.

        Test scenario:
            rename_variable('nonexistent', 'new') → ValueError.
        """
        nc = _make_nc()
        with pytest.raises(ValueError, match="not found"):
            nc.rename_variable("nonexistent", "new_name")

    def test_new_name_already_exists_raises(self):
        """Renaming to an existing name should raise ValueError.

        Test scenario:
            Container with 'temp' and 'pressure'. Rename 'temp' →
            'pressure' → ValueError.
        """
        nc = _make_multi_nc()
        with pytest.raises(ValueError, match="already exists"):
            nc.rename_variable("temp", "pressure")

    def test_classic_mode_raises(self):
        """rename_variable on classic-mode container should raise.

        Test scenario:
            Open in classic mode → ValueError.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        with pytest.raises(ValueError, match="multidimensional"):
            nc.rename_variable("Band1", "precip")


class TestRenameVariableCacheInvalidation:
    """Rename should invalidate cached properties."""

    def test_variable_names_updated(self):
        """variable_names should reflect the rename immediately.

        Test scenario:
            Access variable_names before and after rename.
        """
        nc = _make_nc()
        _ = nc.variable_names
        nc.rename_variable("temperature", "t")
        names = nc.variable_names
        assert "t" in names, f"Cache should be invalidated: {names}"
        assert "temperature" not in names, f"Old name should not be in cache: {names}"

    def test_variables_dict_updated(self):
        """variables dict should reflect the rename.

        Test scenario:
            Access variables dict after rename — new key should exist.
        """
        nc = _make_nc()
        nc.rename_variable("temperature", "t2m")
        assert (
            "t2m" in nc.variables
        ), f"New name should be in variables dict: {list(nc.variables.keys())}"


class TestRenameVariableDiskRoundTrip:
    """Rename then save → reload → verify."""

    def test_rename_survives_disk(self, tmp_path):
        """Renamed variable should be present after save → reload.

        Test scenario:
            Rename → to_file → read_file → check variable_names.
        """
        nc = _make_nc()
        nc.rename_variable("temperature", "air_temp")
        out = str(tmp_path / "renamed.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out)
        assert (
            "air_temp" in reloaded.variable_names
        ), f"Renamed variable not on disk: {reloaded.variable_names}"
        assert (
            "temperature" not in reloaded.variable_names
        ), f"Old name should not be on disk: {reloaded.variable_names}"

    def test_data_preserved_on_disk(self, tmp_path):
        """Data should match after rename → save → reload.

        Test scenario:
            Read data before rename, rename, save, reload, compare.
        """
        nc = _make_nc()
        orig = nc.get_variable("temperature").read_array()
        nc.rename_variable("temperature", "t")
        out = str(tmp_path / "rt.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out)
        loaded = reloaded.get_variable("t").read_array()
        assert_allclose(
            loaded,
            orig,
            rtol=1e-5,
            err_msg="Data changed after rename + disk round-trip",
        )
