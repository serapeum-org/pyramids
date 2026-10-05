"""Tests for NetCDF.rename_dims.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import SEED

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)


def _make_nc(var_name="temperature"):
    """A one-variable cube with a single band dimension `time` (length 3)."""
    arr = np.random.default_rng(SEED).random((3, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=var_name,
        dims=ExtraDimensions(name="time", values=[0, 6, 12]),
    )


def _make_2d_nc():
    """A one-variable cube with two band dimensions `time` (2) and `level` (3)."""
    arr = np.random.default_rng(SEED).random((2, 3, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name="v",
        dims=ExtraDimensions(dims=[("time", [0, 6]), ("level", [1000, 850, 500])]),
    )


def _make_multi_nc():
    """A container with two variables that both span the band dimension `time`."""
    nc = _make_nc("temp")
    arr2 = np.random.default_rng(99).random((3, 5, 8)).astype(np.float64)
    second = NetCDF.from_array(
        arr=arr2,
        geo_ref=GeoReference(geo=GEO),
        variable_name="pressure",
        dims=ExtraDimensions(name="time", values=[0, 6, 12]),
    )
    nc.set_variable("pressure", second.get_variable("pressure"))
    return nc


class TestRenameDimsHappyPath:
    """A rename changes the label, never the cells."""

    def test_single_variable_renames_the_dimension(self):
        """`rename_dims(time='t')` relabels the band dimension on a single variable."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._band_dim_names == ("t",), f"got {out._band_dim_names}"

    def test_the_coordinates_follow_the_rename(self):
        """The renamed dimension keeps its coordinate stamps under the new key."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._band_dim_values_map["t"] == [0, 6, 12], (
            "coordinates must follow the new name"
        )
        assert "time" not in out._band_dim_values_map, "the old key must be gone"

    def test_the_cells_are_unchanged(self):
        """A rename moves no data — the array is identical before and after."""
        var = _make_nc().get_variable("temperature")
        before = np.asarray(var.read_array())
        out = var.rename_dims(time="t")
        assert_allclose(np.asarray(out.read_array()), before)

    def test_a_single_variable_rename_is_cell_free(self):
        """The single-variable path rewraps rather than rebuilds, so it is not marked in-memory."""
        var = _make_nc().get_variable("temperature")
        out = var.rename_dims(time="t")
        assert out._rebuilt_in_memory is False, (
            "a rename must keep the variable's lazy read"
        )

    def test_the_mapping_form_and_kwargs_form_agree(self):
        """`rename_dims({'time': 't'})` equals `rename_dims(time='t')`."""
        var = _make_nc().get_variable("temperature")
        assert (
            var.rename_dims({"time": "t"})._band_dim_names
            == var.rename_dims(time="t")._band_dim_names
        )

    def test_two_dimensions_can_be_swapped(self):
        """Renaming `time->level` and `level->time` in one call swaps the two names."""
        var = _make_2d_nc().get_variable("v")
        out = var.rename_dims(time="level", level="time")
        assert out._band_dim_names == ("level", "time"), f"got {out._band_dim_names}"

    def test_a_container_renames_every_variable_that_spans_the_dim(self):
        """On a container the renamed dimension changes on every variable that has it."""
        cont = _make_multi_nc()
        out = cont.rename_dims(time="t")
        assert out.get_variable("temp")._band_dim_names == ("t",), (
            "temp must be renamed"
        )
        assert out.get_variable("pressure")._band_dim_names == ("t",), (
            "pressure must be renamed"
        )

    def test_a_container_rename_preserves_cells(self):
        """A container rename carries each variable's cells over unchanged."""
        cont = _make_multi_nc()
        before = np.asarray(cont.get_variable("temp").read_array())
        out = cont.rename_dims(time="t")
        assert_allclose(np.asarray(out.get_variable("temp").read_array()), before)


class TestRenameDimsErrors:
    """Every refusal is a ValueError naming the offending input."""

    def test_an_unknown_dimension_is_refused(self):
        """Renaming a name that is not a band dimension raises."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="not a band dimension"):
            var.rename_dims(nope="x")

    def test_a_spatial_target_is_refused(self):
        """Renaming a band dimension to a spatial axis name raises."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="spatial axis name"):
            var.rename_dims(time="lat")

    def test_two_renames_to_the_same_name_are_refused(self):
        """Two band dimensions renamed to one name raises on the duplicate target."""
        var = _make_2d_nc().get_variable("v")
        with pytest.raises(ValueError, match="duplicate"):
            var.rename_dims(time="z", level="z")

    def test_a_target_that_already_exists_is_refused(self):
        """Renaming onto an existing, non-renamed band dimension raises."""
        var = _make_2d_nc().get_variable("v")
        with pytest.raises(ValueError, match="already names a band dimension"):
            var.rename_dims(time="level")


class TestRenameDimsDiskRoundTrip:
    """The rename survives a write/read cycle."""

    def test_renamed_dimension_survives_a_round_trip(self, tmp_path):
        """A container's renamed dimension is still renamed after saving and reloading.

        Written at the container level: ``to_file`` on a lone variable emits a generic multiband
        raster that carries no named dimension, so the round trip that proves a *rename* persists
        is the container one, where the store records the dimension name.
        """
        cont = _make_multi_nc()
        out = cont.rename_dims(time="t")
        path = tmp_path / "renamed.nc"
        out.to_file(str(path))
        reloaded = NetCDF.read_file(str(path)).get_variable("temp")
        assert "t" in reloaded._band_dim_names, f"expected 't' in {reloaded._band_dim_names}"
