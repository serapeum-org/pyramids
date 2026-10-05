"""Tests for NetCDF.assign_coords (scoped: restamp an existing band dimension).

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


class TestAssignCoordsHappyPath:
    """Restamping replaces the coordinate values, never the cells."""

    def test_single_variable_restamps_the_dimension(self):
        """`assign_coords(time=[...])` replaces the band dimension's stamps."""
        var = _make_nc().get_variable("temperature")
        out = var.assign_coords(time=[100, 200, 300])
        assert out._band_dim_values_map["time"] == [100, 200, 300], (
            "stamps must be replaced"
        )

    def test_the_dimension_name_is_unchanged(self):
        """Restamping leaves the dimension's name alone."""
        var = _make_nc().get_variable("temperature")
        out = var.assign_coords(time=[100, 200, 300])
        assert out._band_dim_names == ("time",), f"got {out._band_dim_names}"

    def test_the_cells_are_unchanged(self):
        """Restamping moves no data — the array is identical before and after."""
        var = _make_nc().get_variable("temperature")
        before = np.asarray(var.read_array())
        out = var.assign_coords(time=[100, 200, 300])
        assert_allclose(np.asarray(out.read_array()), before)

    def test_a_single_variable_restamp_is_cell_free(self):
        """The single-variable path rewraps rather than rebuilds, keeping the lazy read."""
        var = _make_nc().get_variable("temperature")
        out = var.assign_coords(time=[100, 200, 300])
        assert out._rebuilt_in_memory is False, (
            "a restamp must keep the variable's lazy read"
        )

    def test_the_mapping_form_and_kwargs_form_agree(self):
        """`assign_coords({'time': v})` equals `assign_coords(time=v)`."""
        var = _make_nc().get_variable("temperature")
        a = var.assign_coords({"time": [1, 2, 3]})._band_dim_values_map["time"]
        b = var.assign_coords(time=[1, 2, 3])._band_dim_values_map["time"]
        assert a == b

    def test_a_container_restamps_every_variable_that_spans_the_dim(self):
        """On a container the restamp reaches every variable that has the dimension."""
        cont = _make_multi_nc()
        out = cont.assign_coords(time=[7, 8, 9])
        assert out.get_variable("temp")._band_dim_values_map["time"] == [7, 8, 9]
        assert out.get_variable("pressure")._band_dim_values_map["time"] == [7, 8, 9]

    def test_a_variable_without_the_dim_is_left_untouched(self):
        """On a container, a variable that does not span the restamped dim is unchanged."""
        cont = _make_nc("temp")
        arr = np.random.default_rng(5).random((2, 5, 8)).astype(np.float64)
        level = NetCDF.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=GEO),
            variable_name="geoid",
            dims=ExtraDimensions(name="level", values=[0, 1]),
        )
        cont.set_variable("geoid", level.get_variable("geoid"))
        out = cont.assign_coords(time=[7, 8, 9])
        assert out.get_variable("temp")._band_dim_values_map["time"] == [7, 8, 9], (
            "temp restamped"
        )
        assert out.get_variable("geoid")._band_dim_values_map["level"] == [0, 1], (
            "geoid untouched"
        )

    def test_an_empty_assign_is_a_no_op_on_a_container(self):
        """`assign_coords()` with nothing to change returns an equivalent container (L2)."""
        cont = _make_multi_nc()
        out = cont.assign_coords()
        assert set(out.variable_names) == set(cont.variable_names), (
            "variables preserved"
        )


class TestAssignCoordsErrors:
    """Every refusal is a ValueError naming the offending input."""

    def test_a_non_dimension_coordinate_is_refused(self):
        """Assigning a name that is not an existing band dimension raises (no index model)."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="not an existing band dimension"):
            var.assign_coords(season=[1, 2, 3])

    def test_a_length_mismatch_is_refused(self):
        """Coordinate values that do not match the dimension length raise."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="has length 3"):
            var.assign_coords(time=[1, 2])

    def test_non_one_dimensional_values_are_refused(self):
        """A 2-D coordinate array raises."""
        var = _make_nc().get_variable("temperature")
        with pytest.raises(ValueError, match="must be a 1-D sequence"):
            var.assign_coords(time=[[1, 2, 3]])

    def test_a_wrong_length_on_an_auxiliary_dimension_raises(self):
        """A wrong length on an aux-only dimension (CF `bnds`, size 2) is refused up front (L1).

        `bnds` is a declared band dimension no *data* variable spans, so the length check used to
        be deferred into the per-variable closure and never ran — the bad input was accepted and
        silently restamped nothing. It is now validated against the container's declared size
        before the per-variable rebuild.
        """
        nc = NetCDF.read_file("tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc")
        with pytest.raises(ValueError, match="has length 2"):
            nc.assign_coords(bnds=[1, 2, 3, 4, 5])


class TestAssignCoordsDiskRoundTrip:
    """The restamp survives a write/read cycle at the container level."""

    def test_restamped_coordinates_survive_a_round_trip(self, tmp_path):
        """A container's restamped coordinates are still present after saving and reloading."""
        cont = _make_multi_nc()
        out = cont.assign_coords(time=[21, 22, 23])
        path = tmp_path / "restamped.nc"
        out.to_file(str(path))
        reloaded = NetCDF.read_file(str(path)).get_variable("temp")
        assert_allclose(reloaded._band_dim_values_map["time"], [21, 22, 23])
