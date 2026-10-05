"""Tests for NetCDF.update (container-only; in-place add/replace of variables).

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.base._errors import AlignmentError
from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import SEED

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)
OTHER_GEO = (100.0, 1.0, 0, 5.0, 0, -1.0)


def _cube(name, geo=GEO, seed=SEED):
    """A one-variable cube with a `time` band dimension (length 3) on the given grid."""
    arr = np.random.default_rng(seed).random((3, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=geo),
        variable_name=name,
        dims=ExtraDimensions(name="time", values=[0, 6, 12]),
    )


class TestUpdateHappyPath:
    """Update adds the new variables and replaces the colliding ones, in place."""

    def test_a_new_variable_is_added(self):
        """A variable only in `other` is added to the receiver."""
        base = _cube("a")
        base.update(_cube("b", seed=1))
        assert set(base.variable_names) == {"a", "b"}, f"got {base.variable_names}"

    def test_a_colliding_variable_is_replaced(self):
        """A variable of the same name in `other` overwrites the receiver's."""
        base = _cube("a", seed=1)
        replacement = _cube("a", seed=2)
        expected = np.asarray(replacement.get_variable("a").read_array())
        base.update(replacement)
        assert_allclose(np.asarray(base.get_variable("a").read_array()), expected)

    def test_update_returns_none_and_mutates_in_place(self):
        """Like xarray, update mutates the receiver and returns None."""
        base = _cube("a")
        result = base.update(_cube("b", seed=1))
        assert result is None, "update must return None"
        assert "b" in base.variable_names, "update must mutate the receiver"

    def test_update_accepts_a_mapping(self):
        """A `{name: variable}` mapping is accepted as well as a container."""
        base = _cube("a")
        donor = _cube("b", seed=1)
        base.update({"b": donor.get_variable("b")})
        assert "b" in base.variable_names, "a mapping donor must be accepted"


class TestUpdateErrors:
    """Refusals: a grid mismatch, a single-variable receiver."""

    def test_a_grid_mismatch_is_refused(self):
        """A variable on a different grid raises rather than resampling."""
        base = _cube("a")
        off_grid = _cube("b", geo=OTHER_GEO)
        with pytest.raises(AlignmentError, match="different grid"):
            base.update(off_grid)

    def test_a_single_variable_receiver_is_refused(self):
        """Called on a lone variable, update redirects to set_variable/merge."""
        var = _cube("a").get_variable("a")
        with pytest.raises(ValueError, match="no .*variable mapping to update"):
            var.update(_cube("b", seed=1))
