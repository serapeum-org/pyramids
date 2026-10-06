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


def _level_cube(name, length=2, geo=GEO):
    """A one-variable cube whose band dimension is `level`, of the given length."""
    arr = np.random.default_rng(length).random((length, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=geo),
        variable_name=name,
        dims=ExtraDimensions(name="level", values=list(range(length))),
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

    def test_update_into_an_empty_container_adopts_the_donor_grid(self):
        """An empty container has no reference grid, so the first donor variable sets it."""
        base = _cube("a")
        base.remove_variable("a")
        assert base.variable_names == [], "precondition: the container is empty"
        base.update(_cube("b", seed=1))
        assert "b" in base.variable_names, "the donor variable must be adopted"


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
        donor = _cube("b", seed=1)
        with pytest.raises(ValueError, match="no .*variable mapping to update"):
            var.update(donor)

    def test_update_is_atomic_when_a_later_donor_mismatches(self):
        """A good donor followed by an off-grid one leaves the receiver untouched (M1).

        The grid of every donor is validated before any is written, so the valid `b` is not
        committed when `c` fails the grid check.
        """
        base = _cube("a")
        on_grid = _cube("b", seed=1).get_variable("b")
        off_grid = _cube("c", geo=OTHER_GEO).get_variable("c")
        with pytest.raises(AlignmentError, match="different grid"):
            base.update({"b": on_grid, "c": off_grid})
        assert base.variable_names == ["a"], (
            f"a refused update must not commit 'b': {base.variable_names}"
        )

    def test_update_rejects_an_unsupported_type(self):
        """A donor that is neither a container nor a mapping raises a typed error (L3)."""
        base = _cube("a")
        with pytest.raises(TypeError, match="NetCDF container or a"):
            base.update([1, 2, 3])

    def test_two_donors_disagreeing_with_each_other_are_refused(self):
        """Two donors introducing the same dimension at different lengths are refused (M2).

        The receiver has no `level`, so neither donor conflicts with *it*; they conflict with each
        other. Snapshotting the receiver's dimensions before the write loop made the second donor's
        conflict invisible, and it landed on a silently renamed `level_4` axis.
        """
        base = _cube("a")
        first = _level_cube("b", length=2).get_variable("b")
        second = _level_cube("c", length=4).get_variable("c")
        with pytest.raises(AlignmentError, match="band dimension 'level'"):
            base.update({"b": first, "c": second})
        assert base.variable_names == ["a"], (
            f"a refused update must commit neither donor: {base.variable_names}"
        )
        assert "level_4" not in (base.dimension_names or []), (
            f"no renamed axis may be created: {base.dimension_names}"
        )

    def test_a_band_dimension_length_conflict_is_refused(self):
        """A donor whose `time` length differs from the container's is refused up front (L2).

        The container's `time` is length 3; a donor carrying a length-5 `time` used to be
        committed under a silently renamed `time_5` axis. The band axes are now validated in the
        same pre-write pass as the spatial grid, so the mismatch raises and the receiver — still
        holding only `a`, with no `time_5` dimension — is left untouched.
        """
        base = _cube("a")
        donor = NetCDF.from_array(
            arr=np.random.default_rng(5).random((5, 5, 8)).astype(np.float64),
            geo_ref=GeoReference(geo=GEO),
            variable_name="b",
            dims=ExtraDimensions(name="time", values=[0, 6, 12, 18, 24]),
        )
        with pytest.raises(AlignmentError, match="band dimension 'time'"):
            base.update(donor)
        assert base.variable_names == ["a"], (
            f"a refused update must not commit 'b': {base.variable_names}"
        )
        assert "time_5" not in (base.dimension_names or []), (
            "the conflicting donor axis must not land on a renamed dimension"
        )
