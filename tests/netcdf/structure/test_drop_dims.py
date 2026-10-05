"""Tests for NetCDF.drop_dims (container-only; removes variables spanning a dimension).

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import SEED

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0, 5.0, 0, -1.0)


def _time_var(name, length=3):
    """A one-variable cube with a `time` band dimension of the given length."""
    arr = np.random.default_rng(SEED).random((length, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=name,
        dims=ExtraDimensions(name="time", values=list(range(length))),
    )


def _level_var(name, length=2):
    """A one-variable cube with a `level` band dimension of the given length."""
    arr = np.random.default_rng(7).random((length, 5, 8)).astype(np.float64)
    return NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=GEO),
        variable_name=name,
        dims=ExtraDimensions(name="level", values=list(range(length))),
    )


def _mixed_container():
    """A container with `temp` along `time` and `geoid` along `level`."""
    cont = _time_var("temp")
    cont.set_variable("geoid", _level_var("geoid").get_variable("geoid"))
    return cont


class TestDropDimsHappyPath:
    """Dropping a dimension removes the variables defined along it, and only those."""

    def test_the_variable_along_the_dropped_dim_is_gone(self):
        """`drop_dims('time')` removes the variable that spans `time`."""
        out = _mixed_container().drop_dims("time")
        assert "temp" not in out.variable_names, "temp spans time and must be dropped"

    def test_a_variable_on_another_dim_survives(self):
        """A variable spanning a different dimension is untouched."""
        out = _mixed_container().drop_dims("time")
        assert "geoid" in out.variable_names, "geoid spans level and must survive"

    def test_the_dropped_dimension_is_gone_from_dims(self):
        """The dimension itself is removed, not just the variables along it (M2).

        Rebuilding from the survivors drops the orphan dimension that in-place removal left
        declared in the store, matching the method name and xarray's `drop_dims`.
        """
        out = _mixed_container().drop_dims("time")
        assert "time" not in (out.dimension_names or []), (
            f"'time' orphaned in {out.dimension_names}"
        )
        assert "level" in (out.dimension_names or []), (
            "the surviving variable's dim must remain"
        )

    def test_drop_dims_does_not_mutate_the_receiver(self):
        """The original container keeps its variable; the drop happens on a copy."""
        cont = _mixed_container()
        cont.drop_dims("time")
        assert "temp" in cont.variable_names, "drop_dims must not mutate the receiver"

    def test_multiple_dims_can_be_dropped(self):
        """Dropping both dimensions empties the container of data variables."""
        out = _mixed_container().drop_dims(["time", "level"])
        assert out.variable_names == [], (
            f"expected no variables, got {out.variable_names}"
        )


class TestDropDimsErrors:
    """Refusals: a single variable, an unknown dimension, a bad errors flag."""

    def test_a_single_variable_is_refused(self):
        """Called on a lone variable, drop_dims redirects to the container/remove_variable."""
        var = _time_var("temp").get_variable("temp")
        with pytest.raises(ValueError, match="removes whole variables"):
            var.drop_dims("time")

    def test_an_unknown_dimension_raises_by_default(self):
        """`errors='raise'` (default) refuses a dimension the container does not have."""
        with pytest.raises(ValueError, match="not a dimension of this container"):
            _mixed_container().drop_dims("season")

    def test_an_unknown_dimension_is_skipped_when_ignored(self):
        """`errors='ignore'` skips an unknown dimension and drops nothing for it."""
        out = _mixed_container().drop_dims("season", errors="ignore")
        assert set(out.variable_names) == {"temp", "geoid"}, "nothing should be dropped"

    def test_a_bad_errors_flag_raises(self):
        """An errors flag other than raise/ignore is refused."""
        with pytest.raises(ValueError, match="errors must be"):
            _mixed_container().drop_dims("time", errors="warn")
