"""Tests for `NetCDF.groupby_bins` — value-interval grouping over `reduce`.

`groupby_bins` cuts a band dimension's coordinates into intervals (`pandas.cut`) and reduces
each bin through the same group-reduce path `reduce(groupby=…)` uses. Each non-empty bin becomes
one output slice, labelled with the bin's left edge, in ascending edge order.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 1.0, 0.0, -1.0)


def _cube(levels, values=None, *, name="t", cols=1):
    """A cube of `len(levels)` bands over a `1 x cols` grid, dimension `level`.

    Args:
        levels: The `level` coordinate values.
        values: The band values; `arange` when omitted.
        name: The variable name.
        cols: The number of grid columns.

    Returns:
        NetCDF: The container.
    """
    n = len(levels)
    arr = (
        np.arange(n * cols, dtype="float64")
        if values is None
        else np.asarray(values, "float64")
    )
    return NetCDF.from_array(
        arr.reshape(n, 1, cols),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name=name,
        dims=ExtraDimensions(name="level", values=list(levels)),
    )


class TestGroupbyBinsReducesEachBin:
    """Each non-empty bin becomes one slice, labelled with its left edge, ascending."""

    def test_explicit_edges_bin_and_reduce(self):
        """Two 500-wide bins average their members and are labelled by their left edges."""
        binned = _cube([100.0, 300.0, 600.0, 900.0]).groupby_bins(
            "level", [0, 500, 1000], "mean"
        )
        var = binned.get_variable("t")
        assert var._band_dim_values_map["level"] == [0.0, 500.0]
        assert_allclose(var.read_array().ravel(), [0.5, 2.5])

    def test_matches_a_hand_built_reduce_groupby(self):
        """`groupby_bins` equals `reduce(groupby=pd.cut(...))` on the same cells."""
        cube = _cube([100.0, 300.0, 600.0, 900.0])
        binned = cube.groupby_bins("level", [0, 500, 1000], "mean").get_variable("t")
        codes = pd.cut([100.0, 300.0, 600.0, 900.0], [0, 500, 1000], labels=False)
        manual = cube.reduce(
            "level", "mean", groupby=[str(c) for c in codes]
        ).get_variable("t")
        assert_allclose(binned.read_array(), manual.read_array())

    def test_an_int_bins_count_spans_the_range(self):
        """An integer `bins` makes that many equal-width bins covering the data range."""
        binned = _cube([100.0, 300.0, 600.0, 900.0]).groupby_bins("level", 2, "mean")
        var = binned.get_variable("t")
        assert len(var._band_dim_values_map["level"]) == 2
        assert_allclose(var.read_array().ravel(), [0.5, 2.5])

    def test_an_empty_bin_is_dropped(self):
        """A bin no coordinate falls in is left out; only non-empty bins appear."""
        binned = _cube([100.0, 150.0, 900.0]).groupby_bins(
            "level", [0, 200, 400, 1000], "mean"
        )
        # bins (0,200], (200,400], (400,1000]; the middle bin is empty and dropped.
        assert binned.get_variable("t")._band_dim_values_map["level"] == [0.0, 400.0]

    def test_right_false_closes_intervals_on_the_left(self):
        """`right=False` puts a coordinate equal to an inner edge in the upper bin."""
        cube = _cube([0.0, 500.0, 900.0])
        left = cube.groupby_bins(
            "level", [0, 500, 1000], "mean", right=False, include_lowest=True
        )
        # [0,500) holds 0.0; [500,1000) holds 500.0 and 900.0 -> two bins, second has two members.
        assert left.get_variable("t")._band_dim_values_map["level"] == [0.0, 500.0]
        assert_allclose(left.get_variable("t").read_array().ravel(), [0.0, 1.5])


class TestGroupbyBinsOnContainersAndVariables:
    """It works on a container (every gridded variable) and on a single variable."""

    def test_a_container_reduces_every_variable(self):
        """A two-variable container bins both variables onto the same binned axis."""
        cube = _cube([100.0, 300.0, 600.0, 900.0], cols=2)
        cube.set_variable("u", cube.get_variable("t") * 10.0)
        binned = cube.groupby_bins("level", [0, 500, 1000], "mean")
        assert sorted(binned.variable_names) == ["t", "u"]
        assert binned.get_variable("u")._band_dim_values_map["level"] == [0.0, 500.0]

    def test_a_variable_reduces_itself(self):
        """Called on a variable, it returns a variable holding the same cells."""
        cube = _cube([100.0, 300.0, 600.0, 900.0])
        from_container = cube.groupby_bins(
            "level", [0, 500, 1000], "mean"
        ).get_variable("t")
        from_variable = cube.get_variable("t").groupby_bins(
            "level", [0, 500, 1000], "mean"
        )
        assert_allclose(from_variable.read_array(), from_container.read_array())


class TestGroupbyBinsRefusals:
    """Inputs that cannot be binned into a valid grid are refused with a clear message."""

    def test_a_coordinate_outside_every_bin_is_refused(self):
        """A coordinate in no bin has no group, so it raises rather than dropping data."""
        cube = _cube([50.0, 300.0, 2000.0])
        with pytest.raises(ValueError, match="outside every bin"):
            cube.groupby_bins("level", [0, 500, 1000], "mean")

    def test_a_spatial_dimension_is_refused(self):
        """Binning a spatial axis would destroy the grid, so it raises."""
        with pytest.raises(ValueError):
            _cube([100.0, 300.0]).groupby_bins("x", [0, 1], "mean")

    def test_non_increasing_edges_are_refused(self):
        """Explicit edges must be strictly increasing."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _cube([100.0, 300.0]).groupby_bins("level", [0, 1000, 500], "mean")

    def test_fewer_than_two_edges_is_refused(self):
        """A single edge forms no bin."""
        with pytest.raises(ValueError, match="at least two bin edges"):
            _cube([100.0, 300.0]).groupby_bins("level", [500], "mean")

    def test_a_text_axis_is_refused(self):
        """A non-numeric coordinate cannot be cut into value intervals."""
        cube = NetCDF.from_array(
            np.arange(2.0).reshape(2, 1, 1),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="scenario", values=["rcp45", "rcp85"]),
        )
        with pytest.raises(ValueError, match="numeric"):
            cube.groupby_bins("scenario", [0, 1], "mean")

    def test_a_coordinate_less_axis_is_refused(self):
        """A band dimension carrying no coordinate values cannot be cut into intervals.

        Test scenario:
            A file-loaded coordinate-less axis tracks `_band_dim_values_map[dim] is None`;
            binning it has nothing to cut, so it raises rather than fabricating labels.
        """
        var = _cube([100.0, 300.0, 600.0]).get_variable("t")
        var._band_dim_values_map["level"] = None
        with pytest.raises(ValueError, match="coordinate-less axis"):
            var.groupby_bins("level", [0, 500, 1000], "mean")

    @pytest.mark.parametrize("bins", [0, -2])
    def test_an_int_bins_below_one_is_refused(self, bins):
        """An integer bin count below one forms no bin, so it raises.

        Args:
            bins: A non-positive integer bin count that cannot form a single bin.
        """
        with pytest.raises(ValueError, match="at least one bin"):
            _cube([100.0, 300.0]).groupby_bins("level", bins, "mean")
