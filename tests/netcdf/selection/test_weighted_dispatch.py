"""Which variables a weighted container reduces, and which axes count as spatial.

A container's gridded variables need not all carry the band dimension being weighted, and a
store need not declare its spatial axes last. Both decide whether a variable is weighted,
carried over unchanged, or refused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines._weighted import _spatial_names

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
DATA = Path(__file__).resolve().parents[2] / "data" / "netcdf"
MIS_ORDERED = DATA / "cf__48v__1d17-3d21-4d10__y-asc.nc"
ERA5_T2M = DATA / "cf__5v__1d4-3d1__geog__y-desc.nc"


def _mixed_container() -> tuple[NetCDF, np.ndarray, np.ndarray]:
    """A container holding one variable over `time` and one without it.

    Returns:
        tuple: The container, the `(time, y, x)` stack of `over_time`, and the `(y, x)` grid of
        `static`.
    """
    stack = np.arange(2 * 3 * 4, dtype="float64").reshape(2, 3, 4)
    flat = np.arange(3 * 4, dtype="float64").reshape(3, 4) * 10.0
    container = NetCDF.from_array(
        stack,
        geo_ref=GEO,
        variable_name="over_time",
        dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
    )
    container.set_variable("static", Dataset.from_array(flat, geo_ref=GEO))
    return container, stack, flat


class TestWhichVariablesAreWeighted:
    """Weighting a band dimension carries a variable that does not have it, as `reduce` does."""

    def test_a_variable_without_the_dimension_is_carried(self):
        """`static` has no `time`, so it comes through unchanged while `over_time` is weighted.

        Test scenario:
            The branch that carries it used to be unreachable — its condition was
            `name in band_names or name not in var._band_dim_names`, which is always true — so
            the whole call raised for a container like this one.
        """
        container, stack, flat = _mixed_container()
        result = container.weighted(np.array([3.0, 1.0]), "time")
        carried = result.get_variable("static")
        assert_array_equal(np.asarray(carried.read_array(), dtype="float64"), flat)
        weighted = result.get_variable("over_time")
        expected = (3.0 * stack[0] + 1.0 * stack[1]) / 4.0
        assert_allclose(np.asarray(weighted.read_array(), dtype="float64"), expected)
        assert tuple(weighted._band_dim_names) == ()

    def test_the_carried_variable_keeps_the_grid(self):
        """Weighting a band dimension leaves every variable on the source grid."""
        container, _, _ = _mixed_container()
        result = container.weighted(np.array([1.0, 1.0]), "time")
        for name in ("over_time", "static"):
            variable = result.get_variable(name)
            assert (variable.rows, variable.columns) == (3, 4), name

    def test_a_dimension_no_variable_has(self):
        """A dimension none of the gridded variables carries is refused."""
        container, _, _ = _mixed_container()
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            container.weighted(np.array([1.0, 1.0]), "level")


class TestWhichAxesAreSpatial:
    """The spatial pair comes from the axes the read resolved, not from the declared order."""

    def test_a_store_that_declares_its_axes_out_of_order(self):
        """`U` declares `(time, lat, lev, lon)`, so the pair is `lat` / `lon`, not `lev` / `lon`.

        Test scenario:
            Taking the last two declared dimensions named `lev`, a band dimension, so the
            default `dims` mixed a band dimension with a spatial axis and the call was refused.
        """
        variable = NetCDF.read_file(str(MIS_ORDERED)).get_variable("U")
        assert variable._md_array_dims[1:4] == ["subset_lat_63_-1_64", "lev", "lon"]
        assert _spatial_names(variable) == ("subset_lat_63_-1_64", "lon")
        result = variable.weighted("area")
        assert (result.rows, result.columns) == (1, 1)
        assert tuple(result._band_dim_names) == ("time", "lev")

    def test_a_store_that_declares_them_last(self):
        """ERA5 declares `(valid_time, latitude, longitude)`, whose pair is the last two."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        assert _spatial_names(variable) == ("latitude", "longitude")

    def test_an_in_memory_variable(self):
        """A variable built by `from_array` names its axes `y` / `x`."""
        container, _, _ = _mixed_container()
        assert _spatial_names(container.get_variable("static")) == ("y", "x")

    def test_the_refusal_lists_each_name_once(self):
        """An unknown dimension is refused with the variable's dimensions, none of them twice."""
        variable = NetCDF.read_file(str(MIS_ORDERED)).get_variable("U")
        with pytest.raises(ValueError, match="cannot weight over 'depth'") as error:
            variable.weighted("area", "depth")
        listed = str(error.value)
        for name in ("time", "lev", "lon", "subset_lat_63_-1_64"):
            assert listed.count(f"'{name}'") == 1, listed
