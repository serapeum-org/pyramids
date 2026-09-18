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
from pyramids.netcdf.engines._weighted import _removed_dimensions, _spatial_names

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
DATA = Path(__file__).resolve().parents[2] / "data" / "netcdf"
MIS_ORDERED = DATA / "cf__48v__1d17-3d21-4d10__y-asc.nc"
SPATIAL_BOUNDS = DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"
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


class TestARasterOfWeights:
    """Weights given as a raster: a container, a variable, or a plain `Dataset`."""

    @staticmethod
    def _parts() -> tuple[NetCDF, np.ndarray, np.ndarray]:
        """A 2x2 variable and a grid of weights on the same grid.

        Returns:
            tuple: The variable, its values, and the weights.
        """
        geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[1.0, 2.0], [3.0, 1.0]])
        variable = NetCDF.from_array(
            values, geo_ref=geo, variable_name="v"
        ).get_variable("v")
        return variable, values, weights

    def test_a_dataset_of_weights_is_read(self):
        """A `Dataset` holds one grid and no variables, so it is read as the weights.

        Test scenario:
            The dispatch accepted anything with `read_array` and then asked it for the band
            dimensions only a `NetCDF` has, so a GeoTIFF of weights raised
            `AttributeError: 'Dataset' object has no attribute '_band_dim_names'` from inside
            the engine.
        """
        variable, values, weights = self._parts()
        raster = Dataset.from_array(
            weights,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        )
        result = variable.weighted(raster)
        expected = float(np.sum(values * weights) / np.sum(weights))
        assert float(np.asarray(result.read_array()).ravel()[0]) == pytest.approx(
            expected
        )

    def test_a_dataset_on_another_grid_is_refused(self):
        """The grid check runs for a `Dataset` as it does for any other raster."""
        variable, _, _ = self._parts()
        raster = Dataset.from_array(
            np.ones((2, 2)),
            geo_ref=GeoReference(geo=(100.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        )
        with pytest.raises(ValueError, match="same grid"):
            variable.weighted(raster)

    def test_a_dataset_matches_the_same_weights_as_an_array(self):
        """Reading the weights from a raster answers what passing the array answers."""
        variable, _, weights = self._parts()
        raster = Dataset.from_array(
            weights,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        )
        assert_allclose(
            np.asarray(variable.weighted(raster).read_array(), dtype="float64"),
            np.asarray(variable.weighted(weights).read_array(), dtype="float64"),
        )


class TestRemovedDimensions:
    """Which dimensions a weighting leaves no full-length axis of, under the store's own names.

    A band dimension weighted over is gone from the result; a spatial axis stays but comes back
    one cell long. Both are answered, because an auxiliary variable spanning either cannot be
    carried at its old length. The spatial ones are answered under the name the **store**
    declares, since that is the name an auxiliary variable's dimensions carry.
    """

    @staticmethod
    def _parts() -> tuple[NetCDF, NetCDF, list[str]]:
        """A y-ascending CF store, its `tos` variable, and the store's dimension names for it.

        The read flips the rows through a view, which renames the row dimension, so the
        variable knows it as `subset_lat_169_-1_170` while the store still declares `lat`.

        Returns:
            tuple: The container, the variable, and the store's dimension names, positionally.
        """
        container = NetCDF.read_file(str(SPATIAL_BOUNDS))
        group = container._working_group()
        variable = container._require_raster_variable("tos")
        return container, variable, container._variable_dim_names(group, "tos")

    def test_the_view_renames_the_row_dimension(self):
        """The premise: the variable and the store name the same axis differently."""
        _, variable, declared = self._parts()
        assert _spatial_names(variable) == ("subset_lat_169_-1_170", "lon"), (
            _spatial_names(variable)
        )
        assert declared == ["time", "lat", "lon"], declared

    def test_a_spatial_axis_is_answered_under_the_stores_name(self):
        """`y` reaches the row axis, and comes back as `lat` — what `lat_bnds` spans.

        Test scenario:
            Answering the view's name (`subset_lat_169_-1_170`) matched no auxiliary variable's
            dimensions, so `lat_bnds` was carried onto a result whose `lat` is one cell long.
        """
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("y", "x"), declared) == ["lat", "lon"], (
            _removed_dimensions(variable, ("y", "x"), declared)
        )

    def test_one_axis_removes_only_that_axis(self):
        """Weighting `x` alone leaves the row axis at its full length, so only `lon` goes."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("x",), declared) == ["lon"], declared

    def test_a_band_dimension_is_answered_as_itself(self):
        """A band dimension is named the same either way, so it passes straight through."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("time",), declared) == ["time"]

    def test_the_order_is_the_order_they_were_named(self):
        """The answer follows `names`, so a message listing them reads as the caller wrote it."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("x", "y"), declared) == ["lon", "lat"], (
            _removed_dimensions(variable, ("x", "y"), declared)
        )

    def test_a_store_naming_too_few_dimensions_falls_back(self):
        """With no store name at that position the variable's own names are answered instead.

        Test scenario:
            The spatial axes are looked up by position in the store's declared names, so a
            store declaring fewer of them than the read resolved would otherwise index past the
            end of the list.
        """
        _, variable, _ = self._parts()
        assert _removed_dimensions(variable, ("y", "x"), ["only-one"]) == [
            "subset_lat_169_-1_170",
            "lon",
        ], _removed_dimensions(variable, ("y", "x"), ["only-one"])

    def test_a_name_that_is_neither_removes_nothing(self):
        """A name matching no band dimension and no spatial axis contributes nothing."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("depth",), declared) == []

    def test_nothing_named_removes_nothing(self):
        """No dimensions asked for, none answered."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, (), declared) == []

    def test_an_unknown_name_does_not_hide_the_ones_beside_it(self):
        """The loop keeps going, so the names it does know are still answered."""
        _, variable, declared = self._parts()
        assert _removed_dimensions(variable, ("depth", "y"), declared) == ["lat"], (
            _removed_dimensions(variable, ("depth", "y"), declared)
        )
