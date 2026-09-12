"""Tests for ``NetCDF.get_dimension_values`` — the coordinate accessor (issue #1125).

``time`` had ``get_time_values`` / ``get_time_variable`` and the spatial axes come off
the geotransform, but a non-spatial, non-time axis (``level``, ``depth``, ``member``)
had no native accessor: its values were reachable only through the optional
``to_xarray`` bridge or by provoking ``sel``'s "Available values" error.

Fixtures:
- ``tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc`` — ``(time=12, level=4, lat, lon)``
  with levels ``[1000, 925, 850, 700]``.
- ``tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc`` — synthetic 4-D CF cube.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

COARDS_PATH = "tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc"
CF_PATH = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"

LEVELS = [1000.0, 925.0, 850.0, 700.0]


@pytest.fixture(scope="module")
def coards_nc() -> NetCDF:
    """The COARDS root container, which declares a 4-level vertical axis."""
    return NetCDF.read_file(COARDS_PATH)


@pytest.fixture(scope="module")
def coards_var(coards_nc) -> NetCDF:
    """The ``rhum`` variable subset of that container."""
    return coards_nc.get_variable("rhum")


class TestRootContainer:
    """A root container answers for any dimension it declares."""

    def test_reads_a_non_spatial_coordinate(self, coards_nc):
        """The vertical axis' values are readable without the xarray bridge.

        Test scenario:
            ``get_dimension_values("level")`` → ``[1000., 925., 850., 700.]``.
        """
        assert_array_equal(
            coards_nc.get_dimension_values("level"),
            np.array(LEVELS),
            err_msg="level coordinates should come straight off the container",
        )

    def test_every_declared_dimension_is_reachable(self, coards_nc):
        """Each name in ``dimension_sizes`` resolves to an array of that length."""
        for name, size in coards_nc.dimension_sizes.items():
            values = coards_nc.get_dimension_values(name)
            assert values is not None, f"{name!r} declared but has no coordinates"
            assert len(values) == size, (
                f"{name!r}: {len(values)} values for size {size}"
            )

    def test_unknown_dimension_is_none(self, coards_nc):
        """A name the dataset does not declare answers ``None`` rather than raising."""
        assert coards_nc.get_dimension_values("depth") is None

    def test_time_axis_reports_stored_offsets(self, coards_nc):
        """The values are the stored ones, matching what ``sel`` matches against."""
        assert_array_equal(
            coards_nc.get_dimension_values("time"),
            coards_nc.get_time_values("time"),
            err_msg="get_time_values is the time-axis spelling of this accessor",
        )


class TestVariableSubset:
    """A variable subset answers from the band-dim coordinates it carries."""

    def test_band_dimension_is_readable(self, coards_var):
        """A subset reports its own band-dim coordinates."""
        assert_array_equal(coards_var.get_dimension_values("level"), np.array(LEVELS))

    def test_reflects_what_the_view_holds_after_sel(self, coards_var):
        """After a ``sel`` the accessor reports the values that view actually holds.

        Test scenario:
            This is how a ``method="nearest"`` caller reads back the coordinate that
            was snapped to.
        """
        pinned = coards_var.sel(level=850)
        assert_array_equal(
            pinned.get_dimension_values("level"),
            np.array([850.0]),
            err_msg="a pinned view should report only the level it kept",
        )

    def test_spatial_axis_has_no_coordinate_variable(self, coards_var):
        """A subset does not track its spatial axes, so they answer ``None``."""
        assert coards_var.get_dimension_values("lat") is None


class TestGetTimeValuesDelegation:
    """``get_time_values`` keeps its contract now that it delegates here."""

    def test_named_axis_still_reads(self):
        """The documented call still returns the raw offsets of the named axis."""
        nc = NetCDF.read_file(CF_PATH)
        assert_array_equal(nc.get_time_values("time"), np.array([0.0, 6.0, 12.0, 18.0]))

    def test_absent_axis_is_none(self, coards_nc):
        """Asking for an axis the store has no dimension for still answers ``None``."""
        assert coards_nc.get_time_values("valid_time") is None


class TestInMemoryContainer:
    """A cube built with ``from_array`` answers for the dimension it declares."""

    def test_extra_dimension_values_are_readable(self):
        """A band dim declared through ``ExtraDimensions`` is readable on both views.

        Test scenario:
            The accessor must not depend on an on-disk MDIM root group — an in-memory
            container and the variable it yields report the same coordinates.
        """
        arr = np.arange(60, dtype=np.float64).reshape(5, 3, 4)
        nc = NetCDF.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0, 3.0, 0, -1.0)),
            variable_name="temp",
            dims=ExtraDimensions(name="time", values=[0, 6, 12, 18, 24]),
        )
        expected = np.array([0, 6, 12, 18, 24])
        assert_array_equal(nc.get_dimension_values("time"), expected)
        assert_array_equal(
            nc.get_variable("temp").get_dimension_values("time"), expected
        )

    def test_unknown_name_on_a_subset_is_none(self):
        """A subset asked for a name it neither tracks nor declares answers ``None``."""
        arr = np.arange(60, dtype=np.float64).reshape(5, 3, 4)
        nc = NetCDF.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0, 3.0, 0, -1.0)),
            variable_name="temp",
            dims=ExtraDimensions(name="time", values=[0, 6, 12, 18, 24]),
        )
        assert nc.get_variable("temp").get_dimension_values("missing") is None
