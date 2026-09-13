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
WRF_PATH = "tests/data/netcdf/none__17v__1d1-2d5-3d6-4d5__stag-str.nc"

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

    def test_each_coards_dimension_resolves_to_its_declared_length(self, coards_nc):
        """Every dimension of *this* fixture resolves to an array of the declared length.

        Test scenario:
            Not a general guarantee — a dimension with no indexing variable correctly
            answers ``None`` (the WRF fixture has eight of them). This one carries a
            coordinate variable for each of its four dimensions.
        """
        for name, size in coards_nc.dimension_sizes.items():
            values = coards_nc.get_dimension_values(name)
            assert values is not None, f"{name!r} declared but has no coordinates"
            assert len(values) == size, (
                f"{name!r}: {len(values)} values for size {size}"
            )

    def test_spatial_coordinates_come_back_in_storage_order(self):
        """A 1-D spatial axis reports what the file stores, not the raster's row order.

        Test scenario:
            The y-ascending fixture stores latitudes south-to-north while pyramids
            presents the raster north-up, so the accessor and `read_array`'s rows run
            opposite ways. That is the documented contract — it matches `sel` and
            `to_xarray().coords` — and this pins it against a silent flip.
        """
        nc = NetCDF.read_file(CF_PATH)
        lats = nc.get_dimension_values("lat")
        assert lats[0] < lats[-1], f"y-asc file should report ascending, got {lats}"
        gt = nc.get_variable("temperature").geotransform
        rows = NetCDF.get_y_lat_dimension_array(
            gt[3], abs(gt[5]), nc.get_variable("temperature").rows
        )
        assert_array_equal(
            np.asarray(rows),
            lats[::-1],
            err_msg="the raster row centres should be the reverse of the stored axis here",
        )

    def test_a_y_descending_file_reports_descending(self, coards_nc):
        """On a north-to-south file storage order and raster order already agree."""
        lats = coards_nc.get_dimension_values("lat")
        assert lats[0] > lats[-1], f"y-desc file should report descending, got {lats}"

    def test_unknown_dimension_is_none(self, coards_nc):
        """A name the dataset does not declare answers ``None`` rather than raising."""
        assert coards_nc.get_dimension_values("depth") is None

    def test_time_axis_reports_stored_offsets(self, coards_nc):
        """The time axis comes back as the raw CF offsets the file stores.

        Test scenario:
            Compared against the underlying read rather than against `get_time_values`,
            which now delegates here — that comparison could not fail.
        """
        assert_array_equal(
            coards_nc.get_dimension_values("time"),
            coards_nc._read_variable("time"),
            err_msg="the accessor should hand back the stored coordinate variable",
        )
        assert coards_nc.get_time_variable("time") is None, (
            "this fixture's units do not parse — the raw offsets are all there is"
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


class TestStringTypedCoordinateVariable:
    """A character coordinate axis (WRF ``Times``) reads rather than raising."""

    def test_container_reads_the_real_timestamps(self):
        """The container reports the stored strings, matching the xarray bridge.

        Test scenario:
            `ReadAsArray` refuses a character MDArray in the SWIG bindings, so the
            accessor used to surface a raw `RuntimeError: String buffer data type not
            supported`. The list-based read that `to_xarray` uses handles it.
        """
        nc = NetCDF.read_file(WRF_PATH)
        values = nc.get_dimension_values("Time")
        assert list(values) == [
            "2000-01-24_12:00:00",
            "2000-01-24_13:00:00",
            "2000-01-24_14:00:00",
        ], f"got {values!r}"

    def test_a_subset_agrees_with_its_container(self):
        """A subset reports the same timestamps its container does, not integer indices.

        Test scenario:
            The subset build hit the same SWIG refusal and recorded `[0, 1, 2]`, so one
            file answered two different vocabularies for one dimension depending on which
            object was asked — and `sel(Time=<timestamp>)` found nothing.
        """
        nc = NetCDF.read_file(WRF_PATH)
        var = nc.get_variable("IVGTYP")
        stamps = [
            "2000-01-24_12:00:00",
            "2000-01-24_13:00:00",
            "2000-01-24_14:00:00",
        ]
        assert list(var.get_dimension_values("Time")) == stamps
        assert list(nc.get_dimension_values("Time")) == stamps

    def test_a_subset_selects_by_that_timestamp(self):
        """The coordinates a subset reports are the ones `sel` matches against."""
        nc = NetCDF.read_file(WRF_PATH)
        var = nc.get_variable("IVGTYP")
        pinned = var.sel(Time="2000-01-24_13:00:00")
        assert list(pinned.get_dimension_values("Time")) == ["2000-01-24_13:00:00"]


class TestUnreadableIndexingVariable:
    """When neither read works, the subset falls back to integer indices."""

    def test_the_placeholder_is_the_last_resort(self, monkeypatch):
        """A coordinate variable no read can handle still yields a usable axis.

        Test scenario:
            `ReadAsArray` already refuses this WRF `Times` axis; with the list-based
            read stubbed out to fail too, the build must not raise — it records
            `[0, 1, ..., size - 1]`, which is what `sel` then matches against.
        """

        def refuse(_md_array):
            raise RuntimeError("no read path available")

        monkeypatch.setattr(NetCDF, "_md_array_to_numpy", staticmethod(refuse))
        nc = NetCDF.read_file(WRF_PATH)
        var = nc.get_variable("IVGTYP")
        assert list(var.get_dimension_values("Time")) == [0, 1, 2]
