"""What a rebuilt container keeps: its spatial axis names, and its CF time units.

`reduce`, `coarsen`, `rolling`, `cumsum` and `diff` build their result through
`NetCDF.from_array`, which names the spatial axes and creates the band axis from scratch.
Three defects live there:

- #1180: the source's `longitude` / `latitude` come out as `x` / `y`, in memory and on the
  written file, so a CF reader sees axes the source never had.
- #1179: the CF `(units, calendar)` the result carries in `_band_dim_time_attrs` is never
  written onto the band axis, so the calendar survives in memory and dies at `to_file`.
- #1194: `set_variable` creates `x` / `y` by name, so writing into a `longitude` /
  `latitude` store leaves it with two pairs of spatial dimensions for one grid.

The fixture is `cf__20v__1d3-3d17__y-desc.nc` — a CF store on `longitude` / `latitude`
whose `time` is `hours since 1900-01-01 00:00:0.0` on the standard calendar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

STORE = Path(__file__).parents[2] / "data" / "netcdf" / "cf__20v__1d3-3d17__y-desc.nc"
UNITS = "hours since 1900-01-01 00:00:0.0"
CALENDAR = "standard"


def _store() -> NetCDF:
    """The CF fixture, freshly opened.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.read_file(str(STORE))


def _time_attrs(container: NetCDF) -> tuple:
    """The `(units, calendar)` a container's first variable reports for `time`.

    Args:
        container: The container to ask.

    Returns:
        tuple: The pair, or `()` when the container carries none.
    """
    first = container.variable_names[0]
    return container.get_variable(first)._resolved_band_dim_time_attrs().get("time", ())


class TestTheFixture:
    """The preconditions every test below depends on."""

    def test_the_source_axes_are_named_for_their_geography(self):
        """`longitude` / `latitude`, not `x` / `y`."""
        assert sorted(_store().dimension_names) == ["latitude", "longitude", "time"]

    def test_the_source_carries_cf_time_units(self):
        """The calendar is on the store, which is what the rebuild must not lose."""
        assert _time_attrs(_store()) == (UNITS, CALENDAR)

    def test_a_plain_round_trip_keeps_both(self, tmp_path):
        """Nothing is lost when no rebuild happens — the defects are the rebuild's.

        Args:
            tmp_path: pytest's temporary directory.
        """
        out = tmp_path / "plain.nc"
        _store().to_file(str(out))
        back = NetCDF.read_file(str(out))
        assert sorted(back.dimension_names) == ["latitude", "longitude", "time"]
        assert _time_attrs(back) == (UNITS, CALENDAR)


class TestARebuildKeepsTheSpatialAxisNames:
    """#1180 — every member that rebuilds along a dimension renamed the grid."""

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("coarsen", lambda nc: nc.coarsen("time", 2)),
            ("reduce", lambda nc: nc.reduce("time", how="mean")),
            ("rolling", lambda nc: nc.rolling("time", 2)),
            ("cumsum", lambda nc: nc.cumsum("time")),
            ("diff", lambda nc: nc.diff("time")),
        ],
    )
    def test_the_axes_keep_their_names(self, member: str, call):
        """The result's spatial dimensions are the source's.

        Args:
            member: The member under test.
            call: How to call it.
        """
        names = list(call(_store()).dimension_names)
        assert "latitude" in names and "longitude" in names, (
            f"{member} renamed the grid: {names}"
        )
        assert "x" not in names and "y" not in names, (
            f"{member} left x / y behind: {names}"
        )

    def test_the_written_file_keeps_them_too(self, tmp_path):
        """The rename reached `to_file`, so the file declared axes the source never had.

        Args:
            tmp_path: pytest's temporary directory.
        """
        out = tmp_path / "coarse.nc"
        _store().coarsen("time", 2).to_file(str(out))
        names = sorted(NetCDF.read_file(str(out)).dimension_names)
        assert names == ["latitude", "longitude", "time"], names

    def test_the_written_file_keeps_its_georeference(self, tmp_path):
        """Naming the axes must not cost the grid — the gap that let a regression through.

        Test scenario:
            `NetCDF.lon` / `NetCDF.lat` knew only `lon` / `x` and `lat` / `y`, so a store
            carrying `latitude` / `longitude` found no coordinate array and
            `_compute_geotransform` fell back to GDAL's index-space placeholder: the
            reopened file reported `(0.0, 1.0, 0, 512.0, 0, -1.0)` and `xy(0, 0)` of
            `(0.5, 511.5)` instead of the source's grid. Asserting the names alone did not
            notice.

        Args:
            tmp_path: pytest's temporary directory.
        """
        store = _store()
        source = store.get_variable(store.variable_names[0]).geotransform
        out = tmp_path / "coarse.nc"
        store.coarsen("time", 2).to_file(str(out))
        back = NetCDF.read_file(str(out))
        assert tuple(back.geotransform) == tuple(source), (
            f"the written file lost its georeference: {tuple(back.geotransform)}"
        )
        assert back.xy(0, 0) == (0.0, 90.0), back.xy(0, 0)

    @pytest.mark.parametrize(
        "names",
        [None, ("latitude", "longitude"), ("lat", "lon"), ("y", "x")],
        ids=["default", "cf-long", "cf-short", "explicit-y-x"],
    )
    def test_every_naming_round_trips_the_same_grid(self, names, tmp_path):
        """Whatever the axes are called, the grid that comes back is the one written.

        Args:
            names: The `(row, column)` names under test, or `None` for the default.
            tmp_path: pytest's temporary directory.
        """
        geo = (10.0, 2.0, 0.0, 50.0, 0.0, -2.0)
        extra = {} if names is None else {"spatial_names": names}
        built = NetCDF.from_array(
            np.arange(12.0).reshape(3, 4),
            geo_ref=GeoReference(geo=geo, epsg=4326),
            variable_name="t",
            **extra,
        )
        out = tmp_path / f"{'-'.join(names) if names else 'default'}.nc"
        built.to_file(str(out))
        assert tuple(NetCDF.read_file(str(out)).geotransform) == geo

    def test_an_in_memory_build_still_gets_y_and_x(self):
        """A variable with no store to inherit from keeps today's names."""
        built = NetCDF.from_array(
            np.arange(4.0).reshape(2, 2),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
        )
        assert sorted(built.dimension_names) == ["x", "y"]


class TestARebuildKeepsTheCfTimeUnits:
    """#1179 — the calendar survived in memory and died at the write."""

    def test_the_result_carries_them_in_memory(self):
        """Unchanged behaviour, and the precondition for the write."""
        assert _time_attrs(_store().coarsen("time", 2)) == (UNITS, CALENDAR)

    def test_they_survive_to_file(self, tmp_path):
        """The written file declares the calendar the source had.

        Args:
            tmp_path: pytest's temporary directory.
        """
        out = tmp_path / "coarse.nc"
        _store().coarsen("time", 2).to_file(str(out))
        assert _time_attrs(NetCDF.read_file(str(out))) == (UNITS, CALENDAR)

    def test_a_frequency_reduce_works_on_the_written_file(self, tmp_path):
        """The symptom the issue was filed for.

        Test scenario:
            `reduce(groupby="1D")` works on the in-memory result and raised
            `ValueError: Cannot group dimension 'time' by frequency '1D': no decodable
            time` on the same result read back from disk.

        Args:
            tmp_path: pytest's temporary directory.
        """
        out = tmp_path / "coarse.nc"
        coarse = _store().coarsen("time", 2)
        coarse.to_file(str(out))
        back = NetCDF.read_file(str(out))
        assert back.reduce("time", how="mean", groupby="1D").variable_names

    def test_the_stamps_are_still_raw_offsets(self, tmp_path):
        """Carrying the units must not change the values themselves.

        Args:
            tmp_path: pytest's temporary directory.
        """
        out = tmp_path / "coarse.nc"
        coarse = _store().coarsen("time", 2)
        before = list(
            coarse.get_variable(coarse.variable_names[0])._band_dim_values_map["time"]
        )
        coarse.to_file(str(out))
        back = NetCDF.read_file(str(out))
        after = list(
            back.get_variable(back.variable_names[0])._band_dim_values_map["time"]
        )
        assert [float(one) for one in after] == [float(one) for one in before]


class TestSetVariableReusesTheStoresAxes:
    """#1194 — writing into a store added a second pair of spatial dimensions."""

    def test_no_second_pair_is_created(self):
        """The store keeps three dimensions, not five."""
        store = _store()
        variable = store.get_variable(store.variable_names[0])
        plane = np.asarray(variable.read_array())[0]
        store.set_variable(
            "added",
            Dataset.from_array(
                plane,
                geo_ref=GeoReference(geo=variable.geotransform, epsg=variable.epsg),
                no_data_value=variable.no_data_value[0],
            ),
        )
        assert sorted(store.dimension_names) == ["latitude", "longitude", "time"], (
            f"set_variable added a second spatial pair: {store.dimension_names}"
        )

    def test_the_written_variable_uses_them(self):
        """The new variable is declared against the store's own axes."""
        store = _store()
        variable = store.get_variable(store.variable_names[0])
        plane = np.asarray(variable.read_array())[0]
        store.set_variable(
            "added",
            Dataset.from_array(
                plane,
                geo_ref=GeoReference(geo=variable.geotransform, epsg=variable.epsg),
                no_data_value=variable.no_data_value[0],
            ),
        )
        names = list(store.get_variable("added").dimension_names or [])
        assert "latitude" in names and "longitude" in names, names

    def test_a_different_grid_still_gets_its_own_axes(self):
        """Reuse is by what the axis holds, so a different grid must not borrow one."""
        store = _store()
        variable = store.get_variable(store.variable_names[0])
        geo = variable.geotransform
        coarser = Dataset.from_array(
            np.zeros((variable.rows // 2, variable.columns // 2)),
            geo_ref=GeoReference(
                geo=(geo[0], geo[1] * 2, geo[2], geo[3], geo[4], geo[5] * 2),
                epsg=variable.epsg,
            ),
            no_data_value=variable.no_data_value[0],
        )
        store.set_variable("coarser", coarser)
        added = list(store.get_variable("coarser").dimension_names or [])
        assert len(added) == 2, added
        assert added != ["latitude", "longitude"], (
            "a half-resolution grid must not reuse the store's own axes"
        )


class TestWhatTheRebuildCarries:
    """`_carried_axis_metadata` decides what a rebuilt store is told about its axes.

    The members reach it with a source variable; these call it directly, because the
    branches below are the ones a store variable never takes — no source at all, a source
    whose axes cannot be resolved, and a dimension that declares units but no calendar.
    """

    def test_no_source_carries_nothing(self):
        """A rebuild with nothing to inherit from keeps today's naming and writes no CF."""
        assert NetCDF._carried_axis_metadata(None) == (None, None)

    def test_a_store_variable_carries_both(self):
        """The ordinary case: the source's axis names and its CF time attributes."""
        store = _store()
        names, attrs = NetCDF._carried_axis_metadata(
            store.get_variable(store.variable_names[0])
        )
        assert sorted(names) == ["latitude", "longitude"], names
        assert attrs == {"time": {"units": UNITS, "calendar": CALENDAR}}, attrs

    def test_an_unresolvable_source_carries_no_names(self):
        """A source that cannot answer about its axes must not break the rebuild.

        Test scenario:
            `_public_spatial_names` reads the parent's dimension list positionally. An
            object that raises on the way through leaves the names unresolved, and the
            rebuild falls back to `y` / `x` rather than failing.
        """

        class _Unresolvable:
            """A source whose axes cannot be read."""

            _band_dim_names = ()

            @property
            def dimension_names(self):
                """Raise, as a half-built object would.

                Raises:
                    AttributeError: Always.
                """
                raise AttributeError("no dimensions here")

            def _resolved_band_dim_time_attrs(self):
                """No CF attributes either.

                Returns:
                    dict: Empty.
                """
                return {}

        assert NetCDF._carried_axis_metadata(_Unresolvable()) == (None, None)

    @pytest.mark.parametrize(
        ("pair", "expected"),
        [
            ((UNITS, CALENDAR), {"units": UNITS, "calendar": CALENDAR}),
            ((UNITS, None), {"units": UNITS}),
            ((None, CALENDAR), {"calendar": CALENDAR}),
        ],
        ids=["both", "units-only", "calendar-only"],
    )
    def test_only_what_the_axis_declares_is_written(self, pair: tuple, expected: dict):
        """A half-declared axis writes the half it has, and an empty one writes nothing.

        Args:
            pair: The `(units, calendar)` the source reports.
            expected: What should be written onto the axis.
        """

        class _Source:
            """A source reporting one band dimension with the given CF pair."""

            _band_dim_names = ("time",)
            dimension_names = ["time", "latitude", "longitude"]
            _parent_nc = None
            _md_spatial_dims = None

            def _resolved_band_dim_time_attrs(self):
                """The CF pair under test.

                Returns:
                    dict: One entry for `time`.
                """
                return {"time": pair}

        _, attrs = NetCDF._carried_axis_metadata(_Source())
        assert attrs == {"time": expected}, attrs

    def test_an_axis_that_declares_neither_is_left_out(self):
        """No CF pair means no attributes, not an empty mapping."""

        class _Bare:
            """A source whose band dimension declares nothing."""

            _band_dim_names = ("time",)
            dimension_names = ["time", "latitude", "longitude"]
            _parent_nc = None
            _md_spatial_dims = None

            def _resolved_band_dim_time_attrs(self):
                """No units, no calendar.

                Returns:
                    dict: One entry holding an empty pair.
                """
                return {"time": (None, None)}

        assert NetCDF._carried_axis_metadata(_Bare())[1] is None


class TestResolvingASpatialDimension:
    """`_spatial_dimension` finds the store's own axis, or creates one.

    The container is built in memory and writable — the fixture on disk is read-only, so
    the "create one" branch cannot run against it. Building it with `spatial_names` also
    exercises the new parameter on the public `from_array`.
    """

    @staticmethod
    def _container() -> NetCDF:
        """A writable container whose axes are named for their geography.

        Returns:
            NetCDF: A 4x4 container on `latitude` / `longitude`.
        """
        return NetCDF.from_array(
            np.arange(16.0).reshape(4, 4),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
            spatial_names=("latitude", "longitude"),
        )

    @staticmethod
    def _resolve(container: NetCDF, preferred: str, values, dim_type):
        """Ask the resolver for an axis.

        Args:
            container: The container to resolve in.
            preferred: The name to create under.
            values: The coordinate values wanted.
            dim_type: The GDAL horizontal axis type.

        Returns:
            The resolved dimension.
        """
        return NetCDF._spatial_dimension(
            container._raster.GetRootGroup(),
            preferred,
            np.asarray(values, dtype="float64"),
            gdal.ExtendedDataType.Create(gdal.GDT_Float64),
            dim_type,
        )

    def test_the_parameter_names_the_axes(self):
        """`spatial_names` is what the store ends up declaring."""
        assert sorted(self._container().dimension_names) == ["latitude", "longitude"]

    def test_an_axis_holding_the_same_values_is_reused(self):
        """The store's own `latitude` answers a request for its own coordinates."""
        container = self._container()
        wanted = container.get_dimension_values("latitude")
        resolved = self._resolve(container, "y", wanted, gdal.DIM_TYPE_HORIZONTAL_Y)
        assert resolved.GetName() == "latitude", resolved.GetName()

    def test_a_different_grid_creates_its_own_axis(self):
        """Half the rows is a different axis, whatever it is called."""
        container = self._container()
        wanted = list(container.get_dimension_values("latitude"))[::2]
        resolved = self._resolve(container, "y", wanted, gdal.DIM_TYPE_HORIZONTAL_Y)
        assert resolved.GetName() != "latitude", "a coarser axis reused the store's own"
        assert resolved.GetSize() == len(wanted), resolved.GetSize()

    def test_the_other_axis_is_not_borrowed(self):
        """An axis of the wrong role is never reused, even when the values would match.

        Test scenario:
            Asked for a horizontal *Y* axis holding `longitude`'s values, the resolver must
            not hand back `longitude` — the role is part of the identity, so a square grid
            cannot make one axis stand in for the other.
        """
        container = self._container()
        wanted = container.get_dimension_values("longitude")
        resolved = self._resolve(container, "y", wanted, gdal.DIM_TYPE_HORIZONTAL_Y)
        assert resolved.GetName() != "longitude", "the X axis was borrowed for Y"
