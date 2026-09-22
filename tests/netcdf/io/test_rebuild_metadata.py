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
