"""Decoding the CF time coordinate of a UGRID mesh (issue #1381).

The reader collects data variables by their ``location`` attribute, and a CF ``time``
coordinate carries none — so the time axis used to read as a bare dimension and
``time_values`` fell back to positional indices even for a file that states its stamps. These
tests pin the decode on read, the coordinate on write, and the positional fallback that must
survive for a file carrying no CF time units.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.dataset import (
    TIME_CALENDAR_ATTR,
    TIME_UNITS_ATTR,
    _time_coordinate_name,
)

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _temporal_mesh(stamps=None, units=None, calendar="standard"):
    """A two-face mesh with a 3-step variable, optionally carrying CF time metadata."""
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"depth": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        data_locations={"depth": "face"},
    )
    ds["depth"].dimensions = ("time", "mesh2d_nFaces")
    if stamps is not None:
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: units,
            TIME_CALENDAR_ATTR: calendar,
        }
    return ds


class TestTimeCoordinateName:
    """Which dimension the decoder treats as the time axis."""

    def test_taken_from_the_variable_dimensions(self):
        """The name comes from the variable's own dimensions at its time_index."""
        ds = _temporal_mesh()
        assert _time_coordinate_name(ds.data_vars) == "time"

    def test_none_when_nothing_is_temporal(self):
        """A static-only dataset has no time coordinate to look for."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"bed": np.array([1.0, 2.0])},
            data_locations={"bed": "face"},
        )
        assert _time_coordinate_name(ds.data_vars) is None

    def test_honours_a_non_default_axis_name(self):
        """A file naming its axis something other than `time` is followed, not assumed."""
        ds = _temporal_mesh()
        ds["depth"].dimensions = ("t", "mesh2d_nFaces")
        assert _time_coordinate_name(ds.data_vars) == "t"


class TestRoundTrip:
    """Write then read must preserve the stamps, units and calendar."""

    def test_stamps_survive_a_round_trip(self, tmp_path):
        """Regression (#1381): the writer emitted the time dimension but no time array, so a
        round trip degraded real dates to positional indices."""
        stamps = np.array(
            ["2020-03-01T00:00:00", "2020-03-01T01:00:00", "2020-03-01T02:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "seconds since 2020-03-01 00:00:00")
        path = tmp_path / "stamped.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert [str(v) for v in back.time_values] == [str(v) for v in stamps]

    def test_units_and_calendar_survive(self, tmp_path):
        """The decoded axis carries its CF units and calendar, which convert_calendar needs."""
        stamps = np.array(
            ["2020-01-01T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "days since 2020-01-01 00:00:00")
        path = tmp_path / "units.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        attrs = back["depth"].attributes
        assert "days since" in str(attrs.get(TIME_UNITS_ATTR))
        assert attrs.get(TIME_CALENDAR_ATTR) == "standard"

    def test_a_non_standard_calendar_round_trips(self, tmp_path):
        """A model calendar decodes through the existing cftime path, not just `standard`."""
        import cftime

        stamps = [
            cftime.DatetimeNoLeap(2020, 1, 1),
            cftime.DatetimeNoLeap(2020, 1, 2),
            cftime.DatetimeNoLeap(2020, 1, 3),
        ]
        ds = _temporal_mesh(stamps, "days since 2020-01-01 00:00:00", calendar="noleap")
        path = tmp_path / "noleap.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back["depth"].attributes.get(TIME_CALENDAR_ATTR) == "noleap"
        assert len(back.time_values) == 3


class TestPositionalFallback:
    """A file with no CF time units must read exactly as it did before."""

    def test_no_time_metadata_reads_positionally(self, tmp_path):
        """Decoding is additive: it must never become a new way for read_file to fail."""
        ds = _temporal_mesh()  # no stamps, no units
        path = tmp_path / "plain.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.time_values == [0, 1, 2]

    def test_a_static_mesh_is_untouched(self, tmp_path):
        """No time dimension at all: nothing to decode, nothing to write."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"bed": np.array([1.0, 2.0])},
            data_locations={"bed": "face"},
        )
        path = tmp_path / "static.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.time_values is None


class TestResampleUnblocked:
    """The wider benefit: resample needs date-like coordinates, which decoding supplies."""

    def test_resample_works_once_stamps_are_decoded(self, tmp_path):
        """resample bins by a pandas offset alias, so it was unusable while the axis was
        positional. Locks in that decoding makes it work."""
        stamps = np.array(
            ["2020-01-01T00:00:00", "2020-01-01T00:30:00", "2020-01-01T01:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "seconds since 2020-01-01 00:00:00")
        path = tmp_path / "resample.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        hourly = back.resample("1h", how="mean")
        assert hourly["depth"].n_time_steps == 2


class TestFittedDimensionNames:
    """`_fitted_dimensions` is what stops a fit result being read as a time series.

    `MeshVariable.time_index` reads `dimensions` when present and otherwise falls back to "the
    leading axis of a >1-D array" - so a `(deg+1, n_face)` fit with no dimension names reported
    `has_time == True` and would have been reduced over its coefficient axis as though that
    were time. These pin both naming paths.
    """

    def test_existing_names_have_only_the_time_axis_replaced(self):
        """A variable read from a file carries real dimension names, which must survive."""
        from pyramids.netcdf.ugrid.dataset import _fitted_dimensions

        ds = _temporal_mesh()
        var = ds["depth"]
        assert var.dimensions == ("time", "mesh2d_nFaces")
        assert _fitted_dimensions(var, 2, 0, "coefficient") == (
            "coefficient",
            "mesh2d_nFaces",
        )

    def test_names_are_synthesised_when_the_variable_carries_none(self):
        """An array-built variable has no names, so they are invented - and none of them may
        match the time-name patterns, or the fallback bug comes straight back."""
        from pyramids.netcdf.ugrid.dataset import _fitted_dimensions

        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.zeros((3, 2))},
            data_locations={"d": "face"},
        )
        names = _fitted_dimensions(ds["d"], 2, 0, "coefficient")
        assert names == ("coefficient", "face")
        assert not any(n.lower().startswith("time") for n in names)

    def test_a_middle_axis_gets_a_placeholder_name(self):
        """A layered fit is 3-D, so the axis between the fit axis and the elements needs a
        name of its own - and it must not look like a time axis either."""
        from pyramids.netcdf.ugrid.dataset import _fitted_dimensions

        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.zeros((3, 2))},
            data_locations={"d": "face"},
        )
        names = _fitted_dimensions(ds["d"], 3, 0, "coefficient")
        assert names == ("coefficient", "dim1", "face")
        assert not any(n.lower().startswith("time") for n in names)

    def test_a_fit_result_never_reports_a_time_axis(self):
        """The property the naming exists to guarantee, asserted end to end."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])},
            data_locations={"d": "face"},
        )
        assert ds["d"].has_time, "the input is a time series"
        assert not ds.polyfit(1)["d"].has_time, "the fit is not"


class TestDecodingIsBestEffort:
    """A broken time coordinate must not become a new way for `read_file` to fail.

    The decode is additive: a file that never decoded still reads exactly as before. These
    make the decoder fail the way a corrupt or exotic time array would, then read a real file
    through it and assert the read succeeds with the positional fallback.
    """

    # Datetime stamps, because that is what a read produces - handing `time_values` raw
    # numeric offsets alongside CF units is a state the reader never creates, and the writer
    # then has nothing to encode.
    _STAMPS = np.array(
        ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
    )

    @staticmethod
    def _written(tmp_path):
        """A real file whose time variable declares CF units, so the decode path runs."""
        ds = _temporal_mesh(
            stamps=list(TestDecodingIsBestEffort._STAMPS),
            units="days since 2001-01-01",
        )
        path = tmp_path / "temporal.nc"
        ds.to_file(str(path))
        return path

    def test_the_decode_path_really_runs_on_this_file(self, tmp_path):
        """Guards the tests below: if the file did not decode, they would prove nothing.

        Asserted against the exact dates, not merely "not an int" - a float offset would
        satisfy that and the guard would pass while nothing had decoded.
        """
        read = UgridDataset.read_file(str(self._written(tmp_path)))
        stamps = read.time_values
        assert [str(value)[:10] for value in stamps] == [
            "2001-01-01",
            "2001-01-02",
            "2001-01-03",
        ], f"the file must decode to real dates; got {stamps!r}"

    @pytest.mark.parametrize(
        "error",
        [ValueError("bad units"), TypeError("bad type"), OverflowError("too big")],
        ids=["value", "type", "overflow"],
    )
    def test_a_decoder_failure_degrades_instead_of_raising(
        self, error, tmp_path, monkeypatch
    ):
        """The read must still succeed, with the stamps it had before decoding.

        Args:
            error: The exception the decoder raises; each is one the guard catches.
            tmp_path: pytest fixture for the written file.
            monkeypatch: pytest fixture used to make the decoder fail.
        """
        path = self._written(tmp_path)
        from pyramids.netcdf.ugrid import dataset as ugrid_dataset

        def exploding(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(ugrid_dataset, "decode_cf_time", exploding)
        read = UgridDataset.read_file(str(path))
        assert read is not None, "a failed decode must not fail the read"
        stamps = read.time_values
        assert stamps is not None
        assert len(stamps) == 3, f"the axis keeps its length; got {stamps!r}"
        assert all(isinstance(value, (int, np.integer)) for value in stamps), (
            f"expected the positional fallback, got {stamps!r}"
        )


class TestWritingIsBestEffortToo:
    """Writing a mesh must never fail over its time metadata.

    The writer re-encodes the decoded stamps back to CF offsets. A stamp it cannot re-encode
    leaves the axis bare - the mesh and its data still write - rather than failing the whole
    `to_file`, which is the same best-effort stance the reader takes.
    """

    def test_a_stamp_that_cannot_be_re_encoded_leaves_the_axis_bare(self, tmp_path):
        """The file must still be written, and still be readable."""
        ds = _temporal_mesh(stamps=["not-a-date"] * 3, units="days since 2001-01-01")
        path = tmp_path / "unencodable.nc"
        ds.to_file(str(path))
        assert path.stat().st_size > 0, "the mesh still wrote"
        read = UgridDataset.read_file(str(path))
        assert read["depth"].n_time_steps == 3, "the data survived"
        assert read.time_values == [0, 1, 2], (
            f"with no coordinate written, the read falls back to positions; got "
            f"{read.time_values!r}"
        )

    def test_the_data_values_are_unaffected(self, tmp_path):
        """A time-metadata problem must not touch the measurements."""
        ds = _temporal_mesh(stamps=["not-a-date"] * 3, units="days since 2001-01-01")
        path = tmp_path / "unencodable2.nc"
        ds.to_file(str(path))
        read = UgridDataset.read_file(str(path))
        assert np.allclose(read["depth"].data, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


class TestSilentLossIsWarned:
    """Losing the time coordinate must say so (review M6).

    This function exists because "a read -> write -> read round trip silently degraded real
    dates to positional indices". Its own swallowed failures re-create exactly that
    degradation, and the write still reports success - so a user who gets positional indices
    back from their own file had no way to find out why.
    """

    def test_an_unencodable_stamp_warns(self, tmp_path):
        ds = _temporal_mesh(stamps=["not-a-date"] * 3, units="days since 2001-01-01")
        with pytest.warns(UserWarning, match="could not write the time coordinate"):
            ds.to_file(str(tmp_path / "unencodable.nc"))

    def test_the_warning_names_the_units_it_could_not_use(self, tmp_path):
        """So the user can see which attribute is the problem."""
        ds = _temporal_mesh(stamps=["not-a-date"] * 3, units="days since 1999-05-05")
        with pytest.warns(UserWarning, match="days since 1999-05-05"):
            ds.to_file(str(tmp_path / "named.nc"))

    def test_the_write_still_succeeds(self, tmp_path):
        """Warning, not raising: the mesh and its data must still be written."""
        path = tmp_path / "still_written.nc"
        ds = _temporal_mesh(stamps=["not-a-date"] * 3, units="days since 2001-01-01")
        with pytest.warns(UserWarning):
            ds.to_file(str(path))
        assert path.stat().st_size > 0
        assert UgridDataset.read_file(str(path))["depth"].n_time_steps == 3

    def test_a_successful_write_warns_about_nothing(self, tmp_path):
        """The ordinary path must stay quiet."""
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        ds = _temporal_mesh(stamps=list(stamps), units="days since 2001-01-01")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ds.to_file(str(tmp_path / "clean.nc"))


class TestWriterTimeDimensionName:
    """The writer's time dimension is always "time", and that is now written down (review L3).

    `_write_time_coordinate` resolved the axis name and then ignored it, addressing `dims` and
    the created array by the literal "time" - logic that read as if it were load bearing. It is
    not: the topology writer and `write_ugrid_data_variable` both address the dimension by that
    name, so making only the coordinate follow a different name breaks the data write outright
    ("Illegal numpy array rank 2"). The name is used as the None gate it really is, with the
    invariant stated beside it.
    """

    def test_a_non_default_axis_name_still_writes_its_stamps(self):
        """The stamps survive - under the writer's "time" axis, not the source's name."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"depth": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            data_locations={"depth": "face"},
        )
        ds["depth"].dimensions = ("nmesh2d_time", "mesh2d_nFaces")
        assert _time_coordinate_name(ds._data_variables) == "nmesh2d_time", (
            "the reader follows the variable's own axis name"
        )

    def test_the_default_axis_name_round_trips_as_before(self, tmp_path):
        """The ordinary case, unchanged."""
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        ds = _temporal_mesh(stamps=list(stamps), units="days since 2001-01-01")
        path = tmp_path / "default_axis.nc"
        ds.to_file(str(path))
        read = UgridDataset.read_file(str(path))
        assert [str(value)[:10] for value in read.time_values] == [
            "2001-01-01",
            "2001-01-02",
            "2001-01-03",
        ]


class TestWriteFailuresAlsoWarn:
    """All three swallowed failures must warn, not just the re-encode (review M6).

    Only the re-encode path is reachable with real data. The other two need GDAL itself to
    refuse, so they are driven against `_write_time_coordinate` directly with a stand-in root
    group - patching GDAL globally is not an option, because the data-variable writer uses the
    same calls and would fail before this function ran. Without these the warning was proven
    at one point of three, and the other two are exactly the ones an unfamiliar driver hits.
    """

    @staticmethod
    def _variables():
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        return _temporal_mesh(
            stamps=list(stamps), units="days since 2001-01-01"
        )._data_variables

    def test_a_refused_array_creation_warns(self, monkeypatch):
        """The `CreateMDArray` collision path.

        Args:
            monkeypatch: used to report no existing array, so creation is attempted.
        """
        from pyramids.netcdf.ugrid import dataset as ugrid_dataset

        monkeypatch.setattr(ugrid_dataset, "open_mdarray", lambda *_a, **_k: None)

        class RefusingGroup:
            """A root group whose array creation collides, as netCDF's can."""

            def CreateMDArray(self, *_args):  # noqa: N802 - matches GDAL's method name
                raise RuntimeError("String match to name in use")

        group = RefusingGroup()
        dims = {"time": object()}
        variables = self._variables()
        with pytest.warns(UserWarning, match="could not create"):
            ugrid_dataset._write_time_coordinate(group, dims, variables)

    def test_a_refused_write_warns(self, monkeypatch):
        """The `array.Write` path, where the driver takes the array then refuses the data.

        Args:
            monkeypatch: used to hand back an array whose Write refuses.
        """
        from pyramids.netcdf.ugrid import dataset as ugrid_dataset

        class RefusingArray:
            """Stands in for a coordinate array whose driver refuses the write."""

            def Write(self, _values):  # noqa: N802 - matches GDAL's method name
                raise RuntimeError("driver refused the write")

        monkeypatch.setattr(
            ugrid_dataset, "open_mdarray", lambda *_a, **_k: RefusingArray()
        )
        monkeypatch.setattr(
            ugrid_dataset, "write_attributes_to_md_array", lambda *_a, **_k: None
        )
        group = object()
        dims = {"time": object()}
        variables = self._variables()
        with pytest.warns(UserWarning, match="refused the time coordinate write"):
            ugrid_dataset._write_time_coordinate(group, dims, variables)


class TestRoundTripWithoutHandSetDimensions:
    """The round trip must work through the public API alone (round 2 review H4).

    Every other round-trip test here hand-sets `.dimensions`, and no public constructor does -
    `from_arrays` leaves it empty. So the guarantee #1381 exists to provide was only ever
    tested on a hand-patched object, while the ordinary path dropped the stamps silently.
    """

    @staticmethod
    def _from_arrays_with_stamps():
        """A dataset built the way a caller builds one: no `.dimensions` assignment."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"depth": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            data_locations={"depth": "face"},
        )
        stamps = np.array(
            ["2020-03-01", "2020-03-02", "2020-03-03"], dtype="datetime64[ns]"
        )
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: "days since 2020-03-01",
            TIME_CALENDAR_ATTR: "standard",
        }
        return ds

    def test_the_stamps_survive_a_round_trip(self, tmp_path):
        """Was `[0, 1, 2]`, with nothing written and nothing warned."""
        path = tmp_path / "from_arrays.nc"
        self._from_arrays_with_stamps().to_file(str(path))
        read = UgridDataset.read_file(str(path))
        assert [str(value)[:10] for value in read.time_values] == [
            "2020-03-01",
            "2020-03-02",
            "2020-03-03",
        ], f"the stamps did not survive: {read.time_values!r}"

    def test_the_name_resolves_without_dimension_names(self):
        """`_time_coordinate_name` needed `axis < len(dimensions)`, which an array-built
        variable can never satisfy - though `has_time` and `time_index` both say there is an
        axis."""
        ds = self._from_arrays_with_stamps()
        assert ds["depth"].dimensions == (), "from_arrays leaves this empty"
        assert ds["depth"].has_time
        assert ds["depth"].time_index == 0
        assert _time_coordinate_name(ds._data_variables) is not None

    def test_a_dataset_that_cannot_write_its_coordinate_says_so(self):
        """The silent case must disappear: all three existing warnings sat AFTER the early
        return, so the most likely way to lose the stamps warned about nothing."""
        from pyramids.netcdf.ugrid import dataset as ugrid_dataset

        variables = self._from_arrays_with_stamps()._data_variables
        group = object()
        # No "time" dimension registered, so the coordinate cannot be placed - which is
        # exactly the state that used to return silently.
        with pytest.warns(UserWarning, match="no 'time' dimension was registered"):
            ugrid_dataset._write_time_coordinate(group, {}, variables)
