"""Tests for `isel()` — selection by position along one or more band dimensions.

`isel` is the positional twin of `sel`: it names a band by its index instead of by the
coordinate value stored at that index. Three claims carry the feature and each has its own
class here — a position and the coordinate stored at it reach the same band, several
dimensions compose in one call, and an axis the store gives **no** coordinates for is served
at all (the case `sel` can only refuse).

Two fixtures are used:

- `tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc` — synthetic, `(time=4, pressure_level=3,
  lat=5, lon=6)`, every cell encoding `t*1000 + l*100 + y*10 + x`, so an expected plane is
  written out by hand rather than re-derived from the code under test.
- `tests/data/netcdf/none__17v__1d1-2d5-3d6-4d5__stag-str.nc` — a WRF store whose `T` is
  `(Time=3, bottom_top=27, south_north=60, west_east=73)`. `bottom_top` has 27 levels and no
  coordinate variable at all.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

CF_PATH = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
WRF_PATH = "tests/data/netcdf/none__17v__1d1-2d5-3d6-4d5__stag-str.nc"

NT, NL, NY, NX = 4, 3, 5, 6
TIME_VALUES = [0.0, 6.0, 12.0, 18.0]
LEVEL_VALUES = [1000.0, 850.0, 500.0]

WRF_TIME_SIZE, WRF_LEVEL_SIZE = 3, 27
WRF_TIME_VALUES = [
    "2000-01-24_12:00:00",
    "2000-01-24_13:00:00",
    "2000-01-24_14:00:00",
]


def _plane(time_index: int, level_index: int) -> np.ndarray:
    """The north-up ``(lat, lon)`` plane the synthetic fixture holds at one ``(t, l)``.

    The generator wrote ``t*1000 + l*100 + y*10 + x`` with ``y`` ascending and pyramids reads
    north-up, so raster row 0 is ``y = NY - 1``. Written out from the encoding rather than read
    back off the file, so a test comparing against it fails if the wrong band is selected.

    Args:
        time_index: Position along ``time``.
        level_index: Position along ``pressure_level``.

    Returns:
        np.ndarray: The ``(NY, NX)`` plane, in raster (north-up) order.
    """
    y_values = np.arange(NY - 1, -1, -1).reshape(NY, 1)
    x_values = np.arange(NX).reshape(1, NX)
    return (
        time_index * 1000.0 + level_index * 100.0 + y_values * 10.0 + x_values
    ).astype("float64")


def _cube(*time_indices: int) -> np.ndarray:
    """Every level of each named time step, stacked in the band order GDAL flattens to.

    Args:
        *time_indices: Positions along ``time``, in the order the result should hold them.

    Returns:
        np.ndarray: A ``(len(time_indices) * NL, NY, NX)`` stack.
    """
    return np.stack(
        [_plane(t, level) for t in time_indices for level in range(NL)]
    ).astype("float64")


def _one_band_dim_variable():
    """An in-memory single-band-dim variable carrying a real no-data sentinel.

    The on-disk fixtures declare no fill value, so the sentinel-collapsing behaviour
    `_subset_along_dim` owns cannot be observed on them. Values ``0..59`` never collide with
    the ``-9999.0`` sentinel.

    Returns:
        NetCDF: The ``temp`` variable subset, with ``time`` coordinates ``[0, 6, 12, 18, 24]``.
    """
    array = np.arange(60, dtype=np.float64).reshape(5, 3, 4)
    container = NetCDF.from_array(
        arr=array,
        geo_ref=GeoReference(geo=(0.0, 1.0, 0, 3.0, 0, -1.0)),
        variable_name="temp",
        no_data_value=-9999.0,
        dims=ExtraDimensions(name="time", values=[0, 6, 12, 18, 24]),
    )
    return container.get_variable("temp")


@pytest.fixture(scope="module")
def cube():
    """The synthetic 4-D ``temperature`` variable, with band dims ``(time, pressure_level)``."""
    return NetCDF.read_file(CF_PATH).get_variable("temperature")


@pytest.fixture(scope="module")
def wrf_t():
    """WRF ``T`` — ``(Time, bottom_top, …)``, where ``bottom_top`` carries no coordinates."""
    return NetCDF.read_file(WRF_PATH).get_variable("T")


@pytest.fixture(scope="module")
def wrf_t2():
    """WRF ``T2`` — a single-band-dim variable, for the ``_band_dim_sizes`` tuple arithmetic."""
    return NetCDF.read_file(WRF_PATH).get_variable("T2")


class TestIselMatchesSelByPosition:
    """Index ``i`` and the coordinate stored at ``i`` must name the same band."""

    @pytest.mark.parametrize("position", range(NT))
    def test_a_position_reads_what_its_coordinate_reads(self, cube, position):
        """``isel(time=i)`` returns the plane the encoding puts at ``i``, and so does ``sel``.

        Args:
            cube: The synthetic 4-D variable.
            position: The index along ``time`` to select.

        Test scenario:
            Two assertions that can fail independently. The first pins the absolute answer
            against the hand-written encoding, so a systematic off-by-one is caught; the second
            pins `isel` and `sel` to each other, which is the claim the feature rests on and
            which two identically-wrong implementations would still satisfy on its own.
        """
        by_index = cube.isel(time=position).read_array()
        by_value = cube.sel(time=TIME_VALUES[position]).read_array()

        assert_array_equal(
            by_index,
            _cube(position),
            err_msg=f"isel(time={position}) must read the encoded planes of that time step",
        )
        assert_array_equal(
            by_index,
            by_value,
            err_msg=f"isel(time={position}) and sel(time={TIME_VALUES[position]}) disagree",
        )

    @pytest.mark.parametrize("position", range(NT))
    def test_the_two_paths_narrow_the_metadata_identically(self, cube, position):
        """The band-dim sizes and the coordinate map come out the same either way.

        Args:
            cube: The synthetic 4-D variable.
            position: The index along ``time`` to select.

        Test scenario:
            `sel` and `isel` share `_subset_along_dim`, so the metadata is meant to be built
            once. Comparing the two results' metadata is what would catch the day one of them
            grows its own copy of that step.
        """
        by_index = cube.isel(time=position)
        by_value = cube.sel(time=TIME_VALUES[position])

        assert by_index._band_dim_sizes == by_value._band_dim_sizes == (1, NL), (
            f"sizes differ: isel {by_index._band_dim_sizes} vs sel {by_value._band_dim_sizes}"
        )
        assert by_index._band_dim_values_map == by_value._band_dim_values_map, (
            f"coordinate maps differ: {by_index._band_dim_values_map} vs "
            f"{by_value._band_dim_values_map}"
        )
        assert by_index._band_dim_values_map["time"] == [TIME_VALUES[position]], (
            f"the kept time coordinate must be {TIME_VALUES[position]}, got "
            f"{by_index._band_dim_values_map['time']}"
        )

    def test_the_grid_survives_the_cut(self, cube):
        """The geotransform and the CRS are carried onto the result.

        Test scenario:
            `_subset_along_dim` rebuilds the raster through `Dataset.from_array`, so the
            georeference is re-stated rather than inherited. A result that lost it would still
            hold the right numbers and plot in the wrong place.
        """
        result = cube.isel(time=2, pressure_level=1)

        assert result.geotransform == (-10.5, 1.0, 0, 44.5, 0, -1.0), (
            f"the grid must be unchanged, got {result.geotransform}"
        )
        assert result.epsg == 4326, f"the CRS must be unchanged, got {result.epsg}"

    def test_the_no_data_sentinel_is_collapsed_to_a_scalar(self):
        """A per-band sentinel tuple becomes one scalar on the result, as it does for ``sel``.

        Test scenario:
            `_subset_along_dim` feeds `Dataset.from_array`, which takes a single value; the
            on-disk fixtures declare no fill value, so this needs the in-memory variable. Both
            callers go through the same collapse, and both are asserted so neither can drift.
        """
        variable = _one_band_dim_variable()

        by_index = variable.isel(time=1)
        by_value = variable.sel(time=6)

        assert by_index.no_data_value == (-9999.0,), (
            f"isel must keep the sentinel, got {by_index.no_data_value}"
        )
        assert by_index.no_data_value == by_value.no_data_value, (
            f"isel {by_index.no_data_value} vs sel {by_value.no_data_value}"
        )


class TestIselSeveralDimensions:
    """Two band dims in one call, and the orders they may be reached in."""

    def test_two_dimensions_reach_a_single_plane(self, cube):
        """``isel(time=1, pressure_level=2)`` collapses the cube to one hand-computed plane.

        Test scenario:
            Both axes pinned is the smallest result there is, and the one where a wrong
            flattening stride shows up as a plane from the wrong level rather than as a shape
            error.
        """
        result = cube.isel(time=1, pressure_level=2)

        assert_array_equal(
            result.read_array(),
            _plane(1, 2),
            err_msg="pinning t=1, l=2 must read the plane encoded at (1, 2)",
        )
        assert result._band_dim_sizes == (1, 1), (
            f"both axes must be pinned, got {result._band_dim_sizes}"
        )
        assert result._band_dim_values_map == {
            "time": [6.0],
            "pressure_level": [500.0],
        }, f"unexpected coordinate map: {result._band_dim_values_map}"

    @pytest.mark.parametrize("time_index", range(NT))
    @pytest.mark.parametrize("level_index", range(NL))
    def test_keyword_order_does_not_change_the_result(
        self, cube, time_index, level_index
    ):
        """Naming the levels first gives what naming the times first gives.

        Args:
            cube: The synthetic 4-D variable.
            time_index: The index along ``time``.
            level_index: The index along ``pressure_level``.

        Test scenario:
            `isel` narrows one dimension at a time in keyword order, and each cut renumbers the
            band grid. Order independence is therefore a property of the arithmetic, not an
            obvious one — it is asserted across every combination rather than for one pair.
        """
        forwards = cube.isel(time=time_index, pressure_level=level_index)
        backwards = cube.isel(pressure_level=level_index, time=time_index)

        assert_array_equal(
            forwards.read_array(),
            backwards.read_array(),
            err_msg=f"keyword order changed the data at (t={time_index}, l={level_index})",
        )
        assert forwards._band_dim_values_map == backwards._band_dim_values_map, (
            f"keyword order changed the coordinates: {forwards._band_dim_values_map} vs "
            f"{backwards._band_dim_values_map}"
        )

    def test_chaining_equals_one_multi_dimension_call(self, cube):
        """``isel(a).isel(b)`` and ``isel(a, b)`` are the same cut.

        Test scenario:
            The chained form re-enters `isel` against an already-narrowed variable, whose
            `_band_dim_sizes` the first cut rewrote. If the sizes were not updated the second
            cut would stride over the original grid and read the wrong bands.
        """
        chained = cube.isel(time=3).isel(pressure_level=0)
        together = cube.isel(time=3, pressure_level=0)

        assert_array_equal(
            chained.read_array(),
            together.read_array(),
            err_msg="the chained form must equal the single multi-dimension call",
        )
        assert_array_equal(
            together.read_array(),
            _plane(3, 0),
            err_msg="both must read the (3, 0) plane",
        )
        assert chained._band_dim_values_map == together._band_dim_values_map, (
            f"{chained._band_dim_values_map} vs {together._band_dim_values_map}"
        )

    def test_isel_and_sel_compose_in_either_order(self, cube):
        """A positional cut and a label cut mix, whichever comes first.

        Test scenario:
            The two produce the same kind of variable, so one must be able to follow the other.
            Reaching (t=1, l=500) by `isel` then `sel` and by `sel` then `isel` is the check
            that the metadata each leaves behind is what the other expects.
        """
        index_first = cube.isel(time=1).sel(pressure_level=500)
        label_first = cube.sel(pressure_level=500).isel(time=1)

        assert_array_equal(
            index_first.read_array(),
            _plane(1, 2),
            err_msg="isel then sel must land on the (1, 2) plane",
        )
        assert_array_equal(
            label_first.read_array(),
            index_first.read_array(),
            err_msg="sel then isel must land on the same plane",
        )

    def test_the_receiver_is_left_alone(self, cube):
        """Selecting returns a new variable and does not narrow the one it was called on.

        Test scenario:
            The module-scoped fixture is shared by every test here, so a cut that mutated its
            receiver would make this suite order-dependent — and would silently shrink a user's
            variable in exactly the same way.
        """
        cube.isel(time=0, pressure_level=0)

        assert cube._band_dim_sizes == (NT, NL), (
            f"the receiver's sizes changed to {cube._band_dim_sizes}"
        )
        assert cube._band_dim_values_map["time"] == TIME_VALUES, (
            f"the receiver's time coordinates changed to "
            f"{cube._band_dim_values_map['time']}"
        )


class TestIselSelectorForms:
    """An ``int``, a list, a tuple, or a ``slice`` of positions."""

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            (0, [0]),
            (3, [3]),
            (-1, [3]),
            (-4, [0]),
            ([1, 2], [1, 2]),
            ((0, 3), [0, 3]),
            (slice(1, 3), [1, 2]),
            (slice(None), [0, 1, 2, 3]),
            (slice(2, None), [2, 3]),
            (slice(None, None, 2), [0, 2]),
            (slice(-2, None), [2, 3]),
        ],
        ids=[
            "first",
            "last",
            "negative-one",
            "negative-size",
            "list",
            "tuple",
            "slice",
            "full-slice",
            "open-ended-slice",
            "strided-slice",
            "negative-bounded-slice",
        ],
    )
    def test_each_form_selects_the_positions_it_names(self, cube, selector, expected):
        """Every accepted selector shape resolves to the positions written out beside it.

        Args:
            cube: The synthetic 4-D variable.
            selector: The selector handed to ``isel(time=...)``.
            expected: The positions along ``time`` it must resolve to.

        Test scenario:
            The expected positions are spelled out rather than computed, so a change to the
            negative-index or `slice.indices` handling has to be restated here to pass. The
            data is asserted as well as the coordinates, because a correct coordinate list on
            wrongly-read bands is exactly the failure that matters.
        """
        result = cube.isel(time=selector)

        assert result._band_dim_values_map["time"] == [
            TIME_VALUES[index] for index in expected
        ], (
            f"{selector!r} kept {result._band_dim_values_map['time']}, expected {expected}"
        )
        assert result._band_dim_sizes == (len(expected), NL), (
            f"{selector!r} left sizes {result._band_dim_sizes}"
        )
        assert_array_equal(
            result.read_array(),
            _cube(*expected),
            err_msg=f"{selector!r} read the wrong bands",
        )

    def test_several_positions_on_the_inner_axis_keep_the_declared_layout(self, cube):
        """``isel(pressure_level=[1, 2])`` must lay its 8 bands out as ``_band_dim_sizes`` says.

        Test scenario:
            `_band_dim_sizes` is the only thing that says how a multi-band result's flat band
            list maps back onto its dimensions, so reshaping the read by it is the contract.
            It held when the outer axis was narrowed and broke when the inner one was:
            `_map_dim_to_band_indices` grouped the bands by the pinned index first, so band 1
            was (t=1, l=850) where the declared layout puts (t=0, l=500) — six of eight planes
            carrying real data attached to the wrong (time, level). Fixed by emitting outer
            blocks before pinned indices; this is the regression test.
        """
        result = cube.isel(pressure_level=[1, 2])
        assert result._band_dim_sizes == (NT, 2), (
            f"expected ({NT}, 2), got {result._band_dim_sizes}"
        )

        reshaped = np.asarray(result.read_array()).reshape(NT, 2, NY, NX)
        expected = np.stack(
            [np.stack([_plane(t, level) for level in (1, 2)]) for t in range(NT)]
        )

        assert_array_equal(
            reshaped,
            expected,
            err_msg="the flat band list must reshape by _band_dim_sizes onto (time, level)",
        )

    def test_a_list_is_deduplicated_and_taken_in_axis_order(self, cube):
        """``isel(time=[2, 0, 2])`` yields positions 0 and 2, in that order.

        Test scenario:
            This is a deliberate divergence from xarray, whose fancy indexing would answer in
            request order and would repeat position 2. Pinning it means a later change to
            request-order semantics cannot land silently.
        """
        result = cube.isel(time=[2, 0, 2])

        assert result._band_dim_values_map["time"] == [0.0, 12.0], (
            f"expected axis order [0.0, 12.0], got {result._band_dim_values_map['time']}"
        )
        assert_array_equal(
            result.read_array(),
            _cube(0, 2),
            err_msg="the duplicate must be dropped and the pair read in axis order",
        )


class TestIselOnAnAxisWithNoCoordinates:
    """The case `isel` exists for: an axis the store declares without coordinate values."""

    def test_the_axis_has_no_coordinates_to_select_by(self, wrf_t):
        """``bottom_top`` is a tracked band dim of 27 levels whose coordinate map holds ``None``.

        Test scenario:
            Everything else in this class rests on the axis genuinely having no coordinates. If
            the store ever started surfacing them the remaining tests would still pass while no
            longer testing what they claim to.
        """
        assert wrf_t._band_dim_names == ("Time", "bottom_top"), (
            f"unexpected band dims {wrf_t._band_dim_names!r}"
        )
        assert wrf_t._band_dim_sizes == (WRF_TIME_SIZE, WRF_LEVEL_SIZE), (
            f"unexpected sizes {wrf_t._band_dim_sizes}"
        )
        assert wrf_t._band_dim_values_map["bottom_top"] is None, (
            "bottom_top must carry no coordinate values, got "
            f"{wrf_t._band_dim_values_map['bottom_top']!r}"
        )

    def test_an_integer_index_reaches_a_level_sel_cannot_name(self, wrf_t):
        """``isel(bottom_top=5)`` reads level 5 of every time step.

        Test scenario:
            GDAL flattens ``(Time=3, bottom_top=27)`` with ``bottom_top`` varying fastest, so
            level 5 lives at flat bands 5, 32 and 59. Those three numbers are written out here
            rather than recomputed, so a change to the stride arithmetic has to be restated.
        """
        result = wrf_t.isel(bottom_top=5)
        every_band = np.asarray(wrf_t.read_array())

        assert result.band_count == WRF_TIME_SIZE, (
            f"one plane per time step expected, got {result.band_count}"
        )
        assert_array_equal(
            result.read_array(),
            every_band[[5, 32, 59]],
            err_msg="level 5 must come from flat bands 5, 32 and 59",
        )

    def test_sel_refuses_the_same_axis_and_points_at_isel(self, wrf_t):
        """``sel(bottom_top=5)`` raises, naming ``isel`` as the way to reach the level.

        Test scenario:
            The refusal is the other half of the feature's justification — the message has to
            send the caller to `isel`, otherwise a coordinate-less axis reads as unreachable.
        """
        with pytest.raises(ValueError, match=r"isel\(bottom_top=<index>\)") as error:
            wrf_t.sel(bottom_top=5)

        assert "No coordinate values available" in str(error.value), (
            f"the refusal must say the axis has no coordinates, got: {error.value}"
        )

    def test_the_result_keeps_no_coordinates_for_that_axis(self, wrf_t):
        """The pinned axis stays ``None`` while the axis that has coordinates keeps its own.

        Test scenario:
            Narrowing could plausibly fabricate `[5]` as the axis' new coordinates. It must not
            — that would make the result claim a value the store never wrote — while the
            sibling `Time` axis must come through untouched.
        """
        result = wrf_t.isel(bottom_top=5)

        assert result._band_dim_values_map["bottom_top"] is None, (
            "a coordinate-less axis must not gain fabricated values, got "
            f"{result._band_dim_values_map['bottom_top']!r}"
        )
        assert result._band_dim_values_map["Time"] == WRF_TIME_VALUES, (
            f"the Time coordinates must survive, got {result._band_dim_values_map['Time']}"
        )
        assert result._band_dim_sizes == (WRF_TIME_SIZE, 1), (
            f"only bottom_top must be pinned, got {result._band_dim_sizes}"
        )

    def test_the_coordinate_bearing_axis_of_the_same_variable_still_selects(
        self, wrf_t
    ):
        """``isel(Time=1)`` keeps all 27 levels of the second time step.

        Test scenario:
            The two axes of this variable are unlike — one has coordinates, one does not — so
            cutting the one that does is what proves `isel` is not quietly special-casing the
            coordinate-less variable as a whole.
        """
        result = wrf_t.isel(Time=1)
        every_band = np.asarray(wrf_t.read_array())

        assert result._band_dim_sizes == (1, WRF_LEVEL_SIZE), (
            f"expected (1, {WRF_LEVEL_SIZE}), got {result._band_dim_sizes}"
        )
        assert_array_equal(
            result.read_array(),
            every_band[27:54],
            err_msg="time step 1 occupies flat bands 27..53",
        )

    def test_both_axes_in_one_call_reach_a_single_plane(self, wrf_t):
        """``isel(Time=0, bottom_top=26)`` is flat band 26 — the last level of the first step.

        Test scenario:
            The top of the model column on the first time step exercises the far corner of the
            stride arithmetic, where an off-by-one in the block size would wrap into the next
            time step instead of raising.
        """
        result = wrf_t.isel(Time=0, bottom_top=26)
        every_band = np.asarray(wrf_t.read_array())

        assert_array_equal(
            result.read_array(),
            every_band[26],
            err_msg="(Time=0, bottom_top=26) must be flat band 26",
        )


class TestIselOnASingleBandDimension:
    """A variable with exactly one band dim — where the stride arithmetic degenerates."""

    def test_one_index_reads_that_band(self, wrf_t2):
        """``T2`` has only ``Time``, so ``isel(Time=1)`` is flat band 1.

        Test scenario:
            With a single band dim `_map_dim_to_band_indices` reduces to the identity, and
            `_band_dim_sizes` is a one-tuple the rebuild has to reconstruct without an index
            error.
        """
        result = wrf_t2.isel(Time=1)
        every_band = np.asarray(wrf_t2.read_array())

        assert wrf_t2._band_dim_names == ("Time",), (
            f"T2 must have one band dim, got {wrf_t2._band_dim_names!r}"
        )
        assert result._band_dim_sizes == (1,), (
            f"the one-tuple must survive, got {result._band_dim_sizes}"
        )
        assert_array_equal(
            result.read_array(),
            every_band[1],
            err_msg="a single-band-dim isel is the identity on the band index",
        )

    def test_a_list_keeps_the_named_bands(self, wrf_t2):
        """``isel(Time=[0, 2])`` drops the middle step and leaves a two-band variable.

        Test scenario:
            The single-band-dim path still has to rebuild a multi-band result, which is the
            branch of `_read_selected_bands` that pre-allocates rather than reading one band.
        """
        result = wrf_t2.isel(Time=[0, 2])
        every_band = np.asarray(wrf_t2.read_array())

        assert result._band_dim_sizes == (2,), (
            f"expected (2,), got {result._band_dim_sizes}"
        )
        assert_array_equal(
            result.read_array(),
            every_band[[0, 2]],
            err_msg="the first and last steps must come through, in axis order",
        )


class TestIselErrors:
    """The refusals, and the one that has to read exactly like `sel`'s."""

    @pytest.mark.parametrize("index", [4, 9, -5, -100])
    def test_an_out_of_range_index_names_the_dimension_and_its_length(
        self, cube, index
    ):
        """An index outside ``[-4, 4)`` raises ``IndexError`` quoting the valid range.

        Args:
            cube: The synthetic 4-D variable.
            index: A position beyond either end of the 4-long ``time`` axis.

        Test scenario:
            "index 9 is out of bounds" alone would not say which of two dimensions overran, so
            the message is asserted to carry the dimension's name and its length.
        """
        with pytest.raises(IndexError) as error:
            cube.isel(time=index)

        message = str(error.value)
        assert f"index {index} is out of range" in message, (
            f"unexpected message: {message}"
        )
        assert "dimension 'time' of length 4" in message, (
            f"unexpected message: {message}"
        )
        assert "Valid indices are -4 to 3" in message, f"unexpected message: {message}"

    def test_an_out_of_range_index_in_a_list_is_caught_too(self, cube):
        """One bad entry refuses the whole list rather than being dropped.

        Test scenario:
            A list is resolved entry by entry; silently skipping the invalid one would return a
            shorter result than the caller asked for, which is the worst of both answers.
        """
        with pytest.raises(IndexError, match=r"index 7 is out of range"):
            cube.isel(time=[0, 7])

    def test_an_unknown_dimension_raises_exactly_what_sel_raises(self, cube):
        """The two report an unknown dimension with the same ``ValueError`` message.

        Test scenario:
            The plan asks for `isel` to refuse an unknown name the way `sel` already does, and
            the only way that stays true is if one helper raises for both. Comparing the two
            messages is what would catch them being written out twice and drifting.
        """
        with pytest.raises(ValueError) as by_index:
            cube.isel(depth=0)
        with pytest.raises(ValueError) as by_value:
            cube.sel(depth=0)

        # Both spellings route through `_assert_band_dimension`, so comparing them to each
        # other alone cannot fail — it would still pass if that helper returned nonsense.
        # The literal is what makes this a test: it pins the message a caller actually
        # reads, naming the dimension they asked for and the ones the variable has.
        expected = (
            "Dimension 'depth' does not match any band dimension of this variable "
            "['time', 'pressure_level']."
        )
        assert str(by_index.value) == expected
        assert str(by_value.value) == expected
        assert "does not match any band dimension" in str(by_index.value), (
            f"unexpected message: {by_index.value}"
        )
        assert "['time', 'pressure_level']" in str(by_index.value), (
            f"the message must list the dimensions there are, got: {by_index.value}"
        )

    def test_a_variable_with_no_band_dimensions_is_refused_by_both(self):
        """A root container tracks no band dims, and both selectors say so in the same way.

        Test scenario:
            The container is the object a user reaches first, so calling `isel` on it instead
            of on a variable is the likely mistake. The caller's name has to appear in the
            message for the advice to make sense.
        """
        container = NetCDF.read_file(CF_PATH)
        assert container._band_dim_names == (), (
            f"a root container must track no band dims, got {container._band_dim_names!r}"
        )

        with pytest.raises(ValueError, match=r"isel\(\) requires a variable"):
            container.isel(time=0)
        with pytest.raises(ValueError, match=r"sel\(\) requires a variable"):
            container.sel(time=0)

    def test_no_indexers_is_refused(self, cube):
        """``isel()`` names nothing to select, so it raises rather than returning the variable.

        Test scenario:
            Returning the receiver unchanged would be a defensible reading, and is the one this
            refuses: a no-argument call is a mistake, not a no-op.
        """
        with pytest.raises(ValueError, match=r"at least one keyword argument"):
            cube.isel()

    @pytest.mark.parametrize(
        "selector",
        [slice(4, 8), slice(2, 2), slice(3, 1), slice(-1, 0)],
        ids=["past-the-end", "empty-range", "reversed", "negative-start-past-stop"],
    )
    def test_a_slice_that_selects_nothing_is_refused(self, cube, selector):
        """An empty slice raises instead of building a variable with no bands.

        Args:
            cube: The synthetic 4-D variable.
            selector: A slice that resolves to no position of the 4-long ``time`` axis.

        Test scenario:
            A zero-band variable fails much later and somewhere else, so the refusal is raised
            at the point the slice is resolved and quotes the slice and the axis length.
        """
        with pytest.raises(ValueError) as error:
            cube.isel(time=selector)

        message = str(error.value)
        assert "selects no index of an axis of length 4" in message, (
            f"unexpected message: {message}"
        )
        assert repr(selector) in message, (
            f"the message must quote the slice {selector!r}, got: {message}"
        )

    @pytest.mark.parametrize(
        "selector",
        [1.0, "0", None, {0}],
        ids=["float", "string", "none", "set"],
    )
    def test_a_selector_that_is_not_a_position_is_refused(self, cube, selector):
        """A non-integer selector raises ``TypeError`` and points at ``sel``.

        Args:
            cube: The synthetic 4-D variable.
            selector: Something ``operator.index()`` refuses.

        Test scenario:
            The gate is ``operator.index()``, Python's own definition of a usable index, so
            a float, a string, ``None`` and a set are all refused while a numpy integer is
            not — see ``TestIselAcceptsAnyIntegerPython``. Booleans are refused too but
            carry their own message, since ``bool`` satisfies ``index()`` and needs an
            explicit guard; they are covered there.
        """
        with pytest.raises(TypeError) as error:
            cube.isel(time=selector)

        message = str(error.value)
        assert "needs an int, a list of ints, a tuple of ints, or a slice" in message, (
            f"unexpected message: {message}"
        )
        assert "sel(time=...)" in message, (
            f"the message must point at sel, got: {message}"
        )

    @pytest.mark.parametrize(
        "selector",
        [[1, 2.5], [0, True], [None], ["1"]],
        ids=["float-entry", "bool-entry", "none-entry", "string-entry"],
    )
    def test_a_list_entry_that_is_not_a_whole_number_is_refused(self, cube, selector):
        """A list is only accepted when every entry is a plain ``int``.

        Args:
            cube: The synthetic 4-D variable.
            selector: A list holding at least one non-integer entry.

        Test scenario:
            The list is validated before any entry is resolved, so a partly-valid list refuses
            as a whole instead of selecting the entries it could read.
        """
        # One gate for scalars and list entries now, so the message is the same for both;
        # a boolean entry gets the dedicated boolean message.
        with pytest.raises(TypeError, match=r"needs an int|does not take booleans"):
            cube.isel(time=selector)


class TestIselFacade:
    """`NetCDF.isel` has to hand back a NetCDF, not a bare raster."""

    def test_the_result_keeps_the_netcdf_api(self, cube):
        """The result is a ``NetCDF`` whose dimension coordinates are still readable.

        Test scenario:
            The facade delegates to an engine that rebuilds the raster through
            `Dataset.from_array`; returning what that builds would strip `get_dimension_values`
            and every other NetCDF-only method off the result and break chaining.
        """
        result = cube.isel(time=0)

        assert isinstance(result, NetCDF), (
            f"isel must return a NetCDF, got {type(result).__name__}"
        )
        assert_array_equal(
            result.get_dimension_values("pressure_level"),
            np.asarray(LEVEL_VALUES),
            err_msg="the untouched axis must still report its coordinates",
        )
        assert_array_equal(
            result.get_dimension_values("time"),
            np.asarray([TIME_VALUES[0]]),
            err_msg="the narrowed axis must report only the coordinate it kept",
        )


class TestIselRefusesEverySelectorThatKeepsNothing:
    """An empty selection must be refused wherever it comes from, not just from a slice."""

    @pytest.mark.parametrize(
        "selector",
        [slice(9, 9), slice(2, 1), slice(0, 0), [], ()],
        ids=[
            "empty-slice",
            "reversed-bounds",
            "zero-width",
            "empty-list",
            "empty-tuple",
        ],
    )
    def test_a_selection_of_nothing_is_refused(self, cube, selector):
        """Every form that resolves to zero positions raises, with the same message.

        Args:
            cube: The 4x3 band-dim fixture.
            selector: A selector keeping no position.

        Test scenario:
            The slice forms were already refused; the list and tuple forms fell straight
            through `sorted(set())` and built a variable with `_band_dim_sizes == (0, 3)`
            and `band_count == 0`, whose `read_array()` then died inside GDAL with
            `AttributeError: 'NoneType' object has no attribute 'GetScale'`. `sel(time=[])`
            has always refused, so the two spellings of the same request disagreed.
        """
        with pytest.raises(ValueError, match="selects no index"):
            cube.isel(time=selector)

    def test_the_refusal_names_the_dimension_and_its_length(self, cube):
        """The message has to say which axis and how long it is.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            A multi-dimension call can refuse on either axis, so "selects no index" alone
            does not say which one the caller got wrong.
        """
        with pytest.raises(ValueError, match="pressure_level") as excinfo:
            cube.isel(pressure_level=[])

        # The selector is echoed as `isel(<dim>=<selector>)`, unquoted, matching how the
        # call was written rather than the quoted `dimension 'x'` of the range message.
        assert "isel(pressure_level=[])" in str(excinfo.value)
        assert "length 3" in str(excinfo.value)


class TestANegativeStepSliceReversesTheAxis:
    """A slice keeps the order its step implies, and the planes follow it."""

    def test_the_coordinates_come_back_descending(self, cube):
        """`isel(time=slice(None, None, -1))` reverses the axis.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            Four docstrings described every selector as resolving to ascending positions,
            which a negative-step slice does not. The behaviour is the useful one and
            agrees with xarray, so the claims were corrected rather than the code — this
            pins the behaviour so a later "fix" to match the old wording has to fail here.
        """
        reversed_axis = cube.isel(time=slice(None, None, -1))

        assert reversed_axis._band_dim_values_map["time"] == [18.0, 12.0, 6.0, 0.0]
        assert reversed_axis._band_dim_sizes == (NT, NL)

    def test_each_plane_stays_attached_to_its_own_coordinate(self, cube):
        """Reversing the labels without reversing the data would be silent corruption.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            The ordering only matters if the planes move with the labels. Each of the four
            reversed planes is compared against the single-index read for the time step its
            label names, so a reversal applied to one and not the other fails here.
        """
        reversed_axis = cube.isel(time=slice(None, None, -1))
        planes = np.asarray(reversed_axis.read_array()).reshape(NT, NL, NY, NX)

        for position, coordinate in enumerate([18.0, 12.0, 6.0, 0.0]):
            source = [0.0, 6.0, 12.0, 18.0].index(coordinate)
            expected = np.asarray(cube.isel(time=source).read_array()).reshape(
                NL, NY, NX
            )
            assert_array_equal(
                planes[position],
                expected,
                err_msg=f"plane {position} is labelled {coordinate} and must hold it",
            )


class TestEveryKeywordIsCheckedBeforeAnythingIsRead:
    """A bad second keyword must not cost a read of the first cut."""

    @staticmethod
    @contextlib.contextmanager
    def _recording_reads():
        """Yield a list that collects the band count of every read performed inside."""
        import pyramids.netcdf.engines.selection as engine

        sizes: list[int] = []
        original = engine._read_selected_bands

        def spy(nc, band_indices):
            sizes.append(len(band_indices))
            return original(nc, band_indices)

        engine._read_selected_bands = spy
        try:
            yield sizes
        finally:
            engine._read_selected_bands = original

    def test_isel_reads_nothing_before_refusing_an_unknown_dimension(self, cube):
        """`isel(time=1, nope=0)` must refuse without reading the `time` cut.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            Validation used to live inside the loop, so the first keyword was resolved,
            cut and **read** -- 3 of the cube's 12 bands -- before the second was looked at
            and rejected. The caller paid for a read whose result was discarded.

            The recorder is a context manager rather than a function returning the list:
            an `assert reads == []` where `reads` is assigned *inside* `pytest.raises`
            never runs the assignment, so it passes whatever the code does. That is the
            shape this test had on its first draft, and it passed before the fix existed.
        """
        with self._recording_reads() as reads:
            with pytest.raises(ValueError, match="does not match any band dimension"):
                cube.isel(time=1, nope=0)

        assert reads == []

    def test_isel_reads_nothing_before_refusing_a_bad_index(self, cube):
        """The same holds for an out-of-range index in the second keyword.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            Position resolution is metadata-only, so it can be done for every keyword up
            front. An index past the end of the *second* axis is now found before the
            first axis is cut.
        """
        with self._recording_reads() as reads:
            with pytest.raises(IndexError, match="out of range"):
                cube.isel(time=1, pressure_level=9)

        assert reads == []

    def test_sel_reads_nothing_before_refusing_an_unknown_dimension(self, cube):
        """`sel` gets the same guarantee for a name it cannot place.

        Args:
            cube: The 4x3 band-dim fixture.
        """
        with self._recording_reads() as reads:
            with pytest.raises(ValueError, match="does not match any band dimension"):
                cube.sel(time=6, nope=0)

        assert reads == []

    def test_sel_reads_nothing_before_refusing_an_unmatched_value(self, cube):
        """A wrong *value* in the second keyword must not cost a read either.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            The first version of this fix hoisted only the dimension-name check for `sel`,
            on the stated grounds that "a preceding cut can narrow that dimension's
            coordinates" so the value had to stay in the loop. That is false: a cut copies
            the coordinate map and replaces only its own dimension's entry, so every other
            dimension's coordinates are exactly what they were. Selectors resolve to the
            same positions either way.

            The consequence was that the commit claiming to "check every keyword before
            cutting anything" did so for a typo'd name and not for a wrong value -- by far
            the more common mistake -- which still read 3 of the cube's 12 bands first.
        """
        with self._recording_reads() as reads:
            with pytest.raises(ValueError, match="No bands match"):
                cube.sel(time=6.0, pressure_level=99999)

        assert reads == []

    def test_a_valid_multi_dimension_call_still_cuts_in_order(self, cube):
        """Eager validation must not change what a good call returns.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            The reads themselves are unchanged -- still one per keyword, narrowing as it
            goes. This test exists so the change is visibly scoped to *when* validation
            happens, not to how the cut is performed.
        """
        with self._recording_reads() as reads:
            result = cube.isel(time=1, pressure_level=2)

        assert reads == [NL, 1]
        assert result.band_count == 1


class TestIselAcceptsAnyIntegerPython:
    """The index gate is `operator.index()`, matching what Python calls an index."""

    @pytest.mark.parametrize(
        "make",
        [lambda: 1, lambda: np.int64(1), lambda: np.int32(1), lambda: np.uint8(1)],
        ids=["int", "int64", "int32", "uint8"],
    )
    def test_a_numpy_integer_selects_the_same_band_as_a_python_int(self, cube, make):
        """A numpy integer is an index everywhere else in Python; it is one here too.

        Args:
            cube: The 4x3 band-dim fixture.
            make: Builds the index to pass.

        Test scenario:
            `isinstance(x, int)` is `False` for every numpy integer, so `isel` refused
            values that fall straight out of `np.argmin`, `np.where(...)[0][0]` or
            iterating an array — while `sel` accepted numpy scalars through
            `numbers.Real`. The two halves of the same API disagreed about what a number
            is. `operator.index()` is Python's own definition and admits exactly the
            integer types.
        """
        assert cube.isel(time=make())._band_dim_values_map["time"] == [6.0]

    @pytest.mark.parametrize(
        "make",
        [lambda: True, lambda: False, lambda: np.bool_(True)],
        ids=["True", "False", "np.bool_"],
    )
    def test_a_boolean_is_still_refused(self, cube, make):
        """`operator.index()` accepts `bool`, so the explicit guard has to stay.

        Args:
            cube: The 4x3 band-dim fixture.
            make: Builds the boolean to pass.

        Test scenario:
            `bool` is an `int` subclass and satisfies `operator.index()`, so widening the
            gate would silently turn `isel(time=True)` into index 1. A caller writing that
            almost certainly means a mask, which is not supported — so it stays a
            `TypeError`.
        """
        with pytest.raises(TypeError, match="does not take booleans"):
            cube.isel(time=make())

    @pytest.mark.parametrize(
        "make",
        [lambda: 1.0, lambda: np.float64(1.0), lambda: "1"],
        ids=["float", "np.float64", "str"],
    )
    def test_a_non_integer_is_still_refused(self, cube, make):
        """Widening to `operator.index()` must not let a float or a string through.

        Args:
            cube: The 4x3 band-dim fixture.
            make: Builds the value to pass.

        Test scenario:
            `operator.index()` refuses all three, which is why it is the right gate rather
            than a looser numeric test — `isel(time=1.5)` has no meaning as a position.
        """
        with pytest.raises(TypeError):
            cube.isel(time=make())

    def test_a_numpy_integer_works_inside_a_list_too(self, cube):
        """The element check is the same gate as the scalar one.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            A list of numpy integers is what `np.where(...)[0]` produces when iterated, so
            widening the scalar path and not the element path would leave the common case
            still refused.
        """
        assert cube.isel(time=[np.int64(2), np.int64(0)])._band_dim_values_map[
            "time"
        ] == [0.0, 12.0]


class TestArrayDimensionalityDecidesAcceptance:
    """`operator.index()` takes a 0-d array; anything higher is not an index."""

    def test_a_zero_dimensional_array_is_accepted(self, cube):
        """`np.array(2)` is a scalar in all but type, and `index()` takes it.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            The commit that widened the gate wrote "a numpy array is refused" in the same
            breath, which is true only from one dimension up. A 0-d array satisfies
            `operator.index()` and selects correctly, so the claim was corrected rather
            than the behaviour -- refusing it would mean special-casing a shape that
            behaves exactly like the scalar it holds.
        """
        assert cube.isel(time=np.array(2))._band_dim_values_map["time"] == [12.0]
        assert cube.isel(time=[np.array(2), np.array(0)])._band_dim_values_map[
            "time"
        ] == [0.0, 12.0]

    @pytest.mark.parametrize(
        "make",
        [lambda: np.array([2]), lambda: np.array([[2]]), lambda: np.arange(2)],
        ids=["1-d", "2-d", "arange"],
    )
    def test_an_array_with_dimensions_is_refused(self, cube, make):
        """One dimension up, it is a sequence rather than an index.

        Args:
            cube: The 4x3 band-dim fixture.
            make: Builds the array to pass.

        Test scenario:
            xarray accepts these as fancy indexing; this does not, and the Notes say so.
            Pinned from this side too, so widening the gate further has to be deliberate.
        """
        with pytest.raises(TypeError, match="needs an int"):
            cube.isel(time=make())
