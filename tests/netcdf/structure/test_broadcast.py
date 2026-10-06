"""Tests for ``NetCDF.broadcast_like`` and ``NetCDF.broadcast_equals``.

``broadcast_like`` gives a cube another cube's band layout, repeating cells along the axes
it gains, and ``broadcast_equals`` asks whether two cubes agree once broadcast against each
other. Neither aligns on coordinates: an axis the two share at two lengths is refused
rather than joined, since there is no index to join on.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from osgeo import gdal, osr

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Container

pytestmark = pytest.mark.core

GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
TIMES = [0.0, 6.0, 12.0]
LEVELS = [1000.0, 850.0, 500.0]
NY, NX = 2, 2


def _geo_ref() -> GeoReference:
    """The one grid every cube in this module sits on.

    Returns:
        GeoReference: A 1-degree WGS 84 grid.
    """
    return GeoReference(geo=GEO, epsg=4326)


def _cube(dims: list[tuple[str, list]], name: str = "v") -> NetCDF:
    """A variable over the shared grid with the given band dimensions.

    Args:
        dims: `(name, coordinate values)` per band dimension, outermost first.
        name: The variable name.

    Returns:
        NetCDF: The variable subset.
    """
    shape = tuple(len(values) for _, values in dims) + (NY, NX)
    values = np.arange(1.0, float(np.prod(shape)) + 1.0).reshape(shape)
    container = NetCDF.from_array(
        values,
        geo_ref=_geo_ref(),
        variable_name=name,
        dims=ExtraDimensions(dims=[(dim, list(stamps)) for dim, stamps in dims]),
    )
    return container.get_variable(name)


def _store_without_time_coordinate() -> gdal.Dataset:
    """A store whose `time` dimension has no coordinate array of any kind.

    Built through GDAL directly because `from_array` always writes the CF same-named
    array, and the point here is a dimension `coords` cannot stamp. The spatial
    coordinates are cell *centres*, which is what GDAL derives the geotransform from, so
    the store lands on the same grid as `_geo_ref()`.

    Returns:
        gdal.Dataset: An in-memory multidimensional store holding `v(time, y, x)`.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("")
    root = store.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    time = root.CreateDimension("time", gdal.DIM_TYPE_TEMPORAL, "", 3)
    y = root.CreateDimension("y", gdal.DIM_TYPE_HORIZONTAL_Y, "NORTH", NY)
    x = root.CreateDimension("x", gdal.DIM_TYPE_HORIZONTAL_X, "EAST", NX)
    for dim, values in ((y, [1.5, 0.5]), (x, [0.5, 1.5])):
        coordinate = root.CreateMDArray(dim.GetName(), [dim], f64)
        coordinate.Write(np.array(values))
        dim.SetIndexingVariable(coordinate)
    data = root.CreateMDArray("v", [time, y, x], f64)
    data.Write(np.arange(3.0 * NY * NX).reshape(3, NY, NX))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    data.SetSpatialRef(reference)
    return store


def _flat(fill: float = 5.0, name: str = "mask") -> NetCDF:
    """A single-band variable with no band dimensions — the mask case.

    Args:
        fill: The constant cell value.
        name: The variable name.

    Returns:
        NetCDF: The variable subset.
    """
    container = NetCDF.from_array(
        np.full((NY, NX), fill), geo_ref=_geo_ref(), variable_name=name
    )
    return container.get_variable(name)


class TestBroadcastLikeAddsDimensions:
    """A dimension the donor has and the receiver lacks is added, cells repeated."""

    def test_a_flat_mask_takes_the_cube_s_time_axis(self):
        """The mask gains `time`, with the donor's size and coordinate stamps.

        Test scenario:
            A `(y, x)` mask broadcast against a 3-step cube comes back with `time`
            of length 3, stamped with the cube's values, and every step holding the
            mask's cells.
        """
        cube = _cube([("time", TIMES)])
        mask = _flat(5.0)

        lifted = mask.broadcast_like(cube)

        assert lifted._band_dim_names == ("time",)
        assert lifted._band_dim_sizes == (3,)
        assert lifted._band_dim_values_map["time"] == TIMES
        assert lifted.band_count == 3
        assert_array_equal(
            np.asarray(lifted.read_array(squeeze=True)), np.full((3, NY, NX), 5.0)
        )

    def test_the_result_matches_the_donor_s_order_when_it_adds_everything(self):
        """With no dimensions of its own, the receiver takes the donor's order exactly.

        Test scenario:
            A flat mask against a `(time, level)` cube comes back `(time, level)`,
            12 bands, so the result can stand in for the donor's layout.
        """
        cube = _cube([("time", TIMES), ("level", LEVELS)])

        lifted = _flat().broadcast_like(cube)

        assert lifted._band_dim_names == ("time", "level")
        assert lifted._band_dim_sizes == (3, 3)
        assert lifted.band_count == 9

    def test_the_receiver_s_own_dimensions_come_first(self):
        """A dimension only the receiver has keeps its place ahead of the donor's.

        Test scenario:
            A `level`-only cube broadcast against a `time`-only cube comes back
            `(level, time)` — the receiver's own axis first, xarray's ordering.
        """
        level_only = _cube([("level", LEVELS)], name="L")
        time_only = _cube([("time", TIMES)], name="T")

        lifted = level_only.broadcast_like(time_only)

        assert lifted._band_dim_names == ("level", "time")
        assert lifted._band_dim_sizes == (3, 3)

    def test_the_cells_repeat_along_the_added_axis(self):
        """Each step of the new axis holds the source's own cells, not a reshuffle.

        Test scenario:
            A `level`-only cube gains `time`; every time step must repeat the full
            level stack unchanged.
        """
        level_only = _cube([("level", LEVELS)], name="L")
        time_only = _cube([("time", TIMES)], name="T")
        source = np.asarray(level_only.read_array(squeeze=True))

        lifted = level_only.broadcast_like(time_only)

        got = np.asarray(lifted.read_array(squeeze=True)).reshape(3, 3, NY, NX)
        for step in range(3):
            assert_array_equal(got[:, step], source)

    def test_a_donor_without_band_dimensions_changes_nothing(self):
        """Broadcasting against a flat donor is a no-op on the layout.

        Test scenario:
            A cube broadcast against a plain mask keeps its own dimensions, sizes,
            coordinates and cells.
        """
        cube = _cube([("time", TIMES)])

        lifted = cube.broadcast_like(_flat())

        assert lifted._band_dim_names == ("time",)
        assert lifted._band_dim_sizes == (3,)
        assert lifted._band_dim_values_map["time"] == TIMES
        assert_array_equal(
            np.asarray(lifted.read_array(squeeze=True)),
            np.asarray(cube.read_array(squeeze=True)),
        )


class TestBroadcastLikeStretchesLengthOneAxes:
    """A shared axis of length one is stretched to the donor's length."""

    def test_a_length_one_axis_takes_the_donor_s_length_and_stamps(self):
        """`level` of one step becomes the donor's three, with the donor's stamps.

        Test scenario:
            A `(time, level=1)` cube against a `(time, level=3)` cube keeps `time`
            as it is and stretches `level`, repeating each time step's single plane.
        """
        receiver = _cube([("time", TIMES), ("level", [1000.0])], name="w")
        donor = _cube([("time", TIMES), ("level", LEVELS)])
        source = np.asarray(receiver.read_array(squeeze=True))

        lifted = receiver.broadcast_like(donor)

        assert lifted._band_dim_sizes == (3, 3)
        assert lifted._band_dim_values_map["level"] == LEVELS
        got = np.asarray(lifted.read_array(squeeze=True)).reshape(3, 3, NY, NX)
        for level in range(3):
            assert_array_equal(got[:, level], source)

    def test_a_donor_s_length_one_axis_never_shortens_the_receiver(self):
        """The receiver's longer axis wins over the donor's single step.

        Test scenario:
            A 3-step `time` against a donor whose `time` is one step keeps 3 steps
            and the receiver's own coordinates.
        """
        receiver = _cube([("time", TIMES)])
        donor = _cube([("time", [99.0])], name="d")

        lifted = receiver.broadcast_like(donor)

        assert lifted._band_dim_sizes == (3,)
        assert lifted._band_dim_values_map["time"] == TIMES

    def test_an_equal_axis_keeps_its_own_coordinates(self):
        """A shared axis of equal length is left alone, stamps included.

        Test scenario:
            Two 3-step cubes with different `time` stamps: the receiver's survive.
        """
        receiver = _cube([("time", TIMES)])
        donor = _cube([("time", [100.0, 200.0, 300.0])], name="d")

        lifted = receiver.broadcast_like(donor)

        assert lifted._band_dim_values_map["time"] == TIMES


class TestBroadcastLikeRefusals:
    """What cannot be broadcast is refused, never joined or resampled."""

    def test_two_lengths_neither_of_them_one_are_refused(self):
        """Broadcasting has no index, so two real lengths cannot be reconciled.

        Test scenario:
            A 2-step `time` against a 3-step `time` raises, naming both lengths and
            saying no join happens.
        """
        receiver = _cube([("time", [0.0, 6.0])], name="short")
        donor = _cube([("time", TIMES)])

        with pytest.raises(ValueError, match="neither is length one"):
            receiver.broadcast_like(donor)

    def test_a_different_grid_is_refused(self):
        """Only the band axes broadcast; the spatial grid must already match.

        Test scenario:
            A donor on a coarser grid raises `AlignmentError` rather than being
            resampled onto the receiver's cells.
        """
        mask = _flat()
        coarse = NetCDF.from_array(
            np.full((1, 1), 1.0),
            geo_ref=GeoReference(geo=(0.0, 2.0, 0.0, 2.0, 0.0, -2.0), epsg=4326),
            variable_name="c",
        ).get_variable("c")

        with pytest.raises(AlignmentError, match="different"):
            mask.broadcast_like(coarse)

    def test_a_plain_raster_donor_is_refused(self):
        """A donor with no band surface has no layout to lend.

        Test scenario:
            A plain `Dataset` raises `TypeError` naming its type.
        """
        plain = Dataset.from_array(np.full((NY, NX), 1.0), geo_ref=_geo_ref())
        mask = _flat()

        with pytest.raises(TypeError, match="NetCDF cube"):
            mask.broadcast_like(plain)


class TestBroadcastLikeAgainstAContainerDonor:
    """A container can be the donor: its layout comes from the store, not a variable."""

    def test_a_container_donor_lends_its_dimensions_and_stamps(self):
        """The donor's sizes come from the store's dimensions and its stamps from the
        indexing arrays, which is a different code path from a variable donor.

        Test scenario:
            A flat mask broadcast against a 3-step *container* gains `time` of length 3
            with the container's own coordinate values.
        """
        donor = NetCDF.from_array(
            np.arange(12.0).reshape(3, NY, NX),
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=TIMES),
        )

        lifted = _flat(5.0).broadcast_like(donor)

        assert lifted._band_dim_names == ("time",)
        assert lifted._band_dim_sizes == (3,)
        assert lifted._band_dim_values_map["time"] == TIMES
        assert_array_equal(
            np.asarray(lifted.read_array(squeeze=True)), np.full((3, NY, NX), 5.0)
        )

    def test_a_container_donor_axis_without_coordinates_lends_none(self):
        """A dimension the store cannot stamp is added unlabelled, not invented.

        Test scenario:
            The donor's `time` dimension carries no coordinate array at all, so
            `coords` omits it. The broadcast result gains the axis at the right length
            with no stamps, rather than inventing positional ones.
        """
        donor = Container(_store_without_time_coordinate())

        lifted = _flat(5.0).broadcast_like(donor)

        assert lifted._band_dim_names == ("time",)
        assert lifted._band_dim_sizes == (3,)
        assert lifted._band_dim_values_map["time"] is None
        assert_array_equal(
            np.asarray(lifted.read_array(squeeze=True)), np.full((3, NY, NX), 5.0)
        )


class TestBroadcastLikeOnAContainer:
    """A container broadcasts each of its variables."""

    def test_every_variable_takes_the_donor_s_axis(self):
        """Both variables of a flat container gain the donor's `time`.

        Test scenario:
            A two-variable flat container broadcast against a 3-step cube comes back
            with both variables on `time` of length 3.
        """
        container = NetCDF.from_array(
            np.full((NY, NX), 1.0), geo_ref=_geo_ref(), variable_name="a"
        )
        container.set_variable("b", _flat(2.0, name="b"))
        donor = _cube([("time", TIMES)])

        lifted = container.broadcast_like(donor)

        assert sorted(lifted.variable_names) == ["a", "b"]
        for name in ("a", "b"):
            assert lifted.get_variable(name)._band_dim_sizes == (3,)

    def test_the_broadcast_result_survives_a_round_trip(self, tmp_path):
        """The repeats and the new axis are real, so they reach the file.

        Args:
            tmp_path: pytest temporary directory.

        Test scenario:
            A flat container broadcast onto a 3-step cube, written with `to_file` and
            reopened, keeps its `time` axis, its stamps and its repeated cells. The
            container is the round-trip subject because a lone variable written to
            NetCDF comes back as `Band1..N` rather than under its own name.
        """
        container = NetCDF.from_array(
            np.full((NY, NX), 5.0), geo_ref=_geo_ref(), variable_name="mask"
        )
        lifted = container.broadcast_like(_cube([("time", TIMES)]))
        path = tmp_path / "lifted.nc"
        lifted.to_file(str(path))

        reopened = NetCDF.read_file(str(path))

        assert list(reopened.get_dimension_values("time")) == TIMES
        variable = reopened.get_variable("mask")
        assert variable._band_dim_sizes == (3,)
        assert_array_equal(
            np.asarray(variable.read_array(squeeze=True)), np.full((3, NY, NX), 5.0)
        )

    def test_the_result_is_selectable(self):
        """Keeping the layout is the point: `sel` reaches the broadcast result.

        Test scenario:
            A broadcast mask can be selected by coordinate value on the axis it
            gained.
        """
        lifted = _flat(5.0).broadcast_like(_cube([("time", TIMES)]))

        step = lifted.sel(time=6.0)

        assert np.asarray(step.read_array(squeeze=True)).shape == (NY, NX)


class TestBroadcastEquals:
    """The weaker equality: equal once both sides are broadcast."""

    def test_a_mask_equals_the_cube_of_that_mask(self):
        """`equals` says no on the ranks, `broadcast_equals` says yes on the values.

        Test scenario:
            A constant mask against a cube whose every step holds that constant.
        """
        mask = _flat(7.0)
        cube = NetCDF.from_array(
            np.full((3, NY, NX), 7.0),
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=TIMES),
        ).get_variable("t")

        assert mask.equals(cube) is False
        assert mask.broadcast_equals(cube) is True

    def test_differing_values_are_not_equal(self):
        """Broadcasting lines the shapes up; the cells still have to agree.

        Test scenario:
            A mask of 7.0 against a cube of ascending values answers False.
        """
        assert _flat(7.0).broadcast_equals(_cube([("time", TIMES)])) is False

    def test_the_answer_is_symmetric(self):
        """Asked from either side, including across disjoint dimension orders.

        Test scenario:
            A constant `level`-only cube and a constant `time`-only cube broadcast to
            `(level, time)` and `(time, level)` respectively; the right operand is
            reordered onto the left's axes, so both directions answer True.
        """
        level_only = NetCDF.from_array(
            np.full((3, NY, NX), 7.0),
            geo_ref=_geo_ref(),
            variable_name="L",
            dims=ExtraDimensions(name="level", values=LEVELS),
        ).get_variable("L")
        time_only = NetCDF.from_array(
            np.full((3, NY, NX), 7.0),
            geo_ref=_geo_ref(),
            variable_name="T",
            dims=ExtraDimensions(name="time", values=TIMES),
        ).get_variable("T")

        assert level_only.broadcast_equals(time_only) is True
        assert time_only.broadcast_equals(level_only) is True

    def test_it_agrees_with_equals_on_a_shared_layout(self):
        """With nothing to broadcast it is exactly `equals`.

        Test scenario:
            Two identical cubes, and a cube against a different one, answer the same
            as `equals` does.
        """
        cube = _cube([("time", TIMES)])
        twin = _cube([("time", TIMES)])
        doubled = cube * 2

        assert cube.equals(twin) is True
        assert cube.broadcast_equals(twin) is True
        assert cube.equals(doubled) is False
        assert cube.broadcast_equals(doubled) is False

    def test_an_unbroadcastable_pair_answers_false(self):
        """A predicate does not raise: it answers False.

        Test scenario:
            Two `time` axes of 2 and 3 steps, which `broadcast_like` refuses, and a
            grid mismatch, both answer False.
        """
        short = _cube([("time", [0.0, 6.0])], name="short")
        coarse = NetCDF.from_array(
            np.full((1, 1), 7.0),
            geo_ref=GeoReference(geo=(0.0, 2.0, 0.0, 2.0, 0.0, -2.0), epsg=4326),
            variable_name="c",
        ).get_variable("c")

        assert short.broadcast_equals(_cube([("time", TIMES)])) is False
        assert _flat(7.0).broadcast_equals(coarse) is False

    def test_a_plain_raster_answers_false(self):
        """A donor with no band surface is not comparable, so it is False.

        Test scenario:
            A plain `Dataset` operand answers False instead of raising `TypeError`.
        """
        plain = Dataset.from_array(np.full((NY, NX), 7.0), geo_ref=_geo_ref())

        assert _flat(7.0).broadcast_equals(plain) is False
