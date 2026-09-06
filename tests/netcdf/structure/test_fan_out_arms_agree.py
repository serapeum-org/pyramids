"""`op()` and `op(path=...)` must produce the same file, not merely the same shape.

`path=` is documented as a memory strategy — the container-wide operation is
streamed slab by slab instead of being built in RAM — so any difference between
the two spellings is a defect. Review round 4 measured 22 of them; what they
cost the user was concrete:

* The eager arm gave its coordinate arrays the **data** variable's dtype, so a
  packed `Int16` ERA5 cube came back with `x = [0, 3, 5, ...]` for a 2.5-degree
  grid and a geotransform of `(-1.5, 3.0, ..., -2.0)` against the true
  `(-1.25, 2.5, ..., -2.5)`; a `UInt16` GOES granule collapsed to
  `(0.0, 0.0, ...)`, a file that can no longer be placed on the earth. An
  epoch-valued `time` axis saturated at 32767.
* The streamed arm wrote no CF attributes on `x` / `y` and lost the `units` of
  every carried auxiliary axis, so one arm's longitudes said `degrees_east` and
  the other said nothing at all.
* The streamed arm declared no `_FillValue` for a variable whose source
  declares one, so masking the streamed result kept the fill cells as data.

Round 5 closed two more:

* The eager arm wrote **every** coordinate axis `Float64`, the non-spatial ones
  included, so an `int64` nanosecond-epoch `time` handed to `from_array` came
  back 21 ns out — float64 carries 53 bits of mantissa. The float64 rule was
  written for the spatial axes, where truncation was the whole problem; an
  integer axis now keeps its own dtype, which also converges the last dtype
  difference between the arms.
* The streamed arm materialised an index coordinate array (`np.arange(size)`)
  for any auxiliary dimension with no indexing variable, so its file gained a
  `bnds` on the CF bounds fixture and a `band`, `number_of_image_bounds` and
  `number_of_time_bounds` on the GOES one — axes the source does not have and
  the eager arm does not write.

What still differs, measured after these fixes and deliberately not changed:
the streamed file keeps an extra per-variable `nodata` attribute beside the
real `_FillValue` — that is the writer's own round-trip channel (#1061), and it
is what carries the fill in the first place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"

# Packed Int16 ERA5: 17 variables on a 2.5-degree grid, each with
# `_FillValue = -32767`. The dtype-sharing bug is visible on every one.
PACKED = DATA / "cf__20v__1d3-3d17__y-desc.nc"

# UInt16 GOES ABI granule; its scan-angle axes are radians, so integer
# truncation destroys the grid outright rather than coarsening it.
GEOS = DATA / "cf__9v__1d7-2d2__geos__y-desc.nc"

# CF bounds store: `lat` / `lon` are carried into the result as auxiliary
# coordinate arrays, which is the path that lost their units.
BOUNDS = DATA / "cf__7v__1d3-2d3-3d1__y-asc.nc"

# Int16, and its file declares no fill at all -- the case where inventing one
# would be worse than carrying none.
NO_FILL = DATA / "coards__4v__1d3-3d1__y-desc.nc"


def _both_arms(fixture: Path, tmp_path: Path) -> tuple[NetCDF, NetCDF]:
    """Reproject one container both ways and return `(eager, streamed)`."""
    eager = NetCDF.read_file(str(fixture)).to_crs(4326)
    streamed_path = tmp_path / f"streamed_{fixture.name}"
    NetCDF.read_file(str(fixture)).to_crs(4326, path=str(streamed_path))
    return eager, NetCDF.read_file(str(streamed_path))


def _array_info(container: NetCDF, name: str) -> tuple[str, dict[str, str], str]:
    """Return `(dtype name, attributes, unit)` of one array of `container`."""
    md_arr = container._working_group().OpenMDArray(name)
    attrs = {a.GetName(): a.ReadAsString() for a in md_arr.GetAttributes()}
    dtype = gdal.GetDataTypeName(md_arr.GetDataType().GetNumericDataType())
    return dtype, attrs, md_arr.GetUnit()


def _effective_units(container: NetCDF, name: str) -> str:
    """The array's CF units, wherever GDAL happens to have put them.

    A netCDF-backed store surfaces `units` in the MDArray's own unit slot and
    an in-memory one leaves it in the attribute dictionary, so a comparison
    that reads only one of the two reports a difference the file does not have.
    """
    _dtype, attrs, unit = _array_info(container, name)
    return attrs.get("units") or unit


class TestTheAxesSurviveAnIntegerVariable:
    """A *spatial* coordinate array is float64, whatever dtype the data happens to be.

    The rule is narrower than "every coordinate array is float64", which is how
    it was first written. A non-spatial axis handed in as integers keeps its
    own dtype — see `TestANonSpatialAxisKeepsItsOwnDtype`.
    """

    @pytest.mark.parametrize(
        ("fixture", "variable"), [(PACKED, "blh"), (GEOS, "CMI")], ids=["era5", "goes"]
    )
    def test_the_two_arms_place_the_result_on_the_same_grid(
        self, fixture, variable, tmp_path
    ):
        """The georeference is the thing, and it was wrong on one arm.

        Args:
            fixture: A store whose data variables are integer-typed.
            variable: A gridded variable of that store.

        Test scenario:
            The geotransform is re-derived from the written coordinate arrays,
            so truncating them silently moves the data. Comparing the two arms'
            geotransforms is what a user would notice: the same reprojection,
            asked for two ways, must put the pixels in the same place.
        """
        eager, streamed = _both_arms(fixture, tmp_path)

        eager_gt = tuple(eager.get_variable(variable).geotransform)
        streamed_gt = tuple(streamed.get_variable(variable).geotransform)

        assert eager_gt == pytest.approx(streamed_gt, rel=1e-9), (
            f"the two arms disagree about the grid: {eager_gt} vs {streamed_gt}"
        )

    def test_the_eager_axis_is_float_and_keeps_its_fractional_values(self, tmp_path):
        """The mechanism, so a regression is diagnosable and not just visible.

        Test scenario:
            `x` on this fixture steps by 2.5 degrees. Written in the data
            variable's `Int16`, it came back `[0, 3, 5, ...]` -- monotonic,
            plausible, and wrong. Asserting the dtype alone would pass on a
            float array of rounded values, so the values are checked too.
        """
        eager, _ = _both_arms(PACKED, tmp_path)

        dtype, _attrs, _unit = _array_info(eager, "x")
        values = eager._working_group().OpenMDArray("x").ReadAsArray()

        assert dtype == "Float64", f"the axis took the data's dtype: {dtype}"
        assert not np.allclose(values, np.round(values)), (
            f"the axis lost its fractional part: {values[:5]}"
        )

    def test_a_large_time_axis_is_not_saturated(self, tmp_path):
        """`Int16` could not hold this axis at all, so it clipped.

        Test scenario:
            The band dimension carries epoch-scaled values. Written in the
            data's `Int16` they all became 32767 -- every step of the time axis
            collapsed onto one instant, which no reader can recover from.
        """
        eager, _ = _both_arms(PACKED, tmp_path)

        values = eager._working_group().OpenMDArray("time").ReadAsArray()

        assert len(set(values.tolist())) == len(values), (
            f"the time axis collapsed onto repeated values: {values[:5]}"
        )


class TestTheAxesDescribeThemselvesOnBothArms:
    """A CF reader must be able to tell degrees from metres on either file."""

    def test_the_streamed_axes_carry_the_same_cf_attributes(self, tmp_path):
        """The streamed file's `x` / `y` said nothing about themselves.

        Test scenario:
            The eager arm stamps `axis` / `standard_name` / `long_name` /
            `units` through `_create_dimension`; the streamed arm wrote empty
            attribute dictionaries. A reprojection to 4326 whose longitudes do
            not declare `degrees_east` is not interpretable without guessing.
        """
        eager, streamed = _both_arms(PACKED, tmp_path)

        for axis, expected_units in (("x", "degrees_east"), ("y", "degrees_north")):
            _dtype, eager_attrs, _unit = _array_info(eager, axis)
            _dtype, streamed_attrs, _unit = _array_info(streamed, axis)
            for key in ("axis", "standard_name", "long_name"):
                assert streamed_attrs.get(key) == eager_attrs.get(key), (
                    f"{axis}.{key}: eager={eager_attrs.get(key)!r} "
                    f"streamed={streamed_attrs.get(key)!r}"
                )
            assert _effective_units(eager, axis) == expected_units, (
                f"the eager {axis} lost its units"
            )
            assert _effective_units(streamed, axis) == expected_units, (
                f"the streamed {axis} declares no units"
            )

    def test_the_unit_lift_is_undone_when_a_carried_array_is_rebuilt(self):
        """GDAL moves `units` out of the attribute dictionary, so copying it loses it.

        Test scenario:
            `read_cf_attributes` returns what GDAL kept in the attribute
            dictionary, and GDAL lifts `units` into the array's own unit slot --
            the same lift that hides `scale_factor`. A carried array rebuilt from
            that dictionary alone therefore described nothing.
            `_aux_attrs_with_unit` puts it back.

            This asserts on the helper rather than on a round trip because no
            fixture in the corpus carries an auxiliary that has units: the axes
            that did (`lat`, `lon`) are deliberately no longer carried at all,
            since a carried spatial axis made the result report the source's
            grid rather than its own.
        """
        source = NetCDF.read_file(str(BOUNDS))
        rg = source._working_group()
        axis = rg.OpenMDArray("time")

        assert axis is not None, "the fixture no longer has a time axis"
        assert axis.GetUnit(), (
            "the fixture's time axis has no unit, so the lift cannot be observed"
        )

        attrs = NetCDF._aux_attrs_with_unit(axis)

        assert attrs.get("units") == axis.GetUnit(), (
            f"the lifted unit {axis.GetUnit()!r} was not restored to the "
            f"attributes a rebuild reads: {attrs}"
        )

    def test_neither_arm_carries_the_source_spatial_axes(self, tmp_path):
        """The two arms have to agree about *not* carrying them, too.

        Test scenario:
            The eager arm was taught to carry a carried auxiliary's dimension
            coordinates so the two arms would report the same
            `variable_names`. They did -- both wrong: a carried `lat`/`lon`
            pair is read by `_compute_geotransform` in preference to the stored
            transform, so both arms reported the source grid on a reprojected
            result. Agreement is now on not carrying them, which this pins on
            both sides so a future "fix" cannot restore one arm alone.
        """
        eager, streamed = _both_arms(BOUNDS, tmp_path)

        for label, result in (("eager", eager), ("streamed", streamed)):
            rg = result._working_group()
            names = set(rg.GetMDArrayNames() or [])
            assert not ({"lat", "lon"} & names), (
                f"the {label} arm carried a source spatial axis: {sorted(names)}"
            )


class TestTheFillValueSurvivesBothArms:
    """Masking the result must blank the same cells whichever spelling was used."""

    def test_a_declared_fill_reaches_the_streamed_file(self, tmp_path):
        """The streamed file declared no fill for a variable that has one.

        Test scenario:
            ERA5 declares `_FillValue = -32767` on every variable. The eager
            result carries it; the streamed one did not, so masking the
            streamed result left the fill cells in as ordinary -32767 data --
            a difference of about 3e4 in the wrong direction.
        """
        eager, streamed = _both_arms(PACKED, tmp_path)

        eager_ndv = eager.get_variable("blh").no_data_value
        streamed_ndv = streamed.get_variable("blh").no_data_value

        assert set(eager_ndv) == {-32767.0}, f"the fixture changed: {eager_ndv}"
        assert set(streamed_ndv) == set(eager_ndv), (
            f"the streamed fill differs: {streamed_ndv} vs {eager_ndv}"
        )

    def test_a_variable_with_no_declared_fill_gets_none_on_both_arms(self, tmp_path):
        """The other half: neither arm may invent a sentinel.

        Test scenario:
            This `Int16` store declares no fill at all. Carrying the warped
            wrapper's no-data across instead of the source array's own would
            stamp GDAL's uninitialised default here, and `0` is an ordinary
            value of every integer type -- masking would blank real cells.
            This is the assertion that makes the fill fix safe rather than
            merely present.
        """
        eager, streamed = _both_arms(NO_FILL, tmp_path)

        eager_ndv = eager.get_variable("air").no_data_value
        streamed_ndv = streamed.get_variable("air").no_data_value

        assert set(eager_ndv) == {None}, f"the eager arm invented a fill: {eager_ndv}"
        assert set(streamed_ndv) == {None}, (
            f"the streamed arm invented a fill: {streamed_ndv}"
        )


class TestANonSpatialAxisKeepsItsOwnDtype:
    """Float64 fixes a spatial axis and breaks a wide integer one.

    A geotransform is re-derived from `x` / `y`, so writing them in the data
    variable's `Int16` truncated a 2.5-degree grid to whole degrees — that is
    the defect the float64 rule exists for, and it stays. A `time` axis is not
    re-derived from anything; it is the values themselves that matter, and
    float64 holds only 53 bits of mantissa. An `int64` nanosecond epoch is
    exactly the case that does not fit.
    """

    def test_an_int64_epoch_axis_round_trips_exactly(self):
        """The values are the whole point of a time axis.

        Test scenario:
            Nanosecond stamps past 2^53 are representable in `int64` and not in
            `float64`. Written as float64 the axis came back 21 ns off — small,
            silent, and enough to make two stamps compare equal that are not.
        """
        stamps = np.array(
            [
                1_700_000_000_123_456_789,
                1_700_000_060_123_456_789,
                1_700_000_120_123_456_789,
            ],
            dtype=np.int64,
        )
        container = NetCDF.from_array(
            np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5),
            geo_ref=GeoReference(
                top_left_corner=(0.0, 10.0), cell_size=0.25, epsg=4326
            ),
            dims=ExtraDimensions(name="time", values=list(stamps)),
        )

        written = np.asarray(
            container._working_group().OpenMDArray("time").ReadAsArray()
        )

        assert np.array_equal(written.astype(np.int64), stamps), (
            f"the time axis lost exactness: {written.astype(np.int64)} vs {stamps}"
        )

    def test_the_spatial_axes_are_still_float64_beside_it(self):
        """The narrowing must not reach the axes the rule was written for.

        Test scenario:
            The same call carries an `Int16` data array. If the narrowing
            leaked into `x` / `y` they would take that dtype and the quarter-
            degree grid would collapse to whole degrees — the exact regression
            the float64 rule prevents.
        """
        container = NetCDF.from_array(
            np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5),
            geo_ref=GeoReference(
                top_left_corner=(0.0, 10.0), cell_size=0.25, epsg=4326
            ),
            dims=ExtraDimensions(name="time", values=[0, 1, 2]),
        )

        x_dtype, _attrs, _unit = _array_info(container, "x")
        x_values = np.asarray(container._working_group().OpenMDArray("x").ReadAsArray())

        assert x_dtype == "Float64", f"a spatial axis lost float64: {x_dtype}"
        assert not np.allclose(x_values, np.round(x_values)), (
            f"the spatial axis lost its fractional part: {x_values}"
        )


class TestNeitherArmInventsACoordinateArray:
    """A dimension with no coordinate variable stays that way on both arms."""

    @pytest.mark.parametrize(
        ("fixture", "invented"),
        [
            (BOUNDS, "bnds"),
            (GEOS, "number_of_image_bounds"),
            (GEOS, "band"),
        ],
        ids=["bounds", "image-bounds", "band"],
    )
    def test_the_streamed_file_does_not_gain_an_index_axis(
        self, fixture, invented, tmp_path
    ):
        """`np.arange(size)` reads as a real axis; the source declares none.

        Test scenario:
            `lat_bnds(lat, bnds)` has no coordinate variable for `bnds`, and
            netCDF is content with that — the eager arm writes nothing there.
            The streamed arm filled the gap with an index range, so one
            spelling of one call produced a file with an extra 1-D array a CF
            reader can only read as a coordinate.

        Args:
            fixture: A store with an auxiliary on an uncoordinated dimension.
            invented: The array name the streamed arm used to add.
            tmp_path: pytest temporary directory.
        """
        eager, streamed = _both_arms(fixture, tmp_path)
        streamed_names = set(streamed._working_group().GetMDArrayNames() or [])
        eager_names = set(eager._working_group().GetMDArrayNames() or [])

        assert invented not in eager_names, (
            f"the fixture changed: {sorted(eager_names)}"
        )
        assert invented not in streamed_names, (
            f"the streamed arm invented {invented!r}: {sorted(streamed_names)}"
        )

    def test_a_bounds_dimension_survives_without_its_coordinate(self, tmp_path):
        """Not writing the axis must not cost the array that needs the dimension.

        Test scenario:
            The point of the skip is that netCDF allows a bare dimension, so
            the carried `lat_bnds` still has to be there with its full shape.
            Asserting only the absence would pass on a file that dropped the
            bounds array along with the axis.
        """
        _eager, streamed = _both_arms(BOUNDS, tmp_path)
        rg = streamed._working_group()

        sizes = [d.GetSize() for d in rg.OpenMDArray("lat_bnds").GetDimensions()]

        assert sizes[-1] == 2, f"the bounds array lost its bnds axis: {sizes}"

    def test_the_arms_write_the_same_arrays(self, tmp_path):
        """The invariant the three cases above are instances of.

        Test scenario:
            Comparing the two files' array names directly is what a user would
            do to check the two spellings agree. It was three names apart on
            the GOES granule.
        """
        eager, streamed = _both_arms(GEOS, tmp_path)
        eager_path = tmp_path / "eager_geos.nc"
        eager.to_file(str(eager_path))
        written = NetCDF.read_file(str(eager_path))

        eager_names = set(written._working_group().GetMDArrayNames() or [])
        streamed_names = set(streamed._working_group().GetMDArrayNames() or [])

        assert streamed_names == eager_names, (
            f"stream-only={sorted(streamed_names - eager_names)} "
            f"eager-only={sorted(eager_names - streamed_names)}"
        )
