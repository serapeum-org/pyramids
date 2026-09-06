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

What still differs, measured after these fixes and deliberately not changed:
the first band dimension's coordinate array is written `Float64` by the eager
arm and in the source's own integer dtype by the streamed one. Both hold the
same values, so nothing is lost; converging it would churn every streamed
file's `time` dtype for no gain. The streamed file also keeps an extra
per-variable `nodata` attribute beside the real `_FillValue` — that is the
writer's own round-trip channel (#1061), and it is what carries the fill in the
first place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf import NetCDF

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
    """A coordinate array is float64, whatever dtype the data happens to be."""

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
