"""Tests for the parity harness itself.

It is test infrastructure every later parity test will inherit, so a fault in it would make
every downstream task agree on the wrong answer — the failure mode §7 of the plan calls the
single point of failure. These tests therefore do two things an ordinary suite would not: they
assert the normalisations are *load-bearing* (skip one and the comparison must fail) and they
assert the harness fails when it should, not only that it passes when it should.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.parity._harness import (
    ParityUnsupported,
    ParityView,
    _gap_mask,
    _packing,
    assert_parity,
    from_pyramids,
    stored_y_ascends,
    to_xr,
    y_dimension,
)
from tests.netcdf.parity.conftest import (
    NAN_SENTINEL,
    PARITY_FIXTURES,
    UNSUPPORTED,
    open_fixture,
)

pytestmark = pytest.mark.interop

# Flattened from the shared catalogue rather than restated: `conftest` records which stores the
# parity suite uses and why, and every later task parametrizes over the same table.
FIXTURES = [
    (case.path, case.variable, case.y_ascends, case.packed) for case in PARITY_FIXTURES
]
FIXTURE_IDS = [case.id for case in PARITY_FIXTURES]
UNSUPPORTED_IDS = ["roms-eta-rho", "curvilinear-y", "goes-index-y"]


def _read(name: str) -> NetCDF:
    """Open one of the parity fixtures by file name.

    Args:
        name: The `.nc` file name under `tests/data/netcdf`.

    Returns:
        NetCDF: The opened container.
    """
    return open_fixture(name)


class OnlyDimensions:
    """A stand-in for `NetCDF` exposing only the two members the y rules read.

    The smallest y axis among the repo's fixtures is five rows, so a one-cell axis cannot be
    reached through a file. `y_dimension` reads `dimension_sizes` and `stored_y_ascends` reads
    `get_dimension_values`; nothing else is needed to drive either of them.
    """

    def __init__(
        self, sizes: dict[str, int], coordinates: dict[str, list[float]]
    ) -> None:
        """Record the dimensions and their coordinate values.

        Args:
            sizes: The dimension sizes, as `NetCDF.dimension_sizes` reports them.
            coordinates: The stored values of each dimension.
        """
        self.dimension_sizes = sizes
        self._coordinates = coordinates

    def get_dimension_values(self, name: str) -> list[float]:
        """The stored coordinate values of one dimension.

        Args:
            name: The dimension to look up.

        Returns:
            The values recorded for it.
        """
        return self._coordinates[name]


class TestTheNoOpParity:
    """`read_array` must match `to_xarray()` on every fixture — the harness's reason to exist."""

    @pytest.mark.parametrize(
        ("name", "variable", "_ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_a_plain_read_matches_the_export(self, name, variable, _ascending, _packed):
        """The same numbers come back from both sides once the five differences are normalised.

        Args:
            name: The fixture file name.
            variable: The variable to compare.
            _ascending: Unused here; the parametrization is shared.
            _packed: Unused here; the parametrization is shared.

        Test scenario:
            This is the identity case. If it fails, every downstream parity assertion is
            meaningless, so it runs across both y orientations, packed and unpacked files, and
            variables with two, one and zero band dimensions.
        """
        nc = _read(name)
        assert_parity(from_pyramids(nc, variable), to_xr(nc, variable))

    @pytest.mark.parametrize(
        ("name", "variable", "_ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_both_sides_agree_on_the_shape(self, name, variable, _ascending, _packed):
        """The pyramids side is rebuilt onto the dimensions xarray keeps, not left flat.

        Args:
            name: The fixture file name.
            variable: The variable to compare.
            _ascending: Unused here.
            _packed: Unused here.
        """
        nc = _read(name)
        assert (
            from_pyramids(nc, variable).values.shape == to_xr(nc, variable).values.shape
        )

    @pytest.mark.parametrize(
        ("name", "variable", "_ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_both_sides_agree_on_the_dimension_names(
        self, name, variable, _ascending, _packed
    ):
        """The names are derived independently on each side and must still match.

        Args:
            name: The fixture file name.
            variable: The variable to compare.
            _ascending: Unused here.
            _packed: Unused here.

        Test scenario:
            pyramids names them from the container's own dimension list and the variable's band
            axes; xarray names them from the exported cube. A mismatch means the pyramids side
            was reshaped against the wrong axis order, which a value comparison on a
            square-ish array could miss.
        """
        nc = _read(name)
        assert from_pyramids(nc, variable).dims == to_xr(nc, variable).dims


class TestTheOrientationRule:
    """Normalisation 1 — and proof that it fires in exactly one direction."""

    @pytest.mark.parametrize(
        ("name", "_variable", "ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_the_stored_direction_is_read_correctly(
        self, name, _variable, ascending, _packed
    ):
        """`stored_y_ascends` reports what the file actually stores.

        Args:
            name: The fixture file name.
            _variable: Unused here.
            ascending: Whether the fixture stores its y axis south-to-north.
            _packed: Unused here.
        """
        assert stored_y_ascends(_read(name)) is ascending

    @pytest.mark.parametrize(
        ("name", "variable", "ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_the_flip_is_load_bearing_in_exactly_one_direction(
        self, name, variable, ascending, _packed
    ):
        """Flipping the *other* way must break the comparison — on both orientations.

        Args:
            name: The fixture file name.
            variable: The variable to compare.
            ascending: Whether the fixture stores its y axis south-to-north.
            _packed: Unused here.

        Test scenario:
            A harness that flipped nothing would still pass on a y-descending file, and one
            that flipped everything would still pass on a y-ascending file. Only asserting that
            the *wrong* orientation fails proves the rule is doing work, which is why the
            fixture set carries both directions.
        """
        nc = _read(name)
        view = to_xr(nc, variable)
        axis = view.dims.index(y_dimension(nc))
        wrong = dataclasses.replace(
            view,
            values=np.flip(view.values, axis),
            gaps=np.flip(view.gaps, axis),
        )
        with pytest.raises(AssertionError):
            assert_parity(from_pyramids(nc, variable), wrong)

    def test_a_single_row_axis_is_never_flipped(self):
        """With one y cell there is no direction to read, so the rule must not guess one.

        Test scenario:
            `values[0] < values[-1]` is `False` for a length-1 axis, which is the answer that
            leaves the array alone — asserted so a future rewrite cannot make it `True` by
            accident.
        """
        nc = _read("cf__5v__1d4-4d1__geog__y-desc.nc")
        single = np.asarray([42.0])
        assert bool(single[0] < single[-1]) is False
        assert stored_y_ascends(nc) is False

    def test_the_y_dimension_is_found_under_each_spelling(self):
        """`lat`, `latitude` and `y` all resolve; the fixtures use all three."""
        assert y_dimension(_read("cf__5v__1d4-4d1__y-asc.nc")) == "lat"
        assert y_dimension(_read("cf__5v__1d4-4d1__geog__y-desc.nc")) == "latitude"
        assert y_dimension(_read("coards__4v__1d2-2d2__scaleoffset__y-asc.nc")) == "y"

    @pytest.mark.parametrize(
        ("name", "variable", "reason"), UNSUPPORTED, ids=UNSUPPORTED_IDS
    )
    def test_an_undecidable_store_is_refused(self, name, variable, reason):
        """A store whose y axis cannot decide the flip must raise, not answer `False`.

        Args:
            name: The fixture file name.
            variable: The variable to attempt.
            reason: The phrase the refusal has to carry.

        Test scenario:
            These three are the reason the rule refuses rather than defaults. Measured on this
            branch before the guard existed: the ROMS store needs the flip on `eta_rho` (the
            xor of the two gap masks drops from 394392 to 0 once flipped), the curvilinear
            store needs it on `y` (134052 to 0), and the GOES store must *not* be flipped even
            though its `y` ascends, because that `y` is the row index `0, 1, … 499`. Answering
            `False` for the first two and `True` for the third is what the suite used to pin.
        """
        nc = _read(name)
        with pytest.raises(ParityUnsupported, match=reason):
            from_pyramids(nc, variable)

    @pytest.mark.parametrize(
        ("name", "variable", "reason"), UNSUPPORTED, ids=UNSUPPORTED_IDS
    )
    def test_the_export_side_refuses_the_same_stores(self, name, variable, reason):
        """`to_xr` must refuse whatever `from_pyramids` refuses, or one side is silently wrong.

        Args:
            name: The fixture file name.
            variable: The variable to attempt.
            reason: The phrase the refusal has to carry.
        """
        nc = _read(name)
        with pytest.raises(ParityUnsupported, match=reason):
            to_xr(nc, variable)

    def test_a_one_cell_y_axis_is_never_flipped(self):
        """A single row carries no direction, so the rule must not read one out of it.

        Test scenario:
            The smallest y axis among the repo's fixtures is five rows, so the degenerate case
            is driven from a stand-in container. `values[0] < values[-1]` would compare the one
            cell with itself; the size guard answers before that, and `False` is also the
            answer that leaves the array alone.
        """
        container = OnlyDimensions({"lat": 1, "lon": 3}, {"lat": [42.0]})
        assert stored_y_ascends(container) is False, (
            "a single-cell y axis has no stored direction, so nothing may be flipped"
        )


class TestTheGapRule:
    """Normalisation 2 — the masks must agree, and be taken before unpacking."""

    def test_a_fixture_with_real_gaps_agrees_on_where_they_are(self):
        """Half of `tcw`'s cells are fill, and both sides must mark the same ones.

        Test scenario:
            Four of the five fixtures declare a sentinel no cell holds, so the gap rule would
            be vacuous without this one — 63072 of 126144 cells really are fill.
        """
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        pyr = from_pyramids(nc, "tcw")
        assert int(pyr.gaps.sum()) == 63072, f"got {int(pyr.gaps.sum())}"
        assert np.array_equal(pyr.gaps, to_xr(nc, "tcw").gaps)

    def test_the_gaps_are_nan_on_both_sides(self):
        """A masked cell is `NaN` in `values`, so a value comparison cannot read fill as data."""
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        pyr = from_pyramids(nc, "tcw")
        assert np.isnan(pyr.values[pyr.gaps]).all(), "every masked cell should be NaN"
        assert not np.isnan(pyr.values[~pyr.gaps]).any(), (
            "no unmasked cell should be NaN"
        )

    def test_the_declared_sentinel_is_not_matched_against_physical_values(self):
        """Why the harness reads `masked=True` rather than comparing against `no_data_value`.

        Test scenario:
            `no_data_value` is a stored value, and an unpacked read carries the fill scaled
            along with the data, so `arr == no_data_value` finds nothing on a packed variable.
            `read_array` documents exactly this and offers `unpack=False` or `masked=True`; the
            harness takes the masked route, and this pins the reason.
        """
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        sentinel = nc.get_variable("tcw").no_data_value[0]
        unpacked = np.asarray(nc.read_array(variable="tcw"))

        assert int((unpacked == sentinel).sum()) == 0, (
            "matching a stored sentinel against physical values should find nothing"
        )
        assert int(from_pyramids(nc, "tcw").gaps.sum()) == 63072

    def test_the_mask_matches_the_stored_fill_cells(self):
        """`masked=True` marks exactly the cells that hold the sentinel before unpacking."""
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        stored = np.asarray(nc.read_array(variable="tcw", unpack=False))
        sentinel = nc.get_variable("tcw").no_data_value[0]
        assert np.array_equal(from_pyramids(nc, "tcw").gaps, stored == sentinel)

    def test_a_disagreement_about_the_gaps_fails(self):
        """One extra masked cell on one side must be caught, not averaged away."""
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        pyr = from_pyramids(nc, "tcw")
        moved = pyr.gaps.copy()
        moved[~moved] = True  # mark every remaining cell as a gap too
        with pytest.raises(AssertionError, match="gap masks differ"):
            assert_parity(dataclasses.replace(pyr, gaps=moved), to_xr(nc, "tcw"))

    def test_a_nan_sentinel_is_found_by_nan_and_not_by_equality(self):
        """A `NaN` fill value needs its own branch, because `nan == nan` is `False`.

        Test scenario:
            The equality path reports no gaps at all on a variable whose declared fill is
            `NaN`, which would silently compare fill against fill. The dedicated branch must
            find them, and the equality it replaces must be shown to find nothing — otherwise
            the branch could be deleted and no test would notice.
        """
        stored = np.asarray([[1.0, np.nan], [np.nan, 4.0]])
        mask = _gap_mask(stored, float("nan"))
        assert mask.tolist() == [[False, True], [True, False]], (
            f"the NaN cells should be the gaps, got {mask.tolist()}"
        )
        assert int((stored == float("nan")).sum()) == 0, (
            "equality against NaN finds nothing, which is the reason the branch exists"
        )

    def test_a_numeric_sentinel_is_found_by_equality(self):
        """A declared number marks exactly the cells holding it, and no others.

        Test scenario:
            The ordinary case, asserted on the mask itself rather than through a file so the
            fill cells and the data cells stay distinguishable by eye.
        """
        stored = np.asarray([[1, -999], [3, -999]])
        mask = _gap_mask(stored, -999)
        assert mask.tolist() == [[False, True], [False, True]], (
            f"only the -999 cells should be marked as gaps, got {mask.tolist()}"
        )

    def test_an_undeclared_sentinel_marks_nothing(self):
        """A variable with no fill value has no gaps, and the mask still matches its shape.

        Test scenario:
            `None` must produce an all-`False` mask shaped like the data, not an empty array:
            `assert_parity` compares the two masks elementwise, and a shape mismatch here would
            fail a comparison that has nothing to do with the values.
        """
        stored = np.zeros((2, 3, 4))
        mask = _gap_mask(stored, None)
        assert mask.shape == stored.shape, (
            f"expected a mask shaped {stored.shape}, got {mask.shape}"
        )
        assert not mask.any(), "nothing is declared as fill, so no cell can be a gap"

    def test_a_variable_whose_declared_fill_is_nan_reaches_parity(self):
        """The `NaN` sentinel is exercised end to end, not only as a unit.

        Test scenario:
            `t2m` on `cf__5v__1d4-3d1__geog__y-desc.nc` is the only variable in the repo that
            declares `NaN` as its fill, so it is the only file that drives `to_xr` down that
            branch. It holds no `NaN` cell, so both sides must report no gaps at all and still
            agree on every value.
        """
        nc = NAN_SENTINEL.open()
        assert np.isnan(nc.get_variable("t2m").no_data_value[0]), (
            "the fixture is here because its declared fill is NaN"
        )
        pyr = from_pyramids(nc, "t2m")
        assert not pyr.gaps.any(), (
            f"no cell of t2m holds the fill value, but {int(pyr.gaps.sum())} are masked"
        )
        assert_parity(pyr, to_xr(nc, "t2m"))


class TestThePackingRule:
    """Normalisation 4 — both sides end in physical units."""

    def test_the_xarray_side_is_unpacked(self):
        """`rhum` is a percentage: stored as `int16`, physical in 0-100.

        Test scenario:
            `to_xarray()` hands over the stored values, so without unpacking the xarray side
            would be tens of thousands where pyramids reports tens.
        """
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        view = to_xr(nc, "rhum")
        stored = np.asarray(nc.read_array(variable="rhum", unpack=False))

        assert view.dtype == np.dtype("int16"), f"the source is stored as {view.dtype}"
        assert float(np.abs(stored).max()) > 1000.0, (
            f"the stored values should be far outside 0-100, got {np.abs(stored).max()}"
        )
        # The bound is not exactly 100: the `float32` scale factor round-trips 100% to
        # 100.000008, which is the packing's own precision rather than a normalisation error.
        assert float(np.nanmin(view.values)) == pytest.approx(0.0, abs=1e-4)
        assert float(np.nanmax(view.values)) == pytest.approx(100.0, abs=1e-4)

    def test_an_unpacked_fixture_is_left_alone(self):
        """A variable with no packing must not be shifted by an identity that is not applied."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        handle = nc.get_variable("temperature")
        assert handle.scale[0] in (None, 1.0)
        assert handle.offset[0] in (None, 0.0)
        assert_parity(from_pyramids(nc, "temperature"), to_xr(nc, "temperature"))

    def test_absent_factors_default_to_the_identity(self):
        """A variable declaring neither `scale_factor` nor `add_offset` is left untouched.

        Test scenario:
            Every fixture in the repo reports both factors explicitly — `1.0` and `0.0` when
            unpacked — so the `None` defaults are driven from a stand-in handle. They have to
            be the identity: any other default would rescale an unpacked variable inside the
            normalisation meant to leave it alone.
        """
        factors = _packing(SimpleNamespace(scale=[None], offset=[None]))
        assert factors == (1.0, 0.0), (
            f"an undeclared packing must be the identity, got {factors}"
        )


class TestTheHarnessFailsWhenItShould:
    """A harness that cannot fail proves nothing."""

    def test_a_perturbed_value_fails(self):
        """One cell moved by more than the tolerance is caught."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        broken = pyr.values.copy()
        broken[0, 0, 0, 0] += 1.0
        with pytest.raises(AssertionError):
            assert_parity(
                dataclasses.replace(pyr, values=broken), to_xr(nc, "temperature")
            )

    def test_a_shape_mismatch_fails_before_the_values_are_compared(self):
        """The shape check runs first, so the message names the shapes rather than the cells."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        with pytest.raises(AssertionError, match="shape differs"):
            assert_parity(
                dataclasses.replace(pyr, values=pyr.values[:2], gaps=pyr.gaps[:2]),
                to_xr(nc, "temperature"),
            )

    def test_a_broken_dtype_contract_fails(self):
        """A stated dtype is checked; `float32` where `float64` was contracted is an error."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        with pytest.raises(AssertionError, match="dtype contract broken"):
            assert_parity(
                from_pyramids(nc, "temperature"),
                to_xr(nc, "temperature"),
                dtype="float32",
            )

    def test_the_dtype_check_is_skipped_by_default(self):
        """A reduction promotes to float64, so the default must not assert the input dtype.

        Test scenario:
            `rhum` is stored as `int16`. Were the check on by default, every parity test over a
            packed variable would fail on a promotion that is the documented behaviour.
        """
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        assert_parity(from_pyramids(nc, "rhum"), to_xr(nc, "rhum"))

    def test_a_declared_dtype_that_holds_passes(self):
        """The contract check is usable, not merely refusable."""
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        assert_parity(from_pyramids(nc, "rhum"), to_xr(nc, "rhum"), dtype="int16")

    def test_an_unknown_variable_is_refused_by_name(self):
        """`to_xr` names the variable and lists what the cube holds."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        with pytest.raises(KeyError, match="nope"):
            to_xr(nc, "nope")


class TestTheParityView:
    """The value object both sides come back as."""

    def test_it_is_immutable(self):
        """A view handed to `assert_parity` cannot be edited by it, or by the test after it."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        view = from_pyramids(nc, "temperature")
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.values = np.zeros(1)

    def test_the_coordinates_are_normalised_with_the_values(self):
        """A flipped y axis must carry its coordinate with it, or `sel` parity would be wrong.

        Test scenario:
            The y-ascending fixture stores `[40, 41, 42, 43, 44]`; north-up raster order is the
            reverse, and both sides must report that.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        assert list(pyr.coords["lat"]) == [44.0, 43.0, 42.0, 41.0, 40.0]
        assert list(to_xr(nc, "temperature").coords["lat"]) == list(pyr.coords["lat"])

    def test_an_unflipped_axis_keeps_its_stored_coordinate(self):
        """The descending fixture is already north-up, so its coordinate is untouched."""
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        stored = list(np.asarray(nc.get_dimension_values("lat")))
        assert list(from_pyramids(nc, "rhum").coords["lat"]) == stored

    def test_it_carries_the_source_dtype_not_the_comparison_one(self):
        """`values` is always float64; `dtype` records what the file held."""
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        view = from_pyramids(nc, "rhum")
        assert view.values.dtype == np.dtype("float64")
        assert view.dtype == np.dtype("int16")

    def test_the_two_sides_report_the_same_source_dtype(self):
        """Both derive it from the stored values, so they must agree."""
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        assert from_pyramids(nc, "rhum").dtype == to_xr(nc, "rhum").dtype

    def test_a_precomputed_result_can_be_normalised(self):
        """`from_pyramids(values=…)` is how a later task will feed an operation's output in.

        Test scenario:
            Every downstream task computes something other than a plain read, so the harness has
            to accept a result rather than always reading one.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        read = np.asarray(nc.read_array(variable="temperature"))
        assert_parity(
            from_pyramids(nc, "temperature", values=read), to_xr(nc, "temperature")
        )


class TestAViewIsItsOwnParity:
    """`assert_parity(x, x)` — the degenerate case the plan asks for explicitly."""

    @pytest.mark.parametrize(
        ("name", "variable", "_ascending", "_packed"), FIXTURES, ids=FIXTURE_IDS
    )
    def test_a_view_matches_itself(self, name, variable, _ascending, _packed):
        """Reflexivity, on every fixture.

        Args:
            name: The fixture file name.
            variable: The variable to compare.
            _ascending: Unused here.
            _packed: Unused here.
        """
        view: ParityView = from_pyramids(_read(name), variable)
        assert_parity(view, view)


class TestTheLabelChecks:
    """`assert_parity` compares the names and the coordinates, not only the numbers (C1)."""

    def test_wrong_dimension_names_fail(self):
        """A mislabelled result is caught even when every cell matches.

        Test scenario:
            The values cannot see a naming error at all, and the plan's deliverable for the
            harness names `dims` among the four things `assert_parity` checks.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        with pytest.raises(AssertionError, match="dimension names differ"):
            assert_parity(
                dataclasses.replace(pyr, dims=("WRONG", "NAMES", "HERE", "TOO")),
                to_xr(nc, "temperature"),
            )

    def test_a_transposed_result_fails(self):
        """A swapped axis order is caught on a grid the value check cannot see it on.

        Test scenario:
            `z` is 21x21 and symmetric under transpose, so the cells compare equal either way.
            Only the axis labels distinguish them, which is why they are asserted.
        """
        nc = _read("coards__4v__1d2-2d2__scaleoffset__y-asc.nc")
        pyr = from_pyramids(nc, "z")
        transposed = dataclasses.replace(
            pyr,
            values=pyr.values.T,
            gaps=pyr.gaps.T,
            dims=pyr.dims[::-1],
            coords={name: pyr.coords[name] for name in pyr.dims[::-1]},
        )
        with pytest.raises(AssertionError, match="dimension names differ"):
            assert_parity(transposed, to_xr(nc, "z"))

    def test_a_shifted_coordinate_fails(self):
        """A result on the right grid but the wrong coordinates is not parity."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        moved = dict(pyr.coords)
        moved["lat"] = moved["lat"] + 1.0
        with pytest.raises(AssertionError, match="'lat' coordinate differs"):
            assert_parity(
                dataclasses.replace(pyr, coords=moved), to_xr(nc, "temperature")
            )

    def test_the_coordinate_check_can_be_waived(self):
        """`coords=False` is for an operation that deliberately changes them."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        moved = dict(pyr.coords)
        moved["lat"] = moved["lat"] + 1.0
        assert_parity(
            dataclasses.replace(pyr, coords=moved),
            to_xr(nc, "temperature"),
            coords=False,
        )

    def test_a_missing_coordinate_fails(self):
        """Dropping a coordinate is a difference, not an omission to tolerate."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        fewer = {k: v for k, v in pyr.coords.items() if k != "lat"}
        with pytest.raises(AssertionError, match="coordinate names differ"):
            assert_parity(
                dataclasses.replace(pyr, coords=fewer), to_xr(nc, "temperature")
            )

    def test_the_time_instants_are_cross_checked(self):
        """Where both sides decode the axis, the instants must agree.

        Test scenario:
            `get_time_variable` and the interop decoder are different code paths, so their
            agreeing is a real check rather than a tautology. They do not always both decode —
            `cf__20v__1d3-3d17__y-desc.nc` is exported as 2002 dates while `get_time_variable`
            declines it — which is why the stored offsets are what is always compared.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        pyr = from_pyramids(nc, "temperature")
        assert "time" in pyr.decoded_coords, "pyramids should decode this axis"
        assert "time" in to_xr(nc, "temperature").decoded_coords

        wrong = dict(pyr.decoded_coords)
        wrong["time"] = wrong["time"] + np.timedelta64(1, "D")
        with pytest.raises(AssertionError, match="'time' instants differ"):
            assert_parity(
                dataclasses.replace(pyr, decoded_coords=wrong),
                to_xr(nc, "temperature"),
            )


class TestAnOperationsResult:
    """The extension point every task after T0 goes through (H4/H5)."""

    def test_a_reduction_can_be_expressed(self):
        """A result that drops an axis is normalised, given its gaps and its dims.

        Test scenario:
            T5 extends `reduce`, and a reduction is the first thing that changes the shape.
            The source mask no longer describes the result, so the caller supplies both — and
            the harness compares it against the same reduction on the xarray side.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        source = from_pyramids(nc, "temperature")
        reduced = np.nanmean(source.values, axis=0)

        view = from_pyramids(
            nc,
            "temperature",
            values=reduced,
            gaps=np.zeros(reduced.shape, dtype=bool),
            dims_override=("pressure_level", "lat", "lon"),
        )
        assert view.dims == ("pressure_level", "lat", "lon")
        assert view.values.shape == reduced.shape
        np.testing.assert_allclose(view.values, reduced)

    def test_a_shape_change_without_gaps_is_refused_by_name(self):
        """The refusal says both shapes and what to pass, rather than a numpy broadcast error.

        Test scenario:
            Before this, a reduction hit
            `ValueError: operands could not be broadcast together with shapes (12,5,6) () (3,5,6)`
            — a message naming neither the harness nor the fix.
        """
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        reduced = np.zeros((3, 5, 6))
        with pytest.raises(ParityUnsupported, match="shape-changing operation"):
            from_pyramids(nc, "temperature", values=reduced)

    def test_a_mismatched_mask_is_refused(self):
        """A `gaps=` that does not describe the result is caught, not broadcast."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        with pytest.raises(ParityUnsupported, match="describe the same array"):
            from_pyramids(
                nc,
                "temperature",
                values=np.zeros((3, 5, 6)),
                gaps=np.zeros((4, 3, 5, 6), dtype=bool),
            )

    def test_a_mask_changing_operation_keeps_its_own_gaps(self):
        """`fillna` fills gaps, so the source mask must not be re-imposed on its result.

        Test scenario:
            T11's whole point is changing which cells are gaps. Re-applying the source mask
            would re-NaN exactly the cells the operation just filled, making the comparison
            against xarray wrong by construction.
        """
        nc = _read("cf__20v__1d3-3d17__y-desc.nc")
        source = from_pyramids(nc, "tcw")
        assert int(source.gaps.sum()) == 63072

        filled = np.nan_to_num(source.values, nan=0.0)
        view = from_pyramids(
            nc, "tcw", values=filled, gaps=np.zeros(filled.shape, dtype=bool)
        )
        assert int(view.gaps.sum()) == 0, "the filled cells must not be re-masked"
        assert not np.isnan(view.values).any()

    def test_the_result_dtype_is_what_a_contract_checks(self):
        """A reduction's float64 output is assertable, which is the point of normalisation 3.

        Test scenario:
            `ParityView.dtype` used to record the file's stored dtype even for a supplied
            result, so `assert_parity(dtype=...)` could only ever restate what the file holds.
        """
        nc = _read("coards__5v__1d4-4d1__y-desc.nc")
        source = from_pyramids(nc, "rhum")
        assert source.dtype == np.dtype("int16"), (
            "a plain read reports the stored dtype"
        )

        promoted = from_pyramids(
            nc,
            "rhum",
            values=source.values.astype("float64"),
            gaps=source.gaps,
        )
        assert promoted.dtype == np.dtype("float64"), f"got {promoted.dtype}"
        assert promoted.source_dtype == np.dtype("int16"), (
            "the stored dtype is kept too"
        )

    def test_the_flat_read_shape_is_accepted(self):
        """A caller holding a plain `read_array` result does not have to reshape it first."""
        nc = _read("cf__5v__1d4-4d1__y-asc.nc")
        flat = np.asarray(nc.read_array(variable="temperature"))
        assert flat.shape == (12, 5, 6)
        view = from_pyramids(nc, "temperature", values=flat)
        assert view.values.shape == (4, 3, 5, 6)
        assert_parity(view, to_xr(nc, "temperature"))
