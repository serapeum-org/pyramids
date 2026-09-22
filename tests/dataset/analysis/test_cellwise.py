"""The Tier 2 cell-wise members — `clip`, `round`, `astype` and `isin`.

Each expected value was measured on xarray 2026.7.0 on the same cells, `1..8` with one gap,
and is quoted in the test that asserts it. The rule shared by all four is the one `where`
and `fillna` already follow: **a gap stays a gap**. Clipping, rounding or casting the
sentinel as if it were data would turn a missing cell into a measurement.

Every member is exercised on a plain raster built with `from_array` *and* on a variable read
from a real store. The second matters: `identical` shipped broken for store variables
(#1178) because every one of its tests used `from_array`, whose metadata is a plain dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines.analysis import _declared_gaps, _holds, _regapped
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
CELLS = np.array([[1.0, 2.0, 3.0, NDV], [5.0, 6.0, 7.0, 8.0]])
STORE = Path(__file__).parents[2] / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"
GAPPY_STORE = (
    Path(__file__).parents[2] / "data" / "netcdf" / "cf__4v__1d3-3d1__proj__y-desc.nc"
)
PACKED_STORE = (
    Path(__file__).parents[2]
    / "data"
    / "netcdf"
    / "coards__4v__1d2-2d2__scaleoffset__y-asc.nc"
)


def _raster(cells: np.ndarray = CELLS, no_data_value: float | None = NDV) -> Dataset:
    """A 2x4 raster holding 1..8 with a gap where 4 would be.

    Args:
        cells: The values.
        no_data_value: The declared sentinel.

    Returns:
        Dataset: The raster.
    """
    return Dataset.from_array(cells, geo_ref=GEO, no_data_value=no_data_value)


def _read(raster: Dataset) -> list:
    """The cells as a flat list, gaps as NaN.

    Args:
        raster: The raster.

    Returns:
        list: One value per cell.
    """
    values = np.asarray(raster.read_array(), dtype="float64")
    sentinel = raster.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values.ravel().tolist()


def _store_variable() -> NetCDF:
    """A variable read from a CF store — the path `from_array` never reaches.

    Returns:
        NetCDF: The first gridded variable.
    """
    store = NetCDF.read_file(str(STORE))
    return store.get_variable(store.variable_names[0])


class TestClip:
    """`clip` bounds the values; a gap is not a value and stays a gap."""

    def test_both_bounds(self):
        """`da.clip(2, 6)` answers `[2, 2, 3, nan, 5, 6, 6, 6]` on xarray."""
        assert_allclose(
            _read(_raster().clip(2.0, 6.0)),
            [2.0, 2.0, 3.0, np.nan, 5.0, 6.0, 6.0, 6.0],
            equal_nan=True,
        )

    def test_a_lower_bound_only(self):
        """`da.clip(min=3)` answers `[3, 3, 3, nan, 5, 6, 7, 8]` on xarray."""
        assert_allclose(
            _read(_raster().clip(min=3.0)),
            [3.0, 3.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
            equal_nan=True,
        )

    def test_the_gap_is_not_clipped_into_data(self):
        """The sentinel `-9999` is below any lower bound, and must not be raised to it.

        Test scenario:
            Clipping the stored array would lift the gap's `-9999.0` to `3.0`, turning a
            missing cell into a measurement.
        """
        clipped = _raster().clip(min=3.0)
        assert np.asarray(clipped.isnull().read_array()).sum() == 1

    def test_the_band_keeps_its_type(self):
        """Bounds the band can hold do not widen it."""
        raster = _raster(CELLS.astype("float32"))
        assert np.asarray(raster.clip(2.0, 6.0).read_array()).dtype == np.float32

    def test_numpy_scalar_bounds_keep_the_band_type(self):
        """Percentiles and statistics arrive as numpy scalars, and must not widen the band.

        Test scenario:
            `np.clip` promotes under NEP 50 when a bound is a `numpy.float64`, so a
            `float32` band came back `float64` and a `uint8` band `int64` — a GDAL Int64
            band written from a byte raster. `where` casts its result back for exactly
            this reason; `clip` computed the same type and then dropped it.
        """
        floats = Dataset.from_array(
            np.array([[1.0, 5.0, 9.0]], dtype="float32"), geo_ref=GEO, no_data_value=NDV
        )
        bytes_ = Dataset.from_array(
            np.array([[1, 5, 9]], dtype="uint8"), geo_ref=GEO, no_data_value=None
        )
        assert floats.clip(np.float64(2.0), np.float64(6.0)).dtype == ["float32"]
        assert bytes_.clip(np.int64(2), np.int64(6)).dtype == ["uint8"]

    def test_a_bound_the_band_cannot_hold_still_widens_it(self):
        """The widening judgement itself is unchanged: a `uint8` band bounded at 300."""
        bytes_ = Dataset.from_array(
            np.array([[1, 5, 9]], dtype="uint8"), geo_ref=GEO, no_data_value=None
        )
        widened = bytes_.clip(min=300.0)
        assert widened.dtype != ["uint8"], (
            f"300 does not fit uint8, got {widened.dtype}"
        )
        assert np.asarray(widened.read_array()).ravel().tolist() == [
            300.0,
            300.0,
            300.0,
        ]

    def test_no_bound_is_refused(self):
        """Clipping to nothing is a caller's mistake."""
        raster = _raster()
        with pytest.raises(ValueError, match="at least one"):
            raster.clip()

    def test_crossed_bounds_are_refused(self):
        """`min > max` is refused, where numpy would quietly set everything to `max`."""
        raster = _raster()
        with pytest.raises(ValueError, match="min"):
            raster.clip(6.0, 2.0)

    @pytest.mark.parametrize("bounds", [{"min": np.nan}, {"max": np.nan}])
    def test_a_nan_bound_is_refused(self, bounds: dict):
        """NaN compares false against everything, so np.clip would blank the raster.

        Test scenario:
            `clip(min=np.nan)` used to answer `[nan, -9999.0, nan]` — every data cell NaN
            while the raster still declares `-9999.0`, so `isnull` flagged none of them.

        Args:
            bounds: The NaN bound under test.
        """
        raster = _raster()
        with pytest.raises(ValueError, match="NaN"):
            raster.clip(**bounds)

    def test_on_a_store_variable(self):
        """A bound that actually moves cells, on a variable read from a file.

        Test scenario:
            The fixture holds 0..3245, so `clip(min=0)` changed nothing and the test
            passed for an implementation that returned its input. The bound here is inside
            the range, so every cell below it must come back at it.
        """
        variable = _store_variable()
        before = np.asarray(variable.read_array(), dtype="float64")
        clipped = np.asarray(variable.clip(100.0, 1000.0).read_array(), dtype="float64")
        assert (before < 100.0).any(), "the fixture must hold cells below the bound"
        assert clipped.min() == 100.0, (
            f"cells below 100 were not raised: {clipped.min()}"
        )
        assert clipped.max() == 1000.0, (
            f"cells above 1000 were not lowered: {clipped.max()}"
        )
        assert np.array_equal(clipped, np.clip(before, 100.0, 1000.0))


class TestRound:
    """`round` rounds the values; a gap stays a gap."""

    def test_to_whole_numbers(self):
        """`(da / 3).round()` answers `[0, 1, 1, nan, 2, 2, 2, 3]` on xarray."""
        thirds = _raster(np.where(CELLS == NDV, NDV, CELLS / 3.0))
        assert_allclose(
            _read(thirds.round()),
            [0.0, 1.0, 1.0, np.nan, 2.0, 2.0, 2.0, 3.0],
            equal_nan=True,
        )

    def test_to_one_decimal(self):
        """`(da / 3).round(1)` answers `[0.3, 0.7, 1.0, nan, 1.7, 2.0, 2.3, 2.7]`."""
        thirds = _raster(np.where(CELLS == NDV, NDV, CELLS / 3.0))
        assert_allclose(
            _read(thirds.round(1)),
            [0.3, 0.7, 1.0, np.nan, 1.7, 2.0, 2.3, 2.7],
            equal_nan=True,
        )

    def test_a_sentinel_with_a_fraction_is_not_rounded_away(self):
        """A gap marked `-9999.5` must still be a gap after rounding.

        Test scenario:
            Rounding the stored array turns `-9999.5` into `-10000.0`, which no longer
            matches the declared sentinel, so the gap would read as a measurement.
        """
        cells = np.array([[1.4, -9999.5], [2.6, 3.5]])
        raster = _raster(cells, no_data_value=-9999.5)
        assert np.asarray(raster.round().isnull().read_array()).sum() == 1

    def test_a_non_integer_decimals_is_refused(self):
        """`decimals` counts digits, so it is a whole number."""
        raster = _raster()
        with pytest.raises(TypeError, match="integer"):
            raster.round(1.5)

    def test_on_a_store_variable(self):
        """Rounding that actually moves cells, on a variable read from a file.

        Test scenario:
            Every value in the fixture is already whole, so `round()` changed nothing.
            `round(-2)` rounds to hundreds, which moves almost all of them.
        """
        variable = _store_variable()
        before = np.asarray(variable.read_array(), dtype="float64")
        rounded = np.asarray(variable.round(-2).read_array(), dtype="float64")
        assert not np.array_equal(before, rounded), "round(-2) changed nothing"
        assert np.array_equal(rounded, np.round(before, -2))
        assert variable.round(-2)._band_dim_names == variable._band_dim_names


class TestAstype:
    """`astype` changes the band type; a gap stays a gap, re-marked for the new type."""

    def test_to_int32(self):
        """`da.fillna(-1).astype("int32")` answers `int32 [1, 2, 3, -1, 5, 6, 7, 8]`.

        Test scenario:
            The source declares `-9999.0`, which `int32` holds, so it carries over and the
            gap is the same gap under the new type.
        """
        cast = _raster().astype("int32")
        values = np.asarray(cast.read_array())
        assert values.dtype == np.int32
        assert_allclose(
            _read(cast), [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0], equal_nan=True
        )
        assert cast.no_data_value[0] == NDV

    def test_the_gap_is_still_a_gap(self):
        """Casting must not turn the missing cell into a measurement."""
        cast = _raster().astype("int32")
        assert np.asarray(cast.isnull().read_array()).sum() == 1

    def test_a_sentinel_the_new_type_cannot_hold_is_refused(self):
        """`-9999` does not fit `uint8`, and a wrapped sentinel would mark nothing.

        Test scenario:
            Without the refusal the gap is cast to `241`, a real `uint8` value, and the
            missing cell becomes data.
        """
        raster = _raster()
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("uint8")

    def test_a_new_sentinel_can_be_given(self):
        """Naming one the new type holds lets the cast go ahead."""
        cast = _raster().astype("uint8", no_data_value=255)
        assert np.asarray(cast.read_array()).dtype == np.uint8
        assert cast.no_data_value[0] == 255
        assert np.asarray(cast.isnull().read_array()).sum() == 1

    def test_a_sentinel_that_collides_with_data_is_refused(self):
        """A sentinel a real cell already holds would delete that measurement.

        Test scenario:
            The recipe the docs used to give — `clip(0, 255).astype("uint8",
            no_data_value=255)` — clamps every cell at or above 255 to 255 and then
            declares 255 missing, so three measured cells (255, 300 and 1200 m) read as
            gaps. This is the mirror of "a gap stays a gap": data has to stay data.
        """
        dem = _raster(np.array([[12.0, 254.0, 255.0, 300.0], [1200.0, NDV, 0.5, 80.0]]))
        with pytest.raises(ValueError, match="already hold"):
            dem.clip(0.0, 255.0).astype("uint8", no_data_value=255)

    def test_a_sentinel_outside_the_data_is_accepted(self):
        """Bounding one below the sentinel is the fix the refusal asks for."""
        dem = _raster(np.array([[12.0, 254.0, 255.0, 300.0], [1200.0, NDV, 0.5, 80.0]]))
        small = dem.clip(0.0, 254.0).astype("uint8", no_data_value=255)
        assert np.asarray(small.read_array()).ravel().tolist() == [
            12,
            254,
            254,
            254,
            254,
            255,
            0,
            80,
        ]
        assert np.asarray(small.isnull().read_array()).sum() == 1

    def test_a_kept_sentinel_that_the_cast_collides_with_is_refused(self):
        """Truncation can move a real value onto the sentinel too.

        Test scenario:
            `-9999.4` is data and `-9999.0` is the gap. Casting to `int32` truncates
            towards zero, so the measurement lands exactly on the sentinel and would be
            read as missing from then on.
        """
        raster = _raster(np.array([[1.0, -9999.4], [-9999.0, 4.0]]))
        with pytest.raises(ValueError, match="already hold"):
            raster.astype("int32")

    def test_a_nan_sentinel_cannot_mark_an_integer_band(self):
        """No integer means NaN, so the caller has to name one."""
        raster = _raster(np.where(CELLS == NDV, np.nan, CELLS), no_data_value=np.nan)
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("int16")

    def test_an_unsupported_type_is_refused(self):
        """GDAL has no boolean band, and says so."""
        raster = _raster()
        with pytest.raises(TypeError, match="bool"):
            raster.astype("bool")

    def test_on_a_store_variable(self):
        """A variable read from a file casts its values, and keeps its band dimensions."""
        variable = _store_variable()
        before = np.asarray(variable.read_array(), dtype="float64")
        cast = variable.astype("int32")
        values = np.asarray(cast.read_array())
        assert values.dtype == np.int32
        assert np.array_equal(values, before.astype("int32"))
        assert cast._band_dim_names == variable._band_dim_names

    def test_a_nan_gap_without_a_declared_sentinel_stays_nan(self):
        """A float cast keeps the gap, whether or not the raster declares a sentinel.

        Test scenario:
            A raster that declares nothing but holds NaN is a gap-holding raster — the same
            `_domain_mask` treats those cells as gaps. xarray's `astype("float32")` keeps
            the NaN; the cells must not come back holding whatever memory was allocated.
        """
        cells = np.array([[1.0, np.nan, 3.0, np.nan]])
        cast = _raster(cells, no_data_value=None).astype("float32")
        assert_allclose(
            np.asarray(cast.read_array(), dtype="float64").ravel(),
            [1.0, np.nan, 3.0, np.nan],
            equal_nan=True,
        )

    def test_an_integer_cast_of_an_unmarked_gap_is_refused(self):
        """An integer band has no NaN, so an unmarked gap has nowhere to go.

        Test scenario:
            `no_data_value=None` asks for a result that declares no sentinel, and the
            raster holds a gap. Casting it into `int32` leaves that cell with no way to
            read as missing, so the caller is asked for a sentinel instead.
        """
        raster = _raster()
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("int32", no_data_value=None)

    def test_a_gapless_raster_casts_with_no_sentinel(self):
        """`no_data_value=None` is fine when there is no gap to mark."""
        cast = _raster(np.array([[1.0, 2.0, 3.0]])).astype("int32", no_data_value=None)
        assert np.asarray(cast.read_array()).ravel().tolist() == [1, 2, 3]
        assert cast.no_data_value[0] is None


class TestABandKeepsItsOwnSentinel:
    """A multi-band raster may declare a different no-data value per band, and must keep it.

    Band 1 declares `-9999` and band 2 declares `-1`, and band 2 holds a real `-9999` — a
    multi-sensor stack, where the sentinel is whatever each source used. Collapsing the two
    onto band 1's rewrote band 2's declaration and turned its measurement into a gap.
    """

    CELLS = np.array([[[1.0, 2.0], [-9999.0, 4.0]], [[-1.0, 6.0], [-9999.0, 8.0]]])

    def _stack(self) -> Dataset:
        """The two-band raster with one sentinel each.

        Returns:
            Dataset: The stack.
        """
        return Dataset.from_array(
            self.CELLS, geo_ref=GEO, no_data_value=[-9999.0, -1.0]
        )

    def test_the_fixture_reads_one_gap_per_band(self):
        """The precondition: each band's own sentinel marks exactly one cell."""
        flags = np.asarray(self._stack().isnull().read_array()).tolist()
        assert flags == [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("round", lambda ds: ds.round()),
            ("clip", lambda ds: ds.clip(min=-10000.0)),
            ("astype", lambda ds: ds.astype("int32")),
        ],
    )
    def test_each_band_keeps_its_declaration(self, member: str, call):
        """The declarations survive the member.

        Args:
            member: The member under test.
            call: How to call it.
        """
        out = call(self._stack())
        assert [float(one) for one in out.no_data_value] == [-9999.0, -1.0], (
            f"{member} rewrote the per-band sentinels as {out.no_data_value}"
        )

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("round", lambda ds: ds.round()),
            ("clip", lambda ds: ds.clip(min=-10000.0)),
            ("astype", lambda ds: ds.astype("int32")),
        ],
    )
    def test_a_real_value_does_not_become_a_gap(self, member: str, call):
        """Band 2's real `-9999` is data, and only band 1's `-9999` is a gap.

        Args:
            member: The member under test.
            call: How to call it.
        """
        flags = np.asarray(call(self._stack()).isnull().read_array()).tolist()
        assert flags == [[[0, 0], [1, 0]], [[1, 0], [0, 0]]], (
            f"{member} changed which cells read as gaps: {flags}"
        )


class TestIsin:
    """`isin` flags the cells whose value is in a set; a gap is in no set."""

    def test_flags(self):
        """`da.isin([2, 5])` answers `[F, T, F, F, T, F, F, F]` on xarray."""
        flags = np.asarray(_raster().isin([2.0, 5.0]).read_array())
        assert flags.ravel().tolist() == [0, 1, 0, 0, 1, 0, 0, 0]

    def test_the_flags_are_uint8(self):
        """GDAL has no boolean band, so the flags are `uint8`, like `isnull`'s."""
        assert np.asarray(_raster().isin([2.0]).read_array()).dtype == np.uint8

    def test_a_gap_is_in_no_set_even_its_own_sentinel(self):
        """Asking for `-9999` does not flag the gap that happens to hold it.

        Test scenario:
            The gap stores `-9999.0`, but it is missing, not the value `-9999`, so it is
            flagged `0` — as xarray flags a NaN `False` whatever is asked for.
        """
        flags = np.asarray(_raster().isin([NDV]).read_array())
        assert flags.sum() == 0

    def test_it_reads_as_a_where_condition(self):
        """The flags are the condition `where` takes, the purpose they are shaped for."""
        raster = _raster()
        kept = raster.where(raster.isin([2.0, 5.0]))
        assert_allclose(
            _read(kept),
            [np.nan, 2.0, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan],
            equal_nan=True,
        )

    def test_a_scalar_is_one_value(self):
        """A bare number is the set of that one number."""
        flags = np.asarray(_raster().isin(6.0).read_array())
        assert flags.ravel().tolist() == [0, 0, 0, 0, 0, 1, 0, 0]

    def test_on_a_store_variable(self):
        """The flags match the cells, on a variable read from a file.

        Test scenario:
            Asserting only the band dimensions passed for an implementation that flagged
            nothing; the fixture holds exactly one `0.0`.
        """
        variable = _store_variable()
        before = np.asarray(variable.read_array(), dtype="float64")
        wanted = [float(before.max()), 0.0]
        flags = variable.isin(wanted)
        assert np.array_equal(
            np.asarray(flags.read_array()).astype(bool), np.isin(before, wanted)
        )
        assert np.asarray(flags.read_array()).sum() >= 2, (
            "the fixture holds both values"
        )
        assert flags._band_dim_names == variable._band_dim_names


class TestAStoreVariableWithGaps:
    """The store path the module docstring calls essential, on a variable that has gaps.

    `cf__5v__1d4-4d1__y-asc.nc` declares no sentinel and holds no gap, so "a gap stays a
    gap" was never exercised on a variable read from a file. This one declares
    `-3.4028e+38` and holds 279 of them.
    """

    @staticmethod
    def _variable() -> NetCDF:
        """The gap-holding store variable.

        Returns:
            NetCDF: The variable.
        """
        store = NetCDF.read_file(str(GAPPY_STORE))
        return store.get_variable("values")

    def test_the_fixture_has_gaps(self):
        """The precondition the three assertions below depend on."""
        variable = self._variable()
        assert np.asarray(variable.isnull().read_array()).sum() == 279
        assert variable.no_data_value[0] is not None

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("clip", lambda v: v.clip(min=0.0)),
            ("round", lambda v: v.round(-1)),
            ("astype", lambda v: v.astype("float32")),
        ],
    )
    def test_the_gaps_survive(self, member: str, call):
        """The same cells read as missing afterwards, and no others.

        Args:
            member: The member under test.
            call: How to call it.
        """
        variable = self._variable()
        before = np.asarray(variable.isnull().read_array())
        after = np.asarray(call(variable).isnull().read_array())
        assert np.array_equal(before, after), (
            f"{member} changed which cells read as gaps: "
            f"{int(before.sum())} -> {int(after.sum())}"
        )

    def test_a_gap_is_in_no_set(self):
        """`isin` flags a gap `0` even when asked for the sentinel itself."""
        variable = self._variable()
        sentinel = float(variable.no_data_value[0])
        flags = np.asarray(variable.isin([sentinel]).read_array())
        assert flags.sum() == 0, f"{int(flags.sum())} gap cells were flagged as data"

    def test_clip_does_not_raise_a_gap_to_the_bound(self):
        """The sentinel is below any bound, and must not be lifted onto it."""
        variable = self._variable()
        clipped = variable.clip(min=0.0)
        gaps = np.asarray(variable.isnull().read_array()).astype(bool)
        values = np.asarray(clipped.read_array(), dtype="float64")
        assert (values[gaps] != 0.0).all(), "a gap was clipped to the lower bound"


class TestAPackedStoreVariable:
    """A CF-packed variable reads physical values, and the members answer in those.

    `z` is stored `float32` with `scale=0.01`, `offset=1.5`, so the cells the member sees
    are the decoded ones. The result carries the decoded values, not the packed ones —
    which is what xarray's decoded arrays hold too.
    """

    @staticmethod
    def _variable() -> NetCDF:
        """The packed store variable.

        Returns:
            NetCDF: The variable.
        """
        store = NetCDF.read_file(str(PACKED_STORE))
        return store.get_variable("z")

    def test_the_fixture_is_packed(self):
        """The precondition: the variable declares CF packing."""
        variable = self._variable()
        assert (variable._scale, variable._offset) == (0.01, 1.5)

    def test_round_answers_in_physical_values(self):
        """`round(1)` rounds what the caller reads, not the stored integers."""
        variable = self._variable()
        physical = np.asarray(variable.read_array(), dtype="float64")
        rounded = np.asarray(variable.round(1).read_array(), dtype="float64")
        assert np.allclose(rounded, np.round(physical, 1))

    def test_clip_answers_in_physical_values(self):
        """A bound is read in the physical units the caller sees."""
        variable = self._variable()
        physical = np.asarray(variable.read_array(), dtype="float64")
        bound = float(np.median(physical))
        clipped = np.asarray(variable.clip(min=bound).read_array(), dtype="float64")
        assert np.allclose(clipped, np.clip(physical, bound, None))


class TestRegappingBandByBand:
    """`_regapped` puts each band's gaps back with that band's own sentinel.

    The members reach it with a list of sentinels, one per band, and a band that declares
    none must be left exactly as the operation left it — there is no value to write there.
    """

    def test_a_band_that_declares_nothing_is_left_alone(self):
        """A `None` entry writes nothing into that band."""
        values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        domain = np.array([[[True, False]], [[False, True]]])
        out = _regapped(values, domain, [-9999.0, None])
        assert out.tolist() == [[[1.0, -9999.0]], [[3.0, 4.0]]]

    def test_a_single_band_with_no_sentinel_is_left_alone(self):
        """The 2-D path, where the one sentinel is `None`."""
        values = np.array([[1.0, 2.0]])
        domain = np.array([[True, False]])
        assert _regapped(values, domain, [None]).tolist() == [[1.0, 2.0]]

    def test_a_short_list_falls_back_to_the_first_sentinel(self):
        """A stack whose sentinel list is shorter than its bands marks the rest alike."""
        values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        domain = np.array([[[True, False]], [[False, True]]])
        assert _regapped(values, domain, [-1.0]).tolist() == [
            [[1.0, -1.0]],
            [[-1.0, 4.0]],
        ]

    def test_a_full_domain_is_returned_untouched(self):
        """With no gap there is nothing to re-mark, and the array is not copied."""
        values = np.array([[1.0, 2.0]])
        domain = np.array([[True, True]])
        assert _regapped(values, domain, [-9999.0]) is values

    def test_declared_gaps_collapses_agreement(self):
        """One sentinel when the bands agree, the list when they do not."""
        assert _declared_gaps([-9999.0, -9999.0]) == -9999.0
        assert _declared_gaps([None, None]) is None
        assert _declared_gaps([-9999.0, -1.0]) == [-9999.0, -1.0]
        assert _declared_gaps([]) == []


class TestAContainerIsRefusedByName:
    """A container has no raster of its own, and the four say so as `where` does."""

    @pytest.mark.parametrize(
        ("member", "arguments"),
        [("clip", (0.0,)), ("round", ()), ("astype", ("float32",)), ("isin", ([0.0],))],
    )
    def test_the_refusal_names_the_member(self, member: str, arguments: tuple):
        """Each names itself and the variable to call it on.

        Args:
            member: The member under test.
            arguments: What to call it with.
        """
        container = NetCDF.read_file(str(STORE))
        call = getattr(container, member)
        with pytest.raises(ValueError, match=rf"^{member}\(\) works on a raster"):
            call(*arguments)


class TestClipRefusesANonNumberBound:
    """A bound is a real number; anything else is refused in words."""

    @pytest.mark.parametrize("bound", ["3", True, complex(1, 2)])
    def test_a_non_number_bound(self, bound):
        """A string, a boolean or a complex number cannot bound a raster.

        Args:
            bound: The bad bound.
        """
        raster = _raster()
        with pytest.raises(TypeError, match="needs a number") as info:
            raster.clip(bound, 6.0)
        assert repr(bound) in str(info.value), (
            f"the refusal should name {bound!r}: {info.value}"
        )

    def test_a_none_lower_bound_is_simply_absent(self):
        """`None` means no bound on that side, not a bad one."""
        clipped = _raster().clip(None, 6.0)
        assert _read(clipped)[-1] == 6.0, (
            f"expected the upper bound only, got {_read(clipped)}"
        )


class TestASentinelAFloatCannotRepresentExactly:
    """A gap is marked by an exact value, so a sentinel that drifts marks the wrong cells.

    `_holds` asked only whether a value was inside the type's range, so a sentinel the type
    rounds to something else was accepted and quietly changed — and the value it landed on
    then read as missing.
    """

    def test_a_sentinel_that_rounds_to_another_value_is_refused(self):
        """`1e-50` is inside float32's range but rounds to `0.0`, which is real data here."""
        raster = _raster(np.array([[0.0, 1e-50, 2.0]]), no_data_value=1e-50)
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("float32")

    def test_float16_refuses_the_default_sentinel(self):
        """`-9999` is inside float16's range but stores as `-10000.0`.

        Test scenario:
            The cast used to declare `-10000.0` as the gap marker, so a real `-10000.0`
            read as missing while the cells marked `-9999` no longer did.
        """
        raster = _raster()
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("float16")

    def test_an_exactly_representable_sentinel_still_passes(self):
        """`-9999` is exact in float32, and that cast is unaffected."""
        cast = _raster().astype("float32")
        assert float(cast.no_data_value[0]) == NDV
        assert np.asarray(cast.isnull().read_array()).sum() == 1


class TestAstypeIntoAFloat:
    """A float target holds any finite sentinel in its range, NaN included."""

    def test_float32_keeps_a_numeric_sentinel(self):
        """`-9999.0` fits `float32`, so it carries over and the gap stays a gap."""
        cast = _raster().astype("float32")
        assert np.asarray(cast.read_array()).dtype == np.float32, (
            "the band type did not change"
        )
        assert cast.no_data_value[0] == NDV, (
            f"the sentinel changed: {cast.no_data_value}"
        )
        assert np.asarray(cast.isnull().read_array()).sum() == 1, "the gap was lost"

    def test_float16_refuses_a_sentinel_past_its_range(self):
        """`float16` tops out at 65504, so `-99999` cannot mark its gaps."""
        raster = _raster(
            np.where(CELLS == NDV, -99999.0, CELLS), no_data_value=-99999.0
        )
        with pytest.raises(ValueError, match="no_data_value"):
            raster.astype("float16")

    def test_a_nan_sentinel_fits_any_float(self):
        """Every float width has NaN, so it is never a reason to refuse."""
        raster = _raster(np.where(CELLS == NDV, np.nan, CELLS), no_data_value=np.nan)
        cast = raster.astype("float32")
        assert np.isnan(cast.no_data_value[0]), (
            f"expected NaN, got {cast.no_data_value}"
        )


class TestHolds:
    """`_holds` decides whether a type can mark a gap with a value."""

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("float32", -9999.0, True),
            ("float32", np.nan, True),
            ("float32", np.inf, True),
            ("float16", -99999.0, False),
            ("float16", 65504.0, True),
            ("float16", -9999.0, False),
            ("float32", 1e-50, False),
            ("float32", 0.1, False),
            ("float64", 0.1, True),
            ("float32", -np.inf, True),
            ("uint8", 255, True),
            ("uint8", 256, False),
            ("uint8", -1, False),
            ("int16", -9999, True),
            ("int16", -9999.5, False),
            ("int16", np.nan, False),
            ("int32", np.inf, False),
        ],
    )
    def test_the_rule(self, dtype: str, value, expected: bool):
        """A float holds a value it can represent **exactly** and every non-finite one; an
        integer only a whole number inside its range.

        Args:
            dtype: The target type.
            value: The candidate sentinel.
            expected: Whether it fits.
        """
        result = _holds(np.dtype(dtype), value)
        assert result is expected, (
            f"_holds({dtype}, {value!r}) gave {result}, expected {expected}"
        )
