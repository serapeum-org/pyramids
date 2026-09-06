"""The exact comparison must not pay for arithmetic it does not use.

`is_stored_no_data` asks for no tolerance at all on an integer band, and that
answer is plain equality. Routing it through `numpy.isclose` computed
`|a - b| <= atol + rtol * |b|` in float64 anyway, materialising two temporaries
the size of the band for a result `==` gives directly -- on a 4000x4000 int32
band, 256 MB of peak memory against 16 MB, and 185 ms against 13 ms. That is
the path `plot_histogram` takes for a whole band by default.

The cost is measured here rather than asserted from the implementation, so the
test fails if the exact branch is removed or bypassed, whatever the reason.
"""

import tracemalloc

import numpy as np
import pytest

from pyramids.base._domain import is_no_data, is_stored_no_data


def _peak_bytes(fn) -> int:
    """Peak allocation of one call, with the warm-up call excluded.

    `tracemalloc.start()` on an already-tracing process does not reset the
    peak, so under `PYTHONTRACEMALLOC=1` the first measurement here inherited
    whatever the interpreter had already accumulated -- tens of megabytes --
    and every ratio below then passed whatever the code did. The tracer is
    reset when it is already running, and left in the state it was found in.

    Args:
        fn: A zero-argument callable to measure.

    Returns:
        int: Peak bytes traced during the measured call.
    """
    fn()
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    if not already_tracing:
        tracemalloc.stop()
    return peak


@pytest.fixture(scope="module")
def integer_band() -> np.ndarray:
    """A band large enough that a float64 temporary is unmissable.

    Returns:
        np.ndarray: A 2000x2000 int32 band holding one sentinel cell.
    """
    band = np.full((2000, 2000), 7, dtype=np.int32)
    band[0, 0] = -9999
    return band


class TestAnIntegerBandIsComparedInItsOwnDtype:
    """The zero-tolerance path must not upcast the band."""

    def test_it_costs_what_equality_costs(self, integer_band: np.ndarray):
        """Peak memory is the mask's, not two float64 copies of the band.

        Args:
            integer_band: A 2000x2000 int32 band.

        Test scenario:
            `arr == sentinel` allocates one boolean mask. `numpy.isclose`
            allocates float64 temporaries the size of the band -- 8 bytes a
            cell against the int32 band's 4 and the mask's 1. Allowing twice
            the equality cost leaves room for an implementation detail while
            still failing on a float64 round trip, which is 16x.
        """
        exact = _peak_bytes(lambda: integer_band == -9999)
        stored = _peak_bytes(
            lambda: is_stored_no_data(integer_band, np.float64(-9999.0))
        )

        assert stored <= 2 * exact, (
            f"is_stored_no_data peaked at {stored / 1e6:.1f} MB where equality peaks at "
            f"{exact / 1e6:.1f} MB; the band is being upcast to float64"
        )


class TestTheExactBranchAnswersTheSameQuestion:
    """Cheaper must not mean different."""

    @pytest.mark.parametrize(
        ("cells", "dtype", "sentinel", "expected"),
        [
            ([1, 2, 3], "int32", 2.0, [False, True, False]),
            ([1, 2, 3], "int32", 2.5, [False, False, False]),
            ([1, 2, 3], "int32", 1e30, [False, False, False]),
            ([254, 255], "uint8", 255.0, [False, True]),
            ([0, 1], "int16", 0.0, [True, False]),
        ],
        ids=["hit", "non-integral", "out-of-range", "byte-max", "zero"],
    )
    def test_an_integer_sentinel_the_dtype_cannot_hold_matches_nothing(
        self, cells: list[int], dtype: str, sentinel: float, expected: list[bool]
    ):
        """A sentinel outside the dtype is absent, not approximately present.

        Args:
            cells: The band's cells.
            dtype: The band's integer dtype.
            sentinel: The declared no-data value.
            expected: The mask each cell should get.

        Test scenario:
            `2.5` and `1e30` cannot be stored in an int32 band, so no cell can
            hold them. Casting them into the dtype first would wrap or
            truncate and mark an unrelated cell; the answer is that nothing
            matches.
        """
        band = np.array(cells, dtype=dtype)

        assert is_stored_no_data(band, sentinel).tolist() == expected, (
            f"{sentinel!r} against a {dtype} band gave the wrong mask"
        )

    def test_an_int64_sentinel_above_2_53_is_not_rounded_into_a_neighbour(self):
        """float64 arithmetic conflated two distinct int64 values.

        Test scenario:
            `9007199254740992` and `9007199254740993` are one apart and both
            round to the former as doubles, so a comparison that goes through
            float64 calls the wrong cell no-data. Compared as int64 they are
            simply different.
        """
        band = np.array([9007199254740992, 9007199254740993], dtype=np.int64)

        mask = is_stored_no_data(band, 9007199254740993)

        assert mask.tolist() == [False, True], (
            f"an int64 sentinel one ULP from its neighbour was rounded into it: {mask.tolist()}"
        )

    def test_a_tolerant_call_still_goes_through_isclose(self):
        """The fast path must not capture callers that asked for a window.

        Test scenario:
            A non-zero `rtol` is a request for a window, and `is_no_data`'s
            default is one. `-9998.95` is inside a `1e-5` window around
            `-9999` and outside an exact comparison, so it tells the two
            branches apart.
        """
        band = np.array([-9999.0, -9998.95, 1.0])

        assert is_no_data(band, -9999.0, rtol=0.00001).tolist() == [
            True,
            True,
            False,
        ], "the tolerant branch stopped honouring rtol"
        assert is_no_data(band, -9999.0, rtol=0.0, atol=0.0).tolist() == [
            True,
            False,
            False,
        ], "the exact branch admitted a cell that is merely near the sentinel"

    def test_a_scalar_still_answers_as_a_boolean(self):
        """The scalar overload is part of the contract.

        Test scenario:
            `is_no_data` is documented to take a scalar and return something
            that behaves as a `bool`. The exact branch has to preserve that;
            returning a 0-d array or a Python `int` would break callers that
            branch on it.
        """
        hit = is_stored_no_data(np.int32(5), 5.0)
        miss = is_stored_no_data(np.int32(5), 6.0)

        assert bool(hit) is True, f"a matching scalar answered {hit!r}"
        assert bool(miss) is False, f"a non-matching scalar answered {miss!r}"


class TestASentinelTooLargeForFloat64ToHold:
    """The two largest integer sentinels are the ones a float round trip loses."""

    @pytest.mark.parametrize(
        ("dtype", "sentinel"),
        [("int64", 2**63 - 1), ("uint64", 2**64 - 1), ("uint64", 2**64 - 2)],
        ids=["int64-max", "uint64-max", "uint64-max-minus-one"],
    )
    def test_it_is_still_found_in_the_band(self, dtype: str, sentinel: int):
        """A 64-bit sentinel must not vanish on the way through the range test.

        Args:
            dtype: A 64-bit integer band dtype.
            sentinel: A value near that dtype's limit.

        Test scenario:
            The range test used to ask `iinfo.min <= float(sentinel) <=
            iinfo.max`, and `float(2**63 - 1)` rounds *up* to
            9223372036854775808.0 -- past the very limit it is compared
            against. The band was therefore judged unable to hold its own
            maximum and the mask came back all-False. These are not exotic
            values: `_coerce_band_no_data` and `_fallback_no_data` fabricate
            `np.uint64(2**64 - 1)` themselves.
        """
        band = np.array([sentinel, 7], dtype=dtype)

        assert is_stored_no_data(band, sentinel).tolist() == [True, False], (
            f"a {dtype} band did not find its own sentinel {sentinel}"
        )

    def test_the_cells_are_not_overwritten_by_fill(self):
        """The consequence, which is silent data loss.

        Test scenario:
            `fill` replaces the cells the predicate calls no-data. With the mask
            all-False the caller got the opposite of what they asked for: every
            cell replaced, including the real data the sentinel existed to
            distinguish.
        """
        sentinel = 2**63 - 1
        band = np.array([[sentinel, 7], [8, sentinel]], dtype="int64")

        found = int(is_stored_no_data(band, sentinel).sum())

        assert found == 2, (
            f"{found} of the 2 sentinel cells were found; `fill` would have "
            "overwritten the 2 real values instead of the sentinels"
        )


class TestAFloatBandsSlackIsNarrowerThanItsOwnUlp:
    """A tolerance of one ULP cannot tell neighbouring values apart."""

    def test_a_float16_sentinel_does_not_swallow_its_neighbours(self):
        """`max(eps(dtype), single_eps)` picked a window wider than the spacing.

        Test scenario:
            Float16's `eps` is 9.77e-4, so around a -10000 sentinel the window
            was about +/-9.8 -- and the neighbouring representable float16
            values are 8 away. Both were reported as no-data, which is the very
            failure the dtype-derived tolerance was introduced to end, in the
            one dtype where it was applied backwards.
        """
        band = np.array([-10000.0, -9992.0, -10008.0, 1.0], dtype="float16")

        mask = is_stored_no_data(band, -10000.0).tolist()

        assert mask == [True, False, False, False], (
            f"a float16 band masked its sentinel's neighbours: {mask}"
        )

    def test_a_sentinel_that_went_through_single_precision_is_still_found(self):
        """The slack is not zero either, and this is what it is for.

        Test scenario:
            A float32 band's sentinel comes back from the driver as a C double,
            a few ULP off the stored value. Comparing exactly would miss it, so
            the float branch keeps single precision's eps -- which this pins,
            since setting the float tolerance to zero passes every other test
            in this module.
        """
        band = np.array([np.float32(1e30), np.float32(2.0)], dtype="float32")
        drifted = float(np.float64(np.float32(1e30))) * (1 + 3e-8)

        assert is_stored_no_data(band, drifted).tolist() == [True, False], (
            "a sentinel a few ULP off the stored value was not matched; the "
            "float slack has been removed"
        )
