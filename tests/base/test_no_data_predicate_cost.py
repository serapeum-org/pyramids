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

    Args:
        fn: A zero-argument callable to measure.

    Returns:
        int: Peak bytes traced during the measured call.
    """
    fn()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
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
        stored = _peak_bytes(lambda: is_stored_no_data(integer_band, np.float64(-9999.0)))

        assert stored <= 2 * exact, (
            f"is_stored_no_data peaked at {stored / 1e6:.1f} MB where equality peaks at "
            f"{exact / 1e6:.1f} MB; the band is being upcast to float64"
        )

    def test_a_float64_sentinel_does_not_promote_it(self, integer_band: np.ndarray):
        """The sentinel is a C double, and that must not decide the dtype.

        Args:
            integer_band: A 2000x2000 int32 band.

        Test scenario:
            Under NEP 50 a `np.float64` scalar is strong, so `arr == sentinel`
            would promote the whole band to float64 and pay the cost equality
            was supposed to avoid. `Dataset.no_data_value` hands back exactly
            such a scalar, so this is the ordinary case, not a corner.
        """
        weak = _peak_bytes(lambda: is_stored_no_data(integer_band, -9999.0))
        strong = _peak_bytes(lambda: is_stored_no_data(integer_band, np.float64(-9999.0)))

        assert strong <= 2 * weak, (
            f"a np.float64 sentinel peaked at {strong / 1e6:.1f} MB against "
            f"{weak / 1e6:.1f} MB for a Python float; it is promoting the band"
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

        assert is_no_data(band, -9999.0, rtol=0.00001).tolist() == [True, True, False], (
            "the tolerant branch stopped honouring rtol"
        )
        assert is_no_data(band, -9999.0, rtol=0.0, atol=0.0).tolist() == [True, False, False], (
            "the exact branch admitted a cell that is merely near the sentinel"
        )

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
