"""Tests for the ``ReadRequest`` option value-object.

Covers ``ReadRequest.__post_init__`` — the option-incompatibility matrix that
``read_array`` runs before resolving its window: each pairing that is never legal
regardless of read path, the exact exception *type* each raises
(``ValueError`` vs ``NotImplementedError``), the message text several read tests
assert on, the valid combinations that must pass, and the check ordering that
decides which error a multiply-invalid request raises.
"""

from __future__ import annotations

import dataclasses

import pytest

from pyramids.dataset.engines._read_request import ReadRequest

pytestmark = pytest.mark.core


def make_request(**overrides) -> ReadRequest:
    """Build a valid ``ReadRequest``, overriding individual option fields.

    Args:
        **overrides: Field values to replace on the otherwise-valid baseline
            (a plain full eager read of band 0).

    Returns:
        ReadRequest: A constructed request (its ``__post_init__`` has run).
    """
    base = {
        "band": 0,
        "chunks": None,
        "lock": None,
        "out_shape": None,
        "resampling": "nearest",
        "boundless": False,
        "fill_value": None,
        "masked": False,
        "threadsafe": False,
    }
    base.update(overrides)
    return ReadRequest(**base)


class TestReadRequest:
    """``ReadRequest`` construction and the ``__post_init__`` matrix."""

    def test_valid_default_request_constructs(self):
        """A plain eager read passes the matrix and keeps its field values.

        Test scenario:
            The baseline request (no special options) constructs and exposes the
            options it was given.
        """
        req = make_request(band=2)
        assert req.band == 2, f"band should round-trip, got {req.band}"
        assert req.chunks is None, "chunks should default to None"
        assert req.resampling == "nearest", "resampling should round-trip"
        assert req.boundless is False, "boundless should round-trip"

    @pytest.mark.parametrize(
        "resampling",
        ["nearest", "NEAREST", " nearest ", "Nearest"],
        ids=["lower", "upper", "padded", "title"],
    )
    def test_nearest_resampling_without_out_shape_is_allowed(self, resampling):
        """``resampling`` naming ``nearest`` (any case/whitespace) needs no out_shape.

        Args:
            resampling: A spelling of ``nearest`` that must be treated as the
                default and therefore not require ``out_shape``.

        Test scenario:
            Guard 6 compares ``resampling.strip().lower()`` to ``"nearest"``; all
            these spellings are the default and must not raise.
        """
        req = make_request(resampling=resampling)
        assert req.resampling == resampling, "resampling value should round-trip"

    def test_is_frozen(self):
        """``ReadRequest`` is immutable; assigning a field raises.

        Test scenario:
            A frozen dataclass rejects attribute assignment.
        """
        req = make_request()
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.boundless = True  # type: ignore[misc]

    def test_fill_value_without_boundless_raises(self):
        """``fill_value`` outside a boundless read is a ``ValueError``.

        Test scenario:
            ``fill_value`` set with ``boundless=False`` → guard 1.
        """
        with pytest.raises(ValueError, match="fill_value") as exc:
            make_request(fill_value=5.0, boundless=False)
        assert "boundless" in str(exc.value), f"message should mention boundless: {exc.value}"

    def test_fill_value_with_boundless_is_allowed(self):
        """``fill_value`` with ``boundless=True`` passes the matrix.

        Test scenario:
            A boundless request may carry a fill value.
        """
        req = make_request(fill_value=5.0, boundless=True)
        assert req.fill_value == 5.0, "fill_value should round-trip on a boundless read"

    def test_boundless_with_chunks_raises_value_error(self):
        """``boundless`` + ``chunks`` is a ``ValueError`` (guard 2).

        Test scenario:
            Boundless fills apply to eager windowed reads only.
        """
        with pytest.raises(ValueError, match="chunks") as exc:
            make_request(boundless=True, chunks=4)
        assert "boundless" in str(exc.value), f"message should mention boundless: {exc.value}"

    def test_boundless_with_out_shape_raises_not_implemented(self):
        """``boundless`` + ``out_shape`` is a ``NotImplementedError`` (guard 3).

        Test scenario:
            Decimated boundless reads are not combined yet.
        """
        with pytest.raises(NotImplementedError, match="out_shape"):
            make_request(boundless=True, out_shape=(4, 4))

    def test_boundless_with_threadsafe_raises_not_implemented(self):
        """``boundless`` + ``threadsafe`` is a ``NotImplementedError`` (guard 4).

        Test scenario:
            The boundless read uses the shared handle, defeating isolation.
        """
        with pytest.raises(NotImplementedError, match="threadsafe=True"):
            make_request(boundless=True, threadsafe=True)

    def test_out_shape_with_threadsafe_raises_not_implemented(self):
        """``out_shape`` + ``threadsafe`` is a ``NotImplementedError`` (guard 5).

        Test scenario:
            The decimated read uses the shared handle, defeating isolation.
        """
        with pytest.raises(NotImplementedError, match="threadsafe=True"):
            make_request(out_shape=(4, 4), threadsafe=True)

    def test_resampling_without_out_shape_raises_value_error(self):
        """A non-nearest ``resampling`` without ``out_shape`` is a ``ValueError`` (guard 6).

        Test scenario:
            ``resampling`` only applies to decimated (``out_shape``) reads.
        """
        with pytest.raises(ValueError, match="resampling") as exc:
            make_request(resampling="bilinear", out_shape=None)
        assert "out_shape" in str(exc.value), f"message should mention out_shape: {exc.value}"

    def test_resampling_with_out_shape_is_allowed(self):
        """A non-nearest ``resampling`` with ``out_shape`` passes the matrix.

        Test scenario:
            ``out_shape`` present → ``resampling`` is meaningful and allowed.
        """
        req = make_request(resampling="bilinear", out_shape=(4, 4))
        assert req.resampling == "bilinear", "resampling should round-trip with out_shape"

    def test_ordering_boundless_out_shape_beats_threadsafe(self):
        """A multiply-invalid request raises the earliest-checked guard.

        Test scenario:
            ``boundless`` + ``out_shape`` + ``threadsafe`` trips guard 3
            (out_shape) before guard 4 (threadsafe), so the ``out_shape`` message
            wins — matching the original inline check order.
        """
        with pytest.raises(NotImplementedError, match="out_shape") as exc:
            make_request(boundless=True, out_shape=(4, 4), threadsafe=True)
        assert "threadsafe" not in str(exc.value) or "out_shape" in str(exc.value), (
            f"guard 3 (out_shape) must fire before guard 4 (threadsafe): {exc.value}"
        )

    def test_ordering_fill_value_beats_resampling(self):
        """``fill_value`` (guard 1) fires before ``resampling`` (guard 6).

        Test scenario:
            ``fill_value`` set (no boundless) + a non-nearest ``resampling`` (no
            out_shape) raises the ``fill_value`` ValueError, not the resampling one.
        """
        with pytest.raises(ValueError, match="fill_value"):
            make_request(fill_value=1.0, boundless=False, resampling="cubic")
