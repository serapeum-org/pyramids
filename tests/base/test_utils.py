"""Tests for helpers in :mod:`pyramids.base._utils`.

Covers :func:`lazy_extra_hint` (the single source of the optional ``[lazy]`` extra
install hint) and :func:`apply_unpack` (the shared scale/offset primitive behind both
the NetCDF CF unpack path and the raster ``read_array(scaled=True)`` path).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._utils import apply_unpack, lazy_extra_hint

pytestmark = pytest.mark.core


class TestApplyUnpack:
    """Tests for the shared scale/offset primitive ``apply_unpack``."""

    def test_none_none_passthrough(self):
        """Both None returns the array unchanged, with no float promotion."""
        arr = np.array([0, 1, 2], dtype="int16")
        out = apply_unpack(arr, None, None)
        assert out is arr, "identity: the same array object is returned"
        assert out.dtype == np.int16, "no float promotion when nothing is declared"

    def test_scalar_scale_and_offset(self):
        """A scalar scale/offset applies as float64."""
        out = apply_unpack(np.array([0, 1, 2]), 0.1, 5.0)
        assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
        np.testing.assert_allclose(out, [5.0, 5.1, 5.2])

    def test_scale_only_and_offset_only(self):
        """Scale-only and offset-only each apply their single operand."""
        np.testing.assert_allclose(
            apply_unpack(np.array([1, 2]), 2.0, None), [2.0, 4.0]
        )
        np.testing.assert_allclose(
            apply_unpack(np.array([1, 2]), None, 3.0), [4.0, 5.0]
        )

    def test_ndarray_broadcast(self):
        """A per-band (bands, 1, 1) scale/offset broadcasts over a 3-D array."""
        arr = np.ones((2, 1, 1))
        scale = np.array([2.0, 3.0]).reshape(-1, 1, 1)
        offset = np.array([1.0, -1.0]).reshape(-1, 1, 1)
        out = apply_unpack(arr, scale, offset)
        np.testing.assert_allclose(out.ravel(), [3.0, 2.0])

    def test_masked_array_mask_preserved(self):
        """A masked-array input keeps its mask across the transform."""
        arr = np.ma.MaskedArray([0, 1, 2], mask=[False, True, False])
        out = apply_unpack(arr, 0.1, 5.0)
        assert isinstance(out, np.ma.MaskedArray), "mask must survive"
        np.testing.assert_array_equal(out.mask, [False, True, False])

    def test_reexported_identity(self):
        """The NetCDF module re-exports the same object (one shared primitive)."""
        from pyramids.netcdf._lazy import _apply_unpack
        from pyramids.netcdf._lazy import apply_unpack as lazy_fn

        assert apply_unpack is lazy_fn is _apply_unpack, "one shared primitive"


class TestLazyExtraHint:
    """Tests for ``lazy_extra_hint``."""

    def test_starts_with_prefix(self):
        """The composed hint begins with the caller's prefix sentence.

        Test scenario:
            A domain-specific prefix is preserved verbatim at the start of the
            message so each call site keeps its own actionable lead sentence.
        """
        prefix = "Zarr IO requires the optional 'dask' / 'zarr' dependencies."
        message = lazy_extra_hint(prefix)
        assert message.startswith(prefix), f"prefix not preserved: {message!r}"

    def test_includes_both_install_commands(self):
        """The hint lists both the PyPI and conda-forge install commands.

        Test scenario:
            Every composed hint must carry the PyPI extra install and the
            conda-forge metapackage install so users on either toolchain get an
            actionable command.
        """
        message = lazy_extra_hint("X requires the optional 'zarr' dependency.")
        assert "pip install 'pyramids-gis[lazy]'" in message, (
            f"PyPI command missing: {message!r}"
        )
        assert "conda install -c conda-forge pyramids-lazy" in message, (
            f"conda-forge command missing: {message!r}"
        )

    def test_exact_format(self):
        """The composed hint matches the documented stacked-list layout exactly.

        Test scenario:
            ``prefix`` + a space + "Install with one of:" then two indented
            bullet lines (PyPI, conda-forge), so callers can rely on the precise
            text. Guards against accidental spacing/format drift.
        """
        prefix = "Op requires the optional 'dask' dependency."
        expected = (
            f"{prefix} Install with one of:\n"
            "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
            "  - conda-forge: conda install -c conda-forge pyramids-lazy"
        )
        assert lazy_extra_hint(prefix) == expected, (
            f"format drift: {lazy_extra_hint(prefix)!r}"
        )

    def test_no_double_space_after_prefix(self):
        """A prefix ending in a period yields a single space before the body.

        Test scenario:
            The composition inserts exactly one space between the prefix's
            trailing period and "Install", never a double space.
        """
        message = lazy_extra_hint("Needs the optional 'dask' dependency.")
        assert ". Install with one of:" in message, (
            f"unexpected spacing before body: {message!r}"
        )
        assert ".  Install" not in message, f"double space after prefix: {message!r}"
