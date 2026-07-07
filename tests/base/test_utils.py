"""Tests for helpers in :mod:`pyramids.base._utils`.

Currently covers :func:`lazy_extra_hint`, the single source of the optional
``[lazy]`` extra install hint reused by the zarr / dask call sites.
"""

from __future__ import annotations

import pytest

from pyramids.base._utils import lazy_extra_hint

pytestmark = pytest.mark.core


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
        assert (
            "pip install 'pyramids-gis[lazy]'" in message
        ), f"PyPI command missing: {message!r}"
        assert (
            "conda install -c conda-forge pyramids-lazy" in message
        ), f"conda-forge command missing: {message!r}"

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
        assert (
            lazy_extra_hint(prefix) == expected
        ), f"format drift: {lazy_extra_hint(prefix)!r}"

    def test_no_double_space_after_prefix(self):
        """A prefix ending in a period yields a single space before the body.

        Test scenario:
            The composition inserts exactly one space between the prefix's
            trailing period and "Install", never a double space.
        """
        message = lazy_extra_hint("Needs the optional 'dask' dependency.")
        assert (
            ". Install with one of:" in message
        ), f"unexpected spacing before body: {message!r}"
        assert ".  Install" not in message, f"double space after prefix: {message!r}"
