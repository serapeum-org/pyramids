"""Unit tests for :meth:`pyramids.netcdf.NetCDF._derive_primary_band_view` (STR-2).

The legacy ``(_band_dim_name, _band_dim_values)`` pair is a derived *view* of the
canonical ``_band_dim_names`` / ``_band_dim_values_map`` / ``_band_dim_sizes`` fields.
``_derive_primary_band_view`` is the single source of truth for that derivation (it
replaced the per-call-site reconciliation in the wrap / build / ``sel`` paths). These
pin its staleness semantics directly, without heavy file I/O.
"""

from __future__ import annotations

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


class TestDerivePrimaryBandView:
    """The staticmethod deriving the legacy primary-band-dim view from canonical state."""

    def test_no_band_dims_returns_none_pair(self):
        """Empty ``names`` yields ``(None, None)`` regardless of the other args.

        Test scenario:
            A variable with no non-spatial dimension has no primary band dim, so both
            the legacy name and values are ``None``.
        """
        name, values = NetCDF._derive_primary_band_view((), {}, (), 0)
        assert name is None, f"expected no name, got {name!r}"
        assert values is None, f"expected no values, got {values!r}"

    def test_single_dim_valid_exposes_values(self):
        """A single band dim whose values length matches the band count exposes them.

        Test scenario:
            ``names=("time",)``, four coords, ``band_count=4`` — the primary view is
            current, so the coords come through unchanged.
        """
        name, values = NetCDF._derive_primary_band_view(
            ("time",), {"time": [0, 6, 12, 18]}, (4,), 4
        )
        assert name == "time", f"expected primary name 'time', got {name!r}"
        assert values == [0, 6, 12, 18], f"expected the coords, got {values!r}"

    def test_single_dim_stale_length_nulls_values(self):
        """A single band dim whose values length diverges from the band count is nulled.

        Test scenario:
            Four cached coords but ``band_count=2`` (a band-shrinking op left the view
            stale) — the name survives but the stale coords are dropped.
        """
        name, values = NetCDF._derive_primary_band_view(
            ("time",), {"time": [0, 6, 12, 18]}, (4,), 2
        )
        assert name == "time", f"name should survive, got {name!r}"
        assert values is None, f"stale coords should be nulled, got {values!r}"

    def test_multi_dim_valid_exposes_primary_values(self):
        """Multi-band-dim: primary values survive when prod(sizes) equals the band count.

        Test scenario:
            ``names=("level","time")`` with sizes ``(3, 4)`` → product 12, ``band_count=12``;
            the primary (``level``) coords are exposed unchanged.
        """
        name, values = NetCDF._derive_primary_band_view(
            ("level", "time"),
            {"level": [1000, 850, 500], "time": [0, 6, 12, 18]},
            (3, 4),
            12,
        )
        assert name == "level", f"expected primary 'level', got {name!r}"
        assert values == [1000, 850, 500], f"expected primary coords, got {values!r}"

    def test_multi_dim_stale_product_nulls_values(self):
        """Multi-band-dim: primary values are nulled when prod(sizes) != band count.

        Test scenario:
            sizes ``(3, 4)`` → product 12 but ``band_count=6`` (total band count diverged
            from the cached sizes), so the now-stale primary view is dropped.
        """
        name, values = NetCDF._derive_primary_band_view(
            ("level", "time"),
            {"level": [1000, 850, 500], "time": [0, 6, 12, 18]},
            (3, 4),
            6,
        )
        assert name == "level", f"name should survive, got {name!r}"
        assert values is None, f"stale multi-dim view should be nulled, got {values!r}"

    def test_zero_band_count_exposes_values_unvalidated(self):
        """A non-positive band count cannot validate, so the cached values pass through.

        Test scenario:
            ``band_count=0`` (raster not yet/ no longer reporting bands) — the helper
            cannot prove staleness, so it returns the cached primary coords as-is.
        """
        name, values = NetCDF._derive_primary_band_view(
            ("time",), {"time": [0, 6, 12, 18]}, (4,), 0
        )
        assert name == "time", f"expected 'time', got {name!r}"
        assert values == [0, 6, 12, 18], f"expected pass-through coords, got {values!r}"

    def test_missing_map_entry_returns_none_values(self):
        """A primary dim with no coordinate entry yields ``None`` values.

        Test scenario:
            ``names=("time",)`` but the map has no ``time`` key (e.g. a non-indexed
            dimension) — the name survives, the values are ``None``.
        """
        name, values = NetCDF._derive_primary_band_view(("time",), {}, (4,), 4)
        assert name == "time", f"expected 'time', got {name!r}"
        assert values is None, f"absent coords should be None, got {values!r}"
