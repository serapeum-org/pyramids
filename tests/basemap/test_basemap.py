"""Tests for pyramids.basemap.basemap.

``add_basemap`` and ``get_provider`` are thin wrappers over
``cleopatra.tiles.add_tiles`` / ``cleopatra.tiles.get_provider`` (the
cleopatra C-6 helpers). These tests cover the delegation contract: the
right cleopatra function is called with the translated kwargs, its return
value is propagated, the missing-extra error path is wired up, and the
public signature has not drifted (downstream code patches against it).
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.plot

pytest.importorskip("cleopatra.tiles", reason="cleopatra[tiles] extra not installed")

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.basemap.basemap import add_basemap, get_provider


class TestGetProvider:
    """``get_provider`` delegates to ``cleopatra.tiles.get_provider``."""

    def test_delegates_to_cleopatra_get_provider(self):
        """The ``name`` argument is forwarded and the result returned verbatim.

        Test scenario:
            Patch ``cleopatra.tiles.get_provider`` with a sentinel and
            confirm ``get_provider("CartoDB.Positron")`` calls it once
            with that name and hands back its return value.
        """
        with patch("cleopatra.tiles.get_provider") as mock_get:
            result = get_provider("CartoDB.Positron")

        mock_get.assert_called_once_with("CartoDB.Positron")
        assert (
            result is mock_get.return_value
        ), "get_provider must return cleopatra.tiles.get_provider's result"

    def test_default_provider_round_trip(self):
        """A no-arg call resolves the real default provider through cleopatra.

        Test scenario:
            With cleopatra installed, ``get_provider()`` should resolve
            the OpenStreetMap.Mapnik default — a smoke test that the
            wrapper does not mangle the call.
        """
        provider = get_provider()
        name = getattr(provider, "name", "") or str(provider)
        assert (
            "openstreetmap" in name.lower()
        ), f"default provider should be OpenStreetMap, got {provider!r}"

    def test_missing_extra_raises(self):
        """When the cleopatra ``[tiles]`` extra is absent the guard fires.

        Test scenario:
            Patch the ``import_basemap`` guard to raise
            ``OptionalPackageDoesNotExist`` and confirm ``get_provider``
            surfaces it (rather than an opaque ``ImportError``).
        """
        with patch(
            "pyramids.basemap.basemap.import_basemap",
            side_effect=OptionalPackageDoesNotExist("no cleopatra[tiles]"),
        ):
            with pytest.raises(OptionalPackageDoesNotExist):
                get_provider("CartoDB.Positron")


class TestAddBasemap:
    """``add_basemap`` delegates to ``cleopatra.tiles.add_tiles``."""

    def test_delegates_with_default_kwargs(self):
        """A bare ``add_basemap(ax)`` forwards the documented defaults.

        Test scenario:
            Patch ``cleopatra.tiles.add_tiles`` and call ``add_basemap``
            with only ``ax``. Every wrapper-level default must reach
            cleopatra as a keyword (``crs=3857``, ``zoom="auto"``, …) and
            the axes must be passed positionally.
        """
        sentinel_ax = object()
        with patch("cleopatra.tiles.add_tiles") as mock_add:
            result = add_basemap(sentinel_ax)

        mock_add.assert_called_once_with(
            sentinel_ax,
            source=None,
            crs=3857,
            zoom="auto",
            alpha=1.0,
            attribution=True,
            zorder=-1,
            interpolation="bilinear",
            timeout=10,
            retries=2,
        )
        assert (
            result is mock_add.return_value
        ), "add_basemap must return cleopatra.tiles.add_tiles's result"

    def test_delegates_with_custom_kwargs(self):
        """Caller-supplied options are forwarded to cleopatra unchanged.

        Test scenario:
            Pass a non-default ``crs``, ``source``, ``alpha``, and
            ``zorder``; the patched ``add_tiles`` must receive exactly
            those values (and the untouched defaults for the rest).
        """
        sentinel_ax = object()
        with patch("cleopatra.tiles.add_tiles") as mock_add:
            add_basemap(
                sentinel_ax,
                crs=4326,
                source="CartoDB.Positron",
                alpha=0.5,
                zorder=-2,
            )

        mock_add.assert_called_once_with(
            sentinel_ax,
            source="CartoDB.Positron",
            crs=4326,
            zoom="auto",
            alpha=0.5,
            attribution=True,
            zorder=-2,
            interpolation="bilinear",
            timeout=10,
            retries=2,
        )

    def test_missing_extra_raises(self):
        """``add_basemap`` surfaces the missing-extra error before delegating.

        Test scenario:
            Patch the ``import_basemap`` guard to raise; ``add_basemap``
            must propagate ``OptionalPackageDoesNotExist`` and never reach
            ``cleopatra.tiles.add_tiles``.
        """
        with patch(
            "pyramids.basemap.basemap.import_basemap",
            side_effect=OptionalPackageDoesNotExist("no cleopatra[tiles]"),
        ):
            with patch("cleopatra.tiles.add_tiles") as mock_add:
                with pytest.raises(OptionalPackageDoesNotExist):
                    add_basemap(object())
        mock_add.assert_not_called()


class TestCleopatraDelegation:
    """The ``add_basemap`` -> ``cleopatra.tiles.add_tiles`` contract.

    ``add_basemap`` is a thin wrapper over ``cleopatra.tiles.add_tiles``
    (shipped in ``cleopatra >= 0.8.0``, pinned via the ``[viz]`` extra as
    ``cleopatra[tiles]``). These tests pin the contract that wrapper relies
    on: the helper is importable, and the two signatures stay compatible —
    cleopatra may add new *optional* params (pyramids just won't expose
    them), but a previously-shared param going *required* upstream, or a
    pyramids-only param, would break the delegation.
    """

    def test_cleopatra_add_tiles_is_importable(self):
        """``cleopatra.tiles.add_tiles`` imports (C-6 contract)."""
        cleopatra_tiles = pytest.importorskip(
            "cleopatra.tiles",
            reason="cleopatra[tiles] extra not installed",
        )
        assert hasattr(cleopatra_tiles, "add_tiles"), (
            "cleopatra.tiles.add_tiles must exist so "
            "pyramids.basemap.add_basemap can delegate to it."
        )

    def test_pyramids_and_cleopatra_share_addmap_signature(self):
        """``add_basemap`` and ``cleopatra.tiles.add_tiles`` line up.

        Test scenario:
            Compare the parameter names directly so a future cleopatra
            release that drifts the signature gets caught here rather than
            at a downstream call site.
        """
        cleopatra_tiles = pytest.importorskip(
            "cleopatra.tiles",
            reason="cleopatra[tiles] extra not installed",
        )
        cleo_sig = inspect.signature(cleopatra_tiles.add_tiles)
        pyr_sig = inspect.signature(add_basemap)
        cleo_params = set(cleo_sig.parameters) - {"ax"}
        pyr_params = set(pyr_sig.parameters) - {"ax"}
        missing = cleo_params - pyr_params
        extra = pyr_params - cleo_params
        # Cleopatra is allowed to grow its surface: new *optional* params
        # (e.g. `user_agent=` / `max_tiles=` in 0.8.0) don't break the
        # `add_basemap(ax, ...) -> cleopatra.tiles.add_tiles(ax, ...)`
        # delegation — pyramids just won't expose them. What *would* break
        # it is cleopatra making a previously-shared (or any) param
        # *required* while pyramids lacks it.
        for name in missing:
            param = cleo_sig.parameters[name]
            assert param.default is not inspect.Parameter.empty, (
                f"cleopatra.tiles.add_tiles added a *required* param {name!r} "
                "not in pyramids.add_basemap — the delegation contract is broken."
            )
        assert not extra, (
            f"pyramids.add_basemap has params not in cleopatra.tiles.add_tiles: "
            f"{extra}. The thin-wrapper delegation will not work until "
            "these are pushed upstream."
        )


class TestAddBasemapSignatureStability:
    """Assert ``add_basemap``'s public signature has not drifted.

    Tests across the suite (and ``pyramids.dataset._plot_helpers``) call
    ``add_basemap`` with positional / keyword combinations. Any signature
    change here breaks those silently — pin the parameter set so a future
    drift surfaces with a clear failure.
    """

    EXPECTED_PARAMS = (
        "ax",
        "crs",
        "source",
        "zoom",
        "alpha",
        "attribution",
        "zorder",
        "interpolation",
        "timeout",
        "retries",
    )

    def test_add_basemap_parameter_names(self):
        """Parameter names (and order) match the documented contract.

        Test scenario:
            Compare ``add_basemap``'s parameter names against the frozen
            ``EXPECTED_PARAMS``. Order matters because most call sites pass
            ``ax`` positionally — a rename or reorder must surface here.
        """
        params = tuple(inspect.signature(add_basemap).parameters.keys())
        assert params == self.EXPECTED_PARAMS, (
            f"add_basemap signature changed; expected {self.EXPECTED_PARAMS}, "
            f"got {params}."
        )
