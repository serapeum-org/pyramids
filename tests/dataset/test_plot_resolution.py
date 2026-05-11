"""Pure-logic tests for the post-PR-1 plot band-resolution policy.

These tests do **not** import :mod:`cleopatra` or trigger the matplotlib
backend, so they run in the main ``pixi run -e dev main`` suite (i.e.
under the ``-m "not plot"`` filter). Tests that exercise the actual
rendering path live in :mod:`tests.dataset.test_plot` and are gated on
the ``plot`` marker.

The split is deliberate: the ``_resolve_plot_band`` policy is the part
of PR-1 most likely to regress quietly (it's a 30-line heuristic with
five branches), so we want it covered on every CI run, not only on the
``plot`` job.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.abstract_dataset import RasterBase
from pyramids.netcdf.netcdf import NetCDF


def _make_nc_subset(n_bands: int) -> NetCDF:
    """Build an in-memory NetCDF variable subset with ``n_bands`` bands.

    Args:
        n_bands: Number of bands (i.e. extra-dim length) on the subset.

    Returns:
        NetCDF: A variable subset (``_is_subset == True``) with the
            requested ``band_count``.
    """
    rng = np.random.default_rng(42)
    arr = rng.random((n_bands, 5, 6)).astype("float32")
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(30.0, 0.5, 0, 35.0, 0, -0.5),
        variable_name="t2m",
        path=None,
        extra_dim_name="time",
        extra_dim_values=list(range(n_bands)),
    )
    return nc.get_variable("t2m")


class TestResolvePlotBandPolicy:
    """Direct unit tests for :meth:`Dataset._resolve_plot_band`.

    The resolver is the per-class band-resolution policy that PR-1
    moved out of the generic ``Analysis.plot`` engine. These tests
    pin every branch of the policy so the documented contract is
    enforced even when the cleopatra extra is not installed.

    See also:
        ``tests/dataset/test_plot.py::TestResolvePlotBand`` — the
        original PR-1 cases (cleopatra-marked, exercising the same
        rules end-to-end through ``Dataset.plot``). This class is the
        wider parametrised matrix; the overlap is intentional.
    """

    def test_explicit_band_pass_through_no_rgb(self):
        """Explicit ``band=N`` returns N unchanged with ``rgb=None``.

        Test scenario:
            User passes ``band=3``. The resolver must not consult the
            colour tags or band count — it returns ``(3, None)``.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((4, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=3, rgb=None)
        assert resolved_band == 3, f"Expected band 3, got {resolved_band}"
        assert resolved_rgb is None, f"rgb should remain None, got {resolved_rgb}"

    def test_explicit_band_overrides_explicit_rgb(self):
        """When both ``band`` and ``rgb`` are supplied, both pass through.

        Test scenario:
            ``band=5`` and ``rgb=[0, 1, 2]`` together. The resolver
            forwards both — it never re-derives the band from rgb[0]
            once an explicit band is provided.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        resolved_band, resolved_rgb = dataset._resolve_plot_band(
            band=5, rgb=[0, 1, 2]
        )
        assert resolved_band == 5
        assert resolved_rgb == [0, 1, 2]

    @pytest.mark.parametrize("n_bands", [1, 2])
    def test_band_count_below_three_defaults_to_zero(self, n_bands):
        """``band_count < 3`` always yields band 0.

        Args:
            n_bands: Either 1 or 2 — both must short-circuit to band 0.

        Test scenario:
            The RGB branch is gated on ``band_count >= 3``. For 1- and
            2-band datasets the resolver returns ``(0, None)``
            regardless of any colour tags the user might set.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((n_bands, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0, (
            f"{n_bands}-band dataset should default to band 0, got {resolved_band}"
        )
        assert resolved_rgb is None

    @pytest.mark.parametrize("n_bands", [3, 4, 12])
    def test_no_color_interpretation_defaults_to_zero(self, n_bands):
        """D-1 regression: ``band_count >= 3`` without colour tags → band 0.

        Args:
            n_bands: 3, 4, or 12 — all should default to band 0.

        Test scenario:
            ``Dataset.create_from_array`` leaves every band's
            ``ColorInterpretation`` as ``GCI_Undefined``. PR-1's D-1
            fix says: *no colour tag means no RGB heuristic*. Verify
            for the threshold (3), Sentinel/RGBA (4), and a large stack
            (12).
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((n_bands, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0, (
            f"{n_bands}-band dataset without colour tags must default to 0, "
            f"got {resolved_band}"
        )
        assert resolved_rgb is None

    def test_full_rgb_tags_resolves_red_band(self):
        """All three R/G/B tags set → resolved band is red, rgb list filled.

        Test scenario:
            ``band_color = {0: 'red', 1: 'green', 2: 'blue'}``. The
            resolver returns ``(0, [0, 1, 2])`` — the canonical happy
            path for tagged Sentinel-style imagery.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green", 2: "blue"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 0, f"Red is band 0, got {resolved_band}"
        assert resolved_rgb == [0, 1, 2], (
            f"Resolved rgb must mirror the colour tags, got {resolved_rgb}"
        )

    def test_red_tagged_on_non_zero_band_resolves_correctly(self):
        """Red on band 1 (not 0) → resolved band index follows the tag.

        Test scenario:
            Tags are reordered: red→1, green→2, blue→0. The resolver
            must pick band 1 as the rendered band and emit
            ``[1, 2, 0]`` as the rgb list.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "blue", 1: "red", 2: "green"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_band == 1, f"Red is on band 1, got {resolved_band}"
        assert resolved_rgb == [1, 2, 0], (
            f"Expected [red, green, blue] = [1, 2, 0], got {resolved_rgb}"
        )

    def test_explicit_rgb_skips_color_lookup(self):
        """``rgb=[2, 1, 0]`` user-supplied → no colour lookup, just pick rgb[0].

        Test scenario:
            On an RGB-tagged dataset, ``rgb=[2, 1, 0]`` wins over the
            tags. The resolver returns ``(2, [2, 1, 0])`` — meaning
            cleopatra renders with the user-supplied band order.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green", 2: "blue"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(
            band=None, rgb=[2, 1, 0]
        )
        assert resolved_band == 2
        assert resolved_rgb == [2, 1, 0]

    def test_partial_tags_blue_missing_falls_back_to_sentinel_default(self):
        """Red+green tagged, blue undefined → fallback ``rgb=[2, 1, 0]``.

        Test scenario:
            ``has_color_interp`` is True (red and green tagged) but
            ``get_band_by_color('blue') is None`` because no band has
            the blue tag. The resolver falls back to the Sentinel-2
            default ``[2, 1, 0]`` and picks band 2 as the rendered band.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=None, rgb=None)
        assert resolved_rgb == [2, 1, 0], (
            f"Partial tags should fall back to [2, 1, 0], got {resolved_rgb}"
        )
        assert resolved_band == 2, f"Resolved band must be rgb[0] = 2, got {resolved_band}"

    def test_explicit_band_zero_on_tagged_dataset(self):
        """Even on a fully-tagged RGB dataset, explicit ``band=0`` wins.

        Test scenario:
            Order of checks is critical: the ``band is not None`` clause
            must short-circuit before the RGB branch. A user explicitly
            asking for band 0 must get back ``(0, None)``.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        dataset.band_color = {0: "red", 1: "green", 2: "blue"}

        resolved_band, resolved_rgb = dataset._resolve_plot_band(band=0, rgb=None)
        assert resolved_band == 0
        assert resolved_rgb is None

    def test_facade_routes_through_resolver(self):
        """:meth:`Dataset.plot` calls ``_resolve_plot_band`` then delegates.

        Test scenario:
            Mock the engine and verify the facade invokes it with the
            resolved kwargs. With a 4-band un-tagged dataset (D-1
            regression case), ``band=0`` and ``rgb=None`` must reach
            the engine.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((4, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with patch.object(type(dataset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            result = dataset.plot()

        assert result == "stub"
        assert mock_plot.call_args.kwargs["band"] == 0, (
            f"Resolver should send band=0, got {mock_plot.call_args.kwargs.get('band')}"
        )
        assert mock_plot.call_args.kwargs["rgb"] is None


class TestRasterBasePlotSignatureContract:
    """M-1 regression: align abstract signature with concrete override."""

    def test_surface_reflectance_default_is_none(self):
        """``RasterBase.plot.surface_reflectance`` defaults to ``None``.

        Test scenario:
            Pre-PR-1 the abstract default was ``10000`` (Sentinel
            reflectance constant). PR-1 changed it to ``None`` to mirror
            the concrete ``Dataset.plot`` override and avoid a confusing
            cross-class default mismatch.
        """
        sig = inspect.signature(RasterBase.plot)
        param = sig.parameters["surface_reflectance"]
        assert param.default is None, (
            f"Expected None default, got {param.default!r}"
        )

    def test_band_default_is_none(self):
        """``RasterBase.plot.band`` defaults to ``None``.

        Test scenario:
            Concrete classes apply per-class resolution policy on
            ``band=None``; the ABC must therefore allow it.
        """
        sig = inspect.signature(RasterBase.plot)
        param = sig.parameters["band"]
        assert param.default is None, (
            f"Expected None default, got {param.default!r}"
        )

    def test_rgb_default_is_none(self):
        """``RasterBase.plot.rgb`` defaults to ``None``.

        Test scenario:
            ``rgb=None`` lets the per-class facade decide whether to
            apply an RGB heuristic — a concrete default would short-
            circuit that decision on the abstract layer.
        """
        sig = inspect.signature(RasterBase.plot)
        param = sig.parameters["rgb"]
        assert param.default is None, (
            f"Expected None default, got {param.default!r}"
        )


class TestNetCDFPlotPolicy:
    """Logic-only tests for :meth:`NetCDF.plot` (no cleopatra import).

    See also:
        ``tests/dataset/test_plot.py::TestNetCDFPlot`` (the PR-1/D-0
        originals) and ``tests/netcdf/test_plot.py`` —
        ``TestNetCDFPlotRejectedKwargs`` / ``TestNetCDFPlotBandKwargRejected``
        / ``TestNetCDFPlotRejectedKwargsCombinations`` cover the same
        forbidden-kwarg gate end-to-end against the post-PR-2 signature;
        this class is the no-cleopatra-needed logic slice of it.
    """

    def test_root_mdim_container_raises_value_error(self):
        """Root MDIM container call to ``plot`` raises :class:`ValueError`.

        Test scenario:
            ``_check_not_container`` must still fire so a user who
            forgets to extract a variable gets a clear instruction to
            call ``get_variable(name)`` first.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 5, 6)).astype("float32")
        nc_root = NetCDF.create_from_array(
            arr=arr,
            geo=(30.0, 0.5, 0, 35.0, 0, -0.5),
            variable_name="t2m",
            path=None,
            extra_dim_name="time",
            extra_dim_values=[0, 1, 2],
        )
        assert nc_root._is_md_array is True
        assert nc_root._is_subset is False
        assert nc_root.band_count == 0

        with pytest.raises(ValueError, match=r"get_variable"):
            nc_root.plot()

    @pytest.mark.parametrize(
        "kwarg, value, expected_substr",
        [
            ("band", 0, "Selectors"),
            ("rgb", [0, 1, 2], "not RGB"),
            ("surface_reflectance", 10000, "Sentinel"),
            ("cutoff", [0.5], "vmin"),
            ("percentile", 2, "robust"),
            ("overview", True, "GeoTIFF/COG"),
            ("overview_index", 0, "GeoTIFF/COG"),
        ],
    )
    def test_forbidden_kwarg_raises_type_error(self, kwarg, value, expected_substr):
        """Each rejected GeoTIFF/Sentinel kwarg raises :class:`TypeError` with hint text.

        Args:
            kwarg: Name of the kwarg under test.
            value: A representative value for that kwarg.
            expected_substr: Substring the error message must contain
                so the replacement-suggestion contract doesn't drift.

        Test scenario:
            ``band`` must point at ``Selectors``; ``rgb`` must say
            "not RGB"; ``surface_reflectance`` must mention Sentinel;
            ``cutoff`` must point at ``vmin``/``vmax``; ``percentile``
            must point at ``robust``; the overview kwargs must mention
            ``GeoTIFF/COG``.
        """
        nc_subset = _make_nc_subset(n_bands=2)
        with pytest.raises(TypeError) as exc_info:
            nc_subset.plot(**{kwarg: value})
        assert expected_substr in str(exc_info.value), (
            f"Expected {expected_substr!r} in TypeError, got: {exc_info.value!r}"
        )

    def test_first_forbidden_kwarg_in_dict_order_wins(self):
        """Multiple forbidden kwargs → only the first (``rgb``) raises.

        Test scenario:
            ``forbidden_kwargs`` is a literal dict whose insertion
            order is ``rgb`` -> ``surface_reflectance`` -> ``cutoff``.
            Iteration follows that order, so passing all three raises
            the ``rgb`` TypeError first.
        """
        nc_subset = _make_nc_subset(n_bands=2)
        with pytest.raises(TypeError, match=r"rgb="):
            nc_subset.plot(rgb=[0, 1, 2], surface_reflectance=10000, cutoff=[0.5])

    @pytest.mark.parametrize("n_bands", [1, 3, 4])
    def test_default_band_is_zero(self, n_bands):
        """Variable subsets always default to ``band=0`` regardless of count.

        Args:
            n_bands: 1 (single-band), 3 (RGB threshold), 4 (Sentinel-shape).

        Test scenario:
            ``NetCDF.plot`` overrides without calling ``super().plot()``,
            so the GeoTIFF RGB heuristic never fires. Verify each
            boundary value individually.
        """
        nc_subset = _make_nc_subset(n_bands=n_bands)
        assert nc_subset.band_count == n_bands

        with patch.object(type(nc_subset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc_subset.plot()

        assert mock_plot.call_args.kwargs["band"] == 0, (
            f"NetCDF default must be band=0, got {mock_plot.call_args.kwargs.get('band')}"
        )

    def test_band_kwarg_rejected_with_migration_hint(self):
        """``NetCDF.plot(band=2)`` raises ``TypeError`` — ``band=`` is not NetCDF vocabulary.

        Test scenario:
            The flat band index was a back-compat shim that has been
            removed; ``band`` now lives in ``_FORBIDDEN_PLOT_KWARGS``.
            The error message must point at ``Selectors(...)``.
        """
        nc_subset = _make_nc_subset(n_bands=4)
        with pytest.raises(TypeError, match=r"band=") as exc_info:
            nc_subset.plot(band=2)
        assert "Selectors" in str(exc_info.value)

    def test_extra_kwargs_forwarded_to_engine(self):
        """Innocent kwargs (``figsize``) propagate past the forbidden-kwarg gate.

        Test scenario:
            Verify the gate doesn't accidentally swallow generic
            cleopatra kwargs that aren't in the forbidden set.
        """
        nc_subset = _make_nc_subset(n_bands=2)
        with patch.object(type(nc_subset.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc_subset.plot(figsize=(3, 3), title="ok")

        kwargs = mock_plot.call_args.kwargs
        assert kwargs["figsize"] == (3, 3)
        assert kwargs["title"] == "ok"
