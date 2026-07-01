"""Plot tests: animation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import FacetSpec, Selectors
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf._plot_helpers import _make_4d_nc, _make_fake_render
from tests.netcdf.conftest import make_plot_3d_nc

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")


class TestNetCDFPlotAnimate:
    """Tests for the PR-5 ``animate=`` kwarg on NetCDF.plot."""

    def test_animate_true_returns_array_glyph(self):
        """``animate=True`` on a single-band-dim variable returns ArrayGlyph.

        Test scenario:
            On a 3-D ``(time, lat, lon)`` variable with ``time`` as
            the only band dim, ``animate=True`` resolves it as the
            target dim and returns a cleopatra ``ArrayGlyph`` wrapping
            the streamed animation. The matplotlib animation object is
            attached to the glyph's figure.
        """
        nc = make_plot_3d_nc(n_times=3)
        result = nc.plot(variable="t2m", animate=True)
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"
        assert result.fig is not None, "Animation must own a Figure"

    def test_animate_string_resolves_named_dim(self):
        """``animate="time"`` walks the named dim and returns ArrayGlyph."""
        nc = make_plot_3d_nc(n_times=4)
        result = nc.plot(variable="t2m", animate="time")
        assert isinstance(result, ArrayGlyph)

    def test_animate_unknown_dim_raises(self):
        """``animate="bogus"`` raises ``KeyError`` (unknown dim name, xarray-style) (N4).

        Test scenario:
            N4 — an unknown *dim name* on ``animate=`` is a missing-key
            error, so it raises ``KeyError`` (mirroring xarray's
            ``ds.sel(unknown_dim=...)``), distinct from the ``ValueError``
            used for invalid *combinations* (pin conflict, faceting
            conflict, ``animate=True`` ambiguity). The message still
            lists the available band dims.
        """
        nc = make_plot_3d_nc()
        with pytest.raises(KeyError, match=r"animate='bogus'") as exc_info:
            nc.plot(variable="t2m", animate="bogus")
        assert "time" in str(
            exc_info.value
        ), f"error should list the available band dims, got: {exc_info.value}"

    def test_animate_with_col_raises_mutually_exclusive(self):
        """``animate=True`` together with ``col=`` is rejected up-front."""
        nc = make_plot_3d_nc()
        spec = FacetSpec(col="time")
        with pytest.raises(ValueError, match=r"mutually exclusive"):
            nc.plot(variable="t2m", animate=True, facet=spec)

    def test_animate_with_pinned_dim_raises(self):
        """``animate="time"`` together with ``time=`` selector conflicts."""
        nc = make_plot_3d_nc()
        sel = Selectors(time=0)
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(variable="t2m", animate="time", selectors=sel)

    def test_animate_true_with_multiple_band_dims_raises(self):
        """``animate=True`` on a 4-D variable without selectors is ambiguous.

        Test scenario:
            A 4-D ``(time, pressure_level, lat, lon)`` variable has
            two free band dims. ``animate=True`` cannot pick one
            automatically — the error message must mention both free
            dims so the caller can name the target via ``animate=<name>``.
        """
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"exactly one free band dim"):
            nc.plot(variable="temperature", animate=True)

    def test_animate_4d_with_level_pin_picks_time(self):
        """4-D variable: pin ``level``, then ``animate=True`` picks ``time``.

        Test scenario:
            Collapsing ``pressure_level`` via ``level=`` leaves
            ``time`` as the only free band dim, so ``animate=True``
            unambiguously walks ``time``.
        """
        nc = _make_4d_nc()
        result = nc.plot(
            variable="temperature",
            selectors=Selectors(level=500),
            animate=True,
        )
        assert isinstance(result, ArrayGlyph)

    def test_animate_uses_cf_decoded_time_labels(self):
        """CF-decoded date strings are used as the animation frame labels.

        Test scenario:
            Build a 3-D NetCDF whose ``time`` dim carries CF
            ``units="days since 2024-01-01"`` so ``get_time_variable()``
            decodes the raw values into ``YYYY-MM-DD`` strings. Patch
            :meth:`NetCDF.get_time_variable` to return a known label
            list and assert those labels reach
            :func:`pyramids.dataset._plot_helpers.render_array` via
            ``animation_axis_values``.
        """
        nc = make_plot_3d_nc(n_times=3)
        labels = ["2024-01-01", "2024-01-02", "2024-01-03"]
        captured: dict = {}

        with patch(
            "pyramids.netcdf.netcdf.NetCDF.get_time_variable",
            return_value=labels,
        ):
            with patch(
                "pyramids.netcdf._plot._render_array",
                side_effect=_make_fake_render(captured),
            ):
                nc.plot(variable="t2m", animate=True)
        assert captured["kw"]["mode"] == "animate"
        assert captured["kw"]["animation_axis_values"] == labels

    def test_animate_data_getter_called_once_per_frame(self):
        """The lazy ``data_getter`` invokes ``sel().read_array`` per frame.

        Test scenario:
            Patch the ``_render_array`` indirection in
            :mod:`pyramids.netcdf.netcdf` so the test captures the
            ``data_getter`` callable that pyramids hands to cleopatra.
            Invoking the callable for indices 0..N-1 must call
            ``sel().read_array(band=0)`` exactly once per call and
            return a 2-D array matching the source slice.
        """
        nc = make_plot_3d_nc(n_times=4)
        captured: dict = {}

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_make_fake_render(captured),
        ):
            nc.plot(variable="t2m", animate=True)
        getter = captured["kw"]["data_getter"]
        var = nc.get_variable("t2m")
        full = var.read_array()
        for i in range(4):
            frame = getter(i)
            assert frame.shape == (5, 5), f"Frame {i} expected (5,5), got {frame.shape}"
            assert_array_equal(
                frame,
                full[i],
                err_msg=f"Frame {i} must match full[{i}]",
            )

    def test_animate_does_not_build_full_stack(self):
        """The animate path never builds a 3-D stack up-front.

        Test scenario:
            Spy on the variable subset's ``read_array`` method and
            confirm that the pre-animate setup calls it at most once
            (for the cleopatra shape template — the first
            ``data_getter(0)`` call). The remaining frames are pulled
            lazily once cleopatra iterates the animation.
        """
        nc = make_plot_3d_nc(n_times=5)
        captured: dict = {}

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_make_fake_render(captured),
        ):
            nc.plot(variable="t2m", animate=True)
        template = captured["kw"]["arr"]
        assert (
            template.ndim == 2
        ), f"Template handed to render_array must be 2-D, got {template.shape}"

    def test_animate_invalid_type_raises(self):
        """``animate=1.0`` (non-bool, non-str) is rejected with a clear error."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"animate="):
            nc.plot(variable="t2m", animate=1.0)


class TestNetCDFPlotAnimateEdges:
    """Extra animate-path coverage beyond the original 11 PR-5 cases."""

    def test_animate_with_sel_pin_other_dim_picks_remaining_time(self):
        """4-D variable: ``animate=True`` + ``sel={'pressure_level': v}``.

        Test scenario:
            Pinning the ``pressure_level`` dim via ``sel=`` (rather than
            via ``level=``) must leave ``time`` as the only free band
            dim. ``animate=True`` then unambiguously resolves ``time``.
        """
        nc = _make_4d_nc()
        captured: dict = {}

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_make_fake_render(captured),
        ):
            nc.plot(
                variable="temperature",
                selectors=Selectors(sel={"pressure_level": 500}),
                animate=True,
            )
        assert captured["kw"]["mode"] == "animate", (
            f"Pinning level via sel must still resolve animate, got "
            f"mode={captured['kw'].get('mode')!r}"
        )
        labels = captured["kw"]["animation_axis_values"]
        assert list(labels) == [0, 6, 12], (
            f"animation labels must come from the time coord values, got {labels}"
        )

    def test_animate_with_isel_pin_animated_dim_raises(self):
        """``animate='time'`` rejects an ``isel`` pin on the same dim.

        Test scenario:
            ``isel={'time': 0}`` collapses the time dim before the
            animate walker can iterate. The pin must lose to a clear
            ValueError that names the already-pinned dim.
        """
        nc = make_plot_3d_nc()
        sel = Selectors(isel={"time": 0})
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(variable="t2m", animate="time", selectors=sel)

    def test_animate_string_when_other_band_dims_free(self):
        """4-D variable: ``animate='pressure_level'`` walks the named dim.

        Test scenario:
            On a 4-D variable both ``time`` and ``pressure_level`` are
            free. ``animate='pressure_level'`` must resolve and the
            frame labels come straight from the dim's coord values
            (1000, 500) — no CF time decoding for non-time dims.
        """
        nc = _make_4d_nc()
        captured: dict = {}

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_make_fake_render(captured),
        ):
            nc.plot(
                variable="temperature",
                selectors=Selectors(time=0),
                animate="pressure_level",
            )
        labels = captured["kw"]["animation_axis_values"]
        assert list(labels) == [
            1000,
            500,
        ], f"Non-time dim labels must be raw coord values, got {labels}"

    def test_animate_false_takes_static_path(self):
        """``animate=False`` is treated like the default; no animate dispatch.

        Test scenario:
            The plot façade guards on ``animate is not None and animate
            is not False``, so ``animate=False`` must fall through to
            the static-plot path. Patch the engine to capture the
            kwargs and verify there is no ``animation_axis_values`` key.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", selectors=Selectors(time=0), animate=False)
        kw = mock_plot.call_args.kwargs
        assert (
            "animation_axis_values" not in kw
        ), f"animate=False must not engage the animate path; got {kw}"

    def test_animate_data_getter_propagates_inner_exception(self):
        """A ``data_getter`` failure on frame N bubbles out of the call.

        Test scenario:
            Capture the ``data_getter`` callable cleopatra would receive
            (post-M3 it reads a flat band index directly — no per-frame
            ``sel()``), then replace ``read_array`` with one that raises
            for the third frame's band. Invoking ``data_getter(2)``
            directly (the equivalent of cleopatra's frame iteration)
            must propagate that exception unchanged so caller-facing
            errors surface with the original traceback.
        """
        nc = make_plot_3d_nc(n_times=4)
        captured: dict = {}

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_make_fake_render(captured),
        ):
            nc.plot(variable="t2m", animate=True)
        getter = captured["kw"]["data_getter"]
        var_cls = type(nc.get_variable("t2m"))
        orig_read = var_cls.read_array

        def _broken_read(self, *args, **kwargs):
            if kwargs.get("band") == 2:
                raise RuntimeError("simulated frame-3 failure")
            return orig_read(self, *args, **kwargs)

        with patch.object(var_cls, "read_array", _broken_read):
            getter(0)
            with pytest.raises(RuntimeError, match=r"frame-3"):
                getter(2)

    def test_animate_does_not_allocate_per_frame_sel_subsets(self):
        """The animate ``data_getter`` reads flat band indices, never calls ``sel`` (M3).

        Test scenario:
            M3 fix — the per-frame fetch used to allocate a fresh
            ``NetCDF`` subset via ``self.sel(...)`` for every frame
            (re-resolving + re-opening the GDAL MDArray view). It now
            computes the flat band index and calls ``read_array(band=N)``
            on the existing handle. Patch ``NetCDF.sel`` and confirm the
            animate render path never touches it; then exercise the
            captured ``data_getter`` for every frame and confirm it
            returns the correct slices.
        """
        nc = make_plot_3d_nc(n_times=4)
        captured: dict = {}

        with patch.object(NetCDF, "sel", autospec=True) as sel_mock:
            with patch(
                "pyramids.netcdf._plot._render_array",
                side_effect=_make_fake_render(captured),
            ):
                nc.plot(variable="t2m", animate=True)
            assert not sel_mock.called, (
                "animate path must not allocate per-frame sel() subsets; "
                f"sel was called {sel_mock.call_count} time(s)"
            )
        getter = captured["kw"]["data_getter"]
        var = nc.get_variable("t2m")
        for i in range(4):
            assert_array_equal(
                np.asarray(getter(i)),
                np.asarray(var.read_array(band=i)),
                err_msg=f"frame {i} should equal band {i} of the variable",
            )

    def test_animate_4d_with_two_free_band_dims_lists_both(self):
        """Error message names both free band dims for the 4-D case.

        Test scenario:
            ``animate=True`` on a 4-D variable with no selectors must
            mention ``time`` and ``pressure_level`` in the error so the
            user knows which names are valid for ``animate=<dim>``.
        """
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"exactly one free band dim") as exc:
            nc.plot(variable="temperature", animate=True)
        msg = str(exc.value)
        assert (
            "time" in msg and "pressure_level" in msg
        ), f"Error must list both free dims, got {msg!r}"

    def test_animate_2d_variable_with_no_band_dims_raises(self):
        """``animate=True`` on a pure 2-D variable raises a clear error.

        Test scenario:
            Build a 2-D variable, then ``animate=True`` must reject the
            request because there is no band dim to iterate. The error
            mentions either "no band" or "exactly one free band dim".
        """
        rng = np.random.default_rng(11)
        arr = rng.random((4, 4)).astype(np.float32)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
            epsg=4326,
            variable_name="flat",
        )
        with pytest.raises(ValueError, match=r"(?:no band|free band dim)"):
            nc.plot(variable="flat", animate=True)
