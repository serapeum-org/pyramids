"""Hoisted ``colorbar`` / ``points`` params on :meth:`NetCDF.plot`.

The cleopatra-0.29 alignment lifts ``colorbar`` and ``points`` out of the opaque
``**kwargs`` and onto the ``NetCDF.plot`` signature (parity with ``Dataset.plot`` /
``DatasetCollection.plot``). Because ``NetCDF.plot`` is a multi-mode facade
(static / animate / facet) and cleopatra's facet path rejects ``points`` outright,
the hoisted params are forwarded **only when set** — a default ``None`` must never
reach the render call. These tests pin both the parity forwarding and that
conditional-injection contract.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.netcdf import FacetSpec
from tests.netcdf.conftest import make_plot_3d_nc

pytestmark = pytest.mark.plot

pytest.importorskip("cleopatra", minversion="0.29", reason="needs cleopatra >= 0.29")
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_cleo_config.Config.set_matplotlib_backend("Agg")
_cleo_array = pytest.importorskip(
    "cleopatra.glyphs.gridded.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
FacetGrid = _cleo_array.FacetGrid
ColorBar = _cleo_array.ColorBar
plt = pytest.importorskip("matplotlib.pyplot", reason="cleopatra not installed")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every figure after each test so the suite does not leak them.

    Yields:
        None: control returns to the test; figures are closed on teardown.
    """
    yield
    plt.close("all")


def _points() -> np.ndarray:
    """Build a minimal cleopatra point overlay for the 5x5 plot grid.

    Returns:
        np.ndarray: A ``(2, 3)`` array of ``(value, row, col)`` rows whose row/col
        indices fall inside the ``make_plot_3d_nc`` 5x5 raster.
    """
    return np.array([[1.0, 1, 1], [2.0, 2, 2]], dtype="float64")


class TestNetCDFHoistedRenderParamsStatic:
    """The hoisted ``colorbar`` / ``points`` params render on the static path.

    cleopatra rejects an unknown keyword, so a clean :class:`ArrayGlyph` proves the
    param was forwarded to ``ArrayGlyph.plot`` rather than silently dropped.
    """

    def test_colorbar_spec_renders(self):
        """A ``ColorBar`` spec on the static path renders to an ``ArrayGlyph``.

        Test scenario:
            ``nc.plot(variable="t2m", colorbar=ColorBar(...))`` on a 3-D variable
            returns an :class:`ArrayGlyph`, proving the hoisted ``colorbar`` reached
            cleopatra.
        """
        nc = make_plot_3d_nc()
        glyph = nc.plot(variable="t2m", colorbar=ColorBar(label="temp"))
        assert isinstance(glyph, ArrayGlyph), f"expected ArrayGlyph, got {type(glyph)}"

    def test_colorbar_false_renders(self):
        """``colorbar=False`` suppresses the colour bar and still renders.

        Test scenario:
            The boolean ``False`` form of the hoisted param forwards cleanly and
            yields an :class:`ArrayGlyph`.
        """
        nc = make_plot_3d_nc()
        glyph = nc.plot(variable="t2m", colorbar=False)
        assert isinstance(glyph, ArrayGlyph), f"expected ArrayGlyph, got {type(glyph)}"

    def test_points_overlay_renders(self):
        """A ``points`` overlay on the static path renders to an ``ArrayGlyph``.

        Test scenario:
            ``nc.plot(variable="t2m", points=<(2, 3) array>)`` forwards the hoisted
            ``points`` param to cleopatra and returns an :class:`ArrayGlyph`.
        """
        nc = make_plot_3d_nc()
        glyph = nc.plot(variable="t2m", points=_points())
        assert isinstance(glyph, ArrayGlyph), f"expected ArrayGlyph, got {type(glyph)}"


class TestNetCDFHoistedParamConditionalInjection:
    """Defaults are not injected; explicit values are.

    The forwarding is spied at ``Analysis.plot`` — the render boundary the facade
    hands off to — to assert the presence/absence of the hoisted keys.
    """

    def test_defaults_not_injected(self):
        """Omitted ``colorbar`` / ``points`` never reach the render call.

        Test scenario:
            A bare ``nc.plot(variable="t2m")`` must not forward ``colorbar`` or
            ``points`` (their default ``None`` is dropped), so the facet path — which
            rejects ``points`` — is never handed a spurious key.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m")
        forwarded = mock_plot.call_args.kwargs
        assert "points" not in forwarded, f"points must not be injected: {forwarded}"
        assert "colorbar" not in forwarded, (
            f"colorbar must not be injected by default: {forwarded}"
        )

    def test_points_injected_when_set(self):
        """An explicit ``points`` value is forwarded to the render call.

        Test scenario:
            ``nc.plot(variable="t2m", points=<array>)`` forwards ``points`` verbatim
            to ``Analysis.plot``.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        pts = _points()
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", points=pts)
        forwarded = mock_plot.call_args.kwargs
        assert "points" in forwarded, f"points must be forwarded when set: {forwarded}"
        assert forwarded["points"] is pts, "points must forward the exact array"


class TestNetCDFFacetPointsContract:
    """The facet path: default ``points`` is fine; an explicit one is rejected.

    This is the exact contract the conditional injection exists to preserve — a
    default ``None`` must not reach cleopatra's facet (which has no ``points``
    parameter), while an explicit ``points`` correctly surfaces the error.
    """

    def test_facet_without_points_renders(self):
        """A facet plot with no ``points`` renders to a ``FacetGrid``.

        Test scenario:
            ``nc.plot(variable="t2m", facet=FacetSpec(col="time"))`` must succeed —
            the default ``points`` is not injected, so the facet path is not handed a
            keyword it rejects.
        """
        nc = make_plot_3d_nc(n_times=4)
        grid = nc.plot(variable="t2m", facet=FacetSpec(col="time"))
        assert isinstance(grid, FacetGrid), f"expected FacetGrid, got {type(grid)}"

    def test_facet_with_explicit_points_raises(self):
        """An explicit ``points`` on the facet path raises a clear error.

        Test scenario:
            cleopatra's facet has no ``points`` parameter, so
            ``nc.plot(variable="t2m", facet=FacetSpec(col="time"), points=<array>)``
            raises ``ValueError`` naming the offending keyword.
        """
        nc = make_plot_3d_nc(n_times=4)
        facet, pts = FacetSpec(col="time"), _points()
        with pytest.raises(ValueError, match="points") as exc_info:
            nc.plot(variable="t2m", facet=facet, points=pts)
        assert "points" in str(exc_info.value), f"unexpected error: {exc_info.value}"


class TestNetCDFGroupParams:
    """Typed render groups (color/contour/cells/data_style) on ``NetCDF.plot``."""

    def test_group_reaches_the_render_call(self):
        """An explicit ``color`` forwards through to ``Analysis.plot``.

        Test scenario:
            ``nc.plot(variable="t2m", color=ColorScaling(...))`` forwards the spec down
            the facade -> NetCDFPlot.run -> Analysis.plot chain.
        """
        from cleopatra.styling.scaling import ColorScaling

        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        color = ColorScaling.power(gamma=0.8)
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", color=color)
        assert mock_plot.call_args.kwargs.get("color") is color, (
            "explicit color must reach Analysis.plot"
        )

    def test_unset_groups_not_forwarded(self):
        """A bare plot forwards none of the typed render groups."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m")
        forwarded = mock_plot.call_args.kwargs
        for name in ("color", "contour", "cells", "data_style"):
            assert name not in forwarded, f"unset {name} must not be forwarded"

    def test_explicit_contour_wins_over_coloropts_levels(self):
        """An explicit ``contour=`` beats a ``ColorOpts(levels=...)`` (no silent override).

        Test scenario:
            ``NetCDF.plot`` exposes both the hoisted ``contour=`` group and the older
            ``ColorOpts(levels=...)`` bag. When both are given the caller's explicit
            ``contour`` must win — the ColorOpts-derived one only fills an unset slot.
        """
        from cleopatra.styling.params import Contour

        from pyramids.netcdf import ColorOpts

        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        explicit = Contour(levels=[1, 2, 3])
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", contour=explicit, colour=ColorOpts(levels=9))
        forwarded = mock_plot.call_args.kwargs.get("contour")
        assert forwarded is explicit, f"explicit contour must win; got {forwarded}"
        assert forwarded.levels == [1, 2, 3], (
            f"the winning contour must keep its levels; got {forwarded.levels}"
        )
