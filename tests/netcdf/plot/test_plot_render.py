"""Plot tests: default render, colour forwarding, colorbar, container behaviour, CRS stamps."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import GeoReference, Selectors
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_plot_3d_nc
from tests.netcdf.plot._plot_helpers import _make_4d_nc

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.glyphs.gridded.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")

_cleo_params = pytest.importorskip(
    "cleopatra.styling.params", reason="cleopatra not installed"
)
Contour = _cleo_params.Contour
DataStyle = _cleo_params.DataStyle


class TestNetCDFPlotColourForwarding:
    """Tests that the colour kwargs forward to cleopatra."""

    def test_robust_forwarded(self):
        """`robust=True` reaches `Analysis.plot` as `robust=True`."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", robust=True)
        assert mock_plot.call_args.kwargs.get("robust") is True

    def test_center_forwarded(self):
        """`center=0.0` reaches `Analysis.plot` as `center=0.0`."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", center=0.0)
        assert mock_plot.call_args.kwargs.get("center") == pytest.approx(0.0)

    def test_robust_default_not_forwarded(self):
        """`robust=False` (the default) is NOT forwarded to keep kwargs lean."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m")
        assert "robust" not in mock_plot.call_args.kwargs

    def test_levels_extend_cbar_kwargs_forwarded(self):
        """`levels` folds into a `Contour` group; `extend`/`cbar_kwargs` stay loose.

        cleopatra 0.30 moved `levels` onto the typed `contour=Contour(levels=...)`
        group, passed directly to `NetCDF.plot`; `extend` and the raw `cbar_kwargs`
        passthrough are still forwarded verbatim as loose kwargs.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        cbar = {"label": "test"}
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(
                variable="t2m",
                contour=Contour(levels=5),
                extend="both",
                cbar_kwargs=cbar,
            )
        kw = mock_plot.call_args.kwargs
        assert kw.get("contour").levels == 5
        assert kw.get("extend") == "both"
        assert kw.get("cbar_kwargs") == cbar


class TestNetCDFPlotDefaultRender:
    """Regression coverage for the default render path."""

    def test_3d_returns_array_glyph(self):
        """`nc.plot(variable=...)` on a 3-D variable returns an ArrayGlyph."""
        nc = make_plot_3d_nc()
        result = nc.plot(variable="t2m")
        assert isinstance(result, ArrayGlyph)

    @pytest.mark.skipif(
        not hasattr(ArrayGlyph, "apply_style"),
        reason="cleopatra < 0.25 has no glyph.apply_style()",
    )
    def test_returned_glyph_supports_apply_style(self):
        """The glyph from `NetCDF.plot` can be restyled in place (cleopatra 0.25).

        `DataStyle.style` documents that the glyph `NetCDF.plot` returns exposes
        `apply_style`; verify the round trip via the documented data_style path.
        """
        nc = make_plot_3d_nc()
        glyph = nc.plot(variable="t2m", data_style=DataStyle(style="flow_accumulation"))
        assert glyph.style == "flow_accumulation"
        glyph.apply_style("topography")
        assert glyph.style == "topography", "apply_style must restyle in place"

    def test_selectors_none_equivalent_to_omitted(self):
        """`selectors=None` renders the same default slice as omitting it.

        Test scenario:
            A missing ``selectors=`` is normalised to ``Selectors()``
            inside ``NetCDF.plot``, so ``selectors=None`` must render
            byte-for-byte the same 2-D array as the bare call.
        """
        nc = make_plot_3d_nc()
        explicit_none = nc.plot(variable="t2m", selectors=None)
        omitted = nc.plot(variable="t2m")
        assert isinstance(explicit_none, ArrayGlyph)
        assert isinstance(omitted, ArrayGlyph)
        assert_array_equal(
            explicit_none.arr,
            omitted.arr,
            err_msg="selectors=None must render the same array as the bare call",
        )

    def test_3d_time_slice_arrayglyph_shape_matches_2d(self):
        """`time=N` selector returns an ArrayGlyph wrapping the 5x5 spatial slice.

        Test scenario:
            Build a 3-time-step container with 5x5 grid, pin
            ``time=1``, and verify the returned ArrayGlyph wraps a 2-D
            array of shape ``(5, 5)`` that equals the corresponding
            band of the source.
        """
        nc = make_plot_3d_nc(n_times=3, rows=5, cols=5)
        var = nc.get_variable("t2m")
        expected = var.read_array()[1]
        result = nc.plot(variable="t2m", selectors=Selectors(time=1))
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )
        assert result.arr.shape == (
            5,
            5,
        ), f"Expected 2-D (5, 5) slice, got shape {result.arr.shape}"
        assert_array_equal(
            result.arr,
            expected,
            err_msg="time=1 slice must equal var.read_array()[1] byte-for-byte",
        )

    def test_4d_time_level_slice_matches_chained_sel(self):
        """4-D `time=` + `level=` returns the same 2-D array as chained ``sel()``.

        Test scenario:
            Plot a 4-D ``(time, pressure_level, lat, lon)`` variable
            with ``time=12, level=500`` and verify the rendered
            ArrayGlyph's array equals ``var.sel(time=12).sel(pressure_level=500).read_array()``.
        """
        nc = _make_4d_nc()
        var = nc.get_variable("temperature")
        expected = var.sel(time=12).sel(pressure_level=500).read_array()
        result = nc.plot(
            variable="temperature",
            selectors=Selectors(time=12, level=500),
        )
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )
        assert_array_equal(
            result.arr,
            expected,
            err_msg="4-D plot slice must match var.sel(...).sel(...).read_array()",
        )

    def test_plot_twice_returns_independent_array_glyphs(self):
        """Two successive plot calls return distinct ArrayGlyph instances.

        Test scenario:
            Calling ``nc.plot`` twice on the same variable must
            produce two ArrayGlyph objects with different ``.fig``
            matplotlib figure instances — each render gets its own
            canvas and no state leaks between calls.
        """
        nc = make_plot_3d_nc()
        first = nc.plot(variable="t2m", selectors=Selectors(time=0))
        second = nc.plot(variable="t2m", selectors=Selectors(time=0))
        assert first is not second, "Successive plot calls returned the same object"
        assert first.fig is not second.fig, (
            "Successive plot calls shared the same matplotlib Figure; "
            "each render must own its canvas"
        )


class TestNetCDFPlotForwardingExtra:
    """Additional kwarg-forwarding edges to cleopatra."""

    def test_cmap_forwarded(self):
        """``cmap="viridis"`` reaches ``Analysis.plot`` verbatim."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", cmap="viridis")
        assert mock_plot.call_args.kwargs.get("cmap") == "viridis", (
            f"cmap must be forwarded, got: {mock_plot.call_args.kwargs}"
        )

    def test_vmin_vmax_forwarded(self):
        """``vmin``/``vmax`` are forwarded to the renderer."""
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", vmin=0.0, vmax=1.0)
        kw = mock_plot.call_args.kwargs
        assert kw.get("vmin") == pytest.approx(0.0), f"vmin not forwarded: {kw}"
        assert kw.get("vmax") == pytest.approx(1.0), f"vmax not forwarded: {kw}"

    def test_basemap_without_epsg_raises(self):
        """``basemap=True`` on a CRS-less subset must surface the underlying ValueError.

        Test scenario:
            The ``Analysis.plot`` engine enforces
            ``"Dataset must have a CRS (epsg) to use basemap."``. We
            null out ``_epsg`` on the variable subset returned by
            ``get_variable`` and confirm NetCDF.plot does not
            short-circuit the basemap contract.
        """
        nc = make_plot_3d_nc()
        real_get_variable = type(nc).get_variable

        def _spy(self_, name, x_dim=None, y_dim=None):
            sub = real_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            # Strip the CRS itself, not just the cached `_epsg`. That cache is no
            # longer authoritative: `epsg` re-derives from the CRS when it is
            # None, so a subset is only genuinely CRS-less once `_get_crs`
            # reports nothing either.
            sub._epsg = None
            sub._get_crs = lambda: ""
            return sub

        with patch.object(type(nc), "get_variable", _spy):
            with pytest.raises(ValueError, match=r"CRS"):
                nc.plot(variable="t2m", basemap=True)


@pytest.mark.plot
class TestNetCDFPlotAddColorbar:
    """Regression coverage for H1 — ``add_colorbar`` is now honoured."""

    def test_add_colorbar_true_default_keeps_cbar(self):
        """``add_colorbar=True`` (default) leaves the cleopatra colorbar in place.

        Test scenario:
            Render a small 3-D variable without touching ``add_colorbar``.
            The returned ArrayGlyph must still expose a non-``None``
            ``.cbar`` attribute (cleopatra attaches one by default).
        """
        nc = make_plot_3d_nc()
        cleo = nc.plot(variable="t2m")
        assert getattr(cleo, "cbar", None) is not None, (
            "Default add_colorbar=True must preserve cleopatra's colorbar; "
            f"got cbar={getattr(cleo, 'cbar', None)!r}"
        )

    def test_colorbar_false_removes_cbar(self):
        """``colorbar=False`` removes the colorbar from the rendered result.

        Test scenario:
            The plot contract: a user who passes ``colorbar=False``
            expects no colorbar in the output. cleopatra renders no
            colorbar natively when the hoisted ``colorbar=False`` is
            forwarded. We assert the ``.cbar`` attribute is ``None``.
        """
        nc = make_plot_3d_nc()
        cleo = nc.plot(variable="t2m", colorbar=False)
        assert getattr(cleo, "cbar", None) is None, (
            "colorbar=False must yield no colorbar; "
            f"got cbar={getattr(cleo, 'cbar', None)!r}"
        )

    def test_colorbar_false_forwarded_to_engine(self):
        """``colorbar=False`` reaches ``Analysis.plot`` as a hoisted kwarg.

        Test scenario:
            cleopatra drops the colorbar natively, so ``NetCDF.plot``
            forwards the hoisted ``colorbar`` value straight through to
            the engine. Patch the engine and confirm ``colorbar=False``
            appears in its kwargs.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = SimpleNamespace(cbar=None)
            nc.plot(variable="t2m", colorbar=False)
        kw = mock_plot.call_args.kwargs
        assert kw.get("colorbar") is False, (
            "colorbar=False must be forwarded to the engine; "
            f"engine call kwargs were: {kw}"
        )


class TestNetCDFPlotContainerBehaviour:
    """Container/subset dispatch wiring covered end-to-end."""

    def test_container_error_lists_available_variables(self):
        """The missing-variable error message includes every available name.

        Test scenario:
            On a container with a single variable ``t2m``, the
            ValueError text must include ``'t2m'`` so users can pick
            from the list verbatim.
        """
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"variable=") as exc_info:
            nc.plot()
        assert "t2m" in str(exc_info.value), (
            f"Error message must list available variables, got: {exc_info.value}"
        )

    def test_subset_with_matching_variable_continues_silently(self):
        """``var.plot(variable=var._source_var_name)`` does not raise.

        Test scenario:
            Mirror of ``read_array``'s contract: passing the pinned
            name on a subset is accepted and the call proceeds to
            render. We patch ``Analysis.plot`` to confirm the call
            reaches the engine.
        """
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            result = var.plot(variable=var._source_var_name)
        assert result == "ok", (
            f"Expected the patched render to return 'ok', got: {result!r}"
        )
        assert mock_plot.called, "Analysis.plot was not invoked"


class TestPlotStampsGlyphCRS:
    """`NetCDF.plot` stamps the dataset EPSG onto the returned glyph (issue #630).

    With cleopatra >= 0.20.0 the glyph's reference-layer helpers default their
    `crs=` to `glyph.crs`, so a stamped glyph lets `glyph.add_features("coastline",
    "50m")` line up with the data without the caller restating the CRS.
    """

    def test_variable_plot_glyph_carries_epsg(self):
        """A plotted variable's glyph exposes the dataset's geographic CRS.

        Test scenario:
            Plot a 2-D EPSG:4326 variable; the returned glyph's `crs` equals 4326,
            so a subsequent `add_features`/`add_tiles` needs no explicit `crs=`.
        """
        nc = NetCDF.create_from_array(
            np.arange(12.0).reshape(3, 4),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0, 3.0, 0, -1.0), epsg=4326),
            variable_name="d",
        )
        glyph = nc.get_variable("d").plot()
        assert glyph.crs == 4326, f"expected glyph.crs == 4326, got {glyph.crs!r}"

    def test_projected_epsg_is_stamped(self):
        """The stamp reflects the dataset's own EPSG, not a hard-coded 4326.

        Test scenario:
            A variable declared EPSG:3857 yields a glyph whose `crs` is 3857.
        """
        nc = NetCDF.create_from_array(
            np.arange(12.0).reshape(3, 4),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0, 3.0, 0, -1.0), epsg=3857),
            variable_name="d",
        )
        glyph = nc.get_variable("d").plot()
        assert glyph.crs == 3857, f"expected glyph.crs == 3857, got {glyph.crs!r}"
