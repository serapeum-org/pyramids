"""Plot tests: the Analysis plot engine, render_array kwarg routing, and the mesh-render helper."""

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset._plot_helpers import render_array
from pyramids.dataset.engines import Analysis

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.glyphs.gridded.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
# cleopatra >= 0.26 bundles the point-overlay styling kwargs into this class,
# and the animate frame-label pair into FrameLabel.
PointOverlay = getattr(_cleo_array, "PointOverlay", None)
FrameLabel = getattr(_cleo_array, "FrameLabel", None)
ColorBar = getattr(_cleo_array, "ColorBar", None)
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("agg")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each plot test to bound memory.

    Plotting tests open figures via cleopatra/pyplot; without this teardown
    the suite accumulates them and matplotlib warns past 20 open figures.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestAnalysisPlotEngine:
    """Tests for the post-PR-1 :meth:`Analysis.plot` engine contract.

    The engine is now band-agnostic: it requires a concrete ``band``
    integer and applies no resolution heuristic.
    """

    @pytest.mark.plot
    def test_explicit_band_renders_array_glyph(self):
        """Calling ``Analysis.plot(band=N)`` directly must work.

        Test scenario:
            Bypass the facade and hit the engine directly with an
            explicit band index — exercises the branch the PR-1 docs
            promise: the engine never resolves and is purely generic.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((3, 6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        result = dataset.analysis.plot(band=2)
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )

    @pytest.mark.plot
    def test_out_of_range_band_raises(self):
        """An out-of-range band must propagate the underlying error.

        Test scenario:
            ``read_array`` raises :class:`ValueError` (or :class:`IndexError`
            on the metadata path) when the requested band is past the
            last available band. The engine performs no resolution, so
            the error should surface to the caller unchanged.
        """
        rng = np.random.default_rng(1337)
        arr = rng.random((6, 6)).astype("float32")
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )

        with pytest.raises((ValueError, IndexError)):
            dataset.analysis.plot(band=42)


class TestRenderArrayKwargRouting:
    """D-4 — fine-grained checks on which kwargs land where."""

    @staticmethod
    def _capture_calls():
        """Build a fake ``ArrayGlyph`` that records ctor/plot kwargs.

        Returns:
            tuple[type, dict, dict, dict, list]: A ``_FakeGlyph`` class
                wrapping ``__init__`` / ``plot`` / ``animate`` / ``facet``,
                plus dicts capturing each call's kwargs and an
                ``animate_args`` list capturing positional args.
        """
        ctor_seen: dict = {}
        plot_seen: dict = {}
        animate_seen: dict = {}
        facet_seen: dict = {}
        animate_args: list = []

        class _FakeAxes:
            def __init__(self):
                self.aspect = "auto"

            def get_xlim(self):
                return (0.0, 1.0)

            def get_ylim(self):
                return (0.0, 1.0)

        class _FakeGlyph:
            @staticmethod
            def option_keys():
                return ArrayGlyph.option_keys()

            def __init__(self, array, **kwargs):
                ctor_seen.clear()
                ctor_seen.update(kwargs)
                self.arr = array
                self.ax = _FakeAxes()
                self.fig = None

            def plot(self, **kwargs):
                plot_seen.clear()
                plot_seen.update(kwargs)
                return (None, self.ax)

            def animate(self, axis_values, **kwargs):
                animate_args.append(axis_values)
                animate_seen.clear()
                animate_seen.update(kwargs)
                return self

            def facet(self, **kwargs):
                facet_seen.clear()
                facet_seen.update(kwargs)
                return self

        return _FakeGlyph, ctor_seen, plot_seen, animate_seen, facet_seen, animate_args

    def test_constructor_owns_cmap_vmin_vmax_cbar(self):
        """Plain style/scale kwargs land on the constructor for plot mode.

        Test scenario:
            ``cmap``, ``vmin``, ``vmax``, ``cbar_kwargs`` must reach
            ``ArrayGlyph.__init__`` so cleopatra's ``default_options`` is set in one
            place. None of them may also reach ``ArrayGlyph.plot``; otherwise the
            value would be overwritten twice (the PR-6 D-4 fix). ``cbar_kwargs`` (the
            raw matplotlib passthrough) is not one of the loose ``cbar_*`` styling
            keywords the ``ColorBar`` spec replaced, so it stays valid.
        """
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(101)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                cmap="plasma",
                vmin=-1.0,
                vmax=2.0,
                cbar_kwargs={"orientation": "horizontal"},
            )
        for key in ("cmap", "vmin", "vmax", "cbar_kwargs"):
            assert key in ctor, f"`{key}` must land on the constructor; ctor={ctor}"
            assert key not in plot, (
                f"`{key}` must NOT also reach cleo.plot; plot={plot}"
            )

    def test_render_call_only_kwargs_reach_plot(self):
        """Render-call-only kwargs reach ``ArrayGlyph.plot`` and nothing else.

        Test scenario:
            A kwarg outside ``ArrayGlyph.option_keys()`` must reach
            ``ArrayGlyph.plot`` exclusively. The cleanup added the
            ``plot_call_only`` set in ``_plot_helpers.render_array``; a
            regression here would resurrect the double-forward bug.

            ``points`` (a render-call-only overlay) and ``full_bleed`` stand in
            for non-option kwargs; ``kind`` is the option that is force-routed to
            the render call. (The loose ``point_*`` style names are no longer used
            here — ``render_array`` now folds them into a ``PointOverlay``.)
        """
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(202)
        arr = rng.random((4, 4)).astype("float32")
        render_only_kwargs = {
            "points": np.array([[1.0, 2, 3]]),
            "full_bleed": True,
            "kind": "imshow",
        }
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                **render_only_kwargs,
            )
        for key in render_only_kwargs:
            assert key in plot, f"`{key}` must reach cleo.plot; plot={plot}"
            assert key not in ctor, (
                f"`{key}` must NOT also be on the constructor; ctor={ctor}"
            )

    def test_animate_mode_merges_both_buckets_into_animate_call(self):
        """``mode='animate'`` — every kwarg flows into ``cleo.animate(...)``.

        Test scenario:
            cleopatra's ``ArrayGlyph.animate`` re-validates every kwarg
            against ``DEFAULT_OPTIONS``, so the D-4 documentation calls
            out the animate path as the exception: both render-call-only
            and constructor buckets merge into a single ``animate_kwargs``
            dict, and the constructor receives nothing.
        """
        fake_cls, ctor, _, animate, _, anim_args = self._capture_calls()
        rng = np.random.default_rng(303)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="animate",
                animation_axis_values=[0, 1, 2],
                cmap="viridis",
                kind="imshow",
                interval=50,
            )
        for key in ("cmap", "kind", "interval"):
            assert key in animate, (
                f"In animate mode, `{key}` must reach cleo.animate; "
                f"animate kwargs={animate}"
            )
            assert key not in ctor, (
                f"In animate mode, `{key}` must NOT be on the constructor; ctor={ctor}"
            )
        assert anim_args == [[0, 1, 2]], (
            f"animation_axis_values must be positional; got {anim_args}"
        )

    def test_facet_mode_routes_kind_to_facet_call(self):
        """``kind`` (render-call-only) reaches ``cleo.facet``, not the ctor.

        Test scenario:
            The facet branch in ``render_array`` calls
            ``cleo.facet(**facet_kwargs, **render_kwargs)``. ``kind`` is
            a render-call-only kwarg, so it must surface inside the
            facet call's kwargs while ``cmap`` (constructor bucket)
            lands on ``__init__``.
        """
        fake_cls, ctor, _, _, facet, _ = self._capture_calls()
        rng = np.random.default_rng(404)
        arr = rng.random((3, 4, 4)).astype("float32")
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="facet",
                facet_kwargs={"col": "time", "col_coords": [0, 1, 2]},
                cmap="magma",
                kind="contourf",
            )
        assert "kind" in facet, f"`kind` should reach cleo.facet; facet kwargs={facet}"
        assert "cmap" in ctor, f"`cmap` should remain on the constructor; ctor={ctor}"
        assert facet.get("col") == "time", (
            f"facet_kwargs must reach cleo.facet via merge; got {facet}"
        )

    def test_split_is_driven_by_option_keys(self):
        """The ctor/render split comes from ``ArrayGlyph.option_keys()``.

        Test scenario:
            A constructor option declared by cleopatra (``add_colorbar``)
            must route to ``__init__`` because it is in ``option_keys()``,
            so the split tracks cleopatra automatically instead of a
            hand-maintained list. ``kind`` is the documented exception: it
            *is* in ``option_keys()`` yet is force-routed to the render
            call (it is an explicit ``plot`` param read from the signature,
            not from ``default_options`` — routing it to the ctor would
            pin every render to ``kind="auto"``).
        """
        assert "add_colorbar" in ArrayGlyph.option_keys()
        assert "kind" in ArrayGlyph.option_keys()
        fake_cls, ctor, plot, _, _, _ = self._capture_calls()
        rng = np.random.default_rng(505)
        arr = rng.random((4, 4)).astype("float32")
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                add_colorbar=False,
                kind="imshow",
            )
        assert "add_colorbar" in ctor and "add_colorbar" not in plot, (
            f"`add_colorbar` is an option_keys() ctor option; ctor={ctor}"
        )
        assert "kind" in plot and "kind" not in ctor, (
            f"`kind` must be force-routed to the render call; plot={plot}"
        )

    def test_loose_cbar_kwarg_rejected(self):
        """A loose ``cbar_*`` kwarg is rejected in favour of the typed ``ColorBar`` spec.

        Test scenario:
            cleopatra 0.30 removed the loose colour-bar styling keywords; pyramids
            exposes a single typed colour-bar surface, so a loose ``cbar_label`` /
            ``ticks_spacing`` raises a pyramids ``ValueError`` naming
            ``pyramids.plot.ColorBar`` rather than being folded or forwarded.
        """
        rng = np.random.default_rng(506)
        arr = rng.random((4, 4)).astype("float32")
        with pytest.raises(ValueError, match="ColorBar"):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                cbar_label="mm",
                ticks_spacing=2,
            )

    def test_loose_point_kwarg_rejected(self):
        """A loose ``point_*`` kwarg surfaces cleopatra's removed-keyword ``ValueError``.

        Test scenario:
            The point-styling keywords moved onto ``PointOverlay``; pyramids wraps a
            bare ``points`` array in a ``PointOverlay`` but no longer translates the
            loose ``point_*`` styling kwargs, so passing one reaches the real cleopatra
            render call, which rejects it.
        """
        rng = np.random.default_rng(507)
        arr = rng.random((4, 4)).astype("float32")
        with pytest.raises(ValueError):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                points=np.array([[1.0, 2, 3]]),
                point_color="red",
            )

    @pytest.mark.plot
    def test_kind_contourf_reaches_plot_not_clobbered(self):
        """Regression: ``kind="contourf"`` renders as contourf, not ``"auto"``.

        Test scenario:
            ``kind`` is in ``option_keys()`` yet ``ArrayGlyph.plot()``
            unconditionally rewrites ``default_options["kind"]`` with its own
            arg. The ``RENDER_ONLY_OVERRIDES`` set forces ``kind`` onto the
            render call so it is not clobbered back to ``"auto"``. Uses the
            real ``ArrayGlyph`` (not the fake) so the clobber path is actually
            exercised; the returned glyph must report ``"contourf"``.
        """
        rng = np.random.default_rng(909)
        arr = rng.random((5, 5)).astype("float32")
        glyph = render_array(
            arr=arr, extent=[0.0, 0.0, 1.0, 1.0], mode="plot", kind="contourf"
        )
        assert glyph.default_options["kind"] == "contourf", (
            "kind must reach ArrayGlyph.plot() and not be clobbered to 'auto'; "
            f"got {glyph.default_options.get('kind')!r}"
        )

    def test_invalid_kwarg_surfaces_cleopatra_valueerror(self):
        """An unknown kwarg is not swallowed — cleopatra raises ``ValueError``.

        Test scenario:
            A key absent from ``option_keys()`` (here ``bogus``) lands in
            ``render_kwargs`` and reaches the real ``ArrayGlyph.plot``,
            which rejects it. The routing must not silently drop unknown
            keys; it must defer to cleopatra's validation.
        """
        rng = np.random.default_rng(606)
        arr = rng.random((4, 4)).astype("float32")
        with pytest.raises(ValueError):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                bogus=1,
            )


class TestStyleHillshadePresets:
    """``style=`` / ``hillshade=`` presets route to cleopatra (#737).

    The presets ship in cleopatra >= 0.24. These tests render against the real
    installed cleopatra when it supports them, and simulate an older cleopatra
    (by hiding ``style`` from ``option_keys()``) to exercise the upgrade guard.
    """

    _supports_style = "style" in ArrayGlyph.option_keys()
    _supports_apply_style = hasattr(ArrayGlyph, "apply_style")

    @pytest.mark.skipif(
        not _supports_style, reason="cleopatra < 0.24 has no style presets"
    )
    def test_style_preset_renders(self):
        """A ``data_style=DataStyle(style=...)`` preset renders end to end.

        Test scenario:
            cleopatra 0.30 moved the preset name onto the typed ``DataStyle`` group,
            passed as ``data_style=``; ``render_array`` forwards it to the render call
            and returns a rendered glyph.
        """
        from cleopatra.styling.params import DataStyle

        rng = np.random.default_rng(737)
        arr = rng.random((6, 6)).astype("float32")
        glyph = render_array(
            arr=arr,
            extent=[0.0, 0.0, 1.0, 1.0],
            mode="plot",
            data_style=DataStyle(style="flow_accumulation"),
        )
        assert isinstance(glyph, ArrayGlyph)

    @pytest.mark.skipif(not _supports_style, reason="cleopatra < 0.24 has no hillshade")
    def test_hillshade_renders(self):
        """A ``data_style=DataStyle(hillshade=True)`` blend renders end to end."""
        from cleopatra.styling.params import DataStyle

        rng = np.random.default_rng(738)
        arr = rng.random((6, 6)).astype("float32")
        glyph = render_array(
            arr=arr,
            extent=[0.0, 0.0, 1.0, 1.0],
            mode="plot",
            data_style=DataStyle(hillshade=True),
        )
        assert isinstance(glyph, ArrayGlyph)

    @pytest.mark.skipif(
        not _supports_style, reason="cleopatra < 0.24 has no style presets"
    )
    def test_dict_hillshade_renders(self):
        """A dict ``hillshade`` on the ``DataStyle`` group renders end to end."""
        from cleopatra.styling.params import DataStyle

        rng = np.random.default_rng(742)
        arr = rng.random((6, 6)).astype("float32")
        glyph = render_array(
            arr=arr,
            extent=[0.0, 0.0, 1.0, 1.0],
            mode="plot",
            data_style=DataStyle(hillshade={"vert_exag": 8}),
        )
        assert isinstance(glyph, ArrayGlyph)

    @pytest.mark.skipif(
        not _supports_style, reason="cleopatra < 0.24 has no style presets"
    )
    def test_style_and_hillshade_animate_renders(self):
        """``data_style`` renders through the real animate path.

        Test scenario:
            Drives cleopatra's ``ArrayGlyph.animate(data_style=...)`` for real over a
            ``(time, rows, cols)`` stack, guarding the advertised animated-shaded path
            against a future cleopatra signature change.
        """
        from cleopatra.styling.params import DataStyle

        rng = np.random.default_rng(743)
        stack = rng.random((3, 6, 6)).astype("float32")
        glyph = render_array(
            arr=stack,
            extent=[0.0, 0.0, 1.0, 1.0],
            mode="animate",
            animation_axis_values=[0, 1, 2],
            data_style=DataStyle(style="flow_accumulation", hillshade=True),
        )
        assert isinstance(glyph, ArrayGlyph)

    @pytest.mark.skipif(
        not _supports_style, reason="cleopatra < 0.24 has no style presets"
    )
    def test_unknown_style_name_surfaces_cleopatra_valueerror(self):
        """An unknown ``style`` name is rejected by cleopatra with the valid list.

        Test scenario:
            pyramids does not duplicate the preset-name check; an invalid name
            defers to cleopatra, which raises ``ValueError`` naming the valid
            ``DATA_STYLES`` keys.
        """
        rng = np.random.default_rng(739)
        arr = rng.random((5, 5)).astype("float32")
        with pytest.raises(ValueError, match="style"):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                style="definitely_not_a_style",
            )

    @pytest.mark.skipif(
        not _supports_apply_style,
        reason="cleopatra < 0.25 has no glyph.apply_style()",
    )
    def test_returned_glyph_supports_apply_style(self):
        """The glyph from ``Dataset.plot`` can be restyled in place (cleopatra 0.25).

        Test scenario:
            cleopatra 0.25 adds ``ArrayGlyph.apply_style(style)`` and a ``style``
            read-back. Because pyramids returns the raw glyph, a caller holding it
            can re-apply a preset by name without rebuilding — verify the round
            trip through the ``Dataset.plot`` facade.
        """
        from cleopatra.styling.params import DataStyle

        rng = np.random.default_rng(750)
        arr = rng.random((1, 8, 8)).astype("float32")
        dataset = Dataset.create_from_array(
            arr=arr, geo=(0, 0.1, 0, 2, 0, -0.1), epsg=4326
        )
        glyph = dataset.plot(
            band=0, data_style=DataStyle(style="flow_accumulation")
        )
        assert glyph.style == "flow_accumulation"
        glyph.apply_style("topography")
        assert glyph.style == "topography", (
            "apply_style must restyle the glyph in place"
        )


class TestMeshRenderHelper:
    """Direct unit tests for the N-6 ``mesh_render`` sibling helper."""

    def test_mesh_render_basemap_without_epsg_raises(self):
        """``basemap=True`` + ``basemap_epsg=None`` raises ValueError.

        Test scenario:
            The mesh helper mirrors :func:`render_array`'s precondition:
            requesting a basemap without an EPSG must fail fast with a
            "CRS" hint before any rendering happens.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        sentinel = object()
        data = np.array([1.0])
        with pytest.raises(ValueError, match=r"CRS"):
            mesh_render(mesh=sentinel, data=data, basemap=True, basemap_epsg=None)

    def test_mesh_render_basemap_false_skips_add_basemap(self):
        """``basemap=False`` short-circuits before ``add_basemap`` is called.

        Test scenario:
            Patch ``plot_mesh_data`` to return a sentinel and patch the
            basemap module's ``add_basemap`` to record calls. With
            ``basemap=None`` the helper must return the sentinel without
            ever calling ``add_basemap``.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        sentinel = object()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=sentinel,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                result = mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    location="face",
                )
        assert result is sentinel, (
            f"mesh_render must return plot_mesh_data's result; got {result!r}"
        )
        mock_add.assert_not_called()

    def test_mesh_render_forwards_kwargs_to_plot_mesh_data(self):
        """Kwargs (``cmap``, ``vmin``, ``vmax``, ``title``) flow through.

        Test scenario:
            ``mesh_render`` is a thin dispatcher — every kwarg except
            ``basemap``/``basemap_epsg`` is forwarded to
            ``plot_mesh_data``. Capture the call and verify each kwarg
            is preserved with the same value the caller supplied.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        captured: dict = {}

        def _fake_plot(mesh, data, **kwargs):
            captured["mesh"] = mesh
            captured["data"] = data
            captured.update(kwargs)
            return "glyph"

        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            side_effect=_fake_plot,
        ):
            mesh = object()
            data = np.array([1.0, 2.0])
            mesh_render(
                mesh=mesh,
                data=data,
                location="node",
                cmap="plasma",
                vmin=0.0,
                vmax=10.0,
                title="t",
            )
        assert captured.get("location") == "node"
        assert captured.get("cmap") == "plasma"
        assert captured.get("vmin") == pytest.approx(0.0)
        assert captured.get("vmax") == pytest.approx(10.0)
        assert captured.get("title") == "t"

    def test_mesh_render_basemap_triggers_add_basemap_with_crs(self):
        """``basemap=True`` calls ``add_basemap`` with the supplied EPSG.

        Test scenario:
            With ``basemap=True`` and ``basemap_epsg=3857`` the helper
            must call ``pyramids.basemap.basemap.add_basemap`` once,
            forwarding the EPSG as the ``crs`` kwarg. The ax it picks up
            comes from the returned glyph's ``ax`` attribute, mirroring
            the raster path's behaviour.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        fake_glyph = type("G", (), {"ax": object()})()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=fake_glyph,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    basemap=True,
                    basemap_epsg=3857,
                )
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs.get("crs") == 3857, f"`crs` must equal basemap_epsg; got {kwargs}"
        assert kwargs.get("source") is None, (
            f"`source` should be None when basemap=True (no provider); got {kwargs}"
        )

    def test_mesh_render_basemap_string_passes_source(self):
        """``basemap='CartoDB.Positron'`` forwards the string as ``source=``.

        Test scenario:
            The mesh helper mirrors the raster path: a basemap string is
            forwarded as the contextily provider name via the ``source``
            kwarg. Boolean ``True`` passes ``source=None`` (already
            covered above); the string variant is verified here.
        """
        from pyramids.dataset._plot_helpers import mesh_render

        fake_glyph = type("G", (), {"ax": object()})()
        with patch(
            "pyramids.netcdf.ugrid.plot.plot_mesh_data",
            return_value=fake_glyph,
        ):
            with patch(
                "pyramids.basemap.basemap.add_basemap",
            ) as mock_add:
                mesh_render(
                    mesh=object(),
                    data=np.array([1.0]),
                    basemap="CartoDB.Positron",
                    basemap_epsg=4326,
                )
        kwargs = mock_add.call_args.kwargs
        assert kwargs.get("source") == "CartoDB.Positron", (
            f"`source` must equal the basemap string; got {kwargs}"
        )


@pytest.mark.skipif(PointOverlay is None, reason="cleopatra < 0.26 has no PointOverlay")
class TestPointOverlay:
    """The `points=` overlay reaches cleopatra from the plot facade (cleopatra 0.26).

    cleopatra 0.26 bundles the point-styling kwargs into a `PointOverlay` object.
    pyramids forwards `points=` verbatim as a render-call-only kwarg, so the object
    flows through untouched — these tests pin the form the docs tell callers to use.
    """

    @staticmethod
    def _dataset():
        """Build a small single-band in-memory dataset."""
        rng = np.random.default_rng(760)
        arr = rng.random((1, 8, 8)).astype("float32")
        return Dataset.create_from_array(
            arr=arr, geo=(0, 0.1, 0, 2, 0, -0.1), epsg=4326
        )

    @staticmethod
    def _points():
        """Points as pyramids documents them: (value, row index, column index)."""
        return np.array([[1.0, 2, 3], [2.0, 4, 5]])

    def test_styled_point_overlay_renders(self):
        """A styled `PointOverlay` renders — the form the docs recommend."""
        glyph = self._dataset().plot(
            band=0,
            points=PointOverlay(self._points(), color="red", label_color="blue"),
        )
        assert isinstance(glyph, ArrayGlyph)

    def test_plain_points_array_renders(self):
        """A bare `points` array renders — the unstyled path."""
        glyph = self._dataset().plot(band=0, points=self._points())
        assert isinstance(glyph, ArrayGlyph)

    def test_point_overlay_reaches_the_render_call(self):
        """`points=` is routed to the render call, not the glyph constructor.

        `points` is not in `ArrayGlyph.option_keys()`, so the D-4 split must send
        it to `cleo.plot(...)`. A fake glyph captures where it lands.
        """
        fake_cls, ctor, plot, *_ = TestRenderArrayKwargRouting._capture_calls()
        overlay = PointOverlay(self._points(), color="red")
        rng = np.random.default_rng(761)
        arr = rng.random((5, 5)).astype("float32")
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr, extent=[0.0, 0.0, 1.0, 1.0], mode="plot", points=overlay
            )
        assert plot.get("points") is overlay, "points must reach the render call"
        assert "points" not in ctor, "points must not land on the constructor"


@pytest.mark.skipif(FrameLabel is None, reason="cleopatra < 0.26 has no FrameLabel")
class TestFrameLabel:
    """`frame_label=` reaches the real `ArrayGlyph.animate` (cleopatra 0.26).

    cleopatra 0.26 replaces `animate()`'s `text_loc` with a `FrameLabel` object.
    `docs/examples/dataset/dataset_collection.ipynb` is otherwise the only thing
    exercising this path, and it runs in a separate CI job — so a change to the
    animate signature would surface in the docs build rather than here.
    """

    @staticmethod
    def _stack():
        """A small `(time, rows, cols)` stack to animate."""
        rng = np.random.default_rng(762)
        return rng.random((3, 6, 6)).astype("float32")

    def test_frame_label_animates(self):
        """`FrameLabel` animates — the form the collection notebook uses.

        Test scenario:
            Drives `render_array(mode="animate", frame_label=FrameLabel(...))`
            against the real glyph, so a change to cleopatra's animate signature
            fails here rather than only in the notebook job.
        """
        glyph = render_array(
            arr=self._stack(),
            extent=[0.0, 0.0, 1.0, 1.0],
            mode="animate",
            animation_axis_values=[0, 1, 2],
            frame_label=FrameLabel(location=(1, 3)),
        )
        assert isinstance(glyph, ArrayGlyph)


class TestRenderArrayGroupParams:
    """Explicit cleopatra render-group objects and their precedence in render_array.

    The typed groups (``color`` / ``contour`` / ``cells`` / ``data_style``) reach the
    render call, and an explicitly-passed group wins over the one built from the loose
    kwargs (``color_scale`` / ``style`` / ...).
    """

    def test_explicit_color_group_reaches_render_call(self):
        """An explicit ``color=ColorScaling`` is forwarded to the render call.

        Test scenario:
            The typed ``color`` group must reach ``ArrayGlyph.plot`` verbatim (the loose
            ``color_scale`` kwarg it replaces was removed and now raises).
        """
        from pyramids.plot import ColorScaling

        fake_cls, _ctor, plot, *_ = TestRenderArrayKwargRouting._capture_calls()
        rng = np.random.default_rng(11)
        arr = rng.random((4, 4)).astype("float32")
        explicit = ColorScaling.power(gamma=0.9)
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                color=explicit,
            )
        assert plot.get("color") is explicit, "explicit color must reach the render call"

    def test_explicit_groups_reach_render_call(self):
        """Explicit ``contour`` / ``cells`` / ``data_style`` reach the render call.

        Test scenario:
            The typed groups other than ``color`` also forward verbatim to
            ``ArrayGlyph.plot``.
        """
        from cleopatra.styling.params import CellValues, Contour, DataStyle

        fake_cls, _ctor, plot, *_ = TestRenderArrayKwargRouting._capture_calls()
        rng = np.random.default_rng(13)
        arr = rng.random((4, 4)).astype("float32")
        contour, cells, data_style = Contour(levels=4), CellValues(show=True), DataStyle(
            style="flow_accumulation"
        )
        with patch("cleopatra.glyphs.gridded.array_glyph.ArrayGlyph", new=fake_cls):
            render_array(
                arr=arr,
                extent=[0.0, 0.0, 1.0, 1.0],
                mode="plot",
                contour=contour,
                cells=cells,
                data_style=data_style,
            )
        assert plot.get("contour") is contour, "explicit contour must reach the render call"
        assert plot.get("cells") is cells, "explicit cells must reach the render call"
        assert plot.get("data_style") is data_style, "explicit data_style must reach it"
