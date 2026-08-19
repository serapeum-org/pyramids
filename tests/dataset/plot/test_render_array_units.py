"""Fast unit tests for the render_array decomposition value objects (#1006).

These target the cleopatra-free logic of the components extracted from
``render_array`` — ``_split_render_kwargs``, ``RgbSpec``, ``ModeSpec``,
``RenderRequest``, and ``BasemapPlan``. The cleopatra classes the RGB methods
need are injected as fakes and ``add_basemap`` is mocked, so this module runs in
the default (non-``plot``) suite without the ``[viz]`` extra.
"""

import numpy as np
import pytest

from pyramids.dataset._plot_helpers import (
    BasemapPlan,
    ModeSpec,
    RenderRequest,
    RgbSpec,
    _split_render_kwargs,
)


def _fake_array_glyph():
    """Return a (fake ArrayGlyph class, calls list) for RgbSpec.composite_animate_frames.

    The compositor instantiates the class internally, so the per-frame stretch kwargs
    are recorded into a closed-over ``calls`` list rather than shared class state — a
    fresh list per factory call keeps tests isolated.
    """
    calls: list = []

    class _Fake:
        def __init__(self, array):
            self.array = array

        def prepare_array(self, frame, *, rgb, surface_reflectance, cutoff, percentile):
            calls.append(
                {
                    "rgb": rgb,
                    "surface_reflectance": surface_reflectance,
                    "cutoff": cutoff,
                    "percentile": percentile,
                }
            )
            rows, cols = frame.shape[-2:]
            return np.zeros((rows, cols, 3), dtype="float32")

    return _Fake, calls


class _FakeRgbBands:
    """Stand-in for cleopatra's RgbBands recording the constructor arguments."""

    def __init__(
        self, indices, *, surface_reflectance=None, cutoff=None, percentile=None
    ):
        """Store the band indices and stretch controls verbatim."""
        self.indices = indices
        self.surface_reflectance = surface_reflectance
        self.cutoff = cutoff
        self.percentile = percentile


class TestSplitRenderKwargs:
    """Tests for _split_render_kwargs."""

    def test_plot_routes_options_to_ctor_and_rest_to_render(self):
        """Option keys land on the ctor bucket; non-option keys on the render bucket.

        Test scenario:
            In ``plot`` mode a key in ``option_keys`` goes to ctor and a render-only
            key goes to render; the animate bucket is empty.
        """
        ctor, render, animate = _split_render_kwargs(
            {"cmap": "viridis", "points": [[1, 2, 3]]}, "plot", {"cmap", "vmin"}
        )
        assert ctor == {"cmap": "viridis"}, f"ctor bucket wrong: {ctor}"
        assert render == {"points": [[1, 2, 3]]}, f"render bucket wrong: {render}"
        assert animate == {}, f"animate bucket must be empty in plot mode: {animate}"

    def test_kind_is_force_routed_to_render_even_when_an_option(self):
        """``kind`` reaches the render call even though it is an option key.

        Test scenario:
            ``kind`` is in ``option_keys`` but must be routed to the render call so
            cleopatra does not clobber it back to ``"auto"``.
        """
        ctor, render, _ = _split_render_kwargs(
            {"cmap": "viridis", "kind": "contourf"}, "plot", {"cmap", "kind"}
        )
        assert "kind" not in ctor, f"kind must not stay on the ctor: {ctor}"
        assert render["kind"] == "contourf", f"kind must reach render: {render}"

    def test_animate_empties_ctor_and_merges_into_animate(self):
        """Animate mode empties the ctor bucket and merges both into the animate bucket.

        Test scenario:
            cleopatra's ``animate`` re-validates every kwarg, so nothing rides on the
            constructor and the animate bucket carries the union.
        """
        ctor, render, animate = _split_render_kwargs(
            {"cmap": "viridis", "kind": "contourf"}, "animate", {"cmap", "kind"}
        )
        assert ctor == {}, f"ctor must be empty in animate mode: {ctor}"
        assert animate == {"cmap": "viridis", "kind": "contourf"}, (
            f"animate bucket must be the merge: {animate}"
        )

    def test_unknown_key_falls_to_render(self):
        """A key outside ``option_keys`` is routed to the render call, not dropped.

        Test scenario:
            An unknown key must reach the render method (which rejects it) rather
            than being silently swallowed.
        """
        ctor, render, _ = _split_render_kwargs({"bogus": 1}, "plot", {"cmap"})
        assert render == {"bogus": 1}, f"unknown key must fall to render: {render}"
        assert ctor == {}, f"ctor must be empty: {ctor}"

    def test_empty_kwargs_yields_empty_buckets(self):
        """No kwargs produces three empty buckets.

        Test scenario:
            An empty kwargs dict returns empty ctor / render / animate buckets.
        """
        assert _split_render_kwargs({}, "plot", {"cmap"}) == ({}, {}, {})


class TestRgbSpec:
    """Tests for the RgbSpec value object."""

    def test_is_set_false_by_default_true_with_rgb(self):
        """``is_set`` reflects whether an rgb band list was supplied.

        Test scenario:
            Empty spec is the single-band path; a spec with ``rgb`` is RGB.
        """
        assert RgbSpec().is_set is False, "empty spec must report is_set False"
        assert RgbSpec(rgb=[0, 1, 2]).is_set is True, "rgb spec must report is_set True"

    def test_to_cleo_bands_none_when_unset(self):
        """An unset spec yields no cleopatra bands.

        Test scenario:
            ``to_cleo_bands`` returns None for the single-band path so the ctor gets
            ``rgb_bands=None``.
        """
        assert RgbSpec().to_cleo_bands(_FakeRgbBands) is None, (
            "unset spec must give None"
        )

    def test_to_cleo_bands_builds_populated_bands(self):
        """A set spec builds bands echoing indices and stretch controls.

        Test scenario:
            ``to_cleo_bands`` forwards indices positionally and each stretch control
            by keyword.
        """
        bands = RgbSpec(
            rgb=[2, 1, 0],
            surface_reflectance=10000,
            cutoff=[0.1, 0.2, 0.3],
            percentile=2,
        ).to_cleo_bands(_FakeRgbBands)
        assert bands.indices == [2, 1, 0], f"indices wrong: {bands.indices}"
        assert bands.surface_reflectance == 10000, "surface_reflectance not forwarded"
        assert bands.cutoff == [0.1, 0.2, 0.3], "cutoff not forwarded"
        assert bands.percentile == 2, "percentile not forwarded"

    def test_composite_animate_frames_stacks_display_ready_frames(self):
        """Compositing a 4-D stack returns a (time, rows, cols, 3) stack.

        Test scenario:
            Each timestep is prepared via the injected glyph and stacked on axis 0;
            the stretch kwargs reach ``prepare_array`` for every frame.
        """
        fake_cls, calls = _fake_array_glyph()
        spec = RgbSpec(rgb=[0, 1, 2], percentile=2)
        arr = np.zeros((4, 3, 5, 6), dtype="float32")
        out = spec.composite_animate_frames(arr, fake_cls)
        assert out.shape == (4, 5, 6, 3), f"composited shape wrong: {out.shape}"
        assert len(calls) == 4, "prepare_array must run once per frame"
        assert calls[0]["percentile"] == 2, "stretch kwargs not forwarded"

    @pytest.mark.parametrize("arr", [None, np.zeros((4, 5, 6), dtype="float32")])
    def test_composite_animate_frames_rejects_non_4d(self, arr):
        """A missing or non-4-D array is rejected before any compositing.

        Args:
            arr: A None array or a 3-D stack — both invalid for RGB animate.

        Test scenario:
            Guards the #538 frame loss by refusing a single-band 3-D stack (or None).
        """
        fake_cls, _ = _fake_array_glyph()
        spec = RgbSpec(rgb=[0, 1, 2])
        with pytest.raises(ValueError, match="4-D"):
            spec.composite_animate_frames(arr, fake_cls)


class TestModeSpec:
    """Tests for the ModeSpec value object."""

    def test_defaults(self):
        """The default ModeSpec is a bare plot request.

        Test scenario:
            Unset fields default to plot mode with no animate/facet inputs.
        """
        spec = ModeSpec()
        assert spec.mode == "plot", f"default mode must be plot: {spec.mode}"
        assert spec.animation_axis_values is None, "animation axis default must be None"
        assert spec.data_getter is None, "data_getter default must be None"
        assert spec.facet_kwargs is None, "facet_kwargs default must be None"


class TestRenderRequest:
    """Tests for the RenderRequest parameter object."""

    def test_defaults_compose_empty_specs(self):
        """The request defaults to an empty RgbSpec and a plot ModeSpec.

        Test scenario:
            Only ``arr`` is required; ``rgb`` / ``mode`` default via factories.
        """
        req = RenderRequest(arr=np.zeros((4, 4)))
        assert req.rgb.is_set is False, "default rgb must be an unset RgbSpec"
        assert req.mode.mode == "plot", "default mode must be a plot ModeSpec"
        assert req.basemap is None, "default basemap must be None"

    def test_validate_accepts_valid_plot(self):
        """A plain plot request validates without error.

        Test scenario:
            A default request has a valid mode and no basemap, so validate passes.
        """
        RenderRequest(arr=np.zeros((4, 4))).validate()

    def test_validate_rejects_unknown_mode(self):
        """An unknown mode is rejected.

        Test scenario:
            ``mode.mode`` outside plot/animate/facet raises naming the bad value.
        """
        request = RenderRequest(arr=np.zeros((4, 4)), mode=ModeSpec(mode="bogus"))
        with pytest.raises(ValueError, match="Invalid mode='bogus'"):
            request.validate()

    def test_validate_requires_animation_axis_values(self):
        """Animate mode requires animation_axis_values.

        Test scenario:
            ``mode='animate'`` without axis values raises.
        """
        request = RenderRequest(arr=np.zeros((4, 4)), mode=ModeSpec(mode="animate"))
        with pytest.raises(ValueError, match="animation_axis_values"):
            request.validate()

    def test_validate_requires_facet_kwargs(self):
        """Facet mode requires facet_kwargs.

        Test scenario:
            ``mode='facet'`` with no facet_kwargs raises.
        """
        request = RenderRequest(arr=np.zeros((4, 4)), mode=ModeSpec(mode="facet"))
        with pytest.raises(ValueError, match="facet_kwargs"):
            request.validate()

    def test_validate_requires_epsg_for_basemap(self):
        """A truthy basemap without a CRS is rejected.

        Test scenario:
            ``basemap`` set while ``basemap_epsg`` is None raises the CRS error.
        """
        request = RenderRequest(arr=np.zeros((4, 4)), basemap="OpenStreetMap")
        with pytest.raises(ValueError, match="CRS"):
            request.validate()


class _FakePanel:
    """A minimal facet panel exposing get_visible for apply_to_facets tests."""

    def __init__(self, visible):
        """Store the panel's visibility flag."""
        self._visible = visible

    def get_visible(self):
        """Return whether this panel is visible."""
        return self._visible


class _FakeGrid:
    """A minimal facet grid exposing an ``axes`` array."""

    def __init__(self, axes):
        """Store the panel axes array."""
        self.axes = axes


class TestBasemapPlan:
    """Tests for the BasemapPlan value object."""

    def test_resolve_provider_string_is_a_tile_plan(self):
        """A provider string resolves to a web-tile plan.

        Test scenario:
            A non-empty string means tile mode with that source and no cleo basemap.
        """
        plan = BasemapPlan.resolve("OpenStreetMap", 4326)
        assert plan.tile is True, "string basemap must be a tile plan"
        assert plan.source == "OpenStreetMap", f"source wrong: {plan.source}"
        assert plan.forwards_cleo_basemap is False, (
            "string must not forward a cleo basemap"
        )
        assert plan.cleo_kwarg == {}, "no cleo basemap -> empty kwarg"

    def test_resolve_true_is_default_provider_tile(self):
        """``True`` resolves to a tile plan with the default provider.

        Test scenario:
            ``basemap=True`` means tile mode with ``source=None`` (default provider).
        """
        plan = BasemapPlan.resolve(True, 3857)
        assert plan.tile is True, f"True must resolve to a tile plan: {plan}"
        assert plan.source is None, f"True must use the default provider: {plan}"

    @pytest.mark.parametrize("value", ["", None, False, {}])
    def test_resolve_falsy_is_no_basemap(self, value):
        """Falsy inputs resolve to a no-op plan.

        Args:
            value: An empty string / None / False / empty dict.

        Test scenario:
            Every falsy basemap means no tile and no forwarded cleo basemap.
        """
        plan = BasemapPlan.resolve(value, 4326)
        assert plan.tile is False, f"falsy basemap must not tile: {value!r}"
        assert plan.forwards_cleo_basemap is False, f"falsy must not forward: {value!r}"

    def test_resolve_object_forwards_cleo_basemap(self):
        """A non-str/bool object resolves to a forwarded cleopatra basemap.

        Test scenario:
            A ``Basemap``-like object is not a tile; it is forwarded on the render call.
        """
        marker = object()
        plan = BasemapPlan.resolve(marker, 4326)
        assert plan.tile is False, "an object basemap must not tile"
        assert plan.forwards_cleo_basemap is True, "an object basemap must forward"
        assert plan.cleo_kwarg == {"basemap": marker}, (
            f"cleo_kwarg wrong: {plan.cleo_kwarg}"
        )

    def test_apply_to_calls_add_basemap(self, monkeypatch):
        """``apply_to`` draws the tile basemap on the given axes via add_basemap.

        Test scenario:
            ``apply_to`` resolves ``add_basemap`` at call time and passes the axes,
            the CRS, and the provider source.
        """
        seen = {}

        def _fake_add_basemap(ax, *, crs, source):
            seen.update(ax=ax, crs=crs, source=source)

        monkeypatch.setattr(
            "pyramids.basemap.basemap.add_basemap", _fake_add_basemap, raising=True
        )
        BasemapPlan.resolve("CartoDB.Positron", 4326).apply_to("AX")
        assert seen == {"ax": "AX", "crs": 4326, "source": "CartoDB.Positron"}, (
            f"add_basemap called with wrong args: {seen}"
        )

    def test_apply_to_facets_only_visible_panels(self, monkeypatch):
        """``apply_to_facets`` draws under visible panels only, skipping hidden/None.

        Test scenario:
            A grid with a visible panel, a hidden panel, and a None slot draws the
            basemap exactly once (under the visible panel).
        """
        drawn = []
        monkeypatch.setattr(
            "pyramids.basemap.basemap.add_basemap",
            lambda ax, *, crs, source: drawn.append(ax),
            raising=True,
        )
        grid = _FakeGrid(
            np.array([_FakePanel(True), _FakePanel(False), None], dtype=object)
        )
        BasemapPlan.resolve("OpenStreetMap", 4326).apply_to_facets(grid)
        assert len(drawn) == 1, f"only the visible panel must be drawn: {len(drawn)}"

    def test_apply_to_facets_no_axes_is_noop(self, monkeypatch):
        """A grid without an ``axes`` attribute is a no-op.

        Test scenario:
            When ``grid.axes`` is None nothing is drawn.
        """
        drawn = []
        monkeypatch.setattr(
            "pyramids.basemap.basemap.add_basemap",
            lambda ax, *, crs, source: drawn.append(ax),
            raising=True,
        )
        BasemapPlan.resolve("OpenStreetMap", 4326).apply_to_facets(_FakeGrid(None))
        assert drawn == [], "a grid with no axes must draw nothing"
