"""Plot tests: multi-panel faceting."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.netcdf import ColourOpts, FacetSpec, Selectors
from pyramids.netcdf._plot import NetCDFPlot
from tests.netcdf._plot_helpers import _make_4d_nc, _make_curvilinear_nc
from tests.netcdf.conftest import make_plot_3d_nc

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
_FacetGrid = _cleo_array.FacetGrid
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")


class TestNetCDFPlotFaceting:
    """PR-4 — `col=`/`row=`/`col_wrap=` build a multi-subplot grid.

    Each test exercises the cleopatra `ArrayGlyph.facet` wiring on
    NetCDF.plot. The returned object is a
    :class:`cleopatra.array_glyph.FacetGrid`; subplot count and the
    `name_dicts` index map confirm the dispatch.
    """

    def test_col_only_returns_facet_grid_with_one_row(self):
        """`col="time"` on a 3-D variable returns N=time_len subplots."""
        nc = make_plot_3d_nc(n_times=4)
        grid = nc.plot(variable="t2m", facet=FacetSpec(col="time"))
        assert isinstance(grid, _FacetGrid)
        assert grid.axes.shape == (1, 4)
        assert len(grid.name_dicts) == 4
        assert all("time" in d for d in grid.name_dicts)

    def test_col_and_row_on_4d_returns_2d_grid(self):
        """`col="time"` + `row="pressure_level"` on a 4-D var → 2-D grid."""
        nc = _make_4d_nc()
        grid = nc.plot(
            variable="temperature",
            facet=FacetSpec(col="time", row="pressure_level"),
        )
        assert isinstance(grid, _FacetGrid)
        # _make_4d_nc has nt=3, nl=2 — so axes shape (ncols, nrows) = (3, 2)
        # cleopatra builds subplots with shape (nrows, ncols) where
        # nrows=n_row, ncols=n_col.
        assert grid.axes.shape == (2, 3)
        assert len(grid.name_dicts) == 6
        for d in grid.name_dicts:
            assert "time" in d
            assert "pressure_level" in d

    def test_col_wrap_wraps_into_grid(self):
        """`col_wrap=3` wraps N=4 panels into a 2x3 grid."""
        nc = make_plot_3d_nc(n_times=4)
        grid = nc.plot(
            variable="t2m",
            facet=FacetSpec(col="time", col_wrap=3),
        )
        assert isinstance(grid, _FacetGrid)
        assert grid.axes.shape == (2, 3)
        # The last panel is hidden when N=4 doesn't fill a 2x3 grid.
        hidden = [ax for ax in grid.axes.ravel() if not ax.get_visible()]
        assert len(hidden) == 2

    def test_pin_one_dim_facet_over_another(self):
        """`level=500, col="time"` pins level first, then facets over time."""
        nc = _make_4d_nc()
        grid = nc.plot(
            variable="temperature",
            selectors=Selectors(level=500),
            facet=FacetSpec(col="time"),
        )
        assert isinstance(grid, _FacetGrid)
        # nt=3 → 3 subplots in a single row.
        assert grid.axes.shape == (1, 3)
        assert len(grid.name_dicts) == 3

    def test_conflict_time_kwarg_and_col_time(self):
        """`time=0, col="time"` is rejected — the same dim cannot be both."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(time=0),
                facet=FacetSpec(col="time"),
            )

    def test_conflict_isel_and_col(self):
        """`isel={"time": 0}, col="time"` is also rejected with the same hint."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(isel={"time": 0}),
                facet=FacetSpec(col="time"),
            )

    def test_conflict_sel_and_col(self):
        """`sel={"time": 0}, col="time"` is also rejected."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(sel={"time": 0}),
                facet=FacetSpec(col="time"),
            )

    def test_invalid_col_wrap_raises(self):
        """`col_wrap=0` or non-int raises ValueError."""
        nc = make_plot_3d_nc(n_times=4)
        with pytest.raises(ValueError, match=r"positive int"):
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time", col_wrap=0),
            )


class TestNetCDFPlotFacetingEdges:
    """PR-4 edge cases not covered by :class:`TestNetCDFPlotFaceting`.

    Coverage targets degenerate grids, ``col_wrap`` bounds, conflict
    error messages, faceting interaction with pinned dims, curvilinear
    coord forwarding to facet cells, and the returned FacetGrid
    attribute contract.
    """

    def test_col_wrap_one_produces_single_column(self):
        """`col_wrap=1` arranges N panels into N rows × 1 col."""
        nc = make_plot_3d_nc(n_times=4)
        grid = nc.plot(
            variable="t2m",
            facet=FacetSpec(col="time", col_wrap=1),
        )
        assert isinstance(grid, _FacetGrid)
        assert grid.axes.shape == (
            4,
            1,
        ), f"col_wrap=1 should yield 4 rows × 1 col, got {grid.axes.shape}"
        assert len(grid.name_dicts) == 4

    def test_col_wrap_larger_than_panel_count(self):
        """`col_wrap=8` with N=4 panels still works (single row, 4 visible).

        Test scenario:
            When ``col_wrap`` exceeds the number of facet panels
            cleopatra produces a single row whose visible-panel count
            equals the panel count. The pyramids layer simply forwards
            the wrap; we confirm the grid is constructed without
            crashing and has the expected shape.
        """
        nc = make_plot_3d_nc(n_times=4)
        grid = nc.plot(
            variable="t2m",
            facet=FacetSpec(col="time", col_wrap=8),
        )
        assert isinstance(grid, _FacetGrid)
        assert grid.axes.shape == (
            1,
            8,
        ), f"col_wrap=8 with N=4 should yield a 1×8 grid, got {grid.axes.shape}"
        visible = [ax for ax in grid.axes.ravel() if ax.get_visible()]
        assert (
            len(visible) == 4
        ), f"Exactly 4 panels should be visible, got {len(visible)}"

    def test_col_with_single_step_degenerate_grid(self):
        """`col="time"` with N=1 yields a 1×1 grid (degenerate but valid).

        Test scenario:
            A variable with a single time step still satisfies the
            facet contract; the resulting grid has one cell. Used to
            guard against off-by-one bugs in the stack builder.
        """
        nc = make_plot_3d_nc(n_times=1)
        grid = nc.plot(variable="t2m", facet=FacetSpec(col="time"))
        assert isinstance(grid, _FacetGrid)
        assert grid.axes.shape == (
            1,
            1,
        ), f"Single-step facet should yield (1, 1), got {grid.axes.shape}"
        assert len(grid.name_dicts) == 1

    def test_facet_dim_unknown_lists_available_dims(self):
        """`col="bogus"` error message lists the actual band dim names.

        Test scenario:
            The validator must include the available band dim names so
            the user can pick a valid one without re-reading the
            variable. Verify the message contains both ``bogus`` and
            ``time`` (the real band dim).
        """
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"not a band dim") as exc_info:
            nc.plot(variable="t2m", facet=FacetSpec(col="bogus"))
        msg = str(exc_info.value)
        assert "bogus" in msg, f"Error must echo the bad name, got: {msg}"
        assert "time" in msg, f"Error must list 'time' as available, got: {msg}"

    def test_row_alone_error_mentions_col_requirement(self):
        """`row=` alone error message explicitly mentions `col=` requirement."""
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"requires `col=`") as exc_info:
            nc.plot(variable="t2m", facet=FacetSpec(row="time"))
        assert "col=" in str(
            exc_info.value
        ), f"Error must mention col= requirement, got: {exc_info.value}"

    def test_facet_with_sel_pinning_other_dim(self):
        """Faceting `col="time"` plus `sel={"pressure_level": 500}` succeeds.

        Test scenario:
            A 4-D variable with both ``time`` and ``pressure_level``
            band dims: pin ``pressure_level`` via ``sel`` and facet
            over ``time``. The validator must not flag a conflict
            because the dims differ. The resulting grid has
            ``time_len`` panels.
        """
        nc = _make_4d_nc()
        grid = nc.plot(
            variable="temperature",
            selectors=Selectors(sel={"pressure_level": 500}),
            facet=FacetSpec(col="time"),
        )
        assert isinstance(grid, _FacetGrid)
        # _make_4d_nc has nt=3
        assert grid.axes.shape == (
            1,
            3,
        ), f"sel-pinned level + col=time should be (1, 3), got {grid.axes.shape}"

    def test_facet_with_kind_pcolormesh_forwarded(self):
        """Faceting + `kind="pcolormesh"` forwards the kind to cleo.facet.

        Test scenario:
            ``kind`` is set via the cleo constructor (stored as a
            default) and consumed by ``ArrayGlyph.facet``. We patch
            ``Analysis.plot`` and confirm both ``facet_kwargs`` and
            ``kind="pcolormesh"`` reach the engine.
        """
        nc = make_plot_3d_nc(n_times=3)
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            nc.plot(variable="t2m", facet=FacetSpec(col="time"), kind="pcolormesh")
        call_kwargs = mock_plot.call_args.kwargs
        assert (
            call_kwargs.get("kind") == "pcolormesh"
        ), f"kind should reach Analysis.plot, got: {call_kwargs}"
        assert (
            "facet_kwargs" in call_kwargs
        ), f"facet_kwargs must be present, got: {list(call_kwargs)}"

    def test_facet_with_curvilinear_coords_forwarded(self):
        """Faceting + curvilinear `coords=` forwards both to the engine.

        Test scenario:
            On a 3-D curvilinear variable, faceting over ``time`` plus
            an explicit ``coords=("XLONG", "XLAT")`` must forward both
            kwargs. We patch ``Analysis.plot`` and inspect: the
            ``coords`` kwarg must carry resolved 2-D arrays and the
            ``facet_kwargs`` dict must include ``col``.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=4, cols=5, n_times=3)
        var = nc.get_variable("CANWAT")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            nc.plot(
                variable="CANWAT",
                facet=FacetSpec(col="time"),
                coords=("XLONG", "XLAT"),
            )
        call_kwargs = mock_plot.call_args.kwargs
        assert "facet_kwargs" in call_kwargs, "facet_kwargs missing"
        assert "coords" in call_kwargs, "coords missing alongside facet_kwargs"
        assert call_kwargs["coords"][0].shape == (
            4,
            5,
        ), f"coords[0] should be 2D (4, 5), got {call_kwargs['coords'][0].shape}"

    def test_facet_with_robust_forwarded(self):
        """Faceting + `robust=True` forwards the percentile-stretch flag.

        Test scenario:
            cleopatra applies the percentile stretch over the full
            stack when ``robust=True`` is set at the constructor
            level. Pyramids must forward the flag verbatim. We patch
            ``Analysis.plot`` and assert ``robust=True`` is in the
            call kwargs alongside ``facet_kwargs``.
        """
        nc = make_plot_3d_nc(n_times=3)
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "stub"
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time"),
                colour=ColourOpts(robust=True),
            )
        call_kwargs = mock_plot.call_args.kwargs
        assert (
            call_kwargs.get("robust") is True
        ), f"robust=True must reach Analysis.plot, got: {call_kwargs}"
        assert "facet_kwargs" in call_kwargs

    def test_facet_grid_exposes_fig_axes_cbar_name_dicts(self):
        """Returned FacetGrid carries `.fig`, `.axes`, `.cbar`, `.name_dicts`."""
        nc = make_plot_3d_nc(n_times=3)
        grid = nc.plot(variable="t2m", facet=FacetSpec(col="time"))
        assert hasattr(grid, "fig"), "FacetGrid must expose .fig"
        assert hasattr(grid, "axes"), "FacetGrid must expose .axes"
        assert hasattr(grid, "cbar"), "FacetGrid must expose .cbar"
        assert hasattr(grid, "name_dicts"), "FacetGrid must expose .name_dicts"
        assert grid.fig is not None, "FacetGrid.fig must not be None"
        assert grid.axes.shape == (1, 3)
        assert isinstance(grid.name_dicts, list)

    def test_facet_invalid_col_wrap_non_integer_raises(self):
        """`col_wrap="three"` (str) raises with the positive-int hint."""
        nc = make_plot_3d_nc(n_times=4)
        with pytest.raises(ValueError, match=r"positive int") as exc_info:
            nc.plot(variable="t2m", facet=FacetSpec(col="time", col_wrap="three"))
        assert "col_wrap" in str(
            exc_info.value
        ), f"Error must reference col_wrap, got: {exc_info.value}"

    def test_facet_4d_col_row_grid_shape(self):
        """4-D facet with `col="time", row="pressure_level"` matches dim sizes.

        Test scenario:
            ``_make_4d_nc`` has ``nt=3`` and ``nl=2``. The faceted
            grid axes attribute has shape ``(nrows, ncols)`` where
            ``nrows=len(row)=2`` and ``ncols=len(col)=3``. The
            ``name_dicts`` list has length ``nt * nl = 6``, and every
            entry references both dim names.
        """
        nc = _make_4d_nc()
        grid = nc.plot(
            variable="temperature",
            facet=FacetSpec(col="time", row="pressure_level"),
        )
        assert grid.axes.shape == (
            2,
            3,
        ), f"4-D col+row grid should be (nrows=2, ncols=3), got {grid.axes.shape}"
        assert (
            len(grid.name_dicts) == 6
        ), f"name_dicts should have nt*nl=6 entries, got {len(grid.name_dicts)}"
        for entry in grid.name_dicts:
            assert "time" in entry, f"Missing 'time' in {entry}"
            assert "pressure_level" in entry, f"Missing 'pressure_level' in {entry}"

    def test_facet_4d_stack_values_match_sel_read(self):
        """The 4-D facet stride read produces the same panels as the old sel() path (gap G6).

        Test scenario:
            ``_build_facet_stack`` reads each panel by a flat band index
            (``ci*col_stride + ri*row_stride``) instead of allocating a ``sel()`` subset per
            panel. Assert that, for a ``(time, pressure_level, lat, lon)`` variable, every
            ``stack[ci][ri]`` panel equals the array the old
            ``sel(time=...).sel(pressure_level=...).read_array(band=0)`` path produces —
            pinning value-equivalence, not just the grid shape.
        """
        nc = _make_4d_nc()
        sub = nc.get_variable("temperature")
        stack, _fkw = NetCDFPlot(sub)._build_facet_stack(
            sub, col="time", row="pressure_level", col_wrap=None
        )
        assert stack.shape[:2] == (3, 2), f"expected (ncol=3, nrow=2) panels, got {stack.shape[:2]}"

        times = [0, 6, 12]
        levels = [1000, 500]
        for ci, t in enumerate(times):
            for ri, lvl in enumerate(levels):
                expected = np.asarray(
                    sub.sel(time=t).sel(pressure_level=lvl).read_array(band=0)
                )
                np.testing.assert_array_equal(
                    stack[ci][ri],
                    expected,
                    err_msg=f"facet panel (time={t}, level={lvl}) != sel().read_array(band=0)",
                )

    def test_facet_with_basemap_adds_one_per_panel(self):
        """`facet=...` + `basemap=True` overlays a tile layer on every panel (M4).

        Test scenario:
            M4 fix — the facet path used to silently drop `basemap=`.
            It now iterates the facet panels and calls `add_basemap`
            on each visible `Axes`. Patch `add_basemap` and confirm it
            fires once per panel (3 panels for `col="time"` over a
            3-time-step variable), each with the subset's `crs`.
        """
        nc = make_plot_3d_nc(n_times=3)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time"),
                basemap=True,
            )
        assert (
            mock_add.call_count == 3
        ), f"expected one add_basemap per facet panel (3), got {mock_add.call_count}"
        for call in mock_add.call_args_list:
            assert (
                call.kwargs.get("crs") == nc.get_variable("t2m").epsg
            ), f"each panel basemap must use the variable's CRS, got: {call.kwargs}"

    def test_facet_with_basemap_skips_hidden_panels(self):
        """`col_wrap` hidden trailing slots get no basemap (M4).

        Test scenario:
            4 time steps with `col_wrap=3` → a 2×3 grid where the last
            two slots are hidden (`set_visible(False)`). `add_basemap`
            must fire exactly 4 times (the visible panels), not 6.
        """
        nc = make_plot_3d_nc(n_times=4)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time", col_wrap=3),
                basemap=True,
            )
        assert (
            mock_add.call_count == 4
        ), f"expected one add_basemap per visible panel (4), got {mock_add.call_count}"

    def test_facet_with_basemap_string_source_propagates(self):
        """`basemap="CartoDB.Positron"` forwards the provider name to every panel."""
        nc = make_plot_3d_nc(n_times=2)
        with patch("pyramids.basemap.basemap.add_basemap") as mock_add:
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time"),
                basemap="CartoDB.Positron",
            )
        assert mock_add.call_count == 2
        for call in mock_add.call_args_list:
            assert (
                call.kwargs.get("source") == "CartoDB.Positron"
            ), f"string basemap must pass through as `source`, got: {call.kwargs}"

    def test_facet_passes_pinned_extent_to_render(self):
        """The facet path supplies the pinned subset's bbox as the render extent (M6).

        Test scenario:
            M6 fix — the facet stack is built from the pinned subset
            but rendered by `Analysis.plot`, which can't derive the
            extent from its own `self._ds` for an *injected* stack.
            `NetCDFPlot.run` now passes `_extent=pinned.bbox`, and the
            engine forwards it as `extent=` to `render_array`. Patch
            `render_array`, plot a faceted variable, and confirm the
            `extent=` it receives equals the variable subset's `bbox`.
        """
        nc = make_plot_3d_nc(n_times=3)
        var = nc.get_variable("t2m")
        with patch("pyramids.dataset.engines.analysis.render_array") as mock_render:
            mock_render.return_value = "ok"
            nc.plot(variable="t2m", facet=FacetSpec(col="time"))
        extent = mock_render.call_args.kwargs.get("extent")
        assert extent == var.bbox, (
            f"facet render extent must be the pinned subset's bbox "
            f"{var.bbox}, got {extent}"
        )
        assert (
            mock_render.call_args.kwargs.get("mode") == "facet"
        ), "this should have gone through the facet render path"

    def test_static_plot_still_uses_self_ds_bbox(self):
        """No `_extent` injected on the non-facet path → engine uses `self._ds.bbox` (M6).

        Test scenario:
            The M6 change must not perturb the self-read path: a plain
            `nc.plot(variable="t2m")` (no facet) has no `_extent`
            kwarg, so `Analysis.plot` falls back to `self._ds.bbox` —
            which equals the variable subset's `bbox`.
        """
        nc = make_plot_3d_nc(n_times=3)
        var = nc.get_variable("t2m")
        with patch("pyramids.dataset.engines.analysis.render_array") as mock_render:
            mock_render.return_value = "ok"
            nc.plot(variable="t2m")
        extent = mock_render.call_args.kwargs.get("extent")
        assert (
            extent == var.bbox
        ), f"static render extent should be self._ds.bbox {var.bbox}, got {extent}"
        assert mock_render.call_args.kwargs.get("mode") == "plot"
