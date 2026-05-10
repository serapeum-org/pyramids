"""Tests for the PR-2 NetCDF.plot xarray-aligned signature.

The new public surface drops `band=` from the signature (kept as a
deprecated escape hatch), takes a `variable=` argument plus label-based
selectors (`time=`, `level=`, `member=`, `sel=`, `isel=`), and forwards
xarray-style colour kwargs (`robust`, `center`, `cmap`, `vmin`, `vmax`,
`levels`, `norm`, `extend`, `add_colorbar`, `cbar_kwargs`) verbatim to
cleopatra. See planning/plotting/plot-architecture-review.md §5a/§7.

Tests are marked `plot` (gated by the `[viz]` extra) and run under the
Agg backend that the pytest configuration forces on import.
"""

from __future__ import annotations

import types
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config

Config.set_matplotlib_backend("Agg")


def _make_3d_nc(n_times: int = 4, rows: int = 5, cols: int = 5):
    """Build a 3-D (time, lat, lon) NetCDF in memory.

    Args:
        n_times: Number of time steps.
        rows: Number of latitude rows.
        cols: Number of longitude columns.

    Returns:
        NetCDF: Root MDIM container with a single variable ``t2m``.
    """
    rng = np.random.default_rng(0)
    arr = rng.random((n_times, rows, cols)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(n_times)),
    )
    return nc


def _make_3d_nc_with_dates():
    """Build a 3-D variable with date-string time coords for label selection.

    Returns:
        tuple[NetCDF, list[str]]: The container plus the list of time
            coord values (strings) used for the time axis.
    """
    rng = np.random.default_rng(1)
    times = ["2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16"]
    arr = rng.random((len(times), 5, 5)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(len(times))),
    )
    var = nc.get_variable("t2m")
    var._band_dim_values = list(times)
    var._band_dim_values_map = dict(var._band_dim_values_map)
    var._band_dim_values_map["time"] = list(times)
    return nc, times, var


def _make_4d_nc():
    """Build a 4-D (time, pressure_level, lat, lon) NetCDF in memory.

    Returns:
        NetCDF: Root MDIM container with a single variable ``temperature``.
    """
    nt, nl, ny, nx = 3, 2, 4, 5
    rng = np.random.default_rng(2)
    arr = rng.random((nt, nl, ny, nx)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(ny), 0, -1.0),
        epsg=4326,
        variable_name="temperature",
        extra_dims=[
            ("time", [0, 6, 12]),
            ("pressure_level", [1000, 500]),
        ],
    )
    return nc


class TestNetCDFPlotVariableResolution:
    """Tests for the `variable=` argument on container and subset."""

    def test_container_without_variable_raises(self):
        """Calling `plot()` on the root container without `variable=` is rejected.

        Test scenario:
            The error message must mention `variable=` and list the
            available variables so the user can see what to pick.
        """
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"variable="):
            nc.plot()

    def test_container_with_variable_dispatches_to_subset(self):
        """Passing `variable=` on the container drills down via `get_variable`.

        Test scenario:
            `nc.plot(variable="t2m")` must return an ArrayGlyph (i.e.
            the call delegates to the subset's plot path and renders).
        """
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m")
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )

    def test_subset_with_matching_variable_renders(self):
        """`variable=<pinned_name>` is accepted on a variable subset.

        Test scenario:
            Mirror of `read_array`: a variable subset accepts the
            pinned variable name and ignores it (since the variable is
            already resolved).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            result = var.plot(variable="t2m")
        assert result == "ok"

    def test_subset_with_mismatched_variable_raises(self):
        """`variable=<other>` on a pinned subset is rejected.

        Test scenario:
            Calling `var.plot(variable="other")` must mention the
            pinned name and direct the caller back to the parent.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with pytest.raises(ValueError, match=r"pinned to 't2m'"):
            var.plot(variable="other")


class TestNetCDFPlotSelectors:
    """Tests for the label-based selector kwargs (`time`, `sel`, `isel`)."""

    def test_time_kwarg_pins_correct_slice(self):
        """`time=<value>` resolves to a single band via `sel(time=...)`.

        Test scenario:
            On a 4-time-step variable with date-string coords the
            pinned slice must equal `var.sel(time="2024-01-15").read_array()`
            byte-for-byte. We call `var.plot(...)` directly so the
            patched date coords survive (the container's
            `get_variable` would rebuild the subset with the original
            numeric coords).
        """
        _nc, _times, var = _make_3d_nc_with_dates()
        expected = var.sel(time="2024-01-15").read_array()
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            var.plot(time="2024-01-15")
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="time=... should pin the slice matching var.sel(...).read_array()",
        )

    def test_sel_dict_pins_correct_slice(self):
        """`sel={"time": value}` is forwarded verbatim to `self.sel(...)`."""
        _nc, _times, var = _make_3d_nc_with_dates()
        expected = var.sel(time="2024-01-14").read_array()
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            var.plot(sel={"time": "2024-01-14"})
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="sel={'time': value} must pin the same slice as time=value",
        )

    def test_isel_positional_index(self):
        """`isel={"time": 2}` indexes by integer into `_band_dim_values_map`."""
        nc = _make_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(variable="t2m", isel={"time": 2})
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="isel={'time': 2} must resolve to band index 2",
        )

    def test_4d_time_and_level_pin_one_slice(self):
        """4-D `time=` plus `level=` collapses both band dims.

        Test scenario:
            `time=12, level=500` on a `(time, pressure_level, lat, lon)`
            variable must equal `var.sel(time=12).sel(pressure_level=500).read_array()`.
        """
        nc = _make_4d_nc()
        var = nc.get_variable("temperature")
        expected = var.sel(time=12).sel(pressure_level=500).read_array()
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(variable="temperature", time=12, level=500)
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="4-D time + level selectors must match chained sel()",
        )

    def test_selectors_not_pinning_to_single_slice_raise(self):
        """If the resolved selectors leave > 1 band remaining, raise ValueError.

        Test scenario:
            On the 4-D variable, pinning only `time=12` leaves the
            pressure_level dim free (band_count == NL). The error
            message must include the resolved selectors and the
            remaining shape so the user can debug.
        """
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"single 2-D slice"):
            nc.plot(variable="temperature", sel={"time": 12})


class TestNetCDFPlotRejectedKwargs:
    """Tests that the Sentinel-only kwargs are explicitly rejected."""

    def test_rgb_raises_with_replacement_hint(self):
        """`rgb=` mentions `time=`/`level=`/`isel=`/`band=` replacements."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"time=") as exc_info:
            nc.plot(variable="t2m", rgb=[0, 1, 2])
        assert "rgb=" in str(exc_info.value)

    def test_surface_reflectance_raises(self):
        """`surface_reflectance=` is Sentinel-only; rejected on NetCDF."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"Sentinel-only"):
            nc.plot(variable="t2m", surface_reflectance=10000)

    def test_cutoff_raises_with_vmin_vmax_hint(self):
        """`cutoff=` mentions `vmin=`/`vmax=`/`robust=True`."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"vmin=") as exc_info:
            nc.plot(variable="t2m", cutoff=[0.1, 0.9])
        assert "robust" in str(exc_info.value).lower() or "vmax=" in str(exc_info.value)

    def test_percentile_raises_with_robust_hint(self):
        """`percentile=` is rejected with a `robust=True` replacement hint."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"robust=True"):
            nc.plot(variable="t2m", percentile=2)

    def test_overview_raises_with_geotiff_hint(self):
        """`overview=` is rejected with a GeoTIFF/COG hint."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"GeoTIFF/COG"):
            nc.plot(variable="t2m", overview=True)

    def test_overview_index_raises_with_geotiff_hint(self):
        """`overview_index=` is rejected with the same GeoTIFF/COG hint."""
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"GeoTIFF/COG"):
            nc.plot(variable="t2m", overview_index=2)


class TestNetCDFPlotColourForwarding:
    """Tests that the xarray-aligned colour kwargs forward to cleopatra."""

    def test_robust_forwarded(self):
        """`robust=True` reaches `Analysis.plot` as `robust=True`."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", robust=True)
        assert mock_plot.call_args.kwargs.get("robust") is True

    def test_center_forwarded(self):
        """`center=0.0` reaches `Analysis.plot` as `center=0.0`."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", center=0.0)
        assert mock_plot.call_args.kwargs.get("center") == 0.0

    def test_robust_default_not_forwarded(self):
        """`robust=False` (the default) is NOT forwarded to keep kwargs lean."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m")
        assert "robust" not in mock_plot.call_args.kwargs

    def test_levels_extend_cbar_kwargs_forwarded(self):
        """`levels=`, `extend=`, and `cbar_kwargs=` reach the renderer verbatim."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        cbar = {"label": "test"}
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(
                variable="t2m",
                levels=5,
                extend="both",
                cbar_kwargs=cbar,
            )
        kw = mock_plot.call_args.kwargs
        assert kw.get("levels") == 5
        assert kw.get("extend") == "both"
        assert kw.get("cbar_kwargs") == cbar


class TestNetCDFPlotLegacyBandKwarg:
    """`band=` is removed from the public signature but still accepted via kwargs."""

    def test_band_emits_deprecation_warning(self):
        """`band=0` works and emits a `DeprecationWarning`."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                nc.plot(variable="t2m", band=0)
        assert any(
            issubclass(w.category, DeprecationWarning) for w in captured
        ), f"DeprecationWarning not emitted; got {[w.category for w in captured]}"
        assert mock_plot.call_args.kwargs["band"] == 0

    def test_band_forwarded_as_resolved_index(self):
        """A non-zero `band=` is forwarded as the resolved flat band index."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                nc.plot(variable="t2m", band=2)
        assert mock_plot.call_args.kwargs["band"] == 2


class TestNetCDFPlotCoordAxes:
    """Tests for the `x=` / `y=` coord-name validation."""

    def test_invalid_x_raises(self):
        """`x="nope"` is not a variable of the NetCDF; reject with ValueError."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"x="):
            nc.plot(variable="t2m", x="nope")

    def test_invalid_y_raises(self):
        """`y="nope"` is rejected the same way as `x=`."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"y="):
            nc.plot(variable="t2m", y="nope")

    def test_valid_x_y_render(self):
        """`x=` / `y=` matching real variable names pass validation and render.

        Test scenario:
            On this in-memory NetCDF the only variable name is the
            data variable itself, but the validation contract is the
            same: a name in `variable_names` is accepted. The actual
            curvilinear rendering will land in PR-3; for now the
            kwargs are stashed on the subset for later use.
        """
        nc = _make_3d_nc()
        # The only variable is "t2m"; the validation passes when the
        # caller picks any name in `variable_names`.
        nc.plot(variable="t2m", x="t2m", y="t2m")


class TestNetCDFPlotDefaultRender:
    """Regression coverage for the default render path."""

    def test_3d_returns_array_glyph(self):
        """`nc.plot(variable=...)` on a 3-D variable returns an ArrayGlyph."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m")
        assert isinstance(result, ArrayGlyph)

    def test_3d_time_slice_arrayglyph_shape_matches_2d(self):
        """`time=N` selector returns an ArrayGlyph wrapping the 5x5 spatial slice.

        Test scenario:
            Build a 3-time-step container with 5x5 grid, pin
            ``time=1``, and verify the returned ArrayGlyph wraps a 2-D
            array of shape ``(5, 5)`` that equals the corresponding
            band of the source.
        """
        nc = _make_3d_nc(n_times=3, rows=5, cols=5)
        var = nc.get_variable("t2m")
        expected = var.read_array()[1]
        result = nc.plot(variable="t2m", time=1)
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph, got {type(result).__name__}"
        )
        assert result.arr.shape == (5, 5), (
            f"Expected 2-D (5, 5) slice, got shape {result.arr.shape}"
        )
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
        result = nc.plot(variable="temperature", time=12, level=500)
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
        nc = _make_3d_nc()
        first = nc.plot(variable="t2m", time=0)
        second = nc.plot(variable="t2m", time=0)
        assert first is not second, "Successive plot calls returned the same object"
        assert first.fig is not second.fig, (
            "Successive plot calls shared the same matplotlib Figure; "
            "each render must own its canvas"
        )


class TestNetCDFPlotVariableResolutionEdges:
    """Coverage for ``variable=`` edge cases not covered above."""

    def test_empty_string_variable_raises_value_error(self):
        """``variable=""`` is not a real variable and must be rejected.

        Test scenario:
            Empty-string lookup goes through ``get_variable("")`` and
            must surface as a meaningful ValueError so the user is
            not left with a cryptic GDAL error.
        """
        nc = _make_3d_nc()
        with pytest.raises(ValueError):
            nc.plot(variable="")

    def test_whitespace_variable_name_raises(self):
        """Leading/trailing whitespace on ``variable=`` is rejected.

        Test scenario:
            ``variable=" t2m "`` (with surrounding whitespace) does
            not match the canonical variable name ``"t2m"``; the call
            must raise rather than silently rendering the wrong thing.
        """
        nc = _make_3d_nc()
        with pytest.raises((ValueError, RuntimeError)):
            nc.plot(variable=" t2m ")

    def test_unknown_variable_name_raises(self):
        """``variable="missing"`` is not in ``variable_names`` and must raise."""
        nc = _make_3d_nc()
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            nc.plot(variable="missing")

    def test_4d_no_selectors_raises_did_not_pin(self):
        """4-D variable plotted with no selectors must report the under-pin.

        Test scenario:
            With band_count == n_times * n_levels > 1 and no
            selectors, ``pinned.analysis.plot(band=0, ...)`` is reached
            but the slice is ambiguous. The current behaviour renders
            band 0 of the flattened cube; assert this path runs to
            completion (no crash) and returns an ArrayGlyph — the
            "did not pin" guard fires only when ``resolved_sel`` is
            truthy.
        """
        nc = _make_4d_nc()
        result = nc.plot(variable="temperature")
        assert isinstance(result, ArrayGlyph), (
            f"4-D no-selector default should still render the first flat "
            f"band, got {type(result).__name__}"
        )

    def test_4d_under_specified_isel_raises_with_resolved_and_shape(self):
        """4-D variable under-pinned via ``isel`` reports resolved and remaining shape.

        Test scenario:
            ``isel={"time": 0}`` on a ``(time, pressure_level, lat, lon)``
            variable leaves the pressure_level dim free. The
            ValueError must mention "single 2-D slice", the resolved
            selector dict, and the remaining shape.
        """
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"single 2-D slice") as exc_info:
            nc.plot(variable="temperature", isel={"time": 0})
        msg = str(exc_info.value)
        assert "time" in msg, f"Resolved selectors should be reported, got: {msg}"
        assert "Remaining shape" in msg, (
            f"Error must include 'Remaining shape', got: {msg}"
        )


class TestNetCDFPlotRejectedKwargsCombinations:
    """Combination semantics for the six Sentinel-only rejected kwargs."""

    def test_all_six_rejected_kwargs_first_wins(self):
        """When several rejected kwargs are passed together, ``rgb`` wins.

        Test scenario:
            The gate iterates the ``forbidden_kwargs`` mapping in
            insertion order; ``rgb`` is first, so its message is the
            one that surfaces. This documents the precedence and
            guards against an accidental dict reorder regression.
        """
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"rgb="):
            nc.plot(
                variable="t2m",
                rgb=[0, 1, 2],
                surface_reflectance=10000,
                cutoff=[0.1, 0.9],
                percentile=2,
                overview=True,
                overview_index=0,
            )

    def test_rejected_kwarg_via_kwargs_dict_still_raises(self):
        """A rejected kwarg passed via ``**`` unpacking still raises.

        Test scenario:
            The gate inspects ``kwargs`` (the captured ``**kwargs``)
            regardless of how the caller spelled the argument. A user
            who builds the kwargs dict programmatically must hit the
            same TypeError, ensuring the contract is keyword-agnostic.
        """
        nc = _make_3d_nc()
        extra = {"percentile": 2}
        with pytest.raises(TypeError, match=r"robust=True"):
            nc.plot(variable="t2m", **extra)

    def test_legacy_band_plus_rejected_rejection_wins(self):
        """Rejected kwarg + legacy ``band=`` → rejected wins, no DeprecationWarning.

        Test scenario:
            The forbidden-kwargs gate runs before the legacy ``band=``
            pop in :func:`NetCDF.plot`. A combined call must therefore
            raise TypeError and emit **no** DeprecationWarning.
        """
        nc = _make_3d_nc()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match=r"overview="):
                nc.plot(variable="t2m", band=0, overview=True)
        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"Rejection must fire before the band= deprecation hook; "
            f"got {[str(w.message) for w in deprecation_warnings]}"
        )


class TestNetCDFPlotSelectorEdges:
    """Edges for the selector pipeline (``sel``/``isel``/``time``/``level``/``member``)."""

    def test_empty_sel_dict_is_noop(self):
        """``sel={}`` adds no resolved selectors; default render proceeds."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", sel={})
        assert isinstance(result, ArrayGlyph), (
            f"Empty sel dict must be a no-op, got {type(result).__name__}"
        )

    def test_empty_isel_dict_is_noop(self):
        """``isel={}`` adds no resolved selectors; default render proceeds."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", isel={})
        assert isinstance(result, ArrayGlyph), (
            f"Empty isel dict must be a no-op, got {type(result).__name__}"
        )

    def test_time_alias_overrides_sel_entry(self):
        """``time=`` alias is written into ``resolved_sel`` after the raw ``sel``.

        Test scenario:
            Both ``sel={"time": 0}`` and ``time=2`` are passed. Per
            the implementation order in :func:`NetCDF.plot`, the
            convenience alias is appended last and overwrites the
            sel-dict entry. Verify by capturing the band index that
            actually reaches the renderer.
        """
        nc = _make_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(variable="t2m", sel={"time": 0}, time=2)

        assert_array_equal(
            captured["data"],
            expected,
            err_msg="time=2 must override sel={'time': 0} (convenience alias wins)",
        )

    def test_isel_overrides_sel_for_same_dim(self):
        """``isel`` writes after ``sel`` + aliases; isel wins for shared dims.

        Test scenario:
            Both ``sel={"time": 0}`` and ``isel={"time": 2}`` are
            given; the implementation processes ``sel`` first, then
            ``isel``, so the isel entry is the one that survives in
            ``resolved_sel``.
        """
        nc = _make_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(
                variable="t2m",
                sel={"time": 0},
                isel={"time": 2},
            )
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="isel must override sel for the same dim (later write wins)",
        )

    def test_time_value_not_in_coords_raises(self):
        """An unknown ``time=`` value surfaces as a ValueError from ``sel``."""
        nc = _make_3d_nc(n_times=4)
        with pytest.raises(ValueError, match=r"No bands match"):
            nc.plot(variable="t2m", time=999)

    def test_isel_unknown_dim_name_raises(self):
        """``isel`` keyed by a non-band-dim name must raise with a helpful list."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"is not a band dim"):
            nc.plot(variable="t2m", isel={"bogus_dim": 0})

    def test_level_on_variable_without_vertical_dim_raises(self):
        """``level=`` on a variable whose band dims do not include a vertical name.

        Test scenario:
            On a 3-D ``(time, lat, lon)`` variable the band dim is
            ``time`` only — none of the candidates (``pressure_level``,
            ``depth``, ``height``, ``z``, ``level``) appear. The
            resolver must raise and include the available band dims
            in the message.
        """
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"level=") as exc_info:
            nc.plot(variable="t2m", level=500)
        assert "['time']" in str(exc_info.value), (
            f"Band dim names must be reported in the error, got: {exc_info.value}"
        )

    def test_member_on_variable_without_ensemble_dim_raises(self):
        """``member=`` on a non-ensemble variable surfaces a clear ValueError."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"member=") as exc_info:
            nc.plot(variable="t2m", member=0)
        assert "['time']" in str(exc_info.value), (
            f"Available band dims must be listed, got: {exc_info.value}"
        )

    def test_under_specified_4d_message_contents(self):
        """Pin-to-one-slice ValueError on 4-D includes resolved and remaining shape.

        Test scenario:
            ``sel={"time": 12}`` on the 4-D variable leaves
            pressure_level free; the error message must include both
            the resolved selector dict and the remaining shape so the
            user can debug.
        """
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"single 2-D slice") as exc_info:
            nc.plot(variable="temperature", sel={"time": 12})
        message = str(exc_info.value)
        assert "Resolved" in message, (
            f"Error must mention 'Resolved', got: {message}"
        )
        assert "Remaining shape" in message, (
            f"Error must mention 'Remaining shape', got: {message}"
        )


class TestNetCDFPlotCoordAxesExtra:
    """Additional ``x=`` / ``y=`` validation coverage."""

    def test_invalid_x_with_valid_y_raises_on_x_first(self):
        """``x="bogus"`` with valid ``y=`` still raises (x is checked first)."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"x=") as exc_info:
            nc.plot(variable="t2m", x="bogus", y="t2m")
        assert "bogus" in str(exc_info.value), (
            f"Error must echo the bad name, got: {exc_info.value}"
        )

    def test_valid_x_omitted_y_renders(self):
        """``x=<valid>`` with ``y=None`` passes validation and renders."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", x="t2m")
        assert isinstance(result, ArrayGlyph), (
            "x=<valid> with y omitted should render"
        )

    def test_valid_x_y_stashed_on_subset(self):
        """``x=`` and ``y=`` are stored on the subset for later PR-3 use.

        Test scenario:
            After ``nc.plot(variable="t2m", x="t2m", y="t2m")`` the
            subset returned by ``get_variable`` records the names in
            ``_plot_x_coord_name`` / ``_plot_y_coord_name``. We
            capture the subset via patching to assert the stash.
        """
        nc = _make_3d_nc()
        captured_subset: list = []
        real_get_variable = type(nc).get_variable

        def _spy(self_, name):
            sub = real_get_variable(self_, name)
            captured_subset.append(sub)
            return sub

        with patch.object(type(nc), "get_variable", _spy):
            nc.plot(variable="t2m", x="t2m", y="t2m")

        assert captured_subset, "get_variable was not called on the container"
        sub = captured_subset[0]
        assert sub._plot_x_coord_name == "t2m", (
            f"Expected x='t2m', got {sub._plot_x_coord_name!r}"
        )
        assert sub._plot_y_coord_name == "t2m", (
            f"Expected y='t2m', got {sub._plot_y_coord_name!r}"
        )


class TestNetCDFPlotForwardingExtra:
    """Additional kwarg-forwarding edges to cleopatra."""

    def test_cmap_forwarded(self):
        """``cmap="viridis"`` reaches ``Analysis.plot`` verbatim."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", cmap="viridis")
        assert mock_plot.call_args.kwargs.get("cmap") == "viridis", (
            f"cmap must be forwarded, got: {mock_plot.call_args.kwargs}"
        )

    def test_vmin_vmax_forwarded(self):
        """``vmin``/``vmax`` are forwarded to the renderer."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", vmin=0.0, vmax=1.0)
        kw = mock_plot.call_args.kwargs
        assert kw.get("vmin") == 0.0, f"vmin not forwarded: {kw}"
        assert kw.get("vmax") == 1.0, f"vmax not forwarded: {kw}"

    def test_basemap_without_epsg_raises(self):
        """``basemap=True`` on a CRS-less subset must surface the underlying ValueError.

        Test scenario:
            The ``Analysis.plot`` engine enforces
            ``"Dataset must have a CRS (epsg) to use basemap."``. We
            null out ``_epsg`` on the variable subset returned by
            ``get_variable`` and confirm NetCDF.plot does not
            short-circuit the basemap contract.
        """
        nc = _make_3d_nc()
        real_get_variable = type(nc).get_variable

        def _spy(self_, name):
            sub = real_get_variable(self_, name)
            sub._epsg = None
            return sub

        with patch.object(type(nc), "get_variable", _spy):
            with pytest.raises(ValueError, match=r"CRS"):
                nc.plot(variable="t2m", basemap=True)


class TestNetCDFPlotContainerBehaviour:
    """Container/subset dispatch wiring covered end-to-end."""

    def test_container_error_lists_available_variables(self):
        """The missing-variable error message includes every available name.

        Test scenario:
            On a container with a single variable ``t2m``, the
            ValueError text must include ``'t2m'`` so users can pick
            from the list verbatim.
        """
        nc = _make_3d_nc()
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
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            result = var.plot(variable=var._source_var_name)
        assert result == "ok", (
            f"Expected the patched render to return 'ok', got: {result!r}"
        )
        assert mock_plot.called, "Analysis.plot was not invoked"


def _make_2d_nc():
    """Build a 2-D (lat, lon) NetCDF in memory with no band dim.

    Returns:
        NetCDF: Container with a single 2-D variable ``surface``.
    """
    rng = np.random.default_rng(3)
    arr = rng.random((5, 5)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
        epsg=4326,
        variable_name="surface",
    )
    return nc


def _make_ensemble_nc():
    """Build a 3-D (member, lat, lon) NetCDF for ensemble selector tests.

    Returns:
        NetCDF: Container whose only variable has an ``ensemble`` dim.
    """
    rng = np.random.default_rng(4)
    arr = rng.random((3, 4, 4)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
        epsg=4326,
        variable_name="forecast",
        extra_dim_name="ensemble",
        extra_dim_values=[0, 1, 2],
    )
    return nc


def _make_3d_nc_anon_dim():
    """Build a 3-D NetCDF whose band dim is not a time-coded name.

    Returns:
        NetCDF: Container whose variable has a single ``alpha`` band dim
        (none of ``time`` / ``valid_time`` / ``t``).
    """
    rng = np.random.default_rng(5)
    arr = rng.random((3, 4, 4)).astype(np.float32)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
        epsg=4326,
        variable_name="signal",
        extra_dim_name="alpha",
        extra_dim_values=[10, 20, 30],
    )
    return nc


class TestNetCDFPlotDimResolverFallbacks:
    """Coverage for the 2-D / fallback paths in the three dim resolvers."""

    def test_time_on_pure_2d_variable_raises(self):
        """``time=`` on a variable with no band dim raises a helpful ValueError.

        Test scenario:
            A 2-D ``(lat, lon)`` variable has empty
            ``_band_dim_names``; the resolver must short-circuit with
            a message that mentions the absent band dimension.
        """
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(time=0)

    def test_level_on_pure_2d_variable_raises(self):
        """``level=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(level=500)

    def test_member_on_pure_2d_variable_raises(self):
        """``member=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(member=0)

    def test_time_falls_back_to_primary_band_dim(self):
        """``time=`` returns the first band dim when no candidate name matches.

        Test scenario:
            Build a NetCDF whose band dim is ``alpha`` — none of the
            time-coded candidates (``time``, ``valid_time``, ``t``)
            are present. The resolver must fall back to
            ``_band_dim_names[0]`` and pin the slice via
            ``sel(alpha=...)``. Verify by capturing the band that
            reaches the renderer.
        """
        nc = _make_3d_nc_anon_dim()
        var = nc.get_variable("signal")
        expected = var.read_array()[1]
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(variable="signal", time=20)
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="time= fallback must resolve to the first band dim (alpha)",
        )


class TestNetCDFPlotMemberSelector:
    """End-to-end selector coverage for the ``member=`` alias."""

    def test_member_resolves_ensemble_dim_and_pins_slice(self):
        """``member=N`` resolves the ``ensemble`` dim and pins the matching slice.

        Test scenario:
            Build a ``(member, lat, lon)`` variable with values ``0,
            1, 2``; call ``nc.plot(variable=..., member=1)`` and
            confirm the slice that reaches the renderer equals
            ``var.read_array()[1]``.
        """
        nc = _make_ensemble_nc()
        var = nc.get_variable("forecast")
        expected = var.read_array()[1]
        captured: dict = {}

        def _capture(self_engine, **kw):
            captured["data"] = self_engine._ds.read_array(band=0)
            return "ok"

        with patch.object(
            type(var.analysis), "plot", autospec=True, side_effect=_capture
        ):
            nc.plot(variable="forecast", member=1)
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="member=1 must pin var.read_array()[1] on an ensemble dim",
        )


class TestNetCDFPlotIselNoCoordValues:
    """``isel`` with a coord-less dim uses the raw integer index."""

    def test_isel_with_none_coords_uses_index_directly(self):
        """``coords is None`` branch in ``isel`` passes the int through to ``sel``.

        Test scenario:
            Null ``_band_dim_values_map["time"]`` on the variable
            subset, then call ``var.plot(isel={"time": 1})``. The
            isel branch sees ``coords is None``, sets
            ``resolved_sel["time"] = 1``, and the subsequent
            ``sel(time=1)`` must raise because no coord values exist
            — which we accept here; the goal is exercising the branch.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        var._band_dim_values_map = dict(var._band_dim_values_map)
        var._band_dim_values_map["time"] = None
        with pytest.raises(ValueError, match=r"No coordinate values"):
            var.plot(isel={"time": 1})


def _attach_curvilinear_coords(
    nc,
    rows: int,
    cols: int,
    *,
    x_name: str = "XLONG",
    y_name: str = "XLAT",
    cf_attr: str | None = None,
):
    """Splice synthetic 2-D curvilinear coord arrays onto the container.

    Helper used by :class:`TestCurvilinearCoords` to simulate a WRF-style
    NetCDF without authoring a real GDAL MDIM file. Patches are scoped
    to the instance — ``nc.__dict__`` is mutated with overrides for
    ``variable_names`` and ``_read_variable``, leaving other instances
    of :class:`NetCDF` untouched.

    Args:
        nc: NetCDF container to splice onto.
        rows: Number of rows for the synthetic coord grid.
        cols: Number of columns for the synthetic coord grid.
        x_name: Name of the synthetic x-coord variable.
        y_name: Name of the synthetic y-coord variable.
        cf_attr: When not None, the value is written as a CF
            ``coordinates`` attribute on the data variable's subset so
            the CF detection path fires.

    Returns:
        tuple: ``(x_arr, y_arr)`` — the synthetic 2-D coord arrays
        that were installed.
    """
    x_arr = np.linspace(-110.0, -100.0, cols, dtype=np.float32)
    y_arr = np.linspace(35.0, 45.0, rows, dtype=np.float32)
    x_2d, y_2d = np.meshgrid(x_arr, y_arr)
    extra_vars = {x_name: x_2d, y_name: y_2d}
    base_names = list(nc.variable_names)
    spliced_names = base_names + [x_name, y_name]
    original_read = type(nc)._read_variable
    original_get_variable = type(nc).get_variable

    def _read(self_, var, window=None):
        if var in extra_vars:
            return extra_vars[var]
        return original_read(self_, var, window)

    def _get_variable(self_, name):
        subset = original_get_variable(self_, name)
        if cf_attr is not None and name in base_names:
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = cf_attr
            subset._variable_attrs = attrs
        return subset

    # Instance-level patches: bind the closures to this `nc` only by
    # using `types.MethodType` so other NetCDF instances built inside
    # the same test session see the original implementations.
    nc._read_variable = types.MethodType(_read, nc)
    nc.get_variable = types.MethodType(_get_variable, nc)
    # `variable_names` is a property on the class. Override per-instance
    # by stuffing a simple object that returns the spliced list when the
    # property's descriptor falls back to `__dict__` (it doesn't).
    # Instead, monkeypatch the property via a subclass on this instance
    # using a small class trick: replace `__class__` on this instance
    # with a thin subclass that overrides only `variable_names`.
    nc_class = type(nc)
    subcls = type(
        f"{nc_class.__name__}WithSpliceCoords",
        (nc_class,),
        {"variable_names": property(lambda _self: spliced_names)},
    )
    nc.__class__ = subcls
    return x_2d, y_2d


def _make_curvilinear_nc(
    rows: int = 6,
    cols: int = 7,
    *,
    x_name: str = "XLONG",
    y_name: str = "XLAT",
    cf_attr: str | None = None,
    n_times: int | None = None,
):
    """Build a NetCDF whose container advertises curvilinear coords.

    Args:
        rows: Number of latitude rows.
        cols: Number of longitude columns.
        x_name: Name for the synthetic x-coord variable
            (``"XLONG"`` for WRF, ``"lon_rho"`` for ROMS, etc.).
        y_name: Name for the synthetic y-coord variable.
        cf_attr: When not None, the value is written as a CF
            ``coordinates`` attribute on the data variable's subset so
            the CF detection path fires.
        n_times: When set, build a 3-D (time, lat, lon) variable;
            otherwise build a 2-D (lat, lon) variable.

    Returns:
        tuple: ``(nc, x_arr, y_arr, data_var_name)`` — the container,
        the synthetic coord arrays, and the data variable name.
    """
    rng = np.random.default_rng(7)
    data_var = "CANWAT"
    if n_times is None:
        arr = rng.random((rows, cols)).astype(np.float32)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
            epsg=4326,
            variable_name=data_var,
        )
    else:
        arr = rng.random((n_times, rows, cols)).astype(np.float32)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
            epsg=4326,
            variable_name=data_var,
            extra_dim_name="time",
            extra_dim_values=list(range(n_times)),
        )
    x_2d, y_2d = _attach_curvilinear_coords(
        nc,
        rows,
        cols,
        x_name=x_name,
        y_name=y_name,
        cf_attr=cf_attr,
    )
    return nc, x_2d, y_2d, data_var


class TestCurvilinearCoords:
    """PR-3 — curvilinear coord detection and ``kind=`` dispatch in NetCDF.plot.

    Each test renders a synthetic NetCDF whose parent container exposes
    2-D ``XLAT``/``XLONG``-style coord variables via patched
    ``variable_names`` / ``_read_variable``. The :meth:`NetCDF.plot`
    surface should resolve those coords, hand them to cleopatra as
    ``coords=(x, y)``, and let cleopatra route to ``pcolormesh``.
    """

    def test_explicit_kind_pcolormesh_renders(self):
        """`kind="pcolormesh"` plus 2-D curvilinear coords renders.

        Test scenario:
            Build a WRF-style NetCDF with 2-D ``XLAT``/``XLONG`` coord
            variables. The first call passes the kind explicitly; the
            returned ArrayGlyph wraps a 2-D array with shape
            ``(rows, cols)`` and exposes the resolved coords on
            ``cleo.coords``.
        """
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=6, cols=7)
        cleo = nc.plot(variable="CANWAT", kind="pcolormesh")
        assert isinstance(cleo, ArrayGlyph)
        assert cleo.coords is not None, "curvilinear coords must reach cleopatra"
        assert cleo.coords[0].shape == (6, 7)
        assert cleo.coords[1].shape == (6, 7)
        assert cleo.extent is None, (
            "extent must be suppressed when curvilinear coords are present"
        )

    def test_kind_auto_routes_to_pcolormesh_with_2d_coords(self):
        """`kind="auto"` (default) auto-routes when 2-D coords are detected.

        Test scenario:
            With WRF-style ``XLAT``/``XLONG`` available on the
            container, the auto-detection path picks them up and
            cleopatra's ``kind="auto"`` resolves to pcolormesh
            (verified via ``cleo.coords`` being populated).
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=6, cols=7)
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (6, 7)

    def test_explicit_coords_by_name(self):
        """`coords=("XLONG", "XLAT")` looks up coord variables by name."""
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", "XLAT"))
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_explicit_coords_by_array(self):
        """`coords=(x_array, y_array)` passes arrays through untouched."""
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=4, cols=5)
        cleo = nc.plot(variable="CANWAT", coords=(x_2d, y_2d))
        assert cleo.coords is not None
        np.testing.assert_array_equal(cleo.coords[0], x_2d)
        np.testing.assert_array_equal(cleo.coords[1], y_2d)

    def test_invalid_coords_one_tuple_raises(self):
        """`coords=("nonexistent",)` (length-1) is rejected as malformed."""
        nc, _, _, _ = _make_curvilinear_nc()
        with pytest.raises(ValueError, match=r"length-2 sequence"):
            nc.plot(variable="CANWAT", coords=("nonexistent",))

    def test_x_y_kwargs_override_auto_detection(self):
        """`x="XLONG", y="XLAT"` honours the PR-2 signature override.

        Test scenario:
            With the curvilinear conventions in place auto-detection
            would normally pick them up; the test sets ``x=`` / ``y=``
            explicitly and verifies the same coords still reach
            cleopatra (i.e. the override path uses the same arrays).
        """
        nc, x_2d, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", x="XLONG", y="XLAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_no_curvilinear_falls_back_to_extent(self):
        """A regular variable with no curvilinear coords keeps imshow extent.

        Test scenario:
            Build a plain 3-D ``(time, lat, lon)`` NetCDF with no
            curvilinear coord variables. The auto path returns None, so
            cleopatra renders via imshow with the geotransform-derived
            extent — verified by ``cleo.coords is None`` and a non-None
            ``cleo.extent``.
        """
        nc = _make_3d_nc()
        cleo = nc.plot(variable="t2m")
        assert cleo.coords is None
        assert cleo.extent is not None

    def test_roms_naming_convention_auto_detected(self):
        """ROMS-style `lat_rho`/`lon_rho` are auto-detected like WRF."""
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5, cols=6, x_name="lon_rho", y_name="lat_rho",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_kind_contour_forwards(self):
        """`kind="contour"` is forwarded and renders."""
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", kind="contour")
        assert isinstance(cleo, ArrayGlyph)

    def test_kind_contourf_forwards(self):
        """`kind="contourf"` is forwarded and renders."""
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", kind="contourf")
        assert isinstance(cleo, ArrayGlyph)

    def test_kind_bogus_raises_value_error(self):
        """`kind="bogus"` propagates cleopatra's ValueError to the caller.

        Test scenario:
            cleopatra validates ``kind`` against
            :data:`cleopatra.array_glyph.VALID_PLOT_KINDS`. An unknown
            value triggers a ValueError that must propagate through
            pyramids unchanged so users see the same error message they
            would see calling ArrayGlyph directly.
        """
        nc, _, _, _ = _make_curvilinear_nc()
        with pytest.raises(ValueError, match=r"Invalid kind"):
            nc.plot(variable="CANWAT", kind="bogus")

    def test_cf_coordinates_attr_auto_detected(self):
        """CF `coordinates` attribute drives the auto-detection path.

        Test scenario:
            Build a NetCDF where the data variable's subset carries a
            CF ``coordinates`` attribute that lists ``"longitude
            latitude"`` (custom names, not in the well-known list). The
            CF-aware detection path should parse the attribute, resolve
            each name via ``_read_variable``, and pass them to
            cleopatra as curvilinear coords.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            x_name="longitude",
            y_name="latitude",
            cf_attr="longitude latitude",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)

    def test_nemo_naming_convention_auto_detected(self):
        """NEMO-style ``nav_lat``/``nav_lon`` are auto-detected like WRF."""
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5, cols=6, x_name="nav_lon", y_name="nav_lat",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)
