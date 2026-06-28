"""Tests for the dataclass-grouped ``NetCDF.plot`` signature.

The public surface drops ``band=`` from the signature (kept as a
deprecated escape hatch), takes a ``variable=`` argument plus three
frozen option dataclasses that replace what used to be a flat 27-kwarg
signature:

- ``selectors=Selectors(time=..., level=..., member=..., sel=...,
  isel=...)`` — label / positional dim selectors.
- ``colour=ColourOpts(cmap=..., vmin=..., vmax=..., robust=...,
  levels=..., norm=..., center=..., extend=..., add_colorbar=...,
  cbar_kwargs=...)`` — xarray-aligned colour controls forwarded
  verbatim to cleopatra.
- ``facet=FacetSpec(col=..., row=..., col_wrap=...)`` — multi-panel
  layout description.

Curvilinear coord names that used to be ``x=`` / ``y=`` are now passed
as ``coords=("XLON", "XLAT")`` (or as a pair of numpy arrays). The
remaining top-level kwargs (``coords``, ``kind``, ``animate``,
``chunks``, ``basemap``, ``exclude_value``, ``title``, ``ax``,
``figsize``) keep their loose form.

Tests are marked ``plot`` (gated by the ``[viz]`` extra) and run under
the Agg backend that the pytest configuration forces on import.
"""

from __future__ import annotations

import logging
import types
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import ColourOpts, FacetSpec, Selectors
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
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"

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
            var.plot(selectors=Selectors(time="2024-01-15"))
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
            var.plot(selectors=Selectors(sel={"time": "2024-01-14"}))
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
            nc.plot(variable="t2m", selectors=Selectors(isel={"time": 2}))
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
            nc.plot(
                variable="temperature",
                selectors=Selectors(time=12, level=500),
            )
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
            nc.plot(
                variable="temperature",
                selectors=Selectors(sel={"time": 12}),
            )


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
            nc.plot(variable="t2m", colour=ColourOpts(robust=True))
        assert mock_plot.call_args.kwargs.get("robust") is True

    def test_center_forwarded(self):
        """`center=0.0` reaches `Analysis.plot` as `center=0.0`."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", colour=ColourOpts(center=0.0))
        assert mock_plot.call_args.kwargs.get("center") == pytest.approx(0.0)

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
                colour=ColourOpts(levels=5, extend="both", cbar_kwargs=cbar),
            )
        kw = mock_plot.call_args.kwargs
        assert kw.get("levels") == 5
        assert kw.get("extend") == "both"
        assert kw.get("cbar_kwargs") == cbar


class TestNetCDFPlotBandKwargRejected:
    """`band=` is rejected outright — a flat band index is the wrong vocabulary for NetCDF."""

    def test_band_raises_type_error_with_migration_hint(self):
        """`band=0` raises `TypeError` pointing at `Selectors(isel=...)`.

        Test scenario:
            `band=` was a back-compat shim that has been removed. It now
            joins the other rejected GeoTIFF/Sentinel kwargs in
            `_FORBIDDEN_PLOT_KWARGS`; the error message must mention
            `Selectors` so the user knows the replacement.
        """
        nc = _make_3d_nc()
        with pytest.raises(TypeError, match=r"band=") as exc_info:
            nc.plot(variable="t2m", band=0)
        msg = str(exc_info.value)
        assert (
            "Selectors" in msg
        ), f"band= rejection should point at Selectors(...), got: {msg}"

    def test_band_rejection_fires_before_render(self):
        """The `band=` gate runs before any engine call.

        Test scenario:
            Patch `Analysis.plot`; passing `band=2` must raise before
            the engine is ever invoked.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            with pytest.raises(TypeError, match=r"band="):
                nc.plot(variable="t2m", band=2)
        assert not mock_plot.called, "engine was called despite the band= rejection"


class TestNetCDFPlotCoordAxes:
    """Tests for explicit curvilinear `coords=` validation."""

    def test_invalid_coords_x_raises(self):
        """`coords=("nope", "t2m")` is rejected because "nope" is unknown."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"coords x="):
            nc.plot(variable="t2m", coords=("nope", "t2m"))

    def test_invalid_coords_y_raises(self):
        """`coords=("t2m", "nope")` is rejected because "nope" is unknown."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"coords y="):
            nc.plot(variable="t2m", coords=("t2m", "nope"))

    def test_valid_coords_render(self):
        """`coords=(<valid>, <valid>)` passes validation and renders.

        Test scenario:
            On this in-memory NetCDF the only variable name is the
            data variable itself. Passing the same name on both axes
            exercises the variable-name lookup branch of
            ``_coerce_coord_spec`` without needing a separate coord
            variable; shape validation falls through to the
            geotransform-derived extent because the data variable's
            shape does not match the slice's 2-D shape.
        """
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", coords=("t2m", "t2m"))
        assert isinstance(result, ArrayGlyph), (
            f"Expected ArrayGlyph from valid-coords render, got {type(result).__name__}"
        )


class TestNetCDFPlotDefaultRender:
    """Regression coverage for the default render path."""

    def test_3d_returns_array_glyph(self):
        """`nc.plot(variable=...)` on a 3-D variable returns an ArrayGlyph."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m")
        assert isinstance(result, ArrayGlyph)

    def test_selectors_none_equivalent_to_omitted(self):
        """`selectors=None` renders the same default slice as omitting it.

        Test scenario:
            A missing ``selectors=`` is normalised to ``Selectors()``
            inside ``NetCDF.plot``, so ``selectors=None`` must render
            byte-for-byte the same 2-D array as the bare call.
        """
        nc = _make_3d_nc()
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
        nc = _make_3d_nc(n_times=3, rows=5, cols=5)
        var = nc.get_variable("t2m")
        expected = var.read_array()[1]
        result = nc.plot(variable="t2m", selectors=Selectors(time=1))
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"
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
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"
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
        first = nc.plot(variable="t2m", selectors=Selectors(time=0))
        second = nc.plot(variable="t2m", selectors=Selectors(time=0))
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
        with pytest.raises(ValueError, match=r"is not a valid variable name"):
            nc.plot(variable="")

    def test_whitespace_variable_name_raises(self):
        """Leading/trailing whitespace on ``variable=`` is rejected.

        Test scenario:
            ``variable=" t2m "`` (with surrounding whitespace) does
            not match the canonical variable name ``"t2m"``; the call
            must raise rather than silently rendering the wrong thing.
        """
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"is not a valid variable name"):
            nc.plot(variable=" t2m ")

    def test_unknown_variable_name_raises(self):
        """``variable="missing"`` is not in ``variable_names`` and must raise."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"is not a valid variable name"):
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
            nc.plot(
                variable="temperature",
                selectors=Selectors(isel={"time": 0}),
            )
        msg = str(exc_info.value)
        assert "time" in msg, f"Resolved selectors should be reported, got: {msg}"
        assert (
            "Remaining shape" in msg
        ), f"Error must include 'Remaining shape', got: {msg}"


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

    def test_band_plus_overview_band_message_wins(self):
        """``band=`` + ``overview=`` → the ``band=`` message surfaces (it's first in the map).

        Test scenario:
            ``band`` is the first key in ``_FORBIDDEN_PLOT_KWARGS`` (it's
            the most likely mistake — it was a real parameter on
            ``main``'s ``NetCDF.plot``). When several rejected kwargs are
            passed together the gate raises on the first matching key in
            insertion order, so ``band=`` wins over ``overview=``. No
            ``DeprecationWarning`` is emitted — the back-compat shim is
            gone, ``band=`` is a hard rejection now.
        """
        nc = _make_3d_nc()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match=r"band="):
                nc.plot(variable="t2m", band=0, overview=True)
        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"band= is a hard rejection now, no deprecation hook; "
            f"got {[str(w.message) for w in deprecation_warnings]}"
        )


class TestNetCDFPlotSelectorEdges:
    """Edges for the selector pipeline (``sel``/``isel``/``time``/``level``/``member``)."""

    def test_empty_sel_dict_is_noop(self):
        """``sel={}`` adds no resolved selectors; default render proceeds."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", selectors=Selectors(sel={}))
        assert isinstance(
            result, ArrayGlyph
        ), f"Empty sel dict must be a no-op, got {type(result).__name__}"

    def test_empty_isel_dict_is_noop(self):
        """``isel={}`` adds no resolved selectors; default render proceeds."""
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", selectors=Selectors(isel={}))
        assert isinstance(
            result, ArrayGlyph
        ), f"Empty isel dict must be a no-op, got {type(result).__name__}"

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
            nc.plot(
                variable="t2m",
                selectors=Selectors(sel={"time": 0}, time=2),
            )

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
                selectors=Selectors(sel={"time": 0}, isel={"time": 2}),
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
            nc.plot(variable="t2m", selectors=Selectors(time=999))

    def test_isel_unknown_dim_name_raises(self):
        """``isel`` keyed by a non-band-dim name must raise with a helpful list."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"is not a band dim"):
            nc.plot(variable="t2m", selectors=Selectors(isel={"bogus_dim": 0}))

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
            nc.plot(variable="t2m", selectors=Selectors(level=500))
        assert "['time']" in str(
            exc_info.value
        ), f"Band dim names must be reported in the error, got: {exc_info.value}"

    def test_member_on_variable_without_ensemble_dim_raises(self):
        """``member=`` on a non-ensemble variable surfaces a clear ValueError."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"member=") as exc_info:
            nc.plot(variable="t2m", selectors=Selectors(member=0))
        assert "['time']" in str(
            exc_info.value
        ), f"Available band dims must be listed, got: {exc_info.value}"

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
            nc.plot(
                variable="temperature",
                selectors=Selectors(sel={"time": 12}),
            )
        message = str(exc_info.value)
        assert "Resolved" in message, f"Error must mention 'Resolved', got: {message}"
        assert (
            "Remaining shape" in message
        ), f"Error must mention 'Remaining shape', got: {message}"


class TestNetCDFPlotCoordAxesExtra:
    """Additional `coords=` validation coverage."""

    def test_invalid_coords_x_with_valid_y_raises_on_x_first(self):
        """`coords=("bogus", "t2m")` raises on the x axis first."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"coords x=") as exc_info:
            nc.plot(variable="t2m", coords=("bogus", "t2m"))
        assert "bogus" in str(
            exc_info.value
        ), f"Error must echo the bad name, got: {exc_info.value}"

    def test_coords_pair_both_valid_renders(self):
        """`coords=(<valid>, <valid>)` passes validation and renders.

        Test scenario:
            Pass the data variable's own name on both axes. The lookup
            via ``_coerce_coord_spec`` succeeds for both; shape
            validation then falls back through the auto-detection
            ladder, and the final render returns an ArrayGlyph.
        """
        nc = _make_3d_nc()
        result = nc.plot(variable="t2m", coords=("t2m", "t2m"))
        assert isinstance(result, ArrayGlyph), "coords=(<valid>, <valid>) should render"


class TestNetCDFPlotForwardingExtra:
    """Additional kwarg-forwarding edges to cleopatra."""

    def test_cmap_forwarded(self):
        """``cmap="viridis"`` reaches ``Analysis.plot`` verbatim."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", colour=ColourOpts(cmap="viridis"))
        assert (
            mock_plot.call_args.kwargs.get("cmap") == "viridis"
        ), f"cmap must be forwarded, got: {mock_plot.call_args.kwargs}"

    def test_vmin_vmax_forwarded(self):
        """``vmin``/``vmax`` are forwarded to the renderer."""
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", colour=ColourOpts(vmin=0.0, vmax=1.0))
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
        nc = _make_3d_nc()
        real_get_variable = type(nc).get_variable

        def _spy(self_, name, x_dim=None, y_dim=None):
            sub = real_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            sub._epsg = None
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
        nc = _make_3d_nc()
        cleo = nc.plot(variable="t2m")
        assert getattr(cleo, "cbar", None) is not None, (
            "Default add_colorbar=True must preserve cleopatra's colorbar; "
            f"got cbar={getattr(cleo, 'cbar', None)!r}"
        )

    def test_add_colorbar_false_removes_cbar(self):
        """``add_colorbar=False`` removes the colorbar from the rendered result.

        Test scenario:
            The xarray-aligned contract: a user who passes
            ``add_colorbar=False`` expects no colorbar in the output.
            Cleopatra always attaches one; the pyramids facade must
            remove it post-render. We assert the ``.cbar`` attribute
            is dropped to ``None``.
        """
        nc = _make_3d_nc()
        cleo = nc.plot(variable="t2m", colour=ColourOpts(add_colorbar=False))
        assert getattr(cleo, "cbar", None) is None, (
            "add_colorbar=False must remove the colorbar; "
            f"got cbar={getattr(cleo, 'cbar', None)!r}"
        )

    def test_add_colorbar_false_engine_call_does_not_receive_kwarg(self):
        """The facade applies removal post-render; cleopatra is not asked.

        Test scenario:
            Cleopatra's ArrayGlyph signature does not accept
            ``add_colorbar=``. The facade must not forward it to
            ``Analysis.plot`` (which would forward to cleopatra and
            raise ``"Unknown option"``). Patch the engine and confirm
            ``add_colorbar`` never appears in its kwargs.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = SimpleNamespace(cbar=None)
            nc.plot(variable="t2m", colour=ColourOpts(add_colorbar=False))
        kw = mock_plot.call_args.kwargs
        assert "add_colorbar" not in kw, (
            "add_colorbar must not be forwarded to cleopatra; "
            f"engine call kwargs were: {kw}"
        )

    def test_remove_colorbar_is_defensive_when_cbar_missing(self):
        """``_remove_colorbar`` is a silent no-op when ``.cbar`` is absent.

        Test scenario:
            Future cleopatra releases may drop or rename ``.cbar``.
            The helper must not raise — it returns silently so the
            user's render call still succeeds even if the colorbar
            cannot be removed.
        """
        from pyramids.netcdf._plot import NetCDFPlot

        NetCDFPlot._remove_colorbar(SimpleNamespace())
        NetCDFPlot._remove_colorbar(SimpleNamespace(cbar=None))


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
        assert "t2m" in str(
            exc_info.value
        ), f"Error message must list available variables, got: {exc_info.value}"

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
        assert (
            result == "ok"
        ), f"Expected the patched render to return 'ok', got: {result!r}"
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
            var.plot(selectors=Selectors(time=0))

    def test_level_on_pure_2d_variable_raises(self):
        """``level=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(selectors=Selectors(level=500))

    def test_member_on_pure_2d_variable_raises(self):
        """``member=`` on a 2-D variable raises with the band-dim hint."""
        nc = _make_2d_nc()
        var = nc.get_variable("surface")
        with pytest.raises(ValueError, match=r"no band dimension"):
            var.plot(selectors=Selectors(member=0))

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
            nc.plot(variable="signal", selectors=Selectors(time=20))
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
            1, 2``; call ``nc.plot(variable=..., selectors=Selectors(member=1))``
            and confirm the slice that reaches the renderer equals
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
            nc.plot(variable="forecast", selectors=Selectors(member=1))
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
            subset, then call
            ``var.plot(selectors=Selectors(isel={"time": 1}))``. The
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
            var.plot(selectors=Selectors(isel={"time": 1}))


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

    def _get_variable(self_, name, x_dim=None, y_dim=None):
        subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
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
        assert (
            cleo.extent is None
        ), "extent must be suppressed when curvilinear coords are present"

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

    def test_coords_override_auto_detection(self):
        """`coords=("XLONG", "XLAT")` overrides auto-detection.

        Test scenario:
            With the curvilinear conventions in place auto-detection
            would normally pick them up; the test sets ``coords=``
            explicitly and verifies the same coords still reach
            cleopatra (i.e. the explicit path uses the same arrays).
        """
        nc, x_2d, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", "XLAT"))
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
            rows=5,
            cols=6,
            x_name="lon_rho",
            y_name="lat_rho",
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
            rows=5,
            cols=6,
            x_name="nav_lon",
            y_name="nav_lat",
        )
        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (5, 6)


_FacetGrid = _cleo_array.FacetGrid


class TestNetCDFPlotFaceting:
    """PR-4 — `col=`/`row=`/`col_wrap=` build a multi-subplot grid.

    Each test exercises the cleopatra `ArrayGlyph.facet` wiring on
    NetCDF.plot. The returned object is a
    :class:`cleopatra.array_glyph.FacetGrid`; subplot count and the
    `name_dicts` index map confirm the dispatch.
    """

    def test_col_only_returns_facet_grid_with_one_row(self):
        """`col="time"` on a 3-D variable returns N=time_len subplots."""
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(time=0),
                facet=FacetSpec(col="time"),
            )

    def test_conflict_isel_and_col(self):
        """`isel={"time": 0}, col="time"` is also rejected with the same hint."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(isel={"time": 0}),
                facet=FacetSpec(col="time"),
            )

    def test_conflict_sel_and_col(self):
        """`sel={"time": 0}, col="time"` is also rejected."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                selectors=Selectors(sel={"time": 0}),
                facet=FacetSpec(col="time"),
            )

    def test_row_without_col_raises(self):
        """`row=` without `col=` is rejected with a clear error."""
        nc = _make_4d_nc()
        with pytest.raises(ValueError, match=r"requires `col=`"):
            nc.plot(variable="temperature", facet=FacetSpec(row="time"))

    def test_facet_dim_not_a_band_dim_raises(self):
        """`col="bogus"` is not a band dim of the variable; raises."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"not a band dim"):
            nc.plot(variable="t2m", facet=FacetSpec(col="bogus"))

    def test_invalid_col_wrap_raises(self):
        """`col_wrap=0` or non-int raises ValueError."""
        nc = _make_3d_nc(n_times=4)
        with pytest.raises(ValueError, match=r"positive int"):
            nc.plot(
                variable="t2m",
                facet=FacetSpec(col="time", col_wrap=0),
            )


class TestCurvilinearCoordsEdges:
    """PR-3 edge cases not covered by :class:`TestCurvilinearCoords`.

    These tests pin down the corner cases of curvilinear coord
    detection — CF attribute ordering, mixed coord-spec forms, shape
    validation, and ``kind=`` interaction with regular vs.
    curvilinear grids.
    """

    def test_explicit_coords_shape_mismatch_warns_and_falls_back(self, caplog):
        """Wrong-shaped `coords=` arrays → `logger.warning` + fall back to extent.

        Test scenario:
            M1 fix — when the caller passes explicit `coords=` whose
            arrays don't match the data slice shape, pyramids must not
            silently ignore them. It logs a WARNING on the
            ``pyramids.netcdf._plot`` logger naming the mismatched
            shapes, then falls through (no conventional coords on this
            NetCDF, so all the way to the geotransform-derived extent).
            The render still succeeds, just without curvilinear coords.
        """
        nc = _make_2d_nc()  # 5x5 `surface`, no XLONG/XLAT auto-detect names
        bad_x = np.zeros((3, 3), dtype=np.float64)
        bad_y = np.zeros((3, 3), dtype=np.float64)
        with caplog.at_level(logging.WARNING, logger="pyramids.netcdf._plot"):
            cleo = nc.plot(variable="surface", coords=(bad_x, bad_y))
        assert cleo.coords is None, (
            "mismatched explicit coords must be dropped, not used; "
            f"got {getattr(cleo, 'coords', None)!r}"
        )
        assert any(
            "don't match the data slice shape" in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records
        ), f"expected a shape-mismatch WARNING, got: {[r.getMessage() for r in caplog.records]}"

    def test_cf_coordinates_lon_then_lat(self):
        """CF `coordinates="XLONG XLAT"` (lon-first) still resolves the pair.

        Test scenario:
            CF Conventions list auxiliary coord variables space-
            separated with no enforced order. The lon-first form must
            still be parsed: the lon/lat name heuristic identifies
            XLONG as the x candidate and XLAT as the y candidate
            regardless of the order in the attribute string.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            cf_attr="XLONG XLAT",
        )
        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is not None
        ), "lon-first CF attribute must still resolve curvilinear coords"
        assert cleo.coords[0].shape == (
            5,
            6,
        ), f"x array should be (5, 6), got {cleo.coords[0].shape}"

    def test_cf_coordinates_lat_then_lon(self):
        """CF `coordinates="XLAT XLONG"` (lat-first) is also accepted.

        Test scenario:
            With the names in the opposite order the same pair must
            resolve — the heuristic looks at the names, not the list
            position. Both axes still match the data slice shape.
        """
        nc, _, _, _ = _make_curvilinear_nc(
            rows=5,
            cols=6,
            cf_attr="XLAT XLONG",
        )
        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is not None
        ), "lat-first CF attribute must still resolve curvilinear coords"
        assert cleo.coords[0].shape == (
            5,
            6,
        ), f"x array should still be (5, 6), got {cleo.coords[0].shape}"

    def test_cf_attribute_wins_over_well_known_naming(self):
        """CF `coordinates` takes priority over the WRF naming convention.

        Test scenario:
            The variable carries both a CF ``coordinates`` attribute
            that names a custom pair (``my_lon``/``my_lat``) AND the
            WRF-style ``XLONG``/``XLAT`` is available on the parent.
            CF detection runs first, so the custom pair wins. We assert
            on the actual coord arrays returned — the CF arrays differ
            from the WRF arrays because they are independent grids.
        """
        rng = np.random.default_rng(42)
        nc = NetCDF.create_from_array(
            arr=rng.random((5, 6)).astype(np.float32),
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            epsg=4326,
            variable_name="CANWAT",
        )
        wrf_x = np.linspace(-110.0, -100.0, 6, dtype=np.float32)
        wrf_y = np.linspace(35.0, 45.0, 5, dtype=np.float32)
        wrf_x_2d, wrf_y_2d = np.meshgrid(wrf_x, wrf_y)
        cf_x_2d = wrf_x_2d + 100.0
        cf_y_2d = wrf_y_2d + 50.0
        extra_vars = {
            "XLONG": wrf_x_2d,
            "XLAT": wrf_y_2d,
            "my_lon": cf_x_2d,
            "my_lat": cf_y_2d,
        }
        spliced_names = list(nc.variable_names) + list(extra_vars)
        original_read = type(nc)._read_variable
        original_get_variable = type(nc).get_variable

        def _read(self_, var, window=None):
            if var in extra_vars:
                return extra_vars[var]
            return original_read(self_, var, window)

        def _get_variable(self_, name, x_dim=None, y_dim=None):
            subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = "my_lon my_lat"
            subset._variable_attrs = attrs
            return subset

        nc._read_variable = types.MethodType(_read, nc)
        nc.get_variable = types.MethodType(_get_variable, nc)
        nc_class = type(nc)
        subcls = type(
            f"{nc_class.__name__}WithBothCoordPairs",
            (nc_class,),
            {"variable_names": property(lambda _self: spliced_names)},
        )
        nc.__class__ = subcls

        cleo = nc.plot(variable="CANWAT")
        assert cleo.coords is not None, "CF coords must resolve"
        # The CF arrays (shifted by 100/50) should reach cleopatra, not
        # the WRF arrays. Check on the x axis (longitude shift = +100).
        np.testing.assert_array_equal(
            cleo.coords[0],
            cf_x_2d,
            err_msg="CF attribute pair must win over WRF naming convention",
        )

    def test_cf_attribute_wrong_shape_falls_back_to_extent(self):
        """A CF `coordinates` attr naming a wrong-shape coord falls back.

        Test scenario:
            The CF attribute names ``my_lon``/``my_lat`` but the
            arrays returned by ``_read_variable`` have a shape that
            does not match the data slice. The detector must silently
            skip and the render must succeed using the geotransform
            extent — i.e. no crash, ``cleo.coords is None``, and the
            extent is populated from the bbox.
        """
        rng = np.random.default_rng(43)
        nc = NetCDF.create_from_array(
            arr=rng.random((5, 6)).astype(np.float32),
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            epsg=4326,
            variable_name="CANWAT",
        )
        bad_x = np.linspace(-1.0, 1.0, 99, dtype=np.float32)
        bad_y = np.linspace(0.0, 1.0, 99, dtype=np.float32)
        extra_vars = {"my_lon": bad_x, "my_lat": bad_y}
        spliced_names = list(nc.variable_names) + list(extra_vars)
        original_read = type(nc)._read_variable
        original_get_variable = type(nc).get_variable

        def _read(self_, var, window=None):
            if var in extra_vars:
                return extra_vars[var]
            return original_read(self_, var, window)

        def _get_variable(self_, name, x_dim=None, y_dim=None):
            subset = original_get_variable(self_, name, x_dim=x_dim, y_dim=y_dim)
            attrs = dict(getattr(subset, "_variable_attrs", {}) or {})
            attrs["coordinates"] = "my_lon my_lat"
            subset._variable_attrs = attrs
            return subset

        nc._read_variable = types.MethodType(_read, nc)
        nc.get_variable = types.MethodType(_get_variable, nc)
        nc_class = type(nc)
        subcls = type(
            f"{nc_class.__name__}WithBadCFShape",
            (nc_class,),
            {"variable_names": property(lambda _self: spliced_names)},
        )
        nc.__class__ = subcls

        cleo = nc.plot(variable="CANWAT")
        assert (
            cleo.coords is None
        ), "Wrong-shape CF coords must be skipped (no crash); got coords"
        assert (
            cleo.extent is not None
        ), "Renderer must fall back to extent when CF coords don't fit"

    def test_explicit_coords_missing_variable_name_raises(self):
        """`coords=("missing", "XLAT")` references a non-variable name.

        Test scenario:
            One of the two names doesn't exist in
            ``parent.variable_names``. The coord-spec coercer raises
            :class:`ValueError`, mentioning the bad name and listing
            available variables. The other valid name must not mask
            the error.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        with pytest.raises(ValueError, match=r"missing") as exc_info:
            nc.plot(variable="CANWAT", coords=("missing", "XLAT"))
        assert "Available" in str(
            exc_info.value
        ), f"Error must list available variables, got: {exc_info.value}"

    def test_explicit_coords_mixed_string_array_forms(self):
        """`coords=(name, array)` mixed-form is accepted.

        Test scenario:
            The first element is a variable name (resolved via
            ``_read_variable``), the second is a raw numpy array. The
            coercer treats each element independently, so mixed forms
            must work. The resulting curvilinear coords must reach
            cleopatra.
        """
        nc, x_2d, y_2d, _ = _make_curvilinear_nc(rows=4, cols=5)
        cleo = nc.plot(variable="CANWAT", coords=("XLONG", y_2d))
        assert cleo.coords is not None, "Mixed-form coords must resolve"
        np.testing.assert_array_equal(cleo.coords[0], x_2d)
        np.testing.assert_array_equal(cleo.coords[1], y_2d)

    def test_explicit_coords_with_nan_values_propagates_matplotlib_error(self):
        """`coords=(x_nan, y_nan)` propagates matplotlib's non-finite-coords error.

        Test scenario:
            Pyramids does not validate coord *values* — only shapes.
            All-NaN coord arrays reach cleopatra which calls
            ``ax.pcolormesh``. Matplotlib rejects non-finite coords
            with a ValueError. The pyramids layer must not mask this
            error (no try/except around the render); it must
            propagate to the caller unchanged so the user can fix
            the upstream data.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=4, cols=5)
        x_nan = np.full((4, 5), np.nan, dtype=np.float32)
        y_nan = np.full((4, 5), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match=r"non-finite"):
            nc.plot(variable="CANWAT", coords=(x_nan, y_nan))

    def test_kind_auto_no_curvilinear_uses_imshow_path(self):
        """`kind="auto"` on a regular grid leaves coords None (imshow path).

        Test scenario:
            A plain 3-D NetCDF with no curvilinear conventions has no
            coords to resolve. With ``kind="auto"`` (the default) the
            renderer should fall through to imshow — verified by
            ``cleo.coords is None`` and a populated extent.
        """
        nc = _make_3d_nc()
        cleo = nc.plot(variable="t2m", kind="auto")
        assert (
            cleo.coords is None
        ), "Regular grid + kind='auto' should keep coords None (imshow path)"
        assert cleo.extent is not None, "Imshow path must carry an extent"

    def test_kind_pcolormesh_without_explicit_coords_renders(self):
        """`kind="pcolormesh"` + coords=None — cleopatra auto-derives a grid.

        Test scenario:
            With no curvilinear coords and an explicit
            ``kind="pcolormesh"``, cleopatra falls back to an
            index-derived grid. The pyramids layer must forward the
            kind verbatim and not crash; cleo handles the rest.
        """
        nc = _make_3d_nc()
        cleo = nc.plot(variable="t2m", kind="pcolormesh")
        assert isinstance(
            cleo, ArrayGlyph
        ), "kind='pcolormesh' without coords must still produce an ArrayGlyph"

    def test_coords_1d_x_1d_y_correct_lengths(self):
        """`coords=(1D x of len cols, 1D y of len rows)` is accepted.

        Test scenario:
            cleopatra accepts 1-D coord pairs (x of length ``cols``,
            y of length ``rows``) and meshgrids them internally. The
            pyramids shape validator must accept this form: assert the
            returned cleo carries the original 1-D arrays.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        x_1d = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        y_1d = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        cleo = nc.plot(variable="CANWAT", coords=(x_1d, y_1d))
        assert cleo.coords is not None
        assert cleo.coords[0].shape == (
            6,
        ), f"x should be 1-D of length 6, got {cleo.coords[0].shape}"
        assert cleo.coords[1].shape == (
            5,
        ), f"y should be 1-D of length 5, got {cleo.coords[1].shape}"

    def test_coords_1d_swapped_lengths_falls_back_to_extent(self):
        """`coords=(1D x of len rows, 1D y of len cols)` shapes mismatch.

        Test scenario:
            Swap the two arrays — now x has the row length and y has
            the col length. ``_coord_shapes_match`` returns False, so
            the explicit-coord branch rejects them. With no other
            curvilinear conventions on the container (plain NetCDF
            from :func:`_make_3d_nc`) the render falls back to
            extent. We assert no crash and ``cleo.coords is None``.
        """
        nc = _make_3d_nc(n_times=1, rows=5, cols=6)
        x_wrong = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        y_wrong = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        cleo = nc.plot(variable="t2m", coords=(x_wrong, y_wrong))
        assert (
            cleo.coords is None
        ), "Swapped-length 1-D coords must skip and fall back to extent"
        assert cleo.extent is not None

    def test_coords_2d_x_1d_y_mixed_dims_accepted(self):
        """`coords=(2D x matching slice, 1D y of len rows)` mixed dims work.

        Test scenario:
            The shape validator accepts each axis independently — 2-D
            x matching the slice plus a 1-D y matching ``rows``
            satisfies both `x_ok` and `y_ok`. Verify the mixed-dim
            arrays reach cleopatra unchanged.
        """
        nc, _, _, _ = _make_curvilinear_nc(rows=5, cols=6)
        x_2d = np.random.default_rng(99).random((5, 6)).astype(np.float32)
        y_1d = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        cleo = nc.plot(variable="CANWAT", coords=(x_2d, y_1d))
        assert cleo.coords is not None, "Mixed (2D, 1D) coords must resolve"
        assert cleo.coords[0].shape == (5, 6)
        assert cleo.coords[1].shape == (5,)


class TestNetCDFPlotFacetingEdges:
    """PR-4 edge cases not covered by :class:`TestNetCDFPlotFaceting`.

    Coverage targets degenerate grids, ``col_wrap`` bounds, conflict
    error messages, faceting interaction with pinned dims, curvilinear
    coord forwarding to facet cells, and the returned FacetGrid
    attribute contract.
    """

    def test_col_wrap_one_produces_single_column(self):
        """`col_wrap=1` arranges N panels into N rows × 1 col."""
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc(n_times=1)
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
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"not a band dim") as exc_info:
            nc.plot(variable="t2m", facet=FacetSpec(col="bogus"))
        msg = str(exc_info.value)
        assert "bogus" in msg, f"Error must echo the bad name, got: {msg}"
        assert "time" in msg, f"Error must list 'time' as available, got: {msg}"

    def test_row_alone_error_mentions_col_requirement(self):
        """`row=` alone error message explicitly mentions `col=` requirement."""
        nc = _make_3d_nc()
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
        nc = _make_3d_nc(n_times=3)
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
        nc = _make_3d_nc(n_times=3)
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
        nc = _make_3d_nc(n_times=3)
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
        nc = _make_3d_nc(n_times=4)
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
        from pyramids.netcdf._plot import NetCDFPlot

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
        nc = _make_3d_nc(n_times=3)
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
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc(n_times=2)
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
        nc = _make_3d_nc(n_times=3)
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
        nc = _make_3d_nc(n_times=3)
        var = nc.get_variable("t2m")
        with patch("pyramids.dataset.engines.analysis.render_array") as mock_render:
            mock_render.return_value = "ok"
            nc.plot(variable="t2m")
        extent = mock_render.call_args.kwargs.get("extent")
        assert (
            extent == var.bbox
        ), f"static render extent should be self._ds.bbox {var.bbox}, got {extent}"
        assert mock_render.call_args.kwargs.get("mode") == "plot"


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
        nc = _make_3d_nc(n_times=3)
        result = nc.plot(variable="t2m", animate=True)
        assert isinstance(
            result, ArrayGlyph
        ), f"Expected ArrayGlyph, got {type(result).__name__}"
        assert result.fig is not None, "Animation must own a Figure"

    def test_animate_string_resolves_named_dim(self):
        """``animate="time"`` walks the named dim and returns ArrayGlyph."""
        nc = _make_3d_nc(n_times=4)
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
        nc = _make_3d_nc()
        with pytest.raises(KeyError, match=r"animate='bogus'") as exc_info:
            nc.plot(variable="t2m", animate="bogus")
        assert "time" in str(
            exc_info.value
        ), f"error should list the available band dims, got: {exc_info.value}"

    def test_animate_with_col_raises_mutually_exclusive(self):
        """``animate=True`` together with ``col=`` is rejected up-front."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"mutually exclusive"):
            nc.plot(variable="t2m", animate=True, facet=FacetSpec(col="time"))

    def test_animate_with_pinned_dim_raises(self):
        """``animate="time"`` together with ``time=`` selector conflicts."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(variable="t2m", animate="time", selectors=Selectors(time=0))

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
        nc = _make_3d_nc(n_times=3)
        labels = ["2024-01-01", "2024-01-02", "2024-01-03"]
        captured: dict = {}

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf.netcdf.NetCDF.get_time_variable",
            return_value=labels,
        ):
            with patch(
                "pyramids.netcdf._plot._render_array",
                side_effect=_fake_render,
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
        nc = _make_3d_nc(n_times=4)
        captured: dict = {}

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_fake_render,
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
        nc = _make_3d_nc(n_times=5)
        captured: dict = {}

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_fake_render,
        ):
            nc.plot(variable="t2m", animate=True)
        template = captured["kw"]["arr"]
        assert (
            template.ndim == 2
        ), f"Template handed to render_array must be 2-D, got {template.shape}"

    def test_animate_invalid_type_raises(self):
        """``animate=1.0`` (non-bool, non-str) is rejected with a clear error."""
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"animate="):
            nc.plot(variable="t2m", animate=1.0)


class TestNetCDFPlotLazy:
    """Tests for the PR-5 ``chunks=`` lazy static-plot path."""

    @pytest.mark.lazy
    def test_chunks_dict_routes_through_dask(self, tmp_path):
        """``chunks={"x": 5, "y": 5}`` switches the read path to dask.

        Test scenario:
            Requires the optional ``dask`` dep and a real on-disk
            NetCDF (the lazy path needs a file the dask graph can
            reopen). With ``chunks=`` set, ``Analysis.plot`` calls
            ``read_array(chunks=...)`` and ``.compute()``s only the
            requested band. The result is the same cleopatra
            ``ArrayGlyph`` shape as the eager path.
        """
        nc_mem = _make_3d_nc(n_times=2, rows=4, cols=4)
        out = tmp_path / "tiny.nc"
        nc_mem.to_file(out)
        nc = NetCDF.read_file(str(out))
        result = nc.plot(
            variable="t2m",
            selectors=Selectors(time=0),
            chunks={"cols": 2, "rows": 2},
        )
        assert isinstance(result, ArrayGlyph)
        assert result.arr.shape == (4, 4)

    def test_chunks_none_preserves_eager_behaviour(self):
        """``chunks=None`` (default) preserves the current eager path.

        Test scenario:
            ``Analysis.plot`` is patched so the test inspects the
            kwargs it receives. ``chunks=None`` (default) must not
            inject any ``_chunks`` kwarg into the call.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", selectors=Selectors(time=0))
        kw = mock_plot.call_args.kwargs
        assert "_chunks" not in kw, f"chunks=None must not inject _chunks; got {kw}"

    def test_chunks_value_forwarded_via_underscore_kwarg(self):
        """A user-supplied ``chunks=`` is forwarded to ``Analysis.plot``.

        Test scenario:
            With ``chunks={"cols": 1}``, ``NetCDF.plot`` must inject
            ``_chunks={"cols": 1}`` into the engine call so ``Analysis.plot``
            can pick it up and route the read through the dask path.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        spec = {"x": 1}
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", selectors=Selectors(time=0), chunks=spec)
        kw = mock_plot.call_args.kwargs
        assert kw.get("_chunks") == spec

    def test_lazy_hint_logged_for_large_variable(self, caplog):
        """Variables above the 100 MB threshold log an informational hint.

        Test scenario:
            Patch the subset's ``shape`` property to report a size
            > 100 MB. The static-plot path (no ``chunks=``) must emit
            an ``INFO``-level log with the ``chunks=`` hint. The plot
            call itself is short-circuited by patching ``Analysis.plot``.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        large_shape = (50, 4000, 4000)
        with patch.object(
            type(var), "shape", new_callable=lambda: property(lambda s: large_shape)
        ):
            with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
                mock_plot.return_value = "ok"
                with caplog.at_level("INFO", logger="pyramids.netcdf._plot"):
                    nc.plot(variable="t2m", selectors=Selectors(time=0))
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "chunks=" in m for m in msgs
        ), f"Expected chunks= hint in logs, got: {msgs}"

    @pytest.mark.lazy
    def test_lazy_hint_not_logged_when_chunks_supplied(self, caplog):
        """No hint is logged when the caller already passed ``chunks=``.

        Test scenario:
            Same oversize shape as the previous test but with
            ``chunks={"cols": 1}`` set. The hint is gated on
            ``chunks is None`` so the log message must be absent.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        large_shape = (50, 4000, 4000)
        with patch.object(
            type(var), "shape", new_callable=lambda: property(lambda s: large_shape)
        ):
            with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
                mock_plot.return_value = "ok"
                with caplog.at_level("INFO", logger="pyramids.netcdf._plot"):
                    nc.plot(
                        variable="t2m",
                        selectors=Selectors(time=0),
                        chunks={"cols": 1},
                    )
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "chunks=" in m for m in msgs
        ), f"Hint must not fire when chunks= is supplied; got: {msgs}"

    @pytest.mark.lazy
    def test_chunks_with_unpinned_band_dims_renders_2d_slice(self, tmp_path):
        """`chunks=` on a multi-band-dim variable renders a 2-D slice, not the whole cube (L3).

        Test scenario:
            L3 fix — `read_array(chunks=...)` returns the variable's
            *native* N-D shape (4-D `(time, pressure_level, lat, lon)`
            for ``_make_4d_nc``), whereas the eager `read_array()`
            flattens the non-spatial dims into a bands axis. The lazy
            branch must flatten to match and index `band=0`, yielding a
            2-D `(rows, cols)` array — not `.compute()` the full 4-D
            cube (which would hand cleopatra a 4-D array and break the
            render). Plotted with no selectors, the rendered slice has
            the same shape as the eager band-0 read.
        """
        nc_mem = _make_4d_nc()
        out = tmp_path / "cube4d.nc"
        nc_mem.to_file(out)
        nc = NetCDF.read_file(str(out))
        eager = nc.get_variable("temperature").read_array(band=0)
        result = nc.plot(variable="temperature", chunks={"cols": 1, "rows": 1})
        assert isinstance(result, ArrayGlyph)
        rendered = np.asarray(result.arr)
        assert rendered.ndim == 2, (
            f"lazy 4-D read must flatten + index a 2-D slice, got "
            f"{rendered.ndim}-D shape {rendered.shape}"
        )
        assert (
            rendered.shape == eager.shape
        ), f"lazy slice shape {rendered.shape} != eager band-0 shape {eager.shape}"


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

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_fake_render,
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
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match=r"already pinned"):
            nc.plot(
                variable="t2m",
                animate="time",
                selectors=Selectors(isel={"time": 0}),
            )

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

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_fake_render,
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
        nc = _make_3d_nc()
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
        nc = _make_3d_nc(n_times=4)
        captured: dict = {}

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        with patch(
            "pyramids.netcdf._plot._render_array",
            side_effect=_fake_render,
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
        nc = _make_3d_nc(n_times=4)
        captured: dict = {}

        def _fake_render(**kw):
            captured["kw"] = kw
            return "ok"

        from pyramids.netcdf.netcdf import NetCDF

        with patch.object(NetCDF, "sel", autospec=True) as sel_mock:
            with patch(
                "pyramids.netcdf._plot._render_array",
                side_effect=_fake_render,
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


class TestNetCDFPlotLazyEdges:
    """Extra ``chunks=`` coverage beyond the original 5 PR-5 cases."""

    def test_chunks_string_value_forwarded(self):
        """``chunks="auto"`` is forwarded into ``_chunks`` unchanged.

        Test scenario:
            String chunk specs (``"auto"``) are accepted alongside dicts;
            the NetCDF.plot façade only checks ``chunks is not None``
            before injecting the engine's ``_chunks`` kwarg. The string
            must be preserved verbatim — the engine decides how to
            interpret it on the dask side.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            mock_plot.return_value = "ok"
            nc.plot(variable="t2m", selectors=Selectors(time=0), chunks="auto")
        kw = mock_plot.call_args.kwargs
        assert (
            kw.get("_chunks") == "auto"
        ), f"String chunks value must be forwarded verbatim; got {kw}"

    def test_lazy_hint_does_not_fire_for_small_variable(self, caplog):
        """Variables under 100 MB never trigger the hint.

        Test scenario:
            The default ``_make_3d_nc`` fixture builds a tiny 4x5x5
            float32 variable (~400 bytes). The hint is gated on a
            100 MB threshold, so no log record must mention ``chunks=``.
        """
        nc = _make_3d_nc()
        with patch(
            "pyramids.netcdf._plot._render_array",
            return_value="ok",
        ):
            with caplog.at_level("INFO", logger="pyramids.netcdf._plot"):
                nc.plot(variable="t2m", selectors=Selectors(time=0))
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "chunks=" in m for m in msgs
        ), f"Small variable must not log lazy hint; got {msgs}"

    def test_lazy_hint_message_contains_chunks_keyword(self, caplog):
        """Hint message names the ``chunks=`` kwarg explicitly.

        Test scenario:
            Force a huge shape so the size > 100 MB and the hint fires.
            The message must contain the literal ``chunks=`` token so
            users can search docs and run-time output for it.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        large_shape = (60, 4000, 4000)
        with patch.object(
            type(var), "shape", new_callable=lambda: property(lambda s: large_shape)
        ):
            with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
                mock_plot.return_value = "ok"
                with caplog.at_level("INFO", logger="pyramids.netcdf._plot"):
                    nc.plot(variable="t2m", selectors=Selectors(time=0))
        hint_msgs = [
            r.getMessage() for r in caplog.records if "chunks=" in r.getMessage()
        ]
        assert hint_msgs, "Expected at least one hint log record"
        joined = " ".join(hint_msgs)
        assert (
            "chunks=" in joined
        ), f"Hint must contain literal `chunks=` token; got {joined!r}"

    def test_lazy_hint_threshold_boundary_one_byte_below(self, caplog):
        """One byte below the 100 MB threshold: hint stays silent.

        Test scenario:
            Pick a shape whose total byte size lands just under
            ``_LAZY_HINT_THRESHOLD_BYTES``. The guard uses strict ``>``
            so the boundary case must NOT log the hint — a regression
            here would fire the hint on every plot of a 99-MB variable.
        """
        from pyramids.netcdf._plot import _LAZY_HINT_THRESHOLD_BYTES

        nc = _make_3d_nc()
        var = nc.get_variable("t2m")
        itemsize = int(np.dtype(var.dtype[0]).itemsize)
        total_elems = _LAZY_HINT_THRESHOLD_BYTES // itemsize
        shape_below = (1, 1, int(total_elems))
        with patch.object(
            type(var), "shape", new_callable=lambda: property(lambda s: shape_below)
        ):
            with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
                mock_plot.return_value = "ok"
                with caplog.at_level("INFO", logger="pyramids.netcdf._plot"):
                    nc.plot(variable="t2m", selectors=Selectors(time=0))
        msgs = [r.getMessage() for r in caplog.records if "chunks=" in r.getMessage()]
        assert (
            not msgs
        ), f"Boundary at threshold (size == 100 MB) must not fire hint; got {msgs}"


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
            geo=(0.0, 1.0, 0, 3.0, 0, -1.0),
            epsg=4326,
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
            geo=(0.0, 1.0, 0, 3.0, 0, -1.0),
            epsg=3857,
            variable_name="d",
        )
        glyph = nc.get_variable("d").plot()
        assert glyph.crs == 3857, f"expected glyph.crs == 3857, got {glyph.crs!r}"
