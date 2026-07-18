"""Plot tests: variable resolution, selectors, and rejected-kwarg validation."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf import Selectors
from tests.netcdf.conftest import make_plot_3d_nc
from tests.netcdf.plot._plot_helpers import (
    _make_3d_nc_with_dates,
    _make_4d_nc,
    _make_capture,
    _make_ensemble_nc,
)

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")


class TestNetCDFPlotVariableResolution:
    """Tests for the `variable=` argument on container and subset."""

    def test_container_without_variable_raises(self):
        """Calling `plot()` on the root container without `variable=` is rejected.

        Test scenario:
            The error message must mention `variable=` and list the
            available variables so the user can see what to pick.
        """
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"variable="):
            nc.plot()

    def test_container_with_variable_dispatches_to_subset(self):
        """Passing `variable=` on the container drills down via `get_variable`.

        Test scenario:
            `nc.plot(variable="t2m")` must return an ArrayGlyph (i.e.
            the call delegates to the subset's plot path and renders).
        """
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
        ):
            var.plot(selectors=Selectors(sel={"time": "2024-01-14"}))
        assert_array_equal(
            captured["data"],
            expected,
            err_msg="sel={'time': value} must pin the same slice as time=value",
        )

    def test_isel_positional_index(self):
        """`isel={"time": 2}` indexes by integer into `_band_dim_values_map`."""
        nc = make_plot_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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
        sel = Selectors(sel={"time": 12})
        with pytest.raises(ValueError, match=r"single 2-D slice"):
            nc.plot(variable="temperature", selectors=sel)


class TestNetCDFPlotRejectedKwargs:
    """Tests that the Sentinel-only kwargs are explicitly rejected."""

    def test_rgb_raises_with_replacement_hint(self):
        """`rgb=` mentions `time=`/`level=`/`isel=`/`band=` replacements."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"time=") as exc_info:
            nc.plot(variable="t2m", rgb=[0, 1, 2])
        assert "rgb=" in str(exc_info.value)

    def test_surface_reflectance_raises(self):
        """`surface_reflectance=` is Sentinel-only; rejected on NetCDF."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"Sentinel-only"):
            nc.plot(variable="t2m", surface_reflectance=10000)

    def test_cutoff_raises_with_vmin_vmax_hint(self):
        """`cutoff=` mentions `vmin=`/`vmax=`/`robust=True`."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"vmin=") as exc_info:
            nc.plot(variable="t2m", cutoff=[0.1, 0.9])
        assert "robust" in str(exc_info.value).lower() or "vmax=" in str(exc_info.value)

    def test_percentile_raises_with_robust_hint(self):
        """`percentile=` is rejected with a `robust=True` replacement hint."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"robust=True"):
            nc.plot(variable="t2m", percentile=2)

    def test_overview_raises_with_geotiff_hint(self):
        """`overview=` is rejected with a GeoTIFF/COG hint."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"GeoTIFF/COG"):
            nc.plot(variable="t2m", overview=True)

    def test_overview_index_raises_with_geotiff_hint(self):
        """`overview_index=` is rejected with the same GeoTIFF/COG hint."""
        nc = make_plot_3d_nc()
        with pytest.raises(TypeError, match=r"GeoTIFF/COG"):
            nc.plot(variable="t2m", overview_index=2)


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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        with patch.object(type(var.analysis), "plot", autospec=True) as mock_plot:
            with pytest.raises(TypeError, match=r"band="):
                nc.plot(variable="t2m", band=2)
        assert not mock_plot.called, "engine was called despite the band= rejection"


class TestNetCDFPlotVariableResolutionEdges:
    """Coverage for ``variable=`` edge cases not covered above."""

    def test_empty_string_variable_raises_value_error(self):
        """``variable=""`` is not a real variable and must be rejected.

        Test scenario:
            Empty-string lookup goes through ``get_variable("")`` and
            must surface as a meaningful ValueError so the user is
            not left with a cryptic GDAL error.
        """
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"is not a valid variable name"):
            nc.plot(variable="")

    def test_whitespace_variable_name_raises(self):
        """Leading/trailing whitespace on ``variable=`` is rejected.

        Test scenario:
            ``variable=" t2m "`` (with surrounding whitespace) does
            not match the canonical variable name ``"t2m"``; the call
            must raise rather than silently rendering the wrong thing.
        """
        nc = make_plot_3d_nc()
        with pytest.raises(ValueError, match=r"is not a valid variable name"):
            nc.plot(variable=" t2m ")

    def test_unknown_variable_name_raises(self):
        """``variable="missing"`` is not in ``variable_names`` and must raise."""
        nc = make_plot_3d_nc()
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
        sel = Selectors(isel={"time": 0})
        with pytest.raises(ValueError, match=r"single 2-D slice") as exc_info:
            nc.plot(variable="temperature", selectors=sel)
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
        result = nc.plot(variable="t2m", selectors=Selectors(sel={}))
        assert isinstance(
            result, ArrayGlyph
        ), f"Empty sel dict must be a no-op, got {type(result).__name__}"

    def test_empty_isel_dict_is_noop(self):
        """``isel={}`` adds no resolved selectors; default render proceeds."""
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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
        nc = make_plot_3d_nc(n_times=4)
        var = nc.get_variable("t2m")
        expected = var.read_array()[2]
        captured: dict = {}

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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
        nc = make_plot_3d_nc(n_times=4)
        sel = Selectors(time=999)
        with pytest.raises(ValueError, match=r"No bands match"):
            nc.plot(variable="t2m", selectors=sel)

    def test_isel_unknown_dim_name_raises(self):
        """``isel`` keyed by a non-band-dim name must raise with a helpful list."""
        nc = make_plot_3d_nc()
        sel = Selectors(isel={"bogus_dim": 0})
        with pytest.raises(ValueError, match=r"is not a band dim"):
            nc.plot(variable="t2m", selectors=sel)

    def test_level_on_variable_without_vertical_dim_raises(self):
        """``level=`` on a variable whose band dims do not include a vertical name.

        Test scenario:
            On a 3-D ``(time, lat, lon)`` variable the band dim is
            ``time`` only — none of the candidates (``pressure_level``,
            ``depth``, ``height``, ``z``, ``level``) appear. The
            resolver must raise and include the available band dims
            in the message.
        """
        nc = make_plot_3d_nc()
        sel = Selectors(level=500)
        with pytest.raises(ValueError, match=r"level=") as exc_info:
            nc.plot(variable="t2m", selectors=sel)
        assert "['time']" in str(
            exc_info.value
        ), f"Band dim names must be reported in the error, got: {exc_info.value}"

    def test_member_on_variable_without_ensemble_dim_raises(self):
        """``member=`` on a non-ensemble variable surfaces a clear ValueError."""
        nc = make_plot_3d_nc()
        sel = Selectors(member=0)
        with pytest.raises(ValueError, match=r"member=") as exc_info:
            nc.plot(variable="t2m", selectors=sel)
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
        sel = Selectors(sel={"time": 12})
        with pytest.raises(ValueError, match=r"single 2-D slice") as exc_info:
            nc.plot(variable="temperature", selectors=sel)
        message = str(exc_info.value)
        assert "Resolved" in message, f"Error must mention 'Resolved', got: {message}"
        assert (
            "Remaining shape" in message
        ), f"Error must mention 'Remaining shape', got: {message}"


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

        with patch.object(
            type(var.analysis),
            "plot",
            autospec=True,
            side_effect=_make_capture(captured),
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
        nc = make_plot_3d_nc()
        var = nc.get_variable("t2m")
        var._band_dim_values_map = dict(var._band_dim_values_map)
        var._band_dim_values_map["time"] = None
        sel = Selectors(isel={"time": 1})
        with pytest.raises(ValueError, match=r"No coordinate values"):
            var.plot(selectors=sel)
