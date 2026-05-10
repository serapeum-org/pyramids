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
