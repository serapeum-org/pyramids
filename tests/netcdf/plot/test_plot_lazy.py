"""Plot tests: lazy (dask-backed) rendering."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.netcdf import Selectors
from pyramids.netcdf._plot import _LAZY_HINT_THRESHOLD_BYTES
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_plot_3d_nc
from tests.netcdf.plot._plot_helpers import _make_4d_nc

pytestmark = pytest.mark.plot

_cleo_array = pytest.importorskip(
    "cleopatra.array_glyph", reason="cleopatra not installed"
)
ArrayGlyph = _cleo_array.ArrayGlyph
_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
Config = _cleo_config.Config
Config.set_matplotlib_backend("Agg")


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
        nc_mem = make_plot_3d_nc(n_times=2, rows=4, cols=4)
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

    @pytest.mark.lazy
    def test_selectors_chunks_renders_selected_slice_not_band_zero(self, tmp_path):
        """`plot(selectors=, chunks=)` renders the SELECTED slice, not storage band 0 (#728 / H1).

        Test scenario:
            The lazy `read_array(chunks=)` ignores the `sel()`-pinned subset and re-reads the whole
            variable, so before the fix the chunks render path drew `lazy[0]` — storage slice 0 —
            regardless of the selector. Each time slice of the fixture has distinct random values, so
            a wrong-slice render is detectable: the chunked render must equal the eager render of the
            same selector and must NOT equal the (different) slice-0 render.
        """
        nc_mem = make_plot_3d_nc(n_times=4, rows=4, cols=4)
        out = tmp_path / "slices.nc"
        nc_mem.to_file(out)
        nc = NetCDF.read_file(str(out))
        eager = np.asarray(nc.plot(variable="t2m", selectors=Selectors(time=2)).arr)
        lazy = np.asarray(
            nc.plot(
                variable="t2m", selectors=Selectors(time=2), chunks={"cols": 2, "rows": 2}
            ).arr
        )
        np.testing.assert_array_equal(
            lazy, eager, err_msg="chunked plot drew a different slice than the eager plot"
        )
        slice0 = np.asarray(nc.plot(variable="t2m", selectors=Selectors(time=0)).arr)
        assert not np.array_equal(
            lazy, slice0
        ), "chunked plot drew storage band 0, not the selected slice"

    @pytest.mark.lazy
    def test_multi_dim_selectors_chunks_renders_intersected_slice(self, tmp_path):
        """`chunks=` with TWO pinned band dims indexes the intersected flat band (#728 / H1).

        Test scenario:
            A 4-D `(time, pressure_level, lat, lon)` variable pinned on both band dims exercises the
            multi-dim intersection in `_flat_band_index`. The chunked render must equal the eager
            render of the same selectors and differ from the (0, 0) corner slice.
        """
        nc_mem = _make_4d_nc()
        out = tmp_path / "cube4d.nc"
        nc_mem.to_file(out)
        nc = NetCDF.read_file(str(out))
        sel = Selectors(time=6, sel={"pressure_level": 500})
        eager = np.asarray(nc.plot(variable="temperature", selectors=sel).arr)
        lazy = np.asarray(
            nc.plot(
                variable="temperature", selectors=sel, chunks={"cols": 1, "rows": 1}
            ).arr
        )
        np.testing.assert_array_equal(
            lazy, eager, err_msg="multi-dim chunked plot drew a different slice than eager"
        )
        corner = np.asarray(
            nc.plot(
                variable="temperature",
                selectors=Selectors(time=0, sel={"pressure_level": 1000}),
            ).arr
        )
        assert not np.array_equal(
            lazy, corner
        ), "chunked plot drew the (0, 0) corner band, not the intersected slice"

    def test_chunks_none_preserves_eager_behaviour(self):
        """``chunks=None`` (default) preserves the current eager path.

        Test scenario:
            ``Analysis.plot`` is patched so the test inspects the
            kwargs it receives. ``chunks=None`` (default) must not
            inject any ``_chunks`` kwarg into the call.
        """
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
            The default ``make_plot_3d_nc`` fixture builds a tiny 4x5x5
            float32 variable (~400 bytes). The hint is gated on a
            100 MB threshold, so no log record must mention ``chunks=``.
        """
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
        nc = make_plot_3d_nc()
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
