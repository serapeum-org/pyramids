"""Smoke tests for the extracted :mod:`pyramids.netcdf._plot` module.

These cover the wiring between :meth:`pyramids.netcdf.NetCDF.plot` (the
public facade) and :class:`pyramids.netcdf._plot.NetCDFPlot` (the
implementation engine). Rendering itself is exercised under ``-m plot``;
here we only assert the delegation and module-boundary contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pyramids.netcdf
from pyramids.netcdf import NetCDF
from pyramids.netcdf._plot import NetCDFPlot
from tests.netcdf.conftest import make_plot_3d_nc


def test_netcdf_plot_engine_importable():
    """The extracted engine is importable from its private module."""
    assert hasattr(NetCDFPlot, "run")


def test_netcdfplot_not_reexported_from_subpackage():
    """``NetCDFPlot`` is an implementation detail, not part of the public API."""
    assert not hasattr(pyramids.netcdf, "NetCDFPlot")
    assert "NetCDFPlot" not in getattr(pyramids.netcdf, "__all__", [])


def test_plot_facade_delegates_to_engine():
    """``NetCDF.plot`` constructs a ``NetCDFPlot`` and calls ``.run``."""
    nc = make_plot_3d_nc()
    sentinel = object()
    with patch.object(
        NetCDFPlot, "run", autospec=True, return_value=sentinel
    ) as mock_run:
        out = nc.plot(variable="t2m")
    assert out is sentinel
    assert mock_run.call_count == 1
    bound_self = mock_run.call_args.args[0]
    assert isinstance(bound_self, NetCDFPlot)
    assert bound_self.nc is nc
    assert mock_run.call_args.args[1] == "t2m"


def test_engine_run_equivalent_to_facade():
    """Calling ``NetCDFPlot(nc).run(...)`` is equivalent to ``nc.plot(...)``.

    Both routes ultimately reach ``Analysis.plot`` with the same kwargs;
    here we patch that engine method and compare the captured calls.
    """
    nc = make_plot_3d_nc()
    var = nc.get_variable("t2m")
    captured: list = []

    def _record(*args, **kwargs):
        captured.append((args, kwargs))
        return "ok"

    with patch.object(type(var.analysis), "plot", side_effect=_record):
        facade_result = nc.plot(variable="t2m")
    with patch.object(type(var.analysis), "plot", side_effect=_record):
        engine_result = NetCDFPlot(nc).run(variable="t2m")

    assert facade_result == engine_result == "ok"
    assert len(captured) == 2
    assert captured[0] == captured[1]
