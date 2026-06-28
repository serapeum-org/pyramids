"""Smoke tests for the extracted :mod:`pyramids.netcdf._plot` module.

These cover the wiring between :meth:`pyramids.netcdf.NetCDF.plot` (the
public facade) and :class:`pyramids.netcdf._plot.NetCDFPlot` (the
implementation engine). Rendering itself is exercised under ``-m plot``;
here we only assert the delegation and module-boundary contract.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import pyramids.netcdf
from pyramids.netcdf import NetCDF
from pyramids.netcdf._plot import NetCDFPlot


def _make_3d_nc(n_times: int = 4, rows: int = 5, cols: int = 5):
    rng = np.random.default_rng(0)
    arr = rng.random((n_times, rows, cols)).astype(np.float32)
    return NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, float(rows), 0, -1.0),
        epsg=4326,
        variable_name="t2m",
        extra_dim_name="time",
        extra_dim_values=list(range(n_times)),
    )


def test_netcdf_plot_engine_importable():
    """The extracted engine is importable from its private module."""
    assert hasattr(NetCDFPlot, "run")


def test_netcdfplot_not_reexported_from_subpackage():
    """``NetCDFPlot`` is an implementation detail, not part of the public API."""
    assert not hasattr(pyramids.netcdf, "NetCDFPlot")
    assert "NetCDFPlot" not in getattr(pyramids.netcdf, "__all__", [])


def test_plot_facade_delegates_to_engine():
    """``NetCDF.plot`` constructs a ``NetCDFPlot`` and calls ``.run``."""
    nc = _make_3d_nc()
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
    nc = _make_3d_nc()
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
