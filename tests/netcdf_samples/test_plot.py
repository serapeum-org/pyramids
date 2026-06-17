"""Plotting (headless): NetCDF.plot renders a 2-D slice without opening a window.

Marked ``plot`` and run under the Agg backend; requires the optional viz dependency.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.plot
pytest.importorskip("cleopatra")


def test_plot_returns_artist(sample):
    """``plot`` on a gridded variable returns a matplotlib artist/handle without raising."""
    nc = NetCDF.read_file(sample("coards__4v__1d3-3d1.nc"))
    try:
        result = nc.plot(variable="air")
        assert result is not None
    finally:
        nc.close()
