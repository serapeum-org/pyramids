"""Plotting (headless, Agg): NetCDF.plot across shapes + Selectors/ColourOpts/FacetSpec and band plots.

Marked ``plot`` and run under the Agg backend; requires the optional viz dependency (cleopatra).
"""

import pytest

from pyramids.netcdf import ColourOpts, FacetSpec, NetCDF, Selectors
from tests.netcdf.samples.conftest import RHUM

pytestmark = pytest.mark.plot
# Skip the whole module (not error) on a no-viz install: the importorskip calls must run before any
# matplotlib/cleopatra use, so a bare-wheel `pytest -m core tests/` collects without these optional
# deps instead of failing collection with ModuleNotFoundError.
pytest.importorskip("cleopatra")
plt = pytest.importorskip("matplotlib.pyplot")


# A known plottable (finite-valued) data variable per gridded sample file.
_PLOT_VAR = {
    "cf__7v__1d3-2d3-3d1__y-asc.nc": "tos",
    "coards__4v__1d3-3d1__y-desc.nc": "air",
    "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc": "ua",
    "cf__20v__1d3-3d17__y-desc.nc": "tcw",
    "cf__48v__1d17-3d21-4d10__y-asc.nc": "T",
    "coards__5v__1d4-4d1__y-desc.nc": "rhum",
    "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc": "salt",
    "none__4v__1d1-2d2-3d1__curv.nc": "Tair",
    "none__17v__1d1-2d5-3d6-4d5__stag-str.nc": "T2",
    "none__5v__1d2-2d2-3d1__curv.nc": "data",
}


@pytest.mark.samples("gridded")
def test_plot_every_gridded_file(sample_name, sample):
    """``plot`` renders a 2-D slice of a data variable for every gridded shape without raising."""
    variable = _PLOT_VAR.get(sample_name)
    if variable is None:
        pytest.skip(f"{sample_name}: no curated plottable variable")
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert nc.plot(variable=variable) is not None
    finally:
        plt.close("all")
        nc.close()


def test_plot_with_colour_options(sample):
    """``ColourOpts`` (cmap / robust limits) is accepted by plot."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        assert nc.plot(variable="rhum", colour=ColourOpts(cmap="viridis", robust=True)) is not None
    finally:
        plt.close("all")
        nc.close()


def test_plot_with_selectors_pins_a_slice(sample):
    """``Selectors`` pinning all non-spatial dims yields a single 2-D plot."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        result = nc.plot(variable="rhum", selectors=Selectors(isel={"time": 0, "level": 0}))
        assert result is not None
    finally:
        plt.close("all")
        nc.close()


def test_plot_with_facet_grid(sample):
    """``FacetSpec`` produces a multi-panel facet grid across a band dimension."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        result = nc.plot(variable="rhum", facet=FacetSpec(col="level", col_wrap=2))
        assert result is not None
    finally:
        plt.close("all")
        nc.close()


def test_plot_histogram(sample):
    """``plot_histogram`` renders a histogram of a variable view."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
    try:
        assert nc.get_variable("tos").plot_histogram() is not None
    finally:
        plt.close("all")
        nc.close()


def test_plot_vector_field(sample):
    """``plot_vector_field`` renders without raising on a variable view."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
    try:
        assert nc.get_variable("tos").plot_vector_field() is not None
    finally:
        plt.close("all")
        nc.close()
