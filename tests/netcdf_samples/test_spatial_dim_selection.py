"""Spatial-plane selection for get_variable / plot: CF auto-detection and x_dim/y_dim override."""

import os

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

os.environ.setdefault("MPLBACKEND", "Agg")

# cf__48v T(time, lat, lev, lon): lat/lon are NOT the trailing dims and carry no CF axis attributes,
# so it is the canonical "needs an explicit override" case (lat=64, lev=6, lon=128).
NONSTD = "cf__48v__1d17-3d21-4d10.nc"
# A CF file whose lat/lon coordinate variables carry units -> auto-detection should fire.
CF = "coards__4v__1d3-3d1.nc"


def test_cf_auto_detection_resolves_lat_lon(sample):
    """On a CF file, the lat/lon dims are detected from their coordinate attributes."""
    nc = NetCDF.read_file(sample(CF))
    try:
        md = nc._raster.GetRootGroup().OpenMDArray("air")
        roles = [NetCDF._axis_role_of_dimension(d) for d in md.GetDimensions()]
        assert "X" in roles and "Y" in roles
    finally:
        nc.close()


def test_explicit_x_dim_y_dim_selects_plane(sample):
    """`x_dim`/`y_dim` pick the raster plane when the dimension order is non-standard (#T plane)."""
    nc = NetCDF.read_file(sample(NONSTD))
    try:
        var = nc.get_variable("T", x_dim="lon", y_dim="lat")
        assert var.read_array().shape[-2:] == (64, 128)  # (lat, lon)
        # Default (no override, no CF attrs) falls back to the last two dims (lev, lon).
        assert nc.get_variable("T").read_array().shape[-2:] == (6, 128)
    finally:
        nc.close()


def test_x_dim_y_dim_same_dimension_raises(sample):
    """Explicitly naming the same dimension for both axes is an error."""
    nc = NetCDF.read_file(sample(NONSTD))
    try:
        with pytest.raises(ValueError, match="different dimensions"):
            nc.get_variable("T", x_dim="lon", y_dim="lon")
    finally:
        nc.close()


@pytest.mark.plot
def test_plot_with_x_dim_y_dim(sample):
    """`plot` forwards x_dim/y_dim so a non-standard-ordered variable renders as a map."""
    pytest.importorskip("cleopatra")
    import matplotlib.pyplot as plt

    nc = NetCDF.read_file(sample(NONSTD))
    try:
        assert nc.plot(variable="T", x_dim="lon", y_dim="lat") is not None
    finally:
        plt.close("all")
        nc.close()
