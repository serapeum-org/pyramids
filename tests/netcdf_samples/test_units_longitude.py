"""Longitude conversion: convert_longitude.

(``convert_units`` is deprecated GIS-out-of-scope domain logic and is intentionally not covered here.)
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_convert_longitude_returns_container(sample):
    """``convert_longitude`` returns a NetCDF container without raising."""
    nc = NetCDF.read_file(sample("coards__5v__1d4-4d1.nc"))
    try:
        result = nc.convert_longitude()
        assert isinstance(result, NetCDF)
    finally:
        nc.close()
