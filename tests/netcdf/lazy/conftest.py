# Shared fixture path for the 3-D NetCDF test file. The autouse `_clear_file_cache` fixture that
# closes parked lazy-read handles lives in the parent `tests/netcdf/conftest.py`, so it covers every
# netcdf test directory, not just this one.
THREE_D_NC_FIXTURE = "tests/data/netcdf/cf__4v__1d3-3d1__proj__y-desc.nc"
