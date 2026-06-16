#!/usr/bin/env bash
# Re-download the NetCDF "zoo" reference set into this directory.
# Files are saved under structural names (see README.md "Naming convention");
# the original published filename is passed as the second argument for traceability.
# Unidata examples moved to the archive host (the old
# www.unidata.ucar.edu/software/netcdf/examples/ path now 404s).
set -euo pipefail
cd "$(dirname "$0")"

A=https://archive.unidata.ucar.edu/software/netcdf/examples
X=https://github.com/pydata/xarray-data/raw/master
G=https://raw.githubusercontent.com/UXARRAY/uxarray/main/test/meshfiles/ugrid

# dl <url> <structural-name>  (the URL basename is the original published filename)
dl() { curl -fsSL --max-time 180 "$1" -o "$2" && echo "OK $2 ($(du -h "$2" | cut -f1))"; }

# Unidata examples (CF / COARDS / AWIPS / netCDF-4 groups / staggered)
dl "$A/testrh.nc"                              none__1v__1d1.nc
dl "$A/tos_O1_2001-2002.nc"                    cf__7v__1d3-2d3-3d1.nc
dl "$A/sresa1b_ncar_ccsm3-example.nc"          cf__12v__1d4-2d5-3d2-4d1.nc
dl "$A/ECMWF_ERA-40_subset.nc"                 cf__20v__1d3-3d17.nc
dl "$A/cami_0000-09-01_64x128_L26_c030918.nc"  cf__48v__1d17-3d21-4d10.nc
dl "$A/OMI-Aura_L2-example.nc"                 cf__40v__1d28-2d9-3d3__nc4.nc
# rhum is downloaded full (365 daily steps), then trimmed to 100 steps (see trim_rhum.py).
dl "$A/rhum.2003.nc"                           coards__5v__1d4-4d1.nc
dl "$A/madis-sao.nc"                           none__111v__1d96-2d13-3d2__str.nc
dl "$A/WMI_Lear.nc"                            none__11v__1d11.nc
dl "$A/IMAGE0002.nc"                           none__5v__1d2-2d2-3d1__curv.nc
# wrfout is downloaded full, then reduced to a 17-variable structural subset (see reduce_wrf.py).
dl "$A/wrfout_v2_Lambert.nc"                   wrfout_v2_Lambert.nc
dl "$A/test_hgroups.nc"                        none__35v__1d35__groups-nc4.nc

# xarray-data tutorial set (curvilinear / staggered / COARDS)
dl "$X/rasm.nc"                                none__4v__1d1-2d2-3d1__curv.nc
dl "$X/ROMS_example.nc"                        cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc
dl "$X/air_temperature.nc"                     coards__4v__1d3-3d1.nc

# UXARRAY UGRID meshfiles (unstructured)
dl "$G/quad-hexagon/grid.nc"                   ugrid__6v__1d5-2d1.nc
dl "$G/quad-hexagon/multi_dim_data.nc"         ugrid__1v__3d1.nc
dl "$G/outCSne30/outCSne30_vortex.nc"          ugrid__1v__1d1.nc

# Reduce wrfout to its 17-variable structural subset, then drop the full 80-var download.
pixi run -e dev python reduce_wrf.py && rm -f wrfout_v2_Lambert.nc

# Trim the COARDS rhum cube from 365 daily steps to 100 (in place).
pixi run -e dev python trim_rhum.py
