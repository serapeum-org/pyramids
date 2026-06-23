"""Hierarchical group navigation: group_names and get_group on a netCDF-4 grouped file."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.mark.samples("groups")
def test_group_names_nonempty(sample_name, sample):
    """A grouped file lists its top-level groups."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert nc.group_names, f"{sample_name}: group_names is empty"
    finally:
        nc.close()


@pytest.mark.samples("groups")
def test_get_group_returns_navigable_netcdf(sample_name, sample):
    """``get_group`` returns a NetCDF whose variables are accessible."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        group = nc.get_group(nc.group_names[0])
        assert isinstance(group, NetCDF)
        assert group.variable_names is not None
    finally:
        nc.close()


@pytest.mark.samples("groups")
def test_metadata_aggregates_group_variables(sample_name, sample, structural):
    """The aggregated metadata variable count matches the structural name (vars live in subgroups)."""
    _conv, nvars, _hist, _feats = structural(sample_name)
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        assert len(meta.groups) > 1, f"{sample_name}: expected subgroups"
        assert len(meta.variables) == nvars
    finally:
        nc.close()
