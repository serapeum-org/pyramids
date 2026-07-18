"""Reduction along a named dimension: reduce(dim, how=...)."""

import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf.samples.conftest import RHUM

pytestmark = pytest.mark.core


def test_reduce_mean_collapses_named_dimension(sample):
    """``reduce('time', how='mean')`` removes the time dimension, leaving the 4 levels (48 -> 4 bands)."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        assert nc.get_variable("rhum").read_array().shape[0] == 48
        reduced = nc.reduce("time", how="mean")
        assert isinstance(reduced, NetCDF)
        assert reduced.get_variable("rhum").read_array().shape[0] == 4
    finally:
        nc.close()


@pytest.mark.parametrize("how", ["mean", "sum", "max", "min"])
def test_reduce_supports_common_reducers(sample, how):
    """The common reducers run over a named dimension and return a NetCDF."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        reduced = nc.reduce("level", how=how)
        assert isinstance(reduced, NetCDF)
        assert (
            reduced.get_variable("rhum").read_array().shape[0] == 12
        )  # 12 time steps remain
    finally:
        nc.close()


def test_reduce_unknown_dimension_raises(sample):
    """Reducing over a dimension no variable has raises a clear ValueError."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        with pytest.raises(ValueError):
            nc.reduce("not_a_dimension")
    finally:
        nc.close()
