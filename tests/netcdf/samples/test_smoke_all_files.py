"""Tier-1 smoke tests: every NetCDF sample file opens, reports self-consistent metadata, loads, and closes.

These run over **all** sample files (the ``sample_name`` parameter is parametrized by ``conftest.py``).
They are the "pyramids handles every NetCDF shape" guarantee; deeper, capability-specific assertions live
in the other test modules.
"""

from collections import Counter

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_opens_and_metadata_matches_name(sample_name, sample, structural):
    """Opening the file yields metadata whose variable count and rank histogram match the structural name."""
    _convention, expected_nvars, expected_histogram, _features = structural(sample_name)
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        assert len(meta.variables) == expected_nvars, (
            f"{sample_name}: metadata reports {len(meta.variables)} variables, "
            f"structural name encodes {expected_nvars}"
        )
        histogram = dict(Counter(len(info.shape) for info in meta.variables.values()))
        assert (
            histogram == expected_histogram
        ), f"{sample_name}: rank histogram {histogram} != structural name {expected_histogram}"
        assert nc.variable_names, f"{sample_name}: variable_names is empty"
    finally:
        nc.close()


def test_every_root_variable_loads(sample_name, sample):
    """Every root variable (any rank, including 1-D coordinate axes and series) materializes.

    ``get_variable`` returns a sub-dataset for >=2-D variables and the underlying MDArray for 1-D ones; in
    both cases it must succeed without raising (regression guard for the 1-D path, issue #582).
    """
    nc = NetCDF.read_file(sample(sample_name))
    try:
        for name in nc.variable_names:
            assert (
                nc.get_variable(name) is not None
            ), f"{sample_name}: get_variable({name!r}) returned None"
    finally:
        nc.close()


def test_close_releases_handle_and_reopens(sample_name, sample):
    """After ``close()`` the file handle is released, so the same path re-opens (Windows lock check)."""
    path = sample(sample_name)
    nc = NetCDF.read_file(path)
    nc.close()
    reopened = NetCDF.read_file(path)
    try:
        assert reopened.variable_names is not None
    finally:
        reopened.close()
