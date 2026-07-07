"""LabeledDataset: reading a non-gridded, label-indexed store (AWIPS station observations)."""

import pytest

from pyramids.netcdf import LabeledDataset

pytestmark = pytest.mark.core

STATION = "none__111v__1d96-2d13-3d2__str__y-desc.nc"  # madis-sao: 111 vars over a recNum record dimension


def test_read_file_exposes_dimensions_and_variables(sample):
    """``LabeledDataset.read_file`` opens the station file and lists its dimensions and variables."""
    ld = LabeledDataset.read_file(sample(STATION))
    try:
        assert ld.dimensions, "expected dimensions"
        assert "recNum" in ld.dimensions
        assert len(ld.variables) > 0
    finally:
        ld.close()


def test_contains_and_getitem(sample):
    """Membership testing and item access return a lazy labeled array for a known variable."""
    ld = LabeledDataset.read_file(sample(STATION))
    try:
        name = ld.variables[0]
        assert name in ld
        assert "definitely_absent_variable" not in ld
        assert ld[name] is not None
    finally:
        ld.close()


def test_context_manager_closes(sample):
    """``LabeledDataset`` works as a context manager."""
    with LabeledDataset.read_file(sample(STATION)) as ld:
        assert ld.variables is not None
