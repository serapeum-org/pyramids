"""LabeledDataset: reading a non-gridded, label-indexed store (AWIPS station observations)."""

import pytest

from pyramids.netcdf import LabeledDataset

pytestmark = pytest.mark.core

STATION = "none__111v__1d96-2d13-3d2__str.nc"  # madis-sao: 111 vars over a recNum record dimension


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


def test_getitem_defaults_the_labels_it_does_not_read(sample):
    """``LabeledDataset["var"]`` still constructs after ``LabeledArray`` grew four fields.

    Args:
        sample: Fixture resolving a sample file name to its path.

    Test scenario:
        ``__getitem__`` builds the wrapper from ``values`` / ``dims`` / ``shape`` alone, so the
        fields added for the ``get_variable`` route have to be optional -- a required one would
        have broken every item access here. Expected: the array, with the four label fields at
        their documented defaults, and ``shape`` agreeing with the values it carries.
    """
    ld = LabeledDataset.read_file(sample(STATION))
    try:
        name = ld.variables[0]

        array = ld[name]

        assert array.name == "", f"__getitem__ declares no name, got {array.name!r}"
        assert array.unit == "", f"__getitem__ declares no unit, got {array.unit!r}"
        assert array.no_data_value is None, (
            f"__getitem__ declares no fill value, got {array.no_data_value!r}"
        )
        assert array.attributes == {}, f"unexpected attributes {array.attributes!r}"
        assert array.shape == array.values.shape, (
            f"shape {array.shape} must describe the values {array.values.shape}"
        )
    finally:
        ld.close()
