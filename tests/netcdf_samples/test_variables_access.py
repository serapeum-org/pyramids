"""Variable access: ``variable_names`` / ``get_variable_names`` / ``variables`` / ``get_variable``.

Covers the read-side variable API across shapes, including the 1-D variable path (regression for issue
#582, where ``get_variable`` on a 1-D numeric variable raised ``Invalid iXDim and/or iYDim``).
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_variable_names_match_get_variable_names(sample_name, sample):
    """The ``variable_names`` property and ``get_variable_names()`` return the same root variables."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert list(nc.variable_names) == list(nc.get_variable_names())
    finally:
        nc.close()


def test_variables_dict_keys_match_names(sample_name, sample):
    """The lazy ``variables`` mapping exposes exactly the root variable names and loads each on access."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert set(nc.variables.keys()) == set(nc.variable_names)
        for name in nc.variable_names:
            assert nc.variables[name] is not None
    finally:
        nc.close()


@pytest.mark.samples("multivar")
def test_get_variable_2d_plus_has_shape(sample_name, sample):
    """For a multi-variable file, every >=2-D variable loads as a dataset exposing a 2-D+ shape."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        twod_plus = [n for n in nc.variable_names if len(meta.variables[n].shape) >= 2]
        for name in twod_plus:
            var = nc.get_variable(name)
            assert var.shape is not None and len(var.shape) >= 2, (
                f"{sample_name}: get_variable({name!r}).shape = {var.shape!r}"
            )
    finally:
        nc.close()


def test_get_variable_1d_does_not_raise(sample_name, sample):
    """``get_variable`` on a 1-D enumerated variable returns it without raising (issue #582).

    Runs over every file; files with no 1-D enumerated variable simply assert nothing.
    """
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        oned = [n for n in nc.variable_names if len(meta.variables[n].shape) == 1]
        for name in oned:
            assert nc.get_variable(name) is not None, (
                f"{sample_name}: get_variable({name!r}) (1-D) returned None"
            )
    finally:
        nc.close()


def test_get_variable_unknown_raises(sample_name, sample):
    """Requesting a non-existent variable raises a clear ``ValueError`` naming the bad variable."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        with pytest.raises(ValueError, match="not a valid variable name"):
            nc.get_variable("definitely_not_a_variable_xyz")
    finally:
        nc.close()
