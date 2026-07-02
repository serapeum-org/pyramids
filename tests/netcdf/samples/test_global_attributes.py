"""Global attribute access and mutation: global_attributes, set_global_attribute, delete_global_attribute."""

import shutil

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_global_attributes_is_mapping(sample_name, sample):
    """``global_attributes`` is a dict on every shape (empty is allowed)."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        attrs = nc.global_attributes
        assert isinstance(attrs, dict)
        assert all(isinstance(k, str) for k in attrs)
    finally:
        nc.close()


def test_set_and_delete_global_attribute(sample, tmp_path):
    """Setting then deleting a global attribute round-trips through a writable copy."""
    work = tmp_path / "attrs.nc"
    shutil.copy(sample("none__1v__1d1.nc"), work)
    nc = NetCDF.read_file(str(work), read_only=False)
    try:
        nc.set_global_attribute("pyramids_test_attr", "hello")
        assert nc.global_attributes.get("pyramids_test_attr") == "hello"
        nc.delete_global_attribute("pyramids_test_attr")
        assert "pyramids_test_attr" not in nc.global_attributes
    finally:
        nc.close()
