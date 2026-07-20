"""Metadata model invariants: NetCDFMetadata / VariableInfo / DimensionInfo / GroupInfo / CFInfo."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_variable_info_shape_matches_dimensions(sample_name, sample):
    """Each VariableInfo has a string dtype and a shape whose rank equals its dimension count."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        for name, info in nc.get_all_metadata().variables.items():
            assert isinstance(info.dtype, str) and info.dtype, (
                f"{sample_name}/{name}: empty dtype"
            )
            assert len(info.shape) == len(info.dimensions), (
                f"{sample_name}/{name}: shape {info.shape} vs dimensions {info.dimensions}"
            )
    finally:
        nc.close()


def test_variable_dimensions_resolve(sample_name, sample):
    """Every dimension referenced by a variable resolves in the metadata's dimension table."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        for name, info in meta.variables.items():
            for dim in info.dimensions:
                assert meta.get_dimension(dim) is not None, (
                    f"{sample_name}/{name}: dimension {dim!r} not found in metadata"
                )
    finally:
        nc.close()


def test_dimension_size_matches_variable_extent(sample_name, sample):
    """A DimensionInfo's size equals the variable extent along that axis."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        for info in meta.variables.values():
            for axis, dim in enumerate(info.dimensions):
                dim_info = meta.get_dimension(dim)
                assert dim_info.size == info.shape[axis], (
                    f"{sample_name}: {info.name} axis {axis} ({dim}) size "
                    f"{dim_info.size} != extent {info.shape[axis]}"
                )
    finally:
        nc.close()


@pytest.mark.samples("packed")
def test_packed_variables_expose_scale_or_offset(sample_name, sample):
    """A packed file exposes a scale_factor and/or add_offset on at least one variable.

    (CF packing may set only scale_factor — e.g. the air_temperature sample has scale but no offset.)
    """
    nc = NetCDF.read_file(sample(sample_name))
    try:
        variables = nc.get_all_metadata().variables.values()
        assert any(
            info.scale is not None or info.offset is not None for info in variables
        ), f"{sample_name}: expected a variable with scale and/or offset"
    finally:
        nc.close()


@pytest.mark.samples("groups")
def test_group_metadata_has_children(sample_name, sample):
    """A hierarchical file reports more than the root group, with child groups recorded."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        groups = nc.get_all_metadata().groups
        assert len(groups) > 1, (
            f"{sample_name}: expected child groups, got {list(groups)}"
        )
    finally:
        nc.close()


def test_convention_field_matches_registry(sample_name, sample, caps):
    """The file's declared ``Conventions`` agrees with the registry's expected convention."""
    expected = caps.get("convention")
    nc = NetCDF.read_file(sample(sample_name))
    try:
        declared = nc.global_attributes.get("Conventions")
        if expected == "none":
            assert declared is None, (
                f"{sample_name}: expected no convention, got {declared!r}"
            )
        elif expected == "coards":
            assert declared and "COARDS" in declared, (
                f"{sample_name}: {declared!r} not COARDS"
            )
        elif expected == "cf":
            assert declared and "CF-" in declared, f"{sample_name}: {declared!r} not CF"
    finally:
        nc.close()
