"""Metadata serialization: to_json / from_json / to_dict round-trips over every sample shape."""

import json

import pytest

from pyramids.netcdf import NetCDF, from_json, to_dict, to_json

pytestmark = pytest.mark.core


def test_to_json_is_string_and_parses(sample_name, sample):
    """``to_json`` returns a non-empty JSON string that ``json.loads`` accepts."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        payload = to_json(nc.get_all_metadata())
        assert isinstance(payload, str) and payload
        json.loads(payload)
    finally:
        nc.close()


def test_from_json_roundtrip_preserves_structure(sample_name, sample):
    """``from_json(to_json(m))`` preserves variable names, dimension names, and convention."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        restored = from_json(to_json(meta))
        assert set(restored.variables) == set(meta.variables), f"{sample_name}: variable set changed"
        assert set(restored.dimensions) == set(meta.dimensions), f"{sample_name}: dimension set changed"
        assert restored.global_attributes.get("Conventions") == meta.global_attributes.get("Conventions")
    finally:
        nc.close()


def test_to_dict_is_json_serializable(sample_name, sample):
    """``to_dict`` returns a mapping that ``json.dumps`` can serialize without custom encoders."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        as_dict = to_dict(nc.get_all_metadata())
        assert isinstance(as_dict, dict)
        json.dumps(as_dict)
    finally:
        nc.close()
