"""Tests for NetCDF global attributes API.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest

from pyramids.netcdf import GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


def _make_nc():
    """Create an in-memory NetCDF container."""
    arr = np.ones((5, 5), dtype=np.float64)
    geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
    return NetCDF.from_array(arr=arr, geo_ref=GeoReference(geo=geo), variable_name="v")


class TestGlobalAttributesProperty:
    """Tests for the global_attributes property."""

    def test_conventions_set_by_default(self):
        """A newly created NetCDF has exactly one global attribute: Conventions=CF-1.8.

        Test scenario:
            from_array always writes Conventions=CF-1.8;
            global_attributes should equal that single-key dict.
        """
        nc = _make_nc()
        attrs = nc.global_attributes
        assert attrs == {"Conventions": "CF-1.8"}, (
            f"Expected {{'Conventions': 'CF-1.8'}}, got {attrs!r}"
        )

    def test_returns_set_attributes(self):
        """After setting attributes, they should appear in the property.

        Test scenario:
            set_global_attribute → global_attributes contains the key.
        """
        nc = _make_nc()
        nc.set_global_attribute("Conventions", "CF-1.8")
        attrs = nc.global_attributes
        assert "Conventions" in attrs, f"Expected 'Conventions' in {attrs}"
        assert attrs["Conventions"] == "CF-1.8", (
            f"Expected 'CF-1.8', got {attrs['Conventions']}"
        )

    def test_returns_fresh_copy_each_time(self):
        """global_attributes should read live from GDAL, not a stale cache.

        Test scenario:
            Read attrs, set a new one, read again — new one should appear.
        """
        nc = _make_nc()
        before = nc.global_attributes
        nc.set_global_attribute("new_key", "new_val")
        after = nc.global_attributes
        assert "new_key" not in before, "Should not be in first read"
        assert "new_key" in after, "Should appear in second read"

    def test_real_file_has_attributes(self, noah_nc_path):
        """A real NetCDF file should have global attributes.

        Test scenario:
            Noah file has Conventions, GDAL, history, etc.
        """
        nc = NetCDF.read_file(noah_nc_path)
        attrs = nc.global_attributes
        assert len(attrs) > 0, "Real file should have global attrs"


class TestSetGlobalAttribute:
    """Tests for set_global_attribute method."""

    def test_string_value(self):
        """Setting a string attribute should store it correctly.

        Test scenario:
            set_global_attribute("history", "test") → readable back.
        """
        nc = _make_nc()
        nc.set_global_attribute("history", "created by test")
        assert nc.global_attributes["history"] == "created by test", (
            "String not stored correctly"
        )

    def test_float_value(self):
        """Setting a float attribute should store it correctly.

        Test scenario:
            set_global_attribute("version", 2.5) → readable back.
        """
        nc = _make_nc()
        nc.set_global_attribute("version", 2.5)
        assert nc.global_attributes["version"] == pytest.approx(2.5), (
            "Float not stored correctly"
        )

    def test_int_value(self):
        """Setting an int attribute should store it correctly.

        Test scenario:
            set_global_attribute("count", 42) → readable back.
        """
        nc = _make_nc()
        nc.set_global_attribute("count", 42)
        assert nc.global_attributes["count"] == 42, "Int not stored correctly"

    def test_overwrite_existing(self):
        """Setting an attribute that already exists should overwrite it.

        Test scenario:
            Set "key" twice with different values → second value wins.
        """
        nc = _make_nc()
        nc.set_global_attribute("key", "first")
        nc.set_global_attribute("key", "second")
        assert nc.global_attributes["key"] == "second", "Overwrite failed"

    def test_multiple_attributes(self):
        """Setting multiple attributes should all be readable.

        Test scenario:
            Set 3 different attributes → all present.
        """
        nc = _make_nc()
        nc.set_global_attribute("a", "1")
        nc.set_global_attribute("b", 2.0)
        nc.set_global_attribute("c", 3)
        attrs = nc.global_attributes
        assert "a" in attrs and "b" in attrs and "c" in attrs, (
            f"Not all attrs present: {list(attrs.keys())}"
        )

    def test_requires_mdim_container(self):
        """set_global_attribute on classic mode should raise.

        Test scenario:
            Open in classic mode → ValueError.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
            open_as_multi_dimensional=False,
        )
        with pytest.raises(ValueError, match="multidimensional"):
            nc.set_global_attribute("key", "value")


class TestDeleteGlobalAttribute:
    """Tests for delete_global_attribute method."""

    def test_deletes_existing(self):
        """Deleting an existing attribute should remove it.

        Test scenario:
            Set "key", delete "key" → not in global_attributes.
        """
        nc = _make_nc()
        nc.set_global_attribute("key", "value")
        assert "key" in nc.global_attributes
        nc.delete_global_attribute("key")
        assert "key" not in nc.global_attributes, "Attribute should be deleted"

    def test_delete_nonexistent_does_not_raise(self):
        """Deleting a non-existent attribute should not raise.

        Test scenario:
            delete_global_attribute("nope") → no error.
        """
        nc = _make_nc()
        nc.delete_global_attribute("nope")

    def test_requires_mdim_container(self):
        """delete_global_attribute on classic mode should raise.

        Test scenario:
            Open in classic mode → ValueError.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
            open_as_multi_dimensional=False,
        )
        with pytest.raises(ValueError, match="multidimensional"):
            nc.delete_global_attribute("key")


class TestGlobalAttributesDiskRoundTrip:
    """Global attributes should survive save → reload."""

    def test_attributes_preserved_on_disk(self, tmp_path):
        """Attributes set before to_file should be readable after reload.

        Test scenario:
            Set attrs → to_file → read_file → verify attrs present.
        """
        nc = _make_nc()
        nc.set_global_attribute("Conventions", "CF-1.8")
        nc.set_global_attribute("history", "test round-trip")
        out = str(tmp_path / "attrs_test.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out)
        attrs = reloaded.global_attributes
        assert "Conventions" in attrs, (
            f"Conventions not preserved: {list(attrs.keys())}"
        )
        assert attrs["Conventions"] == "CF-1.8", (
            f"Expected 'CF-1.8', got {attrs['Conventions']}"
        )


class TestNumericVectorAttribute:
    """set_global_attribute with a numeric list/tuple writes a real CF vector, not a repr."""

    def test_float_list_roundtrips_as_vector(self):
        """A float list is stored as a numeric vector, readable back as its values.

        Test scenario:
            set_global_attribute("valid_range", [0.0, 100.0]) → reads back [0.0, 100.0],
            not the string "[0.0, 100.0]".
        """
        nc = _make_nc()
        nc.set_global_attribute("valid_range", [0.0, 100.0])
        value = nc.global_attributes["valid_range"]
        assert not isinstance(value, str), "numeric list stored as a string repr"
        assert [float(v) for v in value] == [0.0, 100.0]

    def test_int_list_roundtrips_as_vector(self):
        """An int list is stored as a numeric vector (widened to float per CF).

        Test scenario:
            set_global_attribute("bounds", [1, 2, 3]) → reads back [1.0, 2.0, 3.0].
        """
        nc = _make_nc()
        nc.set_global_attribute("bounds", [1, 2, 3])
        value = nc.global_attributes["bounds"]
        assert not isinstance(value, str)
        assert [float(v) for v in value] == [1.0, 2.0, 3.0]

    def test_tuple_accepted_like_list(self):
        """A numeric tuple is stored as a vector too, not stringified.

        Test scenario:
            set_global_attribute("rng", (0.0, 1.0)) → reads back [0.0, 1.0].
        """
        nc = _make_nc()
        nc.set_global_attribute("rng", (0.0, 1.0))
        value = nc.global_attributes["rng"]
        assert not isinstance(value, str)
        assert [float(v) for v in value] == [0.0, 1.0]
