"""Direct unit tests for private helpers in `pyramids.dataset.cog.options`.

These tests complement the existing scenario coverage in
`test_options.py` (which exercises the helpers indirectly through
`to_gdal_options`/`merge_options`). The skill's operating mode calls
for direct coverage of private methods with a leading underscore, so
each helper gets its own :class:`TestClass`.
"""

from __future__ import annotations

import pytest

from pyramids.dataset.cog.options import _parse_list_extra, _stringify

pytestmark = pytest.mark.core


class TestStringify:
    """Direct tests for the `_stringify` private helper."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (True, "YES"),
            (False, "NO"),
            (1, "1"),
            (0, "0"),
            (-9999, "-9999"),
            (3.14, "3.14"),
            ("LZW", "LZW"),
            ("", ""),
            ("YES", "YES"),
        ],
    )
    def test_stringify_various_scalar_types(self, value, expected):
        """Test _stringify converts scalars to their GDAL-style string form.

        Args:
            value: Input scalar.
            expected: Expected stringified output.

        Test scenario:
            Booleans must become "YES"/"NO"; all other scalars use
            str() representation. Zero and empty string are explicit
            boundary cases.
        """
        result = _stringify(value)
        assert (
            result == expected
        ), f"_stringify({value!r}) returned {result!r}, expected {expected!r}"

    def test_stringify_bool_before_int_priority(self):
        """Test that True/False are treated as bool, not int.

        Test scenario:
            Python bool is a subclass of int, so a naive implementation
            might stringify True as "1". The helper must branch on bool
            first.
        """
        assert (
            _stringify(True) == "YES"
        ), f"True must be 'YES', not 'True' or '1', got {_stringify(True)!r}"
        assert (
            _stringify(False) == "NO"
        ), f"False must be 'NO', not 'False' or '0', got {_stringify(False)!r}"

    def test_stringify_none_uses_str(self):
        """Test _stringify on None defers to str().

        Test scenario:
            None should not crash — `str(None)` is `'None'`. (Note
            the public API drops None values in `to_gdal_options`;
            this is a direct-helper test of lower-level contract.)
        """
        assert (
            _stringify(None) == "None"
        ), f"_stringify(None) must defer to str(); got {_stringify(None)!r}"


class TestParseListExtra:
    """Direct tests for the `_parse_list_extra` private helper."""

    def test_single_entry(self):
        """Test parsing a single KEY=VALUE entry.

        Test scenario:
            One entry yields a one-key dict with uppercased key and
            string value preserved.
        """
        result = _parse_list_extra(["COMPRESS=DEFLATE"])
        assert result == {
            "COMPRESS": "DEFLATE"
        }, f"Single-entry parse produced {result!r}"

    def test_multiple_entries_preserve_order(self):
        """Test multiple entries populate the dict correctly.

        Test scenario:
            Each entry becomes one key; all keys are uppercased; all
            values are preserved as strings.
        """
        result = _parse_list_extra(["COMPRESS=LZW", "LEVEL=9", "PREDICTOR=2"])
        assert result == {
            "COMPRESS": "LZW",
            "LEVEL": "9",
            "PREDICTOR": "2",
        }, f"Multi-entry parse produced unexpected result: {result!r}"

    def test_lowercase_key_is_uppercased(self):
        """Test lowercase keys are normalized to uppercase.

        Test scenario:
            User-supplied mixed-case keys must be canonicalized so the
            gate check against `COG_DRIVER_OPTIONS` works uniformly.
        """
        result = _parse_list_extra(["compress=deflate"])
        assert "COMPRESS" in result, f"Key not uppercased in {result!r}"
        assert (
            result["COMPRESS"] == "deflate"
        ), f"Value must be preserved verbatim; got {result['COMPRESS']!r}"

    def test_value_containing_equals_sign(self):
        """Test values containing '=' are preserved intact.

        Test scenario:
            `partition('=')` splits only on the first occurrence, so
            a value like `"key=subval"` survives round-trip.
        """
        result = _parse_list_extra(["TARGET_SRS=EPSG:3857=extra"])
        assert (
            result["TARGET_SRS"] == "EPSG:3857=extra"
        ), f"Value with '=' not preserved; got {result['TARGET_SRS']!r}"

    def test_empty_list_returns_empty_dict(self):
        """Test empty input yields an empty dict.

        Test scenario:
            Boundary: no entries should produce no keys.
        """
        assert _parse_list_extra([]) == {}, "Empty list must produce empty dict"

    def test_malformed_entry_raises_value_error(self):
        """Test entry without '=' raises ValueError.

        Test scenario:
            A free-form string with no '=' cannot be parsed as KEY=VALUE
            and must surface a clear ValueError.
        """
        with pytest.raises(ValueError, match="missing '='") as exc_info:
            _parse_list_extra(["not-a-pair"])
        assert "not-a-pair" in str(
            exc_info.value
        ), f"Error message must echo the malformed entry; got: {exc_info.value}"

    def test_malformed_entry_in_middle(self):
        """Test a malformed entry among valid ones still raises.

        Test scenario:
            Parsing fails at the first malformed entry — no partial
            dict is returned.
        """
        with pytest.raises(ValueError, match="missing '='"):
            _parse_list_extra(["COMPRESS=LZW", "bad", "LEVEL=9"])

    def test_empty_value_allowed(self):
        """Test KEY= (empty value) is allowed.

        Test scenario:
            `'KEY='` partitions to `('KEY', '=', '')` — legal; the
            value is the empty string.
        """
        result = _parse_list_extra(["COMPRESS="])
        assert result == {"COMPRESS": ""}, f"Empty-value entry produced {result!r}"
