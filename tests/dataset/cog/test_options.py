"""Unit tests for pyramids.dataset.cog.options."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.core

from pyramids.dataset.cog.options import (
    COG_DRIVER_OPTIONS,
    merge_options,
    to_gdal_options,
    validate_blocksize,
    validate_option_keys,
)


class TestToGdalOptions:
    def test_basic(self):
        assert to_gdal_options({"COMPRESS": "DEFLATE", "LEVEL": 9}) == [
            "COMPRESS=DEFLATE",
            "LEVEL=9",
        ]

    def test_bool_yes_no(self):
        assert to_gdal_options({"STATISTICS": True, "SPARSE_OK": False}) == [
            "STATISTICS=YES",
            "SPARSE_OK=NO",
        ]

    def test_skips_none(self):
        assert to_gdal_options({"COMPRESS": "LZW", "LEVEL": None}) == ["COMPRESS=LZW"]

    def test_none_input(self):
        assert to_gdal_options(None) == []

    def test_empty_mapping(self):
        assert to_gdal_options({}) == []

    def test_lowercase_keys_are_uppercased(self):
        assert to_gdal_options({"compress": "lzw"}) == ["COMPRESS=lzw"]

    def test_integer_value_stringified(self):
        assert to_gdal_options({"BLOCKSIZE": 512}) == ["BLOCKSIZE=512"]

    def test_float_value_stringified(self):
        assert to_gdal_options({"MAX_Z_ERROR": 0.001}) == ["MAX_Z_ERROR=0.001"]


class TestMergeOptions:
    def test_dict_extras_override_defaults(self):
        result = merge_options({"COMPRESS": "DEFLATE"}, {"COMPRESS": "ZSTD"})
        assert result == {"COMPRESS": "ZSTD"}

    def test_dict_extras_add_new_keys(self):
        result = merge_options({"COMPRESS": "DEFLATE"}, {"LEVEL": 9})
        assert result == {"COMPRESS": "DEFLATE", "LEVEL": 9}

    def test_list_extras(self):
        result = merge_options({}, ["COMPRESS=LZW", "LEVEL=6"])
        assert result == {"COMPRESS": "LZW", "LEVEL": "6"}

    def test_list_extras_override_defaults(self):
        result = merge_options({"COMPRESS": "DEFLATE"}, ["COMPRESS=LZW"])
        assert result == {"COMPRESS": "LZW"}

    def test_malformed_list_entry_raises(self):
        with pytest.raises(ValueError, match="missing '='"):
            merge_options({}, ["no-equals"])

    def test_none_extras_returns_copy_of_defaults(self):
        defaults = {"COMPRESS": "DEFLATE"}
        result = merge_options(defaults, None)
        assert result == {"COMPRESS": "DEFLATE"}
        # Ensure it's a copy, not the same object
        result["COMPRESS"] = "ZSTD"
        assert defaults["COMPRESS"] == "DEFLATE"

    def test_keys_uppercased(self):
        result = merge_options({"compress": "deflate"}, {"level": 9})
        assert result == {"COMPRESS": "deflate", "LEVEL": 9}

    def test_none_values_dropped_from_defaults(self):
        result = merge_options({"COMPRESS": "DEFLATE", "LEVEL": None}, None)
        assert result == {"COMPRESS": "DEFLATE"}

    def test_none_values_dropped_from_dict_extras(self):
        result = merge_options({}, {"COMPRESS": "DEFLATE", "LEVEL": None})
        assert result == {"COMPRESS": "DEFLATE"}


class TestValidateBlocksize:
    @pytest.mark.parametrize("value", [64, 128, 256, 512, 1024, 2048, 4096])
    def test_accepts_powers_of_two_in_range(self, value):
        validate_blocksize(value)

    @pytest.mark.parametrize("value", [500, 300, 1000])
    def test_rejects_non_power_of_two(self, value):
        with pytest.raises(ValueError, match="power of 2"):
            validate_blocksize(value)

    @pytest.mark.parametrize("value", [32, 8192, 0, -64])
    def test_rejects_out_of_range(self, value):
        with pytest.raises(ValueError, match="power of 2"):
            validate_blocksize(value)


class TestValidateOptionKeys:
    def test_accepts_known_keys(self):
        validate_option_keys({"COMPRESS": "DEFLATE", "BLOCKSIZE": 512})

    def test_accepts_lowercase_keys(self):
        validate_option_keys({"compress": "deflate"})

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="NONSENSE"):
            validate_option_keys({"NONSENSE": "x"})

    def test_rejects_unknown_with_known(self):
        with pytest.raises(ValueError, match="NONSENSE"):
            validate_option_keys({"COMPRESS": "DEFLATE", "NONSENSE": "x"})

    def test_empty_mapping_ok(self):
        validate_option_keys({})


class TestCogDriverOptions:
    def test_frozenset_contains_core_options(self):
        for key in ["COMPRESS", "BLOCKSIZE", "BIGTIFF", "OVERVIEW_RESAMPLING"]:
            assert key in COG_DRIVER_OPTIONS

    def test_is_frozenset(self):
        assert isinstance(COG_DRIVER_OPTIONS, frozenset)
