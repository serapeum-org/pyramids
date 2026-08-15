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
    validate_profile,
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
        with pytest.raises(ValueError, match="must be a power of 2"):
            validate_blocksize(value)

    @pytest.mark.parametrize("value", [32, 8192, 0, -64])
    def test_rejects_out_of_range(self, value):
        with pytest.raises(ValueError, match=r"in \[64, 4096\]"):
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


class TestValidateProfile:
    """Per-profile dtype / band-count constraints in `validate_profile`."""

    @pytest.mark.parametrize(
        "name, dtype, bands",
        [
            ("deflate", "Float32", 4),
            ("zstd", "Byte", 1),
            ("lerc", "Float64", 10),
        ],
    )
    def test_unconstrained_profile_passes(self, name, dtype, bands):
        """An unconstrained profile accepts any dtype / band count.

        Args:
            name: An unconstrained profile name.
            dtype: Any GDAL dtype name.
            bands: Any band count.

        Test scenario:
            Profiles without a dtype/band constraint return `None` silently.
        """
        assert validate_profile(name, dtype, bands) is None, f"{name} should pass silently"

    @pytest.mark.parametrize("name, bands", [("jpeg", 1), ("jpeg", 3), ("webp", 3), ("webp", 4)])
    def test_constrained_profile_accepts_valid_source(self, name, bands):
        """JPEG/WEBP accept a Byte source within their band range.

        Args:
            name: A constrained profile name.
            bands: A band count inside the allowed range.

        Test scenario:
            JPEG (1-3) and WEBP (3-4) pass for a Byte source in range.
        """
        assert validate_profile(name, "Byte", bands) is None, f"{name}/{bands} should pass"

    @pytest.mark.parametrize("name", ["jpeg", "webp"])
    def test_constrained_profile_rejects_non_byte_dtype(self, name):
        """JPEG/WEBP reject a non-Byte source dtype.

        Args:
            name: A constrained profile name.

        Test scenario:
            A Float32 source raises `ValueError` naming the dtype constraint.
        """
        with pytest.raises(ValueError, match="requires dtype in") as exc:
            validate_profile(name, "Float32", 3)
        assert name in str(exc.value), f"message should name the profile, got: {exc.value}"

    @pytest.mark.parametrize("name, bands", [("jpeg", 4), ("webp", 1), ("webp", 5)])
    def test_constrained_profile_rejects_bad_band_count(self, name, bands):
        """JPEG/WEBP reject a Byte source outside their band range.

        Args:
            name: A constrained profile name.
            bands: A band count outside the allowed range.

        Test scenario:
            An out-of-range band count raises `ValueError` naming the band range.
        """
        with pytest.raises(ValueError, match="bands; got") as exc:
            validate_profile(name, "Byte", bands)
        assert str(bands) in str(exc.value), f"message should name the band count, got: {exc.value}"
