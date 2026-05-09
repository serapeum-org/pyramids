import pytest

from pyramids.netcdf.dimensions import (
    DimensionsIndex,
    DimMetaData,
    _coerce_scalar,
    _format_braced_list,
    _parse_values_list,
    _smart_split_csv,
    _strip_braces,
    parse_gdal_netcdf_dimensions,
)

pytestmark = pytest.mark.core


class TestStripBraces:
    def test_with_braces(self):
        """Ensure outer braces are removed.

        Input:
            A string with surrounding braces, e.g. "{1,2,3}".

        Expected:
            The content between braces returned without spaces: "1,2,3".

        Checks:
            Correct trimming of outer braces and whitespace.
        """
        assert _strip_braces("{1,2,3}") == "1,2,3"

    def test_without_braces(self):
        """Ensure unbraced input is returned stripped.

        Input:
            A CSV string without braces and with outer whitespace.

        Expected:
            The same content with outer whitespace removed.

        Checks:
            Identity behavior for unbraced input aside from strip().
        """
        assert _strip_braces(" 1,2,3 ") == "1,2,3"

    def test_empty_braces(self):
        """Ensure empty braces return an empty string.

        Input:
            "{}" or variants with whitespace.

        Expected:
            Empty string.

        Checks:
            Special-case handling of empty brace pairs.
        """
        assert _strip_braces("{}") == ""

    def test_strip_braces_type_error(self):
        """_strip_braces raises TypeError when input is not a string.

        Input:
            A non-string value (e.g., None).

        Expected:
            TypeError is raised.

        Checks:
            Strict type checking in helper.
        """
        with pytest.raises(TypeError):
            _strip_braces(None)  # type: ignore[arg-type]


class TestSmartSplitCsv:
    def test_simple_list(self):
        """Split a braced CSV string into tokens.

        Input:
            "{a,b,c}".

        Expected:
            ["a", "b", "c"].

        Checks:
            Comma split and trim behavior.
        """
        assert _smart_split_csv("{a,b,c}") == ["a", "b", "c"]

    def test_with_spaces(self):
        """Trim spaces around tokens.

        Input:
            "{ a , b , c }".

        Expected:
            ["a", "b", "c"].

        Checks:
            Token trimming.
        """
        assert _smart_split_csv("{ a , b , c }") == ["a", "b", "c"]

    def test_empty_content(self):
        """Return empty list for empty braces.

        Input:
            "{}".

        Expected:
            [].

        Checks:
            Empty handling without error.
        """
        assert _smart_split_csv("{}") == []

    def test_no_braces(self):
        """Split unbraced CSV input.

        Input:
            "a, b".

        Expected:
            ["a", "b"].

        Checks:
            Braces are optional.
        """
        assert _smart_split_csv("a, b") == ["a", "b"]

    def test_smart_split_csv_type_error_and_empty_tokens(self):
        """_smart_split_csv raises TypeError for non-strings and ignores empty tokens.

        Input:
            text=None (error case) and text='a,, ,b' (empty tokens included).

        Expected:
            TypeError for non-string; ['a', 'b'] for the second input.

        Checks:
            Type validation and filtering of empty/space-only tokens.
        """
        with pytest.raises(TypeError):
            _smart_split_csv(None)  # type: ignore[arg-type]
        assert _smart_split_csv("a,, ,b") == ["a", "b"]


class TestCoerceScalar:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("42", 42),
            ("-5", -5),
            ("+7", 7),
            ("3.14", 3.14),
            ("1e3", 1000.0),
            ("abc", "abc"),
        ],
    )
    def test_various_tokens(self, token, expected):
        """Coerce numeric and non-numeric tokens appropriately.

        Input:
            A variety of tokens representing integers, floats (including
            scientific notation), and non-numeric strings.

        Expected:
            Integers are returned as int, floats as float, otherwise original
            string.

        Checks:
            Precedence of int over float for integer-like tokens.
        """
        assert _coerce_scalar(token) == expected

    def test_coerce_scalar_type_error_and_non_numeric_token(self):
        """_coerce_scalar raises TypeError for non-strings and preserves non-numeric tokens.

        Input:
            token=None (error) and token='0x10' (hex-like string).

        Expected:
            TypeError for non-string; '0x10' returned unchanged.

        Checks:
            Type validation and conservative coercion behavior.
        """
        with pytest.raises(TypeError):
            _coerce_scalar(None)  # type: ignore[arg-type]
        assert _coerce_scalar("0x10") == "0x10"


class TestParseValuesList:
    def test_mixed_values(self):
        """Parse a mixed list of ints, floats, and strings.

        Input:
            "{1, 2.5, abc}".

        Expected:
            [1, 2.5, "abc"].

        Checks:
            Correct type coercion.
        """
        result = _parse_values_list("{1, 2.5, abc}")
        assert result == [1, 2.5, "abc"]

    def test_empty_list(self):
        """Return empty list for empty braces.

        Input:
            "{}".

        Expected:
            [].

        Checks:
            Empty handling.
        """
        assert _parse_values_list("{}") == []

    def test_parse_values_list_type_error(self):
        """_parse_values_list raises TypeError for non-string input.

        Input:
            text=None.

        Expected:
            TypeError is raised.

        Checks:
            Type validation path.
        """
        with pytest.raises(TypeError):
            _parse_values_list(None)  # type: ignore[arg-type]


class TestFormatBracedList:
    def test_regular_values(self):
        """Format values into GDAL braced form.

        Input:
            [1, 2, 3].

        Expected:
            "{1,2,3}".

        Checks:
            No spaces and comma-join behavior.
        """
        assert _format_braced_list([1, 2, 3]) == "{1,2,3}"

    def test_empty_values(self):
        """Format empty list as empty braces.

        Input:
            [].

        Expected:
            "{}".

        Checks:
            Empty formatting behavior.
        """
        assert _format_braced_list([]) == "{}"

    def test_mixed_types(self):
        """Format mixed-type values without casting errors.

        Input:
            ["a", 1, 2.5].

        Expected:
            "{a,1,2.5}".

        Checks:
            Generic string conversion for all element types.
        """
        assert _format_braced_list(["a", 1, 2.5]) == "{a,1,2.5}"

    def test_format_braced_list_non_iterable_raises(self):
        """_format_braced_list raises TypeError when values is not iterable.

        Input:
            values=None (non-iterable).

        Expected:
            A TypeError due to iteration attempt over None.

        Checks:
            The helper relies on Python iteration semantics and thus raises TypeError.
        """
        with pytest.raises(TypeError):
            _format_braced_list(None)  # type: ignore[arg-type]


class TestDimMetaData:
    def test_basic_construction(self):
        """Construct DimMetaData with explicit fields.

        Input:
            name="time", size=2, values=[0, 31], def_fields=(2, 6).

        Expected:
            Attributes are stored exactly; raw defaults to empty dict.

        Checks:
            Dataclass field assignment and defaults.
        """
        d = DimMetaData(name="time", size=2, values=[0, 31], def_fields=(2, 6))
        assert d.name == "time"
        assert d.size == 2
        assert d.values == [0, 31]
        assert d.def_fields == (2, 6)
        assert d.raw == {}


class TestDimensionsIndex:
    def test_full_example_with_extra_def_values(self):
        """Parse dimensions with EXTRA, DEF and VALUES present.

        Input:
            A metadata dict containing:
                - NETCDF_DIM_EXTRA: "{time,level0}"
                - DEF/VALUES entries for both names

        Expected:
            Dimensions "time" and "level0" exist; sizes/values parsed.

        Checks:
            EXTRA list parsing, DEF first integer = size, VALUES coercion.
        """
        md = {
            'NETCDF_DIM_EXTRA': '{time,level0}',
            'NETCDF_DIM_level0_DEF': '{3,6}',
            'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            'NETCDF_DIM_time_DEF': '{2,6}',
            'NETCDF_DIM_time_VALUES': '{0,31}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert set(idx.names) == {"time", "level0"}
        assert idx["time"].size == 2
        assert idx["level0"].values == [1, 2, 3]
        assert idx["level0"].def_fields == (3, 6)

    def test_without_extra_names_inferred_from_keys(self):
        """Discover dimensions via DEF/VALUES when EXTRA is missing.

        Input:
            Metadata without NETCDF_DIM_EXTRA but with DEF/VALUES entries.

        Expected:
            The dimension is still created.

        Checks:
            Robustness to missing EXTRA list.
        """
        md = {
            'NETCDF_DIM_depth_DEF': '{4,99}',
            'NETCDF_DIM_depth_VALUES': '{0, 10, 20, 30}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx.names == ["depth"]
        assert idx["depth"].size == 4
        assert idx["depth"].values == [0, 10, 20, 30]

    def test_values_define_size_when_def_missing(self):
        """Infer size from VALUES length when DEF is absent.

        Input:
            Only a VALUES entry.

        Expected:
            size == len(values); def_fields == None.

        Checks:
            VALUES-only size inference.
        """
        md = {
            'NETCDF_DIM_time_VALUES': '{0, 31, 59}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx["time"].size == 3
        assert idx["time"].values == [0, 31, 59]
        assert idx["time"].def_fields is None

    def test_def_with_non_integers_ignored_in_def_fields(self):
        """Filter non-integers out of DEF and use first integer as size.

        Input:
            DEF has a mix of ints and non-ints; VALUES present.

        Expected:
            def_fields == (first_integer, ...); size==first_integer.

        Checks:
            Integer filtering, VALUES unaffected.
        """
        md = {
            'NETCDF_DIM_level_DEF': '{3, 2.5, x}',
            'NETCDF_DIM_level_VALUES': '{10,20,30}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx["level"].def_fields == (3,)
        assert idx["level"].size == 3
        assert idx["level"].values == [10, 20, 30]

    def test_missing_values_size_from_def(self):
        """If VALUES missing, size comes from DEF first value."""
        md = {'NETCDF_DIM_var_DEF': '{5,100}'}
        idx = DimensionsIndex.from_metadata(md)
        assert idx["var"].size == 5

    def test_non_prefixed_keys_ignored(self):
        """Keys without NETCDF_DIM_ prefix are ignored."""
        md = {'OTHER_KEY': 'abc'}
        idx = DimensionsIndex.from_metadata(md)
        assert idx.names == []

    def test_extra_key_only(self):
        """EXTRA entry with list of dimensions -> dimensions created."""
        md = {'NETCDF_DIM_EXTRA': '{dim1,dim2}'}
        idx = DimensionsIndex.from_metadata(md)
        assert set(idx.names) == {"dim1", "dim2"}

    def test_def_with_non_ints(self):
        """DEF entry with non-integer tokens -> ignored in def_fields."""
        md = {'NETCDF_DIM_var_DEF': '{a,2}'}
        idx = DimensionsIndex.from_metadata(md)
        assert idx["var"].def_fields == (2,)
        assert idx["var"].size == 2

    def test_unknown_prefixed_keys_are_ignored(self):
        """Ignore prefixed keys that do not match the regex.

        Input:
            A bad key like "NETCDF_DIM_".

        Expected:
            No dimensions created and no exceptions.

        Checks:
            Resilience to unexpected keys under prefix.
        """
        md = {
            'NETCDF_DIM_': '{oops}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert len(idx.names) == 0

    def test_custom_prefix_is_respected(self):
        """Support alternate prefixes when specified.

        Input:
            Keys starting with "CUSTOM_DIM_".

        Expected:
            Default parse finds nothing; custom prefix parses dimension.

        Checks:
            Prefix argument behavior.
        """
        md = {
            'CUSTOM_DIM_time_DEF': '{2,6}',
            'CUSTOM_DIM_time_VALUES': '{0,31}',
        }
        idx_default = DimensionsIndex.from_metadata(md)
        assert len(idx_default.names) == 0
        idx_custom = DimensionsIndex.from_metadata(md, prefix='CUSTOM_DIM_')
        assert idx_custom.names == ["time"]
        assert idx_custom["time"].size == 2
        assert idx_custom["time"].values == [0, 31]

    def test_mixed_case_extra_is_handled(self):
        """Recognize EXTRA regardless of case.

        Input:
            Key "NETCDF_DIM_ExTrA" listing names.

        Expected:
            Names extracted and parsed; DEF reflected in size when available.

        Checks:
            Case-insensitivity for EXTRA name.
        """
        md = {
            'NETCDF_DIM_ExTrA': '{TIME,LEVEL0}',
            'NETCDF_DIM_TIME_VALUES': '{0, 31}',
            'NETCDF_DIM_LEVEL0_DEF': '{3,6}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert set(idx.names) == {"TIME", "LEVEL0"}
        assert idx["LEVEL0"].size == 3


class TestIndexApi:
    def test_names_property(self):
        """Expose the list of dimension names.

        Input:
            One simple dimension.

        Expected:
            names == ["time"].

        Checks:
            names property content.
        """
        md = {'NETCDF_DIM_time_DEF': '{2,6}'}
        idx = DimensionsIndex.from_metadata(md)
        assert idx.names == ["time"]

    def test_getitem_and_contains(self):
        """Support mapping-like behaviors.

        Input:
            One dimension defined by VALUES.

        Expected:
            "level0" in idx and idx["level0"] returns a DimMetaData.

        Checks:
            __contains__ and __getitem__.
        """
        md = {'NETCDF_DIM_level0_VALUES': '{1,2,3}'}
        idx = DimensionsIndex.from_metadata(md)
        assert "level0" in idx
        dim = idx["level0"]
        assert isinstance(dim, DimMetaData)
        assert dim.values == [1, 2, 3]

    def test_len_and_iter(self):
        """Implement len() and iteration over dimensions.

        Input:
            Two dimensions.

        Expected:
            len(idx) == 2 and iter yields DimMetaData objects.

        Checks:
            __len__ and __iter__.
        """
        md = {
            'NETCDF_DIM_a_VALUES': '{1}',
            'NETCDF_DIM_b_VALUES': '{2}',
        }
        idx = DimensionsIndex.from_metadata(md)
        assert len(idx) == 2
        assert all(isinstance(d, DimMetaData) for d in iter(idx))

    def test_to_dict_serialization(self):
        """Serialize to a plain dictionary.

        Input:
            DEF and VALUES provided.

        Expected:
            Dictionary includes size, values, and def_fields.

        Checks:
            to_dict() structure matches content.
        """
        md = {
            'NETCDF_DIM_z_DEF': '{2,6}',
            'NETCDF_DIM_z_VALUES': '{10, 20}',
        }
        idx = DimensionsIndex.from_metadata(md)
        out = idx.to_dict()
        assert out == {"z": {"size": 2, "values": [10, 20], "def_fields": (2, 6)}}


class TestStrMethod:
    def test_str_includes_key_information(self):
        """Render a readable string with size, values and def fields.

        Input:
            Two dimensions with different field availability.

        Expected:
            Output starts with header line and contains each dimension block with
            the appropriate pieces.

        Checks:
            Presence of size, values, and def tuple when provided.
        """
        md = {
            'NETCDF_DIM_level0_DEF': '{3,6}',
            'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            'NETCDF_DIM_time_VALUES': '{0,31}',
        }
        idx = DimensionsIndex.from_metadata(md)
        s = str(idx)
        assert s.splitlines()[0].startswith("DimensionsIndex(2 dims)")
        assert (
            "- level0:" in s
            and "size=3" in s
            and "values=[1, 2, 3]" in s
            and "def=(3, 6)" in s
        )
        assert "- time:" in s and "values=[0, 31]" in s


class TestToMetadataMethod:
    def test_roundtrip_default(self):
        """Serialize to metadata and check keys and formats.

        Input:
            Index with DEF and VALUES for two names.

        Expected:
            Keys NETCDF_DIM_EXTRA, NETCDF_DIM_*_DEF and NETCDF_DIM_*_VALUES with
            braced lists and no spaces.

        Checks:
            Deterministic sorting of names; correct omission when fields are None.
        """
        md = {
            'NETCDF_DIM_level0_DEF': '{3,6}',
            'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            'NETCDF_DIM_time_VALUES': '{0,31}',
        }
        idx = DimensionsIndex.from_metadata(md)
        out = idx.to_metadata()
        # Names should be sorted
        assert out["NETCDF_DIM_EXTRA"] == "{level0,time}"
        assert out["NETCDF_DIM_level0_DEF"] == "{3,6}"
        assert out["NETCDF_DIM_level0_VALUES"] == "{1,2,3}"
        assert out["NETCDF_DIM_time_VALUES"] == "{0,31}"
        assert "NETCDF_DIM_time_DEF" not in out

    def test_no_extra_key(self):
        """Omit EXTRA when include_extra=False.

        Input:
            include_extra=False.

        Expected:
            No NETCDF_DIM_EXTRA key is present.

        Checks:
            Optional inclusion of EXTRA key.
        """
        md = {'NETCDF_DIM_time_VALUES': '{0,31}'}
        idx = DimensionsIndex.from_metadata(md)
        out = idx.to_metadata(include_extra=False)
        assert "NETCDF_DIM_EXTRA" not in out
        assert out["NETCDF_DIM_time_VALUES"] == "{0,31}"

    def test_custom_prefix(self):
        """Use a custom prefix for output keys.

        Input:
            prefix="CUSTOM_DIM_".

        Expected:
            Keys begin with CUSTOM_DIM_.

        Checks:
            Prefix parameter is honored in outputs.
        """
        md = {'NETCDF_DIM_time_VALUES': '{0,31}'}
        idx = DimensionsIndex.from_metadata(md)
        out = idx.to_metadata(prefix="CUSTOM_DIM_")
        assert out["CUSTOM_DIM_EXTRA"] == "{time}"
        assert out["CUSTOM_DIM_time_VALUES"] == "{0,31}"

    def test_unsorted_names(self):
        """Preserve insertion order when sort_names=False.

        Input:
            Multiple names inserted in a specific order.

        Expected:
            EXTRA list reflects insertion order.

        Checks:
            sort_names flag controls ordering.
        """
        md = {
            'NETCDF_DIM_b_VALUES': '{2}',
            'NETCDF_DIM_a_VALUES': '{1}',
        }
        idx = DimensionsIndex.from_metadata(md)
        out_sorted = idx.to_metadata(sort_names=True)
        out_unsorted = idx.to_metadata(sort_names=False)
        assert out_sorted["NETCDF_DIM_EXTRA"] == "{a,b}"
        # Insertion order is whatever dict kept; since we parsed sorted(dim_names)
        # inside from_metadata, the internal order is sorted. To truly test
        # sort_names=False we create an index manually.
        idx2 = DimensionsIndex(
            {
                "b": DimMetaData(name="b", values=[2], size=1),
                "a": DimMetaData(name="a", values=[1], size=1),
            }
        )
        out_manual = idx2.to_metadata(sort_names=False)
        assert out_manual["NETCDF_DIM_EXTRA"] == "{b,a}"

    def test_omit_missing_fields(self):
        """Do not emit DEF/VALUES keys when data are missing.

        Input:
            A dimension with only DEF, and another with only VALUES.

        Expected:
            Corresponding keys emitted only when present.

        Checks:
            Conditional emission for each field.
        """
        idx = DimensionsIndex(
            {
                "onlydef": DimMetaData(name="onlydef", def_fields=(5, 1), size=5),
                "onlyvals": DimMetaData(name="onlyvals", values=[10, 20]),
            }
        )
        out = idx.to_metadata()
        assert out["NETCDF_DIM_EXTRA"] in ("{onlydef,onlyvals}", "{onlyvals,onlydef}")
        assert out["NETCDF_DIM_onlydef_DEF"] == "{5,1}"
        assert "NETCDF_DIM_onlydef_VALUES" not in out
        assert out["NETCDF_DIM_onlyvals_VALUES"] == "{10,20}"
        assert "NETCDF_DIM_onlyvals_DEF" not in out


class TestDimensionsIndexAdditionalEdgeCases:
    def test_to_metadata_empty_index(self):
        """When the index is empty, to_metadata emits no keys.

        Input:
            An empty DimensionsIndex.

        Expected:
            Returned dict is empty; no EXTRA key is produced.

        Checks:
            Guard on empty names list for EXTRA emission.
        """
        idx = DimensionsIndex({})
        out = idx.to_metadata()
        assert out == {}

    def test_dimension_name_regex_allows_punctuation(self):
        """Names with '.', '-' and '_' are supported by the regex used in from_metadata.

        Input:
            Keys like NETCDF_DIM_x.y-1_2_VALUES.

        Expected:
            A dimension with the exact name is created and values are parsed.

        Checks:
            The character class [A-Za-z0-9_.-]+ is honored by the parser.
        """
        md = {
            "NETCDF_DIM_x.y-1_2_VALUES": "{1,2}",
        }
        idx = DimensionsIndex.from_metadata(md)
        assert idx.names == ["x.y-1_2"]
        assert idx["x.y-1_2"].values == [1, 2]


class TestParseGdalNetcdfDimensions:
    def test_wrapper_function(self):
        """Convenience wrapper delegates to from_metadata."""
        md = {'NETCDF_DIM_var_VALUES': '{4,5}'}
        idx = parse_gdal_netcdf_dimensions(md)
        assert idx["var"].values == [4, 5]
