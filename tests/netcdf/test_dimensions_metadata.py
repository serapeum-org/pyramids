import pytest

from pyramids.netcdf.dimensions import (
    DimensionsIndex,
    DimMetaData,
    MetaData,
    parse_dimension_attributes,
)

pytestmark = pytest.mark.core


class TestParseDimensionAttributes:
    def test_parse_all_attributes(self):
        """Parse attribute keys of the form "<name>#<attr>" for all names.

        Input:
            A flat metadata dict with keys like "time#axis" and "lat#units".

        Expected:
            A nested mapping {name: {attr: value}} for each encountered name,
            with attribute keys normalized to lowercase by default.

        Checks:
            Grouping by name and lowercasing of attribute keys.
        """
        md = {
            "lat#bounds": "bounds_lat",
            "lat#Long_Name": "latitude",
            "lat#units": "degrees_north",
            "time#axis": "T",
            "time#long_name": "time",
            "time#units": "days since 1-1-1 0:0:0",
            # unrelated keys should be ignored
            "NETCDF_DIM_time_DEF": "{2,6}",
        }
        out = parse_dimension_attributes(md)
        assert out == {
            "lat": {
                "bounds": "bounds_lat",
                "long_name": "latitude",
                "units": "degrees_north",
            },
            "time": {
                "axis": "T",
                "long_name": "time",
                "units": "days since 1-1-1 0:0:0",
            },
        }

    def test_filter_by_names(self):
        """Only include attributes for provided dimension names.

        Input:
            names=["time"] with metadata containing both lat and time attributes.

        Expected:
            Only the "time" entry is present in the result.

        Checks:
            Filtering by the optional names argument.
        """
        md = {
            "lat#units": "degrees_north",
            "time#axis": "T",
            "time#units": "days",
        }
        out = parse_dimension_attributes(md, names=["time"])  # default normalize=True
        assert out == {"time": {"axis": "T", "units": "days"}}

    def test_preserve_attribute_key_case(self):
        """Preserve original attribute key case when normalize_attr_keys=False.

        Input:
            normalize_attr_keys=False with mixed-case attribute keys.

        Expected:
            Attribute keys are preserved exactly as in the input (e.g., "LongName").

        Checks:
            The normalize_attr_keys flag controls attribute key casing.
        """
        md = {
            "time#Axis": "T",
            "time#LongName": "Time",
        }
        out = parse_dimension_attributes(md, normalize_attr_keys=False)
        assert out == {"time": {"Axis": "T", "LongName": "Time"}}

    def test_ignore_invalid_attribute_keys(self):
        """Ignore keys that do not match the "<name>#<attr>" pattern.

        Input:
            Keys missing a name or attribute (e.g., "#units", "time#").

        Expected:
            These keys are skipped and not present in the output.

        Checks:
            Strict matching against the simple "name#attr" regex; whitespace is not allowed.
        """
        md = {
            "#units": "oops",
            "time#": "oops",
            " time#key1": "val1",
            "time#key2 ": "val2",
            "time##units": "oops",
            # valid entry to ensure normal behavior coexists
            "lat#units": "degrees_north",
        }
        out = parse_dimension_attributes(md)
        assert out == {
            "lat": {"units": "degrees_north"},
            "time": {"key1": "val1", "key2": "val2"},
        }

    def test_empty_input_returns_empty_mapping(self):
        """Return an empty mapping when no attribute-like keys are present.

        Input:
            An empty dict.

        Expected:
            An empty dict.

        Checks:
            Graceful handling of empty inputs.
        """
        assert parse_dimension_attributes({}) == {}


class TestMetaDataFromMetadata:
    def test_builds_from_combined_mapping_default(self):
        """Combine dimension structure and attributes from a single mapping.

        Input:
            NETCDF_DIM_* keys for two dimensions and attribute keys for one.

        Expected:
            MetaData carries both DimensionsIndex and per-name attrs; get_dimension merges them.

        Checks:
            Parsed sizes/values and attribute lookup via get_attrs and get_dimension.
        """
        md = {
            "NETCDF_DIM_EXTRA": "{time,level0}",
            "NETCDF_DIM_time_DEF": "{2,6}",
            "NETCDF_DIM_time_VALUES": "{0,31}",
            "NETCDF_DIM_level0_VALUES": "{1,2,3}",
            "time#axis": "T",
            "level0#axis": "Z",
            "level0#units": "hPa",
        }
        meta = MetaData.from_metadata(md)
        assert sorted(meta.names) == ["level0", "time"]
        assert meta.get_attrs("level0") == {"axis": "Z", "units": "hPa"}
        dim_time = meta.get_dimension("time")
        assert isinstance(dim_time, DimMetaData)
        assert dim_time.size == 2 and dim_time.values == [0, 31]
        assert dim_time.attrs == {"axis": "T"}

    def test_custom_prefix_and_names_filter(self):
        """Honor custom prefix for dimensions and names filter for attributes.

        Input:
            Dimension keys under CUSTOM_DIM_ and attributes for time/lat; names=["time"].

        Expected:
            Only the "time" attributes are collected; dimension index built from custom prefix.

        Checks:
            Prefix and names parameters are correctly propagated.
        """
        md = {
            "CUSTOM_DIM_time_DEF": "{2,6}",
            "CUSTOM_DIM_time_VALUES": "{0,31}",
            "time#axis": "T",
            "lat#units": "degrees_north",
        }
        meta = MetaData.from_metadata(md, prefix="CUSTOM_DIM_", names=["time"])
        assert meta.names == ["time"]
        assert meta.get_attrs("time") == {"axis": "T"}
        assert meta.get_attrs("lat") == {}

    def test_do_not_normalize_attribute_keys(self):
        """Keep attribute key case when normalize_attr_keys=False.

        Input:
            Mixed-case keys after '#'.

        Expected:
            Attributes are stored with original case.

        Checks:
            The normalize_attr_keys flag is respected through MetaData.from_metadata.
        """
        md = {
            "NETCDF_DIM_time_DEF": "{2,6}",
            "time#Axis": "T",
            "time#LongName": "Time",
        }
        meta = MetaData.from_metadata(md, normalize_attr_keys=False)
        assert meta.get_attrs("time") == {"Axis": "T", "LongName": "Time"}


class TestMetaDataNamesProperty:
    def test_exposes_names(self):
        """Expose underlying DimensionsIndex names via property.

        Input:
            One simple dimension.

        Expected:
            names == ["time"].

        Checks:
            Property passthrough to dims.names.
        """
        md = {"NETCDF_DIM_time_DEF": "{2,6}"}
        meta = MetaData.from_metadata(md)
        assert meta.names == ["time"]


class TestMetaDataGetAttrs:
    def test_existing_and_unknown_names(self):
        """Return attribute mapping for known name and empty dict for unknown.

        Input:
            One attribute key for "time" and a query for missing name "lat".

        Expected:
            get_attrs("time") returns dict; get_attrs("lat") returns {}.

        Checks:
            Missing names handled gracefully.
        """
        md = {"NETCDF_DIM_time_DEF": "{2,6}", "time#units": "days"}
        meta = MetaData.from_metadata(md)
        assert meta.get_attrs("time") == {"units": "days"}
        assert meta.get_attrs("lat") == {}


class TestMetaDataGetDimension:
    def test_merge_and_unknown(self):
        """Return a DimMetaData merged with attributes; None for unknown names.

        Input:
            time dimension with DEF/VALUES and one attribute; request "time" and "lat".

        Expected:
            A DimMetaData with fields copied from DimensionsIndex and attrs merged for "time";
            None for "lat".

        Checks:
            get_dimension merges and preserves raw/def/values; unknown name returns None.
        """
        md = {
            "NETCDF_DIM_time_DEF": "{2,6}",
            "NETCDF_DIM_time_VALUES": "{0,31}",
            "time#axis": "T",
        }
        meta = MetaData.from_metadata(md)
        d = meta.get_dimension("time")
        assert isinstance(d, DimMetaData)
        assert (d.name, d.size, d.values, d.def_fields) == ("time", 2, [0, 31], (2, 6))
        assert d.attrs == {"axis": "T"}
        assert meta.get_dimension("lat") is None


class TestMetaDataIterDimensions:
    def test_sorted_iteration(self):
        """Iterate merged dimensions in name-sorted order.

        Input:
            Two dimensions 'b' and 'a'.

        Expected:
            The iteration yields names in ['a', 'b'] order.

        Checks:
            Sorting behavior of iter_dimensions().
        """
        md = {
            "NETCDF_DIM_b_DEF": "{1,0}",
            "NETCDF_DIM_a_DEF": "{2,0}",
        }
        meta = MetaData.from_metadata(md)
        names = [d.name for d in meta.iter_dimensions()]
        assert names == ["a", "b"]


class TestMetaDataToMetadata:
    def test_merge_structure_and_attrs(self):
        """Serialize combined structure and attributes into a flat mapping.

        Input:
            MetaData with DEF/VALUES and attributes for two names.

        Expected:
            Keys include NETCDF_DIM_* for structure and "<name>#<attr>" for attributes.

        Checks:
            Deterministic name and attribute ordering when sort_names=True.
        """
        md = {
            "NETCDF_DIM_level0_DEF": "{3,6}",
            "NETCDF_DIM_level0_VALUES": "{1,2,3}",
            "NETCDF_DIM_time_VALUES": "{0,31}",
            "time#axis": "T",
            "level0#units": "hPa",
            "level0#axis": "Z",
        }
        meta = MetaData.from_metadata(md)
        out = meta.to_metadata()
        # Name ordering is deterministic: level0 before time
        assert out["NETCDF_DIM_EXTRA"] == "{level0,time}"
        # Attribute keys are sorted per-name (axis before units)
        assert list(
            k for k in out.keys() if k.endswith("#axis") or k.endswith("#units")
        ) == [
            "level0#axis",
            "level0#units",
            "time#axis",
        ]

    def test_exclude_attrs_and_extra(self):
        """Optionally exclude attribute keys and EXTRA from serialization.

        Input:
            include_attrs=False and include_extra=False.

        Expected:
            Only structural NETCDF_DIM_* keys are present; no EXTRA and no # keys.

        Checks:
            flags include_attrs and include_extra in MetaData.to_metadata().
        """
        md = {"NETCDF_DIM_time_VALUES": "{0,31}", "time#axis": "T"}
        meta = MetaData.from_metadata(md)
        out = meta.to_metadata(include_attrs=False, include_extra=False)
        assert "NETCDF_DIM_EXTRA" not in out
        assert "time#axis" not in out
        assert out["NETCDF_DIM_time_VALUES"] == "{0,31}"

    def test_sort_names_false_preserves_order(self):
        """When sort_names=False, preserve the provided insertion order for attribute emission.

        Input:
            A MetaData constructed manually with specific dims and attrs order.

        Expected:
            EXTRA reflects insertion order and attribute keys are emitted per that order.

        Checks:
            sort_names flag affects both names and attribute emission ordering.
        """
        meta = MetaData(
            dims=DimensionsIndex(
                {
                    "b": DimMetaData(name="b", size=1, values=[2]),
                    "a": DimMetaData(name="a", size=1, values=[1]),
                }
            ),
            attrs={
                "b": {"axis": "Y"},
                "a": {"axis": "X"},
            },
        )
        out = meta.to_metadata(sort_names=False)
        # Because dims were inserted as b, a the EXTRA should preserve that order
        assert out["NETCDF_DIM_EXTRA"] == "{b,a}"
        # The relative order of attribute keys by name follows b then a
        keys = [k for k in out.keys() if "#" in k]
        assert keys[:2] == ["b#axis", "a#axis"]

    def test_custom_prefix_and_attr_keys_sorted(self):
        """Use a custom prefix for structure and verify per-name attribute keys are sorted.

        Input:
            prefix="CUSTOM_DIM_" and attributes with multiple keys.

        Expected:
            Structural keys start with CUSTOM_DIM_; attribute keys are sorted alphabetically.

        Checks:
            Prefix passthrough and stable attribute key ordering.
        """
        meta = MetaData(
            dims=DimensionsIndex(
                {
                    "x": DimMetaData(name="x", def_fields=(2, 6), size=2),
                }
            ),
            attrs={
                "x": {"units": "m", "axis": "X"},
            },
        )
        out = meta.to_metadata(prefix="CUSTOM_DIM_")
        assert out["CUSTOM_DIM_EXTRA"] == "{x}"
        # Attribute keys for x must be sorted: axis then units
        keys = [k for k in out.keys() if k.startswith("x#")]
        assert keys == ["x#axis", "x#units"]


class TestMetaDataStr:
    def test_str_contains_summary_and_details(self):
        """Render a readable summary including counts and per-dimension details.

        Input:
            Two dimensions, one with values+attrs, another with only DEF.

        Expected:
            First line shows MetaData(<n> dims, attrs for <m> names).
            Subsequent lines contain name, size, number of values, and attr counts.

        Checks:
            Presence of expected substrings and proper counts.
        """
        meta = MetaData(
            dims=DimensionsIndex(
                {
                    "time": DimMetaData(
                        name="time", size=2, values=[0, 31], def_fields=(2, 6)
                    ),
                    "level": DimMetaData(name="level", size=3, def_fields=(3, 6)),
                }
            ),
            attrs={
                "time": {"axis": "T", "units": "days"},
            },
        )
        s = str(meta)
        # Header
        assert s.splitlines()[0].startswith("MetaData(2 dims, attrs for 1 names)")
        # time details
        assert any(
            line.startswith("- time:")
            and "size=2" in line
            and "values=2 items" in line
            and "attrs=2" in line
            for line in s.splitlines()
        )
        # level details
        assert any(
            line.startswith("- level:") and "size=3" in line for line in s.splitlines()
        )
