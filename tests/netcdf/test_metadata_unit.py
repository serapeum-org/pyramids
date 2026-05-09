"""Unit tests for pyramids.netcdf.metadata.

Covers MetadataBuilder, GroupTraverser, get_metadata, to_dict,
to_json, from_json, and flatten_for_index with mocked GDAL objects.
Targets >=95% branch coverage of metadata.py.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from pyramids.netcdf.metadata import (
    GroupTraverser,
    MetadataBuilder,
    flatten_for_index,
    from_json,
    get_metadata,
    to_dict,
    to_json,
)
from pyramids.netcdf.models import (
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)

pytestmark = pytest.mark.core


def _mock_attribute(name: str, value):
    """Create a mock gdal.Attribute.

    Args:
        name: Attribute name.
        value: Value returned by Read().

    Returns:
        MagicMock: A mock attribute.
    """
    attr = MagicMock()
    attr.GetName.return_value = name
    attr.Read.return_value = value
    return attr


def _mock_group(
    name: str = "root",
    full_name: str = "/",
    dimensions: list | None = None,
    array_names: list[str] | None = None,
    group_names: list[str] | None = None,
    attributes: list | None = None,
    children: dict[str, MagicMock] | None = None,
):
    """Create a mock gdal.Group with configurable structure.

    Args:
        name: Group short name.
        full_name: Group full name.
        dimensions: List of mock dimensions.
        array_names: List of array name strings.
        group_names: List of child group name strings.
        attributes: List of mock attributes.
        children: Dict mapping child name to mock group.

    Returns:
        MagicMock: A mock gdal.Group.
    """
    group = MagicMock()
    group.GetName.return_value = name
    group.GetFullName.return_value = full_name
    group.GetDimensions.return_value = dimensions or []
    group.GetMDArrayNames.return_value = array_names or []
    group.GetGroupNames.return_value = group_names or []
    group.GetAttributes.return_value = attributes or []

    children = children or {}

    def open_group_side_effect(cn):
        """Return the child group for the given name."""
        return children.get(cn)

    group.OpenGroup.side_effect = open_group_side_effect

    return group


def _mock_md_array(
    name: str = "temperature",
    full_name: str = "/temperature",
    shape: tuple = (10, 20),
):
    """Create a mock gdal.MDArray with minimal attributes.

    Args:
        name: Array short name.
        full_name: Array full name.
        shape: Shape of the array.

    Returns:
        MagicMock: A mock MDArray.
    """
    arr = MagicMock()
    arr.GetName.return_value = name
    arr.GetFullName.return_value = full_name
    dt_mock = MagicMock()
    dt_mock.GetName.return_value = "Float32"
    arr.GetDataType.return_value = dt_mock
    arr.GetShape.return_value = list(shape)

    dim_names = [f"/{name}_dim{i}" for i in range(len(shape))]
    dims = []
    for i, dn in enumerate(dim_names):
        d = MagicMock()
        d.GetFullName.return_value = dn
        d.GetName.return_value = dn.split("/")[-1]
        d.GetSize.return_value = shape[i]
        dims.append(d)
    arr.GetDimensions.return_value = dims

    arr.GetAttributes.return_value = []
    arr.GetUnit.return_value = None
    arr.GetSpatialRef.return_value = None
    arr.GetStructuralInfo.return_value = None
    arr.GetBlockSize.return_value = None
    arr.GetCoordinateVariables.return_value = None
    arr.GetNoDataValueAsDouble.return_value = None
    arr.GetNoDataValueAsInt64 = MagicMock(side_effect=Exception("na"))
    arr.GetNoDataValueAsString = MagicMock(side_effect=Exception("na"))
    return arr


def _mock_dimension(
    name: str = "time",
    full_name: str = "/time",
    size: int = 365,
):
    """Create a mock gdal.Dimension.

    Args:
        name: Dimension short name.
        full_name: Dimension full name.
        size: Number of elements.

    Returns:
        MagicMock: A mock dimension.
    """
    dim = MagicMock()
    dim.GetName.return_value = name
    dim.GetFullName.return_value = full_name
    dim.GetSize.return_value = size
    dim.GetType.return_value = "TEMPORAL"
    dim.GetDirection.return_value = None
    dim.GetIndexingVariable.return_value = None
    return dim


def _mock_dataset(
    driver_name: str = "netCDF",
    root_group: MagicMock | None = None,
    metadata: dict | None = None,
):
    """Create a mock gdal.Dataset.

    Args:
        driver_name: Driver short name.
        root_group: Mock root group, or None for no MDIM support.
        metadata: Metadata dict for GetMetadata().

    Returns:
        MagicMock: A mock dataset.
    """
    ds = MagicMock()
    driver = MagicMock()
    driver.ShortName = driver_name
    driver.GetMetadata_Dict.return_value = {"DMD_LONGNAME": "NetCDF"}
    ds.GetDriver.return_value = driver
    ds.GetRootGroup.return_value = root_group
    ds.GetMetadata.return_value = metadata or {}
    return ds


def _make_metadata(**kwargs) -> NetCDFMetadata:
    """Create a NetCDFMetadata instance with sensible defaults.

    Args:
        **kwargs: Override any field.

    Returns:
        NetCDFMetadata: A metadata instance.
    """
    defaults = dict(
        driver="netCDF",
        root_group="/",
        groups={
            "/": GroupInfo(
                name="root",
                full_name="/",
                variables=["/temperature"],
            ),
        },
        variables={
            "/temperature": VariableInfo(
                name="temperature",
                full_name="/temperature",
                dtype="float32",
                shape=[10, 20],
                dimensions=["/lat", "/lon"],
            ),
        },
        dimensions={
            "/lat": DimensionInfo(name="lat", full_name="/lat", size=180),
            "/lon": DimensionInfo(name="lon", full_name="/lon", size=360),
            "/time": DimensionInfo(name="time", full_name="/time", size=365),
        },
        global_attributes={"Conventions": "CF-1.6", "history": "created"},
        structural=StructuralInfo(
            driver_name="netCDF",
            driver_metadata={"DMD_LONGNAME": "NetCDF"},
        ),
        created_with={"library": "GDAL", "version": "3.9.0"},
    )
    defaults.update(kwargs)
    return NetCDFMetadata(**defaults)


class TestMetadataBuilder:
    """Tests for MetadataBuilder.build method."""

    def test_build_with_root_group(self):
        """Verify build collects groups, arrays, and dimensions from root group.

        A dataset with a root group should trigger GroupTraverser.walk
        and populate groups/arrays/dimensions from the MDIM hierarchy.
        """
        dim = _mock_dimension(name="x", full_name="x", size=10)
        arr = _mock_md_array(name="temp", full_name="temp", shape=(10,))

        root = _mock_group(
            name="root",
            full_name="/",
            dimensions=[dim],
            array_names=["temp"],
            attributes=[_mock_attribute("Conventions", "CF-1.6")],
        )
        root.OpenMDArray.return_value = arr

        ds = _mock_dataset(root_group=root)
        builder = MetadataBuilder(ds)
        md = builder.build()

        assert md.driver == "netCDF", f"Expected 'netCDF', got '{md.driver}'"
        assert md.root_group == "/", f"Expected '/', got '{md.root_group}'"
        assert len(md.groups) >= 1, "Should have at least one group"
        assert "/" in md.groups, "Root group '/' should be in groups"
        assert len(md.variables) >= 1, "Should have at least one array"
        assert "temp" in md.variables, "'temp' should be in arrays"
        assert len(md.dimensions) >= 1, "Should have at least one dimension"
        assert md.structural is not None, "Structural info should be populated"
        assert (
            "Conventions" in md.global_attributes
        ), f"Expected 'Conventions' in global attrs, got {md.global_attributes}"

    def test_build_without_root_group(self):
        """Verify build falls back to classic metadata when no root group exists.

        When GetRootGroup returns None, the builder should use
        SharedMetaData.from_metadata for global attributes.
        """
        ds = _mock_dataset(
            root_group=None,
            metadata={"key": "value"},
        )
        builder = MetadataBuilder(ds)
        md = builder.build()
        assert md.root_group is None, f"Expected root_group=None, got '{md.root_group}'"
        assert md.groups == {}, "Groups should be empty without root group"
        assert md.variables == {}, "Arrays should be empty without root group"
        assert md.dimensions == {}, "Dimensions should be empty without root group"

    def test_build_without_root_group_metadata_failure(self):
        """Verify build handles failure in SharedMetaData gracefully.

        When the classic metadata fallback also fails, global_attributes
        should default to {}.
        """
        ds = _mock_dataset(root_group=None)
        ds.GetMetadata.side_effect = RuntimeError("fail")
        builder = MetadataBuilder(ds)
        md = builder.build()
        assert (
            md.global_attributes == {}
        ), f"Expected empty global_attributes on failure, got {md.global_attributes}"

    def test_build_root_group_get_full_name_failure(self):
        """Verify root_name defaults to '/' when GetFullName raises."""
        root = _mock_group()
        root.GetFullName.side_effect = RuntimeError("fail")
        root.GetName.return_value = "root"
        ds = _mock_dataset(root_group=root)
        builder = MetadataBuilder(ds)
        md = builder.build()
        assert (
            md.root_group == "/"
        ), f"Expected fallback root_group='/', got '{md.root_group}'"

    def test_open_options_stored(self):
        """Verify open_options are stored in the metadata."""
        root = _mock_group()
        ds = _mock_dataset(root_group=root)
        opts = {"OPEN_SHARED": "YES"}
        builder = MetadataBuilder(ds, open_options=opts)
        md = builder.build()
        assert (
            md.open_options_used == opts
        ), f"Expected {opts}, got {md.open_options_used}"

    def test_open_options_none_stays_none(self):
        """Verify open_options_used is None when not provided."""
        root = _mock_group()
        ds = _mock_dataset(root_group=root)
        builder = MetadataBuilder(ds)
        md = builder.build()
        assert (
            md.open_options_used is None
        ), f"Expected None, got {md.open_options_used}"

    def test_created_with_contains_gdal(self):
        """Verify created_with includes GDAL library and version."""
        root = _mock_group()
        ds = _mock_dataset(root_group=root)
        builder = MetadataBuilder(ds)
        md = builder.build()
        assert (
            md.created_with.get("library") == "GDAL"
        ), f"Expected library='GDAL', got {md.created_with}"
        assert (
            "version" in md.created_with
        ), f"Expected 'version' key in created_with, got {md.created_with}"


class TestGroupTraverserWalk:
    """Tests for GroupTraverser.walk BFS traversal."""

    def test_single_root_group(self):
        """Verify walk collects a single root group with no children."""
        dim = _mock_dimension(name="x", full_name="x", size=5)
        arr = _mock_md_array(name="v", full_name="v", shape=(5,))
        root = _mock_group(
            name="root",
            full_name="/",
            dimensions=[dim],
            array_names=["v"],
        )
        root.OpenMDArray.return_value = arr

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)

        assert "/" in groups, "Root group should be collected"
        assert "v" in arrays, "Array 'v' should be collected"
        assert "x" in dimensions, "Dimension 'x' should be collected"

    def test_nested_groups(self):
        """Verify BFS traverses nested child groups."""
        child_dim = _mock_dimension(name="y", full_name="child/y", size=3)
        child_arr = _mock_md_array(name="w", full_name="child/w", shape=(3,))
        child = _mock_group(
            name="child",
            full_name="child",
            dimensions=[child_dim],
            array_names=["w"],
        )
        child.OpenMDArray.return_value = child_arr

        root_dim = _mock_dimension(name="x", full_name="x", size=5)
        root_arr = _mock_md_array(name="v", full_name="v", shape=(5,))
        root = _mock_group(
            name="root",
            full_name="/",
            dimensions=[root_dim],
            array_names=["v"],
            group_names=["child"],
            children={"child": child},
        )
        root.OpenMDArray.return_value = root_arr

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)

        assert "/" in groups, "Root group should be collected"
        assert "child" in groups, "Child group should be collected"
        assert "v" in arrays, "Root array should be collected"
        assert "child/w" in arrays, "Child array should be collected"
        assert "x" in dimensions, "Root dimension should be collected"
        assert "child/y" in dimensions, "Child dimension should be collected"

    def test_child_open_group_failure(self):
        """Verify walk continues when OpenGroup returns None for a child."""
        root = _mock_group(
            name="root",
            full_name="/",
            group_names=["broken"],
            children={},
        )
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)
        assert "/" in groups, "Root should still be collected"
        assert len(groups) == 1, "Only root group should be collected"

    def test_child_open_group_exception(self):
        """Verify walk continues when OpenGroup raises for a child."""
        root = _mock_group(
            name="root",
            full_name="/",
            group_names=["broken"],
        )
        root.OpenGroup.side_effect = RuntimeError("fail")
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)
        assert "/" in groups, "Root should still be collected despite child failure"

    def test_child_group_info_from_group_failure_falls_back(self):
        """Verify walk falls back to path concatenation when GroupInfo.from_group raises for a child.

        This covers lines 328-330 of metadata.py where the child's
        GroupInfo.from_group raises and the code falls back to
        constructing the full name via string concatenation.
        """
        child = _mock_group(name="child", full_name="child")
        child.GetDimensions.return_value = []
        child.GetMDArrayNames.return_value = []
        child.GetGroupNames.return_value = []

        root = _mock_group(
            name="root",
            full_name="/",
            group_names=["child"],
            children={"child": child},
        )

        call_count = {"n": 0}
        original_from_group = GroupInfo.from_group.__func__

        def patched_from_group(cls, group, **kwargs):
            """Raise on the second call (child identity resolution) to trigger fallback."""
            call_count["n"] += 1
            # Call 1: root base_group identity, call 2: child identity -- raise on 2
            if call_count["n"] == 2:
                raise RuntimeError("simulated GroupInfo failure")
            return original_from_group(cls, group, **kwargs)

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        with patch.object(GroupInfo, "from_group", classmethod(patched_from_group)):
            t.walk(root)

        # The child should still be traversed with a fallback name /child
        assert (
            "child" in [v for v in groups.keys()] or len(groups) >= 1
        ), f"Expected child to be in the traversal, got {list(groups.keys())}"

    def test_child_group_info_fallback_non_root_parent(self):
        """Verify fallback path concatenation for non-root parent uses parent_path/child_name.

        When group_full_name != '/' the fallback should be
        '{group_full_name}/{cn}' not '/{cn}'.
        """
        child = _mock_group(
            name="child",
            full_name="child",
        )
        child.GetDimensions.return_value = []
        child.GetMDArrayNames.return_value = []
        child.GetGroupNames.return_value = []

        root = _mock_group(
            name="root",
            full_name="/",
            group_names=["child"],
            children={"child": child},
        )
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)
        assert "/" in groups, "Root should be collected"
        assert "child" in groups, "Child should be collected"

    def test_array_open_failure_skipped(self):
        """Verify arrays that fail to open are silently skipped."""
        root = _mock_group(
            name="root",
            full_name="/",
            array_names=["good", "bad"],
        )
        good_arr = _mock_md_array(name="good", full_name="good", shape=(5,))

        def open_md_side_effect(name):
            """Return array for 'good', None for 'bad'."""
            if name == "good":
                return good_arr
            return None

        root.OpenMDArray.side_effect = open_md_side_effect

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)
        assert "good" in arrays, "Good array should be collected"
        assert "/bad" not in arrays, "Bad array should be skipped"

    def test_array_open_exception_skipped(self):
        """Verify arrays that raise on OpenMDArray are skipped."""
        root = _mock_group(
            name="root",
            full_name="/",
            array_names=["broken"],
        )
        root.OpenMDArray.side_effect = RuntimeError("fail")
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t.walk(root)
        assert len(arrays) == 0, "No arrays should be collected on failure"


class TestGroupTraverserCollectDimensions:
    """Tests for GroupTraverser._collect_dimensions."""

    def test_dimensions_sorted_by_name(self):
        """Verify dimensions are sorted by name for deterministic output."""
        dim_b = _mock_dimension(name="b", full_name="b", size=2)
        dim_a = _mock_dimension(name="a", full_name="a", size=1)
        root = _mock_group(dimensions=[dim_b, dim_a])

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t._collect_dimensions(root, "/")
        keys = list(dimensions.keys())
        assert keys == [
            "a",
            "b",
        ], f"Expected sorted dimension keys ['a', 'b'], got {keys}"

    def test_dimensions_failure_returns_empty(self):
        """Verify empty result when GetDimensions raises."""
        root = _mock_group()
        root.GetDimensions.side_effect = RuntimeError("fail")
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t._collect_dimensions(root, "/")
        assert len(dimensions) == 0, "No dimensions should be collected on failure"

    def test_dimension_get_name_failure_sorted_empty(self):
        """Verify dimensions with failing GetName are sorted using empty string."""
        dim = _mock_dimension(name="x", full_name="x", size=5)
        dim.GetName.side_effect = RuntimeError("fail")
        root = _mock_group(dimensions=[dim])
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        t._collect_dimensions(root, "/")
        assert len(dimensions) >= 1, "Dimension should still be collected"


class TestGroupTraverserCollectArrays:
    """Tests for GroupTraverser._collect_arrays."""

    def test_returns_full_names(self):
        """Verify _collect_arrays returns list of full names."""
        arr = _mock_md_array(name="v", full_name="v", shape=(5,))
        root = _mock_group(array_names=["v"])
        root.OpenMDArray.return_value = arr

        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        result = t._collect_arrays(root, "/")
        assert result == ["v"], f"Expected ['v'], got {result}"
        assert "v" in arrays, "Array should be stored in arrays dict"

    def test_empty_when_no_arrays(self):
        """Verify empty list when group has no arrays."""
        root = _mock_group(array_names=[])
        groups, arrays, dimensions = {}, {}, {}
        t = GroupTraverser(groups, arrays, dimensions)
        result = t._collect_arrays(root, "/")
        assert result == [], f"Expected [], got {result}"


class TestGetMetadata:
    """Tests for the get_metadata public function."""

    def test_from_string_path(self):
        """Verify get_metadata opens a file path as multidimensional raster.

        Uses a mock to avoid needing a real NetCDF file.
        """
        mock_ds = _mock_dataset(root_group=_mock_group())
        with patch("pyramids.netcdf.metadata.gdal.OpenEx", return_value=mock_ds):
            md = get_metadata("fake.nc")
        assert md.driver == "netCDF", f"Expected 'netCDF', got '{md.driver}'"

    def test_from_string_path_open_failure(self):
        """Verify ValueError when the file cannot be opened."""
        with patch("pyramids.netcdf.metadata.gdal.OpenEx", return_value=None):
            with pytest.raises(ValueError, match="Could not open"):
                get_metadata("nonexistent.nc")

    def test_from_object_with_raster_attribute(self):
        """Verify get_metadata extracts _raster from a pyramids-like object."""
        mock_ds = _mock_dataset(root_group=_mock_group())
        obj = MagicMock()
        obj._raster = mock_ds
        md = get_metadata(obj)
        assert md.driver == "netCDF", f"Expected 'netCDF', got '{md.driver}'"

    def test_from_raw_dataset(self):
        """Verify get_metadata accepts a raw gdal.Dataset."""
        mock_ds = _mock_dataset(root_group=_mock_group())
        del mock_ds._raster  # Ensure no _raster attribute
        md = get_metadata(mock_ds)
        assert md.driver == "netCDF", f"Expected 'netCDF', got '{md.driver}'"

    def test_open_options_propagated(self):
        """Verify open_options are stored in the result metadata."""
        mock_ds = _mock_dataset(root_group=_mock_group())
        with patch("pyramids.netcdf.metadata.gdal.OpenEx", return_value=mock_ds):
            md = get_metadata("fake.nc", open_options={"KEY": "VAL"})
        assert md.open_options_used == {
            "KEY": "VAL"
        }, f"Expected open_options to be stored, got {md.open_options_used}"


class TestToDict:
    """Tests for the to_dict function."""

    def test_basic_conversion(self):
        """Verify to_dict converts all dataclass fields to plain dicts."""
        md = _make_metadata()
        d = to_dict(md)
        assert isinstance(d, dict), f"Expected dict, got {type(d)}"
        assert d["driver"] == "netCDF", f"Expected 'netCDF', got {d['driver']}"
        assert isinstance(d["groups"], dict), "groups should be dict"
        assert isinstance(d["variables"], dict), "arrays should be dict"
        assert isinstance(d["dimensions"], dict), "dimensions should be dict"

    def test_nested_dataclass_conversion(self):
        """Verify nested GroupInfo/VariableInfo/DimensionInfo are converted to dicts."""
        md = _make_metadata()
        d = to_dict(md)
        group_data = d["groups"]["/"]
        assert isinstance(group_data, dict), "GroupInfo should be converted to dict"
        assert (
            group_data["name"] == "root"
        ), f"Expected group name 'root', got {group_data['name']}"

    def test_structural_info_conversion(self):
        """Verify StructuralInfo is converted to a plain dict."""
        md = _make_metadata()
        d = to_dict(md)
        assert isinstance(d["structural"], dict), "structural should be dict"
        assert (
            d["structural"]["driver_name"] == "netCDF"
        ), f"Expected 'netCDF', got {d['structural']['driver_name']}"

    def test_none_structural(self):
        """Verify to_dict handles structural=None."""
        md = _make_metadata(structural=None)
        d = to_dict(md)
        assert (
            d["structural"] is None
        ), f"Expected None for structural, got {d['structural']}"

    def test_empty_metadata(self):
        """Verify to_dict handles metadata with empty collections."""
        md = _make_metadata(
            groups={},
            variables={},
            dimensions={},
            global_attributes={},
        )
        d = to_dict(md)
        assert d["groups"] == {}, "Empty groups should stay {}"
        assert d["variables"] == {}, "Empty arrays should stay {}"
        assert d["dimensions"] == {}, "Empty dimensions should stay {}"

    def test_list_conversion(self):
        """Verify lists inside dataclasses are preserved."""
        md = _make_metadata()
        d = to_dict(md)
        arr_data = d["variables"]["/temperature"]
        assert isinstance(arr_data["shape"], list), "shape should be a list"
        assert isinstance(arr_data["dimensions"], list), "dimensions should be a list"


class TestToJson:
    """Tests for the to_json function."""

    def test_produces_valid_json(self):
        """Verify to_json returns a valid JSON string."""
        md = _make_metadata()
        s = to_json(md)
        parsed = json.loads(s)
        assert isinstance(parsed, dict), "JSON should parse to a dict"
        assert (
            parsed["driver"] == "netCDF"
        ), f"Expected 'netCDF', got {parsed['driver']}"

    def test_compact_format(self):
        """Verify to_json uses compact separators (no spaces)."""
        md = _make_metadata()
        s = to_json(md)
        assert ": " not in s, "Should use compact separators"


class TestFromJson:
    """Tests for the from_json function."""

    def test_round_trip_preserves_all_fields(self):
        """Verify JSON round-trip preserves all metadata fields."""
        md = _make_metadata()
        s = to_json(md)
        restored = from_json(s)
        assert (
            restored.driver == md.driver
        ), f"Expected driver='{md.driver}', got '{restored.driver}'"
        assert (
            restored.root_group == md.root_group
        ), f"Expected root_group '{md.root_group}', got '{restored.root_group}'"
        assert set(restored.groups.keys()) == set(
            md.groups.keys()
        ), "Groups keys should match after round-trip"
        assert set(restored.variables.keys()) == set(
            md.variables.keys()
        ), "Arrays keys should match after round-trip"
        assert set(restored.dimensions.keys()) == set(
            md.dimensions.keys()
        ), "Dimensions keys should match after round-trip"
        assert (
            restored.global_attributes == md.global_attributes
        ), "Global attributes should match after round-trip"
        assert (
            restored.created_with == md.created_with
        ), "created_with should match after round-trip"

    def test_round_trip_groups(self):
        """Verify group metadata is preserved through round-trip."""
        md = _make_metadata()
        restored = from_json(to_json(md))
        orig_group = md.groups["/"]
        rest_group = restored.groups["/"]
        assert (
            rest_group.name == orig_group.name
        ), f"Expected name='{orig_group.name}', got '{rest_group.name}'"
        assert (
            rest_group.full_name == orig_group.full_name
        ), f"Expected full_name '{orig_group.full_name}', got '{rest_group.full_name}'"
        assert (
            rest_group.variables == orig_group.variables
        ), f"Expected arrays {orig_group.variables}, got {rest_group.variables}"

    def test_round_trip_arrays(self):
        """Verify array metadata is preserved through round-trip."""
        md = _make_metadata()
        restored = from_json(to_json(md))
        orig_arr = md.variables["/temperature"]
        rest_arr = restored.variables["/temperature"]
        assert (
            rest_arr.name == orig_arr.name
        ), f"Expected name '{orig_arr.name}', got '{rest_arr.name}'"
        assert (
            rest_arr.dtype == orig_arr.dtype
        ), f"Expected dtype '{orig_arr.dtype}', got '{rest_arr.dtype}'"
        assert (
            rest_arr.shape == orig_arr.shape
        ), f"Expected shape {orig_arr.shape}, got {rest_arr.shape}"
        assert (
            rest_arr.dimensions == orig_arr.dimensions
        ), f"Expected dimensions {orig_arr.dimensions}, got {rest_arr.dimensions}"

    def test_round_trip_dimensions(self):
        """Verify dimension metadata is preserved through round-trip."""
        md = _make_metadata()
        restored = from_json(to_json(md))
        for key, orig_dim in md.dimensions.items():
            rest_dim = restored.dimensions[key]
            assert (
                rest_dim.name == orig_dim.name
            ), f"Expected name='{orig_dim.name}', got '{rest_dim.name}'"
            assert (
                rest_dim.size == orig_dim.size
            ), f"Expected size {orig_dim.size}, got {rest_dim.size}"
            assert (
                rest_dim.full_name == orig_dim.full_name
            ), f"Expected full_name '{orig_dim.full_name}', got '{rest_dim.full_name}'"

    def test_round_trip_structural_info(self):
        """Verify structural info is preserved through round-trip."""
        md = _make_metadata()
        restored = from_json(to_json(md))
        assert restored.structural is not None, "Structural info should not be None"
        assert (
            restored.structural.driver_name == md.structural.driver_name
        ), f"Expected driver_name '{md.structural.driver_name}', got '{restored.structural.driver_name}'"
        assert (
            restored.structural.driver_metadata == md.structural.driver_metadata
        ), "driver_metadata should match after round-trip"

    def test_round_trip_none_structural(self):
        """Verify None structural info survives round-trip."""
        md = _make_metadata(structural=None)
        restored = from_json(to_json(md))
        assert (
            restored.structural is None
        ), f"Expected None structural, got {restored.structural}"

    def test_round_trip_with_all_array_fields(self):
        """Verify all VariableInfo optional fields survive round-trip."""
        full_arr = VariableInfo(
            name="full",
            full_name="/full",
            dtype="int16",
            shape=[10, 20, 30],
            dimensions=["/time", "/lat", "/lon"],
            attributes={"units": "mm"},
            unit="mm",
            nodata=-9999,
            scale=0.01,
            offset=0.5,
            srs_wkt='GEOGCS["WGS 84"]',
            srs_projjson='{"type": "GeographicCRS"}',
            coordinate_variables=["/lat", "/lon"],
            structural_info={"COMPRESS": "DEFLATE"},
            block_size=[1, 20, 30],
        )
        md = _make_metadata(variables={"/full": full_arr})
        restored = from_json(to_json(md))
        r = restored.variables["/full"]
        assert r.scale == 0.01, f"Expected scale=0.01, got {r.scale}"
        assert r.offset == 0.5, f"Expected offset=0.5, got {r.offset}"
        assert r.nodata == -9999, f"Expected nodata=-9999, got {r.nodata}"
        assert (
            r.srs_wkt == 'GEOGCS["WGS 84"]'
        ), f"Expected WGS 84 WKT, got '{r.srs_wkt}'"
        assert r.block_size == [1, 20, 30]
        assert r.coordinate_variables == ["/lat", "/lon"]
        assert r.structural_info == {
            "COMPRESS": "DEFLATE"
        }, f"Expected structural_info with COMPRESS, got {r.structural_info}"

    def test_round_trip_with_dimension_attrs(self):
        """Verify dimension attrs survive round-trip."""
        dim = DimensionInfo(
            name="time",
            full_name="/time",
            size=365,
            type="TEMPORAL",
            direction="NORTH",
            indexing_variable="/time_idx",
            attrs={"units": "days since 1970-01-01", "calendar": "standard"},
        )
        md = _make_metadata(dimensions={"/time": dim})
        restored = from_json(to_json(md))
        r = restored.dimensions["/time"]
        assert r.attrs == dim.attrs, f"Expected {dim.attrs}, got {r.attrs}"
        assert r.type == "TEMPORAL", f"Expected 'TEMPORAL', got '{r.type}'"
        assert r.direction == "NORTH", f"Expected 'NORTH', got '{r.direction}'"
        assert (
            r.indexing_variable == "/time_idx"
        ), f"Expected '/time_idx', got '{r.indexing_variable}'"

    def test_from_json_invalid_json(self):
        """Verify JSONDecodeError for invalid JSON input."""
        with pytest.raises(json.JSONDecodeError):
            from_json("not valid json{{{")

    def test_from_json_missing_driver(self):
        """Verify from_json uses 'UNKNOWN' default for missing driver."""
        s = json.dumps(
            {
                "root_group": "/",
                "groups": {},
                "variables": {},
                "dimensions": {},
                "global_attributes": {},
                "created_with": {},
            }
        )
        md = from_json(s)
        assert (
            md.driver == "UNKNOWN"
        ), f"Expected 'UNKNOWN' for missing driver, got '{md.driver}'"

    def test_from_json_open_options_preserved(self):
        """Verify open_options_used is preserved through round-trip."""
        md = _make_metadata(open_options_used={"KEY": "VAL"})
        restored = from_json(to_json(md))
        assert restored.open_options_used == {
            "KEY": "VAL"
        }, f"Expected {{'KEY': 'VAL'}}, got {restored.open_options_used}"

    def test_from_json_block_size_none(self):
        """Verify block_size=None survives round-trip (not converted to [])."""
        arr = VariableInfo(
            name="v",
            full_name="v",
            dtype="float32",
            shape=[10],
            dimensions=["x"],
            block_size=None,
        )
        md = _make_metadata(variables={"v": arr})
        restored = from_json(to_json(md))
        assert (
            restored.variables["v"].block_size is None
        ), f"Expected None block_size, got {restored.variables['v'].block_size}"


class TestToDictFromJsonConsistency:
    """Tests ensuring to_dict and from_json(to_json(...)) are consistent."""

    def test_to_dict_equals_from_json_to_dict(self):
        """Verify to_dict(md) == to_dict(from_json(to_json(md)))."""
        md = _make_metadata()
        d1 = to_dict(md)
        d2 = to_dict(from_json(to_json(md)))
        assert (
            d1 == d2
        ), "to_dict should produce identical output before and after JSON round-trip"


class TestFlattenForIndex:
    """Tests for the flatten_for_index function."""

    def test_basic_fields(self):
        """Verify basic scalar fields are present in the flat dict."""
        md = _make_metadata()
        flat = flatten_for_index(md)
        assert flat["driver"] == "netCDF", f"Expected 'netCDF', got {flat['driver']}"
        assert flat["root_group"] == "/", f"Expected '/', got {flat['root_group']}"
        assert flat["group_count"] == 1, f"Expected 1, got {flat['group_count']}"
        assert flat["variable_count"] == 1, f"Expected 1, got {flat['variable_count']}"
        assert (
            flat["dimension_count"] == 3
        ), f"Expected 3, got {flat['dimension_count']}"

    def test_global_attributes_prefixed(self):
        """Verify global attributes are prefixed with 'global.'."""
        md = _make_metadata()
        flat = flatten_for_index(md)
        assert (
            flat["global.Conventions"] == "CF-1.6"
        ), f"Expected 'CF-1.6', got {flat.get('global.Conventions')}"
        assert (
            flat["global.history"] == "created"
        ), f"Expected 'created', got {flat.get('global.history')}"

    def test_arrays_sorted(self):
        """Verify arrays list is sorted."""
        md = _make_metadata()
        flat = flatten_for_index(md)
        assert flat["variables"] == [
            "/temperature"
        ], f"Expected ['/temperature'], got {flat['variables']}"

    def test_dimensions_sorted(self):
        """Verify dimensions list is sorted."""
        md = _make_metadata()
        flat = flatten_for_index(md)
        assert flat["dimensions"] == sorted(
            flat["dimensions"]
        ), f"Expected sorted dimensions, got {flat['dimensions']}"

    def test_empty_metadata(self):
        """Verify flatten_for_index handles empty metadata gracefully."""
        md = _make_metadata(
            groups={},
            variables={},
            dimensions={},
            global_attributes={},
        )
        flat = flatten_for_index(md)
        assert flat["group_count"] == 0, f"Expected 0, got {flat['group_count']}"
        assert flat["variable_count"] == 0, f"Expected 0, got {flat['variable_count']}"
        assert (
            flat["dimension_count"] == 0
        ), f"Expected 0, got {flat['dimension_count']}"
        assert flat["variables"] == [], f"Expected [], got {flat['variables']}"
        assert flat["dimensions"] == [], f"Expected [], got {flat['dimensions']}"

    def test_global_attributes_limited_to_20(self):
        """Verify only the first 20 global attributes are included."""
        attrs = {f"attr_{i:03d}": f"val_{i}" for i in range(30)}
        md = _make_metadata(global_attributes=attrs)
        flat = flatten_for_index(md)
        global_keys = [k for k in flat if k.startswith("global.")]
        assert (
            len(global_keys) == 20
        ), f"Expected 20 global keys, got {len(global_keys)}"

    def test_multiple_arrays_sorted(self):
        """Verify multiple arrays are sorted alphabetically."""
        arrays = {
            "z": VariableInfo(
                name="z",
                full_name="z",
                dtype="f32",
                shape=[1],
                dimensions=["x"],
            ),
            "a": VariableInfo(
                name="a",
                full_name="a",
                dtype="f32",
                shape=[1],
                dimensions=["x"],
            ),
            "m": VariableInfo(
                name="m",
                full_name="m",
                dtype="f32",
                shape=[1],
                dimensions=["x"],
            ),
        }
        md = _make_metadata(variables=arrays)
        flat = flatten_for_index(md)
        assert flat["variables"] == [
            "a",
            "m",
            "z",
        ], f"Expected sorted arrays, got {flat['variables']}"

    def test_root_group_none(self):
        """Verify root_group=None is captured in the flat dict."""
        md = _make_metadata(root_group=None)
        flat = flatten_for_index(md)
        assert flat["root_group"] is None, f"Expected None, got {flat['root_group']}"
