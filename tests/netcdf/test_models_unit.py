"""Unit tests for pyramids.netcdf.models.

Covers GroupInfo, DimensionInfo, VariableInfo, StructuralInfo, and
NetCDFMetadata with mocked GDAL objects. Targets >=95% branch
coverage of models.py.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from pyramids.netcdf.models import (
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)

pytestmark = pytest.mark.core


def _mock_group(
    name: str = "root",
    full_name: str = "/",
    attributes: list | None = None,
):
    """Create a mock gdal.Group with configurable name and attributes.

    Args:
        name: Short name returned by GetName().
        full_name: Full name returned by GetFullName().
        attributes: List of mock gdal.Attribute objects.

    Returns:
        MagicMock: A mock gdal.Group.
    """
    group = MagicMock()
    group.GetName.return_value = name
    group.GetFullName.return_value = full_name
    group.GetAttributes.return_value = attributes or []
    return group


def _mock_dimension(
    name: str = "time",
    full_name: str = "/time",
    size: int = 365,
    dtype: str = "TEMPORAL",
    direction: str = "NORTH",
    indexing_variable: MagicMock | None = None,
):
    """Create a mock gdal.Dimension.

    Args:
        name: Dimension short name.
        full_name: Dimension full name.
        size: Size of the dimension.
        dtype: Dimension type string.
        direction: Direction string.
        indexing_variable: Mock indexing variable, or None.

    Returns:
        MagicMock: A mock gdal.Dimension.
    """
    dim = MagicMock()
    dim.GetName.return_value = name
    dim.GetFullName.return_value = full_name
    dim.GetSize.return_value = size
    dim.GetType.return_value = dtype
    dim.GetDirection.return_value = direction
    dim.GetIndexingVariable.return_value = indexing_variable
    return dim


def _mock_md_array(
    name: str = "temperature",
    full_name: str = "/temperature",
    shape: tuple = (365, 180, 360),
    dim_names: list[str] | None = None,
    unit: str = "K",
    nodata: float | None = -9999.0,
    srs: MagicMock | None = None,
):
    """Create a mock gdal.MDArray.

    Args:
        name: Array short name.
        full_name: Array full name.
        shape: Shape of the array.
        dim_names: List of dimension full names.
        unit: Unit string.
        nodata: No-data value or None.
        srs: Mock spatial reference or None.

    Returns:
        MagicMock: A mock gdal.MDArray.
    """
    arr = MagicMock()
    arr.GetName.return_value = name
    arr.GetFullName.return_value = full_name

    dt_mock = MagicMock()
    dt_mock.GetName.return_value = "Float32"
    arr.GetDataType.return_value = dt_mock
    arr.GetShape.return_value = list(shape)

    if dim_names is None:
        dim_names = ["/time", "/lat", "/lon"]
    dims = []
    for i, dn in enumerate(dim_names):
        d = MagicMock()
        d.GetFullName.return_value = dn
        d.GetName.return_value = dn.split("/")[-1]
        d.GetSize.return_value = shape[i] if i < len(shape) else 0
        dims.append(d)
    arr.GetDimensions.return_value = dims

    arr.GetAttributes.return_value = []
    arr.GetUnit.return_value = unit
    arr.GetSpatialRef.return_value = srs
    arr.GetStructuralInfo.return_value = None
    arr.GetBlockSize.return_value = None
    arr.GetCoordinateVariables.return_value = None

    # nodata via attributes (CF _FillValue handled via _read_attributes mock)
    arr.GetNoDataValueAsDouble.return_value = nodata if nodata is not None else None
    arr.GetNoDataValueAsInt64 = MagicMock(side_effect=Exception("not available"))
    arr.GetNoDataValueAsString = MagicMock(side_effect=Exception("not available"))

    return arr


class TestGroupInfoFromGroup:
    """Tests for GroupInfo.from_group class method."""

    def test_basic_construction(self):
        """Verify from_group extracts name, full_name, and pre-supplied arrays/children.

        The factory should delegate to _get_group_name and
        _full_name_with_fallback for identity resolution.
        """
        group = _mock_group(name="root", full_name="/")
        info = GroupInfo.from_group(
            group,
            variables=["/temperature"],
            children=["/forecast"],
            attributes={"Conventions": "CF-1.6"},
        )
        assert info.name == "root", f"Expected name='root', got '{info.name}'"
        assert info.full_name == "/", f"Expected full_name='/', got '{info.full_name}'"
        assert info.variables == [
            "/temperature"
        ], f"Expected variables=['/temperature'], got {info.variables}"
        assert info.children == [
            "/forecast"
        ], f"Expected children=['/forecast'], got {info.children}"
        assert info.attributes == {
            "Conventions": "CF-1.6"
        }, f"Expected attributes with Conventions key, got {info.attributes}"

    def test_attributes_read_from_group_when_none(self):
        """Verify attributes are read from the group when not pre-supplied.

        When attributes=None, the factory calls _read_attributes(group).
        """
        attr_mock = MagicMock()
        attr_mock.GetName.return_value = "history"
        attr_mock.Read.return_value = "created"
        group = _mock_group(name="root", full_name="/", attributes=[attr_mock])
        info = GroupInfo.from_group(
            group,
            variables=[],
            children=[],
            attributes=None,
        )
        assert (
            "history" in info.attributes
        ), f"Expected 'history' in attributes, got {info.attributes}"

    def test_empty_arrays_and_children_become_empty_lists(self):
        """Verify empty iterables are normalised to empty lists.

        Passing empty lists for arrays and children should yield [].
        """
        group = _mock_group()
        info = GroupInfo.from_group(
            group,
            variables=[],
            children=[],
            attributes={},
        )
        assert info.variables == [], f"Expected empty arrays list, got {info.variables}"
        assert info.children == [], f"Expected empty children list, got {info.children}"

    def test_none_arrays_and_children_become_empty_lists(self):
        """Verify None arrays and children are normalised to empty lists."""
        group = _mock_group()
        info = GroupInfo.from_group(
            group,
            variables=None,
            children=None,
            attributes={},
        )
        assert info.variables == [], f"Expected empty arrays list, got {info.variables}"
        assert info.children == [], f"Expected empty children list, got {info.children}"

    def test_group_get_name_failure_returns_empty_name(self):
        """Verify graceful degradation when GetName raises an exception."""
        group = _mock_group()
        group.GetName.side_effect = RuntimeError("GDAL error")
        group.GetFullName.return_value = "/"
        info = GroupInfo.from_group(group, variables=[], children=[], attributes={})
        assert info.name == "", f"Expected empty name on failure, got '{info.name}'"

    def test_group_get_full_name_failure_uses_fallback(self):
        """Verify fallback path when GetFullName raises an exception.

        Should fall back to '/<name>' constructed from GetName().
        """
        group = _mock_group(name="subgroup")
        group.GetFullName.side_effect = RuntimeError("GDAL error")
        info = GroupInfo.from_group(group, variables=[], children=[], attributes={})
        assert (
            info.full_name == "/subgroup"
        ), f"Expected fallback full_name='/subgroup', got '{info.full_name}'"

    def test_group_both_name_methods_fail_gives_root(self):
        """Verify '/' is returned when both GetFullName and GetName fail."""
        group = _mock_group()
        group.GetName.side_effect = RuntimeError("fail")
        group.GetFullName.side_effect = RuntimeError("fail")
        info = GroupInfo.from_group(group, variables=[], children=[], attributes={})
        assert (
            info.full_name == "/"
        ), f"Expected full_name='/' when both methods fail, got '{info.full_name}'"

    def test_none_attributes_dict_becomes_empty(self):
        """Verify that if _read_attributes returns None/empty, attributes={}.

        When attributes is None and the group has no attributes,
        _read_attributes returns {} which maps to {}.
        """
        group = _mock_group()
        group.GetAttributes.return_value = []
        info = GroupInfo.from_group(group, variables=[], children=[], attributes=None)
        assert (
            info.attributes == {}
        ), f"Expected empty attributes, got {info.attributes}"


class TestGroupInfoDataclass:
    """Tests for GroupInfo dataclass defaults and frozen behavior."""

    def test_defaults(self):
        """Verify default factory fields produce empty containers."""
        info = GroupInfo(name="g", full_name="/g")
        assert info.attributes == {}, "Default attributes should be {}"
        assert info.children == [], "Default children should be []"
        assert info.variables == [], "Default arrays should be []"

    def test_frozen_prevents_assignment(self):
        """Verify frozen dataclass raises on attribute assignment."""
        info = GroupInfo(name="g", full_name="/g")
        with pytest.raises(AttributeError):
            info.name = "other"


class TestDimensionInfoFromGdalDim:
    """Tests for DimensionInfo.from_gdal_dim class method."""

    def test_basic_construction(self):
        """Verify all fields are extracted from a fully functional mock dimension."""
        dim = _mock_dimension(
            name="time",
            full_name="/time",
            size=365,
            dtype="TEMPORAL",
            direction="NORTH",
        )
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert info.name == "time", f"Expected name='time', got '{info.name}'"
        assert info.full_name == "/time", f"Expected '/time', got '{info.full_name}'"
        assert info.size == 365, f"Expected size=365, got {info.size}"
        assert info.type == "TEMPORAL", f"Expected type='TEMPORAL', got '{info.type}'"
        assert (
            info.direction == "NORTH"
        ), f"Expected direction='NORTH', got '{info.direction}'"

    def test_indexing_variable_full_name(self):
        """Verify indexing variable full name is captured when available."""
        iv = MagicMock()
        iv.GetFullName.return_value = "/time_idx"
        iv.GetAttributes.return_value = []
        dim = _mock_dimension(indexing_variable=iv)
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.indexing_variable == "/time_idx"
        ), f"Expected '/time_idx', got '{info.indexing_variable}'"

    def test_indexing_variable_name_fallback(self):
        """Verify fallback to GetName when GetFullName is absent on the indexing variable."""
        iv = MagicMock(spec=["GetName", "GetAttributes"])
        iv.GetName.return_value = "time_idx"
        iv.GetAttributes.return_value = []
        del iv.GetFullName
        dim = _mock_dimension(indexing_variable=iv)
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.indexing_variable == "time_idx"
        ), f"Expected 'time_idx', got '{info.indexing_variable}'"

    def test_indexing_variable_none(self):
        """Verify indexing_variable is None when GetIndexingVariable returns None."""
        dim = _mock_dimension(indexing_variable=None)
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.indexing_variable is None
        ), f"Expected None, got '{info.indexing_variable}'"

    def test_indexing_variable_attrs_read(self):
        """Verify attrs are read from the indexing variable when present."""
        attr_mock = MagicMock()
        attr_mock.GetName.return_value = "calendar"
        attr_mock.Read.return_value = "standard"
        iv = MagicMock()
        iv.GetFullName.return_value = "/time"
        iv.GetAttributes.return_value = [attr_mock]
        dim = _mock_dimension(indexing_variable=iv)
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.attrs.get("calendar") == "standard"
        ), f"Expected calendar='standard', got {info.attrs}"

    def test_get_name_failure(self):
        """Verify empty name when GetName raises."""
        dim = _mock_dimension()
        dim.GetName.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert info.name == "", f"Expected empty name, got '{info.name}'"

    def test_get_full_name_failure_uses_fallback(self):
        """Verify full_name fallback uses group_full_name/dim_name."""
        dim = _mock_dimension(name="lat")
        dim.GetFullName.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/root")
        assert (
            info.full_name == "/root/lat"
        ), f"Expected '/root/lat', got '{info.full_name}'"

    def test_get_full_name_fallback_root_group(self):
        """Verify full_name fallback with root group uses /dim_name."""
        dim = _mock_dimension(name="lon")
        dim.GetFullName.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert info.full_name == "/lon", f"Expected '/lon', got '{info.full_name}'"

    def test_get_size_failure(self):
        """Verify size defaults to 0 when GetSize raises."""
        dim = _mock_dimension()
        dim.GetSize.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert info.size == 0, f"Expected size=0, got {info.size}"

    def test_get_type_failure(self):
        """Verify type defaults to None when GetType raises."""
        dim = _mock_dimension()
        dim.GetType.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert info.type is None, f"Expected type=None, got '{info.type}'"

    def test_get_direction_failure(self):
        """Verify direction defaults to None when GetDirection raises."""
        dim = _mock_dimension()
        dim.GetDirection.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.direction is None
        ), f"Expected direction=None, got '{info.direction}'"

    def test_get_indexing_variable_failure(self):
        """Verify indexing_variable is None when GetIndexingVariable raises."""
        dim = _mock_dimension()
        dim.GetIndexingVariable.side_effect = RuntimeError("fail")
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.indexing_variable is None
        ), f"Expected indexing_variable=None, got '{info.indexing_variable}'"
        assert (
            info.attrs == {}
        ), f"Expected empty attrs when indexing variable fails, got {info.attrs}"

    def test_read_attributes_failure_on_indexing_variable(self):
        """Verify attrs defaults to {} when _read_attributes raises on the iv."""
        iv = MagicMock()
        iv.GetFullName.return_value = "/time"
        iv.GetAttributes.side_effect = RuntimeError("fail")
        dim = _mock_dimension(indexing_variable=iv)
        info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.attrs == {}
        ), f"Expected empty attrs on read failure, got {info.attrs}"

    def test_read_attributes_exception_propagation_caught(self):
        """Verify attrs defaults to {} when _read_attributes raises unexpectedly.

        This covers the outer except on line 282-283 of models.py where
        _read_attributes itself raises rather than returning {}.
        """
        iv = MagicMock()
        iv.GetFullName.return_value = "/time"
        iv.GetAttributes.return_value = []
        dim = _mock_dimension(indexing_variable=iv)
        with patch(
            "pyramids.netcdf.models._read_attributes",
            side_effect=RuntimeError("unexpected"),
        ):
            info = DimensionInfo.from_gdal_dim(dim, "/")
        assert (
            info.attrs == {}
        ), f"Expected empty attrs when _read_attributes raises, got {info.attrs}"


class TestDimensionInfoDataclass:
    """Tests for DimensionInfo dataclass defaults."""

    def test_defaults(self):
        """Verify optional fields default to None and empty dict."""
        info = DimensionInfo(name="x", full_name="/x", size=10)
        assert info.type is None, "type should default to None"
        assert info.direction is None, "direction should default to None"
        assert (
            info.indexing_variable is None
        ), "indexing_variable should default to None"
        assert info.attrs == {}, "attrs should default to {}"


class TestVariableInfoFromMdArray:
    """Tests for VariableInfo.from_md_array class method."""

    def test_basic_construction(self):
        """Verify all fields extracted from a fully functional mock array."""
        arr = _mock_md_array(
            name="temperature",
            full_name="/temperature",
            shape=(365, 180, 360),
            dim_names=["/time", "/lat", "/lon"],
            unit="K",
        )
        info = VariableInfo.from_md_array(arr, "temperature", "/")
        assert info.name == "temperature", f"Expected 'temperature', got '{info.name}'"
        assert (
            info.full_name == "/temperature"
        ), f"Expected '/temperature', got '{info.full_name}'"
        assert info.dtype == "float32", f"Expected 'float32', got '{info.dtype}'"
        assert info.shape == [
            365,
            180,
            360,
        ], f"Expected [365, 180, 360], got {info.shape}"
        assert info.dimensions == [
            "/time",
            "/lat",
            "/lon",
        ], f"Expected ['/time', '/lat', '/lon'], got {info.dimensions}"
        assert info.unit == "K", f"Expected 'K', got '{info.unit}'"

    def test_get_name_failure_uses_fallback(self):
        """Verify fallback name when GetName raises."""
        arr = _mock_md_array(name="temp")
        arr.GetName.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "fallback_name", "/")
        assert (
            info.name == "fallback_name"
        ), f"Expected 'fallback_name', got '{info.name}'"

    def test_get_full_name_failure_uses_group_prefix(self):
        """Verify full_name fallback uses group_full_name/name."""
        arr = _mock_md_array(name="temp")
        arr.GetFullName.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/root")
        assert (
            info.full_name == "/root/temp"
        ), f"Expected '/root/temp', got '{info.full_name}'"

    def test_get_full_name_failure_root_group(self):
        """Verify full_name fallback with root group."""
        arr = _mock_md_array(name="temp")
        arr.GetFullName.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.full_name == "/temp", f"Expected '/temp', got '{info.full_name}'"

    def test_dtype_failure(self):
        """Verify dtype defaults to 'unknown' when GetDataType raises."""
        arr = _mock_md_array()
        arr.GetDataType.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.dtype == "unknown", f"Expected 'unknown', got '{info.dtype}'"

    def test_shape_failure_uses_dimensions_fallback(self):
        """Verify shape fallback reads sizes from dimensions when GetShape fails."""
        arr = _mock_md_array(shape=(10, 20), dim_names=["/lat", "/lon"])
        arr.GetShape.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.shape == [
            10,
            20,
        ], f"Expected [10, 20] from dim fallback, got {info.shape}"

    def test_shape_and_dimensions_both_fail(self):
        """Verify shape defaults to [] when both GetShape and dimension sizes fail."""
        arr = _mock_md_array(shape=(10, 20))
        arr.GetShape.side_effect = RuntimeError("fail")
        arr.GetDimensions.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert (
            info.dimensions == []
        ), f"Expected empty dimensions, got {info.dimensions}"

    def test_dimension_get_full_name_failure_falls_back_to_get_name(self):
        """Verify dimension name fallback from GetFullName to GetName."""
        arr = _mock_md_array(shape=(10,), dim_names=["/x"])
        dim = arr.GetDimensions()[0]
        dim.GetFullName.side_effect = RuntimeError("fail")
        dim.GetName.return_value = "x"
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert (
            "x" in info.dimensions
        ), f"Expected 'x' in dimensions, got {info.dimensions}"

    def test_get_unit_failure(self):
        """Verify unit defaults to None when GetUnit raises."""
        arr = _mock_md_array()
        arr.GetUnit.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.unit is None, f"Expected None, got '{info.unit}'"

    def test_spatial_ref_captured(self):
        """Verify srs_wkt and srs_projjson are populated from SpatialRef."""
        srs = MagicMock()
        srs.ExportToWkt.return_value = 'GEOGCS["WGS 84"]'
        srs.ExportToJSON.return_value = '{"type":"GeographicCRS"}'
        arr = _mock_md_array(srs=srs)
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert (
            info.srs_wkt == 'GEOGCS["WGS 84"]'
        ), f"Expected WKT string, got '{info.srs_wkt}'"
        assert (
            info.srs_projjson == '{"type":"GeographicCRS"}'
        ), f"Expected PROJJSON, got '{info.srs_projjson}'"

    def test_spatial_ref_failure(self):
        """Verify srs_wkt and srs_projjson are None when GetSpatialRef raises."""
        arr = _mock_md_array()
        arr.GetSpatialRef.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.srs_wkt is None, f"Expected None, got '{info.srs_wkt}'"
        assert info.srs_projjson is None, f"Expected None, got '{info.srs_projjson}'"

    def test_structural_info_captured(self):
        """Verify structural_info is read and converted to {str: str}."""
        arr = _mock_md_array()
        arr.GetStructuralInfo.return_value = {"COMPRESS": "DEFLATE", "LEVEL": "6"}
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.structural_info == {
            "COMPRESS": "DEFLATE",
            "LEVEL": "6",
        }, f"Expected structural info dict, got {info.structural_info}"

    def test_structural_info_failure(self):
        """Verify structural_info is None when GetStructuralInfo raises."""
        arr = _mock_md_array()
        arr.GetStructuralInfo.side_effect = RuntimeError("fail")
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert (
            info.structural_info is None
        ), f"Expected None, got {info.structural_info}"

    def test_block_size_captured(self):
        """Verify block_size is read and converted to list[int]."""
        arr = _mock_md_array()
        arr.GetBlockSize.return_value = [1, 180, 360]
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.block_size == [
            1,
            180,
            360,
        ], f"Expected [1, 180, 360], got {info.block_size}"

    def test_coordinate_variables_captured(self):
        """Verify coordinate variable names are extracted."""
        cv = MagicMock()
        cv.GetFullName.return_value = "/lat"
        arr = _mock_md_array()
        arr.GetCoordinateVariables.return_value = [cv]
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.coordinate_variables == [
            "/lat"
        ], f"Expected ['/lat'], got {info.coordinate_variables}"

    def test_shape_fallback_dimension_size_failure(self):
        """Verify shape is [] when GetShape fails and dimension GetSize also fails."""
        arr = _mock_md_array(shape=(10,))
        arr.GetShape.side_effect = RuntimeError("fail")
        # Make GetDimensions return dims whose GetSize also fails
        bad_dim = MagicMock()
        bad_dim.GetSize.side_effect = RuntimeError("fail")
        arr.GetDimensions.return_value = [bad_dim]
        info = VariableInfo.from_md_array(arr, "temp", "/")
        assert info.shape == [], f"Expected [], got {info.shape}"


class TestVariableInfoDataclass:
    """Tests for VariableInfo dataclass defaults."""

    def test_defaults(self):
        """Verify optional fields default to None and empty containers."""
        info = VariableInfo(
            name="v",
            full_name="/v",
            dtype="float32",
            shape=[10],
            dimensions=["/x"],
        )
        assert info.attributes == {}, "attributes should default to {}"
        assert info.unit is None, "unit should default to None"
        assert info.nodata is None, "nodata should default to None"
        assert info.scale is None, "scale should default to None"
        assert info.offset is None, "offset should default to None"
        assert info.srs_wkt is None, "srs_wkt should default to None"
        assert info.srs_projjson is None, "srs_projjson should default to None"
        assert (
            info.coordinate_variables == []
        ), "coordinate_variables should default to []"
        assert info.structural_info is None, "structural_info should default to None"
        assert info.block_size is None, "block_size should default to None"


class TestStructuralInfoFromDataset:
    """Tests for StructuralInfo.from_dataset class method."""

    def test_basic_construction(self):
        """Verify driver metadata is read and converted."""
        ds = MagicMock()
        driver = MagicMock()
        driver.GetMetadata_Dict.return_value = {
            "DMD_LONGNAME": "Network Common Data Form",
            "DMD_EXTENSIONS": "nc",
        }
        ds.GetDriver.return_value = driver
        info = StructuralInfo.from_dataset(ds, "netCDF")
        assert (
            info.driver_name == "netCDF"
        ), f"Expected 'netCDF', got '{info.driver_name}'"
        assert info.driver_metadata is not None, "driver_metadata should not be None"
        assert (
            info.driver_metadata["DMD_LONGNAME"] == "Network Common Data Form"
        ), f"Expected correct DMD_LONGNAME, got {info.driver_metadata}"

    def test_empty_metadata_dict(self):
        """Verify driver_metadata is None when GetMetadata_Dict returns empty."""
        ds = MagicMock()
        driver = MagicMock()
        driver.GetMetadata_Dict.return_value = {}
        ds.GetDriver.return_value = driver
        info = StructuralInfo.from_dataset(ds, "netCDF")
        assert (
            info.driver_metadata is None
        ), f"Expected None for empty metadata, got {info.driver_metadata}"

    def test_metadata_dict_none(self):
        """Verify driver_metadata is None when GetMetadata_Dict returns None."""
        ds = MagicMock()
        driver = MagicMock()
        driver.GetMetadata_Dict.return_value = None
        ds.GetDriver.return_value = driver
        info = StructuralInfo.from_dataset(ds, "netCDF")
        assert (
            info.driver_metadata is None
        ), f"Expected None for None metadata, got {info.driver_metadata}"

    def test_get_driver_failure(self):
        """Verify driver_metadata is None when GetDriver raises."""
        ds = MagicMock()
        ds.GetDriver.side_effect = RuntimeError("fail")
        info = StructuralInfo.from_dataset(ds, "netCDF")
        assert (
            info.driver_name == "netCDF"
        ), f"driver_name should still be set, got '{info.driver_name}'"
        assert (
            info.driver_metadata is None
        ), f"Expected None on driver failure, got {info.driver_metadata}"


class TestStructuralInfoDataclass:
    """Tests for StructuralInfo dataclass defaults."""

    def test_defaults(self):
        """Verify driver_metadata defaults to None."""
        info = StructuralInfo(driver_name="GTiff")
        assert info.driver_metadata is None, "driver_metadata should default to None"


def _make_metadata(
    dims: dict[str, DimensionInfo] | None = None,
    arrays: dict[str, VariableInfo] | None = None,
    groups: dict[str, GroupInfo] | None = None,
) -> NetCDFMetadata:
    """Create a NetCDFMetadata instance with sensible defaults.

    Args:
        dims: Optional dimensions dict.
        arrays: Optional arrays dict.
        groups: Optional groups dict.

    Returns:
        NetCDFMetadata: A metadata instance.
    """
    if dims is None:
        dims = {
            "/time": DimensionInfo(name="time", full_name="/time", size=365),
            "/lat": DimensionInfo(name="lat", full_name="/lat", size=180),
            "/lon": DimensionInfo(name="lon", full_name="/lon", size=360),
        }
    if arrays is None:
        arrays = {
            "/temperature": VariableInfo(
                name="temperature",
                full_name="/temperature",
                dtype="float32",
                shape=[365, 180, 360],
                dimensions=["/time", "/lat", "/lon"],
            ),
        }
    if groups is None:
        groups = {
            "/": GroupInfo(name="root", full_name="/", variables=["/temperature"]),
        }

    return NetCDFMetadata(
        driver="netCDF",
        root_group="/",
        groups=groups,
        variables=arrays,
        dimensions=dims,
        global_attributes={"Conventions": "CF-1.6"},
        structural=StructuralInfo(driver_name="netCDF"),
        created_with={"library": "GDAL", "version": "3.9.0"},
    )


class TestNetCDFMetadataGetDimension:
    """Tests for NetCDFMetadata.get_dimension method."""

    def test_lookup_by_full_name(self):
        """Verify exact key lookup by full name works."""
        meta = _make_metadata()
        dim = meta.get_dimension("/time")
        assert dim is not None, "Should find '/time' by full name"
        assert dim.name == "time", f"Expected 'time', got '{dim.name}'"
        assert dim.size == 365, f"Expected 365, got {dim.size}"

    def test_lookup_by_short_name(self):
        """Verify fallback to short name matching works."""
        meta = _make_metadata()
        dim = meta.get_dimension("lat")
        assert dim is not None, "Should find 'lat' by short name"
        assert dim.full_name == "/lat", f"Expected '/lat', got '{dim.full_name}'"

    def test_missing_returns_none(self):
        """Verify None is returned for a non-existent dimension."""
        meta = _make_metadata()
        result = meta.get_dimension("nonexistent")
        assert result is None, f"Expected None for missing dimension, got {result}"

    def test_full_name_takes_precedence_over_short_name(self):
        """Verify full name key lookup is tried before short name iteration.

        If a dimension with full_name="/x" and name="y" exists, looking up
        "/x" should find it but looking up "y" should also find it.
        """
        dims = {"/x": DimensionInfo(name="y", full_name="/x", size=5)}
        meta = _make_metadata(dims=dims)
        by_full = meta.get_dimension("/x")
        assert by_full is not None, "Should find by full name '/x'"
        by_short = meta.get_dimension("y")
        assert by_short is not None, "Should find by short name 'y'"

    def test_empty_dimensions_returns_none(self):
        """Verify None is returned when dimensions dict is empty."""
        meta = _make_metadata(dims={})
        result = meta.get_dimension("time")
        assert result is None, f"Expected None, got {result}"


class TestNetCDFMetadataDataclass:
    """Tests for NetCDFMetadata dataclass defaults and fields."""

    def test_optional_defaults(self):
        """Verify open_options_used defaults to None."""
        meta = _make_metadata()
        assert (
            meta.open_options_used is None
        ), "open_options_used should default to None"

    def test_all_fields_accessible(self):
        """Verify all expected fields are accessible."""
        meta = _make_metadata()
        assert meta.driver == "netCDF", f"Expected 'netCDF', got '{meta.driver}'"
        assert meta.root_group == "/", f"Expected '/', got '{meta.root_group}'"
        assert isinstance(meta.groups, dict), f"Expected dict, got {type(meta.groups)}"
        assert isinstance(
            meta.variables, dict
        ), f"Expected dict, got {type(meta.variables)}"
        assert isinstance(
            meta.dimensions, dict
        ), f"Expected dict, got {type(meta.dimensions)}"
        assert isinstance(
            meta.global_attributes, dict
        ), f"Expected dict, got {type(meta.global_attributes)}"
        assert meta.structural is not None, "Expected not None for structural"
        assert isinstance(
            meta.created_with, dict
        ), f"Expected dict, got {type(meta.created_with)}"
