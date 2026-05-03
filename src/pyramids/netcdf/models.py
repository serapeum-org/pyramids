from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from osgeo import gdal

from pyramids.netcdf.utils import (
    AttributeValue,
    _dtype_to_str,
    _export_srs,
    _full_name_with_fallback,
    _get_array_nodata,
    _get_array_scale_offset,
    _get_block_size,
    _get_coord_variable_names,
    _get_group_name,
    _read_attributes,
)


@dataclass(frozen=True)
class GroupInfo:
    """Immutable metadata for a single MDIM group in a NetCDF file.

    All fields are JSON-serializable and use full names
    (e.g. `"/root/subgroup"`) for stable cross-references
    between groups, variables, and dimensions.

    Note:
        `frozen=True` prevents field reassignment but container
        fields (`attributes`, `children`, `variables`) are
        technically mutable. Treat all contents as **read-only**
        after construction.

    Args:
        name: Short name of the group (e.g. `"root"`).
        full_name: Fully qualified name including the path
            (e.g. `"/"` for the root group).
        attributes: Key-value mapping of group-level
            attributes read from the NetCDF file.
        children: Full names of direct child groups.
        variables: Full names of variables belonging to this
            group.

    Examples:
        - Create a GroupInfo for the root group:
            ```python
            >>> from pyramids.netcdf.models import GroupInfo
            >>> info = GroupInfo(
            ...     name="root",
            ...     full_name="/",
            ...     attributes={"Conventions": "CF-1.6"},
            ...     children=["/forecast"],
            ...     variables=["/temperature", "/pressure"],
            ... )
            >>> info.name
            'root'
            >>> info.full_name
            '/'
            >>> info.attributes
            {'Conventions': 'CF-1.6'}

            ```

        - Create a minimal GroupInfo with defaults:
            ```python
            >>> from pyramids.netcdf.models import GroupInfo
            >>> info = GroupInfo(name="leaf", full_name="/leaf")
            >>> info.children
            []
            >>> info.variables
            []
            >>> info.attributes
            {}

            ```

    See Also:
        VariableInfo: Metadata for individual variables within a group.
        StructuralInfo: Driver-level metadata for the dataset.
    """

    name: str
    full_name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)

    @classmethod
    def from_group(
        cls,
        group: gdal.Group,
        *,
        variables: list[str],
        children: list[str],
        attributes: dict[str, AttributeValue] | None = None,
    ) -> GroupInfo:
        """Build a GroupInfo from a live GDAL Group object.

        Extracts the group name, full name, and attributes
        from the GDAL `Group` handle and bundles them into
        an immutable `GroupInfo` instance.

        Args:
            group: The GDAL multidimensional group object.
            variables: Full names of variables that belong to
                this group (e.g. `["/temperature"]`).
            children: Full names of direct child groups
                (e.g. `["/forecast"]`).
            attributes: Pre-read attributes dictionary. If
                `None`, attributes are read directly from
                the group.

        Returns:
            GroupInfo: Constructed metadata instance for the
                group.

        See Also:
            VariableInfo.from_md_array: Analogous factory for
                variable metadata.
        """
        name = _get_group_name(group)
        full_name = _full_name_with_fallback(group, name)

        if attributes is None:
            attributes = _read_attributes(group)

        return cls(
            name=name,
            full_name=full_name,
            attributes=attributes or {},
            children=list(children) if children else [],
            variables=list(variables) if variables else [],
        )


@dataclass(frozen=True)
class DimensionInfo:
    """Immutable metadata for a single NetCDF dimension.

    Captures the name, size, type, direction, and any
    attributes inherited from the dimension's indexing
    variable (e.g. `units` and `calendar` for a time
    dimension).

    Note:
        `frozen=True` prevents field reassignment but the
        `attrs` dict is technically mutable. Treat its
        contents as **read-only** after construction.

    Args:
        name: Short name of the dimension (e.g. `"time"`).
        full_name: Fully qualified path name
            (e.g. `"/time"`).
        size: Number of elements along this dimension.
        type: Dimension type string reported by GDAL, such
            as `"HORIZONTAL_X"` or `"TEMPORAL"`.
            `None` if unavailable.
        direction: Direction string (e.g. `"EAST"`,
            `"NORTH"`). `None` if unavailable.
        indexing_variable: Full name (or short name) of the
            variable that indexes this dimension.
            `None` if no indexing variable exists.
        attrs: Attributes read from the indexing variable
            (e.g. `{"units": "days since 1970-01-01"}`).

    Examples:
        - Create a spatial dimension:
            ```python
            >>> from pyramids.netcdf.models import DimensionInfo
            >>> dim = DimensionInfo(
            ...     name="lon",
            ...     full_name="/lon",
            ...     size=720,
            ...     type="HORIZONTAL_X",
            ...     direction="EAST",
            ...     indexing_variable="/lon",
            ... )
            >>> dim.name
            'lon'
            >>> dim.size
            720

            ```

        - Create a time dimension with calendar attributes:
            ```python
            >>> from pyramids.netcdf.models import DimensionInfo
            >>> dim = DimensionInfo(
            ...     name="time",
            ...     full_name="/time",
            ...     size=365,
            ...     type="TEMPORAL",
            ...     attrs={
            ...         "units": "days since 1970-01-01",
            ...         "calendar": "standard",
            ...     },
            ... )
            >>> dim.attrs["calendar"]
            'standard'

            ```

    See Also:
        VariableInfo: Metadata for variables that reference these
            dimensions.
        NetCDFMetadata.get_dimension: Look up a dimension by
            name.
    """

    name: str
    full_name: str
    size: int
    type: str | None = None
    direction: str | None = None
    indexing_variable: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_gdal_dim(cls, d: gdal.Dimension, group_full_name: str) -> DimensionInfo:
        """Build a DimensionInfo from a live GDAL Dimension.

        Reads name, size, type, direction, and indexing
        variable information from the GDAL `Dimension`
        handle. Attributes are extracted from the indexing
        variable when one exists (e.g. `units` and
        `calendar` on a time coordinate).

        Args:
            d: The GDAL multidimensional `Dimension` object.
            group_full_name: Full name of the parent group
                (used as a fallback prefix when the dimension
                does not report its own full name).

        Returns:
            DimensionInfo: Constructed metadata instance for
                the dimension.

        See Also:
            GroupInfo.from_group: Analogous factory for group
                metadata.
        """
        try:
            dim_name = d.GetName()
        except Exception:
            dim_name = ""

        try:
            dim_full_name = d.GetFullName()
        except Exception:
            dim_full_name = (
                f"{group_full_name}/{dim_name}"
                if group_full_name!= "/"
                else f"/{dim_name}"
            )

        try:
            dim_size = int(d.GetSize())
        except Exception:
            dim_size = 0

        try:
            dtype = str(d.GetType())  # type: ignore[attr-defined]
        except Exception:
            dtype = None

        try:
            dim_dir = str(d.GetDirection())  # type: ignore[attr-defined]
        except Exception:
            dim_dir = None

        try:
            iv = d.GetIndexingVariable()
            if iv is not None:
                ivname = (
                    iv.GetFullName() if hasattr(iv, "GetFullName") else iv.GetName()
                )
            else:
                ivname = None
        except Exception:
            iv = None
            ivname = None

        # Read attributes from the indexing variable (e.g., "units", "calendar")
        try:
            iv_attrs = _read_attributes(iv) if iv is not None else {}
        except Exception:
            iv_attrs = {}

        return cls(
            name=dim_name,
            full_name=dim_full_name,
            size=dim_size,
            type=dtype,
            direction=dim_dir,
            indexing_variable=ivname,
            attrs=iv_attrs,
        )


@dataclass(frozen=True)
class VariableInfo:
    """Immutable metadata for a single MDIM array (variable).

    Stores everything needed to describe a NetCDF variable
    without holding a reference to the live GDAL object:
    data type, shape, dimension links, CF attributes
    (scale, offset, nodata), spatial reference, and
    chunking information.

    Note:
        `frozen=True` prevents field reassignment but container
        fields (`attributes`, `dimensions`, etc.) are
        technically mutable. Treat all contents as **read-only**
        after construction.

    Args:
        name: Short variable name (e.g. `"temperature"`).
        full_name: Fully qualified name including the group
            path (e.g. `"/temperature"`).
        dtype: NumPy-compatible data type string
            (e.g. `"float32"`, `"int16"`).
        shape: Size along each dimension, in dimension
            order.
        dimensions: Full names of the dimensions this array
            spans, matching the order of `shape`.
        attributes: Key-value mapping of variable-level
            NetCDF attributes.
        unit: Physical unit string from the `units`
            attribute (e.g. `"K"`, `"mm/day"`).
        nodata: No-data / fill value. `None` when no
            fill value is defined.
        scale: CF `scale_factor`. `None` if not set.
        offset: CF `add_offset`. `None` if not set.
        srs_wkt: Spatial reference as WKT string.
            `None` if no SRS is attached.
        srs_projjson: Spatial reference as PROJJSON string.
            `None` if no SRS is attached.
        coordinate_variables: Full names of coordinate
            variables associated with this array.
        structural_info: Driver-specific structural info
            dictionary. `None` if unavailable.
        block_size: Chunk sizes along each dimension.
            `None` if the array is not chunked.

    Examples:
        - Create metadata for a 2-D temperature variable:
            ```python
            >>> from pyramids.netcdf.models import VariableInfo
            >>> arr = VariableInfo(
            ...     name="temperature",
            ...     full_name="/temperature",
            ...     dtype="float32",
            ...     shape=[180, 360],
            ...     dimensions=["/lat", "/lon"],
            ...     unit="K",
            ...     nodata=-9999.0,
            ... )
            >>> arr.dtype
            'float32'
            >>> arr.shape
            [180, 360]

            ```

        - Create metadata for a 3-D variable with scale
          and offset:
            ```python
            >>> from pyramids.netcdf.models import VariableInfo
            >>> arr = VariableInfo(
            ...     name="precip",
            ...     full_name="/precip",
            ...     dtype="int16",
            ...     shape=[365, 180, 360],
            ...     dimensions=["/time", "/lat", "/lon"],
            ...     scale=0.01,
            ...     offset=0.0,
            ...     block_size=[1, 180, 360],
            ... )
            >>> arr.scale
            0.01
            >>> arr.block_size
            [1, 180, 360]

            ```

    See Also:
        DimensionInfo: Metadata for the dimensions that an
            array spans.
        GroupInfo: Metadata for the group containing this
            array.
    """

    name: str
    full_name: str
    dtype: str
    shape: list[int]
    dimensions: list[str]
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    unit: str | None = None
    nodata: int | float | str | None = None
    scale: float | None = None
    offset: float | None = None
    srs_wkt: str | None = None
    srs_projjson: str | None = None
    coordinate_variables: list[str] = field(default_factory=list)
    structural_info: dict[str, str] | None = None
    block_size: list[int] | None = None

    @classmethod
    def from_md_array(
        cls, md_arr: gdal.MDArray, md_arr_name: str, group_full_name: str
    ) -> VariableInfo:
        """Build a VariableInfo from a live GDAL MDArray.

        Extracts name, data type, shape, dimension links,
        attributes, CF conventions (scale, offset, nodata),
        spatial reference, structural info, and chunk sizes
        from the GDAL `MDArray` handle.

        Args:
            md_arr: The GDAL multidimensional array object.
            md_arr_name: Fallback short name used when the
                array's own `GetName()` call fails.
            group_full_name: Full name of the parent group
                (used as a fallback prefix for constructing
                the array's full name).

        Returns:
            VariableInfo: Constructed metadata instance for the
                array.

        See Also:
            DimensionInfo.from_gdal_dim: Analogous factory
                for dimension metadata.
        """
        try:
            md_arr_name = md_arr.GetName()
        except Exception:
            md_arr_name = md_arr_name

        try:
            md_arr_full_name = md_arr.GetFullName()
        except Exception:
            md_arr_full_name = (
                f"{group_full_name}/{md_arr_name}"
                if group_full_name != "/"
                else f"/{md_arr_name}"
            )

        # dtype
        try:
            dt = _dtype_to_str(md_arr.GetDataType())
        except Exception:
            dt = "unknown"

        # shape
        try:
            shape = list(md_arr.GetShape())
        except Exception:
            try:
                shape = [int(d.GetSize()) for d in (md_arr.GetDimensions() or [])]
            except Exception:
                shape = []

        # dimension names
        try:
            dims2 = md_arr.GetDimensions() or []
            dim_names: list[str] = []
            for d in dims2:
                try:
                    dn = d.GetFullName()
                except Exception:
                    dn = d.GetName()
                dim_names.append(dn)
        except Exception:
            dim_names = []

        a_attrs = _read_attributes(md_arr)
        try:
            unit = md_arr.GetUnit()
        except Exception:
            unit = None
        nodata = _get_array_nodata(md_arr, a_attrs)
        scale, offset = _get_array_scale_offset(md_arr, a_attrs)
        try:
            srs = md_arr.GetSpatialRef()
        except Exception:
            srs = None

        wkt, proj_json = _export_srs(srs)
        try:
            sinfo = md_arr.GetStructuralInfo()
            if sinfo is not None:
                sinfo = {str(k): str(v) for k, v in dict(sinfo).items()}
        except Exception:
            sinfo = None
        block_size = _get_block_size(md_arr)
        coord_vars = _get_coord_variable_names(md_arr)

        return cls(
            name=md_arr_name,
            full_name=md_arr_full_name,
            dtype=dt,
            shape=[int(x) for x in (shape or [])],
            dimensions=dim_names,
            attributes=a_attrs,
            unit=unit,
            nodata=nodata,
            scale=scale,
            offset=offset,
            srs_wkt=wkt,
            srs_projjson=proj_json,
            coordinate_variables=coord_vars,
            structural_info=sinfo,
            block_size=block_size,
        )


@dataclass(frozen=True)
class StructuralInfo:
    """Immutable driver-level metadata for a GDAL dataset.

    Captures the driver name and its associated metadata
    dictionary, providing context about how the file was
    opened and which GDAL driver capabilities are
    available.

    Args:
        driver_name: Short name of the GDAL driver
            (e.g. `"netCDF"`, `"GTiff"`).
        driver_metadata: Key-value metadata reported by the
            driver (e.g. creation options, version info).
            `None` if the driver exposes no metadata.

    Examples:
        - Create structural info for a NetCDF driver:
            ```python
            >>> from pyramids.netcdf.models import StructuralInfo
            >>> info = StructuralInfo(
            ...     driver_name="netCDF",
            ...     driver_metadata={
            ...         "DMD_LONGNAME": "Network Common Data Form",
            ...     },
            ... )
            >>> info.driver_name
            'netCDF'

            ```

        - Create structural info without driver metadata:
            ```python
            >>> from pyramids.netcdf.models import StructuralInfo
            >>> info = StructuralInfo(driver_name="netCDF")
            >>> info.driver_metadata is None
            True

            ```

    See Also:
        NetCDFMetadata: Top-level model that includes a
            `StructuralInfo` instance.
    """

    driver_name: str
    driver_metadata: dict[str, str] | None = None

    @classmethod
    def from_dataset(cls, dataset: gdal.Dataset, driver_name: str) -> StructuralInfo:
        """Build a StructuralInfo from a live GDAL Dataset.

        Reads the driver metadata dictionary from the
        dataset's driver handle.

        Args:
            dataset: An open GDAL `Dataset` object.
            driver_name: Short name of the GDAL driver
                (e.g. `"netCDF"`).

        Returns:
            StructuralInfo: Constructed metadata instance
                containing the driver name and its metadata
                dictionary.

        See Also:
            NetCDFMetadata: The top-level model that
                aggregates structural info with groups,
                variables, and dimensions.
        """
        try:
            dmd = dataset.GetDriver().GetMetadata_Dict()
            dmd = {str(k): str(v) for k, v in dmd.items()} if dmd else None
        except Exception:
            dmd = None
        return cls(driver_name=driver_name, driver_metadata=dmd)


MAX_DISPLAY_VARIABLES = 10


@dataclass
class CFInfo:
    """CF convention metadata derived by cross-referencing variables.

    Computed as a post-processing step in `MetadataBuilder.build()`
    after all variables, dimensions, and attributes are collected.

    Args:
        cf_version: CF version string parsed from `Conventions`
            (e.g. `"1.8"`). None if not declared.
        conventions: Parsed `Conventions` attribute as
            `{name: version}` dict.
        classifications: Per-variable CF role classification as
            `{var_name: role}` where role is one of `"data"`,
            `"coordinate"`, `"grid_mapping"`, `"bounds"`,
            `"cell_measure"`, `"ancillary"`, `"mesh_topology"`,
            `"connectivity"`, `"auxiliary_coordinate"`.
        grid_mappings: Grid mapping variable attributes as
            `{var_name: {attr: value}}`.
        bounds_map: Bounds associations as
            `{bounds_var_name: parent_coord_name}`.
        data_variable_names: Variable names classified as `"data"`.
    """

    cf_version: str | None = None
    conventions: dict[str, str] = field(default_factory=dict)
    classifications: dict[str, str] = field(default_factory=dict)
    grid_mappings: dict[str, dict[str, Any]] = field(default_factory=dict)
    bounds_map: dict[str, str] = field(default_factory=dict)
    data_variable_names: list[str] = field(default_factory=list)


@dataclass
class NetCDFMetadata:
    """Top-level metadata model for a NetCDF MDIM dataset.

    Aggregates all structural and scientific metadata
    extracted from a NetCDF file opened through the GDAL
    multidimensional API: groups, variables,
    dimensions, global attributes, and driver information.

    This is the single object returned by the metadata
    extraction pipeline and is intended as a complete,
    JSON-serializable snapshot of the file's structure.

    Note:
        Dictionary keys for `groups`, `variables`, and
        `dimensions` use **short names** with the leading
        `/` stripped (e.g. `"time"` not `"/time"`).
        The root group keeps `"/"` as its key. The
        `full_name` attribute on each object retains the
        original GDAL path (e.g. `"/time"`). Use
        `get_dimension(name)` for lookups that accept
        both forms.

    Args:
        driver: Short name of the GDAL driver used to
            open the file (e.g. `"netCDF"`).
        root_group: Full name of the root group
            (typically `"/"`). `None` when the file
            has no group hierarchy.
        groups: Mapping keyed by short name (or `"/"`
            for root) to `GroupInfo` metadata.
        variables: Mapping keyed by short name to
            `VariableInfo` metadata.
        dimensions: Mapping keyed by short name to
            `DimensionInfo` metadata.
        global_attributes: Key-value mapping of root-level
            NetCDF attributes (e.g. `Conventions`,
            `history`).
        structural: Driver-level metadata, or `None`
            when unavailable.
        created_with: Version information for the tools
            used to extract metadata (e.g.
            `{"gdal": "3.9.0"}`).
        open_options_used: GDAL open options that were
            passed when opening the file. `None` when
            no special options were used.

    Examples:
        - Create a minimal NetCDFMetadata instance:
            ```python
            >>> from pyramids.netcdf.models import (
            ...     NetCDFMetadata,
            ...     GroupInfo,
            ...     VariableInfo,
            ...     DimensionInfo,
            ... )
            >>> dim = DimensionInfo(
            ...     name="time",
            ...     full_name="/time",
            ...     size=12,
            ... )
            >>> arr = VariableInfo(
            ...     name="temp",
            ...     full_name="/temp",
            ...     dtype="float32",
            ...     shape=[12],
            ...     dimensions=["/time"],
            ... )
            >>> grp = GroupInfo(
            ...     name="root",
            ...     full_name="/",
            ...     variables=["/temp"],
            ... )
            >>> meta = NetCDFMetadata(
            ...     driver="netCDF",
            ...     root_group="/",
            ...     groups={"/": grp},
            ...     variables={"/temp": arr},
            ...     dimensions={"/time": dim},
            ...     global_attributes={"Conventions": "CF-1.6"},
            ...     structural=None,
            ...     created_with={"gdal": "3.9.0"},
            ... )
            >>> meta.driver
            'netCDF'
            >>> list(meta.dimensions.keys())
            ['/time']
            >>> meta.get_dimension("time").size
            12

            ```

    See Also:
        GroupInfo: Metadata for a single group.
        VariableInfo: Metadata for a single array.
        DimensionInfo: Metadata for a single dimension.
        StructuralInfo: Driver-level metadata.
    """

    driver: str
    root_group: str | None
    groups: dict[str, GroupInfo]
    variables: dict[str, VariableInfo]
    dimensions: dict[str, DimensionInfo]
    global_attributes: dict[str, AttributeValue]
    structural: StructuralInfo | None
    created_with: dict[str, str]
    open_options_used: dict[str, str] | None = None
    cf: CFInfo | None = None

    def __str__(self) -> str:
        """Human-readable summary of the NetCDF structure."""
        dim_lines = []
        for dim in self.dimensions.values():
            dim_lines.append(f"    {dim.name:20s} size={dim.size}")
        dims_str = "\n".join(dim_lines) if dim_lines else "    (none)"

        var_lines = []
        max_display = MAX_DISPLAY_VARIABLES
        arr_list = list(self.variables.values())
        for arr in arr_list[:max_display]:
            dtype_str = arr.dtype if len(arr.dtype) < 20 else "unknown"
            var_lines.append(f"    {arr.name:20s} {dtype_str:10s} {list(arr.shape)}")
        if len(arr_list) > max_display:
            var_lines.append(f"    ... and {len(arr_list) - max_display} more")
        vars_str = "\n".join(var_lines) if var_lines else "    (none)"

        attr_keys = list(self.global_attributes.keys())
        attrs_str = ", ".join(attr_keys[:5])
        if len(attr_keys) > 5:
            attrs_str += f", ... ({len(attr_keys)} total)"

        lines = [
            "NetCDFMetadata",
            f"  Driver: {self.driver}",
            f"  Root group: {self.root_group}",
            f"  Dimensions ({len(self.dimensions)}):",
            dims_str,
            f"  Variables ({len(self.variables)}):",
            vars_str,
            f"  Global attributes: {attrs_str}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Technical representation with counts and driver info."""
        return (
            f"NetCDFMetadata("
            f"driver={self.driver!r}, "
            f"groups={len(self.groups)}, "
            f"variables={len(self.variables)}, "
            f"dimensions={len(self.dimensions)}, "
            f"attributes={len(self.global_attributes)}"
            f")"
        )

    def get_dimension(self, name: str) -> DimensionInfo | None:
        """Look up a dimension by short name or full name.

        Tries an exact key match against the `dimensions`
        dictionary first (keys are full names with the leading
        `/` stripped, e.g. `"/time"` -> key `"time"`,
        `"/group/time"` -> key `"group/time"`), then falls
        back to matching by the dimension's short `name`
        attribute.

        Args:
            name: Dimension short name (e.g. `"time"`)
                or full name (e.g. `"/time"`).

        Returns:
            DimensionInfo | None: The matching dimension
                metadata, or `None` if no dimension
                matches.

        Examples:
            - Look up by short name:
                ```python
                >>> from pyramids.netcdf.models import (
                ...     NetCDFMetadata,
                ...     DimensionInfo,
                ... )
                >>> dim = DimensionInfo(
                ...     name="time",
                ...     full_name="/time",
                ...     size=365,
                ... )
                >>> meta = NetCDFMetadata(
                ...     driver="netCDF",
                ...     root_group="/",
                ...     groups={},
                ...     variables={},
                ...     dimensions={"/time": dim},
                ...     global_attributes={},
                ...     structural=None,
                ...     created_with={},
                ... )
                >>> meta.get_dimension("time").size
                365

                ```

            - Look up by full name:
                ```python
                >>> from pyramids.netcdf.models import (
                ...     NetCDFMetadata,
                ...     DimensionInfo,
                ... )
                >>> dim = DimensionInfo(
                ...     name="time",
                ...     full_name="/time",
                ...     size=365,
                ... )
                >>> meta = NetCDFMetadata(
                ...     driver="netCDF",
                ...     root_group="/",
                ...     groups={},
                ...     variables={},
                ...     dimensions={"/time": dim},
                ...     global_attributes={},
                ...     structural=None,
                ...     created_with={},
                ... )
                >>> meta.get_dimension("/time").size
                365

                ```

            - Return None for a missing dimension:
                ```python
                >>> from pyramids.netcdf.models import (
                ...     NetCDFMetadata,
                ...     DimensionInfo,
                ... )
                >>> meta = NetCDFMetadata(
                ...     driver="netCDF",
                ...     root_group="/",
                ...     groups={},
                ...     variables={},
                ...     dimensions={},
                ...     global_attributes={},
                ...     structural=None,
                ...     created_with={},
                ... )
                >>> meta.get_dimension("missing") is None
                True

                ```

        See Also:
            NetCDFMetadata.dimensions: The full dimensions
                dictionary keyed by full name.
        """
        # Try full name first (exact key lookup)
        if name in self.dimensions:
            return self.dimensions[name]
        # Try matching by short name
        for dim in self.dimensions.values():
            if dim.name == name:
                return dim
        return None
