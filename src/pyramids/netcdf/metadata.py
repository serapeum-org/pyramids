from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from osgeo import gdal

# Module-level logger so swallowed GDAL traversal errors surface at
# DEBUG instead of disappearing entirely. Every ``except RuntimeError``
# in this file logs through it. Real failures during MDIM traversal
# (corrupted file, missing driver, deleted variable mid-walk) are now
# observable; only the documented fallback path is silent at INFO.
logger = logging.getLogger(__name__)

from pyramids.netcdf.cf import classify_variables, parse_conventions
from pyramids.netcdf.dimensions import MetaData as SharedMetaData
from pyramids.netcdf.models import (
    CFInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
    VariableInfo,
)
from pyramids.netcdf.utils import (
    _get_driver_name,
    _get_root_group,
    _read_attributes,
    _safe_array_names,
    _safe_group_names,
)


class MetadataBuilder:
    """Construct a `NetCDFMetadata` from a GDAL dataset.

    `MetadataBuilder` encapsulates the logic for extracting
    multidimensional metadata from a GDAL `Dataset` opened with
    the NetCDF driver. It delegates group/array/dimension traversal
    to `GroupTraverser` and assembles the result into a single
    `NetCDFMetadata` dataclass.

    Public module functions (`get_metadata`, `to_dict`, etc.)
    delegate to this builder to preserve API compatibility.

    Args:
        src: An already-opened `gdal.Dataset` pointing to a
            NetCDF file.
        open_options: Optional dictionary of GDAL open-options
            that were used when opening the file. Stored as
            informational metadata only.

    Examples:
        >>> import pyramids.netcdf.metadata as meta  # doctest: +SKIP
        >>> from osgeo import gdal  # doctest: +SKIP
        >>> ds = gdal.OpenEx(  # doctest: +SKIP
        ...     "precip.nc", gdal.OF_MULTIDIM_RASTER
        ... )
        >>> builder = meta.MetadataBuilder(ds)  # doctest: +SKIP
        >>> md = builder.build()  # doctest: +SKIP
        >>> md.driver  # doctest: +SKIP
        'netCDF'
    """

    def __init__(
        self,
        src: gdal.Dataset,
        open_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MetadataBuilder.

        Args:
            src: An already-opened `gdal.Dataset` pointing
                to a NetCDF file. Must not be `None`.
            open_options: Optional dictionary of GDAL
                open-options recorded for provenance.
        """
        self.gdal_dataset = src
        self.open_options = open_options or None

    def build(self) -> NetCDFMetadata:
        """Build and return the `NetCDFMetadata` for the dataset.

        Reads the driver name, root group, groups, variables,
        dimensions, global attributes, and structural information
        from the underlying GDAL dataset and assembles them into
        a `NetCDFMetadata` instance.

        Returns:
            NetCDFMetadata: Fully populated metadata for the
                NetCDF file.

        Raises:
            Exception: Propagates any unhandled GDAL errors
                encountered while reading the dataset.

        Examples:
            >>> import pyramids.netcdf.metadata as meta  # doctest: +SKIP
            >>> from osgeo import gdal  # doctest: +SKIP
            >>> ds = gdal.OpenEx(  # doctest: +SKIP
            ...     "precip.nc", gdal.OF_MULTIDIM_RASTER
            ... )
            >>> builder = meta.MetadataBuilder(ds)  # doctest: +SKIP
            >>> md = builder.build()  # doctest: +SKIP
            >>> len(md.variables) > 0  # doctest: +SKIP
            True
        """
        ds = self.gdal_dataset

        # Driver name and root group
        driver_name = _get_driver_name(ds)
        root_group = _get_root_group(ds)

        groups_map: dict[str, GroupInfo] = {}
        variables_map: dict[str, VariableInfo] = {}
        dimensions_map: dict[str, DimensionInfo] = {}

        if root_group is not None:
            traverser = GroupTraverser(groups_map, variables_map, dimensions_map)
            traverser.walk(root_group)

            try:
                root_name = root_group.GetFullName()
            except RuntimeError as exc:
                logger.debug("root_group.GetFullName() failed: %s", exc)
                root_name = "/"
            global_attrs = _read_attributes(root_group)
        else:
            root_name = None
            # Fallback: use classic flattened metadata exposed by MetaData
            try:
                md = SharedMetaData.from_metadata(ds.GetMetadata())
                raw = md.to_metadata() if hasattr(md, "to_metadata") else {}
                global_attrs = {
                    str(k): str(v) for k, v in raw.items()  # type: ignore[arg-type]
                }
            except RuntimeError as exc:
                logger.debug(
                    "SharedMetaData.from_metadata fallback failed: %s", exc
                )
                global_attrs = {}

        structural_info = StructuralInfo.from_dataset(ds, driver_name)
        created_with = {
            "library": "GDAL",
            "version": getattr(gdal, "__version__", "unknown"),
        }

        cf_info = self._build_cf_info(variables_map, dimensions_map, global_attrs)

        return NetCDFMetadata(
            driver=driver_name,
            root_group=root_name,
            groups=groups_map,
            variables=variables_map,
            dimensions=dimensions_map,
            global_attributes=global_attrs,
            structural=structural_info,
            created_with=created_with,
            open_options_used=self.open_options,
            cf=cf_info,
        )

    @staticmethod
    def _build_cf_info(
        variables: dict[str, VariableInfo],
        dimensions: dict[str, DimensionInfo],
        global_attrs: dict[str, Any],
    ) -> CFInfo:
        """Post-process variables to extract CF semantics.

        Args:
            variables: All variables from metadata traversal.
            dimensions: All dimensions from metadata traversal.
            global_attrs: Root-level attributes.

        Returns:
            CFInfo with classifications, grid_mappings, bounds_map.
        """

        conventions = parse_conventions(global_attrs.get("Conventions"))
        classifications = classify_variables(variables, dimensions)

        grid_mappings: dict[str, dict[str, Any]] = {}
        for name, role in classifications.items():
            if role == "grid_mapping" and name in variables:
                grid_mappings[name] = dict(variables[name].attributes)

        bounds_map: dict[str, str] = {}
        for var in variables.values():
            bounds_ref = var.attributes.get("bounds")
            if isinstance(bounds_ref, str):
                bounds_map[bounds_ref] = var.name

        data_vars = [name for name, role in classifications.items() if role == "data"]

        return CFInfo(
            cf_version=conventions.get("CF"),
            conventions=conventions,
            classifications=classifications,
            grid_mappings=grid_mappings,
            bounds_map=bounds_map,
            data_variable_names=data_vars,
        )


class GroupTraverser:
    """Iterative BFS traverser for MDIM groups, variables, and dimensions.

    Performs a breadth-first walk over the GDAL multidimensional
    group hierarchy starting from a root `gdal.Group`. At each
    group it collects dimensions, variables, and child-group
    references, storing them in the caller-provided dictionaries.

    A `collections.deque` is used instead of recursion to avoid
    stack overflow on deeply nested files. Group and variable names
    are sorted for deterministic output ordering.

    Args:
        groups: Mutable dictionary that will be populated with
            `GroupInfo` instances keyed by short name.
        variables: Mutable dictionary that will be populated with
            `VariableInfo` instances keyed by short name.
        dimensions: Mutable dictionary that will be populated
            with `DimensionInfo` instances keyed by full name.

    Examples:
        >>> from collections import deque  # doctest: +SKIP
        >>> from osgeo import gdal  # doctest: +SKIP
        >>> ds = gdal.OpenEx(  # doctest: +SKIP
        ...     "precip.nc", gdal.OF_MULTIDIM_RASTER
        ... )
        >>> root = ds.GetRootGroup()  # doctest: +SKIP
        >>> groups, variables, dims = {}, {}, {}  # doctest: +SKIP
        >>> t = GroupTraverser(  # doctest: +SKIP
        ...     groups, variables, dims
        ... )
        >>> t.walk(root)  # doctest: +SKIP
        >>> list(groups.keys())  # doctest: +SKIP
        ['/']

    See Also:
        MetadataBuilder: High-level builder that uses this
            traverser internally.
    """

    def __init__(
        self,
        groups: dict[str, GroupInfo],
        variables: dict[str, VariableInfo],
        dimensions: dict[str, DimensionInfo],
    ) -> None:
        """Initialize GroupTraverser with output dictionaries.

        Args:
            groups: Dictionary to populate with `GroupInfo`
                objects. Keys are group full names (e.g. `"/"`).
            variables: Dictionary to populate with `VariableInfo`
                objects. Keys are variable full names.
            dimensions: Dictionary to populate with
                `DimensionInfo` objects. Keys are dimension
                full names.
        """
        self.groups = groups
        self.variables = variables
        self.dimensions = dimensions

    def _collect_dimensions(self, group: gdal.Group, group_full_name: str) -> None:
        """Collect all dimensions from *group* into `self.dimensions`.

        Dimensions are sorted by name for deterministic ordering.
        Any GDAL errors when reading individual dimensions are
        silently caught so the traversal can continue.

        Args:
            group: The GDAL group whose dimensions to read.
            group_full_name: Full path of the group, used as a
                fallback when the dimension cannot report its own
                full name.
        """

        try:
            dims = group.GetDimensions() or []
        except RuntimeError as exc:
            logger.debug(
                "group.GetDimensions() failed for %r: %s", group_full_name, exc
            )
            dims = []

        # Sort by name for determinism
        def _dim_name(d: Any) -> str:
            try:
                result = str(d.GetName())
            except RuntimeError as exc:
                logger.debug("dimension.GetName() failed: %s", exc)
                result = ""
            return result

        dims_sorted = sorted(dims, key=_dim_name)

        for d in dims_sorted:
            dim = DimensionInfo.from_gdal_dim(d, group_full_name)
            self.dimensions[dim.full_name.lstrip("/")] = dim

    def _collect_arrays(self, group: gdal.Group, group_full_name: str) -> list[str]:
        """Collect all variables from *group* into `self.variables`.

        Each variable is opened via `group.OpenMDArray` and
        converted to a `VariableInfo` dataclass. Variables that
        cannot be opened are silently skipped.

        Args:
            group: The GDAL group whose variables to read.
            group_full_name: Full path of the parent group,
                used as fallback context for full-name
                resolution.

        Returns:
            list[str]: Full names of the variables that were
                successfully collected.
        """
        variable_full_names: list[str] = []
        for md_arr_name in _safe_array_names(group):
            try:
                md_arr = group.OpenMDArray(md_arr_name)
            except RuntimeError as exc:
                logger.debug(
                    "group.OpenMDArray(%r) failed in %r: %s",
                    md_arr_name,
                    group_full_name,
                    exc,
                )
                md_arr = None

            if md_arr is None:
                continue

            variable_info = VariableInfo.from_md_array(
                md_arr, md_arr_name, group_full_name
            )

            key = variable_info.full_name.lstrip("/")
            self.variables[key] = variable_info
            variable_full_names.append(variable_info.full_name)
        return variable_full_names

    def walk(self, root: gdal.Group) -> None:
        """Traverse the group tree starting from *root* (BFS).

        Visits every reachable group, collecting its dimensions,
        variables, child references, and attributes. All results are
        written into the dictionaries supplied at construction
        time.

        Args:
            root: The root `gdal.Group` returned by
                `gdal.Dataset.GetRootGroup()`.

        Examples:
            >>> from osgeo import gdal  # doctest: +SKIP
            >>> ds = gdal.OpenEx(  # doctest: +SKIP
            ...     "precip.nc", gdal.OF_MULTIDIM_RASTER
            ... )
            >>> root = ds.GetRootGroup()  # doctest: +SKIP
            >>> groups, variables, dims = {}, {}, {}  # doctest: +SKIP
            >>> traverser = GroupTraverser(  # doctest: +SKIP
            ...     groups, variables, dims
            ... )
            >>> traverser.walk(root)  # doctest: +SKIP
            >>> "/" in groups  # doctest: +SKIP
            True
        """
        q = deque([root])

        while q:
            group = q.popleft()

            # Compute group identity (name/full_name) via GroupInfo for separation of concerns
            base_group = GroupInfo.from_group(
                group, variables=[], children=[], attributes={}
            )
            group_full_name = base_group.full_name

            # Dimensions and variables for this group
            self._collect_dimensions(group, group_full_name)
            group_variables = self._collect_arrays(group, group_full_name)

            # Children
            children_full: list[str] = []
            for cn in _safe_group_names(group):
                try:
                    current_group = group.OpenGroup(cn)
                except RuntimeError as exc:
                    logger.debug(
                        "group.OpenGroup(%r) failed in %r: %s",
                        cn,
                        group_full_name,
                        exc,
                    )
                    current_group = None

                if current_group is None:
                    continue

                # Delegate child full-name resolution to GroupInfo
                try:
                    child_info = GroupInfo.from_group(
                        current_group, variables=[], children=[], attributes={}
                    )
                    current_group_full_name = child_info.full_name
                except RuntimeError as exc:
                    logger.debug(
                        "GroupInfo.from_group(%r) failed in %r: %s; "
                        "falling back to path concatenation",
                        cn,
                        group_full_name,
                        exc,
                    )
                    current_group_full_name = (
                        f"{group_full_name}/{cn}"
                        if group_full_name != "/"
                        else f"/{cn}"
                    )

                children_full.append(current_group_full_name)
                q.append(current_group)

            # Record this group entry via GroupInfo factory
            group_info = GroupInfo.from_group(
                group,
                variables=group_variables,
                children=children_full,
            )
            gkey = group_info.full_name
            if gkey != "/":
                gkey = gkey.lstrip("/")
            self.groups[gkey] = group_info


def get_metadata(
    source,
    open_options: dict[str, Any] | None = None,
) -> NetCDFMetadata:
    """Read and normalize all NetCDF MDIM metadata.

    Accepts several source types and delegates to
    `MetadataBuilder` to produce a `NetCDFMetadata` instance.

    Args:
        source (gdal.Dataset | str | object): The data source.
            Accepts a GDAL dataset directly, a file path (opened
            internally with `OF_MULTIDIM_RASTER`), or a pyramids
            `NetCDF`/`Dataset` instance whose internal
            `_raster` attribute is extracted automatically.
        open_options: Optional dictionary of GDAL open-options.
            Stored in the resulting metadata for provenance but
            not used to open the file.

    Returns:
        NetCDFMetadata: Fully populated metadata dataclass.

    Raises:
        ValueError: If *source* is a string path that cannot be
            opened as a multidimensional raster.

    Examples:
        Open from a file path:

        >>> from osgeo import gdal  # doctest: +SKIP
        >>> import pyramids.netcdf.metadata as meta  # doctest: +SKIP
        >>> md = meta.get_metadata(  # doctest: +SKIP
        ...     "precip.nc"
        ... )
        >>> md.driver  # doctest: +SKIP
        'netCDF'

    See Also:
        MetadataBuilder: The builder class used internally.
    """
    if isinstance(source, (str, Path)):
        ds = gdal.OpenEx(str(source), gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            raise ValueError(f"Could not open '{source}' as multidimensional raster")
        builder = MetadataBuilder(ds, open_options)
        result = builder.build()
        ds = None  # close the temporary handle
        return result
    elif hasattr(source, "_raster"):
        builder = MetadataBuilder(source._raster, open_options)
        return builder.build()
    else:
        builder = MetadataBuilder(source, open_options)
        return builder.build()


def to_dict(metadata: NetCDFMetadata) -> dict[str, Any]:
    """Convert `NetCDFMetadata` to plain dicts suitable for JSON.

    Recursively walks all dataclass fields and converts them to
    plain `dict` / `list` / scalar types so the result can be
    passed directly to `json.dumps`.

    Args:
        metadata: A `NetCDFMetadata` instance to convert.

    Returns:
        dict: Nested dictionary with all dataclass fields
            converted to plain dicts.

    Examples:
        Convert a minimal metadata object:

        >>> from pyramids.netcdf.metadata import to_dict
        >>> from pyramids.netcdf.models import (
        ...     NetCDFMetadata, StructuralInfo,
        ... )
        >>> md = NetCDFMetadata(
        ...     driver="netCDF",
        ...     root_group="/",
        ...     groups={},
        ...     variables={},
        ...     dimensions={},
        ...     global_attributes={"title": "test"},
        ...     structural=StructuralInfo(
        ...         driver_name="netCDF"
        ...     ),
        ...     created_with={"library": "GDAL"},
        ... )
        >>> d = to_dict(md)
        >>> d["driver"]
        'netCDF'
        >>> d["global_attributes"]["title"]
        'test'
        >>> d["structural"]["driver_name"]
        'netCDF'

    See Also:
        to_json: Serializes directly to a JSON string.
        from_json: Deserializes a JSON string back to
            `NetCDFMetadata`.
    """

    def convert(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return cast(dict[str, Any], convert(metadata))


def to_json(metadata: NetCDFMetadata) -> str:
    """Serialize `NetCDFMetadata` to a compact JSON string.

    Converts the dataclass tree to plain dicts via `to_dict`
    and then encodes to JSON with no extra whitespace.

    Args:
        metadata: A `NetCDFMetadata` instance to serialize.

    Returns:
        str: JSON-encoded string with no ASCII escaping and
            compact separators (no spaces after `,` or `:`).

    Examples:
        Round-trip a minimal metadata object:

        >>> import json
        >>> from pyramids.netcdf.metadata import to_json
        >>> from pyramids.netcdf.models import (
        ...     NetCDFMetadata, StructuralInfo,
        ... )
        >>> md = NetCDFMetadata(
        ...     driver="netCDF",
        ...     root_group="/",
        ...     groups={},
        ...     variables={},
        ...     dimensions={},
        ...     global_attributes={},
        ...     structural=StructuralInfo(
        ...         driver_name="netCDF"
        ...     ),
        ...     created_with={"library": "GDAL"},
        ... )
        >>> s = to_json(md)
        >>> json.loads(s)["driver"]
        'netCDF'

    See Also:
        to_dict: Converts to plain dicts without JSON encoding.
        from_json: Deserializes the string back to
            `NetCDFMetadata`.
    """
    return json.dumps(to_dict(metadata), ensure_ascii=False, separators=(",", ":"))


def from_json(s: str) -> NetCDFMetadata:
    """Deserialize `NetCDFMetadata` from a JSON string.

    Parses the JSON produced by `to_json` and manually
    reconstructs the dataclass hierarchy (`GroupInfo`,
    `VariableInfo`, `DimensionInfo`, `StructuralInfo`).

    Only the schema produced by `to_dict` / `to_json` is
    supported; arbitrary JSON will likely raise `KeyError`.

    Args:
        s: A JSON string previously produced by `to_json`.

    Returns:
        NetCDFMetadata: Reconstructed metadata instance.

    Raises:
        json.JSONDecodeError: If *s* is not valid JSON.
        KeyError: If required fields are missing from the
            JSON payload.

    Examples:
        Round-trip through JSON:

        >>> from pyramids.netcdf.metadata import (
        ...     to_json, from_json,
        ... )
        >>> from pyramids.netcdf.models import (
        ...     NetCDFMetadata, StructuralInfo,
        ... )
        >>> md = NetCDFMetadata(
        ...     driver="netCDF",
        ...     root_group="/",
        ...     groups={},
        ...     variables={},
        ...     dimensions={},
        ...     global_attributes={"history": "created"},
        ...     structural=StructuralInfo(
        ...         driver_name="netCDF"
        ...     ),
        ...     created_with={"library": "GDAL"},
        ... )
        >>> s = to_json(md)
        >>> restored = from_json(s)
        >>> restored.driver
        'netCDF'
        >>> restored.global_attributes["history"]
        'created'

    See Also:
        to_json: The serialization counterpart.
    """
    d = json.loads(s)

    def build_group(gd: dict[str, Any]) -> GroupInfo:
        return GroupInfo(
            name=gd["name"],
            full_name=gd["full_name"],
            attributes=gd.get("attributes", {}),
            children=gd.get("children", []),
            variables=gd.get("variables", []),
        )

    def build_dim(dd: dict[str, Any]) -> DimensionInfo:
        return DimensionInfo(
            name=dd["name"],
            full_name=dd["full_name"],
            size=int(dd["size"]),
            type=dd.get("type"),
            direction=dd.get("direction"),
            indexing_variable=dd.get("indexing_variable"),
            attrs=dd.get("attrs", {}),
        )

    def build_array(ad: dict[str, Any]) -> VariableInfo:
        return VariableInfo(
            name=ad["name"],
            full_name=ad["full_name"],
            dtype=ad.get("dtype", "unknown"),
            shape=[int(x) for x in ad.get("shape", [])],
            dimensions=[str(x) for x in ad.get("dimensions", [])],
            attributes=ad.get("attributes", {}),
            unit=ad.get("unit"),
            nodata=ad.get("nodata"),
            scale=ad.get("scale"),
            offset=ad.get("offset"),
            srs_wkt=ad.get("srs_wkt"),
            srs_projjson=ad.get("srs_projjson"),
            coordinate_variables=[str(x) for x in ad.get("coordinate_variables", [])],
            structural_info=ad.get("structural_info"),
            block_size=(
                [int(x) for x in ad.get("block_size", [])]
                if ad.get("block_size") is not None
                else None
            ),
        )

    groups = {k: build_group(v) for k, v in d.get("groups", {}).items()}
    variables = {k: build_array(v) for k, v in d.get("variables", {}).items()}
    dims = {k: build_dim(v) for k, v in d.get("dimensions", {}).items()}

    structural = d.get("structural")
    structural_obj = (
        StructuralInfo(
            driver_name=structural.get("driver_name", "UNKNOWN"),
            driver_metadata=structural.get("driver_metadata"),
        )
        if structural is not None
        else None
    )

    return NetCDFMetadata(
        driver=d.get("driver", "UNKNOWN"),
        root_group=d.get("root_group"),
        groups=groups,
        variables=variables,
        dimensions=dims,
        global_attributes=d.get("global_attributes", {}),
        structural=structural_obj,
        open_options_used=d.get("open_options_used"),
        created_with=d.get("created_with", {}),
    )


MAX_INDEXED_GLOBAL_ATTRS = 20


def flatten_for_index(metadata: NetCDFMetadata) -> dict[str, Any]:
    """Return a flat dict of key properties for indexing/search.

    Extracts a small, searchable summary from a full
    `NetCDFMetadata` instance. The result contains scalar
    counts, the first 20 global attributes (prefixed with
    `global.`), and sorted lists of variable and dimension names.

    Args:
        metadata: A `NetCDFMetadata` instance to flatten.

    Returns:
        dict: Flat dictionary containing:

            - `driver` (str): Driver name.
            - `root_group` (str | None): Root group path.
            - `group_count` (int): Number of groups.
            - `variable_count` (int): Number of variables.
            - `dimension_count` (int): Number of dimensions.
            - `global.<key>` entries for the first 20 global
              attributes.
            - `variables` (list[str]): Sorted variable names.
            - `dimensions` (list[str]): Sorted dimension
              names.

    Examples:
        Flatten a metadata object with one variable and one
        dimension:

        >>> from pyramids.netcdf.metadata import flatten_for_index
        >>> from pyramids.netcdf.models import (
        ...     NetCDFMetadata, StructuralInfo,
        ...     VariableInfo, DimensionInfo,
        ... )
        >>> md = NetCDFMetadata(
        ...     driver="netCDF",
        ...     root_group="/",
        ...     groups={},
        ...     variables={
        ...         "/temperature": VariableInfo(
        ...             name="temperature",
        ...             full_name="/temperature",
        ...             dtype="float32",
        ...             shape=[10, 20],
        ...             dimensions=["/lat", "/lon"],
        ...         ),
        ...     },
        ...     dimensions={
        ...         "/time": DimensionInfo(
        ...             name="time",
        ...             full_name="/time",
        ...             size=365,
        ...         ),
        ...     },
        ...     global_attributes={"title": "Sample"},
        ...     structural=StructuralInfo(
        ...         driver_name="netCDF"
        ...     ),
        ...     created_with={"library": "GDAL"},
        ... )
        >>> flat = flatten_for_index(md)
        >>> flat["driver"]
        'netCDF'
        >>> flat["variable_count"]
        1
        >>> flat["dimension_count"]
        1
        >>> flat["global.title"]
        'Sample'
        >>> flat["variables"]
        ['/temperature']
        >>> flat["dimensions"]
        ['/time']

    See Also:
        to_dict: Full recursive conversion to plain dicts.
    """
    d: dict[str, Any] = {
        "driver": metadata.driver,
        "root_group": metadata.root_group,
        "group_count": len(metadata.groups),
        "variable_count": len(metadata.variables),
        "dimension_count": len(metadata.dimensions),
    }
    # include some global attrs
    for k, v in list(metadata.global_attributes.items())[:MAX_INDEXED_GLOBAL_ATTRS]:
        d[f"global.{k}"] = v
    # include names of variables and dims
    d["variables"] = sorted([a for a in metadata.variables.keys()])
    d["dimensions"] = sorted([dname for dname in metadata.dimensions.keys()])
    return d
