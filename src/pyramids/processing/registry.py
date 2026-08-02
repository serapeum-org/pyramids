"""The tool registry — pyramids ops made addressable by name.

Per ADR 0007 the registry is populated with **hand-written** :class:`ToolMetadata`
entries for a curated allowlist of real-signature, serialization-safe ops, rather
than introspected from the (mostly ``(*args, **kwargs)``) public method
signatures. The allowlist is deliberately small for v1 and is trivially
extensible: add a :class:`ToolMetadata` and :func:`register` it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pyramids.base._utils import DEFAULT_RESAMPLING, INTERPOLATION_METHODS
from pyramids.feature.tessellation import QUADTREE_AGG
from pyramids.processing.schema import Parameter, ToolMetadata

_REGISTRY: dict[str, ToolMetadata] = {}

#: Resampling names for to_crs/resample, kept in sync with the GDAL table.
_RESAMPLING_METHODS = tuple(sorted(INTERPOLATION_METHODS))

#: Aggregation names for quadtree(agg=...), kept in sync with the tessellation reducers.
_QUADTREE_AGGS = tuple(sorted(QUADTREE_AGG))

_BAND_DESC = "Zero-based band index."
_RADIUS_DESC = "Neighbourhood radius in cells."


def register(tool: ToolMetadata) -> ToolMetadata:
    """Add ``tool`` to the registry (overwriting any tool of the same name)."""
    _REGISTRY[tool.name] = tool
    return tool


def resolve(name: str) -> ToolMetadata:
    """Return the :class:`ToolMetadata` registered under ``name``.

    Args:
        name: The tool name referenced by a pipeline or the CLI.

    Returns:
        The registered :class:`ToolMetadata`.

    Raises:
        ValueError: If no tool is registered under ``name`` (the message lists the
            available tool names).
    """
    try:
        tool = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown tool {name!r}; registered tools: {tool_names()}"
        ) from exc
    return tool


def tool_names() -> list[str]:
    """Return the registered tool names, sorted."""
    return sorted(_REGISTRY)


def catalog() -> Mapping[str, ToolMetadata]:
    """Return a read-only ``{name: ToolMetadata}`` view of the registered tools.

    Named ``catalog`` (not ``registry``) so it does not shadow the
    ``pyramids.processing.registry`` submodule when re-exported at the package root.
    """
    return MappingProxyType(_REGISTRY)


# Curated v1 allowlist of real-signature, serialization-safe ops (see ADR 0007);
# extensible at runtime via register().
_BUILTINS: tuple[ToolMetadata, ...] = (
    ToolMetadata(
        name="slope",
        input_type="Dataset",
        output_type="Array",
        description="Terrain slope from an elevation raster.",
        parameters=(
            Parameter("band", "Integer", 0, True, _BAND_DESC),
            Parameter(
                "units",
                "OptionList",
                "degrees",
                True,
                "Slope units.",
                choices=("degrees", "radians"),
            ),
        ),
    ),
    ToolMetadata(
        name="aspect",
        input_type="Dataset",
        output_type="Array",
        description="Terrain aspect (compass direction of steepest descent).",
        parameters=(Parameter("band", "Integer", 0, True, _BAND_DESC),),
    ),
    ToolMetadata(
        name="hillshade",
        input_type="Dataset",
        output_type="Array",
        description="Shaded-relief raster from an elevation raster.",
        parameters=(
            Parameter("azimuth", "Float", 315.0, True, "Sun azimuth in degrees."),
            Parameter("altitude", "Float", 45.0, True, "Sun altitude in degrees."),
            Parameter("band", "Integer", 0, True, _BAND_DESC),
        ),
    ),
    ToolMetadata(
        name="to_crs",
        input_type="Dataset",
        output_type="Dataset",
        description="Reproject a raster to a target EPSG.",
        parameters=(
            Parameter("to_epsg", "Integer", None, False, "Target EPSG code."),
            Parameter(
                "method",
                "OptionList",
                DEFAULT_RESAMPLING,
                True,
                "Resampling method.",
                choices=_RESAMPLING_METHODS,
            ),
            Parameter(
                "cell_size", "Float", None, True, "Output cell size (CRS units)."
            ),
        ),
    ),
    ToolMetadata(
        name="resample",
        input_type="Dataset",
        output_type="Dataset",
        description="Resample a raster to a new cell size.",
        parameters=(
            Parameter("cell_size", "Float", None, False, "New cell size (CRS units)."),
            Parameter(
                "method",
                "OptionList",
                DEFAULT_RESAMPLING,
                True,
                "Resampling method.",
                choices=_RESAMPLING_METHODS,
            ),
        ),
    ),
    ToolMetadata(
        name="interpolate_to_raster",
        input_type="FeatureCollection",
        output_type="Dataset",
        description="Interpolate a point column onto a continuous raster surface.",
        parameters=(
            Parameter("column", "Field", None, False, "Numeric column to interpolate."),
            Parameter(
                "method",
                "OptionList",
                "idw",
                True,
                "Interpolation method.",
                choices=("idw",),
            ),
            Parameter(
                "cell_size", "Float", None, True, "Output cell size (CRS units)."
            ),
            Parameter("power", "Float", 2.0, True, "IDW distance exponent."),
            Parameter(
                "n_neighbors", "Integer", None, True, "Nearest points per estimate."
            ),
            Parameter(
                "nodata", "Float", -9999.0, True, "No-data value for empty cells."
            ),
        ),
    ),
    ToolMetadata(
        name="to_h3",
        input_type="FeatureCollection",
        output_type="FeatureCollection",
        description="Tag each point with its H3 cell index.",
        parameters=(
            Parameter("resolution", "Integer", None, False, "H3 resolution 0-15."),
        ),
    ),
    ToolMetadata(
        name="fill",
        input_type="Dataset",
        output_type="Dataset",
        description="Fill every domain cell with a constant value.",
        parameters=(
            Parameter("value", "Float", None, False, "Value to fill the domain with."),
        ),
    ),
    ToolMetadata(
        name="sieve",
        input_type="Dataset",
        output_type="Dataset",
        description="Remove pixel clumps smaller than a threshold (speckle clean-up).",
        parameters=(
            Parameter(
                "threshold", "Integer", None, False, "Minimum clump size in pixels."
            ),
            Parameter("band", "Integer", 0, True, _BAND_DESC),
            Parameter(
                "connectedness", "Integer", 4, True, "Pixel connectedness (4 or 8)."
            ),
        ),
    ),
    ToolMetadata(
        name="focal_mean",
        input_type="Dataset",
        output_type="Array",
        description="Mean of each cell's neighbourhood (smoothing filter).",
        parameters=(
            Parameter("radius", "Integer", 1, True, _RADIUS_DESC),
            Parameter("band", "Integer", 0, True, _BAND_DESC),
        ),
    ),
    ToolMetadata(
        name="focal_std",
        input_type="Dataset",
        output_type="Array",
        description="Standard deviation of each cell's neighbourhood.",
        parameters=(
            Parameter("radius", "Integer", 1, True, _RADIUS_DESC),
            Parameter("band", "Integer", 0, True, _BAND_DESC),
        ),
    ),
    ToolMetadata(
        name="voronoi",
        input_type="FeatureCollection",
        output_type="FeatureCollection",
        description="Voronoi (Thiessen) tessellation of a point layer.",
        parameters=(
            Parameter("values", "Field", None, True, "Column copied onto each cell."),
        ),
    ),
    ToolMetadata(
        name="quadtree",
        input_type="FeatureCollection",
        output_type="FeatureCollection",
        description="Adaptive quad-tree binning of a point layer into cells.",
        parameters=(
            Parameter(
                "column", "Field", None, True, "Numeric column aggregated per cell."
            ),
            Parameter(
                "agg",
                "OptionList",
                "mean",
                True,
                "Aggregation function name.",
                choices=_QUADTREE_AGGS,
            ),
            Parameter(
                "nmax", "Integer", 100, True, "Max points per cell before splitting."
            ),
            Parameter("nmin", "Integer", 0, True, "Min points for a cell to be kept."),
        ),
    ),
    ToolMetadata(
        name="with_centroid",
        input_type="FeatureCollection",
        output_type="FeatureCollection",
        description="Add centroid x/y columns to each feature.",
    ),
    ToolMetadata(
        name="with_coordinates",
        input_type="FeatureCollection",
        output_type="FeatureCollection",
        description="Add per-vertex x/y coordinate columns.",
    ),
)

for _tool in _BUILTINS:
    register(_tool)


#: Tool names in a freshly-imported registry (available inside a worker process);
#: tools added later via register() are rejected by the parallel runner.
BUILTIN_TOOLS = frozenset(_REGISTRY)

_BUILTINS_BY_NAME = {t.name: t for t in _BUILTINS}


def _is_builtin_overridden(name: str) -> bool:
    """Return whether ``name`` is a builtin whose spec was replaced via register().

    Worker processes re-import the registry fresh and always resolve the original
    builtin, so a builtin overridden in the parent process is silently ignored under
    parallel execution — the runner rejects such a pipeline up front.
    """
    original = _BUILTINS_BY_NAME.get(name)
    return original is not None and _REGISTRY.get(name) is not original
