"""The tool registry — pyramids ops made addressable by name.

Per ADR 0007 the registry is populated with **hand-written** :class:`ToolSpec`
entries for a curated allowlist of real-signature, serialization-safe ops, rather
than introspected from the (mostly ``(*args, **kwargs)``) public method
signatures. The allowlist is deliberately small for v1 and is trivially
extensible: add a :class:`ToolSpec` and :func:`register` it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from pyramids.base._utils import INTERPOLATION_METHODS
from pyramids.processing.schema import ParamSpec, ToolSpec

_REGISTRY: dict[str, ToolSpec] = {}

#: Resampling algorithm names accepted by to_crs/resample, sourced from the GDAL
#: resampling table so the OptionList choices stay in sync with what the ops allow.
_RESAMPLING_METHODS = tuple(sorted(INTERPOLATION_METHODS))


def register(spec: ToolSpec) -> ToolSpec:
    """Add ``spec`` to the registry (overwriting any tool of the same name)."""
    _REGISTRY[spec.name] = spec
    return spec


def resolve(name: str) -> ToolSpec:
    """Return the :class:`ToolSpec` registered under ``name``.

    Args:
        name: The tool name referenced by a pipeline or the CLI.

    Returns:
        The registered :class:`ToolSpec`.

    Raises:
        ValueError: If no tool is registered under ``name`` (the message lists the
            available tool names).
    """
    try:
        spec = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown tool {name!r}; registered tools: {tool_names()}"
        ) from exc
    return spec


def tool_names() -> list[str]:
    """Return the registered tool names, sorted."""
    return sorted(_REGISTRY)


def get_registry() -> Mapping[str, ToolSpec]:
    """Return a read-only view of the registry.

    Named ``get_registry`` rather than ``registry`` so it does not shadow the
    ``pyramids.processing.registry`` submodule when re-exported at the package
    root.
    """
    return MappingProxyType(_REGISTRY)


# Curated v1 allowlist (real signatures, serialization-safe params). Deviation
# from the #780 DoD's "~15": v1 ships 7 fully-verified ops rather than a larger
# half-verified set; the registry is extensible (see ADR 0007).

register(
    ToolSpec(
        name="slope",
        receiver="Dataset",
        returns="Array",
        description="Terrain slope from an elevation raster.",
        params=(
            ParamSpec("band", "Integer", 0, True, "Zero-based band index."),
            ParamSpec(
                "units",
                "OptionList",
                "degrees",
                True,
                "Slope units.",
                choices=("degrees", "radians"),
            ),
        ),
    )
)

register(
    ToolSpec(
        name="aspect",
        receiver="Dataset",
        returns="Array",
        description="Terrain aspect (compass direction of steepest descent).",
        params=(ParamSpec("band", "Integer", 0, True, "Zero-based band index."),),
    )
)

register(
    ToolSpec(
        name="hillshade",
        receiver="Dataset",
        returns="Array",
        description="Shaded-relief raster from an elevation raster.",
        params=(
            ParamSpec("azimuth", "Float", 315.0, True, "Sun azimuth in degrees."),
            ParamSpec("altitude", "Float", 45.0, True, "Sun altitude in degrees."),
            ParamSpec("band", "Integer", 0, True, "Zero-based band index."),
        ),
    )
)

register(
    ToolSpec(
        name="to_crs",
        receiver="Dataset",
        returns="Dataset",
        description="Reproject a raster to a target EPSG.",
        params=(
            ParamSpec("to_epsg", "Integer", None, False, "Target EPSG code."),
            ParamSpec(
                "method", "OptionList", None, True, "Resampling method.",
                choices=_RESAMPLING_METHODS,
            ),
            ParamSpec("cell_size", "Float", None, True, "Output cell size (CRS units)."),
        ),
    )
)

register(
    ToolSpec(
        name="resample",
        receiver="Dataset",
        returns="Dataset",
        description="Resample a raster to a new cell size.",
        params=(
            ParamSpec("cell_size", "Float", None, False, "New cell size (CRS units)."),
            ParamSpec(
                "method", "OptionList", None, True, "Resampling method.",
                choices=_RESAMPLING_METHODS,
            ),
        ),
    )
)

register(
    ToolSpec(
        name="interpolate_to_raster",
        receiver="FeatureCollection",
        returns="Dataset",
        description="Interpolate a point column onto a continuous raster surface.",
        params=(
            ParamSpec("column", "Field", None, False, "Numeric column to interpolate."),
            ParamSpec(
                "method",
                "OptionList",
                "idw",
                True,
                "Interpolation method.",
                choices=("idw",),
            ),
            ParamSpec("cell_size", "Float", None, True, "Output cell size (CRS units)."),
            ParamSpec("power", "Float", 2.0, True, "IDW distance exponent."),
            ParamSpec("n_neighbors", "Integer", None, True, "Nearest points per estimate."),
            ParamSpec("nodata", "Float", -9999.0, True, "No-data value for empty cells."),
        ),
    )
)

register(
    ToolSpec(
        name="to_h3",
        receiver="FeatureCollection",
        returns="FeatureCollection",
        description="Tag each point with its H3 cell index.",
        params=(ParamSpec("resolution", "Integer", None, False, "H3 resolution 0-15."),),
    )
)


#: Tool names present in a freshly-imported registry — i.e. available inside a
#: worker process. Tools added later via register() are absent from workers, so
#: the parallel runner rejects a pipeline that uses them (see runner).
BUILTIN_TOOLS = frozenset(_REGISTRY)
