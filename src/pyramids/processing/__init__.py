"""Declarative geoprocessing pipelines over pyramids ops.

The :mod:`pyramids.processing` package turns the existing ``Dataset`` /
``FeatureCollection`` operations into named, self-describing *tools* that can be
chained into a serializable :class:`~pyramids.processing.pipeline.Pipeline` and
run (batched) over one or many inputs.

The public surface grows as the pipeline layer is built (see issue #780); this
module re-exports the stable pieces as they land.
"""

from pyramids.processing.registry import (
    get_registry,
    register,
    resolve,
    tool_names,
)
from pyramids.processing.schema import PARAM_TYPES, ParamSpec, ToolSpec

__all__ = [
    "PARAM_TYPES",
    "ParamSpec",
    "ToolSpec",
    "get_registry",
    "register",
    "resolve",
    "tool_names",
]
