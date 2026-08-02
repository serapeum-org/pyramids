"""Declarative geoprocessing pipelines over pyramids ops.

The :mod:`pyramids.processing` package turns the existing ``Dataset`` /
``FeatureCollection`` operations into named, self-describing *tools* that can be
chained into a serializable :class:`~pyramids.processing.pipeline.Pipeline` and
run (batched) over one or many inputs.

The public surface grows as the pipeline layer is built (see issue #780); this
module re-exports the stable pieces as they land.
"""

from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.provenance import Provenance, StepRecord
from pyramids.processing.registry import (
    catalog,
    register,
    resolve,
    tool_names,
)
from pyramids.processing.runner import RunResult, run
from pyramids.processing.schema import (
    PARAMETER_TYPES,
    Parameter,
    ToolMetadata,
    validate_parameters,
)

__all__ = [
    "PARAMETER_TYPES",
    "Parameter",
    "Pipeline",
    "Provenance",
    "RunResult",
    "Step",
    "StepRecord",
    "ToolMetadata",
    "catalog",
    "register",
    "resolve",
    "run",
    "tool_names",
    "validate_parameters",
]
