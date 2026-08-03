"""Per-run provenance — what produced each output, for reproducibility.

Each successful input in a :func:`~pyramids.processing.runner.run` collects a
:class:`Provenance` record: the source, and per step the tool, parameters, and wall
time. :meth:`Provenance.to_pipeline` re-emits the exact :class:`Pipeline` that
produced the output, so a result can be reproduced or its recipe serialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyramids.processing.pipeline import Pipeline


@dataclass
class StepRecord:
    """One executed step's tool, parameters, and wall-clock duration.

    Attributes:
        tool: The tool name that ran.
        parameters: The parameter mapping it was called with.
        seconds: Wall-clock duration of the step, in seconds.
    """

    tool: str
    parameters: dict[str, Any]
    seconds: float


@dataclass
class Provenance:
    """The recorded recipe + timing for one processed input.

    Attributes:
        source: A label for the input (a path, or a ``<Type>`` marker for an
            in-memory object).
        steps: The executed :class:`StepRecord` entries, in order.
    """

    source: str
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        """Total wall-clock time across all steps."""
        return sum(record.seconds for record in self.steps)

    def to_pipeline(self) -> Pipeline:
        """Re-emit the exact :class:`Pipeline` that produced the output.

        Returns:
            A :class:`Pipeline` equal to the one that was run (so it round-trips
            through ``to_yaml``/``from_yaml`` and reproduces the result).
        """
        return Pipeline([(record.tool, record.parameters) for record in self.steps])
