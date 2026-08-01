"""The :class:`Pipeline` — an ordered, validated chain of tool steps.

A pipeline is a list of ``(tool, params)`` steps where step *N*'s output feeds
step *N+1*. Steps are validated against the registry **at construction** (unknown
tool or schema-invalid params fail immediately, not mid-run). Serialization to and
from a portable YAML "model" file lives in this module too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from pyramids.processing.registry import resolve
from pyramids.processing.schema import validate_params


@dataclass
class Step:
    """One pipeline step: a tool name and its parameter mapping.

    Attributes:
        tool: The registered tool name.
        params: The ``{name: value}`` mapping passed to the tool.
    """

    tool: str
    params: dict[str, Any]


class Pipeline:
    """An ordered, validated chain of geoprocessing steps.

    Args:
        steps: An iterable of ``(tool, params)`` pairs.

    Raises:
        ValueError: If any step names an unknown tool or supplies params that fail
            the tool's schema (validation happens here, at construction).
    """

    def __init__(self, steps: Iterable[tuple[str, dict[str, Any]]]):
        built: list[Step] = []
        for index, item in enumerate(steps):
            try:
                tool, params = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pipeline step {index} must be a (tool, params) pair, got {item!r}"
                ) from exc
            spec = resolve(tool)
            params = dict(params) if params else {}
            validate_params(spec, params)
            built.append(Step(tool, params))
        self._steps = built

    @property
    def steps(self) -> list[Step]:
        """A copy of the pipeline's steps."""
        return list(self._steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        inner = ", ".join(f"({s.tool!r}, {s.params!r})" for s in self._steps)
        return f"Pipeline([{inner}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pipeline):
            return NotImplemented
        mine = [(s.tool, s.params) for s in self._steps]
        theirs = [(s.tool, s.params) for s in other._steps]
        return mine == theirs

    def to_dict(self) -> dict[str, Any]:
        """Return the pipeline as a plain, YAML-ready mapping."""
        return {
            "pipeline": [{"tool": s.tool, "params": s.params} for s in self._steps]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        """Build a pipeline from a mapping produced by :meth:`to_dict`.

        Args:
            data: A mapping with a ``"pipeline"`` list of ``{"tool", "params"}``.

        Returns:
            A validated :class:`Pipeline`.

        Raises:
            ValueError: If ``data`` is not a mapping with a ``"pipeline"`` list, or
                a step is malformed (re-validated through the constructor).
        """
        if not isinstance(data, dict) or "pipeline" not in data:
            raise ValueError(
                "invalid pipeline data: expected a mapping with a 'pipeline' key"
            )
        raw = data["pipeline"]
        if not isinstance(raw, list):
            raise ValueError("invalid pipeline data: 'pipeline' must be a list of steps")
        steps: list[tuple[str, dict[str, Any]]] = []
        for index, step in enumerate(raw):
            if not isinstance(step, dict) or "tool" not in step:
                raise ValueError(
                    f"invalid pipeline step {index}: expected a mapping with a 'tool' key"
                )
            steps.append((step["tool"], step.get("params", {})))
        return cls(steps)
