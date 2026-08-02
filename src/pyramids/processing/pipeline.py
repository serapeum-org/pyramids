"""The :class:`Pipeline` — an ordered, validated chain of tool steps.

A pipeline is a list of ``(tool, parameters)`` steps where step *N*'s output feeds
step *N+1*. Steps are validated against the registry **at construction** (unknown
tool or schema-invalid parameters fail immediately, not mid-run). Serialization to and
from a portable YAML "model" file lives in this module too.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import yaml

from pyramids.processing.registry import resolve
from pyramids.processing.schema import validate_parameters


def _yaml_safe_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Copy ``parameters``, coercing any ``os.PathLike`` value to ``str`` for YAML."""
    return {
        key: (os.fspath(value) if isinstance(value, os.PathLike) else value)
        for key, value in parameters.items()
    }


@dataclass
class Step:
    """One pipeline step: a tool name and its parameter mapping.

    Attributes:
        tool: The registered tool name.
        parameters: The ``{name: value}`` mapping passed to the tool.
    """

    tool: str
    parameters: dict[str, Any]


class Pipeline:
    """An ordered, validated chain of geoprocessing steps.

    Args:
        steps: An iterable of ``(tool, parameters)`` pairs.

    Raises:
        ValueError: If any step names an unknown tool or supplies parameters that fail
            the tool's schema (validation happens here, at construction).
    """

    def __init__(self, steps: Iterable[tuple[str, dict[str, Any]]]):
        built: list[Step] = []
        for index, item in enumerate(steps):
            try:
                name, parameters = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pipeline step {index} must be a (tool, parameters) pair, got {item!r}"
                ) from exc
            tool = resolve(name)
            if parameters is None:
                parameters = {}
            elif not isinstance(parameters, dict):
                raise ValueError(
                    f"pipeline step {index}: parameters must be a mapping, got "
                    f"{type(parameters).__name__}"
                )
            parameters = dict(parameters)
            validate_parameters(tool, parameters)
            built.append(Step(name, parameters))
        self._steps = built

    @property
    def steps(self) -> list[Step]:
        """An independent copy of the pipeline's steps.

        Returns fresh :class:`Step` objects with copied ``parameters`` dicts, so
        mutating the returned steps (or their parameters) never affects the pipeline.
        """
        return [Step(step.tool, dict(step.parameters)) for step in self._steps]

    def __iter__(self) -> Iterator[Step]:
        """Iterate over independent copies of the steps (see :attr:`steps`)."""
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        inner = ", ".join(f"({s.tool!r}, {s.parameters!r})" for s in self._steps)
        return f"Pipeline([{inner}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pipeline):
            result: bool = NotImplemented
        else:
            mine = [(s.tool, s.parameters) for s in self._steps]
            theirs = [(s.tool, s.parameters) for s in other._steps]
            result = mine == theirs
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the pipeline as a plain, YAML-ready mapping.

        Each step's ``parameters`` is copied, and any ``os.PathLike`` value is coerced to
        a plain string so the mapping is safe to `yaml.safe_dump`.
        """
        return {
            "pipeline": [
                {
                    "tool": step.tool,
                    "parameters": _yaml_safe_parameters(step.parameters),
                }
                for step in self._steps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Build a pipeline from a mapping produced by :meth:`to_dict`.

        Args:
            data: A mapping with a ``"pipeline"`` list of ``{"tool", "parameters"}``.

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
            raise ValueError(
                "invalid pipeline data: 'pipeline' must be a list of steps"
            )
        steps: list[tuple[str, dict[str, Any]]] = []
        for index, step in enumerate(raw):
            if not isinstance(step, dict) or "tool" not in step:
                raise ValueError(
                    f"invalid pipeline step {index}: expected a mapping with a 'tool' key"
                )
            steps.append((step["tool"], step.get("parameters", {})))
        return cls(steps)

    def to_yaml(self, path: str) -> None:
        """Write the pipeline to a portable, version-controllable YAML file.

        Every step's parameters are re-validated with ``for_serialization=True`` first,
        so a pipeline carrying a non-serializable value (array / mask / callable /
        in-memory object) raises here instead of writing a file that cannot be
        loaded back.

        Args:
            path: Destination ``.yaml`` path.

        Raises:
            ValueError: If any step carries a non-serializable parameter value.
        """
        for step in self._steps:
            validate_parameters(
                resolve(step.tool), step.parameters, for_serialization=True
            )
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> Pipeline:
        """Read a pipeline from a YAML file written by :meth:`to_yaml`.

        Args:
            path: Path to a pipeline YAML file.

        Returns:
            A validated :class:`Pipeline`.

        Raises:
            ValueError: If the file is not a mapping with a ``"pipeline"`` list, or
                a step references an unknown tool / invalid parameters (re-validated via
                the constructor).
        """
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)
