"""Self-describing tool/parameter schema for the processing registry.

A :class:`ToolMetadata` describes one pyramids op made addressable by name: which
object it runs on (``input_type``), what it produces (``output_type``), and its
parameters. Each :class:`Parameter` carries a tagged ``parameter_type`` plus the
metadata the pipeline layer needs — a default, whether it is optional, and
whether its value can be serialized into a portable pipeline file.

This schema is the single source of truth for CLI help, pipeline validation, and
serialization-safety (see ADR 0007).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

#: Type tags a Parameter may declare.
PARAMETER_TYPES = frozenset(
    {
        "Raster",
        "Vector",
        "NewFile",
        "Float",
        "Integer",
        "Boolean",
        "String",
        "Field",
        "OptionList",
    }
)

#: Param types whose value can be written into a pipeline YAML by value.
_SERIALIZABLE_TYPES = frozenset(
    {"Float", "Integer", "Boolean", "String", "Field", "OptionList", "NewFile"}
)

#: Object types a tool can be invoked on.
INPUT_TYPES = frozenset({"Dataset", "FeatureCollection"})

#: Types a tool may return; "Array" ops are numpy arrays the runner wraps into a Dataset.
OUTPUT_TYPES = frozenset({"Dataset", "FeatureCollection", "Array"})


@dataclass(frozen=True)
class Parameter:
    """Describe a single tool parameter.

    Args:
        name: The keyword-argument name passed to the underlying op.
        parameter_type: One of :data:`PARAMETER_TYPES`.
        default: Display-only default shown in ``help``. The runner passes only the
            parameters a step supplies, so this value is never itself applied — the
            *runtime* default is always whatever the underlying method uses. Set it
            to **mirror** that method's real default so ``help`` advertises the true
            value (e.g. ``"nearest neighbor"`` for a resampling ``method``); leave it
            ``None`` only when the method's default is dynamic or unknown.
        optional: Whether the parameter may be omitted.
        description: Human-readable help text.
        choices: Allowed values for an ``"OptionList"`` parameter.
        serializable: Override for whether the value can be serialized; when
            ``None`` it is derived from ``parameter_type``.

    Raises:
        ValueError: If ``parameter_type`` is unknown or ``choices`` is given for a
            non-``OptionList`` parameter.
    """

    name: str
    parameter_type: str
    default: Any = None
    optional: bool = True
    description: str = ""
    choices: tuple[str, ...] | None = None
    serializable: bool | None = None

    def __post_init__(self) -> None:
        if self.parameter_type not in PARAMETER_TYPES:
            raise ValueError(
                f"unknown parameter_type {self.parameter_type!r} for parameter "
                f"{self.name!r}; valid: {sorted(PARAMETER_TYPES)}"
            )
        if self.choices is not None and self.parameter_type != "OptionList":
            raise ValueError(
                f"parameter {self.name!r}: choices are only valid for an "
                f"'OptionList' parameter_type, not {self.parameter_type!r}"
            )

    @property
    def is_serializable(self) -> bool:
        """Whether a value for this parameter can be written to a pipeline file."""
        if self.serializable is not None:
            result = self.serializable
        else:
            result = self.parameter_type in _SERIALIZABLE_TYPES
        return result

    def validate(self, value: Any) -> None:
        """Validate ``value`` against this parameter's type.

        Args:
            value: The value supplied for the parameter.

        Raises:
            ValueError: If ``value`` does not match ``parameter_type`` (this is what
                rejects a numpy array / mask / callable handed to a scalar param).
        """
        pt = self.parameter_type
        ok = True
        if pt == "Float":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif pt == "Integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif pt == "Boolean":
            ok = isinstance(value, bool)
        elif pt in {"String", "Field"}:
            ok = isinstance(value, str)
        elif pt == "OptionList":
            ok = isinstance(value, str) and (
                self.choices is None or value in self.choices
            )
        elif pt == "NewFile":
            ok = isinstance(value, (str, os.PathLike))
        # Raster/Vector accept an object or a path; left permissive, flagged non-serializable.
        if not ok:
            expected = pt if self.choices is None else f"one of {list(self.choices)}"
            raise ValueError(
                f"parameter {self.name!r} expects {expected}, got "
                f"{type(value).__name__} ({value!r})"
            )

    def coerce(self, raw: str) -> Any:
        """Coerce a raw CLI string into this parameter's type.

        Public API kept for a future CLI ``--set key=value`` path; not yet wired
        into a shipped command (the ``run`` subcommand reads typed parameters from YAML).

        Args:
            raw: The string value from the command line.

        Returns:
            The value converted to the parameter's Python type.

        Raises:
            ValueError: If ``raw`` cannot be converted (e.g. a non-numeric string
                for a ``"Float"`` parameter, or a value outside ``choices``).
        """
        pt = self.parameter_type
        if pt == "Float":
            result: Any = float(raw)
        elif pt == "Integer":
            result = int(raw)
        elif pt == "Boolean":
            low = raw.strip().lower()
            if low in {"1", "true", "yes", "on"}:
                result = True
            elif low in {"0", "false", "no", "off"}:
                result = False
            else:
                raise ValueError(f"parameter {self.name!r}: {raw!r} is not a boolean")
        else:
            result = raw
            if (
                pt == "OptionList"
                and self.choices is not None
                and raw not in self.choices
            ):
                raise ValueError(
                    f"parameter {self.name!r}: {raw!r} not in {list(self.choices)}"
                )
        return result

    def help(self) -> str:
        """Render a one-line CLI/help description of this parameter."""
        flag = "optional" if self.optional else "required"
        default = "" if self.default is None else f", default={self.default!r}"
        choices = "" if self.choices is None else f" {list(self.choices)}"
        desc = f" — {self.description}" if self.description else ""
        return f"{self.name} ({self.parameter_type}{choices}, {flag}{default}){desc}"


@dataclass(frozen=True)
class ToolMetadata:
    """Describe one named, addressable pyramids op.

    Args:
        name: The tool name used in a pipeline and on the CLI.
        input_type: The object the tool runs on — ``"Dataset"`` or
            ``"FeatureCollection"``.
        output_type: The object type the tool produces.
        parameters: The tool's parameters.
        description: Human-readable summary.
        method: The method name on the input object; defaults to ``name``.

    Raises:
        ValueError: If ``input_type``/``output_type`` are not valid object types or a
            parameter name is duplicated.
    """

    name: str
    input_type: str
    output_type: str
    parameters: tuple[Parameter, ...] = ()
    description: str = ""
    method: str | None = None

    def __post_init__(self) -> None:
        if self.input_type not in INPUT_TYPES:
            raise ValueError(
                f"tool {self.name!r}: input_type must be one of "
                f"{sorted(INPUT_TYPES)}, got {self.input_type!r}"
            )
        if self.output_type not in OUTPUT_TYPES:
            raise ValueError(
                f"tool {self.name!r}: output_type must be one of "
                f"{sorted(OUTPUT_TYPES)}, got {self.output_type!r}"
            )
        seen = [p.name for p in self.parameters]
        if len(seen) != len(set(seen)):
            raise ValueError(f"tool {self.name!r}: duplicate parameter names in {seen}")

    @property
    def method_name(self) -> str:
        """The method this tool invokes on its input object."""
        return self.method or self.name

    def param(self, name: str) -> Parameter | None:
        """Return the :class:`Parameter` named ``name`` (or ``None``)."""
        found = None
        for param in self.parameters:
            if param.name == name:
                found = param
                break
        return found

    def help(self) -> str:
        """Render a multi-line help block describing the tool and its parameters."""
        lines = [f"{self.name} ({self.input_type} -> {self.output_type})"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.parameters:
            lines.append("  parameters:")
            lines.extend(f"    {p.help()}" for p in self.parameters)
        else:
            lines.append("  parameters: (none)")
        return "\n".join(lines)


def validate_parameters(
    tool: ToolMetadata, parameters: dict[str, Any], *, for_serialization: bool = False
) -> None:
    """Validate a parameter mapping against a tool's schema.

    Args:
        tool: The tool whose schema ``parameters`` must satisfy.
        parameters: The supplied ``{name: value}`` mapping.
        for_serialization: When ``True``, additionally reject any value whose
            parameter is not pipeline-serializable (arrays, masks, callables,
            in-memory ``Raster``/``Vector`` objects) so ``to_yaml`` never writes a
            file that cannot be loaded back.

    Raises:
        ValueError: If a parameter is unknown, a value fails its type check, a
            non-serializable value is supplied under ``for_serialization``, or a
            required parameter is missing.
    """
    known = {p.name: p for p in tool.parameters}
    for key, value in parameters.items():
        try:
            param = known[key]
        except KeyError as exc:
            raise ValueError(
                f"tool {tool.name!r}: unknown parameter {key!r}; valid: {sorted(known)}"
            ) from exc
        param.validate(value)
        if for_serialization and not param.is_serializable:
            raise ValueError(
                f"tool {tool.name!r}: parameter {key!r} ({param.parameter_type}) is not "
                "pipeline-serializable and cannot be written to a pipeline file"
            )
    missing = [
        p.name for p in tool.parameters if not p.optional and p.name not in parameters
    ]
    if missing:
        raise ValueError(
            f"tool {tool.name!r}: missing required parameter(s): {missing}"
        )
