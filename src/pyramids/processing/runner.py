"""The batch runner — execute a :class:`Pipeline` over one or many inputs.

``run`` opens each input, applies the pipeline's steps in order (step *N*'s output
feeds step *N+1*), and collects the results under an error policy (``"skip"`` to
collect failures and continue, ``"raise"`` to fail fast).
"""

from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass, field
from typing import Any

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.registry import resolve

#: File extensions opened as vector (FeatureCollection) rather than raster.
_VECTOR_EXTS = frozenset(
    {".geojson", ".json", ".shp", ".gpkg", ".fgb", ".gml", ".kml"}
)


@dataclass
class RunResult:
    """The outcome of a batch :func:`run`.

    Attributes:
        outputs: The final object produced for each input that succeeded.
        failures: ``(source, exception)`` pairs for inputs that failed under the
            ``"skip"`` policy.
    """

    outputs: list[Any] = field(default_factory=list)
    failures: list[tuple[Any, Exception]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.outputs)

    @property
    def ok(self) -> bool:
        """Whether every input succeeded (no collected failures)."""
        return not self.failures


def _resolve_inputs(inputs: Any) -> list[Any]:
    """Expand ``inputs`` into a flat list of per-item sources.

    Accepts a single path/object, a glob string, an explicit list/tuple, or a
    ``DatasetCollection`` (duck-typed via its ``datasets`` attribute so the heavy
    collection import is not pulled into every ``import pyramids.processing``).
    """
    if isinstance(inputs, (list, tuple)):
        items = list(inputs)
    elif isinstance(inputs, str):
        if any(ch in inputs for ch in "*?["):
            items = sorted(_glob.glob(inputs))
        else:
            items = [inputs]
    else:
        datasets = getattr(inputs, "datasets", None)
        items = list(datasets) if datasets is not None else [inputs]
    return items


def _receiver_type(obj: Any) -> str:
    """Return ``"Dataset"``/``"FeatureCollection"`` for ``obj`` (else its type name)."""
    if isinstance(obj, FeatureCollection):
        name = "FeatureCollection"
    elif isinstance(obj, Dataset):
        name = "Dataset"
    else:
        name = type(obj).__name__
    return name


def _open(item: Any) -> Any:
    """Open ``item`` into a Dataset/FeatureCollection (pass objects through)."""
    if isinstance(item, (Dataset, FeatureCollection)):
        obj = item
    else:
        ext = os.path.splitext(str(item))[1].lower()
        if ext in _VECTOR_EXTS:
            obj = FeatureCollection.read_file(str(item))
        else:
            obj = Dataset.read_file(str(item))
    return obj


def _apply(step: Step, obj: Any) -> Any:
    """Apply one pipeline step to ``obj`` and return the result.

    Dispatches to ``obj``'s method only when ``obj``'s type matches the tool's
    declared receiver. This is what makes a cross-receiver chain safe: a step whose
    predecessor changed the object type (e.g. ``interpolate_to_raster`` turns a
    ``FeatureCollection`` into a ``Dataset``) is checked against the *new* type, and
    a step applied to the wrong receiver — including chaining past a terminal
    array-returning op — raises a clear ``TypeError`` instead of an opaque
    ``AttributeError``.

    Args:
        step: The pipeline step to apply.
        obj: The current pipeline object.

    Returns:
        The step's output (which may be a different receiver type).

    Raises:
        TypeError: If ``obj``'s type does not match the tool's declared receiver.
    """
    spec = resolve(step.tool)
    actual = _receiver_type(obj)
    if actual != spec.receiver:
        raise TypeError(
            f"tool {step.tool!r} expects a {spec.receiver}, but the current "
            f"pipeline object is a {actual}"
        )
    method = getattr(obj, spec.method_name)
    return method(**step.params)


def _write(obj: Any, source: Any, out_dir: str, index: int) -> str:
    """Write a pipeline output into ``out_dir``, named after its source."""
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(source, str):
        stem = os.path.splitext(os.path.basename(source))[0]
    else:
        stem = f"output_{index}"
    suffix = ".geojson" if _receiver_type(obj) == "FeatureCollection" else ".tif"
    path = os.path.join(out_dir, f"{stem}{suffix}")
    obj.to_file(path)
    return path


def run(
    pipeline: Pipeline,
    inputs: Any,
    *,
    on_error: str = "skip",
    out: str | None = None,
) -> RunResult:
    """Run ``pipeline`` over ``inputs`` and collect the results.

    Args:
        pipeline: The :class:`Pipeline` to apply to each input.
        inputs: A single path/object, a glob string, a list/tuple of those, or a
            ``DatasetCollection``.
        on_error: ``"skip"`` collects ``(source, exception)`` failures and
            continues; ``"raise"`` fails fast on the first error.
        out: Optional output directory; when given, each successful output is
            written there, named after its source.

    Returns:
        A :class:`RunResult` with the successful ``outputs`` and any ``failures``.

    Raises:
        ValueError: If ``on_error`` is not ``"skip"`` or ``"raise"``.
        Exception: The first per-item error when ``on_error="raise"``.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    result = RunResult()
    for index, item in enumerate(_resolve_inputs(inputs)):
        try:
            obj = _open(item)
            for step in pipeline:
                obj = _apply(step, obj)
            if out is not None:
                _write(obj, item, out, index)
            result.outputs.append(obj)
        except Exception as exc:  # noqa: BLE001 - batch policy collects or re-raises
            if on_error == "raise":
                raise
            result.failures.append((item, exc))
    return result
