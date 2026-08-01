"""The batch runner — execute a :class:`Pipeline` over one or many inputs.

``run`` opens each input, applies the pipeline's steps in order (step *N*'s output
feeds step *N+1*), and collects the results under an error policy (``"skip"`` to
collect failures and continue, ``"raise"`` to fail fast).
"""

from __future__ import annotations

import glob as _glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.provenance import Provenance, StepRecord
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
    provenance: list[Provenance] = field(default_factory=list)

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
            if not items:
                raise ValueError(f"no inputs matched glob {inputs!r}")
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


def _source_label(item: Any) -> str:
    """A provenance-friendly label for an input (path, or ``<Type>`` marker)."""
    return item if isinstance(item, str) else f"<{type(item).__name__}>"


def _materialize_array(array: "np.ndarray", source: Dataset) -> Dataset:
    """Wrap a bare array result into a single-band georeferenced ``Dataset``.

    The terrain ops (``slope``/``aspect``/``hillshade``) return a plain numpy array
    on the *same grid* as the raster they ran on. Re-attaching ``source``'s
    geotransform/CRS/no-data makes that output writable to disk and chainable into
    a subsequent ``Dataset`` step.
    """
    data = array if array.ndim == 3 else array[np.newaxis, :, :]
    return Dataset.create_from_array(
        data,
        geo=source.geotransform,
        epsg=source.epsg,
        no_data_value=source.no_data_value[0],
    )


def _run_pipeline_on(pipeline: Pipeline, obj: Any, source: str) -> tuple[Any, Provenance]:
    """Apply every step to ``obj``, timing each, and return output + provenance.

    A step whose tool declares ``returns="Array"`` yields a bare numpy array; it is
    materialized back into a georeferenced :class:`~pyramids.dataset.Dataset` using
    the array's source raster, so terminal terrain ops are writable and chainable.
    """
    prov = Provenance(source=source)
    for step in pipeline:
        source_obj = obj
        start = time.perf_counter()
        obj = _apply(step, obj)
        if resolve(step.tool).returns == "Array" and isinstance(obj, np.ndarray):
            obj = _materialize_array(obj, source_obj)
        prov.steps.append(StepRecord(step.tool, dict(step.params), time.perf_counter() - start))
    return obj, prov


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


def _execute_serial(
    pipeline: Pipeline, items: list[Any], on_error: str, out: str | None
) -> RunResult:
    """Run the pipeline over ``items`` sequentially in this process."""
    result = RunResult()
    for index, item in enumerate(items):
        try:
            obj = _open(item)
            obj, prov = _run_pipeline_on(pipeline, obj, _source_label(item))
            if out is not None:
                _write(obj, item, out, index)
            result.outputs.append(obj)
            result.provenance.append(prov)
        except Exception as exc:  # noqa: BLE001 - batch policy collects or re-raises
            if on_error == "raise":
                raise
            result.failures.append((item, exc))
    return result


def _run_one_worker(payload: tuple[dict[str, Any], str, str, int]) -> tuple[str, Provenance]:
    """Worker entry point: open a path, run the pipeline, write; return path + provenance.

    Runs in a separate process; it rebuilds the pipeline from a plain dict and
    opens the source worker-side so no GDAL handle crosses the process boundary.
    """
    pipe_dict, source, out_dir, index = payload
    pipeline = Pipeline.from_dict(pipe_dict)
    obj = _open(source)
    obj, prov = _run_pipeline_on(pipeline, obj, source)
    return _write(obj, source, out_dir, index), prov


def _execute_parallel(
    pipeline: Pipeline,
    items: list[Any],
    on_error: str,
    out: str,
    max_workers: int | None,
) -> RunResult:
    """Run the pipeline over ``items`` across a process pool (path-in/path-out)."""
    non_paths = [item for item in items if not isinstance(item, str)]
    if non_paths:
        raise ValueError(
            "parallel=True requires file-path inputs — GDAL handles cannot cross "
            "process boundaries, so pass paths/globs, not in-memory objects"
        )
    result = RunResult()
    pipe_dict = pipeline.to_dict()
    payloads = [(pipe_dict, src, out, i) for i, src in enumerate(items)]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_worker, payload): payload[1] for payload in payloads}
        for future in as_completed(futures):
            source = futures[future]
            try:
                path, prov = future.result()
                result.outputs.append(path)
                result.provenance.append(prov)
            except Exception as exc:  # noqa: BLE001 - batch policy collects or re-raises
                if on_error == "raise":
                    raise
                result.failures.append((source, exc))
    return result


def run(
    pipeline: Pipeline,
    inputs: Any,
    *,
    on_error: str = "skip",
    out: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> RunResult:
    """Run ``pipeline`` over ``inputs`` and collect the results.

    Args:
        pipeline: The :class:`Pipeline` to apply to each input.
        inputs: A single path/object, a glob string, a list/tuple of those, or a
            ``DatasetCollection``.
        on_error: ``"skip"`` collects ``(source, exception)`` failures and
            continues; ``"raise"`` fails fast on the first error.
        out: Optional output directory; when given, each successful output is
            written there, named after its source. Required when ``parallel=True``.
        parallel: When ``True``, run the batch across a process pool. Because GDAL
            handles cannot cross process boundaries, this requires **file-path**
            inputs and an ``out`` directory (outputs are written worker-side and
            ``RunResult.outputs`` holds the written paths, in completion order, not
            in-memory objects). Only registry tools registered at import (the
            allowlist) are available in workers.
        max_workers: Worker-process count for ``parallel=True`` (default: the
            pool's default, ~CPU count).

    Returns:
        A :class:`RunResult` with the successful ``outputs`` and any ``failures``.

    Raises:
        ValueError: If ``on_error`` is not ``"skip"``/``"raise"``, or ``parallel``
            is set without an ``out`` directory / with non-path inputs.
        Exception: The first per-item error when ``on_error="raise"``.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    items = _resolve_inputs(inputs)
    if parallel:
        if out is None:
            raise ValueError(
                "parallel=True requires an 'out' directory — outputs are written "
                "worker-side, not returned as objects"
            )
        result = _execute_parallel(pipeline, items, on_error, out, max_workers)
    else:
        result = _execute_serial(pipeline, items, on_error, out)
    return result
