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

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from pyramids.processing.pipeline import Pipeline, Step
from pyramids.processing.provenance import Provenance, StepRecord
from pyramids.processing.registry import (
    BUILTIN_TOOLS,
    _is_builtin_overridden,
    resolve,
)
from pyramids.processing.schema import ToolMetadata

# Extensions opened as vector; .json is included for GeoJSON.
_VECTOR_EXTS = frozenset({".geojson", ".json", ".shp", ".gpkg", ".fgb", ".gml", ".kml"})


@dataclass
class RunResult:
    """The outcome of a batch :func:`run`.

    Attributes:
        outputs: The final object produced for each input that succeeded, in input
            order. In serial mode these are in-memory `Dataset`/`FeatureCollection`
            objects; in `parallel` mode they are the written output **path strings**
            (GDAL handles cannot cross a process boundary).
        failures: ``(source, exception)`` pairs for inputs that failed under the
            ``"skip"`` policy.
        provenance: One :class:`~pyramids.processing.provenance.Provenance` record
            per successful input (tool, parameters, and timing per step).
    """

    outputs: list[Any] = field(default_factory=list)
    failures: list[tuple[Any, Exception]] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)

    def __len__(self) -> int:
        """The number of successful outputs (failures are counted separately)."""
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
    if isinstance(inputs, os.PathLike):
        inputs = os.fspath(inputs)
    if isinstance(inputs, (list, tuple)):
        items = [os.fspath(i) if isinstance(i, os.PathLike) else i for i in inputs]
    elif isinstance(inputs, str):
        if any(ch in inputs for ch in "*?["):
            items = sorted(_glob.glob(inputs, recursive=True))
            if not items:
                raise ValueError(f"no inputs matched glob {inputs!r}")
        else:
            items = [inputs]
    else:
        datasets = getattr(inputs, "datasets", None)
        items = list(datasets) if datasets is not None else [inputs]
    return items


def _object_type(obj: Any) -> str:
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


def _apply(step: Step, obj: Any, tool: ToolMetadata) -> Any:
    """Apply one pipeline step to ``obj`` and return the result.

    Dispatches to ``obj``'s method only when ``obj``'s type matches the tool's
    declared ``input_type``. This is what makes a cross-type chain safe: a step whose
    predecessor changed the object type (e.g. ``interpolate_to_raster`` turns a
    ``FeatureCollection`` into a ``Dataset``) is checked against the *new* type, and
    a step applied to the wrong input type raises a clear ``TypeError`` instead of an
    opaque ``AttributeError``.

    Args:
        step: The pipeline step to apply.
        obj: The current pipeline object.
        tool: The step's resolved :class:`ToolMetadata` (resolved once by the caller).

    Returns:
        The step's output (which may be a different object type).

    Raises:
        TypeError: If ``obj``'s type does not match the tool's declared ``input_type``.
    """
    actual = _object_type(obj)
    if actual != tool.input_type:
        raise TypeError(
            f"tool {step.tool!r} expects a {tool.input_type}, but the current "
            f"pipeline object is a {actual}"
        )
    method = getattr(obj, tool.method_name)
    return method(**step.parameters)


def _source_label(item: Any) -> str:
    """A provenance-friendly label for an input (path, or ``<Type>`` marker)."""
    if isinstance(item, (str, os.PathLike)):
        label = os.fspath(item)
    else:
        label = f"<{type(item).__name__}>"
    return label


def _materialize_array(array: np.ndarray, source: Dataset, band: int = 0) -> Dataset:
    """Wrap a bare array result into a single-band georeferenced ``Dataset``.

    The terrain/focal ops (``slope``/``aspect``/``hillshade``/``focal_*``) return a
    plain numpy array on the *same grid* as the raster they ran on, carrying the
    processed band's no-data sentinel. Re-attaching ``source``'s geotransform/CRS and
    that band's no-data makes the output writable to disk and chainable into a
    subsequent ``Dataset`` step.

    Args:
        array: The op's output array (2-D or 3-D).
        source: The raster the op ran on.
        band: The band index the op processed (its no-data is carried through).
    """
    nodata = source.no_data_value
    if band < len(nodata):
        no_data_value = nodata[band]
    else:
        # A malformed/short per-band no-data list: fall back to band 0's sentinel,
        # or None when the source declares no no-data at all (avoids an IndexError).
        no_data_value = nodata[0] if nodata else None
    data = array if array.ndim == 3 else array[np.newaxis, :, :]
    return Dataset.from_array(
        data,
        no_data_value=no_data_value,
        geo_ref=GeoReference(geo=source.geotransform, epsg=source.epsg or source.crs),
    )


def _run_pipeline_on(
    pipeline: Pipeline, obj: Any, source: str
) -> tuple[Any, Provenance]:
    """Apply every step to ``obj``, timing each, and return output + provenance.

    A step whose tool declares ``output_type="Array"`` yields a bare numpy array; it is
    materialized back into a georeferenced :class:`~pyramids.dataset.Dataset` using
    the array's source raster, so terminal terrain ops are writable and chainable.
    """
    prov = Provenance(source=source)
    for step in pipeline:
        tool = resolve(step.tool)
        source_obj = obj
        start = time.perf_counter()
        obj = _apply(step, obj, tool)
        if tool.output_type == "Array" and isinstance(obj, np.ndarray):
            obj = _materialize_array(obj, source_obj, step.parameters.get("band", 0))
        prov.steps.append(
            StepRecord(step.tool, dict(step.parameters), time.perf_counter() - start)
        )
    return obj, prov


def _write(obj: Any, source: Any, out_dir: str, index: int) -> str:
    """Write a pipeline output into ``out_dir``, named ``<stem>_<index>``.

    The batch index is always part of the name so same-basename inputs from
    different directories never collide (and parallel workers never race on one
    path). A path/``PathLike`` source contributes its basename; an in-memory
    object uses ``output``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(source, (str, os.PathLike)):
        stem = os.path.splitext(os.path.basename(os.fspath(source)))[0]
    else:
        stem = "output"
    suffix = ".geojson" if _object_type(obj) == "FeatureCollection" else ".tif"
    path = os.path.join(out_dir, f"{stem}_{index}{suffix}")
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


def _run_one_worker(
    payload: tuple[dict[str, Any], str, str, int],
) -> tuple[str, Provenance]:
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
    non_paths = [item for item in items if not isinstance(item, (str, os.PathLike))]
    if non_paths:
        raise ValueError(
            "parallel=True requires file-path inputs — GDAL handles cannot cross "
            "process boundaries, so pass paths/globs, not in-memory objects"
        )
    non_builtin = sorted(
        {step.tool for step in pipeline if step.tool not in BUILTIN_TOOLS}
    )
    if non_builtin:
        raise ValueError(
            f"parallel=True cannot use runtime-registered tools {non_builtin}; worker "
            "processes only see the import-time allowlist. Run these serially."
        )
    overridden = sorted({s.tool for s in pipeline if _is_builtin_overridden(s.tool)})
    if overridden:
        raise ValueError(
            f"parallel=True cannot use overridden builtin tools {overridden}; worker "
            "processes resolve the original builtin, not your override. Run serially."
        )
    pipe_dict = pipeline.to_dict()
    payloads = [(pipe_dict, src, out, i) for i, src in enumerate(items)]
    successes: dict[int, tuple[str, Provenance]] = {}
    failures: dict[int, tuple[Any, Exception]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_worker, p): (p[3], p[1]) for p in payloads}
        for future in as_completed(futures):
            index, source = futures[future]
            try:
                successes[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - batch policy collects or re-raises
                if on_error == "raise":
                    pool.shutdown(cancel_futures=True)  # fail fast: drop pending work
                    raise
                failures[index] = (source, exc)
    # Reassemble in input order so outputs/provenance align with the inputs exactly
    # as they do in serial mode (futures complete out of order).
    result = RunResult()
    for index in range(len(items)):
        if index in successes:
            path, prov = successes[index]
            result.outputs.append(path)
            result.provenance.append(prov)
        elif index in failures:
            result.failures.append(failures[index])
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
            ``DatasetCollection``. A string containing glob metacharacters
            (``*``/``?``/``[``) is expanded as a glob — to open a literal path that
            contains those characters, pass it inside a list.
        on_error: ``"skip"`` collects ``(source, exception)`` failures and
            continues; ``"raise"`` fails fast on the first error.
        out: Optional output directory; when given, each successful output is
            written there as ``<source-stem>_<index>`` (the batch index keeps
            same-basename inputs from colliding). Existing files at those paths are
            overwritten — re-running a batch into the same directory replaces prior
            outputs. Required when ``parallel=True``.
        parallel: When ``True``, run the batch across a process pool. Because GDAL
            handles cannot cross process boundaries, this requires **file-path**
            inputs and an ``out`` directory (outputs are written worker-side and
            ``RunResult.outputs`` holds the written paths, in input order — same as
            serial mode — not in-memory objects). Only registry tools registered at
            import (the allowlist) are available in workers. Note that under
            ``on_error="raise"`` the surfaced error is the first worker to *complete*,
            which need not be the first input's (serial raises the first input's error).
        max_workers: Worker-process count for ``parallel=True`` — ignored in serial
            mode (default: the pool's default, ~CPU count).

    Returns:
        A :class:`RunResult` with the successful ``outputs`` (objects in serial mode,
        written paths in ``parallel`` mode), any ``failures``, and per-input
        ``provenance``.

    Raises:
        ValueError: If ``on_error`` is not ``"skip"``/``"raise"``, or ``parallel``
            is set without an ``out`` directory / with non-path inputs.
        Exception: The first per-item error when ``on_error="raise"``.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    if parallel:
        if out is None:
            raise ValueError(
                "parallel=True requires an 'out' directory — outputs are written "
                "worker-side, not returned as objects"
            )
        if max_workers is not None and max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    items = _resolve_inputs(inputs)
    if not items:
        raise ValueError("no inputs to process (the resolved input set is empty)")
    if parallel:
        assert out is not None  # guarded above; narrows str | None -> str for typing
        result = _execute_parallel(pipeline, items, on_error, out, max_workers)
    else:
        result = _execute_serial(pipeline, items, on_error, out)
    return result
