"""zarr-v3-safe kerchunk manifest builder for NetCDF4 / HDF5.

Builds a kerchunk v1 reference manifest by walking the HDF5 container with
:mod:`h5py` and emitting zarr-v2 metadata + byte-range chunk references
**directly** — without ever instantiating a live zarr group. This avoids the
``kerchunk.hdf`` path that drives zarr's synchronous v2 API, which under zarr v3
submits work to a process-global event loop and flakily deadlocks (#530).

The output is the ordinary kerchunk reference document, so the consumer side
(``xr.open_dataset(engine="kerchunk")``) is unchanged.

Scope: the feature surface GDAL's netCDF driver writes — fixed-width numeric and
fixed-length string dtypes, ``deflate``/``shuffle`` filters, dimension scales.
Unsupported HDF5 features raise a clear :class:`ValueError`.
"""

from __future__ import annotations

import base64
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_H5PY_IMPORT_ERROR = (
    "h5py is required for native kerchunk reference manifests. "
    "Install with the [lazy] extra:\n"
    "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-lazy"
)

# HDF5 attribute keys that are container bookkeeping, not user metadata; they are
# stripped from the emitted ``.zattrs`` (their information is carried elsewhere:
# fill_value from _FillValue, dimension names from _ARRAY_DIMENSIONS).
_DROP_ATTRS = frozenset(
    {
        "_Netcdf4Coordinates",
        "_Netcdf4Dimid",
        "_nc3_strict",
        "CLASS",
        "NAME",
        "REFERENCE_LIST",
        "DIMENSION_LIST",
        "_FillValue",
        "_NCProperties",
    }
)

# HDF5 filter ids (see H5Zpublic.h).
_H5_DEFLATE = 1
_H5_SHUFFLE = 2
_H5_FLETCHER32 = 3


def _require_h5py() -> Any:
    """Lazy-import :mod:`h5py`, raising a clear error when it is missing."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(_H5PY_IMPORT_ERROR) from exc
    return h5py


def _to_jsonable(value: Any) -> Any:
    """Convert an HDF5 attribute value into a JSON-serialisable Python value."""
    if isinstance(value, bytes):
        result = value.decode("utf-8", "replace")
    elif isinstance(value, np.ndarray):
        # GDAL writes scalar CF attributes as 1-element arrays; squeeze them to
        # plain scalars (matching kerchunk / CF expectations), keep real vectors.
        if value.size == 1:
            result = _to_jsonable(value.reshape(-1)[0])
        else:
            result = [_to_jsonable(v) for v in value.tolist()]
    elif isinstance(value, (np.bytes_,)):
        result = bytes(value).decode("utf-8", "replace")
    elif isinstance(value, np.floating):
        result = float(value)
    elif isinstance(value, np.integer):
        result = int(value)
    elif isinstance(value, np.bool_):
        result = bool(value)
    else:
        result = value
    return result


def _clean_attrs(attrs: Any) -> dict[str, Any]:
    """Return user-facing attributes, dropping HDF5/NetCDF bookkeeping keys."""
    cleaned: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in _DROP_ATTRS:
            continue
        cleaned[key] = _to_jsonable(value)
    return cleaned


def _basename(path: str) -> str:
    """Last path component of an HDF5 object path (``/g/x`` -> ``x``)."""
    return path.rstrip("/").rsplit("/", 1)[-1]


def _array_dimensions(h5file: Any, dataset: Any) -> list[str]:
    """Resolve a dataset's dimension names for the ``_ARRAY_DIMENSIONS`` attr.

    A dimension-scale dataset is named for itself; a data variable carries a
    ``DIMENSION_LIST`` of object references to its per-axis scales.
    """
    h5py = _require_h5py()
    attrs = dataset.attrs
    if attrs.get("CLASS") == b"DIMENSION_SCALE":
        dims = [_basename(dataset.name)]
    elif "DIMENSION_LIST" in attrs:
        dims = []
        for axis_refs in attrs["DIMENSION_LIST"]:
            ref = axis_refs[0] if len(axis_refs) else None
            if isinstance(ref, h5py.Reference) and ref:
                dims.append(_basename(h5file[ref].name))
            else:
                dims.append(f"phony_dim_{len(dims)}")
    elif dataset.ndim == 0:
        dims = []
    else:
        dims = [f"phony_dim_{i}" for i in range(dataset.ndim)]
    return dims


def _encode_fill_value(dataset: Any) -> Any:
    """Encode ``_FillValue`` for zarr v2 metadata, or ``None`` when absent.

    Only an explicit ``_FillValue`` attribute counts — HDF5's implicit default
    fill (which GDAL leaves on plain coordinate variables) is reported as
    ``None`` so zarr does not treat real data as missing.
    """
    attrs = dataset.attrs
    if "_FillValue" not in attrs:
        return None
    raw = attrs["_FillValue"]
    value = np.asarray(raw)
    if value.size == 1:
        value = value.reshape(()).item() if value.ndim else value.item()
    kind = dataset.dtype.kind
    if kind in ("f", "c"):
        as_float = float(value)
        if math.isnan(as_float):
            encoded: Any = "NaN"
        elif math.isinf(as_float):
            encoded = "Infinity" if as_float > 0 else "-Infinity"
        else:
            encoded = as_float
    elif kind in ("i", "u"):
        encoded = int(value)
    elif kind in ("S", "V", "U"):
        as_bytes = value if isinstance(value, bytes) else bytes(np.asarray(raw))
        encoded = base64.b64encode(as_bytes).decode("ascii")
    else:
        encoded = _to_jsonable(value)
    return encoded


def _compressor_and_filters(dataset: Any) -> tuple[dict | None, list[dict] | None]:
    """Map the HDF5 filter pipeline to a zarr v2 (compressor, filters) pair.

    Supports deflate (-> zlib compressor) and shuffle (-> shuffle filter), and
    silently drops the fletcher32 checksum (zarr has no equivalent; the data
    remains readable). Any other filter id is unsupported.

    zarr decodes the byte-compressor first, then reverses ``filters`` — so the
    mapping is only correct when shuffle is applied *before* deflate in the HDF5
    pipeline (which GDAL's netCDF driver always does). A shuffle that appears
    after deflate would decode wrong, so that ordering is rejected rather than
    silently mis-decoded.
    """
    dcpl = dataset.id.get_create_plist()
    compressor: dict | None = None
    filters: list[dict] = []
    for index in range(dcpl.get_nfilters()):
        filter_id, _flags, cd_values, _name = dcpl.get_filter(index)
        if filter_id == _H5_DEFLATE:
            level = cd_values[0] if cd_values else 4
            compressor = {"id": "zlib", "level": int(level)}
        elif filter_id == _H5_SHUFFLE:
            if compressor is not None:
                raise ValueError(
                    f"Dataset {dataset.name!r} applies shuffle after deflate; the "
                    "native builder only supports shuffle-before-deflate. Use the "
                    "kerchunk backend for this file."
                )
            elementsize = dataset.dtype.itemsize
            filters.append({"id": "shuffle", "elementsize": int(elementsize)})
        elif filter_id == _H5_FLETCHER32:
            continue
        else:
            raise ValueError(
                f"Dataset {dataset.name!r} uses unsupported HDF5 filter id "
                f"{filter_id}; native kerchunk builder handles deflate/shuffle "
                "only. Open an issue or use the kerchunk backend for this file."
            )
    return compressor, (filters or None)


def _chunk_grid_key(chunk_offset: tuple[int, ...], chunks: list[int]) -> str:
    """Zarr chunk key (``"0.1.0"``) from element-space chunk offsets."""
    if not chunks:
        key = "0"
    else:
        key = ".".join(
            str(offset // size) for offset, size in zip(chunk_offset, chunks)
        )
    return key


def _emit_chunk_refs(
    refs: dict[str, Any],
    *,
    dataset: Any,
    name: str,
    src_url: str,
    src_handle: Any,
    chunks: list[int],
    inline_threshold: int,
) -> None:
    """Add this dataset's chunk references (byte-range or inlined) to ``refs``.

    ``src_handle`` is an open binary handle to the local source (or ``None`` for a
    remote source, which is always referenced by byte range, never inlined).
    """

    def add(key: str, offset: int, size: int) -> None:
        if size <= 0:
            return
        if src_handle is not None and size < inline_threshold:
            src_handle.seek(offset)
            blob = src_handle.read(size)
            refs[f"{name}/{key}"] = "base64:" + base64.b64encode(blob).decode("ascii")
        else:
            refs[f"{name}/{key}"] = [src_url, int(offset), int(size)]

    if dataset.chunks is None:
        offset = dataset.id.get_offset()
        size = dataset.id.get_storage_size()
        if offset is not None and size:
            zero_key = "0" if not chunks else ".".join("0" for _ in chunks)
            add(zero_key, offset, size)
        return

    num_chunks = dataset.id.get_num_chunks()
    for index in range(num_chunks):
        info = dataset.id.get_chunk_info(index)
        key = _chunk_grid_key(tuple(info.chunk_offset), chunks)
        add(key, info.byte_offset, info.size)


def _emit_dataset(
    refs: dict[str, Any],
    *,
    h5file: Any,
    name: str,
    dataset: Any,
    src_url: str,
    src_handle: Any,
    inline_threshold: int,
    vlen_encode: str,
) -> None:
    """Emit ``.zarray`` + ``.zattrs`` + chunk refs for one HDF5 dataset."""
    shape = list(dataset.shape)
    if dataset.chunks is not None:
        chunks = list(dataset.chunks)
    else:
        chunks = list(shape)
    if dataset.dtype.kind == "O":
        raise ValueError(
            f"Dataset {name!r} has an object/vlen dtype; native builder does not "
            f"yet support vlen_encode={vlen_encode!r}. Use the kerchunk backend."
        )
    compressor, filters = _compressor_and_filters(dataset)
    zarray = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dataset.dtype.str,
        "fill_value": _encode_fill_value(dataset),
        "order": "C",
        "filters": filters,
        "dimension_separator": ".",
        "compressor": compressor,
        "zarr_format": 2,
    }
    zattrs = _clean_attrs(dataset.attrs)
    zattrs["_ARRAY_DIMENSIONS"] = _array_dimensions(h5file, dataset)

    refs[f"{name}/.zarray"] = json.dumps(zarray, allow_nan=False)
    refs[f"{name}/.zattrs"] = json.dumps(zattrs, allow_nan=False)
    _emit_chunk_refs(
        refs,
        dataset=dataset,
        name=name,
        src_url=src_url,
        src_handle=src_handle,
        chunks=chunks,
        inline_threshold=inline_threshold,
    )


def build_single_manifest(
    src_path: str | Path,
    *,
    src_url: str | None = None,
    inline_threshold: int = 500,
    vlen_encode: str = "embed",
) -> dict[str, Any]:
    """Build a kerchunk v1 reference manifest for one NetCDF4 / HDF5 file.

    Args:
        src_path: Local path used to read chunk byte ranges and HDF5 metadata.
        src_url: URL written into byte-range references. Defaults to
            ``str(src_path)`` so the manifest points back at the source as
            named (matching kerchunk's behaviour for local paths).
        inline_threshold: Chunks smaller than this many bytes are embedded
            directly (base64) rather than referenced by offset.
        vlen_encode: VLEN string handling mode (reserved; object dtypes are
            currently rejected with a clear error).

    Returns:
        The manifest dict ``{"version": 1, "refs": {...}}``.

    Raises:
        ImportError: When h5py is not installed.
        ValueError: When the file uses an unsupported HDF5 feature.
    """
    h5py = _require_h5py()
    src_str = str(src_path)
    src_local = src_str if os.path.exists(src_str) else None
    if src_url is not None:
        url = src_url
    elif src_local is not None:
        # Absolutise local paths so the manifest resolves regardless of the
        # consumer's working directory (a relative path only resolves from the CWD
        # the manifest was written in).
        url = os.path.abspath(src_str)
    else:
        url = src_str

    refs: dict[str, Any] = {}
    # One binary handle for the whole walk so inlined chunks don't reopen the file
    # per chunk; None for a remote source (those are always referenced, not inlined).
    src_handle = open(src_local, "rb") if src_local is not None else None
    try:
        with h5py.File(src_str, "r") as h5file:
            refs[".zgroup"] = json.dumps({"zarr_format": 2})
            refs[".zattrs"] = json.dumps(_clean_attrs(h5file.attrs), allow_nan=False)

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    _emit_dataset(
                        refs,
                        h5file=h5file,
                        name=name,
                        dataset=obj,
                        src_url=url,
                        src_handle=src_handle,
                        inline_threshold=inline_threshold,
                        vlen_encode=vlen_encode,
                    )
                else:
                    refs[f"{name}/.zgroup"] = json.dumps({"zarr_format": 2})
                    refs[f"{name}/.zattrs"] = json.dumps(
                        _clean_attrs(obj.attrs), allow_nan=False
                    )

            h5file.visititems(visit)
    finally:
        if src_handle is not None:
            src_handle.close()

    return {"version": 1, "refs": refs}


def _manifest_refs(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the flat refs mapping from a v0 (flat) or v1 (nested) manifest."""
    return manifest["refs"] if "refs" in manifest else manifest


def _variable_names(refs: dict[str, Any]) -> list[str]:
    """Names of the array variables in a refs mapping (those with a ``.zarray``)."""
    return [key[: -len("/.zarray")] for key in refs if key.endswith("/.zarray")]


def _shift_chunk_key(key: str, axis: int, offset: int) -> str:
    """Shift a zarr chunk key's grid index on ``axis`` by ``offset`` chunks."""
    parts = key.split(".")
    parts[axis] = str(int(parts[axis]) + offset)
    return ".".join(parts)


def _concat_variable(
    name: str,
    per_file_refs: list[dict[str, Any]],
    axis: int,
) -> dict[str, Any]:
    """Stack one variable across files along ``axis``; return its merged refs."""
    base = json.loads(per_file_refs[0][f"{name}/.zarray"])
    chunk_size = base["chunks"][axis]
    total = 0
    prior_chunks = 0
    merged: dict[str, Any] = {}
    for index, refs in enumerate(per_file_refs):
        zarray = json.loads(refs[f"{name}/.zarray"])
        length = zarray["shape"][axis]
        if zarray["chunks"][axis] != chunk_size:
            raise ValueError(
                f"variable {name!r} has inconsistent chunk size on the concat "
                f"axis across files ({zarray['chunks'][axis]} vs {chunk_size}); "
                "cannot concatenate into a uniform-chunk zarr array."
            )
        is_last = index == len(per_file_refs) - 1
        if length % chunk_size and not is_last:
            raise ValueError(
                f"variable {name!r} length {length} on the concat axis is not a "
                f"multiple of its chunk size {chunk_size} in a non-final file; "
                "cannot concatenate without rechunking."
            )
        prefix = f"{name}/"
        for key, value in refs.items():
            if not key.startswith(prefix) or key.endswith(
                (".zarray", ".zattrs", ".zgroup")
            ):
                continue
            chunk_key = key[len(prefix) :]
            merged[f"{prefix}{_shift_chunk_key(chunk_key, axis, prior_chunks)}"] = value
        total += length
        prior_chunks += -(-length // chunk_size)  # ceil division
    base["shape"][axis] = total
    merged[f"{name}/.zarray"] = json.dumps(base)
    merged[f"{name}/.zattrs"] = per_file_refs[0][f"{name}/.zattrs"]
    return merged


def _identical_variable(
    name: str, per_file_refs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Carry a variable that is identical across files; assert metadata parity.

    Metadata (shape/dtype/chunks) must match exactly. For inlined chunks (small
    coordinate arrays) the values are also compared, and a mismatch warns: the
    first file's values are used, so genuinely misaligned inputs would otherwise
    be merged silently.
    """
    prefix = f"{name}/"
    base = json.loads(per_file_refs[0][f"{name}/.zarray"])
    base_chunks = {
        key: value
        for key, value in per_file_refs[0].items()
        if key.startswith(prefix) and not key.endswith((".zarray", ".zattrs", ".zgroup"))
    }
    for refs in per_file_refs[1:]:
        other = json.loads(refs[f"{name}/.zarray"])
        if (other["shape"], other["dtype"], other["chunks"]) != (
            base["shape"],
            base["dtype"],
            base["chunks"],
        ):
            raise ValueError(
                f"variable {name!r} is not identical across files (shape/dtype/"
                "chunks differ) but does not carry the concat dimension; it cannot "
                "be merged. Pass it as a concat dimension or align the inputs."
            )
        for key, value in base_chunks.items():
            inlined = isinstance(value, str) and value.startswith("base64:")
            if inlined and refs.get(key) != value:
                warnings.warn(
                    f"non-concat variable {name!r} differs across files (chunk "
                    f"{key[len(prefix):]}); using the first file's values — the "
                    "inputs may not be co-registered.",
                    stacklevel=3,
                )
                break
    return {key: value for key, value in per_file_refs[0].items() if key.startswith(prefix)}


def combine_manifests(
    per_file: list[dict[str, Any]],
    *,
    concat_dim: str,
) -> dict[str, Any]:
    """Concatenate per-file manifests along ``concat_dim`` into one manifest.

    Every variable whose ``_ARRAY_DIMENSIONS`` contains ``concat_dim`` is stacked
    along that axis (its concat-axis chunk indices are renumbered and its
    ``shape`` summed); every other variable must be identical across files and is
    carried from the first. No zarr group is created, so this cannot hit the
    zarr-v3 ``sync()`` deadlock (#530).

    Args:
        per_file: Per-file manifests, in concatenation order, as returned by
            :func:`build_single_manifest`.
        concat_dim: Dimension name to concatenate along.

    Returns:
        The combined manifest dict ``{"version": 1, "refs": {...}}``.

    Raises:
        ValueError: When ``per_file`` is empty, files disagree on the variable
            set, a concat variable has inconsistent chunking, or a non-concat
            variable is not identical across files.
    """
    if not per_file:
        raise ValueError("combine_manifests requires at least one manifest.")
    refs_list = [_manifest_refs(m) for m in per_file]
    if len(refs_list) == 1:
        return {"version": 1, "refs": dict(refs_list[0])}

    var_sets = [set(_variable_names(refs)) for refs in refs_list]
    if any(names != var_sets[0] for names in var_sets[1:]):
        raise ValueError(
            "cannot combine manifests with differing variable sets: "
            f"{[sorted(names) for names in var_sets]}"
        )

    combined: dict[str, Any] = {}
    for key, value in refs_list[0].items():
        if "/" not in key:  # group-level .zgroup / .zattrs
            combined[key] = value

    for name in _variable_names(refs_list[0]):
        dims = json.loads(refs_list[0][f"{name}/.zattrs"]).get("_ARRAY_DIMENSIONS", [])
        if concat_dim in dims:
            combined.update(_concat_variable(name, refs_list, dims.index(concat_dim)))
        else:
            combined.update(_identical_variable(name, refs_list))

    return {"version": 1, "refs": combined}


__all__ = ["build_single_manifest", "combine_manifests"]
