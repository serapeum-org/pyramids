"""xarray interoperability engine for :class:`pyramids.netcdf.NetCDF`.

Owns the bodies of ``NetCDF.to_xarray`` / ``NetCDF.from_xarray``,
extracted from the ``netcdf.py`` god-object (issue #615, STR-1). The
public ``NetCDF`` methods are thin façades that delegate here — the
conversion goes entirely through GDAL's Multidimensional API (the same
reader/writer the rest of pyramids' NetCDF code uses), so no xarray
NetCDF backend plugin is involved. Behaviour, signatures, and return
types are unchanged by the extraction.
"""

from __future__ import annotations

import os
import tempfile
import traceback
import weakref
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.remote import is_remote
from pyramids.dataset.engines._base import _Engine
from pyramids.netcdf._lazy import build_lazy_array
from pyramids.netcdf._mdim import strip_netcdf_subdataset_prefix
from pyramids.netcdf.cf import write_attributes_to_md_array, write_global_attributes
from pyramids.netcdf.utils import _read_attributes

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Interop(_Engine["NetCDF"]):
    """xarray ↔ pyramids NetCDF conversion collaborator.

    Holds the body of :meth:`NetCDF.to_xarray`. The companion
    :meth:`NetCDF.from_xarray` is a classmethod (it builds a new
    container rather than operating on an existing instance), so its
    body lives in the module-level :func:`from_xarray` function rather
    than on this instance-bound engine.
    """

    def to_xarray(self, chunks: dict | str | int | None = None) -> Any:
        """Convert this NetCDF container to an `xarray.Dataset`.

        Builds an `xarray.Dataset` that mirrors the variables,
        coordinates, dimensions, and global attributes of this pyramids
        NetCDF container.

        The entire conversion goes through GDAL's Multidimensional
        API — the same reader the rest of pyramids' NetCDF code uses.
        No xarray NetCDF engine plugin is involved — pyramids is the
        writer, so xarray does not need to pull a NetCDF backend.

        With the default `chunks=None` the returned `xr.Dataset` holds
        already-materialised numpy arrays. Pass `chunks` (a dict / int /
        `"auto"`) to build each data variable as a lazy dask-backed
        `DataArray` in the file's native axis order, so the dataset is
        assembled without loading every variable into RAM (ARC-48).
        Lazy reads need the optional `[lazy]` (dask) extra and a
        file-backed container; an in-memory container ignores `chunks`
        (its data is already resident).

        Requires the optional `xarray` package. Install with one of:

        - PyPI: ``pip install xarray``
        - conda-forge: ``conda install -c conda-forge xarray``

        Args:
            chunks: Chunk spec forwarded to the lazy reader per data
                variable. `None` (default) reads eagerly.

        Returns:
            xarray.Dataset: An xarray Dataset with the same
            variables, coordinates, and global attributes.

        Raises:
            pyramids.base._errors.OptionalPackageDoesNotExist:
                If `xarray` is not installed.
            ImportError: If `chunks` is given but the `[lazy]` (dask)
                extra is not installed.
            ValueError: If the underlying GDAL handle is not a
                multidimensional container (open the file with
                `open_as_multi_dimensional=True`).

        Examples:
            Convert a pyramids NetCDF to xarray::

                nc = NetCDF.read_file("temperature.nc")
                ds = nc.to_xarray()
                print(ds)

            Build a lazy (dask-backed) dataset::

                lazy = nc.to_xarray(chunks="auto")
                lazy["temperature"].data  # dask.array.Array
        """
        ds = self._ds
        try:
            import xarray as xr
        except ImportError:
            raise OptionalPackageDoesNotExist(
                "xarray is required for to_xarray(). Install with one of:\n"
                "  - PyPI:        pip install xarray\n"
                "  - conda-forge: conda install -c conda-forge xarray"
            )

        rg = ds._working_group()
        if rg is None:
            raise ValueError(
                "to_xarray requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        return xr.Dataset(
            data_vars=_data_vars_from_arrays(rg, ds, chunks),
            coords=_coords_from_dimensions(rg, ds),
            attrs=ds.global_attributes,
        )


def _merge_unit(attrs: dict[str, Any], gdal_obj: Any) -> dict[str, Any]:
    """Fold GDAL's ``GetUnit()`` back into ``attrs`` as a CF ``units`` entry.

    GDAL's netCDF driver normalises the CF ``units`` attribute onto the
    MDArray/indexing-variable unit slot rather than a regular attribute. Merge
    it back so the value round-trips through ``xr.Dataset``. Existing ``units``
    in ``attrs`` win; the dict is mutated in place and returned for chaining.
    """
    unit = gdal_obj.GetUnit()
    if unit and "units" not in attrs:
        attrs["units"] = unit
    return attrs


def _coords_from_dimensions(rg: Any, ds: NetCDF) -> dict[str, Any]:
    """Build the ``xr.Dataset`` ``coords`` mapping from the root group's dimensions.

    Each dimension with an indexing variable becomes a 1-D coordinate; bare
    dimensions (no indexing variable) are skipped.
    """
    coords: dict[str, Any] = {}
    for d in rg.GetDimensions() or []:
        iv = d.GetIndexingVariable()
        if iv is None:
            continue
        dim_name = d.GetName()
        coord_attrs = _merge_unit(_read_attributes(iv), iv)
        coords[dim_name] = ([dim_name], ds._md_array_to_numpy(iv), coord_attrs)
    return coords


def _data_vars_from_arrays(rg: Any, ds: NetCDF, chunks: Any = None) -> dict[str, Any]:
    """Build the ``xr.Dataset`` ``data_vars`` mapping from the container's variables.

    With ``chunks=None`` each variable is read eagerly; otherwise it is read lazily (dask) in the
    file's native axis order via :func:`_lazy_var_data` (ARC-48).
    """
    data_vars: dict[str, Any] = {}
    for var_name in ds.variable_names:
        md_arr = rg.OpenMDArray(var_name)
        if md_arr is None:
            continue
        arr_dim_names = [ad.GetName() for ad in md_arr.GetDimensions() or []]
        if chunks is None:
            arr_data = ds._md_array_to_numpy(md_arr)
        else:
            arr_data = _lazy_var_data(ds, var_name, chunks, md_arr)
        var_attrs = _merge_unit(_read_attributes(md_arr), md_arr)
        data_vars[var_name] = (arr_dim_names, arr_data, var_attrs)
    return data_vars


def _reopenable_path(ds: NetCDF) -> str | None:
    """Return the bare, reopenable file path of a file-backed container, else ``None``.

    Strips any ``NETCDF:"…":var`` subdataset prefix and confirms the path exists on disk or is a
    remote (`/vsi*` / cloud) URL; an in-memory container has no such path.
    """
    path = strip_netcdf_subdataset_prefix(getattr(ds, "_file_name", "") or "")
    if path and (os.path.isfile(path) or is_remote(path)):
        return path
    return None


def _lazy_var_data(ds: NetCDF, var_name: str, chunks: Any, md_arr: Any) -> Any:
    """Return a dask-backed raw read of ``var_name`` in the file's native axis order (ARC-48).

    Deferred counterpart of :meth:`NetCDF._md_array_to_numpy`, so ``to_xarray(chunks=...)`` assembles
    a lazy dataset without materialising every variable. A file-backed container reads through
    :func:`build_lazy_array` with ``orient=False`` (raw, matching the raw coordinate arrays); an
    in-memory container has no file to reopen and its data is already resident, so the eager array is
    returned unchanged (``chunks`` has no benefit there).

    A variable whose dtype a chunked read cannot represent (e.g. a string MDArray such as a CF
    ``expver`` label) falls back to the eager read, matching the default ``to_xarray`` path -- one
    non-chunkable variable must not fail the whole lazy conversion.

    The container's captured cloud config (``_gdal_env``) is carried into the read so each chunk
    task re-opens a signed remote store with the same credentials, matching ``_read_array_lazy`` (#839).
    """
    path = _reopenable_path(ds)
    if path is None:
        return ds._md_array_to_numpy(md_arr)
    try:
        return build_lazy_array(
            path, var_name, chunks, orient=False, gdal_env=ds._gdal_env or None
        )
    except ValueError:
        return ds._md_array_to_numpy(md_arr)


def from_xarray(
    cls: type[NetCDF],
    dataset: Any,
    path: str | Path | None = None,
) -> NetCDF:
    """Create a pyramids NetCDF from an `xarray.Dataset`.

    Extracts dimensions, coordinates, data variables, and
    attributes from the `xarray.Dataset` and writes them to a
    NetCDF file through pyramids' own GDAL Multidimensional
    writer. No xarray NetCDF engine plugin is involved — pyramids
    is the writer, so xarray does not need to pull a NetCDF backend.

    Usage::

        ds = xr.open_dataset("input.nc")
        #... xarray processing...
        nc = NetCDF.from_xarray(ds)
        var = nc.get_variable("temperature")
        cropped = var.crop(mask)

    Requires the optional `xarray` package. Install with one of:

    - PyPI: ``pip install xarray``
    - conda-forge: ``conda install -c conda-forge xarray``

    Args:
        cls: The concrete ``NetCDF`` subclass (``Container``) used to
            read the written file back in. Threaded through by the
            ``NetCDF.from_xarray`` classmethod façade.
        dataset: An `xarray.Dataset` instance.
        path: File path where the NetCDF will be written. If
            `None`, a temp `.nc` is created and cleaned up
            when the returned object is garbage-collected.

    Returns:
        NetCDF: A pyramids NetCDF container backed by the data
        from the xarray Dataset.

    Raises:
        pyramids.base._errors.OptionalPackageDoesNotExist:
            If `xarray` is not installed.
        TypeError: If *dataset* is not an `xarray.Dataset`.
    """
    try:
        import xarray as xr
    except ImportError:
        raise OptionalPackageDoesNotExist(
            "xarray is required for from_xarray(). Install with one of:\n"
            "  - PyPI:        pip install xarray\n"
            "  - conda-forge: conda install -c conda-forge xarray"
        )

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Expected xarray.Dataset, got {type(dataset).__name__}")

    cleanup_temp = False
    if path is not None:
        path = str(path)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        path = tmp.name
        tmp.close()
        cleanup_temp = True

    mem_src = _build_multidim_from_xarray(dataset)
    _create_copy_to_netcdf(mem_src, path)
    mem_src = None

    result = cls.read_file(path, read_only=True)
    if cleanup_temp:
        result._interop_temp_path = path
        weakref.finalize(result, os.unlink, path)
    return result


def _encode_temporal_array(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode datetime64/timedelta64 arrays to CF-numeric seconds (GDAL has no datetime dtype).

    A CF-decoded xarray time axis is `datetime64[ns]`, which `numpy_to_gdal_dtype` cannot map, so
    `from_xarray` crashed on any Dataset opened with the default `decode_cf=True`. Encode such arrays to
    float64 seconds with a CF `units` (and `calendar` for absolute times) so the MDArray stores a
    numeric axis that CF-decodes back to the same instants (ARC-17). A `NaT` maps to `NaN` (a missing
    instant), not the int64 sentinel's bogus ~year-1677 value.

    The CF-portable `seconds since ...` unit is float64, so a timestamp far from the 1970 epoch carries
    only ~sub-microsecond precision (float64 has ~0.35 µs resolution near 2e9 s); an exact nanosecond
    round-trip would need a non-portable `nanoseconds since ...` unit.

    Args:
        values: The raw coordinate / variable array.

    Returns:
        A `(encoded_values, cf_attrs)` pair. For a non-temporal array the values are returned unchanged
        with an empty attribute dict; for a temporal array the values are float64 seconds (`NaN` where
        the input was `NaT`) and `cf_attrs` carries the CF `units` (plus `calendar` for absolute
        datetimes).
    """
    if np.issubdtype(values.dtype, np.datetime64):
        as_ns = values.astype("datetime64[ns]")
        seconds = as_ns.astype("int64").astype("float64") / 1e9
        # `np.where` keeps this scalar-safe: a 0-d input's `/ 1e9` is a NumPy scalar that does not
        # support in-place item assignment (review round-2 M1).
        seconds = np.where(np.isnat(as_ns), np.nan, seconds)
        return seconds, {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    if np.issubdtype(values.dtype, np.timedelta64):
        as_ns = values.astype("timedelta64[ns]")
        seconds = as_ns.astype("int64").astype("float64") / 1e9
        seconds = np.where(np.isnat(as_ns), np.nan, seconds)
        return seconds, {"units": "seconds"}
    return values, {}


def _apply_md_array_attrs(md_arr: gdal.MDArray, attrs: dict[str, Any]) -> None:
    """Write MDArray attributes, routing the CF `units` through `SetUnit`.

    GDAL's netCDF writer moves the CF `units` attribute onto the MDArray's own
    unit slot; if we also write it as a regular attribute it is dropped on the
    next `CreateCopy`. Split it out so the round trip is lossless.

    Args:
        md_arr: The target multidimensional array.
        attrs: Attributes to attach; a `units` key is applied via `SetUnit`.
    """
    if not attrs:
        return
    remaining = dict(attrs)
    unit = remaining.pop("units", None)
    if unit is not None:
        md_arr.SetUnit(str(unit))
    if remaining:
        write_attributes_to_md_array(md_arr, remaining)


def _write_md_array_streamed(md_arr: gdal.MDArray, arr: Any) -> None:
    """Write ``arr`` into ``md_arr``, streaming a dask array one block at a time.

    A dask array is written block by block -- each block is computed, written to its
    hyperslab, then released before the next -- so a lazily-loaded variable never
    becomes fully resident (ARC-48). Any other array-like is written in one hyperslab.

    Args:
        md_arr: The full-shape destination MDArray.
        arr: A dask array (streamed block by block) or a NumPy array (written whole).
    """
    if not (hasattr(arr, "dask") and hasattr(arr, "blocks")):
        md_arr.Write(np.ascontiguousarray(np.asarray(arr)))
        return
    for block_id in np.ndindex(*arr.numblocks):
        block = np.ascontiguousarray(np.asarray(arr.blocks[block_id]))
        starts = [
            int(sum(arr.chunks[axis][: block_id[axis]])) for axis in range(arr.ndim)
        ]
        md_arr.Write(block, array_start_idx=starts, count=list(block.shape))


def _build_multidim(
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    data_vars: dict[str, tuple[tuple[str, ...], Any, dict[str, Any]]],
    global_attrs: dict[str, Any],
) -> gdal.Dataset:
    """Build an in-memory GDAL multidim container from plain arrays and attrs.

    The shared core behind `_build_multidim_from_xarray` and the GDAL-native
    NetCDF writers (e.g. `DatasetCollection.to_netcdf`) — neither needs a
    labeled-array dataset to reach pyramids' own multidimensional writer. Each
    coordinate becomes a 1-D indexing MDArray and each variable an N-D MDArray
    whose dimensions are resolved by name; `numpy` datetime/timedelta axes are
    CF-encoded on the way in and attributes go through pyramids' own CF helpers.

    Args:
        dims: Dimension name to length.
        coords: Coordinate name (which must also be a dimension) to a
            `(values, attrs)` pair. Entries whose name is not a dimension are
            skipped.
        data_vars: Variable name to a `(dimension-name tuple, values, attrs)`
            triple.
        global_attrs: Root-group (global) attributes.

    Returns:
        gdal.Dataset: An in-memory `MEM` multidimensional dataset ready to be
        handed to the netCDF driver's `CreateCopy`.

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate/variable array shape does not match its dimension sizes.
    """
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("pyramids")
    root = src.GetRootGroup()

    gdal_dims: dict[str, gdal.Dimension] = {
        name: root.CreateDimension(name, "", "", int(size))
        for name, size in dims.items()
    }

    for coord_name, (coord_values, coord_attrs) in coords.items():
        if coord_name not in gdal_dims:
            continue
        values, cf_attrs = _encode_temporal_array(np.asarray(coord_values))
        if values.shape != (dims[coord_name],):
            raise ValueError(
                f"coordinate {coord_name!r} has shape {values.shape} but its "
                f"dimension is length {dims[coord_name]}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(coord_name, [gdal_dims[coord_name]], ext)
        md_arr.Write(np.ascontiguousarray(values))
        merged = dict(coord_attrs)
        merged.update(cf_attrs)
        _apply_md_array_attrs(md_arr, merged)

    for var_name, (var_dims, var_values, var_attrs) in data_vars.items():
        unknown = [d for d in var_dims if d not in gdal_dims]
        if unknown:
            raise ValueError(
                f"variable {var_name!r} references unknown dimension(s) "
                f"{unknown} not in dims {sorted(gdal_dims)}"
            )
        # A dask-backed variable (a lazily-loaded xarray var passed through by
        # `_build_multidim_from_xarray`) is written block by block so it never becomes
        # fully resident; a NumPy variable, or a temporal one that must be CF-encoded,
        # is materialised and written in one shot (the prior behaviour).
        dtype = np.dtype(
            getattr(var_values, "dtype", None) or np.asarray(var_values).dtype
        )
        temporal = np.issubdtype(dtype, np.datetime64) or np.issubdtype(
            dtype, np.timedelta64
        )
        stream = hasattr(var_values, "dask") and var_values.ndim > 0 and not temporal
        if stream:
            values: Any = var_values
            cf_attrs: dict[str, Any] = {}
            shape = tuple(var_values.shape)
            write_dtype = dtype
        else:
            values, cf_attrs = _encode_temporal_array(np.asarray(var_values))
            shape = values.shape
            write_dtype = values.dtype
        expected = tuple(dims[d] for d in var_dims)
        if shape != expected:
            raise ValueError(
                f"variable {var_name!r} has shape {shape} but its "
                f"dimensions {tuple(var_dims)} imply {expected}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(np.dtype(write_dtype)))
        md_arr = root.CreateMDArray(var_name, [gdal_dims[d] for d in var_dims], ext)
        _write_md_array_streamed(md_arr, values)
        merged = dict(var_attrs)
        merged.update(cf_attrs)
        _apply_md_array_attrs(md_arr, merged)

    if global_attrs:
        write_global_attributes(root, dict(global_attrs))

    return src


def _build_multidim_from_xarray(dataset: Any) -> gdal.Dataset:
    """Build an in-memory GDAL multidim container from an xarray Dataset.

    Extracts the plain `(dims, coords, data_vars, attrs)` spec from the
    `xarray.Dataset` and delegates to `_build_multidim`, so the GDAL multidim
    assembly lives in one place and this adapter only reads `.sizes` /
    `.coords` / `.data_vars` / `.attrs` off xarray.

    Args:
        dataset: The source `xarray.Dataset`.

    Returns:
        gdal.Dataset: The in-memory `MEM` multidimensional dataset.
    """
    dims = {name: int(size) for name, size in dataset.sizes.items()}
    coords = {
        name: (np.asarray(coord.values), dict(coord.attrs))
        for name, coord in dataset.coords.items()
        if name in dims
    }
    data_vars = {
        # `var.data` hands the underlying array through WITHOUT computing it, so a
        # dask-backed variable stays lazy and `_build_multidim` can stream it block by
        # block (ARC-48); `.values` would force a full materialisation up front.
        name: (tuple(var.dims), var.data, dict(var.attrs))
        for name, var in dataset.data_vars.items()
    }
    return _build_multidim(dims, coords, data_vars, dict(dataset.attrs))


def _create_copy_to_netcdf(mem_src: gdal.Dataset, path: str) -> None:
    """CreateCopy an in-memory multidim dataset to a netCDF file on disk.

    Args:
        mem_src: The in-memory `MEM` multidimensional source dataset.
        path: Destination `.nc` path.

    Raises:
        RuntimeError: When the GDAL netCDF writer returns no dataset.
    """
    dst = gdal.GetDriverByName("netCDF").CreateCopy(path, mem_src, 0)
    if dst is None:
        raise RuntimeError(f"Failed to write NetCDF to {path}")
    dst.FlushCache()
    # Release the write handle here rather than relying on scope-exit GC: an
    # open netCDF write handle can leave the on-disk file unrecognised by a
    # reader that reopens the same path (e.g. from_xarray's read_file).
    dst = None


def write_multidim_netcdf(
    path: str | Path,
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]],
    global_attrs: dict[str, Any],
) -> None:
    """Write a plain multidim spec to a NetCDF file through GDAL.

    Assembles the `(dims, coords, data_vars, global_attrs)` spec into an
    in-memory GDAL multidimensional dataset via `_build_multidim` and copies it
    out with the netCDF driver — the same writer `NetCDF.from_xarray` uses, so a
    caller that already holds `numpy` arrays never has to build a
    labeled-array dataset just to emit a NetCDF.

    Args:
        path: Output `.nc` path.
        dims: Dimension name to length.
        coords: Coordinate name to a `(values, attrs)` pair.
        data_vars: Variable name to a `(dimension-name tuple, values, attrs)`
            triple.
        global_attrs: Root-group (global) attributes.

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate/variable array shape does not match its dimension sizes.
        RuntimeError: When the GDAL netCDF writer fails to create the file.
    """
    mem_src = _build_multidim(dims, coords, data_vars, global_attrs)
    _create_copy_to_netcdf(mem_src, str(path))


class _StreamingMultidimWriter:
    """Fills a netCDF multidim file's data variables one hyperslab at a time.

    Created by :func:`open_streaming_multidim_netcdf`. Each data variable is
    created at its full shape but written incrementally: :meth:`write_slab` writes
    a single index of the leading (streamed) dimension, so the whole cube is never
    resident in memory. The owning context manager finalizes the file on exit.
    """

    def __init__(self, arrays: dict[str, gdal.MDArray]) -> None:
        """Store the per-variable MDArrays to stream into.

        Args:
            arrays: Variable name to its (empty, full-shape) MDArray.
        """
        self._arrays = arrays

    def write_slab(self, var_name: str, index: int, block: np.ndarray) -> None:
        """Write one leading-dimension index of a variable.

        Args:
            var_name: Target data variable.
            index: Position along the leading (streamed) dimension to write at.
            block: The array for this index, i.e. the variable's shape with the
                leading dimension dropped. A length-1 leading axis is prepended
                before the hyperslab write.
        """
        md_arr = self._arrays[var_name]
        block = np.ascontiguousarray(block)
        md_arr.Write(
            block[np.newaxis, ...],
            array_start_idx=[int(index)] + [0] * block.ndim,
            count=[1] + list(block.shape),
        )


def _build_streaming_multidim(
    dataset: gdal.Dataset,
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]],
    global_attrs: dict[str, Any],
) -> dict[str, gdal.MDArray]:
    """Create dims, coord arrays, empty data vars, and global attrs on `dataset`.

    Split out from :func:`open_streaming_multidim_netcdf` so all the transient
    GDAL setup handles (the root group, dimensions, and coordinate MDArrays) live
    in this frame and are released when it returns or raises — leaving the context
    manager only the data-variable MDArrays to drop before `Close()`, which is
    what lets the netCDF driver flush and unlock the file.

    Args:
        dataset: A freshly created netCDF multidim dataset.
        dims: Dimension name to length.
        coords: Coordinate name to a ``(values, attrs)`` pair; entries whose name
            is not a dimension are skipped.
        var_specs: Variable name to a ``(dimension-name tuple, numpy dtype,
            attrs)`` triple.
        global_attrs: Root-group (global) attributes.

    Returns:
        dict[str, gdal.MDArray]: The created (empty, full-shape) data-variable
        MDArrays, keyed by name, for the caller to fill by slab.

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate array shape does not match its dimension size.
    """
    root = dataset.GetRootGroup()
    gdal_dims: dict[str, gdal.Dimension] = {
        name: root.CreateDimension(name, "", "", int(size))
        for name, size in dims.items()
    }

    for coord_name, (coord_values, coord_attrs) in coords.items():
        if coord_name not in gdal_dims:
            continue
        values, cf_attrs = _encode_temporal_array(np.asarray(coord_values))
        if values.shape != (dims[coord_name],):
            raise ValueError(
                f"coordinate {coord_name!r} has shape {values.shape} but its "
                f"dimension is length {dims[coord_name]}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        coord_arr = root.CreateMDArray(coord_name, [gdal_dims[coord_name]], ext)
        coord_arr.Write(np.ascontiguousarray(values))
        merged = dict(coord_attrs)
        merged.update(cf_attrs)
        _apply_md_array_attrs(coord_arr, merged)

    arrays: dict[str, gdal.MDArray] = {}
    for var_name, (var_dims, var_dtype, var_attrs) in var_specs.items():
        unknown = [d for d in var_dims if d not in gdal_dims]
        if unknown:
            raise ValueError(
                f"variable {var_name!r} references unknown dimension(s) "
                f"{unknown} not in dims {sorted(gdal_dims)}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(np.dtype(var_dtype)))
        md_arr = root.CreateMDArray(var_name, [gdal_dims[d] for d in var_dims], ext)
        _apply_md_array_attrs(md_arr, dict(var_attrs))
        arrays[var_name] = md_arr

    if global_attrs:
        write_global_attributes(root, dict(global_attrs))

    return arrays


@contextmanager
def open_streaming_multidim_netcdf(
    path: str | Path,
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]],
    global_attrs: dict[str, Any],
):
    """Create a netCDF multidim file and yield a per-hyperslab writer.

    The streaming counterpart of :func:`write_multidim_netcdf`: instead of
    receiving fully materialised variable arrays, it creates the dimensions,
    writes the (small, 1-D) coordinate arrays whole, creates each data variable
    at its full shape with **no data**, then hands back a
    :class:`_StreamingMultidimWriter` so the caller can fill each variable one
    leading-dimension slab at a time, so the whole ``(T, …)`` cube never has to
    be resident. The write is atomic: the file is created at a temporary sibling
    path and only ``os.replace``-d onto ``path`` after a clean close, so ``path``
    only ever holds a complete file and an existing file there survives a failed
    write. Variable data is written raw at its declared dtype — unlike
    :func:`write_multidim_netcdf`, ``datetime64`` / ``timedelta64`` *variable*
    data is not CF-encoded here (coordinates still are), so variable dtypes must
    be GDAL-mappable numerics (the sole caller streams numeric raster bands).
    Used by :meth:`pyramids.dataset.collection.DatasetCollection.to_netcdf`.

    Args:
        path: Output ``.nc`` path.
        dims: Dimension name to length (the full shape, including the streamed
            leading dimension).
        coords: Coordinate name (also a dimension) to a ``(values, attrs)`` pair;
            written whole up front. Entries whose name is not a dimension are
            skipped.
        var_specs: Variable name to a ``(dimension-name tuple, numpy dtype,
            attrs)`` triple. The first entry of the dimension tuple is the
            streamed (leading) dimension.
        global_attrs: Root-group (global) attributes.

    Yields:
        _StreamingMultidimWriter: Call :meth:`~_StreamingMultidimWriter.write_slab`
        once per leading-dimension index.

    Raises:
        RuntimeError: When the GDAL netCDF driver fails to create the file.
        ValueError: When a variable references an unknown dimension, or a
            coordinate array shape does not match its dimension size.
    """
    final_path = Path(path)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    with suppress(OSError):
        tmp_path.unlink()  # drop any stale temp left by a crashed prior run
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(tmp_path))
    if dataset is None:
        raise RuntimeError(f"Failed to create NetCDF at {path}")
    completed = False
    arrays: dict[str, gdal.MDArray] = {}
    try:
        arrays = _build_streaming_multidim(
            dataset, dims, coords, var_specs, global_attrs
        )
        yield _StreamingMultidimWriter(arrays)
        completed = True
    except BaseException as exc:
        # The in-flight exception's traceback also pins the caller's `writer`
        # (holding the same MDArrays); clear the failure frames so those handles
        # release and the temp can be removed below (the executing generator
        # frame is skipped by clear_frames). Deliberate trade-off: this empties
        # the failure frames' locals, so a post-mortem sees none here — do not
        # "restore" them or the file lock on Windows returns.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # `arrays` is the writer's own dict, so clearing it drops the last refs to
        # the data-variable MDArrays; GDAL only flushes and unlocks the file once
        # those child handles are gone (needed for os.replace / unlink on Windows).
        arrays.clear()
        if completed:
            # Close() drives the flush; let a flush error surface (the write did
            # not truly succeed). Then atomically promote the temp onto `path`.
            dataset.Close()
            os.replace(tmp_path, final_path)
        else:
            # Best-effort cleanup that never masks the in-flight exception and
            # never touches an existing file at `path`: release the handle, then
            # drop the temp.
            with suppress(Exception):
                dataset.Close()
            with suppress(OSError):
                tmp_path.unlink()
