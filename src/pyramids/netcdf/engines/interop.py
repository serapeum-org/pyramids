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
import weakref
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
        No xarray engine plugin (`netcdf4`, `h5netcdf`,
        `scipy.io.netcdf`) is involved, so xarray does not need to
        pull a NetCDF backend: pyramids is the backend.

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
    """
    path = _reopenable_path(ds)
    if path is None:
        return ds._md_array_to_numpy(md_arr)
    try:
        return build_lazy_array(path, var_name, chunks, orient=False)
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
    writer. No xarray engine plugin (`netcdf4`, `h5netcdf`)
    is invoked — pyramids is the writer, so xarray does not
    need to pull a NetCDF backend.

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
    dst = gdal.GetDriverByName("netCDF").CreateCopy(path, mem_src, 0)
    if dst is None:
        raise RuntimeError(f"Failed to write NetCDF to {path}")
    dst.FlushCache()
    dst = None
    mem_src = None

    result = cls.read_file(path, read_only=True)
    if cleanup_temp:
        result._xarray_temp_path = path
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


def _build_multidim_from_xarray(dataset: Any) -> gdal.Dataset:
    """Build an in-memory GDAL multidim container from an xarray Dataset.

    Creates dimensions from `dataset.sizes`, writes each
    coordinate as a 1-D indexing MDArray, writes each data
    variable as an N-D MDArray whose dimensions are resolved by
    name. Variable and global attributes are copied via pyramids'
    own `write_attributes_to_md_array` / `write_global_attributes`
    helpers so every type the CF layer already handles (str, int,
    float, bool, list) round-trips without going through xarray's
    NetCDF writer.
    """
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("from_xarray")
    root = src.GetRootGroup()

    gdal_dims: dict[str, gdal.Dimension] = {}
    for dim_name, dim_size in dataset.sizes.items():
        gdal_dims[dim_name] = root.CreateDimension(
            dim_name,
            "",
            "",
            int(dim_size),
        )

    def _apply_attrs(md_arr: gdal.MDArray, attrs: dict[str, Any]) -> None:
        """Write xarray var attrs, routing `units` through SetUnit.

        GDAL's netCDF writer moves the CF `units` attribute onto
        the MDArray's own unit slot; if we also write it as a regular
        attribute it's dropped on the next CreateCopy. Split it out
        so the round trip is lossless.
        """
        if not attrs:
            return
        remaining = dict(attrs)
        unit = remaining.pop("units", None)
        if unit is not None:
            md_arr.SetUnit(str(unit))
        if remaining:
            write_attributes_to_md_array(md_arr, remaining)

    for coord_name, coord in dataset.coords.items():
        if coord_name not in gdal_dims:
            continue
        values = np.asarray(coord.values)
        values, cf_attrs = _encode_temporal_array(values)
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(
            coord_name,
            [gdal_dims[coord_name]],
            ext,
        )
        md_arr.Write(np.ascontiguousarray(values))
        attrs = dict(coord.attrs)
        attrs.update(cf_attrs)
        _apply_attrs(md_arr, attrs)

    for var_name, var in dataset.data_vars.items():
        values = np.asarray(var.values)
        values, cf_attrs = _encode_temporal_array(values)
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(
            var_name,
            [gdal_dims[d] for d in var.dims],
            ext,
        )
        md_arr.Write(np.ascontiguousarray(values))
        attrs = dict(var.attrs)
        attrs.update(cf_attrs)
        _apply_attrs(md_arr, attrs)

    if dataset.attrs:
        write_global_attributes(root, dict(dataset.attrs))

    return src
