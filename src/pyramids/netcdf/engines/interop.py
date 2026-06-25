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
from pyramids.dataset.engines._base import _Engine
from pyramids.netcdf.cf import write_attributes_to_md_array, write_global_attributes
from pyramids.netcdf.utils import _read_attributes

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Interop(_Engine):
    """xarray ↔ pyramids NetCDF conversion collaborator.

    Holds the body of :meth:`NetCDF.to_xarray`. The companion
    :meth:`NetCDF.from_xarray` is a classmethod (it builds a new
    container rather than operating on an existing instance), so its
    body lives in the module-level :func:`from_xarray` function rather
    than on this instance-bound engine.
    """

    def to_xarray(self) -> Any:
        """Convert this NetCDF container to an `xarray.Dataset`.

        Builds an in-memory `xarray.Dataset` that mirrors the
        variables, coordinates, dimensions, and global attributes of
        this pyramids NetCDF container.

        The entire conversion goes through GDAL's Multidimensional
        API — the same reader the rest of pyramids' NetCDF code uses.
        No xarray engine plugin (`netcdf4`, `h5netcdf`,
        `scipy.io.netcdf`) is involved, so xarray does not need to
        pull a NetCDF backend: pyramids is the backend. The returned
        `xr.Dataset` holds already-
        materialised numpy arrays; for lazy reads use
        :meth:`read_array(chunks=...)` and wrap the result in
        :class:`xarray.DataArray` yourself.

        Requires the optional `xarray` package. Install with one of:

        - PyPI: ``pip install xarray``
        - conda-forge: ``conda install -c conda-forge xarray``

        Returns:
            xarray.Dataset: An xarray Dataset with the same
            variables, coordinates, and global attributes.

        Raises:
            pyramids.base._errors.OptionalPackageDoesNotExist:
                If `xarray` is not installed.
            ValueError: If the underlying GDAL handle is not a
                multidimensional container (open the file with
                `open_as_multi_dimensional=True`).

        Examples:
            Convert a pyramids NetCDF to xarray::

                nc = NetCDF.read_file("temperature.nc")
                ds = nc.to_xarray()
                print(ds)
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

        rg = ds._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "to_xarray requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        coords: dict[str, Any] = {}
        dims = rg.GetDimensions() or []
        for d in dims:
            dim_name = d.GetName()
            iv = d.GetIndexingVariable()
            if iv is None:
                continue
            coord_attrs = _read_attributes(iv)
            unit = iv.GetUnit()
            if unit and "units" not in coord_attrs:
                coord_attrs["units"] = unit
            coords[dim_name] = ([dim_name], ds._md_array_to_numpy(iv), coord_attrs)

        data_vars: dict[str, Any] = {}
        for var_name in ds.variable_names:
            md_arr = rg.OpenMDArray(var_name)
            if md_arr is None:
                continue
            arr_dims = md_arr.GetDimensions() or []
            arr_dim_names = [ad.GetName() for ad in arr_dims]
            arr_data = ds._md_array_to_numpy(md_arr)
            var_attrs = _read_attributes(md_arr)
            # GDAL's netCDF driver normalises the CF `units` attribute
            # to MDArray.GetUnit() / SetUnit() rather than a regular
            # attribute. Merge it back into var_attrs for a clean
            # round-trip through xr.Dataset.
            unit = md_arr.GetUnit()
            if unit and "units" not in var_attrs:
                var_attrs["units"] = unit
            data_vars[var_name] = (arr_dim_names, arr_data, var_attrs)

        result = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=ds.global_attributes,
        )
        return result


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
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(
            coord_name,
            [gdal_dims[coord_name]],
            ext,
        )
        md_arr.Write(np.ascontiguousarray(values))
        _apply_attrs(md_arr, dict(coord.attrs))

    for var_name, var in dataset.data_vars.items():
        values = np.asarray(var.values)
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(
            var_name,
            [gdal_dims[d] for d in var.dims],
            ext,
        )
        md_arr.Write(np.ascontiguousarray(values))
        _apply_attrs(md_arr, dict(var.attrs))

    if dataset.attrs:
        write_global_attributes(root, dict(dataset.attrs))

    return src
