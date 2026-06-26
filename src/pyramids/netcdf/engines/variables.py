"""Variable-mutation engine for :class:`pyramids.netcdf.NetCDF`.

Owns the bodies of the variable add/remove/rename/write family extracted
from the ``netcdf.py`` god-object (issue #615, STR-1):

- :meth:`Variables.set_variable` — write a classic ``Dataset`` back as an
  MDArray variable (the inverse of ``get_variable``).
- :meth:`Variables.add_variable` — copy MDArray variables from another
  container.
- :meth:`Variables.remove_variable` / :meth:`Variables.rename_variable` —
  delete / rename a variable.

The public ``NetCDF`` methods are thin façades delegating here; signatures,
behaviour, and return types are unchanged. Each method reaches the container's
own GDAL plumbing (``_writable_root_group`` / ``_replace_raster`` /
``_invalidate_caches`` / ``_get_or_create_dimension`` /
``_add_md_array_to_group``) through the weakref-proxied back-reference
``self._ds``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.dataset.engines._base import _Engine
from pyramids.netcdf.cf import (
    srs_to_grid_mapping,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf.dimensions import ClassicDimensionInfo

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Variables(_Engine):
    """Variable add / remove / rename / write collaborator for :class:`NetCDF`.

    Owns the bodies of the variable-mutation family. ``NetCDF`` wires one
    instance per container as ``nc.varops`` (deliberately not ``variables`` —
    that name is the read-side property returning the lazy variable dict) and
    exposes thin façades, so ``nc.set_variable(...)`` and
    ``nc.varops.set_variable(...)`` are equivalent. The companion constructor
    :func:`create_from_array` is a module-level function (it builds a new
    container rather than mutating an existing one), reached through the
    ``NetCDF.create_from_array`` classmethod façade.

    Each method reaches the container's GDAL plumbing
    (``_writable_root_group`` / ``_replace_raster`` / ``_invalidate_caches`` /
    ``_get_or_create_dimension`` / ``_add_md_array_to_group``) through the
    weakref-proxied back-reference :attr:`_ds` inherited from
    :class:`~pyramids.dataset.engines._base._Engine`.
    """

    def set_variable(
        self,
        variable_name: str,
        dataset: Dataset,
        band_dim_name: str | None = None,
        band_dim_values: list | None = None,
        attrs: dict | None = None,
    ):
        """Write a classic Dataset back as an MDArray variable in this container.

        This is the reverse of `get_variable()`. After performing GIS
        operations (crop, reproject, etc.) on a variable subset, use this
        method to store the result back into the NetCDF container.

        Args:
            variable_name: Name for the variable in this container. If a
                variable with this name already exists it is replaced.
            dataset: A classic raster dataset, typically the result of a
                GIS operation on a variable obtained via `get_variable()`.
            band_dim_name: Name of the dimension that maps to bands
                (e.g. `"time"`, `"bands"`). Auto-detected from the
                dataset's `_band_dim_name` attribute when available.
                Defaults to None.
            band_dim_values: Coordinate values for the band dimension.
                Auto-detected from `_band_dim_values` when available.
                Defaults to None.
            attrs: Variable attributes to set (e.g. `{"units": "K"}`).
                Auto-detected from `_variable_attrs` when available.
                Defaults to None.

        Raises:
            ValueError: If called on a dataset without a root group
                (not opened in multidimensional mode).
        """
        nc = self._ds
        rg = nc._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )
        # CreateMDArray / DeleteMDArray / CreateDimension are rejected on a file-backed group (netCDF
        # data mode); operate on a writable MEM copy and swap it in, like remove_variable (#587).
        if nc.driver_type != "memory":
            work, rg = nc._writable_root_group()
            nc._replace_raster(work)

        band_dim_name, band_dim_values, attrs, band = _resolve_band_metadata(
            dataset, band_dim_name, band_dim_values, attrs
        )

        # Delete existing variable if present
        if variable_name in nc.variable_names:
            rg.DeleteMDArray(variable_name)

        # Read data from the classic dataset
        arr = dataset.read_array()
        gt: tuple[float, float, float, float, float, float] = dataset.geotransform
        data_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        # Coordinate dimensions must always be float64 to avoid truncation
        # when the data array is integer (e.g., classified rasters).
        coord_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Build spatial dimensions from the geotransform
        x_values = np.array(
            nc.get_x_lon_dimension_array(gt[0], gt[1], dataset.columns)
        )
        y_values = np.array(
            nc.get_y_lat_dimension_array(gt[3], abs(gt[5]), dataset.rows)
        )
        dim_x = nc._get_or_create_dimension(
            rg, "x", x_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_X
        )
        dim_y = nc._get_or_create_dimension(
            rg, "y", y_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_Y
        )

        md_arr = _build_variable_mdarray(
            nc,
            rg,
            variable_name,
            arr,
            dim_y,
            dim_x,
            data_dtype,
            coord_dtype,
            band_dim_name,
            band_dim_values,
            band,
        )

        # Set spatial reference (RT-7: attribute copying)
        if dataset.epsg:
            md_arr.SetSpatialRef(sr_from_epsg(dataset.epsg))

        # Set no-data value
        if dataset.no_data_value and dataset.no_data_value[0] is not None:
            try:
                md_arr.SetNoDataValueDouble(float(dataset.no_data_value[0]))
            except (RuntimeError, TypeError, ValueError):
                pass  # nosec B110

        # Set variable attributes (RT-7)
        if attrs:
            write_attributes_to_md_array(md_arr, attrs)

        nc._invalidate_caches()

    def add_variable(
        self, dataset: Dataset | NetCDF, variable_name: str | None = None
    ):
        """Copy MDArray variables from another NetCDF into this container.

        Args:
            dataset: Source NetCDF dataset whose variables will be copied.
                Must have a root group (opened in MDIM mode).
            variable_name: Specific variable name(s) to copy. If None, all
                variables from the source are copied. If a variable with
                the same name already exists, it is renamed with a
                `"-new"` suffix.
        """
        # Local import breaks the netcdf.py <-> engines.variables import cycle
        # (netcdf.py imports this module at top level for wiring).
        from pyramids.netcdf.netcdf import NetCDF

        nc = self._ds
        var_rg = dataset._raster.GetRootGroup()
        names_to_copy: list[str]
        if variable_name is not None:
            names_to_copy = [variable_name]
        elif isinstance(dataset, NetCDF):
            names_to_copy = dataset.variable_names
        else:
            names_to_copy = []

        # A file-backed root group is opened in netCDF "data mode", which forbids
        # CreateMDArray; operate on a writable MEM copy and swap it in, mirroring
        # remove_variable.
        dst, dst_rg = nc._writable_root_group()

        for var in names_to_copy:
            md_arr = var_rg.OpenMDArray(var)
            # If the variable name already exists in the destination dataset,
            # use a suffixed name to avoid overwriting the original.
            existing = dst_rg.GetMDArrayNames() or []
            target_name = f"{var}-new" if var in existing else var
            nc._add_md_array_to_group(dst_rg, target_name, md_arr)

        nc._replace_raster(dst)
        nc._invalidate_caches()

    def remove_variable(self, variable_name: str):
        """Delete a variable from this container.

        If the dataset is backed by a file on disk, a MEM copy is made first
        so that the on-disk file is not modified. The internal raster
        reference is replaced with the modified copy.

        Args:
            variable_name: Name of the variable to remove.
        """
        nc = self._ds
        dst, rg = nc._writable_root_group()
        rg.DeleteMDArray(variable_name)

        nc._replace_raster(dst)

    def rename_variable(self, old_name: str, new_name: str):
        """Rename a variable in this container.

        Internally extracts the variable data and metadata, creates
        a new variable with the new name, and removes the old one.

        Args:
            old_name: Current name of the variable.
            new_name: Desired new name.

        Raises:
            ValueError: If `old_name` doesn't exist or `new_name`
                already exists.
        """
        nc = self._ds
        if old_name not in nc.variable_names:
            raise ValueError(
                f"Variable '{old_name}' not found. Available: {nc.variable_names}"
            )
        if new_name in nc.variable_names:
            raise ValueError(f"Variable '{new_name}' already exists.")

        if nc._raster.GetRootGroup() is None:
            raise ValueError("rename_variable requires a multidimensional container.")

        # CreateMDArray is rejected on a file-backed group (netCDF data mode);
        # work on a writable MEM copy and swap it in, like remove_variable.
        dst, rg = nc._writable_root_group()
        md_arr = rg.OpenMDArray(old_name)
        nc._add_md_array_to_group(rg, new_name, md_arr)
        rg.DeleteMDArray(old_name)
        nc._replace_raster(dst)
        nc._invalidate_caches()


def _resolve_band_metadata(
    dataset: Dataset,
    band_dim_name: str | None,
    band_dim_values: list | None,
    attrs: dict | None,
) -> tuple[str | None, list | None, dict | None, dict]:
    """Resolve `set_variable` band metadata, auto-detecting from the source dataset.

    Fills ``band_dim_name`` / ``band_dim_values`` / ``attrs`` from the source's
    tracked origin attributes (``_band_dim_name`` etc., set by ``get_variable``)
    when not given explicitly, and bundles the multi-band-dim metadata
    (``_band_dim_names`` / ``_band_dim_sizes`` / ``_band_dim_values_map``) into a
    ``band`` dict consumed by :func:`_build_variable_mdarray`.
    """
    if band_dim_name is None and hasattr(dataset, "_band_dim_name"):
        band_dim_name = dataset._band_dim_name
    if band_dim_values is None and hasattr(dataset, "_band_dim_values"):
        band_dim_values = dataset._band_dim_values
    if attrs is None and hasattr(dataset, "_variable_attrs"):
        attrs = dataset._variable_attrs
    band = {
        "names": tuple(getattr(dataset, "_band_dim_names", ()) or ()),
        "sizes": tuple(getattr(dataset, "_band_dim_sizes", ()) or ()),
        "values_map": dict(getattr(dataset, "_band_dim_values_map", {}) or {}),
    }
    return band_dim_name, band_dim_values, attrs, band


def _create_multi_band_dims(
    nc: NetCDF,
    rg: Any,
    names: tuple[str, ...],
    sizes: tuple[int, ...],
    values_map: dict,
    coord_dtype: Any,
) -> list:
    """Create one GDAL dimension per tracked non-spatial axis (the 4-D+ rebuild path).

    Each axis takes its coordinate values from ``values_map`` (filling integer
    indices when absent); the first axis is tagged ``DIM_TYPE_TEMPORAL``.
    """
    band_dims = []
    for i, dim_name in enumerate(names):
        values = values_map.get(dim_name)
        if values is None:
            values = list(range(int(sizes[i])))
        band_dims.append(
            nc._get_or_create_dimension(
                rg,
                dim_name,
                np.array(values, dtype=np.float64),
                coord_dtype,
                gdal.DIM_TYPE_TEMPORAL if i == 0 else None,
            )
        )
    return band_dims


def _build_variable_mdarray(
    nc: NetCDF,
    rg: Any,
    variable_name: str,
    arr: np.ndarray,
    dim_y: Any,
    dim_x: Any,
    data_dtype: Any,
    coord_dtype: Any,
    band_dim_name: str | None,
    band_dim_values: list | None,
    band: dict,
) -> Any:
    """Create the variable MDArray with the right band dimensions and write ``arr``.

    Three layouts: a multi-band-dim 4-D+ rebuild (reshape the flattened bands
    back into storage order, one GDAL dim per non-spatial axis via
    :func:`_create_multi_band_dims`); the legacy single-band-dim 3-D path; and a
    plain 2-D ``(y, x)`` variable. Returns the written MDArray.
    """
    names, sizes, values_map = band["names"], band["sizes"], band["values_map"]
    if len(names) > 1 and arr.ndim == 3 and sizes:
        arr = arr.reshape(*sizes, arr.shape[-2], arr.shape[-1])
        band_dims = _create_multi_band_dims(nc, rg, names, sizes, values_map, coord_dtype)
        md_arr = rg.CreateMDArray(
            variable_name, [*band_dims, dim_y, dim_x], data_dtype
        )
    elif arr.ndim == 3:
        if band_dim_name is None:
            band_dim_name = "bands"
        if band_dim_values is None:
            band_dim_values = list(range(arr.shape[0]))
        dim_band = nc._get_or_create_dimension(
            rg,
            band_dim_name,
            np.array(band_dim_values, dtype=np.float64),
            coord_dtype,
            gdal.DIM_TYPE_TEMPORAL,
        )
        md_arr = rg.CreateMDArray(
            variable_name, [dim_band, dim_y, dim_x], data_dtype
        )
    else:
        md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], data_dtype)
    md_arr.Write(arr)
    return md_arr


def create_from_array(
    arr: np.ndarray,
    geo: tuple[float, float, float, float, float, float] | None = None,
    epsg: str | int = 4326,
    no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
    path: str | Path | None = None,
    variable_name: str | None = None,
    extra_dim_name: str = "time",
    extra_dim_values: list | None = None,
    extra_dims: list[tuple[str, list | None]] | None = None,
    top_left_corner: tuple[float, float] | None = None,
    cell_size: int | float | None = None,
    chunk_sizes: tuple | list | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    title: str | None = None,
    institution: str | None = None,
    source: str | None = None,
    history: str | None = None,
) -> NetCDF:
    """Create a NetCDF dataset from a NumPy array and geotransform.

    For 3-D arrays the first axis is treated as a non-spatial
    dimension (time, level, depth, etc.) whose name and coordinate
    values are controlled by `extra_dim_name` and
    `extra_dim_values`.

    For 4-D+ arrays — e.g. `(time, level, lat, lon)` — pass
    `extra_dims=[("time", time_values), ("pressure_level", level_values)]`
    in storage order. Every non-spatial dimension is then
    materialised on the resulting NetCDF, preserving the full
    layout. `extra_dims` and the legacy single-dim params
    (`extra_dim_name` / `extra_dim_values`) are mutually exclusive.

    The driver is inferred from `path`: if `path` is `None`
    the dataset is created in memory (MEM driver); if a path is
    provided the netCDF driver writes to disk.

    Args:
        arr: 2-D `(rows, cols)`, 3-D `(extra_dim, rows, cols)`, or
            4-D+ `(d_0, ..., d_{n-1}, rows, cols)` NumPy array.
        geo: Geotransform tuple `(x_min, pixel_size, rotation,
            y_max, rotation, pixel_size)`.
        epsg: EPSG code for the spatial reference.
            Defaults to 4326.
        no_data_value: Sentinel value for cells outside the
            domain. Defaults to DEFAULT_NO_DATA_VALUE.
        path: Output file path. If `None`, the dataset is
            created in memory. Defaults to None.
        variable_name: Name of the data variable in the NetCDF
            file. Defaults to `"data"`.
        extra_dim_name: Legacy single-dim path. Name of the
            non-spatial dimension for 3-D arrays (e.g. `"time"`,
            `"level"`, `"depth"`). Ignored for 2-D arrays.
            Mutually exclusive with `extra_dims`. Defaults to
            `"time"`.
        extra_dim_values: Legacy single-dim path. Coordinate values
            for the non-spatial dimension. Must have length
            `arr.shape[0]` for 3-D arrays. Mutually exclusive with
            `extra_dims`. Defaults to `[0, 1, 2,..., N-1]`.
        extra_dims: Multi-dim path. Ordered list of
            `(dim_name, values)` pairs describing every non-spatial
            dimension in storage order. `len(extra_dims)` must
            equal `arr.ndim - 2`. Each `values` is either a list of
            length `arr.shape[i]` or `None` (use integer indices
            `[0, 1, ..., size - 1]`). Mutually exclusive with
            `extra_dim_name` / `extra_dim_values`.
        top_left_corner: `(x, y)` of the top-left corner. Used
            with `cell_size` to build `geo` when `geo` is
            not provided. Defaults to None.
        cell_size: Pixel size. Used with `top_left_corner` to
            build `geo`. Defaults to None.
        chunk_sizes: Chunk sizes for the data variable as a tuple
            matching the array dimensions (e.g. `(1, 256, 256)`
            for 3-D). Only effective when writing to disk.
            Defaults to None (GDAL default chunking).
        compression: Compression algorithm name (`"DEFLATE"`,
            `"ZSTD"`, etc.). Only effective when writing to
            disk. Defaults to None (no compression).
        compression_level: Compression level (e.g. 1-9 for
            DEFLATE). Defaults to None (GDAL default).
        title: CF global attribute `title`. Short
            description of the dataset. Defaults to None.
        institution: CF global attribute `institution`.
            Where the data was produced. Defaults to None.
        source: CF global attribute `source`. How the
            data was produced. Defaults to None.
        history: CF global attribute `history`. Audit
            trail of processing steps. Defaults to None.

    Returns:
        NetCDF: The newly created NetCDF dataset.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle
    # (netcdf.py imports this module at top level for wiring). create_from_array
    # always returns a Container regardless of which NetCDF subtype the façade
    # was invoked on, sidestepping the deprecated base-NetCDF construction path.
    from pyramids.netcdf.netcdf import Container

    if geo is None and top_left_corner is not None and cell_size is not None:
        geo = (
            top_left_corner[0],
            cell_size,
            0,
            top_left_corner[1],
            0,
            -cell_size,
        )
    if geo is None:
        raise ValueError(
            "Either 'geo' or both 'top_left_corner' and "
            "'cell_size' must be provided."
        )

    rows = int(arr.shape[-2]) if arr.ndim >= 2 else 0
    cols = int(arr.shape[-1]) if arr.ndim >= 2 else 0

    # Reconcile the legacy single-dim params with the new
    # `extra_dims` list-of-pairs API. Result is a normalised list
    # of (name, values) pairs whose length equals
    # `max(arr.ndim - 2, 0)`.
    resolved_extra_dims = _resolve_extra_dims(
        arr=arr,
        extra_dim_name=extra_dim_name,
        extra_dim_values=extra_dim_values,
        extra_dims=extra_dims,
    )

    if arr.ndim == 3:
        ClassicDimensionInfo(
            name=resolved_extra_dims[0][0],
            size=arr.shape[0],
            values=resolved_extra_dims[0][1],
        )

    if variable_name is None:
        variable_name = "data"

    # Collapse the four optional CF global attributes into one mapping so the
    # builder stays within the parameter budget; only provided ones are kept.
    cf_attrs = {
        k: v
        for k, v in (
            ("title", title),
            ("institution", institution),
            ("source", source),
            ("history", history),
        )
        if v is not None
    }

    dst_ds = _create_netcdf_from_array(
        arr,
        variable_name,
        cols,
        rows,
        resolved_extra_dims,
        geo,
        epsg,
        no_data_value,
        path=path,
        chunk_sizes=chunk_sizes,
        compression=compression,
        compression_level=compression_level,
        cf_attrs=cf_attrs,
    )
    result = Container(dst_ds)

    return result


def _resolve_extra_dims(
    arr: np.ndarray,
    extra_dim_name: str,
    extra_dim_values: list | None,
    extra_dims: list[tuple[str, list | None]] | None,
) -> list[tuple[str, list]]:
    """Normalise the legacy + new extra-dim API into a single list.

    Returns an ordered list of `(dim_name, values)` pairs whose
    length equals `max(arr.ndim - 2, 0)`. Each `values` entry is a
    concrete Python list (never `None` — defaults are filled with
    integer indices `[0, 1, ..., size - 1]`).

    Args:
        arr: The data array; only its `ndim` and `shape` are read.
        extra_dim_name: Legacy single-dim name (caller-default
            `"time"`).
        extra_dim_values: Legacy single-dim values, or `None`.
        extra_dims: New multi-dim list of `(name, values)` pairs,
            or `None` for the legacy path.

    Returns:
        list[tuple[str, list]]: Normalised dim specs.

    Raises:
        ValueError: If `extra_dims` is supplied alongside
            `extra_dim_values`; if `extra_dims` length doesn't
            match `arr.ndim - 2`; or if any per-dim `values`
            length doesn't match the corresponding `arr.shape[i]`.
    """
    expected = max(arr.ndim - 2, 0)
    if extra_dims is not None:
        return _resolve_explicit_extra_dims(arr, extra_dims, extra_dim_values, expected)
    if expected == 0:
        return []
    if expected == 1:
        values = (
            list(extra_dim_values)
            if extra_dim_values is not None
            else list(range(int(arr.shape[0])))
        )
        return [(extra_dim_name, values)]
    # 4-D+ array with no `extra_dims` and no legacy values: fall
    # back to anonymous dim names and integer indices so the array
    # can still be written.
    return [(f"dim_{i}", list(range(int(arr.shape[i])))) for i in range(expected)]


def _resolve_explicit_extra_dims(
    arr: np.ndarray,
    extra_dims: list[tuple[str, list | None]],
    extra_dim_values: list | None,
    expected: int,
) -> list[tuple[str, list]]:
    """Normalise the explicit ``extra_dims`` list-of-pairs path.

    Validates it is not combined with the legacy ``extra_dim_values``, that its
    length matches the non-spatial axis count, and that each per-dim ``values``
    matches the corresponding ``arr.shape[i]`` (filling ``None`` with integer
    indices). Helper of :func:`_resolve_extra_dims`.

    Raises:
        ValueError: On mutual-exclusion, length-mismatch, or per-dim
            value-length-mismatch (see :func:`_resolve_extra_dims`).
    """
    if extra_dim_values is not None:
        raise ValueError(
            "extra_dims and extra_dim_values are mutually "
            "exclusive. Use one or the other."
        )
    if len(extra_dims) != expected:
        raise ValueError(
            f"extra_dims must have {expected} entries for a "
            f"{arr.ndim}-D array, got {len(extra_dims)}."
        )
    resolved: list[tuple[str, list]] = []
    for i, (name, values) in enumerate(extra_dims):
        if values is None:
            values = list(range(int(arr.shape[i])))
        elif len(values) != int(arr.shape[i]):
            raise ValueError(
                f"extra_dims[{i}] values length {len(values)} "
                f"does not match arr.shape[{i}]={arr.shape[i]}."
            )
        else:
            values = list(values)
        resolved.append((name, values))
    return resolved


def _build_create_options(
    chunk_sizes: tuple | list | None,
    compression: str | None,
    compression_level: int | None,
) -> list[str]:
    """Assemble GDAL MDArray creation options from the chunking/compression knobs."""
    options: list[str] = []
    if chunk_sizes is not None:
        options.append(f"BLOCKSIZE={','.join(str(s) for s in chunk_sizes)}")
    if compression is not None:
        options.append(f"COMPRESS={compression}")
    if compression_level is not None:
        options.append(f"ZLEVEL={compression_level}")
    return options


def _create_extra_dimensions(
    rg: Any,
    extra_dims: list[tuple[str, list]],
    dtype: Any,
    use_set_indexing: bool,
) -> list:
    """Create one GDAL dimension per non-spatial axis, in storage order.

    The first non-spatial dim is tagged ``DIM_TYPE_TEMPORAL`` (matching the
    legacy 3-D path); the rest are left untagged so the netCDF driver does not
    second-guess their semantics.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle.
    from pyramids.netcdf.netcdf import NetCDF

    gdal_extra_dims = []
    for i, (dim_name, dim_values) in enumerate(extra_dims):
        dim_type = gdal.DIM_TYPE_TEMPORAL if i == 0 else None
        gdal_extra_dims.append(
            NetCDF._create_dimension(
                rg, dim_name, dtype, np.array(dim_values), dim_type, use_set_indexing
            )
        )
    return gdal_extra_dims


def _write_grid_mapping(rg: Any, md_arr: Any, srse: Any) -> None:
    """Write a CF ``grid_mapping`` variable and link it from the data variable.

    Used on the MEM driver only — the netCDF driver creates its own grid mapping
    via ``SetSpatialRef``. The variable is named ``spatial_ref`` to avoid colliding
    with GDAL's automatic ``crs`` during a later ``CreateCopy`` to netCDF.
    """
    gm_name, gm_params = srs_to_grid_mapping(srse)
    gm_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
    gm_var_name = "spatial_ref"
    crs_arr = rg.CreateMDArray(gm_var_name, [], gm_dtype)
    crs_arr.Write(np.array(0, dtype=np.int32))
    gm_params["grid_mapping_name"] = gm_name
    write_attributes_to_md_array(crs_arr, gm_params)
    write_attributes_to_md_array(md_arr, {"grid_mapping": gm_var_name})


def _create_netcdf_from_array(
    arr: np.ndarray,
    variable_name: str,
    cols: int,
    rows: int,
    extra_dims: list[tuple[str, list]] | None = None,
    geo: tuple[float, float, float, float, float, float] | None = None,
    epsg: str | int | None = None,
    no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
    path: str | Path | None = None,
    chunk_sizes: tuple | list | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    cf_attrs: dict[str, str] | None = None,
) -> gdal.Dataset:
    """Build a multidimensional GDAL dataset from an array.

    The driver is inferred from `path`: `None` -> MEM (in-memory),
    otherwise the netCDF driver writes to disk.

    Args:
        arr: 2-D `(rows, cols)`, 3-D `(extra_dim, rows, cols)`, or
            4-D+ `(d_0, ..., d_{n-1}, rows, cols)` NumPy array.
        variable_name: Name of the data variable.
        cols: Number of columns.
        rows: Number of rows.
        extra_dims: Ordered list of `(dim_name, values)` pairs for
            every non-spatial dimension. Length matches
            `arr.ndim - 2`. Empty list for 2-D arrays. Pre-resolved
            by `_resolve_extra_dims` so each `values` entry is a
            concrete list.
        geo: Geotransform tuple. Defaults to None.
        epsg: EPSG code. Defaults to None.
        no_data_value: No-data sentinel. Defaults to
            DEFAULT_NO_DATA_VALUE.
        path: Output file path. If None, created in memory.
            Defaults to None.
        chunk_sizes: Chunk sizes for the variable. Defaults to
            None.
        compression: Compression algorithm. Defaults to None.
        compression_level: Compression level. Defaults to None.
        cf_attrs: Optional CF global attributes (e.g. ``title`` /
            ``institution`` / ``source`` / ``history``) to merge onto the
            root group, alongside the always-written ``Conventions``.
            Defaults to None.

    Returns:
        gdal.Dataset: The created multidimensional GDAL dataset.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle;
    # the static dimension/coordinate helpers stay on NetCDF (shared with
    # set_variable and other call sites) and are reached through the class.
    from pyramids.netcdf.netcdf import NetCDF

    if variable_name is None:
        raise ValueError("Variable_name cannot be None")
    if geo is None:
        raise ValueError("geo cannot be None")
    if epsg is None:
        raise ValueError("epsg cannot be None")

    if extra_dims is None:
        extra_dims = []
    dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
    x_dim_values = NetCDF.get_x_lon_dimension_array(geo[0], geo[1], cols)
    # Y/lat pixel height comes from geo[5] (negative), not geo[1] — using the X cell here would
    # square a non-square grid (e.g. 2° lon, 1° lat). Pass the positive height abs(geo[5]).
    y_dim_values = NetCDF.get_y_lat_dimension_array(geo[3], abs(geo[5]), rows)

    if path is not None:
        driver_type = "netCDF"
    else:
        driver_type = "MEM"
        path = "netcdf"

    src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(str(path))
    rg = src.GetRootGroup()
    write_global_attributes(rg, {"Conventions": "CF-1.8", **(cf_attrs or {})})

    # netCDF driver doesn't support SetIndexingVariable — create
    # dimension arrays manually without linking them.
    use_set_indexing = driver_type == "MEM"
    srse = sr_from_epsg(int(epsg))
    is_geographic = srse.IsGeographic() == 1

    dim_x = NetCDF._create_dimension(
        rg,
        "x",
        dtype,
        np.array(x_dim_values),
        gdal.DIM_TYPE_HORIZONTAL_X,
        use_set_indexing,
        is_geographic=is_geographic,
    )
    dim_y = NetCDF._create_dimension(
        rg,
        "y",
        dtype,
        np.array(y_dim_values),
        gdal.DIM_TYPE_HORIZONTAL_Y,
        use_set_indexing,
        is_geographic=is_geographic,
    )

    gdal_extra_dims = _create_extra_dimensions(rg, extra_dims, dtype, use_set_indexing)
    md_arr = rg.CreateMDArray(
        variable_name,
        [*gdal_extra_dims, dim_y, dim_x],
        dtype,
        _build_create_options(chunk_sizes, compression, compression_level),
    )

    # Set metadata BEFORE writing data — netCDF driver requires
    # nodata to be set before the first Write call.
    # Tolerate both scalar and per-band sequence inputs since
    # callers often pass `Dataset.no_data_value` (now a tuple)
    # straight through.
    ndv_scalar = (
        no_data_value[0]
        if isinstance(no_data_value, (list, tuple)) and no_data_value
        else no_data_value
    )
    if ndv_scalar is not None:
        md_arr.SetNoDataValueDouble(float(ndv_scalar))
    md_arr.SetSpatialRef(srse)
    md_arr.Write(arr)

    if driver_type == "MEM":
        _write_grid_mapping(rg, md_arr, srse)

    return src
