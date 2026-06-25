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

from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import Dataset
from pyramids.dataset.engines._base import _Engine
from pyramids.netcdf.cf import write_attributes_to_md_array

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Variables(_Engine):
    """Variable add/remove/rename/write collaborator for ``NetCDF``."""

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

        # Auto-detect from tracked origin metadata (RT-4)
        if band_dim_name is None and hasattr(dataset, "_band_dim_name"):
            band_dim_name = dataset._band_dim_name
        if band_dim_values is None and hasattr(dataset, "_band_dim_values"):
            band_dim_values = dataset._band_dim_values
        if attrs is None and hasattr(dataset, "_variable_attrs"):
            attrs = dataset._variable_attrs
        # Multi-band-dim metadata: every non-spatial dim and its
        # coords / sizes. Populated by `get_variable` for 4-D+
        # variables. When present, the rebuild materialises each dim
        # separately instead of flattening to a single bands axis.
        band_dim_names: tuple[str, ...] = tuple(
            getattr(dataset, "_band_dim_names", ()) or ()
        )
        band_dim_sizes: tuple[int, ...] = tuple(
            getattr(dataset, "_band_dim_sizes", ()) or ()
        )
        band_dim_values_map: dict = dict(
            getattr(dataset, "_band_dim_values_map", {}) or {}
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

        # 4-D+ rebuild: reshape the flattened bands back into the
        # cached storage order, then create one GDAL dimension per
        # non-spatial axis. Falls through to the legacy single-dim
        # path when only one (or zero) band dim is tracked.
        if len(band_dim_names) > 1 and arr.ndim == 3 and band_dim_sizes:
            arr = arr.reshape(*band_dim_sizes, arr.shape[-2], arr.shape[-1])
            gdal_band_dims = []
            for i, dim_name in enumerate(band_dim_names):
                values = band_dim_values_map.get(dim_name)
                if values is None:
                    values = list(range(int(band_dim_sizes[i])))
                gdal_band_dims.append(
                    nc._get_or_create_dimension(
                        rg,
                        dim_name,
                        np.array(values, dtype=np.float64),
                        coord_dtype,
                        gdal.DIM_TYPE_TEMPORAL if i == 0 else None,
                    )
                )
            md_arr = rg.CreateMDArray(
                variable_name, [*gdal_band_dims, dim_y, dim_x], data_dtype
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

        # Write array data
        md_arr.Write(arr)

        # Set spatial reference (RT-7: attribute copying)
        if dataset.epsg:
            srs = sr_from_epsg(dataset.epsg)
            md_arr.SetSpatialRef(srs)

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
                f"Variable '{old_name}' not found. " f"Available: {nc.variable_names}"
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
