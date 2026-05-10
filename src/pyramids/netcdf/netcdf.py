"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.crs import sr_from_epsg
from pyramids.base.protocols import ArrayLike
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.netcdf._kerchunk import combine_kerchunk, to_kerchunk
from pyramids.netcdf._lazy import _apply_unpack, build_lazy_array
from pyramids.netcdf._mfdataset import open_mfdataset
from pyramids.netcdf.cf import (
    build_coordinate_attrs,
    srs_to_grid_mapping,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf.dimensions import DimMetaData
from pyramids.netcdf.metadata import get_metadata
from pyramids.netcdf.models import NetCDFMetadata
from pyramids.netcdf.utils import create_time_conversion_func


class _LazyVariableDict(dict):
    """Dict that loads NetCDF variables on first access per key.

    Avoids the cost of calling `get_variable()` (which does
    `AsClassicDataset` + Y-flip) for every variable upfront.
    Only the variables actually accessed are loaded.

    Note:
        This class is **not thread-safe**. Concurrent access from
        multiple threads may cause `get_variable()` to be called
        more than once for the same key. Use external locking if
        thread-safety is required.
    """

    def __init__(self, nc: NetCDF) -> None:
        super().__init__()
        self._nc = nc
        self._names: list[str] = nc.get_variable_names()

    def __getitem__(self, key: str) -> NetCDF:
        if not dict.__contains__(self, key) and key in self._names:
            dict.__setitem__(self, key, self._nc.get_variable(key))
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> NetCDF | Any:
        if key in self._names:
            return self[key]
        return default

    def __contains__(self, key: object) -> bool:
        return key in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self):
        return iter(self._names)

    def keys(self) -> list[str]:
        return self._names

    def values(self) -> list[NetCDF]:
        return [self[k] for k in self._names]

    def items(self) -> list[tuple[str, NetCDF]]:
        return [(k, self[k]) for k in self._names]


def _reconstruct_netcdf(
    path: str,
    access: str,
    is_md_array: bool,
    is_subset: bool,
    source_var_name: str | None,
) -> NetCDF:
    """Re-open a :class:`NetCDF` from its pickle recipe tuple.

    Called by :meth:`NetCDF.__reduce__` on unpickle. Carries four
    bits of extra state beyond the base :class:`RasterBase`
    recipe so the reconstructed instance retains identity:

    * `is_md_array` — was the file opened via
      :data:`gdal.OF_MULTIDIM_RASTER` (MDIM mode) or classic mode?
    * `is_subset` — is this instance a container or a single variable?
    * `source_var_name` — when `is_subset` is True, the variable
      path to re-traverse via :meth:`NetCDF.get_variable`.

    Args:
        path: On-disk path or VSI URL to re-open.
        access: `"read_only"` opens read-only; any other value opens
            for update.
        is_md_array: Whether to pass `open_as_multi_dimensional=True`
            to :meth:`NetCDF.read_file`.
        is_subset: If True and `source_var_name` is not None, the
            rebuilt container is then drilled into via
            :meth:`NetCDF.get_variable` before return.
        source_var_name: Variable path for the subset drill-down.

    Returns:
        NetCDF: Container or variable-subset instance.
    """
    read_only = access == "read_only"
    container = NetCDF.read_file(
        path,
        read_only=read_only,
        open_as_multi_dimensional=is_md_array,
    )
    if is_subset and source_var_name is not None:
        result = container.get_variable(source_var_name)
    else:
        result = container
    return result


class NetCDF(Dataset):
    """NetCDF.

    NetCDF class is a recursive data structure or self-referential object.
    The NetCDF class contains methods to deal with NetCDF files.

    NetCDF Creation guidelines:
        https://acdguide.github.io/Governance/create/create-basics.html
    """

    def __reduce__(self):  # type: ignore[override]
        """Emit the extended recipe tuple carrying NetCDF mode flags.

        Overrides :meth:`RasterBase.__reduce__` to include
        `_is_md_array`, `_is_subset`, and `_source_var_name`,
        which are required to reconstruct a container vs a
        variable-subset with matching identity.

        For variable-subset instances the `_file_name` attribute
        reflects the subset's GDAL description, which is typically
        empty or driver-specific. We therefore fall back to the
        parent container's `_file_name` when reconstructing a
        subset.

        Raises:
            TypeError: The NetCDF has no on-disk path (empty
                `_file_name` or a `/vsimem/` path). Pickling an
                in-memory NetCDF is not supported.
        """
        path = self._file_name
        if (not path) and self._is_subset:
            parent = getattr(self, "_parent_nc", None)
            if parent is not None:
                path = parent._file_name
        if not path or path.startswith("/vsimem/"):
            raise TypeError(
                f"NetCDF has no on-disk path (file_name={self._file_name!r}); "
                "pickling an in-memory NetCDF is not supported. Call "
                ".to_file(path) first to anchor it to disk."
            )
        return (
            _reconstruct_netcdf,
            (
                path,
                self._access,
                bool(self._is_md_array),
                bool(self._is_subset),
                self._source_var_name,
            ),
        )

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        open_as_multi_dimensional: bool = True,
    ):
        """Initialize a NetCDF dataset wrapper.

        Args:
            src: A GDAL dataset handle (either classic or multidimensional).
            access: Access mode, either `"read_only"` or `"write"`.
                Defaults to `"read_only"`.
            open_as_multi_dimensional: If True the dataset was opened with
                `gdal.OF_MULTIDIM_RASTER` and supports groups, MDArrays,
                and dimensions. If False it was opened in classic raster
                mode (subdatasets, bands). Defaults to True.
        """
        super().__init__(src, access=access)
        # set the is_subset to false before retrieving the variables
        if open_as_multi_dimensional:
            self._is_md_array = True
            self._is_subset = False
        else:
            self._is_md_array = False
            self._is_subset = False
        # Caches (invalidated by _replace_raster, add_variable, remove_variable)
        self._cached_variables: dict[str, NetCDF] | None = None
        self._cached_meta_data: NetCDFMetadata | None = None
        # Origin-tracking attributes set by get_variable (RT-4)
        self._parent_nc: NetCDF | None = None
        self._source_var_name: str | None = None
        self._gdal_md_arr_ref: Any = None
        self._gdal_rg_ref: Any = None
        self._md_array_dims: list[str] = []
        self._band_dim_name: str | None = None
        self._band_dim_values: list[Any] | None = None
        self._band_dim_names: tuple[str, ...] = ()
        self._band_dim_values_map: dict[str, list[Any] | None] = {}
        self._band_dim_sizes: tuple[int, ...] = ()
        self._variable_attrs: dict[str, Any] = {}
        self._scale: float | None = None
        self._offset: float | None = None

    def _update_inplace(  # type: ignore[override]
        self, src: gdal.Dataset, access: str | None = None
    ) -> None:
        """Swap internal state, preserving NetCDF-specific attributes.

        The base `Dataset._update_inplace` rebuilds via
        `type(self)(src, access)` and overwrites `self.__dict__`.
        For a NetCDF that runs `NetCDF.__init__` with a default
        `open_as_multi_dimensional=True`, which would reset
        `_is_md_array` to True and clear every variable-subset
        attribute. This override snapshots the subset state, runs the
        base swap with the current MDIM mode, then restores the
        snapshot — so a variable subset stays a subset across
        `set_crs`, `apply(inplace=True)`, `change_no_data_value`,
        and the `epsg` setter.
        """
        preserved = {
            "_is_md_array": self._is_md_array,
            "_is_subset": self._is_subset,
            "_parent_nc": self._parent_nc,
            "_source_var_name": self._source_var_name,
            "_gdal_md_arr_ref": self._gdal_md_arr_ref,
            "_gdal_rg_ref": self._gdal_rg_ref,
            "_md_array_dims": self._md_array_dims,
            "_band_dim_name": self._band_dim_name,
            "_band_dim_values": self._band_dim_values,
            "_band_dim_names": self._band_dim_names,
            "_band_dim_values_map": self._band_dim_values_map,
            "_band_dim_sizes": self._band_dim_sizes,
            "_variable_attrs": self._variable_attrs,
            "_scale": self._scale,
            "_offset": self._offset,
        }
        new = NetCDF(
            src,
            access=access or self._access,
            open_as_multi_dimensional=self._is_md_array,
        )
        self.__dict__.update(new.__dict__)
        self.__dict__.update(preserved)
        # collaborators in `new.__dict__` point at
        # `new` via `weakref.proxy`; re-bind to a proxy of `self`
        # so callers using `self.spatial.crop(...)` after this update
        # reach the surviving instance instead of the discarded `new`.
        self_proxy = weakref.proxy(self)
        for attr in ("io", "spatial", "bands", "analysis", "cell", "vectorize", "cog"):
            collab = self.__dict__.get(attr)
            if collab is not None:
                collab._ds = self_proxy

    def __str__(self):
        """Return a human-readable summary of the NetCDF dataset."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            projection: {self.crs}
            Variables: {self.variable_names}
            Metadata: {self.meta_data}
            File: {self.file_name}
        """
        return message

    def __repr__(self):
        """__repr__."""
        return super().__repr__()

    @property
    def top_left_corner(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    def lon(self) -> np.ndarray:
        """Longitude / x-coordinate values as a 1D array.

        Looks for a variable named `"lon"` first, then `"x"`.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither `lon` nor `x` exists in the dataset.
        """
        lon = self._read_variable("lon")
        if lon is None:
            lon = self._read_variable("x")

        result: np.ndarray
        if lon is not None:
            result = lon.reshape(lon.size)
        else:
            result = super().lon
        return result

    @property
    def lat(self) -> np.ndarray:
        """Latitude / y-coordinate values as a 1D array.

        Looks for a variable named `"lat"` first, then `"y"`.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither `lat` nor `y` exists in the dataset.
        """
        lat = self._read_variable("lat")
        if lat is None:
            lat = self._read_variable("y")

        result: np.ndarray
        if lat is not None:
            result = lat.reshape(lat.size)
        else:
            result = super().lat
        return result

    @property
    def x(self) -> np.ndarray:
        """x-coordinate/longitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """y-coordinate/latitude."""
        # Y_coordinate = upper-left corner y - index * cell size - cell-size/2
        return self.lat

    @property
    def geotransform(self):
        """Geotransform.

        Computes from lon/lat coordinate arrays if available.
        Falls back to the parent GDAL GetGeoTransform() otherwise.
        """
        if self.lon is not None and self.lat is not None:
            return (
                self.lon[0] - self.cell_size / 2,
                self.cell_size,
                0,
                self.lat[0] + self.cell_size / 2,
                0,
                -self.cell_size,
            )
        return self._geotransform

    @property
    def variable_names(self) -> list[str]:
        """Names of data variables (excluding dimension coordinate arrays).

        Returns:
            list[str]: Variable names. For MDIM mode these come from
            `GetMDArrayNames()` minus dimension names; for classic mode
            from `GetSubDatasets()`.
        """
        return self.get_variable_names()

    @property
    def variables(self) -> dict[str, NetCDF]:
        """All data variables as a lazy dict of `{name: NetCDF}` subsets.

        Variables are loaded on first access per key, not all at once.
        Cached after loading; invalidated by `add_variable` /
        `remove_variable` / `set_variable`.

        Returns:
            dict[str, NetCDF]: Mapping from variable name to its subset.
        """
        if self._cached_variables is None:
            self._cached_variables = _LazyVariableDict(self)
        return self._cached_variables

    @property
    def no_data_value(self) -> tuple:
        """Per-band nodata markers as an immutable tuple.

        Returns a `tuple` so the read-only contract is explicit —
        assign through the setter to change values.
        """
        return tuple(self._no_data_value)

    @no_data_value.setter
    def no_data_value(self, value: list | tuple | np.ndarray | Number):
        """Set the no-data value that marks cells outside the domain.

        The setter only changes the `no_data_value` attribute; it does
        **not** modify the underlying cell values. Use this to align the
        attribute with whatever sentinel is already stored in the cells.
        To actually rewrite cell values, use `change_no_data_value`.

        Args:
            value: New no-data value. A scalar is broadcast to every
                band; a `list`, `tuple`, or 1-D :class:`numpy.ndarray`
                with `len == band_count` provides one value per band.
                A 0-D ndarray is treated as a scalar.

        Raises:
            ValueError: When `value` is a sequence whose length does
                not equal `band_count`, or a multi-dimensional
                ndarray (only 0-D scalars and 1-D sequences are
                accepted).
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            elif value.ndim == 1:
                value = value.tolist()
            else:
                raise ValueError(
                    f"no_data_value ndarray must be 0-D (scalar) or 1-D "
                    f"(per-band sequence); got ndim={value.ndim}"
                )
        if isinstance(value, (list, tuple)):
            if len(value) != self.band_count:
                raise ValueError(
                    f"no_data_value sequence length {len(value)} does "
                    f"not match band_count {self.band_count}"
                )
            for i, val in enumerate(value):
                self.bands._change_no_data_value_attr(i, val)
        else:
            for i in range(self.band_count):
                self.bands._change_no_data_value_attr(i, value)

    @property
    def file_name(self):
        """File path, with the `NETCDF:"path":var` prefix stripped if present.

        Returns:
            str: Clean file path without the NETCDF prefix.
        """
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    def time_stamp(self):
        """Time coordinate values parsed from the CF-compliant `time` variable.

        Returns:
            list[str] | None: Formatted time strings, or None if no time
                dimension with a `units` attribute is found.
        """
        return self.get_time_variable()

    def _check_not_container(self, operation: str):
        """Raise ValueError if this is a root MDIM container (not a variable subset)."""
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            raise ValueError(
                f"Spatial operations are not supported on the NetCDF container. "
                f"Use nc.get_variable('var_name').{operation}(...) instead."
            )

    def plot(
        self,
        variable: str | None = None,
        *,
        time: Any | None = None,
        level: Any | None = None,
        member: Any | None = None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, int] | None = None,
        x: str | None = None,
        y: str | None = None,
        coords: tuple | list | None = None,
        kind: str = "auto",
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        levels: int | list[float] | None = None,
        norm: Any | None = None,
        center: float | None = None,
        extend: str | None = None,
        add_colorbar: bool = True,
        cbar_kwargs: dict | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
        aspect: str | float | None = None,
        title: str | None = None,
        exclude_value: Any | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ):
        """Plot a 2-D slice of a NetCDF variable using xarray-aligned vocabulary.

        The public surface is shaped around **variables** and **dimensions** — `band`
        is not a NetCDF concept and has been removed from the signature. Variable
        selection is performed by name, and the 2-D slice to render is pinned via
        the label-based selectors `time=`, `level=`, `member=`, `sel=`, or `isel=`,
        which all forward to :meth:`sel`. Colour controls mirror xarray (`vmin`,
        `vmax`, `robust`, `levels`, `norm`, `center`, `extend`, `add_colorbar`,
        `cbar_kwargs`) and are forwarded to cleopatra's renderer.

        On a **root MDIM container** the `variable=` argument is required:

        ```python
        nc.plot(variable="t2m", time="2024-01-15")
        ```

        On a **variable subset** (the result of :meth:`get_variable`) `variable=` may
        be omitted or must equal the pinned variable name; otherwise the call is
        rejected, mirroring the :meth:`read_array` contract.

        Args:
            variable (str, optional):
                Name of the variable to plot. Required on the root MDIM container;
                must be `None` or equal to the pinned variable name on a subset.
                Defaults to None.
            time (Any, optional):
                Convenience label selector for the time dim — equivalent to
                `sel={"time": time}`. Forwarded to :meth:`sel`. Defaults to None.
            level (Any, optional):
                Convenience label selector for the vertical dim. Auto-detected as
                the first of `pressure_level` / `depth` / `height` / `z` present in
                `_band_dim_names` (or the primary band dim if it matches one of
                those names). Defaults to None.
            member (Any, optional):
                Convenience label selector for the ensemble / realization dim
                (`member` / `realization` / `ensemble`). Defaults to None.
            sel (dict, optional):
                Raw label-based selectors forwarded one-by-one to :meth:`sel`. Keys
                must be valid band-dim names of the variable. Defaults to None.
            isel (dict, optional):
                Positional selectors keyed by dim name. Each int is converted to
                the corresponding coord value via `_band_dim_values_map[dim]`; if
                a band dim has no coord values the int is used directly as the
                index. Defaults to None.
            x (str, optional):
                Name of the x-coordinate variable. When set together with
                `y=`, overrides auto-detection: the named coord variables
                are read off the parent NetCDF and passed to cleopatra as
                `coords=(x_arr, y_arr)`, routing the renderer to
                `pcolormesh`. Validated against `variable_names`. Defaults
                to None.
            y (str, optional):
                Name of the y-coordinate variable. Same validation and
                semantics as `x=`. Defaults to None.
            coords (tuple or list, optional):
                Explicit curvilinear `(x, y)` coordinate spec for the
                pcolormesh path. Accepts two forms:

                - A length-2 sequence of strings — each is looked up as a
                  variable name via `_read_variable` on the parent
                  container.
                - A length-2 sequence of numpy arrays — passed straight
                  through to cleopatra. Each array is 1-D (length matches
                  the data x/y axis) or 2-D matching `(rows, cols)`.

                When `coords=` is omitted, pyramids auto-detects
                curvilinear coords via the CF `coordinates` attribute on
                the variable, then via the well-known naming conventions
                (WRF `XLAT`/`XLONG`, ROMS `lat_rho`/`lon_rho`, NEMO
                `nav_lat`/`nav_lon`). When nothing matches, the renderer
                falls back to `extent=self.bbox` (imshow). Defaults to
                None.
            kind (str, optional):
                Render kind forwarded to cleopatra's `ArrayGlyph.plot`.
                One of `"auto"`, `"imshow"`, `"pcolormesh"`, `"contour"`,
                `"contourf"`. `"auto"` routes to `pcolormesh` when
                curvilinear `coords` are present, else `imshow`. Defaults
                to `"auto"`.
            cmap (str, optional):
                Matplotlib colormap name. Forwarded to cleopatra. Defaults to None.
            vmin (float, optional):
                Lower colour limit. Forwarded to cleopatra. Defaults to None.
            vmax (float, optional):
                Upper colour limit. Forwarded to cleopatra. Defaults to None.
            robust (bool, optional):
                If True, clip colour limits to the 2nd / 98th percentile
                (xarray-standard). Forwarded to cleopatra. Defaults to False.
            levels (int or list[float], optional):
                Discrete contour levels — integer count or explicit edges.
                Forwarded to cleopatra. Defaults to None.
            norm (Any, optional):
                Matplotlib `Normalize` instance. Forwarded to cleopatra. Defaults
                to None.
            center (float, optional):
                Diverging-cmap centre (e.g. `0.0` for anomalies). Forwarded to
                cleopatra. Defaults to None.
            extend (str, optional):
                Colorbar arrow style: `"neither"`, `"both"`, `"min"`, or `"max"`.
                Forwarded to cleopatra. Defaults to None.
            add_colorbar (bool, optional):
                Whether to add a colorbar. Forwarded to cleopatra. Defaults to
                True.
            cbar_kwargs (dict, optional):
                Keyword arguments forwarded to `fig.colorbar`. Defaults to None.
            ax (Any, optional):
                Existing matplotlib Axes to draw into. Defaults to None.
            figsize (tuple, optional):
                Figure size in inches. Defaults to None.
            aspect (str or float, optional):
                Axes aspect ratio. Defaults to None.
            title (str, optional):
                Plot title. Defaults to None.
            exclude_value (Any, optional):
                Pixel value to mask out before plotting. Defaults to None.
            basemap (bool or str, optional):
                If truthy, overlay an OpenStreetMap basemap (or a named
                contextily tile provider). Defaults to None.
            **kwargs:
                Additional keyword arguments forwarded to
                :meth:`Analysis.plot <pyramids.dataset.engines.Analysis.plot>`.
                The legacy `band=` kwarg is accepted here for backward
                compatibility but emits a :class:`DeprecationWarning`.

        Returns:
            ArrayGlyph: A cleopatra ``ArrayGlyph`` wrapping the rendered figure.

        Raises:
            TypeError: If any of the Sentinel-only kwargs (`rgb`,
                `surface_reflectance`, `cutoff`, `percentile`, `overview`,
                `overview_index`) is passed. Each rejection message names the
                xarray-aligned replacement.
            ValueError: If called on a root MDIM container without `variable=`,
                if `variable=` is passed on a subset and does not match the pinned
                variable name, if `x=` / `y=` reference unknown variables, or if
                the resolved selectors do not pin to a single 2-D slice.

        Examples:
            - Plot the first time step of a variable on a container. Tagged
              `+SKIP` because rendering requires the optional `[viz]` extra
              (cleopatra + matplotlib):

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> arr = np.random.rand(4, 8, 8).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
              ...     variable_name="t2m",
              ... )
              >>> cleo = nc.plot(variable="t2m", isel={"time": 0})  # doctest: +SKIP
              >>> cleo.fig  # doctest: +SKIP
              <Figure size 800x800 with 2 Axes>

              ```

            - Pick a time slice by label — the `time=` alias is
              equivalent to `sel={"time": value}`:

              ```python
              >>> cleo = nc.plot(variable="t2m", time=2)  # doctest: +SKIP

              ```

            - Pin both time and level on a 4-D `(time, pressure_level,
              lat, lon)` variable. The selectors collapse both band dims
              to a single 2-D slice — equivalent to
              `var.sel(time=12).sel(pressure_level=500)`:

              ```python
              >>> arr4d = np.random.rand(3, 2, 5, 5).astype(np.float32)
              >>> nc4d = NetCDF.create_from_array(  # doctest: +SKIP
              ...     arr=arr4d,
              ...     geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
              ...     epsg=4326,
              ...     variable_name="temperature",
              ...     extra_dims=[
              ...         ("time", [0, 6, 12]),
              ...         ("pressure_level", [1000, 500]),
              ...     ],
              ... )
              >>> cleo = nc4d.plot(  # doctest: +SKIP
              ...     variable="temperature", time=12, level=500
              ... )

              ```

            - Use an explicit `sel=` dict instead of the convenience
              aliases — keys must match the variable's band-dim names:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", sel={"time": 2}
              ... )

              ```

            - Use an `isel=` dict to address slices positionally. Each
              integer is mapped to the corresponding coord value via
              `_band_dim_values_map`:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", isel={"time": 0}
              ... )

              ```

            - All six Sentinel-only kwargs are rejected with a hint at
              the xarray-aligned replacement. These doctests are
              runnable because the kwargs gate fires before any
              cleopatra import:

              ```python
              >>> nc.plot(variable="t2m", rgb=[0, 1, 2])  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...rgb=...

              ```

              ```python
              >>> nc.plot(variable="t2m", surface_reflectance=10000)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...surface_reflectance...

              ```

              ```python
              >>> nc.plot(variable="t2m", cutoff=[0.1, 0.9])  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...cutoff...

              ```

              ```python
              >>> nc.plot(variable="t2m", percentile=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...robust=True...

              ```

              ```python
              >>> nc.plot(variable="t2m", overview=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...overview=...

              ```

              ```python
              >>> nc.plot(variable="t2m", overview_index=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              TypeError: ...overview_index=...

              ```

            - The legacy `band=` kwarg still works as an escape hatch
              but emits a :class:`DeprecationWarning`. Prefer
              `time=`/`level=`/`isel=` for new code:

              ```python
              >>> import warnings
              >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
              ...     warnings.simplefilter("always")
              ...     cleo = nc.plot(variable="t2m", band=2)
              >>> caught[0].category.__name__  # doctest: +SKIP
              'DeprecationWarning'

              ```

            - Render a WRF-style curvilinear NetCDF on its real lat/lon
              grid. With 2-D `XLAT` / `XLONG` coord variables on the
              container, pyramids auto-detects them and routes the
              renderer to `pcolormesh`. This replaces the manual
              `ax.pcolormesh(lon_0, lat_0, band_t, ...)` workaround that
              users previously had to write themselves:

              ```python
              >>> cleo = nc.plot(variable="CANWAT", kind="pcolormesh")  # doctest: +SKIP

              ```

            - Pass an explicit curvilinear coord pair by variable name —
              useful when the variable has no CF `coordinates` attribute
              and the convention does not match WRF/ROMS/NEMO:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="CANWAT", coords=("XLONG", "XLAT"),
              ... )

              ```

            - Pick a non-default render kind. `"contourf"` produces
              filled contours from the same data; `"auto"` (the default)
              picks `pcolormesh` when curvilinear coords are present,
              else falls back to `imshow`:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", kind="contourf", levels=10,
              ... )

              ```
        """
        forbidden_kwargs = {
            "rgb": (
                "NetCDF.plot() does not accept `rgb=`: NetCDF data is not RGB. "
                "Use `time=`, `level=`, `isel=`, or `band=` to select a slice."
            ),
            "surface_reflectance": (
                "NetCDF.plot() does not accept `surface_reflectance=`: "
                "`surface_reflectance` is Sentinel-only; not meaningful for NetCDF."
            ),
            "cutoff": (
                "NetCDF.plot() does not accept `cutoff=`: `cutoff` is Sentinel-only; "
                "use `vmin=`/`vmax=`/`robust=True` instead."
            ),
            "percentile": (
                "NetCDF.plot() does not accept `percentile=`: `percentile` is "
                "Sentinel-only; use `robust=True` (2nd/98th percentile, xarray-style)."
            ),
            "overview": (
                "NetCDF.plot() does not accept `overview=`: Overviews are a "
                "GeoTIFF/COG concept; not applicable to NetCDF."
            ),
            "overview_index": (
                "NetCDF.plot() does not accept `overview_index=`: Overviews are a "
                "GeoTIFF/COG concept; not applicable to NetCDF."
            ),
        }
        for name, message in forbidden_kwargs.items():
            if name in kwargs:
                raise TypeError(message)

        is_container = (
            self._is_md_array and not self._is_subset and self.band_count == 0
        )
        if is_container:
            if variable is None:
                available = self.variable_names
                raise ValueError(
                    "Plotting requires a `variable=` argument on a NetCDF "
                    f"container. Available: {available}. Or call "
                    "`nc.get_variable('name').plot(...)`."
                )
            self._validate_xy_coord_names(x, y)
            subset = self.get_variable(variable)
            return subset.plot(
                time=time,
                level=level,
                member=member,
                sel=sel,
                isel=isel,
                x=x,
                y=y,
                coords=coords,
                kind=kind,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                robust=robust,
                levels=levels,
                norm=norm,
                center=center,
                extend=extend,
                add_colorbar=add_colorbar,
                cbar_kwargs=cbar_kwargs,
                ax=ax,
                figsize=figsize,
                aspect=aspect,
                title=title,
                exclude_value=exclude_value,
                basemap=basemap,
                **kwargs,
            )

        if variable is not None and variable != self._source_var_name:
            raise ValueError(
                f"This subset is pinned to {self._source_var_name!r}; cannot "
                f"re-plot as {variable!r}. Call `plot` on the parent container."
            )

        # On a subset, validate against the parent's variable names when
        # available (the subset's own `variable_names` is empty for the
        # in-memory classic view returned by `get_variable`).
        validator_owner = self._parent_nc if self._parent_nc is not None else self
        validator_owner._validate_xy_coord_names(x, y)
        self._plot_x_coord_name = x
        self._plot_y_coord_name = y

        legacy_band = kwargs.pop("band", None)
        if legacy_band is not None:
            warnings.warn(
                "Pass `time=`/`level=`/`isel=` instead. `band=` remains supported "
                "for now as a low-level escape hatch.",
                DeprecationWarning,
                stacklevel=2,
            )

        resolved_sel: dict[str, Any] = {}
        if sel:
            for dim_name, value in sel.items():
                resolved_sel[dim_name] = value

        if time is not None:
            time_dim = self._resolve_time_dim_name()
            resolved_sel[time_dim] = time
        if level is not None:
            level_dim = self._resolve_level_dim_name()
            resolved_sel[level_dim] = level
        if member is not None:
            member_dim = self._resolve_member_dim_name()
            resolved_sel[member_dim] = member

        if isel:
            for dim_name, idx in isel.items():
                if dim_name not in self._band_dim_names:
                    raise ValueError(
                        f"isel dim {dim_name!r} is not a band dim of this "
                        f"variable {list(self._band_dim_names)!r}."
                    )
                dim_coords = self._band_dim_values_map.get(dim_name)
                if dim_coords is None:
                    resolved_sel[dim_name] = idx
                else:
                    resolved_sel[dim_name] = dim_coords[idx]

        pinned = self
        for dim_name, value in resolved_sel.items():
            pinned = pinned.sel(**{dim_name: value})

        flat_band = 0 if legacy_band is None else legacy_band
        if resolved_sel and pinned.band_count != 1:
            raise ValueError(
                f"Selectors did not pin to a single 2-D slice. Resolved: "
                f"{resolved_sel}. Remaining shape: {pinned.shape}."
            )

        analysis_kwargs: dict[str, Any] = dict(kwargs)
        forwarded_kwargs = (
            ("cmap", cmap),
            ("vmin", vmin),
            ("vmax", vmax),
            ("levels", levels),
            ("norm", norm),
            ("center", center),
            ("extend", extend),
            ("cbar_kwargs", cbar_kwargs),
            ("ax", ax),
            ("figsize", figsize),
            ("aspect", aspect),
            ("title", title),
        )
        for key, value in forwarded_kwargs:
            if value is not None:
                analysis_kwargs[key] = value
        # `robust` carries a default of False; only forward when the caller
        # explicitly enables it. `add_colorbar` is part of the xarray-aligned
        # surface but is not yet honoured by the current cleopatra release;
        # the kwarg is accepted on the signature for forward compatibility
        # but silently dropped before reaching the renderer.
        if robust:
            analysis_kwargs["robust"] = True
        _ = add_colorbar

        # Curvilinear coord resolution. Priority (highest first):
        # 1. Explicit user `x=` / `y=` (PR-2 signature).
        # 2. Explicit user `coords=` (this PR).
        # 3. CF `coordinates` attribute on the variable + well-known
        #    conventions (XLAT/XLONG, lat_rho/lon_rho, nav_lat/nav_lon).
        # When none of the above resolves to a valid coord pair the
        # engine falls back to `extent=self.bbox` (imshow).
        resolved_coord_arrays = pinned._resolve_curvilinear_coords(
            x=x, y=y, coords=coords,
        )
        if resolved_coord_arrays is not None:
            analysis_kwargs["coords"] = resolved_coord_arrays

        # `kind` is forwarded to cleopatra's `ArrayGlyph.plot(kind=...)`
        # dispatch. The default `"auto"` is harmless to forward (cleopatra
        # treats it as the default) but adding it unconditionally would
        # noise the kwargs dict; only forward non-defaults.
        if kind != "auto":
            analysis_kwargs["kind"] = kind
        elif resolved_coord_arrays is not None:
            # When the renderer has curvilinear coords but the caller
            # left `kind="auto"`, forward "auto" anyway so cleopatra can
            # see the routing decision in the kwargs trail (helps when
            # users introspect the call).
            analysis_kwargs["kind"] = "auto"

        analysis_kwargs.setdefault("rgb", None)

        return pinned.analysis.plot(
            band=flat_band,
            exclude_value=exclude_value,
            basemap=basemap,
            **analysis_kwargs,
        )

    def _validate_xy_coord_names(self, x: str | None, y: str | None) -> None:
        """Validate that `x=` / `y=` resolve to a variable of this NetCDF.

        Used by :meth:`plot` to reject typo'd coord names early. The
        validation pool is `self.variable_names`. `None` is always
        accepted — only explicitly supplied names are checked. The
        method does not return anything; it raises on failure and
        returns `None` on success.

        Args:
            x: Candidate x-coord variable name, or ``None``.
            y: Candidate y-coord variable name, or ``None``.

        Raises:
            ValueError: When `x` or `y` is given but is not a variable
                of this NetCDF. The error message lists the available
                variable names so the caller can spot typos.

        Examples:
            - `None` inputs are always accepted and the method returns
              silently:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m",
              ... )
              >>> nc._validate_xy_coord_names(None, None) is None
              True

              ```

            - An unknown coord name raises :class:`ValueError` and
              names the available variables:

              ```python
              >>> nc._validate_xy_coord_names("bogus", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: x='bogus' is not a variable of this NetCDF...

              ```
        """
        if x is not None and x not in self.variable_names:
            raise ValueError(
                f"x={x!r} is not a variable of this NetCDF. "
                f"Available: {self.variable_names}."
            )
        if y is not None and y not in self.variable_names:
            raise ValueError(
                f"y={y!r} is not a variable of this NetCDF. "
                f"Available: {self.variable_names}."
            )

    _CURVILINEAR_NAME_PAIRS = (
        ("XLONG", "XLAT"),
        ("lon_rho", "lat_rho"),
        ("nav_lon", "nav_lat"),
    )

    def _resolve_curvilinear_coords(
        self,
        *,
        x: str | None,
        y: str | None,
        coords: tuple | list | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve curvilinear `(x, y)` coords for the rendered slice.

        Detection priority (first match wins):

        1. Explicit user `x=` / `y=` from the PR-2 signature. Both must
           be given together (one alone is rejected as ambiguous).
        2. Explicit user `coords=` (PR-3). Accepts a length-2 sequence of
           variable-name strings *or* numpy arrays.
        3. The variable's CF `coordinates` attribute, which lists the
           auxiliary coord variables for the data variable.
        4. Well-known curvilinear naming conventions for files that
           omit the CF attribute: WRF (`XLAT`/`XLONG`), ROMS
           (`lat_rho`/`lon_rho`), NEMO (`nav_lat`/`nav_lon`).

        For each candidate pair the helper reads the named variables
        via the parent container's :meth:`_read_variable` (or uses the
        caller-supplied arrays directly), then validates the shapes
        against the rendered slice. Shapes that do not match silently
        skip — the next candidate gets a chance. When nothing resolves
        to a valid pair the helper returns ``None`` so the caller falls
        back to the geotransform-derived ``extent``.

        Args:
            x: Name of the x coord variable. When set together with
                ``y`` it overrides auto-detection.
            y: Name of the y coord variable. Same semantics as ``x``.
            coords: Explicit `(x, y)` spec — either two strings (looked
                up via :meth:`_read_variable`) or two numpy arrays
                (passed straight through after shape validation).

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated
                ``(x_arr, y_arr)`` pair, or ``None`` when no
                curvilinear coords could be resolved.

        Raises:
            ValueError: If ``coords`` is not a length-2 sequence, if
                only one of ``x`` / ``y`` is given, or if user-supplied
                coord variable names do not exist on the parent
                container.
        """
        result: tuple[np.ndarray, np.ndarray] | None = None

        parent = self._parent_nc if self._parent_nc is not None else self
        data_shape = self.shape[-2:] if self.shape else None

        # Only treat `x=`/`y=` as a curvilinear override when BOTH are
        # supplied. A single `x=` (without `y=`) is a backward-compatible
        # tag for downstream code that doesn't intersect with the curvilinear
        # path — leave it alone and fall through to CF / convention
        # auto-detection.
        explicit_xy = x is not None and y is not None
        if explicit_xy:
            user_coords: tuple | None = (x, y)
        elif coords is not None:
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                raise ValueError(
                    "`coords=` must be a length-2 sequence (x, y). Got "
                    f"{type(coords).__name__} of length "
                    f"{len(coords) if hasattr(coords, '__len__') else '?'}."
                )
            user_coords = tuple(coords)
        else:
            user_coords = None

        if user_coords is not None:
            x_in, y_in = user_coords
            x_arr = self._coerce_coord_spec(x_in, parent, "x")
            y_arr = self._coerce_coord_spec(y_in, parent, "y")
            if self._coord_shapes_match(x_arr, y_arr, data_shape):
                result = (x_arr, y_arr)

        if result is None and data_shape is not None:
            cf_pair = self._cf_coordinates_pair(parent)
            if cf_pair is not None:
                x_arr, y_arr = cf_pair
                if self._coord_shapes_match(x_arr, y_arr, data_shape):
                    result = (x_arr, y_arr)

        if result is None and data_shape is not None:
            for x_name, y_name in self._CURVILINEAR_NAME_PAIRS:
                if (
                    x_name in parent.variable_names
                    and y_name in parent.variable_names
                ):
                    x_arr = parent._read_variable(x_name)
                    y_arr = parent._read_variable(y_name)
                    if x_arr is None or y_arr is None:
                        continue
                    x_arr = self._squeeze_leading_axes(x_arr, data_shape)
                    y_arr = self._squeeze_leading_axes(y_arr, data_shape)
                    if self._coord_shapes_match(x_arr, y_arr, data_shape):
                        result = (x_arr, y_arr)
                        break

        return result

    @staticmethod
    def _coerce_coord_spec(
        spec: Any, parent: NetCDF, axis_label: str,
    ) -> np.ndarray:
        """Convert a single coord spec (str or array) to a numpy array.

        Args:
            spec: Either a variable name (str) to look up on the parent
                container, or an array-like that is converted via
                :func:`numpy.asarray`.
            parent: NetCDF container used to resolve string names via
                :meth:`_read_variable`.
            axis_label: ``"x"`` or ``"y"``; used in error messages so the
                caller can spot which axis failed.

        Returns:
            np.ndarray: The resolved coordinate array.

        Raises:
            ValueError: If a string name is not in the parent's
                ``variable_names`` or :meth:`_read_variable` returns
                ``None``.
        """
        if isinstance(spec, str):
            if spec not in parent.variable_names:
                raise ValueError(
                    f"coords {axis_label}={spec!r} is not a variable of "
                    f"the parent NetCDF. Available: {parent.variable_names}."
                )
            arr = parent._read_variable(spec)
            if arr is None:
                raise ValueError(
                    f"coords {axis_label}={spec!r} could not be read via "
                    "`_read_variable`."
                )
            result = arr
        else:
            result = np.asarray(spec)
        return result

    @staticmethod
    def _squeeze_leading_axes(
        arr: np.ndarray, data_shape: tuple[int, int],
    ) -> np.ndarray:
        """Drop leading singleton/time axes so a coord matches the slice shape.

        WRF stores `XLAT` / `XLONG` as `(time, lat, lon)` even though the
        same grid is shared across time — taking time-step 0 gives a 2-D
        view that lines up with the data slice.

        Args:
            arr: Coord array, typically 2-D or 3-D ``(extra, rows, cols)``.
            data_shape: Target shape ``(rows, cols)`` of the data slice.

        Returns:
            np.ndarray: Either ``arr`` unchanged (already 1-D / 2-D
                matching) or the time-step-0 slice of a 3-D array.
        """
        rows, cols = data_shape
        if arr.ndim == 3 and arr.shape[-2:] == (rows, cols):
            result = arr[0]
        else:
            result = arr
        return result

    @staticmethod
    def _coord_shapes_match(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        data_shape: tuple[int, int] | None,
    ) -> bool:
        """Return True when ``(x_arr, y_arr)`` line up with ``data_shape``.

        Accepts the same shape rules as cleopatra's `ArrayGlyph(coords=)`:

        * ``x_arr`` is 1-D matching ``cols`` or 2-D matching the slice.
        * ``y_arr`` is 1-D matching ``rows`` or 2-D matching the slice.

        Args:
            x_arr: Candidate x coordinate array.
            y_arr: Candidate y coordinate array.
            data_shape: ``(rows, cols)`` of the data slice. ``None`` →
                cannot validate, returns ``False``.

        Returns:
            bool: ``True`` when both arrays line up with ``data_shape``.
        """
        result = False
        if data_shape is not None:
            rows, cols = data_shape
            x_ok = (x_arr.ndim == 1 and x_arr.shape[0] == cols) or (
                x_arr.ndim == 2 and x_arr.shape == data_shape
            )
            y_ok = (y_arr.ndim == 1 and y_arr.shape[0] == rows) or (
                y_arr.ndim == 2 and y_arr.shape == data_shape
            )
            result = x_ok and y_ok
        return result

    def _cf_coordinates_pair(
        self, parent: NetCDF,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Parse the CF `coordinates` attribute into an `(x, y)` array pair.

        The CF Conventions allow a data variable to declare auxiliary
        coordinate variables via its ``coordinates`` attribute (a
        space-separated string of variable names). Pyramids reads the
        attribute off ``self._variable_attrs`` (populated by
        :meth:`get_variable`), then resolves each name to an array.

        For each pair (n choose 2 from the listed coord vars) the helper
        picks the first one where one name reads as the x axis (1-D
        ``cols`` or 2-D matching) and the other as the y axis (1-D
        ``rows`` or 2-D matching). When the attribute is missing or no
        valid pair is found returns ``None`` so the caller can fall
        back to the well-known-naming pass.

        Args:
            parent: NetCDF container — coord variables are read off the
                parent (not the subset) via :meth:`_read_variable`.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated x/y
                pair, or ``None`` when nothing matched.
        """
        result = None
        attrs = getattr(self, "_variable_attrs", None) or {}
        coord_attr = attrs.get("coordinates")
        data_shape = self.shape[-2:] if self.shape else None
        if isinstance(coord_attr, str) and data_shape is not None:
            names = [n for n in coord_attr.split() if n]
            candidate_arrays: dict[str, np.ndarray] = {}
            for name in names:
                if name in parent.variable_names:
                    arr = parent._read_variable(name)
                    if arr is not None:
                        candidate_arrays[name] = self._squeeze_leading_axes(
                            arr, data_shape,
                        )
            rows, cols = data_shape
            x_candidates: list[tuple[str, np.ndarray]] = []
            y_candidates: list[tuple[str, np.ndarray]] = []
            for name, arr in candidate_arrays.items():
                if (arr.ndim == 1 and arr.shape[0] == cols) or (
                    arr.ndim == 2 and arr.shape == data_shape
                ):
                    x_candidates.append((name, arr))
                if (arr.ndim == 1 and arr.shape[0] == rows) or (
                    arr.ndim == 2 and arr.shape == data_shape
                ):
                    y_candidates.append((name, arr))
            for x_name, x_arr in x_candidates:
                for y_name, y_arr in y_candidates:
                    if x_name == y_name:
                        continue
                    if self._coord_shapes_match(x_arr, y_arr, data_shape):
                        if self._looks_like_x_then_y(x_name, y_name):
                            result = (x_arr, y_arr)
                            break
                if result is not None:
                    break
            if result is None and x_candidates and y_candidates:
                # Fallback: first viable pair regardless of name heuristic.
                x_arr = x_candidates[0][1]
                y_arr = y_candidates[0][1]
                if self._coord_shapes_match(x_arr, y_arr, data_shape):
                    result = (x_arr, y_arr)
        return result

    @staticmethod
    def _looks_like_x_then_y(x_name: str, y_name: str) -> bool:
        """Heuristic name check: x looks like a longitude, y like a latitude.

        Used to disambiguate the CF `coordinates` attribute when the
        list has two viable candidates per axis. Returns ``True`` when
        ``x_name`` contains ``"lon"`` / ``"long"`` and ``y_name``
        contains ``"lat"`` (case-insensitive). Used purely as a tiebreaker;
        a failed match falls back to the first viable pair.

        Args:
            x_name: Candidate x variable name.
            y_name: Candidate y variable name.

        Returns:
            bool: ``True`` when the names follow the lon/lat convention.
        """
        xl = x_name.lower()
        yl = y_name.lower()
        x_is_lon = "lon" in xl or "long" in xl
        y_is_lat = "lat" in yl
        return x_is_lon and y_is_lat

    def _resolve_time_dim_name(self) -> str:
        """Return the band-dim name that represents the time axis.

        Scans `_band_dim_names` (case-insensitive) for one of `time`,
        `valid_time`, or `t`. When no candidate matches, falls back to
        the **primary** (first) band dim so legacy 3-D files without
        an explicit `time` dim name still work with the `time=`
        convenience selector on :meth:`plot`.

        Returns:
            str: Name of the dim to use as the `time` axis. Either a
                match from the candidate list or the first entry of
                `_band_dim_names` when no candidate is present.

        Raises:
            ValueError: If `_band_dim_names` is empty — i.e. the
                variable is purely 2-D and has no band dim to map a
                `time=` selector onto.

        Examples:
            - A variable whose first non-spatial dim is literally named
              `time` resolves to that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> var._resolve_time_dim_name()
              'time'

              ```

            - When no band dim matches any of `time` / `valid_time` /
              `t`, the helper falls back to the **primary** band dim so
              callers using `time=` on legacy 3-D files still work:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="data", extra_dim_name="depth",
              ... )
              >>> var = nc.get_variable("data")
              >>> var._resolve_time_dim_name()
              'depth'

              ```
        """
        if not self._band_dim_names:
            raise ValueError(
                "`time=` was passed but this variable has no band dimension."
            )
        candidates = ("time", "valid_time", "t")
        for name in self._band_dim_names:
            if name.lower() in candidates:
                return name
        return self._band_dim_names[0]

    def _resolve_level_dim_name(self) -> str:
        """Return the band-dim name that represents the vertical axis.

        Auto-detection scans `_band_dim_names` (case-insensitive) for
        one of `pressure_level`, `depth`, `height`, `z`, or `level`.
        Unlike :meth:`_resolve_time_dim_name` this helper does **not**
        fall back to the primary band dim — a non-time/non-member dim
        that happens to be first is unlikely to actually be a vertical
        axis, so the helper prefers an explicit failure that asks the
        caller to use `sel={dim: value}` instead.

        Returns:
            str: Name of the dim to use as the `level` axis. The first
                entry of `_band_dim_names` whose lowercased name is in
                the candidate set.

        Raises:
            ValueError: If `_band_dim_names` is empty, or if no entry
                matches the candidate vertical-dim names. The error
                message lists the actual band dims to help the caller
                pick the right `sel=` key.

        Examples:
            - A variable with a `pressure_level` band dim resolves to
              that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> arr = np.random.rand(2, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="temperature",
              ...     extra_dim_name="pressure_level",
              ...     extra_dim_values=[1000, 500],
              ... )
              >>> var = nc.get_variable("temperature")
              >>> var._resolve_level_dim_name()
              'pressure_level'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as a vertical axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> var._resolve_level_dim_name()  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `level=` could not be auto-resolved...

              ```
        """
        if not self._band_dim_names:
            raise ValueError(
                "`level=` was passed but this variable has no band dimension."
            )
        candidates = ("pressure_level", "depth", "height", "z", "level")
        for name in self._band_dim_names:
            if name.lower() in candidates:
                return name
        raise ValueError(
            "`level=` could not be auto-resolved. Use `sel={dim: value}` to "
            f"name the vertical dim explicitly. Band dims: "
            f"{list(self._band_dim_names)}."
        )

    def _resolve_member_dim_name(self) -> str:
        """Return the band-dim name that represents the ensemble axis.

        Auto-detection scans `_band_dim_names` (case-insensitive) for
        one of `member`, `realization`, `ensemble`. Like
        :meth:`_resolve_level_dim_name` this helper raises rather than
        falling back to the primary band dim, so a typo or missing
        ensemble dim surfaces as an explicit error.

        Returns:
            str: Name of the dim to use as the `member` axis. The
                first entry of `_band_dim_names` whose lowercased name
                is in the candidate set.

        Raises:
            ValueError: If `_band_dim_names` is empty, or if no entry
                matches the candidate ensemble-dim names. The error
                message lists the actual band dims to help the caller
                pick the right `sel=` key.

        Examples:
            - A variable with a `realization` band dim resolves to
              that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> arr = np.random.rand(5, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m",
              ...     extra_dim_name="realization",
              ...     extra_dim_values=[0, 1, 2, 3, 4],
              ... )
              >>> var = nc.get_variable("t2m")
              >>> var._resolve_member_dim_name()
              'realization'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as an ensemble axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> var._resolve_member_dim_name()  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `member=` could not be auto-resolved...

              ```
        """
        if not self._band_dim_names:
            raise ValueError(
                "`member=` was passed but this variable has no band dimension."
            )
        candidates = ("member", "realization", "ensemble")
        for name in self._band_dim_names:
            if name.lower() in candidates:
                return name
        raise ValueError(
            "`member=` could not be auto-resolved. Use `sel={dim: value}` to "
            f"name the ensemble dim explicitly. Band dims: "
            f"{list(self._band_dim_names)}."
        )

    def read_array(
        self,
        variable: str | None = None,
        band: int | None = None,
        window: list[int] | None = None,
        unpack: bool = False,
        *,
        chunks: Any = None,
        lock: Any = None,
    ) -> ArrayLike:
        """Read array from the dataset (eager by default, lazy with `chunks`).

        Args:
            variable: When this instance is a root MDIM container,
                the variable name to read. When the instance is
                already a variable subset (`nc.get_variable("x")`)
                this argument must be `None` — the variable is
                already pinned.
            band: Band index to read, or None for all bands. Only
                honored on the eager path (`chunks=None`).
            window: Spatial window to read. Only honored on the
                eager path.
            unpack: If True and the variable has CF `scale_factor`
                and/or `add_offset`, apply the transformation
                `real = raw * scale + offset`. Defaults to False.
                Applied lazily via :mod:`dask.array` arithmetic when
                `chunks` is given — the compute graph stays lazy
                until the caller materializes it.
            chunks: Chunking spec for a lazy return. `None` (the
                default) returns an eager :class:`numpy.ndarray` and
                preserves the legacy behavior. Any of `int`,
                `tuple`, `dict`, or the string `"auto"` switches
                to a :class:`dask.array.Array` backed by MDArray
                chunk reads. Defaults chunked at the variable's
                native `GetBlockSize` (see
                :attr:`pyramids.netcdf.models.VariableInfo.block_size`);
                a conservative `(1,..., rows, cols)` fallback is
                used when the driver doesn't advertise one.
            lock: Lock passed to the underlying
                :class:`pyramids.base._file_manager.CachingFileManager`.
                `None` → :func:`pyramids.base._locks.default_lock`
                (a :class:`SerializableLock`, or a
                `dask.distributed.Lock` when a client is active).
                `False` → :class:`pyramids.base._locks.DummyLock`.
                Only meaningful when `chunks` is not `None`.

        Returns:
            np.ndarray or dask.array.Array: The array data, eager
            (numpy) by default or lazy (dask) when `chunks` is
            supplied. The lazy array computes chunk-by-chunk through
            `md_arr.ReadAsArray(array_start_idx=starts, count=counts)`.

        Raises:
            ValueError: If called on a root MDIM container without a
                `variable` argument, or when a subset is called
                with a conflicting `variable` name.
            ImportError: If `chunks` is given but `dask` is not
                installed. Install the `[lazy]` extra.
        """
        is_container = (
            self._is_md_array and not self._is_subset and self.band_count == 0
        )
        if is_container:
            if variable is None:
                self._check_not_container("read_array")
            subset = self.get_variable(variable)
            return subset.read_array(
                band=band,
                window=window,
                unpack=unpack,
                chunks=chunks,
                lock=lock,
            )
        if variable is not None and variable != self._source_var_name:
            raise ValueError(
                f"This NetCDF instance is already pinned to variable "
                f"{self._source_var_name!r}; cannot re-read as "
                f"{variable!r}. Call read_array on the parent container "
                "instead."
            )
        if chunks is None:
            result = super().read_array(band=band, window=window)
            if unpack:
                result = _apply_unpack(
                    result,
                    getattr(self, "_scale", None),
                    getattr(self, "_offset", None),
                )
        else:
            parent = self._parent_nc if self._parent_nc is not None else self
            path = parent._file_name
            if path.startswith("NETCDF"):
                path = path.split(":")[1][1:-1]
            var_name = self._source_var_name
            if var_name is None:
                raise ValueError(
                    "Lazy read requires a variable name; pass "
                    "`variable=` on the container or call read_array "
                    "on a subset from `get_variable()`."
                )
            result = build_lazy_array(
                path=path,
                variable_name=var_name,
                chunks=chunks,
                lock=lock,
            )
            if unpack:
                result = _apply_unpack(
                    result,
                    getattr(self, "_scale", None),
                    getattr(self, "_offset", None),
                )
        return result

    def _preserve_netcdf_metadata(self, result: Dataset) -> NetCDF:
        """Wrap a Dataset result as a NetCDF, preserving variable-subset metadata.

        When spatial operations (crop, to_crs, resample) are called on a
        NetCDF variable subset, the parent `Dataset` mixin returns a
        plain `Dataset`. This helper re-wraps the result as a `NetCDF`
        and copies over the variable-specific attributes so that methods
        like `sel()`, `read_array(unpack=True)`, and further spatial
        operations continue to work with consistent return types.

        Both the legacy single-band-dim fields (`_band_dim_name`,
        `_band_dim_values`) and the multi-band-dim fields
        (`_band_dim_names`, `_band_dim_values_map`, `_band_dim_sizes`)
        are propagated. The legacy length-guard nullifies
        `_band_dim_values` only when its primary-dim view is provably
        stale: for single-band-dim variables it compares
        `len(values) != _band_count`; for multi-band-dim variables it
        compares `prod(_band_dim_sizes) != _band_count` instead, so a
        4-D variable whose total band count diverged from the cached
        sizes (e.g. after a band-shrinking operation outside `sel()`)
        drops the now-stale primary view.

        Args:
            result: The `Dataset` (or `NetCDF`) returned by a parent
                spatial operation.

        Returns:
            NetCDF: The same data wrapped as a `NetCDF` with all
                variable-subset metadata preserved.

        See Also:
            `sel`: produces results that flow through this helper to
                keep the multi-band-dim metadata consistent across
                spatial ops.
        """
        if isinstance(result, NetCDF):
            wrapped = result
        else:
            wrapped = NetCDF(
                result._raster,
                access=result._access,
                open_as_multi_dimensional=False,
            )
        wrapped._is_md_array = self._is_md_array
        wrapped._is_subset = self._is_subset
        wrapped._band_dim_name = self._band_dim_name
        wrapped._band_dim_names = self._band_dim_names
        wrapped._band_dim_sizes = self._band_dim_sizes
        wrapped._band_dim_values_map = dict(self._band_dim_values_map)
        # Length-guard: nullify legacy values only when the primary-dim
        # view is provably stale. For multi-band-dim variables the
        # `_band_count` is the product of every band-dim size, so compare
        # against `prod(_band_dim_sizes)` rather than `len(values)`.
        expected_count = math.prod(self._band_dim_sizes)
        if (
            self._band_dim_values is not None
            and wrapped._band_count > 0
            and len(self._band_dim_names) <= 1
            and len(self._band_dim_values) != wrapped._band_count
        ):
            wrapped._band_dim_values = None
        elif (
            self._band_dim_values is not None
            and wrapped._band_count > 0
            and len(self._band_dim_names) > 1
            and expected_count != wrapped._band_count
        ):
            # Multi-band-dim variable whose total band count diverged from
            # the cached sizes (e.g. after a band-shrinking operation
            # outside sel()). Drop the now-stale primary view.
            wrapped._band_dim_values = None
        else:
            wrapped._band_dim_values = self._band_dim_values
        # Self-heal: if the guard above nulled the legacy values but
        # the per-dim map still carries an entry of the right length
        # for the new band count, refill from there. Makes the helper
        # idempotent under repeat calls and removes the post-call
        # refill requirement on callers like `sel()` for the
        # pin-secondary-dim case (where the primary-dim entry in the
        # map is still valid).
        if (
            wrapped._band_dim_values is None
            and wrapped._band_dim_name is not None
            and wrapped._band_count > 0
        ):
            candidate = wrapped._band_dim_values_map.get(wrapped._band_dim_name)
            if candidate is not None and len(candidate) == wrapped._band_count:
                wrapped._band_dim_values = list(candidate)
        wrapped._variable_attrs = self._variable_attrs
        wrapped._scale = self._scale
        wrapped._offset = self._offset
        wrapped._parent_nc = self._parent_nc
        wrapped._source_var_name = self._source_var_name
        wrapped._gdal_md_arr_ref = None
        wrapped._gdal_rg_ref = None
        return wrapped

    def crop(self, mask: Any, touch: bool = True) -> NetCDF:
        """Crop the dataset using a polygon or raster mask.

        On a **root MDIM container** this crops every variable and
        returns a new in-memory NetCDF container with the cropped
        results. On a **variable subset** it delegates to the
        parent `Dataset.crop()` and wraps the result as `NetCDF`
        to preserve variable metadata (`_band_dim_name`,
        `_band_dim_values`, `sel()`, etc.).

        Args:
            mask: GeoDataFrame with polygon geometry, or a Dataset
                to use as a spatial mask.
            touch: If True, include cells that touch the mask
                boundary. Defaults to True.

        Returns:
            NetCDF: Cropped container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "crop",
                {"mask": mask, "touch": touch},
            )
        else:
            result = super().crop(mask=mask, touch=touch)
            result = self._preserve_netcdf_metadata(result)
        return result

    def _apply_to_all_variables(self, operation, op_kwargs):
        """Apply an operation to every variable in the container.

        Args:
            operation: Name of the Dataset method to call (e.g. "crop").
            op_kwargs: Keyword arguments to pass to the method.

        Returns:
            NetCDF: New container with the operation applied to all variables.

        Raises:
            ValueError: If the container has no data variables.
        """
        if not self.variable_names:
            raise ValueError(
                "Cannot apply operation to an empty container (no data variables)."
            )

        result = None
        for var_name in self.variable_names:
            var = self.get_variable(var_name)
            var_result = getattr(var, operation)(**op_kwargs)
            # to_crs returns a VRT — materialize before the source goes
            # out of scope. read_array also squeezes singleton-band 3-D
            # variables to 2-D, so re-expand when the variable carried a
            # band/time/level dim originally.
            var_arr = var_result.read_array()
            if var_arr.ndim == 2 and var._band_dim_name is not None:
                var_arr = np.expand_dims(var_arr, axis=0)
            # For 4-D+ variables, GDAL classic raster flattened the
            # non-spatial axes into a single bands axis on read — undo
            # that so the rebuild can materialise every band-dim. The
            # cached `_band_dim_sizes` describes the storage order
            # (last non-spatial dim varies fastest, matching GDAL's
            # row-major flatten), so the reshape is the literal
            # inverse of that flatten.
            if (
                len(var._band_dim_names) > 1
                and var_arr.ndim == 3
                and var._band_dim_sizes
            ):
                var_arr = var_arr.reshape(
                    *var._band_dim_sizes, var_arr.shape[-2], var_arr.shape[-1]
                )
            var_ndv = var_result.no_data_value
            var_ndv_scalar = (
                var_ndv[0] if isinstance(var_ndv, list) and var_ndv else var_ndv
            )
            extra_dims = (
                [
                    (name, var._band_dim_values_map.get(name))
                    for name in var._band_dim_names
                ]
                if var._band_dim_names
                else None
            )

            if result is None:
                # First variable: build the container.
                if extra_dims is not None:
                    result = NetCDF.create_from_array(
                        arr=var_arr,
                        geo=var_result.geotransform,
                        epsg=var_result.epsg,
                        no_data_value=var_ndv_scalar,
                        variable_name=var_name,
                        extra_dims=extra_dims,
                    )
                else:
                    result = NetCDF.create_from_array(
                        arr=var_arr,
                        geo=var_result.geotransform,
                        epsg=var_result.epsg,
                        no_data_value=var_ndv_scalar,
                        variable_name=var_name,
                    )
            else:
                # Subsequent variables: drop into the existing container.
                ds = Dataset.create_from_array(
                    var_arr,
                    geo=var_result.geotransform,
                    epsg=var_result.epsg,
                    no_data_value=var_ndv_scalar,
                )
                ds._band_dim_name = var._band_dim_name
                ds._band_dim_values = var._band_dim_values
                ds._band_dim_names = var._band_dim_names
                ds._band_dim_values_map = dict(var._band_dim_values_map)
                ds._band_dim_sizes = var._band_dim_sizes
                result.set_variable(var_name, ds)

        return result

    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> NetCDF:
        """Reproject the dataset to a different CRS.

        On a **root MDIM container** this reprojects every variable
        and returns a new container. On a **variable subset** it
        delegates to `Dataset.to_crs()` and wraps the result as
        `NetCDF` to preserve variable metadata.

        Args:
            to_epsg: Target EPSG code (e.g., 4326, 32637).
            method: Resampling method. Defaults to `"nearest neighbor"`.
            maintain_alignment: If True, keep the same number of rows
                and columns. Defaults to False.

        Returns:
            NetCDF: Reprojected container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "to_crs",
                {
                    "to_epsg": to_epsg,
                    "method": method,
                    "maintain_alignment": maintain_alignment,
                },
            )
        else:
            result = super().to_crs(
                to_epsg=to_epsg,
                method=method,
                maintain_alignment=maintain_alignment,
            )
            result = self._preserve_netcdf_metadata(result)
        return result

    def resample(
        self,
        cell_size: float,
        method: str = "nearest neighbor",
    ) -> NetCDF:
        """Resample the dataset to a different cell size.

        On a **root MDIM container** this resamples every variable
        and returns a new container. On a **variable subset** it
        delegates to `Dataset.resample()` and wraps the result as
        `NetCDF` to preserve variable metadata.

        Args:
            cell_size: New cell size.
            method: Resampling method. Defaults to `"nearest neighbor"`.

        Returns:
            NetCDF: Resampled container or variable subset.
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            result = self._apply_to_all_variables(
                "resample",
                {"cell_size": cell_size, "method": method},
            )
        else:
            result = super().resample(
                cell_size=cell_size,
                method=method,
            )
            result = self._preserve_netcdf_metadata(result)
        return result

    def sel(self, **kwargs: Any) -> NetCDF:
        """Select a subset of bands by coordinate values along a band dim.

        Extracts bands whose coordinate values match the given criteria.
        Works on any variable subset that has at least one non-spatial
        dimension tracked in `_band_dim_names` (set by
        `get_variable()`). For 4-D+ files with multiple non-spatial
        dims (e.g. `(valid_time, pressure_level, lat, lon)` from CDS-Beta
        ERA5), `sel()` may name any of those dims; chaining `sel()`
        pins multiple band dims one at a time.

        The result is always a `NetCDF` instance with the same variable
        metadata preserved, so `sel()` can be chained and NetCDF-only
        methods like `read_array(unpack=True)` remain available.

        Internals: GDAL flattens an MDIM array `(d_0, ..., d_{n-1},
        lat, lon)` row-major over the non-spatial dims, with the last
        non-spatial dim varying fastest. For a band dim at axis `k`
        with sizes `S`, the implementation uses
        `stride = prod(S[k+1:])`, `block = stride * S[k]`, and
        `total = prod(S)` to map each pinned index `p` to the band
        ranges `[outer + p*stride .. outer + (p+1)*stride)` for every
        `outer in range(0, total, block)`. For a single-band-dim
        variable this reduces to the identity
        `band_indices == dim_indices`.

        Args:
            **kwargs: Exactly one keyword argument. The key must name a
                tracked band dim (one of `self._band_dim_names`); the
                value is one of:

                - A single number: select one band by exact value.
                - A list of numbers: select multiple bands.
                - A `slice(start, stop)`: select bands whose coord
                  falls between `start` and `stop` inclusive. Bounds
                  are normalised before matching, so the slice is
                  direction-agnostic — works on both ascending and
                  descending coord axes (e.g. `latitude` stored
                  north-to-south).

        Returns:
            NetCDF: A new variable subset with only the selected bands
                and full metadata preserved. `_band_dim_sizes` reflects
                the pinned axis (e.g. `(4, 1)` after pinning a level on
                a `(4, 3)` cube), and `_band_dim_values_map[dim_name]`
                shrinks to the chosen values. Legacy `_band_dim_values`
                is refreshed from the (possibly updated) primary entry
                in the map.

        Raises:
            ValueError: If exactly one kwarg isn't passed, the variable
                has no tracked band dims, the named dim isn't one of
                `_band_dim_names`, the dim has no coord values
                (`_band_dim_values_map[dim] is None`), or no bands match
                the selector.

        Examples:
            - Pin a pressure level on a 4-D file:
                ```python
                >>> nc = NetCDF.read_file(  # doctest: +SKIP
                ...     "tests/data/netcdf/pyramids-netcdf-4d.nc"
                ... )
                >>> var = nc.get_variable("temperature")  # doctest: +SKIP
                >>> sub = var.sel(pressure_level=500)  # doctest: +SKIP
                >>> sub._band_dim_sizes  # doctest: +SKIP
                (4, 1)

                ```
            - Chain `sel()` to pin both time and level (collapses to 2-D):
                ```python
                >>> sub = var.sel(time=12).sel(pressure_level=500)  # doctest: +SKIP
                >>> sub.read_array().shape  # doctest: +SKIP
                (5, 6)

                ```
            - Use a list selector to keep only two of the levels:
                ```python
                >>> sub = var.sel(pressure_level=[1000, 500])  # doctest: +SKIP
                >>> sub._band_dim_values_map["pressure_level"]  # doctest: +SKIP
                [1000.0, 500.0]

                ```
            - Use a slice selector — direction-agnostic, so the same
              call works on ascending coords (e.g. `[500, 850, 1000]`)
              and on descending coords (e.g. `[1000, 850, 500]`):
                ```python
                >>> sub = var.sel(pressure_level=slice(500, 1000))  # doctest: +SKIP
                >>> sub._band_dim_values_map["pressure_level"]  # doctest: +SKIP
                [1000.0, 850.0, 500.0]

                ```

        Notes:
            All four examples above are tagged `# doctest: +SKIP`
            because they need a real on-disk NetCDF fixture. The
            runnable equivalents live in:

            - `tests/netcdf/test_sel.py::TestSelSingleValue` /
              `TestSelList` / `TestSelSlice` (3-D scenarios — single
              value, list selector, slice selector including the
              direction-agnostic path).
            - `tests/netcdf/test_sel_4d.py::TestSelByPressureLevel` /
              `TestSelByTime` / `TestSelChained` (4-D scenarios —
              pin secondary / primary dim, chained `sel().sel()`).
            - `tests/netcdf/test_sel_4d.py::TestSelErrorMessages` (the
              error contract).

        See Also:
            `get_variable`: builds a variable subset and populates the
                band-dim metadata that `sel()` consumes.
        """
        if len(kwargs) != 1:
            raise ValueError("sel() requires exactly one keyword argument.")

        dim_name, selector = next(iter(kwargs.items()))

        if not self._band_dim_names:
            raise ValueError(
                "sel() requires a variable with at least one non-spatial "
                "dimension. This variable has no band dimensions tracked."
            )
        if dim_name not in self._band_dim_names:
            raise ValueError(
                f"Dimension {dim_name!r} does not match any band dimension "
                f"of this variable {list(self._band_dim_names)!r}."
            )

        coords = self._band_dim_values_map.get(dim_name)
        if coords is None:
            raise ValueError(
                f"No coordinate values available for dimension {dim_name!r}."
            )

        if isinstance(selector, slice):
            start = selector.start if selector.start is not None else coords[0]
            stop = selector.stop if selector.stop is not None else coords[-1]
            # Normalise bounds so the match works on both ascending and
            # descending coord axes (e.g. `latitude = [44, 43, 42, 41,
            # 40]` from CDS-Beta retrievals). Without this, a
            # `slice(None, None)` on a descending axis defaults to
            # `start=44, stop=40`, the test `44 <= v <= 40` matches
            # nothing, and the user gets a confusing "no bands match"
            # error instead of "select everything".
            lo, hi = (start, stop) if start <= stop else (stop, start)
            dim_indices = [i for i, v in enumerate(coords) if lo <= v <= hi]
        elif isinstance(selector, list):
            coord_set = set(selector)
            dim_indices = [i for i, v in enumerate(coords) if v in coord_set]
        else:
            dim_indices = [i for i, v in enumerate(coords) if v == selector]

        if not dim_indices:
            raise ValueError(
                f"No bands match {dim_name}={selector}. "
                f"Available values: {coords}"
            )

        # Map (pinned dim index along dim_name) -> classic-band indices.
        # GDAL flattens (d_0, ..., d_{n-1}, lat, lon) row-major over the
        # non-spatial dims: the last non-spatial dim varies fastest. For
        # a band-dim at axis `k` with sizes S, stride = prod(S[k+1:]) and
        # block = stride * S[k]. For each pinned index p we emit
        # `[outer + p*stride .. outer + (p+1)*stride)` for every outer in
        # range(0, total, block). Reduces to identity (band_indices ==
        # dim_indices) when there is exactly one band dim.
        dim_axis = self._band_dim_names.index(dim_name)
        sizes = self._band_dim_sizes
        stride = math.prod(sizes[dim_axis + 1:])
        block = stride * sizes[dim_axis]
        total = math.prod(sizes)

        band_indices: list[int] = []
        for pinned in dim_indices:
            for outer_start in range(0, total, block):
                base = outer_start + pinned * stride
                band_indices.extend(range(base, base + stride))

        selected_coords = [coords[i] for i in dim_indices]

        # Read only the selected bands instead of loading the full array.
        # Each band index maps to a 1-based GDAL band in the classic
        # dataset view created by get_variable(). Band-by-band reads
        # avoid loading the entire variable into memory.
        band_arrays = [self.read_array(band=i) for i in band_indices]
        if len(band_arrays) == 1:
            selected = band_arrays[0]
        else:
            selected = np.stack(band_arrays, axis=0)

        ndv = self.no_data_value
        ndv_scalar = ndv[0] if isinstance(ndv, list) and ndv else ndv
        ds_result = Dataset.create_from_array(
            selected,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=ndv_scalar,
        )
        result = self._preserve_netcdf_metadata(ds_result)
        new_sizes = tuple(
            len(dim_indices) if i == dim_axis else s
            for i, s in enumerate(sizes)
        )
        result._band_dim_sizes = new_sizes
        result._band_dim_values_map = dict(self._band_dim_values_map)
        result._band_dim_values_map[dim_name] = selected_coords
        # Refresh legacy primary-dim values to match the (possibly
        # updated) primary entry in the map. `_band_dim_name` is
        # guaranteed non-None here: entry to `sel()` requires
        # `_band_dim_names` to be non-empty, and the build path always
        # sets `_band_dim_name = _band_dim_names[0]`.
        result._band_dim_values = result._band_dim_values_map.get(
            result._band_dim_name
        )

        return result

    @classmethod
    def read_file(  # type: ignore[override]
        cls,
        path: str | Path,
        read_only: bool = True,
        open_as_multi_dimensional: bool = True,
    ) -> NetCDF:
        """Open a NetCDF file from disk.

        Args:
            path: Path to the `.nc` file.
            read_only: If True, open in read-only mode. Set to False for
                write access. Defaults to True.
            open_as_multi_dimensional: If True, open with
                `gdal.OF_MULTIDIM_RASTER` to access the full group /
                dimension / variable hierarchy. If False, open in classic
                raster mode where each variable is a subdataset.
                Defaults to True.

        Returns:
            NetCDF: The opened dataset.
        """
        src = _io.read_file(path, read_only, open_as_multi_dimensional)
        if read_only:
            read_only = "read_only"
        else:
            read_only = "write"
        return cls(
            src, access=read_only, open_as_multi_dimensional=open_as_multi_dimensional
        )

    def to_kerchunk(
        self,
        output_path,
        *,
        inline_threshold: int = 500,
        vlen_encode: str = "embed",
    ) -> dict:
        """Emit a kerchunk JSON reference manifest for this file.

        Thin forwarder to :func:`pyramids.netcdf._kerchunk.to_kerchunk`
        using `self._file_name` as the source path. Requires the
        `[netcdf-lazy]` optional extra.

        Args:
            output_path: Path where the manifest JSON is written.
            inline_threshold: Chunks smaller than this many bytes are
                embedded directly. Default 500.
            vlen_encode: VLEN string handling mode. Default `"embed"`.

        Returns:
            dict: The manifest dict that was written.
        """
        return to_kerchunk(
            self._file_name,
            output_path,
            inline_threshold=inline_threshold,
            vlen_encode=vlen_encode,
        )

    @classmethod
    def combine_kerchunk(
        cls,
        paths,
        output_path,
        *,
        concat_dims=("time",),
        identical_dims=("lat", "lon"),
        inline_threshold: int = 500,
    ) -> dict:
        """Emit a combined kerchunk manifest spanning many NetCDFs.

        Thin forwarder to
        :func:`pyramids.netcdf._kerchunk.combine_kerchunk`. Requires
        the `[netcdf-lazy]` optional extra.

        Args:
            paths: Sequence of NetCDF paths to combine.
            output_path: Path where the combined manifest is written.
            concat_dims: Dimension name(s) along which to concatenate.
                Default `("time",)`.
            identical_dims: Dimensions expected to match across all
                files. Default `("lat", "lon")`.
            inline_threshold: Chunks smaller than this inline bytes are
                embedded. Default 500.

        Returns:
            dict: The combined manifest.
        """
        return combine_kerchunk(
            paths,
            output_path,
            concat_dims=concat_dims,
            identical_dims=identical_dims,
            inline_threshold=inline_threshold,
        )

    @classmethod
    def open_mfdataset(
        cls,
        paths,
        variable: str,
        *,
        chunks=None,
        parallel: bool = False,
        preprocess=None,
    ):
        """Open many NetCDFs and stack `variable` into one lazy dask array.

        Thin forwarder to
        :func:`pyramids.netcdf._mfdataset.open_mfdataset`; see that
        function for the full argument contract. Requires the
        `[lazy]` optional extra.

        Args:
            paths: Glob string, explicit path, or sequence of paths.
            variable: Name of the variable to extract from each file.
            chunks: Chunk spec forwarded to
                :meth:`NetCDF.read_array`.
            parallel: Fan out per-file opens through `dask.delayed`.
            preprocess: Optional callable applied to each
                :class:`NetCDF` before extraction.

        Returns:
            dask.array.Array: Stack of shape `(n_files, *var_shape)`.
        """
        return open_mfdataset(
            paths,
            variable,
            chunks=chunks,
            parallel=parallel,
            preprocess=preprocess,
        )

    @property
    def meta_data(self) -> NetCDFMetadata:
        """Structured metadata for this NetCDF.

        Uses the GDAL Multidimensional API (groups, arrays, dimensions) when
        the file was opened with `open_as_multi_dimensional=True`. Falls
        back to the classic `NETCDF_DIM_*` parser (`dimensions.py`) when
        opened in classic mode (no root group available).

        Cached on first access. Invalidated by add_variable/remove_variable.

        Returns:
            NetCDFMetadata
        """
        if self._cached_meta_data is None:
            open_options = {
                "Open Mode": "SHARED" if self.is_subset else "MULTIDIM_RASTER"
            }
            self._cached_meta_data = get_metadata(self._raster, open_options)
        return self._cached_meta_data

    @meta_data.setter
    def meta_data(self, value: dict[str, str] | NetCDFMetadata) -> None:
        """Set metadata on this NetCDF dataset."""
        if isinstance(value, dict):
            for key, val in value.items():
                self._raster.SetMetadataItem(key, val)
        else:
            self._cached_meta_data = value

    def get_all_metadata(self, open_options: dict | None = None) -> NetCDFMetadata:
        """Get full MDIM metadata (uncached).

        Unlike `meta_data` (which is cached), this always re-traverses
        the GDAL multidimensional structure.

        Args:
            open_options: Driver-specific open options forwarded to
                `get_metadata()`. Defaults to None.

        Returns:
            NetCDFMetadata
        """
        result = get_metadata(self._raster, open_options)
        return result

    def get_time_variable(
        self, var_name: str = "time", time_format: str = "%Y-%m-%d"
    ) -> list[str] | None:
        """Parse the time coordinate variable into formatted date strings.

        Reads the `units` attribute (e.g., `"days since 1979-01-01"`)
        from the dimension metadata and converts raw numeric values to
        human-readable date strings.

        Args:
            var_name: Name of the time dimension / variable.
                Defaults to `"time"`.
            time_format: strftime format for the output strings.
                Defaults to `"%Y-%m-%d"`.

        Returns:
            list[str] or None: Formatted time strings, or None if the
            time dimension is not found or lacks a `units` attribute.
        """
        time_stamp = None
        time_dim = self.meta_data.get_dimension(var_name)
        if time_dim is not None:
            units = time_dim.attrs.get("units")
            if units is not None:
                calendar = time_dim.attrs.get("calendar", "standard")
                time_vals = self._read_variable(var_name)
                if time_vals is not None:
                    func = create_time_conversion_func(
                        units, time_format, calendar=calendar
                    )
                    time_stamp = list(map(func, time_vals.reshape(-1)))
        return time_stamp

    def _get_dimension_names(self) -> list[str] | None:
        """Return all dimension names, in storage order.

        On the root MDIM container, this reads from `GetRootGroup()`.
        On a variable subset (returned by `get_variable()`), the
        underlying raster is a classic-mode in-memory `Dataset` whose
        `GetRootGroup()` is `None`, but the source MDArray's dim names
        were captured into `_md_array_dims` at subset-build time. Fall
        through to that field so cube callers see the same public
        surface as container callers.

        Returns:
            list[str] or None: Dim names. `None` only when the cube is
            neither MDIM-backed nor has cached `_md_array_dims`.
        """
        rg = self._raster.GetRootGroup()
        if rg is not None:
            dims = rg.GetDimensions()
            return [dim.GetName() for dim in dims]
        cached = getattr(self, "_md_array_dims", None)
        if cached:
            return list(cached)
        return None

    @property
    def dimension_names(self) -> list[str] | None:
        """Names of all dimensions in storage order.

        On the root MDIM container the names come from the GDAL root
        group (e.g. `["x", "y", "time"]`). On a variable subset
        returned by `get_variable()` the names come from the cached
        `_md_array_dims` captured at subset-build time, so 4-D+ cubes
        report all dims (e.g. `["valid_time", "pressure_level",
        "latitude", "longitude"]`) without touching private state.

        Returns:
            list[str] or None: Dim names. `None` only on a cube that
            has neither a root group nor cached `_md_array_dims`.
        """
        return self._get_dimension_names()

    def _get_dimension(self, name: str) -> gdal.Dimension:
        dim_names = self.dimension_names
        if dim_names is not None and name in dim_names:
            rg = self._raster.GetRootGroup()
            dims = rg.GetDimensions()
            dim = dims[dim_names.index(name)]
        else:
            dim = None
        return dim

    def _needs_y_flip(self, rg, md_arr) -> bool:
        """Check if an MDArray's Y dimension goes south-to-north.

        Uses AsClassicDataset to check the geotransform Y pixel size.
        Returns True if the data needs flipping (positive Y pixel size).
        Returns False for 1-D arrays or when orientation is already correct.

        Args:
            rg: The root group (kept alive to prevent SWIG GC).
            md_arr: The MDArray to check.
        """
        result = False
        dims = md_arr.GetDimensions()
        if len(dims) >= 2:
            try:
                src = md_arr.AsClassicDataset(len(dims) - 1, len(dims) - 2, rg)
                result = src.GetGeoTransform()[5] > 0
            except Exception:
                pass
        return result

    def _read_variable(
        self,
        var: str,
        window: list[tuple[int, int]] | None = None,
    ) -> np.ndarray | None:
        """Read a variable's data as a numpy array, optionally windowed.

        Uses the MDIM root group when available (avoids opening a new GDAL
        handle). Falls back to the classic `NETCDF:file:var` path.

        For arrays with 2+ dimensions, the Y axis is flipped if the data
        is stored south-to-north (matching the flip in `get_variable`).

        Args:
            var: Variable name in the dataset.
            window: Per-dimension window as a list of `(start, count)`
                tuples, one per dimension of the target variable. For
                example, `[(0, 1), (100, 256), (200, 256)]` reads
                time[0:1], y[100:356], x[200:456]. When `None` the
                full variable is read. Only supported in MDIM mode;
                ignored in classic mode.

        Returns:
            np.ndarray or None: The variable data, or None if the
                variable is not found.
        """
        result = None
        rg = self._raster.GetRootGroup()
        if rg is not None:
            try:
                md_arr = rg.OpenMDArray(var)
                if md_arr is not None:
                    if window is not None:
                        starts = [w[0] for w in window]
                        counts = [w[1] for w in window]
                        result = md_arr.ReadAsArray(
                            array_start_idx=starts,
                            count=counts,
                        )
                    else:
                        result = md_arr.ReadAsArray()
                    # Flip Y axis if south-to-north (same as get_variable)
                    if result is not None and result.ndim >= 2:
                        if window is None and self._needs_y_flip(rg, md_arr):
                            y_axis = result.ndim - 2
                            result = np.flip(result, axis=y_axis)
            except Exception:
                pass  # nosec B110
            # Fall back to dimension indexing variable
            if result is None:
                dim = self._get_dimension(var)
                if dim is not None:
                    iv = dim.GetIndexingVariable()
                    if iv is not None:
                        if window is not None and len(window) == 1:
                            starts = [window[0][0]]
                            counts = [window[0][1]]
                            result = iv.ReadAsArray(
                                array_start_idx=starts,
                                count=counts,
                            )
                        else:
                            result = iv.ReadAsArray()
        else:
            # Classic mode: open via subdataset string
            try:
                ds = gdal.Open(f"NETCDF:{self.file_name}:{var}")
                if ds is not None:
                    result = ds.ReadAsArray()
                ds = None
            except (RuntimeError, AttributeError):
                pass
        return result

    @property
    def group_names(self) -> list[str]:
        """Names of sub-groups in the root group.

        Returns:
            list[str]: Sub-group names (e.g. `["forecast", "analysis"]`).
            Empty list if no sub-groups exist or the dataset is in
            classic mode.
        """
        rg = self._raster.GetRootGroup()
        result = []
        if rg is not None:
            try:
                names = rg.GetGroupNames()
                if names:
                    result = list(names)
            except Exception:
                pass
        return result

    def get_group(self, group_name: str) -> NetCDF:
        """Open a sub-group as a NetCDF container.

        The returned object wraps the sub-group's GDAL dataset and
        exposes the sub-group's variables and dimensions via the
        same API as the root container.

        Args:
            group_name: Name of the sub-group. Supports nested paths
                separated by `/` (e.g. `"forecast/surface"`).

        Returns:
            NetCDF: A container backed by the sub-group.

        Raises:
            ValueError: If the group doesn't exist or the dataset
                has no root group.
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError("get_group requires a multidimensional container.")

        # Navigate nested paths: "forecast/surface" → open each level
        group = rg
        parts = group_name.split("/")
        for part in parts:
            try:
                group = group.OpenGroup(part)
            except Exception:
                group = None
            if group is None:
                raise ValueError(
                    f"Group '{group_name}' not found. "
                    f"Available groups: {self.group_names}"
                )

        # Create a multidimensional dataset from the sub-group.
        # GDAL doesn't have a direct "group → dataset" conversion,
        # so we build a MEM MDIM dataset and copy the group's
        # arrays and dimensions into it.
        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("group")
        dst_rg = dst.GetRootGroup()

        # Copy dimensions from the sub-group
        dim_map = {}
        for gdal_dim in group.GetDimensions() or []:
            dim_name = gdal_dim.GetName()
            new_dim = dst_rg.CreateDimension(
                dim_name, gdal_dim.GetType(), None, gdal_dim.GetSize()
            )
            iv = gdal_dim.GetIndexingVariable()
            if iv is not None:
                coord_arr = dst_rg.CreateMDArray(
                    dim_name,
                    [new_dim],
                    gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(iv.ReadAsArray())),
                )
                coord_arr.Write(iv.ReadAsArray())
                new_dim.SetIndexingVariable(coord_arr)
            dim_map[dim_name] = new_dim

        # Copy arrays from the sub-group
        for arr_name in group.GetMDArrayNames() or []:
            md_arr = group.OpenMDArray(arr_name)
            if md_arr is None:
                continue
            arr_dims = md_arr.GetDimensions()
            # Map source dims to destination dims (by name)
            new_dims = []
            for d in arr_dims:
                d_name = d.GetName()
                if d_name in dim_map:
                    new_dims.append(dim_map[d_name])
                else:
                    # Dimension from parent group — create locally
                    new_d = dst_rg.CreateDimension(
                        d_name, d.GetType(), None, d.GetSize()
                    )
                    dim_map[d_name] = new_d
                    new_dims.append(new_d)
            arr_data = md_arr.ReadAsArray()
            arr_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr_data))
            new_arr = dst_rg.CreateMDArray(arr_name, new_dims, arr_dtype)
            new_arr.Write(arr_data)
            ndv = md_arr.GetNoDataValue()
            if ndv is not None:
                new_arr.SetNoDataValueDouble(ndv)
            srs = md_arr.GetSpatialRef()
            if srs is not None:
                new_arr.SetSpatialRef(srs)

        result = NetCDF(dst)
        return result

    def get_variable_names(self) -> list[str]:
        """Return names of data variables, excluding dimension coordinates.

        Uses CF classification when metadata is cached (fast path).
        Otherwise queries `GetMDArrayNames()` and filters out dimension
        arrays and 0-dimensional scalar variables (grid_mapping etc.).
        In classic mode, parses subdataset metadata.

        Returns:
            list[str]: Variable names (e.g., `["temperature", "precipitation"]`).
        """
        if self._cached_meta_data is not None and self._cached_meta_data.cf is not None:
            variable_names = list(self._cached_meta_data.cf.data_variable_names)
        else:
            rg = self._raster.GetRootGroup()
            if rg is not None:
                all_names = rg.GetMDArrayNames()
                dim_names = {dim.GetName() for dim in rg.GetDimensions()}
                filtered = []
                for var in all_names:
                    if var in dim_names:
                        continue
                    md_arr = rg.OpenMDArray(var)
                    if md_arr is not None and len(md_arr.GetDimensions()) == 0:
                        continue
                    filtered.append(var)
                variable_names = filtered
            else:
                variable_names = [
                    var[1].split(" ")[1] for var in self._raster.GetSubDatasets()
                ]

        return variable_names

    def _read_md_array(self, variable_name: str):
        """Convert an MDArray to a classic GDAL dataset via AsClassicDataset.

        The last two dimensions become X (columns) and Y (rows); all
        remaining dimensions are flattened into bands.

        If the Y dimension is stored south-to-north (positive Y pixel
        size), it is reversed via `MDArray.GetView()` **before** the
        conversion. This is a lazy, zero-copy operation — GDAL handles
        the reversed indexing internally without reading the whole array.

        Returns a tuple `(classic_dataset, md_array, root_group)` so
        callers can keep the GDAL objects alive. `AsClassicDataset`
        returns a **view** whose C++ backing depends on the MDArray and
        root group; if the Python SWIG wrappers for those are garbage-
        collected the view becomes a dangling pointer (segfault on
        Windows).
        """
        rg = self._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        dtype = md_arr.GetDataType()
        dims = md_arr.GetDimensions()

        if len(dims) == 1:
            if dtype.GetClass() == gdal.GEDTC_STRING:
                return md_arr, md_arr, rg
            src = md_arr.AsClassicDataset(0, 1, rg)
            return src, md_arr, rg

        iXDim = len(dims) - 1
        iYDim = len(dims) - 2

        # First pass: check if Y orientation needs flipping.
        src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        if src.GetGeoTransform()[5] > 0:
            # Positive Y pixel size = south-to-north (NetCDF convention).
            # Use GetView to reverse the Y dimension — this is lazy and
            # zero-copy; GDAL handles reversed indexing internally.
            slices = ",".join("::-1" if i == iYDim else ":" for i in range(len(dims)))
            md_arr = md_arr.GetView(f"[{slices}]")
            src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        return src, md_arr, rg

    def get_variable(self, variable_name: str) -> NetCDF:
        """Extract a single variable as a classic-raster NetCDF object.

        The returned object carries origin metadata so modified data
        can be written back via `set_variable()`. Every non-spatial
        dim of the variable is tracked: for an N-D MDIM array
        `(d_0, ..., d_{n-1}, lat, lon)` the build path populates
        `_band_dim_names`, `_band_dim_values_map`, and
        `_band_dim_sizes` with all non-spatial dims in storage order,
        while the legacy `_band_dim_name` / `_band_dim_values` keep
        pointing at the first non-spatial dim so existing 3-D
        consumers see no change. 4-D+ files (e.g. CDS-Beta ERA5
        pressure-levels with `(valid_time, pressure_level, lat, lon)`)
        are addressable via `sel()` along any tracked band dim.

        Supports group-qualified names: `"forecast/temperature"` first
        navigates to the `forecast` sub-group, then extracts
        `temperature` from it.

        Args:
            variable_name: Name of the variable to extract. Use `/`
                to separate group path from variable name.

        Returns:
            NetCDF: A subset backed by a classic dataset where every
                non-spatial dimension is mapped onto bands. The new
                `_band_dim_names` / `_band_dim_values_map` /
                `_band_dim_sizes` fields drive `sel()`; the legacy
                `_band_dim_name` / `_band_dim_values` track the first
                non-spatial dim.

        Raises:
            ValueError: If `variable_name` is not present in the dataset.

        Notes:
            String-typed indexing variables (e.g. WRF's `Times` array)
            cannot be read via GDAL SWIG bindings; the build path falls
            back to integer indices `[0, 1, ..., size - 1]` for those
            dims.

        See Also:
            `sel`: subsets the result along any tracked band dim.
        """
        # Handle group-qualified names: "forecast/temperature"
        if "/" in variable_name:
            parts = variable_name.rsplit("/", 1)
            group_nc = self.get_group(parts[0])
            cube = group_nc.get_variable(parts[1])
            return cube  # single return below handles non-group path

        if variable_name not in self.variable_names:
            raise ValueError(
                f"{variable_name} is not a valid variable name in {self.variable_names}"
            )

        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()
        md_arr_ref = None
        rg_ref = None

        if prefix == "MEMORY" or rg is not None:
            src, md_arr_ref, rg_ref = self._read_md_array(variable_name)
            if isinstance(src, gdal.Dataset):
                cube = NetCDF(src)
                cube._is_md_array = True
                # _read_md_array uses GetView to flip the data lazily,
                # and GDAL usually corrects the geotransform. But when
                # the Y dimension has no indexing variable (e.g. WRF
                # "south_north"), the geotransform may still be wrong.
                # Fix it on the wrapper object (no data copy).
                gt = cube._geotransform
                if gt[5] > 0:
                    cube._geotransform = (
                        gt[0],
                        gt[1],
                        gt[2],
                        gt[3] + gt[5] * cube._rows,
                        gt[4],
                        -gt[5],
                    )
                    cube._cell_size = abs(gt[1])
            else:
                cube = src
            # Keep GDAL SWIG references alive — AsClassicDataset returns a
            # view whose C++ backing is owned by the MDArray/root group.
            # Without these the view becomes a dangling pointer on Windows.
            cube._gdal_md_arr_ref = md_arr_ref
            cube._gdal_rg_ref = rg_ref
        else:
            src = gdal.Open(f"{prefix}:{self.file_name}:{variable_name}")
            if src is None:
                raise ValueError(
                    f"Could not open variable '{variable_name}' via "
                    f"'{prefix}:{self.file_name}:{variable_name}'"
                )
            cube = NetCDF(src)
            cube._is_md_array = False

        cube._is_subset = True

        # --- RT-4: Track variable origin for round-trip ---
        cube._parent_nc = self
        cube._source_var_name = variable_name

        md_arr = md_arr_ref if rg is not None else None
        if rg is not None:
            if md_arr is not None:
                dims = md_arr.GetDimensions()
                cube._md_array_dims = [d.GetName() for d in dims]

                # Identify which dimensions became bands (all except X/Y).
                # Track every non-spatial dim so 4-D+ files (e.g. CDS-Beta
                # ERA5 pressure-levels: time, pressure_level, lat, lon)
                # remain addressable via sel(). Legacy fields point at the
                # primary (first) non-spatial dim so 3-D consumers see no
                # change.
                if len(dims) > 2:
                    spatial_indices = {len(dims) - 1, len(dims) - 2}
                    band_dims = [
                        d for i, d in enumerate(dims) if i not in spatial_indices
                    ]
                else:
                    band_dims = []

                if band_dims:
                    cube._band_dim_names = tuple(d.GetName() for d in band_dims)
                    cube._band_dim_sizes = tuple(d.GetSize() for d in band_dims)
                    cube._band_dim_values_map = {}
                    for d in band_dims:
                        iv = d.GetIndexingVariable()
                        try:
                            values = (
                                iv.ReadAsArray().tolist() if iv is not None else None
                            )
                        except RuntimeError:
                            # String-typed indexing variables (e.g. WRF
                            # "Times") can't be read via ReadAsArray in
                            # GDAL SWIG bindings — fall back to indices.
                            values = list(range(d.GetSize()))
                        cube._band_dim_values_map[d.GetName()] = values
                    cube._band_dim_name = cube._band_dim_names[0]
                    cube._band_dim_values = cube._band_dim_values_map[
                        cube._band_dim_name
                    ]
                else:
                    cube._band_dim_name = None
                    cube._band_dim_values = None
                    cube._band_dim_names = ()
                    cube._band_dim_values_map = {}
                    cube._band_dim_sizes = ()

                # Copy variable attributes
                cube._variable_attrs = {}
                try:
                    for attr in md_arr.GetAttributes():
                        cube._variable_attrs[attr.GetName()] = attr.Read()
                except Exception:
                    pass  # nosec B110

                # Scale/offset for CF packed data
                try:
                    cube._scale = md_arr.GetScale()
                    cube._offset = md_arr.GetOffset()
                except Exception:
                    cube._scale = None
                    cube._offset = None
            else:
                cube._md_array_dims = []
                cube._band_dim_name = None
                cube._band_dim_values = None
                cube._band_dim_names = ()
                cube._band_dim_values_map = {}
                cube._band_dim_sizes = ()
                cube._variable_attrs = {}
                cube._scale = None
                cube._offset = None
        else:
            cube._md_array_dims = []
            cube._band_dim_name = None
            cube._band_dim_values = None
            cube._band_dim_names = ()
            cube._band_dim_values_map = {}
            cube._band_dim_sizes = ()
            cube._variable_attrs = {}
            cube._scale = None
            cube._offset = None

        return cube

    def _replace_raster(self, new_raster: gdal.Dataset):
        """Replace the internal GDAL dataset, closing the old one if different.

        Re-derives all base-class state (geotransform, CRS, band info, etc.)
        without resetting NetCDF-specific flags (_is_md_array, _is_subset).
        """
        old = self._raster
        if old is not None and old is not new_raster:
            old.FlushCache()
        # RasterBase state
        self._raster = new_raster
        self._geotransform = new_raster.GetGeoTransform()
        self._cell_size = self._geotransform[1]
        self._file_name = new_raster.GetDescription()
        self._epsg = self._get_epsg()
        self._rows = new_raster.RasterYSize
        self._columns = new_raster.RasterXSize
        self._band_count = new_raster.RasterCount
        self._block_size = [
            new_raster.GetRasterBand(i).GetBlockSize()
            for i in range(1, self._band_count + 1)
        ]
        # Dataset state
        self._no_data_value = [
            new_raster.GetRasterBand(i).GetNoDataValue()
            for i in range(1, self._band_count + 1)
        ]
        self._band_names = self._get_band_names()
        self._band_units = [
            new_raster.GetRasterBand(i).GetUnitType()
            for i in range(1, self._band_count + 1)
        ]
        # Invalidate caches
        self._cached_variables = None
        self._cached_meta_data = None

    def _invalidate_caches(self):
        """Invalidate cached variables and metadata."""
        self._cached_variables = None
        self._cached_meta_data = None

    @property
    def is_subset(self) -> bool:
        """Whether this object represents a single-variable subset.

        Returns:
            bool: True if the dataset is a variable subset extracted
                via `get_variable()`.
        """
        return self._is_subset

    @property
    def is_md_array(self):
        """Whether this dataset was opened in multidimensional mode.

        Returns:
            bool: True if the dataset was opened via
                `gdal.OF_MULTIDIM_RASTER` and supports groups,
                MDArrays, and dimensions.
        """
        return self._is_md_array

    def to_file(  # type: ignore[override]
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Save the dataset to disk.

        For `.nc` / `.nc4` files the full multidimensional structure
        (groups, dimensions, variables, attributes) is preserved via
        `CreateCopy` with the netCDF driver. For other extensions
        (e.g. `.tif`), the parent `Dataset.to_file` is used — but only
        on variable subsets, not on root MDIM containers.

        Args:
            path: Destination file path. The extension determines the
                output driver (`.nc` -> netCDF, `.tif` -> GeoTIFF, etc.).
            **kwargs: Forwarded to `Dataset.to_file` for non-NetCDF
                extensions (e.g. `tile_length`, `creation_options`).

        Raises:
            RuntimeError: If the netCDF `CreateCopy` call fails.
            ValueError: If a root MDIM container is saved to a non-NC
                extension (use `.nc` or extract a variable first).
        """
        path = Path(path)
        extension = path.suffix[1:].lower()
        if extension in ("nc", "nc4"):
            dst = gdal.GetDriverByName("netCDF").CreateCopy(str(path), self._raster, 0)
            if dst is None:
                raise RuntimeError(f"Failed to save NetCDF to {path}")
            dst.FlushCache()
            dst = None
        else:
            if self._is_md_array and not self._is_subset:
                raise ValueError(
                    "Cannot save a multidimensional NetCDF container as "
                    f"'{extension}'. Use .nc extension or extract a "
                    "variable first with .get_variable()."
                )
            super().to_file(path, **kwargs)

    def copy(self, path: str | Path | None = None) -> NetCDF:
        """Create a deep copy of this NetCDF dataset.

        Args:
            path: Destination file path. If None, the copy is created
                in memory using the MEM driver. Defaults to None.

        Returns:
            NetCDF: A new NetCDF object with copied data.

        Raises:
            RuntimeError: If `CreateCopy` fails.
        """
        if path is None:
            path = ""
            driver = "MEM"
        else:
            driver = "netCDF"

        src = gdal.GetDriverByName(driver).CreateCopy(str(path), self._raster)
        if src is None:
            raise RuntimeError(f"Failed to copy NetCDF dataset to '{path}'")
        return NetCDF(src, access="write")

    @staticmethod
    def _create_dimension(
        group: gdal.Group,
        dim_name: str,
        dtype,
        values: np.ndarray,
        dim_type=None,
        set_indexing: bool = True,
        is_geographic: bool = True,
    ) -> gdal.Dimension:
        """Create a dimension with its coordinate array and CF attributes.

        Args:
            group: GDAL root group.
            dim_name: Dimension name.
            dtype: GDAL ExtendedDataType.
            values: Coordinate values.
            dim_type: GDAL dimension type constant.
            set_indexing: If True, call SetIndexingVariable (works
                on MEM driver). If False, skip it (required for
                netCDF driver which doesn't support it).
            is_geographic: If True, coordinate units are degrees.
                If False, units are metres. Defaults to True.

        Returns:
            gdal.Dimension
        """
        dim = group.CreateDimension(dim_name, dim_type, None, values.shape[0])
        coord_arr = group.CreateMDArray(dim_name, [dim], dtype)
        coord_arr.Write(values)
        if set_indexing:
            dim.SetIndexingVariable(coord_arr)
        cf_attrs = build_coordinate_attrs(dim_name, is_geographic)
        if cf_attrs:
            write_attributes_to_md_array(coord_arr, cf_attrs)
        return dim

    @staticmethod
    def create_main_dimension(
        group: gdal.Group, dim_name: str, dtype: int, values: np.ndarray
    ) -> gdal.Dimension:
        """Create a NetCDF dimension with an indexing variable.

        The dimension type is inferred from `dim_name`:
        `y`/`lat`/`latitude` -> horizontal Y,
        `x`/`lon`/`longitude` -> horizontal X,
        `bands`/`time` -> temporal.

        The dimension is registered in the group together with a
        matching MDArray that stores the coordinate values.

        Args:
            group: Root group (or sub-group) of the multidimensional
                dataset.
            dim_name: Name of the dimension to create.
            dtype: GDAL `ExtendedDataType` for the indexing variable.
            values: Coordinate values for the dimension.

        Returns:
            gdal.Dimension: The newly created dimension.
        """
        if dim_name in ["y", "lat", "latitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_Y
        elif dim_name in ["x", "lon", "longitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_X
        elif dim_name in ["bands", "time"]:
            dim_type = gdal.DIM_TYPE_TEMPORAL
        else:
            dim_type = None
        dim = group.CreateDimension(dim_name, dim_type, None, values.shape[0])
        x_values = group.CreateMDArray(dim_name, [dim], dtype)
        x_values.Write(values)
        dim.SetIndexingVariable(x_values)
        return dim

    @classmethod
    def create_from_array(  # type: ignore[override]
        cls,
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
        resolved_extra_dims = cls._resolve_extra_dims(
            arr=arr,
            extra_dim_name=extra_dim_name,
            extra_dim_values=extra_dim_values,
            extra_dims=extra_dims,
        )

        if arr.ndim == 3:
            DimMetaData(
                name=resolved_extra_dims[0][0],
                size=arr.shape[0],
                values=resolved_extra_dims[0][1],
            )

        if variable_name is None:
            variable_name = "data"

        dst_ds = cls._create_netcdf_from_array(
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
            title=title,
            institution=institution,
            source=source,
            history=history,
        )
        result = cls(dst_ds)

        return result

    @staticmethod
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
            for i, entry in enumerate(extra_dims):
                name, values = entry
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
        return [
            (f"dim_{i}", list(range(int(arr.shape[i]))))
            for i in range(expected)
        ]

    @staticmethod
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
        title: str | None = None,
        institution: str | None = None,
        source: str | None = None,
        history: str | None = None,
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
            title: CF global attribute `title`. Defaults to None.
            institution: CF global attribute `institution`.
                Defaults to None.
            source: CF global attribute `source`.
                Defaults to None.
            history: CF global attribute `history`.
                Defaults to None.

        Returns:
            gdal.Dataset: The created multidimensional GDAL dataset.
        """
        if variable_name is None:
            raise ValueError("Variable_name cannot be None")
        if geo is None:
            raise ValueError("geo cannot be None")

        if extra_dims is None:
            extra_dims = []
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        x_dim_values = NetCDF.get_x_lon_dimension_array(geo[0], geo[1], cols)
        y_dim_values = NetCDF.get_y_lat_dimension_array(geo[3], geo[1], rows)

        if path is not None:
            driver_type = "netCDF"
        else:
            driver_type = "MEM"
            path = "netcdf"

        src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(str(path))
        rg = src.GetRootGroup()

        # Set CF global attributes on root group
        cf_global = {"Conventions": "CF-1.8"}
        if title is not None:
            cf_global["title"] = title
        if institution is not None:
            cf_global["institution"] = institution
        if source is not None:
            cf_global["source"] = source
        if history is not None:
            cf_global["history"] = history
        write_global_attributes(rg, cf_global)

        # Build creation options for chunking and compression
        create_options = []
        if chunk_sizes is not None:
            create_options.append(f"BLOCKSIZE={','.join(str(s) for s in chunk_sizes)}")
        if compression is not None:
            create_options.append(f"COMPRESS={compression}")
        if compression_level is not None:
            create_options.append(f"ZLEVEL={compression_level}")

        # netCDF driver doesn't support SetIndexingVariable — create
        # dimension arrays manually without linking them.
        use_set_indexing = driver_type == "MEM"

        # Determine if CRS is geographic (lon/lat) or projected (m)
        is_geographic = True
        if epsg is not None:
            srs_check = sr_from_epsg(int(epsg))
            is_geographic = srs_check.IsGeographic() == 1

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

        # Build one GDAL dimension per non-spatial axis in storage
        # order, then create the MDArray with `(*extra_dims, dim_y,
        # dim_x)`. This generalises the previous 3-D-only branch: a
        # 2-D array yields zero extra dims; a 3-D array yields one;
        # 4-D+ yields the full set. The first non-spatial dim is
        # tagged `DIM_TYPE_TEMPORAL` (matching the legacy 3-D path),
        # and any additional dims are left untagged so the netCDF
        # driver doesn't second-guess their semantics.
        gdal_extra_dims = []
        for i, (dim_name, dim_values) in enumerate(extra_dims):
            dim_type = gdal.DIM_TYPE_TEMPORAL if i == 0 else None
            gd_dim = NetCDF._create_dimension(
                rg,
                dim_name,
                dtype,
                np.array(dim_values),
                dim_type,
                use_set_indexing,
            )
            gdal_extra_dims.append(gd_dim)
        md_arr = rg.CreateMDArray(
            variable_name,
            [*gdal_extra_dims, dim_y, dim_x],
            dtype,
            create_options if create_options else [],
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
        if epsg is None:
            raise ValueError("epsg cannot be None")
        srse = sr_from_epsg(int(epsg))
        md_arr.SetSpatialRef(srse)
        md_arr.Write(arr)

        # Create CF grid_mapping variable (MEM driver only — the netCDF
        # driver creates its own via SetSpatialRef above). Use
        # "spatial_ref" as the variable name to avoid collision with
        # GDAL's automatic "crs" during CreateCopy to netCDF.
        if driver_type == "MEM":
            gm_name, gm_params = srs_to_grid_mapping(srse)
            gm_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
            gm_var_name = "spatial_ref"
            crs_arr = rg.CreateMDArray(gm_var_name, [], gm_dtype)
            crs_arr.Write(np.array(0, dtype=np.int32))
            gm_params["grid_mapping_name"] = gm_name
            write_attributes_to_md_array(crs_arr, gm_params)
            write_attributes_to_md_array(md_arr, {"grid_mapping": gm_var_name})

        return src

    @staticmethod
    def _add_md_array_to_group(dst_group, var_name, src_mdarray):
        """Copy an MDArray from one group to another, preserving data and metadata."""
        src_dims = src_mdarray.GetDimensions()
        arr = src_mdarray.ReadAsArray()
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        new_md_array = dst_group.CreateMDArray(var_name, src_dims, dtype)
        new_md_array.Write(arr)
        ndv = src_mdarray.GetNoDataValue()
        if ndv is not None:
            try:
                new_md_array.SetNoDataValueDouble(ndv)
            except Exception:
                pass

        new_md_array.SetSpatialRef(src_mdarray.GetSpatialRef())

    @staticmethod
    def _get_or_create_dimension(
        rg: gdal.Group, dim_name: str, values: np.ndarray, dtype, dim_type=None
    ) -> gdal.Dimension:
        """Reuse an existing dimension or create a new one.

        If a dimension with `dim_name` already exists in the root group
        and has the same size as `values`, it is returned directly.
        On size mismatch, a new dimension with a `_{size}` suffix is
        created to avoid conflicts.

        Args:
            rg: The root group of the multidimensional dataset.
            dim_name: Name of the dimension (e.g., `"x"`, `"time"`).
            values: Coordinate values for this dimension.
            dtype: GDAL `ExtendedDataType` for the indexing variable.
            dim_type: GDAL dimension type constant (e.g.,
                `gdal.DIM_TYPE_HORIZONTAL_X`). Defaults to None.

        Returns:
            gdal.Dimension: The reused or newly created dimension.
        """
        for existing_dim in rg.GetDimensions() or []:
            if existing_dim.GetName() == dim_name:
                if existing_dim.GetSize() == len(values):
                    return existing_dim
                # Size mismatch — need a new dimension with a unique name
                dim_name = f"{dim_name}_{len(values)}"
                break

        return NetCDF.create_main_dimension(rg, dim_name, dtype, values)

    @property
    def global_attributes(self) -> dict[str, Any]:
        """Global attributes from the root group.

        Returns a live dict read from the GDAL root group each time.
        For MDIM mode, reads from the root group's attributes.
        For classic mode, reads from GDAL's `GetMetadata()`.

        Returns:
            dict[str, Any]: Key-value mapping of global attributes.
        """
        rg = self._raster.GetRootGroup()
        result = {}
        if rg is not None:
            try:
                for attr in rg.GetAttributes():
                    result[attr.GetName()] = attr.Read()
            except Exception:
                pass
        else:
            result = dict(self._raster.GetMetadata())
        return result

    def set_global_attribute(self, name: str, value: Any):
        """Set a global attribute on the root group.

        Creates or updates a single attribute on the root group.

        Args:
            name: Attribute name (e.g. `"history"`,
                `"Conventions"`).
            value: Attribute value. Supports str, int, float.

        Raises:
            ValueError: If the dataset has no root group
                (not opened in MDIM mode).
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_global_attribute requires a multidimensional "
                "container. Open the file with "
                "open_as_multi_dimensional=True."
            )
        # Delete existing attribute if present (GDAL raises on duplicate)
        try:
            rg.DeleteAttribute(name)
        except Exception:
            pass
        if isinstance(value, str):
            attr = rg.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
        elif isinstance(value, float):
            attr = rg.CreateAttribute(
                name, [], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
            )
        elif isinstance(value, int):
            attr = rg.CreateAttribute(
                name, [], gdal.ExtendedDataType.Create(gdal.GDT_Int32)
            )
        else:
            attr = rg.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString())
            value = str(value)
        attr.Write(value)
        self._invalidate_caches()

    def delete_global_attribute(self, name: str):
        """Delete a global attribute from the root group.

        If the attribute does not exist, the call is silently ignored.

        Args:
            name: Attribute name to delete.

        Raises:
            ValueError: If the dataset has no root group.
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "delete_global_attribute requires a multidimensional " "container."
            )
        try:
            rg.DeleteAttribute(name)
        except Exception:
            pass  # attribute may not exist — silently ignored
        self._invalidate_caches()

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
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

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
        if variable_name in self.variable_names:
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
            NetCDF.get_x_lon_dimension_array(gt[0], gt[1], dataset.columns)
        )
        y_values = np.array(
            NetCDF.get_y_lat_dimension_array(gt[3], abs(gt[5]), dataset.rows)
        )
        dim_x = self._get_or_create_dimension(
            rg, "x", x_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_X
        )
        dim_y = self._get_or_create_dimension(
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
                    self._get_or_create_dimension(
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
            dim_band = self._get_or_create_dimension(
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
            except Exception:
                pass  # nosec B110

        # Set variable attributes (RT-7)
        if attrs:
            write_attributes_to_md_array(md_arr, attrs)

        self._invalidate_caches()

    def crop_variable(
        self, variable_name: str, mask: Any, touch: bool = True
    ) -> NetCDF:
        """Crop a single variable and store the result back.

        Convenience method that combines `get_variable` → `crop`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to crop.
            mask: GeoDataFrame with polygon geometry, or a Dataset
                to use as a spatial mask.
            touch: If True, include cells touching the mask boundary.
                Defaults to True.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        cropped = var.crop(mask, touch=touch)
        self.set_variable(variable_name, cropped)
        return self

    def reproject_variable(
        self, variable_name: str, to_epsg: int, method: str = "nearest neighbor"
    ) -> NetCDF:
        """Reproject a single variable and store the result back.

        Convenience method that combines `get_variable` → `to_crs`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to reproject.
            to_epsg: Target EPSG code (e.g. 4326, 32637).
            method: Resampling method. Defaults to
                `"nearest neighbor"`.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        reprojected = var.to_crs(to_epsg, method=method)
        # to_crs returns a VRT-backed dataset — materialize it into
        # a MEM dataset so the data survives after the VRT source
        # (the variable subset) is garbage collected.
        arr = reprojected.read_array()
        no_data_value = reprojected.no_data_value
        ndv_scalar = (
            no_data_value[0]
            if isinstance(no_data_value, (list, tuple)) and no_data_value
            else no_data_value
        )
        materialized = Dataset.create_from_array(
            arr,
            geo=reprojected.geotransform,
            epsg=reprojected.epsg,
            no_data_value=ndv_scalar,
        )
        materialized._band_dim_name = var._band_dim_name
        materialized._band_dim_values = var._band_dim_values
        materialized._band_dim_names = var._band_dim_names
        materialized._band_dim_values_map = dict(var._band_dim_values_map)
        materialized._band_dim_sizes = var._band_dim_sizes
        materialized._variable_attrs = var._variable_attrs
        self.set_variable(variable_name, materialized)
        return self

    def resample_variable(
        self,
        variable_name: str,
        cell_size: int | float,
        method: str = "nearest neighbor",
    ) -> NetCDF:
        """Resample a single variable and store the result back.

        Convenience method that combines `get_variable` → `resample`
        → `set_variable` in one call.

        Args:
            variable_name: Name of the variable to resample.
            cell_size: New cell size.
            method: Resampling method. Defaults to
                `"nearest neighbor"`.

        Returns:
            NetCDF: This container (modified in-place).
        """
        var = self.get_variable(variable_name)
        resampled = var.resample(cell_size, method=method)
        self.set_variable(variable_name, resampled)
        return self

    def add_variable(self, dataset: Dataset | NetCDF, variable_name: str | None = None):
        """Copy MDArray variables from another NetCDF into this container.

        Args:
            dataset: Source NetCDF dataset whose variables will be copied.
                Must have a root group (opened in MDIM mode).
            variable_name: Specific variable name(s) to copy. If None, all
                variables from the source are copied. If a variable with
                the same name already exists, it is renamed with a
                `"-new"` suffix.
        """
        src_rg = self._raster.GetRootGroup()
        var_rg = dataset._raster.GetRootGroup()
        names_to_copy: list[str]
        if variable_name is not None:
            names_to_copy = [variable_name]
        elif isinstance(dataset, NetCDF):
            names_to_copy = dataset.variable_names
        else:
            names_to_copy = []

        for var in names_to_copy:
            md_arr = var_rg.OpenMDArray(var)
            # If the variable name already exists in the destination dataset,
            # use a suffixed name to avoid overwriting the original.
            target_name = f"{var}-new" if var in self.variable_names else var
            self._add_md_array_to_group(src_rg, target_name, md_arr)
        self._invalidate_caches()

    def remove_variable(self, variable_name: str):
        """Delete a variable from this container.

        If the dataset is backed by a file on disk, a MEM copy is made first
        so that the on-disk file is not modified. The internal raster
        reference is replaced with the modified copy.

        Args:
            variable_name: Name of the variable to remove.
        """
        if self.driver_type == "memory":
            dst = self._raster
        else:
            dst = gdal.GetDriverByName("MEM").CreateCopy("", self._raster, 0)

        rg = dst.GetRootGroup()
        rg.DeleteMDArray(variable_name)

        self._replace_raster(dst)

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
        if old_name not in self.variable_names:
            raise ValueError(
                f"Variable '{old_name}' not found. " f"Available: {self.variable_names}"
            )
        if new_name in self.variable_names:
            raise ValueError(f"Variable '{new_name}' already exists.")

        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError("rename_variable requires a multidimensional container.")

        md_arr = rg.OpenMDArray(old_name)
        self._add_md_array_to_group(rg, new_name, md_arr)
        rg.DeleteMDArray(old_name)
        self._invalidate_caches()

    def to_xarray(self) -> Any:
        """Convert this NetCDF container to an `xarray.Dataset`.

        Builds an in-memory `xarray.Dataset` that mirrors the
        variables, coordinates, dimensions, and global attributes of
        this pyramids NetCDF container.

        The entire conversion goes through GDAL's Multidimensional
        API — the same reader the rest of pyramids' NetCDF code uses.
        No xarray engine plugin (`netcdf4`, `h5netcdf`,
        `scipy.io.netcdf`) is involved, so the `[xarray]` extra
        does not need to pull a NetCDF backend: pyramids is the
        backend. The returned `xr.Dataset` holds already-
        materialised numpy arrays; for lazy reads use
        :meth:`read_array(chunks=...)` and wrap the result in
        :class:`xarray.DataArray` yourself.

        Requires the optional `xarray` package. Install it with::

            pip install 'pyramids-gis[xarray]'

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
        try:
            import xarray as xr
        except ImportError:
            raise OptionalPackageDoesNotExist(
                "xarray is required for to_xarray(). "
                "Install it with: pip install 'pyramids-gis[xarray]'"
            )

        rg = self._raster.GetRootGroup()
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
            coord_attrs: dict[str, Any] = {}
            try:
                for attr in iv.GetAttributes():
                    coord_attrs[attr.GetName()] = attr.Read()
            except Exception:
                pass
            unit = iv.GetUnit()
            if unit and "units" not in coord_attrs:
                coord_attrs["units"] = unit
            coords[dim_name] = ([dim_name], iv.ReadAsArray(), coord_attrs)

        data_vars: dict[str, Any] = {}
        for var_name in self.variable_names:
            md_arr = rg.OpenMDArray(var_name)
            if md_arr is None:
                continue
            arr_dims = md_arr.GetDimensions() or []
            arr_dim_names = [ad.GetName() for ad in arr_dims]
            arr_data = md_arr.ReadAsArray()
            var_attrs: dict[str, Any] = {}
            try:
                for attr in md_arr.GetAttributes():
                    var_attrs[attr.GetName()] = attr.Read()
            except Exception:
                pass
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
            attrs=self.global_attributes,
        )
        return result

    @classmethod
    def from_xarray(
        cls,
        dataset: Any,
        path: str | Path | None = None,
    ) -> NetCDF:
        """Create a pyramids NetCDF from an `xarray.Dataset`.

        Extracts dimensions, coordinates, data variables, and
        attributes from the `xarray.Dataset` and writes them to a
        NetCDF file through pyramids' own GDAL Multidimensional
        writer. No xarray engine plugin (`netcdf4`, `h5netcdf`)
        is invoked — pyramids is the writer, so the `[xarray]`
        extra does not need to pull a NetCDF backend.

        Usage::

            ds = xr.open_dataset("input.nc")
            #... xarray processing...
            nc = NetCDF.from_xarray(ds)
            var = nc.get_variable("temperature")
            cropped = var.crop(mask)

        Requires the optional `xarray` package.

        Args:
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
                "xarray is required for from_xarray(). "
                "Install it with: pip install 'pyramids-gis[xarray]'"
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

        mem_src = cls._build_multidim_from_xarray(dataset)
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

    @staticmethod
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
