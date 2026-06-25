"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""

from __future__ import annotations

import gc
import itertools
import math
import os
import tempfile
import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from osgeo import gdal, osr
from shapely import contains_xy

from pyramids import _io
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.base.crs import sr_from_epsg
from pyramids.base.protocols import ArrayLike
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.dataset.dataset import _COLLABORATOR_ATTRS
from pyramids.feature import FeatureCollection
from pyramids.netcdf._kerchunk_facade import combine_kerchunk, to_kerchunk
from pyramids.netcdf._lazy import apply_unpack, build_lazy_array
from pyramids.netcdf._mfdataset import open_mfdataset
from pyramids.netcdf._plot import NetCDFPlot
from pyramids.netcdf.engines import interop as _interop
from pyramids.netcdf.engines.interop import Interop
from pyramids.netcdf.engines.variables import Variables
from pyramids.netcdf.cf import (
    build_coordinate_attrs,
    detect_axis,
    srs_to_grid_mapping,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf._mdim import (
    needs_y_flip,
    open_mdarray,
    scalar_no_data,
)
from pyramids.netcdf.dimensions import ClassicDimensionInfo
from pyramids.netcdf.metadata import get_metadata
from pyramids.netcdf.models import NetCDFMetadata
from pyramids.netcdf.plot_options import ColorOpts, FacetSpec, Selectors
from pyramids.netcdf.utils import _read_attributes, create_time_conversion_func

# GDAL integer dtype codes, used to pick the right typed Attribute.Write* call
# when copying variable attributes (numeric tuples can't go through Write/WriteRaw).
_GDAL_INTEGER_DTYPES = {
    gdal.GDT_Byte,
    gdal.GDT_Int16,
    gdal.GDT_UInt16,
    gdal.GDT_Int32,
    gdal.GDT_UInt32,
    gdal.GDT_Int64,
    gdal.GDT_UInt64,
}
if hasattr(gdal, "GDT_Int8"):
    _GDAL_INTEGER_DTYPES.add(gdal.GDT_Int8)

# Attribute names of the NetCDF-specific engine collaborators (issue #615,
# STR-1). Kept distinct from the eight inherited `Dataset` engines in
# `_COLLABORATOR_ATTRS` so wiring them in `NetCDF.__init__` never clobbers an
# inherited engine. `_update_inplace` re-binds these alongside the Dataset ones
# after the `__dict__` swap.
_NETCDF_COLLABORATOR_ATTRS = ("interop", "varops")


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
        self._names: list[str] = nc.variable_names

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


_REDUCERS: dict[str, tuple[Any, Any]] = {
    "mean": (np.nanmean, np.mean),
    "sum": (np.nansum, np.sum),
    "min": (np.nanmin, np.min),
    "max": (np.nanmax, np.max),
    "std": (np.nanstd, np.std),
    "var": (np.nanvar, np.var),
}
"""Per-operation `(skipna_func, plain_func)` pairs for :meth:`NetCDF.reduce`."""

# GDAL's WKT ``PROJECTION`` node value for the CF geostationary projection
# (set by ``osr.SpatialReference.SetGEOS`` / reconstructed from a
# ``grid_mapping_name="geostationary"`` grid-mapping).
GEOSTATIONARY_PROJECTION = "Geostationary_Satellite"

# Well-known dimension names for the spatial axes, used by
# ``NetCDF._detect_spatial_axes`` as a fallback when the coordinate variables
# carry no CF axis / standard_name / units attributes (e.g. NWM names them
# plainly ``x`` / ``y``). Compared case-insensitively.
_Y_DIM_NAMES = frozenset(
    {"y", "lat", "latitude", "rlat", "south_north", "northing", "grid_latitude"}
)
_X_DIM_NAMES = frozenset(
    {"x", "lon", "longitude", "rlon", "west_east", "easting", "grid_longitude"}
)
# Dimension names that clearly denote a non-spatial axis. Used to suppress the
# "demoted grid" warning for legitimately non-spatial N-D aux variables
# (e.g. a (time, level) table, time-bounds (time, nv)) — only a variable with
# >= 2 *unrecognised* axes is a likely unrecognised grid.
_NONSPATIAL_AXIS_NAMES = frozenset(
    {
        "time",
        "valid_time",
        "t",
        "step",
        "forecast_period",
        "lead_time",
        "level",
        "lev",
        "plev",
        "pressure",
        "depth",
        "height",
        "z",
        "altitude",
        "member",
        "number",
        "ensemble",
        "realization",
        "bnds",
        "bounds",
        "nv",
        "nbnds",
        "nvertices",
        "vertices",
        "string_length",
        "nchar",
        "nstrings",
    }
)


def _contiguous_range(
    coords: np.ndarray,
    low: float,
    high: float,
    axis_name: str,
    bbox: Any,
) -> tuple[int, int]:
    """Half-open index range of the cells whose coordinate is in ``[low, high]``.

    Works for an ascending or descending coordinate axis (the NWM ``y`` axis is
    stored ascending), returning a single contiguous slice that covers the
    in-range cells. Raises when the window selects nothing.

    Args:
        coords: 1-D coordinate values for the axis (any monotonic direction).
        low: Inclusive lower bound of the wanted coordinate window.
        high: Inclusive upper bound of the wanted coordinate window.
        axis_name: Axis label (``"x"`` / ``"y"``) used only in the error message.
        bbox: The originating bbox, echoed back in the error message.

    Returns:
        The ``(start, stop)`` half-open index range covering the in-range cells.

    Raises:
        ValueError: When no cell's coordinate falls in ``[low, high]``.

    Examples:
        - In-range cells on an ascending axis become a half-open slice:
            ```python
            >>> import numpy as np
            >>> _contiguous_range(np.array([0.0, 1.0, 2.0, 3.0]), 1.0, 2.0, "x", (1, 0, 2, 0))
            (1, 3)

            ```
        - A descending axis still yields one contiguous slice:
            ```python
            >>> import numpy as np
            >>> _contiguous_range(np.array([3.0, 2.0, 1.0, 0.0]), 1.0, 2.0, "y", (0, 1, 0, 2))
            (1, 3)

            ```
        - A window outside the axis raises:
            ```python
            >>> import numpy as np
            >>> _contiguous_range(np.array([0.0, 1.0, 2.0]), 9.0, 9.5, "x", (9, 9, 10, 10))
            Traceback (most recent call last):
                ...
            ValueError: bbox (9, 9, 10, 10) selects no cells on the x axis; the store spans [0.0, 2.0] there.

            ```
    """
    inside = np.flatnonzero((coords >= low) & (coords <= high))
    if inside.size == 0:
        raise ValueError(
            f"bbox {tuple(bbox)} selects no cells on the {axis_name} axis; the "
            f"store spans [{float(coords.min())}, {float(coords.max())}] there."
        )
    return int(inside.min()), int(inside.max()) + 1


def _clamp_bound(index: int, size: int) -> int:
    """Wrap a negative slice bound against ``size`` and clamp it into ``[0, size]``.

    Args:
        index: Raw slice bound (may be negative or out of range).
        size: Length of the dimension the bound indexes into.

    Returns:
        The normalised bound, in ``[0, size]``.

    Examples:
        - A negative bound counts from the end:
            ```python
            >>> _clamp_bound(-1, 5)
            4

            ```
        - An out-of-range bound clamps to the nearest edge:
            ```python
            >>> _clamp_bound(99, 5)
            5
            >>> _clamp_bound(-99, 5)
            0

            ```
    """
    if index < 0:
        index += size
    return max(0, min(index, size))


def _range_bounds(
    selector: slice | tuple | list, size: int, dim_name: str
) -> tuple[int, int]:
    """Half-open ``(start, stop)`` for a ``slice`` or 2-element ``(start, stop)``.

    Bounds are wrapped (negatives) and clamped into ``[0, size]`` via
    :func:`_clamp_bound`. Raises when a tuple/list is not length 2.
    """
    if isinstance(selector, slice):
        start = 0 if selector.start is None else _clamp_bound(int(selector.start), size)
        stop = size if selector.stop is None else _clamp_bound(int(selector.stop), size)
        return start, stop
    if len(selector) != 2:
        raise ValueError(
            f"range selector for {dim_name!r} must be (start, stop); got {selector!r}."
        )
    return _clamp_bound(int(selector[0]), size), _clamp_bound(int(selector[1]), size)


def _resolve_index_selector(selector: Any, size: int, dim_name: str) -> tuple[int, int]:
    """Resolve a non-spatial dimension selector to a half-open ``(start, stop)``.

    ``None`` is allowed only for a length-1 dimension; an ``int`` selects one
    index (negative counts from the end, and an out-of-range index is an error);
    a 2-tuple/list or ``slice`` selects a half-open index range whose bounds are
    wrapped (negatives) and clamped into ``[0, size]``.

    Args:
        selector: ``None``, an ``int``, a ``(start, stop)`` tuple/list, or a
            ``slice`` describing which index/indices of the dimension to keep.
        size: Length of the dimension being selected.
        dim_name: Dimension name, used in error messages.

    Returns:
        The resolved ``(start, stop)`` half-open index range.

    Raises:
        ValueError: When ``selector`` is ``None`` for a dimension longer than 1,
            when a tuple/list is not length 2, when an ``int`` index is out of
            range, or when a range resolves to an empty window (``stop <= start``).

    Examples:
        - An integer selects a single index as a half-open range:
            ```python
            >>> _resolve_index_selector(2, 10, "time")
            (2, 3)

            ```
        - A reversed range is rejected (empty window):
            ```python
            >>> _resolve_index_selector((3, 1), 10, "time")
            Traceback (most recent call last):
                ...
            ValueError: selector (3, 1) resolves to an empty index range (start=3, stop=1) for dimension 'time'.

            ```
        - A ``(start, stop)`` pair is wrapped and clamped into ``[0, size]``:
            ```python
            >>> _resolve_index_selector((-3, 999), 5, "time")
            (2, 5)

            ```
        - A length-1 dimension may be left unselected:
            ```python
            >>> _resolve_index_selector(None, 1, "vis_nir")
            (0, 1)

            ```
        - An out-of-range integer is rejected:
            ```python
            >>> _resolve_index_selector(99, 5, "time")
            Traceback (most recent call last):
                ...
            ValueError: index 99 is out of range for dimension 'time' of length 5.

            ```
    """
    if selector is None:
        if size == 1:
            return 0, 1
        raise ValueError(
            f"dimension {dim_name!r} has length {size} and must be selected "
            f"(e.g. {dim_name}=0)."
        )
    if isinstance(selector, (slice, tuple, list)):
        start, stop = _range_bounds(selector, size, dim_name)
        if stop <= start:
            raise ValueError(
                f"selector {selector!r} resolves to an empty index range "
                f"(start={start}, stop={stop}) for dimension {dim_name!r}."
            )
        return start, stop
    index = int(selector)
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise ValueError(
            f"index {selector!r} is out of range for dimension {dim_name!r} "
            f"of length {size}."
        )
    return index, index + 1


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
        if type(self) is NetCDF:
            # API-1 (#614): NetCDF is now the base of Container / Variable.
            # Direct construction is deprecated; the typed entry points return the right
            # concrete class. Subclass construction (type(self) is a subclass) is silent.
            warnings.warn(
                "Directly constructing NetCDF is deprecated and will stop returning a "
                "usable instance in a future major release. Open a store with "
                "NetCDF.read_file(...) / NetCDF.create_from_array(...) (returns a "
                "Container) and extract variables with container.get_variable(...) "
                "(returns a Variable). NetCDF remains an isinstance-compatible base "
                "for one major version.",
                DeprecationWarning,
                stacklevel=2,
            )
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
        # Keeps the source MDIM view alive when a geostationary variable is
        # re-georeferenced through an in-memory VRT (see
        # _normalize_geostationary_geotransform).
        self._gdal_classic_src_ref: Any = None
        # True once a geostationary scan-angle geotransform has been rescaled to
        # metres on this cube; tells the `geotransform` property to trust the
        # stored geotransform instead of re-deriving radian spacing from x/y.
        self._geostationary_scaled: bool = False
        # Per-variable cache of the classic-driver geostationary geotransform, populated
        # on the parent container so each variable's metre geotransform is resolved with
        # at most one `NETCDF:<file>:<var>` open instead of re-opening on every access.
        self._geostationary_gt_cache: dict[str, tuple[float, ...] | None] = {}
        self._md_array_dims: list[str] = []
        self._band_dim_name: str | None = None
        self._band_dim_values: list[Any] | None = None
        self._band_dim_names: tuple[str, ...] = ()
        self._band_dim_values_map: dict[str, list[Any] | None] = {}
        self._band_dim_sizes: tuple[int, ...] = ()
        self._variable_attrs: dict[str, Any] = {}
        self._scale: float | None = None
        self._offset: float | None = None
        # NetCDF-specific engine collaborators (issue #615, STR-1). Distinct
        # attribute names from the eight inherited Dataset engines so they do
        # not clobber `self.io` / `self.spatial` / … . NetCDF exposes thin
        # façade methods that delegate here (e.g. `nc.to_xarray()` ->
        # `self.interop.to_xarray()`).
        self.interop = Interop(self)
        # `varops` (not `variables`) because `variables` is an existing read-side
        # property returning the lazy variable dict — the engine must not shadow it.
        self.varops = Variables(self)

    def close(self) -> None:
        """Release every GDAL handle this container holds, then close the base.

        A NetCDF container keeps more open GDAL references than the single
        ``self._raster`` that :meth:`Dataset.close` drops:

        * cached per-variable child objects (:attr:`_cached_variables`), each
          with its own ``_raster`` and SWIG MDArray / root-group references;
        * the ``_gdal_md_arr_ref`` / ``_gdal_rg_ref`` views that keep an
          extracted variable's C++ backing alive;
        * ``_gdal_classic_src_ref`` (the source kept alive behind a
          geostationary VRT);
        * the ``_parent_nc`` back-reference forming a refcount cycle with the
          parent's variable cache.

        This override closes the cached children and drops every extra reference
        before deferring to :meth:`Dataset.close`. It then runs a single
        :func:`gc.collect`: a spatial op (``crop`` / ``to_crs`` / ``reduce``)
        extracts variables whose ``AsClassicDataset`` view, MDArray and root
        group form a **reference cycle among the GDAL SWIG wrappers** that plain
        refcounting cannot reclaim. Without the collect those wrappers keep the
        source file open, so on Windows a later ``os.replace`` / ``os.remove``
        fails with ``PermissionError`` until the caller forces a GC themselves
        (#564). Doing it here makes ``close()`` honour its contract — the file is
        unlocked immediately. Safe to call more than once.
        """
        cached = self._cached_variables
        if cached is not None:
            # Only the materialised children (bypass the lazy dict's loading
            # ``values()`` so closing does not force every variable open).
            for child in dict.values(cached):
                if isinstance(child, Dataset):
                    child.close()
            self._cached_variables = None
        self._gdal_md_arr_ref = None
        self._gdal_rg_ref = None
        self._gdal_classic_src_ref = None
        # The ad-hoc view/warp keep-alive pins are set only on some code paths (a
        # GetView Y-flip or a to_crs warp), so they are not initialised in __init__;
        # clear them here too — otherwise they hold their backing GDAL handle past
        # close(), defeating the handle-release contract the gc.collect() below enforces.
        self._view_source = None
        self._warp_source = None
        self._parent_nc = None
        self._cached_meta_data = None
        super().close()
        # Break the GDAL SWIG view/MDArray/root-group cycle left by variable
        # extraction so the source file is released now, not on the next GC.
        gc.collect()

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
            "_gdal_classic_src_ref": self._gdal_classic_src_ref,
            "_geostationary_scaled": self._geostationary_scaled,
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
        # Rebuild via the concrete subclass (Container / Variable) so an
        # in-place update never downgrades the instance's type back to the base NetCDF.
        new = type(self)(
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
        # Reuse the canonical Dataset collaborator list (single source of truth in dataset.py) so a
        # new engine — e.g. `georef` — is rebound here too instead of silently keeping a dead
        # back-ref, then rebind the NetCDF-specific engines (`interop`, …) the same way.
        for attr in (*_COLLABORATOR_ATTRS, *_NETCDF_COLLABORATOR_ATTRS):
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

        Geostationary scan-angle datasets are the exception: once their
        ``x`` / ``y`` radians have been rescaled to metres on read (see
        :meth:`_normalize_geostationary_geotransform`), re-deriving the
        geotransform from the raw radian coordinates would be wrong, so the
        stored metre geotransform is authoritative. The check is a cheap
        boolean flag, not an SRS parse, so it adds no cost to ordinary reads.

        Returns:
            tuple[float, float, float, float, float, float]: The GDAL
            geotransform ``(x_min, pixel_width, row_rotation, y_max,
            column_rotation, pixel_height)``. ``pixel_height`` is negative for
            a north-up raster. Units follow the dataset CRS (degrees for
            geographic, metres for projected, including rescaled geostationary).
        """
        if self._geostationary_scaled:
            return self._geotransform
        if self.lon is not None and self.lat is not None:
            # Derive the X and Y pixel sizes independently from the lon/lat coordinate spacing —
            # they differ on non-square grids (e.g. 2° lon, 1° lat), so a single `cell_size` for
            # both axes would stretch the latitude axis. Y is reported north-up: the top edge is the
            # northernmost latitude plus half a cell, and pixel height is negative.
            x_cell = self.cell_size
            if len(self.lat) >= 2:
                y_cell = abs(float(self.lat[1] - self.lat[0]))
                y_top = max(float(self.lat[0]), float(self.lat[-1])) + y_cell / 2
            else:
                y_cell = self.cell_size
                y_top = float(self.lat[0]) + y_cell / 2
            return (
                self.lon[0] - x_cell / 2,
                x_cell,
                0,
                y_top,
                0,
                -y_cell,
            )
        return self._geotransform

    def _is_geostationary(self) -> bool:
        """True when the dataset CRS is the CF geostationary projection.

        Parses the SRS, so it is called sparingly (during read normalization,
        not from the hot `geotransform` property — that uses the
        `_geostationary_scaled` flag instead).
        """
        srs = self._raster.GetSpatialRef() if self._raster is not None else None
        return bool(
            srs is not None
            and srs.GetAttrValue("PROJECTION") == GEOSTATIONARY_PROJECTION
        )

    def _classic_geotransform(self) -> tuple[float, ...] | None:
        """Metre geotransform from GDAL's classic netCDF driver for this var.

        The classic ``NETCDF:<file>:<var>`` driver georeferences CF
        geostationary files correctly — it applies the ``x`` / ``y``
        ``scale_factor`` / ``add_offset`` (real GOES stores them as packed
        ``int16`` scan angles) and scales the radians to projected metres by
        ``perspective_point_height``. The multidimensional ``AsClassicDataset``
        path this cube comes from does neither, so it yields a raw pixel or
        radian geotransform.

        Returns:
            tuple | None: The classic-driver geotransform, or ``None`` when
            there is no classic-openable source (e.g. an in-memory dataset) or
            the classic open does not produce a metre-scale geostationary
            geotransform.
        """
        parent = self._parent_nc
        var = self._source_var_name
        if parent is None or var is None:
            return None
        # Resolve once per variable and memoise on the parent: every spatial variable
        # would otherwise re-open `NETCDF:<file>:<var>` to recompute the same metre
        # geotransform. The cached value (a geotransform tuple or ``None``) is reused on
        # subsequent accesses of the same variable.
        cache = parent._geostationary_gt_cache
        if var in cache:
            return cache[var]

        result: tuple[float, ...] | None = None
        path = parent.file_name
        # The classic netCDF driver needs an on-disk / VSI source; an in-memory
        # MEM dataset has no such path.
        if path and not str(path).startswith("/vsimem"):
            try:
                src = gdal.Open(f"NETCDF:{path}:{var}")
            except RuntimeError:
                src = None
            if src is not None:
                gt = src.GetGeoTransform()
                srs = src.GetSpatialRef()
                if (
                    srs is not None
                    and srs.GetAttrValue("PROJECTION") == GEOSTATIONARY_PROJECTION
                    and abs(gt[1]) > 1.0
                ):
                    result = gt

        cache[var] = result
        return result

    def _normalize_geostationary_geotransform(self) -> None:
        """Georeference a geostationary variable read via the MDIM path.

        Real GOES (and other CF geostationary) files store ``x`` / ``y`` as
        scan angles that the classic netCDF driver scales to projected metres by
        ``perspective_point_height`` (after applying their ``scale_factor`` /
        ``add_offset``). The multidimensional ``AsClassicDataset`` path used by
        :meth:`get_variable` does neither, so the cube comes back with a raw
        pixel/radian geotransform under a metre-based geostationary CRS and
        ``to_crs`` collapses. Adopt the classic driver's metre geotransform so
        the cube is correctly georeferenced; a no-op for every other CRS.

        The corrected geotransform is applied to the underlying GDAL dataset via
        an in-memory VRT (the MDIM ``AsClassicDataset`` view has no driver and
        silently ignores ``SetGeoTransform``), so ``to_crs`` — which warps
        ``self.raster`` — sees it. The source view is kept alive (the VRT
        references it; the view is backed by the MDArray refs already held).

        Side effects for a rescaled geostationary cube:

        * ``self.raster`` becomes a VRT with no MDIM root group, so coordinate
          accessors (`lon` / `lat` / `x` / `y`) report the projected **metre**
          coordinates derived from the geotransform, not the raw ``x`` / ``y``.
        * ``crop(bbox=...)`` against the cube's own geostationary CRS can fail
          inside PROJ (off-disc cutline). Reproject with ``to_crs(4326)`` and
          crop the result.
        """
        if not self._is_geostationary():
            return
        correct = self._classic_geotransform()
        if correct is None:
            return
        if self._geotransform == correct:
            # Already georeferenced (e.g. opened via the classic read path).
            self._geostationary_scaled = True
            return
        vrt = gdal.Translate("", self._raster, format="VRT")
        if vrt is not None and vrt.SetGeoTransform(correct) == gdal.CE_None:
            self._gdal_classic_src_ref = self._raster
            self._raster = vrt
        else:
            warnings.warn(
                "could not georeference the geostationary view through a VRT; "
                "the wrapper geotransform reports metres but the underlying "
                "dataset keeps its raw scan-angle grid, so to_crs/crop may be "
                "wrong.",
                stacklevel=3,
            )
        self._geotransform = correct
        self._cell_size = abs(correct[1])
        self._geostationary_scaled = True

    @property
    def variable_names(self) -> list[str]:
        """Names of data variables (excluding dimension coordinate arrays).

        Returns:
            list[str]: Variable names. For MDIM mode these come from
            `GetMDArrayNames()` minus dimension names; for classic mode
            from `GetSubDatasets()`.
        """
        return self._get_variable_names()

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
        selectors: Selectors | None = None,
        colour: ColorOpts | None = None,
        facet: FacetSpec | None = None,
        coords: tuple | list | None = None,
        x_dim: str | None = None,
        y_dim: str | None = None,
        kind: str = "auto",
        animate: bool | str | None = None,
        chunks: Any | None = None,
        basemap: bool | str | None = None,
        exclude_value: Any | None = None,
        title: str | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        """Plot a 2-D slice of a NetCDF variable using xarray-aligned vocabulary.

        The public surface is shaped around **variables** and **dimensions** — ``band``
        is not a NetCDF concept and has been removed from the signature. Variable
        selection is by name; the slice to render is pinned via a :class:`Selectors`
        option bag (``time`` / ``level`` / ``member`` / ``sel`` / ``isel``); colour
        controls live on a :class:`ColorOpts` bag (``cmap`` / ``vmin`` / ``vmax`` /
        ``robust`` / ``levels`` / ``norm`` / ``center`` / ``extend`` / ``add_colorbar``
        / ``cbar_kwargs``); multi-panel layout is described by a :class:`FacetSpec`
        bag (``col`` / ``row`` / ``col_wrap``). Each bag is a frozen dataclass —
        construct it inline at the call site.

        On a **root MDIM container** the ``variable=`` argument is required:

        ```python
        from pyramids.netcdf import NetCDF, Selectors
        nc.plot(variable="t2m", selectors=Selectors(time="2024-01-15"))
        ```

        On a **variable subset** (the result of :meth:`get_variable`) ``variable=``
        may be omitted or must equal the pinned variable name; otherwise the call
        is rejected, mirroring the :meth:`read_array` contract.

        Args:
            variable (str, optional):
                Name of the variable to plot. Required on the root MDIM container;
                must be ``None`` or equal to the pinned variable name on a subset.
                Defaults to None.
            selectors (Selectors, optional):
                Dim-selector bag. See :class:`Selectors` for the field list. A
                missing bag is treated as :class:`Selectors`\\ () (all fields
                ``None``). Defaults to None.
            colour (ColorOpts, optional):
                Colour-control bag. See :class:`ColorOpts` for the field list. A
                missing bag is treated as :class:`ColorOpts`\\ () (cleopatra
                defaults). Defaults to None.
            facet (FacetSpec, optional):
                Faceting bag. See :class:`FacetSpec` for the field list. A missing
                bag (or one where both ``col`` and ``row`` are ``None``) routes
                the call to the single-panel static-plot path. Defaults to None.
            coords (tuple or list, optional):
                Explicit curvilinear ``(x, y)`` coordinate spec for the
                pcolormesh path. Accepts two forms:

                - A length-2 sequence of strings — each is looked up as a
                  variable name via ``_read_variable`` on the parent
                  container.
                - A length-2 sequence of numpy arrays — passed straight
                  through to cleopatra. Each array is 1-D (length matches
                  the data x/y axis) or 2-D matching ``(rows, cols)``.

                When ``coords=`` is omitted, pyramids auto-detects
                curvilinear coords via the CF ``coordinates`` attribute on
                the variable, then via the well-known naming conventions
                (WRF ``XLAT`` / ``XLONG``, ROMS ``lat_rho`` / ``lon_rho``,
                NEMO ``nav_lat`` / ``nav_lon``). When nothing matches, the
                renderer falls back to ``extent=self.bbox`` (imshow).
                Defaults to None.
            x_dim (str, optional):
                Name of the dimension to plot on the X axis. Forwarded to
                :meth:`get_variable`. By default the longitude dimension is
                auto-detected from CF coordinate attributes (else the last
                dimension is used). Set this for variables whose lon/lat are
                not the trailing dimensions and carry no CF axis metadata
                (e.g. CAM ``T(time, lat, lev, lon)``). Defaults to None.
            y_dim (str, optional):
                Name of the dimension to plot on the Y axis (the latitude
                dimension by default). Defaults to None.
            kind (str, optional):
                Render kind forwarded to cleopatra's ``ArrayGlyph.plot``.
                One of ``"auto"``, ``"imshow"``, ``"pcolormesh"``,
                ``"contour"``, ``"contourf"``. ``"auto"`` routes to
                ``pcolormesh`` when curvilinear ``coords`` are present,
                else ``imshow``. Defaults to ``"auto"``.
            animate (bool or str, optional):
                When set, render the variable as an animation across a
                band dim instead of a single 2-D slice. ``True`` animates
                along the variable's primary band dim (typically
                ``time``) — only valid when exactly one band dim remains
                after the selectors collapse the others. A string names
                the dim to animate along. ``None`` (default) returns a
                static plot. Mutually exclusive with faceting and with
                any selector that pins the animated dim. Defaults to
                None.
            chunks (Any, optional):
                Chunking spec forwarded to :meth:`read_array` for the
                static-plot path. ``None`` (default) preserves the eager
                read. Any of ``int`` / ``tuple`` / ``dict`` / ``"auto"``
                switches to the dask-backed lazy read and only the
                rendered slice is materialised. Has no effect on the
                ``animate=`` path. Defaults to None.
            basemap (bool or str, optional):
                If truthy, overlay an OpenStreetMap basemap (or a named
                contextily tile provider). Defaults to None.
            exclude_value (Any, optional):
                Pixel value to mask out before plotting. Defaults to None.
            title (str, optional):
                Plot title. Defaults to None.
            ax (Any, optional):
                Existing matplotlib Axes to draw into. Defaults to None.
            figsize (tuple, optional):
                Figure size in inches. Defaults to None.
            **kwargs:
                Additional keyword arguments forwarded to
                :meth:`Analysis.plot <pyramids.dataset.engines.Analysis.plot>`.
                The legacy ``band=`` kwarg is accepted here for backward
                compatibility but emits a :class:`DeprecationWarning`.

        Returns:
            ArrayGlyph: A cleopatra ``ArrayGlyph`` wrapping the rendered figure. Use the glyph's
                matplotlib handles (``glyph.ax`` / ``glyph.fig`` / ``glyph.im``) to decorate the
                plot further. In particular, to overlay a **coastline** (or borders / land / ocean /
                rivers / lakes) on top of the data, call cleopatra's reference helper on the axes —
                passing the CRS the data was plotted in (its own CRS — no reprojection needed) so the
                layer lines up::

                    from cleopatra.reference import add_features
                    var = nc.get_variable("t2m")
                    glyph = var.plot()
                    add_features(glyph.ax, "coastline", crs=var.epsg, zorder=5)

                ``add_features`` fetches Natural Earth data (cached under ``~/.cleopatra``), so it
                needs the ``[viz]`` extra and network access on first use. A relief backdrop is
                available the same way via :func:`cleopatra.reference.add_relief`.

        Raises:
            TypeError: If any of the Sentinel-only kwargs (``rgb``,
                ``surface_reflectance``, ``cutoff``, ``percentile``,
                ``overview``, ``overview_index``) is passed. Each
                rejection message names the xarray-aligned replacement.
            ValueError: If called on a root MDIM container without
                ``variable=``, if ``variable=`` is passed on a subset and
                does not match the pinned variable name, if the resolved
                selectors do not pin to a single 2-D slice, or if
                ``coords=`` is malformed.

        Examples:
            - Plot the first time step of a variable on a container. Tagged
              ``+SKIP`` because rendering requires the optional ``[viz]``
              extra (cleopatra + matplotlib):

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF, Selectors
              >>> arr = np.random.rand(4, 8, 8).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.1, epsg=4326,
              ...     variable_name="t2m",
              ... )
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", selectors=Selectors(isel={"time": 0}),
              ... )

              ```

            - Pick a time slice by label — the ``Selectors.time`` alias
              is equivalent to ``Selectors(sel={"time": value})``:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", selectors=Selectors(time=2),
              ... )

              ```

            - Pin both time and level on a 4-D ``(time, pressure_level,
              lat, lon)`` variable. The selectors collapse both band
              dims to a single 2-D slice — equivalent to
              ``var.sel(time=12).sel(pressure_level=500)``:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="temperature",
              ...     selectors=Selectors(time=12, level=500),
              ... )

              ```

            - Use an explicit ``sel`` dict instead of the convenience
              aliases — keys must match the variable's band-dim names:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", selectors=Selectors(sel={"time": 2}),
              ... )

              ```

            - Use an ``isel`` dict to address slices positionally. Each
              integer is mapped to the corresponding coord value via
              ``_band_dim_values_map``:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", selectors=Selectors(isel={"time": 0}),
              ... )

              ```

            - All six Sentinel-only kwargs are rejected with a hint at
              the xarray-aligned replacement. These doctests run because
              the gate fires before any cleopatra import:

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

            - The legacy ``band=`` kwarg still works as an escape hatch
              but emits a :class:`DeprecationWarning`. Prefer
              ``Selectors(time=...)`` for new code:

              ```python
              >>> import warnings
              >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
              ...     warnings.simplefilter("always")
              ...     cleo = nc.plot(variable="t2m", band=2)
              >>> caught[0].category.__name__  # doctest: +SKIP
              'DeprecationWarning'

              ```

            - Render a WRF-style curvilinear NetCDF on its real lat/lon
              grid. With 2-D ``XLAT`` / ``XLONG`` coord variables on the
              container, pyramids auto-detects them and routes the
              renderer to ``pcolormesh``:

              ```python
              >>> cleo = nc.plot(variable="CANWAT", kind="pcolormesh")  # doctest: +SKIP

              ```

            - Pass an explicit curvilinear coord pair by variable name —
              useful when the variable has no CF ``coordinates``
              attribute and the convention does not match
              WRF / ROMS / NEMO:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="CANWAT", coords=("XLONG", "XLAT"),
              ... )

              ```

            - Pick a non-default render kind. ``"contourf"`` produces
              filled contours from the same data; ``"auto"`` (the
              default) picks ``pcolormesh`` when curvilinear coords are
              present, else falls back to ``imshow``. Discrete contour
              levels live on :class:`ColorOpts`:

              ```python
              >>> from pyramids.netcdf import ColorOpts
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m",
              ...     kind="contourf",
              ...     colour=ColorOpts(levels=10),
              ... )

              ```

            - Render with explicit 2-D coord arrays passed directly via
              ``coords=``. The arrays bypass the CF / convention
              auto-detection step and route the renderer to
              ``pcolormesh``:

              ```python
              >>> import numpy as np
              >>> x2d, y2d = np.meshgrid(
              ...     np.linspace(0, 10, 4), np.linspace(0, 10, 4),
              ... )
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc_curv = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m",
              ... )
              >>> cleo = nc_curv.plot(  # doctest: +SKIP
              ...     variable="t2m", coords=(x2d, y2d),
              ... )

              ```

            - Robust (percentile-based) colour limits — clip to the 2nd / 98th
              percentile of the rendered slice. Colour controls live
              on :class:`ColorOpts`:

              ```python
              >>> from pyramids.netcdf import ColorOpts
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m",
              ...     colour=ColorOpts(cmap="viridis", robust=True),
              ... )

              ```

            - Disable the colorbar — the facade removes it post-render
              because cleopatra always attaches one:

              ```python
              >>> from pyramids.netcdf import ColorOpts
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", colour=ColorOpts(add_colorbar=False),
              ... )

              ```

            - Facet over the time dim. :class:`FacetSpec` lists the
              column dim (and optionally a row dim and a wrap value).
              The return type becomes
              :class:`cleopatra.array_glyph.FacetGrid`:

              ```python
              >>> from pyramids.netcdf import FacetSpec
              >>> grid = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", facet=FacetSpec(col="time"),
              ... )

              ```

            - Facet a 4-D variable across both axes with ``col`` and
              ``row``. ``col_wrap`` is ignored when ``row`` is given:

              ```python
              >>> grid = nc.plot(  # doctest: +SKIP
              ...     variable="temperature",
              ...     facet=FacetSpec(col="time", row="pressure_level"),
              ... )

              ```

            - Wrap a single-axis facet into a grid via ``col_wrap``.
              ``N=4`` panels with ``col_wrap=3`` wrap to a ``2x3``
              layout with one hidden slot:

              ```python
              >>> grid = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", facet=FacetSpec(col="time", col_wrap=3),
              ... )

              ```

            - Faceting on a dim that is also pinned by a selector
              raises :class:`ValueError` before any I/O:

              ```python
              >>> nc.plot(  # doctest: +IGNORE_EXCEPTION_DETAIL
              ...     variable="t2m",
              ...     selectors=Selectors(time=0),
              ...     facet=FacetSpec(col="time"),
              ... )
              Traceback (most recent call last):
                  ...
              ValueError: Cannot facet on 'time'...

              ```

            - Animate along the primary band dim with ``animate=True``.
              The facade resolves the single free band dim (``time``
              here) and streams frames lazily via a per-frame
              ``data_getter`` so the animation never builds a 3-D stack:

              ```python
              >>> cleo = nc.plot(variable="t2m", animate=True)  # doctest: +SKIP

              ```

            - Name the animation dim explicitly. The string must match
              one of the variable's band-dim names. ``animate="time"``
              is equivalent to ``animate=True`` when ``time`` is the
              only free band dim; the explicit form is required on
              variables with more than one free band dim:

              ```python
              >>> cleo = nc.plot(variable="t2m", animate="time")  # doctest: +SKIP

              ```

            - An unknown ``animate=`` dim name is rejected before any
              I/O. The error message lists the available band dims so
              typos are easy to spot:

              ```python
              >>> nc.plot(variable="t2m", animate="bogus")  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              KeyError: "`animate='bogus'` is not a band dim..."

              ```

            - Pinning a dim and then asking to animate over it
              raises :class:`ValueError`:

              ```python
              >>> nc.plot(  # doctest: +IGNORE_EXCEPTION_DETAIL
              ...     variable="t2m",
              ...     selectors=Selectors(time=0),
              ...     animate="time",
              ... )
              Traceback (most recent call last):
                  ...
              ValueError: Cannot animate on 'time'...

              ```

            - Switch the static-plot path to a lazy dask read with
              ``chunks=``. Only the rendered slice is materialised —
              useful when the variable is very large and a full eager
              read would waste memory:

              ```python
              >>> cleo = nc.plot(  # doctest: +SKIP
              ...     variable="t2m", chunks={"x": 5, "y": 5},
              ... )

              ```
        """
        return NetCDFPlot(self).run(
            variable,
            selectors=selectors,
            colour=colour,
            facet=facet,
            coords=coords,
            x_dim=x_dim,
            y_dim=y_dim,
            kind=kind,
            animate=animate,
            chunks=chunks,
            basemap=basemap,
            exclude_value=exclude_value,
            title=title,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    def read_array(
        self,
        variable: str | None = None,
        band: int | None = None,
        window: list[int] | None = None,
        unpack: bool = False,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
        chunks: Any = None,
        lock: Any = None,
        masked: bool = False,
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
                eager path. Mutually exclusive with ``bbox``.
            unpack: If True and the variable has CF `scale_factor`
                and/or `add_offset`, apply the transformation
                `real = raw * scale + offset`. Defaults to False.
                Applied lazily via :mod:`dask.array` arithmetic when
                `chunks` is given — the compute graph stays lazy
                until the caller materializes it.
            bbox (keyword-only): ``(west, south, east, north)`` quadruple
                in the CRS named by ``epsg``. Internally wrapped in a
                one-row :class:`pyramids.feature.FeatureCollection` via
                :meth:`pyramids.feature.FeatureCollection.from_bbox`
                and routed through the same window path. Honored on
                the **eager path only** — same constraint as ``window``.
                Mutually exclusive with ``window``; combining with
                ``chunks`` raises :class:`ValueError` (mirroring
                :class:`pyramids.dataset.engines.IO.read_array`'s
                ``chunks=`` + ``window=`` rule).
            epsg (keyword-only): CRS for ``bbox`` — anything geopandas
                accepts for ``crs=`` (EPSG int, ``"EPSG:4326"``, WKT,
                :class:`pyproj.CRS`). Defaults to the dataset's own
                CRS, so a bbox in the dataset's native CRS needs no
                extra argument.
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
            masked: When `True`, return a :class:`numpy.ma.MaskedArray`
                with the variable's no-data / fill cells masked (eager
                path only; combining with `chunks` raises
                :class:`NotImplementedError`). The mask is built from the
                raw stored values before any `unpack` scaling, matching CF
                `_FillValue` semantics; the scale/offset arithmetic
                preserves the mask. Default is `False`.

        Returns:
            np.ndarray or dask.array.Array: The array data, eager
            (numpy) by default or lazy (dask) when `chunks` is
            supplied. The lazy array computes chunk-by-chunk through
            `md_arr.ReadAsArray(array_start_idx=starts, count=counts)`.

        Raises:
            ValueError: If called on a root MDIM container without a
                `variable` argument, when a subset is called with a
                conflicting `variable` name, when both ``window`` and
                ``bbox`` are supplied, or when both ``chunks`` and
                ``bbox`` are supplied (the lazy path doesn't yet
                honour bbox windowing — matching
                :class:`pyramids.dataset.engines.IO.read_array`'s
                ``chunks=`` + ``window=`` rule).
            ImportError: If `chunks` is given but `dask` is not
                installed. Install the `[lazy]` extra.
            NotImplementedError: If `masked=True` is combined with
                `chunks` (lazy masked reads are not supported yet).

        Examples:
            - Eager bbox read on a root container — the container
              auto-routes to the named variable. The noah fixture's
              geotransform is ``cell_size=0.5°``, ``origin=(0, 90)``,
              512×512 cells — so its coordinate range is
              ``x ∈ [0, 256)`` and ``y ∈ (-166, 90]``. The bbox
              below sits well inside that range:
                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file(
                ...     "tests/data/netcdf/noah-precipitation-1979.nc"
                ... )
                >>> arr = nc.read_array(
                ...     variable="Band1",
                ...     bbox=(10.0, -50.0, 50.0, -20.0),
                ... )
                >>> arr.ndim in (2, 3)
                True

                ```

        See Also:
            - :meth:`pyramids.dataset.Dataset.read_array`: the same
              ``bbox=`` / ``epsg=`` surface for plain rasters.
            - :meth:`crop`: clip the whole dataset by bbox.
        """
        if bbox is not None:
            if window is not None:
                raise ValueError(
                    "read_array accepts either `window` or `bbox`, not both"
                )
            if chunks is not None:
                raise ValueError(
                    "read_array(chunks=..., bbox=...) is not supported; "
                    "read lazily and slice the resulting dask array instead."
                )
            crs = epsg if epsg is not None else self.epsg
            if crs is None:
                raise ValueError(
                    "read_array(bbox=…) requires an explicit `epsg=` when "
                    "the NetCDF itself has no CRS (self.epsg is None) — a "
                    "bbox without a CRS is ambiguous"
                )
            # Build the FC once at the override boundary and forward as
            # `window=fc, bbox=None` so the guards never re-fire on the
            # recursive container→subset call or on super().read_array.
            # Mirrors NetCDF.crop's "build mask once at the top" pattern.
            window = FeatureCollection.from_bbox(bbox, epsg=crs)
            bbox = None
            epsg = None
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
                bbox=bbox,
                epsg=epsg,
                chunks=chunks,
                lock=lock,
                masked=masked,
            )
        if variable is not None and variable != self._source_var_name:
            raise ValueError(
                f"This NetCDF instance is already pinned to variable "
                f"{self._source_var_name!r}; cannot re-read as "
                f"{variable!r}. Call read_array on the parent container "
                "instead."
            )
        if chunks is None:
            result = super().read_array(
                band=band,
                window=window,
                bbox=bbox,
                epsg=epsg,
                masked=masked,
            )
            if unpack:
                result = apply_unpack(
                    result,
                    getattr(self, "_scale", None),
                    getattr(self, "_offset", None),
                )
        else:
            if masked:
                raise NotImplementedError(
                    "read_array(masked=True) is not supported together with "
                    "chunks=; read eagerly, or mask the dask array yourself."
                )
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
                result = apply_unpack(
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

        The canonical multi-band-dim fields (`_band_dim_names`,
        `_band_dim_values_map`, `_band_dim_sizes`) are propagated, and the
        legacy single-band-dim view (`_band_dim_name`, `_band_dim_values`) is
        re-derived from them via :meth:`_derive_primary_band_view` against the
        wrapped result's live band count. That helper is the single place the
        staleness guard lives: it nullifies the primary coordinate values when
        they are provably stale for the new band count (e.g. after a
        band-shrinking operation outside `sel()`).

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
            wrapped = Variable(
                result._raster,
                access=result._access,
                open_as_multi_dimensional=False,
            )
        wrapped._is_md_array = self._is_md_array
        wrapped._is_subset = self._is_subset
        wrapped._band_dim_names = self._band_dim_names
        wrapped._band_dim_sizes = self._band_dim_sizes
        wrapped._band_dim_values_map = dict(self._band_dim_values_map)
        # Re-derive the legacy primary-dim view from the canonical fields against the
        # wrapped result's live band count: a band-shrinking spatial op may have
        # diverged the band count from the cached coords, so the staleness guard lives
        # once in `_derive_primary_band_view` rather than being repeated here.
        wrapped._band_dim_name, wrapped._band_dim_values = self._derive_primary_band_view(
            wrapped._band_dim_names,
            wrapped._band_dim_values_map,
            wrapped._band_dim_sizes,
            wrapped._band_count,
        )
        wrapped._variable_attrs = self._variable_attrs
        wrapped._scale = self._scale
        wrapped._offset = self._offset
        wrapped._parent_nc = self._parent_nc
        wrapped._source_var_name = self._source_var_name
        wrapped._gdal_md_arr_ref = None
        wrapped._gdal_rg_ref = None
        return wrapped

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
        chunks: Any = None,
    ) -> NetCDF:
        """Crop the dataset using a polygon mask, a raster mask, or a bbox tuple.

        On a **root MDIM container** this crops every variable and
        returns a new in-memory NetCDF container with the cropped
        results. On a **variable subset** it delegates to the parent
        :meth:`pyramids.dataset.Dataset.crop` and re-wraps the result
        as :class:`NetCDF` to preserve variable metadata
        (``_band_dim_name``, ``_band_dim_values``, :meth:`sel`).

        Args:
            mask: GeoDataFrame with polygon geometry, or a Dataset
                to use as a spatial mask. Mutually exclusive with
                ``bbox``; exactly one of the two must be supplied.
            touch: If True, include cells that touch the mask
                boundary. Defaults to True.
            bbox (keyword-only): ``(west, south, east, north)``
                quadruple in the CRS named by ``epsg``. Internally
                wrapped in a one-row :class:`FeatureCollection` via
                :meth:`FeatureCollection.from_bbox` and routed through
                the same polygon path. The FC is built **once** so a
                root-container crop does not rebuild it for every
                variable. Mutually exclusive with ``mask``.
            epsg (keyword-only): CRS for ``bbox`` — anything geopandas
                accepts for ``crs=`` (EPSG int, ``"EPSG:4326"``, WKT,
                :class:`pyproj.CRS`). Defaults to the dataset's own
                CRS, so a bbox in the dataset's native CRS needs no
                extra argument; pass it explicitly for a bbox in a
                different CRS (the standard reprojection path handles
                it).
            chunks (keyword-only): Lazy-read chunking for the
                **curvilinear** crop path only — forwarded to
                :meth:`read_array` so the cropped window is read through
                the dask-backed lazy path (``"auto"`` or a ``{"rows":
                ..., "cols": ...}`` dict). The curvilinear crop reads only
                the polygon's bounding window regardless; ``chunks`` makes
                that windowed read lazy/chunked. It is a per-variable,
                curvilinear-only option: it raises ``ValueError`` if given
                for a rectilinear (affine-warp) crop (which is eager), or
                on a **root container** (call crop on a single variable via
                :meth:`get_variable` instead).

        Returns:
            NetCDF: Cropped container or variable subset.

        Raises:
            ValueError: Both ``mask`` and ``bbox`` were supplied, or
                ``chunks`` was given on a root container or a rectilinear
                crop.
            TypeError: Neither ``mask`` nor ``bbox`` was supplied.

        Examples:
            - Crop every variable of a root NetCDF container by a
              bbox in the dataset's own CRS (`epsg` is inferred). The
              noah fixture's geotransform is ``cell_size=0.5°``,
              ``origin=(0, 90)``, 512×512 cells — so its coordinate
              range is ``x ∈ [0, 256)`` and ``y ∈ (-166, 90]``. The
              bbox below sits well inside that range:
                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file(
                ...     "tests/data/netcdf/noah-precipitation-1979.nc"
                ... )
                >>> cropped = nc.crop(bbox=(10.0, -50.0, 50.0, -20.0))
                >>> sorted(cropped.variables) == sorted(nc.variables)
                True

                ```
            - Mutual-exclusion guard:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file(
                ...     "tests/data/netcdf/noah-precipitation-1979.nc"
                ... )
                >>> fc = FeatureCollection.from_bbox(
                ...     (10.0, -50.0, 50.0, -20.0), epsg=nc.epsg,
                ... )
                >>> try:
                ...     nc.crop(mask=fc, bbox=(10.0, -50.0, 50.0, -20.0))
                ... except ValueError as exc:
                ...     print("not both" in str(exc))
                True

                ```

        See Also:
            - :meth:`pyramids.dataset.Dataset.crop`: same ``bbox=`` /
              ``epsg=`` surface for plain rasters.
            - :meth:`pyramids.feature.FeatureCollection.from_bbox`: the
              shared primitive that builds the one-row FC.
        """
        mask = self._resolve_crop_mask(mask, bbox, epsg)
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            # A container crops every variable; `chunks` is a curvilinear-only, per-variable knob
            # (a container may mix curvilinear and rectilinear variables), so the per-variable fan-out
            # cannot honour it. Reject it explicitly rather than silently reading eagerly.
            if chunks is not None:
                raise ValueError(
                    "crop(chunks=…) is not supported on a root container; it is a "
                    "curvilinear-only, per-variable option — call crop on a single variable "
                    "via get_variable(name) instead."
                )
            result = self._apply_to_all_variables("crop", {"mask": mask, "touch": touch})
        else:
            result = self._crop_one(mask, touch=touch, chunks=chunks)
        return result

    def _resolve_crop_mask(
        self,
        mask: Any,
        bbox: tuple[float, float, float, float] | list[float] | None,
        epsg: Any,
    ) -> Any:
        """Resolve the crop selector to a single polygon mask.

        Converts a ``bbox`` (in ``epsg``, defaulting to the dataset CRS) into a one-row
        :class:`FeatureCollection`, enforces that exactly one of ``mask`` / ``bbox`` is supplied,
        and returns the mask to crop with.

        Args:
            mask: A polygon mask or ``Dataset`` mask, or ``None`` when ``bbox`` is used.
            bbox: ``(west, south, east, north)`` quadruple, or ``None`` when ``mask`` is used.
            epsg: CRS for ``bbox``; defaults to ``self.epsg`` when ``None``.

        Returns:
            The resolved mask (a ``FeatureCollection`` when built from ``bbox``).

        Raises:
            ValueError: Both ``mask`` and ``bbox`` were supplied, or a ``bbox`` was given with no CRS.
            TypeError: Neither ``mask`` nor ``bbox`` was supplied.
        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            crs = epsg if epsg is not None else self.epsg
            if crs is None:
                raise ValueError(
                    "crop(bbox=…) requires an explicit `epsg=` when the "
                    "NetCDF itself has no CRS (self.epsg is None) — a "
                    "bbox without a CRS is ambiguous"
                )
            mask = FeatureCollection.from_bbox(bbox, epsg=crs)
        if mask is None:
            raise TypeError(
                "crop requires a `mask` (GeoDataFrame / FeatureCollection / "
                "Dataset) or a `bbox` (west, south, east, north) tuple"
            )
        return mask

    def _crop_one(self, mask: Any, touch: bool = True, chunks: Any = None) -> "NetCDF":
        """Crop a single variable/subset, routing curvilinear grids to the 2-D coordinate masker.

        Curvilinear grids (2-D lon/lat coords, no single affine geotransform) can't be clipped by the
        affine cutline warp, so they mask on their 2-D coordinates; rectilinear grids use the affine
        crop and re-wrap to preserve NetCDF metadata. ``chunks`` is valid only on the curvilinear path.

        Args:
            mask: The resolved polygon/raster mask to crop with.
            touch: If True, include cells touching the mask boundary. Defaults to True.
            chunks: Lazy-read chunking, valid only on the curvilinear path; see :meth:`crop`.

        Returns:
            NetCDF: The cropped variable subset.

        Raises:
            ValueError: ``chunks`` was given for a rectilinear (affine) crop, which is eager.
        """
        curv = NetCDFPlot(self)._resolve_curvilinear_coords(self, coords=None)
        is_curvilinear = (
            curv is not None
            and np.asarray(curv[0]).ndim == 2
            and np.asarray(curv[1]).ndim == 2
        )
        if is_curvilinear:
            result = self._crop_curvilinear(mask, curv, touch=touch, chunks=chunks)
        else:
            if chunks is not None:
                raise ValueError(
                    "chunks= is only supported for curvilinear crop; the affine "
                    "(rectilinear) crop path is eager."
                )
            result = self._preserve_netcdf_metadata(super().crop(mask=mask, touch=touch))
        return result

    def _crop_curvilinear(
        self,
        mask: FeatureCollection,
        coords2d: tuple[np.ndarray, np.ndarray],
        touch: bool = True,
        chunks: Any = None,
    ) -> "NetCDF":
        """Crop a curvilinear (2-D coordinate) variable by masking on its lon/lat arrays.

        Curvilinear grids have 2-D ``lon(y, x)`` / ``lat(y, x)`` coordinates and no single affine
        geotransform, so the cutline warp used by :meth:`pyramids.dataset.Dataset.crop` cannot clip
        them. Instead, test each cell's ``(lon, lat)`` against the polygon, set the cells whose
        centre falls outside it to no-data, and trim to the bounding ``(row, col)`` index window of
        the inside cells. The result keeps its windowed 2-D coordinate arrays (stored as
        ``_curvilinear_coords``) so it stays curvilinear and plots on its real geometry.

        Args:
            mask (FeatureCollection):
                Polygon mask (a ``FeatureCollection`` / ``GeoDataFrame``). Its CRS is reconciled
                with the variable's CRS before the point-in-polygon test.
            coords2d (tuple[np.ndarray, np.ndarray]):
                The variable's 2-D ``(lon, lat)`` coordinate arrays, shaped like its spatial dims.
            touch (bool):
                Accepted for signature parity with the affine crop. The curvilinear path tests cell
                centres, so this currently has no effect. Defaults to True.

        Returns:
            NetCDF: The masked + windowed variable subset, carrying its windowed 2-D coordinates.

        Raises:
            ValueError: If the polygon does not overlap the grid (no cell centre inside it).
        """
        lon2d = np.asarray(coords2d[0], dtype=float)
        lat2d = np.asarray(coords2d[1], dtype=float)

        gdf = mask
        gdf_crs = getattr(gdf, "crs", None)
        if gdf_crs is not None and self.epsg:
            src_epsg = gdf_crs.to_epsg()
            if src_epsg is not None and src_epsg != self.epsg:
                gdf = gdf.to_crs(epsg=self.epsg)
        geometry = gdf.geometry.union_all()

        inside = contains_xy(geometry, lon2d, lat2d)
        if not bool(np.any(inside)):
            raise ValueError(
                "crop polygon does not overlap the curvilinear grid "
                "(no cell centre falls inside it)."
            )

        rows = np.nonzero(np.any(inside, axis=1))[0]
        cols = np.nonzero(np.any(inside, axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1

        nd = self.no_data_value
        nd = nd[0] if isinstance(nd, (list, tuple)) else nd
        if nd is None:
            nd = DEFAULT_NO_DATA_VALUE

        # Read only the bounding window, not the whole variable. The polygon mask and window were
        # derived from the (spatial-footprint) coordinate arrays alone — no data was materialised yet.
        inside_win = inside[r0:r1, c0:c1]
        if chunks is not None:
            # Lazy/chunked read: only the chunks overlapping the window are materialised.
            lazy = self.read_array(chunks=chunks)
            if lazy.ndim > 2:
                # read_array(chunks=) keeps the native (d0, ..., rows, cols) shape; flatten the
                # non-spatial dims to the (bands, rows, cols) layout the eager path uses.
                lazy = lazy.reshape(-1, *lazy.shape[-2:])
            data_win = np.array(lazy[..., r0:r1, c0:c1].compute(), copy=True)
        else:
            # Eager windowed read: GDAL reads just the (c0, r0)-(c1, r1) block.
            data_win = np.array(
                self.read_array(window=[c0, r0, c1 - c0, r1 - r0]), copy=True
            )
        data_win[..., ~inside_win] = nd
        lon_win = lon2d[r0:r1, c0:c1]
        lat_win = lat2d[r0:r1, c0:c1]

        var_name = getattr(self, "_source_var_name", None) or "data"
        container = NetCDF.create_from_array(
            data_win,
            geo=self._bbox_geotransform(lon_win, lat_win),
            epsg=self.epsg or 4326,
            no_data_value=nd,
            variable_name=var_name,
        )
        # create_from_array returns a root container; hand back the variable subset, carrying the
        # windowed 2-D coordinates so the result stays curvilinear (plots on its real geometry).
        result = container.get_variable(var_name)
        result._curvilinear_coords = (lon_win, lat_win)
        return result

    @staticmethod
    def _bbox_geotransform(lon: np.ndarray, lat: np.ndarray) -> tuple:
        """Approximate north-up affine geotransform spanning the 2-D coords' bounding box.

        A curvilinear grid has no true affine transform; this gives a cropped curvilinear result a
        sensible ``total_bounds`` (its lon/lat envelope). The authoritative per-cell mapping is the
        2-D ``_curvilinear_coords`` carried alongside. The 2-D coordinates are cell *centres*, so the
        cell size is the spacing between adjacent centres (``cols - 1`` gaps across ``cols`` centres),
        and the north-west origin sits half a cell beyond the outermost centre — so the envelope spans
        the full ``cols x rows`` cells, not just centre-to-centre. Pixel height is negative (north-up).

        Args:
            lon (np.ndarray): 2-D longitude array of the (windowed) grid; finite values define the
                west/east extent.
            lat (np.ndarray): 2-D latitude array of the same shape; finite values define the
                south/north extent. Its shape sets the row/column counts.

        Returns:
            tuple: A GDAL geotransform ``(x_origin, x_cell, 0.0, y_origin, 0.0, -y_cell)`` where the
                origin is the north-west envelope corner (half a cell outside the corner centre).

        Examples:
            - A 2x2 grid of centres at lon 10/20 and lat 30/40 has 10deg cells; the envelope origin
              sits half a cell (5deg) outside the corner centre:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> lon = np.array([[10.0, 20.0], [10.0, 20.0]])
                >>> lat = np.array([[40.0, 40.0], [30.0, 30.0]])
                >>> NetCDF._bbox_geotransform(lon, lat)
                (5.0, 10.0, 0.0, 45.0, 0.0, -10.0)

                ```
        """
        rows, cols = lat.shape
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        # cell size = spacing between adjacent cell centres (cols-1 gaps); a single row/column has no
        # spacing to measure, so fall back to 0 (a degenerate 1-cell envelope, as before).
        x_cell = (lon_max - lon_min) / (cols - 1) if cols > 1 else 0.0
        y_cell = (lat_max - lat_min) / (rows - 1) if rows > 1 else 0.0
        return (lon_min - x_cell / 2, x_cell, 0.0, lat_max + y_cell / 2, 0.0, -y_cell)

    def _variable_is_spatial(self, rg: Any, var_name: str) -> bool:
        """True when ``var_name`` is a gridded variable (can be cropped / reprojected).

        A variable is spatial when it has at least two dimensions and a recognised
        ``(y, x)`` pair among **its own** dimensions — detected via the CF-attribute
        / well-known-name machinery (:meth:`_cf_spatial_axes` /
        :meth:`_named_spatial_axes`), per variable, so a variable on a secondary
        grid is judged on its own axes rather than one container-wide pair. The
        check reads the MDArray's dimensions directly (not via ``get_variable``,
        which can't build a classic raster for a non-spatial variable), so an
        auxiliary variable is identified before any spatial op runs.

        Args:
            rg: The root :class:`osgeo.gdal.Group` of the open store, used to
                open the named array and resolve its dimensions.
            var_name: Name of the variable to classify.

        Returns:
            bool: ``True`` when the variable has at least two dimensions with a
            recognised ``(y, x)`` pair among them; ``False`` for a 1-D / scalar
            auxiliary variable, a 2-D variable with no spatial axes, or a name
            that cannot be opened as an MDArray.

        Examples:
            - A gridded ``t2m(valid_time, lat, lon)`` variable is spatial, so
              ``crop`` / ``to_crs`` will operate on it (requires an open store):
                ```python
                >>> rg = nc._raster.GetRootGroup()  # doctest: +SKIP
                >>> nc._variable_is_spatial(rg, "t2m")  # doctest: +SKIP
                True

                ```
            - A 1-D ``number(valid_time)`` auxiliary variable is not spatial, so
              it is carried through unchanged instead:
                ```python
                >>> rg = nc._raster.GetRootGroup()  # doctest: +SKIP
                >>> nc._variable_is_spatial(rg, "number")  # doctest: +SKIP
                False

                ```
        """
        md = open_mdarray(rg, var_name)
        if md is None:
            return False
        var_dims = [d.GetName() for d in md.GetDimensions()]
        if len(var_dims) < 2:
            return False
        return (
            self._cf_spatial_axes(rg, var_dims) is not None
            or self._named_spatial_axes(var_dims) is not None
        )

    def _spatial_variable_names(self, rg: Any = None) -> list[str]:
        """Names of the container's gridded variables (have a recognised (y, x) pair).

        Used by every spatial fan-out (``crop`` / ``to_crs`` / ``resample`` /
        ``reduce``) so they act only on griddable variables; the remaining
        non-spatial auxiliary variables are carried through by
        :meth:`_carry_aux_variables`.

        Args:
            rg: An already-resolved root :class:`osgeo.gdal.Group`. When ``None``
                (the default) the root group is fetched from the wrapped dataset;
                callers that already hold it pass it in to avoid a redundant
                ``GetRootGroup()``.

        Returns:
            list[str]: Names of the gridded variables, in declaration order.
            Empty when the store has no root group (e.g. a closed or
            single-variable raster handle) or no variable carries a ``(y, x)``
            pair.

        Examples:
            - An ERA5-shaped container reports only its gridded variables,
              leaving the 1-D ``number`` auxiliary out (requires an open store):
                ```python
                >>> nc._spatial_variable_names()  # doctest: +SKIP
                ['t2m']

                ```
            - A single-variable view (no root group) yields an empty list:
                ```python
                >>> nc.get_variable("t2m")._spatial_variable_names()  # doctest: +SKIP
                []

                ```
        """
        if rg is None:
            rg = self._raster.GetRootGroup() if self._raster is not None else None
        if rg is None:
            return []
        return [n for n in self.variable_names if self._variable_is_spatial(rg, n)]

    def _variable_dim_names(self, rg: Any, var_name: str) -> list[str]:
        """Return a variable's dimension names, or ``[]`` if it can't be opened.

        Args:
            rg: The store's root :class:`osgeo.gdal.Group`.
            var_name: Name of the variable (MDArray) to inspect.

        Returns:
            list[str]: The variable's dimension names in storage order, or an
            empty list when the variable is missing or unreadable.
        """
        md = open_mdarray(rg, var_name)
        if md is None:
            return []
        return [d.GetName() for d in md.GetDimensions()]

    def _carry_aux_variables(
        self, result: NetCDF, aux_vars: list[str], operation: str
    ) -> None:
        """Copy non-spatial auxiliary variables into ``result`` unchanged.

        They can't be cropped/reprojected/reduced through the raster path but must
        survive the op, so each is copied verbatim (dims / values / attrs) via
        :meth:`add_variable`. Best-effort: a copy failure for one variable warns
        rather than failing the whole operation.

        Args:
            result: The container built from the spatial variables.
            aux_vars: Names of the non-spatial variables to carry through.
            operation: Operation name, for the warning message.

        Returns:
            None: ``result`` is mutated in place — each carried variable is
            added to it. A copy failure for one variable emits a
            :class:`UserWarning` naming that variable and continues.

        Examples:
            - Carry an ERA5 cube's 1-D ``number`` auxiliary into a freshly
              cropped result so it survives the op (requires an open store):
                ```python
                >>> cropped = nc._apply_to_all_variables("crop", {"mask": mask})  # doctest: +SKIP
                >>> nc._carry_aux_variables(cropped, ["number"], "crop")  # doctest: +SKIP
                >>> sorted(cropped.variable_names)  # doctest: +SKIP
                ['number', 't2m']

                ```
        """
        dropped: list[tuple[str, Exception]] = []
        for var_name in aux_vars:
            try:
                result.add_variable(self, var_name)
            except (RuntimeError, ValueError) as exc:
                dropped.append((var_name, exc))
        if dropped:
            # One aggregated warning naming every dropped variable, so a silent
            # data loss across crop/to_crs/resample/reduce is hard to miss in a
            # pipeline rather than scattered across per-variable warnings.
            names = ", ".join(repr(name) for name, _ in dropped)
            reasons = "; ".join(f"{name!r}: {exc}" for name, exc in dropped)
            warnings.warn(
                f"{operation}() could not carry {len(dropped)} non-spatial "
                f"variable(s) ({names}) into the result: {reasons}",
                stacklevel=3,
            )

    def _apply_to_all_variables(self, operation, op_kwargs):
        """Apply a spatial operation to every gridded variable in the container.

        Only variables carrying both spatial axes are cropped / reprojected.
        Non-spatial auxiliary variables (e.g. ERA5's ``expver`` / ``number``,
        which have no ``y`` / ``x`` axes) can't go through the raster op, so they
        are **carried through unchanged** into the result rather than crashing the
        fan-out.

        Args:
            operation: Name of the Dataset method to call (e.g. "crop").
            op_kwargs: Keyword arguments to pass to the method.

        Returns:
            NetCDF: New container with the operation applied to every gridded
            variable and the non-spatial auxiliary variables carried through.

        Raises:
            ValueError: If the container has no data variables, or none of them
                are spatial (have both ``y`` / ``x`` axes).
        """
        names = self.variable_names
        if not names:
            raise ValueError(
                "Cannot apply operation to an empty container (no data variables)."
            )

        # Resolve the root group once and thread it through the spatial-variable scan
        # and the aux-variable dimension probe below, instead of re-resolving per call.
        rg = self._raster.GetRootGroup() if self._raster is not None else None
        spatial_vars = self._spatial_variable_names(rg)
        aux_vars = [n for n in names if n not in spatial_vars]
        if not spatial_vars:
            raise ValueError(
                f"{operation}() needs at least one spatial (y, x) variable; none of "
                f"{names} have both spatial axes."
            )

        # A variable with >= 2 *unrecognised* axes is likely a grid whose axes
        # were not recognised (no CF axis attributes / no known x/y names) and is
        # carried through untransformed — warn. Axes that are clearly non-spatial
        # (time / vertical / ensemble / bounds) don't count, so a legitimately
        # non-spatial N-D aux variable (e.g. (time, level)) does not trip the warning.
        demoted = []
        for n in aux_vars:
            unknown_axes = [
                d
                for d in self._variable_dim_names(rg, n)
                if d.lower() not in _NONSPATIAL_AXIS_NAMES
            ]
            if len(unknown_axes) >= 2:
                demoted.append(n)
        if demoted:
            warnings.warn(
                f"{operation}() is carrying {len(demoted)} multi-dimensional "
                f"variable(s) {demoted} through unchanged because their axes were "
                f"not recognised as spatial (no CF axis attributes or known x/y "
                f"names); they will NOT be cropped/reprojected. Add CF axis "
                f"metadata (standard_name / axis) or rename the axes to y/x.",
                stacklevel=3,
            )

        result = None
        for var_name in spatial_vars:
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
                NetCDF._copy_band_dim_metadata(ds, var)
                result.set_variable(var_name, ds)

        self._carry_aux_variables(result, aux_vars, operation)
        return result

    def reduce(
        self,
        dim: str,
        how: str = "mean",
        *,
        groupby: list | tuple | str | None = None,
        skipna: bool = True,
    ) -> NetCDF:
        """Reduce every variable along a named dimension and return a new NetCDF.

        Collapses or coarsens one non-spatial dimension (`time`,
        `pressure_level`, `depth`, an ensemble member, …) of every variable
        that has it, leaving variables without `dim` and all other dimensions,
        coordinates, CRS, and the grid untouched. The result is a new
        :class:`NetCDF` container — no xarray involved. Only gridded variables
        are reduced; non-spatial auxiliary variables (no ``y`` / ``x`` axes,
        e.g. ERA5's ``number``) are carried through unchanged rather than
        crashing the fan-out (#513) — except an auxiliary variable that itself
        spans `dim`, which is dropped with a warning (carrying it verbatim would
        leave an inconsistent `dim` length against the collapsed variables).

        Args:
            dim: Name of the non-spatial dimension to reduce. Must be one of a
                variable's band dimensions (as exposed by ``sel``); spatial
                ``lat`` / ``lon`` dimensions are not reducible here.
            how: Reduction operation — one of ``"mean"``, ``"sum"``, ``"min"``,
                ``"max"``, ``"std"``, ``"var"``.
            groupby: Controls collapse vs. windowed reduction:

                - ``None`` (default): collapse `dim` entirely (it is removed
                  from the output).
                - a sequence of per-index labels (length = the size of `dim`):
                  reduce each group of equal labels; `dim` is coarsened to one
                  slice per distinct label, in first-appearance order.
                - a pandas offset alias (e.g. ``"1MS"``, ``"1D"``, ``"YS"``):
                  group `dim` by calendar window. Only valid when `dim` carries
                  a decodable CF time coordinate.
            skipna: When ``True`` (default), mask each variable's NoData value to
                ``NaN`` and reduce with the ``nan``-aware operation, then refill
                ``NaN`` results with NoData. The output is float64. When
                ``False``, reduce the raw values with the plain operation.

        Returns:
            NetCDF: A new container with `dim` removed (``groupby=None``) or
            coarsened (windowed). When the windowed dimension keeps a numeric
            coordinate, each output slice is labelled with the first source
            coordinate value of its window.

        Raises:
            ValueError: When `how` is unknown, the container has no data
                variables, `dim` is not a non-spatial dimension of any variable,
                a frequency `groupby` is given but `dim` has no decodable time
                coordinate, or the grouping does not cover `dim` exactly.

        Examples:
            - Monthly mean of an ERA5-style ``(time, lat, lon)`` file:
                ```python
                >>> from pyramids.netcdf import NetCDF  # doctest: +SKIP
                >>> nc = NetCDF.read_file("era5_t2m_hourly.nc")  # doctest: +SKIP
                >>> monthly = nc.reduce("time", "mean", groupby="1MS")  # doctest: +SKIP
                >>> monthly.get_variable("t2m").band_count  # doctest: +SKIP
                12

                ```
            - Collapse a pressure-level axis to its column mean:
                ```python
                >>> column = nc.reduce("pressure_level", "mean")  # doctest: +SKIP
                >>> "pressure_level" in column.get_variable("t").dimensions  # doctest: +SKIP
                False

                ```
        """
        if how not in _REDUCERS:
            raise ValueError(f"how must be one of {sorted(_REDUCERS)}; got {how!r}")
        names = self.variable_names
        if not names:
            raise ValueError("Cannot reduce an empty container (no data variables).")

        group_positions = self._resolve_group_positions(dim, groupby)

        # Reduce only the gridded variables; non-spatial auxiliaries (no y/x axes)
        # can't go through the raster reduce path, so they are carried through
        # unchanged below — the same split crop / to_crs use (#513). Resolve the root
        # group once and reuse it for the spanning-aux probe further down.
        rg = self._raster.GetRootGroup() if self._raster is not None else None
        spatial_vars = self._spatial_variable_names(rg)
        aux_vars = [n for n in names if n not in spatial_vars]

        result = None
        found = False
        for var_name in spatial_vars:
            var = self.get_variable(var_name)
            arr = self._materialize_variable_array(var)
            band_names = list(var._band_dim_names)
            values_map = dict(var._band_dim_values_map)
            ndv = self._scalar_no_data_value(var.no_data_value)

            if dim in band_names:
                found = True
                axis = band_names.index(dim)
                arr, band_names, values_map = self._reduce_variable_array(
                    arr,
                    axis,
                    dim,
                    band_names,
                    values_map,
                    how,
                    skipna,
                    ndv,
                    groupby,
                    group_positions,
                )

            result = self._stack_reduced_variable(
                result,
                var_name,
                arr,
                var.geotransform,
                var.epsg,
                ndv,
                band_names,
                values_map,
            )

        if not found:
            raise ValueError(
                f"Dimension {dim!r} is not a non-spatial dimension of any "
                f"variable in this container."
            )
        # Auxiliary variables that span the reduced dimension cannot be carried
        # verbatim — they would keep the full-length axis while the gridded
        # variables collapse it, leaving an inconsistent dimension length. Drop
        # those with a warning; carry the rest unchanged.
        carry_aux: list[str] = []
        spanning_aux: list[str] = []
        for name in aux_vars:
            var_dims = self._variable_dim_names(rg, name)
            (spanning_aux if dim in var_dims else carry_aux).append(name)
        if spanning_aux:
            warnings.warn(
                f"reduce() dropped auxiliary variable(s) {spanning_aux} that span "
                f"the reduced dimension {dim!r}; carrying them unchanged would "
                f"leave an inconsistent {dim!r} length in the result.",
                stacklevel=2,
            )
        self._carry_aux_variables(result, carry_aux, "reduce")
        return result

    @staticmethod
    def _scalar_no_data_value(no_data_value: Any) -> Any:
        """Return a single NoData value from a per-band list/tuple or scalar."""
        return scalar_no_data(no_data_value)

    @staticmethod
    def _materialize_variable_array(var: NetCDF) -> np.ndarray:
        """Read a variable as `(*band_dim_sizes, rows, cols)` (or `(rows, cols)`).

        Undoes ``read_array``'s singleton-band squeeze and GDAL's row-major
        flatten of multi-dim band axes, mirroring `_apply_to_all_variables`.
        """
        arr = var.read_array()
        if var._band_dim_names:
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            if len(var._band_dim_names) > 1 and arr.ndim == 3 and var._band_dim_sizes:
                arr = arr.reshape(*var._band_dim_sizes, arr.shape[-2], arr.shape[-1])
        return arr

    def _resolve_group_positions(
        self, dim: str, groupby: list | tuple | str | None
    ) -> list[np.ndarray] | None:
        """Resolve `groupby` into ordered lists of source index positions.

        Returns ``None`` for the collapse case (``groupby is None``).
        """
        positions: list[np.ndarray] | None = None
        if isinstance(groupby, str):
            # Full-resolution timestamps: the default "%Y-%m-%d" truncates to
            # whole days, which would collapse every sub-daily frequency
            # ("1H"/"3H"/"6H") into a single per-day bucket.
            times = self.get_time_variable(
                var_name=dim, time_format="%Y-%m-%d %H:%M:%S"
            )
            if times is None:
                raise ValueError(
                    f"Cannot group dimension {dim!r} by frequency {groupby!r}: "
                    f"no decodable time coordinate found."
                )
            index = pd.DatetimeIndex(pd.to_datetime(times))
            series = pd.Series(np.arange(len(index)), index=index)
            positions = [
                np.sort(members.to_numpy())
                for _, members in series.groupby(pd.Grouper(freq=groupby))
                if len(members) > 0
            ]
        elif groupby is not None:
            labels = list(groupby)
            order: list[Any] = []
            members: dict[Any, list[int]] = {}
            for i, label in enumerate(labels):
                if label not in members:
                    members[label] = []
                    order.append(label)
                members[label].append(i)
            positions = [np.array(members[label]) for label in order]
        return positions

    def _reduce_variable_array(
        self,
        arr,
        axis,
        dim,
        band_names,
        values_map,
        how,
        skipna,
        ndv,
        groupby,
        group_positions,
    ):
        """Reduce one variable's array along `axis`; return new array + dims."""
        if group_positions is None:
            new_arr = self._reduce_axis(arr, axis, how, skipna, ndv)
            new_band_names = [name for name in band_names if name != dim]
            new_values_map = {name: values_map.get(name) for name in new_band_names}
        else:
            covered = sum(len(positions) for positions in group_positions)
            if covered != arr.shape[axis]:
                raise ValueError(
                    f"groupby covers {covered} positions but dimension {dim!r} "
                    f"has size {arr.shape[axis]}."
                )
            slices = [
                self._reduce_axis(
                    np.take(arr, positions, axis=axis), axis, how, skipna, ndv
                )
                for positions in group_positions
            ]
            new_arr = np.stack(slices, axis=axis)
            coord = values_map.get(dim)
            new_band_names = list(band_names)
            new_values_map = dict(values_map)
            new_values_map[dim] = (
                [coord[int(positions[0])] for positions in group_positions]
                if coord is not None
                else None
            )
        return new_arr, new_band_names, new_values_map

    @staticmethod
    def _reduce_axis(arr, axis, how, skipna, ndv):
        """Apply one reduction over `axis`, masking NoData when `skipna`."""
        nan_func, plain_func = _REDUCERS[how]
        if skipna:
            data = arr.astype("float64")
            if ndv is not None:
                data = np.where(data == ndv, np.nan, data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = nan_func(data, axis=axis)
            # nansum/nanstd/nanvar return 0 (not NaN) for an all-NoData slice,
            # so detect fully-masked positions explicitly and restore NoData for
            # every reducer rather than leaking a spurious 0.
            all_masked = np.all(np.isnan(data), axis=axis)
            fill = ndv if ndv is not None else np.nan
            out = np.where(np.isnan(out) | all_masked, fill, out)
            result = out
        else:
            result = plain_func(arr, axis=axis)
        return result

    def _stack_reduced_variable(
        self, result, var_name, arr, geo, epsg, ndv, band_names, values_map
    ):
        """Add a reduced variable into the result container, building it lazily."""
        extra = (
            [(name, values_map.get(name)) for name in band_names]
            if band_names
            else None
        )
        if result is None:
            if extra is not None:
                result = NetCDF.create_from_array(
                    arr=arr,
                    geo=geo,
                    epsg=epsg,
                    no_data_value=ndv,
                    variable_name=var_name,
                    extra_dims=extra,
                )
            else:
                result = NetCDF.create_from_array(
                    arr=arr,
                    geo=geo,
                    epsg=epsg,
                    no_data_value=ndv,
                    variable_name=var_name,
                )
        else:
            # A Dataset stores at most one (flattened) band axis, so collapse the
            # trailing band dimensions to a single (prod(sizes), rows, cols) store.
            # _materialize_variable_array reshapes it back using _band_dim_sizes.
            flat = (
                arr.reshape(-1, arr.shape[-2], arr.shape[-1])
                if len(band_names) > 1
                else arr
            )
            ds = Dataset.create_from_array(flat, geo=geo, epsg=epsg, no_data_value=ndv)
            ds._band_dim_names = tuple(band_names)
            ds._band_dim_values_map = {
                name: values_map.get(name) for name in band_names
            }
            ds._band_dim_sizes = tuple(arr.shape[i] for i in range(len(band_names)))
            ds._band_dim_name, ds._band_dim_values = NetCDF._derive_primary_band_view(
                ds._band_dim_names,
                ds._band_dim_values_map,
                ds._band_dim_sizes,
                ds._band_count,
            )
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

    def warped_view(
        self,
        crs: int | str | Any,
        method: str = "nearest neighbor",
        *,
        cell_size: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> NetCDF:
        """Return a lazy, reprojected view of a **variable subset**.

        Delegates to :meth:`pyramids.dataset.Dataset.warped_view` and re-wraps
        the VRT-backed result as `NetCDF`, preserving the variable-subset
        metadata (band dims, scale/offset, parent reference) so `sel()` and
        `read_array(unpack=True)` keep working on the view.

        A **root MDIM container** cannot be viewed lazily: a warped VRT is a
        classic single-variable raster, and warping every variable eagerly
        would contradict the lazy contract. Use :meth:`get_variable` to pick a
        variable first, or :meth:`to_crs` for an eager whole-container warp.

        Args:
            crs: Target CRS in any form :meth:`pyproj.CRS.from_user_input`
                accepts (EPSG int, ``"EPSG:3857"``, WKT, PROJ4, pyproj CRS).
            method: Resampling method used when windows are read. Defaults to
                ``"nearest neighbor"``.
            cell_size: Optional output pixel size in target-CRS units (applied
                to both axes). ``None`` keeps the source resolution.
            bbox: Optional ``(min_x, min_y, max_x, max_y)`` output extent in
                the **target** CRS; ``None`` covers the warped source extent.

        Returns:
            NetCDF: A read-only, VRT-backed reprojected view of the variable.

        Raises:
            ValueError: Called on a root MDIM container instead of a variable
                subset.

        See Also:
            NetCDF.to_crs: The eager reprojection (handles whole containers).
        """
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            raise ValueError(
                "warped_view works on a single variable, not a root NetCDF "
                "container — call get_variable(<name>) first and warp that, "
                "or use to_crs() for an eager whole-container reprojection."
            )
        pinned = super().warped_view(crs, method, cell_size=cell_size, bbox=bbox)
        result = self._preserve_netcdf_metadata(pinned)
        # Carry the GC pin: the VRT references the source GDAL handle, so the
        # re-wrapped NetCDF view must keep the source alive too. _preserve_netcdf
        # _metadata builds a fresh NetCDF and would otherwise drop _warp_source.
        result._warp_source = getattr(pinned, "_warp_source", self)
        return result

    def _materialize_md_view(self) -> None:
        """Replace a multidimensional ``AsClassicDataset`` view with a window-readable MEM raster.

        A variable subset's backing raster is a GDAL multidimensional ``AsClassicDataset`` view (see
        :meth:`_read_md_array`). GDAL >= 3.13 rejects any partial-window read of that view, raising
        ``RuntimeError: arrayStartIdx[...] >= <dim>``; only a full read succeeds. That breaks every
        eager partial read of the variable — the windowed curvilinear crop, ``resample``, and the COG
        ``CreateCopy``. Copying the view into an in-memory ``MEM`` raster does a single full read (which
        works) and yields a normal raster that GDAL can read with any window, both from pyramids and
        from GDAL-internal ``Warp`` / ``CreateCopy``.

        Idempotent; a no-op on a classic (non-multidim) subset. The lazy ``read_array(chunks=)`` path
        reads from the file directly and never touches this view, so it stays fully lazy.
        """
        if getattr(self, "_md_view_materialized", False):
            return
        # Only a variable subset carries an AsClassicDataset view. A root container's raster is the
        # multidim file dataset (0 bands) and must never be copied here.
        if not (self._is_md_array and self._is_subset) or self._raster is None:
            return
        mem = gdal.GetDriverByName("MEM").CreateCopy("", self._raster)
        # CreateCopy carries the view's geotransform; re-apply the wrapper's (it may hold a Y-flip or
        # coordinate-derived correction the raw view lacks).
        mem.SetGeoTransform(self._geotransform)
        self._raster = mem
        # The MEM copy owns its data; drop the SWIG views that backed the AsClassicDataset.
        self._gdal_md_arr_ref = None
        self._gdal_rg_ref = None
        self._md_view_materialized = True

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
            # resample warps the backing raster; a multidim view can't be window-read by GDAL >= 3.13.
            self._materialize_md_view()
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
                f"No bands match {dim_name}={selector}. " f"Available values: {coords}"
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
        stride = math.prod(sizes[dim_axis + 1 :])
        block = stride * sizes[dim_axis]
        total = math.prod(sizes)

        band_indices: list[int] = []
        for pinned in dim_indices:
            for outer_start in range(0, total, block):
                base = outer_start + pinned * stride
                band_indices.extend(range(base, base + stride))

        selected_coords = [coords[i] for i in dim_indices]

        # Read only the selected classic bands, straight into one pre-allocated buffer
        # (mirroring the all-bands read path in ``IO.read_array``) instead of N
        # ``read_array()`` calls collected into a list and then ``np.stack``-ed — that
        # allocated N throwaway arrays and copied them all a second time. Each 0-based
        # band index maps to a 1-based GDAL band in the classic dataset view built by
        # ``get_variable()``; reading only the selected bands still avoids materialising
        # the whole variable.
        if len(band_indices) == 1:
            selected = self._iloc(band_indices[0]).ReadAsArray()
        else:
            selected = np.empty(
                (len(band_indices), self.rows, self.columns),
                dtype=self.numpy_dtype[0],
            )
            for out_i, band_index in enumerate(band_indices):
                selected[out_i, :, :] = self._iloc(band_index).ReadAsArray()

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
            len(dim_indices) if i == dim_axis else s for i, s in enumerate(sizes)
        )
        result._band_dim_sizes = new_sizes
        result._band_dim_values_map = dict(self._band_dim_values_map)
        result._band_dim_values_map[dim_name] = selected_coords
        # Re-derive the legacy primary-dim view from the (now updated) canonical
        # fields so it tracks the pinned selection — single source of truth in
        # `_derive_primary_band_view`.
        result._band_dim_name, result._band_dim_values = self._derive_primary_band_view(
            result._band_dim_names,
            result._band_dim_values_map,
            result._band_dim_sizes,
            result._band_count,
        )

        return result

    @classmethod
    def read_file(  # type: ignore[override]
        cls,
        path: str | Path,
        read_only: bool = True,
        open_as_multi_dimensional: bool = True,
        file_i: int = 0,
        *,
        vsi: str | None = None,
    ) -> NetCDF:
        """Open a NetCDF file from a path, URL, or archive member.

        Plain local paths, ``/vsi*`` paths, and URL schemes
        (``http(s)://``, ``s3://``, ``gs://``, ``az://`` / ``abfs://``,
        ``file://``) are all accepted — URLs are transparently rewritten
        to GDAL's virtual filesystem. Compressed archives (``.zip`` /
        ``.tar`` / ``.tar.gz`` / ``.gz``) are detected from the
        extension; pass ``vsi=`` to be explicit about the archive kind
        (e.g. an archive without a recognised extension, or to open a
        specific member by index).

        Args:
            path: Path or URL of the ``.nc`` file or archive.
            read_only: If True, open in read-only mode. Set to False for
                write access. Defaults to True.
            open_as_multi_dimensional: If True, open with
                ``gdal.OF_MULTIDIM_RASTER`` to access the full group /
                dimension / variable hierarchy. If False, open in
                classic raster mode where each variable is a subdataset.
                Defaults to True.
            file_i: Which member to open when ``path`` is (or is forced
                to be) a multi-file archive. Default ``0``.
            vsi: Treat ``path`` as an archive of this kind and open
                member ``file_i`` from inside it: ``"zip"``, ``"tar"``
                (also ``"tar.gz"`` / ``"tgz"``), ``"gzip"`` (also
                ``"gz"``), or ``"auto"`` (infer from the extension).
                Default ``None`` — ``path`` is opened directly /
                extension-sniffed as before. GDAL's archive handlers
                key off the file-name extension, so an extension-less
                download URL must first be fetched and saved with a
                ``.zip`` name (or written to ``/vsimem/<name>.zip`` via
                :func:`osgeo.gdal.FileFromMemBuffer`).

                **Platform caveat for NetCDF:** GDAL's netCDF driver
                requires Linux ``userfaultfd`` to open a ``.nc`` from
                any ``/vsi*`` path (archive, ``/vsicurl/``, ``/vsimem/``
                via this route). On Windows / macOS the call raises a
                ``RuntimeError`` from GDAL pointing at the missing
                ``userfaultfd``. Use :meth:`from_bytes` to read a
                downloaded ``.nc`` from memory on those platforms.

        Returns:
            NetCDF: The opened dataset.

        Examples:
            - Open a plain ``.nc`` from disk and list its variables:
                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file(
                ...     "tests/data/netcdf/noah-precipitation-1979.nc"
                ... )
                >>> sorted(nc.variables)
                ['Band1', 'Band2', 'Band3', 'Band4']

                ```
            - Open a NetCDF held inside a zip — ``vsi="auto"`` infers
              the archive kind from the ``.zip`` extension. GDAL's
              netCDF driver needs Linux ``userfaultfd`` to read through
              ``/vsizip/``, so the open actually succeeds only on Linux;
              the ``try`` / ``except`` keeps the doctest runnable on
              Windows / macOS too (where it falls through with the
              ``RuntimeError`` GDAL raises):
                ```python
                >>> import tempfile, zipfile
                >>> from pathlib import Path
                >>> from pyramids.netcdf import NetCDF
                >>> src = Path("tests/data/netcdf/noah-precipitation-1979.nc")
                >>> with tempfile.TemporaryDirectory() as tmp:
                ...     zpath = Path(tmp) / "noah.zip"
                ...     with zipfile.ZipFile(zpath, "w") as zf:
                ...         zf.write(src, arcname="noah.nc")
                ...     try:
                ...         nc = NetCDF.read_file(zpath, vsi="auto")
                ...         variables = sorted(nc.variables)
                ...     except RuntimeError:
                ...         variables = ["Band1", "Band2", "Band3", "Band4"]
                >>> variables
                ['Band1', 'Band2', 'Band3', 'Band4']

                ```

        See Also:
            - :meth:`from_bytes`: open a NetCDF from in-memory bytes.
            - :meth:`pyramids.dataset.Dataset.read_file`: the same
              ``vsi=`` / ``file_i=`` surface for GeoTIFFs.
        """
        src = _io.read_file(
            path,
            read_only,
            open_as_multi_dimensional,
            file_i=file_i,
            vsi=vsi,
        )
        access = "read_only" if read_only else "write"
        return Container(
            src, access=access, open_as_multi_dimensional=open_as_multi_dimensional
        )

    @classmethod
    def from_bytes(  # type: ignore[override]
        cls,
        data: bytes | bytearray | memoryview,
        *,
        suffix: str = ".nc",
        name: str | None = None,
        read_only: bool = True,
        open_as_multi_dimensional: bool = True,
    ) -> NetCDF:
        """Open a NetCDF held in memory as a byte string.

        Writes ``data`` to a temporary GDAL ``/vsimem/`` path and opens
        it as a NetCDF — no on-disk temp file needed. Useful for HTTP
        response bodies, object-store payloads, and test fixtures.

        This is **not** a URL helper — see
        :meth:`pyramids.dataset.Dataset.from_bytes` for the rationale.
        The ``/vsimem/`` entry is removed automatically when the
        returned :class:`NetCDF` is garbage-collected.

        Args:
            data: Raw bytes of a NetCDF file.
            suffix: Extension hint for GDAL's driver detection. Defaults
                to ``".nc"``.
            name: Optional label recorded as :attr:`file_name`
                (cosmetic). Defaults to ``None``.
            read_only: Open read-only. Defaults to ``True``.
            open_as_multi_dimensional: Open with
                ``gdal.OF_MULTIDIM_RASTER`` to access the full group /
                dimension / variable hierarchy. Defaults to ``True``.

        Returns:
            NetCDF: The opened in-memory dataset.

        Raises:
            TypeError: ``data`` is not a bytes-like object.
            ValueError: GDAL could not open the bytes as a NetCDF.

        Examples:
            - Open the bytes of a NetCDF and list its variables (the bytes
              here come from a file, but could be ``requests.get(url).content``):
                ```python
                >>> from pathlib import Path
                >>> from pyramids.netcdf import NetCDF
                >>> data = Path("tests/data/netcdf/noah-precipitation-1979.nc").read_bytes()
                >>> nc = NetCDF.from_bytes(data, name="downloaded.nc")
                >>> list(nc.variables)
                ['Band1', 'Band2', 'Band3', 'Band4']
                >>> nc.epsg
                4326
                >>> nc.file_name
                'downloaded.nc'

                ```
            - An in-memory NetCDF cannot be pickled — anchor it to disk first:
                ```python
                >>> import pickle
                >>> from pathlib import Path
                >>> from pyramids.netcdf import NetCDF
                >>> data = Path("tests/data/netcdf/noah-precipitation-1979.nc").read_bytes()
                >>> try:
                ...     pickle.dumps(NetCDF.from_bytes(data))
                ... except TypeError as exc:
                ...     print("to_file" in str(exc))
                True

                ```

        See Also:
            - :meth:`read_file`: open a NetCDF from a path or URL.
            - :meth:`pyramids.dataset.Dataset.from_bytes`: the GeoTIFF variant.
        """
        src, vsi_path = _io.bytes_to_gdal(
            data,
            suffix=suffix,
            read_only=read_only,
            open_as_multi_dimensional=open_as_multi_dimensional,
        )
        try:
            obj = Container(
                src,
                access="read_only" if read_only else "write",
                open_as_multi_dimensional=open_as_multi_dimensional,
            )
        except Exception as e:
            src = None
            _io.silent_unlink(vsi_path)
            raise ValueError(
                "could not open the supplied bytes as a NetCDF dataset "
                f"(the data may be corrupt or truncated): {e}"
            ) from e
        obj._vsimem_path = vsi_path
        weakref.finalize(obj, _io.silent_unlink, vsi_path)
        if name is not None:
            obj._file_name = str(name)
        return obj

    def to_kerchunk(
        self,
        output_path,
        *,
        inline_threshold: int = 500,
        vlen_encode: str = "embed",
    ) -> dict:
        """Emit a kerchunk JSON reference manifest for this file.

        Thin forwarder to :func:`pyramids.netcdf._kerchunk_facade.to_kerchunk`
        using `self._file_name` as the source path. Requires the
        `[lazy]` optional extra.

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
        :func:`pyramids.netcdf._kerchunk_facade.combine_kerchunk`. Requires
        the `[lazy]` optional extra.

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

    @property
    def dimension_sizes(self) -> dict[str, int]:
        """Logical size of every dimension, in storage order, as ``{name: size}``.

        Reads the true dimension lengths from the multidimensional root group
        (e.g. ``{"time": 128568, "y": 3840, "x": 4608}``). Prefer this over
        :attr:`shape` on a chunked cloud store: ``shape`` reflects the classic
        single-raster view (band count + a chunk-sized window), so a remote Zarr
        whose CF time axis GDAL can't parse reports ``(0, chunk_y, chunk_x)``
        rather than the logical grid. Empty for a variable subset (no root group).

        Returns:
            dict[str, int]: Mapping of dimension name to its length; ``{}`` when
            the cube is a variable subset rather than a root MDIM container.

        Examples:
            - True grid sizes of a NWM retrospective cube (needs the bucket)::

                >>> nc.dimension_sizes  # doctest: +SKIP
                {'soil_layers_stag': 4, 'time': 128568, 'vis_nir': 2, 'x': 4608, 'y': 3840}
        """
        rg = self._raster.GetRootGroup() if self._raster is not None else None
        if rg is None:
            return {}
        return {dim.GetName(): int(dim.GetSize()) for dim in rg.GetDimensions()}

    def get_time_values(self, var_name: str = "time") -> np.ndarray | None:
        """Raw (undecoded) values of the time coordinate, or ``None`` if absent.

        Use this when :meth:`get_time_variable` returns ``None`` because the
        store's CF ``units`` are not parseable — some cloud Zarr stores do not
        surface ``units`` through GDAL, so dates can't be decoded. The raw
        offsets, together with the dimension's ``calendar`` / ``units`` attributes
        (``meta_data.get_dimension(var_name).attrs``) and :attr:`dimension_sizes`,
        let a caller map a date window to integer indices for :meth:`subset` and
        detect out-of-range — instead of hard-coding an unverifiable schedule.

        Args:
            var_name: Name of the time coordinate / dimension. Defaults to
                ``"time"``.

        Returns:
            numpy.ndarray or None: The raw coordinate values, or ``None`` when
            the store has no such dimension.

        Examples:
            - Raw 3-hourly offsets of the NWM retrospective cube (needs the
              bucket)::

                >>> nc.get_time_values("time")[:3]  # doctest: +SKIP
                array([0, 3, 6])
        """
        names = self.dimension_names
        if names is None or var_name not in names:
            return None
        return self._read_variable(var_name)

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
        return needs_y_flip(rg, md_arr)

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
            except (RuntimeError, ValueError):
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
            except RuntimeError:
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
            except RuntimeError:
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

        result = Container(dst)
        return result

    def get_variable_names(self) -> list[str]:
        """Deprecated alias for the :attr:`variable_names` property (API-3).

        Returns:
            list[str]: Same value as :attr:`variable_names`.
        """
        warnings.warn(
            "get_variable_names() is deprecated; use the `variable_names` property "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variable_names

    def _get_variable_names(self) -> list[str]:
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
                variable_names = self._mdim_data_variable_names(rg)
            else:
                variable_names = self._classic_subdataset_variable_names()
        return variable_names

    @staticmethod
    def _mdim_data_variable_names(rg) -> list[str]:
        """Data-variable names from an MDIM root group.

        Drops dimension coordinate arrays and 0-dimensional scalar variables
        (e.g. `grid_mapping` holders) so only true data variables remain.

        Args:
            rg: The store's multidimensional root :class:`osgeo.gdal.Group`.

        Returns:
            list[str]: Names of the data variables in `rg`.
        """
        dim_names = {dim.GetName() for dim in rg.GetDimensions()}
        filtered = []
        for var in rg.GetMDArrayNames():
            if var in dim_names:
                continue
            md_arr = rg.OpenMDArray(var)
            if md_arr is not None and len(md_arr.GetDimensions()) == 0:
                continue
            filtered.append(var)
        return filtered

    def _classic_subdataset_variable_names(self) -> list[str]:
        """Data-variable names parsed from classic-mode subdataset metadata.

        Returns:
            list[str]: The variable name from each `gdal.Dataset.GetSubDatasets()`
            entry (the second whitespace-delimited token of its description).
        """
        return [var[1].split(" ")[1] for var in self._raster.GetSubDatasets()]

    @staticmethod
    def _dimension_index(dim_names: list[str], target: str) -> int:
        """Index of a dimension matched by full name or short (leaf) name."""
        if target in dim_names:
            result = dim_names.index(target)
        else:
            short = target.lstrip("/").split("/")[-1]
            result = next(
                (i for i, name in enumerate(dim_names)
                 if name.lstrip("/").split("/")[-1] == short),
                None,
            )
            if result is None:
                raise ValueError(
                    f"dimension {target!r} not found; available dimensions: {dim_names}"
                )
        return result

    @staticmethod
    def _axis_role_of_dimension(dim) -> str | None:
        """Classify a dimension as ``"X"`` (longitude) or ``"Y"`` (latitude).

        Reads the CF attributes of the dimension's coordinate (indexing) variable —
        ``axis`` (``X``/``Y``), ``standard_name`` (``longitude``/``latitude``), or
        ``units`` (``degrees_east``/``degrees_north``) — and classifies them through the
        shared :func:`pyramids.netcdf.cf.detect_axis` heuristic. Returns ``None`` when the
        role cannot be determined.
        """
        indexing_var = dim.GetIndexingVariable()
        if indexing_var is None:
            return None
        # Attribute-only detection (empty ``name`` disables the name-pattern fallback) so the
        # CF-attribute vs. dimension-name stages stay separated, matching the historical pipeline.
        # Filter to the spatial roles this classifier promises (``detect_axis`` can also return
        # ``"T"``/``"Z"``), mirroring the MDIM ``_axis_role`` sibling so callers see only X/Y/None.
        role = detect_axis("", _read_attributes(indexing_var))
        return role if role in ("X", "Y") else None

    @staticmethod
    def _detect_axis_indices(dims) -> tuple[int | None, int | None]:
        """Indices of the X (longitude) and Y (latitude) dimensions via CF coordinate attributes.

        Returns the first dimension classified as ``"X"`` and the first as ``"Y"`` by
        ``_axis_role_of_dimension`` (each ``None`` when undetected).
        """
        detected_x = detected_y = None
        for i, dim in enumerate(dims):
            role = NetCDF._axis_role_of_dimension(dim)
            if role == "X" and detected_x is None:
                detected_x = i
            elif role == "Y" and detected_y is None:
                detected_y = i
        return detected_x, detected_y

    def _resolve_spatial_dims(
        self, md_arr, x_dim: str | None = None, y_dim: str | None = None
    ) -> tuple[int, int]:
        """Resolve the ``(x_index, y_index)`` raster-plane dimension indices.

        Precedence: explicit ``x_dim``/``y_dim`` names → CF auto-detection from the
        coordinate variables' ``axis``/``standard_name``/``units`` attributes → the
        last-two-dimensions default. This lets variables whose latitude/longitude are
        not the trailing dimensions (e.g. CAM ``T(time, lat, lev, lon)``) be read as a
        proper lat/lon plane instead of a lev/lon cross-section.
        """
        # Guard-clause/early-return style is intentional: a single-return rewrite nests the
        # detection and fallback branches and exceeds SonarCloud's S3776 complexity threshold.
        names = [d.GetName() for d in md_arr.GetDimensions()]
        n = len(names)

        explicit_x = self._dimension_index(names, x_dim) if x_dim is not None else None
        explicit_y = self._dimension_index(names, y_dim) if y_dim is not None else None
        if explicit_x is not None and explicit_y is not None:
            if explicit_x == explicit_y:
                raise ValueError(
                    f"x_dim and y_dim must be different dimensions; both resolved to "
                    f"index {explicit_x}. Dimensions: {names}"
                )
            return explicit_x, explicit_y

        detected_x, detected_y = self._detect_axis_indices(md_arr.GetDimensions())
        # Adopt detection only when it (combined with any single explicit side) yields a
        # complete, distinct (X, Y) pair — otherwise a partially-detected axis must not
        # collide with the last-two fallback (e.g. a 2-D `lon_bnds(lon, bnds)` variable).
        candidate_x = explicit_x if explicit_x is not None else detected_x
        candidate_y = explicit_y if explicit_y is not None else detected_y
        if candidate_x is not None and candidate_y is not None and candidate_x != candidate_y:
            return candidate_x, candidate_y

        # Fallback: the last two dimensions, keeping a non-colliding explicit side.
        x_index = explicit_x if explicit_x is not None else n - 1
        y_index = explicit_y if explicit_y is not None else n - 2
        if x_index == y_index:
            x_index, y_index = n - 1, n - 2
        return x_index, y_index

    def _read_md_array(
        self, variable_name: str, x_dim: str | None = None, y_dim: str | None = None
    ):
        """Convert an MDArray to a classic GDAL dataset via AsClassicDataset.

        The X (columns) and Y (rows) dimensions are resolved by
        `_resolve_spatial_dims` (explicit `x_dim`/`y_dim`, else CF
        auto-detection, else the last two dimensions); all remaining
        dimensions are flattened into bands.

        If the Y dimension is stored south-to-north (positive Y pixel
        size), it is reversed via `MDArray.GetView()` **before** the
        conversion. This is a lazy, zero-copy operation — GDAL handles
        the reversed indexing internally without reading the whole array.

        Returns a tuple `(classic_dataset, md_array, root_group, x_index,
        y_index)` so callers can keep the GDAL objects alive and reuse the
        resolved plane indices (`x_index`/`y_index` are `None` for a 1-D
        variable). `AsClassicDataset`
        returns a **view** whose C++ backing depends on the MDArray and
        root group; if the Python SWIG wrappers for those are garbage-
        collected the view becomes a dangling pointer (segfault on
        Windows).
        """
        rg = self._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        dims = md_arr.GetDimensions()

        if len(dims) == 1:
            # A classic 2-D raster view needs >=2 dimensions, so AsClassicDataset cannot represent a
            # 1-D variable (a coordinate axis or a 1-D data series). Return the MDArray itself, matching
            # the 1-D string path and avoiding GDAL's "Invalid iXDim and/or iYDim" error (#582).
            return md_arr, md_arr, rg, None, None

        # Resolve on the original array — the Y-flip below renames the Y dimension and drops its
        # indexing variable, so the indices must be computed before flipping and returned to the caller.
        x_index, y_index = self._resolve_spatial_dims(md_arr, x_dim, y_dim)

        # First pass: check if Y orientation needs flipping.
        src = md_arr.AsClassicDataset(x_index, y_index, rg)

        if src.GetGeoTransform()[5] > 0:
            # Positive Y pixel size = south-to-north (NetCDF convention).
            # Use GetView to reverse the Y dimension — this is lazy and
            # zero-copy; GDAL handles reversed indexing internally.
            slices = ",".join("::-1" if i == y_index else ":" for i in range(len(dims)))
            md_arr = md_arr.GetView(f"[{slices}]")
            src = md_arr.AsClassicDataset(x_index, y_index, rg)

        return src, md_arr, rg, x_index, y_index

    def get_variable(
        self, variable_name: str, x_dim: str | None = None, y_dim: str | None = None
    ) -> NetCDF:
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
            x_dim: Dimension to map to the raster X axis (columns).
                When omitted, the longitude dimension is auto-detected
                from the coordinate variables' CF attributes (`axis`,
                `standard_name`, `units`), falling back to the last
                dimension. Use this for files whose lon/lat are not the
                trailing dims and lack CF axis metadata.
            y_dim: Dimension to map to the raster Y axis (rows). When
                omitted, the latitude dimension is auto-detected, else
                the second-to-last dimension is used.

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
            cube = group_nc.get_variable(parts[1], x_dim=x_dim, y_dim=y_dim)
            return cube  # single return below handles non-group path

        if variable_name not in self.variable_names:
            raise ValueError(
                f"{variable_name} is not a valid variable name in {self.variable_names}"
            )

        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()
        md_arr_ref = None
        rg_ref = None

        spatial_dim_indices: tuple[int, int] | None = None
        if prefix == "MEMORY" or rg is not None:
            src, md_arr_ref, rg_ref, x_index, y_index = self._read_md_array(
                variable_name, x_dim=x_dim, y_dim=y_dim
            )
            if x_index is not None:
                spatial_dim_indices = (x_index, y_index)
            if isinstance(src, gdal.Dataset):
                cube = Variable(src)
                cube._is_md_array = True
                # _read_md_array flips the data lazily and GDAL usually corrects the geotransform,
                # but a Y dim with no indexing variable (e.g. WRF "south_north") can leave it wrong;
                # fix it on the wrapper (no data copy).
                self._correct_flipped_geotransform(cube)
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
            cube = Variable(src)
            cube._is_md_array = False

        cube._is_subset = True

        # --- RT-4: Track variable origin for round-trip ---
        cube._parent_nc = self
        cube._source_var_name = variable_name

        # Geostationary (GOES) scan-angle x/y come through the MDIM read path in
        # radians; rescale them to projected metres so the cube is correctly
        # georeferenced and to_crs/crop work. No-op for every other CRS. Guarded
        # because a 1-D string variable yields a raw MDArray, not a NetCDF.
        if isinstance(cube, NetCDF):
            cube._normalize_geostationary_geotransform()

        self._attach_variable_metadata(
            cube, md_arr_ref if rg is not None else None, spatial_dim_indices
        )
        cube = self._georeference_index_subset(cube)
        return cube

    @staticmethod
    def _correct_flipped_geotransform(cube: NetCDF) -> None:
        """Flip a south-to-north geotransform (positive Y pixel size) to north-up on the wrapper.

        A no-op unless `cube._geotransform[5] > 0`. Used after a lazy `GetView` Y-flip when the Y
        dimension has no indexing variable, so GDAL could not correct the geotransform itself.
        """
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

    def _georeference_index_subset(self, cube: "NetCDF") -> "NetCDF":
        """Re-georeference a variable subset whose MDArray view came back in index space.

        A subset built from a bare MDArray view can carry an index-space geotransform (cell
        size 1, origin 0) even though the file has real 1-D lon/lat coordinate variables. Those
        coordinates aren't reachable from the subset itself (it has no root group and an empty
        ``file_name``), but they are on this parent container. When they match the subset's grid
        shape and disagree with the view's geotransform, wrap the view in a VRT carrying the
        coordinate-derived geotransform, so warp-based operations (``to_crs`` / ``crop`` /
        ``wrap_longitude``) and basemaps use real degrees instead of pixel indices. The view's
        geotransform is immutable (``SetGeoTransform`` is a no-op on it), hence the VRT wrapper.

        A no-op when: the cube isn't a variable subset; the view is already georeferenced (the
        common case — the derived geotransform matches); or the file has no 1-D lon/lat matching
        the grid shape (curvilinear 2-D coordinates, named coordinate variables, etc.). The
        coordinates are read from the parent rather than via ``cube.lon`` / ``cube.lat`` so those
        accessors keep their existing geotransform-derived (north-up) orientation.
        """
        if isinstance(cube, NetCDF):
            real_gt = self._coordinate_derived_geotransform(cube)
            if real_gt is not None:
                vrt = gdal.Translate("", cube._raster, format="VRT")
                if vrt is not None:
                    vrt.SetGeoTransform(list(real_gt))
                    if cube.epsg:
                        vrt.SetProjection(sr_from_epsg(int(cube.epsg)).ExportToWkt())
                    # The VRT reads through the MDArray view, so keep that view alive.
                    cube._view_source = cube._raster
                    cube._raster = vrt
                    cube._geotransform = real_gt
                    cube._cell_size = real_gt[1]
        return cube

    def _coordinate_derived_geotransform(self, cube: "NetCDF") -> tuple | None:
        """Real-world geotransform from the parent's 1-D lon/lat, or ``None`` if not applicable.

        Returns the north-up affine implied by the parent container's 1-D ``lon``/``lat`` (or
        ``x``/``y``) coordinate variables when they (a) match the subset's grid shape, (b) index the
        subset's spatial dimensions by name (CF coordinate-variable convention — guards a same-shaped
        but different staggered/rotated axis from adopting the wrong coordinates), and (c) actually
        differ from the subset's current (index-space) geotransform. Otherwise ``None``.
        """
        result = None
        lon, lon_name = None, None
        for cand in ("lon", "x"):
            arr = self._read_variable(cand)
            if arr is not None:
                lon, lon_name = np.asarray(arr), cand
                break
        lat = None
        for cand in ("lat", "y"):
            arr = self._read_variable(cand)
            if arr is not None:
                lat = np.asarray(arr)
                break
        # Adopt the parent's lon/lat only when the variable actually has the longitude coordinate
        # dimension (by the CF coordinate-variable convention a 1-D coord var shares its dimension's
        # name). Test membership, not position, so it holds when x_dim/y_dim select a non-trailing
        # plane (e.g. T(time, lat, lev, lon)). Only the X (longitude) dim is checked: the Y-flip in
        # _read_md_array renames the latitude dimension (e.g. ``subset_lat_…``), so the lat name is
        # not reliably present. This still guards a same-shaped but unrelated axis (one with no
        # longitude dimension) from adopting the wrong coordinates.
        dim_names = getattr(cube, "_md_array_dims", None) or []
        names_ok = (not dim_names) or (lon_name in dim_names)
        if (
            lon is not None
            and lat is not None
            and lon.ndim == 1
            and lat.ndim == 1
            and len(lon) == cube.columns
            and len(lat) == cube.rows
            and len(lon) >= 2
            and len(lat) >= 2
            and names_ok
        ):
            x_cell = abs(float(lon[1] - lon[0]))
            y_cell = abs(float(lat[1] - lat[0]))
            y_top = max(float(lat[0]), float(lat[-1])) + y_cell / 2
            real_gt = (float(lon[0]) - x_cell / 2, x_cell, 0.0, y_top, 0.0, -y_cell)
            current = cube._raster.GetGeoTransform()
            if not all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(real_gt, current)):
                result = real_gt
        return result

    def _attach_variable_metadata(
        self, cube: NetCDF, md_arr, spatial_dim_indices: tuple[int, int] | None
    ) -> None:
        """Populate band-dim tracking, variable attributes, and packing on a variable subset.

        When `md_arr` is `None` (e.g. the file-backed gdal.Open path or a 1-D string variable) the
        band/attr metadata is cleared to empty defaults.
        """
        if md_arr is None:
            self._clear_variable_metadata(cube)
            return
        dims = md_arr.GetDimensions()
        cube._md_array_dims = [d.GetName() for d in dims]
        self._track_band_dimensions(cube, dims, spatial_dim_indices)
        self._copy_variable_attrs(cube, md_arr)

    @staticmethod
    def _clear_variable_metadata(cube: NetCDF) -> None:
        """Reset a subset's band-dimension, attribute, and packing metadata to empty defaults."""
        cube._md_array_dims = []
        cube._band_dim_name = None
        cube._band_dim_values = None
        cube._band_dim_names = ()
        cube._band_dim_values_map = {}
        cube._band_dim_sizes = ()
        cube._variable_attrs = {}
        cube._scale = None
        cube._offset = None

    @staticmethod
    def _track_band_dimensions(
        cube: NetCDF, dims, spatial_dim_indices: tuple[int, int] | None
    ) -> None:
        """Map every non-spatial dimension onto bands so `sel()` can address 4-D+ variables.

        The spatial (X/Y) dimensions are taken from `spatial_dim_indices` (resolved on the unflipped
        array) when available, else the last two. The legacy `_band_dim_name`/`_band_dim_values`
        fields point at the first non-spatial dim so existing 3-D consumers are unaffected.
        """
        if len(dims) > 2:
            spatial = (
                set(spatial_dim_indices)
                if spatial_dim_indices is not None
                else {len(dims) - 1, len(dims) - 2}
            )
            band_dims = [d for i, d in enumerate(dims) if i not in spatial]
        else:
            band_dims = []

        if not band_dims:
            cube._band_dim_name = None
            cube._band_dim_values = None
            cube._band_dim_names = ()
            cube._band_dim_values_map = {}
            cube._band_dim_sizes = ()
            return

        cube._band_dim_names = tuple(d.GetName() for d in band_dims)
        cube._band_dim_sizes = tuple(d.GetSize() for d in band_dims)
        cube._band_dim_values_map = {
            d.GetName(): NetCDF._read_band_dim_values(d) for d in band_dims
        }
        cube._band_dim_name, cube._band_dim_values = NetCDF._derive_primary_band_view(
            cube._band_dim_names,
            cube._band_dim_values_map,
            cube._band_dim_sizes,
            cube._band_count,
        )

    @staticmethod
    def _read_band_dim_values(dim):
        """Indexing-variable values for a band dimension, or integer indices when unreadable.

        String-typed indexing variables (e.g. WRF `Times`) cannot be read via `ReadAsArray` in the
        GDAL SWIG bindings, so they fall back to `[0, 1, ..., size - 1]`.
        """
        indexing_var = dim.GetIndexingVariable()
        if indexing_var is None:
            return None
        try:
            return indexing_var.ReadAsArray().tolist()
        except RuntimeError:
            return list(range(dim.GetSize()))

    @staticmethod
    def _copy_variable_attrs(cube: NetCDF, md_arr) -> None:
        """Copy the variable's attributes and CF `scale`/`offset` packing onto the subset."""
        cube._variable_attrs = _read_attributes(md_arr)
        try:
            cube._scale = md_arr.GetScale()
            cube._offset = md_arr.GetOffset()
        except (RuntimeError, AttributeError):
            cube._scale = None
            cube._offset = None

    def _writable_root_group(self) -> tuple[gdal.Dataset, gdal.Group]:
        """Return a ``(dataset, root_group)`` pair that is safe to mutate.

        A file-backed netCDF root group is opened in "data mode", which rejects
        ``CreateMDArray`` / ``DeleteMDArray`` / ``CreateDimension``. So for a file-backed
        container this copies the store into an in-memory ``MEM`` raster and returns that;
        an already in-memory container is returned as-is. The caller is responsible for
        swapping the returned dataset in via :meth:`_replace_raster`.

        Returns:
            tuple: The writable :class:`osgeo.gdal.Dataset` and its root group.
        """
        if self.driver_type == "memory":
            dst = self._raster
        else:
            dst = gdal.GetDriverByName("MEM").CreateCopy("", self._raster, 0)
        return dst, dst.GetRootGroup()

    @staticmethod
    def _derive_primary_band_view(
        names: tuple[str, ...],
        values_map: dict[str, list[Any] | None],
        sizes: tuple[int, ...],
        band_count: int,
    ) -> tuple[str | None, list[Any] | None]:
        """Derive the legacy ``(_band_dim_name, _band_dim_values)`` view from the canonical fields.

        The legacy single-band-dim pair is a *view* of the canonical multi-band-dim
        state: the name is the first (primary) band dimension and the values are that
        dim's entry in ``values_map``. This staticmethod is the single source of truth
        for that derivation — it replaces the per-call-site reconciliation that the
        wrap/build/``sel`` paths used to repeat. The primary coordinate values are
        exposed only when they are provably current for ``band_count``; otherwise they
        come back ``None`` because a band-shrinking operation left the cached primary
        view stale.

        Args:
            names: Canonical band-dimension names (``_band_dim_names``); the first is
                the primary dim the legacy view points at.
            values_map: Per-dim coordinate values (``_band_dim_values_map``).
            sizes: Per-dim sizes (``_band_dim_sizes``); their product is the expected
                band count for a multi-band-dim variable.
            band_count: The live band count of the backing raster.

        Returns:
            tuple: ``(name, values)`` for the primary band dim. ``name`` is ``None``
            when there is no band dimension; ``values`` is ``None`` when there are no
            coordinates or the cached primary view is stale for ``band_count``.
        """
        name = names[0] if names else None
        if name is None:
            return None, None
        values = values_map.get(name)
        if values is None or band_count <= 0:
            return name, values
        if len(names) > 1:
            # Multi-band-dim: the primary view is valid iff the product of the cached
            # sizes still equals the live band count (e.g. a band-shrinking op outside
            # ``sel()`` would diverge them, making the cached primary view stale).
            return name, (values if math.prod(sizes) == band_count else None)
        # Single-band-dim: valid iff the values' own length matches the band count.
        return name, (values if len(values) == band_count else None)

    @staticmethod
    def _copy_band_dim_metadata(dst: Any, src: Any) -> None:
        """Copy the band-dimension bookkeeping from ``src`` onto ``dst``.

        Carries the multi-band-dim fields (``_band_dim_names`` / ``_band_dim_values_map``
        / ``_band_dim_sizes``) and re-derives the legacy single-band-dim view
        (``_band_dim_name`` / ``_band_dim_values``) from them via
        :meth:`_derive_primary_band_view`, so a derived view keeps the same non-spatial
        axis layout. The values map is shallow-copied so the two objects don't share a
        mutable dict.
        """
        dst._band_dim_names = src._band_dim_names
        dst._band_dim_values_map = dict(src._band_dim_values_map)
        dst._band_dim_sizes = src._band_dim_sizes
        dst._band_dim_name, dst._band_dim_values = NetCDF._derive_primary_band_view(
            dst._band_dim_names,
            dst._band_dim_values_map,
            dst._band_dim_sizes,
            dst._band_count,
        )

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
        # Clear the per-variable geostationary geotransform cache too: it is keyed by
        # variable name and derived from the backing geometry, so it must not survive a
        # raster swap / in-place update that could change that geometry (latent staleness).
        self._geostationary_gt_cache = {}

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
            self._write_netcdf(path)
        elif self._is_md_array and not self._is_subset:
            raise ValueError(
                "Cannot save a multidimensional NetCDF container as "
                f"'{extension}'. Use .nc extension or extract a "
                "variable first with .get_variable()."
            )
        else:
            super().to_file(path, **kwargs)

    def _write_netcdf(self, path: Path) -> None:
        """Write this dataset to a netCDF file, preserving the declared (or absent) convention.

        Uses GDAL's ``CreateCopy`` and falls back to a manual multidim copy when that raises on some
        dimension layouts (#584), then strips any writer-injected ``Conventions`` if the source declared
        none (#583).
        """
        source_conventions = self.global_attributes.get("Conventions")
        try:
            dst = gdal.GetDriverByName("netCDF").CreateCopy(str(path), self._raster, 0)
        except RuntimeError:
            # GDAL's netCDF CreateCopy raises on some dimension layouts (re-declaring a dimension
            # name, #584). Fall back to a manual multidim copy that creates each dimension once.
            self._remove_path(path)
            self._manual_netcdf_copy(path)
        else:
            if dst is None:
                raise RuntimeError(f"Failed to save NetCDF to {path}")
            dst.FlushCache()
            dst = None
        if source_conventions is None:
            self._strip_injected_conventions(path)

    @staticmethod
    def _remove_path(path: Path) -> None:
        """Delete a (possibly GDAL-locked) file, retrying once after a GC pass."""
        if not path.exists():
            return
        try:
            path.unlink()
        except OSError:
            gc.collect()
            path.unlink()

    def _manual_netcdf_copy(self, path: str | Path) -> None:
        """Write this container to netCDF by copying each variable explicitly.

        Fallback for files where GDAL's ``CreateCopy`` raises while re-declaring dimensions (#584).
        Each dimension is created once via ``_add_md_array_to_group`` -> ``_resolve_dst_dimensions``.
        Root-level variables and global attributes only; hierarchical groups are not handled here
        (those files copy fine through ``CreateCopy``), so this raises if the source has subgroups.
        """
        src_rg = self._raster.GetRootGroup()
        if src_rg is None:
            raise RuntimeError("manual netCDF copy requires a multidimensional container")
        if src_rg.GetGroupNames():
            raise RuntimeError(
                f"manual netCDF copy of {path} does not support hierarchical groups"
            )
        dst = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(path), ["FORMAT=NC4"])
        dst_rg = dst.GetRootGroup()
        self._copy_md_array_attributes(src_rg, dst_rg)
        for var_name in src_rg.GetMDArrayNames():
            self._add_md_array_to_group(dst_rg, var_name, src_rg.OpenMDArray(var_name))
        dst_rg = None
        dst.FlushCache()
        dst = None
        gc.collect()

    @staticmethod
    def _md_array_to_numpy(md_arr):
        """Read an MDArray to numpy, using the list-based ``Read()`` for string arrays.

        GDAL's SWIG ``ReadAsArray`` raises ("String buffer data type not supported") on string MDArrays,
        so character coordinate/data variables are read via ``Read()`` and wrapped as a numpy array (#586,
        same limitation handled in ``_add_md_array_to_group`` for #565).
        """
        if md_arr.GetDataType().GetClass() == gdal.GEDTC_STRING:
            return np.array(md_arr.Read())
        return md_arr.ReadAsArray()

    @staticmethod
    def _strip_injected_conventions(path: str | Path) -> None:
        """Delete a writer-injected ``Conventions`` global attribute from a just-written netCDF file.

        GDAL's netCDF driver adds a default ``Conventions`` (e.g. ``CF-1.6``) on write when the source
        declares none. Callers use this to keep a no-convention file from being relabelled as CF (#583).
        """
        ds = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER | gdal.OF_UPDATE)
        if ds is None:
            return
        rg = ds.GetRootGroup()
        if rg is not None and any(a.GetName() == "Conventions" for a in rg.GetAttributes()):
            try:
                rg.DeleteAttribute("Conventions")
            except RuntimeError:
                pass  # driver may not support attribute deletion — leave it
        rg = None
        ds.FlushCache()
        ds = None
        gc.collect()

    def copy(self, path: str | Path | None = None) -> NetCDF:
        """Create a deep, standalone copy of this dataset.

        The copy keeps this instance's concrete type (a container copies to a
        ``Container``, a variable to a ``Variable``) and its CF packing metadata
        (``scale`` / ``offset`` / variable attributes / band-dim layout). It
        is **independent**, though: a copied variable is a self-contained classic raster, not
        a live subset of the original's parent, so ``is_subset`` is ``False`` and it carries
        no ``_parent_nc`` / ``_source_var_name``. That keeps pickling sound — a copy
        reconstructs from its own data rather than reaching back into the parent store.

        Args:
            path: Destination file path. If None, the copy is created
                in memory using the MEM driver. Defaults to None.

        Returns:
            NetCDF: A new, standalone NetCDF object (same concrete subclass as ``self``).

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
        # Preserve both the concrete type AND the variable-subset / origin identity: a copy
        # of a container is a container; a copy of a variable is a usable variable subset.
        # The fresh CreateCopy would otherwise reset these flags through __init__ (defaulting
        # open_as_multi_dimensional=True, _is_subset=False), so re-open in the source's mode
        # and carry the subset/origin + cached variable metadata over.
        # A copied variable subset is a standalone single-variable *classic* raster (its
        # backing is the materialized AsClassicDataset view), not an MDIM store — so it must
        # open classic. A container copy keeps the container's MDIM/classic mode.
        copy_is_md = self._is_md_array and not self._is_subset
        result = type(self)(src, access="write", open_as_multi_dimensional=copy_is_md)
        # A copy is an INDEPENDENT, materialized dataset: its data no longer lives at
        # parent_file::source_var_name, so it must NOT inherit the subset/origin identity
        # that __reduce__ uses to reconstruct (that would pickle-reconstruct from the PARENT,
        # silently reading the wrong data, or fail on a self-contained copy whose band names
        # differ). It keeps its concrete type and CF packing metadata, but is standalone.
        result._is_subset = False
        result._parent_nc = None
        result._source_var_name = None
        result._md_array_dims = self._md_array_dims
        result._geostationary_scaled = self._geostationary_scaled
        result._variable_attrs = self._variable_attrs
        result._scale = self._scale
        result._offset = self._offset
        NetCDF._copy_band_dim_metadata(result, self)
        return result

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
            ClassicDimensionInfo(
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
        result = Container(dst_ds)

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
        return [(f"dim_{i}", list(range(int(arr.shape[i])))) for i in range(expected)]

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
    def _resolve_dst_dimensions(dst_group, src_dims):
        """Map source dimensions onto `dst_group`, creating any that are missing.

        Reuses a destination dimension when one already exists with the same
        name and size (so same-group copies like `rename_variable` keep sharing
        their dimensions); otherwise recreates it from the source dimension's
        name, type, direction, and size. This lets `_add_md_array_to_group`
        copy a variable into a *different* container's group.
        """
        existing = {dim.GetName(): dim for dim in (dst_group.GetDimensions() or [])}
        resolved = []
        for dim in src_dims:
            size = dim.GetSize()
            match = existing.get(dim.GetName())
            if match is not None and match.GetSize() == size:
                resolved.append(match)
                continue
            # Name taken by a different-sized dimension: fall back to a size-suffixed name (matching
            # _get_or_create_dimension), reusing a same-size match and uniquifying on any collision.
            name = dim.GetName() if match is None else f"{dim.GetName()}_{size}"
            existing_match = existing.get(name)
            if existing_match is not None and existing_match.GetSize() == size:
                resolved.append(existing_match)
                continue
            suffix = 1
            while name in existing and existing[name].GetSize() != size:
                name = f"{dim.GetName()}_{size}_{suffix}"
                suffix += 1
            new_dim = dst_group.CreateDimension(
                name, dim.GetType(), dim.GetDirection(), size
            )
            existing[name] = new_dim
            resolved.append(new_dim)
        return resolved

    @staticmethod
    def _copy_md_array_attributes(src_mdarray, dst_mdarray):
        """Copy every attribute from one MDArray to another, preserving dtype.

        GDAL's `Attribute.Write` routes through `WriteRaw`, which rejects numeric
        tuples, so each attribute is written with the type-specific call that
        matches its class (string vs integer vs floating point) and arity.
        """
        for attr in src_mdarray.GetAttributes():
            name = attr.GetName()
            data_type = attr.GetDataType()
            count = attr.GetTotalElementsCount()
            dims = [] if count <= 1 else [count]
            if data_type.GetClass() == gdal.GEDTC_STRING:
                new_attr = dst_mdarray.CreateAttribute(
                    name, dims, gdal.ExtendedDataType.CreateString()
                )
                if count <= 1:
                    new_attr.WriteString(attr.ReadAsString())
                else:
                    new_attr.WriteStringArray(attr.ReadAsStringArray())
                continue
            numeric_type = data_type.GetNumericDataType()
            new_attr = dst_mdarray.CreateAttribute(
                name, dims, gdal.ExtendedDataType.Create(numeric_type)
            )
            NetCDF._write_numeric_attribute(new_attr, attr, numeric_type, count)

    @staticmethod
    def _write_numeric_attribute(new_attr, src_attr, numeric_type, count):
        """Write a numeric attribute with the GDAL call matching its dtype class and arity.

        64-bit integer codes (``GDT_Int64`` / ``GDT_UInt64``) use the 64-bit ``WriteInt64`` /
        ``ReadAsInt64`` calls; the 32-bit ``WriteInt`` / ``ReadAsInt`` would truncate large values
        (e.g. an Int64 ``_FillValue``). Other integer codes use the 32-bit integer calls and
        everything else uses the floating-point calls.

        Args:
            new_attr: The destination ``gdal.Attribute`` to write into.
            src_attr: The source ``gdal.Attribute`` to read from.
            numeric_type: The GDAL numeric data type code of the attribute.
            count: Total element count (``<= 1`` is written as a scalar, otherwise as an array).
        """
        is_int64 = numeric_type in (gdal.GDT_Int64, gdal.GDT_UInt64)
        is_integer = numeric_type in _GDAL_INTEGER_DTYPES
        if count <= 1:
            if is_int64:
                new_attr.WriteInt64(src_attr.ReadAsInt64())
            elif is_integer:
                new_attr.WriteInt(src_attr.ReadAsInt())
            else:
                new_attr.WriteDouble(src_attr.ReadAsDouble())
        elif is_int64:
            new_attr.WriteInt64Array(src_attr.ReadAsInt64Array())
        elif is_integer:
            new_attr.WriteIntArray(src_attr.ReadAsIntArray())
        else:
            new_attr.WriteDoubleArray(src_attr.ReadAsDoubleArray())

    @staticmethod
    def _add_md_array_to_group(dst_group, var_name, src_mdarray):
        """Copy an MDArray into `dst_group`, preserving data, packing, and metadata.

        Dimensions are resolved against the destination group, so the source may
        live in a different container. The on-disk packing (`scale`/`offset`),
        `unit`, no-data value, spatial reference, and all variable attributes are
        carried across. Scale/offset/unit/no-data are set before the data is
        written so the netCDF driver accepts the fill value.
        """
        src_dims = NetCDF._resolve_dst_dimensions(
            dst_group, src_mdarray.GetDimensions()
        )
        if src_mdarray.GetDataType().GetClass() == gdal.GEDTC_STRING:
            # String MDArrays can't go through ReadAsArray (numpy) in the GDAL
            # SWIG bindings, but the Python list Read()/Write() path works. Use it
            # so non-spatial string aux vars (e.g. ERA5's 'expver') are carried
            # through container spatial ops instead of being dropped (#565).
            new_md_array = dst_group.CreateMDArray(
                var_name, src_dims, gdal.ExtendedDataType.CreateString()
            )
            NetCDF._copy_md_array_attributes(src_mdarray, new_md_array)
            new_md_array.Write(src_mdarray.Read())
        else:
            arr = src_mdarray.ReadAsArray()
            dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
            new_md_array = dst_group.CreateMDArray(var_name, src_dims, dtype)
            scale = src_mdarray.GetScale()
            if scale is not None:
                new_md_array.SetScale(scale)
            offset = src_mdarray.GetOffset()
            if offset is not None:
                new_md_array.SetOffset(offset)
            unit = src_mdarray.GetUnit()
            if unit:
                new_md_array.SetUnit(unit)
            ndv = src_mdarray.GetNoDataValue()
            if ndv is not None:
                try:
                    new_md_array.SetNoDataValueDouble(ndv)
                except (RuntimeError, TypeError, ValueError):
                    pass
            new_md_array.SetSpatialRef(src_mdarray.GetSpatialRef())
            NetCDF._copy_md_array_attributes(src_mdarray, new_md_array)
            new_md_array.Write(arr)

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
        if rg is not None:
            result = _read_attributes(rg)
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
        except RuntimeError:
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
        except RuntimeError:
            pass  # attribute may not exist — silently ignored
        self._invalidate_caches()

    def set_variable(self, *args, **kwargs):
        """Facade — :meth:`Variables.set_variable <pyramids.netcdf.engines.variables.Variables.set_variable>`."""
        return self.varops.set_variable(*args, **kwargs)

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
        NetCDF._copy_band_dim_metadata(materialized, var)
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

    def add_variable(self, *args, **kwargs):
        """Facade — :meth:`Variables.add_variable <pyramids.netcdf.engines.variables.Variables.add_variable>`."""
        return self.varops.add_variable(*args, **kwargs)

    def remove_variable(self, *args, **kwargs):
        """Facade — :meth:`Variables.remove_variable <pyramids.netcdf.engines.variables.Variables.remove_variable>`."""
        return self.varops.remove_variable(*args, **kwargs)

    def rename_variable(self, *args, **kwargs):
        """Facade — :meth:`Variables.rename_variable <pyramids.netcdf.engines.variables.Variables.rename_variable>`."""
        return self.varops.rename_variable(*args, **kwargs)

    def to_xarray(self, *args, **kwargs) -> Any:
        """Facade — delegates to :meth:`Interop.to_xarray <pyramids.netcdf.engines.interop.Interop.to_xarray>`."""
        return self.interop.to_xarray(*args, **kwargs)

    def subset(
        self,
        variable: str,
        *,
        time: int | slice | tuple[int, int] | None = None,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        crs: int | str = 4326,
        densify: int = 25,
        y_dim: str | None = None,
        x_dim: str | None = None,
        **dims: int | tuple[int, int] | slice,
    ) -> NetCDF:
        """Read a windowed ``(variable, time, bbox)`` slice of a gridded cube.

        Reads only the requested window from a CF/GeoZarr ``(time, y, x[, …])``
        multidimensional store — local or remote — without materialising the
        whole variable, and returns a georeferenced single-variable
        :class:`~pyramids.netcdf.NetCDF` ready for ``to_file`` / ``to_cog`` /
        ``to_crs`` / ``crop`` (a ``Dataset`` subclass, so existing
        ``isinstance(result, Dataset)`` checks keep working).

        Designed for huge cloud cubes (e.g. the NWM retrospective
        ``ldasout.zarr``, an 18 TiB ``(128568, 3840, 4608)`` store) opened
        anonymously via :class:`~pyramids.base.remote.CloudConfig`; only the
        sliced cells are fetched. The output CRS is the variable's own grid
        mapping (read from the multidimensional array), so a Lambert Conformal
        Conic store stays on its native grid.

        Args:
            variable: Data-variable name in the store (e.g. ``"ACCET"``).
            time: Timestep selector along the time dimension. An ``int`` picks
                one step (one output band); a ``(start, stop)`` tuple or
                ``slice`` picks a half-open index range (one band per step);
                ``None`` is allowed only when the time dimension has length 1.
                Selection is by **integer index** — date/label selection needs
                the store to expose CF time ``units``, which many Zarr stores do
                not surface through GDAL, so use indices for those.
            bbox: ``(min_x, min_y, max_x, max_y)`` crop window in ``crs``.
                ``None`` keeps the full grid. The box is reprojected onto the
                store's native grid (so a lon/lat box over a projected grid is
                handled) honouring the variable's grid mapping.
            crs: CRS of ``bbox`` — EPSG int, ``"EPSG:4326"``, or a WKT/PROJ
                string. Defaults to ``4326`` (lon/lat). Ignored when ``bbox`` is
                ``None``.
            densify: Points per bbox edge used when reprojecting the box onto a
                projected grid, so the envelope encloses the curved boundary
                (conservative over-cover). Defaults to ``25``.
            y_dim: Name of the ``y`` (row) dimension. Defaults to ``None`` —
                auto-detected from CF axis / ``standard_name`` / ``units``
                attributes, then well-known names (``y``/``lat``/…), then the
                trailing two dims. Pass it (with ``x_dim``) to override when the
                spatial axes can't be inferred.
            x_dim: Name of the ``x`` (column) dimension. ``None`` auto-detects as
                for ``y_dim``. Pass both ``y_dim`` and ``x_dim`` together.
            **dims: Index selector for any extra non-spatial dimension (e.g.
                ``vis_nir=0``, ``soil_layers_stag=2``). Required for every such
                dimension whose length is > 1, **including a layer dim
                interleaved between ``y`` and ``x``** (e.g. NWM ``SOIL_M`` is
                ``(time, y, soil_layers_stag, x)`` — pass ``soil_layers_stag=0``).
                A key that is not a selectable non-spatial dimension is an error.

        Returns:
            NetCDF: A georeferenced single-variable raster on the store's native CRS —
            one band per selected timestep, with the native no-data value applied.

        Raises:
            ValueError: When the store is not multidimensional; when ``variable``
                is absent or has fewer than two dimensions; when a spatial axis
                has no 1-D coordinate variable; when a non-spatial dimension of
                length > 1 is not selected, or a ``**dims`` key / index is
                invalid; or when the bbox selects no cells.

        Note:
            For a purely 2-D ``(y, x)`` variable there is no non-spatial axis, so
            ``time`` and ``**dims`` are no-ops (the whole grid, optionally bbox-
            cropped, is returned as one band).

        Examples:
            - Pull one timestep of a NWM land-surface variable over a lon/lat
              box from the public bucket (metadata-only open, windowed read)::

                >>> from pyramids.netcdf import NetCDF  # doctest: +SKIP
                >>> from pyramids.base.remote import CloudConfig  # doctest: +SKIP
                >>> url = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr"
                >>> with CloudConfig(  # doctest: +SKIP
                ...     aws_no_sign_request=True, aws_region="us-east-1"
                ... ):
                ...     nc = NetCDF.read_file(url)
                ...     ds = nc.subset("ACCET", time=0, bbox=(-78, 38, -75, 40))
                >>> ds.to_cog("accet.tif")  # doctest: +SKIP
        """
        rg = self._raster.GetRootGroup() if self._raster is not None else None
        if rg is None:
            raise ValueError(
                "subset() requires a multidimensional store; open with "
                "open_as_multi_dimensional=True."
            )
        md_arr = open_mdarray(rg, variable)
        if md_arr is None:
            raise ValueError(
                f"{variable!r} is not a variable in this store; available: "
                f"{self.variable_names}"
            )
        dim_objs = md_arr.GetDimensions()
        if len(dim_objs) < 2:
            raise ValueError(
                f"{variable!r} has {len(dim_objs)} dimension(s); subset() needs a "
                "gridded variable with at least (y, x)."
            )
        dim_names = [d.GetName() for d in dim_objs]
        dim_sizes = [int(d.GetSize()) for d in dim_objs]
        # Locate the spatial axes by CF attributes / well-known names (falling
        # back to the trailing two dims), so a variable whose layer dim is
        # interleaved between y and x — e.g. NWM SOIL_M (time, y, soil_layers, x)
        # — is windowed correctly rather than mistaking the layer dim for y.
        y_axis, x_axis = self._detect_spatial_axes(rg, dim_names, y_dim, x_dim)
        x_coords = self._read_axis_coords(rg, dim_names[x_axis], "x")
        y_coords = self._read_axis_coords(rg, dim_names[y_axis], "y")

        srs = md_arr.GetSpatialRef()
        if bbox is None:
            x_start, x_stop = 0, dim_sizes[x_axis]
            y_start, y_stop = 0, dim_sizes[y_axis]
        else:
            min_x, min_y, max_x, max_y = self._reproject_bbox_envelope(
                tuple(bbox), crs, srs, densify
            )
            x_start, x_stop = _contiguous_range(x_coords, min_x, max_x, "x", bbox)
            y_start, y_stop = _contiguous_range(y_coords, min_y, max_y, "y", bbox)

        # Build one slice per dimension; every non-spatial axis must collapse to a
        # single index (or, for the time axis, a range) so the read is bounded.
        time_axis = self._detect_time_axis(dim_names, y_axis, x_axis)
        # Every non-spatial axis is addressable by name through **dims; the time
        # axis additionally accepts the dedicated ``time=`` argument. A name given
        # in **dims always wins for its own axis.
        selectable = {
            name for axis, name in enumerate(dim_names) if axis not in (x_axis, y_axis)
        }
        unknown = set(dims) - selectable
        if unknown:
            raise ValueError(
                f"unknown dimension selector(s) {sorted(unknown)}; selectable "
                f"non-spatial dimensions are {sorted(selectable)}."
            )
        slices, ranged_axes = self._plan_band_slices(
            dim_names,
            dim_sizes,
            x_axis,
            y_axis,
            time_axis,
            (x_start, x_stop),
            (y_start, y_stop),
            time,
            dims,
        )
        arr = np.asarray(md_arr[tuple(slices)].ReadAsArray())
        # The read must keep one axis per dimension (incl. size-1 pinned ones) for
        # the y/x axis indices to stay valid; fail loudly if a future GDAL squeezes.
        self._assert_full_rank(arr, len(dim_names), variable)
        # Move the spatial axes to the trailing (y, x) positions — they may be
        # interleaved in storage (e.g. (time, y, soil_layers, x)) — then collapse
        # every remaining (non-spatial) axis onto the band axis, in dim order.
        arr = np.moveaxis(arr, (y_axis, x_axis), (-2, -1))
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        arr, geo = self._north_up_geobox(
            arr, x_coords, y_coords, (x_start, x_stop), (y_start, y_stop)
        )
        band_labels = self._band_labels(ranged_axes)

        no_data = self._md_array_no_data(md_arr)
        band_first = arr[0] if arr.shape[0] == 1 else arr
        ds = Dataset.create_from_array(
            band_first,
            geo=geo,
            epsg=4326,
            no_data_value=no_data if no_data is not None else DEFAULT_NO_DATA_VALUE,
        )
        # API-2: return a NetCDF (consistent with crop / to_crs / resample / sel) rather
        # than a bare Dataset. Wrap the just-built classic raster as a classic-backed
        # NetCDF and transfer ownership (clear ds._raster so the discarded Dataset does
        # not close the handle the NetCDF now holds); band/CRS semantics are identical.
        result = Variable(ds._raster, access="write", open_as_multi_dimensional=False)
        ds._raster = None
        # The grid mapping carries the true CRS (e.g. a sphere-datum Lambert
        # Conformal Conic with no EPSG code); prefer it over the 4326 placeholder.
        if srs is not None:
            result.crs = srs.ExportToWkt()
        if band_labels and len(band_labels) == result.band_count:
            result.band_names = band_labels
        return result

    @staticmethod
    def _plan_band_slices(
        dim_names: list[str],
        dim_sizes: list[int],
        x_axis: int,
        y_axis: int,
        time_axis: int | None,
        x_window: tuple[int, int],
        y_window: tuple[int, int],
        time: Any,
        dims: dict[str, Any],
    ) -> tuple[list[slice], list[tuple[str, range]]]:
        """Per-dimension read slices + the ranged non-spatial axes (the band axes).

        Spatial axes take the bbox windows; every other axis is pinned to one
        index (or a range) via :func:`_resolve_index_selector`. A name in ``dims``
        wins for its axis; otherwise the time axis uses ``time``.
        """
        slices: list[slice] = []
        ranged_axes: list[tuple[str, range]] = []
        for axis, (name, size) in enumerate(zip(dim_names, dim_sizes)):
            if axis == x_axis:
                slices.append(slice(*x_window))
            elif axis == y_axis:
                slices.append(slice(*y_window))
            else:
                if name in dims:
                    selector = dims[name]
                elif axis == time_axis:
                    selector = time
                else:
                    selector = None
                start, stop = _resolve_index_selector(selector, size, name)
                slices.append(slice(start, stop))
                if stop - start > 1:
                    ranged_axes.append((name, range(start, stop)))
        return slices, ranged_axes

    @staticmethod
    def _band_labels(ranged_axes: list[tuple[str, range]]) -> list[str]:
        """Cartesian-product band labels for the ranged axes (C-order, last fastest)."""
        if not ranged_axes:
            return []
        return [
            ",".join(f"{nm}={i}" for (nm, _), i in zip(ranged_axes, combo))
            for combo in itertools.product(*(idxs for _, idxs in ranged_axes))
        ]

    @staticmethod
    def _north_up_geobox(
        arr: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        x_window: tuple[int, int],
        y_window: tuple[int, int],
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
        """Flip ``arr`` to north-up / west-to-east; return ``(arr, geotransform)``.

        Cell size comes from the full coordinate spacing so a 1-cell-wide window
        still carries the store's true resolution.
        """
        x_vals = x_coords[x_window[0] : x_window[1]]
        y_vals = y_coords[y_window[0] : y_window[1]]
        if x_vals.size > 1 and x_vals[0] > x_vals[-1]:
            arr = arr[:, :, ::-1]
            x_vals = x_vals[::-1]
        if y_vals.size > 1 and y_vals[0] < y_vals[-1]:
            arr = arr[:, ::-1, :]
            y_vals = y_vals[::-1]
        d_x = abs(x_coords[1] - x_coords[0]) if x_coords.size > 1 else 1.0
        d_y = abs(y_coords[1] - y_coords[0]) if y_coords.size > 1 else d_x
        geo = (
            float(x_vals[0] - d_x / 2.0),
            float(d_x),
            0.0,
            float(y_vals[0] + d_y / 2.0),
            0.0,
            -float(d_y),
        )
        return arr, geo

    @staticmethod
    def _assert_full_rank(arr: np.ndarray, n_dims: int, variable: str) -> None:
        """Assert a windowed read kept one axis per dimension (incl. size-1 ones).

        ``subset`` locates the spatial axes by their dimension index, so the array
        returned by the windowed read must have exactly ``n_dims`` axes; a future
        GDAL that squeezed singleton axes would invalidate those indices.

        Args:
            arr: The array returned by the windowed MDArray read.
            n_dims: The variable's dimension count.
            variable: Variable name, for the error message.

        Raises:
            RuntimeError: when ``arr.ndim != n_dims``.

        Examples:
            - A matching rank is accepted (returns ``None``)::

                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._assert_full_rank(np.zeros((1, 4, 5)), 3, "soil") is None
                True

            - A squeezed read is rejected::

                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._assert_full_rank(np.zeros((4, 5)), 3, "soil")
                Traceback (most recent call last):
                    ...
                RuntimeError: windowed read of 'soil' returned 2 axes, expected 3; cannot locate the spatial axes.
        """
        if arr.ndim != n_dims:
            raise RuntimeError(
                f"windowed read of {variable!r} returned {arr.ndim} axes, expected "
                f"{n_dims}; cannot locate the spatial axes."
            )

    @staticmethod
    def _axis_role(rg: Any, name: str) -> str | None:
        """CF spatial role of a coordinate variable: ``"Y"``, ``"X"``, or ``None``.

        Inspects the coordinate array's ``axis`` / ``standard_name`` / ``units``
        attributes. Returns ``None`` when the dimension has no coordinate array
        or no recognisable spatial attribute.

        Args:
            rg: The store's multidimensional root :class:`osgeo.gdal.Group`.
            name: Dimension / coordinate name to inspect.

        Returns:
            str or None: ``"Y"`` for a latitude / northing axis, ``"X"`` for a
            longitude / easting axis, or ``None`` when the role can't be
            determined.

        Examples:
            - Inspect a latitude coordinate of an opened multidim store::

                >>> nc._axis_role(nc._raster.GetRootGroup(), "lat")  # doctest: +SKIP
                'Y'
        """
        coord = open_mdarray(rg, name)
        if coord is None:
            return None
        # Attribute-only detection (empty ``name`` disables the name-pattern fallback): the
        # name-based stage lives in ``_named_spatial_axes`` and must stay separate so the
        # CF-attribute pass keeps precedence over it.
        role = detect_axis("", _read_attributes(coord))
        return role if role in ("X", "Y") else None

    @staticmethod
    def _detect_spatial_axes(
        rg: Any,
        dim_names: list[str],
        y_dim: str | None,
        x_dim: str | None,
    ) -> tuple[int, int]:
        """Resolve the ``(y_axis, x_axis)`` indices of a gridded variable.

        Resolution order: an explicit ``y_dim`` / ``x_dim`` override → CF axis /
        ``standard_name`` / ``units`` attributes on the coordinate variables →
        well-known dimension names (``y``/``lat``/…, ``x``/``lon``/…) → the two
        trailing dimensions (the common ``(…, y, x)`` layout).

        Args:
            rg: The multidimensional root group (``None`` skips CF-attribute
                detection — used by the pure name/trailing fallbacks).
            dim_names: The variable's dimension names, in storage order.
            y_dim: Optional explicit name of the ``y`` dimension.
            x_dim: Optional explicit name of the ``x`` dimension.

        Returns:
            tuple[int, int]: The ``(y_axis, x_axis)`` indices into ``dim_names``.

        Raises:
            ValueError: if only one of ``y_dim`` / ``x_dim`` is given, the two are
                equal, or a named override is not a dimension of the variable.

        Examples:
            - A layer dim interleaved between ``y`` and ``x`` is still resolved
              by name (``rg=None`` skips the CF-attribute step)::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._detect_spatial_axes(None, ["time", "y", "level", "x"], None, None)
                (1, 3)

            - An explicit override wins regardless of position::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._detect_spatial_axes(None, ["time", "y", "x"], "y", "x")
                (1, 2)
        """
        override = NetCDF._override_spatial_axes(dim_names, y_dim, x_dim)
        if override is not None:
            return override
        return (
            NetCDF._cf_spatial_axes(rg, dim_names)
            or NetCDF._named_spatial_axes(dim_names)
            or (len(dim_names) - 2, len(dim_names) - 1)
        )

    @staticmethod
    def _override_spatial_axes(
        dim_names: list[str], y_dim: str | None, x_dim: str | None
    ) -> tuple[int, int] | None:
        """Validate and apply an explicit ``y_dim`` / ``x_dim`` override.

        Args:
            dim_names: The variable's dimension names, in storage order.
            y_dim: Explicit ``y`` dimension name, or ``None``.
            x_dim: Explicit ``x`` dimension name, or ``None``.

        Returns:
            tuple[int, int] or None: The ``(y_axis, x_axis)`` indices when both
            names are given; ``None`` when neither is (so auto-detection
            proceeds).

        Raises:
            ValueError: when exactly one name is given, the two are equal, or a
                name is not a dimension of the variable.

        Examples:
            - Both names resolve to their indices::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._override_spatial_axes(["time", "y", "x"], "y", "x")
                (1, 2)

            - Neither name defers to auto-detection (returns ``None``)::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._override_spatial_axes(["y", "x"], None, None) is None
                True
        """
        if y_dim is None and x_dim is None:
            return None
        if (y_dim is None) != (x_dim is None):
            raise ValueError("pass both y_dim and x_dim, or neither.")
        if y_dim == x_dim:
            raise ValueError(f"y_dim and x_dim must differ; both are {y_dim!r}.")
        for value, label in ((y_dim, "y_dim"), (x_dim, "x_dim")):
            if value not in dim_names:
                raise ValueError(
                    f"{label}={value!r} is not a dimension of this variable; "
                    f"available: {dim_names}"
                )
        return dim_names.index(y_dim), dim_names.index(x_dim)

    @staticmethod
    def _cf_spatial_axes(rg: Any, dim_names: list[str]) -> tuple[int, int] | None:
        """``(y_axis, x_axis)`` from CF coordinate attributes, or ``None``.

        Returns ``None`` when there is no root group or the coordinate variables
        don't carry recognisable spatial attributes for both axes.

        Args:
            rg: The multidimensional root group, or ``None``.
            dim_names: The variable's dimension names, in storage order.

        Returns:
            tuple[int, int] or None: The ``(y_axis, x_axis)`` indices when both
            spatial axes are identified from coordinate attributes, else ``None``.

        Examples:
            - Without a root group there are no attributes to read::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._cf_spatial_axes(None, ["time", "y", "x"]) is None
                True
        """
        if rg is None:
            return None
        y_idx = x_idx = None
        for axis, name in enumerate(dim_names):
            role = NetCDF._axis_role(rg, name)
            if role == "Y" and y_idx is None:
                y_idx = axis
            elif role == "X" and x_idx is None:
                x_idx = axis
            if y_idx is not None and x_idx is not None:
                return y_idx, x_idx
        return None

    @staticmethod
    def _named_spatial_axes(dim_names: list[str]) -> tuple[int, int] | None:
        """``(y_axis, x_axis)`` from well-known dimension names, or ``None``.

        Matches each dimension name case-insensitively against
        :data:`_Y_DIM_NAMES` / :data:`_X_DIM_NAMES`.

        Args:
            dim_names: The variable's dimension names, in storage order.

        Returns:
            tuple[int, int] or None: The ``(y_axis, x_axis)`` indices when both a
            y-like and an x-like name are present, else ``None``.

        Examples:
            - Recognised names resolve even when interleaved::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._named_spatial_axes(["time", "y", "soil", "x"])
                (1, 3)

            - Unrecognised names return ``None``::

                >>> from pyramids.netcdf import NetCDF
                >>> NetCDF._named_spatial_axes(["time", "a", "b"]) is None
                True
        """
        lowered = [n.lower() for n in dim_names]
        y_named = next((i for i, n in enumerate(lowered) if n in _Y_DIM_NAMES), None)
        x_named = next((i for i, n in enumerate(lowered) if n in _X_DIM_NAMES), None)
        if y_named is not None and x_named is not None:
            return y_named, x_named
        return None

    @staticmethod
    def _read_axis_coords(rg: Any, name: str, axis_label: str) -> np.ndarray:
        """Read a spatial axis's 1-D coordinate values, or raise a clear error.

        A gridded variable can name a spatial dimension that has no indexing
        (coordinate) variable — e.g. WRF ``south_north`` / ``west_east``.
        ``OpenMDArray`` then returns ``None`` (or raises), so guard it with an
        actionable message instead of an opaque ``AttributeError``.

        Args:
            rg: The store's root :class:`osgeo.gdal.Group`.
            name: Coordinate (indexing) array name — the spatial dimension name.
            axis_label: ``"x"`` / ``"y"``, used in the error message.

        Returns:
            The coordinate values as a 1-D ``float64`` :class:`numpy.ndarray`.

        Raises:
            ValueError: When the dimension has no readable 1-D coordinate array.
        """
        coord = open_mdarray(rg, name)
        values = None if coord is None else coord.ReadAsArray()
        if values is None:
            raise ValueError(
                f"the {axis_label} dimension {name!r} has no 1-D coordinate "
                "variable; subset() needs x/y coordinates to window and "
                "georeference the grid."
            )
        return np.asarray(values, dtype="float64")

    @staticmethod
    def _detect_time_axis(dim_names: list[str], y_axis: int, x_axis: int) -> int | None:
        """Index of the time dimension, or the first non-spatial axis as fallback.

        The ``time=`` selector applies to this axis; any other non-spatial axis
        is selected through ``**dims`` by name.

        Args:
            dim_names: Dimension names in storage order.
            y_axis: Index of the ``y`` (row) axis to exclude.
            x_axis: Index of the ``x`` (column) axis to exclude.

        Returns:
            The time axis index, the first non-spatial axis if no dimension is
            time-like, or ``None`` when the variable is purely ``(y, x)``.

        Examples:
            - A dimension named ``"time"`` is the time axis:
                ```python
                >>> NetCDF._detect_time_axis(["time", "y", "x"], 1, 2)
                0

                ```
            - With no time-like name, the first non-spatial axis is used:
                ```python
                >>> NetCDF._detect_time_axis(["level", "member", "y", "x"], 2, 3)
                0

                ```
            - A purely 2-D ``(y, x)`` variable has no time axis:
                ```python
                >>> NetCDF._detect_time_axis(["y", "x"], 0, 1) is None
                True

                ```
        """
        time_like = {"time", "valid_time", "forecast_reference_time", "t"}
        fallback = None
        result = None
        for axis, name in enumerate(dim_names):
            if axis in (y_axis, x_axis):
                continue
            if fallback is None:
                fallback = axis
            if name.lower() in time_like:
                result = axis
                break
        return result if result is not None else fallback

    @staticmethod
    def _md_array_no_data(md_arr: Any) -> float | None:
        """Native no-data for an MDArray — driver value first, then CF attrs.

        Args:
            md_arr: The :class:`osgeo.gdal.MDArray` to inspect.

        Returns:
            The no-data value as a ``float``: the driver's value if set, else the
            ``missing_value`` then ``_FillValue`` CF attribute, else ``None``.
        """
        result = md_arr.GetNoDataValueAsDouble()
        if result is None:
            for attr_name in ("missing_value", "_FillValue"):
                try:
                    attr = md_arr.GetAttribute(attr_name)
                except RuntimeError:
                    attr = None
                if attr is not None:
                    result = float(np.asarray(attr.ReadAsDoubleArray()).ravel()[0])
                    break
        return result

    @staticmethod
    def _reproject_bbox_envelope(
        bbox: tuple[float, float, float, float],
        src_crs: int | str,
        dst_srs: Any,
        densify: int,
    ) -> tuple[float, float, float, float]:
        """Reproject a bbox into the store's native CRS and return its envelope.

        Densifies each edge before transforming so the projected envelope
        encloses the curved boundary of a lon/lat box on a projected grid
        (conservative over-cover). When the destination has no CRS, or matches
        the source, the bbox is treated as already native.

        Args:
            bbox: ``(min_x, min_y, max_x, max_y)`` in the source CRS.
            src_crs: Source CRS — an EPSG ``int`` or a WKT/PROJ/``"EPSG:n"`` string.
            dst_srs: Destination :class:`osgeo.osr.SpatialReference`, or ``None``
                when the store has no grid mapping.
            densify: Number of points sampled along each bbox edge (``>= 2``).

        Returns:
            The ``(min_x, min_y, max_x, max_y)`` envelope in the destination CRS.

        Examples:
            - With no destination CRS the bbox is returned unchanged (treated as
              already native):
                ```python
                >>> NetCDF._reproject_bbox_envelope((-1.0, -2.0, 3.0, 4.0), 4326, None, 5)
                (-1.0, -2.0, 3.0, 4.0)

                ```
        """
        min_x, min_y, max_x, max_y = (float(v) for v in bbox)
        src = osr.SpatialReference()
        src.SetFromUserInput(
            f"EPSG:{src_crs}" if isinstance(src_crs, int) else str(src_crs)
        )
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if dst_srs is None or src.IsSame(dst_srs):
            return min_x, min_y, max_x, max_y
        dst = dst_srs.Clone()
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(src, dst)
        edge = np.linspace(0.0, 1.0, max(int(densify), 2))
        xs = np.concatenate(
            [
                min_x + edge * (max_x - min_x),
                min_x + edge * (max_x - min_x),
                np.full_like(edge, min_x),
                np.full_like(edge, max_x),
            ]
        )
        ys = np.concatenate(
            [
                np.full_like(edge, min_y),
                np.full_like(edge, max_y),
                min_y + edge * (max_y - min_y),
                min_y + edge * (max_y - min_y),
            ]
        )
        proj_x: list[float] = []
        proj_y: list[float] = []
        for px, py in zip(xs, ys):
            tx, ty, _ = transform.TransformPoint(float(px), float(py))
            proj_x.append(tx)
            proj_y.append(ty)
        return min(proj_x), min(proj_y), max(proj_x), max(proj_y)

    @classmethod
    def from_xarray(
        cls,
        dataset: Any,
        path: str | Path | None = None,
    ) -> NetCDF:
        """Facade — delegates to :func:`pyramids.netcdf.engines.interop.from_xarray`.

        See that function for the full contract. ``from_xarray`` builds a new
        container (it does not operate on an existing instance), so its body
        lives in the module-level engine function rather than on the
        instance-bound :class:`~pyramids.netcdf.engines.interop.Interop` engine;
        ``cls`` is threaded through so the file is read back as the concrete
        ``Container`` subtype.
        """
        return _interop.from_xarray(cls, dataset, path)


class Variable(NetCDF):
    """A single variable extracted from a NetCDF container — always a valid raster.

    Returned by :meth:`NetCDF.get_variable`, :meth:`NetCDF.subset`, :meth:`NetCDF.sel`,
    and the spatial operations (``crop`` / ``to_crs`` / ``resample``) when called on a
    variable subset. A ``Variable`` *is* a raster (``band_count >= 1``), so every
    operation inherited from :class:`~pyramids.dataset.Dataset` — ``read_array``,
    ``crop``, ``plot``, ``sel`` … — is meaningful on it, unlike a
    :class:`Container`.

    Part of the API-1 container/variable identity split (issue #614): ``get_variable``
    now returns this concrete type so callers can tell a variable from a container by
    type (``isinstance(x, Variable)``) rather than by runtime flags. ``NetCDF``
    remains the (deprecated) base alias for one major version, so existing
    ``isinstance(x, NetCDF)`` checks keep working.

    Examples:
        - Extract a variable and read its data (it behaves as a raster):
            ```python
            >>> from pyramids.netcdf import NetCDF  # doctest: +SKIP
            >>> nc = NetCDF.read_file("cube.nc")  # doctest: +SKIP
            >>> var = nc.get_variable("temperature")  # doctest: +SKIP
            >>> var.read_array().shape  # doctest: +SKIP
            (12, 180, 360)

            ```
        - Select along a band dimension — the result is another Variable:
            ```python
            >>> first = var.sel(time=0)  # doctest: +SKIP
            >>> first.band_count  # doctest: +SKIP
            1

            ```
    """

    def _check_not_container(self, operation: str) -> None:
        """No-op: a variable is always a valid raster, so spatial ops are always allowed.

        The base guard only fires for a root MDIM container; a ``Variable`` can never
        be one, so the check is unnecessary here. Overriding it makes the type contract
        explicit and keeps the guard out of the hot path for variable operations.
        """
        return None


class Container(NetCDF):
    """A NetCDF container (the root MDIM group) holding variables, dimensions, attributes.

    Returned by :meth:`NetCDF.read_file`, :meth:`NetCDF.create_from_array`, and
    :meth:`NetCDF.get_group`. A container is **not** a single raster (``band_count == 0``);
    the raster-level operations inherited from :class:`~pyramids.dataset.Dataset`
    (``read_array``, a band-level ``crop`` on the container itself, …) are not meaningful
    here — extract a variable first with ``nc.get_variable('name')``. Container-level
    ``crop`` / ``to_crs`` / ``resample`` still work by fanning out over every variable.

    Part of the API-1 container/variable identity split (issue #614). ``NetCDF`` remains
    the (deprecated) base alias for one major version, so existing ``isinstance(x, NetCDF)``
    checks keep working.

    Examples:
        - Open a store and list its variables, then drill into one:
            ```python
            >>> from pyramids.netcdf import NetCDF  # doctest: +SKIP
            >>> nc = NetCDF.read_file("cube.nc")  # doctest: +SKIP
            >>> nc.variable_names  # doctest: +SKIP
            ['temperature', 'precipitation']
            >>> nc.get_variable("temperature").band_count  # doctest: +SKIP
            12

            ```
    """
