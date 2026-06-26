"""Spatial / dimensional selection engine for :class:`pyramids.netcdf.NetCDF`.

Owns the crop / sel / subset / reduce family extracted from the
``netcdf.py`` god-object (issue #615, STR-1). Per the agreed design the
NetCDF-specific cropping (``crop`` and its curvilinear path) is folded in
here rather than living in a separate spatial engine.

The public ``NetCDF`` methods are thin façades delegating to this engine;
signatures, behaviour, and return types are unchanged. Cropping reaches
the container's own plumbing through the weakref-proxied back-reference
``self._ds`` — including ``self._ds.spatial.crop`` for the base affine
crop (equivalent to the ``super().crop`` call the override used) and the
shared helpers ``_apply_to_all_variables`` / ``_preserve_netcdf_metadata``
/ ``_bbox_geotransform`` which stay on ``NetCDF`` because the not-yet-moved
methods still use them.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely import contains_xy

from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.dataset.engines._base import _Engine
from pyramids.feature import FeatureCollection
from pyramids.netcdf._mdim import open_mdarray
from pyramids.netcdf._plot import NetCDFPlot

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Selection(_Engine):
    """Spatial / dimensional selection collaborator for :class:`NetCDF`.

    Owns the bodies of :meth:`crop` (with the curvilinear and rectilinear
    helpers folded in), :meth:`sel` (band selection by coordinate value),
    :meth:`subset` (windowed ``(variable, time, bbox)`` read), and
    :meth:`reduce` (collapse / coarsen a non-spatial dimension). ``NetCDF``
    wires one instance per container as ``nc.selection`` and exposes thin
    façades, so ``nc.crop(...)`` and ``nc.selection.crop(...)`` are equivalent.

    Each method reaches the container through the weakref-proxied
    back-reference :attr:`_ds` inherited from
    :class:`~pyramids.dataset.engines._base._Engine`: the base affine crop via
    ``nc.spatial.crop`` (what the override reached with ``super().crop``), and
    the shared helpers (``_apply_to_all_variables`` /
    ``_preserve_netcdf_metadata`` / the subset axis helpers / the reduce
    helpers) which stay on ``NetCDF``.
    """

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
        nc = self._ds
        mask = self._resolve_crop_mask(mask, bbox, epsg)
        if nc._is_md_array and not nc._is_subset and nc.band_count == 0:
            # A container crops every variable; `chunks` is a curvilinear-only, per-variable knob
            # (a container may mix curvilinear and rectilinear variables), so the per-variable fan-out
            # cannot honour it. Reject it explicitly rather than silently reading eagerly.
            if chunks is not None:
                raise ValueError(
                    "crop(chunks=…) is not supported on a root container; it is a "
                    "curvilinear-only, per-variable option — call crop on a single variable "
                    "via get_variable(name) instead."
                )
            result = nc._apply_to_all_variables("crop", {"mask": mask, "touch": touch})
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
            crs = epsg if epsg is not None else self._ds.epsg
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
        nc = self._ds
        curv = NetCDFPlot(nc)._resolve_curvilinear_coords(nc, coords=None)
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
            # Stamp the variable's known CRS onto its backing raster before the cutline warp.
            # A NetCDF variable tracks its EPSG even when the raster (the AsClassicDataset MDArray
            # view, or a wrap_longitude/materialized MEM raster) carries no projection string;
            # without it GDAL's cutline warp warns ("the input vector layer has a SRS, but the
            # source raster dataset does not") and — for a cutline in a different CRS — would clip
            # the wrong region. The driver-less MDArray view does not persist SetProjection, so
            # materialize it first (the affine crop reads it fully anyway). See issue #629.
            if nc.epsg and nc._raster is not None and not nc._raster.GetProjection():
                wkt = sr_from_epsg(int(nc.epsg)).ExportToWkt()
                nc._raster.SetProjection(wkt)
                if not nc._raster.GetProjection():
                    nc._materialize_md_view()
                    nc._raster.SetProjection(wkt)
            # `nc.spatial.crop` is the base Dataset affine crop — exactly what the
            # NetCDF.crop override reached via `super().crop(...)`, bypassing this engine.
            result = nc._preserve_netcdf_metadata(
                nc.spatial.crop(mask=mask, touch=touch)
            )
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
        nc = self._ds
        lon2d = np.asarray(coords2d[0], dtype=float)
        lat2d = np.asarray(coords2d[1], dtype=float)

        geometry = _reconcile_mask_to_crs(mask, nc.epsg)
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

        nd = _window_no_data(nc)
        # Read only the bounding window, not the whole variable. The polygon mask and window were
        # derived from the (spatial-footprint) coordinate arrays alone — no data was materialised yet.
        data_win = _read_curvilinear_window(nc, r0, r1, c0, c1, chunks)
        data_win[..., ~inside[r0:r1, c0:c1]] = nd
        lon_win = lon2d[r0:r1, c0:c1]
        lat_win = lat2d[r0:r1, c0:c1]

        var_name = getattr(nc, "_source_var_name", None) or "data"
        container = nc.create_from_array(
            data_win,
            geo=nc._bbox_geotransform(lon_win, lat_win),
            epsg=nc.epsg or 4326,
            no_data_value=nd,
            variable_name=var_name,
        )
        # create_from_array returns a root container; hand back the variable subset, carrying the
        # windowed 2-D coordinates so the result stays curvilinear (plots on its real geometry).
        result = container.get_variable(var_name)
        result._curvilinear_coords = (lon_win, lat_win)
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
        nc = self._ds
        if len(kwargs) != 1:
            raise ValueError("sel() requires exactly one keyword argument.")

        dim_name, selector = next(iter(kwargs.items()))

        if not nc._band_dim_names:
            raise ValueError(
                "sel() requires a variable with at least one non-spatial "
                "dimension. This variable has no band dimensions tracked."
            )
        if dim_name not in nc._band_dim_names:
            raise ValueError(
                f"Dimension {dim_name!r} does not match any band dimension "
                f"of this variable {list(nc._band_dim_names)!r}."
            )

        coords = nc._band_dim_values_map.get(dim_name)
        if coords is None:
            raise ValueError(
                f"No coordinate values available for dimension {dim_name!r}."
            )

        dim_indices = _resolve_dim_indices(coords, selector)
        if not dim_indices:
            raise ValueError(
                f"No bands match {dim_name}={selector}. Available values: {coords}"
            )

        dim_axis = nc._band_dim_names.index(dim_name)
        sizes = nc._band_dim_sizes
        band_indices = _map_dim_to_band_indices(dim_axis, sizes, dim_indices)
        selected_coords = [coords[i] for i in dim_indices]
        selected = _read_selected_bands(nc, band_indices)

        ndv = nc.no_data_value
        ndv_scalar = ndv[0] if isinstance(ndv, list) and ndv else ndv
        ds_result = Dataset.create_from_array(
            selected,
            geo=nc.geotransform,
            epsg=nc.epsg,
            no_data_value=ndv_scalar,
        )
        result = nc._preserve_netcdf_metadata(ds_result)
        new_sizes = tuple(
            len(dim_indices) if i == dim_axis else s for i, s in enumerate(sizes)
        )
        result._band_dim_sizes = new_sizes
        result._band_dim_values_map = dict(nc._band_dim_values_map)
        result._band_dim_values_map[dim_name] = selected_coords
        # Re-derive the legacy primary-dim view from the (now updated) canonical
        # fields so it tracks the pinned selection — single source of truth in
        # `_derive_primary_band_view`.
        result._band_dim_name, result._band_dim_values = nc._derive_primary_band_view(
            result._band_dim_names,
            result._band_dim_values_map,
            result._band_dim_sizes,
            result._band_count,
        )

        return result

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
        # Local import breaks the netcdf.py <-> engines.selection import cycle
        # (netcdf.py imports this module at top level for wiring): Variable is the
        # concrete subset subtype, _contiguous_range a module-level helper there.
        from pyramids.netcdf.netcdf import Variable, _contiguous_range

        nc = self._ds
        rg = nc._working_group()
        if rg is None:
            raise ValueError(
                "subset() requires a multidimensional store; open with "
                "open_as_multi_dimensional=True."
            )
        md_arr = open_mdarray(rg, variable)
        if md_arr is None:
            raise ValueError(
                f"{variable!r} is not a variable in this store; available: "
                f"{nc.variable_names}"
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
        y_axis, x_axis = nc._detect_spatial_axes(rg, dim_names, y_dim, x_dim)
        x_coords = nc._read_axis_coords(rg, dim_names[x_axis], "x")
        y_coords = nc._read_axis_coords(rg, dim_names[y_axis], "y")

        srs = md_arr.GetSpatialRef()
        if bbox is None:
            x_start, x_stop = 0, dim_sizes[x_axis]
            y_start, y_stop = 0, dim_sizes[y_axis]
        else:
            min_x, min_y, max_x, max_y = nc._reproject_bbox_envelope(
                tuple(bbox), crs, srs, densify
            )
            x_start, x_stop = _contiguous_range(x_coords, min_x, max_x, "x", bbox)
            y_start, y_stop = _contiguous_range(y_coords, min_y, max_y, "y", bbox)

        # Build one slice per dimension; every non-spatial axis must collapse to a
        # single index (or, for the time axis, a range) so the read is bounded.
        time_axis = nc._detect_time_axis(dim_names, y_axis, x_axis)
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
        slices, ranged_axes = nc._plan_band_slices(
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
        nc._assert_full_rank(arr, len(dim_names), variable)
        # Move the spatial axes to the trailing (y, x) positions — they may be
        # interleaved in storage (e.g. (time, y, soil_layers, x)) — then collapse
        # every remaining (non-spatial) axis onto the band axis, in dim order.
        arr = np.moveaxis(arr, (y_axis, x_axis), (-2, -1))
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        arr, geo = nc._north_up_geobox(
            arr, x_coords, y_coords, (x_start, x_stop), (y_start, y_stop)
        )
        band_labels = nc._band_labels(ranged_axes)

        no_data = nc._md_array_no_data(md_arr)
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
        # Local import breaks the netcdf.py <-> engines.selection import cycle
        # (netcdf.py imports this module at top level for wiring); _REDUCERS is a
        # module-level reducer registry there, shared with the reduce helpers.
        from pyramids.netcdf.netcdf import _REDUCERS

        nc = self._ds
        if how not in _REDUCERS:
            raise ValueError(f"how must be one of {sorted(_REDUCERS)}; got {how!r}")
        names = nc.variable_names
        if not names:
            raise ValueError("Cannot reduce an empty container (no data variables).")

        group_positions = nc._resolve_group_positions(dim, groupby)

        # Reduce only the gridded variables; non-spatial auxiliaries (no y/x axes)
        # can't go through the raster reduce path, so they are carried through
        # unchanged below — the same split crop / to_crs use (#513). Resolve the root
        # group once and reuse it for the spanning-aux probe further down.
        rg = nc._working_group()
        spatial_vars = nc._spatial_variable_names(rg)
        aux_vars = [n for n in names if n not in spatial_vars]

        result = None
        found = False
        for var_name in spatial_vars:
            var = nc.get_variable(var_name)
            arr = nc._materialize_variable_array(var)
            band_names = list(var._band_dim_names)
            values_map = dict(var._band_dim_values_map)
            ndv = nc._scalar_no_data_value(var.no_data_value)

            if dim in band_names:
                found = True
                axis = band_names.index(dim)
                arr, band_names, values_map = nc._reduce_variable_array(
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

            result = nc._stack_reduced_variable(
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
            var_dims = nc._variable_dim_names(rg, name)
            (spanning_aux if dim in var_dims else carry_aux).append(name)
        if spanning_aux:
            warnings.warn(
                f"reduce() dropped auxiliary variable(s) {spanning_aux} that span "
                f"the reduced dimension {dim!r}; carrying them unchanged would "
                f"leave an inconsistent {dim!r} length in the result.",
                # stacklevel=3 (not 2): the user calls NetCDF.reduce, which forwards
                # through the one-line façade to this engine method, so the user's
                # call site is three frames up — keeping the original warning location.
                stacklevel=3,
            )
        nc._carry_aux_variables(result, carry_aux, "reduce")
        return result


def _reconcile_mask_to_crs(mask: FeatureCollection, epsg: int | None) -> Any:
    """Reproject a polygon mask to the variable's CRS and return its unioned geometry.

    Reprojects only when the mask carries a CRS that differs from ``epsg``;
    otherwise the mask is used as-is. Helper of
    :meth:`Selection._crop_curvilinear`.
    """
    gdf = mask
    gdf_crs = getattr(gdf, "crs", None)
    if gdf_crs is not None and epsg:
        src_epsg = gdf_crs.to_epsg()
        if src_epsg is not None and src_epsg != epsg:
            gdf = gdf.to_crs(epsg=epsg)
    return gdf.geometry.union_all()


def _window_no_data(nc: NetCDF) -> Any:
    """Return the scalar no-data value to stamp on out-of-polygon cells.

    Collapses a per-band ``no_data_value`` sequence to its first entry and
    falls back to :data:`DEFAULT_NO_DATA_VALUE` when unset.
    """
    nd = nc.no_data_value
    nd = nd[0] if isinstance(nd, (list, tuple)) else nd
    return DEFAULT_NO_DATA_VALUE if nd is None else nd


def _read_curvilinear_window(
    nc: NetCDF, r0: int, r1: int, c0: int, c1: int, chunks: Any
) -> np.ndarray:
    """Read just the ``(r0:r1, c0:c1)`` bounding window of a curvilinear variable.

    With ``chunks`` the read goes through the dask-backed lazy path (only the
    overlapping chunks materialise) and the native ``(d0, …, rows, cols)`` shape
    is flattened to ``(bands, rows, cols)``; otherwise GDAL reads just the
    ``(c0, r0)``–``(c1, r1)`` block eagerly. Helper of
    :meth:`Selection._crop_curvilinear`.
    """
    if chunks is not None:
        lazy = nc.read_array(chunks=chunks)
        if lazy.ndim > 2:
            lazy = lazy.reshape(-1, *lazy.shape[-2:])
        return np.array(lazy[..., r0:r1, c0:c1].compute(), copy=True)
    return np.array(nc.read_array(window=[c0, r0, c1 - c0, r1 - r0]), copy=True)


def _resolve_dim_indices(coords: list, selector: Any) -> list[int]:
    """Resolve a `sel` selector to the matching indices along a band dimension.

    Supports a ``slice`` (direction-agnostic inclusive bounds — works on both
    ascending and descending coord axes), a ``list`` of exact values, or a
    single exact value. Helper of :meth:`Selection.sel`.
    """
    if isinstance(selector, slice):
        start = selector.start if selector.start is not None else coords[0]
        stop = selector.stop if selector.stop is not None else coords[-1]
        # Normalise bounds so the match works on both ascending and descending
        # coord axes (e.g. `latitude = [44, 43, 42, 41, 40]` from CDS-Beta): a
        # `slice(None, None)` on a descending axis would otherwise test
        # `44 <= v <= 40` and match nothing instead of "select everything".
        lo, hi = (start, stop) if start <= stop else (stop, start)
        return [i for i, v in enumerate(coords) if lo <= v <= hi]
    if isinstance(selector, list):
        coord_set = set(selector)
        return [i for i, v in enumerate(coords) if v in coord_set]
    return [i for i, v in enumerate(coords) if v == selector]


def _map_dim_to_band_indices(
    dim_axis: int, sizes: tuple[int, ...], dim_indices: list[int]
) -> list[int]:
    """Map pinned indices along one band dim to flat classic-band indices.

    GDAL flattens ``(d_0, …, d_{n-1}, lat, lon)`` row-major over the non-spatial
    dims (last varies fastest). For a band dim at axis ``k`` with ``sizes`` S,
    ``stride = prod(S[k+1:])`` and ``block = stride * S[k]``; each pinned index
    ``p`` emits ``[outer + p*stride .. outer + (p+1)*stride)`` for every
    ``outer`` in ``range(0, total, block)``. Reduces to the identity when there
    is a single band dim. Helper of :meth:`Selection.sel`.
    """
    stride = math.prod(sizes[dim_axis + 1 :])
    block = stride * sizes[dim_axis]
    total = math.prod(sizes)
    band_indices: list[int] = []
    for pinned in dim_indices:
        for outer_start in range(0, total, block):
            base = outer_start + pinned * stride
            band_indices.extend(range(base, base + stride))
    return band_indices


def _read_selected_bands(nc: NetCDF, band_indices: list[int]) -> np.ndarray:
    """Read just the selected classic bands into one pre-allocated buffer.

    Mirrors the all-bands read path in ``IO.read_array`` rather than stacking N
    separate ``read_array`` results. Each 0-based band index maps to a 1-based
    GDAL band in the classic view built by ``get_variable``; reading only the
    selected bands avoids materialising the whole variable. Helper of
    :meth:`Selection.sel`.
    """
    if len(band_indices) == 1:
        return nc._iloc(band_indices[0]).ReadAsArray()
    selected = np.empty(
        (len(band_indices), nc.rows, nc.columns), dtype=nc.numpy_dtype[0]
    )
    for out_i, band_index in enumerate(band_indices):
        selected[out_i, :, :] = nc._iloc(band_index).ReadAsArray()
    return selected
