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
import operator
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, contains_xy

from pyramids.base._axes import X_AXIS_NAMES, Y_AXIS_NAMES
from pyramids.base._errors import AlignmentError
from pyramids.base._utils import carry_band_packing
from pyramids.base.crs import crs_equal, crs_spec, sr_from_epsg, sr_from_user_input
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines.spatial import (
    _crop_seam_halves,
    _require_antimeridian_seam,
    _split_lon_bbox,
    _stitch_lon_halves,
)
from pyramids.dataset.transform import GeoTransform
from pyramids.feature import FeatureCollection
from pyramids.netcdf._label_select import (
    FULL_FORMAT,
    first_label,
    has_label,
    label_format,
    label_indices,
    nearest_indices,
    non_label_parts,
    probe_format,
    summarise_values,
)
from pyramids.netcdf._mdim import copy_band_values_map, open_mdarray, scalar_no_data
from pyramids.netcdf._plot import NetCDFPlot
from pyramids.netcdf.array_options import GeoReference
from pyramids.netcdf.engines._along_dim import (
    _apply_per_variable,
    _apply_to_container,
    _apply_to_variable,
    _assert_band_dimension,
    _CumProd,
    _CumSum,
    _Diff,
    _DropNa,
    _Extremum,
    _Interpolate,
    _InterpTo,
    _Pad,
    _Push,
    _Rank,
    _read_no_data,
    _reduces_as_a_variable,
    _Reduction,
    _Rolling,
    _Shift,
)
from pyramids.netcdf.engines._weighted import _WEIGHTED_HOWS, _weighted_result

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF

# The window must cover strictly less than 1/N of the variable's cells before reading it through
# the MDArray earns a second code path: the windowed read pays off by skipping most of the
# variable, and at parity it is the same work plus an extra copy.
#
# Deliberately relative only, with no absolute floor on the variable's size. A floor would spare a
# small local grid a shortcut that gains it nothing — but it gains nothing there either way, the
# two paths are asserted to agree cell for cell, and a floor high enough to matter (thousands of
# cells) would take every test fixture below it and quietly stop exercising this path at all.
_MIN_WINDOW_SAVING = 2


class Selection(_Engine["NetCDF"]):
    """Spatial / dimensional selection collaborator for :class:`NetCDF`.

    Owns the bodies of :meth:`crop` (with the curvilinear and rectilinear
    helpers folded in), :meth:`sel` (band selection by coordinate value),
    :meth:`isel` (the same cut by position), :meth:`subset` (windowed
    `(variable, time, bbox)` read), :meth:`reduce` (collapse or group a
    non-spatial dimension), :meth:`coarsen` (fixed-size windows along one),
    :meth:`rolling` (a moving window that keeps the dimension's length),
    :meth:`diff`, :meth:`cumsum`, :meth:`shift`, the four extremum locators
    :meth:`argmin` / :meth:`argmax` / :meth:`idxmin` / :meth:`idxmax`, and
    :meth:`weighted` (the one member that can reduce the spatial axes).
    `NetCDF` wires one instance per container as
    `nc.selection` and exposes thin façades, so `nc.crop(...)` and
    `nc.selection.crop(...)` are equivalent.

    Each method reaches the container through the weakref-proxied
    back-reference :attr:`_ds` inherited from
    :class:`~pyramids.dataset.engines._base._Engine`: the base affine crop via
    `nc.spatial.crop` (what the override reached with `super().crop`), and
    the shared helpers (`_apply_to_all_variables` /
    `_preserve_netcdf_metadata` / the subset axis helpers / the array-level
    reduce helpers) which stay on `NetCDF`.
    """

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
        chunks: Any = None,
        path: str | Path | None = None,
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
                variable. Mutually exclusive with ``mask``. A *geographic*
                bbox with ``west > east`` (the STAC antimeridian
                convention, e.g. ``(170, -10, -170, 10)``) crosses the
                180° meridian: on a rectilinear variable it is split at
                the 180°/360° seam and stitched into one contiguous strip;
                on a curvilinear variable the split halves become a
                polygon mask over the 2-D coordinates; on a root container
                it fans out to every variable. Behaviour change: a
                *geographic* ``west > east`` bbox is read as the STAC
                antimeridian convention (rather than raising
                ``west < east``) — but only when the dataset's longitude
                extent reaches the 180 seam. On a *regional* grid that
                does not reach the seam it raises a clear error instead,
                catching a transposed / typo'd bbox. A *projected*
                ``west > east`` bbox is still validated and raises.
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
            path (keyword-only): Optional output ``.nc`` path. On a **root
                container** the cropped cube is streamed straight to that
                file one leading-dimension slab at a time (so the whole
                result is never resident) and a **file-backed** :class:`NetCDF`
                reading it is returned; ``None`` (default) builds the result
                in memory.

        Returns:
            NetCDF: Cropped container or variable subset.

        Note:
            The result sits on a **new grid**, so its spatial axes come back named
            `y` / `x` rather than the source's: a cube on `latitude` / `longitude`
            reports `['time', 'x', 'y']` after a crop. See
            :meth:`pyramids.netcdf.NetCDF.to_crs` for why, and for the members that keep
            the source's names.

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
                ...     "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"
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
                ...     "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"
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
        is_container = nc._is_root_container
        antimeridian = self._try_antimeridian(
            bbox, mask, epsg, is_container, touch, chunks
        )
        if antimeridian is not None:
            return cast("NetCDF", antimeridian._persist_to(path))
        mask = self._resolve_crop_mask(mask, bbox, epsg)
        if is_container:
            # A container crops every variable; `chunks` is a curvilinear-only, per-variable knob
            # (a container may mix curvilinear and rectilinear variables), so the per-variable fan-out
            # cannot honour it. Reject it explicitly rather than silently reading eagerly.
            if chunks is not None:
                raise ValueError(
                    "crop(chunks=…) is not supported on a root container; it is a "
                    "curvilinear-only, per-variable option — call crop on a single variable "
                    "via get_variable(name) instead."
                )
            # `path` streams every cropped variable straight to `path` slab-by-slab (bounded memory)
            # and returns a file-backed NetCDF; `path=None` keeps the in-memory fan-out.
            result = nc._apply_to_all_variables(
                "crop", {"mask": mask, "touch": touch}, path=path
            )
        else:
            result = self._crop_one(mask, touch=touch, chunks=chunks)._persist_to(path)
        return cast("NetCDF", result)

    def _try_antimeridian(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None,
        mask: Any,
        epsg: Any,
        is_container: bool,
        touch: bool,
        chunks: Any,
    ) -> NetCDF | None:
        """Return an antimeridian crop when a geographic west>east bbox warrants it.

        Args:
            bbox: The crop bbox, or ``None``.
            mask: The crop mask, or ``None`` (antimeridian is bbox-only).
            epsg: The bbox CRS override, or ``None`` (defaults to the dataset CRS).
            is_container: Whether ``self`` is a root MDIM container.
            touch: Forwarded to the per-half crop.
            chunks: Forwarded to the per-half crop.

        Returns:
            The cropped result when the bbox is a geographic ``west > east``
            antimeridian request on a geographic dataset — a stitched variable
            strip, a masked curvilinear window, or a container with every variable
            cropped — otherwise ``None``.
        """
        result: NetCDF | None = None
        if bbox is not None and mask is None:
            nc = self._ds
            crs = epsg if epsg is not None else nc.epsg
            west, _, east, _ = bbox
            crs_geo = crs is not None and sr_from_user_input(crs).IsGeographic()
            ds_geo = nc.epsg is not None and sr_from_user_input(nc.epsg).IsGeographic()
            if west > east and crs_geo and ds_geo:
                # The unpacking above already asserts the 4-element shape a bbox has.
                bbox_tuple = cast("tuple[float, float, float, float]", tuple(bbox))
                _require_antimeridian_seam(nc, bbox_tuple)
                if is_container:
                    result = self._crop_antimeridian_container(
                        bbox_tuple, crs, touch, chunks
                    )
                else:
                    result = self._crop_antimeridian(bbox_tuple, crs, touch, chunks)
        return result

    def _crop_antimeridian_container(
        self,
        bbox: tuple[float, float, float, float],
        crs: Any,
        touch: bool,
        chunks: Any,
    ) -> NetCDF:
        """Fan an antimeridian bbox out across every variable of a root container.

        The container has no single crop to run — each variable splits the bbox in
        its own longitude frame and stitches (rectilinear) or masks (curvilinear)
        itself, and :meth:`_apply_to_all_variables` reassembles the cropped
        variables into a new container.

        Note: as with any container fan-out, a *curvilinear* variable is rebuilt
        from its cropped result's affine (bbox) geotransform, so its 2-D lon/lat
        coordinates are dropped and it comes back rectilinear-approximated. Crop a
        curvilinear variable directly (``get_variable(name).crop(...)``) to keep
        its 2-D coordinates.

        Args:
            bbox: ``(west, south, east, north)`` with ``west > east``.
            crs: The bbox CRS, forwarded to every per-variable crop.
            touch: Forwarded to every per-variable crop.
            chunks: Must be ``None`` — antimeridian crops are eager.

        Returns:
            NetCDF: A new container with every gridded variable cropped.

        Raises:
            ValueError: ``chunks`` was supplied.
        """
        _reject_antimeridian_chunks(chunks)
        return cast(
            "NetCDF",
            self._ds._apply_to_all_variables(
                "crop", {"bbox": bbox, "epsg": crs, "touch": touch}
            ),
        )

    def _crop_antimeridian(
        self,
        bbox: tuple[float, float, float, float],
        crs: Any,
        touch: bool,
        chunks: Any,
    ) -> NetCDF:
        """Crop a variable with a geographic ``west > east`` (antimeridian) bbox.

        A curvilinear variable (2-D lon/lat coords) is masked on its coordinate
        arrays; a rectilinear one splits the bbox at the grid's longitude seam
        (``180`` on a ``-180..180`` grid, ``360`` on a ``0..360`` grid), crops each
        ``west < east`` half through the normal path, and concatenates the halves
        along longitude into one contiguous variable whose coordinates continue past
        the seam. A half outside the variable's longitude extent is skipped, so a
        single-sided overlap returns just that half.

        Args:
            bbox: ``(west, south, east, north)`` with ``west > east``.
            crs: The bbox CRS.
            touch: Forwarded to the per-half crop / curvilinear mask.
            chunks: Curvilinear-only lazy read; must be ``None`` on the rectilinear
                path, whose merge is eager.

        Returns:
            NetCDF: The cropped strip (rectilinear) or masked window (curvilinear)
            spanning the seam.

        Raises:
            ValueError: ``chunks`` was supplied on the rectilinear path, or the bbox
                does not overlap the variable's longitude extent.
        """
        curv = _curvilinear_coords_2d(self._ds)
        if curv is not None:
            result = self._crop_antimeridian_curvilinear(bbox, crs, curv, touch, chunks)
        else:
            _reject_antimeridian_chunks(chunks)
            result = _crop_seam_halves(
                self._ds,
                bbox,
                lambda half: self.crop(bbox=half, epsg=crs, touch=touch),
                self._merge_lon_halves,
            )
        return result

    def _crop_antimeridian_curvilinear(
        self,
        bbox: tuple[float, float, float, float],
        crs: Any,
        coords2d: tuple[np.ndarray, np.ndarray],
        touch: bool,
        chunks: Any,
    ) -> NetCDF:
        """Crop a curvilinear variable with a ``west > east`` bbox via a split mask.

        Curvilinear grids have no affine seam to stitch across, but their crop
        already masks on the 2-D ``(lon, lat)`` arrays — so the wrap is handled by
        the *mask*, not a stitch. The bbox is split into ``west < east`` halves
        (keyed off the 2-D longitude array's own max, so a 0..360 grid is detected
        without an affine geotransform), turned into a polygon per half (a
        ``MultiPolygon`` on a -180..180 grid, one box on a 0..360 grid), and passed
        to the standard curvilinear point-in-polygon mask + window.

        Args:
            bbox: ``(west, south, east, north)`` with ``west > east``.
            crs: The bbox CRS for the polygon mask.
            coords2d: The variable's 2-D ``(lon, lat)`` coordinate arrays.
            touch: Forwarded to the curvilinear mask (currently a no-op there).
            chunks: Forwarded to the curvilinear windowed read (lazy when set).

        Returns:
            NetCDF: The masked + windowed curvilinear subset spanning the seam.
        """
        lon2d = np.asarray(coords2d[0], dtype=float)
        finite = lon2d[np.isfinite(lon2d)]
        lon_max = float(finite.max()) if finite.size else 0.0
        halves = _split_lon_bbox(bbox, lon_max, _lon_cell_size(lon2d))
        mask = FeatureCollection(
            gpd.GeoDataFrame(geometry=[box(*half) for half in halves], crs=crs)
        )
        return self._crop_curvilinear(mask, coords2d, touch=touch, chunks=chunks)

    def _merge_lon_halves(self, west_part: NetCDF, east_part: NetCDF) -> NetCDF:
        """Concatenate two longitude-adjacent variable crops into one contiguous result.

        `west_part` (the pre-seam half) sits to the left and `east_part` (the
        wrapped half past the seam) to its right; the merged raster keeps
        `west_part`'s north-up geotransform, so the longitude mapping continues
        past the seam. The result is re-wrapped as :class:`NetCDF` so variable
        metadata (band dims, ``sel``) survives.

        Args:
            west_part: Crop of the pre-seam half.
            east_part: Crop of the post-seam half.

        Returns:
            NetCDF: The concatenated variable.
        """
        raster = _stitch_lon_halves(self._ds, west_part, east_part)
        return self._ds._preserve_netcdf_metadata(raster)

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
            # `.epsg` is None for a no-EPSG CRS (e.g. geostationary); fall back to
            # the WKT so a bbox in the grid's own CRS is still honoured (#706).
            crs = epsg if epsg is not None else crs_spec(self._ds.epsg, self._ds.crs)
            if not crs:
                raise ValueError(
                    "crop(bbox=…) requires an explicit `epsg=` when the "
                    "NetCDF has no CRS at all — a bbox without a CRS is ambiguous"
                )
            mask = FeatureCollection.from_bbox(bbox, epsg=crs)
        if mask is None:
            raise TypeError(
                "crop requires a `mask` (GeoDataFrame / FeatureCollection / "
                "Dataset) or a `bbox` (west, south, east, north) tuple"
            )
        return mask

    def _crop_one(self, mask: Any, touch: bool = True, chunks: Any = None) -> NetCDF:
        """Crop a single variable/subset, routing curvilinear grids to the 2-D coordinate masker.

        Curvilinear grids (2-D lon/lat coords, no single affine geotransform) can't be clipped by the
        affine cutline warp, so they mask on their 2-D coordinates; rectilinear grids use the affine
        crop and re-wrap to preserve NetCDF metadata. ``chunks`` is valid only on the curvilinear path.

        The rectilinear path first offers the crop to :meth:`_mask_window_source`, which reads just
        the mask's window from the MDArray when it can prove that equivalent, and otherwise declines
        so the existing full-read crop runs unchanged (#1071). The returned crop is identical either
        way; the difference is a side effect on the *receiver*, which the shortcut skips — a crop
        that declines stamps the variable's CRS onto its backing raster, and may materialize the
        multidim view, so a subsequent operation finds a raster that has already been fixed up. A
        crop that takes the shortcut leaves the receiver untouched.

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
        curv = _curvilinear_coords_2d(nc)
        if curv is not None:
            result = self._crop_curvilinear(
                mask,
                curv,
                touch=touch,
                chunks=chunks,
            )
        else:
            if chunks is not None:
                raise ValueError(
                    "chunks= is only supported for curvilinear crop; the affine "
                    "(rectilinear) crop path is eager."
                )
            # Crop the mask's window rather than the whole variable when that window can be read
            # straight from the MDArray: the affine crop reads its source in full, which over a
            # remote store means fetching the entire variable to clip a few cells. This has to run
            # *before* the CRS stamping below, because materializing drops the root-group reference
            # the windowed read needs — and the windowed raster carries the CRS already (#1071).
            source = self._mask_window_source(mask)
            if source is None:
                # Stamp the variable's known CRS onto its backing raster before the cutline warp.
                # A NetCDF variable tracks its EPSG even when the raster (the AsClassicDataset
                # MDArray view, or a wrap_longitude/materialized MEM raster) carries no projection
                # string; without it GDAL's cutline warp warns ("the input vector layer has a SRS,
                # but the source raster dataset does not") and — for a cutline in a different CRS —
                # would clip the wrong region. The driver-less MDArray view does not persist
                # SetProjection, so materialize it first (that path reads it fully anyway).
                # See issue #629.
                if (
                    nc.epsg
                    and nc._raster is not None
                    and not nc._raster.GetProjection()
                ):
                    wkt = sr_from_epsg(int(nc.epsg)).ExportToWkt()
                    nc._raster.SetProjection(wkt)
                    if not nc._raster.GetProjection():
                        nc._materialize_md_view()
                        nc._raster.SetProjection(wkt)
                source = nc
            # `nc.spatial.crop` is the base Dataset affine crop — exactly what the
            # NetCDF.crop override reached via `super().crop(...)`, bypassing this engine.
            result = nc._preserve_netcdf_metadata(
                source.spatial.crop(mask=mask, touch=touch)
            )
        return result

    def _mask_window_source(self, mask: Any) -> Dataset | None:
        """A `Dataset` over just the mask's window, read straight from the MDArray.

        The affine crop reads its whole source before clipping. That is cheap for a local file
        and expensive for a remote one — the classic view a variable is backed by turns a small
        windowed read into a strided gather, so clipping a few cells out of a 14 GB `/vsicurl`
        NetCDF-4 costs seconds. Reading the window through
        :meth:`~pyramids.netcdf.NetCDF._window_via_mdarray` instead is roughly an order of
        magnitude cheaper, and the clip that follows is identical because the window is built to
        contain the mask.

        Declines (returns ``None``, leaving the caller to crop the full variable) whenever the
        shortcut is not provably equivalent: a rotated or degenerate affine, a mask whose CRS is
        not known to equal the raster's (whose cutline the warp must reproject), non-finite mask
        bounds, a mask that misses the grid, a window that is not appreciably smaller than the
        variable, or a read the MDArray cannot serve.

        Args:
            mask: The resolved mask the crop will clip with — a `FeatureCollection`, a bare
                `geopandas.GeoDataFrame` (passed straight through by `_resolve_crop_mask`), or a
                `Dataset` whose footprint is used. Only `crs` and `total_bounds` are read, so any
                of them is accepted; anything lacking them declines.

        Returns:
            Dataset | None: A raster of the window carrying its sub-affine, or ``None``.
        """
        nc = self._ds
        gt = nc._geotransform
        if not gt or not gt[1] or not gt[5] or gt[2] or gt[4]:
            return None
        # Compare the CRSs themselves, the way `Spatial._cutline_window_bounds` does. An `epsg`
        # comparison fails open twice over: `crop(mask=...)` accepts a bare `GeoDataFrame`, which
        # has no `.epsg` at all, and a grid with no authority code (rotated pole, geostationary)
        # reports `epsg` as `None`. Either way the guard would be skipped and the mask's
        # unreprojected coordinates divided through this raster's affine -- a plausible-looking
        # window over the wrong part of the grid, which is wrong data rather than an error.
        # Unknown on either side is not "equal": decline and let the warp reproject the cutline.
        # `crs` is a pyproj CRS on a GeoDataFrame/FeatureCollection but a plain WKT string on a
        # Dataset mask, so normalise before comparing rather than assume either shape.
        mask_crs = getattr(mask, "crs", None)
        if mask_crs is not None and hasattr(mask_crs, "to_wkt"):
            mask_crs = mask_crs.to_wkt()
        source_crs = nc.crs
        if not source_crs or not mask_crs:
            return None
        if not crs_equal(source_crs, mask_crs):
            return None
        try:
            xmin, ymin, xmax, ymax = (float(bound) for bound in mask.total_bounds)
        except (AttributeError, TypeError, ValueError):
            return None
        # An empty or all-null-geometry mask has non-finite bounds; `math.floor(nan)` raises
        # `ValueError` and `math.floor(inf)` `OverflowError`. The full-read path reports that as a
        # clean "Did not get any cutline features", so decline rather than turn it into a numeric
        # error raised out of an optimisation the caller never asked for.
        if not all(math.isfinite(bound) for bound in (xmin, ymin, xmax, ymax)):
            return None
        columns = [(xmin - gt[0]) / gt[1], (xmax - gt[0]) / gt[1]]
        rows = [(ymax - gt[3]) / gt[5], (ymin - gt[3]) / gt[5]]
        # One cell of slack on every side so `touch=True` and half-open rounding cannot clip a
        # boundary cell the full-read path would have kept.
        x_off = max(0, math.floor(min(columns)) - 1)
        y_off = max(0, math.floor(min(rows)) - 1)
        x_end = min(nc.columns, math.ceil(max(columns)) + 1)
        y_end = min(nc.rows, math.ceil(max(rows)) + 1)
        x_size, y_size = x_end - x_off, y_end - y_off
        if x_size <= 0 or y_size <= 0:
            return None
        if x_size * y_size * _MIN_WINDOW_SAVING >= nc.columns * nc.rows:
            return None
        try:
            raster = nc._window_via_mdarray(x_off, y_off, x_size, y_size)
        except (RuntimeError, AttributeError, ValueError):
            # The shortcut is an optimisation the caller never asked for; anything it fails on must
            # reach the ordinary full-read crop, not surface as an error out of `crop()`.
            raster = None
        return None if raster is None else Dataset(raster)

    def _crop_curvilinear(
        self,
        mask: FeatureCollection,
        coords2d: tuple[np.ndarray, np.ndarray],
        touch: bool = True,
        chunks: Any = None,
    ) -> NetCDF:
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
        container = nc.from_array(
            data_win,
            geo_ref=GeoReference(
                geo=nc._bbox_geotransform(lon_win, lat_win),
                epsg=crs_spec(nc.epsg, nc.crs),
            ),
            no_data_value=nd,
            variable_name=var_name,
        )
        # from_array returns a root container; hand back the variable subset, carrying the
        # windowed 2-D coordinates so the result stays curvilinear (plots on its real geometry).
        result = container._require_raster_variable(var_name)
        # A spatial window leaves the band dimensions untouched, so restore the source variable's
        # band-dim names, coordinate values and sizes onto the rebuild. from_array infers only
        # generic (dim_0, dim_1) axes from the array shape, which would drop ocean_time / s_rho
        # and break sel() by coordinate after the crop (#1241).
        result = nc._preserve_netcdf_metadata(result)
        # The window holds stored counts (`_read_curvilinear_window` asks for them),
        # so the rebuilt variable has to declare what turns them back into
        # measurements, exactly as the affine crop path does.
        result._scale = nc._scale
        result._offset = nc._offset
        for index in range(1, result.raster.RasterCount + 1):
            carry_band_packing(
                nc.raster.GetRasterBand(1), result.raster.GetRasterBand(index)
            )
        result._curvilinear_coords = (lon_win, lat_win)
        return result

    def isel(self, *, drop: bool = False, **indexers: Any) -> NetCDF:
        """Select bands by **position** along one or more band dimensions.

        The positional twin of :meth:`sel`. Where `sel` asks "which band has this
        coordinate value", this asks "which band is at this index" — so it works on an axis
        the store gives no coordinates for, which is the case `sel` cannot serve at all. A
        WRF store's `bottom_top` is the usual example: 27 model levels with no coordinate
        variable, where `sel(bottom_top=...)` can only refuse.

        Several dimensions may be given in one call. They are applied in sequence, and
        because each cut is independent of the others the order does not affect the result.

        Args:
            drop: When `True`, drop the axes a **scalar** (point) selector collapsed —
                `isel(time=0, drop=True)` returns the plane without a length-one `time`,
                matching xarray's `isel(..., drop=True)`. Only a scalar index is
                dimension-reducing: a length-one `list` or `slice` (`isel(time=[0])`,
                `isel(time=slice(0, 1))`) keeps its axis, as xarray keeps it, and a
                pre-existing length-one band dimension the call never indexed (an ensemble
                `member=1`, say) is kept too. Keyword-only, so it is never read as a
                dimension name; `drop=False` (the default) keeps every axis, as before.
            **indexers: One or more `dimension=selector` pairs. Each selector is an index,
                a `list` or `tuple` of indices, or a `slice` of them. "Index" means
                anything `operator.index()` accepts, so a numpy integer counts and needs no
                `int(...)` wrapper. A negative index counts from the end, and a slice's
                `step` is honoured — unlike `sel`'s, where a range of coordinate values has
                no meaningful stride.

        Returns:
            NetCDF: A variable holding the selected bands, with `_band_dim_sizes` and the
            coordinate map narrowed to match. A dimension with no coordinates keeps none.
            With `drop=True`, the axes a scalar selector collapsed are removed.

        Raises:
            ValueError: No indexers were given, the variable tracks no band dimensions, a
                named dimension is not one of them, or a selector keeps no position — an
                empty `list` or `tuple` as much as a `slice` whose bounds cross.
            IndexError: An index is outside the dimension's range.
            TypeError: A selector is not an index, a `list`/`tuple` of indices, or a
                `slice`. That includes a `float`, a `str`, a `range`, a `set`, and an
                array of one or more dimensions. A `bool` is refused separately, with its
                own message, because `operator.index()` would otherwise admit it.

        Examples:
            - Take the first time step of a `(time, pressure_level)` cube, leaving the
              levels untouched:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
              >>> cube = nc["temperature"]
              >>> cube.band_count
              12
              >>> first = cube.isel(time=0)
              >>> first.band_count
              3

              ```
            - Several dimensions in one call, down to a single plane:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
              >>> plane = nc["temperature"].isel(time=1, pressure_level=2)
              >>> plane.band_count
              1
              >>> plane._band_dim_values_map["time"]
              [6.0]

              ```
            - A negative index counts from the end, as it does in xarray:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
              >>> nc["temperature"].isel(time=-1)._band_dim_values_map["time"]
              [18.0]

              ```
            - `drop=True` removes the length-one `time` the selection leaves behind,
              instead of carrying it through `to_file` / `to_xarray`:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
              >>> cube = nc["temperature"]
              >>> cube.isel(time=0)._band_dim_names
              ('time', 'pressure_level')
              >>> cube.isel(time=0, drop=True)._band_dim_names
              ('pressure_level',)

              ```
            - The case `sel` cannot serve — a WRF `bottom_top` axis the store gives no
              coordinates for. The result keeps `None` there rather than inventing an
              axis:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file(
              ...     "tests/data/netcdf/none__17v__1d1-2d5-3d6-4d5__stag-str.nc"
              ... )
              >>> levels = nc["T"]
              >>> levels._band_dim_sizes
              (3, 27)
              >>> level = levels.isel(bottom_top=2)
              >>> level._band_dim_sizes
              (3, 1)
              >>> level._band_dim_values_map["bottom_top"] is None
              True

              ```

        Notes:
            Where this parts company with xarray's `isel`, which indexes fancily:

            - A **list** is sorted and deduplicated before it is applied, so a list can
              neither reorder nor repeat an axis. `isel(time=[2, 0])` leaves `time` as
              `[0.0, 12.0]` where xarray gives `[12.0, 0.0]`, and `isel(time=[2, 2])`
              keeps one band where xarray keeps two.
            - A **slice does** reorder, and agrees with xarray when it does:
              `isel(time=slice(None, None, -1))` reverses the axis to
              `[18.0, 12.0, 6.0, 0.0]`, and the planes are reversed with it — each stays
              attached to its own coordinate. Only the list form is normalised.
            - A slice that selects nothing raises `ValueError` instead of producing a
              zero-length axis. The empty variable would build, then fail much later and
              further away on the first read.
            - A numpy *integer* is accepted — anything `operator.index()` admits is an
              index here, so a value out of `np.argmin`, `np.where(...)[0][0]` or
              iterating an array needs no `int(...)` wrapper. A **0-d** array is accepted
              for the same reason: `operator.index()` takes it, and `np.array(2)` is a
              scalar in all but type.
            - An array of **one or more** dimensions is refused, and so is a boolean
              **mask**; xarray takes both. Pass `list(values)` for the first.
            - A `bool` is refused although `operator.index()` admits it, because
              `isel(time=True)` would quietly mean position 1 and a caller writing it
              almost certainly means a mask.
            - A `tuple` of indices is accepted here and rejected by xarray.

        See Also:
            NetCDF.sel: The same cut, addressed by coordinate value.
        """
        nc = self._ds
        if not indexers:
            raise ValueError(
                "isel() requires at least one keyword argument, e.g. isel(time=0)."
            )

        # Resolve every keyword before cutting anything. Validating inside the loop meant a
        # typo in the second keyword raised only after the first cut had been read from
        # disk — 3 of 12 bands on the CF fixture — so the caller paid for a read whose
        # result was thrown away. Resolution touches metadata only, never pixels.
        resolved: list[tuple[str, list[int]]] = []
        scalar_dims: list[str] = []
        for dim_name, selector in indexers.items():
            _assert_band_dimension(nc, dim_name, caller="isel")
            axis = nc._band_dim_names.index(dim_name)
            size = nc._band_dim_sizes[axis]
            resolved.append(
                (dim_name, _resolve_positional_indices(selector, size, dim_name))
            )
            # A scalar (point) selector is dimension-reducing, as it is in xarray; a
            # `list`/`tuple`/`slice` is not, even when it keeps a single band. Only the
            # former is a `drop=` candidate.
            if not isinstance(selector, (slice, list, tuple)):
                scalar_dims.append(dim_name)

        result = nc
        for dim_name, dim_indices in resolved:
            result = _subset_along_dim(result, dim_name, dim_indices)
        if drop:
            # Match xarray: drop only the axes a *scalar* selector collapsed. A length-one
            # `list` or `slice` keeps its axis there, and a pre-existing length-one band dim
            # the call never indexed is kept too — so this is neither a blanket `squeeze()`
            # nor a drop-every-length-one rule (#1193). A scalar always resolves to one
            # index, so each squeezed axis is length one by construction.
            for dim_name in scalar_dims:
                if dim_name in result._band_dim_names:
                    result = result.squeeze(dim_name)
        return result

    def sel(
        self,
        *,
        method: str | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> NetCDF:
        """Select a subset of bands by coordinate values along a band dim.

                Extracts bands whose coordinate values match the given criteria.
                Works on any variable subset that has at least one non-spatial
                dimension tracked in `_band_dim_names` (set by
                `get_variable()`). For 4-D+ files with multiple non-spatial
                dims (e.g. `(valid_time, pressure_level, lat, lon)` from CDS-Beta
                ERA5), `sel()` may name any of those dims, and several in one
                call: `sel(time=6, pressure_level=850)` is the same cut as
                `sel(time=6).sel(pressure_level=850)`. Each dimension narrows a
                different axis of the same band grid, so the order the keywords
                are written in does not affect the result.

                The result is always a `NetCDF` instance with the same variable
                metadata preserved, so `sel()` can be chained and NetCDF-only
                methods like `read_array(unpack=True)` remain available.

        Every keyword is *resolved* before anything is read, so a bad
                name or an unmatched value costs nothing. The cuts themselves
                are still applied one per keyword, so a call naming two
                dimensions reads twice — the first cut is materialised, then
                narrowed again. Naming the dimension that discards most bands
                first is therefore cheaper today: on a `(time=4, level=3)` cube
                `sel(time=…, pressure_level=…)` reads 3 bands then 1, while the
                reverse reads 4 then 1.

                That second read is removable rather than inherent — the
                positions for every dimension are known before the first cut, so
                one pass could emit the final band list directly. It is left
                alone deliberately: doing it means rewriting
                `_map_dim_to_band_indices`, which this branch has already
                corrected once, and the gain is reads rather than correctness.
                Order never affects the result.

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
                    method: How a selector is matched against the axis.
                        `None` (the default) matches exactly; `"nearest"`
                        snaps each requested value to the closest coordinate,
                        so a caller can ask for "the level nearest 100 m"
                        without knowing the axis values. `"nearest"` needs a
                        numeric selector — it rejects a `slice` (a range has
                        no nearest value) and a date label (select a label
                        exactly; a partial one already names a period). The
                        coordinate it chose is on the result, readable with
                        `get_dimension_values(dim)`.
                    tolerance: The furthest a `method="nearest"` snap may
                        travel. `None` (the default) accepts any distance. A
                        request whose closest coordinate lies further away
                        raises `KeyError`, with the distance and the bound in
                        the message. Rejected without `method="nearest"`, where
                        an exact match has no distance for it to bound.

                        **One bound governs every dimension in the call**, and
                        it is compared against each axis in that axis' own
                        units. `sel(time=5, pressure_level=990,
                        method="nearest", tolerance=20)` allows a 20-hour snap
                        on `time` and a 20-hPa snap on `pressure_level`, which
                        is rarely what a caller means. Bound one dimension per
                        call when the units differ.
                    **kwargs: One or more keyword arguments. Each key must name
                        a tracked band dim (one of `self._band_dim_names`); the
                        value is one of:

                        - A single number: select one band by exact value.
                        - A list of numbers: select multiple bands.
                        - A `slice(start, stop)`: select bands whose coord
                          falls between `start` and `stop` inclusive. Bounds
                          are normalised before matching, so the slice is
                          direction-agnostic — works on both ascending and
                          descending coord axes (e.g. `latitude` stored
                          north-to-south).
                        - A date label, a list of them, or a slice of them, on
                          a CF time axis: `"2024-01-01"`. The axis is stored as
                          raw offsets (`[0.0, 6.0, 12.0, 18.0]`), so a label is
                          matched by decoding the axis with the dimension's
                          `units` / `calendar` at the label's own precision —
                          meaning a partial label matches every step inside the
                          period it names (`"2024-01"` takes the whole month,
                          `"2024-01-01 06:00:00"` takes one step). This is the
                          vocabulary `get_time_variable` hands back — note its
                          **default** `time_format` is `"%Y-%m-%d"`, so feeding
                          one of its labels back selects that whole day; ask for
                          `get_time_variable(dim, "%Y-%m-%d %H:%M:%S")` to get
                          the labels that pin a single step. An axis whose
                          `units` cannot be parsed, or whose values the CF
                          converter cannot decode, has no labels to match, and a
                          label selector on it finds nothing.

                Returns:
                    NetCDF: A new variable subset with only the selected bands
                        and full metadata preserved. `_band_dim_sizes` reflects
                        the pinned axis (e.g. `(4, 1)` after pinning a level on
                        a `(4, 3)` cube), and `_band_dim_values_map[dim_name]`
                        shrinks to the chosen values. Legacy `_band_dim_values`
                        is refreshed from the (possibly updated) primary entry
                        in the map.

                Raises:
                    ValueError: If no kwarg is passed, `method` is neither
                        `None` nor `"nearest"`, `tolerance` is given without
                        `method="nearest"` or is negative, the variable has no
                        tracked band dims, the named dim isn't one of
                        `_band_dim_names`, the dim has no coord values
                        (`_band_dim_values_map[dim] is None` — select by
                        position with `isel()` instead), `"nearest"` is asked
                        of a slice / a date label / a non-numeric axis, or no
                        bands match the selector.
                    KeyError: A `method="nearest"` request found no coordinate
                        within `tolerance`.

                Note:
                    **Two types for one kind of failure.** A selector that
                        matches nothing raises `ValueError` ("No bands match
                        ..."), while a `tolerance` breach raises `KeyError`.
                        Both mean "your selector matched nothing", so
                        `except ValueError` around a `sel` call does not catch
                        the bounded miss and `except KeyError` does not catch
                        the plain one — catch both, or `except Exception`.

                        The split is historical rather than designed:
                        `ValueError` is what `sel` has always raised, and
                        `tolerance` arrived matching xarray, which uses
                        `KeyError`. xarray uses `KeyError` for *both*, so this
                        is not xarray parity. Unifying it would change a
                        released exception type on the commonly hit path, so it
                        is recorded here rather than quietly fixed.

                Examples:
                    - Pin a pressure level on a 4-D `(time, pressure_level)` cube:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> sub = nc.get_variable("temperature").sel(pressure_level=500)
                        >>> sub._band_dim_sizes
                        (4, 1)
                        >>> sub._band_dim_values_map["pressure_level"]
                        [500.0]

                        ```
                    - Name both dims in one call, or chain two calls — the same cut either way,
                      and in either keyword order (collapses to a single 2-D plane):
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> var = nc.get_variable("temperature")
                        >>> var.sel(time=12, pressure_level=500)._band_dim_values_map
                        {'time': [12.0], 'pressure_level': [500.0]}
                        >>> var.sel(pressure_level=500, time=12)._band_dim_values_map
                        {'time': [12.0], 'pressure_level': [500.0]}
                        >>> var.sel(time=12).sel(pressure_level=500).read_array().shape
                        (1, 1, 5, 6)

                        ```
                    - Use a list selector to keep only two of the levels:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> sub = nc.get_variable("temperature").sel(pressure_level=[1000, 500])
                        >>> sub._band_dim_values_map["pressure_level"]
                        [1000.0, 500.0]

                        ```
                    - Use a slice selector — direction-agnostic, so the same
                      call works on ascending coords (e.g. `[500, 850, 1000]`)
                      and on descending ones like this fixture's
                      (`[1000, 850, 500]`):
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> var = nc.get_variable("temperature")
                        >>> var.sel(pressure_level=slice(500, 1000))._band_dim_values_map["pressure_level"]
                        [1000.0, 850.0, 500.0]

                        ```
                    - Snap to the nearest level, then read back which one was
                      chosen:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> sub = nc.get_variable("temperature").sel(pressure_level=900, method="nearest")
                        >>> sub.get_dimension_values("pressure_level")
                        array([850.])

                        ```
                    - Bound the snap with `tolerance`; a request whose closest
                      coordinate lies further away raises instead of snapping:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> var = nc.get_variable("temperature")
                        >>> var.sel(pressure_level=900, method="nearest", tolerance=100)._band_dim_values_map[
                        ...     "pressure_level"
                        ... ]
                        [850.0]
                        >>> var.sel(pressure_level=900, method="nearest", tolerance=10)
                        Traceback (most recent call last):
                            ...
                        KeyError: 'no coordinate within tolerance=10 of 900: the closest is 850.0...'

                        ```
                    - Select a time step by its date label rather than by the
                      raw CF offset. A full-precision label pins one step:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> sub = nc.get_variable("temperature").sel(time="2024-01-01 12:00:00")
                        >>> sub._band_dim_values_map["time"]
                        [12.0]

                        ```
                    - A label from `get_time_variable()` at its default
                      `"%Y-%m-%d"` names a **day**, so it keeps every step in
                      that day — ask for the finer format to pin one:
                        ```python
                        >>> from pyramids.netcdf import NetCDF
                        >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                        >>> var = nc.get_variable("temperature")
                        >>> nc.get_time_variable("time")[1]
                        '2024-01-01'
                        >>> var.sel(time="2024-01-01")._band_dim_values_map["time"]
                        [0.0, 6.0, 12.0, 18.0]
                        >>> fine = nc.get_time_variable("time", "%Y-%m-%d %H:%M:%S")
                        >>> var.sel(time=fine[1])._band_dim_values_map["time"]
                        [6.0]

                        ```

                Notes:
                    A slice's `step` is ignored, on the label path as on the
                    stored-value one: `slice(a, b, 2)` selects the same bands
                    as `slice(a, b)`. Pass a list to pick specific values.

                    `method` and `tolerance` are keywords of this method, so a
                    band dim actually named either cannot be selected through
                    it — the selector would be taken as the option. Use
                    `isel()` for such a dimension: it reserves neither. Both
                    still reserve `self`, which is Python's method binding
                    rather than a keyword of either, and no netCDF dimension is
                    plausibly named that.

                    The examples above run against this repository's own
                    fixtures. Wider scenarios live in:

                    - `tests/netcdf/selection/test_sel_nearest_and_labels.py`
                      (`TestSelNearest` / `TestSelByDateLabel` — snapping and
                      date-label selection, including the vocabulary a failed
                      match reports and the axis whose units do not parse).
                    - `tests/netcdf/selection/test_sel.py::TestSelSingleValue` /
                      `TestSelList` / `TestSelSlice` (3-D scenarios — single
                      value, list selector, slice selector including the
                      direction-agnostic path).
                    - `tests/netcdf/selection/test_sel_4d.py::TestSelByPressureLevel` /
                      `TestSelByTime` / `TestSelChained` (4-D scenarios —
                      pin secondary / primary dim, chained `sel().sel()`).
                    - `tests/netcdf/selection/test_sel_4d.py::TestSelErrorMessages` (the
                      error contract).

                See Also:
                    `get_variable`: builds a variable subset and populates the
                        band-dim metadata that `sel()` consumes.
        """
        nc = self._ds
        if not kwargs:
            raise ValueError(
                "sel() requires at least one keyword argument, e.g. sel(time=6)."
            )
        if method not in (None, "nearest"):
            raise ValueError(
                f"sel() method must be None (exact) or 'nearest', got {method!r}."
            )
        if tolerance is not None and method != "nearest":
            raise ValueError(
                "sel() tolerance= is only meaningful with method='nearest' — without it "
                "a label either matches exactly or does not match at all, and there is no "
                "distance for a tolerance to bound."
            )

        # Resolve every keyword against the receiver before cutting anything, so a bad
        # name *or* a value that matches nothing is refused without having read the earlier
        # keywords' bands.
        #
        # Resolving against the original receiver rather than the progressively narrowed
        # one is equivalent: `_subset_along_dim` copies `_band_dim_values_map` and replaces
        # only its own dimension's entry, so cutting `time` leaves `pressure_level`'s
        # coordinates exactly as they were, and a selector resolves to the same positions
        # either way. An earlier version of this hoisted only the name check, on the stated
        # grounds that a preceding cut could narrow the coordinates a label needs — which
        # is not something any cut does.
        resolved: list[tuple[str, list[int]]] = []
        for dim_name, selector in kwargs.items():
            resolved.append(
                (dim_name, _resolve_one_dim(nc, dim_name, selector, method, tolerance))
            )

        result = nc
        # One cut per dimension, in the order the keywords were written. They compose in
        # any order because each narrows a different axis of the same band grid.
        for dim_name, dim_indices in resolved:
            result = _subset_along_dim(result, dim_name, dim_indices)
        return result

    def head(self, indexers: Any = None, **indexers_kwargs: int) -> NetCDF:
        """Keep the first `n` steps along one or more band dimensions.

        **Band dimensions only**, where xarray's `head` also windows the spatial axes: a
        raster's rows and columns are its grid, so `head()` on a `(time, lat, lon)` cube
        keeps every row and column, while xarray's would keep the first five of each.

        Args:
            indexers: xarray's positional spellings — a count applied to every band
                dimension (`head(2)`), or a mapping of them (`head({"time": 2})`). `None`
                (default) reads the keywords instead, and the two cannot be mixed.
            **indexers_kwargs: `dimension=n` pairs, `n` a whole number of at least 1. With
                neither these nor `indexers`, the first five steps along **every** band
                dimension, which is xarray's default. Asking for more than a dimension
                holds keeps it whole.

        Returns:
            NetCDF: A variable holding the kept steps, its coordinates cut to match.

        Raises:
            ValueError: A dimension is not a band dimension, `n` is below 1 — GDAL has no
                raster of no bands, where xarray answers an empty axis — the receiver has
                no band dimension at all, or both spellings were used at once.
            TypeError: `n` is not an integer.

        Examples:
            - The first two steps of a four-step cube:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.head(time=2)._band_dim_values_map["time"]
              [0.0, 6.0]

              ```
            - xarray's positional spellings, which apply to every band dimension:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.head(2)._band_dim_values_map["time"]
              [0.0, 6.0]
              >>> var.head({"time": 3})._band_dim_values_map["time"]
              [0.0, 6.0, 12.0]
              >>> var.head(2, time=2)
              Traceback (most recent call last):
                  ...
              ValueError: head() takes either a count or head(dimension=n) keywords, not both...

              ```
            - Zero steps is refused — GDAL has no raster of no bands:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.head(time=0)
              Traceback (most recent call last):
                  ...
              ValueError: head() needs a count of at least 1 for 'time', got 0...

              ```

        See Also:
            NetCDF.tail: The last `n` steps instead.
            NetCDF.isel: Any positions, by index.
        """
        _refuse_a_container(self._ds, "head")
        wanted, count = _window_arguments(
            self._ds, indexers, indexers_kwargs, "head", _DEFAULT_WINDOW, _whole_axis
        )
        return _windowed(self._ds, wanted, "head", _head_positions, default=count)

    def tail(self, indexers: Any = None, **indexers_kwargs: int) -> NetCDF:
        """Keep the last `n` steps along one or more band dimensions.

        :meth:`head` from the other end, with the same arguments and refusals, band
        dimensions only.

        Args:
            indexers: A count for every band dimension (`tail(2)`) or a mapping of them,
                as in xarray. `None` (default) reads the keywords instead.
            **indexers_kwargs: `dimension=n` pairs, `n` a whole number of at least 1. With
                neither, the last five steps along every band dimension.

        Returns:
            NetCDF: A variable holding the kept steps, its coordinates cut to match.

        Raises:
            ValueError: A dimension is not a band dimension, `n` is below 1, the receiver
                has no band dimension, or both spellings were used at once.
            TypeError: `n` is not an integer.

        Examples:
            - The last two steps of a four-step cube:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.tail(time=2)._band_dim_values_map["time"]
              [12.0, 18.0]

              ```
            - Asking for more steps than there are keeps the whole axis:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.tail(time=99)._band_dim_values_map["time"]
              [0.0, 6.0, 12.0, 18.0]

              ```

        See Also:
            NetCDF.head: The first `n` steps instead.
            NetCDF.thin: Every `n`-th step.
        """
        _refuse_a_container(self._ds, "tail")
        wanted, count = _window_arguments(
            self._ds, indexers, indexers_kwargs, "tail", _DEFAULT_WINDOW, _whole_axis
        )
        return _windowed(self._ds, wanted, "tail", _tail_positions, default=count)

    def thin(self, indexers: Any = None, **indexers_kwargs: int) -> NetCDF:
        """Keep every `n`-th step along one or more band dimensions, starting at the first.

        Band dimensions only, as :meth:`head` is.

        Args:
            indexers: A step for every band dimension (`thin(2)`) or a mapping of them, as
                in xarray. `None` (default) reads the keywords instead.
            **indexers_kwargs: `dimension=n` pairs, `n` a whole number of at least 1. One
                of the two is needed: unlike `head` and `tail`, xarray gives `thin` no
                default step.

        Returns:
            NetCDF: A variable holding the kept steps, its coordinates cut to match.

        Raises:
            ValueError: No dimension is given, a dimension is not a band dimension, `n` is
                below 1 — xarray refuses a zero step the same way — or both spellings were
                used at once.
            TypeError: `n` is not an integer.

        Examples:
            - Every second step:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.thin(time=2)._band_dim_values_map["time"]
              [0.0, 12.0]

              ```
            - Unlike `head` and `tail`, `thin` needs a step — xarray gives it no default:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.thin()
              Traceback (most recent call last):
                  ...
              ValueError: thin() needs a step: thin(2) for every band dimension, or...

              ```

        See Also:
            NetCDF.head: A contiguous run from the start.
            NetCDF.isel: Any positions, a stepped slice included.
        """
        _refuse_a_container(self._ds, "thin")
        wanted, count = _window_arguments(
            self._ds, indexers, indexers_kwargs, "thin", None, _single_step
        )
        return _windowed(self._ds, wanted, "thin", _thin_positions, default=count)

    def drop_isel(self, **indexers: Any) -> NetCDF:
        """Drop steps by **position** along one or more band dimensions.

        The complement of :meth:`isel`: each selector is read as `isel` reads it — an
        integer array besides, since positions are usually computed rather than typed —
        and the positions it names are the ones removed.

        Args:
            **indexers: `dimension=selector` pairs — an index, a `list` or `tuple` of them,
                a numpy array of them, or a `slice`. `np.flatnonzero(...)` and
                `np.where(...)[0]` are how positions are usually built, so an integer array
                is read here even though :meth:`isel` refuses one.

        Returns:
            NetCDF: A variable without the dropped steps, its coordinates cut to match.

        Raises:
            ValueError: No indexers were given, a dimension is not a band dimension, or
                every step would be dropped — a variable with no bands cannot be built.
                Dropping *nothing* (an empty list, or a slice that selects nothing) is not
                an error: it keeps every step, as xarray's does.
            IndexError: A position is outside the dimension.
            TypeError: A selector is not one `isel` accepts.

        Examples:
            - Drop the second step:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.drop_isel(time=1)._band_dim_values_map["time"]
              [0.0, 12.0, 18.0]

              ```
            - Dropping nothing keeps everything, where `isel` of nothing is refused:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.drop_isel(time=[])._band_dim_values_map["time"]
              [0.0, 6.0, 12.0, 18.0]

              ```
            - A list drops several positions; a negative one counts from the end, as in `isel`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.drop_isel(time=[0, -1])._band_dim_values_map["time"]
              [6.0, 12.0]

              ```

        See Also:
            NetCDF.isel: Keep positions instead of dropping them.
            NetCDF.drop_sel: Drop by coordinate value.
        """
        nc = self._ds
        _refuse_a_container(nc, "drop_isel")
        if not indexers:
            raise ValueError(
                "drop_isel() requires at least one keyword argument, e.g. drop_isel(time=0)."
            )
        keep: dict[str, list[int]] = {}
        for dim_name, selector in indexers.items():
            _assert_band_dimension(nc, dim_name, caller="drop_isel")
            size = nc._band_dim_sizes[nc._band_dim_names.index(dim_name)]
            # Dropping nothing keeps everything, so an empty selector is a no-op copy
            # here where `isel` refuses it: `drop_isel(time=[])` is not a request for a
            # variable with no bands.
            dropped = set(
                _resolve_positional_indices(
                    _as_a_sequence_of_positions(selector),
                    size,
                    dim_name,
                    allow_empty=True,
                )
            )
            keep[dim_name] = [i for i in range(size) if i not in dropped]
        return _kept(nc, keep, "drop_isel")

    def drop_sel(self, *, errors: str = "raise", **labels: Any) -> NetCDF:
        """Drop steps by **coordinate value** along one or more band dimensions.

        The complement of :meth:`sel`. Each label is matched exactly, the way `sel`
        matches one without `method=`, so a CF time string resolves here as it does there.

        Args:
            errors: `"raise"` (default) refuses a label the dimension does not hold, as
                xarray does with `KeyError`; `"ignore"` skips it and drops the rest.
            **labels: `dimension=label` pairs — a coordinate value or a list of them.

        Returns:
            NetCDF: A variable without the dropped steps, its coordinates cut to match.

        Raises:
            ValueError: No labels were given, `errors` is unknown, a dimension is not a band
                dimension or has no coordinates, or every step would be dropped.
            KeyError: A label is not on the dimension and `errors="raise"`.

        Examples:
            - Drop one step by its stamp:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.drop_sel(time=6.0)._band_dim_values_map["time"]
              [0.0, 12.0, 18.0]

              ```
            - A label the dimension does not hold is refused, unless `errors="ignore"`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.drop_sel(time=[6.0, 99.0], errors="ignore")._band_dim_values_map["time"]
              [0.0, 12.0, 18.0]

              ```
            - A CF date string drops the step `sel` would select for it:

              ```python
              >>> from pyramids.netcdf import NetCDF
              >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
              >>> cube = nc["temperature"]
              >>> cube.drop_sel(time="2024-01-01T06:00")._band_dim_values_map["time"]
              [0.0, 12.0, 18.0]

              ```

        See Also:
            NetCDF.sel: Keep labels instead of dropping them.
            NetCDF.drop_isel: Drop by position, for a dimension with no coordinates.
        """
        nc = self._ds
        _refuse_a_container(nc, "drop_sel")
        if errors not in ("raise", "ignore"):
            raise ValueError(
                f"drop_sel() takes errors='raise' or 'ignore', got {errors!r}."
            )
        if not labels:
            raise ValueError(
                "drop_sel() requires at least one keyword argument, e.g. drop_sel(time=6)."
            )
        keep: dict[str, list[int]] = {}
        for dim_name, selector in labels.items():
            _assert_band_dimension(nc, dim_name, caller="drop_sel")
            if nc._band_dim_values_map.get(dim_name) is None:
                raise ValueError(
                    f"drop_sel() drops by coordinate value, and {dim_name!r} has no "
                    f"coordinates. Use drop_isel() to drop by position."
                )
            dropped = _labelled_positions(nc, dim_name, selector, errors)
            size = nc._band_dim_sizes[nc._band_dim_names.index(dim_name)]
            keep[dim_name] = [i for i in range(size) if i not in dropped]
        return _kept(nc, keep, "drop_sel")

    def sortby(self, dim: str, *, ascending: bool = True) -> NetCDF:
        """Reorder a band dimension by its own coordinate values.

        Each plane travels with its stamp, so the cells and the coordinates are reordered
        together. The sort is stable: equal stamps keep the order they were written when
        ascending, and `ascending=False` reverses the whole result, so they come back in
        the opposite order — which is what xarray's `sortby` does too.

        Args:
            dim: The band dimension to sort.
            ascending: `True` (default) for smallest first.

        Returns:
            NetCDF: A variable with `dim` in sorted order.

        Raises:
            ValueError: `dim` is not a band dimension, or has no coordinates to sort by.

        Examples:
            - Put an out-of-order axis back in order:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(3.0).reshape(3, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[12.0, 0.0, 6.0]),
              ... ).get_variable("t")
              >>> var.sortby("time")._band_dim_values_map["time"]
              [0.0, 6.0, 12.0]

              ```
            - Descending, each plane still on its own stamp:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[12.0, 0.0, 18.0, 6.0]),
              ... ).get_variable("t")
              >>> out = var.sortby("time", ascending=False)
              >>> out._band_dim_values_map["time"]
              [18.0, 12.0, 6.0, 0.0]
              >>> out.read_array().ravel().tolist()
              [2.0, 0.0, 3.0, 1.0]

              ```

        See Also:
            NetCDF.drop_duplicates: Remove a repeated stamp once the axis is in order.
            NetCDF.concat: Joining out-of-order parts is what leaves an axis unsorted.
        """
        nc = self._ds
        _refuse_a_container(nc, "sortby")
        coords = _coordinates_of(nc, dim, "sortby")
        order = [int(i) for i in np.argsort(np.asarray(coords), kind="stable")]
        if not ascending:
            order = order[::-1]
        return _kept(nc, {dim: order}, "sortby")

    def drop_duplicates(self, dim: str, *, keep: Any = "first") -> NetCDF:
        """Drop the steps whose stamp repeats one already on the dimension.

        What `concat` of overlapping parts leaves behind — the same stamp twice — and the
        caller's way to undo it. The surviving steps stay in the order they were written.

        Args:
            dim: The band dimension to de-duplicate.
            keep: `"first"` (default) keeps the first plane written for a repeated stamp,
                `"last"` the last one, and `False` drops every stamp that repeats, as
                pandas' `drop_duplicates` does. A repeated NaN stamp counts as a repeat,
                as it does in xarray, even though NaN equals nothing.

        Returns:
            NetCDF: A variable with each stamp at most once.

        Raises:
            ValueError: `dim` is not a band dimension or has no coordinates, `keep` is not
                one of the three, or `keep=False` would drop every step.

        Examples:
            - The stamp `6.0` twice, the first plane kept:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> var.drop_duplicates("time")._band_dim_values_map["time"]
              [0.0, 6.0, 12.0]

              ```
            - `keep="last"` keeps the later plane for the repeated stamp:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> var.drop_duplicates("time", keep="last").read_array().ravel().tolist()
              [0.0, 2.0, 3.0]

              ```

        See Also:
            NetCDF.concat: What leaves a repeated stamp when two parts overlap.
            NetCDF.sortby: Put the axis in order first, if it is not.
        """
        _refuse_a_container(self._ds, "drop_duplicates")
        if keep not in ("first", "last") and keep is not False:
            raise ValueError(
                f"drop_duplicates() takes keep='first', 'last' or False, got {keep!r}."
            )
        nc = self._ds
        coords = [
            _stamp_key(stamp) for stamp in _coordinates_of(nc, dim, "drop_duplicates")
        ]
        counts: dict[Any, int] = {}
        for stamp in coords:
            counts[stamp] = counts.get(stamp, 0) + 1
        if keep is False:
            positions = [i for i, stamp in enumerate(coords) if counts[stamp] == 1]
        else:
            chosen: dict[Any, int] = {}
            for i, stamp in enumerate(coords):
                if keep == "last" or stamp not in chosen:
                    chosen[stamp] = i
            positions = sorted(chosen.values())
        return _kept(nc, {dim: positions}, "drop_duplicates")

    def squeeze(self, dim: str | None = None) -> NetCDF:
        """Drop the band dimensions of length one.

        No cell changes and no plane is reordered — a length-one axis contributes nothing
        to the band count — but the result is a new raster holding a copy of the bands, as
        every derived variable here is; xarray's `squeeze` is a view. **Band dimensions
        only:** xarray's also drops a length-one *spatial* axis, and a raster cannot lose
        one; a one-row raster keeps its row.

        Args:
            dim: The one dimension to drop, which must be length one. `None` (default) drops
                every length-one band dimension.

        Returns:
            NetCDF: A variable without those dimensions.

        Raises:
            ValueError: `dim` is not a band dimension, or is longer than one.

        Examples:
            - A single step left by `isel`, dropped:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.isel(time=[1]).squeeze()._band_dim_names
              ()

              ```
            - A dimension longer than one cannot be squeezed:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.squeeze("time")
              Traceback (most recent call last):
                  ...
              ValueError: squeeze() drops a dimension of length one, and 'time' has length 4...

              ```

        See Also:
            NetCDF.expand_dims: The inverse — add a length-one dimension.
        """
        nc = self._ds
        _refuse_a_container(nc, "squeeze")
        names = list(nc._band_dim_names)
        sizes = list(nc._band_dim_sizes)
        if dim is not None:
            _assert_band_dimension(nc, dim, caller="squeeze")
            length = sizes[names.index(dim)]
            if length != 1:
                raise ValueError(
                    f"squeeze() drops a dimension of length one, and {dim!r} has "
                    f"length {length}. Select one step first with isel({dim}=[0])."
                )
            gone = {dim}
        else:
            gone = {name for name, size in zip(names, sizes) if size == 1}
        if not gone:
            # Nothing to drop: reading and rebuilding every band would copy the whole cube
            # for a call that changes nothing, and would mark the result as rebuilt in
            # memory, forfeiting its lazy read. xarray's squeeze is a free view here.
            return _rewrapped(nc)
        values_map = {
            name: stamps
            for name, stamps in nc._band_dim_values_map.items()
            if name not in gone
        }
        return _relabelled(
            nc,
            tuple(name for name in names if name not in gone),
            tuple(size for name, size in zip(names, sizes) if name not in gone),
            values_map,
        )

    def expand_dims(self, dim: str, value: Any = None) -> NetCDF:
        """Add a band dimension of length one, outermost.

        The step that lifts a raster into a cube before :meth:`NetCDF.concat` joins it to
        others along the new dimension. No cell changes, though the bands are copied into
        the new variable, as :meth:`squeeze` copies them.

        Args:
            dim: The new dimension's name, which must not already be a band dimension or a
                spatial axis of this variable or of the store it came from.
            value: Its single coordinate value. `None` (default) gives the dimension no
                coordinates at all, as xarray's `expand_dims("member")` does — a stamp
                nothing said is not invented here. One value, not a list: xarray's
                `expand_dims(member=[0.0, 1.0])` builds a length-two axis, and the way to
                that here is a plane each, joined with `NetCDF.concat`.

        Returns:
            NetCDF: A variable with `dim` first, length one.

        Raises:
            ValueError: `dim` is already a band dimension, or names a spatial axis.
            TypeError: `value` is a list, tuple, set or array rather than one value.

        Examples:
            - A flat raster lifted onto a `time` axis:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> flat = NetCDF.from_array(
              ...     np.ones((1, 1)),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ... ).get_variable("t")
              >>> lifted = flat.expand_dims("time", 6.0)
              >>> lifted._band_dim_names, lifted._band_dim_values_map["time"]
              (('time',), [6.0])

              ```
            - With no value the dimension carries no coordinates, as xarray's does:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> flat = NetCDF.from_array(
              ...     np.ones((1, 1)),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ... ).get_variable("t")
              >>> lifted = flat.expand_dims("member")
              >>> lifted._band_dim_names, lifted._band_dim_values_map["member"]
              (('member',), None)

              ```
            - Two flat rasters lifted onto `time`, then joined into a cube:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo_ref = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> first = NetCDF.from_array(np.ones((1, 1)), geo_ref=geo_ref, variable_name="t")
              >>> second = NetCDF.from_array(np.zeros((1, 1)), geo_ref=geo_ref, variable_name="t")
              >>> a = first.get_variable("t").expand_dims("time", 0.0)
              >>> b = second.get_variable("t").expand_dims("time", 6.0)
              >>> NetCDF.concat([a, b], "time").get_variable("t").read_array().ravel().tolist()
              [1.0, 0.0]

              ```

        See Also:
            NetCDF.squeeze: The inverse — drop a length-one dimension.
            NetCDF.concat: Join the lifted rasters along the new dimension.
        """
        nc = self._ds
        _refuse_a_container(nc, "expand_dims")
        if dim in nc._band_dim_names:
            raise ValueError(
                f"expand_dims() adds a new dimension, and {dim!r} is already one of "
                f"{tuple(nc._band_dim_names)}."
            )
        spatial = _spatial_dimension_names(nc)
        if dim in spatial:
            raise ValueError(
                f"expand_dims() adds a band dimension, and {dim!r} is a spatial axis of "
                f"this variable or of the store it came from. A raster's rows and columns "
                f"are its grid, not a band axis, and a band dimension under that name "
                f"collides with the store's own when the result is written back."
            )
        if isinstance(value, (list, tuple, set, np.ndarray)):
            raise TypeError(
                f"expand_dims() adds a dimension of length one, so it takes one "
                f"coordinate value, not {value!r}. Build each plane and join them with "
                f"NetCDF.concat() for a longer axis."
            )
        values_map = {
            dim: None if value is None else [value],
            **nc._band_dim_values_map,
        }
        return _relabelled(
            nc,
            (dim, *nc._band_dim_names),
            (1, *nc._band_dim_sizes),
            values_map,
        )

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
        """Read a windowed `(variable, time, bbox)` slice of a gridded cube.

        Reads only the requested window from a CF/GeoZarr `(time, y, x[, …])`
        multidimensional store — local or remote — without materialising the
        whole variable, and returns a georeferenced single-variable
        :class:`~pyramids.netcdf.NetCDF` ready for `to_file` / `to_cog` /
        `to_crs` / `crop` (a `Dataset` subclass, so existing
        `isinstance(result, Dataset)` checks keep working).

        Designed for huge cloud cubes (e.g. the NWM retrospective
        `ldasout.zarr`, an 18 TiB `(128568, 3840, 4608)` store) opened
        anonymously via :class:`~pyramids.base.remote.CloudConfig`; only the
        sliced cells are fetched. The output CRS is the variable's own grid
        mapping (read from the multidimensional array), so a Lambert Conformal
        Conic store stays on its native grid.

        Args:
            variable: Data-variable name in the store (e.g. `"ACCET"`).
            time: Timestep selector along the time dimension. An `int` picks
                one step (one output band); a `(start, stop)` tuple or
                `slice` picks a half-open index range (one band per step);
                `None` is allowed only when the time dimension has length 1.
                Selection is by **integer index** — date/label selection needs
                the store to expose CF time `units`, which many Zarr stores do
                not surface through GDAL, so use indices for those.
            bbox: `(min_x, min_y, max_x, max_y)` crop window in `crs`.
                `None` keeps the full grid. The box is reprojected onto the
                store's native grid (so a lon/lat box over a projected grid is
                handled) honouring the variable's grid mapping.
            crs: CRS of `bbox` — EPSG int, `"EPSG:4326"`, or a WKT/PROJ
                string. Defaults to `4326` (lon/lat). Ignored when `bbox` is
                `None`.
            densify: Points per bbox edge used when reprojecting the box onto a
                projected grid, so the envelope encloses the curved boundary
                (conservative over-cover). Defaults to `25`.
            y_dim: Name of the `y` (row) dimension. Defaults to `None` —
                auto-detected from CF axis / `standard_name` / `units`
                attributes, then well-known names (`y`/`lat`/…), then the
                trailing two dims. Pass it (with `x_dim`) to override when the
                spatial axes can't be inferred.
            x_dim: Name of the `x` (column) dimension. `None` auto-detects as
                for `y_dim`. Pass both `y_dim` and `x_dim` together.
            **dims: Index selector for any extra non-spatial dimension (e.g.
                `vis_nir=0`, `soil_layers_stag=2`). Required for every such
                dimension whose length is > 1, **including a layer dim
                interleaved between `y` and `x`** (e.g. NWM `SOIL_M` is
                `(time, y, soil_layers_stag, x)` — pass `soil_layers_stag=0`).
                A key that is not a selectable non-spatial dimension is an error.

        Returns:
            NetCDF: A georeferenced single-variable raster on the store's native CRS —
            one band per selected timestep, with the native no-data value applied.
            The window holds the stored values and the variable's CF packing
            (`scale_factor` / `add_offset`) is carried onto every band, so a default
            `read_array` of the result answers in physical units.

        Raises:
            ValueError: When the store is not multidimensional; when `variable`
                is absent or has fewer than two dimensions; when a spatial axis
                has no 1-D coordinate variable; when a non-spatial dimension of
                length > 1 is not selected, or a `**dims` key / index is
                invalid; or when the bbox selects no cells.

        Note:
            For a purely 2-D `(y, x)` variable there is no non-spatial axis, so
            `time` and `**dims` are no-ops (the whole grid, optionally bbox-
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
                cast("tuple[float, float, float, float]", tuple(bbox)),
                crs,
                srs,
                densify,
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
        ds = Dataset.from_array(
            band_first,
            no_data_value=no_data if no_data is not None else DEFAULT_NO_DATA_VALUE,
            geo_ref=GeoReference(geo=geo, epsg=4326),
        )
        # `ReadAsArray` on an MDArray answers in stored counts, and `from_array`
        # declares no packing, so without this the windowed shortcut returned raw
        # values where the full-read path it stands in for returns physical ones.
        for index in range(1, ds.raster.RasterCount + 1):
            carry_band_packing(md_arr, ds.raster.GetRasterBand(index))
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
        q: float | None = None,
    ) -> NetCDF:
        """Reduce along a named dimension and return a new NetCDF.

        Collapses or groups one non-spatial dimension (`time`, `pressure_level`, `depth`, an
        ensemble member, ...), leaving the other dimensions and their coordinates, the CRS
        and the grid untouched. The work is done with numpy, streamed through dask when dask is
        installed and the variable is read straight from its file — not a cut, a reprojection, an
        operator result or a variable changed in place (`fill(..., inplace=True)`), which are
        read whole; xarray is not needed.

        On a **container**, every gridded variable that has `dim` is reduced, the gridded
        variables without it are carried over, and the result is a new container.
        Non-spatial auxiliary variables (no `y` / `x` axes, e.g. ERA5's `number`) are
        carried through unchanged (#513) — except an auxiliary variable that itself spans
        `dim`, which is dropped with a warning, since carrying it verbatim would leave an
        inconsistent `dim` length.

        On a **variable** — `nc.get_variable(name)`, a selection, or an operator result —
        that variable alone is reduced and the result is a variable, so
        `nc.get_variable("t").reduce("time")` holds the same cells as
        `nc.reduce("time").get_variable("t")`. A variable with no name of its own (an
        operator result) comes back named `"variable"`.

        Args:
            dim: Name of the non-spatial dimension to reduce. Must be one of a variable's
                band dimensions (as exposed by `sel`); the spatial axes are not reducible
                here.
            how: The reduction. The statistics are `"mean"`, `"sum"`, `"min"`, `"max"`,
                `"std"`, `"var"`, `"median"`, `"prod"` and `"quantile"` (which needs `q`).
                `"count"` answers the number of valid cells as `int64`, and `"all"` /
                `"any"` answer a `uint8` 0/1 flag whether every / some valid cell is
                non-zero — the band format the comparison operators produce.
            groupby: Controls collapse vs. windowed reduction:

                - `None` (default): collapse `dim` entirely (it is removed from the output).
                - a sequence of per-index labels (length = the size of `dim`): reduce each
                  group of equal labels; `dim` is coarsened to one slice per distinct label,
                  in first-appearance order.
                - a pandas offset alias (e.g. `"1MS"`, `"1D"`, `"YS"`): group `dim` by
                  calendar window. Only valid when `dim` carries a decodable CF time
                  coordinate.
            skipna: When `True` (default), gaps — the declared no-data value and NaN — are
                skipped. The statistics then answer float64, and a slice with no valid cell
                holds the variable's no-data value as the read holds it — unpacked, for a
                CF-packed variable — or NaN when it declares none; `all` /
                `any` treat a gap as neutral and answer `255`, their no-data value, for a
                slice with no valid cell. When `False`, the raw stored values are reduced,
                sentinel included, and a statistic keeps the dtype numpy gives it — the
                `min` of an `int16` band stays `int16`. `count` counts valid cells either
                way. Three answers on gaps differ from xarray's: a slice with no valid cell
                makes `sum` and `prod` no-data where xarray answers `0.0` and `1.0`, and
                `all` / `any` `255` where xarray answers `True`; and `any` skips NaN, so
                `[0, NaN, 0]` answers `0` where xarray, reading NaN as true, answers `True`.
            q: The quantile for `how="quantile"`, one number in `[0, 1]`, using numpy's
                default linear interpolation. Required for `"quantile"` and refused for
                every other `how`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with `dim`
            removed (`groupby=None`) or coarsened (windowed). A windowed dimension with a
            numeric coordinate labels each output slice with the first coordinate value of
            its window. The result declares the variable's no-data value in the units the
            reduction read — for a CF-packed variable the unpacked
            `_FillValue * scale_factor + add_offset`, not the stored `_FillValue` — except that
            `count` declares none and `all` / `any` declare `255`.

        Raises:
            ValueError: `how` is unknown; `q` is missing, not a single number in `[0, 1]`,
                or given with a `how` other than `"quantile"`; the container has no data
                variables; `dim` is not a band dimension of any gridded variable (or of this
                variable, or this variable has none); a frequency `groupby` is given but
                `dim` has no decodable time coordinate; or the grouping does not cover `dim`
                exactly.

        Warns:
            UserWarning: A container's auxiliary variable spans `dim` and is dropped.

        Examples:
            - The median and the number of valid steps, on a variable with one gap:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, -9999.0, 10.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=-9999.0,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> float(var.reduce("time", "median").read_array()[0, 0])
              2.0
              >>> counts = var.reduce("time", "count")
              >>> int(counts.read_array()[0, 0]), counts.no_data_value
              (3, (None,))

              ```
            - A quantile needs `q`, and nothing else accepts one:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, 10.0]).reshape(3, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> float(var.reduce("time", "quantile", q=0.25).read_array()[0, 0])
              1.5
              >>> var.reduce("time", "mean", q=0.5)
              Traceback (most recent call last):
                ...
              ValueError: q= is only meaningful with how='quantile', got how='mean' and q=0.5.

              ```
            - Group a container's steps by label, one output step per distinct label:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 24.0, 30.0]),
              ... )
              >>> days = nc.reduce("time", "max", groupby=["day1", "day1", "day2", "day2"])
              >>> days.get_variable("t").read_array().ravel().tolist()
              [1.0, 3.0]
              >>> days.get_variable("t")._band_dim_values_map["time"]
              [0.0, 24.0]

              ```
        """
        # Local import breaks the netcdf.py <-> engines.selection import cycle
        # (netcdf.py imports this module at top level for wiring); the reducer registries
        # are module-level there, shared with the reduce helpers.
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        nc = self._ds
        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        op = _Reduction(
            how=how,
            groups=lambda: nc._resolve_group_positions(dim, groupby),
            skipna=skipna,
            q=q,
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def coarsen(
        self,
        dim: str,
        window: int,
        *,
        how: str = "mean",
        boundary: str = "exact",
        skipna: bool = True,
        q: float | None = None,
    ) -> NetCDF:
        """Block-aggregate a non-spatial dimension into windows of `window` steps.

        The positional sibling of `reduce(groupby=...)`: consecutive runs of `window` steps
        along `dim` are each reduced to one step, so a 24-step hourly axis coarsened by 6
        becomes 4 steps. Each output step is labelled with the **mean** of the coordinate
        values its window holds, as xarray's `coarsen(...).<how>()` labels it — where
        `reduce(groupby=...)` labels a window with its first member. On a variable, a
        dimension without coordinate values stays without them. A container's store numbers
        such a dimension `0, 1, ...`, so its windows are labelled with the mean of those
        positions (`0.5, 2.5` for a window of 2). A dimension whose values are not all
        numbers labels each window with its first member instead, but a text label (a time
        stamp string, say) cannot be stored, so coarsening such a dimension raises
        `ValueError`.

        Works on a container, reducing every variable that has `dim`, and on a single
        variable, returning a variable — the same split `reduce` makes, auxiliary variables
        included.

        `boundary` decides what happens when `window` does not divide the length of `dim`,
        with xarray's vocabulary:

        - `"exact"` (default): refuse.
        - `"trim"`: drop the trailing steps that do not fill a window.
        - `"pad"`: reduce them as a shorter last window. The window is padded with NaN, as
          xarray pads it, so under `skipna` the padding is skipped and the window is
          reduced over its real steps. With `skipna=False` a statistic of that window is
          NaN, `count` still counts only its real cells, and `all` / `any` read the padding
          as true. The window is labelled with the mean of its real steps' coordinates
          either way, where xarray labels it NaN under `skipna=False`.

        Args:
            dim: The non-spatial dimension to coarsen.
            window: Steps per window, an integer of at least 1. Anything `operator.index()`
                accepts works, except a boolean.
            how: The reduction applied to each window — any `how` that `reduce` accepts.
            boundary: `"exact"`, `"trim"` or `"pad"`.
            skipna: Whether gaps are skipped, as `reduce` documents it.
            q: The quantile, for `how="quantile"`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with `dim`
            shortened to one step per window and the other dimensions kept. The no-data
            value is declared as `reduce` declares it.

        Raises:
            TypeError: `window` is not an integer, or is a boolean.
            ValueError: `how` or `q` is refused as `reduce` refuses them (checked before
                `window`); `window` is below 1; `boundary` is unknown; the container has no
                data variables; `dim` is not a band dimension of any gridded variable (or of
                this variable, or this variable has none); `boundary="exact"` and `window`
                does not divide the length of `dim`; `boundary="trim"` and `window` is
                longer than `dim`, which would leave no steps (xarray returns an empty
                result there; a variable with no bands cannot be built); or a window label
                is text.

        Warns:
            UserWarning: A container's auxiliary variable spans `dim` and is dropped. The
                message names `coarsen()`.

        Examples:
            - Six-hourly steps averaged into twelve-hourly ones, labelled at the midpoint:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... )
              >>> half_days = nc.coarsen("time", 2).get_variable("t")
              >>> half_days.read_array().ravel().tolist()
              [0.5, 2.5]
              >>> half_days._band_dim_values_map["time"]
              [3.0, 15.0]

              ```
            - A window that does not divide the axis needs a boundary; `"pad"` reduces the
              steps left over as a shorter last window:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... )
              >>> nc.coarsen("time", 3)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                ...
              ValueError: cannot coarsen 'time' of length 4 into windows of 3
              >>> padded = nc.coarsen("time", 3, boundary="pad").get_variable("t")
              >>> padded.read_array().ravel().tolist()
              [1.0, 3.0]
              >>> padded._band_dim_values_map["time"]
              [6.0, 18.0]

              ```
            - Count the valid steps per window of a variable, trimming the step left over:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, -9999.0, 4.0, 5.0]).reshape(5, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=-9999.0,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0, 3.0, 4.0]),
              ... ).get_variable("t")
              >>> counts = var.coarsen("time", 2, how="count", boundary="trim")
              >>> counts.read_array().ravel().tolist()
              [2, 1]
              >>> counts._band_dim_values_map["time"], counts.no_data_value
              ([0.5, 2.5], (None, None))

              ```
            - A dimension without coordinates: a variable keeps it unlabelled, while a
              container's store numbers it, so its windows are labelled with the mean position:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... )
              >>> var = nc.get_variable("t")
              >>> change = var.isel(time=slice(2, 4)) - var.isel(time=slice(0, 2))
              >>> change.coarsen("time", 2)._band_dim_values_map
              {'time': None}
              >>> numbered = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=None),
              ... )
              >>> numbered.get_dimension_values("time").tolist()
              [0, 1, 2, 3]
              >>> numbered.coarsen("time", 2).get_dimension_values("time").tolist()
              [0.5, 2.5]

              ```
        """
        # Local import breaks the netcdf.py <-> engines.selection import cycle.
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        nc = self._ds
        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        length = _check_window(window, caller="coarsen")
        if boundary not in _BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {list(_BOUNDARIES)}, got {boundary!r}."
            )
        is_variable = _reduces_as_a_variable(nc)
        size = _band_dimension_size(nc, dim, is_variable=is_variable)
        resized, positions = _coarsen_windows(dim, size, length, boundary)
        op = _Reduction(
            how=how,
            groups=lambda: positions,
            skipna=skipna,
            q=q,
            caller="coarsen",
            resize=resized,
            window_mean_coords=True,
        )
        if is_variable:
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def groupby_bins(
        self,
        dim: str,
        bins: int | Sequence[float],
        how: str = "mean",
        *,
        right: bool = True,
        include_lowest: bool = False,
        skipna: bool = True,
        q: float | None = None,
    ) -> NetCDF:
        """Cut a band dimension's coordinates into value intervals and reduce each bin.

        The value sibling of `reduce(groupby=...)`: where `reduce` groups by equal labels or a
        calendar window, this groups by which **interval** of `bins` each coordinate falls in
        — xarray's `groupby_bins`. It is sugar over the same group-reduce path, so `how`,
        `skipna` and `q` mean exactly what they do on `reduce`.

        Each non-empty bin becomes one output slice, labelled with the bin's **left edge**
        (numeric and monotonic, so the result is still addressable with `sel` / `isel`); for
        contiguous bins the right edge is the next bin's left edge. Bins come out in ascending
        edge order whatever order the axis runs in.

        Args:
            dim: The band dimension to bin. Must carry **numeric** coordinates — a text axis or
                a variable's coordinate-less axis is refused — and must not be a spatial axis.
            bins: An `int` number of equal-width bins spanning the data range — right-closed
                with the lowest edge included, so the first label is the data minimum — or an
                explicit sequence of strictly increasing edges (`n` edges make `n - 1` bins),
                whose closure follows `right` / `include_lowest`. A `float`, a `bool`, or any
                other type is refused.
            how: The reduction, as on `reduce` (`"mean"`, `"sum"`, `"min"`, `"max"`, `"std"`,
                `"var"`, `"median"`, `"prod"`, `"quantile"`, `"count"`, `"all"`, `"any"`).
            right: Whether the **explicit-edge** intervals are right-closed `(a, b]` (the
                default, as `pandas.cut`) or left-closed `[a, b)`. An `int` `bins` is always
                right-closed.
            include_lowest: Whether the very first **explicit** edge is included in the first
                bin; an `int` `bins` always includes it.
            skipna: Whether gaps are skipped, as on `reduce`.
            q: The quantile for `how="quantile"`, refused for every other `how`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with `dim` reduced
            to one slice per non-empty bin and its coordinate holding those bins' left edges.

        Raises:
            ValueError: `dim` is not a band dimension, is spatial, is not a dimension of the
                container, or carries non-numeric / no coordinates / a `NaN`; `bins` is not an
                `int` count or a sequence of edges; an `int` `bins` is given for a constant
                axis (no range to divide — pass explicit edges); `how` / `q` are invalid (as on
                `reduce`); the explicit edges are fewer than two or not strictly increasing; or
                a coordinate falls outside every bin.

        Notes:
            Deliberate differences from xarray: an **empty bin is dropped** (xarray keeps it
            as NaN), consistent with how `reduce`'s frequency grouping skips empty windows; the
            binned coordinate is the **left edge**, not a `pandas.Interval` (which a GDAL band
            cannot hold); and a coordinate outside every bin is **refused** rather than dropped
            silently, so widen the bins (or pass `include_lowest=True`) to cover the axis. When
            an empty bin is dropped, the left-edge labels alone no longer reconstruct each
            bin's width (the gap is invisible from the axis); carrying the right edge as an
            attribute is a documented follow-up.

        Examples:
            - Average four levels into two 500-wide bins, labelled by their left edges:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> cube = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="level", values=[100.0, 300.0, 600.0, 900.0]),
              ... )
              >>> binned = cube.groupby_bins("level", [0, 500, 1000], "mean").get_variable("t")
              >>> binned._band_dim_values_map["level"]
              [0.0, 500.0]
              >>> binned.read_array().ravel().tolist()
              [0.5, 2.5]

              ```

        See Also:
            NetCDF.reduce: group by equal labels or a calendar window.
            NetCDF.coarsen: reduce fixed-size positional windows.
        """
        # Local import breaks the netcdf.py <-> engines.selection import cycle, as `reduce` does.
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        nc = self._ds
        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        coords = _bin_coordinates(nc, dim)
        edges, codes = _bin_membership(
            coords, bins, right=right, include_lowest=include_lowest
        )
        outside = int(np.isnan(codes).sum())
        if outside:
            # The reduce path groups must cover the axis exactly; a coordinate in no bin has no
            # group, so rather than drop data silently, refuse and name which coordinates miss.
            # `include_lowest` only rescues a coordinate equal to the lowest edge, so only
            # suggest it when that is what happened.
            outside_vals = sorted(float(v) for v in coords[np.isnan(codes)])
            hint = (
                " (or pass include_lowest=True to include the lowest edge)"
                if not include_lowest and float(edges[0]) in outside_vals
                else ""
            )
            raise ValueError(
                f"groupby_bins() found {outside} {dim!r} coordinate(s) outside every bin "
                f"{[float(e) for e in edges]}: {outside_vals}. Widen the bins{hint} so every "
                "coordinate falls in one."
            )
        codes = codes.astype(int)
        positions: list = []
        left_edges: list = []
        for index in range(len(edges) - 1):
            members = np.nonzero(codes == index)[0]
            if members.size:
                positions.append(members)
                left_edges.append(float(edges[index]))
        op = _Reduction(
            how=how,
            groups=lambda: positions,
            skipna=skipna,
            q=q,
            caller="groupby_bins",
            group_coords=left_edges,
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def rolling(
        self,
        dim: str,
        window: int,
        *,
        how: str = "mean",
        center: bool = False,
        min_periods: int | None = None,
        q: float | None = None,
    ) -> NetCDF:
        """Reduce a moving window along a non-spatial dimension, keeping its length.

        Each step of `dim` becomes the reduction of the `window` steps that end at it, or, with
        `center=True`, that are centred on it — an even window reaching one step further back
        than forward. A window is cut where it would run off the axis, so the steps at the start
        of `dim` own fewer cells, and with `center=True` those at its end do too. Gaps (the
        declared no-data value and NaN) are always skipped, and a step whose window holds fewer
        than `min_periods` valid cells is no-data, which is what makes those short-window steps
        no-data under the default `min_periods` of a whole window. The dimension keeps its length
        and coordinates, so the result selects exactly as the source does.

        Every step holds exactly what a `reduce` over its own window holds, for every `how`. The
        window placement, the short edges and `min_periods` follow xarray's
        `rolling(...).mean()` value for value — a centred even window, and a window longer than
        the axis included; what differs is only how a short window is *marked*, since a band
        declares one no-data value rather than carrying NaN (see `how` below).

        Works on a container, rolling every variable that has `dim`, and on a single variable,
        returning a variable. A container's auxiliary variables are carried over, those that
        span `dim` included, since its length does not change.

        **Cost.** Every step's window is reduced in full, so the work is the axis length times
        the window times the cell count — a window of 100 costs about ten times a window of 10,
        not the same. A 200-step 200x200 in-memory cube measures 1.2 s at `window=10` and 11 s
        at `window=100` here, so a long window over a large grid is correspondingly slow.
        Reducing one strided view of all the windows at once buys part of that back — 6 s for
        the same call — but it holds every window's temporaries at once, 8.3 GB of peak memory
        against 0.6 GB, and it would leave the streamed (dask) path reducing a padded overlap
        graph instead of the store's own chunks; so the windows are taken one at a time on
        purpose.

        Args:
            dim: The non-spatial dimension to roll along.
            window: Steps per window, an integer of at least 1. A window longer than the axis
                is allowed; every step then owns the steps the axis has around it.
            how: The reduction over each window — any `how` that `reduce` accepts. A statistic
                answers float64 and declares the variable's no-data value, or NaN when it declares
                none, since the short windows are gaps; `count` answers `int64` and declares `-1`,
                the value of a window with too few valid cells; `all` / `any` answer a `uint8`
                flag and declare `255`.
            center: Centre each window on its step instead of ending it there.
            min_periods: The valid cells a window needs before its step holds a value, an
                integer between 1 and `window`; `None` (default) asks for a whole window.
                xarray accepts a `min_periods` above `window` and answers all NaN; it is
                refused here, since no window could meet it.
            q: The quantile, for `how="quantile"`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with `dim` and every
            other dimension unchanged in length and coordinates.

        Raises:
            TypeError: `window` or `min_periods` is not an integer, or is a boolean.
            ValueError: `how` or `q` is refused as `reduce` refuses them; `window` is below 1;
                `min_periods` is below 1 or above `window`; the container has no data
                variables; or `dim` is not a band dimension of any gridded variable (or of this
                variable, or this variable has none).

        Examples:
            - A trailing mean over two steps. The first step owns one cell, too few by default, so
              it holds the declared no-data value (`from_array` declares `-9999.0`):

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 3.0, 5.0, 7.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.rolling("time", 2).read_array().ravel().tolist()
              [-9999.0, 2.0, 4.0, 6.0]
              >>> var.rolling("time", 2, min_periods=1).read_array().ravel().tolist()
              [1.0, 2.0, 4.0, 6.0]

              ```
            - A centred window skips a gap, and keeps the dimension's stamps:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.array([1.0, -9999.0, 5.0, 7.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=-9999.0,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... )
              >>> smooth = nc.rolling("time", 3, center=True, min_periods=1).get_variable("t")
              >>> smooth.read_array().ravel().tolist()
              [1.0, 3.0, 6.0, 6.0]
              >>> smooth._band_dim_values_map["time"]
              [0.0, 6.0, 12.0, 18.0]

              ```
            - Count the valid steps in each window; a window with too few is `-1`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, np.nan, 5.0, 7.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> counts = var.rolling("time", 2, how="count")
              >>> counts.read_array().ravel().tolist(), counts.no_data_value[0]
              ([-1, -1, -1, 2], -1)

              ```
        """
        # Local import breaks the netcdf.py <-> engines.selection import cycle.
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _REDUCERS

        nc = self._ds
        _check_how(how, {*_REDUCERS, *_COUNTING_REDUCERS})
        q = _check_quantile(how, q)
        length = _check_window(window, caller="rolling")
        needed = _check_min_periods(min_periods, length)
        op = _Rolling(
            window=length, how=how, center=bool(center), min_periods=needed, q=q
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def diff(self, dim: str, n: int = 1, *, label: str = "upper") -> NetCDF:
        """Difference neighbouring steps along a non-spatial dimension.

        Each step becomes the difference between it and the step before it, so the dimension
        loses one step, and `n` asks for that `n` times over — a second difference is the
        difference of the differences. A difference that meets a gap (the declared no-data value
        or NaN) is a gap. `dim` keeps the stamps of the steps the differences are labelled with:
        the later of each pair by default, as xarray labels them.

        Works on a container, differencing every variable that has `dim`, and on a single
        variable, returning a variable. A container's auxiliary variable spanning `dim` is
        dropped with a warning, since the dimension gets shorter — except for `n=0`, the
        identity, which keeps everything.

        The values match xarray's `diff` for every `n`, and so do the stamps under
        `label="upper"`. Under `label="lower"` they match at `n=1` only: xarray passes `label`
        to the first of its `n` passes and lets the rest fall back to `"upper"`, so its
        `n=2, label="lower"` axis is the second-to-second-last stamp, while here each difference
        keeps the first of the `n + 1` steps it is built from.

        Args:
            dim: The non-spatial dimension to difference along.
            n: The order, an integer of at least 0 and below the length of `dim`. `0` is the
                identity, values, dtype and declared no-data value alike — every other order
                answers float64 for a band whose gaps have to be skipped. An `n` equal to the
                length is refused, where xarray returns an empty result — a variable with no
                bands cannot be built.
            label: `"upper"` (default) labels each difference with the last of the `n + 1` steps
                it is built from, `"lower"` with the first. At `n=1` those are the later and the
                earlier of its pair.

        Returns:
            NetCDF: A container for a container, a variable for a variable, `dim` shorter by
            `n` steps. A float band, or an integer band declaring a no-data value, answers
            float64 and declares that value (NaN when it declares none); an integer band
            declaring none answers in numpy's own type for the difference, as xarray does —
            **which means a narrow one wraps**: `int8` `[-100, 100, 0, 0]` differences to
            `[-56, -100, 0]`, since the true `200` does not fit, exactly as `numpy.diff` and
            `xarray.DataArray.diff` answer it. Declare a no-data value, or read the band as a
            wider type, if the differences can leave its range.

        Raises:
            TypeError: `n` is not an integer, or is a boolean.
            ValueError: `n` is negative or not below the length of `dim`; `label` is neither
                `"upper"` nor `"lower"`; the container has no data variables; or `dim` is not a
                band dimension of any gridded variable (or of this variable, or this variable
                has none).

        Examples:
            - The change between steps, labelled with the later step:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 4.0, 9.0, 16.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> change = var.diff("time")
              >>> change.read_array().ravel().tolist()
              [3.0, 5.0, 7.0]
              >>> change._band_dim_values_map["time"]
              [6.0, 12.0, 18.0]

              ```
            - The second difference, and the leading labels:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 4.0, 9.0, 16.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.diff("time", 2).read_array().ravel().tolist()
              [2.0, 2.0]
              >>> var.diff("time", label="lower")._band_dim_values_map["time"]
              [0.0, 6.0, 12.0]

              ```
            - An integer band that declares no no-data value keeps its own type, and a
              second-order `"lower"` label keeps the earliest step behind each difference:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1, 4, 9, 16], dtype="int16").reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=None,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> change = var.diff("time")
              >>> change.read_array().ravel().tolist(), change.read_array().dtype.name
              ([3, 5, 7], 'int16')
              >>> change.no_data_value
              (None, None, None)
              >>> var.diff("time", 2, label="lower")._band_dim_values_map["time"]
              [0.0, 6.0]

              ```
        """
        nc = self._ds
        order = _check_order(n)
        if label not in _DIFF_LABELS:
            raise ValueError(
                f"label must be one of {list(_DIFF_LABELS)}, got {label!r}."
            )
        op = _Diff(n=order, label=label)
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def rank(self, dim: str, *, pct: bool = False) -> NetCDF:
        """Rank each pixel's values along a band dimension, ties averaged.

        The ordinal position `1..N` of every step among the others along `dim`, tied values sharing
        the average of their positions, and gaps (the declared no-data value or NaN) excluded from
        the ranking and returned as no-data. `pct=True` returns the rank divided by the count of
        valid steps, in `(0, 1]`. The excluded gaps reuse the variable's own no-data value as their
        sentinel, so a no-data value that itself falls inside the rank domain (`1..N`, or `(0, 1]`
        when `pct=True`) would be indistinguishable from a real rank on read-back; real sentinels
        such as `-9999` lie outside that range, so this is a caveat, not a defect. Matches
        `xarray.Dataset.rank`. Band dimensions only — a spatial axis is refused (ranking a
        georeferenced axis is meaningless).

        Works on a container, ranking every variable that has `dim`, and on a single variable.

        Args:
            dim: The band dimension to rank along.
            pct: Return the rank as a fraction of the valid count rather than the `1..N` position.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, the dimension
            unchanged.

        Raises:
            ValueError: `dim` is not a band dimension (spatial or unknown), or the variable has no
                band dimensions.

        Examples:
            - Rank a time series (ties averaged):

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([30.0, 10.0, 10.0, 20.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0, 3.0]),
              ...     no_data_value=None,
              ... ).get_variable("t")
              >>> var.rank("time").read_array().ravel().tolist()
              [4.0, 1.5, 1.5, 3.0]

              ```
        """
        nc = self._ds
        op = _Rank(pct=pct)
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def pad(
        self,
        *,
        mode: str = "constant",
        constant_values: Any = None,
        **pad_width: Any,
    ) -> NetCDF:
        """Pad band or spatial dimensions with a constant fill.

        Extends each named dimension by `(before, after)` steps. A **band** dimension grows with the
        fill and gains NaN coordinate stamps; a **spatial** axis (`x`/`y`/`lon`/`lat`) grows the
        grid and **moves the geotransform** with the data (the padded corner becomes the new
        origin), so the result stays correctly georeferenced. Only `mode="constant"` is supported;
        the fill is `constant_values` when given, otherwise the variable's no-data value (NaN when
        it declares none). Mirrors the constant case of `xarray.Dataset.pad`, except that on a
        container an auxiliary variable spanning a padded **band** dimension is dropped with a
        warning (xarray would extend it) rather than left at an inconsistent length.

        Works on a container (every variable that has the dimension) and on a single variable.

        Args:
            mode: Only `"constant"` is supported for now.
            constant_values: The pad fill; `None` uses the variable's no-data value.
            **pad_width: `dimension=(before, after)` or `dimension=n` (both sides) pairs. A band
                dimension or a spatial axis (`x`/`y`/`lon`/`lat`).

        Returns:
            NetCDF: The padded container or variable.

        Raises:
            ValueError: `mode` is not `"constant"`, no dimension is given, a width is negative or
                not `(before, after)`/`int`, or a band `dimension` is unknown.

        Examples:
            - Pad a band dimension on both sides with no-data:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0]),
              ...     no_data_value=-9999.0,
              ... ).get_variable("t")
              >>> var.pad(time=(1, 0)).read_array().ravel().tolist()
              [-9999.0, 1.0, 2.0]

              ```
        """
        nc = self._ds
        if mode != "constant":
            raise ValueError(
                f"pad() supports only mode='constant' for now, got {mode!r}."
            )
        if not pad_width:
            raise ValueError(
                "pad() needs at least one dimension, e.g. pad(time=(1, 2)) or pad(x=3)."
            )
        result = nc
        for dim, width in pad_width.items():
            before, after = _pad_before_after(width, dim)
            if dim.lower() in _SPATIAL_AXIS_NAMES and dim not in _band_dims_of(result):
                result = _pad_spatial(result, dim, before, after, constant_values)
            else:
                op = _Pad(before=before, after=after, fill_value=constant_values)
                if _reduces_as_a_variable(result):
                    result = _apply_to_variable(result, dim, op)
                else:
                    result = _apply_to_container(result, dim, op)
        return result

    def transpose(self, *dims: Any) -> NetCDF:
        """Reorder the band dimensions; the spatial `(y, x)` plane stays trailing.

        xarray's `transpose`, restricted to what a georeferenced cube allows: the horizontal plane
        is pinned as the trailing `(row, column)` axes by the geotransform, so only the band
        (non-spatial) dimensions may be permuted. Naming a spatial axis is refused. With no
        arguments the band order is reversed; `...` (Ellipsis) expands to the unnamed band
        dimensions in their current order.

        Works on a container (each variable reordered by the subset of `dims` it has) and on a
        single variable.

        Args:
            *dims: The new band-dimension order. Every band dimension must be named, or `...` used
                for the rest; empty reverses them.

        Returns:
            NetCDF: The container or variable with its band axes reordered; values, grid and
            coordinates unchanged.

        Raises:
            ValueError: A named dimension is spatial or not a band dimension, a duplicate is given,
                or (without `...`) not every band dimension is named.

        Examples:
            - Swap two band dimensions:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> cube = NetCDF.from_array(
              ...     np.arange(4.0).reshape(2, 2, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(dims=[("time", [0.0, 1.0]), ("level", [0.0, 1.0])]),
              ... )
              >>> out = cube.transpose("level", "time").get_variable("t")
              >>> out._band_dim_names
              ('level', 'time')

              ```
        """
        nc = self._ds
        _validate_transpose_dims(nc, dims)

        def _fn(var: NetCDF) -> tuple:
            band_names = list(var._band_dim_names)
            order = _transpose_order(band_names, dims)
            perm = [band_names.index(name) for name in order]
            arr = np.asarray(nc._materialize_variable_array(var, lazy=True))
            arr = np.transpose(arr, [*perm, arr.ndim - 2, arr.ndim - 1])
            values_map = {name: var._band_dim_values_map.get(name) for name in order}
            return arr, order, values_map, _read_no_data(var), var.geotransform

        return _apply_per_variable(nc, _fn, caller="transpose")

    def set_coords(self, names: str | Sequence[str]) -> NetCDF:
        """Mark existing variables as CF auxiliary coordinates.

        xarray's `set_coords`, expressed the way CF and GDAL already express it: a variable
        is a coordinate of another because that other names it in its `coordinates`
        attribute. pyramids already *reads* that — `cf.classify_variables` assigns the
        auxiliary-coordinate role from it, and `variable_names` / `data_vars` filter
        themselves by the roles — so this is the write side, and the partition needs no new
        model. A promoted variable therefore leaves `data_vars`, as it does in xarray, and
        everything that iterates variables (`merge`, `concat`, `apply`, `reduce`,
        `to_dataframe`, `to_xarray`) follows without being told.

        What it does **not** do is promote to a *dimension* coordinate. The netCDF driver
        skips GDAL's `SetIndexingVariable`, so a dimension's coordinate is the same-named
        1-D array by CF convention and cannot be reassigned; `rename_variable` is the only
        honest way to make an array a dimension's coordinate. Naming a dimension here is
        refused rather than silently doing nothing.

        A **container** operation: a single variable has no sibling to carry the reference,
        and promoting the only variable would leave a cube with no data at all.
        Non-mutating, like `rename_dims` / `assign_coords` / `drop_dims` — the receiver is
        untouched and a new cube comes back. The variable itself is never moved or copied:
        only its role changes, and `get_variable` still reads it.

        Args:
            names: A variable name, or a sequence of them, to promote.

        Returns:
            NetCDF: A new container in which those variables are auxiliary coordinates.

        Raises:
            ValueError: The receiver is a single variable; a name is not a variable of this
                container; a name is a dimension (already a coordinate by CF convention);
                or no remaining data variable spans the promoted variable's dimensions, so
                nothing can reference it.

        Examples:
            - Promote a per-cell experiment flag out of the data variables:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.full((2, 2), 1.0), geo_ref=geo, variable_name="t2m"
              ... )
              >>> flag = NetCDF.from_array(
              ...     np.full((2, 2), 5.0), geo_ref=geo, variable_name="expver"
              ... ).get_variable("expver")
              >>> cube.set_variable("expver", flag)
              >>> sorted(cube.variable_names)
              ['expver', 't2m']
              >>> promoted = cube.set_coords("expver")
              >>> promoted.variable_names
              ['t2m']
              >>> promoted.get_variable("expver").band_count
              1

              ```

            - Promote two at once, leaving one data variable behind:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.full((1, 1), 1.0), geo_ref=geo, variable_name="t2m"
              ... )
              >>> for label in ("expver", "angle"):
              ...     extra = NetCDF.from_array(
              ...         np.full((1, 1), 2.0), geo_ref=geo, variable_name=label
              ...     ).get_variable(label)
              ...     cube.set_variable(label, extra)
              >>> cube.set_coords(["expver", "angle"]).variable_names
              ['t2m']

              ```

            - A dimension is already its own coordinate, so naming one is refused:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.full((1, 1), 1.0), geo_ref=geo, variable_name="t2m"
              ... )
              >>> cube.set_coords("x")
              Traceback (most recent call last):
                  ...
              ValueError: set_coords(): 'x' is a dimension of this cube, so it is already ...

              ```

        See Also:
            NetCDF.reset_coords: Demotes them back to data variables.
            NetCDF.rename_variable: Renames an array, the only way to make one a
                dimension's coordinate.
        """
        nc = self._ds
        _assert_coordinate_partition(nc, caller="set_coords")
        requested = _requested_names(names)
        promoted = _validated_promotions(nc, requested)
        receivers = _promotion_receivers(nc, promoted)
        return (
            nc._with_coordinate_refs(receivers, add=tuple(promoted))
            if promoted
            else nc.copy()
        )

    def reset_coords(self, names: str | Sequence[str] | None = None) -> NetCDF:
        """Demote auxiliary coordinates back to data variables.

        The inverse of :meth:`set_coords`, and the reason that one is not a one-way door.
        It removes the named variables from every data variable's CF `coordinates`
        attribute — deleting the attribute outright when nothing is left in it — so
        `cf.classify_variables` stops reporting them as coordinates and they reappear in
        `variable_names` / `data_vars`.

        It matters for files that over-declare. A store that lists a per-cell experiment
        flag or a scan angle as a coordinate keeps it out of the data-variable role, so
        anything driven by the roles treats it as a label rather than as data a caller may
        want to analyse. This is how to get it back.

        The variable is never deleted — only its role changes. `remove_variable` and
        `drop_dims` are the members that remove data.

        A **container** operation and non-mutating, as :meth:`set_coords` is. A
        **dimension** coordinate cannot be demoted: it is a coordinate because it shares
        its dimension's name, so naming one here is refused.

        Args:
            names: A variable name, or a sequence of them, to demote. `None` (default)
                demotes every auxiliary coordinate in the container.

        Returns:
            NetCDF: A new container in which those variables are data variables again.

        Raises:
            ValueError: The receiver is a single variable; a name is not an auxiliary
                coordinate of this container; or a name is a dimension coordinate.

        Examples:
            - Promote and then demote, which returns the original roles:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.full((2, 2), 1.0), geo_ref=geo, variable_name="t2m"
              ... )
              >>> flag = NetCDF.from_array(
              ...     np.full((2, 2), 5.0), geo_ref=geo, variable_name="expver"
              ... ).get_variable("expver")
              >>> cube.set_variable("expver", flag)
              >>> promoted = cube.set_coords("expver")
              >>> promoted.variable_names
              ['t2m']
              >>> sorted(promoted.reset_coords().variable_names)
              ['expver', 't2m']

              ```

            - Demote one of two coordinates by name, leaving the other a coordinate:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.full((1, 1), 1.0), geo_ref=geo, variable_name="t2m"
              ... )
              >>> for label in ("expver", "angle"):
              ...     extra = NetCDF.from_array(
              ...         np.full((1, 1), 2.0), geo_ref=geo, variable_name=label
              ...     ).get_variable(label)
              ...     cube.set_variable(label, extra)
              >>> promoted = cube.set_coords(["expver", "angle"])
              >>> sorted(promoted.reset_coords("angle").variable_names)
              ['angle', 't2m']

              ```

        See Also:
            NetCDF.set_coords: The promotion this undoes.
        """
        nc = self._ds
        _assert_coordinate_partition(nc, caller="reset_coords")
        demoted = _validated_demotions(nc, names)
        return (
            nc._with_coordinate_refs(list(nc.variable_names), remove=tuple(demoted))
            if demoted
            else nc.copy()
        )

    def broadcast_like(self, other: Any) -> NetCDF:
        """Give this cube `other`'s band layout, repeating cells along the added axes.

        xarray's `broadcast_like`, restricted to what a georeferenced cube can mean: the
        `(y, x)` plane is pinned by the geotransform, so only the band (non-spatial) axes
        are broadcast and the grids must already match. There is no alignment and no join —
        a dimension the two share at different lengths is refused rather than outer-joined,
        because pyramids has no index to join on.

        What happens to each band dimension:

        - one `other` has and this cube lacks is **added**, with `other`'s size and
          coordinate values, and the cells repeated along it;
        - one both carry, at length one here against `other`'s *n*, is **stretched** the
          same way and takes `other`'s coordinates;
        - one both carry at the same length is left alone, keeping its own coordinates;
        - one this cube has and `other` lacks is kept as it is.

        The result's dimensions are this cube's, in their own order, followed by `other`'s
        that it did not have — xarray's ordering, and the reason `mask.broadcast_like(cube)`
        comes back with exactly `cube`'s layout when the mask has no band dimensions of its
        own.

        Unlike xarray's lazy view this **materialises** the repeats: a mask broadcast over
        twelve steps holds twelve times the cells. For arithmetic you do not need it — a
        single-band operand already broadcasts inside `combine`, without materialising
        anything (`cube * mask`). Reach for this when you need the broadcast result *as a
        cube*: to write it to a file, `concat` it, or hand it to `to_xarray`.

        Args:
            other: The cube whose band layout to take, on this cube's grid.

        Returns:
            NetCDF: This cube on the broadcast layout.

        Raises:
            TypeError: `other` is not a `NetCDF`.
            AlignmentError: `other` is on a different spatial grid.
            ValueError: A band dimension the two share has different lengths and neither is
                one, so there is nothing to stretch and a join would be needed.

        Examples:
            - Lift a plain raster to a cube's `time` axis:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> cube = NetCDF.from_array(
              ...     np.arange(3.0).reshape(3, 1, 1),
              ...     geo_ref=geo,
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> mask = NetCDF.from_array(
              ...     np.full((1, 1), 2.0), geo_ref=geo, variable_name="m"
              ... ).get_variable("m")
              >>> lifted = mask.broadcast_like(cube)
              >>> lifted._band_dim_names, lifted._band_dim_sizes
              (('time',), (3,))
              >>> np.asarray(lifted.read_array(squeeze=True)).ravel().tolist()
              [2.0, 2.0, 2.0]

              ```

            - Stretch an axis that is one step long, taking the donor's stamps for it:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> levels = [("time", [0.0, 6.0]), ("level", [1000.0, 850.0, 500.0])]
              >>> donor = NetCDF.from_array(
              ...     np.zeros((2, 3, 1, 1)),
              ...     geo_ref=geo,
              ...     variable_name="d",
              ...     dims=ExtraDimensions(dims=levels),
              ... ).get_variable("d")
              >>> one_level = NetCDF.from_array(
              ...     np.arange(2.0).reshape(2, 1, 1, 1),
              ...     geo_ref=geo,
              ...     variable_name="v",
              ...     dims=ExtraDimensions(dims=[("time", [0.0, 6.0]), ("level", [1000.0])]),
              ... ).get_variable("v")
              >>> stretched = one_level.broadcast_like(donor)
              >>> stretched._band_dim_sizes
              (2, 3)
              >>> stretched._band_dim_values_map["level"]
              [1000.0, 850.0, 500.0]
              >>> stretched.band_count
              6

              ```

            - Two real lengths cannot be reconciled, since broadcasting never joins axes:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> two = NetCDF.from_array(
              ...     np.zeros((2, 1, 1)),
              ...     geo_ref=geo,
              ...     variable_name="a",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
              ... ).get_variable("a")
              >>> three = NetCDF.from_array(
              ...     np.zeros((3, 1, 1)),
              ...     geo_ref=geo,
              ...     variable_name="b",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("b")
              >>> try:
              ...     two.broadcast_like(three)
              ... except ValueError as refusal:
              ...     print(str(refusal)[:49])
              broadcast_like(): dimension 'time' is 2 long here

              ```

        See Also:
            NetCDF.combine: Broadcasts a single band without materialising it, which is
                what the arithmetic operators use.
            NetCDF.expand_dims: Adds one band dimension of length one.
            NetCDF.broadcast_equals: Compares two cubes after broadcasting.
        """
        nc = self._ds
        donor = _donor_band_layout(other, caller="broadcast_like")
        # Variable to variable: a container's own raster is a 512x512 placeholder, so
        # comparing the containers themselves refuses every container broadcast.
        if not _same_spatial_grid(_grid_reference(nc), _grid_reference(other)):
            raise AlignmentError(
                "broadcast_like() does not resample: the two cubes are on different "
                "spatial grids, so there is no cell-for-cell correspondence to repeat. "
                "Put them on one grid first (`other = other.align(self)`)."
            )

        def _fn(var: NetCDF) -> tuple:
            out_names, out_sizes, values_map = _broadcast_layout(var, donor)
            values = np.asarray(nc._materialize_variable_array(var, lazy=True))
            return (
                _broadcast_values(
                    values,
                    list(var._band_dim_names),
                    list(var._band_dim_sizes),
                    out_names,
                    out_sizes,
                ),
                out_names,
                values_map,
                _read_no_data(var),
                var.geotransform,
            )

        return _apply_per_variable(nc, _fn, caller="broadcast_like")

    def broadcast_equals(self, other: Any) -> bool:
        """Whether two cubes hold the same values once broadcast against each other.

        xarray's `broadcast_equals`: the weaker of the two equality questions. `equals`
        compares the layouts as they are, so a `(y, x)` mask and the `(time, y, x)` cube
        whose every step holds that mask are not equal; this asks whether they describe the
        same values at a common rank, which they do.

        Both operands are put on their common layout with `broadcast_like` and then handed
        to `equals`, so the grid, band dimensions, coordinates and cells are all compared by
        one implementation rather than a second copy of the rules. The right operand is
        reordered onto the left's dimension order first, so the answer does not depend on
        which side it was asked from.

        A pair that cannot be broadcast — a shared dimension at two lengths, a different
        grid, or an operand that is not a cube — answers `False` rather than raising: this
        is a predicate, and `equals` already answers `False` for operands it cannot line
        up. Only those three cases are swallowed; anything else surfaces, so a defect is
        never reported as inequality.

        A **container** receiver is refused rather than answered, for the same reason
        `equals` refuses one: a container has no cells of its own, so there is nothing to
        compare. (Before this was explicit, `equals`' own refusal was caught and turned
        into `False`, which made a container not broadcast-equal to *itself*.)

        Args:
            other: The cube to compare with.

        Returns:
            bool: `True` when the two agree after broadcasting.

        Raises:
            ValueError: The receiver is a container.

        Examples:
            - A mask and the cube whose every step holds it:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> mask = NetCDF.from_array(
              ...     np.full((1, 1), 2.0), geo_ref=geo, variable_name="m"
              ... ).get_variable("m")
              >>> cube = NetCDF.from_array(
              ...     np.full((3, 1, 1), 2.0),
              ...     geo_ref=geo,
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> mask.equals(cube), mask.broadcast_equals(cube)
              (False, True)

              ```

            - Lining the shapes up does not make the cells agree, and a pair that cannot
              be broadcast at all answers `False` rather than raising:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> mask = NetCDF.from_array(
              ...     np.full((1, 1), 2.0), geo_ref=geo, variable_name="m"
              ... ).get_variable("m")
              >>> rising = NetCDF.from_array(
              ...     np.arange(3.0).reshape(3, 1, 1),
              ...     geo_ref=geo,
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> mask.broadcast_equals(rising)
              False
              >>> two_steps = NetCDF.from_array(
              ...     np.zeros((2, 1, 1)),
              ...     geo_ref=geo,
              ...     variable_name="s",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
              ... ).get_variable("s")
              >>> two_steps.broadcast_equals(rising)
              False

              ```

        See Also:
            NetCDF.broadcast_like: The broadcast this is built on.
            Analysis.equals: The comparison it ends in.
        """
        nc = self._ds
        if not _reduces_as_a_variable(nc):
            raise ValueError(
                "broadcast_equals() compares one raster with another, and this is a "
                "container, which has no cells of its own to compare — the same reason "
                "`equals` refuses one. Pick the variables to compare with "
                "`get_variable`, or compare the containers variable by variable."
            )
        answer = False
        try:
            left = nc.broadcast_like(other)
            right = other.broadcast_like(nc)
            ordered = (
                right
                if tuple(right._band_dim_names) == tuple(left._band_dim_names)
                else right.transpose(*left._band_dim_names)
            )
            answer = bool(left.equals(ordered))
        except (AlignmentError, TypeError, _NotBroadcastable):
            # Only the three ways two cubes can be *incomparable* are answered `False`:
            # a different grid, an operand that is not a cube, and axes that cannot be
            # reconciled. Anything else is a defect and must surface, not be reported as
            # inequality -- catching plain `ValueError` here hid both a container
            # refusal and a reshape bug.
            answer = False
        return answer

    def rename_dims(
        self, dims: Mapping[str, str] | None = None, **dims_kwargs: str
    ) -> NetCDF:
        """Rename one or more band dimensions; cells, band count and coordinates unchanged.

        xarray's `rename_dims`, restricted to the band (non-spatial) axes: the geotransform pins
        the `(y, x)` plane, so a spatial axis cannot be renamed and a band dimension cannot take a
        spatial name. A rename moves no cells. On a single variable it is a free re-label that
        keeps the variable's lazy read. A container keeps its dimensions in the *store*, so there
        the rename is store-level surgery: the cube is rebuilt into a fresh root group with the
        dimension re-labelled in place, which leaves no orphan of the old name, keeps the variable
        inventory (CF bounds and sub-groups included) and carries each coordinate array, its
        attributes and the store's global attributes across.

        Args:
            dims: A `{old: new}` mapping of band dimensions to rename.
            **dims_kwargs: The same as `old=new` keywords; merged with `dims`.

        Returns:
            NetCDF: The cube with those band dimensions renamed.

        Raises:
            ValueError: An `old` name is not a band dimension, a `new` name is a spatial axis
                name, two renames target the same name, a `new` name already names a band
                dimension that is not itself being renamed, or a `new` name already names a
                variable (the renamed coordinate array would collide with it).

        Examples:
            - Rename `time` to `t`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="v",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("v")
              >>> var.rename_dims(time="t")._band_dim_names
              ('t',)

              ```

        See Also:
            NetCDF.rename_variable: Rename a *variable* rather than a dimension.
        """
        nc = self._ds
        mapping = {**(dims or {}), **dims_kwargs}
        known = _band_dims_of(nc)
        for old in mapping:
            if old not in known:
                raise ValueError(
                    f"rename_dims(): {old!r} is not a band dimension of this cube; its band "
                    f"dimensions are {sorted(known)}."
                )
        targets = list(mapping.values())
        if len(set(targets)) != len(targets):
            raise ValueError(
                f"rename_dims(): the new names {targets} contain a duplicate."
            )
        for new in targets:
            if new.lower() in _SPATIAL_AXIS_NAMES:
                raise ValueError(
                    f"rename_dims(): {new!r} is a spatial axis name; the geotransform pins the "
                    f"(y, x) plane, so a band dimension cannot take it."
                )
            if new in known and new not in mapping:
                raise ValueError(
                    f"rename_dims(): target {new!r} already names a band dimension."
                )
            if new in nc.variable_names:
                raise ValueError(
                    f"rename_dims(): target {new!r} already names a variable of this cube; the "
                    f"renamed coordinate array would collide with it."
                )
        effective = {old: new for old, new in mapping.items() if old != new}
        if not effective:
            return _unchanged(nc)

        if not _reduces_as_a_variable(nc):
            # A container keeps its dimensions in the store, so a rename is store-level
            # surgery: rebuild into a fresh root group with the dimension re-labelled in
            # place. This leaves no orphan of the old dimension (M2), carries CF bounds
            # variables onto the new name with the inventory unchanged (M2), and lets a
            # renamed axis keep its own CF units instead of the recycled name's (M3) — none
            # of which a per-variable in-memory rebuild could do.
            return nc._rebuilt_container(rename=effective)

        def relabel(var: NetCDF) -> tuple[list[str], dict]:
            names = [mapping.get(name, name) for name in var._band_dim_names]
            values_map = {
                mapping.get(name, name): stamps
                for name, stamps in var._band_dim_values_map.items()
            }
            return names, values_map

        return _relabel_per_variable(nc, relabel, caller="rename_dims")

    def assign_coords(
        self, coords: Mapping[str, Any] | None = None, **coords_kwargs: Any
    ) -> NetCDF:
        """Replace the coordinate stamps of one or more existing band dimensions.

        The honest subset of xarray's `assign_coords`: pyramids has no index model, so it cannot
        attach a *new*, non-dimension coordinate, nor a coordinate that becomes an alignment index —
        those are refused. What it can do is restamp a band dimension that already exists, which is a
        pure re-label (no cells move): on a single variable it keeps the lazy read, on a container
        every variable spanning the dimension is restamped.

        Args:
            coords: A `{dim: values}` mapping; each `dim` must be an existing band dimension and
                `values` a 1-D sequence as long as that dimension.
            **coords_kwargs: The same as `dim=values` keywords; merged with `coords`.

        Returns:
            NetCDF: The cube with those band dimensions restamped.

        Raises:
            ValueError: A name is not an existing band dimension (a new/non-dimension coordinate
                needs an index model pyramids does not have), the values are not 1-D, or their
                length does not match the dimension.

        Examples:
            - Restamp `time` with hours-since-midnight:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="v",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0, 3.0]),
              ... ).get_variable("v")
              >>> var.assign_coords(time=[0, 6, 12, 18])._band_dim_values_map["time"]
              [0, 6, 12, 18]

              ```

        See Also:
            NetCDF.rename_dims: Rename a band dimension rather than restamp it.
        """
        nc = self._ds
        mapping = {**(coords or {}), **coords_kwargs}
        coerced = _validated_restamp(nc, mapping, _band_dims_of(nc))
        if not coerced:
            return _unchanged(nc)

        def relabel(var: NetCDF) -> tuple[list[str], dict]:
            names = list(var._band_dim_names)
            values_map = dict(var._band_dim_values_map)
            for dim, values in coerced.items():
                if dim in names:
                    # Re-checked per variable, not only against the cube's declaration: a
                    # hierarchical container's sub-groups may declare a same-named dimension at
                    # a different length, which the working group's view cannot show (L3).
                    size = var._band_dim_sizes[names.index(dim)]
                    if len(values) != size:
                        raise ValueError(
                            f"assign_coords(): {dim!r} is length {size} on one of this cube's "
                            f"variables, but {len(values)} coordinate values were given."
                        )
                    values_map[dim] = list(values)
            return names, values_map

        return _relabel_per_variable(nc, relabel, caller="assign_coords")

    def drop_dims(
        self, drop_dims: str | Sequence[str], *, errors: str = "raise"
    ) -> NetCDF:
        """Drop one or more band dimensions and every variable that spans them.

        xarray's `drop_dims`: a dimension cannot be dropped on its own — the variables defined
        along it go with it. This is a **container** operation (a single variable *is* its bands,
        so dropping a dimension it spans would leave nothing), and it does not mutate the receiver:
        it rebuilds the survivors into a fresh container — declaring only the dimensions they
        actually use, so the dropped dimension and its coordinate are gone rather than orphaned —
        and keeps each survivor's exact dtype, CF packing and no-data.

        Args:
            drop_dims: A band-dimension name, or a sequence of names, to drop.
            errors: `"raise"` (default) to refuse an unknown dimension, `"ignore"` to skip it.

        Returns:
            NetCDF: A new container without those dimensions or the variables that spanned them.

        Raises:
            ValueError: `errors` is not `"raise"`/`"ignore"`; the receiver is a single variable;
                or (with `errors="raise"`) a named dimension is not a dimension of the container.

        Examples:
            - Drop `time`, removing the variable defined along it:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> cube = NetCDF.from_array(
              ...     np.arange(12.0).reshape(3, 2, 2),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="v",
              ...     dims=ExtraDimensions(name="time", values=[0, 6, 12]),
              ... )
              >>> "v" in cube.drop_dims("time").variable_names
              False

              ```
        """
        nc = self._ds
        if errors not in ("raise", "ignore"):
            raise ValueError(
                f"drop_dims(): errors must be 'raise' or 'ignore', got {errors!r}."
            )
        targets = [drop_dims] if isinstance(drop_dims, str) else list(drop_dims)
        if _reduces_as_a_variable(nc):
            raise ValueError(
                "drop_dims() removes whole variables, and a single variable is its own bands. "
                "Call it on the container, or drop this variable with remove_variable()."
            )
        known = set(_band_dims_of(nc))
        if errors == "raise":
            for dim in targets:
                if dim not in known:
                    raise ValueError(
                        f"drop_dims(): {dim!r} is not a dimension of this container; its "
                        f"dimensions are {sorted(known)}."
                    )
        drop = {dim for dim in targets if dim in known}
        if not drop:
            return nc.copy()
        # One exit, one mechanism: rebuild into a fresh store declaring only the dimensions the
        # surviving arrays actually span, so the dropped dimension (and its coordinate array) is
        # gone rather than orphaned the way `remove_variable` leaves it -- matching the method
        # name and xarray's `drop_dims` (M2). The store-level copy keeps each survivor's exact
        # dtype, CF packing and no-data, which a `merge` rebuild silently unpacked (int16/scale
        # -> float64, M1).
        #
        # There used to be a second path for the case where nothing survives -- `copy()` plus a
        # `remove_variable` per variable -- which reintroduced exactly the orphan this method
        # exists to avoid (H3) and refused a group-qualified name outright, while the survivor
        # scan itself crashed on a hierarchical container because `get_variable` hands back a
        # `LabeledArray` for a 1-D array (M3). The rebuild handles both: it needs no survivor
        # list, and a zero-variable result is just a store with no arrays to copy.
        return nc._rebuilt_container(drop=drop)

    def update(self, other: Any) -> None:
        """Add or replace variables from another cube, in place — xarray's `Dataset.update`.

        A bulk `set_variable`: every variable in `other` is written into this container, replacing
        one of the same name and adding the rest. The blessed mutation path — the read-through
        variables mapping refuses item assignment and points here. Mutates the receiver and returns
        `None`, as xarray does. Grids must match (no resampling), consistent with `merge`/`concat`,
        and every donor is grid-checked **before** any is written, so a mismatch leaves the receiver
        untouched. The reference grid is the receiver's first variable (or, for an empty receiver,
        the first donor variable).

        Args:
            other: A `NetCDF` container, or a `{name: variable}` mapping, whose variables share this
                container's grid.

        Raises:
            ValueError: The receiver is a single variable (it has no variable mapping to update).
            TypeError: `other` is neither a `NetCDF` container nor a `{name: variable}` mapping.
            AlignmentError: A variable in `other` is on a different spatial grid, or spans a band
                dimension sharing a name with one of this container's but of a different length
                (both raised before any write, so the receiver is left untouched).

        Examples:
            - Fold another cube's variable into this container, on the same grid:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> geo = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> dims = ExtraDimensions(name="time", values=[0.0, 1.0])
              >>> a = NetCDF.from_array(np.zeros((2, 1, 1)), geo_ref=geo, variable_name="a", dims=dims)
              >>> b = NetCDF.from_array(np.ones((2, 1, 1)), geo_ref=geo, variable_name="b", dims=dims)
              >>> a.update(b)
              >>> sorted(a.variable_names)
              ['a', 'b']

              ```

        See Also:
            NetCDF.merge: Combine cubes into a new container rather than mutating this one.
        """
        nc = self._ds
        if _reduces_as_a_variable(nc):
            raise ValueError(
                "update() merges variables into a container, and a single variable has no "
                "variable mapping to update. Use set_variable, or merge, instead."
            )
        items = _donor_variables(other)
        _assert_donors_fit(nc, items)
        for name, variable in items:
            nc.set_variable(name, variable)

    def interp(self, method: str = "linear", **coords: Any) -> NetCDF:
        """Interpolate a band dimension onto new coordinate values.

        Puts each pixel's series onto new stamps by 1-D interpolation — xarray's
        `interp(dim=targets)`. The named dimension takes the target coordinates and, in general, a
        new length; the values in between are interpolated, and a target outside the source range
        comes back as a gap (the declared no-data value, or NaN). Several `dim=targets` pairs are
        applied one after another, each an independent 1-D interpolation.

        Only **band (non-spatial) dimensions** are interpolated here. A spatial axis is refused with
        a pointer to the operations that regrid the horizontal plane correctly (they move the
        geotransform with the data): `resample` for a new cell size, `to_crs` for a new CRS,
        `align` onto another dataset's grid, and `extract` / `point` for scattered points.

        Works on a container, interpolating every variable that has the dimension, and on a single
        variable, returning a variable. A container's auxiliary variable spanning the dimension is
        dropped with a warning, since its coordinates change (and its length may too), so carrying
        it verbatim would misalign it with the interpolated axis.

        Args:
            method: The interpolation kind, forwarded to `scipy.interpolate.interp1d`: `"linear"`
                (default), `"nearest"`, `"cubic"`, `"zero"`, `"slinear"`, `"quadratic"`,
                `"previous"` or `"next"`. These match the kinds xarray forwards to scipy for 1-D
                interpolation. The spline kinds `"cubic"` / `"quadratic"` fit the whole series at
                once, so a single gap (a no-data cell or NaN) anywhere makes the entire interpolated
                axis a gap; prefer a local kind (`"linear"` / `"nearest"`) on data with gaps.
            **coords: `dimension=targets` pairs; each `dimension` must be a numeric band dimension
                and `targets` a 1-D sequence (or scalar) of coordinate values to interpolate onto. A
                band dimension literally named `method` cannot be passed here — that keyword is the
                interpolation kind — the same reserved-name limitation as `xarray.Dataset.interp`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, with each named
            dimension relabelled to its targets.

        Raises:
            ValueError: No `coords` given; an unknown `method`; a named dimension that is spatial
                (with the regrid pointer), unknown, text, or coordinate-less; or a target that is
                empty or holds NaN.

        Examples:
            - Interpolate a time axis onto stamps between the source's:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([0.0, 10.0, 20.0]).reshape(3, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 10.0, 20.0]),
              ... ).get_variable("t")
              >>> out = var.interp(time=[5.0, 15.0])
              >>> out.read_array().ravel().tolist()
              [5.0, 15.0]
              >>> out._band_dim_values_map["time"]
              [5.0, 15.0]

              ```
            - Snap each target to its nearest source step with `method="nearest"`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([0.0, 10.0, 20.0]).reshape(3, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 10.0, 20.0]),
              ... ).get_variable("t")
              >>> var.interp(time=[4.0, 16.0], method="nearest").read_array().ravel().tolist()
              [0.0, 20.0]

              ```
        """
        nc = self._ds
        if not coords:
            raise ValueError(
                "interp() needs at least one dimension to interpolate, e.g. interp(time=[...])."
            )
        kind = _resolve_interp_kind(method)
        result = nc
        for dim, target in coords.items():
            result = _run_interp(result, dim, target, kind)
        return result

    def interp_like(self, other: NetCDF, method: str = "linear") -> NetCDF:
        """Interpolate the band dimensions shared with `other` onto `other`'s coordinates.

        The band-axis half of xarray's `interp_like`: for every band dimension this cube shares
        with `other`, interpolate onto `other`'s coordinate values for that dimension. The spatial
        plane is **not** regridded here — if the two spatial grids differ the call is refused, with
        a pointer to `to_crs` / `resample` / `align`, so `interp_like` stays a pure band-axis
        operation (compose `cube.to_crs(other.epsg).interp_like(other)` for the full effect).

        Args:
            other: The cube whose band coordinates this one is interpolated onto. Its spatial grid
                must match this cube's.
            method: The interpolation kind, as in `interp`.

        Returns:
            NetCDF: This cube with each shared band dimension interpolated onto `other`'s
            coordinates.

        Raises:
            ValueError: An unknown `method`; the two spatial grids differ (with the regrid
                pointer); or no band dimension is shared with `other`.

        Examples:
            - Put one cube on another's time axis:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> ref = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
              >>> a = NetCDF.from_array(
              ...     np.array([0.0, 20.0]).reshape(2, 1, 1), geo_ref=ref, variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 20.0]),
              ... ).get_variable("t")
              >>> b = NetCDF.from_array(
              ...     np.zeros((3, 1, 1)), geo_ref=ref, variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 10.0, 20.0]),
              ... ).get_variable("t")
              >>> a.interp_like(b).read_array().ravel().tolist()
              [0.0, 10.0, 20.0]

              ```
        """
        nc = self._ds
        kind = _resolve_interp_kind(method, caller="interp_like")
        if not _same_spatial_grid(nc, other):
            raise ValueError(
                "interp_like() interpolates only band dimensions, but the two spatial grids "
                "differ. Put this cube on other's grid first with to_crs() / resample() / "
                "align(), then interp_like()."
            )
        shared = [dim for dim in _band_dims_of(nc) if dim in _band_dims_of(other)]
        if not shared:
            raise ValueError(
                "interp_like() found no band dimension shared with other to interpolate onto."
            )
        result = nc
        for dim in shared:
            target = _interp_source_coordinates(other, dim, caller="interp_like")
            result = _run_interp(result, dim, target, kind, caller="interp_like")
        return result

    def cumsum(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Total the values along a non-spatial dimension, step by step.

        Each step holds the sum of itself and every step before it. The dimension keeps its
        length and its stamps.

        Works on a container, totalling every variable that has `dim`, and on a single variable,
        returning a variable. A container's auxiliary variables are all carried over, those
        spanning `dim` included, since its length does not change.

        Args:
            dim: The non-spatial dimension to total along.
            skipna: When `True` (default), gaps are skipped: a gap adds nothing and holds the
                total so far, and a step before the first valid cell is a gap — where xarray
                answers `0.0`, a total of nothing that is not invented here, just as
                `reduce(how="sum")` does not invent one for an all-gap slice. When `False` the
                stored values add up as numpy adds them, the sentinel and NaN included, and the
                result declares no no-data value at all — the sentinel is part of the totals, so
                no cell holds it any more and declaring it would mask a total that landed on it.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with every dimension
            unchanged. Skipping gaps the total is float64 and declares the variable's no-data
            value, or NaN when it declares none; otherwise it is numpy's own type for the total
            and declares none.

        Raises:
            ValueError: The container has no data variables, or `dim` is not a band dimension of
                any gridded variable (or of this variable, or this variable has none).

        Examples:
            - The running total, and the same total ending at `reduce(how="sum")`:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.cumsum("time").read_array().ravel().tolist()
              [1.0, 3.0, 6.0, 10.0]
              >>> float(var.reduce("time", "sum").read_array()[0, 0])
              10.0

              ```
            - A gap adds nothing, and a leading gap stays a gap:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([np.nan, 2.0, np.nan, 4.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.cumsum("time").read_array().ravel().tolist()
              [nan, 2.0, 2.0, 6.0]

              ```
            - Without skipping, the stored sentinel is added like any other value, and an
              integer band totals in the wider type numpy gives a cumulative sum:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1, 2, 3, -1], dtype="int16").reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=-1,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> raw = var.cumsum("time", skipna=False)
              >>> raw.read_array().ravel().tolist(), raw.read_array().dtype.name
              ([1, 3, 6, 5], 'int64')
              >>> var.cumsum("time").read_array().ravel().tolist()
              [1.0, 3.0, 6.0, 6.0]

              ```
        """
        nc = self._ds
        op = _CumSum(skipna=bool(skipna))
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def cumprod(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Multiply the values along a non-spatial dimension, step by step.

        :meth:`cumsum`'s multiplicative twin, and it shares everything else: each step holds
        the product of itself and every step before it, the dimension keeps its length and
        its stamps, and a container's auxiliary variables are all carried over.

        Works on a container, multiplying every variable that has `dim`, and on a single
        variable, returning a variable.

        Args:
            dim: The non-spatial dimension to multiply along.
            skipna: When `True` (default), gaps are skipped: a gap multiplies by nothing and
                holds the product so far, and a step before the first valid cell is a gap —
                where xarray answers `1.0`, a product of nothing that is not invented here.
                When `False` the stored values multiply as numpy multiplies them, the
                sentinel included, and the result declares no no-data value, since the
                sentinel went into the products.

        Returns:
            NetCDF: A container for a container, a variable for a variable. Skipping gaps the
            values are float64 and declare the variable's no-data value, or NaN when it
            declares none.

        Raises:
            ValueError: `dim` is not a band dimension of the variable, or of any gridded
                variable in the container.

        Examples:
            - A running product, the gap holding the product so far:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([2.0, 3.0, np.nan, 4.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.cumprod("time").read_array().ravel().tolist()
              [2.0, 6.0, 6.0, 24.0]

              ```
            - `skipna=False` multiplies the stored values as numpy does:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.cumprod("time", skipna=False).read_array().ravel().tolist()
              [1.0, 2.0, 6.0, 24.0]

              ```

        See Also:
            NetCDF.cumsum: The running total instead.
            NetCDF.reduce: `how="prod"` for the whole product at once.
        """
        nc = self._ds
        op = _CumProd(skipna=bool(skipna))
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def shift(self, dim: str, periods: int = 1, *, fill_value: Any = None) -> NetCDF:
        """Move the values along a non-spatial dimension, filling the steps that are vacated.

        A positive `periods` moves the values towards the end of the dimension, so each step
        holds what the step `periods` earlier held; a negative one moves them the other way.
        The dimension keeps its length and its stamps, so a shift is how a step is compared
        with an earlier one (`var - var.shift("time", 1)` is `diff`, with the length kept).

        Works on a container, shifting every variable that has `dim`, and on a single variable,
        returning a variable. A container's auxiliary variables are all carried over, those
        spanning `dim` included, since its length does not change.

        Args:
            dim: The non-spatial dimension to shift along.
            periods: Steps to move, any integer. A shift of at least the length of `dim` leaves
                every step vacated.
            fill_value: What a vacated step holds; `None` (default) asks for the variable's
                no-data value, or NaN when it declares none. A plain Python integer never widens
                an integer band — under NEP 50 it is a weak type — so one the band cannot hold is
                refused rather than promoted; a fractional fill, NaN, or a numpy scalar widens it
                as `numpy.result_type` says.

        Returns:
            NetCDF: A container for a container, a variable for a variable, with every dimension
            unchanged. With no `fill_value`, a band declaring a no-data value keeps its own type
            and fills with that value, while one declaring none fills with NaN and declares it —
            keeping a float band's own floating type, and widening an integer band to float64. A
            `fill_value` is held in `numpy.result_type` of the band and the fill, and leaves the
            declared no-data value alone.

        Raises:
            TypeError: `periods` is not an integer, or is a boolean; `fill_value` is neither
                `None` nor a real number, or is a boolean.
            ValueError: The band cannot hold `fill_value`; the container has no data variables;
                or `dim` is not a band dimension of any gridded variable (or of this variable, or
                this variable has none).

        Examples:
            - One step forward, the vacated step holding the declared no-data value:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.shift("time", 1).read_array().ravel().tolist()
              [nan, 1.0, 2.0, 3.0]
              >>> var.shift("time", -1, fill_value=0.0).read_array().ravel().tolist()
              [2.0, 3.0, 4.0, 0.0]

              ```
            - The stamps stay put, so the shifted step can be compared with its own:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, 2.0, 4.0, 8.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> shifted = var.shift("time", 1)
              >>> shifted._band_dim_values_map["time"]
              [0.0, 6.0, 12.0, 18.0]
              >>> (var - shifted).read_array().ravel().tolist()
              [nan, 1.0, 2.0, 4.0]

              ```
            - An explicit fill leaves the declared no-data value alone, and only a fractional
              one widens an integer band:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1, 2, 3, 4], dtype="int16").reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=-1,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> zeros = var.shift("time", 1, fill_value=0)
              >>> zeros.read_array().ravel().tolist(), zeros.read_array().dtype.name
              ([0, 1, 2, 3], 'int16')
              >>> zeros.no_data_value[0]
              -1.0
              >>> halves = var.shift("time", 1, fill_value=0.5)
              >>> halves.read_array().ravel().tolist(), halves.read_array().dtype.name
              ([0.5, 1.0, 2.0, 3.0], 'float64')

              ```
        """
        nc = self._ds
        steps = _check_periods(periods)
        _check_fill_value(fill_value)
        op = _Shift(periods=steps, fill_value=fill_value)
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def argmin(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Answer the position along a non-spatial dimension where the value is smallest.

        The dimension is removed, as a collapsing `reduce` removes it, and each cell holds the
        zero-based position of its smallest value — the first of them when several tie. A slice
        with no valid cell has no minimum, so it holds `-1`, the declared no-data value, where
        xarray raises `ValueError: All-NaN slice encountered`.

        Works on a container, searching every variable that has `dim`, and on a single variable,
        returning a variable. A container's auxiliary variable spanning `dim` is dropped with a
        warning, since the dimension is removed.

        Args:
            dim: The non-spatial dimension to search along.
            skipna: When `True` (default), gaps (the declared no-data value and NaN) are skipped.
                When `False` the stored values are searched as numpy searches them, where NaN
                wins and a sentinel competes as a value.

        Returns:
            NetCDF: A container for a container, a variable for a variable, `int64`, declaring
            `-1`, with `dim` removed and the other dimensions kept.

        Raises:
            ValueError: The container has no data variables, or `dim` is not a band dimension of
                any gridded variable (or of this variable, or this variable has none).

        Examples:
            - Which step is coldest, and a column with nothing to compare:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[3.0, np.nan], [1.0, np.nan], [2.0, np.nan]]).reshape(3, 1, 2),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> coldest = var.argmin("time")
              >>> coldest.read_array().ravel().tolist(), coldest.no_data_value[0]
              ([1, -1], -1)

              ```
            - Without skipping, the search is numpy's own, where a NaN wins the comparison:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([3.0, np.nan, 2.0]).reshape(3, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> var.argmin("time").read_array().ravel().tolist()
              [2]
              >>> var.argmin("time", skipna=False).read_array().ravel().tolist()
              [1]

              ```
        """
        nc = self._ds
        op = _Extremum(
            extreme="min", coordinate=False, skipna=bool(skipna), caller="argmin"
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def argmax(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Answer the position along a non-spatial dimension where the value is largest.

        The mirror of `argmin`: the dimension is removed, each cell holds the zero-based
        position of its largest value (the first of them when several tie), and a slice with no
        valid cell holds `-1`, the declared no-data value.

        Args:
            dim: The non-spatial dimension to search along.
            skipna: Whether gaps are skipped, as `argmin` documents it.

        Returns:
            NetCDF: A container for a container, a variable for a variable, `int64`, declaring
            `-1`, with `dim` removed.

        Raises:
            ValueError: The container has no data variables, or `dim` is not a band dimension of
                any gridded variable (or of this variable, or this variable has none).

        Examples:
            - Which of three steps is warmest per cell:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 9.0]]).reshape(3, 1, 2),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> var.argmax("time").read_array().ravel().tolist()
              [0, 2]

              ```
        """
        nc = self._ds
        op = _Extremum(
            extreme="max", coordinate=False, skipna=bool(skipna), caller="argmax"
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def idxmin(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Answer the coordinate value along a non-spatial dimension where the value is smallest.

        `argmin`'s answer read through the dimension's own coordinates: the stamp of the
        smallest value rather than its position — "when was it coldest", not "which step". The
        stamps are the raw stored numbers, so a CF time axis answers its offsets, not decoded
        dates. A slice with no valid cell holds NaN, the declared no-data value, as xarray's
        `idxmin` answers there.

        Args:
            dim: The non-spatial dimension to search along. It must carry numeric coordinates.
            skipna: Whether gaps are skipped, as `argmin` documents it.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, declaring
            NaN, with `dim` removed.

        Raises:
            ValueError: `dim` has no coordinate values, or they are not all numbers (use
                `argmin` for the position); the container has no data variables; or `dim` is not
                a band dimension of any gridded variable (or of this variable, or this variable
                has none).

        Examples:
            - The stamp of the coldest step, and NaN where there is nothing to compare:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[3.0, np.nan], [1.0, np.nan], [2.0, np.nan]]).reshape(3, 1, 2),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=np.nan,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
              ... ).get_variable("t")
              >>> var.idxmin("time").read_array().ravel().tolist()
              [6.0, nan]

              ```
            - A dimension whose stamps were dropped — an operator result's — has no coordinate
              to answer with, so `argmin` is the member to reach for:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.arange(4.0).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> change = var.isel(time=slice(2, 4)) - var.isel(time=slice(0, 2))
              >>> change._band_dim_values_map["time"] is None
              True
              >>> change.idxmin("time")  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                ...
              ValueError: idxmin() needs the stamps of 'time'
              >>> change.argmin("time").read_array().ravel().tolist()
              [0]

              ```
        """
        nc = self._ds
        op = _Extremum(
            extreme="min", coordinate=True, skipna=bool(skipna), caller="idxmin"
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def idxmax(self, dim: str, *, skipna: bool = True) -> NetCDF:
        """Answer the coordinate value along a non-spatial dimension where the value is largest.

        The mirror of `idxmin`: the stamp of the largest value, NaN for a slice with no valid
        cell, and the raw stored stamps rather than decoded dates.

        Args:
            dim: The non-spatial dimension to search along. It must carry numeric coordinates.
            skipna: Whether gaps are skipped, as `argmin` documents it.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, declaring
            NaN, with `dim` removed.

        Raises:
            ValueError: `dim` has no coordinate values, or they are not all numbers (use
                `argmax` for the position); the container has no data variables; or `dim` is not
                a band dimension of any gridded variable (or of this variable, or this variable
                has none).

        Examples:
            - Which pressure level holds the maximum of each cell:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[1.0, 9.0], [5.0, 2.0]]).reshape(2, 1, 2),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="level", values=[1000.0, 850.0]),
              ... ).get_variable("t")
              >>> var.idxmax("level").read_array().ravel().tolist()
              [850.0, 1000.0]

              ```
        """
        nc = self._ds
        op = _Extremum(
            extreme="max", coordinate=True, skipna=bool(skipna), caller="idxmax"
        )
        if _reduces_as_a_variable(nc):
            result = _apply_to_variable(nc, dim, op)
        else:
            result = _apply_to_container(nc, dim, op)
        return result

    def weighted(
        self,
        weights: Any,
        dims: Any = None,
        *,
        how: str = "mean",
        skipna: bool = True,
    ) -> NetCDF:
        """Weight the cells along one or more dimensions and reduce them.

        The statistic every cell contributes to in proportion to its weight: a weighted mean is
        `sum(w * x) / sum(w)`. Equal cells of a latitude-longitude grid do not cover equal area,
        so the common use is a **regional or global mean** — `weights="area"`, which is
        `cos(latitude)` per row — over the two spatial axes.

        Weighting the spatial axes leaves no cells for the answer to sit in, so the result is a
        raster of **one cell spanning the source's extent**, keeping the band dimensions and
        their stamps: `nc.weighted("area")` on a `(valid_time, latitude, longitude)` container
        answers one value per step, `reduce` and `to_file` still work on the result, and `sel` /
        `isel` on the variable taken from it, as they do on any container's variable. Weighting
        one spatial axis leaves the other in place, and weighting a band dimension keeps the
        grid and removes that dimension, as `reduce` removes it.

        A gap (the declared no-data value or NaN) leaves both sums, so the answer is the
        statistic of the cells there are. A slice with no valid cell has no statistic and comes
        back NaN, the declared no-data value. Weights that cancel to a total of zero cost only
        the statistics that divide by that total — `mean`, `std` and `var` — while `sum`
        answers the sum it computed and `sum_of_weights` answers `0.0`; xarray answers NaN for
        that `sum_of_weights`.

        **The single cell's footprint.** In memory the result is exact: `geotransform` and
        `cell_size` describe the extent that was reduced, the cell's `lat` / `lon` are its
        centre, and a kept spatial axis keeps its own coordinates. On a source of 2-degree cells
        spanning `[10, 56, 14, 60]`, both the container and the variable report
        `(10.0, 4.0, 0, 60.0, 0, -4.0)` and a `cell_size` of `4.0`. Read `bounds` off the
        **variable** (`[10.0, 56.0, 14.0, 60.0]`, the source's own): a container's `bounds` come
        from its placeholder raster and describe no variable of it, weighted or not.
        **Writing the result to a NetCDF loses the reduced axis' width**: the file records
        coordinate *values*, and one value carries no spacing, so the reopened container reports
        a unit cell around that centre, and a variable taken from it falls back to index space
        altogether, since GDAL declines to georeference a one-pixel-wide subdataset. The values,
        the band dimensions, their stamps and the CRS survive the round trip exactly. Any
        one-cell-wide raster written this way has the same limit; keep the source (or
        `source.bounds`) if the extent has to stay on record in the file.

        Args:
            weights: `"area"` for `cos(latitude)` per row, which needs a geographic CRS and
                takes each row's latitude from the geotransform — exact on a regular lat/lon
                grid, an approximation on a curvilinear one, which pyramids reads through a
                bounding-box affine, and refused when the rows run off the globe, where the
                cosine turns negative; an
                array broadcastable to the weighted axes — `(rows, 1)`, `(1, columns)` or
                `(rows, columns)` for the grid, one weight per step for a band dimension; or a
                raster on the same grid — a `NetCDF` or a `Dataset`, a GeoTIFF of weights
                included — whose first band is read as the weights. Weights may
                be negative, as xarray allows, but must all be finite: replace a NaN or an
                infinity with zero to leave that cell out. Only those are refused — a weights
                raster is read as plain numbers, so its own no-data sentinel would be weighted
                as an ordinary value.
            dims: The dimensions to weight over: `None` (default) for both spatial axes, one
                name, or a sequence of names. A spatial axis is named as the store names it
                (`latitude` / `longitude`) or as `y` / `x`. Spatial axes and band dimensions
                cannot be mixed in one call, since a container cannot hold variables on two
                grids. The spatial pair is the plane the read resolved, so a store that
                declares a band dimension between its spatial axes — CAM's
                `(time, lat, lev, lon)` — is handled by the `None` default too.
            how: `"mean"` (default), `"sum"`, `"sum_of_weights"`, `"std"` or `"var"`. The
                variance is the weighted `sum(w * (x - mean) ** 2) / sum(w)`, as xarray computes
                it. A weighted quantile is not offered; `reduce(how="quantile")` is the
                unweighted one.
            skipna: Whether the declared no-data value counts as a gap. A NaN is left out of
                both sums either way, so `skipna=False` weights the sentinel as an ordinary
                value but still skips NaN — where xarray's `skipna=False` makes the whole answer
                NaN.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64 and
            declaring NaN, on a grid reduced where a spatial axis was weighted.

        Raises:
            TypeError: `weights` is `None`, which names no weighting.
            ValueError: `how` is unknown; `dims` is empty, names a dimension the variable does
                not have, names one twice, or mixes spatial axes with band dimensions; `weights`
                is an unknown name, holds a NaN or an infinity, broadcasts onto neither the
                weighted axes nor the variable's own shape, or is a raster on another grid;
                `"area"` is asked of a grid that is not geographic, or of one whose rows run off
                the globe past 90 degrees; the container has no data variables; or no gridded
                variable of a container carries the band dimension named, as `reduce` refuses
                it. A container's gridded variable that does not carry it is carried over
                unchanged, again as `reduce` carries one it cannot reduce.

        Warns:
            UserWarning: A container's auxiliary variable spans a dimension this weighting
                leaves no full-length axis of — a band dimension that goes, or a spatial axis
                reduced to one cell — so it is dropped rather than carried at a length the rest
                of the container no longer has.

        Examples:
            - The area-weighted mean of each step, on a grid of one cell:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> nc = NetCDF.from_array(
              ...     np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 60.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
              ... )
              >>> mean = nc.weighted("area").get_variable("t")
              >>> (mean.rows, mean.columns)
              (1, 1)
              >>> [round(value, 3) for value in mean.read_array().ravel().tolist()]
              [2.515, 6.515]
              >>> mean._band_dim_values_map["time"]
              [0.0, 6.0]
              >>> mean.geotransform  # the one cell spans the 2 degrees it reduced
              (0.0, 2.0, 0, 60.0, 0, -2.0)
              >>> mean.bounds.total_bounds.tolist()
              [0.0, 58.0, 2.0, 60.0]

              ```
            - Equal weights give the plain mean, and the totals are available too:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[1.0, 2.0], [3.0, 4.0]]),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ... ).get_variable("t")
              >>> float(var.weighted(np.ones((2, 2))).read_array()[0, 0])
              2.5
              >>> float(var.weighted(np.ones((2, 2)), how="sum").read_array()[0, 0])
              10.0
              >>> float(
              ...     var.weighted(np.ones((2, 2)), how="sum_of_weights").read_array()[0, 0]
              ... )
              4.0

              ```
            - Weighting a band dimension keeps the grid and removes the dimension:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([[[1.0, 1.0]], [[3.0, 3.0]]]),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
              ... ).get_variable("t")
              >>> blended = var.weighted(np.array([3.0, 1.0]), "time")
              >>> blended.read_array().ravel().tolist()
              [1.5, 1.5]
              >>> tuple(blended._band_dim_names)
              ()

              ```
        """
        nc = self._ds
        _check_how(how, set(_WEIGHTED_HOWS))
        return _weighted_result(nc, weights, dims, how=how, skipna=bool(skipna))

    def ffill(self, dim: str, *, limit: int | None = None) -> NetCDF:
        """Carry the last valid value along a non-spatial dimension into the gaps after it.

        Each gap takes the nearest valid value **before** it along `dim`. A gap before the
        first valid cell has nothing to take and stays a gap, which is what xarray answers —
        `ffill` carries data forward, it does not invent a start.

        Works on a container, filling every variable that has `dim`, and on a single
        variable, returning a variable. A container's auxiliary variables are all carried
        over, those spanning `dim` included, since its length does not change.

        Args:
            dim: The non-spatial dimension to carry along.
            limit: How many consecutive gaps one valid cell may fill, an integer of at least
                1. `None` (default) lets a value carry as far as the next valid cell. A run
                longer than the limit keeps the gaps beyond it.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, with
            every dimension unchanged in length and coordinates. It declares the variable's
            no-data value, or NaN when it declares none, since a gap the fill could not reach
            is still a gap.

        Raises:
            TypeError: `limit` is not an integer, or is a boolean.
            ValueError: `limit` is below 1; the container has no data variables; or `dim` is
                not a band dimension of any gridded variable (or of this variable, or this
                variable has none).

        Examples:
            - A leading gap has nothing to carry into it; the rest are filled:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([np.nan, 2.0, np.nan, np.nan]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.ffill("time").read_array().ravel().tolist()
              [-9999.0, 2.0, 2.0, 2.0]
              >>> var.ffill("time", limit=1).read_array().ravel().tolist()
              [-9999.0, 2.0, 2.0, -9999.0]

              ```
            - `-9999.0` is the no-data value `from_array` declares when none is given, so
              those are the gaps the fill could not reach:

              ```python
              >>> var.ffill("time").no_data_value[0]
              -9999.0

              ```
        """
        op = _Push(
            backward=False, limit=_check_limit(limit, caller="ffill"), caller="ffill"
        )
        return _along_either(self._ds, dim, op)

    def bfill(self, dim: str, *, limit: int | None = None) -> NetCDF:
        """Carry the next valid value along a non-spatial dimension back into the gaps before it.

        `ffill` read the other way: each gap takes the nearest valid value **after** it along
        `dim`, and a gap after the last valid cell stays a gap.

        Works on a container, filling every variable that has `dim`, and on a single
        variable, returning a variable. A container's auxiliary variables are all carried
        over, those spanning `dim` included, since its length does not change.

        Args:
            dim: The non-spatial dimension to carry along.
            limit: How many consecutive gaps one valid cell may fill, an integer of at least
                1. `None` (default) lets a value carry as far as the previous valid cell.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, with
            every dimension unchanged in length and coordinates, declaring the variable's
            no-data value or NaN when it declares none.

        Raises:
            TypeError: `limit` is not an integer, or is a boolean.
            ValueError: `limit` is below 1; the container has no data variables; or `dim` is
                not a band dimension of any gridded variable (or of this variable, or this
                variable has none).

        Examples:
            - The trailing gap has nothing to carry into it:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([np.nan, 2.0, np.nan, np.nan]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.bfill("time").read_array().ravel().tolist()
              [2.0, 2.0, -9999.0, -9999.0]

              ```
        """
        op = _Push(
            backward=True, limit=_check_limit(limit, caller="bfill"), caller="bfill"
        )
        return _along_either(self._ds, dim, op)

    def dropna(
        self, dim: str, *, how: str = "any", thresh: int | None = None
    ) -> NetCDF:
        """Remove the steps of a non-spatial dimension whose cells are missing.

        The one member here whose result length depends on the values rather than on the
        arguments, so `dim`'s coordinates come back cut to the steps that survived and a
        container's auxiliary variable spanning `dim` is dropped with a warning.

        Works on a container, dropping from every variable that has `dim`, and on a single
        variable, returning a variable.

        **Two places this stops where xarray keeps going**, both because GDAL has no raster
        of no bands to put the answer in: a call that would drop *every* step raises rather
        than returning an empty cube (xarray answers shape `(0, …)`), and `thresh` must be
        at least 1, where xarray reads `thresh=0` or a negative one as "keep everything".
        `how` and `thresh` otherwise mean what they mean in xarray, `thresh` overriding
        `how` included.

        Args:
            dim: The non-spatial dimension to drop steps from.
            how: `"any"` (default) drops a step that holds any gap at all; `"all"` drops only
                a step with no valid cell. Ignored when `thresh` is given, as xarray ignores
                it.
            thresh: Keep a step holding at least this many valid cells, an integer of at
                least 1. `None` (default) defers to `how`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, holding the
            steps that survived with their own values — nothing is computed here, only
            selected. A CF-packed variable comes back unpacked, in physical units and
            declaring the unpacked fill, as it does from every other member here.

        Raises:
            TypeError: `thresh` is not an integer, or is a boolean.
            ValueError: `how` is neither `"any"` nor `"all"`; `thresh` is below 1; no step
                survives, which would leave a variable with no bands; the container has no
                data variables; or `dim` is not a band dimension of any gridded variable.

        Warns:
            UserWarning: A container's auxiliary variable spans `dim` and is dropped. The
                message names `dropna()`.

        Examples:
            - Drop the steps that hold a gap, and then only the empty ones:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, np.nan, 3.0, np.nan]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
              ... ).get_variable("t")
              >>> var.dropna("time").read_array().ravel().tolist()
              [1.0, 3.0]
              >>> var.dropna("time")._band_dim_values_map["time"]
              [0.0, 12.0]

              ```
        """
        if how not in _DROPNA_HOWS:
            raise ValueError(
                f"dropna() takes how={' or '.join(repr(one) for one in _DROPNA_HOWS)}, "
                f"got {how!r}."
            )
        op = _DropNa(how=how, thresh=_check_limit(thresh, caller="dropna"))
        return _along_either(self._ds, dim, op)

    def interpolate_na(
        self,
        dim: str,
        method: str = "linear",
        *,
        limit: int | None = None,
        use_coordinate: bool = True,
    ) -> NetCDF:
        """Fill the gaps along a non-spatial dimension from the valid cells around them.

        The temporal counterpart of the spatial `fill_gaps`: where `ffill` carries one
        neighbour forwards, this reads both sides of a gap and places it between them. A gap
        with a valid cell on only one side — a leading or trailing one — is left alone, as
        xarray leaves it.

        Works on a container, interpolating every variable that has `dim`, and on a single
        variable, returning a variable. A container's auxiliary variables are all carried
        over, those spanning `dim` included, since its length does not change.

        Args:
            dim: The non-spatial dimension to interpolate along.
            method: `"linear"` (default) places a gap between its neighbours in proportion to
                its distance from each; `"nearest"` gives it the closer neighbour's value,
                the earlier one when the distances are equal. The spline methods xarray
                offers are not implemented.
            limit: How many consecutive gaps one run may fill, counted from the valid cell
                before it exactly as `ffill`'s limit is, an integer of at least 1. `None`
                (default) fills a run of any length.
            use_coordinate: Measure the distance between steps along the dimension's own
                coordinate values (default), so an unevenly spaced axis interpolates by how
                far apart its steps really are. `False` measures by position, which is also
                what a dimension carrying no coordinates falls back to.

        Returns:
            NetCDF: A container for a container, a variable for a variable, float64, with
            every dimension unchanged in length and coordinates. It declares the variable's
            no-data value, or NaN when it declares none, since a gap that could not be
            reached is still a gap.

        Raises:
            TypeError: `limit` is not an integer, or is a boolean.
            ValueError: `method` is neither `"linear"` nor `"nearest"`; `limit` is below 1;
                `use_coordinate` was asked for and `dim`'s stamps are not numeric; the
                container has no data variables; or `dim` is not a band dimension of any
                gridded variable.

        Examples:
            - An interior gap is placed between its neighbours; the edges are left alone:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
              >>> var = NetCDF.from_array(
              ...     np.array([1.0, np.nan, np.nan, 7.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=None,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0, 2.0, 3.0]),
              ... ).get_variable("t")
              >>> var.interpolate_na("time").read_array().ravel().tolist()
              [1.0, 3.0, 5.0, 7.0]

              ```
            - On an uneven axis the distance is the coordinate's, not the position's:

              ```python
              >>> uneven = NetCDF.from_array(
              ...     np.array([1.0, np.nan, np.nan, 7.0]).reshape(4, 1, 1),
              ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
              ...     variable_name="t",
              ...     no_data_value=None,
              ...     dims=ExtraDimensions(name="time", values=[0.0, 1.0, 5.0, 6.0]),
              ... ).get_variable("t")
              >>> uneven.interpolate_na("time").read_array().ravel().tolist()
              [1.0, 2.0, 6.0, 7.0]

              ```
        """
        if method not in _INTERPOLATION_METHODS:
            raise ValueError(
                f"interpolate_na() takes method="
                f"{' or '.join(repr(one) for one in _INTERPOLATION_METHODS)}, got "
                f"{method!r}."
            )
        op = _Interpolate(
            method=method,
            limit=_check_limit(limit, caller="interpolate_na"),
            use_coordinate=bool(use_coordinate),
        )
        return _along_either(self._ds, dim, op)


_INTERPOLATION_METHODS = ("linear", "nearest")
"""The interpolations `interpolate_na` offers; xarray's spline methods are not implemented."""

_DROPNA_HOWS = ("any", "all")
"""The `how` modes of `dropna`, in xarray's vocabulary."""


def _along_either(nc: NetCDF, dim: str, op: Any) -> NetCDF:
    """Run `op` along `dim`, on whichever receiver `nc` is.

    Args:
        nc: The container or variable the member was called on.
        dim: The dimension the operation runs along.
        op: The operation.

    Returns:
        NetCDF: A container for a container, a variable for a variable.
    """
    if _reduces_as_a_variable(nc):
        return _apply_to_variable(nc, dim, op)
    return _apply_to_container(nc, dim, op)


def _check_limit(limit: Any, *, caller: str) -> int | None:
    """A `limit` or `thresh` as a positive `int`, or the refusal saying why it is not one.

    Args:
        limit: As passed; `None` asks for no limit.
        caller: The member named in the message. Required rather than defaulted, because a
            default is silently wrong for every member but one — `bfill` inherited `ffill`'s
            and reported a bad limit against a member the caller never called.

    Returns:
        int | None: `None` unchanged, otherwise the value as an `int`.

    Raises:
        TypeError: The value is a boolean or not something `operator.index()` accepts.
        ValueError: The value is below 1, which would fill or keep nothing.
    """
    checked: int | None = None
    if limit is not None:
        if isinstance(limit, (bool, np.bool_)):
            raise TypeError(f"{caller}() needs an integer, got {limit!r}.")
        try:
            checked = operator.index(limit)
        except TypeError:
            raise TypeError(f"{caller}() needs an integer, got {limit!r}.") from None
        if checked < 1:
            raise ValueError(f"{caller}() needs a value of at least 1, got {checked}.")
    return checked


_BOUNDARIES = ("exact", "trim", "pad")
"""The `boundary` modes of `coarsen`, in xarray's vocabulary."""


def _check_how(how: str, known: set[str]) -> None:
    """Refuse a reduction name the caller does not know.

    Shared by `reduce`, `coarsen`, `rolling` and `weighted`, which do not all offer the same
    set — `weighted` passes its own `_WEIGHTED_HOWS` — so the known names come in as an argument
    rather than being looked up here.

    Args:
        how: The requested reduction.
        known: Every reduction name the caller accepts.

    Raises:
        ValueError: `how` is not in `known`; the message lists them sorted.
    """
    if how not in known:
        raise ValueError(f"how must be one of {sorted(known)}; got {how!r}")


def _check_window(window: Any, *, caller: str) -> int:
    """A `coarsen` or `rolling` window as a positive `int`, or the refusal saying why it is not one.

    Args:
        window: The window as passed.
        caller: The member named in the message.

    Returns:
        int: The window length.

    Raises:
        TypeError: `window` is a boolean or not something `operator.index()` accepts.
        ValueError: `window` is below 1.
    """
    if isinstance(window, (bool, np.bool_)):
        raise TypeError(f"{caller}() needs an integer window, got {window!r}.")
    try:
        length = operator.index(window)
    except TypeError:
        raise TypeError(
            f"{caller}() needs an integer window, got {window!r}."
        ) from None
    if length < 1:
        raise ValueError(f"{caller}() needs a window of at least 1, got {length}.")
    return length


_DIFF_LABELS = ("upper", "lower")
"""Which of a difference's two steps labels it, in xarray's vocabulary."""


def _check_order(n: Any) -> int:
    """A `diff` order as a non-negative `int`, or the refusal saying why it is not one.

    Args:
        n: The order as passed.

    Returns:
        int: The order.

    Raises:
        TypeError: `n` is a boolean or not something `operator.index()` accepts.
        ValueError: `n` is negative.
    """
    if isinstance(n, (bool, np.bool_)):
        raise TypeError(f"diff() needs an integer order, got {n!r}.")
    try:
        order = operator.index(n)
    except TypeError:
        raise TypeError(f"diff() needs an integer order, got {n!r}.") from None
    if order < 0:
        raise ValueError(f"diff() needs a non-negative order, got {order}.")
    return order


def _check_periods(periods: Any) -> int:
    """A `shift` distance as an `int`, or the refusal saying why it is not one.

    Args:
        periods: The distance as passed; any sign is fine.

    Returns:
        int: The distance.

    Raises:
        TypeError: `periods` is a boolean or not something `operator.index()` accepts.
    """
    if isinstance(periods, (bool, np.bool_)):
        raise TypeError(f"shift() needs integer periods, got {periods!r}.")
    try:
        steps = operator.index(periods)
    except TypeError:
        raise TypeError(f"shift() needs integer periods, got {periods!r}.") from None
    return steps


def _check_fill_value(fill_value: Any) -> None:
    """Refuse a `shift` fill that is not a real number.

    A boolean is refused by name, as `q` refuses one: `True` is a `Real` equal to 1, and a band
    of flags is not what a boolean fill asks for.

    Args:
        fill_value: The fill as passed; `None` asks for the no-data value.

    Raises:
        TypeError: `fill_value` is neither `None` nor a real number, or is a boolean.
    """
    unusable = fill_value is not None and (
        isinstance(fill_value, (bool, np.bool_)) or not isinstance(fill_value, Real)
    )
    if unusable:
        raise TypeError(f"shift() needs a real fill_value or None, got {fill_value!r}.")


def _check_min_periods(min_periods: Any, window: int) -> int:
    """The valid cells a `rolling` window needs, or the refusal saying why `min_periods` is unusable.

    Args:
        min_periods: As passed; `None` asks for a whole window.
        window: The checked window length.

    Returns:
        int: `window` for `None`, otherwise `min_periods` as an `int`.

    Raises:
        TypeError: `min_periods` is a boolean or not something `operator.index()` accepts —
            a float, NaN included.
        ValueError: `min_periods` is below 1, or above `window`, which no window could meet.
    """
    needed = window
    if min_periods is not None:
        if isinstance(min_periods, (bool, np.bool_)):
            raise TypeError(
                f"rolling() needs an integer min_periods, got {min_periods!r}."
            )
        try:
            needed = operator.index(min_periods)
        except TypeError:
            raise TypeError(
                f"rolling() needs an integer min_periods, got {min_periods!r}."
            ) from None
        if needed < 1:
            raise ValueError(
                f"rolling() needs min_periods of at least 1, got {needed}."
            )
        if needed > window:
            raise ValueError(
                f"rolling() min_periods={needed} can never be met by a window of {window} "
                f"step(s); pass at most {window}."
            )
    return needed


_INTERP_KINDS = (
    "linear",
    "nearest",
    "cubic",
    "zero",
    "slinear",
    "quadratic",
    "previous",
    "next",
)
"""The `scipy.interpolate.interp1d` kinds `interp` / `interp_like` accept as `method`."""

_SPATIAL_AXIS_NAMES = {name.lower() for name in (*X_AXIS_NAMES, *Y_AXIS_NAMES)}
"""Axis names that identify the horizontal plane, which `interp` regrids through the warp path."""

_INTERP_MIN_POINTS = {
    "nearest": 2,
    "previous": 2,
    "next": 2,
    "zero": 2,
    "slinear": 2,
    "linear": 2,
    "quadratic": 3,
    "cubic": 4,
}
"""Minimum source steps each `interp1d` kind needs (spline order + 1; 2 for the piecewise kinds)."""


def _resolve_interp_kind(method: str, caller: str = "interp") -> str:
    """Return `method` if it is a supported `interp1d` kind, else refuse.

    Args:
        method: The interpolation kind the caller passed.
        caller: The member the user called (`"interp"` / `"interp_like"`), named in the refusal.

    Returns:
        str: `method`, unchanged.

    Raises:
        ValueError: `method` is not one of `_INTERP_KINDS`.
    """
    if method not in _INTERP_KINDS:
        raise ValueError(
            f"{caller}() method must be one of {list(_INTERP_KINDS)}, got {method!r}."
        )
    return method


def _assert_coordinate_partition(nc: NetCDF, *, caller: str) -> None:
    """Refuse the coordinate-role members on a single variable.

    CF marks a coordinate by naming it in *another* variable's `coordinates` attribute, so
    a lone variable has no sibling to carry the reference — and promoting the only variable
    there is would leave a cube with no data variables at all.

    A `get_group` view is refused as well, and for a sharper reason: the CF `coordinates`
    attribute names its references *relatively* (`expver`), while `classify_variables`
    reports a sub-group's arrays under their group-qualified names (`inner/expver`), so a
    reference written inside a group is never matched back to the array it names. The write
    would succeed and the role would not change — a silent no-op, which is worse than a
    refusal. The root container is where the roles are read, so that is where they are set.

    Args:
        nc: The receiver.
        caller: The member the user called, named in the refusal.

    Raises:
        ValueError: `nc` is a single variable, or a `get_group` view.
    """
    if _reduces_as_a_variable(nc):
        raise ValueError(
            f"{caller}() changes which of a container's variables are coordinates, and "
            f"this is a single variable: CF marks a coordinate by naming it in another "
            f"variable's `coordinates` attribute, so there is nothing here to name it. "
            f"Call it on the container this variable came from."
        )
    if nc._group_path:
        raise ValueError(
            f"{caller}() is not supported on a get_group() view ({nc._group_path!r}). A CF "
            f"`coordinates` reference is a relative name, while a sub-group's arrays are "
            f"classified under their group-qualified names, so a reference written here "
            f"would never be matched back and the role would silently not change. Open the "
            f"store without get_group() and set the roles on the root container."
        )


def _requested_names(names: str | Sequence[str]) -> list[str]:
    """One or several variable names, as a list, in the order given.

    Args:
        names: A single name or a sequence of them.

    Returns:
        list[str]: The names, de-duplicated with their first-seen order kept.
    """
    requested = [names] if isinstance(names, str) else list(names)
    return list(dict.fromkeys(requested))


def _cf_roles(nc: NetCDF) -> dict[str, str]:
    """The CF role `classify_variables` gives each of the store's arrays.

    Args:
        nc: The container.

    Returns:
        dict[str, str]: Array name to role, empty when the store declares no CF roles.
    """
    cf = nc.meta_data.cf
    return dict(cf.classifications or {}) if cf is not None else {}


def _auxiliary_coordinates(nc: NetCDF) -> list[str]:
    """The container's auxiliary-coordinate variables, in the store's own order.

    Args:
        nc: The container.

    Returns:
        list[str]: The names CF classifies as auxiliary coordinates.
    """
    roles = _cf_roles(nc)
    return [name for name, role in roles.items() if role == "auxiliary_coordinate"]


def _validated_promotions(nc: NetCDF, requested: list[str]) -> list[str]:
    """The names `set_coords` will promote, refusing the ones it cannot.

    A name that is *already* an auxiliary coordinate is dropped rather than refused, so
    promoting twice is idempotent — but a dimension coordinate is refused, because it is a
    coordinate by name convention and asking for it signals a different intent than this
    member can serve.

    Args:
        nc: The container.
        requested: The names the caller asked to promote.

    Returns:
        list[str]: The subset that needs promoting.

    Raises:
        ValueError: A name is a dimension, or is not a variable of this container.
    """
    dimensions = set(nc.dimension_names or [])
    existing = set(_auxiliary_coordinates(nc))
    variables = list(nc.variable_names)
    promoted: list[str] = []
    for name in requested:
        if name in dimensions:
            raise ValueError(
                f"set_coords(): {name!r} is a dimension of this cube, so it is already "
                f"its dimension's coordinate by CF name convention — a coordinate this "
                f"member cannot assign, because the netCDF driver does not support "
                f"reassigning a dimension's indexing variable. Only non-dimension "
                f"(auxiliary) coordinates are set here."
            )
        if name not in existing:
            if name not in variables:
                raise ValueError(
                    f"set_coords(): {name!r} is not a variable of this container; its "
                    f"variables are {sorted(variables)}."
                )
            promoted.append(name)
    return promoted


def _band_axes_of(nc: NetCDF, name: str) -> set[str]:
    """The band dimensions of one of a container's arrays, empty when it has none.

    A container's inventory is not all rasters: a 1-D or otherwise non-gridded array comes
    back from `get_variable` as a `LabeledArray`, which carries no band surface at all.
    `cast` is a type-checker annotation and not a runtime guard, so reading
    `_band_dim_names` off one raised `AttributeError` from inside a public member.

    Args:
        nc: The container.
        name: The array to inspect.

    Returns:
        set[str]: Its band dimension names, or an empty set when it tracks none.
    """
    return set(getattr(nc.get_variable(name), "_band_dim_names", ()) or ())


def _promotion_receivers(nc: NetCDF, promoted: list[str]) -> list[str]:
    """The data variables that will reference the promoted coordinates.

    CF's `coordinates` attribute means "these variables label *my* cells", so only a data
    variable spanning the promoted variable's own band dimensions can carry the reference.
    The promoted names themselves are excluded: a coordinate does not reference itself.

    Only **gridded** variables are offered the reference. A non-gridded array has no cells
    on the grid to label, and `_spatial_variable_names` is the same inventory the rest of
    the fan-out machinery uses; asking a non-gridded one for its band dimensions is what
    raised `AttributeError` on real CF stores.

    Args:
        nc: The container.
        promoted: The names being promoted.

    Returns:
        list[str]: The receiving data variables, in the container's order.

    Raises:
        ValueError: Nothing is left to reference the promotion.
    """
    gridded = nc._spatial_variable_names()
    receivers = [
        name for name in nc.variable_names if name not in promoted and name in gridded
    ]
    needed: set[str] = set()
    for name in promoted:
        needed |= _band_axes_of(nc, name)
    spanning = [name for name in receivers if needed <= _band_axes_of(nc, name)]
    if promoted and not spanning:
        why = (
            "none of the other variables span its dimensions"
            if receivers
            else "it is the container's only variable"
        )
        raise ValueError(
            f"set_coords(): no data variable is left to reference {sorted(promoted)} — "
            f"{why}. CF marks a coordinate by naming it in the `coordinates` attribute "
            f"of the variables it labels, so a promotion nothing can reference would "
            f"take the variable out of `data_vars` and leave it unreachable as a label."
        )
    return spanning


def _validated_demotions(nc: NetCDF, names: str | Sequence[str] | None) -> list[str]:
    """The names `reset_coords` will demote, refusing the ones it cannot.

    Args:
        nc: The container.
        names: The names the caller asked to demote, or `None` for every auxiliary
            coordinate.

    Returns:
        list[str]: The names to demote.

    Raises:
        ValueError: A name is a dimension coordinate, or is not an auxiliary coordinate of
            this container.
    """
    auxiliary = _auxiliary_coordinates(nc)
    dimensions = set(nc.dimension_names or [])
    # With `names=None` the list *is* the auxiliary set, so both checks below pass
    # trivially -- a dimension's coordinate is classified "coordinate", never "auxiliary".
    requested = list(auxiliary) if names is None else _requested_names(names)
    for name in requested:
        if name in dimensions:
            raise ValueError(
                f"reset_coords(): {name!r} is a dimension of this cube, and a dimension's "
                f"coordinate is its same-named array by CF convention, not a reference "
                f"that can be removed. Rename the array (`rename_variable`) if it should "
                f"stop being that dimension's coordinate."
            )
        if name not in auxiliary:
            raise ValueError(
                f"reset_coords(): {name!r} is not an auxiliary coordinate of this "
                f"container; its auxiliary coordinates are {sorted(auxiliary)}."
            )
    return requested


class _NotBroadcastable(ValueError):
    """Two band layouts that cannot be reconciled by repeating a length-one axis.

    A `ValueError` subclass, so `broadcast_like` keeps raising exactly what its `Raises:`
    section documents, while `broadcast_equals` can catch *this* and nothing else. Catching
    plain `ValueError` there turned two real defects into a confident `False`: a container
    receiver, which `equals` refuses, and a reshape mismatch inside `_broadcast_values`.
    """


def _grid_reference(nc: NetCDF) -> NetCDF:
    """A cube whose own raster describes the grid, for a variable-to-variable comparison.

    A root multidimensional container's raster is a placeholder — GDAL reports it as
    512x512 whatever the store holds — so a grid check against the container itself fails
    for every container, and one of its variables has to stand in.

    The variable is taken from `_spatial_variable_names`, the **gridded** inventory that
    `_apply_per_variable` itself fans out over, not from `variable_names`. Those two are
    not the same list and their order is unrelated: on a real CF store `variable_names[0]`
    is often a 1-D array (`hyai` on a hybrid-level store), which `get_variable` answers
    with a `LabeledArray` that has no `epsg`, `rows` or geotransform at all. Reading a grid
    off it raised `AttributeError` from inside a public member.

    A container with no gridded variable has no grid to compare, so it is handed back
    as-is and the caller's own grid check reports the mismatch.

    Args:
        nc: A variable or a container.

    Returns:
        NetCDF: The variable to compare grids with — `nc` itself when it is a variable or a
        container with no gridded variable, else that container's first gridded variable.
    """
    reference = nc
    if not _reduces_as_a_variable(nc):
        gridded = nc._spatial_variable_names()
        if gridded:
            reference = cast("NetCDF", nc.get_variable(gridded[0]))
    return reference


def _donor_band_layout(
    other: Any, *, caller: str
) -> list[tuple[str, int, list | None]]:
    """The band dimensions a broadcast donor offers, as `(name, size, values)`.

    A variable carries its own band bookkeeping; a container's dimensions come from the
    store, minus the spatial axes, with their coordinates read from the indexing arrays. A
    donor with no band dimensions offers nothing, which makes a broadcast against it a
    no-op rather than an error.

    Args:
        other: The donor cube.
        caller: The member the user called, named in the refusal.

    Returns:
        list[tuple[str, int, list | None]]: One entry per band dimension, in the donor's
        order, each with its size and coordinate values (`None` when it has none).

    Raises:
        TypeError: `other` is not a cube carrying the NetCDF band surface.
    """
    if not isinstance(other, Dataset) or not hasattr(other, "_band_dim_names"):
        raise TypeError(
            f"{caller}() takes the band layout from another NetCDF cube, not from "
            f"{type(other).__name__}."
        )
    # The `hasattr` above is the real check -- a plain `Dataset` has no band surface -- so
    # the cast carries that for the type checker rather than widening the signature.
    cube = cast("NetCDF", other)
    names = _band_dims_of(cube)
    if _reduces_as_a_variable(cube):
        sizes = dict(zip(cube._band_dim_names, cube._band_dim_sizes))
        stamps = {name: cube._band_dim_values_map.get(name) for name in names}
    else:
        declared = dict(cube.dimension_sizes or {})
        coordinates = cube.coords
        sizes = {name: int(declared[name]) for name in names if name in declared}
        stamps = {
            name: (list(coordinates[name]) if name in coordinates else None)
            for name in names
        }
    return [(name, sizes[name], stamps.get(name)) for name in names if name in sizes]


def _broadcast_layout(
    var: NetCDF, donor: list[tuple[str, int, list | None]]
) -> tuple[list[str], list[int], dict[str, list | None]]:
    """The band layout `var` takes when broadcast against `donor`.

    This variable's own dimensions keep their order and come first, then the donor's that
    it lacks — xarray's ordering, which makes a broadcast against a donor whose dimensions
    are a superset come back in exactly the donor's order.

    A dimension is taken from the donor when this variable does not have it at all, or has
    it at length one against the donor's longer axis. Otherwise this variable's own length
    and coordinates win, so a donor's length-one axis never shortens anything.

    Args:
        var: The variable being broadcast.
        donor: The donor's layout from `_donor_band_layout`.

    Returns:
        tuple: The result's dimension names, their sizes, and their coordinate values.

    Raises:
        ValueError: A shared dimension has different lengths on the two sides and neither
            is one, so neither can be stretched onto the other.
    """
    mine = list(var._band_dim_names)
    my_sizes = dict(zip(mine, var._band_dim_sizes))
    donor_sizes = {name: size for name, size, _ in donor}
    donor_stamps = {name: values for name, _, values in donor}
    names = [*mine, *[name for name, _, _ in donor if name not in mine]]
    sizes: list[int] = []
    values_map: dict[str, list | None] = {}
    for name in names:
        ours = my_sizes.get(name)
        theirs = donor_sizes.get(name)
        _assert_axes_stretch(name, ours, theirs)
        # Indexed, not `.get`: taking a size from the donor means the name came from the
        # donor, so it has one there -- and the index says so to the type checker too.
        if ours is None or (ours == 1 and theirs not in (None, 1)):
            sizes.append(donor_sizes[name])
            stamps = donor_stamps.get(name)
        else:
            sizes.append(ours)
            stamps = var._band_dim_values_map.get(name)
        values_map[name] = list(stamps) if stamps is not None else None
    return names, sizes, values_map


def _assert_axes_stretch(name: str, ours: int | None, theirs: int | None) -> None:
    """Refuse a shared band dimension neither side can be stretched onto.

    Broadcasting has no index to join on, so two axes of different lengths are only
    reconcilable when one of them is a single step to repeat.

    Args:
        name: The dimension's name.
        ours: Its length here, or `None` when this cube does not have it.
        theirs: Its length on the donor, or `None` when the donor does not have it.

    Raises:
        ValueError: Both sides have it, at different lengths, and neither is one.
    """
    if (
        ours is not None
        and theirs is not None
        and ours != theirs
        and 1 not in (ours, theirs)
    ):
        raise _NotBroadcastable(
            f"broadcast_like(): dimension {name!r} is {ours} long here and {theirs} long "
            f"on the other cube, and neither is length one, so there is nothing to "
            f"stretch. Broadcasting never joins two axes — select or interpolate one of "
            f"them onto the other's steps first."
        )


def _broadcast_values(
    values: np.ndarray,
    mine: list[str],
    my_sizes: list[int],
    names: list[str],
    sizes: list[int],
) -> np.typing.NDArray:
    """Repeat `values` onto the broadcast layout.

    The result's own dimensions lead `names`, so the source's band axes already sit in the
    right order and only the added ones have to be inserted — as length-one axes, which
    `np.broadcast_to` then stretches. The declared sizes drive the reshape rather than the
    array's own, so a materialised read carrying a spare leading axis is normalised here.
    The stretched view is made contiguous at the end: the repeats have to be real cells in
    the rebuilt variable, which is the cost this method is documented to have.

    Args:
        values: The source cells, with the band axes outermost and `(rows, columns)` last.
        mine: The source's band dimension names, outermost first.
        my_sizes: The source's band dimension sizes, aligned to `mine`.
        names: The result's band dimension names.
        sizes: The result's band dimension sizes, aligned to `names`.

    Returns:
        np.ndarray: The cells on the broadcast shape, contiguous.
    """
    spatial = values.shape[-2:]
    declared = dict(zip(mine, my_sizes))
    source = values.reshape((*my_sizes, *spatial))
    lifted = source.reshape((*[declared.get(name, 1) for name in names], *spatial))
    return np.ascontiguousarray(np.broadcast_to(lifted, (*sizes, *spatial)))


def _band_dims_of(nc: NetCDF) -> list[str]:
    """The band (non-spatial) dimension names of a variable or container.

    Args:
        nc: A variable or a container.

    Returns:
        list[str]: The variable's tracked band dimensions, or the container's dimensions minus the
        spatial axes.
    """
    if _reduces_as_a_variable(nc):
        return list(nc._band_dim_names)
    return [
        d for d in (nc.dimension_names or []) if d.lower() not in _SPATIAL_AXIS_NAMES
    ]


def _same_spatial_grid(nc: NetCDF, other: NetCDF) -> bool:
    """Whether two cubes occupy the same horizontal grid (CRS, size and geotransform).

    Compared from the raster properties directly rather than through `Dataset.same_grid`, so it
    holds for a root MDIM container whose geotransform is derived.

    Args:
        nc: One cube.
        other: The other cube.

    Returns:
        bool: `True` iff both share EPSG, row/column counts and geotransform.
    """
    return bool(
        nc.epsg == other.epsg
        and nc.rows == other.rows
        and nc.columns == other.columns
        and np.allclose(
            np.asarray(nc.geotransform, dtype="float64"),
            np.asarray(other.geotransform, dtype="float64"),
        )
    )


def _refuse_spatial_interp(nc: NetCDF, dim: str, *, caller: str) -> None:
    """Refuse interpolating a spatial axis, pointing at the operations that regrid it correctly.

    Args:
        nc: The cube being interpolated.
        dim: The dimension the caller named.
        caller: The member the user called, named in the message.

    Raises:
        ValueError: `dim` is a spatial axis (and not a band dimension of `nc`).
    """
    if dim.lower() in _SPATIAL_AXIS_NAMES and dim not in _band_dims_of(nc):
        raise ValueError(
            f"{caller}() interpolates only band (non-spatial) dimensions; {dim!r} is a spatial "
            "axis. For a new spatial grid use resample() (new cell size), to_crs() (new CRS) or "
            "align() (onto another dataset's grid); for scattered points use extract() / point()."
        )


def _interp_source_coordinates(
    nc: NetCDF, dim: str, caller: str = "interp"
) -> np.ndarray:
    """The numeric source coordinates of band dimension `dim`, for `interp` to interpolate from.

    Read from the variable's `_band_dim_values_map` (a variable) or the store's dimension values (a
    container), mirroring `_bin_coordinates`.

    Args:
        nc: The container or variable being interpolated (or `other`, read for its targets).
        dim: The band dimension to read.
        caller: The member the user called (`"interp"` / `"interp_like"`), named in refusals.

    Returns:
        np.ndarray: The coordinates as `float64`.

    Raises:
        ValueError: `dim` is not one of a variable's band dimensions (or not a dimension of the
            container), or its coordinates are absent (coordinate-less), non-numeric (text) or hold
            NaN.
    """
    coords: Any
    if _reduces_as_a_variable(nc):
        _assert_band_dimension(nc, dim, caller=caller)
        coords = nc._band_dim_values_map.get(dim)
    else:
        names = list(nc.dimension_names or [])
        if dim not in names:
            raise ValueError(
                f"{caller}() got {dim!r}, which is not a dimension of this container; its "
                f"dimensions are {names}."
            )
        coords = nc.get_dimension_values(dim)
    if coords is None:
        raise ValueError(
            f"{caller}() needs coordinate values on {dim!r} to interpolate from, but it carries "
            "none (a coordinate-less axis)."
        )
    values = np.asarray(coords)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"{caller}() needs a numeric {dim!r} axis to interpolate; its coordinates are not "
            "numbers."
        )
    values = values.astype("float64")
    if np.isnan(values).any():
        raise ValueError(
            f"{caller}() cannot interpolate along {dim!r}: its coordinates contain NaN."
        )
    if np.unique(values).size != values.size:
        # interp1d is uniquely sensitive to a repeated sample point -- it returns an arbitrary,
        # order-dependent value at the tie rather than erroring -- so refuse a duplicate stamp here.
        raise ValueError(
            f"{caller}() cannot interpolate along {dim!r}: its coordinates have duplicate values, "
            "which make the interpolation ambiguous. Deduplicate the axis first."
        )
    return values


def _interp_targets(target: Any, dim: str, caller: str = "interp") -> np.ndarray:
    """The target coordinate values to interpolate onto, as a validated 1-D float64 array.

    Args:
        target: A scalar or 1-D sequence of coordinate values.
        dim: The dimension the targets are for, named in refusals.
        caller: The member the user called (`"interp"` / `"interp_like"`), named in refusals.

    Returns:
        np.ndarray: The targets as a 1-D `float64` array.

    Raises:
        ValueError: `target` is not 1-D, is empty, or holds NaN.
    """
    values = np.atleast_1d(np.asarray(target, dtype="float64"))
    if values.ndim != 1:
        raise ValueError(
            f"{caller}() target for {dim!r} must be one-dimensional; got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError(f"{caller}() target for {dim!r} is empty.")
    if np.isnan(values).any():
        raise ValueError(f"{caller}() target for {dim!r} contains NaN.")
    return values


def _run_interp(
    nc: NetCDF, dim: str, target: Any, kind: str, caller: str = "interp"
) -> NetCDF:
    """Interpolate one band dimension of `nc` onto `target`, container or variable.

    Args:
        nc: The container or variable to interpolate.
        dim: The band dimension to interpolate along.
        target: The coordinate values to interpolate onto.
        kind: The resolved `interp1d` kind.
        caller: The member the user called (`"interp"` / `"interp_like"`), threaded into every
            refusal and the dropped-auxiliary warning so they name the real entry point.

    Returns:
        NetCDF: The interpolated container or variable.
    """
    _refuse_spatial_interp(nc, dim, caller=caller)
    source = _interp_source_coordinates(nc, dim, caller=caller)
    minimum = _INTERP_MIN_POINTS[kind]
    if source.size < minimum:
        raise ValueError(
            f"{caller}() method {kind!r} needs at least {minimum} source steps along {dim!r}, "
            f"but it has {source.size}. Use a lower-order method or a longer axis."
        )
    targets = _interp_targets(target, dim, caller=caller)
    op = _InterpTo(target=targets, kind=kind, caller=caller)
    if _reduces_as_a_variable(nc):
        return _apply_to_variable(nc, dim, op)
    return _apply_to_container(nc, dim, op)


def _pad_before_after(width: Any, dim: str) -> tuple[int, int]:
    """Parse a `pad` width for `dim` into a `(before, after)` pair of non-negative ints.

    Args:
        width: `(before, after)` or a single `int` applied to both sides.
        dim: The dimension the width is for, named in refusals.

    Returns:
        tuple[int, int]: The validated `(before, after)`.

    Raises:
        ValueError: `width` is not a 2-tuple or an int, holds a fractional or non-scalar
            value, or holds a negative value.
    """
    if isinstance(width, (tuple, list)):
        if len(width) != 2:
            raise ValueError(
                f"pad() width for {dim!r} must be (before, after) or an int, got {width!r}."
            )
        before, after = _pad_width_int(width[0], dim), _pad_width_int(width[1], dim)
    else:
        before = after = _pad_width_int(width, dim)
    if before < 0 or after < 0:
        raise ValueError(
            f"pad() widths for {dim!r} must be non-negative, got ({before}, {after})."
        )
    return before, after


def _pad_width_int(value: Any, dim: str) -> int:
    """Coerce a single pad width to an int, refusing a fractional or non-scalar value.

    Args:
        value: One side count, or one element of a `(before, after)` pair.
        dim: The dimension the width is for, named in refusals.

    Returns:
        int: The width as an int.

    Raises:
        ValueError: `value` is not a scalar whole number — a float with a fraction, an array,
            `None`, or a non-numeric string all raise here rather than being truncated with
            `int(...)` or leaking a numpy `TypeError`.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"pad() width for {dim!r} must be an int or (before, after), got {value!r}."
        ) from None
    if not number.is_integer():
        raise ValueError(
            f"pad() width for {dim!r} must be a whole number, got {value!r}."
        )
    return int(number)


def _pad_spatial(
    nc: NetCDF, dim: str, before: int, after: int, constant_values: Any
) -> NetCDF:
    """Pad a spatial axis, growing the grid and moving the geotransform with the data.

    Padding `before` cells on the left/top shifts the origin so the padded corner becomes the new
    top-left; the rebuilt grid's coordinates come from that geotransform. The fill is
    `constant_values` when given, else the variable's no-data value (NaN when none).

    Args:
        nc: The container or variable to pad.
        dim: The spatial axis (`x`/`y`/`lon`/`lat`).
        before: Cells to add at the left (x) or top (y).
        after: Cells to add at the right (x) or bottom (y).
        constant_values: The fill; `None` uses the variable's no-data value.

    Returns:
        NetCDF: The spatially padded container or variable.
    """
    is_x = dim.lower() in {name.lower() for name in X_AXIS_NAMES}

    def _fn(var: NetCDF) -> tuple:
        arr = np.asarray(
            nc._materialize_variable_array(var, lazy=True), dtype="float64"
        )
        ndv = _read_no_data(var)
        no_data: Any = np.nan if ndv is None else ndv
        fill = no_data if constant_values is None else constant_values
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1 if is_x else -2] = (before, after)
        padded = np.pad(arr, pad_width, mode="constant", constant_values=fill)
        gt = GeoTransform(*var.geotransform)
        new_gt = (
            gt._replace(x_origin=gt.x_origin - before * gt.pixel_width)
            if is_x
            else gt._replace(y_origin=gt.y_origin - before * gt.pixel_height)
        )
        return (
            padded,
            list(var._band_dim_names),
            dict(var._band_dim_values_map),
            no_data,
            tuple(new_gt),
        )

    return _apply_per_variable(nc, _fn, caller="pad")


def _transpose_order(band_names: list[str], dims: tuple) -> list[str]:
    """The new band-dimension order for one variable, given the requested `dims`.

    Empty `dims` reverses the band order; `...` (Ellipsis) expands to the band dimensions not named,
    in their current order; otherwise every band dimension must be named.

    Args:
        band_names: The variable's current band dimensions.
        dims: The requested order (may contain `...`).

    Returns:
        list[str]: The new band order for this variable.

    Raises:
        ValueError: `dims` has no `...` and does not name every band dimension.
    """
    present = [d for d in dims if d is not Ellipsis and d in band_names]
    if not dims:
        order: list[str] = list(reversed(band_names))
    elif Ellipsis in dims:
        rest = [name for name in band_names if name not in present]
        order = []
        for d in dims:
            if d is Ellipsis:
                order.extend(rest)
            elif d in band_names:
                order.append(d)
    elif set(present) != set(band_names):
        raise ValueError(
            f"transpose() must name every band dimension {sorted(band_names)} (or use ...); "
            f"got {[d for d in dims if d is not Ellipsis]!r}."
        )
    else:
        order = present
    return order


def _validate_transpose_dims(nc: NetCDF, dims: tuple) -> None:
    """Refuse a `transpose` that names a spatial axis, an unknown dim, or a duplicate.

    Args:
        nc: The container or variable being transposed.
        dims: The requested order (may contain `...`).

    Raises:
        ValueError: A named entry is not a string/`...`, `...` is given more than once, an entry is
            a spatial axis, is not a band dimension of the cube, or is duplicated.
    """
    if sum(1 for d in dims if d is Ellipsis) > 1:
        raise ValueError(
            "transpose() got more than one ...; an ellipsis (...) may be used at most once."
        )
    explicit = [d for d in dims if d is not Ellipsis]
    for d in explicit:
        if not isinstance(d, str):
            raise ValueError(
                f"transpose() dimensions must be strings or ..., got {d!r}."
            )
    band = set(_band_dims_of(nc))
    spatial = [
        d for d in explicit if d.lower() in _SPATIAL_AXIS_NAMES and d not in band
    ]
    if spatial:
        raise ValueError(
            f"transpose() reorders only band (non-spatial) dimensions; {spatial!r} is/are spatial "
            "axes pinned as the trailing (row, column) plane by the geotransform."
        )
    unknown = [d for d in explicit if d not in band]
    if unknown:
        raise ValueError(
            f"transpose() got {unknown!r}, which are not band dimensions of this cube; its band "
            f"dimensions are {sorted(band)}."
        )
    if len(set(explicit)) != len(explicit):
        raise ValueError(f"transpose() got duplicate dimensions: {explicit!r}.")


def _bin_coordinates(nc: NetCDF, dim: str) -> np.ndarray:
    """The numeric coordinates of band dimension `dim`, for `groupby_bins` to cut into bins.

    Read from the variable's `_band_dim_values_map` (a variable) or the store's dimension
    values (a container). On a variable, `_assert_band_dimension` refuses a spatial or unknown
    name up front with the wrong-dimension message `sel` / `isel` share. A text or
    coordinate-less axis (including a container dimension with no coordinate variable) is
    refused here, since it cannot be cut into value intervals; a container's spatial axis reads
    back its own numeric coordinates and is left for the reduce path to refuse.

    Args:
        nc: The container or variable `groupby_bins` was called on.
        dim: The band dimension to read.

    Returns:
        np.ndarray: The coordinates as `float64`.

    Raises:
        ValueError: `dim` is not one of a variable's band dimensions (or not a dimension of the
            container at all), or its coordinates are absent (a coordinate-less axis),
            non-numeric (a text axis), or contain `NaN`.
    """
    coords: Any
    if _reduces_as_a_variable(nc):
        _assert_band_dimension(nc, dim, caller="groupby_bins")
        coords = nc._band_dim_values_map.get(dim)
    else:
        names = list(nc.dimension_names or [])
        if dim not in names:
            raise ValueError(
                f"groupby_bins() got {dim!r}, which is not a dimension of this container; "
                f"its dimensions are {names}."
            )
        coords = nc.get_dimension_values(dim)
    if coords is None:
        raise ValueError(
            f"groupby_bins() needs numeric coordinates for {dim!r} to cut into bins, but it "
            "carries none (a coordinate-less axis)."
        )
    values = np.asarray(coords)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"groupby_bins() needs a numeric {dim!r} axis to cut into bins; its coordinates "
            "are not numbers."
        )
    values = values.astype("float64")
    if np.isnan(values).any():
        raise ValueError(
            f"groupby_bins() cannot bin {dim!r}: its coordinates contain NaN, which falls in "
            "no interval. Drop or fill the NaN coordinate first."
        )
    return values


def _bin_membership(
    coords: np.ndarray,
    bins: int | Sequence[float],
    *,
    right: bool,
    include_lowest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """The bin edges and each coordinate's bin index, via `pandas.cut`.

    Args:
        coords: The numeric coordinates to bin.
        bins: An `int` count of equal-width bins, or a sequence of strictly increasing edges.
        right: Whether the intervals are right-closed `(a, b]`.
        include_lowest: Whether the first edge joins the first bin (explicit edges only; an
            `int` `bins` always includes it, as `pandas.cut` does).

    Returns:
        tuple: `(edges, codes)` — the `float64` edges (`n` of them, `n - 1` bins) and one
        bin index per coordinate, `NaN` for a coordinate outside every bin.

    Raises:
        ValueError: `bins` is not an `int` count or a sequence of edges (a `bool` or a `float`
            is refused); an `int` count below one; an `int` count for a constant axis (every
            coordinate shares one value, so there is no range to divide — pass explicit edges);
            or an explicit sequence with fewer than two edges or edges that are not strictly
            increasing.
    """
    if isinstance(bins, bool):
        raise ValueError(
            "groupby_bins() bins must be an int count or a sequence of edges, not a bool."
        )
    if isinstance(bins, (int, np.integer)):
        if int(bins) < 1:
            raise ValueError(
                f"groupby_bins() needs at least one bin; got bins={bins!r}."
            )
        if float(coords.min()) == float(coords.max()):
            # A constant axis has no range to divide, and `np.histogram_bin_edges` would expand
            # it by +-0.5 so the first label falls below the value. Refuse it (pass explicit
            # edges for a constant axis) rather than mislabel the bin.
            raise ValueError(
                f"groupby_bins() cannot form {int(bins)} equal-width bins from a constant axis "
                f"(every coordinate is {float(coords.min())}); pass explicit edges instead."
            )
        # Edges span exactly [min, max], so the first label is the data minimum rather than
        # pandas' range-expanded edge (`cut(x, int)` widens the ends ~0.1%, which for a level
        # axis starting at 0 gives a negative first label). The int form is right-closed with
        # the lowest edge included; `right` / `include_lowest` apply to explicit edges only.
        edges = np.histogram_bin_edges(coords, bins=int(bins))
        codes = pd.cut(coords, edges, right=True, include_lowest=True, labels=False)
    elif isinstance(bins, (list, tuple, np.ndarray)):
        edges = np.asarray(list(bins), dtype="float64")
        if edges.size < 2:
            raise ValueError(
                f"groupby_bins() needs at least two bin edges to form a bin; got {list(edges)}."
            )
        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                f"groupby_bins() needs strictly increasing bin edges; got {list(edges)}."
            )
        codes = pd.cut(
            coords, edges, right=right, include_lowest=include_lowest, labels=False
        )
    else:
        raise ValueError(
            "groupby_bins() bins must be an int count of equal-width bins or a sequence of "
            f"edges; got {type(bins).__name__}."
        )
    return np.asarray(edges, dtype="float64"), np.asarray(codes, dtype="float64")


def _band_dimension_size(nc: NetCDF, dim: str, *, is_variable: bool) -> int:
    """The length of band dimension `dim`, from the variable or the container's first gridded one.

    Args:
        nc: A variable or a container.
        dim: The dimension.
        is_variable: Whether `nc` is a single variable.

    Returns:
        int: The number of steps along `dim`.

    Raises:
        ValueError: The variable has no band dimension `dim`; the container has no data
            variables; or none of its gridded variables has `dim`.
    """
    size = None
    if is_variable:
        _assert_band_dimension(nc, dim, caller="coarsen")
        size = nc._band_dim_sizes[list(nc._band_dim_names).index(dim)]
    else:
        if not nc.variable_names:
            raise ValueError("Cannot coarsen an empty container (no data variables).")
        for name in nc._spatial_variable_names(nc._working_group()):
            var = nc._require_raster_variable(name)
            if dim in var._band_dim_names:
                size = var._band_dim_sizes[list(var._band_dim_names).index(dim)]
                break
        if size is None:
            raise ValueError(
                f"Dimension {dim!r} is not a non-spatial dimension of any "
                f"variable in this container."
            )
    return int(size)


def _coarsen_windows(
    dim: str, size: int, window: int, boundary: str
) -> tuple[int, list[np.ndarray]]:
    """The axis length `coarsen` reduces over, and the positions in each window.

    Args:
        dim: The dimension, for the messages.
        size: Its length.
        window: Steps per window.
        boundary: `"exact"`, `"trim"` or `"pad"`. Any other value is treated as `"pad"`,
            so the caller validates it first.

    Returns:
        tuple: The resized length — `size` for `exact`, the largest multiple of `window`
        not above it for `trim`, the smallest not below it for `pad` — and one array of
        positions per window over that length.

    Raises:
        ValueError: `exact` and `window` does not divide `size`, or `trim` and `window`
            is longer than `size`.
    """
    if boundary == "exact":
        if size % window:
            raise ValueError(
                f"cannot coarsen {dim!r} of length {size} into windows of {window} with "
                f"boundary='exact': {size % window} step(s) would be left over. Pass "
                f"boundary='trim' to drop them, or boundary='pad' to reduce them as a "
                f"shorter last window."
            )
        resized = size
    elif boundary == "trim":
        resized = size // window * window
        if resized == 0:
            raise ValueError(
                f"boundary='trim' would leave nothing of {dim!r}: its length {size} is "
                f"shorter than the window {window}. Pass boundary='pad' to reduce it as "
                f"one window."
            )
    else:
        resized = -(-size // window) * window
    positions = [
        np.arange(start, start + window) for start in range(0, resized, window)
    ]
    return resized, positions


def _check_quantile(how: str, q: Any) -> float | None:
    """Refuse a `q` that is missing for `"quantile"` or given to any other reducer.

    NaN fails the range test by itself — every comparison with it is false — so it needs no
    case of its own; `bool` is refused by name because `True` is a `Real` equal to 1.

    An accepted `q` is handed back as a plain `float`. Any `numbers.Real` passes the check,
    and numpy types some of them — `fractions.Fraction(1, 2)` — as `object`, which its
    quantile functions cannot take; the operators narrow such a scalar the same way.

    Args:
        how: The requested reduction.
        q: The quantile argument as passed.

    Returns:
        float | None: `q` as a `float` for `"quantile"`, `None` for every other `how`.

    Raises:
        ValueError: `how` is `"quantile"` and `q` is not one real number in `[0, 1]`, or
            `how` is anything else and `q` is not `None`.
    """
    if how == "quantile":
        usable = (
            isinstance(q, Real)
            and not isinstance(q, (bool, np.bool_))
            and 0 <= float(q) <= 1
        )
        if not usable:
            raise ValueError(
                f"how='quantile' needs q= set to one number in [0, 1], got {q!r}."
            )
        checked = float(q)
    elif q is not None:
        raise ValueError(
            f"q= is only meaningful with how='quantile', got how={how!r} and q={q!r}."
        )
    else:
        checked = None
    return checked


def _curvilinear_coords_2d(
    nc: NetCDF,
) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
    """Return the variable's 2-D ``(lon, lat)`` coords when it is curvilinear.

    A curvilinear variable carries 2-D longitude/latitude arrays (no single affine
    geotransform). Returns the coordinate pair when both are 2-D, else ``None`` for
    a rectilinear grid. Shared by the plain and antimeridian crop paths.
    """
    curv = NetCDFPlot(nc)._resolve_curvilinear_coords(nc, coords=None)
    is_2d = (
        curv is not None
        and np.asarray(curv[0]).ndim == 2
        and np.asarray(curv[1]).ndim == 2
    )
    result = None
    if is_2d:
        result = cast("tuple[np.typing.NDArray, np.typing.NDArray]", curv)
    return result


def _reject_antimeridian_chunks(chunks: Any) -> None:
    """Reject ``chunks`` on an eager antimeridian path (rectilinear stitch / container).

    Only the curvilinear antimeridian path (a windowed read) honours ``chunks``; the
    rectilinear stitch and the container fan-out read the wrapped halves eagerly.
    """
    if chunks is not None:
        raise ValueError(
            "chunks= is not supported for an antimeridian crop; it is eager "
            "(the wrapped halves are read and concatenated)."
        )


def _lon_cell_size(lon2d: np.typing.NDArray) -> float:
    """Return the median centre-to-centre longitude spacing of a 2-D lon array.

    Used as the one-cell seam tolerance for a curvilinear grid. Each row of a
    -180..180 grid has one ~360 jump at the dateline; the median rejects it only
    when it is a strict minority, i.e. for ``nx >= 4`` columns (for ``nx <= 3`` the
    median collapses toward the jump, inflating the estimate). This is harmless
    here: the inflated tolerance only widens the ``lon_max > 180 + cell_x`` test in
    :func:`_split_lon_bbox`, and a -180..180 grid has ``lon_max < 180``, so the
    0..360 branch is never wrongly taken. A single-column grid (or all-NaN
    coordinates) has no spacing to measure and yields 0.0; the all-finite guard
    avoids a NumPy "All-NaN slice" RuntimeWarning on degenerate arrays.
    """
    size = 0.0
    if lon2d.shape[-1] >= 2:
        diffs = np.abs(np.diff(lon2d, axis=-1))
        if np.any(np.isfinite(diffs)):
            size = float(np.nanmedian(diffs))
    return size


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
) -> np.typing.NDArray:
    """Read just the ``(r0:r1, c0:c1)`` bounding window of a curvilinear variable.

    With ``chunks`` the read goes through the dask-backed lazy path (only the
    overlapping chunks materialise); otherwise GDAL reads just the ``(c0, r0)``–
    ``(c1, r1)`` block eagerly. Both paths return the variable's native
    dimension-preserving ``(*band_sizes, rows, cols)`` shape (a size-1 band axis kept,
    a 2-D plane for a variable with no band dimensions), so the crop keeps the band
    structure the caller's :meth:`from_array` rebuild reconstructs from that shape — the
    eager and lazy crops produce the identical layout (#1241). Helper of
    :meth:`Selection._crop_curvilinear`.

    Reads with ``unpack=False``. The caller stamps the variable's **stored** no-data
    sentinel into the cells outside the cutline and rebuilds the variable around the
    result, carrying the packing with it — a copy of the store, so it moves counts.
    A physical read would put ``-98.49``-shaped values next to a ``-9999`` fill and
    declare the recipe over both.
    """
    if chunks is not None:
        # Keep the native ``(*band_sizes, rows, cols)`` shape (no flatten) so the lazy crop
        # rebuilds the same band structure the eager crop does (#1241).
        lazy = nc.read_array(chunks=chunks, unpack=False)
        return np.array(cast("Any", lazy[..., r0:r1, c0:c1]).compute(), copy=True)
    return np.array(
        nc.read_array(window=[c0, r0, c1 - c0, r1 - r0], unpack=False), copy=True
    )


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


def _undecodable_label_hint(
    nc: NetCDF, dim_name: str, coords: list, selector: Any
) -> str:
    """Explain a failed label match on an axis whose CF values would not decode.

    Without this the caller sees the stored offsets and no reason why their label found
    nothing — the axis *does* declare `units`, so "it is not a time axis" would be the
    wrong conclusion to draw.

    Two probes, both only on the failure path. The first coordinate answers "is this a time
    axis at all", so an axis with no CF `units` gets no hint; the whole axis answers "was
    the caller shown stored numbers", because an axis that decodes end to end was shown
    dates and this sentence would contradict the list above it.

    Args:
        nc: The variable subset being selected.
        dim_name: Name of the band dimension.
        coords: That dimension's stored coordinate values.
        selector: The selector that matched nothing.

    Returns:
        str: A trailing sentence for the error, or `""` when the axis simply has no
            CF `units` (in which case the stored values are the whole story), and when the
            axis decodes end to end (in which case the caller was shown dates, not stored
            values, and the sentence would contradict them).

    Examples:
        - An axis that decodes end to end gets no hint, because the values the caller was
          shown are already dates:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _undecodable_label_hint
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> coords = [0.0, 6.0, 12.0, 18.0]
            >>> _undecodable_label_hint(nc, "time", coords, "1800-01-01")
            ''

            ```

        - A value the converter cannot handle leaves the axis undecodable, and the caller
          is told why their label matched nothing:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _undecodable_label_hint
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> hint = _undecodable_label_hint(
            ...     nc, "time", [float("nan")], "1800-01-01"
            ... )
            >>> hint[:52]
            " The 'time' axis declares CF units, but a coordinate"

            ```

        - A non-temporal axis gets no hint either, since its stored values are the whole
          story:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _undecodable_label_hint
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> _undecodable_label_hint(
            ...     nc, "pressure_level", [1000.0, 850.0, 500.0], "1800-01-01"
            ... )
            ''

            ```

    See Also:
        _decodes: Answers each of the two probes.
    """
    hint = ""
    if has_label(selector) and coords:
        # Two questions, not one. The first value answers "is this a time axis at all" — an
        # axis with no CF units decodes nothing and gets no hint. The whole axis answers
        # "was the caller shown stored numbers": if every value decodes, the vocabulary in
        # the message is dates and the hint would contradict it. Probing only the first
        # value conflated the two, so once an axis started decoding (#1140) the message read
        # "could not be decoded ... these are its stored values" above a list of dates.
        looks_temporal = _decodes(nc, dim_name, coords[:1], on_error=True)
        shown_as_stored = not _decodes(nc, dim_name, coords, on_error=False)
        if looks_temporal and shown_as_stored:
            hint = (
                f" The {dim_name!r} axis declares CF units, but a coordinate value could"
                " not be decoded, so it has no labels to match — these are its stored"
                " values."
            )
    return hint


def _decodes(nc: NetCDF, dim_name: str, coords: Any, *, on_error: bool) -> bool:
    """Whether `coords` decode to time labels on `dim_name`.

    Never raises: a decode that blows up is reported as `on_error`, so a probe on the error
    path cannot itself become the error the caller sees.

    Args:
        nc: The cube the dimension belongs to.
        dim_name: The band dimension's name.
        coords: The stored values to try.
        on_error: The answer when the decode *raises*, which the two probes in
            `_undecodable_label_hint` read differently. The probe asking whether the axis is
            temporal passes `True`, so a raise counts as "a time axis that failed"; the one
            asking whether the caller was shown stored numbers passes `False`, so the same
            raise counts as "not fully decoded" and the hint is kept.

    Returns:
        `True` when every value decodes, `False` when the axis has no parseable CF `units`,
        and `on_error` when the decode raised.

    Examples:
        - A CF time axis decodes, so both answers agree and `on_error` never comes up:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _decodes
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> _decodes(nc, "time", [0.0, 6.0], on_error=True)
            True
            >>> _decodes(nc, "time", [0.0, 6.0], on_error=False)
            True

            ```

        - A non-temporal axis has no CF `units` to decode with, which is a `False` of its
          own rather than an error:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _decodes
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> _decodes(nc, "pressure_level", [1000.0], on_error=True)
            False

            ```

        - A value the converter chokes on is where the two answers part company:
            ```python
            >>> from pyramids.netcdf.netcdf import NetCDF
            >>> from pyramids.netcdf.engines.selection import _decodes
            >>> nc = NetCDF.read_file(
            ...     "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
            ... )
            >>> _decodes(nc, "time", [float("nan")], on_error=True)
            True
            >>> _decodes(nc, "time", [float("nan")], on_error=False)
            False

            ```

    See Also:
        _undecodable_label_hint: The caller, which probes twice with opposite `on_error`.
    """
    try:
        decoded = nc._decode_time_labels(dim_name, coords, FULL_FORMAT) is not None
    except Exception:
        decoded = on_error
    return decoded


def _probe_label_format(
    dim_name: str, selector: Any, decode: Callable[[str], list[str]]
) -> str | None:
    """The precision a label selector needs, after rejecting the ones that make no sense.

    Helper of :func:`_resolve_selector_indices`, split out to keep the mode choice there
    readable. Refuses a selector that mixes the two vocabularies, and -- on an axis that
    *does* decode -- a string that is not a date-label shape, which is a malformed label
    rather than a stored value.

    Args:
        dim_name: Name of the band dimension, for the error messages.
        selector: The selector, already known to carry at least one string.
        decode: The memoised axis decoder.

    Returns:
        str or None: The strftime format to match at, or ``None`` when the selector's
            string is not date-shaped and the axis has no labels anyway.

    Raises:
        ValueError: The selector mixes labels with stored values, or names a date
            precision that is not supported on an axis that decodes.
    """
    stray = non_label_parts(selector)
    if stray:
        raise ValueError(
            f"{dim_name}={selector!r} mixes date labels with stored values "
            f"({stray[0]!r}). Select by label or by stored value, not both."
        )
    probe = probe_format(selector)
    if probe is None and decode(FULL_FORMAT):
        # The axis decodes as time, so a string that is not a date-label shape is a
        # malformed label, not a stored value -- `label_format` raises with the
        # precisions that would have worked. On an axis that does *not* decode, the same
        # string is simply a stored value (a string-valued coordinate variable, an
        # ensemble member's name) and falls through to exact matching.
        label_format(cast("str", first_label(selector)))
    return probe


def _nearest_or_raise(
    dim_name: str,
    coords: list,
    selector: Any,
    probe: str | None,
    *,
    is_label: bool,
    tolerance: float | None = None,
) -> list[int]:
    """Snap a numeric selector, or explain why this one cannot be snapped.

    Helper of :func:`_resolve_selector_indices`. The two refusals differ because the
    advice does: a date label already names a period, while a non-date string has nothing
    to measure at all.

    Args:
        dim_name: Name of the band dimension, for the error messages.
        coords: That dimension's stored coordinate values.
        selector: The selector handed to ``sel``.
        probe: The label format resolved for the selector, or ``None``.
        is_label: Whether the selector carries a string at all.
        tolerance: The furthest a snap may travel; ``None`` accepts any distance.

    Returns:
        list[int]: Indices of the snapped coordinates.

    Raises:
        ValueError: The selector is a date label, or is otherwise not a number. Also
            raised by :func:`nearest_indices` for a non-numeric axis, a non-finite
            selector, and a negative ``tolerance``.
        KeyError: Raised through :func:`nearest_indices` when a request's closest
            coordinate lies further away than ``tolerance``.
    """
    if is_label and probe is not None:
        raise ValueError(
            f"method='nearest' needs a numeric selector; {dim_name}={selector!r} is a "
            "date label. Select a label exactly -- a partial label such as '2024-01' "
            "already matches every step inside it."
        )
    if is_label:
        # Not date-shaped, so the date-label advice would be nonsense: this is a string
        # on an axis that has no labels at all (a pressure level written "850", an
        # ensemble member's name).
        raise ValueError(
            f"method='nearest' needs a numeric selector; {dim_name}={selector!r} is "
            "not a number. Snapping compares distances, so it has nothing to measure."
        )
    return nearest_indices(coords, selector, tolerance)


def _resolve_selector_indices(
    nc: NetCDF,
    dim_name: str,
    coords: list,
    selector: Any,
    method: str | None = None,
    tolerance: float | None = None,
) -> tuple[list[int], list]:
    """Resolve one ``sel`` selector to band-dim indices, and the values it matched against.

    The single place the three matching modes are chosen between, so :meth:`Selection.sel`
    and the plot engine's ``_flat_band_index`` cannot drift apart:

    * ``method="nearest"`` snaps each numeric request to the closest coordinate.
    * A selector carrying a date-shaped string is a **label**; the axis is decoded with
      the dimension's CF ``units`` / ``calendar`` and matched as text. An axis that cannot
      be decoded has no labels, so the stored-value path runs and the caller reports "no
      bands match" against the raw values.
    * Everything else matches the stored values exactly.

    Args:
        nc: The variable subset being selected (carries the CF dimension attributes).
        dim_name: Name of the band dimension the selector applies to.
        coords: That dimension's stored coordinate values.
        selector: The value, list, or :class:`slice` handed to ``sel``.
        method: ``None`` for exact matching, ``"nearest"`` to snap. Defaults to ``None``.
        tolerance: The furthest a ``method="nearest"`` snap may travel; ``None`` (the
            default) accepts any distance, and it is forwarded only on the ``"nearest"``
            path. ``Selection.sel`` refuses a bound whenever ``method`` is not
            ``"nearest"``, so the label and stored-value paths below never see one. With
            ``method="nearest"`` *and* a date label the bound does arrive here, and
            :func:`_nearest_or_raise` then refuses the **selector** rather than the
            bound.

    Returns:
        tuple[list[int], list]: The matching indices along ``dim_name``, and the values
            they were matched against -- the decoded labels on a label selection, the
            stored coordinates otherwise -- so the caller's error message quotes the
            vocabulary the caller actually used.

    Raises:
        ValueError: The selector mixes labels with stored values, names an unsupported
            date precision, or asks ``"nearest"`` of something that is not a number.
        KeyError: A ``method="nearest"`` request found no coordinate within ``tolerance``.
    """
    decoded: dict[str, list[str]] = {}

    def decode(fmt: str) -> list[str]:
        """Decode the axis at one precision, once; empty when it cannot be decoded at all.

        ``strict=False`` is the selection contract: a coordinate **value** the converter
        chokes on -- a ``_FillValue``, an infinity, an out-of-range offset, a string, or
        anything cftime refuses on a non-standard calendar -- means this axis has no
        labels, not that the caller's ``sel`` should abort. The stored-value path can
        still answer it.
        """
        if fmt not in decoded:
            decoded[fmt] = (
                nc._decode_time_labels(dim_name, coords, fmt, strict=False) or []
            )
        return decoded[fmt]

    # Probe at the precision this selector needs rather than always at FULL_FORMAT: the
    # match then answers from the same memoised pass, so a decodable axis is decoded once
    # per distinct precision instead of up to four times over the whole coordinate
    # variable -- which on a 128k-step cloud axis is the difference between one cftime
    # pass per `sel` and four, one of them only ever used to build an error string.
    is_label = has_label(selector)
    probe = _probe_label_format(dim_name, selector, decode) if is_label else None
    if method == "nearest":
        indices = _nearest_or_raise(
            dim_name, coords, selector, probe, is_label=is_label, tolerance=tolerance
        )
        available = coords
    elif probe is not None and decode(probe):
        indices = label_indices(decode, selector)
        # Only a failed match needs the vocabulary spelled out, and only then is the
        # full-precision decode worth paying for.
        available = cast("list", decode(FULL_FORMAT)) if not indices else coords
    else:
        try:
            indices = _resolve_dim_indices(coords, selector)
        except TypeError:
            # A stored-value comparison the types cannot answer -- a string-bounded slice
            # against a numeric axis, which is what a label slice degrades to when the
            # axis has no labels. A scalar or a list simply never equals any coordinate
            # and reports "no bands match"; a slice used to raise `TypeError` instead,
            # outside the documented contract. Nothing matches, which is what the caller
            # is then told.
            indices = []
        available = coords
    return indices, available


def _labelled_positions(
    nc: NetCDF, dim_name: str, selector: Any, errors: str
) -> set[int]:
    """The positions `drop_sel` should remove along one dimension.

    Args:
        nc: The variable.
        dim_name: The dimension, already validated as one with coordinates.
        selector: What the caller passed for it — a label, a sequence of them, or a mask.
        errors: `"raise"` to refuse a label the dimension does not hold, `"ignore"` to
            skip it.

    Returns:
        set[int]: The positions to drop.

    Raises:
        KeyError: A label is nowhere on the dimension and `errors="raise"`. The message
            pairs each missing label with the resolver's own reason for it.
    """
    coords = nc._band_dim_values_map.get(dim_name)
    wanted = _as_a_sequence_of_labels(selector)
    # A mask is one selector, not a sequence of labels: dropping what `sel` would keep is
    # the complement the two members are paired as. xarray raises
    # `KeyError: '[True, False, ...] not found in axis'` here instead.
    masked = _mask_positions(wanted, coords, dim_name)
    if masked is not None:
        return set(masked)
    if not isinstance(wanted, (list, tuple)):
        wanted = [wanted]
    dropped: set[int] = set()
    missing = []
    reasons = []
    for label in wanted:
        try:
            dropped.update(_resolve_one_dim(nc, dim_name, label, None, None))
        except (ValueError, KeyError) as unmatched:
            missing.append(label)
            reasons.append(str(unmatched).strip("\"'"))
    if missing and errors == "raise":
        explained = "; ".join(
            f"{label!r}: {reason}" for label, reason in zip(missing, reasons)
        )
        raise KeyError(
            f"drop_sel() found {missing!r} nowhere on {dim_name!r}. "
            f"{explained.rstrip('.')}. "
            f"Pass errors='ignore' to drop the labels that are there and skip the rest."
        )
    return dropped


def _mask_positions(
    selector: Any, coords: list | None, dim_name: str
) -> list[int] | None:
    """The positions a boolean mask selects, or `None` when `selector` is not a mask.

    xarray reads a boolean array or list of the axis' own length as a mask —
    `da.sel(time=[True, False, True, False])` keeps the first and third steps — and so does
    this. Read as labels instead, `True` and `False` match the coordinates `1.0` and `0.0`,
    so a mask quietly answered one band; that is a wrong answer rather than a loud one, and
    the ndarray spelling is exactly what `other.time.values > 6` hands over.

    Args:
        selector: The already-normalised selector.
        coords: The dimension's coordinate values, or `None` when it has none.
        dim_name: The dimension, for the refusals.

    Returns:
        list[int] | None: The selected positions, or `None` when this is not a mask.

    Raises:
        ValueError: The flags do not cover the axis, or the mask selects no step at all —
            a variable with no bands cannot be built.
    """
    positions = None
    values = list(selector) if isinstance(selector, (list, tuple)) else None
    if values and all(isinstance(one, (bool, np.bool_)) for one in values):
        length = len(coords) if coords is not None else 0
        if len(values) != length:
            raise ValueError(
                f"A boolean selector is a mask, and this one carries {len(values)} flags "
                f"for {dim_name!r}, which has {length}. Pass one flag per step, or select "
                f"by value."
            )
        positions = [index for index, flag in enumerate(values) if flag]
        if not positions:
            raise ValueError(
                f"A boolean mask that selects no step of {dim_name!r} would build a "
                f"variable with no bands, which GDAL has no raster for."
            )
    return positions


def _stamp_key(stamp: Any) -> Any:
    """A stamp as something that can be counted, NaN included.

    NaN equals nothing, itself included, so two NaN stamps landed in two dictionary
    entries of one each and a repeated NaN was never a duplicate — the case
    `keep=False` exists for. `NetCDF._dimension_holds` treats two NaNs at the same
    position as agreeing for the same reason.

    Args:
        stamp: One coordinate value.

    Returns:
        Any: The stamp, or a marker standing for "not a number".
    """
    key = stamp
    if isinstance(stamp, float) and math.isnan(stamp):
        key = ("nan",)
    return key


def _as_a_sequence_of_positions(selector: Any) -> Any:
    """`selector` with an integer array spelled as a list, everything else untouched.

    Positions are computed, not typed: `np.flatnonzero(mask)`, `np.where(...)[0]` and
    `np.argsort(...)` all hand back an integer `numpy.ndarray`. `isel` deliberately refuses
    one — its Notes say so, and a test pins it — but `drop_isel` is new, and refusing the
    canonical way of producing positions while pointing the caller at `sel` (which takes
    *values*) helps nobody.

    A boolean array is left alone, so the resolver's own "does not take booleans" refusal
    still answers it: a mask belongs to `sel`, where it selects by flag.

    Args:
        selector: What the caller passed for one dimension.

    Returns:
        Any: A `list` of `int` for an integer array, `selector` itself otherwise.
    """
    resolved = selector
    if isinstance(selector, np.ndarray) and selector.dtype.kind in "iu":
        resolved = [int(one) for one in selector.ravel().tolist()]
    return resolved


def _as_a_sequence_of_labels(selector: Any) -> Any:
    """`selector` with an array or a set spelled as a list, everything else untouched.

    Labels rarely come in typed by hand: they come from `other.time.values`, from
    `np.unique(...)`, from a set built to deduplicate. Those arrive as a `numpy.ndarray` or
    a `set`, which the resolvers read as one opaque label — an array then reached a truth
    test and raised `The truth value of an array with more than one element is ambiguous`,
    while a set matched nothing at all. A slice, a scalar, a string and a list are returned
    as they are.

    Args:
        selector: What the caller passed for one dimension.

    Returns:
        Any: A `list` for an array or a set, `selector` itself otherwise.
    """
    resolved = selector
    if isinstance(selector, np.ndarray):
        resolved = selector.ravel().tolist() if selector.ndim else selector.item()
    elif isinstance(selector, (set, frozenset)):
        resolved = list(selector)
    return resolved


def _resolve_one_dim(
    nc: NetCDF,
    dim_name: str,
    selector: Any,
    method: str | None,
    tolerance: float | None,
) -> list[int]:
    """Resolve one `sel` keyword to positions, without cutting anything.

    Everything `sel` used to do for a single keyword except the cut itself, so the whole
    call can be validated before the first band is read. Splitting it out is what lets
    `sel` refuse a wrong *value* in a later keyword as cheaply as a wrong name.

    Args:
        nc: The variable the keyword is resolved against.
        dim_name: The dimension to narrow.
        selector: A coordinate value, a list of them, a boolean mask of the axis' own
            length, or a slice.
        method: `None` for an exact match, `"nearest"` to snap.
        tolerance: The furthest a `"nearest"` snap may travel.

    Returns:
        list[int]: Positions along `dim_name` to keep.

    Raises:
        ValueError: The dimension is unknown, has no coordinates, or nothing matched.
        KeyError: A `"nearest"` request found nothing within `tolerance`.
    """
    _assert_band_dimension(nc, dim_name, caller="sel")
    selector = _as_a_sequence_of_labels(selector)

    coords = nc._band_dim_values_map.get(dim_name)
    masked = _mask_positions(selector, coords, dim_name)
    if masked is not None:
        return masked
    if coords is None:
        raise ValueError(
            f"No coordinate values available for dimension {dim_name!r}. "
            f"Select by position instead: isel({dim_name}=<index>)."
        )

    dim_indices, available = _resolve_selector_indices(
        nc, dim_name, coords, selector, method, tolerance
    )
    if not dim_indices:
        hint = _undecodable_label_hint(nc, dim_name, coords, selector)
        raise ValueError(
            f"No bands match {dim_name}={selector}. "
            f"Available values: {summarise_values(available)}{hint}"
        )

    return dim_indices


def _resolve_positional_indices(
    selector: Any, size: int, dim_name: str, *, allow_empty: bool = False
) -> list[int]:
    """Turn one `isel` selector into ascending, deduplicated positions along an axis.

    Accepts Python's own index types and nothing else: an `int`, a `list` or `tuple` of
    `int`, or a `slice` of indices. A negative index counts from the end, as everywhere
    else in Python, and a slice's `step` is honoured.

    Narrower than xarray's `isel` in two ways, and wider in one. xarray also takes a numpy
    integer, a numpy array and a boolean mask, all of which are refused here, and it keeps
    a list's order and its repeats where this sorts and deduplicates; a `tuple` goes the
    other way, accepted here and rejected by xarray.

    Args:
        selector: An `int`, a `list`/`tuple` of `int`, or a `slice`.
        size: The length of the axis, used to normalise negatives and to bound-check.
        dim_name: The dimension's name, for the error messages.
        allow_empty: `True` when selecting nothing is a legitimate answer — `drop_isel`,
            where dropping nothing keeps every step. `False` (default) refuses it, since
            `isel` of nothing would build a variable with no bands.

    Returns:
        list[int]: The positions to keep. A list or an int resolves to ascending,
            deduplicated order; a **slice keeps the order its step implies**, so a
            negative step yields descending positions and reverses the axis.

    Raises:
        IndexError: An index is outside `[-size, size)`. The message names the dimension
            and its length, because "index 7 is out of bounds" alone does not say which of
            several dimensions was overrun.
        TypeError: The selector is not something `operator.index()` admits, nor a
            `list`/`tuple` of such, nor a `slice`. A `bool` is refused rather than read as
            the `int` it subclasses, even though `index()` admits it, because
            `isel(dim=True)` would quietly mean position 1. Every integer type
            `index()` accepts — `np.int64`, `np.int32`, `np.uint8` — is taken, which
            matches `sel`'s numeric path accepting numpy scalars.
        ValueError: The selector keeps no position — an empty `list` or `tuple` as much as
            a `slice` whose bounds cross — which would otherwise build a zero-band variable
            that fails much later and further away, inside GDAL.

    Examples:
        - A negative index counts from the end:

          ```python
          >>> from pyramids.netcdf.engines.selection import _resolve_positional_indices
          >>> _resolve_positional_indices(-1, 4, "time")
          [3]

          ```
        - A slice keeps axis order, and a list is sorted and deduplicated:

          ```python
          >>> from pyramids.netcdf.engines.selection import _resolve_positional_indices
          >>> _resolve_positional_indices(slice(1, 3), 4, "time")
          [1, 2]
          >>> _resolve_positional_indices([2, 0, 2], 4, "time")
          [0, 2]

          ```
        - An out-of-range index names the dimension and its size:

          ```python
          >>> from pyramids.netcdf.engines.selection import _resolve_positional_indices
          >>> _resolve_positional_indices(9, 4, "time")
          Traceback (most recent call last):
              ...
          IndexError: index 9 is out of range for dimension 'time' of length 4...

          ```
        - A numpy integer is an index, so it needs no conversion:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.engines.selection import _resolve_positional_indices
          >>> _resolve_positional_indices(np.int64(1), 4, "time")
          [1]
          >>> _resolve_positional_indices([np.int64(2), np.int32(0)], 4, "time")
          [0, 2]

          ```
        - A boolean is not, even though Python would let it act as one:

          ```python
          >>> from pyramids.netcdf.engines.selection import _resolve_positional_indices
          >>> _resolve_positional_indices(True, 4, "time")
          Traceback (most recent call last):
              ...
          TypeError: isel() does not take booleans for 'time'...

          ```
    """
    if isinstance(selector, slice):
        sliced = list(range(*selector.indices(size)))
        return (
            sliced
            if sliced or allow_empty
            else _refuse_empty_selection(selector, dim_name, size)
        )
    wanted = list(selector) if isinstance(selector, (list, tuple)) else [selector]
    positions = {
        _normalise_index(_as_index(value, selector, dim_name), size, dim_name)
        for value in wanted
    }
    resolved = sorted(positions)
    return (
        resolved
        if resolved or allow_empty
        else _refuse_empty_selection(selector, dim_name, size)
    )


def _is_boolean(value: Any) -> bool:
    """Whether a selector entry is a boolean, in any spelling numpy offers.

    `operator.index()` admits a Python `bool`, so it has to be excluded by name or
    `isel(time=True)` quietly means position 1. A 0-d boolean array is caught here too:
    `index()` refuses it anyway, but with the generic "needs an int" message, which does
    not tell a caller reaching for a mask what is actually wrong.

    Args:
        value: One entry of an `isel` selector.

    Returns:
        bool: `True` for `bool`, `numpy.bool_`, and a 0-d boolean array.

    Examples:
        - Every spelling counts:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.engines.selection import _is_boolean
          >>> _is_boolean(True), _is_boolean(np.bool_(False)), _is_boolean(np.array(True))
          (True, True, True)

          ```
        - An integer does not, whatever its width:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf.engines.selection import _is_boolean
          >>> _is_boolean(1), _is_boolean(np.int64(0)), _is_boolean(np.array(1))
          (False, False, False)

          ```
    """
    if isinstance(value, (bool, np.bool_)):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == np.bool_


def _as_index(value: Any, selector: Any, dim_name: str) -> int:
    """One selector entry as a Python `int`, or the refusal explaining why it is not.

    Args:
        value: The entry to convert.
        selector: The whole selector, quoted back so a bad list entry names its list.
        dim_name: The dimension being selected, for the message.

    Returns:
        int: The entry as a plain `int`. `operator.index()` normalises the width, so a
        numpy integer arrives here as a Python one and the later arithmetic can neither
        overflow nor wrap.

    Raises:
        TypeError: The entry is a boolean, or is not something `operator.index()` admits.
    """
    if _is_boolean(value):
        raise TypeError(
            f"isel() does not take booleans for {dim_name!r}, got {selector!r}. A "
            f"boolean mask is not supported; pass the integer positions instead."
        )
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(
            f"isel() needs an int, a list of ints, a tuple of ints, or a slice for "
            f"{dim_name!r}, got {selector!r}. Select by coordinate value with "
            f"sel({dim_name}=...) instead."
        ) from None


def _normalise_index(value: int, size: int, dim_name: str) -> int:
    """An index counted from the end turned into one counted from the start.

    Args:
        value: An index, possibly negative.
        size: The axis' length.
        dim_name: The dimension being selected, for the message.

    Returns:
        int: The equivalent non-negative position.

    Raises:
        IndexError: `value` is outside `[-size, size)`. The message names the dimension and
            its length, because "index 7 is out of bounds" alone does not say which of
            several dimensions was overrun.
    """
    if not -size <= value < size:
        raise IndexError(
            f"index {value} is out of range for dimension {dim_name!r} of length "
            f"{size}. Valid indices are {-size} to {size - 1}."
        )
    return value + size if value < 0 else value


def _refuse_empty_selection(selector: Any, dim_name: str, size: int) -> NoReturn:
    """Refuse a selector that keeps no position, whatever form it arrived in.

    Reached from both arms of :func:`_resolve_positional_indices` — a `slice` whose bounds
    cross or coincide, and an empty list or tuple. Guarding only the slice left
    `isel(time=[])` building a variable that declares `_band_dim_sizes` with a zero on the
    selected axis and `band_count == 0`, whose first `read_array()` fails inside GDAL with
    an `AttributeError` about `GetScale` — a long way from the call that caused it. The
    label twin `sel(time=[])` has always refused.

    Args:
        selector: The selector that matched no position, quoted back to the caller.
        dim_name: The dimension it was applied to.
        size: That dimension's length.

    Raises:
        ValueError: Always.

    Examples:
        - The message names the selector, the dimension and its length:

          ```python
          >>> from pyramids.netcdf.engines.selection import _refuse_empty_selection
          >>> _refuse_empty_selection([], "time", 4)
          Traceback (most recent call last):
              ...
          ValueError: isel(time=[]) selects no index of an axis of length 4...

          ```
    """
    raise ValueError(
        f"isel({dim_name}={selector!r}) selects no index of an axis of length {size}. "
        f"A variable with no bands cannot be built."
    )


def _whole_axis(size: int) -> int:
    """The `head`/`tail` count that keeps an axis of `size` whole.

    Args:
        size: The axis length.

    Returns:
        int: `size`.
    """
    return size


def _single_step(size: int) -> int:
    """The `thin` step that keeps an axis whole, whatever its length.

    Args:
        size: The axis length, unused — every step is kept at a step of one.

    Returns:
        int: `1`.
    """
    return 1


_DEFAULT_WINDOW = 5
"""How many steps `head` and `tail` take when called with no arguments — xarray's."""


def _window_size(n: Any, dim_name: str, caller: str) -> int:
    """A window's step count, or the refusal saying why it is not one.

    Args:
        n: The count the caller passed.
        dim_name: The dimension, for the message.
        caller: The member, for the message.

    Returns:
        int: The count.

    Raises:
        TypeError: `n` is not an integer, or is a boolean.
        ValueError: `n` is below 1.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"{caller}() needs an integer for {dim_name!r}, got {n!r}.")
    if n < 1:
        raise ValueError(
            f"{caller}() needs a count of at least 1 for {dim_name!r}, got {n}: a raster "
            f"of no bands cannot be built."
        )
    return int(n)


def _head_positions(size: int, n: int) -> list[int]:
    """The first `n` positions of an axis of `size`.

    Args:
        size: The axis length.
        n: The count.

    Returns:
        list[int]: The positions.
    """
    return list(range(min(n, size)))


def _tail_positions(size: int, n: int) -> list[int]:
    """The last `n` positions of an axis of `size`.

    Args:
        size: The axis length.
        n: The count.

    Returns:
        list[int]: The positions.
    """
    return list(range(max(size - n, 0), size))


def _thin_positions(size: int, n: int) -> list[int]:
    """Every `n`-th position of an axis of `size`, from the first.

    Args:
        size: The axis length.
        n: The step.

    Returns:
        list[int]: The positions.
    """
    return list(range(0, size, n))


def _window_arguments(
    nc: NetCDF,
    indexers: Any,
    keywords: dict,
    caller: str,
    default: int | None,
    whole: Any,
) -> tuple[dict, int | None]:
    """Read the three spellings xarray accepts for a window into one plan.

    `head(time=2)` is this package's own; `head(2)` and `head({"time": 2})` are xarray's,
    and both used to raise `TypeError: head() takes 1 positional argument but 2 were
    given`. A bare count applies to every band dimension, which is what a bare call does.

    Args:
        nc: The receiver, whose band dimensions an empty mapping spans.
        indexers: The positional argument — a count, a mapping, or `None`.
        keywords: The `dimension=n` keywords.
        caller: The member, for the refusals.
        default: The count a bare call takes, or `None` when a bare call is refused.
        whole: `size -> n`, the count that keeps a dimension of that length whole. The
            axis length for `head` and `tail`, a step of one for `thin`.

    Returns:
        tuple: The per-dimension counts, and the count to apply when there are none.

    Raises:
        ValueError: Both spellings were used at once, as xarray refuses too.
        TypeError: The positional argument is neither a mapping nor an integer.
    """
    if indexers is not None and keywords:
        raise ValueError(
            f"{caller}() takes either a count or {caller}(dimension=n) keywords, not "
            f"both, as xarray refuses the same mixture."
        )
    wanted = dict(keywords)
    count = default
    if isinstance(indexers, Mapping):
        wanted = dict(indexers)
        if not wanted:
            # An empty mapping names no dimension to window, which xarray answers with the
            # whole axis — not with the five-step default a *bare* call takes.
            wanted = {
                name: whole(size)
                for name, size in zip(nc._band_dim_names, nc._band_dim_sizes)
            }
    elif indexers is not None:
        if isinstance(indexers, bool) or not isinstance(indexers, (int, np.integer)):
            raise TypeError(
                f"{caller}() takes an integer count or a mapping of them, got "
                f"{indexers!r}."
            )
        count = int(indexers)
    return wanted, count


def _windowed(
    nc: NetCDF, indexers: dict, caller: str, positions: Any, *, default: int | None
) -> NetCDF:
    """Cut `nc` to a positional window along each named band dimension.

    Args:
        nc: The variable.
        indexers: `dimension=n` pairs.
        caller: The member, for the refusals.
        positions: `(size, n) -> list[int]`, which positions the window keeps.
        default: The count applied along every band dimension when `indexers` is empty,
            or `None` to refuse an empty call — `thin`'s case, which xarray gives no default.

    Returns:
        NetCDF: The cut variable.

    Raises:
        ValueError: `indexers` is empty and there is no default, or the receiver has no
            band dimension for an empty call to window.
    """
    if not indexers:
        if default is None:
            raise ValueError(
                f"{caller}() needs a step: {caller}(2) for every band dimension, or "
                f"{caller}(time=2) for one."
            )
        indexers = dict.fromkeys(nc._band_dim_names, default)
        if not indexers:
            raise ValueError(
                f"{caller}() needs a band dimension to window, and this has none — a "
                f"single raster plane, or a container, carries no axis to cut. Call it "
                f"on a variable that has one."
            )
    keep: dict[str, list[int]] = {}
    for dim_name, n in indexers.items():
        _assert_band_dimension(nc, dim_name, caller=caller)
        size = nc._band_dim_sizes[nc._band_dim_names.index(dim_name)]
        keep[dim_name] = positions(size, _window_size(n, dim_name, caller))
    return _kept(nc, keep, caller)


def _kept(nc: NetCDF, keep: dict[str, list[int]], caller: str) -> NetCDF:
    """Cut `nc` to the given positions along each dimension, refusing an empty result.

    Every Tier 2 positional member ends here, so the one band-cutting primitive `sel` and
    `isel` already share does the reading.

    Args:
        nc: The variable.
        keep: The positions to keep along each dimension, in the order to keep them.
        caller: The member, for the refusal.

    Returns:
        NetCDF: The cut variable.

    Raises:
        ValueError: A dimension would keep no positions, or no dimension is named at all —
            with nothing to cut the receiver itself would be handed back, and that is the
            engine's `weakref.proxy` to its dataset, which dies with the object it proxies.
    """
    if not keep:
        raise ValueError(
            f"{caller}() names no dimension to cut, so there is nothing to build a "
            f"variable from."
        )
    for dim_name, positions in keep.items():
        if not positions:
            raise ValueError(
                f"{caller}() would leave {dim_name!r} with no steps, and a variable with "
                f"no bands cannot be built."
            )
    result = nc
    for dim_name, positions in keep.items():
        result = _subset_along_dim(result, dim_name, positions)
    return result


def _refuse_a_container(nc: NetCDF, caller: str) -> None:
    """Refuse a container by name: the band members cut one variable's bands.

    A container's raster is a placeholder — its variables hold the cells — so there is
    nothing to cut along. Without this the caller met whichever internal guard came first:
    `squeeze()` and `expand_dims()` died with `IndexError: list index out of range` from
    inside the band reader, which names nothing the caller wrote.

    Args:
        nc: The receiver.
        caller: The member named in the refusal.

    Raises:
        ValueError: The receiver is a container.
    """
    if not _reduces_as_a_variable(nc):
        variables = getattr(nc, "variable_names", None) or []
        where = (
            f" Call it on one of them: `nc.get_variable({variables[0]!r}).{caller}(...)`."
            if variables
            else " Call it on one of the container's variables."
        )
        raise ValueError(
            f"{caller}() works on one variable's bands, and a container has none of its "
            f"own — its variables do.{where}"
        )


def _spatial_dimension_names(nc: NetCDF) -> set[str]:
    """The names of `nc`'s spatial axes — its own, and the grid axes of its store.

    Two sources, each asked only what it can answer:

    - the variable's own dimensions minus its own band dimensions, which is the pair that
      makes up its grid — often a materialised spelling such as `subset_lat_4_-1_5`;
    - the parent container's dimensions that *name* a grid axis (`lat`, `lon`, `y`, `x`,
      `rlat`, `easting`, …), since a store's own spatial axes keep their plain names.

    The parent's **non**-spatial dimensions are deliberately not taken. Subtracting this
    variable's band dimensions from the parent's whole list called every other axis
    spatial, `time` included, so `squeeze()` on a store variable — whose own dimension
    list is empty, being freshly built in memory — could not be undone by `expand_dims`,
    the inverse the two docstrings point at.

    Args:
        nc: The variable.

    Returns:
        set[str]: The spatial names, empty when nothing declares any.
    """
    names = set(nc.dimension_names or []) - set(nc._band_dim_names)
    parent = getattr(nc, "_parent_nc", None)
    if parent is not None:
        names.update(
            name
            for name in (parent.dimension_names or [])
            if name.lower() in X_AXIS_NAMES or name.lower() in Y_AXIS_NAMES
        )
    return names


def _coordinates_of(nc: NetCDF, dim_name: str, caller: str) -> list:
    """A band dimension's coordinates, refusing a dimension that has none.

    Args:
        nc: The variable.
        dim_name: The dimension.
        caller: The member, for the refusals.

    Returns:
        list: The coordinate values.

    Raises:
        ValueError: The dimension is not a band dimension, or has no coordinates.
    """
    _assert_band_dimension(nc, dim_name, caller=caller)
    coords = nc._band_dim_values_map.get(dim_name)
    if coords is None:
        raise ValueError(
            f"{caller}() reads {dim_name!r}'s coordinate values, and it has none."
        )
    return list(coords)


def _rewrapped(nc: NetCDF) -> NetCDF:
    """`nc` under its own layout again, without reading a single band.

    The answer to a squeeze that drops nothing. It is not `nc` itself — inside an engine
    that is a `weakref.proxy` to the dataset, which dies with the object it proxies — but a
    fresh wrapper over the **same** raster, so no cells are copied and the result still
    reads as whatever `nc` reads as: a store variable keeps its lazy read, and a variable
    already rebuilt in memory stays rebuilt.

    Args:
        nc: The variable.

    Returns:
        NetCDF: A wrapper carrying `nc`'s layout and metadata.
    """
    # Local import breaks the netcdf.py <-> engines.selection cycle, as `subset` does.
    from pyramids.netcdf.netcdf import Variable

    shared = Variable(nc._raster, access=nc._access, open_as_multi_dimensional=False)
    result = nc._preserve_netcdf_metadata(shared)
    result._rebuilt_in_memory = nc._rebuilt_in_memory
    result._store_raster = getattr(nc, "_store_raster", None)
    return result


def _relabelled(nc: NetCDF, names: tuple, sizes: tuple, values_map: dict) -> NetCDF:
    """Every band of `nc`, rebuilt under a new band-dimension layout.

    `squeeze` and `expand_dims` change the layout without moving any data, and neither can
    go through the cut primitive — a flat raster has no band dimension to cut along — so
    this reads every band once and rebuilds the way that primitive does, then states the
    new layout.

    Args:
        nc: The variable.
        names: The new band dimension names.
        sizes: Their lengths, which must multiply to the band count.
        values_map: Their coordinates.

    Returns:
        NetCDF: The relabelled variable.
    """
    selected = _read_selected_bands(nc, list(range(nc.band_count)))
    rebuilt = Dataset.from_array(
        selected,
        no_data_value=scalar_no_data(nc.no_data_value),
        geo_ref=GeoReference(geo=nc.geotransform, epsg=crs_spec(nc.epsg, nc.crs)),
    )
    result = nc._preserve_netcdf_metadata(rebuilt)
    result._band_dim_names = tuple(names)
    result._band_dim_sizes = tuple(sizes)
    result._band_dim_values_map = dict(values_map)
    result._band_dim_time_attrs = {
        name: attrs
        for name, attrs in result._resolved_band_dim_time_attrs().items()
        if name in names
    }
    result._band_dim_name, result._band_dim_values = nc._derive_primary_band_view(
        result._band_dim_names,
        result._band_dim_values_map,
        result._band_dim_sizes,
        result._band_count,
    )
    return result


def _declared_band_sizes(nc: NetCDF, known: list[str]) -> dict[str, int]:
    """Each band dimension's declared length, as the receiver itself declares it.

    A `Variable` carries its own `(name, size)` pairs; a `Container` declares its dimensions in the
    store, including auxiliary-only ones no data variable spans (CF `bnds`), which is exactly the
    case a per-variable length check cannot see.

    Args:
        nc: The receiver.
        known: The band-dimension names to report, from `_band_dims_of`.

    Returns:
        dict[str, int]: The declared length of each known band dimension.
    """
    if _reduces_as_a_variable(nc):
        sizes = dict(zip(nc._band_dim_names, nc._band_dim_sizes))
    else:
        sizes = {
            name: size for name, size in nc.dimension_sizes.items() if name in known
        }
    return sizes


def _validated_restamp(
    nc: NetCDF, mapping: dict[str, Any], known: list[str]
) -> dict[str, list]:
    """The requested coordinate restamp, checked against the cube's declared dimensions.

    Validating up front — rather than inside the per-variable closure, which only sees the
    dimensions a data variable spans — is what makes a wrong length on an auxiliary-only dimension
    an error instead of a silent no-op (L1).

    Args:
        nc: The receiver.
        mapping: The requested `{dim: values}`.
        known: The receiver's band-dimension names.

    Returns:
        dict[str, list]: The accepted restamp, each entry a list of the dimension's length.

    Raises:
        ValueError: A name is not an existing band dimension, the values are not 1-D, or their
            length does not match the dimension's declared length.
    """
    declared = _declared_band_sizes(nc, known)
    coerced: dict[str, list] = {}
    for dim, values in mapping.items():
        if dim not in known:
            raise ValueError(
                f"assign_coords(): {dim!r} is not an existing band dimension (have "
                f"{sorted(known)}). pyramids has no index model, so a new or non-dimension "
                f"coordinate cannot be attached — only an existing band dimension can be "
                f"restamped."
            )
        if np.ndim(values) != 1:
            raise ValueError(
                f"assign_coords(): {dim!r} coordinates must be a 1-D sequence, got "
                f"{np.ndim(values)}-D."
            )
        size = declared.get(dim)
        if size is None:
            # `known` and `declared` are built from the same declaration, so a name in one is in
            # the other. Failing loudly beats the old `if size is not None` guard, which skipped
            # the length check instead of reporting that the two had drifted apart (L3).
            raise ValueError(
                f"assign_coords(): {dim!r} is a band dimension of this cube but its declared "
                f"length is unknown, so the restamp cannot be checked."
            )
        if len(values) != size:
            raise ValueError(
                f"assign_coords(): {dim!r} has length {size}, but {len(values)} "
                f"coordinate values were given."
            )
        coerced[dim] = list(values)
    return coerced


def _donor_variables(other: Any) -> list[tuple[str, NetCDF]]:
    """The `(name, variable)` pairs `update` will write, from either accepted donor form.

    Args:
        other: A `NetCDF` container, or a `{name: variable}` mapping.

    Returns:
        list[tuple[str, NetCDF]]: The donors, in the order they will be written.

    Raises:
        TypeError: `other` is neither a container nor a `{name: variable}` mapping.
    """
    if isinstance(other, Mapping):
        items = list(other.items())
    elif hasattr(other, "variable_names") and hasattr(other, "get_variable"):
        items = [
            (name, cast("NetCDF", other.get_variable(name)))
            for name in other.variable_names
        ]
    else:
        raise TypeError(
            "update() accepts a NetCDF container or a {name: variable} mapping, got "
            f"{type(other).__name__}."
        )
    return items


def _assert_band_axes_fit(
    name: str, variable: NetCDF, band_sizes: dict[str, int]
) -> None:
    """Refuse a donor spanning a same-named band axis of a different length.

    Left unchecked, `set_variable` resolves the conflict by writing the donor onto a renamed
    `<dim>_<len>` axis, so the donor silently lands on a different dimension.

    Args:
        name: The donor's variable name, for the message.
        variable: The donor variable.
        band_sizes: The lengths already fixed for each dimension name — the receiver's, plus
            those the donors checked before this one introduced.

    Raises:
        AlignmentError: A band dimension's length disagrees with the length already fixed.
    """
    for dim_name, dim_size in zip(variable._band_dim_names, variable._band_dim_sizes):
        existing = band_sizes.get(dim_name)
        if existing is not None and existing != dim_size:
            raise AlignmentError(
                f"update(): variable {name!r} spans band dimension {dim_name!r} of "
                f"length {dim_size}, but {dim_name!r} is already length {existing} here; "
                f"align the band axis first (interp / sel) — update does not reconcile band "
                f"dimensions, and writing it as-is would land it on a renamed "
                f"{dim_name}_{dim_size} axis."
            )


def _assert_donors_fit(nc: NetCDF, items: list[tuple[str, NetCDF]]) -> None:
    """Refuse the whole update unless every donor fits the receiver's grid and band axes.

    The pre-write pass that makes `update` all-or-nothing: checking as we wrote left a valid donor
    committed when a later one failed. The spatial grid is compared variable-to-variable, because a
    container's own raster is a placeholder — the reference is the receiver's first variable, or,
    for an empty receiver, the first donor. Band axes are checked against the receiver's *and*
    against each other, so two donors cannot disagree about one dimension's length either.

    Args:
        nc: The receiving container.
        items: The donors from `_donor_variables`.

    Raises:
        AlignmentError: A donor is on a different spatial grid, or disagrees on a band axis with
            the receiver or with an earlier donor.
    """
    reference: NetCDF | None = (
        cast("NetCDF", nc.get_variable(nc.variable_names[0]))
        if nc.variable_names
        else None
    )
    # Each accepted donor's axes join the map, so the donors are checked against one another and
    # not only against the receiver: a dimension a donor introduces has to mean the same length
    # for every later donor. Checking a map snapshotted before the loop let donor k+1 conflict
    # with donor k's new axis and still land on a renamed `<dim>_<len>` axis, and left the donors
    # of an empty receiver unchecked entirely (M2).
    band_sizes = dict(nc.dimension_sizes)
    for name, variable in items:
        if reference is None:
            reference = variable
        if not _same_spatial_grid(reference, variable):
            raise AlignmentError(
                f"update(): variable {name!r} is on a different grid than this container; "
                f"align it first (resample / to_crs / align) — update does not resample."
            )
        _assert_band_axes_fit(name, variable, band_sizes)
        band_sizes.update(zip(variable._band_dim_names, variable._band_dim_sizes))


def _unchanged(nc: NetCDF) -> NetCDF:
    """A cheap, non-mutating copy of the receiver for a no-op relabel.

    A single `Variable` is `_rewrapped` (shares the raster, keeps the lazy read); a `Container` is
    a `copy` (a single store copy, not the per-variable in-memory rebuild `_apply_per_variable`
    would do for a guaranteed no-op).

    Args:
        nc: The receiver.

    Returns:
        NetCDF: An equivalent, independent cube.
    """
    if _reduces_as_a_variable(nc):
        return _rewrapped(nc)
    return nc.copy()


def _relabel_per_variable(
    nc: NetCDF,
    relabel: Callable[[NetCDF], tuple[list[str], dict]],
    *,
    caller: str,
) -> NetCDF:
    """Relabel a variable's band dimensions without moving a single cell, or every variable's.

    The cell-free sibling of `_apply_per_variable`: `assign_coords` (and the single-variable
    path of `rename_dims`) change a band dimension's *label* — its name or its coordinate
    stamps — never its cells or the band count, so a single `Variable` is answered by
    `_rewrapped` (shares the raster, keeps a lazy read) with its band metadata overwritten,
    exactly as a no-op `squeeze` is. A `Container` keeps its band dimensions in the store rather
    than in memory, so it is rebuilt per variable through `_apply_per_variable` — the same path
    `transpose` takes — with each variable's cells carried over unchanged. (A container
    *rename* takes the store-level path instead, `NetCDF._rebuilt_container`, which renames the
    dimension in place; this per-variable rebuild is the `assign_coords` container path.)

    On the single-variable path the CF time attributes are re-keyed from the **resolved** view
    (`_resolved_band_dim_time_attrs`), as `_rewrapped` and `_copy_band_dim_metadata` already do.
    That normalises the result's raw dict for *both* callers, not just the renaming one: a
    non-CF-time `units` (a pressure level's `millibar`, say) is dropped and an entry resolved from
    a parent cube is added, so `assign_coords` — which renames nothing — can still come back with
    a different raw dict than it went in with. Every consumer reads these through
    `_carried_time_attrs`, which applies the same CF-time filter, so the normalisation is not
    observable in the cube's behaviour; it is called out here because nothing else records it.

    Args:
        nc: The receiver, a variable or a container.
        relabel: Given one variable, returns its `(new band names, new values map)`. The band
            count and axis order must be unchanged — this relabels, it does not restructure.
        caller: The member named in any refusal or warning.

    Returns:
        NetCDF: The relabelled variable or container; cells and band count unchanged.
    """
    if _reduces_as_a_variable(nc):
        names, values_map = relabel(nc)
        rename_map = dict(zip(nc._band_dim_names, names))
        result = _rewrapped(nc)
        result._band_dim_names = tuple(names)
        result._band_dim_values_map = dict(values_map)
        # Re-key from the resolved (CF-time-only) view, matching `_copy_band_dim_metadata` and
        # the container path: a non-time entry (a pressure level's `millibar`) is dropped rather
        # than carried under the new key, so the single-variable and container paths agree (N1).
        result._band_dim_time_attrs = {
            rename_map.get(name, name): attrs
            for name, attrs in nc._resolved_band_dim_time_attrs().items()
        }
        result._band_dim_name, result._band_dim_values = nc._derive_primary_band_view(
            result._band_dim_names,
            result._band_dim_values_map,
            nc._band_dim_sizes,
            result._band_count,
        )
        return result

    def _fn(var: NetCDF) -> tuple:
        names, values_map = relabel(var)
        arr = np.asarray(nc._materialize_variable_array(var, lazy=True))
        return arr, list(names), dict(values_map), _read_no_data(var), var.geotransform

    return _apply_per_variable(nc, _fn, caller=caller)


def _subset_along_dim(nc: NetCDF, dim_name: str, dim_indices: list[int]) -> NetCDF:
    """Build the variable holding only `dim_indices` along `dim_name`.

    Everything after "which positions do we want" — the band arithmetic, the read, and
    rebuilding the band-dim metadata on the result. `sel` reaches it by resolving a label
    to positions and `isel` by being handed them, so the two produce identical results for
    the same positions by construction rather than by agreement.

    A dimension with no coordinate values keeps `None` on the result rather than gaining a
    fabricated axis: that is the case `isel` exists to serve, and inventing coordinates for
    it would make the result claim to know something the store never said.

    Args:
        nc: The variable subset to cut.
        dim_name: A dimension of `nc`, already validated.
        dim_indices: Positions along that dimension, in the order the result should
            carry them. Ascending from a list or an int; descending from a
            negative-step slice, which reverses the axis.

    Returns:
        NetCDF: A variable with `len(dim_indices)` planes along `dim_name`.
    """
    dim_axis = nc._band_dim_names.index(dim_name)
    sizes = nc._band_dim_sizes
    band_indices = _map_dim_to_band_indices(dim_axis, sizes, dim_indices)
    coords = nc._band_dim_values_map.get(dim_name)
    selected_coords = None if coords is None else [coords[i] for i in dim_indices]
    selected = _read_selected_bands(nc, band_indices)

    ndv = nc.no_data_value
    # no_data_value is a TUPLE; the old `isinstance(ndv, list)` test never fired (ARC-29). Route
    # through the shared helper (handles list AND tuple) like the reduce path below.
    ndv_scalar = scalar_no_data(ndv)
    ds_result = Dataset.from_array(
        selected,
        no_data_value=ndv_scalar,
        geo_ref=GeoReference(geo=nc.geotransform, epsg=crs_spec(nc.epsg, nc.crs)),
    )
    result = nc._preserve_netcdf_metadata(ds_result)
    new_sizes = tuple(
        len(dim_indices) if i == dim_axis else s for i, s in enumerate(sizes)
    )
    result._band_dim_sizes = new_sizes
    result._band_dim_values_map = copy_band_values_map(nc._band_dim_values_map)
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


def _map_dim_to_band_indices(
    dim_axis: int, sizes: tuple[int, ...], dim_indices: list[int]
) -> list[int]:
    """Map pinned indices along one band dim to flat classic-band indices.

    GDAL flattens ``(d_0, …, d_{n-1}, lat, lon)`` row-major over the non-spatial
    dims (last varies fastest). For a band dim at axis ``k`` with ``sizes`` S,
    ``stride = prod(S[k+1:])`` and ``block = stride * S[k]``; each pinned index
    ``p`` emits ``[outer + p*stride .. outer + (p+1)*stride)`` for every
    ``outer`` in ``range(0, total, block)``. Reduces to the identity when there
    is a single band dim. Helper of :meth:`Selection.sel` and
    :meth:`Selection.isel`.

    **The emitted order is row-major over the narrowed sizes**, outer blocks before
    pinned indices, because the caller labels the result with those sizes and nothing
    else records how the flat list maps back onto dimensions.

    Args:
        dim_axis: Position of the pinned dimension in ``sizes``.
        sizes: The variable's band-dim sizes, outermost first, as tracked in
            ``_band_dim_sizes``.
        dim_indices: The positions kept along ``dim_axis``, in result order —
            ascending from a list, descending from a negative-step slice. The
            emitted bands follow that order, so it is **not** safe to assume the
            input is sorted. (`_plot._flat_band_index`, the only other caller,
            wraps the result in a `set`, so ordering is immaterial to it.)

    Returns:
        list[int]: ``len(dim_indices) * prod(sizes) // sizes[dim_axis]`` flat 0-based
            band indices, row-major over the sizes the result will declare.

    Examples:
        - Pinning the *inner* dim of a ``(time=4, level=3)`` cube interleaves the kept
          levels within each time step:

          ```python
          >>> from pyramids.netcdf.engines.selection import _map_dim_to_band_indices
          >>> _map_dim_to_band_indices(1, (4, 3), [1, 2])
          [1, 2, 4, 5, 7, 8, 10, 11]

          ```
        - Pinning the outermost dim takes contiguous blocks, and a single band dim is the
          identity:

          ```python
          >>> from pyramids.netcdf.engines.selection import _map_dim_to_band_indices
          >>> _map_dim_to_band_indices(0, (4, 3), [1, 2])
          [3, 4, 5, 6, 7, 8]
          >>> _map_dim_to_band_indices(0, (4,), [1, 2])
          [1, 2]

          ```
    """
    stride = math.prod(sizes[dim_axis + 1 :])
    block = stride * sizes[dim_axis]
    total = math.prod(sizes)
    band_indices: list[int] = []
    # `outer_start` outside `pinned`, not the reverse. The result declares
    # `_band_dim_sizes` with `dim_axis` narrowed, and that tuple is the only thing saying
    # how the flat band list maps back onto dimensions — so the bands have to come out
    # row-major over it, outer blocks first. Grouping by the pinned index instead returned
    # `(t0,l1)(t1,l1)(t2,l1)(t3,l1)(t0,l2)…` while declaring `(time=4, level=2)`, which
    # mislabels six of eight planes the moment anything reshapes by the declared sizes.
    # Identical output whenever one index is kept, or whenever there is a single outer
    # block — `prod(sizes[:dim_axis]) == 1`, which covers `dim_axis == 0` and also a
    # `dim_axis` whose preceding dims are all size 1. The two orders therefore agree on
    # most shapes, which is why the defect went unnoticed for so long. Where they differ
    # the old one contradicts the declared sizes by construction, since it groups by the
    # pinned index while the sizes say the outer axis varies slowest.
    for outer_start in range(0, total, block):
        for pinned in dim_indices:
            base = outer_start + pinned * stride
            band_indices.extend(range(base, base + stride))
    return band_indices


def _read_selected_bands(nc: NetCDF, band_indices: list[int]) -> np.typing.NDArray:
    """Read just the selected classic bands into one pre-allocated buffer.

    Mirrors the all-bands read path in ``IO.read_array`` rather than stacking N
    separate ``read_array`` results. Each 0-based band index maps to a 1-based
    GDAL band in the classic view built by ``get_variable``; reading only the
    selected bands avoids materialising the whole variable. Helper of
    :meth:`Selection.sel`.
    """
    if len(band_indices) == 1:
        return cast("np.typing.NDArray", nc._iloc(band_indices[0]).ReadAsArray())
    selected: np.typing.NDArray = np.empty(
        (len(band_indices), nc.rows, nc.columns), dtype=nc.numpy_dtype[0]
    )
    for out_i, band_index in enumerate(band_indices):
        selected[out_i, :, :] = nc._iloc(band_index).ReadAsArray()
    return selected
