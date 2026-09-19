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
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import geopandas as gpd
import numpy as np
from shapely import box, contains_xy

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
    _apply_to_container,
    _apply_to_variable,
    _assert_band_dimension,
    _CumSum,
    _Diff,
    _DropNa,
    _Extremum,
    _Push,
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

    def isel(self, **indexers: Any) -> NetCDF:
        """Select bands by **position** along one or more band dimensions.

        The positional twin of :meth:`sel`. Where `sel` asks "which band has this
        coordinate value", this asks "which band is at this index" — so it works on an axis
        the store gives no coordinates for, which is the case `sel` cannot serve at all. A
        WRF store's `bottom_top` is the usual example: 27 model levels with no coordinate
        variable, where `sel(bottom_top=...)` can only refuse.

        Several dimensions may be given in one call. They are applied in sequence, and
        because each cut is independent of the others the order does not affect the result.

        Args:
            **indexers: One or more `dimension=selector` pairs. Each selector is an index,
                a `list` or `tuple` of indices, or a `slice` of them. "Index" means
                anything `operator.index()` accepts, so a numpy integer counts and needs no
                `int(...)` wrapper. A negative index counts from the end, and a slice's
                `step` is honoured — unlike `sel`'s, where a range of coordinate values has
                no meaningful stride.

        Returns:
            NetCDF: A variable holding the selected bands, with `_band_dim_sizes` and the
            coordinate map narrowed to match. A dimension with no coordinates keeps none.

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
        for dim_name, selector in indexers.items():
            _assert_band_dimension(nc, dim_name, caller="isel")
            axis = nc._band_dim_names.index(dim_name)
            size = nc._band_dim_sizes[axis]
            resolved.append(
                (dim_name, _resolve_positional_indices(selector, size, dim_name))
            )

        result = nc
        for dim_name, dim_indices in resolved:
            result = _subset_along_dim(result, dim_name, dim_indices)
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
                        (5, 6)

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
        op = _Push(backward=False, limit=_check_limit(limit), caller="ffill")
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
        op = _Push(backward=True, limit=_check_limit(limit), caller="bfill")
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

        Args:
            dim: The non-spatial dimension to drop steps from.
            how: `"any"` (default) drops a step that holds any gap at all; `"all"` drops only
                a step with no valid cell. Ignored when `thresh` is given, as xarray ignores
                it.
            thresh: Keep a step holding at least this many valid cells, an integer of at
                least 1. `None` (default) defers to `how`.

        Returns:
            NetCDF: A container for a container, a variable for a variable, holding the steps
            that survived with their own values, type and no-data value — nothing is
            computed here, only selected.

        Raises:
            TypeError: `thresh` is not an integer, or is a boolean.
            ValueError: `how` is neither `"any"` nor `"all"`; `thresh` is below 1; no step
                survives, which would leave a variable with no bands; the container has no
                data variables; or `dim` is not a band dimension of any gridded variable.

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


def _check_limit(limit: Any, *, caller: str = "ffill") -> int | None:
    """A `limit` or `thresh` as a positive `int`, or the refusal saying why it is not one.

    Args:
        limit: As passed; `None` asks for no limit.
        caller: The member named in the message.

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
    overlapping chunks materialise) and the native ``(d0, …, rows, cols)`` shape
    is flattened to ``(bands, rows, cols)``; otherwise GDAL reads just the
    ``(c0, r0)``–``(c1, r1)`` block eagerly. Helper of
    :meth:`Selection._crop_curvilinear`.

    Reads with ``unpack=False``. The caller stamps the variable's **stored** no-data
    sentinel into the cells outside the cutline and rebuilds the variable around the
    result, carrying the packing with it — a copy of the store, so it moves counts.
    A physical read would put ``-98.49``-shaped values next to a ``-9999`` fill and
    declare the recipe over both.
    """
    if chunks is not None:
        lazy = nc.read_array(chunks=chunks, unpack=False)
        if lazy.ndim > 2:
            lazy = lazy.reshape(-1, *lazy.shape[-2:])
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
        selector: A coordinate value, a list of them, or a slice.
        method: `None` for an exact match, `"nearest"` to snap.
        tolerance: The furthest a `"nearest"` snap may travel.

    Returns:
        list[int]: Positions along `dim_name` to keep.

    Raises:
        ValueError: The dimension is unknown, has no coordinates, or nothing matched.
        KeyError: A `"nearest"` request found nothing within `tolerance`.
    """
    _assert_band_dimension(nc, dim_name, caller="sel")

    coords = nc._band_dim_values_map.get(dim_name)
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


def _resolve_positional_indices(selector: Any, size: int, dim_name: str) -> list[int]:
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
        return list(range(*selector.indices(size))) or _refuse_empty_selection(
            selector, dim_name, size
        )
    wanted = list(selector) if isinstance(selector, (list, tuple)) else [selector]
    positions = {
        _normalise_index(_as_index(value, selector, dim_name), size, dim_name)
        for value in wanted
    }
    return sorted(positions) or _refuse_empty_selection(selector, dim_name, size)


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
