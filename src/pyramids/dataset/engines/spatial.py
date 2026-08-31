"""Spatial engine.

Owns the Spatial family of operations on a Dataset. Accessed as
``ds.spatial``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.spatial.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload
from xml.sax.saxutils import escape  # nosec B406 - output escaping only

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr
from pyproj import Transformer

from pyramids.base._domain import is_no_data
from pyramids.base._utils import DEFAULT_RESAMPLING, resolve_resampling
from pyramids.base.crs import (
    crs_equal,
    crs_from_user_input,
    crs_spec,
    epsg_of_crs,
    reproject_coordinates,
    require_crs_spec,
    sr_from_epsg,
    sr_from_user_input,
    sr_from_wkt,
)
from pyramids.dataset.abstract_dataset import RasterBase
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr
from pyramids.feature.bbox import split_antimeridian

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

from pyramids.base.georeference import GeoReference
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._warp import carry_raster_metadata, warp_to_dataset
from pyramids.dataset.engines._warp import dst_srs_arg as _dst_srs_arg
from pyramids.dataset.engines.vectorize import Vectorize


@overload
def _resolve_resolution(
    cell_size: float | tuple[float, float] | list[float],
) -> tuple[float, float]: ...


@overload
def _resolve_resolution(cell_size: None) -> tuple[None, None]: ...


def _resolve_resolution(
    cell_size: float | tuple[float, float] | list[float] | None,
) -> tuple[float | None, float | None]:
    """Resolve a ``cell_size`` argument to a positive ``(x_res, y_res)`` pair.

    Accepts a scalar (square cells), an ``(x_res, y_res)`` pair (non-square cells), or ``None``
    (returned as ``(None, None)`` so callers can let GDAL infer the output resolution).

    Args:
        cell_size: Output pixel size — a scalar, an ``(x_res, y_res)`` sequence, or ``None``.

    Returns:
        tuple: ``(x_res, y_res)``; both ``None`` when ``cell_size`` is ``None``.

    Raises:
        ValueError: If the pair is not length 2, or any resolution is not positive.
    """
    result: tuple[float | None, float | None]
    if cell_size is None:
        result = (None, None)
    else:
        if isinstance(cell_size, (tuple, list)):
            if len(cell_size) != 2:
                raise ValueError(
                    f"cell_size must be a scalar or an (x_res, y_res) pair, got {cell_size!r}."
                )
            x_res, y_res = float(cell_size[0]), float(cell_size[1])
        else:
            x_res = y_res = float(cell_size)
        if x_res <= 0 or y_res <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}.")
        result = (x_res, y_res)
    return result


def _check_lon_halves_concatenable(
    west_part: RasterBase, east_part: RasterBase
) -> None:
    """Assert the invariant that two longitude-adjacent crop halves are stitchable.

    Both halves are cropped from the same source lattice, so equal row/band counts
    and a shared cell boundary at the 180/360 seam are expected to hold — this is a
    defensive guard that turns any future violation into a clear error instead of a
    raw NumPy shape error or a silently shifted `np.concatenate` result.

    Args:
        west_part: Crop of the pre-seam half.
        east_part: Crop of the post-seam half (wrapped past the seam).

    Raises:
        ValueError: The halves have mismatched row/band counts, or the grid has no
            cell boundary at the seam so the halves are not seam-aligned.
    """
    if west_part.rows != east_part.rows or west_part.band_count != east_part.band_count:
        raise ValueError(
            "antimeridian halves are not concatenable "
            f"(rows {west_part.rows}/{east_part.rows}, "
            f"bands {west_part.band_count}/{east_part.band_count})"
        )
    w_gt = west_part.geotransform
    seam_gap = abs(
        (w_gt[0] + west_part.columns * w_gt[1]) - (east_part.geotransform[0] + 360.0)
    )
    if seam_gap > 0.5 * abs(w_gt[1]):
        raise ValueError(
            "antimeridian halves are not seam-aligned; the grid has no cell "
            "boundary at the 180/360 seam, so the halves cannot be stitched"
        )


def _split_lon_bbox(
    bbox: tuple[float, float, float, float], lon_max: float, cell_x: float
) -> list[tuple[float, float, float, float]]:
    """Split a geographic ``west > east`` bbox into ``west < east`` halves.

    Pure numeric core shared by the rectilinear and curvilinear crop paths. A
    0..360 grid (``lon_max`` reaching past 180 by more than a cell) has the bbox
    shifted into its own frame, which usually removes the wrap (one half); a
    -180..180 grid is split at 180 via :func:`split_antimeridian`.

    The frame is inferred from ``lon_max`` alone: a 0..360 grid must actually
    reach past 180 to be detected as such. An eastern-hemisphere-only grid that
    ends at or below 180 is treated as -180..180 — harmless, since an
    antimeridian bbox barely overlaps it and the non-overlapping half is skipped.

    Args:
        bbox: ``(west, south, east, north)`` with ``west > east``.
        lon_max: The grid's maximum longitude (its own frame's east edge).
        cell_x: The grid's longitude cell size, used as a one-cell tolerance so a
            -180..180 grid whose ``lon_max`` floats a hair over 180 is not
            mistaken for a 0..360 grid.

    Returns:
        One or two ``west < east`` sub-bboxes to crop and stitch, in west-to-east
        order.
    """
    west, south, east, north = bbox
    if lon_max > 180.0 + cell_x:
        # 0..360 grid: bring the STAC (-180..180) bbox into the grid's frame.
        west = west + 360.0 if west < 0 else west
        east = east + 360.0 if east < 0 else east
        if west <= east:
            halves = [(west, south, east, north)]
        else:
            halves = [(west, south, 360.0, north), (0.0, south, east, north)]
    else:
        halves = split_antimeridian(bbox)
    return halves


def _antimeridian_halves(
    ds: RasterBase, bbox: tuple[float, float, float, float]
) -> list[tuple[float, float, float, float]]:
    """Split a ``west > east`` bbox using a rectilinear dataset's affine frame.

    Reads the grid's east edge and cell size from the affine ``bbox`` /
    ``geotransform`` and delegates to :func:`_split_lon_bbox`. Curvilinear grids
    have no single affine frame and key the split off their 2-D longitude array
    instead.

    Args:
        ds: The dataset whose longitude frame and cell size set the seam.
        bbox: ``(west, south, east, north)`` with ``west > east``.

    Returns:
        One or two ``west < east`` sub-bboxes, in west-to-east order.
    """
    return _split_lon_bbox(bbox, float(ds.bbox[2]), abs(ds.geotransform[1]))


def _reaches_antimeridian_seam(ds: RasterBase) -> bool:
    """Whether the grid can serve a *wrapping* antimeridian crop.

    A two-half (wrapping) crop needs the grid to actually reach the seam it wraps
    across: a 0..360 grid must span (nearly) the full 0..360 (so the wrap at the
    0/360 edge has data on both sides), and a -180..180 grid must touch +180 or
    -180 (within a cell). A regional grid that reaches neither (e.g. Europe, lon
    -10..40) — or a partial 0..360 grid that stops well short of 360 (e.g. lon
    0..256) — cannot serve a wrap, so a ``west > east`` bbox producing one there is
    a transposed/typo bbox rather than a genuine crossing.

    Args:
        ds: The dataset whose affine ``bbox`` / ``geotransform`` set the extent.

    Returns:
        True when the extent reaches the seam; False otherwise.
    """
    lon_min, lon_max = float(ds.bbox[0]), float(ds.bbox[2])
    cell_x = abs(ds.geotransform[1])
    if lon_max > 180.0 + cell_x:
        # 0..360 frame: the wrap is at the 0/360 edge, so require a near-full span.
        return lon_max >= 360.0 - cell_x and lon_min <= cell_x
    # -180..180 frame: the extent must touch +180 or -180.
    return lon_max >= 180.0 - cell_x or lon_min <= -180.0 + cell_x


def _require_antimeridian_seam(
    ds: RasterBase, bbox: tuple[float, float, float, float]
) -> None:
    """Raise when a *wrapping* antimeridian bbox targets a grid that can't serve it.

    Guards the ``west > east`` reinterpretation. A contiguous (single-half) split —
    e.g. ``170..190`` on a 0..360 grid — is just a normal crop and needs no check;
    only a two-half **wrapping** split requires the grid to actually reach the seam.
    Raising here turns a transposed/typo bbox into a clear error instead of a
    truncated single-half crop or a downstream empty-crop failure.

    Args:
        ds: The dataset being cropped.
        bbox: ``(west, south, east, north)`` with ``west > east``.

    Raises:
        ValueError: A wrapping split targets a grid that does not reach the seam.
    """
    wrapping = len(_antimeridian_halves(ds, bbox)) >= 2
    if wrapping and not _reaches_antimeridian_seam(ds):
        raise ValueError(
            "bbox has west > east (antimeridian) but the dataset's longitude "
            "extent does not reach the 180 seam - the bbox may be transposed, or "
            "the dataset does not cover the antimeridian region."
        )


def _crop_seam_halves(
    ds: RasterBase,
    bbox: tuple[float, float, float, float],
    crop_half: Any,
    merge_halves: Any,
) -> Any:
    """Crop the overlapping antimeridian halves of ``bbox`` and stitch them.

    Shared by the raster (`Spatial`) and NetCDF (`Selection`) crop engines: each
    half that overlaps the dataset's longitude extent is cropped through the normal
    path via ``crop_half``; a single overlapping half is handed straight back, two
    are stitched with ``merge_halves``. Non-adopted parts are always closed.

    Args:
        ds: The dataset whose ``bbox`` sets the longitude extent to test overlap
            against.
        bbox: The original ``(west, south, east, north)`` with ``west > east``.
        crop_half: Callable cropping one ``west < east`` sub-bbox to a part.
        merge_halves: Callable concatenating two parts into the final result.

    Returns:
        The single overlapping half or the stitched strip spanning the seam.

    Raises:
        ValueError: The bbox does not overlap the dataset's longitude extent.
    """
    xmin, xmax = float(ds.bbox[0]), float(ds.bbox[2])
    halves = _antimeridian_halves(ds, bbox)
    parts: list = []
    try:
        for half in halves:
            if half[0] < xmax and half[2] > xmin:
                parts.append(crop_half(half))
        if not parts:
            raise ValueError(
                f"antimeridian bbox {bbox!r} does not overlap the dataset extent"
            )
        if len(parts) == 1:
            result, parts = parts[0], []  # hand ownership to the caller
        else:
            result = merge_halves(parts[0], parts[1])
    finally:
        for part in parts:
            part.close()
    return result


def _stitch_lon_halves(ds: RasterBase, west_part: Any, east_part: Any) -> Dataset:
    """Concatenate two longitude-adjacent crops into one contiguous raster Dataset.

    `west_part` (pre-seam) sits to the left of `east_part` (wrapped past the seam);
    the merged raster keeps `west_part`'s north-up geotransform, so the longitude
    mapping continues past the seam (e.g. 170..180 then 180..190). Shared by both
    crop engines; the NetCDF engine re-wraps the result to preserve variable
    metadata.

    Args:
        ds: The dataset supplying the band names for the merged raster.
        west_part: Crop of the pre-seam half.
        east_part: Crop of the post-seam half.

    Returns:
        Dataset: The concatenated raster.
    """
    # Local import breaks the engines <-> Dataset cycle; the merged result must be a
    # plain raster Dataset (from_array on a variable view would build a NetCDF
    # container).
    from pyramids.dataset.dataset import Dataset

    _check_lon_halves_concatenable(west_part, east_part)
    merged = np.concatenate([west_part.read_array(), east_part.read_array()], axis=-1)
    # epsg is None only for a no-EPSG CRS reported as such (a NetCDF
    # geostationary grid); from_array raises CRSError on None, so fall back to
    # the WKT. No-op for a plain Dataset (reports 4326) (#706).
    out = Dataset.from_array(
        merged,
        geo_ref=GeoReference(
            geo=west_part.geotransform,
            epsg=crs_spec(west_part.epsg, west_part.crs),
        ),
        no_data_value=west_part.no_data_value,
    )
    out.band_names = ds.band_names
    return out


class Spatial(_Engine["Dataset"]):
    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        return str(self._ds.raster.GetProjection())

    def set_crs(self, crs: str | None = None, epsg: int | None = None) -> None:
        """Set the Coordinate Reference System (CRS).

        Assign the CRS of the raster in place, from either a WKT string (``crs``) or an EPSG
        code (``epsg``). Exactly one of the two must be supplied.

        Args:
            crs (str | None):
                Optional if epsg is specified. WKT string. i.e.
                    ```
                    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84", 6378137,298.257223563,AUTHORITY["EPSG","7030"],
                    AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
                    0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],
                    AUTHORITY["EPSG","4326"]]'
                    ```
            epsg (int | None):
                Optional if crs is specified. EPSG code specifying the projection.

        Returns:
            None: The CRS is set on the underlying dataset in place.

        Raises:
            ReadOnlyError: If the dataset is opened read-only.
            TypeError: If the dataset is backed by an ASCII driver, which cannot store a CRS.
            ValueError: If neither ``crs`` nor ``epsg`` is provided.
        """
        # ASCII cannot store a CRS in any mode, so that TypeError takes precedence
        # over the read-only guard below.
        if self._ds.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then "
                "reset the crs"
            )
        # Validate the arguments before the read-only guard so an invalid call
        # (neither crs nor epsg) reports the actionable ValueError regardless of
        # access mode, rather than a ReadOnlyError that hides the real mistake.
        if crs is None and epsg is None:
            raise ValueError("Either crs or epsg must be provided.")
        self._ds._require_writable("set the CRS")
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if crs is not None:
            self._ds.raster.SetProjection(crs)
            # An empty crs string means "no CRS", which propagates as None
            # rather than being tagged WGS 84 (ARC-26).
            self._ds._epsg = epsg_of_crs(crs)
        else:
            # crs is None here, so epsg is not None (the neither-None check above
            # rejects both being None); cast narrows it for the type checker without
            # a redundant always-true runtime condition.
            sr = sr_from_epsg(cast(int, epsg))
            self._ds.raster.SetProjection(sr.ExportToWkt())
            self._ds._epsg = epsg
        # A NetCDF container derives its CRS from its variables and memoises the
        # result, so both caches are stale now. Without this, clearing a
        # container's CRS is a no-op: `_get_crs` falls straight back to the
        # cached borrowed value (ARC-26).
        # The inferred-CF cache exists on every Dataset; the container ones only
        # on NetCDF. All three derive from the projection just replaced.
        self._ds.__dict__.pop("_cf_crs_cache", None)
        if hasattr(self._ds, "_container_crs_cache"):
            self._ds._container_crs_cache = None
            self._ds._crs_cache = None  # type: ignore[attr-defined]
            # Leave it False: the property re-resolves on the next read. Setting
            # it True here (as an earlier revision did, two lines below) pinned
            # the pre-set answer and made clearing a CRS a no-op.
            self._ds._epsg_resolved = False  # type: ignore[attr-defined]

    def to_crs(
        self,
        to_epsg: int | str | Any,
        method: str = DEFAULT_RESAMPLING,
        maintain_alignment: bool = False,
        *,
        cell_size: float | tuple[float, float] | None = None,
    ) -> Dataset:
        """Reproject the dataset to any projection.

            (default the WGS84 web mercator projection, without resampling)

        Args:
            to_epsg (int | str | pyproj.CRS):
                The target CRS. Accepts any form :meth:`pyproj.CRS.from_user_input`
                understands: an EPSG reference number (``3857``), an authority string
                (``"EPSG:3857"``, ``"ESRI:54030"`` for Robinson, ``"ESRI:54009"`` for
                Mollweide), a bare numeric string (``"3857"``), a WKT or PROJ4 string
                (``"+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84"``), or a
                :class:`pyproj.CRS`. Projections without an EPSG code (orthographic,
                Robinson, Mollweide, polar-stereographic variants) are warped directly
                against the spatial reference; cells outside the projection domain
                are filled with the source's nodata value when one is configured, or
                with GDAL's dtype-default fill value otherwise.
            method (str):
                Resampling method, case-insensitive. Default is "nearest neighbor". Allowed values: "nearest"
                (alias "nearest neighbor"), "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                "mode", "max", "min", "med", "q1", "q3", "sum", and "rms" (the GDAL warp algorithms;
                "sum"/"rms" need GDAL >= 3.1/3.3). See https://gisgeography.com/raster-resampling/.
                Note: the aggregating algorithms ("min", "max", "average", "mode", "med", "q1", "q3", "sum",
                "rms") honour the source no-data value when the raster declares one — no-data cells are excluded
                from the kernel, and a kernel that is entirely no-data yields no-data. A raster with no no-data
                marker has nothing to exclude, so every cell (including a sentinel fill such as 9999) counts as
                valid data.
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the raster the same after reprojection.
                Default is False.
            cell_size (float | tuple, keyword-only):
                Optional output pixel size in target-CRS units. A scalar gives square cells; an
                ``(x_res, y_res)`` pair gives non-square cells. ``None`` (default) lets GDAL pick the
                output resolution. Not supported together with ``maintain_alignment=True``.

        Returns:
            Dataset:
                A new reprojected Dataset.

        Raises:
            CRSError:
                ``to_epsg`` cannot be interpreted as a CRS.
            TypeError:
                ``method`` is not a string.
            ValueError:
                ``method`` is not one of the supported interpolation methods.

        Examples:
            - Reproject a small 4326 raster to Web Mercator (EPSG:3857). The
              source cell size of 0.05° expands to roughly 5566 m near the
              equator and the EPSG of the result confirms the warp:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 5, 5)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
              ... )
              >>> dataset.epsg
              4326
              >>> reprojected = dataset.to_crs(to_epsg=3857)
              >>> reprojected.epsg
              3857
              >>> reprojected.band_count
              4

              ```
            - Reproject to a non-EPSG CRS via an ESRI authority string
              (Robinson, ``ESRI:54030``):

              ```python
              >>> import numpy as np
              >>> from osgeo import osr
              >>> from pyramids.dataset import Dataset
              >>> arr = np.ones((5, 5), dtype=np.float32)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
              ... )
              >>> robinson = dataset.to_crs(to_epsg="ESRI:54030")
              >>> "Robinson" in osr.SpatialReference(wkt=robinson.crs).GetName()
              True

              ```
            - Reproject to a bespoke orthographic projection via a proj4 string
              (no authority code at all):

              ```python
              >>> import numpy as np
              >>> from osgeo import osr
              >>> from pyramids.dataset import Dataset
              >>> arr = np.ones((5, 5), dtype=np.float32)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
              ... )
              >>> proj4 = "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"
              >>> ortho = dataset.to_crs(to_epsg=proj4)
              >>> osr.SpatialReference(wkt=ortho.crs).IsProjected()
              1
              >>> ortho.epsg is None  # a bespoke projection has no EPSG code
              True

              ```
            - Contrast ``maintain_alignment=False`` (default) with
              ``maintain_alignment=True``. At 60°N a 4326 → 3857 warp distorts
              cell sizes substantially, so the default `gdal.Warp` heuristic
              picks a different output shape from the source; the alignment-
              preserving path keeps the source row/column count and absorbs the
              distortion into the per-axis cell size instead:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.ones((10, 10), dtype=np.float32)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(10.0, 60.5), cell_size=0.1, epsg=4326),
              ... )
              >>> default_warp = dataset.to_crs(to_epsg=3857)
              >>> (default_warp.rows, default_warp.columns)
              (13, 6)
              >>> aligned = dataset.to_crs(to_epsg=3857, maintain_alignment=True)
              >>> (aligned.rows, aligned.columns)
              (10, 10)

              ```

        See Also:
            - :meth:`Spatial.set_crs`: Tag the dataset with a new CRS *without*
              warping the pixels (use when the source CRS metadata is wrong,
              not when you want a reprojection).
            - :meth:`Spatial.resample`: Change the cell size without changing
              the CRS.
            - :func:`pyramids.base.crs.sr_from_user_input`: The helper that
              resolves every accepted CRS form to an
              :class:`osr.SpatialReference`.

        """
        # Reprojection is meaningless without a source CRS: GDAL would warp from
        # an unknown frame and the result would carry `to_epsg` as an unearned
        # claim, with the geotransform untouched. This is the operation
        # `require_crs_spec` exists for, so guard it here rather than only at the
        # callers (ARC-26).
        require_crs_spec(self._ds.epsg, self._ds.crs, "reproject to another CRS")
        dst_sr = sr_from_user_input(to_epsg)
        resampling_method: int = resolve_resampling(method)

        if maintain_alignment:
            # Reject cell_size before validating it, so the more specific "not supported with
            # maintain_alignment" error wins over the generic shape/positivity check.
            if cell_size is not None:
                raise ValueError(
                    "cell_size is not supported with maintain_alignment=True (that path keeps the "
                    "source row/column count). Use maintain_alignment=False to set the output cell size."
                )
            dst_obj = self._reproject_with_ReprojectImage(dst_sr, resampling_method)
        else:
            # cell_size may be a scalar (square) or an (x_res, y_res) pair (non-square output).
            x_res, y_res = _resolve_resolution(cell_size)
            dst_obj = warp_to_dataset(
                self._ds,
                gdal.WarpOptions(
                    dstSRS=_dst_srs_arg(dst_sr),
                    format="VRT",
                    resampleAlg=resampling_method,
                    xRes=x_res,
                    yRes=y_res,
                ),
                error_message="GDAL could not reproject the dataset.",
            )

        return dst_obj

    def warped_view(
        self,
        crs: int | str | Any,
        method: str = DEFAULT_RESAMPLING,
        *,
        cell_size: float | tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Dataset:
        """Return a lazy, reprojected **view** of the dataset (no pixels warped yet).

        Builds an in-memory warped VRT: nothing is resampled until a window is
        read, and a windowed read warps **only that window**. This is the lazy
        counterpart of :meth:`to_crs` — prefer it for tile serving, partial
        reads of reprojected data, and chained virtual pipelines; prefer
        :meth:`to_crs` when you will consume the whole reprojected raster.

        The returned Dataset keeps a reference to its source, so the source
        handle cannot be garbage-collected underneath the view.

        Note:
            The view captures its source **by handle, not by value**: the VRT
            re-reads the source's geotransform, projection, and pixels lazily on
            each windowed read. Mutating the source in place after the view is
            built (for example :meth:`set_crs` or anything that rewrites the
            geotransform) leaves the view reading from the now-changed source and
            is undefined. Treat the source as read-only for the lifetime of the
            view, or rebuild the view after mutating the source.

        Args:
            crs: Target CRS in any form :meth:`pyproj.CRS.from_user_input`
                accepts (EPSG int, ``"EPSG:3857"``, WKT, PROJ4, pyproj CRS).
            method: Resampling method used when windows are read. Any name
                accepted by :func:`pyramids.base._utils.resolve_resampling`
                (case- and whitespace-insensitive). Default is
                ``"nearest neighbor"``.
            cell_size: Optional output pixel size in target-CRS units. A scalar
                applies to both axes (square cells); an ``(x_res, y_res)`` pair
                gives non-square cells. ``None`` lets GDAL pick the size that
                preserves the source resolution.
            bbox: Optional ``(min_x, min_y, max_x, max_y)`` output extent in
                the **target** CRS; ``None`` covers the warped source extent.

        Returns:
            Dataset: A read-only, VRT-backed reprojected view.

        Raises:
            CRSError: ``crs`` cannot be interpreted as a CRS.
            TypeError: ``method`` is not a string.
            ValueError: ``method`` is not a supported resampling method.
            RuntimeError: GDAL could not build the warped VRT.

        Examples:
            - A view reports the warped CRS without materialising pixels, and
              a windowed read matches the eager reprojection:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> src = Dataset.from_array(
                ...     np.random.rand(8, 8).astype("float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 8), cell_size=0.01, epsg=4326),
                ... )
                >>> view = src.warped_view(3857)
                >>> view.epsg
                3857
                >>> eager = src.to_crs(3857)
                >>> bool(np.allclose(view.read_array(), eager.read_array()))
                True

                ```
            - The view holds its source alive (safe to drop the original):
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> src = Dataset.from_array(
                ...     np.ones((4, 4), dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=0.01, epsg=4326),
                ... )
                >>> view = src.warped_view(3857)
                >>> del src
                >>> view.read_array().shape == (view.rows, view.columns)
                True

                ```

        See Also:
            Spatial.to_crs: The eager reprojection (materialises the result).
        """
        dst_sr = sr_from_user_input(crs)
        resample_alg: int = resolve_resampling(method)
        dst_srs_arg = _dst_srs_arg(dst_sr)
        x_res, y_res = _resolve_resolution(cell_size)
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError(
                    f"bbox must be (min_x, min_y, max_x, max_y), got {bbox!r}."
                )
            min_x, min_y, max_x, max_y = bbox
            if min_x >= max_x or min_y >= max_y:
                raise ValueError(
                    f"bbox must have min_x < max_x and min_y < max_y, got {bbox!r}."
                )
        options = gdal.WarpOptions(
            format="VRT",
            dstSRS=dst_srs_arg,
            resampleAlg=resample_alg,
            xRes=x_res,
            yRes=y_res,
            outputBounds=bbox,
            multithread=True,
        )
        # The VRT references the source GDAL handle, so the result has to pin
        # it; `warp_to_dataset` is the one place that does.
        return warp_to_dataset(
            self._ds,
            options,
            access="read_only",
            error_message=f"GDAL could not build a warped VRT onto {dst_srs_arg!r}.",
        )

    def _get_epsg(self) -> int | None:
        """Get the EPSG number.

            This function reads the projection of a GEOGCS file or tiff file.

        Returns:
            int: EPSG number.
        """
        prj = self._get_crs()
        # No projection means no EPSG; None propagates instead of a
        # fabricated WGS 84 code (ARC-26).
        epsg = epsg_of_crs(prj)

        return epsg

    def wrap_longitude(self) -> Dataset:
        """Wrap a global raster's longitude from the 0/360 frame to the -180/180 frame.

        The wrap is a pure column roll (no resampling): the columns whose longitude is greater than
        180 (the western hemisphere in the -180/180 frame) move to the front, the remaining columns
        follow, and the geotransform's top-left x is moved to -180. The raster must span the whole
        globe (its last longitude must exceed 180).

        Two execution paths, selected automatically by the source:

        - **File-backed source** (a real on-disk raster): the roll is built as a lazy two-source VRT,
          so no pixel data is read until the result is used (read, plotted, cropped, or written).
        - **In-memory source** (e.g. a NetCDF variable view from ``get_variable``, which has no
          filename for a VRT to reference): an eager fallback copies the dataset once via
          ``MEM.CreateCopy`` (preserving all metadata) and rolls the columns in place, so the source
          is read only once.

        Returns:
            Dataset:
                A new dataset of the same class on the -180/180 grid. Same shape, dtype, band count,
                no-data value, and CRS as the source; only the columns and the top-left x change.
                File-backed inputs yield a VRT-backed (lazy) dataset; in-memory inputs an MEM-backed
                one.

        Raises:
            ValueError: If the grid is not a global 0-360 grid — it must span ~360° of longitude
                (within one cell) and lie in the 0-360 frame (its last longitude exceeds 180).
                Regional windows and grids already in the -180/180 frame are rejected.

        Examples:
            - Shift an in-memory 0-360 global raster and inspect the new extent:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.arange(360, dtype=np.float32).reshape(1, 360)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.5), cell_size=1.0, epsg=4326),
                ... )
                >>> shifted = ds.wrap_longitude()
                >>> shifted.top_left_corner[0]
                -180.0
                >>> bool(shifted.lon.max() < 180)
                True
                >>> shifted.read_array(band=0).shape
                (1, 360)

                ```
            - A raster that does not span the globe raises ``ValueError``:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.from_array(
                ...     np.ones((3, 3), dtype=np.float32),
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326),
                ... )
                >>> ds.wrap_longitude()  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: wrap_longitude requires a global grid ...

                ```

        See Also:
            to_crs: Reproject to a different CRS (a full warp, not a column roll).
        """
        lon = self._ds.lon
        # Require a grid that actually spans the globe in the 0-360 frame: the longitudinal extent
        # (n_columns * cell) must be ~360° (within one cell), and the last longitude must exceed 180.
        # This rejects regional windows (e.g. 200-330) and grids already in the -180/180 frame, which
        # the bare `lon[-1] > 180` check would have silently mis-wrapped.
        cell = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.0
        spans_globe = cell > 0 and abs(len(lon) * cell - 360.0) <= cell
        if not (spans_globe and lon[-1] > 180):
            raise ValueError(
                "wrap_longitude requires a global grid spanning ~360° in the 0-360 longitude "
                f"frame; got {len(lon)} columns covering "
                f"{float(lon[0]):g}..{float(lon[-1]):g}°."
            )

        src = self._ds.raster
        n_columns = src.RasterXSize
        first_to_translated = int(np.nonzero(lon > 180)[0][0])
        gt = list(src.GetGeoTransform())
        gt[0] = self._ds.top_left_corner[0] - 180

        # Route to the lazy VRT only when the source is referenceable by a real on-disk path
        # (a plain file). In-memory views — e.g. a NetCDF variable via AsClassicDataset — report the
        # backing file in GetFileList() but expose no usable description for a VRT SourceFilename, so
        # they take the eager path.
        description = src.GetDescription()
        # Path.exists() returns False (it never raises) for a non-path description — an empty
        # in-memory view or a `NETCDF:"file":var` subdataset string — so those take the eager path.
        is_file_backed = bool(description) and Path(description).exists()

        if is_file_backed:
            # A — lazy: file-backed source, roll columns via a two-source VRT (no data read).
            dst = self._wrap_longitude_vrt(src, first_to_translated, gt)
        else:
            # B — eager: in-memory source has no filename for a VRT, so materialise once via
            # CreateCopy (which preserves all metadata) and roll the columns in place, reading the
            # cheap in-memory copy instead of re-reading the source a second time.
            dst = gdal.GetDriverByName("MEM").CreateCopy("", src, 0)
            order = list(range(first_to_translated, n_columns)) + list(
                range(0, first_to_translated)
            )
            for band in range(src.RasterCount):
                gdal_band = dst.GetRasterBand(band + 1)
                gdal_band.WriteArray(gdal_band.ReadAsArray()[:, order])
            dst.SetGeoTransform(gt)
        return self._ds.__class__(dst)

    @staticmethod
    def _wrap_longitude_vrt(src, first_to_translated: int, gt: list) -> gdal.Dataset:
        """Build a lazy two-source VRT that rolls 0-360 columns to -180-180 without reading data.

        Each band gets two ``SimpleSource`` entries that reference the source file by an absolute path:
        the columns ``>= first_to_translated`` (longitudes > 180, i.e. the western hemisphere in the
        -180/180 frame) are mapped to the front, and the remaining columns follow. The source
        projection, dataset metadata, and per-band no-data values are carried across. Reads against the
        returned VRT are deferred to the backing file, so no pixel data is read here.

        Args:
            src (gdal.Dataset):
                The file-backed source dataset (its ``GetDescription()`` must be a resolvable path).
            first_to_translated (int):
                Index of the first column whose longitude exceeds 180; the split point of the roll.
            gt (list):
                The destination geotransform (the source geotransform with its top-left x set to -180).

        Returns:
            gdal.Dataset:
                An in-memory VRT dataset that lazily rolls the columns when read.
        """
        n_columns, n_rows, n_bands = src.RasterXSize, src.RasterYSize, src.RasterCount
        right_width = n_columns - first_to_translated
        # Use an absolute path so the in-memory VRT resolves the source regardless of CWD; leave
        # non-path descriptions (e.g. `NETCDF:"file":var` subdatasets) untouched.
        description = src.GetDescription()
        source_name = (
            str(Path(description).resolve())
            if Path(description).exists()
            else description
        )
        source_name = escape(source_name)

        vrt = gdal.GetDriverByName("VRT").Create("", n_columns, n_rows, 0)
        vrt.SetGeoTransform(gt)
        projection = src.GetProjection()
        if projection:
            vrt.SetProjection(projection)
        vrt.SetMetadata(src.GetMetadata())

        def simple_source(
            band_index: int, src_x_off: int, dst_x_off: int, width: int
        ) -> str:
            dtype = gdal.GetDataTypeName(src.GetRasterBand(band_index).DataType)
            return (
                f"<SimpleSource>"
                f'<SourceFilename relativeToVRT="0">{source_name}</SourceFilename>'
                f"<SourceBand>{band_index}</SourceBand>"
                f'<SourceProperties RasterXSize="{n_columns}" RasterYSize="{n_rows}" '
                f'DataType="{dtype}"/>'
                f'<SrcRect xOff="{src_x_off}" yOff="0" xSize="{width}" ySize="{n_rows}"/>'
                f'<DstRect xOff="{dst_x_off}" yOff="0" xSize="{width}" ySize="{n_rows}"/>'
                f"</SimpleSource>"
            )

        for band_index in range(1, n_bands + 1):
            source_band = src.GetRasterBand(band_index)
            vrt.AddBand(source_band.DataType)
            vrt_band = vrt.GetRasterBand(band_index)
            no_data = source_band.GetNoDataValue()
            if no_data is not None:
                vrt_band.SetNoDataValue(no_data)
            vrt_band.SetMetadataItem(
                "source_0",
                simple_source(band_index, first_to_translated, 0, right_width),
                "new_vrt_sources",
            )
            vrt_band.SetMetadataItem(
                "source_1",
                simple_source(band_index, 0, right_width, first_to_translated),
                "new_vrt_sources",
            )
        return vrt

    def resample(
        self,
        cell_size: int | float | tuple[float, float],
        method: str = DEFAULT_RESAMPLING,
    ) -> Dataset:
        """Resample a raster to a new cell size.

        Resample the raster to ``cell_size`` using the requested interpolation method, keeping the
        existing CRS and extent. Returns a new in-memory Dataset; the source is left unchanged.

        Args:
            cell_size (int | float | tuple):
                New cell size to resample the raster to, in the units of the raster CRS. A scalar
                applies to both axes (square cells); an ``(x_res, y_res)`` pair gives non-square
                cells (e.g. ``(2.0, 1.0)`` for 2° longitude by 1° latitude).
            method (str):
                Resampling method, case-insensitive. Default is "nearest neighbor". Allowed values: "nearest"
                (alias "nearest neighbor"), "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                "mode", "max", "min", "med", "q1", "q3", "sum", and "rms" (the GDAL warp algorithms;
                "sum"/"rms" need GDAL >= 3.1/3.3). Note: the aggregating algorithms ("average", "mode",
                "med", "q1", "q3", "sum", "rms") are not no-data-aware on this warp path — no-data cells
                inside a resampling kernel are mixed into the result. Prefer "nearest" on rasters that
                carry a no-data marker.

        Returns:
            Dataset:
                A new resampled Dataset.

        Raises:
            TypeError: If ``method`` is not a string.
            ValueError: If ``method`` is not one of the supported interpolation methods.

        Examples:
            - Create a 4-band 10×10 dataset at lon/lat (0, 0) with a 0.05° cell size, then resample to a
              coarser 0.1° cell. Halving the resolution halves the row/column count in each dimension
              (10 → 5), and the source CRS and band count carry through unchanged:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 10, 10)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> (dataset.rows, dataset.columns, dataset.band_count)
              (10, 10, 4)
              >>> resampled = dataset.resample(cell_size=0.1)
              >>> (resampled.rows, resampled.columns, resampled.band_count, resampled.epsg)
              (5, 5, 4, 4326)
              >>> resampled.geotransform[1]
              0.1

              ```
              ![resample-source](./../../_images/dataset/resample-source.png)
              ![resample-new](./../../_images/dataset/resample-new.png)
        """
        resampling_method: int = resolve_resampling(method)
        # cell_size may be a scalar (square) or an (x_res, y_res) pair (non-square output).
        x_res, y_res = _resolve_resolution(cell_size)

        sr_src = sr_from_wkt(self._ds.crs)
        # NetCDF variable views expose their CRS as an EPSG code (derived from CF coordinates) rather
        # than WKT on the raster, so `crs` can be empty even when `epsg` is known. Fall back to epsg to
        # avoid building a corrupt SpatialReference (which fails on ExportToWkt) (#588).
        if not self._ds.crs and self._ds.epsg:
            sr_src = sr_from_epsg(self._ds.epsg)

        ulx = self._ds.geotransform[0]
        uly = self._ds.geotransform[3]
        # transform the right lower corner point
        lrx = self._ds.geotransform[0] + self._ds.geotransform[1] * self._ds.columns
        lry = self._ds.geotransform[3] + self._ds.geotransform[5] * self._ds.rows

        # new geotransform — separate X/Y cell sizes so non-square output is supported
        new_geo = (
            self._ds.geotransform[0],
            x_res,
            self._ds.geotransform[2],
            self._ds.geotransform[3],
            self._ds.geotransform[4],
            -1 * y_res,
        )
        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / x_res))
        rows = int(np.round(abs(uly - lry) / y_res))
        dtype = self._ds.gdal_dtype[0]
        bands = self._ds.band_count

        dst_obj = self._ds.__class__._build_dataset(
            cols,
            rows,
            bands,
            dtype,
            new_geo,
            sr_src.ExportToWkt(),
            self._ds.no_data_value,
        )
        gdal.ReprojectImage(
            self._ds.raster,
            dst_obj.raster,
            sr_src.ExportToWkt(),
            sr_src.ExportToWkt(),
            resampling_method,
        )

        return dst_obj

    def _reproject_with_ReprojectImage(
        self,
        dst_sr: osr.SpatialReference,
        method: int = gdal.GRA_NearestNeighbour,
    ) -> Dataset:
        """Reproject the dataset by deriving an extent from corner reprojection.

        Drives the alignment-preserving branch of :meth:`to_crs` — chosen by
        ``maintain_alignment=True``. Reprojects the source corners through
        :func:`pyramids.base.crs.reproject_coordinates` to compute the output
        extent, measures the X/Y cell-step independently (so a non-square
        output aspect is honoured), allocates the destination raster, and
        finally runs :func:`gdal.ReprojectImage` to fill it.

        Both source and destination spatial references are normalised to
        ``OAMS_TRADITIONAL_GIS_ORDER`` before the identity check. This lets
        :meth:`osr.SpatialReference.IsSame` report semantic equality even when
        the two SRSes were built from different axis-order strategies (the
        common case: a ``sr_from_wkt(self._ds.crs)`` source + a
        ``sr_from_user_input`` target), which is what enables the same-CRS
        shortcut to actually fire. See issue #418 for the underlying bug.

        For a geographic source whose left edge sits past longitude 180, the
        edge is shifted into the western hemisphere (``- 360``) before
        reprojection so the corner-derived extent does not collapse across
        the dateline.

        Args:
            dst_sr: Target spatial reference. Any axis-mapping strategy is
                accepted; the function normalises only the *source* side.
                Built from ``Spatial.to_crs(..., maintain_alignment=True)``
                via :func:`pyramids.base.crs.sr_from_user_input`, but callers
                may pass any pre-built SRS.
            method: GDAL resampling algorithm constant (e.g.
                ``gdal.GRA_NearestNeighbour``, ``gdal.GRA_Bilinear``,
                ``gdal.GRA_Cubic``). Resolve a method *name* through
                :func:`pyramids.base._utils.resolve_resampling` when calling
                from outside :meth:`to_crs`.

        Returns:
            Dataset: A new ``Dataset`` covering the reprojected extent. Cell
            size equals the corner-derived per-axis cell-step on the target
            CRS; row and column counts are derived from the extent / cell-step
            ratio (so the output shape is approximately, not exactly, the
            source shape — corner-sampled spacings are accurate for affine
            reprojections and approximate for footprints spanning large
            latitude ranges, where the gdal.Warp path is preferred).

        Examples:
            - Identity reprojection: passing the source's own CRS hits the
              ``IsSame`` shortcut and preserves the source geotransform
              bit-exactly. Use the public :meth:`to_crs` facade rather than
              calling this private method directly:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.ones((5, 5), dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(10.0, 50.0), cell_size=0.5, epsg=4326),
                ... )
                >>> result = ds.to_crs(to_epsg=4326, maintain_alignment=True)
                >>> result.geotransform == ds.geotransform
                True
                >>> (result.rows, result.columns) == (ds.rows, ds.columns)
                True

                ```
            - Cross-CRS alignment-preserving reproject: 4326 → 3857 keeps the
              source row/column count and changes the cell size to metres.
              At 60°N the longitudinal cell size is roughly half the
              latitudinal cell size, so the output is non-square:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((10, 10), dtype=np.float32)
                >>> ds = Dataset.from_array(
                ...     arr,
                ...     no_data_value=-9999.0,
                ...     geo_ref=GeoReference(top_left_corner=(10.0, 60.5), cell_size=0.1, epsg=4326),
                ... )
                >>> result = ds.to_crs(to_epsg=3857, maintain_alignment=True)
                >>> result.epsg
                3857
                >>> abs(result.geotransform[5]) > abs(result.geotransform[1])
                True

                ```

        See Also:
            - :meth:`Spatial.to_crs`: Public facade that picks this method
              when ``maintain_alignment=True`` and routes through
              :func:`gdal.Warp` otherwise.
            - :func:`pyramids.base.crs.reproject_coordinates`: Reprojects the
              corner / step coordinate pairs used to derive the destination
              extent and cell size.
        """
        src_gt = self._ds.geotransform
        src_x = self._ds.columns
        src_y = self._ds.rows

        src_sr = sr_from_wkt(self._ds.crs)
        # Normalise to traditional GIS axis order (lon/easting first). sr_from_wkt
        # preserves GDAL's default OAMS_AUTHORITY_COMPLIANT order, which is
        # lat-first for geographic CRSes; dst_sr comes from sr_from_user_input,
        # which always uses traditional order. Aligning both sides here lets
        # IsSame() report semantic equality (instead of WKT-byte equality, which
        # fails for two SRSes that differ only in axis-mapping strategy — #418)
        # and removes any axis-order surprise from downstream reprojection math.
        src_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        src_wkt = src_sr.ExportToWkt()
        dst_wkt = dst_sr.ExportToWkt()
        same_crs = bool(src_sr.IsSame(dst_sr))

        if not same_crs:
            # In a geographic source whose longitudes wrap past 180, shift the
            # left edge into the western hemisphere before reprojecting so the
            # corner-derived extent does not collapse across the dateline.
            west_edge = (
                src_gt[0] - 360
                if src_sr.IsGeographic() and src_gt[0] > 180
                else src_gt[0]
            )
            xs = [west_edge, west_edge + src_gt[1] * src_x]
            ys = [src_gt[3], src_gt[3] + src_gt[5] * src_y]
            [ulx, lrx], [uly, lry] = reproject_coordinates(
                xs, ys, from_crs=src_wkt, to_crs=dst_wkt
            )
        else:
            ulx = src_gt[0]
            uly = src_gt[3]
            lrx = src_gt[0] + src_gt[1] * src_x
            lry = src_gt[3] + src_gt[5] * src_y

        # measure the X and Y cell-size separately by reprojecting a
        # one-pixel step on each axis. The previous code only stepped
        # X (passing `ys = [src_gt[3], src_gt[3]]`) and reused the X
        # spacing for Y, which forced square output pixels and
        # silently squashed non-square reprojections (e.g. 4326 →
        # 3857 at non-zero latitude). Corner-sampled spacings are
        # exact for affine transforms (UTM ↔ lat-lon, equal-area)
        # and approximate for footprints spanning large latitude
        # ranges where local pixel size varies — for those cases
        # route through the gdal.Warp path in `Spatial.to_crs`.
        x_pair_xs = [src_gt[0], src_gt[0] + src_gt[1]]
        x_pair_ys = [src_gt[3], src_gt[3]]
        y_pair_xs = [src_gt[0], src_gt[0]]
        y_pair_ys = [src_gt[3], src_gt[3] + src_gt[5]]

        if not same_crs:
            # x_pair_xs and x_pair_ys are horizontally spaced by the cell size, after reprojection gives the cell size
            # in x
            new_x_xs, _ = reproject_coordinates(
                x_pair_xs,
                x_pair_ys,
                from_crs=src_wkt,
                to_crs=dst_wkt,
                precision=6,
            )
            # y_pair_xs and y_pair_ys are vertically spaced by the cell size, after reprojection gives the cell size
            # in y
            _, new_y_ys = reproject_coordinates(
                y_pair_xs,
                y_pair_ys,
                from_crs=src_wkt,
                to_crs=dst_wkt,
                precision=6,
            )
        else:
            new_x_xs = x_pair_xs
            new_y_ys = y_pair_ys

        x_spacing = np.abs(new_x_xs[0] - new_x_xs[1])
        y_spacing = np.abs(new_y_ys[0] - new_y_ys[1])

        cols = int(np.round(abs(lrx - ulx) / x_spacing))
        rows = int(np.round(abs(uly - lry) / y_spacing))

        dtype = self._ds.gdal_dtype[0]
        new_geo = (
            ulx,
            x_spacing,
            src_gt[2],
            uly,
            src_gt[4],
            np.sign(src_gt[-1]) * y_spacing,
        )
        dst_obj = self._ds.__class__._build_dataset(
            cols,
            rows,
            self._ds.band_count,
            dtype,
            new_geo,
            dst_sr.ExportToWkt(),
            self._ds.no_data_value,
        )
        gdal.ReprojectImage(
            self._ds.raster,
            dst_obj.raster,
            src_sr.ExportToWkt(),
            dst_sr.ExportToWkt(),
            method,
        )
        return dst_obj

    def fill_gaps(self, mask, src_array: np.ndarray) -> np.typing.NDArray:
        """Fill gaps in src_array using nearest neighbors where mask indicates valid cells.

        Args:
            mask (Dataset | np.ndarray):
                Mask dataset or array used to determine valid cells.
            src_array (np.ndarray):
                Source array whose gaps will be filled.

        Returns:
            np.ndarray: The source array with gaps filled where applicable.
        """
        # align function only equate the no of rows and columns only
        # match no_data_value inserts no_data_value in src raster to all places like mask
        # still places that has no_data_value in the src raster, but it is not no_data_value in the mask
        # and now has to be filled with values
        # compare no of element that is not no_data_value in both rasters to make sure they are matched
        # if both inputs are rasters
        # read_array() is called with no chunks=, so it always returns a plain
        # ndarray here (the dask.Array arm of ArrayLike is unreachable).
        mask_array = cast(np.typing.NDArray, mask.read_array())
        mask_noval = mask.no_data_value[0]

        if isinstance(mask, RasterBase) and isinstance(self._ds, RasterBase):
            src_no_data = is_no_data(src_array, self._ds.no_data_value[0])
            mask_no_data = is_no_data(mask_array, mask_noval)
            elem_src = src_array.size - np.count_nonzero(src_array[src_no_data])
            elem_mask = mask_array.size - np.count_nonzero(mask_array[mask_no_data])

            # Cells that are out-of-domain in src but in-domain in mask
            # need to be interpolated from neighbors.
            if elem_mask > elem_src:
                gap_rows, gap_cols = np.nonzero(src_no_data & ~mask_no_data)
                src_array = Vectorize._nearest_neighbour(
                    src_array,
                    self._ds.no_data_value[0],
                    gap_rows.tolist(),
                    gap_cols.tolist(),
                )
        return src_array

    def _assert_crop_aligned(
        self, mask: gdal.Dataset | np.ndarray, row: int, col: int
    ) -> None:
        """Raise if source and mask differ in size, grid origin/cell size, or CRS."""
        if row != self._ds.rows or col != self._ds.columns:
            raise ValueError(
                "Two rasters have different number of columns or rows, please resample or match both rasters"
            )
        if isinstance(mask, RasterBase):
            if (
                self._ds.top_left_corner != mask.top_left_corner
                or self._ds.cell_size != mask.cell_size
            ):
                raise ValueError(
                    "the location of the upper left corner of both rasters is not the same or cell size is "
                    "different please match both rasters first "
                )
            if mask.epsg != self._ds.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate systems please reproject one of them to "
                    "the other raster coordinate system"
                )

    def _apply_mask_nodata(
        self,
        src_array: np.ndarray,
        mask_no_data: np.ndarray,
        band_count: int,
        no_data_value: list | None = None,
    ) -> None:
        """Write the source no-data value into the masked cells (per band).

        `no_data_value` may be a caller-precomputed, dtype-checked per-band list; the
        tiled crop passes it so the coercion runs once instead of per tile. When
        `None` the multi-band path validates it here as before.
        """
        if band_count > 1:
            # check the no_data_value complies with the src dtype before writing it
            # into cells (a band full of values may never use its no_data_value).
            if no_data_value is None:
                no_data_value = self._ds._check_no_data_value(self._ds.no_data_value)
            for band in range(self._ds.band_count):
                src_array[band, mask_no_data] = no_data_value[band]
        else:
            src_array[mask_no_data] = self._ds.no_data_value[0]

    def _write_bands(
        self, dst_obj: Any, src_array: np.ndarray, band_count: int
    ) -> None:
        """Write the (possibly multi-band) array into the destination raster."""
        if band_count > 1:
            for band in range(band_count):
                dst_obj.raster.GetRasterBand(band + 1).WriteArray(src_array[band, :, :])
        else:
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)

    def _crop_aligned(
        self,
        mask: gdal.Dataset | np.ndarray,
        mask_noval: int | float | None = None,
        fill_gaps: bool = False,
    ) -> Dataset:
        """Clip/crop by matching the nodata layout from mask to the source raster.

        Both rasters must have the same dimensions (rows and columns). Use MatchRasterAlignment prior to this
        method to align both rasters.

        Args:
            mask (Dataset | np.ndarray):
                Mask raster to get the location of the NoDataValue and where it is in the array.
            mask_noval (int | float, optional):
                In case the mask is a numpy array, the mask_noval has to be given.
            fill_gaps (bool):
                Whether to fill gaps after cropping. Default is False.

        Returns:
            Dataset:
                The raster with NoDataValue stored in its cells exactly the same as the source raster.
        """
        if isinstance(mask, RasterBase):
            mask_gt = mask.geotransform
            row = mask.rows
            col = mask.columns
            mask_noval = mask.no_data_value[0]
        elif isinstance(mask, np.ndarray):
            if mask_noval is None:
                raise ValueError(
                    "You have to enter the value of the no_val parameter when the mask is a numpy array"
                )
            row, col = mask.shape
        else:
            raise TypeError(
                "The second parameter 'mask' has to be either gdal.Dataset or numpy array"
                f"given - {type(mask)}"
            )

        band_count = self._ds.band_count
        src_sref = sr_from_wkt(self._ds.crs)

        self._assert_crop_aligned(mask, row, col)

        # No path -> in-memory, which is what driver="MEM" used to say explicitly.
        dst = self._ds.__class__._create_dataset(
            col, row, band_count, self._ds.gdal_dtype[0]
        )
        # if the mask is a numpy array there's no geotransform / CRS
        # to copy from it; fall back to the source raster's because
        # the contract requires both rasters to be already aligned.
        if isinstance(mask, RasterBase):
            dst.SetGeoTransform(mask_gt)
            dst.SetProjection(mask.crs)
        else:
            dst.SetGeoTransform(self._ds.geotransform)
            dst.SetProjection(src_sref.ExportToWkt())

        dst_obj = self._ds.__class__(dst)
        # set the no data value
        dst_obj._set_no_data_value(self._ds.no_data_value)

        # Apply the mask's no-data layout tile by tile for the raster-mask,
        # no-gap-fill case, so neither the full source (all bands) nor the full
        # mask is ever materialised at once (#969). The mask-apply is purely
        # per-pixel, so the tiled result is byte-identical to the whole-array
        # path. The numpy-array mask (already in memory) and the gap-filling
        # path (which interpolates across the whole array) stay eager below.
        if isinstance(mask, RasterBase) and not fill_gaps:
            self._crop_aligned_tiled(mask, mask_noval, dst_obj, band_count)
            return dst_obj

        # read_array() is called with no chunks=, so it always returns a plain
        # ndarray here (the dask.Array arm of ArrayLike is unreachable).
        if isinstance(mask, RasterBase):
            mask_array = cast(np.typing.NDArray, mask.read_array(band=0))
        else:
            mask_array = mask.copy()
        src_array = cast(np.typing.NDArray, self._ds.read_array())

        mask_no_data = is_no_data(mask_array, mask_noval)
        self._apply_mask_nodata(src_array, mask_no_data, band_count)

        if fill_gaps:
            src_array = self.fill_gaps(mask, src_array)

        self._write_bands(dst_obj, src_array, band_count)
        return dst_obj

    def _crop_aligned_tiled(
        self,
        mask: RasterBase,
        mask_noval: int | float | None,
        dst_obj: Any,
        band_count: int,
    ) -> None:
        """Stamp the mask's no-data layout onto the source one window at a time.

        Reads the source (all bands) and the mask band a square tile at a time,
        applies the source no-data value where the mask is no-data, and writes
        each masked block straight into `dst_obj`, so the full arrays are never
        held together. The operation is purely per-pixel, hence byte-identical
        to the whole-array path (#969).

        Args:
            mask: The already-aligned raster mask supplying the no-data layout.
            mask_noval: The mask's no-data value used to locate masked cells.
            dst_obj: The destination Dataset the masked blocks are written into.
            band_count: Number of bands in the source raster.
        """
        # Coerce the per-band no-data value once here rather than on every tile.
        no_data_value = (
            self._ds._check_no_data_value(self._ds.no_data_value)
            if band_count > 1
            else None
        )
        for xoff, yoff, xsize, ysize in self._ds.io._tile_offsets():
            window = [xoff, yoff, xsize, ysize]
            # read_array() is called with no chunks=, so it always returns a
            # plain ndarray here (the dask.Array arm of ArrayLike is unreachable).
            mask_tile = cast(np.typing.NDArray, mask.read_array(band=0, window=window))
            src_tile = cast(np.typing.NDArray, self._ds.read_array(window=window))
            mask_no_data = is_no_data(mask_tile, mask_noval)
            self._apply_mask_nodata(src_tile, mask_no_data, band_count, no_data_value)
            if band_count > 1:
                for band in range(band_count):
                    dst_obj.raster.GetRasterBand(band + 1).WriteArray(
                        src_tile[band, :, :], xoff, yoff
                    )
            else:
                dst_obj.raster.GetRasterBand(1).WriteArray(src_tile, xoff, yoff)

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster."""
        if not isinstance(mask, RasterBase):
            raise TypeError("The second parameter should be a Dataset")

        return self._ds.rows == mask.rows and self._ds.columns == mask.columns

    def align(
        self,
        alignment_src: Dataset,
        *,
        method: str = DEFAULT_RESAMPLING,
    ) -> Dataset:
        """Align the current dataset (rows and columns) to match a given dataset.

        Copies spatial properties from alignment_src to the current raster:
            - The coordinate system
            - The number of rows and columns
            - Cell size
        Then resamples values from the current dataset onto that grid using ``method`` (nearest neighbor by
        default, so the historical behaviour is unchanged).

        Args:
            alignment_src (Dataset):
                Spatial information source raster to get the spatial information (coordinate system, number of rows and
                columns). The data values of the current dataset are resampled to this alignment.
            method (str, keyword-only):
                Resampling method, case-insensitive. Default is "nearest neighbor". Accepts the same algorithm
                names as :meth:`Spatial.to_crs`: "nearest" (alias "nearest neighbor"), "bilinear", "cubic",
                "cubic_spline", "lanczos", "average", "mode", "max", "min", "med", "q1", "q3", "sum", and "rms"
                (the GDAL warp algorithms; "sum"/"rms" need GDAL >= 3.1/3.3). The aggregating algorithms
                ("min", "max", "average", "mode", "med", "q1", "q3", "sum", "rms") honour the source no-data
                value when the raster declares one: no-data cells are excluded from the kernel and a kernel that
                is entirely no-data yields no-data. A raster that carries no no-data marker has nothing to
                exclude, so every cell — including a sentinel fill such as 9999 — is treated as valid data.

        Returns:
            Dataset: A new aligned Dataset.

        Note:
            - **`method` is keyword-only.** Pass it by name (`align(template, method="bilinear")`), matching
              :meth:`DatasetCollection.align`. This deliberately differs from :meth:`Spatial.to_crs` /
              :meth:`Spatial.resample`, which accept `method` positionally — a bare positional resampling name
              after a `Dataset` template reads poorly and would collide with the collection API's `inplace` slot.
            - **Output dtype follows the template, not the source.** The aligned raster is built with
              `alignment_src`'s data type, so resampling a floating-point source onto an integer-typed template
              with an interpolating `method` ("bilinear"/"cubic"/...) silently drops the fractional part (the
              interpolated values are cast to the template's integer type). Match the template dtype to the
              source, or use "nearest", to avoid it.
            - **Cross-CRS aligns resample twice.** When the source and `alignment_src` CRSes differ, the data is
              first reprojected onto an intermediate grid and then resampled onto the template grid, so a
              non-nearest `method` is applied twice. For interpolating kernels ("bilinear"/"cubic") that means the
              result is a little more smoothed than a same-CRS align with the identical `method`; for aggregating
              kernels it changes the statistic itself — "sum" double-aggregates and inflates the total, and
              "average"/"mode"/... become differently weighted, not smoother. The output grid is always exact;
              only pixel values differ. Reproject the source to the template CRS first when an aggregating `method`
              must be exact (a single-pass warp).

        Examples:
            - The source dataset has a `top_left_corner` at (0, 0) with a 5*5 alignment, and a 0.05 degree cell size.

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(5, 5)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> (dataset.rows, dataset.columns, dataset.epsg, dataset.band_count)
              (5, 5, 4326, 1)

              ```

            - The dataset to be aligned has a top_left_corner at (-0.1, 0.1) (i.e., it has two more rows on top of the
              dataset, and two columns on the left of the dataset).

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr_target = np.random.rand(10, 10)
              >>> dataset_target = Dataset.from_array(
              ...     arr_target,
              ...     geo_ref=GeoReference(top_left_corner=(-0.1, 0.1), cell_size=0.07, epsg=4326),
              ... )
              >>> (dataset_target.rows, dataset_target.columns, dataset_target.geotransform[1])
              (10, 10, 0.07)

              ```

            ![align-source-target](./../../_images/dataset/align-source-target.png)

            - Now call the `align` method and use the source dataset as the alignment template. The aligned
              dataset adopts the source's cell size, dimensions, and CRS:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> source = Dataset.from_array(
              ...     np.random.rand(5, 5),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> target = Dataset.from_array(
              ...     np.random.rand(10, 10),
              ...     geo_ref=GeoReference(top_left_corner=(-0.1, 0.1), cell_size=0.07, epsg=4326),
              ... )
              >>> aligned = target.align(source)
              >>> (aligned.rows, aligned.columns, aligned.geotransform[1], aligned.epsg)
              (5, 5, 0.05, 4326)

              ```

            ![align-result](./../../_images/dataset/align-result.png)

            - Choose a different resampling method (e.g. bilinear) while still landing on the template's exact
              grid. The default stays nearest neighbor, so only callers that opt in change behaviour:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> source = Dataset.from_array(
              ...     np.arange(25, dtype=np.float32).reshape(5, 5),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> template = Dataset.from_array(
              ...     np.zeros((10, 10), dtype=np.float32),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.025, epsg=4326),
              ... )
              >>> aligned = source.align(template, method="bilinear")
              >>> (aligned.rows, aligned.columns, aligned.epsg)
              (10, 10, 4326)

              ```

        Raises:
            TypeError: `alignment_src` is not a `RasterBase`, or `method` is not a string.
            ValueError: `method` is not one of the supported interpolation methods.
            CRSError: Either raster has no CRS. Aligning needs both, and
                pyramids will not assume WGS 84 for an unprojected grid
                (ARC-26).
        """
        if isinstance(alignment_src, RasterBase):
            src = alignment_src
        else:
            raise TypeError(
                "First parameter should be a Dataset read using Dataset.openRaster or a path to the raster, "
                f"given {type(alignment_src)}"
            )

        # Validate/resolve the method up front so a bad name is rejected before any warp runs, even on the
        # same-CRS path (where `to_crs` below is skipped and would otherwise be the only validator).
        resample_alg: int = resolve_resampling(method)

        # reproject the raster to match the projection of alignment_src
        # Both sides are required, and the check must sit OUTSIDE the inequality
        # below: two CRS-less rasters both report `epsg is None`, so they compare
        # equal, skip the branch, and the result is still stamped with the
        # reference's projection further down (ARC-26).
        own_crs = require_crs_spec(
            self._ds.epsg, self._ds.crs, "align a raster onto another grid"
        )
        target_crs = require_crs_spec(
            src.epsg, src.crs, "align onto this reference grid"
        )
        reprojected_raster_b: Dataset = self._ds
        # Compare the resolved CRS, not `epsg` alone: two grids with different
        # real CRSes can both report `epsg is None` (the geostationary shape),
        # compare equal, skip the warp, and be stamped with the reference's
        # projection further down (ARC-26). `crs_equal` rather than `!=`, because
        # `crs_spec` falls back to the raw WKT for a CRS with no EPSG authority
        # -- exactly that case -- so two spellings of one CRS would otherwise
        # warp the data through an identity transform.
        if not crs_equal(own_crs, target_crs):
            reprojected_raster_b = self.to_crs(target_crs, method=method)  # type: ignore[assignment]
        dst_obj = self._ds.__class__._build_dataset(
            src.columns,
            src.rows,
            self._ds.band_count,
            src.gdal_dtype[0],
            src.geotransform,
            src.crs,
            self._ds.no_data_value,
        )
        # resample the reprojected_RasterB onto the freshly built target grid using the requested algorithm
        gdal.ReprojectImage(
            reprojected_raster_b.raster,
            dst_obj.raster,
            src.crs,
            src.crs,
            resample_alg,
        )
        # ReprojectImage moves only pixels onto the freshly built grid, so the
        # output would otherwise lose the class legend, RAT, and band/dataset
        # metadata. Carry them over from the (possibly reprojected) source (#1029).
        carry_raster_metadata(reprojected_raster_b.raster, dst_obj.raster)

        return dst_obj

    def _crop_with_raster(
        self,
        mask: gdal.Dataset | str,
    ) -> Dataset:
        """Crop this raster using another raster as a mask.

        Args:
            mask (Dataset | str):
                The raster you want to use as a mask to crop this raster; it can be a path or a GDAL Dataset.

        Returns:
            Dataset:
                The cropped raster.
        """
        # get information from the mask raster
        if isinstance(mask, (str, Path)):
            mask = self._ds.__class__.read_file(mask)
        elif not isinstance(mask, RasterBase):
            raise TypeError(
                "The second parameter has to be either path to the mask raster or a gdal.Dataset object"
            )
        if not self._check_alignment(mask):
            # first align the mask with the src raster
            mask = mask.align(self._ds)
        # crop the src raster with the aligned mask
        dst_obj = self._crop_aligned(mask)

        # Hand the trim the *base* Dataset, not the subclass. `_crop_aligned` builds its result as
        # `self._ds.__class__(...)`, so on a NetCDF receiver it is a NetCDF — and
        # Historically (#1073) this died with `TypeError: got an unexpected keyword
        # argument 'geo'`, because the NetCDF override took a different keyword set
        # from the base. #1075 converged the signatures, so that specific divergence
        # is gone -- but a subclass may still override a constructor, so intermediate
        # GDAL results are still built through the base class rather than `type(self)`.
        dst_obj = Spatial._correct_wrap_cutline_error(Spatial._as_base_dataset(dst_obj))
        return dst_obj

    @staticmethod
    def _base_dataset_class(source: type) -> type[Dataset]:
        """The plain `Dataset` class sitting directly above `RasterBase` in `source`'s MRO.

        Intermediate GDAL results must not be built through a subclass: a subclass may override the
        constructors shared raster code calls. `NetCDF.from_array` and the base now agree on a
        keyword-only `geo_ref` core (#1075), so that particular mismatch no longer bites; the
        guard remains because an override may still narrow behaviour — `NetCDF.from_array`
        returns a `Container` and adds `variable_name` / `dims` / `encoding` / `attrs`.

        Args:
            source: The class to walk, normally `type(some_dataset)`.

        Returns:
            type[Dataset]: The base class, or `source` unchanged when the walk finds nothing. That
            leaves such a caller exactly where it was, which is safer than the bare `next()` this
            replaced — whose `StopIteration` would have aborted the crop outright. (`Dataset`
            itself is not usable as the default: this module imports it lazily to break a cycle,
            so the name does not exist at call time.)
        """
        found = next(
            (c for c in source.__mro__ if RasterBase in getattr(c, "__bases__", ())),
            source,
        )
        return cast("type[Dataset]", found)

    @staticmethod
    def _as_base_dataset(src: RasterBase) -> Dataset:
        """Re-wrap `src` as the plain `Dataset` above `RasterBase`, preserving how it was opened.

        See :meth:`_base_dataset_class` for why the subclass must not carry into an intermediate.
        Returns `src` unchanged when it already is the base class, so the common path allocates
        nothing; otherwise the new wrapper is given the same access mode and GDAL environment, or
        it would silently come back read-only and without the source's open options.

        Args:
            src: The dataset to normalise.

        Returns:
            Dataset: `src` itself, or a base-class wrapper over the same GDAL dataset.
        """
        base_cls = Spatial._base_dataset_class(src.__class__)
        if type(src) is base_cls:
            return cast("Dataset", src)
        carried = {
            name: getattr(src, name)
            for name in ("access", "gdal_env", "open_options")
            if getattr(src, name, None) is not None
        }
        return cast("Dataset", base_cls(src.raster, **carried))

    @staticmethod
    def _cutline_window_bounds(
        src: RasterBase, feature: FeatureCollection | GeoDataFrame
    ) -> tuple[float, float, float, float] | None:
        """The source-grid window that contains every cell the cutline can touch.

        Returns `(west, south, east, north)` snapped to source pixel edges and grown by
        one cell on each side so a "touch" crop keeps every cell the polygon merely
        grazes, then clipped to the source extent. `gdal.Warp` bounded to this window
        masks and trims to exactly the same cells the full-source warp would, because the
        cutline masks every cell outside the polygon regardless of the window — only the
        read shrinks.

        The window is only computed when the cutline is **already in the source CRS**. A
        reprojected cutline is not eligible: `geopandas` reprojects a polygon by moving
        its vertices, leaving straight edges, whereas GDAL densifies the cutline when it
        transforms it, so its masked region bulges past the vertex envelope on a curving
        projection — the window would under-cover it and the crop would come back
        truncated. A same-CRS cutline is applied by GDAL with no transform and so no
        densification, so its edges stay straight in source coordinates and the polygon's
        `total_bounds` envelope is exact — for any polygon, not only axis-aligned ones.

        Returns `None`, falling back to the full-source warp, whenever the optimisation
        cannot be applied safely: a rotated, sheared or non-north-up geotransform; a
        missing or differing CRS; a degenerate (non-finite) envelope; or a cutline that
        does not overlap the source (left for the existing "no valid pixels" error).

        Args:
            src: The raster being cropped.
            feature: The cutline.

        Returns:
            tuple[float, float, float, float] | None:
                The clipped window, or None to fall back.
        """
        window: tuple[float, float, float, float] | None = None
        x0, dx, row_skew, y0, col_skew, dy = src._raster.GetGeoTransform()
        source_crs = src.crs
        north_up = not row_skew and not col_skew and dx > 0 and dy < 0
        same_crs = (
            bool(source_crs)
            and feature.crs is not None
            and crs_equal(source_crs, feature.crs.to_wkt())
        )
        if north_up and same_crs:
            minx, miny, maxx, maxy = (float(v) for v in feature.total_bounds)
            if all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
                src_west, src_east = x0, x0 + dx * src.columns
                src_south, src_north = y0 + dy * src.rows, y0  # dy < 0
                # Snap outward to whole pixels, then one cell of "touch" margin per side.
                west = x0 + (math.floor((minx - x0) / dx) - 1) * dx
                east = x0 + (math.ceil((maxx - x0) / dx) + 1) * dx
                north = y0 + (math.floor((y0 - maxy) / -dy) - 1) * dy
                south = y0 + (math.ceil((y0 - miny) / -dy) + 1) * dy
                west, east = max(west, src_west), min(east, src_east)
                south, north = max(south, src_south), min(north, src_north)
                if west < east and south < north:
                    window = (west, south, east, north)
        return window

    @staticmethod
    def _cutline_segment_length(
        src: RasterBase, feature: FeatureCollection
    ) -> float | None:
        """Densification step for the cutline, about one source pixel wide.

        Dividing the cutline's envelope into a fixed number of parts is scale-free and
        therefore wrong in general: 64 segments across a 64 km window is a 1 km chord,
        fine against 1 km pixels, but 64 segments across a continental lon/lat bbox is
        a ~50 km chord whose sagitta swallows many cells of a 30 m grid. Since the
        reprojected envelope becomes `outputBounds`, that under-coverage truncates the
        crop.

        The step is therefore measured, not assumed: one source pixel is transformed
        into the cutline's own CRS, which converts between their units without either
        being named. The result is floored at 1/4096 of the envelope so a very fine
        raster under a very large window cannot explode the vertex count.

        Args:
            src: The raster being cropped.
            feature: The cutline, in its own CRS.

        Returns:
            float | None: The segment length in the cutline's units, or ``None`` when
            the envelope is degenerate or the scale cannot be measured — in which case
            the caller reprojects the outline undensified, as it did before.
        """
        minx, miny, maxx, maxy = (float(v) for v in feature.total_bounds)
        span = max(maxx - minx, maxy - miny)
        if not (math.isfinite(span) and span > 0):
            return None
        length: float | None = None
        try:
            x0, dx, _, y0, _, dy = src._raster.GetGeoTransform()
            pixel = min(abs(dx), abs(dy))
            to_cutline = Transformer.from_crs(
                crs_from_user_input(src.crs), feature.crs, always_xy=True
            )
            ax, ay = to_cutline.transform(x0, y0)
            bx, by = to_cutline.transform(x0 + pixel, y0)
            step = math.hypot(bx - ax, by - ay)
            if math.isfinite(step) and step > 0:
                length = max(step, span / 4096.0)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            # Narrow rather than bare: these are what a missing geotransform, an
            # unparseable CRS or a transform PROJ refuses actually raise. Catching
            # everything would swallow genuine bugs in the measurement itself and
            # silently degrade the crop instead of failing.
            length = None
        return length

    @staticmethod
    def _cutline_in_source_crs(
        src: RasterBase, feature: FeatureCollection
    ) -> FeatureCollection:
        """Reproject a cutline into the raster's own CRS, densifying it first.

        A cutline in a different CRS used to be handed to GDAL as-is. That silently
        broke `touch=True` crops: :meth:`_cutline_window_bounds` refuses a differing
        CRS, so no `outputBounds` was set and `cropToCutline` is false under `touch`,
        leaving the whole source grid in place. The crop then depended entirely on
        :meth:`_correct_wrap_cutline_error` trimming an all-no-data border — and a
        raster that declares **no no-data value** has no border to detect, so
        `crop(bbox=..., epsg=<other CRS>)` returned the raster *uncropped*, with no
        error. Setting a no-data value on the same raster made the identical call crop
        correctly, which is what made the bug so easy to miss.

        Bringing the cutline into the source CRS here fixes it at the root: the window
        optimisation applies again and GDAL needs no cutline transform, so the crop is
        bounded by the cutline's own window rather than by an incidental no-data
        border. The two cases are not made identical — with a no-data value the
        subsequent trim still tightens the result by the touch margin and the
        reprojection slack, so a band that declares one crops a few cells closer —
        but neither case can any longer return the raster uncropped.

        The edges are densified before reprojecting. A straight edge in one CRS is
        generally a *curve* in another, so reprojecting only the four corners of a bbox
        would cut inside the requested area wherever the true edge bows outward — and
        because the reprojected envelope becomes `outputBounds`, an under-covered edge
        truncates the crop rather than merely blurring it. The segment length comes
        from :meth:`_cutline_segment_length`, which measures one source pixel in the
        cutline's own units, so the density follows the raster's resolution instead of
        a fixed division of the envelope.

        Args:
            src: The raster being cropped.
            feature: The cutline, in any CRS.

        Returns:
            FeatureCollection: The cutline in `src`'s CRS, or `feature` unchanged when
            either side has no CRS or they already agree.
        """
        source_crs = src.crs
        result = feature
        needs_reprojection = (
            bool(source_crs)
            and feature.crs is not None
            and not crs_equal(source_crs, feature.crs.to_wkt())
        )
        if needs_reprojection:
            densified = feature.copy()
            step = Spatial._cutline_segment_length(src, feature)
            if step is not None:
                densified.geometry = densified.geometry.segmentize(step)
            result = densified.to_crs(crs_from_user_input(source_crs))
        return result

    def _crop_bbox_windowed(
        self,
        bbox: tuple[float, float, float, float] | list[float],
        crs: Any,
    ) -> Dataset | None:
        """Crop an axis-aligned bbox by reading only its pixel window.

        The default bbox crop wraps the box in a one-row cutline and runs
        ``gdal.Warp``; over a huge or ``/vsicurl`` source that is far heavier than
        a plain windowed read (#957). When the crop needs neither reprojection nor
        antimeridian handling, this reads just the AOI window straight from the
        source and rebuilds the geotransform, so a small crop out of a very large
        remote raster never triggers a full read.

        Semantics: the crop keeps **every pixel the box overlaps** — the cells from
        ``floor`` of the west/north edges to ``ceil`` of the east/south edges in
        pixel space — the same all-touched convention :meth:`Dataset.read_array`
        uses for its ``bbox=`` argument. The ``bbox=`` crop's warp fallback
        (reprojection / rotated grids) is run with an all-touched cutline so it
        keeps the same cells; only a user-supplied polygon ``mask=`` keeps GDAL's
        centre-containment default. The windowed read is passed through the same
        :meth:`_correct_wrap_cutline_error` trim, so an all-no-data crop raises the
        same "no valid pixels" error and interior all-no-data rows/cols drop the
        same way.

        Eligibility (returns ``None`` to fall back to the warp path otherwise):

        - the source is north-up (no rotation/shear, ``dx > 0``, ``dy < 0``);
        - the bbox CRS equals the source CRS, so no reprojection is needed;
        - the box is a normal ``west < east``, ``south < north`` quadruple that
          overlaps the source (a transposed or non-overlapping box is left to the
          warp path, which reports it with a clear error).

        Args:
            bbox (tuple[float, float, float, float] | list[float]):
                ``(west, south, east, north)`` in the CRS named by ``crs``.
            crs (Any):
                CRS of ``bbox`` — anything :func:`crs_equal` accepts (EPSG code,
                WKT/authority string).

        Returns:
            Dataset | None:
                The windowed crop, or ``None`` when the fast path does not apply.
        """
        result: Dataset | None = None
        x0, dx, row_skew, y0, col_skew, dy = self._ds._raster.GetGeoTransform()
        north_up = not row_skew and not col_skew and dx > 0 and dy < 0
        source_crs = self._ds.crs
        west, south, east, north = (float(v) for v in bbox)
        eligible = (
            north_up
            and bool(source_crs)
            and crs_equal(source_crs, crs)
            and west < east
            and south < north
        )
        if eligible:
            columns, rows = self._ds.columns, self._ds.rows
            # Cells whose extent intersects the box (overlap / all-touched), snapped
            # outward to whole pixels, then clipped to the source grid. A tiny epsilon
            # (in pixel units) absorbs binary-float noise so an edge that is
            # mathematically on a pixel boundary does not flip floor/ceil by one pixel;
            # it is applied in the snap direction (floor edges up, ceil edges down) so
            # it drops nothing but noise-scale (< ~1e-9 px) overlaps, which are
            # sub-nanometre at any realistic cell size.
            eps = 1e-9
            xoff = min(max(math.floor((west - x0) / dx + eps), 0), columns)
            x_far = min(max(math.ceil((east - x0) / dx - eps), 0), columns)
            yoff = min(max(math.floor((y0 - north) / -dy + eps), 0), rows)
            y_far = min(max(math.ceil((y0 - south) / -dy - eps), 0), rows)
            x_size, y_size = x_far - xoff, y_far - yoff
            if x_size > 0 and y_size > 0:
                array = self._ds.read_array(window=[xoff, yoff, x_size, y_size])
                new_gt = (x0 + xoff * dx, dx, 0.0, y0 + yoff * dy, 0.0, dy)
                # Rebuild on the base Dataset class (never a subclass like NetCDF), whose
                # from_array has the plain array->raster behaviour this path needs.
                base_cls = Spatial._base_dataset_class(self._ds.__class__)
                dst = cast(
                    "Dataset",
                    base_cls.from_array(
                        array,
                        no_data_value=self._ds.no_data_value,
                        geo_ref=GeoReference(geo=new_gt),
                    ),
                )
                # Preserve the source CRS from its WKT so _correct_wrap_cutline_error
                # carries it onto the trimmed result (a custom CRS with no EPSG survives).
                dst.crs = source_crs
                # Same trim the warp path applies with touch=True: drop all-no-data
                # rows/cols and raise "no valid pixels" when the whole window is
                # no-data. The read above covered only the AOI, so this stays off the
                # full source.
                result = Spatial._correct_wrap_cutline_error(dst)
        return result

    def _crop_with_polygon_warp(
        self,
        feature: FeatureCollection | GeoDataFrame,
        touch: bool = True,
        cutline_all_touched: bool = False,
    ) -> Dataset:
        """Crop raster with polygon.

            - Do not convert the polygon into a raster but rather use it directly to crop the raster using the
            gdal.warp function.

        Args:
            feature (FeatureCollection | GeoDataFrame):
                Vector mask.
            touch (bool):
                Include cells that touch the polygon, not only those entirely inside the polygon mask. Defaults to True.
            cutline_all_touched (bool):
                Rasterize the cutline with `CUTLINE_ALL_TOUCHED=TRUE`, keeping every cell the cutline overlaps
                rather than only cells whose centre it contains. Set by the `bbox=` crop so its reprojection /
                rotated-grid fallback matches the windowed fast path's overlap semantics. Defaults to False (the
                GDAL centre-containment default) for a user-supplied polygon mask. Ignored when `touch` is False.

        Returns:
            Dataset:
                Cropped dataset.
        """
        if isinstance(feature, GeoDataFrame):
            feature = FeatureCollection(feature)
        else:
            if not isinstance(feature, FeatureCollection):
                raise TypeError(
                    f"The function takes only a FeatureCollection or GeoDataFrame, given {type(feature)}"
                )

        # gdal.Warp's cutlineDSName needs a *path*; stage the vector in
        # /vsimem/ through the internal OGR bridge. The path is unlinked
        # automatically when the with-block exits.
        # Intermediate GDAL warp results are built on the base Dataset class, never a subclass:
        # `_correct_wrap_cutline_error` calls `from_array`, whose NetCDF override takes a
        # different signature. See `_base_dataset_class`.
        base_cls = Spatial._base_dataset_class(self._ds.__class__)

        # The warp output (VRT) may resolve the cutline lazily, so we must
        # complete every access that could touch the cutline path inside
        # the with-block that keeps that path alive.
        # With touch=True the warp keeps the whole source grid (cropToCutline is
        # False) and _correct_wrap_cutline_error reads it back to trim the no-data
        # border, so peak memory tracked the *source*, not the crop (#854). Bound the
        # warp to the cutline's own window first: the trimmed result is identical
        # because every non-no-data cell lives inside that window, but the read shrinks
        # from the source to the crop. cropToCutline already bounds the touch=False path.
        feature = self._cutline_in_source_crs(self._ds, feature)
        window = self._cutline_window_bounds(self._ds, feature) if touch else None
        # Pin the resolution to the source's own so the windowed warp is a pixel-exact
        # subset and cannot resample; only needed when a window is set.
        gt = self._ds._raster.GetGeoTransform() if window else None
        with _feature_ogr.as_vsimem_path(feature) as cutline_path:
            warp_options = gdal.WarpOptions(
                format="VRT",
                cropToCutline=not touch,
                cutlineDSName=cutline_path,
                # State the cutline's CRS rather than letting GDAL read it back from
                # the staged file. The cutline is staged as GeoJSON, which can name a
                # CRS only as an OGC URN, so a CRS with no authority code -- an
                # orthographic or Robinson grid, or one rescued from GDAL's PROJ
                # database -- is written with no CRS at all. GDAL then assumes the
                # GeoJSON default of CRS84 and transforms metre coordinates as
                # lon/lat, failing with "Invalid latitude" (issue #964).
                cutlineSRS=(None if feature.crs is None else feature.crs.to_wkt()),
                multithread=True,
                outputBounds=window,
                xRes=abs(gt[1]) if gt else None,
                yRes=abs(gt[5]) if gt else None,
                warpOptions=(
                    ["CUTLINE_ALL_TOUCHED=TRUE"]
                    if touch and cutline_all_touched
                    else None
                ),
            )
            # `_base_dataset_class` already returns a `type[Dataset]`; the cast is for the
            # checker's benefit only.
            # Routed through warp_to_dataset for the pin: with `touch` false the
            # result is the raw VRT, which reads through to the source on every
            # access and previously held nothing alive.
            dst_obj = cast(
                "Dataset",
                warp_to_dataset(
                    self._ds,
                    warp_options,
                    dataset_class=cast("type[Dataset]", base_cls),
                    error_message="GDAL could not crop the dataset with the cutline.",
                ),
            )
            if touch:
                dst_obj = Spatial._correct_wrap_cutline_error(dst_obj)

        return dst_obj

    @staticmethod
    def _correct_wrap_cutline_error(src: Dataset) -> Dataset:
        """Trim the all-nodata border GDAL leaves after a cutline warp.

        ``gdal.Warp`` with ``cropToCutline=False`` (the ``touch=True``
        crop path) keeps the source grid and fills the cells outside the
        cutline with the no-data value, producing a frame of fully-nodata
        rows and columns around the real data. This rebuilds the dataset
        from the array with those edge rows/columns removed and the
        geotransform shifted to the new top-left corner.

        The output CRS is copied from the source **WKT** (``src.crs``)
        rather than round-tripped through ``src.epsg``: a custom CRS with
        no resolvable EPSG (e.g. a spherical-earth GRIB GEOGCS) would
        otherwise be relabelled — or, before issue #403 was fixed, crash
        on ``sr_from_epsg`` — so the exact source CRS is preserved. When the
        source is unprojected (``src.crs`` is empty) the copy is skipped, so
        the rebuilt dataset keeps the :meth:`Dataset.from_array`
        default CRS instead of having its projection wiped to empty.

        Args:
            src (Dataset): Result of the cutline warp, expected to carry a
                fully-nodata border. Its single no-data value
                (``src.no_data_value[0]``) marks the cells to trim. The
                backing array must be 2D (single band) or 3D
                (band, row, col).

        Returns:
            Dataset: A new in-memory dataset with the all-nodata border
            rows/columns removed, the geotransform shifted to the trimmed
            top-left corner, and the no-data value and band count preserved.
            The CRS is the source CRS, or the ``from_array`` default
            when the source is unprojected.

        Raises:
            ValueError: If the source array is neither 2D nor 3D.

        See Also:
            Spatial.crop: Caller that applies this correction when
                ``touch=True``.

        References:
            https://github.com/serapeum-org/pyramids/issues/74
        """
        big_array = src.read_array()
        value_to_remove = src.no_data_value[0]
        # `is_no_data`, not `==`: a NaN sentinel never equals itself, so `==` marks
        # nothing and the all-no-data frame GDAL leaves after a cutline warp
        # survives -- an oversized crop carrying a no-data border. The helper is
        # already imported and used for exactly this three times in this module.
        no_data_mask = is_no_data(big_array, value_to_remove)
        # Find rows and columns to be removed
        if big_array.ndim == 2:
            rows_to_remove = np.all(no_data_mask, axis=1)
            cols_to_remove = np.all(no_data_mask, axis=0)
            # Use boolean indexing to remove rows and columns
            small_array = big_array[~rows_to_remove][:, ~cols_to_remove]
        elif big_array.ndim == 3:
            rows_to_remove = np.all(no_data_mask, axis=(0, 2))
            cols_to_remove = np.all(no_data_mask, axis=(0, 1))
            # Use boolean indexing to remove rows and columns
            # first remove the rows then the columns
            small_array = big_array[:, ~rows_to_remove, :]
            small_array = small_array[:, :, ~cols_to_remove]
            n_rows = np.count_nonzero(~rows_to_remove)
            n_cols = np.count_nonzero(~cols_to_remove)
            small_array = small_array.reshape((src.band_count, n_rows, n_cols))
        else:
            raise ValueError("Array must be 2D or 3D")

        valid_rows = np.nonzero(~rows_to_remove)[0]
        valid_cols = np.nonzero(~cols_to_remove)[0]
        if valid_rows.size == 0 or valid_cols.size == 0:
            raise ValueError(
                "crop produced no valid pixels: the bbox / polygon does not "
                "overlap any valid (non-no-data) data in the dataset."
            )
        x_ind = valid_rows[0]
        y_ind = valid_cols[0]
        # Use the source's separate X/Y pixel sizes (gt[1], gt[5]) rather than a single cell_size, so
        # a non-square grid (e.g. 2° lon, 1° lat) keeps its true latitude spacing. Identical to the
        # old cell_size form on square grids (gt[1] == -gt[5] == cell_size).
        warp_gt = src._raster.GetGeoTransform()
        x_cell, y_cell = warp_gt[1], warp_gt[5]
        new_x = src.x[y_ind] - x_cell / 2
        new_y = src.y[x_ind] - y_cell / 2
        new_gt = (new_x, x_cell, 0, new_y, 0, y_cell)
        new_src = src.from_array(
            small_array,
            no_data_value=src.no_data_value,
            geo_ref=GeoReference(geo=new_gt),
        )
        # Preserve the source CRS from its WKT rather than round-tripping
        # through src.epsg: a custom CRS with no EPSG (e.g. a spherical-earth
        # GRIB GEOGCS) has no resolvable code, so passing epsg=src.epsg would
        # relabel — or, before issue #403 was fixed, crash on — the output.
        # Skip when the source is unprojected: setting an empty WKT would
        # wipe the from_array default, so leave that default in place.
        if src.crs:
            new_src.crs = src.crs
        return new_src

    def _crop_antimeridian(
        self,
        bbox: tuple[float, float, float, float],
        crs: Any,
        touch: bool,
    ) -> Dataset:
        """Crop with a geographic bbox whose ``west > east`` crosses the antimeridian.

        Splits the bbox at the grid's longitude seam (``180`` on a ``-180..180``
        grid, ``360`` on a ``0..360`` grid), crops each ``west < east`` half through
        the normal path, and concatenates the halves along longitude into one
        contiguous raster. A half that falls outside the dataset's longitude extent
        is skipped, so a single-sided overlap returns just that half.

        Args:
            bbox: ``(west, south, east, north)`` with ``west > east``.
            crs: The bbox CRS (defaults to the dataset's own upstream).
            touch: Forwarded to the per-half crop.

        Returns:
            Dataset: The cropped strip spanning the seam.

        Raises:
            ValueError: The bbox does not overlap the dataset's longitude extent.
        """
        # _crop_seam_halves is shared with the NetCDF Selection engine and returns
        # Any accordingly; here crop_half/merge_halves are both Dataset-returning,
        # so the result is a Dataset.
        return cast(
            "Dataset",
            _crop_seam_halves(
                self._ds,
                bbox,
                lambda half: self.crop(bbox=half, epsg=crs, touch=touch),
                self._merge_lon_halves,
            ),
        )

    def _merge_lon_halves(self, west_part: Dataset, east_part: Dataset) -> Dataset:
        """Concatenate two longitude-adjacent crops into one contiguous raster.

        `west_part` (the pre-seam half) sits to the left and `east_part` (the
        wrapped half past the seam) to its right; the merged raster keeps
        `west_part`'s north-up geotransform, so the linear longitude mapping simply
        continues past the seam (e.g. 170..180 then 180..190).

        Args:
            west_part: Crop of the pre-seam half.
            east_part: Crop of the post-seam half.

        Returns:
            Dataset: The concatenated raster.
        """
        return _stitch_lon_halves(self._ds, west_part, east_part)

    def _crop_from_bbox(
        self,
        bbox: tuple[float, float, float, float] | list[float],
        mask: GeoDataFrame | FeatureCollection | None,
        epsg: Any,
        touch: bool,
    ) -> Dataset | FeatureCollection:
        """Resolve a `crop(bbox=...)` request to a finished crop or a cutline mask.

        Handles the three bbox outcomes so :meth:`crop` stays flat: an
        antimeridian-crossing geographic box is split and cropped, an eligible
        axis-aligned box is cropped by the windowed fast path, and anything else is
        wrapped in a one-row :class:`FeatureCollection` for the cutline warp.

        Args:
            bbox (tuple[float, float, float, float] | list[float]):
                ``(west, south, east, north)`` in the CRS named by ``epsg``.
            mask (GeoDataFrame | FeatureCollection | None):
                The caller's ``mask`` argument, rejected here if also supplied.
            epsg (Any):
                CRS for ``bbox``; defaults to the dataset's own CRS when ``None``.
            touch (bool):
                Forwarded to the fast path / warp (``True`` enables the fast path).

        Returns:
            Dataset | FeatureCollection:
                A finished :class:`Dataset` crop (antimeridian or fast path), or a
                :class:`FeatureCollection` for the caller to warp.

        Raises:
            ValueError: Both ``mask`` and ``bbox`` were supplied.
        """
        if mask is not None:
            raise ValueError("crop accepts either `mask` or `bbox`, not both")
        # `.epsg` is None for a no-EPSG CRS (e.g. geostationary); fall back to the WKT
        # so a bbox in the grid's own CRS is still honoured (#706).
        crs = (
            epsg
            if epsg is not None
            else require_crs_spec(
                self._ds.epsg, self._ds.crs, "crop by a bbox without an explicit epsg="
            )
        )
        west, _, east, _ = bbox
        crs_geo = bool(crs) and sr_from_user_input(crs).IsGeographic()
        ds_epsg = self._ds.epsg
        ds_geo = ds_epsg is not None and sr_from_user_input(ds_epsg).IsGeographic()
        if west > east and crs_geo and ds_geo:
            # bbox is validated 4-long above; tuple(bbox) loses that fixed arity
            # statically (it may start as a list), so restore it.
            bbox_4 = cast(tuple[float, float, float, float], tuple(bbox))
            _require_antimeridian_seam(self._ds, bbox_4)
            return self._crop_antimeridian(bbox_4, crs, touch)
        # Fast path: an axis-aligned box in the source CRS is just a window, so read
        # only that window instead of warping a cutline over the whole source (#957).
        # Falls through to the warp path when it does not apply (reprojection, a rotated
        # grid, or a degenerate/off-grid box).
        if touch:
            windowed = self._crop_bbox_windowed(bbox, crs)
            if windowed is not None:
                return windowed
        return FeatureCollection.from_bbox(bbox, epsg=crs)

    def crop(
        self,
        mask: GeoDataFrame | FeatureCollection | None = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
    ) -> Dataset:
        """Crop dataset using a polygon mask, a raster mask, or a bbox tuple.

            Crop/Clip the Dataset object using a polygon/raster — or, as a
            convenience, a plain ``(west, south, east, north)`` bbox tuple
            in some EPSG (no need to wrap it in a :class:`FeatureCollection`
            by hand).

        Args:
            mask (GeoDataFrame | Dataset | None):
                GeoDataFrame with a polygon geometry, or a Dataset object.
                Mutually exclusive with ``bbox``; exactly one of the two must
                be supplied.
            touch (bool):
                Include the cells that touch the polygon, not only those that lie entirely inside the polygon mask.
                Default is True.
            bbox (tuple[float, float, float, float] | None, keyword-only):
                ``(west, south, east, north)`` quadruple in the CRS named by
                ``epsg``. Internally wrapped in a one-row
                :class:`FeatureCollection` and routed through the same polygon
                path. Mutually exclusive with ``mask``. A *geographic* bbox with
                ``west > east`` (the STAC convention for an antimeridian-crossing
                area, e.g. ``(170, -10, -170, 10)``) is split at the 180°/360°
                seam, each half cropped, and the halves stitched into one
                contiguous strip whose longitudes continue past the seam
                (``170..190``). Works for ``-180..180`` and ``0..360`` grids.
                Behaviour change: a *geographic* ``west > east`` bbox is read as
                the STAC antimeridian convention (rather than raising
                ``bbox must satisfy west < east``) — but only when the dataset's
                longitude extent actually reaches the 180 seam. On a *regional*
                grid that does not reach the seam (e.g. Europe, lon ``-10..40``) a
                ``west > east`` bbox cannot be a genuine crossing, so it raises a
                clear error instead of silently returning a truncated crop —
                catching a transposed / typo'd bbox. A *projected* ``west > east``
                bbox is still validated and raises, since the antimeridian has no
                meaning off a geographic CRS.
            epsg (Any, keyword-only):
                CRS for ``bbox`` — anything ``geopandas`` accepts for ``crs=``
                (EPSG int, ``"EPSG:4326"``, WKT, ``pyproj.CRS``). Defaults to
                the dataset's own CRS, so a bbox in the dataset's native CRS
                needs no extra argument; pass it explicitly for a bbox in a
                different CRS (the standard reprojection path takes care of it).

        Returns:
            Dataset:
                A new cropped Dataset.

        Hint:
            - If the mask is a dataset with multi-bands, the `crop` method will use the first band as the mask.

        Note:
            A ``bbox`` crop keeps **every pixel the box overlaps** (the all-touched
            convention :meth:`read_array` uses for ``bbox=``), which for a box not
            aligned to pixel edges can be up to one pixel wider per side than a
            polygon ``mask=`` crop (GDAL's centre-containment default). When the box
            is axis-aligned in the dataset's own CRS (no ``epsg`` reprojection, no
            antimeridian split, a north-up grid, ``touch=True``) it is cropped by
            reading only that pixel window straight from the source, so a small crop
            out of a very large or ``/vsicurl`` raster reads only the AOI instead of
            the full source. Ineligible cases (a reprojecting bbox, a rotated grid)
            fall back to the cutline warp, run all-touched so the result matches the
            windowed path; ``touch=False`` keeps the entirely-inside cutline crop.

        Examples:
            - Crop the raster using a polygon mask.

              - The polygon covers 4 cells in the 3rd and 4th rows and 3rd and 4th column `arr[2:4, 2:4]`, so the result
                dataset will have the same number of bands `4`, 2 rows and 2 columns.
              - First, create the dataset to have 4 bands, 10 rows and 10 columns; the dataset has a cell size of 0.05
                degree, the top left corner of the dataset is (0, 0).

              ```python
              >>> import numpy as np
              >>> import geopandas as gpd
              >>> from shapely.geometry import Polygon
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 10, 10)
              >>> cell_size = 0.05
              >>> top_left_corner = (0, 0)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```
            - Second, create the polygon using shapely polygon, and use the xmin, ymin, xmax, ymax = [0.1, -0.2, 0.2 -0.1]
                to cover the 4 cells.

                ```python
                >>> mask = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])], crs=4326)

                ```
            - Pass the `geodataframe` to the crop method using the `mask` parameter.

              ```python
              >>> cropped_dataset = dataset.crop(mask=mask)

              ```
            - Check the cropped dataset:

              ```python
              >>> print(cropped_dataset.shape)
              (4, 2, 2)
              >>> print(cropped_dataset.geotransform)
              (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> print(cropped_dataset.read_array(band=0))# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]
              >>> print(arr[0, 2:4, 2:4])# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]

              ```
            - Crop a raster using another raster mask:

              - Create a mask dataset with the same extent of the polygon we used in the previous example.

              ```python
              >>> geotransform = (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> mask_dataset = Dataset.from_array(
              ...     np.random.rand(2, 2),
              ...     geo_ref=GeoReference(geo=geotransform, epsg=4326),
              ... )

              ```
            - Then use the mask dataset to crop the dataset.

              ```python
              >>> cropped_dataset_2 = dataset.crop(mask=mask_dataset)
              >>> print(cropped_dataset_2.shape)
              (4, 2, 2)

              ```
            - Check the cropped dataset:

              ```python
              >>> print(cropped_dataset_2.geotransform)
              (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> print(cropped_dataset_2.read_array(band=0))# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]
              >>> print(arr[0, 2:4, 2:4])# doctest: +SKIP
               [[0.00921161 0.90841171]
               [0.355636   0.18650262]]

              ```

            - Crop using a ``(west, south, east, north)`` bbox tuple instead of
              a hand-built ``FeatureCollection`` (the bbox CRS defaults to the
              dataset's own):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr_int = np.arange(100, dtype="int16").reshape(10, 10)
              >>> dataset_bbox = Dataset.from_array(
              ...     arr_int,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> cropped_bbox = dataset_bbox.crop(bbox=(0.1, -0.2, 0.2, -0.1))
              >>> cropped_bbox.shape
              (1, 2, 2)
              >>> cropped_bbox.epsg
              4326

              ```

            - Crop across the antimeridian with a ``west > east`` geographic bbox
              (STAC convention); the two sides are stitched into one contiguous
              strip whose longitudes continue past the 180° seam:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> grid = Dataset.from_array(
              ...     np.arange(180 * 360, dtype="float32").reshape(180, 360),
              ...     geo_ref=GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326),
              ... )
              >>> strip = grid.crop(bbox=(170.0, -10.0, -170.0, 10.0))
              >>> strip.shape
              (1, 20, 20)
              >>> strip.bbox
              [170.0, -10.0, 190.0, 10.0]

              ```

            - Supplying both ``mask`` and ``bbox`` is rejected:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> from pyramids.feature import FeatureCollection
              >>> dataset_excl = Dataset.from_array(
              ...     np.zeros((4, 5), dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> fc = FeatureCollection.from_bbox((0.0, -0.1, 0.1, 0.0), epsg=4326)
              >>> try:
              ...     dataset_excl.crop(mask=fc, bbox=(0.0, -0.1, 0.1, 0.0))
              ... except ValueError as exc:
              ...     print("not both" in str(exc))
              True

              ```

        """
        bbox_mask = False
        if bbox is not None:
            # Resolve the bbox: a completed crop (antimeridian split or windowed fast
            # path) is returned as-is; otherwise it yields a FeatureCollection to warp.
            resolved = self._crop_from_bbox(bbox, mask, epsg, touch)
            if isinstance(resolved, RasterBase):
                return cast("Dataset", resolved)
            mask = resolved
            bbox_mask = True
        if mask is None:
            raise TypeError(
                "crop requires a `mask` (GeoDataFrame / FeatureCollection / "
                "Dataset) or a `bbox` (west, south, east, north) tuple"
            )
        if isinstance(mask, GeoDataFrame):
            # A bbox that fell through the fast path (reprojection / rotated grid) still
            # crops by overlap, so its warp uses the all-touched cutline to match the fast
            # path; a user-supplied polygon mask keeps GDAL's centre-containment default.
            dst = self._crop_with_polygon_warp(
                mask, touch=touch, cutline_all_touched=bbox_mask
            )
        elif isinstance(mask, RasterBase):
            dst = self._crop_with_raster(mask)
        else:
            raise TypeError(
                "The second parameter: mask could be either GeoDataFrame or Dataset object"
            )

        return dst
