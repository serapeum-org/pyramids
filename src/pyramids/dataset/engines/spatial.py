"""Spatial engine.

Owns the Spatial family of operations on a Dataset. Accessed as
``ds.spatial``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.spatial.<method>(...)`` are equivalent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload
from xml.sax.saxutils import escape  # nosec B406 - output escaping only

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr

from pyramids.base._domain import is_no_data
from pyramids.base._utils import DEFAULT_RESAMPLING, resolve_resampling
from pyramids.base.crs import (
    epsg_from_wkt,
    reproject_coordinates,
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

from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines.vectorize import Vectorize


def _dst_srs_arg(dst_sr: osr.SpatialReference) -> str:
    """Derive the ``dstSRS`` argument to hand to :func:`gdal.Warp`.

    Prefer the ``"<AUTHORITY>:<code>"`` form when one exists so the output WKT
    GDAL writes is the canonical GDAL/PROJ form (matching historical bytes for
    EPSG codes and avoiding a GDAL warning when the authority is ESRI). Fall
    back to the explicit WKT for CRSes carrying no authority at all (custom
    orthographic proj4 strings, etc.). See #418.

    Args:
        dst_sr: The target spatial reference.

    Returns:
        str: An authority string such as ``"EPSG:3857"``, or the full WKT when
            the SRS carries no authority.
    """
    dst_auth = dst_sr.GetAuthorityName(None)
    dst_code = dst_sr.GetAuthorityCode(None)
    if dst_auth is not None and dst_code is not None:
        srs_arg = f"{dst_auth}:{dst_code}"
    else:
        srs_arg = dst_sr.ExportToWkt()
    return srs_arg


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
    # plain raster Dataset (create_from_array on a variable view would build a NetCDF
    # container).
    from pyramids.dataset.dataset import Dataset

    _check_lon_halves_concatenable(west_part, east_part)
    merged = np.concatenate([west_part.read_array(), east_part.read_array()], axis=-1)
    out = Dataset.create_from_array(
        merged,
        geo=west_part.geotransform,
        # epsg is None only for a no-EPSG CRS reported as such (a NetCDF
        # geostationary grid); create_from_array raises CRSError on None, so fall
        # back to the WKT. No-op for a plain Dataset (reports 4326) (#706).
        epsg=west_part.epsg or west_part.crs,
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
            # fallback to 4326 when crs is an empty string
            # (get_epsg_from_prj raises in that case); epsg_from_wkt
            # absorbs the fallback in one place.
            self._ds._epsg = epsg_from_wkt(crs)
        elif epsg is not None:
            sr = sr_from_epsg(epsg)
            self._ds.raster.SetProjection(sr.ExportToWkt())
            self._ds._epsg = epsg

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
                Note: the aggregating algorithms ("average", "mode", "med", "q1", "q3", "sum", "rms")
                are not no-data-aware on this warp path — no-data cells inside a resampling kernel are
                mixed into the result. Prefer "nearest" on rasters that carry a no-data marker.
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
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(4, 5, 5)
              >>> dataset = Dataset.create_from_array(
              ...     arr,
              ...     top_left_corner=(0.0, 0.0),
              ...     cell_size=0.05,
              ...     epsg=4326,
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
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326
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
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326
              ... )
              >>> proj4 = "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"
              >>> ortho = dataset.to_crs(to_epsg=proj4)
              >>> osr.SpatialReference(wkt=ortho.crs).IsProjected()
              1
              >>> ortho.epsg
              4326

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
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(10.0, 60.5), cell_size=0.1, epsg=4326
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
            dst = gdal.Warp(
                "",
                self._ds.raster,
                dstSRS=_dst_srs_arg(dst_sr),
                format="VRT",
                resampleAlg=resampling_method,
                xRes=x_res,
                yRes=y_res,
            )
            dst_obj = self._ds.__class__(dst)

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
                >>> from pyramids.dataset import Dataset
                >>> src = Dataset.create_from_array(
                ...     np.random.rand(8, 8).astype("float32"),
                ...     top_left_corner=(0, 8), cell_size=0.01, epsg=4326,
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
                >>> src = Dataset.create_from_array(
                ...     np.ones((4, 4), dtype="float32"),
                ...     top_left_corner=(0, 4), cell_size=0.01, epsg=4326,
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
        vrt = gdal.Warp("", self._ds.raster, options=options)
        if vrt is None:
            raise RuntimeError(
                f"GDAL could not build a warped VRT onto {dst_srs_arg!r}."
            )
        view = self._ds.__class__(vrt, access="read_only")
        # The VRT references the source GDAL handle; pin the source Dataset on
        # the view so Python cannot garbage-collect it underneath the VRT.
        view._warp_source = self._ds
        return view

    def _get_epsg(self) -> int | None:
        """Get the EPSG number.

            This function reads the projection of a GEOGCS file or tiff file.

        Returns:
            int: EPSG number.
        """
        prj = self._get_crs()
        # get_epsg_from_prj raises on empty input; epsg_from_wkt
        # absorbs the historical 4326 fallback for datasets without a
        # projection.
        epsg = epsg_from_wkt(prj)

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
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(360, dtype=np.float32).reshape(1, 360)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0.0, 0.5), cell_size=1.0, epsg=4326,
                ...     no_data_value=-9999.0,
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
                >>> ds = Dataset.create_from_array(
                ...     np.ones((3, 3), dtype=np.float32), top_left_corner=(0.0, 0.0),
                ...     cell_size=0.05, epsg=4326, no_data_value=-9999.0,
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
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(4, 10, 10)
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
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
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((5, 5), dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     arr,
                ...     top_left_corner=(10.0, 50.0),
                ...     cell_size=0.5,
                ...     epsg=4326,
                ...     no_data_value=-9999.0,
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
                >>> ds = Dataset.create_from_array(
                ...     arr,
                ...     top_left_corner=(10.0, 60.5),
                ...     cell_size=0.1,
                ...     epsg=4326,
                ...     no_data_value=-9999.0,
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
        self, src_array: np.ndarray, mask_no_data: np.ndarray, band_count: int
    ) -> None:
        """Write the source no-data value into the masked cells (per band)."""
        if band_count > 1:
            # check the no_data_value complies with the src dtype before writing it
            # into cells (a band full of values may never use its no_data_value).
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
            # read_array() is called with no chunks=, so it always returns a plain
            # ndarray here (the dask.Array arm of ArrayLike is unreachable).
            mask_array = cast(np.typing.NDArray, mask.read_array(band=0))
        elif isinstance(mask, np.ndarray):
            if mask_noval is None:
                raise ValueError(
                    "You have to enter the value of the no_val parameter when the mask is a numpy array"
                )
            mask_array = mask.copy()
            row, col = mask.shape
        else:
            raise TypeError(
                "The second parameter 'mask' has to be either gdal.Dataset or numpy array"
                f"given - {type(mask)}"
            )

        band_count = self._ds.band_count
        src_sref = sr_from_wkt(self._ds.crs)
        # read_array() is called with no chunks=, so it always returns a plain
        # ndarray here (the dask.Array arm of ArrayLike is unreachable).
        src_array = cast(np.typing.NDArray, self._ds.read_array())

        self._assert_crop_aligned(mask, row, col)

        mask_no_data = is_no_data(mask_array, mask_noval)
        self._apply_mask_nodata(src_array, mask_no_data, band_count)

        if fill_gaps:
            src_array = self.fill_gaps(mask, src_array)

        dst = self._ds.__class__._create_dataset(
            col, row, band_count, self._ds.gdal_dtype[0], driver="MEM"
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
        self._write_bands(dst_obj, src_array, band_count)
        return dst_obj

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster."""
        if not isinstance(mask, RasterBase):
            raise TypeError("The second parameter should be a Dataset")

        return self._ds.rows == mask.rows and self._ds.columns == mask.columns

    def align(
        self,
        alignment_src: Dataset,
    ) -> Dataset:
        """Align the current dataset (rows and columns) to match a given dataset.

        Copies spatial properties from alignment_src to the current raster:
            - The coordinate system
            - The number of rows and columns
            - Cell size
        Then resamples values from the current dataset using the nearest neighbor interpolation.

        Args:
            alignment_src (Dataset):
                Spatial information source raster to get the spatial information (coordinate system, number of rows and
                columns). The data values of the current dataset are resampled to this alignment.

        Returns:
            Dataset: A new aligned Dataset.

        Examples:
            - The source dataset has a `top_left_corner` at (0, 0) with a 5*5 alignment, and a 0.05 degree cell size.

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(5, 5)
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
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
              >>> dataset_target = Dataset.create_from_array(
              ...     arr_target, top_left_corner=(-0.1, 0.1), cell_size=0.07, epsg=4326
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
              >>> source = Dataset.create_from_array(
              ...     np.random.rand(5, 5),
              ...     top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
              ... )
              >>> target = Dataset.create_from_array(
              ...     np.random.rand(10, 10),
              ...     top_left_corner=(-0.1, 0.1), cell_size=0.07, epsg=4326,
              ... )
              >>> aligned = target.align(source)
              >>> (aligned.rows, aligned.columns, aligned.geotransform[1], aligned.epsg)
              (5, 5, 0.05, 4326)

              ```

            ![align-result](./../../_images/dataset/align-result.png)
        """
        if isinstance(alignment_src, RasterBase):
            src = alignment_src
        else:
            raise TypeError(
                "First parameter should be a Dataset read using Dataset.openRaster or a path to the raster, "
                f"given {type(alignment_src)}"
            )

        # reproject the raster to match the projection of alignment_src
        reprojected_raster_b: Dataset = self._ds
        if self._ds.epsg != src.epsg:
            reprojected_raster_b = self.to_crs(src.epsg or src.crs)  # type: ignore[assignment]
        dst_obj = self._ds.__class__._build_dataset(
            src.columns,
            src.rows,
            self._ds.band_count,
            src.gdal_dtype[0],
            src.geotransform,
            src.crs,
            self._ds.no_data_value,
        )
        method = gdal.GRA_NearestNeighbour
        # resample the reprojected_RasterB
        gdal.ReprojectImage(
            reprojected_raster_b.raster,
            dst_obj.raster,
            src.crs,
            src.crs,
            method,
        )

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

        dst_obj = Spatial._correct_wrap_cutline_error(dst_obj)
        return dst_obj

    def _crop_with_polygon_warp(
        self, feature: FeatureCollection | GeoDataFrame, touch: bool = True
    ) -> Dataset:
        """Crop raster with polygon.

            - Do not convert the polygon into a raster but rather use it directly to crop the raster using the
            gdal.warp function.

        Args:
            feature (FeatureCollection | GeoDataFrame):
                Vector mask.
            touch (bool):
                Include cells that touch the polygon, not only those entirely inside the polygon mask. Defaults to True.

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
        # Use the base Dataset class (not a subclass like NetCDF) for intermediate GDAL warp results
        # because _correct_wrap_cutline_error calls create_from_array which has different behavior in
        # subclasses.
        base_cls = next(
            c
            for c in self._ds.__class__.__mro__
            if RasterBase in getattr(c, "__bases__", ())
        )

        # The warp output (VRT) may resolve the cutline lazily, so we must
        # complete every access that could touch the cutline path inside
        # the with-block that keeps that path alive.
        with _feature_ogr.as_vsimem_path(feature) as cutline_path:
            warp_options = gdal.WarpOptions(
                format="VRT",
                cropToCutline=not touch,
                cutlineDSName=cutline_path,
                multithread=True,
            )
            dst = gdal.Warp("", self._ds.raster, options=warp_options)
            # base_cls is a dynamic MRO walk that always resolves to Dataset itself
            # (the class directly above RasterBase; see the comment above), never a
            # subclass, so this is guaranteed to actually be a Dataset at runtime.
            dst_obj = cast("Dataset", base_cls(dst))
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
        the rebuilt dataset keeps the :meth:`Dataset.create_from_array`
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
            The CRS is the source CRS, or the ``create_from_array`` default
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
        # Find rows and columns to be removed
        if big_array.ndim == 2:
            rows_to_remove = np.all(big_array == value_to_remove, axis=1)
            cols_to_remove = np.all(big_array == value_to_remove, axis=0)
            # Use boolean indexing to remove rows and columns
            small_array = big_array[~rows_to_remove][:, ~cols_to_remove]
        elif big_array.ndim == 3:
            rows_to_remove = np.all(big_array == value_to_remove, axis=(0, 2))
            cols_to_remove = np.all(big_array == value_to_remove, axis=(0, 1))
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
        new_src = src.create_from_array(
            small_array, geo=new_gt, no_data_value=src.no_data_value
        )
        # Preserve the source CRS from its WKT rather than round-tripping
        # through src.epsg: a custom CRS with no EPSG (e.g. a spherical-earth
        # GRIB GEOGCS) has no resolvable code, so passing epsg=src.epsg would
        # relabel — or, before issue #403 was fixed, crash on — the output.
        # Skip when the source is unprojected: setting an empty WKT would
        # wipe the create_from_array default, so leave that default in place.
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
              >>> from pyramids.dataset import Dataset
              >>> arr = np.random.rand(4, 10, 10)
              >>> cell_size = 0.05
              >>> top_left_corner = (0, 0)
              >>> dataset = Dataset.create_from_array(
              ...         arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
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
              >>> mask_dataset = Dataset.create_from_array(np.random.rand(2, 2), geo=geotransform, epsg=4326)

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
              >>> dataset_bbox = Dataset.create_from_array(
              ...     arr_int, top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
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
              >>> grid = Dataset.create_from_array(
              ...     np.arange(180 * 360, dtype="float32").reshape(180, 360),
              ...     top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326,
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
              >>> dataset_excl = Dataset.create_from_array(
              ...     np.zeros((4, 5), dtype="int16"),
              ...     top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
              ... )
              >>> fc = FeatureCollection.from_bbox((0.0, -0.1, 0.1, 0.0), epsg=4326)
              >>> try:
              ...     dataset_excl.crop(mask=fc, bbox=(0.0, -0.1, 0.1, 0.0))
              ... except ValueError as exc:
              ...     print("not both" in str(exc))
              True

              ```

        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            # `.epsg` is None for a no-EPSG CRS (e.g. geostationary); fall back to
            # the WKT so a bbox in the grid's own CRS is still honoured (#706).
            crs = epsg if epsg is not None else (self._ds.epsg or self._ds.crs)
            west, _, east, _ = bbox
            crs_geo = bool(crs) and sr_from_user_input(crs).IsGeographic()
            ds_epsg = self._ds.epsg
            ds_geo = ds_epsg is not None and sr_from_user_input(ds_epsg).IsGeographic()
            if west > east and crs_geo and ds_geo:
                # bbox is validated 4-long above; tuple(bbox) loses that fixed
                # arity statically (it may start as a list), so restore it.
                bbox_4 = cast(tuple[float, float, float, float], tuple(bbox))
                _require_antimeridian_seam(self._ds, bbox_4)
                return self._crop_antimeridian(bbox_4, crs, touch)
            mask = FeatureCollection.from_bbox(bbox, epsg=crs)
        if mask is None:
            raise TypeError(
                "crop requires a `mask` (GeoDataFrame / FeatureCollection / "
                "Dataset) or a `bbox` (west, south, east, north) tuple"
            )
        if isinstance(mask, GeoDataFrame):
            dst = self._crop_with_polygon_warp(mask, touch=touch)
        elif isinstance(mask, RasterBase):
            dst = self._crop_with_raster(mask)
        else:
            raise TypeError(
                "The second parameter: mask could be either GeoDataFrame or Dataset object"
            )

        return dst
