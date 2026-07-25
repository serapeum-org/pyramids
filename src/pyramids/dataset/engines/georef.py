"""Georeferencing engine: ground-control points and rational-polynomial coefficients.

Accessed as ``ds.georef``; the Dataset exposes same-named facades (``ds.gcps``,
``ds.set_gcps``, ``ds.georeference``, ``ds.rpcs``, ``ds.set_rpcs``,
``ds.orthorectify``). Everything routes through GDAL — pyramids stays a generic
GDAL toolkit and does not implement any sensor model itself.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.base._utils import DEFAULT_RESAMPLING, resolve_resampling
from pyramids.base.crs import sr_from_user_input
from pyramids.dataset._gcp import GroundControlPoint
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._warp import dst_srs_arg as _dst_srs_arg

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

logger = logging.getLogger(__name__)

# The offsets, scales, and four coefficient vectors every RPC sensor model needs.
_REQUIRED_RPC_KEYS: frozenset[str] = frozenset(
    {
        "LINE_OFF",
        "SAMP_OFF",
        "LAT_OFF",
        "LONG_OFF",
        "HEIGHT_OFF",
        "LINE_SCALE",
        "SAMP_SCALE",
        "LAT_SCALE",
        "LONG_SCALE",
        "HEIGHT_SCALE",
        "LINE_NUM_COEFF",
        "LINE_DEN_COEFF",
        "SAMP_NUM_COEFF",
        "SAMP_DEN_COEFF",
    }
)


class Georef(_Engine["Dataset"]):
    """Ground-control-point and RPC georeferencing for a :class:`Dataset`.

    A normal raster is georeferenced by an affine geotransform. Raw imagery
    (scanned maps, drone mosaics, un-orthorectified satellite scenes) instead
    carries **ground-control points** (pixel↔map tie points) or **rational
    polynomial coefficients** (a vendor sensor model). This engine reads and
    attaches both, and warps from them into an affine-geotransform raster.
    """

    @property
    def gcps(self: Georef) -> list[GroundControlPoint]:
        """The dataset's ground-control points (empty list when it has none).

        Returns:
            list[GroundControlPoint]: one per attached GCP, in GDAL order.

        Examples:
            - Read back the points attached with :meth:`set_gcps`:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.dataset._gcp import GroundControlPoint
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
                ... )
                >>> ds.set_gcps([GroundControlPoint(row=0, col=0, x=10.0, y=50.0)], 4326)
                >>> (ds.gcps[0].col, ds.gcps[0].x, ds.gcps[0].y)
                (0.0, 10.0, 50.0)

                ```
        """
        return [GroundControlPoint.from_gdal(gcp) for gcp in self._ds.raster.GetGCPs()]

    @property
    def gcp_count(self: Georef) -> int:
        """Number of ground-control points attached (``0`` when none).

        Returns:
            int: the GCP count.
        """
        return cast(int, self._ds.raster.GetGCPCount())

    @property
    def gcp_projection(self: Georef) -> str | None:
        """WKT of the GCPs' CRS, or ``None`` when the dataset has no GCPs.

        Returns:
            str | None: the GCP-projection WKT, else ``None``.
        """
        wkt = self._ds.raster.GetGCPProjection()
        return wkt if self._ds.raster.GetGCPCount() else None

    @property
    def has_gcps(self: Georef) -> bool:
        """``True`` when the dataset carries at least one ground-control point.

        Returns:
            bool: whether any GCP is attached.
        """
        return cast(int, self._ds.raster.GetGCPCount()) > 0

    @property
    def rpcs(self: Georef) -> dict[str, str] | None:
        """The dataset's rational-polynomial coefficients, or ``None`` if absent.

        RPCs live in GDAL's ``"RPC"`` metadata domain — a vendor sensor model of
        ~90 string coefficients (``LINE_NUM_COEFF``, ``HEIGHT_OFF``, ...) shipped
        with raw high-resolution satellite imagery.

        Returns:
            dict[str, str] | None: the RPC coefficient mapping, or ``None`` when
            the dataset has no RPC metadata.

        Examples:
            - A plain raster has no RPCs:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
                ... )
                >>> ds.rpcs is None
                True

                ```
        """
        metadata = self._ds.raster.GetMetadata("RPC")
        return metadata or None

    @property
    def has_rpcs(self: Georef) -> bool:
        """``True`` when the dataset carries RPC metadata.

        Returns:
            bool: whether an RPC sensor model is attached.
        """
        return bool(self._ds.raster.GetMetadata("RPC"))

    def set_rpcs(self: Georef, rpc: Mapping[str, str | float]) -> None:
        """Attach rational-polynomial coefficients (an RPC sensor model).

        Values are stringified before being written to GDAL's ``"RPC"`` metadata
        domain. The dataset must be writable.

        Args:
            rpc: The RPC coefficient mapping. Must contain every required key:
                the five ``*_OFF`` offsets, the five ``*_SCALE`` scales, and the
                four coefficient vectors (``LINE_NUM_COEFF``, ``LINE_DEN_COEFF``,
                ``SAMP_NUM_COEFF``, ``SAMP_DEN_COEFF``).

        Raises:
            ReadOnlyError: The dataset is opened read-only.
            ValueError: One or more required RPC keys are missing (the error
                lists them).

        Examples:
            - Round-trip a coefficient set through the RPC domain:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> rpc = {k: "0" for k in (
                ...     "LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF",
                ...     "LINE_SCALE", "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE",
                ...     "HEIGHT_SCALE", "LINE_NUM_COEFF", "LINE_DEN_COEFF",
                ...     "SAMP_NUM_COEFF", "SAMP_DEN_COEFF",
                ... )}
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
                ... )
                >>> ds.set_rpcs(rpc)
                >>> ds.rpcs["HEIGHT_OFF"]
                '0'

                ```
        """
        if self._ds.access == "read_only":
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                "read_only=False to attach RPC metadata."
            )
        missing = _REQUIRED_RPC_KEYS - set(rpc)
        if missing:
            raise ValueError(
                f"RPC metadata is missing required keys: {sorted(missing)}"
            )
        stringified = {key: str(value) for key, value in rpc.items()}
        self._ds.raster.SetMetadata(stringified, "RPC")

    def orthorectify(
        self: Georef,
        *,
        dem: str | Path | Dataset | None = None,
        to_epsg: int | str | None = None,
        method: str = "bilinear",
        rpc_height: float | None = None,
        cell_size: float | None = None,
        lazy: bool = False,
    ) -> Dataset:
        """Orthorectify the dataset from its RPC sensor model onto a map grid.

        Uses the attached rational-polynomial coefficients (attach them first
        with :meth:`set_rpcs`) and, ideally, a digital elevation model to remove
        terrain-induced distortion and produce a map-projected raster.

        Args:
            dem: Elevation model used to evaluate the RPCs — a raster path or a
                :class:`Dataset` (an in-memory dataset is staged to ``/vsimem/``).
                ``None`` falls back to a constant height.
            to_epsg: Target CRS. ``None`` keeps the RPCs' native geographic CRS.
            method: Resampling method. Default ``"bilinear"``.
            rpc_height: Constant elevation (map units) to use when no ``dem`` is
                given. If both ``dem`` and ``rpc_height`` are ``None`` GDAL uses
                height ``0`` and a ``logging.WARNING`` is emitted.
            cell_size: Output pixel size in target-CRS units. ``None`` lets GDAL
                pick.
            lazy: ``True`` returns a VRT-backed view; ``False`` (default)
                materialises the result.

        Returns:
            Dataset: the orthorectified raster.

        Raises:
            ValueError: the dataset has no RPC metadata.
            RuntimeError: GDAL could not build the warp.
        """
        if not self._ds.raster.GetMetadata("RPC"):
            raise ValueError("dataset has no RPC metadata; call set_rpcs first.")
        transformer_options = ["METHOD=RPC"]
        dem_path = self._resolve_dem_path(dem)
        if dem_path is not None:
            transformer_options.append(f"RPC_DEM={dem_path}")
        elif rpc_height is not None:
            transformer_options.append(f"RPC_HEIGHT={rpc_height}")
        else:
            logger.warning(
                "orthorectify: no DEM and no rpc_height given; GDAL will use "
                "height 0, which is rarely correct over real terrain."
            )
        warp_kwargs: dict = {
            "format": "VRT" if lazy else "MEM",
            "resampleAlg": resolve_resampling(method),
            "xRes": cell_size,
            "yRes": cell_size,
            "transformerOptions": transformer_options,
        }
        if to_epsg is not None:
            warp_kwargs["dstSRS"] = _dst_srs_arg(sr_from_user_input(to_epsg))
        # Only a DEM staged by this call is ours to free; a caller-supplied path
        # must be left alone.
        staged_dem = dem_path is not None and dem_path.startswith(
            "/vsimem/orthorectify_dem_"
        )
        try:
            dst = gdal.Warp(
                "", self._ds.raster, options=gdal.WarpOptions(**warp_kwargs)
            )
            if dst is None:
                raise RuntimeError("GDAL could not orthorectify the dataset.")
            result = self._ds.__class__(dst, access="read_only")
        except BaseException:
            # The failure paths returned without unlinking, leaving the staged
            # copy in /vsimem for the lifetime of the process.
            if staged_dem:
                gdal.Unlink(dem_path)
            raise

        if lazy:
            result._warp_source = self._ds.raster
            if staged_dem:
                # A lazy result reads the DEM on every access, so it cannot be
                # freed here -- but nothing freed it later either, so it leaked.
                # Tie its lifetime to the result instead.
                weakref.finalize(result, gdal.Unlink, dem_path)
        elif staged_dem:
            # A materialised result no longer references the staged DEM, so free
            # the /vsimem copy we made from a MEM Dataset.
            gdal.Unlink(dem_path)
        return result

    @staticmethod
    def _resolve_dem_path(dem: str | Path | Dataset | None) -> str | None:
        """Resolve a DEM argument to a GDAL-openable path.

        Args:
            dem: ``None``, a filesystem path, or a Dataset. A file-backed Dataset
                yields its path; an in-memory one is staged to ``/vsimem/``.

        Returns:
            str | None: the path GDAL's ``RPC_DEM`` should open, or ``None``.
        """
        if dem is None:
            result = None
        elif isinstance(dem, (str, Path)):
            result = str(dem)
        else:
            description = dem.raster.GetDescription()
            if description:
                result = description
            else:
                result = f"/vsimem/orthorectify_dem_{uuid4().hex}.tif"
                gdal.Translate(result, dem.raster)
        return result

    def set_gcps(
        self: Georef,
        gcps: Sequence[GroundControlPoint],
        projection: int | str,
    ) -> None:
        """Attach ground-control points (and their CRS) to the dataset.

        Replaces any existing GCPs. The dataset must be opened writable
        (``read_only=False``); a MEM-backed dataset (e.g. from
        :meth:`Dataset.create_from_array`) is always writable.

        Args:
            gcps: One or more :class:`GroundControlPoint` tie points.
            projection: The GCPs' CRS, in any form
                :func:`pyramids.base.crs.sr_from_user_input` accepts (EPSG int,
                ``"EPSG:4326"``, WKT, PROJ4, ...).

        Raises:
            ReadOnlyError: The dataset is opened read-only.
            ValueError: ``gcps`` is empty.

        Examples:
            - Attach four corner points in EPSG:4326 to an in-memory raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.dataset._gcp import GroundControlPoint
                >>> ds = Dataset.create_from_array(
                ...     np.ones((8, 8), "float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
                ... )
                >>> pts = [
                ...     GroundControlPoint(row=0, col=0, x=10.0, y=50.0),
                ...     GroundControlPoint(row=0, col=8, x=11.0, y=50.0),
                ...     GroundControlPoint(row=8, col=0, x=10.0, y=49.0),
                ...     GroundControlPoint(row=8, col=8, x=11.0, y=49.0),
                ... ]
                >>> ds.set_gcps(pts, 4326)
                >>> ds.raster.GetGCPCount()
                4

                ```
        """
        if self._ds.access == "read_only":
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                "read_only=False to attach GCPs."
            )
        gcp_list = list(gcps)
        if not gcp_list:
            raise ValueError("set_gcps requires at least one GroundControlPoint.")
        wkt = sr_from_user_input(projection).ExportToWkt()
        self._ds.raster.SetGCPs([point.to_gdal() for point in gcp_list], wkt)

    def georeference(
        self: Georef,
        *,
        to_epsg: int | str | None = None,
        method: str = DEFAULT_RESAMPLING,
        transform: str = "polynomial",
        order: int = 1,
        cell_size: float | None = None,
        lazy: bool = False,
    ) -> Dataset:
        """Warp the dataset **from its GCPs** into an affine-geotransform raster.

        Fits a transform through the attached ground-control points and resamples
        the pixels onto a regular grid — the step that turns a GCP-tagged scan or
        mosaic into a normal georeferenced raster. The GCPs are read from the
        source automatically (attach them first with :meth:`set_gcps`).

        Args:
            to_epsg: Target CRS for the output. ``None`` warps into the GCPs' own
                CRS; otherwise reprojects to ``to_epsg`` in the same pass (any
                form :func:`pyramids.base.crs.sr_from_user_input` accepts).
            method: Resampling method (see :meth:`Spatial.to_crs`). Default
                ``"nearest neighbor"``.
            transform: ``"polynomial"`` (default) fits a polynomial of degree
                ``order``; ``"tps"`` fits a thin-plate spline (good for many,
                irregularly-spaced points / local warps).
            order: Polynomial degree, one of ``1``/``2``/``3``. Ignored when
                ``transform="tps"``.
            cell_size: Output pixel size in target-CRS units (both axes). ``None``
                lets GDAL pick a size that preserves the source resolution.
            lazy: ``True`` returns a VRT-backed view (pixels warped per read);
                ``False`` (default) materialises the result in memory.

        Returns:
            Dataset: the georeferenced raster.

        Raises:
            ValueError: the dataset has no GCPs, or ``transform``/``order`` is
                invalid.
            RuntimeError: GDAL could not build the warp.

        Examples:
            - Georeference an 8x8 image from four corner GCPs in EPSG:4326:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from pyramids.dataset._gcp import GroundControlPoint
                >>> ds = Dataset.create_from_array(
                ...     np.ones((8, 8), "float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
                ... )
                >>> ds.set_gcps([
                ...     GroundControlPoint(row=0, col=0, x=10.0, y=50.0),
                ...     GroundControlPoint(row=0, col=8, x=11.0, y=50.0),
                ...     GroundControlPoint(row=8, col=0, x=10.0, y=49.0),
                ...     GroundControlPoint(row=8, col=8, x=11.0, y=49.0),
                ... ], 4326)
                >>> out = ds.georeference()
                >>> out.epsg
                4326

                ```
        """
        if self._ds.raster.GetGCPCount() == 0:
            raise ValueError("dataset has no GCPs; call set_gcps first.")
        if transform not in {"polynomial", "tps"}:
            raise ValueError(
                f"transform must be 'polynomial' or 'tps', got {transform!r}."
            )
        if transform == "polynomial" and order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3, got {order!r}.")
        # Force the GCP transformer: GDAL otherwise prefers an affine geotransform
        # when the source carries one, ignoring the GCPs. METHOD=GCP_* makes the
        # warp fit the points regardless.
        if transform == "tps":
            transformer_options = ["METHOD=GCP_TPS"]
        else:
            transformer_options = ["METHOD=GCP_POLYNOMIAL", f"MAX_GCP_ORDER={order}"]
        warp_kwargs: dict = {
            "format": "VRT" if lazy else "MEM",
            "resampleAlg": resolve_resampling(method),
            "xRes": cell_size,
            "yRes": cell_size,
            "transformerOptions": transformer_options,
        }
        if to_epsg is not None:
            warp_kwargs["dstSRS"] = _dst_srs_arg(sr_from_user_input(to_epsg))
        dst = gdal.Warp("", self._ds.raster, options=gdal.WarpOptions(**warp_kwargs))
        if dst is None:
            raise RuntimeError("GDAL could not warp the dataset from its GCPs.")
        result = self._ds.__class__(dst, access="read_only")
        if lazy:
            # The VRT references the source GDAL handle; pin the source Dataset so
            # it cannot be garbage-collected underneath the view.
            result._warp_source = self._ds.raster
        return result
