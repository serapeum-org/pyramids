"""Georeferencing engine: ground-control points and rational-polynomial coefficients.

Accessed as ``ds.georef``; the Dataset exposes same-named facades (``ds.gcps``,
``ds.set_gcps``, ``ds.georeference``, ``ds.rpcs``, ``ds.set_rpcs``,
``ds.orthorectify``). Everything routes through GDAL — pyramids stays a generic
GDAL toolkit and does not implement any sensor model itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pyramids.base._errors import ReadOnlyError
from pyramids.base.crs import sr_from_user_input
from pyramids.dataset._gcp import GroundControlPoint
from pyramids.dataset.engines._base import _Engine

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


class Georef(_Engine):
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
        return self._ds.raster.GetGCPCount()

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
        return self._ds.raster.GetGCPCount() > 0

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
