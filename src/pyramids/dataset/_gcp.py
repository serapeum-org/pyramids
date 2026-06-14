"""Ground-control-point value object for georeferencing rasters.

A :class:`GroundControlPoint` ties one raster pixel to one map coordinate. A set
of them (with a CRS) lets GDAL fit a polynomial / thin-plate-spline transform and
warp an otherwise-ungeoreferenced raster onto a real grid — see
:meth:`pyramids.dataset.Dataset.georeference`.
"""

from __future__ import annotations

from dataclasses import dataclass

from osgeo import gdal


@dataclass(frozen=True)
class GroundControlPoint:
    """An immutable ground-control point: a pixel mapped to a map coordinate.

    A GCP states "pixel ``(col, row)`` is at map coordinate ``(x, y[, z])``". The
    pixel axes follow GDAL: ``col`` is the pixel (x) index and ``row`` is the line
    (y) index, both measured from the raster's top-left corner.

    Args:
        row: Pixel-space line (y / row) index of the control point.
        col: Pixel-space pixel (x / column) index of the control point.
        x: Map-space X (easting / longitude) at that pixel.
        y: Map-space Y (northing / latitude) at that pixel.
        z: Map-space Z (elevation). Defaults to ``0.0``.
        id: Optional point identifier. Defaults to ``None``.
        info: Optional free-text label. Defaults to ``None``.

    Examples:
        - Build a point and read its fields:
            ```python
            >>> from pyramids.dataset._gcp import GroundControlPoint
            >>> gcp = GroundControlPoint(row=0.0, col=0.0, x=10.0, y=50.0)
            >>> (gcp.col, gcp.row, gcp.x, gcp.y, gcp.z)
            (0.0, 0.0, 10.0, 50.0, 0.0)

            ```
        - Round-trip through the GDAL representation:
            ```python
            >>> from pyramids.dataset._gcp import GroundControlPoint
            >>> gcp = GroundControlPoint(row=4.0, col=2.0, x=11.5, y=46.2, id="tl")
            >>> back = GroundControlPoint.from_gdal(gcp.to_gdal())
            >>> (back.col, back.row, back.x, back.y, back.id)
            (2.0, 4.0, 11.5, 46.2, 'tl')

            ```
    """

    row: float
    col: float
    x: float
    y: float
    z: float = 0.0
    id: str | None = None
    info: str | None = None

    def to_gdal(self) -> gdal.GCP:
        """Convert to an :class:`osgeo.gdal.GCP`.

        Returns:
            gdal.GCP: a GDAL GCP whose ``pixel``/``line`` are this point's
            ``col``/``row`` and whose ``(x, y, z)`` are the map coordinate.

        Examples:
            - The GDAL GCP carries the pixel and map coordinates:
                ```python
                >>> from pyramids.dataset._gcp import GroundControlPoint
                >>> g = GroundControlPoint(row=3.0, col=7.0, x=1.0, y=2.0).to_gdal()
                >>> (g.GCPPixel, g.GCPLine, g.GCPX, g.GCPY)
                (7.0, 3.0, 1.0, 2.0)

                ```
        """
        return gdal.GCP(
            self.x, self.y, self.z, self.col, self.row, self.info or "", self.id or ""
        )

    @classmethod
    def from_gdal(cls, gcp: gdal.GCP) -> GroundControlPoint:
        """Build a :class:`GroundControlPoint` from an :class:`osgeo.gdal.GCP`.

        Args:
            gcp: A GDAL GCP (fields ``GCPPixel``/``GCPLine``/``GCPX``/``GCPY``/
                ``GCPZ``/``Id``/``Info``). Empty ``Id``/``Info`` map to ``None``.

        Returns:
            GroundControlPoint: the equivalent value object.

        Examples:
            - Convert a GDAL GCP into the value object:
                ```python
                >>> from osgeo import gdal
                >>> from pyramids.dataset._gcp import GroundControlPoint
                >>> gp = GroundControlPoint.from_gdal(gdal.GCP(1.0, 2.0, 0.0, 7.0, 3.0))
                >>> (gp.col, gp.row, gp.x, gp.y)
                (7.0, 3.0, 1.0, 2.0)

                ```
        """
        return cls(
            row=gcp.GCPLine,
            col=gcp.GCPPixel,
            x=gcp.GCPX,
            y=gcp.GCPY,
            z=gcp.GCPZ,
            id=gcp.Id or None,
            info=gcp.Info or None,
        )
