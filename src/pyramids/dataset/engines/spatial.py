"""Spatial engine.

Owns the Spatial family of operations on a Dataset. Accessed as
``ds.spatial``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.spatial.<method>(...)`` are equivalent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr

from pyramids.base._domain import is_no_data
from pyramids.base._utils import INTERPOLATION_METHODS
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

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines.vectorize import Vectorize


class Spatial(_Engine):

    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        return str(self._ds.raster.GetProjection())

    def set_crs(self, crs: str | None = None, epsg: int | None = None) -> None:
        """Set the Coordinate Reference System (CRS).

            Set the Coordinate Reference System (CRS) of a

        Args:
            crs (str):
                Optional if epsg is specified. WKT string. i.e.
                    ```
                    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84", 6378137,298.257223563,AUTHORITY["EPSG","7030"],
                    AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
                    0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],
                    AUTHORITY["EPSG","4326"]]'
                    ```
            epsg (int):
                Optional if crs is specified. EPSG code specifying the projection.
        """
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if self._ds.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then "
                "reset the crs"
            )
        else:
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
            else:
                raise ValueError("Either crs or epsg must be provided.")

    def to_crs(
        self,
        to_epsg: int | str | Any,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
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
                resampling method. Default is "nearest neighbor". See https://gisgeography.com/raster-resampling/.
                Allowed values: "nearest neighbor", "cubic", "bilinear".
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the raster the same after reprojection.
                Default is False.

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
        if not isinstance(method, str):
            raise TypeError(
                "Please enter a correct method, for more information, see documentation "
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method: {method} does not exist, existing methods are "
                f"{INTERPOLATION_METHODS.keys()}"
            )

        resampling_method: Any = INTERPOLATION_METHODS.get(method)

        if maintain_alignment:
            dst_obj = self._reproject_with_ReprojectImage(dst_sr, resampling_method)
        else:
            # Prefer the "<AUTHORITY>:<code>" form when one exists so the
            # output WKT GDAL writes is the canonical GDAL/PROJ form
            # (matching historical bytes for EPSG codes and avoiding a
            # GDAL warning when the authority is ESRI). Fall back to the
            # explicit WKT for CRSes carrying no authority at all
            # (custom orthographic proj4 strings, etc.). See #418.
            dst_auth = dst_sr.GetAuthorityName(None)
            dst_code = dst_sr.GetAuthorityCode(None)
            dst_srs_arg = (
                f"{dst_auth}:{dst_code}"
                if dst_auth is not None and dst_code is not None
                else dst_sr.ExportToWkt()
            )
            dst = gdal.Warp(
                "", self._ds.raster, dstSRS=dst_srs_arg, format="VRT"
            )
            dst_obj = self._ds.__class__(dst)

        return dst_obj

    def _get_epsg(self) -> int:
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

    def convert_longitude(self) -> Dataset:
        """Convert Longitude.

        - convert the longitude from 0-360 to -180 - 180.
        - currently the function works correctly if the raster covers the whole world, it means that the columns
            in the rasters covers from longitude 0 to 360.

        Returns:
            Dataset:
                A new Dataset with longitude converted to -180/180.
        """
        # dst = gdal.Warp(
        #     "",
        #     self._ds.raster,
        #     dstSRS="+proj=longlat +ellps=WGS84 +datum=WGS84 +lon_0=0 +over",
        #     format="VRT",
        # )
        lon = self._ds.lon
        src = self._ds.raster
        # create a copy
        drv = gdal.GetDriverByName("MEM")
        dst = drv.CreateCopy("", src, 0)
        # convert the 0 to 360 to -180 to 180
        if lon[-1] <= 180:
            raise ValueError("The raster should cover the whole globe")

        first_to_translated = np.where(lon > 180)[0][0]

        ind = list(range(first_to_translated, len(lon)))
        ind_2 = list(range(0, first_to_translated))

        for band in range(self._ds.band_count):
            arr = self._ds.read_array(band=band)
            arr_rearranged = arr[:, ind + ind_2]
            dst.GetRasterBand(band + 1).WriteArray(arr_rearranged)

        # correct the geotransform
        top_left_corner = self._ds.top_left_corner
        gt = list(self._ds.geotransform)
        if lon[-1] > 180:
            new_gt = top_left_corner[0] - 180
            gt[0] = new_gt

        dst.SetGeoTransform(gt)
        return self._ds.__class__(dst)

    def resample(
        self, cell_size: int | float, method: str = "nearest neighbor"
    ) -> Dataset:
        """resample.

        resample method reprojects a raster to any projection (default the WGS84 web mercator projection,
        without resampling). The function returns a GDAL in-memory file object.

        Args:
            cell_size (int):
                New cell size to resample the raster. If None, raster will not be resampled.
            method (str):
                Resampling method: "nearest neighbor", "cubic", or "bilinear". Default is "nearest neighbor".

        Returns:
            Dataset:
                A new resampled Dataset.

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
        if not isinstance(method, str):
            raise TypeError(
                "Please enter a correct method, for more information, see documentation"
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method does not exist, existing methods are "
                f"{INTERPOLATION_METHODS.keys()}"
            )

        resampling_method: Any = INTERPOLATION_METHODS.get(method)

        sr_src = sr_from_wkt(self._ds.crs)

        ulx = self._ds.geotransform[0]
        uly = self._ds.geotransform[3]
        # transform the right lower corner point
        lrx = self._ds.geotransform[0] + self._ds.geotransform[1] * self._ds.columns
        lry = self._ds.geotransform[3] + self._ds.geotransform[5] * self._ds.rows

        # new geotransform
        new_geo = (
            self._ds.geotransform[0],
            cell_size,
            self._ds.geotransform[2],
            self._ds.geotransform[3],
            self._ds.geotransform[4],
            -1 * cell_size,
        )
        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / cell_size))
        rows = int(np.round(abs(uly - lry) / cell_size))
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
        method: str = "nearest neighbor",
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
            method: GDAL resampling algorithm (e.g. ``gdal.GRA_NearestNeighbour``,
                ``gdal.GRA_Bilinear``, ``gdal.GRA_Cubic``). The default string
                ``"nearest neighbor"`` is a placeholder for the typed enum;
                pass the resolved enum through :data:`INTERPOLATION_METHODS`
                when calling from outside :meth:`to_crs`.

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
            west_edge = src_gt[0] - 360 if src_sr.IsGeographic() and src_gt[0] > 180 else src_gt[0]
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

    def fill_gaps(self, mask, src_array: np.ndarray) -> np.ndarray:
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
        mask_array = mask.read_array()
        mask_noval = mask.no_data_value[0]

        if isinstance(mask, RasterBase) and isinstance(self._ds, RasterBase):
            src_no_data = is_no_data(src_array, self._ds.no_data_value[0])
            mask_no_data = is_no_data(mask_array, mask_noval)
            elem_src = src_array.size - np.count_nonzero(src_array[src_no_data])
            elem_mask = mask_array.size - np.count_nonzero(mask_array[mask_no_data])

            # Cells that are out-of-domain in src but in-domain in mask
            # need to be interpolated from neighbors.
            if elem_mask > elem_src:
                gap_rows, gap_cols = np.where(src_no_data & ~mask_no_data)
                src_array = Vectorize._nearest_neighbour(
                    src_array,
                    self._ds.no_data_value[0],
                    gap_rows.tolist(),
                    gap_cols.tolist(),
                )
        return src_array

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
            mask_epsg = mask.epsg
            row = mask.rows
            col = mask.columns
            mask_noval = mask.no_data_value[0]
            mask_array = mask.read_array(band=0)
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
        src_array = self._ds.read_array()

        if not row == self._ds.rows or not col == self._ds.columns:
            raise ValueError(
                "Two rasters have different number of columns or rows, please resample or match both rasters"
            )

        if isinstance(mask, RasterBase):
            if (
                not self._ds.top_left_corner == mask.top_left_corner
                or not self._ds.cell_size == mask.cell_size
            ):
                raise ValueError(
                    "the location of the upper left corner of both rasters is not the same or cell size is "
                    "different please match both rasters first "
                )

            if not mask_epsg == self._ds.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate systems please reproject one of them to "
                    "the other raster coordinate system"
                )

        mask_no_data = is_no_data(mask_array, mask_noval)
        if band_count > 1:
            # check if the no data value for the src complies with the dtype of the src as sometimes the band is full
            # of values and the no_data_value is not used at all in the band, and when we try to replace any value in
            # the array with the no_data_value it will raise an error.
            no_data_value = self._ds._check_no_data_value(self._ds.no_data_value)
            for band in range(self._ds.band_count):
                src_array[band, mask_no_data] = no_data_value[band]
        else:
            src_array[mask_no_data] = self._ds.no_data_value[0]

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
        if band_count > 1:
            for band in range(band_count):
                dst_obj.raster.GetRasterBand(band + 1).WriteArray(src_array[band, :, :])
        else:
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)
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
            reprojected_raster_b = self.to_crs(src.epsg)  # type: ignore[assignment]
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
        elif isinstance(mask, RasterBase):
            mask = mask
        else:
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
            dst_obj = base_cls(dst)
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

        x_ind = np.where(~rows_to_remove)[0][0]
        y_ind = np.where(~cols_to_remove)[0][0]
        new_x = src.x[y_ind] - src.cell_size / 2
        new_y = src.y[x_ind] + src.cell_size / 2
        new_gt = (new_x, src.cell_size, 0, new_y, 0, -src.cell_size)
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
                path. Mutually exclusive with ``mask``.
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
            crs = epsg if epsg is not None else self._ds.epsg
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
