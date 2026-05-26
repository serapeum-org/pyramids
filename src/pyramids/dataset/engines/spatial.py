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
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> Dataset:
        """Reproject the dataset to any projection.

            (default the WGS84 web mercator projection, without resampling)

        Args:
            to_epsg (int):
                reference number to the new projection (https://epsg.io/). Default 3857 is the reference number of WGS84
                web mercator.
            method (str):
                resampling method. Default is "nearest neighbor". See https://gisgeography.com/raster-resampling/.
                Allowed values: "nearest neighbor", "cubic", "bilinear".
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the raster the same after reprojection.
                Default is False.

        Returns:
            Dataset:
                A new reprojected Dataset.

        Examples:
            - Create a dataset and reproject it:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> print(dataset.crs)
              GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
              >>> print(dataset.epsg)
              4326
              >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)
              >>> print(reprojected_dataset)
              <BLANKLINE>
                          Cell size: 5565.983370404396
                          Dimension: 5 * 5
                          EPSG: 3857
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> print(reprojected_dataset.crs)
              PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]
              >>> print(reprojected_dataset.epsg)
              3857

              ```

        """
        if not isinstance(to_epsg, int):
            raise TypeError(
                "please enter correct integer number for to_epsg more information "
                f"https://epsg.io/, given {type(to_epsg)}"
            )
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
            dst_obj = self._reproject_with_ReprojectImage(to_epsg, resampling_method)
        else:
            dst = gdal.Warp("", self._ds.raster, dstSRS=f"EPSG:{to_epsg}", format="VRT")
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
            - Create a Dataset with 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 10 * 10
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File: ...
              <BLANKLINE>
              >>> dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
              ![resample-source](./../../_images/dataset/resample-source.png)

            - Resample the raster to a new cell size of 0.1:

              ```python
              >>> new_dataset = dataset.resample(cell_size=0.1)
              >>> print(new_dataset)
              <BLANKLINE>
                          Cell size: 0.1
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> new_dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
              ![resample-new](./../../_images/dataset/resample-new.png)

            - Resampling the dataset from cell_size 0.05 to 0.1 degrees reduced the number of cells to 5 in each
              dimension instead of 10.
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
        self, to_epsg: int, method: str = "nearest neighbor"
    ) -> Dataset:
        src_gt = self._ds.geotransform
        src_x = self._ds.columns
        src_y = self._ds.rows

        src_sr = sr_from_wkt(self._ds.crs)
        src_epsg = self._ds.epsg

        dst_sr = sr_from_epsg(to_epsg)

        # in case the source crs is GCS and longitude is in the west hemisphere, gdal
        # reads longitude from 0 to 360 and a transformation factor wont work with values
        # greater than 180
        if src_epsg != to_epsg:
            if src_epsg == "4326" and src_gt[0] > 180:
                lng_new = src_gt[0] - 360
                # transformation factors
                tx = osr.CoordinateTransformation(src_sr, dst_sr)
                # transform the right upper corner point
                ulx, uly, ulz = tx.TransformPoint(lng_new, src_gt[3])
                # transform the right lower corner point
                lrx, lry, lrz = tx.TransformPoint(
                    lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                )
            else:
                xs = [src_gt[0], src_gt[0] + src_gt[1] * src_x]
                ys = [src_gt[3], src_gt[3] + src_gt[5] * src_y]

                # reproject_coordinates takes (x, y) and returns (x, y).
                [ulx, lrx], [uly, lry] = reproject_coordinates(
                    xs, ys, from_crs=src_epsg, to_crs=to_epsg
                )
                # old transform
                # # transform the right upper corner point
                # (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
                # # transform the right lower corner point
                # (lrx, lry, lrz) = tx.TransformPoint(
                #     src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                # )

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

        if src_epsg != to_epsg:
            # x_pair_xs and x_pair_ys are horizontally spaced by the cell size, after reprojection gives the cell size
            # in x
            new_x_xs, _ = reproject_coordinates(
                x_pair_xs,
                x_pair_ys,
                from_crs=src_epsg,
                to_crs=to_epsg,
                precision=6,
            )
            # y_pair_xs and y_pair_ys are vertically spaced by the cell size, after reprojection gives the cell size
            # in y
            _, new_y_ys = reproject_coordinates(
                y_pair_xs,
                y_pair_ys,
                from_crs=src_epsg,
                to_crs=to_epsg,
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
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              ```

            - The dataset to be aligned has a top_left_corner at (-0.1, 0.1) (i.e., it has two more rows on top of the
              dataset, and two columns on the left of the dataset).

              ```python
              >>> arr = np.random.rand(10, 10)
              >>> top_left_corner = (-0.1, 0.1)
              >>> cell_size = 0.07
              >>> dataset_target = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,
              ... epsg=4326)
              >>> print(dataset_target)
              <BLANKLINE>
                          Cell size: 0.07
                          Dimension: 10 * 10
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              ```

            ![align-source-target](./../../_images/dataset/align-source-target.png)

            - Now call the `align` method and use the dataset as the alignment source.

              ```python
              >>> aligned_dataset = dataset_target.align(dataset)
              >>> print(aligned_dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

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
        """Correct wrap cutline error.

        https://github.com/serapeum-org/pyramids/issues/74
        """
        big_array = src.read_array()
        value_to_remove = src.no_data_value[0]
        """Remove rows and columns that are all filled with a certain value from a 2D array."""
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
