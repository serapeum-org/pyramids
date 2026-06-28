"""Analysis engine.

Owns the Analysis family of operations on a Dataset. Accessed as
``ds.analysis``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.analysis.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, get_pixels2
from osgeo import gdal
from pandas import DataFrame

from pyramids.base._domain import inside_domain, is_no_data
from pyramids.base._errors import AlignmentError, OutOfBoundsError, ReadOnlyError
from pyramids.base._utils import gdal_to_numpy_dtype, require_cleopatra
from pyramids.dataset._mask import MaskFlags
from pyramids.dataset._plot_helpers import render_array
from pyramids.dataset.window import Window
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from cleopatra.array_glyph import ArrayGlyph
    from pyramids.dataset.dataset import Dataset

from pyramids.dataset.engines._base import _Engine


class Analysis(_Engine):
    """Mixin providing analysis, statistics, and data extraction operations for Dataset."""

    def stats(
        self, band: int | None = None, mask: GeoDataFrame | None = None
    ) -> DataFrame:
        """Get statistics of a band [Min, max, mean, std].

        Args:
            band (int, optional):
                Band index. If None, the statistics of all bands will be returned.
            mask (Polygon GeoDataFrame or Dataset, optional):
                GeodataFrame with a geometry of polygon type.

        Returns:
            DataFrame:
                DataFrame wit the stats of each band, the dataframe has the following columns
                [min, max, mean, std], the index of the dataframe is the band names.

                ```text

                                   Min         max        mean       std
                    Band_1  270.369720  270.762299  270.551361  0.154270
                    Band_2  269.611938  269.744751  269.673645  0.043788
                    Band_3  273.641479  274.168823  273.953979  0.198447
                    Band_4  273.991516  274.540344  274.310669  0.205754
                ```

        Notes:
            - The value of the stats will be stored in an xml file by the name of the raster file with the extension of
              .aux.xml.
            - The content of the file will be like the following:

              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Metadata>
                        <MDI key="RepresentationType">ATHEMATIC</MDI>
                        <MDI key="STATISTICS_MAXIMUM">88</MDI>
                        <MDI key="STATISTICS_MEAN">7.9662921348315</MDI>
                        <MDI key="STATISTICS_MINIMUM">0</MDI>
                        <MDI key="STATISTICS_STDDEV">18.294377743948</MDI>
                        <MDI key="STATISTICS_VALID_PERCENT">48.9</MDI>
                      </Metadata>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Get the statistics of all bands in the dataset:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> geotransform = (0, 0.05, 0, 0, 0, -0.05)
              >>> dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
              >>> print(dataset.stats()) # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.006443  0.942943  0.468935  0.266634
              Band_2  0.020377  0.978130  0.477189  0.306864
              Band_3  0.019652  0.992184  0.537215  0.286502
              Band_4  0.011955  0.984313  0.503616  0.295852
              >>> print(dataset.stats(band=1))  # doctest: +SKIP
                           min      max      mean       std
              Band_2  0.020377  0.97813  0.477189  0.306864

              ```

            - Get the statistics of all the bands using a mask polygon.

              - Create the polygon using shapely polygon, and use the xmin, ymin, xmax, ymax = [0.1, -0.2,
                0.2 -0.1] to cover the 4 cells.
              ```python
              >>> from shapely.geometry import Polygon
              >>> import geopandas as gpd
              >>> mask = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],crs=4326)
              >>> print(dataset.stats(mask=mask))  # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.193441  0.702108  0.541478  0.202932
              Band_2  0.281281  0.932573  0.665602  0.239410
              Band_3  0.031395  0.982235  0.493086  0.377608
              Band_4  0.079562  0.930965  0.591025  0.341578

              ```

        """
        dst: Dataset | None = None
        if mask is not None:
            dst = self._ds.crop(mask, touch=True)

        if band is None:
            df = pd.DataFrame(
                index=self._ds.band_names,
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            for i in range(self._ds.band_count):
                if mask is not None and dst is not None:
                    df.iloc[i, :] = dst.analysis._get_stats(i)
                else:
                    df.iloc[i, :] = self._get_stats(i)
        else:
            df = pd.DataFrame(
                index=[self._ds.band_names[band]],
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            if mask is not None and dst is not None:
                df.iloc[0, :] = dst.analysis._get_stats(band)
            else:
                df.iloc[0, :] = self._get_stats(band)

        return df

    def _get_stats(self, band: int | None = None) -> list[float]:
        """Return summary statistics for one band.

        Reads GDAL band statistics, computing them on the fly when the cached values are
        absent or empty.

        Args:
            band (int | None):
                Zero-based band index. Defaults to the first band (0) when None.

        Returns:
            list[float]: The ``[minimum, maximum, mean, standard_deviation]`` values.
        """
        band_index = band if band is not None else 0
        band_i = self._ds._iloc(band_index)
        try:
            vals = band_i.GetStatistics(True, True)
        except RuntimeError:
            # when the GetStatistics gives an error "RuntimeError: Failed to compute statistics, no valid pixels
            # found in sampling."
            vals = [0]

        if sum(vals) == 0:
            warnings.warn(
                f"Band {band} has no statistics, and the statistics are going to be calculate"
            )
            vals = band_i.ComputeStatistics(False)

        return list(vals)

    def count_domain_cells(self, band: int = 0) -> int:
        """Count cells inside the domain.

        Args:
            band (int):
                Band index. Default is 0.

        Returns:
            int:
                Number of cells.
        """
        arr = self._ds.read_array(band=band)
        domain_count = np.size(arr[:, :]) - np.count_nonzero(
            arr[is_no_data(arr, self._ds.no_data_value[band])]
        )
        return int(domain_count)

    def apply(self, func, band: int = 0, inplace: bool = False) -> Dataset:
        """Apply a function to all domain cells.

        - apply method executes a mathematical operation on the raster array.
        - The function is applied to all domain cells at once using vectorized NumPy operations.

        Args:
            func (function):
                Defined function that takes one input (the cell value).
            band (int):
                Band number.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created.
                Default is False.

        Returns:
            Dataset:
                A new Dataset with the function applied. If inplace is True, returns self.

        Examples:
            - Create a dataset from an array filled with values between -1 and 1:

              ```python
              >>> import numpy as np
              >>> arr = np.random.uniform(-1, 1, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[ 0.94997539 -0.80083622 -0.30948769 -0.77439961 -0.83836424]
               [-0.36810158 -0.23979251  0.88051216 -0.46882913  0.64511056]
               [ 0.50585374 -0.46905902  0.67856589  0.2779605   0.05589759]
               [ 0.63382852 -0.49259597  0.18471423 -0.49308984 -0.52840286]
               [-0.34076174 -0.53073014 -0.18485789 -0.40033474 -0.38962938]]

              ```

            - Apply the absolute function to the dataset:

              ```python
              >>> abs_dataset = dataset.apply(np.abs)
              >>> print(abs_dataset.read_array()) # doctest: +SKIP
              [[0.94997539 0.80083622 0.30948769 0.77439961 0.83836424]
               [0.36810158 0.23979251 0.88051216 0.46882913 0.64511056]
               [0.50585374 0.46905902 0.67856589 0.2779605  0.05589759]
               [0.63382852 0.49259597 0.18471423 0.49308984 0.52840286]
               [0.34076174 0.53073014 0.18485789 0.40033474 0.38962938]]

              ```
        """
        if not callable(func):
            raise TypeError("The second argument should be a function")

        no_data_value = self._ds.no_data_value[band]
        src_array = self._ds.read_array(band)
        dtype = self._ds.gdal_dtype[band]

        new_array = np.full(
            (self._ds.rows, self._ds.columns), no_data_value, dtype=src_array.dtype
        )
        domain_mask = inside_domain(src_array, no_data_value)
        domain_values = src_array[domain_mask]
        try:
            new_array[domain_mask] = func(domain_values)
        except (ValueError, TypeError):
            new_array[domain_mask] = np.vectorize(func)(domain_values)

        dst_obj = self._ds.__class__._build_dataset(
            self._ds.columns,
            self._ds.rows,
            1,
            dtype,
            self._ds.geotransform,
            self._ds.crs,
            no_data_value,
        )
        dst_obj.raster.GetRasterBand(1).WriteArray(new_array)

        if inplace:
            self._ds._update_inplace(dst_obj.raster)
            return None
        return dst_obj

    def fill(
        self, value: float | int, inplace: bool = False, path: str | Path | None = None
    ) -> Dataset:
        """Fill the domain cells with a certain value.

            Fill takes a raster and fills it with one value

        Args:
            value (float | int):
                Numeric value to fill.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created. Default is False.
            path (str):
                Path including the extension (.tif).

        Returns:
            Dataset:
                A new Dataset with cells filled. If inplace is True, returns self.

        Examples:
            - Create a Dataset with 1 band, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 5, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[1 1 3 1 2]
               [2 2 2 1 2]
               [2 2 3 1 3]
               [3 4 3 3 4]
               [4 4 2 1 1]]
              >>> new_dataset = dataset.fill(10)
              >>> print(new_dataset.read_array())
              [[10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]]

              ```
        """
        no_data_value = self._ds.no_data_value[0]
        src_array = self._ds.raster.ReadAsArray()

        # rtol=1e-6 is intentionally tighter than the package default
        # (1e-3): `fill` writes user-supplied values into every domain
        # cell, so a too-loose match would clobber legitimate cells that
        # happen to lie within ~0.1% of the no-data sentinel.
        src_array[inside_domain(src_array, no_data_value, rtol=0.000001)] = value

        dst = self._ds.__class__.dataset_like(self._ds, src_array, path=path)
        if inplace:
            self._ds._update_inplace(dst.raster)
            return None
        return dst

    def extract(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        mask: FeatureCollection | GeoDataFrame | None = None,
    ) -> np.typing.NDArray:
        """Extract.

        - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
        - If the mask parameter is given, the raster will be clipped to the extent of the given mask and the
          values within the mask are extracted.

        Args:
            band (int, optional):
                Band index. Default is None.
            exclude_value (Numeric, optional):
                Values to exclude from extracted values. If the dataset is multi-band, the values in `exclude_value`
                will be filtered out from the first band only.
            mask (FeatureCollection | GeoDataFrame, optional):
                Vector data containing point geometries at which to extract the values. Default is None.

        Returns:
            np.ndarray:
                The extracted values from each band in the dataset will be in one row in the returned array.

        Examples:
            - Extract all values from the dataset:

              - First, create a dataset with 2 bands, 4 rows and 4 columns:

                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(1, 5, size=(2, 4, 4))
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
                >>> print(dataset)
                <BLANKLINE>
                            Cell size: 0.05
                            Dimension: 4 * 4
                            EPSG: 4326
                            Number of Bands: 2
                            Band names: ['Band_1', 'Band_2']
                            Mask: -9999.0
                            Data type: int32
                            File:...
                <BLANKLINE>
                >>> print(dataset.read_array()) # doctest: +SKIP
                [[[1 3 3 4]
                  [1 4 2 4]
                  [2 4 2 1]
                  [1 3 2 3]]
                 [[3 2 1 3]
                  [4 3 2 2]
                  [2 2 3 4]
                  [1 4 1 4]]]

                ```

              - Now, extract the values in the dataset:

                ```python
                >>> values = dataset.extract()
                >>> print(values) # doctest: +SKIP
                [[1 3 3 4 1 4 2 4 2 4 2 1 1 3 2 3]
                 [3 2 1 3 4 3 2 2 2 2 3 4 1 4 1 4]]

                ```

              - Extract all the values except 2:

                ```python
                >>> values = dataset.extract(exclude_value=2)
                >>> print(values) # doctest: +SKIP

                ```

            - Extract values at the location of the given point geometries:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import Point
              ```

              - Create the points using shapely and GeoPandas to cover the 4 cells with xmin, ymin, xmax, ymax = [0.1, -0.2, 0.2, -0.1]:

                ```python
                >>> points = gpd.GeoDataFrame(geometry=[Point(0.1, -0.1), Point(0.1, -0.2), Point(0.2, -0.2), Point(0.2, -0.1)],crs=4326)
                >>> values = dataset.extract(mask=points)
                >>> print(values) # doctest: +SKIP
                [[4 3 3 4]
                 [3 4 4 2]]

                ```
        """
        # Optimize: make the read_array return only the array for inside the mask feature, and not to read the whole
        #  raster
        arr = self._ds.read_array(band=band)
        no_data_value = (
            self._ds.no_data_value[0]
            if self._ds.no_data_value[0] is not None
            else np.nan
        )
        if mask is None:
            exclude_list = (
                [no_data_value, exclude_value]
                if exclude_value is not None
                else [no_data_value]
            )
            values = get_pixels2(arr, exclude_list)
        else:
            geom_types = set(getattr(mask, "geom_type", []))
            # map(str, ...) — missing geometries yield float nan, which is not
            # orderable against the str type names.
            if geom_types - {"Point"}:
                raise ValueError(
                    "extract(mask=...) expects Point geometries — one value is read "
                    f"per point; got {sorted(map(str, geom_types))}. For polygon "
                    "zones use Dataset.zonal_stats(); to clip a raster use "
                    "Dataset.crop(); explode MultiPoint masks into single points "
                    "first."
                )
            indices = self._ds.map_to_array_coordinates(mask)
            if arr.ndim > 2:
                values = arr[:, indices[:, 0], indices[:, 1]]
            else:
                values = arr[indices[:, 0], indices[:, 1]]

        return np.asarray(values)

    def _points_to_xy(
        self, points: FeatureCollection | GeoDataFrame | DataFrame
    ) -> np.typing.NDArray:
        """Extract an ``(N, 2)`` float array of ``(x, y)`` coordinates from points.

        Args:
            points: A point :class:`~pyramids.feature.FeatureCollection` /
                :class:`~geopandas.GeoDataFrame`, or a :class:`~pandas.DataFrame`
                carrying ``x`` and ``y`` columns.

        Returns:
            np.ndarray: Coordinates with shape ``(N, 2)`` as ``float``.

        Raises:
            ValueError: A ``DataFrame`` lacking ``x``/``y`` columns.
            TypeError: ``points`` is not a supported type.
        """
        if isinstance(points, FeatureCollection):
            verts = points.with_coordinates()
            return verts.loc[:, ["x", "y"]].to_numpy(dtype=float)
        if isinstance(points, GeoDataFrame):
            verts = FeatureCollection(points).with_coordinates()
            return verts.loc[:, ["x", "y"]].to_numpy(dtype=float)
        if isinstance(points, DataFrame):
            if not all(col in points.columns for col in ("x", "y")):
                raise ValueError(
                    "If the input is a DataFrame, it must have 'x' and 'y' columns."
                )
            return points.loc[:, ["x", "y"]].to_numpy(dtype=float)
        raise TypeError(
            "points must be a FeatureCollection, GeoDataFrame, or DataFrame with "
            f"x/y columns - given {type(points)}."
        )

    def sample(
        self,
        points: FeatureCollection | GeoDataFrame | DataFrame,
        *,
        bands: int | list[int] | None = None,
        masked: bool = False,
        on_out_of_bounds: str = "nodata",
    ) -> np.typing.NDArray:
        """Sample band values at point coordinates.

        The memory- and out-of-bounds-safe counterpart to
        :meth:`extract` with a point mask. Each point is mapped to its
        containing pixel with a **vectorised inverse geotransform** (``O(1)`` per
        point) and read with a **1x1 windowed read** — so a handful of points on
        a multi-gigabyte raster touches only those pixels, never the whole array.
        Points falling outside the raster are handled explicitly instead of being
        silently snapped to the nearest edge cell.

        Args:
            points (FeatureCollection | GeoDataFrame | DataFrame):
                Point locations to sample. A ``FeatureCollection`` /
                ``GeoDataFrame`` with point geometry, or a ``DataFrame`` with
                ``x`` and ``y`` columns. Coordinates must already be in the
                raster's CRS (no reprojection is performed).
            bands (int | list[int] | None):
                Which band(s) to sample, zero-based. ``None`` (default) samples
                every band and returns a ``(n_bands, n_points)`` array; a single
                ``int`` returns a 1-D ``(n_points,)`` array; a list returns a
                ``(len(bands), n_points)`` array in the requested order.
            masked (bool):
                When ``True`` return a :class:`numpy.ma.MaskedArray` with
                out-of-bounds points masked. Defaults to ``False``.
            on_out_of_bounds (str):
                How to treat points outside the raster extent:

                - ``"nodata"`` (default): fill with the band's no-data value
                  (``NaN`` when the band has none).
                - ``"raise"``: raise :class:`OutOfBoundsError`.
                - ``"snap"``: clamp to the nearest edge pixel (the legacy
                  :meth:`extract` behaviour).

        Returns:
            np.ndarray:
                Sampled values, ordered to match ``points``. Shape is
                ``(n_points,)`` for a single ``int`` band, otherwise
                ``(n_bands, n_points)``. A :class:`numpy.ma.MaskedArray` when
                ``masked=True``.

        Raises:
            ValueError: ``on_out_of_bounds`` is not one of the allowed values, or
                ``bands`` references a band outside the raster.
            OutOfBoundsError: ``on_out_of_bounds="raise"`` and a point lies
                outside the raster extent.
            TypeError: ``points`` is not a supported type.

        Examples:
            - Sample a 2-band raster at three points and read the per-band values:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(2 * 5 * 5, dtype="float32").reshape(2, 5, 5)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
                ... )
                >>> pts = GeoDataFrame(
                ...     geometry=[Point(0.5, 4.5), Point(2.5, 2.5)], crs=4326
                ... )
                >>> ds.sample(pts).tolist()
                [[0.0, 12.0], [25.0, 37.0]]

                ```
            - Sample a single band and get a flat array of values:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
                ... )
                >>> pts = GeoDataFrame(geometry=[Point(0.5, 4.5), Point(4.5, 0.5)], crs=4326)
                >>> ds.sample(pts, bands=0).tolist()
                [0.0, 24.0]

                ```
            - Points outside the extent become no-data instead of snapping:
                ```python
                >>> import numpy as np
                >>> from geopandas import GeoDataFrame
                >>> from shapely.geometry import Point
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326,
                ...     no_data_value=-9999.0,
                ... )
                >>> pts = GeoDataFrame(geometry=[Point(2.5, 2.5), Point(100, 100)], crs=4326)
                >>> ds.sample(pts, bands=0).tolist()
                [12.0, -9999.0]

                ```
        """
        if on_out_of_bounds not in ("nodata", "raise", "snap"):
            raise ValueError(
                "on_out_of_bounds must be one of 'nodata', 'raise', 'snap'; got "
                f"{on_out_of_bounds!r}."
            )

        band_count = self._ds.band_count
        if bands is None:
            band_list = list(range(band_count))
            squeeze = False
        elif isinstance(bands, int):
            band_list = [bands]
            squeeze = True
        else:
            band_list = list(bands)
            squeeze = False
        for b in band_list:
            if b < 0 or b >= band_count:
                raise ValueError(
                    f"band {b} is out of range for a {band_count}-band dataset."
                )

        xy = self._points_to_xy(points)
        n_points = xy.shape[0]

        x0, dx, rxy, y0, ryx, dy = self._ds.geotransform
        det = dx * dy - rxy * ryx
        delta_x = xy[:, 0] - x0
        delta_y = xy[:, 1] - y0
        col = np.floor((dy * delta_x - rxy * delta_y) / det).astype(int)
        row = np.floor((-ryx * delta_x + dx * delta_y) / det).astype(int)

        n_rows, n_cols = self._ds.rows, self._ds.columns
        out_of_bounds = (row < 0) | (row >= n_rows) | (col < 0) | (col >= n_cols)
        if on_out_of_bounds == "raise" and out_of_bounds.any():
            raise OutOfBoundsError(
                f"{int(out_of_bounds.sum())} of {n_points} points fall outside the "
                "raster extent."
            )
        if on_out_of_bounds == "snap":
            row = np.clip(row, 0, n_rows - 1)
            col = np.clip(col, 0, n_cols - 1)
            out_of_bounds = np.zeros(n_points, dtype=bool)

        in_bounds_idx = np.flatnonzero(~out_of_bounds)
        rows_out: list[np.ndarray] = []
        for b in band_list:
            gdal_band = self._ds.raster.GetRasterBand(b + 1)
            no_data_value = gdal_band.GetNoDataValue()
            band_dtype = np.dtype(gdal_to_numpy_dtype(gdal_band.DataType))
            if no_data_value is None:
                fill: Any = np.nan
                out_dtype = (
                    band_dtype
                    if np.issubdtype(band_dtype, np.floating)
                    else np.dtype("float64")
                )
            else:
                fill = no_data_value
                out_dtype = band_dtype
            band_values = np.full(n_points, fill, dtype=out_dtype)
            for i in in_bounds_idx:
                window = gdal_band.ReadAsArray(int(col[i]), int(row[i]), 1, 1)
                band_values[i] = window[0, 0]
            rows_out.append(band_values)

        stacked = np.vstack(rows_out) if rows_out else np.empty((0, n_points))
        result: np.ndarray = stacked[0] if squeeze else stacked
        if masked:
            mask = (
                out_of_bounds
                if squeeze
                else np.broadcast_to(out_of_bounds, result.shape)
            )
            result = np.ma.masked_array(result, mask=np.array(mask))
        return result

    def sieve(
        self,
        threshold: int,
        *,
        band: int = 0,
        connectedness: int = 4,
        mask: Dataset | None = None,
    ) -> Dataset:
        """Remove small pixel clumps with ``gdal.SieveFilter``.

        Raster polygons — connected groups of identical-value pixels — smaller
        than ``threshold`` pixels are dissolved into their largest neighbour.
        This is the standard clean-up for "salt-and-pepper" speckle in
        classification rasters. Implemented natively via GDAL; returns a new
        single-band :class:`~pyramids.dataset.Dataset`.

        Args:
            threshold (int):
                Minimum polygon size to keep, in pixels. Clumps with fewer
                pixels are merged away. Must be ``>= 1``.
            band (int):
                Zero-based index of the band to sieve. Defaults to ``0``.
            connectedness (int):
                Pixel connectivity used to define a clump: ``4`` (edge-adjacent,
                the default) or ``8`` (edge- and diagonal-adjacent).
            mask (Dataset | None):
                Optional single-band mask. Pixels where the mask is zero are
                excluded from sieving. ``None`` (default) uses the source band's
                no-data mask.

        Returns:
            Dataset:
                A new single-band dataset with small clumps removed, sharing the
                source geotransform, CRS, and no-data value.

        Raises:
            ValueError: ``threshold < 1``, ``connectedness`` is not 4 or 8, or
                ``band`` is out of range.

        Examples:
            - Remove an isolated speckle pixel from a classified raster:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((6, 6), dtype="int32")
                >>> arr[0:3, 0:3] = 2      # a 9-pixel clump (kept)
                >>> arr[5, 5] = 2          # a lone pixel (removed)
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
                ... )
                >>> cleaned = ds.sieve(threshold=4).read_array()
                >>> int(cleaned[5, 5])     # merged into the background
                1
                >>> int(cleaned[0, 0])     # large clump survives
                2

                ```
            - 8-connectivity joins diagonal neighbours that 4-connectivity keeps
              separate:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.ones((5, 5), dtype="int32")
                >>> arr[1, 1] = 2
                >>> arr[2, 2] = 2          # touches (1,1) only diagonally
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
                ... )
                >>> int(ds.sieve(threshold=2, connectedness=8).read_array()[1, 1])
                2

                ```
        """
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}.")
        if connectedness not in (4, 8):
            raise ValueError(f"connectedness must be 4 or 8, got {connectedness}.")
        if band < 0 or band >= self._ds.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self._ds.band_count}-band dataset."
            )

        src_band = self._ds.raster.GetRasterBand(band + 1)
        out_ds = gdal.GetDriverByName("MEM").Create(
            "", self._ds.columns, self._ds.rows, 1, src_band.DataType
        )
        out_ds.SetGeoTransform(self._ds.geotransform)
        out_ds.SetProjection(self._ds.crs)
        dst_band = out_ds.GetRasterBand(1)
        dst_band.WriteArray(src_band.ReadAsArray())
        no_data_value = src_band.GetNoDataValue()
        if no_data_value is not None:
            dst_band.SetNoDataValue(no_data_value)

        mask_band = mask.raster.GetRasterBand(1) if mask is not None else None
        gdal.SieveFilter(dst_band, mask_band, dst_band, threshold, connectedness)
        dst_band.FlushCache()
        return self._ds.__class__(out_ds, access="write")

    def proximity(
        self,
        *,
        band: int = 0,
        target_values: list[int] | None = None,
        distance_units: str = "GEO",
        max_distance: float | None = None,
        nodata: float | None = None,
    ) -> Dataset:
        """Compute per-pixel distance to the nearest target pixel (``gdal.ComputeProximity``).

        The GDAL-native equivalent of ``gdal_proximity``: every output pixel
        holds the Euclidean distance to the closest "target" pixel in the source
        band. Targets are the pixels whose value is in ``target_values`` (or any
        non-zero pixel when ``target_values`` is ``None``). Useful for
        distance-to-coast, distance-to-river, buffer analyses, etc.

        Args:
            band (int):
                Zero-based index of the source band. Defaults to ``0``.
            target_values (list[int] | None):
                Pixel values that count as targets. ``None`` (default) treats
                every non-zero pixel as a target.
            distance_units (str):
                ``"GEO"`` (default) measures distance in the CRS's georeferenced
                units; ``"PIXEL"`` measures it in pixels.
            max_distance (float | None):
                Stop searching beyond this distance. Pixels farther than this get
                ``nodata`` when given, otherwise ``max_distance``. ``None``
                (default) searches the whole raster.
            nodata (float | None):
                Value written to the output band's no-data slot and used to fill
                pixels beyond ``max_distance``. ``None`` (default) sets no
                no-data value.

        Returns:
            Dataset:
                A new single-band ``Float32`` dataset of distances, sharing the
                source geotransform and CRS.

        Raises:
            ValueError: ``distance_units`` is not ``"GEO"``/``"PIXEL"``,
                ``band`` is out of range, or ``max_distance`` is negative.

        Examples:
            - Distance (in pixels) from every cell to a single target pixel:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.zeros((5, 5), dtype="int32")
                >>> arr[2, 2] = 1
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
                ... )
                >>> dist = ds.proximity(distance_units="PIXEL").read_array()
                >>> float(dist[2, 2])      # the target itself
                0.0
                >>> float(dist[2, 0])      # two cells to the left
                2.0

                ```
            - GEO units scale distances by the cell size:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.zeros((5, 5), dtype="int32")
                >>> arr[2, 2] = 1
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 10), cell_size=2.0, epsg=4326
                ... )
                >>> dist = ds.proximity(distance_units="GEO").read_array()
                >>> float(dist[2, 0])      # two cells x 2.0 units
                4.0

                ```
        """
        if distance_units not in ("GEO", "PIXEL"):
            raise ValueError(
                f"distance_units must be 'GEO' or 'PIXEL', got {distance_units!r}."
            )
        if band < 0 or band >= self._ds.band_count:
            raise ValueError(
                f"band {band} is out of range for a {self._ds.band_count}-band dataset."
            )
        if max_distance is not None and max_distance < 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}.")

        src_band = self._ds.raster.GetRasterBand(band + 1)
        out_ds = gdal.GetDriverByName("MEM").Create(
            "", self._ds.columns, self._ds.rows, 1, gdal.GDT_Float32
        )
        out_ds.SetGeoTransform(self._ds.geotransform)
        out_ds.SetProjection(self._ds.crs)
        prox_band = out_ds.GetRasterBand(1)

        options = [f"DISTUNITS={distance_units}"]
        if target_values is not None:
            options.append("VALUES=" + ",".join(str(v) for v in target_values))
        if max_distance is not None:
            options.append(f"MAXDIST={max_distance}")
        if nodata is not None:
            options.append(f"NODATA={nodata}")
            prox_band.SetNoDataValue(float(nodata))

        gdal.ComputeProximity(src_band, prox_band, options=options)
        prox_band.FlushCache()
        return self._ds.__class__(out_ds, access="write")

    def overlay(
        self: Dataset,
        classes_map,
        band: int = 0,
        exclude_value: float | int | None = None,
    ) -> dict[float, list[float]]:
        """Overlay.

        Overlay method extracts all the values in the dataset for each class in the given class map.

        Args:
            classes_map (Dataset):
                Dataset object for the raster that has classes you want to overlay with the raster.
            band (int):
                If the raster is multi-band, choose the band you want to overlay with the classes map. Default is 0.
            exclude_value (Numeric, optional):
                Values you want to exclude from extracted values. Default is None.

        Returns:
            Dict:
                Dictionary with class values as keys (from the class map), and for each key a list of all the intersected
                values in the base map.

        Examples:
            - Read the dataset:

              ```python
              >>> dataset = Dataset.read_file("examples/data/geotiff/raster-folder/MSWEP_1979.01.01.tif")
              >>> dataset.plot(figsize=(6, 8)) # doctest: +SKIP

              ```

              ![rhine-rainfall](./../../_images/dataset/rhine-rainfall.png)

            - Read the classes dataset:

              ```python
              >>> classes = Dataset.read_file("examples/data/geotiff/rhine-classes.tif")
              >>> classes.plot(figsize=(6, 8), color_scale="boundary-norm", bounds=[1,2,3,4,5,6]) # doctest: +SKIP

              ```

              ![rhine-classes](./../../_images/dataset/rhine-classes.png)

            - Overlay the dataset with the classes dataset:

              ```python
              >>> classes_dict = dataset.overlay(classes)
              >>> print(classes_dict.keys()) # doctest: +SKIP
              dict_keys([1, 2, 3, 4, 5])

              ```

            - You can use the key `1` to get the values that overlay class 1.
        """
        if not self._ds.spatial._check_alignment(classes_map):
            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, please use the method "
                "'align' to align both rasters."
            )
        arr = self._ds.read_array(band=band)
        no_data_value = (
            self._ds.no_data_value[0]
            if self._ds.no_data_value[0] is not None
            else np.nan
        )
        mask = (
            [no_data_value, exclude_value]
            if exclude_value is not None
            else [no_data_value]
        )
        ind = get_indices2(arr, mask)
        classes = classes_map.read_array()
        values: dict[Any, list[Any]] = dict()

        # extract values
        for i, ind_i in enumerate(ind):
            # first check if the sub-basin has a list in the dict if not create a list
            key = classes[ind_i[0], ind_i[1]]
            if key not in list(values.keys()):
                values[key] = list()

            values[key].append(arr[ind_i[0], ind_i[1]])

        return values

    def get_mask(self, band: int = 0) -> np.typing.NDArray:
        """Get the mask array.

        Args:
            band (int):
                Band index. Default is 0.

        Returns:
            np.ndarray:
                Array of the mask. 0 value for cells out of the domain, and 255 for cells in the domain.
        """
        arr = np.asarray(self._ds._iloc(band).GetMaskBand().ReadAsArray())
        return arr

    def mask_flags(self, band: int = 0) -> MaskFlags:
        """Decode the GDAL mask flags of ``band`` into a :class:`MaskFlags`.

        Tells you *why* a band is masked (or not): a fully-valid band, a shared
        per-dataset mask, an alpha-band mask, or a no-data-derived mask.

        Args:
            band: Band index. Default 0.

        Returns:
            MaskFlags: the four decoded boolean flags.

        Examples:
            - A band with a no-data value reports ``nodata``:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0),
                ...     cell_size=1.0, no_data_value=-9999.0,
                ... )
                >>> ds.mask_flags().nodata
                True

                ```
        """
        flags = self._ds._iloc(band).GetMaskFlags()
        return MaskFlags(
            all_valid=bool(flags & gdal.GMF_ALL_VALID),
            per_dataset=bool(flags & gdal.GMF_PER_DATASET),
            alpha=bool(flags & gdal.GMF_ALPHA),
            nodata=bool(flags & gdal.GMF_NODATA),
        )

    def read_masks(
        self,
        band: int | None = None,
        *,
        window: Window | None = None,
    ) -> np.typing.NDArray:
        """Read per-band mask arrays (``0`` invalid, ``255`` valid).

        The companion to :meth:`Dataset.read_array(masked=True) <read_array>`:
        instead of applying the mask, it returns the mask itself, so you can
        inspect *which* pixels are masked.

        Args:
            band: Band index. ``None`` (default) returns every band's mask
                stacked as ``(band_count, rows, cols)``; an index returns a
                single ``(rows, cols)`` mask.
            window: Optional :class:`Window` to read only a sub-block.

        Returns:
            numpy.ndarray: the mask array(s); ``0`` marks out-of-domain pixels
            and ``255`` marks valid pixels.

        Examples:
            - The mask of a no-data raster is ``0`` exactly at the no-data cells:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.array([[1.0, -9999.0, 3.0, 4.0]] * 4, dtype="float32")
                >>> ds = Dataset.create_from_array(
                ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, no_data_value=-9999.0,
                ... )
                >>> mask = ds.read_masks(0)
                >>> mask.shape
                (4, 4)
                >>> bool((mask[:, 1] == 0).all())
                True

                ```
        """
        if window is None:
            read_args: tuple = ()
        else:
            clamped = window.crop(self._ds.rows, self._ds.columns)
            if clamped is None:
                raise OutOfBoundsError(
                    f"window {window} lies entirely outside the raster "
                    f"({self._ds.rows}x{self._ds.columns})."
                )
            read_args = clamped.to_read_args()
        bands = [band] if band is not None else range(self._ds.band_count)
        masks = [
            np.asarray(self._ds._iloc(index).GetMaskBand().ReadAsArray(*read_args))
            for index in bands
        ]
        result = masks[0] if band is not None else np.stack(masks)
        return result

    def create_mask_band(self, *, per_dataset: bool = True) -> None:
        """Create a mask band on the dataset.

        Args:
            per_dataset: ``True`` (default) creates a single mask shared by every
                band (``GMF_PER_DATASET``); ``False`` creates a per-band mask.

        Raises:
            ReadOnlyError: The dataset is opened read-only.

        Examples:
            - After creating a per-dataset mask, the flags report it:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> import tempfile, os
                >>> path = os.path.join(tempfile.mkdtemp(), "m.tif")
                >>> Dataset.create_from_array(
                ...     np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
                ... ).to_file(path)
                >>> ds = Dataset.read_file(path, read_only=False)
                >>> ds.create_mask_band()
                >>> ds.mask_flags().per_dataset
                True

                ```
        """
        if self._ds.access == "read_only":
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                "read_only=False to create a mask band."
            )
        self._ds.raster.CreateMaskBand(gdal.GMF_PER_DATASET if per_dataset else 0)

    def footprint(
        self: Dataset,
        band: int = 0,
        exclude_values: list[Any] | None = None,
    ) -> GeoDataFrame | None:
        """Extract the real coverage of the values in a certain band.

        Args:
            band (int):
                Band index. Default is 0.
            exclude_values (List[Any] | None):
                If you want to exclude a certain value in the raster with another value inter the two values as a
                list of tuples a [(value_to_be_exclude_valuesd, new_value)].

                - Example of exclude_values usage:

                  ```python
                  >>> exclude_values = [0]

                  ```

                - This parameter is introduced particularly in the case of rasters that has the no_data_value stored in
                  the `no_data_value` property does not match the value stored in the band, so this option can correct
                  this behavior.

        Returns:
            GeoDataFrame:
                - geodataframe containing the polygon representing the extent of the raster. the extent column should
                  contain a value of 2 only.
                - if the dataset had separate polygons, each polygon will be in a separate row.

        Examples:
            - The following raster dataset has flood depth stored in its values, and the non-flooded cells are filled with
              zero, so to extract the flood extent, we need to exclude the zero flood depth cells.

              ```python
              >>> dataset = Dataset.read_file("examples/data/geotiff/rhine-flood.tif")
              >>> dataset.plot()
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            ![dataset-footprint-rhine-flood](./../../_images/dataset/dataset-footprint-rhine-flood.png)

            - Now, to extract the footprint of the dataset band, we need to specify the `exclude_values` parameter with the
              value of the non-flooded cells.

              ```python
              >>> extent = dataset.footprint(band=0, exclude_values=[0])
              >>> print(extent)
                 Band_1                                           geometry
              0     2.0  POLYGON ((4070974.182 3181069.473, 4070974.182...
              1     2.0  POLYGON ((4077674.182 3181169.473, 4077674.182...
              2     2.0  POLYGON ((4091174.182 3169169.473, 4091174.182...
              3     2.0  POLYGON ((4088574.182 3176269.473, 4088574.182...
              4     2.0  POLYGON ((4082974.182 3167869.473, 4082974.182...
              5     2.0  POLYGON ((4092274.182 3168269.473, 4092274.182...
              6     2.0  POLYGON ((4072474.182 3181169.473, 4072474.182...

              >>> extent.plot()
              <Axes: >

              ```

            ![dataset-footprint-rhine-flood-extent](./../../_images/dataset/dataset-footprint-rhine-flood-extent.png)

        """
        arr = self._ds.read_array(band=band)
        no_data_val = self._ds.no_data_value[band]

        if no_data_val is None:
            if not (np.isnan(arr)).any():
                self._ds.logger.warning(
                    "The nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the no_data_value stored in the raster is"
                    " not correct"
                )
        else:
            if not (np.isclose(arr, no_data_val, rtol=0.00001)).any():
                self._ds.logger.warning(
                    "the nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the no_data_value stored in the raster is"
                    " not correct"
                )
        # if you want to exclude_values any value in the raster
        if exclude_values:
            for val in exclude_values:
                try:
                    # in case the val2 is None, and the array is int type, the following line will give error as None
                    # is considered as float
                    arr[np.isclose(arr, val)] = no_data_val
                except TypeError:
                    arr = arr.astype(np.float32)
                    arr[np.isclose(arr, val)] = no_data_val

        # replace all the values with 2
        if no_data_val is None:
            # check if the whole raster is full of no_data_value
            if (np.isnan(arr)).all():
                self._ds.logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isnan(arr)] = 2
        else:
            # check if the whole raster is full of no_data_value
            if (np.isclose(arr, no_data_val, rtol=0.00001)).all():
                self._ds.logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isclose(arr, no_data_val, rtol=0.00001)] = 2
        new_dataset = self._ds.create_from_array(
            arr,
            geo=self._ds.geotransform,
            epsg=self._ds.epsg,
            no_data_value=self._ds.no_data_value,
        )
        # then convert the raster into polygon
        gdf = new_dataset.cluster2(band=band)
        gdf.rename(columns={"Band_1": self._ds.band_names[band]}, inplace=True)

        return gdf

    @staticmethod
    def normalize(array: np.ndarray) -> np.typing.NDArray:
        """Normalize numpy arrays into scale 0.0-1.0.

        Args:
            array (np.ndarray): Numpy array to normalize.

        Returns:
            np.ndarray: Normalized array.
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return np.asarray(val)

    @staticmethod
    def _rescale(array: np.ndarray, min_value: float, max_value: float) -> np.typing.NDArray:
        val = (array - min_value) / (max_value - min_value)
        return val

    def get_histogram(
        self: Dataset,
        band: int = 0,
        bins: int = 6,
        min_value: float | None = None,
        max_value: float | None = None,
        include_out_of_range: bool = False,
        approx_ok: bool = False,
    ) -> tuple[list, list[tuple[Any, Any]]]:
        """Get histogram.

        Args:
            band (int, optional):
                Band index. Default is 1.
            bins (int, optional):
                Number of bins. Default is 6.
            min_value (float, optional):
                Minimum value. Default is None.
            max_value (float, optional):
                Maximum value. Default is None.
            include_out_of_range (bool, optional):
                If True, add out-of-range values into the first and last buckets. Default is False.
            approx_ok (bool, optional):
                If True, compute an approximate histogram by using subsampling or overviews. Default is False.

        Returns:
            tuple[list, list[tuple[Any, Any]]]:
                Histogram values and bin edges.

        Hint:
            - The value of the histogram will be stored in an xml file by the name of the raster file with the extension
                of .aux.xml.

            - The content of the file will be like the following:
              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Histograms>
                        <HistItem>
                          <HistMin>0</HistMin>
                          <HistMax>88</HistMax>
                          <BucketCount>6</BucketCount>
                          <IncludeOutOfRange>0</IncludeOutOfRange>
                          <Approximate>0</Approximate>
                          <HistCounts>75|6|0|4|2|1</HistCounts>
                        </HistItem>
                      </Histograms>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Create `Dataset` consists of 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 12, size=(10, 10))
              >>> print(arr)    # doctest: +SKIP
              [[ 4  1  1  2  6  9  2  5  1  8]
               [ 1 11  5  6  2  5  4  6  6  7]
               [ 5  2 10  4  8 11  4 11 11  1]
               [ 2  3  6  3  1  5 11 10 10  7]
               [ 8  2 11  3  1  3  5  4 10 10]
               [ 1  2  1  6 10  3  6  4  2  8]
               [ 9  5  7  9  7  8  1 11  4  4]
               [ 7  7  2  2  5  3  7  2  9  9]
               [ 2 10  3  2  1 11  5  9  8 11]
               [ 1  5  6 11  3  3  8  1  2  1]]
               >>> top_left_corner = (0, 0)
               >>> cell_size = 0.05
               >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

               ```

            - Now, let's get the histogram of the first band using the `get_histogram` method with the default
                parameters:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0)
                >>> print(hist)  # doctest: +SKIP
                [28, 17, 10, 15, 13, 7]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 2.67), (2.67, 4.34), (4.34, 6.0), (6.0, 7.67), (7.67, 9.34), (9.34, 11.0)]

                ```
            - we can also exclude values from the histogram by using the `min_value` and `max_value`:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, min_value=5, max_value=10)
                >>> print(hist)  # doctest: +SKIP
                [10, 8, 7, 7, 6, 0]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 1.835), (1.835, 2.67), (2.67, 3.5), (3.5, 4.34), (4.34, 5.167), (5.167, 6.0)]

                ```
            - For datasets with big dimensions, computing the histogram can take some time; approximating the computation
                of the histogram can save a lot of computation time. When using the parameter `approx_ok` with a `True`
                value the histogram will be calculated from resampling the band or from the overviews if they exist.
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, approx_ok=True)
                >>> print(hist)  # doctest: +SKIP
                [28, 17, 10, 15, 13, 7]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 2.67), (2.67, 4.34), (4.34, 6.0), (6.0, 7.67), (7.67, 9.34), (9.34, 11.0)]

                ```
            - As you see for small datasets, the approximation of the histogram will be the same as without approximation.

        """
        band_obj = self._ds._iloc(band)
        min_val, max_val = band_obj.ComputeRasterMinMax()
        if min_value is None:
            min_value = min_val
        if max_value is None:
            max_value = max_val

        bin_width = (max_value - min_value) / bins
        ranges = [
            (min_val + i * bin_width, min_val + (i + 1) * bin_width)
            for i in range(bins)
        ]

        hist = band_obj.GetHistogram(
            min=min_value,
            max=max_value,
            buckets=bins,
            include_out_of_range=include_out_of_range,
            approx_ok=approx_ok,
        )
        return hist, ranges

    def plot_histogram(
        self,
        band: int = 0,
        bins: int = 15,
        exclude_value: Any | None = None,
        ax: Any | None = None,
        **kwargs: Any,
    ):
        """Plot the value distribution of a band as a histogram.

        Backed by cleopatra's
        :class:`~cleopatra.statistical_glyph.StatisticalGlyph`. The band is
        read into memory, the band's no-data value and ``exclude_value``
        (and any ``NaN`` for floating-point bands) are dropped, and only the
        remaining valid samples reach the glyph. Requires the ``[viz]`` extra.

        Args:
            band (int, optional):
                Band index to read. Default is ``0``.
            bins (int, optional):
                Number of histogram bins. Default is ``15``.
            exclude_value (Any, optional):
                An extra value to drop from the samples, in addition to the
                band's no-data value and ``NaN``. Default is ``None``.
            ax (matplotlib.axes.Axes, optional):
                Axes to draw on. A new figure/axes is created when ``None``.
            **kwargs:
                Style options forwarded to the ``StatisticalGlyph``
                constructor, filtered via
                :meth:`StatisticalGlyph.filter_kwargs` so only accepted keys
                are passed.

        Returns:
            tuple:
                ``(fig, ax, hist)`` from
                :meth:`StatisticalGlyph.histogram` — the
                :class:`matplotlib.figure.Figure`, the
                :class:`matplotlib.axes.Axes`, and the histogram ``dict``.

        Raises:
            ValueError: If the band has no valid samples left after masking
                the no-data value, ``exclude_value``, and ``NaN``.

        Examples:
            - Plot the distribution of a band and reuse the matplotlib
              handles (tagged ``+SKIP`` — needs the ``[viz]`` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(100, dtype="float32").reshape(10, 10)
                >>> ds = Dataset.create_from_array(arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326)
                >>> fig, ax, hist = ds.plot_histogram(band=0, bins=8)  # doctest: +SKIP
                >>> _ = ax.set_title("band 0 distribution")  # doctest: +SKIP
                ```
            - Drop a sentinel value before binning:

                ```python
                >>> arr = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]], dtype="float32")
                >>> ds = Dataset.create_from_array(arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326)
                >>> fig, ax, hist = ds.plot_histogram(band=0, exclude_value=99.0)  # doctest: +SKIP
                ```
        """
        require_cleopatra()
        from cleopatra.statistical_glyph import StatisticalGlyph

        arr = self._ds.read_array(band=band).flatten()
        no_data_value = self._ds.no_data_value[band]
        mask = np.ones(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isnan(arr)
        if no_data_value is not None and not (
            isinstance(no_data_value, float) and np.isnan(no_data_value)
        ):
            mask &= arr != no_data_value
        if exclude_value is not None:
            mask &= arr != exclude_value
        values = arr[mask]
        if values.size == 0:
            raise ValueError(
                f"Band {band} has no valid samples to histogram after masking "
                "no-data / exclude_value / NaN."
            )
        glyph = StatisticalGlyph(
            values, ax=ax, **StatisticalGlyph.filter_kwargs(kwargs)
        )
        result = glyph.histogram(bins=bins)
        return result

    def to_image(
        self,
        band: int = 0,
        cmap: str = "viridis",
        exclude_value: Any | None = None,
    ):
        """Export a band as a colour-mapped RGB image.

        Reads the band, masks the no-data value (and an optional
        ``exclude_value``), applies a matplotlib colormap via cleopatra's
        :meth:`ArrayGlyph.apply_colormap`, and returns the result as a
        :class:`PIL.Image.Image`. Masked / no-data pixels are rendered with
        the colormap's "bad" fill colour. Requires the ``[viz]`` extra.

        Args:
            band (int, optional):
                Band index to export. Default is ``0``.
            cmap (str, optional):
                Matplotlib colormap name. Default is ``"viridis"``.
            exclude_value (Any, optional):
                An extra value to mask out, in addition to the band's
                no-data value. Default is ``None``.

        Returns:
            PIL.Image.Image:
                An RGB image of the colour-mapped band, the same width and
                height as the raster band.

        Raises:
            ValueError: If the band has no valid (non-nodata) pixels left
                after masking the no-data value, ``exclude_value``, and
                ``NaN`` — there is then nothing to colour-map.

        Examples:
            - Export a band as a viridis thumbnail, inspect its size, and
              save it to disk (tagged ``+SKIP`` — needs the ``[viz]`` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> arr = np.arange(48, dtype="float32").reshape(6, 8)
                >>> ds = Dataset.create_from_array(arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326)
                >>> img = ds.to_image(band=0, cmap="viridis")  # doctest: +SKIP
                >>> img.size  # (width, height) == (columns, rows)  # doctest: +SKIP
                (8, 6)
                >>> img.save("band0.png")  # doctest: +SKIP
                ```
        """
        require_cleopatra()
        from cleopatra.array_glyph import ArrayGlyph

        arr = self._ds.read_array(band=band)
        no_data_value = self._ds.no_data_value[band]
        exclude: list = []
        if no_data_value is not None and not (
            isinstance(no_data_value, float) and np.isnan(no_data_value)
        ):
            exclude.append(no_data_value)
        if exclude_value is not None:
            exclude.append(exclude_value)
        valid = np.ones(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            valid &= ~np.isnan(arr)
        for excluded in exclude:
            valid &= arr != excluded
        if not valid.any():
            raise ValueError(
                f"Band {band} has no valid (non-nodata) pixels to render to "
                "an image after masking no-data / exclude_value / NaN."
            )
        glyph = ArrayGlyph(arr, exclude_value=exclude if exclude else np.nan)
        image = glyph.to_image(glyph.apply_colormap(cmap))
        return image

    def plot_vector_field(
        self,
        u_band: int = 0,
        v_band: int = 1,
        kind: str = "quiver",
        ax: Any | None = None,
        **kwargs: Any,
    ):
        """Plot two bands as a 2-component vector field.

        Reads ``u_band`` and ``v_band`` as the vector components over the
        dataset's cell-centre coordinate grid (built from the geotransform)
        and renders them via cleopatra's
        :class:`~cleopatra.vector_glyph.VectorGlyph` as arrows, wind barbs,
        or streamlines, coloured by vector magnitude. Requires the ``[viz]``
        extra.

        The grid is taken from the dataset's 1-D ``x``/``y`` cell-centre
        arrays, so an **axis-aligned (north-up, unrotated)** geotransform is
        assumed — as elsewhere in pyramids' extent-based plotting. ``v`` is
        treated as the northward (``+y``) component. Because ``streamplot``
        requires strictly-increasing coordinates while a north-up raster's
        ``y`` is descending, the axis is flipped to ascending and the data
        rows/cols are mirrored to match; this is a pure relabelling, so each
        vector stays at its true location for every ``kind``.

        Args:
            u_band (int, optional):
                Band index of the x-component (``u``). Default is ``0``.
            v_band (int, optional):
                Band index of the y-component (``v``). Default is ``1``.
            kind (str, optional):
                Render kind: ``"quiver"`` (default), ``"barbs"``, or
                ``"streamplot"``.
            ax (matplotlib.axes.Axes, optional):
                Axes to draw on. A new figure/axes is created when ``None``.
            **kwargs:
                Style options forwarded to the ``VectorGlyph`` constructor,
                filtered via :meth:`VectorGlyph.filter_kwargs` (e.g.
                ``density``, ``scale``, ``cmap``, ``add_colorbar``). Pass
                ``add_colorbar=False`` when composing onto a shared map.

        Returns:
            tuple:
                ``(fig, ax, im)`` from :meth:`VectorGlyph.plot` — the
                :class:`matplotlib.figure.Figure`, the
                :class:`matplotlib.axes.Axes`, and the mappable coloured by
                vector magnitude.

        Raises:
            ValueError: If ``u_band`` or ``v_band`` is out of range for the
                dataset, or if ``kind`` is not one of ``"quiver"``,
                ``"barbs"``, or ``"streamplot"``.

        Examples:
            - Render a two-band ``(u, v)`` stack as arrows (tagged ``+SKIP``
              — needs the ``[viz]`` extra):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> rng = np.random.default_rng(0)
                >>> uv = rng.standard_normal((2, 6, 6)).astype("float32")
                >>> ds = Dataset.create_from_array(uv, top_left_corner=(0, 0), cell_size=1.0, epsg=4326)
                >>> fig, ax, im = ds.plot_vector_field(u_band=0, v_band=1, kind="quiver")  # doctest: +SKIP
                ```
            - Draw streamlines without the magnitude colorbar (e.g. to add a
              shared one later):

                ```python
                >>> fig, ax, im = ds.plot_vector_field(kind="streamplot", add_colorbar=False)  # doctest: +SKIP
                ```
        """
        require_cleopatra()
        from cleopatra.vector_glyph import VectorGlyph

        band_count = self._ds.band_count
        for name, idx in (("u_band", u_band), ("v_band", v_band)):
            if idx < 0 or idx >= band_count:
                raise ValueError(
                    f"{name}={idx} is out of range for a {band_count}-band "
                    "dataset; plot_vector_field needs two in-range bands "
                    "(u, v components)."
                )
        u = self._ds.read_array(band=u_band)
        v = self._ds.read_array(band=v_band)
        x = self._ds.x
        y = self._ds.y
        # matplotlib's ``streamplot`` requires strictly-increasing 1-D
        # coordinates, but a north-up raster's ``y`` (and occasionally ``x``)
        # is descending. Flip the axis to ascending and mirror the data
        # rows/cols so the field stays spatially correct for every kind
        # (``quiver``/``barbs`` are direction-agnostic; ``streamplot`` is not).
        if y[0] > y[-1]:
            y = y[::-1]
            u = u[::-1, :]
            v = v[::-1, :]
        if x[0] > x[-1]:
            x = x[::-1]
            u = u[:, ::-1]
            v = v[:, ::-1]
        xx, yy = np.meshgrid(x, y)
        glyph = VectorGlyph(xx, yy, u, v, ax=ax, **VectorGlyph.filter_kwargs(kwargs))
        result = glyph.plot(kind=kind)
        return result

    def plot(
        self,
        band: int,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        percentile: int | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> ArrayGlyph:
        """Plot the values/overviews of a given band.

        This is the generic rendering engine. It assumes ``band`` has already been resolved
        by the caller (typically a per-class facade such as :meth:`Dataset.plot` or
        :meth:`NetCDF.plot`). It does **not** apply any band-resolution policy (no RGB
        heuristic, no `ColorInterpretation` lookup, no default-to-zero fallback) \u2014 those
        are dataset-type-specific decisions that belong on the facades.

        The plot function uses the `cleopatra` as a backend to plot the raster data, for more information check
        [ArrayGlyph](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/#cleopatra.array_glyph.ArrayGlyph.plot).

        Implementation note: this method is a thin caller around the
        shared :func:`pyramids.dataset._plot_helpers.render_array`
        helper. It resolves the data (``arr``), extent, exclude value,
        and curvilinear coords from the underlying ``Dataset``, then
        forwards to ``render_array(..., mode="plot", ...)`` for a
        single 2-D slice or ``mode="facet"`` when ``NetCDF.plot``
        injects a pre-built ``_facet_stack`` and ``facet_kwargs``.
        ``DatasetCollection.plot`` reuses the same helper with
        ``mode="animate"``. The shared helper owns the actual
        ``ArrayGlyph`` construction and dispatch — see the module
        docstring of :mod:`pyramids.dataset._plot_helpers` for the
        three-mode contract.

        Args:
            band (int):
                Concrete band index to render. Must be provided \u2014 the engine does not resolve
                bands.
            exclude_value (Any, optional):
                Value to exclude from the plot. Default is None.
            rgb (List[int], optional):
                The indices of the red, green, and blue bands in the `Dataset`. the `rgb` parameter can be a list of
                three values, or a list of four values if the alpha band is also included. Only meaningful for
                Sentinel-style multi-band rasters; pass-through to cleopatra.
            surface_reflectance (int, optional):
                Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff (List, optional):
                clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
                and scale them back to between 0 and 1). Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            percentile: int
                The percentile value to be used for scaling.
            basemap (bool or str, optional):
                If True, add an OpenStreetMap basemap underneath the plot. If a string, use it as
                the tile provider name (e.g. "CartoDB.Positron"). Default is None (no basemap).
                Requires the [viz] extra (mercantile, xyzservices, Pillow).
        kwargs:
                | Parameter                   | Type                | Description |
                |-----------------------------|---------------------|-------------|
                | `points`                    | array               | 3 column array with the first column as the value to display for the point, the second as the row index, and the third as the column index in the array. The second and third columns tell the location of the point. |
                | `point_color`               | str                 | Color of the point. |
                | `point_size`                | Any                 | Size of the point. |
                | `pid_color`                 | str                 | Color of the annotation of the point. Default is blue. |
                | `pid_size`                  | Any                 | Size of the point annotation. |
                | `figsize`                   | tuple, optional     | Figure size. Default is `(8, 8)`. |
                | `title`                     | str, optional       | Title of the plot. Default is `'Total Discharge'`. |
                | `title_size`                | int, optional       | Title size. Default is `15`. |
                | `orientation`               | str, optional       | Orientation of the color bar (`horizontal` or `vertical`). Default is `'vertical'`. |
                | `rotation`                  | number, optional    | Rotation of the color bar label. Default is `-90`. |
                | `cbar_length`               | float, optional     | Ratio to control the height of the color bar. Default is `0.75`. |
                | `ticks_spacing`             | int, optional       | Spacing between color bar ticks. Default is `2`. |
                | `cbar_label_size`           | int, optional       | Size of the color bar label. Default is `12`. |
                | `cbar_label`                | str, optional       | Label of the color bar. Default is `'Discharge m\u00b3/s'`. |
                | `color_scale`               | str, optional       | Color-scale mode. One of `"linear"`, `"power"`, `"sym-lognorm"`, `"boundary-norm"`, `"midpoint"` (case-insensitive), or a `cleopatra.styles.ColorScale` member. Integer codes are no longer accepted. Default is `"linear"`. |
                | `gamma`                     | float, optional     | Exponent for the `"power"` color scale. Default is `1/2`. |
                | `line_threshold`            | float, optional     | `linthresh` for the `"sym-lognorm"` color scale. Default is `0.0001`. |
                | `line_scale`                | float, optional     | `linscale` for the `"sym-lognorm"` color scale. Default is `0.001`. |
                | `bounds`                    | list, optional      | Discrete bounds for the `"boundary-norm"` color scale. Default is `None`. |
                | `midpoint`                  | float, optional     | Midpoint value for the `"midpoint"` color scale. Default is `0`. |
                | `cmap`                      | str, optional       | Color map style. Default is `'coolwarm_r'`. |
                | `display_cell_value`        | bool, optional      | Whether to display cell values as text. |
                | `num_size`                  | int, optional       | Size of numbers plotted on top of each cell. Default is `8`. |
                | `background_color_threshold`| float or int, optional | Threshold for deciding text color over cells: if value > threshold -> black text; else white text. If `None`, max value / 2 is used. Default is `None`. |
                | `add_colorbar`              | bool, optional      | Whether to draw the colour bar. Default is `True`. When `False`, no colorbar is created and the returned glyph's `cbar` is `None`. |
        Returns:
            ArrayGlyph:
                A cleopatra ``ArrayGlyph`` wrapping the rendered figure. The underlying matplotlib
                primitives are exposed on the glyph \u2014 use them as the escape hatch when you need
                to further customise the plot with raw matplotlib calls:

                - ``cleo.fig`` / ``cleo.ax`` \u2014 the :class:`matplotlib.figure.Figure` and
                  :class:`matplotlib.axes.Axes`.
                - ``cleo.im`` \u2014 the colour-mapped mappable, populated for every ``kind=``
                  (imshow/pcolormesh/contour/contourf); e.g. ``cleo.im.set_clim(0, 100)``.
                - ``cleo.cbar`` \u2014 the auto-created :class:`matplotlib.colorbar.Colorbar`, or
                  ``None`` when ``add_colorbar=False`` (or for RGB renders).

                For the full ``ArrayGlyph`` API see the
                [ArrayGlyph reference](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/).
        Examples:
            - Plot a certain band:
              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326)
              >>> dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using power scale.
              ```python
              >>> dataset.plot(band=0, color_scale="power")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using SymLogNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="sym-lognorm")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using PowerNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="boundary-norm", bounds=[0, 0.2, 0.4, 0.6, 0.8, 1])
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using BoundaryNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="midpoint")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
        """
        no_data_value = [np.nan if i is None else i for i in self._ds.no_data_value]
        # `coords` is the PR-3 curvilinear kwarg; the helper handles the
        # mutually-exclusive `extent` swap. `facet_kwargs` (PR-4) is
        # forwarded by `NetCDF.plot` to switch the helper to the
        # `mode="facet"` branch; the pre-built stack arrives alongside as
        # `_facet_stack` and its spatial extent as `_extent` (the facet
        # stack is *injected*, not read from `self._ds`, so the engine
        # can't derive the extent from `self._ds.bbox` — the caller must
        # supply it). `_chunks` (PR-5) is injected by `NetCDF.plot` to
        # switch the static-plot read path to the dask-backed lazy read;
        # only the rendered slice is materialised.
        coords = kwargs.pop("coords", None)
        facet_kwargs = kwargs.pop("facet_kwargs", None)
        facet_stack = kwargs.pop("_facet_stack", None)
        injected_extent = kwargs.pop("_extent", None)
        chunks = kwargs.pop("_chunks", None)
        mode = "facet" if facet_kwargs else "plot"
        if mode == "facet":
            arr = facet_stack
        elif chunks is not None:
            # Lazy read path: build a dask array of the variable, then
            # materialise only the requested slice via `.compute()`.
            # `read_array(chunks=...)` is only meaningful on NetCDF —
            # plain Dataset doesn't support `chunks`. The kwarg arrives
            # here only because NetCDF.plot injected it, so the call is
            # safe to issue.
            lazy = self._ds.read_array(chunks=chunks)
            if hasattr(lazy, "compute"):
                if lazy.ndim > 2:
                    # `read_array(chunks=...)` returns the variable's
                    # native `(d0, d1, ..., rows, cols)` shape, whereas
                    # the eager `read_array()` flattens the non-spatial
                    # dims into a single bands axis. Match that flatten so
                    # `band` indexes the same slice. The reshape stays
                    # lazy — `read_array(chunks=...)` already chunks the
                    # non-spatial dims at size 1, so it's a pure relabel —
                    # and only the chosen band's chunks get computed.
                    lazy = lazy.reshape(-1, *lazy.shape[-2:])
                    arr = np.asarray(lazy[band].compute())
                else:
                    arr = np.asarray(lazy.compute())
            else:
                arr = lazy if band is None else lazy[band]
        else:
            # When ``rgb`` is supplied, cleopatra's ArrayGlyph needs the full
            # multi-band ``(bands, rows, cols)`` array so it can pick the
            # colour channels itself. In all other cases we render just the
            # requested band as a 2-D array.
            read_band = None if rgb is not None else band
            if overview:
                arr = self._ds.read_overview_array(
                    band=read_band,
                    overview_index=(
                        overview_index if overview_index is not None else 0
                    ),
                )
            else:
                arr = self._ds.read_array(band=read_band)
        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )
        ax = kwargs.pop("ax", None)
        fig = kwargs.pop("fig", None)
        # On the self-read paths (`mode="plot"` / `_chunks`) the data and
        # the extent both come from `self._ds`. On the injected-stack path
        # (`mode="facet"`) the caller passes `_extent` so the panels are
        # placed at the stack's own spatial domain rather than implicitly
        # trusting that it matches `self._ds.bbox`.
        effective_extent = (
            injected_extent if injected_extent is not None else self._ds.bbox
        )
        return render_array(
            arr=arr,
            extent=effective_extent,
            coords=coords,
            exclude_value=exclude_value,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            percentile=percentile,
            mode=mode,
            facet_kwargs=facet_kwargs,
            ax=ax,
            fig=fig,
            basemap=basemap,
            basemap_epsg=self._ds.epsg,
            **kwargs,
        )

    @staticmethod
    def _process_color_table(color_table: DataFrame) -> DataFrame:
        require_cleopatra()
        from cleopatra.colors import Colors

        # if the color_table does not contain the red, green, and blue columns, assume it has one column with
        # the color as hex and then, convert the color to rgb.
        if all(elem in color_table.columns for elem in ["red", "green", "blue"]):
            color_df = color_table.loc[:, ["values", "red", "green", "blue"]]
        elif "color" in color_table.columns:
            color = Colors(color_table["color"].tolist())
            color_rgb = color.to_rgb(normalized=False)
            color_df = DataFrame(columns=["values"])
            color_df["values"] = color_table["values"].to_list()
            color_df.loc[:, ["red", "green", "blue"]] = color_rgb
        else:
            raise ValueError(
                f"color_table must contain either red, green, blue, or color columns. given columns are: "
                f"{color_table.columns}"
            )
        if "alpha" not in color_table.columns:
            color_df.loc[:, "alpha"] = 255
        else:
            color_df.loc[:, "alpha"] = color_table["alpha"]
        return color_df
