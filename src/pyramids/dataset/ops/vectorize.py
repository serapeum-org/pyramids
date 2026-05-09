"""Vectorization, clustering, and translate mixin for Dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids.base._errors import CRSError
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset
    from pyramids.feature import FeatureCollection

logger = logging.getLogger(__name__)


def rasterize_features(
    features: FeatureCollection,
    dataset_cls: type[Dataset],
    *,
    cell_size: Any | None = None,
    template: Dataset | None = None,
    column_name: str | list[str] | None = None,
) -> Dataset:
    """Burn a :class:`FeatureCollection` into a new raster.

    Free-function form of :meth:`Dataset.from_features`. `dataset_cls`
    is the class to instantiate (passed in by the classmethod
    forwarder so this module does not need a runtime import of
    :class:`~pyramids.dataset.Dataset`).

    Args:
        features: The vector to rasterize.
        dataset_cls: The :class:`~pyramids.dataset.Dataset` class (or a
            subclass) used to allocate the output raster.
        cell_size: Cell size for the new raster. Required unless
            `template` is given.
        template: Optional template raster. When supplied, the output
            inherits its geotransform and no-data value.
        column_name: Attribute column(s) to burn as band values.
            `None` burns every non-geometry column as a separate band.

    Returns:
        Dataset: The burned raster. When the burn column is integer
        dtyped and the template's no-data is `None`, the output
        raster's no-data is `dataset_cls.default_no_data_value`
        rather than `NaN` â€” NaN cannot be stored in integer rasters
        without silent coercion.

    Raises:
        ValueError: If `cell_size` is missing or non-positive, or
            if `column_name` is empty / references missing columns.
        TypeError: If `template` is not a Dataset, or
            `column_name` is not `str` / `list` / `None`.
        CRSError: If the FeatureCollection has no CRS, or
            `template.epsg!= features.epsg`.
    """
    if cell_size is None and template is None:
        raise ValueError("You have to enter either cell size or Dataset object.")
    if cell_size is not None and cell_size <= 0:
        raise ValueError(f"cell_size must be positive; got {cell_size!r}.")
    if column_name is not None and not isinstance(column_name, (str, list)):
        raise TypeError(
            f"column_name must be str, list[str], or None; "
            f"got {type(column_name).__name__}."
        )

    ds_epsg = features.epsg
    if ds_epsg is None:
        raise CRSError(
            "FeatureCollection must have a CRS before rasterisation. "
            "Set one via `fc.set_crs('EPSG:...')` or construct the FC "
            "with `crs='EPSG:...'`."
        )
    if template is not None:
        if not isinstance(template, dataset_cls):
            raise TypeError(
                "The template parameter must be a pyramids Dataset "
                "(see pyramids.dataset.Dataset.read_file)."
            )
        if template.epsg != ds_epsg:
            raise CRSError(
                f"Dataset and vector are not the same EPSG. "
                f"{template.epsg} != {ds_epsg}"
            )
        xmin, ymax = template.top_left_corner
        no_data_value = (
            template.no_data_value[0]
            if template.no_data_value[0] is not None
            else np.nan
        )
        rows = template.rows
        columns = template.columns
        cell_size = template.cell_size
    else:
        xmin, ymin, xmax, ymax = features.total_bounds
        no_data_value = dataset_cls.default_no_data_value
        columns = int(np.ceil((xmax - xmin) / cell_size))
        rows = int(np.ceil((ymax - ymin) / cell_size))

    if column_name is None:
        column_name = [c for c in features.columns if c != "geometry"]

    if isinstance(column_name, list):
        if not column_name:
            raise ValueError(
                "column_name list must be non-empty. Pass None to "
                "burn every non-geometry column, or name at least "
                "one column."
            )
        missing = [c for c in column_name if c not in features.columns]
        if missing:
            raise ValueError(
                f"column_name references columns not in the "
                f"FeatureCollection: {missing}. Available columns: "
                f"{list(features.columns)}."
            )
        # Multi-band burn: every band shares the same dataset dtype, so
        # promote to the smallest dtype that can hold every selected
        # column without lossy cast. Mixed [int8, float32] â†’ float32,
        # mixed [int8, int16] â†’ int16, etc. Previously the dtype was
        # taken from column_name[0] only, which silently truncated
        # wider columns.
        numpy_dtype = np.result_type(*[features.dtypes[c] for c in column_name])
    else:
        if column_name not in features.columns:
            raise ValueError(
                f"column_name {column_name!r} is not in the "
                f"FeatureCollection. Available columns: "
                f"{list(features.columns)}."
            )
        numpy_dtype = features.dtypes[column_name]

    # Integer raster dtypes cannot represent NaN. If the template
    # supplied None as no_data_value (defaulted to NaN above) and the
    # burn column's dtype is integer, fall back to the class default
    # sentinel so GDAL does not silently coerce NaN into an arbitrary
    # integer value.
    if np.issubdtype(numpy_dtype, np.integer):
        try:
            if np.isnan(no_data_value):
                no_data_value = dataset_cls.default_no_data_value
        except (TypeError, ValueError):
            pass

    bands_count = 1 if not isinstance(column_name, list) else len(column_name)
    dataset_n = dataset_cls.create(
        float(cell_size),
        rows,
        columns,
        str(numpy_dtype),
        bands_count,
        (xmin, ymax),
        ds_epsg,
        no_data_value,
    )

    with _feature_ogr.as_datasource(features, gdal_dataset=True) as vector_ds:
        for ind in range(bands_count):
            attribute = (
                column_name[ind] if isinstance(column_name, list) else column_name
            )
            rasterize_opts = gdal.RasterizeOptions(
                bands=[ind + 1],
                burnValues=None,
                attribute=attribute,
                allTouched=True,
            )
            gdal.Rasterize(dataset_n.raster, vector_ds, options=rasterize_opts)

    return dataset_n
