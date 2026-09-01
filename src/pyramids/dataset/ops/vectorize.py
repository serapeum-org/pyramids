"""Vectorization, clustering, and translate mixin for Dataset."""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids.base._errors import CRSError
from pyramids.base.georeference import GeoReference
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
    snap_to_template: bool = False,
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
            inherits its geotransform and no-data value (the vector is
            burned onto the template's fixed grid). Features that fall
            entirely outside the template extent, or an empty
            FeatureCollection, produce an all-nodata raster and emit a
            ``UserWarning`` (issue #46).
        snap_to_template: When ``True`` (requires `template`), the output
            keeps the template's cell size and grid alignment but is sized
            to the features' bounds snapped outward onto the template's
            grid lines — a small, template-co-registered raster instead of
            one spanning the whole template (issue #46). Requires a square,
            axis-aligned template and a FeatureCollection with valid
            (non-NaN) geometry bounds. The output is sized to the features,
            so a fine-celled template with far-apart features can allocate a
            large raster.
        column_name: Attribute column(s) to burn as band values.
            `None` burns every non-geometry column as a separate band.

    Returns:
        Dataset: The burned raster. When the burn column is integer
        dtyped and the template's no-data is `None`, the output
        raster's no-data is `dataset_cls.default_no_data_value`
        rather than `NaN` â€” NaN cannot be stored in integer rasters
        without silent coercion.

    Raises:
        ValueError: If `cell_size` is missing or non-positive, if
            `column_name` is empty / references missing columns, if
            `snap_to_template` is set without a `template`, or (in snap
            mode) the template is rotated / non-square or the features
            have no valid (non-NaN) geometry bounds.
        TypeError: If `template` is not a Dataset, or
            `column_name` is not `str` / `list` / `None`.
        CRSError: If the FeatureCollection has no CRS, or
            `template.epsg!= features.epsg`.
    """
    _validate_rasterize_args(cell_size, template, snap_to_template, column_name)

    ds_epsg = features.epsg
    if ds_epsg is None:
        raise CRSError(
            "FeatureCollection must have a CRS before rasterisation. "
            "Set one via `fc.set_crs('EPSG:...')` or construct the FC "
            "with `crs='EPSG:...'`."
        )

    xmin, ymax, rows, columns, cell_size, no_data_value = _resolve_raster_geometry(
        features, dataset_cls, template, cell_size, ds_epsg, snap_to_template
    )

    # Resolve (and validate) the burn columns before warning, so an invalid column_name
    # raises rather than emitting a warning about an output that is never produced.
    column_name, numpy_dtype = _resolve_burn_columns(features, column_name)

    # In snap mode the output is sized to the features, so it always covers them — the
    # outside-template warning applies only to the full-template mode.
    if (
        template is not None
        and not snap_to_template
        and _features_outside_template(features, template)
    ):
        warnings.warn(
            "FeatureCollection is empty or falls entirely outside the template extent; "
            "the output raster will likely be all-nodata. Check that the template covers "
            "the features and that both use the same CRS.",
            # stacklevel targets the caller of Dataset.from_features (the primary API);
            # a direct rasterize_features(...) call points one frame higher.
            stacklevel=3,
        )

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
        rows,
        columns,
        str(numpy_dtype),
        bands_count,
        geo_ref=GeoReference(
            top_left_corner=(xmin, ymax), cell_size=float(cell_size), epsg=ds_epsg
        ),
        no_data_value=no_data_value,
    )

    # An empty FeatureCollection has no layer field to burn (gdal.Rasterize would raise
    # "Failed to find field ..."); the freshly created raster is already all-nodata, so
    # skip the burn and return it (the warning above has already flagged the empty input).
    if len(features):
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


def _validate_rasterize_args(
    cell_size: Any | None,
    template: Dataset | None,
    snap_to_template: bool,
    column_name: str | list[str] | None,
) -> None:
    """Validate the scalar arguments of :func:`rasterize_features`.

    Raises:
        ValueError: Neither ``cell_size`` nor ``template`` given, a non-positive
            ``cell_size``, or ``snap_to_template`` without a ``template``.
        TypeError: ``column_name`` is not ``str`` / ``list`` / ``None``.
    """
    if cell_size is None and template is None:
        raise ValueError("You have to enter either cell size or Dataset object.")
    if cell_size is not None and cell_size <= 0:
        raise ValueError(f"cell_size must be positive; got {cell_size!r}.")
    if snap_to_template and template is None:
        raise ValueError("snap_to_template=True requires a template Dataset.")
    if column_name is not None and not isinstance(column_name, (str, list)):
        raise TypeError(
            f"column_name must be str, list[str], or None; "
            f"got {type(column_name).__name__}."
        )


def _resolve_raster_geometry(
    features: FeatureCollection,
    dataset_cls: type[Dataset],
    template: Dataset | None,
    cell_size: Any | None,
    ds_epsg: int,
    snap_to_template: bool = False,
) -> tuple[float, float, int, int, Any, Any]:
    """Resolve the output raster geometry for :func:`rasterize_features`.

    With a ``template`` the geometry (origin, size, cell size, no-data) is inherited from
    it after validating its type and CRS; with ``snap_to_template`` the template supplies
    only the cell size and grid alignment while the extent is the feature bounds snapped
    onto the template grid; otherwise it is derived from the feature bounds and the
    requested ``cell_size``.

    Returns:
        ``(xmin, ymax, rows, columns, cell_size, no_data_value)``.

    Raises:
        TypeError: ``template`` is not an instance of ``dataset_cls``.
        CRSError: ``template.epsg`` differs from the FeatureCollection's EPSG.
        ValueError: ``snap_to_template`` with a rotated/non-square template or an empty
            FeatureCollection.
    """
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
        if snap_to_template:
            xmin, ymax, rows, columns, cell_size = _snap_extent_to_template(
                features, template
            )
        else:
            xmin, ymax = template.top_left_corner
            rows = template.rows
            columns = template.columns
            cell_size = template.cell_size
        no_data_value = (
            template.no_data_value[0]
            if template.no_data_value[0] is not None
            else np.nan
        )
    else:
        xmin, ymin, xmax, ymax = features.total_bounds
        no_data_value = dataset_cls.default_no_data_value
        columns = int(np.ceil((xmax - xmin) / cell_size))
        rows = int(np.ceil((ymax - ymin) / cell_size))
    return xmin, ymax, rows, columns, cell_size, no_data_value


_SNAP_EPS = 1e-9
"""Cell-unit nudge so floor/ceil snapping is stable for grid-aligned bounds (issue #46)."""

_ROTATION_REL_TOL = 1e-6
"""A geotransform rotation term below this fraction of the pixel size is treated as zero."""


def _snap_extent_to_template(
    features: FeatureCollection, template: Dataset
) -> tuple[float, float, int, int, float]:
    """Size the output to the features, snapped onto the template's grid (issue #46).

    The output keeps the template's cell size and grid alignment (its pixel edges line up
    with the template's), but spans only the feature bounds snapped *outward* to the
    template's grid lines — a small raster co-registered with the template rather than one
    covering the whole template.

    Args:
        features: The vector being rasterised (must have valid, non-NaN bounds).
        template: A square, axis-aligned template raster.

    Returns:
        ``(xmin, ymax, rows, columns, cell_size)`` for the snapped output grid.

    Raises:
        ValueError: The template is rotated or non-square, or the features have no valid
            (non-NaN) geometry bounds.
    """
    x0, px, rx, y0, ry, py = template.geotransform
    # Tolerate ULP-level float noise: real GeoTIFFs (warped/reprojected) carry rotation
    # terms of ~1e-16 and X/Y pixel sizes that differ by a few ULPs, which exact ==/!=
    # comparisons would reject outright.
    if abs(rx) > abs(px) * _ROTATION_REL_TOL or abs(ry) > abs(py) * _ROTATION_REL_TOL:
        raise ValueError("snap_to_template does not support a rotated template.")
    if not math.isclose(abs(px), abs(py), rel_tol=1e-9):
        raise ValueError(
            "snap_to_template requires a square template (equal X/Y pixel size); "
            f"got {abs(px)} x {abs(py)}."
        )
    fxmin, fymin, fxmax, fymax = features.total_bounds
    if any(np.isnan(v) for v in (fxmin, fymin, fxmax, fymax)):
        raise ValueError(
            "snap_to_template needs a FeatureCollection with valid (non-NaN) geometry "
            "bounds to size the output."
        )
    size = abs(px)
    # Nudge by _SNAP_EPS (in cell units) so a bound mathematically on a grid line but a
    # few ULPs off does not push floor/ceil one cell too far — the snapped box stays tight
    # and the output size is stable across CRSs. Trade-off: a genuine feature edge lying
    # within _SNAP_EPS (1e-9 cell) past a grid line is treated as on it, so it never
    # over-covers by a spurious cell at the cost of that sub-nanocell ambiguity.
    snap_xmin = x0 + np.floor((fxmin - x0) / size + _SNAP_EPS) * size
    snap_xmax = x0 + np.ceil((fxmax - x0) / size - _SNAP_EPS) * size
    snap_ymax = y0 - np.floor((y0 - fymax) / size + _SNAP_EPS) * size
    snap_ymin = y0 - np.ceil((y0 - fymin) / size - _SNAP_EPS) * size
    columns = max(1, int(round((snap_xmax - snap_xmin) / size)))
    rows = max(1, int(round((snap_ymax - snap_ymin) / size)))
    return float(snap_xmin), float(snap_ymax), rows, columns, size


def _features_outside_template(features: FeatureCollection, template: Dataset) -> bool:
    """Whether the feature bounds fall entirely outside the template extent (issue #46).

    A template rasterisation burns onto the template's fixed grid, so features that lie
    wholly outside it produce an all-nodata raster. Detecting the disjoint case lets
    :func:`rasterize_features` warn rather than silently return an empty raster. A
    partial overlap is not flagged — clipping features to the template grid is the
    intended behaviour of a template.

    The template extent comes from :attr:`Dataset.bbox`, which uses the separate X and Y
    pixel sizes, so the check is correct for non-square grids (a single ``cell_size``
    would mis-measure the Y extent). An empty (or all-empty-geometry) FeatureCollection
    has ``NaN`` bounds and also burns to all-nodata, so it is treated as outside too.
    Rotated or south-up geotransforms are not handled — ``Dataset.bbox`` assumes an
    axis-aligned, north-up grid (which ``Dataset.create`` always produces).

    Args:
        features: The vector being rasterised.
        template: The template raster whose extent the features are tested against.

    Returns:
        bool: ``True`` when the feature bounds are empty (``NaN``) or disjoint from the
        template extent. A zero-area feature *inside* the extent (e.g. a single point)
        returns ``False``.
    """
    txmin, tymin, txmax, tymax = template.bbox
    fxmin, fymin, fxmax, fymax = features.total_bounds
    empty = any(np.isnan(v) for v in (fxmin, fymin, fxmax, fymax))
    return bool(
        empty or fxmax <= txmin or fxmin >= txmax or fymax <= tymin or fymin >= tymax
    )


def _resolve_burn_columns(
    features: FeatureCollection, column_name: str | list[str] | None
) -> tuple[str | list[str], Any]:
    """Resolve and validate the burn column(s) and the output dtype.

    ``None`` expands to every non-geometry column. A list is validated as
    non-empty and fully present; the dtype is promoted to the smallest that
    holds every selected column (mixed ``[int8, float32]`` -> ``float32``),
    fixing the earlier ``column_name[0]``-only dtype that truncated wider
    columns. A single column is validated as present.

    Returns:
        ``(column_name, numpy_dtype)`` with ``column_name`` normalised.

    Raises:
        ValueError: The list is empty, or a named column is missing.
    """
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
        numpy_dtype = np.result_type(*[features.dtypes[c] for c in column_name])
    else:
        if column_name not in features.columns:
            raise ValueError(
                f"column_name {column_name!r} is not in the "
                f"FeatureCollection. Available columns: "
                f"{list(features.columns)}."
            )
        numpy_dtype = features.dtypes[column_name]
    return column_name, numpy_dtype
