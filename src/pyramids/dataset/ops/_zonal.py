"""Zonal statistics over a :class:`Dataset` + polygon
:class:`FeatureCollection`.

Single-pass rasterize of every polygon into an integer label grid,
then numpy-based per-label reductions. Follows xvec's
`method="rasterize"` pattern (single rasterize, then groupby).

Scope kept narrow:

* Supported stats: `mean`, `sum`, `min`, `max`, `count`,
  `std`, `var`.
* One band at a time (caller selects via `band=`).
* Raster and vector must share a CRS — caller reprojects first if
  not. (A later enhancement can auto-align.)
* Output is a `pandas.DataFrame` indexed by the FeatureCollection
  row index, with one column per stat.

Area-weighted zonal statistics (where a cell's contribution is
scaled by the fraction of its area inside the polygon) are planned
as a `method="fractional"` follow-up — see
`planning/dask/zonal_stats/` for the full implementation plan.
Until that lands, callers who need area-weighted semantics must
either upsample the raster or rasterize the polygons finely and
then reduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from pyramids.base._domain import is_no_data
from pyramids.base._reductions import reduce_by_label
from pyramids.base._utils import apply_unpack
from pyramids.base.crs import sr_from_epsg, sr_from_wkt

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature import FeatureCollection


def _rasterize_labels(ds: Dataset, fc: FeatureCollection) -> np.typing.NDArray:
    """Rasterize `fc` into an integer label array shaped like `ds`.

    Label values are 0-based feature-index integers. Pixels not
    covered by any polygon get -1.

    H4: the FeatureCollection's CRS is attached to the OGR layer so
    GDAL's `RasterizeLayer` reprojects coordinates when the vector
    and raster CRSes disagree. A mismatch that `pyproj` considers
    incompatible raises :class:`ValueError` early rather than silently
    producing a mis-aligned label grid.
    """
    fc_crs = getattr(fc, "crs", None)
    ds_epsg = int(ds.epsg) if ds.epsg else None
    if fc_crs is not None and ds_epsg is not None:
        fc_epsg = fc_crs.to_epsg()
        if fc_epsg is not None and fc_epsg != ds_epsg:
            raise ValueError(
                f"zonal_stats: FeatureCollection CRS (EPSG:{fc_epsg}) does "
                f"not match Dataset CRS (EPSG:{ds_epsg}). Reproject the "
                "FeatureCollection via fc.to_crs(ds.epsg) first."
            )

    if fc_crs is not None:
        srs = sr_from_wkt(fc_crs.to_wkt())
    elif ds_epsg is not None:
        srs = sr_from_epsg(ds_epsg)
    else:
        srs = osr.SpatialReference()
    mem_driver = ogr.GetDriverByName("MEM")
    ds_vec = mem_driver.CreateDataSource("zonal_mem")
    layer = ds_vec.CreateLayer("features", srs=srs, geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn("pid", ogr.OFTInteger)
    layer.CreateField(id_field)
    for idx, geom in enumerate(fc.geometry):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("pid", int(idx))
        ogr_geom = ogr.CreateGeometryFromWkb(geom.wkb)
        feat.SetGeometry(ogr_geom)
        layer.CreateFeature(feat)
        feat = None

    mem_drv = gdal.GetDriverByName("MEM")
    label_ds = mem_drv.Create("", ds.columns, ds.rows, 1, gdal.GDT_Int32)
    label_ds.SetGeoTransform(ds.geotransform)
    label_ds.SetProjection(ds.raster.GetProjection())
    label_ds.GetRasterBand(1).Fill(-1)
    gdal.RasterizeLayer(
        label_ds,
        [1],
        layer,
        options=["ATTRIBUTE=pid", "ALL_TOUCHED=FALSE"],
    )
    labels = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None
    ds_vec = None
    return labels


def _rasterize_zonal_stats(
    ds: Dataset,
    fc: FeatureCollection,
    stats: Sequence[str],
    band: int,
    no_data: float | None,
) -> pd.DataFrame:
    """Compute stats via single-rasterize + the shared per-label reducer.

    The polygons are rasterised once into an integer label grid, then
    :func:`pyramids.base._reductions.reduce_by_label` reduces the band per label
    (bincount for sum/count/mean, a sorted-group pass for min/max/std/var).

    The band is read once in stored units. The no-data cells are found there, where
    the sentinel lives, and blanked to `NaN`; the statistics are then taken over the
    physical values, unpacked with `ds._effective_packing(band)`.

    Args:
        ds: The source dataset.
        fc: The polygons, one zone per row.
        stats: Statistic names, in output column order.
        band: Zero-based band index.
        no_data: The band's declared (stored) sentinel, or `None` when it has none.

    Returns:
        pd.DataFrame: Indexed by `fc.index`; one `float64` column per stat, `NaN` for
            a zone with no valid cell (`count` is `0.0` there).

    Raises:
        ValueError: An unknown stat name.
    """
    # Masked against the stored counts, where the sentinel lives, and the statistics
    # then taken over the physical values. `no_data` is a stored value, so comparing it
    # with a physical read of a packed band matched nothing and every gap entered the
    # zone's mean, count and minimum as a measurement.
    stored = np.asarray(ds.read_array(band=band, unpack=False))
    raster = np.asarray(
        apply_unpack(stored, *ds._effective_packing(band)), dtype=np.float64
    )
    if no_data is not None:
        # `is_no_data`, not `==`: a NaN sentinel never equals itself, so the
        # comparison marked nothing and every no-data cell entered the stats.
        raster = np.where(is_no_data(stored, no_data), np.nan, raster)
    labels = _rasterize_labels(ds, fc)
    n_features = len(fc)

    # The per-label reduction (bincount for sum/count/mean, a sorted-group pass for
    # min/max/std/var) is the grid-agnostic `reduce_by_label`: unassigned pixels carry -1
    # from `_rasterize_labels`, and the physical NaN-blanked `raster` is the value array.
    columns = reduce_by_label(raster, labels, n_features, list(stats), unassigned=-1)
    ordered = {stat: columns[stat] for stat in stats}
    return pd.DataFrame(ordered, index=fc.index)


def zonal_stats(
    ds: Dataset,
    fc: FeatureCollection,
    *,
    stats: Sequence[str] = ("mean",),
    method: str = "rasterize",
    band: int = 0,
) -> pd.DataFrame:
    """Compute zonal statistics of `ds` over polygons in `fc`.

    The statistics are taken over physical values, as `read_array` returns them: a
    CF-packed band (`scale_factor` / `add_offset`) is unpacked. The band's no-data
    cells are found in its stored values, where the sentinel lives, and left out of
    every statistic, so a packed band's gaps do not enter a zone's mean, count or
    minimum as the physical number a default read shows there.

    Args:
        ds: The source :class:`~pyramids.dataset.Dataset`.
        fc: A :class:`~pyramids.feature.FeatureCollection` of polygons.
            CRS must match `ds` — reproject first if needed.
        stats: Statistics to compute per polygon. One or more of
            `"mean"`, `"sum"`, `"min"`, `"max"`, `"std"`,
            `"var"`, `"count"`.
        method: `"rasterize"` is the only supported value today.
            Uses a single-pass GDAL rasterize then numpy groupby per
            label — fast and accurate when polygons are comparable to
            or larger than the raster pixel. An area-weighted
            `"fractional"` method is planned; see
            `planning/dask/zonal_stats/`.
        band: Zero-based band index on `ds`. Default 0.

    Returns:
        pandas.DataFrame: Indexed by `fc.index`; one column per
        stat.

    Raises:
        ValueError: Unknown `stat` name or unknown `method`.

    Examples:
        - Compute the mean value of a constant-valued raster over one
          polygon — the answer must equal the raster value itself:
            ```python
            >>> import geopandas as gpd
            >>> import numpy as np
            >>> from shapely.geometry import box
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.dataset.ops._zonal import zonal_stats
            >>> from pyramids.feature import FeatureCollection
            >>> arr = np.full((4, 4), 5.0, dtype=np.float32)
            >>> ds = Dataset.from_array(
            ...     arr,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
            ... )
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"zone": ["a"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:4326",
            ... ))
            >>> out = zonal_stats(ds, fc, stats=("mean",))
            >>> float(out["mean"].iloc[0])
            5.0

            ```
        - On a CF-packed band the statistics are physical and the gap is left out:
            ```python
            >>> import geopandas as gpd
            >>> import numpy as np
            >>> from shapely.geometry import box
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.dataset.ops._zonal import zonal_stats
            >>> from pyramids.feature import FeatureCollection
            >>> packed = Dataset.from_array(
            ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ...     no_data_value=-9999,
            ... )
            >>> packed.scale = [0.01]
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"zone": ["a"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326",
            ... ))
            >>> out = zonal_stats(packed, fc, stats=("mean", "count", "min"))
            >>> out.to_dict("records")
            [{'mean': 2.0, 'count': 3.0, 'min': 1.0}]

            ```
    """
    if method == "rasterize":
        no_data_list = ds.no_data_value
        no_data = no_data_list[band] if no_data_list else None
        result = _rasterize_zonal_stats(ds, fc, stats, band, no_data)
    else:
        raise ValueError(f"method must be 'rasterize', got {method!r}")
    return result


__all__ = ["zonal_stats"]
