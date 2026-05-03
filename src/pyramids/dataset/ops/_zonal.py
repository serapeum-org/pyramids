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

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from pyramids.base.crs import sr_from_epsg, sr_from_wkt

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature import FeatureCollection


_STAT_FUNCS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "var": np.nanvar,
    "count": lambda vals: float(np.sum(~np.isnan(vals))),
}


def _rasterize_labels(ds: Dataset, fc: FeatureCollection) -> np.ndarray:
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
    mem_driver = ogr.GetDriverByName("Memory")
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


_BINCOUNT_STATS = {"mean", "sum", "count"}


def _rasterize_zonal_stats(
    ds: Dataset,
    fc: FeatureCollection,
    stats: Sequence[str],
    band: int,
    no_data: float | None,
) -> pd.DataFrame:
    """Compute stats via single-rasterize + vectorised groupby.

    mean / sum / count stats route through :func:`numpy.bincount`
    weighted reductions instead of a per-polygon Python loop. Non-
    linear stats (std / var / min / max) still use the loop because
    bincount can't express them without per-label sort.
    """
    raster = np.asarray(ds.read_array(band=band), dtype=np.float64)
    if no_data is not None:
        raster = np.where(raster == no_data, np.nan, raster)
    labels = _rasterize_labels(ds, fc)
    n_features = len(fc)

    stats_bincount = [s for s in stats if s in _BINCOUNT_STATS]
    stats_loop = [s for s in stats if s not in _BINCOUNT_STATS]

    columns: dict[str, np.ndarray] = {}
    if stats_bincount:
        columns.update(_bincount_stats(raster, labels, n_features, stats_bincount))
    if stats_loop:
        loop_cols: dict[str, list[float]] = {s: [] for s in stats_loop}
        for pid in range(n_features):
            vals = raster[labels == pid]
            for stat in stats_loop:
                loop_cols[stat].append(_apply_stat(stat, vals))
        for stat, vals_list in loop_cols.items():
            columns[stat] = np.asarray(vals_list, dtype=np.float64)

    ordered = {stat: columns[stat] for stat in stats}
    return pd.DataFrame(ordered, index=fc.index)


def _bincount_stats(
    raster: np.ndarray,
    labels: np.ndarray,
    n_features: int,
    stats: list[str],
) -> dict[str, np.ndarray]:
    """Vectorised sum / count / mean via :func:`numpy.bincount`.

    `labels` uses -1 for unassigned pixels; we shift by +1 so
    bincount's non-negative-index requirement is satisfied, then
    drop the leading "unassigned" bucket.
    """
    flat_labels = labels.ravel() + 1
    flat_vals = raster.ravel()
    valid_mask = ~np.isnan(flat_vals)
    minlength = n_features + 1
    sums = np.bincount(
        flat_labels[valid_mask],
        weights=flat_vals[valid_mask],
        minlength=minlength,
    )[1:]
    counts = np.bincount(
        flat_labels[valid_mask],
        minlength=minlength,
    )[
        1:
    ].astype(np.float64)
    out: dict[str, np.ndarray] = {}
    if "sum" in stats:
        out["sum"] = sums
    if "count" in stats:
        out["count"] = counts
    if "mean" in stats:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / counts, np.nan)
        out["mean"] = mean
    return out


def _apply_stat(stat: str, vals: np.ndarray) -> float:
    """Safe per-polygon stat — empty cohort or all-NaN returns NaN."""
    if vals.size == 0:
        result = float("nan")
    else:
        valid = vals[~np.isnan(vals)] if stat != "count" else vals
        if stat != "count" and valid.size == 0:
            result = float("nan")
        else:
            try:
                func = _STAT_FUNCS[stat]
            except KeyError as exc:
                raise ValueError(
                    f"unknown stat {stat!r}; supported: {sorted(_STAT_FUNCS)}"
                ) from exc
            result = float(func(vals))
    return result


def zonal_stats(
    ds: Dataset,
    fc: FeatureCollection,
    *,
    stats: Sequence[str] = ("mean",),
    method: str = "rasterize",
    band: int = 0,
) -> pd.DataFrame:
    """Compute zonal statistics of `ds` over polygons in `fc`.

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
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._zonal import zonal_stats
            >>> from pyramids.feature import FeatureCollection
            >>> arr = np.full((4, 4), 5.0, dtype=np.float32)
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326,
            ... )
            >>> fc = FeatureCollection(gpd.GeoDataFrame(
            ...     {"zone": ["a"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:4326",
            ... ))
            >>> out = zonal_stats(ds, fc, stats=("mean",))
            >>> float(out["mean"].iloc[0])
            5.0

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
