"""Grid inference and reshape shared by the `to_dataframe` / `from_dataframe` pair.

A `MultiIndex` DataFrame that names its axes — the innermost two the `(y, x)` grid, any
outer levels the band axis/dimensions — describes a regular raster grid without carrying its
georeferencing. Turning such a frame back into an array means the same three steps whether the
caller rebuilds a raster `Dataset` or a NetCDF cube: read the axis names, recover the affine
geotransform from the cell centres (a regular grid is required), and reindex the frame onto the
full Cartesian product so each value column reshapes to the grid.

That logic lives here rather than in either engine so `pyramids.dataset` and `pyramids.netcdf`
share one implementation with no cross-boundary import (`base` is below both). The functions are
pure — NumPy and pandas only — and know nothing about `Dataset`, `NetCDF`, or GDAL. The error
messages say `from_dataframe()` because that is the caller on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd


def frame_axes(
    df: pd.DataFrame, x: str | None, y: str | None
) -> tuple[list[str], str, str]:
    """Resolve the band-axis names and the row / column axis names of a frame.

    Args:
        df: The DataFrame to read the index of.
        x: The column-axis level name, or `None` for the innermost level.
        y: The row-axis level name, or `None` for the second-innermost level.

    Returns:
        tuple[list[str], str, str]: `(band_names, row_name, col_name)`, the band names in index
        order (outermost first).

    Raises:
        ValueError: The index is not a `MultiIndex` of at least two named levels, a named
            `x` / `y` level is missing, or the two coincide.
    """
    index = df.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        raise ValueError(
            "from_dataframe() needs a DataFrame indexed by its dimensions — a MultiIndex "
            "of at least two named levels, the innermost two being the (y, x) grid axes. A "
            "tidy frame on a plain index is scattered rows, not a raster; name the axes "
            "first, e.g. df.set_index([...])."
        )
    names = list(index.names)
    if any(nm is None for nm in names):
        raise ValueError(
            f"from_dataframe() needs every index level named; got {names}. Name them via "
            "df.rename_axis([...]) or set_index."
        )
    row_name = y if y is not None else names[-2]
    col_name = x if x is not None else names[-1]
    for role, nm in (("y", row_name), ("x", col_name)):
        if nm not in names:
            raise ValueError(
                f"from_dataframe() was told the {role} axis is {nm!r}, which is not an "
                f"index level; the levels are {names}."
            )
    if row_name == col_name:
        raise ValueError(
            f"from_dataframe() got {row_name!r} for both the y and x axes; they must be "
            "different index levels."
        )
    band_names = [nm for nm in names if nm not in (row_name, col_name)]
    return band_names, row_name, col_name


def value_columns(df: pd.DataFrame, variables: str | Sequence[Any] | None) -> list:
    """The columns that become data — the original labels, in order.

    The **original** column labels are returned, not stringified ones, because the caller
    indexes the frame with them (`ordered[col]`); only a NetCDF variable name is stringified, at
    the point it is written. Returning `str(...)`-normalised labels made `ordered[col]` raise
    `KeyError` on any non-string column.

    Args:
        df: The DataFrame whose columns are the candidates.
        variables: A label, a sequence of labels, or `None` for every column.

    Returns:
        list: The chosen column labels, in the given order, never empty.

    Raises:
        ValueError: The frame has no columns, a requested label is not a column, a label was
            given more than once, or an empty selection was given.
    """
    available = list(df.columns)
    if not available:
        raise ValueError(
            "from_dataframe() needs at least one value column to become data; the frame has "
            "none."
        )
    if variables is None:
        return available
    # A list/tuple is a set of labels; anything else — a str, or a scalar label such as an int
    # column name — is a single label (`list(7)` would raise).
    names = list(variables) if isinstance(variables, (list, tuple)) else [variables]
    if not names:
        raise ValueError(
            "from_dataframe() was given an empty selection; pass `variables=None` for every "
            f"column, or one of {available}."
        )
    unknown = [nm for nm in names if nm not in available]
    if unknown:
        raise ValueError(
            f"from_dataframe() cannot take {unknown!r}: the frame's columns are {available}."
        )
    repeated = [nm for nm in dict.fromkeys(names) if names.count(nm) > 1]
    if repeated:
        raise ValueError(
            f"from_dataframe() was asked for {repeated!r} more than once; a label can only "
            "become one column of data."
        )
    return names


def check_no_duplicate_index(df: pd.DataFrame) -> None:
    """Refuse a frame whose index has duplicate rows — an ambiguous cell.

    Args:
        df: The DataFrame to check.

    Raises:
        ValueError: The index has duplicate rows, so a cell would carry more than one value.
    """
    if df.index.duplicated().any():
        raise ValueError(
            "from_dataframe() found duplicate index rows, so a cell has more than one "
            "value. Resolve them first, e.g. "
            "df.groupby(level=list(df.index.names)).mean()."
        )


def reshape_to_grid(
    df: pd.DataFrame, band_names: list[str], row_name: str, col_name: str
) -> tuple[list, np.ndarray, np.ndarray, tuple, pd.DataFrame, tuple]:
    """Recover the grid coordinates + geotransform and reindex the frame onto the full product.

    The row axis is sorted **descending** and the column axis ascending, so the grid is
    north-up whatever order the frame's rows arrived in; band coordinates keep first-appearance
    order. The frame is reindexed against the full `(*band, y, x)` Cartesian product so every
    value column reshapes to `shape` (absent cells become `NaN`).

    Args:
        df: The DataFrame to reshape.
        band_names: The band-axis level names, outermost first (may be empty).
        row_name: The row (y) index level name.
        col_name: The column (x) index level name.

    Returns:
        tuple: `(band_coords, y_coords, x_coords, geo, ordered, shape)` — the per-band-level
        unique coordinates, the descending y and ascending x centres, the inferred affine
        geotransform, the frame reindexed onto the full product, and the `(*band, y, x)` shape.

    Raises:
        ValueError: The x or y axis is irregular or has fewer than two coordinates, so no
            geotransform can be inferred.
    """
    band_coords = [pd.unique(df.index.get_level_values(nm)) for nm in band_names]
    y_coords = np.unique(
        np.asarray(df.index.get_level_values(row_name), dtype="float64")
    )[::-1]
    x_coords = np.unique(
        np.asarray(df.index.get_level_values(col_name), dtype="float64")
    )
    geo = geotransform_from_centres(x_coords, y_coords)
    ordered = df.reorder_levels([*band_names, row_name, col_name])
    full = pd.MultiIndex.from_product(
        [*band_coords, list(y_coords), list(x_coords)],
        names=[*band_names, row_name, col_name],
    )
    ordered = ordered.reindex(full)
    shape = (*[len(c) for c in band_coords], len(y_coords), len(x_coords))
    return band_coords, y_coords, x_coords, geo, ordered, shape


def column_array(ordered: pd.DataFrame, col: Any, shape: tuple) -> np.ndarray:
    """One value column as a float64 array of the target shape, refusing the bad cases.

    Args:
        ordered: The frame reindexed against the full dimension product.
        col: The column label to read.
        shape: The target `(*band_sizes, rows, cols)` shape.

    Returns:
        np.ndarray: The column's cells, `float64`, shaped `shape`.

    Raises:
        ValueError: The label matches more than one column (it cannot become one array), or the
            column is not numeric — each named, rather than a raw numpy reshape/convert error.
    """
    series = ordered[col]
    if isinstance(series, pd.DataFrame):
        raise ValueError(
            f"from_dataframe() found more than one column labelled {col!r}, so it cannot "
            "become one array. Give each value column a unique label."
        )
    try:
        values = series.to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"from_dataframe() cannot read column {col!r} as numbers: {exc}. A value column "
            "must be numeric."
        ) from exc
    # `to_numpy` is typed `Any` in the pandas stubs, so reshape is too; the cast keeps the
    # declared `-> np.ndarray` honest under `warn_return_any`.
    return cast("np.ndarray", values.reshape(shape))


def geotransform_from_centres(
    x_coords: np.ndarray, y_coords: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Infer a north-up geotransform from ascending x and descending y cell centres.

    The two axes are assumed regular; `regular_step` refuses an irregular one. The centres are
    half a cell inside the edges, so the origin steps back half a cell in each axis: with
    `dy < 0`, `y_max - dy / 2` is half a cell **above** the topmost centre.

    Args:
        x_coords: The unique column coordinates, ascending, at least two of them.
        y_coords: The unique row coordinates, descending, at least two of them.

    Returns:
        tuple: The affine geotransform `(x_min, dx, 0.0, y_max, 0.0, dy)`, `dy` negative.

    Raises:
        ValueError: Either axis is irregular or has fewer than two coordinates.
    """
    dx = regular_step(x_coords, "x")
    dy = regular_step(y_coords, "y")
    x_min = float(x_coords[0]) - dx / 2.0
    y_max = float(y_coords[0]) - dy / 2.0
    return (x_min, dx, 0.0, y_max, 0.0, dy)


def regular_step(coords: np.ndarray, axis: str) -> float:
    """The constant spacing of a coordinate axis, refusing an irregular or too-short one.

    Args:
        coords: The axis' unique coordinates, already sorted (ascending x, descending y).
        axis: `"x"` or `"y"`, for the message.

    Returns:
        float: The step between consecutive coordinates — positive for x, negative for y.

    Raises:
        ValueError: Fewer than two coordinates (no spacing to infer), or the spacing varies (an
            irregular grid has no affine transform).
    """
    if coords.size < 2:
        raise ValueError(
            f"from_dataframe() cannot infer the {axis} cell size from a single {axis} "
            f"coordinate; give an axis with at least two cells, or resample to a grid first."
        )
    # `coords` are `np.unique`'d, hence strictly monotonic, so every diff is non-zero; the only
    # failure to guard is uneven spacing, which `np.allclose` catches.
    diffs = np.diff(coords)
    step = float(diffs[0])
    if not np.allclose(diffs, step, rtol=1e-6, atol=0.0):
        raise ValueError(
            f"from_dataframe() needs a regular {axis} axis to build a geotransform, but its "
            f"spacing varies. An irregular grid has no affine transform; resample to a "
            f"regular grid first."
        )
    return step
