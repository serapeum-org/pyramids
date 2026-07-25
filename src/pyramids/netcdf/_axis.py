"""CF axis-role detection helpers extracted from the NetCDF class (ARC-37).

Pure functions that classify an MDArray dimension as the longitude (``"X"``) or latitude (``"Y"``)
raster axis from its coordinate variable's CF attributes. This is the first, self-contained slice of
the planned `AxisResolver` engine; the larger spatial/time axis-resolution cluster
(`_detect_spatial_axes` and friends) and the other engines (geostationary handling, MDIM read,
netCDF writer) remain on `NetCDF` pending a dedicated extraction follow-up.
"""

from __future__ import annotations

from typing import Any

from pyramids.netcdf.cf import detect_axis
from pyramids.netcdf.utils import _read_attributes


def axis_role_of_dimension(dim: Any) -> str | None:
    """Classify a dimension as `"X"` (longitude) or `"Y"` (latitude) from its coordinate CF attributes.

    Reads the CF attributes of the dimension's coordinate (indexing) variable — `axis` (`X`/`Y`),
    `standard_name` (`longitude`/`latitude`), or `units` (`degrees_east`/`degrees_north`) — and
    classifies them through the shared `pyramids.netcdf.cf.detect_axis` heuristic. Returns `None` when
    the role cannot be determined.

    Args:
        dim: A GDAL MDArray dimension whose indexing variable carries the CF attributes.

    Returns:
        `"X"`, `"Y"`, or `None`.
    """
    indexing_var = dim.GetIndexingVariable()
    if indexing_var is None:
        return None
    # Attribute-only detection (empty `name` disables the name-pattern fallback) so the CF-attribute
    # vs. dimension-name stages stay separated, matching the historical pipeline. Filter to the
    # spatial roles this classifier promises (`detect_axis` can also return `"T"`/`"Z"`).
    role = detect_axis("", _read_attributes(indexing_var))
    return role if role in ("X", "Y") else None


def detect_axis_indices(dims: Any) -> tuple[int | None, int | None]:
    """Indices of the X (longitude) and Y (latitude) dimensions via CF coordinate attributes.

    Returns the first dimension classified as `"X"` and the first as `"Y"` by
    `axis_role_of_dimension` (each `None` when undetected).

    Args:
        dims: The MDArray's dimensions, in storage order.

    Returns:
        `(x_index, y_index)`, each `None` when the axis is not detected.
    """
    detected_x = detected_y = None
    for i, dim in enumerate(dims):
        role = axis_role_of_dimension(dim)
        if role == "X" and detected_x is None:
            detected_x = i
        elif role == "Y" and detected_y is None:
            detected_y = i
    return detected_x, detected_y
