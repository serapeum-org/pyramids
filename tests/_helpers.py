"""Shared test-helper utilities used across the pyramids test suite."""

from __future__ import annotations

from pyramids.dataset import Dataset


def write_raster(path, arr, top_left, *, epsg=4326, cell_size=1.0, nodata=-9999.0):
    """Write ``arr`` to ``path`` as a GeoTIFF and return the path string.

    Args:
        path: Output path for the GeoTIFF.
        arr: Array to write.
        top_left: Top-left corner of the raster.
        epsg: EPSG code of the source CRS (default 4326).
        cell_size: Pixel size in CRS units.
        nodata: No-data marker stamped on the output.

    Returns:
        str: The output path as a string.
    """
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=nodata,
    )
    ds.to_file(str(path))
    return str(path)
