"""Shared fixtures for the STAC dataset tests.

These are the rasters and item dicts that several STAC tasks reuse: a packed
int16 raster with a known scale (so rescale assertions are exact), and a small
set of local items carrying distinct datetimes plus a groupable property.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset


@pytest.fixture
def packed_raster_path(tmp_path):
    """A local int16 GeoTIFF with a known scale so rescale tests are exact.

    Stored values are ``[[0, 100], [200, 300]]`` with nodata ``0``; a scale of
    ``0.01`` therefore makes the physical values ``[[nodata, 1.0], [2.0, 3.0]]``.
    The scale is **not** written into the file — the STAC item declares it, which
    is exactly the case `rescale` has to handle.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the written raster.
    """
    path = str(tmp_path / "packed.tif")
    dataset = Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    dataset.to_file(path)
    return path


@pytest.fixture
def packed_item(packed_raster_path):
    """A STAC item over `packed_raster_path` declaring scale 0.01 and nodata 0.

    Args:
        packed_raster_path: The packed int16 raster.

    Returns:
        dict: An item whose single asset carries a `raster:bands` scale/offset.
    """
    return {
        "type": "Feature",
        "id": "packed",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
        },
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {
            "data": {
                "href": packed_raster_path,
                "type": "image/tiff",
                "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
            }
        },
        "stac_extensions": [],
    }


@pytest.fixture
def three_local_items(tmp_path):
    """Three STAC item dicts over real local rasters.

    Each item has a distinct datetime and an ``orbit`` property — two share
    ``orbit=1`` and one has ``orbit=2`` — so a property `groupby` collapses them
    from three timesteps to two. ``eo:cloud_cover`` varies so property-to-cube
    attachment has something to carry.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        list[dict]: Three STAC item dicts, in datetime order.
    """
    items = []
    for index, orbit in enumerate((1, 1, 2)):
        path = str(tmp_path / f"scene{index}.tif")
        Dataset.from_array(
            np.full((3, 3), float(index + 1), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        ).to_file(path)
        items.append(
            {
                "type": "Feature",
                "id": f"scene{index}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]],
                },
                "bbox": [0.0, 0.0, 3.0, 3.0],
                "properties": {
                    "datetime": f"2023-06-0{index + 1}T00:00:00Z",
                    "orbit": orbit,
                    "eo:cloud_cover": 10 * index,
                },
                "assets": {"data": {"href": path, "type": "image/tiff"}},
                "stac_extensions": [],
            }
        )
    return items
