"""A Zarr store must round-trip its full affine, not just origin and cell size.

`Dataset.from_zarr` rebuilt the georeference from `top_left_corner` and
`cell_size` alone. That pair cannot express the two rotation terms, and it
cannot express a positive pixel height, so a rotated or south-up store came back
silently re-gridded as a north-up square one -- no error, just different
coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

try:  # pragma: no cover - the zarr-using test is @pytest.mark.lazy gated
    import zarr  # noqa: F401
except ImportError:  # pragma: no cover
    zarr = None

AFFINES = {
    "north_up_square": (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
    "non_square_cells": (0.0, 2.0, 0.0, 4.0, 0.0, -0.5),
    "rotated": (0.0, 1.0, 0.5, 4.0, 0.25, -1.0),
    "south_up": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
}


@pytest.mark.lazy
@pytest.mark.parametrize("name", list(AFFINES), ids=list(AFFINES))
def test_the_affine_survives_a_zarr_round_trip(name: str, tmp_path: Path):
    """Every element of the geotransform comes back unchanged.

    The source is written to disk first: the Zarr writer reads through the lazy
    `read_array(chunks=...)` path, which needs a real file to reopen.
    """
    geotransform = AFFINES[name]
    Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4),
        geo_ref=GeoReference(geo=geotransform, epsg=4326),
    ).to_file(str(tmp_path / "src.tif"))
    source = Dataset.read_file(str(tmp_path / "src.tif"))
    store = tmp_path / f"{name}.zarr"
    source.to_zarr(str(store))

    restored = Dataset.from_zarr(str(store))

    assert tuple(restored.geotransform) == pytest.approx(geotransform)
