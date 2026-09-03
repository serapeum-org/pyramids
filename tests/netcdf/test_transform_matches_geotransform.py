"""`transform` and `geotransform` must describe the same affine.

`RasterBase.transform` built its `GeoTransform` from `_geotransform`, the value
cached at construction, while `RasterBase.geotransform` is a property that
`NetCDF` overrides to derive the affine from the file's CF coordinate variables.
For a netCDF container the cached value is GDAL's identity fallback, so the two
disagreed on every one of them -- and code reading `transform` got pixel indices
where it expected map coordinates.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
FIXTURES = sorted(Path(p).name for p in glob.glob(str(DATA / "*.nc")))


@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURES)
def test_transform_agrees_with_geotransform(fixture: str):
    """The two views of the affine agree, for every netCDF fixture."""
    try:
        dataset = NetCDF.read_file(str(DATA / fixture))
    except RuntimeError as exc:  # pragma: no cover - fixture-dependent
        pytest.skip(f"fixture not readable: {exc}")

    assert tuple(dataset.transform) == tuple(dataset.geotransform)


def test_a_plain_raster_is_unaffected():
    """An ordinary `Dataset` has no override, so nothing changes for it."""
    dataset = Dataset.from_array(
        np.ones((4, 4), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )

    assert tuple(dataset.transform) == tuple(dataset.geotransform)
    assert tuple(dataset.transform) == tuple(dataset._geotransform)
