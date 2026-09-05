"""`transform` and `geotransform` must describe the same affine.

`RasterBase.transform` built its `GeoTransform` from `_geotransform`, the value
cached at construction, while `RasterBase.geotransform` is a property that
`NetCDF` overrides to derive the affine from the file's CF coordinate variables.
For a netCDF container the cached value is GDAL's identity fallback, so the two
disagreed on every one of them -- and code reading `transform` got pixel indices
where it expected map coordinates.

`transform` now returns `GeoTransform(*self.geotransform)`, so asserting the two
are equal is an identity and can never fail. What these tests assert instead is
that `transform` is *not* the construction-time cache -- the value the defect
returned -- which is a comparison the defect fails and the fix passes.
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
def test_transform_is_not_the_geotransform_cached_at_construction(fixture: str):
    """`transform` follows the derived affine, for every netCDF fixture.

    Args:
        fixture: A netCDF fixture file name.

    Test scenario:
        Asserting `transform == geotransform` cannot fail: `transform` is
        literally `GeoTransform(*self.geotransform)`. The regression it was
        meant to catch is `transform` reading `_geotransform` instead -- GDAL's
        identity fallback, which every one of these fixtures reports and none
        of their derived affines equals. That comparison *can* fail, so it is
        the one made here.
    """
    try:
        dataset = NetCDF.read_file(str(DATA / fixture))
    except RuntimeError as exc:  # pragma: no cover - fixture-dependent
        pytest.skip(f"fixture not readable: {exc}")

    cached = tuple(dataset._geotransform)

    assert tuple(dataset.geotransform) != cached, (
        "the fixture no longer exercises the divergence this test is about"
    )
    assert tuple(dataset.transform) != cached


def test_a_plain_raster_is_unaffected():
    """An ordinary `Dataset` has no override, so `transform` is still the cache.

    Test scenario:
        The mirror of the netCDF case: with no derivation in the way, the
        affine `transform` reports has to be the one cached at construction.
        This fails if the netCDF override ever leaks onto the base class.
    """
    dataset = Dataset.from_array(
        np.ones((4, 4), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )

    assert tuple(dataset.transform) == tuple(dataset._geotransform)
