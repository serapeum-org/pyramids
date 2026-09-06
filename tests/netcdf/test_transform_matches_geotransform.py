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

from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"


def _netcdf_fixtures() -> list[str]:
    """Every netCDF fixture in the corpus, or a loud failure when there are none.

    Returns:
        list[str]: The fixture file names, sorted.

    Raises:
        FileNotFoundError: When the directory holds no `.nc` file. Discovering
            the corpus is what keeps this test covering every shape of file in
            it, but a rename or a move would otherwise parametrise over an
            empty list -- zero tests collected, and a green run reporting that
            nothing is wrong.
    """
    names = sorted(path.name for path in DATA.glob("*.nc"))
    if not names:
        raise FileNotFoundError(f"no netCDF fixtures found under {DATA}")
    return names


FIXTURES = _netcdf_fixtures()


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
        the one made here, together with the map coordinate the affine puts the
        first pixel at: the defect placed it at the fallback's origin.
    """
    dataset = NetCDF.read_file(str(DATA / fixture))

    cached = tuple(dataset._geotransform)
    derived = tuple(dataset.geotransform)

    assert derived != cached, (
        "the fixture no longer exercises the divergence this test is about"
    )
    assert tuple(dataset.transform) != cached, (
        f"`transform` returned the construction-time cache {cached}"
    )
    assert dataset.transform * (0, 0) == (derived[0], derived[3]), (
        f"`transform` does not map the first pixel onto the derived origin {derived}"
    )


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

    assert tuple(dataset.transform) == tuple(dataset._geotransform), (
        "`transform` no longer reports the affine an ordinary raster was built with"
    )
    assert dataset.transform * (0, 0) == (0.0, 4.0), (
        "`transform` does not map the first pixel onto the top-left corner"
    )
