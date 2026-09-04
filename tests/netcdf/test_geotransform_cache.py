"""`NetCDF.geotransform` is derived once, then memoised.

Deriving it reads the `lon` / `lat` MDArrays. `RasterBase.transform` reads the
property on every access, and `xy` / `_get_indices` read `transform` per call,
so an uncached derivation made a loop of point lookups re-read the coordinate
arrays once per point -- about 13 ms each.

Caching is only safe if it is invalidated wherever the geotransform can change,
so that is what these tests pin, not the timing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
SOURCE = DATA / "cf__5v__1d4-4d1__y-asc.nc"


class TestTheDerivedGeotransformIsCached:
    """One derivation per object, and the same answer every time."""

    def test_repeated_reads_return_an_equal_geotransform(self):
        """Memoising must not change what the property reports.

        Test scenario:
            The first read derives from the coordinate arrays and the rest are
            served from the cache; all of them have to agree, or a caller's
            second lookup lands somewhere else on the grid.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")

        first = tuple(variable.geotransform)

        assert all(tuple(variable.geotransform) == first for _ in range(5))

    def test_the_transform_property_agrees_with_it(self):
        """`transform` reads this property, so the two cannot diverge.

        Test scenario:
            `RasterBase.transform` was changed to read `geotransform` rather
            than the cached `_geotransform` attribute, which is what made the
            cost visible. The two must still describe the same grid.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")

        assert tuple(variable.transform) == tuple(variable.geotransform)

    def test_it_is_derived_from_the_coordinates_not_the_gdal_fallback(self):
        """The cache must hold the derived value, not GDAL's identity.

        Test scenario:
            A multidim-opened container reports an index-space geotransform
            from GDAL (origin 0, pixel size 1). The point of deriving from the
            coordinate arrays is to get the real one, so caching the wrong one
            would be worse than not caching.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")

        geotransform = tuple(variable.geotransform)

        assert geotransform[1] != 1.0 or geotransform[0] != 0.0

    def test_the_cache_starts_empty(self):
        """Nothing is derived until the property is first read.

        Test scenario:
            Opening a container should not pay for a coordinate read it may
            never need.
        """
        dataset = NetCDF.read_file(str(SOURCE))

        assert dataset._derived_geotransform is None

    def test_it_is_cleared_when_the_geotransform_is_replaced(self):
        """A stale cache would outlive the value it describes.

        Test scenario:
            The invalidation is what makes the cache safe. Writing a new
            geotransform and clearing the cache the way the internals do must
            make the next read report the new grid.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")
        _ = variable.geotransform

        variable._derived_geotransform = None
        variable._geotransform = (100.0, 2.0, 0.0, 50.0, 0.0, -2.0)
        variable._geostationary_scaled = True

        assert tuple(variable.geotransform) == (100.0, 2.0, 0.0, 50.0, 0.0, -2.0)

    def test_a_point_lookup_round_trips_through_the_cached_transform(self):
        """The consumer the cost was measured on.

        Test scenario:
            `xy` and `rowcol` both read `transform`. A cell's centre
            converted to a coordinate and back has to return that cell.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")

        x, y = variable.xy(1, 2)

        assert variable.rowcol(x, y) == (1, 2)
