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
from osgeo import gdal

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
SOURCE = DATA / "cf__5v__1d4-4d1__y-asc.nc"
# A grid nothing in the fixture could derive by accident, so a read after the
# raster swap can only report it if the memo was really dropped.
REPLACEMENT_GEOTRANSFORM = (100.0, 2.0, 0.0, 50.0, 0.0, -2.0)


class _CountingDerivation:
    """A `_compute_geotransform` stand-in that answers identically and counts.

    Installed on the instance, so it shadows the class method for that one
    object and leaves every other test untouched.
    """

    def __init__(self, original):
        """Store the bound method being shadowed.

        Args:
            original: The object's real `_compute_geotransform`.
        """
        self.original = original
        self.calls = 0

    def __call__(self):
        """Count the derivation, then return exactly what the real one returns.

        Returns:
            tuple: The derived geotransform.
        """
        self.calls += 1
        return self.original()


class TestTheDerivedGeotransformIsCached:
    """One derivation per object, and the same answer every time."""

    def test_it_is_derived_once_however_often_it_is_asked(self, monkeypatch):
        """Six reads, one derivation -- counted, not inferred.

        Args:
            monkeypatch: Fixture used to install the counting stand-in.

        Test scenario:
            Equality across repeated reads cannot fail on its own: an uncached
            derivation returns an equal tuple every time, so the assertion the
            memo is meant to justify holds just as well with the memo removed.
            What the memo changes is how many times the `lon` / `lat` MDArrays
            are read, so the derivations are counted -- a `geotransform` that
            dropped the memo would derive six times while still answering the
            same.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")
        counter = _CountingDerivation(variable._compute_geotransform)
        monkeypatch.setattr(variable, "_compute_geotransform", counter)

        first = tuple(variable.geotransform)
        rest = [tuple(variable.geotransform) for _ in range(5)]

        assert counter.calls == 1, (
            f"the coordinate arrays were re-read {counter.calls} times, expected once"
        )
        assert all(value == first for value in rest), (
            f"repeated reads disagree: {first} then {rest}"
        )

    def test_the_transform_property_goes_through_the_memo(self):
        """Reading `transform` populates the derivation cache, so it used it.

        Test scenario:
            Asserting `transform == geotransform` cannot fail -- `transform`
            is `GeoTransform(*self.geotransform)` -- and the "not the
            construction-time cache" form has no bite on a *variable*, whose
            `_geotransform` GDAL has already corrected to the right value (the
            container is where the two differ, and
            `test_transform_matches_geotransform.py` pins that for every
            fixture). What is left to pin here, and what this file is about, is
            that `transform` reads the memoised property rather than reaching
            past it to `_geotransform`: bypassing it would put the per-point
            coordinate re-read straight back.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")
        assert variable._derived_geotransform is None, "the cache started warm"

        _ = variable.transform

        assert variable._derived_geotransform is not None, (
            "`transform` reached past the memoised property to `_geotransform`"
        )

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

        assert geotransform[1] != 1.0 or geotransform[0] != 0.0, (
            f"the cache holds GDAL's index-space fallback: {geotransform}"
        )

    def test_the_cache_starts_empty(self):
        """Nothing is derived until the property is first read.

        Test scenario:
            Opening a container should not pay for a coordinate read it may
            never need.
        """
        dataset = NetCDF.read_file(str(SOURCE))

        assert dataset._derived_geotransform is None, (
            "opening derived a geotransform nobody asked for: "
            f"{dataset._derived_geotransform}"
        )

    def test_swapping_the_backing_raster_clears_the_cache(self):
        """A stale cache would outlive the value it describes.

        Test scenario:
            `_replace_raster` is the production invalidation path -- every
            variable mutation (`set_variable`, `add_variable`,
            `remove_variable`, `rename_variable`) swaps the backing raster
            through it. Clearing the memo by hand in the test asserts nothing
            about the code, so instead the memo is warmed, a raster on a
            different grid is swapped in, and the next read has to report that
            grid. Dropping the clear from `_replace_raster` leaves the old
            value memoised and fails here.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")
        warm = tuple(variable.geotransform)
        assert warm != REPLACEMENT_GEOTRANSFORM, "the fixture already has the new grid"
        replacement = gdal.GetDriverByName("MEM").Create("", 4, 3, 1, gdal.GDT_Float32)
        replacement.SetGeoTransform(REPLACEMENT_GEOTRANSFORM)

        variable._replace_raster(replacement)

        assert tuple(variable.geotransform) == REPLACEMENT_GEOTRANSFORM, (
            f"the memo survived the raster swap: still reporting {warm}"
        )

    def test_a_point_lookup_round_trips_through_the_cached_transform(self):
        """The consumer the cost was measured on.

        Test scenario:
            `xy` and `rowcol` both read `transform`. A cell's centre
            converted to a coordinate and back has to return that cell.
        """
        variable = NetCDF.read_file(str(SOURCE)).get_variable("temperature")

        x, y = variable.xy(1, 2)

        assert variable.rowcol(x, y) == (1, 2), (
            f"({x}, {y}) came back as {variable.rowcol(x, y)}, expected (1, 2)"
        )
