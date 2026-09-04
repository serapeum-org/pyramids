"""Two guards the second review round found one `if` and one `Clone()` short.

`_dataarray_to_dataset` looked its spatial coordinates up by name and never
checked them against the array's shape, so a `(y, x, time)` DataArray became a
raster of the wrong shape georeferenced by axes it is not laid out on --
silently, with a plausible-looking geotransform.

`_reproject_with_ReprojectImage` normalised only the source axis order before
an identity check that is axis-order sensitive, so its own docstring's promise
("any axis-mapping strategy is accepted") held only for the one destination
the in-repo caller happens to build.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base.crs import sr_from_epsg
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.cog.facade import _dataarray_to_dataset
from pyramids.dataset.engines import spatial as spatial_engine

try:
    import xarray as xr
except ImportError:  # pragma: no cover - exercised only without the extra
    xr = None

# Guarded rather than a module-level `importorskip`, so the file still collects
# without the labeled-array extra; the tests that need it are marked.
needs_xarray = pytest.mark.skipif(xr is None, reason="requires xarray")

X = 499985.0 + 30.0 * np.arange(6)
Y = 4600015.0 - 30.0 * np.arange(4)


class TestTheSpatialCoordinatesMustBeTheLastTwoAxes:
    """A coordinate that does not label the axis it is used for is refused."""

    @pytest.mark.lazy
    @needs_xarray
    def test_a_trailing_time_axis_is_refused(self):
        """The regression: a 4x6x3 array became a 6x3 raster with 4 bands.

        Test scenario:
            `(y, x, time)` puts `time` last, so the array's rows and columns
            are `x` and `time` -- but the lookup found `y` and `x` by name and
            georeferenced the result from those, giving a geotransform built
            from axes the data is not shaped by. Nothing raised.
        """
        da = xr.DataArray(
            np.zeros((4, 6, 3)),
            dims=("y", "x", "time"),
            coords={"y": Y, "x": X},
            attrs={"crs": 32636},
        )

        with pytest.raises(ValueError, match="do not match its last two axes"):
            _dataarray_to_dataset(da, None, None)

    @pytest.mark.lazy
    @needs_xarray
    def test_the_message_says_how_to_fix_it(self):
        """A refusal the caller can act on names the axes and the remedy.

        Test scenario:
            The array is one `transpose` away from working, so the message
            has to say that rather than only that something is wrong.
        """
        da = xr.DataArray(
            np.zeros((4, 6, 3)),
            dims=("y", "x", "time"),
            coords={"y": Y, "x": X},
            attrs={"crs": 32636},
        )

        with pytest.raises(ValueError) as excinfo:
            _dataarray_to_dataset(da, None, None)

        message = str(excinfo.value)
        assert "Transpose" in message
        assert "'y' (4)" in message and "'x' (6)" in message

    @pytest.mark.lazy
    @needs_xarray
    def test_a_two_dimensional_array_still_builds(self):
        """The ordinary case must be untouched.

        Test scenario:
            `(y, x)` is already its own last two dimensions, so the guard has
            nothing to say and the raster comes out 4 rows by 6 columns.
        """
        da = xr.DataArray(
            np.zeros((4, 6)),
            dims=("y", "x"),
            coords={"y": Y, "x": X},
            attrs={"crs": 32636},
        )

        dataset = _dataarray_to_dataset(da, None, None)

        assert (dataset.rows, dataset.columns) == (4, 6)
        assert dataset.band_count == 1

    @pytest.mark.lazy
    @needs_xarray
    def test_a_leading_band_axis_still_builds(self):
        """`(time, y, x)` is the shape a multi-band export actually has.

        Test scenario:
            The guard checks the *last* two dimensions precisely so a leading
            band or time axis keeps working; a stricter "exactly two"
            rule would have broken every multi-band DataArray.
        """
        da = xr.DataArray(
            np.zeros((3, 4, 6)),
            dims=("time", "y", "x"),
            coords={"y": Y, "x": X},
            attrs={"crs": 32636},
        )

        dataset = _dataarray_to_dataset(da, None, None)

        assert (dataset.rows, dataset.columns) == (4, 6)
        assert dataset.band_count == 3

    @pytest.mark.lazy
    @needs_xarray
    def test_transposed_spatial_axes_are_refused(self):
        """`(x, y)` is the subtler half of the same mistake.

        Test scenario:
            Both axes are spatial and both are last, but they are the wrong
            way round -- the array's rows are `x`. A check that only asked
            "are they present and last" would let this through.
        """
        da = xr.DataArray(
            np.zeros((6, 4)),
            dims=("x", "y"),
            coords={"y": Y, "x": X},
            attrs={"crs": 32636},
        )

        with pytest.raises(ValueError, match="do not match its last two axes"):
            _dataarray_to_dataset(da, None, None)


class TestTheIdentityCheckAcceptsAnyAxisMappingStrategy:
    """What the docstring promised; the code normalised only one side."""

    @pytest.fixture
    def wgs84_raster(self) -> Dataset:
        """A small north-up WGS 84 raster.

        Returns:
            Dataset: Built in memory, so the test says nothing about I/O.
        """
        return Dataset.from_array(
            np.ones((4, 6), dtype="float32"),
            geo_ref=GeoReference(
                top_left_corner=(-10.0, 50.0), cell_size=0.25, epsg=4326
            ),
        )

    @pytest.mark.core
    def test_an_authority_order_destination_takes_the_identity_branch(
        self, wgs84_raster, monkeypatch
    ):
        """#418 verbatim, through the destination the docstring invites.

        Args:
            wgs84_raster: The fixture raster.
            monkeypatch: Fixture used to spy on the reprojection helper.

        Test scenario:
            `sr_from_epsg(4326)` keeps GDAL's authority axis order, and
            `IsSame` is mapping-sensitive for a geographic CRS -- so a WGS 84
            raster compared unequal to WGS 84 and took the full reprojection
            path into its own CRS. Asserted on the *branch*, by spying on
            `reproject_coordinates`: a 4326-to-4326 warp is numerically a
            no-op, so the output alone cannot tell the two paths apart, which
            is exactly why the bug survived.
        """
        destination = sr_from_epsg(4326)
        assert destination.GetAxisMappingStrategy() != osr.OAMS_TRADITIONAL_GIS_ORDER
        calls: list[tuple] = []
        real = spatial_engine.reproject_coordinates

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(spatial_engine, "reproject_coordinates", spy)

        result = wgs84_raster.spatial._reproject_with_ReprojectImage(
            destination, gdal.GRA_NearestNeighbour
        )

        assert calls == [], "the same-CRS shortcut did not fire"
        assert tuple(result.geotransform) == tuple(wgs84_raster.geotransform)

    @pytest.mark.core
    def test_the_callers_spatial_reference_is_not_mutated(self, wgs84_raster):
        """Normalising on a clone, because the SRS belongs to the caller.

        Test scenario:
            Stamping the strategy onto the argument would change an object
            the caller may reuse -- and would make this method's effect
            depend on how many times it had been called.
        """
        destination = sr_from_epsg(4326)
        before = destination.GetAxisMappingStrategy()

        wgs84_raster.spatial._reproject_with_ReprojectImage(
            destination, gdal.GRA_NearestNeighbour
        )

        assert destination.GetAxisMappingStrategy() == before

    @pytest.mark.core
    def test_a_genuinely_different_crs_still_reprojects(
        self, wgs84_raster, monkeypatch
    ):
        """Widening the identity check must not swallow a real difference.

        Args:
            wgs84_raster: The fixture raster.
            monkeypatch: Fixture used to spy on the reprojection helper.

        Test scenario:
            Degrees into EPSG:3857 has to come back as metres, through the
            reprojection path -- an over-eager shortcut would take the
            identity branch and hand back the degrees unchanged.
        """
        calls: list[tuple] = []
        real = spatial_engine.reproject_coordinates

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(spatial_engine, "reproject_coordinates", spy)

        result = wgs84_raster.spatial._reproject_with_ReprojectImage(
            sr_from_epsg(3857), gdal.GRA_NearestNeighbour
        )

        assert calls, "the reprojection path was skipped for a different CRS"
        assert abs(result.geotransform[0]) > 1e5
        assert tuple(result.geotransform) != tuple(wgs84_raster.geotransform)
