"""Tests for :meth:`Dataset.contour` / ``Vectorize.contour`` (PB-4).

Covers contour-line and contour-polygon generation via ``gdal.ContourGenerateEx``:
regular ``interval`` levels, explicit ``fixed_levels``, the ``base`` anchor,
custom attribute naming, no-data handling, the empty-result case, and every
guard clause. All fixtures are in-memory ``create_from_array`` rasters.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def ramp() -> Dataset:
    """A 10x10 raster ramping 0..9 across columns (constant down each column).

    Returns:
        Dataset: EPSG:4326 ramp suitable for predictable horizontal contours.
    """
    arr = np.tile(np.arange(10, dtype=np.float32), (10, 1))
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 10), cell_size=1.0, epsg=4326
    )


@pytest.fixture(scope="function")
def flat() -> Dataset:
    """A 6x6 constant raster (value 5) — produces no contour crossings.

    Returns:
        Dataset: EPSG:4326 constant raster.
    """
    arr = np.full((6, 6), 5.0, dtype=np.float32)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
    )


class TestContour:
    """Tests for the ``Vectorize.contour`` engine method via ``Dataset.contour``."""

    def test_interval_lines(self, ramp):
        """A regular interval yields one LineString per level it crosses.

        Test scenario:
            A 0..9 ramp contoured at interval=2 (base=0) crosses levels
            2, 4, 6, 8 — four LineString features carrying those elevations.
        """
        fc = ramp.contour(interval=2.0)
        assert isinstance(fc, FeatureCollection), (
            f"Expected FeatureCollection, got {type(fc)}"
        )
        assert len(fc) == 4, f"Expected 4 contour lines, got {len(fc)}"
        assert sorted(fc["elev"].tolist()) == [
            2.0,
            4.0,
            6.0,
            8.0,
        ], f"Unexpected elevations: {sorted(fc['elev'].tolist())}"
        assert fc.geometry.geom_type.unique().tolist() == ["LineString"], (
            f"Expected LineString geometries, got {fc.geometry.geom_type.unique().tolist()}"
        )

    def test_crs_preserved(self, ramp):
        """The output FeatureCollection carries the source raster CRS.

        Test scenario:
            An EPSG:4326 source yields contours reporting EPSG:4326.
        """
        fc = ramp.contour(interval=2.0)
        assert fc.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {fc.crs}"

    def test_fixed_levels(self, ramp):
        """Explicit fixed_levels contour exactly the requested values.

        Test scenario:
            fixed_levels=[3, 5, 7] yields three lines at those elevations.
        """
        fc = ramp.contour(fixed_levels=[3.0, 5.0, 7.0])
        assert sorted(fc["elev"].tolist()) == [
            3.0,
            5.0,
            7.0,
        ], f"Unexpected elevations: {sorted(fc['elev'].tolist())}"

    def test_base_offset(self, ramp):
        """The base anchor shifts the regular interval grid.

        Test scenario:
            interval=2, base=1 yields odd levels 1, 3, 5, 7, 9 within 0..9.
        """
        fc = ramp.contour(interval=2.0, base=1.0)
        assert sorted(fc["elev"].tolist()) == [
            1.0,
            3.0,
            5.0,
            7.0,
            9.0,
        ], f"Unexpected elevations: {sorted(fc['elev'].tolist())}"

    def test_custom_attribute_name(self, ramp):
        """The elevation attribute is written under the requested column name.

        Test scenario:
            attribute='height' produces a 'height' column, not 'elev'.
        """
        fc = ramp.contour(interval=2.0, attribute="height")
        assert "height" in fc.columns, f"'height' column missing: {list(fc.columns)}"
        assert "elev" not in fc.columns, f"'elev' should be absent: {list(fc.columns)}"

    def test_polygonize(self, ramp):
        """Polygon mode emits filled bands with min/max elevation columns.

        Test scenario:
            polygonize=True yields (Multi)Polygon geometries and the
            'elev_min' / 'elev_max' attribute pair.
        """
        fc = ramp.contour(interval=2.0, polygonize=True)
        geom_types = set(fc.geometry.geom_type.unique().tolist())
        assert geom_types <= {
            "Polygon",
            "MultiPolygon",
        }, f"Unexpected geom types: {geom_types}"
        assert "elev_min" in fc.columns and "elev_max" in fc.columns, (
            f"min/max columns missing: {list(fc.columns)}"
        )

    def test_polygonize_custom_attribute(self, ramp):
        """Polygon mode honours the custom attribute base name.

        Test scenario:
            attribute='z' with polygonize=True produces 'z_min' and 'z_max'.
        """
        fc = ramp.contour(interval=2.0, polygonize=True, attribute="z")
        assert "z_min" in fc.columns and "z_max" in fc.columns, (
            f"z_min/z_max columns missing: {list(fc.columns)}"
        )

    def test_band_selection(self):
        """A non-default band is contoured when band= is supplied.

        Test scenario:
            For a 2-band raster whose band 1 ramps 0..4, contouring band=1 at
            interval=2 crosses levels 2 and 4.
        """
        flat_band = np.zeros((5, 5), dtype=np.float32)
        ramp_band = np.tile(np.arange(5, dtype=np.float32), (5, 1))
        arr = np.stack([flat_band, ramp_band])
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 5), cell_size=1.0, epsg=4326
        )
        fc = ds.contour(interval=2.0, band=1)
        assert sorted(fc["elev"].tolist()) == [
            2.0,
            4.0,
        ], f"Unexpected elevations for band 1: {sorted(fc['elev'].tolist())}"

    def test_nodata_handled(self):
        """A band with a no-data value still contours its valid pixels.

        Test scenario:
            A ramp whose first column is no-data still yields interval contours
            from the remaining valid gradient.
        """
        arr = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        arr[:, 0] = -9999.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0, 10),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        fc = ds.contour(interval=2.0)
        assert len(fc) > 0, "Expected contours from the valid gradient, got none"
        assert min(fc["elev"].tolist()) >= 2.0, (
            f"No-data region should not produce negative contours: {sorted(fc['elev'].tolist())}"
        )

    def test_no_nodata_value(self):
        """A band with no no-data value set still contours (NODATA option omitted).

        Test scenario:
            A ramp created with no_data_value=None has GetNoDataValue() == None,
            exercising the branch that skips the NODATA contour option.
        """
        arr = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0, 10), cell_size=1.0, epsg=4326, no_data_value=None
        )
        assert ds.raster.GetRasterBand(1).GetNoDataValue() is None, (
            "fixture should have no nodata"
        )
        fc = ds.contour(interval=2.0)
        assert sorted(fc["elev"].tolist()) == [
            2.0,
            4.0,
            6.0,
            8.0,
        ], f"Unexpected elevations: {sorted(fc['elev'].tolist())}"

    def test_flat_raster_empty(self, flat):
        """A constant raster produces an empty FeatureCollection, not an error.

        Test scenario:
            A flat value-5 raster contoured at interval=2 has no level crossings.
        """
        fc = flat.contour(interval=2.0)
        assert isinstance(fc, FeatureCollection), (
            f"Expected FeatureCollection, got {type(fc)}"
        )
        assert len(fc) == 0, f"Expected an empty collection, got {len(fc)} features"

    def test_neither_interval_nor_levels_raises(self, ramp):
        """Omitting both interval and fixed_levels raises ValueError.

        Test scenario:
            With no level specification the call cannot proceed.
        """
        with pytest.raises(ValueError, match="exactly one of interval or fixed_levels"):
            ramp.contour()

    def test_both_interval_and_levels_raises(self, ramp):
        """Supplying both interval and fixed_levels raises ValueError.

        Test scenario:
            The two level specifications are mutually exclusive.
        """
        with pytest.raises(ValueError, match="exactly one of interval or fixed_levels"):
            ramp.contour(interval=2.0, fixed_levels=[1.0])

    def test_non_positive_interval_raises(self, ramp):
        """A non-positive interval raises ValueError.

        Test scenario:
            interval=0 is rejected.
        """
        with pytest.raises(ValueError, match="interval must be positive"):
            ramp.contour(interval=0.0)

    def test_empty_fixed_levels_raises(self, ramp):
        """An empty fixed_levels list raises ValueError.

        Test scenario:
            fixed_levels=[] specifies no levels.
        """
        with pytest.raises(ValueError, match="at least one level"):
            ramp.contour(fixed_levels=[])

    def test_band_out_of_range_raises(self, ramp):
        """A band index beyond the raster raises ValueError.

        Test scenario:
            band=5 on a 1-band raster is out of range.
        """
        with pytest.raises(ValueError, match="out of range"):
            ramp.contour(interval=2.0, band=5)

    def test_datasource_creation_failure_raises(self, ramp, monkeypatch):
        """A failed in-memory OGR DataSource creation raises RuntimeError.

        Test scenario:
            Monkeypatching the MEM driver to return None for CreateDataSource
            triggers the defensive guard.
        """
        from pyramids.dataset.engines import vectorize as vec_mod

        class _NullDriver:
            def CreateDataSource(self, *args, **kwargs):
                return None

        monkeypatch.setattr(vec_mod.ogr, "GetDriverByName", lambda name: _NullDriver())
        with pytest.raises(
            RuntimeError, match="Failed to create in-memory OGR DataSource"
        ):
            ramp.contour(interval=2.0)
