"""Tests for :mod:`pyramids.dataset.ops.interpolate` (PB-2).

Covers :func:`grid_points` (and, by extension, the :meth:`Dataset.from_points`
classmethod facade): happy-path interpolation with several ``gdal.Grid``
algorithms, explicit ``width``/``height`` vs ``cell_size`` sizing, ``bbox`` and
``epsg`` overrides, CRS-less input, and every guard clause.
"""

from __future__ import annotations

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point

from pyramids.base._errors import FailedToSaveError
from pyramids.dataset import Dataset
from pyramids.dataset.ops.interpolate import _DEFAULT_ALGORITHM, grid_points
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def corner_points() -> FeatureCollection:
    """Four corner points of a 10x10 box, each with a distinct value.

    Returns:
        FeatureCollection: EPSG:4326 points at (0,0), (10,0), (0,10), (10,10)
        with values 10, 20, 30, 40 in column ``val``.
    """
    gdf = GeoDataFrame(
        {"val": [10.0, 20.0, 30.0, 40.0]},
        geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture(scope="function")
def crsless_points() -> FeatureCollection:
    """Same four corner points but without a CRS, to exercise the no-SRS path.

    Returns:
        FeatureCollection: CRS-less points with values in column ``val``.
    """
    gdf = GeoDataFrame(
        {"val": [10.0, 20.0, 30.0, 40.0]},
        geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
    )
    return FeatureCollection(gdf)


class TestGridPoints:
    """Tests for :func:`grid_points`."""

    def test_cell_size_sizing_and_bounds(self, corner_points):
        """grid_points derives width/height/extent from cell_size and bounds.

        Test scenario:
            A 0..10 box at cell_size=1 yields a 10x10 single-band raster whose
            geotransform starts at the top-left corner (0, 10).
        """
        ds = grid_points(corner_points, "val", Dataset, cell_size=1.0)
        assert ds.rows == 10, f"Expected 10 rows, got {ds.rows}"
        assert ds.columns == 10, f"Expected 10 columns, got {ds.columns}"
        assert ds.band_count == 1, f"Expected 1 band, got {ds.band_count}"
        geo = ds.geotransform
        assert geo[0] == pytest.approx(0.0), f"x-origin not at minx: {geo[0]}"
        assert geo[3] == pytest.approx(10.0), f"y-origin not at maxy: {geo[3]}"

    def test_invdist_interpolates_within_value_range(self, corner_points):
        """Default IDW produces values bounded by the sample values.

        Test scenario:
            Inverse-distance weighting of samples in [10, 40] must yield an
            interpolated surface whose values stay within that range.
        """
        ds = grid_points(corner_points, "val", Dataset, cell_size=1.0)
        arr = ds.read_array()
        assert float(np.nanmin(arr)) >= 10.0 - 1e-6, (
            f"min below range: {np.nanmin(arr)}"
        )
        assert float(np.nanmax(arr)) <= 40.0 + 1e-6, (
            f"max above range: {np.nanmax(arr)}"
        )

    def test_explicit_width_height_overrides_cell_size(self, corner_points):
        """Explicit width/height set the output shape directly.

        Test scenario:
            Passing width=20, height=15 (no cell_size) yields a 15x20 raster.
        """
        ds = grid_points(corner_points, "val", Dataset, width=20, height=15)
        assert ds.columns == 20, f"Expected 20 columns, got {ds.columns}"
        assert ds.rows == 15, f"Expected 15 rows, got {ds.rows}"

    def test_nearest_algorithm(self, corner_points):
        """A non-default algorithm string is honoured.

        Test scenario:
            algorithm='nearest' produces a valid raster of the requested size.
        """
        ds = grid_points(
            corner_points, "val", Dataset, algorithm="nearest", width=8, height=8
        )
        assert (ds.rows, ds.columns) == (
            8,
            8,
        ), f"Unexpected shape: {ds.rows}x{ds.columns}"

    def test_bbox_override_sets_extent(self, corner_points):
        """An explicit bbox overrides the points' total bounds.

        Test scenario:
            bbox=(-5, -5, 15, 15) with cell_size=1 yields a 20x20 raster whose
            top-left corner is (-5, 15).
        """
        ds = grid_points(
            corner_points, "val", Dataset, cell_size=1.0, bbox=(-5.0, -5.0, 15.0, 15.0)
        )
        assert (ds.rows, ds.columns) == (
            20,
            20,
        ), f"Unexpected shape: {ds.rows}x{ds.columns}"
        geo = ds.geotransform
        assert geo[0] == pytest.approx(-5.0), f"x-origin not at bbox minx: {geo[0]}"
        assert geo[3] == pytest.approx(15.0), f"y-origin not at bbox maxy: {geo[3]}"

    def test_epsg_override(self, corner_points):
        """An explicit epsg sets the output CRS regardless of the input CRS.

        Test scenario:
            epsg=3857 produces a raster reporting EPSG:3857.
        """
        ds = grid_points(corner_points, "val", Dataset, cell_size=1.0, epsg=3857)
        assert ds.epsg == 3857, f"Expected EPSG 3857, got {ds.epsg}"

    def test_crsless_points_produce_no_srs(self, crsless_points):
        """CRS-less input still interpolates; output carries no projection.

        Test scenario:
            With neither epsg nor a points CRS, gdal.Grid still returns a raster.
        """
        ds = grid_points(crsless_points, "val", Dataset, cell_size=1.0)
        assert (ds.rows, ds.columns) == (
            10,
            10,
        ), f"Unexpected shape: {ds.rows}x{ds.columns}"

    def test_default_algorithm_constant(self):
        """The module's default algorithm is inverse-distance weighting.

        Test scenario:
            The exported default string starts with ``invdist``.
        """
        assert _DEFAULT_ALGORITHM.startswith("invdist"), (
            f"Default algorithm should be IDW, got {_DEFAULT_ALGORITHM!r}"
        )

    def test_missing_value_column_raises(self, corner_points):
        """A value_column not present in the layer raises ValueError.

        Test scenario:
            Requesting an absent column reports it with the available columns.
        """
        with pytest.raises(ValueError, match="not in the points columns") as exc:
            grid_points(corner_points, "missing", Dataset, cell_size=1.0)
        assert "missing" in str(exc.value), (
            f"Column name absent from error: {exc.value}"
        )

    def test_degenerate_bounds_raises(self, corner_points):
        """A zero-area bbox raises ValueError before calling gdal.Grid.

        Test scenario:
            A collapsed bbox (minx==maxx) is rejected as degenerate.
        """
        with pytest.raises(ValueError, match="degenerate output bounds"):
            grid_points(
                corner_points, "val", Dataset, cell_size=1.0, bbox=(0.0, 0.0, 0.0, 10.0)
            )

    def test_collinear_points_degenerate_bounds_raises(self):
        """Collinear points produce zero-area bounds and are rejected.

        Test scenario:
            Three points sharing a y of 0 give maxy == miny -> degenerate.
        """
        gdf = GeoDataFrame(
            {"val": [1.0, 2.0, 3.0]},
            geometry=[Point(0, 0), Point(5, 0), Point(10, 0)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="degenerate output bounds"):
            grid_points(fc, "val", Dataset, cell_size=1.0)

    def test_no_sizing_raises(self, corner_points):
        """Omitting both cell_size and width/height raises ValueError.

        Test scenario:
            Without any sizing information the call cannot proceed.
        """
        with pytest.raises(ValueError, match="cell_size or both width and height"):
            grid_points(corner_points, "val", Dataset)

    def test_only_width_without_cell_size_raises(self, corner_points):
        """Supplying width but not height (nor cell_size) raises ValueError.

        Test scenario:
            Partial sizing (width only) is insufficient.
        """
        with pytest.raises(ValueError, match="cell_size or both width and height"):
            grid_points(corner_points, "val", Dataset, width=10)

    def test_failed_grid_raises(self, corner_points, monkeypatch):
        """A ``None`` return from gdal.Grid surfaces as FailedToSaveError.

        Test scenario:
            Monkeypatching gdal.Grid to return None triggers the guard.
        """
        from pyramids.dataset.ops import interpolate as interp_mod

        monkeypatch.setattr(interp_mod.gdal, "Grid", lambda *a, **k: None)
        with pytest.raises(FailedToSaveError, match="gdal.Grid returned no dataset"):
            grid_points(corner_points, "val", Dataset, cell_size=1.0)


class TestDatasetFromPoints:
    """Tests for the :meth:`Dataset.from_points` classmethod facade."""

    def test_from_points_delegates(self, corner_points):
        """Dataset.from_points returns an interpolated Dataset.

        Test scenario:
            The classmethod produces the same shape as the underlying
            grid_points call for equivalent arguments.
        """
        ds = Dataset.from_points(corner_points, "val", cell_size=1.0)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert (ds.rows, ds.columns) == (
            10,
            10,
        ), f"Unexpected shape: {ds.rows}x{ds.columns}"

    def test_from_points_algorithm_and_epsg(self, corner_points):
        """Dataset.from_points forwards algorithm and epsg overrides.

        Test scenario:
            A nearest-neighbour grid reprojected to EPSG:3857 of explicit size.
        """
        ds = Dataset.from_points(
            corner_points, "val", algorithm="nearest", width=12, height=12, epsg=3857
        )
        assert (ds.rows, ds.columns) == (
            12,
            12,
        ), f"Unexpected shape: {ds.rows}x{ds.columns}"
        assert ds.epsg == 3857, f"Expected EPSG 3857, got {ds.epsg}"
