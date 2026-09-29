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
from osgeo import gdal
from shapely.geometry import Point, Polygon

from pyramids.base._errors import FailedToSaveError
from pyramids.dataset import Dataset
from pyramids.dataset.ops import interpolate as interp_mod
from pyramids.dataset.ops.interpolate import _DEFAULT_ALGORITHM, grid_points
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr

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


class TestGridArrayFastPath:
    """The all-points fast path (CSV + OGR VRT) and its guards.

    A point layer skips the GeoJSON serialization and goes straight from
    coordinate arrays into ``gdal.Grid``; a non-point layer keeps the GeoJSON
    fallback. Both must produce a correct raster.
    """

    def test_fast_path_matches_geojson_reference(self, corner_points):
        """The array path grids identically to an independent GeoJSON grid.

        Test scenario:
            Grid four points through grid_points (the fast CSV+VRT path) and,
            separately, through gdal.Grid on a GeoJSON serialization of the same
            points; the two rasters must be pixel-for-pixel identical.
        """
        fast = np.asarray(
            grid_points(corner_points, "val", Dataset, cell_size=1.0).read_array()
        )
        options = gdal.GridOptions(
            format="MEM",
            algorithm=_DEFAULT_ALGORITHM,
            zfield="val",
            outputBounds=[0.0, 10.0, 10.0, 0.0],
            width=10,
            height=10,
            outputSRS=corner_points.crs.to_wkt(),
        )
        with _feature_ogr.as_vsimem_path(corner_points) as src_path:
            reference = gdal.Grid("", src_path, options=options)
        assert np.array_equal(fast, np.asarray(reference.ReadAsArray())), (
            "fast CSV+VRT path and the GeoJSON reference produced different grids"
        )

    def test_fast_path_matches_geojson_reference_for_nearest(self, corner_points):
        """The array path also matches GeoJSON under a non-default algorithm.

        Test scenario:
            The pixel-for-pixel equivalence between the CSV+VRT fast path and the
            GeoJSON round trip is otherwise only pinned for the default invdist
            algorithm; grid the same four points with ``nearest`` through both the
            fast path and an independent gdal.Grid-on-GeoJSON reference and require
            identical rasters, so a source-encoding divergence on a non-default
            algorithm cannot slip through.
        """
        fast = np.asarray(
            grid_points(
                corner_points, "val", Dataset, algorithm="nearest", cell_size=1.0
            ).read_array()
        )
        options = gdal.GridOptions(
            format="MEM",
            algorithm="nearest",
            zfield="val",
            outputBounds=[0.0, 10.0, 10.0, 0.0],
            width=10,
            height=10,
            outputSRS=corner_points.crs.to_wkt(),
        )
        with _feature_ogr.as_vsimem_path(corner_points) as src_path:
            reference = gdal.Grid("", src_path, options=options)
        assert np.array_equal(fast, np.asarray(reference.ReadAsArray())), (
            "fast CSV+VRT path and the GeoJSON reference diverged under nearest"
        )

    def test_value_column_named_x_survives_fast_path(self):
        """A value column named 'x' does not clash with the CSV's x coordinate.

        Test scenario:
            The fast path writes fixed x/y/z CSV columns, so gridding a column
            literally named 'x' interpolates the values, not the coordinates.
        """
        gdf = GeoDataFrame(
            {"x": [10.0, 20.0, 30.0, 40.0]},
            geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
            crs="EPSG:4326",
        )
        ds = grid_points(FeatureCollection(gdf), "x", Dataset, cell_size=1.0)
        arr = np.asarray(ds.read_array())
        assert float(np.nanmin(arr)) >= 10.0 - 1e-6, f"min below range: {arr.min()}"
        assert float(np.nanmax(arr)) <= 40.0 + 1e-6, f"max above range: {arr.max()}"

    def test_non_point_layer_uses_fallback(self):
        """A layer whose geometry is not all points still grids (GeoJSON fallback).

        Test scenario:
            A polygon layer carrying a value column is gridded via the fallback
            branch and returns a raster of the requested size.
        """
        gdf = GeoDataFrame(
            {"val": [1.0, 2.0]},
            geometry=[
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(8, 8), (10, 8), (10, 10), (8, 10)]),
            ],
            crs="EPSG:4326",
        )
        ds = grid_points(
            FeatureCollection(gdf), "val", Dataset, cell_size=1.0, bbox=(0, 0, 10, 10)
        )
        assert (ds.rows, ds.columns) == (10, 10), (
            f"fallback produced unexpected shape: {ds.rows}x{ds.columns}"
        )


class TestDatasetFromPointArrays:
    """Tests for :meth:`Dataset.from_point_arrays` — gridding from raw arrays."""

    def test_grids_from_arrays(self):
        """Raw x/y/z arrays grid to a Dataset without any FeatureCollection.

        Test scenario:
            Four corner readings gridded at cell_size=1 over their own 0..10 extent
            give a 10x10 single-band Dataset labelled EPSG:4326.
        """
        ds = Dataset.from_point_arrays(
            [0.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 20.0, 30.0, 40.0],
            cell_size=1.0,
            epsg=4326,
        )
        assert (ds.rows, ds.columns, ds.band_count) == (10, 10, 1), (
            f"unexpected shape: {ds.rows}x{ds.columns}x{ds.band_count}"
        )
        assert ds.epsg == 4326, f"Expected EPSG 4326, got {ds.epsg}"

    def test_agrees_with_from_points_entry(self):
        """The two public entries stay in agreement (both delegate to the core).

        Test scenario:
            After the refactor Dataset.from_points (point layer) and
            Dataset.from_point_arrays both funnel through grid_arrays, so this is a
            contract test that the two entries keep producing the same raster for
            the same points -- a guard against one entry drifting -- not an
            independent-implementation comparison (that guard is
            test_fast_path_matches_geojson_reference, which grids via GeoJSON).
        """
        x = [0.0, 10.0, 0.0, 10.0]
        y = [0.0, 0.0, 10.0, 10.0]
        z = [10.0, 20.0, 30.0, 40.0]
        gdf = GeoDataFrame(
            {"z": z}, geometry=[Point(a, b) for a, b in zip(x, y)], crs="EPSG:4326"
        )
        via_fc = Dataset.from_points(
            FeatureCollection(gdf), "z", cell_size=1.0
        ).read_array()
        via_arrays = Dataset.from_point_arrays(
            x, y, z, cell_size=1.0, epsg=4326
        ).read_array()
        assert np.array_equal(np.asarray(via_fc), np.asarray(via_arrays)), (
            "from_point_arrays and from_points produced different grids"
        )

    def test_bbox_and_explicit_size(self):
        """bbox sets the extent and width/height set the shape directly.

        Test scenario:
            A bbox wider than the points, with explicit width/height, produces a
            raster of exactly that shape and origin.
        """
        ds = Dataset.from_point_arrays(
            [0.0, 5.0],
            [0.0, 5.0],
            [1.0, 2.0],
            width=20,
            height=15,
            bbox=(-5, -5, 15, 15),
        )
        assert (ds.rows, ds.columns) == (15, 20), (
            f"unexpected shape: {ds.rows}x{ds.columns}"
        )
        assert ds.geotransform[0] == pytest.approx(-5.0), "x-origin not at bbox minx"

    def test_crs_argument_labels_output(self):
        """A crs= argument (not just epsg) labels the output raster.

        Test scenario:
            crs="EPSG:3857" is honoured the same as epsg=3857.
        """
        ds = Dataset.from_point_arrays(
            [0.0, 5.0, 0.0, 5.0],
            [0.0, 0.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 4.0],
            cell_size=1.0,
            crs="EPSG:3857",
        )
        assert ds.epsg == 3857, f"Expected EPSG 3857, got {ds.epsg}"

    def test_empty_arrays_raise(self):
        """Empty inputs raise ValueError before touching gdal.Grid.

        Test scenario:
            Zero-length arrays are rejected with a clear message.
        """
        with pytest.raises(ValueError, match="at least one point"):
            Dataset.from_point_arrays([], [], [], cell_size=1.0)

    def test_length_mismatch_raises(self):
        """Unequal-length arrays raise ValueError.

        Test scenario:
            x/y of length 2 with z of length 3 is rejected.
        """
        with pytest.raises(ValueError, match="equal length"):
            Dataset.from_point_arrays(
                [0.0, 1.0], [0.0, 1.0], [1.0, 2.0, 3.0], cell_size=1.0
            )

    def test_negative_cell_size_raises(self):
        """A negative cell_size is rejected, not clamped to a 1x1 raster.

        Test scenario:
            Before the guard, max(1, round(negative)) silently produced a 1x1 grid.
        """
        with pytest.raises(ValueError, match="cell_size must be positive"):
            Dataset.from_point_arrays(
                [0.0, 10.0], [0.0, 10.0], [1.0, 2.0], cell_size=-1.0
            )

    def test_zero_cell_size_raises_valueerror(self):
        """A zero cell_size raises ValueError, not a bare ZeroDivisionError.

        Test scenario:
            The documented exception type is ValueError; 0.0 used to divide by zero.
        """
        with pytest.raises(ValueError, match="cell_size must be positive"):
            Dataset.from_point_arrays(
                [0.0, 10.0], [0.0, 10.0], [1.0, 2.0], cell_size=0.0
            )

    def test_no_sizing_raises(self):
        """Omitting both cell_size and width/height raises ValueError.

        Test scenario:
            Without any sizing the call cannot proceed.
        """
        with pytest.raises(ValueError, match="cell_size or both width and height"):
            Dataset.from_point_arrays([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])

    def test_two_dimensional_input_raises(self):
        """A 2-D coordinate array is rejected by the array core.

        Test scenario:
            Passing a (2, 2) array for x reports that 1-D arrays are required.
        """
        with pytest.raises(ValueError, match="grid_arrays expects 1-D arrays"):
            Dataset.from_point_arrays(
                np.array([[0.0, 1.0], [2.0, 3.0]]),
                np.array([0.0, 1.0]),
                np.array([1.0, 2.0]),
                cell_size=1.0,
            )

    def test_crs_and_epsg_label_consistently(self):
        """crs= (string or int) and epsg= all stamp the same EPSG on the output.

        Test scenario:
            The review noted the crs= path builds its output SRS via a different WKT
            flavour than the epsg= path; this pins that all three spellings resolve
            to the same EPSG on the result.
        """
        pts = ([0.0, 5.0, 0.0, 5.0], [0.0, 0.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0])
        by_epsg = Dataset.from_point_arrays(*pts, cell_size=1.0, epsg=3857).epsg
        by_crs_str = Dataset.from_point_arrays(
            *pts, cell_size=1.0, crs="EPSG:3857"
        ).epsg
        by_crs_int = Dataset.from_point_arrays(*pts, cell_size=1.0, crs=3857).epsg
        assert by_epsg == by_crs_str == by_crs_int == 3857, (
            f"crs/epsg spellings disagree: {by_epsg}, {by_crs_str}, {by_crs_int}"
        )


class TestNonFiniteInputs:
    """Non-finite handling in the array grid core (findings M1, M2).

    A NaN *value* is a missing reading and must be dropped (matching the GeoJSON
    path), not fed to gdal.Grid as a real 0.0; a non-finite *coordinate* or bbox is
    malformed and must raise a clear ValueError.
    """

    def test_nan_value_matches_geojson_drop(self):
        """A partial-NaN value column grids identically through both paths.

        Test scenario:
            Four points with one NaN reading, gridded via the public
            Dataset.from_points (the CSV fast path) must equal an independent
            gdal.Grid-on-GeoJSON reference, which drops the NaN feature. Before the
            fix the CSV path read NaN as 0.0 and diverged badly (23.99 -> 1.38 at
            the corner).
        """
        x = [0.0, 10.0, 0.0, 10.0]
        y = [0.0, 0.0, 10.0, 10.0]
        z = [10.0, 20.0, float("nan"), 40.0]
        gdf = GeoDataFrame(
            {"z": z}, geometry=[Point(a, b) for a, b in zip(x, y)], crs="EPSG:4326"
        )
        fast = np.asarray(
            Dataset.from_points(FeatureCollection(gdf), "z", cell_size=2.0).read_array()
        )
        options = gdal.GridOptions(
            format="MEM",
            algorithm=_DEFAULT_ALGORITHM,
            zfield="z",
            outputBounds=[0.0, 10.0, 10.0, 0.0],
            width=5,
            height=5,
            outputSRS=gdf.crs.to_wkt(),
        )
        with _feature_ogr.as_vsimem_path(FeatureCollection(gdf)) as src_path:
            ref = np.asarray(gdal.Grid("", src_path, options=options).ReadAsArray())
        assert np.allclose(fast, ref, equal_nan=True), (
            "NaN value handling diverges from the GeoJSON path"
        )

    def test_partial_nan_equals_explicit_drop(self):
        """Gridding with a NaN value equals gridding the finite points alone.

        Test scenario:
            The same four points with one NaN, and the three finite points, over an
            identical bbox, produce the identical raster.
        """
        bbox = (0.0, 0.0, 10.0, 10.0)
        with_nan = Dataset.from_point_arrays(
            [0.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 20.0, float("nan"), 40.0],
            cell_size=2.0,
            bbox=bbox,
        ).read_array()
        finite_only = Dataset.from_point_arrays(
            [0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0],
            [10.0, 20.0, 40.0],
            cell_size=2.0,
            bbox=bbox,
        ).read_array()
        assert np.array_equal(np.asarray(with_nan), np.asarray(finite_only)), (
            "dropping a NaN value must equal omitting the point"
        )

    def test_vsimem_cleanup_tolerates_a_missing_file(self, monkeypatch):
        """When the first write fails, the finally unlinks absent paths harmlessly.

        Test scenario:
            The CSV write raises, so neither /vsimem file is created; the finally's
            Unlink runs against absent paths, its guard swallows the resulting
            error, and the original failure still propagates.
        """
        monkeypatch.setattr(
            interp_mod.gdal,
            "FileFromMemBuffer",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("write blew up")),
        )
        with pytest.raises(RuntimeError, match="write blew up"):
            Dataset.from_point_arrays(
                [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0], cell_size=1.0
            )

    def test_no_vsimem_leak_when_the_vrt_write_fails(self, monkeypatch):
        """The real leak scenario: CSV written, VRT write fails, CSV must be freed.

        Test scenario:
            The first write (CSV) succeeds, the second (VRT) raises. The earlier
            cleanup test only forced the *first* write to fail, so no file was ever
            created; this drives the branch the leak guard exists for and asserts no
            ``grid_*`` file is left in /vsimem afterwards.
        """
        real_write = interp_mod.gdal.FileFromMemBuffer
        calls = {"n": 0}

        def write_then_fail(path, buf):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("vrt write blew up")
            return real_write(path, buf)

        monkeypatch.setattr(interp_mod.gdal, "FileFromMemBuffer", write_then_fail)
        with pytest.raises(RuntimeError, match="vrt write blew up"):
            Dataset.from_point_arrays(
                [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0], cell_size=1.0
            )
        leftovers = [
            name for name in (gdal.ReadDir("/vsimem") or []) if name.startswith("grid_")
        ]
        assert leftovers == [], (
            f"/vsimem leaked files after a failed write: {leftovers}"
        )

    def test_all_nan_values_raise(self):
        """An all-non-finite value column leaves nothing to interpolate.

        Test scenario:
            Every value NaN raises a clear ValueError rather than an empty grid.
        """
        with pytest.raises(ValueError, match="every value is non-finite"):
            Dataset.from_point_arrays(
                [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [float("nan")] * 3, cell_size=1.0
            )

    def test_nan_coordinate_raises(self):
        """A NaN coordinate is malformed and raises a clear ValueError.

        Test scenario:
            Before the guard this surfaced as 'cannot convert float NaN to integer'.
        """
        with pytest.raises(ValueError, match="x and y must be finite"):
            Dataset.from_point_arrays(
                [0.0, float("nan")], [0.0, 1.0], [1.0, 2.0], cell_size=1.0
            )

    def test_inf_coordinate_raises(self):
        """An infinite coordinate raises ValueError, not OverflowError.

        Test scenario:
            inf in x is rejected up front instead of reaching round()/gdal.Grid.
        """
        with pytest.raises(ValueError, match="x and y must be finite"):
            Dataset.from_point_arrays(
                [0.0, float("inf")], [0.0, 1.0], [1.0, 2.0], cell_size=1.0
            )

    def test_nonfinite_bbox_raises(self):
        """A non-finite bbox is rejected before it reaches gdal.Grid.

        Test scenario:
            With explicit width/height (which skips the sizing round()), an inf bbox
            edge would otherwise flow straight into gdal.GridOptions.
        """
        with pytest.raises(ValueError, match="bbox must be finite"):
            Dataset.from_point_arrays(
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 2.0],
                width=5,
                height=5,
                bbox=(0.0, 0.0, float("inf"), 10.0),
            )

    def test_extent_from_all_coords_when_boundary_value_is_nan(self):
        """With no bbox, a NaN-valued boundary point still sets the extent.

        Test scenario:
            The sole extreme point (20, 20) carries a NaN value, so it is dropped
            before interpolation -- but the extent must be derived from *all*
            coordinates first, so the raster still spans to (20, 20). Were the drop
            applied before the bounds computation (extent from survivors only), the
            grid would silently shrink to the 0..10 box of the four finite points.
            This is the load-bearing "bounds from all coords, values from survivors"
            split that both NaN tests miss by passing an explicit bbox.
        """
        ds = Dataset.from_point_arrays(
            [0.0, 10.0, 0.0, 10.0, 20.0],
            [0.0, 0.0, 10.0, 10.0, 20.0],
            [10.0, 20.0, 30.0, 40.0, float("nan")],
            cell_size=1.0,
            epsg=4326,
        )
        assert (ds.rows, ds.columns) == (20, 20), (
            f"extent shrank to survivors: {ds.rows}x{ds.columns} (expected 20x20)"
        )
        assert ds.geotransform[3] == pytest.approx(20.0), (
            f"y-origin not at the NaN point's maxy: {ds.geotransform[3]}"
        )
        bottom_right_x = ds.geotransform[0] + ds.columns * ds.geotransform[1]
        assert bottom_right_x == pytest.approx(20.0), (
            f"extent does not reach the NaN point's maxx: {bottom_right_x}"
        )

    def test_nan_point_coordinate_raises_through_from_points(self):
        """A NaN point geometry now raises through the from_points FC entry.

        Test scenario:
            Before the refactor a point FeatureCollection carrying POINT (nan nan)
            flowed through the GeoJSON path into gdal.Grid; the shared grid_arrays
            core now rejects the non-finite coordinate up front, so the public
            Dataset.from_points entry raises a clear ValueError.
        """
        gdf = GeoDataFrame(
            {"z": [1.0, 2.0, 3.0]},
            geometry=[Point(0, 0), Point(10, 0), Point(float("nan"), float("nan"))],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="x and y must be finite"):
            Dataset.from_points(fc, "z", cell_size=1.0)


class TestGridPointsFallbackBranch:
    """The non-point GeoJSON fallback in :func:`grid_points`."""

    @staticmethod
    def _polygon_layer() -> FeatureCollection:
        """Two disjoint square polygons carrying a value column.

        Returns:
            FeatureCollection: A non-point layer that forces the fallback path.
        """
        gdf = GeoDataFrame(
            {"val": [1.0, 2.0]},
            geometry=[
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(8, 8), (10, 8), (10, 10), (8, 10)]),
            ],
            crs="EPSG:4326",
        )
        return FeatureCollection(gdf)

    def test_fallback_without_bbox_uses_total_bounds(self):
        """A non-point layer with no bbox derives the extent from total_bounds.

        Test scenario:
            Two polygons spanning 0..10 gridded at cell_size=1 (no bbox) give a
            10x10 raster whose origin is the layer's own top-left.
        """
        ds = grid_points(self._polygon_layer(), "val", Dataset, cell_size=1.0)
        assert (ds.rows, ds.columns) == (10, 10), (
            f"unexpected shape from total_bounds: {ds.rows}x{ds.columns}"
        )
        assert ds.geotransform[3] == pytest.approx(10.0), "y-origin not at maxy"

    def test_fallback_failed_grid_raises(self, monkeypatch):
        """A None from gdal.Grid on the fallback path surfaces FailedToSaveError.

        Test scenario:
            With gdal.Grid stubbed to return None, gridding a polygon layer raises.
        """
        monkeypatch.setattr(interp_mod.gdal, "Grid", lambda *a, **k: None)
        layer = self._polygon_layer()
        with pytest.raises(FailedToSaveError, match="gdal.Grid returned no dataset"):
            grid_points(layer, "val", Dataset, cell_size=1.0)
