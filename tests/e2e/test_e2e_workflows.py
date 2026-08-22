"""End-to-end workflow tests.

These tests exercise multi-step pipelines that combine reading, creating,
cropping, reprojecting, aligning, and round-tripping raster and vector data.

Workflows covered:
1. Create GeoTIFF from array -> crop with polygon -> extract values -> verify
2. Create DatasetCollection -> save -> reload -> verify shapes
3. FeatureCollection -> to_dataset (rasterize) -> extract -> verify round-trip
4. Read GeoTIFF -> reproject -> align with another -> verify dimensions match
"""

import shutil
import tempfile
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr
from shapely.geometry import Point, box

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.ops.vectorize import _features_outside_template
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


def _make_dataset(
    rows: int = 10,
    cols: int = 10,
    epsg: int = 32636,
    cell_size: float = 1000.0,
    top_left: tuple = (500000.0, 3400000.0),
    no_data: float = -9999.0,
    fill_value: float = 0.0,
) -> Dataset:
    """Create a simple in-memory Dataset."""
    src = Dataset.create(
        cell_size=cell_size,
        rows=rows,
        columns=cols,
        dtype="float32",
        bands=1,
        top_left_corner=top_left,
        epsg=epsg,
        no_data_value=no_data,
    )
    arr = np.full((rows, cols), fill_value, dtype=np.float32)
    src.raster.GetRasterBand(1).WriteArray(arr)
    src.raster.FlushCache()
    return src


class TestCreateCropExtract:
    """Create a raster, crop it with a polygon mask, then extract values."""

    def test_create_crop_extract(self):
        """Full pipeline: create -> populate -> crop -> verify extracted values."""
        rows, cols = 20, 20
        cell_size = 1000.0
        epsg = 32636
        top_left = (500000.0, 3400000.0)

        # Step 1 - Create dataset with sequential values
        src = Dataset.create(
            cell_size=cell_size,
            rows=rows,
            columns=cols,
            dtype="float32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=-9999.0,
        )
        arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        src.raster.GetRasterBand(1).WriteArray(arr)

        # Step 2 - Create a polygon that covers the top-left 5x5 cells
        x0, y0 = top_left
        poly = box(x0, y0 - 5 * cell_size, x0 + 5 * cell_size, y0)
        mask_gdf = gpd.GeoDataFrame(geometry=[poly], crs=f"EPSG:{epsg}")

        # Step 3 - Crop
        cropped = src.crop(mask_gdf)

        # Step 4 - Verify
        assert cropped is not None, "crop should return a new Dataset"
        cropped_arr = cropped.read_array()
        assert cropped_arr.shape[0] <= rows, "Cropped rows should be <= original"
        assert cropped_arr.shape[1] <= cols, "Cropped cols should be <= original"
        # The cropped area (top-left 5x5) should contain values 0-4, 20-24 etc.
        non_nodata = cropped_arr[
            ~np.isclose(cropped_arr, cropped.no_data_value[0], rtol=0.001)
        ]
        assert non_nodata.size > 0, "Cropped raster should contain some valid data"


class TestDatasetCollectionRoundTrip:
    """Create a DatasetCollection, save it, reload, and verify."""

    def test_save_and_reload(self):
        """Write DatasetCollection to disk, read back, compare shapes."""
        rows, cols = 8, 10
        time_steps = 3

        base = _make_dataset(rows=rows, cols=cols, fill_value=1.0)
        md = DatasetCollection.from_dataset(base, time_length=time_steps)
        values = (
            np.random.default_rng(0).random((time_steps, rows, cols)).astype(np.float64)
        )
        md.values = values

        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "multidataset_output"
        try:
            md.to_file(out_dir)

            # Reload
            reloaded = DatasetCollection.read_multiple_files(out_dir, with_order=False)
            assert reloaded.time_length == time_steps, (
                f"Expected {time_steps} files, got {reloaded.time_length}"
            )
            assert reloaded.base.rows == rows, (
                f"Reloaded rows mismatch: expected {rows}"
            )
            assert reloaded.base.columns == cols, (
                f"Reloaded columns mismatch: expected {cols}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestRasterizeRoundTrip:
    """Rasterize a FeatureCollection and verify the burned values."""

    @pytest.fixture
    def utm_template(self):
        """A 10x10 UTM (EPSG:32636) template at a fixed origin, shared by template tests."""
        return _make_dataset(
            rows=10,
            cols=10,
            cell_size=1000.0,
            top_left=(500000.0, 3400000.0),
            epsg=32636,
        )

    @staticmethod
    def _fc_32636(geometry, value=42):
        """A single-feature EPSG:32636 FeatureCollection with a `class_id` column."""
        return FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": [value]}, geometry=[geometry], crs="EPSG:32636"
            )
        )

    @staticmethod
    def _mem_template(geotransform, epsg=4326):
        """A 5x5 in-memory Dataset with a custom geotransform (non-square/rotated tests)."""
        raster = gdal.GetDriverByName("MEM").Create("", 5, 5, 1, gdal.GDT_Float32)
        raster.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        raster.SetProjection(srs.ExportToWkt())
        return Dataset(raster)

    def test_rasterize_polygon(self):
        """Burn a polygon attribute into a raster and verify the value."""
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        # Create a polygon covering a 5x5 area
        x0, y0 = top_left
        poly = box(x0, y0 - 5 * cell_size, x0 + 5 * cell_size, y0)
        gdf = gpd.GeoDataFrame({"burn_val": [7]}, geometry=[poly], crs=f"EPSG:{epsg}")
        fc = FeatureCollection(gdf)

        # Rasterize: use cell_size (no reference dataset)
        raster = Dataset.from_features(fc, cell_size=cell_size, column_name="burn_val")
        arr = raster.read_array()

        # Verify burned value
        burned = arr[np.isclose(arr, 7.0)]
        assert burned.size > 0, "At least some cells should contain the burned value 7"
        assert raster.epsg == epsg, (
            f"Rasterized EPSG should be {epsg}, got {raster.epsg}"
        )

    def test_rasterize_with_reference_dataset(self, utm_template):
        """Burn an inside polygon onto a template; it appears and emits no outside warning."""
        x0, y0 = utm_template.top_left_corner
        cell = utm_template.cell_size
        inside = box(x0, y0 - 3 * cell, x0 + 3 * cell, y0)
        fc = FeatureCollection(
            gpd.GeoDataFrame({"class_id": [42]}, geometry=[inside], crs="EPSG:32636")
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raster = Dataset.from_features(
                fc, template=utm_template, column_name="class_id"
            )

        arr = raster.read_array()
        assert arr.shape == (10, 10), (
            f"shape should match the template, got {arr.shape}"
        )
        assert np.any(np.isclose(arr, 42.0)), (
            "burned value 42 should appear in the raster"
        )
        assert not [w for w in caught if "template extent" in str(w.message)], (
            "an inside polygon must not emit the outside-template warning"
        )

    def test_from_features_warns_when_features_outside_template(self, utm_template):
        """Features disjoint from the template warn and yield an all-nodata raster (#46)."""
        x0, y0 = utm_template.top_left_corner
        far = box(
            x0 + 100000.0, y0 - 3000.0, x0 + 103000.0, y0
        )  # ~100 km east, outside
        fc = FeatureCollection(
            gpd.GeoDataFrame({"class_id": [42]}, geometry=[far], crs="EPSG:32636")
        )

        with pytest.warns(UserWarning, match="outside the template extent"):
            raster = Dataset.from_features(
                fc, template=utm_template, column_name="class_id"
            )

        assert not np.any(np.isclose(raster.read_array(), 42.0)), (
            "a polygon outside the template burns nothing"
        )

    @pytest.mark.parametrize(
        "feature_box, expected",
        [
            ((20.0, 2.0, 30.0, 8.0), True),  # east of the template
            ((-30.0, 2.0, -20.0, 8.0), True),  # west of the template
            ((2.0, 20.0, 8.0, 30.0), True),  # north of the template
            ((2.0, -30.0, 8.0, -20.0), True),  # south of the template
            ((2.0, 2.0, 8.0, 8.0), False),  # inside the template
            ((5.0, 5.0, 15.0, 15.0), False),  # partial overlap (not flagged)
        ],
    )
    def test_features_outside_template_by_direction(self, feature_box, expected):
        """`_features_outside_template` flags a disjoint bbox in every direction (#46).

        Args:
            feature_box: `(minx, miny, maxx, maxy)` of the feature relative to a template
                covering x[0, 10], y[0, 10].
            expected: Whether the helper should report the feature as fully outside.

        Test scenario:
            A template at origin (0, 10), 10x10, cell size 1 covers x[0, 10] and y[0, 10].
            Boxes to the east/west/north/south are disjoint (`True`); an inside box and a
            partially overlapping box are not flagged (`False`).
        """
        minx, miny, maxx, maxy = feature_box
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        template = _make_dataset(
            rows=10, cols=10, epsg=4326, cell_size=1.0, top_left=(0.0, 10.0)
        )
        result = _features_outside_template(fc, template)
        assert result is expected, (
            f"box {feature_box} outside-template should be {expected}, got {result}"
        )

    def test_features_outside_non_square_template_on_y_axis(self):
        """A non-square template flags a feature outside on the Y axis (#46 M1).

        Test scenario:
            A template with X pixel 2 and Y pixel 1 (`gt=(0, 2, 0, 10, 0, -1)`, 5x5) has
            true bbox x[0, 10], y[5, 10]. A feature at `box(2, 2, 4, 4)` is south of the
            true `ymin=5`, so it must be flagged — a single-`cell_size` check would use the
            X pixel for Y and miss it.
        """
        template = self._mem_template((0.0, 2.0, 0.0, 10.0, 0.0, -1.0))
        south = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]}, geometry=[box(2.0, 2.0, 4.0, 4.0)], crs="EPSG:4326"
            )
        )
        assert _features_outside_template(south, template) is True, (
            "a feature south of a non-square template must be flagged outside"
        )

    def test_features_touching_template_edge_is_outside(self):
        """A feature touching a template edge (zero-area overlap) is flagged outside (#46).

        Test scenario:
            A box whose left edge coincides with the template's right edge (`fxmin == txmax`)
            has zero-area overlap and is treated as outside, pinning the `>=`/`<=` semantics.
        """
        template = _make_dataset(
            rows=10, cols=10, epsg=4326, cell_size=1.0, top_left=(0.0, 10.0)
        )
        touching = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]}, geometry=[box(10.0, 2.0, 12.0, 8.0)], crs="EPSG:4326"
            )
        )
        assert _features_outside_template(touching, template) is True, (
            "a feature touching the template edge (zero overlap) is treated as outside"
        )

    @pytest.mark.parametrize(
        "class_ids, geometry",
        [
            (pd.Series([], dtype="int32"), []),  # truly empty (len 0)
            ([7], [None]),  # rows with null geometry (len>0, NaN bounds)
        ],
        ids=["empty_collection", "null_geometry_rows"],
    )
    def test_from_features_warns_and_returns_all_nodata_without_geometry(
        self, class_ids, geometry
    ):
        """Empty or null-geometry collections warn and return an all-nodata raster (#46).

        Args:
            class_ids: Burn-column values — empty for the zero-row case, one value for the
                null-geometry row.
            geometry: The geometry column — empty, or a single ``None``.

        Test scenario:
            Both an empty FeatureCollection (len 0, which skips the burn) and rows with null
            geometry (len>0, NaN bounds, which burns nothing) warn and yield an all-nodata
            raster on the template grid, without the cryptic GDAL "field not found" crash.
        """
        template = _make_dataset(
            rows=5, cols=5, epsg=4326, cell_size=1.0, top_left=(0.0, 5.0)
        )
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": class_ids}, geometry=geometry, crs="EPSG:4326"
            )
        )

        with pytest.warns(UserWarning, match="empty or falls entirely outside"):
            raster = Dataset.from_features(
                fc, template=template, column_name="class_id"
            )

        arr = raster.read_array()
        assert arr.shape == (5, 5), "output should adopt the template grid"
        assert not np.any(np.isclose(arr, 7.0)), "no geometry burns nothing"

    def test_interior_zero_area_point_is_not_flagged(self):
        """A zero-area point inside the template is not flagged as outside (#46).

        Test scenario:
            `Point(5, 5)` has bounds `[5, 5, 5, 5]` (not NaN) and lies inside a template
            covering x[0, 10], y[0, 10], so the helper must return `False`.
        """
        template = _make_dataset(
            rows=10, cols=10, epsg=4326, cell_size=1.0, top_left=(0.0, 10.0)
        )
        point = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[Point(5.0, 5.0)], crs="EPSG:4326")
        )
        assert _features_outside_template(point, template) is False, (
            "an interior zero-area point must not be flagged outside"
        )

    def test_cell_size_mode_does_not_warn(self):
        """cell_size mode (no template) never emits the outside-template warning (#46)."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]}, geometry=[box(0.0, 0.0, 3.0, 3.0)], crs="EPSG:4326"
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Dataset.from_features(fc, cell_size=1.0, column_name="v")

        template_warnings = [w for w in caught if "template extent" in str(w.message)]
        assert not template_warnings, (
            f"cell_size mode must not warn about a template; got: {template_warnings}"
        )

    def test_snap_to_template_crops_to_features_and_co_registers(self, utm_template):
        """snap_to_template crops to the features but keeps the template's grid (#46 point 2).

        Test scenario:
            A small polygon inside the 10x10 template yields a smaller raster whose origin
            lies on the template's grid lines and whose cell size equals the template's, so
            it co-registers pixel-for-pixel; the burned value is present.
        """
        x0, y0 = utm_template.top_left_corner
        cell = utm_template.cell_size
        inside = box(x0 + 2 * cell, y0 - 4 * cell, x0 + 5 * cell, y0 - 1 * cell)
        out = Dataset.from_features(
            self._fc_32636(inside),
            template=utm_template,
            snap_to_template=True,
            column_name="class_id",
        )

        assert (out.rows, out.columns) == (3, 3), (
            f"snap output should crop to the features, got {(out.rows, out.columns)}"
        )
        ox, oy = out.top_left_corner
        assert (ox - x0) % cell == 0, "snapped x-origin must lie on the template grid"
        assert (y0 - oy) % cell == 0, "snapped y-origin must lie on the template grid"
        assert out.cell_size == cell, "snapped output keeps the template cell size"
        assert np.any(np.isclose(out.read_array(), 42.0)), (
            "the polygon should be burned"
        )

    def test_snap_to_template_covers_features_outside_footprint(self, utm_template):
        """snap_to_template covers features beyond the template footprint, no warning (#46)."""
        x0, y0 = utm_template.top_left_corner
        cell = utm_template.cell_size
        far = box(x0 + 100 * cell, y0 - 3 * cell, x0 + 103 * cell, y0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = Dataset.from_features(
                self._fc_32636(far),
                template=utm_template,
                snap_to_template=True,
                column_name="class_id",
            )

        assert np.any(np.isclose(out.read_array(), 42.0)), (
            "features beyond the footprint are still covered in snap mode"
        )
        assert not [w for w in caught if "template extent" in str(w.message)], (
            "snap mode sizes to the features, so it must not warn"
        )
        ox, oy = out.top_left_corner
        assert (ox - x0) % cell == 0, "snapped x-origin still on the template grid"
        assert (y0 - oy) % cell == 0, "snapped y-origin still on the template grid"

    def test_snap_to_template_requires_template(self):
        """snap_to_template without a template raises ValueError (#46)."""
        fc = self._fc_32636(box(0.0, 0.0, 3.0, 3.0))
        with pytest.raises(
            ValueError, match="snap_to_template=True requires a template"
        ):
            Dataset.from_features(
                fc, cell_size=1.0, snap_to_template=True, column_name="class_id"
            )

    def test_snap_to_template_rejects_non_square_template(self):
        """snap_to_template with a non-square template raises ValueError (#46)."""
        template = self._mem_template((0.0, 2.0, 0.0, 10.0, 0.0, -1.0))
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": [42]}, geometry=[box(1.0, 6.0, 3.0, 8.0)], crs="EPSG:4326"
            )
        )
        with pytest.raises(ValueError, match="requires a square template"):
            Dataset.from_features(
                fc, template=template, snap_to_template=True, column_name="class_id"
            )

    def test_snap_to_template_rejects_rotated_template(self):
        """snap_to_template with a rotated template raises ValueError (#46)."""
        template = self._mem_template((0.0, 1.0, 0.5, 10.0, 0.5, -1.0))
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": [42]}, geometry=[box(1.0, 6.0, 3.0, 8.0)], crs="EPSG:4326"
            )
        )
        with pytest.raises(ValueError, match="rotated template"):
            Dataset.from_features(
                fc, template=template, snap_to_template=True, column_name="class_id"
            )

    def test_snap_to_template_rejects_empty_features(self, utm_template):
        """snap_to_template with an empty FeatureCollection raises ValueError (#46)."""
        empty = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": pd.Series([], dtype="int32")},
                geometry=[],
                crs="EPSG:32636",
            )
        )
        with pytest.raises(ValueError, match=r"valid \(non-NaN\) geometry bounds"):
            Dataset.from_features(
                empty,
                template=utm_template,
                snap_to_template=True,
                column_name="class_id",
            )

    def test_snap_to_template_degenerate_point_yields_covering_pixel(
        self, utm_template
    ):
        """A point exactly on a grid corner snaps to a 1x1 pixel that covers it (#46).

        Test scenario:
            A zero-area `Point` on a template grid corner collapses the snapped span to
            zero cells; the `max(1, ...)` clamp keeps a single covering pixel.
        """
        x0, y0 = utm_template.top_left_corner
        cell = utm_template.cell_size
        corner = Point(x0 + 2 * cell, y0 - 3 * cell)
        out = Dataset.from_features(
            self._fc_32636(corner),
            template=utm_template,
            snap_to_template=True,
            column_name="class_id",
        )
        assert out.rows == 1, f"degenerate point should give one row, got {out.rows}"
        assert out.columns == 1, (
            f"degenerate point should give one column, got {out.columns}"
        )
        assert np.any(np.isclose(out.read_array(), 42.0)), (
            "the point's pixel must be burned"
        )

    def test_snap_to_template_fractional_grid_is_tight(self):
        """Grid-aligned bounds on a fractional-coordinate template give tight dims (#46).

        Test scenario:
            A WGS84 template with 0.001 cells and a feature whose bounds are exactly 37→60
            cells in X and 12→40 in Y must snap to exactly 23x28 — no float off-by-one.
        """
        template = self._mem_template((-123.456, 0.001, 0.0, 45.678, 0.0, -0.001))
        xmin = -123.456 + 37 * 0.001
        xmax = -123.456 + 60 * 0.001
        ymax = 45.678 - 12 * 0.001
        ymin = 45.678 - 40 * 0.001
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": [42]},
                geometry=[box(xmin, ymin, xmax, ymax)],
                crs="EPSG:4326",
            )
        )
        out = Dataset.from_features(
            fc, template=template, snap_to_template=True, column_name="class_id"
        )
        assert (out.rows, out.columns) == (28, 23), (
            f"snap should be tight (28, 23), got {(out.rows, out.columns)}"
        )

    def test_snap_to_template_supports_south_up_template(self):
        """A south-up template (positive Y pixel) snaps onto the same lattice (#46).

        Test scenario:
            north-up is not required — a south-up template still covers the feature in a
            correct north-up output.
        """
        template = self._mem_template((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"class_id": [42]}, geometry=[box(1.0, 1.0, 4.0, 4.0)], crs="EPSG:4326"
            )
        )
        out = Dataset.from_features(
            fc, template=template, snap_to_template=True, column_name="class_id"
        )
        assert np.any(np.isclose(out.read_array(), 42.0)), (
            "a south-up template should still cover the feature"
        )

    def test_from_features_rejects_non_positive_cell_size(self):
        """D-M2: cell_size=0 and negative values raise ``ValueError``."""
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="cell_size must be positive"):
            Dataset.from_features(fc, cell_size=0, column_name="v")
        with pytest.raises(ValueError, match="cell_size must be positive"):
            Dataset.from_features(fc, cell_size=-10.0, column_name="v")

    def test_from_features_rejects_empty_column_list(self):
        """D-M2: empty ``column_name`` list raises ``ValueError``."""
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="non-empty"):
            Dataset.from_features(fc, cell_size=0.1, column_name=[])

    def test_from_features_rejects_unknown_column_string(self):
        """D-M2: unknown ``column_name`` string raises with the valid list."""
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="not in the FeatureCollection"):
            Dataset.from_features(fc, cell_size=0.1, column_name="nope")

    def test_from_features_rejects_non_str_non_list_column_name(self):
        """M4: ``column_name`` must be str, list, or None — typed TypeError.

        Test scenario:
            Passing an int for ``column_name`` previously surfaced as a
            misleading ValueError about the column not being in the
            FeatureCollection. The typed TypeError now points at the
            real issue (wrong input type) and names the allowed set.
        """
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(TypeError, match=r"str, list\[str\], or None"):
            Dataset.from_features(fc, cell_size=0.1, column_name=123)

    def test_from_features_rejects_unknown_column_in_list(self):
        """D-M2: unknown name inside a ``column_name`` list also raises."""
        gdf = gpd.GeoDataFrame(
            {"a": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match=r"not in the FeatureCollection.*'b'"):
            Dataset.from_features(fc, cell_size=0.1, column_name=["a", "b"])

    def test_from_features_negative_cell_size_with_template(self):
        """D-M2 boundary: negative cell_size is rejected even when template given.

        Test scenario:
            The guard fires on the cell_size kwarg unconditionally (it
            does not wait until the non-template branch dereferences
            ``cell_size``). A template-path caller who passes a
            negative cell_size gets the same error up front.
        """
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)
        template = Dataset.create(
            cell_size=cell_size,
            rows=5,
            columns=5,
            dtype="int32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=-1,
        )
        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0.0, 0.0, 1.0, 1.0)],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)
        with pytest.raises(ValueError, match="cell_size must be positive"):
            Dataset.from_features(
                fc,
                cell_size=-1.0,
                template=template,
                column_name="v",
            )

    def test_from_features_raises_on_crs_less_features(self):
        """C5: CRS-less FeatureCollection fails fast with CRSError.

        Regression for the pr-review-merged C5 finding: rasterising a
        FeatureCollection whose ``crs`` is ``None`` previously produced
        a raster with an undefined projection, failing downstream with
        cryptic GDAL errors. Now the method raises a typed
        :class:`CRSError` at the top of ``from_features``.
        """
        from pyramids.base._errors import CRSError

        gdf = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0.0, 0.0, 1.0, 1.0)],
            # Explicitly no CRS.
        )
        fc = FeatureCollection(gdf)
        assert fc.epsg is None

        with pytest.raises(CRSError, match="must have a CRS"):
            Dataset.from_features(fc, cell_size=0.1, column_name="v")

    def test_rasterize_integer_dtype_with_none_nodata_template(self):
        """C2: integer burn with a template having no-data=None falls back
        to the class default sentinel instead of NaN.

        Regression for the pr-review-merged C2 finding: when the template's
        no-data is None and the burn column is integer-typed, the previous
        code assigned ``np.nan`` to the output raster's no-data — invalid
        on integer rasters and silently coerced into an arbitrary sentinel.
        """
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        template = Dataset.create(
            cell_size=cell_size,
            rows=10,
            columns=10,
            dtype="int32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=None,
        )
        # Precondition: template carries None as its no-data.
        assert template.no_data_value[0] is None

        x0, y0 = top_left
        poly = box(x0, y0 - 5 * cell_size, x0 + 5 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"class_id": np.array([7], dtype=np.int32)},
            geometry=[poly],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)

        raster = Dataset.from_features(fc, template=template, column_name="class_id")

        nodata = raster.no_data_value[0]
        assert nodata is not None, "integer raster must have a non-None no-data"
        assert not (isinstance(nodata, float) and np.isnan(nodata)), (
            "integer raster's no-data must not be NaN (C2)"
        )
        assert nodata == Dataset.default_no_data_value

    @pytest.mark.parametrize(
        "int_dtype,sample",
        [
            ("int16", -42),
            ("int64", 2_000_000_000),
        ],
        ids=["int16", "int64"],
    )
    def test_rasterize_integer_dtype_variants(self, int_dtype, sample):
        """C2 parametrized: signed integer dtypes trigger the fallback.

        Test scenario:
            Build a template with ``no_data_value=None`` and a burn
            column of the given signed integer dtype. The rasterizer
            picks ``cls.default_no_data_value`` (``-9999``) instead of
            silently coercing NaN into an arbitrary integer. Unsigned
            integer dtypes (``uint8``/``uint16``) are excluded: the
            class default ``-9999`` cannot be stored in an unsigned
            type at all — that is a separate, pre-existing defect in
            ``band_metadata`` orthogonal to C2.
        """
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        template = Dataset.create(
            cell_size=cell_size,
            rows=5,
            columns=5,
            dtype=int_dtype,
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=None,
        )

        x0, y0 = top_left
        poly = box(x0, y0 - 3 * cell_size, x0 + 3 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"v": np.array([sample], dtype=int_dtype)},
            geometry=[poly],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)

        raster = Dataset.from_features(fc, template=template, column_name="v")
        nodata = raster.no_data_value[0]
        assert nodata is not None, f"{int_dtype}: no-data is None"
        assert not (isinstance(nodata, float) and np.isnan(nodata)), (
            f"{int_dtype}: no-data is NaN — C2 regression"
        )

    def test_rasterize_float_dtype_keeps_nan_nodata(self):
        """C2 negative: float dtype templates keep the NaN fallback.

        Test scenario:
            The C2 guard only kicks in for integer dtypes. A float32
            burn column with ``template.no_data_value=None`` must still
            carry NaN as its no-data (since float32 can represent it).
        """
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        template = Dataset.create(
            cell_size=cell_size,
            rows=5,
            columns=5,
            dtype="float32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=None,
        )
        x0, y0 = top_left
        poly = box(x0, y0 - 3 * cell_size, x0 + 3 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"x": np.array([3.14], dtype=np.float32)},
            geometry=[poly],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)

        raster = Dataset.from_features(fc, template=template, column_name="x")
        nodata = raster.no_data_value[0]
        # NaN on float is valid and preserved.
        assert nodata is not None, "float raster should still have a no-data"
        assert isinstance(nodata, float) and np.isnan(nodata), (
            f"float raster should keep NaN no-data; got {nodata!r}"
        )

    def test_rasterize_integer_dtype_keeps_explicit_template_nodata(self):
        """C2 negative: an explicit integer no-data on the template is preserved.

        Test scenario:
            When the template already carries a concrete integer no-data
            (e.g. ``-1``), the C2 guard must not overwrite it with the
            class default. Only the NaN → default fallback path fires.
        """
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        template = Dataset.create(
            cell_size=cell_size,
            rows=5,
            columns=5,
            dtype="int32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=-1,
        )
        assert template.no_data_value[0] == -1

        x0, y0 = top_left
        poly = box(x0, y0 - 3 * cell_size, x0 + 3 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"v": np.array([7], dtype=np.int32)},
            geometry=[poly],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)

        raster = Dataset.from_features(fc, template=template, column_name="v")
        assert raster.no_data_value[0] == -1, (
            "explicit template no-data must not be overwritten"
        )

    def test_rasterize_then_pickle_roundtrip_chain(self):
        """C2 + C3 chained: rasterize → pickle FC → unpickle → rasterize again.

        Test scenario:
            Exercise C2's integer-dtype guard and C3's ``_metadata``
            dedup together. Build an integer-typed FC, pickle/unpickle
            it, verify the CRS/epsg cache and geometry column survive,
            then rasterize through a None-nodata template and confirm
            both runs produce the same no-data sentinel.
        """
        import pickle

        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        x0, y0 = top_left
        poly = box(x0, y0 - 3 * cell_size, x0 + 3 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"class_id": np.array([9], dtype=np.int32)},
            geometry=[poly],
            crs=f"EPSG:{epsg}",
        )
        fc = FeatureCollection(gdf)

        restored = pickle.loads(pickle.dumps(fc))
        assert isinstance(restored, FeatureCollection)
        assert restored.epsg == epsg
        assert "geometry" in restored.columns

        template = Dataset.create(
            cell_size=cell_size,
            rows=5,
            columns=5,
            dtype="int32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=None,
        )
        r1 = Dataset.from_features(fc, template=template, column_name="class_id")
        r2 = Dataset.from_features(restored, template=template, column_name="class_id")
        assert r1.no_data_value[0] == r2.no_data_value[0]
        assert r1.no_data_value[0] == Dataset.default_no_data_value


class TestReprojectAlignWorkflow:
    """Reproject a raster and then align another to its grid."""

    def test_reproject_and_verify(self):
        """Reproject a UTM raster to WGS84 and verify the EPSG changes."""
        src = _make_dataset(
            rows=10, cols=10, epsg=32636, cell_size=1000.0, fill_value=5.0
        )
        _ = src.read_array()
        original_epsg = src.epsg
        assert original_epsg == 32636, "Starting EPSG should be 32636"

        reprojected = src.to_crs(to_epsg=4326)
        assert reprojected.epsg == 4326, (
            f"Reprojected EPSG should be 4326, got {reprojected.epsg}"
        )
        repr_arr = reprojected.read_array()
        assert repr_arr.shape[0] > 0, "Reprojected raster should have rows"
        assert repr_arr.shape[1] > 0, "Reprojected raster should have cols"

    def test_align_to_reference(self):
        """Align one raster to match another's grid."""
        # Reference raster (smaller)
        ref = _make_dataset(
            rows=5,
            cols=5,
            epsg=32636,
            cell_size=2000.0,
            top_left=(500000.0, 3400000.0),
            fill_value=0.0,
        )

        # Source raster (different grid)
        src = _make_dataset(
            rows=10,
            cols=10,
            epsg=32636,
            cell_size=1000.0,
            top_left=(500000.0, 3400000.0),
            fill_value=7.0,
        )

        aligned = src.align(ref)
        assert aligned.rows == ref.rows, (
            f"Aligned rows should be {ref.rows}, got {aligned.rows}"
        )
        assert aligned.columns == ref.columns, (
            f"Aligned columns should be {ref.columns}, got {aligned.columns}"
        )


class TestDatasetCollectionProcessingPipeline:
    """Create a DatasetCollection, apply a function, then iterate and verify."""

    def test_apply_then_iterate(self):
        """Apply a transformation and iterate to check every time step."""
        rows, cols = 6, 8
        time_steps = 4

        base = _make_dataset(rows=rows, cols=cols, fill_value=10.0)
        md = DatasetCollection.from_dataset(base, time_length=time_steps)

        # Fill with known values: each time step has value = step_index + 1
        values = np.zeros((time_steps, rows, cols), dtype=np.float64)
        for t in range(time_steps):
            values[t, :, :] = float(t + 1)
        md.values = values

        # Apply np.sqrt — out-of-place after the L-3 refactor.
        md = md.apply(np.sqrt)

        # Verify each time step via iteration
        for i, slice_arr in enumerate(md):
            expected_val = np.sqrt(float(i + 1))
            non_nodata = slice_arr[~np.isclose(slice_arr, -9999.0, rtol=0.001)]
            if non_nodata.size > 0:
                assert np.allclose(non_nodata, expected_val, atol=0.01), (
                    f"Time step {i}: expected ~{expected_val}, got {non_nodata[0]}"
                )

    def test_head_tail_first_last(self):
        """Verify head/tail/first/last return correct shapes."""
        rows, cols = 4, 5
        time_steps = 6

        base = _make_dataset(rows=rows, cols=cols)
        md = DatasetCollection.from_dataset(base, time_length=time_steps)
        values = np.random.default_rng(0).random((time_steps, rows, cols))
        md.values = values

        assert md.head(3).shape == (3, rows, cols), "head(3) shape mismatch"
        assert md.tail(-2).shape == (2, rows, cols), "tail(-2) shape mismatch"
        assert md.first().shape == (rows, cols), "first() shape mismatch"
        assert md.last().shape == (rows, cols), "last() shape mismatch"

        # Verify first/last content
        np.testing.assert_array_equal(
            md.first(), values[0], err_msg="first() content mismatch"
        )
        np.testing.assert_array_equal(
            md.last(), values[-1], err_msg="last() content mismatch"
        )


class TestFeatureCollectionPropertiesE2E:
    """End-to-end property checks for FeatureCollection."""

    def test_subclass_identity_preserves_data(self):
        """After ARC-1a FeatureCollection IS a GeoDataFrame — check round-trip.

        Verifies that wrapping a GeoDataFrame in FeatureCollection and
        constructing a plain GeoDataFrame back from it preserves EPSG,
        geometry, and attributes without any OGR-side conversion.
        """
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame({"val": [1]}, geometry=[poly], crs="EPSG:4326")
        fc = FeatureCollection(gdf)
        assert isinstance(fc, gpd.GeoDataFrame)
        assert fc.epsg == 4326

        round_trip = gpd.GeoDataFrame(fc)
        assert round_trip.crs.to_epsg() == 4326
        assert len(round_trip) == 1
        assert round_trip["val"].iloc[0] == 1

    def test_save_and_reload_vector(self):
        """Save a FeatureCollection to disk and read it back."""
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame({"score": [99.5]}, geometry=[poly], crs="EPSG:4326")
        fc = FeatureCollection(gdf)

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "test_output.geojson"
        try:
            fc.to_file(path)
            assert path.exists(), "File should exist after to_file"
            reloaded = FeatureCollection.read_file(path)
            # FeatureCollection IS a GeoDataFrame, no `.feature` indirection.
            assert isinstance(reloaded, gpd.GeoDataFrame)
            assert len(reloaded) == 1, "Reloaded GDF should have 1 row"
            assert abs(reloaded["score"].iloc[0] - 99.5) < 0.01, (
                "Reloaded score value should be ~99.5"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestClusterE2E:
    """End-to-end workflows combining cluster with other Dataset operations."""

    def test_create_cluster_save_reload(self):
        """Create dataset -> cluster -> write cluster array to file -> reload and verify.

        Test scenario:
            Full round-trip: create a raster with known values, cluster it,
            save the cluster array as a new GeoTIFF, reload it, and verify
            the cluster labels survive the disk round-trip.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 8.0, 8.0],
                [0.0, 0.0, 0.0, 8.0, 8.0],
            ],
            dtype=np.float32,
        )
        src = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
        )
        cluster_array, count, position, _ = src.cluster(1, 10)

        assert count == 3, f"Expected 2 clusters, got {count - 1}"
        assert len(position) == 8, f"Expected 8 cells clustered, got {len(position)}"

        result = Dataset.create_from_array(
            cluster_array.astype(np.float32),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        )

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "cluster_result.tif"
        try:
            result.to_file(path)
            assert path.exists(), "Cluster GeoTIFF should be written"

            reloaded = Dataset.read_file(path)
            reloaded_arr = reloaded.read_array()
            np.testing.assert_array_equal(
                reloaded_arr,
                cluster_array,
                err_msg="Cluster array should survive disk round-trip",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_crop_then_cluster(self):
        """Create a large raster -> crop to subset -> cluster the cropped region.

        Test scenario:
            Verify that cropping a dataset and then clustering the result
            produces correct clusters based on the cropped data, not the
            original extent.
        """
        arr = np.zeros((20, 20), dtype=np.float32)
        arr[2:5, 2:5] = 7.0
        arr[15:18, 15:18] = 7.0
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
        )

        crop_poly = box(-0.5, -5.5, 6.5, 0.5)
        crop_mask = gpd.GeoDataFrame(geometry=[crop_poly], crs="EPSG:4326")
        cropped = src.crop(crop_mask)

        _, count, _, values = cropped.cluster(5, 10)

        assert count == 2, f"Expected 1 cluster in cropped region, got {count - 1}"
        for v in values:
            assert 5 <= v <= 10, f"Clustered value {v} outside bounds [5, 10]"

    def test_cluster_reproject_preserves_count(self):
        """Create dataset -> cluster -> reproject -> re-cluster -> compare counts.

        Test scenario:
            Create a dataset in EPSG:4326, cluster it, reproject to
            EPSG:32636 (UTM), re-cluster, and verify a similar number of
            clusters exist (exact match not expected due to resampling).
        """
        rng = np.random.default_rng(77)
        arr = rng.choice([0.0, 5.0], size=(10, 10), p=[0.6, 0.4]).astype(np.float32)
        src = Dataset.create_from_array(
            arr, top_left_corner=(30.0, 31.0), cell_size=0.01, epsg=4326
        )

        _, count_orig, _, _ = src.cluster(4, 6)

        reprojected = src.to_crs(to_epsg=32636)
        _, count_reproj, _, _ = reprojected.cluster(4, 6)

        assert count_reproj >= 1, "Reprojected dataset should have at least 1 cluster"
        assert abs(count_reproj - count_orig) <= count_orig, (
            f"Cluster count after reproject ({count_reproj}) diverged too far "
            f"from original ({count_orig})"
        )

    def test_cluster_to_vector_polygons(self):
        """Create dataset -> cluster -> convert clusters to vector polygons.

        Test scenario:
            Create a dataset, cluster it, then use to_polygons (GDAL
            Polygonize) on the cluster array to produce vector polygons.
            Verify the polygons are valid and cover the clustered region.
        """
        arr = np.array(
            [
                [5.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        src = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
        )

        cluster_array, _, _, _ = src.cluster(4, 6)

        cluster_ds = Dataset.create_from_array(
            cluster_array.astype(np.float32),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
        )
        gdf = cluster_ds.to_polygons()

        assert isinstance(gdf, gpd.GeoDataFrame), (
            f"Expected GeoDataFrame, got {type(gdf)}"
        )
        assert len(gdf) > 0, "Should produce at least one polygon"
        assert all(geom.is_valid for geom in gdf.geometry), (
            "All polygons should be valid geometries"
        )

    def test_large_cluster_no_recursion_e2e(self):
        """Create a 300x300 raster -> cluster all cells -> verify no crash.

        Test scenario:
            End-to-end verification that the iterative BFS handles a
            90,000-cell connected region through the full Dataset.cluster
            pipeline without hitting recursion limits.
        """
        arr = np.ones((300, 300), dtype=np.float32) * 5
        src = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.01, epsg=4326
        )

        cluster_array, count, position, _ = src.cluster(1, 10)

        assert count == 2, f"Expected 1 cluster, got {count - 1}"
        assert len(position) == 90000, f"Expected 90000 cells, got {len(position)}"
        assert np.all(cluster_array == 1), "All cells should be cluster 1"


class TestApplyE2E:
    """End-to-end workflows combining apply with other Dataset operations."""

    def test_apply_save_reload(self):
        """Apply a function -> save to GeoTIFF -> reload -> verify values survive round-trip.

        Test scenario:
            Create a dataset, apply np.square, save the result to disk,
            reload it, and verify the squared values are preserved.
        """
        arr = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = src.apply(np.square)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "apply_result.tif"
        try:
            result.to_file(path)
            assert path.exists(), "GeoTIFF should be written"

            reloaded = Dataset.read_file(path)
            np.testing.assert_array_almost_equal(
                reloaded.read_array(),
                expected,
                decimal=2,
                err_msg="Squared values should survive disk round-trip",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_apply_then_crop(self):
        """Apply a function -> crop the result with a polygon -> verify cropped values.

        Test scenario:
            Create a 10x10 dataset, apply doubling, crop to a 5x5 sub-region,
            and verify the cropped values are doubled.
        """
        arr = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        doubled = src.apply(lambda x: x * 2)

        crop_poly = box(0.5, -4.5, 4.5, -0.5)
        crop_mask = gpd.GeoDataFrame(geometry=[crop_poly], crs="EPSG:4326")
        cropped = doubled.crop(crop_mask)

        cropped_arr = cropped.read_array()
        nodata = cropped.no_data_value[0]
        domain_vals = cropped_arr[~np.isclose(cropped_arr, nodata, rtol=0.001)]
        assert len(domain_vals) > 0, "Cropped result should have domain cells"
        assert np.all(domain_vals % 2 == 0), (
            "All cropped domain values should be even (doubled from integers)"
        )

    def test_apply_chained(self):
        """Chain multiple apply calls -> verify cumulative transformation.

        Test scenario:
            Create a dataset with value 2, apply x+3 -> then apply x*10.
            The result should be (2+3)*10 = 50.
        """
        arr = np.full((3, 3), 2.0, dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        step1 = src.apply(lambda x: x + 3)
        step2 = step1.apply(lambda x: x * 10)
        result_arr = step2.read_array()
        assert np.allclose(result_arr, 50.0), (
            f"Expected all cells to be 50.0 after chaining, got {result_arr}"
        )

    def test_apply_scalar_function_e2e(self):
        """Apply a scalar if/elif classification function end-to-end.

        Test scenario:
            Create a dataset with values spanning multiple classification
            bins, apply a scalar function that uses if/elif, and verify
            each cell gets the correct class.
        """

        def classify(val):
            if val < 5:
                return 1.0
            elif val < 15:
                return 2.0
            else:
                return 3.0

        arr = np.array(
            [[1.0, 5.0, 20.0], [3.0, 10.0, 25.0], [4.0, 14.0, 30.0]],
            dtype=np.float32,
        )
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = src.apply(classify)
        result_arr = result.read_array()
        expected = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            result_arr,
            expected,
            err_msg="Scalar classify should produce correct classification",
        )

    def test_apply_with_nodata_save_reload(self):
        """Apply a function to a dataset with no-data cells -> save -> reload -> verify.

        Test scenario:
            No-data cells should remain as no-data through the apply,
            disk save, and reload pipeline.
        """
        arr = np.array([[10.0, -9999.0], [-9999.0, 20.0]], dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = src.apply(lambda x: x + 5)

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "apply_nodata.tif"
        try:
            result.to_file(path)
            reloaded = Dataset.read_file(path)
            reloaded_arr = reloaded.read_array()
            assert np.isclose(reloaded_arr[0, 0], 15.0), (
                f"Domain cell should be 15.0, got {reloaded_arr[0, 0]}"
            )
            assert np.isclose(reloaded_arr[1, 1], 25.0), (
                f"Domain cell should be 25.0, got {reloaded_arr[1, 1]}"
            )
            nodata = reloaded.no_data_value[0]
            assert np.isclose(reloaded_arr[0, 1], nodata, rtol=0.001), (
                f"No-data cell should remain {nodata}, got {reloaded_arr[0, 1]}"
            )
            assert np.isclose(reloaded_arr[1, 0], nodata, rtol=0.001), (
                f"No-data cell should remain {nodata}, got {reloaded_arr[1, 0]}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_apply_inplace_then_save(self):
        """Apply inplace -> save the modified dataset -> reload -> verify.

        Test scenario:
            Using inplace=True should modify the original dataset, and
            saving + reloading should reflect those modifications.
        """
        arr = np.array([[3.0, 6.0], [9.0, 12.0]], dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        src.apply(lambda x: x / 3, inplace=True)

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "apply_inplace.tif"
        try:
            src.to_file(path)
            reloaded = Dataset.read_file(path)
            expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.testing.assert_array_almost_equal(
                reloaded.read_array(),
                expected,
                decimal=2,
                err_msg="Inplace apply should be reflected after save/reload",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestToFeatureCollectionE2E:
    """End-to-end workflows combining to_feature_collection with other operations."""

    def test_to_feature_collection_save_geojson_reload(self):
        """Create dataset -> to_feature_collection with geometry -> save GeoJSON -> reload.

        Test scenario:
            Convert a dataset to a GeoDataFrame with point geometry, save
            it as GeoJSON, reload it, and verify the values and geometry
            survive the round-trip.
        """
        arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        gdf = src.to_feature_collection(add_geometry="point")

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "test_fc.geojson"
        try:
            gdf.to_file(path, driver="GeoJSON")
            assert path.exists(), "GeoJSON file should exist"

            reloaded = gpd.read_file(path)
            assert len(reloaded) == 4, f"Expected 4 rows, got {len(reloaded)}"
            assert "geometry" in reloaded.columns, "Should have geometry column"
            assert all(g.geom_type == "Point" for g in reloaded.geometry), (
                "All geometries should be Points after reload"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_crop_then_to_feature_collection(self):
        """Create dataset -> crop -> to_feature_collection -> verify subset.

        Test scenario:
            Crop a 10x10 dataset to a 3x3 region, then convert to
            DataFrame. The result should have fewer rows than the full
            dataset.
        """
        arr = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        poly = box(1.5, -3.5, 4.5, -0.5)
        mask = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        cropped = src.crop(mask)
        df = cropped.to_feature_collection()

        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        assert len(df) < 100, (
            f"Cropped result should have fewer than 100 rows, got {len(df)}"
        )
        assert len(df) > 0, "Should have some domain cells"

    def test_apply_then_to_feature_collection(self):
        """Create dataset -> apply function -> to_feature_collection -> verify transformed values.

        Test scenario:
            Apply x*10 to a dataset, then convert to DataFrame. All
            values in the DataFrame should be multiples of 10.
        """
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        transformed = src.apply(lambda x: x * 10)
        df = transformed.to_feature_collection()

        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"
        assert all(v % 10 == 0 for v in df.iloc[:, 0]), (
            "All values should be multiples of 10"
        )

    def test_multiband_to_feature_collection_polygon_geometry(self):
        """Create multi-band dataset -> to_feature_collection with polygon -> verify.

        Test scenario:
            A 2-band dataset converted with polygon geometry should
            produce a GeoDataFrame with 2 value columns plus geometry.
        """
        arr = np.random.default_rng(42).random((2, 4, 4)).astype(np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        gdf = src.to_feature_collection(add_geometry="polygon")

        assert isinstance(gdf, gpd.GeoDataFrame), (
            f"Expected GeoDataFrame, got {type(gdf)}"
        )
        value_cols = [c for c in gdf.columns if c != "geometry"]
        assert len(value_cols) == 2, f"Expected 2 value columns, got {len(value_cols)}"
        assert all(g.geom_type == "Polygon" for g in gdf.geometry), (
            "All geometries should be Polygons"
        )
        assert len(gdf) == 16, f"Expected 16 rows (4x4), got {len(gdf)}"


class TestContextManagerE2E:
    """End-to-end workflows using the context manager protocol."""

    def test_context_manager_save_and_reload(self):
        """Create dataset -> use in with block -> save -> reload outside block.

        Test scenario:
            Create a dataset, save it to disk inside a with block, then
            reload it outside the block after the original is closed.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "ctx_test.tif"
        try:
            ds = Dataset.create_from_array(
                arr,
                top_left_corner=(0.0, 0.0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=-9999.0,
            )
            with ds:
                ds.to_file(path)
            assert ds._raster is None, "Dataset should be closed after with block"

            reloaded = Dataset.read_file(path)
            np.testing.assert_array_almost_equal(
                reloaded.read_array(),
                arr,
                decimal=2,
                err_msg="Reloaded values should match original",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_context_manager_apply_then_save(self):
        """Create dataset -> apply inside with block -> save -> verify.

        Test scenario:
            Apply a transformation inside a with block, save the result,
            and verify the pipeline works end-to-end with cleanup.
        """
        arr = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "ctx_apply.tif"
        try:
            ds = Dataset.create_from_array(
                arr,
                top_left_corner=(0.0, 0.0),
                cell_size=1.0,
                epsg=4326,
                no_data_value=-9999.0,
            )
            with ds:
                result = ds.apply(lambda x: x * 2)
                result.to_file(path)

            reloaded = Dataset.read_file(path)
            expected = np.array([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32)
            np.testing.assert_array_almost_equal(
                reloaded.read_array(),
                expected,
                decimal=2,
                err_msg="Applied values should be doubled",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_context_manager_exception_no_file_leak(self):
        """Exception inside with block should not leave file locks.

        Test scenario:
            Create a dataset, raise inside the with block, then verify
            we can still write to the same path (no lingering file lock).
        """
        arr = np.ones((3, 3), dtype=np.float32) * 42
        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "ctx_exception.tif"
        try:
            ds = Dataset.create_from_array(
                arr,
                top_left_corner=(0.0, 0.0),
                cell_size=1.0,
                epsg=4326,
            )
            with pytest.raises(ValueError):
                with ds:
                    raise ValueError("intentional error")

            ds2 = Dataset.create_from_array(
                arr,
                top_left_corner=(0.0, 0.0),
                cell_size=1.0,
                epsg=4326,
            )
            ds2.to_file(path)
            assert path.exists(), "Should be able to write after exception cleanup"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestGeoTiffRoundTrip:
    """Write an in-memory Dataset to GeoTIFF, reload, verify."""

    def test_write_read_geotiff(self):
        """Create an in-memory raster, save to disk, reload and verify array."""
        rows, cols = 12, 15
        src = _make_dataset(
            rows=rows,
            cols=cols,
            fill_value=42.0,
            epsg=4326,
            cell_size=0.1,
            top_left=(10.0, 50.0),
        )
        arr_original = src.read_array()

        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / "test_raster.tif"
        try:
            src.to_file(path)
            assert path.exists(), "GeoTIFF should be written"

            reloaded = Dataset.read_file(path)
            arr_reloaded = reloaded.read_array()
            assert arr_reloaded.shape == (
                rows,
                cols,
            ), f"Reloaded shape mismatch: {arr_reloaded.shape}"
            np.testing.assert_array_almost_equal(
                arr_reloaded,
                arr_original,
                decimal=2,
                err_msg="Reloaded array values differ from original",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
