"""Tests for COG partial reads: read_part / preview / point / read_tile.

Covers overview-decimated geographic reads, whole-image previews, single-point
sampling, Web-Mercator XYZ tiles, the `_xyz_bounds_3857` helper, CRS
reprojection of the request window, and out-of-bounds / invalid-arg handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OutOfBoundsError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines.cog import COG, _xyz_bounds_3857
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


@pytest.fixture
def ramp_4326() -> Dataset:
    """A 100x100 Float32 ramp on EPSG:4326 where value == row*100 + col.

    Returns:
        Dataset: An in-memory dataset with a deterministic ramp.
    """
    arr = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    return Dataset.from_array(
        arr, geo_ref=GeoReference(geo=COG_GEOTRANSFORM, epsg=4326)
    )


@pytest.fixture
def tile_3857(tmp_path) -> Dataset:
    """A 256x256 Float32 ramp on EPSG:3857 covering exactly XYZ tile (1, 0, 0).

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dataset: An in-memory dataset aligned to the zoom-1 NW tile bounds.
    """
    west, _, east, north = _xyz_bounds_3857(1, 0, 0)
    cell = (east - west) / 256.0
    gt = (west, cell, 0.0, north, 0.0, -cell)
    mem = gdal.GetDriverByName("MEM").Create("", 256, 256, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3857)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(
        np.arange(256 * 256, dtype="float32").reshape(256, 256)
    )
    mem.FlushCache()
    return Dataset(mem)


@pytest.fixture
def straddle_3857() -> Dataset:
    """A 3857 dataset centred on the origin, straddling the four zoom-1 tiles.

    Returns:
        Dataset: A 200x200 dataset covering roughly the central quarter of the
        Web-Mercator world, so the zoom-1 NW tile (1, 0, 0) only *partially*
        overlaps it.
    """
    r = 20037508.342789244
    a = r * 0.5
    cell = (2 * a) / 200.0
    gt = (-a, cell, 0.0, a, 0.0, -cell)
    mem = gdal.GetDriverByName("MEM").Create("", 200, 200, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3857)
    mem.SetProjection(sr.ExportToWkt())
    mem.GetRasterBand(1).WriteArray(np.ones((200, 200), dtype="float32"))
    mem.GetRasterBand(1).SetNoDataValue(-1.0)
    mem.FlushCache()
    return Dataset(mem)


class TestXyzBounds3857:
    """Tests for the _xyz_bounds_3857 helper."""

    def test_zoom0_world(self):
        """Zoom 0 tile (0,0) spans the whole Web-Mercator world.

        Test scenario:
            west/south are the negative half-extent; east/north the positive.
        """
        w, s, e, n = _xyz_bounds_3857(0, 0, 0)
        assert round(w) == -20037508 and round(s) == -20037508, (w, s)
        assert round(e) == 20037508 and round(n) == 20037508, (e, n)

    def test_zoom1_nw_quadrant(self):
        """Zoom 1 tile (0,0) is the north-west quadrant.

        Test scenario:
            east and south meet at the origin (0, 0).
        """
        w, s, e, n = _xyz_bounds_3857(1, 0, 0)
        assert round(e) == 0 and round(s) == 0, (e, s)
        assert round(w) == -20037508 and round(n) == 20037508, (w, n)


class TestReadPart:
    """Tests for COG.read_part."""

    def test_full_window_native_size(self, ramp_4326):
        """Reading the full bbox at native size returns the full array.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            The dataset bbox at no explicit output size yields (100, 100).
        """
        arr = ramp_4326.read_part(tuple(ramp_4326.bbox), bbox_crs=4326, band=0)
        assert arr.shape == (100, 100), f"unexpected shape {arr.shape}"

    def test_decimated_output_shape(self, ramp_4326):
        """An explicit dst size decimates the read to that shape.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            Requesting 25x25 over the full bbox returns a 25x25 array.
        """
        arr = ramp_4326.read_part(
            tuple(ramp_4326.bbox), dst_width=25, dst_height=25, bbox_crs=4326, band=0
        )
        assert arr.shape == (25, 25), f"unexpected shape {arr.shape}"

    def test_all_bands_shape(self, ramp_4326):
        """Reading all bands of a single-band raster returns a 2-D array.

        Args:
            ramp_4326: Fixture ramp Dataset (single band).

        Test scenario:
            band=None reads every band; GDAL's ReadAsArray collapses a
            single-band read to 2-D (rows, cols) — a multiband source would
            instead yield (bands, rows, cols).
        """
        arr = ramp_4326.read_part(tuple(ramp_4326.bbox), bbox_crs=4326)
        assert arr.shape == (100, 100), f"unexpected shape {arr.shape}"

    def test_invalid_resampling_raises(self, ramp_4326):
        """An unknown resampling name raises ValueError.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            resampling='bogus' is rejected before any read.
        """
        bbox = tuple(ramp_4326.bbox)
        with pytest.raises(ValueError, match="unknown resampling"):
            ramp_4326.read_part(bbox, resampling="bogus")

    def test_nearest_neighbor_alias_accepted(self, ramp_4326):
        """The 'nearest neighbor' alias resolves like the warp family (L1).

        Test scenario:
            The decimated-read registry accepts the historical
            ``"nearest neighbor"`` spelling (used by to_crs/resample), not only
            ``"nearest"``, so the resampling vocabulary is consistent across
            read paths.
        """
        via_alias = ramp_4326.read_part(
            tuple(ramp_4326.bbox), bbox_crs=4326, resampling="nearest neighbor"
        )
        via_short = ramp_4326.read_part(
            tuple(ramp_4326.bbox), bbox_crs=4326, resampling="nearest"
        )
        np.testing.assert_array_equal(
            via_alias, via_short, err_msg="'nearest neighbor' must equal 'nearest'"
        )

    def test_non_intersecting_bbox_raises(self, ramp_4326):
        """A bbox outside the raster extent raises OutOfBoundsError.

        Args:
            ramp_4326: Fixture ramp Dataset (lon 0..1, lat 9..10).

        Test scenario:
            A far-away bbox does not intersect the raster.
        """
        with pytest.raises(OutOfBoundsError):
            ramp_4326.read_part((50.0, 50.0, 51.0, 51.0), bbox_crs=4326)

    def test_partial_overlap_pads_with_nodata(self, ramp_4326):
        """A partially-overlapping window is padded, not stretched (M1).

        Args:
            ramp_4326: Fixture ramp Dataset (lon 0..1, lat 9..10, 100x100).

        Test scenario:
            The window (0.5, 9.5, 1.5, 10.5) overlaps only the raster's
            top-right quarter (lon 0.5..1, lat 9.5..10). The result keeps the
            requested 100x100 size; the overlapping sub-region carries real
            data while the out-of-raster remainder is filled with NoData — so
            the data stays aligned to the requested window instead of being
            stretched to fill it.
        """
        nd = ramp_4326._raster.GetRasterBand(1).GetNoDataValue()

        def is_fill(v):
            return bool(np.isnan(v)) if nd is None else v == nd

        arr = ramp_4326.read_part(
            (0.5, 9.5, 1.5, 10.5),
            dst_width=100,
            dst_height=100,
            bbox_crs=4326,
            band=0,
        )
        assert arr.shape == (100, 100), f"requested size must be kept: {arr.shape}"
        # Top half of the window is above the raster -> NoData.
        assert is_fill(arr[10, 10]), "above-raster region should be NoData-filled"
        # Right half of the window is east of the raster -> NoData.
        assert is_fill(arr[75, 90]), "east-of-raster region should be NoData-filled"
        # Bottom-left quadrant of the window overlaps the raster -> real data.
        assert not is_fill(arr[75, 10]), "overlapping region should hold real data"

    def test_partial_overlap_all_bands_shape(self, ramp_4326):
        """Partial overlap with band=None preserves the (rows, cols) shape.

        Args:
            ramp_4326: Fixture ramp Dataset (single band).

        Test scenario:
            A single-band partial read returns a 2-D padded array of the
            requested size.
        """
        arr = ramp_4326.read_part(
            (0.5, 9.5, 1.5, 10.5), dst_width=64, dst_height=64, bbox_crs=4326
        )
        assert arr.shape == (64, 64), f"unexpected shape {arr.shape}"

    def test_partial_overlap_multiband_shape(self):
        """Partial overlap on a multi-band raster keeps the (bands, H, W) shape.

        Test scenario:
            A 3-band ramp read with band=None over a window that overlaps only
            the top-right quarter returns a padded (3, 80, 80) array.
        """
        arr = np.arange(3 * 100 * 100, dtype="float32").reshape(3, 100, 100)
        ds = Dataset.from_array(
            arr, geo_ref=GeoReference(geo=COG_GEOTRANSFORM, epsg=4326)
        )
        out = ds.read_part(
            (0.5, 9.5, 1.5, 10.5), dst_width=80, dst_height=80, bbox_crs=4326
        )
        assert out.shape == (3, 80, 80), f"unexpected shape {out.shape}"

    def test_fully_inside_window_is_not_padded(self, ramp_4326):
        """A fully-inside window holds only real data — no NoData fill.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            A central sub-window lies entirely within the raster, so the result
            contains no NoData fill values.
        """
        nd = ramp_4326._raster.GetRasterBand(1).GetNoDataValue()
        arr = ramp_4326.read_part(
            (0.2, 9.2, 0.8, 9.8), dst_width=60, dst_height=60, bbox_crs=4326, band=0
        )
        assert arr.shape == (60, 60), f"unexpected shape {arr.shape}"
        if nd is not None:
            assert not np.any(arr == nd), "fully-inside window must not be padded"

    def test_partial_overlap_uses_explicit_nodata(self):
        """Padding uses the raster's explicit NoData value when set.

        Test scenario:
            A raster created with no_data_value=-1 pads the out-of-raster
            remainder of a partial window with -1.
        """
        arr = np.arange(100 * 100, dtype="float32").reshape(100, 100)
        ds = Dataset.from_array(
            arr,
            no_data_value=-1.0,
            geo_ref=GeoReference(geo=COG_GEOTRANSFORM, epsg=4326),
        )
        out = ds.read_part(
            (0.5, 9.5, 1.5, 10.5), dst_width=100, dst_height=100, bbox_crs=4326, band=0
        )
        assert out[10, 10] == -1.0, (
            f"expected NoData -1 in padded region, got {out[10, 10]}"
        )

    @pytest.mark.parametrize(
        "bad", [{"dst_width": 0}, {"dst_height": 0}, {"dst_width": -3}]
    )
    def test_a_non_positive_output_size_is_a_clear_error(self, ramp_4326, bad):
        """A zero or negative dst size is rejected with a message, not a ZeroDivisionError.

        Args:
            ramp_4326: Fixture ramp Dataset.
            bad: A single non-positive output dimension.

        Test scenario:
            The output geotransform divides by the output size, so a zero or
            negative `dst_*` used to surface as a bare `ZeroDivisionError` -- and,
            once the transform was computed unconditionally, on the plain-array
            path too. It must name the offending argument instead.
        """
        with pytest.raises(ValueError, match="must be a positive pixel count"):
            ramp_4326.read_part(tuple(ramp_4326.bbox), bbox_crs=4326, band=0, **bad)


class TestReadPartReturnTransform:
    """`read_part(return_transform=True)` reports the window it actually read."""

    @staticmethod
    def _col_index_grid() -> Dataset:
        """An 8x8 EPSG:3857 grid, top-left (0, 8), cell 1, value == column index.

        Returns:
            Dataset: A grid whose value at a cell equals the source column, so a
            returned value reveals the source-x it was sampled from.
        """
        cols = np.tile(np.arange(8, dtype="float64"), (8, 1))
        return Dataset.from_array(
            cols,
            geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=3857),
        )

    def test_default_still_returns_the_bare_array(self):
        """Omitting the flag keeps the original return type.

        Test scenario:
            The existing contract is a bare ndarray; a caller that never asks for
            the transform must not start receiving a tuple.
        """
        grid = self._col_index_grid()

        result = grid.read_part((1.5, 1.5, 4.5, 4.5), dst_width=3, dst_height=3, band=0)

        assert isinstance(result, np.ndarray), (
            f"expected a bare array, got {type(result)}"
        )

    def test_the_transform_places_the_cells_where_the_data_is(self):
        """The transform labels the snapped window, not the requested bbox.

        Test scenario:
            bbox (1.5, 1.5, 4.5, 4.5) snaps outward to source x-pixels [1, 5], read
            to width 3, so the cells sit at world-x centres 1.667 / 3.0 / 4.333 --
            not the 2.0 / 3.0 / 4.0 a caller labelling from the requested bbox would
            get. Because value == column index, each returned value names the
            source column its cell was sampled from, so `value + 0.5` is the
            world-x the transform must reproduce.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part(
            (1.5, 1.5, 4.5, 4.5),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )

        assert gt == pytest.approx((1.0, 4 / 3, 0.0, 5.0, 0.0, -4 / 3))
        transform_x = [gt[0] + (i + 0.5) * gt[1] for i in range(3)]
        sampled_x = [value + 0.5 for value in array[0].tolist()]
        assert transform_x == pytest.approx(sampled_x, abs=0.05), (
            f"transform {transform_x} must match the data at {sampled_x}"
        )

    def test_the_transform_round_trips_through_a_dataset(self):
        """Building a Dataset from `(array, transform)` reproduces the window.

        Test scenario:
            The whole point is to place the result without redoing the snap, so
            the transform must be a valid geotransform: a Dataset built from it
            carries the snapped origin and the decimated cell size.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part(
            (1.5, 1.5, 4.5, 4.5),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )
        placed = Dataset.from_array(array, geo_ref=GeoReference(geo=gt, epsg=3857))

        assert placed.top_left_corner == pytest.approx((1.0, 5.0))
        assert placed.cell_size == pytest.approx(4 / 3)

    def test_a_partial_overlap_transform_covers_the_padded_buffer(self):
        """The transform describes the full buffer, padding included.

        Test scenario:
            A window straddling the left edge is padded with NoData on the outside,
            and the returned buffer -- padding and all -- is aligned to the snapped
            window. So the transform's origin is the snapped left edge (x = -2, the
            floor of the requested -1.5), out to the raster and beyond, not the
            raster's own edge at x = 0.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part(
            (-1.5, 1.5, 1.5, 4.5),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )

        assert array.shape == (3, 3)
        assert gt[0] == pytest.approx(-2.0), f"origin must snap to -2, got {gt[0]}"
        assert gt[1] == pytest.approx(4 / 3), "four source pixels read to width three"

    @staticmethod
    def _worst_cell_offset(array, gt) -> float:
        """Largest gap between a data cell's source-x and its transform centre.

        Args:
            array: The returned buffer; value equals source column, so `value +
                0.5` is the source-x the cell sampled (exact under bilinear on the
                linear ramp).
            gt: The returned geotransform.

        Returns:
            float: The worst offset over the non-NoData cells of the middle row.
        """
        row = array if array.ndim == 1 else array[array.shape[0] // 2]
        worst = 0.0
        for column in range(array.shape[-1]):
            value = float(row[column])
            if np.isnan(value) or value < -1000:
                continue
            centre = gt[0] + (column + 0.5) * gt[1]
            worst = max(worst, abs(centre - (value + 0.5)))
        return worst

    def test_a_native_resolution_read_places_every_cell_exactly(self):
        """Native resolution is the only regime the docstring promises is cell-exact.

        Test scenario:
            A window off the left edge, read at native resolution (no `dst_*`),
            must land every data cell on its transform centre -- this is the
            cell-exact escape hatch the docstring points callers at.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part((-1.0, 5.0, 5.0, 9.0), band=0, return_transform=True)

        assert self._worst_cell_offset(array, gt) == pytest.approx(0.0, abs=1e-9)

    def test_a_decimated_straddle_drifts_within_one_cell_and_keeps_its_extent(self):
        """A decimated straddle stays under one output cell; the extent is exact.

        Test scenario:
            A window that both straddles the edge and is *decimated* pads the
            out-of-raster remainder to whole output cells before decimating, so
            the sampled cells shift within the buffer -- worst at the padded edge,
            but still under one output cell for the decimation regime. The outer
            extent matches the snapped window exactly. This is the milder of the
            two resampled-straddle regimes; the upsampling one drifts far more
            (see `test_an_upsampled_straddle_can_drift_several_cells`).
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part(
            (-1.0, 5.0, 5.0, 9.0),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )

        assert self._worst_cell_offset(array, gt) < abs(gt[1]), (
            "edge drift exceeds one cell"
        )
        left = gt[0]
        right = gt[0] + array.shape[-1] * gt[1]
        assert left == pytest.approx(-1.0), "left extent must match the snapped window"
        assert right == pytest.approx(5.0), "right extent must match the snapped window"

    def test_an_upsampled_straddle_can_drift_several_cells(self):
        """The transform's outer extent stays exact even where per-cell drift is large.

        Test scenario:
            Upsampling a window that overruns the raster edge is the regime an
            earlier draft's "at most about one output cell" bound got wrong: the
            padded remainder, stretched across many output cells, pushes data
            several cells off its transform centre. This pins the true contract --
            per-cell placement is *not* bounded to one cell here (it is measured
            well above one output cell), yet the buffer's outer extent still
            matches the snapped window exactly, which is the guarantee callers can
            rely on for any resampled read.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part(
            (-0.6, 1.5, 0.4, 6.5),
            dst_width=13,
            dst_height=13,
            band=0,
            return_transform=True,
        )

        assert self._worst_cell_offset(array, gt) > abs(gt[1]), (
            "an upsampled straddle should breach the one-cell bound the old wording claimed"
        )
        left = gt[0]
        right = gt[0] + array.shape[-1] * gt[1]
        assert left == pytest.approx(-1.0), "left extent must match the snapped window"
        assert right == pytest.approx(1.0), "right extent must match the snapped window"

    def test_a_south_up_source_keeps_its_pixel_step_sign(self):
        """A positive y-step source is not forced north-up.

        Test scenario:
            `_output_geotransform` composes the source affine rather than assuming
            a north-up grid, so a south-up source (positive `y_size`) comes back
            with a positive `y_size`, and a caller placing the cells does not flip
            the raster.
        """
        cols = np.tile(np.arange(8, dtype="float64"), (8, 1))
        south_up = Dataset.from_array(
            cols, geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0), epsg=3857)
        )

        _, gt = south_up.read_part(
            (1.5, 1.5, 4.5, 4.5),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )

        assert gt[5] == pytest.approx(4 / 3), f"y-step must stay positive, got {gt[5]}"

    def test_no_decimation_yields_the_source_cell_size(self):
        """Omitting the output size makes the transform carry the source cell size.

        Test scenario:
            With no `dst_width`/`dst_height` the buffer is the snapped source
            window at native resolution (`out_w`/`out_h` default to the source
            window size), so the transform's pixel step is the source cell size
            (1.0), not a decimated one, and its origin is the snapped whole-pixel
            corner. The existing tests always decimate 4 source pixels to width 3,
            so the identity `out_w == req_xsize` branch is otherwise unexercised.
        """
        grid = self._col_index_grid()

        array, gt = grid.read_part((1.5, 1.5, 4.5, 4.5), band=0, return_transform=True)

        assert array.shape == (4, 4), f"native-size snapped window, got {array.shape}"
        assert gt == pytest.approx((1.0, 1.0, 0.0, 5.0, 0.0, -1.0)), (
            f"native-resolution window must carry the source cell size, got {gt}"
        )

    def test_the_transform_is_in_the_dataset_crs_not_the_bbox_crs(self, tile_3857):
        """The transform is in the dataset CRS, whatever `bbox_crs` the caller uses.

        Args:
            tile_3857: Fixture EPSG:3857 dataset covering the zoom-1 NW quadrant.

        Test scenario:
            The window is asked for in EPSG:4326 degrees while the dataset is
            EPSG:3857 metres. The transform must describe the buffer in the
            dataset's own metric CRS -- every component metre-scale, never the
            degree bbox -- with the origin sitting strictly inside the dataset
            extent (so it tracks the reprojected window, not the dataset corner)
            and the north-up y-step kept negative.
        """
        west, south, east, north = _xyz_bounds_3857(1, 0, 0)

        _, gt = tile_3857.read_part(
            (-100.0, 20.0, -80.0, 40.0),
            dst_width=32,
            dst_height=32,
            bbox_crs=4326,
            band=0,
            return_transform=True,
        )

        assert gt[2] == 0.0 and gt[4] == 0.0, f"axis-aligned dataset, got {gt}"
        assert min(abs(gt[0]), abs(gt[1]), abs(gt[3]), abs(gt[5])) > 180, (
            f"transform must be metres in the dataset CRS, not degrees: {gt}"
        )
        assert west < gt[0] < east, (
            f"origin-x must track the window inside {west}..{east}: {gt[0]}"
        )
        assert south < gt[3] < north, (
            f"origin-y must track the window inside {south}..{north}: {gt[3]}"
        )
        assert gt[5] < 0, f"north-up dataset keeps a negative y-step, got {gt[5]}"

    def test_all_bands_read_returns_a_3d_array_with_the_same_transform(self):
        """`band=None` returns a 3-D buffer, and the transform is band-independent.

        Test scenario:
            The transform describes the window, not the band selection, so a
            three-band read with `band=None` returns a `(bands, rows, cols)` array
            whose transform is identical to the single-band read of the same
            window -- selecting bands never moves the cells. The existing return
            transform tests all pass `band=0`, so the all-bands arm pairing a 3-D
            array with the transform is otherwise untested.
        """
        cols = np.tile(np.arange(8, dtype="float64"), (8, 1))
        multi = np.stack([cols, cols + 10.0, cols + 20.0])
        dataset = Dataset.from_array(
            multi,
            geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=3857),
        )

        all_bands, gt_all = dataset.read_part(
            (1.5, 1.5, 4.5, 4.5), dst_width=3, dst_height=3, return_transform=True
        )
        _, gt_one = dataset.read_part(
            (1.5, 1.5, 4.5, 4.5),
            dst_width=3,
            dst_height=3,
            band=0,
            return_transform=True,
        )

        assert all_bands.shape == (3, 3, 3), (
            f"expected (bands, rows, cols), got {all_bands.shape}"
        )
        assert gt_all == pytest.approx(gt_one), (
            f"the transform must not depend on band selection: {gt_all} vs {gt_one}"
        )


class TestOutputGeotransform:
    """Tests for the COG._output_geotransform static helper."""

    def test_a_rotated_source_carries_its_skew_terms_through(self):
        """The affine composition holds for a rotated source (non-zero skew).

        Test scenario:
            A rotated source geotransform has non-zero `gt2`/`gt4` skew that the
            north-up and south-up cases never exercise. Composing the source
            affine with the window offset and the output scale must map every
            output pixel to the same world point the source grid maps its
            corresponding fractional source pixel to -- checked against GDAL's own
            `ApplyGeoTransform`. The window and output sizes are deliberately
            distinct (4x6 read to 2x2) so the x- and y-scalings cannot be swapped
            unnoticed.
        """
        source_gt = (100.0, 2.0, 0.5, 200.0, 0.3, -1.5)
        req_xoff, req_yoff, req_xsize, req_ysize, out_w, out_h = 5, 7, 4, 6, 2, 2

        out_gt = COG._output_geotransform(
            source_gt, req_xoff, req_yoff, req_xsize, req_ysize, out_w, out_h
        )

        assert out_gt == pytest.approx((113.5, 4.0, 1.5, 191.0, 0.6, -4.5)), (
            f"rotated composition changed, got {out_gt}"
        )
        for col, row in [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]:
            source_px = req_xoff + col * req_xsize / out_w
            source_line = req_yoff + row * req_ysize / out_h
            expected = gdal.ApplyGeoTransform(list(source_gt), source_px, source_line)
            got = gdal.ApplyGeoTransform(list(out_gt), col, row)
            assert got == pytest.approx(expected), (
                f"output pixel ({col}, {row}) must map to {expected}, got {got}"
            )


class TestPreview:
    """Tests for COG.preview."""

    def test_max_size_long_edge(self, ramp_4326):
        """preview caps the long edge at max_size.

        Args:
            ramp_4326: Fixture ramp Dataset (100x100).

        Test scenario:
            max_size=40 on a square raster yields a 40x40 thumbnail.
        """
        thumb = ramp_4326.preview(max_size=40, band=0)
        assert max(thumb.shape) == 40, f"unexpected shape {thumb.shape}"

    def test_small_raster_not_upsampled(self, ramp_4326):
        """A raster already smaller than max_size is returned at native size.

        Args:
            ramp_4326: Fixture ramp Dataset (100x100).

        Test scenario:
            max_size=512 leaves the 100x100 raster unchanged.
        """
        thumb = ramp_4326.preview(max_size=512, band=0)
        assert thumb.shape == (100, 100), f"unexpected shape {thumb.shape}"


class TestPoint:
    """Tests for COG.point."""

    def test_samples_expected_value(self, ramp_4326):
        """point samples the ramp value at a known pixel.

        Args:
            ramp_4326: Fixture ramp Dataset where value == row*100 + col.

        Test scenario:
            A coordinate at the centre of pixel (col=20, row=30) returns
            30*100 + 20 == 3020.
        """
        x = (20 + 0.5) * 0.01
        y = 10.0 - (30 + 0.5) * 0.01
        value = ramp_4326.point(x, y, point_crs=4326, band=0)
        assert float(value) == pytest.approx(3020.0), f"got {value}"

    def test_all_bands_returns_vector(self, ramp_4326):
        """point with band=None returns a 1-D per-band vector.

        Args:
            ramp_4326: Fixture ramp Dataset (single band).

        Test scenario:
            A valid coordinate yields a length-1 array (one band).
        """
        x = (10 + 0.5) * 0.01
        y = 10.0 - (10 + 0.5) * 0.01
        vec = ramp_4326.point(x, y, point_crs=4326)
        assert vec.shape == (1,), f"unexpected shape {vec.shape}"

    def test_out_of_bounds_raises(self, ramp_4326):
        """A coordinate outside the extent raises OutOfBoundsError.

        Args:
            ramp_4326: Fixture ramp Dataset.

        Test scenario:
            A far-away lon/lat is rejected.
        """
        with pytest.raises(OutOfBoundsError):
            ramp_4326.point(80.0, -10.0, point_crs=4326)

    def test_reprojects_point_crs(self, tile_3857):
        """point reprojects from point_crs to the dataset CRS.

        Args:
            tile_3857: Fixture EPSG:3857 dataset covering the NW quadrant.

        Test scenario:
            A 4326 lon/lat inside the NW quadrant (e.g. -90, 45) samples a
            finite value after reprojection to 3857.
        """
        value = tile_3857.point(-90.0, 45.0, point_crs=4326, band=0)
        assert np.isfinite(float(value)), f"expected a finite sample, got {value}"


class TestReadTile:
    """Tests for COG.read_tile."""

    def test_aligned_tile_shape(self, tile_3857):
        """Reading the aligned XYZ tile returns a tilesize square.

        Args:
            tile_3857: Fixture dataset aligned to zoom-1 tile (0,0).

        Test scenario:
            read_tile(1, 0, 0) over the matching 3857 dataset yields 256x256.
        """
        tile = tile_3857.read_tile(1, 0, 0, tilesize=256, band=0)
        assert tile.shape == (256, 256), f"unexpected shape {tile.shape}"

    def test_custom_tilesize(self, tile_3857):
        """read_tile honours a custom tilesize.

        Args:
            tile_3857: Fixture dataset aligned to zoom-1 tile (0,0).

        Test scenario:
            tilesize=128 yields a 128x128 tile.
        """
        tile = tile_3857.read_tile(1, 0, 0, tilesize=128, band=0)
        assert tile.shape == (128, 128), f"unexpected shape {tile.shape}"

    def test_non_overlapping_tile_raises(self, tile_3857):
        """A tile that does not overlap the raster raises OutOfBoundsError.

        Args:
            tile_3857: Fixture dataset covering only the NW quadrant.

        Test scenario:
            Tile (1, 1, 1) is the south-east quadrant and does not intersect.
        """
        with pytest.raises(OutOfBoundsError):
            tile_3857.read_tile(1, 1, 1, tilesize=256, band=0)

    def test_edge_tile_is_padded(self, straddle_3857):
        """An edge tile that partially overlaps is padded to tilesize (M1).

        Args:
            straddle_3857: Fixture dataset straddling the zoom-1 tile boundary.

        Test scenario:
            The NW tile (1, 0, 0) only covers the dataset's NW corner, so the
            tile is returned at full 256x256 with the non-overlapping remainder
            filled with the dataset's NoData value (-1) — not stretched.
        """
        tile = straddle_3857.read_tile(1, 0, 0, tilesize=256, band=0)
        assert tile.shape == (256, 256), f"edge tile must keep tilesize: {tile.shape}"
        assert (tile == -1.0).any(), "edge tile should have NoData-padded margin"
        assert np.isclose(tile, 1.0).any(), "edge tile should also contain real data"
