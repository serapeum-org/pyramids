"""Antimeridian (``minx > maxx``) bboxes through the WCS and OGC API coverage readers.

Both readers used to refuse such a bbox outright — ``ValueError: bbox must have
minx < maxx and miny < maxy`` — while ``Dataset.crop`` had served it for a long
time by splitting at the seam and stitching. These tests pin the new behaviour
(#1088): the request is split into a ``west..180`` and a ``-180..east`` half, each
half is fetched through the reader's normal path, and the two are concatenated
along longitude into one raster whose geotransform continues past the seam.

Everything here is offline. The WCS side drives GDAL's real WCS driver against
``wcs_mock_server``; the OGC API side drives GDAL's real ``OGCAPI`` driver against
the in-process mock in ``test_ogc_coverages``. Both mocks can now advertise a
coverage spanning the whole globe, which is what a seam-crossing request needs to
have data either side of the split.

The proof of a correct stitch is the same in every case: fetch the identical
region as two *separate non-wrapping* requests and require the merged raster to be
their concatenation, pixel for pixel, with the west half's geotransform.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._coverage import window_overlaps
from pyramids.dataset import Dataset, _ogc_coverages, _wcs
from tests.dataset.remote.test_ogc_coverages import GLOBAL_BOUNDS as OGC_GLOBAL_BOUNDS
from tests.dataset.remote.test_ogc_coverages import _serving
from tests.dataset.remote.wcs_mock_server import GLOBAL_BOUNDS, WcsMock

# Deliberately asymmetric: 10 degrees west of the seam, 5 east of it. Equal halves
# would let a swapped or offset stitch pass unnoticed.
WRAP = (170.0, -10.0, -175.0, 10.0)
WEST = (170.0, -10.0, 180.0, 10.0)
EAST = (-180.0, -10.0, -175.0, 10.0)


def _arr(ds: Dataset) -> np.ndarray:
    """The dataset's single band as a 2-D array."""
    return np.asarray(ds.read_array())


def _assert_is_concatenation(merged: Dataset, west: Dataset, east: Dataset) -> None:
    """The merged raster is exactly `west` then `east`, on the west geotransform.

    This is the whole correctness claim in one place: the right width, the west
    half's geotransform continued past the seam (so 170..180 runs on into
    180..190 instead of jumping back to -180), and both halves' pixels where they
    belong.
    """
    m, w, e = _arr(merged), _arr(west), _arr(east)
    assert m.shape == (w.shape[0], w.shape[1] + e.shape[1])
    assert merged.geotransform == pytest.approx(west.geotransform)
    assert np.array_equal(m[:, : w.shape[1]], w)
    assert np.array_equal(m[:, w.shape[1] :], e)
    # The seam is where the west half ends; past it the geotransform keeps
    # counting up rather than wrapping to -180.
    gt = merged.geotransform
    assert gt[0] + w.shape[1] * gt[1] == pytest.approx(180.0)
    assert gt[0] + m.shape[1] * gt[1] > 180.0
    assert merged.epsg == west.epsg


class TestWindowOverlaps:
    """The overlap test that keeps a seam half that misses the coverage unfetched."""

    @staticmethod
    def _src(
        geotransform: tuple[float, float, float, float, float, float] = (
            0.0,
            1.0,
            0.0,
            10.0,
            0.0,
            -1.0,
        ),
    ) -> gdal.Dataset:
        """A 10x10 MEM raster carrying `geotransform`.

        Args:
            geotransform: The GDAL geotransform to stamp. Defaults to the ordinary
                north-up, east-positive one covering `0..10` on both axes.

        Returns:
            osgeo.gdal.Dataset: The in-memory raster.
        """
        src = gdal.GetDriverByName("MEM").Create("", 10, 10, 1)
        src.SetGeoTransform(geotransform)
        return src

    def test_window_inside_overlaps(self):
        assert window_overlaps([2.0, 8.0, 4.0, 6.0], self._src())

    def test_window_partly_outside_overlaps(self):
        assert window_overlaps([-5.0, 8.0, 4.0, 6.0], self._src())

    @pytest.mark.parametrize(
        "projwin",
        [
            [20.0, 8.0, 30.0, 6.0],  # east of the raster
            [-30.0, 8.0, -20.0, 6.0],  # west of it
            [2.0, 30.0, 4.0, 20.0],  # north of it
            [2.0, -20.0, 4.0, -30.0],  # south of it
        ],
    )
    def test_window_outside_does_not_overlap(self, projwin):
        assert not window_overlaps(projwin, self._src())

    def test_touching_edge_does_not_overlap(self):
        """A zero-area intersection has no pixels to read, so it is not an overlap."""
        assert not window_overlaps([10.0, 8.0, 20.0, 6.0], self._src())

    def test_a_rotated_grid_is_measured_by_its_far_corner(self):
        """A rotated geotransform reaches further east than its pixel width suggests.

        Test scenario:
            With a rotation term (`gt[2] = 0.5`) a 10-column raster spans `0 .. 15`
            in x, not `0 .. 10`: the far corner is
            `gt[0] + width*gt[1] + height*gt[2]`. A window at `11 .. 14` therefore
            does overlap. Measuring the extent from `gt[1]` alone would place the
            east edge at `10` and skip a half that genuinely has data.
        """
        rotated = self._src((0.0, 1.0, 0.5, 10.0, 0.5, -1.0))
        result = window_overlaps([11.0, 9.0, 14.0, 6.0], rotated)
        assert result, (
            "a window inside the rotated footprint should overlap; measuring the "
            "extent without the rotation terms would put the east edge at 10"
        )

    def test_a_rotated_grid_still_excludes_what_is_beyond_it(self):
        """The far-corner extent bounds the raster rather than unbounding it.

        Test scenario:
            The same rotated raster reaches `15` in x, so a window at `16 .. 20` is
            outside it. This is the companion to the previous case: widening the
            extent to catch the rotation must not widen it without limit.
        """
        rotated = self._src((0.0, 1.0, 0.5, 10.0, 0.5, -1.0))
        result = window_overlaps([16.0, 9.0, 20.0, 6.0], rotated)
        assert not result, (
            "a window east of the rotated footprint should not overlap, got True"
        )

    def test_a_south_up_grid_is_not_read_as_empty(self):
        """A positive `gt[5]` puts the origin at the raster's south edge.

        Test scenario:
            With `gt[3] = 0` and `gt[5] = +1` the raster runs from `0` up to `10` in
            y, so the origin is its *minimum*. Taking `gt[3]` as the maximum without
            ordering the pair yields `miny > maxy`, which makes the intersection
            test false for every window and would skip both halves of a seam read.
        """
        south_up = self._src((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        result = window_overlaps([2.0, 8.0, 4.0, 6.0], south_up)
        assert result, (
            "a window inside a south-up raster should overlap; an unordered y "
            "extent makes every window miss"
        )

    def test_a_west_positive_grid_is_not_read_as_empty(self):
        """The same ordering problem on the x axis, with a negative `gt[1]`.

        Test scenario:
            `gt[0] = 10` with `gt[1] = -1` runs the columns westward, so the origin
            is the raster's *maximum* x. A window at `2 .. 4` is inside it, and only
            ordering the pair keeps that true.
        """
        west_positive = self._src((10.0, -1.0, 0.0, 10.0, 0.0, -1.0))
        result = window_overlaps([2.0, 8.0, 4.0, 6.0], west_positive)
        assert result, (
            "a window inside a west-positive raster should overlap; an unordered x "
            "extent makes every window miss"
        )


class TestWindowSizes:
    """Sizing the halves of an OGC API coverage read onto one shared grid."""

    def test_single_window_is_sized_as_before(self):
        """One window keeps the plain read_size behaviour: cap the longer side."""
        # 2 deg of longitude against 4 of latitude: the tall side takes the cap.
        assert _ogc_coverages._window_sizes([[0.0, 10.0, 2.0, 6.0]], None) == [
            (512, 1024)
        ]

    def test_two_windows_share_one_resolution(self):
        """Both halves come back on one grid: equal height, widths summing to the cap."""
        west = [170.0, 10.0, 180.0, -10.0]
        east = [-180.0, 10.0, -175.0, -10.0]
        (w_width, w_height), (e_width, e_height) = _ogc_coverages._window_sizes(
            [west, east], None
        )
        assert w_height == e_height
        # 15 deg of longitude against 20 of latitude: the tall side takes the cap.
        assert w_height == 1024
        assert (w_width, e_width) == (512, 256)
        # Same pixel size on both, which is what makes them stitchable.
        assert 10.0 / w_width == pytest.approx(5.0 / e_width)

    def test_explicit_resolution_is_used_for_both(self):
        west = [170.0, 10.0, 180.0, -10.0]
        east = [-180.0, 10.0, -175.0, -10.0]
        assert _ogc_coverages._window_sizes([west, east], (0.1, 0.1)) == [
            (100, 200),
            (50, 200),
        ]


class TestWcsDirect:
    """Direct KVP GetCoverage: one request per half, then a stitch."""

    def test_wrapping_bbox_is_the_concatenation_of_its_halves(self):
        with WcsMock(version="2.0.1") as server:
            merged = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=WRAP,
                version="2.0.1",
                direct=True,
            )
            west = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=WEST,
                version="2.0.1",
                direct=True,
            )
            east = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=EAST,
                version="2.0.1",
                direct=True,
            )
        assert merged.shape == (1, 200, 150)
        _assert_is_concatenation(merged, west, east)

    def test_one_request_per_half_carrying_that_half_s_subset(self):
        """The URL built for each half asks for that half, not for the wrap."""
        with WcsMock(version="2.0.1") as server:
            Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=WRAP,
                version="2.0.1",
                direct=True,
            )
            requests = server.getcoverage_requests()
        assert len(requests) == 2
        assert "SUBSET=Long(170.0,180.0)" in requests[0]
        assert "SUBSET=Long(-180.0,-175.0)" in requests[1]
        for request in requests:
            assert "SUBSET=Lat(-10.0,10.0)" in request
            # The wrapping bbox itself must never reach the wire.
            assert "170.0,-175.0" not in request

    def test_output_crs_is_applied_once_to_the_merged_raster(self):
        with WcsMock(version="2.0.1") as server:
            merged = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=WRAP,
                version="2.0.1",
                direct=True,
                output_crs="EPSG:3857",
            )
        assert merged.epsg == 3857

    def test_output_writes_one_reopenable_stitched_file(self, tmp_path):
        out = tmp_path / "wcs_seam.tif"
        with WcsMock(version="2.0.1") as server:
            Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=WRAP,
                version="2.0.1",
                direct=True,
                output=out,
            )
        assert out.exists()
        assert Dataset.read_file(str(out)).shape == (1, 200, 150)


class TestWcsDiscovery:
    """Full GetCapabilities → DescribeCoverage → GetCoverage cycle across the seam."""

    @pytest.mark.parametrize("version", ["1.0.0", "2.0.1"])
    def test_wrapping_bbox_is_the_concatenation_of_its_halves(self, version):
        with WcsMock(version=version, bounds=GLOBAL_BOUNDS) as server:
            merged = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=WRAP, version=version
            )
            west = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=WEST, version=version
            )
            east = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=EAST, version=version
            )
        assert merged.shape == (1, 200, 150)
        _assert_is_concatenation(merged, west, east)

    def test_non_wrapping_bbox_is_unchanged(self):
        """The ordinary request keeps the exact raster it had before the split."""
        with WcsMock(version="1.0.0") as server:
            ds = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=(2.0, 2.0, 4.0, 4.0),
                version="1.0.0",
            )
        assert ds.shape == (1, 20, 20)
        assert ds.geotransform == pytest.approx((2.0, 0.1, 0.0, 4.0, 0.0, -0.1))

    def test_half_that_misses_the_coverage_is_not_requested(self):
        """A coverage reaching the seam from the west only returns just that half."""
        with WcsMock(version="1.0.0", bounds=(170.0, -20.0, 180.0, 20.0)) as server:
            merged = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=WRAP, version="1.0.0"
            )
            west = Dataset.from_wcs(
                server.url, coverage="test_cov", bbox=WEST, version="1.0.0"
            )
        assert merged.shape == west.shape == (1, 200, 100)
        assert np.array_equal(_arr(merged), _arr(west))
        assert merged.geotransform == pytest.approx(west.geotransform)

    def test_no_overlapping_half_raises(self):
        """A regional coverage nowhere near the seam gets a clear refusal."""
        with WcsMock(version="1.0.0") as server:
            with pytest.raises(ValueError, match="neither half overlaps"):
                Dataset.from_wcs(
                    server.url, coverage="test_cov", bbox=WRAP, version="1.0.0"
                )

    def test_a_single_window_that_misses_is_left_to_gdal(self):
        """The overlap filter must not change what a *non-wrapping* request does.

        Test scenario:
            An ordinary bbox nowhere near the coverage is one window, not two, so
            the filter is skipped entirely and GDAL's own lenient behaviour
            survives: it warns that the source window falls outside the raster and
            fills the result with no-data rather than raising. Applying the filter
            unconditionally would turn that long-standing outcome into
            `neither half overlaps`, which is why the guard reads
            `len(windows) > 1`.
        """
        with WcsMock(version="1.0.0") as server:
            ds = Dataset.from_wcs(
                server.url,
                coverage="test_cov",
                bbox=(50.0, 50.0, 52.0, 52.0),
                version="1.0.0",
            )
        assert ds.shape == (1, 20, 20), (
            f"a single off-coverage window should still return a raster, got {ds.shape}"
        )


@pytest.mark.skipif(
    gdal.GetDriverByName("OGCAPI") is None,
    reason="GDAL build lacks the OGCAPI driver; the real-driver path cannot run",
)
class TestOgcCoverages:
    """The same behaviour through GDAL's real OGCAPI driver."""

    def test_wrapping_bbox_is_the_concatenation_of_its_halves(self):
        with _serving(OGC_GLOBAL_BOUNDS) as url:
            merged = Dataset.from_ogc_coverages(url, coverage="demo", bbox=WRAP)
            west = Dataset.from_ogc_coverages(url, coverage="demo", bbox=WEST)
            east = Dataset.from_ogc_coverages(url, coverage="demo", bbox=EAST)
        _assert_is_concatenation(merged, west, east)

    def test_no_resolution_keeps_the_whole_seam_read_inside_one_pixel_budget(self):
        """The cap is spent on the combined span, not twice over on each half."""
        with _serving(OGC_GLOBAL_BOUNDS) as url:
            merged = Dataset.from_ogc_coverages(url, coverage="demo", bbox=WRAP)
        # 15 deg of longitude against 20 of latitude: the tall side takes the
        # 1024 px cap and the width follows the aspect ratio, exactly as an
        # unwrapped 15x20 deg window would.
        assert merged.shape == (1, 1024, 768)

    def test_explicit_resolution_grids_both_halves_the_same(self):
        with _serving(OGC_GLOBAL_BOUNDS) as url:
            merged = Dataset.from_ogc_coverages(
                url, coverage="demo", bbox=WRAP, resolution=0.05
            )
        assert merged.shape == (1, 400, 300)
        assert merged.geotransform == pytest.approx(
            (170.0, 0.05, 0.0, 10.0, 0.0, -0.05)
        )

    def test_half_that_misses_the_coverage_is_not_requested(self):
        with _serving((170.0, -20.0, 180.0, 20.0)) as url:
            merged = Dataset.from_ogc_coverages(url, coverage="demo", bbox=WRAP)
            west = Dataset.from_ogc_coverages(url, coverage="demo", bbox=WEST)
        assert merged.shape == west.shape
        assert np.array_equal(_arr(merged), _arr(west))

    def test_no_overlapping_half_raises(self):
        with _serving((0.0, 0.0, 10.0, 8.0)) as url:
            with pytest.raises(ValueError, match="neither half overlaps"):
                Dataset.from_ogc_coverages(url, coverage="demo", bbox=WRAP)

    def test_a_single_window_that_misses_is_left_to_gdal(self):
        """The same guard, on the reader that filters a whole list at once.

        Test scenario:
            `_fetch_windows` filters `projwins` only when there is more than one, so
            an ordinary off-coverage bbox keeps GDAL's no-data fill instead of
            gaining a refusal it never had. Without the length guard the list would
            empty and `_window_sizes` would be handed nothing.
        """
        with _serving((0.0, 0.0, 10.0, 8.0)) as url:
            ds = Dataset.from_ogc_coverages(
                url, coverage="demo", bbox=(50.0, 50.0, 52.0, 52.0), resolution=0.1
            )
        assert ds.shape == (1, 20, 20), (
            f"a single off-coverage window should still return a raster, got {ds.shape}"
        )

    def test_output_crs_is_applied_once_to_the_merged_raster(self):
        with _serving(OGC_GLOBAL_BOUNDS) as url:
            merged = Dataset.from_ogc_coverages(
                url, coverage="demo", bbox=WRAP, resolution=0.05, output_crs="EPSG:3857"
            )
        assert merged.epsg == 3857
