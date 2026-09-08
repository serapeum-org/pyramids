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
from osgeo import gdal, osr

from pyramids.base._coverage import check_seam_bbox, seam_offset, window_overlaps
from pyramids.dataset import Dataset, _ogc_coverages, _wcs
from pyramids.dataset.engines.spatial import _stitch_lon_halves
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
            in x, not `0 .. 10`. A window at `11 .. 14` therefore does overlap.
            Measuring the extent from `gt[1]` alone would place the east edge at
            `10` and skip a half that genuinely has data.
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

    def test_a_rotated_grid_is_bounded_on_the_axis_the_diagonal_misses(self):
        """Two opposite corners do not bound a rotated grid; four do.

        Test scenario:
            The same rotated raster's true corners are `(0,10) (10,15) (5,0)
            (15,5)`, so it spans `0 .. 15` in y. Taking only the origin and the
            diagonally-opposite corner gives `5 .. 10` and misses a third of it: a
            window at y `1 .. 2` holds real pixels and was reported as no overlap,
            so a seam half over it was silently dropped.
        """
        rotated = self._src((0.0, 1.0, 0.5, 10.0, 0.5, -1.0))
        result = window_overlaps([5.0, 2.0, 6.0, 1.0], rotated)
        assert result, (
            "a window inside the rotated footprint should overlap; the origin and "
            "its diagonal alone bound y as 5..10 instead of 0..15"
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


class TestTheSeamGuardReachesEveryRasterReader:
    """The refusals `from_wms` always had, now on the two coverage readers too.

    A network reader has no grid to check a bbox against, so once a wrap is
    accepted these two checks are the only thing standing between a transposed
    bbox and a confidently wrong answer.
    """

    def test_wcs_refuses_a_wrapping_bbox_in_a_projected_crs(self):
        """A projected CRS has no 180 degree meridian to split at.

        Test scenario:
            An ordinary transposed Web Mercator box. Split at +/-180 *metres* it
            becomes two windows over central Europe -- nowhere near what was
            asked for, and returned without an error. `from_wcs` takes an explicit
            `crs`, so the caller can reach this.
        """
        box = (2_000_000.0, 6_000_000.0, 1_000_000.0, 6_100_000.0)
        with pytest.raises(ValueError, match="not a geographic"):
            check_seam_bbox(box, "EPSG:3857")

    def test_a_wrapping_bbox_may_not_overhang_the_seam(self):
        """A corner past 180 leaves "west of the seam" meaning nothing.

        Test scenario:
            `(190, -10, -170, 10)` splits into a `(190, ..., 180)` half whose
            native projwin comes out inverted, which the overlap filter then
            discards -- so 20 of the 30 requested degrees vanish silently. It is
            refused instead.
        """
        with pytest.raises(ValueError, match="within -180..180"):
            check_seam_bbox((190.0, -10.0, -170.0, 10.0), "EPSG:4326")

    def test_a_geographic_crs_off_greenwich_is_refused(self):
        """`IsGeographic()` alone does not put the seam at 180.

        Test scenario:
            EPSG:4807 (NTF, Paris) is geographic, but counts longitude from a
            meridian 2.34 degrees east of Greenwich and expresses it in grads,
            where the half-turn is 200. Splitting at 180 would cut such a bbox in
            the wrong place and then measure the halves against a 360 that is not
            the width of the world in those units.
        """
        with pytest.raises(ValueError, match="degrees from Greenwich"):
            check_seam_bbox((170.0, -10.0, -170.0, 10.0), "EPSG:4807")

    def test_an_ordinary_bbox_passes_in_any_crs(self):
        """The guard is a no-op unless the box actually wraps."""
        assert check_seam_bbox((0.0, 0.0, 1e6, 1e6), "EPSG:3857") is None
        assert check_seam_bbox((5.0, 51.0, 6.0, 52.0), "EPSG:4326") is None

    def test_the_wcs_reader_applies_it_before_any_request(self):
        """The refusal happens client-side, so no request is issued.

        Test scenario:
            The mock server is never started -- an endpoint that cannot be reached
            proves the guard fires before the network is touched.
        """
        with pytest.raises(ValueError, match="not a geographic"):
            Dataset.from_wcs(
                "http://127.0.0.1:1/wcs",
                coverage="test_cov",
                bbox=(2_000_000.0, 6_000_000.0, 1_000_000.0, 6_100_000.0),
                crs="EPSG:3857",
                version="1.0.0",
            )

    def test_the_coverages_reader_applies_the_range_check(self):
        """Its bbox is contractually CRS84, so only the corner range can fire."""
        with pytest.raises(ValueError, match="within -180..180"):
            Dataset.from_ogc_coverages(
                "http://127.0.0.1:1/ogcapi",
                coverage="demo",
                bbox=(190.0, -10.0, -170.0, 10.0),
            )


class TestWindowSizes:
    """Sizing the halves of an OGC API coverage read onto one shared grid."""

    def test_single_window_is_sized_as_before(self):
        """One window keeps the plain read_size behaviour: cap the longer side."""
        # 2 deg of longitude against 4 of latitude: the tall side takes the cap.
        assert _ogc_coverages._window_sizes([[0.0, 10.0, 2.0, 6.0]], None) == [
            (512, 1024)
        ]

    def test_the_two_halves_land_on_exactly_one_cell_size(self):
        """The halves must share a grid, not merely be sized from one.

        Test scenario:
            Rounding each half's width independently leaves them with cell sizes
            differing in the sixth decimal -- measured at 0.01953125 against
            0.019526627 before the fix, which puts the merged raster's declared
            east edge about 87 m from where the data ends. The east width is now
            the remainder of the combined width, and the windows are snapped to it.
        """
        halves = [[170.0, 10.0, 180.0, -10.0], [-180.0, 10.0, -176.7, -10.0]]
        sizes = _ogc_coverages._window_sizes(halves, None)
        snapped = _ogc_coverages._align_to_sizes(halves, sizes)
        cells = [
            (window[2] - window[0]) / width
            for window, (width, _) in zip(snapped, sizes)
        ]
        assert cells[0] == pytest.approx(cells[1], rel=1e-12), (
            f"the halves must share one cell size, got {cells}"
        )
        assert sizes[0][1] == sizes[1][1], "the halves must share one row count"

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


class TestTheSeamOffsetIsMeasuredNotAssumed:
    """A coverage is windowed in its own CRS, which need not be degrees.

    Both coverage readers window the source in the coverage's native CRS and then
    stitch. The stitch guard used to hard-code a 360 seam offset, which is right
    only for halves in degrees -- so for any coverage whose native CRS is projected
    (Web Mercator, UTM, a national grid, polar stereographic) a seam read failed
    every time, with a message blaming the caller's grid.
    """

    WORLD = 20037508.342789244

    @staticmethod
    def _half(origin_x: float, columns: int, cell: float, epsg: int) -> Dataset:
        """A one-band raster whose geotransform is stamped before it is wrapped.

        Args:
            origin_x: The window's west edge in the CRS's own units.
            columns: Pixel width.
            cell: Pixel size in the CRS's own units.
            epsg: The CRS to stamp.

        Returns:
            Dataset: The raster half.
        """
        mem = gdal.GetDriverByName("MEM").Create("", columns, 10, 1, gdal.GDT_Byte)
        mem.SetGeoTransform((origin_x, cell, 0.0, 1_000_000.0, 0.0, -cell))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        mem.SetSpatialRef(srs)
        return Dataset(mem, access="write")

    def test_a_geographic_seam_measures_360_degrees(self):
        """The default is still right for halves in degrees."""
        native = osr.SpatialReference()
        native.ImportFromEPSG(4326)
        assert seam_offset(WRAP, "EPSG:4326", native) == pytest.approx(360.0)

    def test_a_web_mercator_seam_measures_the_world_width(self):
        """The same seam is about 40,075,017 m once the coverage is projected."""
        native = osr.SpatialReference()
        native.ImportFromEPSG(3857)
        assert seam_offset(WRAP, "EPSG:4326", native) == pytest.approx(
            2 * self.WORLD, rel=1e-6
        )

    def test_projected_halves_stitch_with_the_measured_offset(self):
        """The case that failed for every projected coverage before the fix.

        Test scenario:
            Two halves in metres meeting exactly at the Web Mercator antimeridian.
            With the measured offset they concatenate; the merged raster is as wide
            as both and keeps the west half's origin.
        """
        west = self._half(self.WORLD - 100_000.0, 100, 1000.0, 3857)
        east = self._half(-self.WORLD, 50, 1000.0, 3857)
        merged = _stitch_lon_halves(west, west, east, 2 * self.WORLD)
        assert merged.columns == 150, (
            f"the stitch should be as wide as both halves, got {merged.columns}"
        )
        assert merged.geotransform[0] == pytest.approx(self.WORLD - 100_000.0)

    def test_the_same_halves_are_refused_under_the_degree_default(self):
        """Which is exactly what both readers used to do to every projected coverage.

        Test scenario:
            The companion to the previous case. Left at the 360 default the guard
            compares metres against degrees, so a perfectly well-formed pair is
            rejected -- the regression this finding was about.
        """
        west = self._half(self.WORLD - 100_000.0, 100, 1000.0, 3857)
        east = self._half(-self.WORLD, 50, 1000.0, 3857)
        with pytest.raises(ValueError, match="not seam-aligned"):
            _stitch_lon_halves(west, west, east)

    def test_halves_at_different_resolutions_are_refused(self):
        """The check the shared guard was missing.

        Test scenario:
            Two halves whose cell sizes differ by more than a rounding wobble. The
            stitch keeps the west geotransform, so accepting these would silently
            misplace the east half's declared edge.
        """
        west = self._half(170.0, 100, 0.1, 4326)
        east = self._half(-180.0, 50, 0.2, 4326)
        with pytest.raises(ValueError, match="different resolutions"):
            _stitch_lon_halves(west, west, east)


class TestNothingIsStrandedWhenAHalfFails:
    """A seam read fetches twice and adopts twice; either can raise part-way.

    Before the split there was one window to fetch and one raster to adopt, so
    neither loop could fail with something already in hand. Both can now, and the
    rest of these readers is deliberately explicit about ownership -- a dedicated
    finally in `_collect_halves`, `src = None` in three finally blocks, `parts`
    closed in a finally. These two loops were the lapse.
    """

    @staticmethod
    def _tracking_translate(monkeypatch, module, closed: list[int]):
        """Replace the module's `_translate_window` with one that fails second.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
            module: The reader module to patch.
            closed: Collects the id of every raster the reader closes.

        Returns:
            None
        """
        calls = {"n": 0}
        real = module._translate_window

        def fake(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("second half exploded")
            mem = real(*args, **kwargs)
            original_close = mem.Close

            def close():
                closed.append(id(mem))
                original_close()

            mem.Close = close
            return mem

        monkeypatch.setattr(module, "_translate_window", fake)

    def test_wcs_closes_the_first_half_when_the_second_fetch_raises(self, monkeypatch):
        """The first MEM raster must not be left to the garbage collector."""
        closed: list[int] = []
        self._tracking_translate(monkeypatch, _wcs, closed)
        with WcsMock(version="1.0.0", bounds=GLOBAL_BOUNDS) as server:
            with pytest.raises(RuntimeError, match="second half exploded"):
                Dataset.from_wcs(
                    server.url, coverage="test_cov", bbox=WRAP, version="1.0.0"
                )
        assert len(closed) == 1, (
            f"the successfully fetched half should be closed on the way out, "
            f"got {len(closed)} close(s)"
        )

    def test_wcs_closes_the_halves_it_could_not_adopt(self):
        """A raise during adoption leaves the untouched tail wrapped by nothing.

        Test scenario:
            `parts` closes what it wrapped, but a raise part-way through means the
            remaining `mems` entries were never wrapped, so nothing else would
            close them. `from_wcs` takes `dataset_cls`, so a stub that fails on its
            second construction reaches the path directly.
        """
        adopted_closed: list[int] = []
        mem_closed: list[int] = []

        class FailsOnSecond:
            made = 0

            def __init__(self, mem, access="read"):
                FailsOnSecond.made += 1
                # Track the raw handle's own Close, which is what the sweep must
                # call for the half that was never wrapped.
                original = mem.Close
                mem.Close = lambda: (mem_closed.append(id(mem)), original())[1]
                if FailsOnSecond.made > 1:
                    raise RuntimeError("adoption exploded")
                self._mem = mem

            def close(self):
                adopted_closed.append(id(self._mem))

        with WcsMock(version="1.0.0", bounds=GLOBAL_BOUNDS) as server:
            with pytest.raises(RuntimeError, match="adoption exploded"):
                _wcs.from_wcs(
                    FailsOnSecond,
                    server.url,
                    coverage="test_cov",
                    bbox=WRAP,
                    crs="EPSG:4326",
                    version="1.0.0",
                    output_crs=None,
                    resolution=None,
                    coverage_crs=None,
                    wcs_format=None,
                    output=None,
                    resample="nearest",
                    direct=False,
                    subset_axes=None,
                    auth=None,
                    timeout=60.0,
                    extra_params=None,
                )
        assert len(adopted_closed) == 1, (
            f"the one part that was adopted is closed by `parts`, got "
            f"{len(adopted_closed)}"
        )
        assert len(mem_closed) == 1, (
            f"the half that was never wrapped has nothing else to close it, so "
            f"the mems tail sweep must; got {len(mem_closed)} raw close(s)"
        )

    @pytest.mark.skipif(
        gdal.GetDriverByName("OGCAPI") is None,
        reason="GDAL build lacks the OGCAPI driver",
    )
    def test_coverages_closes_the_first_half_when_the_second_fetch_raises(
        self, monkeypatch
    ):
        """The same guarantee on the reader that fetches a whole list at once."""
        closed: list[int] = []
        self._tracking_translate(monkeypatch, _ogc_coverages, closed)
        with _serving(OGC_GLOBAL_BOUNDS) as url:
            with pytest.raises(RuntimeError, match="second half exploded"):
                Dataset.from_ogc_coverages(
                    url, coverage="demo", bbox=WRAP, resolution=0.05
                )
        assert len(closed) == 1, (
            f"the successfully fetched half should be closed on the way out, "
            f"got {len(closed)} close(s)"
        )


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
