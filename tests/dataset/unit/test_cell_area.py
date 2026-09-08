"""Ground area of cells, and the area-weighted domain that sums it.

`cell_size` answers in the CRS's units and `count_domain_cells` weighs every
cell alike, so neither can say how much ground a geographic raster covers -- a
1-degree cell spans 12 309 km2 at the equator and 2 272 km2 at 80 degrees
north. `get_cell_polygons().area` cannot either: those polygons are in degrees,
so every cell on the grid reports the same number.

The area is taken on the CRS's own ellipsoid rather than a sphere of an assumed
radius. That is not a detail: against WGS84 the two differ by 0.45 % at the
equator and 0.85 % at 80 degrees north, and the ellipsoidal sum reproduces the
true ellipsoid surface area exactly where the spherical one is 0.00005 % out.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from osgeo import gdal
from pyproj import CRS

from pyramids.base._domain import is_stored_no_data
from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

GEOGRAPHIC = GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326)
WGS84_ELLIPSOID_KM2 = 510_065_622.0
SPHERICAL_DATUM_WKT = (
    'GEOGCS["Sphere",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _global_grid(values: np.ndarray | None = None, no_data=None) -> Dataset:
    """A 1-degree global raster.

    Args:
        values: Band values; ones when omitted.
        no_data: The sentinel to declare.

    Returns:
        Dataset: The raster.
    """
    if values is None:
        values = np.ones((180, 360), dtype="float32")
    return Dataset.from_array(values, geo_ref=GEOGRAPHIC, no_data_value=no_data)


class TestCellAreaOnAProjectedGrid:
    """A linear CRS gives one area for every cell, from the geotransform."""

    def test_a_square_cell_is_its_side_squared(self):
        """Test scenario: a 30 m UTM cell covers 900 m2, in the CRS's own unit."""
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
        raster = Dataset.from_array(np.ones((3, 4), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area()

        assert areas.shape == (3, 4)
        assert np.allclose(areas, 900.0)

    def test_a_rotated_cell_uses_the_determinant(self):
        """Rotation shears the cell; its area is the determinant, not dx*dy.

        Test scenario:
            A rotated geotransform still describes a parallelogram, whose area
            is `|dx*dy - rx*ry|`. Taking `dx*dy` alone would report the area of
            the bounding rectangle instead, which is larger.
        """
        geo_ref = GeoReference(geo=(0.0, 30.0, 10.0, 0.0, 10.0, -30.0), epsg=32636)
        raster = Dataset.from_array(np.ones((2, 2), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area()

        assert np.allclose(areas, abs(30.0 * -30.0 - 10.0 * 10.0))

    def test_a_projected_cell_does_not_vary_by_row(self):
        """Test scenario: only a geographic grid varies with latitude.

        The value is asserted alongside the uniformity. The projected branch
        broadcasts a single scalar, so one distinct value is guaranteed by
        construction whatever that scalar is -- returning twice the correct
        area passed this test until the `allclose` was added.
        """
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
        raster = Dataset.from_array(np.ones((5, 5), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area()

        assert len(np.unique(areas)) == 1
        assert np.allclose(areas, 900.0)


class TestCellAreaOnAGeographicGrid:
    """Latitude decides the answer, and the ellipsoid decides it exactly."""

    def test_cells_shrink_towards_the_pole(self):
        """The reason the method exists.

        Test scenario:
            `cell_size` reports 1.0 for every row of this raster. The ground
            area is not remotely constant: the equatorial row is more than five
            times the row at 80 degrees north.
        """
        areas = _global_grid().cell_area(unit="km2")

        equator = float(areas[90, 0])
        high_latitude = float(areas[10, 0])
        # Bounded on both sides against the true ratio of ~5.42. A bare
        # `> 5` sat almost on top of it, so it was closer to becoming flaky
        # than to catching a change.
        assert 5.3 < equator / high_latitude < 5.6

    def test_it_matches_the_exact_quadrilateral_area(self):
        """Pinned against the closed form derived here, not against the code.

        Test scenario:
            A cell is bounded by parallels and meridians, so its exact area is
            the ellipsoidal area element integrated between its two latitudes.
            Re-deriving that here, independently of the implementation, is what
            rules out both an off-by-one in the row edges and a return to a
            geodesic polygon -- which bows the north and south edges poleward
            and is wrong by 2.6e-05 per cell even at 1 degree.
        """
        areas = _global_grid().cell_area()
        geod = CRS.from_epsg(4326).get_geod()
        a, f = geod.a, geod.f
        e2 = f * (2.0 - f)
        e = np.sqrt(e2)

        def zone(degrees: float) -> float:
            """Area south of a latitude, per radian of longitude."""
            s = np.sin(np.deg2rad(degrees))
            return (
                a
                * a
                * (1 - e2)
                * (s / (2 * (1 - e2 * s * s)) + np.arctanh(e * s) / (2 * e))
            )

        for row in (0, 45, 90, 179):
            top = 90.0 - row
            expected = np.deg2rad(1.0) * (zone(top) - zone(top - 1.0))
            assert float(areas[row, 0]) == pytest.approx(expected, rel=1e-12)

    def test_a_geodesic_polygon_would_not_pass(self):
        """The defect the closed form replaces, pinned so it cannot return.

        Test scenario:
            `Geod.polygon_area_perimeter` joins the corners with geodesics, and
            a parallel is not one. The bias is small per cell and cancels
            exactly over a whole globe -- which is why an ungapped-globe check
            cannot catch it and this test must.
        """
        areas = _global_grid().cell_area()
        geod = CRS.from_epsg(4326).get_geod()

        geodesic, _ = geod.polygon_area_perimeter(
            [0.0, 1.0, 1.0, 0.0], [90.0, 90.0, 89.0, 89.0]
        )
        assert float(areas[0, 0]) != pytest.approx(abs(geodesic), rel=1e-6)

    def test_the_ellipsoid_is_used_rather_than_a_sphere(self):
        """The difference is 0.45 % at the equator -- larger than a rounding.

        Test scenario:
            A sphere of the authalic radius gets the global total right by
            construction but every individual cell wrong. Pinning the gap stops
            a future simplification to `R**2 * dlam * (sin a - sin b)` from
            passing unnoticed.
        """
        equator = float(_global_grid().cell_area(unit="km2")[90, 0])

        spherical = (
            6371.0088**2
            * np.deg2rad(1.0)
            * (np.sin(np.deg2rad(0.0)) - np.sin(np.deg2rad(-1.0)))
        )
        assert equator != pytest.approx(spherical, rel=1e-4)
        assert equator == pytest.approx(spherical, rel=5e-3)

    def test_the_rows_are_symmetric_about_the_equator(self):
        """Test scenario: an ellipsoid of revolution is symmetric, so its rows are."""
        areas = _global_grid().cell_area()

        assert np.allclose(areas[:90, 0], areas[179:89:-1, 0], rtol=1e-12)

    def test_the_result_is_a_view_over_one_value_per_row(self):
        """A global grid must not cost one float per pixel.

        Test scenario:
            The areas are constant along a row, so the 2-D result is broadcast
            from 180 values rather than materialising 64 800. Complaining that
            `get_cell_polygons` builds one geometry per pixel and then doing
            the same here would miss the point of the issue.
        """
        areas = _global_grid().cell_area()

        assert areas.shape == (180, 360)
        # Asked through the public flags rather than `.base`, a numpy
        # internal: a broadcast view owns no data and cannot be written.
        assert not areas.flags.owndata
        assert not areas.flags.writeable


class TestTheUnits:
    """`m2`, `km2` and `ha`, and a clear refusal for anything else."""

    @pytest.mark.parametrize(
        ("unit", "divisor"), [("m2", 1.0), ("km2", 1e6), ("ha", 1e4)]
    )
    def test_each_unit_scales_the_same_answer(self, unit: str, divisor: float):
        """Each unit is the square-metre answer divided by a constant.

        Args:
            unit: The requested unit.
            divisor: Square metres in one of it.

        Test scenario:
            The unit changes the number, never the geometry, so every answer is
            the square-metre one divided by a constant.
        """
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=100.0, epsg=32636)
        raster = Dataset.from_array(np.ones((2, 2), "float32"), geo_ref=geo_ref)

        assert float(raster.cell_area(unit=unit)[0, 0]) == pytest.approx(
            10_000.0 / divisor
        )

    def test_an_unknown_unit_is_refused_by_name(self):
        """Test scenario: the message lists what is accepted, not just what is not."""
        raster = _global_grid()

        with pytest.raises(ValueError, match="unknown area unit"):
            raster.cell_area(unit="acres")


class TestWhatCannotBeAnswered:
    """Two cases where the honest answer is a refusal."""

    def test_a_raster_with_no_crs_is_refused(self):
        """Cells have no ground area until the grid is located on the earth.

        Test scenario:
            Built through GDAL directly rather than `from_array`, which
            defaults an omitted `epsg` to 4326 and so cannot produce this. A
            raster read from a file carrying no projection can.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        raster = Dataset(handle)

        # `require_crs_spec` phrases it, so the message names the operation
        # and the fix; `CRSError` is a `ValueError`, so the contract holds.
        with pytest.raises(ValueError, match="cannot compute cell area"):
            raster.cell_area()

    def test_a_rotated_geographic_raster_is_refused(self):
        """Rotation breaks the one assumption the per-row shortcut rests on.

        Test scenario:
            Cells in a row no longer share a latitude band, so a per-row answer
            would be wrong and a per-cell one would spend a geodesic call on
            every pixel. The message names the fix rather than guessing.
        """
        geo_ref = GeoReference(geo=(0.0, 1.0, 0.2, 90.0, 0.2, -1.0), epsg=4326)
        raster = Dataset.from_array(np.ones((4, 4), "float32"), geo_ref=geo_ref)

        with pytest.raises(ValueError, match="rotated geographic"):
            raster.cell_area()


class TestTheRefusalsAddedAfterReview:
    """Inputs that used to leak an internal error instead of a clear one."""

    @pytest.mark.parametrize("unit", [["km2"], {"km2": 1}, None, 2])
    def test_any_bad_unit_gives_the_same_refusal(self, unit):
        """A dict lookup fails two ways; the caller should not see the difference.

        Args:
            unit: An argument that is not a unit this package converts to.

        Test scenario:
            An unhashable argument raises `TypeError` from the lookup itself
            rather than missing it, so it escaped a guard that caught only
            `KeyError` -- leaving `unit=["km2"]` with numpy's "unhashable
            type" while `unit=None` got the documented message.
        """
        with pytest.raises(ValueError, match="unknown area unit"):
            _global_grid().cell_area(unit=unit)

    def test_a_band_out_of_range_is_named(self):
        """Test scenario: indexing the sentinel tuple first gave `IndexError`."""
        with pytest.raises(ValueError, match="out of range for a 1-band"):
            _global_grid().domain_area(band=5)

    def test_a_geocentric_crs_is_refused(self):
        """Axes that are not a ground plane have no cell area.

        Test scenario:
            EPSG:4978 is geocentric -- metres from the earth's centre on three
            axes. It is neither geographic nor projected, so it used to fall
            into the projected branch and get the determinant of a
            geotransform that describes nothing planar.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        handle.SetProjection(CRS.from_epsg(4978).to_wkt())
        raster = Dataset(handle)

        with pytest.raises(ValueError, match="neither geographic nor projected"):
            raster.cell_area()

    def test_a_degenerate_geotransform_is_refused(self):
        """A cell with no extent is not a cell of zero area.

        Test scenario:
            A zero cell size, or a rotation that collapses the parallelogram,
            gives a determinant of zero. Answering `0.0` would make
            `domain_area` report no ground for a raster full of data.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((0.0, 10.0, 10.0, 0.0, 10.0, 10.0))
        handle.SetProjection(CRS.from_epsg(32636).to_wkt())
        raster = Dataset(handle)

        with pytest.raises(ValueError, match="no area"):
            raster.cell_area()

    def test_a_half_cell_past_the_pole_keeps_the_globe_exact(self):
        """The cell-centred convention puts the first edge half a cell outside.

        Test scenario:
            ERA5 and friends centre their top row on 90, so its upper edge sits
            at 90.125. Only the half of that cell which exists is real ground,
            which is exactly what clipping the edge to the pole integrates --
            so the grid must still total the ellipsoid. Asserting merely that
            the answer is finite, as this test first did, held whether or not
            the clip was there at all: `arctanh(e sin(phi))` cannot be `nan`
            for any real latitude, so there was never a `nan` to prevent.
        """
        geo_ref = GeoReference(
            geo=(-180.0, 0.25, 0.0, 90.125, 0.0, -0.25), epsg=4326
        )
        raster = Dataset.from_array(np.ones((721, 1440), "float32"), geo_ref=geo_ref)

        total = raster.cell_area(unit="km2").sum() / 1e6

        assert total == pytest.approx(WGS84_ELLIPSOID_KM2 / 1e6, rel=1e-9)

    def test_a_row_entirely_past_the_pole_is_refused_by_its_latitudes(self):
        """Ground that is not on the ellipsoid is not ground of zero area.

        Test scenario:
            Clipping turns such a row into an exact 0.0, which used to reach
            the degenerate-geotransform guard and refuse the whole raster --
            all 180 well-defined rows with it -- while blaming the cell size
            and the rotation terms, neither of which is wrong here.
        """
        geo_ref = GeoReference(geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326)
        raster = Dataset.from_array(np.ones((200, 360), "float32"), geo_ref=geo_ref)

        with pytest.raises(ValueError, match="runs off the ellipsoid") as caught:
            raster.cell_area()

        assert "19 of its 200 rows" in str(caught.value)

    def test_the_clip_is_what_keeps_an_overshoot_from_reading_as_a_lower_row(self):
        """Past the pole the sine turns back down, so 100 degrees reads as 80.

        Test scenario:
            The hazard the clip addresses is a plausible wrong number, not a
            `nan`. A raster whose rows straddle the pole keeps only the real
            ground: the half-cell above 90 must contribute less than the full
            cell below it, never the mirrored area of the 80-degree band.
        """
        geo_ref = GeoReference(geo=(-180.0, 1.0, 0.0, 90.5, 0.0, -1.0), epsg=4326)
        raster = Dataset.from_array(np.ones((3, 360), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area(unit="km2")

        # The 89.5-90 zone, not the 90-90.5 one reflected back down: were the
        # clip removed, the top edge would read as 89.5 and the row would
        # collapse to 0 instead.
        assert float(areas[0, 0]) == pytest.approx(27.2172, rel=1e-5)
        assert float(areas[0, 0]) < float(areas[1, 0])


class TestDomainArea:
    """The area-weighted sibling of `count_domain_cells`."""

    def test_an_ungapped_globe_covers_the_ellipsoid(self):
        """The acceptance test the issue proposed, on the right figure.

        Test scenario:
            Summing every cell of a 1-degree global grid must reproduce the
            surface area of the CRS's ellipsoid. That is what proves the
            weighting is right rather than merely plausible -- an error in the
            row edges or the band formula would not survive it.
        """
        total = _global_grid().domain_area(unit="km2")

        assert total == pytest.approx(WGS84_ELLIPSOID_KM2, rel=1e-6)

    def test_it_honours_the_bands_no_data(self):
        """The same cells `count_domain_cells` counts, weighted.

        Test scenario:
            Masking everything below 60 degrees north leaves the polar cap. Its
            area must match the rows that survive, and nothing else -- which is
            what ties this to the existing domain concept rather than inventing
            a second one.
        """
        values = np.ones((180, 360), dtype="float32")
        latitudes = 90.0 - np.arange(180) - 0.5
        values[latitudes < 60.0, :] = -9999.0
        cap = _global_grid(values, no_data=-9999.0)

        areas = cap.cell_area(unit="km2")
        expected = float(areas[latitudes >= 60.0, :].sum())
        assert cap.domain_area(unit="km2") == pytest.approx(expected, rel=1e-12)

    def test_the_unweighted_count_overstates_a_polar_domain(self):
        """The defect the issue is about, stated as a ratio.

        Test scenario:
            Counting cells and multiplying by a nominal area overstates the
            cap north of 60 degrees roughly fourfold, because every cell in it
            is far smaller than one at the equator.
        """
        values = np.ones((180, 360), dtype="float32")
        latitudes = 90.0 - np.arange(180) - 0.5
        values[latitudes < 60.0, :] = -9999.0
        cap = _global_grid(values, no_data=-9999.0)

        weighted = cap.domain_area(unit="km2")
        nominal = float(cap.cell_area(unit="km2")[90, 0])
        naive = cap.count_domain_cells() * nominal

        # Bounded both ways around the true ~3.9 that the `domain_area`
        # doctest pins exactly. A bare `> 3.5` was satisfied by anything that
        # drove `weighted` toward zero, which is the failure most worth
        # catching here.
        assert 3.8 < naive / weighted < 4.0

    def test_a_fully_masked_band_has_no_area(self):
        """Test scenario: no valid cell, no ground -- and no division by zero."""
        values = np.full((10, 10), -9999.0, dtype="float32")
        raster = Dataset.from_array(values, geo_ref=GEOGRAPHIC, no_data_value=-9999.0)

        assert raster.domain_area() == 0.0

    def test_a_raster_taller_than_one_strip_weighs_each_row(self):
        """The row offset only matters once the reader makes a second call.

        Test scenario:
            `stream_reduce` cuts the band into 256-row strips, so every raster
            shorter than that arrives in a single call with `yoff == 0` and the
            slice `areas[0:rows]` is a no-op. A 0.25 degree global grid is 720
            rows -- three strips -- and is the size issue #1085 names as the
            live case. Ignoring `yoff` here answers 385 731 207 km2 instead of
            the ellipsoid's 510 065 622, a 24 % error that no shorter raster
            can reveal.
        """
        geo_ref = GeoReference(
            top_left_corner=(-180.0, 90.0), cell_size=0.25, epsg=4326
        )
        grid = Dataset.from_array(np.ones((720, 1440), "float32"), geo_ref=geo_ref)

        assert grid.rows > 256, "the raster must span more than one strip"
        assert grid.domain_area(unit="km2") == pytest.approx(
            WGS84_ELLIPSOID_KM2, rel=1e-6
        )

    def test_a_tall_partial_domain_is_weighed_strip_by_strip(self):
        """A gap in the last strip must still be weighed by its own latitudes.

        Test scenario:
            The globe total is symmetric, so a mis-sliced weighting can still
            land on it by cancellation. Masking a band that falls in the third
            strip removes that escape: the answer has to match the rows that
            actually survive.
        """
        geo_ref = GeoReference(
            top_left_corner=(-180.0, 90.0), cell_size=0.25, epsg=4326
        )
        values = np.ones((720, 1440), dtype="float32")
        values[600:, :] = -9999.0
        grid = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)

        areas = grid.cell_area(unit="km2")
        expected = float(areas[:600, :].sum())
        assert grid.domain_area(unit="km2") == pytest.approx(expected, rel=1e-12)

    def test_it_matches_a_whole_band_sum_despite_streaming(self):
        """The strips must line up with the rows they weigh.

        Test scenario:
            The areas vary by row and the band is read in row strips, so a
            window offset applied to the wrong axis -- or not at all -- would
            weigh cells with another latitude's area. Summing the whole band at
            once is the reference.
        """
        rng = np.random.default_rng(1337)
        values = rng.random((180, 360)).astype("float32")
        values[values < 0.25] = -9999.0
        raster = _global_grid(values, no_data=-9999.0)

        areas = raster.cell_area()
        # The same predicate the implementation uses, rather than a second
        # idea of what counts as no-data: `is_stored_no_data` gives integer
        # bands no tolerance at all and floating ones single precision's `eps`
        # with no absolute term, which an `np.isclose` reference would not
        # reproduce.
        inside = ~is_stored_no_data(np.asarray(raster.read_array()), -9999.0)
        expected = float((areas * inside).sum())

        assert raster.domain_area() == pytest.approx(expected, rel=1e-9)

    def test_a_projected_domain_is_the_count_times_the_cell(self):
        """Test scenario: with constant cells the weighted answer is the simple one."""
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
        values = np.ones((6, 6), dtype="float32")
        values[0, :] = -9999.0
        raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)

        assert raster.domain_area() == pytest.approx(
            raster.count_domain_cells() * 900.0
        )


class TestTheEdgesTheHappyPathMisses:
    """Cases a global WGS84 grid never exercises."""

    def test_a_projected_crs_in_feet_is_converted(self):
        """The linear unit is read from the CRS, not assumed to be metres.

        Test scenario:
            EPSG:2225 measures in US survey feet, so a 100-unit cell is 30.48 m
            on a side and covers 929 m2, not 10 000. Assuming metres would
            overstate every area on such a raster by a factor of ten.
        """
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=100.0, epsg=2225)
        raster = Dataset.from_array(np.ones((2, 2), "float32"), geo_ref=geo_ref)

        side_m = 100.0 * CRS.from_epsg(2225).axis_info[0].unit_conversion_factor
        assert float(raster.cell_area()[0, 0]) == pytest.approx(side_m**2, rel=1e-12)

    def test_each_band_is_weighed_by_its_own_domain(self):
        """`band=` selects which mask is applied, as it does for the count.

        Test scenario:
            The second band masks a row the first does not, so it must report
            less area. Ignoring `band` would give both the same answer.
        """
        geo_ref = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        values = np.ones((2, 4, 4), dtype="float32")
        values[1, 0, :] = -9999.0
        raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)

        # Pinned to the rows each band actually keeps, not merely ordered: a
        # band that returned 0.0 from an unrelated fault would satisfy
        # `band1 < band0` while being entirely wrong.
        areas = raster.cell_area()
        assert raster.domain_area(band=0) == pytest.approx(float(areas.sum()))
        assert raster.domain_area(band=1) == pytest.approx(float(areas[1:, :].sum()))

    def test_a_band_declaring_no_sentinel_is_wholly_domain(self):
        """No sentinel means no gaps, so every cell counts.

        Test scenario:
            This is the shape `count_domain_cells` already answers with the
            full cell count, and the weighted sibling has to agree with it
            rather than treating an absent sentinel as masking everything.
        """
        geo_ref = GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326)
        raster = Dataset.from_array(
            np.ones((3, 3), "float32"), geo_ref=geo_ref, no_data_value=None
        )

        assert raster.count_domain_cells() == 9
        assert raster.domain_area() == pytest.approx(float(raster.cell_area().sum()))

    def test_a_south_up_grid_has_positive_areas(self):
        """A positive `dy` walks the rows the other way; area has no sign.

        Test scenario:
            The row edges are built from the geotransform, so a south-up raster
            gives each band's lower edge first. `polygon_area_perimeter` signs
            its answer by winding order, which is why the magnitude is taken.
        """
        south_up = Dataset.from_array(
            np.ones((4, 4), "float32"),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, -10.0, 0.0, 1.0), epsg=4326),
        )
        north_up = Dataset.from_array(
            np.ones((4, 4), "float32"),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, -6.0, 0.0, -1.0), epsg=4326),
        )

        areas = south_up.cell_area()

        assert np.all(np.isfinite(areas))
        # The two describe the same four latitude bands in opposite order, so
        # one is the other reversed. Asserting only "finite and positive" would
        # pass for any answer at all.
        assert np.allclose(areas[:, 0], north_up.cell_area()[::-1, 0], rtol=1e-12)


class TestASphericalDatum:
    """The weather-model datum: a real ellipsoid whose flattening is zero."""

    def test_the_sphere_is_integrated_in_closed_form_too(self):
        """A sphere is not a special case to the caller, only to the integral.

        Test scenario:
            The general antiderivative divides by the eccentricity, which is
            zero here. The limit as it vanishes is `a^2 sin(phi)`, so a global
            grid must come to exactly `4 pi R^2` -- the closed form of the
            surface it is integrating. GRIB rasters from every major weather
            model carry precisely this datum, so the branch is ordinary input.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 360, 180, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((-180.0, 1.0, 0.0, 90.0, 0.0, -1.0))
        handle.SetProjection(SPHERICAL_DATUM_WKT)
        raster = Dataset(handle)

        total = float(raster.cell_area(unit="km2").sum())

        assert total == pytest.approx(4.0 * np.pi * 6371.229**2, rel=1e-12)

    def test_the_sphere_disagrees_with_the_ellipsoid_where_it_should(self):
        """Zero flattening is a different earth, not a rounding difference.

        Test scenario:
            Were the flattening ignored -- or the eccentricity branch taken
            with `e = 0` folded in wrongly -- the two datums would answer alike.
            They must not: a sphere puts more area at the equator and less at
            the pole than WGS84 does, while the totals stay within 0.01 %.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 360, 180, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((-180.0, 1.0, 0.0, 90.0, 0.0, -1.0))
        handle.SetProjection(SPHERICAL_DATUM_WKT)
        sphere = Dataset(handle).cell_area(unit="km2")
        ellipsoid = _global_grid().cell_area(unit="km2")

        assert float(sphere[90, 0]) > float(ellipsoid[90, 0])
        assert float(sphere[0, 0]) < float(ellipsoid[0, 0])
        assert float(sphere.sum()) == pytest.approx(float(ellipsoid.sum()), rel=1e-4)


class TestTheGuardsNoPublicInputReaches:
    """A defensive branch, forced open.

    `get_geod()` returned an ellipsoid for every geographic CRS tried,
    including a datum naming none, so nothing public reaches this. It is kept
    because the return is typed `Geod | None`; the test forces the state
    rather than producing it, and asserts only that the refusal is the
    documented one.
    """

    def test_a_geographic_crs_without_an_ellipsoid_is_refused(self, monkeypatch):
        """There is no figure of the earth to integrate over.

        Args:
            monkeypatch: Returns a CRS whose `get_geod()` is `None`.
        """
        monkeypatch.setattr(CRS, "get_geod", lambda self: None)

        with pytest.raises(ValueError, match="declares no ellipsoid"):
            _global_grid().cell_area()


class TestACompoundCrs:
    """A DEM carrying a vertical datum is still a horizontal grid."""

    @pytest.mark.parametrize(
        "epsg, geo, expected",
        [
            (5972, (0.0, 30.0, 0.0, 0.0, 0.0, -30.0), 900.0),
            (9518, (0.0, 1.0, 0.0, 1.0, 0.0, -1.0), 12308.0e6),
        ],
    )
    def test_it_takes_the_branch_its_horizontal_part_names(self, epsg, geo, expected):
        """Compound CRSs answer; they do not fall into the geocentric refusal.

        Args:
            epsg: A compound CRS -- projected, then geographic.
            geo: Its geotransform.
            expected: The area of one cell in square metres.

        Test scenario:
            The refusal branch's comment used to claim compound CRSs reached
            it, which would have refused every DEM with a vertical datum.
            pyproj reads `is_geographic` and `is_projected` through to the
            horizontal part, so they do not -- and nothing in the suite said
            so either way.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        handle.SetGeoTransform(geo)
        handle.SetProjection(CRS.from_epsg(epsg).to_wkt())
        raster = Dataset(handle)

        assert float(raster.cell_area()[0, 0]) == pytest.approx(expected, rel=1e-3)


class TestWhatItCostsToAsk:
    """The view is the design; a guard that expands it is a defect."""

    @pytest.mark.parametrize(
        "epsg, geo, ceiling_kb",
        [
            (4326, (-180.0, 0.01, 0.0, 90.0, 0.0, -0.009), 4096),
            (32636, (0.0, 30.0, 0.0, 0.0, 0.0, -30.0), 64),
        ],
    )
    def test_a_huge_raster_costs_rows_not_pixels(self, epsg, geo, ceiling_kb):
        """A 20000x20000 grid must not allocate per pixel anywhere in the call.

        Args:
            epsg: The CRS, to take each branch in turn.
            geo: Its geotransform.
            ceiling_kb: Generous cap that still excludes a dense array.

        Test scenario:
            Validating `areas > 0.0` on the broadcast result rather than on the
            `rows` values behind it allocated a dense boolean -- 400 MB here,
            and a `MemoryError` at Copernicus GLO-30 sizes -- before returning
            a view whose whole purpose is to avoid exactly that. The projected
            branch is the sharper probe: its only real array is a 0-d scalar,
            so anything above a few kilobytes can only be the guard.
        """
        handle = gdal.GetDriverByName("MEM").Create("", 20000, 20000, 1, gdal.GDT_Byte)
        handle.SetGeoTransform(geo)
        handle.SetProjection(CRS.from_epsg(epsg).to_wkt())
        raster = Dataset(handle)

        tracemalloc.start()
        try:
            areas = raster.cell_area()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert areas.shape == (20000, 20000)
        assert peak < ceiling_kb * 1024
