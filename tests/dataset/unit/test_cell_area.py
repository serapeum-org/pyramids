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

import numpy as np
import pytest
from osgeo import gdal
from pyproj import CRS

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.core

GEOGRAPHIC = GeoReference(top_left_corner=(-180.0, 90.0), cell_size=1.0, epsg=4326)
WGS84_ELLIPSOID_KM2 = 510_065_622.0


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
        """Test scenario: only a geographic grid varies with latitude."""
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
        raster = Dataset.from_array(np.ones((5, 5), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area()

        assert len(np.unique(areas)) == 1


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
        assert equator > high_latitude * 5

    def test_it_matches_an_independent_geodesic_computation(self):
        """Pinned against pyproj directly, not against a remembered constant.

        Test scenario:
            The implementation asks `Geod` once per row; this asks it again,
            here, cell by cell for a handful of rows. They must agree exactly,
            which is what rules out an off-by-one in the row edges.
        """
        areas = _global_grid().cell_area()
        geod = CRS.from_epsg(4326).get_geod()

        for row in (0, 45, 90, 179):
            top = 90.0 - row
            expected, _ = geod.polygon_area_perimeter(
                [0.0, 1.0, 1.0, 0.0], [top, top, top - 1.0, top - 1.0]
            )
            assert float(areas[row, 0]) == pytest.approx(abs(expected), rel=1e-12)

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
        assert areas.base is not None
        assert areas.base.size <= 180


class TestTheUnits:
    """`m2`, `km2` and `ha`, and a clear refusal for anything else."""

    @pytest.mark.parametrize(
        ("unit", "divisor"), [("m2", 1.0), ("km2", 1e6), ("ha", 1e4)]
    )
    def test_each_unit_scales_the_same_answer(self, unit: str, divisor: float):
        """Args: unit: The requested unit. divisor: Square metres in it.

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

        with pytest.raises(ValueError, match="declares no CRS"):
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

        assert naive / weighted > 3.5

    def test_a_fully_masked_band_has_no_area(self):
        """Test scenario: no valid cell, no ground -- and no division by zero."""
        values = np.full((10, 10), -9999.0, dtype="float32")
        raster = Dataset.from_array(values, geo_ref=GEOGRAPHIC, no_data_value=-9999.0)

        assert raster.domain_area() == 0.0

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
        inside = ~np.isclose(np.asarray(raster.read_array()), -9999.0)
        expected = float((areas * inside).sum())

        assert raster.domain_area() == pytest.approx(expected, rel=1e-9)

    def test_a_projected_domain_is_the_count_times_the_cell(self):
        """Test scenario: with constant cells the weighted answer is the simple one."""
        geo_ref = GeoReference(top_left_corner=(0.0, 0.0), cell_size=30.0, epsg=32636)
        values = np.ones((6, 6), dtype="float32")
        values[0, :] = -9999.0
        raster = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=-9999.0)

        assert raster.domain_area() == pytest.approx(raster.count_domain_cells() * 900.0)


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

        assert raster.domain_area(band=1) < raster.domain_area(band=0)

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
        geo_ref = GeoReference(geo=(0.0, 1.0, 0.0, -10.0, 0.0, 1.0), epsg=4326)
        raster = Dataset.from_array(np.ones((4, 4), "float32"), geo_ref=geo_ref)

        areas = raster.cell_area()

        assert np.all(np.isfinite(areas))
        assert np.all(areas > 0.0)
