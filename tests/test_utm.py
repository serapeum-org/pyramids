"""Tests for the public UTM helpers in `pyramids.utm`."""

from __future__ import annotations

import geopandas as gpd
import pytest
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import box

from pyramids.base._errors import CRSError
from pyramids.dataset._stac import _utm_epsg
from pyramids.utm import project_to_utm, utm_epsg, utm_epsg_for_polygon, utm_zone


def _pyproj_utm_epsg(lon: float, lat: float, d: float = 0.01) -> int:
    """Reference UTM EPSG for a point via pyproj's own query, as an oracle."""
    infos = query_utm_crs_info(
        "WGS 84", AreaOfInterest(lon - d, lat - d, lon + d, lat + d)
    )
    return int(infos[0].code)


class TestUtmZone:
    """The lon-only UTM zone number."""

    @pytest.mark.parametrize(
        "lon, expected",
        [
            (0.0, 31),  # zone 31 spans 0-6E (western boundary inclusive)
            (5.0, 31),  # Bergen longitude — plain band, not the MGRS zone-32 shift
            (11.0, 32),
            (-76.2, 18),
            (-180.0, 1),  # first zone
            (180.0, 60),  # clamped to the last zone
            (200.0, 60),  # out-of-range clamps to 60
            (-200.0, 1),  # out-of-range clamps to 1
        ],
    )
    def test_zone_number(self, lon, expected):
        """Longitude maps to the plain 6-degree UTM band, clamped to 1..60."""
        assert utm_zone(lon) == expected


class TestUtmEpsg:
    """The EPSG code for a point, and agreement with pyproj / EPSG."""

    @pytest.mark.parametrize(
        "lon, lat, expected",
        [
            (5.0, 60.0, 32631),  # Bergen, Norway (EPSG-correct, not MGRS 32632)
            (7.0, 78.0, 32632),  # Svalbard west (EPSG-correct, not MGRS 32631)
            (31.25, 30.05, 32636),  # Cairo
            (31.25, -25.0, 32736),  # southern hemisphere -> 327xx band
            (11.0, 46.0, 32632),  # Italian Alps
            (-58.0, -34.0, 32721),  # Buenos Aires
            (200.0, 10.0, 32660),  # out-of-range lon clamps to zone 60N
            (-200.0, -10.0, 32701),  # out-of-range lon clamps to zone 1S
        ],
    )
    def test_epsg_code(self, lon, lat, expected):
        """Each point resolves to its EPSG-correct UTM zone (lon clamped to 1..60)."""
        assert utm_epsg(lon, lat) == expected

    @pytest.mark.parametrize(
        "lon, lat",
        [(5.0, 60.0), (7.0, 78.0), (31.25, 30.05), (31.25, -25.0), (11.0, 46.0)],
    )
    def test_agrees_with_pyproj(self, lon, lat):
        """The helper matches pyproj's own UTM-CRS lookup for interior points."""
        assert utm_epsg(lon, lat) == _pyproj_utm_epsg(lon, lat)

    def test_no_norway_mgrs_exception(self):
        """Bergen stays in zone 31 (its EPSG area of use), not the MGRS zone 32."""
        assert utm_epsg(5.0, 60.0) == 32631
        assert utm_epsg(5.0, 60.0) != 32632

    def test_no_svalbard_mgrs_exception(self):
        """Svalbard-west stays in zone 32, not the MGRS zone 31."""
        assert utm_epsg(7.0, 78.0) == 32632
        assert utm_epsg(7.0, 78.0) != 32631


class TestStacDelegation:
    """The private STAC helper delegates to the public one, unchanged."""

    @pytest.mark.parametrize(
        "lon, lat", [(5.0, 60.0), (11.0, 46.0), (-58.0, -34.0), (0.0, 0.0)]
    )
    def test_stac_utm_epsg_delegates(self, lon, lat):
        """`_stac._utm_epsg` returns exactly `pyramids.utm.utm_epsg`."""
        assert _utm_epsg(lon, lat) == utm_epsg(lon, lat)


class TestPolygonHelpers:
    """UTM selection and reprojection for vector layers."""

    def test_epsg_for_polygon_4326(self):
        """A 4326 layer around Bergen selects UTM 31N from its bounds centre."""
        gdf = gpd.GeoDataFrame(geometry=[box(4.5, 59.5, 5.5, 60.5)], crs="EPSG:4326")
        assert utm_epsg_for_polygon(gdf) == 32631

    def test_epsg_for_polygon_reprojects_non_4326(self):
        """A non-4326 layer is reprojected to 4326 before the centroid lookup."""
        gdf = gpd.GeoDataFrame(
            geometry=[box(4.5, 59.5, 5.5, 60.5)], crs="EPSG:4326"
        ).to_crs(3857)
        assert utm_epsg_for_polygon(gdf) == 32631

    def test_epsg_for_polygon_without_crs_raises(self):
        """A layer with no CRS raises CRSError rather than guessing."""
        gdf = gpd.GeoDataFrame(geometry=[box(4.5, 59.5, 5.5, 60.5)])
        with pytest.raises(CRSError, match="no CRS"):
            utm_epsg_for_polygon(gdf)

    def test_epsg_for_polygon_empty_raises(self):
        """An empty (finite-bounds-less) layer raises a clear ValueError, not NaN."""
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="no finite bounds"):
            utm_epsg_for_polygon(gdf)

    def test_epsg_for_polygon_antimeridian_raises(self):
        """A layer whose bounds span >180° of longitude is rejected, not mis-zoned."""
        gdf = gpd.GeoDataFrame(
            geometry=[box(179.0, 0.0, 179.9, 1.0), box(-179.9, 0.0, -179.0, 1.0)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="no single UTM zone"):
            utm_epsg_for_polygon(gdf)

    def test_project_to_utm_returns_reprojected_and_epsg(self):
        """`project_to_utm` returns the layer in its UTM zone and that EPSG."""
        gdf = gpd.GeoDataFrame(geometry=[box(4.5, 59.5, 5.5, 60.5)], crs="EPSG:4326")
        projected, epsg = project_to_utm(gdf)
        assert epsg == 32631
        assert projected.crs.to_epsg() == 32631

    def test_project_to_utm_does_not_mutate_input(self):
        """`project_to_utm` leaves the input layer's CRS untouched."""
        gdf = gpd.GeoDataFrame(geometry=[box(4.5, 59.5, 5.5, 60.5)], crs="EPSG:4326")
        project_to_utm(gdf)
        assert gdf.crs.to_epsg() == 4326
