"""`_same_grid` must compare CRSes, not just their EPSG codes.

It tested `a.epsg == b.epsg`. `epsg` is `None` for any CRS with no EPSG
authority, so two *different* such CRSes both reported `None` and compared
equal. Two geostationary rasters at different sub-satellite longitudes were
therefore read as one grid, and `from_band_files` silently kept only the first
band instead of refusing to stack them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.dataset import _same_grid

pytestmark = pytest.mark.core


def geostationary_wkt(central_meridian: float) -> str:
    """A geostationary projection WKT, which carries no EPSG code."""
    return (
        f'PROJCS["geos_{central_meridian:g}",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Geostationary_Satellite"],'
        f'PARAMETER["central_meridian",{central_meridian}],'
        'PARAMETER["satellite_height",35785831],'
        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
        'UNIT["metre",1]]'
    )


def build(path: Path, wkt: str) -> Dataset:
    """A 4x4 raster on `wkt`, written to disk and reopened."""
    Dataset.from_array(
        np.ones((4, 4), dtype="float32"),
        geo_ref=GeoReference(geo=(0.0, 1000.0, 0.0, 4000.0, 0.0, -1000.0), epsg=wkt),
    ).to_file(str(path))
    return Dataset.read_file(str(path))


class TestSameGridComparesTheCrs:
    """Two CRSes without EPSG codes are not automatically the same CRS."""

    def test_two_geostationary_crses_are_not_one_grid(self, tmp_path: Path):
        """Different sub-satellite longitudes are different grids."""
        first = build(tmp_path / "a.tif", geostationary_wkt(0.0))
        second = build(tmp_path / "b.tif", geostationary_wkt(140.0))

        assert first.epsg == second.epsg, "premise: neither carries an EPSG code"
        assert _same_grid(first, second) is False

    def test_the_same_geostationary_crs_is_one_grid(self, tmp_path: Path):
        """The predicate stays true for genuinely identical grids."""
        first = build(tmp_path / "a.tif", geostationary_wkt(0.0))
        second = build(tmp_path / "b.tif", geostationary_wkt(0.0))

        assert _same_grid(first, second) is True

    def test_stacking_mismatched_geostationary_bands_is_refused(self, tmp_path: Path):
        """`from_band_files` raises instead of silently dropping bands."""
        build(tmp_path / "a.tif", geostationary_wkt(0.0))
        build(tmp_path / "b.tif", geostationary_wkt(140.0))

        with pytest.raises(AlignmentError):
            Dataset.from_band_files(
                [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")], align=False
            )


class TestGeotransformToleranceUnchanged:
    """The numeric half of the predicate keeps its tolerance."""

    def test_cell_sizes_differing_beyond_the_tolerance_are_different_grids(
        self, tmp_path: Path
    ):
        """1/120 and 0.008333 are not the same grid at rtol=1e-7.

        This is the guard that would catch a tolerance being widened while
        fixing the CRS half.
        """
        first = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(
                geo=(0.0, 1 / 120, 0.0, 4.0, 0.0, -1 / 120), epsg=4326
            ),
        )
        second = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(
                geo=(0.0, 0.008333, 0.0, 4.0, 0.0, -0.008333), epsg=4326
            ),
        )

        assert _same_grid(first, second) is False
