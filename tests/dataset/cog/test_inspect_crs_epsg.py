"""`cog_info` reports the EPSG code a CRS declares, or nothing.

`COGInfo.crs_epsg` read the authority code straight off the spatial reference.
That returns a number for any authority, so an ESRI-authority CRS reported its
ESRI number as though it were an EPSG code (#965), and an `OGC:CRS84` raster
raised outright. It now goes through `epsg_of_crs`, the package's answer to that
question everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.cog.inspect import cog_info

pytestmark = pytest.mark.core


def write_cog(path: Path, srs_setter) -> str:
    """A small COG whose CRS is stamped by `srs_setter`."""
    source = str(path / "src.tif")
    Dataset.from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0, epsg=4326),
    ).to_file(source)

    handle = gdal.Open(source, gdal.GA_Update)
    srs = osr.SpatialReference()
    srs_setter(srs)
    handle.SetSpatialRef(srs)
    handle = None

    destination = str(path / "out.tif")
    gdal.GetDriverByName("COG").CreateCopy(destination, gdal.Open(source))
    return destination


class TestCrsEpsg:
    """What `crs_epsg` reports for each kind of CRS."""

    def test_an_epsg_crs_reports_its_code(self, tmp_path: Path):
        """The ordinary case is unchanged."""
        path = write_cog(tmp_path, lambda srs: srs.ImportFromEPSG(4326))

        assert cog_info(path).crs_epsg == 4326

    def test_an_esri_authority_crs_reports_no_epsg_code(self, tmp_path: Path):
        """An ESRI code is not an EPSG code, so none is reported (#965).

        Reading the authority code directly returned `102008` here, which is a
        valid-looking integer belonging to a different authority.
        """
        path = write_cog(tmp_path, lambda srs: srs.SetFromUserInput("ESRI:102008"))

        assert cog_info(path).crs_epsg is None

    def test_ogc_crs84_normalises_to_4326_through_the_writer(self, tmp_path: Path):
        """`OGC:CRS84` is not preserved as such, so it cannot reach the reader.

        The old code would have raised `int('CRS84')` on a CRS that kept that
        authority, but GDAL's COG writer rewrites CRS84 to EPSG:4326, so no
        file-backed raster exercises that branch. Pinned so the assumption is
        visible rather than assumed.
        """
        path = write_cog(tmp_path, lambda srs: srs.SetFromUserInput("OGC:CRS84"))

        assert cog_info(path).crs_epsg == 4326

    def test_a_raster_without_a_crs_reports_none(self, tmp_path: Path):
        """No CRS means no code, not a fabricated one."""
        source = str(tmp_path / "bare.tif")
        driver = gdal.GetDriverByName("GTiff")
        handle = driver.Create(source, 8, 8, 1, gdal.GDT_Float32)
        handle.SetGeoTransform((0.0, 1.0, 0.0, 8.0, 0.0, -1.0))
        handle = None
        destination = str(tmp_path / "bare_cog.tif")
        gdal.GetDriverByName("COG").CreateCopy(destination, gdal.Open(source))

        assert cog_info(destination).crs_epsg is None
