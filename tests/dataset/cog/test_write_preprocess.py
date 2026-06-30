"""Tests for to_cog write-time pre-processing: indexes / out_dtype / nodata (PB-4).

Covers band subsetting + reorder, dtype rescale (with post-cast predictor
resolution), explicit NoData, and their combination — all routed through an
in-memory ``gdal.Translate`` before the COG write.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


@pytest.fixture
def multiband_float() -> Dataset:
    """A 64x64 4-band Float32 Dataset on EPSG:4326.

    Returns:
        Dataset: An in-memory multiband dataset; band i is filled with value i.
    """
    mem = gdal.GetDriverByName("MEM").Create("", 64, 64, 4, gdal.GDT_Float32)
    mem.SetGeoTransform(COG_GEOTRANSFORM)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    mem.SetProjection(sr.ExportToWkt())
    for i in range(4):
        mem.GetRasterBand(i + 1).WriteArray(np.full((64, 64), i, dtype="float32"))
    mem.FlushCache()
    return Dataset(mem)


def _open(path):
    """Open a raster with GDAL for assertions.

    Args:
        path: Path to a raster.

    Returns:
        gdal.Dataset: The opened dataset.
    """
    return gdal.Open(str(path))


class TestToCogPreprocess:
    """Tests for to_cog indexes / out_dtype / nodata."""

    def test_band_subset(self, multiband_float, tmp_path):
        """indexes selects a subset of bands in order.

        Args:
            multiband_float: 4-band fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            indexes=[0] keeps a single band; the COG has 1 band.
        """
        out = multiband_float.to_cog(tmp_path / "one.tif", indexes=[0])
        ds = _open(out)
        assert ds.RasterCount == 1, f"expected 1 band, got {ds.RasterCount}"
        ds = None

    def test_band_reorder(self, multiband_float, tmp_path):
        """indexes reorders bands (0-based, GDAL 1-based internally).

        Args:
            multiband_float: 4-band fixture (band i holds value i).
            tmp_path: pytest temp directory.

        Test scenario:
            indexes=[2, 0] yields a 2-band COG whose band 1 == old band 2 (value
            2) and band 2 == old band 0 (value 0).
        """
        out = multiband_float.to_cog(tmp_path / "reorder.tif", indexes=[2, 0])
        ds = _open(out)
        assert ds.RasterCount == 2, f"expected 2 bands, got {ds.RasterCount}"
        assert ds.GetRasterBand(1).ReadAsArray()[0, 0] == pytest.approx(2.0)
        assert ds.GetRasterBand(2).ReadAsArray()[0, 0] == pytest.approx(0.0)
        ds = None

    def test_out_dtype_cast_and_predictor(self, multiband_float, tmp_path):
        """out_dtype casts the output and re-resolves the predictor.

        Args:
            multiband_float: 4-band float fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            Casting float32 -> int16 yields an Int16 COG whose predictor is 2
            (the integer predictor), proving the predictor is resolved from the
            post-cast dtype, not the float source.
        """
        out = multiband_float.to_cog(
            tmp_path / "cast.tif", indexes=[0], out_dtype="int16"
        )
        ds = _open(out)
        assert (
            gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) == "Int16"
        ), "cast failed"
        pred = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE")
        ds = None
        assert pred == "2", f"post-cast int COG should have predictor 2, got {pred}"

    def test_nodata_set(self, multiband_float, tmp_path):
        """nodata sets the output NoData value.

        Args:
            multiband_float: 4-band float fixture (no NoData).
            tmp_path: pytest temp directory.

        Test scenario:
            nodata=-1 is written to band 1 of the COG.
        """
        out = multiband_float.to_cog(tmp_path / "nd.tif", indexes=[0], nodata=-1.0)
        ds = _open(out)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        assert nd == pytest.approx(-1.0), f"expected nodata -1, got {nd}"

    def test_no_preprocess_when_unset(self, multiband_float, tmp_path):
        """Without indexes/out_dtype/nodata all bands are written unchanged.

        Args:
            multiband_float: 4-band float fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            A plain to_cog keeps all 4 bands and the float predictor (3).
        """
        out = multiband_float.to_cog(tmp_path / "all.tif")
        ds = _open(out)
        assert ds.RasterCount == 4, f"expected 4 bands, got {ds.RasterCount}"
        pred = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE")
        ds = None
        assert pred == "3", f"float COG should keep predictor 3, got {pred}"

    def test_result_is_valid_cog(self, multiband_float, tmp_path):
        """A pre-processed write still produces a valid COG.

        Args:
            multiband_float: 4-band float fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            Subset + cast + nodata together yield a valid COG.
        """
        out = multiband_float.to_cog(
            tmp_path / "combo.tif", indexes=[1, 0], out_dtype="int16", nodata=0
        )
        assert Dataset.read_file(str(out)).validate_cog().is_valid, "combo COG invalid"
