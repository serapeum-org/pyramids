"""Tests for Dataset.to_terrain_rgb (terrain-RGB encoding + XYZ tiling)."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _dem_3857(no_data_value=None):
    """A small DEM already in EPSG:3857 (so no reprojection on encode)."""
    arr = np.array([[0.0, 100.0], [2000.0, 8848.0]], dtype="float32")
    return Dataset.create_from_array(
        arr=arr,
        geo=(0.0, 30.0, 0.0, 6000000.0, 0.0, -30.0),
        epsg=3857,
        no_data_value=no_data_value,
    )


def _decode_mapbox(r, g, b, base_val=-10000.0, interval=0.1):
    """Mapbox Terrain-RGB decoder (inverse of the encoder)."""
    return base_val + (r * 65536 + g * 256 + b) * interval


def _read_bands(path):
    """Return the per-band int arrays of a raster on disk."""
    ds = gdal.Open(str(path))
    bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)]
    return ds.RasterCount, bands


class TestToTerrainRgbRoundtrip:
    """Decoding a written raster recovers the source elevation."""

    def test_mapbox_roundtrip_within_interval(self, tmp_path):
        """Mapbox decode recovers each height to within one ``interval``."""
        dem = _dem_3857()
        out = dem.to_terrain_rgb(tmp_path / "dem.png", tiles=False, encoding="mapbox")
        count, (r, g, b) = _read_bands(out)
        assert count == 3, f"no-nodata source must yield 3-band RGB, got {count}"
        decoded = _decode_mapbox(
            r.astype("int64"), g.astype("int64"), b.astype("int64")
        )
        source = np.array([[0.0, 100.0], [2000.0, 8848.0]])
        assert np.max(np.abs(decoded - source)) <= 0.1, (
            f"decode must be within 0.1 m, got {np.abs(decoded - source)}"
        )

    def test_terrarium_roundtrip_within_quantum(self, tmp_path):
        """Terrarium decode recovers each height to within 1/256 m."""
        dem = _dem_3857()
        out = dem.to_terrain_rgb(tmp_path / "t.png", tiles=False, encoding="terrarium")
        _, (r, g, b) = _read_bands(out)
        decoded = (r.astype("int64") * 256 + g + b / 256.0) - 32768
        source = np.array([[0.0, 100.0], [2000.0, 8848.0]])
        assert np.max(np.abs(decoded - source)) <= 1 / 256, "terrarium within 1/256 m"

    def test_encoding_is_case_insensitive(self, tmp_path):
        """``encoding`` is normalised, so upper-case names are accepted."""
        out = _dem_3857().to_terrain_rgb(
            tmp_path / "u.png", tiles=False, encoding="MapBox"
        )
        assert out.exists(), "upper-case encoding name must be accepted"


class TestToTerrainRgbNoData:
    """No-data handling: transparent alpha where the source is masked."""

    def test_nodata_becomes_transparent(self, tmp_path):
        """A nodata cell becomes alpha 0; valid cells alpha 255 (RGBA output)."""
        arr = np.array([[0.0, -9999.0], [2000.0, 8848.0]], dtype="float32")
        dem = Dataset.create_from_array(
            arr=arr,
            geo=(0.0, 30.0, 0.0, 6000000.0, 0.0, -30.0),
            epsg=3857,
            no_data_value=-9999.0,
        )
        out = dem.to_terrain_rgb(tmp_path / "n.png", tiles=False)
        count, bands = _read_bands(out)
        assert count == 4, f"nodata source must yield 4-band RGBA, got {count}"
        alpha = bands[3]
        assert alpha[0, 1] == 0, "nodata cell must be transparent"
        assert alpha[0, 0] == 255, "valid cell must be opaque"

    def test_no_nodata_yields_rgb(self, tmp_path):
        """A source without a nodata value yields a 3-band RGB raster."""
        out = _dem_3857(no_data_value=None).to_terrain_rgb(
            tmp_path / "r.png", tiles=False
        )
        count, _ = _read_bands(out)
        assert count == 3, f"expected 3-band RGB without nodata, got {count}"

    def test_reproject_preserves_nodata_transparency(self, tmp_path):
        """A 4326 source with nodata stays transparent after the warp to 3857."""
        arr = np.array([[100.0, -9999.0], [2000.0, 3000.0]], dtype="float32")
        dem = Dataset.create_from_array(
            arr=arr, geo=(10.0, 0.01, 0.0, 47.0, 0.0, -0.01), epsg=4326,
            no_data_value=-9999.0,
        )
        out = dem.to_terrain_rgb(tmp_path / "rn.png", tiles=False)
        count, bands = _read_bands(out)
        assert count == 4, f"reprojected nodata source must be RGBA, got {count}"
        assert (bands[3] == 0).any(), "nodata cell must survive reprojection as alpha 0"


class TestToTerrainRgbOutputs:
    """File-format and reprojection behaviour."""

    def test_geotiff_when_not_png(self, tmp_path):
        """A non-``.png`` suffix writes a GeoTIFF carrying the 3857 projection."""
        out = _dem_3857().to_terrain_rgb(tmp_path / "dem.tif", tiles=False)
        ds = gdal.Open(str(out))
        assert ds.GetDriver().ShortName == "GTiff", "non-png suffix must write GeoTIFF"
        assert "3857" in ds.GetProjection(), "output must be Web Mercator"

    def test_reprojects_4326_source(self, tmp_path):
        """A non-3857 source is reprojected, not rejected."""
        arr = np.ones((4, 4), dtype="float32") * 500.0
        dem = Dataset.create_from_array(
            arr=arr, geo=(10.0, 0.01, 0.0, 47.0, 0.0, -0.01), epsg=4326,
            no_data_value=None,
        )
        out = dem.to_terrain_rgb(tmp_path / "w.tif", tiles=False)
        assert "3857" in gdal.Open(str(out)).GetProjection(), "must reproject to 3857"


class TestToTerrainRgbTiles:
    """XYZ ``{z}/{x}/{y}.png`` pyramid output."""

    def test_writes_xyz_layout(self, tmp_path):
        """``tiles=True`` writes valid 256x256 RGBA PNGs under ``{z}/{x}/{y}.png``."""
        root = _dem_3857(no_data_value=None).to_terrain_rgb(
            tmp_path / "tiles", tiles=True, min_zoom=4, max_zoom=6
        )
        pngs = glob.glob(os.path.join(str(root), "**", "*.png"), recursive=True)
        assert pngs, "at least one tile must be written"
        for png in pngs:
            rel = os.path.relpath(png, str(root)).replace(os.sep, "/")
            z, x, name = rel.split("/")
            assert name.endswith(".png") and z.isdigit() and x.isdigit(), (
                f"tile path must be z/x/y.png, got {rel}"
            )
            tile = gdal.Open(png)
            assert (tile.RasterXSize, tile.RasterYSize) == (256, 256), "256x256 tiles"

    def test_zoom_levels_span_min_to_max(self, tmp_path):
        """Every requested zoom level produces a directory."""
        root = _dem_3857(no_data_value=None).to_terrain_rgb(
            tmp_path / "z", tiles=True, min_zoom=3, max_zoom=5
        )
        zooms = {int(d) for d in os.listdir(root) if d.isdigit()}
        assert {3, 4, 5} <= zooms, f"zooms 3-5 must be written, got {zooms}"


class TestToTerrainRgbErrors:
    """Input validation."""

    def test_invalid_encoding_raises(self, tmp_path):
        """An unknown encoding name raises ValueError."""
        with pytest.raises(ValueError, match="encoding must be one of"):
            _dem_3857().to_terrain_rgb(tmp_path / "x.png", encoding="rainbow")

    def test_invalid_resampling_raises(self, tmp_path):
        """An unknown resampling name raises ValueError."""
        with pytest.raises(ValueError, match="does not exist|resampling"):
            _dem_3857().to_terrain_rgb(tmp_path / "x.png", resampling="bogus")

    def test_max_zoom_below_min_zoom_raises(self, tmp_path):
        """``max_zoom < min_zoom`` raises ValueError."""
        with pytest.raises(ValueError, match="max_zoom"):
            _dem_3857(no_data_value=None).to_terrain_rgb(
                tmp_path / "t", tiles=True, min_zoom=8, max_zoom=4
            )

    def test_non_positive_interval_raises(self, tmp_path):
        """A non-positive mapbox ``interval`` raises instead of dividing by zero."""
        with pytest.raises(ValueError, match="interval must be positive"):
            _dem_3857().to_terrain_rgb(tmp_path / "x.png", tiles=False, interval=0.0)

    def test_negative_min_zoom_raises(self, tmp_path):
        """A negative ``min_zoom`` raises ValueError."""
        with pytest.raises(ValueError, match="min_zoom must be >= 0"):
            _dem_3857(no_data_value=None).to_terrain_rgb(
                tmp_path / "t", tiles=True, min_zoom=-1
            )


class TestToTerrainRgbExposure:
    """Facade wiring."""

    def test_exposed_on_dataset(self):
        """``to_terrain_rgb`` is a callable method on the Dataset facade."""
        assert callable(getattr(Dataset, "to_terrain_rgb", None)), (
            "Dataset must expose to_terrain_rgb"
        )
