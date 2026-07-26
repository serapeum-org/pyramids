"""Tests for Dataset.to_terrain_rgb (terrain-RGB encoding + XYZ tiling)."""

from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.engines.io import (
    IO,
    _encode_terrain_rgb,
    _overview_levels_for_tiling,
    _terrain_rgba_stack,
)

pytestmark = pytest.mark.core

# A north-up EPSG:3857 geotransform (30 m pixels) reused across the DEM fixtures.
_GEO_3857 = (0.0, 30.0, 0.0, 6000000.0, 0.0, -30.0)


def _dem_3857(no_data_value=None):
    """A small DEM already in EPSG:3857 (so no reprojection on encode)."""
    arr = np.array([[0.0, 100.0], [2000.0, 8848.0]], dtype="float32")
    return Dataset.create_from_array(
        arr=arr,
        geo=_GEO_3857,
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
            geo=_GEO_3857,
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
            arr=arr,
            geo=(10.0, 0.01, 0.0, 47.0, 0.0, -0.01),
            epsg=4326,
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
            arr=arr,
            geo=(10.0, 0.01, 0.0, 47.0, 0.0, -0.01),
            epsg=4326,
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
        dem = _dem_3857()
        out = tmp_path / "x.png"
        with pytest.raises(ValueError, match="encoding must be one of"):
            dem.to_terrain_rgb(out, encoding="rainbow")

    def test_invalid_resampling_raises(self, tmp_path):
        """An unknown resampling name raises ValueError."""
        dem = _dem_3857()
        out = tmp_path / "x.png"
        with pytest.raises(ValueError, match="does not exist|resampling"):
            dem.to_terrain_rgb(out, resampling="bogus")

    def test_max_zoom_below_min_zoom_raises(self, tmp_path):
        """``max_zoom < min_zoom`` raises ValueError."""
        dem = _dem_3857(no_data_value=None)
        out = tmp_path / "t"
        with pytest.raises(ValueError, match="max_zoom"):
            dem.to_terrain_rgb(out, tiles=True, min_zoom=8, max_zoom=4)

    def test_non_positive_interval_raises(self, tmp_path):
        """A non-positive mapbox ``interval`` raises instead of dividing by zero."""
        dem = _dem_3857()
        out = tmp_path / "x.png"
        with pytest.raises(ValueError, match="interval must be positive"):
            dem.to_terrain_rgb(out, tiles=False, interval=0.0)

    def test_negative_min_zoom_raises(self, tmp_path):
        """A negative ``min_zoom`` raises ValueError."""
        dem = _dem_3857(no_data_value=None)
        out = tmp_path / "t"
        with pytest.raises(ValueError, match="min_zoom must be >= 0"):
            dem.to_terrain_rgb(out, tiles=True, min_zoom=-1)

    def test_out_of_range_band_raises(self, tmp_path):
        """A band index past the source band count raises a clear ValueError."""
        dem = _dem_3857()
        out = tmp_path / "x.png"
        with pytest.raises(ValueError, match="is out of range for a"):
            dem.to_terrain_rgb(out, tiles=False, band=5)


class TestEncodeTerrainRgb:
    """Unit tests for the pure ``_encode_terrain_rgb`` packer."""

    def test_mapbox_known_value(self):
        """0 m with the default base/interval packs to the spec's R, G, B bytes."""
        rgb = _encode_terrain_rgb(
            np.array([[0.0]]), encoding="mapbox", base_val=-10000.0, interval=0.1
        )
        # v = (0 - -10000)/0.1 = 100000 -> R=1, G=134, B=160
        assert rgb.shape == (3, 1, 1), f"expected (3,1,1), got {rgb.shape}"
        assert rgb.dtype == np.uint8, f"expected uint8, got {rgb.dtype}"
        assert tuple(rgb[:, 0, 0]) == (1, 134, 160), f"wrong bytes: {rgb[:, 0, 0]}"

    @pytest.mark.parametrize(
        "elevation, expected",
        [(1e12, (255, 255, 255)), (-1e9, (0, 0, 0))],
    )
    def test_mapbox_clamps_out_of_range(self, elevation, expected):
        """Out-of-range elevations clamp to the encodable extremes, not wrap.

        Args:
            elevation: An elevation far outside the encodable window.
            expected: The clamped ``(R, G, B)`` byte triple.
        """
        rgb = _encode_terrain_rgb(
            np.array([[elevation]]), encoding="mapbox", base_val=-10000.0, interval=0.1
        )
        assert tuple(rgb[:, 0, 0]) == expected, f"clamp failed: {rgb[:, 0, 0]}"

    def test_terrarium_known_value(self):
        """0 m terrarium-encodes to (128, 0, 0) and decodes back to 0."""
        rgb = _encode_terrain_rgb(
            np.array([[0.0]]), encoding="terrarium", base_val=0.0, interval=1.0
        )
        r, g, b = (int(v) for v in rgb[:, 0, 0])
        assert (r, g, b) == (128, 0, 0), f"terrarium 0 m wrong: {(r, g, b)}"
        decoded = (r * 256 + g + b / 256.0) - 32768
        assert abs(decoded) < 1e-9, f"terrarium decode of 0 m must be 0, got {decoded}"

    def test_terrarium_fractional_metre(self):
        """The terrarium blue byte carries the sub-metre fraction (1/256 m)."""
        rgb = _encode_terrain_rgb(
            np.array([[0.5]]), encoding="terrarium", base_val=0.0, interval=1.0
        )
        b = int(rgb[2, 0, 0])
        assert b == 128, f"0.5 m must encode B=128 (0.5*256), got {b}"


class TestTerrainRgbaStack:
    """Unit tests for the ``_terrain_rgba_stack`` band-count / alpha logic."""

    def test_no_nodata_returns_three_bands(self):
        """Without a nodata value the stack is plain 3-band RGB."""
        stack = _terrain_rgba_stack(
            np.array([[100.0]]),
            None,
            encoding="mapbox",
            base_val=-10000.0,
            interval=0.1,
        )
        assert stack.shape[0] == 3, f"expected 3 bands, got {stack.shape[0]}"

    def test_nodata_adds_transparent_alpha(self):
        """A nodata cell yields a 4th alpha band that is 0 there and 255 elsewhere."""
        elev = np.array([[100.0, -9999.0]])
        stack = _terrain_rgba_stack(
            elev, -9999.0, encoding="mapbox", base_val=-10000.0, interval=0.1
        )
        assert stack.shape[0] == 4, f"expected RGBA, got {stack.shape[0]} bands"
        assert stack[3, 0, 0] == 255 and stack[3, 0, 1] == 0, (
            f"alpha must be 255 valid / 0 nodata, got {stack[3, 0]}"
        )


class TestTerrainTileMath:
    """Unit tests for the slippy-tile index and native-zoom helpers."""

    def test_zoom_zero_is_single_tile(self):
        """At zoom 0 the whole world is a single tile (0, 0)."""
        r = 20037508.34
        tiles = list(IO._terrain_tile_indices(0, -r, -r, r, r))
        assert tiles == [(0, 0)], f"zoom 0 must be one tile (0,0), got {tiles}"

    def test_indices_stay_in_range(self):
        """Every emitted tile index is within ``[0, 2**zoom)``."""
        tiles = list(IO._terrain_tile_indices(5, 0.0, 0.0, 1_000_000.0, 1_000_000.0))
        assert tiles, "a covered region must yield at least one tile"
        assert all(0 <= x < 32 and 0 <= y < 32 for x, y in tiles), (
            f"indices out of [0, 32) at zoom 5: {tiles}"
        )

    def test_native_zoom_floored_at_min_zoom(self):
        """A coarse pixel size would give a low zoom; ``min_zoom`` is the floor."""
        # huge pixel size -> computed zoom is small/negative -> min_zoom wins
        assert IO._native_terrain_zoom(1e7, 256, min_zoom=3) == 3, "min_zoom must floor"

    def test_native_zoom_matches_resolution(self):
        """A finer pixel size yields a higher zoom than a coarser one."""
        fine = IO._native_terrain_zoom(30.0, 256, 0)
        coarse = IO._native_terrain_zoom(1000.0, 256, 0)
        assert fine > coarse, f"finer pixels need a higher zoom: {fine} <= {coarse}"


class TestToTerrainRgbClamping:
    """End-to-end clamping through the public method."""

    def test_extreme_elevation_clamps_to_max(self, tmp_path):
        """An elevation above the encodable range writes the max RGB, not garbage."""
        arr = np.array([[1e9, 0.0]], dtype="float64").astype("float32")
        dem = Dataset.create_from_array(
            arr=arr,
            geo=_GEO_3857,
            epsg=3857,
            no_data_value=None,
        )
        out = dem.to_terrain_rgb(tmp_path / "c.png", tiles=False)
        _, (r, g, b) = _read_bands(out)
        assert (r[0, 0], g[0, 0], b[0, 0]) == (
            255,
            255,
            255,
        ), f"clamped cell must be (255,255,255), got {(r[0, 0], g[0, 0], b[0, 0])}"


class TestTilingSourcePreparation:
    """ARC-51: tiling reads a staged, pyramided source rather than the original."""

    @staticmethod
    def _capture_source(monkeypatch) -> list:
        """Record what `_terrain_rgb_tiles` is handed, without tiling.

        Inspects the dataset inside the spy rather than storing it: the staged
        source is closed and deleted by the time `to_terrain_rgb` returns, so a
        stored handle would be dead when the assertions run.

        Args:
            monkeypatch: pytest's monkeypatch fixture.

        Returns:
            list: filled with one `(overview_count, raster_handle)` tuple once
            `to_terrain_rgb(tiles=True)` runs.
        """
        seen: list = []

        def spy(self, source, path, **kwargs):
            seen.append(
                (source.raster.GetRasterBand(1).GetOverviewCount(), source.raster)
            )
            return path

        monkeypatch.setattr(IO, "_terrain_rgb_tiles", spy)
        return seen

    def test_a_reprojected_source_is_staged_with_overviews(self, tmp_path, monkeypatch):
        """A source needing reprojection reaches the tiler with a pyramid.

        Test scenario:
            `to_crs` hands back a warped VRT that re-warps from the original on
            every read, and a source without overviews makes every low-zoom tile
            resample full-resolution pixels. The staged GTiff carries the
            overviews the tiler needs.
        """
        seen = self._capture_source(monkeypatch)
        dem = Dataset.create_from_array(
            arr=np.linspace(0.0, 8848.0, 600 * 600, dtype="float32").reshape(600, 600),
            top_left_corner=(0.0, 6.0),
            cell_size=0.01,
            epsg=4326,
        )
        dem.to_terrain_rgb(tmp_path / "tiles", tiles=True, min_zoom=3, max_zoom=5)
        assert len(seen) == 1, f"expected one tiling call, got {len(seen)}"
        assert seen[0][0] > 0, "the staged tiling source must carry overviews"

    def test_a_source_already_in_3857_is_staged_too(self, tmp_path, monkeypatch):
        """The common pre-prepared case gets the pyramid as well.

        Test scenario:
            The preparation used to be gated on whether a reprojection happened,
            so a dataset already in EPSG:3857 -- the usual input for tiling --
            got neither the staging nor the overviews and kept paying
            full-resolution reads per tile.
        """
        seen = self._capture_source(monkeypatch)
        dem = Dataset.create_from_array(
            arr=np.linspace(0.0, 8848.0, 600 * 600, dtype="float32").reshape(600, 600),
            geo=_GEO_3857,
            epsg=3857,
        )
        dem.to_terrain_rgb(tmp_path / "tiles", tiles=True, min_zoom=3, max_zoom=5)
        assert len(seen) == 1, f"expected one tiling call, got {len(seen)}"
        assert seen[0][1] is not dem.raster, (
            "a source already in EPSG:3857 must still be staged, not read directly"
        )
        assert seen[0][0] > 0, (
            "a source already in EPSG:3857 must be staged with overviews too"
        )

    def test_the_untiled_path_does_not_stage_anything(self, tmp_path, monkeypatch):
        """`tiles=False` reads once, so it keeps the original source.

        Test scenario:
            The staging exists to amortise repeated reads across tiles and zoom
            levels. A single encode has nothing to amortise, so it must not pay
            for a GTiff copy -- the encoder receives the caller's own dataset
            handle, not a staged duplicate.
        """
        seen: list = []

        def spy(self, source, path, **kwargs):
            seen.append(source)
            return path

        monkeypatch.setattr(IO, "_terrain_rgb_single", spy)
        dem = _dem_3857()
        dem.to_terrain_rgb(tmp_path / "dem.png", tiles=False)
        assert len(seen) == 1, f"expected one encode call, got {len(seen)}"
        assert seen[0].raster is dem.raster, (
            "the untiled path must hand the encoder the original raster handle"
        )
        assert seen[0].raster.GetRasterBand(1).GetOverviewCount() == 0, (
            "the untiled path must not build overviews"
        )


class TestOverviewLevelsForTiling:
    """The pyramid depth follows the raster size, not a fixed ladder."""

    def test_a_raster_inside_one_tile_gets_no_levels(self):
        """Nothing to decimate when the whole raster fits in a tile.

        Test scenario:
            The fixed ladder built five levels regardless, the coarsest of them
            smaller than a single pixel of the output.
        """
        assert _overview_levels_for_tiling(200, 200, 256) == (), (
            "a raster smaller than one tile needs no pyramid"
        )

    def test_the_ladder_reaches_roughly_one_tile(self):
        """The coarsest level is about tile-sized, whatever the raster's size.

        Test scenario:
            A 4096-pixel raster tiled at 256 needs decimations up to 16x; the
            old fixed ladder stopped at 32x for every raster, so a continental
            source's lowest zoom levels still resampled a level far finer than
            they needed.
        """
        assert _overview_levels_for_tiling(4096, 4096, 256) == (2, 4, 8, 16), (
            f"got {_overview_levels_for_tiling(4096, 4096, 256)}"
        )
        deep = _overview_levels_for_tiling(65536, 65536, 256)
        assert deep[-1] == 256, f"a large raster must decimate further, got {deep}"

    def test_the_longer_edge_drives_the_depth(self):
        """A wide, short raster still gets levels for its width.

        Test scenario:
            Taking the shorter edge would stop the pyramid early and leave the
            long axis resampling full-resolution pixels.
        """
        assert _overview_levels_for_tiling(4096, 300, 256) == (2, 4, 8, 16), (
            f"got {_overview_levels_for_tiling(4096, 300, 256)}"
        )


class TestTilingScratchCleanup:
    """H3/M2: the staged source lives on disk and is removed on every path."""

    @staticmethod
    def _scratch_dirs() -> set[str]:
        """Temp directories this method has staged into.

        Returns:
            set[str]: full paths of every live staging directory.
        """
        root = Path(tempfile.gettempdir())
        return {str(p) for p in root.glob("pyramids_terrain_rgb_*")}

    @pytest.fixture(scope="function")
    def dem(self) -> Dataset:
        """A 600x600 EPSG:3857 DEM, big enough to warrant a pyramid."""
        return Dataset.create_from_array(
            arr=np.linspace(0.0, 8848.0, 600 * 600, dtype="float32").reshape(600, 600),
            geo=_GEO_3857,
            epsg=3857,
        )

    def test_the_staged_source_is_removed_after_a_successful_run(self, dem, tmp_path):
        """Tiling leaves no temp directory behind.

        Test scenario:
            The staging exists only for the duration of the tiling; keeping it
            would leak a full copy of the reprojected raster per call.
        """
        before = self._scratch_dirs()
        dem.to_terrain_rgb(tmp_path / "tiles", tiles=True, min_zoom=3, max_zoom=4)
        assert self._scratch_dirs() == before, (
            f"tiling left a staging directory behind: {self._scratch_dirs() - before}"
        )

    def test_a_failure_during_staging_is_cleaned_up(self, dem, tmp_path, monkeypatch):
        """A raise between the copy and the tiling still removes the scratch.

        Test scenario:
            The staging used to sit outside the try/finally that cleans it up,
            so an allocation failure in CreateCopy or an unsupported
            BuildOverviews stranded a copy of the whole raster for the lifetime
            of the process -- the same defect the orthorectify cleanup was filed
            for, one function over.
        """
        before = self._scratch_dirs()

        def explode(self, *args, **kwargs):
            raise RuntimeError("synthetic BuildOverviews failure")

        monkeypatch.setattr(gdal.Dataset, "BuildOverviews", explode)
        with pytest.raises(RuntimeError, match="synthetic BuildOverviews failure"):
            dem.to_terrain_rgb(tmp_path / "tiles", tiles=True, min_zoom=3, max_zoom=4)
        assert self._scratch_dirs() == before, (
            f"a failed staging left {self._scratch_dirs() - before} behind"
        )

    def test_the_source_is_staged_on_disk_not_in_memory(
        self, dem, tmp_path, monkeypatch
    ):
        """The intermediate is a real file, so a continental DEM streams.

        Test scenario:
            /vsimem and MEM are both process heap, so staging into either holds
            the whole reprojected raster in memory -- on the continental DEMs
            this method targets that is an allocation failure, not a slowdown.
            Captures the path the staged dataset was opened from.
        """
        seen: list = []

        def spy(self, source, path, **kwargs):
            seen.append(source.raster.GetDescription())
            return path

        monkeypatch.setattr(IO, "_terrain_rgb_tiles", spy)
        dem.to_terrain_rgb(tmp_path / "tiles", tiles=True, min_zoom=3, max_zoom=4)
        assert len(seen) == 1, f"expected one tiling call, got {len(seen)}"
        assert not seen[0].startswith("/vsimem"), (
            f"the staged source must not live in process memory, got {seen[0]!r}"
        )
        assert "pyramids_terrain_rgb_" in seen[0], (
            f"expected an on-disk staging path, got {seen[0]!r}"
        )
