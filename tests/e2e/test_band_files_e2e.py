"""End-to-end workflow tests for ``Dataset.from_band_files`` / ``stack_bands``.

Simulates the consumer pattern this helper exists for: a download (e.g. an
Earth Engine default ``getDownloadURL`` ZIP) yields one GeoTIFF per band on
disk; you stack them into a single multi-band raster, then crop / persist /
reload it.
"""

from __future__ import annotations

import os
import zipfile

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.merge import stack_bands

pytestmark = pytest.mark.core


def _write_band(
    directory,
    name,
    value,
    *,
    shape=(6, 8),
    cell_size=10.0,
    top_left=(500000.0, 4000000.0),
    epsg=32636,
    no_data_value=0,
):
    """Write one constant-valued single-band GeoTIFF and return its path."""
    path = os.path.join(str(directory), name)
    Dataset.create_from_array(
        np.full(shape, value, dtype="uint16"),
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=no_data_value,
        path=path,
    ).close()
    return path


class TestBandFilesDownloadStackPersist:
    """E2E: per-band tifs -> from_band_files -> crop -> save -> reload."""

    def test_per_band_tifs_to_multiband_file_then_crop(self, tmp_path):
        """Per-band tifs stack into a named multi-band raster; crop subsets it.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``<asset>.B2.tif`` / ``.B3.tif`` / ``.B4.tif`` -> ``from_band_files``
            (band names ``["B2", "B3", "B4"]``, values preserved) -> ``to_file``
            -> ``read_file`` (names + values survive) -> ``crop`` to a sub-box
            (fewer pixels, still 3 bands).
        """
        d = tmp_path / "download"
        d.mkdir()
        files = [
            _write_band(d, "S2_scene.B2.tif", 100),
            _write_band(d, "S2_scene.B3.tif", 200),
            _write_band(d, "S2_scene.B4.tif", 300),
        ]
        ds = Dataset.from_band_files(files)
        assert ds.band_names == ["B2", "B3", "B4"], f"band names: {ds.band_names}"
        assert [int(ds.read_array(band=i).flat[0]) for i in range(3)] == [
            100,
            200,
            300,
        ], "per-band values not carried into the stack"

        out = tmp_path / "stacked.tif"
        ds.to_file(str(out))
        assert out.exists(), "stacked multi-band raster was not written"
        reloaded = Dataset.read_file(str(out))
        assert reloaded.band_count == 3, f"expected 3 bands, got {reloaded.band_count}"
        assert reloaded.band_names == [
            "B2",
            "B3",
            "B4",
        ], f"band names lost on disk: {reloaded.band_names}"
        assert [int(reloaded.read_array(band=i).flat[0]) for i in range(3)] == [
            100,
            200,
            300,
        ], "per-band values changed on round-trip"

        minx, miny, maxx, maxy = reloaded.bbox
        sub = box(minx, (miny + maxy) / 2, (minx + maxx) / 2, maxy)
        mask = gpd.GeoDataFrame(geometry=[sub], crs=reloaded.epsg)
        cropped = reloaded.crop(mask)
        assert cropped.band_count == 3, f"crop changed band count: {cropped.band_count}"
        assert cropped.rows < reloaded.rows or cropped.columns < reloaded.columns, (
            "crop to a sub-box should shrink the raster"
        )

    def test_unzip_then_stack(self, tmp_path):
        """Mimic the GEE default download: a ZIP of per-band tifs -> one raster.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Build a ``.zip`` containing ``asset.B2.tif`` / ``.B3.tif`` /
            ``.B4.tif``, extract it, then ``stack_bands`` the members — expected:
            a 3-band dataset with the band names taken from the member names.
        """
        src = tmp_path / "src"
        src.mkdir()
        members = [
            _write_band(src, "asset.B2.tif", 1),
            _write_band(src, "asset.B3.tif", 2),
            _write_band(src, "asset.B4.tif", 3),
        ]
        zip_path = tmp_path / "download.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for m in members:
                zf.write(m, arcname=os.path.basename(m))

        extract_dir = tmp_path / "unzipped"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        extracted = sorted(str(extract_dir / n) for n in os.listdir(extract_dir))

        ds = stack_bands(extracted)
        assert ds.band_count == 3, f"expected 3 bands, got {ds.band_count}"
        assert ds.band_names == ["B2", "B3", "B4"], f"unexpected names: {ds.band_names}"

    def test_stacked_raster_feeds_a_collection(self, tmp_path):
        """Stacked per-bucket rasters compose into a time collection.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            For each of two time buckets, stack 2 band tifs and write the
            result; then ``DatasetCollection.from_files`` over the two stacked
            rasters — expected: a 2-timestep collection whose template has the
            stacked band count.
        """
        timestep_paths = []
        for t in range(2):
            d = tmp_path / f"t{t}"
            d.mkdir()
            files = [
                _write_band(d, f"t{t}.B2.tif", t),
                _write_band(d, f"t{t}.B3.tif", t + 10),
            ]
            out = tmp_path / f"stacked_t{t}.tif"
            Dataset.from_band_files(files, path=str(out))
            timestep_paths.append(str(out))

        collection = DatasetCollection.from_files(timestep_paths)
        assert collection.time_length == 2, (
            f"expected 2 timesteps, got {collection.time_length}"
        )
        assert collection.base.band_count == 2, "stacked template should have 2 bands"
