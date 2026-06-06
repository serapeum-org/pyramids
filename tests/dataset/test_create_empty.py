"""Tests for the out-of-core empty-raster allocators ``Dataset.create_empty`` / ``empty_like``.

These cover issue #470 task A1: header-only, disk-backed, tiled / sparse / BigTIFF
allocation that out-of-core algorithms fill window-by-window via
``write_array(window=)``.
"""

from __future__ import annotations

import subprocess
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.dataset import OUT_OF_CORE_CREATION_OPTIONS

pytestmark = pytest.mark.core


class TestCreateEmpty:
    """Tests for :meth:`pyramids.dataset.Dataset.create_empty`."""

    def test_mem_allocation_stamps_nodata_metadata(self):
        """An unwritten MEM raster has the configured shape and no-data metadata.

        Test scenario:
            ``create_empty(4, 5, driver_type="MEM")`` allocates a 1-band 4x5 raster
            whose cells are untouched. Shape and band count match the request and the
            no-data sentinel is stamped on the band metadata. (This asserts the
            *metadata* only — see ``test_mem_unwritten_cells_read_as_zero`` for the
            distinct cell-contents behaviour of the MEM driver.)
        """
        ds = Dataset.create_empty(
            4, 5, dtype="float32", no_data_value=-9999.0, driver_type="MEM"
        )
        assert (ds.rows, ds.columns, ds.band_count) == (4, 5, 1), (
            f"shape mismatch: {(ds.rows, ds.columns, ds.band_count)}"
        )
        assert ds.no_data_value[0] == -9999.0, (
            f"nodata not stamped, got {ds.no_data_value[0]}"
        )

    def test_mem_unwritten_cells_read_as_nodata(self):
        """MEM unwritten cells read back as the no-data sentinel, not 0.

        Test scenario:
            ``_build_dataset`` fills every band with the no-data value at allocation
            (``GDALRasterBand.Fill``), so a never-written MEM cell reads back as -9999,
            matching the sparse-GTiff behaviour in
            ``test_unwritten_block_reads_as_nodata``. This pins the cross-driver
            guarantee that unwritten cells are no-data, not 0.
        """
        ds = Dataset.create_empty(
            4, 4, dtype="float32", no_data_value=-9999.0, driver_type="MEM"
        )
        whole = ds.read_array()
        assert np.all(whole == -9999.0), (
            f"unwritten MEM cells should read as nodata -9999, got {np.unique(whole)}"
        )

    def test_window_write_read_roundtrip_mem(self):
        """A window scattered into an empty MEM raster reads back unchanged.

        Test scenario:
            Allocate empty, ``write_array(block, window=(1, 1, 2, 2))``, then read the
            same window — the values round-trip exactly.
        """
        ds = Dataset.create_empty(4, 4, dtype="float32", driver_type="MEM")
        block = np.arange(4, dtype="float32").reshape(2, 2)
        ds.write_array(block, window=(1, 1, 2, 2))
        back = ds.read_array(window=[1, 1, 2, 2])
        assert back.tolist() == block.tolist(), f"window did not round-trip: {back}"

    def test_window_write_read_roundtrip_gtiff(self, tmp_path: Path):
        """A window scattered into an empty disk GTiff round-trips after reopen.

        Test scenario:
            Allocate a disk-backed GTiff, write one window, drop the handle, reopen the
            file, and read the window back. Proves the disk path (not just MEM) works
            end-to-end.
        """
        path = tmp_path / "empty.tif"
        ds = Dataset.create_empty(64, 64, dtype="float32", epsg=3857, path=path)
        block = np.full((8, 8), 5.0, dtype="float32")
        ds.write_array(block, window=(0, 0, 8, 8))
        del ds
        reopened = Dataset.read_file(str(path))
        back = reopened.read_array(window=[0, 0, 8, 8])
        assert back.tolist() == block.tolist(), f"GTiff window did not round-trip: {back}"

    def test_geo_crs_nodata_preserved_on_reopen(self, tmp_path: Path):
        """Geotransform, CRS, and no-data survive a disk round-trip.

        Test scenario:
            Allocate with an explicit geotransform, EPSG, and no-data; reopen and
            confirm all three are preserved.
        """
        path = tmp_path / "geo.tif"
        geo = (100.0, 0.25, 0.0, 200.0, 0.0, -0.25)
        ds = Dataset.create_empty(
            32, 48, dtype="int16", geo=geo, epsg=32636, no_data_value=-1.0, path=path
        )
        del ds
        reopened = Dataset.read_file(str(path))
        assert reopened.geotransform == geo, (
            f"geotransform drift: {reopened.geotransform} != {geo}"
        )
        assert reopened.epsg == 32636, f"epsg drift: {reopened.epsg}"
        assert reopened.no_data_value[0] == -1.0, (
            f"nodata drift: {reopened.no_data_value[0]}"
        )

    def test_unwritten_block_reads_as_nodata(self, tmp_path: Path):
        """A never-written block of a sparse GTiff reads back as no-data, not 0.

        Test scenario:
            Allocate a sparse tiled GTiff larger than one block, write only the
            top-left window, then read a far block that was never touched. With
            SPARSE_OK + a stamped nodata, the unwritten block must read as the nodata
            sentinel — downstream code relies on this to distinguish "unwritten" from
            "zero".
        """
        path = tmp_path / "sparse.tif"
        ds = Dataset.create_empty(
            1024, 1024, dtype="float32", no_data_value=-9999.0, path=path
        )
        ds.write_array(np.ones((4, 4), dtype="float32"), window=(0, 0, 4, 4))
        del ds
        reopened = Dataset.read_file(str(path))
        far = reopened.read_array(window=[1000, 1000, 4, 4])
        assert np.all(far == -9999.0), (
            f"unwritten block should read as nodata -9999, got {np.unique(far)}"
        )

    def test_default_options_are_tiled_sparse_bigtiff(self, tmp_path: Path):
        """The default GTiff is tiled, sparse, and BigTIFF.

        Test scenario:
            Allocate with default options and inspect the written file via gdalinfo:
            the block size is 512x512 (TILED), and the file is BigTIFF. Confirms the
            out-of-core option set actually reaches GDAL.
        """
        path = tmp_path / "opts.tif"
        ds = Dataset.create_empty(600, 600, dtype="float32", path=path)
        del ds
        info = gdal.Info(str(path))
        assert "Block=512x512" in info, f"expected 512x512 blocks, gdalinfo:\n{info}"

    def test_custom_options_override_default(self, tmp_path: Path):
        """An explicit ``options`` list overrides the out-of-core defaults.

        Test scenario:
            Pass a 256x256 block size; gdalinfo reports 256x256, proving the override
            reaches GDAL instead of the 512 default.
        """
        path = tmp_path / "custom.tif"
        ds = Dataset.create_empty(
            512,
            512,
            dtype="float32",
            path=path,
            options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "SPARSE_OK=TRUE"],
        )
        del ds
        info = gdal.Info(str(path))
        assert "Block=256x256" in info, f"expected 256x256 blocks, gdalinfo:\n{info}"

    def test_default_geo_when_omitted(self):
        """Omitting ``geo`` yields a unit-pixel grid at the origin.

        Test scenario:
            ``create_empty`` with no geotransform defaults to
            ``(0, 1, 0, 0, 0, -1)`` so a caller that only cares about pixel coordinates
            gets a sane identity grid.
        """
        ds = Dataset.create_empty(3, 3, driver_type="MEM")
        assert ds.geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, -1.0), (
            f"default geo mismatch: {ds.geotransform}"
        )

    def test_sparse_allocation_is_small_on_disk(self, tmp_path: Path):
        """A large empty sparse GTiff costs almost no disk before any write.

        Test scenario:
            ``create_empty`` allocates a 10 000 x 10 000 float32 raster — 400 MB if
            materialised — but with SPARSE_OK and the no-data fill optimised away by
            GDAL (an all-no-data block is not allocated), the file on disk must stay
            tiny (header + metadata only). This is the headline out-of-core guarantee:
            never-written blocks cost no disk.
        """
        path = tmp_path / "sparse.tif"
        ds = Dataset.create_empty(10_000, 10_000, dtype="float32", path=path)
        del ds
        size = path.stat().st_size
        materialised = 10_000 * 10_000 * 4
        assert size < materialised // 100, (
            f"sparse GTiff is {size / 1e6:.2f} MB; a materialised raster would be "
            f"{materialised / 1e6:.0f} MB — SPARSE_OK is not in effect"
        )

    def test_gtiff_without_path_raises(self):
        """``create_empty`` with the default GTiff driver but no path raises ValueError.

        Test scenario:
            ``create_empty(rows, cols)`` defaults to ``driver_type="GTiff"`` but with no
            ``path`` the underlying driver would silently fall back to MEM and drop the
            tiled / sparse / BigTIFF options. The method must reject that combination
            loudly rather than hand back a surprising in-memory raster.
        """
        with pytest.raises(ValueError, match="needs a path"):
            Dataset.create_empty(4, 4)

    @pytest.mark.slow
    def test_bigtiff_past_4gb_ceiling(self, tmp_path: Path):
        """A logical raster past the 4 GB classic-TIFF ceiling allocates and writes.

        Test scenario:
            A 50 000 x 90 000 int8 raster is ~4.5 GB logically — past the classic-TIFF
            4 GB limit. With BIGTIFF=YES + SPARSE_OK the header allocates and a single
            far-corner window write succeeds, while on-disk footprint stays tiny
            (only the touched block is materialised). A classic TIFF would refuse the
            header.
        """
        path = tmp_path / "big.tif"
        ds = Dataset.create_empty(50_000, 90_000, dtype="int8", path=path)
        # write_array window is (row_off, col_off, n_rows, n_cols); the far corner
        # of a 50_000-row x 90_000-col raster.
        ds.write_array(np.ones((4, 4), dtype="int8"), window=(49_996, 89_996, 4, 4))
        del ds
        info = gdal.Info(str(path))
        assert "BigTIFF" in info or "BIGTIFF" in info.upper(), (
            f"file should be BigTIFF, gdalinfo:\n{info[:400]}"
        )
        reopened = Dataset.read_file(str(path))
        # read_array window is (col_off, row_off, n_cols, n_rows) — the opposite axis
        # order of write_array (it forwards straight to GDAL ReadAsArray(xoff, yoff,
        # xsize, ysize)). Mirror the write corner with the axes swapped.
        back = reopened.read_array(window=[89_996, 49_996, 4, 4])
        assert np.all(back == 1), f"far-corner write past 4GB failed: {np.unique(back)}"

    @pytest.mark.slow
    def test_allocation_allocates_no_full_python_buffer(self, tmp_path: Path):
        """Allocating a huge raster materialises no full-size NumPy buffer on the Python side.

        Test scenario:
            A 20 000 x 20 000 float32 dense array would need ~1.6 GB of RAM. The
            header-only ``create_empty`` must not build one — ``tracemalloc`` (which
            sees Python-level allocations only, not GDAL's C++ block cache) should show
            a peak far below the dense-array size. This guards against an accidental
            NumPy full-array materialisation in the Python code path; it does not bound
            GDAL's native allocations (see the review's L2 note).
        """
        path = tmp_path / "o1.tif"
        dense_bytes = 20_000 * 20_000 * 4
        tracemalloc.start()
        ds = Dataset.create_empty(20_000, 20_000, dtype="float32", path=path)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del ds
        assert peak < dense_bytes // 10, (
            f"create_empty peaked at {peak / 1e6:.1f} MB; a dense buffer would be "
            f"{dense_bytes / 1e6:.1f} MB — allocation is not header-only"
        )


class TestEmptyLike:
    """Tests for :meth:`pyramids.dataset.Dataset.empty_like`."""

    @pytest.fixture
    def template(self) -> Dataset:
        """A small 3-band float32 template with geo / epsg / nodata set.

        Returns:
            Dataset: 3 x 4 x 5 in-memory raster at EPSG:4326, nodata -9999.
        """
        return Dataset.create_from_array(
            np.ones((3, 4, 5), dtype="float32"),
            top_left_corner=(0.0, 10.0),
            cell_size=0.5,
            epsg=4326,
            no_data_value=-9999.0,
        )

    def test_copies_template_footprint(self, template: Dataset):
        """``empty_like`` reproduces the template's shape, geo, epsg, and nodata.

        Test scenario:
            With no overrides, the output matches every spatial property of the
            template and stamps the same nodata.
        """
        out = Dataset.empty_like(template)
        assert (out.rows, out.columns, out.band_count) == (4, 5, 3), (
            f"shape mismatch: {(out.rows, out.columns, out.band_count)}"
        )
        assert out.geotransform == template.geotransform, "geotransform drift"
        assert out.epsg == template.epsg, "epsg drift"
        assert out.no_data_value[0] == template.no_data_value[0], "nodata drift"

    def test_dtype_override(self, template: Dataset):
        """``dtype`` overrides the template's data type.

        Test scenario:
            Requesting ``int16`` produces an int16 raster while keeping every spatial
            property from the float32 template.
        """
        out = Dataset.empty_like(template, dtype="int16")
        assert out.dtype[0] == "int16", f"dtype not overridden: {out.dtype[0]}"
        assert (out.rows, out.columns) == (4, 5), "shape changed under dtype override"

    def test_bands_override(self, template: Dataset):
        """``bands`` overrides the template's band count.

        Test scenario:
            Requesting a single band from a 3-band template yields a 1-band output
            with the template's footprint.
        """
        out = Dataset.empty_like(template, bands=1)
        assert out.band_count == 1, f"band count not overridden: {out.band_count}"

    def test_nodata_override(self, template: Dataset):
        """An explicit ``no_data_value`` overrides the template's sentinel.

        Test scenario:
            Passing ``no_data_value=0`` stamps 0 instead of inheriting the template's
            -9999.
        """
        out = Dataset.empty_like(template, no_data_value=0)
        assert out.no_data_value[0] == 0, (
            f"nodata override ignored: {out.no_data_value[0]}"
        )

    def test_disk_backed_roundtrip(self, template: Dataset, tmp_path: Path):
        """A disk-backed ``empty_like`` writes, reopens, and round-trips a window.

        Test scenario:
            With a path, the output is a GTiff; a window written into it survives a
            reopen.
        """
        path = tmp_path / "like.tif"
        out = Dataset.empty_like(template, path=path)
        # template is 3-band, so target band 0 explicitly when writing a 2-D block.
        out.write_array(
            np.full((2, 2), 7.0, dtype="float32"), band=0, window=(0, 0, 2, 2)
        )
        del out
        reopened = Dataset.read_file(str(path))
        back = reopened.read_array(band=0, window=[0, 0, 2, 2])
        assert back.tolist() == [[7.0, 7.0], [7.0, 7.0]], (
            f"empty_like disk window did not round-trip: {back}"
        )


class TestCreateOptionsBackCompat:
    """The new ``options`` threading must not change existing factory output."""

    def test_create_from_array_unchanged_default_compression(self, tmp_path: Path):
        """``create_from_array`` to disk still uses LZW (the historical default).

        Test scenario:
            With no ``options`` passed anywhere, a disk-backed ``create_from_array``
            keeps the historical ``COMPRESS=LZW`` creation option — the new ``options``
            parameter defaults to None and must not perturb existing callers.
        """
        path = tmp_path / "compat.tif"
        Dataset.create_from_array(
            np.ones((4, 4), dtype="float32"),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            driver_type="GTiff",
            path=str(path),
        )
        info = gdal.Info(str(path))
        assert "COMPRESSION=LZW" in info.upper(), (
            f"existing create_from_array compression changed, gdalinfo:\n{info}"
        )

    def test_out_of_core_option_set_shape(self):
        """The module-level option set is the documented tiled / sparse / BigTIFF list.

        Test scenario:
            Guards the default option set against accidental edits — every out-of-core
            allocation depends on these exact keys being present.
        """
        joined = ";".join(OUT_OF_CORE_CREATION_OPTIONS)
        for key in ("TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES"):
            assert key in joined, f"{key} missing from OUT_OF_CORE_CREATION_OPTIONS"
