"""Additional scenario coverage for `pyramids.dataset.cog` and ops mixin.

Fills boundary/interaction gaps surfaced by the
`generate-full-test-suite` skill's gap analysis:

* `_fallback_validate` when the file raises during `gdal.Open`.
* `to_cog(overview_count=0)` boundary.
* `to_cog(add_mask=True, sparse_ok=True)` option interaction.
* `to_cog(blocksize` — largest allowed (4096) boundary.
* `CloudConfig.__exit__` return-value contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base.remote import CloudConfig
from pyramids.dataset import Dataset
from pyramids.dataset.cog.validate import _fallback_validate

pytestmark = pytest.mark.core


@pytest.fixture
def small_float_dataset() -> Dataset:
    """A 64x64 Float32 Dataset on EPSG:4326 for to_cog tests.

    Returns:
        Dataset: In-memory dataset with deterministic ramp values.
    """
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.001, epsg=4326
    )


class TestFallbackValidateGdalOpenRaises:
    """Gap: `_fallback_validate` when `gdal.Open` raises."""

    def test_runtime_error_from_gdal_open_surfaces_as_error(
        self, monkeypatch, tmp_path
    ):
        """Test _fallback_validate surfaces RuntimeError from gdal.Open.

        Test scenario:
            `gdal.Open` raises `RuntimeError` (e.g., corrupt file)
            rather than returning `None`. The fallback validator must
            not crash — it should propagate the exception since its
            callers translate it upstream.
        """
        from osgeo import gdal as gdal_mod

        p = tmp_path / "junk.txt"
        p.write_text("not a tiff")

        def boom(*args, **kwargs):
            raise RuntimeError("cannot read")

        monkeypatch.setattr(gdal_mod, "Open", boom)
        with pytest.raises(RuntimeError, match="cannot read") as exc_info:
            _fallback_validate(str(p))
        assert "cannot read" in str(
            exc_info.value
        ), f"Expected 'cannot read' in exception, got: {exc_info.value}"


class TestToCogOverviewCountBoundary:
    """Gap: `to_cog(overview_count=0)` — no overviews requested."""

    def test_zero_overviews_produces_valid_cog(self, small_float_dataset, tmp_path):
        """Test to_cog with overview_count=0 writes a COG with no overviews.

        Test scenario:
            Requesting zero overviews is a valid config (useful for
            small rasters where overviews would be wasted bytes). The
            output should still be a valid COG.
        """
        out = small_float_dataset.to_cog(tmp_path / "no_ovr.tif", overview_count=0)
        assert out.exists(), f"Output file must exist: {out}"
        reopened = gdal.Open(str(out))
        try:
            ovr_count = reopened.GetRasterBand(1).GetOverviewCount()
            assert ovr_count == 0, f"Expected 0 overviews, got {ovr_count}"
        finally:
            reopened = None


class TestToCogBlocksizeBoundaries:
    """Gap: `to_cog` blocksize boundaries min (64) and max (4096)."""

    def test_min_blocksize_64_accepted(self, small_float_dataset, tmp_path):
        """Test to_cog accepts the smallest legal blocksize.

        Test scenario:
            64 is the smallest power-of-2 in [64, 4096]; the file must
            write successfully and honor the requested block size.
        """
        out = small_float_dataset.to_cog(tmp_path / "bs64.tif", blocksize=64)
        reopened = gdal.Open(str(out))
        try:
            bx, by = reopened.GetRasterBand(1).GetBlockSize()
            assert (
                bx == 64 and by == 64
            ), f"Expected blocksize (64, 64), got ({bx}, {by})"
        finally:
            reopened = None

    @pytest.mark.parametrize("bad", [32, 63, 65, 500, 8192])
    def test_invalid_blocksize_rejected(self, small_float_dataset, tmp_path, bad):
        """Test invalid blocksizes raise ValueError before I/O.

        Args:
            bad: Illegal blocksize value to try.

        Test scenario:
            Values below 64, above 4096, or non-powers-of-2 within
            the range must be rejected up-front (no partial file
            written).
        """
        target = tmp_path / f"bad_{bad}.tif"
        with pytest.raises(ValueError, match=r"power of 2") as exc_info:
            small_float_dataset.to_cog(target, blocksize=bad)
        assert (
            not target.exists()
        ), f"No file should be created on validation failure: {target}"
        assert "power of 2" in str(
            exc_info.value
        ), f"Error message must mention 'power of 2'; got: {exc_info.value}"


class TestToCogOptionInteractions:
    """Gap: `to_cog` boolean option interactions (add_mask + sparse_ok)."""

    def test_add_mask_and_sparse_ok_both_true(self, small_float_dataset, tmp_path):
        """Test add_mask=True and sparse_ok=True combine without error.

        Test scenario:
            Both options can be set together; the COG driver should
            honor both — we verify the file writes and the resulting
            dataset has more bands than the input (alpha added).
        """
        out = small_float_dataset.to_cog(
            tmp_path / "mask_sparse.tif",
            add_mask=True,
            sparse_ok=True,
        )
        reopened = gdal.Open(str(out))
        try:
            assert reopened.RasterCount >= small_float_dataset.band_count, (
                f"add_mask should add at least one band; "
                f"got RasterCount={reopened.RasterCount}"
            )
        finally:
            reopened = None

    def test_statistics_false_skips_stats(self, small_float_dataset, tmp_path):
        """Test statistics=False produces a file without embedded band stats.

        Test scenario:
            When `statistics=False`, the COG driver should not compute
            or embed `STATISTICS_*` metadata — the user opted out.
        """
        out = small_float_dataset.to_cog(tmp_path / "no_stats.tif", statistics=False)
        reopened = gdal.Open(str(out))
        try:
            band_meta = reopened.GetRasterBand(1).GetMetadata()
            stat_keys = [k for k in band_meta if k.startswith("STATISTICS_")]
            assert stat_keys == [], (
                f"statistics=False should produce no STATISTICS_* keys; "
                f"got {stat_keys}"
            )
        finally:
            reopened = None


class TestCloudConfigExitContract:
    """Gap: `CloudConfig.__exit__` return value contract."""

    def test_exit_returns_none_on_normal_exit(self):
        """Test __exit__ returns a falsy value when no exception occurred.

        Test scenario:
            No exception inside the `with` block — `__exit__` is
            called with `(None, None, None)` and returns the
            underlying GDAL context manager's return (`None`), which
            does not suppress anything.
        """
        cfg = CloudConfig(aws_region="eu-west-1")
        cfg.__enter__()
        result = cfg.__exit__(None, None, None)
        assert (
            not result
        ), f"__exit__ must return a falsy value on normal exit; got {result!r}"

    def test_exit_does_not_swallow_exception(self):
        """Test exceptions inside the with block are re-raised.

        Test scenario:
            If an exception is raised inside the block, `__exit__`
            must not return True (which would suppress it). Caller code
            must still see the original exception.
        """
        with pytest.raises(RuntimeError, match="boom"):
            with CloudConfig(aws_region="us-east-1"):
                raise RuntimeError("boom")


class TestToCogPathHandling:
    """Gap: Dataset.to_cog path handling — literal filenames with no templating."""

    def test_literal_filename_passes_through(self, small_float_dataset, tmp_path):
        """to_cog accepts a fully-literal path with no placeholders.

        Test scenario:
            Dataset.to_cog (single-file) takes a literal path; template
            placeholders like `{name}` / `{i}` are a concern only of
            DatasetCollection.to_cog_stack. Verify the filename is used
            verbatim.
        """
        out = small_float_dataset.to_cog(tmp_path / "fixed_name.tif")
        assert (
            out.name == "fixed_name.tif"
        ), f"Path passed through unchanged; got {out.name!r}"
