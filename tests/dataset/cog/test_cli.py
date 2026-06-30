"""Tests for the pyramids cog CLI (PD-1).

Exercises the in-process entry point `pyramids.cli.main` for the
create / validate / info subcommands, including exit codes and printed output.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.cli import main
from pyramids.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


@pytest.fixture
def source_tif(tmp_path) -> str:
    """A plain on-disk GeoTIFF to feed the CLI.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the source raster.
    """
    arr = (np.random.default_rng(7).random((600, 600)) * 100).astype("float32")
    ds = Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)
    path = tmp_path / "src.tif"
    ds.to_file(path)
    return str(path)


@pytest.fixture
def cog_tif(tmp_path, source_tif) -> str:
    """A valid COG on disk to validate/inspect.

    Args:
        tmp_path: pytest temp directory.
        source_tif: Fixture source raster path.

    Returns:
        str: Path to the COG.
    """
    out = Dataset.read_file(source_tif).to_cog(tmp_path / "cog.tif")
    return str(out)


class TestCli:
    """Tests for the cog CLI subcommands."""

    def test_create_writes_valid_cog(self, source_tif, tmp_path, capsys):
        """`cog create` writes a COG and reports it valid (exit 0).

        Args:
            source_tif: Fixture source raster.
            tmp_path: pytest temp directory.
            capsys: pytest stdout/stderr capture.

        Test scenario:
            Creating with a profile produces a valid COG on disk.
        """
        out = tmp_path / "out.tif"
        code = main(["cog", "create", source_tif, str(out), "--profile", "deflate"])
        assert code == 0, "create should succeed"
        assert "valid COG" in capsys.readouterr().out, "should report validity"
        assert out.exists(), "output file should exist"

    def test_create_refuses_existing_without_overwrite(
        self, source_tif, tmp_path, capsys
    ):
        """`cog create` refuses to clobber an existing output unless --overwrite.

        Args:
            source_tif: Fixture source raster.
            tmp_path: pytest temp directory.
            capsys: pytest stdout/stderr capture.

        Test scenario:
            A second create to the same path exits 1 with an "already exists"
            message; passing --overwrite then succeeds.
        """
        out = tmp_path / "exists.tif"
        assert main(["cog", "create", source_tif, str(out)]) == 0
        capsys.readouterr()
        code = main(["cog", "create", source_tif, str(out)])
        err = capsys.readouterr().err
        assert code == 1, "a second create without --overwrite must be refused"
        assert "already exists" in err, f"error should name the existing file: {err}"
        assert main(["cog", "create", source_tif, str(out), "--overwrite"]) == 0

    def test_validate_valid_returns_0(self, cog_tif, capsys):
        """`cog validate` returns 0 for a valid COG.

        Args:
            cog_tif: Fixture COG path.
            capsys: pytest capture.

        Test scenario:
            A valid COG exits 0 and prints 'valid COG'.
        """
        code = main(["cog", "validate", cog_tif])
        assert code == 0, "valid COG should exit 0"
        assert "valid COG" in capsys.readouterr().out, "should print validity"

    def test_validate_invalid_returns_1(self, source_tif, capsys):
        """`cog validate` returns 1 for a non-COG (large stripped GeoTIFF).

        Args:
            source_tif: Fixture plain stripped raster.
            capsys: pytest capture.

        Test scenario:
            A 600x600 stripped GeoTIFF is not a valid COG -> exit 1.
        """
        # ensure it is genuinely not a COG by attaching external overviews
        ds = gdal.Open(source_tif, gdal.GA_ReadOnly)
        ds.BuildOverviews("NEAREST", [2])
        ds = None
        code = main(["cog", "validate", source_tif])
        assert code == 1, "non-COG should exit 1"

    def test_info_prints_fields(self, cog_tif, capsys):
        """`cog info` prints the structured metadata and exits 0.

        Args:
            cog_tif: Fixture COG path.
            capsys: pytest capture.

        Test scenario:
            The output contains the compression and overview lines.
        """
        code = main(["cog", "info", cog_tif])
        out = capsys.readouterr().out
        assert code == 0, "info should exit 0"
        assert "compression:" in out and "overviews:" in out, f"missing fields:\n{out}"
