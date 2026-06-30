"""Tests for web-optimized (TileMatrixSet) COG output (PC-5).

pyramids forwards tiling_scheme / zoom_level / zoom_level_strategy / aligned_levels
to the GDAL COG driver — the GDAL-native equivalent of rio-cogeo's
web_optimized=True. These verify the output SRS and overview alignment, and the
documented mutual exclusion with target_srs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core

# lon 0..6, lat 4..10 — well within the Web-Mercator valid latitude band.

_POWERS_OF_TWO = {2**k for k in range(1, 16)}


@pytest.fixture
def big_dataset() -> Dataset:
    """A 600x600 Float32 EPSG:4326 Dataset (large enough for overviews).

    Returns:
        Dataset: An in-memory float32 dataset.
    """
    arr = (np.random.default_rng(6).random((600, 600)) * 100).astype("float32")
    return Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)


class TestWebOptimizedCog:
    """Tests for tiling_scheme=GoogleMapsCompatible output."""

    def test_output_is_epsg_3857(self, big_dataset, tmp_path):
        """A GoogleMapsCompatible COG is reprojected to EPSG:3857.

        Args:
            big_dataset: 600x600 EPSG:4326 fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            The reopened web-optimized COG reports EPSG:3857.
        """
        out = big_dataset.to_cog(
            tmp_path / "web.tif", tiling_scheme="GoogleMapsCompatible"
        )
        reopened = Dataset.read_file(str(out))
        assert reopened.epsg == 3857, f"expected EPSG:3857, got {reopened.epsg}"

    def test_overviews_power_of_two_aligned(self, big_dataset, tmp_path):
        """A web-optimized COG has power-of-two overview decimations.

        Args:
            big_dataset: 600x600 EPSG:4326 fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            Every overview level decimates by a power of two (zoom-aligned).
        """
        out = big_dataset.to_cog(
            tmp_path / "web.tif", tiling_scheme="GoogleMapsCompatible"
        )
        info = Dataset.read_file(str(out)).cog_info()
        assert info.overviews, "web-optimized COG should carry overviews"
        for ovr in info.overviews:
            assert (
                ovr.decimation in _POWERS_OF_TWO
            ), f"overview decimation {ovr.decimation} is not a power of two"

    def test_result_is_valid_cog(self, big_dataset, tmp_path):
        """A web-optimized write still validates as a COG.

        Args:
            big_dataset: 600x600 EPSG:4326 fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            The GoogleMapsCompatible output passes validate_cog.
        """
        out = big_dataset.to_cog(
            tmp_path / "web.tif", tiling_scheme="GoogleMapsCompatible"
        )
        assert Dataset.read_file(str(out)).validate_cog().is_valid, "invalid web COG"

    def test_tiling_scheme_and_target_srs_warns(self, big_dataset, tmp_path):
        """Passing both tiling_scheme and target_srs warns (scheme wins).

        Args:
            big_dataset: 600x600 EPSG:4326 fixture.
            tmp_path: pytest temp directory.

        Test scenario:
            The two are mutually exclusive; tiling_scheme wins and a UserWarning
            is emitted.
        """
        with pytest.warns(UserWarning, match="tiling_scheme"):
            big_dataset.to_cog(
                tmp_path / "w.tif",
                tiling_scheme="GoogleMapsCompatible",
                target_srs=4326,
            )
