"""Regression test for the COG PREDICTOR value form (ARC-8).

The dtype-aware resolver emits numeric predictors (2 for integer, 3 for float).
This locks in that the GDAL COG driver accepts both the numeric forms and the
string aliases (STANDARD / FLOATING_POINT / YES) and that all round-trip to the
expected IMAGE_STRUCTURE PREDICTOR token — so no int->string normalisation is
needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def float_dataset() -> Dataset:
    """A 64x64 Float32 Dataset on EPSG:4326.

    Returns:
        Dataset: An in-memory float32 dataset.
    """
    arr = (np.random.default_rng(seed=2).random((64, 64)) * 100).astype("float32")
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


def _predictor_token(path) -> str:
    """Return the IMAGE_STRUCTURE PREDICTOR token of a raster.

    Args:
        path: Path to a raster.

    Returns:
        str: The predictor token (e.g. "2"/"3"), or "".
    """
    ds = gdal.Open(str(path))
    tok = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE") or ""
    ds = None
    return tok


class TestPredictorValueForm:
    """Tests that all accepted PREDICTOR forms round-trip on a float source."""

    @pytest.mark.parametrize(
        "predictor, expected_token",
        [
            (2, "2"),
            (3, "3"),
            ("STANDARD", "2"),
            ("FLOATING_POINT", "3"),
            ("YES", "3"),
        ],
    )
    def test_predictor_round_trips(
        self, float_dataset, tmp_path, predictor, expected_token
    ):
        """Each accepted predictor form writes a valid COG with the right token.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.
            predictor: The predictor value passed to to_cog.
            expected_token: The IMAGE_STRUCTURE PREDICTOR token expected on disk.

        Test scenario:
            Both numeric (2/3) and string (STANDARD/FLOATING_POINT/YES) forms are
            accepted by the COG driver and round-trip to the numeric token.
        """
        out = float_dataset.to_cog(tmp_path / f"p_{predictor}.tif", predictor=predictor)
        assert (
            _predictor_token(out) == expected_token
        ), f"predictor={predictor!r} should yield token {expected_token!r}"
        assert (
            Dataset.read_file(str(out)).validate_cog().is_valid
        ), f"predictor={predictor!r} produced an invalid COG"
