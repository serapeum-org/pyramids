"""Tests for the per-overview layout enrichment of validate().details (PC-4)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.dataset.cog import validate

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def big_cog(tmp_path) -> str:
    """A 600x600 float COG (large enough to carry overviews).

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the COG.
    """
    arr = (np.random.default_rng(seed=4).random((600, 600)) * 100).astype("float32")
    ds = Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)
    return str(ds.to_cog(tmp_path / "big.tif"))


class TestValidateOverviewDetails:
    """Tests that validate().details carries the overview layout."""

    def test_details_include_overviews(self, big_cog):
        """A valid COG's report lists its overview pyramid.

        Args:
            big_cog: Fixture path to a 600x600 COG.

        Test scenario:
            details["overviews"] is a non-empty list whose first level decimates
            by >= 2, and details["blocksize"] is reported.
        """
        report = validate(big_cog)
        assert report.is_valid, f"expected valid, errors {report.errors}"
        assert report.details.get("overviews"), "overviews missing from details"
        assert report.details["overviews"][0]["decimation"] >= 2, "bad decimation"
        assert report.details.get("blocksize") == [512, 512], report.details.get("blocksize")

    def test_overview_count_matches(self, big_cog):
        """details['overview_count'] equals the number of listed overviews.

        Args:
            big_cog: Fixture path to a 600x600 COG.

        Test scenario:
            The count and the list length agree.
        """
        report = validate(big_cog)
        assert report.details["overview_count"] == len(report.details["overviews"]), (
            "overview_count must match the overviews list length"
        )
