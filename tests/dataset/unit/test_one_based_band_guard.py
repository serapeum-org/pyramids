"""One refusal for a 1-based band number that is out of range.

Three entry points guarded a 1-based band number, and each phrased the refusal
its own way: "band index 9 is out of range for a 3-band dataset (valid 1..3)",
"band 9 is out of range for a 3-band dataset (bands are 1-based)", and "band
number 9 out of range 1..3". Same defect, three messages, so what a user was
told depended on which door they came through.

The 0-based `validate_band_index` stays as it is. Both conventions are real --
GDAL numbers bands from one, the array-facing APIs index from zero -- so the
two guards are siblings, not duplicates, and the tests below hold them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.engines._validate import (
    validate_band_index,
    validate_one_based_band,
)

pytestmark = pytest.mark.core


class TestValidateOneBasedBand:
    """The range, its ends, and what the message says."""

    @pytest.mark.parametrize("band", [1, 2, 3])
    def test_every_band_in_range_passes(self, band: int):
        """Both ends are inclusive for a 1-based count.

        Args:
            band: A band number inside a 3-band dataset.

        Test scenario:
            1 and `band_count` are valid numbers, not boundaries to exclude.
            An off-by-one at either end would reject a real band.
        """
        assert validate_one_based_band(band, 3) is None

    @pytest.mark.parametrize("band", [0, -1, 4, 99])
    def test_a_band_outside_the_range_is_refused(self, band: int):
        """Zero is out of range here, which is the whole point of the sibling.

        Args:
            band: A band number outside a 3-band dataset.

        Test scenario:
            Under the 1-based convention band 0 does not exist. The 0-based
            guard accepts it, so routing a 1-based caller through that one
            would let an invalid band through to GDAL.
        """
        with pytest.raises(ValueError, match="out of range for a 3-band dataset"):
            validate_one_based_band(band, 3)

    def test_the_message_says_which_convention_it_used(self):
        """A bare "0 is out of range" reads like an off-by-one in pyramids.

        Test scenario:
            The refusal has to say that bands start at 1, otherwise a caller
            passing a 0-based index sees a message that looks like a bug rather
            than instructions.
        """
        with pytest.raises(ValueError) as exc_info:
            validate_one_based_band(0, 3)

        assert "bands are 1-based" in str(exc_info.value)

    def test_the_message_names_the_caller_s_own_argument(self):
        """`grib` calls it `variable`; the message must not say `band`.

        Test scenario:
            The GRIB reader's parameter is `variable=`, so a message about
            `band` sends the user looking for an argument that does not exist.
        """
        with pytest.raises(ValueError) as exc_info:
            validate_one_based_band(9, 2, name="variable")

        assert str(exc_info.value).startswith("variable 9 is out of range")

    def test_the_two_guards_disagree_about_zero_on_purpose(self):
        """The reason both exist, stated as an assertion.

        Test scenario:
            Band 0 is valid 0-based and invalid 1-based; band `count` is the
            reverse. Collapsing the two would silently shift every range by
            one, so the difference is pinned here.
        """
        assert validate_band_index(0, 3) is None
        with pytest.raises(ValueError):
            validate_one_based_band(0, 3)

        assert validate_one_based_band(3, 3) is None
        with pytest.raises(ValueError):
            validate_band_index(3, 3)


class TestTheCallSitesShareTheWording:
    """All three entry points now refuse the same way."""

    @pytest.fixture
    def dataset(self):
        """A three-band in-memory raster."""
        return Dataset.from_array(
            np.ones((3, 4, 5), dtype=np.float32),
            geo_ref=GeoReference(top_left_corner=(0.0, 5.0), cell_size=1.0, epsg=4326),
        )

    @pytest.mark.parametrize("band", [0, 4])
    def test_a_band_selector_out_of_range_is_refused(self, dataset, band: int):
        """`bands.select` takes 1-based numbers, and said "valid 1..3" before.

        Args:
            dataset: A three-band raster fixture.
            band: A 1-based band number outside it.

        Test scenario:
            `Bands.select` documents its `bands=` as 1-based, so 0 is out of
            range there even though it is a valid index for the 0-based
            `read_array`. Both ends must refuse with the shared wording.
        """
        with pytest.raises(ValueError, match="out of range for a 3-band dataset"):
            dataset.bands.select([band])

    def test_the_zero_based_reader_still_accepts_band_zero(self, dataset):
        """The sibling convention, reached through a real API.

        Args:
            dataset: A three-band raster fixture.

        Test scenario:
            `read_array(band=0)` is the first band, not an error. Converging
            the two guards would have broken exactly this.
        """
        assert dataset.read_array(band=0).shape == (4, 5)
