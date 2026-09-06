"""Two pictures of one band, masking one set of pixels.

`plot_histogram` and `to_image` read the same band and both drop its no-data
before handing the rest to cleopatra. The consolidation moved the histogram onto
the shared `is_no_data` predicate and left `to_image` on the two-step form it had
always used -- `is_nan_sentinel` to decide whether the sentinel is comparable,
then an exact `!=` against it. So a cell just off the declared sentinel was
no-data to the histogram and ordinary data to the image: two views of one raster
that agreed before the refactor and did not after.

Both now ask `is_stored_no_data`, so the masks agree from either side: a band of
nothing but the sentinel is refused by both, and a band of cells that merely lie
near it is drawn by both. The fixture the first pair used to be built from --
cells `1e-5` away from `-9999`, no-data under the fixed `rtol=1e-5` the renderers
briefly shared -- encoded the very window that made the histogram drop real data
(round 4, L1), so it has been replaced by a band that is genuinely all sentinel.
The masks are still compared through the one thing both renderers expose about
them: the refusal each raises when masking leaves nothing to draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.plot

# Nothing but the sentinel, so masking leaves neither renderer anything.
_ALL_SENTINEL = np.array([[-9999.0, -9999.0], [-9999.0, -9999.0]], dtype="float64")

# Ordinary cells a hair from the sentinel. 0.05 is 5e-6 of -9999, so the
# `rtol=1e-5` the renderers shared masked all three; the dtype's own slack is
# 1.2e-3 wide, so they are data. float64, because float32 cannot hold the
# distinction at this magnitude.
_NEAR_SENTINEL = np.array([[-9999.0, -9998.95], [-9999.05, -9998.98]], dtype="float64")

# The same sentinel with cells far enough off it to be real data. -9990 is
# inside the window `DEFAULT_RTOL` (1e-3) would mask and outside the one the
# renderers use, so it is what tells a shared predicate from a loose one.
_REAL_DATA = np.array([[-9999.0, -9995.0], [-9990.0, -100.0]], dtype="float64")


def _band(values: np.ndarray) -> Dataset:
    """A single-band raster carrying `values` with a -9999 sentinel.

    Args:
        values: The band contents.

    Returns:
        Dataset: An in-memory raster declaring `-9999.0` as its no-data.
    """
    return Dataset.from_array(
        values,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close the figures cleopatra opens, so the suite does not accumulate them.

    Yields:
        None: Control returns to the test; figures are closed afterwards.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestTheTwoRenderersMaskTheSameCells:
    """A band that is entirely no-data to one has to be to the other."""

    def test_the_histogram_refuses_a_band_of_sentinel_cells(self):
        """The reference behaviour, which `to_image` has to match.

        Test scenario:
            Every cell is the sentinel, so after masking there is nothing to
            bin and the histogram says so.
        """
        dataset = _band(_ALL_SENTINEL)

        with pytest.raises(ValueError, match="no valid samples"):
            dataset.plot_histogram(band=0)

    def test_to_image_refuses_it_too(self):
        """The regression: it rendered the band the histogram called empty.

        Test scenario:
            The exact `!=` matched only the cell that is bit-for-bit -9999, so
            cells the sibling renderer had already dropped were still coloured.
        """
        dataset = _band(_ALL_SENTINEL)

        with pytest.raises(ValueError, match=r"no valid \(non-nodata\) pixels"):
            dataset.to_image(band=0)

    def test_the_two_refusals_are_about_the_same_thing(self):
        """Both name the band and say masking left nothing.

        Test scenario:
            A caller who hits one and switches to the other should not be told
            two different stories about the same raster.
        """
        dataset = _band(_ALL_SENTINEL)
        with pytest.raises(ValueError) as histogram:
            dataset.plot_histogram(band=0)
        with pytest.raises(ValueError) as image:
            dataset.to_image(band=0)

        assert "Band 0" in str(histogram.value)
        assert "Band 0" in str(image.value)


class TestNeitherRendererDropsCellsNearTheSentinel:
    """Agreeing on a predicate must not mean agreeing to lose data (L1)."""

    def test_the_histogram_counts_the_near_sentinel_cells(self):
        """Three of the four cells are data and are binned.

        Test scenario:
            `rtol=1e-5` around -9999 masks everything in [-9999.1, -9998.9],
            which is all three -- so the histogram drew one bar for a band with
            three ordinary cells in it, and the picture disagreed with
            `get_histogram`, which never masked them.
        """
        _fig, _ax, hist = _band(_NEAR_SENTINEL).plot_histogram(band=0, bins=3)

        assert int(np.sum(hist["n"][0])) == 3, hist

    def test_to_image_draws_them(self):
        """The same three cells, through the sibling renderer.

        Test scenario:
            The image is the other half of the agreement: had only one of the
            two been tightened, the pair would disagree again, in the opposite
            direction from the divergence this module was written for.
        """
        from PIL import Image

        image = _band(_NEAR_SENTINEL).to_image(band=0)

        assert isinstance(image, Image.Image)
        assert image.size == (2, 2)


class TestRealDataIsStillDrawn:
    """Agreeing on a tolerance must not mean masking everything near it."""

    def test_to_image_keeps_cells_that_are_only_near_the_sentinel(self):
        """-9995 and -9990 are data, and stay data.

        Test scenario:
            `is_no_data`'s package default is `rtol=1e-3`, which for a sentinel
            of -9999 masks everything in [-10009, -9989] -- both of these. The
            renderers ask with the tolerance their own module already uses, so
            the band still has three cells to colour.
        """
        from PIL import Image

        image = _band(_REAL_DATA).to_image(band=0)

        assert isinstance(image, Image.Image)
        assert image.size == (2, 2)

    def test_the_histogram_counts_them_too(self):
        """The same three cells, through the sibling renderer.

        Test scenario:
            This is the pair the divergence was visible in: the histogram
            counting one cell where the image drew three.
        """
        _fig, _ax, hist = _band(_REAL_DATA).plot_histogram(band=0, bins=3)

        assert int(np.sum(hist["n"][0])) == 3, hist
