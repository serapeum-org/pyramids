"""Two pictures of one band, masking one set of pixels.

`plot_histogram` and `to_image` read the same band and both drop its no-data
before handing the rest to cleopatra. The consolidation moved the histogram onto
the shared `is_no_data` predicate and left `to_image` on the two-step form it had
always used -- `is_nan_sentinel` to decide whether the sentinel is comparable,
then an exact `!=` against it. So a cell just off the declared sentinel was
no-data to the histogram and ordinary data to the image: two views of one raster
that agreed before the refactor and did not after.

The masks are compared through the one thing both renderers expose about them --
the refusal each raises when masking leaves nothing to draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference

pytestmark = pytest.mark.plot

# A sentinel of -9999 with cells 1e-5 off it. `_warn_if_nodata_absent` and
# `footprint` both ask `is_no_data(..., rtol=1e-5)` of the same band, so these
# are no-data to every other reader in the module. float64, because float32
# cannot hold the fifth decimal of a five-digit number and the whole band would
# collapse onto the sentinel before either renderer saw it.
_NEAR_SENTINEL = np.array(
    [[-9999.0, -9999.00001], [-9998.99999, -9999.00002]], dtype="float64"
)

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

    def test_the_histogram_refuses_a_band_of_near_sentinel_cells(self):
        """The reference behaviour, which `to_image` has to match.

        Test scenario:
            Every cell is the sentinel or within the tolerance of it, so after
            masking there is nothing to bin and the histogram says so.
        """
        with pytest.raises(ValueError, match="no valid samples"):
            _band(_NEAR_SENTINEL).plot_histogram(band=0)

    def test_to_image_refuses_it_too(self):
        """The regression: it rendered the band the histogram called empty.

        Test scenario:
            The exact `!=` matched only the cell that is bit-for-bit -9999, so
            the other three counted as data and the image drew three coloured
            pixels for cells its sibling had already dropped.
        """
        with pytest.raises(ValueError, match=r"no valid \(non-nodata\) pixels"):
            _band(_NEAR_SENTINEL).to_image(band=0)

    def test_the_two_refusals_are_about_the_same_thing(self):
        """Both name the band and say masking left nothing.

        Test scenario:
            A caller who hits one and switches to the other should not be told
            two different stories about the same raster.
        """
        dataset = _band(_NEAR_SENTINEL)
        with pytest.raises(ValueError) as histogram:
            dataset.plot_histogram(band=0)
        with pytest.raises(ValueError) as image:
            dataset.to_image(band=0)

        assert "Band 0" in str(histogram.value)
        assert "Band 0" in str(image.value)


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
