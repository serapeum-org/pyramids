"""`plot_histogram` on a CF-packed band: physical values, with the gap left out.

The mask is judged against the stored counts, where the sentinel lives, and the histogram
is then drawn over the physical values. Masking a physical read against the stored `-9999`
matched nothing, so every gap landed in the lowest bucket and stretched the axis down to
`-98.49`.
"""

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.plot

_cleo_config = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_cleo_config.Config.set_matplotlib_backend("agg")

# After the cleopatra importorskip: matplotlib arrives with the [viz] extra, so importing
# it at the top of the file would error instead of skipping on a no-viz install.
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every matplotlib figure after each test to bound memory."""
    yield
    plt.close("all")


def _packed(counts: np.ndarray) -> Dataset:
    """A one-band `int16` raster packed at `0.01` / `1.5`, declaring `-9999`.

    Args:
        counts: The stored counts.

    Returns:
        Dataset: The packed raster.
    """
    dataset = Dataset.from_array(
        counts,
        geo_ref=GeoReference(
            top_left_corner=(0, counts.shape[0]), cell_size=1.0, epsg=4326
        ),
        no_data_value=-9999,
    )
    dataset.scale = [0.01]
    dataset.offset = [1.5]
    return dataset


class TestPlotHistogramOnAPackedBand:
    """`Analysis.plot_histogram` masks in stored units and bins physical values."""

    def test_the_bins_span_the_physical_values_without_the_gap(self):
        """The edges run from `2.5` to `4.5` and the three real cells are all counted.

        Test scenario:
            A 2x2 band with one gap. Had the gap survived the mask, the lowest edge
            would be the physical sentinel `-98.49`; had the counts been binned, the
            edges would run from `100` to `300`.
        """
        dataset = _packed(np.array([[-9999, 100], [200, 300]], dtype="int16"))

        _fig, _ax, hist = dataset.plot_histogram(band=0, bins=2)

        edges = np.asarray(hist["bins"], dtype="float64").ravel()
        assert edges[0] == pytest.approx(2.5), f"lowest edge {edges[0]}"
        assert edges[-1] == pytest.approx(4.5), f"highest edge {edges[-1]}"
        assert float(np.sum(hist["n"])) == pytest.approx(3.0), hist["n"]

    def test_a_decimated_read_is_masked_in_stored_units_too(self):
        """With `max_samples`, the decimated samples are judged against the stored sentinel.

        Test scenario:
            A 4x4 band read at a 2x2 budget samples one cell per 2x2 block, and the
            top-left block is all gaps. Only the three data blocks may reach the bins.
        """
        counts = np.full((4, 4), 100, dtype="int16")
        counts[:2, :2] = -9999
        counts[2:, 2:] = 300
        dataset = _packed(counts)

        _fig, _ax, hist = dataset.plot_histogram(band=0, bins=2, max_samples=4)

        edges = np.asarray(hist["bins"], dtype="float64").ravel()
        assert float(np.sum(hist["n"])) == pytest.approx(3.0), hist["n"]
        assert edges[0] == pytest.approx(2.5), f"lowest edge {edges[0]}"
        assert edges[-1] == pytest.approx(4.5), f"highest edge {edges[-1]}"

    def test_a_physical_exclude_value_is_dropped_after_unpacking(self):
        """`exclude_value` is compared with the physical values the bins are drawn over."""
        dataset = _packed(np.array([[-9999, 100], [200, 300]], dtype="int16"))

        _fig, _ax, hist = dataset.plot_histogram(band=0, bins=2, exclude_value=4.5)

        edges = np.asarray(hist["bins"], dtype="float64").ravel()
        assert float(np.sum(hist["n"])) == pytest.approx(2.0), hist["n"]
        assert edges[-1] == pytest.approx(3.5), f"highest edge {edges[-1]}"
