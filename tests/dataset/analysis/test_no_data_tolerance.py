"""Cells near the sentinel are data, and every reader of the band has to agree.

The consolidation moved the analysis engine's no-data masking onto one shared
predicate, which is right, and then asked it with a fixed `rtol=1e-5`, which is
not: a relative tolerance scales with the sentinel, so `-9999` masked everything
within `0.1` of it and a `2e9` integer sentinel masked everything within
`20 000`. Real cells stopped being counted, drawn, filled and polygonised, with
no warning that anything had been dropped.

The tolerance now comes from the band's dtype -- none at all for an integer
band, single precision's slack for a floating one -- so what these tests pin is
that ordinary data a hair away from the sentinel survives every entry point,
while the sentinel itself is still found.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

# One sentinel cell, two ordinary cells close to it, one far away. 0.05 is
# 5e-6 of -9999 -- inside the window `rtol=1e-5` masked, and 40 times wider
# than anything float64 storage can account for.
_NEAR = np.array(
    [[-9999.0, -9998.95], [-9999.05, 12.0]],
    dtype="float64",
)

# The same shape on an integer band, where the sentinel's magnitude makes the
# relative window enormous: 1e-5 of 2e9 is 20 000.
_NEAR_INT = np.array(
    [[2_000_000_000, 1_999_990_000], [2_000_010_000, 12]],
    dtype="int32",
)


def _band(values: np.ndarray, no_data_value: float) -> Dataset:
    """A single-band raster carrying `values` and declaring `no_data_value`.

    Args:
        values: The band contents.
        no_data_value: The sentinel to declare on the band.

    Returns:
        Dataset: An in-memory single-band raster, on a projected CRS so that a
            polygonised footprint's area is in the same units as its cells.
    """
    return Dataset.from_array(
        values,
        no_data_value=no_data_value,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=3857),
    )


class TestTheDomainIsEveryCellButTheSentinel:
    """Counting, applying and filling all decide from the same mask."""

    def test_count_domain_cells_counts_the_near_sentinel_cells(self):
        """Three of the four cells are data.

        Test scenario:
            `count_domain_cells` asked with the package default (`rtol=1e-3`),
            which around `-9999` masks everything down to `-10009` -- so two
            ordinary cells were counted as no-data and the raster reported one
            domain cell too few, twice over.
        """
        assert _band(_NEAR, -9999.0).count_domain_cells(band=0) == 3

    def test_apply_transforms_the_near_sentinel_cells(self):
        """`apply` writes through every cell that is not the sentinel.

        Test scenario:
            The domain mask `apply` builds decided which cells `func` ever
            sees. A cell it wrongly called no-data was left at its original
            value while its neighbours were transformed, so the output silently
            mixed transformed and untransformed data.
        """
        result = _band(_NEAR, -9999.0).apply(lambda v: v * 0.0 + 7.0).read_array()

        assert result[0, 1] == 7.0, "-9998.95 was left untransformed"
        assert result[1, 0] == 7.0, "-9999.05 was left untransformed"
        assert result[0, 0] == -9999.0, "the sentinel cell was transformed"

    def test_fill_writes_into_the_near_sentinel_cells(self):
        """`fill` replaces every domain cell and leaves the sentinel alone.

        Test scenario:
            `fill` asked with a tolerance of its own (`rtol=1e-6`), so the two
            near cells were preserved as though they were no-data while the
            rest of the band was overwritten.
        """
        filled = _band(_NEAR, -9999.0).fill(3.0).read_array()

        assert filled[0, 1] == 3.0, "-9998.95 was treated as no-data"
        assert filled[1, 0] == 3.0, "-9999.05 was treated as no-data"
        assert filled[0, 0] == -9999.0, "the sentinel cell was filled"


class TestTheSentinelIsStillFound:
    """Loosening nothing is not the same as tightening to exact equality."""

    def test_a_band_of_nothing_but_the_sentinel_has_no_domain(self):
        """The counterpart to the tests above: the sentinel is still masked.

        Test scenario:
            A predicate that kept everything would pass every test above and be
            just as wrong, so one band is pinned from the other side -- every
            cell is the declared sentinel, and none of them is data.
        """
        values = np.full((2, 2), -9999.0, dtype="float64")

        assert _band(values, -9999.0).count_domain_cells(band=0) == 0

    def test_a_sentinel_that_went_through_single_precision_is_masked(self):
        """A float32 band whose declared sentinel is the wider double.

        Test scenario:
            `1e30` is not representable in float32, so the cells hold
            `1.0000000150474662e+30` while the band declares `1e30`. Under
            exact equality the sentinel matches nothing and the band reports
            four domain cells; the dtype's own slack finds it.
        """
        values = np.array([[1e30, 2.0], [3.0, 4.0]], dtype="float32")

        assert _band(values, 1e30).count_domain_cells(band=0) == 3

    def test_a_zero_sentinel_does_not_swallow_small_values(self):
        """A sentinel of `0` has no slack, so nothing near zero is no-data.

        Test scenario:
            `numpy.isclose`'s default `atol=1e-8` decides the answer for a
            zero sentinel, because its relative window is empty -- so a
            float band's genuinely small cells read as no-data.
        """
        values = np.array([[0.0, 1e-9], [-1e-9, 5.0]], dtype="float64")

        assert _band(values, 0.0).count_domain_cells(band=0) == 3


class TestTheIntegerBandGetsNoToleranceAtAll:
    """An integer sentinel is stored exactly; a relative window is pure loss."""

    def test_cells_ten_thousand_away_from_the_sentinel_are_data(self):
        """`rtol=1e-5` of `2e9` is a window of 20 000 counts.

        Test scenario:
            The two ordinary cells are 10 000 from the sentinel -- values a
            32-bit band can hold and distinguish -- and were masked as no-data
            by every reader that asked with a relative tolerance.
        """
        assert _band(_NEAR_INT, 2_000_000_000).count_domain_cells(band=0) == 3

    def test_the_integer_sentinel_itself_is_still_masked(self):
        """The exact sentinel is no-data on an integer band too.

        Test scenario:
            The complement of the test above, so "no tolerance" cannot be
            mistaken for "no masking".
        """
        footprint = _band(_NEAR_INT, 2_000_000_000).footprint(band=0)

        assert footprint is not None, "an all-no-data band was reported"
        assert len(footprint) >= 1, "the covered cells produced no polygon"


class TestTheWarningAndTheFootprintReadTheSameCells:
    """Two answers about one band, from one predicate."""

    def test_a_band_whose_sentinel_only_nearly_appears_is_reported_absent(self, caplog):
        """The sentinel is declared but no cell holds it.

        Args:
            caplog: pytest fixture capturing the logger output.

        Test scenario:
            Every cell is ordinary data, two of them near `-9999`. Asked with
            `rtol=1e-5` the warning saw those two as the sentinel and stayed
            silent, so a raster with a wrong no-data value looked fine.
        """
        values = np.array([[-9998.95, -9999.05], [1.0, 2.0]], dtype="float64")

        with caplog.at_level("WARNING"):
            _band(values, -9999.0).footprint(band=0)

        assert "does not exist in the raster" in caplog.text

    def test_the_footprint_covers_the_near_sentinel_cells(self):
        """Cells near the sentinel are inside the raster's coverage.

        Test scenario:
            `footprint` polygonises the domain mask, so a near-sentinel cell
            wrongly masked became a hole in the reported extent of the data.
        """
        footprint = _band(_NEAR, -9999.0).footprint(band=0)

        assert footprint is not None, "an all-no-data band was reported"
        covered = float(footprint.geometry.area.sum())
        # An explicit absolute window: `approx`'s relative default is 1e-6 of
        # 3.0, which is fine here but says nothing about what tolerance the
        # geometry actually needs. One part in ten thousand of a single cell is
        # far below the difference a wrongly-masked cell makes, which is 1.0.
        assert covered == pytest.approx(3.0, abs=1e-4), (
            f"three 1x1 cells are data, but the footprint covers {covered}"
        )
