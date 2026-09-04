"""One rule for what `band=None` means, shared by both read paths.

The plain windowed read and the decimated read each spelled it out: `None` on a
multi-band raster means every band stacked on a leading axis, anything else is a
single band read flat, and `None` on a single-band raster means band 0 -- read
flat, not stacked. Two copies of one rule, and the squeeze-or-stack decision is
the part that is easiest to get subtly wrong.

The unwindowed multi-band read keeps its own branch: `ReadAsArray()` fetches the
whole cube in one GDAL call, which a per-band stack cannot match. That is an
optimisation, not a spelling difference, so it stays visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines.io import _stack_bands

pytestmark = pytest.mark.core


class TestTheStackingRule:
    """The helper on its own, over the three cases it decides between."""

    def test_none_on_a_multi_band_raster_stacks(self):
        """Test scenario: the leading axis is bands, in index order."""
        result = _stack_bands(lambda i: np.full((2, 2), i), None, 3)

        assert result.shape == (3, 2, 2)
        assert result[0, 0, 0] == 0 and result[2, 0, 0] == 2

    def test_a_named_band_is_read_flat(self):
        """Test scenario: asking for one band gives a 2-D array, not a 1xNxM."""
        result = _stack_bands(lambda i: np.full((2, 2), i), 1, 3)

        assert result.shape == (2, 2)
        assert result[0, 0] == 1

    def test_none_on_a_single_band_raster_is_read_flat(self):
        """The case a naive "None means stack" rule gets wrong.

        Test scenario:
            A one-band raster read with `band=None` returns 2-D. Stacking it
            would give `(1, rows, cols)` and change the shape every consumer
            of a single-band read expects.
        """
        result = _stack_bands(lambda i: np.full((2, 2), i), None, 1)

        assert result.shape == (2, 2)

    def test_band_zero_is_not_confused_with_none(self):
        """`0` is falsy, which is exactly how this rule gets broken.

        Test scenario:
            An `if not band` in place of `if band is None` would treat an
            explicit `band=0` on a multi-band raster as "all bands" and return
            a 3-D array.
        """
        result = _stack_bands(lambda i: np.full((2, 2), i), 0, 3)

        assert result.shape == (2, 2), "band=0 was treated as band=None"

    def test_each_band_is_read_exactly_once(self):
        """The callable is not re-invoked per element.

        Test scenario:
            A per-band read is a GDAL call; invoking it twice per band would
            double the I/O of every full-cube read without changing the
            result, so nothing else would catch it.
        """
        seen: list[int] = []

        def read(index: int):
            seen.append(index)
            return np.zeros((2, 2))

        _stack_bands(read, None, 4)

        assert seen == [0, 1, 2, 3]


class TestBothReadPathsAgree:
    """End to end, on a raster whose bands differ."""

    @pytest.fixture
    def raster(self) -> Dataset:
        """A 3-band raster whose bands hold distinguishable values.

        Returns:
            Dataset: Band i is filled with i, so a mis-stacked read shows up.
        """
        array = np.stack(
            [np.full((4, 5), value, dtype="float32") for value in (0.0, 1.0, 2.0)]
        )
        return Dataset.from_array(
            array,
            geo_ref=GeoReference(top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326),
        )

    def test_the_windowed_read_matches_the_unwindowed_one(self, raster):
        """The two branches of the plain read, which are different code.

        Test scenario:
            Unwindowed goes through `ReadAsArray()` in one GDAL call;
            windowed stacks per band through the helper. A full-extent window
            must give exactly what no window gives.
        """
        full = [0, 0, raster.columns, raster.rows]

        unwindowed = np.asarray(raster.read_array())
        windowed = np.asarray(raster.read_array(window=full))

        assert np.array_equal(unwindowed, windowed)

    def test_the_stack_is_in_band_order(self, raster):
        """A reversed or shuffled stack would still have the right shape.

        Test scenario:
            Shape assertions alone cannot catch a mis-ordered stack, so the
            fixture makes each band's values name its index.
        """
        stacked = np.asarray(raster.read_array())

        assert [float(stacked[i, 0, 0]) for i in range(3)] == [0.0, 1.0, 2.0]

    def test_a_single_band_read_matches_its_plane_of_the_stack(self, raster):
        """The two ways of asking for one band must agree.

        Test scenario:
            `read_array(band=1)` and `read_array()[1]` are the same request,
            and the helper is what keeps the squeeze consistent between them.
        """
        stacked = np.asarray(raster.read_array())

        for index in range(3):
            single = np.asarray(raster.read_array(band=index))
            assert single.shape == (4, 5)
            assert np.array_equal(single, stacked[index])
