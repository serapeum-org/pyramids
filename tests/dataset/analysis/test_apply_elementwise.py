"""Tests for the opt-in streaming path of ``Analysis.apply`` (``elementwise=True``).

The tiled path must be byte-identical to the default whole-array pass for a per-pixel
``func``, including across 256-px tile seams, for both a plain band and a band carrying a
no-data value, and it must honour band selection and the ``inplace`` flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestApplyElementwise:
    """The ``elementwise=True`` tiled path matches the default whole-array apply."""

    @pytest.mark.parametrize(
        "func",
        [np.abs, lambda v: v * 2 + 1, np.sqrt],
        ids=["abs", "affine", "sqrt"],
    )
    def test_matches_whole_array_across_tile_boundaries(self, func):
        """A raster larger than one 256-px tile transforms identically streamed and whole.

        Args:
            func: A per-pixel callable applied to the domain values.

        Test scenario:
            ``apply(elementwise=True)`` on a 300x300 raster equals ``apply()`` cell for cell.
        """
        arr = (np.random.default_rng(0).random((300, 300)) * 50).astype("float64")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 15.0), cell_size=0.05, epsg=4326
        )
        streamed = ds.apply(func, elementwise=True).read_array()
        whole = ds.apply(func, elementwise=False).read_array()
        np.testing.assert_array_equal(
            streamed, whole, err_msg="Streamed apply must match the whole-array apply"
        )

    def test_preserves_no_data_cells(self):
        """No-data cells stay no-data and only domain cells are transformed.

        Test scenario:
            A raster with scattered no-data in different tiles keeps those cells and doubles
            the rest, matching the whole-array pass.
        """
        arr = np.ones((300, 300), dtype="float64") * 3.0
        arr[10, 10] = -9999.0
        arr[290, 295] = -9999.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 15.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        streamed = ds.apply(lambda v: v * 2, elementwise=True).read_array()
        whole = ds.apply(lambda v: v * 2, elementwise=False).read_array()
        np.testing.assert_array_equal(
            streamed, whole, err_msg="No-data layout and domain map must match"
        )
        assert streamed[10, 10] == -9999.0, "No-data cell must be preserved"

    def test_band_selection(self):
        """A non-default band is streamed and returned as a single-band result.

        Test scenario:
            ``apply(elementwise=True, band=1)`` on a 2-band raster transforms band 1 and the
            result matches the whole-array apply on the same band.
        """
        arr = np.stack(
            [
                np.zeros((300, 300), dtype="float64"),
                (np.random.default_rng(1).random((300, 300)) * 10).astype("float64"),
            ]
        )
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 15.0), cell_size=0.05, epsg=4326
        )
        streamed = ds.apply(np.abs, band=1, elementwise=True).read_array()
        whole = ds.apply(np.abs, band=1, elementwise=False).read_array()
        assert streamed.shape == (300, 300), (
            f"Expected single band, got {streamed.shape}"
        )
        np.testing.assert_array_equal(
            streamed, whole, err_msg="Band-1 streamed apply must match whole-array"
        )

    def test_inplace_streaming(self):
        """``elementwise=True`` with ``inplace=True`` updates the source in place.

        Test scenario:
            The call returns ``self`` and the source array is the doubled result.
        """
        arr = (np.random.default_rng(2).random((300, 300)) * 4).astype("float64")
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 15.0), cell_size=0.05, epsg=4326
        )
        result = ds.apply(lambda v: v * 2, inplace=True, elementwise=True)
        assert result is ds, "inplace streaming apply should return self"
        np.testing.assert_array_equal(
            ds.read_array(), arr * 2, err_msg="In-place streamed result must be doubled"
        )

    def test_all_nodata_tile_with_vectorize_fallback_func(self):
        """A fully-no-data tile does not crash the np.vectorize fallback path.

        Test scenario:
            A raster whose last 256-px tile is entirely no-data, transformed with a scalar
            conditional func (which forces the np.vectorize fallback on array input), streams
            without raising and matches the whole-array pass.
        """
        arr = (np.random.default_rng(3).random((300, 300)) * 10).astype("float64")
        arr[256:300, 256:300] = -9999.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 15.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        func = lambda v: 1.0 if v > 5 else 0.0  # noqa: E731 - forces vectorize fallback
        streamed = ds.apply(func, elementwise=True).read_array()
        whole = ds.apply(func, elementwise=False).read_array()
        np.testing.assert_array_equal(
            streamed,
            whole,
            err_msg="empty-domain tile must not crash and must match the whole-array pass",
        )
        assert streamed[270, 270] == -9999.0, "no-data cell in the empty tile preserved"
