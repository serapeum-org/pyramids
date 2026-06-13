"""Tests for the windowed / band-targeted update path of ``IO.write_array`` (PB-8).

Covers the rasterio-``write(window=…)`` parity added to ``Dataset.write_array``:
the ``window`` and ``top_left_corner`` placements, single-band targeting,
multi-band writes, the default whole-array write, and the read-only /
out-of-bounds / shape / band guards. Fixtures are in-memory
``create_from_array`` rasters; the read-only case uses a ``tmp_path`` file.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OutOfBoundsError, ReadOnlyError
from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def blank() -> Dataset:
    """A writable 5x5 raster of zeros (EPSG:4326, cell size 1, top-left (0,5)).

    Returns:
        Dataset: in-memory zero raster.
    """
    return Dataset.create_from_array(
        np.zeros((5, 5), dtype="float32"),
        top_left_corner=(0, 5),
        cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture(scope="function")
def blank_multiband() -> Dataset:
    """A writable 2-band 5x5 raster of zeros.

    Returns:
        Dataset: in-memory 2-band zero raster.
    """
    return Dataset.create_from_array(
        np.zeros((2, 5, 5), dtype="float32"),
        top_left_corner=(0, 5),
        cell_size=1.0,
        epsg=4326,
    )


class TestWriteArrayWindow:
    """Tests for ``IO.write_array`` window/band/guard behaviour."""

    def test_window_patches_region(self, blank):
        """A window write updates exactly the targeted sub-region.

        Test scenario:
            Window(1, 1, 2, 2) sets the inner 2x2 to ones; surrounding cells stay 0.
        """
        blank.write_array(np.ones((2, 2)), window=Window(1, 1, 2, 2))
        arr = blank.read_array()
        assert arr[1:3, 1:3].tolist() == [
            [1.0, 1.0],
            [1.0, 1.0],
        ], f"Window not written: {arr}"
        assert arr[0, 0] == 0.0 and arr[4, 4] == 0.0, "Cells outside the window changed"

    def test_top_left_corner_still_works(self, blank):
        """The legacy top_left_corner placement is unchanged.

        Test scenario:
            A 2x2 patch at [2,2] lands at rows/cols 2..3.
        """
        blank.write_array(np.array([[7.0, 8.0], [9.0, 10.0]]), top_left_corner=[2, 2])
        arr = blank.read_array()
        assert arr[2:4, 2:4].tolist() == [
            [7.0, 8.0],
            [9.0, 10.0],
        ], f"tlc patch wrong: {arr}"

    def test_default_writes_at_origin(self, blank):
        """With neither window nor top_left_corner the array writes at (0,0).

        Test scenario:
            A full 5x5 array of ones overwrites the whole raster.
        """
        blank.write_array(np.ones((5, 5)))
        assert blank.read_array().sum() == 25.0, "Default write did not fill the raster"

    def test_band_targeted_window(self, blank_multiband):
        """A band-targeted window write touches only that band.

        Test scenario:
            band=1 with Window(0, 0, 2, 2) sets band-1's corner; band 0 stays 0.
        """
        blank_multiband.write_array(
            np.full((2, 2), 3.0), band=1, window=Window(0, 0, 2, 2)
        )
        arr = blank_multiband.read_array()
        assert arr[1, 0, 0] == 3.0, f"Band 1 not patched: {arr[1, 0, 0]}"
        assert arr[0, 0, 0] == 0.0, f"Band 0 should be untouched: {arr[0, 0, 0]}"

    def test_band_targeted_top_left_corner(self, blank_multiband):
        """Band targeting also works with top_left_corner.

        Test scenario:
            band=0 patched at [3,3] leaves band 1 untouched.
        """
        blank_multiband.write_array(
            np.full((2, 2), 5.0), band=0, top_left_corner=[3, 3]
        )
        arr = blank_multiband.read_array()
        assert arr[0, 3, 3] == 5.0, f"Band 0 not patched: {arr[0, 3, 3]}"
        assert arr[1, 3, 3] == 0.0, f"Band 1 should be untouched: {arr[1, 3, 3]}"

    def test_multiband_array_write(self, blank_multiband):
        """A 3D array writes across bands when band is not given.

        Test scenario:
            A (2,5,5) array of ones fills both bands.
        """
        blank_multiband.write_array(np.ones((2, 5, 5)))
        arr = blank_multiband.read_array()
        assert (
            arr[0].sum() == 25.0 and arr[1].sum() == 25.0
        ), "Multiband write incomplete"

    def test_window_at_edge_is_allowed(self, blank):
        """A window flush against the bottom-right edge is valid.

        Test scenario:
            Window(3, 3, 2, 2) ends exactly at row/col 5 (the raster size).
        """
        blank.write_array(np.ones((2, 2)), window=Window(3, 3, 2, 2))
        assert (
            blank.read_array()[3:5, 3:5].sum() == 4.0
        ), "Edge-aligned window not written"

    def test_window_shape_mismatch_raises(self, blank):
        """An array whose shape differs from the window raises ValueError.

        Test scenario:
            A 3x3 array given a 2x2 window is rejected.
        """
        with pytest.raises(ValueError, match="does not match the window size"):
            blank.write_array(np.ones((3, 3)), window=Window(0, 0, 2, 2))

    def test_legacy_tuple_wrong_length_raises_clear_error(self, blank):
        """A deprecated tuple of the wrong length gives a clear error (L6).

        Test scenario:
            A 3-element legacy tuple used to raise a bare "not enough values to
            unpack"; it must now raise a ValueError naming the expected 4-integer
            (row_off, col_off, n_rows, n_cols) form. The deprecation warning still
            fires first.
        """
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="tuple of 4 integers"):
                blank.write_array(np.ones((2, 2)), window=(0, 0, 2))

    def test_window_out_of_bounds_raises(self, blank):
        """A window extending past the raster raises OutOfBoundsError.

        Test scenario:
            Window(4, 4, 3, 3) would reach row/col 7 on a 5x5 raster.
        """
        with pytest.raises(OutOfBoundsError, match="falls outside"):
            blank.write_array(np.ones((3, 3)), window=Window(4, 4, 3, 3))

    def test_top_left_corner_out_of_bounds_raises(self, blank):
        """A top_left_corner placement past the raster raises OutOfBoundsError.

        Test scenario:
            A 2x2 patch at [4,4] would reach row/col 6 on a 5x5 raster.
        """
        with pytest.raises(OutOfBoundsError, match="falls outside"):
            blank.write_array(np.ones((2, 2)), top_left_corner=[4, 4])

    def test_negative_offset_raises(self, blank):
        """A negative offset raises OutOfBoundsError.

        Test scenario:
            Window(0, -1, 2, 2) has a negative row offset.
        """
        with pytest.raises(OutOfBoundsError, match="falls outside"):
            blank.write_array(np.ones((2, 2)), window=Window(0, -1, 2, 2))

    def test_band_out_of_range_raises(self, blank_multiband):
        """A band index beyond the raster raises ValueError.

        Test scenario:
            band=9 on a 2-band raster is out of range.
        """
        with pytest.raises(ValueError, match="out of range"):
            blank_multiband.write_array(
                np.ones((2, 2)), band=9, window=Window(0, 0, 2, 2)
            )

    def test_band_write_requires_2d_array(self, blank_multiband):
        """A band-targeted write with a non-2D array raises ValueError.

        Test scenario:
            band=0 given a 3D array is rejected.
        """
        with pytest.raises(ValueError, match="requires a 2D array"):
            blank_multiband.write_array(np.ones((2, 2, 2)), band=0)

    def test_read_only_dataset_raises(self, tmp_path):
        """Writing into a read-only dataset raises ReadOnlyError.

        Test scenario:
            A file reopened with read_only=True rejects write_array.
        """
        path = tmp_path / "ro.tif"
        Dataset.create_from_array(
            np.zeros((3, 3), dtype="float32"),
            top_left_corner=(0, 3),
            cell_size=1.0,
            epsg=4326,
            path=str(path),
        )
        read_only = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError, match="read-only"):
            read_only.write_array(np.ones((2, 2)), window=Window(0, 0, 2, 2))
