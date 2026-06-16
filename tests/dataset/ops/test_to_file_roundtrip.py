"""Regression for #570: a multi-band ``to_file`` round-trips through a 2nd handle.

``to_file`` writes a compressed GeoTIFF (the default ``COMPRESS=DEFLATE``). The
compressed strips and the TIFF directory must be fully flushed to disk before
``to_file`` returns, otherwise a *separately*-opened reader sees the no-data fill
instead of the real pixels. The bug only surfaced on Linux/macOS: ``CreateCopy``
left everything buffered in the write handle (``FlushCache`` was not enough), so
a fresh ``read_file`` of the same path read back an all-``-9999`` array.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def test_multiband_to_file_roundtrips_through_second_handle(tmp_path):
    """``to_file`` then a fresh ``read_file`` recovers the pixels, not the fill.

    Test scenario:
        A 2-band float32 raster (no-data ``-9999``, large enough to span several
        GeoTIFF strips) is written through the default compressed path, then
        reopened with a separate handle. Every pixel must equal the source
        (#570 read it back as all ``-9999`` on Unix).
    """
    a = np.arange(2 * 32 * 48, dtype=np.float32).reshape(2, 32, 48)
    ds = Dataset.create_from_array(
        a,
        top_left_corner=(0.0, 32.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999.0,
    )
    out = str(tmp_path / "mb.tif")
    ds.to_file(out)
    reread = np.asarray(Dataset.read_file(out).read_array())
    np.testing.assert_array_equal(reread, a)
