"""Tests for :func:`pyramids.base.crs.reproject_arrays` (issue #1135, item 1).

The NumPy-native, z-aware coordinate transform: arrays in, arrays out, no
per-element rounding and no ``.tolist()``. Covers the 2-D, 3-D and no-op-CRS
cases, equivalence with the list-based :func:`reproject_coordinates`, the shape
guards, and CRS-parse failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import CRSError
from pyramids.base.crs import reproject_arrays, reproject_coordinates

pytestmark = pytest.mark.core


class TestReprojectArrays:
    """Tests for the array-native coordinate transform."""

    def test_returns_numpy_arrays(self):
        """The 2-D form returns exactly two float64 ndarrays.

        Test scenario:
            Two WGS84 points reprojected to Web Mercator come back as a 2-tuple of
            ndarrays matching the input length.
        """
        out = reproject_arrays(
            np.array([31.0, 32.0]), np.array([30.0, 29.0]), from_crs=4326, to_crs=3857
        )
        assert len(out) == 2, f"expected (x, y), got {len(out)} arrays"
        assert all(isinstance(a, np.ndarray) for a in out), "outputs must be ndarrays"
        assert all(a.dtype == np.float64 for a in out), "outputs must be float64"
        assert out[0].shape == (2,), f"unexpected shape: {out[0].shape}"

    def test_matches_reproject_coordinates_unrounded(self):
        """It agrees with the list function to full precision (no per-element round).

        Test scenario:
            reproject_coordinates(precision=None) is the unrounded list transform;
            reproject_arrays must produce the same values, element for element.
        """
        x = [31.0, 20.0, -45.5, 100.25]
        y = [30.0, -10.0, 60.0, -80.0]
        ref_x, ref_y = reproject_coordinates(
            x, y, from_crs=4326, to_crs=3857, precision=None
        )
        got_x, got_y = reproject_arrays(
            np.array(x), np.array(y), from_crs=4326, to_crs=3857
        )
        assert np.allclose(got_x, ref_x, rtol=0, atol=0), (
            "x diverged from the list form"
        )
        assert np.allclose(got_y, ref_y, rtol=0, atol=0), (
            "y diverged from the list form"
        )

    def test_noop_crs_returns_input(self):
        """Reprojecting to the same CRS returns the coordinates unchanged.

        Test scenario:
            from_crs == to_crs is an identity transform.
        """
        x = np.array([31.0, 32.0, 33.0])
        y = np.array([30.0, 29.0, 28.0])
        out_x, out_y = reproject_arrays(x, y, from_crs=4326, to_crs=4326)
        assert np.array_equal(out_x, x), f"x changed on no-op: {out_x}"
        assert np.array_equal(out_y, y), f"y changed on no-op: {out_y}"

    def test_three_dim_returns_three_arrays(self):
        """A z array is carried through and returned as a third array.

        Test scenario:
            With z given, the result is a 3-tuple; under a no-op CRS the elevation
            is unchanged.
        """
        out = reproject_arrays(
            np.array([31.0]),
            np.array([30.0]),
            np.array([12.5]),
            from_crs=4326,
            to_crs=4326,
        )
        assert len(out) == 3, f"expected (x, y, z), got {len(out)} arrays"
        assert float(out[2][0]) == pytest.approx(12.5), f"z changed: {out[2][0]}"

    def test_accepts_python_lists(self):
        """Plain lists are accepted, not only ndarrays.

        Test scenario:
            List inputs produce the same result as their ndarray equivalents.
        """
        from_list = reproject_arrays([31.0], [30.0], from_crs=4326, to_crs=3857)
        from_arr = reproject_arrays(
            np.array([31.0]), np.array([30.0]), from_crs=4326, to_crs=3857
        )
        assert np.array_equal(from_list[0], from_arr[0]), "list vs array x differ"
        assert np.array_equal(from_list[1], from_arr[1]), "list vs array y differ"

    def test_xy_shape_mismatch_raises(self):
        """Unequal x/y shapes raise ValueError.

        Test scenario:
            x of length 3 and y of length 2 is rejected.
        """
        with pytest.raises(ValueError, match="x and y must share a shape"):
            reproject_arrays(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_z_shape_mismatch_raises(self):
        """A z that does not match x's shape raises ValueError.

        Test scenario:
            Two points but three z values is rejected.
        """
        with pytest.raises(ValueError, match="z must share x's shape"):
            reproject_arrays(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])
            )

    def test_bad_crs_raises_crserror(self):
        """A malformed CRS raises pyramids' CRSError, not a raw pyproj error.

        Test scenario:
            An unparseable from_crs is wrapped with a message naming both CRSes.
        """
        with pytest.raises(CRSError, match="reproject_arrays failed to parse CRS"):
            reproject_arrays(np.array([1.0]), np.array([1.0]), from_crs="not-a-crs")

    def test_exported_from_all(self):
        """reproject_arrays is part of the module's public surface.

        Test scenario:
            The name is listed in pyramids.base.crs.__all__.
        """
        import pyramids.base.crs as crs_module

        assert "reproject_arrays" in crs_module.__all__, "not exported in __all__"
