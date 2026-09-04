"""The COG facade understands the same axis spellings as the rest of pyramids.

`write_cog` accepts a labeled `DataArray`, and looked for its spatial
coordinates under three names per axis -- `x` / `longitude` / `lon`. A
rotated-pole grid's `rlon` / `rlat`, or a projected store's `easting` /
`northing`, were refused with "could not find longitude/latitude (or x/y)
coordinates" even though the rest of the package recognises them.

The lookup keeps a second condition the shared vocabulary does not imply: the
coordinate must be **1-D**. The geotransform here is derived from the first two
values of each axis, which describes a grid only for a vector. A curvilinear
store's 2-D `nav_lon` / `nav_lat` carry known names but not that shape, so they
must still fall through to the explicit error rather than produce a transform
built from two cells of a raster of positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._axes import X_AXIS_NAMES, Y_AXIS_NAMES
from pyramids.dataset.cog.facade import _dataarray_to_dataset, _first_1d_coord

try:
    import xarray as xr
except ImportError:  # pragma: no cover - exercised only without the extra
    xr = None

needs_xarray = pytest.mark.skipif(xr is None, reason="requires xarray")

pytestmark = pytest.mark.core


def _grid(x_name: str, y_name: str):
    """A small 2-D DataArray with its axes under the given names."""
    return xr.DataArray(
        np.arange(24, dtype="float32").reshape(4, 6),
        dims=(y_name, x_name),
        coords={
            y_name: np.linspace(50.0, 47.0, 4),
            x_name: np.linspace(3.0, 8.0, 6),
        },
    )


class TestFirstOneDimensionalCoordinate:
    """The lookup itself, without building a whole dataset."""

    @needs_xarray
    @pytest.mark.parametrize(
        ("x_name", "y_name"),
        [
            ("x", "y"),
            ("lon", "lat"),
            ("longitude", "latitude"),
            ("rlon", "rlat"),
            ("easting", "northing"),
            ("xc", "yc"),
        ],
    )
    def test_it_finds_every_shared_spelling(self, x_name: str, y_name: str):
        """The last three were refused before, for their names alone.

        Args:
            x_name: The x-axis coordinate name.
            y_name: The y-axis coordinate name.

        Test scenario:
            Each pair is a real convention -- CF, rotated-pole, projected. All
            are already understood elsewhere in pyramids, so this reader has to
            understand them too.
        """
        da = _grid(x_name, y_name)

        assert _first_1d_coord(da, X_AXIS_NAMES) == x_name
        assert _first_1d_coord(da, Y_AXIS_NAMES) == y_name

    @needs_xarray
    def test_a_two_dimensional_coordinate_does_not_qualify(self):
        """The condition the shared vocabulary does not carry.

        Test scenario:
            A curvilinear store names its 2-D position fields `nav_lon` /
            `nav_lat`, which are in the shared vocabulary. Accepting one here
            would build a geotransform from `x[1] - x[0]` across a raster of
            positions -- a plausible-looking number that is not a cell size.
        """
        da = xr.DataArray(
            np.zeros((4, 6), dtype="float32"),
            dims=("j", "i"),
            coords={
                "nav_lon": (("j", "i"), np.zeros((4, 6))),
                "nav_lat": (("j", "i"), np.zeros((4, 6))),
            },
        )

        assert _first_1d_coord(da, X_AXIS_NAMES) is None

    @needs_xarray
    def test_an_array_with_no_spatial_coordinates_yields_none(self):
        """The caller turns this into its own error message."""
        da = xr.DataArray(np.zeros((2, 2)), dims=("a", "b"))

        assert _first_1d_coord(da, X_AXIS_NAMES) is None


class TestConvertingTheDataArray:
    """End to end, through the facade's own normalisation."""

    @needs_xarray
    @pytest.mark.parametrize(
        ("x_name", "y_name"),
        [("x", "y"), ("rlon", "rlat"), ("easting", "northing")],
    )
    def test_a_grid_under_any_spelling_becomes_a_dataset(self, x_name, y_name):
        """The two unusual pairs used to raise instead.

        Args:
            x_name: The x-axis coordinate name.
            y_name: The y-axis coordinate name.

        Test scenario:
            The resulting dataset must carry the grid's real shape and cell
            size regardless of what the coordinates were called.
        """
        dataset = _dataarray_to_dataset(_grid(x_name, y_name), crs=4326, nodata=None)

        assert dataset.shape[-2:] == (4, 6)
        assert dataset.cell_size == pytest.approx(1.0)

    @needs_xarray
    def test_a_curvilinear_array_is_still_refused_clearly(self):
        """A wrong transform would be worse than a refusal.

        Test scenario:
            2-D coordinates cannot describe an affine grid. The reader has to
            say so and point at the way round it, not guess.
        """
        da = xr.DataArray(
            np.zeros((4, 6), dtype="float32"),
            dims=("j", "i"),
            coords={
                "nav_lon": (("j", "i"), np.zeros((4, 6))),
                "nav_lat": (("j", "i"), np.zeros((4, 6))),
            },
        )

        with pytest.raises(ValueError, match="Could not find 1-D"):
            _dataarray_to_dataset(da, crs=4326, nodata=None)
