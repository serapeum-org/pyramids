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

from pyramids.base._axes import (
    X_AXIS_NAMES,
    X_AXIS_NAMES_ORDERED,
    Y_AXIS_NAMES,
    Y_AXIS_NAMES_ORDERED,
)
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


def _utm_grid_with_degree_aliases():
    """A 30 m UTM grid that also carries auxiliary 1-D lon/lat coordinates."""
    return xr.DataArray(
        np.zeros((4, 6), dtype="float32"),
        dims=("y", "x"),
        coords={
            "y": np.linspace(4_600_000, 4_599_910, 4),
            "x": np.linspace(500_000, 500_150, 6),
            "lon": ("x", np.linspace(32.9, 32.902, 6)),
            "lat": ("y", np.linspace(41.5, 41.4992, 4)),
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

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) == x_name
        assert _first_1d_coord(da, Y_AXIS_NAMES_ORDERED) == y_name

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

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) is None

    @needs_xarray
    def test_an_array_with_no_spatial_coordinates_yields_none(self):
        """The caller turns this into its own error message."""
        da = xr.DataArray(np.zeros((2, 2)), dims=("a", "b"))

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) is None


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


class TestPreferenceOrderWhenSeveralCoordinatesMatch:
    """The order is load-bearing, not incidental.

    A projected grid whose axes are `x` / `y` in metres commonly also carries
    auxiliary 1-D `lon` / `lat` in degrees. Choosing between them alphabetically
    picks `lon`, builds a geotransform in degrees, and stamps the projected CRS
    on it -- a silently mis-georeferenced raster rather than an error.
    """

    @needs_xarray
    def test_the_grid_dimension_wins_over_an_auxiliary_alias(self):
        """`x` outranks `lon`, which sorting alphabetically reverses.

        Test scenario:
            A UTM grid of 30 m cells with auxiliary degree coordinates. The
            lookup must return `x` / `y`; returning `lon` / `lat` yields a
            0.0004 cell size labelled EPSG:32636.
        """
        da = _utm_grid_with_degree_aliases()

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) == "x"
        assert _first_1d_coord(da, Y_AXIS_NAMES_ORDERED) == "y"

    @needs_xarray
    def test_the_resulting_grid_is_in_the_crs_units_it_was_given(self):
        """The consequence, measured on the dataset rather than the name.

        Test scenario:
            The cell size must come back as 30 -- metres, matching EPSG:32636 --
            not as a fraction of a degree.
        """
        dataset = _dataarray_to_dataset(
            _utm_grid_with_degree_aliases(), crs=32636, nodata=None
        )

        assert dataset.cell_size == pytest.approx(30.0)
        assert dataset.epsg == 32636

    @needs_xarray
    def test_a_projected_alias_also_outranks_the_geographic_one(self):
        """`easting` is the same axis as `x`, and equally not `lon`.

        Test scenario:
            Some projected stores name their axes `easting` / `northing`. Those
            must win over auxiliary `lon` / `lat` too.
        """
        da = xr.DataArray(
            np.zeros((4, 6), dtype="float32"),
            dims=("northing", "easting"),
            coords={
                "northing": np.linspace(4_600_000, 4_599_910, 4),
                "easting": np.linspace(500_000, 500_150, 6),
                "lon": ("easting", np.linspace(32.9, 32.902, 6)),
                "lat": ("northing", np.linspace(41.5, 41.4992, 4)),
            },
        )

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) == "easting"
        assert _first_1d_coord(da, Y_AXIS_NAMES_ORDERED) == "northing"

    @needs_xarray
    def test_a_geographic_grid_with_only_aliases_still_resolves(self):
        """Preferring `x` must not mean requiring it.

        Test scenario:
            A plain lat/lon store has no `x` at all. The first name it does
            carry has to be chosen.
        """
        da = _grid("longitude", "latitude")

        assert _first_1d_coord(da, X_AXIS_NAMES_ORDERED) == "longitude"
        assert _first_1d_coord(da, Y_AXIS_NAMES_ORDERED) == "latitude"

    def test_the_two_sequences_match_their_sets(self):
        """The sets are derived from the sequences, so they cannot diverge."""
        assert frozenset(X_AXIS_NAMES_ORDERED) == X_AXIS_NAMES
        assert frozenset(Y_AXIS_NAMES_ORDERED) == Y_AXIS_NAMES
        assert len(X_AXIS_NAMES_ORDERED) == len(X_AXIS_NAMES)
        assert len(Y_AXIS_NAMES_ORDERED) == len(Y_AXIS_NAMES)
