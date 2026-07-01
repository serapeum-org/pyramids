"""Unit tests for `pyramids.dataset._plot_helpers` (no cleopatra needed).

Covers `_unwrap_geographic_longitude`, the upstream fix for the curvilinear
antimeridian-seam smear (#669 / serapeum-org/cleopatra#179): a wrapping
geographic longitude is made continuous before it reaches cleopatra, while
non-geographic, unknown-CRS, and non-wrapping coords are left untouched.
"""
import numpy as np

from pyramids.dataset._plot_helpers import _unwrap_geographic_longitude


def _wrapping_grid():
    """A 2-D lon/lat grid whose rows cross the 0/360 seam (350 -> 0 jump)."""
    lon1d = np.arange(-30, 40, 10.0) % 360.0  # [330 340 350 0 10 20 30]
    lon = np.tile(lon1d, (6, 1))
    lat = np.tile(np.linspace(-20, 20, 6)[:, None], (1, lon1d.size))
    return lon, lat


def _max_adjacent_step(lon):
    """Largest longitude gap between horizontally-adjacent cells."""
    return float(np.nanmax(np.abs(np.diff(np.asarray(lon), axis=-1))))


def test_unwrap_removes_the_antimeridian_seam():
    """A wrapping geographic longitude is unwrapped so no adjacent gap exceeds 180 degrees."""
    lon, lat = _wrapping_grid()
    assert _max_adjacent_step(lon) > 180.0  # the seam is present before
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert _max_adjacent_step(x) < 180.0


def test_unwrap_preserves_latitude():
    """Unwrapping the longitude leaves the latitude coordinate unchanged."""
    lon, lat = _wrapping_grid()
    _, y = _unwrap_geographic_longitude((lon, lat), 4326)
    assert np.array_equal(np.asarray(y), lat)


def test_projected_crs_is_left_untouched():
    """A projected (non-geographic) CRS is never unwrapped, even if x jumps > 180."""
    lon, lat = _wrapping_grid()
    x, _ = _unwrap_geographic_longitude((lon, lat), 3857)
    assert x is lon


def test_unknown_crs_is_left_untouched():
    """With no CRS (epsg is None) pyramids does not assume degrees, so coords pass through."""
    lon, lat = _wrapping_grid()
    result = _unwrap_geographic_longitude((lon, lat), None)
    assert result[0] is lon


def test_nonwrapping_geographic_grid_is_left_untouched():
    """A geographic grid with no seam (all steps <= 180) is returned unchanged."""
    lon = np.tile(np.linspace(-40, 40, 20), (5, 1))
    lat = np.tile(np.linspace(-10, 10, 5)[:, None], (1, 20))
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert x is lon


def test_none_coords_pass_through():
    """`None` coords (non-curvilinear plot) are returned as-is."""
    assert _unwrap_geographic_longitude(None, 4326) is None
