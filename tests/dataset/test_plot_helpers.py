"""Unit tests for `pyramids.dataset._plot_helpers` (no cleopatra needed).

Covers `_unwrap_geographic_longitude`, the upstream fix for the curvilinear
antimeridian-seam smear (#669 / serapeum-org/cleopatra#179): a wrapping
geographic longitude is made continuous before it reaches cleopatra, while
non-geographic, unknown-CRS, and non-wrapping coords are left untouched.
"""
import numpy as np

from pyramids.dataset._plot_helpers import (
    _is_degree_geographic,
    _unwrap_geographic_longitude,
)


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


def test_interior_nan_does_not_propagate():
    """A NaN in the longitude stays put; the unwrap never spreads it down the row."""
    lon = np.array(
        [[330.0, 340.0, 350.0, 0.0, 10.0], [330.0, 340.0, np.nan, 0.0, 10.0]]
    )
    lat = np.zeros_like(lon)
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert np.array_equal(np.isnan(np.asarray(x)), np.isnan(lon))


def test_input_float32_longitude_stays_float32():
    """Unwrapping preserves the incoming float precision instead of upcasting to float64."""
    lon = (np.arange(-30, 40, 10.0) % 360.0).astype(np.float32)
    lon = np.tile(lon, (4, 1))
    lat = np.zeros_like(lon)
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert np.asarray(x).dtype == np.float32


def test_is_degree_geographic_gate():
    """`_is_degree_geographic` accepts EPSG:4326, rejects a projected CRS and an unresolvable code."""
    assert _is_degree_geographic(4326) is True
    assert _is_degree_geographic(3857) is False
    assert _is_degree_geographic(999999) is False  # unassigned code -> CRSError -> False


def test_row_wrapping_more_than_once_is_fully_unwrapped():
    """A row that wraps twice (>720 deg) accumulates the wrap count and unwraps cleanly."""
    lon = np.tile(np.array([0.0, 120.0, 240.0, 0.0, 120.0, 240.0, 0.0]), (3, 1))
    lat = np.zeros_like(lon)
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert _max_adjacent_step(x) < 180.0


def test_single_column_grid_is_noop():
    """A single-column grid has no horizontal neighbours, so it is returned unchanged."""
    lon = np.array([[10.0], [20.0], [30.0]])
    lat = np.zeros_like(lon)
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    assert x is lon


def test_interior_nan_off_seam_still_unwraps_the_row():
    """A NaN away from the seam is preserved while the row's real wrap is still unwrapped."""
    lon = np.array(
        [[330.0, 340.0, 350.0, 0.0, 10.0], [330.0, np.nan, 350.0, 0.0, 10.0]]
    )
    lat = np.zeros_like(lon)
    x, _ = _unwrap_geographic_longitude((lon, lat), 4326)
    x = np.asarray(x)
    assert np.array_equal(np.isnan(x), np.isnan(lon))  # NaN stays put
    assert x[1, 3] == 360.0 and x[1, 4] == 370.0  # seam still unwrapped around the NaN
