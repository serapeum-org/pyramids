"""Tests for GCP/RPC georeferencing (the Georef engine, issue-driven GR-* tasks).

Fixtures are synthetic and offline: a small in-memory raster with four corner
ground-control points, generated via ``set_gcps`` rather than shipping a binary.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.dataset._gcp import GroundControlPoint

pytestmark = pytest.mark.core


@pytest.fixture
def corner_gcps() -> list[GroundControlPoint]:
    """Four corner control points of an 8x8 raster in EPSG:4326.

    Returns:
        list[GroundControlPoint]: top-left, top-right, bottom-left, bottom-right.
    """
    return [
        GroundControlPoint(row=0, col=0, x=10.0, y=50.0, id="tl"),
        GroundControlPoint(row=0, col=8, x=11.0, y=50.0, id="tr"),
        GroundControlPoint(row=8, col=0, x=10.0, y=49.0, id="bl"),
        GroundControlPoint(row=8, col=8, x=11.0, y=49.0, id="br"),
    ]


@pytest.fixture
def writable_dataset() -> Dataset:
    """A writable in-memory 8x8 float32 dataset.

    Returns:
        Dataset: MEM-backed (always writable), no GCPs yet.
    """
    return Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
    )


class TestGroundControlPoint:
    """Tests for the GroundControlPoint value object."""

    def test_to_gdal_maps_pixel_and_map_coords(self):
        """`to_gdal` puts col/row on pixel/line and keeps the map coordinate.

        Test scenario:
            A point at (col=7, row=3) -> (x=1, y=2) becomes a gdal.GCP with the
            same pixel/line and X/Y.
        """
        g = GroundControlPoint(row=3.0, col=7.0, x=1.0, y=2.0).to_gdal()
        assert (g.GCPPixel, g.GCPLine, g.GCPX, g.GCPY) == (7.0, 3.0, 1.0, 2.0)

    def test_round_trip_through_gdal(self):
        """`from_gdal(to_gdal())` preserves all fields.

        Test scenario:
            A fully-populated point survives the GDAL round-trip.
        """
        original = GroundControlPoint(
            row=4.0, col=2.0, x=11.5, y=46.2, z=3.0, id="p1", info="note"
        )
        back = GroundControlPoint.from_gdal(original.to_gdal())
        assert back == original

    def test_empty_id_info_become_none(self):
        """Empty GDAL Id/Info come back as None, not empty strings.

        Test scenario:
            A point with no id/info round-trips to None id/info.
        """
        back = GroundControlPoint.from_gdal(
            GroundControlPoint(row=0, col=0, x=0.0, y=0.0).to_gdal()
        )
        assert back.id is None and back.info is None


class TestSetGCPs:
    """Tests for Georef.set_gcps (and the Dataset facade)."""

    def test_attaches_gcps_and_projection(self, writable_dataset, corner_gcps):
        """set_gcps writes the points and an EPSG:4326 projection to the raster.

        Test scenario:
            After set_gcps the underlying GDAL dataset reports 4 GCPs and a
            4326 projection.
        """
        writable_dataset.set_gcps(corner_gcps, 4326)
        raster = writable_dataset.raster
        assert raster.GetGCPCount() == 4
        assert "4326" in raster.GetGCPProjection()

    def test_empty_list_raises_value_error(self, writable_dataset):
        """An empty GCP list is rejected.

        Test scenario:
            set_gcps([], 4326) raises ValueError.
        """
        with pytest.raises(ValueError, match="at least one"):
            writable_dataset.set_gcps([], 4326)

    def test_read_only_raises(self, corner_gcps, tmp_path):
        """A read-only dataset rejects set_gcps.

        Test scenario:
            A dataset opened read_only=True raises ReadOnlyError.
        """
        path = tmp_path / "plain.tif"
        Dataset.create_from_array(
        np.ones((8, 8), dtype="float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
    ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.set_gcps(corner_gcps, 4326)
