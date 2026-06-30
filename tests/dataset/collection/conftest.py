"""Shared fixtures and constants for tests/dataset/collection/ tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

# Shared NetCDF path (time-coordinate, dims: time, pressure_level, lat, lon).
NC_FIXTURE = "tests/data/netcdf/pyramids-netcdf-4d.nc"


@pytest.fixture
def three_files(tmp_path):
    """Three GeoTIFFs filled with value i (0, 1, 2), shape 4x5, top-left (0, 4)."""
    paths = []
    for i in range(3):
        arr = np.full((4, 5), i, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.fixture
def three_files_ramp(tmp_path):
    """Three GeoTIFFs filled with value i+1 (1, 2, 3), shape 3x4, top-left (0, 3)."""
    paths = []
    for i in range(3):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths
