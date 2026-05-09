"""Integration tests for the UGRID subpackage.

Tests full lifecycle chains: create -> write -> read -> clip ->
interpolate -> Dataset to verify cross-module interactions.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.ugrid.dataset import UgridDataset

pytestmark = pytest.mark.core


class TestFullLifecycle:
    """Test complete create -> write -> read -> clip -> interpolate chain."""

    def test_create_write_read_clip_interpolate(self, tmp_path):
        """Test the full UGRID lifecycle.

        Test scenario:
            1. Create a 4-face mesh from arrays with face data.
            2. Write to NetCDF.
            3. Read back from file.
            4. Clip to a polygon.
            5. Interpolate to a regular grid (Dataset).
            6. Verify the Dataset has valid data.
        """

        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            face_node_connectivity=np.array(
                [
                    [0, 1, 4, 3],
                    [1, 2, 5, 4],
                    [3, 4, 7, 6],
                    [4, 5, 8, 7],
                ]
            ),
            data={"temperature": np.array([10.0, 20.0, 30.0, 40.0])},
            data_locations={"temperature": "face"},
            epsg=32631,
        )
        assert ds.n_face == 4, f"Step 1 failed: expected 4 faces, got {ds.n_face}"

        out_path = tmp_path / "lifecycle.nc"
        ds.to_file(out_path)
        assert out_path.exists(), "Step 2 failed: file not created"

        ds2 = UgridDataset.read_file(out_path)
        assert ds2.n_face == 4, f"Step 3 failed: expected 4 faces, got {ds2.n_face}"
        assert (
            "temperature" in ds2.data_variable_names
        ), "Step 3 failed: temperature variable missing"

        mask = box(-0.1, -0.1, 1.1, 2.1)
        clipped = ds2.clip(mask, touch=False)
        assert clipped.n_face > 0, "Step 4 failed: clip produced 0 faces"
        assert (
            clipped.n_face < ds2.n_face
        ), f"Step 4 failed: clip should reduce faces: {clipped.n_face} vs {ds2.n_face}"

        raster = clipped.to_dataset("temperature", cell_size=0.5)
        assert (
            raster.rows > 0
        ), f"Step 5 failed: expected positive rows, got {raster.rows}"
        assert (
            raster.columns > 0
        ), f"Step 5 failed: expected positive cols, got {raster.columns}"

    def test_create_geodataframe_round_trip(self):
        """Test create -> to_geodataframe -> verify geometry.

        Test scenario:
            Create mesh, convert to GeoDataFrame, verify polygon count
            and data values match.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.5])},
            data_locations={"depth": "face"},
        )
        gdf = ds.to_geodataframe("depth", location="face")
        assert len(gdf) == 1, f"Expected 1 row, got {len(gdf)}"
        assert (
            gdf["depth"].iloc[0] == 5.5
        ), f"Expected depth 5.5, got {gdf['depth'].iloc[0]}"
        assert (
            gdf.geometry.iloc[0].geom_type == "Polygon"
        ), f"Expected Polygon, got {gdf.geometry.iloc[0].geom_type}"

    def test_reproject_then_interpolate(self):
        """Test CRS reprojection followed by interpolation.

        Test scenario:
            Create mesh in UTM, reproject to WGS84, then interpolate
            to a regular grid.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"salinity": np.array([35.0])},
            data_locations={"salinity": "face"},
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected.epsg == 4326, f"Expected EPSG 4326, got {reprojected.epsg}"

        raster = reprojected.to_dataset("salinity", cell_size=0.0005)
        assert raster.rows > 0, f"Expected positive rows, got {raster.rows}"

    def test_ugrid_convention_nc_full_pipeline(self, ugrid_convention_nc_path):
        """Test full pipeline on the UGRID convention NC real dataset.

        Test scenario:
            Read -> subset -> to_dataset -> verify raster dimensions.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)

        xmin, ymin, xmax, ymax = ds.bounds
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        subset = ds.subset_by_bounds(xmin, ymin, mid_x, mid_y)
        assert subset.n_face > 0, "Subset should have faces"

        raster = subset.to_dataset("mesh2d_node_z", cell_size=500.0)
        assert raster.rows > 0, f"Expected positive rows, got {raster.rows}"
        arr = raster.read_array()
        valid_count = np.sum(arr != -9999.0)
        assert valid_count > 0, "Raster should have some valid (non-nodata) cells"
