"""Unit tests for pyramids.netcdf.ugrid.io.

Covers topology detection (parse_ugrid_topology), single topology
parsing, and CRS detection using the UGRID convention NC test file.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.io import (
    _MeshArrayScan,
    _detect_face_dim,
    _parse_single_topology,
    parse_ugrid_topology,
    write_ugrid_topology,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.utils import _read_attributes

pytestmark = pytest.mark.core


class TestParseUgridTopology:
    """Tests for parse_ugrid_topology() function."""

    @pytest.fixture
    def ugrid_convention_nc_rg(self, ugrid_convention_nc_path):
        """Open the UGRID convention NC file and return the root group.

        Returns:
            GDAL root group for the UGRID test file.
        """
        ds = gdal.OpenEx(str(ugrid_convention_nc_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        return rg

    def test_detects_mesh_topology(self, ugrid_convention_nc_rg):
        """Test that topology detection finds the mesh2d topology.

        Test scenario:
            The UGRID convention NC file has a single mesh2d topology
            that should be detected.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        assert (
            len(topologies) >= 1
        ), f"Expected at least 1 topology, got {len(topologies)}"
        topo = topologies[0]
        assert (
            topo.mesh_name == "mesh2d"
        ), f"Expected mesh_name 'mesh2d', got '{topo.mesh_name}'"

    def test_topology_dimension(self, ugrid_convention_nc_rg):
        """Test that topology_dimension is 2 for the UGRID convention NC mesh.

        Test scenario:
            The UGRID convention NC is a 2D surface mesh.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert (
            topo.topology_dimension == 2
        ), f"Expected topology_dimension 2, got {topo.topology_dimension}"

    def test_node_coordinate_vars(self, ugrid_convention_nc_rg):
        """Test that node coordinate variable names are parsed.

        Test scenario:
            UGRID convention NC should have mesh2d_node_x and mesh2d_node_y.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert (
            topo.node_x_var == "mesh2d_node_x"
        ), f"Expected node_x_var 'mesh2d_node_x', got '{topo.node_x_var}'"
        assert (
            topo.node_y_var == "mesh2d_node_y"
        ), f"Expected node_y_var 'mesh2d_node_y', got '{topo.node_y_var}'"

    def test_face_node_connectivity_var(self, ugrid_convention_nc_rg):
        """Test that face_node_connectivity variable name is parsed.

        Test scenario:
            UGRID convention NC should have mesh2d_face_nodes.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert (
            topo.face_node_var == "mesh2d_face_nodes"
        ), f"Expected 'mesh2d_face_nodes', got '{topo.face_node_var}'"

    def test_edge_node_connectivity_var(self, ugrid_convention_nc_rg):
        """Test that edge_node_connectivity variable name is parsed.

        Test scenario:
            UGRID convention NC should have mesh2d_edge_nodes.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert (
            topo.edge_node_var == "mesh2d_edge_nodes"
        ), f"Expected 'mesh2d_edge_nodes', got '{topo.edge_node_var}'"

    def test_data_variables_detected(self, ugrid_convention_nc_rg):
        """Test that data variables with mesh= attribute are detected.

        Test scenario:
            UGRID convention NC should detect mesh2d_node_z (node)
            and mesh2d_edge_type (edge).
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert (
            len(topo.data_variables) >= 2
        ), f"Expected at least 2 data variables, got {len(topo.data_variables)}"
        assert (
            "mesh2d_node_z" in topo.data_variables
        ), f"Expected 'mesh2d_node_z' in data_variables, got {list(topo.data_variables.keys())}"
        assert (
            topo.data_variables["mesh2d_node_z"] == "node"
        ), f"Expected location 'node', got '{topo.data_variables['mesh2d_node_z']}'"

    def test_face_coordinates_detected(self, ugrid_convention_nc_rg):
        """Test that face center coordinate variables are detected.

        Test scenario:
            UGRID convention NC should have face center coordinates.
        """
        topologies = parse_ugrid_topology(ugrid_convention_nc_rg)
        topo = topologies[0]
        assert topo.face_x_var is not None, "Expected face_x_var to be set"
        assert topo.face_y_var is not None, "Expected face_y_var to be set"

    def test_no_topology_returns_empty(self):
        """Test that a non-UGRID structured NetCDF file returns empty list.

        Test scenario:
            The Noah precipitation file is a regular structured NetCDF
            without mesh_topology — should return [].
        """
        nc_path = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"
        ds = gdal.OpenEx(str(nc_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        assert topologies == [], f"Expected empty list, got {topologies}"


class TestTopologyParsingEdgeCases:
    """Tests for topology parsing edge cases (Issue #4)."""

    def test_optional_connectivity_detected(self, ugrid_convention_nc_path):
        """Test that optional connectivity variables are parsed.

        Test scenario:
            UGRID convention NC has edge_node_connectivity. Verify it is
            detected and stored in the topology info.
        """
        ds = gdal.OpenEx(str(ugrid_convention_nc_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        topo = topologies[0]
        assert (
            topo.edge_node_var is not None
        ), "Expected edge_node_connectivity to be detected"

    def test_data_variable_locations(self, ugrid_convention_nc_path):
        """Test that data variable locations are correctly parsed.

        Test scenario:
            mesh2d_node_z should be 'node', mesh2d_edge_type should be 'edge'.
        """
        ds = gdal.OpenEx(str(ugrid_convention_nc_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        topo = topologies[0]
        assert (
            topo.data_variables.get("mesh2d_node_z") == "node"
        ), f"Expected 'node', got '{topo.data_variables.get('mesh2d_node_z')}'"
        assert (
            topo.data_variables.get("mesh2d_edge_type") == "edge"
        ), f"Expected 'edge', got '{topo.data_variables.get('mesh2d_edge_type')}'"

    def test_parse_single_topology_no_topo_dim_returns_none(
        self, ugrid_convention_nc_path
    ):
        """Test that a variable without topology_dimension returns None.

        Test scenario:
            Open a data variable (not topology) and try to parse it.
            Should return None.
        """

        ds = gdal.OpenEx(str(ugrid_convention_nc_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray("mesh2d_node_x")
        result = _parse_single_topology(_MeshArrayScan(rg), "mesh2d_node_x", md_arr)
        assert result is None, f"Expected None for non-topology variable, got {result}"


class TestWriteUgridTopology:
    """Direct unit tests for write_ugrid_topology() (H6)."""

    def _write_and_reopen(self, tmp_path, mesh, mesh_name="mesh2d", crs_wkt=None):
        """Helper: write via UgridDataset.to_file and reopen with GDAL.

        Args:
            tmp_path: pytest tmp_path fixture.
            mesh: Mesh2d instance to write.
            mesh_name: Topology variable name.
            crs_wkt: Optional CRS WKT string.

        Returns:
            Tuple of (nc_path, gdal_root_group).
        """
        from pyramids.netcdf.ugrid.dataset import UgridDataset
        from pyramids.netcdf.ugrid.models import MeshTopologyInfo

        topo = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=2,
            node_x_var=f"{mesh_name}_node_x",
            node_y_var=f"{mesh_name}_node_y",
            face_node_var=f"{mesh_name}_face_nodes",
            crs_wkt=crs_wkt,
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={},
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=topo,
            crs_wkt=crs_wkt,
        )
        nc_path = tmp_path / f"{mesh_name}.nc"
        ds.to_file(nc_path)

        ds2 = gdal.OpenEx(str(nc_path), gdal.OF_MULTIDIM_RASTER)
        rg2 = ds2.GetRootGroup()
        return nc_path, rg2

    def test_writes_topology_variable_with_cf_role(self, tmp_path):
        """Test that the topology variable has cf_role=mesh_topology.

        Test scenario:
            Write a simple mesh, reopen with raw GDAL, verify cf_role.
        """
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        _, rg = self._write_and_reopen(tmp_path, mesh)
        topo_arr = rg.OpenMDArray("mesh2d")
        attrs = _read_attributes(topo_arr)
        assert (
            attrs.get("cf_role") == "mesh_topology"
        ), f"Expected cf_role='mesh_topology', got '{attrs.get('cf_role')}'"
        assert (
            attrs.get("topology_dimension") == 2
        ), f"Expected topology_dimension=2, got {attrs.get('topology_dimension')}"

    def test_writes_node_coordinates(self, tmp_path):
        """Test that node coordinate arrays are written correctly.

        Test scenario:
            Write mesh with known node coords, verify raw values.
        """
        node_x = np.array([0.0, 1.0, 0.5])
        mesh = Mesh2d(
            node_x=node_x,
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        _, rg = self._write_and_reopen(tmp_path, mesh)
        x_arr = rg.OpenMDArray("mesh2d_node_x")
        np.testing.assert_array_almost_equal(
            x_arr.ReadAsArray(),
            node_x,
            err_msg="Node x-coordinates should match",
        )

    def test_writes_connectivity_with_fill_and_start_index(self, tmp_path):
        """Test that connectivity arrays have correct attributes.

        Test scenario:
            Write mixed connectivity, verify _FillValue and start_index.
        """
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 4, 3], [1, 2, 5, -1]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        _, rg = self._write_and_reopen(tmp_path, mesh)
        fnc_arr = rg.OpenMDArray("mesh2d_face_nodes")
        attrs = _read_attributes(fnc_arr)
        assert (
            attrs.get("cf_role") == "face_node_connectivity"
        ), f"Expected cf_role 'face_node_connectivity', got '{attrs.get('cf_role')}'"
        assert (
            attrs.get("start_index") == 0
        ), f"Expected start_index=0, got {attrs.get('start_index')}"
        raw_data = fnc_arr.ReadAsArray()
        assert (
            raw_data[1, 3] == -999
        ), f"Expected fill value -999 at [1,3], got {raw_data[1, 3]}"

    def test_writes_crs_variable(self, tmp_path):
        """Test that CRS variable is written when crs_wkt provided.

        Test scenario:
            Write mesh with CRS WKT, verify crs variable has crs_wkt attr.
        """
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        crs_wkt = srs.ExportToWkt()

        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        _, rg = self._write_and_reopen(tmp_path, mesh, crs_wkt=crs_wkt)
        crs_arr = rg.OpenMDArray("crs")
        assert crs_arr is not None, "CRS variable should exist"
        crs_attrs = _read_attributes(crs_arr)
        assert "crs_wkt" in crs_attrs, "CRS variable should have crs_wkt attribute"


def _face_node_group(*, reversed_order: bool, with_face_coord: bool):
    """Build an in-memory multidim group with a ``face_node_connectivity`` for face-dim detection.

    Args:
        reversed_order: When True the connectivity is stored ``(max_nodes, n_face)`` (the reverse of
            the conventional ``(n_face, max_nodes)``), so the face dim is the *second* connectivity
            dimension.
        with_face_coord: When True a 1-D ``face_x`` variable with ``standard_name=longitude`` is added
            on the face dimension (the face-located signal :func:`_detect_face_dim` keys off).

    Returns:
        tuple: ``(dataset, root_group, conn_dims, all_array_names)``. The dataset is returned so the
            caller keeps it alive (the root group dangles otherwise).
    """
    ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("mesh")
    rg = ds.GetRootGroup()
    dim_face = rg.CreateDimension("n_face", "", "", 5)
    dim_nodes = rg.CreateDimension("max_face_nodes", "", "", 4)
    conn_order = [dim_nodes, dim_face] if reversed_order else [dim_face, dim_nodes]
    conn = rg.CreateMDArray(
        "face_nodes", conn_order, gdal.ExtendedDataType.Create(gdal.GDT_Int32)
    )
    role = conn.CreateAttribute("cf_role", [], gdal.ExtendedDataType.CreateString())
    role.WriteString("face_node_connectivity")
    if with_face_coord:
        face_x = rg.CreateMDArray(
            "face_x", [dim_face], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        std = face_x.CreateAttribute(
            "standard_name", [], gdal.ExtendedDataType.CreateString()
        )
        std.WriteString("longitude")
    conn_dims = [d.GetName() for d in conn.GetDimensions()]
    return ds, rg, conn_dims, (rg.GetMDArrayNames() or [])


class TestDetectFaceDim:
    """Tests for _detect_face_dim() connectivity face-dimension detection."""

    def test_detects_face_dim_when_connectivity_reversed(self):
        """A face-located variable identifies the face dim even when connectivity is reversed.

        Test scenario:
            Connectivity stored as (max_face_nodes, n_face) with a 1-D face_x (standard_name
            longitude) on n_face. Detection must return 'n_face', NOT conn_dims[0]
            ('max_face_nodes').
        """
        _, rg, conn_dims, _ = _face_node_group(reversed_order=True, with_face_coord=True)
        assert conn_dims[0] == "max_face_nodes", f"fixture not reversed: {conn_dims}"
        face_dim = _detect_face_dim(_MeshArrayScan(rg), conn_dims)
        assert (
            face_dim == "n_face"
        ), f"expected 'n_face' from the face coordinate variable, got {face_dim!r}"

    def test_falls_back_to_first_conn_dim_without_face_signal(self):
        """With no face-located variable, detection falls back to the first connectivity dim.

        Test scenario:
            Connectivity stored (n_face, max_face_nodes) and no face coordinate / location=face
            variable — detection returns conn_dims[0] (the conventional (face, node) order).
        """
        _, rg, conn_dims, _ = _face_node_group(reversed_order=False, with_face_coord=False)
        face_dim = _detect_face_dim(_MeshArrayScan(rg), conn_dims)
        assert (
            face_dim == conn_dims[0]
        ), f"expected fallback to conn_dims[0]={conn_dims[0]!r}, got {face_dim!r}"
