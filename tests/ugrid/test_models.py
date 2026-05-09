"""Unit tests for pyramids.netcdf.ugrid.models.

Covers MeshTopologyInfo, MeshVariable, and UgridMetadata dataclasses
with comprehensive scenario coverage. Targets >=95% line+branch coverage.

Style: Google-style docstrings, <=120 char lines, descriptive assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable, UgridMetadata

pytestmark = pytest.mark.core


class TestMeshTopologyInfo:
    """Tests for the MeshTopologyInfo frozen dataclass."""

    def test_init_required_fields_only(self):
        """Test construction with only required fields.

        Test scenario:
            Minimum valid MeshTopologyInfo with mesh_name, topology_dimension,
            node_x_var, and node_y_var. All optional fields should default to None
            or empty dict.
        """
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="node_x",
            node_y_var="node_y",
        )
        assert (
            topo.mesh_name == "mesh2d"
        ), f"Expected mesh_name 'mesh2d', got '{topo.mesh_name}'"
        assert (
            topo.topology_dimension == 2
        ), f"Expected topology_dimension 2, got {topo.topology_dimension}"
        assert (
            topo.node_x_var == "node_x"
        ), f"Expected node_x_var 'node_x', got '{topo.node_x_var}'"
        assert (
            topo.node_y_var == "node_y"
        ), f"Expected node_y_var 'node_y', got '{topo.node_y_var}'"
        assert (
            topo.face_node_var is None
        ), f"Expected face_node_var None, got {topo.face_node_var}"
        assert (
            topo.edge_node_var is None
        ), f"Expected edge_node_var None, got {topo.edge_node_var}"
        assert (
            topo.face_edge_var is None
        ), f"Expected face_edge_var None, got {topo.face_edge_var}"
        assert (
            topo.face_face_var is None
        ), f"Expected face_face_var None, got {topo.face_face_var}"
        assert (
            topo.edge_face_var is None
        ), f"Expected edge_face_var None, got {topo.edge_face_var}"
        assert (
            topo.boundary_node_var is None
        ), f"Expected boundary_node_var None, got {topo.boundary_node_var}"
        assert (
            topo.face_x_var is None
        ), f"Expected face_x_var None, got {topo.face_x_var}"
        assert (
            topo.face_y_var is None
        ), f"Expected face_y_var None, got {topo.face_y_var}"
        assert (
            topo.edge_x_var is None
        ), f"Expected edge_x_var None, got {topo.edge_x_var}"
        assert (
            topo.edge_y_var is None
        ), f"Expected edge_y_var None, got {topo.edge_y_var}"
        assert (
            topo.data_variables == {}
        ), f"Expected empty data_variables, got {topo.data_variables}"
        assert topo.crs_wkt is None, f"Expected crs_wkt None, got {topo.crs_wkt}"

    def test_init_all_fields(self):
        """Test construction with all fields populated.

        Test scenario:
            Full 2D mesh topology with all connectivity, coordinate,
            data variable, and CRS fields set.
        """
        data_vars = {"water_level": "face", "velocity": "edge"}
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="mesh2d_node_x",
            node_y_var="mesh2d_node_y",
            face_node_var="mesh2d_face_nodes",
            edge_node_var="mesh2d_edge_nodes",
            face_edge_var="mesh2d_face_edges",
            face_face_var="mesh2d_face_faces",
            edge_face_var="mesh2d_edge_faces",
            boundary_node_var="mesh2d_boundary",
            face_x_var="mesh2d_face_x",
            face_y_var="mesh2d_face_y",
            edge_x_var="mesh2d_edge_x",
            edge_y_var="mesh2d_edge_y",
            data_variables=data_vars,
            crs_wkt='GEOGCS["WGS 84"]',
        )
        assert (
            topo.face_node_var == "mesh2d_face_nodes"
        ), f"Expected 'mesh2d_face_nodes', got '{topo.face_node_var}'"
        assert (
            topo.data_variables == data_vars
        ), f"data_variables mismatch: {topo.data_variables}"
        assert topo.crs_wkt == 'GEOGCS["WGS 84"]', f"CRS WKT mismatch: {topo.crs_wkt}"

    def test_frozen_immutability(self):
        """Test that MeshTopologyInfo is immutable (frozen dataclass).

        Test scenario:
            Attempting to set an attribute after construction should raise
            FrozenInstanceError.
        """
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="node_x",
            node_y_var="node_y",
        )
        with pytest.raises(AttributeError):
            topo.mesh_name = "changed"

    def test_equality(self):
        """Test equality comparison between two identical MeshTopologyInfo instances.

        Test scenario:
            Two MeshTopologyInfo objects with the same field values should be equal.
        """
        kwargs = dict(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="node_x",
            node_y_var="node_y",
        )
        topo1 = MeshTopologyInfo(**kwargs)
        topo2 = MeshTopologyInfo(**kwargs)
        assert topo1 == topo2, "Identical MeshTopologyInfo objects should be equal"

    @pytest.mark.parametrize(
        "topo_dim",
        [1, 2, 3],
        ids=["1d-network", "2d-surface", "3d-volume"],
    )
    def test_topology_dimensions(self, topo_dim):
        """Test that all valid topology dimensions are accepted.

        Args:
            topo_dim: Topology dimension value to test.

        Test scenario:
            Topology dimensions 1, 2, and 3 should all be valid.
        """
        topo = MeshTopologyInfo(
            mesh_name="mesh",
            topology_dimension=topo_dim,
            node_x_var="x",
            node_y_var="y",
        )
        assert (
            topo.topology_dimension == topo_dim
        ), f"Expected topology_dimension {topo_dim}, got {topo.topology_dimension}"


class TestMeshVariable:
    """Tests for the MeshVariable dataclass."""

    @pytest.fixture
    def face_var_1d(self):
        """Create a 1D face-centered MeshVariable with eager data.

        Returns:
            MeshVariable with shape (5,) and float64 data.
        """
        return MeshVariable(
            name="water_level",
            location="face",
            mesh_name="mesh2d",
            shape=(5,),
            attributes={"units": "m", "standard_name": "sea_surface_height"},
            nodata=-999.0,
            units="m",
            standard_name="sea_surface_height",
            _data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

    @pytest.fixture
    def temporal_var(self):
        """Create a 2D temporal face-centered MeshVariable.

        Returns:
            MeshVariable with shape (3, 5) — 3 time steps, 5 faces.
        """
        data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.1, 2.1, 3.1, 4.1, 5.1],
                [1.2, 2.2, 3.2, 4.2, 5.2],
            ]
        )
        return MeshVariable(
            name="water_level",
            location="face",
            mesh_name="mesh2d",
            shape=(3, 5),
            _data=data,
        )

    def test_init_with_eager_data(self, face_var_1d):
        """Test construction with eagerly-loaded data.

        Test scenario:
            MeshVariable with _data provided should store it directly.
        """
        assert (
            face_var_1d.name == "water_level"
        ), f"Expected name 'water_level', got '{face_var_1d.name}'"
        assert (
            face_var_1d.location == "face"
        ), f"Expected location 'face', got '{face_var_1d.location}'"
        assert (
            face_var_1d.mesh_name == "mesh2d"
        ), f"Expected mesh_name 'mesh2d', got '{face_var_1d.mesh_name}'"
        assert face_var_1d.shape == (
            5,
        ), f"Expected shape (5,), got {face_var_1d.shape}"
        assert (
            face_var_1d.nodata == -999.0
        ), f"Expected nodata -999.0, got {face_var_1d.nodata}"
        assert (
            face_var_1d.units == "m"
        ), f"Expected units 'm', got '{face_var_1d.units}'"
        assert (
            face_var_1d.standard_name == "sea_surface_height"
        ), f"Expected standard_name 'sea_surface_height', got '{face_var_1d.standard_name}'"

    def test_init_defaults(self):
        """Test construction with minimal required fields.

        Test scenario:
            Optional fields should default to None or empty dict.
        """
        var = MeshVariable(name="v", location="node", mesh_name="m", shape=(10,))
        assert var.attributes == {}, f"Expected empty attributes, got {var.attributes}"
        assert var.nodata is None, f"Expected nodata None, got {var.nodata}"
        assert var.units is None, f"Expected units None, got {var.units}"
        assert (
            var.standard_name is None
        ), f"Expected standard_name None, got {var.standard_name}"
        assert var._data is None, "Expected _data None by default"
        assert var._loader is None, "Expected _loader None by default"

    def test_data_property_eager(self, face_var_1d):
        """Test data property returns eager data directly.

        Test scenario:
            When _data is set, data property should return it without calling loader.
        """
        result = face_var_1d.data
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_data_property_lazy_load(self):
        """Test data property triggers lazy loading via _loader.

        Test scenario:
            When _data is None and _loader is set, accessing .data should
            call the loader exactly once and cache the result.
        """
        call_count = 0
        lazy_data = np.array([10.0, 20.0, 30.0])

        def loader():
            nonlocal call_count
            call_count += 1
            return lazy_data

        var = MeshVariable(
            name="v",
            location="face",
            mesh_name="m",
            shape=(3,),
            _loader=loader,
        )
        result = var.data
        np.testing.assert_array_equal(result, lazy_data)
        assert (
            call_count == 1
        ), f"Loader should be called once, was called {call_count} times"

        result2 = var.data
        np.testing.assert_array_equal(result2, lazy_data)
        assert (
            call_count == 1
        ), f"Loader should still be called once (cached), was called {call_count} times"

    def test_data_property_no_data_no_loader(self):
        """Test data property returns None when no data and no loader.

        Test scenario:
            MeshVariable with neither _data nor _loader should return None.
        """
        var = MeshVariable(name="v", location="face", mesh_name="m", shape=(3,))
        assert var.data is None, f"Expected None, got {var.data}"

    def test_n_elements_1d(self, face_var_1d):
        """Test n_elements returns last dimension for 1D variable.

        Test scenario:
            For shape (5,), n_elements should be 5.
        """
        assert face_var_1d.n_elements == 5, f"Expected 5, got {face_var_1d.n_elements}"

    def test_n_elements_2d(self, temporal_var):
        """Test n_elements returns last dimension for 2D variable.

        Test scenario:
            For shape (3, 5), n_elements should be 5 (spatial dim).
        """
        assert (
            temporal_var.n_elements == 5
        ), f"Expected 5, got {temporal_var.n_elements}"

    def test_n_elements_empty_shape(self):
        """Test n_elements returns 0 for empty shape.

        Test scenario:
            Shape () should yield n_elements = 0.
        """
        var = MeshVariable(name="v", location="face", mesh_name="m", shape=())
        assert var.n_elements == 0, f"Expected 0, got {var.n_elements}"

    def test_has_time_false(self, face_var_1d):
        """Test has_time is False for 1D variable.

        Test scenario:
            A variable with shape (5,) has no time dimension.
        """
        assert face_var_1d.has_time is False, "Expected has_time=False for 1D variable"

    def test_has_time_true(self, temporal_var):
        """Test has_time is True for 2D variable.

        Test scenario:
            A variable with shape (3, 5) has a time dimension.
        """
        assert temporal_var.has_time is True, "Expected has_time=True for 2D variable"

    def test_n_time_steps_no_time(self, face_var_1d):
        """Test n_time_steps returns 0 for non-temporal variable.

        Test scenario:
            1D variable should have 0 time steps.
        """
        assert (
            face_var_1d.n_time_steps == 0
        ), f"Expected 0, got {face_var_1d.n_time_steps}"

    def test_n_time_steps_with_time(self, temporal_var):
        """Test n_time_steps returns correct count for temporal variable.

        Test scenario:
            Variable with shape (3, 5) should have 3 time steps.
        """
        assert (
            temporal_var.n_time_steps == 3
        ), f"Expected 3, got {temporal_var.n_time_steps}"

    def test_dtype_from_data(self, face_var_1d):
        """Test dtype returns the dtype of the data array.

        Test scenario:
            float64 data should yield float64 dtype.
        """
        assert face_var_1d.dtype == np.dtype(
            "float64"
        ), f"Expected float64, got {face_var_1d.dtype}"

    def test_dtype_int_data(self):
        """Test dtype for integer data.

        Test scenario:
            int32 data should yield int32 dtype.
        """
        var = MeshVariable(
            name="v",
            location="face",
            mesh_name="m",
            shape=(3,),
            _data=np.array([1, 2, 3], dtype=np.int32),
        )
        assert var.dtype == np.dtype("int32"), f"Expected int32, got {var.dtype}"

    def test_dtype_no_data_default(self):
        """Test dtype defaults to float64 when no data.

        Test scenario:
            Variable with no data should default to float64 dtype.
        """
        var = MeshVariable(name="v", location="face", mesh_name="m", shape=(3,))
        assert var.dtype == np.dtype(
            "float64"
        ), f"Expected float64 default, got {var.dtype}"

    def test_dtype_explicit_without_data(self):
        """Test dtype returns explicit _dtype when data is not loaded.

        Test scenario:
            Variable with _dtype=int32 but _data=None should return int32
            without triggering a lazy load.
        """
        var = MeshVariable(
            name="v",
            location="face",
            mesh_name="m",
            shape=(3,),
            _dtype=np.dtype("int32"),
        )
        assert var.dtype == np.dtype(
            "int32"
        ), f"Expected explicit int32, got {var.dtype}"
        assert var._data is None, "Data should NOT be loaded by dtype access"

    def test_sel_time_valid(self, temporal_var):
        """Test sel_time returns correct time step.

        Test scenario:
            Selecting time step 1 from a (3, 5) variable should return
            the second row.
        """
        result = temporal_var.sel_time(1)
        expected = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sel_time_first_step(self, temporal_var):
        """Test sel_time with index 0.

        Test scenario:
            First time step selection.
        """
        result = temporal_var.sel_time(0)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sel_time_last_step(self, temporal_var):
        """Test sel_time with last index.

        Test scenario:
            Last time step selection using index 2.
        """
        result = temporal_var.sel_time(2)
        expected = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sel_time_no_time_raises(self, face_var_1d):
        """Test sel_time raises ValueError for non-temporal variable.

        Test scenario:
            Calling sel_time on a 1D variable should raise ValueError.
        """
        with pytest.raises(ValueError, match="no time dimension"):
            face_var_1d.sel_time(0)

    def test_sel_time_range_valid(self, temporal_var):
        """Test sel_time_range returns a new MeshVariable with sliced data.

        Test scenario:
            Selecting time range [0:2] from a (3, 5) variable should return
            a new MeshVariable with shape (2, 5).
        """
        result = temporal_var.sel_time_range(0, 2)
        assert isinstance(
            result, MeshVariable
        ), f"Expected MeshVariable, got {type(result)}"
        assert result.shape == (2, 5), f"Expected shape (2, 5), got {result.shape}"
        assert result.name == temporal_var.name, "Name should be preserved"
        assert result.location == temporal_var.location, "Location should be preserved"
        assert (
            result.mesh_name == temporal_var.mesh_name
        ), "Mesh name should be preserved"
        expected_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]])
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_sel_time_range_preserves_metadata(self, temporal_var):
        """Test sel_time_range preserves nodata, units, and standard_name.

        Test scenario:
            Metadata attributes should carry over to the sliced variable.
        """
        temporal_var.nodata = -999.0
        temporal_var.units = "m"
        temporal_var.standard_name = "sea_surface_height"
        result = temporal_var.sel_time_range(1, 3)
        assert result.nodata == -999.0, f"Expected nodata -999.0, got {result.nodata}"
        assert result.units == "m", f"Expected units 'm', got '{result.units}'"
        assert (
            result.standard_name == "sea_surface_height"
        ), f"Expected standard_name 'sea_surface_height', got '{result.standard_name}'"

    def test_sel_time_range_no_time_raises(self, face_var_1d):
        """Test sel_time_range raises ValueError for non-temporal variable.

        Test scenario:
            Calling sel_time_range on a 1D variable should raise ValueError.
        """
        with pytest.raises(ValueError, match="no time dimension"):
            face_var_1d.sel_time_range(0, 1)


class TestUgridMetadata:
    """Tests for the UgridMetadata frozen dataclass."""

    def test_init_defaults(self):
        """Test construction with default values.

        Test scenario:
            All fields should have sensible defaults.
        """
        meta = UgridMetadata()
        assert (
            meta.mesh_topologies == ()
        ), f"Expected empty tuple, got {meta.mesh_topologies}"
        assert (
            meta.data_variables == {}
        ), f"Expected empty dict, got {meta.data_variables}"
        assert (
            meta.global_attributes == {}
        ), f"Expected empty dict, got {meta.global_attributes}"
        assert meta.conventions is None, f"Expected None, got {meta.conventions}"
        assert meta.n_nodes == 0, f"Expected 0, got {meta.n_nodes}"
        assert meta.n_faces == 0, f"Expected 0, got {meta.n_faces}"
        assert meta.n_edges == 0, f"Expected 0, got {meta.n_edges}"

    def test_init_with_topology(self):
        """Test construction with a mesh topology and data variables.

        Test scenario:
            Full UgridMetadata with a single 2D mesh topology.
        """
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="node_x",
            node_y_var="node_y",
        )
        meta = UgridMetadata(
            mesh_topologies=(topo,),
            data_variables={"water_level": "face"},
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            conventions="CF-1.8 UGRID-1.0",
            n_nodes=100,
            n_faces=80,
            n_edges=200,
        )
        assert (
            len(meta.mesh_topologies) == 1
        ), f"Expected 1 topology, got {len(meta.mesh_topologies)}"
        assert (
            meta.mesh_topologies[0].mesh_name == "mesh2d"
        ), "Topology mesh_name mismatch"
        assert meta.data_variables == {"water_level": "face"}, "data_variables mismatch"
        assert (
            meta.conventions == "CF-1.8 UGRID-1.0"
        ), f"conventions mismatch: {meta.conventions}"
        assert meta.n_nodes == 100, f"Expected n_nodes 100, got {meta.n_nodes}"
        assert meta.n_faces == 80, f"Expected n_faces 80, got {meta.n_faces}"
        assert meta.n_edges == 200, f"Expected n_edges 200, got {meta.n_edges}"

    def test_frozen_immutability(self):
        """Test that UgridMetadata is immutable.

        Test scenario:
            Attempting to set an attribute should raise FrozenInstanceError.
        """
        meta = UgridMetadata()
        with pytest.raises(AttributeError):
            meta.n_nodes = 42

    def test_equality(self):
        """Test equality between identical UgridMetadata instances.

        Test scenario:
            Two UgridMetadata objects with the same values should be equal.
        """
        kwargs = dict(conventions="CF-1.8", n_nodes=10, n_faces=5, n_edges=15)
        m1 = UgridMetadata(**kwargs)
        m2 = UgridMetadata(**kwargs)
        assert m1 == m2, "Identical UgridMetadata objects should be equal"
