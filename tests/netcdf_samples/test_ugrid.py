"""UGRID unstructured-mesh support and the underlying NetCDF structure of the mesh files."""

import pytest

from pyramids.netcdf import NetCDF, UgridDataset

pytestmark = pytest.mark.core

_UNSTRUCTURED_DIMS = {"n_face", "n_node", "n_edge", "ncol"}
MESH = "ugrid__6v__1d5-2d1.nc"               # a true mesh: nodes/faces + face_node_connectivity
DATA_ONLY = ["ugrid__1v__3d1.nc", "ugrid__1v__1d1.nc"]  # data over a mesh dim, no topology in-file


@pytest.mark.samples("ugrid")
def test_ugrid_files_have_unstructured_dimensions(sample_name, sample):
    """Each UGRID file exposes an unstructured dimension (n_face / n_node / ncol) via the NetCDF reader."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        dims = set(nc.dimension_sizes)
        assert dims & _UNSTRUCTURED_DIMS, f"{sample_name}: no unstructured dimension in {sorted(dims)}"
    finally:
        nc.close()


def test_ugrid_dataset_reads_mesh(sample):
    """``UgridDataset.read_file`` loads a UXARRAY-style mesh declared via cf_role connectivity (#589)."""
    ug = UgridDataset.read_file(sample(MESH))
    assert ug.mesh is not None
    assert ug.mesh.n_node == 16
    assert ug.mesh.n_face == 4


@pytest.mark.parametrize("name", DATA_ONLY)
def test_ugrid_data_only_files_have_no_mesh(name, sample):
    """Files carrying data on a mesh dimension but no topology raise a clear error (no mesh to build)."""
    with pytest.raises(ValueError, match="mesh topology"):
        UgridDataset.read_file(sample(name))
