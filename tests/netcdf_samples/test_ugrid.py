"""UGRID unstructured-mesh support and the underlying NetCDF structure of the mesh files."""

import pytest

from pyramids.netcdf import NetCDF, UgridDataset

pytestmark = pytest.mark.core

_UNSTRUCTURED_DIMS = {"n_face", "n_node", "n_edge", "ncol"}


@pytest.mark.samples("ugrid")
def test_ugrid_files_have_unstructured_dimensions(sample_name, sample):
    """Each UGRID file exposes an unstructured dimension (n_face / n_node / ncol) via the NetCDF reader."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        dims = set(nc.dimension_sizes)
        assert dims & _UNSTRUCTURED_DIMS, f"{sample_name}: no unstructured dimension in {sorted(dims)}"
    finally:
        nc.close()


@pytest.mark.samples("ugrid")
@pytest.mark.xfail(
    reason="UgridDataset needs a central mesh_topology var; UXARRAY files declare it via cf_role (#589)",
    strict=False,
)
def test_ugrid_dataset_reads_mesh(sample_name, sample):
    """``UgridDataset.read_file`` should load the mesh (currently raises, issue #589)."""
    ug = UgridDataset.read_file(sample(sample_name))
    assert ug.mesh is not None
