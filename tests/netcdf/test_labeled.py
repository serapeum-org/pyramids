"""Tests for the label-indexed (non-gridded) NetCDF/Zarr reader (PY-G / P-A).

Builds a tiny ``feature_id x time`` streamflow store — the shape of an NWM
``channel_rt`` table — and checks :class:`LabeledDataset` opens it lazily and
exposes its dims, coords, and variables without forcing a raster interpretation.
"""
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from pyramids.netcdf import LabeledDataset

pytestmark = pytest.mark.xarray

N_TIME, N_FEAT = 4, 3


def _streamflow_dataset() -> "xr.Dataset":
    """A ``(time, feature_id)`` streamflow store with NWM-like coordinates."""
    feature_id = np.array([101, 202, 303], dtype="int64")
    time = np.array(
        ["2010-06-01", "2010-06-02", "2010-06-03", "2010-06-04"], dtype="datetime64[ns]"
    )
    streamflow = np.arange(N_TIME * N_FEAT, dtype="f4").reshape(N_TIME, N_FEAT)
    return xr.Dataset(
        {"streamflow": (("time", "feature_id"), streamflow)},
        coords={
            "time": ("time", time),
            "feature_id": ("feature_id", feature_id),
            "latitude": ("feature_id", np.array([40.0, 41.0, 42.0], "f8")),
            "longitude": ("feature_id", np.array([-75.0, -76.0, -77.0], "f8")),
            "gage_id": ("feature_id", np.array(["01010000", "01010500", "01011000"])),
        },
    )


@pytest.fixture
def nc_store(tmp_path: Path) -> Path:
    """Write the streamflow store to a NetCDF file."""
    path = tmp_path / "channel_rt.nc"
    ds = _streamflow_dataset()
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture
def zarr_store(tmp_path: Path) -> Path:
    """Write the streamflow store to a Zarr store."""
    path = tmp_path / "channel_rt.zarr"
    _streamflow_dataset().to_zarr(path, mode="w")
    return path


class TestLabeledDatasetRead:
    """P-A: open a label-indexed store and expose its structure."""

    def test_netcdf_exposes_dims_coords_variables(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        assert store.sizes == {"time": N_TIME, "feature_id": N_FEAT}
        assert "feature_id" in store.coordinates
        assert "gage_id" in store.coordinates
        assert store.variables == ["streamflow"]

    def test_zarr_exposes_dims_coords_variables(self, zarr_store: Path):
        store = LabeledDataset.read_file(zarr_store)
        assert store.sizes == {"time": N_TIME, "feature_id": N_FEAT}
        assert "feature_id" in store.coordinates
        assert store.variables == ["streamflow"]

    def test_engine_override_opens_zarr(self, zarr_store: Path):
        # a .zarr dir opened with an explicit engine still works.
        store = LabeledDataset.read_file(zarr_store, engine="zarr")
        assert store.variables == ["streamflow"]

    def test_variables_subset(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store, variables=["streamflow"])
        assert store.variables == ["streamflow"]

    def test_getitem_and_contains(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        assert "streamflow" in store
        assert "missing" not in store
        arr = store["streamflow"]
        assert arr.dims == ("time", "feature_id")
        assert arr.shape == (N_TIME, N_FEAT)

    def test_feature_id_coord_values(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        assert list(store["feature_id"].values) == [101, 202, 303]

    def test_repr_is_structure_only(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        text = repr(store)
        assert "LabeledDataset" in text
        assert "streamflow" in text
        assert "feature_id" in text

    def test_dimensions_property(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        assert store.dimensions == ["time", "feature_id"]

    def test_read_with_explicit_engine(self, nc_store: Path):
        pytest.importorskip("h5netcdf")
        store = LabeledDataset.read_file(nc_store, engine="h5netcdf")
        assert store.variables == ["streamflow"]


class TestLabeledDatasetSelect:
    """P-C: select by label dimension and by a secondary 1-D coord."""

    def test_select_feature_id_subset(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[101, 303])
        assert sub.sizes["feature_id"] == 2
        assert list(sub["feature_id"].values) == [101, 303]
        assert sub.sizes["time"] == N_TIME

    def test_select_returns_new_instance(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[202])
        assert isinstance(sub, LabeledDataset)
        assert store.sizes["feature_id"] == N_FEAT  # original untouched

    def test_select_unknown_feature_id_reported(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="999"):
            store.select(feature_id=[101, 999])

    def test_select_unknown_dim_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select(bogus=[1])

    def test_select_by_coord_gage_id(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_by_coord("gage_id", ["01010500"])
        assert sub.sizes["feature_id"] == 1
        assert list(sub["feature_id"].values) == [202]

    def test_select_by_coord_unknown_reported(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="99999999"):
            store.select_by_coord("gage_id", ["99999999"])

    def test_select_by_coord_preserves_order(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        # request out of order; result keeps the store's original order.
        sub = store.select_by_coord("gage_id", ["01011000", "01010000"])
        assert list(sub["feature_id"].values) == [101, 303]

    def test_select_scalar_value(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=202)
        # a scalar selector drops the feature_id dimension.
        assert "feature_id" not in sub.sizes
        assert int(sub["feature_id"].values) == 202

    def test_select_by_coord_unknown_coord_name_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select_by_coord("nope", ["x"])

    def test_select_by_coord_requires_1d(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        # add a 2-D coordinate, then assert selecting on it is rejected.
        ds2 = store.dataset.assign_coords(
            grid=(("time", "feature_id"), np.zeros((N_TIME, N_FEAT)))
        )
        store2 = LabeledDataset(ds2)
        with pytest.raises(KeyError, match="must be 1-D"):
            store2.select_by_coord("grid", [0.0])


class TestLabeledDatasetLaziness:
    """Opening reads metadata only; data is not materialised."""

    def test_default_open_does_not_load_values(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        # xarray keeps the data lazy until accessed; the backing variable is not
        # an in-memory numpy array on open.
        assert store.dataset["streamflow"].variable._in_memory is False

    def test_chunks_back_arrays_with_dask(self, nc_store: Path):
        pytest.importorskip("dask")
        store = LabeledDataset.read_file(nc_store, chunks={})
        # dask-backed arrays expose a .chunks tuple (chunked, out-of-core).
        assert store["streamflow"].chunks is not None
