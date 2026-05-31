"""Tests for the label-indexed (non-gridded) NetCDF/Zarr reader (PY-G / P-A).

Builds a tiny ``feature_id x time`` streamflow store — the shape of an NWM
``channel_rt`` table — and checks :class:`LabeledDataset` opens it lazily and
exposes its dims, coords, and variables without forcing a raster interpretation.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from pyramids.base._errors import OptionalPackageDoesNotExist
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

    def test_select_on_coordless_dim_raises_clearly(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        # 'extra' is a dimension with no coordinate variable -> cannot be
        # selected by value; expect the clear "not a coordinate" error.
        ds2 = store.dataset.expand_dims({"extra": 2})
        with pytest.raises(KeyError, match="not a coordinate"):
            LabeledDataset(ds2).select(extra=[0])

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


class TestLabeledDatasetTimeSlice:
    """P-E: slice the time axis, composing with label selection."""

    def test_select_time_window(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_time("2010-06-02", "2010-06-03")
        assert sub.sizes["time"] == 2
        assert sub.sizes["feature_id"] == N_FEAT

    def test_select_time_open_ended_start(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_time(end="2010-06-02")
        assert sub.sizes["time"] == 2

    def test_select_time_open_ended_end(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_time(start="2010-06-03")
        assert sub.sizes["time"] == 2

    def test_select_time_composes_with_select(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[101, 202]).select_time("2010-06-01", "2010-06-02")
        assert sub.sizes == {"time": 2, "feature_id": 2}

    def test_out_of_range_window_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(ValueError, match="no timesteps in window"):
            store.select_time("2020-01-01", "2020-12-31")

    def test_unknown_time_dim_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select_time("2010-06-01", time_dim="when")

    def test_cftime_axis(self, tmp_path: Path):
        cftime = pytest.importorskip("cftime")
        times = [cftime.DatetimeNoLeap(2010, 6, d) for d in (1, 2, 3, 4)]
        ds = xr.Dataset(
            {"q": (("time", "feature_id"), np.zeros((4, 2), "f4"))},
            coords={"time": ("time", times), "feature_id": ("feature_id", [1, 2])},
        )
        path = tmp_path / "cftime.nc"
        ds.to_netcdf(path)
        ds.close()
        store = LabeledDataset.read_file(path)
        sub = store.select_time("2010-06-02", "2010-06-03")
        assert sub.sizes["time"] == 2


class TestLabeledDatasetBbox:
    """P-D: bbox subset via the in-file 1-D lat/lon coords."""

    def test_bbox_keeps_inside_features(self, nc_store: Path):
        # lats [40,41,42], lons [-75,-76,-77] over feature_id [101,202,303];
        # this box selects only feature 202.
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_bbox((-76.5, 40.5, -74.5, 41.5))
        assert list(sub["feature_id"].values) == [202]

    def test_bbox_inclusive_bounds(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_bbox((-77.0, 40.0, -75.0, 42.0))
        assert list(sub["feature_id"].values) == [101, 202, 303]

    def test_bbox_composes_with_time(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_bbox((-76.5, 40.5, -74.5, 41.5)).select_time(
            "2010-06-01", "2010-06-02"
        )
        assert sub.sizes == {"time": 2, "feature_id": 1}

    def test_empty_bbox_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(ValueError, match="no labels inside bbox"):
            store.select_bbox((0.0, 0.0, 1.0, 1.0))

    def test_unknown_coord_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select_bbox((-77, 40, -75, 42), lon="x")

    def test_non_1d_coord_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        ds2 = store.dataset.assign_coords(
            grid=(("time", "feature_id"), np.zeros((N_TIME, N_FEAT)))
        )
        with pytest.raises(KeyError, match="must be 1-D"):
            LabeledDataset(ds2).select_bbox((-77, 40, -75, 42), lon="grid")

    def test_mismatched_dims_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        ds2 = store.dataset.assign_coords(t_lon=(("time",), np.zeros(N_TIME)))
        with pytest.raises(KeyError, match="same dimension"):
            LabeledDataset(ds2).select_bbox((-77, 40, -75, 42), lon="t_lon")


class TestLabeledDatasetWrite:
    """P-F: typed tabular write-out (DataFrame / Parquet / CSV)."""

    def test_to_dataframe_is_tidy(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        df = store.to_dataframe()
        assert len(df) == N_TIME * N_FEAT
        for col in ("time", "feature_id", "streamflow"):
            assert col in df.columns

    def test_to_dataframe_after_select(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        df = store.select(feature_id=[101]).to_dataframe()
        assert len(df) == N_TIME
        assert set(df["feature_id"]) == {101}

    def test_to_parquet_round_trip(self, tmp_path: Path, nc_store: Path):
        pytest.importorskip("pyarrow")
        store = LabeledDataset.read_file(nc_store)
        out = store.to_parquet(tmp_path / "q.parquet")
        assert out.exists()
        back = pd.read_parquet(out)
        assert len(back) == N_TIME * N_FEAT
        assert set(back["feature_id"]) == {101, 202, 303}

    def test_to_csv_round_trip(self, tmp_path: Path, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        out = store.to_csv(tmp_path / "q.csv")
        assert out.exists()
        back = pd.read_csv(out)
        assert len(back) == N_TIME * N_FEAT

    def test_small_store_does_not_warn(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            store.to_dataframe()

    def test_large_store_warns_before_realising(self, nc_store: Path, monkeypatch):
        monkeypatch.setattr("pyramids.netcdf.labeled._LARGE_REALISE_BYTES", 0)
        store = LabeledDataset.read_file(nc_store)
        with pytest.warns(UserWarning, match="realising"):
            store.to_dataframe()

    def test_to_parquet_missing_pyarrow_raises(self, tmp_path: Path, nc_store, monkeypatch):
        def _raise(_msg):
            raise OptionalPackageDoesNotExist("no pyarrow")

        monkeypatch.setattr("pyramids.netcdf.labeled.import_pyarrow", _raise)
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(OptionalPackageDoesNotExist):
            store.to_parquet(tmp_path / "q.parquet")


class TestLabeledDatasetRemoteOpen:
    """P-B: anonymous remote open wiring (no network)."""

    def _fake_zarr(self):
        return xr.Dataset(
            {"streamflow": (("feature_id",), np.zeros(2, "f4"))},
            coords={"feature_id": ("feature_id", [1, 2])},
        )

    def test_anon_injects_storage_options(self, monkeypatch):
        captured = {}

        def fake_open_zarr(source, **kwargs):
            captured["source"] = source
            captured.update(kwargs)
            return self._fake_zarr()

        monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
        store = LabeledDataset.read_file("s3://bucket/chrtout.zarr", anon=True)
        assert captured["source"] == "s3://bucket/chrtout.zarr"
        assert captured["storage_options"] == {"anon": True}
        assert "streamflow" in store.variables

    def test_explicit_storage_options_override_anon(self, monkeypatch):
        captured = {}

        def fake_open_zarr(source, **kwargs):
            captured.update(kwargs)
            return self._fake_zarr()

        monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
        LabeledDataset.read_file(
            "s3://b/x.zarr", anon=True, storage_options={"anon": False, "region": "us-east-1"}
        )
        assert captured["storage_options"] == {"anon": False, "region": "us-east-1"}

    def test_anon_threads_to_netcdf_open(self, monkeypatch):
        captured = {}

        def fake_open_dataset(source, **kwargs):
            captured.update(kwargs)
            return self._fake_zarr()

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        LabeledDataset.read_file("s3://b/channel_rt.nc", anon=True)
        assert captured["storage_options"] == {"anon": True}


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYRAMIDS_RUN_NWM_ZARR_TEST") != "1",
    reason="set PYRAMIDS_RUN_NWM_ZARR_TEST=1 to open the NWM retrospective Zarr from S3",
)
class TestLabeledDatasetRemoteIntegration:
    """P-B: open the real NWM retrospective Zarr anonymously (opt-in, network)."""

    def test_open_nwm_retro_chrtout_anonymously(self):
        pytest.importorskip("s3fs")
        url = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
        try:
            store = LabeledDataset.read_file(url, anon=True, chunks={})
        except OSError as exc:
            # connection / DNS / timeout / permission -> skip; a LabeledDataset
            # logic error (KeyError/ValueError/...) still fails the test.
            pytest.skip(f"NWM retrospective Zarr unreachable: {exc}")
        assert "feature_id" in store.dimensions, f"dims: {store.dimensions}"
        assert "time" in store.dimensions, f"dims: {store.dimensions}"
        assert len(store.variables) > 0, "no data variables read"


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
