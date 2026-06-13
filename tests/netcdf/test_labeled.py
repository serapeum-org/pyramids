"""Tests for the label-indexed (non-gridded) NetCDF/Zarr reader (GDAL multidim).

Builds a tiny ``feature_id x time`` streamflow store — the shape of an NWM
``channel_rt`` table — and checks :class:`LabeledDataset` opens it through GDAL's
multidimensional API (no xarray/dask) and exposes its dims, coords, and variables
without forcing a raster interpretation.

Fixtures are written with xarray purely as test scaffolding (the production class
reads them via GDAL); they are gated on the optional ``xarray`` dependency.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.netcdf import LabeledDataset
from pyramids.netcdf import labeled as labeled_mod
from pyramids.netcdf.labeled import _is_remote_url, _is_zarr_store

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.xarray

N_TIME, N_FEAT = 4, 3


def _streamflow_dataset(**extra_coords) -> xr.Dataset:
    """A ``(time, feature_id)`` streamflow store with NWM-like coordinates."""
    feature_id = np.array([101, 202, 303], dtype="int64")
    time = np.array(
        ["2010-06-01", "2010-06-02", "2010-06-03", "2010-06-04"], dtype="datetime64[ns]"
    )
    streamflow = np.arange(N_TIME * N_FEAT, dtype="f4").reshape(N_TIME, N_FEAT)
    coords = {
        "time": ("time", time),
        "feature_id": ("feature_id", feature_id),
        "latitude": ("feature_id", np.array([40.0, 41.0, 42.0], "f8")),
        "longitude": ("feature_id", np.array([-75.0, -76.0, -77.0], "f8")),
        "gage_id": ("feature_id", np.array(["01010000", "01010500", "01011000"])),
    }
    coords.update(extra_coords)
    return xr.Dataset(
        {"streamflow": (("time", "feature_id"), streamflow)}, coords=coords
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
    """Write the streamflow store to a Zarr v2 store.

    GDAL 3.12's Zarr driver cannot read the ``fixed_length_utf32`` string encoding
    Zarr v3 uses, so the fixture is written as v2 (which GDAL reads natively).
    """
    path = tmp_path / "channel_rt.zarr"
    _streamflow_dataset().to_zarr(path, mode="w", zarr_format=2)
    return path


class TestLabeledDatasetRead:
    """Open a label-indexed store and expose its structure."""

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
        store = LabeledDataset.read_file(zarr_store, engine="zarr")
        assert store.variables == ["streamflow"]

    def test_zarr_v3_string_coord_degrades_gracefully(self, tmp_path: Path):
        """A Zarr v3 string coord GDAL can't read is skipped with a warning.

        Test scenario:
            GDAL's Zarr driver rejects Zarr v3 string arrays
            (https://github.com/OSGeo/gdal/issues/13782). On GDAL versions where
            the store still opens, the unreadable string coord (``gage_id``) is
            dropped with a warning and the numeric data stays available.
        """
        path = tmp_path / "v3.zarr"
        _streamflow_dataset().to_zarr(path, mode="w", zarr_format=3)
        with pytest.warns(UserWarning, match="13782"):
            store = LabeledDataset.read_file(path)
        assert "gage_id" not in store.coordinates, "v3 string coord must be dropped"
        assert "feature_id" in store.coordinates
        assert store.variables == ["streamflow"]
        np.testing.assert_array_equal(
            store["streamflow"].values,
            np.arange(N_TIME * N_FEAT, dtype="f4").reshape(N_TIME, N_FEAT),
        )

    def test_variables_subset(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store, variables=["streamflow"])
        assert store.variables == ["streamflow"]

    def test_variables_unknown_name_raises(self, nc_store: Path):
        """An unknown name in ``variables=`` raises KeyError and releases the handle.

        The KeyError must not leak the open GDAL handle (a leaked read handle keeps
        the file locked on Windows), so the store file is unlinkable afterward.
        """
        with pytest.raises(KeyError, match="not found"):
            LabeledDataset.read_file(nc_store, variables=["streamflow", "typo"])
        nc_store.unlink()  # raises PermissionError on Windows if the handle leaked
        assert not nc_store.exists()

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

    def test_time_coord_decoded_to_datetime(self, nc_store: Path):
        """A CF time coordinate is decoded to datetime64, not raw numbers."""
        store = LabeledDataset.read_file(nc_store)
        times = store["time"].values
        assert np.issubdtype(times.dtype, np.datetime64), f"got {times.dtype}"
        assert str(times[0]).startswith("2010-06-01")

    def test_repr_is_structure_only(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        text = repr(store)
        assert "LabeledDataset" in text
        assert "streamflow" in text
        assert "feature_id" in text

    def test_dimensions_property(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        assert store.dimensions == ["time", "feature_id"]

    def test_unopenable_store_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="could not open"):
            LabeledDataset.read_file(tmp_path / "does-not-exist.nc")

    def test_missing_group_raises_and_releases_handle(self, nc_store: Path):
        """A bad ``group`` name raises and does not leak the GDAL handle.

        Test scenario:
            ``read_file(group="nope")`` raises ValueError; the store file is
            then removable, proving the open handle was released before the
            raise (a leaked read handle keeps the file locked on Windows).
        """
        with pytest.raises(ValueError, match="not found"):
            LabeledDataset.read_file(nc_store, group="nope")
        nc_store.unlink()  # raises PermissionError on Windows if the handle leaked
        assert not nc_store.exists()


class TestLabeledDatasetSelect:
    """Select by label dimension and by a secondary 1-D coord."""

    def test_select_feature_id_subset(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[101, 303])
        assert sub.sizes["feature_id"] == 2
        assert list(sub["feature_id"].values) == [101, 303]
        assert sub.sizes["time"] == N_TIME

    def test_select_reads_only_selected_data(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[101, 303])
        np.testing.assert_array_equal(
            sub["streamflow"].values, store["streamflow"].values[:, [0, 2]]
        )

    def test_select_accepts_0d_array_as_scalar(self, nc_store: Path):
        """A 0-d numpy array label selects exactly like a Python scalar (N3).

        Before the fix, a 0-d ndarray was mis-classified as a sequence and
        ``select`` raised on ``list(0d_array)``; now it squeezes the dimension
        just like the equivalent Python scalar.
        """
        store = LabeledDataset.read_file(nc_store)
        by_scalar = store.select(feature_id=101)
        by_0d = store.select(feature_id=np.array(101))
        assert by_0d.sizes == by_scalar.sizes, "0-d array must match scalar select"
        np.testing.assert_array_equal(
            by_0d["streamflow"].values, by_scalar["streamflow"].values
        )

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

    def test_select_on_coordless_name_raises_clearly(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        # 'string8' is an internal dimension with no coordinate variable.
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select(string8=[0])

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
        sub = store.select_by_coord("gage_id", ["01011000", "01010000"])
        assert list(sub["feature_id"].values) == [101, 303]

    def test_select_scalar_value(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=202)
        assert "feature_id" not in sub.sizes
        assert int(sub["feature_id"].values) == 202

    def test_select_by_coord_unknown_coord_name_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(KeyError, match="not a coordinate"):
            store.select_by_coord("nope", ["x"])

    def test_select_by_coord_requires_1d(self, tmp_path: Path):
        ds = _streamflow_dataset(
            grid=(("time", "feature_id"), np.zeros((N_TIME, N_FEAT)))
        )
        path = tmp_path / "grid.nc"
        ds.to_netcdf(path)
        ds.close()
        store = LabeledDataset.read_file(path)
        with pytest.raises(KeyError, match="must be 1-D"):
            store.select_by_coord("grid", [0.0])

    def test_select_empty_list_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(ValueError, match="empty selection list"):
            store.select(feature_id=[])

    def test_select_by_coord_empty_list_raises(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(ValueError, match="empty selection list"):
            store.select_by_coord("gage_id", [])

    def test_select_tuple_treated_as_list(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        from_tuple = store.select(feature_id=(101, 202))
        from_list = store.select(feature_id=[101, 202])
        assert from_tuple.sizes == from_list.sizes == {"time": N_TIME, "feature_id": 2}
        assert list(from_tuple["feature_id"].values) == [101, 202]

    def test_select_numpy_array_selector_keeps_request_order(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=np.array([303, 101]))
        assert list(sub["feature_id"].values) == [303, 101]


class TestLabeledDatasetTimeSlice:
    """Slice the time axis, composing with label selection."""

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

    def test_select_time_full_range_keeps_all(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select_time()
        assert sub.sizes["time"] == N_TIME

    def test_select_time_composes_with_select(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        sub = store.select(feature_id=[101, 202]).select_time(
            "2010-06-01", "2010-06-02"
        )
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
        # a non-standard calendar decodes to cftime objects, not datetime64.
        decoded = store["time"].values
        assert isinstance(decoded[0], cftime.DatetimeNoLeap), f"got {type(decoded[0])}"


class TestLabeledDatasetClose:
    """close() / context manager release the GDAL store handle."""

    def test_close_releases_handle(self, nc_store: Path):
        """After close() the file is unlocked and can be removed (Windows)."""
        store = LabeledDataset.read_file(nc_store)
        assert store["streamflow"].shape == (N_TIME, N_FEAT)
        store.close()
        nc_store.unlink()
        assert not nc_store.exists(), "file must be removable after close()"

    def test_context_manager_closes(self, nc_store: Path):
        """The store is usable inside a with-block and closed on exit."""
        with LabeledDataset.read_file(nc_store) as store:
            assert store.variables == ["streamflow"]
        nc_store.unlink()
        assert not nc_store.exists(), "file must be removable after the with-block"


class TestLabeledDatasetBbox:
    """Bbox subset via the in-file 1-D lat/lon coords."""

    def test_bbox_keeps_inside_features(self, nc_store: Path):
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

    def test_non_1d_coord_raises(self, tmp_path: Path):
        ds = _streamflow_dataset(
            grid=(("time", "feature_id"), np.zeros((N_TIME, N_FEAT)))
        )
        path = tmp_path / "grid.nc"
        ds.to_netcdf(path)
        ds.close()
        store = LabeledDataset.read_file(path)
        with pytest.raises(KeyError, match="must be 1-D"):
            store.select_bbox((-77, 40, -75, 42), lon="grid")

    def test_mismatched_dims_raises(self, tmp_path: Path):
        ds = _streamflow_dataset(t_lon=(("time",), np.zeros(N_TIME)))
        path = tmp_path / "mismatch.nc"
        ds.to_netcdf(path)
        ds.close()
        store = LabeledDataset.read_file(path)
        with pytest.raises(KeyError, match="same dimension"):
            store.select_bbox((-77, 40, -75, 42), lon="t_lon")

    def test_bbox_custom_lon_lat_names(self, tmp_path: Path):
        ds = _streamflow_dataset().rename({"longitude": "x", "latitude": "y"})
        path = tmp_path / "xy.nc"
        ds.to_netcdf(path)
        ds.close()
        store = LabeledDataset.read_file(path)
        sub = store.select_bbox((-76.5, 40.5, -74.5, 41.5), lon="x", lat="y")
        assert list(sub["feature_id"].values) == [202]


class TestLabeledDatasetWrite:
    """Typed tabular write-out (DataFrame / Parquet / CSV)."""

    def test_to_dataframe_is_tidy(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        df = store.to_dataframe()
        assert len(df) == N_TIME * N_FEAT
        for col in ("time", "feature_id", "streamflow"):
            assert col in df.columns

    def test_to_dataframe_time_is_datetime(self, nc_store: Path):
        """The tidy table's time column is decoded datetimes, not raw numbers."""
        store = LabeledDataset.read_file(nc_store)
        df = store.to_dataframe()
        assert np.issubdtype(df["time"].dtype, np.datetime64), f"got {df['time'].dtype}"

    def test_to_dataframe_values_match(self, nc_store: Path):
        store = LabeledDataset.read_file(nc_store)
        df = store.select(feature_id=[101]).to_dataframe()
        assert len(df) == N_TIME
        assert set(df["feature_id"]) == {101}
        # streamflow for feature_id 101 (column 0) across the 4 timesteps.
        assert list(df["streamflow"]) == [0.0, 3.0, 6.0, 9.0]

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

    def test_to_parquet_missing_pyarrow_raises(
        self, tmp_path: Path, nc_store, monkeypatch
    ):
        def _raise(_msg):
            raise OptionalPackageDoesNotExist("no pyarrow")

        monkeypatch.setattr("pyramids.netcdf.labeled.import_pyarrow", _raise)
        store = LabeledDataset.read_file(nc_store)
        with pytest.raises(OptionalPackageDoesNotExist):
            store.to_parquet(tmp_path / "q.parquet")


class TestLabeledDatasetRemoteOpen:
    """Remote open wiring: URL -> /vsi translation + anon config (no network)."""

    def test_remote_zarr_translated_and_anon(self, monkeypatch):
        captured = {}

        def fake_openex(path, flags):
            captured["path"] = path
            captured["no_sign"] = gdal.GetConfigOption("AWS_NO_SIGN_REQUEST")
            return None

        monkeypatch.setattr(labeled_mod.gdal, "OpenEx", fake_openex)
        with pytest.raises(ValueError, match="could not open"):
            LabeledDataset.read_file("s3://bucket/chrtout.zarr", anon=True)
        assert captured["path"] == 'ZARR:"/vsis3/bucket/chrtout.zarr"'
        assert captured["no_sign"] == "YES"

    def test_remote_netcdf_translated_without_anon(self, monkeypatch):
        captured = {}

        def fake_openex(path, flags):
            captured["path"] = path
            captured["no_sign"] = gdal.GetConfigOption("AWS_NO_SIGN_REQUEST")
            return None

        monkeypatch.setattr(labeled_mod.gdal, "OpenEx", fake_openex)
        with pytest.raises(ValueError, match="could not open"):
            LabeledDataset.read_file("s3://bucket/channel_rt.nc")
        assert captured["path"] == "/vsis3/bucket/channel_rt.nc"
        assert captured["no_sign"] in (None, "", "NO")


class TestIsRemoteUrl:
    """Unit tests for the ``_is_remote_url`` scheme classifier."""

    @pytest.mark.parametrize(
        "scheme", ["s3", "gs", "gcs", "az", "abfs", "http", "https", "S3", "HTTPS"]
    )
    def test_remote_schemes_are_remote(self, scheme: str):
        """Every supported object-store / web scheme is classified remote."""
        assert _is_remote_url(f"{scheme}://host/store") is True

    @pytest.mark.parametrize("scheme", ["ftp", "sftp", "file", "mailto"])
    def test_unknown_schemes_are_not_remote(self, scheme: str):
        """A scheme outside the known set is not classified remote."""
        assert _is_remote_url(f"{scheme}://host/store") is False

    @pytest.mark.parametrize(
        "source",
        ["/data/x.nc", "relative/path/x.zarr", r"C:\data\x.nc", "x.zarr", ""],
    )
    def test_local_paths_are_not_remote(self, source: str):
        """Local filesystem paths are not classified remote."""
        assert _is_remote_url(source) is False


class TestIsZarrStore:
    """Unit tests for the ``_is_zarr_store`` kind classifier."""

    @pytest.mark.parametrize(
        "engine, expected", [("zarr", True), ("netcdf4", False), ("h5netcdf", False)]
    )
    def test_explicit_engine_wins(self, engine: str, expected: bool):
        """An explicit ``engine`` overrides any suffix heuristic."""
        path = "store.nc" if expected else "store.zarr"
        assert _is_zarr_store(path, engine) is expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("store.zarr", True),
            ("store.zarr/", True),
            ("store.zarr\\", True),
            ("s3://bucket/store.zarr", True),
            ("store.nc", False),
            ("store", False),
        ],
    )
    def test_suffix_heuristic_when_no_engine(self, path: str, expected: bool):
        """With ``engine=None`` the ``.zarr`` suffix (trailing slash trimmed) decides."""
        assert _is_zarr_store(path, None) is expected
