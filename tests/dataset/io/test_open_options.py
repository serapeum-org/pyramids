"""Tests for GDAL open options threaded through Dataset.read_file (#1025).

`Dataset.read_file(open_options=...)` forwards driver-specific open options to
GDAL and captures them on the returned dataset, so the read paths that reopen
the file rather than reuse the handle — `threadsafe=True` per-thread handles,
lazy `chunks=` reads inside dask tasks, and unpickling on a worker — reopen with
the same options. `GTiff`'s `GEOREF_SOURCES` gives an observable effect with no
special fixture: `NONE` drops the internal georeference (identity geotransform),
`INTERNAL` keeps it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from pyramids._io import normalize_open_options
from pyramids._io import read_file as io_read_file
from pyramids.base._file_manager import gdal_raster_open
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset import collection as collection_module
from pyramids.dataset.engines import io as io_engine
from pyramids.netcdf import NetCDF
from tests._helpers import write_raster
from tests._marks import requires_dask

_NETCDF_FIXTURE = (
    Path(__file__).resolve().parents[2] / "data" / "netcdf" / "none__1v__1d1.nc"
)

pytestmark = pytest.mark.core

_REAL_GT = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)
_IDENTITY_GT = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def georeferenced_tif(tmp_path):
    """A GeoTIFF carrying an internal geotransform.

    Returns:
        str: Path to an 8x8 raster whose internal georef is `_REAL_GT`.
    """
    return write_raster(
        tmp_path / "geo.tif",
        np.ones((8, 8), "float32"),
        (100.0, 200.0),
        cell_size=1.0,
    )


class TestNormalizeOpenOptions:
    """Tests for the `dict` / list / `None` normaliser."""

    def test_dict_becomes_key_value_list(self):
        """A mapping is rendered as GDAL's `KEY=VALUE` list."""
        assert normalize_open_options({"L1B_MODE": "DATASTRIP"}) == ["L1B_MODE=DATASTRIP"]

    def test_list_passes_through(self):
        """A native list is returned as a list, unchanged in content."""
        assert normalize_open_options(["A=1", "B=2"]) == ["A=1", "B=2"]

    def test_tuple_becomes_list(self):
        """A tuple (the hashable form the file managers carry) becomes a list."""
        result = normalize_open_options(("A=1", "B=2"))
        assert result == ["A=1", "B=2"], f"expected a list, got {result!r}"
        assert isinstance(result, list), "tuple input must be normalised to a list"

    def test_none_stays_none(self):
        """`None` (the no-options case) is preserved."""
        assert normalize_open_options(None) is None


class TestReadFileOpenOptions:
    """Tests for `Dataset.read_file(open_options=...)` end to end."""

    def test_default_open_captures_nothing(self, georeferenced_tif):
        """An ordinary open captures no options and keeps the real georef."""
        dataset = Dataset.read_file(georeferenced_tif)
        assert dataset.open_options == [], "an ordinary open must capture nothing"
        assert dataset.geotransform == _REAL_GT

    def test_dict_form_takes_effect(self, georeferenced_tif):
        """A `dict` option reaches GDAL: `GEOREF_SOURCES=NONE` drops the georef.

        Test scenario:
            Without the option the geotransform is `_REAL_GT`; with it, GDAL
            ignores the internal georef and returns the identity transform.
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        assert dataset.geotransform == _IDENTITY_GT, "the option did not reach GDAL"
        assert dataset.open_options == ["GEOREF_SOURCES=NONE"]

    def test_list_form_takes_effect(self, georeferenced_tif):
        """GDAL's native `["KEY=VALUE"]` form is accepted too."""
        dataset = Dataset.read_file(
            georeferenced_tif, open_options=["GEOREF_SOURCES=INTERNAL"]
        )
        assert dataset.geotransform == _REAL_GT
        assert dataset.open_options == ["GEOREF_SOURCES=INTERNAL"]

    def test_options_survive_pickle_reopen(self, georeferenced_tif):
        """A worker reopen (unpickle) reapplies the captured options.

        Test scenario:
            Losing the options on the worker would silently change driver
            behaviour there — the geotransform would flip back to `_REAL_GT`.
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        reopened = pickle.loads(pickle.dumps(dataset))
        assert reopened.open_options == ["GEOREF_SOURCES=NONE"]
        assert reopened.geotransform == _IDENTITY_GT, "options lost on reopen"

    def test_options_survive_threadsafe_reopen(self, georeferenced_tif, mocker):
        """A per-thread reopen forwards the captured options to the opener.

        Test scenario:
            `read_array(threadsafe=True)` reopens the file through the per-thread
            manager. `GEOREF_SOURCES` is inert for array values, so asserting the
            array alone cannot detect option loss on this path — instead spy on
            the reopen opener and prove it receives the captured options.
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        real = io_engine.gdal_raster_open
        seen: list = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("open_options"))
            return real(*args, **kwargs)

        mocker.patch.object(io_engine, "gdal_raster_open", side_effect=spy)
        result = np.asarray(dataset.read_array(threadsafe=True))
        assert result.shape == (8, 8), f"unexpected shape {result.shape}"
        assert seen, "the per-thread reopen opener was never called"
        assert all(o == ("GEOREF_SOURCES=NONE",) for o in seen), seen

    @requires_dask
    def test_options_survive_lazy_chunked_reopen(self, georeferenced_tif, mocker):
        """A lazy chunked read forwards the captured options to the per-chunk opener.

        Test scenario:
            `read_array(chunks=...)` builds a dask array whose per-chunk opener is
            a `CachingFileManager`; spy on the reopen opener to prove it is fed the
            captured options (again the georef option is inert for array values, so
            the array assertion alone could not catch a dropped option).
        """
        dataset = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        real = io_engine.gdal_raster_open
        seen: list = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("open_options"))
            return real(*args, **kwargs)

        mocker.patch.object(io_engine, "gdal_raster_open", side_effect=spy)
        lazy = dataset.read_array(chunks=4)
        assert hasattr(lazy, "dask"), "expected a lazy dask array"
        result = np.asarray(lazy.compute())
        assert result.shape == (8, 8), f"unexpected shape {result.shape}"
        assert np.allclose(result, 1.0), "lazy chunked read altered the values"
        assert seen, "the per-chunk reopen opener was never called"
        assert all(o == ("GEOREF_SOURCES=NONE",) for o in seen), seen

    def test_two_opens_different_options_do_not_share_a_handle(self, georeferenced_tif):
        """The shared cache keys on the options, so the two reads differ.

        Test scenario:
            Read-only opens with options go through `OpenEx(OF_SHARED)`; this pins
            the GDAL guarantee the branch relies on — that the shared-dataset
            cache key includes the open options — so if a future GDAL ignored
            them, the second open would inherit the first's georef and this test
            would fail loudly rather than returning stale data.
        """
        first = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "NONE"}
        )
        second = Dataset.read_file(
            georeferenced_tif, open_options={"GEOREF_SOURCES": "INTERNAL"}
        )
        assert first.geotransform == _IDENTITY_GT
        assert second.geotransform == _REAL_GT

    def test_no_options_dataset_survives_pickle(self, georeferenced_tif):
        """A plain (no-options) dataset still round-trips through pickle.

        Test scenario:
            The pickle recipe grew a fifth element (the captured options); an
            ordinary dataset must still carry an empty tuple through it and
            reopen with no options and the real georef — i.e. the reconstruct
            path's `open_options is None` branch.
        """
        dataset = Dataset.read_file(georeferenced_tif)
        reopened = pickle.loads(pickle.dumps(dataset))
        assert reopened.open_options == [], "a plain dataset must capture nothing"
        assert reopened.geotransform == _REAL_GT, "georef lost on plain reopen"


class TestReadFileLowLevel:
    """Tests for `_io.read_file` open branches not reachable via the wrapper.

    `Dataset.read_file` always opens read-only, so the update-mode and
    multidimensional open branches of the low-level `_io.read_file` are covered
    here directly.
    """

    def test_update_mode_with_options_reaches_gdal(self, georeferenced_tif):
        """Update-mode + options goes through `OpenEx(OF_UPDATE)` carrying them.

        Test scenario:
            `read_only=False` with `GEOREF_SOURCES=NONE` must still forward the
            option to the driver — the returned handle reports the identity
            geotransform, proving the update branch passed the option through.
        """
        src = io_read_file(
            georeferenced_tif,
            read_only=False,
            open_options={"GEOREF_SOURCES": "NONE"},
        )
        try:
            assert src is not None, "update-mode open with options returned nothing"
            assert src.GetGeoTransform() == _IDENTITY_GT, "option lost in update mode"
        finally:
            src = None

    def test_multidim_with_options_opens(self):
        """Multidimensional open forwards options through `OpenEx` (smoke/coverage).

        Test scenario:
            Opening a NetCDF with `open_as_multi_dimensional=True` and a
            recognised NetCDF open option must return a live handle, exercising
            the `open_options=options or []` multidim branch with a non-empty
            option list. This is a smoke/line-coverage check only: GDAL treats an
            unrecognised or ignored open option as a warning (not a `CE_Failure`),
            so a successful open does not by itself prove the option was honoured —
            proving an effect would need a fixture engineered around a specific
            multidim option.
        """
        assert _NETCDF_FIXTURE.exists(), f"missing fixture: {_NETCDF_FIXTURE}"
        src = io_read_file(
            str(_NETCDF_FIXTURE),
            open_as_multi_dimensional=True,
            open_options=["HONOUR_VALID_RANGE=YES"],
        )
        try:
            assert src is not None, "multidim open with options returned nothing"
        finally:
            src = None


class TestGdalRasterOpen:
    """Tests for `gdal_raster_open` — the file managers' reopen opener."""

    def test_no_options_uses_plain_open(self, georeferenced_tif):
        """With no options the opener takes the plain `gdal.Open` path.

        Test scenario:
            The falsy-options branch keeps the real georef (no option applied).
        """
        src = gdal_raster_open(georeferenced_tif)
        try:
            assert src.GetGeoTransform() == _REAL_GT, "plain open must not alter georef"
        finally:
            src = None

    def test_read_options_take_effect(self, georeferenced_tif):
        """Read-only + options reopens via `OpenEx` (flags 0) carrying them."""
        src = gdal_raster_open(
            georeferenced_tif, open_options=("GEOREF_SOURCES=NONE",)
        )
        try:
            assert src.GetGeoTransform() == _IDENTITY_GT, "read option not applied"
        finally:
            src = None

    def test_update_access_with_options(self, georeferenced_tif):
        """Update access + options selects the `OF_UPDATE` flag and applies them.

        Test scenario:
            `access="update"` with an option must resolve to the `OF_UPDATE`
            ternary branch and still forward the option (identity geotransform).
        """
        src = gdal_raster_open(
            georeferenced_tif,
            access="update",
            open_options=("GEOREF_SOURCES=NONE",),
        )
        try:
            assert src is not None, "update-access open with options returned nothing"
            assert src.GetGeoTransform() == _IDENTITY_GT, "update-mode option lost"
        finally:
            src = None


class TestCollectionOpenOptions:
    """Tests for `DatasetCollection.from_files(open_options=...)`."""

    @pytest.fixture
    def two_files(self, tmp_path):
        """Two georeferenced GeoTIFFs sharing one header.

        Returns:
            list[str]: Paths to two 8x8 rasters with the same `_REAL_GT`.
        """
        return [
            write_raster(
                tmp_path / f"g{i}.tif", np.ones((8, 8), "float32"), (100.0, 200.0)
            )
            for i in range(2)
        ]

    def test_from_files_threads_the_option(self, two_files):
        """The option reaches the eagerly-opened template.

        Test scenario:
            `GEOREF_SOURCES=NONE` on a two-file collection must drop the georef
            on the template and be exposed on the `open_options` property.
        """
        collection = DatasetCollection.from_files(
            two_files, open_options={"GEOREF_SOURCES": "NONE"}
        )
        assert collection.open_options == ["GEOREF_SOURCES=NONE"]
        assert collection.base.geotransform == _IDENTITY_GT

    def test_from_files_validate_threads_the_option(self, two_files):
        """`validate=True` threads the option into the header check.

        Test scenario:
            The per-file `_validate_headers` open must carry the option too — all
            files then report the identity georef, so the headers still agree and
            validation passes.
        """
        collection = DatasetCollection.from_files(
            two_files, open_options={"GEOREF_SOURCES": "NONE"}, validate=True
        )
        assert collection.open_options == ["GEOREF_SOURCES=NONE"]
        assert collection.base.geotransform == _IDENTITY_GT

    def test_datasets_property_threads_the_option(self, two_files):
        """Each eagerly-materialised per-timestep handle carries the option.

        Test scenario:
            The `datasets` property opens every file; each `Dataset` must both
            apply the option (identity georef) and capture it.
        """
        collection = DatasetCollection.from_files(
            two_files, open_options={"GEOREF_SOURCES": "NONE"}
        )
        handles = collection.datasets
        assert len(handles) == 2, f"expected two handles, got {len(handles)}"
        for handle in handles:
            assert handle.geotransform == _IDENTITY_GT, "option not applied per-file"
            assert handle.open_options == ["GEOREF_SOURCES=NONE"], "option not captured"

    def test_dataset_at_threads_the_option(self, two_files):
        """The single-timestep accessor opens one file with the option.

        Test scenario:
            `_dataset_at` opens only index 0; that handle must apply and capture
            the option without materialising the whole set.
        """
        collection = DatasetCollection.from_files(
            two_files, open_options={"GEOREF_SOURCES": "NONE"}
        )
        handle = collection._dataset_at(0)
        assert handle.geotransform == _IDENTITY_GT, "option not applied at index"
        assert handle.open_options == ["GEOREF_SOURCES=NONE"], "option not captured"

    @requires_dask
    def test_lazy_data_forwards_the_option(self, two_files, mocker):
        """The lazy `.data` graph threads the option into every per-timestep open.

        Test scenario:
            Building the dask stack calls `_lazy_timestep` once per file; each
            call must receive the captured options as the hashable tuple form.
        """
        collection = DatasetCollection.from_files(
            two_files, open_options={"GEOREF_SOURCES": "NONE"}
        )
        real = collection_module._lazy_timestep
        seen: list = []

        def spy(path, meta, gdal_env, lock, **kwargs):
            seen.append(kwargs.get("open_options"))
            return real(path, meta, gdal_env, lock, **kwargs)

        mocker.patch.object(collection_module, "_lazy_timestep", side_effect=spy)
        _ = collection.data
        assert seen == [("GEOREF_SOURCES=NONE",), ("GEOREF_SOURCES=NONE",)], seen


class TestNetCDFOpenOptions:
    """`NetCDF.read_file` honours the widened `open_options` contract (M2)."""

    def test_read_file_accepts_and_captures(self):
        """A `dict` option is accepted (no `TypeError`) and captured on the container."""
        nc = NetCDF.read_file(
            str(_NETCDF_FIXTURE), open_options={"HONOUR_VALID_RANGE": "NO"}
        )
        assert nc.open_options == ["HONOUR_VALID_RANGE=NO"], "option not captured"

    def test_default_captures_nothing(self):
        """An ordinary NetCDF open captures no options (backward-compatible)."""
        nc = NetCDF.read_file(str(_NETCDF_FIXTURE))
        assert nc.open_options == [], "a plain NetCDF open must capture nothing"

    def test_options_survive_pickle_reopen(self):
        """The captured options round-trip through the NetCDF pickle recipe.

        Test scenario:
            The recipe grew a seventh element (the captured options); a worker
            reopen (unpickle) must reapply them rather than silently dropping.
        """
        nc = NetCDF.read_file(
            str(_NETCDF_FIXTURE), open_options={"HONOUR_VALID_RANGE": "NO"}
        )
        reopened = pickle.loads(pickle.dumps(nc))
        assert reopened.open_options == ["HONOUR_VALID_RANGE=NO"], "options lost on reopen"
