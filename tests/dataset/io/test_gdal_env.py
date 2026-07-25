"""Tests for the cloud config a Dataset captures at open time (ARC-24).

A signed remote open (`pyramids.stac.load_asset` with a Requester-Pays or
bearer signer) authenticates the handle it creates, but several read paths do
not use that handle: `threadsafe=True` opens one per thread, a lazy `chunks=`
read opens inside the dask task, and unpickling on a worker re-opens from the
path. `Dataset.read_file(..., gdal_env=...)` captures the config so every one of
those re-opens installs it again.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.engines import io as io_engine
from tests._helpers import write_raster

pytestmark = pytest.mark.core

ENV = {"AWS_REQUEST_PAYER": "requester"}


@pytest.fixture
def tif(tmp_path):
    """A small single-band GeoTIFF path.

    Returns:
        str: Path to a 4x4 EPSG:4326 raster filled with 7.0.
    """
    return write_raster(tmp_path / "signed.tif", np.full((4, 4), 7.0, "float32"), (0.0, 4.0))


class TestGdalEnvCapture:
    """Tests for `Dataset.read_file(gdal_env=...)` and the `gdal_env` property."""

    def test_default_open_captures_nothing(self, tif):
        """An ordinary open leaves the captured config empty.

        Test scenario:
            No `gdal_env=` argument -> `{}`, so unsigned reads pay nothing.
        """
        assert Dataset.read_file(tif).gdal_env == {}, "an ordinary open must capture nothing"

    def test_open_captures_the_env(self, tif):
        """`gdal_env=` is retained on the dataset.

        Test scenario:
            A Requester-Pays config survives the open.
        """
        assert Dataset.read_file(tif, gdal_env=ENV).gdal_env == ENV, "env not captured"

    def test_property_returns_a_copy(self, tif):
        """Mutating the returned mapping cannot corrupt the dataset's own copy.

        Test scenario:
            Clearing the returned dict leaves the dataset's config intact.
        """
        ds = Dataset.read_file(tif, gdal_env=ENV)
        ds.gdal_env.clear()
        assert ds.gdal_env == ENV, "the property must hand out a copy"

    def test_env_is_active_during_the_open(self, tif, monkeypatch):
        """The config is installed while the file is opened, not only after.

        Test scenario:
            A reader stub reads `AWS_REQUEST_PAYER` at open time.
        """
        captured = {}
        original = Dataset.read_file.__func__

        def spy(path, *args, **kwargs):
            captured["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return None

        monkeypatch.setattr("pyramids._io.read_file", spy)
        with pytest.raises(Exception):
            original(Dataset, tif, gdal_env=ENV)
        assert captured["payer"] == "requester", f"env not active at open: {captured}"


class TestGdalEnvOnReads:
    """The captured config is re-installed around reads that re-open the file."""

    def test_shared_handle_read_installs_the_env(self, tif, monkeypatch):
        """A plain eager read runs under the captured config.

        Test scenario:
            A VRT-backed dataset opens its sources here, so the config has to be
            active even though the dataset's own handle already exists.
        """
        seen = {}
        original = io_engine.IO._read_block

        def spy(self, *args, **kwargs):
            seen["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(io_engine.IO, "_read_block", spy)
        Dataset.read_file(tif, gdal_env=ENV).read_array(window=[0, 0, 2, 2])
        assert seen["payer"] == "requester", f"env not active during read: {seen}"

    def test_threadsafe_read_installs_the_env(self, tif, monkeypatch):
        """A per-thread re-open runs under the captured config.

        Test scenario:
            `threadsafe=True` opens a fresh handle from the path — without the
            captured config that handle is built with no credentials.
        """
        seen = {}
        original = io_engine.IO._read_via_handle

        def spy(self, *args, **kwargs):
            seen["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(io_engine.IO, "_read_via_handle", spy)
        Dataset.read_file(tif, gdal_env=ENV).read_array(threadsafe=True)
        assert seen["payer"] == "requester", f"env not active in threadsafe read: {seen}"

    def test_unsigned_read_installs_nothing(self, tif, monkeypatch):
        """An ordinary dataset leaves the option unset during a read.

        Test scenario:
            No captured config -> no GDAL config is touched.
        """
        seen = {}
        original = io_engine.IO._read_block

        def spy(self, *args, **kwargs):
            seen["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(io_engine.IO, "_read_block", spy)
        Dataset.read_file(tif).read_array(window=[0, 0, 2, 2])
        assert seen["payer"] is None, f"unexpected config for an unsigned open: {seen}"

    def test_values_are_unchanged_by_the_env(self, tif):
        """Capturing a config does not alter what is read.

        Test scenario:
            The pixels match a plain open of the same file.
        """
        signed = Dataset.read_file(tif, gdal_env=ENV).read_array()
        plain = Dataset.read_file(tif).read_array()
        np.testing.assert_array_equal(signed, plain)


class TestGdalEnvSurvivesPickle:
    """The recipe that re-opens on a dask worker carries the config."""

    def test_reduce_carries_the_env(self, tif):
        """`__reduce__` puts the captured config in the recipe tuple.

        Test scenario:
            The fourth recipe argument is the config dict.
        """
        _fn, args = Dataset.read_file(tif, gdal_env=ENV).__reduce__()
        assert args[3] == ENV, f"env missing from the pickle recipe: {args}"

    def test_round_trip_restores_the_env(self, tif):
        """An unpickled dataset re-opens with the same config attached.

        Test scenario:
            pickle -> unpickle keeps `gdal_env` and the pixel values.
        """
        ds = Dataset.read_file(tif, gdal_env=ENV)
        restored = pickle.loads(pickle.dumps(ds))
        assert restored.gdal_env == ENV, f"env lost across pickle: {restored.gdal_env}"
        np.testing.assert_array_equal(restored.read_array(), ds.read_array())

    def test_legacy_three_element_recipe_still_loads(self, tif):
        """A recipe written before the config existed still reconstructs.

        Test scenario:
            Calling the reconstructor with three arguments yields a dataset with
            an empty config rather than a TypeError.
        """
        from pyramids.dataset.abstract_dataset import _reconstruct_dataset

        ds = _reconstruct_dataset(Dataset, tif, "read_only")
        assert ds.gdal_env == {}, f"legacy recipe should capture nothing: {ds.gdal_env}"


class TestCollectionHandsEnvToTimesteps:
    """A signed collection passes its config down to each timestep Dataset."""

    def test_per_timestep_datasets_carry_the_env(self, tmp_path):
        """Path A datasets inherit the collection's captured config.

        Test scenario:
            Without it, a per-thread or lazy read of a timestep re-opens the
            file unauthenticated even though the collection knows the config.
        """
        paths = [
            write_raster(tmp_path / f"t{i}.tif", np.full((3, 3), float(i), "float32"), (0.0, 3.0))
            for i in range(2)
        ]
        collection = DatasetCollection.from_files(paths, gdal_env=ENV)
        assert [ds.gdal_env for ds in collection.datasets] == [ENV, ENV], (
            "timestep datasets did not inherit the collection env"
        )

    def test_unsigned_collection_leaves_timesteps_clean(self, tmp_path):
        """An ordinary collection hands down nothing.

        Test scenario:
            No config on the collection -> none on its timesteps.
        """
        paths = [
            write_raster(tmp_path / f"u{i}.tif", np.full((3, 3), float(i), "float32"), (0.0, 3.0))
            for i in range(2)
        ]
        collection = DatasetCollection.from_files(paths)
        assert all(ds.gdal_env == {} for ds in collection.datasets), (
            "an unsigned collection must not attach a config"
        )


class TestGdalEnvOnLazyReads:
    """The captured config travels with the dask tasks of a lazy read."""

    def test_chunk_tasks_receive_the_env(self, tif, monkeypatch):
        """`_read_chunk` is handed the dataset's captured config.

        Test scenario:
            Chunks open the file inside the task, after (and possibly in another
            process from) the call that built the graph, so the config cannot be
            an ambient one — it has to travel as a plain dict.
        """
        pytest.importorskip("dask")
        from pyramids.dataset.ops import io as ops_io

        seen = {}
        original = ops_io._read_chunk

        def spy(block_info=None, *, gdal_env=None, **kwargs):
            seen["gdal_env"] = gdal_env
            return original(block_info, gdal_env=gdal_env, **kwargs)

        monkeypatch.setattr(ops_io, "_read_chunk", spy)
        Dataset.read_file(tif, gdal_env=ENV).read_array(chunks=-1).compute()
        assert seen["gdal_env"] == ENV, f"env not shipped to the chunk task: {seen}"

    def test_unsigned_chunk_tasks_receive_none(self, tif, monkeypatch):
        """An ordinary lazy read ships no config at all.

        Test scenario:
            `None` rather than an empty dict, so the task pickles a little
            smaller and the worker takes the nullcontext path.
        """
        pytest.importorskip("dask")
        from pyramids.dataset.ops import io as ops_io

        seen = {}
        original = ops_io._read_chunk

        def spy(block_info=None, *, gdal_env=None, **kwargs):
            seen["gdal_env"] = gdal_env
            return original(block_info, gdal_env=gdal_env, **kwargs)

        monkeypatch.setattr(ops_io, "_read_chunk", spy)
        Dataset.read_file(tif).read_array(chunks=-1).compute()
        assert seen["gdal_env"] is None, f"unexpected env on an unsigned read: {seen}"
