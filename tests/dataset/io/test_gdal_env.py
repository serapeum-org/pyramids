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
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.collection import _SIDECAR_SUFFIXES
from pyramids.dataset.engines import io as io_engine
from tests._helpers import write_raster

pytestmark = pytest.mark.core

ENV = {"AWS_REQUEST_PAYER": "requester"}


class _StopOpen(Exception):
    """Sentinel raised by the reader spy so the open aborts after its probe."""


@pytest.fixture
def tif(tmp_path):
    """A small single-band GeoTIFF path.

    Returns:
        str: Path to a 4x4 EPSG:4326 raster filled with 7.0.
    """
    return write_raster(
        tmp_path / "signed.tif", np.full((4, 4), 7.0, "float32"), (0.0, 4.0)
    )


class TestGdalEnvCapture:
    """Tests for `Dataset.read_file(gdal_env=...)` and the `gdal_env` property."""

    def test_default_open_captures_nothing(self, tif):
        """An ordinary open leaves the captured config empty.

        Test scenario:
            No `gdal_env=` argument -> `{}`, so unsigned reads pay nothing.
        """
        assert Dataset.read_file(tif).gdal_env == {}, (
            "an ordinary open must capture nothing"
        )

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

        def spy(path, *args, **kwargs):
            captured["payer"] = gdal.GetConfigOption("AWS_REQUEST_PAYER")
            raise _StopOpen

        monkeypatch.setattr("pyramids._io.read_file", spy)
        with pytest.raises(_StopOpen):
            Dataset.read_file(tif, gdal_env=ENV)
        assert captured.get("payer") == "requester", (
            f"env not active at open: {captured}"
        )


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
        assert seen["payer"] == "requester", (
            f"env not active in threadsafe read: {seen}"
        )

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
            write_raster(
                tmp_path / f"t{i}.tif",
                np.full((3, 3), float(i), "float32"),
                (0.0, 3.0),
            )
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
            write_raster(
                tmp_path / f"u{i}.tif",
                np.full((3, 3), float(i), "float32"),
                (0.0, 3.0),
            )
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


class TestGdalEnvOnEveryReadPath:
    """Coverage is structural: every decorated read primitive installs the config.

    The first cut wrapped three call sites by hand, so `read_array()` carried the
    credentials but `read_array(out_shape=...)`, `get_tile()` and the COG-engine
    reads did not — an inconsistency users would hit as a puzzle.
    """

    @staticmethod
    def _payer_during(callable_):
        """Run `callable_` and return AWS_REQUEST_PAYER as GDAL saw it."""
        seen = {}
        originals = (gdal.Band.ReadAsArray, gdal.Dataset.ReadAsArray)

        def make_spy(original):
            def spy(self, *args, **kwargs):
                seen.setdefault("payer", gdal.GetConfigOption("AWS_REQUEST_PAYER"))
                return original(self, *args, **kwargs)

            return spy

        gdal.Band.ReadAsArray = make_spy(originals[0])
        gdal.Dataset.ReadAsArray = make_spy(originals[1])
        try:
            callable_()
        finally:
            gdal.Band.ReadAsArray, gdal.Dataset.ReadAsArray = originals
        return seen.get("payer")

    def test_decimated_read_installs_the_env(self, tif):
        """`out_shape=` decimated reads carry the config.

        Test scenario:
            The natural way to preview a large scene must not 403 where a plain
            read succeeds.
        """
        ds = Dataset.read_file(tif, gdal_env=ENV)
        payer = self._payer_during(lambda: ds.read_array(out_shape=(2, 2)))
        assert payer == "requester", f"decimated read ran without the config: {payer}"

    def test_boundless_read_installs_the_env(self, tif):
        """Boundless reads carry the config.

        Test scenario:
            A window extending past the raster still reads real pixels inside it.
        """
        ds = Dataset.read_file(tif, gdal_env=ENV)
        payer = self._payer_during(
            lambda: ds.read_array(window=[-1, -1, 3, 3], boundless=True)
        )
        assert payer == "requester", f"boundless read ran without the config: {payer}"

    def test_get_tile_installs_the_env(self, tif):
        """Tiled iteration carries the config.

        Test scenario:
            `get_tile` walks the raster block by block off the shared handle.
        """
        ds = Dataset.read_file(tif, gdal_env=ENV)
        payer = self._payer_during(lambda: list(ds.get_tile(size=2)))
        assert payer == "requester", f"get_tile ran without the config: {payer}"

    def test_cog_preview_installs_the_env(self, tmp_path):
        """A COG-engine read carries the config.

        Test scenario:
            `ds.cog.preview()` opens and decimates through the COG engine, which
            has its own read path independent of `IO`.
        """
        source = Dataset.create_from_array(
            np.full((64, 64), 5.0, "float32"),
            top_left_corner=(0.0, 64.0),
            cell_size=1.0,
            epsg=4326,
        )
        cog_path = str(source.to_cog(tmp_path / "c.tif"))
        ds = Dataset.read_file(cog_path, gdal_env=ENV)
        payer = self._payer_during(lambda: ds.cog.preview(max_size=8))
        assert payer == "requester", f"COG preview ran without the config: {payer}"


_READDIR = "GDAL_DISABLE_READDIR_ON_OPEN"


def _write_folder_of_tifs(folder, count=2):
    """Write ``count`` tiny 3x3 GeoTIFFs into ``folder`` and return their paths."""
    return [
        write_raster(
            folder / f"t{i}.tif", np.full((3, 3), float(i), "float32"), (0.0, 3.0)
        )
        for i in range(count)
    ]


class TestFromFilesDirScanDefault:
    """`from_files` defaults GDAL_DISABLE_READDIR_ON_OPEN for sidecar-free folders (#1010)."""

    def test_sidecar_free_folder_defaults_empty_dir(self, tmp_path):
        """A folder with no sidecars gets GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR.

        Test scenario:
            The auto-default is persisted on the collection and handed to every timestep,
            so each open skips GDAL's per-open directory rescan.
        """
        _write_folder_of_tifs(tmp_path)
        col = DatasetCollection.from_files(str(tmp_path))
        assert col._gdal_env.get(_READDIR) == "EMPTY_DIR", (
            f"sidecar-free folder should default EMPTY_DIR, got {col._gdal_env}"
        )
        assert all(ds.gdal_env.get(_READDIR) == "EMPTY_DIR" for ds in col.datasets), (
            "timesteps must inherit the directory-scan default"
        )

    def test_lazy_index_open_carries_the_env(self, tmp_path):
        """The single-index lazy path (iloc -> _dataset_at) also carries EMPTY_DIR.

        Test scenario:
            iloc opens one timestep without materialising the bulk ``datasets`` list,
            so its handle must still inherit the directory-scan default.
        """
        _write_folder_of_tifs(tmp_path)
        col = DatasetCollection.from_files(str(tmp_path))
        assert col.iloc(0).gdal_env.get(_READDIR) == "EMPTY_DIR", (
            "the single-index lazy open must inherit the directory-scan default"
        )

    def test_folder_with_aux_xml_leaves_scan_on(self, tmp_path):
        """A folder holding a .aux.xml sidecar keeps the per-open rescan on (no EMPTY_DIR)."""
        _write_folder_of_tifs(tmp_path)
        (tmp_path / "t0.tif.aux.xml").write_text(
            "<PAMDataset></PAMDataset>", encoding="utf-8"
        )
        col = DatasetCollection.from_files(str(tmp_path))
        assert _READDIR not in col._gdal_env, (
            f"a folder with a sidecar must not disable the scan: {col._gdal_env}"
        )
        assert col._gdal_env == {}, f"no env should be installed, got {col._gdal_env}"

    def test_explicit_gdal_env_overrides_default(self, tmp_path):
        """A caller's GDAL_DISABLE_READDIR_ON_OPEN wins over the sidecar-free default."""
        _write_folder_of_tifs(tmp_path)
        col = DatasetCollection.from_files(str(tmp_path), gdal_env={_READDIR: "FALSE"})
        assert col._gdal_env.get(_READDIR) == "FALSE", (
            f"an explicit gdal_env value must override the default, got {col._gdal_env}"
        )

    def test_explicit_gdal_env_merges_with_default(self, tmp_path):
        """An unrelated caller key is layered on top of the sidecar-free default."""
        _write_folder_of_tifs(tmp_path)
        col = DatasetCollection.from_files(
            str(tmp_path), gdal_env={"AWS_REQUEST_PAYER": "requester"}
        )
        assert col._gdal_env.get(_READDIR) == "EMPTY_DIR", (
            "the auto-default should remain"
        )
        assert col._gdal_env.get("AWS_REQUEST_PAYER") == "requester", (
            "the caller's key should be merged in"
        )

    def test_explicit_sequence_leaves_scan_on(self, tmp_path):
        """An explicit file sequence gets no directory-scan default (no listing to trust)."""
        paths = _write_folder_of_tifs(tmp_path)
        col = DatasetCollection.from_files(paths)
        assert col._gdal_env == {}, (
            f"a file sequence must not default EMPTY_DIR, got {col._gdal_env}"
        )


class TestResolveFilesAndScanSafe:
    """Unit tests for `_resolve_files_and_scan_safe` and the `_resolve_files` delegate."""

    def test_folder_without_sidecars_is_scan_safe(self, tmp_path):
        """A folder of plain rasters resolves the matched list with scan_safe True."""
        _write_folder_of_tifs(tmp_path)
        resolved, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.tif"
        )
        assert [Path(p).name for p in resolved] == ["t0.tif", "t1.tif"], (
            f"resolved list wrong: {resolved}"
        )
        assert scan_safe is True, "no sidecars -> scan is safe to skip"

    @pytest.mark.parametrize("suffix", _SIDECAR_SUFFIXES)
    def test_each_sidecar_suffix_is_detected(self, tmp_path, suffix):
        """Every companion suffix in _SIDECAR_SUFFIXES flips scan_safe to False.

        Args:
            suffix: A recognised companion suffix, derived from the constant so a new
                suffix added to _SIDECAR_SUFFIXES is covered automatically.
        """
        _write_folder_of_tifs(tmp_path)
        (tmp_path / f"d{suffix}").write_text("", encoding="utf-8")
        _, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.tif"
        )
        assert scan_safe is False, (
            f"a {suffix!r} companion must keep the directory scan on"
        )

    def test_uppercase_sidecar_is_detected(self, tmp_path):
        """An uppercase sidecar is detected (case-insensitive match, #1010 M1).

        Test scenario:
            On case-insensitive filesystems GDAL discovers ``T0.TIF.OVR`` regardless
            of case, so the scan must stay on even though the suffix is upper-case.
        """
        _write_folder_of_tifs(tmp_path)
        (tmp_path / "T0.TIF.OVR").write_text("", encoding="utf-8")
        _, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.tif"
        )
        assert scan_safe is False, (
            "an uppercase sidecar must keep the directory scan on"
        )

    def test_non_tiff_world_file_is_detected(self, tmp_path):
        """A format-specific world file (.pgw for PNG) blocks the scan-skip (#1010 M2)."""
        (tmp_path / "a.png").write_text("", encoding="utf-8")
        (tmp_path / "b.png").write_text("", encoding="utf-8")
        (tmp_path / "a.pgw").write_text("", encoding="utf-8")
        resolved, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.png"
        )
        assert [Path(p).name for p in resolved] == ["a.png", "b.png"], (
            f"only the .png rasters should resolve: {resolved}"
        )
        assert scan_safe is False, "a .pgw world file must keep the directory scan on"

    def test_sidecar_not_matching_glob_still_counts(self, tmp_path):
        """A sidecar is detected even though it does not match the raster glob."""
        _write_folder_of_tifs(tmp_path)
        (tmp_path / "overviews.ovr").write_text("", encoding="utf-8")
        resolved, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.tif"
        )
        assert all(p.endswith(".tif") for p in resolved), (
            "the .ovr must not be a timestep"
        )
        assert scan_safe is False, "a non-matching sidecar must still be detected"

    def test_single_file_is_not_scan_safe(self, tmp_path):
        """A single explicit file resolves with scan_safe False (nothing scanned)."""
        path = write_raster(
            tmp_path / "one.tif", np.zeros((2, 2), "float32"), (0.0, 2.0)
        )
        resolved, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            path, "*.tif"
        )
        assert [Path(p).name for p in resolved] == ["one.tif"], (
            f"wrong resolved: {resolved}"
        )
        assert scan_safe is False, "an explicit file is never auto-skipped"

    def test_sequence_is_not_scan_safe(self, tmp_path):
        """An explicit sequence resolves (order preserved) with scan_safe False."""
        paths = _write_folder_of_tifs(tmp_path)
        resolved, scan_safe = DatasetCollection._resolve_files_and_scan_safe(
            paths, "*.tif"
        )
        assert [Path(p).name for p in resolved] == ["t0.tif", "t1.tif"], (
            f"sequence order should be preserved: {resolved}"
        )
        assert scan_safe is False, "an explicit sequence is never auto-skipped"

    def test_resolve_files_delegates_and_returns_list(self, tmp_path):
        """`_resolve_files` returns exactly the path list from the scan-safe resolver."""
        _write_folder_of_tifs(tmp_path)
        expected, _ = DatasetCollection._resolve_files_and_scan_safe(
            str(tmp_path), "*.tif"
        )
        assert DatasetCollection._resolve_files(str(tmp_path), "*.tif") == expected, (
            "_resolve_files must still return the plain resolved list"
        )

    def test_missing_folder_raises(self, tmp_path):
        """A folder that does not exist raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DatasetCollection._resolve_files_and_scan_safe(
                str(tmp_path / "nope"), "*.tif"
            )

    def test_no_glob_match_raises(self, tmp_path):
        """A folder with no file matching ``glob`` raises ``FileNotFoundError``."""
        (tmp_path / "note.txt").write_text("", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="matched glob"):
            DatasetCollection._resolve_files_and_scan_safe(str(tmp_path), "*.tif")

    def test_empty_sequence_raises(self):
        """An empty sequence raises ``ValueError``."""
        with pytest.raises(ValueError, match="at least one path"):
            DatasetCollection._resolve_files_and_scan_safe([], "*.tif")
