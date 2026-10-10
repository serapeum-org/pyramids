"""Tests for from_stac(rescale=...) and from_stac(cfg=...) (STAC-04 / STAC-07)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyramids.base import _artifacts as artifacts
from pyramids.base._errors import StacAssetError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection, Grid
from pyramids.dataset._stac import from_stac
from pyramids.stac._config import AssetMetadataWarning

pytestmark = pytest.mark.core

_PHYSICAL = [[np.nan, 1.0], [2.0, 3.0]]


def _timestep(collection, index=0):
    """Return one timestep of a cube as a 2-D float array.

    Args:
        collection: The built :class:`DatasetCollection`.
        index: Which timestep to read.

    Returns:
        np.ndarray: The timestep's first band as floats.
    """
    values = np.asarray(collection.iloc(index).read_array(), dtype="float64")
    return values if values.ndim == 2 else values[0]


def _assert_physical(values, label):
    """Assert a timestep carries the packed fixture's physical values.

    Args:
        values: A 2-D array of the timestep's pixels.
        label: What is being checked, for the assertion message.
    """
    assert np.isnan(values[0, 0]), f"{label}: no-data must stay no-data, got {values}"
    assert values[1, 1] == pytest.approx(3.0), f"{label}: 300 * 0.01 != {values[1, 1]}"
    assert values[0, 1] == pytest.approx(1.0), f"{label}: 100 * 0.01 != {values[0, 1]}"


@pytest.fixture
def packed_items(packed_item):
    """Two copies of the packed item, so the cube has two timesteps."""
    second = {**packed_item, "id": "packed-2"}
    second["properties"] = {"datetime": "2023-06-02T00:00:00Z"}
    return [packed_item, second]


@pytest.fixture
def nodataless_item(tmp_path):
    """An item whose asset declares neither a no-data value nor a unit."""
    path = str(tmp_path / "nodataless.tif")
    Dataset.from_array(
        np.array([[1, 2], [3, 4]], dtype="int16"),
        no_data_value=None,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    ).to_file(path)
    return {
        "type": "Feature",
        "id": "thin",
        "collection": "c",
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {"B05": {"href": path, "type": "image/tiff"}},
        "stac_extensions": [],
    }


@pytest.fixture
def file_packed_item(tmp_path):
    """A packed item whose raster declares the scale 0.01 in the file itself.

    The shared `packed_item` fixture deliberately leaves the file's packing at
    identity, which makes "the rebuilt result declares identity packing" true
    of the source as well — an assertion that cannot fail. Here the source
    declares `0.01`, so the two sides can disagree.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        dict: An item over a file-declared-scale raster, with the same scale in
        its `raster:bands`.
    """
    path = str(tmp_path / "file_packed.tif")
    source = Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    source.scale = [0.01]
    source.to_file(path)
    return {
        "type": "Feature",
        "id": "file-packed",
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {
            "data": {
                "href": path,
                "type": "image/tiff",
                "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
            }
        },
        "stac_extensions": [],
    }


@pytest.fixture
def file_packed_items(file_packed_item):
    """Two copies of the file-declared-scale item, so the cube has two timesteps."""
    second = {**file_packed_item, "id": "file-packed-2"}
    second["properties"] = {"datetime": "2023-06-02T00:00:00Z"}
    return [file_packed_item, second]


class TestSingleAssetRescale:
    """Tests for the single-asset path under `rescale=True`."""

    def test_default_keeps_stored_counts(self, packed_items):
        """Without `rescale` the cube still carries the stored counts.

        Test scenario:
            Two packed items, read with today's defaults.
        """
        collection = from_stac(packed_items, asset="data")
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(300.0), (
            f"rescale=False must keep the stored counts, got {values}"
        )

    def test_rescale_gives_physical_values(self, packed_items):
        """`rescale=True` applies the asset's raster:bands packing per timestep.

        Test scenario:
            Two packed items declaring scale 0.01.
        """
        collection = from_stac(packed_items, asset="data", rescale=True)
        assert collection.time_length == 2, (
            f"expected 2 timesteps, got {collection.time_length}"
        )
        _assert_physical(_timestep(collection, 0), "timestep 0")
        _assert_physical(_timestep(collection, 1), "timestep 1")

    def test_rescale_materialises_off_the_asset_url(
        self, packed_items, packed_raster_path
    ):
        """A rescaled timestep is backed by a local copy, not the asset href.

        Test scenario:
            The cube's backing files are compared against the asset path.
        """
        collection = from_stac(packed_items, asset="data", rescale=True)
        backing = str(collection.iloc(0).file_name)
        assert packed_raster_path not in backing, (
            f"a rescaled timestep must not be backed by the raw asset: {backing}"
        )

    def test_rescale_keeps_the_url_when_nothing_is_declared(self, three_local_items):
        """An asset without raster:bands keeps its lazy href.

        Test scenario:
            Items whose assets declare no packing at all.
        """
        collection = from_stac(three_local_items, asset="data", rescale=True)
        backing = str(collection.iloc(0).file_name)
        expected = three_local_items[0]["assets"]["data"]["href"]
        assert backing.replace("/", "\\") == expected.replace("/", "\\"), (
            f"nothing to rescale must keep the asset href, got {backing}"
        )

    def test_unreadable_asset_raises_while_materialising(self, packed_item, tmp_path):
        """A rescale that cannot open its asset raises instead of staying silent.

        Test scenario:
            The item declares a scale, so the materialiser must open the asset —
            but the href points at a missing file and `errors_as_nodata` is off.
        """
        asset = {**packed_item["assets"]["data"], "href": str(tmp_path / "gone.tif")}
        item = {**packed_item, "assets": {"data": asset}}
        with pytest.raises(FileNotFoundError):
            from_stac([item], asset="data", rescale=True)

    def test_unreadable_asset_is_tolerated_under_errors_as_nodata(
        self, packed_item, tmp_path
    ):
        """With errors_as_nodata the failed materialisation is left to the caller.

        Test scenario:
            The second item's href is missing; the materialiser stays silent and
            the timestep is substituted with a no-data plane instead.
        """
        asset = {**packed_item["assets"]["data"], "href": str(tmp_path / "gone.tif")}
        second = {
            **packed_item,
            "id": "packed-2",
            "properties": {"datetime": "2023-06-02T00:00:00Z"},
            "assets": {"data": asset},
        }
        with pytest.warns(RuntimeWarning, match="errors_as_nodata"):
            collection = from_stac(
                [packed_item, second],
                asset="data",
                rescale=True,
                errors_as_nodata=True,
            )
        assert collection.time_length == 2, (
            f"the unreadable timestep should be kept, got {collection.time_length}"
        )
        assert np.isnan(_timestep(collection, 1)).all(), (
            "the substituted timestep should be entirely no-data"
        )

    def test_rescale_result_declares_identity_packing(self, file_packed_items):
        """A rescaled timestep declares identity packing, unlike its source.

        Test scenario:
            The source *file* declares scale 0.01 (the shared `packed_item`
            fixture leaves the file at identity, so there the claim is true of
            the source too and cannot fail). The rescaled timestep must declare
            `[1.0]`, and one `read_array(unpack=True)` must give the physical
            value once — not 0.01x it again.
        """
        source = Dataset.read_file(file_packed_items[0]["assets"]["data"]["href"])
        assert source.scale == [0.01], (
            "the fixture must declare the scale in the file, or the result's identity "
            f"packing is vacuous, got {source.scale}"
        )
        collection = from_stac(file_packed_items, asset="data", rescale=True)
        timestep = collection.iloc(0)
        assert timestep.scale == [1.0], (
            f"the rebuilt timestep must declare identity scale, got {timestep.scale}"
        )
        assert timestep.offset == [0.0], (
            f"the rebuilt timestep must declare identity offset, got {timestep.offset}"
        )
        unpacked = np.asarray(timestep.read_array(unpack=True), dtype="float64")
        values = unpacked if unpacked.ndim == 2 else unpacked[0]
        assert values[1, 1] == pytest.approx(3.0), (
            "300 counts x 0.01 is 3.0; a second application would give 0.03, got "
            f"{values[1, 1]}"
        )

    def test_rescale_combines_with_a_target_grid(self, packed_items):
        """`rescale` and `grid=` apply together.

        Test scenario:
            The packed items matched onto a finer 4x4 template.
        """
        template = Dataset.from_array(
            np.zeros((4, 4), dtype="float32"),
            no_data_value=np.nan,
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=0.5, epsg=4326),
        )
        collection = from_stac(
            packed_items, asset="data", rescale=True, grid=Grid(like=template)
        )
        values = _timestep(collection)
        assert values.shape == (4, 4), f"expected the template grid, got {values.shape}"
        assert np.nanmax(values) == pytest.approx(3.0), (
            f"the aligned cube must still be in physical units, got {values}"
        )


class TestGroupedRescale:
    """Tests for the grouped (mosaicking) path under `rescale=True`."""

    def test_grouped_mosaic_is_physical(self, packed_items):
        """Each group is mosaicked from physical-unit sources.

        Test scenario:
            Two items grouped by id -> two timesteps, both rescaled.
        """
        collection = from_stac(packed_items, asset="data", groupby="id", rescale=True)
        assert collection.time_length == 2, (
            f"expected 2 id groups, got {collection.time_length}"
        )
        _assert_physical(_timestep(collection, 0), "group 0")

    def test_grouped_default_is_unchanged(self, packed_items):
        """Without `rescale` the grouped mosaic keeps the stored counts.

        Test scenario:
            The same items grouped by id, no rescale.
        """
        collection = from_stac(packed_items, asset="data", groupby="id")
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(300.0), (
            f"the grouped default must keep the counts, got {values}"
        )

    def test_grouped_fuse_func_sees_physical_values(self, packed_items):
        """A `fuse_func` is handed rescaled values when `rescale=True`.

        Test scenario:
            A max-fuser over one group of two packed items.
        """

        def _fuse(dst, src):
            """Keep the larger of the two planes, in place."""
            np.copyto(dst, np.fmax(dst, src))

        collection = from_stac(
            packed_items,
            asset="data",
            groupby="solar_day",
            rescale=True,
            fuse_func=_fuse,
        )
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(3.0), (
            f"the fuser must see physical values, got {values}"
        )


class TestMultiAssetRescale:
    """Tests for the multi-asset (band-stacking) path under `rescale=True`."""

    def test_each_asset_is_rescaled_before_stacking(self, packed_raster_path):
        """Per-asset packings are applied before the bands are stacked.

        Test scenario:
            Two assets over the same counts with scales 0.01 and 10.
        """
        item = {
            "id": "two-assets",
            "bbox": [0.0, 0.0, 2.0, 2.0],
            "properties": {"datetime": "2023-06-01T00:00:00Z"},
            "assets": {
                "a": {
                    "href": packed_raster_path,
                    "type": "image/tiff",
                    "raster:bands": [{"scale": 0.01, "nodata": 0}],
                },
                "b": {
                    "href": packed_raster_path,
                    "type": "image/tiff",
                    "raster:bands": [{"scale": 10.0, "nodata": 0}],
                },
            },
        }
        collection = from_stac([item], asset=["a", "b"], rescale=True)
        timestep = collection.iloc(0)
        assert timestep.band_count == 2, (
            f"expected a 2-band timestep, got {timestep.band_count}"
        )
        values = np.asarray(timestep.read_array(), dtype="float64")
        assert values[0][1, 1] == pytest.approx(3.0), (
            f"band a must use scale 0.01, got {values[0]}"
        )
        assert values[1][1, 1] == pytest.approx(3000.0), (
            f"band b must use scale 10, got {values[1]}"
        )


class TestFromStacConfig:
    """Tests for from_stac(cfg=...) — overrides and band aliases."""

    def test_alias_names_the_asset(self, nodataless_item):
        """An alias is resolved per item before the href is looked up.

        Test scenario:
            cfg maps "rededge" -> "B05" for collection "c".
        """
        cfg = {"c": {"aliases": {"rededge": "B05"}}}
        collection = from_stac([nodataless_item], asset="rededge", cfg=cfg)
        values = _timestep(collection)
        assert values[1, 1] == pytest.approx(4.0), (
            f"the aliased asset must be the one read, got {values}"
        )

    def test_unknown_alias_still_raises_for_a_missing_asset(self, nodataless_item):
        """An unaliased, absent asset key fails as it does today.

        Test scenario:
            A key that is neither an alias nor an asset.
        """
        cfg = {"c": {"aliases": {"rededge": "B05"}}}
        with pytest.raises(StacAssetError, match="asset 'nir' not found"):
            from_stac([nodataless_item], asset="nir", cfg=cfg)

    def test_missing_nodata_is_supplied(self, nodataless_item):
        """A cfg nodata fills the gap the item leaves.

        Test scenario:
            The asset declares no no-data; cfg supplies 1.
        """
        cfg = {"c": {"assets": {"*": {"nodata": 1}}}}
        collection = from_stac([nodataless_item], asset="B05", cfg=cfg)
        assert collection.iloc(0).no_data_value[0] == pytest.approx(1.0), (
            f"expected the configured no-data, got {collection.iloc(0).no_data_value}"
        )

    def test_wildcard_collection_applies_without_an_item_collection(
        self, three_local_items
    ):
        """The '*' cfg section reaches items that declare no collection.

        Test scenario:
            Items without a `collection` member and a '*' unit override.
        """
        cfg = {"*": {"assets": {"*": {"unit": "K"}}}}
        collection = from_stac(three_local_items, asset="data", cfg=cfg)
        assert collection.iloc(0).band_units == ["K"], (
            f"expected the configured unit, got {collection.iloc(0).band_units}"
        )

    def test_declared_value_is_not_replaced(self, packed_items):
        """A nodata the asset already declares survives, with a warning.

        Test scenario:
            The packed items declare nodata 0; cfg asks for 7.
        """
        cfg = {"*": {"assets": {"*": {"nodata": 7}}}}
        with pytest.warns(AssetMetadataWarning, match="already declares"):
            collection = from_stac(packed_items, asset="data", cfg=cfg)
        assert collection.iloc(0).no_data_value[0] == pytest.approx(0.0), (
            "an override must not replace a value the asset declares"
        )

    def test_cfg_none_is_byte_identical(self, nodataless_item):
        """Omitting cfg leaves the asset backing the timestep directly.

        Test scenario:
            The default read keeps the href as the backing file.
        """
        collection = from_stac([nodataless_item], asset="B05")
        backing = str(collection.iloc(0).file_name).replace("/", "\\")
        expected = nodataless_item["assets"]["B05"]["href"].replace("/", "\\")
        assert backing == expected, (
            f"the default read must back the timestep with the href, got {backing}"
        )

    def test_classmethod_default_path_still_works(self, three_local_items):
        """The public classmethod is unaffected by the new keywords.

        Test scenario:
            DatasetCollection.from_stac with no new arguments.
        """
        collection = DatasetCollection.from_stac(three_local_items, asset="data")
        assert collection.time_length == 3, (
            f"expected 3 timesteps, got {collection.time_length}"
        )


class TestFailedMaterialisationIsNeverSilent:
    """H3: a failed rescale must not leave a raw-DN timestep in the cube.

    `_materialise_asset` answers `None` to mean "nothing had to be applied",
    which makes the caller keep the asset's own href as the timestep's backing.
    Tolerating a *materialisation* failure that way put stored counts in a cube
    whose other timesteps carry physical units, with no warning — the numbers
    all look plausible, so any reduction over the time axis is wrong by the
    packing factor for that subset. Only the **open** is tolerated now.
    """

    def test_a_materialisation_failure_raises_even_when_tolerated(
        self, packed_items, monkeypatch
    ):
        """errors_as_nodata does not swallow a failure after the open.

        Test scenario:
            The asset opens, then `materialise` raises `RuntimeError` — a
            truncated range response or a decompression error mid-read. With
            `errors_as_nodata=True` that used to return the unrescaled href.
        """
        import pyramids.stac._config as config

        def boom(_dataset, _overrides):
            raise RuntimeError("truncated range response mid-read")

        monkeypatch.setattr(config, "materialise", boom)
        with pytest.raises(RuntimeError, match="truncated range response"):
            from_stac(packed_items, asset="data", rescale=True, errors_as_nodata=True)

    def test_no_timestep_is_left_in_stored_counts(self, packed_items, monkeypatch):
        """The failure cannot produce a cube mixing counts and physical units.

        Test scenario:
            The materialiser fails for the second item only. Before the fix the
            build succeeded with timestep 0 physical (1.0 / 2.0 / 3.0) and
            timestep 1 raw (100 / 200 / 300); now nothing is returned at all.
        """
        import pyramids.stac._config as config

        real = config.materialise
        calls = {"n": 0}

        def flaky(dataset, overrides):
            calls["n"] += 1
            if calls["n"] == 1:
                return real(dataset, overrides)
            raise OSError("the second asset died mid-read")

        monkeypatch.setattr(config, "materialise", flaky)
        with pytest.raises(OSError, match="died mid-read"):
            from_stac(packed_items, asset="data", rescale=True, errors_as_nodata=True)

    def test_an_open_failure_is_still_tolerated(self, packed_item, tmp_path):
        """The narrowing kept the case `errors_as_nodata` exists for.

        Test scenario:
            The second item's href does not exist, so the *open* fails and the
            timestep becomes a no-data plane rather than an error.
        """
        asset = {**packed_item["assets"]["data"], "href": str(tmp_path / "gone.tif")}
        second = {
            **packed_item,
            "id": "packed-2",
            "properties": {"datetime": "2023-06-02T00:00:00Z"},
            "assets": {"data": asset},
        }
        with pytest.warns(RuntimeWarning, match="errors_as_nodata"):
            collection = from_stac(
                [packed_item, second],
                asset="data",
                rescale=True,
                errors_as_nodata=True,
            )
        assert collection.time_length == 2, (
            f"the unreadable timestep should be kept, got {collection.time_length}"
        )
        assert np.isnan(_timestep(collection, 1)).all(), (
            "the substituted timestep should be entirely no-data"
        )


class TestMaterialisedIntermediatesAreReclaimed:
    """M8: the copies that do not back the collection are deleted in-call.

    Every `rescale` / `cfg` build writes a full-resolution raster per timestep
    under the process artefact root, and nothing there is reclaimed before the
    interpreter exits. The copies the *returned collection* is backed by have
    to stay, but a grouped build's per-source copies and a multi-asset build's
    per-band copies are consumed inside the call.
    """

    @staticmethod
    def _artefacts():
        """Return every file currently under the process artefact root."""
        root = artifacts._ROOT
        found = []
        if root is not None:
            for base, _dirs, names in os.walk(root):
                found.extend(os.path.join(base, name) for name in names)
        return found

    def test_multi_asset_keeps_only_the_band_stacks(self, packed_raster_path):
        """The per-band copies are gone; one stack per timestep remains.

        Test scenario:
            Two items x two packed assets rescaled. Four band copies are
            written and consumed, so only the two stacked timesteps survive.
        """
        asset = {
            "href": packed_raster_path,
            "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
        }
        items = [
            {
                "id": f"item-{index}",
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "properties": {"datetime": f"2023-06-0{index + 1}T00:00:00Z"},
                "assets": {"red": dict(asset), "nir": dict(asset)},
            }
            for index in range(2)
        ]
        before = set(self._artefacts())
        from_stac(items, asset=["red", "nir"], rescale=True)
        added = [path for path in self._artefacts() if path not in before]
        leftovers = [path for path in added if "_band_" in os.path.basename(path)]
        assert not leftovers, (
            "the per-band copies are consumed by the stack and must be reclaimed, "
            f"found {[os.path.basename(p) for p in leftovers]}"
        )
        assert len(added) == 2, (
            "only the two band stacks should survive the call, got "
            f"{[os.path.basename(p) for p in added]}"
        )

    def test_grouped_keeps_only_the_group_mosaics(self, packed_items):
        """The per-source copies are gone; one mosaic per group remains.

        Test scenario:
            Two packed items grouped by id, so two sources are materialised and
            two mosaics are written.
        """
        before = set(self._artefacts())
        from_stac(packed_items, asset="data", groupby="id", rescale=True)
        added = [path for path in self._artefacts() if path not in before]
        leftovers = [
            path for path in added if os.path.basename(path).startswith("source_")
        ]
        assert not leftovers, (
            "the materialised sources are consumed by the mosaic and must be "
            f"reclaimed, found {[os.path.basename(p) for p in leftovers]}"
        )
        assert len(added) == 2, (
            "only the two group mosaics should survive the call, got "
            f"{[os.path.basename(p) for p in added]}"
        )

    def test_single_asset_copies_survive_because_they_back_the_cube(self, packed_items):
        """The reclaim is scoped: a timestep's own backing file is not deleted.

        Test scenario:
            A single-asset rescale, whose materialised copies *are* the
            collection's files, so they must still be readable afterwards.
        """
        before = set(self._artefacts())
        collection = from_stac(packed_items, asset="data", rescale=True)
        added = [path for path in self._artefacts() if path not in before]
        assert len(added) == 2, (
            f"one materialised copy per timestep is expected, got {len(added)}"
        )
        _assert_physical(_timestep(collection, 1), "the second timestep")
