"""Tests for :meth:`DatasetCollection.from_stac`.

DASK-19: thin STAC loader — takes a sequence of STAC Items (duck-
typed: :class:`pystac.Item` objects, raw JSON dicts, or anything
with an ``.assets`` dict mapping asset keys to objects / dicts
bearing an ``href``), extracts a named asset's href from each, and
builds a :class:`DatasetCollection`. No pystac dependency in
pyramids — tests use raw dicts to prove the duck-typed contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import (
    AlignmentError,
    OptionalPackageDoesNotExist,
    StacAssetError,
)
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._stac import (
    _horizontal_bounds,
    _item_centroid_lon,
    _item_intersects_bbox,
    _lon_overlaps,
    _solar_day,
    _validate_lonlat_bbox,
)

pytestmark = pytest.mark.core


@pytest.fixture
def three_tifs(tmp_path):
    """Three small GeoTIFFs, each with a different fill value."""
    paths = []
    for i in range(3):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"tile_{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.fixture
def stac_items(three_tifs):
    """Wrap three local GeoTIFFs as raw STAC JSON dict items.

    This fixture deliberately uses plain dicts rather than
    :class:`pystac.Item` objects — the pyramids ``from_stac`` loader
    is fully duck-typed and must work for callers who hand-build JSON
    from an HTTP response without touching pystac.
    """
    return [
        {
            "id": f"item-{i}",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "assets": {"data": {"href": path}},
        }
        for i, path in enumerate(three_tifs)
    ]


class TestFromStac:
    """Happy-path: iterable of STAC items → DatasetCollection."""

    def test_returns_dataset_collection(self, stac_items):
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        assert isinstance(collection, DatasetCollection)
        assert collection.time_length == 3

    def test_files_match_asset_hrefs(self, stac_items, three_tifs):
        """Asset hrefs should round-trip to the same on-disk files.

        STAC hrefs are URL-shaped (forward slashes) on every platform,
        but tmp_path fixtures yield native-separator paths on Windows.
        Compare normalised forms so the test passes regardless.
        """
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        left = [Path(p).resolve() for p in collection.files]
        right = [Path(p).resolve() for p in three_tifs]
        assert (
            left == right
        ), f"files mismatch (normalised): got {left}, expected {right}"

    def test_lazy_data_computes(self, stac_items):
        try:
            import_dask("dask not installed")
        except OptionalPackageDoesNotExist:
            pytest.skip("dask not installed")
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        arr = collection.data.compute()
        assert arr.shape[0] == 3
        for i in range(3):
            assert (arr[i] == i + 1).all()


class TestPatchUrl:
    """patch_url rewrites every href before it becomes a file path."""

    def test_patch_url_called_per_href(self, stac_items):
        seen: list[str] = []

        def patch(href: str) -> str:
            seen.append(href)
            return href

        DatasetCollection.from_stac(stac_items, asset="data", patch_url=patch)
        assert len(seen) == 3


class _RecordingSigner:
    """Signer stand-in recording every href and advertising a GDAL env (H3)."""

    def __init__(self, env=None, suffix=""):
        self._env = dict(env or {})
        self.suffix = suffix
        self.seen: list[str] = []

    def sign_href(self, href):
        """Record the href and append the configured suffix."""
        self.seen.append(href)
        return f"{href}{self.suffix}"

    def gdal_env(self):
        """Return the advertised GDAL config mapping."""
        return dict(self._env)


class TestFromStacSigner:
    """H3: from_stac gains signer= (sign every href + capture gdal_env)."""

    def test_sign_href_applied_per_item(self, stac_items, three_tifs):
        """signer.sign_href fires once per item, before files are opened.

        Test scenario:
            An identity signer records each resolved href; its ``seen`` list
            equals the three asset hrefs and the collection still builds.
        """
        signer = _RecordingSigner()
        coll = DatasetCollection.from_stac(stac_items, asset="data", signer=signer)
        assert coll.time_length == 3, f"expected 3 timesteps, got {coll.time_length}"
        assert (
            signer.seen == three_tifs
        ), f"sign_href should see each href once, got {signer.seen}"

    def test_gdal_env_captured_on_collection(self, stac_items):
        """The signer's gdal_env() is persisted on the returned collection.

        Test scenario:
            A signer advertising GDAL_HTTP_TIMEOUT=30 leaves that mapping on
            collection._gdal_env (a harmless local-read option here).
        """
        signer = _RecordingSigner(env={"GDAL_HTTP_TIMEOUT": "30"})
        coll = DatasetCollection.from_stac(stac_items, asset="data", signer=signer)
        assert coll._gdal_env == {
            "GDAL_HTTP_TIMEOUT": "30"
        }, f"signer env not captured: {coll._gdal_env}"

    def test_no_signer_empty_gdal_env(self, stac_items):
        """Without a signer the collection captures no GDAL config.

        Test scenario:
            from_stac(signer=None) leaves _gdal_env empty (no behaviour change).
        """
        coll = DatasetCollection.from_stac(stac_items, asset="data")
        assert coll._gdal_env == {}, f"expected empty env, got {coll._gdal_env}"

    def test_patch_url_runs_before_signer(self, stac_items, monkeypatch):
        """patch_url is applied before signer.sign_href; both reach from_files.

        Test scenario:
            patch_url appends '?p' and the signer appends '?s'; from_files (stubbed
            to capture, no file open) receives hrefs ending '?p?s' and the
            signer's gdal_env.
        """
        captured: dict = {}

        def fake_from_files(files, *, meta=None, gdal_env=None):
            captured["files"] = list(files)
            captured["gdal_env"] = gdal_env
            return "COLL"

        monkeypatch.setattr(
            DatasetCollection,
            "from_files",
            classmethod(
                lambda cls, files, *, meta=None, gdal_env=None: fake_from_files(
                    files, meta=meta, gdal_env=gdal_env
                )
            ),
        )
        signer = _RecordingSigner(env={"AWS_REQUEST_PAYER": "requester"}, suffix="?s")
        DatasetCollection.from_stac(
            stac_items, asset="data", patch_url=lambda h: f"{h}?p", signer=signer
        )
        assert all(
            f.endswith("?p?s") for f in captured["files"]
        ), f"patch_url should run before signer: {captured['files']}"
        assert captured["gdal_env"] == {
            "AWS_REQUEST_PAYER": "requester"
        }, f"signer env not forwarded to from_files: {captured['gdal_env']}"

    def test_from_files_gdal_env_persisted(self, three_tifs):
        """from_files(gdal_env=...) persists the mapping on the collection.

        Test scenario:
            A direct from_files call with a GDAL env stores it on _gdal_env.
        """
        coll = DatasetCollection.from_files(
            three_tifs, gdal_env={"GDAL_HTTP_TIMEOUT": "15"}
        )
        assert coll._gdal_env == {
            "GDAL_HTTP_TIMEOUT": "15"
        }, f"env not persisted: {coll._gdal_env}"


class TestCollectionGdalEnvLazyReads:
    """H4: the persisted gdal_env is installed around every lazy read."""

    def test_path_a_datasets_open_under_env(self, three_tifs, monkeypatch):
        """Path A: the `datasets` property opens each file under the env.

        Test scenario:
            A spy on Dataset.read_file records the sentinel GDAL option at open
            time; with a persisted env every open (template + per-timestep) sees
            it. Proves a signed file-backed collection authenticates Path A.
        """
        from pyramids.dataset import collection as coll_mod

        seen: list[str | None] = []
        real = coll_mod.Dataset.read_file

        def spy(*args, **kwargs):
            seen.append(gdal.GetConfigOption("PYRAMIDS_TEST_KEY"))
            return real(*args, **kwargs)

        monkeypatch.setattr(coll_mod.Dataset, "read_file", staticmethod(spy))
        coll = DatasetCollection.from_files(
            three_tifs, gdal_env={"PYRAMIDS_TEST_KEY": "on"}
        )
        _ = coll.datasets
        assert seen and all(
            v == "on" for v in seen
        ), f"env not active for every open: {seen}"

    def test_path_a_no_env_leaves_option_unset(self, three_tifs, monkeypatch):
        """Path A: without a persisted env the sentinel option stays unset.

        Test scenario:
            from_files with no gdal_env opens under a nullcontext, so the
            sentinel is None at open time.
        """
        from pyramids.dataset import collection as coll_mod

        seen: list[str | None] = []
        real = coll_mod.Dataset.read_file

        def spy(*args, **kwargs):
            seen.append(gdal.GetConfigOption("PYRAMIDS_TEST_KEY"))
            return real(*args, **kwargs)

        monkeypatch.setattr(coll_mod.Dataset, "read_file", staticmethod(spy))
        coll = DatasetCollection.from_files(three_tifs)
        _ = coll.datasets
        assert seen and all(
            v is None for v in seen
        ), f"unexpected env without signer: {seen}"

    def test_path_b_read_time_step_installs_env(self, three_tifs, monkeypatch):
        """Path B: `_read_time_step` installs the env around the worker open.

        Test scenario:
            A spy on the module-level opener records the sentinel option; calling
            _read_time_step with an env must see it active.
        """
        from pyramids.dataset import collection as coll_mod

        captured: dict[str, str | None] = {}
        real_open = coll_mod.gdal_raster_open

        def spy(*args, **kwargs):
            captured["v"] = gdal.GetConfigOption("PYRAMIDS_TEST_KEY")
            return real_open(*args, **kwargs)

        monkeypatch.setattr(coll_mod, "gdal_raster_open", spy)
        coll_mod._read_time_step(three_tifs[0], {"PYRAMIDS_TEST_KEY": "on"})
        assert captured["v"] == "on", f"env not active during Path B open: {captured}"

    def test_path_b_read_time_step_reads_array(self, three_tifs):
        """Path B: `_read_time_step` returns a (1, R, C) array for a 1-band file.

        Test scenario:
            The reader still works (env defaults to None) and shapes correctly.
        """
        from pyramids.dataset.collection import _read_time_step

        arr = _read_time_step(three_tifs[0])
        assert (
            arr.shape[0] == 1
        ), f"single-band read should be (1, R, C), got {arr.shape}"

    def test_gdal_env_survives_pickle(self, three_tifs):
        """H4: the persisted env survives pickling (so it reaches dask workers).

        Test scenario:
            A round-trip through pickle preserves _gdal_env (the dask `data`
            graph ships it into each worker task).
        """
        import pickle

        coll = DatasetCollection.from_files(
            three_tifs, gdal_env={"AWS_REQUEST_PAYER": "requester"}
        )
        restored = pickle.loads(pickle.dumps(coll))
        assert restored._gdal_env == {
            "AWS_REQUEST_PAYER": "requester"
        }, f"env lost across pickle: {restored._gdal_env}"


class TestBboxAndMaxItems:
    """M6: bbox filter + max_items cap before href resolution."""

    def test_bbox_filters_items(self, stac_items):
        collection = DatasetCollection.from_stac(
            stac_items,
            asset="data",
            bbox=(0.0, 0.0, 0.5, 0.5),
        )
        # Every fixture item claims bbox [0,0,1,1] so they all intersect.
        assert collection.time_length == 3

    def test_bbox_excludes_non_intersecting(self, stac_items):
        # A valid lon/lat box that does not intersect the items' [0,0,1,1] bbox
        # (L1 now rejects projected / out-of-range boxes up front).
        with pytest.raises(ValueError, match="at least one path"):
            DatasetCollection.from_stac(
                stac_items,
                asset="data",
                bbox=(10.0, 10.0, 20.0, 20.0),
            )

    def test_max_items_caps(self, stac_items):
        collection = DatasetCollection.from_stac(
            stac_items,
            asset="data",
            max_items=2,
        )
        assert collection.time_length == 2


class TestAssetMissing:
    """Missing asset keys raise KeyError with available assets listed."""

    def test_unknown_asset_raises(self, stac_items):
        with pytest.raises(KeyError, match="not found"):
            DatasetCollection.from_stac(stac_items, asset="doesnotexist")


class TestAssetShapes:
    """``assets[key]`` can be a pystac.Asset (attribute) or a dict."""

    def test_asset_as_attribute_object(self, three_tifs):
        """Emulate pystac.Asset: object with an ``href`` attribute."""
        from types import SimpleNamespace

        items = [
            {"assets": {"data": SimpleNamespace(href=path)}} for path in three_tifs
        ]
        collection = DatasetCollection.from_stac(items, asset="data")
        assert collection.time_length == 3

    def test_asset_dict_without_href_raises(self):
        """An asset dict lacking ``href`` is a malformed STAC Item."""
        items = [{"assets": {"data": {"type": "image/tiff"}}}]
        with pytest.raises(KeyError, match="has no 'href'"):
            DatasetCollection.from_stac(items, asset="data")


class TestHorizontalBounds:
    """M1: ``_horizontal_bounds`` extracts (west, south, east, north) from 2D/3D bboxes."""

    def test_2d_bbox_returned_verbatim(self):
        """A 4-element bbox returns its four members as floats.

        Test scenario:
            ``[w, s, e, n]`` → ``(w, s, e, n)``.
        """
        assert _horizontal_bounds([1.0, 2.0, 3.0, 4.0]) == (1.0, 2.0, 3.0, 4.0)

    def test_3d_bbox_drops_elevation(self):
        """A 6-element 3D bbox drops the elevation members.

        Test scenario:
            ``[w, s, min_z, e, n, max_z]`` → ``(w, s, e, n)`` (indices 0,1,3,4).
        """
        result = _horizontal_bounds([1.0, 2.0, 100.0, 3.0, 4.0, 500.0])
        assert result == (
            1.0,
            2.0,
            3.0,
            4.0,
        ), f"3D bbox horizontal extent wrong: {result}"

    def test_integer_members_coerced_to_float(self):
        """Integer bbox members are returned as floats.

        Test scenario:
            An all-int bbox yields a float tuple.
        """
        result = _horizontal_bounds([0, 0, 10, 10])
        assert result == (0.0, 0.0, 10.0, 10.0)
        assert all(
            isinstance(v, float) for v in result
        ), f"members not floats: {result}"

    @pytest.mark.parametrize(
        "bad", [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7], []]
    )
    def test_invalid_length_raises(self, bad):
        """A bbox that is neither 4- nor 6-element raises ValueError.

        Args:
            bad: A bbox of an unsupported length.

        Test scenario:
            Lengths 0, 3, 5, 7 are rejected with a clear message.
        """
        with pytest.raises(ValueError, match="4 .2D. or 6 .3D. elements"):
            _horizontal_bounds(bad)


class TestItemIntersectsBbox3D:
    """M1: ``_item_intersects_bbox`` handles 3D item/query bboxes without crashing."""

    def test_3d_item_bbox_intersecting(self):
        """A 3D item bbox overlapping the 2D query box intersects.

        Test scenario:
            Item ``[0,0,minz,2,2,maxz]`` overlaps query ``(1,1,3,3)`` → True
            (previously raised ValueError on the 6-element unpack).
        """
        item = {"bbox": [0.0, 0.0, 100.0, 2.0, 2.0, 500.0]}
        assert _item_intersects_bbox(item, (1.0, 1.0, 3.0, 3.0)) is True

    def test_3d_item_bbox_disjoint(self):
        """A 3D item bbox outside the query box does not intersect.

        Test scenario:
            Item far to the east of the query box → False, no crash.
        """
        item = {"bbox": [10.0, 10.0, 0.0, 12.0, 12.0, 50.0]}
        assert _item_intersects_bbox(item, (0.0, 0.0, 1.0, 1.0)) is False

    def test_3d_query_bbox_against_2d_item(self):
        """A 3D query bbox compares only its horizontal extent.

        Test scenario:
            6-element query box overlapping a 2D item bbox → True.
        """
        item = {"bbox": [0.0, 0.0, 5.0, 5.0]}
        assert _item_intersects_bbox(item, [1.0, 1.0, 0.0, 3.0, 3.0, 999.0]) is True

    def test_item_without_bbox_is_permissive(self):
        """An item with no bbox is treated as intersecting.

        Test scenario:
            Missing ``bbox`` → True regardless of the query box.
        """
        assert _item_intersects_bbox({"id": "x"}, (0.0, 0.0, 1.0, 1.0)) is True


class TestValidateLonLatBbox:
    """L1: from_stac validates the query bbox is lon/lat (WGS84)."""

    @pytest.mark.parametrize(
        "bbox",
        [
            (-180.0, -90.0, 180.0, 90.0),
            (0.0, 0.0, 1.0, 1.0),
            [10.0, 20.0, 0.0, 30.0, 40.0, 500.0],
        ],
    )
    def test_valid_lonlat_passes(self, bbox):
        """A box within +/-180 / +/-90 (2D or 3D) is accepted.

        Args:
            bbox: A valid lon/lat box.

        Test scenario:
            The validator returns None (no raise) for in-range boxes.
        """
        assert _validate_lonlat_bbox(bbox) is None

    @pytest.mark.parametrize(
        "bbox",
        [
            (600000.0, 5000000.0, 601000.0, 5001000.0),
            (0.0, -91.0, 1.0, 1.0),
            (-181.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_projected_or_out_of_range_raises(self, bbox):
        """A projected / out-of-range box raises ValueError.

        Args:
            bbox: A box with coordinates outside the lon/lat domain.

        Test scenario:
            UTM metres and out-of-range lat/lon are rejected with a clear
            message rather than silently matching nothing.
        """
        with pytest.raises(ValueError, match="lon/lat"):
            _validate_lonlat_bbox(bbox)

    def test_from_stac_rejects_projected_bbox(self, stac_items):
        """from_stac surfaces the lon/lat validation to the caller.

        Test scenario:
            A UTM-metre bbox passed to from_stac raises before any read.
        """
        with pytest.raises(ValueError, match="lon/lat"):
            DatasetCollection.from_stac(
                stac_items,
                asset="data",
                bbox=(500000.0, 4000000.0, 501000.0, 4001000.0),
            )

    def test_from_stac_filters_3d_bbox_items(self, three_tifs):
        """``from_stac`` bbox-filters items carrying 3D bboxes end-to-end.

        Test scenario:
            Items with 6-element bboxes are filtered by a 2D query box without
            raising — the regression M1 fixes (the loader previously crashed
            unpacking a 3D item bbox).
        """
        items = [
            {
                "id": f"i{i}",
                "bbox": [0.0, 0.0, 10.0, 1.0, 1.0, 200.0],
                "assets": {"data": {"href": p}},
            }
            for i, p in enumerate(three_tifs)
        ]
        coll = DatasetCollection.from_stac(
            items, asset="data", bbox=(0.0, 0.0, 0.5, 0.5)
        )
        assert (
            coll.time_length == 3
        ), f"all 3 overlapping 3D-bbox items should pass, got {coll.time_length}"


@pytest.fixture
def multi_asset_items(tmp_path):
    """Two scenes, each with red/green/blue single-band assets on a shared grid.

    Each asset is a 3x4 EPSG:4326 raster (top-left (0, 3), cell 1) filled with a
    distinct constant (red=1, green=2, blue=3), so band order is verifiable.

    Returns:
        list[dict]: two raw STAC item dicts, each with three assets.
    """
    items = []
    values = {"red": 1.0, "green": 2.0, "blue": 3.0}
    for scene in range(2):
        assets = {}
        for name, val in values.items():
            ds = Dataset.create_from_array(
                np.full((3, 4), val, dtype=np.float32),
                top_left_corner=(0.0, 3.0),
                cell_size=1.0,
                epsg=4326,
            )
            p = str(tmp_path / f"scene{scene}_{name}.tif")
            ds.to_file(p)
            assets[name] = {"href": p}
        items.append(
            {"id": f"scene-{scene}", "bbox": [0.0, 0.0, 1.0, 1.0], "assets": assets}
        )
    return items


class TestFromStacMultiAsset:
    """PB-2: from_stac(asset=[...]) stacks assets band-wise per timestep."""

    def test_returns_multiband_timesteps(self, multi_asset_items):
        """A list of asset keys yields one multi-band Dataset per item.

        Test scenario:
            asset=["red","green","blue"] over two scenes -> 2 timesteps, each a
            3-band raster whose band names are the asset keys.
        """
        coll = DatasetCollection.from_stac(
            multi_asset_items, asset=["red", "green", "blue"]
        )
        assert coll.time_length == 2, f"expected 2 timesteps, got {coll.time_length}"
        first = coll.datasets[0]
        assert first.band_count == 3, f"expected 3 bands, got {first.band_count}"
        assert first.band_names == [
            "red",
            "green",
            "blue",
        ], f"band names: {first.band_names}"

    def test_band_order_preserved(self, multi_asset_items):
        """Bands carry each asset's values in the requested order.

        Test scenario:
            Band 1 == red(1), band 2 == green(2), band 3 == blue(3).
        """
        coll = DatasetCollection.from_stac(
            multi_asset_items, asset=["red", "green", "blue"]
        )
        arr = coll.datasets[0].read_array()
        assert arr.shape[0] == 3, f"expected 3 bands, got {arr.shape}"
        assert float(arr[0, 0, 0]) == pytest.approx(1.0), f"band1 should be red=1, got {arr[0, 0, 0]}"
        assert (
            float(arr[1, 0, 0]) == pytest.approx(2.0)
        ), f"band2 should be green=2, got {arr[1, 0, 0]}"
        assert float(arr[2, 0, 0]) == pytest.approx(3.0), f"band3 should be blue=3, got {arr[2, 0, 0]}"

    def test_band_order_follows_asset_sequence(self, multi_asset_items):
        """Reordering the asset list reorders the output bands.

        Test scenario:
            asset=["blue","red"] -> band1==blue(3), band2==red(1).
        """
        coll = DatasetCollection.from_stac(multi_asset_items, asset=["blue", "red"])
        first = coll.datasets[0]
        assert first.band_names == ["blue", "red"], f"band names: {first.band_names}"
        arr = first.read_array()
        assert (
            float(arr[0, 0, 0]) == pytest.approx(3.0)
            and float(arr[1, 0, 0]) == pytest.approx(1.0)
        ), f"order wrong: {arr[:, 0, 0]}"

    def test_single_asset_str_is_single_band(self, multi_asset_items):
        """A plain str keeps the single-asset (single-band) behaviour.

        Test scenario:
            asset="red" -> 2 single-band timesteps (back-compat with the
            pre-PB-2 contract).
        """
        coll = DatasetCollection.from_stac(multi_asset_items, asset="red")
        assert coll.time_length == 2, f"expected 2 timesteps, got {coll.time_length}"
        assert coll.datasets[0].band_count == 1, f"single asset should be 1 band"

    def test_missing_asset_raises(self, multi_asset_items):
        """A requested asset absent from an item raises StacAssetError.

        Test scenario:
            "nir" is not present on any scene -> StacAssetError (a KeyError).
        """
        with pytest.raises(StacAssetError, match="not found"):
            DatasetCollection.from_stac(multi_asset_items, asset=["red", "nir"])

    def test_skip_missing_drops_item(self, multi_asset_items):
        """skip_missing=True drops items lacking a requested asset.

        Test scenario:
            Scene 0 loses its "blue" asset; with skip_missing only scene 1
            survives the ["red","blue"] request.
        """
        del multi_asset_items[0]["assets"]["blue"]
        coll = DatasetCollection.from_stac(
            multi_asset_items, asset=["red", "blue"], skip_missing=True
        )
        assert (
            coll.time_length == 1
        ), f"expected 1 surviving item, got {coll.time_length}"

    def test_skip_missing_all_gone_raises(self, multi_asset_items):
        """When every item is skipped, a clear ValueError is raised.

        Test scenario:
            No item has "nir"; skip_missing drops them all -> ValueError.
        """
        with pytest.raises(ValueError, match="produced no items"):
            DatasetCollection.from_stac(
                multi_asset_items, asset=["red", "nir"], skip_missing=True
            )

    def test_align_true_resamples_mixed_resolution(self, tmp_path):
        """align=True (default) resamples a coarser asset onto the first's grid.

        Test scenario:
            A 10 m red asset and a 20 m green asset stack into one 2-band raster
            on red's grid without raising.
        """
        red = Dataset.create_from_array(
            np.full((4, 4), 1.0, dtype=np.float32),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        green = Dataset.create_from_array(
            np.full((2, 2), 2.0, dtype=np.float32),
            top_left_corner=(0.0, 4.0),
            cell_size=2.0,
            epsg=4326,
        )
        rp, gp = str(tmp_path / "r.tif"), str(tmp_path / "g.tif")
        red.to_file(rp)
        green.to_file(gp)
        items = [{"assets": {"red": {"href": rp}, "green": {"href": gp}}}]
        coll = DatasetCollection.from_stac(items, asset=["red", "green"], align=True)
        out = coll.datasets[0]
        assert out.band_count == 2, f"expected 2 bands, got {out.band_count}"
        assert out.read_array().shape[1:] == (
            4,
            4,
        ), f"should be on red's 4x4 grid, got {out.read_array().shape}"

    def test_align_false_mismatch_raises(self, tmp_path):
        """align=False raises AlignmentError on a grid mismatch.

        Test scenario:
            The 10 m / 20 m pair cannot stack without resampling.
        """
        red = Dataset.create_from_array(
            np.full((4, 4), 1.0, dtype=np.float32),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        green = Dataset.create_from_array(
            np.full((2, 2), 2.0, dtype=np.float32),
            top_left_corner=(0.0, 4.0),
            cell_size=2.0,
            epsg=4326,
        )
        rp, gp = str(tmp_path / "r.tif"), str(tmp_path / "g.tif")
        red.to_file(rp)
        green.to_file(gp)
        items = [{"assets": {"red": {"href": rp}, "green": {"href": gp}}}]
        with pytest.raises(AlignmentError):
            DatasetCollection.from_stac(items, asset=["red", "green"], align=False)


@pytest.fixture
def solar_day_items(tmp_path):
    """Three tiles: two on 2021-06-01 (same grid), one on 2021-06-05.

    Returns:
        list[dict]: raw STAC items with properties.datetime + a "data" asset.
    """
    grids = [
        ("2021-06-01T10:00:00Z", 10.0),
        ("2021-06-01T10:00:30Z", 11.0),
        ("2021-06-05T10:00:00Z", 12.0),
    ]
    items = []
    for i, (when, val) in enumerate(grids):
        ds = Dataset.create_from_array(
            np.full((4, 4), val, dtype="float32"),
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        p = str(tmp_path / f"t{i}.tif")
        ds.to_file(p)
        items.append(
            {
                "id": f"item-{i}",
                "bbox": [0.0, 0.0, 4.0, 4.0],
                "properties": {"datetime": when},
                "assets": {"data": {"href": p}},
            }
        )
    return items


class TestSolarDayHelper:
    """PC-1: the _solar_day label helper."""

    def test_utc_date(self):
        """A near-zero-longitude item keeps its UTC calendar date.

        Test scenario:
            A 10:00Z item at lon 0 -> that same date.
        """
        item = {
            "bbox": [-0.5, 0.0, 0.5, 1.0],
            "properties": {"datetime": "2021-06-01T10:00:00Z"},
        }
        assert _solar_day(item) == "2021-06-01", f"got {_solar_day(item)}"

    def test_longitude_shift_crosses_midnight(self):
        """A high-longitude late-UTC item shifts into the next solar day.

        Test scenario:
            23:00Z at lon ~150E shifts +10h -> next calendar day.
        """
        item = {
            "bbox": [149.0, 0.0, 151.0, 1.0],
            "properties": {"datetime": "2021-06-01T23:00:00Z"},
        }
        assert _solar_day(item) == "2021-06-02", f"got {_solar_day(item)}"


class TestFromStacSolarDay:
    """PC-1: groupby='solar_day' mosaics same-day items into one timestep."""

    def test_groups_same_day(self, solar_day_items):
        """Two same-solar-day tiles fuse; a third day is its own timestep.

        Test scenario:
            3 items over 2 solar days -> time_length 2.
        """
        coll = DatasetCollection.from_stac(
            solar_day_items, asset="data", groupby="solar_day"
        )
        assert (
            coll.time_length == 2
        ), f"expected 2 solar-day timesteps, got {coll.time_length}"

    def test_chronological_order(self, solar_day_items):
        """The per-day mosaics are stacked in chronological order.

        Test scenario:
            06-01 (first-valid mosaic of value 10) precedes 06-05 (value 12).
        """
        coll = DatasetCollection.from_stac(
            solar_day_items, asset="data", groupby="solar_day"
        )
        first = coll.datasets[0].read_array()
        last = coll.datasets[1].read_array()
        assert (
            float(first[0, 0]) == pytest.approx(10.0)
        ), f"first day should be first-valid 10, got {first[0, 0]}"
        assert float(last[0, 0]) == pytest.approx(12.0), f"second day should be 12, got {last[0, 0]}"

    def test_invalid_groupby_raises(self, solar_day_items):
        """An unsupported groupby value raises ValueError.

        Test scenario:
            groupby='month' is not supported.
        """
        with pytest.raises(ValueError, match="groupby must be"):
            DatasetCollection.from_stac(solar_day_items, asset="data", groupby="month")

    def test_groupby_multi_asset_raises(self, solar_day_items):
        """groupby with a multi-asset sequence is rejected.

        Test scenario:
            solar-day fusing is single-asset only.
        """
        with pytest.raises(ValueError, match="single asset"):
            DatasetCollection.from_stac(
                solar_day_items, asset=["data", "data"], groupby="solar_day"
            )


class TestAntimeridian:
    """L3: antimeridian-aware longitude overlap + centroid."""

    def test_lon_overlaps_within_wrapping_box(self):
        """A wrapping query box overlaps a point just east of the dateline.

        Test scenario:
            Query [170, -170] (crosses 180) overlaps item [175, 178].
        """
        assert _lon_overlaps(170.0, -170.0, 175.0, 178.0) is True

    def test_lon_overlaps_excludes_far_side(self):
        """A wrapping box does not overlap a box near lon 0.

        Test scenario:
            Query [170, -170] vs item [-10, 10] -> no overlap.
        """
        assert _lon_overlaps(170.0, -170.0, -10.0, 10.0) is False

    def test_lon_overlaps_two_wrapping_boxes(self):
        """Two wrapping boxes overlap across the dateline.

        Test scenario:
            [170, -170] and [160, -150] both wrap and share the seam.
        """
        assert _lon_overlaps(170.0, -170.0, 160.0, -150.0) is True

    def test_item_intersects_wrapping_box(self):
        """from_stac's filter matches an item under a wrapping query box.

        Test scenario:
            Item near lon 178 intersects the wrapping query [170,-10,-170,10].
        """
        item = {"bbox": [177.0, 0.0, 179.0, 5.0]}
        assert _item_intersects_bbox(item, (170.0, -10.0, -170.0, 10.0)) is True

    def test_item_excluded_by_wrapping_box(self):
        """An item near lon 0 is excluded by a wrapping query box.

        Test scenario:
            Item at lon 0 does not intersect the dateline-wrapping box.
        """
        item = {"bbox": [-5.0, 0.0, 5.0, 5.0]}
        assert _item_intersects_bbox(item, (170.0, -10.0, -170.0, 10.0)) is False

    def test_centroid_lon_of_wrapping_box(self):
        """The centroid of an antimeridian box lands near the dateline.

        Test scenario:
            [170, -170] -> centroid ~180 (not the wrong ~0 from a naive mean).
        """
        assert _item_centroid_lon({"bbox": [170.0, 0.0, -170.0, 5.0]}) == pytest.approx(180.0)

    def test_centroid_lon_normalised_past_dateline(self):
        """An asymmetric wrapping box normalises its centroid into [-180, 180].

        Test scenario:
            [170, -150] -> centroid -170 after the +360 shift + normalise.
        """
        assert _item_centroid_lon({"bbox": [170.0, 0.0, -150.0, 5.0]}) == -170.0

    def test_solar_day_uses_wrapping_centroid(self):
        """solar_day uses the antimeridian-aware centroid for its shift.

        Test scenario:
            A 23:00Z item wrapping the dateline (centroid ~180, +12h) rolls into
            the next solar day.
        """
        item = {
            "bbox": [170.0, 0.0, -170.0, 5.0],
            "properties": {"datetime": "2021-06-01T23:00:00Z"},
        }
        assert _solar_day(item) == "2021-06-02", f"got {_solar_day(item)}"


class TestFromStacMultiAssetUint16:
    """#362 regression on the PB-2 path: multi-asset align over uint16 bands."""

    def test_multi_asset_uint16_mixed_resolution(self, tmp_path):
        """from_stac(asset=[...], align=True) stacks uint16 10 m + 20 m bands.

        Test scenario:
            The Sentinel-2 case that triggered #362 through the multi-asset
            from_stac -> from_band_files(align=True) path: two uint16 assets at
            10 m and 20 m on one item build a 2-band uint16 cube without the
            -9999 template OverflowError.
        """
        b10 = Dataset.create_from_array(
            np.arange(16, dtype="uint16").reshape(4, 4),
            top_left_corner=(0.0, 40.0),
            cell_size=10.0,
            epsg=32630,
            no_data_value=0,
        )
        b20 = Dataset.create_from_array(
            (np.arange(4, dtype="uint16") + 1).reshape(2, 2),
            top_left_corner=(0.0, 40.0),
            cell_size=20.0,
            epsg=32630,
            no_data_value=0,
        )
        p10, p20 = str(tmp_path / "B04.tif"), str(tmp_path / "B05.tif")
        b10.to_file(p10)
        b20.to_file(p20)
        items = [{"assets": {"B04": {"href": p10}, "B05": {"href": p20}}}]
        coll = DatasetCollection.from_stac(items, asset=["B04", "B05"], align=True)
        out = coll.datasets[0]
        assert out.band_count == 2, f"expected 2 bands, got {out.band_count}"
        assert out.dtype[0] == "uint16", f"expected uint16, got {out.dtype}"
        assert out.band_names == ["B04", "B05"], f"band names: {out.band_names}"
        assert (out.rows, out.columns) == (
            4,
            4,
        ), f"grid should match the 10 m band: {(out.rows, out.columns)}"
