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

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset._stac import _horizontal_bounds, _item_intersects_bbox

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
        with pytest.raises(ValueError, match="at least one path"):
            DatasetCollection.from_stac(
                stac_items,
                asset="data",
                bbox=(100.0, 100.0, 200.0, 200.0),
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
        assert result == (1.0, 2.0, 3.0, 4.0), f"3D bbox horizontal extent wrong: {result}"

    def test_integer_members_coerced_to_float(self):
        """Integer bbox members are returned as floats.

        Test scenario:
            An all-int bbox yields a float tuple.
        """
        result = _horizontal_bounds([0, 0, 10, 10])
        assert result == (0.0, 0.0, 10.0, 10.0)
        assert all(isinstance(v, float) for v in result), f"members not floats: {result}"

    @pytest.mark.parametrize("bad", [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7], []])
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

    def test_from_stac_filters_3d_bbox_items(self, three_tifs):
        """``from_stac`` bbox-filters items carrying 3D bboxes end-to-end.

        Test scenario:
            Items with 6-element bboxes are filtered by a 2D query box without
            raising — the regression M1 fixes (the loader previously crashed
            unpacking a 3D item bbox).
        """
        items = [
            {"id": f"i{i}", "bbox": [0.0, 0.0, 10.0, 1.0, 1.0, 200.0], "assets": {"data": {"href": p}}}
            for i, p in enumerate(three_tifs)
        ]
        coll = DatasetCollection.from_stac(items, asset="data", bbox=(0.0, 0.0, 0.5, 0.5))
        assert coll.time_length == 3, f"all 3 overlapping 3D-bbox items should pass, got {coll.time_length}"
