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
