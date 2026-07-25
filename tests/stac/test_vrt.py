"""Unit tests for pyramids.stac._vrt.build_vrt_from_stac (PB-5).

Builds a lazy GDAL VRT mosaic over one STAC asset across items. Tests use local
GeoTIFF tiles (no network); the VRT references them and reads lazily.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base import _artifacts
from pyramids.dataset import Dataset
from pyramids.stac import _vrt
from pyramids.stac._vrt import (
    _check_dropped_sources,
    _dropped_sources,
    build_vrt_from_stac,
)
from tests._helpers import write_raster

pytestmark = pytest.mark.core


def vsimem_entries() -> set[str]:
    """Return the current `/vsimem/` directory listing as a set.

    Returns:
        set[str]: Entry names (not full paths) currently in `/vsimem/`.
    """
    return set(gdal.ReadDir("/vsimem/") or [])


@pytest.fixture
def adjacent_tiles(tmp_path):
    """Two 4x4 EPSG:4326 tiles abutting horizontally (union is 4x8).

    Tile A (value 10) covers columns 0..3; tile B (value 20) columns 4..7.

    Returns:
        list[dict]: two raw STAC items each exposing a "data" asset.
    """
    a = np.full((4, 4), 10.0, dtype="float32")
    b = np.full((4, 4), 20.0, dtype="float32")
    pa = write_raster(tmp_path / "a.tif", a, (0.0, 4.0))
    pb = write_raster(tmp_path / "b.tif", b, (4.0, 4.0))
    return [{"assets": {"data": {"href": pa}}}, {"assets": {"data": {"href": pb}}}]


@pytest.fixture
def mismatched_band_tiles(tmp_path):
    """One single-band tile plus one 3-band tile GDAL cannot mosaic with it.

    `gdalbuildvrt` refuses heterogeneous band counts: it warns and *skips* the
    3-band source instead of failing, which is the silent-drop case
    :func:`build_vrt_from_stac` guards against.

    Returns:
        list[dict]: two raw STAC items exposing a "data" asset each.
    """
    one = write_raster(tmp_path / "one.tif", np.full((4, 4), 1.0, "float32"), (0.0, 4.0))
    three = write_raster(
        tmp_path / "three.tif", np.full((3, 4, 4), 2.0, "float32"), (4.0, 4.0)
    )
    return [{"assets": {"data": {"href": one}}}, {"assets": {"data": {"href": three}}}]


class TestBuildVrtFromStac:
    """Tests for build_vrt_from_stac."""

    def test_returns_dataset_over_union(self, adjacent_tiles):
        """The VRT mosaics the two tiles into one Dataset over their union.

        Test scenario:
            Two 4x4 tiles abutting horizontally -> a 4x8 single-band Dataset.
        """
        ds = build_vrt_from_stac(adjacent_tiles, asset="data")
        assert isinstance(ds, Dataset), f"expected a Dataset, got {type(ds)}"
        arr = ds.read_array()
        assert arr.shape == (4, 8), f"expected union shape (4, 8), got {arr.shape}"

    def test_mosaic_values_lazy_read(self, adjacent_tiles):
        """Reading the VRT pulls each source's pixels into the right place.

        Test scenario:
            Left half reads as tile A (10), right half as tile B (20).
        """
        ds = build_vrt_from_stac(adjacent_tiles, asset="data")
        arr = ds.read_array()
        assert float(arr[0, 0]) == pytest.approx(10.0), (
            f"left half should be tile A=10, got {arr[0, 0]}"
        )
        assert float(arr[0, 7]) == pytest.approx(20.0), (
            f"right half should be tile B=20, got {arr[0, 7]}"
        )

    def test_separate_stacks_bands(self, tmp_path):
        """separate=True makes one band per source (same-grid sources).

        Test scenario:
            Two same-grid tiles -> a 2-band VRT.
        """
        a = np.full((3, 3), 1.0, dtype="float32")
        b = np.full((3, 3), 2.0, dtype="float32")
        pa = write_raster(tmp_path / "sa.tif", a, (0.0, 3.0))
        pb = write_raster(tmp_path / "sb.tif", b, (0.0, 3.0))
        items = [{"assets": {"d": {"href": pa}}}, {"assets": {"d": {"href": pb}}}]
        ds = build_vrt_from_stac(items, asset="d", separate=True)
        assert ds.band_count == 2, (
            f"separate=True should give 2 bands, got {ds.band_count}"
        )

    def test_signer_applied_to_each_source(self, adjacent_tiles):
        """signer.sign_href is applied to every source href before the build.

        Test scenario:
            An identity-recording signer sees both source hrefs; the build still
            succeeds on the local files.
        """

        class _RecordingSigner:
            def __init__(self):
                self.seen = []

            def sign_href(self, href):
                self.seen.append(href)
                return href

            def gdal_env(self):
                return {}

        signer = _RecordingSigner()
        build_vrt_from_stac(adjacent_tiles, asset="data", signer=signer)
        assert len(signer.seen) == 2, (
            f"sign_href should fire per source, got {signer.seen}"
        )

    def test_empty_items_raises(self):
        """No items raises a clear ValueError.

        Test scenario:
            An empty iterable cannot build a mosaic.
        """
        with pytest.raises(ValueError, match="no items"):
            build_vrt_from_stac([], asset="data")

    def test_missing_asset_raises(self, adjacent_tiles):
        """A missing asset surfaces StacAssetError from resolved_href.

        Test scenario:
            Requesting an absent asset key fails before the build.
        """
        from pyramids.base._errors import StacAssetError

        with pytest.raises(StacAssetError, match="not found"):
            build_vrt_from_stac(adjacent_tiles, asset="nope")


class TestDroppedSources:
    """Tests for the _dropped_sources helper."""

    def test_reports_source_absent_from_the_vrt(self):
        """A requested source the VRT does not reference is reported.

        Test scenario:
            Two requested, one retained -> the other is reported as dropped.
        """
        assert _dropped_sources(["a.tif", "b.tif"], ["a.tif"]) == ["b.tif"], (
            "the retained-source difference should name b.tif"
        )

    def test_reports_nothing_when_all_kept(self):
        """No source is reported when every requested path was retained.

        Test scenario:
            Extra entries in the retained list (e.g. the VRT itself) are ignored.
        """
        assert _dropped_sources(["a.tif"], ["a.tif", "/vsimem/x.vrt"]) == [], (
            "an extra retained entry must not make a kept source look dropped"
        )

    def test_preserves_requested_order(self):
        """Dropped sources come back in the order they were requested.

        Test scenario:
            Three requested, the middle one kept -> [first, last].
        """
        dropped = _dropped_sources(["a.tif", "b.tif", "c.tif"], ["b.tif"])
        assert dropped == ["a.tif", "c.tif"], f"order not preserved: {dropped}"

    def test_empty_retained_drops_everything(self):
        """An empty retained list means nothing survived the build.

        Test scenario:
            `GetFileList()` returning nothing -> every source is dropped.
        """
        assert _dropped_sources(["a.tif", "b.tif"], []) == ["a.tif", "b.tif"], (
            "an empty retained list should report every requested source"
        )


class TestCheckDroppedSources:
    """Tests for the _check_dropped_sources guard."""

    def test_no_drops_is_silent(self, recwarn):
        """An empty drop list neither raises nor warns.

        Args:
            recwarn: pytest fixture recording warnings raised in the block.

        Test scenario:
            The happy path must stay free of both errors and warnings.
        """
        _check_dropped_sources([], 2, "data", strict=True)
        assert len(recwarn) == 0, f"unexpected warnings: {[str(w) for w in recwarn]}"

    def test_strict_raises_with_counts_and_asset(self):
        """strict=True raises a RuntimeError naming the counts and the asset.

        Test scenario:
            1 of 3 sources dropped for asset "B04".
        """
        with pytest.raises(RuntimeError) as exc:
            _check_dropped_sources(["b.tif"], 3, "B04", strict=True)
        message = str(exc.value)
        assert "skipped 1 of 3" in message, f"counts missing from message: {message}"
        assert "'B04'" in message, f"asset key missing from message: {message}"
        assert "strict=False" in message, f"no escape hatch hinted: {message}"

    def test_strict_false_warns_instead(self):
        """strict=False downgrades the failure to a UserWarning.

        Test scenario:
            The same drop that raises under strict=True only warns here.
        """
        with pytest.warns(UserWarning, match="skipped 1 of 3"):
            _check_dropped_sources(["b.tif"], 3, "B04", strict=False)

    def test_long_drop_list_is_truncated(self):
        """More than five dropped sources are summarised, not dumped in full.

        Test scenario:
            Eight dropped paths -> five listed plus a "+3 more" suffix.
        """
        dropped = [f"s{i}.tif" for i in range(8)]
        with pytest.raises(RuntimeError) as exc:
            _check_dropped_sources(dropped, 10, "data", strict=True)
        message = str(exc.value)
        assert "(+3 more)" in message, f"long list not truncated: {message}"
        assert "s7.tif" not in message, f"truncated entry still listed: {message}"


class TestBuildVrtSourceCompleteness:
    """Tests for the strict source-completeness guard (ARC-79)."""

    def test_band_count_mismatch_raises(self, mismatched_band_tiles):
        """A source GDAL skips for a band-count mismatch fails the build.

        Test scenario:
            A 1-band and a 3-band tile: GDAL warns and drops the 3-band source,
            so the mosaic would silently cover half the requested footprint.
        """
        with pytest.raises(RuntimeError, match="skipped 1 of 2"):
            build_vrt_from_stac(mismatched_band_tiles, asset="data")

    def test_unreadable_source_raises(self, adjacent_tiles, tmp_path):
        """An unreadable href (404 / expired URL) fails the build.

        Test scenario:
            One good tile plus a path that does not exist.
        """
        items = [
            adjacent_tiles[0],
            {"assets": {"data": {"href": str(tmp_path / "gone.tif")}}},
        ]
        with pytest.raises(RuntimeError, match="skipped 1 of 2"):
            build_vrt_from_stac(items, asset="data")

    def test_error_names_the_dropped_source(self, mismatched_band_tiles):
        """The error lists the href that was dropped, not just a count.

        Test scenario:
            The 3-band tile's path appears in the message so it is actionable.
        """
        dropped_href = mismatched_band_tiles[1]["assets"]["data"]["href"]
        with pytest.raises(RuntimeError) as exc:
            build_vrt_from_stac(mismatched_band_tiles, asset="data")
        assert dropped_href in str(exc.value), (
            f"dropped href {dropped_href!r} missing from: {exc.value}"
        )

    def test_strict_false_warns_and_returns_partial(self, mismatched_band_tiles):
        """strict=False returns the partial mosaic with a warning.

        Test scenario:
            The same drop warns, and the returned Dataset covers only the tile
            GDAL kept (4x4 rather than the requested 4x8 union).
        """
        with pytest.warns(UserWarning, match="skipped 1 of 2"):
            ds = build_vrt_from_stac(mismatched_band_tiles, asset="data", strict=False)
        arr = ds.read_array()
        assert arr.shape == (4, 4), f"expected the kept tile only, got {arr.shape}"

    def test_complete_mosaic_does_not_raise(self, adjacent_tiles):
        """A build where GDAL keeps every source is unaffected by the guard.

        Test scenario:
            Two compatible tiles -> the full 4x8 union, no error, no warning.
        """
        ds = build_vrt_from_stac(adjacent_tiles, asset="data", strict=True)
        assert ds.read_array().shape == (4, 8), "the complete mosaic must survive"

    def test_all_sources_unreadable_raises_build_error(self, tmp_path):
        """When nothing is usable, gdal.BuildVRT returns None and that is raised.

        Test scenario:
            Every href missing -> the pre-existing "returned None" guard fires
            rather than the drop guard.
        """
        items = [{"assets": {"data": {"href": str(tmp_path / "nope.tif")}}}]
        with pytest.raises(RuntimeError, match="returned None"):
            build_vrt_from_stac(items, asset="data")


class TestBuildVrtArtifactCleanup:
    """Tests for the /vsimem VRT lifetime (ARC-10)."""

    def test_vsimem_registered_on_success(self, adjacent_tiles):
        """A successful build tracks its VRT for the process-exit sweep.

        Test scenario:
            The new `/vsimem` entry is also present in the artefact registry.
        """
        before = vsimem_entries()
        build_vrt_from_stac(adjacent_tiles, asset="data")
        created = vsimem_entries() - before
        assert len(created) == 1, f"expected exactly one new /vsimem entry: {created}"
        tracked = {path.rsplit("/", 1)[-1] for path in _artifacts._VSIMEM_PATHS}
        assert created <= tracked, f"{created} not tracked for cleanup"

    def test_vsimem_removed_when_open_fails(self, adjacent_tiles, monkeypatch):
        """A failing `Dataset.read_file` unlinks the VRT instead of orphaning it.

        Args:
            adjacent_tiles: Fixture providing two mosaickable items.
            monkeypatch: pytest fixture used to break the VRT open.

        Test scenario:
            The build succeeds, the open raises, and `/vsimem` is left clean.
        """

        class _BoomDataset:
            @staticmethod
            def read_file(*args, **kwargs):
                raise RuntimeError("simulated open failure")

        monkeypatch.setattr(_vrt, "Dataset", _BoomDataset)
        before = vsimem_entries()
        with pytest.raises(RuntimeError, match="simulated open failure"):
            build_vrt_from_stac(adjacent_tiles, asset="data")
        assert vsimem_entries() - before == set(), (
            "the VRT must be unlinked when the open fails"
        )

    def test_vsimem_removed_when_strict_check_fails(self, mismatched_band_tiles):
        """A strict drop failure also unlinks the VRT it had already written.

        Test scenario:
            The guard raises after `BuildVRT` wrote `/vsimem`; nothing is left.
        """
        before = vsimem_entries()
        with pytest.raises(RuntimeError, match="skipped"):
            build_vrt_from_stac(mismatched_band_tiles, asset="data")
        assert vsimem_entries() - before == set(), (
            "the VRT must be unlinked when the completeness guard fails"
        )
