"""Tests for :mod:`pyramids.basemap.features` (Natural Earth vector accessor).

Network access is avoided except for a single ``slow``-marked end-to-end test: the
download is monkeypatched and the read path is exercised against a locally-built
shapefile zip placed in a temporary cache directory.
"""

from __future__ import annotations

import urllib.error
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from pyramids.basemap import features
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def cache_dir(tmp_path, monkeypatch):
    """Redirect the Natural Earth cache to a temp directory via PYRAMIDS_CACHE_DIR.

    Args:
        tmp_path: pytest temp directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Path: the ``naturalearth`` cache directory features.py will use.
    """
    monkeypatch.setenv("PYRAMIDS_CACHE_DIR", str(tmp_path))
    return tmp_path / "naturalearth"


def _make_coastline_zip(destination: Path, n_features: int = 2) -> int:
    """Write a tiny ``ne_110m_coastline.zip`` (real shapefile) into ``destination``.

    Args:
        destination: Cache directory to place the zip in (created if missing).
        n_features: Number of LineString features to include.

    Returns:
        The number of features written (for assertion convenience).
    """
    destination.mkdir(parents=True, exist_ok=True)
    shp_dir = destination / "_build"
    shp_dir.mkdir(exist_ok=True)
    lines = [LineString([(i, 0), (i + 1, 1)]) for i in range(n_features)]
    gdf = gpd.GeoDataFrame({"id": range(n_features)}, geometry=lines, crs="EPSG:4326")
    shp_path = shp_dir / "ne_110m_coastline.shp"
    gdf.to_file(shp_path)
    zip_path = destination / "ne_110m_coastline.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for sidecar in shp_dir.glob("ne_110m_coastline.*"):
            archive.write(sidecar, sidecar.name)
    return n_features


def _make_coastline_zip_with_stems(destination: Path, stems: dict[str, int]) -> None:
    """Write ``ne_110m_coastline.zip`` containing one shapefile per given stem.

    Args:
        destination: Cache directory to place the zip in (created if missing).
        stems: Mapping of shapefile stem (without extension) to the number of
            LineString features that shapefile should contain.
    """
    destination.mkdir(parents=True, exist_ok=True)
    shp_dir = destination / "_build_stems"
    shp_dir.mkdir(exist_ok=True)
    for stem, n_features in stems.items():
        lines = [LineString([(i, 0), (i + 1, 1)]) for i in range(n_features)]
        gdf = gpd.GeoDataFrame(
            {"id": range(n_features)}, geometry=lines, crs="EPSG:4326"
        )
        gdf.to_file(shp_dir / f"{stem}.shp")
    zip_path = destination / "ne_110m_coastline.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for sidecar in shp_dir.iterdir():
            archive.write(sidecar, sidecar.name)


def _make_sidecar_only_zip(destination: Path) -> None:
    """Write ``ne_110m_coastline.zip`` with shapefile sidecars but no ``.shp``."""
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "ne_110m_coastline.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ne_110m_coastline.dbf", b"not a real dbf")
        archive.writestr("ne_110m_coastline.prj", b"not a real prj")


class TestAvailableLayers:
    """Tests for :func:`pyramids.basemap.features.available_layers`."""

    def test_contains_expected_layers(self):
        """available_layers returns the six supported Natural Earth layers.

        Test scenario:
            The list contains every documented layer and is sorted.
        """
        result = features.available_layers()
        expected = ["borders", "coastline", "lakes", "land", "ocean", "rivers"]
        assert result == expected, f"Unexpected layers: {result}"


class TestAvailableResolutions:
    """Tests for :func:`pyramids.basemap.features.available_resolutions`."""

    def test_returns_three_scales_coarsest_first(self):
        """available_resolutions returns 110m/50m/10m, coarsest first.

        Test scenario:
            The ordering matters for documentation; assert exact sequence.
        """
        assert features.available_resolutions() == ["110m", "50m", "10m"]


class TestDatasetStem:
    """Tests for :func:`pyramids.basemap.features._dataset_stem`."""

    @pytest.mark.parametrize(
        "layer, resolution, expected",
        [
            ("coastline", "110m", "ne_110m_coastline"),
            ("land", "50m", "ne_50m_land"),
            ("rivers", "10m", "ne_10m_rivers_lake_centerlines"),
            ("borders", "110m", "ne_110m_admin_0_boundary_lines_land"),
        ],
    )
    def test_stem_format(self, layer, resolution, expected):
        """_dataset_stem builds ``ne_{resolution}_{suffix}``.

        Args:
            layer: Natural Earth layer name.
            resolution: Resolution string.
            expected: Expected dataset stem.

        Test scenario:
            Each layer maps to its documented Natural Earth file stem.
        """
        assert features._dataset_stem(layer, resolution) == expected


class TestDownloadUrl:
    """Tests for :func:`pyramids.basemap.features._download_url`."""

    @pytest.mark.parametrize("resolution", ["110m", "50m", "10m"])
    @pytest.mark.parametrize(
        "layer, category",
        [
            ("coastline", "physical"),
            ("land", "physical"),
            ("ocean", "physical"),
            ("rivers", "physical"),
            ("lakes", "physical"),
            ("borders", "cultural"),
        ],
    )
    def test_url_uses_correct_category(self, layer, category, resolution):
        """_download_url routes physical vs cultural layers to the right sub-path.

        Args:
            layer: Natural Earth layer name.
            category: Expected CDN category ("physical" or "cultural").
            resolution: Resolution string.

        Test scenario:
            The URL embeds the resolution, category and dataset stem and ends in .zip.
        """
        url = features._download_url(layer, resolution)
        stem = features._dataset_stem(layer, resolution)
        assert url == f"{features._BASE_URL}/{resolution}/{category}/{stem}.zip", url


class TestCacheDir:
    """Tests for :func:`pyramids.basemap.features._cache_dir`."""

    def test_honors_env_override(self, tmp_path, monkeypatch):
        """_cache_dir uses PYRAMIDS_CACHE_DIR and creates the naturalearth subdir.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With the env var set, the returned path is
            ``<override>/naturalearth`` and exists on disk.
        """
        monkeypatch.setenv("PYRAMIDS_CACHE_DIR", str(tmp_path))
        result = features._cache_dir()
        assert result == tmp_path / "naturalearth", f"Wrong cache dir: {result}"
        assert result.is_dir(), "cache dir should be created"

    def test_defaults_to_home(self, monkeypatch):
        """_cache_dir falls back to ~/.pyramids/naturalearth without the env var.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With PYRAMIDS_CACHE_DIR unset and a redirected home, the cache lives under
            the home directory.
        """
        monkeypatch.delenv("PYRAMIDS_CACHE_DIR", raising=False)
        fake_home = Path(features.__file__).parent  # any existing dir works
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        result = features._cache_dir()
        assert result == fake_home / ".pyramids" / "naturalearth", f"{result}"


class TestEnsureCached:
    """Tests for :func:`pyramids.basemap.features._ensure_cached`."""

    def test_downloads_when_absent(self, cache_dir, monkeypatch):
        """_ensure_cached downloads the archive when it is not already cached.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With no cached file, _download is invoked once and the returned path is
            the expected cached archive.
        """
        calls = []

        def fake_download(url, destination):
            calls.append((url, destination))
            destination.write_bytes(b"PK\x03\x04")

        monkeypatch.setattr(features, "_download", fake_download)
        result = features._ensure_cached("coastline", "110m")
        assert result == cache_dir / "ne_110m_coastline.zip", f"Wrong path: {result}"
        assert len(calls) == 1, f"Expected one download, got {len(calls)}"

    def test_skips_download_when_present(self, cache_dir, monkeypatch):
        """_ensure_cached returns the cached archive without downloading.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With the archive already on disk, _download is never called.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "ne_110m_coastline.zip"
        archive.write_bytes(b"PK\x03\x04")

        def fail_download(url, destination):
            raise AssertionError("download must not be called on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        result = features._ensure_cached("coastline", "110m")
        assert result == archive, f"Wrong path: {result}"


class TestDownload:
    """Tests for :func:`pyramids.basemap.features._download`."""

    def test_success_writes_file_atomically(self, tmp_path, monkeypatch):
        """_download streams the response body to the destination path.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With urlopen returning a fake byte stream, _download writes exactly those
            bytes to the destination (no real network).
        """
        import io

        payload = b"PK\x03\x04 fake zip bytes"

        def fake_urlopen(request, *args, **kwargs):
            return io.BytesIO(payload)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        destination = tmp_path / "out.zip"
        features._download("https://example.com/x.zip", destination)
        assert destination.read_bytes() == payload, "downloaded bytes mismatch"

    def test_non_http_url_rejected(self, tmp_path):
        """_download refuses non-HTTP(S) URLs (guards against file:// reads).

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A file:// URL raises ValueError before any network/file access.
        """
        with pytest.raises(ValueError, match="non-HTTP"):
            features._download("file:///etc/passwd", tmp_path / "x.zip")

    def test_network_error_wrapped_as_oserror(self, tmp_path, monkeypatch):
        """_download wraps a urllib error as OSError with the URL and cache hint.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            When urlopen raises URLError, _download raises OSError mentioning the
            source URL, and chains the original error.
        """

        def boom(request, *args, **kwargs):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(OSError, match="failed to download basemap data") as exc:
            features._download("https://example.com/x.zip", tmp_path / "x.zip")
        assert "x.zip" in str(exc.value), f"URL missing from message: {exc.value}"
        assert exc.value.__cause__ is not None, "original error should be chained"

    def test_failed_download_leaves_no_temp_file(self, tmp_path, monkeypatch):
        """_download cleans up its temp file when the download fails.

        Args:
            tmp_path: pytest temp directory (the download destination's parent).
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With urlopen raising URLError, _download raises OSError and leaves no
            files behind in the destination directory (no stray temp .zip).
        """

        def boom(request, *args, **kwargs):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(OSError):
            features._download("https://example.com/x.zip", tmp_path / "x.zip")
        leftovers = list(tmp_path.iterdir())
        assert leftovers == [], f"temp file(s) left behind after failure: {leftovers}"


class TestNaturalEarth:
    """Tests for :func:`pyramids.basemap.features.natural_earth`."""

    def test_unknown_layer_raises(self):
        """natural_earth rejects an unknown layer with the valid-options list.

        Test scenario:
            Passing a non-existent layer raises ValueError naming the supported
            layers.
        """
        with pytest.raises(ValueError, match="unknown Natural Earth layer") as exc:
            features.natural_earth("countries")
        assert "coastline" in str(exc.value), f"Options missing: {exc.value}"

    def test_unknown_resolution_raises(self):
        """natural_earth rejects an unknown resolution with the valid-options list.

        Test scenario:
            Passing a non-existent resolution raises ValueError naming the supported
            resolutions.
        """
        with pytest.raises(ValueError, match="unknown Natural Earth resolution") as exc:
            features.natural_earth("coastline", resolution="1m")
        assert "110m" in str(exc.value), f"Options missing: {exc.value}"

    def test_cache_hit_returns_feature_collection(self, cache_dir, monkeypatch):
        """natural_earth reads a cached archive without downloading.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With a locally-built coastline zip in the cache, natural_earth returns a
            FeatureCollection with the expected feature count and never downloads.
        """
        n_features = _make_coastline_zip(cache_dir, n_features=3)

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        result = features.natural_earth("coastline", "110m")
        assert isinstance(result, FeatureCollection), f"Got {type(result)}"
        assert len(result) == n_features, f"Expected {n_features}, got {len(result)}"

    def test_reads_shapefile_with_unexpected_stem(self, cache_dir, monkeypatch):
        """natural_earth reads the archive's .shp even when its name is unexpected.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            The cached zip contains a shapefile named ``coastline_renamed.shp`` (not the
            conventional ``ne_110m_coastline.shp``). natural_earth still selects it by
            listing the archive members, returns the right feature count, and does not
            download.
        """
        _make_coastline_zip_with_stems(cache_dir, {"coastline_renamed": 4})

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        result = features.natural_earth("coastline", "110m")
        assert isinstance(result, FeatureCollection), f"Got {type(result)}"
        assert len(result) == 4, f"Expected 4 features, got {len(result)}"

    def test_prefers_conventional_stem_over_others(self, cache_dir, monkeypatch):
        """natural_earth prefers ``ne_{res}_{name}.shp`` when several .shp exist.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            The archive holds two shapefiles — the conventional ``ne_110m_coastline.shp``
            (2 features) and an ``extra.shp`` (5 features). natural_earth picks the
            conventional one, identified by its distinct feature count.
        """
        _make_coastline_zip_with_stems(cache_dir, {"ne_110m_coastline": 2, "extra": 5})

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        result = features.natural_earth("coastline", "110m")
        assert (
            len(result) == 2
        ), f"Expected the conventional 2-feature shp, got {len(result)}"

    def test_archive_without_shapefile_raises(self, cache_dir, monkeypatch):
        """natural_earth raises when the cached archive contains no .shp member.

        Args:
            cache_dir: Redirected cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            A zip with only sidecar files (no ``.shp``) raises FileNotFoundError naming
            the ``*.shp`` pattern it looked for.
        """
        _make_sidecar_only_zip(cache_dir)

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        with pytest.raises(FileNotFoundError, match=r"\*\.shp"):
            features.natural_earth("coastline", "110m")

    @pytest.mark.slow
    def test_real_download_coastline(self, cache_dir):
        """End-to-end fetch of the 110m coastline from the Natural Earth CDN.

        Test scenario:
            Network-dependent; skipped offline. Downloads the real 110m coastline and
            asserts a non-empty FeatureCollection is returned.
        """
        try:
            result = features.natural_earth("coastline", "110m")
        except OSError as exc:
            pytest.skip(f"Natural Earth CDN unreachable: {exc}")
        assert isinstance(result, FeatureCollection), f"Got {type(result)}"
        assert len(result) > 0, "real coastline layer should have features"


@pytest.fixture(scope="function")
def relief_cache_dir(tmp_path, monkeypatch):
    """Redirect the relief cache to a temp directory via PYRAMIDS_CACHE_DIR.

    Args:
        tmp_path: pytest temp directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Path: the ``relief`` cache directory features.py will use.
    """
    monkeypatch.setenv("PYRAMIDS_CACHE_DIR", str(tmp_path))
    return tmp_path / "relief"


def _make_relief_tif(path: Path, rows: int = 18, cols: int = 36) -> None:
    """Write a tiny 3-band RGB EPSG:4326 global GeoTIFF (synthetic relief) to ``path``.

    Args:
        path: Destination file (its parent is created if missing).
        rows: Raster height; with ``cols`` it spans the globe at ``180/rows`` degrees.
        cols: Raster width.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(3, rows, cols), dtype="uint8")
    Dataset.create_from_array(
        arr,
        top_left_corner=(-180.0, 90.0),
        cell_size=180.0 / rows,
        epsg=4326,
        driver_type="GTiff",
        path=str(path),
    )


class TestAvailableReliefResolutions:
    """Tests for :func:`pyramids.basemap.features.available_relief_resolutions`."""

    def test_returns_low_then_medium(self):
        """available_relief_resolutions returns ``["low", "medium"]`` (coarsest first).

        Test scenario:
            The ordering is documented; assert the exact sequence.
        """
        assert features.available_relief_resolutions() == ["low", "medium"]


class TestReliefUrl:
    """Tests for :func:`pyramids.basemap.features._relief_url`."""

    @pytest.mark.parametrize("resolution", ["low", "medium"])
    def test_url_points_at_release_asset(self, resolution):
        """_relief_url builds the release-asset URL for each resolution.

        Args:
            resolution: Relief resolution key.

        Test scenario:
            The URL is ``{_RELIEF_BASE_URL}/{product-filename}`` and ends in ``.tif``.
        """
        url = features._relief_url(resolution)
        product = features._RELIEF_PRODUCTS[resolution]
        assert url == f"{features._RELIEF_BASE_URL}/{product}", url
        assert url.endswith(".tif"), url


class TestRelief:
    """Tests for :func:`pyramids.basemap.features.relief`."""

    def test_unknown_resolution_raises(self):
        """relief rejects an unknown resolution with the valid-options list.

        Test scenario:
            Passing a non-existent resolution raises ValueError naming the supported
            ``low``/``medium`` values.
        """
        with pytest.raises(ValueError, match="unknown relief resolution") as exc:
            features.relief("high")
        assert "low" in str(exc.value), f"Options missing: {exc.value}"

    def test_cache_dir_uses_relief_subdir(self, tmp_path, monkeypatch):
        """_cache_dir('relief') resolves under the cache root, separate from vectors.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With PYRAMIDS_CACHE_DIR set, the relief cache is ``<root>/relief`` and the
            Natural Earth cache stays ``<root>/naturalearth`` — they don't collide.
        """
        monkeypatch.setenv("PYRAMIDS_CACHE_DIR", str(tmp_path))
        assert features._cache_dir("relief") == tmp_path / "relief"
        assert features._cache_dir() == tmp_path / "naturalearth"

    def test_cache_hit_returns_rgb_dataset(self, relief_cache_dir, monkeypatch):
        """relief reads a cached raster as a 3-band EPSG:4326 Dataset, no download.

        Args:
            relief_cache_dir: Redirected relief cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With a synthetic relief GeoTIFF already in the cache, relief returns a
            3-band (RGB), EPSG:4326 Dataset and never downloads.
        """
        _make_relief_tif(relief_cache_dir / features._RELIEF_PRODUCTS["low"])

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        result = features.relief("low")
        assert isinstance(result, Dataset), f"Got {type(result)}"
        assert result.band_count == 3, f"Expected 3 bands, got {result.band_count}"
        assert result.epsg == 4326, f"Expected EPSG:4326, got {result.epsg}"

    def test_downloads_when_absent(self, relief_cache_dir, monkeypatch):
        """relief downloads the raster once when it is not already cached.

        Args:
            relief_cache_dir: Redirected relief cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            With no cached file, _download is invoked exactly once (writing a synthetic
            raster) and relief returns the resulting Dataset.
        """
        calls = []

        def fake_download(url, destination):
            calls.append(url)
            _make_relief_tif(Path(destination))

        monkeypatch.setattr(features, "_download", fake_download)
        result = features.relief("low")
        assert len(calls) == 1, f"Expected one download, got {len(calls)}"
        assert calls[0] == features._relief_url("low"), f"Wrong URL: {calls[0]}"
        assert isinstance(result, Dataset), f"Got {type(result)}"

    def test_reprojectable_via_to_crs(self, relief_cache_dir, monkeypatch):
        """The relief Dataset reprojects via to_crs (EPSG:3857 here).

        Args:
            relief_cache_dir: Redirected relief cache directory fixture.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            relief returns an EPSG:4326 raster that ``to_crs`` can reproject to another
            CRS without error. (``to_crs`` takes an EPSG code; orthographic/globe
            rendering is handled downstream by the visualization layer.)
        """
        _make_relief_tif(relief_cache_dir / features._RELIEF_PRODUCTS["low"])

        def fail_download(url, destination):
            raise AssertionError("download must not run on a cache hit")

        monkeypatch.setattr(features, "_download", fail_download)
        reprojected = features.relief("low").to_crs(to_epsg=3857)
        assert isinstance(reprojected, Dataset), f"Got {type(reprojected)}"
        assert reprojected.epsg == 3857, f"Expected EPSG:3857, got {reprojected.epsg}"

    @pytest.mark.slow
    def test_real_download_low(self, relief_cache_dir):
        """End-to-end fetch of the low-res relief raster from the release assets.

        Test scenario:
            Network-dependent; skipped offline. Downloads the real
            ``ne_hypso_rgb_720x360.tif`` from the ``basemap-data-v1`` release and
            asserts a 3-band EPSG:4326 Dataset of the expected size is returned.
        """
        try:
            result = features.relief("low")
        except OSError as exc:
            pytest.skip(f"relief release asset unreachable: {exc}")
        assert isinstance(result, Dataset), f"Got {type(result)}"
        assert result.band_count == 3, f"Expected 3 bands, got {result.band_count}"
        assert result.epsg == 4326, f"Expected EPSG:4326, got {result.epsg}"
        assert (result.columns, result.rows) == (
            720,
            360,
        ), f"Expected 720x360, got {result.columns}x{result.rows}"
