"""Unit tests for pyramids.stac._vrt.build_vrt_from_stac (PB-5).

Builds a lazy GDAL VRT mosaic over one STAC asset across items. Tests use local
GeoTIFF tiles (no network); the VRT references them and reads lazily.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base import _artifacts
from pyramids.dataset import Dataset
from pyramids.stac import _vrt
from pyramids.stac._vrt import (
    _check_dropped_sources,
    _dropped_sources,
    _embed_source_options,
    _source_config,
    _warn_unembeddable_credentials,
    build_vrt_from_stac,
    redact,
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
    one = write_raster(
        tmp_path / "one.tif", np.full((4, 4), 1.0, "float32"), (0.0, 4.0)
    )
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
        """A requested source the VRT does not reference is reported by href.

        Test scenario:
            Two requested, one retained -> the other is reported as dropped.
        """
        pairs = [("a.tif", "a.tif"), ("b.tif", "b.tif")]
        assert _dropped_sources(pairs, ["a.tif"]) == ["b.tif"], (
            "the retained-source difference should name b.tif"
        )

    def test_reports_nothing_when_all_kept(self):
        """No source is reported when every requested path was retained.

        Test scenario:
            Extra entries in the retained list (e.g. the VRT itself) are ignored.
        """
        kept = _dropped_sources([("a.tif", "a.tif")], ["a.tif", "/vsimem/x.vrt"])
        assert kept == [], (
            "an extra retained entry must not make a kept source look dropped"
        )

    def test_preserves_requested_order(self):
        """Dropped sources come back in the order they were requested.

        Test scenario:
            Three requested, the middle one kept -> [first, last].
        """
        pairs = [(n, n) for n in ("a.tif", "b.tif", "c.tif")]
        dropped = _dropped_sources(pairs, ["b.tif"])
        assert dropped == ["a.tif", "c.tif"], f"order not preserved: {dropped}"

    def test_empty_retained_drops_everything(self):
        """An empty retained list means nothing survived the build.

        Test scenario:
            `GetFileList()` returning nothing -> every source is dropped.
        """
        pairs = [("a.tif", "a.tif"), ("b.tif", "b.tif")]
        assert _dropped_sources(pairs, []) == ["a.tif", "b.tif"], (
            "an empty retained list should report every requested source"
        )


class TestRedact:
    """Tests for the redact helper that keeps credentials out of messages."""

    def test_query_string_is_replaced(self):
        """A SAS-signed href keeps its identity but loses the token.

        Test scenario:
            The path stays readable so the message is still actionable.
        """
        assert redact("https://h/B04.tif?sv=2021&sig=SECRET") == (
            "https://h/B04.tif?<redacted>"
        ), "the query string should be replaced wholesale"

    def test_unsigned_href_passes_through(self):
        """An href with no query string is unchanged.

        Test scenario:
            Nothing to hide -> nothing altered.
        """
        assert redact("https://h/B04.tif") == "https://h/B04.tif"

    def test_local_windows_path_passes_through(self):
        """A drive-letter path is not mistaken for a query string.

        Test scenario:
            `C:/data/a.tif` has no `?` and must survive intact.
        """
        assert redact("C:/data/a.tif") == "C:/data/a.tif"

    def test_embedded_source_path_is_redacted(self):
        """The `/vsicurl?` form this module builds is redacted to its prefix.

        Test scenario:
            Everything after the `?` — including `header.Authorization` — goes.
        """
        embedded = "/vsicurl?header.Authorization=Bearer%20tok&url=https%3A%2F%2Fh%2Fa"
        assert redact(embedded) == "/vsicurl?<redacted>", "the header must not survive"


class TestCredentialsStayOutOfMessages:
    """A credential must never reach an exception, a warning, or a log."""

    @staticmethod
    def _signer():
        """Return a bearer signer whose token is easy to grep for."""

        class _BearerSigner:
            def sign_href(self, href):
                return href

            def gdal_env(self):
                return {"GDAL_HTTP_HEADERS": "Authorization: Bearer SUPERSECRET"}

        return _BearerSigner()

    def test_strict_error_hides_the_token(self, tmp_path):
        """The drop error names the href, never the embedded credential.

        Test scenario:
            An expired href is the *expected* failure on a large mosaic, so its
            error is the most likely thing to reach a log aggregator.
        """
        good = write_raster(
            tmp_path / "g.tif", np.full((2, 2), 1.0, "float32"), (0.0, 2.0)
        )
        items = [
            {"assets": {"d": {"href": good}}},
            {"assets": {"d": {"href": "https://host.invalid/gone.tif"}}},
        ]
        with pytest.raises(RuntimeError) as exc:
            build_vrt_from_stac(items, asset="d", signer=self._signer(), strict=True)
        assert "SUPERSECRET" not in str(exc.value), (
            f"the bearer token leaked into the error: {exc.value}"
        )
        assert "host.invalid/gone.tif" in str(exc.value), (
            f"the dropped href should still be named: {exc.value}"
        )

    def test_warning_hides_the_token(self, tmp_path):
        """The non-strict warning is redacted the same way.

        Test scenario:
            `strict=False` warns instead of raising — same exposure risk.
        """
        good = write_raster(
            tmp_path / "w.tif", np.full((2, 2), 1.0, "float32"), (0.0, 2.0)
        )
        items = [
            {"assets": {"d": {"href": good}}},
            {"assets": {"d": {"href": "https://host.invalid/gone.tif"}}},
        ]
        with pytest.warns(UserWarning) as caught:
            build_vrt_from_stac(items, asset="d", signer=self._signer(), strict=False)
        joined = " ".join(str(w.message) for w in caught)
        assert "SUPERSECRET" not in joined, f"the token leaked into a warning: {joined}"

    def test_url_signed_token_is_hidden_too(self, tmp_path):
        """A SAS token in the href itself is redacted, not just embedded headers.

        Test scenario:
            URL-signing signers put the credential in the query string.
        """
        good = write_raster(
            tmp_path / "s.tif", np.full((2, 2), 1.0, "float32"), (0.0, 2.0)
        )

        class _SasSigner:
            def sign_href(self, href):
                return f"{href}?sig=SASSECRET" if href.startswith("http") else href

            def gdal_env(self):
                return {}

        items = [
            {"assets": {"d": {"href": good}}},
            {"assets": {"d": {"href": "https://host.invalid/gone.tif"}}},
        ]
        with pytest.raises(RuntimeError) as exc:
            build_vrt_from_stac(items, asset="d", signer=_SasSigner(), strict=True)
        assert "SASSECRET" not in str(exc.value), (
            f"the SAS token leaked into the error: {exc.value}"
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
            build_vrt_from_stac(mismatched_band_tiles, asset="data", strict=True)

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
            build_vrt_from_stac(items, asset="data", strict=True)

    def test_error_names_the_dropped_source(self, mismatched_band_tiles):
        """The error lists the href that was dropped, not just a count.

        Test scenario:
            The 3-band tile's path appears in the message so it is actionable.
        """
        dropped_href = mismatched_band_tiles[1]["assets"]["data"]["href"]
        with pytest.raises(RuntimeError) as exc:
            build_vrt_from_stac(mismatched_band_tiles, asset="data", strict=True)
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
            build_vrt_from_stac(items, asset="data", strict=True)


class TestSourceCredentialEmbedding:
    """Header credentials must ride the source path (ARC-24).

    GDAL opens a VRT's sources on the first pixel read and ignores the
    thread-local config when it does, so a signer env installed around the read
    never reaches them. Header credentials are embedded per source instead.
    """

    def test_bearer_header_is_embedded_in_the_source(self):
        """A bearer header becomes a `/vsicurl?header.…` source path.

        Test scenario:
            The token is URL-encoded into the path GDAL stores in the VRT.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        source = _embed_source_options("https://h/a.tif", env)
        assert source.startswith("/vsicurl?header.Authorization=Bearer%20tok"), (
            f"header not embedded: {source}"
        )
        assert source.endswith("url=https%3A%2F%2Fh%2Fa.tif"), (
            f"source url not encoded: {source}"
        )

    def test_readdir_skip_rides_along(self):
        """The readdir skip is carried into the read-time opens.

        Test scenario:
            Without it each source re-probes its sidecars on every open — 14
            wasted requests for two sources, measured. The build already runs
            under the same skip, so embedding keeps build and read consistent.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        assert "empty_dir=yes" in _embed_source_options("https://h/a.tif", env), (
            "the readdir skip should be embedded alongside the header"
        )

    def test_already_vsi_href_is_still_rewritten(self):
        """An href handed in as `/vsicurl/...` still gets its credentials.

        Test scenario:
            Matching on the raw href would silently skip this shape.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        source = _embed_source_options("/vsicurl/https://h/a.tif", env)
        assert source.startswith("/vsicurl?header.Authorization="), (
            f"a pre-rewritten href should still be embedded: {source}"
        )

    def test_archive_chaining_is_preserved(self):
        """A zipped source keeps its `/vsizip/` prefix instead of losing it.

        Test scenario:
            Rewriting it to the query form would drop the chaining and the
            source would fail to open.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        source = _embed_source_options("https://h/scene.zip/a.tif", env)
        assert source == "/vsizip//vsicurl/https://h/scene.zip/a.tif", (
            f"archive chaining lost: {source}"
        )

    def test_multiple_headers_all_embedded(self):
        """Every header in the env reaches the source path.

        Test scenario:
            A newline-separated pair produces two `header.` options.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok\r\nX-Trace: 42"}
        source = _embed_source_options("https://h/a.tif", env)
        assert "header.Authorization=Bearer%20tok" in source, f"missing auth: {source}"
        assert "header.X-Trace=42" in source, f"missing second header: {source}"

    def test_no_signer_uses_the_plain_rewrite(self):
        """Without credentials the ordinary VSI rewrite is kept.

        Test scenario:
            The unsigned path must not change shape.
        """
        assert _embed_source_options("https://h/a.tif", None) == (
            "/vsicurl/https://h/a.tif"
        ), "an unsigned source should keep the plain form"

    def test_non_http_scheme_is_left_alone(self):
        """Only HTTP(S) sources can carry headers in the path.

        Test scenario:
            An `s3://` href keeps its `/vsis3/` rewrite.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        assert _embed_source_options("s3://bucket/a.tif", env) == "/vsis3/bucket/a.tif"

    def test_local_source_is_left_alone(self):
        """A local path is never rewritten to a curl source.

        Test scenario:
            Header credentials are irrelevant to a local file.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        assert _embed_source_options("/data/a.tif", env) == "/data/a.tif"

    def test_embedded_source_reads(self, tmp_path):
        """A VRT over embedded-option sources still builds and reads.

        Test scenario:
            The `/vsicurl?` form is only produced for HTTP hrefs, so this covers
            the local fallback end to end: the mosaic is unaffected.
        """
        a = write_raster(
            tmp_path / "e1.tif", np.full((2, 2), 3.0, "float32"), (0.0, 2.0)
        )
        b = write_raster(
            tmp_path / "e2.tif", np.full((2, 2), 4.0, "float32"), (2.0, 2.0)
        )

        class _HeaderSigner:
            def sign_href(self, href):
                return href

            def gdal_env(self):
                return {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}

        items = [{"assets": {"d": {"href": a}}}, {"assets": {"d": {"href": b}}}]
        ds = build_vrt_from_stac(items, asset="d", signer=_HeaderSigner())
        assert ds.read_array().shape == (2, 4), "the local mosaic should be unaffected"


class TestUnembeddableCredentialWarning:
    """Credentials with no `/vsicurl?` equivalent are called out at build time."""

    def test_requester_pays_env_warns_for_remote_sources(self):
        """An AWS env cannot reach the read-time opens, so the build warns.

        Test scenario:
            A remote source plus `AWS_REQUEST_PAYER` -> a clear warning.
        """
        env = {"AWS_REQUEST_PAYER": "requester"}
        with pytest.warns(UserWarning, match="AWS_REQUEST_PAYER"):
            _warn_unembeddable_credentials(env, ["s3://bucket/a.tif"])

    def test_mixed_env_still_warns_for_the_stranded_half(self):
        """Headers plus AWS keys: the AWS half is stranded, so it still warns.

        Test scenario:
            Classifying the rewritten `/vsicurl?...` path as non-remote used to
            suppress this warning entirely.
        """
        env = {
            "GDAL_HTTP_HEADERS": "Authorization: Bearer tok",
            "AWS_REQUEST_PAYER": "requester",
        }
        with pytest.warns(UserWarning, match="AWS_REQUEST_PAYER"):
            _warn_unembeddable_credentials(env, ["https://h/a.tif"])

    def test_header_only_env_does_not_warn(self, recwarn):
        """A header signer is fully supported, so nothing is warned about.

        Args:
            recwarn: pytest fixture recording warnings raised in the block.

        Test scenario:
            `GDAL_HTTP_HEADERS` is embeddable — no warning.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        _warn_unembeddable_credentials(env, ["https://h/a.tif"])
        assert len(recwarn) == 0, f"unexpected warnings: {[str(w) for w in recwarn]}"

    def test_tuning_knobs_do_not_warn(self, recwarn):
        """Performance knobs are not credentials, so losing them is not an error.

        Test scenario:
            `GDAL_HTTP_MULTIPLEX` alone must not trip the warning.
        """
        env = {"GDAL_HTTP_MULTIPLEX": "YES"}
        _warn_unembeddable_credentials(env, ["https://h/a.tif"])
        assert len(recwarn) == 0, f"unexpected warnings: {[str(w) for w in recwarn]}"

    def test_local_sources_do_not_warn(self, recwarn):
        """An all-local build needs no credentials at all.

        Test scenario:
            Even a stranded AWS key is irrelevant for local sources.
        """
        _warn_unembeddable_credentials({"AWS_REQUEST_PAYER": "requester"}, ["/d/a.tif"])
        assert len(recwarn) == 0, f"unexpected warnings: {[str(w) for w in recwarn]}"


class TestSourceConfig:
    """Tests for the build-time GDAL config chosen by the source mix."""

    def test_remote_sources_get_the_fast_read_preset(self):
        """A remote build installs the `/vsicurl/` tuning preset.

        Test scenario:
            An un-tuned build costs a directory listing plus sidecar probes per
            source.
        """
        config = _source_config(["https://h/a.tif"], None).as_gdal_config()
        assert config["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR", config
        assert config["GDAL_HTTP_MULTIRANGE"] == "YES", config

    def test_header_signed_remote_build_gets_the_preset(self):
        """The header-signed build — the case with the largest measured saving.

        Test scenario:
            Deciding remoteness on the rewritten `/vsicurl?...` path instead of
            the href silently opted exactly this case out of the preset.
        """
        env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
        config = _source_config(["https://h/a.tif"], env).as_gdal_config()
        assert config["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR", config
        assert config["GDAL_HTTP_MULTIPLEX"] == "YES", config
        assert config["GDAL_HTTP_HEADERS"] == "Authorization: Bearer tok", config

    def test_signer_env_overrides_the_preset(self):
        """The signer env wins on a key conflict.

        Test scenario:
            An explicit HTTP version overrides the preset's default.
        """
        config = _source_config(
            ["s3://b/a.tif"], {"GDAL_HTTP_VERSION": "1.1"}
        ).as_gdal_config()
        assert config["GDAL_HTTP_VERSION"] == "1.1", f"signer env did not win: {config}"

    def test_local_sources_skip_the_preset(self):
        """An all-local build installs the signer env only.

        Test scenario:
            The preset's readdir skip would hide a local source's `.aux.xml`.
        """
        config = _source_config(["C:/data/a.tif"], {"CPL_CURL_VERBOSE": "YES"})
        assert config.as_gdal_config() == {"CPL_CURL_VERBOSE": "YES"}, (
            "a local build must not pull in the /vsicurl preset"
        )

    def test_local_build_without_signer_is_a_no_op(self):
        """Nothing at all is installed for a plain local build.

        Test scenario:
            The returned context manager is usable and changes no config.
        """
        with _source_config(["C:/data/a.tif"], None):
            assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") is None


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
            build_vrt_from_stac(mismatched_band_tiles, asset="data", strict=True)
        assert vsimem_entries() - before == set(), (
            "the VRT must be unlinked when the completeness guard fails"
        )


class TestStrictDefaultDeprecation:
    """The default flips to strict=True next minor, so the default warns now."""

    def test_default_warns_and_returns_the_partial_mosaic(self, mismatched_band_tiles):
        """Leaving strict unset keeps today's behaviour and flags the change.

        Test scenario:
            A skipped source warns twice — once about the incomplete mosaic, once
            that this default is going away — and still returns the partial VRT.
        """
        with pytest.warns(DeprecationWarning, match="becomes True in the next"):
            ds = build_vrt_from_stac(mismatched_band_tiles, asset="data")
        assert ds.read_array().shape == (4, 4), "the partial mosaic should be returned"

    def test_default_also_warns_about_the_skip(self, mismatched_band_tiles):
        """The completeness warning still fires under the default.

        Test scenario:
            The deprecation notice must not replace the actual problem report.
        """
        with pytest.warns(UserWarning, match="skipped 1 of 2"):
            build_vrt_from_stac(mismatched_band_tiles, asset="data")

    def test_explicit_false_does_not_deprecation_warn(self, mismatched_band_tiles):
        """A caller who pinned strict=False has nothing to migrate.

        Test scenario:
            Only the completeness warning fires.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_vrt_from_stac(mismatched_band_tiles, asset="data", strict=False)
        kinds = [w.category for w in caught]
        assert DeprecationWarning not in kinds, f"unexpected deprecation: {kinds}"
        assert UserWarning in kinds, f"the skip should still warn: {kinds}"

    def test_complete_build_never_warns(self, adjacent_tiles):
        """A clean build under the default is silent.

        Test scenario:
            The deprecation notice fires only when the default actually changes
            the outcome — i.e. only when a source was skipped.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_vrt_from_stac(adjacent_tiles, asset="data")
        assert not caught, (
            f"a complete build should not warn: {[str(w) for w in caught]}"
        )
