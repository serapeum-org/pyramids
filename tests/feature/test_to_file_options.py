"""ARC-26: FeatureCollection.to_file(layer=, mode=, **creation_options).

Covers the three newly-promoted first-class kwargs on ``to_file``:
layer selection, append mode, and driver creation options. All three
forward to the underlying pyogrio / fiona engine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def fc_rivers() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"name": ["r1", "r2"]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def fc_lakes() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"name": ["l1", "l2", "l3"]},
            geometry=[Point(10, 10), Point(11, 11), Point(12, 12)],
            crs="EPSG:4326",
        )
    )


class TestLayerKwarg:
    """layer= writes into a named layer of a multi-layer driver (GPKG)."""

    def test_write_single_layer_gpkg(
        self, tmp_path: Path, fc_rivers: FeatureCollection
    ):
        p = tmp_path / "single.gpkg"
        fc_rivers.to_file(p, driver="gpkg", layer="rivers")
        reloaded = FeatureCollection.read_file(p, layer="rivers")
        assert len(reloaded) == 2
        assert set(reloaded["name"]) == {"r1", "r2"}

    def test_write_two_layers_same_gpkg(
        self,
        tmp_path: Path,
        fc_rivers: FeatureCollection,
        fc_lakes: FeatureCollection,
    ):
        p = tmp_path / "multi.gpkg"
        fc_rivers.to_file(p, driver="gpkg", layer="rivers")
        fc_lakes.to_file(p, driver="gpkg", layer="lakes")
        rivers = FeatureCollection.read_file(p, layer="rivers")
        lakes = FeatureCollection.read_file(p, layer="lakes")
        assert len(rivers) == 2 and len(lakes) == 3


class TestModeAppend:
    """mode='a' appends to an existing file / layer."""

    def test_append_to_existing_gpkg_layer(
        self, tmp_path: Path, fc_rivers: FeatureCollection
    ):
        p = tmp_path / "append.gpkg"
        fc_rivers.to_file(p, driver="gpkg", layer="rivers", mode="w")
        fc_rivers.to_file(p, driver="gpkg", layer="rivers", mode="a")
        reloaded = FeatureCollection.read_file(p, layer="rivers")
        # Wrote 2 rows twice -> 4 rows total.
        assert len(reloaded) == 4

    def test_invalid_mode_raises(self, tmp_path: Path, fc_rivers: FeatureCollection):
        with pytest.raises(ValueError, match="'w' .* 'a'"):
            fc_rivers.to_file(tmp_path / "x.geojson", mode="x")


class TestCreationOptions:
    """Driver creation options are forwarded through **kwargs."""

    def test_gpkg_spatial_index(self, tmp_path: Path, fc_rivers: FeatureCollection):
        """SPATIAL_INDEX='YES' should produce an rtree_<layer>_geom table."""
        p = tmp_path / "indexed.gpkg"
        fc_rivers.to_file(p, driver="gpkg", layer="rivers", SPATIAL_INDEX="YES")
        # Peek into the GPKG via sqlite3: rtree companion table is
        # named 'rtree_<layer>_<geom-column>'. Default geom-column
        # is 'geom' for GPKG, so we expect 'rtree_rivers_geom'.
        with sqlite3.connect(p) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name LIKE 'rtree_%'"
            ).fetchall()
        names = {r[0] for r in rows}
        assert any(
            n.startswith("rtree_rivers") for n in names
        ), f"expected an rtree_rivers_* table; found {names}"

    def test_geojson_rfc7946(self, tmp_path: Path, fc_rivers: FeatureCollection):
        """GeoJSON RFC7946=YES writes RFC 7946-compliant output.

        RFC 7946 mandates CRS84 (= EPSG:4326 with axis order lon/lat).
        The test just verifies the option is accepted — nothing raises.
        """
        p = tmp_path / "rfc7946.geojson"
        fc_rivers.to_file(p, driver="geojson", RFC7946="YES")
        assert p.exists()


class TestPlainPathStillWorks:
    """Back-compat: no new kwargs still writes a GeoJSON like before."""

    def test_default_geojson(self, tmp_path: Path, fc_rivers: FeatureCollection):
        p = tmp_path / "plain.geojson"
        fc_rivers.to_file(p)
        assert p.exists()
        reloaded = FeatureCollection.read_file(p)
        assert len(reloaded) == 2


class TestCreationOptionValidation:
    """C8: pyogrio already rejects unknown creation options at write-time.

    The original C8 concern ("GDAL silently ignores unknown options") is
    mooted by the pyogrio engine which geopandas 1.0+ uses by default —
    pyogrio validates options against both the dataset-level and
    layer-level creation-option metadata and raises
    :class:`ValueError` with the message
    ``"unrecognized option '<name>' for driver '<driver>'"`` before
    the write ever reaches GDAL. These tests pin that behaviour so
    regressions in the geopandas/pyogrio stack surface here rather
    than producing silently-different files.
    """

    def test_unknown_option_raises_value_error(
        self, tmp_path: Path, fc_rivers: FeatureCollection
    ):
        """Nonsense option raises ``ValueError`` naming the option + driver."""
        p = tmp_path / "warn.gpkg"
        with pytest.raises(ValueError, match="NOT_A_REAL_OPTION.*GPKG"):
            fc_rivers.to_file(p, driver="gpkg", NOT_A_REAL_OPTION="YES")

    def test_known_option_accepted(self, tmp_path: Path, fc_rivers: FeatureCollection):
        """A legitimate option completes the write successfully."""
        p = tmp_path / "ok.gpkg"
        fc_rivers.to_file(p, driver="gpkg", SPATIAL_INDEX="YES")
        assert p.exists()

    def test_mix_of_known_and_unknown_options_still_raises(
        self, tmp_path: Path, fc_rivers: FeatureCollection
    ):
        """Any single unknown option triggers the ValueError.

        Test scenario:
            pyogrio iterates the kwargs and stops on the first option
            that matches neither the dataset nor the layer option list.
            Supplying one known + one unknown option still raises.
        """
        p = tmp_path / "mix.gpkg"
        with pytest.raises(ValueError, match="BOGUS"):
            fc_rivers.to_file(
                p,
                driver="gpkg",
                SPATIAL_INDEX="YES",
                BOGUS="x",
            )

    def test_option_case_insensitive_accepted(
        self, tmp_path: Path, fc_rivers: FeatureCollection
    ):
        """Options are case-insensitive in the pyogrio layer.

        Test scenario:
            pyogrio uppercases the option key before checking the
            driver's metadata — so ``spatial_index`` and
            ``SPATIAL_INDEX`` are both accepted for GPKG. This pins
            the behaviour so a future pyogrio version that tightens
            to case-sensitive matching surfaces here rather than
            silently dropping the option.
        """
        p = tmp_path / "case.gpkg"
        fc_rivers.to_file(p, driver="gpkg", spatial_index="YES")
        assert p.exists()


class TestEnginePin:
    """C28: to_file pins ``engine="pyogrio"`` when forwarding to geopandas.

    Matches the pin on :meth:`read_file` and :meth:`iter_features`.
    A ``geopandas.options.io_engine = "fiona"`` global no longer
    changes the write path silently.
    """

    def test_passes_engine_pyogrio(
        self, tmp_path: Path, fc_rivers: FeatureCollection, monkeypatch
    ):
        """The ``engine="pyogrio"`` kwarg reaches the underlying writer."""
        captured: list = []
        from geopandas import GeoDataFrame as _GDF

        real_to_file = _GDF.to_file

        def _spy(self, path, **kwargs):
            captured.append(kwargs)
            return real_to_file(self, path, **kwargs)

        monkeypatch.setattr(_GDF, "to_file", _spy)
        fc_rivers.to_file(tmp_path / "pin.geojson")
        assert captured, "to_file must be invoked at least once"
        assert (
            captured[0].get("engine") == "pyogrio"
        ), f"expected engine='pyogrio', got {captured[0]}"
