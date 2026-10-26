"""The vector-source driver registry and ``FeatureCollection.read_file`` dispatch, no plugin installed.

Every test builds its own fake driver *class* (claims are class-level, so discovery never constructs
one). Two fixtures make that possible: ``registered`` injects classes straight into the registry
cache, and ``fake_entry_points`` monkeypatches ``importlib.metadata.entry_points`` with in-memory
``EntryPoint`` objects so real discovery runs without a distribution on disk.

The negative cases matter most: a plugin that raises on import, declares a future API version, is
missing its ``extra`` dependency, or collides on a name must be skipped with a warning, because the
alternative is a broken third-party package stopping a vector file from opening through OGR.

Unlike the raster hook (GDAL-first, driver as a post-failure fallback), a claiming vector driver
pre-empts OGR: OGR would otherwise silently mis-read a look-alike (a CityJSON ``.json`` handed to
its GeoJSON driver), so the dispatch tests assert the driver wins *before* the built-in reader.
"""

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._driver_discovery import extract_tokens
from pyramids.base._errors import DriverNotExistError
from pyramids.feature import FeatureCollection, _drivers
from pyramids.feature._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    VECTOR_DRIVER_API_VERSION,
    clear_driver_cache,
    describe_drivers,
    drivers_for_scheme,
    drivers_for_suffix,
    resolve_driver,
    select_driver,
    vector_drivers,
)
from pyramids.feature.driver_conformance import (
    assert_vector_driver_conforms,
    check_vector_driver,
)
from tests._driver_discovery_fixtures import make_publish_entry_points, make_register

pytestmark = pytest.mark.core

GAUGES = Path(__file__).resolve().parents[1] / "data" / "coello-gauges.geojson"
GAUGES_FEATURES = 6


def _gdf() -> gpd.GeoDataFrame:
    """Build the smallest GeoDataFrame a fake driver can return.

    Returns:
        gpd.GeoDataFrame: a one-row point frame in EPSG:4326.
    """
    return gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")


class GoodDriver:
    """A minimal conforming vector driver class."""

    name = "cityjson"
    long_name = "CityJSON"
    api_version = VECTOR_DRIVER_API_VERSION
    suffixes = (".cityjson",)
    schemes = ()
    capabilities = frozenset({"read"})
    extra = None

    @classmethod
    def can_open(cls, source):
        """Claim a ``.cityjson`` source by suffix."""
        return isinstance(source, str) and source.endswith(".cityjson")

    def read(self, source, **kwargs):
        """Return a stub one-row GeoDataFrame."""
        return _gdf()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the discovery cache around every test so one never leaks into the next."""
    clear_driver_cache()
    yield
    clear_driver_cache()


@pytest.fixture
def registered(monkeypatch):
    """Inject driver classes directly into the registry cache (shared helper)."""
    return make_register(monkeypatch, _drivers)


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Monkeypatch ``entry_points`` so real discovery runs over in-memory entry points."""
    publish, published = make_publish_entry_points(
        monkeypatch, _drivers, "_fake_vector_driver_"
    )
    yield publish
    for attribute in published:
        monkeypatch.delattr(_drivers, attribute, raising=False)


class TestValidation:
    """Discovery rejects a driver class that does not meet the contract."""

    @pytest.mark.parametrize(
        "mutate, fragment",
        [
            ({"name": ""}, "non-blank string name"),
            ({"name": "ogr"}, "reserved name"),
            ({"long_name": 5}, "non-string long_name"),
            ({"api_version": 99}, "speaks version"),
            ({"capabilities": "read"}, "expected a set of tokens"),
            (
                {"capabilities": frozenset({"read", "bogus"})},
                "unknown capability token",
            ),
            ({"capabilities": frozenset()}, "required 'read' capability"),
            ({"suffixes": 5}, "declares suffixes 5"),
            ({"schemes": 5}, "declares schemes 5"),
            ({"suffixes": (), "schemes": ()}, "neither suffixes nor schemes"),
            ({"suffixes": ("cityjson",)}, "malformed suffix"),
            ({"suffixes": (".",)}, "malformed suffix"),
            ({"schemes": ("cityjson://",)}, "malformed scheme"),
            ({"schemes": ("",)}, "malformed scheme"),
            ({"schemes": (" ",)}, "malformed scheme"),
            ({"schemes": ("c c",)}, "malformed scheme"),
        ],
    )
    def test_a_malformed_claim_is_skipped_with_a_warning(
        self, fake_entry_points, mutate, fragment
    ):
        """Each contract violation skips the driver and warns naming the problem.

        Args:
            fake_entry_points: Fixture publishing in-memory entry points.
            mutate: The single attribute override that breaks the contract.
            fragment: A substring the warning must contain.

        Test scenario:
            A driver built from ``GoodDriver`` with one broken claim is discovered, skipped, and
            warned about, leaving an empty registry.
        """
        attrs = {**GoodDriver.__dict__}
        bad = type("Bad", (), {**attrs, **mutate})
        fake_entry_points(("bad", bad))
        with pytest.warns(RuntimeWarning, match=fragment):
            assert vector_drivers() == {}, "a malformed driver must not register"

    def test_missing_attribute_is_skipped(self, fake_entry_points):
        """A class missing a required claim is skipped with a warning.

        Test scenario:
            A class with only ``api_version`` is missing ``name`` and the rest, so discovery skips
            it with a "missing attribute" warning.
        """

        class Nameless:
            api_version = VECTOR_DRIVER_API_VERSION

        fake_entry_points(("nameless", Nameless))
        with pytest.warns(RuntimeWarning, match="missing attribute"):
            assert vector_drivers() == {}, "a class missing claims must not register"

    def test_a_driver_missing_a_promised_method_is_skipped(self, fake_entry_points):
        """A driver declaring 'read' but not implementing it is skipped with a 'does not implement' warning.

        Test scenario:
            The class is otherwise valid but defines no ``read`` method, so ``_validate_claims``
            reports the missing method and discovery drops it, leaving an empty registry.
        """

        class NoRead:
            name, long_name = "noread", "No Read"
            api_version = VECTOR_DRIVER_API_VERSION
            suffixes, schemes = (".noread",), ()
            capabilities, extra = frozenset({"read"}), None

            @classmethod
            def can_open(cls, source):
                return False

        fake_entry_points(("noread", NoRead))
        with pytest.warns(RuntimeWarning, match="does not implement"):
            assert vector_drivers() == {}, (
                "a driver missing a promised method must not register"
            )

    def test_every_capability_token_is_accepted(self, fake_entry_points):
        """A driver may declare every known token (``read`` + the declarative ``needs_auth``).

        Test scenario:
            ``needs_auth`` promises no method, so a driver declaring both tokens loads normally.
        """

        class Everything(GoodDriver):
            name = "everything"
            capabilities = frozenset(CAPABILITIES)

        fake_entry_points(("everything", Everything))
        assert sorted(vector_drivers()) == ["everything"], (
            "a driver declaring every known token should load"
        )

    def test_driver_failing_to_load_is_skipped_with_a_warning(self, monkeypatch):
        """An entry point whose target cannot be resolved costs itself, not the registry.

        Test scenario:
            One entry point points at a missing attribute and is skipped with a warning; the other
            resolves and registers, so one broken plugin does not take out a good one.
        """
        monkeypatch.setattr(_drivers, "_good_target", GoodDriver, raising=False)
        entries = (
            EntryPoint(
                name="broken",
                value="pyramids.feature._drivers:_missing_target_xyz",
                group=ENTRY_POINT_GROUP,
            ),
            EntryPoint(
                name="cityjson",
                value="pyramids.feature._drivers:_good_target",
                group=ENTRY_POINT_GROUP,
            ),
        )
        monkeypatch.setattr(
            _drivers,
            "entry_points",
            lambda *, group=None: entries if group == ENTRY_POINT_GROUP else (),
        )
        monkeypatch.setattr(_drivers, "_CACHE", None)
        with pytest.warns(RuntimeWarning, match="could not be loaded"):
            assert sorted(vector_drivers()) == ["cityjson"], (
                "the resolvable driver should still register"
            )
        monkeypatch.delattr(_drivers, "_good_target", raising=False)

    def test_duplicate_name_keeps_the_first(self, fake_entry_points):
        """A name registered twice keeps the first and warns about the second.

        Test scenario:
            Two entry points both named ``cityjson`` — the first wins, the second is skipped with a
            "registered twice" warning.
        """

        class Other(GoodDriver):
            long_name = "Other"

        fake_entry_points(("a", GoodDriver), ("b", Other))
        with pytest.warns(RuntimeWarning, match="registered twice"):
            drivers = vector_drivers()
        assert list(drivers) == ["cityjson"], "the first registration must win"
        assert drivers["cityjson"].long_name == "CityJSON", (
            "the kept driver is the first"
        )


class TestExtraGating:
    """A driver is skipped when its declared optional dependency is absent."""

    def test_missing_extra_is_skipped_silently(self, fake_entry_points):
        """An absent extra skips the driver without importing its backend or warning.

        Test scenario:
            A driver whose ``extra`` names a non-existent package is dropped at discovery with no
            warning (it is a legitimate opt-out, not an error).
        """

        class NeedsBackend(GoodDriver):
            extra = "no_such_backend_xyz"

        fake_entry_points(("needs", NeedsBackend))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert vector_drivers() == {}, "a missing extra must gate the driver out"

    def test_present_extra_is_loaded(self, fake_entry_points):
        """An importable extra (here the stdlib ``os``) does not gate the driver out.

        Test scenario:
            ``extra="os"`` is always importable, so the driver registers normally.
        """

        class HasBackend(GoodDriver):
            extra = "os"

        fake_entry_points(("has", HasBackend))
        assert sorted(vector_drivers()) == ["cityjson"], (
            "an importable extra must not gate the driver out"
        )

    def test_unresolvable_extra_is_treated_as_missing(self, fake_entry_points):
        """An extra naming a missing parent package (find_spec raises) gates the driver out.

        Test scenario:
            ``find_spec`` raises ``ModuleNotFoundError`` for ``no_such_parent_xyz.child``; the guard
            treats that as missing and silently drops the driver.
        """

        class NeedsChild(GoodDriver):
            extra = "no_such_parent_xyz.child"

        fake_entry_points(("needs", NeedsChild))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert vector_drivers() == {}, (
                "an unresolvable extra must gate the driver out"
            )


class TestHelpers:
    """The standalone resolver helpers."""

    def test_resolve_driver_returns_the_class(self, registered):
        """resolve_driver returns the registered class for a known name."""
        registered(GoodDriver)
        assert resolve_driver("cityjson") is GoodDriver, "exact name must resolve"

    def test_resolve_driver_unknown_raises(self, registered):
        """resolve_driver raises DriverNotExistError for an unregistered name."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            resolve_driver("nope")

    def test_resolve_driver_is_case_sensitive(self, registered):
        """A driver name is an identifier; a different case does not resolve it."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="CityJSON"):
            resolve_driver("CityJSON")

    def test_select_driver_with_an_explicit_name_delegates_to_resolve(self, registered):
        """select_driver with a name returns that driver, bypassing the can_open scan."""
        registered(GoodDriver)
        assert select_driver("anything", "cityjson") is GoodDriver, (
            "an explicit name must bypass the scan"
        )

    def test_select_driver_returns_none_on_a_bare_install(self):
        """With nothing registered, the scan claims nothing and the caller falls back to OGR."""
        assert select_driver("anything.cityjson", None) is None, (
            "a bare install claims nothing"
        )


class TestDiscovery:
    """Discovery order, the copy contract, and the lookup helpers."""

    def test_discovery_preserves_entry_point_order(self, fake_entry_points):
        """The registry keeps drivers in the order discovery found them."""

        class Second(GoodDriver):
            name = "citygml"
            suffixes = (".gml",)

        fake_entry_points(("cityjson", GoodDriver), ("citygml", Second))
        assert list(vector_drivers()) == ["cityjson", "citygml"], (
            "discovery order must be preserved"
        )

    def test_vector_drivers_returns_a_copy(self, registered):
        """Mutating the returned mapping does not reach the registry."""
        registered(GoodDriver)
        drivers = vector_drivers()
        drivers.clear()
        assert sorted(vector_drivers()) == ["cityjson"], (
            "the returned mapping must be a copy"
        )

    def test_reentrant_discovery_during_load_registers_the_driver(self, monkeypatch):
        """A plugin that consults the registry during its own load is still registered."""

        class ReentrantEntryPoint:
            name = "cityjson"
            value = "your_package:CityJSONDriver"

            def load(self):
                vector_drivers()
                return GoodDriver

        monkeypatch.setattr(_drivers, "_CACHE", None)
        monkeypatch.setattr(
            _drivers,
            "entry_points",
            lambda *, group=None: (
                (ReentrantEntryPoint(),) if group == ENTRY_POINT_GROUP else ()
            ),
        )
        assert list(vector_drivers()) == ["cityjson"], (
            "re-entrant load must not drop the driver"
        )

    def test_a_concurrent_publish_keeps_the_first_result(self, monkeypatch):
        """If another thread publishes between discovery and the lock, the first result wins."""
        first = {"cityjson": GoodDriver}

        def _discover_then_race():
            _drivers._CACHE = first
            return {"other": GoodDriver}

        monkeypatch.setattr(_drivers, "_CACHE", None)
        monkeypatch.setattr(_drivers, "_discover", _discover_then_race)
        result = vector_drivers()
        assert list(result) == ["cityjson"], "the already-published mapping must win"
        assert result is not first, "callers get a copy, not the live cache"

    def test_drivers_for_suffix_is_case_and_dot_insensitive(self, registered):
        """A suffix resolves with or without its dot and in any case."""
        registered(GoodDriver)
        assert [d.name for d in drivers_for_suffix(".cityjson")] == ["cityjson"]
        assert [d.name for d in drivers_for_suffix("CITYJSON")] == ["cityjson"]
        assert drivers_for_suffix(".shp") == [], "an unclaimed suffix yields nothing"

    def test_drivers_for_scheme_strips_separator(self, registered):
        """A scheme resolves with or without a trailing ``://``."""

        class Scheme(GoodDriver):
            name = "cityjson-url"
            suffixes = ()
            schemes = ("cityjson",)

        registered(Scheme)
        assert [d.name for d in drivers_for_scheme("cityjson")] == ["cityjson-url"]
        assert [d.name for d in drivers_for_scheme("CITYJSON://")] == ["cityjson-url"]
        assert drivers_for_scheme("s3") == [], "an unclaimed scheme yields nothing"

    def test_describe_drivers_is_serialisable(self, registered):
        """describe_drivers returns a plain-dict summary per driver."""
        registered(GoodDriver)
        assert describe_drivers() == [
            {
                "name": "cityjson",
                "long_name": "CityJSON",
                "suffixes": [".cityjson"],
                "schemes": [],
                "capabilities": ["read"],
            }
        ], "describe_drivers must return a serialisable summary"


class TestSourceTokens:
    """The shared ``extract_tokens`` suffix/scheme extraction that gates the scan's pre-filter."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("city.cityjson", {".cityjson"}),
            ("/data/city.CityJSON", {".cityjson"}),
            ("https://host/city.cityjson", {".cityjson", "https"}),
            ("s3://bucket/roads.gml", {".gml", "s3"}),
            ("s3://bucket", {"s3"}),
            ("no-suffix-no-scheme", set()),
            ("/data/.bashrc", set()),
            ("foo.tar.gz", {".gz"}),
            ("/data/dir/", set()),
            ("https://host/path/x.json?token=abc", {".json", "https"}),
            ("https://host/x.gml#frag", {".gml", "https"}),
            (r"C:\data\city.cityjson", {".cityjson"}),
        ],
    )
    def test_extracts_scheme_and_suffix(self, source, expected):
        """Scheme (no ``://``) and the final dot-suffix are both pulled, lowercased.

        Args:
            source: The source string.
            expected: The expected token set.
        """
        assert extract_tokens(source) == expected, f"{source!r} -> {expected}"


class TestCacheColdness:
    """A bare install does at most one cached entry-point scan across many vector reads."""

    def test_bare_registry_is_empty(self):
        """With no plugin installed the registry is empty."""
        assert vector_drivers() == {}, "a bare install registers nothing"

    def test_repeated_reads_discover_at_most_once(self, monkeypatch):
        """Two normal reads warm the discovery cache exactly once, then reuse it.

        Test scenario:
            ``entry_points`` is wrapped with a counter. Reading a real GeoJSON twice calls it once
            (first read discovers an empty, cached registry; the second read reuses the cache), and
            the file still opens through OGR with all its features.
        """
        calls = {"n": 0}
        real_entry_points = _drivers.entry_points

        def _counting(*, group=None):
            calls["n"] += 1
            return real_entry_points(group=group)

        monkeypatch.setattr(_drivers, "entry_points", _counting)
        clear_driver_cache()

        first = FeatureCollection.read_file(str(GAUGES))
        second = FeatureCollection.read_file(str(GAUGES))

        assert calls["n"] == 1, f"expected one discovery scan, got {calls['n']}"
        assert _drivers._CACHE == {}, "discovery must cache an empty registry"
        assert len(first) == GAUGES_FEATURES, "OGR must still read the file"
        assert len(second) == GAUGES_FEATURES, "the second read must match the first"


class TestDispatch:
    """``FeatureCollection.read_file`` routing: driver= override, can_open pre-emption, OGR."""

    def test_explicit_driver_bypasses_ogr(self, registered):
        """``driver=`` routes straight to the named plugin without touching OGR.

        Test scenario:
            The path has a ``.shp`` suffix the stub does not claim, but ``driver="cityjson"`` forces
            it; the stub's one-row frame proves OGR never ran.
        """
        registered(GoodDriver)
        fc = FeatureCollection.read_file("whatever.shp", driver="cityjson")
        assert len(fc) == 1, "the explicit driver's stub frame should be returned"

    def test_unknown_driver_raises(self, registered):
        """An unregistered ``driver=`` name is a clear error."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            FeatureCollection.read_file("x.cityjson", driver="nope")

    def test_driver_with_dask_backend_raises(self, registered):
        """A driver-selected read is eager-only: ``backend='dask'`` with a claim raises, not downgrades.

        Test scenario:
            ``city.cityjson`` is claimed by the stub, and a plugin driver returns an eager frame, so
            asking for the lazy backend must raise a clear error naming the driver rather than
            silently returning an eager ``FeatureCollection``.
        """
        registered(GoodDriver)
        with pytest.raises(ValueError, match="cityjson.*backend.*not supported"):
            FeatureCollection.read_file("city.cityjson", backend="dask")

    def test_bad_backend_is_rejected_on_the_driver_path(self, registered):
        """An invalid ``backend=`` is validated even when a driver would claim the source.

        Test scenario:
            With a driver installed and explicitly named, a nonsense ``backend`` must still raise the
            same "backend must be" error it raises on the OGR path, not be swallowed by the dispatch.
        """
        registered(GoodDriver)
        with pytest.raises(ValueError, match="backend must be"):
            FeatureCollection.read_file(
                "city.cityjson", driver="cityjson", backend="nonsense"
            )

    def test_can_open_claim_preempts_ogr(self, registered):
        """A registered driver whose can_open claims the source wins before OGR.

        Test scenario:
            ``city.cityjson`` is claimed by the stub's suffix check, so the read never reaches OGR
            (the file does not even exist) and returns the stub's one-row frame.
        """
        registered(GoodDriver)
        fc = FeatureCollection.read_file("city.cityjson")
        assert len(fc) == 1, "the claiming driver should pre-empt OGR"

    def test_non_claiming_source_falls_through_to_ogr(self, registered):
        """A source no driver claims still opens through the built-in OGR reader.

        Test scenario:
            With the ``.cityjson``-only stub installed, a real GeoJSON is not claimed, so OGR reads
            it and all its features come back.
        """
        registered(GoodDriver)
        fc = FeatureCollection.read_file(str(GAUGES))
        assert len(fc) == GAUGES_FEATURES, (
            "an unclaimed source must fall through to OGR"
        )

    def test_driver_returning_non_geodataframe_raises_typeerror(self, registered):
        """A driver whose read returns a non-GeoDataFrame is rejected with a clear TypeError."""

        class Bad(GoodDriver):
            def read(self, source, **kwargs):
                return {"not": "a gdf"}

        registered(Bad)
        with pytest.raises(TypeError, match="cityjson.*not a GeoDataFrame"):
            FeatureCollection.read_file("x.cityjson", driver="cityjson")

    def test_passthrough_filters_reach_the_driver(self, registered):
        """The read_file filters (bbox/where/…) are forwarded to the driver's read."""
        seen = {}

        class Rec(GoodDriver):
            def read(self, source, **kwargs):
                seen.update(kwargs)
                return _gdf()

        registered(Rec)
        FeatureCollection.read_file(
            "x.cityjson", driver="cityjson", bbox=(0, 0, 1, 1), where="pop > 1"
        )
        assert seen.get("bbox") == (0, 0, 1, 1), f"bbox not forwarded: {seen}"
        assert seen.get("where") == "pop > 1", f"where not forwarded: {seen}"

    def test_a_can_open_that_raises_is_skipped_and_ogr_reads(self, registered):
        """A driver whose can_open raises is skipped with a warning; OGR still reads the source.

        Test scenario:
            Scanning a real GeoJSON hits the exploding can_open, which is warned about and treated
            as "does not claim", so the read falls through to OGR and returns every feature.
        """

        class Explodes(GoodDriver):
            suffixes = (".geojson",)

            @classmethod
            def can_open(cls, source):
                raise RuntimeError("bad can_open")

        registered(Explodes)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fc = FeatureCollection.read_file(str(GAUGES))
        assert len(fc) == GAUGES_FEATURES, "OGR must read the source after the skip"
        assert any(
            "raised from can_open" in str(record.message) for record in caught
        ), "expected the skipped-can_open RuntimeWarning"

    def test_can_open_is_not_called_for_a_non_matching_suffix(self, registered):
        """The scan pre-filters by suffix/scheme, so can_open runs only for a plausible driver.

        Test scenario:
            A ``.cityjson``-only driver whose can_open would claim anything is installed, and a
            ``.geojson`` is read. The suffix pre-filter means can_open is never consulted, so OGR
            reads the file and every feature comes back — one slow/over-broad plugin cannot tax or
            hijack an unrelated read.
        """
        calls = []

        class Nosy(GoodDriver):
            @classmethod
            def can_open(cls, source):
                calls.append(source)
                return True

        registered(Nosy)
        fc = FeatureCollection.read_file(str(GAUGES))
        assert len(fc) == GAUGES_FEATURES, "an unmatched-suffix source must reach OGR"
        assert calls == [], (
            f"can_open must not run for a non-matching suffix, got {calls}"
        )

    def test_the_scan_does_not_instantiate_a_non_claiming_driver(self, registered):
        """can_open is a classmethod; only the chosen driver is ever constructed."""
        built = {"nonclaim": 0, "claim": 0}

        class NonClaim(GoodDriver):
            name = "nonclaim"
            suffixes = (".other",)

            @classmethod
            def can_open(cls, source):
                return False

            def __init__(self):
                built["nonclaim"] += 1

        class Claim(GoodDriver):
            name = "claim"

            @classmethod
            def can_open(cls, source):
                return str(source).endswith(".cityjson")

            def __init__(self):
                built["claim"] += 1

        registered(NonClaim, Claim)
        FeatureCollection.read_file("x.cityjson")
        assert built == {"nonclaim": 0, "claim": 1}, (
            "only the claiming driver should be constructed"
        )

    def test_scheme_claimed_source_dispatches_end_to_end(self, registered):
        """A driver matched by its declared *scheme* is dispatched through read_file, not just OGR.

        Test scenario:
            A driver declaring the ``cityjson`` scheme (no suffix) claims a ``cityjson://…`` source;
            the scheme token matches the pre-filter, can_open confirms, and the stub frame returns —
            exercising the scheme path through select_driver that the suffix tests do not.
        """

        class SchemeDriver(GoodDriver):
            name = "cityjson-url"
            suffixes = ()
            schemes = ("cityjson",)

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("cityjson://")

        registered(SchemeDriver)
        fc = FeatureCollection.read_file("cityjson://host/city")
        assert len(fc) == 1, "a scheme-claimed source should dispatch to the driver"


class TestMalformedDeclaredTokens:
    """select_driver tolerates a directly-injected driver with a malformed suffixes/schemes (S1).

    Production drivers only enter via validated discovery, but the registry can be injected directly
    (the tests do), so the scan's token read must never raise on a non-container or missing
    suffixes/schemes — mirroring the raster hook's ``_declared_preempts`` robustness.
    """

    @pytest.mark.parametrize("bad", [True, 5, object(), ".cityjson", {".cityjson": 1}])
    def test_a_non_container_suffixes_is_tolerated(self, registered, bad):
        """A non-(tuple/list/set/frozenset) suffixes yields no tokens instead of raising.

        Args:
            bad: A malformed suffixes value injected onto a driver (a bare string must not be
                treated as a one-tuple of characters either).
        """

        class Malformed(GoodDriver):
            suffixes = bad

        registered(Malformed)
        assert select_driver("city.cityjson") is None, (
            f"{bad!r} must not claim or raise"
        )

    def test_a_suffixes_whose_iter_raises_is_tolerated(self, registered):
        """A suffixes object whose ``__iter__``/``__bool__`` raises is rejected before iteration."""

        class Weird:
            def __iter__(self):
                raise ValueError("no iter")

            def __bool__(self):
                raise ValueError("no bool")

        class Malformed(GoodDriver):
            suffixes = Weird()

        registered(Malformed)
        assert select_driver("city.cityjson") is None, (
            "a raising container must be tolerated"
        )

    def test_a_missing_suffixes_attribute_is_tolerated(self, registered):
        """A driver missing ``suffixes`` entirely contributes only its schemes, without raising."""

        class NoSuffixes:
            name, long_name = "nosuf", "No Suffixes"
            api_version = VECTOR_DRIVER_API_VERSION
            schemes = ("cityjson",)
            capabilities, extra = frozenset({"read"}), None

            @classmethod
            def can_open(cls, source):
                return True

        registered(NoSuffixes)
        assert select_driver("roads.shp") is None, (
            "a missing suffixes attr must be tolerated"
        )


class TestConformanceKit:
    """The reusable conformance kit a downstream vector plugin runs against its driver."""

    def test_a_conforming_driver_has_no_problems(self):
        """A well-formed class passes both the static and behavioural checks."""
        assert check_vector_driver(GoodDriver) == [], "GoodDriver should conform"
        assert_vector_driver_conforms(GoodDriver)

    def test_a_static_violation_is_reported_first(self):
        """A contract violation surfaces as the first listed problem and raises on assert."""

        class Bad(GoodDriver):
            capabilities = frozenset()

        assert "required 'read'" in check_vector_driver(Bad)[0], "static problem first"
        with pytest.raises(AssertionError, match="does not conform"):
            assert_vector_driver_conforms(Bad)

    def test_a_can_open_that_raises_is_reported(self):
        """can_open must never raise; a kit probe catches one that does."""

        class Raises(GoodDriver):
            @classmethod
            def can_open(cls, source):
                raise ValueError("no")

        problems = check_vector_driver(Raises)
        assert any("must never raise" in problem for problem in problems), problems

    def test_a_driver_needing_constructor_args_is_reported(self):
        """A driver read_file cannot build with no arguments is caught by the kit."""

        class NeedsArg(GoodDriver):
            def __init__(self, required):
                self.required = required

        problems = check_vector_driver(NeedsArg)
        assert any("no arguments" in problem for problem in problems), problems

    def test_an_instance_shadowing_read_with_a_non_callable_is_reported(self):
        """A driver whose instance shadows read with a non-callable is caught."""

        class ShadowsRead(GoodDriver):
            def __init__(self):
                self.read = "not callable"

        problems = check_vector_driver(ShadowsRead)
        assert any("no callable read" in problem for problem in problems), problems

    def test_a_non_bool_can_open_is_reported(self):
        """can_open must return a bool; a kit probe catches a truthy non-bool."""

        class Truthy(GoodDriver):
            @classmethod
            def can_open(cls, source):
                return "yes"

        problems = check_vector_driver(Truthy)
        assert any("must return a bool" in problem for problem in problems), problems
