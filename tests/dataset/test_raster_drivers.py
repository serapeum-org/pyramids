"""The raster-source driver registry and ``Dataset.read_file`` dispatch, with no plugin installed.

Every test builds its own fake driver *class* (claims are class-level, so discovery never
constructs one). Two fixtures make that possible: ``registered`` injects classes straight into the
registry cache, and ``fake_entry_points`` monkeypatches ``importlib.metadata.entry_points`` with
in-memory ``EntryPoint`` objects so real discovery runs without a distribution on disk.

The negative cases matter most: a plugin that raises on import, declares a future API version,
is missing its ``extra`` dependency, or collides on a name must be skipped with a warning, because
the alternative is a broken third-party package stopping a raster from opening through GDAL.
"""

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint

import numpy as np
import pytest

from pyramids.base._driver_discovery import extract_tokens
from pyramids.base._errors import DriverNotExistError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, _drivers
from pyramids.dataset._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    RASTER_DRIVER_API_VERSION,
    clear_driver_cache,
    describe_drivers,
    drivers_for_scheme,
    drivers_for_suffix,
    raster_drivers,
    select_driver,
    select_preempting_driver,
)
from pyramids.dataset.driver_conformance import (
    assert_raster_driver_conforms,
    check_raster_driver,
)
from tests._driver_discovery_fixtures import make_publish_entry_points, make_register

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326)


def _dataset() -> Dataset:
    """Build the smallest Dataset a fake driver can return.

    Returns:
        Dataset: a 2x2 in-memory raster.
    """
    return Dataset.from_array(np.ones((2, 2), dtype="float32"), geo_ref=GEO)


class GoodDriver:
    """A minimal conforming raster driver class."""

    name = "fci"
    long_name = "EUMETSAT FCI"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes = (".nat",)
    schemes = ()
    capabilities = frozenset({"read"})
    extra = None

    @classmethod
    def can_open(cls, source):
        """Claim a ``.nat`` granule."""
        return isinstance(source, str) and source.endswith(".nat")

    def read(self, source, *, bbox=None, bands=None, crs=None, options=None, auth=None):
        """Return a stub dataset."""
        return _dataset()


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
        monkeypatch, _drivers, "_fake_raster_driver_"
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
            ({"name": "gdal"}, "reserved name"),
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
            ({"suffixes": ("nat",)}, "malformed suffix"),
            ({"suffixes": (".",)}, "malformed suffix"),
            ({"schemes": ("ee://",)}, "malformed scheme"),
            ({"schemes": ("",)}, "malformed scheme"),
            ({"schemes": (" ",)}, "malformed scheme"),
            ({"schemes": ("e e",)}, "malformed scheme"),
            ({"schemes": ("ee ",)}, "malformed scheme"),
            ({"preempts": True}, "declares preempts"),
            ({"preempts": ".safe"}, "declares preempts"),
            ({"preempts": (".SAFE",)}, "malformed preempts"),
            ({"preempts": (".safe", 123)}, "malformed preempts"),
            ({"preempts": ("ee://",)}, "malformed preempts"),
        ],
    )
    def test_a_malformed_claim_is_skipped_with_a_warning(
        self, fake_entry_points, mutate, fragment
    ):
        """Each contract violation skips the driver and warns naming the problem."""
        attrs = {**GoodDriver.__dict__}
        bad = type("Bad", (), {**{k: v for k, v in attrs.items()}, **mutate})
        fake_entry_points(("bad", bad))
        with pytest.warns(RuntimeWarning, match=fragment):
            assert raster_drivers() == {}

    def test_a_valid_preempts_passes_discovery(self, fake_entry_points):
        """A driver declaring well-formed ``preempts`` loads without warning."""
        good = type(
            "WithPreempts", (), {**GoodDriver.__dict__, "preempts": (".nat", "ee")}
        )
        fake_entry_points(("wp", good))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = raster_drivers()
        assert "fci" in loaded, (
            f"a valid-preempts driver should load, got {sorted(loaded)}"
        )

    def test_missing_attribute_is_skipped(self, fake_entry_points):
        """A class missing a required claim is skipped with a warning."""

        class Nameless:
            api_version = RASTER_DRIVER_API_VERSION

        fake_entry_points(("nameless", Nameless))
        with pytest.warns(RuntimeWarning, match="missing attribute"):
            assert raster_drivers() == {}

    def test_read_collection_capability_requires_the_method(self, fake_entry_points):
        """Declaring read_collection without implementing it is rejected."""

        class NoMethod(GoodDriver):
            capabilities = frozenset({"read", "read_collection"})

        fake_entry_points(("nomethod", NoMethod))
        with pytest.warns(RuntimeWarning, match="does not implement"):
            assert raster_drivers() == {}

    def test_every_capability_token_is_accepted(self, fake_entry_points):
        """A driver may declare every known token when it implements what they promise."""

        class Everything(GoodDriver):
            name = "everything"
            capabilities = frozenset(CAPABILITIES)

            def read_collection(self, source, *, time=None, **kw):
                return None

        fake_entry_points(("everything", Everything))
        assert sorted(raster_drivers()) == ["everything"]

    def test_driver_failing_to_load_is_skipped_with_a_warning(self, monkeypatch):
        """An entry point whose target cannot be resolved costs itself, not the registry."""
        monkeypatch.setattr(_drivers, "_good_target", GoodDriver, raising=False)
        entries = (
            EntryPoint(
                name="broken",
                value="pyramids.dataset._drivers:_missing_target_xyz",
                group=ENTRY_POINT_GROUP,
            ),
            EntryPoint(
                name="fci",
                value="pyramids.dataset._drivers:_good_target",
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
            assert sorted(raster_drivers()) == ["fci"]
        monkeypatch.delattr(_drivers, "_good_target", raising=False)

    def test_duplicate_name_keeps_the_first(self, fake_entry_points):
        """A name registered twice keeps the first and warns about the second."""

        class Other(GoodDriver):
            long_name = "Other"

        fake_entry_points(("a", GoodDriver), ("b", Other))
        with pytest.warns(RuntimeWarning, match="registered twice"):
            drivers = raster_drivers()
        assert list(drivers) == ["fci"]
        assert drivers["fci"].long_name == "EUMETSAT FCI"


class TestExtraGating:
    """A driver is skipped when its declared optional dependency is absent."""

    def test_missing_extra_is_skipped_silently(self, fake_entry_points):
        """An absent extra skips the driver without importing its backend or warning."""

        class NeedsBackend(GoodDriver):
            extra = "no_such_backend_xyz"

        fake_entry_points(("needs", NeedsBackend))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert raster_drivers() == {}

    def test_present_extra_is_loaded(self, fake_entry_points):
        """An importable extra (here the stdlib ``os``) does not gate the driver out."""

        class HasBackend(GoodDriver):
            extra = "os"

        fake_entry_points(("has", HasBackend))
        assert sorted(raster_drivers()) == ["fci"]

    def test_unresolvable_extra_is_treated_as_missing(self, fake_entry_points):
        """An extra naming a missing parent package (find_spec raises) gates the driver out."""

        class NeedsChild(GoodDriver):
            extra = "no_such_parent_xyz.child"

        fake_entry_points(("needs", NeedsChild))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert raster_drivers() == {}


class TestHelpers:
    """The standalone resolver and message helpers."""

    def test_resolve_driver_returns_the_class(self, registered):
        """resolve_driver returns the registered class for a known name."""
        registered(GoodDriver)
        assert _drivers.resolve_driver("fci") is GoodDriver

    def test_resolve_driver_unknown_raises(self, registered):
        """resolve_driver raises for an unregistered name."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            _drivers.resolve_driver("nope")

    def test_select_driver_with_an_explicit_name_delegates_to_resolve(self, registered):
        """select_driver with a name returns that driver, bypassing the can_open scan."""
        registered(GoodDriver)
        assert select_driver("anything", "fci") is GoodDriver


class TestDiscovery:
    """Discovery order, the copy contract, and the lookup helpers."""

    def test_discovery_preserves_entry_point_order(self, fake_entry_points):
        """The registry keeps drivers in the order discovery found them."""

        class Second(GoodDriver):
            name = "second"
            suffixes = (".two",)

        fake_entry_points(("fci", GoodDriver), ("second", Second))
        assert list(raster_drivers()) == ["fci", "second"]

    def test_raster_drivers_returns_a_copy(self, registered):
        """Mutating the returned mapping does not reach the registry."""
        registered(GoodDriver)
        drivers = raster_drivers()
        drivers.clear()
        assert sorted(raster_drivers()) == ["fci"]

    def test_reentrant_discovery_during_load_registers_the_driver(self, monkeypatch):
        """A plugin that consults the registry during its own load is still registered."""

        class ReentrantEntryPoint:
            name = "fci"
            value = "your_package:FciDriver"

            def load(self):
                # A plugin checking for a companion driver during its own import.
                raster_drivers()
                return GoodDriver

        monkeypatch.setattr(_drivers, "_CACHE", None)
        monkeypatch.setattr(
            _drivers,
            "entry_points",
            lambda *, group=None: (
                (ReentrantEntryPoint(),) if group == ENTRY_POINT_GROUP else ()
            ),
        )
        assert list(raster_drivers()) == ["fci"], (
            "re-entrant load must not drop the driver"
        )

    def test_a_concurrent_publish_keeps_the_first_result(self, monkeypatch):
        """If another thread publishes between discovery and the lock, the first result wins."""
        first = {"fci": GoodDriver}

        def _discover_then_race():
            # Simulate a second thread finishing discovery and publishing first.
            _drivers._CACHE = first
            return {"other": GoodDriver}

        monkeypatch.setattr(_drivers, "_CACHE", None)
        monkeypatch.setattr(_drivers, "_discover", _discover_then_race)
        result = raster_drivers()
        assert list(result) == ["fci"], "the already-published mapping must win"
        assert result is not first, "callers get a copy, not the live cache"

    def test_drivers_for_suffix_is_case_and_dot_insensitive(self, registered):
        """A suffix resolves with or without its dot and in any case."""
        registered(GoodDriver)
        assert [d.name for d in drivers_for_suffix(".nat")] == ["fci"]
        assert [d.name for d in drivers_for_suffix("NAT")] == ["fci"]
        assert drivers_for_suffix(".tif") == []

    def test_drivers_for_scheme_strips_separator(self, registered):
        """A scheme resolves with or without a trailing ``://``."""

        class EE(GoodDriver):
            name = "earthengine"
            suffixes = ()
            schemes = ("ee",)

        registered(EE)
        assert [d.name for d in drivers_for_scheme("ee")] == ["earthengine"]
        assert [d.name for d in drivers_for_scheme("EE://")] == ["earthengine"]
        assert drivers_for_scheme("s3") == []

    def test_describe_drivers_is_serialisable(self, registered):
        """describe_drivers returns a plain-dict summary per driver."""
        registered(GoodDriver)
        assert describe_drivers() == [
            {
                "name": "fci",
                "long_name": "EUMETSAT FCI",
                "suffixes": [".nat"],
                "schemes": [],
                "capabilities": ["read"],
            }
        ]


class TestCacheColdness:
    """A bare install does at most one cached entry-point scan on a normal GDAL read."""

    def test_bare_registry_is_empty(self):
        """With no plugin installed the registry is empty."""
        assert raster_drivers() == {}

    def test_a_read_with_a_suffix_discovers_at_most_once(
        self, tmp_path, fake_entry_points
    ):
        """A readable raster *with a suffix* opens through GDAL; the pre-empt check does one scan.

        Pre-emption (#1344) consults the registry before GDAL for a source with a suffix/scheme, so
        such a read now warms the discovery cache exactly once — to an empty mapping on a bare
        install (no entry points), cached for every later read. It never pays per-read discovery,
        and with nothing registered nothing pre-empts, so GDAL still opens the file.

        ``fake_entry_points()`` stubs discovery to an empty entry-point set, so the ``== {}`` check
        is independent of any raster-driver plugin that may be co-installed in the test environment.
        """
        fake_entry_points()
        path = tmp_path / "x.tif"
        _dataset().to_file(str(path))
        clear_driver_cache()
        opened = Dataset.read_file(str(path))
        assert opened.band_count == 1
        assert _drivers._CACHE == {}, (
            "a bare install discovers once to an empty, cached registry"
        )

    def test_a_tokenless_source_never_touches_the_registry(self, tmp_path):
        """A source with no suffix or scheme short-circuits before the scan, leaving the cache cold.

        ``select_preempting_driver`` returns early when the source yields no token, so
        ``raster_drivers()`` is never called and ``_CACHE`` stays ``None`` — the exception to the
        one-scan rule above.
        """
        src = tmp_path / "x.tif"
        _dataset().to_file(str(src))
        tokenless = tmp_path / "x"  # no suffix, no scheme
        tokenless.write_bytes(
            src.read_bytes()
        )  # GDAL opens a GeoTIFF by content, not extension
        clear_driver_cache()
        opened = Dataset.read_file(str(tokenless))
        assert opened.band_count == 1
        assert _drivers._CACHE is None, (
            "a tokenless read must not warm the discovery cache"
        )


class TestDispatch:
    """``Dataset.read_file`` routing: driver= override, GDAL, then plugin fallback."""

    def test_explicit_driver_bypasses_gdal(self, registered):
        """``driver=`` routes straight to the named plugin without touching GDAL."""
        registered(GoodDriver)
        opened = Dataset.read_file("whatever.nat", driver="fci")
        assert opened.rows == 2

    def test_unknown_driver_raises(self, registered):
        """An unregistered ``driver=`` name is a clear error."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            Dataset.read_file("x.nat", driver="nope")

    def test_list_and_tuple_open_options_reach_the_plugin(self, registered):
        """A list or tuple open_options is passed to the plugin unchanged, not dropped."""
        seen = {}

        class Rec(GoodDriver):
            def read(
                self,
                source,
                *,
                bbox=None,
                bands=None,
                crs=None,
                options=None,
                auth=None,
            ):
                seen["options"] = options
                return _dataset()

        registered(Rec)
        Dataset.read_file("x.nat", driver="fci", open_options=["L1B_MODE=DATASTRIP"])
        assert seen["options"] == ["L1B_MODE=DATASTRIP"]
        Dataset.read_file("x.nat", driver="fci", open_options=("A=B",))
        assert seen["options"] == ("A=B",)

    def test_plugin_returning_non_dataset_raises_typeerror(self, registered):
        """A driver whose read returns a non-Dataset is rejected with a clear TypeError."""

        class Bad(GoodDriver):
            def read(
                self,
                source,
                *,
                bbox=None,
                bands=None,
                crs=None,
                options=None,
                auth=None,
            ):
                return "NOT-A-DATASET"

        registered(Bad)
        with pytest.raises(TypeError, match="not a Dataset"):
            Dataset.read_file("x.nat", driver="fci")

    def test_plugin_dispatch_warns_about_ignored_gdal_only_args(self, registered):
        """Selecting a plugin while passing GDAL-only args warns that they are ignored."""
        registered(GoodDriver)
        with pytest.warns(UserWarning, match="ignored"):
            Dataset.read_file("x.nat", driver="fci", vsi="zip")

    def test_unknown_driver_does_not_warn_before_raising(self, registered):
        """An unknown driver= raises DriverNotExistError without a spurious ignored-args warning."""
        registered(GoodDriver)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DriverNotExistError):
                Dataset.read_file("x.nat", driver="nope", vsi="zip")

    def test_gdal_failure_falls_back_to_a_claiming_driver(self, registered):
        """A source GDAL cannot open is handed to the first driver whose can_open claims it."""

        class EE(GoodDriver):
            name = "earthengine"
            suffixes = ()
            schemes = ("ee",)

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("ee://")

        registered(EE)
        opened = Dataset.read_file("ee://MODIS/006/MOD13Q1")
        assert opened.rows == 2

    def test_gdal_failure_no_claimant_reraises_gdal_error(self, registered):
        """A source no driver claims re-raises GDAL's own error, even with drivers installed."""
        registered(GoodDriver)  # claims only .nat, not this path
        with pytest.raises(FileNotFoundError):
            Dataset.read_file("definitely-not-a-raster.bogus")

    def test_missing_file_keeps_filenotfounderror_regardless_of_plugins(
        self, registered
    ):
        """A missing path stays FileNotFoundError whether or not a plugin is installed."""
        registered(GoodDriver)
        with pytest.raises(FileNotFoundError):
            Dataset.read_file("definitely-not-a-raster.bogus")
        clear_driver_cache()
        with pytest.raises(FileNotFoundError):
            Dataset.read_file("definitely-not-a-raster.bogus")

    def test_the_scan_does_not_instantiate_a_non_claiming_driver(self, registered):
        """can_open is a classmethod; only the chosen driver is ever constructed."""
        built = {"nonclaim": 0, "claim": 0}

        class NonClaim(GoodDriver):
            name = "nonclaim"
            schemes = ("zzz",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                return False

            def __init__(self):
                built["nonclaim"] += 1

        class Claim(GoodDriver):
            name = "claim"
            schemes = ("ee",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("ee://")

            def __init__(self):
                built["claim"] += 1

        registered(NonClaim, Claim)
        Dataset.read_file("ee://asset")
        assert built == {"nonclaim": 0, "claim": 1}

    def test_a_can_open_that_raises_is_skipped_with_a_warning(self, registered):
        """A driver whose can_open raises is skipped so one bad driver cannot block the scan."""

        class Explodes(GoodDriver):
            name = "explodes"
            schemes = ("ee",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                raise RuntimeError("bad can_open")

        # A plain missing path raises FileNotFoundError on every platform (unlike an ``ee://``
        # scheme, whose GDAL error type differs across OSes); can_open is still scanned on the
        # GDAL failure, raises, is skipped with a warning, and the original error propagates.
        # The warning is captured manually so pytest.raises wraps exactly the one call.
        registered(Explodes)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(FileNotFoundError):
                Dataset.read_file("definitely-not-a-raster.bogus")
        assert any(
            "raised from can_open" in str(record.message) for record in caught
        ), "expected the skipped-can_open RuntimeWarning"

    def test_select_driver_returns_none_on_a_bare_install(self):
        """With nothing registered, the scan claims nothing and the caller falls back to GDAL."""
        assert select_driver("anything.nat", None) is None


class _PreemptDriver(GoodDriver):
    """A driver that pre-empts GDAL for ``.tif`` and returns a recognisable 2x2 stub."""

    name = "preempt"
    suffixes = (".tif",)
    schemes = ()
    preempts = (".tif",)

    @classmethod
    def can_open(cls, source):
        return str(source).lower().endswith(".tif")


class TestPreemptTokens:
    """The shared ``extract_tokens`` suffix/scheme extraction that gates a ``preempts`` match."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("/data/S2_X.SAFE", {".safe"}),
            ("/data/S2_X.SAFE/", {".safe"}),
            (r"C:\data\S2_X.SAFE", {".safe"}),
            ("ee://MODIS/006/MOD13Q1", {"ee"}),
            ("s3://bucket/key.tif", {"s3", ".tif"}),
            ("plain.tif", {".tif"}),
            ("no-suffix-no-scheme", set()),
            ("http://host/path/x.jp2?token=abc", {"http", ".jp2"}),
            ("http://host/x.tif#frag", {"http", ".tif"}),
            ("/data/.bashrc", set()),
            ("foo.tar.gz", {".gz"}),
            ("/data/archive.", set()),
        ],
    )
    def test_extracts_scheme_and_suffix(self, source, expected):
        """Scheme (no ``://``) and dot-prefixed suffix are both pulled, lowercased.

        Args:
            source: The source string.
            expected: The expected token set.
        """
        assert extract_tokens(source) == expected


def _real_geotiff(tmp_path) -> str:
    """Write a genuine GDAL-openable 3x3 GeoTIFF and return its path.

    A 3x3 so GDAL's result (``rows == 3``) is distinguishable from the driver stub's 2x2
    (``rows == 2``).

    Args:
        tmp_path: pytest temp dir.

    Returns:
        str: The path to the written GeoTIFF.
    """
    real = tmp_path / "real.tif"
    Dataset.from_array(np.ones((3, 3), dtype="float32"), geo_ref=GEO).to_file(str(real))
    return str(real)


class TestPreemption:
    """``read_file`` lets a ``preempts`` driver win *before* GDAL for a source it claims (#1344)."""

    def test_preempting_driver_wins_before_gdal(self, registered, tmp_path):
        """A driver that pre-empts ``.tif`` reads a real GeoTIFF instead of GDAL.

        Test scenario:
            The file is a genuine GDAL-openable 3x3 GeoTIFF, so without pre-emption GDAL opens it
            and returns ``rows == 3``. The pre-empting driver's stub returns 2x2, so ``rows == 2``
            proves the driver won before GDAL.
        """
        real = _real_geotiff(tmp_path)
        registered(_PreemptDriver)
        opened = Dataset.read_file(real)
        assert opened.rows == 2, (
            "the pre-empting driver's 2x2 stub should be returned, not GDAL's"
        )

    def test_a_driver_without_preempts_does_not_preempt(self, registered, tmp_path):
        """A driver declaring no ``preempts`` leaves a GDAL-openable source to GDAL (opt-in)."""
        real = _real_geotiff(tmp_path)
        registered(GoodDriver)  # .nat only, no preempts
        opened = Dataset.read_file(real)
        assert opened.rows == 3, (
            "GDAL should still open the 3x3 .tif when no driver pre-empts it"
        )

    def test_preempt_requires_can_open_confirmation(self, registered, tmp_path):
        """A ``preempts`` match whose ``can_open`` returns False does not pre-empt; GDAL opens it."""

        class Coarse(_PreemptDriver):
            name = "coarse"

            @classmethod
            def can_open(cls, source):
                return False

        real = _real_geotiff(tmp_path)
        registered(Coarse)
        assert select_preempting_driver(real) is None
        opened = Dataset.read_file(real)
        assert opened.rows == 3, (
            "GDAL opens the file because can_open declined the pre-emption"
        )

    def test_select_preempting_driver_matches_suffix_and_scheme(self, registered):
        """The selector claims a source whose suffix or scheme is in ``preempts`` and can_open."""

        class SchemePreempt(GoodDriver):
            name = "scheme"
            suffixes = ()
            schemes = ("ee",)
            preempts = ("ee",)

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("ee://")

        registered(_PreemptDriver, SchemePreempt)
        assert select_preempting_driver("x.tif") is _PreemptDriver
        assert select_preempting_driver("ee://asset") is SchemePreempt
        assert select_preempting_driver("x.jp2") is None

    def test_no_preempts_driver_is_not_consulted(self, registered):
        """A driver with no ``preempts`` is skipped by the pre-empt scan entirely."""
        registered(GoodDriver)  # claims .nat via can_open, but declares no preempts
        assert select_preempting_driver("x.nat") is None

    def test_a_source_with_no_suffix_or_scheme_never_preempts(self, registered):
        """A source yielding no token short-circuits to ``None`` without scanning drivers."""
        registered(_PreemptDriver)
        assert select_preempting_driver("no-suffix-no-scheme") is None

    def test_a_raising_can_open_is_skipped_with_a_warning(self, registered):
        """A ``preempts`` driver whose ``can_open`` raises is warned about, not propagated."""

        class Boom(_PreemptDriver):
            name = "boom"

            @classmethod
            def can_open(cls, source):
                raise RuntimeError("kaboom")

        registered(Boom)
        with pytest.warns(RuntimeWarning, match="raised from can_open"):
            assert select_preempting_driver("x.tif") is None

    @pytest.mark.parametrize(
        "bad, matches",
        [
            (True, False),
            (5, False),
            (object(), False),
            (".tif", False),
            ({".tif": 1}, False),
            ([".tif", 123], True),
            (frozenset({".tif"}), True),
        ],
    )
    def test_malformed_preempts_is_tolerated_not_raised(self, registered, bad, matches):
        """A non-iterable, bare-string, or junk-member ``preempts`` never raises from the scan (H1).

        Args:
            bad: A malformed ``preempts`` value injected onto a driver.
            matches: Whether a usable ``.tif`` token survives (so the driver still claims ``x.tif``).

        Test scenario:
            Without the guard, ``preempts=True`` iterates a bool and ``[".tif", 123]`` calls
            ``123.lower()`` — both raise out of the scan and abort the read. The guard filters to
            valid string tokens, so a bare string / non-iterable yields no match while a list with
            one good member still matches on it.
        """

        class Malformed(_PreemptDriver):
            name = "malformed"
            preempts = bad

        registered(Malformed)
        result = select_preempting_driver("x.tif")
        assert (result is Malformed) == matches, (
            f"{bad!r} should match={matches}, got {result}"
        )

    def test_a_preempts_whose_bool_or_iter_raises_is_tolerated(self, registered):
        """A ``preempts`` whose ``__bool__`` / ``__iter__`` raises (e.g. a numpy array) is ignored (L1).

        The guard rejects anything that is not a tuple/list/set/frozenset *before* truth-testing or
        iterating it, so such a value never reaches ``bool()`` or ``for`` and cannot propagate.
        """

        class Weird:
            def __bool__(self):
                raise ValueError("truth value is ambiguous")

            def __iter__(self):
                raise ValueError("iteration is ambiguous")

        class Driver(_PreemptDriver):
            name = "weird"
            preempts = Weird()

        registered(Driver)
        assert select_preempting_driver("x.tif") is None

    def test_a_malformed_preempts_driver_does_not_break_a_gdal_read(
        self, registered, tmp_path
    ):
        """A driver with a non-iterable ``preempts`` installed still lets a plain GeoTIFF open via GDAL."""

        class Malformed(_PreemptDriver):
            name = "malformed"
            preempts = True

        real = _real_geotiff(tmp_path)
        registered(Malformed)
        opened = Dataset.read_file(real)
        assert opened.rows == 3, (
            "GDAL must still open the .tif despite a malformed-preempts driver"
        )

    def test_explicit_vsi_skips_the_preempt_scan(self, tmp_path, monkeypatch):
        """An explicit ``vsi=`` (archive intent) suppresses the pre-empt scan, like ``driver=`` (L3)."""
        import pyramids.dataset.dataset as ds_mod

        seen: list[str] = []
        monkeypatch.setattr(
            ds_mod,
            "select_preempting_driver",
            lambda source: seen.append(source) or None,
        )
        real = _real_geotiff(tmp_path)
        Dataset.read_file(real)
        assert seen == [real], "a normal read consults the pre-empt scan"
        seen.clear()
        try:
            Dataset.read_file(real, vsi="zip")
        except Exception:
            pass  # a .tif is not a zip; the point is only that the scan was skipped
        assert seen == [], "an explicit vsi= must skip the pre-empt scan"

    def test_two_preempting_drivers_first_registered_wins(self, registered):
        """When two drivers pre-empt the same source, the first in discovery order wins (L4)."""

        class First(_PreemptDriver):
            name = "first"

        class Second(_PreemptDriver):
            name = "second"

        registered(First, Second)
        assert select_preempting_driver("x.tif") is First


class TestConformanceKit:
    """The reusable conformance kit a downstream plugin runs against its driver."""

    def test_a_conforming_driver_has_no_problems(self):
        """A well-formed class passes both static and behavioural checks."""
        assert check_raster_driver(GoodDriver) == []
        assert_raster_driver_conforms(GoodDriver)

    def test_a_static_violation_is_reported_first(self):
        """A contract violation surfaces as the first listed problem."""

        class Bad(GoodDriver):
            capabilities = frozenset()

        assert "required 'read'" in check_raster_driver(Bad)[0]
        with pytest.raises(AssertionError, match="does not conform"):
            assert_raster_driver_conforms(Bad)

    def test_a_can_open_that_raises_is_reported(self):
        """can_open must never raise; a kit probe catches one that does."""

        class Raises(GoodDriver):
            @classmethod
            def can_open(cls, source):
                raise ValueError("no")

        problems = check_raster_driver(Raises)
        assert any("must never raise" in problem for problem in problems)

    def test_a_driver_needing_constructor_args_is_reported(self):
        """A driver read_file cannot build with no arguments is caught by the kit."""

        class NeedsArg(GoodDriver):
            def __init__(self, required):
                self.required = required

        problems = check_raster_driver(NeedsArg)
        assert any("no arguments" in problem for problem in problems)

    def test_an_instance_shadowing_read_with_a_non_callable_is_reported(self):
        """A driver whose instance shadows read with a non-callable is caught."""

        class ShadowsRead(GoodDriver):
            def __init__(self):
                self.read = "not callable"

        problems = check_raster_driver(ShadowsRead)
        assert any("no callable read" in problem for problem in problems)

    def test_a_non_bool_can_open_is_reported(self):
        """can_open must return a bool; a kit probe catches a truthy non-bool."""

        class Truthy(GoodDriver):
            @classmethod
            def can_open(cls, source):
                return "yes"

        problems = check_raster_driver(Truthy)
        assert any("must return a bool" in problem for problem in problems)
