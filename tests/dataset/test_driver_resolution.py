"""Tests for :mod:`pyramids.dataset._driver`.

#1075 removed the `driver_type` parameter from the raster constructors, so the
output driver is now derived from the destination path. That makes this module
the single place deciding memory-vs-disk and the on-disk format, and the place
where two very different failures have to stay distinguishable: an extension the
catalog has never heard of, and a catalogued format that cannot be built with
`Create` at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from osgeo import gdal

from pyramids.base._errors import DriverNotExistError, FileFormatNotSupportedError
from pyramids.base._utils import Catalog
from pyramids.dataset._driver import MEMORY_DRIVER, resolve_output_driver

pytestmark = pytest.mark.core


class TestResolveOutputDriverInMemory:
    """No path means an in-memory raster."""

    def test_none_resolves_to_the_memory_driver(self):
        """`None` is the only way to ask for an in-memory raster.

        Test scenario:
            With no destination there is nowhere to write, so MEM is forced.
            This is what makes the old `driver_type` argument redundant: it
            could only agree with `path` or contradict it.
        """
        assert resolve_output_driver(None) == MEMORY_DRIVER, (
            f"expected {MEMORY_DRIVER}, got {resolve_output_driver(None)}"
        )

    def test_the_memory_driver_constant_matches_gdal(self):
        """The constant names a driver GDAL actually provides."""
        assert gdal.GetDriverByName(MEMORY_DRIVER) is not None, (
            f"{MEMORY_DRIVER} is not a GDAL driver"
        )


class TestResolveOutputDriverFromExtension:
    """The extension selects the format."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("out.tif", "GTiff"),
            ("out.tiff", "GTiff"),
            ("out.nc", "netCDF"),
            ("out.nc4", "netCDF"),
            ("out.img", "HFA"),
        ],
        ids=[
            "geotiff",
            "geotiff-tiff-alias",
            "netcdf",
            "netcdf-nc4-alias",
            "erdas-imagine",
        ],
    )
    def test_a_creatable_extension_resolves_to_its_gdal_driver(
        self, filename, expected
    ):
        """Each catalogued, creatable format maps to its GDAL short name.

        Args:
            filename: Destination file name.
            expected: The GDAL driver short name.

        Test scenario:
            The catalog stores pyramids' own keys (`geotiff`, `netcdf`), so a
            resolver that returned those verbatim would fail inside
            `gdal.GetDriverByName`. These assert the GDAL spelling.
        """
        assert resolve_output_driver(filename) == expected, (
            f"{filename} should resolve to {expected}, "
            f"got {resolve_output_driver(filename)}"
        )

    def test_a_path_object_is_accepted_as_well_as_a_string(self):
        """Both `str` and `Path` destinations work.

        Test scenario:
            The constructors annotate `path: str | Path | None`, so refusing
            one of them here would contradict the public signature.
        """
        assert resolve_output_driver(Path("out.tif")) == resolve_output_driver(
            "out.tif"
        )

    @pytest.mark.parametrize("filename", ["out.TIF", "out.Tif", "OUT.TiF"])
    def test_the_extension_match_is_case_insensitive(self, filename):
        """Upper-case extensions resolve like their lower-case spelling.

        Args:
            filename: A destination whose extension differs only in case.

        Test scenario:
            `.TIF` is common on data shipped from Windows tooling; rejecting it
            would be a surprising, purely cosmetic failure.
        """
        assert resolve_output_driver(filename) == "GTiff", (
            f"{filename} should resolve to GTiff, got {resolve_output_driver(filename)}"
        )

    def test_a_full_directory_path_resolves_on_its_suffix_alone(self):
        """Only the suffix matters, not the surrounding path.

        Test scenario:
            Directory components can contain dots; resolution must read the
            suffix rather than scanning the whole string.
        """
        assert resolve_output_driver("/data/v1.2/scenes/out.tif") == "GTiff"


class TestResolveOutputDriverFailures:
    """Two distinct failures that need two distinct errors."""

    @pytest.mark.parametrize("filename", ["out.zzz", "out.unknown", "out.q"])
    def test_an_uncatalogued_extension_raises_driver_not_exist(self, filename):
        """An extension pyramids has never heard of is `DriverNotExistError`.

        Args:
            filename: A destination with an unknown suffix.

        Test scenario:
            This is "I do not know this format", which is a different problem
            from "I know it but cannot write it this way" — see the copy-only
            case below. Collapsing them would lose that distinction.
        """
        with pytest.raises(DriverNotExistError):
            resolve_output_driver(filename)

    @pytest.mark.parametrize(
        "filename, driver",
        [
            ("out.png", "PNG"),
            ("out.vrt", "VRT"),
            ("out.jp2", "JP2OpenJPEG"),
            ("out.j2k", "JP2OpenJPEG"),
            ("out.jpeg", "JPEG"),
            ("out.jpg", "JPEG"),
        ],
        ids=["png", "vrt", "jp2", "jp2-j2k-alias", "jpeg", "jpeg-jpg-alias"],
    )
    def test_a_copy_only_format_raises_before_gdal(self, filename, driver):
        """A catalogued but `CreateCopy`-only format fails up front.

        Args:
            filename: A destination in a write-by-copy-only format.
            driver: The GDAL driver that format maps to.

        Test scenario:
            The constructors build with `Create()`, which PNG and JP2OpenJPEG
            do not support. Reaching GDAL would surface an opaque driver error;
            the message here names the extension, the driver and the reason.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            resolve_output_driver(filename)
        message = str(excinfo.value)
        assert filename.split(".")[-1] in message, (
            f"message must name the extension: {message}"
        )
        assert driver in message, f"message must name the driver: {message}"
        assert "copy" in message.lower(), f"message must give the reason: {message}"

    @pytest.mark.parametrize(
        "filename, driver",
        [("out.png", "PNG"), ("out.jpg", "JPEG"), ("out.jp2", "JP2OpenJPEG")],
    )
    def test_for_copy_accepts_what_create_cannot_build(self, filename, driver):
        """`for_copy=True` lifts the refusal for a copy-only format.

        Args:
            filename: A destination in a write-by-copy-only format.
            driver: The GDAL driver it resolves to.

        Test scenario:
            The `Creation` flag records `Create` support, so callers that write
            with `CreateCopy` -- `translate`, `copy`, `to_terrain_rgb`,
            `from_band_files`' VRT branch -- must not inherit the refusal.
            They could produce these files all along; routing them through the
            default gate briefly rejected a `.png` they can write.
        """
        assert resolve_output_driver(filename, for_copy=True) == driver
        with pytest.raises(FileFormatNotSupportedError):
            resolve_output_driver(filename)

    def test_for_copy_still_refuses_an_unknown_extension(self):
        """`for_copy` lifts the capability gate, not the catalog lookup.

        Test scenario:
            An extension nothing claims is still unresolvable however the
            caller intends to write it.
        """
        with pytest.raises(DriverNotExistError):
            resolve_output_driver("out.zzz", for_copy=True)

    def test_the_two_failures_are_different_exception_types(self):
        """A caller can tell "unknown format" from "cannot Create" by type.

        Test scenario:
            Neither error is a subclass of the other, so `except
            DriverNotExistError` will not silently swallow the copy-only case.
        """
        assert not issubclass(DriverNotExistError, FileFormatNotSupportedError)
        assert not issubclass(FileFormatNotSupportedError, DriverNotExistError)

    @pytest.mark.parametrize("bad_path", [123, 4.5, ["out.tif"], {"path": "out.tif"}])
    def test_a_non_path_argument_raises_type_error(self, bad_path):
        """A destination that is neither `str` nor `Path` is rejected.

        Args:
            bad_path: A value of the wrong type.

        Test scenario:
            Catching this here keeps the failure at the call boundary rather
            than deep inside `Path()` or GDAL.
        """
        with pytest.raises(TypeError, match="string or Path"):
            resolve_output_driver(bad_path)

    def test_a_path_with_no_extension_raises_driver_not_exist(self):
        """An extensionless destination cannot select a format.

        Test scenario:
            With `driver_type` gone the extension is the only signal, so a bare
            name has to fail rather than silently defaulting to GTiff. The
            message must name the path rather than reporting an empty
            extension, which is what the generic catalog lookup would say.
        """
        with pytest.raises(DriverNotExistError) as excinfo:
            resolve_output_driver("out")
        message = str(excinfo.value)
        assert "no file extension" in message, f"unhelpful message: {message}"
        assert "out" in message, f"the message must name the path: {message}"


class TestSiblingExtensionsAgree:
    """Two spellings of one format must resolve alike."""

    @pytest.mark.parametrize(
        "canonical, alias",
        [("out.tif", "out.tiff"), ("out.jp2", "out.j2k"), ("out.jpeg", "out.jpg")],
        ids=["tif-tiff", "jp2-j2k", "jpeg-jpg"],
    )
    def test_an_alias_behaves_exactly_like_its_canonical_spelling(
        self, canonical, alias
    ):
        """An alias resolves to the same driver, or fails the same way.

        Args:
            canonical: The catalogued spelling.
            alias: Another spelling GDAL reports for the same driver.

        Test scenario:
            `.tif` resolved while `.tiff` raised `DriverNotExistError`, and
            `.jpeg` reported "cannot create" while `.jpg` reported "unknown
            format" -- the same file format giving two different answers. The
            premise of deriving the driver from the path is that the extension
            names the format, and for these it did not.
        """

        def outcome(name):
            try:
                result = ("driver", resolve_output_driver(name))
            except Exception as exc:
                result = ("error", type(exc).__name__)
            return result

        assert outcome(canonical) == outcome(alias), (
            f"{canonical} gives {outcome(canonical)} but {alias} gives {outcome(alias)}"
        )

    @pytest.mark.parametrize("empty", ["", None])
    def test_an_empty_extension_is_refused_by_the_catalog(self, empty):
        """The lookup guards its argument, not just the catalog rows.

        Args:
            empty: An argument naming no extension.

        Test scenario:
            The per-row `extension is not None` guard had to go so a row
            carrying only `aliases` could be found. That leaves the argument
            itself unguarded, and the `memory` row holds a null extension --
            so `None == None` would resolve a lookup for nothing to the
            in-memory driver, writing a file that writes nothing.
        """
        catalog = Catalog(raster_driver=True)
        with pytest.raises(DriverNotExistError):
            catalog.get_driver_name_by_extension(empty)

    def test_a_row_with_only_aliases_is_still_reachable(self):
        """An entry carrying aliases but no canonical extension resolves.

        Test scenario:
            This is what removing the per-row guard buys. No shipped row is
            shaped this way yet, so it is constructed here -- the guard's
            absence is a property of the lookup, and a future YAML edit should
            not have to rediscover it.
        """
        catalog = Catalog(raster_driver=True)
        catalog.drivers = dict(catalog.drivers)
        catalog.drivers["synthetic"] = {"GDAL Name": "GTiff", "aliases": ["synthext"]}
        assert catalog.get_driver_name_by_extension("synthext") == "synthetic"

    def test_the_memory_row_is_not_reachable_by_extension(self):
        """The in-memory driver has a real YAML null, not the string "None".

        Test scenario:
            Unquoted `None` in YAML parses as the string `"None"`, which made
            the MEM row reachable by extension lookup. It was harmless only
            because the resolver lowercases the suffix, so `path="x.None"`
            would have resolved to MEM -- a disk path building a raster that
            writes nothing -- had that `.lower()` ever been dropped.
        """
        catalog = Catalog(raster_driver=True)
        assert catalog.get_driver("memory")["extension"] is None, (
            "the memory row's extension must be a real null"
        )
        for spelling in ("None", "none", "NONE"):
            with pytest.raises(DriverNotExistError):
                resolve_output_driver(f"out.{spelling}")


class TestCatalogAgreesWithGdal:
    """The catalog's `Creation` flag is what the resolver trusts."""

    @pytest.mark.parametrize(
        "extension, creatable",
        [
            ("tif", True),
            ("tiff", True),
            ("nc", True),
            ("nc4", True),
            ("img", True),
            ("asc", False),
            ("png", False),
            ("jpeg", False),
            ("jpg", False),
            ("jp2", False),
            ("j2k", False),
        ],
    )
    def test_the_creation_flag_matches_the_real_driver_capability(
        self, extension, creatable
    ):
        """Each flag the resolver reads agrees with GDAL's own metadata.

        Args:
            extension: The catalogued extension.
            creatable: Whether GDAL reports `DCAP_CREATE`.

        Test scenario:
            The flag decides whether a path is accepted or refused, so a wrong
            one either rejects a writable format or lets an unwritable one
            through to an opaque GDAL failure. Two entries (`hdf4`, `adrg`)
            were wrong before #1075 for exactly this reason. Every catalogued,
            resolver-reachable extension is covered -- canonical spellings and
            aliases alike -- so the same drift cannot recur unnoticed in the
            ones left out. `vrt` is the documented exception, asserted
            separately below.
        """
        catalog = Catalog(raster_driver=True)
        entry = catalog.get_driver(catalog.get_driver_name_by_extension(extension))
        driver = gdal.GetDriverByName(entry["GDAL Name"])
        if driver is None:
            # HFA and JP2OpenJPEG are optional GDAL components, absent from some
            # builds (the win_arm64 vcpkg closure, source builds). The resolver
            # never calls GetDriverByName -- the catalog answers first -- so
            # failing here would fail for a reason the code does not depend on.
            pytest.skip(f"{entry['GDAL Name']} is not in this GDAL build")
        real = driver.GetMetadata().get("DCAP_CREATE") == "YES"
        assert real == creatable, (
            f".{extension} -> {entry['GDAL Name']}: GDAL says Create={real}, "
            f"test expects {creatable}"
        )
        assert bool(entry["Creation"]) == real, (
            f".{extension}: catalog says Creation={entry['Creation']}, GDAL says {real}"
        )

    def test_vrt_is_refused_despite_gdal_reporting_create(self):
        """`.vrt` is the one place the flag deliberately disagrees with GDAL.

        Test scenario:
            GDAL reports `DCAP_CREATE=YES` for VRT, but a VRT owns no pixel
            storage: `Create()` succeeds and the `WriteArray` that follows dies
            with "Writing through VRTSourcedRasterBand is not supported". The
            `Creation` flag answers "can the pyramids constructors build it",
            which is the question the resolver actually asks, so it is `No`
            here and the refusal happens up front instead of inside GDAL.
        """
        catalog = Catalog(raster_driver=True)
        entry = catalog.get_driver(catalog.get_driver_name_by_extension("vrt"))
        driver = gdal.GetDriverByName(entry["GDAL Name"])
        assert driver.GetMetadata().get("DCAP_CREATE") == "YES", (
            "the divergence this test documents only exists while GDAL reports "
            "DCAP_CREATE for VRT"
        )
        assert not entry["Creation"], "the catalog must refuse .vrt regardless"
        with pytest.raises(FileFormatNotSupportedError):
            resolve_output_driver("out.vrt")

    def test_every_resolver_reachable_extension_is_covered_above(self):
        """The parametrisation above lists every extension the resolver can reach.

        Test scenario:
            The agreement test is only as good as its list. This derives the
            list from the catalog itself, so adding a driver row without a
            matching case fails here rather than silently going unchecked --
            which is how `asc` and `jpeg` came to be omitted.
        """
        catalog = Catalog(raster_driver=True)
        reachable: set[str] = set()
        for value in catalog.drivers.values():
            if value.get("extension") is not None:
                reachable.add(value["extension"])
            # Aliases are reachable by the resolver too, so a row that gains one
            # must not escape this guard -- which is the whole point of it.
            reachable.update(value.get("aliases") or ())
        covered = {
            "tif",
            "tiff",
            "nc",
            "nc4",
            "vrt",
            "img",
            "asc",
            "png",
            "jpeg",
            "jpg",
            "jp2",
            "j2k",
        }
        assert reachable == covered, (
            f"catalog extensions {sorted(reachable)} != covered {sorted(covered)}; "
            "add the new one to test_the_creation_flag_matches_the_real_driver_capability"
        )
