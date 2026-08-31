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
            ("out.nc", "netCDF"),
            ("out.vrt", "VRT"),
            ("out.img", "HFA"),
        ],
        ids=["geotiff", "netcdf", "vrt", "erdas-imagine"],
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
        [("out.png", "PNG"), ("out.jp2", "JP2OpenJPEG")],
        ids=["png", "jp2"],
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
            name has to fail rather than silently defaulting to GTiff.
        """
        with pytest.raises(DriverNotExistError):
            resolve_output_driver("out")


class TestCatalogAgreesWithGdal:
    """The catalog's `Creation` flag is what the resolver trusts."""

    @pytest.mark.parametrize(
        "extension, creatable",
        [
            ("tif", True),
            ("nc", True),
            ("vrt", True),
            ("img", True),
            ("png", False),
            ("jp2", False),
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
            were wrong before #1075 for exactly this reason.
        """
        catalog = Catalog(raster_driver=True)
        entry = catalog.get_driver(catalog.get_driver_name_by_extension(extension))
        driver = gdal.GetDriverByName(entry["GDAL Name"])
        assert driver is not None, f"{entry['GDAL Name']} missing from this GDAL build"
        real = driver.GetMetadata().get("DCAP_CREATE") == "YES"
        assert real == creatable, (
            f".{extension} -> {entry['GDAL Name']}: GDAL says Create={real}, "
            f"test expects {creatable}"
        )
        assert bool(entry["Creation"]) == real, (
            f".{extension}: catalog says Creation={entry['Creation']}, GDAL says {real}"
        )
