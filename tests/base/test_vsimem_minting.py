"""One way of minting an in-memory artefact path, and it tracks itself.

Four call sites minted `/vsimem` paths by hand -- the COG writer, the
orthorectify DEM, the dtype probe and the WCS reader -- with four naming
conventions between them, so a `/vsimem` listing during debugging gave no
consistent way to tell whose artefact was whose. None of the four registered
with the exit sweep, so anything they failed to unlink themselves stayed in
memory for the life of the process, while the STAC reader's artefacts were
reclaimed.

Minting and tracking in one call is what makes the registration hard to forget.
It is only half the contract, though: the registry is process-wide and swept at
exit, so a site that unlinks its own artefact and says nothing leaves an entry
behind for the life of the process. Twenty-five `to_cog_bytes` calls left
twenty-five dead entries; a service calling it per request accumulates one per
request forever, and the exit sweep walks the whole list.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base import _artifacts
from pyramids.base._artifacts import mint_vsimem, unregister_vsimem
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset._wcs import _open_getcoverage_bytes
from pyramids.dataset.ops.io import _driver_preserves_dtype

pytestmark = pytest.mark.core


def _small_raster() -> Dataset:
    """A 4x4 float32 raster, the cheapest thing every site under test accepts.

    Returns:
        Dataset: An in-memory raster with a real georeference.
    """
    return Dataset.from_array(
        np.arange(16, dtype="float32").reshape(4, 4),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def tracked() -> list[str]:
    """The paths minted by a test, unregistered afterwards.

    The registry is process-wide and armed with an `atexit` hook, so a test
    that leaves entries behind changes what a later `cleanup()` unlinks.

    Yields:
        list[str]: Append minted paths here to have them released.
    """
    minted: list[str] = []
    yield minted
    for path in minted:
        unregister_vsimem(path)


class TestTheMintedPath:
    """Shape and uniqueness."""

    def test_the_purpose_leads_the_name(self, tracked):
        """Args: tracked: Fixture releasing what the test mints.

        Test scenario:
            The four hand-rolled spellings put the purpose in different
            places, or nowhere -- one was a bare uuid. Leading with it is what
            makes a `/vsimem` listing readable.
        """
        path = mint_vsimem("cog")
        tracked.append(path)

        assert path.startswith("/vsimem/cog_"), path
        assert path.endswith(".tif"), path

    def test_the_suffix_is_the_callers_choice(self, tracked):
        """Args: tracked: Fixture releasing what the test mints.

        Test scenario:
            The dtype probe wants no extension at all, so GDAL infers the
            format. A helper that always appended `.tif` could not serve it.
        """
        path = mint_vsimem("dtype_probe", "")
        tracked.append(path)

        assert path.startswith("/vsimem/dtype_probe_")
        assert not path.endswith(".tif")

    def test_two_calls_never_collide(self, tracked):
        """The point of minting rather than naming.

        Args:
            tracked: Fixture releasing what the test mints.

        Test scenario:
            Two concurrent COG writes must not share a path; the second would
            overwrite the first's artefact mid-write.
        """
        first, second = mint_vsimem("probe", ""), mint_vsimem("probe", "")
        tracked.extend([first, second])

        assert first != second


class TestTheMintedPathIsTracked:
    """The half every hand-rolled site forgot."""

    def test_it_is_registered_for_the_exit_sweep(self, tracked):
        """Args: tracked: Fixture releasing what the test mints.

        Test scenario:
            `cleanup()` unlinks what the registry holds. A path that was never
            registered survives until the process ends, which is the leak the
            four sites each had their own version of.
        """
        path = mint_vsimem("cog")
        tracked.append(path)

        assert path in _artifacts._VSIMEM_PATHS

    def test_unregistering_removes_it_again(self):
        """A caller that unlinks early can say so.

        Test scenario:
            Leaving a dead entry makes a later `cleanup()` unlink a path that
            no longer exists. That is harmless -- the sweep is best-effort --
            but the registry should not grow for the life of the process.
        """
        path = mint_vsimem("cog")

        unregister_vsimem(path)

        assert path not in _artifacts._VSIMEM_PATHS

    def test_unregistering_an_unknown_path_is_silent(self):
        """It runs from failure handlers, where raising would mask the error.

        Test scenario:
            A build that fails and reclaims its own artefact calls this from
            an `except`. A `ValueError` there would replace the error the
            caller is actually reporting.
        """
        unregister_vsimem("/vsimem/never_minted.tif")


class TestASiteThatUnlinksAlsoUnregisters:
    """The other half: minting registers, so reclaiming has to unregister.

    Every site below frees its own `/vsimem` artefact in a `finally`, which is
    right -- waiting for the exit sweep would hold the memory for the life of
    the process. What none of them did was tell the registry, so the path
    string stayed in `_VSIMEM_PATHS` pointing at a file that no longer exists.
    Bounded per call, unbounded over a process.
    """

    def test_to_cog_bytes_leaves_no_entry_behind(self):
        """The COG encoder, which a tile service calls per request.

        Test scenario:
            Twenty-five encodes left twenty-five dead entries. The files
            themselves were reclaimed, so nothing looked wrong until the
            registry was read.
        """
        ds = _small_raster()
        before = len(_artifacts._VSIMEM_PATHS)

        for _ in range(5):
            ds.to_cog_bytes()

        assert len(_artifacts._VSIMEM_PATHS) == before, (
            f"entries left behind: {_artifacts._VSIMEM_PATHS[before:]}"
        )

    def test_the_dtype_probe_leaves_no_entry_behind(self):
        """The 1x1 write that asks a driver what it does with a dtype.

        Test scenario:
            The probe is `functools.cache`d per (driver, dtype), so it is
            called through `__wrapped__` here to run the body more than once
            and show the growth is per call rather than per cache miss.
        """
        before = len(_artifacts._VSIMEM_PATHS)

        for _ in range(5):
            _driver_preserves_dtype.__wrapped__("GTiff", gdal.GDT_Float32)

        assert len(_artifacts._VSIMEM_PATHS) == before, (
            f"entries left behind: {_artifacts._VSIMEM_PATHS[before:]}"
        )

    def test_the_wcs_response_decoder_leaves_no_entry_behind(self, tmp_path):
        """The `/vsimem` file a GetCoverage body is decoded through.

        Args:
            tmp_path: pytest temp directory, used to make a real GeoTIFF body.

        Test scenario:
            `from_wcs` mints one of these per request, so a long-lived
            harvester grows the registry with every coverage it pulls.
        """
        source = tmp_path / "coverage.tif"
        _small_raster().to_file(str(source))
        payload = source.read_bytes()
        before = len(_artifacts._VSIMEM_PATHS)

        for _ in range(5):
            _open_getcoverage_bytes(payload, "dem")

        assert len(_artifacts._VSIMEM_PATHS) == before, (
            f"entries left behind: {_artifacts._VSIMEM_PATHS[before:]}"
        )
