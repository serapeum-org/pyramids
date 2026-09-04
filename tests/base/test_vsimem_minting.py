"""One way of minting an in-memory artefact path, and it tracks itself.

Four call sites minted `/vsimem` paths by hand -- the COG writer, the
orthorectify DEM, the dtype probe and the WCS reader -- with four naming
conventions between them, so a `/vsimem` listing during debugging gave no
consistent way to tell whose artefact was whose. None of the four registered
with the exit sweep, so anything they failed to unlink themselves stayed in
memory for the life of the process, while the STAC reader's artefacts were
reclaimed.

Minting and tracking in one call is what makes the registration hard to forget.
"""

from __future__ import annotations

import pytest

from pyramids.base import _artifacts
from pyramids.base._artifacts import mint_vsimem, unregister_vsimem

pytestmark = pytest.mark.core


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
