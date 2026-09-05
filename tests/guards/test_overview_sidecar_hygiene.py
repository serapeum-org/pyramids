"""Guard the generated era5 overview sidecar against dirtying the working tree.

``tests/dataset/spatial/test_overviews.py`` builds an external ``.ovr`` next to the committed
``tests/data/geotiff/era5_land_monthly_averaged.tif`` fixture. On Windows the ``gdal.Dataset`` that
built it can still hold the sidecar open when the sweep in ``tests/conftest.py`` runs, so the
removal is best-effort: it is hygiene, not a precondition, and letting a lost race escape a
*session-scoped* fixture would turn one race into an ERROR on every remaining test.

Best-effort must not mean invisible, and it must not mean a dirty checkout. These two guards pin
the pair of properties that make the swallow honest: the lost race is reported as a warning, and
the sidecar is listed in ``.gitignore`` so a leftover cannot show up as an untracked file.
"""

import warnings
from pathlib import Path

import pytest

from tests.conftest import _unlink_best_effort

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"

# The fixture the overview tests decorate, and the sidecar GDAL writes beside it. Spelled
# repo-relative because that is the spelling `.gitignore` matches on.
ERA5_FIXTURE = "tests/data/geotiff/era5_land_monthly_averaged.tif"
ERA5_SIDECAR = f"{ERA5_FIXTURE}.ovr"


def _gitignore_patterns() -> list[str]:
    """Every non-comment, non-blank line of the repository's ``.gitignore``."""
    lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#")
    ]


class TestALostUnlinkRaceIsReported:
    """The sweep swallows the failure it cannot act on, but says that it did."""

    def test_a_permission_error_is_warned_about_rather_than_swallowed(
        self, tmp_path, monkeypatch
    ):
        """A sidecar another handle is holding open produces a warning, not silence.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            monkeypatch: Fixture used to make ``Path.unlink`` lose the race.

        Test scenario:
            The swallow exists so a session-scoped fixture cannot fail the rest of a
            worker's run. Swallowing in silence, though, is how a leftover artefact goes
            unnoticed: the caller is told nothing and the file is still on disk. The
            helper must report it.
        """
        target = tmp_path / "held_open.tif.ovr"
        target.write_bytes(b"")

        def _refuse(self, missing_ok: bool = False) -> None:
            raise PermissionError(32, "The process cannot access the file")

        monkeypatch.setattr(Path, "unlink", _refuse)

        with pytest.warns(UserWarning, match="overview sidecar"):
            _unlink_best_effort(target)

    def test_a_sidecar_it_can_delete_is_deleted_without_a_warning(self, tmp_path):
        """The ordinary path stays quiet, so the warning means something.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A warning on every sweep would be noise and would train a reader to ignore
            the one that matters, so the uncontended case must remove the file and say
            nothing at all.
        """
        target = tmp_path / "free.tif.ovr"
        target.write_bytes(b"")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _unlink_best_effort(target)

        assert not target.exists(), "an unheld sidecar must actually be removed"


class TestTheGeneratedSidecarCannotDirtyTheTree:
    """A leftover is ignored by git, so a partial run leaves the checkout clean."""

    def test_the_era5_sidecar_is_gitignored(self):
        """``.gitignore`` names the sidecar the overview tests generate.

        Test scenario:
            The sweep is best-effort, so a run that loses the race — an interrupted run,
            a ``-k`` filtered one, or a worker whose sibling holds the handle — leaves the
            ``.ovr`` on disk. The sidecar is generated, never committed, so git must not
            report it as an untracked file when that happens.
        """
        assert ERA5_SIDECAR in _gitignore_patterns(), (
            f"{ERA5_SIDECAR} must be listed in .gitignore; the overview tests generate it "
            "and the sweep that removes it is best-effort"
        )

    def test_the_ignored_path_is_the_one_the_fixture_names(self):
        """The ignore rule tracks the fixture, so renaming the fixture breaks loudly.

        Test scenario:
            A stale ignore rule is worse than none: it keeps passing while the real
            sidecar, now under a different name, dirties the tree again. Pinning the
            rule to the fixture that exists on disk is what couples the two.
        """
        assert (REPO_ROOT / ERA5_FIXTURE).is_file(), (
            f"{ERA5_FIXTURE} is the fixture the ignore rule is written for; if it moved, "
            "move the .gitignore entry with it"
        )
