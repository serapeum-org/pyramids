"""Three archive kinds, one way of naming a member inside them.

Each handler indexed its own member list, and they disagreed on both halves of
the job:

- **Selecting.** The zip handler listed the archive and named a member, so
  `multiple_compressed_files.zip` opened its first file. The tar handler only
  prefixed `/vsitar/` and never listed anything, handing GDAL a *directory* --
  a single-member tar happened to resolve anyway, a multi-member one failed
  with a raw `RuntimeError` quoting the internal `/vsitar/` path, and `file_i`
  was accepted by the caller and silently discarded.
- **Refusing.** Asking for a member past the end let a bare
  `IndexError: list index out of range` escape from the zip handler, while a
  plain gzip -- one stream, no member list -- answered any index at all with
  the stream itself.

So the same request produced an opened file, an `IndexError`, a GDAL
`RuntimeError` or a quiet wrong answer depending only on which archive it was
addressed to.

A member name is also the archive's *own* bytes, so naming a member is where
the tar-slip / zip-slip shape lives: `/vsitar/x.tar/../../etc/passwd` is a path
GDAL resolves outside the archive. `_member_at` therefore refuses any segment
that navigates and rebuilds the result from the segments that cleared, rather
than checking the name and passing the original string through.
"""

from __future__ import annotations

import gzip
import tarfile
import zipfile
from pathlib import Path

import pytest

from pyramids._io import _get_tar_path, _member_at, _only_member_suffix
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

ARCHIVES = Path(__file__).parent / "data" / "virtual-file-system"
KINDS = ["zip", "tar", "gz"]


class TestAMultiMemberArchiveOpensItsFirstMember:
    """The selecting half, which only the zip handler did."""

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_naming_the_archive_alone_reads_the_first_file(self, kind: str):
        """The regression: a multi-member tar raised out of GDAL.

        Args:
            kind: The archive extension under test.

        Test scenario:
            `/vsitar/x.tar` is a directory, not a file. The zip handler had
            always appended a member name; the tar handler had not, so the
            same two-file archive opened as a zip and failed as a tar.
        """
        dataset = Dataset.read_file(str(ARCHIVES / f"multiple_compressed_files.{kind}"))

        assert dataset.band_count == 1, f"{kind}: expected one band"
        assert dataset.rows > 0, f"{kind}: the member read back with no rows"
        assert dataset.columns > 0, f"{kind}: the member read back with no columns"

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_an_index_reaches_a_later_member(self, kind: str):
        """`file_i` was accepted for tar and thrown away.

        Args:
            kind: The archive extension under test.

        Test scenario:
            `_parse_path` passes `file_i` to the zip and gzip handlers and, as
            written, called the tar handler without it -- so no index could
            reach the second file of a tar at all.
        """
        second = Dataset.read_file(
            str(ARCHIVES / f"multiple_compressed_files.{kind}"), file_i=1
        )

        assert second.band_count == 1, f"{kind}: expected one band"

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_the_two_indices_are_not_the_same_file(self, kind: str):
        """Selecting must actually select, not ignore the argument.

        Args:
            kind: The archive extension under test.

        Test scenario:
            Both members of the fixture are the same shape, so a handler that
            ignored `file_i` would pass every assertion above. The resolved
            VSI paths are what has to differ.
        """
        first = Dataset.read_file(
            str(ARCHIVES / f"multiple_compressed_files.{kind}"), file_i=0
        )
        second = Dataset.read_file(
            str(ARCHIVES / f"multiple_compressed_files.{kind}"), file_i=1
        )

        assert first.file_name != second.file_name, (
            f"{kind}: file_i=0 and file_i=1 resolved to {first.file_name}"
        )


class TestAnIndexPastTheEndIsRefusedTheSameWay:
    """The refusing half, which all three did differently."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_single_member_archive_refuses_a_second_member(self, kind: str):
        """One typed error, whatever the archive is.

        Args:
            kind: The archive extension under test.

        Test scenario:
            The zip handler raised `IndexError` from inside a list lookup, the
            gzip handler answered index 3 with its only stream, and the tar
            handler never indexed. A caller could not write one `except`
            around the three.
        """
        with pytest.raises(FileFormatNotSupportedError):
            Dataset.read_file(str(ARCHIVES / f"one_compressed_file.{kind}"), file_i=1)

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_the_message_says_what_is_actually_there(self, kind: str):
        """A refusal the caller can act on names the members.

        Args:
            kind: The archive extension under test.

        Test scenario:
            `IndexError: list index out of range` says nothing about the
            archive. The count and the member list are what tell the caller
            which index to ask for instead.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            Dataset.read_file(str(ARCHIVES / f"one_compressed_file.{kind}"), file_i=9)

        message = str(excinfo.value)
        assert "index 9" in message
        assert "1 file(s)" in message

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_first_member_is_still_the_default(self, kind: str):
        """Refusing a bad index must not refuse a good one.

        Args:
            kind: The archive extension under test.

        Test scenario:
            Index 0 is what every caller that never passes `file_i` asks for,
            so a guard that fired on it would break every archive read.
        """
        dataset = Dataset.read_file(str(ARCHIVES / f"one_compressed_file.{kind}"))

        assert dataset.band_count == 1, f"{kind}: expected one band"


class TestAMemberNameThatWouldLeaveTheArchiveIsRefused:
    """The tar-slip / zip-slip guard, on the name handed to GDAL.

    The member name is read out of the archive, and the archive is the
    untrusted input: `/vsitar/x.tar/../../etc/passwd` is a path GDAL resolves
    *outside* the archive, so a crafted tar or zip could turn a read of what
    looks like a self-contained file into a read of an arbitrary one.
    """

    @pytest.mark.parametrize(
        "member",
        ["../../etc/passwd", "a/../../x", "..", "a/../b.asc", "sub/.."],
        ids=["leading", "mid-path", "bare", "between-names", "trailing"],
    )
    def test_a_parent_segment_anywhere_is_refused(self, member: str):
        """`..` never reaches the VSI path, wherever in the name it sits.

        Args:
            member: An archive member name carrying a `..` segment.

        Test scenario:
            A guard that only inspected the *start* of the name would let
            `a/../../x` through, and one that normalised the name first would
            let the archive decide where the read lands. Every position has to
            be refused, or a crafted archive reads an arbitrary file.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", [member], 0, "tar")

        assert "escapes it" in str(excinfo.value), (
            f"{member!r} was refused, but not as a path leaving the archive: "
            f"{excinfo.value}"
        )

    def test_a_backslash_separated_traversal_is_refused(self):
        """A Windows-authored archive escapes with backslashes, not slashes.

        Test scenario:
            Separators are normalised to `/` before the segments are checked,
            so `..\\..\\etc\\passwd` has to be refused for the same reason its
            POSIX spelling is. Checking the raw string for `../` would miss it
            entirely, and GDAL resolves both on Windows.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", [r"..\..\etc\passwd"], 0, "tar")

        assert "escapes it" in str(excinfo.value), (
            f"a backslash traversal was refused for the wrong reason: {excinfo.value}"
        )

    def test_a_member_of_nothing_but_dots_is_refused(self):
        """`...` is not a name, and the allow-list cannot match one.

        Test scenario:
            The segment pattern excludes an all-dot segment by construction —
            that is *why* `..` cannot match — rather than comparing against the
            literal `".."`. A longer run of dots would slip past such a
            comparison on any filesystem that folds it, so it is refused here.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", ["..."], 0, "tar")

        assert "escapes it" in str(excinfo.value), (
            f"an all-dot member was refused for the wrong reason: {excinfo.value}"
        )

    @pytest.mark.parametrize(
        "member",
        ["/etc/passwd", r"C:\Windows\x", r"\Windows\x", r"\\srv\share\x", "//"],
        ids=[
            "posix-root",
            "windows-drive",
            "windows-rooted",
            "unc-share",
            "bare-root",
        ],
    )
    def test_an_absolute_member_is_refused(self, member: str):
        """An absolute name needs no `..` to leave the archive.

        Args:
            member: An archive member named by an absolute path.

        Test scenario:
            Appending an absolute name to the `/vsitar/` prefix still yields a
            path GDAL resolves away from the archive, and both spellings have
            to be caught on both platforms: a POSIX root, a drive letter, a
            rooted Windows path and a UNC share.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", [member], 0, "tar")

        assert "absolute path" in str(excinfo.value), (
            f"{member!r} was refused, but not as an absolute path: {excinfo.value}"
        )

    @pytest.mark.parametrize(
        "member",
        [r"Z:..\x.tif", "C:../x.tif", r"c:..\..\etc\passwd"],
        ids=["drive-traversal", "drive-traversal-posix", "drive-traversal-deep"],
    )
    def test_a_drive_designator_glued_to_a_traversal_is_refused(self, member: str):
        """The one traversal spelling both other checks miss.

        Args:
            member: A member whose leading drive designator is glued to `..`.

        Test scenario:
            `ntpath.isabs("Z:..\\x")` is False -- there is no separator after
            the colon -- so the absolute check does not fire; and the `..` is
            not a segment of its own once the designator is glued to it, so the
            dots-only deny-list does not see it either. It navigates all the
            same.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.zip", [member], 0, "zip")

        assert "glues the parent traversal" in str(excinfo.value), (
            f"{member!r} was refused, but not by the drive check: {excinfo.value}"
        )

    @pytest.mark.parametrize(
        "member",
        ["C:x.tif", "a:b.tif", "c:data/x.tif", "1:2.asc"],
        ids=["drive-shaped", "single-letter", "nested", "digit"],
    )
    def test_a_colon_name_that_does_not_navigate_still_opens(self, member: str):
        """Refusing every drive-shaped name would be the allow-list mistake again.

        Args:
            member: A member whose name contains a colon.

        Test scenario:
            `a:b.tif` and `C:x.tif` are the same string shape, and on POSIX
            both are ordinary filenames. A member is appended to
            `/vsizip/<archive>/`, where a bare designator names a member rather
            than a drive -- only the `..` navigates, and only that is refused.
            Rejecting real names to catch a threat that is not there is exactly
            what the round-3 allow-list did.
        """
        assert _member_at("x.zip", [member], 0, "zip") == member

    def test_a_colon_that_is_not_a_drive_designator_still_opens(self):
        """The drive check must not turn back into a ban on the colon.

        Test scenario:
            `:` is one of the characters the round-3 ASCII allow-list refused,
            and it is pinned as legitimate elsewhere in this module. The
            traversal check reads a leading designator only as the prefix of
            `..`; a colon anywhere else is an ordinary character in a name.
        """
        resolved = _member_at("x.zip", ["semi:colon.tif"], 0, "zip")

        assert resolved == "semi:colon.tif", (
            f"an interior colon was not preserved: {resolved!r}"
        )

    @pytest.mark.parametrize(
        "member",
        ["", ".", "./", "./."],
        ids=["empty", "dot", "dot-slash", "dot-slash-dot"],
    )
    def test_a_name_that_normalises_to_nothing_is_refused(self, member: str):
        """Dropping the no-op segments must not leave an empty name.

        Args:
            member: A member name whose every segment is skipped.

        Test scenario:
            `.` and empty segments are normalised away, so a name made only of
            them would otherwise resolve to `/vsitar/x.tar/` — the archive
            root, not a file in it.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", [member], 0, "tar")

        assert "normalises to nothing" in str(excinfo.value), (
            f"{member!r} was refused, but not as a name that normalises away: "
            f"{excinfo.value}"
        )

    @pytest.mark.parametrize(
        "member", [".", "./", "./."], ids=["dot", "dot-slash", "dot-slash-dot"]
    )
    def test_the_refusal_quotes_the_name_that_was_actually_there(self, member: str):
        """A name that normalised away was not an empty name, and is not called one.

        Args:
            member: A member name that is not empty but normalises to nothing.

        Test scenario:
            `.` and `./.` are names; they are refused because *normalising* them
            leaves nothing, which is a different fact about the archive than
            "this member has no name". Reporting the second sends a reader
            looking for a zero-length entry that is not there, so the message
            has to quote the name it actually saw.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _member_at("x.tar", [member], 0, "tar")

        assert repr(member) in str(excinfo.value), (
            f"the refusal for {member!r} did not quote it: {excinfo.value}"
        )

    def test_a_real_tar_carrying_a_crafted_member_is_refused_on_read(self, tmp_path):
        """The guard fires through the public read, not only in isolation.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `tarfile` writes whatever `arcname` it is given, so an attacker's
            archive is a two-line fixture. `Dataset.read_file` on it must
            refuse before GDAL is handed the path, otherwise the traversing
            name resolves and the caller reads a file outside the archive.
        """
        payload = tmp_path / "payload.txt"
        payload.write_text("x")
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(payload, arcname="../../evil.txt")
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert names == ["../../evil.txt"], (
            f"the fixture archive does not carry the crafted name: {names}"
        )

        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            Dataset.read_file(str(archive))

        assert "escapes it" in str(excinfo.value), (
            f"the crafted member was refused for the wrong reason: {excinfo.value}"
        )


class TestALegitimateMemberNameStillOpens:
    """The other half of a guard: it must not refuse what real archives hold."""

    @pytest.mark.parametrize(
        "member",
        [
            "1.asc",
            "a/b/1.asc",
            "my file (1).asc",
            "data-2020_v1.tif",
            "x+y#z@w[1]{2}~3%4,5'6.asc",
            ".hidden",
        ],
        ids=[
            "plain",
            "nested",
            "spaces-and-parens",
            "hyphen-and-underscore",
            "punctuation",
            "leading-dot",
        ],
    )
    def test_it_comes_back_unchanged(self, member: str):
        """An allow-list is only safe if it admits the ordinary names.

        Args:
            member: A member name a real archive plausibly carries.

        Test scenario:
            The guard rebuilds the name from matched segments, so a character
            missing from the allow-list would not raise — it would silently
            change the name and open the wrong member, or none. Equality with
            the input is what pins the admitted alphabet.
        """
        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == member, (
            f"a legitimate member name was rewritten: {member!r} -> {resolved!r}"
        )

    @pytest.mark.parametrize(
        ("member", "expected"),
        [("./1.asc", "1.asc"), ("a//b.asc", "a/b.asc"), ("a/./b.asc", "a/b.asc")],
        ids=["leading-dot-slash", "empty-segment", "inner-dot-segment"],
    )
    def test_a_no_op_segment_is_normalised_away_rather_than_refused(
        self, member: str, expected: str
    ):
        """`.` and empty segments are noise, not an escape attempt.

        Args:
            member: A member name carrying a no-op segment.
            expected: The name the guard should resolve it to.

        Test scenario:
            Archivers write `./name` routinely. Refusing it would reject
            ordinary archives, and passing it through would leave a `.`
            segment in the VSI path; it is dropped instead.
        """
        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == expected, (
            f"{member!r} normalised to {resolved!r}, expected {expected!r}"
        )

    def test_a_backslash_separator_becomes_a_forward_slash(self):
        """A Windows-authored archive names its nested members with `\\`.

        Test scenario:
            GDAL's VSI paths are `/`-separated, so the separator is normalised
            before the segments are checked. A member written as `sub\\file`
            has to open, and to open as `sub/file`.
        """
        resolved = _member_at("x.zip", [r"sub\file.asc"], 0, "zip")

        assert resolved == "sub/file.asc", (
            f"a backslash-separated member resolved to {resolved!r}"
        )

    def test_the_returned_name_is_rebuilt_from_the_matched_segments(self):
        """The property that breaks the taint, not merely a passing check.

        Test scenario:
            The guard validates *and rebuilds*: the string handed back is
            joined from the segments the deny-list cleared, so the name GDAL
            resolves is one this function composed. The observable side of that
            is normalisation — a name carrying segments the rebuild drops comes
            back without them, which a check-then-pass-through could not
            produce. Object identity is deliberately *not* asserted: the
            helper's own docstring says the guarantee is about the value, and
            for a single-segment name CPython hands back the input object
            itself, so pinning identity would pin a `str.join` implementation
            detail rather than the contract.
        """
        member = "a/./b//1.asc"

        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == "a/b/1.asc", (
            f"{member!r} was not rebuilt from its segments; got {resolved!r}"
        )

    def test_a_name_with_nothing_to_normalise_survives_intact(self):
        """Rebuilding must not perturb a name that was already clean.

        Test scenario:
            The counterpart to the normalisation above: the rebuild exists to
            drop no-op segments, not to alter the name, so a nested name with
            none of them has to come back byte-for-byte.
        """
        member = "a/b/1.asc"

        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == member, "a safe nested name must survive intact"


class TestTheOnlyStreamOfAMemberLessArchive:
    """`_only_member_suffix`: index 0 appends nothing, anything else refuses."""

    def test_the_first_index_appends_nothing(self):
        """A gzip's single stream *is* the file, so the VSI path ends there.

        Test scenario:
            The caller interpolates the return value straight after the
            archive path. Anything but the empty string would append a member
            name to a `/vsigzip/` path that has no member list to name.
        """
        suffix = _only_member_suffix("x.gz", 0, "gzip")

        assert suffix == "", f"index 0 appended {suffix!r} to the VSI path"

    @pytest.mark.parametrize("file_i", [1, 2, 9])
    def test_any_other_index_is_refused(self, file_i: int):
        """Index 3 of a one-stream archive is a question with no answer.

        Args:
            file_i: An index past the single stream.

        Test scenario:
            The old handler answered every index with the one stream, so a
            caller asking for the second member of a gzip got the first and
            never learned it had asked for something that does not exist.
        """
        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _only_member_suffix("x.gz", file_i, "gzip")

        message = str(excinfo.value)
        assert f"index {file_i}" in message, (
            f"the refusal does not name the index asked for: {message}"
        )
        assert "single stream" in message, (
            f"the refusal does not say why there is no such member: {message}"
        )


class TestATarThatCannotBeListedFallsBackToTheBarePrefix:
    """Listing is best-effort; GDAL owns the "does this exist" message."""

    def test_a_missing_archive_is_left_for_gdal_to_report(self, tmp_path):
        """`tarfile.open` raises `OSError`, which is caught, not propagated.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            A missing file makes `tarfile.open` raise `FileNotFoundError` — an
            `OSError`, not a `tarfile.TarError`. Catching only the latter would
            let it escape from a path *builder*, moving the "no such file"
            message off GDAL and onto every caller of a tar path.
        """
        missing = tmp_path / "not-here.tar"
        assert not missing.exists(), "the fixture path must not exist"

        resolved = _get_tar_path(str(missing))

        assert resolved == f"/vsitar/{missing}", (
            f"a missing tar resolved to {resolved!r} instead of the bare prefix"
        )

    def test_reading_a_missing_archive_still_fails_and_names_the_path(self, tmp_path):
        """Falling back must not turn a missing file into a silent success.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The bare prefix is handed to GDAL precisely so *it* reports the
            missing archive. The read has to fail, and the message has to name
            the path asked for, or the fallback hides the error instead of
            delegating it.
        """
        missing = tmp_path / "not-here.tar"

        with pytest.raises(RuntimeError) as excinfo:
            Dataset.read_file(str(missing))

        assert "/vsitar/" in str(excinfo.value), (
            f"GDAL's failure does not name the vsitar path: {excinfo.value}"
        )

    def test_an_archive_of_only_directories_falls_back_too(self, tmp_path):
        """No *file* members means there is nothing to append.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            Only regular files are candidates, so a tar holding just directory
            entries lists nothing. Indexing that empty list would refuse with
            "holds 0 file(s)" from a path builder; the bare prefix is handed
            over instead, exactly as for an unreadable archive.
        """
        inner = tmp_path / "sub"
        inner.mkdir()
        archive = tmp_path / "dirs-only.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(inner, arcname="sub")

        resolved = _get_tar_path(str(archive))

        assert resolved == f"/vsitar/{archive}", (
            f"a directory-only tar resolved to {resolved!r} instead of the bare prefix"
        )


class TestANegativeIndexIsRefusedByEveryKind:
    """The bound test only guarded one end, so below zero the three diverged."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("file_i", [-1, -2, -9])
    def test_a_negative_index_is_refused(self, kind: str, file_i: int):
        """The regression: `-9` leaked the bare `IndexError`, `-1` opened the last member.

        Args:
            kind: The archive extension under test.
            file_i: A negative index.

        Test scenario:
            `file_i >= len(members)` never fires below zero, so `-9` fell
            through to the list lookup and raised `IndexError: list index out
            of range` -- verbatim the failure this helper was written to
            replace. And `-1` quietly returned the *last* member for a zip or
            tar while the gzip helper refused it, reproducing the three-way
            divergence the consolidation removed, one sign flip away.
        """
        with pytest.raises(FileFormatNotSupportedError):
            Dataset.read_file(
                str(ARCHIVES / f"multiple_compressed_files.{kind}"), file_i=file_i
            )

    @pytest.mark.parametrize("kind", KINDS)
    def test_no_index_leaks_an_index_error(self, kind: str):
        """`IndexError` is the thing that must never escape, at either end.

        Args:
            kind: The archive extension under test.

        Test scenario:
            A caller writing `except FileFormatNotSupportedError` around an
            archive read must not be surprised by a bare `IndexError` from
            inside a list lookup, whichever side of the range they overshot.
        """
        archive = str(ARCHIVES / f"multiple_compressed_files.{kind}")

        for file_i in (-100, -1, 99):
            with pytest.raises(FileFormatNotSupportedError):
                Dataset.read_file(archive, file_i=file_i)


class TestTheMemberListIsFilesOnlyForEveryKind:
    """`file_i` must mean the same thing whatever the container is."""

    @pytest.fixture
    def tree_archives(self, tmp_path) -> dict[str, Path]:
        """A zip and a tar holding the same subdirectory plus one file.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Returns:
            dict[str, Path]: The written archives, keyed by extension.
        """
        payload = tmp_path / "inner.asc"
        payload.write_text(
            "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n"
            "NODATA_value -9999\n1 2\n3 4\n"
        )
        zip_path = tmp_path / "tree.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("subdir/", "")
            archive.write(payload, "subdir/inner.asc")
        tar_path = tmp_path / "tree.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(payload, arcname="subdir/inner.asc")
        return {"zip": zip_path, "tar": tar_path}

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_index_zero_is_the_first_file_not_a_directory(
        self, tree_archives, kind: str
    ):
        """The regression: the same index meant different things per container.

        Args:
            tree_archives: The fixture archives.
            kind: Which one to open.

        Test scenario:
            The zip handler indexed `namelist()`, which includes directory
            entries; the tar handler filtered to `isfile()`. On a tree with a
            subdirectory, `file_i=0` was the directory for a zip -- opening it
            gave a raw GDAL error -- and the first file for a tar.
        """
        dataset = Dataset.read_file(str(tree_archives[kind]))

        assert dataset.band_count == 1, f"{kind}: expected the file, not a directory"

    @pytest.mark.parametrize("kind", ["zip", "tar"])
    def test_only_the_file_is_counted(self, tree_archives, kind: str):
        """A directory entry must not consume an index either.

        Args:
            tree_archives: The fixture archives.
            kind: Which one to open.

        Test scenario:
            One file means one member, so index 1 is past the end. If the
            directory were still counted the zip would have two and this would
            open something.
        """
        with pytest.raises(FileFormatNotSupportedError):
            Dataset.read_file(str(tree_archives[kind]), file_i=1)


class TestNamesTheOldAllowListRefused:
    """The regressions `_io.py` names, pinned so they cannot come back."""

    @pytest.mark.parametrize(
        ("label", "member"),
        [
            ("accented", "H\u00f6he.tif"),
            ("french", "donn\u00e9es.shp"),
            ("cjk", "\u5317\u4eac.tif"),
            ("greek", "\u03a9_flux.tif"),
            ("cyrillic", "\u0440\u0430\u0441\u0442\u0440.tif"),
            ("hive partition", "year=2020/data.tif"),
            ("ampersand", "R&D.tif"),
            ("bang", "file!important.tif"),
            ("dollar", "cost$.tif"),
            ("semicolon", "a;b.tif"),
            ("colon", "semi:colon.tif"),
            ("caret", "tmp^1.tif"),
        ],
    )
    def test_it_comes_back_unchanged(self, label: str, member: str):
        """Args: label: What the name exercises. member: The member name.

        Test scenario:
            The first version of this guard was an allow-list of ASCII
            punctuation, so every one of these was refused as "not a plain
            name" -- a regression dressed as a security fix. Each is named in
            the module comment as a reason the allow-list was wrong, so each
            is pinned here.
        """
        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == member, f"{label}: {member!r} came back as {resolved!r}"


# Large enough that inflating it is unmistakable in the byte count below, small enough
# that building it costs nothing: a run of zeros gzips to a few kilobytes on disk.
_LARGE_MEMBER_BYTES = 8_000_000


class _CountingGzipFile(gzip.GzipFile):
    """A :class:`gzip.GzipFile` that records how far into the decompressed stream it went.

    ``tarfile.open`` does ``from gzip import GzipFile`` inside ``gzopen``, so substituting
    this for :class:`gzip.GzipFile` puts the probe on the exact object the tar reader
    inflates through. Both reading and seeking are recorded: `tarfile` skips a member's
    payload with ``seek``, and on a gzip stream a forward seek is a decompression too --
    counting only ``read`` would report a walk of the whole archive as nearly free.
    """

    reached = 0

    def read(self, size: int = -1) -> bytes:
        """Read as :class:`gzip.GzipFile` does, recording the offset that leaves us at."""
        chunk = super().read(size)
        _CountingGzipFile.reached = max(_CountingGzipFile.reached, self.tell())
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek as :class:`gzip.GzipFile` does, recording how far into the stream it lands."""
        position = super().seek(offset, whence)
        _CountingGzipFile.reached = max(_CountingGzipFile.reached, position)
        return position


def _tar_gz_with_a_large_second_member(tmp_path) -> Path:
    """A two-member ``.tar.gz`` whose second member is large, written to `tmp_path`."""
    first = tmp_path / "1.asc"
    first.write_text("ncols 1\n")
    second = tmp_path / "2.asc"
    second.write_bytes(b"\0" * _LARGE_MEMBER_BYTES)
    archive = tmp_path / "big.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(first, arcname="1.asc")
        tar.add(second, arcname="2.asc")
    return archive


class TestACompressedTarIsNotInflatedToNameItsFirstMember:
    """Listing a `.tar.gz` member must not decompress the archive Python-side first.

    A plain `.tar` is a seek-walk, but a `.tar.gz` has no member index: reaching the *last*
    header means inflating every byte before it. Materialising the whole member table to
    answer `file_i=0` therefore paid a full decompression before GDAL -- which decompresses
    it again -- was even called. Walking the headers and stopping at the member asked for
    keeps the common read at one header.
    """

    def test_naming_the_first_member_reads_only_its_header(self, tmp_path, monkeypatch):
        """Resolving `file_i=0` inflates a header, not the archive.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            monkeypatch: Fixture used to install the counting gzip reader.

        Test scenario:
            The archive's second member is 8 MB. Enumerating every member -- what
            `getmembers()` does -- has to inflate all of it to read the header that
            follows; naming the first member needs only the 512-byte header that opens
            the stream.
        """
        archive = _tar_gz_with_a_large_second_member(tmp_path)
        _CountingGzipFile.reached = 0
        monkeypatch.setattr(gzip, "GzipFile", _CountingGzipFile)

        resolved = _get_tar_path(str(archive))

        assert resolved.endswith("/1.asc"), (
            f"expected the first member, got {resolved!r}"
        )
        assert _CountingGzipFile.reached < 100_000, (
            f"the reader inflated {_CountingGzipFile.reached} bytes to name the first member of a "
            f"{_LARGE_MEMBER_BYTES}-byte archive; the member table was materialised eagerly"
        )

    def test_a_later_member_is_still_reachable(self, tmp_path):
        """Stopping early must not cost the ability to select a later member.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The walk stops at the member asked for, so `file_i=1` has to keep walking
            past the first one. An off-by-one in the stop condition would show up here as
            the wrong member or a spurious refusal.
        """
        archive = _tar_gz_with_a_large_second_member(tmp_path)

        resolved = _get_tar_path(str(archive), file_i=1)

        assert resolved.endswith("/2.asc"), (
            f"expected the second member, got {resolved!r}"
        )

    def test_an_index_past_the_end_still_reports_the_true_count(self, tmp_path):
        """The refusal counts every member, not the truncated walk.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The walk stops early only when it found what it was asked for. An index past
            the end is exactly the case that has to keep going, because the message names
            how many members there actually are -- reporting the length of a partial walk
            would tell the caller a number that is not true of the archive.
        """
        archive = _tar_gz_with_a_large_second_member(tmp_path)

        with pytest.raises(FileFormatNotSupportedError) as excinfo:
            _get_tar_path(str(archive), file_i=7)

        assert "holds 2 file(s)" in str(excinfo.value), (
            f"the refusal did not report both members: {excinfo.value}"
        )
