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
GDAL resolves outside the archive. `_member_at` therefore validates the name
against an allow-list of segments and rebuilds the result from what matched,
rather than checking it and passing the original string through.
"""

from __future__ import annotations

import tarfile
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

        assert "empty name" in str(excinfo.value), (
            f"{member!r} was refused, but not as an empty name: {excinfo.value}"
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
            joined from the segments the allow-list matched, so none of the
            archive's own bytes reach the path GDAL resolves. A
            check-then-pass-through would let the untrusted object itself flow
            on, which is the shape every taint analysis (rightly) flags.
        """
        member = "a/b/1.asc"

        resolved = _member_at("x.zip", [member], 0, "zip")

        assert resolved == member, "a safe nested name must survive intact"
        assert resolved is not member, (
            "the archive's own string was passed through rather than rebuilt"
        )


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
