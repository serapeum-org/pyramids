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
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
