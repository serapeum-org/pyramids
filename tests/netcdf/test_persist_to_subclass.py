"""`_persist_to` reopens through the class it was called on.

Four `path=` operations shared one persist-and-reopen trailer, and one of the
call sites it replaced reopened with `type(result).read_file` while the shared
version hard-coded `NetCDF.read_file`. That is the deliberate-divergence trap
this branch is about: a subclass asking for `crop(path=...)` got the base class
back, and any behaviour it had overridden was gone from the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
SOURCE = DATA / "cf__5v__1d4-4d1__y-asc.nc"


class TestPersistToUsesTheCallersClass:
    """The reopen is `type(self).read_file`, not `NetCDF.read_file`."""

    def test_the_reopen_goes_through_the_instance_s_own_class(
        self, tmp_path, monkeypatch
    ):
        """Hard-coding `NetCDF.read_file` bypassed any subclass override.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            monkeypatch: Fixture used to observe which classmethod runs.

        Test scenario:
            `read_file` is patched on the *instance's own* class. If the
            trailer called `NetCDF.read_file` directly, the patch on a
            subclass would never fire -- which is exactly how a subclass lost
            its identity through `crop(path=...)`.
        """
        dataset = NetCDF.read_file(str(SOURCE))
        own_class = type(dataset)
        assert own_class is not NetCDF, "fixture no longer exercises a subclass"

        calls: list[str] = []
        original = own_class.read_file

        def recording(cls, *args, **kwargs):
            """Record the call, then defer to the real implementation."""
            calls.append(str(args[0]) if args else "")
            return original(*args, **kwargs)

        monkeypatch.setattr(own_class, "read_file", classmethod(recording))
        destination = tmp_path / "persisted.nc"

        dataset._persist_to(destination)

        assert calls == [str(destination)], (
            "the reopen did not go through the instance's own class"
        )

    def test_no_path_returns_the_same_object_untouched(self, tmp_path):
        """`path=None` is the in-memory case and must not reopen anything.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The trailer is shared by operations that take an optional `path`.
            With none given it has to hand back the very object it was called
            on, still open.
        """
        dataset = NetCDF.read_file(str(SOURCE))

        assert dataset._persist_to(None) is dataset

    def test_the_file_is_written_and_reopenable(self, tmp_path):
        """The other half: a path really does produce a file-backed result.

        Args:
            tmp_path: Fixture supplying a temporary directory.
        """
        dataset = NetCDF.read_file(str(SOURCE))
        destination = tmp_path / "written.nc"

        result = dataset._persist_to(destination)

        assert destination.exists()
        assert result.variable_names
