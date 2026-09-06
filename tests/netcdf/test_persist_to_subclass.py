"""`_persist_to` reopens through the class it was called on.

Four `path=` operations shared one persist-and-reopen trailer, and the call
sites it replaced disagreed: one reopened with `type(result).read_file`, the
others hard-coded `NetCDF.read_file`. The shared trailer keeps the `type(self)`
form, so the *lookup* goes through the receiver's class and an override of
`read_file` on a subclass is honoured rather than skipped.

What it does **not** do -- and what these tests therefore do not claim -- is
hand a subclass an instance of itself. `NetCDF.read_file` ends in
`return Container(...)` and ignores `cls`, so persisting a `Variable` yields a
`Container` here exactly as it did before (`netcdf.py:2783-2790` says so). The
divergence this closes is which classmethod runs, not which class comes back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"
SOURCE = DATA / "cf__5v__1d4-4d1__y-asc.nc"


class _RecordingReadFile:
    """A `read_file` stand-in that records the path asked for, then defers to the real one.

    Installed as a plain attribute on the *instance's own* class, so it answers
    a `type(self).read_file(...)` lookup and is invisible to a hard-coded
    `NetCDF.read_file(...)` one.
    """

    def __init__(self, original):
        """Store the real classmethod this recorder defers to.

        Args:
            original: The bound `read_file` classmethod being shadowed.
        """
        self.original = original
        self.calls: list[str] = []

    def __call__(self, *args, **kwargs):
        """Record the requested path, then run the real `read_file`.

        Args:
            *args: Positional arguments; the first is the path.
            **kwargs: Keyword arguments, forwarded untouched.

        Returns:
            NetCDF: Whatever the real `read_file` returns.
        """
        self.calls.append(str(args[0]) if args else "")
        return self.original(*args, **kwargs)


class TestPersistToUsesTheCallersClass:
    """The reopen is `type(self).read_file`, not `NetCDF.read_file`."""

    def test_the_reopen_goes_through_the_instance_s_own_class(
        self, tmp_path, monkeypatch
    ):
        """Hard-coding `NetCDF.read_file` would bypass a subclass override.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            monkeypatch: Fixture used to install the recorder.

        Test scenario:
            `read_file` is replaced on the *instance's own* class, which is a
            `Container` and not `NetCDF` itself. A trailer spelled
            `NetCDF.read_file(...)` resolves on the base class and never sees
            that replacement, so a recorder left empty is exactly what the
            hard-coded form looks like from here.
        """
        dataset = NetCDF.read_file(str(SOURCE))
        own_class = type(dataset)
        assert own_class is not NetCDF, "fixture no longer exercises a subclass"

        recorder = _RecordingReadFile(own_class.read_file)
        monkeypatch.setattr(own_class, "read_file", recorder)
        destination = tmp_path / "persisted.nc"

        dataset._persist_to(destination)

        assert recorder.calls == [str(destination)], (
            f"the reopen did not go through the instance's own class: {recorder.calls}"
        )

    def test_no_path_returns_the_same_object_untouched(self):
        """`path=None` is the in-memory case and must not reopen anything.

        Test scenario:
            The trailer is shared by operations that take an optional `path`.
            With none given it has to hand back the very object it was called
            on, still open.
        """
        dataset = NetCDF.read_file(str(SOURCE))

        assert dataset._persist_to(None) is dataset, (
            "path=None must return the receiver itself, not a reopen of it"
        )

    def test_the_file_is_written_and_reopenable(self, tmp_path):
        """The other half: a path really does produce a file-backed result.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            The reopen has to be of the file just written, so the result must
            enumerate the source's variables rather than come back empty.
        """
        dataset = NetCDF.read_file(str(SOURCE))
        expected = sorted(dataset.variable_names)
        destination = tmp_path / "written.nc"

        result = dataset._persist_to(destination)

        assert destination.exists(), f"{destination} was not written"
        assert sorted(result.variable_names) == expected, (
            f"the reopen lists {sorted(result.variable_names)}, expected {expected}"
        )
