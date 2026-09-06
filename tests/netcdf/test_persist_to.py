"""`NetCDF._persist_to` releases the in-memory handle even when the write fails.

Four spatial-operation paths honoured a `path=` argument the same way: write the
in-memory result out, drop the handle, reopen the file. Three of them closed only
on success, so a failed write -- a full disk, a locked target -- leaked the handle
and, on Windows, left the file locked against the retry.

These tests pin the failure path, which is the half no existing test covered.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import GeoReference, NetCDF

pytestmark = pytest.mark.core


def _raise_disk_full(self, *args, **kwargs) -> None:
    """Stand in for a `to_file` that fails part-way through the write.

    Args:
        self: The receiver, since this replaces an unbound method.
        *args: The destination and any writer options, all ignored.
        **kwargs: Ignored writer options.

    Raises:
        OSError: Always, standing in for a full disk or a locked target.
    """
    raise OSError("disk full")


def _recording_close(calls: list[str], real_close: Callable) -> Callable:
    """Build a `close` that records the call before closing for real.

    Args:
        calls: The list each call appends to.
        real_close: The unbound `close` the wrapper delegates to.

    Returns:
        Callable: A replacement for the unbound `close` method.
    """

    def close(self) -> None:
        """Record the call, then release the handle."""
        calls.append("close")
        real_close(self)

    return close


@pytest.fixture
def container() -> NetCDF:
    """A small in-memory container with one gridded variable."""
    return NetCDF.from_array(
        np.arange(8, dtype="float32").reshape(2, 2, 2),
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        variable_name="v",
    )


class TestPersistToReleasesTheHandle:
    """The in-memory handle is closed whether the write succeeds or raises."""

    def test_a_failed_write_still_closes_the_handle(
        self, container: NetCDF, tmp_path: Path, monkeypatch
    ):
        """When `to_file` raises, the handle is closed and the error propagates.

        Test scenario:
            A write fails on a full disk or a locked target. `to_file` is
            replaced with one that raises and `close` with one that records
            the call, so both halves of the contract are observable: the
            caller still sees the `OSError`, and the handle was released
            before it arrived. Closing only on success -- what three of the
            call sites this helper replaced did -- leaves `closed` empty.
        """
        closed: list[str] = []
        monkeypatch.setattr(type(container), "to_file", _raise_disk_full)
        monkeypatch.setattr(
            type(container), "close", _recording_close(closed, type(container).close)
        )

        with pytest.raises(OSError, match="disk full"):
            container._persist_to(tmp_path / "out.nc")

        assert closed == ["close"], "the handle was not closed on the failure path"

    def test_a_successful_write_returns_a_file_backed_reopen(
        self, container: NetCDF, tmp_path: Path
    ):
        """The happy path writes the file and hands back a reopened dataset.

        Test scenario:
            The point of the helper is that the caller ends up holding the
            file, not the in-memory result: the destination has to exist, the
            returned object has to be a different one, and the variable has to
            have survived the round trip through it.
        """
        destination = tmp_path / "out.nc"

        result = container._persist_to(destination)

        assert destination.exists(), "the result was not written to the destination"
        assert result is not container, "the in-memory result was handed back unwritten"
        assert "v" in result.variable_names, (
            f"the variable did not survive the round trip: {result.variable_names}"
        )

    def test_no_path_returns_the_same_object(self, container: NetCDF):
        """`path=None` is a no-op that keeps the in-memory result.

        Test scenario:
            Every call site passes the user's `path=` straight through, so
            `None` -- the default on all four of them -- must neither write
            nor close: the same live object comes back, and it is still
            readable afterwards.
        """
        result = container._persist_to(None)

        assert result is container, "`path=None` did not return the receiver"
        assert "v" in result.variable_names, "`path=None` closed the in-memory handle"
