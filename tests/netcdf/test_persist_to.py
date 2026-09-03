"""`NetCDF._persist_to` releases the in-memory handle even when the write fails.

Four spatial-operation paths honoured a `path=` argument the same way: write the
in-memory result out, drop the handle, reopen the file. Three of them closed only
on success, so a failed write -- a full disk, a locked target -- leaked the handle
and, on Windows, left the file locked against the retry.

These tests pin the failure path, which is the half no existing test covered.
"""

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import GeoReference, NetCDF

pytestmark = pytest.mark.core

OPERATIONS = ["to_file"]


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
        """When `to_file` raises, the handle is closed and the error propagates."""
        closed: list[bool] = []
        monkeypatch.setattr(
            type(container),
            "to_file",
            lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        real_close = type(container).close
        monkeypatch.setattr(
            type(container),
            "close",
            lambda self: (closed.append(True), real_close(self))[1],
        )

        with pytest.raises(OSError, match="disk full"):
            container._persist_to(tmp_path / "out.nc")

        assert closed, "the handle was not closed on the failure path"

    def test_a_successful_write_returns_a_file_backed_reopen(
        self, container: NetCDF, tmp_path: Path
    ):
        """The happy path writes the file and hands back a reopened dataset."""
        destination = tmp_path / "out.nc"

        result = container._persist_to(destination)

        assert destination.exists()
        assert result is not container
        assert "v" in result.variable_names

    def test_no_path_returns_the_same_object(self, container: NetCDF):
        """`path=None` is a no-op that keeps the in-memory result."""
        assert container._persist_to(None) is container
