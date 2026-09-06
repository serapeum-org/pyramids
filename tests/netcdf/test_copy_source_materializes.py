"""`CreateCopy` must not read through a reversed multidimensional view.

A NetCDF variable subset is backed by a GDAL `AsClassicDataset` view, and when
the raw Y axis is bottom-up that view is reversed. GDAL cannot service a partial
window through a negative step, so it raises
`arrayStartIdx[...] + (count-1)*arrayStep >= <dim>`.

The windowed read paths already call `_materialize_md_view` first. The three
`CreateCopy` call sites did not, even though `CreateCopy` reads block by block
and hits the same restriction: `to_file` (`netcdf.py`), `copy` (`netcdf.py`) and
`change_no_data_value` (`dataset/engines/bands.py`) failed on exactly the
datasets the read paths handle. All three are exercised here, and each is
checked on the array that came out rather than only on the file existing --
an upside-down copy also produces a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"

# Named, not globbed. A glob evaluated at collection time reports *zero* tests
# when the data directory moves, instead of failing, and a `variable_names[:2]`
# slice silently decides what is covered. These three are the fixtures that
# actually reproduce the defect -- reading their view unmaterialised is what
# raises `arrayStartIdx[...]`, so most Y-reversed fixtures would pin nothing --
# and each differs from its own Y-reversal, which `source_array` re-checks at
# run time so a fixture that stopped doing either fails loudly rather than
# passing vacuously.
FLIPPED = [
    ("cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc", "zeta"),
    ("cf__40v__1d28-2d9-3d3__nc4.nc", "APrioriCovarianceMatrix"),
]
IDS = [f"{Path(fixture).stem}:{name}" for fixture, name in FLIPPED]


def open_flipped(fixture: str, variable: str) -> NetCDF:
    """Open one fixture variable, checking it really is backed by a reversed view.

    Args:
        fixture: File name under `tests/data/netcdf`.
        variable: The variable to narrow to.

    Returns:
        NetCDF: The variable subset, asserted to be Y-reversed.
    """
    view = NetCDF.read_file(str(DATA / fixture)).get_variable(variable)
    assert view._md_y_flipped, (
        f"{fixture}:{variable} is no longer Y-reversed, so it exercises nothing"
    )
    return view


def source_array(fixture: str, variable: str) -> np.ndarray:
    """The variable's array, checked to differ from its own Y-reversal.

    A fixture symmetric about its Y axis compares equal to an upside-down copy,
    which would leave every round-trip assertion below without bite.

    Args:
        fixture: File name under `tests/data/netcdf`.
        variable: The variable to read.

    Returns:
        np.ndarray: The source array, `(rows, cols)` or `(bands, rows, cols)`.
    """
    array = np.asarray(open_flipped(fixture, variable).read_array())
    plane = array if array.ndim == 2 else array[0]
    assert not np.array_equal(plane, plane[::-1, :]), (
        f"{fixture}:{variable} is symmetric about Y, so a flipped copy would pass"
    )
    return array


def written_array(path: Path) -> np.ndarray:
    """Read a written netCDF back as one array, re-stacking its bands.

    `to_file` writes a variable subset out as `Band1 … BandN`, so the read-back
    goes per band and re-stacks, giving something directly comparable with the
    source's `read_array()`.

    Args:
        path: The written file.

    Returns:
        np.ndarray: The stored array in the source's shape.
    """
    container = NetCDF.read_file(str(path))
    try:
        stacked = np.stack(
            [
                np.asarray(container.get_variable(name).read_array())
                for name in container.variable_names
            ]
        )
        result = stacked[0] if stacked.shape[0] == 1 else stacked
    finally:
        container.close()
    return result


@pytest.mark.parametrize(("fixture", "variable"), FLIPPED, ids=IDS)
class TestWritingAReversedView:
    """The three `CreateCopy` sites, driven over a view GDAL cannot window-read."""

    def test_to_file_writes_the_array_the_right_way_up(
        self, fixture: str, variable: str, tmp_path: Path
    ):
        """`to_file` completes, and the file holds the source array unreversed.

        Args:
            fixture: The Y-reversed fixture.
            variable: The variable to write.
            tmp_path: Where the file is written.

        Test scenario:
            Handing `self._raster` to `CreateCopy` instead of
            `self._copy_source` raises `arrayStartIdx[...]` part-way through
            the copy. Asserting only that the file exists would also pass for a
            copy that came out upside down, so the written array is compared
            with the source's.
        """
        expected = source_array(fixture, variable)
        destination = tmp_path / "out.nc"

        open_flipped(fixture, variable).to_file(str(destination))

        assert destination.exists(), f"{destination} was not written"
        assert np.array_equal(written_array(destination), expected), (
            f"{fixture}:{variable} did not survive to_file the right way up"
        )

    def test_copy_returns_the_array_the_right_way_up(
        self, fixture: str, variable: str, tmp_path: Path
    ):
        """`copy` completes, and the copy reads back as the source did.

        Args:
            fixture: The Y-reversed fixture.
            variable: The variable to copy.
            tmp_path: Where the copy is written.

        Test scenario:
            This is the site that raised `IReadBlock failed … arrayStartIdx`.
            The returned object is a usable variable subset, so its own
            `read_array` is what the source array is compared against.
        """
        expected = source_array(fixture, variable)
        destination = tmp_path / "copy.nc"

        copied = open_flipped(fixture, variable).copy(str(destination))

        assert destination.exists(), f"{destination} was not written"
        assert np.array_equal(np.asarray(copied.read_array()), expected), (
            f"{fixture}:{variable} did not survive copy the right way up"
        )

    def test_change_no_data_value_keeps_the_array_the_right_way_up(
        self, fixture: str, variable: str, tmp_path: Path
    ):
        """The third site, which the module docstring named but nothing called.

        Args:
            fixture: The Y-reversed fixture.
            variable: The variable whose sentinel is swapped.
            tmp_path: Where the disk-backed clone is written.

        Test scenario:
            `change_no_data_value` clones the raster with `CreateCopy` before
            streaming the swap into it, so it meets the same restriction. The
            clone is asked for on disk (`path=`), because the in-memory `MEM`
            clone reads the source in one go and never exercises the block
            walk that trips over a reversed view. The swap must land on every
            band and must not disturb the pixels: the fixture holds no
            `-9999.0` cell, so the array has to come back unchanged, and the
            right way up.
        """
        expected = source_array(fixture, variable)
        destination = tmp_path / "sentinel.nc"

        changed = open_flipped(fixture, variable).change_no_data_value(
            -9999.0, path=destination
        )

        assert set(changed.no_data_value) == {-9999.0}, (
            f"the new sentinel did not reach every band: {changed.no_data_value}"
        )
        assert np.array_equal(np.asarray(changed.read_array()), expected), (
            f"{fixture}:{variable} was altered by change_no_data_value"
        )


class TestCopySourceMaterializes:
    """`_copy_source` is the materialising accessor the copy sites read through."""

    def test_it_returns_the_backing_raster(self):
        """For an ordinary raster it is just the raster, materialisation a no-op.

        Test scenario:
            A plain `Dataset` is already window-readable, so the accessor must
            hand back the very raster it was asked about rather than a copy.
        """
        dataset = Dataset.from_array(
            np.ones((2, 2), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )

        assert dataset._copy_source is dataset._raster, (
            "an ordinary raster must be handed to CreateCopy untouched"
        )

    def test_it_swaps_a_reversed_view_for_a_window_readable_raster(self):
        """For a Y-reversed variable it materialises first, and says so by identity.

        Test scenario:
            The whole point of the accessor is that the object `CreateCopy`
            receives is not the reversed `AsClassicDataset` view. Reading it
            therefore has to leave the subset holding a *different* raster,
            which is also the one returned.
        """
        view = open_flipped(*FLIPPED[0])
        before = view._raster

        source = view._copy_source

        assert source is not before, (
            "the reversed view was handed to CreateCopy unmaterialised"
        )
        assert source is view._raster, (
            "the materialised raster must also become the subset's own"
        )
