"""`CreateCopy` must not read through a reversed multidimensional view.

A NetCDF variable subset is backed by a GDAL `AsClassicDataset` view, and when
the raw Y axis is bottom-up that view is reversed. GDAL cannot service a partial
window through a negative step, so it raises
`arrayStartIdx[...] + (count-1)*arrayStep >= <dim>`.

The windowed read paths already call `_materialize_md_view` first. The three
`CreateCopy` call sites did not, even though `CreateCopy` reads block by block
and hits the same restriction: `to_file`, `copy` and `change_no_data_value`
failed on exactly the datasets the read paths handle.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[1] / "data" / "netcdf"


def flipped_variables() -> list[tuple[str, str]]:
    """Every (fixture, variable) whose backing view is Y-reversed."""
    found: list[tuple[str, str]] = []
    for path in sorted(glob.glob(str(DATA / "*.nc"))):
        try:
            dataset = NetCDF.read_file(path)
            for name in dataset.variable_names[:2]:
                if getattr(dataset.get_variable(name), "_md_y_flipped", False):
                    found.append((path, name))
        except Exception:  # pragma: no cover - fixture-dependent
            continue
    return found


FLIPPED = flipped_variables()
IDS = [f"{Path(p).stem}:{n}" for p, n in FLIPPED]


@pytest.mark.skipif(not FLIPPED, reason="no Y-reversed fixtures available")
@pytest.mark.parametrize("path,variable", FLIPPED, ids=IDS)
class TestWritingAReversedView:
    """The write paths handle what the read paths already handled."""

    def test_to_file_writes_a_netcdf(self, path: str, variable: str, tmp_path: Path):
        """`to_file` completes for a variable backed by a reversed view."""
        dataset = NetCDF.read_file(path).get_variable(variable)
        destination = tmp_path / "out.nc"

        dataset.to_file(str(destination))

        assert destination.exists()

    def test_copy_completes(self, path: str, variable: str, tmp_path: Path):
        """`copy` completes for a variable backed by a reversed view.

        This is the site that raised `IReadBlock failed ... arrayStartIdx`.
        """
        dataset = NetCDF.read_file(path).get_variable(variable)
        destination = tmp_path / "copy.nc"

        dataset.copy(str(destination))

        assert destination.exists()


class TestCopySourceMaterializes:
    """`_copy_source` is the materialising accessor the copy sites read through."""

    def test_it_returns_the_backing_raster(self):
        """For an ordinary raster it is just the raster, materialisation a no-op."""
        dataset = Dataset.from_array(
            np.ones((2, 2), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )

        assert dataset._copy_source is dataset._raster
