"""`read_resource` must hand a netCDF to the netCDF reader.

`_read_raster` sent every raster-family suffix to `Dataset.read_file`, which
opens a netCDF as a plain raster: the variables and the time dimension the
`NetCDF` reader recovers were simply not there. The family sniff was right --
`sniff_kind("cube.nc")` is `"raster"` -- but the family does not determine the
reader.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pyramids._resource import read_resource, sniff_kind
from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

FIXTURE = Path(__file__).parent / "data" / "netcdf" / "cf__6v__1d2-2d4__geog__y-asc.nc"


class TestNetcdfResourcesReachTheNetcdfReader:
    """A `.nc` resource comes back as a `NetCDF`, not a bare `Dataset`."""

    def test_a_netcdf_reads_as_netcdf(self, tmp_path: Path):
        """The reader is chosen by suffix, not just by family."""
        destination = tmp_path / "cube.nc"
        shutil.copy(FIXTURE, destination)

        result = read_resource(str(destination))

        assert isinstance(result, NetCDF)

    def test_the_variables_are_recovered(self, tmp_path: Path):
        """This is what the plain raster reader was losing."""
        destination = tmp_path / "cube.nc"
        shutil.copy(FIXTURE, destination)

        result = read_resource(str(destination))

        assert result.variable_names, "no variables recovered"

    def test_a_netcdf_is_still_a_dataset(self, tmp_path: Path):
        """`NetCDF` subclasses `Dataset`, so no caller's type check breaks."""
        destination = tmp_path / "cube.nc"
        shutil.copy(FIXTURE, destination)

        assert isinstance(read_resource(str(destination)), Dataset)

    def test_the_family_sniff_is_unchanged(self):
        """A netCDF is still in the raster family; only the reader changed."""
        assert sniff_kind("cube.nc") == "raster"

    @pytest.mark.parametrize("suffix", [".nc", ".nc4", ".cdf"])
    def test_every_netcdf_suffix_routes_the_same_way(self, suffix: str, tmp_path: Path):
        """The same format under three names, content deciding the reader.

        Args:
            suffix: A netCDF file extension.
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `.cdf` is the one that needs saying. It is absent from
            `_RASTER_SUFFIXES`, which makes the netCDF suffix set look as
            though it could never be consulted for such a file -- but an
            unlisted suffix falls through to `sniff_format`, which maps `.cdf`
            to "nc" and back to the raster kind. Leaving it out of this
            parametrize is what let it be dropped from the suffix set
            unnoticed, sending the file to the plain raster reader and back as
            a band-less `Dataset`.
        """
        destination = tmp_path / f"cube{suffix}"
        shutil.copy(FIXTURE, destination)

        resource = read_resource(str(destination))

        assert isinstance(resource, NetCDF)
        assert resource.variable_names, f"{suffix} enumerated no variables"

    def test_a_geotiff_still_reads_as_a_plain_dataset(self, tmp_path: Path):
        """The other raster suffixes are untouched."""
        source = tmp_path / "plain.tif"
        Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        ).to_file(str(source))

        result = read_resource(str(source))

        assert isinstance(result, Dataset)
        assert not isinstance(result, NetCDF)
