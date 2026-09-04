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


class TestTheContainerReplacesAFlatBandStack:
    """What changed for a caller, on a fixture where it is visible.

    `read_resource` used to hand a multi-variable netCDF to the plain raster
    reader, which stacked every variable's every timestep into one flat band
    list -- a 12-band `Dataset` with no way to tell which band was which
    variable. It now returns a container that names them.

    That is a better answer and a breaking one: the returned type changes, and
    `read_array()` on the container refuses rather than returning the stack. The
    file this suite's other tests use has a base band count of zero, so the
    change is invisible there; this pins it where it shows.
    """

    FIXTURE = Path(__file__).parent / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"

    def test_the_container_names_its_variables(self):
        """The gain: the variables are addressable instead of anonymous bands.

        Test scenario:
            The old 12-band stack gave no way to say which band belonged to
            which variable or timestep.
        """
        resource = read_resource(str(self.FIXTURE))

        assert resource.variable_names == ["temperature"]

    def test_reading_the_container_directly_is_refused_with_advice(self):
        """The break: `read_array()` no longer returns the flat stack.

        Test scenario:
            A caller who previously did `read_resource(path).read_array()` now
            gets a `ValueError`. It has to name the way forward rather than
            fail blankly, since this is the call that used to work.
        """
        resource = read_resource(str(self.FIXTURE))

        with pytest.raises(ValueError, match="get_variable"):
            resource.read_array()

    def test_the_variable_reads_the_data_the_stack_used_to_carry(self):
        """The migration path, exercised end to end.

        Test scenario:
            `get_variable(...).read_array()` is what replaces the old call, and
            it returns the variable's own bands rather than every variable's.
        """
        resource = read_resource(str(self.FIXTURE))

        array = resource.get_variable("temperature").read_array()

        assert array.ndim >= 2
