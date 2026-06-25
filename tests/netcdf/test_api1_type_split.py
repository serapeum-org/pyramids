"""Contract tests for the API-1 container/variable type split (issue #614).

``NetCDF`` is now the (deprecated) base of two concrete public types:

* :class:`pyramids.netcdf.NetCDFContainer` — returned by the file-open / build entry
  points (``read_file`` / ``from_bytes`` / ``create_from_array`` / ``get_group``).
* :class:`pyramids.netcdf.NetCDFVariable` — returned by ``get_variable`` and the
  variable-level operations (``subset`` / ``sel`` / ``crop`` / ``to_crs`` / ``resample``).

Both subclass ``NetCDF`` so existing ``isinstance(x, NetCDF)`` checks keep working.
These tests pin the routing, the deprecation of direct base construction, the
type-preservation across copy / in-place updates, and the formalized container guard.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import NetCDF, NetCDFContainer, NetCDFVariable

pytestmark = pytest.mark.core

THREE_D = "tests/data/netcdf/pyramids-netcdf-3d.nc"
CLASSIC = "tests/data/netcdf/noah-precipitation-1979.nc"


@pytest.fixture(scope="function")
def container() -> NetCDFContainer:
    """An MDIM container opened from the 3-D fixture."""
    return NetCDF.read_file(THREE_D, open_as_multi_dimensional=True)


@pytest.fixture(scope="function")
def variable(container) -> NetCDFVariable:
    """A single variable extracted from the container."""
    return container.get_variable(container.variable_names[0])


def _make_container() -> NetCDFContainer:
    """Build a small in-memory container via create_from_array."""
    arr = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    return NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
        epsg=4326,
        variable_name="t",
        extra_dims=[("time", [0, 1])],
    )


class TestContainerRouting:
    """The file-open / build entry points return NetCDFContainer."""

    def test_read_file_mdim_returns_container(self, container):
        """``read_file(open_as_multi_dimensional=True)`` returns a NetCDFContainer.

        Test scenario:
            The MDIM open yields the canonical container type, which is still an
            ``isinstance`` of ``NetCDF`` so legacy checks keep passing.
        """
        assert type(container) is NetCDFContainer, f"got {type(container)}"
        assert isinstance(container, NetCDF), "container must remain a NetCDF instance"

    def test_read_file_classic_returns_container(self):
        """``read_file(open_as_multi_dimensional=False)`` also returns a NetCDFContainer.

        Test scenario:
            Opening a file is a container operation regardless of mode; the classic-mode
            handle is still a NetCDFContainer (and a NetCDF).
        """
        nc = NetCDF.read_file(CLASSIC, open_as_multi_dimensional=False)
        assert type(nc) is NetCDFContainer, f"got {type(nc)}"
        assert isinstance(nc, NetCDF)

    def test_from_bytes_returns_container(self):
        """``from_bytes`` returns a NetCDFContainer.

        Test scenario:
            Opening from in-memory bytes is a file-open, so it yields a NetCDFContainer.
        """
        nc = NetCDF.from_bytes(Path(CLASSIC).read_bytes())
        assert type(nc) is NetCDFContainer, f"got {type(nc)}"

    def test_create_from_array_returns_container(self):
        """``create_from_array`` returns a NetCDFContainer.

        Test scenario:
            Building a store from arrays produces a container holding the variable(s).
        """
        nc = _make_container()
        assert type(nc) is NetCDFContainer, f"got {type(nc)}"
        assert "t" in nc.variable_names, "the built variable should be present"


class TestVariableRouting:
    """get_variable and variable-level ops return NetCDFVariable."""

    def test_get_variable_returns_variable(self, variable):
        """``get_variable`` returns a NetCDFVariable that is also a NetCDF.

        Test scenario:
            A single extracted variable is the raster-bearing type; callers can dispatch
            on ``isinstance(x, NetCDFVariable)``.
        """
        assert type(variable) is NetCDFVariable, f"got {type(variable)}"
        assert isinstance(variable, NetCDF)

    def test_crop_on_variable_returns_variable(self, variable):
        """A spatial op on a variable returns a NetCDFVariable.

        Test scenario:
            Cropping the variable to its own bounding box keeps every cell and must
            return a NetCDFVariable (consistent-return-type contract).
        """
        cropped = variable.crop(bbox=variable.bbox)
        assert isinstance(cropped, NetCDFVariable), f"crop returned {type(cropped)}"

    def test_subset_returns_variable(self, container):
        """``subset`` returns a NetCDFVariable.

        Test scenario:
            A windowed variable subset is a raster, so it is a NetCDFVariable.
        """
        sub = container.subset("values", time=0)
        assert isinstance(sub, NetCDFVariable), f"subset returned {type(sub)}"


class TestDirectConstructionDeprecation:
    """Direct base NetCDF construction is deprecated; subclass construction is silent."""

    def test_direct_base_construction_warns(self, container):
        """Constructing the base ``NetCDF`` directly emits a DeprecationWarning.

        Test scenario:
            ``NetCDF(gdal_dataset)`` is the deprecated path; it must warn and point at the
            typed entry points.
        """
        with pytest.warns(DeprecationWarning, match="Directly constructing NetCDF"):
            NetCDF(container._raster)

    def test_subclass_construction_is_silent(self, container):
        """Constructing a subclass directly does not warn.

        Test scenario:
            ``NetCDFContainer(gdal_dataset)`` / ``NetCDFVariable(...)`` are the canonical
            types; their construction must be warning-free even under ``error`` filtering.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            c = NetCDFContainer(container._raster)
            assert isinstance(c, NetCDF)


class TestTypePreservation:
    """copy() and in-place updates keep the concrete subclass."""

    def test_copy_preserves_container_type(self, container):
        """Copying a container yields a container.

        Test scenario:
            ``container.copy()`` (in-memory MEM copy) must return a NetCDFContainer, not
            the base NetCDF.
        """
        assert type(container.copy()) is NetCDFContainer, "copy downgraded the type"

    def test_copy_preserves_variable_type(self, variable):
        """Copying a variable yields a variable.

        Test scenario:
            ``variable.copy()`` must return a NetCDFVariable.
        """
        assert type(variable.copy()) is NetCDFVariable, "copy downgraded the type"

    def test_epsg_setter_inplace_preserves_variable_type(self, variable):
        """An in-place op (the ``epsg`` setter) does not downgrade a NetCDFVariable.

        Test scenario:
            The ``epsg`` setter rebuilds the instance via ``_update_inplace`` → ``type(self)``;
            re-setting the EPSG on a variable must keep it a NetCDFVariable rather than
            reverting to the base ``NetCDF``.
        """
        variable.epsg = variable.epsg
        assert type(variable) is NetCDFVariable, f"in-place op downgraded to {type(variable)}"


class TestContainerGuard:
    """The container rejects band-level ops; the variable allows them."""

    def test_mdim_container_rejects_read_array(self, container):
        """Calling a band-level op on an MDIM container raises a guiding ValueError.

        Test scenario:
            ``read_array`` on the root MDIM container (band_count == 0) must raise and tell
            the user to extract a variable first.
        """
        with pytest.raises(ValueError, match="get_variable"):
            container.read_array()

    def test_variable_allows_read_array(self, variable):
        """A NetCDFVariable's container guard is a no-op, so read_array works.

        Test scenario:
            ``read_array`` on the extracted variable returns a real array — the variable
            is always a valid raster.
        """
        arr = variable.read_array()
        assert arr is not None and arr.size > 0, "variable read_array should return data"

    def test_variable_check_not_container_is_noop(self, variable):
        """``NetCDFVariable._check_not_container`` returns None and never raises.

        Test scenario:
            The override makes the type contract explicit — a variable is never a
            container, so the guard is a no-op.
        """
        assert variable._check_not_container("read_array") is None


class TestPublicExports:
    """Both types are importable from the documented paths and are identical objects."""

    def test_exports_from_package_and_variable_module(self):
        """``pyramids.netcdf`` and ``pyramids.netcdf.variable`` expose the same classes.

        Test scenario:
            The re-export module must hand back the very same class objects (no shadow
            copies), and both must subclass ``NetCDF``.
        """
        from pyramids.netcdf.variable import NetCDFContainer as VC
        from pyramids.netcdf.variable import NetCDFVariable as VV

        assert VC is NetCDFContainer, "variable.NetCDFContainer must be the same object"
        assert VV is NetCDFVariable, "variable.NetCDFVariable must be the same object"
        assert issubclass(NetCDFContainer, NetCDF) and issubclass(NetCDFVariable, NetCDF)
