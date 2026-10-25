"""Driver dispatch for `Network1d.read_file` (issue #1351).

A NetCDF suffix short-circuits ahead of driver discovery inside `_detected_read_driver`, so a
`.nc` carrying a 1D network never reached the plugin able to read it: `can_open` was not
called even once. `UgridDataset.read_file` recovers from that short-circuit through
`_netcdf_fallback_driver`; these tests cover `Network1d.read_file` doing the same, and the
error message it raises when nothing claims the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import _drivers
from pyramids.netcdf.ugrid._drivers import (
    MESH_DRIVER_API_VERSION,
    UnsupportedMeshFormatError,
)
from pyramids.netcdf.ugrid.network import Network1d

pytestmark = pytest.mark.core

CHAIN = (
    np.array([0.0, 1.0, 2.0, 3.0]),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([[0, 1], [1, 2], [2, 3]]),
)


class NetworkNetcdfDriver:
    """A driver reading a 1D network out of a `.nc`, like `pyramids-mesh`'s `deltares1d`.

    Declares the same suffix and `read_mesh` capability as the real one, and records every
    `can_open` it is asked, which is what the dispatch assertions turn on.
    """

    name = "net1d"
    long_name = "Deltares 1D network (fake)"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def __init__(self) -> None:
        """Start with empty call logs."""
        self.sniffed: list[str] = []
        self.read_calls: list[tuple[str, str | None]] = []

    def can_open(self, path: str) -> bool:
        """Claim any file offered, recording the path.

        Args:
            path: The file to inspect.

        Returns:
            True, always.
        """
        self.sniffed.append(path)
        return True

    def read(self, path: str, mesh_name: str | None = None, lazy: bool = True):
        """Return a fixed 4-node chain, recording the call.

        Args:
            path: The file to read.
            mesh_name: Topology requested, recorded as given.
            lazy: Accepted and ignored.

        Returns:
            Network1d: A 4-node, 3-edge chain.
        """
        self.read_calls.append((path, mesh_name))
        return Network1d.from_arrays(*CHAIN)


class SurfaceNetcdfDriver(NetworkNetcdfDriver):
    """A `.nc` driver that returns a surface mesh, so `Network1d` must refuse what it gets."""

    name = "surf2d"
    long_name = "Surface mesh in NetCDF (fake)"

    def read(self, path: str, mesh_name: str | None = None, lazy: bool = True):
        """Return a `UgridDataset`, which is the wrong type for this entry point.

        Args:
            path: The file to read.
            mesh_name: Topology requested, recorded as given.
            lazy: Accepted and ignored.

        Returns:
            UgridDataset: A two-triangle surface mesh.
        """
        from tests.ugrid.conftest import make_two_triangle_ds

        self.read_calls.append((path, mesh_name))
        return make_two_triangle_ds()


@pytest.fixture
def registered(monkeypatch):
    """Inject drivers straight into the registry cache.

    Returns:
        A callable taking driver instances and returning the first one, so a test keeps a
        handle on the instance the registry hands to pyramids.
    """

    def _register(*drivers):
        monkeypatch.setattr(
            _drivers, "_CACHE", {driver.name: driver for driver in drivers}
        )
        return drivers[0] if len(drivers) == 1 else drivers

    return _register


@pytest.fixture
def netcdf_path(tmp_path):
    """An empty file with a `.nc` suffix.

    The driver claims every file it is offered, so the bytes never matter — only that the
    path exists, since `_require_existing_file` runs before any driver is consulted.

    Returns:
        pathlib.Path: The file.
    """
    target = tmp_path / "korte_woerden_1d_FlowFM_net.nc"
    target.write_bytes(b"")
    return target


class TestNetcdfFallback:
    """A `.nc` network reaches a driver claiming the suffix (issue #1351)."""

    def test_read_file_dispatches_to_a_netcdf_driver(self, netcdf_path, registered):
        """`read_file(path)` with no `driver=` finds the driver and returns its network.

        Test scenario:
            This is the issue's repro. Before the fix the suffix short-circuit returned the
            built-in reader, `selected` was None, and `UnsupportedMeshFormatError` was raised
            without the driver's `can_open` being called at all.
        """
        driver = registered(NetworkNetcdfDriver())
        result = Network1d.read_file(netcdf_path)
        assert isinstance(result, Network1d), f"got {type(result).__name__}"
        assert (result.n_node, result.n_edge) == (4, 3)
        assert driver.sniffed == [str(netcdf_path)], (
            f"can_open should be consulted exactly once, got {driver.sniffed}"
        )
        assert driver.read_calls == [(str(netcdf_path), None)], (
            f"the driver's read should have run, got {driver.read_calls}"
        )

    def test_mesh_name_is_forwarded_through_the_fallback(self, netcdf_path, registered):
        """The fallback route forwards `mesh_name`, as the named-driver route does."""
        driver = registered(NetworkNetcdfDriver())
        Network1d.read_file(netcdf_path, mesh_name="network1d")
        assert driver.read_calls == [(str(netcdf_path), "network1d")]

    def test_an_explicit_driver_still_bypasses_detection(self, netcdf_path, registered):
        """Naming the driver dispatches without sniffing — the documented opt-out.

        Test scenario:
            `driver=` means the caller chose the reader, so neither detection nor the
            fallback should second-guess it. This is the call that already worked before the
            fix, and it must keep working without a `can_open` round trip.
        """
        driver = registered(NetworkNetcdfDriver())
        result = Network1d.read_file(netcdf_path, driver="net1d")
        assert isinstance(result, Network1d)
        assert driver.sniffed == [], (
            f"an explicit driver should not be sniffed, got {driver.sniffed}"
        )

    def test_a_driver_returning_a_surface_mesh_is_refused(
        self, netcdf_path, registered
    ):
        """The fallback does not weaken the type check at the end of the read."""
        registered(SurfaceNetcdfDriver())
        with pytest.raises(UnsupportedMeshFormatError, match="not a Network1d"):
            Network1d.read_file(netcdf_path)


class TestNoDriverMessage:
    """What `read_file` says when no installed driver claims the file (issue #1351)."""

    def test_the_message_names_the_installed_drivers_and_the_explicit_call(
        self, netcdf_path, registered
    ):
        """A driver that declines the file produces a message naming the way through.

        Test scenario:
            The old message sent the caller to `UgridDataset.read_file`, which refuses a
            network by design and sends them straight back — a loop naming neither `driver=`
            nor the drivers installed.
        """

        class DecliningDriver(NetworkNetcdfDriver):
            """Claims the suffix but declines every file."""

            name = "declines"

            def can_open(self, path: str) -> bool:
                """Decline the file, recording the path.

                Args:
                    path: The file offered.

                Returns:
                    False, always.
                """
                self.sniffed.append(path)
                return False

        registered(DecliningDriver())
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(netcdf_path)
        message = str(caught.value)
        assert "as a 1D network" in message
        assert "declines" in message, "a driver claiming the suffix must be listed"
        assert "driver='declines'" in message, (
            f"the message must name the explicit-driver call, got:\n{message}"
        )
        assert "surface mesh" in message, (
            f"a .nc may well be a surface mesh, and that is the most likely mistake here, "
            f"so the message must say where to read one; got:\n{message}"
        )

    def test_the_message_says_so_when_nothing_is_installed(
        self, netcdf_path, registered
    ):
        """With no drivers at all the message asks for one rather than naming a call."""
        registered()
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(netcdf_path)
        message = str(caught.value)
        assert "none installed" in message
        assert "Install a mesh driver plugin" in message
        assert "driver=" not in message, (
            f"with nothing claiming the suffix there is no driver to name; got:\n{message}"
        )

    def test_a_missing_file_is_a_missing_file(self, tmp_path, registered):
        """A path that does not exist reports as missing, not as an unsupported format."""
        registered(NetworkNetcdfDriver())
        with pytest.raises(FileNotFoundError):
            Network1d.read_file(tmp_path / "absent_net.nc")
