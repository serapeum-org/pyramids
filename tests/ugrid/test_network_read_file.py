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

    def test_a_reserved_driver_name_is_refused_with_the_documented_reason(
        self, netcdf_path, registered
    ):
        """`driver="ugrid"` is refused by name, not reported as "none installed" (M3).

        Test scenario:
            The docstring says `"ugrid"` is refused here because the built-in reader builds a
            `Mesh2d` whatever dimension a file declares. No such refusal existed: the
            reserved name resolved to the built-in reader, the fallback declined because a
            driver was named, and the generic message then claimed no driver was installed --
            false, since `ugrid` is the built-in one.
        """
        registered(NetworkNetcdfDriver())
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(netcdf_path, driver="ugrid")
        message = str(caught.value)
        assert "ugrid" in message
        assert "2D mesh" in message or "Mesh2d" in message, (
            f"the refusal must give the documented reason, got:\n{message}"
        )
        assert "none installed" not in message, (
            f"'ugrid' is the built-in reader, so this is false; got:\n{message}"
        )

    def test_a_reserved_name_is_refused_case_insensitively(
        self, netcdf_path, registered
    ):
        """Reserved names are matched the way `_named_read_driver` matches them (M3).

        Test scenario:
            `_named_read_driver` lowercases and strips before testing membership, so the
            refusal has to as well -- otherwise `driver="UGRID"` slips past it and lands on
            the generic "no installed driver" message again. The name is echoed back as the
            caller spelled it.
        """
        registered(NetworkNetcdfDriver())
        with pytest.raises(
            UnsupportedMeshFormatError, match="built-in UGRID reader"
        ) as caught:
            Network1d.read_file(netcdf_path, driver="UGRID")
        assert "'UGRID'" in str(caught.value), (
            f"the name is echoed as spelled, got: {caught.value}"
        )

    @pytest.mark.parametrize(
        "driver",
        [None, "ugrid", "net1d", "not_installed"],
        ids=["detect", "reserved", "installed", "unknown"],
    )
    def test_a_missing_file_reports_as_missing_whatever_driver_is_named(
        self, tmp_path, registered, driver
    ):
        """The existence check comes first on every route into `read_file`.

        Test scenario:
            `_resolve_read_driver` checks existence before it looks at a driver name, and
            `read_file` documents `FileNotFoundError` for an absent path. The reserved-name
            refusal added for M3 was placed ahead of that check, so `driver="ugrid"` on a
            path that does not exist reported a format problem instead -- the one route of
            four that disagreed.
        """
        registered(NetworkNetcdfDriver())
        absent = tmp_path / "absent_net.nc"
        kwargs = {} if driver is None else {"driver": driver}
        with pytest.raises(FileNotFoundError):
            Network1d.read_file(absent, **kwargs)


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

    def test_a_non_netcdf_suffix_gets_no_surface_mesh_pointer(
        self, tmp_path, registered
    ):
        """The `UgridDataset.read_file` line is for a NetCDF suffix only.

        Test scenario:
            The pointer exists because a `.nc` handed to the 1D entry point is most likely a
            surface mesh. For a format that cannot be one -- a `.2dm`, say -- the line would
            be noise, so the message omits it. Every other test here uses `.nc`, which left
            this arm of the branch unexercised.
        """
        registered()
        target = tmp_path / "channels.2dm"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(target)
        message = str(caught.value)
        assert "'.2dm'" in message, f"the suffix must be named, got:\n{message}"
        assert "surface mesh" not in message, (
            f"a .2dm cannot be read by the built-in UGRID reader, so pointing at "
            f"UgridDataset.read_file would be misleading; got:\n{message}"
        )
        assert "Install a mesh driver plugin" in message, (
            f"nothing claims '.2dm' here, so a plugin is the way forward; got:\n{message}"
        )


class TestResultsOnlyDriver:
    """A `needs_mesh` driver reached through the fallback (issue #1351, L10)."""

    def test_the_advice_does_not_name_a_method_network1d_lacks(
        self, netcdf_path, registered
    ):
        """A results-only driver must not send a network caller to `attach_data` (L10).

        Test scenario:
            `_needs_mesh_error` was written for `UgridDataset.read_file`, so its recipe is
            `mesh = UgridDataset.read_file(...)` then `mesh.attach_data(...)`. Reached from
            `Network1d.read_file` that is wrong twice over: the caller asked for a network,
            and `Network1d` has no `attach_data` at all.
        """

        class ResultsOnlyDriver(NetworkNetcdfDriver):
            """Carries values but no topology, like a TUFLOW FV results reader."""

            name = "results_only"
            long_name = "Results-only NetCDF (fake)"
            capabilities = frozenset({"needs_mesh", "read_data"})

        registered(ResultsOnlyDriver())
        assert not hasattr(Network1d, "attach_data"), (
            "the premise of this test is that Network1d has no attach_data"
        )
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(netcdf_path)
        message = str(caught.value)
        assert "no attach_data" in message, (
            f"the message must say why the usual recipe does not apply, got:\n{message}"
        )
        assert "UgridDataset.read_file" in message, (
            f"it must still name where the mesh comes from, got:\n{message}"
        )
        assert "mesh.attach_data" not in message, (
            f"the UgridDataset recipe must not be offered to a network caller, "
            f"got:\n{message}"
        )
