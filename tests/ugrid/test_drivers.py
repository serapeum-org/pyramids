"""The mesh driver registry and dispatch, exercised with no plugin installed.

Every test here builds its own fake driver. Two fixtures make that possible: `registered`
injects drivers straight into the registry cache, and `fake_entry_points` monkeypatches
`importlib.metadata.entry_points` with in-memory `EntryPoint` objects so real discovery runs
without a distribution on disk.

The negative cases matter more than the positive ones. A plugin that raises on import, declares
a future API version or collides on a name must be skipped with a warning, because the
alternative is a broken third-party package stopping a NetCDF file from opening.
"""

from __future__ import annotations

import threading
import warnings
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import DriverNotExistError, FileFormatNotSupportedError
from pyramids.netcdf.ugrid import _drivers
from pyramids.netcdf.ugrid._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    MESH_DRIVER_API_VERSION,
    NoMeshTopologyError,
    UnsupportedMeshFormatError,
    clear_driver_cache,
    describe_drivers,
    drivers_for_suffix,
    mesh_drivers,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset, _netcdf_fallback_driver
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from pyramids.netcdf.ugrid.network import Network1d


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the discovery cache around every test.

    Without this the cache leaks between tests and `test_netcdf_read_does_no_discovery` passes
    for the wrong reason -- an already-populated cache is never refilled, so the assertion that
    it stayed empty would hold even if the NetCDF path did consult the registry.
    """
    clear_driver_cache()
    yield
    clear_driver_cache()


def _square_mesh() -> UgridDataset:
    """Build the smallest valid dataset a fake driver can return.

    Returns:
        A one-quad mesh carrying a single face-located variable.
    """
    faces = Connectivity(
        data=np.array([[0, 1, 2, 3]]),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=faces,
    )
    variable = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(1,),
        _data=np.array([2.5]),
    )
    return UgridDataset(
        mesh=mesh,
        data_variables={"depth": variable},
        global_attributes={"Conventions": "UGRID-1.0"},
    )


class FakeDriver:
    """A minimal driver that reads one made-up format."""

    name = "fake"
    long_name = "Fake Mesh Format"
    suffixes = (".fake",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "write_mesh", "write_data"})

    def __init__(self) -> None:
        """Record what the driver was asked to do, so tests can assert on it."""
        self.read_calls: list[tuple[str, str | None]] = []
        self.write_calls: list[tuple[str, dict]] = []
        self.sniffed: list[str] = []
        self.returns: object | None = None

    def can_open(self, path: str) -> bool:
        """Accept any path this driver is offered.

        Args:
            path: The file to inspect.

        Returns:
            True, always; tests that need a veto use `VetoDriver`.
        """
        self.sniffed.append(path)
        return True

    def read(self, path, *, mesh_name=None, lazy=True):
        """Return the prepared dataset.

        Args:
            path: The file to read.
            mesh_name: Which topology to read.
            lazy: Defer reading values.

        Returns:
            Whatever `returns` was set to, or a one-quad mesh.
        """
        self.read_calls.append((path, mesh_name))
        return _square_mesh() if self.returns is None else self.returns

    def read_data(self, path, dataset, *, lazy=True):
        """Attach nothing; `needs_mesh` drivers have their own fake.

        Args:
            path: The results file.
            dataset: The mesh to attach to.
            lazy: Defer reading values.

        Returns:
            The dataset unchanged.
        """
        return dataset

    def mesh_names(self, path: str) -> list[str]:
        """Report two topologies, so multi-mesh routing is observable.

        Args:
            path: The file to inspect.

        Returns:
            Two names.
        """
        return ["first", "second"]

    def write(self, dataset, path, **options) -> None:
        """Record the write instead of producing a file.

        Args:
            dataset: The dataset to write.
            path: The destination.
            **options: Format-specific settings.
        """
        self.write_calls.append((path, options))


class VetoDriver(FakeDriver):
    """A driver that claims the suffix but refuses the file."""

    name = "veto"
    long_name = "Veto Format"
    suffixes = (".fake",)

    def can_open(self, path: str) -> bool:
        """Refuse every file.

        Args:
            path: The file to inspect.

        Returns:
            False, always.
        """
        self.sniffed.append(path)
        return False


class AttachDriver(FakeDriver):
    """A results-only driver: values, no topology."""

    name = "attach"
    long_name = "Attachable Results"
    suffixes = (".res",)
    capabilities = frozenset({"needs_mesh", "read_data"})

    def __init__(self, declared_elements: int | None = None) -> None:
        """Set up the driver.

        Args:
            declared_elements: The element count the file claims, so a mismatch can be tested.
                ``None`` matches whatever mesh it is given.
        """
        super().__init__()
        self.declared_elements = declared_elements

    def read(self, path, *, mesh_name=None, lazy=True):
        """Refuse: this format carries no topology.

        Args:
            path: The file.
            mesh_name: Unused.
            lazy: Unused.

        Raises:
            UnsupportedMeshFormatError: Always.
        """
        raise UnsupportedMeshFormatError("results-only; use attach_data")

    def read_data(self, path, dataset, *, lazy=True):
        """Return the variables read from the file, after checking the element count.

        Args:
            path: The results file.
            dataset: The mesh the values attach to.
            lazy: Defer reading values.

        Returns:
            The variables read, keyed by name. Only the driver's own — `attach_data` merges.

        Raises:
            ValueError: If the declared element count does not match the mesh.
        """
        declared = (
            dataset.n_face if self.declared_elements is None else self.declared_elements
        )
        if declared != dataset.n_face:
            raise ValueError(
                f"{path} declares {declared} elements; the mesh has {dataset.n_face}."
            )
        return {
            "velocity": MeshVariable(
                name="velocity",
                location="face",
                mesh_name=dataset.mesh_name,
                shape=(dataset.n_face,),
                _data=np.zeros(dataset.n_face),
            )
        }


class FrameOnlyDriver(FakeDriver):
    """A driver that writes a mesh frame but no variables."""

    name = "frame"
    long_name = "Frame Only"
    suffixes = (".frm",)
    capabilities = frozenset({"read_mesh", "write_mesh"})


class ReadOnlyDriver(FakeDriver):
    """A driver that cannot write."""

    name = "readonly"
    long_name = "Read Only"
    suffixes = (".ro",)
    capabilities = frozenset({"read_mesh"})


@pytest.fixture
def registered(monkeypatch):
    """Inject drivers directly into the registry cache.

    Returns:
        A callable taking driver instances and returning them, so a test can keep a handle on
        the instance the registry will hand to pyramids.
    """

    def _register(*drivers):
        monkeypatch.setattr(
            _drivers, "_CACHE", {driver.name: driver for driver in drivers}
        )
        return drivers[0] if len(drivers) == 1 else drivers

    return _register


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Monkeypatch `entry_points` so real discovery runs over in-memory entry points.

    Returns:
        A callable taking ``(name, factory)`` pairs. Each factory is published under a module
        attribute that the EntryPoint's ``value`` resolves to, so ``EntryPoint.load`` follows its
        normal path rather than being stubbed.
    """
    published: dict[str, object] = {}

    def _publish(*pairs):
        entries = []
        for index, (name, factory) in enumerate(pairs):
            attribute = f"_fake_driver_{index}"
            published[attribute] = factory
            setattr(_drivers, attribute, factory)
            entries.append(
                EntryPoint(
                    name=name,
                    value=f"pyramids.netcdf.ugrid._drivers:{attribute}",
                    group=ENTRY_POINT_GROUP,
                )
            )

        def _entry_points(*, group=None):
            return tuple(entries) if group == ENTRY_POINT_GROUP else ()

        monkeypatch.setattr(_drivers, "entry_points", _entry_points)

    yield _publish
    for attribute in published:
        monkeypatch.delattr(_drivers, attribute, raising=False)


class TestValidation:
    """Registration rejects a driver that does not meet the contract."""

    @pytest.mark.parametrize(
        ("attribute", "value", "expected"),
        [
            ("api_version", 99, "api_version"),
            ("suffixes", (), "no suffixes"),
            ("suffixes", ("slf",), "malformed suffix"),
            ("suffixes", (".SLF",), "malformed suffix"),
            (
                "capabilities",
                frozenset({"read_mesh", "teleport"}),
                "unknown capability",
            ),
            (
                "capabilities",
                frozenset({"read_mesh", "needs_mesh"}),
                "declares exactly one",
            ),
            ("capabilities", frozenset({"read_data"}), "declares exactly one"),
        ],
    )
    def test_invalid_driver_is_skipped_with_a_warning(
        self, fake_entry_points, attribute, value, expected
    ):
        """A driver breaking any contract rule is skipped, and the warning says which."""

        class Broken(FakeDriver):
            pass

        setattr(Broken, attribute, value)
        fake_entry_points(("broken", Broken))
        with pytest.warns(RuntimeWarning, match=expected):
            assert mesh_drivers() == {}

    def test_driver_missing_an_attribute_is_skipped(self, fake_entry_points):
        """A class lacking a contract attribute is reported by name, not an AttributeError."""

        class Nameless:
            suffixes = (".x",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"read_mesh"})

        fake_entry_points(("nameless", Nameless))
        with pytest.warns(RuntimeWarning, match="missing attribute"):
            assert mesh_drivers() == {}

    def test_every_capability_token_is_accepted(self, fake_entry_points):
        """The documented vocabulary is exactly what validation allows."""

        class Everything(FakeDriver):
            capabilities = CAPABILITIES - {"needs_mesh"}

        fake_entry_points(("everything", Everything))
        assert set(mesh_drivers()) == {"fake"}


class TestDiscovery:
    """Discovery is lazy, cached, and survives a broken plugin."""

    def test_driver_raising_on_load_is_skipped_with_a_warning(self, fake_entry_points):
        """A plugin whose import fails -- a missing optional dependency, say -- costs itself only."""

        def explode():
            raise ImportError("No module named 'mikeio'")

        fake_entry_points(("exploding", explode), ("fake", FakeDriver))
        with pytest.warns(RuntimeWarning, match="could not be loaded"):
            resolved = mesh_drivers()
        assert set(resolved) == {"fake"}

    def test_duplicate_name_keeps_the_first_and_warns(self, fake_entry_points):
        """Two plugins claiming one name is a conflict the user must be told about."""

        class Other(FakeDriver):
            long_name = "The Second One"

        fake_entry_points(("first", FakeDriver), ("second", Other))
        with pytest.warns(RuntimeWarning, match="registered twice"):
            resolved = mesh_drivers()
        assert resolved["fake"].long_name == "Fake Mesh Format"

    def test_discovery_runs_once_and_instantiates_once(self, fake_entry_points):
        """A driver may cache parsed state, so it must not be rebuilt per read."""
        built = []

        class Counted(FakeDriver):
            def __init__(self) -> None:
                super().__init__()
                built.append(self)

        fake_entry_points(("counted", Counted))
        first = mesh_drivers()
        second = mesh_drivers()
        assert len(built) == 1, f"Built {len(built)} times, expected 1"
        # The mapping is a fresh copy each call (so a caller cannot mutate the registry),
        # but the driver inside it is the one shared instance.
        assert first["fake"] is second["fake"], (
            "The driver instance must be shared across calls"
        )

    def test_clear_cache_forces_rediscovery(self, fake_entry_points):
        """The test-only reset is what keeps the no-discovery guard honest."""
        fake_entry_points(("fake", FakeDriver))
        mesh_drivers()
        assert _drivers._CACHE is not None
        clear_driver_cache()
        assert _drivers._CACHE is None

    def test_drivers_for_suffix_normalises_and_preserves_order(self, registered):
        """Suffix lookup is case-insensitive, dot-optional, and in registration order."""
        first, second = registered(FakeDriver(), VetoDriver())
        assert drivers_for_suffix(".fake") == [first, second]
        assert drivers_for_suffix("FAKE") == [first, second]
        assert drivers_for_suffix(".missing") == []


class TestReadDispatch:
    """`read_file` resolution order."""

    def test_driver_kwarg_selects_that_driver(self, tmp_path, registered):
        """An explicit driver is used whatever the suffix says."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.whatever"
        target.write_bytes(b"")
        result = UgridDataset.read_file(target, driver="fake")
        assert result.n_face == 1
        assert driver.read_calls == [(str(target), None)]

    def test_driver_kwarg_does_not_sniff(self, tmp_path, registered):
        """Naming a driver skips `can_open`: the caller has already decided."""
        driver = registered(VetoDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target, driver="veto")
        assert driver.sniffed == []

    def test_unknown_driver_lists_what_is_installed(self, tmp_path, registered):
        """The error names the alternatives rather than only the failure."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(DriverNotExistError, match="fake"):
            UgridDataset.read_file(target, driver="nope")

    def test_can_open_veto_falls_through_to_gdal(self, tmp_path, registered):
        """A driver claiming the suffix but refusing the file does not capture the read."""
        driver = registered(VetoDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"not a mesh at all")
        with pytest.raises(UnsupportedMeshFormatError):
            UgridDataset.read_file(target)
        assert driver.sniffed == [str(target)]

    def test_unclaimed_suffix_reports_installed_drivers(self, tmp_path, registered):
        """GDAL's "not recognized" would point at a corrupt file, not a missing plugin."""
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01\x02")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert ".slf" in message
        assert "fake (.fake)" in message
        assert "mesh driver plugin" in message

    def test_unsupported_format_error_is_catchable_as_the_old_one(
        self, tmp_path, registered
    ):
        """Existing `except FileFormatNotSupportedError` callers keep working."""
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00")
        with pytest.raises(FileFormatNotSupportedError):
            UgridDataset.read_file(target)

    def test_needs_mesh_driver_points_at_attach_data(self, tmp_path, registered):
        """A results-only file reached through `read_file` must say where to go."""
        registered(AttachDriver())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="attach_data"):
            UgridDataset.read_file(target)

    def test_driver_returning_network1d_is_reported(self, tmp_path, registered):
        """A 1D topology is a different class, so handing it back would break the caller."""
        driver = registered(FakeDriver())
        driver.returns = Network1d(
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 0.0]),
            edge_node_connectivity=Connectivity(
                data=np.array([[0, 1]]),
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )
        target = tmp_path / "network.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="Network1d"):
            UgridDataset.read_file(target)

    def test_driver_failure_names_the_driver(self, tmp_path, registered):
        """A parse failure is attributed, so the blame is unambiguous."""

        class Failing(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise ValueError("record length mismatch at byte 80")

        registered(Failing())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="'fake' driver failed"):
            UgridDataset.read_file(target)

    @pytest.mark.parametrize("suffix", [".nc", ".nc4", ".NC", ".cdf"])
    def test_netcdf_read_does_no_discovery(
        self, tmp_path, suffix, ugrid_convention_nc_path
    ):
        """The performance contract: a NetCDF read must not walk every installed distribution.

        Asserted by watching the cache rather than by timing, so it cannot flake.
        """
        target = tmp_path / f"mesh{suffix}"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        UgridDataset.read_file(target)
        assert _drivers._CACHE is None

    def test_vsi_path_does_no_discovery(self, ugrid_convention_nc_path):
        """A virtual-filesystem name has no suffix to dispatch on and must stay on GDAL."""
        UgridDataset.from_bytes(Path(ugrid_convention_nc_path).read_bytes())
        assert _drivers._CACHE is None

    def test_extensionless_file_does_not_crash(self, tmp_path, registered):
        """`Path.suffix` is empty here; dispatch must handle it, not raise."""
        registered(FakeDriver())
        target = tmp_path / "mesh"
        target.write_bytes(b"\x00")
        with pytest.raises(UnsupportedMeshFormatError, match="no suffix"):
            UgridDataset.read_file(target)

    def test_suffix_match_is_case_insensitive(self, tmp_path, registered):
        """`.FAKE` is a real spelling on Windows."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.FAKE"
        target.write_bytes(b"")
        UgridDataset.read_file(target)
        assert len(driver.read_calls) == 1

    def test_missing_file_still_raises_file_not_found(self, tmp_path, registered):
        """Dispatch must not mask the simplest error of all."""
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "absent.nc")


class TestAttachData:
    """`attach_data` for results-only formats."""

    def test_attach_returns_a_new_dataset_and_leaves_the_original(
        self, tmp_path, registered
    ):
        """The immutable-derivation contract every other pyramids operation keeps."""
        registered(AttachDriver())
        mesh = _square_mesh()
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        result = mesh.attach_data(target)
        assert "velocity" in result.data_variable_names
        assert "velocity" not in mesh.data_variable_names
        assert result is not mesh

    def test_attach_refuses_an_element_count_mismatch(self, tmp_path, registered):
        """The only safeguard against attaching results to the wrong mesh."""
        registered(AttachDriver(declared_elements=4096))
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(ValueError, match="4096"):
            mesh.attach_data(target)

    def test_attach_refuses_a_driver_that_reads_its_own_mesh(
        self, tmp_path, registered
    ):
        """A `read_mesh` driver has nothing to attach."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="nothing to attach"):
            mesh.attach_data(target, driver="fake")

    def test_attach_with_no_driver_for_the_suffix_reports_installed(
        self, tmp_path, registered
    ):
        """Detection failing says what is available, as the read path does."""
        registered(AttachDriver())
        target = tmp_path / "velocity.unknown"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="No installed driver"):
            mesh.attach_data(target)

    def test_attach_rejects_a_variable_name_collision(self, tmp_path, registered):
        """Silently keeping either copy would lose data."""

        class Colliding(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return {
                    "depth": MeshVariable(
                        name="depth",
                        location="face",
                        mesh_name=dataset.mesh_name,
                        shape=(dataset.n_face,),
                        _data=np.ones(dataset.n_face),
                    )
                }

        registered(Colliding())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(ValueError, match="already defines"):
            mesh.attach_data(target)

    def test_attach_unknown_driver_raises(self, tmp_path, registered):
        """Naming a driver that is not installed is an error, not a fallback."""
        registered(AttachDriver())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(DriverNotExistError):
            mesh.attach_data(target, driver="nope")


class TestWriteDispatch:
    """`to_file` with a driver."""

    def test_default_write_is_unchanged(self, tmp_path):
        """No driver means the built-in UGRID writer, byte for byte as before."""
        target = tmp_path / "mesh.nc"
        _square_mesh().to_file(target)
        assert target.exists()
        assert UgridDataset.read_file(target).n_face == 1

    def test_driver_write_passes_options_through_verbatim(self, tmp_path, registered):
        """pyramids must not interpret or rename a format's own settings."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        _square_mesh().to_file(target, driver="fake", endian=">", precision="double")
        assert driver.write_calls == [
            (str(target), {"endian": ">", "precision": "double"})
        ]

    def test_write_driver_without_write_mesh_raises(self, tmp_path, registered):
        """A read-only driver says what it can do instead."""
        registered(ReadOnlyDriver())
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="cannot write"):
            mesh.to_file(tmp_path / "mesh.ro", driver="readonly")

    def test_frame_only_driver_warns_that_variables_are_dropped(
        self, tmp_path, registered
    ):
        """Losing data silently is the failure mode worth a warning."""
        registered(FrameOnlyDriver())
        mesh = _square_mesh()
        with pytest.warns(RuntimeWarning, match="no data"):
            mesh.to_file(tmp_path / "mesh.frm", driver="frame")

    def test_a_failed_frame_only_write_does_not_warn(self, tmp_path, registered):
        """The warning follows the write, so a failed write does not claim a file exists.

        The warning used to fire before `write` was called, so a write that raised still told
        the user that variables "will not be in" a file which was never created. Asserting
        the warning *happens* cannot catch that — it happens under either ordering. This
        asserts it does **not** happen when the write fails, which only holds if the warning
        comes after.

        Test scenario:
            A frame-only driver whose `write` raises produces the write error and no
            RuntimeWarning about dropped variables.
        """

        class FailingFrameOnly(FrameOnlyDriver):
            def write(self, dataset, path, **options):
                raise ValueError("cannot encode this mesh")

        registered(FailingFrameOnly())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = _square_mesh()
            with pytest.raises(UnsupportedMeshFormatError):
                mesh.to_file(tmp_path / "mesh.frm", driver="frame")
        dropped = [
            str(record.message) for record in caught if "no data" in str(record.message)
        ]
        assert dropped == [], (
            f"A failed write must not warn that variables are missing from a file that was "
            f"never created; got: {dropped}"
        )

    def test_suffix_does_not_select_a_driver(self, tmp_path, registered):
        """Inferring Selafin from `.slf` would change behaviour for existing callers."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        _square_mesh().to_file(target)
        assert driver.write_calls == []
        assert UgridDataset.read_file(target, driver="fake").n_face == 1

    def test_options_without_a_driver_are_rejected(self, tmp_path):
        """Dropping them silently would hide a caller's mistake."""
        mesh = _square_mesh()
        with pytest.raises(TypeError, match="driver="):
            mesh.to_file(tmp_path / "mesh.nc", endian=">")

    def test_driver_write_failure_names_the_driver(self, tmp_path, registered):
        """A driver's own bug on write is attributed to it, as on the read path.

        Test scenario:
            The driver raises a format-level error (a bad value it cannot encode); the
            message names the driver rather than letting a bare ValueError reach the caller.
        """

        class Failing(FakeDriver):
            def write(self, dataset, path, **options):
                raise ValueError("cannot encode a 6-node face in this format")

        registered(Failing())
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="'fake' driver failed"):
            mesh.to_file(tmp_path / "mesh.fake", driver="fake")

    def test_driver_write_oserror_passes_through(self, tmp_path, registered):
        """A filesystem failure on write stays a filesystem failure.

        "disk full" attributed to "the 'fake' driver failed" sends the user to inspect the
        plugin for something that is not its fault, so OSError passes through with its own
        message — matching the read path, where a multi-file format's missing sibling does
        the same.

        Test scenario:
            The driver raises OSError; the caller sees OSError, not a format error.
        """

        class DiskFull(FakeDriver):
            def write(self, dataset, path, **options):
                raise OSError("No space left on device")

        registered(DiskFull())
        mesh = _square_mesh()
        with pytest.raises(OSError, match="No space left") as caught:
            mesh.to_file(tmp_path / "mesh.fake", driver="fake")
        assert not isinstance(caught.value, UnsupportedMeshFormatError), (
            "A disk-full error must not be reported as an unsupported format"
        )

    def test_write_unknown_driver_raises(self, tmp_path, registered):
        """Consistent with the read and attach paths."""
        registered(FakeDriver())
        mesh = _square_mesh()
        with pytest.raises(DriverNotExistError):
            mesh.to_file(tmp_path / "mesh.fake", driver="nope")


class TestIntrospection:
    """`info(drivers=True)`, `mesh_names` and `read_all` routing."""

    def test_info_lists_the_builtin_reader_with_no_plugins(self, capsys):
        """The listing must read sensibly on a bare install, not render an empty section."""
        _square_mesh().info(drivers=True)
        captured = capsys.readouterr().out
        assert "built-in + 0 plugin" in captured, (
            f"A bare install has no plugins; got: {captured}"
        )
        assert "ugrid (built-in)" in captured, (
            f"The built-in reader should be listed; got: {captured}"
        )
        assert ".netcdf" in captured, (
            "The listed suffixes must match what dispatch accepts, not a shorter hand-written "
            f"list; got: {captured}"
        )

    def test_info_lists_a_registered_driver(self, capsys, registered):
        """pyramids' answer to MDAL's driver introspection."""
        registered(FakeDriver())
        _square_mesh().info(drivers=True)
        captured = capsys.readouterr().out
        assert "fake  Fake Mesh Format  .fake" in captured

    def test_info_without_the_flag_does_no_discovery(self, capsys):
        """Summarising one dataset must not walk every installed distribution."""
        _square_mesh().info()
        capsys.readouterr()
        assert _drivers._CACHE is None

    def test_describe_drivers_reports_capabilities(self, registered):
        """Each line carries what the driver can actually do."""
        registered(FakeDriver())
        lines = describe_drivers()
        assert any("read_mesh" in line and "fake" in line for line in lines)

    def test_mesh_names_routes_to_the_driver(self, tmp_path, registered):
        """A multi-mesh third-party format is discoverable the same way UGRID is."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        assert UgridDataset.mesh_names(target) == ["first", "second"]

    def test_mesh_names_on_netcdf_does_no_discovery(self, ugrid_convention_nc_path):
        """The same short-circuit as `read_file`."""
        UgridDataset.mesh_names(ugrid_convention_nc_path)
        assert _drivers._CACHE is None

    def test_read_all_routes_through_the_driver(self, tmp_path, registered):
        """`read_all` is `mesh_names` plus a read per name, so both must dispatch."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        result = UgridDataset.read_all(target)
        assert sorted(result) == ["first", "second"]
        assert [call[1] for call in driver.read_calls] == ["first", "second"]


class TestBareInstall:
    """Behaviour with no plugin installed at all — the commonest real case.

    Worth its own class because every other test here registers something. The paths below run
    only when the registry is genuinely empty, which is what a fresh `pip install pyramids-core`
    looks like.
    """

    def test_unsupported_message_says_none_are_installed(self, tmp_path):
        """The error reads sensibly with an empty registry rather than a dangling colon.

        Test scenario:
            With no driver registered, a file of an unknown format reports `none` for the
            installed list, still names the built-in reader, and still offers the install hint.
        """
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert "Plugin drivers   : none installed" in message, (
            f"An empty registry should say so explicitly, got: {message}"
        )
        assert "ugrid (.cdf .nc .nc3 .nc4 .netcdf)" in message, (
            "The built-in list must be every suffix dispatch accepts, not a shorter "
            f"hand-written one, got: {message}"
        )
        assert "mesh driver plugin" in message, (
            f"The install hint should still be offered, got: {message}"
        )

    def test_unknown_driver_reports_none_installed(self, tmp_path):
        """`driver=` against an empty registry names no alternatives, and says so.

        Test scenario:
            Asking for a driver on a bare install raises DriverNotExistError whose message
            reads `none` rather than listing an empty set.
        """
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(DriverNotExistError) as caught:
            UgridDataset.read_file(target, driver="selafin")
        assert "none" in str(caught.value), (
            f"Expected 'none' for an empty registry, got: {caught.value}"
        )

    def test_attach_data_on_a_bare_install_reports_none(self, tmp_path):
        """`attach_data` degrades the same way the read path does.

        Test scenario:
            With nothing registered, attaching a results file reports the empty catalogue
            instead of raising a KeyError or an IndexError from an empty candidate list.
        """
        target = tmp_path / "velocity.dat"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="none installed"):
            mesh.attach_data(target)

    def test_write_driver_on_a_bare_install_reports_none(self, tmp_path):
        """A write through a named driver fails the same way as a read.

        Test scenario:
            `to_file(driver=...)` on a bare install raises DriverNotExistError naming `none`.
        """
        mesh = _square_mesh()
        with pytest.raises(DriverNotExistError, match="none"):
            mesh.to_file(tmp_path / "mesh.slf", driver="selafin")


class TestErrorPassthrough:
    """A driver's own `UnsupportedMeshFormatError` must not be wrapped in another one.

    Dispatch attributes a driver's failure by name, which is right for an unexpected exception —
    a `struct.error` or a `KeyError` from a parser. But a driver that has already raised the
    package's own "I cannot read this" error has said everything useful, and re-wrapping turns a
    clear message into `UnsupportedMeshFormatError: The 'x' driver failed ...
    UnsupportedMeshFormatError(...)`.
    """

    def test_read_passes_an_unsupported_error_through_unwrapped(
        self, tmp_path, registered
    ):
        """A driver raising UnsupportedMeshFormatError from `read` keeps its own message.

        Test scenario:
            The driver raises UnsupportedMeshFormatError("this file is version 9"); the caller
            sees exactly that, with no "driver failed to read" wrapper around it.
        """

        class Picky(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise UnsupportedMeshFormatError(
                    "this file is version 9, which I cannot read"
                )

        registered(Picky())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert message == "this file is version 9, which I cannot read", (
            f"The driver's own message should survive unwrapped, got: {message}"
        )
        assert "driver failed to read" not in message, (
            f"An already-specific error must not be re-wrapped, got: {message}"
        )

    def test_write_passes_an_unsupported_error_through_unwrapped(
        self, tmp_path, registered
    ):
        """A driver raising UnsupportedMeshFormatError from `write` keeps its own message.

        Test scenario:
            The driver refuses to write a quad mesh and says so; the caller sees that reason,
            not a generic "driver failed to write" wrapper.
        """

        class RefusesQuads(FakeDriver):
            def write(self, dataset, path, **options):
                raise UnsupportedMeshFormatError("this format stores triangles only")

        registered(RefusesQuads())
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            mesh.to_file(tmp_path / "mesh.fake", driver="fake")
        message = str(caught.value)
        assert message == "this format stores triangles only", (
            f"The driver's own message should survive unwrapped, got: {message}"
        )

    def test_write_passes_a_driver_not_exist_error_through(self, tmp_path, registered):
        """A driver raising DriverNotExistError is not converted into a format error.

        Test scenario:
            A driver delegating to a sub-driver it cannot find raises DriverNotExistError;
            that type reaches the caller rather than being flattened into
            UnsupportedMeshFormatError, so `except DriverNotExistError` still works.
        """

        class Delegating(FakeDriver):
            def write(self, dataset, path, **options):
                raise DriverNotExistError("the sub-driver 'x' is not installed")

        registered(Delegating())
        mesh = _square_mesh()
        with pytest.raises(DriverNotExistError, match="sub-driver"):
            mesh.to_file(tmp_path / "mesh.fake", driver="fake")


class TestAttachThroughVirtualFilesystem:
    """`attach_data` with a GDAL virtual-filesystem name."""

    def test_vsi_path_has_no_suffix_to_dispatch_on(self, registered):
        """A `/vsi` name is not an OS path, so suffix detection must not be run on it.

        `Path("/vsimem/x").suffix` would read whatever follows the last dot in a virtual name,
        which is not a format hint. The resolver therefore treats a `/vsi` target as having no
        suffix, and reports that no driver claims it rather than guessing from the text.

        Test scenario:
            Attaching `/vsimem/results.res` finds no candidate — the registered driver claims
            `.res` but the virtual path contributes no suffix — and the error says so.
        """
        registered(AttachDriver())
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="no suffix"):
            mesh.attach_data("/vsimem/results.res")

    def test_attach_with_an_explicit_needs_mesh_driver_succeeds(
        self, tmp_path, registered
    ):
        """Naming a valid `needs_mesh` driver skips detection and attaches.

        The counterpart to `test_attach_refuses_a_driver_that_reads_its_own_mesh`: that one
        proves the capability gate rejects the wrong driver, this one proves it lets the right
        one through. Without it the gate's passing branch is never executed, so an inverted
        condition would still look tested.

        Test scenario:
            `attach_data(path, driver="attach")` on a file whose suffix the driver does not
            claim still attaches, because an explicit driver bypasses suffix detection.
        """
        registered(AttachDriver())
        target = tmp_path / "velocity.unclaimed"
        target.write_bytes(b"")
        result = _square_mesh().attach_data(target, driver="attach")
        assert "velocity" in result.data_variable_names, (
            f"Expected the attached variable, got: {result.data_variable_names}"
        )


class TestValidationIsContained:
    """Validation must never let a malformed plugin raise out of `mesh_drivers`.

    `_validate` reads attributes a plugin author controls, so it coerces values it has not
    type-checked — `suffix.startswith`, `set(capabilities)`. Running it outside the guarded
    block meant a plugin with `suffixes = (1,)` raised `AttributeError` straight out of
    `read_file`, breaking every non-NetCDF read and contradicting the module's documented
    "every failure is a warning, never an exception" guarantee.
    """

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("suffixes", (1,)),
            ("suffixes", 42),
            ("suffixes", None),
            ("capabilities", 7),
            ("capabilities", "read_mesh"),
            ("capabilities", None),
            ("name", 123),
            ("api_version", "one"),
        ],
    )
    def test_wrongly_typed_attribute_warns_and_does_not_raise(
        self, fake_entry_points, attribute, value
    ):
        """A plugin attribute of the wrong type is skipped with a warning, never raised.

        Args:
            attribute: The contract attribute to corrupt.
            value: A wrongly-typed value a plugin author could plausibly write.

        Test scenario:
            Discovery must survive each of these and return an empty registry, rather than
            propagating the TypeError or AttributeError that coercing the value produces.
        """

        class Malformed(FakeDriver):
            pass

        setattr(Malformed, attribute, value)
        fake_entry_points(("malformed", Malformed))
        with pytest.warns(RuntimeWarning):
            assert mesh_drivers() == {}, "A malformed driver must not register"

    def test_a_malformed_plugin_does_not_break_a_working_one(self, fake_entry_points):
        """One broken plugin costs itself, not the whole registry.

        Test scenario:
            A driver with integer suffixes registered alongside a valid driver leaves the
            valid one usable — which is the point of containing the failure.
        """

        class Malformed(FakeDriver):
            name = "malformed"
            suffixes = (1,)

        fake_entry_points(("malformed", Malformed), ("fake", FakeDriver))
        with pytest.warns(RuntimeWarning):
            resolved = mesh_drivers()
        assert set(resolved) == {"fake"}, (
            f"The valid driver should survive, got: {sorted(resolved)}"
        )

    def test_a_driver_missing_its_methods_is_rejected_at_registration(
        self, fake_entry_points
    ):
        """A driver without `can_open`/`read` is caught now, not at first use.

        Registering it and letting the AttributeError surface later reports a missing method
        as a *format* error, which sends the user looking at their file instead of the plugin.

        Test scenario:
            A class carrying every contract attribute but no methods is skipped, and the
            warning names the missing method.
        """

        class NoMethods:
            name = "nomethods"
            long_name = "No Methods"
            suffixes = (".nm",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"read_mesh"})

        fake_entry_points(("nomethods", NoMethods))
        with pytest.warns(RuntimeWarning, match="can_open"):
            assert mesh_drivers() == {}, "A method-less driver must not register"

    def test_a_needs_mesh_driver_must_implement_read_data(self, fake_entry_points):
        """The method a capability promises is the method validation requires.

        Test scenario:
            A `needs_mesh` driver without `read_data` is rejected, naming that method —
            `read` alone cannot serve a results-only format.
        """

        class NoReadData:
            name = "noreaddata"
            long_name = "No read_data"
            suffixes = (".nrd",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"needs_mesh"})

            def can_open(self, path):
                return True

        fake_entry_points(("noreaddata", NoReadData))
        with pytest.warns(RuntimeWarning, match="read_data"):
            assert mesh_drivers() == {}, "A needs_mesh driver needs read_data"

    def test_a_write_mesh_driver_must_implement_write(self, fake_entry_points):
        """Same rule for the write side.

        Test scenario:
            A driver declaring `write_mesh` without a `write` method is rejected at
            registration rather than failing mid-write with a half-created file.
        """

        class NoWrite(FakeDriver):
            capabilities = frozenset({"read_mesh", "write_mesh"})
            write = None

        fake_entry_points(("nowrite", NoWrite))
        with pytest.warns(RuntimeWarning, match="write"):
            assert mesh_drivers() == {}, "A write_mesh driver needs write"

    @pytest.mark.parametrize("name", ["ugrid", "UGRID", "", "   "])
    def test_a_reserved_or_blank_name_is_rejected(self, fake_entry_points, name):
        """A plugin must not be able to shadow the built-in reader or register nameless.

        Args:
            name: A name a plugin should not be allowed to claim.

        Test scenario:
            `ugrid` is the built-in reader's identity in `info()` listings and in `driver=`;
            a plugin taking it would make the listing ambiguous and `driver="ugrid"` resolve
            to the plugin. A blank name is unusable in `driver=` at all.
        """

        class Shadowing(FakeDriver):
            pass

        Shadowing.name = name
        fake_entry_points(("shadowing", Shadowing))
        with pytest.warns(RuntimeWarning):
            assert mesh_drivers() == {}, f"The name {name!r} must be rejected"


class TestMissingFileReporting:
    """A path that does not exist must say so, on every route.

    `read_file` documents `FileNotFoundError`, but the existence check lived only in the
    NetCDF reader, and the dispatch wrapper rewrote a driver's own `FileNotFoundError` into
    `UnsupportedMeshFormatError`. A filename typo therefore reported "no installed driver
    reads this format" — sending the user to install a plugin they already had.
    """

    def test_read_file_missing_driver_path_raises_file_not_found(
        self, tmp_path, registered
    ):
        """A missing file whose suffix a driver claims still raises FileNotFoundError.

        Test scenario:
            `mesh.fake` does not exist and the `fake` driver claims `.fake`. The error must
            name the missing file, not the format.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "typo.fake")

    def test_read_file_missing_with_explicit_driver_raises_file_not_found(
        self, tmp_path, registered
    ):
        """The explicit-driver route checks existence too.

        Test scenario:
            `driver="fake"` bypasses detection, so it needs its own existence check rather
            than inheriting the detection path's.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "typo.whatever", driver="fake")

    def test_attach_data_missing_file_raises_file_not_found(self, tmp_path, registered):
        """`attach_data` reports a missing results file as missing.

        Test scenario:
            A results file that is not there is a typo, not an unsupported format.
        """
        registered(AttachDriver())
        mesh = _square_mesh()
        with pytest.raises(FileNotFoundError):
            mesh.attach_data(tmp_path / "typo.res")

    def test_mesh_names_missing_driver_path_raises_file_not_found(
        self, tmp_path, registered
    ):
        """`mesh_names` matches `read_file` here, as its docstring claims.

        Test scenario:
            The same missing-file report on the topology-listing route.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.mesh_names(tmp_path / "typo.fake")

    def test_a_driver_raising_file_not_found_is_not_rewritten(
        self, tmp_path, registered
    ):
        """A driver's own FileNotFoundError passes through unwrapped.

        A driver may open sibling files of its own — FLO-2D and ADCIRC are multi-file
        formats — and a missing sibling is a missing file, not an unsupported format.

        Test scenario:
            The primary file exists, the driver raises FileNotFoundError for a sibling, and
            the caller sees FileNotFoundError naming that sibling.
        """

        class NeedsSibling(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise FileNotFoundError(
                    "the companion file 'domain.sibling' is missing"
                )

        registered(NeedsSibling())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="companion file"):
            UgridDataset.read_file(target)

    def test_a_directory_is_not_reported_as_an_unsupported_format(
        self, tmp_path, registered
    ):
        """A directory gets an error about being a directory, not a plugin-install hint.

        A driver is still allowed to claim a directory — Esri TIN is a directory format — so
        this covers the *unclaimed* case, which is the one that used to misreport.

        Test scenario:
            With a driver that vetoes the path, the directory falls through to the built-in
            reader, which previously produced "no installed driver reads '.fake' — pip
            install pyramids-mesh" for what is really a wrong-path mistake.
        """
        registered(VetoDriver())
        folder = tmp_path / "a_directory.fake"
        folder.mkdir()
        with pytest.raises(
            (IsADirectoryError, FileNotFoundError, ValueError)
        ) as caught:
            UgridDataset.read_file(folder)
        assert "driver plugin" not in str(caught.value), (
            f"A directory must not be reported as a missing plugin: {caught.value}"
        )


class VariablesOnlyAttachDriver:
    """A `needs_mesh` driver under the contract where `read_data` returns only new variables."""

    name = "varattach"
    long_name = "Variables-Only Results"
    suffixes = (".vres",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"needs_mesh", "read_data"})

    def __init__(self, produces=("velocity",), declared_elements=None):
        """Set up the driver.

        Args:
            produces: Names the driver claims to have read from the file.
            declared_elements: The element count the file declares, for the mismatch case.
        """
        self.produces = produces
        self.declared_elements = declared_elements
        self.lazy_seen: list[bool] = []

    def can_open(self, path):
        """Accept any offered path."""
        return True

    def read_data(self, path, dataset, *, lazy=True):
        """Return one variable per name in `produces`, after the element-count check."""
        self.lazy_seen.append(lazy)
        declared = (
            dataset.n_face if self.declared_elements is None else self.declared_elements
        )
        if declared != dataset.n_face:
            raise ValueError(
                f"{path} declares {declared} elements; the mesh has {dataset.n_face}."
            )
        return {
            name: MeshVariable(
                name=name,
                location="face",
                mesh_name=dataset.mesh_name,
                shape=(dataset.n_face,),
                _data=np.zeros(dataset.n_face),
            )
            for name in self.produces
        }


class TestAttachMergeContract:
    """`read_data` returns its own variables; pyramids owns the merge.

    The first contract had the driver return a whole merged `UgridDataset`, and pyramids
    detected collisions with `attached[name] is not self._data_variables.get(name)`. That
    encoded an undocumented by-reference requirement: a driver rebuilding the dict — which
    is what `dataclasses.replace` produces, and how pyramids' own `rename_variable` builds
    its result — tripped a spurious `ValueError` naming variables that were not colliding,
    with a remedy ("rename one side") the user could not act on. It also could not tell a
    driver that silently dropped the mesh's own variables from a well-behaved one.

    Returning only the new variables removes the ambiguity: a name already on the dataset is
    unambiguously a collision, and the driver never touches what was already there.
    """

    def test_attach_merges_the_drivers_variables_onto_the_mesh(
        self, tmp_path, registered
    ):
        """The mesh keeps its own variables and gains the driver's.

        Test scenario:
            A mesh carrying `depth` attaches a file producing `velocity`; the result has
            both, and the original is untouched.
        """
        registered(VariablesOnlyAttachDriver())
        mesh = _square_mesh()
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        result = mesh.attach_data(target)
        assert sorted(result.data_variable_names) == ["depth", "velocity"], (
            f"Expected both variables, got: {sorted(result.data_variable_names)}"
        )
        assert mesh.data_variable_names == ["depth"], (
            f"The original must be unchanged, got: {mesh.data_variable_names}"
        )

    def test_a_rebuilt_variable_is_not_a_spurious_collision(self, tmp_path, registered):
        """A driver that rebuilds objects rather than reusing them attaches cleanly.

        This is the exact shape that produced M3: identity no longer participates, so a
        driver may construct its variables however it likes.

        Test scenario:
            The driver returns freshly-built `MeshVariable` objects; attaching succeeds.
        """
        registered(VariablesOnlyAttachDriver(produces=("velocity", "salinity")))
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        result = _square_mesh().attach_data(target)
        assert sorted(result.data_variable_names) == [
            "depth",
            "salinity",
            "velocity",
        ], f"All three expected, got: {sorted(result.data_variable_names)}"

    def test_a_genuine_name_collision_is_refused(self, tmp_path, registered):
        """A driver producing a name the mesh already defines is an unambiguous collision.

        Test scenario:
            The driver claims to have read `depth`, which the mesh already has; silently
            keeping either copy would lose data, so this raises and names `depth`.
        """
        registered(VariablesOnlyAttachDriver(produces=("depth",)))
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(ValueError, match="depth"):
            mesh.attach_data(target)

    def test_attach_forwards_lazy_to_the_driver(self, tmp_path, registered):
        """`lazy` reaches `read_data`, so the capability is not inert.

        Test scenario:
            `attach_data(..., lazy=False)` is observable inside the driver.
        """
        driver = registered(VariablesOnlyAttachDriver())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        _square_mesh().attach_data(target, lazy=False)
        _square_mesh().attach_data(target, lazy=True)
        assert driver.lazy_seen == [False, True], (
            f"lazy must be forwarded verbatim, got: {driver.lazy_seen}"
        )

    def test_a_driver_returning_the_wrong_type_is_reported(self, tmp_path, registered):
        """A driver returning a dataset instead of a mapping is told so.

        The contract changed from "return the merged dataset" to "return your variables", so
        a driver written against the old shape must get a clear message rather than a
        confusing failure deep in the merge.

        Test scenario:
            `read_data` returns a `UgridDataset`; the error names the driver and the
            expected shape.
        """

        class OldStyle(VariablesOnlyAttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return dataset

        registered(OldStyle())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="mapping"):
            mesh.attach_data(target)

    def test_a_driver_returning_a_non_mesh_variable_is_reported(
        self, tmp_path, registered
    ):
        """Values in the mapping must be MeshVariable objects.

        Test scenario:
            A driver returning a bare numpy array per name is caught at the boundary, not
            stored and left to fail at first use.
        """

        class RawArrays(VariablesOnlyAttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return {"velocity": np.zeros(dataset.n_face)}

        registered(RawArrays())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="MeshVariable"):
            mesh.attach_data(target)


class TestLazyForwarding:
    """`lazy` must reach the driver, or the `lazy_data` capability is decoration.

    `read_file` had no `lazy` parameter and `_read_through_driver` called
    `driver.read(path, mesh_name=...)` without one, so a driver advertising `lazy_data` was
    always invoked with its own default. The docs told authors to honour `lazy=True`, which
    nothing ever set.
    """

    def test_read_file_forwards_lazy_to_the_driver(self, tmp_path, registered):
        """`lazy` is observable inside `driver.read`.

        Test scenario:
            Reading with `lazy=False` then `lazy=True` records both values in the driver.
        """

        class Recording(FakeDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(Recording())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target, lazy=False)
        UgridDataset.read_file(target, lazy=True)
        assert driver.lazy_seen == [False, True], (
            f"lazy must be forwarded verbatim, got: {driver.lazy_seen}"
        )

    def test_lazy_defaults_to_true(self, tmp_path, registered):
        """The default matches what the docs promise a driver will receive.

        Test scenario:
            Calling without `lazy` reaches the driver as True.
        """

        class Recording(FakeDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(Recording())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target)
        assert driver.lazy_seen == [True], (
            f"The default must arrive as True, got: {driver.lazy_seen}"
        )

    def test_lazy_is_rejected_on_the_netcdf_path(self, ugrid_convention_nc_path):
        """The built-in reader has no lazy switch, so asking for one must not be silent.

        pyramids' NetCDF variables are already lazy by construction. Accepting `lazy=False`
        and ignoring it would promise an eager read the reader does not perform.

        Test scenario:
            `read_file(..., lazy=False)` on a `.nc` file raises rather than silently
            doing a lazy read.
        """
        with pytest.raises(TypeError, match="lazy"):
            UgridDataset.read_file(ugrid_convention_nc_path, lazy=False)


class Network1dDriver(FakeDriver):
    """A driver for a 1D network format — 2DM's E2L elements, for instance."""

    name = "net1d"
    long_name = "1D Network Format"
    suffixes = (".net",)

    def read(self, path, *, mesh_name=None, lazy=True):
        """Return a two-node, one-edge network."""
        self.read_calls.append((path, mesh_name))
        return Network1d(
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 0.0]),
            edge_node_connectivity=Connectivity(
                data=np.array([[0, 1]]),
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )


class TestNetwork1dEntryPoint:
    """A driver returning `Network1d` must have somewhere to return it to.

    The dispatch error and the authoring guide both told the user to call
    `Network1d.read_file` — a method that did not exist. So the documented way to ship a 1D
    format produced an object nothing could read, and following the instructions led to an
    `AttributeError`.
    """

    def test_network1d_read_file_exists_and_reads_through_a_driver(
        self, tmp_path, registered
    ):
        """`Network1d.read_file` dispatches to a driver and returns the network.

        Test scenario:
            A driver claiming `.net` returns a two-node network; reading the file yields it
            with its nodes and edges intact.
        """
        registered(Network1dDriver())
        target = tmp_path / "channels.net"
        target.write_bytes(b"")
        network = Network1d.read_file(target)
        assert network.n_node == 2, f"Expected 2 nodes, got {network.n_node}"
        assert network.n_edge == 1, f"Expected 1 edge, got {network.n_edge}"

    def test_network1d_read_file_refuses_a_2d_mesh(self, tmp_path, registered):
        """A driver handing back a `UgridDataset` is reported, not returned.

        Test scenario:
            The mirror of `read_file`'s guard: asking for a network and receiving a surface
            mesh names the other entry point.
        """
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="UgridDataset.read_file"):
            Network1d.read_file(target)

    def test_read_file_pointing_at_network1d_names_a_method_that_exists(
        self, tmp_path, registered
    ):
        """The advice `read_file` gives must be followable.

        Test scenario:
            `UgridDataset.read_file` on a 1D format names `Network1d.read_file`, and that
            attribute resolves to a callable.
        """
        registered(Network1dDriver())
        target = tmp_path / "channels.net"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        assert "Network1d.read_file" in str(caught.value), (
            f"The error should name the 1D entry point, got: {caught.value}"
        )
        assert callable(getattr(Network1d, "read_file", None)), (
            "Network1d.read_file must exist for that advice to be followable"
        )

    def test_network1d_read_file_on_a_netcdf_path_is_refused_clearly(
        self, ugrid_convention_nc_path, registered
    ):
        """There is no built-in 1D NetCDF reader, so say that rather than failing obscurely.

        pyramids' UGRID reader builds a `Mesh2d` whatever topology dimension a file declares,
        so a 1D NetCDF has no native route. Since #1351 the fallback does consult a driver
        claiming `.nc` first, so this is the case where none is installed.

        Test scenario:
            With no driver registered, a `.nc` path raises `UnsupportedMeshFormatError`
            that says nothing claims the suffix and asks for a plugin. It also points at
            `UgridDataset.read_file` -- but as the answer to "this might be a surface mesh",
            not as a bare redirect. The loop #1351 reported is broken by
            `_read_through_driver` naming `driver=`, not by deleting the pointer, and
            deleting it left the most likely mistake with no correct advice at all.
        """
        registered()
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            Network1d.read_file(ugrid_convention_nc_path)
        message = str(caught.value)
        assert "as a 1D network" in message
        assert "none installed" in message, (
            f"the message should say no driver is installed, got:\n{message}"
        )
        assert "surface mesh" in message, (
            f"a .nc is most likely a surface mesh handed to the wrong entry point, so the "
            f"message must point at UgridDataset.read_file for that case; got:\n{message}"
        )
        assert "driver=" not in message, (
            f"with no driver claiming the suffix there is none to suggest; got:\n{message}"
        )


class TestMeshNamesContract:
    """`mesh_names` and `read_all` must behave as their docstrings claim.

    Both say dispatch mirrors `read_file`. It did not: an unclaimed suffix fell through to
    GDAL and surfaced a raw `RuntimeError` where `read_file` raises
    `UnsupportedMeshFormatError`, and a driver's own failure was neither attributed nor
    type-checked the way `read`'s is.
    """

    def test_mesh_names_unclaimed_suffix_raises_unsupported_format(
        self, tmp_path, registered
    ):
        """The same error type `read_file` raises, for the same input.

        Test scenario:
            A `.slf` file no driver claims reports the installed drivers, rather than GDAL's
            "not recognized as a supported file format".
        """
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedMeshFormatError, match="mesh driver plugin"):
            UgridDataset.mesh_names(target)

    def test_read_all_unclaimed_suffix_raises_unsupported_format(
        self, tmp_path, registered
    ):
        """`read_all` inherits it, since it delegates to `mesh_names`.

        Test scenario:
            As above, through the read-every-topology entry point.
        """
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedMeshFormatError):
            UgridDataset.read_all(target)

    def test_mesh_names_attributes_a_driver_failure(self, tmp_path, registered):
        """A driver raising inside `mesh_names` is named, as it is for `read`.

        Test scenario:
            The driver raises a parser error; the message names the driver rather than
            letting a bare `struct.error` reach the caller.
        """

        class Failing(FakeDriver):
            def mesh_names(self, path):
                raise ValueError("record length mismatch at byte 80")

        registered(Failing())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="'fake' driver failed"):
            UgridDataset.mesh_names(target)

    def test_mesh_names_rejects_a_non_list_return(self, tmp_path, registered):
        """The return shape is checked at the boundary.

        Test scenario:
            A driver returning a bare string instead of a list of names is caught here,
            rather than silently iterating into single characters downstream.
        """

        class WrongShape(FakeDriver):
            def mesh_names(self, path):
                return "mesh2d"

        registered(WrongShape())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="list of topology names"):
            UgridDataset.mesh_names(target)

    def test_mesh_names_rejects_non_string_names(self, tmp_path, registered):
        """Names must be strings, since `read_file(mesh_name=...)` takes them.

        Test scenario:
            A driver returning integers is rejected rather than producing a `read_all`
            dictionary keyed by ints.
        """

        class WrongNames(FakeDriver):
            def mesh_names(self, path):
                return [0, 1]

        registered(WrongNames())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="list of topology names"):
            UgridDataset.mesh_names(target)

    def test_read_all_resolves_the_driver_once(self, tmp_path, registered):
        """`read_all` must not re-sniff the file per topology.

        It called `mesh_names` and then `read_file` per name, each re-running detection, so
        an N-mesh file invoked `can_open` N+1 times. Passing the resolved driver name through
        makes it once.

        Test scenario:
            A two-mesh driver is sniffed exactly once for a `read_all` call.
        """
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_all(target)
        assert len(driver.sniffed) == 1, (
            f"Expected one can_open call, got {len(driver.sniffed)}"
        )


class TestRegistryRobustness:
    """The registry is process-wide shared state, so it must not be mutable or racy.

    `mesh_drivers()` is documented and public, and it returned the live cache dict — so any
    caller could inject a driver into, or clear, every other caller's registry. Discovery was
    also an unlocked check-then-set, letting two threads both run it and hold different
    instances of the same driver.
    """

    def test_mesh_drivers_returns_a_copy(self, registered):
        """Mutating the returned mapping must not reach the registry.

        Test scenario:
            Deleting from the returned dict leaves the next call unaffected.
        """
        registered(FakeDriver())
        first = mesh_drivers()
        del first["fake"]
        assert "fake" in mesh_drivers(), (
            "Mutating the returned mapping must not change the registry"
        )

    def test_mesh_drivers_returns_a_new_mapping_each_call(self, registered):
        """Successive calls hand out independent mappings.

        Test scenario:
            Two calls are equal but not the same object, so neither caller can affect the
            other through it.
        """
        registered(FakeDriver())
        first, second = mesh_drivers(), mesh_drivers()
        assert first is not second, "Each call must return its own mapping"
        assert first == second, "...but with the same contents"

    def test_a_can_open_that_raises_is_treated_as_a_veto(self, tmp_path, registered):
        """A broken `can_open` costs its own driver, not the read.

        `can_open` is documented as never raising, but a plugin can still break it — and an
        exception there propagated out of `read_file`, so one broken driver broke reads of
        every other format sharing its suffix.

        Test scenario:
            A driver whose `can_open` raises is skipped with a warning, and a working driver
            registered after it still gets the file.
        """

        class Exploding(FakeDriver):
            name = "exploding"

            def can_open(self, path):
                raise RuntimeError("the sniff itself is broken")

        driver = registered(Exploding(), FakeDriver())[1]
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.warns(RuntimeWarning, match="can_open"):
            result = UgridDataset.read_file(target)
        assert result.n_face == 1, "The working driver should have read the file"
        assert len(driver.read_calls) == 1, (
            f"The second driver should have been reached, calls: {driver.read_calls}"
        )

    def test_a_can_open_that_raises_does_not_hide_an_unreadable_file(
        self, tmp_path, registered
    ):
        """With no working driver behind it, the file still reports as unsupported.

        Test scenario:
            The only driver's `can_open` raises; the read falls through and reports the
            format, having warned about the driver.
        """

        class Exploding(FakeDriver):
            def can_open(self, path):
                raise RuntimeError("the sniff itself is broken")

        registered(Exploding())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"\x00")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(UnsupportedMeshFormatError):
                UgridDataset.read_file(target)
        sniff_warnings = [
            str(record.message)
            for record in caught
            if "can_open" in str(record.message)
        ]
        assert len(sniff_warnings) == 1, (
            f"The broken sniff should warn exactly once, got: {sniff_warnings}"
        )


class TestRemainingDispatchBranches:
    """The passthrough and fallback branches the earlier tests left unexecuted.

    Each is a behaviour the module documents, so an untested one is a claim nothing checks.
    """

    def test_a_non_string_long_name_is_rejected(self, fake_entry_points):
        """`long_name` reaches `info()` output, so it must be a string.

        Test scenario:
            A driver whose `long_name` is an int is skipped with a warning naming the
            attribute, rather than registering and rendering as `123` in the listing.
        """

        class NumericLongName(FakeDriver):
            long_name = 123

        fake_entry_points(("numeric", NumericLongName))
        with pytest.warns(RuntimeWarning, match="long_name"):
            assert mesh_drivers() == {}, "A non-string long_name must be rejected"

    def test_mesh_names_on_an_unclaimed_directory_reports_a_directory(
        self, tmp_path, registered
    ):
        """`mesh_names` matches `read_file` on the directory case too.

        Test scenario:
            A directory no driver claims raises `IsADirectoryError` rather than telling the
            user to install a plugin.
        """
        registered(VetoDriver())
        folder = tmp_path / "a_directory.fake"
        folder.mkdir()
        with pytest.raises(IsADirectoryError) as caught:
            UgridDataset.mesh_names(folder)
        assert "driver plugin" not in str(caught.value), (
            f"A directory must not be reported as a missing plugin: {caught.value}"
        )

    def test_mesh_names_passes_an_oserror_through(self, tmp_path, registered):
        """A driver's own I/O failure while listing topologies stays an I/O failure.

        Test scenario:
            The driver cannot open a sibling index file; the caller sees `OSError`, not a
            format error.
        """

        class MissingIndex(FakeDriver):
            def mesh_names(self, path):
                raise OSError("the companion index 'domain.idx' is missing")

        registered(MissingIndex())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(OSError, match="companion index") as caught:
            UgridDataset.mesh_names(target)
        assert not isinstance(caught.value, UnsupportedMeshFormatError), (
            "A missing sibling file must not be reported as an unsupported format"
        )

    def test_attach_data_attributes_an_unexpected_driver_failure(
        self, tmp_path, registered
    ):
        """A driver bug inside `read_data` is attributed, like `read`'s and `write`'s.

        `ValueError` is deliberately excluded from attribution — it is how the element-count
        mismatch is reported — so this uses a different exception to reach the wrapper.

        Test scenario:
            The driver raises `KeyError`; the message names the driver.
        """

        class Buggy(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                raise KeyError("TIMEUNITS")

        registered(Buggy())
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="'attach' driver failed"):
            mesh.attach_data(target)

    def test_attach_data_passes_an_oserror_through(self, tmp_path, registered):
        """And a filesystem failure there is not a format failure either.

        Test scenario:
            The driver raises `OSError`; the caller sees it unwrapped.
        """

        class DiskTrouble(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                raise OSError("Input/output error")

        registered(DiskTrouble())
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(OSError, match="Input/output error") as caught:
            mesh.attach_data(target)
        assert not isinstance(caught.value, UnsupportedMeshFormatError), (
            "An I/O error must not be reported as an unsupported format"
        )


class TestDiscoveryDoesNotDeadlock:
    """Plugin code must never run while the registry's lock is held.

    Round 1 added a `threading.Lock` around the whole of discovery, including
    `entry_point.load()()`. A plugin that consults the registry while constructing — to check
    whether a companion driver is installed, say — then re-entered a non-reentrant lock and
    hung forever, and the hung thread blocked every later reader too. That inverted the
    module's own guarantee: instead of one broken plugin being skipped with a warning, one
    plugin froze the process.

    `RLock` is not the fix. It would let the nested call recurse into discovery rather than
    deadlock, which is a different wrong answer.
    """

    def test_a_plugin_using_the_registry_while_constructing_does_not_hang(
        self, fake_entry_points
    ):
        """Discovery completes when a plugin calls `mesh_drivers()` from `__init__`.

        Run on a worker thread with a timeout, because the failure mode is a hang: asserting
        on a return value in the main thread would simply never finish.

        Test scenario:
            A driver whose constructor calls `mesh_drivers()` is discovered, and the thread
            doing the discovery returns within the timeout.
        """

        class NestedUse(FakeDriver):
            name = "nested"

            def __init__(self):
                super().__init__()
                mesh_drivers()

        fake_entry_points(("nested", NestedUse))
        clear_driver_cache()
        finished = threading.Event()
        failures: list[BaseException] = []

        def discover():
            try:
                mesh_drivers()
            except BaseException as error:  # noqa: BLE001 - recording whatever happens
                failures.append(error)
            finished.set()

        worker = threading.Thread(target=discover, daemon=True)
        worker.start()
        assert finished.wait(timeout=10.0), (
            "Discovery deadlocked: a plugin that uses the registry while constructing must "
            "not hang, which means plugin code cannot run under the lock"
        )
        assert not failures, f"Discovery raised: {failures}"
        # And it must LOAD, not be dropped. Unlocking discovery alone turned the nested call
        # into infinite recursion, which `_instantiate` caught as "could not be loaded" --
        # finite, but it silently discarded a plugin for asking a reasonable question.
        assert "nested" in mesh_drivers(), (
            f"The plugin should be registered, not skipped; got {sorted(mesh_drivers())}"
        )

    def test_a_nested_call_sees_what_has_been_resolved_so_far(self, fake_entry_points):
        """The re-entrant answer is defined, not accidental.

        Test scenario:
            A plugin records what `mesh_drivers()` returns from inside its own constructor.
            It gets a mapping — the drivers resolved before it — rather than recursing.
        """
        observed: list[object] = []

        class Observer(FakeDriver):
            name = "observer"

            def __init__(self):
                super().__init__()
                observed.append(mesh_drivers())

        fake_entry_points(("observer", Observer))
        clear_driver_cache()
        mesh_drivers()
        assert len(observed) == 1, (
            f"The constructor should have run once, got {len(observed)}"
        )
        assert isinstance(observed[0], dict), (
            f"A nested call must return a mapping, got {type(observed[0]).__name__}"
        )

    def test_every_caller_sees_the_same_driver_instance(self, fake_entry_points):
        """Concurrent callers agree on the object, which is the property that matters.

        Two threads may both run discovery — it is deliberately unlocked — but only one
        result is published, so a driver holding a parsed-format cache is not halved by
        handing out two instances. This replaces a test that asserted `__init__` ran exactly
        once, which the unlocked discovery no longer guarantees and which could pass with the
        lock removed entirely.

        Test scenario:
            Sixteen threads released together each take the driver from `mesh_drivers()`;
            every one holds the same object.
        """
        fake_entry_points(("counted", FakeDriver))
        clear_driver_cache()
        seen: list[object] = []
        barrier = threading.Barrier(16)

        def grab():
            barrier.wait()
            seen.append(mesh_drivers()["fake"])

        threads = [threading.Thread(target=grab) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(seen) == 16, (
            f"Every thread should have recorded one driver, got {len(seen)}"
        )
        assert len({id(driver) for driver in seen}) == 1, (
            "All callers must share one published instance"
        )

    def test_clear_driver_cache_is_safe_while_readers_run(self, fake_entry_points):
        """Clearing the cache concurrently must not expose a half-published registry.

        Test scenario:
            Readers and a clearer run together; every read returns a mapping (empty or
            populated) rather than raising.
        """
        fake_entry_points(("counted", FakeDriver))
        results: list[object] = []
        errors: list[BaseException] = []

        def read():
            for _ in range(50):
                try:
                    results.append(mesh_drivers())
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)

        def clear():
            for _ in range(50):
                clear_driver_cache()

        threads = [threading.Thread(target=read), threading.Thread(target=clear)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, f"Concurrent clear/read raised: {errors[:3]}"
        assert all(isinstance(result, dict) for result in results), (
            "Every read must return a mapping"
        )


class TestContractTruth:
    """The validation and the documented contract must agree with the code."""

    def test_a_driver_without_mesh_names_is_rejected(self, fake_entry_points):
        """`mesh_names` is called unconditionally, so it is required unconditionally.

        Validation required `read`, `read_data` and `write` per capability but omitted
        `mesh_names` — the one method pyramids calls on every driver. A driver lacking it
        registered, and `mesh_names`/`read_all` then reported "the 'nmn' driver failed to
        list the topologies … AttributeError": precisely the misattribution that function's
        docstring says it prevents.

        Test scenario:
            A driver with `can_open` and `read` but no `mesh_names` is skipped at
            registration, and the warning names the missing method.
        """

        class NoMeshNames:
            name = "nmn"
            long_name = "No mesh_names"
            suffixes = (".nmn",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"read_mesh"})

            def can_open(self, path):
                return True

            def read(self, path, *, mesh_name=None, lazy=True):
                return None

        fake_entry_points(("nmn", NoMeshNames))
        with pytest.warns(RuntimeWarning, match="mesh_names"):
            assert mesh_drivers() == {}, "A driver without mesh_names must not register"

    def test_driver_ugrid_reaches_the_builtin_reader(
        self, ugrid_convention_nc_path, registered
    ):
        """`driver="ugrid"` names the built-in reader, as RESERVED_NAMES implies.

        `ugrid` is reserved precisely because it identifies pyramids' own reader in listings
        and in `driver=`. But asking for it raised `DriverNotExistError … pip install
        pyramids-mesh` — telling the user to install a plugin to reach a reader that ships
        with pyramids.

        Test scenario:
            Reading a UGRID file with `driver="ugrid"` succeeds and returns the mesh.
        """
        registered(FakeDriver())
        result = UgridDataset.read_file(ugrid_convention_nc_path, driver="ugrid")
        assert result.n_node > 0, "The built-in reader should have read the file"

    def test_driver_ugrid_bypasses_a_plugin_claiming_the_suffix(
        self, tmp_path, registered, ugrid_convention_nc_path
    ):
        """Naming the built-in reader overrides a plugin that would otherwise claim the file.

        Test scenario:
            A driver claims `.fake`; a UGRID file under that suffix read with
            `driver="ugrid"` goes to the built-in reader, and the plugin is never asked.
        """
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        result = UgridDataset.read_file(target, driver="ugrid")
        assert result.n_node > 0, "The built-in reader should have read it"
        assert driver.read_calls == [], (
            f"The plugin must not be consulted, got: {driver.read_calls}"
        )

    def test_to_file_driver_ugrid_writes_netcdf(self, tmp_path, registered):
        """The write side accepts it too, for symmetry.

        Test scenario:
            `to_file(driver="ugrid")` produces a readable UGRID NetCDF file.
        """
        registered(FakeDriver())
        target = tmp_path / "mesh.nc"
        _square_mesh().to_file(target, driver="ugrid")
        assert UgridDataset.read_file(target).n_face == 1, (
            "driver='ugrid' should write through the built-in writer"
        )


class TestReadAllResolution:
    """`read_all` resolves once on every route, forwards `lazy`, and refuses bad name lists."""

    def test_read_all_sniffs_once_on_the_unclaimed_route(self, tmp_path, registered):
        """The resolve-once fix must cover the fallback route too.

        Passing a driver *name* through only short-circuited the route where detection had
        found something. With nothing claiming the suffix the name was `None`, so `mesh_names`
        and every per-topology read re-ran full detection — two `can_open` calls for one
        `read_all`, growing per topology.

        Test scenario:
            A vetoing driver on a `.fake` path: `can_open` is asked exactly once even though
            the read then falls through to the built-in reader.
        """
        driver = registered(VetoDriver())
        target = tmp_path / "model.fake"
        target.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedMeshFormatError):
            UgridDataset.read_all(target)
        assert len(driver.sniffed) == 1, (
            f"Expected one can_open call on the fallback route, got {len(driver.sniffed)}"
        )

    def test_read_all_forwards_lazy(self, tmp_path, registered):
        """A multi-topology third-party format must be readable eagerly.

        Test scenario:
            `read_all(..., lazy=False)` reaches the driver as False for every topology.
        """

        class Recording(FakeDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(Recording())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_all(target, lazy=False)
        assert driver.lazy_seen == [False, False], (
            f"lazy must reach every per-topology read, got: {driver.lazy_seen}"
        )

    def test_duplicate_topology_names_are_refused(self, tmp_path, registered):
        """`read_all` keys by name, so a duplicate would silently drop a topology.

        Test scenario:
            A driver reporting the same name twice is refused, naming the duplicate.
        """

        class Duplicating(FakeDriver):
            def mesh_names(self, path):
                return ["mesh2d", "mesh2d"]

        registered(Duplicating())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="duplicate"):
            UgridDataset.read_all(target)

    def test_an_empty_topology_list_is_refused(self, tmp_path, registered):
        """Claiming a file and then reporting nothing in it is a driver bug.

        Test scenario:
            An empty list is refused rather than turned into an empty `read_all` result.
        """

        class Empty(FakeDriver):
            def mesh_names(self, path):
                return []

        registered(Empty())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="no topologies"):
            UgridDataset.mesh_names(target)

    def test_a_plugin_bug_in_read_data_is_attributed(self, tmp_path, registered):
        """A `ValueError` that is not about the mesh is a plugin bug, and is named.

        The element-count mismatch is reported as `ValueError` and must pass through
        untouched — it is the one check standing between the user and a silently wrong
        dataset. But waving the whole class through left genuine plugin bugs unattributed.

        Test scenario:
            A driver raising `ValueError("invalid literal for int()")` is attributed to the
            driver, while the element-count message still passes through verbatim.
        """

        class Buggy(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                raise ValueError("invalid literal for int() with base 10: 'x'")

        registered(Buggy())
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="'attach' driver failed"):
            mesh.attach_data(target)

    def test_the_element_count_message_still_passes_through(self, tmp_path, registered):
        """The mismatch report is never reworded or attributed.

        Test scenario:
            The driver's own "declares N elements; the mesh has M" reaches the caller as a
            plain ValueError.
        """
        registered(AttachDriver(declared_elements=4096))
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(ValueError, match="4096") as caught:
            mesh.attach_data(target)
        assert not isinstance(caught.value, UnsupportedMeshFormatError), (
            "The element-count mismatch must not be dressed up as a format error"
        )

    def test_non_string_variable_keys_are_refused(self, tmp_path, registered):
        """Keys are variable names, so they must be strings.

        Mixed-type keys also made the collision check's `sorted()` raise `TypeError`.

        Test scenario:
            A driver returning an int key is refused, naming the key.
        """

        class IntKeys(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return {
                    0: MeshVariable(
                        name="v",
                        location="face",
                        mesh_name=dataset.mesh_name,
                        shape=(dataset.n_face,),
                        _data=np.zeros(dataset.n_face),
                    )
                }

        registered(IntKeys())
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        mesh = _square_mesh()
        with pytest.raises(UnsupportedMeshFormatError, match="non-string key"):
            mesh.attach_data(target)


class TestPublishRace:
    """The branch taken when another thread published the registry first."""

    def test_a_losing_discoverer_adopts_the_published_registry(
        self, monkeypatch, fake_entry_points
    ):
        """Two threads may both discover; the loser must adopt the winner's mapping.

        Discovery is deliberately unlocked, so two threads can both run it. Only the first
        result is published and the second is discarded — which is what keeps every caller on
        one instance. Simulated deterministically by having `_discover` publish a different
        mapping as a side effect, standing in for the thread that got there first.

        Test scenario:
            A discoverer whose result loses the race returns the published mapping, not its
            own.
        """
        winner = FakeDriver()
        loser = FakeDriver()

        def discover_then_lose():
            _drivers._CACHE = {"fake": winner}
            return {"fake": loser}

        fake_entry_points(("fake", FakeDriver))
        clear_driver_cache()
        monkeypatch.setattr(_drivers, "_discover", discover_then_lose)
        assert mesh_drivers()["fake"] is winner, (
            "The already-published registry must win, so every caller shares one instance"
        )


class TestAttachResolutionOrder:
    """Attach resolution walks candidates until one claims the file."""

    def test_a_vetoing_needs_mesh_driver_is_skipped(self, tmp_path, registered):
        """A `needs_mesh` driver that declines the file does not block the next one.

        Both drivers claim the suffix; only the second recognises the content. The loop must
        continue past the first rather than give up or pick it anyway.

        Test scenario:
            A vetoing results driver registered before a working one; the attach succeeds
            through the second, and both were asked.
        """

        class VetoAttach(AttachDriver):
            name = "vetoattach"

            def can_open(self, path):
                self.sniffed.append(path)
                return False

        first = VetoAttach()
        second = AttachDriver()
        second.suffixes = (".res",)
        registered(first, second)
        target = tmp_path / "v.res"
        target.write_bytes(b"")
        result = _square_mesh().attach_data(target)
        assert "velocity" in result.data_variable_names, (
            f"The second driver should have attached, got: {result.data_variable_names}"
        )
        assert first.sniffed == [str(target)], (
            f"The first driver should have been asked and declined, got: {first.sniffed}"
        )


class NetcdfDriver(FakeDriver):
    """A driver claiming the NetCDF suffix, like a 3Di or TUFLOW FV reader.

    The case the fallback exists for: the built-in reader opens the file as NetCDF, finds no
    UGRID topology, and only then is this consulted.
    """

    name = "nc3di"
    long_name = "3Di results (fake)"
    suffixes = (".nc",)
    capabilities = frozenset({"read_mesh", "read_data"})


def _plain_netcdf(path: str) -> None:
    """Write a NetCDF file that opens but declares no UGRID topology.

    A 4x4 single-band raster is enough: GDAL opens it through the multidimensional driver, so
    the built-in reader gets as far as ``parse_ugrid_topology`` and raises
    :class:`NoMeshTopologyError` rather than a GDAL open error -- which is the failure the
    fallback keys on. This is what a 3Di ``.nc`` looks like to the built-in reader.

    Args:
        path: Where to write the file.
    """
    driver = gdal.GetDriverByName("netCDF")
    dataset = driver.Create(path, 4, 4, 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).Fill(1.0)
    dataset = None


class TestNetcdfDriverFallback:
    """A NetCDF-suffixed file the built-in reader cannot read falls back to a driver.

    The behaviour #1327 adds: a ``.nc`` that opens but carries no UGRID topology -- a 3Di or
    TUFLOW FV result -- reaches a driver claiming ``.nc``, while a real UGRID read still pays
    nothing and the narrow trigger keeps every other failure on its old path.
    """

    def test_netcdf_with_no_topology_dispatches_to_the_driver(
        self, tmp_path, registered
    ):
        """A `.nc` the built-in reader rejects reaches a driver claiming the suffix."""
        driver = registered(NetcdfDriver())
        target = tmp_path / "results_3di.nc"
        _plain_netcdf(str(target))
        result = UgridDataset.read_file(target)
        assert result.n_face == 1, "the driver's dataset should be returned"
        assert driver.sniffed == [str(target)], (
            f"the driver's can_open should have been consulted once, got: {driver.sniffed}"
        )
        assert driver.read_calls == [(str(target), None)], (
            f"the driver's read should have been called, got: {driver.read_calls}"
        )

    def test_real_ugrid_read_never_consults_a_netcdf_driver(
        self, tmp_path, registered, ugrid_convention_nc_path
    ):
        """The fallback fires only after the built-in read fails, so a real UGRID read is untouched.

        This is the performance contract: a driver claiming `.nc` is installed, yet a genuine
        UGRID file reads without the driver ever being asked.
        """
        driver = registered(NetcdfDriver())
        target = tmp_path / "mesh.nc"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        UgridDataset.read_file(target)
        assert driver.sniffed == [], (
            f"a successful UGRID read must not consult the driver, got: {driver.sniffed}"
        )
        assert driver.read_calls == [], (
            f"a successful UGRID read must not call the driver, got: {driver.read_calls}"
        )

    def test_driver_kwarg_still_wins_outright_on_netcdf(self, tmp_path, registered):
        """An explicit `driver=` reaches the driver directly, with no built-in attempt first."""
        driver = registered(NetcdfDriver())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        result = UgridDataset.read_file(target, driver="nc3di")
        assert result.n_face == 1
        assert driver.sniffed == [], (
            f"an explicit driver skips can_open, got: {driver.sniffed}"
        )
        assert driver.read_calls == [(str(target), None)]

    def test_unclaimed_netcdf_still_raises_the_original_value_error(
        self, tmp_path, registered
    ):
        """No driver claims `.nc`, so a non-UGRID NetCDF keeps raising the ValueError it always did.

        The error type for a plain NetCDF with no plugin installed is unchanged: existing
        callers that `except ValueError` on "No UGRID mesh topology" keep working.
        """
        registered(FakeDriver())  # claims `.fake`, not `.nc`
        target = tmp_path / "plain.nc"
        _plain_netcdf(str(target))
        with pytest.raises(ValueError, match="No UGRID mesh topology") as caught:
            UgridDataset.read_file(target)
        assert not isinstance(caught.value, FileFormatNotSupportedError), (
            "an unclaimed NetCDF must stay a ValueError, not become UnsupportedMeshFormatError"
        )

    def test_a_wrong_mesh_name_does_not_fall_back(
        self, tmp_path, registered, ugrid_convention_nc_path
    ):
        """A real UGRID file with a bad `mesh_name` is a different failure and must not dispatch.

        The fallback keys on `NoMeshTopologyError` alone; "no topology named X" is a plain
        ValueError, so the driver is never consulted for it.
        """
        driver = registered(NetcdfDriver())
        target = tmp_path / "mesh.nc"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        with pytest.raises(ValueError, match="No UGRID topology named"):
            UgridDataset.read_file(target, mesh_name="does_not_exist")
        assert driver.sniffed == [], (
            f"a wrong mesh_name must not reach the driver, got: {driver.sniffed}"
        )

    def test_mesh_names_falls_back_to_the_driver(self, tmp_path, registered):
        """`mesh_names` shares the fallback, so a driver-backed NetCDF lists its topologies."""
        registered(NetcdfDriver())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        assert UgridDataset.mesh_names(target) == ["first", "second"]

    def test_read_all_falls_back_to_the_driver(self, tmp_path, registered):
        """`read_all` shares the fallback, so every topology of a driver-backed NetCDF is read."""
        registered(NetcdfDriver())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        result = UgridDataset.read_all(target)
        assert sorted(result) == ["first", "second"], (
            f"read_all should route through the driver's mesh_names, got: {sorted(result)}"
        )

    def test_mesh_names_reraises_when_no_driver_claims_the_netcdf(
        self, tmp_path, registered
    ):
        """`mesh_names` on a non-UGRID NetCDF no driver claims keeps raising, not swallowing."""
        registered(FakeDriver())  # claims `.fake`, not `.nc`
        target = tmp_path / "plain.nc"
        _plain_netcdf(str(target))
        with pytest.raises(ValueError, match="No UGRID mesh topology"):
            UgridDataset.mesh_names(target)

    def test_read_all_reraises_when_no_driver_claims_the_netcdf(
        self, tmp_path, registered
    ):
        """`read_all` on a non-UGRID NetCDF no driver claims keeps raising, not swallowing."""
        registered(FakeDriver())  # claims `.fake`, not `.nc`
        target = tmp_path / "plain.nc"
        _plain_netcdf(str(target))
        with pytest.raises(ValueError, match="No UGRID mesh topology"):
            UgridDataset.read_all(target)

    def test_fallback_skips_a_vetoing_driver_for_the_next(self, tmp_path, registered):
        """Among drivers sharing `.nc`, one whose `can_open` refuses is passed over for the next."""

        class VetoNetcdf(NetcdfDriver):
            name = "ncveto"

            def can_open(self, path):
                self.sniffed.append(path)
                return False

        veto = VetoNetcdf()
        accept = NetcdfDriver()
        registered(veto, accept)
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        result = UgridDataset.read_file(target)
        assert result.n_face == 1, "the second, accepting driver should have read it"
        assert veto.sniffed == [str(target)], (
            "the vetoing driver should have been asked first"
        )
        assert accept.read_calls == [(str(target), None)], (
            f"the accepting driver should have read it, got: {accept.read_calls}"
        )

    def test_fallback_driver_ignores_a_non_netcdf_suffix(self, registered):
        """The fallback is for NetCDF suffixes only; any other suffix resolves to no driver.

        Guards the branch reached when the built-in reader's ``NoMeshTopologyError`` comes from
        the step-4 fall-through (a non-NetCDF suffix nothing claimed), where re-running
        discovery would be pointless -- detection already found nothing.
        """
        registered(NetcdfDriver())  # claims `.nc`
        assert _netcdf_fallback_driver("model.slf", None) is None, (
            "a non-NetCDF suffix must not resolve to a NetCDF driver"
        )

    def test_reserved_driver_name_does_not_fall_back(self, tmp_path, registered):
        """`driver="ugrid"`/`"netcdf"` forces the built-in reader and must never reach a plugin.

        A reserved name resolves to the native branch, but an explicit `driver=` request is
        documented never to fall back, so hitting no topology must raise, not dispatch.
        """
        driver = registered(NetcdfDriver())
        target = tmp_path / "plain.nc"
        _plain_netcdf(str(target))
        for reserved in ("ugrid", "netcdf"):
            with pytest.raises(ValueError, match="No UGRID mesh topology"):
                UgridDataset.read_file(target, driver=reserved)
        assert driver.read_calls == [], (
            f"a reserved driver name must not consult a plugin, got: {driver.read_calls}"
        )

    def test_fallback_declines_a_vsi_path(self, registered):
        """A `/vsi` path stays on GDAL, as detection keeps it; the fallback must not feed a plugin one.

        `_resolve_read_driver` short-circuits `/vsi` to the built-in reader without detection, so
        the fallback has to decline it too or a plugin would run on a path it never sees otherwise.
        """
        registered(NetcdfDriver())  # claims `.nc`
        assert _netcdf_fallback_driver("/vsimem/results.nc", None) is None, (
            "a /vsi NetCDF path must not resolve to a plugin via the fallback"
        )

    def test_needs_mesh_driver_on_netcdf_points_to_attach_data(
        self, tmp_path, registered
    ):
        """A results-only driver claiming `.nc` gets the same attach_data guidance detection gives.

        Reached by the fallback rather than detection, a `needs_mesh` driver must still be
        refused with the actionable "read the mesh first" message, not whatever its own `read`
        happens to raise.
        """

        class NetcdfAttach(AttachDriver):
            name = "ncattach"
            long_name = "3Di results (results-only)"
            suffixes = (".nc",)

        registered(NetcdfAttach())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        with pytest.raises(
            UnsupportedMeshFormatError, match="carry values but no mesh"
        ):
            UgridDataset.read_file(target)

    def test_lazy_false_reaches_a_driver_backed_netcdf(self, tmp_path, registered):
        """`lazy=False` is forwarded to a `.nc` driver as False, not rejected before the fallback.

        The built-in reader cannot do an eager read, but a driver that claims the file can, so
        the `lazy=False` guard must not fire ahead of the fallback -- `read_all` already forwards
        `lazy` to the driver, and the two entry points must agree. The driver records the `lazy`
        it received, so this proves the value is forwarded, not merely that the driver was reached.
        """

        class RecordingNetcdf(NetcdfDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(RecordingNetcdf())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        result = UgridDataset.read_file(target, lazy=False)
        assert result.n_face == 1, "the driver should have read the file"
        assert driver.lazy_seen == [False], (
            f"lazy=False must be forwarded to the driver, got: {driver.lazy_seen}"
        )

    def test_lazy_false_still_rejected_for_a_real_ugrid_read(
        self, tmp_path, ugrid_convention_nc_path
    ):
        """A genuine UGRID file still rejects `lazy=False`: the built-in reader has no eager mode."""
        target = tmp_path / "mesh.nc"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        with pytest.raises(TypeError, match="lazy=False"):
            UgridDataset.read_file(target, lazy=False)

    def test_read_all_rejects_lazy_false_on_a_native_ugrid_read(
        self, tmp_path, ugrid_convention_nc_path
    ):
        """`read_all(lazy=False)` on a genuine UGRID file raises, exactly as `read_file` does.

        The built-in reader has no eager mode, so the two entry points must agree rather than
        `read_all` silently dropping `lazy=False` and returning lazy datasets.
        """
        target = tmp_path / "mesh.nc"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        with pytest.raises(TypeError, match="lazy=False"):
            UgridDataset.read_all(target, lazy=False)

    def test_mesh_names_needs_mesh_driver_points_to_attach_data(
        self, tmp_path, registered
    ):
        """`mesh_names` fires the needs_mesh guard on the fallback route, like `read_file`."""

        class NetcdfAttach(AttachDriver):
            name = "ncattach"
            long_name = "3Di results (results-only)"
            suffixes = (".nc",)

        registered(NetcdfAttach())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        with pytest.raises(
            UnsupportedMeshFormatError, match="carry values but no mesh"
        ):
            UgridDataset.mesh_names(target)

    def test_read_all_needs_mesh_driver_points_to_attach_data(
        self, tmp_path, registered
    ):
        """`read_all` fires the needs_mesh guard on the fallback route, like `read_file`."""

        class NetcdfAttach(AttachDriver):
            name = "ncattach"
            long_name = "3Di results (results-only)"
            suffixes = (".nc",)

        registered(NetcdfAttach())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        with pytest.raises(
            UnsupportedMeshFormatError, match="carry values but no mesh"
        ):
            UgridDataset.read_all(target)

    def test_mesh_names_and_read_all_never_consult_a_driver_on_a_ugrid_read(
        self, tmp_path, registered, ugrid_convention_nc_path
    ):
        """The perf contract holds for `mesh_names`/`read_all` too: a real UGRID read is untouched.

        Asserts the no-discovery guarantee for the other two entry points, not just `read_file`:
        a driver claiming `.nc` is installed, yet a genuine UGRID file never reaches it.
        """
        driver = registered(NetcdfDriver())
        target = tmp_path / "mesh.nc"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        UgridDataset.mesh_names(target)
        UgridDataset.read_all(target)
        assert driver.sniffed == [], (
            f"a genuine UGRID read must not consult the driver, got: {driver.sniffed}"
        )
        assert driver.read_calls == [], (
            f"a genuine UGRID read must not call the driver, got: {driver.read_calls}"
        )

    def test_mesh_names_and_read_all_reach_the_driver_can_open(
        self, tmp_path, registered
    ):
        """The fallback actually sniffs on these routes, not only returns the right names.

        Pairs with the perf-contract test: when the file is non-UGRID the driver's `can_open`
        is consulted, so the no-consult assertion above means something.
        """
        driver = registered(NetcdfDriver())
        target = tmp_path / "results.nc"
        _plain_netcdf(str(target))
        UgridDataset.mesh_names(target)
        UgridDataset.read_all(target)
        assert driver.sniffed == [str(target), str(target)], (
            f"can_open should be consulted on each entry point, got: {driver.sniffed}"
        )

    def test_no_mesh_topology_error_is_publicly_importable(self):
        """`NoMeshTopologyError` is exported from the package, so callers can catch it by name.

        The subclass exists partly so a caller can distinguish "no UGRID topology" from other
        `ValueError`s; that is only usable if it is reachable without the private `_drivers` path.
        """
        from pyramids.netcdf.ugrid import NoMeshTopologyError as Exported

        assert Exported is NoMeshTopologyError, (
            "the package must re-export the same class"
        )

    def test_the_fallback_trigger_is_a_value_error_subclass(self):
        """`NoMeshTopologyError` stays an `except ValueError` for callers that never registered one."""
        assert issubclass(NoMeshTopologyError, ValueError)
