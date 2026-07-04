"""Shared fixtures and the capability registry for the ground-up netcdf-subpackage suite.

Each NetCDF sample file in ``tests/data/netcdf/`` is a distinct structural shape (convention,
dimensionality, dtype mix, grid type). ``SAMPLES`` tags every file with the properties that decide which
tests apply; ``files_with(*flags)`` selects the subset carrying all the given flags.

Tests parametrize via the ``sample_name`` parameter:

- declaring ``sample_name`` runs the test over **all** sample files (universal/invariant tier);
- adding ``@pytest.mark.samples("packed", "level")`` narrows it to the files carrying those capability
  flags (capability tier). An empty subset is a hard collection error, never a silent skip.

See ``planning/netcdf/testing/netcdf-test-plan.md`` for the full model.
"""

from collections import Counter
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "netcdf"

# Shared sample-file name constants — imported by individual test modules to avoid duplication.
TOS = "cf__7v__1d3-2d3-3d1.nc"   # tos(time=24, lat=170, lon=180), EPSG:4326, non-square cells
RHUM = "coards__5v__1d4-4d1.nc"  # rhum(time=12, level=4, lat=37, lon=72)
AIR = "coards__4v__1d3-3d1.nc"   # air(time, lat, lon) — has a concat 'time' dim
MESH = "ugrid__6v__1d5-2d1.nc"   # quad-hexagon UGRID mesh: 16 nodes, 4 faces

# Capability registry — the single source of truth for which files exercise which behaviours.
#
# Flags: convention(cf|coards|none) · gridded · packed · time · level · fourd · curvilinear · staggered ·
#        groups · nc4 · string_vars · multivar · bounds · ugrid · labeled.
#
# ``convention`` matches the file's on-disk ``Conventions`` attribute (``none`` = no attribute declared).
SAMPLES = {
    "none__1v__1d1.nc": {"convention": "none"},
    "cf__7v__1d3-2d3-3d1.nc": {
        "convention": "cf", "gridded": True, "time": True, "bounds": True,
    },
    "coards__4v__1d3-3d1.nc": {
        "convention": "coards", "gridded": True, "time": True, "packed": True,
    },
    "cf__12v__1d4-2d5-3d2-4d1.nc": {
        "convention": "cf", "gridded": True, "time": True, "level": True, "fourd": True,
        "multivar": True, "bounds": True,
    },
    "cf__20v__1d3-3d17.nc": {
        "convention": "cf", "gridded": True, "time": True, "packed": True, "multivar": True,
    },
    "cf__48v__1d17-3d21-4d10.nc": {
        "convention": "cf", "gridded": True, "time": True, "level": True, "fourd": True,
        "multivar": True, "string_vars": True,
    },
    "coards__5v__1d4-4d1.nc": {
        "convention": "coards", "gridded": True, "time": True, "level": True, "fourd": True,
        "packed": True,
    },
    "cf__40v__1d28-2d9-3d3__nc4.nc": {
        "convention": "cf", "nc4": True, "multivar": True,
    },
    "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc": {
        "convention": "cf", "gridded": True, "time": True, "level": True, "fourd": True,
        "curvilinear": True,
    },
    "none__4v__1d1-2d2-3d1__curv.nc": {
        "convention": "none", "gridded": True, "time": True, "curvilinear": True,
    },
    "none__17v__1d1-2d5-3d6-4d5__stag-str.nc": {
        "convention": "none", "gridded": True, "time": True, "level": True, "fourd": True,
        "staggered": True, "string_vars": True, "multivar": True,
    },
    "none__5v__1d2-2d2-3d1__curv.nc": {
        "convention": "none", "gridded": True, "curvilinear": True, "string_vars": True,
        "multivar": True,
    },
    "none__111v__1d96-2d13-3d2__str.nc": {
        "convention": "none", "string_vars": True, "multivar": True, "labeled": True,
    },
    "none__11v__1d11.nc": {
        "convention": "none", "time": True, "multivar": True,
    },
    "none__35v__1d35__groups-nc4.nc": {
        "convention": "none", "groups": True, "nc4": True, "string_vars": True, "multivar": True,
    },
    "ugrid__6v__1d5-2d1.nc": {"ugrid": True},
    "ugrid__1v__3d1.nc": {"ugrid": True, "time": True, "level": True},
    "ugrid__1v__1d1.nc": {"ugrid": True},
}

SAMPLE_NAMES = list(SAMPLES)


def files_with(*flags):
    """Return the sample filenames whose registry entry has every flag in ``flags`` truthy."""
    return [name for name, caps in SAMPLES.items() if all(caps.get(flag) for flag in flags)]


def parse_structural_name(filename):
    """Decode a structural filename into ``(convention, nvars, {rank: count}, [features])``.

    Example: ``cf__12v__1d4-2d5-3d2-4d1.nc`` ->
    ``("cf", 12, {1: 4, 2: 5, 3: 2, 4: 1}, [])``.
    """
    parts = filename[:-3].split("__")
    convention = parts[0]
    nvars = int(parts[1].rstrip("v"))
    histogram = {}
    for token in parts[2].split("-"):
        rank, count = token.split("d")
        histogram[int(rank)] = int(count)
    features = parts[3].split("-") if len(parts) > 3 else []
    return convention, nvars, histogram, features


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "samples(*flags): restrict a `sample_name`-parametrized test to files carrying these "
        "capability flags (see SAMPLES in conftest).",
    )


def pytest_generate_tests(metafunc):
    """Parametrize any test taking ``sample_name`` over all files, or a capability subset.

    A ``@pytest.mark.samples(*flags)`` marker narrows the set; an empty subset raises (a registry bug
    should fail loudly, not skip silently).
    """
    if "sample_name" not in metafunc.fixturenames:
        return
    names = SAMPLE_NAMES
    marker = metafunc.definition.get_closest_marker("samples")
    if marker is not None:
        names = files_with(*marker.args)
        if not names:
            raise ValueError(
                f"@pytest.mark.samples{marker.args!r} matched no sample files — check the SAMPLES registry"
            )
    metafunc.parametrize("sample_name", names, ids=lambda value: value[:-3])


@pytest.fixture
def sample():
    """Resolve a sample filename to its absolute path, skipping the test if the file is absent."""

    def _resolve(name):
        path = DATA_DIR / name
        if not path.exists():
            pytest.skip(f"sample file not present: {name}")
        return str(path)

    return _resolve


@pytest.fixture
def structural():
    """Return the structural-name parser (``parse_structural_name``)."""
    return parse_structural_name


@pytest.fixture
def caps(sample_name):
    """The capability-registry entry (flags dict) for the current parametrized sample file."""
    return SAMPLES[sample_name]


@pytest.fixture
def rank_histogram():
    """Return a helper computing ``{rank: count}`` from a ``NetCDFMetadata.variables`` mapping."""

    def _hist(variables):
        return dict(Counter(len(info.shape) for info in variables.values()))

    return _hist


@pytest.fixture
def tos(sample):
    """Open the TOS sample and yield its ``tos`` variable view; close the parent on teardown."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        yield nc.get_variable("tos")
    finally:
        nc.close()
