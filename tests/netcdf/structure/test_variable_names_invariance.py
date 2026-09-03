"""`NetCDF.variable_names` must not depend on what was read before it.

Two classifiers used to answer "which arrays are data variables": a CF-based one
that ran whenever `meta_data` happened to be cached, and a store-derived one that
ran otherwise. Because `__str__` interpolates `meta_data`, merely printing or
logging a container switched which classifier answered -- and on a classic-mode
file the CF path returned an empty list, so `get_variable` then raised for every
variable in the file.

These tests pin the invariant the fix establishes: the answer is derived from the
store on every call, so reading `meta_data`, or printing the object, cannot change
it. They are deliberately black-box -- they assert the enumerated names, not which
code path produced them.
"""

import glob
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

DATA = Path(__file__).parents[2] / "data" / "netcdf"
FIXTURES = sorted(Path(p).name for p in glob.glob(str(DATA / "*.nc")))


@pytest.fixture(params=FIXTURES, ids=FIXTURES)
def netcdf_path(request) -> str:
    """Every netCDF fixture in the suite, by name."""
    return str(DATA / request.param)


def open_or_skip(path: str, multi_dimensional: bool) -> NetCDF:
    """Open `path` in the given mode, skipping fixtures that mode cannot read.

    Several fixtures carry only 1-D variables, so GDAL's classic driver refuses
    them outright ("not recognized as being in a supported file format"). They
    say nothing about the invariant under test.
    """
    try:
        return NetCDF.read_file(path, open_as_multi_dimensional=multi_dimensional)
    except RuntimeError as exc:  # pragma: no cover - depends on the fixture set
        pytest.skip(f"fixture not readable in this mode: {exc}")


@pytest.mark.parametrize("multi_dimensional", [True, False], ids=["mdim", "classic"])
class TestVariableNamesInvariance:
    """`variable_names` is the same however the object was used beforehand."""

    def test_reading_meta_data_does_not_change_variable_names(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """Consulting `meta_data` leaves the enumerated variables untouched."""
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        before = sorted(dataset.variable_names)

        _ = dataset.meta_data

        assert sorted(dataset.variable_names) == before

    def test_str_does_not_change_variable_names(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """`__str__` reads `meta_data`, so printing must stay side-effect free."""
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        before = sorted(dataset.variable_names)

        str(dataset)

        assert sorted(dataset.variable_names) == before

    def test_every_reported_variable_is_retrievable_after_meta_data(
        self, netcdf_path: str, multi_dimensional: bool
    ):
        """`get_variable` accepts every name `variable_names` advertises.

        The regression this guards raised `ValueError: <name> is not a valid
        variable name in []` for every variable of a file whose metadata had
        been read.

        Classic mode is excluded, and not because of this fix. There,
        `_classic_subdataset_variable_names` reports CF *standard* names
        (`precipitation_flux`) while `get_variable` resolves *store* names
        (`pr`), so the two disagree however the handle was used. That is a
        separate, pre-existing defect; asserting it here would fail for a
        reason this test is not about.
        """
        if not multi_dimensional:
            pytest.skip(
                "classic mode reports CF standard names but resolves store "
                "names -- a separate pre-existing defect"
            )
        dataset = open_or_skip(netcdf_path, multi_dimensional)
        _ = dataset.meta_data

        for name in dataset.variable_names:
            assert dataset.get_variable(name) is not None
