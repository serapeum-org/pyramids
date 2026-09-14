"""The fixture catalogue for the xarray parity suite.

The files below are chosen to cover the normalisations `_harness` applies rather than to be
representative stores: between them they run the y axis in both directions, pack and do not
pack, and carry two, one and no band dimensions. The last, the gapped store, is the only file
in the repo that combines packing with cells that really hold the fill value — the case where
taking the gaps after unpacking would silently compare fill against fill.

The catalogue is here rather than in the test module because it is shared knowledge: every
later task in `planning/xarray/missing-functionality-plan.md` writes its parity tests against
the same set, and the properties that decide which normalisations fire (`y_ascends`, `packed`)
are what those tests parametrize over. Keeping the table and the loader in one place is what
stops a second copy drifting from this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pyramids.netcdf.netcdf import NetCDF

DATA = Path(__file__).resolve().parents[2] / "data" / "netcdf"


@dataclass(frozen=True)
class ParityFixture:
    """One store in the parity catalogue, with the properties its tests parametrize over.

    Attributes:
        path: The file name under `tests/data/netcdf`.
        variable: The variable to compare.
        y_ascends: Whether the file stores its y axis south-to-north, so the harness flips it.
        packed: Whether the variable declares `scale_factor` / `add_offset`.
        covers: What this store is in the catalogue for.
    """

    path: str
    variable: str
    y_ascends: bool
    packed: bool
    covers: str

    @property
    def id(self) -> str:
        """A short pytest id: the convention prefix and the variable."""
        return f"{self.path.split('__')[0]}-{self.variable}"

    def open(self) -> NetCDF:
        """Open the store.

        Returns:
            NetCDF: The opened container.
        """
        return NetCDF.read_file(str(DATA / self.path))


PARITY_FIXTURES: tuple[ParityFixture, ...] = (
    ParityFixture(
        "cf__5v__1d4-4d1__y-asc.nc",
        "temperature",
        y_ascends=True,
        packed=False,
        covers=(
            "The flip, and nothing else. `temperature` is float64 over "
            "`(time, pressure_level, lat, lon)` = `(4, 3, 5, 6)` and declares neither packing "
            "nor a fill value; its latitudes run 40.0 up to 44.0 and its time axis decodes to "
            "`datetime64[ns]`."
        ),
    ),
    ParityFixture(
        "coards__5v__1d4-4d1__y-desc.nc",
        "rhum",
        y_ascends=False,
        packed=True,
        covers=(
            "Packing on a descending axis. `rhum` is int16 over `(time, level, lat, lon)` = "
            "`(12, 4, 37, 72)`, unpacking by `0.01 * stored + 302.66` into 0-100 percent. It "
            "declares 32766 as its fill value but no cell holds it, so this file tests the "
            "packing rather than the gap rule. Its `hours since 1-1-1` time axis is exported "
            "as stored offsets, with a `TimeDecodingWarning`."
        ),
    ),
    ParityFixture(
        "cf__5v__1d4-4d1__geog__y-desc.nc",
        "t",
        y_ascends=False,
        packed=False,
        covers=(
            "That the flip is read from the coordinate, not the geotransform. This container "
            "reports the placeholder `(0.0, 1.0, 0, 0.0, 0, -1.0)` while its latitudes run "
            "65.0 down to 63.25 at a 0.25 degree step. `t` is float64 over "
            "`(valid_time, pressure_level, latitude, longitude)` = `(4, 1, 8, 10)`, and its y "
            "dimension is spelled `latitude`."
        ),
    ),
    ParityFixture(
        "coards__4v__1d2-2d2__scaleoffset__y-asc.nc",
        "z",
        y_ascends=True,
        packed=True,
        covers=(
            "Ascending *and* packed, with no band dimensions at all. `z` is float32 on a 21x21 "
            "`(y, x)` grid with one band, so the band-axis rebuild has nothing to do. Its y is "
            "spelled plainly `y`, and its variables report a renamed `subset_y_20_-1_21` "
            "dimension — which is why `from_pyramids` takes the band names from the variable "
            "rather than from that list."
        ),
    ),
    ParityFixture(
        "cf__20v__1d3-3d17__y-desc.nc",
        "tcw",
        y_ascends=False,
        packed=True,
        covers=(
            "The gap rule, on cells that really are gaps. `tcw` is int16 over "
            "`(time, latitude, longitude)` = `(12, 73, 144)`, unpacking by "
            "`0.0013500981745480953 * stored + 44.3250482744756`, and exactly 63072 of its "
            "126144 cells hold the declared -32767. Unpacked, that fill reads back as 0.0864 "
            "— an ordinary physical value — which is why both sides take their gaps first."
        ),
    ),
)

#: The repo's only variable declaring a NaN fill value, so the only one that exercises the
#: `isnan` arm of the gap rule end to end.
NAN_SENTINEL = ParityFixture(
    "cf__5v__1d4-3d1__geog__y-desc.nc",
    "t2m",
    y_ascends=False,
    packed=False,
    covers="A declared NaN fill value, which needs `isnan` rather than `==` to find.",
)

#: Stores with no y dimension at all: a 1-D time series, and a curvilinear grid whose axes are
#: `eta_rho` / `xi_rho`. Both must leave the orientation rule with nothing to decide.
WITHOUT_Y = ("none__11v__1d11.nc", "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc")


def open_fixture(path: str) -> NetCDF:
    """Open a store from the parity data directory by file name.

    Args:
        path: The `.nc` file name under `tests/data/netcdf`.

    Returns:
        NetCDF: The opened container.
    """
    return NetCDF.read_file(str(DATA / path))


@pytest.fixture(params=PARITY_FIXTURES, ids=[case.id for case in PARITY_FIXTURES])
def parity_case(request: pytest.FixtureRequest) -> ParityFixture:
    """Each catalogue entry in turn, for a test that should hold across the whole set.

    Args:
        request: The pytest request carrying the parametrized entry.

    Returns:
        ParityFixture: The entry under test.
    """
    return request.param
