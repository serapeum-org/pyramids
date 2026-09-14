"""The fixture catalogue for the xarray parity suite.

The files below are chosen to cover the normalisations `_harness` applies rather than to be
representative stores: between them they run the y axis in both directions, pack and do not
pack, and carry two, one and no band dimensions. The last, the gapped store, is the only file
in the repo that combines packing with cells that really hold the fill value — the case where
taking the gaps after unpacking would silently compare fill against fill.

It is a module of its own rather than `conftest` content because it is ordinary shared data
that several modules import: every later task in `planning/xarray/missing-functionality-plan.md`
writes its parity tests against the same set, and the properties that decide which
normalisations fire (`y_ascends`, `packed`) are what those tests parametrize over. Keeping the
table and the loader in one place is what stops a second copy drifting from this one; keeping
it out of `conftest` means importing it does not depend on pytest's collection semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
            "A y axis spelled `latitude`, on a container whose geotransform is the "
            "placeholder `(0.0, 1.0, 0, 0.0, 0, -1.0)` while its latitudes run 65.0 down to "
            "63.25 at a 0.25 degree step — so the coordinate and the geotransform disagree "
            "about the *values*, though not about the direction. `t` is float64 over "
            "`(valid_time, pressure_level, latitude, longitude)` = `(4, 1, 8, 10)`."
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
        "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc",
        "pr",
        y_ascends=True,
        packed=False,
        covers=(
            "A band dimension of size 1, which `read_array` squeezes away entirely: `pr` reads "
            "back `(128, 256)` where `to_xarray` reports `('time', 'lat', 'lon')`. `isel` and "
            "a single-label `sel` produce exactly this shape, so T3 and T4 would otherwise "
            "have had nothing to compare. Its container also declares dimensions `pr` does not "
            "use, which is what caught the axis labels being taken from the container."
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

#: A variable declaring a NaN fill value, which exercises the `isnan` arm of the gap rule
#: end to end — `==` cannot find a NaN. It is not the only one in the repo, but it is the
#: one on a store the harness can otherwise handle.
NAN_SENTINEL = ParityFixture(
    "cf__5v__1d4-3d1__geog__y-desc.nc",
    "t2m",
    y_ascends=False,
    packed=False,
    covers="A declared NaN fill value, which needs `isnan` rather than `==` to find.",
)

#: Stores the harness refuses, with the phrase the refusal carries. Each is a case where the
#: orientation rule cannot be decided from the y coordinate, and answering anyway would hand
#: every downstream parity test a silently wrong reference: the ROMS and curvilinear stores
#: have a y axis with no coordinate variable (both in fact need the flip), and the GOES store's
#: `y` is the synthesised row index `0, 1, … 499`, which ascends but must not be flipped.
UNSUPPORTED = (
    ("cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc", "salt", "no coordinate variable"),
    ("none__4v__1d1-2d2-3d1__curv.nc", "Tair", "no coordinate variable"),
    ("cf__9v__1d7-2d2__geos__y-desc.nc", "CMI", "synthesised row index"),
)


def open_fixture(path: str) -> NetCDF:
    """Open a store from the parity data directory by file name.

    Args:
        path: The `.nc` file name under `tests/data/netcdf`.

    Returns:
        NetCDF: The opened container.
    """
    return NetCDF.read_file(str(DATA / path))
