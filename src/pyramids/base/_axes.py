"""The names a horizontal coordinate axis goes by.

Readers meet the same handful of spellings for "this array is a coordinate,
not data" -- `x` / `lon` / `longitude` from CF, `rlon` from a rotated-pole
grid, `nav_lon` from NEMO, `easting` from a projected one. Every reader that
has to tell a coordinate from a data variable needs the same list, and each had
kept its own: the GeoTIFF path knew eleven x-spellings, the Zarr reader six.

That mattered. The Zarr reader picks its data array by elimination -- whatever
is not a known coordinate -- so a NEMO store, whose `nav_lon` / `nav_lat` are
full 2-D fields, offered them as candidates alongside the real variable and
returned a latitude field as the raster.

Kept under `base` so any reader can import it without reaching sideways into
another reader's module.
"""

from __future__ import annotations

# Spellings of the x / easting / longitude axis.
X_AXIS_NAMES = frozenset(
    {
        "x",
        "xc",
        "xdim",
        "x_dim",
        "lon",
        "longitude",
        "long",
        "rlon",
        "east",
        "easting",
        "nav_lon",
    }
)

# Spellings of the y / northing / latitude axis. Deliberately the mirror of
# `X_AXIS_NAMES`: a rule that requires a *pair* of axes to agree can only work
# if the two halves are maintained together.
Y_AXIS_NAMES = frozenset(
    {
        "y",
        "yc",
        "ydim",
        "y_dim",
        "lat",
        "latitude",
        "rlat",
        "north",
        "northing",
        "nav_lat",
    }
)

# Both halves. Derived rather than restated -- the third copy of this list used
# to be written out, and could disagree with the two it was meant to union.
AXIS_NAMES = X_AXIS_NAMES | Y_AXIS_NAMES

__all__ = ["AXIS_NAMES", "X_AXIS_NAMES", "Y_AXIS_NAMES"]
