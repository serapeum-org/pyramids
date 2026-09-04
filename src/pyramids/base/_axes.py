"""The names a horizontal coordinate axis goes by, in preference order.

Readers meet the same handful of spellings for "this array is a coordinate,
not data" -- `x` / `lon` / `longitude` from CF, `rlon` from a rotated-pole
grid, `nav_lon` from NEMO, `easting` from a projected one. Every reader that
has to tell a coordinate from a data variable needs the same list, and each had
kept its own: the GeoTIFF path knew eleven x-spellings, the Zarr reader six.

That mattered. The Zarr reader picks its data array by elimination -- whatever
is not a known coordinate -- so a NEMO store, whose `nav_lon` / `nav_lat` are
full 2-D fields, offered them as candidates alongside the real variable and
returned a latitude field as the raster.

**Order is part of the definition, not an accident.** A reader that has to pick
*one* axis from an array carrying several must prefer the grid's own dimension
coordinate over an auxiliary one. The CF pattern that forces this is a
projected grid with `x` / `y` in metres that also carries 1-D `lon` / `lat` in
degrees: alphabetical order picks `lon`, and the resulting geotransform
describes a degree grid while the CRS says UTM. So the sequences below run
dimension and projected names first and the geographic aliases last, and the
frozensets are derived from them rather than written out separately.

Kept under `base` so any reader can import it without reaching sideways into
another reader's module.
"""

from __future__ import annotations

# Spellings of the x / easting / longitude axis, most specific to the grid
# first. `x` and its variants name the grid's own dimension; `easting` / `east`
# are the projected spelling of the same thing; `rlon` is a rotated-pole grid's
# native axis. The geographic aliases trail because on a projected grid they
# are auxiliary coordinates describing the same cells in other units.
X_AXIS_NAMES_ORDERED = (
    "x",
    "xc",
    "xdim",
    "x_dim",
    "easting",
    "east",
    "rlon",
    "lon",
    "long",
    "longitude",
    "nav_lon",
)

# The mirror of `X_AXIS_NAMES_ORDERED`, in the same order: a rule that requires
# a *pair* of axes to agree can only work if the two halves are maintained
# together, and a reader that picks `x` must be able to pick the matching `y`.
Y_AXIS_NAMES_ORDERED = (
    "y",
    "yc",
    "ydim",
    "y_dim",
    "northing",
    "north",
    "rlat",
    "lat",
    "latitude",
    "nav_lat",
)

X_AXIS_NAMES = frozenset(X_AXIS_NAMES_ORDERED)
Y_AXIS_NAMES = frozenset(Y_AXIS_NAMES_ORDERED)

# Both halves. Derived rather than restated -- the third copy of this list used
# to be written out, and could disagree with the two it was meant to union.
AXIS_NAMES = X_AXIS_NAMES | Y_AXIS_NAMES

__all__ = [
    "AXIS_NAMES",
    "X_AXIS_NAMES",
    "X_AXIS_NAMES_ORDERED",
    "Y_AXIS_NAMES",
    "Y_AXIS_NAMES_ORDERED",
]
