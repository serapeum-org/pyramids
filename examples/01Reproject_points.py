"""Reproject coordinates between EPSG codes.

Demonstrates :func:`pyramids.base.crs.reproject_coordinates`. Argument
and return order is ``(x, y)`` throughout.
"""

from pyramids.feature import FeatureCollection

# %% WGS84 → UTM 18N
x = [-180.0, -179.5]
y = [90.0, 90.0]

x_out, y_out = FeatureCollection.reproject_coordinates(
    x, y, from_crs=4326, to_crs=32618, precision=9
)

# %% Brazil
x_in = [4522693.11]
y_in = [7423522.55]

lon, lat = FeatureCollection.reproject_coordinates(
    x_in, y_in, from_crs=5641, to_crs=4326, precision=4
)

assert lat[0] == -22.6895 and lon[0] == -47.2903, "Error ReprojectPoints error 1"

# %% WGS84 → ETRS89 / UTM zone 32N + zE-N (EPSG:4647)
x = [-32.0, 71.0]
y = [32.0, 83.0]

x_out, y_out = FeatureCollection.reproject_coordinates(
    x, y, from_crs=4326, to_crs=4647, precision=4
)

assert (
    y_out[0] == 4390682.5383 and y_out[1] == 9629641.4604
), "Error ReprojectPoints error 2y"
assert (
    x_out[0] == 28494364.9445 and x_out[1] == 33190988.6123
), "Error ReprojectPoints error 2x"
