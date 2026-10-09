"""The cube as plain Python objects: `to_dict` out, `from_dict` back.

Every other export goes to a *file* or a *foreign object* — `to_file`, `to_zarr`, `to_kerchunk`,
`to_dataframe`, `to_xarray`, `to_stac_item`. This is the one that goes to nothing but `dict`,
`list`, `str` and numbers, which is what inspection, diffing two cubes, JSON transport and
asserting on structure in a test all want.

**The schema is a superset of xarray's, and that is the whole design decision.** xarray's
`to_dict` is keyed `dims` / `coords` / `data_vars` / `attrs` and has **no slot for a CRS or a
geotransform** — a `crs` attribute survives there only because it happens to ride in `attrs`. For
a GDAL-backed cube the georeferencing *is* the object, so matching xarray exactly would make the
round trip silently drop it. Instead those four keys keep their xarray meaning exactly, and
everything GDAL needs lives under one extra `pyramids` key. A reader that knows only xarray's
schema still finds what it expects; a round trip through here loses nothing.

What round-trips: dimensions and their order, every band dimension's coordinates, variable arrays
and dtypes, per-variable attributes, the CRS, the geotransform, the no-data sentinels and the
spatial dimension names. What does not: arbitrary *global* attributes. `from_array` takes globals
as a `CFAttributes`, which is a fixed set of CF fields rather than a free mapping, so `to_dict`
reports every global attribute it finds and `from_dict` restores the CF ones it recognises. That
is a stated limit, not a silent loss — the dict is complete, the constructor is the narrow part.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.base.georeference import GeoReference
from pyramids.netcdf.array_options import CFAttributes, ExtraDimensions

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import Container, NetCDF

SCHEMA_VERSION = 1
"""The version stamped into the `pyramids` block, so a future change is detectable."""

_REQUIRED = ("dims", "coords", "data_vars", "attrs", "pyramids")
"""The top-level keys `from_dict` refuses a payload without."""

_REQUIRED_GEO = ("epsg", "geotransform", "spatial_dims")
"""The keys inside the `pyramids` block that a cube cannot be rebuilt without."""


def _plain(value: Any) -> Any:
    """Convert NumPy containers and scalars to plain Python, recursively.

    NaN is left as a float NaN rather than mapped to `None`: it is what the array holds, and
    Python's `json` emits and reads it back by default. Mapping it would make the round trip lossy
    in exactly the cells a caller most needs to see.

    Args:
        value: Anything that might be a NumPy array, a NumPy scalar, or a container of them.

    Returns:
        Any: The same value built from `list`, `dict`, `str`, `bool`, `int`, `float` and `None`.
    """
    if isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, np.generic):
        result = value.item()
    elif isinstance(value, dict):
        result = {str(key): _plain(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_plain(item) for item in value]
    else:
        result = value
    return result


def _shaped_array(var: NetCDF, flat: Any) -> Any:
    """Restore a variable's band axes, which `read_array` returns flattened into one.

    A variable with band dimensions `(4, 3)` over a `5x6` grid reads back as `(12, 5, 6)`, since
    the store addresses bands by a single index. The declared sizes say how to split that again,
    and the dict records the array in its declared shape so `dims` and the data agree.

    Args:
        var: The variable, for its declared band sizes.
        flat: What `read_array` returned.

    Returns:
        Any: The array shaped `(*band_sizes, rows, cols)`.
    """
    array = np.asarray(flat)
    sizes = tuple(int(size) for size in var._band_dim_sizes)
    return array.reshape(*sizes, *array.shape[-2:]) if sizes else array


def to_dict(nc: NetCDF, *, data: bool = True) -> dict[str, Any]:
    """Export a cube's full structure as a nested dict.

    Args:
        nc: The container or variable to export.
        data: Whether to include the values. `False` emits structure only — every variable keeps
            its `dims`, `dtype`, `shape` and `attrs` but no `data` key — which is the useful mode
            for inspecting or diffing two cubes without reading them.

    Returns:
        dict[str, Any]: The payload, keyed `dims`, `coords`, `data_vars`, `attrs` in their xarray
        meanings, plus `pyramids` carrying the CRS, geotransform, spatial dimension names, no-data
        sentinels and schema version.

    Raises:
        ValueError: The cube has no data variables, or declares no CRS, so the payload could
            only rebuild into an unreferenced cube.

    Examples:
        - The four xarray keys, plus the one that carries the georeferencing:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> payload = cube.to_dict()
          >>> sorted(payload)
          ['attrs', 'coords', 'data_vars', 'dims', 'pyramids']
          >>> payload["dims"]
          {'time': 2, 'y': 1, 'x': 1}
          >>> payload["coords"]["time"]["data"]
          [0.0, 6.0]

          ```
        - The `pyramids` block is what xarray's schema has nowhere to put:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> block = cube.to_dict()["pyramids"]
          >>> block["epsg"], block["spatial_dims"]
          (4326, ['y', 'x'])
          >>> block["geotransform"]
          [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]

          ```
        - `data=False` keeps the shape and drops the values, for inspecting or diffing:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> entry = cube.to_dict(data=False)["data_vars"]["t"]
          >>> entry["shape"], entry["dtype"]
          ([2, 1, 1], 'float64')
          >>> "data" in entry
          False

          ```

    See Also:
        from_dict: Rebuilds a cube from what this returns.
        pyramids.netcdf.NetCDF.to_dataframe: The cube as a pandas frame instead, which is a
            flat table rather than the structure.
    """
    names = list(nc.variable_names)
    if not names:
        raise ValueError(
            "to_dict() needs at least one data variable, and this cube has none."
        )
    epsg = nc.epsg
    if epsg is None:
        raise ValueError(
            "to_dict() needs a CRS to export, and this cube declares none. Set one with "
            "to_crs() first — the payload carries it so from_dict() can georeference the "
            "result, and a dict without it would rebuild into an unreferenced cube."
        )
    group = nc._working_group()
    dims: dict[str, int] = {}
    coords: dict[str, Any] = {}
    data_vars: dict[str, Any] = {}
    no_data: dict[str, Any] = {}
    spatial: list[str] = []
    for name in names:
        var = cast("NetCDF", nc.get_variable(name))
        declared = list(nc._variable_dim_names(group, name))
        band = [str(entry) for entry in var._band_dim_names]
        spatial = declared[len(band) :]
        array = _shaped_array(var, var.read_array())
        for position, size in zip(declared, array.shape):
            dims[position] = int(size)
        for coord_name, values in var.coords.items():
            coords[str(coord_name)] = {
                "dims": [str(coord_name)],
                "data": _plain(np.asarray(values)),
                "attrs": {},
            }
        entry: dict[str, Any] = {
            "dims": declared,
            "attrs": _plain(dict(var.attrs)),
            "dtype": str(array.dtype),
            "shape": [int(size) for size in array.shape],
        }
        if data:
            entry["data"] = _plain(array)
        data_vars[name] = entry
        sentinels = var.no_data_value
        # Per-band on a variable, and uniform within one, so the first entry speaks for it.
        # Annotated as possibly scalar, which a cube built by some routes really is.
        first = sentinels[0] if isinstance(sentinels, (tuple, list)) else sentinels
        no_data[name] = _plain(first)
    return {
        "dims": dims,
        "coords": coords,
        "data_vars": data_vars,
        "attrs": _plain(dict(nc.attrs)),
        "pyramids": {
            "schema": SCHEMA_VERSION,
            "epsg": int(epsg),
            "geotransform": [float(value) for value in nc.geotransform],
            "spatial_dims": spatial,
            "no_data_value": no_data,
        },
    }


def _assert_payload(payload: Any) -> dict[str, Any]:
    """Refuse a payload that cannot describe a cube, naming the key at fault.

    `from_dict` takes untrusted input — a file, a request body, a hand-written literal — so every
    refusal names what is wrong rather than letting a malformed payload reach GDAL and fail there
    with a message about the store.

    Args:
        payload: What the caller passed.

    Returns:
        dict[str, Any]: The payload, once it is known to have the required shape.

    Raises:
        TypeError: `payload` is not a mapping.
        ValueError: A required top-level key, or a required key of the `pyramids` block, is
            missing, or there are no data variables.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"from_dict() needs a dict as to_dict() returns, but got "
            f"{type(payload).__name__}."
        )
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        raise ValueError(
            f"from_dict() needs the keys {list(_REQUIRED)}, and this payload is missing "
            f"{missing}. A dict from to_dict() always has them."
        )
    geo = payload["pyramids"]
    if not isinstance(geo, dict):
        raise TypeError(
            f"from_dict() needs the 'pyramids' key to hold a dict, but got "
            f"{type(geo).__name__}."
        )
    missing_geo = [key for key in _REQUIRED_GEO if key not in geo]
    if missing_geo:
        raise ValueError(
            f"from_dict() cannot georeference the result: the 'pyramids' block is missing "
            f"{missing_geo}. Without them the rebuilt cube would not be georeferenced, which "
            f"is why the schema carries them at all."
        )
    if not payload["data_vars"]:
        raise ValueError("from_dict() needs at least one entry in 'data_vars'.")
    return payload


def _variable_array(name: str, entry: Any, dims: dict[str, Any]) -> np.ndarray:
    """Build one variable's array, checking it against the dims it declares.

    Args:
        name: The variable's name, for the refusals.
        entry: Its entry in `data_vars`.
        dims: The payload's `dims`, for the expected sizes.

    Returns:
        np.ndarray: The array, in its declared shape.

    Raises:
        ValueError: The entry has no `data` (it came from `to_dict(data=False)`), declares a
            dimension the payload's `dims` does not list, or holds an array whose shape disagrees
            with those dimensions.
    """
    if not isinstance(entry, dict) or "data" not in entry:
        raise ValueError(
            f"from_dict() needs values for {name!r}, and this payload has none. A dict from "
            f"to_dict(data=False) carries structure only and cannot be rebuilt."
        )
    declared = [str(item) for item in entry.get("dims", [])]
    unknown = [item for item in declared if item not in dims]
    if unknown:
        raise ValueError(
            f"from_dict() cannot place {name!r}: it declares the dimensions {unknown}, which "
            f"the payload's 'dims' does not list."
        )
    array = np.asarray(entry["data"], dtype=entry.get("dtype") or None)
    expected = tuple(int(dims[item]) for item in declared)
    if array.shape != expected:
        raise ValueError(
            f"from_dict() found {name!r} shaped {array.shape}, but its dimensions {declared} "
            f"say it should be {expected}."
        )
    return array


def _cf_attributes(attrs: Any) -> CFAttributes:
    """Keep the global attributes `CFAttributes` has a field for, and drop the rest.

    The drop is the documented limit of the round trip: `from_array` takes globals as this fixed
    set of CF fields rather than a free mapping, so an arbitrary global attribute is reported by
    `to_dict` and cannot be restored here.

    Args:
        attrs: The payload's global `attrs`.

    Returns:
        CFAttributes: The recognised subset.
    """
    known = {field.name for field in fields(CFAttributes)}
    taken = {
        key: value
        for key, value in (attrs or {}).items()
        if key in known and value is not None
    }
    return CFAttributes(**taken)


def from_dict(payload: Any) -> Container:
    """Rebuild a cube from a nested dict as :func:`to_dict` produces.

    The acceptance criterion is the round trip, not the parse: `to_dict` then `from_dict` gives
    back an equivalent cube, georeferencing included.

    Args:
        payload: The dict to rebuild from.

    Returns:
        Container: The rebuilt cube.

    Raises:
        TypeError: `payload` is not a dict, or its `pyramids` block is not a dict.
        ValueError: A required key is missing; `data_vars` is empty; the geotransform is not six
            values; a variable has no values because the payload came from `to_dict(data=False)`;
            a variable declares an unknown dimension; or a variable's array shape disagrees with
            its dimensions.

    Examples:
        - The round trip, which is the contract: values and stamps come back:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> rebuilt = NetCDF.from_dict(cube.to_dict())
          >>> rebuilt.get_variable("t").read_array().ravel().tolist()
          [1.0, 2.0]
          >>> np.asarray(rebuilt.get_variable("t").coords["time"]).tolist()
          [0.0, 6.0]

          ```
        - The georeferencing survives, which is why the schema carries it at all:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> rebuilt = NetCDF.from_dict(cube.to_dict())
          >>> rebuilt.epsg
          4326
          >>> tuple(rebuilt.geotransform)
          (0.0, 1.0, 0, 1.0, 0, -1.0)

          ```
        - A payload carrying structure only cannot be rebuilt, and says so:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> cube = NetCDF.from_array(
          ...     np.array([1.0, 2.0]).reshape(2, 1, 1),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> NetCDF.from_dict(cube.to_dict(data=False))  # doctest: +ELLIPSIS
          Traceback (most recent call last):
              ...
          ValueError: from_dict() needs values for 't', ... structure only and cannot be rebuilt.

          ```

    See Also:
        to_dict: Produces the payload this consumes.
        pyramids.netcdf.NetCDF.from_array: The constructor this builds on, and the reason only
            the CF-recognised global attributes are restored.
    """
    # Local import breaks the netcdf.py <-> dict_io import cycle: netcdf.py's facades call these
    # functions, and the constructor here is netcdf.py's own.
    from pyramids.netcdf.netcdf import NetCDF

    payload = _assert_payload(payload)
    dims = payload["dims"]
    geo = payload["pyramids"]
    spatial = [str(item) for item in geo["spatial_dims"]]
    coords = payload["coords"]
    sentinels = geo.get("no_data_value") or {}
    transform = [float(value) for value in geo["geotransform"]]
    if len(transform) != 6:
        raise ValueError(
            f"from_dict() needs a 6-value geotransform, but this payload's has "
            f"{len(transform)}."
        )
    reference = GeoReference(
        geo=(
            transform[0],
            transform[1],
            transform[2],
            transform[3],
            transform[4],
            transform[5],
        ),
        epsg=geo["epsg"],
    )
    attributes = _cf_attributes(payload.get("attrs"))
    names_pair = (spatial[-2], spatial[-1]) if len(spatial) >= 2 else None
    built: Container | None = None
    for name, entry in payload["data_vars"].items():
        array = _variable_array(str(name), entry, dims)
        declared = [str(item) for item in entry["dims"]]
        band = [item for item in declared if item not in spatial]
        extra = (
            ExtraDimensions(
                dims=[(item, _plain(coords.get(item, {}).get("data"))) for item in band]
            )
            if band
            else None
        )
        one = NetCDF.from_array(
            array,
            geo_ref=reference,
            no_data_value=sentinels.get(name),
            variable_name=str(name),
            dims=extra,
            attrs=attributes,
            spatial_names=names_pair,
        )
        if built is None:
            built = one
        else:
            built.add_variable(one, str(name))
    return cast("Container", built)
