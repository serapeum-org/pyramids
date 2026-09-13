"""xarray interoperability engine for :class:`pyramids.netcdf.NetCDF`.

Owns the bodies of ``NetCDF.to_xarray`` / ``NetCDF.from_xarray``,
extracted from the ``netcdf.py`` god-object (issue #615, STR-1). The
public ``NetCDF`` methods are thin façades that delegate here — the
conversion goes entirely through GDAL's Multidimensional API (the same
reader/writer the rest of pyramids' NetCDF code uses), so no xarray
NetCDF backend plugin is involved. Behaviour, signatures, and return
types are unchanged by the extraction.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import traceback
import warnings
import weakref
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cftime
import numpy as np
from osgeo import gdal, osr

from pyramids.base._errors import TimeDecodingWarning
from pyramids.base._utils import import_xarray, numpy_to_gdal_dtype
from pyramids.base.remote import is_remote
from pyramids.dataset.engines._base import _Engine
from pyramids.netcdf._lazy import build_lazy_array
from pyramids.netcdf._mdim import open_mdarray, strip_netcdf_subdataset_prefix
from pyramids.netcdf.cf import (
    build_coordinate_attrs,
    srs_from_wkt,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf.utils import (
    CF_EPOCH_CALENDAR,
    cf_epoch_units,
    is_cf_time_units,
    read_cf_attributes,
)

# The window `datetime64[ns]` can represent, in microseconds: 1677-09-21T00:12:43.145225
# to 2262-04-11T23:47:16.854775. Numpy wraps silently outside it. Derived from the int64
# tick count rather than written out, because a hand-picked bound that is even a day too
# narrow refuses representable instants while telling the user the type cannot hold them.
# The nanosecond ends are rounded inward to whole microseconds (ceil the floor, floor the
# ceiling) so a value inside these bounds always survives the cast up to nanoseconds.
_NS_TICKS = np.iinfo("int64")
_NS_MIN = np.datetime64((_NS_TICKS.min + 1 + 999) // 1000, "us")
_NS_MAX = np.datetime64(_NS_TICKS.max // 1000, "us")

# A coordinate as the writers take it: `(values, attrs)`, or `(values, attrs, encoding)`
# when the caller knows the CF units a decoded time axis has to go back out in.
CoordSpec = (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
)

# A variable as the writers take it: `(dims, values, attrs)`, plus the same optional
# encoding slot, which carries a decoded time array back out in its own CF units.
VarSpec = (
    tuple[tuple[str, ...], Any, dict[str, Any]]
    | tuple[tuple[str, ...], Any, dict[str, Any], dict[str, Any]]
)

_XARRAY_HINT = (
    "xarray is required for {func}(). Install with one of:\n"
    "  - PyPI:        pip install xarray\n"
    "  - conda-forge: conda install -c conda-forge xarray"
)

# Everything under this directory is pyramids' own code, so a warning raised
# from a frame inside it is never the frame the user wrote.
_PACKAGE_DIR = os.path.normcase(
    os.path.join(str(Path(__file__).resolve().parents[2]), "")
)

# netCDF reserves `/` as the group path separator and rejects it in a variable
# name; zarr reads it as a hierarchy separator too. The flat `xr.Dataset`
# namespace therefore cannot key a sub-group array by its store path, so the
# separator is substituted rather than the path dropped -- reducing
# `flight_a/CO` to `CO` would collide with every other group's `CO`, which is
# the very thing the qualification exists to tell apart.
_GROUP_PATH_SEPARATOR = "/"
_FLAT_NAME_SEPARATOR = "_"


def _caller_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel that lands on the caller outside pyramids.

    A counted ``stacklevel=`` pins the warning to a frame by arithmetic over
    the call chain that reaches it, so it silently starts pointing at the wrong
    line the moment a façade, an engine hop, or a helper is added or removed --
    and a warning that names a pyramids source line instead of the user's is
    both useless to read and unfilterable by ``module=``. This walks the chain
    instead: the first frame whose file is not under :data:`_PACKAGE_DIR` is
    the caller, whichever spelling they used (``NetCDF.to_xarray`` and
    ``NetCDF.interop.to_xarray`` sit at different depths).

    Returns:
        int: The ``stacklevel`` to pass to :func:`warnings.warn` from the frame
        that calls this. ``1`` when the immediate caller is already outside the
        package; the depth of the outermost pyramids frame when the whole stack
        is pyramids' own.
    """
    level = 1
    current = inspect.currentframe()
    frame = None if current is None else current.f_back
    while frame is not None and os.path.normcase(
        os.path.abspath(frame.f_code.co_filename)
    ).startswith(_PACKAGE_DIR):
        parent = frame.f_back
        if parent is None:
            break
        frame = parent
        level += 1
    return level


#: Stamped on an exported variable whose key is not the store's name for it, so
#: a caller holding the key can still reach `get_variable`. Only on the rewritten
#: ones: on a flat store every key already *is* the store name, and stamping the
#: whole export would put a pyramids-private attribute into files that have no
#: use for it.
_STORE_NAME_ATTR = "pyramids_store_name"


def _flat_export_name(variable_name: str, taken: set[str]) -> str:
    """Return the flat `xr.Dataset` key for a store variable.

    The group path is kept and only its separator substituted, so two groups'
    identically named leaves stay distinct (`flight_a/CO` and `flight_b/CO`
    become `flight_a_CO` and `flight_b_CO`). Substitution can still land on a
    name already in the dataset -- a root array literally called `flight_a_CO`,
    a dimension coordinate of that name, or another group path that flattens
    the same way -- and those are resolved by appending `_2`, `_3`, ...

    The substitution is therefore **not injective**: `_` is a legal character
    in a group name and in an array name, so group `a_b` + array `c` and group
    `a` + array `b_c` both flatten to `a_b_c`, and no rule reads the store name
    back out of the key. The suffix keeps the export legal but records nothing
    -- it says only that *some* name was taken first. Recovering the pre-image
    is the caller's job, and it does it by stamping the store name on the
    variable as its :data:`_STORE_NAME_ATTR` attribute; a caller holding the
    returned dataset reads it from there rather than trying to invert the key.

    Which of the two keeps the plain name is **not** decided by enumeration
    order. The caller reserves every unqualified store name before the loop
    starts, so a name the store holds literally always wins it and a flattened
    sub-group name is the one that moves. An earlier version of this docstring
    claimed the order was safe "because the enumeration lists a group's own
    arrays before it recurses" -- true of `_mdim_data_variable_names`, but the
    export iterates `_readable_variable_names()`, which emits the CF-classified
    data variables first and everything else after. A root array the
    classification does not call `data` -- an auxiliary coordinate, an
    ordinary CF construct -- therefore sorted *behind* a sub-group array that
    flattens onto its name, and the sub-group array took it: the export handed
    back one array's values under the other's name.

    Args:
        variable_name: The store's name for the variable, group-qualified with
            `/` when it lives in a sub-group.
        taken: Names already claimed in the dataset -- the data variables
            emitted so far plus the dimension coordinates.

    Returns:
        str: The key to use. It used to be returned with a flag saying whether a
        numeric suffix had been appended, which gated the caller's rename
        warning -- so the far commoner plain `/`-to-`_` rewrite went unannounced
        even though its key is just as unusable with `get_variable`. The caller
        compares the key with the store name instead, which catches both.

    Examples:
        - A sub-group array keeps its group as a prefix:
            ```python
            >>> from pyramids.netcdf.engines.interop import _flat_export_name
            >>> _flat_export_name("flight_a/CO", set())
            'flight_a_CO'

            ```
        - A collision with a name already claimed is suffixed, and the
          sub-group array is the one that moves:
            ```python
            >>> _flat_export_name("flight_a/CO", {"flight_a_CO"})
            'flight_a_CO_2'

            ```
        - The root array of that name keeps it, whichever order they are
          enumerated in, because its own reservation is lifted before it asks:
            ```python
            >>> _flat_export_name("flight_a_CO", {"flight_a_CO_2"})
            'flight_a_CO'

            ```
        - Two different group paths can flatten onto one name, and the key
          alone does not say which store name it came from -- only that this
          one asked second:
            ```python
            >>> _flat_export_name("a_b/c", set())
            'a_b_c'
            >>> _flat_export_name("a/b_c", {"a_b_c"})
            'a_b_c_2'

            ```
        - A root-level name is handed back untouched:
            ```python
            >>> _flat_export_name("temperature", set())
            'temperature'

            ```
    """
    flattened = variable_name.replace(_GROUP_PATH_SEPARATOR, _FLAT_NAME_SEPARATOR)
    candidate = flattened
    suffix = 1
    while candidate in taken:
        suffix += 1
        candidate = f"{flattened}_{suffix}"
    return candidate


if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


class Interop(_Engine["NetCDF"]):
    """xarray ↔ pyramids NetCDF conversion collaborator.

    Holds the body of :meth:`NetCDF.to_xarray`. The companion
    :meth:`NetCDF.from_xarray` is a classmethod (it builds a new
    container rather than operating on an existing instance), so its
    body lives in the module-level :func:`from_xarray` function rather
    than on this instance-bound engine.
    """

    def to_xarray(
        self, chunks: dict | str | int | None = None, *, decode_times: bool = True
    ) -> Any:
        """Convert this NetCDF container to an `xarray.Dataset`.

        Builds an `xarray.Dataset` that mirrors the variables,
        coordinates, dimensions, and global attributes of this pyramids
        NetCDF container.

        The entire conversion goes through GDAL's Multidimensional
        API — the same reader the rest of pyramids' NetCDF code uses.
        No xarray NetCDF engine plugin is involved — pyramids is the
        writer, so xarray does not need to pull a NetCDF backend.

        With the default `chunks=None` the returned `xr.Dataset` holds
        already-materialised numpy arrays. Pass `chunks` (a dict / int /
        `"auto"`) to build each data variable as a lazy dask-backed
        `DataArray` in the file's native axis order, so the dataset is
        assembled without loading every variable into RAM (ARC-48).
        Lazy reads need the optional `[lazy]` (dask) extra and a
        file-backed container; an in-memory container ignores `chunks`
        (its data is already resident).

        Requires the optional `xarray` package. Install with one of:

        - PyPI: ``pip install xarray``
        - conda-forge: ``conda install -c conda-forge xarray``

        A decoded time axis moves its CF `units` / `calendar` out of `attrs` and
        into the coordinate's `encoding`, where xarray itself keeps them: the
        values are no longer expressed in those units, so leaving them in `attrs`
        would describe the axis wrongly and invite a second decode on write.
        :func:`from_xarray` reads them back from `encoding`, so a container read
        here and written back keeps the origin and calendar it arrived with.

        Args:
            chunks: Chunk spec forwarded to the lazy reader per data
                variable. `None` (default) reads eagerly.
            decode_times: When `True` (the default) a dimension whose CF `units`
                name a time origin is exported as datetimes, so xarray's own
                `resample` / `.dt` / `groupby("time.month")` work on the result.
                `False` exports the stored offsets unchanged, `units` and all.

        Returns:
            xarray.Dataset: An xarray Dataset with the same
            variables, coordinates, and global attributes.

        Raises:
            pyramids.base._errors.OptionalPackageDoesNotExist:
                If `xarray` is not installed.
            ImportError: If `chunks` is given but the `[lazy]` (dask)
                extra is not installed.
            ValueError: If the underlying GDAL handle is not a
                multidimensional container (open the file with
                `open_as_multi_dimensional=True`).

        Warns:
            TimeDecodingWarning: A dimension declares CF time `units` that could
                not be decoded — a non-standard calendar, an instant outside
                `datetime64[ns]`, or a conversion failure. The axis is exported as
                its stored offsets and the export continues; see
                :func:`_decode_time_coordinate` for the three reasons.

        Examples:
            - Export a CF container and use xarray's own time machinery on the
              result:

                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                >>> xds = nc.to_xarray()
                >>> str(xds.coords["time"].dtype)
                'datetime64[ns]'
                >>> [int(hour) for hour in xds.coords["time"].dt.hour.values]
                [0, 6, 12, 18]
                >>> xds["temperature"].resample(time="1D").mean().shape
                (1, 3, 5, 6)

                ```
            - The decoded axis keeps its CF units in `encoding`, not in `attrs`,
              so the write-back path can re-use the file's own origin:

                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                >>> time = nc.to_xarray().coords["time"]
                >>> time.attrs
                {}
                >>> time.encoding["units"]
                'hours since 2024-01-01'

                ```
            - Ask for the stored offsets instead, and the axis comes back numeric
              with its `units` still on it:

                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                >>> raw = nc.to_xarray(decode_times=False)
                >>> [float(offset) for offset in raw.coords["time"].values]
                [0.0, 6.0, 12.0, 18.0]
                >>> raw.coords["time"].attrs["units"]
                'hours since 2024-01-01'

                ```
            - Build the cube lazily instead, so each data variable arrives as a
              dask-backed array that is only read when it is computed:

                ```python
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file("tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc")
                >>> lazy = nc.to_xarray(chunks="auto")
                >>> lazy["temperature"].chunks
                ((4,), (3,), (5,), (6,))
                >>> round(float(lazy["temperature"].mean().compute()), 1)
                1622.5

                ```
            - An axis that declares CF time units but cannot be decoded warns and
              degrades to those offsets rather than failing the export:

                ```python
                >>> import warnings
                >>> from pyramids.netcdf import NetCDF
                >>> nc = NetCDF.read_file("tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc")
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     legacy = nc.to_xarray()
                >>> str(caught[0].message).split(" but ")[0]
                "time coordinate 'time' declares 'hours since 1-1-1 00:00:0.0'"
                >>> str(legacy.coords["time"].dtype)
                'float64'
                >>> legacy.coords["time"].attrs["units"]
                'hours since 1-1-1 00:00:0.0'

                ```

        See Also:
            from_xarray: The inverse, which re-encodes a decoded axis into the
                `units` carried in its `encoding`.
            pyramids.errors.TimeDecodingWarning: The category emitted when an
                axis is left undecoded, and why.
        """
        ds = self._ds
        xr = import_xarray(_XARRAY_HINT.format(func="to_xarray"))

        rg = ds._working_group()
        if rg is None:
            raise ValueError(
                "to_xarray requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        coords, bounds_encodings = _coords_from_dimensions(
            rg, ds, decode_times=decode_times
        )
        data_vars, export_names = _data_vars_from_arrays(rg, ds, chunks, set(coords))
        exported = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=ds.global_attributes,
        )
        return _promote_cf_non_data_arrays(
            exported, ds, export_names, bounds_encodings=bounds_encodings
        )


# The CF roles xarray represents as coordinates rather than data variables.
# `ancillary` is deliberately absent: CF ancillary variables are quality flags
# and uncertainty estimates, which xarray and cf-xarray both carry as ordinary
# data variables, so promoting them would be the opposite correction.
_COORDINATE_CF_ROLES = frozenset(
    {
        "bounds",
        "coordinate",
        "auxiliary_coordinate",
        "cell_measure",
        "grid_mapping",
    }
)


def _promote_cf_non_data_arrays(
    exported: Any,
    ds: NetCDF,
    export_names: dict[str, str],
    *,
    bounds_encodings: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Move the exported CF non-data arrays from ``data_vars`` into ``coords``.

    The export loop reads every *readable* variable, which is what stops an
    aux array from vanishing -- but it hands `lat_bnds` and the 2-D
    `lat_rho` / `lon_rho` to `xr.Dataset` as data variables. In xarray's
    convention `lat_bnds` belongs under `lat`'s `bounds` attribute and the
    `*_rho` pair are auxiliary coordinates; exported as data variables, a
    `to_netcdf()` round-trip writes them back as ordinary variables and the CF
    relationship is gone.

    Which roles move is :data:`_COORDINATE_CF_ROLES`, whose own comment records
    why `ancillary` is not one of them.

    The CF classification is keyed by *store* names and the dataset by *export*
    names, and :func:`_flat_export_name` makes those differ for a grouped store,
    so the roles are translated through `export_names` before they are looked
    up. The translation is the identity for every store the suite holds -- a
    sub-group array that classifies as a coordinate is filtered out of the
    enumeration, so it is never a data variable to begin with -- but the two
    namespaces are joined here and nowhere else, and leaving that join implicit
    is how it would quietly stop matching the next time the enumeration widens.

    Args:
        exported: The `xr.Dataset` just built from the container.
        ds: The container it was built from, for its CF classification.
        export_names: Store name to the key it was exported under, from
            :func:`_data_vars_from_arrays`. A name absent from it is its own
            key.
        bounds_encodings: Bounds-variable **store** name to the CF `units` / `calendar`
            of the decoded coordinate that names it, from
            :func:`_coords_from_dimensions`. Translated to export names here, as the
            roles are, and each such array is decoded with them once promoted.

    Returns:
        xr.Dataset: The same dataset with those names promoted. Unchanged when
            the store carries no CF classification, so a non-CF file is not
            second-guessed.

    Notes:
        `exported` and the return are typed `Any` rather than `xr.Dataset`
        because xarray is an optional extra -- naming the class would mean
        importing it at module level.

        There is no doctest here: reaching this needs a container with a real
        CF classification, which means a file on disk. The behaviour is covered
        by `tests/netcdf/test_round2_followups.py` and
        `tests/netcdf/test_xarray_round_trip.py` against the suite's fixtures.

    See Also:
        _data_vars_from_arrays: Builds the mapping this reclassifies.
        _coords_from_dimensions: The dimension coordinates it joins them to.
    """
    cf = ds.meta_data.cf
    if cf is None:
        result = exported
    else:
        candidates = (
            export_names.get(name, name)
            for name, role in cf.classifications.items()
            if role in _COORDINATE_CF_ROLES
        )
        promote = [key for key in candidates if key in exported.data_vars]
        result = exported.set_coords(promote) if promote else exported
        # A coordinate's CF `bounds` attribute names a *store* array, and the keys here
        # are export names -- the same two namespaces the roles above are translated
        # between, and for the same reason. The two are identical for every store in
        # the suite, which is exactly why an implicit join would go unnoticed.
        by_export = {
            export_names.get(name, name): encoding
            for name, encoding in (bounds_encodings or {}).items()
        }
        for key in promote:
            result = _decode_bounds_coordinate(result, key, by_export.get(key))
    return result


def _decode_bounds_coordinate(
    exported: Any, key: str, encoding: dict[str, Any] | None
) -> Any:
    """Decode a promoted CF bounds array with its parent coordinate's time units.

    CF says a bounds variable carries no `units` of its own and inherits the
    coordinate that names it, so the decoder cannot recognise one by looking at its
    attributes -- and a `time` exported as `datetime64[ns]` beside a numeric
    `time_bnds` is an internally inconsistent CF object: nothing downstream can
    relate the two. xarray decodes bounds the same way, by lending them the parent's
    attributes first.

    Args:
        exported: The `xr.Dataset` whose coordinate is to be replaced.
        key: The promoted coordinate's name.
        encoding: The parent coordinate's CF `units` / `calendar`, or `None` when this
            array is not the bounds of a decoded time axis.

    Returns:
        The dataset with that coordinate decoded, or unchanged when it is not a
        decodable time bounds array.
    """
    result = exported
    if encoding:
        variable = exported[key]
        decoded = _decode_time_coordinate(
            np.asarray(variable.values), dict(encoding), key
        )
        if decoded is not None:
            result = exported.assign_coords(
                {key: (variable.dims, decoded, dict(variable.attrs), dict(encoding))}
            )
    return result


def _warn_not_decoded(name: str, units: Any, reason: str) -> None:
    """Report that a CF time axis was exported as its stored offsets.

    Args:
        name: The dimension's name, or `""` when the caller did not supply one.
        units: The CF `units` string that made the axis a decoding candidate.
        reason: Why it was not decoded, phrased to follow "because".
    """
    labelled = f"{name!r} " if name else ""
    warnings.warn(
        f"time coordinate {labelled}declares {units!r} but was exported as stored "
        f"offsets because {reason}. xarray's resample/.dt/groupby will not work on it; "
        "pass decode_times=False to ask for the offsets deliberately.",
        TimeDecodingWarning,
        # Walked, not counted: the coordinate path and the bounds path reach here
        # through different depths, and either would drift the moment a hop is added.
        stacklevel=_caller_stacklevel(),
    )


def _decode_time_coordinate(values: Any, attrs: dict, name: str = "") -> Any | None:
    """Decode a CF time axis' stored offsets into datetimes, or ``None`` if it is not one.

    A CF time coordinate is stored as numbers against an origin (``hours since
    2024-01-01``). Handing those raw numbers to xarray leaves it with a plain numeric
    index, which its own time machinery cannot use: ``resample``, ``.dt`` and
    ``groupby("time.month")`` all fail on the result. Decoding here is what makes the
    exported object a cube xarray can actually work on (#1137).

    Only a **standard** calendar whose instants fit ``datetime64[ns]`` is decoded. A non-standard one
    (``360_day``, ``noleap``) would decode to ``cftime`` objects — an object-dtype array
    GDAL cannot write back, so exporting it would fix the xarray side at the cost of the
    round trip. Those axes keep their stored offsets; re-encoding them on write is the
    work that would lift that restriction.

    An axis that declares CF time ``units`` and is then declined warns with
    :class:`~pyramids.errors.TimeDecodingWarning` naming the dimension and the reason.
    The export still degrades rather than aborting — a bad coordinate is not the
    export's to discover — but silently not decoding left the caller meeting only the
    consequence, an xarray error about a non-datetime index. An axis whose ``units`` are
    not time units at all is not a candidate and says nothing.

    Args:
        values: The dimension's stored coordinate values.
        attrs: That dimension's CF attributes, read from the indexing variable.
        name: The dimension's name, used only to name it in the warning.

    Returns:
        The decoded array, or ``None`` when the axis declares no CF time ``units``, the
        values cannot be decoded, or the decoded instants fall outside the
        ``datetime64[ns]`` range — in which case the caller keeps the raw numbers.

    Warns:
        TimeDecodingWarning: The axis declares CF time ``units`` but was left as stored
            offsets — a non-standard calendar, an instant outside ``datetime64[ns]``, or
            a conversion failure.
    """
    units = attrs.get("units")
    if not is_cf_time_units(units):
        return None
    calendar = attrs.get("calendar") or "standard"
    decoded: Any = None
    try:
        converted = cftime.num2date(
            np.asarray(values), units, calendar, only_use_cftime_datetimes=False
        )
    except (ValueError, TypeError, OverflowError) as error:
        # A malformed origin, an out-of-range offset or a fill value in the axis. The
        # export must not fail because a coordinate could not be decoded, so fall back
        # to the raw offsets -- the same degrade-rather-than-abort contract `sel` uses.
        # Narrow on purpose: anything else raised in there is a defect of ours, and
        # swallowing it would turn it into a quietly numeric axis with no trace of why.
        _warn_not_decoded(name, units, f"{type(error).__name__}: {error}")
    else:
        array = np.asarray(converted)
        if not (array.size and isinstance(array.flat[0], datetime)):
            # A non-standard calendar (`360_day`, `noleap`) decodes to `cftime` objects,
            # an object-dtype array GDAL has no band type for: exporting it would give
            # xarray a usable index but break the write-back round trip, which is a
            # worse trade than leaving the offsets alone. Those axes keep their numbers.
            produced = type(array.flat[0]).__name__ if array.size else "no"
            _warn_not_decoded(
                name,
                units,
                f"it decodes to {produced} objects, which GDAL has no band type for "
                f"(the {calendar!r} calendar, or an origin before the 1582 reform)",
            )
        else:
            # `datetime64[ns]` spans 1677-09-21 to 2262-04-11 and numpy *wraps* an
            # instant outside it rather than raising, so an unchecked cast turns `hours
            # since 1600-01-01` into dates in 2184 with no error anywhere. Decode at
            # microsecond resolution first, which reaches well beyond any CF axis, and
            # hand the axis back undecoded when it will not fit. Paleo reconstructions
            # and post-2262 climate projections are the real cases.
            micro = array.astype("datetime64[us]")
            if micro.min() < _NS_MIN or micro.max() > _NS_MAX:
                _warn_not_decoded(
                    name,
                    units,
                    f"{micro.min()} to {micro.max()} falls outside datetime64[ns]",
                )
            else:
                decoded = micro.astype("datetime64[ns]")
    return decoded


def _coords_from_dimensions(
    rg: Any, ds: NetCDF, *, decode_times: bool = True
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the ``xr.Dataset`` ``coords`` mapping from the root group's dimensions.

    Each dimension with an indexing variable becomes a 1-D coordinate; bare
    dimensions (no indexing variable) are skipped. A CF time axis is decoded to
    datetimes unless ``decode_times`` is ``False`` — see
    :func:`_decode_time_coordinate`.

    Args:
        rg: The container's root group.
        ds: The container being exported, for its array reader.
        decode_times: Whether to decode a CF time axis to datetimes.

    Returns:
        The coordinate mapping, and the CF ``units`` / ``calendar`` of each decoded
        axis keyed by the ``bounds`` variable it names — which
        :func:`_decode_bounds_coordinate` needs, because a bounds array declares no
        units of its own.
    """
    coords: dict[str, Any] = {}
    bounds_encodings: dict[str, dict[str, Any]] = {}
    for d in rg.GetDimensions() or []:
        iv = d.GetIndexingVariable()
        if iv is None:
            continue
        dim_name = d.GetName()
        coord_attrs = read_cf_attributes(iv)
        raw = ds._md_array_to_numpy(iv)
        encoding: dict[str, Any] = {}
        decoded = (
            _decode_time_coordinate(raw, coord_attrs, dim_name)
            if decode_times
            else None
        )
        if decoded is not None:
            raw = decoded
            # The values are no longer expressed in those units, so carrying them in
            # `attrs` would describe the coordinate wrongly and invite a double decode
            # on write. They move to `encoding` instead -- where xarray itself puts
            # them, and where `from_xarray` looks to re-encode the axis in the units
            # it arrived in rather than inventing an epoch of its own.
            encoding = {
                k: coord_attrs[k] for k in ("units", "calendar") if k in coord_attrs
            }
            bounds = coord_attrs.get("bounds")
            if bounds:
                bounds_encodings[str(bounds)] = encoding
            coord_attrs = {
                k: v for k, v in coord_attrs.items() if k not in ("units", "calendar")
            }
        coords[dim_name] = ([dim_name], raw, coord_attrs, encoding)
    return coords, bounds_encodings


def _data_vars_from_arrays(
    rg: Any, ds: NetCDF, chunks: Any = None, reserved: set[str] | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the ``xr.Dataset`` ``data_vars`` mapping from the container's variables.

    With ``chunks=None`` each variable is read eagerly; otherwise it is read lazily (dask) in the
    file's native axis order via :func:`_lazy_var_data` (ARC-48).

    An `xr.Dataset` is one flat namespace with one size per dimension name, and
    a grouped netCDF store is neither: two sub-groups may each declare their own
    `air_press` at different lengths. Such a variable cannot be represented
    alongside the one already taken, so it is skipped with a warning naming it
    -- the alternative is `xarray` raising and the whole export failing.

    The same flatness decides the *keys*: a sub-group variable is keyed by
    :func:`_flat_export_name`, not by its store path, because netCDF rejects the
    `/` a store path carries. Exporting the raw path produced a Dataset that
    `NetCDF.from_xarray` could not write back (`NetCDF: Name contains illegal
    characters`) and that `to_netcdf` either rejected or wrote as a group,
    depending on the engine (round-4 N5). The store name is still what
    `get_variable` takes, so the skip warning below keeps naming that.

    Which leaves the user holding keys their own container refuses: on the
    suite's grouped fixture eight of the nine exported data variables are keys
    `get_variable` rejects. Every rewritten variable therefore both appears in
    the warning below -- it used to fire only for the suffixed minority, so the
    plain `/`-to-`_` rewrite, which is the case that actually happens, was
    silent -- and carries the store's name for it as its `pyramids_store_name`
    attribute. The attribute is the half that matters: a warning is text a
    caller cannot look a name up in, and the flattening is not injective, so
    the key cannot be inverted without it.

    Args:
        rg: The container's working (root) group.
        ds: The container being exported.
        chunks: Chunk spec per data variable; `None` reads eagerly.
        reserved: Names already claimed in the dataset under construction --
            the dimension coordinates, which share one namespace with the data
            variables and so must not be collided with.

    Returns:
        tuple[dict[str, Any], dict[str, str]]: The `data_vars` mapping keyed by
        export name, and the store name to export name map for the variables it
        emitted -- which is what lets :func:`_promote_cf_non_data_arrays` look
        the CF classification's store names up in the exported namespace.
    """
    data_vars: dict[str, Any] = {}
    export_names: dict[str, str] = {}
    dim_sizes: dict[str, int] = {}
    conflicted: list[str] = []
    renamed: list[str] = []
    taken: set[str] = set(reserved or ())
    readable = list(ds._readable_variable_names())
    # Reserve every unqualified store name before assigning any, so which of a
    # root array and a sub-group array keeps the plain name does not depend on
    # the order they are enumerated in. Without this the loser is whichever
    # comes second, and `_readable_variable_names()` puts the CF-classified
    # data variables first -- so a root auxiliary coordinate lost its own name
    # to a sub-group array, and the export returned that array's values under
    # it. Each root name's own reservation is lifted when its turn comes.
    taken |= {name for name in readable if _GROUP_PATH_SEPARATOR not in name}
    # Readable names, not the data-variable enumeration: an export that
    # dropped the store's aux arrays (`expver`, bounds) would not round-trip.
    for var_name in readable:
        # Through the helper: a name from the variable enumeration may be
        # group-qualified, which `OpenMDArray` alone does not resolve.
        md_arr = open_mdarray(rg, var_name)
        if md_arr is None:
            continue
        dims = md_arr.GetDimensions() or []
        arr_dim_names = [ad.GetName() for ad in dims]
        sizes = {ad.GetName(): ad.GetSize() for ad in dims}
        if any(dim_sizes.get(n, size) != size for n, size in sizes.items()):
            conflicted.append(var_name)
            continue
        if chunks is None:
            arr_data = ds._md_array_to_numpy(md_arr)
        else:
            # The store name, not the export name: the lazy reader reopens the
            # file and resolves the path against it.
            arr_data = _lazy_var_data(ds, var_name, chunks, md_arr)
        var_attrs = read_cf_attributes(md_arr)
        # A root array is claiming the name it already holds, so its own
        # reservation must not read as a collision with itself.
        taken.discard(var_name)
        export_name = _flat_export_name(var_name, taken)
        if export_name != var_name:
            renamed.append(f"{var_name} -> {export_name}")
            # Every rewritten key, not just the suffixed ones: `get_variable`
            # refuses the flattened form as flatly as the suffixed form, and the
            # flattening is not invertible from the key alone (group `a_b` +
            # array `c` and group `a` + array `b_c` both give `a_b_c`), so the
            # pre-image has to travel with the variable. `export_names` is
            # call-local and the warning is text, so neither is something a
            # caller holding the returned Dataset can look a name up in.
            var_attrs[_STORE_NAME_ATTR] = var_name
        taken.add(export_name)
        export_names[var_name] = export_name
        data_vars[export_name] = (arr_dim_names, arr_data, var_attrs)
        dim_sizes.update(sizes)
    if renamed:
        warnings.warn(
            f"to_xarray() renamed {len(renamed)} variable(s): {renamed}. An xarray "
            "Dataset is one flat namespace and netCDF forbids '/' in a name, so a "
            "sub-group array's path is flattened with '_', and where that lands on a "
            "name the store already holds a numeric suffix breaks the tie. The name "
            "before the arrow is the store's, and the only one get_variable() takes; "
            f"each renamed variable also carries it as its {_STORE_NAME_ATTR!r} "
            "attribute, because the key alone cannot be turned back into it.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    if conflicted:
        warnings.warn(
            f"to_xarray() skipped {len(conflicted)} variable(s) whose dimensions "
            f"clash with one already exported: {conflicted}. An xarray Dataset has "
            "one size per dimension name, which a grouped store need not honour; "
            "read those variables individually with get_variable() instead.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    return data_vars, export_names


def _reopenable_path(ds: NetCDF) -> str | None:
    """Return the bare, reopenable file path of a file-backed container, else ``None``.

    Strips any ``NETCDF:"…":var`` subdataset prefix and confirms the path exists on disk or is a
    remote (`/vsi*` / cloud) URL; an in-memory container has no such path.
    """
    path = strip_netcdf_subdataset_prefix(getattr(ds, "_file_name", "") or "")
    if path and (os.path.isfile(path) or is_remote(path)):
        return path
    return None


def _lazy_var_data(ds: NetCDF, var_name: str, chunks: Any, md_arr: Any) -> Any:
    """Return a dask-backed raw read of ``var_name`` in the file's native axis order (ARC-48).

    Deferred counterpart of :meth:`NetCDF._md_array_to_numpy`, so ``to_xarray(chunks=...)`` assembles
    a lazy dataset without materialising every variable. A file-backed container reads through
    :func:`build_lazy_array` with ``orient=False`` (raw, matching the raw coordinate arrays); an
    in-memory container has no file to reopen and its data is already resident, so the eager array is
    returned unchanged (``chunks`` has no benefit there).

    A variable whose dtype a chunked read cannot represent (e.g. a string MDArray such as a CF
    ``expver`` label) falls back to the eager read, matching the default ``to_xarray`` path -- one
    non-chunkable variable must not fail the whole lazy conversion.

    The container's captured cloud config (``_gdal_env``) is carried into the read so each chunk
    task re-opens a signed remote store with the same credentials, matching ``_read_array_lazy`` (#839).
    """
    path = _reopenable_path(ds)
    if path is None:
        return ds._md_array_to_numpy(md_arr)
    try:
        return build_lazy_array(
            path, var_name, chunks, orient=False, gdal_env=ds._gdal_env or None
        )
    except ValueError:
        return ds._md_array_to_numpy(md_arr)


def from_xarray(
    cls: type[NetCDF],
    dataset: Any,
    path: str | Path | None = None,
) -> NetCDF:
    """Create a pyramids NetCDF from an `xarray.Dataset`.

    Extracts dimensions, coordinates, data variables, and
    attributes from the `xarray.Dataset` and writes them to a
    NetCDF file through pyramids' own GDAL Multidimensional
    writer. No xarray NetCDF engine plugin is involved — pyramids
    is the writer, so xarray does not need to pull a NetCDF backend.

    Usage::

        ds = xr.open_dataset("input.nc")
        #... xarray processing...
        nc = NetCDF.from_xarray(ds)
        var = nc.get_variable("temperature")
        cropped = var.crop(mask)

    A CF-decoded time axis — `datetime64`, which GDAL has no band type for — is
    re-encoded on the way in. The CF `units` and `calendar` in the array's xarray
    `encoding` are used when they are there, so a container read with
    :meth:`Interop.to_xarray` and written back keeps the origin and calendar it
    arrived with; only an axis that carries no such encoding falls back to the
    writer's own `seconds since 1970-01-01` epoch. The same applies to a data
    variable and to an auxiliary array such as a CF bounds variable.

    Requires the optional `xarray` package. Install with one of:

    - PyPI: ``pip install xarray``
    - conda-forge: ``conda install -c conda-forge xarray``

    Args:
        cls: The concrete ``NetCDF`` subclass (``Container``) used to
            read the written file back in. Threaded through by the
            ``NetCDF.from_xarray`` classmethod façade.
        dataset: An `xarray.Dataset` instance.
        path: File path where the NetCDF will be written. If
            `None`, a temp `.nc` is created and cleaned up
            when the returned object is garbage-collected.

    Returns:
        NetCDF: A pyramids NetCDF container backed by the data
        from the xarray Dataset.

    Raises:
        pyramids.base._errors.OptionalPackageDoesNotExist:
            If `xarray` is not installed.
        TypeError: If *dataset* is not an `xarray.Dataset`.

    See Also:
        Interop.to_xarray: The inverse, which puts the CF `units` this reads into
            the exported coordinate's `encoding`.
    """
    xr = import_xarray(_XARRAY_HINT.format(func="from_xarray"))

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Expected xarray.Dataset, got {type(dataset).__name__}")

    cleanup_temp = False
    if path is not None:
        path = str(path)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        path = tmp.name
        tmp.close()
        cleanup_temp = True

    mem_src = _build_multidim_from_xarray(dataset)
    _create_copy_to_netcdf(mem_src, path)
    mem_src = None

    result = cls.read_file(path, read_only=True)
    if cleanup_temp:
        result._interop_temp_path = path
        weakref.finalize(result, os.unlink, path)
    return result


def _warn_not_encoded(name: str, units: Any, reason: str) -> None:
    """Report that a time array was written against the epoch, not its declared units.

    The write is the side that changes a file, so a silent fallback here is worse than
    the decode-side one it mirrors: the caller asked for one epoch and another was put
    on disk, with the instants intact but the axis rebased.

    Args:
        name: The array's name, or `""` when the caller did not supply one.
        units: The CF `units` the array's encoding declared.
        reason: Why they could not be used, phrased to follow "because".
    """
    labelled = f"{name!r} " if name else ""
    warnings.warn(
        f"time array {labelled}declares {units!r} in its encoding but was written as "
        f"{cf_epoch_units('seconds')!r} because {reason}. The instants are unchanged; "
        "the stored offsets and units are not.",
        TimeDecodingWarning,
        # Walked, not counted, for the reason `_warn_not_decoded` gives: a coordinate,
        # a data variable and a streamed write all reach here at different depths.
        stacklevel=_caller_stacklevel(),
    )


def _encode_in_declared_units(
    values: np.ndarray, units: str, calendar: str, name: str = ""
) -> np.ndarray | None:
    """Encode datetime64 values back into the CF ``units`` they were decoded from.

    Args:
        values: A `datetime64` array.
        units: The CF time units to encode into, e.g. `"hours since 2024-01-01"`.
        calendar: The CF calendar those units are counted in.
        name: The array's name, used only to name it in the warning.

    Returns:
        The float64 offsets in `units` (`NaN` where the input was `NaT`), or `None`
        when the array carries no valid instant to anchor on or `cftime` refuses the
        units — in which case the caller falls back to the epoch encoding.

    Warns:
        TimeDecodingWarning: The declared units could not be used, so the caller will
            write the array against the 1970 epoch instead.
    """
    as_us = values.astype("datetime64[us]")
    flat = as_us.ravel()
    missing = np.isnat(flat)
    result: np.ndarray | None = None
    if missing.all():
        _warn_not_encoded(
            name,
            units,
            "every instant is NaT, leaving nothing to anchor the offsets on",
        )
    else:
        instants = flat.astype(object)
        # `date2num` has no notion of a missing instant, so the `NaT` slots are handed a
        # real one to encode and overwritten with `NaN` afterwards.
        instants = np.where(missing, instants[~missing][0], instants)
        try:
            numbers = np.asarray(
                cftime.date2num(instants.tolist(), units, calendar), dtype="float64"
            )
        except (ValueError, TypeError, OverflowError) as error:
            # An unknown calendar, a unit `cftime` does not count in, a units string
            # with no `since`, an unparseable origin. Narrow on purpose, for the reason
            # the decode side gives: anything else raised in there is a defect, and
            # swallowing it would rebase the axis with no trace of the cause.
            _warn_not_encoded(name, units, f"{type(error).__name__}: {error}")
        else:
            numbers[missing] = np.nan
            result = numbers.reshape(values.shape)
    return result


def _encode_temporal_array(
    values: np.ndarray, encoding: dict[str, Any] | None = None, name: str = ""
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode datetime64/timedelta64 arrays to CF-numeric seconds (GDAL has no datetime dtype).

    A CF-decoded xarray time axis is `datetime64[ns]`, which `numpy_to_gdal_dtype` cannot map, so
    `from_xarray` crashed on any Dataset opened with the default `decode_cf=True`. Encode such arrays to
    float64 seconds with a CF `units` (and `calendar` for absolute times) so the MDArray stores a
    numeric axis that CF-decodes back to the same instants (ARC-17). A `NaT` maps to `NaN` (a missing
    instant), not the int64 sentinel's bogus ~year-1677 value.

    The CF-portable `seconds since ...` unit is float64, so a timestamp far from the 1970 epoch carries
    only ~sub-microsecond precision (float64 has ~0.35 µs resolution near 2e9 s); an exact nanosecond
    round-trip would need a non-portable `nanoseconds since ...` unit.

    An `encoding` naming CF time `units` takes precedence over that epoch: it is what the axis was
    decoded from, so encoding back into it returns the file's own numbers. Without it a file read with
    `to_xarray()` and written back came home as `seconds since 1970-01-01` on a `proleptic_gregorian`
    calendar it never declared — the same instants, but a rewritten axis (review round-1 M4).

    Args:
        values: The raw coordinate / variable array.
        encoding: The xarray `encoding` of the array, if any. A CF `units` key (with an optional
            `calendar`) is encoded back into; anything else is ignored.
        name: The array's name, used only to name it if the declared units cannot be used.

    Returns:
        A `(encoded_values, cf_attrs)` pair. For a non-temporal array the values are returned unchanged
        with an empty attribute dict; for a temporal array the values are float64 offsets (`NaN` where
        the input was `NaT`) and `cf_attrs` carries the CF `units` (plus `calendar` for absolute
        datetimes) they are counted in.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        declared = (encoding or {}).get("units")
        if is_cf_time_units(declared):
            stated_calendar = (encoding or {}).get("calendar")
            offsets = _encode_in_declared_units(
                values, str(declared), stated_calendar or "standard", name
            )
            if offsets is not None:
                # An undeclared calendar stays undeclared: `standard` is the CF
                # default, so writing it would add an attribute the source never had.
                cf: dict[str, Any] = {"units": declared}
                if stated_calendar:
                    cf["calendar"] = stated_calendar
                return offsets, cf
        as_ns = values.astype("datetime64[ns]")
        seconds = as_ns.astype("int64").astype("float64") / 1e9
        # `np.where` keeps this scalar-safe: a 0-d input's `/ 1e9` is a NumPy scalar that does not
        # support in-place item assignment (review round-2 M1).
        seconds = np.where(np.isnat(as_ns), np.nan, seconds)
        # Counting in *fractional* seconds is this path's own choice: it has to
        # carry `NaT` as `NaN`, which an integer count cannot. Dividing the
        # nanosecond count rather than casting to `datetime64[s]` is deliberate
        # too -- the cast truncates, losing sub-second times. Only the epoch and
        # calendar are shared, with the module that decodes them.
        return seconds, {
            "units": cf_epoch_units("seconds"),
            "calendar": CF_EPOCH_CALENDAR,
        }
    if np.issubdtype(values.dtype, np.timedelta64):
        as_ns = values.astype("timedelta64[ns]")
        seconds = as_ns.astype("int64").astype("float64") / 1e9
        seconds = np.where(np.isnat(as_ns), np.nan, seconds)
        return seconds, {"units": "seconds"}
    return values, {}


def _apply_md_array_attrs(md_arr: gdal.MDArray, attrs: dict[str, Any]) -> None:
    """Write MDArray attributes, routing the CF `units` through `SetUnit`.

    GDAL's netCDF writer moves the CF `units` attribute onto the MDArray's own
    unit slot; if we also write it as a regular attribute it is dropped on the
    next `CreateCopy`. Split it out so the round trip is lossless.

    Args:
        md_arr: The target multidimensional array.
        attrs: Attributes to attach; a `units` key is applied via `SetUnit`.
    """
    if not attrs:
        return
    remaining = dict(attrs)
    unit = remaining.pop("units", None)
    if unit is not None:
        md_arr.SetUnit(str(unit))
    if remaining:
        write_attributes_to_md_array(md_arr, remaining)


def _write_md_array_streamed(md_arr: gdal.MDArray, arr: Any) -> None:
    """Write ``arr`` into ``md_arr``, streaming a dask array one block at a time.

    A dask array is written block by block -- each block is computed, written to its
    hyperslab, then released before the next -- so a lazily-loaded variable never
    becomes fully resident (ARC-48). Any other array-like is written in one hyperslab.

    Args:
        md_arr: The full-shape destination MDArray.
        arr: A dask array (streamed block by block) or a NumPy array (written whole).
    """
    if not (hasattr(arr, "dask") and hasattr(arr, "blocks")):
        md_arr.Write(np.ascontiguousarray(np.asarray(arr)))
        return
    for block_id in np.ndindex(*arr.numblocks):
        block = np.ascontiguousarray(np.asarray(arr.blocks[block_id]))
        starts = [
            int(sum(arr.chunks[axis][: block_id[axis]])) for axis in range(arr.ndim)
        ]
        md_arr.Write(block, array_start_idx=starts, count=list(block.shape))


def _write_data_var(
    root: gdal.Group,
    gdal_dims: dict[str, Any],
    dims: dict[str, int],
    var_name: str,
    var_dims: tuple[str, ...],
    var_values: Any,
    var_attrs: dict[str, Any],
    var_encoding: dict[str, Any] | None = None,
) -> gdal.MDArray:
    """Create and fill one data variable's MDArray, streaming a dask-backed one block by block.

    A dask-backed variable (a lazily-loaded xarray var passed through by
    `_build_multidim_from_xarray`) is written block by block so it never becomes fully resident; a
    NumPy variable, or a temporal one that must be CF-encoded, is materialised and written in one
    shot (the prior behaviour).

    Args:
        root: The root group the MDArray is created in.
        gdal_dims: Dimension name to the `gdal.Dimension` already created for it.
        dims: Dimension name to length, used to check the variable's shape.
        var_name: The variable's name.
        var_dims: The dimension names the variable spans, outermost first.
        var_values: The variable's values — a NumPy array, or a dask-backed array to stream.
        var_attrs: The variable's own CF attributes.
        var_encoding: The xarray `encoding` of the variable, if any. Only a temporal variable
            reads it, to be written back in the CF `units` it was decoded from rather than in
            the writer's own epoch; `None` (the default) is what every GDAL-native caller passes.

    Returns:
        gdal.MDArray: The created (and filled) data-variable MDArray, so the caller can attach a
            `grid_mapping` link to it.

    Raises:
        ValueError: When the variable references an unknown dimension, or its shape does not match
            the sizes implied by its dimensions.
    """
    unknown = [d for d in var_dims if d not in gdal_dims]
    if unknown:
        raise ValueError(
            f"variable {var_name!r} references unknown dimension(s) "
            f"{unknown} not in dims {sorted(gdal_dims)}"
        )
    dtype = np.dtype(getattr(var_values, "dtype", None) or np.asarray(var_values).dtype)
    temporal = np.issubdtype(dtype, np.datetime64) or np.issubdtype(
        dtype, np.timedelta64
    )
    stream = hasattr(var_values, "dask") and var_values.ndim > 0 and not temporal
    if stream:
        values: Any = var_values
        cf_attrs: dict[str, Any] = {}
        shape = tuple(var_values.shape)
        write_dtype = dtype
    else:
        values, cf_attrs = _encode_temporal_array(
            np.asarray(var_values), var_encoding, var_name
        )
        shape = values.shape
        write_dtype = values.dtype
    expected = tuple(dims[d] for d in var_dims)
    if shape != expected:
        raise ValueError(
            f"variable {var_name!r} has shape {shape} but its "
            f"dimensions {tuple(var_dims)} imply {expected}"
        )
    if np.dtype(write_dtype).kind in ("U", "S", "O"):
        # `numpy_to_gdal_dtype` has no numeric code for a character column, and
        # `ReadAsArray` / `Write` cannot carry one through SWIG. GDAL's string
        # extended type plus the Python list `Write()` can -- the same channel
        # `_add_md_array_to_group` uses to carry ERA5's `expver` through a
        # container op (#565). Without it a CF label array is unwritable, so an
        # export carrying one either raised or lost it.
        ext = gdal.ExtendedDataType.CreateString()
        md_arr = root.CreateMDArray(var_name, [gdal_dims[d] for d in var_dims], ext)
        md_arr.Write(np.asarray(values).astype(str).tolist())
    else:
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(np.dtype(write_dtype)))
        md_arr = root.CreateMDArray(var_name, [gdal_dims[d] for d in var_dims], ext)
        _write_md_array_streamed(md_arr, values)
    merged = dict(var_attrs)
    merged.update(cf_attrs)
    _apply_md_array_attrs(md_arr, merged)
    return md_arr


def _coord_entry(
    entry: tuple[Any, ...],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Split a coordinate spec into its ``(values, attrs, encoding)`` parts.

    The encoding is optional: the GDAL-native writers pass a `(values, attrs)` pair and
    only the xarray adapter has an encoding to carry, so both shapes are accepted here
    rather than forcing every caller to append an empty dict.

    Args:
        entry: A `(values, attrs)` pair or a `(values, attrs, encoding)` triple.

    Returns:
        The values, the attributes, and the encoding (empty when the entry omitted it).

    Examples:
        - A GDAL-native caller's pair gets an empty encoding:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.engines.interop import _coord_entry
            >>> values, attrs, encoding = _coord_entry((np.array([0.0, 6.0]), {"axis": "T"}))
            >>> values.tolist(), attrs, encoding
            ([0.0, 6.0], {'axis': 'T'}, {})

            ```
        - The xarray adapter's triple carries the units the axis is written back in:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.engines.interop import _coord_entry
            >>> _, _, encoding = _coord_entry(
            ...     (np.array([0.0, 6.0]), {}, {"units": "hours since 2024-01-01"})
            ... )
            >>> encoding["units"]
            'hours since 2024-01-01'

            ```
    """
    values, attrs = entry[0], entry[1]
    encoding = entry[2] if len(entry) > 2 else None
    return values, attrs, encoding or {}


def _var_entry(
    entry: tuple[Any, ...],
) -> tuple[tuple[str, ...], Any, dict[str, Any], dict[str, Any]]:
    """Split a variable spec into its ``(dims, values, attrs, encoding)`` parts.

    The encoding is optional for the same reason it is on a coordinate: only the
    xarray adapter has one to carry, and the GDAL-native writers pass a triple.

    Args:
        entry: A `(dims, values, attrs)` triple or a `(dims, values, attrs, encoding)`
            quadruple.

    Returns:
        The dimension names, the values, the attributes, and the encoding (empty when
        the entry omitted it).

    Examples:
        - A GDAL-native caller's triple gets an empty encoding:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.engines.interop import _var_entry
            >>> dims, values, attrs, encoding = _var_entry(
            ...     (("time", "lat"), np.zeros((2, 3)), {"units": "K"})
            ... )
            >>> dims, attrs, encoding
            (('time', 'lat'), {'units': 'K'}, {})
            >>> values.shape
            (2, 3)

            ```
        - The xarray adapter's quadruple keeps the encoding it read off the variable:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.engines.interop import _var_entry
            >>> *_, encoding = _var_entry(
            ...     (("time",), np.zeros(2), {}, {"units": "days since 2000-01-01"})
            ... )
            >>> encoding["units"]
            'days since 2000-01-01'

            ```
    """
    dims, values, attrs = entry[0], entry[1], entry[2]
    encoding = entry[3] if len(entry) > 3 else None
    return dims, values, attrs, encoding or {}


def _dim_type(name: str) -> str:
    """CF dimension type for a coordinate name, or ``""`` when it is not spatial/temporal.

    Declaring the horizontal / temporal dimension types lets GDAL's netCDF writer place
    the CRS on the right axes (``SetSpatialRef`` otherwise warns that it is *assuming* the
    last two dimensions are lon/lat).

    Args:
        name: The dimension name (case-insensitive).

    Returns:
        str: ``gdal.DIM_TYPE_HORIZONTAL_X`` / ``_Y`` / ``gdal.DIM_TYPE_TEMPORAL``, or ``""``.
    """
    lowered = name.lower()
    if lowered in ("x", "lon", "longitude"):
        dim_type = str(gdal.DIM_TYPE_HORIZONTAL_X)
    elif lowered in ("y", "lat", "latitude"):
        dim_type = str(gdal.DIM_TYPE_HORIZONTAL_Y)
    elif lowered in ("time", "t"):
        dim_type = str(gdal.DIM_TYPE_TEMPORAL)
    else:
        dim_type = ""
    return dim_type


def _cf_coord_attrs(
    coord_name: str,
    coord_attrs: dict[str, Any],
    temporal_attrs: dict[str, Any],
    srs: osr.SpatialReference | None,
) -> dict[str, Any]:
    """Merge a coordinate's attributes, adding CF ``axis``/``standard_name``/``units`` for x/y.

    GDAL's multidim writer emits only what it is handed, so a spatial coordinate written with a
    bare attribute dict is unusable to a CF reader (Panoply: "X-dimension index is not set"). This
    stamps the CF axis attributes from :func:`pyramids.netcdf.cf.build_coordinate_attrs` onto x/y
    (lon/lat) coordinates. The caller's own attributes and the temporal encoding win over the CF
    defaults; a non-spatial coordinate (e.g. ``band``, or a ``time`` axis already carrying its
    ``units``/``calendar``) is left untouched.

    Args:
        coord_name: The coordinate/dimension name.
        coord_attrs: The caller-supplied coordinate attributes.
        temporal_attrs: The CF temporal encoding (``units``/``calendar``) for a datetime axis.
        srs: The dataset's spatial reference, or None. Only decides degrees vs metres units;
            the ``axis`` role is written even without a CRS.

    Returns:
        dict: The merged attribute dict to apply to the coordinate MDArray.
    """
    merged = dict(coord_attrs)
    merged.update(temporal_attrs)
    is_geographic = None if srs is None else bool(srs.IsGeographic())
    cf = build_coordinate_attrs(coord_name, is_geographic)
    if cf.get("axis") in ("X", "Y"):
        cf.update(merged)
        merged = cf
    return merged


def _apply_grid_mapping(
    srs: osr.SpatialReference, data_arrays: dict[str, gdal.MDArray]
) -> None:
    """Attach the CRS to each data MDArray so GDAL emits a CF ``grid_mapping`` variable.

    Calling ``SetSpatialRef`` on a data MDArray makes GDAL's netCDF writer auto-generate a
    scalar CF ``grid_mapping`` variable (named by GDAL from the projection — ``crs`` for
    geographic, e.g. ``transverse_mercator`` for a projected CRS — carrying
    ``grid_mapping_name`` + ``crs_wkt`` + the projection params) and link the data
    variable to it via
    ``<var>#grid_mapping`` — the same mechanism ``from_array`` uses on the netCDF driver.
    The generated variable is hidden from the multidim array listing, so it never leaks into
    ``get_variable_names`` / ``variables``, and the CRS round-trips (``MDArray.GetSpatialRef``).

    Args:
        srs: The dataset's spatial reference.
        data_arrays: Data-variable name to its MDArray.
    """
    for md_arr in data_arrays.values():
        md_arr.SetSpatialRef(srs)


def _build_multidim(
    dims: dict[str, int],
    coords: Mapping[str, CoordSpec],
    data_vars: Mapping[str, VarSpec],
    global_attrs: dict[str, Any],
    crs_wkt: str | None = None,
    aux_vars: Mapping[str, VarSpec] | None = None,
) -> gdal.Dataset:
    """Build an in-memory GDAL multidim container from plain arrays and attrs.

    The shared core behind `_build_multidim_from_xarray` and the GDAL-native
    NetCDF writers (e.g. `DatasetCollection.to_netcdf`) — neither needs a
    labeled-array dataset to reach pyramids' own multidimensional writer. Each
    coordinate becomes a 1-D indexing MDArray and each variable an N-D MDArray
    whose dimensions are resolved by name; `numpy` datetime/timedelta axes are
    CF-encoded on the way in and attributes go through pyramids' own CF helpers.

    When `crs_wkt` is given, the x/y coordinates gain CF `axis`/`standard_name`/`units`
    attributes and a scalar CF grid-mapping variable (`crs` / `transverse_mercator` /
    ..., named by GDAL from the projection) is written and linked
    from every data variable, so the file is georeferenceable by a CF reader (Panoply,
    QGIS, xarray); without it the coordinates keep only the caller's attributes.

    Args:
        dims: Dimension name to length.
        coords: Coordinate name (which must also be a dimension) to a
            `(values, attrs)` pair, or a `(values, attrs, encoding)` triple whose
            encoding names the CF `units` a decoded time axis is written back in.
            Entries whose name is not a dimension are skipped.
        data_vars: Variable name to a `(dimension-name tuple, values, attrs)`
            triple, or a `(dimension-name tuple, values, attrs, encoding)`
            quadruple carrying the same CF `units` slot a coordinate's encoding
            carries. A dask-backed `values` is streamed block by block.
        global_attrs: Root-group (global) attributes.
        crs_wkt: The dataset CRS as a WKT string, or None. Drives the CF coordinate
            attributes and the `grid_mapping` variable.
        aux_vars: Auxiliary arrays -- CF bounds, 2-D curvilinear coordinate
            fields, label columns -- in the same shape as `data_vars`, optional
            encoding slot included. They are written as ordinary MDArrays
            but are **not** linked to the grid mapping, because a coordinate
            variable is not itself georeferenced by one. `coords` cannot carry
            them: an entry there must name a dimension, and these do not.

    Returns:
        gdal.Dataset: An in-memory `MEM` multidimensional dataset ready to be
        handed to the netCDF driver's `CreateCopy`.

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate/variable array shape does not match its dimension sizes.
    """
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("pyramids")
    root = src.GetRootGroup()
    srs = srs_from_wkt(crs_wkt)

    gdal_dims: dict[str, gdal.Dimension] = {
        name: root.CreateDimension(name, _dim_type(name), "", int(size))
        for name, size in dims.items()
    }

    for coord_name, coord_spec in coords.items():
        if coord_name not in gdal_dims:
            continue
        coord_values, coord_attrs, coord_encoding = _coord_entry(coord_spec)
        values, cf_attrs = _encode_temporal_array(
            np.asarray(coord_values), coord_encoding, coord_name
        )
        if values.shape != (dims[coord_name],):
            raise ValueError(
                f"coordinate {coord_name!r} has shape {values.shape} but its "
                f"dimension is length {dims[coord_name]}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        md_arr = root.CreateMDArray(coord_name, [gdal_dims[coord_name]], ext)
        md_arr.Write(np.ascontiguousarray(values))
        _apply_md_array_attrs(
            md_arr, _cf_coord_attrs(coord_name, coord_attrs, cf_attrs, srs)
        )

    data_arrays: dict[str, gdal.MDArray] = {}
    for var_name, var_spec in data_vars.items():
        var_dims, var_values, var_attrs, var_encoding = _var_entry(var_spec)
        data_arrays[var_name] = _write_data_var(
            root,
            gdal_dims,
            dims,
            var_name,
            var_dims,
            var_values,
            var_attrs,
            var_encoding,
        )

    for var_name, var_spec in (aux_vars or {}).items():
        var_dims, var_values, var_attrs, var_encoding = _var_entry(var_spec)
        _write_data_var(
            root,
            gdal_dims,
            dims,
            var_name,
            var_dims,
            var_values,
            var_attrs,
            var_encoding,
        )

    if srs is not None:
        _apply_grid_mapping(srs, data_arrays)

    if global_attrs:
        write_global_attributes(root, dict(global_attrs))

    return src


def _crs_wkt_from_xarray(dataset: Any) -> str | None:
    """Best-effort CRS WKT from an xarray Dataset's grid-mapping variable or global attrs.

    Reads a ``spatial_ref`` / ``crs`` variable's ``crs_wkt`` / ``spatial_ref`` attribute
    (the rioxarray / CF convention), then the dataset's global attributes. Returns None
    when the source carries no CRS — ``from_xarray`` never fabricates one.

    Args:
        dataset: The source ``xarray.Dataset``.

    Returns:
        str or None: The CRS WKT, or None when the dataset declares no CRS.
    """
    result: str | None = None
    for name in ("spatial_ref", "crs"):
        if name in dataset.variables:
            attrs = dataset.variables[name].attrs
            wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
            if wkt:
                result = str(wkt)
                break
    if result is None:
        wkt = dataset.attrs.get("crs_wkt")
        result = str(wkt) if wkt else None
    return result


def _without_store_name(attrs: Any) -> dict[str, Any]:
    """An exported variable's attributes minus the export's own provenance note.

    :func:`_data_vars_from_arrays` stamps `pyramids_store_name` on a variable
    whose export key is not the store's name for it, so a caller can get from
    the key back to the name `get_variable` takes. That names the *source*
    store: in the file being written the variable really is called what the key
    says, so carrying the note across would leave a flat file asserting a group
    path it does not have, and a second export of that file would repeat it.

    Args:
        attrs: An xarray variable's or coordinate's attribute mapping.

    Returns:
        dict[str, Any]: A plain dict of the attributes, without that key.
    """
    return {
        name: value for name, value in dict(attrs).items() if name != _STORE_NAME_ATTR
    }


def _build_multidim_from_xarray(dataset: Any) -> gdal.Dataset:
    """Build an in-memory GDAL multidim container from an xarray Dataset.

    Extracts the plain `(dims, coords, aux_vars, data_vars, attrs)` spec from
    the `xarray.Dataset` and delegates to `_build_multidim`, so the GDAL
    multidim assembly lives in one place and this adapter only reads `.sizes` /
    `.coords` / `.data_vars` / `.attrs` off xarray. When the dataset carries a CRS,
    it is passed down so the x/y coordinates gain CF attributes and a `grid_mapping`
    variable is written (see :func:`_build_multidim`).

    A coordinate that is not also a dimension is passed on as an auxiliary array
    rather than dropped: xarray keeps CF bounds and 2-D curvilinear coordinate
    fields in `coords`, and for a ROMS-style store that pair is the only
    georeferencing the file has. A rank-0 (scalar) coordinate is the exception
    and is still skipped -- GDAL's multidim writer cannot take one.

    Args:
        dataset: The source `xarray.Dataset`.

    Returns:
        gdal.Dataset: The in-memory `MEM` multidimensional dataset.
    """
    crs_wkt = _crs_wkt_from_xarray(dataset)
    dims = {name: int(size) for name, size in dataset.sizes.items()}
    coords = {
        # The encoding rides along so a CF-decoded time axis is written back in the
        # units it was decoded from instead of the writer's own epoch (round-1 M4).
        name: (
            np.asarray(coord.values),
            _without_store_name(coord.attrs),
            dict(coord.encoding),
        )
        for name, coord in dataset.coords.items()
        if name in dims
    }
    # When we re-generate a grid_mapping from the resolved CRS, drop any pre-existing
    # scalar grid-mapping variable so the output carries a single one.
    skip = {"spatial_ref", "crs"} if crs_wkt is not None else set()
    # A coordinate that is not a dimension -- a CF bounds array, a 2-D
    # curvilinear `lat_rho`, a label column -- belongs to neither mapping above:
    # `coords` takes only dimension coordinates, and xarray does not list it
    # among `data_vars`. Left out of both it was written nowhere, so
    # `to_xarray()` -> `from_xarray()` silently dropped exactly the arrays the
    # export's CF promotion had just moved into `coords` (round-4 M1).
    # A rank-0 coordinate stays out: GDAL's numpy write path refuses a 0-d
    # array ("Illegal numpy array rank 1"), and pyramids' own enumeration drops
    # 0-dimensional MDArrays anyway, so no `to_xarray` export can carry one in.
    aux_vars = {
        name: (
            tuple(coord.dims),
            coord.data,
            _without_store_name(coord.attrs),
            dict(coord.encoding),
        )
        for name, coord in dataset.coords.items()
        if name not in dims and coord.ndim > 0
    }
    data_vars = {
        # `var.data` hands the underlying array through WITHOUT computing it, so a
        # dask-backed variable stays lazy and `_build_multidim` can stream it block by
        # block (ARC-48); `.values` would force a full materialisation up front.
        name: (
            tuple(var.dims),
            var.data,
            _without_store_name(var.attrs),
            dict(var.encoding),
        )
        for name, var in dataset.data_vars.items()
        if not (name in skip and var.ndim == 0)
    }
    return _build_multidim(
        dims,
        coords,
        data_vars,
        dict(dataset.attrs),
        crs_wkt=crs_wkt,
        aux_vars=aux_vars,
    )


def _create_copy_to_netcdf(mem_src: gdal.Dataset, path: str) -> None:
    """CreateCopy an in-memory multidim dataset to a netCDF file on disk.

    Args:
        mem_src: The in-memory `MEM` multidimensional source dataset.
        path: Destination `.nc` path.

    Raises:
        RuntimeError: When the GDAL netCDF writer returns no dataset.
    """
    dst = gdal.GetDriverByName("netCDF").CreateCopy(path, mem_src, 0)
    if dst is None:
        raise RuntimeError(f"Failed to write NetCDF to {path}")
    dst.FlushCache()
    # Release the write handle here rather than relying on scope-exit GC: an
    # open netCDF write handle can leave the on-disk file unrecognised by a
    # reader that reopens the same path (e.g. from_xarray's read_file).
    dst = None


def write_multidim_netcdf(
    path: str | Path,
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]],
    global_attrs: dict[str, Any],
    crs_wkt: str | None = None,
) -> None:
    """Write a plain multidim spec to a NetCDF file through GDAL.

    Assembles the `(dims, coords, data_vars, global_attrs)` spec into an
    in-memory GDAL multidimensional dataset via `_build_multidim` and copies it
    out with the netCDF driver — the same writer `NetCDF.from_xarray` uses, so a
    caller that already holds `numpy` arrays never has to build a
    labeled-array dataset just to emit a NetCDF.

    Args:
        path: Output `.nc` path.
        dims: Dimension name to length.
        coords: Coordinate name to a `(values, attrs)` pair.
        data_vars: Variable name to a `(dimension-name tuple, values, attrs)`
            triple.
        global_attrs: Root-group (global) attributes.
        crs_wkt: The dataset CRS as a WKT string, or None. When given, the x/y
            coordinates gain CF attributes and a `grid_mapping` variable is written
            (see :func:`_build_multidim`).

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate/variable array shape does not match its dimension sizes.
        RuntimeError: When the GDAL netCDF writer fails to create the file.
    """
    mem_src = _build_multidim(dims, coords, data_vars, global_attrs, crs_wkt)
    _create_copy_to_netcdf(mem_src, str(path))


class _StreamingMultidimWriter:
    """Fills a netCDF multidim file's data variables one hyperslab at a time.

    Created by :func:`open_streaming_multidim_netcdf`. Each data variable is
    created at its full shape but written incrementally: :meth:`write_slab` writes
    a single index of the leading (streamed) dimension, so the whole cube is never
    resident in memory. The owning context manager finalizes the file on exit.
    """

    def __init__(self, arrays: dict[str, gdal.MDArray]) -> None:
        """Store the per-variable MDArrays to stream into.

        Args:
            arrays: Variable name to its (empty, full-shape) MDArray.
        """
        self._arrays = arrays

    def write_slab(self, var_name: str, index: int, block: np.ndarray) -> None:
        """Write one leading-dimension index of a variable.

        Args:
            var_name: Target data variable.
            index: Position along the leading (streamed) dimension to write at.
            block: The array for this index, i.e. the variable's shape with the
                leading dimension dropped. A length-1 leading axis is prepended
                before the hyperslab write.
        """
        md_arr = self._arrays[var_name]
        block = np.ascontiguousarray(block)
        md_arr.Write(
            block[np.newaxis, ...],
            array_start_idx=[int(index)] + [0] * block.ndim,
            count=[1] + list(block.shape),
        )

    def write_whole(self, var_name: str, array: np.ndarray) -> None:
        """Write an entire (non-streamed) variable in one hyperslab.

        For a variable with no streamed leading dimension -- a 2-D ``(y, x)`` grid, or a small
        carried-through auxiliary variable -- there is no slab to iterate, so the full array is
        written at once.

        Args:
            var_name: Target variable.
            array: The variable's full array, matching its declared shape.
        """
        self._arrays[var_name].Write(np.ascontiguousarray(np.asarray(array)))


def _build_streaming_multidim(
    dataset: gdal.Dataset,
    dims: dict[str, int],
    coords: Mapping[str, CoordSpec],
    var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]],
    global_attrs: dict[str, Any],
    crs_wkt: str | None = None,
) -> dict[str, gdal.MDArray]:
    """Create dims, coord arrays, empty data vars, and global attrs on `dataset`.

    Split out from :func:`open_streaming_multidim_netcdf` so all the transient
    GDAL setup handles (the root group, dimensions, and coordinate MDArrays) live
    in this frame and are released when it returns or raises — leaving the context
    manager only the data-variable MDArrays to drop before `Close()`, which is
    what lets the netCDF driver flush and unlock the file.

    When `crs_wkt` is given, the x/y coordinates gain CF `axis`/`standard_name`/`units`
    attributes and a scalar CF grid-mapping variable (`crs` / `transverse_mercator` /
    ..., named by GDAL from the projection) is written and linked
    from every data variable, so a CF reader can georeference the streamed file.

    Args:
        dataset: A freshly created netCDF multidim dataset.
        dims: Dimension name to length.
        coords: Coordinate name to a ``(values, attrs)`` pair, or a
            ``(values, attrs, encoding)`` triple whose encoding names the CF
            ``units`` a decoded time axis is written back in; entries whose name
            is not a dimension are skipped.
        var_specs: Variable name to a ``(dimension-name tuple, numpy dtype,
            attrs)`` triple.
        global_attrs: Root-group (global) attributes.
        crs_wkt: The dataset CRS as a WKT string, or None. Drives the CF coordinate
            attributes and the `grid_mapping` variable.

    Returns:
        dict[str, gdal.MDArray]: The created (empty, full-shape) data-variable
        MDArrays, keyed by name, for the caller to fill by slab.

    Raises:
        ValueError: When a variable references an unknown dimension, or a
            coordinate array shape does not match its dimension size.
    """
    root = dataset.GetRootGroup()
    srs = srs_from_wkt(crs_wkt)
    gdal_dims: dict[str, gdal.Dimension] = {
        name: root.CreateDimension(name, _dim_type(name), "", int(size))
        for name, size in dims.items()
    }

    for coord_name, coord_spec in coords.items():
        if coord_name not in gdal_dims:
            continue
        coord_values, coord_attrs, coord_encoding = _coord_entry(coord_spec)
        values, cf_attrs = _encode_temporal_array(
            np.asarray(coord_values), coord_encoding, coord_name
        )
        if values.shape != (dims[coord_name],):
            raise ValueError(
                f"coordinate {coord_name!r} has shape {values.shape} but its "
                f"dimension is length {dims[coord_name]}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values))
        coord_arr = root.CreateMDArray(coord_name, [gdal_dims[coord_name]], ext)
        coord_arr.Write(np.ascontiguousarray(values))
        _apply_md_array_attrs(
            coord_arr, _cf_coord_attrs(coord_name, coord_attrs, cf_attrs, srs)
        )

    arrays: dict[str, gdal.MDArray] = {}
    for var_name, (var_dims, var_dtype, var_attrs) in var_specs.items():
        unknown = [d for d in var_dims if d not in gdal_dims]
        if unknown:
            raise ValueError(
                f"variable {var_name!r} references unknown dimension(s) "
                f"{unknown} not in dims {sorted(gdal_dims)}"
            )
        ext = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(np.dtype(var_dtype)))
        md_arr = root.CreateMDArray(var_name, [gdal_dims[d] for d in var_dims], ext)
        # Declare a real CF `_FillValue` *before* the caller streams any slab: netCDF rejects a
        # fill value once data exists, so this must happen at creation. GDAL surfaces it as the
        # `_FillValue` attribute, which CF readers (Panoply, xarray, QGIS) mask missing data on —
        # they never honor the bare `nodata` attribute this writer also keeps for round-trip (#1061).
        fill = var_attrs.get("nodata")
        if fill is not None:
            md_arr.SetNoDataValueDouble(float(fill))
        _apply_md_array_attrs(md_arr, dict(var_attrs))
        arrays[var_name] = md_arr

    if srs is not None:
        _apply_grid_mapping(srs, arrays)

    if global_attrs:
        write_global_attributes(root, dict(global_attrs))

    return arrays


@contextmanager
def open_streaming_multidim_netcdf(
    path: str | Path,
    dims: dict[str, int],
    coords: dict[str, tuple[np.ndarray, dict[str, Any]]],
    var_specs: dict[str, tuple[tuple[str, ...], np.dtype | str, dict[str, Any]]],
    global_attrs: dict[str, Any],
    crs_wkt: str | None = None,
):
    """Create a netCDF multidim file and yield a per-hyperslab writer.

    The streaming counterpart of :func:`write_multidim_netcdf`: instead of
    receiving fully materialised variable arrays, it creates the dimensions,
    writes the (small, 1-D) coordinate arrays whole, creates each data variable
    at its full shape with **no data**, then hands back a
    :class:`_StreamingMultidimWriter` so the caller can fill each variable one
    leading-dimension slab at a time, so the whole ``(T, …)`` cube never has to
    be resident. The write is atomic: the file is created at a temporary sibling
    path and only ``os.replace``-d onto ``path`` after a clean close, so ``path``
    only ever holds a complete file and an existing file there survives a failed
    write. Variable data is written raw at its declared dtype — unlike
    :func:`write_multidim_netcdf`, ``datetime64`` / ``timedelta64`` *variable*
    data is not CF-encoded here (coordinates still are), so variable dtypes must
    be GDAL-mappable numerics (the sole caller streams numeric raster bands).
    Used by :meth:`pyramids.dataset.collection.DatasetCollection.to_netcdf`.

    Args:
        path: Output ``.nc`` path.
        dims: Dimension name to length (the full shape, including the streamed
            leading dimension).
        coords: Coordinate name (also a dimension) to a ``(values, attrs)`` pair;
            written whole up front. Entries whose name is not a dimension are
            skipped.
        var_specs: Variable name to a ``(dimension-name tuple, numpy dtype,
            attrs)`` triple. The first entry of the dimension tuple is the
            streamed (leading) dimension.
        global_attrs: Root-group (global) attributes.
        crs_wkt: The dataset CRS as a WKT string, or None. When given, the x/y
            coordinates gain CF attributes and a `grid_mapping` variable is written
            (see :func:`_build_streaming_multidim`).

    Yields:
        _StreamingMultidimWriter: Call :meth:`~_StreamingMultidimWriter.write_slab`
        once per leading-dimension index.

    Raises:
        RuntimeError: When the GDAL netCDF driver fails to create the file.
        ValueError: When a variable references an unknown dimension, or a
            coordinate array shape does not match its dimension size.
    """
    final_path = Path(path)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    with suppress(OSError):
        tmp_path.unlink()  # drop any stale temp left by a crashed prior run
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(tmp_path))
    if dataset is None:
        raise RuntimeError(f"Failed to create NetCDF at {path}")
    completed = False
    arrays: dict[str, gdal.MDArray] = {}
    try:
        arrays = _build_streaming_multidim(
            dataset, dims, coords, var_specs, global_attrs, crs_wkt
        )
        yield _StreamingMultidimWriter(arrays)
        completed = True
    except BaseException as exc:
        # The in-flight exception's traceback also pins the caller's `writer`
        # (holding the same MDArrays); clear the failure frames so those handles
        # release and the temp can be removed below (the executing generator
        # frame is skipped by clear_frames). Deliberate trade-off: this empties
        # the failure frames' locals, so a post-mortem sees none here — do not
        # "restore" them or the file lock on Windows returns.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # `arrays` is the writer's own dict, so clearing it drops the last refs to
        # the data-variable MDArrays; GDAL only flushes and unlocks the file once
        # those child handles are gone (needed for os.replace / unlink on Windows).
        arrays.clear()
        if completed:
            # Close() drives the flush; let a flush error surface (the write did
            # not truly succeed). Then atomically promote the temp onto `path`.
            dataset.Close()
            os.replace(tmp_path, final_path)
        else:
            # Best-effort cleanup that never masks the in-flight exception and
            # never touches an existing file at `path`: release the handle, then
            # drop the temp.
            with suppress(Exception):
                dataset.Close()
            with suppress(OSError):
                tmp_path.unlink()
