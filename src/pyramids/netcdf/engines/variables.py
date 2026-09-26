"""Variable-mutation engine for :class:`pyramids.netcdf.NetCDF`.

Owns the bodies of the variable add/remove/rename/write family extracted
from the ``netcdf.py`` god-object (issue #615, STR-1):

- :meth:`Variables.set_variable` — write a classic ``Dataset`` back as an
  MDArray variable (the inverse of ``get_variable``).
- :meth:`Variables.add_variable` — copy MDArray variables from another
  container.
- :meth:`Variables.remove_variable` / :meth:`Variables.rename_variable` —
  delete / rename a variable.

The public ``NetCDF`` methods are thin façades delegating here; signatures,
behaviour, and return types are unchanged. Each method reaches the container's
own GDAL plumbing (``_writable_root_group`` / ``_replace_raster`` /
``_invalidate_caches`` / ``_get_or_create_dimension`` /
``_add_md_array_to_group``) through the weakref-proxied back-reference
``self._ds``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from osgeo import gdal, osr

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.base._utils import (
    _is_identity_packing,
    numpy_to_gdal_dtype,
    write_packing,
)
from pyramids.base.crs import sr_from_epsg, sr_from_user_input
from pyramids.base.georeference import GeoReference
from pyramids.dataset import DEFAULT_NO_DATA_VALUE, Dataset
from pyramids.dataset._driver import MEMORY_DRIVER, resolve_output_driver
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.transform import GeoTransform
from pyramids.netcdf._mdim import open_mdarray, scalar_no_data, unflatten_band_axes
from pyramids.netcdf.array_options import (
    CFAttributes,
    Encoding,
    ExtraDimensions,
)
from pyramids.netcdf.cf import (
    srs_to_grid_mapping,
    write_attributes_to_md_array,
    write_global_attributes,
)
from pyramids.netcdf.dimensions import COLUMN_AXIS, ROW_AXIS, ClassicDimensionInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import Container, NetCDF


class Variables(_Engine["NetCDF"]):
    """Variable add / remove / rename / write collaborator for :class:`NetCDF`.

    Owns the bodies of the variable-mutation family. ``NetCDF`` wires one
    instance per container as ``nc.varops`` (deliberately not ``variables`` —
    that name is the read-side property returning the lazy variable dict) and
    exposes thin façades, so ``nc.set_variable(...)`` and
    ``nc.varops.set_variable(...)`` are equivalent. The companion constructor
    :func:`from_array` is a module-level function (it builds a new
    container rather than mutating an existing one), reached through the
    ``NetCDF.from_array`` classmethod façade.

    Each method reaches the container's GDAL plumbing
    (``_writable_root_group`` / ``_replace_raster`` / ``_invalidate_caches`` /
    ``_get_or_create_dimension`` / ``_add_md_array_to_group``) through the
    weakref-proxied back-reference :attr:`_ds` inherited from
    :class:`~pyramids.dataset.engines._base._Engine`.
    """

    def set_variable(
        self,
        variable_name: str,
        dataset: Dataset,
        band_dim_name: str | None = None,
        band_dim_values: list | None = None,
        attrs: dict | None = None,
        *,
        copy: bool = True,
        dim_attrs: dict[str, dict[str, str]] | None = None,
    ):
        """Write a classic Dataset back as an MDArray variable in this container.

        This is the reverse of `get_variable()`. After performing GIS
        operations (crop, reproject, etc.) on a variable subset, use this
        method to store the result back into the NetCDF container.

        The dataset's **stored** values are written (`read_array(unpack=False)`) --
        this copies the store -- and the recipe that describes them is set on the
        new variable as `scale_factor` / `add_offset`, so a CF-packed raster stays
        packed and still reads back in physical units. The recipe is the one a read
        of the dataset applies, `dataset._effective_packing(0)`: a `NetCDF`
        variable's own `_scale` / `_offset` when it carries them (what `sel()`
        builds, over a band that declares none), else band 1's. An identity (or
        unusable) pair is not written as a declaration. A destination that refuses
        the packing is logged at `DEBUG` level and the counts are written without
        it. An MDArray holds a single packing, so a multi-band raster whose bands
        are packed differently keeps only that one recipe.

        Args:
            variable_name: Name for the variable in this container. If a
                variable with this name already exists it is replaced.
            dataset: A classic raster dataset, typically the result of a
                GIS operation on a variable obtained via `get_variable()`.
            band_dim_name: Name of the dimension that maps to bands
                (e.g. `"time"`, `"bands"`). Auto-detected from the
                dataset's `_band_dim_name` attribute when available.
                Defaults to None.
            band_dim_values: Coordinate values for the band dimension.
                Auto-detected from `_band_dim_values` when available.
                Defaults to None.
            attrs: Variable attributes to set (e.g. `{"units": "K"}`).
                Auto-detected from `_variable_attrs` when available.
                Defaults to None.
            dim_attrs: CF `(units, calendar)` to write onto a band dimension this call
                creates, keyed by dimension name — `{"level": {"units": "hPa"}}`. Only a
                newly created, labelled dimension is stamped; a dimension already in the
                store keeps what it was created with. A join uses this so a variable it
                adds after the first keeps its band axis' calendar through `to_file`
                (#1179). Defaults to None.
            copy: When True (the default) an in-memory container is copied
                before mutation so that any handle sharing the same backing
                `gdal.Dataset` (a `get_group()` view, or a caller holding
                `_raster`) is not corrupted — see #143. Pass False only from a
                caller that exclusively owns this container (e.g. the internal
                per-variable fan-out builders) to mutate it in place and avoid a
                per-call copy. Ignored (a copy is always made) for file-backed
                containers, which must copy to escape netCDF data mode, and for a
                `get_group()` view, whose raster is shared with its parent.
                Defaults to True.

        Raises:
            ValueError: If called on a dataset without a root group
                (not opened in multidimensional mode).

        Examples:
            - A packed raster is stored as its counts under its own recipe, so the new
              variable still reads back in physical units:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from pyramids.netcdf import NetCDF
                >>> geo_ref = GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326)
                >>> nc = NetCDF.from_array(
                ...     np.zeros((1, 2, 2), dtype="float32"), geo_ref=geo_ref, variable_name="base"
                ... )
                >>> packed = Dataset.from_array(
                ...     np.array([[100, 200], [300, 400]], dtype="int16"), geo_ref=geo_ref
                ... )
                >>> packed.scale = [0.01]
                >>> nc.set_variable("packed", packed)
                >>> stored = nc.get_variable("packed")
                >>> np.asarray(stored.read_array(unpack=False)).tolist()
                [[100, 200], [300, 400]]
                >>> stored.read_array().tolist()
                [[1.0, 2.0], [3.0, 4.0]]

                ```
        """
        nc = self._ds
        rg = nc._working_group()
        if rg is None:
            raise ValueError(
                "set_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )
        # CreateMDArray / DeleteMDArray / CreateDimension are rejected on a file-backed group (netCDF
        # data mode) so must go through a MEM copy. For an in-memory container, copying also prevents
        # corrupting a handle that shares the same gdal.Dataset (a get_group() view, or a caller holding
        # _raster) — #143. Skip the copy only when the caller exclusively owns this container and opts
        # out with copy=False, mutating the live working group in place. A get_group() view (_group_path
        # set) shares its parent's raster, so it always copies regardless of the flag.
        if copy or nc.driver_type != "memory" or nc._group_path:
            work, rg = nc._writable_root_group()
            nc._replace_raster(work)

        band_dim_name, band_dim_values, attrs, band = _resolve_band_metadata(
            dataset, band_dim_name, band_dim_values, attrs
        )

        # Delete existing variable if present. Existence is a question about
        # the store, so an aux array under this name counts as a collision.
        if variable_name in nc._readable_variable_names():
            rg.DeleteMDArray(variable_name)

        # Read data from the classic dataset. `unpack=False`: this writes the raster
        # back into the store, and the packing is carried onto the MDArray below, so
        # what belongs in it is the counts that recipe describes.
        arr = dataset.read_array(unpack=False)
        gt: tuple[float, float, float, float, float, float] = dataset.geotransform
        data_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        # Spatial coordinate dimensions must always be float64 to avoid
        # truncation when the data array is integer (e.g., classified rasters).
        coord_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Build the spatial coordinate axes from the geotransform. x_axis/y_axis read
        # the signed pixel width/height (geotransform[1]/[5]); the geo is the input
        # Dataset's own, not a read-normalised one, so a south-up input (gt[5] > 0)
        # reaches this site and writes an ascending y coordinate within its extent
        # (abs(gt[5]) would write a descending axis below the extent).
        x_values = GeoTransform(*gt).x_axis(dataset.columns)
        y_values = GeoTransform(*gt).y_axis(dataset.rows)
        # By what the axis holds, not by what it is called: a store on `longitude` /
        # `latitude` used to gain an `x` / `y` pair beside its own, describing the same
        # grid, with the new variable declared against the pair the store never had
        # (#1194).
        dim_x, dim_y = nc._spatial_axes(
            rg, x_values, y_values, coord_dtype, (ROW_AXIS, COLUMN_AXIS)
        )

        md_arr = _build_variable_mdarray(
            nc,
            rg,
            variable_name,
            arr,
            dim_y,
            dim_x,
            data_dtype,
            coord_dtype,
            band_dim_name,
            band_dim_values,
            band,
            dim_attrs,
        )

        # Set spatial reference (RT-7: attribute copying). Carry a no-EPSG CRS
        # (e.g. geostationary) through its WKT so set_variable / add_variable does
        # not silently erase the georeference (#706).
        if dataset.epsg:
            md_arr.SetSpatialRef(sr_from_epsg(dataset.epsg))
        elif dataset.crs:
            md_arr.SetSpatialRef(sr_from_user_input(dataset.crs))

        # GDAL keeps `scale_factor` / `add_offset` in the MDArray's own slots rather
        # than in its attribute dictionary, so the attribute write below cannot carry
        # them; without this a packed raster written back would lose the recipe for
        # the counts just stored.
        #
        # Through `_effective_packing`, not off band 1. A `NetCDF` variable can hold its
        # recipe only in `_scale` / `_offset` over a band that declares none -- which is
        # what `sel()` builds -- so reading the band wrote the counts back with no recipe
        # at all: `sel` -> process -> `set_variable`, the round trip this method exists
        # for, stored 3.5 as a bare 200. The identity is not written out as a
        # declaration, for the same reason the cube writer skips it.
        scale, offset = dataset._effective_packing(0)
        if not _is_identity_packing(scale, offset) and not write_packing(
            md_arr, scale, offset
        ):
            # The one carry site that used to swallow a refusal silently; every other
            # one reports it, so a driver that cannot store packing is visible here too.
            logger.debug("the destination refused the packing for %r", variable_name)

        # Set no-data value
        if dataset.no_data_value and dataset.no_data_value[0] is not None:
            try:
                md_arr.SetNoDataValueDouble(float(dataset.no_data_value[0]))
            except (RuntimeError, TypeError, ValueError):
                pass  # nosec B110

        # Set variable attributes (RT-7)
        if attrs:
            write_attributes_to_md_array(md_arr, attrs)

        nc._invalidate_caches()

    def add_variable(
        self,
        dataset: Dataset | NetCDF,
        variable_name: str | None = None,
        *,
        copy: bool = True,
    ):
        """Copy MDArray variables from another NetCDF into this container.

        Args:
            dataset: Source NetCDF dataset whose variables will be copied.
                Must have a root group (opened in MDIM mode).
            variable_name: Specific variable name(s) to copy. If None, all
                variables from the source are copied. A group-qualified source
                name (`"flight_03/CO"`) is copied under its leaf name, because
                this container's root group is flat and netCDF-4 forbids `/` in
                a name; where that collides with a name already present the copy
                is suffixed (`"-new"`, then numbered) rather than overwriting.
            copy: When True (the default) an in-memory container is copied
                before mutation so a shared `gdal.Dataset` handle is not
                corrupted — see #143. Pass False only from a caller that
                exclusively owns this container (e.g. the internal aux-variable
                carry loop) to mutate it in place and avoid a per-call copy.
                Ignored (a copy is always made) for file-backed containers and
                for a `get_group()` view. Defaults to True.

        Raises:
            ValueError: If called on a dataset without a root group (not
                opened in multidimensional mode), or if a requested
                `variable_name` does not resolve in the source. An unresolved
                name used to be skipped, so a typo returned `None` and left
                this container unchanged with nothing said.
        """
        nc = self._ds
        working_group = nc._working_group()
        if working_group is None:
            raise ValueError(
                "add_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )
        # A NetCDF source may be a group view; read its working group so variables
        # are copied from the active sub-group. A plain Dataset has no group view.
        var_rg = (
            dataset._working_group()
            if hasattr(dataset, "_working_group")
            else dataset._raster.GetRootGroup()
        )
        names_to_copy = _names_to_copy(dataset, variable_name)

        # A file-backed root group is opened in netCDF "data mode", which forbids
        # CreateMDArray; and mutating an in-memory container in place would corrupt a
        # handle sharing the same gdal.Dataset (#143). Copy and swap unless the caller
        # exclusively owns this container and opts out with copy=False (a get_group()
        # view always copies — it shares its parent's raster).
        in_place = not copy and nc.driver_type == "memory" and not nc._group_path
        if in_place:
            dst, dst_rg = nc._raster, working_group
        else:
            dst, dst_rg = nc._writable_root_group()

        for var in names_to_copy:
            # Group-qualified names reach here from the variable
            # enumeration, so resolve through the helper that walks them.
            md_arr = open_mdarray(var_rg, var)
            if md_arr is None:
                _refuse_unresolved_source(dataset, var_rg, var)
            # If the variable name already exists in the destination dataset,
            # use a suffixed name to avoid overwriting the original.
            existing = dst_rg.GetMDArrayNames() or []
            target_name = _free_target_name(var, existing)
            nc._add_md_array_to_group(dst_rg, target_name, md_arr)

        if not in_place:
            nc._replace_raster(dst)
        nc._invalidate_caches()

    def remove_variable(self, variable_name: str):
        """Delete a variable from this container.

        An independent MEM copy is always made first and the internal raster is
        replaced with it, so neither the on-disk file (for a file-backed dataset)
        nor any handle sharing the in-memory raster is mutated in place (#143).

        Args:
            variable_name: Name of the variable to remove, relative to the
                container's working group. A group-qualified name is refused;
                see `Raises`.

        Raises:
            ValueError: If `variable_name` is group-qualified (open the group
                and remove the leaf name from the view instead), or if the
                working group holds no array by that name. Note that the name
                need not be in :attr:`variable_names` or the readable superset:
                a dimension coordinate array (`lat`) is in neither and is
                removable, so the guard resolves the array rather than
                consulting an enumeration.
        """
        nc = self._ds
        _refuse_group_qualified(variable_name, "remove_variable")
        dst, rg = nc._writable_root_group()
        try:
            rg.DeleteMDArray(variable_name)
        except RuntimeError as exc:
            # GDAL answers a missing name with "Array <name> is not an array of
            # this group", which reads as an internal error rather than a typo.
            # `rename_variable` already promised `ValueError` for the same
            # mistake, so both mutators now agree.
            raise ValueError(
                f"Variable '{variable_name}' not found in this container's "
                f"working group. Available: {nc._readable_variable_names()} "
                "(a dimension coordinate array is removable too, though it is "
                "not listed there)."
            ) from exc

        nc._replace_raster(dst)

    def rename_variable(self, old_name: str, new_name: str):
        """Rename a variable in this container.

        Internally extracts the variable data and metadata, creates
        a new variable with the new name, and removes the old one.

        Args:
            old_name: Current name of the variable, relative to the container's
                working group.
            new_name: Desired new name, in that same group.

        Raises:
            ValueError: If `old_name` does not resolve in the working group, if
                it resolves to a **dimension coordinate array** (which cannot
                be renamed — the dimension of that name would keep pointing at
                the deleted array), if `new_name` already exists, or if either
                name is group-qualified — a rename acts within one group, so
                `"flight_03/CO"` is refused with a pointer at
                `get_group("flight_03").rename_variable("CO", ...)`, which does
                work.

        Note:
            Existence is decided by resolving the array, as
            :meth:`remove_variable` decides it, not by consulting
            :meth:`_readable_variable_names`. Gating on the enumeration made
            the two mutators contradict each other: `remove_variable("lat")`
            deleted the array while this method reported the same name "not
            found".
        """
        nc = self._ds
        _refuse_group_qualified(old_name, "rename_variable", ", <new name>")
        if "/" in new_name:
            # A qualified target would create a root-level array literally named
            # "group/leaf" -- netCDF-4 forbids `/` in a name, so the container
            # would only fail at `to_file` with "Name contains illegal
            # characters", long after the call that caused it (the same trap
            # `_free_target_name` closes for `add_variable`).
            raise ValueError(
                f"rename_variable() cannot rename into a group: new_name "
                f"{new_name!r} must not contain '/'. Open the destination "
                "group with `get_group(...)` and rename within it."
            )
        # Both checks ask what the store holds: an aux array is renameable,
        # and its name is equally taken.
        readable = nc._readable_variable_names()
        working_group = nc._working_group()
        if working_group is None:
            raise ValueError("rename_variable requires a multidimensional container.")
        source = open_mdarray(working_group, old_name)
        if source is None:
            # Existence is resolved against the store, the way
            # `remove_variable` resolves it, so the two mutators cannot
            # disagree about what the container holds. Gating on `readable`
            # alone made them contradict each other in the user's face:
            # `remove_variable("lat")` deleted the array while
            # `rename_variable("lat", ...)` called the same name "not found"
            # and offered a list that proved nothing of the sort.
            raise ValueError(f"Variable '{old_name}' not found. Available: {readable}")
        if _is_dimension_coordinate(working_group, old_name, source):
            # The rename is a create-then-delete, and a dimension keeps
            # pointing at its indexing variable: renaming `lat` left
            # `latitude(lat)` beside a `lat` dimension whose indexing variable
            # had been deleted, and the very next `geotransform` read died with
            # GDAL's "This object has been deleted. No action on it is
            # possible". So this is refused rather than gated out of existence
            # -- the array *is* there, and saying otherwise is what was wrong.
            raise ValueError(
                f"Variable '{old_name}' is a dimension coordinate array and "
                "cannot be renamed: the dimension of that name would keep "
                "pointing at the deleted array. Rebuild the container with the "
                "axis you want, or remove the array with remove_variable()."
            )
        if new_name in readable:
            raise ValueError(f"Variable '{new_name}' already exists.")

        # CreateMDArray is rejected on a file-backed group (netCDF data mode);
        # work on a writable MEM copy and swap it in, like remove_variable.
        dst, rg = nc._writable_root_group()
        md_arr = open_mdarray(rg, old_name)
        nc._add_md_array_to_group(rg, new_name, md_arr)
        rg.DeleteMDArray(old_name)
        nc._replace_raster(dst)
        nc._invalidate_caches()


def _is_dimension_coordinate(rg: Any, name: str, md_arr: Any) -> bool:
    """Whether ``name`` is the coordinate variable of a dimension, not a variable of its own.

    The same rule :meth:`NetCDF._group_data_array_names` uses to decide what
    the enumeration leaves out: an array is a coordinate variable when its name
    matches a dimension of its **own group** or a dimension **it is indexed
    by**. Reading both is what makes a sub-group's axis answer the same as the
    root's -- netCDF-4 dimensions are visible in every descendant group, so a
    sub-group that declares none of its own still holds coordinate arrays for
    its parents'.

    Args:
        rg: The group the array was resolved against.
        name: The array's name, relative to that group.
        md_arr: The resolved :class:`osgeo.gdal.MDArray`.

    Returns:
        bool: True when the array is a dimension coordinate.
    """
    leaf = name.rsplit("/", 1)[-1]
    own = {d.GetName().rsplit("/", 1)[-1] for d in (md_arr.GetDimensions() or [])}
    declared = {d.GetName().rsplit("/", 1)[-1] for d in (rg.GetDimensions() or [])}
    return leaf in own or leaf in declared


def _refuse_group_qualified(name: str, method: str, extra_args: str = "") -> None:
    """Refuse a sub-group name, pointing at the group view where it works.

    The variable enumeration walks sub-groups, so ``"flight_03/CO"`` is a name
    the container reports and :meth:`NetCDF.get_variable` accepts. Both
    mutators check existence against that same list, which let a qualified name
    through to GDAL, where it failed with *"Array flight_03/CO is not an array
    of this group"* -- a driver message from methods that document
    :class:`ValueError`.

    Refusing is the right answer rather than resolving the group, because these
    methods mutate exactly one group: the container's working group. GDAL will
    not delete another group's array from the root, and creating the renamed
    copy at the root would *move* the variable rather than rename it.
    ``get_group(...)`` already returns a writable view of the sub-group in
    which the bare leaf name works, so the message names that call.

    Args:
        name: The variable name the caller passed.
        method: The public method name, used in the message and the suggestion.
        extra_args: Text appended inside the suggested call's parentheses, for
            a method that takes more than the name.

    Raises:
        ValueError: When ``name`` carries a group path.

    Examples:
        - A bare name is accepted silently:
            ```python
            >>> _refuse_group_qualified("CO", "remove_variable") is None
            True

            ```
        - A qualified one names the group and the call that works:
            ```python
            >>> try:
            ...     _refuse_group_qualified("flight_03/CO", "remove_variable")
            ... except ValueError as exc:
            ...     print("flight_03" in str(exc), "get_group" in str(exc))
            True True

            ```
    """
    group, _, leaf = name.rpartition("/")
    if group:
        raise ValueError(
            f"{method}() cannot act on {name!r}: it lives in the sub-group "
            f"{group!r}, and this container mutates only its own working "
            f"group. Open the group first: "
            f"nc.get_group({group!r}).{method}({leaf!r}{extra_args})."
        )


def _names_to_copy(dataset: Any, variable_name: str | None) -> list[str]:
    """Which of the source's variables `add_variable` should bring across.

    Args:
        dataset: The source container or raster.
        variable_name: The single name the caller asked for, or `None` for all
            of them.

    Returns:
        list[str]: The names to copy. A named variable is taken at its word --
            it is resolved later, and refused there if it does not exist. With
            no name, a `NetCDF` source gives everything it holds rather than
            only its data variables, because a copy that dropped the auxiliary
            arrays would not be a copy; a plain raster has no variables to
            enumerate.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle
    # (netcdf.py imports this module at top level for wiring).
    from pyramids.netcdf.netcdf import NetCDF

    if variable_name is not None:
        names = [variable_name]
    elif isinstance(dataset, NetCDF):
        names = list(dataset._readable_variable_names())
    else:
        names = []
    return names


def _refuse_unresolved_source(dataset: Any, var_rg: Any, name: str) -> None:
    """Refuse a source variable that will not resolve, naming what is on offer.

    Skipping it was added so the internal auxiliary carry could survive a name
    that will not walk. It also turned a typo in the public
    `add_variable(dataset, "does_not_exist")` into a silent no-op that returned
    `None` and left the container untouched. The carry loop catches
    `ValueError` and folds it into one warning naming every variable it could
    not bring across, so the signal is kept on both paths.

    Args:
        dataset: The source container or raster.
        var_rg: The source's working group, for a raster that cannot enumerate.
        name: The name that did not resolve.

    Raises:
        ValueError: Always -- that is what this helper is for.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle
    # (netcdf.py imports this module at top level for wiring).
    from pyramids.netcdf.netcdf import NetCDF

    available = (
        dataset._readable_variable_names()
        if isinstance(dataset, NetCDF)
        else list(var_rg.GetMDArrayNames() or [])
    )
    raise ValueError(
        f"add_variable() could not resolve {name!r} in the source container. "
        f"Available: {available}"
    )


def _free_target_name(source_name: str, existing: Sequence[str]) -> str:
    """Pick the destination array name for a copied variable.

    The destination root group is **flat**, and netCDF-4 forbids ``/`` in a
    variable name, so a group-qualified source name (``"flight_03/CO"``) cannot
    be used verbatim: creating an array under it built a container that
    ``to_file`` later refused with *"NetCDF: Name contains illegal
    characters"*, long after the ``add_variable`` call that caused it. The leaf
    segment is the name, and collisions are resolved by suffixing rather than
    overwriting -- ``"-new"`` first, as before, then numbered, so a store whose
    sub-groups repeat a leaf name copies all of them instead of clobbering one
    with the next.

    Args:
        source_name: The source array's name, possibly group-qualified.
        existing: Names already present in the destination group.

    Returns:
        str: A name not already taken in ``existing``.

    Examples:
        - A free name is used unchanged:
            ```python
            >>> _free_target_name("CO", [])
            'CO'

            ```
        - A group-qualified name is reduced to its leaf:
            ```python
            >>> _free_target_name("flight_03/CO", [])
            'CO'

            ```
        - A taken name is suffixed rather than overwritten:
            ```python
            >>> _free_target_name("flight_03/CO", ["CO"])
            'CO-new'
            >>> _free_target_name("flight_04/CO", ["CO", "CO-new"])
            'CO-new-2'

            ```
    """
    leaf = source_name.rpartition("/")[2]
    taken = set(existing)
    candidate = leaf
    suffix = 1
    while candidate in taken:
        suffix += 1
        candidate = f"{leaf}-new" if suffix == 2 else f"{leaf}-new-{suffix - 1}"
    return candidate


def _resolve_band_metadata(
    dataset: Dataset,
    band_dim_name: str | None,
    band_dim_values: list | None,
    attrs: dict | None,
) -> tuple[str | None, list | None, dict | None, dict]:
    """Resolve `set_variable` band metadata, auto-detecting from the source dataset.

    Fills ``band_dim_name`` / ``band_dim_values`` / ``attrs`` from the source's
    tracked origin attributes (``_band_dim_name`` etc., set by ``get_variable``)
    when not given explicitly, and bundles the multi-band-dim metadata
    (``_band_dim_names`` / ``_band_dim_sizes`` / ``_band_dim_values_map``) into a
    ``band`` dict consumed by :func:`_build_variable_mdarray`.
    """
    if band_dim_name is None and hasattr(dataset, "_band_dim_name"):
        band_dim_name = dataset._band_dim_name
    if band_dim_values is None and hasattr(dataset, "_band_dim_values"):
        band_dim_values = dataset._band_dim_values
    if attrs is None and hasattr(dataset, "_variable_attrs"):
        attrs = dataset._variable_attrs
    band = {
        "names": tuple(getattr(dataset, "_band_dim_names", ()) or ()),
        "sizes": tuple(getattr(dataset, "_band_dim_sizes", ()) or ()),
        "values_map": dict(getattr(dataset, "_band_dim_values_map", {}) or {}),
    }
    return band_dim_name, band_dim_values, attrs, band


def _is_text_axis(values: np.ndarray) -> bool:
    """Whether a coordinate axis holds text rather than numbers.

    A NumPy string array (`kind` `U`/`S`) is text outright. An **object** array
    (`kind` `O`) is text only when every element is a `str` or `bytes`: that is how a
    `pandas.Index` of strings, an xarray string coordinate's `.values`, and an explicit
    `np.array([...], dtype=object)` all present, none of which a `kind` check alone
    catches, yet each is a realistic caller of `set_variable` / `from_array` (#1181).

    Args:
        values: The coordinate values.

    Returns:
        bool: `True` when the axis is text.
    """
    if values.dtype.kind in ("U", "S"):
        text = True
    elif values.dtype.kind == "O":
        text = values.size > 0 and all(
            isinstance(one, (str, bytes)) for one in values.ravel()
        )
    else:
        text = False
    return text


def _coordinate_dtype(values: np.ndarray, default: Any) -> Any:
    """The GDAL type for a band-coordinate axis, chosen from what it holds.

    An integer axis keeps its own integer type; a **text** axis — WRF's `Time` of
    `'2000-01-24_12:00:00'` stamps, a scenario name, a station id, whether NumPy string
    or object-of-strings (see `_is_text_axis`) — is stored as GDAL strings; everything
    else is `default` (the float64 coordinate type). Reducing over a variable's *other*
    dimension has to carry this one through the rebuild, and coercing a text axis to
    float64 turned every stamp into `could not convert string to float` (#1181).

    Args:
        values: The coordinate values.
        default: The type for a non-integer, non-text axis (float64).

    Returns:
        The GDAL :class:`osgeo.gdal.ExtendedDataType` to create the axis with.

    Raises:
        ValueError: The axis mixes text with non-text values (e.g. a string
            `pandas.Index` carrying a `None`/`NaN` gap). Such an axis has no single
            storage type; without this it fell through to float64 and raised the
            misleading `could not convert string to float` (#1181).
    """
    if np.issubdtype(values.dtype, np.integer):
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(values.dtype))
    elif _is_text_axis(values):
        dtype = gdal.ExtendedDataType.CreateString()
    else:
        _reject_partial_text_axis(values)
        dtype = default
    return dtype


def _reject_partial_text_axis(values: np.ndarray) -> None:
    """Refuse an object axis that carries text alongside non-text values.

    `_is_text_axis` is all-or-nothing, so a text axis with a gap — a string
    `pandas.Index` with a `None` or a float `nan` — reaches the float64 default and
    raised `could not convert string to float`, the exact #1181 error, from deep in the
    GDAL write. Storing the gap would mean inventing a value for it (an empty string, a
    sentinel), which is the caller's decision, so this refuses with a message that names
    the real problem instead.

    Args:
        values: The coordinate values, already known not to be integer or all-text.

    Raises:
        ValueError: An object array holds at least one `str`/`bytes` element but is not
            all text.
    """
    if values.dtype.kind == "O" and any(
        isinstance(one, (str, bytes)) for one in values.ravel()
    ):
        raise ValueError(
            "a coordinate axis that mixes text with non-text values (a gap, a number) "
            f"has no single storage type: {list(values)!r}. Give every coordinate a "
            "value of one kind."
        )


def _carry_band_dim_attrs(
    rg: Any,
    dim_name: str,
    created: Any,
    labelled: bool,
    preexisting: bool,
    dim_attrs: dict[str, dict[str, str]] | None,
) -> None:
    """Write a band dimension's carried CF attributes onto its coordinate array.

    This is the `set_variable` counterpart of `_create_extra_dimensions`' carry, for the
    variables a rebuild adds after the first (#1179). It writes only for an axis that is
    both **newly created** — a dimension the first variable already made carries the
    attributes it was created with, and re-writing would duplicate them — and **labelled**,
    since the fabricated `0..n-1` of an unlabelled axis are positions, not measurements
    (the rule `_coordinate_attrs` enforces on the first-variable path).

    Args:
        rg: The root group.
        dim_name: The dimension's name.
        created: The dimension just returned by `_get_or_create_dimension`.
        labelled: Whether the caller supplied this axis' coordinate values.
        preexisting: Whether a dimension of this name was already in the store.
        dim_attrs: CF attributes keyed by dimension name, or `None`.
    """
    if preexisting or not labelled:
        return
    carried = (dim_attrs or {}).get(dim_name)
    if carried:
        # `SetIndexingVariable` is skipped on the netCDF driver, so the coordinate array is
        # reached by name there. A dimension just created on this MEM group always has one,
        # so the `is not None` guard covers a driver that refuses rather than any supported
        # path — it is not covered, and cannot be (as in `_create_extra_dimensions`).
        indexing = created.GetIndexingVariable() or rg.OpenMDArray(dim_name)
        if indexing is not None:
            write_attributes_to_md_array(indexing, carried)


def _create_multi_band_dims(
    nc: NetCDF,
    rg: Any,
    names: tuple[str, ...],
    sizes: tuple[int, ...],
    values_map: dict,
    coord_dtype: Any,
    dim_attrs: dict[str, dict[str, str]] | None = None,
) -> list:
    """Create one GDAL dimension per tracked non-spatial axis (the 4-D+ rebuild path).

    Each axis takes its coordinate values from `values_map`; the first axis is tagged
    `DIM_TYPE_TEMPORAL`. An axis *tracked* as coordinate-less — its `values_map` entry is
    present and `None`, which a T20 operator disagreement produces — is created with no
    indexing variable via `NetCDF._coordinateless_dimension`, so it reads back as `None`
    (#1192). An axis simply absent from `values_map` keeps the positional `range(size)`
    default, the same decision the single-band path makes, matching `from_array`. A newly
    created, labelled axis also carries its CF `(units, calendar)` from `dim_attrs` onto its
    coordinate array, so a variable a join adds after the first keeps them through `to_file`
    (#1179).
    """
    band_dims = []
    known = {dimension.GetName() for dimension in rg.GetDimensions() or []}
    for i, dim_name in enumerate(names):
        dim_type = gdal.DIM_TYPE_TEMPORAL if i == 0 else None
        # Decide from the same evidence the single-band path uses: an axis *tracked* as
        # coordinate-less is the key present with a `None` value (a T20 disagreement); a key
        # simply absent is an omitted axis that keeps the positional `range(size)` default,
        # matching `from_array`. For a well-formed `values_map` (every name a key) this is
        # exactly `values is None`; the split only guards a future invariant slip (#1201 N3).
        if dim_name in values_map and values_map[dim_name] is None:
            # No coordinates for this axis: write it with no indexing variable so it reads
            # back `None`, not a fabricated `range(size)` (#1192).
            band_dims.append(
                nc._coordinateless_dimension(rg, dim_name, int(sizes[i]), dim_type)
            )
            continue
        values = values_map.get(dim_name)
        labelled = values is not None
        if values is None:
            values = list(range(int(sizes[i])))
        # `np.asarray`, not `dtype=np.float64`: a text axis stays text so it can be stored
        # as strings, where the float cast raised on WRF's `Time` stamps (#1181).
        values = np.asarray(values)
        created = nc._get_or_create_dimension(
            rg, dim_name, values, _coordinate_dtype(values, coord_dtype), dim_type
        )
        _carry_band_dim_attrs(
            rg, dim_name, created, labelled, dim_name in known, dim_attrs
        )
        band_dims.append(created)
    return band_dims


def _build_variable_mdarray(
    nc: NetCDF,
    rg: Any,
    variable_name: str,
    arr: np.ndarray,
    dim_y: Any,
    dim_x: Any,
    data_dtype: Any,
    coord_dtype: Any,
    band_dim_name: str | None,
    band_dim_values: list | None,
    band: dict,
    dim_attrs: dict[str, dict[str, str]] | None = None,
) -> Any:
    """Create the variable MDArray with the right band dimensions and write `arr`.

    Three layouts: a multi-band-dim 4-D+ rebuild (reshape the flattened bands back into
    storage order, one GDAL dim per non-spatial axis via :func:`_create_multi_band_dims`);
    the legacy single-band-dim 3-D path; and a plain 2-D `(y, x)` variable. An axis the
    source *tracks* as coordinate-less — its `values_map` entry is present and `None`, which
    a T20 operator disagreement produces — is created with no indexing variable via
    `NetCDF._coordinateless_dimension`, so it reads back as `None` (#1192). A caller who
    merely omits `band_dim_values` on a plain raster is not that case: the axis keeps the
    documented positional `range(size)` default, matching `from_array`. A newly created,
    labelled band axis carries its CF `(units, calendar)` from `dim_attrs` onto its
    coordinate array so a join's later variable keeps them through `to_file` (#1179).
    Returns the written MDArray.
    """
    names, sizes, values_map = band["names"], band["sizes"], band["values_map"]
    if len(names) > 1 and arr.ndim == 3 and sizes:
        arr = unflatten_band_axes(arr, names, sizes)
        band_dims = _create_multi_band_dims(
            nc, rg, names, sizes, values_map, coord_dtype, dim_attrs
        )
        md_arr = rg.CreateMDArray(variable_name, [*band_dims, dim_y, dim_x], data_dtype)
    elif arr.ndim == 3:
        if band_dim_name is None:
            band_dim_name = "bands"
        # Only a source that *tracks* this axis as coordinate-less writes a
        # coordinate-less dimension — a T20 operator disagreement leaves
        # `_band_dim_values_map[name] is None` (the key present, the value `None`). A
        # caller who merely omits `band_dim_values` on a plain raster is not that case:
        # it still gets the documented positional `range(size)` default, matching
        # `from_array` (#1192, #1201).
        coordinate_less = (
            band_dim_name in values_map and values_map[band_dim_name] is None
        )
        if band_dim_values is None and coordinate_less:
            # Write the dimension with no indexing variable so it reads back `None`,
            # rather than adopting the store's own stamps (#1192).
            dim_band = nc._coordinateless_dimension(
                rg, band_dim_name, arr.shape[0], gdal.DIM_TYPE_TEMPORAL
            )
        else:
            labelled = band_dim_values is not None
            if band_dim_values is None:
                band_dim_values = list(range(arr.shape[0]))
            preexisting = band_dim_name in {
                dimension.GetName() for dimension in rg.GetDimensions() or []
            }
            # `np.asarray`, not `dtype=np.float64`: a text axis stays text (#1181).
            band_values = np.asarray(band_dim_values)
            dim_band = nc._get_or_create_dimension(
                rg,
                band_dim_name,
                band_values,
                _coordinate_dtype(band_values, coord_dtype),
                gdal.DIM_TYPE_TEMPORAL,
            )
            _carry_band_dim_attrs(
                rg, band_dim_name, dim_band, labelled, preexisting, dim_attrs
            )
        md_arr = rg.CreateMDArray(variable_name, [dim_band, dim_y, dim_x], data_dtype)
    else:
        md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], data_dtype)
    md_arr.Write(arr)
    return md_arr


def from_array(
    arr: np.ndarray,
    *,
    geo_ref: GeoReference,
    path: str | Path | None = None,
    variable_name: str | None = None,
    no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
    dims: ExtraDimensions | None = None,
    encoding: Encoding | None = None,
    attrs: CFAttributes | None = None,
    spatial_names: tuple[str, str] | None = None,
) -> Container:
    """Create a NetCDF dataset from a NumPy array and geotransform.

    For 3-D arrays the first axis is treated as a non-spatial
    dimension (time, level, depth, etc.) whose name and coordinate
    values are controlled by `extra_dim_name` and
    `extra_dim_values`.

    For 4-D+ arrays — e.g. `(time, level, lat, lon)` — pass
    `extra_dims=[("time", time_values), ("pressure_level", level_values)]`
    in storage order. Every non-spatial dimension is then
    materialised on the resulting NetCDF, preserving the full
    layout. `extra_dims` and the legacy single-dim params
    (`extra_dim_name` / `extra_dim_values`) are mutually exclusive.

    `path` decides only *where* the store goes, not what format it is: `None`
    creates it in memory (MEM driver), and any path is written by the netCDF
    driver, because this builds a multidimensional store no other driver here
    can carry. The extension is therefore checked rather than obeyed — a path
    naming a different format (`"lies.tif"`) raises
    :class:`~pyramids.base._errors.FileFormatNotSupportedError` instead of
    silently writing netCDF bytes under a GeoTIFF name. Use
    :meth:`pyramids.dataset.Dataset.from_array` to write those formats.

    `arr` may be a `dask.array.Array` instead of a NumPy array. It is
    then written to disk one block at a time (memory-bounded streaming),
    so an array too large to hold in RAM can still be written — peak
    source memory is a single dask block rather than the whole array. The
    output is byte-identical to passing the materialised NumPy array.
    When `chunk_sizes` is not given, the on-disk chunking defaults to the
    dask block shape. Writing to memory (`path=None`) still materialises
    one block at a time but the in-memory result holds the full array.

    The write options are organised into grouped, validated dataclasses
    (:class:`~pyramids.base.georeference.GeoReference`,
    :class:`~pyramids.netcdf.array_options.ExtraDimensions`,
    :class:`~pyramids.netcdf.array_options.Encoding`,
    :class:`~pyramids.netcdf.array_options.CFAttributes`) — see each class for
    its fields. `GeoReference` is shared vocabulary and lives in
    :mod:`pyramids.base.georeference`; the other three are netCDF-specific. All
    four are importable from the subpackage, e.g.
    ``from pyramids.netcdf import GeoReference``.

    Args:
        arr: 2-D `(rows, cols)`, 3-D `(extra_dim, rows, cols)`, or
            4-D+ `(d_0, ..., d_{n-1}, rows, cols)` array. Either a NumPy
            array (written eagerly) or a `dask.array.Array` (streamed
            block-by-block, see above).
        geo_ref: How the array maps to space — a
            :class:`~pyramids.base.georeference.GeoReference` holding either
            `geo` / `epsg` or `top_left_corner` + `cell_size`. Required and
            keyword-only, with no default: it must resolve to a geotransform,
            so an empty `GeoReference()` raises rather than placing the array
            somewhere arbitrary.
        path: Output file path, which must name netCDF (`.nc`, or its `.nc4`
            alias). If `None` (default), the store is created in memory.
        variable_name: Name of the data variable in the NetCDF
            file. Defaults to `"data"`.
        no_data_value: Sentinel value for cells outside the
            domain. Defaults to DEFAULT_NO_DATA_VALUE.
        dims: The non-spatial dimensions of a 3-D+ array, as an
            :class:`~pyramids.netcdf.array_options.ExtraDimensions`. Ignored
            for 2-D arrays. Defaults to an empty `ExtraDimensions()` (dim name
            `"time"`, integer-index coordinates).
        encoding: On-disk write options (chunking, compression) as an
            :class:`~pyramids.netcdf.array_options.Encoding`. Only effective
            when `path` is given. Defaults to an empty `Encoding()`.
        attrs: CF global attributes (`title`, `institution`, `source`,
            `history`) as a
            :class:`~pyramids.netcdf.array_options.CFAttributes`. Defaults to
            an empty `CFAttributes()`.
        spatial_names: `(row, column)` names for the two spatial dimensions. `None`
            (default) names them `y` / `x`. A rebuild passes the source store's own
            names, so a `reduce` / `coarsen` result keeps `latitude` / `longitude`
            rather than renaming the grid (#1180). Must be two distinct non-empty
            strings; anything else is refused here rather than surfacing later as an
            unpacking error or a GDAL duplicate-dimension message.

    Returns:
        Container: The newly created store. Always a `Container`, never a bare
            `NetCDF`, regardless of the subtype the facade was invoked on —
            the annotation says so rather than leaving the caller's declared
            `-> Container` resting on an engine that claims otherwise.

    Raises:
        ValueError: `geo_ref` resolves to no geotransform — it carries neither
            a `geo` nor a complete `top_left_corner` + `cell_size` pair — or
            the requested extra dimensions do not match `arr`'s non-spatial
            axes; `spatial_names` is not two distinct non-empty strings; or
            `dims.attrs` is keyed by a dimension this array does not have (a
            misspelled key wrote nothing at all before, which looked exactly like
            the bug the parameter exists to fix).
        DriverNotExistError: `path` has no extension, or one the driver catalog
            does not know.
        FileFormatNotSupportedError: `path`'s extension names a driver other
            than netCDF. The store written here is multidimensional, so the
            format cannot be honoured; build the raster with
            :meth:`pyramids.dataset.Dataset.from_array` instead.

    See Also:
        - :meth:`pyramids.netcdf.NetCDF.from_array`: The classmethod facade
          this function backs.
        - :class:`~pyramids.base.georeference.GeoReference`: The georeferencing
          value object `geo_ref` takes.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle
    # (netcdf.py imports this module at top level for wiring). from_array
    # always returns a Container regardless of which NetCDF subtype the façade
    # was invoked on, sidestepping the deprecated base-NetCDF construction path.
    from pyramids.netcdf.netcdf import Container

    dims = dims or ExtraDimensions()
    encoding = encoding or Encoding()
    attrs = attrs or CFAttributes()

    # `GeoReference` owns the geo/(corner + cell_size) reconciliation and raises when neither
    # can produce a geotransform.
    geo = geo_ref.resolve_geotransform()

    rows = int(arr.shape[-2]) if arr.ndim >= 2 else 0
    cols = int(arr.shape[-1]) if arr.ndim >= 2 else 0

    # Reconcile the legacy single-dim params with the new
    # `dims` list-of-pairs API. Result is a normalised list
    # of (name, values) pairs whose length equals
    # `max(arr.ndim - 2, 0)`.
    resolved_extra_dims = _resolve_extra_dims(
        arr=arr,
        extra_dim_name=dims.name,
        extra_dim_values=dims.values,
        extra_dims=dims.dims,
    )

    if arr.ndim == 3:
        ClassicDimensionInfo(
            name=resolved_extra_dims[0][0],
            size=arr.shape[0],
            values=resolved_extra_dims[0][1],
        )

    if variable_name is None:
        variable_name = "data"

    _require_spatial_names(spatial_names)
    _require_known_dimensions(dims.attrs, resolved_extra_dims)
    cf_attrs = attrs.as_dict()

    dst_ds = _create_netcdf_from_array(
        arr,
        variable_name,
        cols,
        rows,
        resolved_extra_dims,
        geo,
        geo_ref.epsg,
        no_data_value,
        path=path,
        encoding=encoding,
        cf_attrs=cf_attrs,
        spatial_names=spatial_names,
        dim_attrs=_coordinate_attrs(dims),
    )
    result = Container(dst_ds)

    return result


def from_dataframe(
    df: pd.DataFrame,
    *,
    crs: str | int | None = None,
    x: str | None = None,
    y: str | None = None,
    variables: str | Sequence[str] | None = None,
    no_data_value: Any = DEFAULT_NO_DATA_VALUE,
    path: str | Path | None = None,
) -> Container:
    """Rebuild a NetCDF cube from a `MultiIndex` DataFrame — the inverse of `to_dataframe`.

    The frame must be indexed by its dimensions, the two innermost index levels being the
    `(y, x)` grid axes and any outer levels the band dimensions; each non-index column
    becomes a data variable. That is exactly the shape `to_dataframe` returns, so
    `from_dataframe(nc.to_dataframe(), crs=nc.epsg)` reproduces `nc` — to floating-point
    tolerance, since the geotransform is recovered by differencing the stored cell centres.
    The one cube it cannot round-trip is a single-row or single-column raster: one centre on
    an axis carries no spacing to recover, so that axis is refused (see `Raises`).

    A DataFrame carries no georeferencing, so this recovers it: the geotransform is
    **inferred** from the `x` / `y` cell-centre coordinates assuming a regular grid (an
    irregular axis is refused — it has no affine transform), and the CRS comes from `crs`
    (a DataFrame has none, so it is left unset when `crs` is `None`). The result is always
    north-up (`y` descending, `x` ascending), whatever order the frame's rows are in. Band
    dimensions are not sorted: their coordinates keep the order they first appear in the
    frame, which is the order `to_dataframe` emitted them.

    Args:
        df: A DataFrame on a `pandas.MultiIndex` of at least two named levels. The innermost
            two are the row (`y`) and column (`x`) axes unless `x` / `y` name them; any
            further levels are band dimensions, outermost first. A tidy frame on a plain
            index is refused — it describes scattered rows, not a grid.
        crs: The CRS for the result, as an EPSG code or a CRS string. `None` (default)
            leaves the CRS unset, because a DataFrame does not carry one — a full
            `to_dataframe` → `from_dataframe` round trip therefore needs `crs=nc.epsg` to
            recover it.
        x: The index level holding the column (x) coordinates. Defaults to the innermost
            level.
        y: The index level holding the row (y) coordinates. Defaults to the second-innermost
            level.
        variables: Which columns become data variables, as a name or a sequence of names.
            `None` (default) takes every column.
        no_data_value: Sentinel for the gaps. `NaN` cells (and cells absent from the frame)
            are stored as this value. Defaults to `DEFAULT_NO_DATA_VALUE`.
        path: Destination. `None` (default) builds the store in memory; a `.nc` path writes
            it, exactly as `from_array`.

    Returns:
        Container: The rebuilt store, one variable per chosen column, on the inferred grid.

    Raises:
        ValueError: `df` is not indexed by a `MultiIndex` of at least two named levels; a
            named `x` / `y` level is missing or the two coincide; there are no value columns
            or a requested one is absent; the index has duplicate rows (an ambiguous cell);
            or the `x` / `y` axis is irregular or has fewer than two coordinates, so no
            geotransform can be inferred.

    Examples:
        - Round-trip a two-step cube through pandas and back:

          ```python
          >>> import numpy as np
          >>> from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
          >>> nc = NetCDF.from_array(
          ...     np.arange(8.0).reshape(2, 2, 2),
          ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
          ...     variable_name="t",
          ...     dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
          ... )
          >>> back = NetCDF.from_dataframe(nc.to_dataframe(), crs=nc.epsg)
          >>> back.get_variable("t")._band_dim_values_map["time"]
          [0.0, 6.0]
          >>> back.get_variable("t").read_array().tolist()
          [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]

          ```
    """
    band_names, row_name, col_name = _dataframe_axes(df, x, y)
    columns = _dataframe_value_columns(df, variables)
    if df.index.duplicated().any():
        raise ValueError(
            "from_dataframe() found duplicate index rows, so a cell has more than one "
            "value. Resolve them first, e.g. "
            "df.groupby(level=list(df.index.names)).mean()."
        )

    band_coords = [pd.unique(df.index.get_level_values(nm)) for nm in band_names]
    y_coords = np.unique(
        np.asarray(df.index.get_level_values(row_name), dtype="float64")
    )[::-1]
    x_coords = np.unique(
        np.asarray(df.index.get_level_values(col_name), dtype="float64")
    )
    geo = _geotransform_from_centres(x_coords, y_coords)

    ordered = df.reorder_levels([*band_names, row_name, col_name])
    full = pd.MultiIndex.from_product(
        [*band_coords, list(y_coords), list(x_coords)],
        names=[*band_names, row_name, col_name],
    )
    ordered = ordered.reindex(full)
    shape = (*[len(c) for c in band_coords], len(y_coords), len(x_coords))

    geo_ref = GeoReference(geo=geo, epsg=crs)
    dims = (
        ExtraDimensions(dims=[(nm, list(c)) for nm, c in zip(band_names, band_coords)])
        if band_names
        else None
    )
    built = []
    for col in columns:
        arr = _dataframe_column_array(ordered, col, shape)
        arr = np.where(np.isnan(arr), no_data_value, arr)
        built.append(
            from_array(
                arr,
                geo_ref=geo_ref,
                variable_name=str(col),
                no_data_value=no_data_value,
                dims=dims,
            )
        )
    result = built[0] if len(built) == 1 else built[0].merge(built[1:])
    if path is not None:
        result.to_file(path)
        result = result.read_file(path)
    return cast("Container", result)


def _dataframe_axes(
    df: pd.DataFrame, x: str | None, y: str | None
) -> tuple[list[str], str, str]:
    """Resolve the band-dimension names and the row / column axis names of a frame.

    Args:
        df: The DataFrame to read the index of.
        x: The column-axis level name, or `None` for the innermost level.
        y: The row-axis level name, or `None` for the second-innermost level.

    Returns:
        tuple[list[str], str, str]: `(band_dim_names, row_name, col_name)`, the band names in
        index order (outermost first).

    Raises:
        ValueError: The index is not a `MultiIndex` of at least two named levels, a named
            `x` / `y` level is missing, or the two coincide.
    """
    index = df.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        raise ValueError(
            "from_dataframe() needs a DataFrame indexed by its dimensions — a MultiIndex "
            "of at least two named levels, the innermost two being the (y, x) grid axes. A "
            "tidy frame on a plain index is scattered rows, not a raster; name the axes "
            "first, e.g. df.set_index([...])."
        )
    names = list(index.names)
    if any(nm is None for nm in names):
        raise ValueError(
            f"from_dataframe() needs every index level named; got {names}. Name them via "
            "df.rename_axis([...]) or set_index."
        )
    row_name = y if y is not None else names[-2]
    col_name = x if x is not None else names[-1]
    for role, nm in (("y", row_name), ("x", col_name)):
        if nm not in names:
            raise ValueError(
                f"from_dataframe() was told the {role} axis is {nm!r}, which is not an "
                f"index level; the levels are {names}."
            )
    if row_name == col_name:
        raise ValueError(
            f"from_dataframe() got {row_name!r} for both the y and x axes; they must be "
            "different index levels."
        )
    band_names = [nm for nm in names if nm not in (row_name, col_name)]
    return band_names, row_name, col_name


def _dataframe_value_columns(
    df: pd.DataFrame, variables: str | Sequence[Any] | None
) -> list:
    """The columns that become data variables, in order.

    The **original** column labels are returned, not stringified ones, because the caller
    indexes the frame with them (`ordered[col]`); only the NetCDF variable name is
    stringified, at the point it is written. Returning `str(...)`-normalised labels made
    `ordered[col]` raise `KeyError` on any non-string column (#1203 L1).

    Args:
        df: The DataFrame whose columns are the candidate variables.
        variables: A label, a sequence of labels, or `None` for every column.

    Returns:
        list: The chosen column labels, in the given order, never empty.

    Raises:
        ValueError: The frame has no columns, a requested label is not a column, a label was
            given more than once, or an empty selection was given.
    """
    available = list(df.columns)
    if not available:
        raise ValueError(
            "from_dataframe() needs at least one value column to become a data variable; "
            "the frame has none."
        )
    if variables is None:
        chosen = available
    else:
        # A list/tuple is a set of labels; anything else — a str, or a scalar label such as
        # an int column name — is a single label (`list(7)` would raise).
        names = list(variables) if isinstance(variables, (list, tuple)) else [variables]
        if not names:
            raise ValueError(
                "from_dataframe() was given an empty selection; pass `variables=None` for "
                f"every column, or one of {available}."
            )
        unknown = [nm for nm in names if nm not in available]
        if unknown:
            raise ValueError(
                f"from_dataframe() cannot take {unknown!r} as variables: the frame's "
                f"columns are {available}."
            )
        repeated = [nm for nm in dict.fromkeys(names) if names.count(nm) > 1]
        if repeated:
            raise ValueError(
                f"from_dataframe() was asked for {repeated!r} more than once; a label can "
                "only become one variable."
            )
        chosen = names
    return chosen


def _dataframe_column_array(
    ordered: pd.DataFrame, col: Any, shape: tuple
) -> np.ndarray:
    """One value column as a float64 array of the cube's shape, refusing the bad cases.

    Args:
        ordered: The frame reindexed against the full dimension product.
        col: The column label to read.
        shape: The target `(*band_sizes, rows, cols)` shape.

    Returns:
        np.ndarray: The column's cells, `float64`, shaped `shape`.

    Raises:
        ValueError: The label matches more than one column (it cannot become one variable),
            or the column is not numeric — each named, rather than surfacing as a raw numpy
            reshape / conversion error (#1203 L2).
    """
    series = ordered[col]
    if isinstance(series, pd.DataFrame):
        raise ValueError(
            f"from_dataframe() found more than one column labelled {col!r}, so it cannot "
            "become one variable. Give each value column a unique label."
        )
    try:
        values = series.to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"from_dataframe() cannot read column {col!r} as numbers: {exc}. A value column "
            "must be numeric."
        ) from exc
    return values.reshape(shape)


def _geotransform_from_centres(
    x_coords: np.ndarray, y_coords: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Infer a north-up geotransform from ascending x and descending y cell centres.

    The two axes are assumed regular; `_regular_step` refuses an irregular one. The centres
    are half a cell inside the edges, so the origin steps back half a cell in each axis:
    with `dy < 0`, `y_max - dy / 2` is half a cell **above** the topmost centre.

    Args:
        x_coords: The unique column coordinates, ascending, at least two of them.
        y_coords: The unique row coordinates, descending, at least two of them.

    Returns:
        tuple: The affine geotransform `(x_min, dx, 0.0, y_max, 0.0, dy)`, `dy` negative.

    Raises:
        ValueError: Either axis is irregular or has fewer than two coordinates.
    """
    dx = _regular_step(x_coords, "x")
    dy = _regular_step(y_coords, "y")
    x_min = float(x_coords[0]) - dx / 2.0
    y_max = float(y_coords[0]) - dy / 2.0
    return (x_min, dx, 0.0, y_max, 0.0, dy)


def _regular_step(coords: np.ndarray, axis: str) -> float:
    """The constant spacing of a coordinate axis, refusing an irregular or too-short one.

    Args:
        coords: The axis' unique coordinates, already sorted (ascending x, descending y).
        axis: `"x"` or `"y"`, for the message.

    Returns:
        float: The step between consecutive coordinates — positive for x, negative for y.

    Raises:
        ValueError: Fewer than two coordinates (no spacing to infer), or the spacing varies
            (an irregular grid has no affine transform).
    """
    if coords.size < 2:
        raise ValueError(
            f"from_dataframe() cannot infer the {axis} cell size from a single {axis} "
            f"coordinate; give an axis with at least two cells, or resample to a grid first."
        )
    diffs = np.diff(coords)
    step = float(diffs[0])
    if step == 0.0 or not np.allclose(diffs, step, rtol=1e-6, atol=0.0):
        raise ValueError(
            f"from_dataframe() needs a regular {axis} axis to build a geotransform, but its "
            f"spacing varies. An irregular grid has no affine transform; resample to a "
            f"regular grid first."
        )
    return step


def _require_spatial_names(spatial_names: tuple[str, str] | None) -> None:
    """Refuse a `spatial_names` the dimension creation would only fail on later.

    Unchecked, the four ways to get it wrong surface as an unpacking error, a SWIG
    argument-type message, or `RuntimeError: A dimension with same name already exists` —
    none of which names the parameter the caller passed.

    Args:
        spatial_names: The `(row, column)` names, or `None` for the default.

    Raises:
        ValueError: `spatial_names` is not two distinct non-empty strings.
    """
    if spatial_names is not None:
        names = tuple(spatial_names)
        if len(names) != 2 or not all(isinstance(one, str) and one for one in names):
            raise ValueError(
                "spatial_names must be two non-empty strings naming the row and column "
                f"axes, got {spatial_names!r}."
            )
        if names[0] == names[1]:
            raise ValueError(
                f"spatial_names must name two different axes, got {names[0]!r} for both."
            )


def _require_known_dimensions(
    attrs: dict[str, dict[str, str]] | None, resolved: list[tuple[str, list]]
) -> None:
    """Refuse CF attributes addressed to a dimension the array does not have.

    A misspelled key, or one left over from a dimension the caller dropped, used to do
    nothing at all — the attributes were quietly not written and the axis came back bare,
    which is the same symptom as #1179 with none of its cause.

    Args:
        attrs: The `ExtraDimensions.attrs` mapping, or `None`.
        resolved: The `(name, values)` pairs of every non-spatial dimension.

    Raises:
        ValueError: A key of `attrs` names no non-spatial dimension.
    """
    known = {name for name, _ in resolved}
    unknown = sorted(set(attrs or {}) - known)
    if unknown:
        raise ValueError(
            f"attrs names {', '.join(repr(one) for one in unknown)}, which "
            f"{'are' if len(unknown) > 1 else 'is'} not among the dimensions of this "
            f"array ({', '.join(sorted(known)) or 'none'})."
        )


def _coordinate_attrs(dims: ExtraDimensions) -> dict[str, dict[str, str]] | None:
    """The CF attributes of `dims`, less any axis that was given no coordinate values.

    An axis handed in as `None` is filled with `[0, 1, ..., size - 1]`, which are
    positions, not measurements. Writing `units = "hours since 1900-01-01"` over them
    does not describe the axis — it invents timestamps for it, and every later read
    decodes step 0 as 1900-01-01 rather than as the unlabelled position it is. Only an
    axis whose values the caller supplied can be said to be in those units.

    Args:
        dims: The dimensions as the caller described them, before `_resolve_extra_dims`
            fills the missing coordinates in.

    Returns:
        dict | None: The attributes to write, or `None` when none survive.
    """
    if not dims.attrs:
        result = None
    else:
        if dims.dims is not None:
            labelled = {name for name, values in dims.dims if values is not None}
        else:
            labelled = {dims.name} if dims.values is not None else set()
        result = {
            name: attrs for name, attrs in dims.attrs.items() if name in labelled
        } or None
    return result


def _resolve_extra_dims(
    arr: np.ndarray,
    extra_dim_name: str,
    extra_dim_values: list | None,
    extra_dims: list[tuple[str, list | None]] | None,
) -> list[tuple[str, list]]:
    """Normalise the legacy + new extra-dim API into a single list.

    Returns an ordered list of `(dim_name, values)` pairs whose
    length equals `max(arr.ndim - 2, 0)`. Each `values` entry is a
    concrete Python list (never `None` — defaults are filled with
    integer indices `[0, 1, ..., size - 1]`).

    Args:
        arr: The data array; only its `ndim` and `shape` are read.
        extra_dim_name: Legacy single-dim name (caller-default
            `"time"`).
        extra_dim_values: Legacy single-dim values, or `None`.
        extra_dims: New multi-dim list of `(name, values)` pairs,
            or `None` for the legacy path.

    Returns:
        list[tuple[str, list]]: Normalised dim specs.

    Raises:
        ValueError: If `extra_dims` is supplied alongside
            `extra_dim_values`; if `extra_dims` length doesn't
            match `arr.ndim - 2`; or if any per-dim `values`
            length doesn't match the corresponding `arr.shape[i]`.
    """
    expected = max(arr.ndim - 2, 0)
    if extra_dims is not None:
        return _resolve_explicit_extra_dims(arr, extra_dims, extra_dim_values, expected)
    if expected == 0:
        return []
    if expected == 1:
        values = (
            list(extra_dim_values)
            if extra_dim_values is not None
            else list(range(int(arr.shape[0])))
        )
        return [(extra_dim_name, values)]
    # 4-D+ array with no `extra_dims` and no legacy values: fall
    # back to anonymous dim names and integer indices so the array
    # can still be written.
    return [(f"dim_{i}", list(range(int(arr.shape[i])))) for i in range(expected)]


def _resolve_explicit_extra_dims(
    arr: np.ndarray,
    extra_dims: list[tuple[str, list | None]],
    extra_dim_values: list | None,
    expected: int,
) -> list[tuple[str, list]]:
    """Normalise the explicit ``extra_dims`` list-of-pairs path.

    Validates it is not combined with the legacy ``extra_dim_values``, that its
    length matches the non-spatial axis count, and that each per-dim ``values``
    matches the corresponding ``arr.shape[i]`` (filling ``None`` with integer
    indices). Helper of :func:`_resolve_extra_dims`.

    Raises:
        ValueError: On mutual-exclusion, length-mismatch, or per-dim
            value-length-mismatch (see :func:`_resolve_extra_dims`).
    """
    if extra_dim_values is not None:
        raise ValueError(
            "extra_dims and extra_dim_values are mutually "
            "exclusive. Use one or the other."
        )
    if len(extra_dims) != expected:
        raise ValueError(
            f"extra_dims must have {expected} entries for a "
            f"{arr.ndim}-D array, got {len(extra_dims)}."
        )
    resolved: list[tuple[str, list]] = []
    for i, (name, values) in enumerate(extra_dims):
        if values is None:
            values = list(range(int(arr.shape[i])))
        elif len(values) != int(arr.shape[i]):
            raise ValueError(
                f"extra_dims[{i}] values length {len(values)} "
                f"does not match arr.shape[{i}]={arr.shape[i]}."
            )
        else:
            values = list(values)
        resolved.append((name, values))
    return resolved


def _build_create_options(
    chunk_sizes: tuple | list | None,
    compression: str | None,
    compression_level: int | None,
) -> list[str]:
    """Assemble GDAL MDArray creation options from the chunking/compression knobs."""
    options: list[str] = []
    if chunk_sizes is not None:
        options.append(f"BLOCKSIZE={','.join(str(s) for s in chunk_sizes)}")
    if compression is not None:
        options.append(f"COMPRESS={compression}")
    if compression_level is not None:
        options.append(f"ZLEVEL={compression_level}")
    return options


def _create_extra_dimensions(
    rg: Any,
    extra_dims: list[tuple[str, list]],
    dtype: Any,
    use_set_indexing: bool,
    dim_attrs: dict[str, dict[str, str]] | None = None,
) -> list:
    """Create one GDAL dimension per non-spatial axis, in storage order.

    The first non-spatial dim is tagged ``DIM_TYPE_TEMPORAL`` (matching the
    legacy 3-D path); the rest are left untagged so the netCDF driver does not
    second-guess their semantics.

    The coordinate type is chosen by ``_coordinate_dtype``: an **integer** axis
    keeps its own integer dtype, a **text** axis (``_is_text_axis`` — WRF's
    ``Time`` stamps, a scenario name) is stored as GDAL strings, and everything
    else is written in ``dtype`` (float64). The float64 rule was written for the
    *spatial* axes, where sharing the data array's integer type truncated a
    2.5-degree grid to whole degrees -- a real defect it fixes. Applied to a
    non-spatial axis it costs the opposite: an `int64` nanosecond-epoch `time`
    handed in as `1700000000123456789` came back `1700000000123456768`, because
    float64 carries 53 bits of mantissa. Nothing is lost by keeping the integer
    type, and the streamed arm (`_add_aux_var_spec`) already copies the source
    dtype, so the two arms of a fan-out now agree about `time` as well.

    Visible consequence relative to `origin/main`: a band dim left to default is
    `list(range(size))`, so it is integer too, and anything reading those values
    back gets `0` where it used to get `0.0` -- facet panel labels most
    noticeably (`NetCDFPlot._build_facet_stack`).

    Args:
        rg: The group to create the dimensions in.
        extra_dims: ``(name, values)`` per non-spatial axis, in storage order.
        dim_attrs: CF attributes to write onto an axis' coordinate array, keyed by
            dimension name — `{"time": {"units": …, "calendar": …}}`. A computed result
            carries its `(units, calendar)` on the Python object; without writing them
            here the store never had them, so `to_file` had nothing to copy and the
            calendar died at the write (#1179).
        dtype: The coordinate :class:`osgeo.gdal.ExtendedDataType` to use for
            any axis that is not integer-valued.
        use_set_indexing: Whether ``SetIndexingVariable`` is supported by the
            driver being written to.

    Returns:
        list: The created :class:`osgeo.gdal.Dimension` objects, in order.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle.
    from pyramids.netcdf.netcdf import NetCDF

    gdal_extra_dims = []
    for i, (dim_name, dim_values) in enumerate(extra_dims):
        dim_type = gdal.DIM_TYPE_TEMPORAL if i == 0 else None
        values = np.asarray(dim_values)
        dim_dtype = _coordinate_dtype(values, dtype)
        created = NetCDF._create_dimension(
            rg, dim_name, dim_dtype, values, dim_type, use_set_indexing
        )
        carried = (dim_attrs or {}).get(dim_name)
        if carried:
            # `SetIndexingVariable` is skipped on the netCDF driver, so the coordinate
            # array is reached by name there. `_create_dimension` has just created it, so
            # the `is not None` below is a guard against a driver that refuses rather than
            # a path any supported store takes — it is not covered, and cannot be.
            indexing = created.GetIndexingVariable() or rg.OpenMDArray(dim_name)
            if indexing is not None:
                write_attributes_to_md_array(indexing, carried)
        gdal_extra_dims.append(created)
    return gdal_extra_dims


def _write_grid_mapping(rg: Any, md_arr: Any, srse: Any) -> None:
    """Write a CF ``grid_mapping`` variable and link it from the data variable.

    Used on the MEM driver only — the netCDF driver creates its own grid mapping
    via ``SetSpatialRef``. The variable is named ``spatial_ref`` to avoid colliding
    with GDAL's automatic ``crs`` during a later ``CreateCopy`` to netCDF.
    """
    gm_name, gm_params = srs_to_grid_mapping(srse)
    gm_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
    gm_var_name = "spatial_ref"
    crs_arr = rg.CreateMDArray(gm_var_name, [], gm_dtype)
    crs_arr.Write(np.array(0, dtype=np.int32))
    gm_params["grid_mapping_name"] = gm_name
    write_attributes_to_md_array(crs_arr, gm_params)
    write_attributes_to_md_array(md_arr, {"grid_mapping": gm_var_name})


def _is_dask_array(obj: Any) -> bool:
    """Return whether ``obj`` is a dask array, without importing dask.

    Duck-types on the public block-iteration surface the streaming write path
    relies on (``compute`` / ``blocks`` / ``chunks`` / ``numblocks``) plus a
    ``dask`` module origin. A plain ``np.ndarray`` (or any non-dask object) is
    therefore never mistaken for one, and dask stays an optional dependency that
    this module never imports — the caller supplies the live dask array.

    Args:
        obj: Any candidate array passed as ``from_array``'s ``arr``.

    Returns:
        bool: True only for a dask array.
    """
    module = getattr(type(obj), "__module__", "") or ""
    return (
        (module == "dask" or module.startswith("dask."))
        and hasattr(obj, "compute")
        and hasattr(obj, "blocks")
        and hasattr(obj, "chunks")
        and hasattr(obj, "numblocks")
    )


def _iter_block_windows(dask_arr: Any):
    """Yield ``(block_index, starts, counts)`` for every dask block in storage order.

    ``starts`` / ``counts`` are the per-axis ``array_start_idx`` / ``count``
    windows for a GDAL windowed write, derived from the cumulative
    ``dask_arr.chunks`` offsets so the blocks tile the array exactly with no
    overlap or gap (ragged final chunks included).

    Args:
        dask_arr: A dask array (see :func:`_is_dask_array`).

    Yields:
        tuple: ``(block_index, starts, counts)`` per block, where ``block_index``
        is the tuple index into ``dask_arr.blocks``.
    """
    offsets = [np.cumsum((0,) + tuple(axis_chunks)) for axis_chunks in dask_arr.chunks]
    for block_index in np.ndindex(*dask_arr.numblocks):
        starts = [int(offsets[ax][bi]) for ax, bi in enumerate(block_index)]
        counts = [int(dask_arr.chunks[ax][bi]) for ax, bi in enumerate(block_index)]
        yield block_index, starts, counts


def _write_blocks_streaming(md_arr: Any, dask_arr: Any) -> None:
    """Write a dask array into ``md_arr`` one block at a time (memory-bounded).

    Materialises a single dask block per iteration and writes it to its window
    via ``md_arr.Write(block, array_start_idx=starts, count=counts)``, so peak
    source memory is one block rather than the whole array. This is the streaming
    equivalent of the eager single ``md_arr.Write(arr)`` call in
    :func:`_create_netcdf_from_array` and produces byte-identical output.

    Args:
        md_arr: The freshly created GDAL ``MDArray`` to populate. Its no-data and
            spatial reference must already be set (netCDF requires no-data before
            the first write).
        dask_arr: The dask array to stream, in ``(extra..., y, x)`` storage order.
    """
    for block_index, starts, counts in _iter_block_windows(dask_arr):
        block = np.asarray(dask_arr.blocks[block_index].compute())
        md_arr.Write(block, array_start_idx=starts, count=counts)


def _require_create_inputs(
    variable_name: str | None, geo: Sequence[float] | None
) -> None:
    """Validate the required inputs for `_create_netcdf_from_array`.

    Args:
        variable_name: Name of the data variable.
        geo: Geotransform tuple.

    Raises:
        ValueError: If `variable_name` or `geo` is None. `epsg` is not
            validated: it may legitimately be absent, which builds an
            ungeoreferenced variable (ARC-26).
    """
    if variable_name is None:
        raise ValueError("Variable_name cannot be None")
    if geo is None:
        raise ValueError("geo cannot be None")
    # `epsg` may legitimately be None: a source with no CRS produces an
    # ungeoreferenced result rather than one stamped with a default (ARC-26).
    # The creator below skips the spatial reference in that case.


def _resolve_write_crs(
    epsg: str | int | None,
) -> tuple[osr.SpatialReference | None, bool | None]:
    """Spatial reference and axis kind to write for a variable.

    Args:
        epsg: EPSG code, WKT/user-input CRS string, or a falsy value meaning the
            source has no CRS.

    Returns:
        tuple: `(srs_or_None, is_geographic)` where `is_geographic` is `True`
        for degrees axes, `False` for projected (metre) axes, and `None` when
        the CRS is unknown — the third state, which writes the axis role without
        asserting any units. Stamping metres there would be the same fabrication
        ARC-26 removed on the read side.
    """
    if not epsg:
        return None, None
    try:
        srse = sr_from_epsg(int(epsg))
    except (TypeError, ValueError):
        srse = sr_from_user_input(epsg)
    return srse, srse.IsGeographic() == 1


def _require_netcdf_destination(path: str | Path) -> None:
    """Refuse a destination that names a driver other than netCDF.

    This builds a multidimensional store, which only the netCDF driver carries,
    so the driver is fixed rather than resolved. What the extension decides is
    whether the caller asked for something else: `path="lies.tif"` used to
    write a netCDF under a GeoTIFF name without a word.

    Args:
        path: The destination the caller supplied.

    Raises:
        FileFormatNotSupportedError: `path` does not name the netCDF driver.
    """
    try:
        # `for_copy=True` so a copy-only extension reaches the message below
        # rather than raising the Create-gate error inside the resolver, whose
        # advice ("build it as GTiff, then convert") is wrong for a caller who
        # asked a netCDF container to write a PNG.
        resolved: str | None = resolve_output_driver(path, for_copy=True)
    except FileFormatNotSupportedError:
        # Refused outright (a reference-only format such as `.vrt`). Its advice
        # -- "write a format that owns its pixels, e.g. '.tif'" -- is advice
        # this method would also refuse, so answer in the terms of the method
        # the caller actually reached for.
        resolved = None
    if resolved == "netCDF":
        return
    names = f" -- it names {resolved}" if resolved else ""
    raise FileFormatNotSupportedError(
        "NetCDF.from_array writes a multidimensional netCDF store, but "
        f"{str(path)!r} does not name the netCDF driver{names}. Use a '.nc' or "
        "'.nc4' path, or build the raster with Dataset.from_array."
    )


def _create_netcdf_from_array(
    arr: np.ndarray,
    variable_name: str,
    cols: int,
    rows: int,
    extra_dims: list[tuple[str, list]] | None = None,
    geo: tuple[float, float, float, float, float, float] | None = None,
    epsg: str | int | None = None,
    no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
    path: str | Path | None = None,
    encoding: Encoding | None = None,
    cf_attrs: dict[str, str] | None = None,
    spatial_names: tuple[str, str] | None = None,
    dim_attrs: dict[str, dict[str, str]] | None = None,
) -> gdal.Dataset:
    """Build a multidimensional GDAL dataset from an array.

    The driver is inferred from `path`: `None` -> MEM (in-memory),
    otherwise the netCDF driver writes to disk. A `dask.array.Array`
    `arr` is streamed block-by-block via windowed writes (ARC-11); a
    NumPy `arr` is written in a single call.

    Args:
        arr: 2-D `(rows, cols)`, 3-D `(extra_dim, rows, cols)`, or
            4-D+ `(d_0, ..., d_{n-1}, rows, cols)` array — NumPy
            (eager) or `dask.array.Array` (streamed).
        variable_name: Name of the data variable.
        cols: Number of columns.
        rows: Number of rows.
        extra_dims: Ordered list of `(dim_name, values)` pairs for
            every non-spatial dimension. Length matches
            `arr.ndim - 2`. Empty list for 2-D arrays. Pre-resolved
            by `_resolve_extra_dims` so each `values` entry is a
            concrete list.
        geo: Geotransform tuple. Defaults to None.
        epsg: EPSG code. Defaults to None.
        no_data_value: No-data sentinel. Defaults to
            DEFAULT_NO_DATA_VALUE.
        path: Output file path. If None, created in memory.
            Defaults to None.
        encoding: On-disk write options — chunk sizes, compression and its level —
            as an :class:`~pyramids.netcdf.array_options.Encoding`. `None` uses the
            GDAL defaults. Defaults to None.
        cf_attrs: Optional CF global attributes (e.g. ``title`` /
            ``institution`` / ``source`` / ``history``) to merge onto the
            root group, alongside the always-written ``Conventions``.
            Defaults to None.

    Returns:
        gdal.Dataset: The created multidimensional GDAL dataset.
    """
    # Local import breaks the netcdf.py <-> engines.variables import cycle;
    # the static dimension/coordinate helpers stay on NetCDF (shared with
    # set_variable and other call sites) and are reached through the class.
    from pyramids.netcdf.netcdf import NetCDF

    encoding = encoding or Encoding()
    _require_create_inputs(variable_name, geo)
    # `_require_create_inputs` raises `ValueError` on a None `geo`; restate that
    # invariant so the geotransform indexing below is guarded. `epsg` is NOT
    # restated — it may legitimately be None, which builds an ungeoreferenced
    # variable rather than one stamped with a default (ARC-26).
    assert geo is not None

    if extra_dims is None:
        extra_dims = []
    # `arr.dtype` is a `np.dtype` for both a NumPy array and a dask array, so the
    # GDAL type is derived without materialising a (possibly out-of-core) dask input.
    dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr.dtype))
    # Coordinate arrays are always float64, never the data array's dtype --
    # exactly the rule `set_variable` already states. Sharing `dtype` truncated
    # every axis of an integer-typed variable to whole units: a container-wide
    # `to_crs` of the packed `Int16` ERA5 fixture wrote `x = [0, 3, 5, ...]` for
    # a 2.5-degree grid, so the geotransform re-derived from those coordinates
    # came back `(-1.5, 3.0, ..., -2.0)` against the true `(-1.25, 2.5, ...,
    # -2.5)`, and a `UInt16` GOES granule collapsed to `(0.0, 0.0, ...)` -- a
    # file that can no longer be placed on the earth. An epoch-valued `time`
    # axis saturated at 32767. The streamed arm (`_stream_apply_to_file`) always
    # wrote float64 here, so this also converges the two arms.
    #
    # This is the rule for the *spatial* axes, which is where truncation was the
    # whole problem. `_create_extra_dimensions` narrows it for a non-spatial
    # axis: an integer one keeps its own dtype, because float64 cannot hold an
    # int64 nanosecond epoch exactly.
    coord_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    # Build the spatial coordinate axes from the geotransform. y_axis reads the signed
    # geo[5] (not geo[1], which would square a non-square grid, e.g. 2° lon / 1° lat);
    # the geo is caller-supplied (resolve_geotransform returns it verbatim), not
    # read-normalised, so a south-up input (geo[5] > 0) writes an ascending y coordinate
    # within its extent (abs(geo[5]) would write a descending axis below the extent).
    x_dim_values = GeoTransform(*geo).x_axis(cols)
    y_dim_values = GeoTransform(*geo).y_axis(rows)

    if path is not None:
        _require_netcdf_destination(path)
        driver_type = "netCDF"
    else:
        driver_type = MEMORY_DRIVER
        path = "netcdf"

    src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(str(path))
    rg = src.GetRootGroup()
    write_global_attributes(rg, {"Conventions": "CF-1.8", **(cf_attrs or {})})

    # netCDF driver doesn't support SetIndexingVariable — create
    # dimension arrays manually without linking them.
    use_set_indexing = driver_type == "MEM"
    # `epsg` is normally an EPSG int/numeric string; keep the exact `sr_from_epsg`
    # path for those. A geostationary (and other no-EPSG) CRS is carried through
    # as a WKT string so the fan-out that rebuilds each variable preserves it
    # instead of crashing on a None EPSG code (#706).
    srse, is_geographic = _resolve_write_crs(epsg)

    # A rebuilt result names its grid after the store it came from — `longitude` /
    # `latitude` stay themselves rather than becoming `x` / `y`, which made every
    # `reduce`, `coarsen`, `rolling`, `cumsum` and `diff` hand back axes the source
    # never had, on the object and on the written file (#1180). A caller who names
    # none, and every in-memory build, keeps `y` / `x`.
    row_name, column_name = spatial_names or (ROW_AXIS, COLUMN_AXIS)
    dim_x = NetCDF._create_dimension(
        rg,
        column_name,
        coord_dtype,
        x_dim_values,
        gdal.DIM_TYPE_HORIZONTAL_X,
        use_set_indexing,
        is_geographic=is_geographic,
    )
    dim_y = NetCDF._create_dimension(
        rg,
        row_name,
        coord_dtype,
        y_dim_values,
        gdal.DIM_TYPE_HORIZONTAL_Y,
        use_set_indexing,
        is_geographic=is_geographic,
    )

    gdal_extra_dims = _create_extra_dimensions(
        rg, extra_dims, coord_dtype, use_set_indexing, dim_attrs
    )
    # For a dask input with no explicit on-disk chunking, align the netCDF storage
    # BLOCKSIZE with the dask block shape so the streamed windows map onto whole
    # storage chunks. An explicit `chunk_sizes` always wins.
    chunk_sizes = encoding.chunk_sizes
    if chunk_sizes is None and _is_dask_array(arr):
        chunk_sizes = tuple(
            int(axis_chunks[0]) for axis_chunks in cast("Any", arr).chunks
        )
    md_arr = rg.CreateMDArray(
        variable_name,
        [*gdal_extra_dims, dim_y, dim_x],
        dtype,
        _build_create_options(
            chunk_sizes, encoding.compression, encoding.compression_level
        ),
    )

    # Set metadata BEFORE writing data — netCDF driver requires
    # nodata to be set before the first Write call.
    # Tolerate both scalar and per-band sequence inputs since
    # callers often pass `Dataset.no_data_value` (now a tuple)
    # straight through.
    ndv_scalar = scalar_no_data(no_data_value)
    if ndv_scalar is not None:
        md_arr.SetNoDataValueDouble(float(ndv_scalar))
    if srse is not None:
        md_arr.SetSpatialRef(srse)
    # Eager NumPy input writes in one call; a dask input streams block-by-block so
    # peak source memory is one block rather than the whole array (ARC-11). Both
    # produce byte-identical output. nodata + SRS are set above first — the netCDF
    # driver requires nodata before the initial write.
    if _is_dask_array(arr):
        _write_blocks_streaming(md_arr, arr)
    else:
        md_arr.Write(arr)

    if driver_type == "MEM" and srse is not None:
        _write_grid_mapping(rg, md_arr, srse)

    return src
