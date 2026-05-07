from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

Number = int | float


def _strip_braces(value: str) -> str:
    """Extract the inner content of a braced string.

    This utility trims surrounding whitespace and, if the value is wrapped in
    a single pair of curly braces, returns the content inside those braces.
    Otherwise, it returns the stripped input unchanged.

    Args:
        value (str):
            Input string which may look like "{1,2,3}" or include surrounding
            whitespace.

    Returns:
        str: The substring without the outer braces and surrounding whitespace.

    Raises:
        TypeError: If `value` is not a string.

    Examples:
        - Typical usage: remove outer braces and spaces
            ```python
            >>> from pyramids.netcdf.dimensions import _strip_braces
            >>> _strip_braces(" {1,2,3} ")
            '1,2,3'

            ```
        - No braces: returns stripped content
            ```python
            >>> _strip_braces(" a, b ")
            'a, b'

            ```
        - Empty braces: yields empty string
            ```python
            >>> _strip_braces("{}")
            ''

            ```

    See Also:
        - :func:`_smart_split_csv`: Uses this helper to normalize values.
    """
    if not isinstance(value, str):
        raise TypeError("value must be a str")
    s = value.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1].strip()
    return s


def _smart_split_csv(text: str) -> list[str]:
    """Split a comma-separated string, honoring optional outer braces.

    The function trims whitespace, removes a single pair of surrounding braces
    if present, and then splits on commas, returning only non-empty tokens.

    Args:
        text (str):
            The input string. May be braced (e.g., "{a, b, c}") or unbraced
            (e.g., "a, b, c").

    Returns:
        list[str]: A list of non-empty, trimmed tokens. Returns an empty list
        if the content is empty.

    Raises:
        TypeError: If `text` is not a string.

    Examples:
        - Braced input with spaces
            ```python
            >>> from pyramids.netcdf.dimensions import _smart_split_csv
            >>> _smart_split_csv('{ a, b, c }')
            ['a', 'b', 'c']

            ```
        - Unbraced input
            ```python

            >>> _smart_split_csv('x, y')
            ['x', 'y']

            ```
        - Empty content
            ```python
            >>> _smart_split_csv('{}')
            []

            ```

    See Also:
        - :func:`_strip_braces`: Used to remove the optional braces.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    core = _strip_braces(text)
    if core:
        vals = [item.strip() for item in core.split(",") if item.strip()]
    else:
        vals = []
    return vals


def _coerce_scalar(token: str) -> str | Number:
    """Convert a string token to int or float when possible.

    The conversion strategy is conservative:
      - If the token matches an integer pattern (optional sign, digits only),
        it's converted to `int`.
      - Otherwise, if `float(token)` succeeds, it's converted to `float`.
      - On any failure, the original string is returned unchanged.

    Args:
        token (str): The input string token.

    Returns:
        str | Number: The coerced scalar (`int` or `float`) or the
        original string if not numeric.

    Raises:
        TypeError: If `token` is not a string.

    Examples:
        - Integer-like token
            ```python
            >>> from pyramids.netcdf.dimensions import _coerce_scalar
            >>> _coerce_scalar('42')
            42

            ```
        - Float-like token
            ```python
            >>> _coerce_scalar('3.14')
            3.14

            ```
        - Non-numeric token stays a string
            ```python
            >>> _coerce_scalar('foo')
            'foo'

            ```
    """
    if not isinstance(token, str):
        raise TypeError("token must be a str")
    # Try int
    try:
        # Account for +/-, hex/float-like strings should fail here
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
    except ValueError:
        pass
    # Try float
    try:
        return float(token)
    except ValueError:
        return token


def _parse_values_list(text: str) -> list[str | Number]:
    """Parse a CSV/braced list into typed scalars.

    This helper combines :func:`_smart_split_csv` and :func:`_coerce_scalar` to
    produce a list of tokens converted to `int`/`float` where possible.

    Args:
        text (str): The metadata text containing values (e.g., "{0,31}").

    Returns:
        list[str | Number]: A list of numbers and/or strings depending on
        the input tokens.

    Raises:
        TypeError: If `text` is not a string.

    Examples:
        - Mixed numeric and text values
            ```python
            >>> from pyramids.netcdf.dimensions import _parse_values_list
            >>> _parse_values_list('{0, 31, foo}')
            [0, 31, 'foo']

            ```
        - Empty list
            ```python
            >>> _parse_values_list('{}')
            []

            ```
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return [_coerce_scalar(t) for t in _smart_split_csv(text)]


def _format_braced_list(values: Iterable[str | Number]) -> str:
    """Format a sequence as a GDAL-style braced list without spaces.

    Args:
        values (Iterable[str | Number]):
            Iterable of scalars to format. Each element is converted with `str`.

    Returns:
        str: A string like "{1,2,foo}". Empty iterables yield "{}".

    Raises:
        TypeError: If `values` is not iterable.

    Examples:
        - Regular values
            ```python
            >>> from pyramids.netcdf.dimensions import _format_braced_list
            >>> _format_braced_list([1, 2, 3])
            '{1,2,3}'

            ```
        - Mixed types
            ```python
            >>> _format_braced_list(['a', 1, 2.5])
            '{a,1,2.5}'

            ```
        - Empty input
            ```python
            >>> _format_braced_list([])
            '{}'

            ```
    """
    return "{" + ",".join(str(v) for v in values) + "}"


@dataclass(frozen=True)
class DimMetaData:
    """Unified information for a single netCDF dimension.

    This immutable dataclass captures both the structural information the GDAL
    netCDF driver exposes via `NETCDF_DIM_*` keys and, optionally, the
    per-dimension attribute mapping collected from keys of the form
    `"<name>#<attr>"` (e.g., `time#units`).

    It subsumes the previous "Dimension" helper by adding an `attrs` field
    while still preserving the original `raw` bucket that stores the exact
    strings parsed from metadata.

    Args:
        name (str):
            Dimension name (e.g., "time", "level0").
        size (int | None):
            Dimension length (if known). Often derived from the first integer
            in `*_DEF` or the length of `*_VALUES`.
        values (list[str | Number] | None):
            Parsed scalar values from the `*_VALUES` entry, if provided by the
            GDAL netCDF driver.
        def_fields (tuple[int,...] | None):
            Parsed integers from the `*_DEF` entry. The meaning is driver-
            specific; commonly the first value corresponds to the dimension size.
        raw (dict[str, str]):
            Raw strings captured from metadata for this dimension (e.g., the
            original `DEF` and `VALUES` content).
        attrs (dict[str, str]):
            Optional attribute dictionary associated with the same dimension
            name (e.g., `{"axis": "T", "units": "days since..."}`).

    Raises:
        ValueError: If `size` is negative.

    Examples:
        - Construct manually for testing
            ```python
            >>> from pyramids.netcdf.dimensions import DimMetaData
            >>> d = DimMetaData(name='time', size=2, values=[0, 31], def_fields=(2, 6))
            >>> d.name, d.size, d.values, d.def_fields
            ('time', 2, [0, 31], (2, 6))

            ```
        - With attributes merged
            ```python

            >>> d = DimMetaData(name='time', size=2, values=[0, 31], def_fields=(2, 6), attrs={'axis': 'T'})
            >>> d.attrs['axis']
            'T'

            ```

    See Also:
        - :class:`DimensionsIndex`: Factory that populates structural entries.
        - :class:`MetaData`: Provides convenient construction with merged attrs.
    """

    name: str
    size: int | None = None
    values: list[str | Number] | None = None
    def_fields: tuple[int, ...] | None = None
    raw: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimension fields for consistency."""
        if not self.name:
            raise ValueError("Dimension name cannot be empty.")
        if self.size is not None and self.size < 0:
            raise ValueError(
                f"Dimension '{self.name}': size cannot be negative, "
                f"got {self.size}."
            )
        if self.values is not None and self.size is not None:
            if len(self.values) != self.size:
                raise ValueError(
                    f"Dimension '{self.name}': values length "
                    f"({len(self.values)}) does not match size "
                    f"({self.size})."
                )


@dataclass
class DimensionsIndex:
    """Index of netCDF dimensions parsed from GDAL metadata.

    A thin mapping-like container that stores :class:`DimMetaData` objects
    keyed by dimension name. Use :meth:`from_metadata` to construct an index
    from a GDAL metadata mapping (e.g., `gdal.Dataset.GetMetadata()`).

    Behavior:
      - Accepts dimensions listed under `<prefix>EXTRA`.
      - Also recognizes any `<prefix><name>_DEF` and `<prefix><name>_VALUES`
        keys, even when `<name>` is not listed in `EXTRA`.
      - Coerces numeric tokens to `int`/`float` where possible.

    Notes:
        The default prefix is `NETCDF_DIM_` but any prefix can be supplied to
        :meth:`from_metadata` and :meth:`to_metadata`.

    See Also:
        - :class:`DimMetaData`
        - :class:`MetaData` for a higher-level view that merges attributes
          like `time#units` with dimension structure.

    Examples:
        - Build from typical NETCDF_DIM_* keys
            ```python
            >>> from pyramids.netcdf.dimensions import DimensionsIndex
            >>> md = {
            ...     'NETCDF_DIM_EXTRA': '{time,level0}',
            ...     'NETCDF_DIM_level0_DEF': '{3,6}',
            ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            ...     'NETCDF_DIM_time_DEF': '{2,6}',
            ...     'NETCDF_DIM_time_VALUES': '{0,31}',
            ... }
            >>> idx = DimensionsIndex.from_metadata(md)
            >>> sorted(idx.names)
            ['level0', 'time']
            >>> idx['time'].size
            2
            >>> idx['level0'].values
            [1, 2, 3]

            ```
        - Using a custom prefix
            ```python
            >>> md = {
            ...     'CUSTOM_DIM_time_DEF': '{2,6}',
            ...     'CUSTOM_DIM_time_VALUES': '{0,31}',
            ... }
            >>> DimensionsIndex.from_metadata(md).names
            []
            >>> DimensionsIndex.from_metadata(md, prefix='CUSTOM_DIM_').names
            ['time']

            ```
    """

    _dims: dict[str, DimMetaData] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        prefix: str = "NETCDF_DIM_",
    ) -> DimensionsIndex:
        """Parse dimensions from a GDAL metadata dictionary.

        Args:
            metadata (Mapping[str, str]):
                GDAL metadata mapping (e.g., from `Dataset.GetMetadata()`).
            prefix (str, optional):
                Key prefix to filter on (default: `'NETCDF_DIM_'`). Custom
                prefixes are supported, e.g., `'CUSTOM_DIM_'`.

        Returns:
            DimensionsIndex: Parsed index of dimensions.

        Raises:
            TypeError: If `metadata` is not a mapping or contains non-string
                keys/values.

        Examples:
            - Basic usage with standard prefix
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> md = {
                ...     'NETCDF_DIM_EXTRA': '{time,level0}',
                ...     'NETCDF_DIM_level0_DEF': '{3,6}',
                ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
                ...     'NETCDF_DIM_time_DEF': '{2,6}',
                ...     'NETCDF_DIM_time_VALUES': '{0,31}',
                ... }
                >>> idx = DimensionsIndex.from_metadata(md)
                >>> idx['time'].size
                2

                ```
            - Using a custom prefix
                ```python
                >>> md = {
                ...     'CUSTOM_DIM_time_DEF': '{2,6}',
                ...     'CUSTOM_DIM_time_VALUES': '{0,31}',
                ... }
                >>> DimensionsIndex.from_metadata(md).names
                []
                >>> DimensionsIndex.from_metadata(md, prefix='CUSTOM_DIM_').names
                ['time']

                ```
        """
        # Gather candidate keys first
        dim_names: set[str] = set()
        buckets: dict[str, dict[str, str]] = {}

        # Build a regex that respects the provided prefix
        # Example: ^NETCDF_DIM_(<name>)(?:_(DEF|VALUES))?$ when prefix is default
        _DIM_KEY_RE = re.compile(
            rf"^{re.escape(prefix)}([A-Za-z0-9_.-]+?)(?:_(DEF|VALUES))?$"
        )

        for key, value in metadata.items():
            if not key.startswith(prefix):
                continue
            m = _DIM_KEY_RE.match(key)
            if not m:
                # Keep unknown but prefixed keys under a synthetic "_misc" bucket
                buckets.setdefault("_misc", {})[key] = value
                continue
            name, suffix = m.groups()

            # Special case: EXTRA carries a comma-separated list of dim names
            if name.upper() == "EXTRA" and suffix is None:
                for nm in _smart_split_csv(value):
                    if nm:
                        dim_names.add(nm)
                continue

            # Normal dimension attributes (DEF/VALUES)
            buckets.setdefault(name, {})[suffix or "_root"] = value
            dim_names.add(name)

        dims: dict[str, DimMetaData] = {}
        for name in sorted(dim_names):
            raw_bucket = buckets.get(name, {})
            # DEF may contain integers, often first item is size
            def_fields: tuple[int, ...] | None = None
            size: int | None = None
            if "DEF" in raw_bucket:
                def_list = _parse_values_list(raw_bucket["DEF"])
                # Only keep integers in def_fields; ignore non-ints gracefully
                ints = tuple(int(x) for x in def_list if isinstance(x, int))
                def_fields = ints or None
                if ints:
                    size = ints[0]

            # VALUES: keep as numbers/strings
            values: list[str | Number] | None = None
            if "VALUES" in raw_bucket:
                values = _parse_values_list(raw_bucket["VALUES"]) or None
                # If size wasn't in DEF, infer from VALUES length
                if size is None and values is not None:
                    size = len(values)

            dims[name] = DimMetaData(
                name=name,
                size=size,
                values=values,
                def_fields=def_fields,
                raw=dict(raw_bucket),
            )

        return cls(dims)

    @property
    def names(self) -> list[str]:
        """Return the list of dimension names.

        Returns:
            list[str]: Names in insertion order (or sorted order, depending on
            construction) matching the keys of the index.

        Examples:
            - Simple index
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_x_DEF': '{2,0}'})
                >>> idx.names
                ['x']

                ```
        """
        return list(self._dims.keys())

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Number of dimensions in the index.

        Returns:
            int: Count of stored dimensions.

        Examples:
            - Length of a simple index
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_time_DEF': '{2,6}'})
                >>> len(idx)
                1

                ```
        """
        return len(self._dims)

    def __iter__(self) -> Iterable[DimMetaData]:  # pragma: no cover - trivial
        """Iterate over stored dimensions.

        Yields:
            DimMetaData: Each stored dimension in unspecified order.

        Examples:
          - Iterate names
            ```python
            >>> from pyramids.netcdf.dimensions import DimensionsIndex
            >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_a_DEF': '{1,0}', 'NETCDF_DIM_b_DEF': '{2,0}'})
            >>> [d.name for d in idx]
            ['a', 'b']

            ```
        """
        return iter(self._dims.values())

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        """Check if a dimension name exists in the index.

        Args:
            name (str): Dimension name to check.

        Returns:
            bool: `True` if present, else `False`.

        Examples:
            - Membership test
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_time_DEF': '{2,6}'})
                >>> 'time' in idx, 'lat' in idx
                (True, False)

                ```
        """
        return name in self._dims

    def __getitem__(self, name: str) -> DimMetaData:
        """Get a dimension by name.

        Args:
            name (str): Dimension name.

        Returns:
            DimMetaData: The matching dimension.

        Raises:
            KeyError: If the name is not present in the index.

        Examples:
            - Access an existing dimension
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_time_DEF': '{2,6}'})
                >>> idx['time'].size
                2

                ```
        """
        return self._dims[name]

    def __str__(self) -> str:
        """Return a compact, human-readable summary of the index.

        Returns:
            str: A multi-line string listing each dimension with size, values
            and DEF fields when available.

        Examples:
            - Pretty-print a small index
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'NETCDF_DIM_time_VALUES': '{0,31}'}
                >>> str(DimensionsIndex.from_metadata(md)).splitlines()[0]
                'DimensionsIndex(1 dims)'

                ```
        """
        lines: list[str] = [f"DimensionsIndex({len(self)} dims)"]
        for name in sorted(self._dims):
            d = self._dims[name]
            parts: list[str] = []
            if d.size is not None:
                parts.append(f"size={d.size}")
            if d.values is not None:
                vals = ", ".join(str(v) for v in d.values)
                parts.append(f"values=[{vals}]")
            if d.def_fields is not None:
                parts.append(f"def={tuple(d.def_fields)}")
            detail = ", ".join(parts) if parts else "(no details)"
            lines.append(f"- {name}: {detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize the index to a plain dictionary.

        Useful for logging, debugging, or JSON/YAML output.

        Returns:
            dict[str, dict[str, object]]: Mapping from dimension name to a
            structure with `size`, `values` and `def_fields` fields.

        Examples:
            - Convert to a simple dict
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> idx = DimensionsIndex.from_metadata({'NETCDF_DIM_time_DEF': '{2,6}', 'NETCDF_DIM_time_VALUES': '{0,31}'})
                >>> d = idx.to_dict()
                >>> sorted(list(d['time'].keys()))
                ['def_fields', 'size', 'values']

                ```
        """
        out: dict[str, dict[str, object]] = {}
        for k, d in self._dims.items():
            out[k] = {
                "size": d.size,
                "values": d.values,
                "def_fields": d.def_fields,
            }
        return out

    def to_metadata(
        self,
        *,
        prefix: str = "NETCDF_DIM_",
        include_extra: bool = True,
        sort_names: bool = True,
    ) -> dict[str, str]:
        """Serialize the index back to GDAL netCDF metadata keys.

        This produces keys compatible with the netCDF GDAL driver such as
        `<prefix>EXTRA`, `<prefix><name>_DEF` and `<prefix><name>_VALUES`.

        Args:
            prefix (str): Metadata key prefix (defaults to `"NETCDF_DIM_"`).
            include_extra (bool): Whether to include the `<prefix>EXTRA` key
                listing dimension names.
            sort_names (bool): Whether to sort names deterministically in
                outputs.

        Returns:
            dict[str, str]: A dictionary suitable for use with GDAL's metadata API.

        Examples:
            - Round-trip a simple index
                ```python
                >>> from pyramids.netcdf.dimensions import DimensionsIndex
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'NETCDF_DIM_time_VALUES': '{0,31}'}
                >>> idx = DimensionsIndex.from_metadata(md)
                >>> out = idx.to_metadata()
                >>> sorted(out.keys())
                ['NETCDF_DIM_EXTRA', 'NETCDF_DIM_time_DEF', 'NETCDF_DIM_time_VALUES']

                ```
        """
        md: dict[str, str] = {}
        names = list(self._dims.keys())
        if sort_names:
            names.sort()
        if include_extra and names:
            md[f"{prefix}EXTRA"] = _format_braced_list(names)
        for name in names:
            d = self._dims[name]
            if d.def_fields:
                md[f"{prefix}{name}_DEF"] = _format_braced_list(d.def_fields)
            if d.values is not None:
                md[f"{prefix}{name}_VALUES"] = _format_braced_list(d.values)
        return md


def parse_gdal_netcdf_dimensions(metadata: Mapping[str, str]) -> DimensionsIndex:
    """Parse netCDF dimension info from GDAL metadata.

    A convenience wrapper around :meth:`DimensionsIndex.from_metadata` that
    uses the default `NETCDF_DIM_` prefix.

    Args:
        metadata (Mapping[str, str]): GDAL metadata mapping (e.g., from
            `Dataset.GetMetadata()`).

    Returns:
        DimensionsIndex: Parsed index of dimensions.

    Raises:
        TypeError: If `metadata` is not a mapping.

    Examples:
        - Typical usage
            ```python
            >>> md = {
            ...     'NETCDF_DIM_EXTRA': '{time,level0}',
            ...     'NETCDF_DIM_level0_DEF': '{3,6}',
            ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            ...     'NETCDF_DIM_time_DEF': '{2,6}',
            ...     'NETCDF_DIM_time_VALUES': '{0,31}',
            ... }
            >>> idx = parse_gdal_netcdf_dimensions(md)
            >>> idx.to_dict()['time']['size']
            2
            >>> idx.to_dict()['level0']['values']
            [1, 2, 3]

            ```

    See Also:
        - :class:`DimensionsIndex`
    """
    return DimensionsIndex.from_metadata(metadata)


def parse_dimension_attributes(
    metadata: Mapping[str, str],
    names: Iterable[str] | None = None,
    *,
    normalize_attr_keys: bool = True,
) -> dict[str, dict[str, str]]:
    """Extract per-dimension attributes from GDAL netCDF metadata.

    This helper scans metadata entries whose keys look like "<name>#<attr>"
    (e.g., "time#axis", "lat#units", "level0#positive") and groups them by
    dimension name.

    Args:
        metadata (Mapping[str, str]):
            Mapping of metadata keys to values (e.g., from GDAL).
        names (Iterable[str] | None):
            Optional iterable of dimension names to include. If provided, only
            attributes for these names are captured.
        normalize_attr_keys (bool):
            If True, attribute names after the "#" are converted to lowercase in
            the output. If False, original case is preserved.

    Returns:
        dict[str, dict[str, str]]: A mapping from dimension name to a dictionary
        of attributes for that dimension.

    Raises:
        TypeError: If `metadata` is not a mapping or contains non-string keys.

    Examples:
        - Parse all attributes for any name
            ```python
            >>> md = {
            ...     'lat#bounds': 'bounds_lat',
            ...     'lat#long_name': 'latitude',
            ...     'lat#units': 'degrees_north',
            ...     'time#axis': 'T',
            ...     'time#long_name': 'time',
            ...     'time#units': 'days since 1-1-1 0:0:0',
            ... }
            >>> parse_dimension_attributes(md)
            {'lat': {'bounds': 'bounds_lat', 'long_name': 'latitude', 'units': 'degrees_north'}, 'time': {'axis': 'T', 'long_name': 'time', 'units': 'days since 1-1-1 0:0:0'}}

            ```
        - Restrict to provided names and preserve attribute case
            ```python
            >>> parse_dimension_attributes(md, names=['time'], normalize_attr_keys=False)
            {'time': {'axis': 'T', 'long_name': 'time', 'units': 'days since 1-1-1 0:0:0'}}

            ```

    See Also:
        - :class:`MetaData`: Combines these attributes with dimension structure.
    """
    # Build a quick lookup for allowed names if provided
    allowed = set(names) if names is not None else None
    out: dict[str, dict[str, str]] = {}

    # Simple pattern: everything before first '#' is the dimension name; after is attribute
    # Keep it permissive but avoid empty parts.
    key_re = re.compile(r"^([^#\s]+)#([^#\s]+)$")

    for k, v in metadata.items():
        m = key_re.match(k.strip())
        if not m:
            continue
        name, attr = m.group(1), m.group(2)
        if allowed is not None and name not in allowed:
            continue
        if normalize_attr_keys:
            attr = attr.lower()
        bucket = out.setdefault(name, {})
        bucket[attr] = v

    return out


@dataclass
class MetaData:
    """Aggregate of dimension structure and per-dimension attributes.

    This class ties together two complementary pieces of information commonly
    exposed by the GDAL netCDF driver:
      - A :class:`DimensionsIndex` parsed from `NETCDF_DIM_*` keys, describing
        dimension sizes, DEF fields, and VALUES.
      - Per-dimension attribute dictionaries collected from keys of the form
        `"<name>#<attr>"` (e.g., `time#units`, `lat#axis`).

    Examples:
        - Build from a combined metadata mapping and inspect
            ```python
            >>> from pyramids.netcdf.dimensions import MetaData
            >>> md = {
            ...     'NETCDF_DIM_EXTRA': '{time,level0}',
            ...     'NETCDF_DIM_time_DEF': '{2,6}',
            ...     'NETCDF_DIM_time_VALUES': '{0,31}',
            ...     'NETCDF_DIM_level0_DEF': '{3,6}',
            ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
            ...     'time#axis': 'T',
            ...     'time#units': 'days since 1-1-1 0:0:0',
            ...     'level0#axis': 'Z',
            ... }
            >>> meta = MetaData.from_metadata(md)
            >>> sorted(meta.names)
            ['level0', 'time']
            >>> meta.get_attrs('time')['axis']
            'T'

            ```

    See Also:
        - :class:`DimensionsIndex`
        - :func:`parse_dimension_attributes`
    """

    dims: DimensionsIndex
    attrs: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        prefix: str = "NETCDF_DIM_",
        normalize_attr_keys: bool = True,
        names: Iterable[str] | None = None,
    ) -> MetaData:
        """Build a MetaData object by parsing a GDAL metadata mapping.

        Args:
            metadata (Mapping[str, str]):
                GDAL metadata map (e.g., `Dataset.GetMetadata()`).
            prefix (str):
                Prefix used for dimension entries (defaults to `NETCDF_DIM_`).
            normalize_attr_keys (bool):
                Normalize attribute keys (the part after `#`) to lowercase.
            names (Iterable[str] | None):
                If provided, limit attribute parsing to these names. By default
                uses the dimension names discovered under the prefix.

        Returns:
            MetaData: Combined structure and attributes parsed from metadata.

        Raises:
            TypeError: If the input mapping contains non-string keys/values.

        Examples:
            - Typical usage
                ```python
                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {
                ...     'NETCDF_DIM_time_DEF': '{2,6}',
                ...     'NETCDF_DIM_time_VALUES': '{0,31}',
                ...     'time#axis': 'T',
                ... }
                >>> meta = MetaData.from_metadata(md)
                >>> meta.get_dimension('time').size
                2

                ```
        """
        dims = DimensionsIndex.from_metadata(metadata, prefix=prefix)
        # Decide which names we keep attributes for
        attr_names = list(names) if names is not None else dims.names
        attrs = parse_dimension_attributes(
            metadata, names=attr_names, normalize_attr_keys=normalize_attr_keys
        )
        return cls(dims=dims, attrs=attrs)

    @property
    def names(self) -> list[str]:
        """Return the list of dimension names represented in this metadata.

        Returns:
            list[str]: Names present in the underlying :class:`DimensionsIndex`.

        Examples:
          - Inspect names
            ```python
            >>> from pyramids.netcdf.dimensions import MetaData
            >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'time#axis': 'T'}
            >>> MetaData.from_metadata(md).names
            ['time']

            ```
        """
        return self.dims.names

    def get_attrs(self, name: str) -> dict[str, str]:
        """Return attributes for a given dimension name.

        Args:
            name (str): Dimension name.

        Returns:
            dict[str, str]: Attribute dictionary; empty if the name is unknown
            or has no attributes.

        Examples:
            - Access attributes safely
                ```python
                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'time#units': 'days'}
                >>> meta = MetaData.from_metadata(md)
                >>> meta.get_attrs('time')
                {'units': 'days'}
                >>> meta.get_attrs('lat')
                {}

                ```
        """
        return self.attrs.get(name, {})

    def get_dimension(self, name: str) -> DimMetaData | None:
        """Return a DimMetaData with merged attributes for a given name, if present.

        Combines structural info from :class:`DimensionsIndex` with the
        attribute dictionary captured for the same name and returns a new
        :class:`DimMetaData` instance that includes both sets of information.

        Args:
            name (str): Dimension name.

        Returns:
            DimMetaData | None: The merged view if available, else `None`.

        Examples:
            - Get a merged DimMetaData and inspect attributes
                ```python
                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'time#axis': 'T'}
                >>> meta = MetaData.from_metadata(md)
                >>> dim = meta.get_dimension('time')
                >>> (dim.name, dim.size, dim.attrs['axis'])
                ('time', 2, 'T')

                ```
            - Unknown name returns None
                ```python
                >>> meta.get_dimension('lat') is None
                True

                ```
        """
        if name not in self.dims:
            return None
        d = self.dims[name]
        return DimMetaData(
            name=d.name,
            size=d.size,
            values=d.values,
            def_fields=d.def_fields,
            raw=dict(d.raw),
            attrs=self.get_attrs(name),
        )

    def iter_dimensions(self) -> Iterable[DimMetaData]:
        """Iterate over merged DimMetaData objects in name-sorted order.

        Yields:
            DimMetaData: Each dimension with merged structure and attributes.

        Examples:
            - Iterate and collect names
                ```python

                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {'NETCDF_DIM_b_DEF': '{1,0}', 'NETCDF_DIM_a_DEF': '{2,0}'}
                >>> meta = MetaData.from_metadata(md)
                >>> [d.name for d in meta.iter_dimensions()]
                ['a', 'b']

                ```
        """
        for name in sorted(self.names):
            dim = self.get_dimension(name)
            if dim is not None:
                yield dim

    def to_metadata(
        self,
        *,
        prefix: str = "NETCDF_DIM_",
        include_extra: bool = True,
        sort_names: bool = True,
        include_attrs: bool = True,
    ) -> dict[str, str]:
        """Serialize back to a GDAL metadata mapping.

        Combines the dimension keys produced by :meth:`DimensionsIndex.to_metadata`
        with flattened attribute keys of the form `"<name>#<attr>"`.

        Args:
            prefix (str): Metadata key prefix for dimension keys. Defaults to
                `"NETCDF_DIM_"`.
            include_extra (bool): Include an `<prefix>EXTRA` key listing
                dimension names.
            sort_names (bool): Sort dimension names when serializing for
                determinism.
            include_attrs (bool): Whether to include `"<name>#<attr>"` keys
                for known attributes.

        Returns:
            dict[str, str]: A single flattened mapping suitable for GDAL.

        Examples:
            - Merge structure and attributes
                ```python
                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'time#axis': 'T'}
                >>> meta = MetaData.from_metadata(md)
                >>> out = meta.to_metadata()
                >>> sorted(out.keys())
                ['NETCDF_DIM_EXTRA', 'NETCDF_DIM_time_DEF', 'time#axis']

                ```
        """
        md = self.dims.to_metadata(
            prefix=prefix, include_extra=include_extra, sort_names=sort_names
        )
        if include_attrs and self.attrs:
            names = list(self.names)
            if sort_names:
                names.sort()
            for name in names:
                a = self.attrs.get(name) or {}
                # Sort attributes to keep deterministic order
                for k in sorted(a.keys()):
                    md[f"{name}#{k}"] = a[k]
        return md

    def __str__(self) -> str:
        """Return a readable summary of dimensions and attributes.

        Returns:
            str: A multi-line summary listing each dimension name with basic
            statistics (size, number of values, attribute count).

        Examples:
            - Pretty-print a MetaData summary
                ```python

                >>> from pyramids.netcdf.dimensions import MetaData
                >>> md = {'NETCDF_DIM_time_DEF': '{2,6}', 'time#axis': 'T'}
                >>> s = str(MetaData.from_metadata(md))
                >>> s.splitlines()[0].startswith('MetaData(')
                True

                ```
        """
        lines: list[str] = [
            f"MetaData({len(self.dims)} dims, attrs for {len(self.attrs)} names)"
        ]
        # Show a compact, aligned summary for each dimension
        for name in sorted(self.dims.names):
            d = self.dims[name]
            parts: list[str] = []
            if d.size is not None:
                parts.append(f"size={d.size}")
            if d.values is not None:
                parts.append(f"values={len(d.values)} items")
            a = self.attrs.get(name)
            if a:
                parts.append(f"attrs={len(a)}")
            detail = ", ".join(parts) if parts else "(no details)"
            lines.append(f"- {name}: {detail}")
        return "\n".join(lines)
