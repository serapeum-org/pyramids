"""Label and nearest-value matching primitives for :meth:`NetCDF.sel`.

Pure helpers — no pyramids imports — shared by the eager selection engine
(:mod:`pyramids.netcdf.engines.selection`) and the plot engine's flat-band index
(:mod:`pyramids.netcdf._plot`), so the two paths resolve a selector identically.

Two concerns live here:

* **Date-label selection.** A CF time axis is *stored* as raw offsets
  (``[0.0, 6.0, 12.0, 18.0]`` for ``hours since 2024-01-01``), while every accessor
  that decodes it — :meth:`NetCDF.get_time_variable` — hands back date strings. These
  helpers turn a date label, whole or partial (``"2024"`` through
  ``"2024-01-01 18:00:00"``), into the strftime format that decodes the axis at the
  *same* precision, so matching is a plain string comparison and a partial label
  matches every step inside the period it names.
* **Nearest-value selection.** ``method="nearest"`` snaps a numeric selector to the
  closest coordinate on the axis, which is how a caller asks for "the level nearest
  100 m" without knowing the axis values up front.

Ranges are compared as text, which is exact while ``%Y`` renders a zero-padded
four-digit year — true for every calendar cftime decodes in the CE range, and the reason
:data:`_UPPER_TEMPLATE` can stop at 9999. A palaeo or idealised axis outside that range
would not sort correctly, and is not supported. The *labels* themselves are whatever the
dimension's CF ``calendar`` produces, so a ``360_day`` axis with a real ``2024-02-30``
sorts and pads like any other — nothing here assumes a Gregorian month length.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable
from typing import Any

FULL_FORMAT = "%Y-%m-%d %H:%M:%S"
"""strftime format of a fully-qualified date label — the widest precision supported."""

_LOWER_TEMPLATE = "0001-01-01 00:00:00"
_UPPER_TEMPLATE = "9999-12-31 23:59:59"

# A label is recognised by its *shape*, not merely its length: "control" is seven
# characters like "2024-01" but is an ensemble member's name, not a date, and belongs on
# the stored-value path.
_LABEL_PATTERN = re.compile(
    r"\d{4}(?:-\d{2}(?:-\d{2}(?:[ ]\d{2}(?::\d{2}(?::\d{2})?)?)?)?)?\Z"
)

_PRECISION_FORMATS = {
    4: "%Y",
    7: "%Y-%m",
    10: "%Y-%m-%d",
    13: "%Y-%m-%d %H",
    16: "%Y-%m-%d %H:%M",
    19: FULL_FORMAT,
}


def normalise_label(text: str) -> str:
    """Normalise a date label to the ``YYYY-MM-DD HH:MM:SS`` spelling this module matches on.

    Accepts the ISO ``T`` separator and a trailing ``Z``, so a label copied out of an
    ISO timestamp resolves the same as the space-separated form the CF decoders emit.

    Args:
        text: A whole or partial date label, e.g. ``"2024-01-01T06:00:00Z"``.

    Returns:
        str: The same instant spelled with a space separator and no zone suffix.

    Examples:
        - The ISO ``T`` separator and trailing ``Z`` are normalised away:
            ```python
            >>> from pyramids.netcdf._label_select import normalise_label
            >>> normalise_label("2024-01-01T06:00:00Z")
            '2024-01-01 06:00:00'

            ```
        - A partial label passes through unchanged apart from surrounding space:
            ```python
            >>> from pyramids.netcdf._label_select import normalise_label
            >>> normalise_label("  2024-01 ")
            '2024-01'

            ```
    """
    label = text.strip()
    if label.endswith("Z"):
        label = label[:-1].strip()
    if len(label) > 10 and label[10] == "T":
        label = f"{label[:10]} {label[11:]}"
    return label


def label_format(text: str) -> str:
    """Return the strftime format that decodes an axis at ``text``'s precision.

    Args:
        text: A whole or partial date label (normalised or not).

    Returns:
        str: The matching strftime format, e.g. ``"%Y-%m"`` for ``"2024-01"``.

    Raises:
        ValueError: The label is not one of the supported precisions.

    Examples:
        - A date-only label decodes the axis to dates:
            ```python
            >>> from pyramids.netcdf._label_select import label_format
            >>> label_format("2024-01-01")
            '%Y-%m-%d'

            ```
        - A year-only label decodes the axis to years:
            ```python
            >>> from pyramids.netcdf._label_select import label_format
            >>> label_format("2024")
            '%Y'

            ```
        - A label between two precisions is rejected, listing the ones that work:
            ```python
            >>> from pyramids.netcdf._label_select import label_format
            >>> label_format("2024-01-01 06:0")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: '2024-01-01 06:0' is not a supported date label. Write one of...

            ```

    See Also:
        pad_label: extends a partial label to the edge of the period it names.
    """
    label = normalise_label(text)
    fmt = _PRECISION_FORMATS.get(len(label)) if _LABEL_PATTERN.match(label) else None
    if fmt is None:
        raise ValueError(
            f"{text!r} is not a supported date label. Write one of "
            "'2024', '2024-01', '2024-01-01', '2024-01-01 06', '2024-01-01 06:00', "
            "'2024-01-01 06:00:00'."
        )
    return fmt


def pad_label(text: str, *, upper: bool) -> str:
    """Extend a partial date label to full precision, at the start or the end of its period.

    ``"2024-01"`` names a *period*, not an instant: as the lower bound of a range it
    means that period's first second, as the upper bound its last. Padding with the
    matching template makes an inclusive string comparison behave that way. The upper
    template pads to day 31 whatever the month's real length — the result is only ever
    compared against decoded labels, never parsed, and no real label sorts above it.

    Args:
        text: A whole or partial date label.
        upper: ``True`` pads to the last instant of the period, ``False`` to the first.

    Returns:
        str: A fully-qualified ``YYYY-MM-DD HH:MM:SS`` label.

    Examples:
        - A month as a lower bound is its first second:
            ```python
            >>> from pyramids.netcdf._label_select import pad_label
            >>> pad_label("2024-01", upper=False)
            '2024-01-01 00:00:00'

            ```
        - The same month as an upper bound is its last:
            ```python
            >>> from pyramids.netcdf._label_select import pad_label
            >>> pad_label("2024-01", upper=True)
            '2024-01-31 23:59:59'

            ```
    """
    label = normalise_label(text)
    label_format(label)  # validates the precision; the format itself is not needed here
    template = _UPPER_TEMPLATE if upper else _LOWER_TEMPLATE
    return label + template[len(label) :]


def has_label(selector: Any) -> bool:
    """Return whether ``selector`` selects by date label rather than by stored value.

    Args:
        selector: A ``sel`` selector — a scalar, a list, or a :class:`slice`.

    Returns:
        bool: ``True`` when the selector carries at least one string.

    Examples:
        - A date string selects by label:
            ```python
            >>> from pyramids.netcdf._label_select import has_label
            >>> has_label("2024-01-01")
            True

            ```
        - A slice of stored values does not:
            ```python
            >>> from pyramids.netcdf._label_select import has_label
            >>> has_label(slice(500, 1000))
            False

            ```
        - One label anywhere in a list is enough to make it a label selection:
            ```python
            >>> from pyramids.netcdf._label_select import has_label
            >>> has_label(["2024-01-01", "2024-01-02"])
            True

            ```

    See Also:
        label_indices: resolves the selectors this predicate accepts.
        nearest_indices: resolves the numeric selectors it rejects.
    """
    if isinstance(selector, slice):
        parts: tuple[Any, ...] = (selector.start, selector.stop)
    elif isinstance(selector, list):
        parts = tuple(selector)
    else:
        parts = (selector,)
    return any(isinstance(part, str) for part in parts)


def non_label_parts(selector: Any) -> list[Any]:
    """Parts of a label selector that are not labels — empty when it is purely labels.

    :func:`has_label` is true when *any* part of a selector is a string, but the label
    path assumes *every* part is. This names the parts that would break it, so the caller
    can refuse a half-converted selector such as ``slice("2024-01-01", 12)`` with a
    message instead of an ``AttributeError`` from inside the matcher. An open slice bound
    (``None``) is not a stray part — it means "the end of the axis".

    Args:
        selector: A ``sel`` selector — a scalar, a list, or a :class:`slice`.

    Returns:
        list: The non-string parts, in the order they appear; empty when every part is a
            label (or an open slice bound).

    Examples:
        - A selector that is entirely labels has no stray parts:
            ```python
            >>> from pyramids.netcdf._label_select import non_label_parts
            >>> non_label_parts(["2024-01-01", "2024-01-02"])
            []

            ```
        - A half-converted range names the part that does not belong:
            ```python
            >>> from pyramids.netcdf._label_select import non_label_parts
            >>> non_label_parts(slice("2024-01-01", 12))
            [12]

            ```
        - An open bound is not a stray part:
            ```python
            >>> from pyramids.netcdf._label_select import non_label_parts
            >>> non_label_parts(slice("2024-01-01", None))
            []

            ```
    """
    if isinstance(selector, slice):
        parts: tuple[Any, ...] = tuple(
            part for part in (selector.start, selector.stop) if part is not None
        )
    elif isinstance(selector, list):
        parts = tuple(selector)
    else:
        parts = (selector,)
    return [part for part in parts if not isinstance(part, str)]


def first_label(selector: Any) -> str | None:
    """The first string in a selector, or ``None`` when it carries none.

    Args:
        selector: A ``sel`` selector — a scalar, a list, or a :class:`slice`.

    Returns:
        str or None: The first string part, in the order the selector writes them.

    Examples:
        - A list answers with its first label:
            ```python
            >>> from pyramids.netcdf._label_select import first_label
            >>> first_label(["2024-01-02", "2024-01-03"])
            '2024-01-02'

            ```
        - A purely numeric selector has none:
            ```python
            >>> from pyramids.netcdf._label_select import first_label
            >>> first_label(slice(500, 1000)) is None
            True

            ```
    """
    if isinstance(selector, slice):
        parts: tuple[Any, ...] = (selector.start, selector.stop)
    elif isinstance(selector, list):
        parts = tuple(selector)
    else:
        parts = (selector,)
    return next((part for part in parts if isinstance(part, str)), None)


def probe_format(selector: Any) -> str | None:
    """The precision an axis must decode at to answer this label selector.

    Lets a caller test "does this axis decode at all" with the *same* format the match
    will use, so a memoised decoder answers both from one pass over the axis instead of
    decoding the whole coordinate variable again.

    Args:
        selector: A label selector — a label, a list of them, or a :class:`slice`.

    Returns:
        str or None: :data:`FULL_FORMAT` for a slice, whose bounds are compared at full
            precision; the format of the selector's first label otherwise; and ``None``
            when the selector carries no string, or its string is not a date-label shape
            at all (``"control"``, ``"850"``) — such a selector is a stored value, not a
            label, and belongs on the exact-match path.

    Examples:
        - A date-only label only needs the axis decoded to dates:
            ```python
            >>> from pyramids.netcdf._label_select import probe_format
            >>> probe_format("2024-01-01")
            '%Y-%m-%d'

            ```
        - A slice compares fully-qualified labels:
            ```python
            >>> from pyramids.netcdf._label_select import probe_format
            >>> probe_format(slice("2024-01", "2024-03"))
            '%Y-%m-%d %H:%M:%S'

            ```
        - A string that is not a date label at all is not a label selection:
            ```python
            >>> from pyramids.netcdf._label_select import probe_format
            >>> probe_format("control") is None
            True

            ```
    """
    label = first_label(selector)
    if isinstance(selector, slice) and label is not None:
        fmt: str | None = FULL_FORMAT
    elif label is None:
        fmt = None
    else:
        normalised = normalise_label(label)
        fmt = (
            _PRECISION_FORMATS.get(len(normalised))
            if _LABEL_PATTERN.match(normalised)
            else None
        )
    return fmt


def _label_slice_indices(
    decode: Callable[[str], list[str]], selector: slice
) -> list[int]:
    """Indices whose decoded label falls inside an inclusive label slice.

    Helper of :func:`label_indices`; bounds are padded to full precision so a partial
    one covers its whole period, and swapped when the axis runs newest-first.
    """
    labels = decode(FULL_FORMAT)
    if not labels:
        # An axis with nothing decoded has no label to fall inside the range. The engine
        # guards this before calling, but the primitive is shared and doctested on its
        # own, so it must not die in `min()` on an empty axis.
        return []
    start = None if selector.start is None else normalise_label(selector.start)
    stop = None if selector.stop is None else normalise_label(selector.stop)
    # Order the bounds *before* padding: which end a partial label extends to depends on
    # whether it is the low or the high bound, not on which slot it was written in. An
    # open bound falls back to the axis extreme rather than to its first/last value, so
    # the slice stays direction-agnostic on an axis written newest-first.
    if start is not None and stop is not None and start > stop:
        start, stop = stop, start
    low = min(labels) if start is None else pad_label(start, upper=False)
    high = max(labels) if stop is None else pad_label(stop, upper=True)
    return [i for i, label in enumerate(labels) if low <= label <= high]


def label_indices(decode: Callable[[str], list[str]], selector: Any) -> list[int]:
    """Resolve a date-label selector against a CF time axis.

    Each label is matched at its own precision: the axis is decoded with the format
    :func:`label_format` returns for that label, so ``"2024-01-01"`` matches every step
    of that day while ``"2024-01-01 06:00:00"`` matches one. A list unions its labels;
    a :class:`slice` compares fully-qualified labels with inclusive, period-aware bounds.

    Args:
        decode: Callable turning a strftime format into the axis' decoded labels, one
            per coordinate value (``nc._decode_time_labels`` bound to the dimension).
        selector: A label, a list of labels, or a :class:`slice` of labels.

    Returns:
        list[int]: Ascending indices of the matching coordinates; empty when none match.

    Raises:
        ValueError: A label is not one of the supported precisions.

    Examples:
        - A date-only label matches every step inside that day:
            ```python
            >>> from datetime import datetime, timedelta
            >>> from pyramids.netcdf._label_select import label_indices
            >>> steps = [
            ...     datetime(2024, 1, 1) + timedelta(hours=h) for h in (0, 6, 12, 18)
            ... ]
            >>> def decode(fmt):
            ...     return [step.strftime(fmt) for step in steps]
            >>> label_indices(decode, "2024-01-01")
            [0, 1, 2, 3]

            ```
        - A fully-qualified label matches exactly one:
            ```python
            >>> label_indices(decode, "2024-01-01 12:00:00")
            [2]

            ```
        - A slice takes an inclusive range:
            ```python
            >>> label_indices(decode, slice("2024-01-01 06:00", "2024-01-01 12:00"))
            [1, 2]

            ```
    """
    if isinstance(selector, slice):
        indices = _label_slice_indices(decode, selector)
    else:
        wanted = selector if isinstance(selector, list) else [selector]
        found: set[int] = set()
        for label in wanted:
            normalised = normalise_label(label)
            decoded = decode(label_format(normalised))
            found.update(i for i, value in enumerate(decoded) if value == normalised)
        indices = sorted(found)
    return indices


def _is_number(value: Any) -> bool:
    """Return whether ``value`` is a real number — ``bool`` excluded. Helper of ``nearest_indices``.

    Tests against :class:`numbers.Real` rather than ``(int, float)`` so a numpy scalar
    off a coordinate array counts: ``np.int64`` is registered as ``Integral`` but is not
    an ``int`` subclass.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def nearest_indices(coords: list, selector: Any) -> list[int]:
    """Snap a numeric selector to the closest coordinate(s) on an axis.

    A request exactly between two coordinates resolves to the **smaller** one, whichever
    end of the axis it is stored at, so the same physical request answers the same way on
    an ascending and a descending axis.

    Args:
        coords: The axis' stored coordinate values.
        selector: A number, or a list of numbers (each snapped independently).

    Returns:
        list[int]: Indices of the snapped coordinates, in **axis** order rather than
            request order, deduplicated when two requests snap to the same coordinate.
            Non-finite coordinates (a ``_FillValue`` in the axis) are never snapped to.

    Raises:
        ValueError: The selector is a :class:`slice` (a range has no nearest value), the
            selector is not a finite number, the axis is not numeric, or the axis holds
            no finite coordinate to snap to.

    Examples:
        - A value between two levels snaps to the closer one:
            ```python
            >>> from pyramids.netcdf._label_select import nearest_indices
            >>> nearest_indices([1000.0, 925.0, 850.0, 700.0], 900.0)
            [1]

            ```
        - Each value in a list snaps independently:
            ```python
            >>> from pyramids.netcdf._label_select import nearest_indices
            >>> nearest_indices([1000.0, 925.0, 850.0, 700.0], [990.0, 710.0])
            [0, 3]

            ```
        - A range has no nearest value, so a slice is refused:
            ```python
            >>> from pyramids.netcdf._label_select import nearest_indices
            >>> nearest_indices([1000.0, 925.0], slice(900, 1000))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: method='nearest' does not accept a slice selector...

            ```

    See Also:
        label_indices: the date-label counterpart, which this deliberately refuses.
    """
    if isinstance(selector, slice):
        raise ValueError(
            "method='nearest' does not accept a slice selector — a range has no nearest "
            "value. Drop `method` to take the range, or pass a value or a list of values."
        )
    wanted = selector if isinstance(selector, list) else [selector]
    if not all(_is_number(value) for value in wanted):
        raise ValueError(
            f"method='nearest' needs numeric selector values, got {selector!r}."
        )
    if not all(math.isfinite(value) for value in wanted):
        raise ValueError(
            f"method='nearest' needs finite selector values, got {selector!r}."
        )
    if not all(_is_number(coord) for coord in coords):
        raise ValueError(
            f"method='nearest' needs a numeric coordinate axis, got {coords!r}."
        )
    # A `_FillValue` in a coordinate axis arrives as NaN, which compares false against
    # everything — so a plain `min` over the distances would hand back whichever slot it
    # was seeded with, silently snapping to the fill value's plane. Scan only the real
    # coordinates, and say so when there are none.
    candidates = [
        (position, coord)
        for position, coord in enumerate(coords)
        if math.isfinite(coord)
    ]
    if not candidates:
        raise ValueError(
            f"method='nearest' found no finite coordinate to snap to on this axis: {coords!r}."
        )
    found: set[int] = set()
    for value in wanted:
        # Rank by (distance, coordinate) rather than by position, so a request that falls
        # exactly between two coordinates resolves to the same one whether the file
        # stores the axis ascending or descending — the direction-agnostic rule `sel`'s
        # slice path already advertises. The smaller coordinate wins a tie.
        _, _, position = min(
            (abs(coord - value), coord, index) for index, coord in candidates
        )
        found.add(position)
    return sorted(found)
