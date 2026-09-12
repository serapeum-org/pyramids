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
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any

FULL_FORMAT = "%Y-%m-%d %H:%M:%S"
"""strftime format of a fully-qualified date label — the widest precision supported."""

_LOWER_TEMPLATE = "0001-01-01 00:00:00"
_UPPER_TEMPLATE = "9999-12-31 23:59:59"

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
    """
    label = normalise_label(text)
    fmt = _PRECISION_FORMATS.get(len(label))
    if fmt is None:
        supported = ", ".join(sorted(_PRECISION_FORMATS.values(), key=len))
        raise ValueError(
            f"{text!r} is not a supported date label. Supported precisions: {supported}."
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
    label_format(label)
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
    """
    if isinstance(selector, slice):
        parts: tuple[Any, ...] = (selector.start, selector.stop)
    elif isinstance(selector, list):
        parts = tuple(selector)
    else:
        parts = (selector,)
    return any(isinstance(part, str) for part in parts)


def _label_slice_indices(
    decode: Callable[[str], list[str]], selector: slice
) -> list[int]:
    """Indices whose decoded label falls inside an inclusive label slice.

    Helper of :func:`label_indices`; bounds are padded to full precision so a partial
    one covers its whole period, and swapped when the axis runs newest-first.
    """
    labels = decode(FULL_FORMAT)
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

    Args:
        coords: The axis' stored coordinate values.
        selector: A number, or a list of numbers (each snapped independently).

    Returns:
        list[int]: Ascending indices of the snapped coordinates — one per requested
            value, deduplicated when two requests snap to the same coordinate.

    Raises:
        ValueError: The selector is a :class:`slice` (a range has no nearest value), or
            either the selector or the axis is not numeric.

    Examples:
        - A value between two levels snaps to the closer one:
            ```python
            >>> from pyramids.netcdf._label_select import nearest_indices
            >>> nearest_indices([1000.0, 925.0, 850.0, 700.0], 900.0)
            [1]

            ```
        - Each value in a list snaps independently:
            ```python
            >>> nearest_indices([1000.0, 925.0, 850.0, 700.0], [990.0, 710.0])
            [0, 3]

            ```
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
    if not all(_is_number(value) for value in coords):
        raise ValueError(
            f"method='nearest' needs a numeric coordinate axis, got {coords!r}."
        )
    found: set[int] = set()
    for value in wanted:
        distances = [abs(coord - value) for coord in coords]
        found.add(distances.index(min(distances)))
    return sorted(found)
