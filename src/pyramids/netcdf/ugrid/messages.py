"""Message text shared by the two halves of the UgridDataset API.

`UgridDataset` resolves a variable name in two places — the facade mixin's `_facade_variable`
and the dataset's own `_require_variable_name` — and they must say the same thing about the
same mistake. Keeping the wording here is what makes that true by construction rather than by
two copies happening to agree.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["unknown_variable_message"]


def unknown_variable_message(
    operation: str, variable_name: str, available: Sequence[str]
) -> str:
    """The message for a ``variable_name`` the dataset does not have.

    Args:
        operation: The calling method's name.
        variable_name: The name that was not found.
        available: The names the dataset does hold.

    Returns:
        str: A message naming the operation, the bad name and the available ones.

    Examples:
        - A typo, with the real names listed:
            ```python
            >>> unknown_variable_message("polyfit", "dpeth", ["depth", "velocity"])
            "polyfit: 'dpeth' is not a data variable of this dataset. Available: ['depth', 'velocity']."

            ```
        - An empty dataset says so rather than printing an empty list on its own:
            ```python
            >>> unknown_variable_message("slope", "z", [])
            "slope: 'z' is not a data variable of this dataset. It holds no data variables."

            ```
    """
    if not available:
        return (
            f"{operation}: {variable_name!r} is not a data variable of this dataset. "
            f"It holds no data variables."
        )
    return (
        f"{operation}: {variable_name!r} is not a data variable of this dataset. "
        f"Available: {sorted(available)}."
    )
