"""The `pyramids.stac` package surface matches the tour its docstring gives.

`pyramids.stac.__doc__` is the structured tour of the subpackage — "Search and
discovery", "Reading assets", "Item metadata" — and is what a reader meets
first, in the API docs and in `help(pyramids.stac)`. A name exported without a
line there is undiscoverable except by reading `__all__`, which is how sixteen
of them accumulated at once.
"""

from __future__ import annotations

import pytest

import pyramids.stac as stac_package

pytestmark = pytest.mark.core


class TestExportedNamesAreDocumented:
    """Every exported name appears in the module docstring, and vice versa.

    Test scenario:
        `__all__` and the docstring are maintained by hand in the same file, so
        they drift in both directions: a new export that nobody wrote a line
        for, and a line left behind by a name that was renamed or removed.
    """

    def test_every_exported_name_is_in_the_docstring(self):
        """No public name is missing from the tour."""
        doc = stac_package.__doc__ or ""
        missing = sorted(name for name in stac_package.__all__ if name not in doc)
        assert missing == [], (
            f"{len(missing)} exported name(s) are absent from the "
            f"pyramids.stac module docstring: {missing}"
        )

    def test_every_exported_name_is_importable(self):
        """`__all__` names nothing the package does not actually expose."""
        absent = sorted(
            name for name in stac_package.__all__ if not hasattr(stac_package, name)
        )
        assert absent == [], f"__all__ names unexported attribute(s): {absent}"

    def test_the_docstring_references_no_unexported_name(self):
        """A `:func:`/`:class:` reference in the tour resolves to an export.

        Test scenario:
            A renamed export leaves a stale cross-reference behind, which
            renders as a broken link in the docs rather than failing anywhere.
        """
        doc = stac_package.__doc__ or ""
        referenced = {
            fragment.split("`", 1)[0]
            for marker in (":func:`", ":class:`", ":meth:`")
            for fragment in doc.split(marker)[1:]
        }
        local = {name for name in referenced if "." not in name}
        unexported = sorted(local - set(stac_package.__all__))
        assert unexported == [], (
            f"the docstring cross-references name(s) the package does not "
            f"export: {unexported}"
        )
