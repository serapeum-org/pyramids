"""One composer for the optional-extra install hints.

Every guard for an optional dependency raised the same two-line install block
with only the extra's name changing, and each had written it out. `extra_hint`
composes it; `lazy_extra_hint` is now its named case for the `[lazy]` extra, so
the dozen call sites that use that wording are unaffected.
"""

from __future__ import annotations

import builtins

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import extra_hint, import_xarray, lazy_extra_hint

pytestmark = pytest.mark.core


class TestExtraHint:
    """The lead is the caller's; the install commands are not."""

    def test_the_lead_sentence_is_preserved_verbatim(self):
        """The lead says which operation needed the dependency."""
        lead = "Zarr IO requires the optional 'zarr' dependency."

        assert extra_hint(lead, "lazy").startswith(lead)

    @pytest.mark.parametrize("extra", ["lazy", "stac", "parquet", "viz"])
    def test_both_commands_name_the_extra(self, extra: str):
        """PyPI and conda-forge instructions both track the extra."""
        hint = extra_hint("Needs something.", extra)

        assert f"pyramids-gis[{extra}]" in hint
        assert f"conda-forge pyramids-{extra}" in hint

    def test_the_header_line_is_stable(self):
        """Call sites and their tests match on this wording."""
        assert extra_hint("X.", "lazy").splitlines()[0] == "X. Install with one of:"


class TestLazyExtraHintUnchanged:
    """The existing helper keeps its exact output."""

    def test_it_is_the_lazy_case_of_extra_hint(self):
        """Delegating changed nothing for the dozen `[lazy]` call sites."""
        lead = "Op needs the optional 'dask' dependency."

        assert lazy_extra_hint(lead) == extra_hint(lead, "lazy")

    def test_the_pinned_substrings_survive(self):
        """Both commands its own tests assert on are still produced."""
        hint = lazy_extra_hint("X requires the optional 'zarr' dependency.")

        assert "pip install 'pyramids-gis[lazy]'" in hint
        assert "conda install -c conda-forge pyramids-lazy" in hint


# Needs the [interop] extra: `to_xarray` / `from_xarray` import xarray, which a
# bare install does not have. The module-level `core` mark still selects these
# under `-m core`, and `tests/conftest.py` turns this marker into the matching
# skip when xarray is absent -- which is what the pure-wheel job needs.
@pytest.mark.interop
class TestImportXarray:
    """The two xarray guards share one importer.

    Both call sites need the live module back -- one to build a `Dataset`, the
    other to type-check the argument it was handed -- so this importer returns
    it rather than only asserting the import worked.
    """

    def test_it_returns_the_module_itself(self):
        """A guard that returned None would force a second import.

        Test scenario:
            With xarray installed, the call yields the module, and the module
            is usable: constructing a `DataArray` from it works.
        """
        xr = import_xarray("unused: xarray is installed in this environment")

        assert xr.__name__ == "xarray"
        assert xr.DataArray([1, 2, 3]).shape == (3,)

    def test_calling_it_twice_returns_the_same_module(self):
        """Guards run per call; re-importing must not build a new module.

        Test scenario:
            Python caches modules in `sys.modules`, so two guarded imports have
            to yield the identical object -- otherwise an `isinstance` check
            against a type from one would fail for a value made by the other.
        """
        first = import_xarray("unused")
        second = import_xarray("unused")

        assert first is second

    def test_a_missing_xarray_raises_the_hint_it_was_given(self, monkeypatch):
        """The message is the caller's, so the guard must not rewrite it.

        Args:
            monkeypatch: Fixture used to make the import fail.

        Test scenario:
            `builtins.__import__` is patched to reject xarray, simulating an
            environment without the package. The raised error must carry the
            caller's own install hint verbatim -- that hint names the operation
            that needed xarray, which a generic message would lose.
        """
        hint = "to_xarray() needs xarray. Install with: pip install xarray"
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("No module named 'xarray'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

        with pytest.raises(OptionalPackageDoesNotExist) as exc_info:
            import_xarray(hint)

        assert hint in str(exc_info.value), f"hint not carried: {exc_info.value}"
