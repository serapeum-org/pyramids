"""One composer for the optional-extra install hints.

Every guard for an optional dependency raised the same two-line install block
with only the extra's name changing, and each had written it out. `extra_hint`
composes it; `lazy_extra_hint` is now its named case for the `[lazy]` extra, so
the dozen call sites that use that wording are unaffected.
"""

from __future__ import annotations

import pytest

from pyramids.base._utils import extra_hint, lazy_extra_hint

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
