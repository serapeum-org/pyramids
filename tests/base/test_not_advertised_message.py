"""One refusal for a name the service does not advertise.

The WCS reader, the WMTS reader and the OGC API Coverages reader each wrote
this message out, with the same wording, the same ten-name preview and the same
ellipsis. Two of them sorted the advertised names and the third listed them in
whatever order the capabilities document used -- so the same mistake against a
WMTS endpoint showed an arbitrary ten names where a WCS endpoint showed the
first ten alphabetically.

The vector readers already had a shared helper for the same refusal --
`pyramids.feature._ogc.require_advertised`, used by WFS and OGC API Features --
so "one refusal" only holds while the two share a message and a preview cap.
`require_advertised` decides *whether* to refuse; `not_advertised` decides what
the refusal says.
"""

from __future__ import annotations

import pytest

from pyramids.base import _ogc_api
from pyramids.base._ogc_api import not_advertised
from pyramids.feature._ogc import require_advertised

pytestmark = pytest.mark.core


class TestTheMessage:
    """What the caller is told."""

    def test_it_names_the_request_and_the_endpoint(self):
        """Both halves of "what did I ask for, and of whom".

        Test scenario:
            The name is usually a typo or a stale identifier, and the endpoint
            is which of several services rejected it.
        """
        error = not_advertised("coverage", "dem", "http://x/wcs", ["a", "b"])

        assert "coverage 'dem'" in str(error)
        assert "'http://x/wcs'" in str(error)

    def test_the_kind_is_pluralised_for_the_list(self):
        """A WMTS refusal reads about layers, a WCS one about coverages.

        Test scenario:
            The services call the thing different names, and the message has
            to use the caller's vocabulary rather than a shared abstraction's.
        """
        assert "Available layers:" in str(not_advertised("layer", "z", "u", ["a"]))
        assert "Available coverages:" in str(
            not_advertised("coverage", "z", "u", ["a"])
        )

    def test_the_names_are_sorted_whatever_order_they_arrive_in(self):
        """The regression: one caller listed them in server order.

        Test scenario:
            A capabilities document has no meaningful order, so listing the
            first ten as they appear shows an arbitrary subset. Sorting makes
            the preview the same list every time and easy to scan.
        """
        error = not_advertised("coverage", "z", "u", ["c", "a", "b"])

        assert "['a', 'b', 'c']" in str(error)

    def test_a_long_list_is_trimmed_and_marked(self):
        """A service can advertise hundreds; a traceback should stay readable.

        Test scenario:
            The preview shows the shape of what is there. The ellipsis is what
            tells the caller the list did not end where it appears to.
        """
        error = not_advertised("layer", "z", "u", [f"{i:02d}" for i in range(30)])

        assert str(error).endswith("…")
        assert "'09'" in str(error)
        assert "'10'" not in str(error)

    def test_a_short_list_is_not_marked(self):
        """The ellipsis must mean something when it appears.

        Test scenario:
            Ten or fewer names are shown in full, so an ellipsis there would
            claim there is more when there is not.
        """
        error = not_advertised("layer", "z", "u", ["a", "b"])

        assert not str(error).endswith("…")

    def test_it_is_returned_rather_than_raised(self):
        """So a caller can chain it onto the service error it replaces.

        Test scenario:
            The WMTS reader catches a `WMSError` and re-raises this `from` it.
            A helper that raised internally could not be used that way.
        """
        result = not_advertised("layer", "z", "u", ["a"])

        assert isinstance(result, ValueError)


class TestTheVectorHelperRaisesTheSameRefusal:
    """`require_advertised` is the vector readers' half of the same check."""

    def test_the_two_produce_the_same_sentence(self):
        """A WFS typo and a WCS typo read alike.

        Test scenario:
            Both helpers name the thing, quote the endpoint and list what is
            advertised. Written out twice they agreed only until one of them
            was edited.
        """
        with pytest.raises(ValueError) as raised:
            require_advertised(
                "c1", frozenset({"a", "b"}), noun="collection", endpoint="http://x"
            )
        expected = not_advertised("collection", "c1", "http://x", ["a", "b"])

        assert str(raised.value) == str(expected)

    def test_one_preview_cap_governs_both(self):
        """The cap is a constant, not a literal repeated per helper.

        Test scenario:
            Two copies of `[:10]` drift the moment either is retuned. Moving
            the cap must move the vector refusal too, which is only true while
            the message has a single implementation.
        """
        advertised = frozenset(f"{i:02d}" for i in range(30))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_ogc_api, "_ADVERTISED_PREVIEW", 3)
            with pytest.raises(ValueError) as raised:
                require_advertised(
                    "zz", advertised, noun="feature type", endpoint="http://x"
                )

        assert "['00', '01', '02']" in str(raised.value)

    def test_an_empty_advertised_set_still_skips_the_check(self):
        """A discovery document that enumerates nothing is not evidence.

        Test scenario:
            Some services advertise no names at all; refusing every read
            against them would break working callers. The driver is left to
            fail if the name is truly unknown.
        """
        assert (
            require_advertised("anything", frozenset(), noun="collection", endpoint="u")
            is None
        )

    def test_an_advertised_name_passes(self):
        """The check only fires on a name the document does not list."""
        assert (
            require_advertised(
                "a", frozenset({"a", "b"}), noun="collection", endpoint="u"
            )
            is None
        )
