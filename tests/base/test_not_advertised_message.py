"""One refusal for a name the service does not advertise.

The WCS reader, the WMTS reader and the OGC API Coverages reader each wrote
this message out, with the same wording, the same ten-name preview and the same
ellipsis. Two of them sorted the advertised names and the third listed them in
whatever order the capabilities document used -- so the same mistake against a
WMTS endpoint showed an arbitrary ten names where a WCS endpoint showed the
first ten alphabetically.
"""

from __future__ import annotations

import pytest

from pyramids.base._ogc_api import not_advertised

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
