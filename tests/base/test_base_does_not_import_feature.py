"""`base` must not reach up into `feature`.

`base/_coverage.py` imported `pyramids.feature.bbox` for a bbox reprojection,
which inverted the package's own layering -- and `pyramids.feature` pulls in
geopandas, so importing a `base` module alone loaded it. `base` modules are the
ones most likely to be imported early and in isolation.

The function moved down to `base/_bbox.py`, and `feature.bbox` re-exports it so
the name callers already use still resolves.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.core


def _imports_after(module: str) -> set[str]:
    """Module names present in `sys.modules` after importing `module` alone."""
    code = (
        "import sys, json;"
        f" import {module};"
        " print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    import json

    return set(json.loads(completed.stdout.splitlines()[-1]))


class TestTheCoverageReaderStaysInBase:
    """The layering, asserted in a fresh interpreter."""

    def test_importing_it_does_not_pull_in_feature(self):
        """A `base` module reaching up into `feature` is the inversion.

        Test scenario:
            Checked in a subprocess because the test session has already
            imported most of the package; only a fresh interpreter shows what
            one module actually costs.
        """
        loaded = _imports_after("pyramids.base._coverage")

        assert "pyramids.feature" not in loaded

    def test_importing_it_does_not_pull_in_geopandas(self):
        """The measurable cost of that inversion.

        Test scenario:
            `pyramids.feature` imports geopandas, which is the slowest import
            in the dependency set. A `base` module should not pay for it.
        """
        loaded = _imports_after("pyramids.base._coverage")

        assert "geopandas" not in loaded


class TestTheReExportStillResolves:
    """Moving it must not break the name callers already use."""

    def test_feature_bbox_transform_is_the_base_one(self):
        """Re-exported, not reimplemented, so there is still one function."""
        from pyramids.base._bbox import transform as base_transform
        from pyramids.feature.bbox import transform as feature_transform

        assert feature_transform is base_transform

    def test_it_still_reprojects(self):
        """The move is a relocation, not a rewrite.

        Test scenario:
            A same-CRS transform is the identity, and a real reprojection still
            produces the Web Mercator easting it did before.
        """
        from pyramids.feature.bbox import transform

        assert [round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)] == [
            -10.0,
            -5.0,
            10.0,
            5.0,
        ]
        assert round(transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)[2]) == 1113195
