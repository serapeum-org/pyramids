"""A lower package must not reach up into a higher one.

`base/_coverage.py` imported `pyramids.feature.bbox` for a bbox reprojection,
which inverted the package's own layering -- and `pyramids.feature` pulls in
geopandas, so importing a `base` module alone loaded it. `base` modules are the
ones most likely to be imported early and in isolation.

The function moved down to `base/_bbox.py`, and `feature.bbox` re-exports it so
the name callers already use still resolves.

`dataset/_cube_time.py` had the same shape one layer up: it imported the CF
epoch constants from `pyramids.netcdf.utils`, while `pyramids.netcdf.netcdf`
imports `pyramids.dataset`. That resolved only because nothing imported
`_cube_time` at module level -- one edit away from a real cycle. Same remedy:
the constants moved down to `base/_cf_epoch.py`, and `netcdf.utils` re-exports
them.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.core


def _imports_after(module: str) -> set[str]:
    """Module names present in `sys.modules` after importing `module` alone."""
    code = f"import sys, json; import {module}; print(json.dumps(sorted(sys.modules)))"
    # The parent's import path is passed through explicitly. Without it the
    # child resolves `pyramids` from the editable install, which points at the
    # repo root -- so in a worktree this measured a different tree entirely and
    # the guard proved nothing about the code under test.
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(entry for entry in sys.path if entry),
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
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

        assert [
            round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)
        ] == [
            -10.0,
            -5.0,
            10.0,
            5.0,
        ]
        assert round(transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)[2]) == 1113195


class TestTheCollectionTimeAxisStaysBelowNetcdf:
    """`dataset` must not reach up into `netcdf` either."""

    def test_importing_the_time_axis_writer_does_not_pull_in_netcdf(self):
        """The inversion, one layer up from the `base` case above.

        Test scenario:
            `pyramids.netcdf.netcdf` imports `pyramids.dataset`, so a
            `dataset` module importing `pyramids.netcdf` closes the loop. It
            held together only because `collection.py` does not import
            `_cube_time` at module level, which is not a property anyone
            declared or would think to preserve.
        """
        loaded = _imports_after("pyramids.dataset._cube_time")

        assert "pyramids.netcdf" not in loaded

    def test_the_shared_constants_are_importable_from_base_alone(self):
        """What they were moved into has to stand on its own.

        Test scenario:
            The point of moving them is that they carry no netCDF dependency.
            If importing them from `base` still loaded `netcdf`, the move
            would have relocated the problem rather than removed it.
        """
        loaded = _imports_after("pyramids.base._cf_epoch")

        assert "pyramids.netcdf" not in loaded
        assert "pyramids.dataset" not in loaded

    def test_the_re_export_still_resolves(self):
        """Moving them must not break the name callers already use.

        Test scenario:
            `netcdf.utils` is where `decode_cf_time` lives, so it is the
            natural place to import the epoch from. Re-exported, not
            redefined -- two definitions of the epoch is the failure the
            sharing exists to prevent.
        """
        from pyramids.base._cf_epoch import CF_EPOCH, cf_epoch_units
        from pyramids.netcdf.utils import CF_EPOCH as reexported_epoch
        from pyramids.netcdf.utils import cf_epoch_units as reexported_units

        assert reexported_units is cf_epoch_units
        assert reexported_epoch is CF_EPOCH

    def test_both_writers_count_from_the_same_epoch(self):
        """The property that made sharing them worth doing.

        Test scenario:
            The collection axis counts in nanoseconds and the interop path in
            fractional seconds -- deliberately different resolutions. What
            they must not differ on is the epoch, because a writer that
            drifted to another one would produce a file the reader silently
            decodes to the wrong dates.
        """
        from pyramids.base._cf_epoch import CF_EPOCH, cf_epoch_units

        assert cf_epoch_units("nanoseconds") == f"nanoseconds since {CF_EPOCH}"
        assert cf_epoch_units("seconds") == f"seconds since {CF_EPOCH}"

    def test_the_seconds_axis_decodes_back_through_the_reader(self):
        """One resolution round-trips end to end; the other cannot.

        Test scenario:
            `decode_cf_time` goes through `cftime`, which accepts `seconds`
            but not `nanoseconds` -- so the interop path's axis reads back
            here and the collection's does not. That asymmetry predates this
            move and is not changed by it; the test pins the half that does
            work, so relocating the constants cannot break it unnoticed.
        """
        import numpy as np

        from pyramids.base._cf_epoch import cf_epoch_units
        from pyramids.netcdf.utils import decode_cf_time

        decoded = decode_cf_time(np.array([86400.0]), cf_epoch_units("seconds"))

        assert "1970-01-02" in str(decoded[0])
