"""The lazy `orthorectify` result frees its staged DEM from the registry too.

`mint_vsimem` registers every path it mints so a process-exit sweep can reclaim
anything a call site failed to unlink. That makes unlinking only half of
reclaiming: a site that frees its own `/vsimem` file and says nothing leaves the
path string in `_VSIMEM_PATHS` forever, pointing at a file that is gone, and the
exit sweep walks it.

`orthorectify` stages an in-memory DEM into `/vsimem` and has three release
paths -- a failed warp, a materialised result, and a lazy (VRT) result, which
cannot free the DEM until the GDAL handle reading it is collected. The first two
are covered in `test_georef.py`. This is the third: the `weakref.finalize` the
lazy path arms has to name the release that also unregisters, not a bare unlink,
and nothing asserted that. A tile service handing out lazy orthorectified views
would otherwise grow the registry by one dead string per view, for the life of
the process.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from pyramids.base import _artifacts
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines import georef as georef_module

pytestmark = pytest.mark.core


def _rpc_coeff(term_index: int) -> str:
    """A 20-term RPC coefficient string with a single 1 at `term_index`.

    Args:
        term_index: Which polynomial term carries the 1.

    Returns:
        str: The coefficient list, space separated.
    """
    coeffs = ["0"] * 20
    coeffs[term_index] = "1"
    return " ".join(coeffs)


# A near-identity RPC. The warp is stubbed out in every test here, so these
# coefficients are never evaluated -- they exist because `orthorectify` refuses a
# dataset with no RPC domain before it stages anything.
_RPC_SAMPLE: dict[str, str] = {
    "HEIGHT_OFF": "100",
    "HEIGHT_SCALE": "50",
    "LAT_OFF": "49.5",
    "LAT_SCALE": "0.5",
    "LONG_OFF": "10.5",
    "LONG_SCALE": "0.5",
    "LINE_OFF": "4",
    "LINE_SCALE": "4",
    "SAMP_OFF": "4",
    "SAMP_SCALE": "4",
    "SAMP_NUM_COEFF": _rpc_coeff(1),
    "SAMP_DEN_COEFF": _rpc_coeff(0),
    "LINE_NUM_COEFF": _rpc_coeff(2),
    "LINE_DEN_COEFF": _rpc_coeff(0),
}


def _small_raster(size: int = 4) -> Dataset:
    """A square in-memory float32 raster.

    Args:
        size: Rows and columns.

    Returns:
        Dataset: The raster, georeferenced north-up at one unit per pixel.
    """
    return Dataset.from_array(
        np.zeros((size, size), "float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, float(size)), cell_size=1.0),
    )


@pytest.fixture
def rpc_dataset() -> Dataset:
    """An 8x8 raster carrying the near-identity RPC sensor model.

    Returns:
        Dataset: The raster `orthorectify` is called on.
    """
    dataset = _small_raster(8)
    dataset.raster.SetMetadata(_RPC_SAMPLE, "RPC")
    return dataset


@pytest.fixture
def mem_dem() -> Dataset:
    """A MEM-backed DEM, which `orthorectify` has to stage to `/vsimem`.

    Returns:
        Dataset: A DEM with no path of its own, so staging is unavoidable.
    """
    return Dataset.from_array(
        np.full((8, 8), 100.0, "float32"),
        geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0),
    )


@pytest.fixture
def stub_warp(monkeypatch):
    """Replace the warp with one that returns a fresh materialised raster.

    A real RPC warp against a DEM is GDAL-version-fragile with synthetic
    coefficients, and what is under test is the cleanup around the warp.

    Args:
        monkeypatch: pytest fixture, used to swap the module-level callable.
    """
    monkeypatch.setattr(
        georef_module,
        "warp_to_dataset",
        lambda *args, **kwargs: _small_raster(),
    )


class TestTheLazyResultReleasesItsRegistryEntry:
    """The fourth `mint_vsimem` release site, and the one nothing pinned."""

    def test_the_finalizer_drops_the_entry_when_the_handle_dies(
        self, rpc_dataset, mem_dem, stub_warp
    ):
        """The lazy path's release has to unregister, not only unlink.

        Args:
            rpc_dataset: The RPC-carrying raster fixture.
            mem_dem: The MEM-backed DEM fixture.
            stub_warp: Fixture replacing the warp with a materialised raster.

        Test scenario:
            A lazy result reads the staged DEM on every access, so it can only
            be freed once the GDAL handle is collected -- which is why the
            release is armed as a `weakref.finalize` rather than run inline.
            Arming that finalizer with a bare unlink reclaims the file and
            leaves the registry entry behind, which is invisible until the
            registry is read: one dead string per lazy view, for the life of the
            process.
        """
        before = len(_artifacts._VSIMEM_PATHS)

        view = rpc_dataset.orthorectify(dem=mem_dem, lazy=True)

        assert len(_artifacts._VSIMEM_PATHS) == before + 1, (
            "the staged DEM should be registered while the view is alive, "
            f"registry grew by {len(_artifacts._VSIMEM_PATHS) - before}"
        )
        handle = view.raster
        del view
        gc.collect()
        assert len(_artifacts._VSIMEM_PATHS) == before + 1, (
            "the DEM must stay registered while the GDAL handle can still read it"
        )
        del handle
        gc.collect()

        assert len(_artifacts._VSIMEM_PATHS) == before, (
            f"entries left behind: {_artifacts._VSIMEM_PATHS[before:]}"
        )

    def test_repeated_lazy_views_do_not_accumulate_entries(
        self, rpc_dataset, mem_dem, stub_warp
    ):
        """The growth is per call, which is what makes it unbounded.

        Args:
            rpc_dataset: The RPC-carrying raster fixture.
            mem_dem: The MEM-backed DEM fixture.
            stub_warp: Fixture replacing the warp with a materialised raster.

        Test scenario:
            One leaked string is invisible; a service building a lazy
            orthorectified view per request accumulates one per request, and the
            exit sweep then iterates the whole list. Five calls whose results are
            all dropped must leave the registry exactly as they found it.
        """
        before = len(_artifacts._VSIMEM_PATHS)

        for _ in range(5):
            rpc_dataset.orthorectify(dem=mem_dem, lazy=True)
        gc.collect()

        assert len(_artifacts._VSIMEM_PATHS) == before, (
            f"entries left behind: {_artifacts._VSIMEM_PATHS[before:]}"
        )
