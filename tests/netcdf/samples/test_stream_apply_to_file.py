"""Streaming-to-file variant of the container spatial ops (issue #976).

``crop`` / ``to_crs`` / ``resample`` on a root MDIM container accept a ``path=`` keyword. With it set,
the op streams every transformed variable straight to that ``.nc`` file one leading-dimension slab at a
time -- instead of materialising the whole reprojected/cropped cube in an in-memory MEM container -- and
returns a **file-backed** :class:`NetCDF`. ``path=None`` keeps the historical in-memory fan-out.

These tests assert the streamed result is value-identical to the in-memory result, that the 3-D write
really is slab-by-slab (not a single whole-array write), and that a shape the slab writer cannot represent
(a variable with two band dimensions) still round-trips through the eager fallback.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import NetCDF
from pyramids.netcdf.engines import interop
from pyramids.netcdf.netcdf import NetCDF as _NetCDF
from tests.netcdf.samples.conftest import RHUM, TOS

pytestmark = pytest.mark.core


def _snapshot(nc):
    """Materialise every variable's array, keyed by name, for a value comparison."""
    return {
        name: np.asarray(nc.get_variable(name).read_array()) for name in nc.variables
    }


def _assert_same(mem, streamed):
    """Assert two containers hold the same variables with value-identical arrays."""
    assert sorted(mem.variables) == sorted(streamed.variables)
    left, right = _snapshot(mem), _snapshot(streamed)
    for name in left:
        assert left[name].shape == right[name].shape, name
        if np.issubdtype(left[name].dtype, np.floating):
            assert_allclose(left[name], right[name], equal_nan=True, err_msg=name)
        else:
            assert_array_equal(left[name], right[name], err_msg=name)


OPS = {
    "crop": lambda nc, path: nc.crop(bbox=(0, -40, 60, 40), path=path),
    "to_crs": lambda nc, path: nc.to_crs(3857, path=path),
    "resample": lambda nc, path: nc.resample(nc.cell_size * 2, path=path),
}


@pytest.mark.parametrize("op", list(OPS))
def test_streamed_matches_in_memory(sample, tmp_path, op):
    """``op(path=…)`` writes a file-backed container value-identical to the in-memory result."""
    call = OPS[op]
    mem = call(NetCDF.read_file(sample(TOS)), None)
    out = tmp_path / f"{op}.nc"
    streamed = call(NetCDF.read_file(sample(TOS)), str(out))
    try:
        assert out.exists()
        assert streamed._raster is not None
        _assert_same(mem, streamed)
    finally:
        mem.close()
        streamed.close()


def test_three_d_variable_is_written_slab_by_slab(sample, tmp_path, monkeypatch):
    """The 3-D variable is filled one band slab at a time, not in a single whole-array write."""
    slab_calls = []
    original = interop._StreamingMultidimWriter.write_slab

    def counting(self, var_name, index, block):
        slab_calls.append((var_name, index))
        return original(self, var_name, index, block)

    monkeypatch.setattr(interop._StreamingMultidimWriter, "write_slab", counting)

    out = tmp_path / "to_crs.nc"
    streamed = NetCDF.read_file(sample(TOS)).to_crs(3857, path=str(out))
    try:
        bands = streamed.get_variable("tos").shape[0]
        tos_slabs = [index for name, index in slab_calls if name == "tos"]
        # One write_slab per band of the only 3-D variable, in order 0..bands-1.
        assert tos_slabs == list(range(bands))
    finally:
        streamed.close()


def test_two_band_dim_variable_falls_back_to_eager_write(sample, tmp_path, monkeypatch):
    """A variable with two band dims isn't slab-streamable: stream returns None, eager write matches."""
    returned = {}
    real = _NetCDF._stream_apply_to_file

    def spy(self, operation, op_kwargs, spatial_vars, aux_vars, path):
        result = real(self, operation, op_kwargs, spatial_vars, aux_vars, path)
        returned["is_none"] = result is None
        return result

    monkeypatch.setattr(_NetCDF, "_stream_apply_to_file", spy)

    # `resample` avoids any reprojection (RHUM spans past Web Mercator's ±85° limit); the point is
    # the 4-D (time, level) variable, which the slab writer cannot represent.
    cell = NetCDF.read_file(sample(RHUM)).cell_size
    mem = NetCDF.read_file(sample(RHUM)).resample(cell * 2)
    out = tmp_path / "rhum.nc"
    streamed = NetCDF.read_file(sample(RHUM)).resample(cell * 2, path=str(out))
    try:
        assert returned["is_none"] is True  # 4-D (time, level) var rejects the slab writer
        assert out.exists()
        _assert_same(mem, streamed)
    finally:
        mem.close()
        streamed.close()


def test_subset_crop_honours_path(sample, tmp_path):
    """``path=`` on a single-variable (non-container) crop writes and re-opens a file-backed result.

    A single-variable crop is a plain raster, so writing it out yields a generic multi-band raster
    (``Band1..N``); the point here is that ``path=`` is honoured and the pixel data round-trips.
    """
    in_memory = (
        NetCDF.read_file(sample(TOS)).get_variable("tos").crop(bbox=(0, -40, 60, 40))
    )
    out = tmp_path / "tos_subset.nc"
    on_disk = NetCDF.read_file(sample(TOS)).get_variable("tos").crop(
        bbox=(0, -40, 60, 40), path=str(out)
    )
    try:
        assert out.exists()
        bands = in_memory.shape[0]
        # The raster export splits the 24-band variable into Band1..Band24 — stack them back.
        stacked = np.stack(
            [
                np.asarray(on_disk.get_variable(f"Band{i + 1}").read_array())
                for i in range(bands)
            ]
        )
        assert_allclose(stacked, np.asarray(in_memory.read_array()), equal_nan=True)
    finally:
        in_memory.close()
        on_disk.close()
