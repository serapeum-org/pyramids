"""File round-trip invariants: ``to_file`` -> ``read_file`` preserves structure and convention.

Includes the convention-preservation regression for issue #583 (``to_file`` was stamping
``Conventions="CF-1.6"`` onto files that declared no convention).
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_roundtrip_preserves_variable_set(sample_name, sample, tmp_path):
    """Writing then re-reading preserves the variable count and the rank histogram."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        before = nc.get_all_metadata()
        before_count = len(before.variables)
        before_ranks = sorted(len(v.shape) for v in before.variables.values())
    finally:
        out = str(tmp_path / "roundtrip.nc")
        nc.to_file(out)
        nc.close()

    reopened = NetCDF.read_file(out)
    try:
        after = reopened.get_all_metadata()
        assert (
            len(after.variables) == before_count
        ), f"{sample_name}: variable count {len(after.variables)} != {before_count} after round-trip"
        assert (
            sorted(len(v.shape) for v in after.variables.values()) == before_ranks
        ), f"{sample_name}: rank histogram changed after round-trip"
    finally:
        reopened.close()


def test_roundtrip_preserves_convention(sample_name, sample, tmp_path):
    """``to_file`` keeps the source's ``Conventions`` verbatim — including *no* convention (issue #583)."""
    nc = NetCDF.read_file(sample(sample_name))
    source_conventions = nc.global_attributes.get("Conventions")
    out = str(tmp_path / "roundtrip.nc")
    try:
        nc.to_file(out)
    finally:
        nc.close()

    reopened = NetCDF.read_file(out)
    try:
        assert reopened.global_attributes.get("Conventions") == source_conventions, (
            f"{sample_name}: Conventions changed on write "
            f"({source_conventions!r} -> {reopened.global_attributes.get('Conventions')!r})"
        )
    finally:
        reopened.close()


@pytest.mark.samples("packed")
def test_roundtrip_preserves_packing(sample_name, sample, tmp_path):
    """A packed (int16 scale/offset) variable keeps its scale, offset, and dtype across a round-trip."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        packed = {
            name: (info.scale, info.offset, info.dtype)
            for name, info in meta.variables.items()
            if info.scale is not None
        }
        assert packed, f"{sample_name}: expected at least one packed variable"
    finally:
        out = str(tmp_path / "roundtrip.nc")
        nc.to_file(out)
        nc.close()

    reopened = NetCDF.read_file(out)
    try:
        after = reopened.get_all_metadata().variables
        for name, (scale, offset, dtype) in packed.items():
            info = after[name]
            assert info.scale == pytest.approx(
                scale
            ), f"{sample_name}/{name}: scale changed"
            assert info.offset == pytest.approx(
                offset
            ), f"{sample_name}/{name}: offset changed"
            assert info.dtype == dtype, f"{sample_name}/{name}: dtype changed"
    finally:
        reopened.close()
