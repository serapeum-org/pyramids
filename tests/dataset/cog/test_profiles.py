"""Tests for named COG profiles (PB-5).

Covers the PROFILES presets, profile_options / validate_profile helpers, and the
`profile=` kwarg on to_cog (precedence vs explicit kwargs, and the jpeg/webp
dtype/band constraints).
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.cog import PROFILES, profile_options, validate_profile

pytestmark = pytest.mark.core

_GEOTRANSFORM = (0.0, 0.01, 0.0, 10.0, 0.0, -0.01)


@pytest.fixture
def float_dataset() -> Dataset:
    """A 64x64 Float32 Dataset on EPSG:4326.

    Returns:
        Dataset: An in-memory float32 dataset.
    """
    rng = np.random.default_rng(seed=5)
    arr = (rng.random((64, 64)) * 100).astype("float32")
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


@pytest.fixture
def byte_dataset() -> Dataset:
    """A 64x64 single-band Byte Dataset on EPSG:4326.

    Returns:
        Dataset: An in-memory uint8 dataset.
    """
    arr = (np.arange(64 * 64) % 200).astype("uint8").reshape(64, 64)
    return Dataset.create_from_array(arr, geo=_GEOTRANSFORM, epsg=4326)


def _compression(path) -> str:
    """Return the IMAGE_STRUCTURE compression of a raster.

    Args:
        path: Path to a raster.

    Returns:
        str: Compression token, or "".
    """
    ds = gdal.Open(str(path))
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or ""
    ds = None
    return comp


class TestProfileHelpers:
    """Tests for profile_options and validate_profile."""

    def test_profile_options_returns_copy(self):
        """profile_options returns an independent copy of the preset.

        Test scenario:
            Mutating the result does not affect the PROFILES registry.
        """
        opts = profile_options("deflate")
        opts["LEVEL"] = 1
        assert PROFILES["deflate"]["LEVEL"] == 9, "registry must not be mutated"

    def test_profile_options_case_insensitive(self):
        """Profile names are case-insensitive.

        Test scenario:
            'ZSTD' resolves to the zstd preset.
        """
        assert profile_options("ZSTD")["COMPRESS"] == "ZSTD", "case-insensitive lookup"

    def test_unknown_profile_raises(self):
        """An unknown profile name raises ValueError.

        Test scenario:
            'bogus' is not a registered profile.
        """
        with pytest.raises(ValueError, match="unknown COG profile"):
            profile_options("bogus")

    def test_validate_profile_unconstrained_passes(self):
        """An unconstrained profile passes for any dtype/band count.

        Test scenario:
            deflate has no dtype constraint.
        """
        validate_profile("deflate", "Float32", 4)

    def test_validate_profile_jpeg_dtype(self):
        """JPEG rejects non-Byte dtypes.

        Test scenario:
            Float32 violates the JPEG Byte-only constraint.
        """
        with pytest.raises(ValueError, match="jpeg profile requires dtype"):
            validate_profile("jpeg", "Float32", 1)

    def test_validate_profile_webp_bands(self):
        """WEBP rejects a band count outside 3-4.

        Test scenario:
            A single Byte band violates the WEBP 3-4 band constraint.
        """
        with pytest.raises(ValueError, match="webp profile requires"):
            validate_profile("webp", "Byte", 1)


class TestToCogProfile:
    """Tests for the to_cog profile= kwarg."""

    def test_profile_sets_compression(self, float_dataset, tmp_path):
        """profile='zstd' produces a ZSTD-compressed COG.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            The written COG's compression is ZSTD.
        """
        out = float_dataset.to_cog(tmp_path / "z.tif", profile="zstd")
        assert _compression(out) == "ZSTD", "profile should set ZSTD compression"

    def test_explicit_compress_overrides_profile(self, float_dataset, tmp_path):
        """An explicit compress kwarg wins over the profile.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            profile='zstd' but compress='DEFLATE' yields DEFLATE.
        """
        out = float_dataset.to_cog(
            tmp_path / "o.tif", profile="zstd", compress="DEFLATE"
        )
        assert _compression(out) == "DEFLATE", "explicit compress must override profile"

    def test_lzw_profile(self, float_dataset, tmp_path):
        """profile='lzw' produces an LZW COG.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            The written COG's compression is LZW.
        """
        out = float_dataset.to_cog(tmp_path / "l.tif", profile="lzw")
        assert _compression(out) == "LZW", "profile should set LZW compression"

    def test_jpeg_profile_rejects_float(self, float_dataset, tmp_path):
        """profile='jpeg' on a float source raises before writing.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            The JPEG Byte-only constraint is enforced up-front.
        """
        with pytest.raises(ValueError, match="jpeg profile requires dtype"):
            float_dataset.to_cog(tmp_path / "j.tif", profile="jpeg")

    def test_unknown_profile_rejected(self, float_dataset, tmp_path):
        """An unknown profile name raises ValueError.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            profile='nope' is not registered.
        """
        with pytest.raises(ValueError, match="unknown COG profile"):
            float_dataset.to_cog(tmp_path / "n.tif", profile="nope")

    def test_profile_result_is_valid_cog(self, float_dataset, tmp_path):
        """A profile-driven write still produces a valid COG.

        Args:
            float_dataset: Fixture float32 Dataset.
            tmp_path: pytest temp directory.

        Test scenario:
            profile='deflate' yields a valid COG.
        """
        out = float_dataset.to_cog(tmp_path / "d.tif", profile="deflate")
        assert (
            Dataset.read_file(str(out)).validate_cog().is_valid
        ), "profile COG invalid"
