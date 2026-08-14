"""Unit tests for the grouped COG option dataclasses and their validators.

Covers each ``__post_init__`` validation branch and ``Compression.coerce`` for
`Compression`, `Overviews`, `Tiling`, `BandSelection`, `Tags`, and `Layout`.
"""

from __future__ import annotations

import pytest

from pyramids.dataset.cog import (
    PROFILES,
    BandSelection,
    Compression,
    Layout,
    Overviews,
    Tags,
    Tiling,
)

_COERCED_PROFILE_KEYS = {"COMPRESS", "LEVEL", "QUALITY", "MAX_Z_ERROR"}


def test_all_profiles_use_only_coerced_keys():
    """Every PROFILES entry uses only keys `Compression.coerce` carries.

    Guards against a future profile adding a key (e.g. `PREDICTOR`, `NBITS`) that
    the string-coercion path would silently drop.
    """
    for name, opts in PROFILES.items():
        extra = set(opts) - _COERCED_PROFILE_KEYS
        assert not extra, f"profile {name!r} uses keys coerce would drop: {extra}"

pytestmark = pytest.mark.core


class TestCompression:
    """Validation and coercion for `Compression`."""

    def test_defaults_are_none(self):
        """A bare `Compression` leaves every field `None`."""
        c = Compression()
        assert (c.compress, c.level, c.quality, c.predictor, c.max_z_error) == (
            None,
            None,
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("quality", [0, 101, -1, 200])
    def test_quality_out_of_range_rejected(self, quality):
        """`quality` outside 1..100 raises `ValueError`.

        Args:
            quality: An out-of-range quality value.
        """
        with pytest.raises(ValueError, match="quality must be in 1..100"):
            Compression(quality=quality)

    @pytest.mark.parametrize("quality", [1, 50, 100])
    def test_quality_in_range_accepted(self, quality):
        """`quality` within 1..100 is accepted.

        Args:
            quality: An in-range quality value.
        """
        assert Compression(quality=quality).quality == quality

    @pytest.mark.parametrize(
        "predictor", [1, 2, 3, "YES", "NO", "STANDARD", "FLOATING_POINT"]
    )
    def test_valid_predictors_accepted(self, predictor):
        """Every GDAL predictor token is accepted.

        Args:
            predictor: A valid predictor value.
        """
        assert Compression(predictor=predictor).predictor == predictor

    @pytest.mark.parametrize("predictor", ["1", "2", "3"])
    def test_string_numeric_predictor_accepted(self, predictor):
        """String-numeric predictors (forwarded by the old flat API) are accepted.

        Args:
            predictor: A numeric predictor in string form.
        """
        assert Compression(predictor=predictor).predictor == predictor

    @pytest.mark.parametrize("predictor", ["bogus", 4, "maybe"])
    def test_invalid_predictor_rejected(self, predictor):
        """An unknown predictor raises `ValueError`.

        Args:
            predictor: An invalid predictor value.
        """
        with pytest.raises(ValueError, match="predictor must be one of"):
            Compression(predictor=predictor)

    def test_coerce_profile_string(self):
        """A profile-name string expands to its preset."""
        assert Compression.coerce("zstd") == Compression(compress="ZSTD", level=9)

    def test_coerce_lerc_carries_max_z_error(self):
        """The lerc profile carries `MAX_Z_ERROR` into `max_z_error`."""
        assert Compression.coerce("lerc").max_z_error == 0.0

    def test_coerce_none_passes_through(self):
        """`None` coerces to `None`."""
        assert Compression.coerce(None) is None

    def test_coerce_instance_is_identity(self):
        """An existing `Compression` is returned unchanged."""
        c = Compression(compress="LZW")
        assert Compression.coerce(c) is c

    def test_coerce_unknown_profile_rejected(self):
        """A non-profile string raises `ValueError`."""
        with pytest.raises(ValueError, match="unknown COG profile"):
            Compression.coerce("nope")


class TestOverviews:
    """Validation for `Overviews`."""

    def test_negative_count_rejected(self):
        """A negative overview count raises `ValueError`."""
        with pytest.raises(ValueError, match="overview count must be >= 0"):
            Overviews(count=-1)

    def test_zero_count_accepted(self):
        """A zero overview count is accepted."""
        assert Overviews(count=0).count == 0


class TestTiling:
    """Validation for `Tiling`."""

    @pytest.mark.parametrize("strategy", ["auto", "lower", "upper"])
    def test_valid_strategies_accepted(self, strategy):
        """Every valid zoom-level strategy is accepted.

        Args:
            strategy: A valid zoom-level strategy.
        """
        assert Tiling(zoom_level_strategy=strategy).zoom_level_strategy == strategy

    def test_invalid_strategy_rejected(self):
        """An unknown zoom-level strategy raises `ValueError`."""
        with pytest.raises(ValueError, match="zoom_level_strategy must be"):
            Tiling(zoom_level_strategy="sideways")


class TestBandSelection:
    """Validation for `BandSelection`."""

    def test_negative_index_rejected(self):
        """A negative band index raises `ValueError`."""
        with pytest.raises(ValueError, match="band indexes must be >= 0"):
            BandSelection(indexes=[0, -1])

    def test_zero_based_indexes_accepted(self):
        """Non-negative 0-based indices are accepted in order."""
        assert BandSelection(indexes=[2, 0, 1]).indexes == [2, 0, 1]


class TestLayout:
    """Validation for `Layout`."""

    def test_default_blocksize(self):
        """The default `Layout` uses a 512 blocksize."""
        assert Layout().blocksize == 512

    def test_bad_blocksize_rejected(self):
        """A non-power-of-2 blocksize raises `ValueError`."""
        with pytest.raises(ValueError, match="blocksize must be a power of 2"):
            Layout(blocksize=100)

    @pytest.mark.parametrize("bigtiff", ["IF_SAFER", "YES", "NO", "IF_NEEDED"])
    def test_valid_bigtiff_accepted(self, bigtiff):
        """Every valid BIGTIFF token is accepted.

        Args:
            bigtiff: A valid BIGTIFF value.
        """
        assert Layout(bigtiff=bigtiff).bigtiff == bigtiff

    def test_bad_bigtiff_rejected(self):
        """An unknown BIGTIFF value raises `ValueError`."""
        with pytest.raises(ValueError, match="bigtiff must be"):
            Layout(bigtiff="MAYBE")


class TestTags:
    """`Tags` is a plain carrier with no validation."""

    def test_fields_round_trip(self):
        """The three fields are stored verbatim."""
        t = Tags(band_tags={0: {"name": "x"}}, metadata={"a": "b"})
        assert t.band_tags == {0: {"name": "x"}} and t.metadata == {"a": "b"}
