"""Tests for the HTTP retry / timeout / VSI_CACHE knobs on ``CloudConfig`` (PY-6).

Covers the four new typed fields added to :class:`pyramids.base.remote.CloudConfig`
— ``http_max_retry``, ``http_retry_delay``, ``http_timeout``, ``vsi_cache`` — and
their mapping in :meth:`CloudConfig.as_gdal_config`. The e2e tests verify that
the context manager actually sets and restores the corresponding GDAL config
options (which are process- / thread-local, so we seed pre-existing values
with :func:`osgeo.gdal.SetConfigOption` and assert restoration on exit).
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.base.remote import CloudConfig

pytestmark = pytest.mark.core

_HTTP_KEYS = (
    "GDAL_HTTP_MAX_RETRY",
    "GDAL_HTTP_RETRY_DELAY",
    "GDAL_HTTP_TIMEOUT",
    "VSI_CACHE",
)


@pytest.fixture()
def clean_gdal_http_config():
    """Snapshot the GDAL HTTP / VSI_CACHE options and restore them after the test.

    Yields:
        None: tests run with the snapshotted state captured, then on teardown
        every key listed in ``_HTTP_KEYS`` is restored to whatever it was at
        entry (or cleared if it wasn't set).
    """
    snapshot = {key: gdal.GetConfigOption(key) for key in _HTTP_KEYS}
    try:
        yield
    finally:
        for key, value in snapshot.items():
            gdal.SetConfigOption(key, value)


class TestCloudConfigHttpFields:
    """Mapping tests for the four new fields on :class:`CloudConfig`."""

    def test_default_cloudconfig_does_not_emit_http_keys(self):
        """A bare ``CloudConfig()`` produces an empty dict — no HTTP / VSI_CACHE keys.

        Test scenario:
            ``CloudConfig().as_gdal_config()`` — expected: empty mapping (every
            new field defaults to ``None`` and therefore drops out).
        """
        cfg = CloudConfig().as_gdal_config()
        assert cfg == {}, f"expected empty dict, got {cfg!r}"

    @pytest.mark.parametrize(
        "field, value, expected_key, expected_str",
        [
            ("http_max_retry", 5, "GDAL_HTTP_MAX_RETRY", "5"),
            ("http_max_retry", 0, "GDAL_HTTP_MAX_RETRY", "0"),
            ("http_retry_delay", 1.5, "GDAL_HTTP_RETRY_DELAY", "1.5"),
            ("http_retry_delay", 0.0, "GDAL_HTTP_RETRY_DELAY", "0.0"),
            ("http_timeout", 60, "GDAL_HTTP_TIMEOUT", "60"),
        ],
        ids=[
            "http_max_retry-5",
            "http_max_retry-0",
            "http_retry_delay-1.5",
            "http_retry_delay-0.0",
            "http_timeout-60",
        ],
    )
    def test_typed_field_maps_to_gdal_key_with_str_value(
        self, field, value, expected_key, expected_str
    ):
        """Each numeric field maps to its GDAL config key with a ``str``-coerced value.

        Args:
            field: Attribute name on ``CloudConfig``.
            value: Value to set on that field.
            expected_key: GDAL config option key that should appear in the output.
            expected_str: Stringified value that should appear under that key.

        Test scenario:
            Build ``CloudConfig(**{field: value})``, call ``as_gdal_config()`` —
            expected: ``{expected_key: expected_str}`` and nothing else.
        """
        cfg = CloudConfig(**{field: value}).as_gdal_config()
        assert cfg == {
            expected_key: expected_str
        }, f"unexpected mapping for {field}={value!r}: {cfg!r}"

    @pytest.mark.parametrize(
        "value, expected",
        [(True, "TRUE"), (False, "FALSE")],
        ids=["vsi_cache=True", "vsi_cache=False"],
    )
    def test_vsi_cache_bool_maps_to_truefalse_string(self, value, expected):
        """``vsi_cache=True`` / ``False`` becomes ``VSI_CACHE=TRUE`` / ``FALSE``.

        Args:
            value: The bool passed to ``vsi_cache``.
            expected: The expected string value under ``VSI_CACHE``.

        Test scenario:
            Construct ``CloudConfig(vsi_cache=value)`` — expected: a one-entry
            mapping ``{"VSI_CACHE": expected}``.
        """
        cfg = CloudConfig(vsi_cache=value).as_gdal_config()
        assert cfg == {"VSI_CACHE": expected}, f"unexpected vsi_cache mapping: {cfg!r}"

    def test_vsi_cache_none_drops_out(self):
        """``vsi_cache=None`` (the default) emits no ``VSI_CACHE`` key.

        Test scenario:
            ``CloudConfig(vsi_cache=None).as_gdal_config()`` — expected: empty
            mapping (the process-wide setting, if any, is left alone).
        """
        cfg = CloudConfig(vsi_cache=None).as_gdal_config()
        assert cfg == {}, f"vsi_cache=None should drop out, got {cfg!r}"

    @pytest.mark.parametrize(
        "value, expected",
        [(True, "TRUE"), (False, "FALSE")],
        ids=["aws_virtual_hosting=True", "aws_virtual_hosting=False"],
    )
    def test_aws_virtual_hosting_bool_maps_to_truefalse(self, value, expected):
        """``aws_virtual_hosting`` maps to ``AWS_VIRTUAL_HOSTING=TRUE``/``FALSE`` (#560).

        Args:
            value: The bool passed to ``aws_virtual_hosting``.
            expected: The expected string value under ``AWS_VIRTUAL_HOSTING``.

        Test scenario:
            Path-style (``False``) is what the anonymous-S3 read path uses to
            avoid GDAL's unfollowed 301 on the data-chunk GET.
        """
        cfg = CloudConfig(aws_virtual_hosting=value).as_gdal_config()
        assert cfg == {
            "AWS_VIRTUAL_HOSTING": expected
        }, f"unexpected aws_virtual_hosting mapping: {cfg!r}"

    def test_aws_virtual_hosting_none_drops_out(self):
        """``aws_virtual_hosting=None`` (the default) emits no key.

        Test scenario:
            ``CloudConfig(aws_virtual_hosting=None).as_gdal_config()`` — expected:
            empty mapping, so GDAL's default addressing is left untouched.
        """
        cfg = CloudConfig(aws_virtual_hosting=None).as_gdal_config()
        assert cfg == {}, f"aws_virtual_hosting=None should drop out, got {cfg!r}"

    def test_all_four_fields_combine(self):
        """The four fields combine into one mapping in a single ``CloudConfig``.

        Test scenario:
            Build a config with every new knob set — expected: four entries
            in the mapping, none missing, each with the right type-coerced value.
        """
        cfg = CloudConfig(
            http_max_retry=3,
            http_retry_delay=0.5,
            http_timeout=30,
            vsi_cache=True,
        ).as_gdal_config()
        assert cfg == {
            "GDAL_HTTP_MAX_RETRY": "3",
            "GDAL_HTTP_RETRY_DELAY": "0.5",
            "GDAL_HTTP_TIMEOUT": "30",
            "VSI_CACHE": "TRUE",
        }, f"unexpected combined mapping: {cfg!r}"

    def test_mixed_with_aws_credentials(self):
        """HTTP knobs coexist with credential fields in one ``CloudConfig``.

        Test scenario:
            ``CloudConfig(aws_region=..., http_max_retry=..., vsi_cache=True)``
            — expected: AWS and HTTP keys appear together in the mapping.
        """
        cfg = CloudConfig(
            aws_region="us-east-1",
            http_max_retry=3,
            vsi_cache=True,
        ).as_gdal_config()
        assert cfg == {
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "GDAL_HTTP_MAX_RETRY": "3",
            "VSI_CACHE": "TRUE",
        }, f"unexpected credentials+http mapping: {cfg!r}"

    def test_extra_overrides_typed_field_on_conflict(self):
        """``extra`` is the escape hatch and wins on key conflict.

        Test scenario:
            ``CloudConfig(http_max_retry=3, extra={"GDAL_HTTP_MAX_RETRY": "9"})``
            — expected: the ``extra`` value (``"9"``) is what comes out.
        """
        cfg = CloudConfig(
            http_max_retry=3,
            extra={"GDAL_HTTP_MAX_RETRY": "9"},
        ).as_gdal_config()
        assert cfg == {
            "GDAL_HTTP_MAX_RETRY": "9"
        }, f"extra= should override typed field, got {cfg!r}"

    def test_extra_overrides_vsi_cache(self):
        """``extra`` also overrides ``vsi_cache``'s ``TRUE``/``FALSE`` rendering.

        Test scenario:
            ``CloudConfig(vsi_cache=True, extra={"VSI_CACHE": "NO"})`` — expected:
            ``VSI_CACHE=NO`` (the ``extra`` mapping wins).
        """
        cfg = CloudConfig(
            vsi_cache=True,
            extra={"VSI_CACHE": "NO"},
        ).as_gdal_config()
        assert cfg == {
            "VSI_CACHE": "NO"
        }, f"extra should override vsi_cache, got {cfg!r}"


class TestCloudConfigHttpContextManager:
    """End-to-end: the ``with`` block sets and restores GDAL options."""

    def test_sets_then_restores_when_previous_value_unset(self, clean_gdal_http_config):
        """Inside ``with``, GDAL reports the new value; outside, it goes back to ``None``.

        Args:
            clean_gdal_http_config: Fixture that snapshots and restores the keys.

        Test scenario:
            Pre-condition: ``GDAL_HTTP_MAX_RETRY`` is unset (``None``). Enter a
            ``CloudConfig(http_max_retry=7)`` block — expected: ``GetConfigOption``
            reports ``"7"`` inside, and the key is back to ``None`` after exit.
        """
        gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", None)
        assert (
            gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY") is None
        ), "precondition: key should start unset"
        with CloudConfig(http_max_retry=7):
            inside = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")
        after = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")
        assert inside == "7", f"expected '7' inside the block, got {inside!r}"
        assert after is None, f"expected unset after the block, got {after!r}"

    def test_restores_previous_explicit_value(self, clean_gdal_http_config):
        """A seeded pre-existing value is restored on exit, not blanked.

        Args:
            clean_gdal_http_config: Snapshot/restore fixture.

        Test scenario:
            Seed ``GDAL_HTTP_TIMEOUT="prev"``, enter a ``CloudConfig(http_timeout=42)``
            block — expected: ``"42"`` inside, ``"prev"`` after.
        """
        gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "prev")
        with CloudConfig(http_timeout=42):
            inside = gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
        after = gdal.GetConfigOption("GDAL_HTTP_TIMEOUT")
        assert inside == "42", f"expected '42' inside, got {inside!r}"
        assert after == "prev", f"expected 'prev' restored after exit, got {after!r}"

    @pytest.mark.parametrize(
        "value, expected_str",
        [(True, "TRUE"), (False, "FALSE")],
        ids=["True", "False"],
    )
    def test_vsi_cache_applied_inside_block(
        self, clean_gdal_http_config, value, expected_str
    ):
        """``vsi_cache`` toggles ``VSI_CACHE`` inside the block.

        Args:
            clean_gdal_http_config: Snapshot/restore fixture.
            value: The bool to pass.
            expected_str: The expected string under ``VSI_CACHE``.

        Test scenario:
            With the key unset beforehand, enter ``CloudConfig(vsi_cache=value)``
            — expected: ``VSI_CACHE`` reads ``expected_str`` inside, unset after.
        """
        gdal.SetConfigOption("VSI_CACHE", None)
        with CloudConfig(vsi_cache=value):
            inside = gdal.GetConfigOption("VSI_CACHE")
        after = gdal.GetConfigOption("VSI_CACHE")
        assert (
            inside == expected_str
        ), f"expected {expected_str!r} inside, got {inside!r}"
        assert after is None, f"expected unset after exit, got {after!r}"

    def test_vsi_cache_none_leaves_existing_value_untouched(
        self, clean_gdal_http_config
    ):
        """``vsi_cache=None`` (default) does not stomp a pre-existing ``VSI_CACHE``.

        Args:
            clean_gdal_http_config: Snapshot/restore fixture.

        Test scenario:
            Seed ``VSI_CACHE="TRUE"`` and enter a ``CloudConfig()`` block that
            only sets credentials — expected: ``VSI_CACHE`` still reads ``"TRUE"``
            both inside and after.
        """
        gdal.SetConfigOption("VSI_CACHE", "TRUE")
        with CloudConfig(aws_region="us-east-1"):
            inside = gdal.GetConfigOption("VSI_CACHE")
        after = gdal.GetConfigOption("VSI_CACHE")
        assert inside == "TRUE", f"expected 'TRUE' inside, got {inside!r}"
        assert after == "TRUE", f"expected 'TRUE' after, got {after!r}"

    def test_nested_blocks_restore_inner_value(self, clean_gdal_http_config):
        """Nesting two ``CloudConfig`` blocks restores the outer value when the inner exits.

        Args:
            clean_gdal_http_config: Snapshot/restore fixture.

        Test scenario:
            Outer ``http_max_retry=3``, inner ``http_max_retry=9``. While inside
            the inner block ``GetConfigOption`` reads ``"9"``; after the inner
            exits but still inside the outer it reads ``"3"``; after the outer
            exits it returns to the snapshotted (here unset) value.
        """
        gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", None)
        with CloudConfig(http_max_retry=3):
            outer_value = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")
            with CloudConfig(http_max_retry=9):
                inner_value = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")
            after_inner = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")
        after_outer = gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY")

        assert outer_value == "3", f"outer expected '3', got {outer_value!r}"
        assert inner_value == "9", f"inner expected '9', got {inner_value!r}"
        assert (
            after_inner == "3"
        ), f"after inner expected '3' (outer restored), got {after_inner!r}"
        assert after_outer is None, f"after outer expected unset, got {after_outer!r}"

    def test_exception_inside_block_still_restores(self, clean_gdal_http_config):
        """An exception inside the block does not leak the temporary GDAL setting.

        Args:
            clean_gdal_http_config: Snapshot/restore fixture.

        Test scenario:
            Seed ``GDAL_HTTP_MAX_RETRY="prev"``, raise inside the block —
            expected: the exception propagates and the previous value is back.
        """
        gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "prev")
        with pytest.raises(RuntimeError, match="boom"):
            with CloudConfig(http_max_retry=11):
                raise RuntimeError("boom")
        assert (
            gdal.GetConfigOption("GDAL_HTTP_MAX_RETRY") == "prev"
        ), "previous value not restored after exception inside the block"
