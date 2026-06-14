"""Unit tests for pyramids.base._bootstrap (vendored-osgeo activation).

Focus: the curl CA-bundle wiring added for issue #412. The bootstrap must
re-point GDAL / PROJ / libcurl at the cacert.pem vendored into the wheel,
without clobbering a user/system override (the setdefault contract).
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import pytest

from pyramids.base._bootstrap import activate_vendored_osgeo

pytestmark = pytest.mark.core

_CA_ENV_VARS = (
    "GDAL_HTTP_CAINFO",
    "PROJ_CURL_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
)

# Env vars the driver-path / data-dir logic touches, plus the opt-out.
_GDAL_ENV_VARS = (
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "PROJ_DATA",
    "PROJ_LIB",
    "PYRAMIDS_KEEP_GDAL_ENV",
)


@pytest.fixture
def vendored_pkg(tmp_path: Path):
    """Build a fake vendored-wheel package dir (ABI-matching _gdal ext).

    Yields the package dir. `with_ca` controls whether the cacert.pem is
    present so individual tests can toggle it.
    """

    def _build(with_ca: bool = False, with_plugins: bool = False) -> Path:
        osgeo = tmp_path / "_vendor" / "osgeo"
        osgeo.mkdir(parents=True)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        (osgeo / f"_gdal{ext_suffix}").touch()
        if with_ca:
            ssl_dir = tmp_path / "_data" / "ssl"
            ssl_dir.mkdir(parents=True)
            (ssl_dir / "cacert.pem").write_text("# fake bundle\n", encoding="utf-8")
        if with_plugins:
            plugins = tmp_path / "_data" / "gdalplugins"
            plugins.mkdir(parents=True)
            (plugins / "gdal_netCDF.so").touch()
        return tmp_path

    return _build


@pytest.fixture(autouse=True)
def _isolate_env_and_syspath():
    """Snapshot the CA env vars + sys.path; restore after each test.

    activate_vendored_osgeo mutates os.environ (via setdefault) and
    inserts the vendor dir into sys.path — both are process-global, so
    restore them to keep tests order-independent.
    """
    tracked = _CA_ENV_VARS + _GDAL_ENV_VARS
    saved_env = {k: os.environ.get(k) for k in tracked}
    saved_path = list(sys.path)
    for k in tracked:
        os.environ.pop(k, None)
    yield
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    sys.path[:] = saved_path


class TestCaBundle:
    """CA-bundle env wiring (#412)."""

    def test_sets_all_ca_vars_to_bundled_cert(self, vendored_pkg):
        """All CA env vars point at the vendored cacert.pem when present.

        Test scenario:
            A vendored package dir ships _data/ssl/cacert.pem; activation
            sets every CA env var to that file.
        """
        pkg = vendored_pkg(with_ca=True)
        expected = str(pkg / "_data" / "ssl" / "cacert.pem")

        assert activate_vendored_osgeo(pkg) is True
        for var in _CA_ENV_VARS:
            assert os.environ.get(var) == expected, f"{var} not pointed at bundle"

    def test_does_not_override_user_setting(self, vendored_pkg):
        """A pre-set CA var wins over the bundled cert (setdefault contract).

        Test scenario:
            CURL_CA_BUNDLE is already set; activation leaves it untouched
            while still setting the unset vars.
        """
        pkg = vendored_pkg(with_ca=True)
        os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

        activate_vendored_osgeo(pkg)

        assert os.environ["CURL_CA_BUNDLE"] == "/etc/ssl/certs/ca-certificates.crt"
        assert os.environ["GDAL_HTTP_CAINFO"] == str(
            pkg / "_data" / "ssl" / "cacert.pem"
        )

    def test_no_ca_vars_when_cert_absent(self, vendored_pkg):
        """No CA var is set when the wheel ships no cacert.pem.

        Test scenario:
            A vendored package dir without _data/ssl/cacert.pem activates
            (the osgeo ext is present) but sets none of the CA env vars.
        """
        pkg = vendored_pkg(with_ca=False)

        assert activate_vendored_osgeo(pkg) is True
        for var in _CA_ENV_VARS:
            assert var not in os.environ, f"{var} should be unset without a bundle"


class TestDriverPath:
    """GDAL_DRIVER_PATH override for the bundled, version-locked plugins (#465)."""

    def test_sets_driver_path_when_unset(self, vendored_pkg):
        """GDAL_DRIVER_PATH points at the bundled plugin dir when unset.

        Test scenario:
            A vendored package dir ships _data/gdalplugins; activation
            with no inherited GDAL_DRIVER_PATH sets it to that dir.
        """
        pkg = vendored_pkg(with_plugins=True)
        expected = str(pkg / "_data" / "gdalplugins")

        assert activate_vendored_osgeo(pkg) is True
        assert os.environ["GDAL_DRIVER_PATH"] == expected

    def test_overrides_inherited_driver_path(self, vendored_pkg):
        """A conda-style inherited GDAL_DRIVER_PATH is FORCED to the bundle.

        Test scenario:
            GDAL_DRIVER_PATH is pre-set to a foreign dir (as an activated
            conda env exports it). Because the bundled plugins are
            ABI-locked to the bundled libgdal, activation overrides it
            rather than honoring the inherited value (#465).
        """
        pkg = vendored_pkg(with_plugins=True)
        os.environ["GDAL_DRIVER_PATH"] = "/opt/conda/envs/x/lib/gdalplugins"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DRIVER_PATH"] == str(pkg / "_data" / "gdalplugins")

    def test_opt_out_keeps_inherited_driver_path(self, vendored_pkg):
        """PYRAMIDS_KEEP_GDAL_ENV=1 restores the old setdefault behavior.

        Test scenario:
            With the opt-out set, a pre-set GDAL_DRIVER_PATH is preserved.
        """
        pkg = vendored_pkg(with_plugins=True)
        os.environ["PYRAMIDS_KEEP_GDAL_ENV"] = "1"
        os.environ["GDAL_DRIVER_PATH"] = "/opt/conda/envs/x/lib/gdalplugins"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DRIVER_PATH"] == "/opt/conda/envs/x/lib/gdalplugins"

    def test_data_dirs_still_respect_user_override(self, vendored_pkg):
        """GDAL_DATA / PROJ_DATA keep setdefault semantics (not forced).

        Test scenario:
            A pre-set GDAL_DATA is left untouched even though the driver
            path is forced — the data dirs are version-tolerant and the
            override contract for them is intentional.
        """
        pkg = vendored_pkg(with_plugins=True)
        (pkg / "_data" / "gdal_data").mkdir(parents=True)
        (pkg / "_data" / "proj_data").mkdir(parents=True)
        os.environ["GDAL_DATA"] = "/custom/gdal_data"

        activate_vendored_osgeo(pkg)

        assert os.environ["GDAL_DATA"] == "/custom/gdal_data"
        assert os.environ["PROJ_DATA"] == str(pkg / "_data" / "proj_data")
        assert os.environ["GDAL_DRIVER_PATH"] == str(pkg / "_data" / "gdalplugins")
