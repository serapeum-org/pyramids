import io
import logging
import threading
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
from osgeo import gdal

from pyramids.base import config as config_mod
from pyramids.base.config import (
    PACKAGE_LOGGER_NAME,
    Config,
    LoggerManager,
    _gdal_error_handler,
)

pytestmark = pytest.mark.core


@contextmanager
def isolated_package_logging():
    """
    Temporarily isolate the ``pyramids`` logger's handlers and level so tests
    don't interfere with each other or the global test suite.

    ARC-40: pyramids configures its own namespace, never the root logger, so the
    isolation target is ``logging.getLogger("pyramids")``.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    old_level = package_logger.level
    old_handlers = list(package_logger.handlers)
    # `propagate` must be restored too: opting into logging sets it False, and
    # leaking that would silently cut pytest's caplog (whose handler lives on the
    # root logger) off from every pyramids record for the rest of the session.
    old_propagate = package_logger.propagate
    try:
        for h in package_logger.handlers[:]:
            package_logger.removeHandler(h)
        yield package_logger
    finally:
        for h in package_logger.handlers[:]:
            package_logger.removeHandler(h)
        for h in old_handlers:
            package_logger.addHandler(h)
        package_logger.setLevel(old_level)
        package_logger.propagate = old_propagate


@contextmanager
def reinstallable_error_handler():
    """Let a test observe ``_set_error_handler`` installing, then restore the guard."""
    saved = config_mod._gdal_error_handler_installed
    config_mod._gdal_error_handler_installed = False
    try:
        yield
    finally:
        config_mod._gdal_error_handler_installed = saved


def test_console_logging_colored_and_message(capsys):
    with isolated_package_logging():
        # An explicit level opts into the coloured console handler.
        LoggerManager(level="DEBUG")

        # Emitted by setup_logging
        out = capsys.readouterr()
        stderr_text = out.err
        # Should have at least one ANSI escape (from ColorFormatter)
        assert "\x1b[" in stderr_text
        assert "Logging is configured." in stderr_text
        assert "pyramids.base.config" in stderr_text  # logger name

        # Also test that subsequent pyramids logs go to the console
        logging.getLogger("pyramids.tests.console").info("hello world")
        out2 = capsys.readouterr()
        assert "hello world" in out2.err
        assert "\x1b[" in out2.err  # colored level name


def test_default_level_configures_nothing(capsys):
    """ARC-40: ``LoggerManager()`` with no level installs only a NullHandler.

    Test scenario:
        Importing pyramids runs ``Config()``. That must not print anything, must
        not add a console handler, and must not set a level — logging policy
        belongs to the host application.
    """
    with isolated_package_logging() as package_logger:
        LoggerManager()
        out = capsys.readouterr()
        assert out.err == "", f"default LoggerManager must be silent; got {out.err!r}"
        assert out.out == "", f"default LoggerManager must be silent; got {out.out!r}"
        assert all(
            isinstance(h, logging.NullHandler) for h in package_logger.handlers
        ), (
            "default LoggerManager must add only a NullHandler; got "
            f"{[type(h).__name__ for h in package_logger.handlers]}"
        )
        assert package_logger.level == logging.NOTSET, (
            "default LoggerManager must not set a level on the pyramids logger"
        )


def test_root_logger_is_never_touched():
    """ARC-40: no pyramids handler ever lands on the root logger.

    Test scenario:
        Configuring pyramids at DEBUG must leave the root logger's handler list
        and level exactly as they were — a library that adds a root handler
        silently reconfigures logging for the whole application.
    """
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    with isolated_package_logging():
        LoggerManager(level="DEBUG", log_file=None)
    assert list(root.handlers) == handlers_before, (
        "pyramids must not add or remove root logger handlers"
    )
    assert root.level == level_before, "pyramids must not change the root logger level"


def test_file_logging_no_colors_and_writes(tmp_path: Path):
    log_file = tmp_path / "test.log"
    with isolated_package_logging():
        LoggerManager(level=logging.DEBUG, log_file=log_file)

        # setup_logging should log a message already
        # Now log something extra
        logging.getLogger("pyramids.tests.file_handler").debug("file handler check")

    # Read file and assert contents
    text = log_file.read_text(encoding="utf-8")
    assert "Logging is configured." in text
    assert "file handler check" in text
    # Ensure no ANSI color sequences in the file
    assert "\x1b[" not in text


def test_idempotent_handlers(tmp_path: Path):
    log_file = tmp_path / "dup.log"
    with isolated_package_logging() as package_logger:
        LoggerManager(level="INFO", log_file=log_file)
        # Call again with the same parameters
        LoggerManager(level="INFO", log_file=log_file)

        # Expect exactly one StreamHandler (console) and one FileHandler
        stream_handlers = [
            h
            for h in package_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [
            h for h in package_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
        assert len(file_handlers) == 1
        # And that file handler targets the same file
        assert Path(file_handlers[0].baseFilename) == log_file


def test_set_error_handler_prints_for_low_error_class():
    # Invoke the handler with an error class lower than CE_Warning to trigger printing
    buf = io.StringIO()
    with redirect_stdout(buf):
        _gdal_error_handler(0, 42, "oops")
    out = buf.getvalue().strip()

    assert out == "GDAL error (class 0, number 42): oops"


def _collect_log_messages(records, logger_name: str):
    return [r for r in records if r.name == logger_name]


def test_set_error_handler_logs_severities(capsys):
    # Isolate pyramids logging and configure
    with isolated_package_logging():
        LoggerManager(level="DEBUG")
        capsys.readouterr()  # drop the "Logging is configured." line

        # Emit messages: warning and higher are routed to the configured logger -> console (stderr)
        _gdal_error_handler(gdal.CE_Warning, 22, "warn msg")
        _gdal_error_handler(gdal.CE_Failure, 33, "fail msg")
        _gdal_error_handler(gdal.CE_Fatal, 44, "fatal msg")
        # Unknown class -> ERROR fallback
        _gdal_error_handler(999, 55, "unknown class msg")

        out = capsys.readouterr()
        err_text = out.err
        # Assert substrings exist in stderr (avoid brittle timestamp/ANSI sequences)
        assert "pyramids.base.config.gdal | GDAL[22] warn msg" in err_text
        assert "pyramids.base.config.gdal | GDAL[33] fail msg" in err_text
        assert "pyramids.base.config.gdal | GDAL[44] fatal msg" in err_text
        assert (
            "pyramids.base.config.gdal | GDAL(class=999, code=55) unknown class msg"
            in err_text
        )


@patch("osgeo.gdal.PushErrorHandler")
def test_error_handler_installed_only_once(mock_push):
    """ARC-40: repeated construction must not stack GDAL error handlers.

    Test scenario:
        ``gdal.PushErrorHandler`` pushes onto a stack that pyramids never pops,
        so an unguarded install made GDAL log every error once per push. Two
        LoggerManager constructions must produce at most one push.
    """
    with reinstallable_error_handler(), isolated_package_logging():
        LoggerManager()
        LoggerManager()
    assert mock_push.call_count == 1, (
        f"expected exactly one PushErrorHandler call, got {mock_push.call_count}"
    )
    assert mock_push.call_args[0][0] is _gdal_error_handler


@patch("osgeo.gdal.PushErrorHandler")
def test_error_handler_not_reinstalled_when_already_present(mock_push):
    """The install guard short-circuits once the handler is in place.

    Test scenario:
        Set the flag explicitly rather than relying on the package import
        having already set it — asserting against ambient module state
        would pass just as happily against a `_set_error_handler` that did
        nothing at all.
    """
    with reinstallable_error_handler():
        config_mod._gdal_error_handler_installed = True
        with isolated_package_logging():
            LoggerManager()
    mock_push.assert_not_called()


def test_error_handler_install_is_serialised_across_threads():
    """Concurrent `Config()` construction still pushes exactly one handler.

    Test scenario:
        The guard is a check-then-set on a module global. Without the lock,
        several threads can all read it as unset and each push a handler —
        reintroducing the stacking (and the duplicated GDAL log lines) the
        guard exists to prevent.
    """
    pushed: list = []
    barrier = threading.Barrier(8)

    def install():
        barrier.wait()
        LoggerManager._set_error_handler()

    with reinstallable_error_handler(), patch(
        "osgeo.gdal.PushErrorHandler", side_effect=lambda h: pushed.append(h)
    ):
        config_mod._gdal_error_handler_installed = False
        threads = [threading.Thread(target=install) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert len(pushed) == 1, (
        f"8 racing installs must push exactly one handler, got {len(pushed)}"
    )


def test_setup_logging_invalid_level_string_raises():
    with isolated_package_logging():
        with pytest.raises(ValueError):
            LoggerManager(level="NOT_A_LEVEL")


@patch("pyramids.base.config.Config.set_env_conda")
def test_dynamic_env_variables_returns_early_when_conda_provides_path(mock_set_env):
    # Return a specific path from set_env_conda to ensure early return
    expected = Path("/fake/conda/env/Library/lib/gdalplugins")
    mock_set_env.return_value = expected
    cfg = object.__new__(Config)
    cfg.logger = logging.getLogger("tests.config.coverage")
    with patch("sys.platform", new="linux"):
        result = cfg.dynamic_env_variables()
    assert result == expected


@patch("osgeo.gdal.SetConfigOption")
@patch("osgeo.gdal.AllRegister")
@patch("pyramids.base.config.Config.dynamic_env_variables")
def test_initialize_gdal_sets_options_and_conditional_driver_path(
    mock_dyn, mock_register, mock_setopt
):
    # Create instance without running __init__ side-effects
    cfg = object.__new__(Config)
    cfg.logger = logging.getLogger("tests.config.coverage")
    cfg.settings = {
        "gdal": {"GDAL_CACHEMAX": "256"},
        "ogr": {"OGR_SRS_PARSER": "strict"},
    }

    # Case 1: dynamic_env_variables returns None -> no GDAL_DRIVER_PATH set
    mock_dyn.return_value = None
    cfg.initialize_gdal()
    # Called for provided options
    mock_setopt.assert_any_call("GDAL_CACHEMAX", "256")
    mock_setopt.assert_any_call("OGR_SRS_PARSER", "strict")
    # Ensure GDAL_DRIVER_PATH was not set in this branch
    assert ("GDAL_DRIVER_PATH",) not in [c.args[:1] for c in mock_setopt.call_args_list]

    mock_setopt.reset_mock()

    # Case 2: dynamic_env_variables returns a Path -> GDAL_DRIVER_PATH set
    path = Path("/some/plugins")
    mock_dyn.return_value = path
    cfg.initialize_gdal()
    mock_setopt.assert_any_call("GDAL_DRIVER_PATH", str(path))
    mock_register.assert_called()


def test_error_handler_exception_fallback_logs_error(capsys):
    # Install handler via LoggerManager
    with isolated_package_logging():
        LoggerManager(level="DEBUG")
        # Cause a TypeError inside the handler's try block by passing a non-orderable err_class
        _gdal_error_handler(object(), 66, "boom")
        err = capsys.readouterr().err
    # The fallback except path logs an error with the generic format
    assert "GDAL(class=" in err
    assert "code=66" in err
    assert "boom" in err


def test_opt_in_logging_does_not_double_print_under_basicconfig(capsys):
    """S3: pyramids records reach the console once, not twice.

    Test scenario:
        A host that ran ``logging.basicConfig()`` has a root handler. With
        pyramids' own handler on the ``pyramids`` logger and propagation
        left on, every record was formatted by both. Owning a handler must
        therefore also stop propagation.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    stream = io.StringIO()
    host_handler = logging.StreamHandler(stream)
    host_handler.setFormatter(logging.Formatter("ROOT:%(name)s:%(message)s"))
    try:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        root.addHandler(host_handler)
        root.setLevel(logging.DEBUG)
        with isolated_package_logging():
            LoggerManager(level="INFO")
            capsys.readouterr()
            logging.getLogger("pyramids.tests.double").info("only once please")
            captured = capsys.readouterr()
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)

    assert captured.err.count("only once please") == 1, (
        f"pyramids' own handler must emit the record exactly once, got {captured.err!r}"
    )
    assert "only once please" not in stream.getvalue(), (
        "the record must not also reach the host's root handler; got "
        f"{stream.getvalue()!r}"
    )


def test_default_level_leaves_propagation_intact():
    """With no level, host-configured root logging still sees pyramids records.

    Test scenario:
        The import path must not claim the namespace. Propagation stays on
        so an application that configures root logging keeps receiving
        pyramids output — the whole point of the opt-in default.
    """
    with isolated_package_logging() as package_logger:
        package_logger.propagate = True
        LoggerManager()
        assert package_logger.propagate is True, (
            "the no-level path must leave propagation alone"
        )


def test_third_party_loggers_untouched_by_default():
    """S5: opting into pyramids logging does not re-level other libraries.

    Test scenario:
        A host that deliberately set matplotlib to DEBUG must keep it.
        The convenience is still available, but only when asked for.
    """
    victim = logging.getLogger("matplotlib")
    saved = victim.level
    try:
        victim.setLevel(logging.DEBUG)
        with isolated_package_logging():
            LoggerManager(level="INFO")
        assert victim.level == logging.DEBUG, (
            "the host's third-party log level must survive by default"
        )
        with isolated_package_logging():
            LoggerManager(level="INFO", quiet_third_party=True)
        assert victim.level == logging.WARNING, (
            "quiet_third_party=True must still pin the noisy loggers"
        )
    finally:
        victim.setLevel(saved)
