import io
import logging
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
from osgeo import gdal

from pyramids.base.config import Config, LoggerManager

pytestmark = pytest.mark.core


@contextmanager
def isolated_root_logging():
    """
    Temporarily isolate root logger handlers and level so tests don't
    interfere with each other or the global test suite.
    """
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
        yield root
    finally:
        # Clear any handlers added by the test
        for h in list(root.handlers):
            root.removeHandler(h)
        # Restore originals
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)


def test_console_logging_colored_and_message(capsys):
    with isolated_root_logging():
        # Configure logging at INFO level using LoggerManager
        LoggerManager(level="INFO")

        # Emitted by setup_logging
        out = capsys.readouterr()
        stderr_text = out.err
        # Should have at least one ANSI escape (from ColorFormatter)
        assert "\x1b[" in stderr_text
        assert "Logging is configured." in stderr_text
        assert "pyramids.base.config" in stderr_text  # logger name

        # Also test that subsequent logs go to the console
        logging.getLogger(__name__).info("hello world")
        out2 = capsys.readouterr()
        assert "hello world" in out2.err
        assert "\x1b[" in out2.err  # colored level name


def test_file_logging_no_colors_and_writes(tmp_path: Path):
    log_file = tmp_path / "test.log"
    with isolated_root_logging():
        LoggerManager(level=logging.DEBUG, log_file=log_file)

        # setup_logging should log a message already
        # Now log something extra
        logging.getLogger("tests.config.test_logger").debug("file handler check")

    # Read file and assert contents
    text = log_file.read_text(encoding="utf-8")
    assert "Logging is configured." in text
    assert "file handler check" in text
    # Ensure no ANSI color sequences in the file
    assert "\x1b[" not in text


def test_idempotent_handlers(tmp_path: Path):
    log_file = tmp_path / "dup.log"
    with isolated_root_logging() as root:
        LoggerManager(level="INFO", log_file=log_file)
        # Call again with the same parameters
        LoggerManager(level="INFO", log_file=log_file)

        # Expect exactly one StreamHandler (console) and one FileHandler
        stream_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
        assert len(file_handlers) == 1
        # And that file handler targets the same file
        assert Path(file_handlers[0].baseFilename) == log_file


@patch("osgeo.gdal.PushErrorHandler")
def test_set_error_handler_prints_for_low_error_class(mock_push):
    # Install the handler via LoggerManager and capture it from the patched GDAL entry point
    LoggerManager()
    handler = mock_push.call_args[0][0]

    # Invoke the handler with an error class lower than CE_Warning to trigger printing
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler(0, 42, "oops")
    out = buf.getvalue().strip()

    assert out == "GDAL error (class 0, number 42): oops"


def _collect_log_messages(records, logger_name: str):
    return [r for r in records if r.name == logger_name]


@patch("osgeo.gdal.PushErrorHandler")
def test_set_error_handler_logs_severities(mock_push, capsys):
    # Isolate root logging and configure
    with isolated_root_logging():
        LoggerManager(level="DEBUG")

        # Capture the handler installed during construction
        handler = mock_push.call_args[0][0]

        # Emit messages: warning and higher are routed to the configured logger -> console (stderr)
        handler(gdal.CE_Warning, 22, "warn msg")
        handler(gdal.CE_Failure, 33, "fail msg")
        handler(gdal.CE_Fatal, 44, "fatal msg")
        # Unknown class -> ERROR fallback
        handler(999, 55, "unknown class msg")

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


def test_setup_logging_invalid_level_string_raises():
    with isolated_root_logging():
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
    cfg.config = {
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


@patch("osgeo.gdal.PushErrorHandler")
def test_error_handler_exception_fallback_logs_error(mock_push, capsys):
    # Install handler via LoggerManager
    with isolated_root_logging():
        LoggerManager(level="DEBUG")
        handler = mock_push.call_args[0][0]
        # Cause a TypeError inside the handler's try block by passing a non-orderable err_class
        handler(object(), 66, "boom")
        err = capsys.readouterr().err
    # The fallback except path logs an error with the generic format
    assert "GDAL(class=" in err
    assert "code=66" in err
    assert "boom" in err
