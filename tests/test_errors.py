"""Tests for pyramids.base._errors custom exception classes.

Ensures every exception class inherits from _PyramidsError, str(e) returns
the message, construction logs at DEBUG (not ERROR), and caught exceptions
are still traceable via DEBUG logs.
"""

import logging

import pytest

from pyramids.base._errors import (
    AlignmentError,
    DatasetNotFoundError,
    DriverNotExistError,
    FailedToSaveError,
    FileFormatNotSupportedError,
    NoDataValueError,
    OptionalPackageDoesNotExist,
    OutOfBoundsError,
    ReadOnlyError,
    _PyramidsError,
)

pytestmark = pytest.mark.core

ALL_ERRORS = [
    ReadOnlyError,
    DatasetNotFoundError,
    NoDataValueError,
    AlignmentError,
    DriverNotExistError,
    FileFormatNotSupportedError,
    OptionalPackageDoesNotExist,
    FailedToSaveError,
    OutOfBoundsError,
]


class TestPyramidsErrorBase:
    """Tests for the _PyramidsError base class."""

    def test_str_returns_message(self):
        """str(_PyramidsError) should return the error message.

        Test scenario:
            The base class calls super().__init__(message), so str()
            must return the message, not an empty string.
        """
        exc = _PyramidsError("base error")
        assert str(exc) == "base error", f"Expected 'base error', got '{str(exc)}'"

    def test_args_contains_message(self):
        """_PyramidsError.args should contain the message.

        Test scenario:
            Standard args tuple should have the message as first element.
        """
        exc = _PyramidsError("args test")
        assert exc.args == ("args test",), f"Expected ('args test',), got {exc.args}"

    def test_logs_debug_with_class_name(self, caplog):
        """_PyramidsError should log at DEBUG with the class name prefix.

        Test scenario:
            The log message should include '_PyramidsError: <message>' for
            traceability.
        """
        with caplog.at_level(logging.DEBUG):
            _PyramidsError("base debug test")
        assert (
            "_PyramidsError: base debug test" in caplog.text
        ), f"Expected '_PyramidsError: base debug test' in log, got: {caplog.text}"

    def test_inherits_from_exception(self):
        """_PyramidsError should inherit from Exception.

        Test scenario:
            The base class itself must be a subclass of Exception.
        """
        assert issubclass(
            _PyramidsError, Exception
        ), "_PyramidsError should inherit from Exception"


class TestExceptionHierarchy:
    """Tests for all 9 concrete exception classes."""

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_str_returns_message(self, exc_class):
        """str(exception) should return the error message.

        Test scenario:
            Construct the exception with a message and verify str()
            returns it (not an empty string).
        """
        msg = f"test message for {exc_class.__name__}"
        exc = exc_class(msg)
        assert (
            str(exc) == msg
        ), f"str({exc_class.__name__}) should return the message, got '{str(exc)}'"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_inherits_from_pyramids_error(self, exc_class):
        """All custom exceptions should inherit from _PyramidsError.

        Test scenario:
            Every concrete exception must be a subclass of the base class,
            enabling catch-all with `except _PyramidsError`.
        """
        assert issubclass(
            exc_class, _PyramidsError
        ), f"{exc_class.__name__} should inherit from _PyramidsError"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_inherits_from_exception(self, exc_class):
        """All custom exceptions should inherit from Exception.

        Test scenario:
            Verify the class is a subclass of Exception so it can be
            caught with `except Exception`.
        """
        assert issubclass(
            exc_class, Exception
        ), f"{exc_class.__name__} should inherit from Exception"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_raise_and_catch(self, exc_class):
        """Each exception should be raisable and catchable by its type.

        Test scenario:
            Raise the exception and catch it by its specific type,
            verifying the message is preserved through the raise/catch cycle.
        """
        msg = f"catch test for {exc_class.__name__}"
        with pytest.raises(exc_class, match="catch test"):
            raise exc_class(msg)

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_catchable_as_pyramids_error(self, exc_class):
        """Each exception should be catchable via `except _PyramidsError`.

        Test scenario:
            Raise a concrete exception and catch it using the base class.
            This verifies the inheritance chain works for catch-all handlers.
        """
        msg = f"base catch for {exc_class.__name__}"
        with pytest.raises(_PyramidsError, match="base catch"):
            raise exc_class(msg)

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_logs_debug_on_construction(self, exc_class, caplog):
        """Exception construction should log at DEBUG level (not ERROR).

        Test scenario:
            Create an exception and verify a DEBUG record is emitted
            (for traceability) but no ERROR record (to avoid log pollution).
        """
        with caplog.at_level(logging.DEBUG):
            exc_class("trace this message")
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert (
            len(debug_records) >= 1
        ), f"{exc_class.__name__} should log at DEBUG on construction"
        assert (
            "trace this message" in caplog.text
        ), "DEBUG log should contain the exception message"
        assert len(error_records) == 0, (
            f"{exc_class.__name__} should NOT log at ERROR on construction, "
            f"but found {len(error_records)} ERROR records"
        )

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_debug_log_includes_class_name(self, exc_class, caplog):
        """DEBUG log message should include the exception class name.

        Test scenario:
            The log format is '{ClassName}: {message}' for easy grep/filtering.
        """
        with caplog.at_level(logging.DEBUG):
            exc_class("class name test")
        expected = f"{exc_class.__name__}: class name test"
        assert (
            expected in caplog.text
        ), f"Expected '{expected}' in log, got: {caplog.text}"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_caught_exception_still_logs_debug(self, exc_class, caplog):
        """Caught-and-handled exceptions should still produce a DEBUG log.

        Test scenario:
            Raise and catch the exception inside a try/except. The DEBUG
            log should still appear because it's emitted at construction
            time, before the exception is raised or caught.
        """
        with caplog.at_level(logging.DEBUG):
            try:
                raise exc_class("caught but logged")
            except exc_class:
                pass
        assert (
            "caught but logged" in caplog.text
        ), f"Caught {exc_class.__name__} should still produce DEBUG log"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_args_tuple_contains_message(self, exc_class):
        """Exception.args should contain the message.

        Test scenario:
            The standard `args` tuple should have the message as its
            first element, enabling framework-level error reporting.
        """
        msg = f"args test for {exc_class.__name__}"
        exc = exc_class(msg)
        assert exc.args == (msg,), f"Expected args=({msg!r},), got {exc.args}"


class TestErrorsReExport:
    """Tests for the public :mod:`pyramids.errors` re-export surface.

    The module is a thin re-export of the private ``pyramids.base._errors``;
    these tests guard the surface contract (every name in ``__all__`` is
    importable, resolves to a class, and participates in the
    ``PyramidsError`` hierarchy).
    """

    def test_module_importable(self):
        """:mod:`pyramids.errors` imports cleanly.

        Test scenario:
            A downstream user who follows the import advice in the module
            docstring (``from pyramids import errors``) must not see
            a deferred :class:`ImportError`.
        """
        import pyramids.errors  # noqa: F401

    def test_pyramids_error_exposed_as_public_name(self):
        """:class:`PyramidsError` is the public alias of the private base.

        Test scenario:
            The module docstring promises ``except errors.PyramidsError``
            works, so the public name must resolve to the underlying
            :class:`_PyramidsError` base class.
        """
        import pyramids.errors as errs

        assert (
            errs.PyramidsError is _PyramidsError
        ), "pyramids.errors.PyramidsError must alias the private base"

    def test_all_entries_are_importable_attributes(self):
        """Every name in ``__all__`` resolves to a module attribute.

        Test scenario:
            A star-import (or static analyser) must not see dangling
            names. Every listed name must be an attribute on the module.
        """
        import pyramids.errors as errs

        missing = [n for n in errs.__all__ if not hasattr(errs, n)]
        assert not missing, f"Names in __all__ with no attribute: {missing}"

    @pytest.mark.parametrize("exc_class", ALL_ERRORS, ids=lambda c: c.__name__)
    def test_each_concrete_class_reexported(self, exc_class):
        """Every private concrete exception is also re-exported publicly.

        Args:
            exc_class: One of the concrete exception classes that lives
                in :mod:`pyramids.base._errors`.

        Test scenario:
            Iterate the full exception hierarchy and assert each one is
            accessible via :mod:`pyramids.errors` by its unqualified
            class name. This is the promise of the public re-export.
        """
        import pyramids.errors as errs

        assert (
            getattr(errs, exc_class.__name__, None) is exc_class
        ), f"{exc_class.__name__} must be re-exported from pyramids.errors"
