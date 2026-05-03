"""
Configuration utilities for the pyramids package.

This module exposes helpers to:
- Configure application logging with colored console output.
- Load package configuration from YAML.
- Initialize GDAL/OGR (errors routed to Python logging).
- Discover and export environment variables (e.g., GDAL_DRIVER_PATH) across platforms.

Examples:
- Set up logging and route GDAL errors into Python logging

    ```python

    >>> from pyramids.base.config import Config
    >>> cfg = Config(level="DEBUG")  # doctest: +SKIP
    >>> # - Creates a colored console handler and optionally a file handler
    >>> # - Installs a GDAL error handler that forwards messages to logging

    ```

- Load configuration values from the default config.yaml

    ```python

    >>> from pyramids.base.config import Config
    >>> cfg = Config(level="WARNING")  # doctest: +SKIP
    >>> settings = cfg.config  # doctest: +SKIP
    >>> isinstance(settings, dict)  # doctest: +SKIP
    True

    ```

See Also:
- Config: Main entry point to configure logging and GDAL.
- LoggerManager: Internal helper that performs logging configuration.
- ColorFormatter: Adds ANSI colors to console logs.
"""

from __future__ import annotations

import copy as _copy
import logging
import os
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from osgeo import gdal, ogr

from pyramids import __path__ as root_path


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to the log level name for console output.

    This formatter wraps the levelname (e.g., INFO, WARNING) with an ANSI color
    escape sequence appropriate for the record's level. It is intended for
    console handlers and leaves the message text untouched.

    Examples:
    - Demonstrate how the formatter colors the levelname

        ```python
        >>> import logging
        >>> from pyramids.base.config import ColorFormatter  # doctest: +SKIP
        >>> logger = logging.getLogger("example.color")
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
        >>> logger.addHandler(handler)
        >>> logger.setLevel(logging.INFO)
        >>> logger.info("hello")  # doctest: +SKIP
        INFO - hello

        ```

    See Also:
        - Config.setup_logging: Uses this formatter for console logging.
    """

    RESET = "\x1b[0m"
    LEVEL_COLORS = {
        logging.DEBUG: "\x1b[36m",  # Cyan
        logging.INFO: "\x1b[32m",  # Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",  # Red
        logging.CRITICAL: "\x1b[35m",  # Magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record by applying a color to its level name.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted message where the levelname is wrapped in an ANSI
                color escape sequence suitable for consoles.

        Raises:
            None: This method does not raise exceptions on its own.

        Examples:
        - Use the formatter with a logger to colorize level names

            ```python

            >>> import logging
            >>> from pyramids.base.config import ColorFormatter
            >>> handler = logging.StreamHandler()
            >>> handler.setFormatter(ColorFormatter('%(levelname)s - %(message)s'))
            >>> logger = logging.getLogger('example.color.format')
            >>> _ = [logger.removeHandler(h) for h in list(logger.handlers)]
            >>> logger.addHandler(handler)
            >>> logger.setLevel(logging.WARNING)
            >>> logger.warning('warn')  # doctest: +SKIP

            ```
        """
        colored = _copy.copy(record)
        color = self.LEVEL_COLORS.get(record.levelno, "")
        colored.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(colored)


class LoggerManager:
    """Encapsulates logging setup and GDAL error handler installation for Pyramids.

    This helper centralizes all logging-related concerns so that the public
    configuration class can remain small. It configures a colored console handler,
    an optional file handler, and redirects GDAL errors to the logging subsystem.

    Examples:
    - Basic usage to configure logging and register the GDAL error handler

        ```python

        >>> from pyramids.base.config import LoggerManager
        >>> _ = LoggerManager(level="INFO")  # doctest: +SKIP
        2025-09-09 23:01:39 | INFO | pyramids.base.config | Logging is configured.
        >>> # - Creates a console handler with colorized levels
        >>> # - Optionally creates a file handler when log_file is provided
        >>> # - Installs a GDAL error handler that forwards messages to logging

        ```

    See Also:
        - ColorFormatter: Used by the console handler for colored levels.
        - Config.setup_logging: Public API that delegates to this class.
    """

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"
    LEVELS = ["FATAL", 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

    def __init__(
        self,
        level: int | str = logging.INFO,
        log_file: str | Path | None = None,
    ):
        """Create a LoggerManager and configure logging.

        Args:
            level (int | str, optional): Logging level as an int (e.g., logging.INFO)
                or as a case-insensitive string ("DEBUG", "INFO", ...). Defaults to logging.INFO.
            log_file (str | pathlib.Path | None, optional): Optional path to a log file to
                also write logs to. If None, no file handler is added. Defaults to None.

        Raises:
            ValueError: If an invalid level string is provided (e.g., "VERBOS").

        Examples:
        - Configure logging at DEBUG level and write to a file

            ```python

            >>> from pyramids.base.config import LoggerManager
            >>> _ = LoggerManager(level="DEBUG", log_file="pyramids.log")  # doctest: +SKIP
            2025-09-09 23:02:23 | INFO | pyramids.base.config | Logging is configured.

            ```
        """
        self._setup_logging(level=level, log_file=log_file)
        self._set_error_handler()

    def _setup_logging(
        self,
        level: int | str = logging.INFO,
        log_file: str | Path | None = None,
    ) -> None:
        """Configure application-wide logging for Pyramids.

        Args:
            level (int | str, optional): Logging level as an int or one of
                "FATAL", "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG" (case-insensitive).
                Defaults to logging.INFO.
            log_file (str | pathlib.Path | None, optional): Optional path to a log file to
                write logs to in addition to console output. If None, no file is used.

        Returns:
            None: This method configures the root logger in-place.

        Raises:
            ValueError: If an invalid level string is provided.
            OSError: If the file handler cannot be created (e.g., invalid path or permissions).

        Examples:
        - Configure only console logging at WARNING level
            ```python
            >>> from pyramids.base.config import LoggerManager
            >>> _ = LoggerManager(level="WARNING")  # doctest: +SKIP
            ```

        - Configure console and file logging
            ```python
            >>> from pyramids.base.config import LoggerManager
            >>> _ = LoggerManager(level="INFO", log_file="pyramids.log")  # doctest: +SKIP
            ```

        See Also:
            ColorFormatter: Colorizes level names for console output.
        """
        # Normalize level
        if isinstance(level, str):
            if level.upper() not in self.LEVELS:
                raise ValueError(f"Invalid log level: {level}")
            level = getattr(logging, level.upper(), logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Determine if a console handler already exists
        console_handler = None
        file_handler_exists_for = set()
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                console_handler = h
            if isinstance(h, logging.FileHandler):
                try:
                    file_handler_exists_for.add(Path(h.baseFilename))
                except Exception:
                    pass  # nosec B110

        # Create or update console handler
        if console_handler is None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(
                ColorFormatter(fmt=self.FMT, datefmt=self.DATE_FMT)
            )
            root_logger.addHandler(console_handler)
        else:
            console_handler.setLevel(level)
            # Always ensure colored formatter on console
            if not isinstance(console_handler.formatter, ColorFormatter):
                console_handler.setFormatter(
                    ColorFormatter(fmt=self.FMT, datefmt=self.DATE_FMT)
                )

        # Create file handler if requested and not already present
        if log_file is not None:
            log_file_path = Path(log_file)
            if log_file_path not in file_handler_exists_for:
                fh = logging.FileHandler(log_file_path, encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(logging.Formatter(fmt=self.FMT, datefmt=self.DATE_FMT))
                root_logger.addHandler(fh)

        # Reduce noise from common third-party libraries
        for noisy in ("fiona", "rasterio", "shapely", "matplotlib", "urllib3", "osgeo"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Announce configuration via the module logger so tests can assert on it
        logging.getLogger(__name__).info("Logging is configured.")

    @staticmethod
    def _set_error_handler() -> None:
        """Install a GDAL error handler that forwards messages to logging.

        The handler maps GDAL error classes to Python logging levels. Messages below
        CE_Warning are printed to stdout to preserve expected behaviors in some
        GDAL workflows and tests.
        """
        # Use a child of the module logger so it inherits the handlers/format configured in setup_logging()
        log = logging.getLogger(__name__).getChild("gdal")

        def gdal_error_handler(err_class, err_num, err_msg):
            """Error handler for GDAL mapped to logging levels."""
            try:
                # For error classes lower than CE_Warning, print to stdout (expected by tests)
                if err_class is not None and err_class < getattr(gdal, "CE_Warning", 2):
                    print(
                        f"GDAL error (class {err_class}, number {err_num}): {err_msg}"
                    )
                    return

                # Map GDAL error classes to logging levels
                if err_class == gdal.CE_Debug:
                    log.debug(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Warning:
                    log.warning(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Failure:
                    log.error(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Fatal:
                    log.critical(f"GDAL[{err_num}] {err_msg}")
                else:
                    log.error(f"GDAL(class={err_class}, code={err_num}) {err_msg}")
            except Exception:
                # Fallback to error level if mapping fails for any reason
                log.error(f"GDAL(class={err_class}, code={err_num}) {err_msg}")

        gdal.PushErrorHandler(gdal_error_handler)


@dataclass
class EnvironmentVariables:
    """Utility helpers to inspect and modify PATH-related environment variables.

    This small helper provides convenient accessors for the PATH string and its
    components, as well as a prepend operation; combine with if_exists() to avoid
    duplicates.

    Examples:
    - Create the helper and inspect PATH entries

        ```python

        >>> from pyramids.base.config import EnvironmentVariables
        >>> env = EnvironmentVariables()
        >>> isinstance(env.paths, list)
        True

        ```

    See Also:
        - Plugins: Uses EnvironmentVariables to adjust PATH when GDAL bins are found.
    """

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    def path(self) -> str:
        """Return the raw PATH environment variable.

        Returns:
            str: The current value of the PATH environment variable, or an empty
                string if not set.

        Examples:
        - Read PATH as a string

            ```python
            >>> from pyramids.base.config import EnvironmentVariables
            >>> ev = EnvironmentVariables()
            >>> isinstance(ev.path, str)
            True

            ```
        """
        return os.environ.get("PATH", "")

    @property
    def paths(self) -> list[str]:
        """Return PATH components as a list, split by the OS separator.

        Returns:
            list[str]: A list of absolute/relative directories contained in PATH.

        Examples:
        - Split PATH into components

            ```python
            >>> from pyramids.base.config import EnvironmentVariables
            >>> ev = EnvironmentVariables()
            >>> isinstance(ev.paths, list)
            True

            ```
        """
        path = self.path
        paths = path.split(os.pathsep) if path else []
        return paths

    def if_exists(self, path: str | Path) -> bool:
        """Check whether a directory exists in PATH.

        Args:
            path (str | pathlib.Path): Directory to look for.

        Returns:
            bool: True if the directory is already present in PATH, False otherwise.

        Examples:
        - Check for a directory in PATH

            ```python
            >>> from pyramids.base.config import EnvironmentVariables
            >>> env = EnvironmentVariables()
            >>> env.if_exists("C:/") in (True, False)
            True

            ```
        """
        return str(path) in self.paths

    def prepend(self, path: str | Path) -> None:
        """Prepend a directory to PATH if not already present.

        Args:
            path (str | pathlib.Path): Directory to prepend to PATH.

        Examples:
            - Prepend a directory to PATH safely
                ```python
                >>> import os
                >>> from pyramids.base.config import EnvironmentVariables
                >>> env = EnvironmentVariables()
                >>> original = env.path
                >>> env.prepend("C:/example/bin")
                >>> print(env.paths[0])
                C:/example/bin

                ```
        """
        p = str(path)
        os.environ["PATH"] = p + (os.pathsep + self.path)
        self.logger.debug(f"Prepended to PATH: {p}")


@dataclass
class Plugins:
    """Discover and export GDAL-related paths within a Python site-packages tree.

    Given a site-packages directory, this helper resolves conventional GDAL
    locations used by Conda-forge (Library/Lib/gdalplugins, Library/bin, etc.)
    and can export them into environment variables suitable for GDAL loading.

    Args:
        site_packages_path (str | pathlib.Path): Root path to a site-packages directory
            (e.g., one of values from site.getsitepackages()).

    Attributes:
        plugins_path (pathlib.Path | None): Expected GDAL plugins directory.
        bin_path (pathlib.Path | None): Expected DLL bin directory.
        data_path (pathlib.Path | None): Expected GDAL data directory.
        proj_path (pathlib.Path | None): Expected PROJ data directory.

    Examples:
        - Discover GDAL paths and export to the environment
            ```python
            >>> import site
            >>> from pyramids.base.config import Plugins
            >>> sp = next(iter(site.getsitepackages()), None)  # doctest: +SKIP
            >>> if sp:  # doctest: +SKIP
            ...     p = Plugins(site_packages_path=sp)  # doctest: +SKIP
            ...     _ = p.check_path()  # doctest: +SKIP

            ```

    See Also:
        - EnvironmentVariables: Used to manage PATH updates.
        - Config.dynamic_env_variables: Uses this class to probe for GDAL on Windows.
    """

    site_packages_path: str | Path
    plugins_path: Path = field(init=False)
    bin_path: Path = field(init=False)
    data_path: Path = field(init=False)
    proj_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived GDAL-related paths based on site_packages_path."""
        self.logger = logging.getLogger(__name__)
        self.plugins_path = Path(self.site_packages_path) / "Library/Lib/gdalplugins"
        base_path = Path(self.site_packages_path) / "Library"
        self.bin_path = base_path / "bin"
        self.data_path = base_path / "share" / "gdal"
        self.proj_path = base_path / "share" / "proj"

    def check_path(self) -> Path | None:
        """Probe known locations under site-packages and set GDAL env variables.

        This method checks for the presence of the GDAL plugins folder and, if
        found, sets the following environment variables where appropriate:
        - GDAL_DRIVER_PATH
        - Optionally prepends Library/bin to PATH (so GDAL plugin DLLs resolve)
        - Optionally sets GDAL_DATA and PROJ_LIB if those directories exist

        Returns:
            pathlib.Path | None: The detected plugins_path if found, otherwise None.

        Examples:
            - Probe a site-packages tree and set environment variables
                ```python
                >>> import site
                >>> from pyramids.base.config import Plugins
                >>> sp = next(iter(site.getsitepackages()), None)  # doctest: +SKIP
                >>> if sp:  # doctest: +SKIP
                ...     p = Plugins(site_packages_path=sp)  # doctest: +SKIP
                ...     _ = p.check_path()  # doctest: +SKIP

                ```
        """
        if self.plugins_path.exists():
            os.environ["GDAL_DRIVER_PATH"] = str(self.plugins_path)
            self.logger.debug(f"GDAL_DRIVER_PATH set to: {self.plugins_path}")
            if self.bin_path.exists():
                env_vars = EnvironmentVariables()
                bin_str = str(self.bin_path)

                if not env_vars.if_exists(bin_str):
                    env_vars.prepend(bin_str)

            # Optionally set GDAL_DATA and PROJ_LIB
            if self.data_path.exists() and os.environ.get("GDAL_DATA") != str(
                self.data_path
            ):
                os.environ["GDAL_DATA"] = str(self.data_path)
                self.logger.debug(f"GDAL_DATA set to: {self.data_path}")
            if self.proj_path.exists() and os.environ.get("PROJ_LIB") != str(
                self.proj_path
            ):
                os.environ["PROJ_LIB"] = str(self.proj_path)
                self.logger.debug(f"PROJ_LIB set to: {self.proj_path}")

            path = self.plugins_path
        else:
            path = None

        return path


class Config:
    """High-level configuration entry point for logging and GDAL environment.

    This class orchestrates:
    - Loading user/package configuration from YAML.
    - Configuring Python logging (console with colors, optional file).
    - Initializing GDAL/OGR and discovering GDAL-related environment variables.

    Args:
        level (int | str, optional): Logging level. Defaults to logging.INFO.
        log_file (str | pathlib.Path | None, optional): Optional path to a file for logs.
        config_file (str, optional): YAML filename to read from the package's base folder.
            Defaults to "config.yaml".

    Attributes:
        config (dict): Parsed configuration values from YAML.
        logger (logging.Logger): Module logger configured by setup_logging().

    Examples:
    - Create a configuration with console logging

        ```python

        >>> from pyramids.base.config import Config  # doctest: +SKIP
        >>> cfg = Config(level="INFO")  # doctest: +SKIP
        2025-09-09 23:10:28 | INFO | pyramids.base.config | Logging is configured.
        >>> print(cfg.config)  # doctest: +SKIP
        {'gdal': {'GDAL_CACHEMAX': '512',
          'GDAL_PAM_ENABLED': 'YES',
          'GDAL_VRT_ENABLE_PYTHON': 'YES',
          'GDAL_TIFF_INTERNAL_MASK': 'NO'},
         'ogr': {'OGR_SRS_PARSER': 'strict'},
         'logging': {'level': 'DEBUG',
          'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          'file': 'pyramids.log'}}

        ````

    See Also:
        - LoggerManager: Implements logging configuration details.
        - Config.initialize_gdal: Applies GDAL/OGR options and registers drivers.
    """

    def __init__(
        self,
        level: int | str = logging.INFO,
        log_file: str | Path | None = None,
        config_file="config.yaml",
    ):
        """Construct a Config, load YAML, configure logging, and initialize GDAL.

        Args:
            level (int | str, optional):
                Logging level (e.g., logging.INFO or "DEBUG").
            log_file (str | pathlib.Path | None, optional):
                Optional path to a log file.
            config_file (str, optional):
                Name of the YAML configuration file shipped in pyramids/base. Defaults to "config.yaml".

        Raises:
            FileNotFoundError: If the YAML file cannot be found.
            yaml.YAMLError: If parsing the YAML fails.
            Exception: Any GDAL-related error if GDAL initialization fails.

        Examples:
            - Create a configuration with INFO logging
                ```python
                >>> from pyramids.base.config import Config  # doctest: +SKIP
                >>> cfg = Config(level="INFO")  # doctest: +SKIP
                2025-09-09 23:11:52 | INFO | pyramids.base.config | Logging is configured.

                ```
        """
        self.setup_logging(level=level, log_file=log_file)
        self.config_file = config_file
        self.config = self.load_config()
        self.initialize_gdal()

    def load_config(self):
        """Load configuration from the package YAML file.

        The YAML file is expected to live under pyramids/base/<config_file>.

        Returns:
            dict[str, Any]: Parsed configuration values.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If the YAML cannot be parsed.

        Examples:
            - Load settings and verify the result is a dictionary
                ```python
                >>> from pyramids.base.config import Config
                >>> cfg = Config(config_file="config.yaml")
                >>> print(cfg.config)  # doctest: +SKIP
                {'gdal': {'GDAL_CACHEMAX': '512',
                  'GDAL_PAM_ENABLED': 'YES',
                  'GDAL_VRT_ENABLE_PYTHON': 'YES',
                  'GDAL_TIFF_INTERNAL_MASK': 'NO'},
                 'ogr': {'OGR_SRS_PARSER': 'strict'},
                 'logging': {'level': 'DEBUG',
                  'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                  'file': 'pyramids.log'}}

                ```
        """
        config_file = Path(root_path[0]) / "base/data" / self.config_file
        with open(config_file) as file:
            return yaml.safe_load(file)

    def initialize_gdal(self):
        """Initialize GDAL/OGR options and register drivers.

        This method:
        - Enables exceptions in GDAL and OGR.
        - Applies options from the loaded YAML under keys `gdal` and `ogr`.
        - Attempts to detect and export the GDAL_DRIVER_PATH using dynamic_env_variables().
        - Calls gdal.AllRegister() to ensure drivers are available.

        Raises:
            RuntimeError: If GDAL initialization encounters issues (propagated from GDAL).

        Examples:
            - Initialize GDAL and then query GDAL_DRIVER_PATH
                ```python
                >>> from pyramids.base.config import Config
                >>> cfg = Config()  # doctest: +SKIP
                >>> # Depending on the environment, GDAL_DRIVER_PATH may or may not be set
                >>> _ = cfg.initialize_gdal()  # doctest: +SKIP

                ```

        See Also:
            - Config.dynamic_env_variables: Locates plugin directories across platforms.
        """
        # Enable exceptions so GDAL/OGR raise on error instead of printing to stdout
        gdal.UseExceptions()
        ogr.UseExceptions()
        # Apply GDAL/OGR options from configuration
        for key, value in self.config.get("gdal", {}).items():
            gdal.SetConfigOption(key, value)
        for key, value in self.config.get("ogr", {}).items():
            gdal.SetConfigOption(key, value)

        gdal_plugins_path = self.dynamic_env_variables()

        if gdal_plugins_path:
            gdal.SetConfigOption("GDAL_DRIVER_PATH", str(gdal_plugins_path))
        gdal.AllRegister()

    def set_env_conda(self) -> Path | None:
        """Set GDAL-related environment variables in a Conda environment.

        This method looks up the active Conda environment (via CONDA_PREFIX) and
        configures:
        - GDAL_DRIVER_PATH (if Library/lib/gdalplugins exists)
        - PATH (prepends Library/bin so dependent DLLs can be found)
        - GDAL_DATA and PROJ_LIB (if their directories exist)

        Returns:
            pathlib.Path | None: The GDAL plugins path if found, else None.

        Examples:
            - Configure environment variables when running inside Conda
                ```python
                >>> from pyramids.base.config import Config  # doctest: +SKIP
                >>> cfg = Config()  # doctest: +SKIP
                >>> _ = cfg.set_env_conda()  # doctest: +SKIP

                ```
        """
        conda_prefix = os.getenv("CONDA_PREFIX")

        if not conda_prefix:
            self.logger.debug("CONDA_PREFIX is not set. Ensure Conda is activated.")
            return None

        conda_prefix_path = Path(conda_prefix)
        gdal_plugins_path = conda_prefix_path / "Library" / "lib" / "gdalplugins"
        library_bin_path = conda_prefix_path / "Library" / "bin"
        gdal_data_path = conda_prefix_path / "Library" / "share" / "gdal"
        proj_lib_path = conda_prefix_path / "Library" / "share" / "proj"

        # Set GDAL plugins path
        if gdal_plugins_path.exists():
            os.environ["GDAL_DRIVER_PATH"] = str(gdal_plugins_path)
            self.logger.debug(f"GDAL_DRIVER_PATH set to: {gdal_plugins_path}")
        else:
            self.logger.debug(
                f"GDAL plugins path not found at: {gdal_plugins_path}. Please check your GDAL installation."
            )

        # Ensure dependent DLLs are on PATH (fixes error 126 on Windows when loading plugins like HDF5)
        if library_bin_path.exists():
            current_path = os.environ.get("PATH", "")
            bin_str = str(library_bin_path)
            path_parts = current_path.split(os.pathsep) if current_path else []
            if bin_str not in path_parts:
                os.environ["PATH"] = bin_str + (
                    os.pathsep + current_path if current_path else ""
                )
                self.logger.debug(f"Prepended to PATH: {bin_str}")
        else:
            self.logger.debug(f"Library bin path not found at: {library_bin_path}")

        # Optionally set GDAL_DATA and PROJ_LIB if available
        if gdal_data_path.exists():
            if os.environ.get("GDAL_DATA") != str(gdal_data_path):
                os.environ["GDAL_DATA"] = str(gdal_data_path)
                self.logger.debug(f"GDAL_DATA set to: {gdal_data_path}")
        else:
            self.logger.debug(f"GDAL data path not found at: {gdal_data_path}")

        if proj_lib_path.exists():
            if os.environ.get("PROJ_LIB") != str(proj_lib_path):
                os.environ["PROJ_LIB"] = str(proj_lib_path)
                self.logger.debug(f"PROJ_LIB set to: {proj_lib_path}")
        else:
            self.logger.debug(f"PROJ lib path not found at: {proj_lib_path}")

        return gdal_plugins_path if gdal_plugins_path.exists() else None

    def dynamic_env_variables(self) -> Path | None:
        """Locate GDAL plugin directories and export GDAL_DRIVER_PATH.

        The search proceeds in this order:
        - If inside Conda, use set_env_conda().
        - On Windows, probe site.getsitepackages() using Plugins helper.
        - On POSIX, probe common locations like /usr/local/lib/gdalplugins.

        Returns:
            pathlib.Path | None: The detected GDAL plugins path, or None if not found.

        Examples:
            - Attempt to discover plugin directory in the current environment
                ```python
                >>> from pyramids.base.config import Config
                >>> cfg = Config()
                >>> _ = cfg.dynamic_env_variables()  # doctest: +SKIP

                ```

        See Also:
            - Plugins: Windows helper for site-packages probing.
            - Config.set_env_conda: Conda-specific environment configuration.
        """
        # Check if we're in a Conda environment
        gdal_plugins_path = self.set_env_conda()

        if gdal_plugins_path is None:

            # For Windows, check Python site-packages
            if sys.platform == "win32":
                for site_path in site.getsitepackages():
                    plugins = Plugins(site_packages_path=site_path)
                    path = plugins.check_path()
                    if path:
                        gdal_plugins_path = path
            else:

                # Check typical system locations (Linux/MacOS)
                system_paths = [
                    "/usr/local/lib/gdalplugins",
                    "/usr/lib/gdalplugins",
                ]
                for path in system_paths:
                    path = Path(path)
                    if path.exists():
                        os.environ["GDAL_DRIVER_PATH"] = str(path)
                        gdal_plugins_path = path
                        self.logger.debug(f"GDAL_DRIVER_PATH set to: {path}")

                # If the path is not found
                # print("GDAL plugins path could not be found. Please check your GDAL installation.")
        return gdal_plugins_path

    def setup_logging(
        self,
        level: int | str = logging.INFO,
        log_file: str | Path | None = None,
    ):
        """
        Configure application-wide logging for Pyramids by delegating to LoggerManager.

        This method preserves the public API while separating responsibilities. It delegates
        the actual logging configuration to the LoggerManager constructor and then sets
        self.logger and self._logging_configured for convenience/compatibility.
        """
        LoggerManager(level=level, log_file=log_file)
        self._logging_configured = True
        self.logger = logging.getLogger(__name__)
