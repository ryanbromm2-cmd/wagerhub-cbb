"""
Logging configuration for the CBB Totals Model.
Provides Rich-formatted console output and rotating file logs.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Try to import Rich; fall back to standard logging if not available
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Try to load dotenv for LOG_LEVEL env var
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "cbb_totals.log"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT = 5

# Noisy third-party loggers to suppress
_SUPPRESS = [
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "charset_normalizer",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "sqlalchemy.dialects",
    "apscheduler",
    "matplotlib",
    "PIL",
    "streamlit",
    "watchdog",
    "asyncio",
]

_INITIALIZED: set[str] = set()


def _get_log_level() -> int:
    """Read log level from environment variable, default INFO."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    return level


def _build_file_handler() -> RotatingFileHandler:
    """Create a rotating file handler with detailed formatting."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)  # File always captures DEBUG+
    return handler


def _build_console_handler() -> logging.Handler:
    """Create a Rich console handler (or fallback StreamHandler)."""
    level = _get_log_level()
    if RICH_AVAILABLE:
        handler = RichHandler(
            level=level,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        # RichHandler uses its own formatter; keep it minimal
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
    return handler


def _suppress_noisy_loggers() -> None:
    """Set WARNING level on known noisy third-party loggers."""
    for name in _SUPPRESS:
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    On first call the root 'cbb_totals' logger is configured with both a
    Rich console handler and a rotating file handler.  Subsequent calls for
    any name under that hierarchy simply return a child logger.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        logging.Logger instance ready for use.

    Example::

        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """
    # Use a project-scoped root logger
    root_name = "cbb_totals"
    root_logger = logging.getLogger(root_name)

    if root_name not in _INITIALIZED:
        _INITIALIZED.add(root_name)

        root_logger.setLevel(logging.DEBUG)  # Handlers control effective level
        root_logger.propagate = False        # Don't bubble to root logger

        # Remove any pre-existing handlers (e.g. from previous imports in tests)
        root_logger.handlers.clear()

        root_logger.addHandler(_build_file_handler())
        root_logger.addHandler(_build_console_handler())

        _suppress_noisy_loggers()

    # For names already under cbb_totals hierarchy, return as-is
    if name.startswith(root_name):
        return logging.getLogger(name)

    # Prefix caller's __name__ so it appears under the project hierarchy
    child_name = f"{root_name}.{name}" if name else root_name
    return logging.getLogger(child_name)


# ── Module-level logger for this file ─────────────────────────────────────────
logger = get_logger(__name__)
