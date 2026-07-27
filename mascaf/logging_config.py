"""Configure logging for MASCAF applications and CLI scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

Level = Union[int, str]

# Loggers that become noisy at DEBUG without adding MASCAF insight.
_DEFAULT_QUIET_LOGGERS = (
    "matplotlib",
    "PIL",
    "trimesh",
    "urllib3",
    "choreographer",
    "kaleido",
    "swctools",
    "asyncio",
    "browser_proc",
)


def configure_logging(
    level: Level = logging.INFO,
    *,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    fmt: str = "%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    quiet_loggers: Optional[Iterable[str]] = _DEFAULT_QUIET_LOGGERS,
    quiet_level: int = logging.WARNING,
) -> None:
    """Set up root logging for MASCAF pipeline runs.

    Configures the root logger with optional console and file handlers,
    sets ``mascaf`` (and submodules) to ``level``, and raises common
    third-party loggers to ``quiet_level`` so DEBUG runs stay readable.

    Parameters
    ----------
    level :
        Root and ``mascaf`` log level (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    log_file :
        Optional path to append log output (parent dirs are created).
    console :
        When ``True``, also emit logs to stderr.
    fmt, datefmt :
        ``logging.Formatter`` strings (timestamp included by default).
    quiet_loggers :
        Logger names to cap at ``quiet_level``. ``None`` uses the default
        quiet list; pass ``()`` to disable quieting entirely.
    quiet_level :
        Level for ``quiet_loggers`` (default ``WARNING``).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if quiet_loggers is None:
        quiet_loggers = _DEFAULT_QUIET_LOGGERS

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    mascaf_logger = logging.getLogger("mascaf")
    mascaf_logger.setLevel(level)

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(quiet_level)
