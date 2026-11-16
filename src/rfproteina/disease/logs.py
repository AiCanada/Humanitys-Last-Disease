"""Run log for the disease-binder flow.

Every step and every error is appended to one file, so a run that stopped can be
read afterwards. The path is fixed rather than derived from the output directory,
so it is the same answer whichever options a run used: it sits beside the key
store, which ``RFD4_KEY_DIR`` already moves, and ``--log-file`` or
``HUMANITYS_LAST_DISEASE_LOG`` override it outright.

API keys are never written. The panel's key fields are replaced before any
submitted form reaches the log, and a provider error is recorded through the
same redaction the terminal uses.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rfproteina.disease.key_store import key_directory

LOGGER_NAME = "Humanitys-Last-Disease"
LOG_NAME = "Humanitys-Last-Disease.log"
#: Bytes kept before the log rolls over, and how many rolled files are kept.
LOG_BYTES = 2_000_000
LOG_BACKUPS = 3
_SECRET_FIELDS = ("grok_key", "claude_key", "openai_key", "nvidia_key", "openrouter_key")
_REDACTED = "[redacted]"


def log() -> logging.Logger:
    """The run logger. Writes nowhere until :py:func:`start_log` opens a file."""
    return logging.getLogger(LOGGER_NAME)


def log_path(override: Path | None = None) -> Path:
    """Where the run log goes. ``--log-file`` wins, then ``HUMANITYS_LAST_DISEASE_LOG``."""
    if override is not None:
        return override
    chosen = os.environ.get("HUMANITYS_LAST_DISEASE_LOG", "").strip()
    if chosen:
        return Path(chosen)
    return key_directory() / LOG_NAME


def start_log(override: Path | None = None) -> Path | None:
    """Open the run log and return its path, or None when it cannot be opened.

    A log that cannot be written is reported once and then skipped, because a
    read-only directory is not a reason to refuse the run itself.
    """
    path = log_path(override)
    logger = log()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=LOG_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"The run log could not be opened at {path}: {exc}")
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return path


def stop_log() -> None:
    """Close the run log. Leaving a rolled handler open would hold the file.

    ``propagate`` goes back with the handlers. ``start_log`` turns it off so a
    run's records are not repeated by whatever configured the root logger, and
    leaving it off outlived the log it was for: afterwards this logger had no
    handler of its own and could reach none above it, so every record fell
    through to logging's last-resort stderr and was invisible to anything
    watching the root -- an embedder's handler, or a test's ``caplog``, which is
    how this surfaced.
    """
    logger = log()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True


def redact_fields(fields: dict[str, list[str]]) -> dict[str, str]:
    """A submitted panel form as one loggable mapping, with every key replaced.

    A key field is recorded as present or absent and never by value, so the log
    shows that a key was typed without showing the key.
    """
    cleaned: dict[str, str] = {}
    for name in sorted(fields):
        values = [value for value in fields[name] if value.strip()]
        if name in _SECRET_FIELDS:
            cleaned[name] = _REDACTED if values else ""
            continue
        cleaned[name] = ",".join(values)
    return cleaned
