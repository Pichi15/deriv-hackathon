"""Rich stdout logging + JSONL run.log writer."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from rich.console import Console

from .config import RUN_LOG_PATH

console = Console()

# Serialises JSONL writes — without this, concurrent appends from 5 workers
# can interleave mid-line and corrupt the log.
_log_lock = threading.Lock()

_LEVEL_STYLE = {
    "INFO": "green",
    "WARN": "yellow",
    "ERROR": "red",
    "QA": "cyan",
}


def _write_jsonl(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with RUN_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(level: str, step: str, message: str, **fields) -> None:
    """Write a single structured event to stdout + output/run.log."""
    level = level.upper()
    style = _LEVEL_STYLE.get(level, "white")
    console.print(f"[{style}][{level}][/] [{step}] {message}")
    _write_jsonl(
        {
            "ts": _now(),
            "level": level,
            "step": step,
            "message": message,
            **fields,
        }
    )


def info(step: str, message: str, **fields) -> None:
    log("INFO", step, message, **fields)


def warn(step: str, message: str, **fields) -> None:
    log("WARN", step, message, **fields)


def error(step: str, message: str, **fields) -> None:
    log("ERROR", step, message, **fields)


def qa(step: str, message: str, **fields) -> None:
    log("QA", step, message, **fields)
