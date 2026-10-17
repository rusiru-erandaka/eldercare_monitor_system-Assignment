"""Small JSON-lines logger for run diagnostics; CLI reports remain on stdout."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FIELDS = (
    "run_id", "track_id", "model", "requested_device", "execution_devices",
    "path", "sha256", "tick_count", "elapsed_sec", "status", "error_type",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name,
            "event": getattr(record, "event", "message"),
            "message": record.getMessage(),
        }
        for name in _FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    logger = logging.getLogger("eldercare")
    logger.setLevel(level.upper())
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        path = Path(log_file).expanduser()
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def validate_log_file(log_file: str | None, output: Path) -> None:
    if not log_file:
        return
    path = Path(log_file).expanduser().absolute()
    if path.suffix not in {".log", ".jsonl"}:
        raise ValueError("--log-file must have a .log or .jsonl suffix")
    if path.parent.resolve() == output.resolve() and path.name != "run.log":
        raise ValueError("A log inside --out must be named run.log")
    if path.parent.resolve() != output.resolve() and not path.parent.is_dir():
        raise ValueError(f"Log directory does not exist: {path.parent}")
