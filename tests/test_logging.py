"""Machine-readable diagnostics do not alter the human CLI report."""
from __future__ import annotations

import json
import logging

import pytest

from eldercare.logging_utils import JsonFormatter, validate_log_file


def test_json_log_record_has_required_fields():
    record = logging.LogRecord(
        name="eldercare.run", level=logging.INFO, pathname=__file__, lineno=1,
        msg="Model ready", args=(), exc_info=None,
    )
    record.event = "model_ready"
    record.model = "pose"
    parsed = json.loads(JsonFormatter().format(record))
    assert set(("timestamp", "level", "component", "event", "message")) <= parsed.keys()
    assert parsed["event"] == "model_ready"
    assert parsed["model"] == "pose"


def test_log_path_must_be_safe_and_recognizable(tmp_path):
    output = tmp_path / "results"
    validate_log_file(str(output / "run.log"), output)
    with pytest.raises(ValueError, match="named run.log"):
        validate_log_file(str(output / "private.jsonl"), output)
    with pytest.raises(ValueError, match="suffix"):
        validate_log_file(str(tmp_path / ".env"), output)
