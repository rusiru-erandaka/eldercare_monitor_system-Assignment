"""Pinned model files must be present and byte-for-byte verified."""
from __future__ import annotations

import hashlib
import json

import pytest

from eldercare import models


def test_verified_model_rejects_corruption(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    detector = model_root / "detector"
    detector.mkdir(parents=True)
    xml = detector / "model.xml"
    xml.write_bytes(b"original model")
    digest = hashlib.sha256(xml.read_bytes()).hexdigest()
    manifest = model_root / "models.lock.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "models": {
                "detector": {
                    "directory": "detector",
                    "files": {"model.xml": digest},
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "MODEL_ROOT", model_root)
    monkeypatch.setattr(models, "MANIFEST", manifest)

    assert models.verify_model("detector") == {str(xml.resolve()): digest}
    xml.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        models.verify_model("detector")
