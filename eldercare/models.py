"""Pinned local model assets and SHA-256 verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
MANIFEST = MODEL_ROOT / "models.lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    with MANIFEST.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), dict):
        raise ValueError(f"Invalid model manifest: {MANIFEST}")
    return data


def model_directory(name: str) -> Path:
    entry = load_manifest()["models"].get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Unknown model name: {name}")
    directory = entry.get("directory")
    if not isinstance(directory, str) or Path(directory).name != directory:
        raise ValueError(f"Unsafe model directory in {MANIFEST}: {directory!r}")
    return MODEL_ROOT / directory


def model_xml(name: str) -> Path:
    entry = load_manifest()["models"].get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Unknown model name: {name}")
    xml_files = [filename for filename in entry["files"] if filename.endswith(".xml")]
    if len(xml_files) != 1:
        raise ValueError(f"Expected exactly one XML for {name}")
    return model_directory(name) / xml_files[0]


def verify_model(name: str) -> dict[str, str]:
    entry = load_manifest()["models"].get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Unknown model name: {name}")
    directory = model_directory(name)
    files = entry.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"No files listed for {name} in {MANIFEST}")
    verified: dict[str, str] = {}
    for filename, expected in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe model filename in {MANIFEST}: {filename!r}")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Pinned {name} model file missing: {path}. Run scripts/fetch_models.py.")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Pinned {name} model hash mismatch: {path} "
                f"(expected {expected}, got {actual}). Refusing to load it.")
        verified[str(path.resolve())] = actual
    return verified
