"""Output guards and manifest verification must fail closed."""

from __future__ import annotations

import json

import pytest

from eldercare.models import PROJECT_ROOT
from eldercare.provenance import (
    artifact_hashes,
    config_hash,
    file_identity,
    manifest_template,
    prepare_output,
    redacted_command,
    verify_result,
    write_manifest,
)


def test_unsafe_output_targets_are_rejected():
    for unsafe in (PROJECT_ROOT, PROJECT_ROOT.parent):
        with pytest.raises(ValueError, match="Refusing --out"):
            prepare_output(str(unsafe), "source", "config", overwrite=True)


def test_legacy_artifacts_archive_outside_output(tmp_path, capsys):
    output = tmp_path / "results"
    output.mkdir()
    (output / "timeline.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="no run.json"):
        prepare_output(str(output), "source", "config", overwrite=False)

    prepared, archive = prepare_output(str(output), "source", "config", overwrite=True)
    assert prepared == output
    assert archive is not None
    assert archive.parent == tmp_path / "_archive"
    assert (archive / "timeline.json").read_text(encoding="utf-8") == "[]"
    assert list(output.iterdir()) == []
    message = capsys.readouterr().out
    assert str(archive) in message
    assert "2 bytes" in message


def test_unknown_file_blocks_archive_without_moving_anything(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    unknown = output / "notes.txt"
    unknown.write_text("personal data", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or non-file"):
        prepare_output(str(output), "source", "config", overwrite=True)
    assert unknown.is_file()
    assert not (tmp_path / "_archive").exists()


def test_different_source_or_config_requires_overwrite(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")
    (output / "run.json").write_text(
        json.dumps({"input": {"source_sha256": "old"}, "config_sha256": "config"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        prepare_output(str(output), "new", "config", overwrite=False)
    with pytest.raises(ValueError, match="provenance mismatch"):
        prepare_output(str(output), "old", "new-config", overwrite=False)
    assert prepare_output(str(output), "old", "config", overwrite=False) == (output, None)


def test_verify_detects_tampered_artifact(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    summary = output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    settings = {"analysis_fps": 4.0}
    manifest = manifest_template(
        ["eldercare.run", "--video", str(source)],
        {
            "kind": "video",
            "video": file_identity(source),
            "source_sha256": file_identity(source)["sha256"],
        },
        settings,
        "GPU",
    )
    assert manifest["config_sha256"] == config_hash(settings)
    manifest["artifacts"] = artifact_hashes(output)
    write_manifest(output, manifest)
    assert verify_result(output) == []
    summary.write_text('{"changed": true}', encoding="utf-8")
    assert any("artifact hash mismatch" in problem for problem in verify_result(output))


def test_sensitive_cli_values_are_redacted():
    assert redacted_command(["--api-key", "secret-value", "--token=abc", "--video", "x"]) == [
        "--api-key",
        "<redacted>",
        "--token=<redacted>",
        "--video",
        "x",
    ]
