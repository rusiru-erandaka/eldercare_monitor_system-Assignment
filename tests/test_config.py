"""Configuration values must be validated and intentional."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from eldercare.config import Config
from eldercare.provenance import config_hash
from eldercare.run import build_config, parse_args


def test_defaults_keep_the_original_snapshot_hash():
    assert config_hash(Config().to_dict()) == (
        "2e684cbc923349c461161024c5c663dcd140677f8b1cc5116a0ca241296c5ddb"
    )


def test_yaml_loads_and_cli_has_precedence(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("analysis_fps: 2.0\nperson_conf: 0.4\n", encoding="utf-8")
    args = parse_args(["--video", "clip.mp4", "--config", str(profile), "--analysis-fps", "3.0"])
    config = build_config(args)
    assert config.analysis_fps == 3.0
    assert config.person_conf == 0.4
    assert config.monitor_unknown_sec == Config().monitor_unknown_sec


def test_unknown_key_and_invalid_range_are_rejected(tmp_path):
    profile = tmp_path / "invalid.yaml"
    profile.write_text("not_a_setting: 2\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="not_a_setting"):
        Config.from_yaml(profile)
    with pytest.raises(ValidationError, match="person_conf"):
        Config(person_conf=1.5)
    with pytest.raises(ValidationError, match="analysis_fps"):
        Config(analysis_fps=0)
    with pytest.raises(ValidationError, match="return_start_overlap_max"):
        Config(return_start_overlap_max=0.7)


def test_config_fields_are_used_or_explicitly_reviewed():
    package = Path(__file__).resolve().parents[1] / "eldercare"
    used_attributes: set[str] = set()
    for source in package.rglob("*.py"):
        if source.name == "config.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        used_attributes.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )

    # Reviewed Phase 2 debt; do not silently wire these into current policy:
    # - normal_middle_sit_sec: 10-minute reading/resting policy is descriptive.
    # - exit_long_out_sec: exit confirmation currently uses spatial evidence or
    #   exit_min_out_sec; this 30-second value has no consumer.
    # - min_event_conf: event confidence is metadata, not a risk gate today.
    reviewed_unused = {
        "normal_middle_sit_sec",
        "exit_long_out_sec",
        "min_event_conf",
    }
    unused = set(Config.model_fields) - used_attributes
    assert unused == reviewed_unused, f"Unreviewed/changed unused config fields: {unused}"
