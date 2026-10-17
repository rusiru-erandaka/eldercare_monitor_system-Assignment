"""Lock the original CV-only outputs; known-bad behavior is intentional here."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eldercare.alerts import apply_vlm_safety_escalations
from eldercare.config import Config
from eldercare.pipeline import load_ticks
from eldercare.vlm import FrameSample, analyze_video

GOLDEN_ROOT = Path(__file__).parent / "golden"
SOURCES = (
    "a7f7143a12bd0d55d420084e9bdd1ad712f12e91923e10222295f119e5dc80d8",
    "b0ec3cc87a2375f89e5520c8d440b0a9f5c913ff135aeebf18d453a77fdfa5dc",
)


def normalized_summary(path: Path) -> bytes:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["analysis"].pop("input_path", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("source_sha", SOURCES)
def test_cv_from_ticks_matches_starting_commit(tmp_path: Path, source_sha: str) -> None:
    fixture = GOLDEN_ROOT / source_sha
    meta = json.loads((fixture / "ticks.meta.json").read_text(encoding="utf-8"))
    assert meta["source_video"]["sha256"] == source_sha
    output = tmp_path / "result"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "eldercare.run",
            "--from-ticks",
            str(fixture / "ticks.jsonl"),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    for name in ("timeline.json", "events.json"):
        assert (output / name).read_bytes() == (fixture / name).read_bytes()
    assert normalized_summary(output / "summary.json") == normalized_summary(
        fixture / "summary.json"
    )


def test_saved_floor_review_escalates_but_never_downgrades() -> None:
    review = json.loads((GOLDEN_ROOT / "vlm_floor_review.json").read_text(encoding="utf-8"))
    records = load_ticks(str(GOLDEN_ROOT / SOURCES[1] / "ticks.jsonl"))
    cfg = Config()

    without_review, cv_decision, _ = apply_vlm_safety_escalations({}, [], "NORMAL", cfg, records)
    assert without_review == []
    assert cv_decision == "NORMAL"

    alerts, decision, escalations = apply_vlm_safety_escalations(
        review, [], cv_decision, cfg, records
    )
    assert decision == "ALERT"
    assert len(alerts) == len(escalations) == 1
    assert alerts[0].rule == "A1_vlm_floor_confirmed"

    _, preserved, _ = apply_vlm_safety_escalations(review, [], "ALERT", cfg, records)
    assert preserved == "ALERT"
    benign_review = {"reviews": [{**review["reviews"][0], "accepted": False}]}
    _, preserved, _ = apply_vlm_safety_escalations(benign_review, [], "ALERT", cfg, records)
    assert preserved == "ALERT"


def test_mock_gemini_client_plans_once_and_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    import types

    from eldercare import vlm

    saved = json.loads((GOLDEN_ROOT / "vlm_floor_review.json").read_text(encoding="utf-8"))
    answer = saved["reviews"][0]["vlm_result"]
    calls: list[dict] = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(text=json.dumps(answer))

    fake_client = types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=generate_content)
    )
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda api_key: fake_client
    fake_types = types.ModuleType("google.genai.types")
    fake_types.Part = types.SimpleNamespace(from_bytes=lambda **kwargs: kwargs)
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs
    fake_types.AutomaticFunctionCallingConfig = lambda **kwargs: kwargs
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setattr(vlm, "load_project_env", lambda: tmp_path / "unused")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        vlm,
        "extract_frames",
        lambda path, times, config: [FrameSample(t, b"fake-jpeg") for t in times],
    )

    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"not-a-real-video")
    records = load_ticks(str(GOLDEN_ROOT / SOURCES[1] / "ticks.jsonl"))
    cache = tmp_path / "vlm_reviews.json"
    first = analyze_video(str(video), records, [], Config(), str(cache), model="mock-gemini")
    assert first["review_count"] == 1
    assert len(calls) == 1
    assert first["reviews"][0]["accepted"] is True
    second = analyze_video(str(video), records, [], Config(), str(cache), model="mock-gemini")
    assert second["cache_used"] is True
    assert len(calls) == 1
