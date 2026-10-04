import json

from eldercare.config import Config
from eldercare.alerts import apply_vlm_safety_escalations
from eldercare.events import Event
from eldercare.pipeline import save_ticks, validate_ticks_video
from eldercare.schema import TickRecord
from eldercare.vlm import (FrameSample, analyze_video, plan_reviews,
                           report_summary)


def tick(t, state, *, confidence=0.9, overlap=0.8, clipped=False):
  return TickRecord(
      t=t, state=state, confidence=confidence, observed=True, track_id=1,
      bed_known=True, bed_overlap=overlap, body_clipped=clipped,
      bbox=[10, 10, 80, 100], bed=[0, 0, 100, 100])


class FakeBackend:
  model = "fake-gemini"

  def __init__(self):
    self.calls = 0

  def analyze(self, prompt, frames):
    self.calls += 1
    assert "before/current/after" in prompt
    assert len(frames) <= 5
    return {
        "observation": "The elder moves from the bed toward the room.",
        "previous_segment_finding": "The elder was lying in bed.",
        "current_segment_finding": "The elder becomes upright beside the bed.",
        "following_segment_finding": "The elder walks away from the bed.",
        "conclusion": "BED_EXIT",
        "resolved_activity": "WALKING",
        "person_location": "OUTSIDE_BED",
        "confidence": 0.91,
        "evidence": ["Bed overlap decreases", "Movement is outward"],
        "needs_more_context": False,
      "human_review_required": False,
    }


def test_event_json_excludes_private_runtime_metadata():
  event = Event(
      "bed_exit", 1.0, 1.5, "SITTING_ON_BED", "OUT_OF_BED", 0.8,
      extra={"track_id": 1, "_out_run": (1.5, 20.0)})

  payload = event.to_dict()

  assert payload["track_id"] == 1
  assert "_out_run" not in payload
  assert event.extra["_out_run"] == (1.5, 20.0)


def exit_records():
  return [
      tick(0.0, "LYING_IN_BED", overlap=0.95),
      tick(0.5, "SITTING_ON_BED", overlap=0.80),
      tick(1.0, "UNKNOWN", confidence=0.1, overlap=0.60, clipped=True),
      tick(1.5, "WALKING", overlap=0.25),
  ]


def test_review_planner_uses_one_context_window_for_event_and_unknown():
  cfg = Config()
  event = Event("bed_exit", 1.0, 1.5, "SITTING_ON_BED", "OUT_OF_BED", 0.8)
  windows = plan_reviews(exit_records(), [event], cfg)
  assert len(windows) == 1
  assert windows[0].expected_event == "BED_EXIT"
  assert "unknown_activity" in windows[0].trigger
  assert "deterministic_bed_exit_visual_confirmation" in windows[0].trigger


def test_agent_review_is_structured_cached_and_does_not_mutate_ticks(tmp_path, monkeypatch):
  cfg = Config()
  records = exit_records()
  original = [r.to_dict() for r in records]
  event = Event("bed_exit", 1.0, 1.5, "SITTING_ON_BED", "OUT_OF_BED", 0.8)
  video = tmp_path / "room.mp4"
  video.write_bytes(b"test-video-fingerprint")
  output = tmp_path / "vlm_reviews.json"
  monkeypatch.setattr(
      "eldercare.vlm.extract_frames",
      lambda path, times, config: [FrameSample(t, b"jpeg") for t in times])
  backend = FakeBackend()

  result = analyze_video(
      str(video), records, [event], cfg, str(output), model="fake-gemini",
      backend=backend)
  assert backend.calls == 1
  assert result["reviews"][0]["accepted"] is True
  assert result["reviews"][0]["vlm_result"]["conclusion"] == "BED_EXIT"
  assert result["raw_timeline_changed"] is False
  assert [r.to_dict() for r in records] == original
  assert json.loads(output.read_text())["reviews"][0]["accepted"] is True

  cached = analyze_video(
      str(video), records, [event], cfg, str(output), model="fake-gemini",
      backend=backend)
  assert cached["cache_used"] is True
  assert backend.calls == 1
  summary = report_summary(cached)
  assert summary["accepted_review_count"] == 1
  assert summary["raw_timeline_changed"] is False


def test_confident_routine_timeline_needs_no_api_key_or_backend(tmp_path):
  cfg = Config()
  records = [tick(0.0, "LYING_IN_BED", overlap=0.95),
             tick(0.5, "LYING_IN_BED", overlap=0.95)]
  video = tmp_path / "routine.mp4"
  video.write_bytes(b"routine")
  output = tmp_path / "vlm_reviews.json"
  result = analyze_video(
      str(video), records, [], cfg, str(output), model="fake-gemini")
  assert result["review_count"] == 0
  assert result["reviews"] == []
  assert output.exists()


def test_cached_ticks_are_cryptographically_bound_to_their_video(tmp_path):
  source = tmp_path / "source.mp4"
  other = tmp_path / "other.mp4"
  source.write_bytes(b"waking-video-content")
  other.write_bytes(b"return-video-content")
  ticks = tmp_path / "ticks.jsonl"
  save_ticks(str(ticks), exit_records(), str(source))

  metadata = validate_ticks_video(str(ticks), str(source))
  assert metadata["source_video"]["path"] == str(source)

  try:
    validate_ticks_video(str(ticks), str(other))
  except ValueError as exc:
    message = str(exc)
  else:
    raise AssertionError("different video was accepted for cached ticks")
  assert "Cached ticks/video mismatch" in message
  assert "remove --from-ticks" in message


def test_vlm_contradiction_is_not_automatically_accepted(tmp_path, monkeypatch):
  class ContradictingBackend(FakeBackend):
    def analyze(self, prompt, frames):
      answer = super().analyze(prompt, frames)
      answer["conclusion"] = "BED_RETURN"
      answer["resolved_activity"] = "LYING_IN_BED"
      answer["person_location"] = "IN_BED"
      answer["confidence"] = 0.99
      return answer

  cfg = Config()
  records = exit_records()
  event = Event("bed_exit", 1.0, 1.5, "SITTING_ON_BED", "OUT_OF_BED", 0.8)
  video = tmp_path / "room.mp4"
  video.write_bytes(b"same-video")
  monkeypatch.setattr(
      "eldercare.vlm.extract_frames",
      lambda path, times, config: [FrameSample(t, b"jpeg") for t in times])

  result = analyze_video(
      str(video), records, [event], cfg, str(tmp_path / "review.json"),
      model="fake-gemini", backend=ContradictingBackend())
  review = result["reviews"][0]
  assert review["accepted"] is False
  assert review["cv_vlm_conflict"] is True
  assert review["vlm_result"]["human_review_required"] is True
  assert review["agent_trace"][-1]["finding"] == "CV_VLM_CONFLICT"


def test_clear_vlm_floor_confirmation_escalates_short_clip_to_alert():
  cfg = Config()
  result = {
      "reviews": [{
          "accepted": True,
          "window": {"start_sec": 6.0, "end_sec": 8.0},
          "vlm_result": {
              "conclusion": "LYING_ON_FLOOR",
              "resolved_activity": "LYING_ON_FLOOR",
              "person_location": "OUTSIDE_BED",
              "confidence": 0.95,
              "human_review_required": False,
          },
      }],
  }
  alerts, overall, escalations = apply_vlm_safety_escalations(
      result, [], "NORMAL", cfg)
  assert overall == "ALERT"
  assert len(alerts) == len(escalations) == 1
  assert alerts[0].rule == "A1_vlm_floor_confirmed"
  assert alerts[0].time == 6.0


def test_uncertain_or_low_confidence_floor_result_cannot_raise_alert():
  cfg = Config()
  result = {
      "reviews": [{
          "accepted": False,
          "window": {"start_sec": 6.0},
          "vlm_result": {
              "conclusion": "LYING_ON_FLOOR",
              "resolved_activity": "LYING_ON_FLOOR",
              "person_location": "OUTSIDE_BED",
              "confidence": 0.89,
              "human_review_required": True,
          },
      }],
  }
  alerts, overall, escalations = apply_vlm_safety_escalations(
      result, [], "NORMAL", cfg)
  assert overall == "NORMAL"
  assert alerts == escalations == []
