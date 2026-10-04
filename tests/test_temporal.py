from eldercare.config import Config
from eldercare.events import detect_events
from eldercare.report import analyze
from eldercare.schema import ActivityState as S, TickRecord
from eldercare.temporal import Segment
from tests.synth import make_stream

cfg = Config()


def run(script, noise=0.0, seed=0):
  recs = make_stream(script, cfg, noise, seed)
  segs, events, runs, alerts, overall, rep = analyze(recs, cfg)
  return segs, events, alerts, overall, rep


def states(segs):
  return [s.state.value for s in segs]


FULL = [(270, "LYING_IN_BED"), (36, "SITTING_ON_BED"), (12, "STANDING"),
        (140, "WALKING"), (95, "SITTING_OUTSIDE_BED"), (27, "WALKING"),
        (20, "SITTING_ON_BED"), (300, "LYING_IN_BED")]


def test_full_story_clean():
  segs, events, alerts, overall, rep = run(FULL)
  assert states(segs) == ["LYING_IN_BED", "SITTING_ON_BED", "STANDING", "WALKING",
                          "SITTING_OUTSIDE_BED", "WALKING", "SITTING_ON_BED", "LYING_IN_BED"]
  assert [e.event for e in events] == ["bed_exit", "bed_return"]
  # boundaries within 2 s of truth
  truth = [0, 270, 306, 318, 458, 553, 580, 600]
  for s, t in zip(segs, truth):
    assert abs(s.start - t) <= 2.0, (s, t)
  assert rep["summary"]["final_state"] == "lying_in_bed"


def test_full_story_noisy():
  segs, events, alerts, overall, rep = run(FULL, noise=0.15, seed=3)
  assert [e.event for e in events] == ["bed_exit", "bed_return"]
  assert rep["summary"]["bed_exit_count"] == 1
  total = sum(s.duration for s in segs)
  assert abs(total - 900) < 1.0


def test_sit_up_is_not_exit():
  segs, events, *_ = run([(60, "LYING_IN_BED"), (40, "SITTING_ON_BED"), (60, "LYING_IN_BED")])
  assert events == []


def test_stand_briefly_then_sit_back_is_not_exit():
  segs, events, *_ = run([(60, "LYING_IN_BED"), (20, "SITTING_ON_BED"), (5, "STANDING"),
                          (30, "SITTING_ON_BED"), (60, "LYING_IN_BED")])
  assert events == []


def test_bed_exit_ignores_many_unknown_segments_between_required_steps():
  segs, events, *_ = run([
      (30, "LYING_IN_BED"), (12, "UNKNOWN"),
      (10, "SITTING_ON_BED"), (15, "UNKNOWN"),
      (10, "STANDING"), (20, "UNKNOWN"),
      (12, "WALKING"), (20, "OUT_OF_BED")])
  assert [e.event for e in events] == ["bed_exit"]
  assert events[0].current_state == "OUT_OF_BED"


def test_bed_exit_requires_every_main_step_and_initial_lying():
  _, events, *_ = run([
      (30, "LYING_IN_BED"), (10, "SITTING_ON_BED"),
      (12, "WALKING"), (20, "OUT_OF_BED")])
  assert events == []  # STANDING was never established
  _, events, *_ = run([
      (20, "SITTING_ON_BED"), (10, "STANDING"),
      (12, "WALKING"), (20, "OUT_OF_BED")])
  assert events == []  # video/activity path did not start from lying


def test_visitor_track_cannot_complete_elders_exit_sequence():
  segs = [
      Segment(S.LYING_IN_BED, 0, 30, 0.9, 1, track_id=1),
      Segment(S.UNKNOWN, 30, 35, 0.3, 1),
      Segment(S.SITTING_ON_BED, 35, 45, 0.9, 1, track_id=2),
      Segment(S.STANDING, 45, 55, 0.9, 1, track_id=2),
      Segment(S.WALKING, 55, 70, 0.9, 1, track_id=2),
      Segment(S.OUT_OF_BED, 70, 90, 0.8, 1, track_id=2),
  ]
  events, _ = detect_events(segs, cfg)
  assert events == []


def test_occlusion_in_bed_is_reported_as_unknown_without_fabricating_lying_time():
  segs, events, alerts, overall, rep = run(
      [(60, "LYING_IN_BED"), (120, "LOST"), (60, "LYING_IN_BED")])
  assert states(segs) == ["LYING_IN_BED", "UNKNOWN", "LYING_IN_BED"]
  assert rep["summary"]["activity_duration_sec"]["lying_in_bed"] == 120
  assert rep["summary"]["activity_duration_sec"]["unknown"] == 120
  assert events == []


def test_occluded_sitting_is_not_held_indefinitely():
  segs, events, alerts, overall, rep = run(
      [(20, "SITTING_ON_BED"), (20, "LOST")])
  assert segs[-1].state.value == "UNKNOWN"


def test_leaving_camera_view_outside_bed_is_unknown_and_away():
  segs, events, alerts, overall, rep = run(
      [(60, "LYING_IN_BED"), (10, "SITTING_ON_BED"), (10, "STANDING"), (20, "WALKING"),
       (120, "LOST")])
  assert segs[-1].state.value == "UNKNOWN"
  assert [e.event for e in events] == ["bed_exit"]
  # Event context knows the elder remains away, but factual location accounting
  # must retain the unobserved interval as UNKNOWN.
  assert rep["summary"]["total_out_of_bed_sec"] == 30
  assert rep["summary"]["total_location_unknown_sec"] == 120


def test_report_keeps_every_raw_tick_state_and_real_timestamp_duration():
  recs = [
      TickRecord(t=0.00, state="LYING_IN_BED", confidence=0.9, observed=True),
      TickRecord(t=0.48, state="SITTING_ON_BED", confidence=0.8, observed=True),
      TickRecord(t=0.96, state="STANDING", confidence=0.2, observed=True),
      TickRecord(t=1.44, state="WALKING", confidence=0.7, observed=True),
      TickRecord(t=1.92, state="UNKNOWN", confidence=0.0, observed=True),
      TickRecord(t=2.40, state="LYING_IN_BED", confidence=0.9, observed=False),
  ]
  segs, _, _, _, _, rep = analyze(recs, cfg)
  assert states(segs) == ["LYING_IN_BED", "SITTING_ON_BED", "STANDING",
                          "WALKING", "UNKNOWN"]
  dur = rep["summary"]["activity_duration_sec"]
  assert dur["lying_in_bed"] == dur["sitting_on_bed"] == 0.48
  assert dur["standing"] == dur["walking"] == 0.48
  assert dur["unknown"] == 0.96
  assert rep["summary"]["observation_duration_sec"] == 2.88


def _spatial_tick(t, state, overlap, bbox, *, torso=5.0, knee=170.0,
                  clipped=False, confidence=0.9):
  return TickRecord(t=t, state=state, confidence=confidence, observed=True,
                    track_id=1, bed_known=True, bed_overlap=overlap,
                    torso_tilt=torso, knee_angle=knee, speed_bh_s=0.2,
                    body_clipped=clipped, bbox=bbox, bed=[0, 0, 100, 100])


def test_spatial_crossing_confirms_exit_when_standing_label_is_missing():
  recs = [
      _spatial_tick(0, "LYING_IN_BED", 1.0, [10, 30, 80, 80], torso=85),
      _spatial_tick(1, "SITTING_ON_BED", 0.85, [30, 10, 95, 100]),
      _spatial_tick(2, "UNKNOWN", 0.60, [55, 0, 105, 100], clipped=True),
      _spatial_tick(3, "WALKING", 0.25, [90, 0, 130, 100]),
  ]
  *_, rep = analyze(recs, cfg)
  assert rep["summary"]["bed_exit_count"] == 1
  event = rep["events"][0]
  assert event["confirmation"] == "spatial_bed_crossing"
  assert "standing_inferred" in event["sequence"]


def test_spatial_crossing_confirms_exit_when_sitting_label_is_missing():
  recs = [
      _spatial_tick(0, "LYING_IN_BED", 1.0, [10, 30, 80, 80], torso=85),
      _spatial_tick(1, "UNKNOWN", 0.80, [30, 10, 95, 100]),
      _spatial_tick(2, "STANDING", 0.60, [55, 0, 105, 100]),
      _spatial_tick(3, "WALKING", 0.20, [90, 0, 130, 100]),
  ]
  *_, rep = analyze(recs, cfg)
  assert rep["summary"]["bed_exit_count"] == 1
  assert "sitting_inferred" in rep["events"][0]["sequence"]


def test_walking_label_without_bed_boundary_crossing_is_not_an_exit():
  recs = [
      _spatial_tick(0, "LYING_IN_BED", 1.0, [10, 30, 80, 80], torso=85),
      _spatial_tick(1, "SITTING_ON_BED", 0.85, [25, 10, 90, 100]),
      _spatial_tick(2, "UNKNOWN", 0.78, [30, 0, 95, 100], clipped=True),
      _spatial_tick(3, "WALKING", 0.70, [35, 0, 95, 100]),
  ]
  *_, rep = analyze(recs, cfg)
  assert rep["summary"]["bed_exit_count"] == 0


def test_standalone_bed_return_ignores_unknown_gaps():
  recs = [
      _spatial_tick(0, "OUT_OF_BED", 0.0, [110, 0, 150, 100]),
      _spatial_tick(1, "UNKNOWN", 0.05, [105, 0, 145, 100]),
      _spatial_tick(2, "WALKING", 0.20, [90, 0, 130, 100]),
      _spatial_tick(3, "UNKNOWN", 0.40, [75, 0, 115, 100]),
      _spatial_tick(4, "STANDING", 0.60, [60, 0, 105, 100]),
      _spatial_tick(5, "SITTING_ON_BED", 0.85, [30, 10, 95, 100]),
      _spatial_tick(6, "LYING_IN_BED", 1.0, [10, 30, 90, 80], torso=85),
  ]
  *_, rep = analyze(recs, cfg)
  assert rep["summary"]["bed_return_count"] == 1
  event = next(e for e in rep["events"] if e["event"] == "bed_return")
  assert event["confirmation"] == "spatial_bed_entry"
  assert event["missing_steps"] == []


def test_bed_return_allows_two_missing_middle_steps_with_spatial_proof():
  recs = [
      _spatial_tick(0, "OUT_OF_BED", 0.0, [110, 0, 150, 100]),
      _spatial_tick(1, "UNKNOWN", 0.10, [100, 0, 140, 100]),
      _spatial_tick(2, "WALKING", 0.25, [90, 0, 130, 100]),
      _spatial_tick(3, "UNKNOWN", 0.65, [55, 0, 105, 100]),
      _spatial_tick(4, "LYING_IN_BED", 1.0, [10, 30, 90, 80], torso=85),
  ]
  *_, rep = analyze(recs, cfg)
  event = next(e for e in rep["events"] if e["event"] == "bed_return")
  assert event["missing_steps"] == ["standing", "sitting_on_bed"]
  assert event["spatial_evidence"]["center_started_outside_bed"]
  assert event["spatial_evidence"]["center_finished_inside_bed"]


def test_bed_return_requires_final_lying_and_real_spatial_entry():
  no_lie = [
      _spatial_tick(0, "OUT_OF_BED", 0.0, [110, 0, 150, 100]),
      _spatial_tick(1, "WALKING", 0.25, [90, 0, 130, 100]),
      _spatial_tick(2, "SITTING_ON_BED", 0.85, [30, 10, 95, 100]),
  ]
  *_, rep = analyze(no_lie, cfg)
  assert rep["summary"]["bed_return_count"] == 0

  no_crossing = [
      _spatial_tick(0, "OUT_OF_BED", 0.70, [30, 0, 90, 100]),
      _spatial_tick(1, "WALKING", 0.72, [30, 0, 90, 100]),
      _spatial_tick(2, "LYING_IN_BED", 0.80, [25, 20, 95, 80], torso=85),
  ]
  *_, rep = analyze(no_crossing, cfg)
  assert rep["summary"]["bed_return_count"] == 0


def test_visitor_track_cannot_complete_elders_bed_return():
  recs = [
      _spatial_tick(0, "OUT_OF_BED", 0.0, [110, 0, 150, 100]),
      _spatial_tick(1, "WALKING", 0.25, [90, 0, 130, 100]),
      _spatial_tick(2, "SITTING_ON_BED", 0.85, [30, 10, 95, 100]),
      _spatial_tick(3, "LYING_IN_BED", 1.0, [10, 30, 90, 80], torso=85),
  ]
  recs[-1].track_id = 2
  *_, rep = analyze(recs, cfg)
  assert rep["summary"]["bed_return_count"] == 0


def test_short_observed_absence_is_normal_under_new_policy():
  segs, events, alerts, overall, rep = run(
      [(60, "LYING_IN_BED"), (10, "SITTING_ON_BED"), (10, "STANDING"), (700, "WALKING")])
  assert overall == "NORMAL"
  assert alerts == []


def test_edge_sitting_three_minutes_is_monitor_but_middle_sitting_is_normal():
  edge = [_spatial_tick(i * 0.5, "SITTING_ON_BED", 0.70,
                        [20, 0, 95, 100]) for i in range(362)]
  *_, alerts, overall, rep = analyze(edge, cfg)
  assert overall == "MONITOR"
  assert any(a.rule == "M2_long_edge_sit" for a in alerts)

  middle = [_spatial_tick(i * 0.5, "SITTING_ON_BED", 0.85,
                          [10, 0, 90, 100]) for i in range(1202)]
  *_, alerts, overall, rep = analyze(middle, cfg)
  assert overall == "NORMAL"
  assert alerts == []


def test_fragmented_unknown_ticks_are_summed_until_next_known_activity():
  recs = [_spatial_tick(0.0, "SITTING_ON_BED", 0.85, [10, 0, 90, 100])]
  for i in range(1, 182):
    rec = _spatial_tick(i * 0.5, "UNKNOWN", 0.7, [20, 0, 95, 100],
                        confidence=0.3)
    if i % 2 == 0:
      rec.observed = False
      rec.track_id = None
    recs.append(rec)
  recs.append(_spatial_tick(91.0, "STANDING", 0.3, [80, 0, 120, 100]))
  *_, alerts, overall, rep = analyze(recs, cfg)
  assert overall == "MONITOR"
  unknown = [a for a in alerts if a.rule == "M4_unknown"]
  assert len(unknown) == 1


def test_observed_out_of_bed_30_minutes_is_monitor_not_alert():
  segs, events, alerts, overall, rep = run([(1801, "WALKING")])
  assert overall == "MONITOR"
  assert any(a.rule == "M3_long_absence" for a in alerts)
  assert not any(a.level == "ALERT" for a in alerts)


def test_out_of_bed_and_unseen_40_minutes_is_alert():
  segs, events, alerts, overall, rep = run([(10, "WALKING"), (2401, "LOST")])
  assert overall == "ALERT"
  assert any(a.rule == "A3_lost_outside_bed" for a in alerts)


def test_floor_lying_30_seconds_is_alert():
  segs, events, alerts, overall, rep = run([(31, "LYING_ON_FLOOR")])
  assert overall == "ALERT"
  assert any(a.rule == "A1_floor_lying" for a in alerts)


def test_normal_night_is_normal():
  *_, overall, rep = run([(600, "LYING_IN_BED")])
  assert overall == "NORMAL"


def test_durations_sum_to_observation():
  segs, events, alerts, overall, rep = run(FULL, noise=0.1, seed=7)
  assert abs(sum(rep["summary"]["activity_duration_sec"].values()) - rep["summary"]["observation_duration_sec"]) <= 4
