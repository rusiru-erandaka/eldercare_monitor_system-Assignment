import os
import tempfile

import numpy as np
import openvino as ov
import openvino.opset13 as ops
from types import SimpleNamespace

from eldercare.config import Config
from eldercare.classifier import TickState, classify_tick
from eldercare.features import Features, compute_features
from eldercare.motion import MotionEstimator
from eldercare.perception.fusion import fuse
from eldercare.perception.detector import ObjectDetector
from eldercare.perception.pose import PoseEstimator
from eldercare.perception.tracker import PersonTracker
from eldercare.perception.yolo_ov import YoloOpenVINO
from eldercare.schema import ActivityState as S, BoundingBox, Keypoint, PersonObs

cfg = Config()
BED = BoundingBox(100, 300, 900, 650)


def _save(model, name):
  path = os.path.join(tempfile.mkdtemp(), name)
  ov.save_model(model, path)
  return path


def fake_pose_ir(cands):
  """IR with input [1,3,640,640] f32 and a constant [1,56,8400] output holding `cands`."""
  out = np.zeros((1, 56, 8400), np.float32)
  for i, c in enumerate(cands):
    out[0, :5, i] = [c["cx"], c["cy"], c["w"], c["h"], c["score"]]
    for k in range(17):
      out[0, 5 + 3 * k, i] = c["kp"][k][0]
      out[0, 6 + 3 * k, i] = c["kp"][k][1]
      out[0, 7 + 3 * k, i] = c["kp"][k][2]
  x = ops.parameter([1, 3, 640, 640], np.float32, name="images")
  zero = ops.multiply(ops.reduce_sum(x, ops.constant(np.array([0, 1, 2, 3], np.int64)), False),
                      ops.constant(np.float32(0)))
  y = ops.add(ops.constant(out), zero)
  return _save(ov.Model([y], [x], "fake_pose"), "pose.xml")


def test_preprocessing_is_rgb_and_scaled():
  x = ops.parameter([1, 3, 640, 640], np.float32, name="images")
  m = ops.reduce_mean(x, ops.constant(np.array([2, 3], np.int64)), False)  # -> [1,3]
  path = _save(ov.Model([m], [x], "probe"), "probe.xml")
  net = YoloOpenVINO(path, None, "CPU", normalize=True)
  img = np.zeros((640, 640, 3), np.uint8)
  img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30           # BGR
  res = net.request.infer({0: img[None]})
  vals = np.asarray(res[net.compiled.output(0)]).ravel()
  np.testing.assert_allclose(vals, np.array([30, 20, 10]) / 255.0, atol=1e-4)  # RGB, 0..1


def test_pose_decode_letterbox_nms_and_ghost():
  person = {"cx": 320, "cy": 320, "w": 100, "h": 200, "score": 0.9,
            "kp": [(320 + 3 * k, 250 + 8 * k, 0.9) for k in range(17)]}
  dup = dict(person, score=0.6)                                   # removed by NMS
  other = {"cx": 100, "cy": 100, "w": 40, "h": 80, "score": 0.4,
           "kp": [(100, 100 + k, 0.4) for k in range(17)]}
  path = fake_pose_ir([person, dup, other])
  est = PoseEstimator(path, None, "CPU", conf_threshold=0.25)
  frame = np.zeros((720, 1280, 3), np.uint8)                       # r=0.5, top=140
  poses = est.infer(frame)
  assert len(poses) == 2
  b = poses[0].bbox
  assert abs(b.xmin - 540) < 2 and abs(b.xmax - 740) < 2
  assert abs(b.ymin - 160) < 2 and abs(b.ymax - 560) < 2
  assert abs(poses[0].keypoints[0].x - 640) < 2 and abs(poses[0].keypoints[0].y - 220) < 2


def kp_dict(pts, conf=0.9):
  return {i: Keypoint(x, y, conf) for i, (x, y) in pts.items()}


def standing_obs(cx=1000, top=100, h=400):
  w = h * 0.35
  pts = {5: (cx - 30, top + 90), 6: (cx + 30, top + 90), 11: (cx - 20, top + 230),
         12: (cx + 20, top + 230), 13: (cx - 20, top + 310), 14: (cx + 20, top + 310),
         15: (cx - 20, top + 395), 16: (cx + 20, top + 395)}
  return PersonObs(BoundingBox(cx - w / 2, top, cx + w / 2, top + h, 0.9),
                   kp_dict(pts), "det+pose")


def lying_obs(x0=200, y=450):
  pts = {5: (x0 + 80, y - 20), 6: (x0 + 80, y + 20), 11: (x0 + 250, y - 15),
         12: (x0 + 250, y + 15), 13: (x0 + 330, y - 15), 14: (x0 + 330, y + 15),
         15: (x0 + 420, y - 15), 16: (x0 + 420, y + 15)}
  return PersonObs(BoundingBox(x0, y - 60, x0 + 450, y + 60, 0.9), kp_dict(pts), "det+pose")


def sitting_edge_obs(cx=500, top=150):
  pts = {5: (cx - 30, top + 80), 6: (cx + 30, top + 80), 11: (cx - 20, top + 210),
         12: (cx + 20, top + 210), 13: (cx - 90, top + 230), 14: (cx - 80, top + 250),
         15: (cx - 95, top + 330), 16: (cx - 85, top + 340)}   # knees ~90 deg
  return PersonObs(BoundingBox(cx - 100, top, cx + 40, top + 345, 0.9), kp_dict(pts), "det+pose")


def classify(obs, speed=None, bed=BED):
  f = compute_features(obs, bed, [], speed, cfg)
  return classify_tick(obs, f, cfg), f


def test_classifier_core_states():
  assert classify(lying_obs())[0].state.value == "LYING_IN_BED"
  assert classify(sitting_edge_obs())[0].state.value == "SITTING_ON_BED"
  assert classify(standing_obs(cx=1000), speed=0.02)[0].state.value == "STANDING"
  assert classify(standing_obs(cx=1000), speed=0.5)[0].state.value == "WALKING"
  assert classify(lying_obs(x0=950, y=700), bed=BoundingBox(100, 300, 900, 650))[0].state.value == "LYING_ON_FLOOR"


def test_standing_beside_bed_is_not_lying():
  # straight-legged person whose box overlaps the bed edge (the old Rule 1 bug)
  obs = standing_obs(cx=880, top=250, h=400)   # box x 810..950 -> ~64% inside bed ROI width-wise
  st, f = classify(obs, speed=0.02)
  assert st.state.value in ("STANDING", "SITTING_ON_BED") and st.state.value != "LYING_IN_BED"


def test_blanket_occlusion_defaults_to_lying():
  obs = PersonObs(BoundingBox(200, 380, 700, 520, 0.2), None, "det", det_conf=0.2)
  st, _ = classify(obs)
  assert st.state.value == "LYING_IN_BED" and st.confidence < 0.7


def test_bed_unknown_gives_unknown():
  st, _ = classify(standing_obs(), bed=None)
  assert st.state.value == "UNKNOWN"


def test_partial_upright_body_in_bed_is_unknown():
  obs = standing_obs(cx=160, top=300, h=420)
  f = compute_features(obs, BED, [], 0.02, cfg, frame_w=1280, frame_h=720)
  st = classify_tick(obs, f, cfg)
  assert f.body_clipped
  assert st.state.value == "UNKNOWN"
  assert st.reason == "partial_body_at_frame_edge"


def _features(*, overlap, inside, speed, clipped):
  return Features(
      bed_overlap=overlap, center_in_bed=inside, inside_bed=inside,
      torso_tilt=10.0, knee_min=None, knee_max=None, aspect=2.0,
      n_kps=8, near_chair=0.0, speed=speed, body_clipped=clipped)


def test_partial_body_uses_recent_same_track_activity_then_expires():
  from eldercare.pipeline import PerceptionPipeline
  c = Config()
  pipe = PerceptionPipeline.__new__(PerceptionPipeline)
  pipe.cfg = c
  pipe._activity_memory = {}

  clear = TickState(S.SITTING_ON_BED, 0.82, "upright_in_bed")
  pipe._resolve_partial_body(
      0.0, 7, clear, _features(overlap=0.80, inside=True,
                              speed=0.01, clipped=False))
  unknown = TickState(S.UNKNOWN, 0.30, "partial_body_at_frame_edge")
  recovered = pipe._resolve_partial_body(
      0.5, 7, unknown, _features(overlap=0.72, inside=True,
                                speed=0.01, clipped=True))
  assert recovered.state == S.SITTING_ON_BED
  assert recovered.reason == "partial_body_tracked_from_sitting_on_bed"

  expired = pipe._resolve_partial_body(
      c.partial_track_hold_sec + 0.1, 7, unknown,
      _features(overlap=0.72, inside=True, speed=0.01, clipped=True))
  assert expired.state == S.UNKNOWN


def test_partial_body_tracking_uses_motion_and_bed_departure_for_walking():
  from eldercare.pipeline import PerceptionPipeline
  c = Config()
  pipe = PerceptionPipeline.__new__(PerceptionPipeline)
  pipe.cfg = c
  pipe._activity_memory = {}
  pipe._resolve_partial_body(
      0.0, 3, TickState(S.SITTING_ON_BED, 0.8, "upright_in_bed"),
      _features(overlap=0.85, inside=True, speed=0.01, clipped=False))
  result = pipe._resolve_partial_body(
      0.5, 3, TickState(S.UNKNOWN, 0.3, "partial_body_at_frame_edge"),
      _features(overlap=0.60, inside=True, speed=0.20, clipped=True))
  assert result.state == S.WALKING
  assert result.reason == "partial_body_tracked_motion"


def test_balanced_default_inference_rates():
  c = Config()
  assert c.analysis_fps == 4.0
  assert c.analysis_fps / c.det_every_n_ticks == 2.0


def test_weak_pose_only_candidate_does_not_claim_posture():
  obs = sitting_edge_obs()
  obs.source = "pose"
  obs.pose_conf = 0.10
  st, _ = classify(obs)
  assert st.state.value == "UNKNOWN"
  assert st.reason == "weak_pose_candidate"


def test_fusion_rejects_ghost_pose_and_keeps_blanket_person():
  ghost = PersonObs(BoundingBox(1200, 85, 1230, 170, 0.4),
                    kp_dict({i: (1205 + (i % 3) * 8, 90 + i * 4) for i in range(17)}, conf=0.4),
                    "pose", pose_conf=0.4)
  low_person = BoundingBox(250, 350, 700, 520, 0.18)   # < 0.30 but inside the bed ROI
  out = fuse([low_person], True, [ghost], BED, 1293, 720, cfg)
  assert len(out) == 1 and out[0].source == "det" and out[0].keypoints is None
  out2 = fuse([], True, [ghost], BED, 1293, 720, cfg)
  assert out2 == []
  strong = standing_obs()
  strong.source = "pose"
  strong.pose_conf = 0.9
  out3 = fuse([], True, [strong], BED, 1293, 720, cfg)
  assert len(out3) == 1 and out3[0].source == "pose"


def test_fusion_discards_keypoints_outside_det_box():
  obs = standing_obs()
  obs.keypoints[15] = Keypoint(5, 5, 0.9)               # stray joint far outside the box
  out = fuse([obs.bbox], True, [obs], BED, 1293, 720, cfg)
  assert out[0].keypoints[15].confidence == 0.0


def test_model_api_detection_result_arrays_are_parsed():
  detector = ObjectDetector.__new__(ObjectDetector)
  detector.cfg = cfg
  detector._model_api = lambda frame: SimpleNamespace(
      bboxes=np.array([[10, 20, 110, 220], [100, 200, 500, 600],
                       [700, 200, 900, 600]], dtype=np.float32),
      labels=np.array([cfg.person_class_id, cfg.bed_class_id, cfg.chair_class_id]),
      scores=np.array([0.9, 0.8, 0.7], dtype=np.float32))
  result = detector._detect_model_api(np.zeros((720, 1280, 3), np.uint8))
  assert len(result.persons) == len(result.beds) == len(result.chairs) == 1
  assert result.persons[0].to_list() == [10.0, 20.0, 110.0, 220.0]


def test_motion_estimator_walk_vs_still():
  m = MotionEstimator(cfg)
  sp = None
  for i in range(10):
    sp = m.update(i * 0.5, BoundingBox(100 + i * 60, 100, 200 + i * 60, 500))
  assert sp is not None and sp > cfg.walk_speed_bh_s
  m2 = MotionEstimator(cfg)
  for i in range(10):
    sp2 = m2.update(i * 0.5, BoundingBox(100 + (i % 2) * 3, 100, 200, 500))
  assert sp2 is not None and sp2 < 0.05


def test_tracker_keeps_elder_and_ignores_caregiver():
  tr = PersonTracker(cfg)
  elder = lambda: PersonObs(BoundingBox(200, 380, 700, 520, 0.8), None, "det")
  t = 0.0
  for _ in range(40):                                   # elder alone in bed 20 s
    res = tr.update(t, [elder()], BED)
    t += 0.5
  eid = res.elder_id
  assert eid is not None
  for i in range(20):                                   # caregiver walks in
    cg = PersonObs(BoundingBox(1000 - i * 20, 100, 1100 - i * 20, 500, 0.9), None, "det")
    res = tr.update(t, [elder(), cg], BED)
    t += 0.5
  assert res.elder_id == eid and res.caregiver_present
  assert res.elder.bbox.xmin == 200                      # the returned obs is the elder's
  for _ in range(30):                                   # elder undetected for 15 s
    res = tr.update(t, [], BED)
    t += 0.5
  assert res.elder is None
  res = tr.update(t, [PersonObs(BoundingBox(210, 385, 705, 525, 0.5), None, "det")], BED)
  assert res.elder_id == eid                             # re-linked to the same identity


def test_single_elder_tracker_recovers_from_box_fragmentation():
  tr = PersonTracker(cfg)
  head = PersonObs(BoundingBox(100, 300, 160, 370, 0.5), None, "det")
  full = PersonObs(BoundingBox(700, 180, 1100, 680, 0.9), None, "det")
  first_frame = np.zeros((720, 1280, 3), np.uint8)
  first_frame[300:370, 100:160] = (0, 0, 255)
  next_frame = np.zeros((720, 1280, 3), np.uint8)
  next_frame[180:680, 700:1100] = (0, 255, 0)
  first = tr.update(0.0, [head], BED, first_frame)
  old_id = first.elder_id
  # The very different box creates another track; after a short continuity
  # window the only visible person must become the elder in single-resident mode.
  tr.update(0.5, [full], BED, next_frame)
  tr.update(1.0, [full], BED, next_frame)
  result = tr.update(1.5, [full], BED, next_frame)
  assert result.elder is full
  assert result.elder_id == old_id  # fragmented observation merged into logical elder ID


def test_visitor_born_outside_does_not_inherit_hidden_elder_identity():
  tr = PersonTracker(cfg)
  elder = PersonObs(BoundingBox(200, 380, 700, 520, 0.8), None, "det")
  visitor = PersonObs(BoundingBox(1000, 120, 1120, 620, 0.9), None, "det")
  t = 0.0
  for _ in range(20):
    result = tr.update(t, [elder], BED)
    t += 0.5
  elder_id = result.elder_id
  tr.update(t, [elder, visitor], BED)  # visitor track originates outside the bed
  t += 0.5
  for _ in range(40):
    result = tr.update(t, [visitor], BED)
    t += 0.5
  assert tr.elder_id == elder_id
  assert result.elder is None


class FakeDetector:
  def __init__(self, script): self.script, self.i = script, 0
  def detect(self, frame):
    from eldercare.schema import Detections
    d = self.script(self.i)
    self.i += 1
    return d


class FakePose:
  def __init__(self, script): self.script, self.i = script, 0
  def infer(self, frame):
    r = self.script(self.i)
    self.i += 1
    return r


def test_pipeline_end_to_end_with_fakes():
  from eldercare.pipeline import PerceptionPipeline
  from eldercare.report import analyze
  from eldercare.schema import Detections
  c = Config()
  c.det_every_n_ticks = 1

  def tri(t, period=6.0, amp=350.0):
    ph = (t % period) / period
    return amp * (2 * ph if ph < 0.5 else 2 * (1 - ph))

  def person(i):
    t = i * 0.5
    if t < 30: return lying_obs(x0=450, y=450)
    if t < 40: return sitting_edge_obs(cx=800)
    if t < 50: return standing_obs(cx=900)
    if t < 80: return standing_obs(cx=900 + tri(t - 50))
    if t < 88: return standing_obs(cx=900)
    if t < 98: return sitting_edge_obs(cx=800)
    return lying_obs(x0=450, y=450)

  det = lambda i: Detections([person(i).bbox], [BED], [])
  pose = lambda i: [PersonObs(person(i).bbox, person(i).keypoints, "pose", pose_conf=0.9)]
  pipe = PerceptionPipeline(c, FakeDetector(det), FakePose(pose))
  frame = np.zeros((720, 1293, 3), np.uint8)
  recs = [pipe.process(frame, i * 0.5) for i in range(240)]
  assert all(r.observed for r in recs)            # identity never lost
  assert {r.track_id for r in recs} == {1}
  segs, events, runs, alerts, overall, rep = analyze(recs, c)
  seq = [s.state.value for s in segs]
  assert seq[0] == "LYING_IN_BED" and seq[-1] == "LYING_IN_BED"
  assert "WALKING" in seq and "STANDING" in seq and "SITTING_ON_BED" in seq
  assert [e.event for e in events] == ["bed_exit", "bed_return"]
  assert overall == "NORMAL"  # routine confirmed exit + safe return
