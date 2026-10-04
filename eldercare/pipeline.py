"""Run video perception and store timestamped activity observations.

Cached tick records allow event and alert rules to be evaluated again without
re-running the OpenVINO models.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .classifier import TickState, classify_tick
from .config import Config
from .features import compute_features
from .motion import MotionEstimator
from .perception.bed import BedCalibrator
from .perception.detector import ObjectDetector
from .perception.fusion import fuse
from .perception.pose import PoseEstimator
from .perception.tracker import PersonTracker
from .schema import (ActivityState, BoundingBox, Detections, Keypoint, PersonObs,
                     TickRecord)


@dataclass
class _ActivityMemory:
  t: float
  state: ActivityState
  confidence: float
  inside_bed: Optional[bool]
  bed_overlap: Optional[float]


class PerceptionPipeline:

  def __init__(self, cfg: Config, detector: ObjectDetector, pose: PoseEstimator,
               manual_bed: Optional[BoundingBox] = None):
    self.cfg = cfg
    self.detector = detector
    self.pose = pose
    self.bed = BedCalibrator(cfg, manual_bed)
    self.tracker = PersonTracker(cfg)
    self.motion = MotionEstimator(cfg)
    self._tick = 0
    self._last_det: Optional[Detections] = None
    self._last_det_t = -1e9
    self._motion_track: Optional[int] = None
    self._activity_memory = {}
    self.last_obs: Optional[PersonObs] = None   # for visualisation
    self.last_poses: List[PersonObs] = []

  def process(self, frame: np.ndarray, t: float) -> TickRecord:
    cfg = self.cfg
    h, w = frame.shape[:2]

    det_fresh = self._tick % cfg.det_every_n_ticks == 0 or self._last_det is None
    if det_fresh:
      self._last_det = self.detector.detect(frame)
      self._last_det_t = t
      self.bed.update(self._last_det.beds)
    elif t - self._last_det_t > cfg.det_max_age_sec:
      det_fresh = False
    det = self._last_det
    self._tick += 1

    poses = self.pose.infer(frame)
    self.last_poses = poses
    bed = self.bed.roi
    obs_list = fuse(det.persons, det_fresh, poses, bed, w, h, cfg)
    # A covered elder can be too small/ambiguous in the full frame. On a complete
    # miss, magnify the static bed ROI and retry pose at a low candidate threshold.
    # The normal pose-only fusion gates (joint count, mean confidence and size)
    # still reject blanket folds and other weak ghost skeletons.
    if not obs_list and bed is not None and cfg.enable_bed_crop_pose:
      crop_poses = self._infer_bed_crop(frame, bed)
      fallback = fuse([], True, crop_poses, bed, w, h, cfg)
      if fallback:
        obs_list = fallback
        self.last_poses.extend(crop_poses)
    res = self.tracker.update(t, obs_list, bed, frame)

    rec = TickRecord(t=round(t, 3), bed_known=bed is not None,
                     caregiver_present=res.caregiver_present,
                     bed=bed.to_list() if bed else None)
    self.last_obs = res.elder
    if res.elder is None:
      rec.reason = "elder_not_observed"
      return rec

    if self._motion_track != res.elder_id:   # identity change -> motion history invalid
      self.motion.reset()
      self._motion_track = res.elder_id
    speed = self.motion.update(t, res.elder.bbox)
    f = compute_features(res.elder, bed, det.chairs, speed, cfg, w, h)
    ts = classify_tick(res.elder, f, cfg)
    ts = self._resolve_partial_body(t, res.elder_id, ts, f)

    rec.state, rec.confidence, rec.reason = ts.state.value, ts.confidence, ts.reason
    rec.observed, rec.source, rec.track_id = True, res.elder.source, res.elder_id
    rec.bed_overlap = None if f.bed_overlap is None else round(f.bed_overlap, 3)
    rec.torso_tilt = None if f.torso_tilt is None else round(f.torso_tilt, 1)
    rec.knee_angle = None if f.knee_max is None else round(f.knee_max, 1)
    rec.speed_bh_s = None if f.speed is None else round(f.speed, 3)
    rec.body_clipped = f.body_clipped
    rec.bbox = res.elder.bbox.to_list()
    return rec

  def _resolve_partial_body(self, t: float, track_id: Optional[int],
                            raw: TickState, f) -> TickState:
    """Recover a short clipped-body gap from the same elder track.

    The fallback only changes an UNKNOWN caused by frame-edge
    clipping, requires recent same-track evidence, checks bed-location continuity,
    and expires after two seconds. Horizontal posture is already classified before
    this point, so the fallback is limited to upright activities.
    """
    if track_id is None:
      return raw
    memory = self._activity_memory.get(track_id)

    def remember(state: TickState):
      if state.state != ActivityState.UNKNOWN and state.confidence >= self.cfg.min_conf:
        self._activity_memory[track_id] = _ActivityMemory(
            t, state.state, state.confidence, f.inside_bed, f.bed_overlap)
      return state

    if not f.body_clipped or raw.state != ActivityState.UNKNOWN or \
        raw.reason != "partial_body_at_frame_edge":
      return remember(raw)
    if memory is None or t - memory.t > self.cfg.partial_track_hold_sec:
      return raw

    age = max(0.0, t - memory.t)
    speed = f.speed
    overlap_change = (abs(f.bed_overlap - memory.bed_overlap)
                      if f.bed_overlap is not None and memory.bed_overlap is not None
                      else 0.0)
    prior = memory.state
    upright_states = {
        ActivityState.SITTING_ON_BED, ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.STANDING, ActivityState.WALKING,
    }

    # Strong tracked displacement while the upright elder is clipped is still
    # valid walking evidence. A sitting-to-exit transition additionally needs a
    # meaningful decrease in bed overlap to avoid calling upper-body sway walking.
    moving = speed is not None and speed >= self.cfg.walk_speed_bh_s
    overlap_drop = (memory.bed_overlap - f.bed_overlap
                    if memory.bed_overlap is not None and f.bed_overlap is not None
                    else 0.0)
    if prior in upright_states and moving and \
        (prior != ActivityState.SITTING_ON_BED or
         overlap_drop >= self.cfg.exit_overlap_drop_min * 0.5):
      return TickState(
          ActivityState.WALKING, max(0.62, raw.confidence),
          "partial_body_tracked_motion")

    location_consistent = (
        (prior == ActivityState.SITTING_ON_BED and f.inside_bed is True and
         overlap_change <= self.cfg.partial_track_overlap_tolerance) or
        (prior == ActivityState.SITTING_OUTSIDE_BED and f.inside_bed is False) or
        (prior == ActivityState.STANDING and
         (f.inside_bed is False or
          (f.bed_overlap is not None and
           f.bed_overlap < self.cfg.edge_sit_overlap_max))) or
        (prior == ActivityState.WALKING and
         (speed is None or speed >= self.cfg.walk_speed_bh_s * 0.5)))
    if not location_consistent:
      return raw

    decay = max(0.60, 1.0 - 0.20 * age / self.cfg.partial_track_hold_sec)
    confidence = max(self.cfg.partial_track_min_conf,
                     min(0.80, memory.confidence * decay))
    return TickState(prior, round(confidence, 3),
                     f"partial_body_tracked_from_{prior.value.lower()}")

  def _infer_bed_crop(self, frame: np.ndarray,
                      bed: BoundingBox) -> List[PersonObs]:
    """Run low-threshold pose on a magnified bed crop and map it to frame pixels."""
    h, w = frame.shape[:2]
    b = bed.expand(self.cfg.bed_crop_padding)
    x1, y1 = max(0, int(b.xmin)), max(0, int(b.ymin))
    x2, y2 = min(w, int(b.xmax)), min(h, int(b.ymax))
    if x2 - x1 < 32 or y2 - y1 < 32:
      return []
    local = self.pose.infer(frame[y1:y2, x1:x2], self.cfg.bed_crop_pose_conf)
    mapped: List[PersonObs] = []
    for p in local:
      box = BoundingBox(p.bbox.xmin + x1, p.bbox.ymin + y1,
                        p.bbox.xmax + x1, p.bbox.ymax + y1,
                        p.bbox.confidence)
      kps = None if p.keypoints is None else {
          i: Keypoint(kp.x + x1, kp.y + y1, kp.confidence)
          for i, kp in p.keypoints.items()
      }
      mapped.append(PersonObs(box, kps, "pose", pose_conf=p.pose_conf))
    return mapped


# ---------------------------------------------------------------------------------
def iter_video(path: str, analysis_fps: float) -> Iterator[tuple]:
  """Yield (t_seconds, frame) at `analysis_fps` using the video's own timestamps."""
  cap = cv2.VideoCapture(path)
  if not cap.isOpened():
    raise ValueError(f"Unable to open video: {path}")
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  step = max(1, int(round(fps / analysis_fps)))
  idx = 0
  try:
    while True:
      if idx % step == 0:
        ok, frame = cap.read()
        if not ok:
          break
        yield idx / fps, frame
      else:
        if not cap.grab():
          break
      idx += 1
  finally:
    cap.release()


def video_identity(path: str) -> dict:
  """Return an exact, content-based identity for a source video.

  File names are not sufficient: a user may rename a video, or accidentally
  pair ``waking`` ticks with ``return-to-bed`` frames.  A streaming SHA-256 is
  computed once when ticks are produced and checked before cached ticks are
  used for visual review.
  """
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    while True:
      chunk = f.read(4 * 1024 * 1024)
      if not chunk:
        break
      digest.update(chunk)
  return {
      "path": os.path.abspath(path),
      "size_bytes": os.path.getsize(path),
      "sha256": digest.hexdigest(),
  }


def tick_metadata_path(path: str) -> str:
  root, _ = os.path.splitext(path)
  return root + ".meta.json"


def save_tick_metadata(path: str, source_video: str, tick_count: int):
  metadata = {
      "schema_version": 1,
      "tick_count": tick_count,
      "source_video": video_identity(source_video),
  }
  meta_path = tick_metadata_path(path)
  temp = meta_path + ".tmp"
  with open(temp, "w") as f:
    json.dump(metadata, f, indent=2)
  os.replace(temp, meta_path)


def save_ticks(path: str, records: List[TickRecord], source_video: Optional[str] = None):
  with open(path, "w") as f:
    for r in records:
      f.write(json.dumps(r.to_dict()) + "\n")
  if source_video:
    save_tick_metadata(path, source_video, len(records))


def load_ticks(path: str) -> List[TickRecord]:
  with open(path) as f:
    return [TickRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def validate_ticks_video(ticks_path: str, video_path: str) -> dict:
  """Verify that cached ticks were generated from ``video_path``.

  Old result folders are migrated only when their summary still names an
  existing source video whose content exactly matches the supplied video.
  Records with unknown provenance are rejected.
  """
  meta_path = tick_metadata_path(ticks_path)
  metadata = None
  if os.path.exists(meta_path):
    with open(meta_path) as f:
      metadata = json.load(f)
  else:
    summary_path = os.path.join(os.path.dirname(os.path.abspath(ticks_path)),
                                "summary.json")
    if os.path.exists(summary_path):
      try:
        with open(summary_path) as f:
          report = json.load(f)
        analysis = report.get("analysis", {})
        old_source = analysis.get("input_path") if analysis.get("input_type") == "video" else None
        if old_source and os.path.isfile(old_source):
          old_identity = video_identity(old_source)
          requested = video_identity(video_path)
          if old_identity["sha256"] == requested["sha256"]:
            records = load_ticks(ticks_path)
            save_tick_metadata(ticks_path, video_path, len(records))
            with open(meta_path) as f:
              metadata = json.load(f)
      except (OSError, ValueError, json.JSONDecodeError):
        metadata = None

  if metadata is None:
    raise ValueError(
        f"Cannot verify the source video for cached ticks: {ticks_path}. "
        "Run perception on this video without --from-ticks to create a verified cache.")

  expected = metadata.get("source_video", {})
  actual = video_identity(video_path)
  if (expected.get("sha256") != actual["sha256"] or
      expected.get("size_bytes") != actual["size_bytes"]):
    source = expected.get("path", "unknown video")
    raise ValueError(
        "Cached ticks/video mismatch.\n"
        f"  ticks were generated from: {source}\n"
        f"  video supplied now:       {actual['path']}\n"
        "--from-ticks skips object and pose detection; it cannot be paired with "
        "a different video. Use that video's matching ticks, or remove --from-ticks.")
  return metadata


# ---------------------------------------------------------------------------------
COLORS = {
    "LYING_IN_BED": (255, 200, 0), "SITTING_ON_BED": (0, 255, 255),
    "SITTING_OUTSIDE_BED": (255, 150, 0), "STANDING": (0, 255, 0),
    "WALKING": (0, 165, 255), "OUT_OF_BED": (0, 100, 255),
    "LYING_ON_FLOOR": (0, 0, 255), "UNKNOWN": (160, 160, 160),
}


def draw_overlay(frame: np.ndarray, pipe: PerceptionPipeline, rec: TickRecord, cfg: Config):
  bed = pipe.bed.roi
  if bed:
    cv2.rectangle(frame, (int(bed.xmin), int(bed.ymin)), (int(bed.xmax), int(bed.ymax)),
                  (255, 120, 0), 2)
    cv2.putText(frame, "BED", (int(bed.xmin), max(15, int(bed.ymin) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
  obs = pipe.last_obs
  if obs:
    b = obs.bbox
    cv2.rectangle(frame, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)),
                  (0, 255, 255), 2)
    if obs.keypoints:
      for kp in obs.keypoints.values():
        if kp.confidence > cfg.kp_conf:
          cv2.circle(frame, (int(kp.x), int(kp.y)), 4, (0, 0, 255), -1)
  cv2.rectangle(frame, (0, 0), (frame.shape[1], 52), (0, 0, 0), -1)
  label = f"{rec.state}  conf={rec.confidence:.2f}  [{rec.reason}]"
  if rec.body_clipped:
    label += "  PARTIAL_BODY"
  if rec.caregiver_present:
    label += "  +caregiver(ignored)"
  cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
              COLORS.get(rec.state, (255, 255, 255)), 2)
