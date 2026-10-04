"""Classify one fused person observation into an activity state.

The classifier combines posture, movement and the person's position relative to
the bed. Temporal smoothing and event detection are handled by later stages.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .features import Features
from .schema import ActivityState as S, PersonObs


@dataclass
class TickState:
  state: S
  confidence: float
  reason: str


def _orientation(f: Features, cfg: Config):
  """-> (label, margin 0..1). label in upright | horizontal | unclear."""
  if f.torso_tilt is not None:
    if f.torso_tilt <= cfg.upright_tilt_deg:
      return "upright", min(1.0, (cfg.upright_tilt_deg - f.torso_tilt) / cfg.upright_tilt_deg + 0.4)
    if f.torso_tilt >= cfg.horizontal_tilt_deg:
      return "horizontal", min(1.0, (f.torso_tilt - cfg.horizontal_tilt_deg) / 30.0 + 0.5)
    return "unclear", 0.3
  # joints hidden (blanket): fall back to the box shape, with lower confidence
  if f.aspect >= cfg.upright_aspect:
    return "upright", 0.45
  if f.aspect <= cfg.horizontal_aspect:
    return "horizontal", 0.5
  return "unclear", 0.25


def classify_tick(obs: PersonObs, f: Features, cfg: Config) -> TickState:
  if f.inside_bed is None:
    return TickState(S.UNKNOWN, 0.2, "bed_unknown")
  # Low-confidence crop poses are useful for presence detection, but do not
  # contain enough evidence for a posture decision.
  if obs.source == "pose" and obs.pose_conf < cfg.pose_conf:
    return TickState(S.UNKNOWN, max(0.1, round(obs.pose_conf, 3)),
                     "weak_pose_candidate")

  orient, margin = _orientation(f, cfg)
  quality = 1.0 if obs.keypoints and f.torso_tilt is not None else 0.8

  def confidence(value: float) -> float:
    return round(max(0.05, min(0.99, value * quality)), 3)

  legs_bent = f.knee_max is not None and f.knee_max <= cfg.knee_sit_deg
  legs_straight = f.knee_max is not None and f.knee_max >= cfg.knee_straight_deg
  moving = f.speed is not None and f.speed >= cfg.walk_speed_bh_s

  # ---------------- inside the bed ROI ----------------
  if f.inside_bed:
    if orient == "horizontal":
      return TickState(S.LYING_IN_BED, confidence(0.55 + 0.4 * margin), "horizontal_in_bed")
    # When the body continues beyond the camera boundary, an upright torso alone
    # cannot distinguish sitting on the bed from standing beside it. Do not turn
    # missing legs into a confident sitting label; temporal/VLM context resolves it.
    if f.body_clipped:
      return TickState(S.UNKNOWN, confidence(0.30), "partial_body_at_frame_edge")
    if orient == "upright":
      # upright but overlapping the bed only partly and legs straight => standing beside it
      if (f.bed_overlap is not None and f.bed_overlap < 0.60 and legs_straight
          and not legs_bent):
        if moving:
          return TickState(S.WALKING, confidence(0.5 + 0.3 * margin), "upright_straight_legs_moving_at_bed_edge")
        return TickState(S.STANDING, confidence(0.5 + 0.3 * margin), "upright_straight_legs_at_bed_edge")
      return TickState(S.SITTING_ON_BED, confidence(0.5 + 0.4 * margin), "upright_in_bed")
    # unclear (propped on pillows / under blanket): default to lying, low confidence
    return TickState(S.LYING_IN_BED, confidence(0.45), "unclear_posture_in_bed_default_lying")

  # ---------------- outside the bed ROI ----------------
  if orient == "horizontal":
    return TickState(S.LYING_ON_FLOOR, confidence(0.5 + 0.4 * margin), "horizontal_outside_bed")
  if orient == "upright":
    if moving:
      return TickState(S.WALKING, confidence(0.6 + 0.3 * margin), "upright_moving")
    if legs_bent or f.near_chair > 0.4:
      c = 0.55 + 0.3 * margin + (0.1 if f.near_chair > 0.4 else 0.0)
      return TickState(S.SITTING_OUTSIDE_BED, confidence(c), "upright_bent_legs_or_chair")
    if f.speed is None:  # not enough history yet to rule walking in or out
      return TickState(S.STANDING, confidence(0.4 * margin + 0.2), "upright_motion_unknown")
    return TickState(S.STANDING, confidence(0.55 + 0.35 * margin), "upright_still")
  return TickState(S.OUT_OF_BED, confidence(0.4), "outside_bed_unclassified_posture")
