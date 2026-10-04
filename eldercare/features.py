"""Geometric evidence extracted from one fused person observation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .schema import (BoundingBox, Keypoint, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
                    PersonObs, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER)


@dataclass
class Features:
  bed_overlap: Optional[float]      # fraction of person box inside bed ROI (None = bed unknown)
  center_in_bed: Optional[bool]
  inside_bed: Optional[bool]        # final in/out decision (None = bed unknown)
  torso_tilt: Optional[float]       # 0 = vertical, 90 = horizontal (image plane)
  knee_min: Optional[float]
  knee_max: Optional[float]         # straightest visible leg
  aspect: float                     # bbox h/w
  n_kps: int
  near_chair: float                 # fraction of person box overlapping a chair
  speed: Optional[float]            # body-heights / s
  body_clipped: bool                # bbox reaches image edge; posture may be incomplete


def _ok(kp: Optional[Keypoint], thr: float) -> bool:
  return kp is not None and kp.confidence > thr


def _mid(points: List[Keypoint]):
  return (sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))


def torso_tilt(obs: PersonObs, thr: float) -> Optional[float]:
  if not obs.keypoints:
    return None
  k = obs.keypoints
  sh = [k[i] for i in (L_SHOULDER, R_SHOULDER) if _ok(k.get(i), thr)]
  hp = [k[i] for i in (L_HIP, R_HIP) if _ok(k.get(i), thr)]
  if not sh or not hp:
    return None
  (sx, sy), (hx, hy) = _mid(sh), _mid(hp)
  dx, dy = abs(hx - sx), abs(hy - sy)
  if dx + dy < 1e-6:
    return None
  return math.degrees(math.atan2(dx, dy))


def joint_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
  """Angle at b (degrees) between b->a and b->c."""
  v1, v2 = (a.x - b.x, a.y - b.y), (c.x - b.x, c.y - b.y)
  m1, m2 = math.hypot(*v1), math.hypot(*v2)
  if m1 * m2 == 0:
    return 180.0
  cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)))
  return math.degrees(math.acos(cos))


def knee_angles(obs: PersonObs, thr: float):
  if not obs.keypoints:
    return None, None
  k = obs.keypoints
  angles = []
  for h, kn, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
    if _ok(k.get(h), thr) and _ok(k.get(kn), thr) and _ok(k.get(a), thr):
      angles.append(joint_angle(k[h], k[kn], k[a]))
  if not angles:
    return None, None
  return min(angles), max(angles)


def bed_relation(box: BoundingBox, bed: Optional[BoundingBox], cfg: Config):
  if bed is None or box.area <= 0:
    return None, None, None
  overlap = box.intersection_area(bed) / box.area
  cx, cy = box.center
  center_in = bed.contains(cx, cy)
  inside = (center_in and overlap >= cfg.bed_inside_center_overlap) or \
           overlap >= cfg.bed_inside_overlap
  return overlap, center_in, inside


def compute_features(obs: PersonObs, bed: Optional[BoundingBox],
                     chairs: List[BoundingBox], speed: Optional[float],
                     cfg: Config, frame_w: Optional[int] = None,
                     frame_h: Optional[int] = None) -> Features:
  overlap, center_in, inside = bed_relation(obs.bbox, bed, cfg)
  kmin, kmax = knee_angles(obs, cfg.kp_conf)
  chair = 0.0
  if chairs and obs.bbox.area > 0:
    chair = max(obs.bbox.intersection_area(c) / obs.bbox.area for c in chairs)
  clipped = False
  if frame_w and frame_h:
    mx, my = frame_w * cfg.frame_edge_margin_frac, frame_h * cfg.frame_edge_margin_frac
    clipped = (obs.bbox.xmin <= mx or obs.bbox.ymin <= my or
               obs.bbox.xmax >= frame_w - mx or obs.bbox.ymax >= frame_h - my)
  return Features(
      bed_overlap=overlap, center_in_bed=center_in, inside_bed=inside,
      torso_tilt=torso_tilt(obs, cfg.kp_conf), knee_min=kmin, knee_max=kmax,
      aspect=obs.bbox.aspect_ratio, n_kps=obs.n_visible(cfg.kp_conf),
      near_chair=chair, speed=speed, body_clipped=clipped)
