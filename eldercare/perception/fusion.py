"""Fuse object detections and pose estimates into person observations.

Policy:
  1. YOLO11 person boxes are the spatial authority (context-aware threshold:
     lower inside the bed ROI so a person under a blanket is still found).
  2. A pose is only trusted if it matches a detector person; joints outside the
     detector box (+margin) are discarded.
  3. A pose with NO detector support is accepted only if it is a clearly real
     skeleton (many joints, high confidence, big enough). Curtain folds/shadows fail.
  4. A detector person without a pose is kept as a keypoint-less observation
     (blanket occlusion) and handled downstream.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import Config
from ..schema import BoundingBox, Keypoint, PersonObs


def _kp_box(kps: Dict[int, Keypoint], thr: float) -> Optional[BoundingBox]:
  vis = [k for k in kps.values() if k.confidence > thr]
  if len(vis) < 3:
    return None
  xs, ys = [k.x for k in vis], [k.y for k in vis]
  return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def filter_persons(persons: List[BoundingBox], bed: Optional[BoundingBox],
                   cfg: Config) -> List[BoundingBox]:
  keep = []
  for p in persons:
    if p.confidence >= cfg.person_conf:
      keep.append(p)
    elif bed is not None and p.confidence >= cfg.person_conf_in_bed:
      cx, cy = p.center
      if bed.expand(0.05).contains(cx, cy):
        keep.append(p)
  return keep


def fuse(det_persons: List[BoundingBox], det_fresh: bool,
         poses: List[PersonObs], bed: Optional[BoundingBox], frame_w: int,
         frame_h: int, cfg: Config) -> List[PersonObs]:
  persons = filter_persons(det_persons, bed, cfg)
  used = set()
  out: List[PersonObs] = []

  for det in persons:
    best_i, best_score = -1, 0.0
    for i, pose in enumerate(poses):
      if i in used:
        continue
      iou = det.iou(pose.bbox)
      cx, cy = pose.bbox.center
      inside = det.expand(0.1).contains(cx, cy)
      score = iou + (0.2 if inside else 0.0)
      if (iou >= cfg.fuse_iou or inside) and score > best_score:
        best_i, best_score = i, score
    if best_i < 0:
      out.append(PersonObs(det, None, "det", det_conf=det.confidence))
      continue

    used.add(best_i)
    pose = poses[best_i]
    margin_box = det.expand(cfg.kp_box_margin)
    kps = {}
    for idx, kp in pose.keypoints.items():
      ok = margin_box.contains(kp.x, kp.y)
      kps[idx] = Keypoint(kp.x, kp.y, kp.confidence if ok else 0.0)
    obs = PersonObs(det, kps, "det+pose", det_conf=det.confidence,
                    pose_conf=pose.pose_conf)
    # stale detector box (detector runs slower than pose): prefer the fresh pose box
    if not det_fresh and obs.n_visible(cfg.kp_conf) >= 6:
      obs.bbox = BoundingBox(pose.bbox.xmin, pose.bbox.ymin, pose.bbox.xmax,
                             pose.bbox.ymax, det.confidence)
    out.append(obs)

  frame_area = float(frame_w * frame_h)
  for i, pose in enumerate(poses):
    if i in used:
      continue
    vis = [k for k in pose.keypoints.values() if k.confidence > cfg.kp_conf]
    if len(vis) < cfg.pose_only_min_kps:
      continue
    if sum(k.confidence for k in vis) / len(vis) < cfg.pose_only_min_mean_conf:
      continue
    if pose.bbox.area / frame_area < cfg.pose_only_min_area_frac:
      continue
    if pose.bbox.height / frame_h < cfg.pose_only_min_height_frac:
      continue
    out.append(PersonObs(pose.bbox, pose.keypoints, "pose",
                         pose_conf=pose.pose_conf))
  return out
