"""Shared domain types: states, geometry, observations and per-tick records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ActivityState(str, Enum):
  LYING_IN_BED = "LYING_IN_BED"
  SITTING_ON_BED = "SITTING_ON_BED"
  SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
  STANDING = "STANDING"
  WALKING = "WALKING"
  OUT_OF_BED = "OUT_OF_BED"
  # Lying outside the bed is kept separate because it may indicate a fall.
  LYING_ON_FLOOR = "LYING_ON_FLOOR"
  UNKNOWN = "UNKNOWN"


S = ActivityState
IN_BED_STATES = {S.LYING_IN_BED, S.SITTING_ON_BED}
OUT_OF_BED_STATES = {
    S.SITTING_OUTSIDE_BED, S.STANDING, S.WALKING, S.OUT_OF_BED,
    S.LYING_ON_FLOOR,
}

# COCO keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class BoundingBox:
  xmin: float
  ymin: float
  xmax: float
  ymax: float
  confidence: float = 1.0

  @property
  def width(self) -> float:
    return max(0.0, self.xmax - self.xmin)

  @property
  def height(self) -> float:
    return max(0.0, self.ymax - self.ymin)

  @property
  def area(self) -> float:
    return self.width * self.height

  @property
  def aspect_ratio(self) -> float:
    """height / width  (>1 tall, <1 wide)."""
    return self.height / self.width if self.width > 0 else 0.0

  @property
  def center(self) -> Tuple[float, float]:
    return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)

  def contains(self, x: float, y: float) -> bool:
    return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

  def intersection_area(self, other: "BoundingBox") -> float:
    w = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
    h = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
    return max(0.0, w) * max(0.0, h)

  def iou(self, other: "BoundingBox") -> float:
    inter = self.intersection_area(other)
    union = self.area + other.area - inter
    return inter / union if union > 0 else 0.0

  def expand(self, frac: float) -> "BoundingBox":
    dx, dy = self.width * frac, self.height * frac
    return BoundingBox(self.xmin - dx, self.ymin - dy, self.xmax + dx,
                       self.ymax + dy, self.confidence)

  def to_list(self) -> List[float]:
    return [round(self.xmin, 1), round(self.ymin, 1),
            round(self.xmax, 1), round(self.ymax, 1)]

  @staticmethod
  def from_list(v: List[float], conf: float = 1.0) -> "BoundingBox":
    return BoundingBox(v[0], v[1], v[2], v[3], conf)


@dataclass
class Keypoint:
  x: float
  y: float
  confidence: float


@dataclass
class PersonObs:
  """One person observation after fusing detector + pose evidence."""
  bbox: BoundingBox
  keypoints: Optional[Dict[int, Keypoint]]  # None => detector-only (e.g. blanket)
  source: str  # "det+pose" | "det" | "pose"
  det_conf: float = 0.0
  pose_conf: float = 0.0

  def n_visible(self, thr: float = 0.3) -> int:
    if not self.keypoints:
      return 0
    return sum(1 for k in self.keypoints.values() if k.confidence > thr)


@dataclass
class Detections:
  persons: List[BoundingBox]
  beds: List[BoundingBox]
  chairs: List[BoundingBox]


@dataclass
class TickRecord:
  """Everything the temporal layer needs for one analysis tick (JSON-serialisable)."""
  t: float
  state: str = ActivityState.UNKNOWN.value
  confidence: float = 0.0
  reason: str = ""
  observed: bool = False          # was the elder observed this tick?
  source: str = "none"
  track_id: Optional[int] = None
  caregiver_present: bool = False
  bed_known: bool = False
  bed_overlap: Optional[float] = None
  torso_tilt: Optional[float] = None
  knee_angle: Optional[float] = None
  speed_bh_s: Optional[float] = None
  body_clipped: bool = False
  bbox: Optional[List[float]] = None
  bed: Optional[List[float]] = None

  def to_dict(self) -> dict:
    return asdict(self)

  @staticmethod
  def from_dict(d: dict) -> "TickRecord":
    return TickRecord(**d)


def fmt_mmss(seconds: float) -> str:
  minutes = int(seconds // 60)
  sec = seconds - minutes * 60
  if abs(sec - round(sec)) < 0.005:
    return f"{minutes:02d}:{int(round(sec)):02d}"
  return f"{minutes:02d}:{sec:05.2f}".rstrip("0").rstrip(".")


def fmt_hhmmss(seconds: float) -> str:
  hours = int(seconds // 3600)
  minutes = int((seconds - hours * 3600) // 60)
  sec = seconds - hours * 3600 - minutes * 60
  if abs(sec - round(sec)) < 0.005:
    sec_text = f"{int(round(sec)):02d}"
  else:
    sec_text = f"{sec:05.2f}".rstrip("0").rstrip(".")
  return f"{hours:02d}:{minutes:02d}:{sec_text}"


def fmt_dur(seconds: float) -> str:
  seconds = round(max(0.0, seconds), 2)
  if seconds >= 60:
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    rem_text = f"{rem:.2f}".rstrip("0").rstrip(".")
    return f"{minutes}m {rem_text.zfill(2)}s"
  value = f"{seconds:.2f}".rstrip("0").rstrip(".")
  return f"{value}s"
