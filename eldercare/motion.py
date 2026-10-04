"""Scale-invariant motion estimate (body-heights per second) over a short window."""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

from .config import Config
from .schema import BoundingBox


class MotionEstimator:

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.hist: Deque[Tuple[float, float, float, float]] = deque()

  def reset(self):
    self.hist.clear()

  def update(self, t: float, box: BoundingBox) -> Optional[float]:
    cx, cy = box.center
    self.hist.append((t, cx, cy, box.height))
    while self.hist and t - self.hist[0][0] > self.cfg.motion_window_sec:
      self.hist.popleft()
    if len(self.hist) < 3 or self.hist[-1][0] - self.hist[0][0] < 1.0:
      return None
    heights = sorted(h for _, _, _, h in self.hist)
    h_med = max(heights[len(heights) // 2], 1.0)
    path = 0.0
    pts = list(self.hist)
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(pts, pts[1:]):
      step = math.hypot(x1 - x0, y1 - y0) / h_med
      if step >= self.cfg.motion_step_deadband_bh:
        path += step
    span = pts[-1][0] - pts[0][0]
    return path / span if span > 0 else None
