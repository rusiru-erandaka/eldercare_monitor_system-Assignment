"""Bed ROI calibration: the bed is static, so use a running median of detections
instead of trusting any single frame. A manual ROI can override detection."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from ..config import Config
from ..schema import BoundingBox


class BedCalibrator:

  def __init__(self, cfg: Config, manual_roi: Optional[BoundingBox] = None):
    self.cfg = cfg
    self.manual = manual_roi
    self.history: Deque[BoundingBox] = deque(maxlen=cfg.bed_calib_window)
    self.roi: Optional[BoundingBox] = manual_roi
    self._outliers = 0

  def update(self, beds: List[BoundingBox]) -> Optional[BoundingBox]:
    if self.manual is not None or not beds:
      return self.roi
    best = max(beds, key=lambda b: b.confidence)
    if self.roi is not None and best.iou(self.roi) < self.cfg.bed_outlier_iou:
      # ignore one-off outliers, but accept a persistent change (camera moved)
      self._outliers += 1
      if self._outliers < 5:
        return self.roi
      self.history.clear()
    self._outliers = 0
    self.history.append(best)
    arr = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in self.history])
    med = np.median(arr, axis=0)
    self.roi = BoundingBox(*map(float, med), confidence=best.confidence)
    return self.roi
