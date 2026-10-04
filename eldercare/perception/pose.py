"""YOLOv8-Pose decoding -> list of PersonObs (multi-person, NMS'd)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..schema import BoundingBox, Keypoint, PersonObs
from .yolo_ov import YoloOpenVINO, nms, unletterbox_boxes


class PoseEstimator(YoloOpenVINO):

  def __init__(self, model_xml, model_bin=None, device_name="GPU",
               conf_threshold=0.25, nms_iou=0.5, max_persons=5, normalize=True):
    super().__init__(model_xml, model_bin, device_name, normalize, tag="PoseEstimator")
    self.conf_threshold = conf_threshold
    self.nms_iou = nms_iou
    self.max_persons = max_persons

  def infer(self, frame: np.ndarray,
            conf_threshold: Optional[float] = None) -> List[PersonObs]:
    out, r, left, top, w, h = self._run(frame)
    return self.decode(out, r, left, top, w, h, conf_threshold)

  def decode(self, out, r, left, top, w, h,
             conf_threshold: Optional[float] = None) -> List[PersonObs]:
    """out: [56, N] = cx,cy,w,h,score, 17*(x,y,conf) in letterboxed input pixels."""
    threshold = self.conf_threshold if conf_threshold is None else conf_threshold
    scores = out[4]
    idx = np.where(scores > threshold)[0]
    if idx.size == 0:
      return []
    d = out[:, idx]
    boxes = np.stack([d[0] - d[2] / 2, d[1] - d[3] / 2,
                      d[0] + d[2] / 2, d[1] + d[3] / 2], axis=1)
    keep = nms(boxes, d[4], self.nms_iou)[:self.max_persons]
    boxes = unletterbox_boxes(boxes, r, left, top, w, h)

    people = []
    for k in keep:
      b = boxes[k]
      kp_raw = d[5:5 + 51, k].reshape(17, 3)
      kps = {
          i: Keypoint(x=float(np.clip((kp_raw[i, 0] - left) / r, 0, w)),
                      y=float(np.clip((kp_raw[i, 1] - top) / r, 0, h)),
                      confidence=float(kp_raw[i, 2]))
          for i in range(17)
      }
      people.append(PersonObs(
          bbox=BoundingBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                           float(d[4, k])),
          keypoints=kps, source="pose", pose_conf=float(d[4, k])))
    return people
