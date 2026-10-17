"""YOLO11 object detection for people, beds and chairs.

OpenVINO IR files are decoded locally. Model API directories and model IDs use
``Model.from_pretrained`` with the configured confidence thresholds.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..config import Config
from ..schema import BoundingBox, Detections
from .yolo_ov import YoloOpenVINO, nms, unletterbox_boxes


class ObjectDetector:

  def __init__(self, model_path: str, cfg: Config, device_name: str = "GPU",
               normalize: bool = True):
    self.cfg = cfg
    self.low_conf = min(cfg.person_conf_in_bed, cfg.bed_conf, cfg.chair_conf)
    self._raw = None
    self._model_api = None
    if model_path.endswith(".xml"):
      self._raw = YoloOpenVINO(model_path, None, device_name, normalize, tag="ObjectDetector")
    else:
      from model_api.models import Model  # lazy: only needed for this backend
      print(f"[ObjectDetector] model_api backend on {device_name}")
      local_path = Path(model_path)
      if local_path.is_dir():
        xml_files = list(local_path.glob("*.xml"))
        if len(xml_files) != 1:
          raise ValueError(f"Expected exactly one detector XML in {local_path}")
        self._model_api = Model.create_model(
            model=str(xml_files[0]), device=device_name,
            configuration={"confidence_threshold": self.low_conf})
      else:
        self._model_api = Model.from_pretrained(
            model_path, device=device_name,
            configuration={"confidence_threshold": self.low_conf})

  @property
  def execution_devices(self) -> list[str]:
    if self._raw is not None:
      return self._raw.execution_devices
    adapter = self._model_api.inference_adapter
    return list(adapter.compiled_model.get_property("EXECUTION_DEVICES"))

  @property
  def resolved_model_path(self) -> str:
    if self._raw is not None:
      return self._raw.model_xml_path
    return str(self._model_api.inference_adapter.model_path)

  # -- public ---------------------------------------------------------------
  def detect(self, frame: np.ndarray) -> Detections:
    if self._raw is not None:
      return self._detect_raw(frame)
    return self._detect_model_api(frame)

  # -- OpenVINO backend --------------------------------------------------------
  def _detect_raw(self, frame) -> Detections:
    out, r, left, top, w, h = self._raw._run(frame)   # [4+nc, N]
    cfg = self.cfg
    xyxy = np.stack([out[0] - out[2] / 2, out[1] - out[3] / 2,
                     out[0] + out[2] / 2, out[1] + out[3] / 2], axis=1)

    def per_class(cid: int, thr: float, top_k: int) -> List[BoundingBox]:
      scores = out[4 + cid]
      idx = np.where(scores > thr)[0]
      if idx.size == 0:
        return []
      keep = nms(xyxy[idx], scores[idx], 0.5)[:top_k]
      boxes = unletterbox_boxes(xyxy[idx][keep], r, left, top, w, h)
      return [BoundingBox(*map(float, b), confidence=float(scores[idx][k]))
              for b, k in zip(boxes, keep)]

    return Detections(
        persons=per_class(cfg.person_class_id, cfg.person_conf_in_bed, 6),
        beds=per_class(cfg.bed_class_id, cfg.bed_conf, 2),
        chairs=per_class(cfg.chair_class_id, cfg.chair_conf, 4))

  # -- model_api backend ---------------------------------------------------------
  def _detect_model_api(self, frame) -> Detections:
    result = self._model_api(frame)
    cfg = self.cfg

    # Current openvino-model-api returns a DetectionResult with parallel numpy
    # arrays (bboxes, labels, scores).  Older releases returned iterable object
    # records, so keep both paths.  Treating DetectionResult as an iterable
    # silently produced no detections and made every activity UNKNOWN because
    # the bed was never calibrated.
    if all(hasattr(result, name) for name in ("bboxes", "labels", "scores")):
      persons, beds, chairs = [], [], []
      for coords, cid, score in zip(result.bboxes, result.labels, result.scores):
        score = float(score)
        box = BoundingBox(*map(float, coords[:4]), confidence=score)
        if int(cid) == cfg.person_class_id and score >= cfg.person_conf_in_bed:
          persons.append(box)
        elif int(cid) == cfg.bed_class_id and score >= cfg.bed_conf:
          beds.append(box)
        elif int(cid) == cfg.chair_class_id and score >= cfg.chair_conf:
          chairs.append(box)
      for lst in (persons, beds, chairs):
        lst.sort(key=lambda b: -b.confidence)
      return Detections(persons=persons[:6], beds=beds[:2], chairs=chairs[:4])

    if hasattr(result, "objects"):
      dets = result.objects
    elif hasattr(result, "detections"):
      dets = result.detections
    else:
      try:
        dets = list(result)
      except TypeError:
        dets = []
    persons, beds, chairs = [], [], []
    for obj in dets:
      cid = getattr(obj, "id", getattr(obj, "label_id", getattr(obj, "label", None)))
      score = float(getattr(obj, "score", getattr(obj, "confidence", 0.0)))
      if hasattr(obj, "xmin"):
        box = BoundingBox(float(obj.xmin), float(obj.ymin), float(obj.xmax),
                          float(obj.ymax), score)
      elif hasattr(obj, "box"):
        b = list(map(float, obj.box))
        box = BoundingBox(b[0], b[1], b[2], b[3], score)
      else:
        continue
      if cid == cfg.person_class_id and score >= cfg.person_conf_in_bed:
        persons.append(box)
      elif cid == cfg.bed_class_id and score >= cfg.bed_conf:
        beds.append(box)
      elif cid == cfg.chair_class_id and score >= cfg.chair_conf:
        chairs.append(box)
    for lst in (persons, beds, chairs):
      lst.sort(key=lambda b: -b.confidence)
    return Detections(persons=persons, beds=beds[:2], chairs=chairs[:4])
