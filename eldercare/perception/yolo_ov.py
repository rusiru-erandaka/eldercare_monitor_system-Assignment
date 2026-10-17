"""OpenVINO utilities shared by the YOLO object and pose models.

The preprocessing stage performs letterbox resizing, BGR-to-RGB conversion and
normalisation. Decoding supports non-maximum suppression and multiple people.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import openvino as ov
from openvino import Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
  """Resize keeping aspect ratio and pad to size x size. Returns (img, r, left, top)."""
  h, w = img.shape[:2]
  r = min(size / h, size / w)
  nh, nw = int(round(h * r)), int(round(w * r))
  resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
  left, top = (size - nw) // 2, (size - nh) // 2
  canvas = np.full((size, size, 3), 114, np.uint8)
  canvas[top:top + nh, left:left + nw] = resized
  return canvas, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
  """Plain greedy NMS. boxes: [N,4] xyxy."""
  order = scores.argsort()[::-1]
  keep = []
  while order.size > 0:
    i = int(order[0])
    keep.append(i)
    if order.size == 1:
      break
    rest = order[1:]
    xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
    yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
    xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
    yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
    area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
    iou = inter / (area_i + area_r - inter + 1e-9)
    order = rest[iou <= iou_thr]
  return keep


def unletterbox_boxes(boxes: np.ndarray, r: float, left: int, top: int,
                      w: int, h: int) -> np.ndarray:
  out = boxes.copy()
  out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
  out[:, [1, 3]] = (out[:, [1, 3]] - top) / r
  out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
  out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
  return out


class YoloOpenVINO:
  """Compiles an OpenVINO IR once and keeps it resident on the device."""

  def __init__(self, model_xml: str, model_bin: Optional[str] = None,
               device_name: str = "GPU", normalize: bool = True, tag: str = "YOLO"):
    self.model_xml_path = model_xml
    self.model_bin_path = model_bin
    self.core = ov.Core()
    model = self.core.read_model(model_xml, model_bin) if model_bin else \
        self.core.read_model(model_xml)

    shape = model.input(0).partial_shape
    self.size = 640
    if shape.rank.is_static and len(shape) == 4 and shape[2].is_static:
      self.size = int(shape[2].get_length())

    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(Type.u8).set_layout(
        Layout("NHWC")).set_color_format(ColorFormat.BGR)
    ppp.input().model().set_layout(Layout("NCHW"))
    steps = ppp.input().preprocess().convert_element_type(Type.f32)
    if normalize:
      steps.convert_color(ColorFormat.RGB).scale(255.0)
    else:
      steps.convert_color(ColorFormat.RGB)
    model = ppp.build()

    self.device_name = device_name
    try:
      self.compiled = self.core.compile_model(model, device_name)
    except Exception as exc:  # e.g. no Intel GPU on the reviewer's machine
      print(f"[{tag}] compile on {device_name} failed ({exc}); falling back to CPU")
      self.device_name = "CPU"
      self.compiled = self.core.compile_model(model, "CPU")
    self.request = self.compiled.create_infer_request()
    print(f"[{tag}] ready on {self.device_name} (input {self.size}x{self.size})")

  @property
  def execution_devices(self) -> list[str]:
    """Report what OpenVINO actually compiled, including CPU fallback."""
    return list(self.compiled.get_property("EXECUTION_DEVICES"))

  def _run(self, frame: np.ndarray):
    h, w = frame.shape[:2]
    img, r, left, top = letterbox(frame, self.size)
    res = self.request.infer({0: np.expand_dims(img, 0)})
    out = np.asarray(res[self.compiled.output(0)])
    if out.ndim == 3:
      out = out[0]
    if out.shape[0] > out.shape[1]:  # [N, C] -> [C, N]
      out = out.T
    return out, r, left, top, w, h
