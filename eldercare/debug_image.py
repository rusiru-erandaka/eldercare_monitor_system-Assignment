"""Inspect object and pose model output for a single image.

  python -m eldercare.debug_image --image frame.png --det-model ... --pose-model ...
The command prints detections, fused observations and activity classifications,
then writes an annotated image.
"""
import argparse

import cv2

from .classifier import classify_tick
from .config import Config
from .features import compute_features
from .perception.detector import ObjectDetector
from .perception.fusion import fuse
from .perception.pose import PoseEstimator
from .schema import BoundingBox


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--image", required=True)
  p.add_argument("--det-model", default="OpenVINO/YOLO11n-fp16-ov")
  p.add_argument("--pose-model", default="../yolov8n-pose_openvino_model/yolov8n-pose.xml")
  p.add_argument("--device", default="GPU")
  p.add_argument("--bed-roi")
  p.add_argument("--no-normalize", action="store_true")
  p.add_argument("--out", default="debug_out.png")
  a = p.parse_args()

  cfg = Config()
  frame = cv2.imread(a.image)
  if frame is None:
    raise SystemExit(f"Unable to read image: {a.image}")
  h, w = frame.shape[:2]
  det = ObjectDetector(a.det_model, cfg, a.device, normalize=not a.no_normalize).detect(frame)
  poses = PoseEstimator(a.pose_model, None, a.device, cfg.pose_conf, cfg.pose_nms_iou,
                        cfg.pose_max_persons, normalize=not a.no_normalize).infer(frame)
  bed = BoundingBox(*[float(v) for v in a.bed_roi.split(",")]) if a.bed_roi else \
      (max(det.beds, key=lambda b: b.confidence) if det.beds else None)

  print(f"beds   : {[(b.to_list(), round(b.confidence, 2)) for b in det.beds]}")
  print(f"persons: {[(b.to_list(), round(b.confidence, 2)) for b in det.persons]}")
  print(f"chairs : {[(b.to_list(), round(b.confidence, 2)) for b in det.chairs]}")
  for i, ps in enumerate(poses):
    print(f"pose {i}: box={ps.bbox.to_list()} score={ps.pose_conf:.2f} visible_kps={ps.n_visible(cfg.kp_conf)}")
  obs = fuse(det.persons, True, poses, bed, w, h, cfg)
  print(f"fused observations kept: {len(obs)}  (ghost poses rejected: {len(poses) - sum(1 for o in obs if o.source != 'det')})")

  if bed:
    cv2.rectangle(frame, (int(bed.xmin), int(bed.ymin)), (int(bed.xmax), int(bed.ymax)), (255, 120, 0), 2)
  for o in obs:
    f = compute_features(o, bed, det.chairs, None, cfg, w, h)
    ts = classify_tick(o, f, cfg)
    print(f"  obs[{o.source}] -> {ts.state.value} conf={ts.confidence} reason={ts.reason} "
          f"tilt={f.torso_tilt} knee_max={f.knee_max} overlap={f.bed_overlap}")
    b = o.bbox
    cv2.rectangle(frame, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 255, 255), 2)
    cv2.putText(frame, ts.state.value, (int(b.xmin), max(15, int(b.ymin) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if o.keypoints:
      for kp in o.keypoints.values():
        if kp.confidence > cfg.kp_conf:
          cv2.circle(frame, (int(kp.x), int(kp.y)), 4, (0, 0, 255), -1)
  if not cv2.imwrite(a.out, frame):
    raise SystemExit(f"Unable to write image: {a.out}")
  print(f"saved {a.out}")


if __name__ == "__main__":
  main()
