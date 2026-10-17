"""Command-line entry point for the eldercare monitoring pipeline.

Examples
--------
Run perception and activity analysis::

  python -m eldercare.run --video room.mp4 --device GPU --show --out results

Reuse verified perception records::

  python -m eldercare.run --from-ticks results/ticks.jsonl --out results
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2

from .config import Config
from .models import model_directory, model_xml, verify_model
from .pipeline import (PerceptionPipeline, draw_overlay, iter_video, load_ticks,
                       save_ticks, validate_ticks_video)
from .report import analyze, timeline_lines, write_outputs
from .schema import BoundingBox


def parse_bed_roi(value: str) -> BoundingBox:
  """Parse a manual bed region in x1,y1,x2,y2 format."""
  try:
    coordinates = [float(item.strip()) for item in value.split(",")]
  except ValueError as exc:
    raise argparse.ArgumentTypeError("bed ROI values must be numbers") from exc
  if len(coordinates) != 4:
    raise argparse.ArgumentTypeError("bed ROI must contain four comma-separated values")
  x1, y1, x2, y2 = coordinates
  if x2 <= x1 or y2 <= y1:
    raise argparse.ArgumentTypeError("bed ROI must satisfy x2 > x1 and y2 > y1")
  return BoundingBox(x1, y1, x2, y2)


def parse_args(argv: Optional[Sequence[str]] = None):
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--video")
  parser.add_argument("--from-ticks", help="analyse verified cached tick records")
  parser.add_argument("--det-model", default=str(model_directory("detector")))
  parser.add_argument(
      "--pose-model",
      default=str(model_xml("pose")))
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--out", default="results")
  parser.add_argument(
      "--analysis-fps", type=float, default=4.0,
      help="pose sampling rate; detector runs at half this rate (default: 4)")
  parser.add_argument(
      "--bed-roi", type=parse_bed_roi,
      help="manual bed box x1,y1,x2,y2 in pixels (overrides detection)")
  parser.add_argument(
      "--no-normalize", action="store_true",
      help="skip RGB/255 preprocessing when it is already part of the IR model")
  parser.add_argument("--show", action="store_true", help="show the annotated video")
  parser.add_argument(
      "--save-video", action="store_true", help="write annotated overlay.mp4")
  parser.add_argument("--max-seconds", type=float)
  parser.add_argument(
      "--vlm", action="store_true",
      help="review ambiguous or safety-relevant windows with Gemini")
  parser.add_argument(
      "--vlm-model", default="gemini-3.5-flash-lite",
      help="Gemini model used for visual review")
  parser.add_argument(
      "--vlm-api-key-env", default="GEMINI_API_KEY",
      help="environment variable containing the Gemini API key")
  parser.add_argument(
      "--vlm-max-reviews", type=int,
      help="maximum Gemini calls (default from Config: 6)")
  parser.add_argument(
      "--vlm-refresh", action="store_true",
      help="ignore an existing matching VLM review cache")
  return parser.parse_args(argv)


@dataclass
class PreviewClock:
  """Synchronise sampled preview frames with their source-video timestamps."""

  video_start: Optional[float] = None
  wall_start: Optional[float] = None

  def delay_ms(self, video_time: float) -> int:
    if self.video_start is None:
      self.video_start = video_time
      self.wall_start = time.monotonic()
    assert self.wall_start is not None
    target = self.wall_start + (video_time - self.video_start)
    return max(1, int(round((target - time.monotonic()) * 1000)))


def build_pipeline(args, cfg: Config) -> PerceptionPipeline:
  """Initialise the object detector, pose estimator and perception pipeline."""
  from .perception.detector import ObjectDetector
  from .perception.pose import PoseEstimator

  for name, selected in (("detector", args.det_model), ("pose", args.pose_model)):
    pinned = model_directory(name) if name == "detector" else model_xml(name)
    if os.path.abspath(selected) == str(pinned.resolve()):
      for path, digest in verify_model(name).items():
        print(f"[{name}] verified model {path} sha256={digest}")
  normalize = not args.no_normalize
  detector = ObjectDetector(args.det_model, cfg, args.device, normalize=normalize)
  pose = PoseEstimator(
      args.pose_model, None, args.device, cfg.pose_conf, cfg.pose_nms_iou,
      cfg.pose_max_persons, normalize=normalize)
  return PerceptionPipeline(cfg, detector, pose, args.bed_roi)


def open_video_writer(path: str, frame, fps: float):
  """Create an MP4 writer using the dimensions of the first processed frame."""
  height, width = frame.shape[:2]
  writer = cv2.VideoWriter(
      path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
  if not writer.isOpened():
    writer.release()
    raise RuntimeError(f"Unable to create output video: {path}")
  return writer


def run_perception(args, cfg: Config):
  """Process a video and return timestamped activity records."""
  pipeline = build_pipeline(args, cfg)
  records = []
  writer = None
  preview = PreviewClock()
  started = time.perf_counter()

  try:
    for video_time, frame in iter_video(args.video, cfg.analysis_fps):
      if args.max_seconds is not None and video_time > args.max_seconds:
        break

      record = pipeline.process(frame, video_time)
      records.append(record)

      if not (args.show or args.save_video):
        continue
      draw_overlay(frame, pipeline, record, cfg)

      if args.save_video:
        if writer is None:
          writer = open_video_writer(
              os.path.join(args.out, "overlay.mp4"), frame, cfg.analysis_fps)
        writer.write(frame)

      if args.show:
        cv2.imshow("eldercare", frame)
        if cv2.waitKey(preview.delay_ms(video_time)) & 0xFF == ord("q"):
          break
  finally:
    if writer is not None:
      writer.release()
    if args.show:
      cv2.destroyAllWindows()

  save_ticks(os.path.join(args.out, "ticks.jsonl"), records, args.video)
  elapsed = time.perf_counter() - started
  print(f"[run] {len(records)} ticks in {elapsed:.1f}s wall time")
  return records


def load_or_create_records(args, cfg: Config):
  if args.from_ticks:
    if args.video:
      try:
        validate_ticks_video(args.from_ticks, args.video)
      except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return load_ticks(args.from_ticks)
  return run_perception(args, cfg)


def apply_vlm_review(args, cfg: Config, records, events, alerts, overall: str,
                     report: dict):
  """Run selective Gemini review and apply escalation-only safety decisions."""
  from .alerts import apply_vlm_safety_escalations
  from .vlm import analyze_video, report_summary

  output_path = os.path.join(args.out, "vlm_reviews.json")
  result = analyze_video(
      args.video, records, events, cfg, output_path,
      model=args.vlm_model, api_key_env=args.vlm_api_key_env,
      refresh=args.vlm_refresh, max_reviews=args.vlm_max_reviews)
  report["agentic_analysis"] = report_summary(result)

  alerts, overall, escalations = apply_vlm_safety_escalations(
      result, alerts, overall, cfg, records)
  report["alerts"] = [alert.to_dict() for alert in alerts]
  report["summary"]["overall_decision"] = overall
  report["agentic_analysis"]["safety_escalations"] = escalations
  return result, overall


def print_results(segments, report: dict, overall: str, vlm_result=None):
  print("\n=== TIMELINE ===")
  print("\n".join(timeline_lines(segments)))
  print("\n=== SUMMARY ===")
  summary = {
      key: report[key]
      for key in ("total_observation_time", "activity_summary", "bed_summary")
  }
  print(json.dumps(summary, indent=2))
  print("\n=== EVENTS ===")
  print(json.dumps(report["events"], indent=2))

  if vlm_result is not None:
    print("\n=== AGENTIC VLM ANALYSIS ===")
    print(json.dumps(report["agentic_analysis"], indent=2))
    for review in vlm_result.get("reviews", []):
      window = review["window"]
      answer = review.get("vlm_result", {})
      conclusion = answer.get("conclusion", "INSUFFICIENT_EVIDENCE")
      confidence = answer.get("confidence", 0)
      print(
          f"  {window['start']}–{window['end']} {window['trigger']}: "
          f"{conclusion} conf={confidence:.2f} "
          f"accepted={review.get('accepted', False)}")

  print(f"\nOVERALL DECISION: {overall}")
  for alert in report["alerts"]:
    print(
        f"  [{alert['level']}] {alert['time']} {alert['rule']}: "
        f"{alert['message']}")


def validate_args(args):
  if not args.video and not args.from_ticks:
    raise SystemExit("--video or --from-ticks is required")
  if args.vlm and not args.video:
    raise SystemExit("--vlm requires --video so review frames can be extracted")
  if args.analysis_fps <= 0:
    raise SystemExit("--analysis-fps must be greater than zero")
  if args.max_seconds is not None and args.max_seconds <= 0:
    raise SystemExit("--max-seconds must be greater than zero")


def main(argv: Optional[Sequence[str]] = None):
  args = parse_args(argv)
  validate_args(args)
  cfg = Config(analysis_fps=args.analysis_fps)
  os.makedirs(args.out, exist_ok=True)

  records = load_or_create_records(args, cfg)
  segments, events, _runs, alerts, overall, report = analyze(records, cfg)
  report["analysis"]["input_type"] = "ticks" if args.from_ticks else "video"
  report["analysis"]["input_path"] = os.path.abspath(
      args.from_ticks or args.video)
  if args.video:
    report["analysis"]["source_video"] = os.path.abspath(args.video)

  vlm_result = None
  if args.vlm:
    vlm_result, overall = apply_vlm_review(
        args, cfg, records, events, alerts, overall, report)

  write_outputs(args.out, report, segments)
  with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg.to_dict(), f, indent=2)
  print_results(segments, report, overall, vlm_result)


if __name__ == "__main__":
  main()
