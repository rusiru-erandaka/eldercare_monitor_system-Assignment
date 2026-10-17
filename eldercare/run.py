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
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
from pydantic import ValidationError

from .config import Config
from .logging_utils import configure_logging, validate_log_file
from .models import model_directory, model_xml, verify_model
from .pipeline import (PerceptionPipeline, draw_overlay, iter_video, load_ticks,
                       save_ticks, validate_ticks_video)
from .provenance import (artifact_hashes, config_hash, input_identity,
                         manifest_template, model_file_hashes, prepare_output,
                         utc_now, write_manifest)
from .report import analyze, timeline_lines, write_outputs
from .schema import BoundingBox

logger = logging.getLogger("eldercare.run")  # works for python -m and console entry point


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
  parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                      default="INFO")
  parser.add_argument("--log-file", help="optional JSON-lines log file")
  parser.add_argument("--config", help="YAML profile; CLI values override it")
  parser.add_argument("--overwrite", action="store_true",
                      help="archive approved existing output artifacts before running")
  parser.add_argument(
      "--analysis-fps", type=float,
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
        logger.info("Pinned model file verified", extra={
            "event": "model_verified", "model": name,
            "path": path, "sha256": digest,
        })
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
  model_runtime = {
      "detector": {
          "execution_devices": pipeline.detector.execution_devices,
          "files": model_file_hashes(pipeline.detector.resolved_model_path),
      },
      "pose": {
          "execution_devices": pipeline.pose.execution_devices,
          "files": model_file_hashes(pipeline.pose.model_xml_path),
      },
  }
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
  logger.info("Video perception finished", extra={
      "event": "perception_complete", "tick_count": len(records),
      "elapsed_sec": round(elapsed, 3),
  })
  return records, model_runtime


def load_or_create_records(args, cfg: Config):
  if args.from_ticks:
    if args.video:
      try:
        validate_ticks_video(args.from_ticks, args.video)
      except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return load_ticks(args.from_ticks), None
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
  if args.analysis_fps is not None and args.analysis_fps <= 0:
    raise SystemExit("--analysis-fps must be greater than zero")
  if args.max_seconds is not None and args.max_seconds <= 0:
    raise SystemExit("--max-seconds must be greater than zero")


def build_config(args) -> Config:
  overrides = {}
  if args.analysis_fps is not None:
    overrides["analysis_fps"] = args.analysis_fps
  if args.vlm_max_reviews is not None:
    overrides["vlm_max_reviews"] = args.vlm_max_reviews
  try:
    return (Config.from_yaml(args.config, overrides) if args.config else
            Config(**overrides))
  except (OSError, ValueError, ValidationError) as exc:
    raise SystemExit(f"Invalid configuration: {exc}") from exc


def main(argv: Optional[Sequence[str]] = None):
  args = parse_args(argv)
  validate_args(args)
  cfg = build_config(args)
  original_ticks = Path(args.from_ticks).resolve() if args.from_ticks else None
  original_out = Path(args.out).absolute().resolve()
  if args.from_ticks and args.video:
    validate_ticks_video(args.from_ticks, args.video)
  source = input_identity(args.video, args.from_ticks)
  settings = cfg.to_dict()
  try:
    validate_log_file(args.log_file, original_out)
    output, archived = prepare_output(
        args.out, source["source_sha256"], config_hash(settings), args.overwrite)
  except ValueError as exc:
    raise SystemExit(str(exc)) from exc

  if archived and original_ticks is not None and original_ticks.parent == original_out:
    # The user may intentionally re-analyse ticks in the same output folder.
    # Restore exact input bytes after archiving, before reading the cache.
    for filename in ("ticks.jsonl", "ticks.meta.json"):
      old_file = archived / filename
      if old_file.is_file():
        shutil.copy2(old_file, output / filename)
    args.from_ticks = str(output / original_ticks.name)
  args.out = str(output)
  configure_logging(args.log_level, args.log_file)
  command = ["eldercare.run", *(argv if argv is not None else sys.argv[1:])]
  manifest = manifest_template(command, source, settings, args.device)
  write_manifest(output, manifest)
  logger.info("Run started", extra={
      "event": "run_start", "run_id": manifest["run_id"],
      "requested_device": args.device,
  })

  try:
    records, model_runtime = load_or_create_records(args, cfg)
    if model_runtime is not None:
      for name, runtime in model_runtime.items():
        manifest["models"][name].update(runtime)
        logger.info("Model execution device recorded", extra={
            "event": "model_execution", "run_id": manifest["run_id"],
            "model": name, "requested_device": args.device,
            "execution_devices": runtime["execution_devices"],
        })
        for path, digest in runtime["files"].items():
          logger.info("Loaded model file hashed", extra={
              "event": "model_loaded", "run_id": manifest["run_id"],
              "model": name, "path": path, "sha256": digest,
          })
        if args.device.upper() == "GPU" and "GPU" not in runtime["execution_devices"]:
          warning = f"{name} did not execute on GPU; actual devices: {runtime['execution_devices']}"
          manifest["warnings"].append(warning)
          logger.warning(warning, extra={
              "event": "device_fallback", "run_id": manifest["run_id"],
              "model": name, "requested_device": args.device,
              "execution_devices": runtime["execution_devices"],
          })
    else:
      logger.info("Using cached ticks; no inference device executed", extra={
          "event": "inference_skipped", "run_id": manifest["run_id"],
      })
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
      json.dump(settings, f, indent=2)
    manifest["finished_at_utc"] = utc_now()
    manifest["status"] = "complete"
    logger.info("Run completed", extra={
        "event": "run_end", "run_id": manifest["run_id"],
        "status": "complete", "tick_count": len(records),
    })
    manifest["artifacts"] = artifact_hashes(output)
    write_manifest(output, manifest)
    print_results(segments, report, overall, vlm_result)
  except Exception as exc:
    manifest["finished_at_utc"] = utc_now()
    manifest["status"] = "failed"
    manifest["error_type"] = type(exc).__name__
    logger.error("Run failed", extra={
        "event": "run_end", "run_id": manifest["run_id"],
        "status": "failed", "error_type": type(exc).__name__,
    })
    manifest["artifacts"] = artifact_hashes(output)
    write_manifest(output, manifest)
    raise


if __name__ == "__main__":
  main()
