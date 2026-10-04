import os, tempfile
from eldercare.config import Config
from eldercare.evaluate import evaluate, parse_gt_timeline
from eldercare.report import analyze
from tests.synth import make_stream

def test_evaluate_roundtrip():
  # Keep this seeded evaluation fixture at its original sampling rate; default
  # production inference frequency is tested separately in test_perception.
  cfg = Config(analysis_fps=2.0)
  gt_text = """00:00 – 01:00 LYING_IN_BED
01:00 - 01:10 SITTING_ON_BED
01:10 - 01:20 STANDING
01:20 - 02:00 WALKING
02:00 - 02:20 SITTING_ON_BED
02:20 - 03:00 LYING_IN_BED
"""
  p = os.path.join(tempfile.mkdtemp(), "gt.txt"); open(p, "w").write(gt_text)
  gt = parse_gt_timeline(p)
  recs = make_stream([(60, "LYING_IN_BED"), (10, "SITTING_ON_BED"), (10, "STANDING"),
                      (40, "WALKING"), (20, "SITTING_ON_BED"), (40, "LYING_IN_BED")], cfg, noise=0.1, seed=2)
  segs, events, runs, alerts, overall, rep = analyze(recs, cfg)
  res = evaluate(segs, rep, gt, None, cfg)
  assert res["accuracy"] > 0.9
  assert res["bed_events"]["bed_exit"]["recall"] == 1.0
  assert res["bed_events"]["bed_exit"]["precision"] == 1.0
  assert res["duration_mae_sec"] < 6


def test_parse_gt_timeline_with_fractional_seconds():
  gt_text = """00:00.00 - 00:02.50 LYING_IN_BED
00:02.50 - 00:07.25 SITTING_ON_BED
"""
  p = os.path.join(tempfile.mkdtemp(), "gt_fractional.txt")
  open(p, "w").write(gt_text)
  gt = parse_gt_timeline(p)
  assert [(s.start, s.end) for s in gt] == [(0.0, 2.5), (2.5, 7.25)]
