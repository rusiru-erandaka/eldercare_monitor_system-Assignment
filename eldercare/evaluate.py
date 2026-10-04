"""Evaluation against hand-labelled ground truth.

Ground-truth timeline format, one segment per line:
    00:00 - 04:32  LYING_IN_BED
    04:32 - 05:08  SITTING_ON_BED
Optional GT events JSON: {"bed_exits": [308, 540], "bed_returns": [600]}  (seconds).
If omitted, GT events are derived from the GT timeline with the same event rules.

Reports: frame-level accuracy (1 Hz), confusion matrix, per-state precision/recall,
per-state duration error, bed-exit/return precision/recall/F1 (+ false positives),
and the longest mismatch intervals (use them as your failure-case examples).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .config import Config
from .events import detect_events
from .schema import ActivityState as S, fmt_dur, fmt_mmss
from .temporal import Segment

# MM:SS, HH:MM:SS, and fractional seconds such as MM:SS.ss.
_TS = r"(\d+(?::\d{2}){1,2}(?:\.\d+)?)"
LINE = re.compile(rf"^\s*{_TS}\s*[-–—]\s*{_TS}\s+([A-Z_]+)\s*$")


def parse_ts(s: str) -> float:
  parts = [float(p) for p in s.split(":")]
  sec = 0.0
  for p in parts:
    sec = sec * 60 + p
  return sec


def parse_gt_timeline(path: str) -> List[Segment]:
  segs = []
  with open(path, encoding="utf-8") as f:
    for ln in f:
      if not ln.strip() or ln.lstrip().startswith("#"):
        continue
      m = LINE.match(ln.strip())
      if not m:
        raise ValueError(f"Bad GT line: {ln!r}")
      segs.append(Segment(S(m.group(3)), parse_ts(m.group(1)), parse_ts(m.group(2)), 1.0, 1))
  return segs


def load_pred(outdir: str) -> Tuple[List[Segment], dict]:
  with open(os.path.join(outdir, "timeline.json"), encoding="utf-8") as f:
    tl = json.load(f)
  segs = [Segment(S(d["state"]), d["start"], d["end"],
                  d.get("confidence", 1.0), 1,
                  track_id=d.get("track_id")) for d in tl]
  with open(os.path.join(outdir, "events.json"), encoding="utf-8") as f:
    ev = json.load(f)
  return segs, ev


def state_at(segs: List[Segment], t: float) -> Optional[str]:
  for s in segs:
    if s.start <= t < s.end:
      return s.state.value
  return None


def event_times(events_json: dict, name: str) -> List[float]:
  return [parse_ts(e["start_time"]) for e in events_json["events"] if e["event"] == name]


def match_events(pred: List[float], gt: List[float], tol: float):
  pred_left, tp, fn = list(pred), 0, []
  for g in gt:
    best = min(pred_left, key=lambda p: abs(p - g), default=None)
    if best is not None and abs(best - g) <= tol:
      tp += 1
      pred_left.remove(best)
    else:
      fn.append(g)
  fp = pred_left
  prec = tp / (tp + len(fp)) if tp + len(fp) else 1.0
  rec = tp / (tp + len(fn)) if tp + len(fn) else 1.0
  f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
  return {"tp": tp, "false_positives": [fmt_mmss(x) for x in fp],
          "false_negatives": [fmt_mmss(x) for x in fn],
          "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def evaluate(pred_segs: List[Segment], pred_events: dict, gt_segs: List[Segment],
             gt_events: Optional[dict], cfg: Config, tol: float = 10.0,
             hz: float = 1.0) -> dict:
  gt_has_floor = any(s.state == S.LYING_ON_FLOOR for s in gt_segs)
  def norm(st):
    return S.OUT_OF_BED.value if (st == S.LYING_ON_FLOOR.value and not gt_has_floor) else st

  t_end = min(gt_segs[-1].end, pred_segs[-1].end)
  n = int(t_end * hz)
  conf: Dict[str, Counter] = defaultdict(Counter)
  correct, total = 0, 0
  mismatch: List[Tuple[float, str, str]] = []
  for i in range(n):
    t = i / hz + 0.5 / hz
    g, p = state_at(gt_segs, t), state_at(pred_segs, t)
    if g is None or p is None:
      continue
    p = norm(p)
    conf[g][p] += 1
    total += 1
    correct += int(g == p)
    if g != p:
      mismatch.append((t, g, p))

  states = sorted({s for c in conf.values() for s in c} | set(conf))
  per_state = {}
  for st in states:
    tp = conf[st][st]
    gt_n = sum(conf[st].values())
    pr_n = sum(conf[g][st] for g in conf)
    per_state[st] = {"precision": round(tp / pr_n, 3) if pr_n else None,
                     "recall": round(tp / gt_n, 3) if gt_n else None, "gt_sec": gt_n}

  def dur(segs, collapse):
    d: Dict[str, float] = defaultdict(float)
    for s in segs:
      st = norm(s.state.value) if collapse else s.state.value
      d[st] += max(0.0, min(s.end, t_end) - min(s.start, t_end))
    return d
  gd, pd_ = dur(gt_segs, False), dur(pred_segs, True)
  duration = {}
  for st in sorted(set(gd) | set(pd_)):
    duration[st] = {"ground_truth": fmt_dur(gd.get(st, 0)), "predicted": fmt_dur(pd_.get(st, 0)),
                    "abs_error_sec": round(abs(gd.get(st, 0) - pd_.get(st, 0)), 1)}
  mae = sum(v["abs_error_sec"] for v in duration.values()) / max(1, len(duration))

  if gt_events is None:
    ev, _ = detect_events(gt_segs, cfg)
    gt_events = {"bed_exits": [e.start_time for e in ev if e.event == "bed_exit"],
                 "bed_returns": [e.start_time for e in ev if e.event == "bed_return"]}
  bed = {"bed_exit": match_events(event_times(pred_events, "bed_exit"), gt_events["bed_exits"], tol),
         "bed_return": match_events(event_times(pred_events, "bed_return"), gt_events["bed_returns"], tol),
         "tolerance_sec": tol}

  # failure cases: merge consecutive mismatching seconds with same (gt, pred)
  cases, cur = [], None
  for t, g, p in mismatch:
    if cur and cur["gt"] == g and cur["pred"] == p and t - cur["end"] <= 1.5 / hz:
      cur["end"] = t
    else:
      cur = {"start": t, "end": t, "gt": g, "pred": p}
      cases.append(cur)
  cases = [dict(c, duration_sec=round(c["end"] - c["start"] + 1 / hz, 1)) for c in cases]
  cases.sort(key=lambda c: -c["duration_sec"])
  for c in cases:
    c["start"], c["end"] = fmt_mmss(c["start"]), fmt_mmss(c["end"])

  return {"accuracy": round(correct / total, 4) if total else None, "evaluated_sec": total,
          "per_state": per_state, "confusion_matrix": {g: dict(c) for g, c in conf.items()},
          "duration_error": duration, "duration_mae_sec": round(mae, 1),
          "bed_events": bed, "top_failure_intervals": cases[:8]}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument("--pred", required=True, help="results dir from run.py")
  ap.add_argument("--gt", required=True, help="GT timeline text file")
  ap.add_argument("--gt-events")
  ap.add_argument("--tolerance", type=float, default=10.0)
  ap.add_argument("--out")
  a = ap.parse_args()
  segs, ev = load_pred(a.pred)
  gt = parse_gt_timeline(a.gt)
  if a.gt_events:
    with open(a.gt_events, encoding="utf-8") as f:
      gt_ev = json.load(f)
  else:
    gt_ev = None
  res = evaluate(segs, ev, gt, gt_ev, Config(), a.tolerance)
  out = a.out or os.path.join(a.pred, "evaluation.json")
  with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
