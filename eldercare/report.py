"""Turn segments/events/alerts into the deliverable outputs (timeline, durations, JSON)."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

from .alerts import Alert, evaluate as evaluate_alerts
from .config import Config
from .events import Event, Run, detect_events
from .schema import (ActivityState as S, IN_BED_STATES, OUT_OF_BED_STATES,
                     TickRecord, fmt_dur, fmt_mmss)
from .temporal import (Segment, TemporalStateMachine, segments_from_ticks,
                       tick_period)

CORE = [S.LYING_IN_BED, S.SITTING_ON_BED, S.SITTING_OUTSIDE_BED, S.STANDING,
        S.WALKING, S.OUT_OF_BED, S.UNKNOWN]


def durations(segments: List[Segment]) -> Dict[S, float]:
  d: Dict[S, float] = {}
  for s in segments:
    d[s.state] = d.get(s.state, 0.0) + s.duration
  return d


def timeline_lines(segments: List[Segment]) -> List[str]:
  return [f"{fmt_mmss(s.start)} – {fmt_mmss(s.end)}  {s.state.value}" for s in segments]


def build_report(segments: List[Segment], events: List[Event], runs: List[Run],
                 alerts: List[Alert], overall: str, t_start: float,
                 t_end: float) -> dict:
  total = max(0.0, t_end - t_start)
  dur = durations(segments)
  states = list(CORE) + ([S.LYING_ON_FLOOR] if S.LYING_ON_FLOOR in dur else [])
  # Location durations are factual state accounting. Event reasoning may bridge
  # UNKNOWN gaps to confirm that an exit continued, but that inference must not
  # rewrite uncertain time as observed in/out-of-bed time in the final report.
  in_bed = sum(s.duration for s in segments if s.state in IN_BED_STATES)
  out_bed = sum(s.duration for s in segments if s.state in OUT_OF_BED_STATES)
  unk_loc = sum(s.duration for s in segments if s.state == S.UNKNOWN)
  exits = [e for e in events if e.event == "bed_exit"]
  returns = [e for e in events if e.event == "bed_return"]
  longest_out = max([r.duration for r in runs if r.loc == "out"] or [0.0])

  return {
      "total_observation_time": fmt_dur(total),
      "activity_summary": {s.value.lower(): fmt_dur(dur.get(s, 0.0)) for s in states},
      "bed_summary": {
          "time_in_bed": fmt_dur(in_bed),
          "time_out_of_bed": fmt_dur(out_bed),
          "time_location_unknown": fmt_dur(unk_loc),
          "bed_exit_count": len(exits),
          "bed_return_count": len(returns),
      },
      "summary": {
          "observation_duration_sec": round(total, 2),
          "activity_duration_sec": {
              s.value.lower(): round(dur.get(s, 0.0), 2) for s in states},
          "bed_exit_count": len(exits),
          "bed_return_count": len(returns),
          "total_in_bed_sec": round(in_bed, 2),
          "total_out_of_bed_sec": round(out_bed, 2),
          "total_location_unknown_sec": round(unk_loc, 2),
          "longest_out_of_bed_period_sec": round(longest_out, 2),
          "final_state": segments[-1].state.value.lower() if segments else "unknown",
          "overall_decision": overall,
      },
      "events": [e.to_dict() for e in events],
      "alerts": [a.to_dict() for a in alerts],
      "timeline": [s.to_dict() for s in segments],
  }


def write_outputs(outdir: str, report: dict, segments: List[Segment]):
  os.makedirs(outdir, exist_ok=True)
  with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in report.items() if k != "timeline"}, f, indent=2)
  with open(os.path.join(outdir, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump(report["timeline"], f, indent=2)
  with open(os.path.join(outdir, "timeline.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(timeline_lines(segments)) + "\n")
  with open(os.path.join(outdir, "events.json"), "w", encoding="utf-8") as f:
    json.dump({"events": report["events"], "alerts": report["alerts"]}, f, indent=2)


def analyze(records: Iterable[TickRecord], cfg: Config):
  """Build the activity timeline, events, alerts and final report."""
  records = sorted(list(records), key=lambda r: r.t)
  if not records:
    raise ValueError("no ticks to analyse")
  segments = segments_from_ticks(records, cfg)
  t_first = records[0].t
  t_end = records[-1].t + tick_period(records, cfg)
  # The stable timeline is retained only as context for event/alert reasoning.
  # It must never be used for the reported activity durations, because doing so
  # hides short model outputs (especially STANDING/WALKING/UNKNOWN).
  reasoning = TemporalStateMachine(cfg)
  for rec in records:
    reasoning.update(rec)
  reasoning.finalize(t_end)
  events, runs = detect_events(reasoning.segments, cfg, records)
  alerts, overall = evaluate_alerts(reasoning.segments, events, runs, cfg, records)
  rep = build_report(segments, events, runs, alerts, overall, t_first, t_end)
  rep["analysis"] = {
      "timeline_source": "raw_perception_ticks",
      "tick_count": len(records),
      "measured_tick_period_sec": round(tick_period(records, cfg), 4),
      "smoothing_applied_to_activity_durations": False,
      "temporal_reasoning_used_for_events_only": True,
  }
  return segments, events, runs, alerts, overall, rep
