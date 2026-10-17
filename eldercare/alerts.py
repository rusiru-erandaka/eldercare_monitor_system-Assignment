"""Contextual alert policy (NORMAL < MONITOR < ALERT) - every rule is explicit.

Rationale
---------
ALERT:
  A1  VLM-confirmed lying on the floor immediately, or CV-only floor lying for
      the configured duration
  A3  out of bed AND actually unobserved for >= 40 minutes
MONITOR:
  M2  sitting at the bed edge (<75% person-box overlap) for >= 3 minutes
  M3  out of bed for >= 30 minutes
  M4  total UNKNOWN evidence >= 90 seconds between two known activities
NORMAL:
  lying/sitting/standing/walking, confirmed bed exits and returns, and middle-of-bed
  sitting (including reading/resting for 10 minutes or longer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Config
from .events import Event, Run
from .schema import ActivityState as S, TickRecord, fmt_hhmmss
from .temporal import Segment, tick_period

LEVELS = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


def _max(a: str, b: str) -> str:
  return a if LEVELS[a] >= LEVELS[b] else b


@dataclass
class Alert:
  time: float
  level: str
  rule: str
  message: str

  def to_dict(self) -> dict:
    return {"time": fmt_hhmmss(self.time), "level": self.level,
            "rule": self.rule, "message": self.message}


def _tick_spans(records: List[TickRecord], cfg: Config):
  """Yield each raw tick with the exact interval it represents."""
  records = sorted(records, key=lambda r: r.t)
  if not records:
    return []
  last_dt = tick_period(records, cfg)
  return [(rec, rec.t,
           records[i + 1].t if i + 1 < len(records) else rec.t + last_dt)
          for i, rec in enumerate(records)]


def _unknown_episodes(records: List[TickRecord], cfg: Config):
  """Merge all adjacent UNKNOWN ticks even when reason/track/source changes."""
  episodes, start, end = [], None, None
  for rec, t0, t1 in _tick_spans(records, cfg):
    unknown = not rec.observed or rec.state == S.UNKNOWN.value
    if unknown:
      if start is None:
        start = t0
      end = t1
    elif start is not None:
      episodes.append((start, end))
      start = end = None
  if start is not None:
    episodes.append((start, end))
  return episodes


def _unseen_episodes(records: List[TickRecord], cfg: Config):
  """Continuous periods with no elder observation (not merely ambiguous posture)."""
  episodes, start, end = [], None, None
  for rec, t0, t1 in _tick_spans(records, cfg):
    if not rec.observed:
      if start is None:
        start = t0
      end = t1
    elif start is not None:
      episodes.append((start, end))
      start = end = None
  if start is not None:
    episodes.append((start, end))
  return episodes


def _floor_episodes(records: List[TickRecord], cfg: Config):
  """Find continuous, observed floor postures in the raw activity ticks."""
  episodes, start, end = [], None, None
  for rec, t0, t1 in _tick_spans(records, cfg):
    on_floor = (rec.observed and rec.state == S.LYING_ON_FLOOR.value and
                rec.confidence >= cfg.min_conf)
    if on_floor:
      if start is None:
        start = t0
      end = t1
    elif start is not None:
      episodes.append((start, end))
      start = end = None
  if start is not None:
    episodes.append((start, end))
  return episodes


def _edge_sitting_episodes(records: List[TickRecord], cfg: Config):
  """Return edge-sitting periods, bridging only brief UNKNOWN detection gaps."""
  pieces = []
  for rec, t0, t1 in _tick_spans(records, cfg):
    if (rec.observed and rec.state == S.SITTING_ON_BED.value and
        rec.bed_overlap is not None and rec.bed_overlap < cfg.edge_sit_overlap_max):
      label = "edge"
    elif not rec.observed or rec.state == S.UNKNOWN.value:
      label = "unknown"
    else:
      label = "other"
    if pieces and pieces[-1][0] == label:
      pieces[-1] = (label, pieces[-1][1], t1)
    else:
      pieces.append((label, t0, t1))

  episodes, i = [], 0
  while i < len(pieces):
    label, start, end = pieces[i]
    if label != "edge":
      i += 1
      continue
    j = i
    while (j + 2 < len(pieces) and pieces[j + 1][0] == "unknown" and
           pieces[j + 1][2] - pieces[j + 1][1] < cfg.unknown_gap_sec and
           pieces[j + 2][0] == "edge"):
      end = pieces[j + 2][2]
      j += 2
    episodes.append((start, end))
    i = j + 1
  return episodes


def evaluate(segments: List[Segment], events: List[Event], runs: List[Run],
             cfg: Config, records: Optional[List[TickRecord]] = None) -> Tuple[List[Alert], str]:
  alerts: List[Alert] = []

  floor_periods = (_floor_episodes(records, cfg) if records is not None else
                   [(s.start, s.end) for s in segments if s.state == S.LYING_ON_FLOOR])
  for start, end in floor_periods:
    duration = end - start
    if duration >= cfg.alert_floor_lying_sec:
      alerts.append(Alert(start + cfg.alert_floor_lying_sec, "ALERT", "A1_floor_lying",
                          f"Lying on the floor for {duration:.0f}s (possible fall)"))

  if records is not None:
    for start, end in _edge_sitting_episodes(records, cfg):
      if end - start >= cfg.monitor_sit_edge_sec:
        alerts.append(Alert(start + cfg.monitor_sit_edge_sec, "MONITOR",
                            "M2_long_edge_sit",
                            f"Sitting at bed edge for {end - start:.0f}s"))
    for start, end in _unknown_episodes(records, cfg):
      if end - start >= cfg.monitor_unknown_sec:
        alerts.append(Alert(start + cfg.monitor_unknown_sec, "MONITOR", "M4_unknown",
                            f"Activity undetermined for {end - start:.0f}s"))
  else:
    # Geometry-free/manual timelines cannot distinguish bed edge from middle.
    for s in segments:
      if s.state == S.UNKNOWN and s.duration >= cfg.monitor_unknown_sec:
        alerts.append(Alert(s.start + cfg.monitor_unknown_sec, "MONITOR", "M4_unknown",
                            f"Activity undetermined for {s.duration:.0f}s"))

  for r in runs:
    if r.loc != "out":
      continue
    if r.duration >= cfg.monitor_out_of_bed_sec:
      alerts.append(Alert(r.start + cfg.monitor_out_of_bed_sec, "MONITOR", "M3_long_absence",
                          f"Out of bed for {r.duration:.0f}s"))
    if records is not None:
      for unseen_start, unseen_end in _unseen_episodes(records, cfg):
        start, end = max(r.start, unseen_start), min(r.end, unseen_end)
        if end - start >= cfg.alert_lost_sec:
          alerts.append(Alert(start + cfg.alert_lost_sec, "ALERT",
                              "A3_lost_outside_bed",
                              f"Out of bed and not visible for {end - start:.0f}s"))

  # event-level decisions
  for ev in events:
    # A normal exit for meals/washroom and a safe return are both NORMAL.
    # Confidence remains in the event JSON for review but is not a risk signal.
    level = "NORMAL"
    out_run = ev.extra.get("_out_run")
    if out_run:
      for a in alerts:
        if out_run[0] <= a.time <= out_run[1] + 1e-6:
          level = _max(level, a.level)
    ev.decision = level

  alerts.sort(key=lambda a: a.time)
  overall = "NORMAL"
  for a in alerts:
    overall = _max(overall, a.level)
  for ev in events:
    overall = _max(overall, ev.decision)
  return alerts, overall


def apply_vlm_safety_escalations(vlm_result: dict, alerts: List[Alert],
                                 overall: str, cfg: Config,
                                 records: Optional[List[TickRecord]] = None):
  """Allow accepted VLM evidence to escalate safety, never to downgrade it.

  A short clip cannot satisfy the configured CV-only floor duration.
  However, when the selective visual review independently confirms that the
  person is lying outside the bed with very high confidence, waiting another
  several seconds would delay a possible-fall alert. This function changes only the
  risk decision; raw activities, timestamps and event counts remain untouched.
  """
  escalations = []
  for review in vlm_result.get("reviews", []):
    answer = review.get("vlm_result", {})
    clearly_floor = (
        review.get("accepted") is True and
        answer.get("resolved_activity") == S.LYING_ON_FLOOR.value and
        answer.get("person_location") == "OUTSIDE_BED" and
        float(answer.get("confidence", 0.0)) >= cfg.vlm_floor_alert_conf and
        not answer.get("human_review_required", False))
    if not clearly_floor:
      continue
    window = review.get("window", {})
    start = float(window.get("start_sec", 0.0))
    end = float(window.get("end_sec", start))
    if records is not None:
      floor_ticks = [rec.t for rec in records
                     if start <= rec.t < end and rec.observed and
                     rec.state == S.LYING_ON_FLOOR.value]
      if floor_ticks:
        start = min(floor_ticks)
    confidence = float(answer["confidence"])
    # Avoid two alerts for the same episode if the longer CV rule also fired.
    duplicate = any(a.rule.startswith("A1_") and abs(a.time - start) <= 30.0
                    for a in alerts)
    if not duplicate:
      alert = Alert(start, "ALERT", "A1_vlm_floor_confirmed",
                    f"VLM confirmed lying on the floor "
                    f"(confidence={confidence:.2f}; possible fall)")
      alerts.append(alert)
      escalations.append(alert.to_dict())
    overall = _max(overall, "ALERT")
  alerts.sort(key=lambda a: a.time)
  return alerts, overall, escalations
