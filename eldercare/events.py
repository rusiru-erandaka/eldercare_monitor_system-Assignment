"""Identity-aware, ordered bed-event reasoning on committed segments.

BED_EXIT normally follows the main activity path:
  LYING_IN_BED -> SITTING_ON_BED -> STANDING -> WALKING outside the bed

UNKNOWN segments are evidence gaps and are ignored by the sequence matcher, even when several
occur between main states. One sitting/standing step may be missed when same-track spatial and
pose evidence proves the elder became upright and crossed the bed boundary while walking.
Location alone can never create a bed exit. A visitor who starts by sitting on the bed therefore
cannot inherit the elder's earlier lying state and create an event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config
from .schema import (ActivityState as S, IN_BED_STATES, OUT_OF_BED_STATES,
                     TickRecord, fmt_hhmmss)
from .temporal import Segment, segments_from_ticks


@dataclass
class Run:
  loc: str                      # "bed" | "out" | "unknown"
  start: float
  end: float
  segs: List[Segment] = field(default_factory=list)

  @property
  def duration(self) -> float:
    return self.end - self.start


@dataclass
class Event:
  event: str                    # "bed_exit" | "bed_return"
  start_time: float
  confirmed_time: float
  previous_state: str
  current_state: str
  confidence: float
  decision: str = "NORMAL"
  extra: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    payload = {
        "event": self.event,
        "start_time": fmt_hhmmss(self.start_time),
        "confirmed_time": fmt_hhmmss(self.confirmed_time),
        "previous_state": self.previous_state.lower(),
        "current_state": self.current_state.lower(),
        "confidence": round(self.confidence, 2),
        "decision": self.decision,
    }
    payload.update({key: value for key, value in self.extra.items()
                    if not key.startswith("_")})
    return payload


def _loc(state: S) -> str:
  if state in IN_BED_STATES:
    return "bed"
  if state in OUT_OF_BED_STATES:
    return "out"
  return "unknown"


def build_runs(segments: List[Segment], cfg: Config) -> List[Run]:
  runs: List[Run] = []
  for s in segments:
    loc = _loc(s.state)
    if runs and runs[-1].loc == loc:
      runs[-1].end = s.end
      runs[-1].segs.append(s)
    else:
      runs.append(Run(loc, s.start, s.end, [s]))

  changed = True
  while changed:
    changed = False
    for i, r in enumerate(runs):
      if r.loc != "unknown":
        continue
      prev = runs[i - 1] if i > 0 else None
      nxt = runs[i + 1] if i + 1 < len(runs) else None
      target = None
      if prev and nxt and prev.loc == nxt.loc and prev.loc != "unknown" \
          and r.duration < cfg.unknown_gap_sec:
        target = prev.loc
      elif prev and prev.loc == "out" and (nxt is None or nxt.loc == "out"):
        target = "out"   # out of bed and out of view: still away from the bed
      if target is None:
        continue
      r.loc = target
      merged: List[Run] = []
      for x in runs:
        if merged and merged[-1].loc == x.loc:
          merged[-1].end = x.end
          merged[-1].segs.extend(x.segs)
        else:
          merged.append(x)
      runs[:] = merged
      changed = True
      break
  return runs


def _mean_conf(segs: List[Segment]) -> float:
  tot = sum(s.duration for s in segs)
  if tot <= 0:
    return 0.0
  return sum(s.confidence * s.duration for s in segs) / tot


@dataclass
class ExitPath:
  lying: Segment
  sitting: Optional[Segment]
  standing: Optional[Segment]
  walking: Segment
  actor: Optional[int]
  run_index: Optional[int]
  confirmed_time: float
  spatial: dict = field(default_factory=dict)


@dataclass
class ReturnPath:
  outside: Segment
  walking: Optional[Segment]
  standing: Optional[Segment]
  sitting: Optional[Segment]
  lying: Segment
  actor: Optional[int]
  run_index: Optional[int]
  confirmed_time: float
  spatial: dict = field(default_factory=dict)


def _actor_matches(actor: Optional[int], seg: Segment) -> bool:
  """Unknown identity is tolerated; two known, different tracks never mix."""
  return actor is None or seg.track_id is None or actor == seg.track_id


def _departure_run(runs: List[Run], walking_t: float, cfg: Config) -> Optional[int]:
  """First sustained out run after walking, allowing intervening UNKNOWN evidence."""
  for i, r in enumerate(runs):
    if r.end <= walking_t:
      continue
    if r.loc == "bed" and r.start > walking_t:
      return None  # returned before departure could be confirmed
    if r.loc == "out" and r.end - walking_t >= cfg.exit_min_out_sec:
      return i
  return None


def _center_outside_bed(rec: TickRecord) -> Optional[bool]:
  if not rec.bbox or not rec.bed:
    return None
  cx = (rec.bbox[0] + rec.bbox[2]) / 2.0
  cy = (rec.bbox[1] + rec.bbox[3]) / 2.0
  return not (rec.bed[0] <= cx <= rec.bed[2] and
              rec.bed[1] <= cy <= rec.bed[3])


def _center_inside_bed(rec: TickRecord) -> Optional[bool]:
  outside = _center_outside_bed(rec)
  return None if outside is None else not outside


def _spatial_exit_evidence(records: List[TickRecord], lying: Segment,
                           sitting: Optional[Segment], standing: Optional[Segment],
                           walking: Segment, actor: Optional[int], cfg: Config) -> dict:
  """Check whether the elder physically moves from the bed into walking outside it.

  Same-track geometry can recover a standing step hidden by camera clipping while
  preventing a visitor or a one-frame walking error from creating an exit event.
  """
  anchor = sitting or standing or lying
  relevant = [r for r in records
              if anchor.start - 1e-6 <= r.t <= walking.end + 1e-6 and
              (actor is None or r.track_id is None or r.track_id == actor)]
  overlaps = [(r.t, r.bed_overlap) for r in relevant if r.bed_overlap is not None]
  walking_records = [r for r in relevant
                     if walking.start - 1e-6 <= r.t <= walking.end + 1e-6 and
                     r.observed]
  final = walking_records[-1] if walking_records else None
  start_overlap = max((v for _, v in overlaps), default=None)
  final_overlap = final.bed_overlap if final is not None else None
  overlap_drop = (start_overlap - final_overlap
                  if start_overlap is not None and final_overlap is not None else None)

  changes = [b - a for (_, a), (_, b) in zip(overlaps, overlaps[1:])]
  outward = sum(1 for d in changes if d < -0.02)
  inward = sum(1 for d in changes if d > 0.02)
  outward_ratio = outward / max(1, outward + inward)

  infer_after = sitting.end if sitting is not None else lying.end
  upright_unknown = [r for r in relevant
                     if r.t >= infer_after - 1e-6 and r.observed and
                     r.state == S.UNKNOWN.value and r.torso_tilt is not None and
                     r.torso_tilt <= cfg.upright_tilt_deg and
                     ((r.knee_angle is not None and
                       r.knee_angle >= cfg.exit_inferred_standing_knee_deg) or
                      r.body_clipped)]
  inferred_standing_time = upright_unknown[0].t if upright_unknown else None
  center_outside = _center_outside_bed(final) if final is not None else None
  moving = bool(final is not None and
                (final.state == S.WALKING.value or
                 (final.speed_bh_s is not None and
                  final.speed_bh_s >= cfg.walk_speed_bh_s)))
  crossed = bool(
      start_overlap is not None and start_overlap >= cfg.bed_inside_overlap and
      final_overlap is not None and final_overlap <= cfg.exit_cross_overlap_max and
      center_outside is True and overlap_drop is not None and
      overlap_drop >= cfg.exit_overlap_drop_min and
      outward_ratio >= cfg.exit_outward_ratio_min and moving)
  # If STANDING was not labelled, require upright skeletal evidence in the
  # intervening UNKNOWN frames before treating it as a missed step.
  confirmed = crossed and (standing is not None or inferred_standing_time is not None)
  return {
      "confirmed": confirmed,
      "start_bed_overlap": None if start_overlap is None else round(start_overlap, 3),
      "final_bed_overlap": None if final_overlap is None else round(final_overlap, 3),
      "bed_overlap_drop": None if overlap_drop is None else round(overlap_drop, 3),
      "outward_ratio": round(outward_ratio, 2),
      "center_outside_bed": center_outside,
      "inferred_standing_time": inferred_standing_time,
      "walking_observed": moving,
  }


def _spatial_return_evidence(records: List[TickRecord], outside: Segment,
                             lying: Segment, actor: Optional[int],
                             cfg: Config) -> dict:
  """Prove that the same elder moved from outside the bed to lying inside it."""
  relevant = [r for r in records
              if outside.start - 1e-6 <= r.t <= lying.end + 1e-6 and
              (actor is None or r.track_id is None or r.track_id == actor)]
  overlaps = [(r.t, r.bed_overlap) for r in relevant if r.bed_overlap is not None]
  outside_records = [r for r in relevant if r.observed and r.bed_overlap is not None]
  lying_records = [r for r in relevant
                   if lying.start - 1e-6 <= r.t <= lying.end + 1e-6 and r.observed and
                   r.state == S.LYING_IN_BED.value]
  # Use the clearest outside observation in the approach. A person can briefly
  # oscillate at the bed edge before turning and entering the bed.
  first = min(outside_records, key=lambda r: r.bed_overlap) if outside_records else None
  final = lying_records[-1] if lying_records else None
  start_overlap = min((v for _, v in overlaps), default=None)
  final_overlap = final.bed_overlap if final is not None else None
  overlap_rise = (final_overlap - start_overlap
                  if start_overlap is not None and final_overlap is not None else None)

  changes = [b - a for (_, a), (_, b) in zip(overlaps, overlaps[1:])]
  inward = sum(1 for d in changes if d > 0.02)
  outward = sum(1 for d in changes if d < -0.02)
  inward_ratio = inward / max(1, inward + outward)
  center_started_outside = _center_outside_bed(first) if first is not None else None
  center_finished_inside = _center_inside_bed(final) if final is not None else None
  confirmed = bool(
      first is not None and final is not None and
      start_overlap is not None and start_overlap <= cfg.return_start_overlap_max and
      final_overlap is not None and final_overlap >= cfg.return_cross_overlap_min and
      overlap_rise is not None and overlap_rise >= cfg.return_overlap_rise_min and
      inward_ratio >= cfg.return_inward_ratio_min and
      center_started_outside is True and center_finished_inside is True)
  return {
      "confirmed": confirmed,
      "start_bed_overlap": None if start_overlap is None else round(start_overlap, 3),
      "final_bed_overlap": None if final_overlap is None else round(final_overlap, 3),
      "bed_overlap_rise": None if overlap_rise is None else round(overlap_rise, 3),
      "inward_ratio": round(inward_ratio, 2),
      "center_started_outside_bed": center_started_outside,
      "center_finished_inside_bed": center_finished_inside,
      "lying_in_bed_observed": final is not None,
  }


def evidence_segments(records: List[TickRecord], cfg: Config) -> List[Segment]:
  """Keep confident raw transition evidence that timeline smoothing may suppress.

  UNKNOWN and weak observations are omitted. Consecutive equal states are grouped,
  but an evidence gap starts a new group so its duration is never fabricated.
  """
  return [s for s in segments_from_ticks(records, cfg)
          if s.state != S.UNKNOWN and s.confidence >= cfg.min_conf]


def find_exit_paths(segments: List[Segment], runs: List[Run], cfg: Config,
                    records: Optional[List[TickRecord]] = None) -> List[ExitPath]:
  """Match ordered elder paths, allowing one visually supported missed posture."""
  candidate: Optional[Dict[str, object]] = None
  paths: List[ExitPath] = []

  for seg in segments:
    if seg.state == S.UNKNOWN:
      continue

    # A fresh lying observation is the only way to arm a new exit candidate.
    if seg.state == S.LYING_IN_BED:
      candidate = {"lying": seg, "sitting": None, "standing": None,
                   "actor": seg.track_id}
      continue
    if candidate is None:
      continue

    lying = candidate["lying"]
    if seg.start - lying.start > cfg.exit_sequence_max_sec:
      candidate = None
      continue
    actor = candidate["actor"]
    if not _actor_matches(actor, seg):
      # This is another tracked person (for example a visitor); it cannot
      # advance or cancel the elder's sequence.
      continue
    if actor is None and seg.track_id is not None:
      actor = seg.track_id
      candidate["actor"] = actor

    if seg.state == S.SITTING_ON_BED:
      # Sitting again after standing means the earlier stand did not complete
      # the departure; wait for a new stand or infer it from spatial evidence.
      candidate.update(sitting=seg, standing=None)
      continue
    if seg.state == S.STANDING:
      candidate["standing"] = seg
      continue
    if seg.state == S.WALKING:
      sitting = candidate.get("sitting")
      standing = candidate.get("standing")
      accepted = False
      # At least one intermediate posture must be directly observed. The other
      # may be missed only when the geometry proves an actual bed-boundary exit.
      if sitting is not None or standing is not None:
        run_i = _departure_run(runs, seg.start, cfg)
        sustained = run_i is not None
        spatial = (_spatial_exit_evidence(records, lying, sitting, standing,
                                           seg, actor, cfg)
                   if records is not None else {"confirmed": False})
        complete_path = sitting is not None and standing is not None
        if (complete_path and sustained) or spatial.get("confirmed", False):
          confirmed = (seg.start if spatial.get("confirmed", False)
                       else seg.start + cfg.exit_min_out_sec)
          paths.append(ExitPath(lying, sitting, standing, seg, actor, run_i,
                                confirmed, spatial))
          accepted = True
      # Motion estimation can briefly call the sit-to-stand displacement
      # WALKING while the person is still over the bed. Keep the candidate for
      # a later real crossing unless this unsupported label persists.
      if accepted or seg.duration > cfg.exit_transient_tolerance_sec:
        candidate = None
      continue
    if seg.duration > cfg.exit_transient_tolerance_sec:
      candidate = None
  return paths


def _return_run(runs: List[Run], lying_t: float, cfg: Config) -> Optional[int]:
  for i, run in enumerate(runs):
    if run.loc == "bed" and run.start <= lying_t < run.end and \
        run.end - lying_t >= cfg.return_min_in_sec:
      return i
  return None


def find_return_paths(segments: List[Segment], runs: List[Run], cfg: Config,
                      records: Optional[List[TickRecord]] = None) -> List[ReturnPath]:
  """Find OUT -> WALK -> STAND -> SIT -> LIE, tolerating two missing middle steps.

  UNKNOWN evidence never advances or cancels the path. A return may be found at
  the beginning of a standalone return video; it does not depend on a previous
  BED_EXIT event in the same recording.
  """
  candidate: Optional[Dict[str, object]] = None
  paths: List[ReturnPath] = []
  outside_states = {S.OUT_OF_BED, S.WALKING, S.STANDING,
                    S.SITTING_OUTSIDE_BED, S.LYING_ON_FLOOR}

  for seg in segments:
    if seg.state == S.UNKNOWN:
      continue

    if candidate is None:
      if seg.state not in outside_states:
        continue
      candidate = {"outside": seg, "walking": None, "standing": None,
                   "sitting": None, "actor": seg.track_id}
      if seg.state == S.WALKING:
        candidate["walking"] = seg
      elif seg.state == S.STANDING:
        candidate["standing"] = seg
      continue

    outside = candidate["outside"]
    if seg.start - outside.start > cfg.exit_sequence_max_sec:
      candidate = None
      if seg.state in outside_states:
        candidate = {"outside": seg, "walking": seg if seg.state == S.WALKING else None,
                     "standing": seg if seg.state == S.STANDING else None,
                     "sitting": None, "actor": seg.track_id}
      continue
    actor = candidate["actor"]
    if not _actor_matches(actor, seg):
      continue
    if actor is None and seg.track_id is not None:
      actor = seg.track_id
      candidate["actor"] = actor

    if seg.state == S.WALKING:
      # Walking after sitting means the attempted return was abandoned; begin a
      # fresh approach and wait for a new stand/sit/lie sequence.
      candidate["walking"] = seg
      candidate["standing"] = None
      candidate["sitting"] = None
      continue
    if seg.state == S.STANDING:
      candidate["standing"] = seg
      candidate["sitting"] = None
      continue
    if seg.state == S.SITTING_ON_BED:
      candidate["sitting"] = seg
      continue
    if seg.state == S.LYING_IN_BED:
      walking = candidate.get("walking")
      standing = candidate.get("standing")
      sitting = candidate.get("sitting")
      observed_middle = sum(x is not None for x in (walking, standing, sitting))
      if observed_middle >= 1:  # at most two of the three main middle steps may be absent
        spatial = (_spatial_return_evidence(records, outside, seg, actor, cfg)
                   if records is not None else {"confirmed": False})
        run_i = _return_run(runs, seg.start, cfg)
        # Standalone returns from real video require geometry. A separate legacy
        # fallback is applied later only when a confirmed exit exists first.
        if spatial.get("confirmed", False):
          paths.append(ReturnPath(outside, walking, standing, sitting, seg,
                                  actor, run_i, seg.start, spatial))
      candidate = None
      continue

    # A long unrelated in-bed/outside posture cancels this approach, while short
    # classifier flicker is tolerated just like it is for bed exits.
    if seg.duration > cfg.exit_transient_tolerance_sec and seg.state not in outside_states:
      candidate = None
  return paths


def detect_events(segments: List[Segment], cfg: Config,
                  records: Optional[List[TickRecord]] = None):
  """Returns (events, runs)."""
  runs = build_runs(segments, cfg)
  evidence = evidence_segments(records, cfg) if records is not None else segments
  exit_paths = find_exit_paths(evidence, runs, cfg, records)
  return_paths = find_return_paths(evidence, runs, cfg, records)
  events: List[Event] = []
  for path in exit_paths:
    observed = [s for s in (path.lying, path.sitting, path.standing, path.walking)
                if s is not None]
    missing = int(path.sitting is None) + int(path.standing is None)
    conf = sum(s.confidence for s in observed) / len(observed)
    conf *= 0.92 ** missing
    standing_t = (path.standing.start if path.standing is not None
                  else path.spatial.get("inferred_standing_time", path.walking.start))
    sequence = [path.lying.state.value.lower()]
    sequence.append("sitting_on_bed" if path.sitting is not None else "sitting_inferred")
    sequence.append("standing" if path.standing is not None else "standing_inferred")
    sequence.extend(["walking", "out_of_bed"])
    times = {s.state.value.lower(): fmt_hhmmss(s.start) for s in observed}
    if path.standing is None and standing_t is not None:
      times["standing_inferred"] = fmt_hhmmss(standing_t)
    extra = {
        "sequence": sequence,
        "sequence_times": times,
        "track_id": path.actor,
        "confirmation": "spatial_bed_crossing" if path.spatial.get("confirmed")
                        else "sustained_out_of_bed",
        "spatial_evidence": {k: v for k, v in path.spatial.items() if k != "confirmed"},
    }
    if path.run_index is not None:
      r = runs[path.run_index]
      extra["_out_run"] = (r.start, r.end)
    previous = path.sitting.state.value if path.sitting is not None else path.lying.state.value
    events.append(Event("bed_exit", standing_t or path.walking.start,
                        path.confirmed_time, previous, S.OUT_OF_BED.value,
                        conf, extra=extra))

  for path in return_paths:
    middle = [("walking", path.walking), ("standing", path.standing),
              ("sitting_on_bed", path.sitting)]
    observed = [path.outside] + [s for _, s in middle if s is not None] + [path.lying]
    missing = [name for name, seg in middle if seg is None]
    conf = sum(s.confidence for s in observed) / len(observed)
    conf *= 0.92 ** len(missing)
    sequence = ["out_of_bed"] + [name if seg is not None else f"{name}_not_observed"
                                  for name, seg in middle] + ["lying_in_bed"]
    times = {name: fmt_hhmmss(seg.start) for name, seg in middle if seg is not None}
    times["out_of_bed"] = fmt_hhmmss(path.outside.start)
    times["lying_in_bed"] = fmt_hhmmss(path.lying.start)
    extra = {
        "sequence": sequence,
        "sequence_times": times,
        "missing_steps": missing,
        "track_id": path.actor,
        "confirmation": "spatial_bed_entry" if path.spatial.get("confirmed")
                        else "sustained_in_bed",
        "spatial_evidence": {k: v for k, v in path.spatial.items() if k != "confirmed"},
    }
    events.append(Event("bed_return",
                        path.walking.start if path.walking is not None else path.outside.start,
                        path.confirmed_time, S.OUT_OF_BED.value,
                        S.LYING_IN_BED.value, conf, extra=extra))

  # Geometry-free timelines (unit tests, manually labelled GT, or old cached
  # ticks) cannot prove a standalone return. Preserve a conservative fallback
  # only after an exit was already confirmed in the same stream.
  geometry_available = bool(records is not None and any(
      r.bbox is not None and r.bed is not None for r in records))
  if not geometry_available:
    for i, path in enumerate(exit_paths):
      next_exit = exit_paths[i + 1].confirmed_time if i + 1 < len(exit_paths) else float("inf")
      already = any(e.event == "bed_return" and
                    path.confirmed_time < e.confirmed_time < next_exit for e in events)
      if already:
        continue
      for run_i, run in enumerate(runs):
        if not (path.confirmed_time < run.start < next_exit):
          continue
        if run.loc != "bed" or run.duration < cfg.return_min_in_sec:
          continue
        prev = next((x for x in reversed(runs[:run_i]) if x.loc == "out"), None)
        out_states = [s for s in (prev.segs if prev else []) if s.state != S.UNKNOWN]
        prev_state = out_states[-1].state.value if out_states else S.OUT_OF_BED.value
        first = next((s for s in run.segs if s.state != S.UNKNOWN), run.segs[0])
        conf = _mean_conf(run.segs) * min(1.0, run.duration / (2 * cfg.return_min_in_sec))
        events.append(Event(
            "bed_return", run.start, run.start + cfg.return_min_in_sec,
            prev_state, first.state.value, conf,
            extra={"confirmation": "sustained_in_bed_after_confirmed_exit",
                   "missing_steps": [], "track_id": first.track_id}))
        break
  events.sort(key=lambda e: e.confirmed_time)
  return events, runs
