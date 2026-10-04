"""Convert noisy per-tick classifications into stable activity segments.

Three mechanisms (all explainable on a whiteboard):
  1. Occlusion policy   - person not observed: if they were in bed we assume a blanket
                          hides them (hold state); outside bed we hold only a few seconds
                          then say UNKNOWN (left camera view).
  2. Sliding vote       - confidence-weighted majority over ~2 s kills 1-tick flicker.
  3. Dwell hysteresis   - a new state must persist for its minimum dwell time before it is
                          committed; the segment is then BACK-DATED to when the new state
                          first appeared. Direct jumps that make no physical sense
                          (lying -> walking) need twice the dwell.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Dict, List, Optional, Tuple

from .config import Config
from .schema import ActivityState as S, TickRecord

IMPLAUSIBLE = {
    frozenset({S.LYING_IN_BED, S.WALKING}),
    frozenset({S.LYING_IN_BED, S.STANDING}),
    frozenset({S.LYING_IN_BED, S.SITTING_OUTSIDE_BED}),
}


@dataclass
class Segment:
  state: S
  start: float
  end: float
  conf_sum: float = 0.0
  n: int = 0
  note: str = ""
  track_id: Optional[int] = None

  @property
  def duration(self) -> float:
    return max(0.0, self.end - self.start)

  @property
  def confidence(self) -> float:
    return self.conf_sum / self.n if self.n else 0.0

  def to_dict(self) -> dict:
    return {"state": self.state.value, "start": round(self.start, 2),
            "end": round(self.end, 2), "confidence": round(self.confidence, 3),
            "note": self.note, "track_id": self.track_id}


def tick_period(records: List[TickRecord], cfg: Config) -> float:
  """Return the real sampling period represented by a list of video ticks.

  Video frame rates are commonly 24, 25, or 29.97 FPS, so asking for 2 analysis
  FPS does not always produce timestamps exactly 0.5 seconds apart.  Use the
  timestamps written by the perception pass and only fall back to the configured
  rate when a stream contains a single tick.
  """
  gaps = [b.t - a.t for a, b in zip(records, records[1:]) if b.t > a.t]
  return median(gaps) if gaps else 1.0 / cfg.analysis_fps


def segments_from_ticks(records: List[TickRecord], cfg: Config) -> List[Segment]:
  """Build the factual activity timeline directly from model output ticks.

  This function performs no voting, dwell filtering, back-dating, or
  occlusion-state holding. A visible tick keeps exactly the state shown in the
  preview; an unobserved/invalid tick is UNKNOWN.  Only adjacent equal outputs
  for the same logical person are merged for readability.
  """
  if not records:
    return []
  records = sorted(records, key=lambda r: r.t)
  dt_last = tick_period(records, cfg)
  out: List[Segment] = []
  for i, rec in enumerate(records):
    end = records[i + 1].t if i + 1 < len(records) else rec.t + dt_last
    if end <= rec.t:  # defensive handling for duplicate/malformed timestamps
      end = rec.t + dt_last
    try:
      state = S(rec.state) if rec.observed else S.UNKNOWN
    except ValueError:
      state = S.UNKNOWN
    track_id = rec.track_id if rec.observed else None
    note = "raw_model_output" if rec.observed else "elder_not_observed"
    conf = max(0.0, rec.confidence) if rec.observed else 0.0
    contiguous = bool(
        out and out[-1].state == state and out[-1].track_id == track_id and
        abs(out[-1].end - rec.t) <= 1e-6)
    if contiguous:
      out[-1].end = end
      out[-1].conf_sum += conf
      out[-1].n += 1
    else:
      out.append(Segment(state, rec.t, end, conf, 1, note, track_id))
  return out


class TemporalStateMachine:

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.dt = 1.0 / cfg.analysis_fps
    self.segments: List[Segment] = []
    self.state: Optional[S] = None
    self.t0: Optional[float] = None
    self._lost_since: Optional[float] = None
    self._recent: Deque[Tuple[float, S, float]] = deque()
    self._raw: Deque[Tuple[float, S]] = deque()
    self._cand: Optional[S] = None
    self._cand_since = 0.0

  # ---- helpers ---------------------------------------------------------------
  def _effective(self, rec: TickRecord) -> Tuple[S, float]:
    cfg = self.cfg
    if rec.observed:
      self._lost_since = None
      st = S(rec.state)
      if st == S.UNKNOWN or rec.confidence < cfg.min_conf:
        return S.UNKNOWN, max(rec.confidence, 0.1)
      return st, rec.confidence
    if self._lost_since is None:
      self._lost_since = rec.t
    lost = rec.t - self._lost_since
    # Long blanket occlusion is plausible while lying. A sitting label must not be
    # held for minutes after the body disappears: they may have lain down or left.
    if self.state == S.LYING_IN_BED and lost <= cfg.hold_in_bed_sec:
      return self.state, max(0.40, 0.60 - 0.20 * lost / cfg.hold_in_bed_sec)
    if self.state == S.SITTING_ON_BED and lost <= cfg.hold_sitting_in_bed_sec:
      return self.state, 0.40
    if self.state not in (None, S.UNKNOWN, S.SITTING_ON_BED) and lost <= cfg.lost_grace_sec:
      return self.state, 0.40
    return S.UNKNOWN, 0.30

  def _vote(self) -> S:
    score: Dict[S, float] = defaultdict(float)
    for _, st, c in self._recent:
      score[st] += c
    return max(score, key=score.get)

  def _backdate(self, target: S, floor: float) -> float:
    """Earliest time of the most recent run of `target`, tolerating single-tick noise."""
    start, miss = self._raw[-1][0], 0
    for t, st in reversed(self._raw):
      if t < floor:
        break
      if st == target:
        start, miss = t, 0
      else:
        miss += 1
        if miss >= 2:
          break
    return max(start, floor)

  def _need(self, new: S) -> float:
    need = self.cfg.min_dwell_sec.get(new.value, 2.0)
    if self.state is not None and frozenset({self.state, new}) in IMPLAUSIBLE:
      need *= self.cfg.implausible_dwell_mult
    return need

  # ---- main --------------------------------------------------------------------
  def update(self, rec: TickRecord):
    cfg = self.cfg
    if self.t0 is None:
      self.t0 = rec.t
    eff, conf = self._effective(rec)
    self._raw.append((rec.t, eff))
    while self._raw and rec.t - self._raw[0][0] > 180.0:
      self._raw.popleft()
    self._recent.append((rec.t, eff, conf))
    while self._recent and rec.t - self._recent[0][0] > cfg.smooth_window_sec:
      self._recent.popleft()
    smoothed = self._vote()

    if self.state is None:  # bootstrap the very first segment
      if rec.t - self.t0 + self.dt >= cfg.min_dwell_sec.get(smoothed.value, 2.0):
        self.state = smoothed
        self.segments.append(Segment(smoothed, self.t0, rec.t, conf, 1,
                                     track_id=rec.track_id))
      return

    cur = self.segments[-1]
    cur.end = rec.t + self.dt
    if smoothed == self.state:
      self._cand = None
      if cur.track_id is None and rec.track_id is not None:
        cur.track_id = rec.track_id
      cur.conf_sum += conf
      cur.n += 1
      return

    if self._cand != smoothed:
      self._cand, self._cand_since = smoothed, rec.t
    if rec.t - self._cand_since + self.dt >= self._need(smoothed):
      start = self._backdate(smoothed, cur.start)
      cur.end = start
      note = "implausible_direct" if frozenset({self.state, smoothed}) in IMPLAUSIBLE else ""
      self.segments.append(Segment(smoothed, start, rec.t + self.dt, conf, 1, note,
                                   rec.track_id))
      self.state, self._cand = smoothed, None
    else:
      cur.conf_sum += conf
      cur.n += 1

  def finalize(self, t_end: float):
    if not self.segments:
      if self.t0 is not None:
        self.segments.append(Segment(S.UNKNOWN, self.t0, t_end, 0.1, 1, "no_stable_state"))
      return
    self.segments[-1].end = max(t_end, self.segments[-1].start)
    # drop zero-length artefacts created by back-dating
    self.segments = [s for s in self.segments if s.duration > 1e-6] or self.segments
