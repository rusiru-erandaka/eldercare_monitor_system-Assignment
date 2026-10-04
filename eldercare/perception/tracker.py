"""Track people and maintain the identity of the monitored elder.

Matching combines bounding-box overlap, centre distance and an HSV appearance
signature. Time spent near the bed provides an additional identity cue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..config import Config
from ..schema import BoundingBox, PersonObs


def appearance_signature(frame: Optional[np.ndarray], box: BoundingBox) -> Optional[np.ndarray]:
  """Normalised HSV histogram of the person crop (cheap re-ID cue)."""
  if frame is None:
    return None
  h, w = frame.shape[:2]
  x1, y1 = max(0, int(box.xmin)), max(0, int(box.ymin))
  x2, y2 = min(w, int(box.xmax)), min(h, int(box.ymax))
  if x2 - x1 < 8 or y2 - y1 < 8:
    return None
  hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
  s = hist.sum()
  return hist / s if s > 0 else None


def _sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
  if a is None or b is None:
    return 0.5
  return float(np.sum(np.sqrt(a * b)))  # Bhattacharyya coefficient in [0,1]


@dataclass
class Track:
  tid: int
  bbox: BoundingBox
  obs: PersonObs
  last_seen: float
  first_seen: float
  sig: Optional[np.ndarray] = None
  bed_time: float = 0.0
  hits: int = 1
  last_tick_t: float = 0.0
  born_in_bed: bool = False


@dataclass
class TrackResult:
  elder: Optional[PersonObs]
  elder_id: Optional[int]
  caregiver_present: bool
  n_active: int


class PersonTracker:

  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.tracks: Dict[int, Track] = {}
    self._next_id = 1
    self.elder_id: Optional[int] = None

  # ---- matching ---------------------------------------------------------------
  def _affinity(self, tr: Track, ob: PersonObs, sig, gate: float, t: float,
                min_sim: float = 0.0) -> float:
    iou = tr.bbox.iou(ob.bbox)
    (cx1, cy1), (cx2, cy2) = tr.bbox.center, ob.bbox.center
    scale = max(tr.bbox.height, ob.bbox.height, 1.0)
    dist = math.hypot(cx1 - cx2, cy1 - cy2) / scale
    gate = gate + 0.5 * max(0.0, t - tr.last_seen)   # allow for motion while unseen
    if iou < 0.10 and dist > gate:
      return -1.0
    sim = _sim(tr.sig, sig)
    if min_sim > 0.0 and tr.sig is not None and sig is not None and sim < min_sim:
      return -1.0                                     # clearly a different person
    return iou + 0.6 * max(0.0, 1.0 - dist / gate) + 0.5 * sim

  def _assign(self, t: float, obs: List[PersonObs], sigs, track_ids: List[int],
              gate: float, taken_obs: set, min_sim: float = 0.0) -> Dict[int, int]:
    pairs = []
    for ti in track_ids:
      for oi, ob in enumerate(obs):
        if oi in taken_obs:
          continue
        a = self._affinity(self.tracks[ti], ob, sigs[oi], gate, t, min_sim)
        if a >= 0:
          pairs.append((a, ti, oi))
    pairs.sort(reverse=True)
    got, used_t = {}, set()
    for a, ti, oi in pairs:
      if ti in used_t or oi in taken_obs:
        continue
      got[oi] = ti
      used_t.add(ti)
      taken_obs.add(oi)
    return got

  # ---- main update ---------------------------------------------------------------
  def update(self, t: float, obs: List[PersonObs], bed: Optional[BoundingBox],
             frame: Optional[np.ndarray] = None) -> TrackResult:
    cfg = self.cfg
    sigs = [appearance_signature(frame, o.bbox) for o in obs]
    alive = [i for i, tr in self.tracks.items() if t - tr.last_seen <= cfg.track_lost_sec]
    active = [i for i in alive if t - self.tracks[i].last_seen <= cfg.track_active_sec]

    taken: set = set()
    assigned = self._assign(t, obs, sigs, active, cfg.track_gate_dist, taken)
    # stage 2: leftover observations may re-link to any still-unmatched track (active or
    # lost) over a wider gate, but only if the person *looks* the same (colour signature)
    leftovers = [i for i in alive if i not in set(assigned.values())]
    assigned.update(self._assign(t, obs, sigs, leftovers, cfg.relink_gate_dist, taken,
                                 cfg.relink_min_sim))

    seen_ids = set()
    for oi, ob in enumerate(obs):
      if oi in assigned:
        tr = self.tracks[assigned[oi]]
        tr.bbox, tr.obs, tr.last_seen = ob.bbox, ob, t
        tr.hits += 1
        if sigs[oi] is not None:  # slow EMA keeps the signature stable
          tr.sig = sigs[oi] if tr.sig is None else 0.9 * tr.sig + 0.1 * sigs[oi]
      else:
        tid = self._next_id
        self._next_id += 1
        born_in_bed = False
        if bed is not None and ob.bbox.area > 0:
          born_in_bed = ob.bbox.intersection_area(bed) / ob.bbox.area > 0.3
        tr = Track(tid, ob.bbox, ob, t, t, sigs[oi], born_in_bed=born_in_bed)
        self.tracks[tid] = tr
      seen_ids.add(tr.tid)
      # behavioural elder cue: time spent on the bed
      if bed is not None and ob.bbox.area > 0:
        ov_frac = ob.bbox.intersection_area(bed) / ob.bbox.area
        if ov_frac > 0.3:
          tr.bed_time += max(0.0, t - tr.last_tick_t) if tr.last_tick_t else 0.0
      tr.last_tick_t = t

    # forget very old tracks
    for i in [i for i, tr in self.tracks.items() if t - tr.last_seen > cfg.track_lost_sec]:
      if i != self.elder_id:
        del self.tracks[i]

    self._choose_elder(t, seen_ids)
    elder_obs = None
    if self.elder_id in seen_ids:
      elder_obs = self.tracks[self.elder_id].obs
    n_active = len(seen_ids)
    caregiver = n_active > (1 if self.elder_id in seen_ids else 0)
    return TrackResult(elder_obs, self.elder_id if elder_obs else None, caregiver, n_active)

  def _choose_elder(self, t: float, seen_ids: set):
    cfg = self.cfg
    if not self.tracks:
      return
    if self.elder_id is None or self.elder_id not in self.tracks:
      # bootstrap: most bed time, else the largest / most confident visible track
      cands = [self.tracks[i] for i in seen_ids] or list(self.tracks.values())
      self.elder_id = max(cands, key=lambda tr: (tr.bed_time, tr.bbox.area)).tid
      return
    elder = self.tracks[self.elder_id]
    unseen = t - elder.last_seen
    # During the initial fixed-camera/single-resident phase, a drastic posture
    # change can fragment a track (for example blanket/head box -> full body as
    # the elder sits up).  If there is exactly one visible person, reacquire it
    # promptly instead of returning "elder_not_observed" for many seconds.
    if cfg.single_elder_mode and len(seen_ids) == 1 and self.elder_id not in seen_ids \
        and unseen >= cfg.elder_reacquire_sec:
      new_id = next(iter(seen_ids))
      candidate = self.tracks[new_id]
      if candidate.born_in_bed:
        # This is normally the same elder fragmented by a drastic posture/box
        # change. Merge it into the stable logical elder ID so event sequences
        # cannot mix identities and legitimate transitions keep one actor.
        elder.bbox = candidate.bbox
        elder.obs = candidate.obs
        elder.last_seen = candidate.last_seen
        elder.sig = candidate.sig
        elder.bed_time = max(elder.bed_time, candidate.bed_time)
        elder.hits += candidate.hits
        elder.last_tick_t = candidate.last_tick_t
        del self.tracks[new_id]
        seen_ids.discard(new_id)
        seen_ids.add(self.elder_id)
        return
    # Switch identity only while the elder is missing (a caregiver standing at the bedside
    # must not steal it): either someone out-scores the elder's bed time by a margin, or the
    # elder has been gone a while and another visible track has been on the bed for a while
    # (their track was orphaned by a missed-detection gap).
    for tr in self.tracks.values():
      if tr.tid == self.elder_id or tr.tid not in seen_ids:
        continue
      if cfg.single_elder_mode and not tr.born_in_bed:
        continue  # visitor entered from outside; never inherit the elder identity
      if unseen > cfg.track_active_sec and \
          tr.bed_time > elder.bed_time + cfg.elder_switch_margin_sec:
        self.elder_id = tr.tid
        break
      if unseen > cfg.elder_orphan_sec and tr.bed_time >= cfg.elder_orphan_bed_sec:
        self.elder_id = tr.tid
        break
