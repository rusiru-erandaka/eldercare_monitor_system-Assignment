"""Synthetic TickRecord streams to test the temporal layer without any model."""
import random
from typing import List, Tuple

from eldercare.config import Config
from eldercare.schema import TickRecord


def make_stream(script: List[Tuple[float, str]], cfg: Config, noise: float = 0.0,
                seed: int = 0, observed_script=None) -> List[TickRecord]:
  """script: [(seconds, STATE)]. A state of 'LOST' means person not observed."""
  rnd = random.Random(seed)
  dt = 1.0 / cfg.analysis_fps
  recs, t = [], 0.0
  all_states = ["LYING_IN_BED", "SITTING_ON_BED", "STANDING", "WALKING",
                "SITTING_OUTSIDE_BED", "OUT_OF_BED", "UNKNOWN"]
  for dur, st in script:
    n = int(round(dur / dt))
    for _ in range(n):
      if st == "LOST":
        recs.append(TickRecord(t=t, observed=False, bed_known=True))
      else:
        s = st
        if noise and rnd.random() < noise:
          s = rnd.choice(all_states)
        recs.append(TickRecord(t=t, state=s, confidence=0.8, observed=True,
                               bed_known=True))
      t += dt
  return recs
