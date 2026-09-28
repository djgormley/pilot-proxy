"""Marker-bin exchangeability with the bulk: the OS-CFAR closed form, measured.

On quiet frames a null bin tested against the rank ``rho`` of the bulk
exceeds it at the combinatorial rate ``(|B| + 1 - rho) / (|B| + 1)`` when the
bins are exchangeable. The bin under test must lie outside the bulk: within
one frame exactly ``|B| - rho`` of the bulk's own bins exceed the rank of the
others whatever their distribution, so a leave-one-out test says nothing. The
bins tested are the marker bins (the designated window ``D`` of the measured
anchor), which on coarse-quiet frames should carry no marker and are exactly
the set the decision ``max_D T > eta T_(rho)`` reads: the per-bin rate checks
that quiet frames are quiet and that a marker bin is exchangeable with the
bulk, and the rate of the maximum over ``D`` is the fine rule's own
exceedance rate at ``eta = 1`` on those frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Exchangeability:
    rho: int
    bulk_size: int
    frames: int
    test_bins: tuple[int, ...]
    trials: int
    measured_rate: float               # per-bin exceedance of the designated bins against T_(rho) of the bulk
    predicted_rate: float              # (|B| + 1 - rho) / (|B| + 1)
    max_over_test_rate: float          # fraction of frames with max_D T > T_(rho): the fine rule at eta = 1

    def as_dict(self) -> dict:
        return {"exch_rho": self.rho, "exch_bulk": self.bulk_size, "exch_frames": self.frames,
                "exch_test_bins": ";".join(str(b) for b in self.test_bins), "exch_trials": self.trials,
                "exch_measured": self.measured_rate, "exch_predicted": self.predicted_rate,
                "exch_max_over_designated": self.max_over_test_rate}


def exchangeability_rate(fine_t: np.ndarray, bulk_mask: np.ndarray, rho: int, test_bins) -> Exchangeability:
    """Exceedance rate of bins outside the bulk against the rank ``rho`` of the bulk.

    ``fine_t`` is ``(frames, 256)`` over quiet frames; ``test_bins`` are the
    bins under test (the designated window). A test bin exceeds when ``T`` is
    strictly larger than the ``rho``-th smallest bulk value of its frame (ties
    do not exceed, the conservative direction for a decision ``> eta T_(rho)``).
    """
    t = np.asarray(fine_t, dtype=float)
    bulk = np.flatnonzero(np.asarray(bulk_mask, dtype=bool))
    tests = tuple(int(b) for b in test_bins)
    n = bulk.size
    if t.ndim != 2 or n < 1 or not (1 <= int(rho) <= n) or not tests:
        raise ValueError("exchangeability needs (frames, bins) values, a non-empty bulk, 1 <= rho <= |B| and test bins")
    if set(tests) & set(bulk.tolist()):
        raise ValueError("test bins must lie outside the bulk")
    t_rho = np.sort(t[:, bulk], axis=1)[:, int(rho) - 1]             # (frames,)
    tested = t[:, tests]                                              # (frames, len(tests))
    exceed = tested > t_rho[:, None]
    return Exchangeability(int(rho), int(n), int(t.shape[0]), tests, int(exceed.size),
                           float(exceed.mean()) if exceed.size else math.nan, (n + 1 - int(rho)) / (n + 1),
                           float(exceed.any(axis=1).mean()) if exceed.size else math.nan)


__all__ = ["Exchangeability", "exchangeability_rate"]
