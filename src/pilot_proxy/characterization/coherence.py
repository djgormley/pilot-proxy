"""Coherence gain of a residual read over a finite number of frames.

A residual that stays coherent for a time ``tau`` integrates up by the block
count ``G = min(tau, cap) / T_frame`` (the zero-frequency gain). A statistic
formed only from products of distinct frames cannot see a coherence shorter
than the dump, so the least ``G`` that any ``tau`` allows for an ``n``-frame
reading (the admissible minimum) is the lower of two autocorrelation models,
exponential and block, over the whole range of ``tau``: 8.41 for 11 frames,
21.06 for 33 and 4.06 for 4.
"""
from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=None)
def gmin_models(n):
    """Lowest power-form R(tau)/R(G = 1) over tau for a cross-frame amplitude over n frames, per autocorrelation model.
    The estimator uses only products of distinct frames, so a residual with autocorrelation rho(d) responds as
    resp = mean_{i != j} rho(|i - j|); the true power is A^2 / resp and G = max(tau / T_frame, 1)."""
    d = np.arange(1, n); wts = 2 * (n - d) / (n * (n - 1)); out = {}
    for model in ("exp", "block"):
        best = np.inf
        for tf in np.logspace(-2, 4, 6001):
            rho = np.exp(-d / tf) if model == "exp" else np.clip(1 - d / tf, 0, None)
            resp = float(np.sum(wts * rho))
            if resp <= 0: continue
            best = min(best, max(tf, 1.0) / resp)
        out[model] = best
    return out


def admissible_min_G(n):
    """The lower of the two models (item 3): 8.41 for 11 frames, 21.06 for 33, 4.06 for 4."""
    return min(gmin_models(n).values())


def zero_frequency_gain(tau, cap, frame_seconds):
    """The zero-frequency block count min(tau, cap) / T_frame."""
    return min(tau, cap) / frame_seconds


__all__ = ["admissible_min_G", "gmin_models", "zero_frequency_gain"]
