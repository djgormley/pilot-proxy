"""The named configurations that reproduce the archive releases of the 2026 campaign.

Three releases characterize the same products; they differ in three
declared choices:

- ``archive_author_eras_2026_09_23`` (``results/archive_author_eras_2026-09-23``):
  the integration model books the variance split where the correlation time
  is usable (``booked_when_tau_usable``), and least-residual choices break
  ties by the first minimum in enumeration order (:data:`FIRST_MINIMUM`, the
  release code at e041d5e). Era list ``eras/author_eras_2026-09-23.json``.
- ``archive_no_split_2026_09_24_r2`` (``results/archive_no_split_2026-09-24_r2``):
  no variance split (``off``), and ties within a relative 1e-9 go to the
  selector's order (:data:`pilot_proxy.characterization.surface.SELECTOR_ORDER`,
  the release code at f72bf34). Era list ``eras/author_eras_2026-09-23.json``.
- ``archive_no_split_2026_09_24_r3`` (``results/archive_no_split_2026-09-24_r3``,
  the default): r2 with channel 17's current era dated to 2021-12 inside its
  data gap (the release code at 87c8f74). Era list
  ``eras/author_eras_2026-09-28.json``, the project's own.

Each record names the era list it ran (a file of the project, relative to its
directory) and that file's sha256; the run refuses a list whose bytes differ.
``pilot-proxy characterize archive --record NAME`` selects one; without it
the project's integration model, era list and the selector-order rule apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from pilot_proxy.characterization.surface import MIN_KEPT, SELECTOR_ORDER, FrontierPoint, TieRule
from pilot_proxy.config.integration_model import IntegrationModel


class FirstMinimum:
    """The least residual is the first exact minimum in enumeration order (release code e041d5e).

    The frontier sorts by ``(masked fraction, r_sys)`` and admits a point when
    its residual is strictly below every smaller fraction's.
    """

    name = "first_minimum"

    def least(self, rows, residual, tie_key):
        rows = list(rows)
        return min(rows, key=residual) if rows else None

    def frontier(self, points: Sequence[dict], min_kept: int = MIN_KEPT) -> list[FrontierPoint]:
        return pareto_frontier(points, min_kept=min_kept)


def pareto_frontier(points: Sequence[dict], *, min_kept: int = MIN_KEPT) -> list[FrontierPoint]:
    """The lower envelope of ``r_sys`` against masked fraction, ascending in ``f``.

    A point keeping fewer than ``min_kept`` frames is not on the frontier: the
    selector will not choose one, and its residual is an average over too few
    frames to mean anything.
    """
    rows = [p for p in points if p.get("kept", 0) >= min_kept and math.isfinite(p.get("r_sys", math.nan))]
    if not rows:
        return []
    rows.sort(key=lambda p: (p["masked_fraction"], p["r_sys"]))
    out: list[FrontierPoint] = []
    best = math.inf
    for p in rows:
        if p["r_sys"] < best:
            best = p["r_sys"]
            out.append(FrontierPoint(int(p["rho"]), int(p["eta_q16"]), float(p["eta"]), float(p["masked_fraction"]),
                                     int(p["kept"]), float(p["r_sys"]), float(p.get("exposure_cost_uniform_loss", math.nan))))
    return out


FIRST_MINIMUM = FirstMinimum()


@dataclass(frozen=True)
class ArchiveRecord:
    """One release's declared choices."""

    name: str
    release: str                   # the results directory it reproduces
    variance_split: str            # the integration model's rule
    ties: TieRule
    era_list: str                  # the author-dated era list, relative to the project directory
    era_list_sha256: str           # its bytes as released (the release's author_eras.json)

    def integration_model(self, base: IntegrationModel) -> IntegrationModel:
        return replace(base, variance_split=self.variance_split)


ERAS_2026_09_23 = ("eras/author_eras_2026-09-23.json",
                   "a988e813fcaca817670aa88b4db59b9e5ad92f130159a46524bc0bb722c95ffd")
ERAS_2026_09_28 = ("eras/author_eras_2026-09-28.json",
                   "a37b5fa6b28b22f51f310856e8271cfeab1ee31033f3c05f65f51f0dab2fe5b3")

RECORDS = {
    "archive_author_eras_2026_09_23": ArchiveRecord(
        "archive_author_eras_2026_09_23", "results/archive_author_eras_2026-09-23",
        "booked_when_tau_usable", FIRST_MINIMUM, *ERAS_2026_09_23),
    "archive_no_split_2026_09_24_r2": ArchiveRecord(
        "archive_no_split_2026_09_24_r2", "results/archive_no_split_2026-09-24_r2",
        "off", SELECTOR_ORDER, *ERAS_2026_09_23),
    "archive_no_split_2026_09_24_r3": ArchiveRecord(
        "archive_no_split_2026_09_24_r3", "results/archive_no_split_2026-09-24_r3",
        "off", SELECTOR_ORDER, *ERAS_2026_09_28),
}


def record(name: str) -> ArchiveRecord:
    try:
        return RECORDS[name]
    except KeyError as exc:
        raise KeyError(f"unknown archive record {name!r}; choose from {sorted(RECORDS)}") from exc


__all__ = ["ArchiveRecord", "FIRST_MINIMUM", "FirstMinimum", "RECORDS", "pareto_frontier", "record"]
