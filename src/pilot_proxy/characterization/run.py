"""``pilot-proxy characterize archive``: the detector characterization of every band.

Per band, in order: the product's geometry and frame accounting; the eras
(the predeclared rule on the selected frames, replaced by the project's
author-dated era list where it dates the band); the band's dated
transmitter-off population and the two tests that would make it a verified
signal-free null; the chronological calibration/evaluation split of the
current era; the anchors; the per-frame spectrum containment; the residual
chain under the project's integration model; the null calibration and the
floor; the operating surface of the calibration block, its knee and its
least-residual point replayed on the evaluation block; the coarse rule's
surface and the coarse retention ladder; the exchangeability at the
least-residual rank; the baseline flaggers; the occupancy class; the
false-alarm limit; and the ledger record.

Nothing here reads a science tolerance. The outputs are the handoff files
(:mod:`.oc_table`) and the detector tables in their frozen layouts:

    <out>/characterization/{oc_table.csv, oc_table.csv.gz, oc_summary.csv, manifest.json}
    <out>/characterization/tables/{eras, eras_channels, anchors, containment, kstar, nulls, chain,
                                   flaggers, held_out_spectra}.csv
    <out>/characterization/channels/chNN/{eras.json, anchor_contrast.csv, spectra_window.json,
                                          held_out_spectra.csv, held_out_spectra.npz}
    <out>/characterization/ledger/{channels/chNN_fidF.json, run.json, ledger.csv}

The frozen layouts. The exchangeability columns of ``nulls.csv``, the
reported-point row of ``flaggers.csv`` and the held-out spectra exist in the
releases only on the bands where the release's selection ran. The bands named
by ``--replay-points`` get them in that layout (a replay of the release's
selection, read from the releases, not from a tolerance); every band gets the
same quantities in the operating-characteristic files.

Bands run in parallel processes; each opens its own product and writes its
own per-band files. The parent writes the tables, the K* rule over all bands,
the handoff files and the ledger.

    pilot-proxy characterize archive --project projects/chime_atsc --products DIR --out DIR
        [--record NAME] [--replay-points FILE] [--bands 14,29] [--campaign-last-month YYYY-MM]
        [--bootstrap-replicates 1000] [--bootstrap-seed 20260907] [--workers 8]
        [--control-product FILE --control-run-dir DIR]
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.project import Project, default_project, load_project
from pilot_proxy.detectors.narrowband_marker import (
    NarrowbandMarkerAdapter,
    anchors,
    psd,
)
from pilot_proxy.detectors.narrowband_marker.scores import build_score_bundle
from pilot_proxy.products.reader import COARSE_BIN_HZ, FINE_BIN_HZ, Product, sha256_of

from . import (
    blocks,
    coarse_ladder,
    eras,
    false_alarm,
    flaggers,
    ledger,
    masked_spectra,
    nulls,
    oc_table,
    occupancy,
    residual_chain,
    stability,
    surface,
)
from .surface import SELECTOR_ORDER, TieRule


def producer_identity(roots: Mapping[str, Path] | None = None) -> dict:
    """The producing code, informational: its commit, dirty flag and a digest of the package sources.

    Delegates to :func:`pilot_proxy.provenance.producer_identity`.
    """
    from pilot_proxy.provenance import producer_identity as identity
    return identity(roots)


# the profile parts the general layers bind at import (the register's floor, null, chain, stability and
# false-alarm constants; the detector configuration's null degrees of freedom; the instrument's geometry)
DEFAULT_BOUND_PARTS = ("register", "detector_config", "instrument")


def _require_default_layers(project: Project) -> None:
    """Refuse a project whose register, detector configuration or instrument differ from the default project's.

    The characterization modules read those parts from the default project
    when they are imported, so another project's values would otherwise be
    mixed silently with the default's. A project that differs only in its
    frequency plan, eras, transmitter-off record, interference template or
    integration model is threaded through and accepted.
    """
    default = default_project()
    theirs, ours = project.file_sha256(), default.file_sha256()
    differ = [k for k in ("register", "detector_config") if theirs[k] != ours[k]]
    if project.instrument_name != default.instrument_name:
        differ.append("instrument")
    if differ:
        raise ProfileError(f"project {project.name!r} at {project.directory}: its {', '.join(differ)} differ from the "
                           f"default project's, which the characterization binds at import; refused rather than mixed")


def _quiet_cohort_is_null(product: Product, mask) -> bool:
    """Whether a mask's coarse-quiet frames are a null population: the bulk's centre lies at mu_0."""
    frames = np.asarray(mask, dtype=bool) & product.selected
    if frames.sum() < nulls.MIN_NULL_FRAMES:
        return True
    widths = nulls.describe_null(product.statistic[frames], nulls.COARSE_DOF)
    return bool(math.isfinite(widths.centre) and abs(widths.centre - 1.0) <= nulls.OFF_CENTRE_TOLERANCE)


def _fine_bin_of_rf(rf_offset_hz: float, geometry) -> int:
    """The padded fine bin an RF offset from the nominal marker falls on (the inverse of anchors.rf_offset_of_bin)."""
    fine_hz = geometry.grid_residual_hz + float(np.sign(geometry.sense) or 1) * float(rf_offset_hz)
    return int(round(fine_hz / FINE_BIN_HZ)) % anchors.FINE_BINS


def _rate(flag: np.ndarray, block: np.ndarray) -> float:
    """Fraction of a block's frames carrying ``flag`` (NaN on an empty block)."""
    block = np.asarray(block, dtype=bool)
    return float(np.asarray(flag, dtype=bool)[block].mean()) if block.any() else math.nan


def _label(era: eras.Era | None) -> str:
    return f"{era.first_label}..{era.last_label} ({era.state})" if era else "no era"


def _write_csv(rows: Sequence[dict], path: Path) -> Path | None:
    import csv
    rows = [r for r in rows if r]
    if not rows:
        return None
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) or v is None else repr(v) if isinstance(v, float) else v)
                        for k, v in ((k, r.get(k)) for k in keys)})
    return path


def _month_index(label: str) -> int:
    y, m = (int(x) for x in label.split("-"))
    return y * 12 + m - 1


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


# ------------------------------------------------------------------ the handoff rows of one band
def _band_key(band, table: eras.EraTable) -> dict:
    era = table.current_era
    return {"band_id": band.label, "band_low_mhz": float(band.low_mhz), "band_high_mhz": float(band.high_mhz),
            "era_label": _label(era), "era_start": era.first_label if era else "", "era_end": era.last_label if era else "",
            "era_state": era.state if era else ""}


def _marks(rho: int, eta_q16: int, knee_point, least_row) -> str:
    out = []
    if knee_point is not None and (knee_point.rho, knee_point.eta_q16) == (rho, eta_q16):
        out.append("knee")
    if least_row and (least_row.get("rho"), least_row.get("eta_q16")) == (rho, eta_q16):
        out.append("least_residual")
    return ";".join(out)


def _fine_rows(key: dict, block: surface.BlockCharacterization, knee_op, replays: dict, pfa_status: str,
               rule: TieRule) -> list[dict]:
    knee_point = knee_op.point if knee_op is not None else None
    selected = []
    if knee_point is not None:
        selected.append((knee_point.rho, knee_point.eta_q16))
    if block.least_residual:
        selected.append((block.least_residual["rho"], block.least_residual["eta_q16"]))
    thinned = surface.thin_points(block.points, selected or None, rule=rule)
    base = {**key, "candidate_set": "fine_surface", "population": "current_era_calibration_block",
            "statistic": "Z_rho", "pfa": None, "pfa_status": pfa_status}
    rows = []
    for p in thinned:
        rows.append({**base, "block": "calibration", "threshold_family": f"rho={p['rho']}",
                     "candidate_order": p["candidate_order"], "eta": p["eta"], "eta_exact": f"{p['eta_q16']}/65536",
                     "frames": p["frames"], "kept": p["kept"], "masked": p["frames"] - p["kept"],
                     "masked_fraction": p["masked_fraction"], "exposure_cost_uniform_loss": p["exposure_cost_uniform_loss"],
                     "r_var": 0.0, "r_sys": p["r_sys"], "r_sys_incoherent": p["r_sys_incoherent"],
                     "r_unmasked": block.unmasked_residual, "floor_share": p["floor_share"],
                     "mark": _marks(p["rho"], p["eta_q16"], knee_point, block.least_residual),
                     "rho": p["rho"], "bulk_size": block.bulk_size, "rank_fraction": p["rank_fraction"],
                     "eta_q16": p["eta_q16"]})
    by_key = {(p["rho"], p["eta_q16"]): p for p in block.points}
    for (rho, eta_q16), rep in sorted(replays.items(), key=lambda kv: by_key[kv[0]]["candidate_order"]):
        p = by_key[(rho, eta_q16)]
        rows.append({**base, "block": "evaluation", "threshold_family": f"rho={rho}",
                     "candidate_order": p["candidate_order"], "eta": p["eta"], "eta_exact": f"{eta_q16}/65536",
                     "frames": rep.frames, "kept": rep.kept, "masked": rep.frames - rep.kept,
                     "masked_fraction": rep.masked_fraction,
                     "exposure_cost_uniform_loss": 1.0 / (1.0 - rep.masked_fraction) if rep.masked_fraction < 1.0 else math.nan,
                     "r_var": 0.0, "r_sys": rep.retained_residual, "r_sys_incoherent": rep.retained_residual_incoherent,
                     "r_unmasked": rep.unmasked_residual, "floor_share": rep.floor_share,
                     "mark": _marks(rho, eta_q16, knee_point, block.least_residual),
                     "rho": rho, "bulk_size": block.bulk_size, "rank_fraction": p["rank_fraction"], "eta_q16": eta_q16})
    return rows


def _row_pfa(population, eta):
    return false_alarm.exceedance(population, eta) if population is not None else None


def _coarse_rows(key: dict, frontier: list[dict], unmasked: float, pfa_status: str, population=None) -> list[dict]:
    rows = []
    for order, f in enumerate(frontier):
        if not f["evaluable"]:
            continue
        num, den = float(f["eta_c"]).as_integer_ratio()
        rows.append({**key, "block": "calibration", "candidate_set": "coarse_surface",
                     "population": "current_era_calibration_block", "threshold_family": "Q", "candidate_order": order,
                     "statistic": "Q", "eta": f["eta_c"], "eta_exact": f"{num}/{den}", "frames": f["frames"],
                     "kept": f["kept"], "masked": f["frames"] - f["kept"], "masked_fraction": f["masked_fraction"],
                     "exposure_cost_uniform_loss": 1.0 / (1.0 - f["masked_fraction"]) if f["masked_fraction"] < 1.0 else math.nan,
                     "r_var": 0.0, "r_sys": f["r_sys"], "r_sys_incoherent": f["r_sys_incoherent"],
                     "r_unmasked": unmasked, "floor_share": f["floor_share"], "pfa": _row_pfa(population, f["eta_c"]),
                     "pfa_status": pfa_status, "mark": ""})
    return rows


def _ladder_rows(key: dict, rungs: list[dict], pfa_status: str, population=None) -> list[dict]:
    rows = []
    keep_all = next((r for r in rungs if r["policy"] == "keep_all"), None)
    for block_name in ("calibration", "evaluation"):
        unmasked = keep_all[block_name]["chain_allowance"] if keep_all else math.nan
        for order, rung in enumerate(rungs):
            rec = rung[block_name]
            if rec["kept"] < coarse_ladder.MINIMUM:
                continue
            eta = rung["eta"]
            exact = f"{eta.as_integer_ratio()[0]}/{eta.as_integer_ratio()[1]}" if eta else ""
            rows.append({**key, "block": block_name, "candidate_set": "coarse_ladder", "population": rung["population"],
                         "threshold_family": "Q", "candidate_order": order, "statistic": "Q",
                         "eta": eta if eta is not None else None, "eta_exact": exact, "frames": rec["frames"],
                         "kept": rec["kept"], "masked": rec["frames"] - rec["kept"],
                         "masked_fraction": (rec["frames"] - rec["kept"]) / rec["frames"] if rec["frames"] else math.nan,
                         "exposure_cost_uniform_loss": rec["mask_only_cost"], "r_var": 0.0,
                         "r_sys": rec["chain_allowance"], "r_sys_incoherent": rec["G1_allowance"],
                         "r_unmasked": unmasked, "floor_share": None, "pfa": _row_pfa(population, eta),
                         "pfa_status": pfa_status, "mark": "",
                         "policy": rung["policy"], "kept_acquisitions": rec["kept_acquisitions"],
                         "kept_days": rec["kept_days"], "kept_months": rec["kept_supported_months"]})
    return rows


def _point_columns(prefix: str, point: dict | None, replay=None) -> dict:
    point = point or {}
    return {f"{prefix}_rho": point.get("rho"), f"{prefix}_eta_q16": point.get("eta_q16"),
            f"{prefix}_eta": point.get("eta", math.nan), f"{prefix}_masked_fraction": point.get("masked_fraction", math.nan),
            f"{prefix}_kept": point.get("kept"), f"{prefix}_r_sys": point.get("r_sys", math.nan),
            f"{prefix}_r_sys_incoherent": point.get("r_sys_incoherent", math.nan),
            f"{prefix}_floor_share": point.get("floor_share", math.nan),
            f"{prefix}_exposure_cost_uniform_loss": point.get("exposure_cost_uniform_loss", math.nan),
            f"{prefix}_masked_fraction_evaluation": replay.masked_fraction if replay is not None else math.nan,
            f"{prefix}_r_sys_evaluation": replay.retained_residual if replay is not None else math.nan}


def _knee_dict(op) -> dict | None:
    if op is None or op.point is None:
        return None
    p = op.point
    return {"rho": p.rho, "eta_q16": p.eta_q16, "eta": p.eta, "masked_fraction": p.masked_fraction, "kept": p.kept,
            "r_sys": p.r_sys, "exposure_cost_uniform_loss": p.cost}


# ------------------------------------------------------------------ one band
def characterize_band(path: str, out_dir: str, *, project_dir: str, campaign_last_month: int, replicates: int, seed: int,
                      era_config: eras.EraConfig | None = None, era_spec: Sequence[Sequence[str]] | None = None,
                      record_name: str | None = None, replay: bool = False, role: str = "screened") -> dict:
    """The whole characterization of one product; returns small rows, writes large files."""
    t0 = time.time()
    project = load_project(project_dir)
    _require_default_layers(project)
    record_config = project.record_module("archive_releases").record(record_name) if record_name else None
    model = record_config.integration_model(project.integration_model) if record_config else project.integration_model
    rule: TieRule = record_config.ties if record_config else SELECTOR_ORDER
    out = Path(out_dir)
    p = Product(path, require_health=True)
    g = p.geometry
    ch = g.physical_channel
    band = project.frequency_plan.band(str(ch))
    ch_dir = out / "channels" / f"ch{ch:02d}"
    ch_dir.mkdir(parents=True, exist_ok=True)
    config = era_config or eras.DEFAULT_CONFIG
    record = ledger.ChannelRecord(ch, g.freq_id, p.path.name, sha256_of(p.path))
    notes = record.notes

    # 0. what the product is: its geometry and its frame accounting
    lo_mhz, hi_mhz = float(band.low_mhz), float(band.high_mhz)
    record.add("geometry", {
        "pilot_hz": g.pilot_hz, "centre_hz": g.centre_hz, "sense": g.sense, "grid_residual_hz": g.grid_residual_hz,
        "nominal_fine_bin": g.nominal_fine_bin, "nominal_psd_bin": g.nominal_psd_bin,
        "centre_line_rf_offset_hz": g.centre_line_rf_offset_hz, "stored_window_centre": g.stored_window_centre,
        "allocation_low_mhz": lo_mhz, "allocation_high_mhz": hi_mhz, "coarse_bin_hz": COARSE_BIN_HZ, "fine_bin_hz": FINE_BIN_HZ,
        "fine_pad_factor": p.fine_pad_factor, "fine_guard_bins": p.fine_guard_bins,
        "census_excluded_bins": ";".join(str(int(b)) for b in np.atleast_1d(p.fine_census_excluded_bins)),
        "mu0": p.mu0, "shelf_offset_db": float(p.view.shelf_offset_db), "band_role": role,
    })
    months_all = eras.frame_months(p)
    sel_months = months_all[p.selected & (months_all >= 0)]
    record.add("product", {
        "n_frames": p.n_frames, "n_valid": int(p.valid.sum()), "n_selected": int(p.selected.sum()),
        "n_units": int(np.unique(p.frame_unit_index[p.selected]).size) if p.selected.any() else 0,
        "n_rejected_flag": int((p.selected & p.rejected).sum()),
        "n_with_shelf_estimate": int((p.selected & np.isfinite(p.shelf_db)).sum()),
        "n_without_time": int((p.selected & ~np.isfinite(p.frame_time)).sum()),
        "health_schema": p.health.schema, "health_excluded": int((p.valid & ~p.health.include).sum()),
        "health_reasons": ";".join(f"{k}:{v}" for k, v in sorted(p.health.reason_counts.items())),
        "first_month": blocks.month_label(int(sel_months.min())) if sel_months.size else "",
        "last_month": blocks.month_label(int(sel_months.max())) if sel_months.size else "",
        "per_frame_spectra": "psd_frame_db_i16" in p.archive.files,
    })

    # 1. eras
    # Dates inferred from this archive are diagnostics, not an independent
    # vote supporting a detected transition.
    table = eras.era_table(p, config=config, campaign_last_month=campaign_last_month, station_record={})
    if era_spec:
        # an author-dated era list replaces the rule's eras; the rule's month record is kept beside it
        table = eras.impose_eras(table, p, era_spec)
    months = eras.frame_months(p)
    off = nulls.transmitter_off(p, months, project.eras.transmitter_off.get(str(ch), ()), table)
    off_through, off_from = off.off_through, off.off_from
    record.add("archive_inferred_events", {"events": nulls.station_records(off_from, off_through),
                                           "independently_verified": off.independently_verified,
                                           "external_record": off.external_record})
    eras.write_era_json(table, ch_dir / "eras.json")
    era = table.current_era
    timed = np.isfinite(p.frame_time)
    era_mask_all = table.current_era_mask(p)
    era_mask = era_mask_all & timed          # one block definition for every module: frames without a time are excluded
    untimed_excluded = int((era_mask_all & ~timed).sum())
    # Only a verified signal-free population (independently verified and null-like) calibrates the null and the
    # floor; a dated off epoch that is not one is recorded, and excluded.
    off_mask = off.mask if off.signal_free else None
    if off.mask is not None and not off.signal_free:
        if off.independently_verified:
            notes.append(f"independently verified off population is not null-like ({off.null_check}); "
                         "excluded from null calibration and false-alarm claims")
        else:
            notes.append("archive-inferred low epoch is not independently verified; excluded from null calibration and false-alarm claims")
    if off.note:
        notes.append(off.note)
    # the current era is an off era when the verified off population covers most of it; whether that
    # population also reads as a null is reported beside it
    off_ok, off_widths, off_reason = nulls.off_population_check(p, off_mask)
    off_majority = bool(off_mask is not None and era_mask.any() and off_mask[era_mask].mean() > 0.5)
    off_era_current = off_majority
    if off_mask is not None and not off_ok:
        notes.append(f"recorded off population is not null-like ({off_reason}): a carrier persists after the record")
    record.add("era", {**eras.channel_row(table), "off_record_frames": off.record_frames, "off_population_frames": off.off_frames,
                       "off_null_like": off_ok if off_widths is not None else None, "off_majority_of_current_era": off_majority,
                       "off_era_current": off_era_current, "current_level_median_db": era.level_median_db if era else math.nan,
                       "off_independently_verified": off.independently_verified,
                       "off_population_null_like": off.null_like, "off_population_check": off.null_check,
                       "off_population_r68": off.null_r68, "off_signal_free": off.signal_free})
    era_label = _label(era)
    # on an off era the transmitter's position, containment and chain are read from the last on era (the previous era)
    reference_index = table.current_index - 1 if (off_era_current and table.current_index > 0) else table.current_index
    if reference_index >= 0 and reference_index != table.current_index:
        reference_mask = table.era_mask(p, reference_index) & timed
        reference_label = f"previous era {_label(table.eras[reference_index])} (current era is off)"
        notes.append(f"anchor, containment and chain read on the {reference_label}")
    else:
        reference_mask, reference_label = era_mask, f"current era {era_label}"

    # 2. blocks (month support on the era procedure's own gate)
    split = blocks.split_blocks(p.frame_unit_index, p.unit_time, era_mask, frame_time=p.frame_time, minimum_months=1,
                                month_kwargs=dict(min_frames=config.min_frames, min_units=config.min_units, min_days=config.min_days))
    # Eras remain retrospective. Within the selected era, every fitted PSD
    # fallback and chain term is restricted to pre-evaluation samples.
    reference_mask = reference_mask & ~split.evaluation
    reference_label = reference_label + " / calibration-only"
    record.add("evaluation_contract", {
        "status": "retrospective era-conditioned replay",
        "era_discovery": "full archive; not an untouched prospective holdout",
        "fitted_anchor_psd_chain": "evaluation frames excluded",
        "station_event_basis": ("independently verified" if off.independently_verified else "archive-inferred, not independently verified"),
    })
    off_for_null = (off_mask & ~split.evaluation) if off_mask is not None else None      # the evaluation block never enters the null
    off_for_eval = (off_mask & split.evaluation) if off_mask is not None else None
    record.add("blocks", {
        "status": split.status, "detail": split.detail, "boundary_time": split.boundary_time,
        "calibration_frames": split.calibration_frames, "evaluation_frames": split.evaluation_frames,
        "calibration_units": split.calibration_units, "evaluation_units": split.evaluation_units,
        "calibration_months": len(split.calibration_months), "evaluation_months": len(split.evaluation_months),
        "frames_without_time_excluded": untimed_excluded,
        "boundary_month": blocks.month_label(int(blocks.month_index([split.boundary_time])[0])) if math.isfinite(split.boundary_time) else "",
        "calibration_finite_estimate_rate": _rate(np.isfinite(p.shelf_db), split.calibration),
        "evaluation_finite_estimate_rate": _rate(np.isfinite(p.shelf_db), split.evaluation),
        "calibration_flag_rate": _rate(p.rejected, split.calibration), "evaluation_flag_rate": _rate(p.rejected, split.evaluation),
        "calibration_first_month": blocks.month_label(split.calibration_months[0].month) if split.calibration_months else "",
        "calibration_last_month": blocks.month_label(split.calibration_months[-1].month) if split.calibration_months else "",
        "evaluation_first_month": blocks.month_label(split.evaluation_months[0].month) if split.evaluation_months else "",
        "evaluation_last_month": blocks.month_label(split.evaluation_months[-1].month) if split.evaluation_months else "",
    })

    # 3. anchors. The on-minus-quiet estimator needs a quiet cohort that is a null: where a mask's coarse bulk sits
    # far above mu_0 the frames below mu_0 are the carrier's lower tail, and the plain median is used instead.
    ratio = anchors.fine_ratio(p)
    anchor_results = []
    anc = {}
    quiet_ok = {}
    masks = [("calibration", split.calibration), ("evaluation", split.evaluation), ("current_era", era_mask)]
    if table.current_index > 0:
        masks.append(("previous_era", table.era_mask(p, table.current_index - 1) & timed))
    for label, mask in masks:
        quiet_ok[label] = _quiet_cohort_is_null(p, mask)
        anc[label] = anchors.anchor(p, mask, label, ratio=ratio, replicates=replicates, seed=seed, quiet_usable=quiet_ok[label])
        anchor_results.append(anc[label])
    anchors.write_contrast_curves(anchor_results, ch_dir / "anchor_contrast.csv")
    # the table's anchor: the current era's (eq 8.1), or the previous era's on an off-era band
    if reference_index != table.current_index and anc.get("previous_era") is not None and anc["previous_era"].status == "ok":
        table_anchor = anc["previous_era"]
    else:
        table_anchor = anc["current_era"]
    # the selector's anchor: the calibration block's (held out from the evaluation block)
    selector_anchor = anc["calibration"]
    record.add("anchor", {**anchors.anchor_record(table_anchor), "source": table_anchor.label,
                          "quiet_cohort_is_null": quiet_ok.get(table_anchor.label)})
    record.add("anchor_calibration", {**anchors.anchor_record(selector_anchor), "source": selector_anchor.label,
                                      "quiet_cohort_is_null": quiet_ok.get(selector_anchor.label)})
    record.add("anchor_evaluation", anchors.anchor_record(anc["evaluation"]))
    record.add("anchor_era", anchors.anchor_record(anc["current_era"]))
    if "previous_era" in anc:
        row = anchors.anchor_record(anc["previous_era"])
        both = anc["previous_era"].status == "ok" and anc["current_era"].status == "ok"
        row["shift_from_previous_bins"] = anchors.anchor_shift_bins(anc["current_era"], anc["previous_era"]) if both else None
        record.add("anchor_previous", row)
    else:
        record.add("anchor_previous", None)

    # 4. containment on the reference era (needs the per-frame spectra the campaign products carry). The table's
    # fine anchor is compared with the spectrum's in-span lobe: a disagreement beyond the designated half-width
    # sets the sentinel; an out-of-window anchor is an alias only when it folds onto the out-of-span feature by
    # one coarse bin.
    anchor_suspect, anchor_note, lobe_bin = False, "", None
    if reference_mask.any() and "psd_frame_db_i16" in p.archive.files:
        spectrum = psd.accumulate_spectra(p, reference_mask)
        first = psd.analyse(spectrum, g)
        lobe_hz = first.row.in_span_refined_offset_hz if first.row.in_span_recovered else math.nan
        anchor_hz = float(selector_anchor.anchor_rf_offset_hz) if selector_anchor.status == "ok" else math.nan
        offset_bins = psd.anchor_lobe_offset_bins(anchor_hz, lobe_hz)
        disagree = math.isfinite(offset_bins) and abs(offset_bins) > anchors.DESIGNATED_HALF_WIDTH + 0.5
        folds = False
        if math.isfinite(first.row.out_of_span_offset_hz) and math.isfinite(anchor_hz):
            tol = (anchors.DESIGNATED_HALF_WIDTH + 0.5) * FINE_BIN_HZ
            folds = any(abs(anchor_hz - (first.row.out_of_span_offset_hz + k * COARSE_BIN_HZ)) <= tol for k in (-1, 1))
        reasons = []
        if folds:
            reasons.append(f"fine anchor aliases the out-of-span feature at {first.row.out_of_span_offset_hz:.0f} Hz by one coarse bin")
        elif selector_anchor.status == "ok" and selector_anchor.aliased_out_of_window:
            notes.append("fine anchor lies beyond the +-30-bin acquisition window (no fold onto an out-of-span feature)")
        if disagree:
            reasons.append(f"fine anchor {offset_bins:+.1f} bins from the PSD in-span lobe")
        anchor_note = "; ".join(reasons)
        cont = psd.analyse(spectrum, g, anchor_aliases=bool(folds or disagree), anchor_note=anchor_note)
        psd.write_spectra_json([cont], ch_dir / "spectra_window.json", provenance=reference_label)
        containment_row = cont.row
        mode_mass = float(selector_anchor.boot_mode_mass) if selector_anchor.status == "ok" else math.nan
        anchor_suspect = bool(disagree or folds or (math.isfinite(mode_mass) and mode_mass < 0.5))
        if first.row.in_span_recovered:
            lobe_bin = _fine_bin_of_rf(lobe_hz, g)
        record.add("containment", {**dataclasses.asdict(cont.row), "anchor_lobe_offset_bins": offset_bins,
                                   "anchor_lobe_disagree": bool(disagree), "anchor_folds_out_of_span": bool(folds),
                                   "anchor_suspect": anchor_suspect, "lobe_fine_bin": lobe_bin, "era": reference_label})
    else:
        cont, containment_row = None, None
        record.add("containment", None)
        if reference_mask.any():
            notes.append("containment skipped: product carries no per-frame spectra")

    # 5. chain on the reference era, with the archive-wide chain beside it for comparison
    gain_basis = ""
    try:
        ch_res = residual_chain.chain_on_frames(p, reference_mask, model, population=reference_label,
                                                off_through=off_through, off_from=off_from)
        record.add("chain", ch_res.as_row())
        gain, gain_basis = ch_res.gain, "era chain"
    except Exception as exc:  # the chain refuses loudly on some bands; record, do not stop
        ch_res = None
        record.add("chain", None)
        notes.append(f"chain on the {reference_label}: {type(exc).__name__}: {exc}")
        gain = math.nan
    try:
        ch_all = residual_chain.chain(p.path, model, off_through=off_through, off_from=off_from)
        record.add("chain_archive", ch_all.as_row())
        # Archive-wide values are descriptive only. Falling back to them
        # would fit the evaluation data after calibration had refused.
    except Exception as exc:
        record.add("chain_archive", None)
        notes.append(f"archive-wide chain: {type(exc).__name__}: {exc}")

    # 6. null calibration on the calibration block (floor first; exchangeability after the surface). The
    # evaluation block never enters the null: the floor and the widths that score both blocks are read from the
    # calibration block, or from a verified signal-free population outside the evaluation block. When the fine
    # anchor is suspect the nominal window is excluded from the fine null's bulk (the bulk may carry the marker).
    # The selector's anchor is the calibration block's; when that anchor is suspect and the spectrum recovered an
    # in-span lobe, the lobe's fine bin is used instead, labelled.
    anchor_bin = int(selector_anchor.anchor_bin) if selector_anchor.status == "ok" else g.nominal_fine_bin
    anchor_source = selector_anchor.label if selector_anchor.status == "ok" else "nominal bin"
    if anchor_suspect and lobe_bin is not None:
        anchor_bin, anchor_source = int(lobe_bin), "psd in-span lobe (fine anchor suspect)"
        notes.append(f"selector anchor taken from the PSD in-span lobe (bin {anchor_bin}): {anchor_note or 'anchor bootstrap mode mass below 0.5'}")
    bulk = anchors.bulk_mask(anchor_bin, pad_factor=p.fine_pad_factor, guard_fine_bins=p.fine_guard_bins,
                             census_excluded_bins=p.fine_census_excluded_bins)
    exclude = anchors.window_bins(g.nominal_fine_bin, anchors.WINDOW_HALF_WIDTH) if anchor_suspect else None
    if split.calibration.any():
        null_block, null_label = split.calibration, f"{era_label}/calibration"
    else:
        null_block, null_label = era_mask, era_label
        if era_mask.any():
            notes.append("null calibrated on the whole era: no calibration block")
    null_cal = nulls.calibrate_null(p, null_block, anchor_bin=anchor_bin, bulk_mask=bulk, era_label=null_label,
                                    off_era=off_for_null, fine_t=ratio, exclude_fine_bins=exclude)
    floor = surface.Floor(null_cal.floor.db, null_cal.floor.evidence, null_cal.floor.population)

    # 7. the operating surface on the calibration block, its least-residual point replayed on the evaluation block
    if not math.isfinite(gain):
        notes.append("surface skipped: no chain gain (both chains refused)")
    if split.calibration.any() and math.isfinite(gain):
        block = surface.characterize_block(
            p, split.calibration, split.evaluation, anchor_bin=anchor_bin, bulk_mask=bulk, floor=floor,
            era_label=era_label, gain=gain, latest_era=True, off_era=off_era_current,
            bootstrap_replicates=replicates, bootstrap_seed=seed, rule=rule)
        block = dataclasses.replace(block, anchor_sentinel=anchor_note)
    else:
        block = None
        notes.append("surface skipped: no calibration frames or no chain gain")
    exch_rho = block.least_residual.get("rho") if block is not None else None
    null_cal_exch = null_cal
    if exch_rho is not None:
        null_cal_exch = nulls.calibrate_null(p, null_block, anchor_bin=anchor_bin, bulk_mask=bulk, era_label=null_label,
                                             off_era=off_for_null, rho=int(exch_rho), quiet_block=split.evaluation,
                                             fine_t=ratio, exclude_fine_bins=exclude)
    # the frozen layout carries the exchangeability only on the replayed bands
    null_row_cal = null_cal_exch if replay else null_cal
    record.add("null", {**null_cal_exch.as_row(), "anchor_bin": anchor_bin, "anchor_source": anchor_source,
                        "exchangeability_rank_basis": ("least-residual point" if exch_rho is not None else "")})
    # the same description on the evaluation block
    if split.evaluation.any():
        null_eval = nulls.calibrate_null(p, split.evaluation, anchor_bin=anchor_bin, bulk_mask=bulk,
                                         era_label=f"{era_label}/evaluation", off_era=off_for_eval, fine_t=ratio,
                                         exclude_fine_bins=exclude)
        record.add("null_evaluation", null_eval.as_row())
    else:
        record.add("null_evaluation", None)
    keep, drift, frontier_rows, knee_op, replays = {}, {}, [], None, {}
    coarse_summary = surface.coarse_least([], rule)
    if block is not None:
        # the keep-everything residual on each block (r_keep of the chain)
        for name, blk in (("calibration", split.calibration), ("evaluation", split.evaluation)):
            rows = np.flatnonzero(np.asarray(blk, dtype=bool))
            try:
                keep[f"keep_everything_r_sys_{name}"] = float(residual_chain.frame_residuals(p, rows, floor, gain).mean()) if rows.size else math.nan
            except ValueError:
                keep[f"keep_everything_r_sys_{name}"] = math.nan
        # why the drift screen refused: the candidate it refused on, and the drift a selected point would see
        try:
            bundle = build_score_bundle(
                p.path, split.calibration, anchor_bin=int(anchor_bin),
                designated_half_width=surface.DESIGNATED_HALF_WIDTH, bulk_mask=bulk)
            drift = stability.drift_diagnostic(
                bundle, residual_chain.frame_residuals(p, bundle.source_row_index, floor, gain),
                p.frame_time[bundle.source_row_index])
        except Exception as exc:
            notes.append(f"drift diagnostic: {type(exc).__name__}: {exc}")
            drift = {"drift_status": f"{type(exc).__name__}"}
        # the coarse rule's own surface (f, r_sys) on the calibration block, beside the fine surface
        frontier_rows = surface.coarse_surface(p, split.calibration, floor, gain)
        coarse_summary = surface.coarse_least(frontier_rows, rule)
        record.add("surface", {**block.as_row(), **surface.surface_summary(block.points, rule), **keep,
                               "anchor_source": anchor_source, "gain_basis": gain_basis, **coarse_summary, **drift,
                               "tie_rule": rule.name})
        # the knee of the calibration block's own mask-against-residual frontier, and both notable points on the
        # evaluation block
        if block.points:
            knee_op = surface.knee(ch, list(block.points), rule=rule)
            record.add("knee", knee_op.as_row())
            for note in knee_op.notes:
                notes.append(f"knee: {note}")
        if block.least_residual and block.evaluation is not None:
            replays[(block.least_residual["rho"], block.least_residual["eta_q16"])] = block.evaluation
        if knee_op is not None and knee_op.point is not None and split.evaluation.any():
            key = (knee_op.point.rho, knee_op.point.eta_q16)
            if key not in replays:
                try:
                    replays[key] = surface.replay(p, split.evaluation, anchor_bin=anchor_bin, bulk_mask=bulk,
                                                  rho=key[0], eta_q16=key[1], floor=floor, gain=gain,
                                                  off_era=off_era_current, replicates=replicates, seed=seed)
                except Exception as exc:
                    notes.append(f"knee replay: {type(exc).__name__}: {exc}")
    else:
        record.add("surface", None)
    if knee_op is None:
        record.add("knee", None)

    # 7b. what the knee removes from the held-out spectrum (the frozen layout: replayed bands only)
    spectra = []
    op = knee_op if replay else None
    if op is not None and op.point is not None and split.evaluation.any() and "psd_frame_db_i16" in p.archive.files:
        for basis, rho, eta_q16, eta in (("operating point", op.point.rho, op.point.eta_q16, op.point.eta),):
            try:
                kept_frames = masked_spectra.kept_at_point(p, split.evaluation, anchor_bin=anchor_bin, bulk_mask=bulk,
                                                           rho=rho, eta_q16=eta_q16)
                spectra.append(masked_spectra.measure(p, split.evaluation, kept_frames, basis=basis, rho=rho,
                                                      eta_q16=eta_q16, eta=eta))
            except Exception as exc:
                notes.append(f"held-out spectra ({basis}): {type(exc).__name__}: {exc}")
        # the survey flag on the same block, as the reference the knee is measured against
        try:
            spectra.append(masked_spectra.measure(p, split.evaluation, split.evaluation & ~p.rejected,
                                                  basis="survey flag", rho=0, eta_q16=surface.Q16_SCALE, eta=1.0))
        except Exception as exc:
            notes.append(f"held-out spectra (survey flag): {type(exc).__name__}: {exc}")
    if spectra:
        masked_spectra.write_spectra_rows(spectra, ch_dir / "held_out_spectra.csv")
        masked_spectra.write_spectra_npz(spectra, ch_dir / "held_out_spectra.npz")
        record.add("held_out", {f"{s.basis.replace(' ', '_')}_{k}": v for s in spectra for k, v in s.as_row().items()
                                if k not in ("channel", "freq_id", "basis")})
    else:
        record.add("held_out", None)

    # 8. the baseline flaggers on the same frames (the reported point only on the replayed bands)
    try:
        flag_cmp = flaggers.compare(
            p, era_mask, floor_db=floor.db, floor_evidence=floor.evidence, era_label=era_label,
            diagnostic=(block.least_residual if (replay and block is not None) else None), anchor_bin=anchor_bin, bulk_mask=bulk)
        record.add("flaggers", flaggers.channel_row(flag_cmp))
    except Exception as exc:
        flag_cmp = None
        record.add("flaggers", None)
        notes.append(f"flaggers: {type(exc).__name__}: {exc}")

    # 9. the coarse retention ladder on the calibration block
    rungs = []
    if split.calibration.any():
        try:
            rungs = coarse_ladder.ladder(p, split.calibration, split.evaluation, floor, gain)
            record.add("ladder", {r["policy"]: {"eta": r["eta"], "calibration_kept": r["calibration"]["kept"],
                                                "evaluation_kept": r["evaluation"]["kept"]} for r in rungs})
        except Exception as exc:
            notes.append(f"coarse ladder: {type(exc).__name__}: {exc}")

    # 10. occupancy and the false-alarm limit
    flag_rate = float(p.rejected[era_mask].mean()) if era_mask.any() else math.nan
    occ = occupancy.occupancy(flag_rate, era_state=era.state if era else "",
                              era_level_db=float(era.level_median_db) if era else math.nan)
    record.add("occupancy", {**occ.as_row(), "survey_flag_rate_era": flag_rate})
    limit = false_alarm.false_alarm_limit(band.label, role=role, off_population=off, q=p.statistic,
                                          frame_time=p.frame_time, current_era=era_mask)
    record.add("false_alarm", limit)
    # P_fa of each Q row on the frames eta_Pfa reads (a verified signal-free population in the current era); a fine
    # (Z_rho) row carries no Q value (H2: eta_Pfa and P_fa are estimated on Q only)
    pfa_frames = false_alarm.verified_population(off, era_mask)
    pfa_population = p.statistic[pfa_frames] if pfa_frames is not None else None
    q_status = false_alarm.row_pfa_status(limit, "Q", pfa_population is not None)
    fine_status = false_alarm.row_pfa_status(limit, "Z_rho", pfa_population is not None)

    # 11. the handoff rows
    key = _band_key(band, table)
    oc_rows = []
    if block is not None and block.points:
        oc_rows += _fine_rows(key, block, knee_op, replays, fine_status, rule)
    oc_rows += _coarse_rows(key, frontier_rows, block.unmasked_residual if block is not None else math.nan, q_status,
                            pfa_population)
    oc_rows += _ladder_rows(key, rungs, q_status, pfa_population)
    fine_design = float(p.scalar("fine_p_fa"))
    exch = null_cal_exch.exchangeability
    limit_columns = ("pfa_target", "eta_pfa", "eta_pfa_status", "null_source", "null_frames", "pfa_effective_samples",
                     "pfa_upper_95", "null_rejection_reason", "eta_pfa_diagnostic", "eta_pfa_diagnostic_boot_low",
                     "eta_pfa_diagnostic_boot_high", "eta_pfa_diagnostic_n_eff", "eta_pfa_diagnostic_status",
                     "eta_pfa_diagnostic_population", "eta_pfa_diagnostic_frames")
    q_limit = {k: limit[k] for k in limit_columns}
    # the Q quantities on Q rows only, the fine stage's OS-CFAR design value on fine rows only
    q_columns = {**q_limit, "pfa_design_model": math.nan, "eta_pfa_ideal_model": false_alarm.ideal_model_threshold()}
    fine_columns = {**{k: v for k, v in false_alarm.fine_row_columns(q_limit).items() if k in limit_columns},
                    "pfa_design_model": fine_design, "eta_pfa_ideal_model": math.nan}
    common = {**{k: v for k, v in key.items()}, "band_role": role,
              "coarse_raw_width_factor": null_cal.coarse.raw_width_factor, "coarse_core_width_factor": null_cal.coarse.core_width_factor,
              "fine_raw_width_factor": null_cal.fine.raw_width_factor, "fine_core_width_factor": null_cal.fine.core_width_factor,
              "exch_rho": exch.rho if exch else exch_rho, "exch_frames": exch.frames if exch else None,
              "exch_measured": exch.measured_rate if exch else math.nan,
              "exch_predicted": exch.predicted_rate if exch else math.nan,
              "exch_max_over_designated": exch.max_over_test_rate if exch else math.nan,
              "exch_note": "; ".join(n for n in null_cal_exch.notes if n.startswith("exchangeability")),
              "floor_db": floor.db, "floor_evidence": floor.evidence, "floor_population": floor.population,
              "floor_verified": bool(floor.evidence == "measured" and off.signal_free),
              "chain_gain": gain, "chain_gain_status": ch_res.gain_status if ch_res else "unmeasured",
              "chain_components": ch_res.components_text() if ch_res else "",
              "variance_split": model.variance_split,
              "tau_c_minutes": ch_res.tau_c_minutes if ch_res else math.nan,
              "tau_c_low_minutes": ch_res.tau_c_low / 60.0 if (ch_res and math.isfinite(ch_res.tau_c_low)) else math.nan,
              "tau_c_high_minutes": ch_res.tau_c_high / 60.0 if (ch_res and math.isfinite(ch_res.tau_c_high)) else math.nan,
              "tau_quality": ch_res.tau_quality if ch_res else "", "tau_outcome": ch_res.tau_outcome if ch_res else "",
              "on_shelf_db": ch_res.on_shelf_db if ch_res else math.nan,
              "intraday_share": ch_res.intraday_share if ch_res else math.nan,
              "fast_share": ch_res.fast_share if ch_res else math.nan,
              "residual_convention": "floor-bounded shelf linear x chain gain, variance 0",
              "keep_everything_r_sys_calibration": keep.get("keep_everything_r_sys_calibration", math.nan),
              "keep_everything_r_sys_evaluation": keep.get("keep_everything_r_sys_evaluation", math.nan),
              "stability_status": block.stability.get("status", "") if block else "",
              "stability_reason": block.stability.get("reason", "") if block else "",
              "stability_points_checked": block.stability.get("points_checked") if block else None,
              "stability_points_skipped": block.stability.get("points_skipped") if block else None,
              **{k: v for k, v in drift.items() if k.startswith("drift_")},
              "occupancy_class": occ.occupancy_class, "occupancy_reasons": "; ".join(occ.reasons),
              "survey_flag_rate_era": flag_rate,
              "claim_status": block.claim_status if block else "", "refusal": block.refusal if block else ""}
    summary_rows = []
    if block is not None:
        lr = block.least_residual
        ev = block.evaluation
        mf = (ev.masked_fraction_bootstrap or {}) if ev else {}
        rr = (ev.retained_residual_bootstrap or {}) if ev else {}
        knee_key = (knee_op.point.rho, knee_op.point.eta_q16) if (knee_op and knee_op.point) else None
        notable = {**_point_columns("knee", _knee_dict(knee_op), replays.get(knee_key) if knee_key else None),
                   **_point_columns("least_residual", lr, ev),
                   "least_residual_masked_fraction_evaluation_q16": mf.get("q0.16", math.nan),
                   "least_residual_masked_fraction_evaluation_q84": mf.get("q0.84", math.nan),
                   "least_residual_r_sys_evaluation_q16": rr.get("q0.16", math.nan),
                   "least_residual_r_sys_evaluation_q84": rr.get("q0.84", math.nan),
                   "least_residual_bootstrap_blocks_evaluation": mf.get("blocks", 0) if ev else None,
                   "least_residual_r_sys_unmasked_evaluation": ev.unmasked_residual if ev else math.nan,
                   "least_residual_kept_evaluation": ev.kept if ev else None,
                   "least_residual_survey_flag_rate_evaluation": ev.survey_flag_rate if ev else math.nan,
                   "least_residual_false_alarm_rate": ev.false_alarm_rate if ev else math.nan,
                   "least_residual_false_alarm_basis": ev.false_alarm_basis if ev else ""}
        if knee_key is not None and knee_op.point is not None:
            for p_row in block.points:
                if (p_row["rho"], p_row["eta_q16"]) == knee_key:
                    notable["knee_r_sys_incoherent"] = p_row["r_sys_incoherent"]
                    notable["knee_floor_share"] = p_row["floor_share"]
                    break
        by_rho: dict[int, list[dict]] = {}
        for p_row in block.points:
            by_rho.setdefault(int(p_row["rho"]), []).append(p_row)
        for rho in sorted(block.candidates_by_rho):
            family = by_rho.get(rho, [])
            summary_rows.append({**common, **fine_columns, **notable, "candidate_set": "fine_surface",
                                 "population": "current_era_calibration_block", "threshold_family": f"rho={rho}",
                                 "eta_eval": family[0]["eta"] if family else math.nan,
                                 "candidates_total": block.candidates_by_rho[rho], "candidates_evaluable": len(family)})
    if frontier_rows:
        evaluable = [f for f in frontier_rows if f["evaluable"]]
        least = next((f for f in evaluable if f["eta_c"] == coarse_summary["coarse_min_r_sys_eta"]), None)
        coarse_point = ({"eta": least["eta_c"], "masked_fraction": least["masked_fraction"], "kept": least["kept"],
                         "r_sys": least["r_sys"], "r_sys_incoherent": least["r_sys_incoherent"],
                         "floor_share": least["floor_share"],
                         "exposure_cost_uniform_loss": 1.0 / (1.0 - least["masked_fraction"]) if least["masked_fraction"] < 1.0 else math.nan}
                        if least else None)
        summary_rows.append({**common, **q_columns, **_point_columns("least_residual", coarse_point),
                             "candidate_set": "coarse_surface",
                             "population": "current_era_calibration_block", "threshold_family": "Q",
                             "eta_eval": evaluable[0]["eta_c"] if evaluable else math.nan,
                             "candidates_total": len(frontier_rows), "candidates_evaluable": len(evaluable)})
    if rungs:
        evaluable = [r for r in rungs if r["calibration"]["kept"] >= coarse_ladder.MINIMUM]
        summary_rows.append({**common, **q_columns, "candidate_set": "coarse_ladder", "population": rungs[0]["population"],
                             "threshold_family": "Q",
                             "eta_eval": min((r["eta"] for r in evaluable if r["eta"] is not None), default=math.nan),
                             "candidates_total": len(rungs), "candidates_evaluable": len(evaluable)})

    p.close()
    return {
        "record": record,
        "era_rows": eras.era_rows(table), "era_channel_row": eras.channel_row(table),
        "anchor_rows": [anchors.anchor_row(a) for a in anchor_results],
        "containment_row": containment_row,
        "null_row": null_row_cal.as_row(),
        "chain_row": ch_res.as_row() if ch_res is not None else None,
        "flagger_rows": [r.as_row() for r in flag_cmp.rows] if flag_cmp is not None else [],
        "held_out_rows": [s.as_row() for s in spectra],
        "oc_rows": oc_rows, "summary_rows": summary_rows,
        "seconds": time.time() - t0,
    }


def _worker(args):
    path, out_dir, kwargs = args
    try:
        return characterize_band(path, out_dir, **kwargs)
    except Exception:
        return {"error": traceback.format_exc(), "path": path}


# ------------------------------------------------------------------ the replay file
def read_replay_points(path: Path | str | None) -> dict[int, dict]:
    """The bands whose frozen-layout replay is written: ``band_id`` and, optionally, the point it must reproduce."""
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "band_id" not in rows[0]:
        raise ValueError(f"{path}: the replay file needs a band_id column")
    return {int(r["band_id"]): r for r in rows}


def _check_replay(results: Sequence[dict], replay_points: Mapping[int, dict]) -> list[str]:
    """Where a replay row names the point it reproduces, the least-residual point must be it."""
    problems = []
    for r in results:
        ch = r["record"].channel
        want = replay_points.get(ch)
        if not want or not want.get("least_residual_rho"):
            continue
        section = r["record"].sections.get("surface") or {}
        got = (section.get("least_residual_rho"), section.get("least_residual_eta_q16"))
        if (str(got[0]), str(got[1])) != (want["least_residual_rho"], want["least_residual_eta_q16"]):
            problems.append(f"band {ch}: least-residual point {got} differs from the replay file's "
                            f"({want['least_residual_rho']}, {want['least_residual_eta_q16']})")
    return problems


# ------------------------------------------------------------------ the whole archive
def _sort_oc(rows: list[dict]) -> list[dict]:
    order = {name: i for i, name in enumerate(oc_table.CANDIDATE_SETS)}
    return sorted(rows, key=lambda r: (int(r["band_id"]), order[r["candidate_set"]], r["population"],
                                       oc_table.BLOCKS.index(r["block"]), int(r["candidate_order"])))


def _handoff_manifest(project: Project, bands_written, products: Mapping[str, str], *, record_name, rule, model,
                      replay_file, producer) -> dict:
    adapter = NarrowbandMarkerAdapter(project)
    law = adapter.null_law()
    instrument, detector = project.instrument, project.detector_config
    return {
        "mask_rule": oc_table.MASK_RULE,
        "statistics": {
            "Q": {"definition": "coarse statistic F / mu_0", "null_law": law.name, "null_dof": list(law.dof),
                  "eta": "a multiplier on Q"},
            "Z_rho": {"definition": "fine order-statistic CFAR of one-based background rank rho: the marker-bin "
                                    "maximum over the rank-rho bulk value", "eta": "eta_q16 / 65536"},
        },
        "r_reference": {"noise_bandwidth_hz": project.interference.band_width_hz,
                        "frame_seconds": model.frame_seconds, "feed_sum": detector.feed_sum,
                        "summed_terms": detector.summed_terms(instrument),
                        "definition": "residual interference power over system noise power per frame, in the band"},
        "residual_shape": project.interference.residual_shape,
        "integration_model": {"frame_seconds": model.frame_seconds, "coherence_cap_seconds": model.coherence_cap_seconds,
                              "variance_split": model.variance_split},
        "detector_parameter_columns": list(oc_table.DETECTOR_PARAMETER_COLUMNS),
        "bands": [{"band_id": b.label, "low_mhz": float(b.low_mhz), "high_mhz": float(b.high_mhz), "role": b.role,
                   "marker_freq_id": int(project.target_freq_id(b))} for b in bands_written],
        "null_policy": {"alpha": float(false_alarm.ALPHA),
                        "availability_rule": (f"eta_Pfa is available when n_eff >= "
                                              f"{false_alarm.MIN_FRAMES_PER_FALSE_ALARM:g} / alpha on a verified "
                                              "signal-free population (independently verified, null-like, in the "
                                              "current era)"),
                        "estimator": ("higher (1 - alpha) order statistic; day-block bootstrap "
                                      f"(B = {false_alarm.B}); n_eff = n / max(DEFF, 1); one-sided 95% "
                                      "Clopper-Pearson bound with n_eff trials"),
                        "minimum_frames_per_false_alarm": false_alarm.MIN_FRAMES_PER_FALSE_ALARM},
        "inputs": {"products": dict(sorted(products.items())), "profile": project.file_sha256(),
                   "era_list_sha256": project.eras.source_sha256,
                   "transmitter_off_sha256": project.eras.transmitter_off_sha256,
                   "replay_points": ({"path": str(replay_file), "sha256": oc_table.sha256_file(replay_file)}
                                     if replay_file else None)},
        "record": record_name, "tie_rule": rule.name,
        "fine_surface_rows": "thinned: at most 200 candidates per rank, with each rank's least-residual candidate "
                             "and the marked points",
        "producer": producer,
    }


def characterize_archive(products_dir: Path | str, out_dir: Path | str, *, project_dir: Path | str,
                         workers: int = 6, replicates: int = blocks.DEFAULT_REPLICATES, seed: int = blocks.DEFAULT_SEED,
                         bands: Sequence[int] | None = None, campaign_last_month: str | None = None,
                         era_config: eras.EraConfig | None = None, record_name: str | None = None,
                         replay_points: Path | str | None = None, control_product: Path | str | None = None,
                         control_run_dir: Path | str | None = None, generated: str | None = None) -> dict:
    """Characterize every screened band's product (and, optionally, one control band into its own directory)."""
    project = load_project(project_dir)
    _require_default_layers(project)
    project_dir = str(project.directory)
    products_dir, out = Path(products_dir), Path(out_dir) / "characterization"
    out.mkdir(parents=True, exist_ok=True)
    producer = producer_identity()                   # read once, before any work: the code that runs is the code named
    rule = SELECTOR_ORDER
    model = project.integration_model
    if record_name:
        rec = project.record_module("archive_releases").record(record_name)
        rule, model = rec.ties, rec.integration_model(project.integration_model)
    screened = {int(b.label) for b in project.frequency_plan.bands("screened")}
    paths = sorted(products_dir.glob("*.npz"))
    opened = [Product(p, require_health=True) for p in paths]
    all_bands = {p.geometry.physical_channel: p for p in opened if p.geometry.physical_channel in screened}
    config = era_config or eras.DEFAULT_CONFIG
    # the campaign snapshot is the last populated month over every product present, whatever subset runs
    campaign_last = (_month_index(campaign_last_month) if campaign_last_month
                     else eras.campaign_last_populated_month(list(all_bands.values()), config))
    by_band = {c: all_bands[c] for c in bands} if bands else all_bands
    products = {p.path.name: sha256_of(p.path) for c, p in sorted(all_bands.items())}
    replay = read_replay_points(replay_points)
    overrides = project.eras.overrides_spec()
    jobs = [(str(p.path), str(out), dict(project_dir=project_dir, campaign_last_month=campaign_last, replicates=replicates,
                                         seed=seed, era_config=config, era_spec=overrides.get(c), record_name=record_name,
                                         replay=c in replay))
            for c, p in sorted(by_band.items())]
    for p in opened:
        p.close()
    results, errors = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if "error" in r:
                errors.append(r)
                print(f"FAILED {Path(r['path']).name}:\n{r['error']}", flush=True)
            else:
                results.append(r)
                occ = (r["record"].sections.get("occupancy") or {}).get("occupancy_class", "")
                print(f"done ch{r['record'].channel:02d} in {r['seconds']:.0f}s: {occ}", flush=True)
    results.sort(key=lambda r: r["record"].channel)
    replay_problems = _check_replay(results, replay)
    for problem in replay_problems:
        print(f"REPLAY MISMATCH {problem}", flush=True)
    _write_outputs(out, results, project=project, products=products,
                   bands_written=[project.frequency_plan.band(str(r["record"].channel)) for r in results],
                   record_name=record_name, rule=rule, model=model, replay_file=replay_points, producer=producer,
                   run={"generated": generated or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "products_dir": str(products_dir), "channels": sorted(by_band),
                        "campaign_last_month": blocks.month_label(campaign_last),
                        "era_config": json.loads(config.canonical_json()), "era_config_digest": config.digest,
                        "era_overrides": {str(c): [list(e) for e in v] for c, v in sorted(overrides.items())},
                        "era_overrides_sha256": hashlib.sha256(json.dumps(
                            {str(c): [list(e) for e in v] for c, v in sorted(overrides.items())},
                            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "bootstrap": {"replicates": replicates, "seed": seed},
                        "record": record_name, "tie_rule": rule.name,
                        "integration_model": dataclasses.asdict(model),
                        "replay_bands": sorted(replay), "replay_problems": replay_problems,
                        "errors": [{"path": e["path"], "error": e["error"].splitlines()[-1]} for e in errors],
                        "seconds": time.time() - t0})
    control = None
    if control_product is not None:
        if control_run_dir is None:
            raise ValueError("a control product needs its own --control-run-dir")
        control = characterize_control(control_product, control_run_dir, project=project, campaign_last=campaign_last,
                                       replicates=replicates, seed=seed, era_config=config, record_name=record_name,
                                       rule=rule, model=model, producer=producer)
    return {"channels": [r["record"].channel for r in results], "errors": errors, "out": str(out),
            "replay_problems": replay_problems, "control": control, "seconds": time.time() - t0}


def _write_outputs(out: Path, results: Sequence[dict], *, project: Project, products: Mapping[str, str], bands_written,
                   record_name, rule, model, replay_file, producer, run: dict) -> None:
    tables = out / "tables"
    _write_csv([row for r in results for row in r["era_rows"]], tables / "eras.csv")
    _write_csv([r["era_channel_row"] for r in results], tables / "eras_channels.csv")
    _write_csv([row for r in results for row in r["anchor_rows"]], tables / "anchors.csv")
    cont_rows = [r["containment_row"] for r in results if r["containment_row"] is not None]
    if cont_rows:
        psd.write_containment_csv(cont_rows, tables / "containment.csv")
        psd.write_kstar_csv(psd.k_star_table(cont_rows), tables / "kstar.csv")
    _write_csv([r["null_row"] for r in results], tables / "nulls.csv")
    _write_csv([r["chain_row"] for r in results], tables / "chain.csv")
    _write_csv([row for r in results for row in r["flagger_rows"]], tables / "flaggers.csv")
    _write_csv([row for r in results for row in r["held_out_rows"]], tables / "held_out_spectra.csv")
    oc_table.write_rows(_sort_oc([row for r in results for row in r["oc_rows"]]), out / "oc_table.csv")
    oc_table.write_gzip_copy(out / "oc_table.csv")
    oc_table.write_rows([row for r in results for row in r["summary_rows"]], out / "oc_summary.csv",
                        oc_table.SUMMARY_COLUMNS)
    manifest = _handoff_manifest(project, bands_written, products, record_name=record_name, rule=rule, model=model,
                                 replay_file=replay_file, producer=producer)
    manifest["tables"] = {p.name: oc_table.sha256_file(p) for p in sorted(tables.glob("*.csv"))}
    oc_table.write_manifest(out, manifest, ("oc_table.csv", "oc_table.csv.gz", "oc_summary.csv"))
    book = ledger.Ledger(run={**run, "producer": producer, "products": dict(products),
                              "provisional": {"stability.minimum_half_retained_frames": stability.PROVISIONAL_MIN_HALF_RETAINED,
                                              "stability.maximum_cost_ratio": stability.PROVISIONAL_MAX_COST_RATIO,
                                              "stability.maximum_systematic_residual_ratio": stability.PROVISIONAL_MAX_SYSTEMATIC_RATIO,
                                              "occupancy_wall_flag_rate": occupancy.OCCUPANCY_WALL_FLAG_RATE,
                                              "null_scale_probes": list(nulls.CORE_PROBES),
                                              "containment_window_hz": psd.WINDOW_HZ, "e_min": psd.E_MIN}})
    for r in results:
        book.add(r["record"])
    book.write(out / "ledger")


def characterize_control(product: Path | str, run_dir: Path | str, *, project: Project, campaign_last: int,
                         replicates: int, seed: int, era_config: eras.EraConfig, record_name, rule, model,
                         producer) -> dict:
    """One control band (no transmitter) into its own directory: the rule's eras, no dated off epoch, no eta_Pfa."""
    product = Path(product)
    out = Path(run_dir) / "characterization"
    out.mkdir(parents=True, exist_ok=True)
    with Product(product, require_health=True) as p:
        band = project.frequency_plan.band(str(p.geometry.physical_channel))
        if band.role != "control":
            raise ValueError(f"band {band.label} has role {band.role!r}, not control")
        if int(p.geometry.freq_id) != int(project.target_freq_id(band)):
            raise ValueError(f"control band {band.label} is read at freq_id {project.target_freq_id(band)}")
    result = characterize_band(str(product), str(out), project_dir=str(project.directory), campaign_last_month=campaign_last,
                               replicates=replicates, seed=seed, era_config=era_config, era_spec=None,
                               record_name=record_name, replay=False, role="control")
    _write_outputs(out, [result], project=project, products={product.name: sha256_of(product)}, bands_written=[band],
                   record_name=record_name, rule=rule, model=model, replay_file=None, producer=producer,
                   run={"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "control_product": str(product), "channels": [int(band.label)], "role": "control",
                        "campaign_last_month": blocks.month_label(campaign_last),
                        "bootstrap": {"replicates": replicates, "seed": seed}})
    return {"band": band.label, "out": str(out)}


def main(argv=None) -> int:
    import argparse

    from pilot_proxy.config.project import default_project_dir

    parser = argparse.ArgumentParser(prog="pilot-proxy characterize archive", description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, default=None, help="project profile directory (default: the repository's)")
    parser.add_argument("--products", type=Path, required=True, help="directory of per-pilot products (*.npz)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (characterization/ is written in it)")
    parser.add_argument("--record", default=None, help="a named record configuration (records/chime_atsc_2026)")
    parser.add_argument("--replay-points", type=Path, default=None,
                        help="CSV naming the bands whose frozen-layout replay is written (band_id column)")
    parser.add_argument("--bands", default=None, help="comma-separated band labels (default: every screened band)")
    parser.add_argument("--campaign-last-month", default=None, help="YYYY-MM of the campaign snapshot")
    parser.add_argument("--bootstrap-replicates", type=int, default=blocks.DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=blocks.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--control-product", type=Path, default=None)
    parser.add_argument("--control-run-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if (args.control_product is None) != (args.control_run_dir is None):
        parser.error("--control-product and --control-run-dir go together")
    result = characterize_archive(
        args.products, args.out, project_dir=args.project or default_project_dir(), workers=args.workers,
        replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
        bands=[int(b) for b in args.bands.split(",")] if args.bands else None,
        campaign_last_month=args.campaign_last_month, record_name=args.record, replay_points=args.replay_points,
        control_product=args.control_product, control_run_dir=args.control_run_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=1, default=str), flush=True)
    return 1 if (result["errors"] or result["replay_problems"]) else 0


__all__ = ["characterize_archive", "characterize_band", "characterize_control", "main", "producer_identity",
           "read_replay_points"]
