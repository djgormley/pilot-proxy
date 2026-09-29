"""The operating-characteristic handoff: ``oc_table.csv``, ``oc_summary.csv`` and ``manifest.json``.

What "operating characteristic" means here. To an electrical engineer an
operating characteristic is P_d against SNR at a fixed P_fa. This table is
not that. For each candidate threshold ``eta`` of a band it gives the
**masked fraction** ``f(eta)`` and the **kept residual** ``r_sys(eta)``;
where a verified signal-free population exists it adds ``P_fa(eta)``. The
science side (its verdict) reads these files and nothing else from
the detector: they are the whole contract, and a producer whose files
validate against the schema can feed the verdict.

Conventions (repeated in the schema's column descriptions):

- two statistics share one ``eta`` column and ``statistic`` says which:
  ``Q`` (the coarse statistic ``F / mu_0``; ``eta`` a multiplier on it) and
  ``Z_rho`` (the fine order-statistic CFAR of rank ``rho``;
  ``eta = eta_q16 / 2^16``);
- the mask rule is ``statistic > eta``: a frame is masked if and only if its
  keep boundary exceeds ``eta``; equality keeps it; a larger ``eta`` masks
  fewer frames;
- ``r`` values are dimensionless power ratios (residual interference power
  over system noise power per frame) in the reference the manifest declares
  (``r_reference``);
- only evaluable candidates are written (at least 30 kept frames, and
  eligible: both calendar halves of the era keep enough frames);
- the fine surface is written thinned: at most 200 candidates per rank,
  evenly spaced in ``eta``, always with the first, the last, the rank's
  least-residual candidate and the marked points; the counts and the
  tightest evaluable threshold of every family are in ``oc_summary``.

Floats are written as Python ``repr`` (round-trip exact); NaN and None as an
empty cell. The manifest is canonical JSON (sorted keys, no NaN) and records
the sha256 of every file it lists.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA_NAME = "operating_characteristic"
SCHEMA_VERSION = 1
SCHEMA_ID = f"{SCHEMA_NAME}_v{SCHEMA_VERSION}"
SCHEMA_FILE = Path(__file__).resolve().parent / "schemas" / f"{SCHEMA_ID}.schema.json"
MASK_RULE = "statistic > eta"
CANDIDATE_SETS = ("fine_surface", "coarse_surface", "coarse_ladder")
POPULATIONS = ("current_era_calibration_block", "all_valid_frames")
BLOCKS = ("calibration", "evaluation")
STATISTICS = ("Q", "Z_rho")
MARKS = ("knee", "least_residual")
NOT_COMPUTED_FINE = "not computed: statistic Z_rho"
# the status vocabularies: exact values, then prefixes followed by a reason
PFA_STATUS_VALUES = ("measured", "undefined: no alpha declared", NOT_COMPUTED_FINE)
PFA_STATUS_PREFIXES = ("unavailable: ", "not defined: ")
ETA_PFA_STATUS_VALUES = ("available", "undefined: no alpha declared", "unsupported: too few frames", NOT_COMPUTED_FINE)
ETA_PFA_STATUS_PREFIXES = ("unsupported: n_eff = ", "unavailable: ", "not defined: ")

# (name, type, unit, description). Types: str, int, float, bool.
OC_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("band_id", "str", "", "Band label from the frequency plan; the join key, with the edges."),
    ("band_low_mhz", "float", "MHz", "Lower band edge from the plan."),
    ("band_high_mhz", "float", "MHz", "Upper band edge from the plan."),
    ("era_label", "str", "", "The current era, e.g. '2018-12..2026-08 (proxy-low)'."),
    ("era_start", "str", "YYYY-MM", "First month of the current era."),
    ("era_end", "str", "YYYY-MM", "Last month of the current era."),
    ("era_state", "str", "", "The era's state from the era list's vocabulary."),
    ("block", "str", "", "calibration or evaluation (the chronological split of the current era)."),
    ("candidate_set", "str", "", "fine_surface, coarse_surface or coarse_ladder."),
    ("population", "str", "", "The frames the candidates were computed on (current_era_calibration_block)."),
    ("threshold_family", "str", "", "The family a deployable set is formed in: 'rho=<rank>' (fine) or 'Q' (coarse)."),
    ("candidate_order", "int", "", "The detector's declared order within (band, candidate set, population, block): "
                                  "rank then multiplier (fine), the grid order (coarse), the rung order (ladder). "
                                  "Ties on the science side are broken by it."),
    ("statistic", "str", "", "Q (coarse, F/mu_0) or Z_rho (fine order-statistic CFAR of rank rho)."),
    ("eta", "float", "statistic units", "The threshold; the mask rule is statistic > eta (equality keeps)."),
    ("eta_exact", "str", "", "The threshold as an exact rational num/den: eta_q16/65536 (fine), "
                             "float(eta).as_integer_ratio() (coarse)."),
    ("frames", "int", "frames", "Frames of the block."),
    ("kept", "int", "frames", "Frames the threshold keeps."),
    ("masked", "int", "frames", "Frames the threshold masks."),
    ("masked_fraction", "float", "", "f = masked/frames. A mixture quantity (detections and false alarms counted "
                                     "together), not a false-alarm rate and not a figure of merit."),
    ("exposure_cost_uniform_loss", "float", "", "1/(1 - f): observing time per unit of kept time if the lost "
                                                "frames are spread evenly (ladder: frames/kept)."),
    ("r_var", "float", "", "Variance residual; 0 by convention (the products carry no variance residual)."),
    ("r_sys", "float", "", "Kept-frame mean of r_i = max(10^(s_i/10), floor)·G, G the chain gain of the declared "
                           "integration model (ladder: the mean at G = 1, times G)."),
    ("r_sys_incoherent", "float", "", "The same kept-frame mean at G = 1: the residual if it averages down "
                                      "incoherently. Never used in place of r_sys."),
    ("r_unmasked", "float", "", "The keep-everything residual on the same block."),
    ("floor_share", "float", "", "Fraction of kept frames whose residual is the floor (no shelf estimate, or the "
                                 "floor at or above the shelf). Near 1, r_sys measures the measurement's "
                                 "sensitivity, not the interference."),
    ("pfa", "float", "", "Empirical P_fa of this eta on the verified signal-free population eta_Pfa reads (Q rows; "
                         "empty when there is none, and on Z_rho rows)."),
    ("pfa_status", "str", "", "measured (pfa is filled), unavailable: <reason>, undefined: no alpha declared, "
                              "not defined: control band ..., or not computed: statistic Z_rho (fine rows: P_fa is "
                              "estimated on Q only)."),
    ("mark", "str", "", "knee, least_residual, both ';'-joined, or empty."),
    ("rho", "int", "", "One-based background rank (cfar_rank + 1); fine rows."),
    ("bulk_size", "int", "bins", "Size of the bulk B; fine rows."),
    ("rank_fraction", "float", "", "rho/(B + 1); fine rows."),
    ("eta_q16", "int", "", "The 16-bit fixed-point encoding of eta ('Q16' is a number format, not the statistic Q)."),
    ("policy", "str", "", "The ladder rung: cal_q0.5 keeps the frames whose Q is at or below the calibration-block "
                          "median of Q; keep_all keeps every frame."),
    ("kept_acquisitions", "int", "", "Acquisitions with a kept frame (ladder rows)."),
    ("kept_days", "int", "", "UTC days with a kept frame (ladder rows)."),
    ("kept_months", "int", "", "Months with sampling support among the kept frames (ladder rows)."),
)
DETECTOR_PARAMETER_COLUMNS = ("rho", "bulk_size", "rank_fraction", "eta_q16", "policy", "kept_acquisitions",
                              "kept_days", "kept_months")

_DRIFT = []
for _tag in ("0", "0p01", "0p05", "0p2"):
    _DRIFT += [(f"drift_candidates_at_{_tag}", "int", "", "Candidates keeping at least this fraction of each half."),
               (f"drift_max_cost_ratio_at_{_tag}", "float", "", "Worst early/late cost ratio over them."),
               (f"drift_max_systematic_ratio_at_{_tag}", "float", "", "Worst early/late r_sys ratio over them.")]


def _point(prefix: str, what: str) -> list[tuple[str, str, str, str]]:
    return [(f"{prefix}_rho", "int", "", f"{what}: rank (fine)."),
            (f"{prefix}_eta_q16", "int", "", f"{what}: Q16 multiplier (fine)."),
            (f"{prefix}_eta", "float", "statistic units", f"{what}: threshold."),
            (f"{prefix}_masked_fraction", "float", "", f"{what}: masked fraction, calibration block."),
            (f"{prefix}_kept", "int", "frames", f"{what}: kept frames, calibration block."),
            (f"{prefix}_r_sys", "float", "", f"{what}: r_sys, calibration block."),
            (f"{prefix}_r_sys_incoherent", "float", "", f"{what}: r_sys at G = 1, calibration block."),
            (f"{prefix}_floor_share", "float", "", f"{what}: floor share, calibration block."),
            (f"{prefix}_exposure_cost_uniform_loss", "float", "", f"{what}: 1/(1 - f)."),
            (f"{prefix}_masked_fraction_evaluation", "float", "", f"{what}: masked fraction replayed on the "
                                                                  "evaluation block."),
            (f"{prefix}_r_sys_evaluation", "float", "", f"{what}: r_sys replayed on the evaluation block.")]


SUMMARY_COLUMNS: tuple[tuple[str, str, str, str], ...] = tuple([
    ("band_id", "str", "", "Band label (join key)."),
    ("band_low_mhz", "float", "MHz", "Lower band edge."),
    ("band_high_mhz", "float", "MHz", "Upper band edge."),
    ("band_role", "str", "", "screened or control."),
    ("era_label", "str", "", "The current era."),
    ("era_start", "str", "YYYY-MM", "First month of the current era."),
    ("era_end", "str", "YYYY-MM", "Last month of the current era."),
    ("era_state", "str", "", "The era's state."),
    ("candidate_set", "str", "", "fine_surface, coarse_surface or coarse_ladder."),
    ("population", "str", "", "The frames the candidates were computed on."),
    ("threshold_family", "str", "", "rho=<rank> or Q."),
    ("eta_eval", "float", "statistic units", "The tightest evaluable threshold of the family; below it the kept set "
                                             "is too small (< 30 frames) or unsupported in an era half."),
    ("candidates_total", "int", "", "Candidates of the family, evaluable or not."),
    ("candidates_evaluable", "int", "", "Evaluable candidates of the family."),
    ("pfa_target", "float", "", "The declared alpha (detection.false_alarm_target)."),
    ("eta_pfa", "float", "statistic units", "The false-alarm limit, in Q: the higher (1 - alpha) quantile of the "
                                            "verified signal-free population. Filled only when eta_pfa_status is "
                                            "available; empty on Z_rho rows."),
    ("eta_pfa_status", "str", "", "available, unsupported: n_eff = N (the point is a diagnostic), unavailable: "
                                  "<reason>, undefined: no alpha declared, not defined: control band ..., or not "
                                  "computed: statistic Z_rho (fine rows)."),
    ("null_source", "str", "", "Where the null population comes from."),
    ("null_frames", "int", "frames", "Frames of the verified signal-free population the estimator read (0 when "
                                     "there is none)."),
    ("pfa_effective_samples", "float", "", "n_eff = n / max(DEFF, 1), from a day-block bootstrap."),
    ("pfa_upper_95", "float", "", "One-sided 95% Clopper-Pearson upper bound on P_fa at eta_Pfa, n_eff trials."),
    ("null_rejection_reason", "str", "", "Why the band has no verified signal-free population."),
    ("eta_pfa_diagnostic", "float", "statistic units", "A diagnostic, never eta_Pfa: the estimator on the current "
                                                       "era of an independently verified but not signal-free band, "
                                                       "or an unsupported point of a verified one (Q rows)."),
    ("eta_pfa_diagnostic_boot_low", "float", "", "Its day-bootstrap 95% interval, low."),
    ("eta_pfa_diagnostic_boot_high", "float", "", "Its day-bootstrap 95% interval, high."),
    ("eta_pfa_diagnostic_n_eff", "float", "", "Its effective sample count."),
    ("eta_pfa_diagnostic_status", "str", "", "Its support status."),
    ("eta_pfa_diagnostic_population", "str", "", "The frames it read."),
    ("eta_pfa_diagnostic_frames", "int", "frames", "How many frames it read (0 when there is no diagnostic)."),
    ("pfa_design_model", "float", "", "The product's fine_p_fa: the fine stage's OS-CFAR design value under "
                                      "i.i.d. bulk; model-conditional; not verified; never eta_Pfa (Z_rho rows only)."),
    ("eta_pfa_ideal_model", "float", "", "The ideal F(2P, 4P) (1 - alpha) quantile of Q divided by the law's mean "
                                         "(mean-scaled; reference only; never used; Q rows only)."),
    ("coarse_raw_width_factor", "float", "", "Coarse null raw width / ideal width (nulls.csv)."),
    ("coarse_core_width_factor", "float", "", "Coarse null core width / ideal width (nulls.csv)."),
    ("fine_raw_width_factor", "float", "", "Fine null raw width / ideal width (nulls.csv)."),
    ("fine_core_width_factor", "float", "", "Fine null core width / ideal width (nulls.csv)."),
    ("exch_rho", "int", "", "Rank of the exchangeability test (the least-residual point's)."),
    ("exch_frames", "int", "frames", "Quiet evaluation-block frames it read."),
    ("exch_measured", "float", "", "Measured per-bin exceedance of the marker bins against T_(rho) of the bulk."),
    ("exch_predicted", "float", "", "The OS-CFAR closed form (|B| + 1 - rho)/(|B| + 1)."),
    ("exch_max_over_designated", "float", "", "Fraction of frames with max_D T > T_(rho)."),
    ("exch_note", "str", "", "Why the test did not run, when it did not."),
    ("floor_db", "float", "dB", "The floor that bounds r_i."),
    ("floor_evidence", "str", "", "measured, stated or refused."),
    ("floor_population", "str", "", "Where the floor comes from."),
    ("floor_verified", "bool", "", "True only for a floor measured on a verified signal-free population."),
    ("chain_gain", "float", "", "G, the coherence gain of the declared integration model."),
    ("chain_gain_status", "str", "", "measured (tau_c measured), bounded (tau_c bounded above) or cap (tau_c "
                                     "refused: G is the cap and r_sys an upper bound)."),
    ("chain_components", "str", "", "The booked (share:n_coh) pairs, ';'-joined, so another integration model can "
                                    "recompute G."),
    ("variance_split", "str", "", "The integration model's variance-split rule."),
    ("tau_c_minutes", "float", "min", "Correlation time (measured value or bound)."),
    ("tau_c_low_minutes", "float", "min", "Its 16th percentile (day-block bootstrap)."),
    ("tau_c_high_minutes", "float", "min", "Its 84th percentile."),
    ("tau_quality", "str", "", "measured, bounded_above or refused."),
    ("tau_outcome", "str", "", "measured, bound or refused (cap)."),
    ("on_shelf_db", "float", "dB", "On-air shelf level."),
    ("intraday_share", "float", "", "Intra-day share of the shelf power (reported where the split is booked)."),
    ("fast_share", "float", "", "Sub-acquisition share (reported where the split is booked)."),
    ("residual_convention", "str", "", "floor-bounded shelf linear x chain gain, variance 0."),
    ("keep_everything_r_sys_calibration", "float", "", "r_sys keeping every frame, calibration block."),
    ("keep_everything_r_sys_evaluation", "float", "", "r_sys keeping every frame, evaluation block."),
    ("stability_status", "str", "", "The within-era drift screen: passed or refused_*. A refusal is a detector "
                                    "refusal."),
    ("stability_reason", "str", "", "The screen's reason."),
    ("stability_points_checked", "int", "", "Candidates the screen checked."),
    ("stability_points_skipped", "int", "", "Candidates it skipped."),
    ("drift_status", "str", "", "The drift diagnostic's status."),
    ("drift_early_frames", "int", "frames", "Frames in the early half."),
    ("drift_late_frames", "int", "frames", "Frames in the late half."),
    ("drift_refused_rho", "int", "", "The candidate the screen refused on: rank."),
    ("drift_refused_eta", "float", "", "Its threshold."),
    ("drift_refused_early_kept", "int", "frames", "Its kept frames, early half."),
    ("drift_refused_late_kept", "int", "frames", "Its kept frames, late half."),
    ("drift_refused_kept_fraction", "float", "", "Its kept fraction."),
    *_DRIFT,
    *_point("knee", "The Pareto knee of the family's candidate set (surface-wide)"),
    *_point("least_residual", "The least-residual point of the candidate set (surface-wide)"),
    ("least_residual_masked_fraction_evaluation_q16", "float", "", "Acquisition-bootstrap 16th percentile."),
    ("least_residual_masked_fraction_evaluation_q84", "float", "", "Acquisition-bootstrap 84th percentile."),
    ("least_residual_r_sys_evaluation_q16", "float", "", "Acquisition-bootstrap 16th percentile."),
    ("least_residual_r_sys_evaluation_q84", "float", "", "Acquisition-bootstrap 84th percentile."),
    ("least_residual_bootstrap_blocks_evaluation", "int", "", "Acquisitions resampled."),
    ("least_residual_r_sys_unmasked_evaluation", "float", "", "Keep-everything r_sys of the evaluation block."),
    ("least_residual_kept_evaluation", "int", "frames", "Kept frames on the evaluation block."),
    ("least_residual_survey_flag_rate_evaluation", "float", "", "Survey-flag rate of the evaluation block."),
    ("least_residual_false_alarm_rate", "float", "", "Masked fraction on the evaluation block when it is a "
                                                     "verified off era (empty otherwise)."),
    ("least_residual_false_alarm_basis", "str", "", "Why a false-alarm rate is or is not measurable there."),
    ("occupancy_class", "str", "", "occupancy wall (survey-flag rate >= 0.90) or below it."),
    ("occupancy_reasons", "str", "", "The reasons, with the release's strings."),
    ("survey_flag_rate_era", "float", "", "Survey-flag rate of the current era."),
    ("claim_status", "str", "", "screening (the stability screen passed) or diagnostic (it refused)."),
    ("refusal", "str", "", "A detector refusal (no floor, bundle, preparation, stability), or empty."),
])


def column_names(columns=OC_COLUMNS) -> tuple[str, ...]:
    return tuple(c[0] for c in columns)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    return value


def write_rows(rows: Sequence[Mapping], path: Path | str, columns=OC_COLUMNS) -> Path:
    """Write rows in the declared column order; floats as repr, NaN and None as empty."""
    names = column_names(columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = sorted({k for r in rows for k in r} - set(names))
    if unknown:
        raise ValueError(f"{path.name}: undeclared columns {unknown}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(names), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in names})
    return path


def write_gzip_copy(path: Path | str) -> Path:
    """A deterministic gzip copy (``path + '.gz'``, mtime 0) for vendoring."""
    path = Path(path)
    out = path.with_name(path.name + ".gz")
    with path.open("rb") as src, out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(src.read())
    return out


def read_rows(path: Path | str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, indent=1, allow_nan=False, ensure_ascii=True) + "\n"


def _column_schema(columns) -> list[dict]:
    return [{"name": n, "type": t, "unit": u, "description": d} for n, t, u, d in columns]


def schema() -> dict:
    """The operating-characteristic schema: the manifest (JSON Schema) and the two tables' columns."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/WVURAIL/pilot-proxy/docs/schemas/{SCHEMA_ID}.schema.json",
        "title": "Operating characteristic handoff (detector to science)",
        "description": __doc__.strip().split("\n\n")[0],
        "type": "object",
        "required": ["schema", "schema_version", "files", "mask_rule", "statistics", "r_reference",
                     "residual_shape", "integration_model", "detector_parameter_columns", "bands", "null_policy",
                     "inputs"],
        "properties": {
            "schema": {"const": SCHEMA_NAME},
            "schema_version": {"const": SCHEMA_VERSION},
            "files": {"type": "object", "minProperties": 2,
                      "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                      "required": ["oc_table.csv", "oc_summary.csv"]},
            "mask_rule": {"const": MASK_RULE},
            "statistics": {"type": "object", "required": list(STATISTICS)},
            "r_reference": {"type": "object",
                            "required": ["noise_bandwidth_hz", "frame_seconds", "feed_sum", "summed_terms"]},
            "residual_shape": {"type": "string"},
            "integration_model": {"type": "object",
                                  "required": ["frame_seconds", "coherence_cap_seconds", "variance_split"]},
            "detector_parameter_columns": {"type": "array", "items": {"type": "string"}},
            "bands": {"type": "array", "items": {"type": "object",
                                                  "required": ["band_id", "low_mhz", "high_mhz", "role"]}},
            "null_policy": {"type": "object", "required": ["alpha", "availability_rule"]},
            "inputs": {"type": "object"},
            "producer": {"type": "object", "description": "Informational; a consumer never requires it."},
        },
        "x-oc-table-columns": _column_schema(OC_COLUMNS),
        "x-oc-summary-columns": _column_schema(SUMMARY_COLUMNS),
        "x-row-rules": {
            "order": ["band_id", "candidate_set", "population", "block", "candidate_order"],
            "evaluable": "kept >= 30 and eligible (both era halves keep enough frames)",
            "fine_surface": "thinned: at most 200 candidates per rank, with each rank's least-residual candidate "
                            "and the marked points",
            "candidate_sets": list(CANDIDATE_SETS), "populations": list(POPULATIONS), "blocks": list(BLOCKS),
            "marks": list(MARKS),
            "pfa_status": {"values": list(PFA_STATUS_VALUES), "prefixes": list(PFA_STATUS_PREFIXES)},
            "eta_pfa_status": {"values": list(ETA_PFA_STATUS_VALUES), "prefixes": list(ETA_PFA_STATUS_PREFIXES)},
            "false_alarm": ("eta_Pfa and P_fa are estimated on Q only: Z_rho rows carry "
                            f"'{NOT_COMPUTED_FINE}' and no Q value; eta_pfa is filled only when available; "
                            "pfa_status measured implies a pfa in [0, 1]; pfa_design_model on Z_rho rows only"),
        },
    }


def schema_text() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _parse(value: str, kind: str):
    if value == "":
        return None
    if kind == "int":
        return int(value)
    if kind == "float":
        out = float(value)
        if repr(out) != value:
            raise ValueError(f"float {value!r} is not in repr form")
        return out
    if kind == "bool":
        if value not in ("True", "False"):
            raise ValueError(f"bool {value!r}")
        return value == "True"
    return value


def check_table(path: Path | str, columns) -> list[str]:
    """Problems with one table against its declared columns ([] when it validates)."""
    names = column_names(columns)
    kinds = {n: t for n, t, _, _ in columns}
    problems = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or tuple(header) != names:
            return [f"{Path(path).name}: header differs from the schema"]
        for number, row in enumerate(reader, start=2):
            if len(row) != len(names):
                problems.append(f"{Path(path).name}:{number}: {len(row)} cells, expected {len(names)}")
                continue
            for name, value in zip(names, row):
                try:
                    _parse(value, kinds[name])
                except ValueError as exc:
                    problems.append(f"{Path(path).name}:{number}: {name}: {exc}")
    return problems


def check_oc_rules(path: Path | str) -> list[str]:
    """Row rules of oc_table.csv: vocabularies, evaluability, order, exact eta and the mask arithmetic."""
    from fractions import Fraction

    problems = []
    rows = read_rows(path)
    last_key = None
    for number, row in enumerate(rows, start=2):
        where = f"oc_table.csv:{number}"
        if row["candidate_set"] not in CANDIDATE_SETS:
            problems.append(f"{where}: candidate_set {row['candidate_set']!r}")
        if row["population"] not in POPULATIONS:
            problems.append(f"{where}: population {row['population']!r}")
        if row["block"] not in BLOCKS:
            problems.append(f"{where}: block {row['block']!r}")
        if row["statistic"] not in STATISTICS:
            problems.append(f"{where}: statistic {row['statistic']!r}")
        if row["mark"] and not set(row["mark"].split(";")) <= set(MARKS):
            problems.append(f"{where}: mark {row['mark']!r}")
        status = row["pfa_status"]
        if not _in_vocabulary(status, PFA_STATUS_VALUES, PFA_STATUS_PREFIXES):
            problems.append(f"{where}: pfa_status {status!r}")
        if status == "measured":
            if row["statistic"] != "Q":
                problems.append(f"{where}: a Z_rho row carries a measured pfa")
            if not row["pfa"] or not 0.0 <= float(row["pfa"]) <= 1.0:
                problems.append(f"{where}: pfa_status measured without a pfa in [0, 1]")
        elif row["pfa"]:
            problems.append(f"{where}: pfa filled with pfa_status {status!r}")
        if row["statistic"] == "Z_rho" and status != NOT_COMPUTED_FINE:
            problems.append(f"{where}: a Z_rho row's pfa_status is not {NOT_COMPUTED_FINE!r}")
        frames, kept, masked = int(row["frames"]), int(row["kept"]), int(row["masked"])
        if kept + masked != frames:
            problems.append(f"{where}: kept + masked != frames")
        if row["block"] == "calibration" and kept < 30:
            problems.append(f"{where}: a non-evaluable candidate is written")
        if row["eta"] and row["eta_exact"]:
            num, den = (int(v) for v in row["eta_exact"].split("/"))
            if Fraction(num, den) != Fraction(float(row["eta"])) and row["statistic"] == "Q":
                problems.append(f"{where}: eta_exact is not the exact value of eta")
            if row["statistic"] == "Z_rho" and (den != 65536 or num != int(row["eta_q16"])
                                                or float(row["eta"]) != num / den):
                problems.append(f"{where}: fine eta is not eta_q16/65536")
        key = (row["band_id"], row["candidate_set"], row["population"], row["block"], int(row["candidate_order"]))
        order_key = (int(key[0]) if key[0].isdigit() else key[0], CANDIDATE_SETS.index(key[1]) if key[1] in CANDIDATE_SETS else 9,
                     key[2], BLOCKS.index(key[3]) if key[3] in BLOCKS else 9, key[4])
        if last_key is not None and order_key <= last_key:
            problems.append(f"{where}: rows out of order")
        last_key = order_key
    return problems


def _in_vocabulary(value: str, values: Sequence[str], prefixes: Sequence[str]) -> bool:
    """An exact value, or a prefix followed by a non-empty reason."""
    return value in values or any(value.startswith(p) and len(value) > len(p) for p in prefixes)


# the columns that carry a value of the coarse estimator (Q), empty on fine rows
_Q_ONLY = ("eta_pfa", "pfa_effective_samples", "pfa_upper_95", "eta_pfa_diagnostic", "eta_pfa_diagnostic_boot_low",
           "eta_pfa_diagnostic_boot_high", "eta_pfa_diagnostic_n_eff", "eta_pfa_ideal_model")
_DIAGNOSTIC = ("eta_pfa_diagnostic", "eta_pfa_diagnostic_boot_low", "eta_pfa_diagnostic_boot_high",
               "eta_pfa_diagnostic_n_eff")


def check_summary_rules(path: Path | str) -> list[str]:
    """Row rules of oc_summary.csv's false-alarm columns (H2, M1, M2, L10): vocabulary, eta_Pfa only when
    available, Q quantities on Q rows and the fine design value on fine rows only."""
    problems = []
    for number, row in enumerate(read_rows(path), start=2):
        where = f"oc_summary.csv:{number}"
        status = row["eta_pfa_status"]
        fine = row["candidate_set"] == "fine_surface"
        if not _in_vocabulary(status, ETA_PFA_STATUS_VALUES, ETA_PFA_STATUS_PREFIXES):
            problems.append(f"{where}: eta_pfa_status {status!r}")
        if bool(row["eta_pfa"]) != (status == "available"):
            problems.append(f"{where}: eta_pfa {'filled' if row['eta_pfa'] else 'empty'} with status {status!r}")
        if status == "available" and not row["pfa_upper_95"]:
            problems.append(f"{where}: an available eta_pfa without its bound")
        if fine:
            if status != NOT_COMPUTED_FINE:
                problems.append(f"{where}: a fine row's eta_pfa_status is not {NOT_COMPUTED_FINE!r}")
            filled = [c for c in _Q_ONLY if row[c]]
            if filled or row["eta_pfa_diagnostic_status"] or row["null_frames"] not in ("", "0"):
                problems.append(f"{where}: a fine row carries Q values {filled}")
            if not row["pfa_design_model"]:
                problems.append(f"{where}: a fine row lacks the design value")
        else:
            if row["pfa_design_model"]:
                problems.append(f"{where}: a Q row carries the fine design value")
            if row["eta_pfa_diagnostic_status"] and not row["eta_pfa_diagnostic"]:
                problems.append(f"{where}: a diagnostic status without a diagnostic value")
            if any(row[c] for c in _DIAGNOSTIC) and not row["eta_pfa_diagnostic_status"]:
                problems.append(f"{where}: a diagnostic value without its status")
            if row["eta_pfa_diagnostic_frames"] not in ("", "0") and not row["eta_pfa_diagnostic"]:
                problems.append(f"{where}: diagnostic frames without a diagnostic")
    return problems


def check_manifest(manifest: Mapping, directory: Path | str | None = None) -> list[str]:
    """Problems with a manifest (required keys, constants, and the file digests when ``directory`` is given)."""
    spec = schema()
    problems = [f"manifest: missing {k}" for k in spec["required"] if k not in manifest]
    if manifest.get("schema") != SCHEMA_NAME:
        problems.append("manifest: schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("manifest: schema_version")
    if manifest.get("mask_rule") != MASK_RULE:
        problems.append("manifest: mask_rule")
    for key in spec["properties"]["files"]["required"]:
        if key not in manifest.get("files", {}):
            problems.append(f"manifest: files lacks {key}")
    if directory is not None:
        for name, digest in manifest.get("files", {}).items():
            path = Path(directory) / name
            if not path.is_file() or sha256_file(path) != digest:
                problems.append(f"manifest: {name} digest differs")
    return problems


def validate_directory(directory: Path | str) -> list[str]:
    """Every problem with one characterization directory's handoff files ([] when they validate)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    problems = check_manifest(manifest, directory)
    problems += check_table(directory / "oc_table.csv", OC_COLUMNS)
    problems += check_table(directory / "oc_summary.csv", SUMMARY_COLUMNS)
    if not problems:
        problems += check_oc_rules(directory / "oc_table.csv")
        problems += check_summary_rules(directory / "oc_summary.csv")
    return problems


def write_manifest(directory: Path | str, manifest: Mapping, files: Iterable[str]) -> Path:
    """Write ``manifest.json`` with the sha256 of each listed file (relative to ``directory``)."""
    directory = Path(directory)
    body = dict(manifest)
    body["schema"] = SCHEMA_NAME
    body["schema_version"] = SCHEMA_VERSION
    body["files"] = {name: sha256_file(directory / name) for name in sorted(files)}
    path = directory / "manifest.json"
    path.write_text(canonical_json(body), encoding="utf-8")
    return path


__all__ = ["BLOCKS", "CANDIDATE_SETS", "DETECTOR_PARAMETER_COLUMNS", "ETA_PFA_STATUS_PREFIXES", "ETA_PFA_STATUS_VALUES",
           "MARKS", "MASK_RULE", "NOT_COMPUTED_FINE", "OC_COLUMNS", "PFA_STATUS_PREFIXES", "PFA_STATUS_VALUES",
           "POPULATIONS", "SCHEMA_FILE", "SCHEMA_ID", "SCHEMA_NAME", "SCHEMA_VERSION", "STATISTICS",
           "SUMMARY_COLUMNS", "canonical_json", "check_manifest", "check_oc_rules", "check_summary_rules",
           "check_table", "column_names",
           "read_rows", "schema", "schema_text", "sha256_file", "validate_directory", "write_gzip_copy",
           "write_manifest", "write_rows"]
