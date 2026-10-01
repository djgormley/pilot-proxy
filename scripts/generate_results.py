#!/usr/bin/env python3
"""PilotProxy survey results generator (v2: amendment modes + full depth).

One-shot post-processing driver for an (optionally interrupted) production
`pilot-proxy chime-scan`. Runs CPU-only against the atomic per-pilot
checkpoints on disk -- no GPU session required. Stages:

  1. discover   locate run dirs under --runs-root (or take --run-dir),
                inventory per-pilot products, auto-detect a pilot-free
                control run for the cleaning-tradeoff residual metric
  2. integrity  scripts/check_scan.py invariants on every per-pilot product
                (failing products are excluded everywhere, loudly)
  3. subset     choose the stacked combine subset:
                  --stack-mode preregistered  (default) the registered greedy
                     rule (docs/PAPER_PLAN.md@30e50c2 decision 1, 2026-07-08): drop the
                     most event-constraining channel while common events grow
                     >= --growth-percent, retain >= --min-channels
                  --stack-mode max-events     documented amendment: exact
                     search over event-presence signatures for the subset
                     maximizing common events subject to the same floor
                  --stack-freq-ids <csv>      documented amendment: stack
                     exactly these channels
                The registered greedy outcome, its full drop-curve, and the
                exact best-per-k table are recorded in every mode.
  4. combine    pilot-proxy chime-combine into a fresh combined dir
  5. validate   pilot-proxy validate-products (+ JSON report)
  6. plot       pilot-proxy chime-plot (figures/, tables/)
  7. h0         H0 zero-point tables: stack quicklook AND full-depth
                per-channel mean F vs null_power_ratio over every valid frame, with the
                pilot's offset from the coarse-channel DC in fine bins
  8. tradeoff   pilot-proxy analyze-cleaning-tradeoff on the stack
  9. fulldepth  per-channel full-depth tradeoff: single-channel combine +
                analyze-cleaning-tradeoff for EVERY usable channel, merged
                into one CSV + recovered-bandwidth headline over all
                channels at full depth (the Table 3 / Figs 7-8 basis)
 10. census     pilot-proxy analyze-transmitter-census --lines-from-run
 11. bundle     small tar.gz of figures/tables/JSON/CSV summaries to carry
                off CANFAR (big NPZ products stay on /arc)

Typical use on a CANFAR notebook session (venv already activated):

    python generate_results.py --stack-mode max-events
    python generate_results.py --run-dir ~/pilot_proxy_runs/chime-pilots
    python generate_results.py --reuse-combined <results>/combined --skip census

The script never modifies scan products; every output goes to a fresh
results directory.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pilot_proxy.product_contract import (
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    null_power_ratio_of,
)

# CHIME freq_ids of the ATSC physical-channel 14-36 pilot set (CANFAR_RUNBOOK.md).
PILOT_FREQ_IDS = frozenset({
    506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675,
    690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844,
})
DETECTOR_SCHEMA = PER_PILOT_PRODUCT_SCHEMA_TOKEN
COMBINED_REQUIRED = ("chime_detector_outputs.npz", "chime_integrated_spectra.npz")
CONTROL_NAME_HINT = re.compile(r"control|h0|null|quiet|blank", re.IGNORECASE)
BUNDLE_NPZ_ALLOWLIST = {"h0_fstat_histograms.npz", "chime_frame_identity.npz",
                        "h0_fulldepth_fstat_histograms.npz",
                        "event_presence_signatures.npz"}

STAGES = ("integrity", "combine", "validate", "plot", "h0", "tradeoff",
          "fulldepth", "census", "bundle")
CHIME_COARSE_MHZ = 400.0 / 1024.0
FINE_BIN_HZ = 390625.0 / 128.0


# --------------------------------------------------------------------------
# logging / subprocess plumbing
# --------------------------------------------------------------------------

class Runner:
    def __init__(self, out_dir: Path):
        self.log_dir = out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.master = (self.log_dir / "generate_results.log").open("a", encoding="utf-8")
        self.status: dict[str, dict] = {}

    def say(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.master.write(msg + "\n")
        self.master.flush()

    def banner(self, title: str) -> None:
        self.say("\n" + "=" * 74)
        self.say(f"== {title}")
        self.say("=" * 74)

    def run(self, stage: str, cmd: list[str], *, env: dict | None = None,
            cwd: Path | None = None, quiet: bool = False) -> int:
        log_path = self.log_dir / f"{stage}.log"
        if not quiet:
            self.say(f"[{stage}] $ {' '.join(str(c) for c in cmd)}")
        t0 = time.time()
        full_env = dict(os.environ)
        full_env.setdefault("MPLBACKEND", "Agg")
        if env:
            full_env.update(env)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
            try:
                proc = subprocess.Popen(
                    [str(c) for c in cmd], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, env=full_env,
                    cwd=str(cwd) if cwd else None)
            except FileNotFoundError as exc:
                self.say(f"[{stage}] FAILED to launch: {exc}")
                log.write(f"launch failure: {exc}\n")
                return 127
            assert proc.stdout is not None
            for line in proc.stdout:
                if not quiet:
                    sys.stdout.write(line)
                    self.master.write(line)
                log.write(line)
            proc.wait()
        elapsed = time.time() - t0
        self.say(f"[{stage}] exit={proc.returncode} ({elapsed:.1f}s)")
        return proc.returncode

    def record(self, stage: str, ok: bool | None, detail: str = "") -> None:
        self.status[stage] = {"ok": ok, "detail": detail}
        flag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        self.say(f"[{stage}] {flag}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def _walk_limited(root: Path, max_depth: int):
    root = root.resolve()
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if len(p.parts) - base >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in ("staging", "figures")]
        yield p, dirnames


def find_work_dirs(root: Path, max_depth: int = 5) -> list[Path]:
    """Directories named _per_pilot that hold <freq_id>.npz products."""
    found = []
    if not root.is_dir():
        return found
    for p, dirnames in _walk_limited(root, max_depth):
        if "_per_pilot" in dirnames:
            wd = p / "_per_pilot"
            if any(c.stem.isdigit() and c.suffix == ".npz" for c in wd.iterdir()):
                found.append(wd)
    return sorted(found)


def product_paths(work_dir: Path) -> list[Path]:
    return sorted((p for p in work_dir.glob("*.npz")
                   if p.stem.isdigit()), key=lambda p: int(p.stem))


def load_product_meta(path: Path) -> dict:
    meta = {"path": path, "fid": int(path.stem), "ok": True, "why": ""}
    try:
        with np.load(str(path)) as z:
            schema = str(np.asarray(z["schema_version"]).reshape(-1)[0])
            if schema != DETECTOR_SCHEMA:
                meta.update(ok=False, why=f"schema {schema!r}")
                return meta
            meta["physical_channel"] = int(
                np.asarray(z["physical_channel"]).reshape(-1)[0])
            meta["fid"] = int(np.asarray(z["freq_id"]).reshape(-1)[0]) \
                if "freq_id" in z.files else meta["fid"]
            meta["n_frames"] = int(np.asarray(z["frame_index"]).reshape(-1).size)
            meta["n_valid"] = int(np.asarray(z["valid"]).reshape(-1).astype(bool).sum())
            if "source_event_keys" in z.files:
                meta["events"] = set(
                    np.asarray(z["source_event_keys"]).reshape(-1).astype(str).tolist())
            else:
                meta["events"] = set()
            t0 = np.asarray(z["unit_time0_ctime"]).reshape(-1) \
                if "unit_time0_ctime" in z.files else np.asarray([])
            finite = t0[np.isfinite(t0)] if t0.size else t0
            meta["t_min"] = float(finite.min()) if finite.size else None
            meta["t_max"] = float(finite.max()) if finite.size else None
    except Exception as exc:  # unreadable/corrupt product
        meta.update(ok=False, why=f"unreadable: {exc}")
    meta["mtime"] = path.stat().st_mtime
    meta["size_mb"] = path.stat().st_size / 1e6
    return meta


def pick_production(work_dirs: list[Path], metas: dict[Path, list[dict]]) -> Path | None:
    def score(wd: Path):
        ms = metas[wd]
        n_pilot = sum(1 for m in ms if m["fid"] in PILOT_FREQ_IDS)
        frames = sum(m.get("n_frames", 0) for m in ms)
        return (n_pilot, frames)
    ranked = sorted(work_dirs, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0])[0] > 0 else None


def find_control(work_dirs: list[Path], metas: dict[Path, list[dict]],
                 production_wd: Path) -> dict | None:
    """Best pilot-free control candidate across every discovered run."""
    candidates = []
    for wd in work_dirs:
        run_dir = wd.parent
        ms = metas[wd]
        all_nonpilot = all(m["fid"] not in PILOT_FREQ_IDS for m in ms)
        for m in ms:
            if m["fid"] in PILOT_FREQ_IDS or not m.get("ok"):
                continue
            has_combined = all((run_dir / f).exists() for f in COMBINED_REQUIRED)
            hint = bool(CONTROL_NAME_HINT.search(str(run_dir)))
            candidates.append({
                "run_dir": run_dir, "product": m["path"], "fid": m["fid"],
                "n_valid": m.get("n_valid", 0),
                "combined_usable": has_combined and all_nonpilot,
                "hint": hint,
            })
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c["hint"], c["combined_usable"], c["n_valid"]),
                    reverse=True)
    return candidates[0]


# --------------------------------------------------------------------------
# integrity (scripts/check_scan.py)
# --------------------------------------------------------------------------

def import_check_scan(repo_dir: Path):
    path = repo_dir / "scripts" / "check_scan.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("check_scan", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def integrity_check(check_scan, paths: list[Path], runner: Runner) -> dict[int, dict]:
    """Per-product invariants; returns {fid: {'fail': int, 'warn': int}}."""
    results: dict[int, dict] = {}
    for p in paths:
        rep = check_scan.Report()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                check_scan.check_per_pilot(p, rep)
        except Exception as exc:
            rep.fail += 1
            buf.write(f"  [FAIL] exception while checking: {exc}\n")
        text = buf.getvalue()
        with (runner.log_dir / "integrity.log").open("a", encoding="utf-8") as fh:
            fh.write(text)
        tail = [ln for ln in text.splitlines() if "FAIL" in ln or "summary:" in ln]
        for ln in tail:
            runner.say(ln)
        results[int(p.stem)] = {"fail": rep.fail, "warn": rep.warn}
    return results


# --------------------------------------------------------------------------
# stacked-subset selection
# --------------------------------------------------------------------------

def preregistered_subset(event_sets: dict[int, set], *, min_channels: int,
                         growth_percent: float) -> dict:
    """docs/PAPER_PLAN.md@30e50c2 pre-registered decision 1 (registered 2026-07-08).

    Greedy: repeatedly identify the most event-constraining channel (the one
    whose removal maximizes the remaining common-event count; ties broken by
    smaller own event count, then smaller freq_id). Accept the drop while it
    grows the common-event count by at least `growth_percent` (an empty
    starting intersection counts any non-empty growth as acceptance), subject
    to retaining at least `min_channels` channels. The greedy curve is also
    simulated past the stopping point, down to max(2, N//2) channels, for the
    appendix.
    """
    fids = sorted(event_sets)
    n_start = len(fids)

    def inter(keys) -> int:
        keys = list(keys)
        if not keys:
            return 0
        return len(set.intersection(*(event_sets[k] for k in keys)))

    def most_constraining(keys: list[int]) -> tuple[int, int]:
        best = None
        for c in keys:
            n = inter([k for k in keys if k != c])
            tie = (n, -len(event_sets[c]), -c)
            if best is None or tie > best[0]:
                best = (tie, c, n)
        return best[1], best[2]

    curve = []
    sim = list(fids)
    floor = max(2, n_start // 2)
    i_all = inter(sim)
    while len(sim) > floor:
        drop, n_after = most_constraining(sim)
        sim = [k for k in sim if k != drop]
        curve.append({"drop_fid": drop, "n_channels_after": len(sim),
                      "intersection_events_after": n_after})

    kept = list(fids)
    i_cur = i_all
    steps = []
    stop = "no candidate grew the intersection enough"
    for entry in curve:
        if len(kept) <= min_channels:
            stop = f"pre-registered floor: at least {min_channels} channels retained"
            break
        i_new = entry["intersection_events_after"]
        grew = (i_cur == 0 and i_new > 0) or (
            i_cur > 0 and i_new >= i_cur * (1.0 + growth_percent / 100.0))
        steps.append({**entry, "intersection_events_before": i_cur,
                      "growth_ok": bool(grew)})
        if not grew:
            break
        kept = [k for k in kept if k != entry["drop_fid"]]
        i_cur = i_new
    else:
        stop = "greedy curve exhausted"

    return {
        "rule": ("drop most event-constraining channel while common events grow "
                 f">= {growth_percent:g}%, retain >= {min_channels} channels"),
        "registered": "2026-07-08 (docs/PAPER_PLAN.md@30e50c2, decision 1)",
        "start_channels": fids,
        "start_intersection_events": i_all,
        "kept_channels": kept,
        "dropped_channels": [f for f in fids if f not in kept],
        "final_intersection_events": i_cur,
        "stopping_reason": stop,
        "decision_steps": steps,
        "full_drop_curve": curve,
    }


def exact_subset_search(event_sets: dict[int, set], *, min_channels: int,
                        out_npz: Path | None = None) -> dict:
    """Exact best common-event count at each allowed subset size."""
    fids = sorted(event_sets)
    n = len(fids)
    if not 1 <= min_channels <= n:
        raise ValueError(
            f"min_channels must be between 1 and {n}, got {min_channels}."
        )
    presence: dict[str, int] = {}
    for i, f in enumerate(fids):
        for e in event_sets[f]:
            presence[e] = presence.get(e, 0) | (1 << i)
    sig_counts = collections.Counter(presence.values())
    sigs = np.asarray(list(sig_counts.keys()), dtype=np.int64)
    cnts = np.asarray([sig_counts[int(s)] for s in sigs], dtype=np.int64)
    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_npz),
                            freq_ids=np.asarray(fids, dtype=np.int64),
                            signature=sigs, count=cnts)
    by_k = []
    subsets_evaluated = 0
    for k in range(n, min_channels - 1, -1):
        best_total = -1
        best_indices = None
        for indices in itertools.combinations(range(n), k):
            mask = sum(1 << index for index in indices)
            total = int(cnts[(sigs & mask) == mask].sum())
            subsets_evaluated += 1
            if total > best_total:
                best_total = total
                best_indices = indices
        assert best_indices is not None
        chans = [fids[index] for index in best_indices]
        by_k.append({"k": k, "common_events": best_total, "channels": chans,
                     "excluded": [f for f in fids if f not in chans]})
    eligible = [r for r in by_k
                if r["k"] >= min_channels and r["common_events"] > 0]
    selected = (max(eligible, key=lambda r: (r["common_events"], r["k"]))
                if eligible else None)
    return {"n_signatures": int(sigs.size), "n_events": int(cnts.sum()),
            "min_channels": min_channels,
            "subsets_evaluated": subsets_evaluated,
            "by_k": by_k, "selected": selected}


# --------------------------------------------------------------------------
# H0 zero-point tables
# --------------------------------------------------------------------------

def h0_quicklook(combined: Path, out_dir: Path, runner: Runner) -> dict:
    """Stack-based quicklook over the combined (common-event) products."""
    det = np.load(str(combined / "chime_detector_outputs.npz"))
    valid = np.asarray(det["valid"]).astype(bool)
    mask = np.asarray(det["mask"]).astype(bool)
    null_power_ratio = np.asarray(det["null_power_ratio"], np.float64).reshape(-1)
    chan = np.asarray(det["physical_channel"]).reshape(-1).astype(int)
    if "coarse_power_ratio" in det:
        fstat = np.asarray(det["coarse_power_ratio"], np.float64)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            fstat = 2.0 * np.asarray(det["p_target_u64"], np.float64) / \
                np.asarray(det["p_ref_sum_u64"], np.float64)
    stats = {}
    stats_path = combined / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    fid_by_pilot = stats.get("freq_id_by_pilot") or [None] * chan.size

    rows, hists = [], {}
    for i, ch in enumerate(chan):
        v = valid[:, i]
        f = fstat[v & np.isfinite(fstat[:, i]), i] if v.any() else np.asarray([])
        n = int(f.size)
        mean = float(f.mean()) if n else float("nan")
        sem = float(f.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        gap = abs(mean - null_power_ratio[i]) if n else float("nan")
        bound = abs(null_power_ratio[i] - 1.0) / 3.0
        mask_frac = float((mask[:, i] & v).sum() / v.sum()) if v.any() else float("nan")
        rows.append({
            "physical_channel": int(ch),
            "freq_id": fid_by_pilot[i],
            "n_valid": n,
            "mean_fstat": mean,
            "sem_fstat": sem,
            "null_power_ratio": float(null_power_ratio[i]),
            "abs_mean_minus_null_power_ratio": gap,
            "acceptance_bound_abs_null_power_ratio_minus_1_over_3": bound,
            "mean_tracks_null_power_ratio": bool(n and gap < bound),
            "mask_fraction_valid": mask_frac,
        })
        if n:
            hi = max(2.0, float(np.percentile(f, 99.9)) * 1.05)
            counts, edges = np.histogram(f, bins=2048, range=(0.0, hi))
            hists[f"ch{ch}_counts"] = counts.astype(np.int64)
            hists[f"ch{ch}_edges"] = edges.astype(np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "h0_zero_point.csv"
    import csv as _csv
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if hists:
        np.savez_compressed(str(out_dir / "h0_fstat_histograms.npz"), **hists)
    n_track = sum(1 for r in rows if r["mean_tracks_null_power_ratio"])
    return {"rows": rows, "csv": str(csv_path),
            "channels_tracking_null_power_ratio": n_track, "channels_total": len(rows)}


def h0_fulldepth_table(metas: list[dict], out_dir: Path, runner: Runner) -> dict:
    """Per-channel mean F vs null_power_ratio over EVERY valid frame (full depth), with
    the pilot's offset from the coarse-channel DC in fine bins."""
    rows, hists = [], {}
    for m in sorted(metas, key=lambda m: m.get("physical_channel", 0)):
        with np.load(str(m["path"])) as z:
            valid = np.asarray(z["valid"]).reshape(-1).astype(bool)
            rej = np.asarray(z["reject_mask"]).reshape(-1).astype(bool)
            f = np.asarray(z["coarse_power_ratio"], np.float64).reshape(-1)[valid]
            f = f[np.isfinite(f)]
            # not stored since schema v3; exact from the two weight norms
            null_power_ratio = null_power_ratio_of(z)
            pilot = float(np.asarray(z["pilot_frequency_hz"]).reshape(-1)[0])
            center = float(np.asarray(z["chime_frequency_hz"]).reshape(-1)[0])
        n = int(f.size)
        mean = float(f.mean()) if n else float("nan")
        sem = float(f.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        gap = abs(mean - null_power_ratio) if n else float("nan")
        bound = abs(null_power_ratio - 1.0) / 3.0
        ch = m.get("physical_channel")
        rows.append({
            "physical_channel": ch,
            "freq_id": m["fid"],
            "n_valid": n,
            "mean_fstat": mean,
            "sem_fstat": sem,
            "null_power_ratio": null_power_ratio,
            "abs_mean_minus_null_power_ratio": gap,
            "acceptance_bound_abs_null_power_ratio_minus_1_over_3": bound,
            "mean_tracks_null_power_ratio": bool(n and gap < bound),
            "mask_fraction_valid":
                float(rej.sum() / valid.sum()) if valid.any() else float("nan"),
            "pilot_offset_from_dc_finebins": (pilot - center) / FINE_BIN_HZ,
        })
        if n:
            hi = max(2.0, float(np.percentile(f, 99.9)) * 1.05)
            counts, edges = np.histogram(f, bins=2048, range=(0.0, hi))
            hists[f"ch{ch}_counts"] = counts.astype(np.int64)
            hists[f"ch{ch}_edges"] = edges.astype(np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    csv_path = out_dir / "h0_fulldepth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if hists:
        np.savez_compressed(str(out_dir / "h0_fulldepth_fstat_histograms.npz"),
                            **hists)
    n_track = sum(1 for r in rows if r["mean_tracks_null_power_ratio"])
    return {"rows": rows, "csv": str(csv_path),
            "channels_tracking_null_power_ratio": n_track, "channels_total": len(rows)}


# --------------------------------------------------------------------------
# full-depth per-channel cleaning tradeoff
# --------------------------------------------------------------------------

def fulldepth_tradeoff(metas: list[dict], out_dir: Path, runner: Runner, *,
                       control_dir: Path | None, survey_hours: float | None,
                       formats: str, keep_products: bool) -> dict:
    """Single-channel combine + analyze-cleaning-tradeoff per channel at full
    depth; merged CSV, recovered-bandwidth headline, and summary figures."""
    base = out_dir / "fulldepth"
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    op_by_channel: dict[int, dict] = {}
    failed: list[int] = []
    for m in sorted(metas, key=lambda m: m["fid"]):
        fid = m["fid"]
        cdir = base / f"combined_{fid}"
        tdir = base / f"tradeoff_{fid}"
        rc_c = runner.run(f"fd_combine_{fid}", [
            "pilot-proxy", "chime-combine", "--product", m["path"],
            "--output-dir", cdir], quiet=True)
        rc_t = 1
        if rc_c == 0:
            cmd = ["pilot-proxy", "analyze-cleaning-tradeoff",
                   "--run-dir", cdir, "--output-dir", tdir]
            if control_dir is not None:
                cmd += ["--control-run-dir", control_dir]
            rc_t = runner.run(f"fd_tradeoff_{fid}", cmd, quiet=True,
                              env={"PILOT_PROXY_FIGURE_FORMATS": formats})
        summ = read_json(tdir / "cleaning_tradeoff_summary.json") \
            if rc_t == 0 else None
        if summ is None:
            failed.append(fid)
            continue  # keep cdir for diagnosis
        for r in summ.get("rows", []):
            rows.append({"freq_id": fid, **r})
            if float(r.get("excess_db", -1)) == 0.0:
                op_by_channel[fid] = r
        if not keep_products:
            shutil.rmtree(cdir, ignore_errors=True)

    result: dict = {"failed_freq_ids": failed,
                    "n_channels": len(op_by_channel)}
    if rows:
        import csv as _csv
        fields = ["freq_id"] + [k for k in rows[0] if k != "freq_id"]
        merged = base / "fulldepth_cleaning_tradeoff.csv"
        with merged.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        result["merged_csv"] = str(merged)

        recovered_by_x: dict[float, float] = {}
        for r in rows:
            kf = r.get("kept_fraction")
            if kf is None or not np.isfinite(kf):
                continue
            x = float(r["excess_db"])
            recovered_by_x[x] = recovered_by_x.get(x, 0.0) + kf * CHIME_COARSE_MHZ
        n_ch = len({r["freq_id"] for r in rows})
        total_mhz = n_ch * CHIME_COARSE_MHZ
        headline = {
            "n_channels": n_ch,
            "recovered_mhz_at_null_power_ratio": recovered_by_x.get(0.0),
            "total_affected_mhz": total_mhz,
            "recovered_percent_at_null_power_ratio":
                (100.0 * recovered_by_x[0.0] / total_mhz
                 if total_mhz and 0.0 in recovered_by_x else None),
            "control_run_dir": str(control_dir) if control_dir else None,
        }
        if survey_hours is not None and 0.0 in recovered_by_x:
            headline["survey_hours"] = float(survey_hours)
            headline["recovered_mhz_hours"] = \
                recovered_by_x[0.0] * float(survey_hours)
        result["headline"] = headline
        result["recovered_mhz_by_excess_db"] = \
            {f"{k:g}": v for k, v in sorted(recovered_by_x.items())}
        result["operating_point_by_freq_id"] = op_by_channel
        (base / "fulldepth_summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = sorted(recovered_by_x)
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.plot(xs, [recovered_by_x[x] for x in xs], "-o", ms=3)
            ax.axvline(0.0, color="k", ls="--", lw=0.8, label=r"$\tau=\mu_0$")
            ax.set_xlabel("mask-threshold excess x (dB)")
            ax.set_ylabel("recovered bandwidth (MHz)")
            ax.set_title(f"Full-depth recovered bandwidth vs threshold "
                         f"({n_ch} channels)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(base / "fulldepth_recovered_bandwidth_vs_threshold.png",
                        dpi=200)
            plt.close(fig)
            if control_dir is not None and any("residual_db" in r for r in rows):
                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                by_fid: dict[int, list] = collections.defaultdict(list)
                for r in rows:
                    if "residual_db" in r:
                        by_fid[r["freq_id"]].append(
                            (r["masked_fraction"], r["residual_db"]))
                for fid, pts in sorted(by_fid.items()):
                    pts.sort()
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            "-", lw=1, label=str(fid))
                ax.set_xlabel("masked fraction (valid frames)")
                ax.set_ylabel("residual above control floor (dB)")
                ax.set_title("Full-depth cleaning operating curves")
                ax.legend(fontsize=6, ncol=3)
                fig.tight_layout()
                fig.savefig(base / "fulldepth_operating_curves.png", dpi=200)
                plt.close(fig)
        except Exception as exc:
            runner.say(f"[fulldepth] figures skipped ({exc})")
    return result


# --------------------------------------------------------------------------
# misc helpers
# --------------------------------------------------------------------------

def data_hours(combined: Path, kept_metas: list[dict]) -> dict:
    out = {}
    try:
        stats = json.loads((combined / "stats.json").read_text(encoding="utf-8"))
        spec = np.load(str(combined / "chime_integrated_spectra.npz"))
        sr = float(np.asarray(spec["sample_rate_hz"]).reshape(-1)[0])
        fss = float(stats.get("frame_size_samples", 16384))
        frame_s = fss / sr if np.isfinite(sr) and sr > 0 else float("nan")
        n_common = int(stats.get("num_frames", 0))
        out["frame_seconds"] = frame_s
        out["common_frames"] = n_common
        out["common_frame_hours_per_channel"] = (
            n_common * frame_s / 3600.0 if np.isfinite(frame_s) else None)
    except Exception as exc:
        out["error"] = str(exc)
    t_mins = [m["t_min"] for m in kept_metas if m.get("t_min") is not None]
    t_maxs = [m["t_max"] for m in kept_metas if m.get("t_max") is not None]
    if t_mins and t_maxs:
        out["survey_span_start_ctime"] = min(t_mins)
        out["survey_span_end_ctime"] = max(t_maxs)
        out["survey_span_days"] = (max(t_maxs) - min(t_mins)) / 86400.0
    return out


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_bundle(out_dir: Path, bundle_path: Path, max_file_mb: float,
                runner: Runner) -> dict:
    included, excluded = [], []
    with tarfile.open(str(bundle_path), "w:gz") as tar:
        for p in sorted(out_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(out_dir)
            size_mb = p.stat().st_size / 1e6
            if p.suffix == ".npz" and p.name not in BUNDLE_NPZ_ALLOWLIST:
                excluded.append((str(rel), size_mb))
                continue
            if size_mb > max_file_mb:
                excluded.append((str(rel), size_mb))
                continue
            tar.add(str(p), arcname=str(Path(bundle_path.stem.replace(".tar", "")) / rel))
            included.append((str(rel), size_mb))
    manifest = {
        "bundle": str(bundle_path),
        "bundle_size_mb": bundle_path.stat().st_size / 1e6,
        "included_files": len(included),
        "excluded_large_or_npz": [
            {"file": f, "size_mb": round(s, 1)} for f, s in excluded],
    }
    runner.say(f"[bundle] {bundle_path}  ({manifest['bundle_size_mb']:.1f} MB, "
               f"{len(included)} files; {len(excluded)} big/NPZ files left on /arc)")
    return manifest


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path,
                    default=Path.home() / "pilot_proxy_runs",
                    help="Where to search for run directories (default: "
                         "~/pilot_proxy_runs).")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Production run dir (contains _per_pilot/). "
                         "Default: auto-pick the discovered run with the most "
                         "pilot-set products.")
    ap.add_argument("--repo-dir", type=Path, default=Path.home() / "pilot-proxy",
                    help="pilot-proxy checkout (census CSV + check_scan.py).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Results directory (default: "
                         "<runs-root>/results_<run>_<UTC>).")
    ap.add_argument("--control-run-dir", type=Path, default=None,
                    help="Combined pilot-free control run dir for the tradeoff "
                         "residual; overrides auto-detection.")
    ap.add_argument("--no-control", action="store_true",
                    help="Skip control detection; tradeoffs run without the "
                         "residual metric.")
    ap.add_argument("--survey-hours", type=float, default=None,
                    help="Passed through for the MHz*hours headline (optional).")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated freq_ids to exclude everywhere "
                         "(e.g. a knowingly bad channel).")
    ap.add_argument("--stack-mode", choices=["preregistered", "max-events"],
                    default="preregistered",
                    help="Stacked-subset selector: the registered greedy rule "
                         "(default), or the documented amendment maximizing "
                         "common events subject to the --min-channels floor.")
    ap.add_argument("--stack-freq-ids", default=None,
                    help="Documented amendment: stack exactly these freq_ids "
                         "(comma-separated); overrides --stack-mode.")
    ap.add_argument("--min-channels", type=int, default=16,
                    help="Pre-registered retention floor (default 16).")
    ap.add_argument("--growth-percent", type=float, default=50.0,
                    help="Pre-registered growth threshold in percent "
                         "(default 50).")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write PDF figures (needs no TeX; formats env).")
    ap.add_argument("--keep-perchannel-products", action="store_true",
                    help="Keep the per-channel combined dirs the full-depth "
                         "stage creates (default: removed after a successful "
                         "per-channel tradeoff).")
    ap.add_argument("--reuse-combined", type=Path, default=None,
                    help="Existing combined dir: skip integrity/subset/combine "
                         "and run the later stages against it.")
    ap.add_argument("--skip", default="",
                    help=f"Comma-separated stages to skip ({','.join(STAGES)}).")
    ap.add_argument("--bundle-max-file-mb", type=float, default=64.0)
    ap.add_argument("--bundle-out", type=Path, default=None,
                    help="Bundle tar.gz path (default: $HOME/results_bundle_"
                         "<UTC>.tar.gz).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # absolute paths everywhere: pilot-proxy re-execs some subcommands, which
    # breaks relative --run-dir/--output-json paths.
    for name in ("runs_root", "run_dir", "repo_dir", "output_dir",
                 "control_run_dir", "reuse_combined", "bundle_out"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    bad = skip - set(STAGES)
    if bad:
        print(f"unknown --skip stage(s): {sorted(bad)}; choose from {STAGES}")
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- discover ---------------------------------------------------------
    roots = [args.runs_root]
    if args.run_dir is not None:
        roots.insert(0, args.run_dir)
    work_dirs: list[Path] = []
    for root in roots:
        for wd in find_work_dirs(Path(root)):
            if wd not in work_dirs:
                work_dirs.append(wd)
    if not work_dirs:
        print(f"no _per_pilot product directories found under {roots}; "
              "pass --run-dir explicitly")
        return 1

    metas = {wd: [load_product_meta(p) for p in product_paths(wd)]
             for wd in work_dirs}

    if args.run_dir is not None:
        rd = args.run_dir
        if rd.name == "_per_pilot":
            rd = rd.parent
        wd_hint = rd / "_per_pilot"
        production_wd = wd_hint if wd_hint in work_dirs else None
        if production_wd is None:
            inside = [wd for wd in work_dirs if str(wd).startswith(str(rd))]
            production_wd = inside[0] if inside else None
    else:
        production_wd = pick_production(work_dirs, metas)
    if production_wd is None:
        print("could not identify a production run (no pilot-set products); "
              "pass --run-dir")
        return 1
    run_dir = production_wd.parent
    run_name = run_dir.name

    out_dir = (args.output_dir or
               args.runs_root / f"results_{run_name}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(out_dir)
    runner.banner(f"PilotProxy results generator v2  ({stamp})")
    runner.say(f"production run : {run_dir}")
    runner.say(f"results dir    : {out_dir}")
    runner.say(f"repo dir       : {args.repo_dir}")
    runner.say(f"stack mode     : "
               f"{'explicit' if args.stack_freq_ids else args.stack_mode}")

    runner.say("\ndiscovered run directories:")
    for wd in work_dirs:
        ms = metas[wd]
        n_pilot = sum(1 for m in ms if m["fid"] in PILOT_FREQ_IDS)
        tag = "  <-- production" if wd == production_wd else ""
        runner.say(f"  {wd.parent}  ({len(ms)} products, {n_pilot} pilot-set, "
                   f"{sum(m.get('n_frames', 0) for m in ms)} frames){tag}")

    prod_metas = metas[production_wd]
    ok_metas = [m for m in prod_metas if m["ok"]]
    for m in prod_metas:
        if not m["ok"]:
            runner.say(f"  !! {m['path'].name}: {m['why']} (excluded)")
    med_events = float(np.median([len(m["events"]) for m in ok_metas])) \
        if ok_metas else 0.0
    runner.say("\nper-pilot inventory (production run):")
    for m in sorted(ok_metas, key=lambda m: m["fid"]):
        partial = "   <-- looks partial" if len(m["events"]) < 0.5 * med_events else ""
        runner.say(f"  freq_id {m['fid']:>4}  ch{m.get('physical_channel', '?'):>3}  "
                   f"{len(m['events']):>6} events  {m.get('n_frames', 0):>8} frames  "
                   f"{m.get('n_valid', 0):>8} valid{partial}")

    # ---- integrity --------------------------------------------------------
    # bad_fids: products unusable everywhere (unreadable, failed invariants,
    # user-excluded). Channels the subset rule drops stay OUT of this set --
    # they still feed the census/full-depth analyses.
    bad_fids: set[int] = set()
    stack_only: list[int] = []
    if args.exclude:
        bad_fids |= {int(x) for x in args.exclude.split(",") if x.strip()}
        runner.say(f"\n--exclude: {sorted(bad_fids)}")
    bad_fids |= {m["fid"] for m in prod_metas if not m["ok"]}

    if args.reuse_combined is None and "integrity" not in skip:
        runner.banner("integrity: scripts/check_scan.py per-pilot invariants")
        check_scan = import_check_scan(args.repo_dir)
        if check_scan is None:
            runner.record("integrity", None,
                          f"check_scan.py not found under {args.repo_dir}/scripts")
        else:
            res = integrity_check(
                check_scan,
                [m["path"] for m in ok_metas if m["fid"] not in bad_fids],
                runner)
            failing = {fid for fid, r in res.items() if r["fail"]}
            if failing:
                runner.say(f"  excluding failing product(s): {sorted(failing)}")
                bad_fids |= failing
            runner.record("integrity", not failing,
                          f"{len(res)} products checked, "
                          f"{len(failing)} excluded, "
                          f"{sum(r['warn'] for r in res.values())} warnings")
    else:
        runner.record("integrity", None, "skipped")

    # ---- subset + combine -------------------------------------------------
    subset = None
    if args.reuse_combined is not None:
        combined = args.reuse_combined
        runner.record("combine", None, f"reusing {combined}")
    elif "combine" in skip:
        runner.record("combine", None, "skipped")
        combined = None
    else:
        runner.banner("stacked combine subset")
        candidates = {m["fid"]: m["events"] for m in ok_metas
                      if m["fid"] in PILOT_FREQ_IDS
                      and m["fid"] not in bad_fids and m["events"]}
        no_events = [m["fid"] for m in ok_metas
                     if m["fid"] in PILOT_FREQ_IDS
                     and m["fid"] not in bad_fids and not m["events"]]
        if no_events:
            runner.say(f"  products without event metadata (excluded from "
                       f"event-keyed stack only): {sorted(no_events)}")
            stack_only.extend(no_events)
        if len(candidates) < 2:
            runner.record("combine", False,
                          f"only {len(candidates)} usable pilot products")
            combined = None
        else:
            reg = preregistered_subset(
                candidates, min_channels=args.min_channels,
                growth_percent=args.growth_percent)
            exact = exact_subset_search(
                candidates, min_channels=args.min_channels,
                out_npz=out_dir / "event_presence_signatures.npz")
            runner.say(f"  registered greedy rule: kept "
                       f"{len(reg['kept_channels'])} channels, "
                       f"{reg['final_intersection_events']} common events "
                       f"({reg['stopping_reason']})")
            runner.say("  exact best common events per subset size "
                       "(exhaustive search):")
            for r in exact["by_k"][:8]:
                runner.say(f"    k={r['k']:>2}  common={r['common_events']:>6}  "
                           f"excluded={r['excluded']}")

            explicit = None
            if args.stack_freq_ids:
                explicit = sorted({int(x) for x in
                                   args.stack_freq_ids.split(",") if x.strip()})
                unknown = [f for f in explicit if f not in candidates]
                if unknown:
                    runner.say(f"  !! --stack-freq-ids not usable here: "
                               f"{unknown} (ignored)")
                    explicit = [f for f in explicit if f in candidates]
            if explicit:
                mode, kept = "explicit-amendment", explicit
            elif args.stack_mode == "max-events" and exact["selected"]:
                mode = "max-events-amendment"
                kept = exact["selected"]["channels"]
            else:
                mode, kept = "preregistered", reg["kept_channels"]
            inter = len(set.intersection(*(candidates[f] for f in kept))) \
                if kept else 0
            if mode == "max-events-amendment" and \
                    inter != exact["selected"]["common_events"]:
                runner.say(f"  !! intersection check: {inter} != selected "
                           f"{exact['selected']['common_events']}")
            amendment_note = None
            if mode == "max-events-amendment":
                amendment_note = (
                    "Amendment to docs/PAPER_PLAN.md@30e50c2 pre-registered decision 1 "
                    "(registered 2026-07-08): stacked subset chosen to "
                    "maximize the common-event count subject to retaining at "
                    f"least {args.min_channels} channels (exhaustive subset "
                    "search over event-presence signatures). The registered "
                    "greedy rule's outcome and full drop-curve are recorded "
                    "here and reported in the appendix.")
            elif mode == "explicit-amendment":
                amendment_note = (
                    "Amendment: explicit stacked subset via --stack-freq-ids; "
                    "the registered greedy rule's outcome and full drop-curve "
                    "are recorded here and reported in the appendix.")
            subset = {
                "mode": mode,
                "amendment_note": amendment_note,
                "kept_channels": kept,
                "dropped_channels": [f for f in sorted(candidates)
                                     if f not in kept],
                "final_intersection_events": inter,
                "start_channels": sorted(candidates),
                "start_intersection_events": reg["start_intersection_events"],
                "registered_rule": reg,
                "exact_best_by_k": exact["by_k"],
            }
            runner.say(f"  selected [{mode}]: {len(kept)} channels, "
                       f"{inter} common events")
            (out_dir / "combine_subset_decision.json").write_text(
                json.dumps(subset, indent=2), encoding="utf-8")

            # explicit --product lists everywhere: chime-combine --work-dir
            # loads every .npz in the dir before --drop applies, so a corrupt
            # or excluded file would sink the whole combine.
            report_cmd = ["pilot-proxy", "chime-combine", "--report"]
            for m in sorted(ok_metas, key=lambda m: m["fid"]):
                report_cmd += ["--product", m["path"]]
            runner.run("combine_report", report_cmd, quiet=True)

            if inter == 0:
                runner.record("combine", False,
                              "selected subset shares no common events; "
                              "inspect the drop-curve and rerun with "
                              "--stack-freq-ids/--exclude")
                combined = None
            else:
                combined = out_dir / "combined"
                kept_set = set(kept)
                cmd = ["pilot-proxy", "chime-combine", "--output-dir", combined]
                for m in sorted(ok_metas, key=lambda m: m["fid"]):
                    if m["fid"] in kept_set:
                        cmd += ["--product", m["path"]]
                rc = runner.run("combine", cmd)
                runner.record("combine", rc == 0,
                              f"{len(kept)} channels -> {combined}"
                              if rc == 0 else f"exit {rc}")
                if rc != 0:
                    combined = None

    if combined is not None and not all(
            (combined / f).exists() for f in COMBINED_REQUIRED):
        runner.say(f"  !! combined dir missing {COMBINED_REQUIRED}; "
                   "downstream stages will be limited")

    # ---- validate ---------------------------------------------------------
    if combined is not None and "validate" not in skip:
        runner.banner("validate-products")
        rc = runner.run("validate", [
            "pilot-proxy", "validate-products", "--run-dir", combined,
            "--output-json", combined / "product_validation.json"])
        runner.record("validate", rc == 0, f"exit {rc}")
    else:
        runner.record("validate", None, "skipped (no combined dir)"
                      if combined is None else "skipped")

    # ---- plot -------------------------------------------------------------
    formats = "png,pdf" if args.pdf else "png"
    if combined is not None and "plot" not in skip:
        runner.banner("chime-plot")
        rc = runner.run("plot", [
            "pilot-proxy", "chime-plot", "--run-dir", combined,
            "--clean-figures"],
            env={"PILOT_PROXY_FIGURE_FORMATS": formats})
        runner.record("plot", rc == 0, f"figures/ in {combined}")
    else:
        runner.record("plot", None, "skipped")

    # ---- h0 tables ---------------------------------------------------------
    h0 = None
    h0fd = None
    if "h0" not in skip:
        runner.banner("H0 zero-point tables")
        if combined is not None:
            try:
                h0 = h0_quicklook(combined, out_dir / "h0_quicklook", runner)
            except Exception as exc:
                runner.say(f"[h0] stack quicklook failed: {exc}")
        try:
            h0fd = h0_fulldepth_table(
                [m for m in ok_metas if m["fid"] in PILOT_FREQ_IDS
                 and m["fid"] not in bad_fids],
                out_dir / "h0_quicklook", runner)
        except Exception as exc:
            runner.say(f"[h0] full-depth table failed: {exc}")
        detail = []
        if h0fd:
            detail.append(f"full-depth {h0fd['channels_tracking_null_power_ratio']}/"
                          f"{h0fd['channels_total']} track null_power_ratio")
        if h0:
            detail.append(f"stack {h0['channels_tracking_null_power_ratio']}/"
                          f"{h0['channels_total']}")
        runner.record("h0", bool(h0fd or h0), "; ".join(detail) or "no table")
    else:
        runner.record("h0", None, "skipped")

    # ---- control + stack tradeoff ------------------------------------------
    control_dir = None
    control_note = "none"
    if args.no_control:
        control_note = "disabled (--no-control)"
    elif args.control_run_dir is not None:
        control_dir = args.control_run_dir
        control_note = f"user-supplied: {control_dir}"
    elif not ({"tradeoff", "fulldepth"} <= skip):
        cand = find_control(work_dirs, metas, production_wd)
        if cand is None:
            control_note = ("no pilot-free product found under the runs root; "
                            "tradeoffs run without the residual metric")
        elif cand["combined_usable"]:
            control_dir = cand["run_dir"]
            control_note = f"auto-detected combined control: {control_dir}"
        else:
            ctrl_out = out_dir / f"control_combined_{cand['fid']}"
            rc = runner.run("control_combine", [
                "pilot-proxy", "chime-combine", "--product", cand["product"],
                "--output-dir", ctrl_out])
            if rc == 0:
                control_dir = ctrl_out
                control_note = (f"auto-detected product freq_id {cand['fid']} "
                                f"({cand['product']}), combined to {ctrl_out}")
            else:
                control_note = (f"control combine failed (exit {rc}); "
                                "tradeoffs run without the residual metric")
    runner.say(f"\ncontrol run: {control_note}")

    if combined is not None and "tradeoff" not in skip:
        runner.banner("analyze-cleaning-tradeoff (stacked products)")
        cmd = ["pilot-proxy", "analyze-cleaning-tradeoff",
               "--run-dir", combined,
               "--output-dir", out_dir / "cleaning_tradeoff"]
        if control_dir is not None:
            cmd += ["--control-run-dir", control_dir]
        if args.survey_hours is not None:
            cmd += ["--survey-hours", str(args.survey_hours)]
        rc = runner.run("tradeoff", cmd, env={
            "PILOT_PROXY_FIGURE_FORMATS": formats})
        runner.record("tradeoff", rc == 0, f"exit {rc}")
    else:
        runner.record("tradeoff", None, "skipped")

    # ---- full-depth per-channel tradeoff ------------------------------------
    fd = None
    fd_metas = [m for m in ok_metas if m["fid"] in PILOT_FREQ_IDS
                and m["fid"] not in bad_fids]
    if "fulldepth" not in skip and fd_metas:
        runner.banner("full-depth per-channel cleaning tradeoff "
                      f"({len(fd_metas)} channels)")
        fd = fulldepth_tradeoff(
            fd_metas, out_dir, runner, control_dir=control_dir,
            survey_hours=args.survey_hours, formats=formats,
            keep_products=args.keep_perchannel_products)
        head = fd.get("headline") or {}
        ok = not fd["failed_freq_ids"] and head.get("recovered_mhz_at_null_power_ratio") is not None
        detail = ""
        if head.get("recovered_mhz_at_null_power_ratio") is not None:
            detail = (f"recovered {head['recovered_mhz_at_null_power_ratio']:.2f} of "
                      f"{head['total_affected_mhz']:.2f} MHz at tau=null_power_ratio over "
                      f"{head['n_channels']} channels")
        if fd["failed_freq_ids"]:
            detail += f"; FAILED freq_ids {fd['failed_freq_ids']}"
        runner.record("fulldepth", ok, detail)
    else:
        runner.record("fulldepth", None, "skipped")

    # ---- census ------------------------------------------------------------
    # Full survey depth: every usable product feeds the case study, including
    # channels the subset rule dropped from the stack. Only bad_fids stay out
    # (via a symlink farm when needed).
    census_src = production_wd
    if bad_fids:
        farm = out_dir / "census_products"
        farm.mkdir(parents=True, exist_ok=True)
        for m in ok_metas:
            if m["fid"] in bad_fids:
                continue
            link = farm / m["path"].name
            if not link.exists():
                try:
                    link.symlink_to(m["path"].resolve())
                except OSError:
                    shutil.copy2(m["path"], link)
        census_src = farm
    census_csv = args.repo_dir / "data" / "census" / "census.csv"
    if "census" not in skip and census_csv.exists():
        runner.banner("analyze-transmitter-census (--lines-from-run)")
        rc = runner.run("census", [
            "pilot-proxy", "analyze-transmitter-census",
            "--census", census_csv,
            "--lines-from-run", census_src,
            "--output-dir", out_dir / "transmitter_census"],
            env={"PILOT_PROXY_FIGURE_FORMATS": formats})
        runner.record("census", rc == 0, f"exit {rc}")
    else:
        runner.record("census", None,
                      "skipped" if "census" in skip
                      else f"census CSV not found: {census_csv}")

    # ---- summary -----------------------------------------------------------
    runner.banner("summary")
    kept_fids = set(subset["kept_channels"]) if subset else None
    kept_metas = [m for m in ok_metas
                  if kept_fids is None or m["fid"] in kept_fids]
    hours = data_hours(combined, kept_metas) if combined is not None else {}
    tradeoff_summary = read_json(out_dir / "cleaning_tradeoff" /
                                 "cleaning_tradeoff_summary.json")
    census_summary = read_json(out_dir / "transmitter_census" / "summary.json")
    validation = read_json(combined / "product_validation.json") \
        if combined is not None else None
    mask_table = None
    if combined is not None:
        mt = combined / "tables" / "mask_summary_by_pilot.csv"
        mask_table = str(mt) if mt.exists() else None

    summary = {
        "generated_utc": stamp,
        "production_run_dir": str(run_dir),
        "results_dir": str(out_dir),
        "stage_status": runner.status,
        "per_pilot_inventory": [
            {"freq_id": m["fid"], "physical_channel": m.get("physical_channel"),
             "n_events": len(m.get("events") or ()),
             "n_frames": m.get("n_frames"),
             "n_valid": m.get("n_valid"), "ok": m["ok"], "why": m["why"]}
            for m in sorted(prod_metas, key=lambda m: m["fid"])],
        "unusable_freq_ids": sorted(bad_fids),
        "stack_only_excluded_freq_ids": sorted(set(stack_only)),
        "stack_subset": subset,
        "combined_dir": str(combined) if combined is not None else None,
        "control": control_note,
        "data_hours": hours,
        "h0_stack": ({k: v for k, v in h0.items() if k != "rows"}
                     if h0 else None),
        "h0_stack_rows": (h0 or {}).get("rows"),
        "h0_fulldepth": ({k: v for k, v in h0fd.items() if k != "rows"}
                         if h0fd else None),
        "h0_fulldepth_rows": (h0fd or {}).get("rows"),
        "stack_tradeoff_operating_point":
            (tradeoff_summary or {}).get("operating_point"),
        "stack_tradeoff_control_floor_db":
            (tradeoff_summary or {}).get("control_floor_db"),
        "fulldepth": ({k: v for k, v in fd.items()
                       if k != "operating_point_by_freq_id"} if fd else None),
        "census_headline": {
            k: (census_summary or {}).get(k)
            for k in ("spearman_rho", "spearman_ci95",
                      "spearman_rho_all_channels", "n_channels",
                      "n_channels_qualifying", "n_channels_excluded",
                      "by_class")
        } if census_summary else None,
        "mask_summary_table": mask_table,
        "validation_valid": (validation or {}).get("valid"),
    }
    (out_dir / "RESULTS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [f"PilotProxy results summary  ({stamp})",
             f"run: {run_dir}", ""]
    for stage in ("integrity", "combine", "validate", "plot", "h0",
                  "tradeoff", "fulldepth", "census"):
        st = runner.status.get(stage, {"ok": None, "detail": ""})
        flag = {True: "PASS", False: "FAIL", None: "SKIP"}[st["ok"]]
        lines.append(f"  {stage:<10} {flag:<5} {st['detail']}")
    if subset:
        lines += ["",
                  f"stack subset [{subset['mode']}]: kept "
                  f"{len(subset['kept_channels'])} channels "
                  f"{subset['kept_channels']}",
                  f"  dropped from stack: {subset['dropped_channels']}",
                  f"  common events: {subset['start_intersection_events']} "
                  f"(all) -> {subset['final_intersection_events']} (kept)"]
        if subset["mode"] == "preregistered":
            lines.append(f"  stopped because: "
                         f"{subset['registered_rule']['stopping_reason']}")
        else:
            lines.append("  amendment recorded in combine_subset_decision.json")
    if hours.get("common_frame_hours_per_channel") is not None:
        lines.append(f"  common-frame data per channel: "
                     f"{hours['common_frame_hours_per_channel']:.3f} h "
                     f"({hours['common_frames']} frames x "
                     f"{hours['frame_seconds']*1e3:.2f} ms)")
    if hours.get("survey_span_days") is not None:
        lines.append(f"  survey span: {hours['survey_span_days']:.1f} days")
    if tradeoff_summary:
        op = tradeoff_summary.get("operating_point", {})
        lines.append(f"  stack recovered bandwidth at tau=null_power_ratio: "
                     f"{op.get('recovered_mhz', float('nan')):.2f} MHz of "
                     f"{op.get('total_affected_mhz', float('nan')):.2f} MHz")
        if tradeoff_summary.get("control_floor_db") is not None:
            lines.append(f"  control floor: "
                         f"{tradeoff_summary['control_floor_db']:.2f} dB")
    if fd and fd.get("headline"):
        h = fd["headline"]
        if h.get("recovered_mhz_at_null_power_ratio") is not None:
            lines.append(
                f"  FULL-DEPTH recovered bandwidth at tau=null_power_ratio: "
                f"{h['recovered_mhz_at_null_power_ratio']:.2f} of "
                f"{h['total_affected_mhz']:.2f} MHz "
                f"({h.get('recovered_percent_at_null_power_ratio', float('nan')):.1f}%) "
                f"over {h['n_channels']} channels")
    if h0fd:
        lines.append(f"  H0 full depth: {h0fd['channels_tracking_null_power_ratio']}/"
                     f"{h0fd['channels_total']} channels track null_power_ratio within "
                     "|null_power_ratio-1|/3")
    text = "\n".join(lines) + "\n"
    (out_dir / "RESULTS_SUMMARY.txt").write_text(text, encoding="utf-8")
    runner.say("\n" + text)

    # ---- bundle ------------------------------------------------------------
    if "bundle" not in skip:
        bundle_path = (args.bundle_out or
                       Path.home() / "archive" /
                       f"results_bundle_{run_name}_{stamp}.tar.gz")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = make_bundle(out_dir, bundle_path, args.bundle_max_file_mb,
                               runner)
        (out_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        runner.say(f"\nDownload the bundle via the Jupyter file browser: "
                   f"{bundle_path}")
        runner.record("bundle", True, f"{manifest['bundle_size_mb']:.1f} MB")

    core_ok = all(runner.status.get(s, {}).get("ok") in (True, None)
                  for s in ("combine", "validate"))
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
