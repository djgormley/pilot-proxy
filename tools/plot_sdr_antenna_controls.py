#!/usr/bin/env python3
"""Plot all completed antenna-control frame ratios without fitting a reference.

Read only finalized analysis and hash-bound frame arrays. No capture, hardware
API, source correction, thermal-noise qualification or statistical gate is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODES = ("noise", "txzero", "tone")
TITLES = {"noise": "Receiver-only ambient", "txzero": "TX-active zero IQ", "tone": "Commanded steady tone"}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
FRAME_SAMPLES = 16384
OUTPUT_RATE_HZ = 390625
PLOT_NAME = "antenna-control-frame-ratios"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_completed(study):
    """Require final orchestration/analysis and recount all displayed statuses."""
    study = Path(study).resolve()
    # This receipt is created only when orchestration has completed or stopped.
    run = json.loads((study / "run-receipt.json").read_text())
    require(run["status"] in {"complete", "stopped_failed"}, "Capture sequence has no final status")
    plan = json.loads((study / "plan.json").read_text())
    summary_path = study / "analysis/summary.json"
    summary = json.loads(summary_path.read_text())
    digest = sha(study / "plan.json")
    require(digest == run["plan_sha256"] == summary["plan_sha256"] ==
            json.loads((study / "plan-digest.json").read_text())["sha256"], "Study plan identities differ")
    require(summary["schema"] == "sdr-antenna-controls-summary-v1" and
            summary["thermal_noise_qualified"] is False and summary["absolute_power_qualified"] is False and
            summary["stationarity_qualified"] is False, "Descriptive antenna-control scope differs")
    require(plan["analysis_config"] == {"input_rate_hz": 2000000, "mixer_hz": 100000.,
                                         "target_bin": 0, "sample_scale": 500.}, "Fixed analysis configuration differs")
    specifications = {item["directory"]: item for item in plan["attempts"]}
    states = {item["directory"]: item for item in run["attempts"]}
    require(len(specifications) == len(states) == 15 and set(specifications) == set(states),
            "Fixed fifteen-attempt coverage differs")
    require(summary["completed_records"] == len(summary["records"]) and
            len(summary["records"]) + len(summary["failed_records"]) + len(summary["unattempted_records"]) == 15,
            "Final completed/failed/unattempted accounting differs")
    seen = set()
    records = []
    for record in summary["records"]:
        name = record["id"]
        require(name in specifications and name not in seen, "Completed record identity differs")
        seen.add(name)
        spec = specifications[name]
        require(states[name]["status"] == "complete" and record["mode"] == spec["mode"] and
                record["triplet_index"] == spec["triplet_index"] and 0 <= record["triplet_index"] < 5,
                "Completed record mode, triplet or execution status differs")
        metadata = record["adapter_metadata"]
        require(metadata["frame_size_samples"] == FRAME_SAMPLES and metadata["frame_count"] == 47 and
                metadata["K"] == metadata["L"] == 128 and metadata["num_input_streams"] == 1,
                "Single-input upgrade frame geometry differs")
        path = study / "analysis" / name / "frames.npz"
        require(path.resolve().is_relative_to(study / "analysis"), "Frame path escapes study")
        require(sha(path) == record["frame_array_sha256"], "Frame-array identity differs: " + name)
        with np.load(path, allow_pickle=False) as arrays:
            times = arrays["frame_start_seconds"].copy()
            values = {stage: arrays[stage + "_ratio"].copy() for stage in ("natural", "packed")}
        expected_times = 81.92e-6 + np.arange(47) * FRAME_SAMPLES / OUTPUT_RATE_HZ
        require(times.shape == (47,) and np.allclose(times, expected_times, rtol=0, atol=1e-12),
                "Local frame time lattice differs")
        for stage, ratios in values.items():
            require(ratios.shape == (47,) and not np.any(ratios < 0), "Ratio shape or nonnegative support differs")
            counts = record[stage + "_ratio"]
            require(counts["total_count"] == 47 and counts["finite_count"] == np.isfinite(ratios).sum() and
                    counts["undefined_count"] == np.isnan(ratios).sum() and
                    counts["infinite_count"] == np.isinf(ratios).sum(), "Displayed ratio-status counts differ")
        quantization = metadata["sample_quantization"]
        require(quantization["component_count"] == 2 * 47 * FRAME_SAMPLES and
                0 <= quantization["saturated_component_count"] <= quantization["component_count"],
                "Clipping denominator differs")
        records.append({"summary": record, "times": times, **values})
    require(seen == {name for name, state in states.items() if state["status"] == "complete"},
            "A complete record is omitted from analysis")
    return study, plan, summary, records


def render(study, output=None):
    study, plan, summary, records = load_completed(study)
    output = study / "analysis/figures" if output is None else Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / (PLOT_NAME + extension) for extension in (".pdf", ".png")]
    require(not any(path.exists() for path in paths), "Use a new output location; plot files already exist")
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{lmodern}",
                         "font.family": "serif", "font.serif": ["Latin Modern Roman"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(12, 8.9), sharex=True)
    figure.subplots_adjust(left=.075, right=.985, top=.81, bottom=.23, hspace=.47, wspace=.24)
    for column, mode in enumerate(MODES):
        selected = sorted((record for record in records if record["summary"]["mode"] == mode),
                          key=lambda record: record["summary"]["triplet_index"])
        finite = [record[stage][np.isfinite(record[stage])] for record in selected for stage in ("natural", "packed")]
        largest = max((float(np.max(value)) for value in finite if value.size), default=0.)
        upper = 1.12 * largest if largest > 0 else 1.
        for row, stage in enumerate(("natural", "packed")):
            ax = axes[row, column]
            counts = {key: 0 for key in ("frames", "undefined", "infinite", "zero")}
            for record in selected:
                ratios, time = record[stage], record["times"]
                color = COLORS[record["summary"]["triplet_index"]]
                is_finite = np.isfinite(ratios)
                ax.plot(time, np.where(is_finite, ratios, np.nan), color=color, lw=.75,
                        marker=".", markersize=2.7, alpha=.88)
                undefined = np.isnan(ratios)
                infinite = np.isinf(ratios)
                if undefined.any():
                    ax.scatter(time[undefined], np.full(undefined.sum(), .012), marker="x", s=25,
                               color=color, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
                if infinite.any():
                    ax.scatter(time[infinite], np.full(infinite.sum(), .995), marker="^", s=27,
                               color=color, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
                counts["frames"] += ratios.size
                counts["undefined"] += int(undefined.sum())
                counts["infinite"] += int(infinite.sum())
                counts["zero"] += int(np.count_nonzero(ratios == 0))
            ax.set_xlim(-.035, 2.0)
            ax.set_ylim(0, upper)
            ax.set_xticks([0, .5, 1., 1.5, 2.])
            ax.grid(alpha=.18, lw=.6)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ax.text(.0, 1.018,
                    f"{len(selected)}/5 records; {counts['frames']} frames; "
                    f"undefined {counts['undefined']}; infinite {counts['infinite']}; zeros {counts['zero']}",
                    transform=ax.transAxes, fontsize=7.7, ha="left", va="bottom")
            if not selected:
                ax.text(.5, .5, "No complete records", transform=ax.transAxes, ha="center", color="#666666")
            elif counts["frames"] == counts["undefined"] + counts["infinite"]:
                ax.text(.5, .5, "No finite ratios", transform=ax.transAxes, ha="center", color="#666666")
            if row == 0:
                ax.set_title(TITLES[mode], fontsize=11.5, pad=29)
            else:
                clipped = sum(r["summary"]["adapter_metadata"]["sample_quantization"]["saturated_component_count"]
                              for r in selected)
                components = sum(r["summary"]["adapter_metadata"]["sample_quantization"]["component_count"]
                                 for r in selected)
                ax.set_xlabel("Time from each record's start [s]")
                ax.text(.0, -.28, f"Clipped components: {clipped:,} / {components:,}",
                        transform=ax.transAxes, fontsize=8.6, ha="left", va="top")
    axes[0, 0].set_ylabel("Natural floating\nframe ratio $Q$")
    axes[1, 0].set_ylabel("Packed integer\nframe ratio $Q$")
    handles = [Line2D([0], [0], color=color, lw=1.5, label=f"Triplet {index + 1}")
               for index, color in enumerate(COLORS)]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .916), ncol=5,
                  frameon=False, fontsize=10, handlelength=2.3, columnspacing=2.)
    figure.suptitle("Antenna-connected controls: complete frame observations", fontsize=16, y=.985)
    figure.text(.5, .946,
                f"{summary['completed_records']}/15 complete records; {len(summary['failed_records'])} failed; "
                f"{len(summary['unattempted_records'])} unattempted. All finite values shown; no fitted model or correction.",
                ha="center", fontsize=10)
    figure.text(.025, .112,
                r"One input; $16{,}384$ samples/frame; $K=L=128$; fixed sample scale $500$. Each record has its own time origin.",
                fontsize=9)
    figure.text(.025, .084,
                "Vertical limits include every finite value and are shared within each mode column. Colours preserve the frozen triplet identity.",
                fontsize=9)
    figure.text(.025, .056,
                r"If present: $\times$ at the lower border denotes undefined ratios; triangles at the upper border denote infinite ratios. Neither is a finite value.",
                fontsize=8.7)
    figure.text(.025, .028,
                "Receiver-only ambient is not qualified thermal noise. These records do not establish stationary signals, absolute RF power or physical calibration.",
                fontsize=8.7)
    stamp = datetime(2026, 9, 9, tzinfo=timezone.utc)
    figure.savefig(paths[0], metadata={"Title": "Antenna-connected frame-ratio controls", "CreationDate": stamp, "ModDate": stamp})
    figure.savefig(paths[1], dpi=170)
    plt.close(figure)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print("\n".join(str(path) for path in render(args.study, args.output)))
