#!/usr/bin/env python3
"""Two ideal CANFAR coarse density references from actual GNU projection streams.

No SDR data or physical calibration is inferred. All histogram entries are full
frame ratios after P=2048*128 branch-power contributions have been accumulated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate, stats

from pilot_proxy.detectors.narrowband_marker.reference_models import P, steady_law as law

N = 1024
GAMMA = 0.02
EDGES = np.linspace(0.988, 1.032, 89)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frames(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float64) for k in ("A", "B", "C", "Q")}
    for name, values in arrays.items():
        if values.shape != (N,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{path}: invalid {name}; require {N} positive finite full-frame entries")
    reconstructed = 2 * arrays["A"] / (arrays["B"] + arrays["C"])
    if not np.allclose(arrays["Q"], reconstructed, rtol=2e-15, atol=0):
        raise ValueError("Q must be the ratio of branch-power sums, not the mean of ratios")
    return arrays


def summarize(arrays, gamma):
    q = arrays["Q"]
    distribution = law(gamma)
    mean, var = map(float, distribution.stats(moments="mv"))
    sd = np.sqrt(var)
    counts, _ = np.histogram(q, bins=EDGES)
    density = counts / (q.size * np.diff(EDGES))
    # Use the full sample count, never renormalize just the plotted subset.
    expected = q.size * np.diff(distribution.cdf(EDGES))
    bounds = distribution.ppf([1e-10, 1 - 1e-10])
    area, error = integrate.quad(distribution.pdf, *bounds, epsabs=1e-9)
    if abs(area - (1 - 2e-10)) > 5e-7:
        raise ValueError("Numerical theoretical PDF does not integrate correctly")
    ks = stats.kstest(q, distribution.cdf)
    mean_z = float((q.mean() - mean) / (sd / np.sqrt(N)))
    branch_expected = [P * (1 + gamma), P, P]
    branch_se = [np.sqrt(P * (1 + 2 * gamma) / N), np.sqrt(P / N), np.sqrt(P / N)]
    branch_z = {
        k: float((arrays[k].mean() - e) / se)
        for k, e, se in zip(("A", "B", "C"), branch_expected, branch_se)
    }
    return {
        "frames": N, "powers_per_branch_per_frame": P,
        "gamma_projected_signal_to_noise_power": gamma,
        "total_noncentrality": 2 * P * gamma,
        "degrees_of_freedom": [2 * P, 4 * P],
        "theory_mean": mean, "theory_sd": float(sd),
        "gnu_mean": float(q.mean()), "gnu_sd": float(q.std(ddof=1)),
        "mean_standard_errors_from_theory": mean_z,
        "branch_mean_standard_errors_from_theory": branch_z,
        "ks_distance": float(ks.statistic), "ks_pvalue_diagnostic_only": float(ks.pvalue),
        "mean_within_six_standard_errors": abs(mean_z) < 6,
        "branch_means_within_six_standard_errors": all(abs(z) < 6 for z in branch_z.values()),
        "pdf_integral_between_theory_1e_minus10_quantiles": area,
        "pdf_integral_error_estimate": error,
        "histogram_edges": EDGES.tolist(), "histogram_counts": counts.tolist(),
        "histogram_density": density.tolist(),
        "expected_counts_per_bin_at_this_frame_count": expected.tolist(),
        "outside_displayed_bins": int(N - counts.sum()),
        "displayed_empirical_probability_mass": float(np.sum(density * np.diff(EDGES))),
        "displayed_theoretical_probability_mass": float(distribution.cdf(EDGES[-1]) - distribution.cdf(EDGES[0])),
    }


def draw(arrays, gamma, summary, output):
    distribution = law(gamma)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.78, bottom=0.235)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9aa7b0")
    color = "#137c91" if gamma == 0 else "#a64d19"
    ax.stairs(summary["histogram_density"], EDGES, fill=True,
              facecolor=color, alpha=0.16, linewidth=0)
    histogram = ax.stairs(summary["histogram_density"], EDGES, color=color,
                         linewidth=1.25, label=f"GNU Radio simulation ({N:,} frames)")
    grid = np.linspace(EDGES[0], EDGES[-1], 2400)
    theory, = ax.plot(grid, distribution.pdf(grid), color="#172d40", linewidth=2.1,
                     label="Theoretical PDF")
    ax.set(xlim=(EDGES[0], EDGES[-1]), ylim=(0, 210),
           xlabel=r"Per-frame statistic $Q = F/\mu_0$", ylabel="Probability density")
    ax.set_xticks([0.99, 1.00, 1.01, 1.02, 1.03])
    ax.tick_params(labelsize=10, colors="#273d4a")
    ax.grid(axis="y", color="#d9e1e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[theory, histogram], loc="upper right" if gamma == 0 else "upper left",
              frameon=False, fontsize=10)
    title = "Noise only" if gamma == 0 else "Steady signal + noise"
    subtitle = ("No transmitter signal in the target or reference branches"
                if gamma == 0 else "Illustrative fixed signal: 2% of target-branch noise power")
    fig.text(0.105, 0.94, title, fontsize=20, weight="bold", color="#172d40", va="top")
    fig.text(0.105, 0.872, subtitle, fontsize=11, color="#465c69", va="top")
    fig.text(0.105, 0.127,
             f"Expected centre {summary['theory_mean']:.6f}   |   Standard deviation {summary['theory_sd']:.6f}",
             fontsize=10, color="#273d4a")
    fig.text(0.105, 0.071,
             "CANFAR pooling: 2,048 inputs x 128 windows per frame. One histogram entry per frame.",
             fontsize=8.8, color="#465c69")
    fig.text(0.105, 0.035,
             "Ideal Gaussian projection model; independent branches and signal-free, equal-variance references.",
             fontsize=8.8, color="#465c69")
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.input / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "canfar-coarse-gnuradio-plan-v2"
            or plan.get("frames_per_state") != N
            or plan.get("projections_per_frame") != P
            or plan.get("noise_source_amplitude") != 1.0
            or plan.get("complex_noise_variance_ideal") != 1.0):
        raise ValueError("GNU generation plan does not match the plotted model")
    if sha(plan_path) != json.loads((args.input / "plan-digest.json").read_text())["sha256"]:
        raise ValueError("GNU generation plan digest differs")
    evidence = {str(plan_path.absolute()): sha(plan_path)}
    seeds = []
    for state, gamma in (("noise", 0.0), ("steady", GAMMA)):
        if plan["states"][state]["gamma"] != gamma:
            raise ValueError("GNU signal level differs from theoretical model")
        seeds.extend(plan["states"][state]["seeds"])
        receipt_path = args.input / (state + "-receipt.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "canfar-coarse-gnuradio-receipt-v2"
                or receipt.get("success") is not True
                or receipt.get("actual_gnuradio_generation") is not True
                or receipt.get("hardware_attempted") is not False
                or receipt.get("plan_sha256") != sha(plan_path)
                or receipt.get("frames") != N or receipt.get("gamma") != gamma
                or receipt.get("projections_per_frame") != P):
            raise ValueError("Successful matching GNU generation receipt required")
        evidence[str(receipt_path.absolute())] = sha(receipt_path)
        for name, digest in receipt["artifacts"].items():
            if Path(name).name != name or sha(args.input / name) != digest:
                raise ValueError("GNU simulation artifact identity differs")
            evidence[str((args.input / name).absolute())] = digest
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("Require distinct declared seeds for all branches and states")
    for name, digest in plan["inputs"].items():
        if sha(name) != digest:
            raise ValueError("GNU generation source or runtime identity differs")
    evidence.update(plan["inputs"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "canfar-coarse-reference-figures-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Ideal equal-reference independent Gaussian projection model, using CANFAR coarse aggregation",
        "not_physical_sdr_data": True, "not_packed_input_end_to_end_simulation": True,
        "histogram_meaning": "Count full-frame Q values; each Q is formed after summing branch powers within its frame",
        "signal_level": "Illustrative fixed received projected level; not fitted to CANFAR or SDR and not measured shelf SNR",
        "normalization": "Counts divided by total N and linear Q bin width; outside-range mass retained",
        "inputs": {**evidence, str(Path(__file__).absolute()): sha(__file__)},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__},
        "sources": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html",
                    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html"],
        "states": {},
    }
    for state, gamma in (("noise", 0.0), ("steady", GAMMA)):
        path = args.input / (state + ".npz")
        arrays = load_frames(path)
        report["inputs"][str(path.absolute())] = sha(path)
        summary = summarize(arrays, gamma)
        report["states"][state] = summary
        draw(arrays, gamma, summary, args.output / (state + "-density"))
    report["artifacts"] = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({state: {k: v for k, v in info.items() if not isinstance(v, (list, dict))}
                      for state, info in report["states"].items()}, indent=2))


if __name__ == "__main__":
    main()
