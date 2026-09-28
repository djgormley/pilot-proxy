#!/usr/bin/env python3
"""Plot exact noncentral-F median and standard deviation versus projected power.

Current CANFAR coarse pooling: P=2048*128 independent complex projections.
The x coordinate is received projected signal/noise power, not transmitter watts.
This is an ideal analytical model; no hardware or new simulation is used.
"""
from __future__ import annotations

import argparse
import csv
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
from scipy import stats

from pilot_proxy.detectors.narrowband_marker.reference_models import DF1, DF2, P, power_response as response


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model():
    gamma = np.r_[0.0, np.logspace(-4, 1, 101)]
    curve = response(gamma)
    scipy_std = np.empty_like(gamma)
    scipy_std[0] = stats.f.std(DF1, DF2)
    scipy_std[1:] = stats.ncf.std(DF1, DF2, DF1 * gamma[1:])
    cdf = np.empty_like(gamma)
    cdf[0] = stats.f.cdf(curve["median"][0], DF1, DF2)
    cdf[1:] = stats.ncf.cdf(curve["median"][1:], DF1, DF2, DF1 * gamma[1:])
    # Independently form E[Q^2] using E[A^2] and E[D^-2], A/D = Q/2.
    # A = sum target powers; D = sum two reference branch powers.
    ea2 = P * (1 + 2 * gamma) + (P * (1 + gamma))**2
    eq2 = 4 * ea2 / ((2 * P - 1) * (2 * P - 2))
    independent_variance = eq2 - curve["mean"]**2
    checks = {
        "all_medians_finite": bool(np.isfinite(curve["median"]).all()),
        "median_strictly_increases": bool(np.all(np.diff(curve["median"]) > 0)),
        "absolute_std_strictly_increases": bool(np.all(np.diff(curve["std"]) > 0)),
        "exact_variance_matches_scipy": bool(np.allclose(curve["std"], scipy_std, rtol=2e-13, atol=0)),
        "exact_variance_matches_independent_moments": bool(np.allclose(curve["std"]**2, independent_variance, rtol=2e-9, atol=0)),
        "median_cdf_within_1e_minus7": bool(np.max(np.abs(cdf - 0.5)) < 1e-7),
        "central_limit_matches_previous_reference": bool(abs(curve["std"][0] - 0.0023920874311778497) < 1e-15),
    }
    if not all(checks.values()):
        raise ValueError(f"Model validation failed: {checks}")
    return {
        "checks": checks,
        "largest_median_cdf_error": float(np.max(np.abs(cdf - 0.5))),
        "max_std_relative_error_vs_scipy": float(np.max(np.abs(curve["std"] / scipy_std - 1))),
        "max_std_relative_error_large_P_approximation": float(np.max(np.abs(curve["std_large_P_approximation"] / curve["std"] - 1))),
        "max_median_absolute_error_large_P_approximation": float(np.max(np.abs(curve["median"] - (1 + gamma)))),
        "verified_gamma_range": [0.0, 10.0],
    }


def draw(gamma, output, db_axis=False):
    values = response(gamma)
    x = 10 * np.log10(gamma) if db_axis else 100 * gamma
    earlier = response([0.02])
    null = response([0.0])
    xp = 10 * np.log10(0.02) if db_axis else 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=180)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.27, top=0.76, wspace=0.27)
    fig.patch.set_facecolor("white")
    colors = ["#137c91", "#a64d19"]
    for ax, key, title, color in zip(axes, ("median", "std"),
                                     ("Median", "Standard deviation"), colors):
        ax.set_facecolor("#fbfcfd")
        ax.plot(x, values[key], color=color, lw=2.5)
        ax.axhline(null[key][0], color="#7c8f99", linestyle=(0, (3, 3)), lw=1)
        ax.axvline(xp, color="#8497a2", linestyle=(0, (2, 3)), lw=0.9)
        ax.scatter([xp], earlier[key], s=42, facecolor=color, edgecolor="white", lw=0.8, zorder=5)
        ax.set_title(title, loc="left", fontsize=13, weight="bold", color="#172d40", pad=13)
        ax.set_ylabel(r"Median of $Q$" if key == "median" else r"Standard deviation of $Q$", fontsize=10)
        ax.set_xlim((x[0], x[-1]) if db_axis else (-1.5, 101.5))
        if not db_axis:
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_ylim((0.95, 2.08) if key == "median" else (0.0022, 0.0046))
        ax.tick_params(labelsize=10, colors="#273d4a")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(color="#dce4e9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#9aa7b0")
        note = f"Earlier 2% example\n{title}: {earlier[key][0]:.6f}"
        ax.text(0.06, 0.91, note, transform=ax.transAxes, va="top", fontsize=10,
                color="#273d4a", bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#dce4e9"})
    fig.text(0.08, 0.945, "Signal power sets both centre and width", fontsize=21,
             weight="bold", color="#172d40", va="top")
    fig.text(0.08, 0.865,
             r"CANFAR coarse model: $Q\sim F_{\rm nc}(524288,1048576;\lambda)$,  $\lambda=524288\,\gamma$",
             fontsize=11.5, color="#465c69", va="top")
    xlabel = (r"Received projected signal-to-noise power, $10\log_{10}\gamma$ (dB)"
              if db_axis else "Received signal power (% of target-branch noise power)")
    fig.text(0.525, 0.17, xlabel, ha="center", fontsize=11, color="#273d4a")
    fig.text(0.08, 0.092,
             "Exact noncentral-F median and standard deviation. Dashed horizontal lines: noise-only limits.",
             fontsize=9.5, color="#465c69")
    fig.text(0.08, 0.043,
             "Fixed noise and linear response; ideal independent projections and signal-free references. Transmitter watts are uncalibrated.",
             fontsize=9, color="#465c69")
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), facecolor="white")
    plt.close(fig)


def write_csv(path, data):
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(data)
        writer.writerows(zip(*data.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.reference_plan.read_text())
    if (plan.get("projections_per_frame") != P or plan.get("receivers_count_model") != 2048
            or plan.get("windows_per_receiver_model") != 128):
        raise ValueError("Reference plan does not match current coarse degrees of freedom")
    validation = check_model()
    args.output.mkdir(parents=True, exist_ok=False)
    linear_gamma = np.linspace(0, 1, 501)
    broad_gamma = np.logspace(-4, 1, 501)
    draw(linear_gamma, args.output / "power-response")
    draw(broad_gamma, args.output / "power-response-db", db_axis=True)
    write_csv(args.output / "power-response.csv", response(linear_gamma))
    write_csv(args.output / "power-response-db.csv", response(broad_gamma))
    checkpoints = response([0.0, 0.001, 0.01, 0.02, 0.1, 1.0, 10.0])
    rows = [{k: float(v[i]) for k, v in checkpoints.items()} for i in range(len(checkpoints["gamma"]))]
    report = {
        "schema": "canfar-power-response-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Ideal analytical signal-power response of the CANFAR coarse statistic",
        "hardware_used": False, "new_simulation_used": False,
        "pooled_projections_per_frame": P, "degrees_of_freedom": [DF1, DF2],
        "gamma_definition": "Sum of deterministic target projected signal powers divided by total target projected noise power",
        "noncentrality": "lambda = degrees_of_freedom_numerator * gamma",
        "median_method": "Numerical inverse CDF at 0.5; central F used explicitly at gamma=0",
        "variance_exact": "2*(df2/df1)^2*((df1+lambda)^2+(df1+2*lambda)*(df2-2))/((df2-2)^2*(df2-4))",
        "median_large_P_approximation": "1 + gamma",
        "std_large_P_approximation": "sqrt((gamma^2+6*gamma+3)/(2*P))",
        "transmitter_power_mapping": "For fixed linear response z_p=sqrt(Ptx)*h_p+n_p and common noise variance v: gamma=Ptx*sum(abs(h_p)^2)/(P*v); coefficient is not measured here",
        "assumptions": ["Independent proper complex Gaussian noise contributions", "Equal noise variances and independent target/reference branches", "Fixed deterministic received projected signal", "Signal-free references", "Constant pooling, noise level and linear receiver response"],
        "checkpoints": rows, "validation": validation,
        "inputs": {str(Path(__file__).absolute()): sha(__file__), str(args.reference_plan.absolute()): sha(args.reference_plan)},
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "sources": ["https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/dist_ref/dists/nc_f_dist.html", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html"],
    }
    report["artifacts"] = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"validation": validation, "checkpoints": rows}, indent=2))


if __name__ == "__main__":
    main()
