"""Descriptive coarse histogram comparison against the median-matched ideal F law.

Reads the exported health/era/time-qualified full-frame Q values
(:mod:`.histogram_frames`) and compares each era's histogram with the two
reference laws of the detector adapter
(:mod:`pilot_proxy.detectors.narrowband_marker.reference_models`): the ideal
noise law F(2P, 4P) and the noncentral law matched to the era's median. No
policy is selected and no fitted model is claimed as physical calibration. A
control-band frame export is compared the same way, with no threshold drawn.

    pilot-proxy characterize histograms --frames DIR --plan FILE
        --exploratory-addendum FILE [--thresholds FILE] --output DIR
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy import stats

from pilot_proxy.config.project import default_project
from pilot_proxy.detectors.narrowband_marker.reference_models import (A, B, NULL, NULL_MEDIAN, NULL_SD, bin_masses,
                                                                      median_matched, moments)

PROBS = np.array([.001, .01, .025, .15865525393145707, .25, .5, .75, .8413447460685429, .975, .99, .999])
NOISE_LOWER = {"001": float(NULL.ppf(.001)), "01": float(NULL.ppf(.01))}
BLUE, ORANGE, DARK = "#147f95", "#b85b24", "#173347"
ZOOM_TAIL_SD = 8.0        # each reference model is held to this many of its own standard deviations
ZOOM_ETA_SD = 2.0         # the policy threshold eta is held with this many matched-model standard deviations beside it
ZOOM_MAX_GAP_SD = 120.0   # features farther apart than this, in matched-model standard deviations, get separate segments
ZOOM_MARGIN = 0.10        # margin on each side of a segment, as a fraction of its width
ZOOM_BIN_SD = 0.5         # widest zoom bin, in matched-model standard deviations


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump(path, value):
    with Path(path).open("x") as f:
        json.dump(clean(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
def measure(q):
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0 or not np.isfinite(q).all():
        raise ValueError("Require a nonempty finite frame population")
    quant = np.quantile(q, PROBS, method="linear")
    median = float(quant[5])
    law, gamma, status, error = median_matched(median)
    mq = law.ppf(PROBS)
    if not np.isfinite(mq).all() or not np.all(np.diff(mq) > 0):
        raise ValueError("Invalid theoretical quantiles")
    mean, sd = moments(gamma)
    raw = float(np.std(q, ddof=1)) if q.size > 1 else np.nan
    half = float((quant[7] - quant[3]) / 2)
    mh = float((mq[7] - mq[3]) / 2)
    iqr = float(quant[6] - quant[4])
    miqr = float(mq[6] - mq[4])
    result = {
        "n_frames": q.size, "supported_30_frames": q.size >= 30,
        "median": median, "mean": float(q.mean()), "std": raw, "minimum": float(q.min()), "maximum": float(q.max()),
        "central68_halfwidth": half, "iqr": iqr,
        "median_equivalent_gamma": gamma if status == "median_matched" else None,
        "median_equivalent_lambda": A * gamma if status == "median_matched" else None,
        "model_gamma_for_reference": gamma, "model_lambda_for_reference": A * gamma,
        "model_status": status, "median_cdf_error": error,
        "model_mean": mean, "model_std": sd, "model_central68_halfwidth": mh, "model_iqr": miqr,
        "std_ratio": raw / sd, "central68_width_ratio": half / mh, "iqr_ratio": iqr / miqr,
        "central68_upper_lower_width_ratio": float((quant[7]-median)/(median-quant[3])) if median > quant[3] else np.nan,
        "quantiles": dict(zip(PROBS, quant)), "model_quantiles": dict(zip(PROBS, mq)),
        "nonpositive_frames": int(np.sum(q <= 0)),
    }
    for prob, index in ((.001, 0), (.01, 1)):
        tag = "001" if prob == .001 else "01"
        result["lower_tail_fraction_" + tag] = float(np.mean(q < mq[index]))
        result["upper_tail_fraction_" + tag] = float(np.mean(q > mq[-index-1]))
        result["lower_tail_ratio_" + tag] = result["lower_tail_fraction_" + tag] / prob
        result["upper_tail_ratio_" + tag] = result["upper_tail_fraction_" + tag] / prob
    for tag, threshold in NOISE_LOWER.items():
        result["exploratory_noise_lower_threshold_"+tag]=threshold
        result["exploratory_noise_lower_count_"+tag]=int(np.sum(q<threshold))
        result["exploratory_noise_lower_fraction_"+tag]=float(np.mean(q<threshold))
    return clean(result)


def variance_parts(q, group):
    keys, inv, n = np.unique(group, return_inverse=True, return_counts=True)
    mean = q.mean()
    delta = q - mean
    group_delta = np.bincount(inv, weights=delta) / n
    total = float(np.mean(delta**2))
    within = float(np.mean((delta - group_delta[inv])**2))
    between = float(np.sum(n * group_delta**2) / q.size)
    if not np.isclose(total, within + between, rtol=2e-12, atol=1e-20):
        raise ValueError("Variance decomposition identity failed")
    return {
        "groups": keys.size, "frames": q.size, "total_population_variance": total,
        "within_population_variance": within, "between_population_variance": between,
        "between_variance_fraction": between/total if total else None,
        "iid_equal_variance_expected_between_fraction": (keys.size-1)/(q.size-1) if q.size > 1 else None,
        "within_group_pooled_sd": np.sqrt(within*q.size/(q.size-keys.size)) if q.size > keys.size else None,
        "minimum_frames_per_group": int(n.min()), "median_frames_per_group": float(np.median(n)),
        "maximum_frames_per_group": int(n.max()),
    }


def month_label(month):
    return f"{int(month)//12:04d}-{int(month)%12+1:02d}"


def month_date(month):
    return np.datetime64(month_label(month) + "-15")

def csv_rows(path, rows):
    columns = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (dict, list)) and key not in columns:
                columns.append(key)
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: clean(row.get(k)) for k in columns})


def style_axis(ax):
    ax.set_facecolor("#fafcfd")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#9aaab3")
    ax.tick_params(labelsize=9, colors=DARK)
    ax.grid(axis="y", color="#dce4e8", lw=.6)
    ax.set_axisbelow(True)


def overview(current, output):
    channels = np.array([r["channel"] for r in current])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True, dpi=180)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.15, hspace=.18)
    for ax in axes:
        style_axis(ax)
    for key, label, marker, color, offset in (
            ("std_ratio", "Full standard deviation / model SD", "o", BLUE, -.1),
            ("central68_width_ratio", "Central 68% width / model width", "D", ORANGE, .1)):
        axes[0].plot(channels+offset, [r[key] for r in current], marker=marker, ls="none", ms=6, color=color, label=label)
    axes[0].axhline(1, color=DARK, lw=1, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Observed / steady-model width", fontsize=11)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0,1.015), ncol=2, fontsize=9, frameon=False)
    for key, label, marker, color, offset in (
            ("upper_tail_fraction_001", "Above model's 99.9th percentile", "^", BLUE, -.1),
            ("lower_tail_fraction_001", "Below model's 0.1st percentile", "v", ORANGE, .1)):
        axes[1].plot(channels+offset, [100*r[key] for r in current], marker=marker, ls="none", ms=6, color=color, label=label)
    axes[1].axhline(.1, color=DARK, lw=1, linestyle="--")
    axes[1].set_ylabel("Frames beyond model tail (%)", fontsize=11)
    axes[1].set_xlabel("Television channel", fontsize=11)
    axes[1].set_xticks(channels)
    axes[1].legend(loc="lower left", bbox_to_anchor=(0,1.015), ncol=2, fontsize=9, frameon=False)
    fig.text(.08, .958, "CANFAR histograms versus a steady-signal reference", fontsize=21, weight="bold", color=DARK, va="top")
    fig.text(.08, .907, "Current saved era of every channel; ideal reference matched to that era's observed median", fontsize=11, color="#465e6b", va="top")
    fig.text(.08, .067, "Dashed lines: model width ratio = 1; expected fraction in each extreme tail = 0.1%.", fontsize=10, color=DARK)
    fig.text(.08, .031, "Descriptive comparison of timed, health-selected frames. Matching the median is not independent validation or measured transmitter power.", fontsize=9, color="#465e6b")
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("."+ext), facecolor="white")
    plt.close(fig)


def zoom_segments(median, sd, eta=None):
    """Edges of the zoom segments that hug the matched model, the noise model and eta.

    Each model is held to ZOOM_TAIL_SD of its own standard deviations and eta to ZOOM_ETA_SD matched-model
    standard deviations; features that lie within ZOOM_MAX_GAP_SD matched-model standard deviations of each
    other share one segment, and a segment is widened by ZOOM_MARGIN of its width on each side. A linear axis
    cannot resolve features hundreds of model widths apart, so those go to separate segments (a split axis).
    """
    features = [(median - ZOOM_TAIL_SD*sd, median + ZOOM_TAIL_SD*sd),
                (NULL_MEDIAN - ZOOM_TAIL_SD*NULL_SD, NULL_MEDIAN + ZOOM_TAIL_SD*NULL_SD)]
    if eta is not None and np.isfinite(eta):
        features.append((eta - ZOOM_ETA_SD*sd, eta + ZOOM_ETA_SD*sd))
    features.sort()
    groups = [list(features[0])]
    for lo, hi in features[1:]:
        if lo - groups[-1][1] <= ZOOM_MAX_GAP_SD*sd:
            groups[-1][1] = max(groups[-1][1], hi)
        else:
            groups.append([lo, hi])
    out = []
    for lo, hi in groups:
        pad = ZOOM_MARGIN*(hi - lo)
        lo, hi = lo - pad, hi + pad
        out.append(np.linspace(lo, hi, int(np.clip(np.ceil((hi - lo)/(ZOOM_BIN_SD*sd)), 40, 400)) + 1))
    return out


def channel_page(channel, metadata, arrays, eras, monthly, output, atlas, eta=None):
    current = next(r for r in eras if r["is_current"])
    q = arrays["Q"][arrays["current_histogram_eligible"]]
    law = NULL if current["model_gamma_for_reference"] == 0 else stats.ncf(A, B, current["model_lambda_for_reference"])
    model_label = "Median-matched steady model" if current["model_status"] == "median_matched" else "Central boundary reference (unmatched)"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), dpi=140)
    fig.subplots_adjust(left=.085, right=.975, top=.785, bottom=.12, hspace=.47, wspace=.24)
    for ax in axes.flat:
        style_axis(ax)
    # Zoom: segments that hug both reference models and eta (a split axis where they lie too far apart
    # for one linear window). Histogram normalization still uses ALL frames.
    median, sd = current["median"], current["model_std"]
    segments = []
    for edges in zoom_segments(median, sd, eta):
        counts, _ = np.histogram(q, edges)
        segments.append({"edges": edges, "counts": counts, "observed_density": counts/(q.size*np.diff(edges)),
                         "model_bin_mass": bin_masses(law, edges), "noise_bin_mass": bin_masses(NULL, edges)})
    shown = int(sum(seg["counts"].sum() for seg in segments))
    positive = np.concatenate([np.r_[seg["observed_density"][seg["observed_density"] > 0],
                                     (seg["model_bin_mass"]/np.diff(seg["edges"]))[seg["model_bin_mass"] > 0],
                                     (seg["noise_bin_mass"]/np.diff(seg["edges"]))[seg["noise_bin_mass"] > 0]] for seg in segments])
    widths = np.concatenate([np.diff(seg["edges"]) for seg in segments])
    ylim = (.4/(q.size*widths.max()), max(positive.max()*1.5, 1/(q.size*widths.min())))
    zoom_spec = axes[0,0].get_subplotspec()
    axes[0,0].remove()
    grid = zoom_spec.subgridspec(1, len(segments), wspace=.06, width_ratios=[len(seg["counts"]) for seg in segments])
    zoom_axes = [fig.add_subplot(grid[0, i]) for i in range(len(segments))]
    for i, (ax, seg) in enumerate(zip(zoom_axes, segments)):
        style_axis(ax)
        edges = seg["edges"]
        matched = seg["model_bin_mass"]/np.diff(edges)
        noise = seg["noise_bin_mass"]/np.diff(edges)
        ax.stairs(np.where(seg["observed_density"] > 0, seg["observed_density"], np.nan), edges, color=BLUE, lw=1.25)
        ax.stairs(np.where(matched > 0, matched, np.nan), edges, color=ORANGE, lw=1.4)
        if noise.max() > 0:
            ax.stairs(np.where(noise > 0, noise, np.nan), edges, color="#657888", lw=1, ls="--")
        if eta is not None and edges[0] <= eta <= edges[-1]:
            ax.axvline(eta, color=DARK, ls=":", lw=1.1)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(edges[0], edges[-1])
        ax.xaxis.set_major_locator(MaxNLocator(3))
        if i:
            ax.tick_params(which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
        if i < len(segments) - 1:
            ax.spines["right"].set_visible(False)
        # break marks between segments
        for x in ((1,) if i < len(segments) - 1 else ()) + ((0,) if i else ()):
            ax.plot([x - .015, x + .015], [-.02, .02], transform=ax.transAxes, color=DARK, lw=.8, clip_on=False)
    zoom_axes[0].set_ylabel("Probability density per Q (log)")
    zoom_axes[0].set_title(f"Zoom on the models and eta: {shown/q.size:.1%} of frames shown", loc="left", fontsize=11, weight="bold")
    fig.text(np.mean([a.get_position().x0 for a in zoom_axes[:1]] + [zoom_axes[-1].get_position().x1]), zoom_axes[0].get_position().y0 - .045,
             r"$Q=F/\mu_0$", ha="center", va="top", fontsize=10)
    # The full-range panel includes noise even when it is outside the central zoom.
    fig.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.25, label="CANFAR frames"),
        Line2D([], [], color=ORANGE, lw=1.4, label=model_label),
        Line2D([], [], color="#657888", lw=1, ls="--", label="Noise-only model"),
    ], loc="center", bbox_to_anchor=(.53,.833), ncol=3, fontsize=9, frameon=False)
    # Complete positive range with logarithmic bins; actual expected counts.
    positive_q = q[q > 0]
    full_lo = min(float(positive_q.min()), float(NULL.ppf(.0001)))
    full_hi = max(float(positive_q.max()), float(law.ppf(.9999)))
    logedges = np.geomspace(full_lo*(1-1e-12), full_hi*(1+1e-12), 121)
    full_counts, _ = np.histogram(positive_q, logedges)
    axes[0,1].stairs(np.where(full_counts > 0, full_counts, np.nan), logedges, color=BLUE, lw=1.25)
    for dist, color, linestyle in ((law, ORANGE, "-"), (NULL, "#657888", "--")):
        expected = q.size*bin_masses(dist, logedges)
        axes[0,1].stairs(np.where(expected > 0, expected, np.nan), logedges, color=color, lw=1.2, ls=linestyle)
    axes[0,1].set(xscale="log", yscale="log", xlim=(logedges[0], logedges[-1]), ylim=(.45, q.size*1.5))
    axes[0,1].set_title("Full range: logarithmic Q bins", loc="left", fontsize=11, weight="bold")
    axes[0,1].set_xlabel(r"$Q=F/\mu_0$ (log)")
    axes[0,1].set_ylabel("Frames per bin (log)")
    if np.sum(q <= 0):
        axes[0,1].text(.02,.06,f"Nonpositive Q outside log axis: {np.sum(q <= 0)}",transform=axes[0,1].transAxes,fontsize=8)
    # Fixed saved current era, month by month, with sampling support marked.
    months = [r for r in monthly if r["era_id"] == current["era_id"]]
    supported = [r for r in months if r["well_sampled"]]
    if supported:
        # Missing/unsupported months are explicit gaps, not interpolated observations.
        month_grid=np.arange(min(r["month_index"] for r in months),max(r["month_index"] for r in months)+1)
        by_month={r["month_index"]:r for r in supported}
        x = [month_date(i) for i in month_grid]
        med = np.array([by_month[i]["median"] if i in by_month else np.nan for i in month_grid])
        qlo = np.array([by_month[i]["quantiles"][str(PROBS[3])] if i in by_month else np.nan for i in month_grid])
        qhi = np.array([by_month[i]["quantiles"][str(PROBS[7])] if i in by_month else np.nan for i in month_grid])
        axes[1,0].fill_between(x,qlo,qhi,color=BLUE,alpha=.2,label="Monthly central 68% interval")
        axes[1,0].vlines(x,qlo,qhi,color=BLUE,alpha=.28,lw=.7)
        axes[1,0].plot(x,med,color=BLUE,marker=".",ms=4,lw=.8,label="Monthly median")
        axes[1,0].set_yscale("log")
        span=int(month_grid[-1]-month_grid[0]+1)
        if span <= 36:
            axes[1,0].xaxis.set_major_locator(mdates.MonthLocator(interval=3 if span<=18 else 6))
            axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            axes[1,0].xaxis.set_major_locator(mdates.YearLocator(base=2 if span>72 else 1))
            axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1,0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
        positive_lower = qlo[np.isfinite(qlo) & (qlo > 0)]
        if positive_lower.size and np.nanmax(qhi) / positive_lower.min() >= 10:
            # A label at every log minor tick overlaps across multiple decades.
            axes[1,0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axes[1,0].yaxis.set_minor_formatter(NullFormatter())
        else:
            axes[1,0].yaxis.set_minor_formatter(FuncFormatter(lambda v,pos:f"{v:g}"))
    else:
        axes[1,0].text(.5,.5,"No months meet fixed sampling support",transform=axes[1,0].transAxes,ha="center")
    axes[1,0].set_title(f"Monthly median and central 68% range\n{len(supported)}/{len(months)} months meet sampling support",loc="left",fontsize=10.5,weight="bold")
    axes[1,0].set_ylabel(r"$Q$ (log)")
    axes[1,0].set_xlabel("UTC month; support: 30 frames, 5 acquisitions, 3 days")
    for key,color,marker,label in (("std_ratio",BLUE,"o","Full SD / model SD"),("central68_width_ratio",ORANGE,"D","Central 68% width / model width")):
        rows=[r for r in eras if r["supported_30_frames"]]
        axes[1,1].plot([r["era_id"] for r in rows],[r[key] for r in rows],ls="none",marker=marker,color=color,label=label)
    axes[1,1].axhline(1,color=DARK,ls="--",lw=1)
    axes[1,1].axvspan(current["era_id"]-.25,current["era_id"]+.25,color="#dce4e8",alpha=.8)
    axes[1,1].set(yscale="log",xticks=[r["era_id"] for r in eras],xlabel="Saved historical era (shaded = current)",ylabel="Observed / model width")
    axes[1,1].set_xticklabels([str(r["era_id"])+("*" if r["model_status"]!="median_matched" else "") for r in eras])
    axes[1,1].set_title("All saved eras: width ratios\nSD: circles; central 68%: diamonds",loc="left",fontsize=10.5,weight="bold")
    fig.text(.075,.962,f"Channel {channel}: coarse histogram reference comparison",fontsize=20,weight="bold",color=DARK,va="top")
    fig.text(.075,.909,f"Current era {current['era_id']}: {current['first_month']} to {current['last_month']} ({current['state']}); {q.size:,} timed frames, {current['n_acquisitions']:,} acquisitions",fontsize=10.5,color="#465e6b",va="top")
    fig.text(.075,.868,f"Median {median:.6g}  |  SD {current['std']:.6g}  |  Central-width/model {current['central68_width_ratio']:.2f}x  |  Above model 99.9% threshold: {100*current['upper_tail_fraction_001']:.2f}%",fontsize=10.5,color=DARK,va="top")
    fig.text(.085,.055,"Model curves are integrated over the displayed bins. Zoom normalization retains all frames. Monthly shading is data spread, not uncertainty.",fontsize=8.5,color="#465e6b")
    fig.text(.085,.023,"Retrospective comparison under ideal independent Gaussian projections and clean references. * denotes an unmatched central boundary reference.",fontsize=8.5,color="#465e6b")
    for ext in ("png","pdf"):
        fig.savefig(output.with_suffix("."+ext),facecolor="white")
    atlas.savefig(fig,facecolor="white")
    plt.close(fig)
    return {"channel":channel,"zoom_eta":eta,"zoom_segments":segments,
            "zoom_excluded_frames":int(q.size-shown),"full_log_edges":logedges,"full_log_counts":full_counts,
            "full_log_nonpositive_frames":int(np.sum(q<=0)),"full_model_bin_mass":bin_masses(law,logedges)}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="pilot-proxy characterize histograms",description=__doc__)
    parser.add_argument("--frames",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--exploratory-addendum",type=Path,required=True)
    parser.add_argument("--thresholds",type=Path,help="coarse policy table (tradeoffs.csv) whose cal_q0.5 eta the zoom holds")
    args=parser.parse_args(argv)
    screened=[int(label) for label in default_project().frequency_plan.labels("screened")]
    etas={}
    if args.thresholds:
        with args.thresholds.open(newline="") as fh:
            etas={int(r["channel"]):float(r["eta"]) for r in csv.DictReader(fh) if r["policy"]=="cal_q0.5"}
        if sorted(etas)!=screened:raise ValueError(f"Thresholds must give cal_q0.5 eta for all {len(screened)} screened bands")
    plan=json.loads(args.plan.read_text())
    if (plan.get("schema")!="canfar-coarse-histogram-comparison-plan-v1"
        or plan.get("degrees_of_freedom")!=[A,B]
        or not np.array_equal(np.asarray(plan.get("quantile_probabilities",[]))[1:-1],PROBS)
        or plan.get("monthly_well_sampled")!="atleast30frames,5acquisitions,3UTCdays; retain other months with explicit flag"):
        raise ValueError("Comparison plan differs from resolved model or support settings")
    addendum=json.loads(args.exploratory_addendum.read_text())
    if addendum.get("schema")!="coarse-histogram-exploratory-addendum-v1":raise ValueError("Exploratory addendum required")
    manifest_path=args.frames/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="coarse-histogram-frames-manifest-v1" or manifest.get("passed") is not True:
        raise ValueError("Successful authenticated frame export is required")
    control=manifest.get("role")=="control"
    if control and args.thresholds:raise ValueError("A control band takes no threshold")
    channels=[int(item["channel"]) for item in manifest["channels"]] if control else screened
    for name,digest in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(args.frames/name)!=digest:
            raise ValueError(f"Frame export artifact identity differs: {name}")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"channels").mkdir()
    inputs={str(Path(__file__).absolute()):sha(__file__),str(args.plan.absolute()):sha(args.plan),str(manifest_path.absolute()):sha(manifest_path),str(args.exploratory_addendum.absolute()):sha(args.exploratory_addendum)}
    if args.thresholds:inputs[str(args.thresholds.absolute())]=sha(args.thresholds)
    all_eras=[];all_months=[];all_acquisitions=[];channel_records=[];sources={}
    for channel in channels:
        file=args.frames/f"ch{channel:02d}.npz"
        metafile=file.with_suffix(".json")
        metadata=json.loads(metafile.read_text())
        for path in (file,metafile):inputs[str(path.absolute())]=sha(path)
        with np.load(file,allow_pickle=False) as z:
            arrays={k:z[k] for k in ("Q","histogram_eligible","era_id","current_histogram_eligible","frame_time","frame_month","acquisition_id")}
        if metadata["channel"]!=channel:raise ValueError("Channel identity differs")
        if control and metadata.get("role")!="control":raise ValueError("A control run holds only control-band exports")
        if not np.array_equal(arrays["current_histogram_eligible"], arrays["histogram_eligible"] & (arrays["era_id"]==int(metadata["current_era"]))):
            raise ValueError("Current mask differs from exact saved era membership")
        q_all=arrays["Q"]
        if np.any(arrays["histogram_eligible"] & ~np.isfinite(q_all)):raise ValueError("Eligible nonfinite Q")
        if int(arrays["current_histogram_eligible"].sum())!=metadata["counts"]["current_histogram_eligible"]:raise ValueError("Current count differs")
        eras=[];monthly=[];acqs=[]
        for era in metadata["era_definitions"]:
            eid=int(era["era"])
            mask=arrays["histogram_eligible"] & (arrays["era_id"]==eid)
            q=q_all[mask]
            if not q.size:
                raise ValueError(f"No timed finite frames in channel{channel} era{eid}; retain explicitly before continuing")
            row=measure(q)
            row.update(channel=channel,freq_id=metadata["freq_id"],era_id=eid,
                       is_current=eid==int(metadata["current_era"]),first_month=era["first_month"],last_month=era["last_month"],state=era["state"])
            acq=arrays["acquisition_id"][mask];months=arrays["frame_month"][mask];times=arrays["frame_time"][mask]
            row["n_acquisitions"]=int(np.unique(acq).size)
            row["n_utc_days"]=int(np.unique(np.floor(times/86400)).size)
            row["acquisition_variance"]=clean(variance_parts(q,acq))
            row["month_variance"]=clean(variance_parts(q,months))
            row["between_acquisition_variance_fraction"]=row["acquisition_variance"]["between_variance_fraction"]
            row["between_month_variance_fraction"]=row["month_variance"]["between_variance_fraction"]
            row["within_acquisition_pooled_sd"]=row["acquisition_variance"]["within_group_pooled_sd"]
            row["within_acquisition_sd_ratio"]=row["within_acquisition_pooled_sd"]/row["model_std"] if row["within_acquisition_pooled_sd"] is not None else None
            for month in np.unique(months):
                take=months==month
                mr=measure(q[take]);nacq=np.unique(acq[take]).size;nday=np.unique(np.floor(times[take]/86400)).size
                mr.update(channel=channel,era_id=eid,is_current=row["is_current"],month_index=int(month),month=month_label(month),n_acquisitions=int(nacq),n_utc_days=int(nday),well_sampled=q[take].size>=30 and nacq>=5 and nday>=3)
                monthly.append(mr)
            order=np.argsort(acq,kind="stable")
            ids,starts,counts=np.unique(acq[order],return_index=True,return_counts=True)
            for aid,start,count in zip(ids,starts,counts):
                take=order[start:start+count];v=q[take]
                acqs.append({"channel":channel,"era_id":eid,"is_current":row["is_current"],"acquisition_id":int(aid),"frames":int(v.size),"first_time":float(times[take].min()),"last_time":float(times[take].max()),"median":float(np.median(v)),"mean":float(v.mean()),"std":float(v.std(ddof=1)) if v.size>1 else None})
            supported=[m for m in monthly if m["era_id"]==eid and m["well_sampled"]]
            row["well_sampled_months"]=len(supported)
            row["monthly_median_min"]=min((m["median"] for m in supported),default=None)
            row["monthly_median_max"]=max((m["median"] for m in supported),default=None)
            row["monthly_central_width_ratio_median"]=float(np.median([m["central68_width_ratio"] for m in supported])) if supported else None
            row["monthly_central_width_ratio_min"]=min((m["central68_width_ratio"] for m in supported),default=None)
            eras.append(row)
        current=next(r for r in eras if r["is_current"])
        if current["n_frames"]!=int(arrays["current_histogram_eligible"].sum()):raise ValueError("Current era and current histogram membership differ")
        sources[channel]=(metadata,arrays,eras,monthly)
        all_eras.extend(eras);all_months.extend(monthly);all_acquisitions.extend(acqs)
        channel_records.append({"channel":channel,"counts":metadata["counts"],"current":current})
        dump(args.output/"channels"/f"ch{channel:02d}-statistics.json",{"metadata":metadata,"eras":eras,"months":monthly})
        print(f"Channel {channel}: {len(eras)} eras, current N={current['n_frames']}, median={current['median']:.6g}, SD/model={current['std_ratio']:.2f}, central68/model={current['central68_width_ratio']:.2f}",flush=True)
    current=[r["current"] for r in channel_records]
    csv_rows(args.output/"current-eras.csv",current)
    csv_rows(args.output/"all-eras.csv",all_eras)
    csv_rows(args.output/"monthly.csv",all_months)
    csv_rows(args.output/"acquisitions.csv",all_acquisitions)
    overview(current,args.output/"current-era-comparison")
    with PdfPages(args.output/"channel-atlas.pdf") as atlas:
        for channel,(metadata,arrays,eras,monthly) in sources.items():
            bins=channel_page(channel,metadata,arrays,eras,monthly,args.output/"channels"/f"ch{channel:02d}",atlas,eta=etas.get(channel))
            dump(args.output/"channels"/f"ch{channel:02d}-histogram-bins.json",bins)
            print(f"Rendered channel {channel}",flush=True)
    report={"schema":"canfar-coarse-histogram-comparison-v5","created_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"Retrospective descriptive comparison; no independent significance, physical-power calibration, verified state or policy certification", "degrees_of_freedom":[A,B],"central_median":NULL_MEDIAN,
            "channels":channel_records,"eras":all_eras,"monthly":all_months,"exploratory_addendum":addendum,
            "totals":{"channels":len(current),"eras":len(all_eras),"monthly_groups":len(all_months),"acquisition_era_groups":len(all_acquisitions),"current_frames":sum(r["n_frames"] for r in current),"all_era_frames":sum(r["n_frames"] for r in all_eras),"histogram_eligible_all":sum(r["counts"]["histogram_eligible"] for r in channel_records),"selected_untimed":sum(r["counts"]["untimed_selected"] for r in channel_records),"timed_unassigned_era":sum(r["counts"]["unassigned_histogram_eligible"] for r in channel_records),"health_selected":sum(r["counts"]["selected"] for r in channel_records)},
            "resolved_settings":{"quantile_probabilities":PROBS.tolist(),"monthly_min_frames":30,"monthly_min_acquisitions":5,"monthly_min_utc_days":3,"sample_std_ddof":1,"fit_scope":"same-sample descriptive median match","unmatched_equivalent_parameters":"null; central boundary model parameters retained separately"},
            "inputs":inputs,"runtime":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    if control:report["role"]="control"
    report["artifacts"]={str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    dump(args.output/"report.json",report)
    print(json.dumps(report["totals"],indent=2))


__all__ = ["PROBS", "channel_page", "main", "measure", "variance_parts", "zoom_segments"]
