#!/usr/bin/env python3
"""Single-row floating reference-ratio comparison; never opens radio hardware.

Fixed geometry:2MHz complex IQ,K128,target+100kHz,references +/-2 tap-grid bins.
The central/noncentral F curves require independent proper complex Gaussian
projections with equal noise variance and central reference branches. These are
conditional comparison models, not assumed properties of an SDR measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import welch
from scipy.stats import f, ncf

K = 128
FS = 2_000_000.
TARGET = 100_000.
OFFSET_BINS = 2
SAMPLES = 4_000_000
MODES = ("noise", "txzero", "tone")
CONFIG = {"sample_rate_hz": FS, "window_samples": K, "target_hz": TARGET,
          "reference_offset_bins": OFFSET_BINS, "samples_per_capture": SAMPLES,
          "projection": "z=x@conj(exp(+2j*pi*f*n/fs)).T; terms target,lower,upper",
          "ratio": "2*abs(z0)**2/(abs(zlower)**2+abs(zupper)**2)",
          "arithmetic": "complex128 projection of unmodified complex64 IQ, no quantization",
          "row_stride_samples": K, "rows_pooled_before_ratio": 1,
          "strong_calibration_baseline": "txzero", "calibration_triplets": 2,
          "comparison_triplets": 3}
SCOPE = ("Descriptive single-RF-path comparison using floating single-row projections. "
         "No physical power calibration, packed-kernel/full-array validation, false-alarm "
         "certification, scientific acceptance or prospective observing policy.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def frequencies():
    delta = OFFSET_BINS * FS / K
    return np.array([TARGET, TARGET-delta, TARGET+delta])


def weights():
    return np.exp(2j*np.pi*frequencies()[:, None]*np.arange(K)[None, :]/FS)


def geometry():
    w = weights()
    # Cov(z_r,conj(z_s))/sigma2 = sum conj(w_r)*w_s.
    gram = np.conj(w) @ w.T
    norm = np.real(np.diag(gram))
    corr = gram/np.sqrt(norm[:, None]*norm[None, :])
    return {"frequencies_hz": frequencies().tolist(), "weight_norm_squared": norm.tolist(),
            "gram_real": gram.real.tolist(), "gram_imag": gram.imag.tolist(),
            "normalized_gram_max_off_diagonal_magnitude": float(np.max(np.abs(corr-np.eye(3)))),
            "null_ratio_of_expected_powers": float(2*norm[0]/(norm[1]+norm[2])),
            "central_F_mean": 2., "central_F_variance": "infinite",
            "central_F_median": float(f.median(2,4)),
            "note": "Ratio of expected powers is1; it is not E[the single-row ratio], which is2 under the ideal null."}


def projections(iq):
    iq = np.asarray(iq)
    if iq.ndim != 1 or iq.size == 0 or iq.size % K:
        raise ValueError("IQ must be nonempty and contain an exact multiple of128 samples")
    if not np.isfinite(iq).all():
        raise ValueError("nonfinite IQ must not enter comparisons")
    return np.asarray(iq.reshape(-1, K), dtype=np.complex128) @ np.conj(weights()).T


def ratios(z):
    p = np.abs(z)**2
    den = p[:, 1]+p[:, 2]
    valid = den > 0
    values = np.full(den.shape, np.nan)
    values[valid] = 2*p[valid, 0]/den[valid]
    values[(den == 0) & (p[:, 0] > 0)] = np.inf
    counts = {"positive_denominator": int(valid.sum()),
              "infinite_ratio_positive_numerator_zero_denominator": int(((den == 0)&(p[:, 0] > 0)).sum()),
              "undefined_both_zero": int(((den == 0)&(p[:, 0] == 0)).sum())}
    return values, counts


def simulation(rows, lam, seed):
    """Independent unnormalized proper white Gaussian samples, then the same projection.

    E|noise[k]|^2=1. Aligned deterministic amplitude sqrt(lambda/(2K))
    gives target noncentrality2*(A*K)^2/K=lambda. No projection, exact-energy
    normalization or conditioning is applied to the noise realization. The raw
    IQ is rounded to complex64 to match the capture representation before the
    common complex128 projection; analytic F is the continuous-noise ideal.
    """
    if rows <= 0 or not np.isfinite(lam) or lam < 0:
        raise ValueError("positive rows and finite nonnegative lambda required")
    rng = np.random.default_rng(seed)
    output = []
    for start in range(0, rows, 4096):
        size = min(4096, rows-start)*K
        noise = (rng.standard_normal(size)+1j*rng.standard_normal(size))/np.sqrt(2)
        n = np.arange(start*K, start*K+size, dtype=np.float64)
        iq = noise+np.sqrt(lam/(2*K))*np.exp(2j*np.pi*TARGET*n/FS)
        output.append(projections(iq.astype(np.complex64)))
    return np.concatenate(output)


def acf(values, max_lag=32):
    a = np.asarray(values, dtype=np.float64)
    a = a-a.mean()
    den = np.dot(a,a)
    if den <= 0:
        return [None]*(max_lag+1)
    return [float(np.dot(a[:len(a)-lag], a[lag:])/den) for lag in range(max_lag+1)]


def diagnostics(iq, z):
    p = np.abs(z)**2
    r, counts = ratios(z)
    centered = z-z.mean(axis=0)
    covariance = centered.T @ centered.conj()/len(z)
    variance = np.real(np.diag(covariance))
    divisor = np.sqrt(variance[:, None]*variance[None, :])
    corr = np.divide(covariance, divisor, out=np.zeros_like(covariance), where=divisor>0)
    pseudocov = centered.T @ centered/len(z)
    pseudo = np.divide(pseudocov, divisor, out=np.zeros_like(pseudocov), where=divisor>0)
    # Fixed contiguous20-bin summaries, with every row used. No stationarity-based exclusions.
    windows = []
    for indexes in np.array_split(np.arange(len(z)), 20):
        vals = r[indexes]
        finite = vals[np.isfinite(vals)]
        windows.append({"first_row": int(indexes[0]), "stop_row_exclusive": int(indexes[-1]+1),
                        "mean_branch_power": p[indexes].mean(axis=0).tolist(),
                        "median_ratio": float(np.median(finite)) if finite.size else None,
                        "q95_ratio": float(np.quantile(finite,.95)) if finite.size else None})
    # Use bounded indicator/log diagnostics, since the ideal ratio has infinite variance.
    safe = np.where(np.isfinite(r), r, 0.)
    result = {"rows": len(z), "ratio_status": counts, "mean_branch_power": p.mean(axis=0).tolist(),
              "reference_power_lower_over_upper": float(p[:,1].mean()/p[:,2].mean()) if p[:,2].mean()>0 else None,
              "centered_projection_correlation_real": corr.real.tolist(),
              "centered_projection_correlation_imag": corr.imag.tolist(),
              "centered_projection_pseudocorrelation_magnitude": np.abs(pseudo).tolist(),
              "correlation_scope": "Observed projection dependence/properness diagnostic; coherent tone rotation and offset can affect these moments",
              "acf_ratio_above_ideal_median": acf((safe>f.median(2,4)).astype(float)),
              "acf_log1p_ratio": acf(np.log1p(safe)),
              "acf_target_power": acf(p[:,0]), "stationarity_windows": windows,
              "iq_mean_real": float(np.real(iq).mean()), "iq_mean_imag": float(np.imag(iq).mean()),
              "iq_mean_power": float(np.mean(np.abs(iq.astype(np.complex128))**2)),
              "iq_component_peak": float(max(np.max(np.abs(iq.real)),np.max(np.abs(iq.imag)))),
              "iq_component_at_or_above_098_count": int(np.sum((np.abs(iq.real)>=.98)|(np.abs(iq.imag)>=.98))),
              "iq_guard_note": "No digital overload is not proof of analog linearity or absence of clipping upstream",
              "invalid_ratio_policy": "All invalid ratios counted; comparison plots refuse invalid observations instead of silently deleting or replacing them"}
    freq, psd = welch(iq, fs=FS, window="hann", nperseg=8192, noverlap=4096,
                      detrend=False, return_onesided=False, scaling="density")
    order = np.argsort(freq)
    return result, freq[order], psd[order]


def distribution(lam):
    # Use central f explicitly at lambda0 (some SciPy ncf versions have sf edge behavior).
    return f(2,4) if lam == 0 else ncf(2,4,lam)


def ecdf(values, grid):
    values = np.sort(np.asarray(values))
    if not values.size or not np.isfinite(values).all():
        raise ValueError("ECDF requires all observed ratios finite; inspect invalid counts")
    return np.searchsorted(values,grid,side="right")/len(values)


def distribution_figure(output, name, observed, simulated, lam, label):
    model = distribution(lam)
    hi = float(max(300., model.ppf(.9995)))
    grid = np.geomspace(.001, hi, 1200)
    bins = np.geomspace(.001, hi, 100)
    q = np.concatenate([np.linspace(.001,.99,350), np.linspace(.991,.999,30)])
    fig, axes = plt.subplots(2, 2, figsize=(11.5,8.0), layout="constrained")
    pooled = np.concatenate([item[1] for item in observed])
    if not np.isfinite(pooled).all():
        raise ValueError("invalid observed ratios: preserve diagnostics, do not silently censor")
    theoretical_label = r"Ideal $F_{2,4}$" if lam==0 else rf"Conditional $F_{{2,4}}(\lambda={lam:.3g})$"
    for i,(identifier, values) in enumerate(observed):
        axes[0,1].plot(grid, ecdf(values,grid), color="#ca702c", alpha=.3, lw=.9,
                       label=("Individual captures" if label.startswith("SDR") else "Simulation A") if i==0 else None)
        tail = 1-ecdf(values,grid)
        axes[1,0].plot(grid,np.where(tail>0,tail,np.nan),color="#ca702c",alpha=.3,lw=.9)
    for values, color, text in ((pooled,"#bb5b1b",label),(simulated,"#1879a0","Matched raw-IQ white-Gaussian simulation")):
        counts, _ = np.histogram(values,bins=bins)
        # Density divides by ALL rows, not only rows falling within displayed limits.
        axes[0,0].stairs(counts/(len(values)*np.diff(bins)),bins,color=color,label=text)
        cdf = ecdf(values,grid)
        axes[0,1].plot(grid,cdf,color=color,lw=1.8,label=text)
        sf = 1-cdf
        axes[1,0].plot(grid,np.where(sf>0,sf,np.nan),color=color,lw=1.8,label=text)
        axes[1,1].plot(model.ppf(q),np.quantile(values,q),color=color,lw=1.5,label=text)
    axes[0,0].plot(grid,model.pdf(grid),"k--",lw=1.5,label=theoretical_label)
    axes[0,1].plot(grid,model.cdf(grid),"k--",lw=1.5,label=theoretical_label)
    axes[1,0].plot(grid,model.sf(grid),"k--",lw=1.5,label=theoretical_label)
    axes[1,1].plot(grid,grid,"k--",lw=1.5,label="Equal quantiles")
    for ax in axes.flat:
        ax.set_xscale("log"); ax.grid(alpha=.2); ax.set_xlabel("Single-row power ratio")
    axes[0,0].set_yscale("log"); axes[0,0].set_ylabel("Probability density per unit ratio")
    axes[0,1].set_ylabel("ECDF / ideal CDF"); axes[0,1].set_ylim(0,1)
    axes[1,0].set_yscale("log"); axes[1,0].set_ylabel("Fraction strictly above ratio"); axes[1,0].set_ylim(1/(len(pooled)*2),1)
    axes[1,1].set_yscale("log"); axes[1,1].set_xlabel("Ideal model quantile"); axes[1,1].set_ylabel("Observed / simulated quantile")
    axes[0,0].legend(fontsize=7.3); axes[0,1].legend(fontsize=7.3)
    fig.suptitle(f"{name}: one row, K=128, three floating projections",fontsize=14)
    footer = "Persistent spectral lines, serial dependence and signal leakage remain in observed data." if label.startswith("SDR") else "Simulation only; no radio data. Ideal white Gaussian IQ is rounded to the same complex64 storage format."
    fig.supxlabel("Descriptive comparisons; no IID row p-values or Gaussian error bars. Ideal single-row variance is infinite.\n"+footer,fontsize=9)
    for ext in ("png","pdf"): fig.savefig(output/f"{name}-distributions.{ext}",dpi=170)
    plt.close(fig)
    cdf_diff = ecdf(pooled,grid)-model.cdf(grid)
    return {"rows": len(pooled), "model_lambda": lam,
            "grid_max_abs_cdf_difference": float(np.max(np.abs(cdf_diff))),
            "cdf_difference_is_descriptive_not_a_pvalue": True,
            "displayed_pdf_range": [.001,hi],
            "outside_display_range_count": int(((pooled<.001)|(pooled>hi)).sum()),
            "quantiles": {str(prob): float(np.quantile(pooled,prob)) for prob in (.01,.1,.5,.9,.99,.999)}}, (grid,cdf_diff)


def preview(output, rows=93750):
    """Offline proof-of-method only; lambda20 is illustrative and never fitted."""
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    report = {"schema":"noise-signal-simulation-preview-v1","scope":"No radio data: independent raw-IQ Gaussian simulations versus ideal laws", "config":CONFIG,"geometry":geometry(),"source_sha256":sha(__file__),"rows_each":rows,"lambda_tone":20.,"seeds":{}}
    for mode,lam,index in (("noise",0.,0),("steady-signal",20.,1)):
        seed1, seed2 = 81000+index, 91000+index
        first,_ = ratios(simulation(rows,lam,seed1)); second,_ = ratios(simulation(rows,lam,seed2))
        value,_ = distribution_figure(output,mode,[("simulationA",first)],second,lam,"Independent raw-IQ simulation A")
        report[mode] = value; report["seeds"][mode] = [seed1,seed2]
    write_json(output/"report.json",report)
    return report


def check_hashes(inputs):
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"Evidence changed: {path}")


def utc_timestamp(value):
    stamp=datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("provenance timestamp must include timezone")
    return stamp


def load_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != "noise-signal-reference-inputs-v1":
        raise ValueError("unsupported input manifest schema")
    if value.get("analysis_source_sha256") != sha(__file__) or value.get("config") != CONFIG:
        raise ValueError("analysis source/config must match the preassigned manifest")
    entries = value.get("captures", [])
    expected = [(mode,"calibration" if index<2 else "comparison") for index in range(5) for mode in MODES]
    if [(v.get("mode"),v.get("role")) for v in entries] != expected:
        raise ValueError("require five ordered noise/txzero/tone triplets; first two calibration, last three comparison")
    utc_timestamp(value["assigned_utc"])
    ids = [entry["id"] for entry in entries]
    directories = [str(Path(entry["directory"]).resolve()) for entry in entries]
    if len(set(ids))!=15 or len(set(directories))!=15:
        raise ValueError("capture ids and directories must be unique")
    if any(not v or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in v) for v in ids):
        raise ValueError("capture ids may contain only letters,digits,hyphen,underscore")
    return value, {str(path):sha(path),str(Path(__file__).resolve()):sha(__file__)}


def read_capture(entry):
    directory = Path(entry["directory"]).resolve()
    receipt_path = directory/"receipt.json"
    receipt = json.loads(receipt_path.read_text())
    report = receipt.get("worker_report", {})
    if (receipt.get("schema") != "lime-reference-receipt-v1" or
        report.get("schema") != "lime-reference-capture-v1" or
        receipt.get("returncode") != 0 or
        any(receipt.get(key) is not True for key in ("success","cleanup_confirmed","accepted_interval_confirmed","hardware_attempted")) or
        any(report.get(key) is not True for key in ("success","accepted_interval_passed","rx_qualified")) or
        receipt.get("mode") != entry["mode"] or report.get("mode") != entry["mode"]):
        raise ValueError("capture must have a successful matching hardware/interval/cleanup receipt")
    plan_path = directory/"plan.json"
    if receipt.get("plan_sha256") != sha(plan_path):
        raise ValueError("receipt plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if (plan.get("mode") != entry["mode"] or plan.get("record_samples") != SAMPLES or
        plan.get("sample_rate_hz") != FS or plan.get("tone_frequency_hz") != TARGET or
        plan.get("frequency_hz") != 500_000_000. or plan.get("native_rx_gain_db") != 30):
        raise ValueError("hardware plan differs from fixed comparison settings")
    if entry["mode"] != "noise" and (plan.get("requested_native_tx_gain_db") != 50 or
        plan.get("expected_native_tx_gain_readback_db") != 50 or
        plan.get("tone_peak_component") != (.005 if entry["mode"]=="tone" else 0.)):
        raise ValueError("TX-enabled comparison settings differ")
    inputs = dict(plan["inputs"])
    inputs[str(receipt_path)] = sha(receipt_path)
    for name, expected in receipt["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("receipt artifact must be a local filename")
        inputs[str(directory/name)] = expected
    for name in ("accepted.cfile","rx.cfile","worker-status.json","plan.json","rx-chunks.jsonl","launch.json","schedule.json","startup.cfile"):
        if name not in receipt["artifacts"]:
            raise ValueError(f"receipt lacks required artifact {name}")
    check_hashes(inputs)
    if json.loads((directory/"worker-status.json").read_text()) != report:
        raise ValueError("receipt worker report differs from original status")
    if (report.get("accepted_samples") != SAMPLES or abs(report.get("rx_rate_hz",0)-FS)>1 or
        abs(report.get("rx_frequency_hz",0)-500_000_000.)>100 or report.get("native_rx_gain_db")!=30):
        raise ValueError("sample count, rate or RX setting mismatch")
    start, stop = report["accepted_first_timestamp"], report["accepted_stop_timestamp_exclusive"]
    if stop-start != SAMPLES or any(report.get("qualified_rx_"+key)!=0 for key in ("underrun","overrun","dropped")):
        raise ValueError("accepted timestamps or qualified RX counters invalid")
    if report["next_rx_timestamp"]-report["first_rx_timestamp"] != report["captured_samples"]:
        raise ValueError("qualified capture timestamps do not span exact saved sample count")
    schedule=json.loads((directory/"schedule.json").read_text())
    if (schedule.get("schema")!="lime-reference-schedule-v1" or schedule.get("mode")!=entry["mode"] or
        any(schedule.get(key)!=report.get(key) for key in ("schedule_base_timestamp","tx_start_timestamp","accepted_first_timestamp","accepted_stop_timestamp_exclusive")) or
        schedule.get("record_samples")!=SAMPLES):
        raise ValueError("saved schedule differs from completed accepted interval")
    base=report["schedule_base_timestamp"]
    if entry["mode"] == "noise":
        if (start!=base+int(.2*FS) or report.get("tx_start_timestamp")!=0 or
            report.get("tx_samples_sent") != 0 or report.get("tx_chain_active_for_payload") is not False or
            report.get("tx_attempted") is not False or report.get("payload_submission_attempted") is not False):
            raise ValueError("noise reference must have TX disabled")
        expected = {"TXEN_A":0,"EN_TXTSP":0,"EN_G_TRF":0,"PD_TXPAD_TRF":1}
        for phase in ("before","after"):
            if report.get("tx_disabled_register_readbacks",{}).get(phase) != expected:
                raise ValueError("noise capture lacks exact TX-disabled register evidence")
    elif (start != report["tx_start_timestamp"]+int(.15*FS) or report["tx_start_timestamp"]!=base+int(.2*FS) or
        report.get("tx_samples_sent")!=plan["payload_samples"] or report.get("native_tx_gain_db")!=50 or
        report.get("tx_overrun")!=0 or report.get("tx_dropped")!=0):
        raise ValueError("TX-enabled settings or accepted interval differs from contract")
    full_path = directory/"rx.cfile"
    if full_path.stat().st_size != report["captured_samples"]*8:
        raise ValueError("qualified raw byte count differs")
    accepted_path = directory/"accepted.cfile"
    if accepted_path.stat().st_size != SAMPLES*8:
        raise ValueError("accepted raw byte count differs")
    full = np.memmap(full_path, dtype="<c8", mode="r")
    offset = start-report["first_rx_timestamp"]
    if offset<0 or offset+SAMPLES>len(full):
        raise ValueError("accepted interval is outside qualified capture")
    iq = np.fromfile(accepted_path,dtype="<c8")
    if not np.array_equal(iq,full[offset:offset+SAMPLES]):
        raise ValueError("accepted IQ is not the exact timestamped slice of qualified raw")
    verify_chunk_receipts(directory,report,full,start,stop)
    return iq, inputs, receipt



def verify_chunk_receipts(directory,report,full,start,stop):
    q=report["qualification_samples"]
    startup_offset=report["qualification_overlap_startup_sample_offset"]
    startup_path=directory/"startup.cfile"
    if startup_path.stat().st_size!=report["startup_samples"]*8 or q<int(.1*FS):
        raise ValueError("startup/candidate byte count or clean duration invalid")
    startup=np.memmap(startup_path,dtype="<c8",mode="r")
    if startup_offset<0 or startup_offset+q>len(startup) or not np.array_equal(startup[startup_offset:startup_offset+q],full[:q]):
        raise ValueError("qualified prefix differs from retained startup candidate")
    offset=q;expected_timestamp=report["first_rx_timestamp"]+q;accepted_offset=0
    for line in (directory/"rx-chunks.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row["phase"]!="qualified":continue
        count=row["received_count"]
        if (count<=0 or row["file_sample_offset"]!=offset or row["timestamp"]!=expected_timestamp or
            row["timestamp_gap"] is not False or row["status_return"]!=0 or
            row["stream_active"] is not True or row["finite"] is not True or
            any(row[key]!=0 for key in ("underrun_delta","overrun_delta","dropped_delta")) or
            row["peak_component"]>=.98 or row["rms"]>=.35):
            raise ValueError("qualified chunk continuity/status/overload receipt invalid")
        count_accepted=max(0,min(expected_timestamp+count,stop)-max(expected_timestamp,start))
        if row["accepted_file_sample_offset"]!=accepted_offset or row["accepted_samples_from_chunk"]!=count_accepted:
            raise ValueError("chunk accepted-file mapping differs from predetermined interval")
        accepted_offset+=count_accepted;offset+=count;expected_timestamp+=count
    if offset!=len(full) or accepted_offset!=SAMPLES or expected_timestamp!=report["next_rx_timestamp"]:
        raise ValueError("qualified chunk receipts do not cover all retained samples")

def process_role(manifest, role, output):
    captures = []
    inputs = {}
    for entry in manifest["captures"]:
        if entry["role"] != role:
            continue
        iq, evidence, receipt = read_capture(entry)
        launch=json.loads((Path(entry["directory"])/"launch.json").read_text())
        if utc_timestamp(launch["started_utc"]) <= utc_timestamp(manifest["assigned_utc"]):
            raise ValueError("capture launch must follow preassigned mode/role manifest")
        inputs.update(evidence)
        z = projections(iq)
        info, frequency, psd = diagnostics(iq,z)
        r, _ = ratios(z)
        info.update(id=entry["id"],mode=entry["mode"],role=role,
                    receipt_completed_utc=receipt["completed_utc"],
                    launch_started_utc=launch["started_utc"])
        np.savez_compressed(output/f"{entry['id']}-projections.npz",projections=z,ratio=r,
                            frequency_hz=frequency,psd=psd)
        write_json(output/f"{entry['id']}-diagnostics.json",info)
        captures.append({"entry":entry,"diagnostics":info,"ratio":r,"frequency":frequency,"psd":psd})
    return captures,inputs


def power_calibration(captures):
    means = {}
    for mode in MODES:
        arrays = [item["diagnostics"]["mean_branch_power"] for item in captures if item["entry"]["mode"]==mode]
        if len(arrays)!=2:
            raise ValueError("require exactly two calibration captures per mode")
        means[mode] = np.mean(arrays,axis=0)
    if any(np.any(~np.isfinite(v)) or np.any(v<=0) for v in means.values()):
        raise ValueError("calibration branch powers must be positive and finite")
    target_excess = float(means["tone"][0]-means["txzero"][0])
    lam = 2*max(target_excess,0.)/means["txzero"][0]
    return {"lambda":float(lam),"primary_baseline":"txzero",
            "formula":"2*max(mean_target_power_tone-mean_target_power_txzero,0)/mean_target_power_txzero",
            "unclamped_target_power_excess":target_excess,"clamped_at_zero":target_excess<0,
            "mean_branch_power":{key:value.tolist() for key,value in means.items()},
            "tone_over_txzero_branch_power":(means["tone"]/means["txzero"]).tolist(),
            "txzero_over_noise_branch_power":(means["txzero"]/means["noise"]).tolist(),
            "noise_baseline_lambda_diagnostic_only":float(2*max(means["tone"][0]-means["noise"][0],0)/means["noise"][0]),
            "interpretation":"Target power of the separate TX-zero calibration is a putative noise variance. A coherent baseline line, colored/correlated noise, time drift or tone leakage into references can invalidate the ideal noncentral-F interpretation. No evaluation data are used to estimate lambda.",
            "calibration_uncertainty":"Point estimate from two captures per mode. Per-capture powers are retained; uncertainty in lambda is not included in the conditional reference curve."}


def calibrate(manifest_path, output):
    manifest, inputs = load_manifest(manifest_path)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    captures, captured_inputs=process_role(manifest,"calibration",output)
    inputs.update(captured_inputs)
    result={"schema":"noise-signal-reference-calibration-v1","scope":SCOPE,"config":CONFIG,
            "manifest_sha256":sha(manifest_path),"analysis_source_sha256":sha(__file__),
            "calibration":power_calibration(captures),"geometry":geometry(),
            "captures":[item["diagnostics"] for item in captures],"inputs":inputs,
            "comparison_files_read":False,"completed_utc":datetime.now(timezone.utc).isoformat()}
    check_hashes(inputs)
    write_json(output/"calibration.json",result)
    return result


def diagnostics_figure(output,captures):
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout="constrained")
    for row,mode in enumerate(MODES):
        for item in captures:
            if item["entry"]["mode"]!=mode:continue
            identifier=item["entry"]["id"]; info=item["diagnostics"]
            axes[row,0].plot(item["frequency"]/1000,10*np.log10(np.maximum(item["psd"],1e-300)),lw=.8,label=identifier)
            powers=np.array([v["mean_branch_power"] for v in info["stationarity_windows"]])
            for term,style in enumerate(("-","--",":")):
                axes[row,1].plot((np.arange(20)+.5)*.1,10*np.log10(powers[:,term]),style,lw=1,alpha=.75)
            axes[row,2].plot(np.arange(33)*K/FS*1000,info["acf_log1p_ratio"],lw=1,label=identifier)
        axes[row,0].axvline(TARGET/1000,color="black",lw=.8,ls="--")
        axes[row,0].axvline(310,color="gray",lw=.8,ls=":")
        axes[row,0].set_ylabel(f"{mode}\nPSD (digital units/Hz, dB)");axes[row,0].set_xlabel("Baseband frequency (kHz)")
        axes[row,0].legend(fontsize=7)
        axes[row,1].set_ylabel("Mean branch power (digital units, dB)");axes[row,1].set_xlabel("Time inside accepted capture (s)")
        axes[row,2].set_ylabel("ACF of log(1+ratio)");axes[row,2].set_xlabel("Row lag (ms)");axes[row,2].set_ylim(-1,1)
    axes[0,0].set_title("Whole-band spectra; no line removal")
    axes[0,1].set_title("20 fixed contiguous windows\nTarget solid; lower dashed; upper dotted")
    axes[0,2].set_title("Serial dependence diagnostics")
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle("Measured assumptions and stability checks",fontsize=14)
    fig.supxlabel("No diagnostic selects, trims or corrects data. Repeated captures share one RF path and are not certified independent trials.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(output/f"diagnostics.{ext}",dpi=170)
    plt.close(fig)


def compare(manifest_path,calibration_path,output):
    manifest,inputs=load_manifest(manifest_path)
    calibration_path=Path(calibration_path).resolve()
    cal=json.loads(calibration_path.read_text())
    if (cal.get("schema")!="noise-signal-reference-calibration-v1" or cal.get("config")!=CONFIG or
        cal.get("manifest_sha256")!=sha(manifest_path) or cal.get("analysis_source_sha256")!=sha(__file__) or
        cal.get("comparison_files_read") is not False):
        raise ValueError("calibration does not match the source/config/assignments")
    check_hashes(cal["inputs"])
    inputs.update(cal["inputs"]);inputs[str(calibration_path)]=sha(calibration_path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    captures,record_inputs=process_role(manifest,"comparison",output);inputs.update(record_inputs)
    if any(utc_timestamp(item["diagnostics"]["launch_started_utc"]) <= utc_timestamp(cal["completed_utc"]) for item in captures):
        raise ValueError("comparison captures must launch after completed calibration; do not present post-acquisition fitting as frozen")
    results={};differences={}
    for index,mode in enumerate(MODES):
        observed=[(item["entry"]["id"],item["ratio"]) for item in captures if item["entry"]["mode"]==mode]
        lam=cal["calibration"]["lambda"] if mode=="tone" else 0.
        simulated,_=ratios(simulation(sum(len(v) for _,v in observed),lam,920260909+index))
        np.savez_compressed(output/f"{mode}-matched-simulation.npz",ratio=simulated)
        results[mode],differences[mode]=distribution_figure(output,mode,observed,simulated,lam,"SDR comparison captures pooled")
        results[mode]["simulation_seed"]=920260909+index
    diagnostics_figure(output,captures)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    for ax,mode in zip(axes,MODES):
        grid,delta=differences[mode];ax.semilogx(grid,delta,color="#bb5b1b");ax.axhline(0,color="black",lw=.8)
        ax.set_title(mode);ax.set_xlabel("Single-row ratio");ax.set_ylabel("SDR ECDF minus ideal CDF");ax.grid(alpha=.2)
    fig.suptitle("Descriptive CDF differences; no IID goodness-of-fit p-values")
    for ext in ("png","pdf"):fig.savefig(output/f"cdf-differences.{ext}",dpi=170)
    plt.close(fig)
    result={"schema":"noise-signal-reference-comparison-v1","scope":SCOPE,"config":CONFIG,
            "calibration_file":str(calibration_path),"calibration_sha256":sha(calibration_path),
            "refit_performed":False,"calibration":cal["calibration"],"geometry":geometry(),
            "results":results,"captures":[item["diagnostics"] for item in captures],"inputs":inputs,
            "versions":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
            "limitations":["Ideal F(2,4) assumes proper Gaussian white noise and orthogonal equal-norm weights; radio colored noise and spectral lines can violate this.",
                "The strong reference is conditional on its independent calibration point estimate; parameter uncertainty is not plotted.",
                "A steady tone is added in complex amplitude before taking powers; it is not a mixture of quiet and strong populations.",
                "Nonoverlap avoids shared raw samples but does not establish temporal or between-capture independence.",
                "The single-row ideal mean is2+lambda and its variance is infinite; no ratio-mean standard-error claims are made.",
                "TX-disabled, TX-enabled zero IQ and tone captures are different hardware states; no baseline equivalence is assumed.",
                "The raw spectrum includes the known persistent approximately310kHz line; no cause, RF cleanliness or antenna versus internal-coupling attribution is established."]}
    check_hashes(inputs)
    write_json(output/"comparison.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preview","calibrate","compare"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--preview-rows",type=int,default=93750)
    args=parser.parse_args()
    if args.stage=="preview":result=preview(args.output,args.preview_rows)
    elif args.stage=="calibrate":
        if args.manifest is None:parser.error("calibrate needs --manifest")
        result=calibrate(args.manifest,args.output)
    else:
        if args.manifest is None or args.calibration is None:parser.error("compare needs --manifest and --calibration")
        result=compare(args.manifest,args.calibration,args.output)
    print(json.dumps({"schema":result["schema"],"output":str(args.output.resolve()),"scope":result["scope"]},indent=2))


if __name__=="__main__":main()
