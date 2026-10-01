#!/usr/bin/env python3
"""Produce exactly two density figures: ideal law, actual GNU Radio, measured SDR.

Uses the frozen v1 single-row projection and capture-validation functions. All
power calibration is loaded from reserved captures; no comparison data refit.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CORE_PATH=Path(__file__).with_name("analyze_noise_signal_references.py")
spec=importlib.util.spec_from_file_location("reference_core_v1",CORE_PATH)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def read_gnuradio(directory,lam):
    directory=Path(directory).resolve()
    receipt_path=directory/"receipt.json";plan_path=directory/"plan.json"
    receipt=json.loads(receipt_path.read_text());plan=json.loads(plan_path.read_text())
    if (receipt.get("schema")!="gnuradio-reference-receipt-v1" or receipt.get("success") is not True or
        receipt.get("actual_gnuradio_blocks") is not True or receipt.get("hardware_attempted") is not False or
        receipt.get("saved_addition_exact") is not True or receipt.get("plan_sha256")!=core.sha(plan_path)):
        raise ValueError("GNU Radio successful source/addition receipt required")
    mode=plan.get("mode")
    if (plan.get("schema")!="gnuradio-reference-plan-v1" or mode not in ("noise","tone") or
        plan.get("lambda")!=(lam if mode=="tone" else 0.) or plan.get("samples")!=core.SAMPLES or
        plan.get("sample_rate_hz")!=core.FS or plan.get("window_samples")!=core.K or
        plan.get("target_hz")!=core.TARGET or plan.get("noise_source_amplitude")!=1.):
        raise ValueError("GNU Radio source settings differ from comparison model")
    inputs=dict(plan["inputs"]);inputs[str(receipt_path)]=core.sha(receipt_path)
    for name,digest in receipt["artifacts"].items():
        if Path(name).name!=name:raise ValueError("GNU artifact path must be a filename")
        inputs[str(directory/name)]=digest
    for name in ("plan.json","noise.cfile","signal.cfile","iq.cfile"):
        if name not in receipt["artifacts"]:raise ValueError("GNU receipt lacks source/raw artifact")
    core.check_hashes(inputs)
    for name in ("noise.cfile","signal.cfile","iq.cfile"):
        if (directory/name).stat().st_size!=8*core.SAMPLES:raise ValueError("GNU raw count mismatch")
    noise=np.fromfile(directory/"noise.cfile",dtype="<c8")
    signal=np.fromfile(directory/"signal.cfile",dtype="<c8")
    iq=np.fromfile(directory/"iq.cfile",dtype="<c8")
    if not np.array_equal(iq,noise+signal):raise ValueError("saved GNU add_cc source sum mismatch")
    ratio,status=core.ratios(core.projections(iq))
    if not np.isfinite(ratio).all():raise ValueError("invalid GNU ratio must not be silently deleted")
    return ratio,inputs,{"directory":str(directory),"mode":mode,"seed":plan["seed"],"plan":plan,"receipt":receipt,"ratio_status":status}


def density_per_dex(values,edges):
    values=np.asarray(values)
    if not values.size or np.any(values<=0) or not np.isfinite(values).all():
        raise ValueError("per-dex density requires all finite positive ratios; inspect invalid or zero count")
    counts,_=np.histogram(np.log10(values),bins=edges)
    return counts/(len(values)*np.diff(edges)),int(len(values)-counts.sum())


def draw_density(output,mode,observed,simulated,lam):
    model=core.distribution(lam)
    lo=min(-3.,float(np.log10(model.ppf(.0001))))
    hi=max(float(np.log10(300)),float(np.log10(model.ppf(.9995))))
    edges=np.linspace(lo,hi,121);x=np.linspace(lo,hi,1800);r=10**x
    theoretical=np.log(10)*r*model.pdf(r)
    radio_density,radio_outside=density_per_dex(observed,edges)
    gnu_density,gnu_outside=density_per_dex(simulated,edges)
    fig,ax=plt.subplots(figsize=(8.4,5.3),layout="constrained")
    ax.plot(x,theoretical,color="#292d32",lw=2.2,label=(r"Theory: $F_{2,4}$" if lam==0 else rf"Conditional theory: $F_{{2,4}}(\lambda={lam:.4g})$"))
    ax.stairs(gnu_density,edges,color="#267ca5",lw=1.6,label=f"GNU Radio simulation ({len(simulated):,} rows)")
    ax.stairs(radio_density,edges,color="#ce681c",lw=1.5,label=f"SDR comparison ({len(observed):,} rows)")
    ax.set_xlim(lo,hi);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    ax.set_xlabel(r"$x=\log_{10}R$, where $R=2|z_0|^2/(|z_-|^2+|z_+|^2)$")
    ax.set_ylabel(r"Probability density per dex: $p_x(x)=\ln(10)\,10^x p_R(10^x)$")
    ax.set_title("Noise-only reference" if mode=="noise" else "Steady signal plus noise reference",fontsize=15)
    ax.legend(fontsize=9,loc="upper right")
    note=("Single-row reference: one RF path, K=128, floating projections; no quantization.\n"
          "CANFAR pools powers over many inputs/windows; its fine statistic also searches and normalizes bins.")
    if mode=="tone":note+="\nStrong-signal parameter comes from separate calibration; curve is conditional on ideal noise/reference assumptions."
    fig.supxlabel(note,fontsize=8.6)
    for extension in ("png","pdf"):fig.savefig(output/f"{mode}-density.{extension}",dpi=190)
    plt.close(fig)
    grid=np.geomspace(10**lo,10**hi,1800)
    return {"mode":mode,"lambda":lam,"ordinate":"probability density per unit log10(R), not per unit R or logcounts",
            "radio_rows":len(observed),"gnuradio_rows":len(simulated),"log10_ratio_limits":[lo,hi],
            "bins":len(edges)-1,"radio_outside_display":radio_outside,"gnuradio_outside_display":gnu_outside,
            "normalization":"histogram counts divided by all rows and bin width; displayed tails are not renormalized",
            "radio_grid_max_abs_cdf_difference":float(np.max(np.abs(core.ecdf(observed,grid)-model.cdf(grid)))),
            "gnuradio_grid_max_abs_cdf_difference":float(np.max(np.abs(core.ecdf(simulated,grid)-model.cdf(grid)))),
            "cdf_differences_are_descriptive_only":True,
            "density_bin_edges":edges.tolist(),"radio_density_per_dex":radio_density.tolist(),"gnuradio_density_per_dex":gnu_density.tolist()}


def run(manifest_path,calibration_path,gnu_directories,output,analysis_core=None):
    global core,CORE_PATH
    if analysis_core is not None:
        CORE_PATH=Path(analysis_core).resolve()
        module_spec=importlib.util.spec_from_file_location("selected_reference_core",CORE_PATH)
        core=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(core)
    manifest,inputs=core.load_manifest(manifest_path)
    calibration_path=Path(calibration_path).resolve();cal=json.loads(calibration_path.read_text())
    if (cal.get("schema")!="noise-signal-reference-calibration-v1" or cal.get("manifest_sha256")!=core.sha(manifest_path) or
        cal.get("analysis_source_sha256")!=core.sha(CORE_PATH) or cal.get("config")!=core.CONFIG or
        cal.get("comparison_files_read") is not False):raise ValueError("independent calibration contract mismatch")
    core.check_hashes(cal["inputs"]);inputs.update(cal["inputs"])
    inputs[str(calibration_path)]=core.sha(calibration_path);inputs[str(Path(__file__).resolve())]=core.sha(__file__)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    diagnostics_dir=output/"supporting-diagnostics";diagnostics_dir.mkdir()
    captures,hardware_inputs=core.process_role(manifest,"comparison",diagnostics_dir);inputs.update(hardware_inputs)
    if any(datetime.fromisoformat(item["diagnostics"]["launch_started_utc"])<=datetime.fromisoformat(cal["completed_utc"]) for item in captures):
        raise ValueError("comparison hardware must launch after completed independent calibration")
    lam=cal["calibration"]["lambda"]
    simulations={"noise":[],"tone":[]};receipts=[];seeds=[];directories=[]
    for directory in gnu_directories:
        values,evidence,info=read_gnuradio(directory,lam)
        mode=info["mode"];simulations[mode].append(values);inputs.update(evidence)
        seeds.append(info["seed"]);directories.append(info["directory"]);receipts.append(info)
        plan=info["plan"]
        if mode=="tone" and (plan.get("calibration_path")!=str(calibration_path) or
            plan["inputs"].get(str(calibration_path))!=core.sha(calibration_path)):
            raise ValueError("GNU tone must bind the same independent calibration")
    if len(seeds)!=6 or len(set(seeds))!=6 or len(set(directories))!=6 or any(len(v)!=3 for v in simulations.values()):
        raise ValueError("need three independent-seed GNU runs per displayed mode, six distinct directories/seeds")
    results={}
    for mode in ("noise","tone"):
        observed=np.concatenate([v["ratio"] for v in captures if v["entry"]["mode"]==mode])
        generated=np.concatenate(simulations[mode])
        results[mode]=draw_density(output,mode,observed,generated,lam if mode=="tone" else 0.)
    report={"schema":"gnu-radio-sdr-two-reference-densities-v1","scope":core.SCOPE,"config":core.CONFIG,
            "figures":["noise-density.pdf","tone-density.pdf"],"results":results,
            "calibration":cal["calibration"],"calibration_path":str(calibration_path),"refit_performed":False,
            "gnuradio_generations":receipts,"capture_diagnostics":[v["diagnostics"] for v in captures],
            "geometry":core.geometry(),"inputs":inputs,
            "limits":["These are single-row F(2,4) primitive references, not the pooled CANFAR coarse or fine-decision distributions.",
                "The ideal single-row mean is2+lambda and variance is infinite. No ratio-mean Gaussian error bars or IID row p-values are reported.",
                "Strong theory/GNU use a point estimate from separate TX-zero/tone calibration, not a fit to comparison ratios; parameter uncertainty is not plotted.",
                "SDR serial dependence, colored noise, reference leakage, drift, spurs and nonlinearity can produce disagreement; no observed rows are trimmed or whitened.",
                "TX-zero is retained only as separate hardware-state control/calibration. It is not assumed equivalent to TX-disabled noise."]}
    core.check_hashes(inputs);core.write_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--gnu-dir",type=Path,action="append",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--analysis-core",type=Path,help="explicit separately versioned core; its source hash must match calibration/manifest")
    a=p.parse_args();result=run(a.manifest,a.calibration,a.gnu_dir,a.output,a.analysis_core)
    print(json.dumps({"schema":result["schema"],"figures":result["figures"],"output":str(a.output.resolve())},indent=2))
