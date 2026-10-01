#!/usr/bin/env python3
"""Freeze and generate finite GNU Radio Gaussian-plus-tone IQ; no SDR APIs.

Run with PYTHONNOUSERSITE=1 /usr/bin/python3. The actual GNU Radio noise_source_c,
sig_source_c and add_cc blocks produce the saved streams. NumPy only summarizes
those files after generation; it does not generate or normalize their samples.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

FS=2_000_000.
K=128
TONE_HZ=100_000.


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())


def runtime_identity():
    from gnuradio import gr,analog,blocks
    import numpy as np
    if os.environ.get("PYTHONNOUSERSITE")!="1":raise ValueError("set PYTHONNOUSERSITE=1")
    if gr.version()!="3.10.9.2":raise ValueError("this versioned generator requires GNU Radio3.10.9.2")
    paths={Path(sys.executable).resolve(),Path(__file__).resolve()}
    for module in (gr,analog,blocks,np):
        paths.add(Path(module.__file__).resolve())
        paths.update(v.resolve() for v in Path(module.__file__).parent.glob("*.so"))
    for line in Path("/proc/self/maps").read_text().splitlines():
        path=line.split()[-1]
        if path.startswith("/") and any(term in path for term in ("libgnuradio-","libvolk","libfmt","libspdlog")):
            paths.add(Path(path).resolve())
    return {"gnuradio":gr.version(),"numpy":np.__version__,"python":sys.version,
            "executable":str(Path(sys.executable).resolve()),"platform":platform.platform(),
            "files":{str(path):sha(path) for path in sorted(paths)}}


def prepare(output,*,mode,seed,lam,samples=4_000_000,calibration=None,evidence_dir=None):
    if mode not in ("noise","txzero","tone"):raise ValueError("unknown reference mode")
    if type(seed) is not int or not 1<=seed<=2**31-1:raise ValueError("explicit nonzero31-bit seed required")
    if type(samples) is not int or samples<=0 or samples>12_000_000 or samples%K:raise ValueError("finite sample count must be a positive multiple of128, at most12M")
    if not math.isfinite(lam) or lam<0 or (mode!="tone" and lam!=0):raise ValueError("finite nonnegative lambda; zero for noise/txzero")
    identity=runtime_identity()
    inputs=dict(identity["files"])
    if calibration is not None:
        calibration=Path(calibration).resolve();cal=json.loads(calibration.read_text())
        if cal.get("schema")!="noise-signal-reference-calibration-v1":raise ValueError("unknown independent calibration schema")
        if mode=="tone" and cal["calibration"]["lambda"]!=lam:raise ValueError("tone lambda differs from independent calibration")
        inputs[str(calibration)]=sha(calibration)
    if evidence_dir is not None:
        for path in Path(evidence_dir).resolve().iterdir():
            if path.is_file():inputs[str(path)]=sha(path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan={"schema":"gnuradio-reference-plan-v1","created_utc":datetime.now(timezone.utc).isoformat(),
          "scope":"Actual GNU Radio software simulation; no hardware or measured RF power",
          "mode":mode,"seed":seed,"lambda":lam,"samples":samples,"sample_rate_hz":FS,
          "target_hz":TONE_HZ,"window_samples":K,"noise_source_amplitude":1.,
          "complex_noise_variance_ideal":1.,"tone_amplitude":math.sqrt(lam/(2*K)),
          "source_graph":"noise_source_c(GR_GAUSSIAN,1,seed)->head(N)->add_cc[0]; sig_source_c(fs,GR_COS_WAVE,100k,sqrt(lambda/(2K)),0,phase0)->head(N)->add_cc[1]; each source and sum saved complex64",
          "conditioning":"None: no forced energy, orthogonalization, mean subtraction, whitening or filtering",
          "noise_normalization_source":"GNU Radio3.10.9.2 complex noise constructor divides requested amplitude by sqrt2; rayleigh_complex returns two standard Gaussian draws",
          "normalization_references":["https://raw.githubusercontent.com/gnuradio/gnuradio/v3.10.9.2/gr-analog/lib/noise_source_impl.cc","https://raw.githubusercontent.com/gnuradio/gnuradio/v3.10.9.2/gnuradio-runtime/lib/math/random.cc"],
          "runtime":identity,"inputs":inputs,"calibration_path":str(calibration) if calibration is not None else None,
          "generation_started":False}
    write_json(output/"plan.json",plan);write_json(output/"plan-digest.json",{"sha256":sha(output/"plan.json")})
    (output/"generator-source.py").write_bytes(Path(__file__).read_bytes())
    return plan


def generate(output):
    from gnuradio import gr,analog,blocks
    import numpy as np
    output=Path(output).resolve();plan=json.loads((output/"plan.json").read_text())
    if plan.get("schema")!="gnuradio-reference-plan-v1" or sha(output/"plan.json")!=json.loads((output/"plan-digest.json").read_text())["sha256"]:
        raise ValueError("frozen plan changed")
    for path,digest in plan["inputs"].items():
        if sha(path)!=digest:raise ValueError(f"frozen input changed: {path}")
    if runtime_identity()!=plan["runtime"]:raise ValueError("GNU Radio runtime changed since plan")
    write_json(output/"launch.json",{"started_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":sha(output/"plan.json"),"hardware_attempted":False})
    sinks=[];tb=None
    receipt={"schema":"gnuradio-reference-receipt-v1","success":False,"plan_sha256":sha(output/"plan.json"),"hardware_attempted":False,"mode":plan["mode"]}
    try:
        tb=gr.top_block("finite reference Gaussian plus complex tone")
        noise=analog.noise_source_c(analog.GR_GAUSSIAN,1.,plan["seed"])
        tone=analog.sig_source_c(FS,analog.GR_COS_WAVE,TONE_HZ,plan["tone_amplitude"],0.,0.)
        heads=[blocks.head(gr.sizeof_gr_complex,plan["samples"]) for _ in range(2)]
        add=blocks.add_cc(1)
        for name in ("noise.cfile","signal.cfile","iq.cfile"):
            fd=os.open(output/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
            try:sinks.append(blocks.file_descriptor_sink(gr.sizeof_gr_complex,fd))
            except BaseException:os.close(fd);raise
        tb.connect(noise,heads[0]);tb.connect(tone,heads[1])
        tb.connect(heads[0],(add,0));tb.connect(heads[1],(add,1))
        tb.connect(heads[0],sinks[0]);tb.connect(heads[1],sinks[1]);tb.connect(add,sinks[2])
        tb.run(max_noutput_items=4096)
        for name in ("noise.cfile","signal.cfile","iq.cfile"):
            if (output/name).stat().st_size!=plan["samples"]*8:raise ValueError("GNU Radio sample count differs from finite plan")
        noise_iq=np.fromfile(output/"noise.cfile",dtype="<c8")
        signal_iq=np.fromfile(output/"signal.cfile",dtype="<c8")
        mixed=np.fromfile(output/"iq.cfile",dtype="<c8")
        if not all(np.isfinite(v).all() for v in (noise_iq,signal_iq,mixed)):raise ValueError("nonfinite generated IQ")
        addition_exact=bool(np.array_equal(mixed,noise_iq+signal_iq))
        if not addition_exact:raise ValueError("saved add_cc output differs from saved source sum")
        noise_power=float(np.mean(np.abs(noise_iq.astype(np.complex128))**2))
        signal_power=float(np.mean(np.abs(signal_iq.astype(np.complex128))**2))
        receipt.update(success=True,samples=plan["samples"],actual_gnuradio_blocks=True,
            saved_addition_exact=addition_exact,noise_mean_power=noise_power,
            noise_real_variance=float(np.var(noise_iq.real.astype(np.float64))),
            noise_imag_variance=float(np.var(noise_iq.imag.astype(np.float64))),
            noise_real_imag_covariance=float(np.mean((noise_iq.real-noise_iq.real.mean())*(noise_iq.imag-noise_iq.imag.mean()))),
            signal_mean_power=signal_power,
            generator_note="Reported sample moments are diagnostics and never used to renormalize generated streams")
    except BaseException as error:
        receipt["error"]=repr(error);raise
    finally:
        if tb is not None:tb.stop();tb.wait()
        receipt["completed_utc"]=datetime.now(timezone.utc).isoformat()
        receipt["artifacts"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="receipt.json"}
        write_json(output/"receipt.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","generate"));p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("noise","txzero","tone"));p.add_argument("--seed",type=int)
    p.add_argument("--lambda",dest="lam",type=float);p.add_argument("--samples",type=int,default=4_000_000)
    p.add_argument("--calibration",type=Path);p.add_argument("--evidence-dir",type=Path)
    a=p.parse_args()
    if a.stage=="prepare":
        if a.mode is None or a.seed is None or a.lam is None:p.error("prepare requires mode,seed and lambda")
        result=prepare(a.output,mode=a.mode,seed=a.seed,lam=a.lam,samples=a.samples,calibration=a.calibration,evidence_dir=a.evidence_dir)
    else:result=generate(a.output)
    print(json.dumps({"schema":result["schema"],"mode":result["mode"],"success":result.get("success"),"output":str(a.output.resolve())},indent=2))


if __name__=="__main__":main()
