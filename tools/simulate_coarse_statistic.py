#!/usr/bin/env python3
"""Actual GNU Radio ideal projection-space simulation of the pooled CANFAR coarse ratio.

No radio access; no packed-input or correlated-telescope simulation. Three ordinary
Gaussian projection streams feed float32 individual powers and float64 block sums.
Only the aggregate A,B,C,Q frame products are retained, not billions of projections.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

P=2048*128
STATES={"noise":{"gamma":0.,"seeds":[926091301,926091302,926091303]},
        "steady":{"gamma":.02,"seeds":[926091401,926091402,926091403]}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1048576),b""):h.update(part)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())


def runtime():
    from gnuradio import gr,analog,blocks
    import numpy as np
    if os.environ.get("PYTHONNOUSERSITE")!="1":raise ValueError("set PYTHONNOUSERSITE=1")
    if gr.version()!="3.10.9.2":raise ValueError("GNU Radio3.10.9.2 required")
    paths={Path(__file__).resolve(),Path(sys.executable).resolve()}
    for module in (gr,analog,blocks,np):
        paths.add(Path(module.__file__).resolve())
        paths.update(p.resolve() for p in Path(module.__file__).parent.glob("*.so"))
    for line in Path("/proc/self/maps").read_text().splitlines():
        path=line.split()[-1]
        if path.startswith("/") and any(v in path for v in ("libgnuradio-","libvolk","libfmt","libspdlog")):
            paths.add(Path(path).resolve())
    return {"gnuradio":gr.version(),"numpy":np.__version__,"python":sys.version,
            "executable":str(Path(sys.executable).resolve()),"platform":platform.platform(),
            "files":{str(p):sha(p) for p in sorted(paths)}}


def theoretical_moments(gamma):
    expected_a=P*(1+gamma);var_a=P*(1+2*gamma)
    mean=2*expected_a/(2*P-1)
    second=4*(expected_a**2+var_a)/((2*P-1)*(2*P-2))
    return {"A_mean":expected_a,"A_variance":var_a,"B_mean":P,"B_variance":P,
            "C_mean":P,"C_variance":P,"Q_mean":mean,"Q_variance":second-mean**2,
            "df_numerator":2*P,"df_denominator":4*P,"noncentrality":2*P*gamma}


def prepare(output,frames,evidence_dir=None):
    if type(frames)is not int or not 1<=frames<=4096:raise ValueError("frames must be1..4096")
    identity=runtime();inputs=dict(identity["files"])
    if evidence_dir:
        for path in Path(evidence_dir).resolve().iterdir():
            if path.is_file():inputs[str(path)]=sha(path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan={"schema":"canfar-coarse-gnuradio-plan-v2","created_utc":datetime.now(timezone.utc).isoformat(),
          "scope":"Ideal independent Gaussian projection-space simulation of the CANFAR coarse aggregation; not packed raw telescope data or a physical2048-receiver simulation",
          "frames_per_state":frames,"projections_per_frame":P,"receivers_count_model":2048,"windows_per_receiver_model":128,
          "statistic":"A=sum|ztarget|^2;B=sum|zlower|^2;C=sum|zupper|^2;Q=2A/(B+C); powers sum before ratio",
          "states":STATES,"noise_source_amplitude":1.,"complex_noise_variance_ideal":1.,
          "stream_model":"Independent CN(0,1) projections; deterministic real target mean sqrt(gamma); central references",
          "actual_gnuradio_blocks":["analog.noise_source_c(GR_GAUSSIAN,1,seed) x3","analog.sig_source_c(GR_CONST_WAVE,sqrt(gamma))","blocks.add_cc","blocks.complex_to_mag_squared x3","blocks.head x3","custom gr.decim_block with np.sum(dtype=float64)","blocks.file_descriptor_sink"],
          "precision":"GNU complex64 projections and individual float32 squared magnitudes; each P-term branch sum accumulated with NumPy float64, then float64 ratio",
          "conditioning":"No exact-energy normalization, orthogonalization, mean subtraction or fitted scaling",
          "output_schema":"noise.npz and steady.npz contain float64 length-N arrays A,B,C,Q; aggregate binary rows contain A,B,C,Q in that order",
          "retention":"Aggregate frames only; no full projection streams retained. Seeds, runtime hashes and source recreate simulation.",
          "runtime":identity,"inputs":inputs,
          "source_references":["https://raw.githubusercontent.com/gnuradio/gnuradio/v3.10.9.2/gr-analog/lib/noise_source_impl.cc","https://raw.githubusercontent.com/gnuradio/gnuradio/v3.10.9.2/gnuradio-runtime/lib/math/random.cc"]}
    write_json(output/"plan.json",plan);write_json(output/"plan-digest.json",{"sha256":sha(output/"plan.json")})
    (output/"generator-source.py").write_bytes(Path(__file__).read_bytes())
    return plan


def aggregate_float64(powers):
    """Independent testable reduction of the actual float32 GNU branch powers."""
    import numpy as np
    rows=min(len(v)//P for v in powers)
    result=np.empty((rows,4),dtype=np.float64)
    for index,power in enumerate(powers):
        result[:,index]=np.sum(np.asarray(power[:rows*P]).reshape(rows,P),axis=1,dtype=np.float64)
    den=result[:,1]+result[:,2]
    if np.any(den<=0) or not np.isfinite(result[:,:3]).all():raise ValueError("invalid power sums")
    result[:,3]=2*result[:,0]/den
    return result


def generate(output,state):
    from gnuradio import gr,analog,blocks
    import numpy as np
    output=Path(output).resolve();plan=json.loads((output/"plan.json").read_text())
    if plan.get("schema")!="canfar-coarse-gnuradio-plan-v2" or sha(output/"plan.json")!=json.loads((output/"plan-digest.json").read_text())["sha256"]:
        raise ValueError("frozen plan changed")
    if plan["states"]!=STATES or plan["projections_per_frame"]!=P:raise ValueError("state/grouping plan differs")
    for path,digest in plan["inputs"].items():
        if sha(path)!=digest:raise ValueError(f"frozen input changed:{path}")
    if runtime()!=plan["runtime"]:raise ValueError("runtime differs from frozen plan")
    if state not in STATES:raise ValueError("unknown state")
    config=STATES[state];frames=plan["frames_per_state"];count=frames*P
    class PowerSums(gr.decim_block):
        def __init__(self):
            gr.decim_block.__init__(self,name="float64 independent branch power sums",in_sig=[np.float32]*3,out_sig=[(np.float64,4)],decim=P)
        def work(self,input_items,output_items):
            rows=min(len(output_items[0]),min(len(v)//P for v in input_items))
            values=aggregate_float64([v[:rows*P] for v in input_items])
            output_items[0][:rows]=values
            return rows
    receipt={"schema":"canfar-coarse-gnuradio-receipt-v2","state":state,"success":False,
             "hardware_attempted":False,"actual_gnuradio_generation":True,"plan_sha256":sha(output/"plan.json")}
    write_json(output/f"{state}-launch.json",{"started_utc":datetime.now(timezone.utc).isoformat(),"state":state,"plan_sha256":sha(output/"plan.json")})
    tb=None;start=time.monotonic()
    try:
        tb=gr.top_block("finite ideal CANFAR coarse reference")
        sources=[analog.noise_source_c(analog.GR_GAUSSIAN,1.,seed) for seed in config["seeds"]]
        constant=analog.sig_source_c(1.,analog.GR_CONST_WAVE,0.,math.sqrt(config["gamma"]),0.)
        add=blocks.add_cc(1);tb.connect(sources[0],(add,0));tb.connect(constant,(add,1))
        branches=[add,sources[1],sources[2]]
        powers=[blocks.complex_to_mag_squared(1) for _ in range(3)]
        heads=[blocks.head(gr.sizeof_float,count) for _ in range(3)]
        reducer=PowerSums()
        for index in range(3):
            tb.connect(branches[index],powers[index],heads[index],(reducer,index))
        fd=os.open(output/f"{state}-aggregates.f64",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        try:sink=blocks.file_descriptor_sink(32,fd)
        except BaseException:os.close(fd);raise
        tb.connect(reducer,sink);tb.run(max_noutput_items=262144)
        values=np.fromfile(output/f"{state}-aggregates.f64",dtype="<f8")
        if values.size!=frames*4:raise ValueError("aggregate frame count differs")
        values=values.reshape(frames,4)
        if np.any(values[:,:3]<0) or not np.isfinite(values).all():raise ValueError("aggregate values invalid")
        if not np.array_equal(values[:,3],2*values[:,0]/(values[:,1]+values[:,2])):raise ValueError("ratio differs from retained power sums")
        with (output/f"{state}.npz").open("xb") as stream:
            np.savez_compressed(stream,A=values[:,0],B=values[:,1],C=values[:,2],Q=values[:,3])
        theoretical=theoretical_moments(config["gamma"])
        checks={};observed={}
        for index,key in enumerate(("A","B","C","Q")):
            mean=float(values[:,index].mean());variance=float(values[:,index].var(ddof=1)) if frames>1 else None
            theory_mean=theoretical[key+"_mean"];theory_variance=theoretical[key+"_variance"]
            observed[key]={"mean":mean,"sample_variance":variance,
                "mean_standardized_mc_error":(mean-theory_mean)/math.sqrt(theory_variance/frames),
                "sample_variance_over_ideal":variance/theory_variance if variance is not None else None}
            if frames>=128:
                checks[key+"_mean_within_5_theoretical_MC_SE"]=abs(observed[key]["mean_standardized_mc_error"])<5
                checks[key+"_variance_within_25_percent"]=abs(variance/theory_variance-1)<.25
        receipt.update(success=True,frames=frames,projections_per_frame=P,branch_projections_generated_each=count,
                       gamma=config["gamma"],seeds=config["seeds"],theoretical_moments=theoretical,
                       observed_moments=observed,moment_sanity_checks=checks,
                       moment_sanity_passed=all(checks.values()) if checks else None,
                       moment_check_scope="Predetermined broad Monte Carlo sanity checks; no retuning, seed rejection or claim of physical model acceptance")
    except BaseException as error:
        receipt["error"]=repr(error);raise
    finally:
        if tb is not None:tb.stop();tb.wait()
        receipt["elapsed_seconds"]=time.monotonic()-start
        receipt["completed_utc"]=datetime.now(timezone.utc).isoformat()
        receipt["artifacts"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name.startswith(state) and p.name!=f"{state}-receipt.json"}
        write_json(output/f"{state}-receipt.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","generate"));p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=1024);p.add_argument("--state",choices=tuple(STATES))
    p.add_argument("--evidence-dir",type=Path)
    a=p.parse_args()
    if a.stage=="prepare":result=prepare(a.output,a.frames,a.evidence_dir)
    else:
        if a.state is None:p.error("generate requires --state")
        result=generate(a.output,a.state)
    print(json.dumps({key:result.get(key) for key in ("schema","state","frames","success","elapsed_seconds","moment_sanity_passed")},indent=2))


if __name__=="__main__":main()
