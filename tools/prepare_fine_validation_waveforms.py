#!/usr/bin/env python3
"""Freeze, generate and qualify four fresh deterministic ATSC payload fixtures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAIL = REPO.parent
FS = 4_500_000.0 / 286.0 * 684.0
PILOT = -3_000_000.0 + (6_000_000.0 - FS / 2.0) / 2.0
RETAINED = 4_000_000
DISCARD = 200_000
PACKETS = 16_384
WINDOW = math.ceil(((16_384 + 4 - 1) * 2048 - 1) * FS / 800_000_000.0) + 1
WINDOW_COUNT = RETAINED // WINDOW
REFERENCE_WIDTH = 390625.0 / 128
REFERENCE_DISTANCE = 2 * REFERENCE_WIDTH
SPAN_LIMITS_DB = {"total_rms": 0.25, "coherent_pilot_amplitude": 1.0,
                  "lower_reference_band_power": 2.0, "upper_reference_band_power": 2.0}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def scalar(value):
    return float(value) if np.isfinite(value) else None


def qualify(path):
    """Independent IQ-domain qualification; no detector or PFB implementation."""
    iq = np.memmap(path, dtype=np.complex64, mode='r')
    if iq.size != RETAINED:
        raise ValueError('Unexpected retained IQ length')
    time = np.arange(WINDOW, dtype=np.float64) / FS
    carrier = np.exp(2j * np.pi * PILOT * time)
    hann = np.hanning(WINDOW)
    frequency = np.fft.fftfreq(WINDOW, 1 / FS)
    masks = {}
    for name, sign in (('lower', -1), ('upper', 1)):
        centre = PILOT + sign * REFERENCE_DISTANCE
        masks[name] = np.abs(frequency - centre) <= REFERENCE_WIDTH / 2
    windows = []
    for i in range(WINDOW_COUNT):
        start = i * WINDOW
        part = np.asarray(iq[start:start+WINDOW], dtype=np.complex128)
        finite = bool(np.isfinite(part).all())
        total_rms = math.sqrt(float(np.mean(np.abs(part)**2)))
        pilot = abs(np.vdot(carrier, part) / WINDOW)
        # Integrated two-sided complex-IQ periodogram. The Hann normalization
        # makes the sum an estimate of average waveform power, not ASD units.
        spectrum = np.fft.fft(part * hann)
        band_power = np.abs(spectrum)**2 / (WINDOW * np.sum(hann**2))
        rows = {'window_index': i, 'start_sample': start, 'stop_sample_exclusive': start+WINDOW,
                'samples': WINDOW, 'source_array_sha256': hashlib.sha256(np.asarray(iq[start:start+WINDOW]).tobytes()).hexdigest(),
                'finite': finite, 'total_rms': scalar(total_rms),
                'coherent_pilot_amplitude': scalar(pilot)}
        for name, mask in masks.items():
            rows[name+'_reference_band_power'] = scalar(np.sum(band_power[mask]))
        windows.append(rows)
    checks = {}
    for name, limit in SPAN_LIMITS_DB.items():
        vals = [row[name] for row in windows]
        valid = all(value is not None and value > 0 for value in vals)
        factor = 20 if name in ('total_rms','coherent_pilot_amplitude') else 10
        span = factor * math.log10(max(vals)/min(vals)) if valid else None
        checks[name] = {'max_min_span_db': span, 'limit_db': limit,
                        'passed': valid and span <= limit,
                        'minimum': min(vals) if valid else None,
                        'maximum': max(vals) if valid else None}
    return {'schema': 'fine-waveform-stationarity-v1', 'input_iq': str(path), 'input_sha256': sha(path),
            'sample_rate_hz': FS, 'qualification_interval_half_open': [0,WINDOW_COUNT*WINDOW],
            'number_of_disjoint_windows': WINDOW_COUNT, 'samples_per_window': WINDOW,
            'excluded_tail_samples': RETAINED-WINDOW_COUNT*WINDOW,
            'exclusion_rule': 'Remainder after the predeclared maximal count of disjoint full-frame-length windows; not selected from outcomes.',
            'reference_band_fft_bin_counts': {name:int(mask.sum()) for name,mask in masks.items()},
            'windows': windows, 'checks': checks,
            'stationarity_engineering_passed': all(row['finite'] for row in windows) and all(c['passed'] for c in checks.values()),
            'scope': 'Pre-PFB complex-IQ engineering consistency only; no physical stationarity or statistical coverage claim. Disjoint time windows from one deterministic payload are not independent waveform seeds.'}


def run(args):
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Preparation output must be a new directory')
    old = RAIL/'results/channel29_residual_controls_2026-09-09/preflight'
    exclusions = old/'seed-exclusions.npz'
    used_controls = old/'planned-frame-seeds.npz'
    with np.load(exclusions,allow_pickle=False) as z:
        forbidden = set(int(v) for v in z['raw_seed'])
        forbidden63 = set(int(v) for v in z['effective_seed63'])
        forbidden32 = set(int(v) for v in z['conservative_seed32'])
    with np.load(used_controls,allow_pickle=False) as z:
        controls = [int(v) for v in z['raw_seed']]
        forbidden.update(controls)
        forbidden63.update(int(v) for v in z['effective_seed63'])
        forbidden32.update(v & ((1<<32)-1) for v in controls)
    seeds = [int.from_bytes(hashlib.sha256(f'fine-detector-validation-v1:payload:{i}'.encode()).digest()[:4],'big') for i in range(4)]
    if len(set(seeds)) != 4 or any(s in forbidden or s in forbidden63 or s in forbidden32 for s in seeds):
        raise ValueError('Proposed payload seeds collide; freeze a reviewed new namespace before generation')
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 2_000_000_000:
        raise ValueError('Need at least 2GB free before waveform preparation')
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    probe = "import json,sys,numpy;from gnuradio import gr;print(json.dumps({'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'gnuradio':gr.version()}))"
    runtime = json.loads(subprocess.check_output([args.gnuradio_python,'-c',probe],env=env,text=True))
    out.mkdir(parents=True)
    snap = out/'source_snapshots/pilot-proxy'
    sources = sorted((REPO/'src/pilot_proxy').rglob('*.py')) + [REPO/'tools/generate_settled_atsc.py',Path(__file__).resolve()]
    snapshots = {}
    for source in sources:
        target = snap/source.relative_to(REPO)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        snapshots[str(target.relative_to(out))] = sha(target)
    plan = {'schema':'fine-waveform-preparation-plan-v1','frozen_utc':utc(),
            'generation_runtime':runtime,'qualification_runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__},
            'source_snapshots_sha256':snapshots,
            'seed_recipe':'First4 SHA256 bytes big-endian of UTF8 fine-detector-validation-v1:payload:{index}, indices0..3. No retries.',
            'payload_seeds':seeds,'seed_exclusion_inputs':{str(exclusions):sha(exclusions),str(used_controls):sha(used_controls)},
            'seed_exclusion_counts':{'raw':len(forbidden),'effective63':len(forbidden63),'conservative32':len(forbidden32)},
            'seed_collision_check_passed':True,'new_detector_noise_draws':0,
            'retained_samples_per_payload':RETAINED,'discard_samples':DISCARD,'ts_packets':PACKETS,
            'sample_rate_hz':FS,'symbol_rate_hz':FS,'analytic_generator_pilot_baseband_hz':PILOT,
            'qualification':{'method':'Independent NumPy read of predeclared disjoint full-frame-length IQ windows; RMS, known-frequency coherent pilot amplitude and Hann-integrated lower/upper reference-band power.',
                'window_samples':WINDOW,'window_count':WINDOW_COUNT,'windows':[[i*WINDOW,(i+1)*WINDOW] for i in range(WINDOW_COUNT)],
                'qualified_interval_half_open':[0,WINDOW_COUNT*WINDOW],'unused_tail_samples':RETAINED-WINDOW_COUNT*WINDOW,
                'reference_width_hz':REFERENCE_WIDTH,'reference_offsets_from_pilot_hz':[-REFERENCE_DISTANCE,REFERENCE_DISTANCE],
                'max_min_span_limits_db':SPAN_LIMITS_DB,
                'tolerance_rationale':'Engineering consistency tolerances frozen before generation:0.25dB total RMS and1dB pilot amplitude limit gross gain/startup drift;2dB reference-band power permits finite-band spectral variation across8 windows (about128 Fourier bins per arm), while catching order-unity drift. These are not a confidence guarantee or fitted physical tolerances.',
                'existing_broad_waveform_audit':'Unmodified generate_settled_atsc audit/receipt retained; its spectral quality check uses first262144 retained samples.',
                'failure_rule':'Record every payload and failed metric. No trimming, seed replacement, window reselection or tolerance tuning after outcomes.',
                'reference_scope':'Pre-PFB spectral sidebands only. Not actual per-channel int4 matched-filter reference projections; qualify those separately in per-channel caches.'},
            'scope':'Four deterministic digital payload fixtures with explicit startup discard. Qualification does not establish physical RF stationarity, waveform-population generalization, or independent detector evaluation.'}
    write(out/'plan.json',plan)
    print(json.dumps({'frozen_plan_sha256':sha(out/'plan.json'),'payload_seeds':seeds,'window_samples':WINDOW,'window_count':WINDOW_COUNT}),flush=True)
    results=[]
    wrapper=snap/'tools/generate_settled_atsc.py'
    for index,seed in enumerate(seeds):
        dest=out/f'payload-{index:02d}'
        command=[sys.executable,str(wrapper),'--output-dir',str(dest),'--num-iq-samples',str(RETAINED),
                 '--discard-samples',str(DISCARD),'--num-ts-packets',str(PACKETS),'--seed',str(seed),
                 '--gnuradio-python',args.gnuradio_python]
        start=utc()
        with (out/f'payload-{index:02d}.log').open('x') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'index':index,'seed':seed,'started_utc':start,'completed_utc':utc(),'command':command,'returncode':proc.returncode}
        if (dest/'receipt.json').exists():
            receipt=json.loads((dest/'receipt.json').read_text())
            for filename,key in [('settled.cfile','settled_iq_sha256'),('as-generated.cfile','raw_iq_sha256'),
                                 ('plan.json','plan_sha256'),('waveform-audit.json','waveform_audit_sha256'),
                                 ('transport-stream.ts','transport_stream_sha256')]:
                if sha(dest/filename)!=receipt[key]:
                    raise ValueError('Generated artifact receipt mismatch:'+filename)
            stationarity=qualify(dest/'settled.cfile')
            write(dest/'stationarity.json',stationarity)
            record.update(waveform_quality_passed=receipt['waveform_quality_passed'],
                          stationarity_engineering_passed=stationarity['stationarity_engineering_passed'],
                          exact_trim_verified=receipt['exact_trim_verified'],
                          qualified=proc.returncode==0 and receipt['waveform_quality_passed'] and stationarity['stationarity_engineering_passed'],
                          settled_iq=str(dest/'settled.cfile'),settled_iq_sha256=sha(dest/'settled.cfile'),
                          stationarity_sha256=sha(dest/'stationarity.json'),generation_receipt_sha256=sha(dest/'receipt.json'),
                          qualification_interval_half_open=[0,WINDOW_COUNT*WINDOW],window_samples=WINDOW,window_count=WINDOW_COUNT)
        else:
            record.update(qualified=False,reason='Generation receipt absent; inspect retained log. No replacement attempted.')
        results.append(record)
        write(out/f'payload-{index:02d}-completion.json',record)
        print(json.dumps({'payload':index,'returncode':proc.returncode,'qualified':record['qualified']}),flush=True)
    for name,expected in snapshots.items():
        if sha(out/name)!=expected:
            raise ValueError('Frozen generating source changed:'+name)
    summary={'schema':'fine-waveform-preparation-result-v1','completed_utc':utc(),'plan_sha256':sha(out/'plan.json'),
             'payloads':results,'all_four_qualified':all(row['qualified'] for row in results),
             'qualified_payload_count':sum(row['qualified'] for row in results),
             'scope':plan['scope'],'seed_history_note':'The four payload seeds and all reserved historical control identities must be excluded from later noise/evaluation allocations.'}
    write(out/'summary.json',summary)
    write(out/'manifest.json',{'schema':'fine-waveform-preparation-manifest-v1','files':{
        str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()}})
    print(json.dumps({'all_four_qualified':summary['all_four_qualified'],'output':str(out)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gnuradio-python',default='/usr/bin/python3')
    run(parser.parse_args())
