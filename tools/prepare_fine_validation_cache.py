#!/usr/bin/env python3
"""Freeze and build phase-preserving full-reference-PFB fine-validation fixtures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAIL = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))

from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.detector_reference import unpack_packed_complex
from pilot_proxy.detector_geometry import stream_time_block_to_detector_matrix
from pilot_proxy.reference_channelizer import (
    ReferenceChannelizerSpec, complex_envelope_to_real_adc_blocks,
    complex_envelope_to_real_adc_blocks_gpu,
    channelize_real_blocks_to_reference_channels,
    channelize_real_blocks_to_reference_channels_gpu,
    sinc_hamming_pfb_response,
)

FS = 4_500_000.0/286.0*684.0
COARSE_FS = 390625.0
FRAME = 16384
K = 128
FINE = 256
STEP = Fraction(390625,32768)
WINDOW = math.ceil(((FRAME+3)*2048-1)*FS/800e6)+1
PILOT_MODEL = -3e6+(6e6-FS/2)/2
PILOT_NOMINAL = -3e6+309441.0
ROLE = {0:'calibration_engineering',1:'reference_variation_stress',2:'primary_evaluation',3:'primary_evaluation'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def runtime(backend):
    result={'python':platform.python_version(),'python_executable':sys.executable,'numpy':np.__version__,
            'platform':platform.platform(),'backend':backend}
    if backend=='gpu':
        import cupy as cp
        prop=cp.cuda.runtime.getDeviceProperties(0)
        name=prop['name']; name=name.decode() if isinstance(name,bytes) else name
        result.update(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      device_name=name,compute_capability=f"{prop['major']}{prop['minor']}")
    return result


def geometry_cases(layout):
    """Exact rational frequency design, independent of any generated signal."""
    delta=Fraction(str(layout['dtv_pilot_hz']))-Fraction(str(layout['coarse_channel_center_hz']))
    nominal_k=round(delta/STEP)
    align=nominal_k*STEP-delta
    definitions=[('nominal',Fraction(0)),('aligned',align),('half',align+STEP/2),
                 ('cfo_m1000',Fraction(-1000)),('cfo_p1000',Fraction(1000)),
                 ('cfo_m1400',Fraction(-1400)),('cfo_p1400',Fraction(1400))]
    out=[]
    for name,cfo in definitions:
        coordinate=(delta+cfo)/STEP
        # A half-bin tie uses the lower aligned bin, frozen before any IQ is read.
        chosen=nominal_k if name=='half' else round(coordinate)
        out.append({'name':name,'actual_cfo_hz':float(cfo),
                    'actual_cfo_fraction':[cfo.numerator,cfo.denominator],
                    'carrier_minus_coarse_center_hz':float(delta+cfo),
                    'continuous_unwrapped_fine_coordinate':float(coordinate),
                    'continuous_wrapped_fine_coordinate':float(coordinate%256),
                    'calibrated_anchor_bin_normal':int(chosen%256),
                    'calibrated_anchor_bin_inverted':int((-chosen)%256),
                    'nominal_anchor_bin_normal':int(nominal_k%256),
                    'nominal_anchor_bin_inverted':int((-nominal_k)%256),
                    'fractional_offset_from_chosen_anchor_bins':float(coordinate-chosen),
                    'anchor_rule':'Known declared synthetic carrier; nearest integer, except half uses the lower aligned integer. No evaluation peak fitting.',
                    'stress_rule':'Replay these same powers with unchanged nominal anchor; no additional cache or recentering.'})
    return out


def freeze(args):
    out=args.output.resolve()
    if out.exists(): raise ValueError('Cache output must be a new directory')
    prep=args.preparation.resolve()
    preparation=json.loads((prep/'summary.json').read_text())
    manifest=json.loads((prep/'manifest.json').read_text())
    if preparation['plan_sha256']!=sha(prep/'plan.json'):
        raise ValueError('Preparation plan binding differs')
    for relative,expected in manifest['files'].items():
        if sha(prep/relative)!=expected: raise ValueError('Preparation manifest mismatch:'+relative)
    payloads=[]
    for record in preparation['payloads']:
        index=record['index']; path=Path(record['settled_iq'])
        if sha(path)!=record['settled_iq_sha256']:
            raise ValueError('Waveform differs')
        if index in (0,2,3) and record['qualified'] is not True:
            raise ValueError('Primary payload failed preparation qualification')
        if index==1 and record['qualified'] is not False:
            raise ValueError('Variation fixture classification differs; review the design')
        payloads.append({'index':index,'role':ROLE[index],'path':str(path),'sha256':sha(path),
                         'qualified':record['qualified'],'window_index':0,'window_start':0,'window_stop_exclusive':WINDOW,
                         'preparation_interval':record['qualification_interval_half_open']})
    weights=args.weights.resolve(); adjacent=weights.with_suffix(weights.suffix+'.manifest.json')
    bank=DetectorWeightBank(explicit_path=weights)
    channels=bank.supported_physical_channels()
    if set(channels)!=set(range(14,37)): raise ValueError('Expected all23 weight profiles')
    profiles=[]
    for channel in range(14,37):
        layout=bank.layout_for_physical_channel(channel)
        packed,valid=bank.get_weights_for_physical_channel(channel)
        if not valid or packed is None: raise ValueError('Invalid weight profile')
        unpacked=unpack_packed_complex(packed,4,dtype=np.float64)
        norms=np.sum(np.abs(unpacked)**2,axis=1)
        if int(round(norms[0]))!=layout['target_norm_sq'] or int(round(norms[1]+norms[2]))!=layout['reference_norm_sum_sq']:
            raise ValueError('Packed coefficient norms disagree with manifest')
        reference_index=int(round((float(layout['coarse_channel_center_hz'])-400e6)/COARSE_FS))-1
        if 400e6+(reference_index+1)*COARSE_FS!=layout['coarse_channel_center_hz']:
            raise ValueError('Reference index centre differs from actual weight profile')
        profiles.append({'physical_channel':channel,'layout':dict(layout),'reference_channel_index':reference_index,
                         'packed_profile_sha256':hashlib.sha256(np.asarray(packed).tobytes()).hexdigest(),
                         'cases':geometry_cases(layout)})
    # These calibrate a deterministic digital normalization only; they are not
    # independent detector-noise trials or physical thermal measurements.
    gain_seeds={str(c):int.from_bytes(hashlib.sha256(f'fine-validation-pfb-gain-v1:{c}'.encode()).digest()[:8],'big') for c in range(14,37)}
    prior=RAIL/'results/channel29_residual_controls_2026-09-09/preflight'
    with np.load(prior/'seed-exclusions.npz',allow_pickle=False) as z:
        raw=set(int(x) for x in z['raw_seed']); eff=set(int(x) for x in z['effective_seed63'])
    with np.load(prior/'planned-frame-seeds.npz',allow_pickle=False) as z:
        raw.update(int(x) for x in z['raw_seed']); eff.update(int(x) for x in z['effective_seed63'])
    payload_seeds=json.loads((prep/'plan.json').read_text())['payload_seeds']
    raw.update(payload_seeds);eff.update(payload_seeds)
    values=list(gain_seeds.values())
    if len(set(values))!=23 or any(x in raw or (x&((1<<63)-1)) in eff for x in values):
        raise ValueError('PFB normalization seed collision')
    sources={str(Path(__file__).resolve()):sha(__file__)}
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source and Path(source).suffix=='.py' and Path(source).is_relative_to(ROOT):
            sources[str(Path(source).resolve())]=sha(source)
    out.mkdir(parents=True)
    for source in sources:
        target=out/'source_snapshots/pilot-proxy'/Path(source).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    plan={'schema':'fine-validation-pfb-cache-plan-v1','frozen_utc':utc(),'runtime':runtime(args.backend),
          'source_sha256':sources,'preparation':str(prep),'preparation_plan_sha256':sha(prep/'plan.json'),
          'preparation_manifest_sha256':sha(prep/'manifest.json'),'preparation_summary_sha256':sha(prep/'summary.json'),
          'weights_path':str(weights),'weights_sha256':sha(weights),'weight_manifest_sha256':sha(adjacent),
          'profiles':profiles,'payloads':payloads,'cache_count':644,
          'window_index':0,'window_input_samples':WINDOW,'output_samples':FRAME,
          'pfb':{'real_adc_sample_rate_hz':800e6,'fft_size':2048,'taps':4,'response':'sinc_hamming_pfb_response checked-in implementation',
                 'source_iq_sample_rate_hz':FS,'output_sample_rate_hz':COARSE_FS,'input_interpolation':'piecewise linear as implemented by full reference ADC model'},
          'analytic_generator_pilot_baseband_hz':PILOT_MODEL,'nominal_pilot_baseband_hz':PILOT_NOMINAL,
          'rf_center_correction_hz':PILOT_NOMINAL-PILOT_MODEL,
          'fine_bin_spacing_hz':float(STEP),'unpadded_bin_spacing_hz':float(2*STEP),
          'gain_seeds':gain_seeds,'gain_seed_recipe':'First8 SHA256 bytes big-endian of UTF8 fine-validation-pfb-gain-v1:{physical_channel}; NumPy PCG64 full64 seed.',
          'gain_normalization':'One deterministic unit-variance circular complex Gaussian IQ sequence per channel, full reference PFB at corrected nominal RF centre. Divide all clean/tone streams of that channel by sqrt(mean(abs(noise_stream)**2)). Reused conditional normalization, not a fresh frame-noise trial.',
          'coordinate_system':'Already normalized, centred reference-rFFT complex samples. No raw-archive phase convention or input reversal applied here.',
          'tone_rule':'Project normalized full16384 clean stream onto exp(2pi i actual_RF_minus_coarse_center*n/Fs); use resulting complex coefficient times the same continuous-frequency carrier across all16384 samples. Finite-frame data leakage into this coefficient is retained and labelled.',
          'pilot_frequency_rule':'Only analytic generator-vs-nominal correction and predeclared CFO; no spectrum-derived frequency retuning.',
          'engineering_gates':{'finite_arrays':True,'exact_array_shapes':True,'positive_normalization':True,'positive_projected_pilot_amplitude':True,
                               'clean_peak_max_circular_distance_from_calibrated_anchor_bins':2,
                               'pure_tone_peak_max_circular_distance_from_calibrated_anchor_bins':1},
          'limits':['Calibrated anchors use known synthetic offsets; this does not validate estimating unknown transmitter offsets.',
                    'Fixed nominal-anchor stress is a separate replay of the same powers.',
                    'Payload01 failed pre-PFB reference-band engineering stationarity; it is outside the primary qualified family.',
                    'No physical residual, measured CHIME PFB or independent telescope validation is supplied.'],
          'case_order':'channel ascending, each profile case list order, payload index ascending',
          'existing_failures_policy':'Write and report failed caches; no hidden replacement or retuning. Complete failure-free status is required for a primary cache-ready claim.'}
    write(out/'plan.json',plan)
    print(json.dumps({'plan_sha256':sha(out/'plan.json'),'cache_count':644,'gain_seeds':gain_seeds}),flush=True)


def verify_plan(out):
    plan=json.loads((out/'plan.json').read_text())
    for source,expected in plan['source_sha256'].items():
        if sha(source)!=expected: raise ValueError('Generating source changed:'+source)
    if sha(plan['weights_path'])!=plan['weights_sha256']:
        raise ValueError('Weight bank changed')
    if sha(Path(plan['weights_path']).with_suffix('.bin.manifest.json'))!=plan['weight_manifest_sha256']:
        raise ValueError('Weight manifest changed')
    prep=Path(plan['preparation'])
    for name,key in [('plan.json','preparation_plan_sha256'),('manifest.json','preparation_manifest_sha256'),('summary.json','preparation_summary_sha256')]:
        if sha(prep/name)!=plan[key]: raise ValueError('Preparation identity changed')
    if runtime(plan['runtime']['backend'])!=plan['runtime']:
        raise ValueError('Cache execution runtime changed')
    return plan


def channelize(iq,rf_center,reference_index,backend):
    builder=complex_envelope_to_real_adc_blocks_gpu if backend=='gpu' else complex_envelope_to_real_adc_blocks
    pfb=channelize_real_blocks_to_reference_channels_gpu if backend=='gpu' else channelize_real_blocks_to_reference_channels
    blocks=builder(iq,iq_sample_rate_hz=FS,rf_center_hz=rf_center,adc_sample_rate_hz=800e6,
                   band_lower_hz=400e6,n_blocks=FRAME+3,block_size=2048)
    return pfb(blocks,channel_indices=[reference_index],response=sinc_hamming_pfb_response(4,2048),
               spec=ReferenceChannelizerSpec())[0]


def gain(out,plan,profile):
    directory=out/'normalization';directory.mkdir(exist_ok=True)
    channel=profile['physical_channel'];path=directory/f'ch{channel:02d}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['plan_sha256']!=sha(out/'plan.json'):raise ValueError('Cached gain plan mismatch')
        if sha(directory/saved['array_file'])!=saved['array_sha256']:raise ValueError('Cached gain array mismatch')
        with np.load(directory/saved['array_file'], allow_pickle=False) as archive:
            recounted = float(np.mean(np.abs(archive['noise_stream'].astype(np.complex128))**2))
        if recounted != saved['power']: raise ValueError('Cached gain value differs from saved noise stream')
        return saved['power']
    rng=np.random.default_rng(plan['gain_seeds'][str(channel)])
    # Draw order and dtype are part of the frozen builder identity.
    iq=(rng.standard_normal(WINDOW,dtype=np.float32)+1j*rng.standard_normal(WINDOW,dtype=np.float32))/math.sqrt(2)
    centre=profile['layout']['dtv_pilot_hz']+3e6-309441.0+plan['rf_center_correction_hz']
    clean=channelize(np.asarray(iq,dtype=np.complex64),centre,profile['reference_channel_index'],plan['runtime']['backend'])
    power=float(np.mean(np.abs(clean.astype(np.complex128))**2))
    if not math.isfinite(power) or power<=0:raise ValueError('Invalid PFB gain')
    array=directory/f'ch{channel:02d}.npz'
    np.savez_compressed(array,noise_stream=clean)
    write(path,{'plan_sha256':sha(out/'plan.json'),'power':power,'seed':plan['gain_seeds'][str(channel)],
                'array_file':array.name,'array_sha256':sha(array),'normalization_scope':plan['gain_normalization']})
    return power


def generate(args):
    out=args.output.resolve();plan=verify_plan(out);bank=DetectorWeightBank(explicit_path=plan['weights_path'])
    cases=[(profile,case,payload) for profile in plan['profiles'] for case in profile['cases'] for payload in plan['payloads']]
    indices=list(range(len(cases))) if args.case_index is None else args.case_index
    if len(indices)!=len(set(indices)) or any(i<0 or i>=len(cases) for i in indices):
        raise ValueError('Invalid requested case indices')
    directory=out/'cases';directory.mkdir(exist_ok=True)
    import time
    for index in indices:
        profile,case,payload=cases[index];stem=f'{index:04d}_ch{profile["physical_channel"]}_{case["name"]}_p{payload["index"]:02d}'
        path=directory/(stem+'.npz');receipt=directory/(stem+'.json')
        if path.exists() or receipt.exists():
            if not path.exists() or not receipt.exists():raise ValueError('Incomplete cache; investigate before retry')
            previous=json.loads(receipt.read_text())
            if previous['plan_sha256']!=sha(out/'plan.json') or previous['array_sha256']!=sha(path):raise ValueError('Existing cache binding differs')
            with np.load(path, allow_pickle=False) as archive:
                embedded = json.loads(str(archive['meta_json'].item()))
            if any(previous.get(key) != value for key, value in embedded.items()):
                raise ValueError('Existing cache receipt differs from embedded metadata')
            print(json.dumps({'index':index,'status':'verified_existing','passed':previous['passed']}),flush=True)
            continue
        started=time.monotonic();source=Path(payload['path'])
        if sha(source)!=payload['sha256']:raise ValueError('Payload identity differs')
        with source.open('rb') as stream:
            stream.seek(payload['window_start']*8);iq=np.fromfile(stream,dtype=np.complex64,count=WINDOW)
        phase=np.exp(2j*np.pi*case['actual_cfo_hz']*np.arange(WINDOW,dtype=np.float64)/FS)
        shifted=np.asarray(iq*phase,dtype=np.complex64)
        layout=profile['layout'];centre=layout['dtv_pilot_hz']+3e6-309441.0+plan['rf_center_correction_hz']
        power=gain(out,plan,profile)
        clean=np.asarray(channelize(shifted,centre,profile['reference_channel_index'],plan['runtime']['backend'])/math.sqrt(power),dtype=np.complex64)
        carrier=np.exp(2j*np.pi*case['carrier_minus_coarse_center_hz']*np.arange(FRAME,dtype=np.float64)/COARSE_FS)
        amplitude=np.vdot(carrier,clean.astype(np.complex128))/FRAME
        tone=np.asarray(amplitude*carrier,dtype=np.complex64)
        rows=stream_time_block_to_detector_matrix(clean[None,:],detector_window_samples=K)
        tone_rows=stream_time_block_to_detector_matrix(tone[None,:],detector_window_samples=K)
        packed,valid=bank.get_weights_for_physical_channel(profile['physical_channel'])
        if not valid:raise ValueError('Invalid weights')
        weight=unpack_packed_complex(packed,4,dtype=np.float64)/7
        clean_columns=rows.astype(np.complex128)@np.conjugate(weight).T
        tone_columns=tone_rows.astype(np.complex128)@np.conjugate(weight).T
        clean_peak=int(np.argmax(np.abs(np.fft.fft(clean_columns[:,0],n=FINE))))
        tone_peak=int(np.argmax(np.abs(np.fft.fft(tone_columns[:,0],n=FINE))))
        anchor=case['calibrated_anchor_bin_normal']
        distance=lambda v:min((v-anchor)%FINE,(anchor-v)%FINE)
        gates={'finite':bool(np.isfinite(clean).all() and np.isfinite(tone).all()),
               'shape':clean.shape==(FRAME,) and tone.shape==(FRAME,) and rows.shape==(128,128) and tone_rows.shape==(128,128),
               'positive_amplitude':bool(np.isfinite(amplitude) and abs(amplitude)>0),
               'clean_peak_position':distance(clean_peak)<=2,'tone_peak_position':distance(tone_peak)<=1}
        metadata={'plan_sha256':sha(out/'plan.json'),'case_index':index,'payload':payload,'profile_channel':profile['physical_channel'],
                  'case':case,'pfb_noise_gain':power,'rf_center_hz':centre,'packed_profile_sha256':profile['packed_profile_sha256'],
                  'coherent_line_amplitude_real':float(amplitude.real),'coherent_line_amplitude_imag':float(amplitude.imag),
                  'clean_mean_power':float(np.mean(np.abs(clean.astype(np.complex128))**2)),
                  'clean_coarse_projection_powers':np.mean(np.abs(clean_columns)**2,axis=0).tolist(),
                  'tone_coarse_projection_powers':np.mean(np.abs(tone_columns)**2,axis=0).tolist(),
                  'clean_fine_peak_bin':clean_peak,'tone_fine_peak_bin':tone_peak,
                  'clean_peak_circular_distance':distance(clean_peak),'tone_peak_circular_distance':distance(tone_peak),
                  'coordinate_system':plan['coordinate_system'],'tone_scope':plan['tone_rule'],'gates':gates,'passed':all(gates.values())}
        np.savez_compressed(path,clean_stream=clean,atsc_rows=rows,ideal_tone_stream=tone,
                            ideal_tone_rows=tone_rows,packed_weights=packed,meta_json=np.asarray(json.dumps(metadata,sort_keys=True)))
        receipt_data={**metadata,'array_file':path.name,'array_sha256':sha(path),'elapsed_seconds':time.monotonic()-started,'completed_utc':utc()}
        write(receipt,receipt_data)
        print(json.dumps({'index':index,'passed':metadata['passed'],'elapsed_seconds':receipt_data['elapsed_seconds']}),flush=True)
    verify_plan(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','generate'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,default=RAIL/'results/fine_detector_validation_2026-09-09/preparation')
    parser.add_argument('--weights',type=Path,default=ROOT/'weights/chime_dtv_weights_k128.bin')
    parser.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--case-index',type=int,action='append')
    arguments=parser.parse_args()
    freeze(arguments) if arguments.stage=='freeze' else generate(arguments)
