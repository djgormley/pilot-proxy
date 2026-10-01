#!/usr/bin/env python3
"""Frozen CPU GNU Radio distribution study through the unchanged SDR adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import scipy
from scipy.stats import f, ncf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pilot_proxy.integration.sdr_upgrade_adapter import (  # noqa: E402
    DigitalAdapterConfig, adapt_record,
)
from pilot_proxy.detector_reference import unpack_packed_complex  # noqa: E402

STAGES = ('natural_float', 'quantized_weights_float', 'packed')
CONFIG = DigitalAdapterConfig(2_000_000, 100_000., 0, 5.)
RECORDS = 24
SAMPLES = 4_000_000
AMPLITUDE = .1171875


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def sources():
    return [Path(__file__).resolve(), ROOT/'tools/generate_gnuradio_reference.py',
            ROOT/'src/pilot_proxy/integration/sdr_upgrade_adapter.py',
            ROOT/'src/pilot_proxy/detector_reference.py']


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    records = [{'case': case, 'record': i,
                'seed': 1_409_000_000+1000*i+(17 if case == 'noise' else 113)}
               for case in ('noise', 'steady') for i in range(RECORDS)]
    plan = {
        'schema': 'sdr-distribution-reference-plan-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Fresh CPU software reference; no SDR operation, physical calibration or full-array validation',
        'records': records, 'samples_per_record': SAMPLES,
        'frames_per_record': 47, 'frames_per_case': 1128,
        'input_complex_noise_variance': 1., 'signal_amplitude': AMPLITUDE,
        'generator_lambda_per_raw_128_sample_window': 2*128*AMPLITUDE**2,
        'ideal_output_sample_variance': 25/128, 'ideal_row_noise_power': 25.,
        'ideal_frame_noncentrality': 2304., 'ideal_dof': [256, 512],
        'ideal_model': 'Proper complex white Gaussian noise through ideal full-output-Nyquist brickwall, unquantized orthogonal natural weights; steady tone exactly centered after mixing',
        'adapter_config': {'input_rate_hz': 2_000_000, 'mixer_hz': 100_000., 'target_bin': 0, 'sample_scale': 5.},
        'frame_samples': 16384, 'K': 128, 'L': 128, 'inputs': 1,
        'stages': list(STAGES),
        'noise_histogram_edges': np.linspace(.6, 1.6, 41).tolist(),
        'steady_histogram_edges': np.linspace(5, 16, 45).tolist(),
        'interpretation': 'Descriptive distribution comparison; no formal F-law or physical null acceptance; dependent within-record frames are not independent confidence replicates',
        'statistics': ['frame mean, median, sample standard deviation, 0.05/0.50/0.95 quantiles',
                       'PDF bin counts normalized by all valid frame observations, with tails explicit',
                       'record-level means and spreads; same-record lag-one frame ratio correlation',
                       'descriptive row projection means, proper/pseudo covariance and adjacent-row covariance; steady covariance includes deterministic GNU Radio mean drift',
                       'unmodified source means/variances; quantizer clipping and invalid ratios'],
        'acceptance': 'Fixed finite source counts, source/hash identities, finite output and exact packed projection/power recount; distribution discrepancies are measured, never tuned away',
        'failure_policy': 'Stop on generation/integrity error; preserve failed records, no seed replacements or sample extension',
        'source_sha256': {str(p): sha(p) for p in sources()},
        'analysis_runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'campaign_modified': False, 'gpu_used': False, 'hardware_used': False,
    }
    write_json(out/'plan.json', plan)
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    for p in sources():
        target = out/'source_snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    print('Frozen 48 independently seeded records; 1128 complete frames per case.', flush=True)


def verify_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if sha(out/'plan.json') != (out/'plan.sha256').read_text().strip():
        raise ValueError('Plan hash differs')
    for name, digest in plan['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen source changed: '+name)
    if plan['adapter_config'] != vars(CONFIG):
        raise ValueError('Adapter configuration differs')
    return plan


def complex_json(value):
    value = np.asarray(value)
    return {'real': value.real.tolist(), 'imag': value.imag.tolist()}


def powers_and_ratio(z):
    powers = np.sum(abs(z)**2, axis=2)
    denominator = powers[:, 1]+powers[:, 2]
    if not np.all(denominator > 0):
        raise ValueError('Zero reference power; preserve failed record')
    return powers, 2*powers[:, 0]/denominator


def packed_recount(result):
    """Independent int64 nibble arithmetic (no projection or unpack helper)."""
    x = result['packed_frames'].astype(np.uint8)
    w = result['packed_weights'].astype(np.uint8)
    def decode(a):
        return np.where(a >= 8, a.astype(np.int64)-16, a.astype(np.int64))
    xr, xi = decode(x >> 4), decode(x & 15)
    wr, wi = decode(w >> 4), decode(w & 15)
    re = np.einsum('frk,tk->ftr', xr, wr)+np.einsum('frk,tk->ftr', xi, wi)
    im = np.einsum('frk,tk->ftr', xi, wr)-np.einsum('frk,tk->ftr', xr, wi)
    check = np.stack([re, im], axis=-1)
    if not np.array_equal(check, result['projections_i32']):
        raise ValueError('Independent packed projection mismatch')
    power = np.sum(re*re+im*im, axis=2)
    if not np.array_equal(power, result['term_power_sums']):
        raise ValueError('Independent integer frame power mismatch')
    return int(re.size)


def run(out):
    plan = verify_plan(out)
    if (out/'run.json').exists():
        raise ValueError('Study already attempted; no implicit retry or overwrite')
    write_json(out/'run.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                              'plan_sha256': sha(out/'plan.json')})
    (out/'records').mkdir()
    generator = ROOT/'tools/generate_gnuradio_reference.py'
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    all_records = []
    for spec in plan['records']:
        case, index = spec['case'], spec['record']
        folder = out/'records'/f'{case}-{index:02d}'
        mode, lam = ('noise', 0.) if case == 'noise' else ('tone', plan['generator_lambda_per_raw_128_sample_window'])
        prepare = ['/usr/bin/python3', str(generator), 'prepare', '--output', str(folder),
                   '--mode', mode, '--seed', str(spec['seed']), '--lambda', str(lam), '--samples', str(SAMPLES)]
        generate = ['/usr/bin/python3', str(generator), 'generate', '--output', str(folder)]
        for command in (prepare, generate):
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout+completed.stderr)
        receipt = json.loads((folder/'receipt.json').read_text())
        if not receipt['success'] or not receipt['saved_addition_exact']:
            raise ValueError('GNU Radio source acceptance failed')
        iq = np.fromfile(folder/'iq.cfile', dtype='<c8')
        result = adapt_record(iq, CONFIG)
        if result['metadata']['frame_count'] != plan['frames_per_record']:
            raise ValueError('Complete frame count differs')
        projections_checked = packed_recount(result)
        used = result['resampled_frames']
        natural = np.einsum('frk,tk->ftr', used, result['natural_weights'].conj())
        weights = unpack_packed_complex(result['packed_weights'], 4)/7
        weight_float = np.einsum('frk,tk->ftr', used, weights.conj())
        packed = (result['projections_i32'][..., 0]+1j*result['projections_i32'][..., 1])/(CONFIG.sample_scale*7)
        arrays = {name: z for name, z in zip(STAGES, (natural, weight_float, packed))}
        arrays.update({name+'_ratio': powers_and_ratio(z)[1] for name, z in list(arrays.items())})
        if not np.allclose(arrays['packed_ratio'], result['coarse_ratio'], rtol=1e-14, atol=1e-14):
            raise ValueError('Packed normalization changes ratio')
        arrays['frame_start_seconds'] = result['frame_start_seconds']
        np.savez_compressed(folder/'projections.npz', **arrays)
        record = {**spec, 'directory': str(folder.relative_to(out)),
                  'generator_receipt_sha256': sha(folder/'receipt.json'),
                  'projection_sha256': sha(folder/'projections.npz'),
                  'input_iq_sha256': sha(folder/'iq.cfile'),
                  'independent_complex_projections_checked': projections_checked,
                  'source_noise_mean_power': receipt['noise_mean_power'],
                  'source_signal_mean_power': receipt['signal_mean_power'],
                  'adapter_metadata': result['metadata']}
        write_json(folder/'analysis-receipt.json', record)
        all_records.append(record)
        print(f'{case} {index+1}/{RECORDS}: 47 frames; {projections_checked} exact projections', flush=True)
    write_json(out/'records.json', all_records)
    verify_plan(out)
    print('All generation/adapter records complete.', flush=True)


def describe(values):
    return {'count': len(values), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'std_ddof1': float(np.std(values, ddof=1)),
            'quantiles_05_50_95': np.quantile(values, [.05, .5, .95]).tolist()}


def report(out):
    plan = verify_plan(out)
    records = json.loads((out/'records.json').read_text())
    if [(x['case'], x['record'], x['seed']) for x in records] != [(x['case'], x['record'], x['seed']) for x in plan['records']]:
        raise ValueError('Record coverage differs')
    summary = {'schema': 'sdr-distribution-reference-summary-v1', 'plan_sha256': sha(out/'plan.json'),
               'cases': {}, 'hardware_qualified': False, 'campaign_modified': False,
               'distribution_acceptance_claim': False, 'independent_record_count_per_case': RECORDS,
               'frame_samples': 16384, 'K': 128, 'L': 128, 'inputs': 1,
               'total_complex_integer_projections_recounted': sum(x['independent_complex_projections_checked'] for x in records)}
    for case in ('noise', 'steady'):
        members = [r for r in records if r['case'] == case]
        cubes = {stage: [] for stage in STAGES}
        for r in members:
            path = out/r['directory']/'projections.npz'
            if sha(path) != r['projection_sha256']:
                raise ValueError('Projection hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                for stage in STAGES:
                    cubes[stage].append(data[stage])
        distribution = f(256, 512) if case == 'noise' else ncf(256, 512, 2304)
        info = {'ideal_reference': {'dof': [256, 512], 'noncentrality': 0 if case == 'noise' else 2304,
                                    'mean': float(distribution.mean()), 'std': float(distribution.std()),
                                    'median': float(distribution.median()),
                                    'quantiles_05_50_95': distribution.ppf([.05, .5, .95]).tolist()},
                'saturated_components': sum(r['adapter_metadata']['sample_quantization']['saturated_component_count'] for r in members),
                'sample_components': sum(r['adapter_metadata']['sample_quantization']['component_count'] for r in members),
                'source_noise_mean_power_by_record': [r['source_noise_mean_power'] for r in members],
                'source_signal_mean_power_by_record': [r['source_signal_mean_power'] for r in members],
                'stages': {}}
        edges = np.array(plan[case+'_histogram_edges'])
        for stage in STAGES:
            zs = cubes[stage]
            qs = [powers_and_ratio(z)[1] for z in zs]
            q = np.concatenate(qs)
            rows = [z.transpose(0, 2, 1).reshape(-1, 3) for z in zs]
            pooled = np.concatenate(rows)
            mean = np.mean(pooled, axis=0)
            centered = pooled-mean
            cov = centered.T@centered.conj()/len(centered)
            pseudo = centered.T@centered/len(centered)
            late = np.concatenate([r[1:] for r in rows])-mean
            early = np.concatenate([r[:-1] for r in rows])-mean
            lag_cov = late.T@early.conj()/len(late)
            a = np.concatenate([r[:-1] for r in qs])
            b = np.concatenate([r[1:] for r in qs])
            a = a-a.mean()
            b = b-b.mean()
            counts = np.histogram(q, bins=edges)[0]
            row = {**describe(q), 'histogram_edges': edges.tolist(), 'histogram_counts': counts.tolist(),
                   'histogram_underflow': int(np.count_nonzero(q < edges[0])),
                   'histogram_overflow': int(np.count_nonzero(q > edges[-1])),
                   'record_summaries': [describe(v) for v in qs],
                   'frame_lag1_correlation': float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))),
                   'row_projection_mean': complex_json(mean), 'row_projection_covariance': complex_json(cov),
                   'row_projection_pseudocovariance': complex_json(pseudo),
                   'adjacent_row_covariance': complex_json(lag_cov),
                   'row_mean_power': np.mean(abs(pooled)**2, axis=0).tolist(),
                   'empirical_cdf_sup_distance_to_ideal': float(max(np.max(np.arange(1,len(q)+1)/len(q)-distribution.cdf(np.sort(q))), np.max(distribution.cdf(np.sort(q))-np.arange(len(q))/len(q))))}
            if len(q) != plan['frames_per_case'] or sum(counts)+row['histogram_underflow']+row['histogram_overflow'] != len(q):
                raise ValueError('Histogram or frame population mismatch')
            info['stages'][stage] = row
        summary['cases'][case] = info
    write_json(out/'summary.json', summary)
    print(json.dumps({case: {stage: {k: summary['cases'][case]['stages'][stage][k] for k in ('mean','median','std_ddof1')} for stage in STAGES} for case in ('noise','steady')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'report': report}[args.stage](args.output.resolve())
