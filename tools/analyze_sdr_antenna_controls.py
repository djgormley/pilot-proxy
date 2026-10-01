#!/usr/bin/env python3
"""Descriptive fixed-geometry analysis of complete antenna-control captures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pilot_proxy.integration.sdr_upgrade_adapter import (  # noqa: E402
    DigitalAdapterConfig, adapt_record,
)
from run_sdr_distribution_reference import packed_recount  # noqa: E402

CONFIG = DigitalAdapterConfig(2_000_000, 100_000., 0, 500.)
LABELS = {'noise': 'Receiver-only ambient', 'txzero': 'TX-active zero IQ',
          'tone': 'Commanded steady tone'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def moments(values):
    """Keep every observation in counts; finite-only summaries are explicit."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'total_count': int(values.size), 'finite_count': int(finite.size),
            'undefined_count': int(np.isnan(values).sum()),
            'infinite_count': int(np.isinf(values).sum()),
            'finite_mean': float(finite.mean()) if finite.size else None,
            'finite_median': float(np.median(finite)) if finite.size else None,
            'finite_std_ddof1': float(finite.std(ddof=1)) if finite.size > 1 else None,
            'finite_min': float(finite.min()) if finite.size else None,
            'finite_max': float(finite.max()) if finite.size else None}


def powers_ratio(z):
    powers = np.sum(abs(z)**2, axis=2)
    denominator = powers[:, 1]+powers[:, 2]
    ratio = np.full(len(powers), np.nan)
    np.divide(2*powers[:, 0], denominator, out=ratio, where=denominator > 0)
    ratio[(denominator == 0) & (powers[:, 0] > 0)] = np.inf
    return powers, ratio


def analyze_capture(folder, destination):
    folder, destination = Path(folder), Path(destination)
    receipt = json.loads((folder/'receipt.json').read_text())
    plan = json.loads((folder/'plan.json').read_text())
    if not receipt.get('success') or not receipt.get('cleanup_confirmed'):
        raise ValueError('Only complete transport-qualified records may enter analysis')
    if receipt['plan_sha256'] != sha(folder/'plan.json'):
        raise ValueError('Capture plan hash differs')
    for name, expected in receipt['artifacts'].items():
        if sha(folder/name) != expected:
            raise ValueError('Capture artifact changed: '+name)
    if plan['sample_rate_hz'] != 2_000_000 or plan['record_samples'] != 4_000_000:
        raise ValueError('Capture geometry differs')
    iq = np.fromfile(folder/'accepted.cfile', dtype='<c8')
    if iq.size != 4_000_000 or not np.all(np.isfinite(iq)):
        raise ValueError('Capture length or finiteness differs')
    result = adapt_record(iq, CONFIG)
    if result['metadata']['frame_count'] != 47:
        raise ValueError('Complete-frame count differs')
    checked = packed_recount(result)
    natural = np.einsum('frk,tk->ftr', result['resampled_frames'],
                        result['natural_weights'].conj())
    powers, ratio = powers_ratio(natural)
    packed = result['coarse_ratio']
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'frames.npz', natural_projections=natural,
                        natural_powers=powers, natural_ratio=ratio,
                        packed_projections_i32=result['projections_i32'],
                        packed_powers=result['term_power_sums'], packed_ratio=packed,
                        frame_start_seconds=result['frame_start_seconds'])
    rows = natural.transpose(0, 2, 1).reshape(-1, 3)
    centered = rows-rows.mean(axis=0)
    cov = centered.conj().T@centered/(len(rows)-1)
    denom = np.sqrt(np.outer(cov.diagonal().real, cov.diagonal().real))
    coherence = np.divide(abs(cov), denom, out=np.zeros((3, 3)), where=denom > 0)
    mean_powers = powers.mean(axis=0)
    record = {
        'schema': 'sdr-antenna-controls-record-v1', 'mode': plan['mode'],
        'label': LABELS[plan['mode']], 'capture_directory': str(folder.resolve()),
        'capture_receipt_sha256': sha(folder/'receipt.json'),
        'input_sha256': sha(folder/'accepted.cfile'),
        'frame_array_sha256': sha(destination/'frames.npz'),
        'adapter_metadata': result['metadata'],
        'independent_complex_projections_checked': checked,
        'natural_ratio': moments(ratio), 'packed_ratio': moments(packed),
        'natural_mean_frame_term_powers': mean_powers.tolist(),
        'natural_lower_upper_reference_power_ratio':
            float(mean_powers[1]/mean_powers[2]) if mean_powers[2] > 0 else None,
        'natural_frame_term_power_std_ddof1': powers.std(axis=0, ddof=1).tolist(),
        'centered_row_covariance': {'real': cov.real.tolist(), 'imag': cov.imag.tolist()},
        'centered_row_coherence_magnitude': coherence.tolist(),
        'zero_variance_rows': (cov.diagonal().real <= 0).tolist(),
        'scope': 'Descriptive single-input antenna record; no thermal-null, stationary-source, absolute-power or physical-path qualification. Centering includes any deterministic tone drift; frame observations may depend.',
    }
    write_json(destination/'record.json', record)
    return record


def analyze(study):
    study = Path(study).resolve()
    plan = json.loads((study/'plan.json').read_text())
    if sha(study/'plan.json') != json.loads((study/'plan-digest.json').read_text())['sha256']:
        raise ValueError('Frozen study plan changed')
    for name, expected in plan['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen source changed: '+name)
    if plan['analysis_config'] != vars(CONFIG):
        raise ValueError('Frozen analysis configuration differs')
    run = json.loads((study/'run-receipt.json').read_text())
    if run['plan_sha256'] != sha(study/'plan.json'):
        raise ValueError('Execution plan identity differs')
    output = study/'analysis'
    output.mkdir(exist_ok=False)
    records, failed, unattempted = [], [], []
    for spec, state in zip(plan['attempts'], run['attempts'], strict=True):
        identity = spec['directory']
        folder = study/spec['directory']
        if state['directory'] != identity or state['mode'] != spec['mode']:
            raise ValueError('Execution attempt identity differs')
        if sha(folder/'plan.json') != spec['plan_sha256']:
            raise ValueError('Attempt plan identity differs')
        if state['status'] == 'unattempted':
            if (folder/'receipt.json').exists():
                raise ValueError('Unattempted record has a receipt')
            unattempted.append(identity)
            continue
        if not (folder/'receipt.json').exists():
            failed.append({'id': identity, 'receipt_sha256': None})
            continue
        if sha(folder/'receipt.json') != state['receipt_sha256']:
            raise ValueError('Execution receipt identity differs')
        receipt = json.loads((folder/'receipt.json').read_text())
        if state['status'] != 'complete' or not receipt.get('success'):
            failed.append({'id': identity, 'receipt_sha256': sha(folder/'receipt.json')})
            continue
        records.append({'id': identity, 'triplet_index': spec['triplet_index'],
                        **analyze_capture(folder, output/identity)})
    summary = {'schema': 'sdr-antenna-controls-summary-v1',
               'plan_sha256': sha(study/'plan.json'),
               'completed_records': len(records), 'failed_records': failed,
               'unattempted_records': unattempted, 'records': records,
               'thermal_noise_qualified': False, 'absolute_power_qualified': False,
               'stationarity_qualified': False, 'gpu_used': False,
               'scope': 'Fixed descriptive control sequence. Failures and unattempted records retained; no F-law acceptance or fitted correction.'}
    write_json(output/'summary.json', summary)
    with (output/'records.csv').open('x') as stream:
        fields = ['id', 'mode', 'frames', 'natural_ratio_median', 'packed_ratio_median',
                  'packed_undefined', 'packed_infinite', 'clipped_components',
                  'lower_upper_reference_power_ratio']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({'id': r['id'], 'mode': r['mode'],
                             'frames': r['natural_ratio']['total_count'],
                             'natural_ratio_median': r['natural_ratio']['finite_median'],
                             'packed_ratio_median': r['packed_ratio']['finite_median'],
                             'packed_undefined': r['packed_ratio']['undefined_count'],
                             'packed_infinite': r['packed_ratio']['infinite_count'],
                             'clipped_components': r['adapter_metadata']['sample_quantization']['saturated_component_count'],
                             'lower_upper_reference_power_ratio': r['natural_lower_upper_reference_power_ratio']})
    print(json.dumps({key: summary[key] for key in ('completed_records', 'failed_records', 'unattempted_records')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    analyze(parser.parse_args().study)
