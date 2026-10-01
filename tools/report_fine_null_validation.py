#!/usr/bin/env python3
"""Report independent literal H0 outcomes without changing calibration.

Scientific gate failures are retained in the report and do not raise an
exception. Missing, altered or incorrectly labelled inputs do raise, since
they cannot support a report with the frozen denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

from pilot_proxy.testbench.fine_validation_stats import false_alarm_validation
from reduce_fine_validation import (
    CAL_SCHEMA, FLOAT_STAGES, Q16_STAGE, load_contract, load_score_manifest,
    phase_ids, read_json, read_raw,
    reduce_powers, rehash, sha, utc, write_json,
)
from pilot_proxy.testbench.fine_validation_scores import construct_geometry


MAX_Q16 = (1 << 64) - 1


def aware(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Receipt timestamps must include a timezone')
    return parsed


def threshold_value(value, *, exact):
    if exact:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Q16 calibration threshold must be an exact integer')
        if not 1 <= int(value) <= MAX_Q16:
            raise ValueError('Q16 calibration threshold is outside the deployable uint64 range')
        return int(value)
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError('Floating calibration threshold must be finite nonnegative')
    return float(value)


def count_outcomes(samples, valid, threshold, *, always=None):
    """Count strict decisions without dropping invalid or sentinel observations.

    Explicit uint64 threshold conversion prevents mixed-scalar float coercion
    near 2**64. Logical 2**64 is represented only by zero plus always=True;
    it exceeds every deployable threshold. Invalid zero is a distinct state.
    """
    samples, valid = np.asarray(samples), np.asarray(valid)
    if samples.ndim != 1 or valid.shape != samples.shape or valid.dtype != np.dtype(bool):
        raise ValueError('Decision samples and Boolean validity must be aligned vectors')
    undefined = np.zeros(samples.shape, dtype=bool)
    if always is not None:
        threshold = threshold_value(threshold, exact=True)
        always = np.asarray(always)
        if (samples.dtype != np.dtype(np.uint64) or always.shape != samples.shape
                or always.dtype != np.dtype(bool)):
            raise ValueError('Exact Q16 decisions require uint64 samples and Boolean sentinel flags')
        if (np.any(always & ~valid) or np.any((always | ~valid) & (samples != 0))
                or np.any(valid & ~always & (samples == 0))):
            raise ValueError('Inconsistent Q16 zero/invalid/sentinel encoding')
        mask = valid & (np.greater(samples, np.uint64(threshold)) | always)
        sentinels = int(np.count_nonzero(always))
        zero = 0
    else:
        threshold = threshold_value(threshold, exact=False)
        if samples.dtype.kind != 'f' or np.any(samples < 0):
            raise ValueError('Floating responses must have floating dtype and be nonnegative')
        # NaN cannot be silently interpreted as a non-exceedance. It remains
        # in the denominator and is explicitly invalid; +inf is an exceedance.
        undefined = valid & np.isnan(samples)
        valid = valid & ~np.isnan(samples)
        mask = valid & (samples > threshold)
        sentinels = 0
        zero = int(np.count_nonzero(valid & (samples == 0)))
    return {
        'exceedances': int(np.count_nonzero(mask)),
        'invalid_trials': int(np.count_nonzero(~valid)),
        'valid_nonexceedances': int(np.count_nonzero(valid & ~mask)),
        'always_masked_trials': sentinels,
        'undefined_float_trials': int(np.count_nonzero(undefined)),
        'valid_zero_response_trials': zero,
    }


def validate_calibration(out, plan, geometry, contract_bindings):
    calibration = read_json(out / 'calibration.json')
    if (calibration['schema'] != CAL_SCHEMA or calibration['scope'] != plan['scope']
            or calibration['coordinate_system'] != plan['coordinate_system']
            or calibration['null_trials'] != plan['counts']['null_calibration']
            or calibration['status'] != 'calibrated' or calibration['refusals']
            or calibration['independent_null_validation'] is not False):
        raise ValueError('A complete correctly scoped frozen calibration is required')
    if calibration['plan_sha256'] != sha(out / 'plan.json'):
        raise ValueError('Calibration plan differs')
    if (out / 'calibration.sha256').read_text() != sha(out / 'calibration.json') + '\n':
        raise ValueError('Calibration completion marker differs')
    frozen = aware(calibration['frozen_utc'])
    if not aware(plan['frozen_utc']) <= frozen <= datetime.now(timezone.utc):
        raise ValueError('Calibration freeze chronology differs')
    _, expected = load_score_manifest(out, plan, 'null_calibration')
    source_path = out / 'reduction/source-binding.json'
    source = read_json(source_path)
    if (source['schema'] != 'literal-fine-reducer-source-v1'
            or source['source_sha256'] != calibration['source_sha256']
            or not aware(plan['frozen_utc']) <= aware(source['frozen_utc']) <= frozen):
        raise ValueError('Calibration reducer source binding differs')
    rehash(out, source['source_sha256'])
    rehash(out, source['snapshots_sha256'])
    expected.update(contract_bindings)
    expected.update({'reduction/source-binding.json': sha(source_path), **source['snapshots_sha256']})
    bound = calibration['calibration_inputs_sha256']
    if any(bound.get(name) != digest for name, digest in expected.items()):
        raise ValueError('Calibration omits or changes a required dependency')
    allowed = ('raw/null_calibration/', 'scores/null_calibration/',
               'source_snapshots/', 'reduction/source_snapshots/')
    singletons = {'plan.json', 'trial-identities.npy', 'reduction/source-binding.json',
                  'reduction/cache-plan.json'}
    for name in bound:
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts
                or (name not in singletons and not name.startswith(allowed))):
            raise ValueError('Calibration binds forbidden or unrecognized inputs')
        if name.endswith('.json') and name.startswith(('raw/null_calibration/', 'scores/null_calibration/')):
            if not aware(plan['frozen_utc']) <= aware(read_json(out / name)['completed_utc']) <= frozen:
                raise ValueError('Calibration was frozen before its inputs completed')
    rehash(out, bound)
    pfas = plan['diagnostic_pfa'] + plan['primary_pfa']
    expected_channels = {f'ch{c}' for c in plan['channels']}
    if set(calibration['fine']) != expected_channels or set(calibration['coarse']) != expected_channels:
        raise ValueError('Calibration channel coverage differs')
    for channel in plan['channels']:
        fine = calibration['fine'][f'ch{channel}']
        if set(fine) != {str(a) for a in geometry[channel]['anchors']}:
            raise ValueError('Calibration anchor coverage differs')
        for anchor, stages in fine.items():
            if set(stages) != set((*FLOAT_STAGES, Q16_STAGE)):
                raise ValueError('Calibration stage coverage differs')
            ranks = {str(int(r)) for r in construct_geometry(int(anchor)).ranks}
            for stage, records in stages.items():
                if set(records) != ranks:
                    raise ValueError('Calibration rank coverage differs')
                for levels in records.values():
                    check_threshold_records(levels, pfas, plan, exact=stage == Q16_STAGE)
        coarse = calibration['coarse'][f'ch{channel}']
        if set(coarse) != set(FLOAT_STAGES):
            raise ValueError('Calibration coarse stage coverage differs')
        for levels in coarse.values():
            check_threshold_records(levels, pfas, plan, exact=False)
    return calibration


def check_threshold_records(records, pfas, plan, *, exact):
    if set(records) != {str(p) for p in pfas}:
        raise ValueError('Calibration Pfa coverage differs')
    for pfa in pfas:
        record = records[str(pfa)]
        if (record['status'] != 'available' or record['trials'] != plan['counts']['null_calibration']
                or record['nominal_pfa'] != pfa or record['independent_validation'] is not False):
            raise ValueError('Calibration threshold provenance differs')
        threshold_value(record['threshold'], exact=exact)


def independent_inventory(out, plan, ids, calibration):
    selected = phase_ids(plan, ids, 'null_validation')
    frozen = aware(calibration['frozen_utc'])
    # The campaign identity table uses its general fixture-choice recipe even
    # for H0. These labels are unused: the raw null phase injects amplitude 0.
    expected_labels = [2 + (hashlib.sha256(
        f"{plan['schema']}:fixture-choice:null_validation:{int(trial)}".encode()
    ).digest()[0] & 1) for trial in selected['trial']]
    if selected['payload'].tolist() != expected_labels:
        raise ValueError('Independent null identity labels differ from the frozen recipe')
    expected_files, inventory, bindings = set(), [], {}
    for start in range(0, len(selected), 32):
        chunk = selected[start:start+32]
        path = out / 'raw/null_validation' / f'{start:06d}.npz'
        receipt = path.with_suffix('.json')
        saved = read_json(receipt)
        metadata = {'plan_sha256': sha(out / 'plan.json'), 'phase': 'null_validation',
                    'start': start, 'stop': start+len(chunk), 'channels': plan['channels'],
                    'streams': 2048, 'float_stages': plan['float_stages'],
                    'raw_seeds': [int(v) for v in chunk['raw_seed']],
                    'axes': ['trial', 'channel', 'stage_if_float', 'term', 'bin_if_fine']}
        if saved['metadata'] != metadata or sha(path) != saved['sha256']:
            raise ValueError(f'Independent null shard identity differs: {path}')
        if not frozen <= aware(saved['completed_utc']) <= datetime.now(timezone.utc):
            raise ValueError('Independent null shard chronology differs from frozen calibration')
        expected_files.update((path, receipt))
        inventory.append((path, metadata))
        bindings[str(path.relative_to(out))] = saved['sha256']
        bindings[str(receipt.relative_to(out))] = sha(receipt)
    actual = {p for p in (out / 'raw/null_validation').iterdir() if p.suffix in ('.json', '.npz')}
    if actual != expected_files:
        raise ValueError('Independent null coverage has missing or extra shards')
    return inventory, bindings


def run(out):
    out = Path(out).resolve()
    if not (out / 'reduction/cache-plan.json').is_file():
        raise ValueError('Frozen reducer cache snapshot is missing')
    plan, ids, geometry, bindings = load_contract(out)
    gate = plan['false_alarm_gate']
    expected_n = 2 if plan['scope'] == 'engineering_only' else 10000
    if (plan['counts']['null_validation'] != expected_n or gate['confidence'] != 0.95
            or gate['family_tests'] != 1334 or gate['cap_multiple'] != 2.0
            or gate['pointwise_width_max_multiple_nominal'] != 1.0):
        raise ValueError('Independent null count or frozen acceptance family differs')
    calibration = validate_calibration(out, plan, geometry, bindings)
    bindings.update(calibration['calibration_inputs_sha256'])
    bindings.update({name: sha(out / name) for name in ('calibration.json', 'calibration.sha256')})
    inventory, raw_bindings = independent_inventory(out, plan, ids, calibration)
    bindings.update(raw_bindings)
    target = out / 'reports/null_validation'
    if target.exists():
        raise ValueError('Refusing to overwrite an independent null report')
    target.mkdir(parents=True)
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).suffix == '.py' and Path(name).resolve().is_relative_to(ROOT):
            sources[str(Path(name).resolve())] = sha(name)
    import shutil
    snapshots = {}
    for name, digest in sources.items():
        path = target / 'source_snapshots' / Path(name).relative_to(ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
        if sha(path) != digest:
            raise ValueError('Report source changed while snapshotting')
        snapshots[str(path.relative_to(out))] = digest
    pieces = {c: [] for c in plan['channels']}
    for path, metadata in inventory:
        raw = read_raw(path, metadata, len(plan['channels']))
        if any(np.any(raw[name] >= np.uint64(1 << 63)) for name in ('fixed_fine', 'fixed_coarse')):
            raise ValueError('Raw fixed power exceeds the frozen signed accumulation capacity')
        for column, channel in enumerate(plan['channels']):
            g = geometry[channel]
            reduced = reduce_powers(raw['float_fine'][:, column], raw['fixed_fine'][:, column],
                                    raw['float_coarse'][:, column], raw['fixed_coarse'][:, column],
                                    g['anchors'], g['mu0'])
            reduced['clip_count'] = raw['clip_count'][:, column]
            pieces[channel].append(reduced)
    rows, artifacts = [], {}
    n = plan['counts']['null_validation']
    pfas = plan['diagnostic_pfa'] + plan['primary_pfa']

    def append(row, samples, valid, threshold, pfa, primary, always=None):
        if samples.shape != (n,) or valid.shape != (n,):
            raise ValueError('Independent denominator differs')
        counts = count_outcomes(samples, valid, threshold, always=always)
        stats = false_alarm_validation(counts['exceedances'], n, nominal_pfa=pfa,
                    cap_pfa=gate['cap_multiple']*pfa,
                    family_tests=gate['family_tests'] if primary else 1,
                    confidence=gate['confidence'])
        failures = []
        if counts['invalid_trials']:
            failures.append('invalid_trials_present')
        if not stats['cap_demonstrated']:
            failures.append('simultaneous_cap_not_demonstrated')
        if not stats['pointwise_width_at_most_nominal']:
            failures.append('pointwise_precision_not_met')
        rows.append({**row, **counts, 'pfa': pfa, 'threshold': threshold,
                     'primary_family': primary,
                     'statistics': stats,
                     'gate_status': 'passed' if primary and not failures else 'failed' if primary else 'diagnostic_only',
                     'gate_failure_reasons': failures if primary else [],
                     'gate_passed': primary and not failures})

    for channel in plan['channels']:
        data = {name: np.concatenate([p[name] for p in pieces[channel]])
                for name in pieces[channel][0]}
        path = target / f'ch{channel}.npz'
        with path.open('xb') as stream:
            np.savez(stream, **data)
        artifacts[str(path.relative_to(out))] = sha(path)
        for slot, anchor in enumerate(geometry[channel]['anchors']):
            g = construct_geometry(anchor)
            for stage_index, stage in enumerate((*FLOAT_STAGES, Q16_STAGE)):
                for rank_index, rank in enumerate(g.ranks):
                    for pfa in pfas:
                        threshold = calibration['fine'][f'ch{channel}'][str(anchor)][stage][str(int(rank))][str(pfa)]['threshold']
                        exact = stage == Q16_STAGE
                        primary = exact and slot < 7 and pfa in plan['primary_pfa']
                        row = {'channel': channel, 'kind': 'fine', 'geometry_slot': slot,
                               'anchor': anchor, 'rank_index': rank_index, 'rank': int(rank),
                               'stage': stage}
                        if exact:
                            append(row, data['required_q16'][:, slot, rank_index],
                                   data['fixed_valid'][:, slot, rank_index], threshold, pfa,
                                   primary, data['always_masked'][:, slot, rank_index])
                        else:
                            append(row, data['fine_float'][:, slot, stage_index, rank_index],
                                   data['fine_valid'][:, slot, stage_index, rank_index],
                                   threshold, pfa, primary)
        for stage_index, stage in enumerate(FLOAT_STAGES):
            for pfa in pfas:
                threshold = calibration['coarse'][f'ch{channel}'][stage][str(pfa)]['threshold']
                append({'channel': channel, 'kind': 'coarse', 'stage': stage},
                       data['coarse_float'][:, stage_index], data['coarse_valid'][:, stage_index],
                       threshold, pfa,
                       stage_index == len(FLOAT_STAGES)-1 and pfa in plan['primary_pfa'])
    primary = [r for r in rows if r['primary_family']]
    expected = len(plan['channels'])*(7*4*2+2)
    if len(primary) != expected or (plan['scope'] != 'engineering_only' and expected != gate['family_tests']):
        raise ValueError('Primary false-alarm family denominator differs')
    report = {'schema': 'literal-fine-independent-null-report-v1', 'completed_utc': utc(),
              'plan_sha256': sha(out / 'plan.json'), 'scope': plan['scope'],
              'trials_per_policy': n, 'primary_family_size': gate['family_tests'],
              'primary_policy_rows': len(primary), 'passed_primary_rows': sum(r['gate_passed'] for r in primary),
              'failed_primary_rows': sum(not r['gate_passed'] for r in primary),
              'all_primary_gates_passed': all(r['gate_passed'] for r in primary),
              'thresholds_changed': False, 'calibration_trials_reused': False,
              'null_fixture_labels': 'The frozen identity table retains its randomized payload2/3 labels; H0 generation ignores those labels and injects amplitude zero. Labels do not imply an injected waveform.',
              'coordinate_system': plan['coordinate_system'],
              'engineering_gate_interpretation': 'Two-trial engineering mode is incapable of meeting the declared precision/cap gates.' if plan['scope'] == 'engineering_only' else None,
              'coarse_scope': 'Floating normalized coarse ratio from exact fixed coarse powers; not an integer coarse-decision replay.',
              'rows': rows, 'inputs_sha256': bindings, 'outputs_sha256': artifacts,
              'source_sha256': sources, 'snapshots_sha256': snapshots,
              'physical_certification': False,
              'interpretation': 'Independent evaluation of frozen thresholds. Primary simultaneous cap is twice nominal Pfa; no claim of equality to nominal. All0.001 rows and nonprimary paths are pointwise diagnostics. Failed gates are retained.'}
    rehash(out, bindings)
    rehash(out, sources)
    rehash(out, snapshots)
    rehash(out, artifacts)
    write_json(target / 'report.json', report)
    with (target / 'report.sha256').open('x') as stream:
        stream.write(sha(target / 'report.json')+'\n')
    print(json.dumps({k: report[k] for k in ('scope', 'trials_per_policy', 'primary_policy_rows',
                                            'passed_primary_rows', 'all_primary_gates_passed')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
