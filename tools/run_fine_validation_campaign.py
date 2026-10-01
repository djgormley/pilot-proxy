#!/usr/bin/env python3
"""Execute a frozen long fine-validation campaign with explicit checkpoints.

This is a local process supervisor, not a scheduler. It performs no git or RF
operations. A scientific failure remains in its report; an execution/integrity
failure stops the sequence. Finishing generation is not scientific acceptance.
"""
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

TOOLS = Path(__file__).resolve().parent
RAW = TOOLS / 'validate_fine_detector.py'
REDUCE = TOOLS / 'reduce_fine_validation.py'
NULL_REPORT = TOOLS / 'report_fine_null_validation.py'
SENSITIVITY_REPORT = TOOLS / 'report_fine_sensitivity.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data, *, exclusive=False):
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n'
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix+'.writing')
        with temporary.open('w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def steps(out):
    def raw(phase):
        return (phase, [str(RAW), '--output', str(out), '--stage', phase])

    def reduce(phase):
        return (phase, [str(REDUCE), '--output', str(out), '--stage', phase])

    return [raw('null_calibration'), reduce('score-null'), reduce('calibrate'),
            raw('null_validation'),
            ('report-null-validation', [str(NULL_REPORT), '--output', str(out)]),
            raw('discovery'), reduce('score-discovery'), reduce('freeze-grid'),
            raw('evaluation'),
            ('report-evaluation', [str(SENSITIVITY_REPORT), '--output', str(out), '--phase', 'evaluation']),
            raw('stress'),
            ('report-stress', [str(SENSITIVITY_REPORT), '--output', str(out), '--phase', 'stress']),
            raw('spatial_null'), raw('spatial_signal')]


def grid_projection(out):
    """Operational estimate only; never changes the frozen sampling plan."""
    grid = json.loads((out / 'evaluation-grid.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    counts = [len(values) for values in grid['grids'].values()]
    if not counts or grid['plan_sha256'] != sha(out / 'plan.json'):
        raise ValueError('Cannot project a missing or differently bound grid')
    n_eval = sum(counts)*plan['counts']['evaluation']
    n_stress = sum(counts)*plan['counts']['stress']
    return {'cells': len(counts), 'minimum_snr_points': min(counts),
            'maximum_snr_points': max(counts), 'mean_snr_points': sum(counts)/len(counts),
            'evaluation_frames': n_eval, 'stress_frames': n_stress,
            'raw_array_bytes_evaluation_and_stress': (n_eval+n_stress)*37012,
            'projected_compute_hours_evaluation_and_stress':
                [(n_eval+n_stress)*seconds/3600 for seconds in (0.10, 0.14)],
            'basis': 'Engineering literal-frame timings0.10–0.14seconds; excludes reports, I/O, other phases and hardware-load changes. This is an estimate, not a completion deadline.',
            'sampling_plan_changed': False}


def freeze(out):
    plan = json.loads((out / 'plan.json').read_text())
    if plan['scope'] != 'digital_literal_M2048_validation':
        raise ValueError('Long-run supervisor requires a scientific campaign plan')
    if (out / 'plan.sha256').read_text() != sha(out / 'plan.json')+'\n':
        raise ValueError('Campaign plan digest differs')
    target = out / 'execution'
    if target.exists():
        raise ValueError('Execution freeze requires a new execution directory')
    sequence = steps(out)
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for _, command in sequence:
        sources[command[0]] = sha(command[0])
    target.mkdir()
    for source in sources:
        shutil.copyfile(source, target / Path(source).name)
        if sha(target / Path(source).name) != sources[source]:
            raise ValueError('Execution source changed while snapshotting')
    record = {'schema': 'literal-fine-execution-plan-v1', 'frozen_utc': utc(),
              'study': str(out), 'campaign_plan_sha256': sha(out / 'plan.json'),
              'python': sys.executable, 'source_sha256': sources,
              'steps': [{'name': name, 'argv': [sys.executable, *command]} for name, command in sequence],
              'completion_meaning': 'Raw generation and the listed reports finished. Independent full-run audit, spatial/frontier analysis, figures and dissertation import remain required. No physical acceptance is implied.',
              'failure_policy': 'Stop on any nonzero command; preserve every completed output and log. No automatic seed/count/grid changes or retries.'}
    write(target / 'plan.json', record, exclusive=True)
    with (target / 'plan.sha256').open('x') as stream:
        stream.write(sha(target / 'plan.json')+'\n')
    print(json.dumps({'status': 'execution_frozen', 'steps': len(sequence),
                      'execution_plan_sha256': sha(target / 'plan.json')}), flush=True)


def verify(out):
    path = out / 'execution/plan.json'
    if (path.with_suffix('.sha256')).read_text() != sha(path)+'\n':
        raise ValueError('Execution plan digest differs')
    plan = json.loads(path.read_text())
    if (plan['schema'] != 'literal-fine-execution-plan-v1'
            or plan['study'] != str(out) or plan['python'] != sys.executable):
        raise ValueError('Execution location/runtime differs')
    if plan['campaign_plan_sha256'] != sha(out / 'plan.json'):
        raise ValueError('Campaign plan changed')
    expected_sources = {str(Path(__file__).resolve())} | {argv[0] for _, argv in steps(out)}
    if (set(plan['source_sha256']) != expected_sources
            or len({Path(p).name for p in expected_sources}) != len(expected_sources)):
        raise ValueError('Execution source inventory differs or snapshot names collide')
    for source, digest in plan['source_sha256'].items():
        if sha(source) != digest or sha(out / 'execution' / Path(source).name) != digest:
            raise ValueError(f'Execution source/snapshot changed: {source}')
    expected = [{'name': name, 'argv': [sys.executable, *argv]} for name, argv in steps(out)]
    if plan['steps'] != expected:
        raise ValueError('Execution sequence differs')
    return plan


def _run_locked(out):
    import fcntl
    target = out / 'execution'
    with (target / 'supervisor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A supervisor already owns this campaign') from error
        try:
            return _run_owned(out)
        except BaseException as error:
            # Only the process holding the lock may change status. Initial
            # verification failures also need a failed state; a stale prior
            # PID or missing state must not leave them looking successful.
            state_path = target / 'state.json'
            try:
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if not isinstance(state, dict):
                    raise ValueError('State is not an object')
            except (ValueError, TypeError):
                preserved = target / f'state-unreadable-{utc().replace(":", "-")}.json'
                shutil.copyfile(state_path, preserved)
                state = {'unreadable_previous_state': str(preserved)}
            state.setdefault('started_utc', utc())
            state.setdefault('completed_steps', [])
            if (target / 'plan.json').is_file():
                state.setdefault('execution_plan_sha256', sha(target / 'plan.json'))
            state.update(status='failed', pid=os.getpid(), stopped_utc=utc(),
                         error_type=type(error).__name__, error=str(error),
                         scientific_acceptance=False)
            write(state_path, state)
            raise


def _run_owned(out):
    target = out / 'execution'
    plan = verify(out)
    digest = sha(target / 'plan.json')
    state_path = target / 'state.json'
    state = {'status': 'running', 'pid': os.getpid(), 'started_utc': utc(),
             'execution_plan_sha256': digest, 'completed_steps': [],
             'scientific_acceptance': False}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous['execution_plan_sha256'] != digest:
            raise ValueError('Previous execution state belongs to another plan')
        state['completed_steps'] = previous['completed_steps']
        state['previous_started_utc'] = previous['started_utc']
    expected_prefix = [s['name'] for s in plan['steps'][:len(state['completed_steps'])]]
    if state['completed_steps'] != expected_prefix:
        raise ValueError('Completed-step journal is not a valid execution prefix')
    write(state_path, state)
    for index, step in enumerate(plan['steps']):
        receipt_path = target / f'{index:02d}_{step["name"]}.json'
        if index < len(state['completed_steps']):
            receipt = json.loads(receipt_path.read_text())
            if (receipt['execution_plan_sha256'] != digest or receipt['step'] != step
                    or receipt['returncode'] != 0 or receipt['post_step_integrity_verified'] is not True):
                raise ValueError('Completed-step receipt differs')
            if sha(receipt_path.with_suffix('.log')) != receipt['log_sha256']:
                raise ValueError('Completed-step log changed')
            continue
        verify(out)
        log = target / f'{index:02d}_{step["name"]}.log'
        if log.exists() or receipt_path.exists():
            raise ValueError('Interrupted step retained; inspect its log and raw checkpoints before a manual resume')
        state.update(current_step=step['name'], current_step_index=index,
                     step_started_utc=utc(), log_path=str(log))
        write(state_path, state)
        print(json.dumps({'status': 'starting', 'step': step['name'], 'utc': utc()}), flush=True)
        with log.open('x') as output:
            process = subprocess.run(step['argv'], cwd=TOOLS.parent, stdout=output,
                                     stderr=subprocess.STDOUT, check=False)
        receipt = {'step': step, 'execution_plan_sha256': digest,
                   'started_utc': state['step_started_utc'], 'completed_utc': utc(),
                   'returncode': process.returncode, 'log_sha256': sha(log),
                   'post_step_integrity_verified': False}
        if process.returncode == 0:
            try:
                verify(out)
                receipt['post_step_integrity_verified'] = True
            except BaseException as error:
                receipt['post_step_integrity_error'] = str(error)
                write(receipt_path, receipt, exclusive=True)
                raise
        write(receipt_path, receipt, exclusive=True)
        if process.returncode != 0:
            state.update(status='failed', failed_step=step['name'], stopped_utc=utc())
            write(state_path, state)
            print(json.dumps({'status': 'failed', 'step': step['name'], 'log': str(log)}), flush=True)
            return process.returncode
        if step['name'] == 'freeze-grid':
            state['grid_projection'] = grid_projection(out)
            print(json.dumps({'status': 'grid_frozen', **state['grid_projection']}), flush=True)
        state['completed_steps'].append(step['name'])
        state['last_completed_utc'] = utc()
        write(state_path, state)
    verify(out)
    state.update(status='generation_and_listed_reports_complete', completed_utc=utc(),
                 current_step=None, remaining_work=plan['completion_meaning'])
    write(state_path, state)
    print(json.dumps(state), flush=True)
    return 0


def run(out):
    return _run_locked(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['freeze', 'run', 'status'], required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.stage == 'freeze':
        freeze(directory)
    elif args.stage == 'status':
        print((directory / 'execution/state.json').read_text())
    else:
        raise SystemExit(run(directory))
