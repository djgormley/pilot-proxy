"""Long-run failures must preserve checkpoints and stop dependent work."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def supervisor():
    path = Path(__file__).resolve().parents[2] / 'tools/run_fine_validation_campaign.py'
    spec = importlib.util.spec_from_file_location('fine_supervisor_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def study(tmp_path, module):
    out = tmp_path / 'study'
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps({'scope': 'digital_literal_M2048_validation'}))
    (out / 'plan.sha256').write_text(module.sha(out / 'plan.json')+'\n')
    return out


def test_completed_step_is_not_executed_twice(tmp_path, monkeypatch, supervisor):
    out = study(tmp_path, supervisor)
    marker = tmp_path / 'calls.txt'
    child = tmp_path / 'child.py'
    child.write_text('from pathlib import Path\n'
                     f'with Path({str(marker)!r}).open("a") as f: f.write("called\\n")\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('one', [str(child)])])
    supervisor.freeze(out)
    assert supervisor.run(out) == 0
    assert supervisor.run(out) == 0
    assert marker.read_text() == 'called\n'
    state = json.loads((out / 'execution/state.json').read_text())
    assert state['status'] == 'generation_and_listed_reports_complete'
    assert state['scientific_acceptance'] is False


def test_execution_error_stops_following_step_and_preserves_log(tmp_path, monkeypatch, supervisor):
    out = study(tmp_path, supervisor)
    failing = tmp_path / 'failing.py'
    failing.write_text('print("explicit failure")\nraise SystemExit(7)\n')
    following = tmp_path / 'following.py'
    marker = tmp_path / 'should-not-exist'
    following.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('first', [str(failing)]), ('second', [str(following)])])
    supervisor.freeze(out)
    assert supervisor.run(out) == 7
    assert not marker.exists()
    log = out / 'execution/00_first.log'
    assert 'explicit failure' in log.read_text()
    original = log.read_bytes()
    with pytest.raises(ValueError, match='Interrupted step retained'):
        supervisor.run(out)
    assert log.read_bytes() == original


@pytest.mark.parametrize('mutation', ['source', 'snapshot', 'plan', 'campaign'])
def test_changed_frozen_input_refuses_execution(tmp_path, monkeypatch, supervisor, mutation):
    out = study(tmp_path, supervisor)
    child = tmp_path / 'child.py'
    child.write_text('print("ok")\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('one', [str(child)])])
    supervisor.freeze(out)
    target = {'source': child, 'snapshot': out / 'execution/child.py',
              'plan': out / 'execution/plan.json', 'campaign': out / 'plan.json'}[mutation]
    target.write_bytes(target.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='changed|differs'):
        supervisor.run(out)
    assert not (out / 'execution/00_one.log').exists()
    state = json.loads((out / 'execution/state.json').read_text())
    assert state['status'] == 'failed'
    assert state['scientific_acceptance'] is False
    assert 'execution_plan_sha256' in state


def test_lock_contention_does_not_change_the_owner_state(tmp_path, monkeypatch, supervisor):
    import fcntl
    out = study(tmp_path, supervisor)
    child = tmp_path / 'child.py'
    child.write_text('print("ok")\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('one', [str(child)])])
    supervisor.freeze(out)
    state_path = out / 'execution/state.json'
    state_path.write_text('{"status":"running","pid":999999}\n')
    original = state_path.read_bytes()
    with (out / 'execution/supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='already owns'):
            supervisor.run(out)
    assert state_path.read_bytes() == original
    assert not (out / 'execution/00_one.log').exists()


def test_final_child_source_mutation_cannot_mark_campaign_complete(tmp_path, monkeypatch, supervisor):
    out = study(tmp_path, supervisor)
    child = tmp_path / 'child.py'
    child.write_text('from pathlib import Path\np=Path(__file__)\np.write_text(p.read_text()+"\\n")\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('one', [str(child)])])
    supervisor.freeze(out)
    with pytest.raises(ValueError, match='source/snapshot changed'):
        supervisor.run(out)
    state = json.loads((out / 'execution/state.json').read_text())
    assert state['status'] == 'failed'
    assert state['completed_steps'] == []
    receipt = json.loads((out / 'execution/00_one.json').read_text())
    assert receipt['returncode'] == 0
    assert receipt['post_step_integrity_verified'] is False
    assert 'post_step_integrity_error' in receipt


def test_frozen_sequence_preserves_calibration_and_independent_evaluation_order(supervisor, tmp_path):
    names = [name for name, _ in supervisor.steps(tmp_path)]
    assert names == ['null_calibration', 'score-null', 'calibrate', 'null_validation',
        'report-null-validation', 'discovery', 'score-discovery', 'freeze-grid',
        'evaluation', 'report-evaluation', 'stress', 'report-stress',
        'spatial_null', 'spatial_signal']


def test_omitted_source_binding_is_rejected_even_with_matching_plan_marker(tmp_path, monkeypatch, supervisor):
    out = study(tmp_path, supervisor)
    child = tmp_path / 'child.py'
    child.write_text('print("ok")\n')
    monkeypatch.setattr(supervisor, 'steps', lambda _: [('one', [str(child)])])
    supervisor.freeze(out)
    path = out / 'execution/plan.json'
    plan = json.loads(path.read_text())
    plan['source_sha256'].pop(str(child))
    path.write_text(json.dumps(plan))
    path.with_suffix('.sha256').write_text(supervisor.sha(path)+'\n')
    with pytest.raises(ValueError, match='source inventory differs'):
        supervisor.run(out)
    assert not (out / 'execution/00_one.log').exists()
