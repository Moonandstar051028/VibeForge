"""Exercise real process loss/restart on CPU, including the separately owned engine."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from vibeforge_v2.config import ROOT, load
from vibeforge_v2.loop import Pipeline
from vf_runtime.engine.process import same_process, stop_record
from vf_runtime.storage.atomic import read_json, write_json
from test_pipeline import fixture_connection, fixture_edit

spec = importlib.util.spec_from_file_location('run_supervisor', ROOT / 'scripts/supervise_run.py')
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def wait_for(predicate, process, log, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(log.read_text()[-4000:])
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    pytest.fail('Supervisor deadline exceeded: ' + log.read_text()[-4000:])


def test_cpu_process_loss_cleans_orphan_and_resumes_without_round_limit(tmp_path):
    run = tmp_path / 'run'
    workspace = tmp_path / 'workspace/experiment'
    workspace.mkdir(parents=True)
    Pipeline(load(ROOT / 'configs/toy.json'), run,
             edit=fixture_edit, connect=fixture_connection).execute(rounds=1)
    seal = supervisor.recovery_seal(run, ROOT)
    assert supervisor.recovery_check(run, seal)['completed_rounds'] == 1
    config = read_json(run / 'config.json')
    config['evaluation']['warmup'] += 1
    write_json(run / 'config.json', config)
    with pytest.raises(RuntimeError, match='Recovery input changed'):
        supervisor.recovery_check(run, seal)
    seal = supervisor.recovery_seal(run, ROOT)
    state = read_json(run / 'state.json')
    write_json(run / 'state.json', {**state, 'integrity_failed': True})
    with pytest.raises(RuntimeError, match='integrity failure'):
        supervisor.recovery_check(run, seal)
    write_json(run / 'state.json', state)
    active = workspace.parent / 'active.json'
    write_json(active, {'run': str(run), 'workspace': str(workspace)})
    log = workspace / 'supervisor.log'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1'}
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/supervise_run.py'),
            '--run', str(run), '--workspace', str(workspace), '--interval', '.1', '--retry-delay', '.1'],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record_path = run / 'state/active.engine.json'
    try:
        first = wait_for(lambda: read_json(active).get('pipeline_pid'), process, log)
        old_engine = wait_for(lambda: read_json(record_path) if record_path.exists() else None, process, log)
        os.kill(first, signal.SIGKILL)
        second = wait_for(lambda: (p if (p := read_json(active).get('pipeline_pid')) != first else None),
                          process, log)
        assert second != first
        assert not same_process(old_engine)
        recovered = wait_for(lambda: (s if (s := read_json(run / 'state.json')).get('current_round', 0) >= 3
                                     else None), process, log)
        assert recovered['continuous'] and recovered['round_limit'] is None
        assert read_json(run / 'rounds/0002/decision.json')['outcome'] == 'interrupted'
        assert 'unexpected_pipeline_exit' in (workspace / 'operations.jsonl').read_text()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        assert process.returncode == 0
        assert read_json(workspace / 'supervisor-status.json')['phase'] == 'stopped_by_operator'
        assert not record_path.exists()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
        stop_record(record_path)
