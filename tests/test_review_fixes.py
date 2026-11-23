"""Crash recovery, fatal integrity failures and real dependency/command isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vibeforge_v2.agent import Shell
from vibeforge_v2.checks import CandidateCheck
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.environment import Environments, ExperimentIntegrityError
from vibeforge_v2.feedback import write_feedback
from vibeforge_v2.loop import Pipeline
from vf_runtime.storage.atomic import read_json, write_json
from vf_runtime.versioning.artifacts import source_hash
from test_pipeline import fixture_connection, fixture_edit


def test_shell_timeout_kills_process_group_and_keeps_partial_output(tmp_path):
    tool = Shell(tmp_path, tmp_path / 'log', timeout_seconds=.15)
    result = tool.call(json.dumps({'command': "printf STARTED; sleep 30 & wait"}))
    assert tool.calls[-1]['timed_out']
    assert tool.calls[-1]['exit_code'] == 124
    assert 'STARTED' in result and 'timed out' in result
    # A subsequent command can still execute after cancellation.
    assert 'exit_code: 0' in tool.call(json.dumps({'command': 'printf recovered'}))


def test_comments_and_json_formatting_are_not_runtime_optimizations(tmp_path):
    tmp_path = tmp_path / 'engine'
    tmp_path.mkdir()
    (tmp_path / 'engine.py').write_text('def generate(): return 1\n')
    (tmp_path / 'engine.json').write_text('{"backend":"toy"}')
    checker = CandidateCheck(tmp_path, tmp_path.parent / 'checks')
    (tmp_path / 'engine.py').write_text('# faster!\ndef generate(): return 1\n')
    (tmp_path / 'engine.json').write_text('{\n  "backend": "toy"\n}\n')
    report = checker.inspect()
    assert report['static_passed'] and not report['ready_for_host_evaluation']
    (tmp_path / 'engine.py').write_text('def generate(): return 2\n')
    assert checker.inspect()['ready_for_host_evaluation']


def test_crash_after_decision_before_checkpoint_replays_without_overwrite(tmp_path):
    output = tmp_path / 'run'
    config = load(ROOT / 'configs/toy.json')
    Pipeline(config, output, edit=fixture_edit, connect=fixture_connection).execute(rounds=1)
    committed = (output / 'rounds/0001/decision.json').read_bytes()
    state = read_json(output / 'state.json')
    state.update(completed_rounds=0, stable='initial/engine', stable_hash=source_hash(output / 'initial/engine'))
    write_json(output / 'state.json', state)
    result = Pipeline(config, output).execute(rounds=1, resume=True)
    assert (output / 'rounds/0001/decision.json').read_bytes() == committed
    assert result['stable'] == 'rounds/0001/engine'
    assert result['completed_rounds'] == 1


def test_frozen_input_edit_is_fatal_and_no_next_round_runs(tmp_path):
    def edit(url, config, workspace, folder, context):
        fixture_edit(url, config, workspace, folder, context)
        path = Path(workspace).parents[2] / 'cases.json'
        path.write_text(path.read_text() + '\n')
    output = tmp_path / 'run'
    config = load(ROOT / 'configs/toy.json')
    with pytest.raises(ExperimentIntegrityError, match='Frozen evaluation input changed'):
        Pipeline(config, output, edit=edit, connect=fixture_connection).execute(rounds=3)
    assert not (output / 'rounds/0002').exists()
    assert read_json(output / 'state.json')['stable'] == 'initial/engine'
    assert read_json(output / 'state.json')['integrity_failed']
    with pytest.raises(ExperimentIntegrityError, match='previously failed'):
        Pipeline(config, output).execute(rounds=3, resume=True)


def test_source_environments_keep_rejected_dependencies_out_of_stable(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    config['agent']['isolate_dependencies'] = True
    envs = Environments(config, tmp_path)
    stable, candidate = tmp_path / 'initial/engine', tmp_path / 'rounds/0001/engine'
    stable.mkdir(parents=True)
    candidate.mkdir(parents=True)
    envs.prepare(stable)
    envs.seal(stable)
    envs.prepare(candidate, stable)
    # Install a local package with no network or build dependency changes.
    import zipfile
    wheel = tmp_path / 'vf_review_marker-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('vf_review_marker.py', 'VALUE = 42\n')
        archive.writestr('vf_review_marker-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: vf-review-marker\nVersion: 1.0\n')
        archive.writestr('vf_review_marker-1.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nGenerator: review\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr('vf_review_marker-1.0.dist-info/RECORD', '')
    subprocess.run([envs.python(candidate), '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)],
                   check=True, capture_output=True)
    subprocess.run([envs.python(candidate), '-c', 'import vf_review_marker; assert vf_review_marker.VALUE == 42'], check=True)
    result = subprocess.run([envs.python(stable), '-c', 'import vf_review_marker'], capture_output=True)
    assert result.returncode != 0
    envs.verify(stable)
    envs.seal(candidate)
    envs.verify(candidate)


def test_feedback_includes_paired_losses_and_actual_failed_output(tmp_path):
    write_json(tmp_path / 'quality/report.json', {'passed': False, 'errors': ['paired_quality_not_established'],
        'paired': {'losses': ['task-a'], 'loss_rate_upper': .06},
        'runs': [{'cases': [{'id': 'task-a', 'passed': False, 'text': 'actual bad answer'}]}]})
    write_feedback(tmp_path, {'reason': 'quality_failed'})
    text = (tmp_path / 'outcome.md').read_text()
    assert 'task-a' in text and 'actual bad answer' in text and '0.06' in text
    assert '正式性能未执行' in text


def test_throughput_improvement_cannot_hide_latency_regression():
    from vf_runtime.controller.admission import decide
    protocol = {'objective': 'throughput', 'min_gain': .02, 'confidence': .95,
                'max_metric_regressions': {'latency_mean_seconds': .05}}
    a = [{'valid': True, 'metrics': {'throughput': 10, 'latency_mean_seconds': 1}}] * 5
    b = [{'valid': True, 'metrics': {'throughput': 12, 'latency_mean_seconds': 1.2}}] * 5
    assert decide({'passed': True}, a, b, protocol).reason.startswith('metric_regression:')
    b = [{'valid': True, 'metrics': {'throughput': 12, 'latency_mean_seconds': 1.01}}] * 5
    assert decide({'passed': True}, a, b, protocol).outcome == 'accept'


def test_interrupt_during_activation_does_not_commit_acceptance(tmp_path):
    def connection(url, config, workspace, folder):
        if Path(folder).name == 'activation':
            raise KeyboardInterrupt()
        return fixture_connection(url, config, workspace, folder)
    output = tmp_path / 'run'
    with pytest.raises(KeyboardInterrupt):
        Pipeline(load(ROOT / 'configs/toy.json'), output, edit=fixture_edit, connect=connection).execute(rounds=1)
    decision = read_json(output / 'rounds/0001/decision.json')
    assert decision['outcome'] == 'interrupted'
    assert read_json(output / 'state.json')['stable'] == 'initial/engine'


def test_host_source_mutation_is_fatal_before_evaluation(tmp_path):
    import shutil
    project = tmp_path / 'project'
    shutil.copytree(ROOT / 'src', project / 'src')
    shutil.copytree(ROOT / 'prompts', project / 'prompts')
    config = load(ROOT / 'configs/toy.json')
    config['project'] = str(project)
    def edit(url, config, workspace, folder, context):
        (project / 'src/review_mutation.py').write_text('# actual host change\n')
    output = tmp_path / 'run'
    with pytest.raises(ExperimentIntegrityError, match='Host source changed'):
        Pipeline(config, output, edit=edit, connect=fixture_connection).execute(rounds=2)
    assert not (output / 'rounds/0002').exists()
    assert not (output / 'rounds/0001/build.json').exists()


def test_identical_rejected_candidate_requires_source_or_dependency_change(tmp_path):
    from vibeforge_v2.environment import package_snapshot
    old, current = tmp_path / 'rounds/0001/engine', tmp_path / 'rounds/0002/engine'
    for path in (old, current):
        path.mkdir(parents=True)
        (path / 'engine.py').write_text('def generate(): return 1\n')
        (path / 'engine.json').write_text('{"backend":"toy"}')
    write_json(old.parent / 'decision.json', {'outcome': 'reject', 'reason': 'correctness_failed'})
    write_json(old.parent / 'environment.json', {'candidate_after': package_snapshot(sys.executable)})
    checker = CandidateCheck(current, tmp_path / 'checks')
    assert checker.repeated_failure(sys.executable)['round'] == '0001'
    (current / 'engine.py').write_text('def generate(): return 2\n')
    assert checker.repeated_failure(sys.executable) is None


def test_shell_does_not_expose_cuda_to_duplicate_model_loads(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    tool = Shell(tmp_path, tmp_path / 'logs')
    result = tool.call(json.dumps({'command': "python -c \"import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))\""}))
    assert "''" in result
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'


def test_dependency_overlay_inherits_selected_runtime_without_system_packages(tmp_path):
    from vibeforge_v2.environment import package_snapshot
    config = load(ROOT / 'configs/toy.json')
    config['agent']['isolate_dependencies'] = True
    envs = Environments(config, tmp_path)
    source = tmp_path / 'initial/engine'
    source.mkdir(parents=True)
    python = envs.prepare(source)
    def identity(snapshot):
        return {(row['name'].lower().replace('_', '-'), row['version'], row['metadata_hash'])
                for row in snapshot['packages'] if row['name'].lower() not in {'pip', 'setuptools', 'wheel'}}
    assert identity(package_snapshot(python)) == identity(package_snapshot(config['runtime_python']))


def test_paired_quality_can_stop_when_observed_losses_cannot_recover(tmp_path, monkeypatch):
    import vf_runtime.evaluation.quality as module
    from types import SimpleNamespace
    calls = []
    def command(argv, folder, log, timeout, **kwargs):
        calls.append(argv)
        write_json(Path(folder) / 'result.json', {'passed': True, 'metrics': {'accuracy': .9},
                   'cases': [{'id': str(i), 'passed': i >= 10} for i in range(100)]})
        return {'exit_code': 0}
    monkeypatch.setattr(module, 'run_command', command)
    settings = {'command': ['fake', '{output}'], 'repeats': 3, 'timeout_seconds': 1,
                'min_scores': {}, 'max_regressions': {},
                'paired': {'min_cases': 100, 'max_loss_rate': .05, 'confidence': .95}}
    baseline = {'cases': [{'id': str(i), 'passed': True} for i in range(100)]}
    report = module.run_quality(SimpleNamespace(url='http://unused'), settings, tmp_path,
                                {'project': str(ROOT)}, baseline)
    assert not report['passed'] and len(calls) == 1
    assert report['repeat_count'] == 1 and report['planned_repeat_count'] == 3
    assert report['early_stop']['reason'] == 'paired_failure_cannot_recover_with_more_repeats'


def test_old_quality_reports_do_not_turn_missing_diagnostics_into_zero(tmp_path):
    write_json(tmp_path / 'quality/report.json', {'passed': False,
        'cases': [{'id': 'old', 'passed': False, 'text': 'bad'}]})
    write_feedback(tmp_path, {'reason': 'quality_failed'})
    text = (tmp_path / 'outcome.md').read_text()
    assert "'failed_exact_match': None" in text
    assert 'not zero failures' in text
