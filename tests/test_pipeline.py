"""Real CPU subprocess tests of accept/reject/rollback, not Qwen performance claims."""

import json
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, read_json, write_json
from vf_runtime.versioning.artifacts import source_hash
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.loop import Pipeline


def fixture_connection(url, config, workspace, folder):
    result = {'passed': True, 'fixture': True, 'note': 'CPU controller fixture; no real language model'}
    write_json(Path(folder) / 'connection.json', result)
    return result


def fixture_edit(url, config, workspace, folder, context):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    number = int(folder.parent.name)
    atomic_write(folder / 'context.md', context)
    path = Path(workspace) / 'engine.py'
    source = path.read_text()
    if number == 1:
        source = source.replace('sum(range(120000))', '(120000 * 119999 // 2)')
    elif number == 2:
        source = source.replace('/ 997 for index in indices]', '/ 997 + 0.1 for index in indices]')
    else:
        source += '\ndef intentionally_invalid(:\n'
    path.write_text(source)
    atomic_write(folder.parent / 'analysis.md', 'CPU fixture: plain explanation, no JSON submission.\n')


def test_real_three_round_pipeline(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    output = tmp_path / 'run'
    pipeline = Pipeline(config, output, edit=fixture_edit, connect=fixture_connection)
    state = pipeline.execute(rounds=3)
    decisions = [read_json(output / f'rounds/{i:04d}/decision.json') for i in (1, 2, 3)]
    assert [d['outcome'] for d in decisions] == ['accept', 'reject', 'reject']
    assert decisions[1]['reason'] == 'correctness_failed'
    assert decisions[2]['stage'] == 'build'
    assert not read_json(output / 'rounds/0003/build.json')['passed']
    assert state['stable'] == 'rounds/0001/engine'
    assert state['stable_hash'] == source_hash(output / state['stable'])
    assert state['completed_rounds'] == 3 and state['status'] == 'completed'
    surveys = [read_json(output / f'rounds/{i:04d}/observations/observations.json') for i in (1, 2, 3)]
    assert surveys[0]['source_hash'] != surveys[1]['source_hash']
    assert surveys[1]['source_hash'] == surveys[2]['source_hash']
    assert surveys[0]['collected_at'] < surveys[1]['collected_at']
    # Round 3 profiles the same stable source, dependencies and workload as round 2: reused, labelled.
    assert surveys[2]['collected_at'] == surveys[1]['collected_at']
    assert surveys[2]['reused']['from'] == str(output / 'rounds/0002/observations')
    assert '复用' in (output / 'rounds/0003/observations/summary.md').read_text()
    assert (output / 'rounds/0003/observations/digest.md').exists()
    assert (output / 'experience.md').exists()
    assert all(s['profiles'] and s['concurrency'] for s in surveys)
    assert (output / 'rounds/0001/probe/probe.json').exists()
    assert (output / 'rounds/0001/performance/samples.jsonl').exists()
    assert not (output / 'state/active.engine.json').exists()
    assert 'correctness_failed' in (output / 'history.md').read_text()
    assert '未测量' in (output / 'history.md').read_text()
    assert 'rounds/0001' in (output / 'rounds/0002/agent/context.md').read_text()
    for number in (1, 2, 3):
        folder = output / f'rounds/{number:04d}'
        assert (folder / 'observations/runtime.json').exists()
        manifest = read_json(folder / 'agent/context-files.json')
        assert len(manifest['included_files']) == 1
        assert len(manifest['available_source_files']) == 2
        assert manifest['stable_source_hash'] == read_json(folder / 'parent.json')['source_hash']
    assert '(120000 * 119999 // 2)' in (output / 'current/engine/engine.py').read_text()
    current = read_json(output / 'current/manifest.json')
    assert current['source'] == state['stable'] and current['source_hash'] == state['stable_hash']
    # Two rejected candidates did not switch the canonical accepted-engine pointer.
    assert (output / 'current/engine').resolve() == output / 'rounds/0001/engine'


def test_activation_failure_keeps_original_stable(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    def connection(url, config, workspace, folder):
        if Path(folder).name == 'activation':
            raise RuntimeError('fixture: candidate cannot serve an agent')
        return fixture_connection(url, config, workspace, folder)
    output = tmp_path / 'run'
    state = Pipeline(config, output, edit=fixture_edit, connect=connection).execute(rounds=1)
    decision = read_json(output / 'rounds/0001/decision.json')
    assert decision['outcome'] == 'reject' and decision['stage'] == 'activation'
    assert state['stable'] == 'initial/engine'
    assert 'candidate cannot serve an agent' in decision['error']


def test_activation_cannot_promote_files_changed_after_evaluation(tmp_path):
    def connection(url, config, workspace, folder):
        if Path(folder).name == 'activation':
            path = Path(workspace) / 'engine.py'
            path.write_text(path.read_text() + '\n# changed after formal measurement\n')
        return fixture_connection(url, config, workspace, folder)
    output = tmp_path / 'run'
    state = Pipeline(load(ROOT / 'configs/toy.json'), output,
                     edit=fixture_edit, connect=connection).execute(rounds=1)
    decision = read_json(output / 'rounds/0001/decision.json')
    assert decision['outcome'] == 'reject' and decision['stage'] == 'activation'
    assert 'changed after evaluation' in decision['error']
    assert state['stable'] == 'initial/engine'


def test_independent_quality_rejects_an_uncovered_task_regression(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('original_toy', ROOT / 'engines/toy/engine.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    expected = original.generate('quality-only', max_new_tokens=2)['token_ids']
    wrong = [(expected[0] + 1) % 32, expected[1]]
    config = load(ROOT / 'configs/toy.json')
    config['evaluation']['quality'] = {
        'command': ['{python}', '{project}/tests/quality_fixture.py', '--request', '{request}',
                    '--output', '{output}', '--expected', json.dumps(expected)],
        'files': [], 'file_hashes': {}, 'repeats': 2, 'timeout_seconds': 10,
        'min_scores': {'accuracy': 1}, 'max_regressions': {'accuracy': 0}}
    def edit(url, config, workspace, folder, context):
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(workspace) / 'engine.py'
        source = path.read_text()
        start = source.index('\n', source.index('def generate('))
        source = source[:start] + (f"\n    if prompt == 'quality-only':\n"
            f"        return {{'input_ids': [ord(c) % 128 for c in prompt], 'token_ids': {wrong!r}, 'text': 'wrong'}}") + source[start:]
        path.write_text(source)
        atomic_write(Path(folder).parent / 'analysis.md', 'Deliberate independent quality failure fixture.\n')
    output = tmp_path / 'run'
    state = Pipeline(config, output, edit=edit, connect=fixture_connection).execute(rounds=1)
    assert read_json(output / 'rounds/0001/correctness.json')['passed']
    assert read_json(output / 'rounds/0001/decision.json')['reason'] == 'quality_failed'
    assert not read_json(output / 'rounds/0001/quality/report.json')['passed']
    assert state['stable'] == 'initial/engine'
    assert not (output / 'rounds/0001/performance').exists()


def test_resume_preserves_an_interrupted_round_and_surveys_again(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    def unchanged(url, config, workspace, folder, context):
        Path(folder).mkdir(parents=True, exist_ok=True)
        atomic_write(Path(folder).parent / 'analysis.md', 'No change fixture.\n')
    output = tmp_path / 'run'
    Pipeline(config, output, edit=unchanged, connect=fixture_connection).execute(rounds=1)
    incomplete = output / 'rounds/0002'
    incomplete.mkdir()
    (incomplete / 'partial.log').write_text('Preserve crash evidence.')
    state = Pipeline(read_json(output / 'config.json'), output, edit=unchanged,
                     connect=fixture_connection).execute(rounds=3, resume=True)
    assert read_json(incomplete / 'decision.json')['outcome'] == 'interrupted'
    assert (incomplete / 'partial.log').read_text() == 'Preserve crash evidence.'
    assert (output / 'rounds/0003/observations/observations.json').exists()
    assert state['stable'] == 'initial/engine'
    assert state['completed_rounds'] == 3


def test_vendor_snapshot_is_exact_namespace_copy():
    import hashlib
    manifest = read_json(ROOT / 'vendor-manifest.json')
    for name, item in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == item['vendored_sha256'], name
    assert not (ROOT / 'src/vf_runtime/agent').exists()
    assert not (ROOT / 'src/vf_runtime/controller/supervisor.py').exists()


def test_agent_protocol_has_no_submission_schema():
    from vibeforge_v2.agent import Shell
    assert list(Shell.parameters['properties']) == ['command', 'timeout_seconds']
    assert Shell.parameters['required'] == ['command']
    prompt = (ROOT / 'prompts/engineer.md').read_text()
    assert '性能瓶颈' in prompt and '自然语言' in prompt


def test_context_keeps_last_actual_error_and_only_recent_history(tmp_path):
    from types import SimpleNamespace
    output = tmp_path / 'run'
    output.mkdir()
    history = ['| 轮次 | 结果 | 原因 | 收益 |'] + [f'| {i:04d} | reject | reason-{i} | 未测量 |' for i in range(1, 11)]
    atomic_write(output / 'history.md', '\n'.join(history))
    atomic_write(output / 'rounds/0010/outcome.md', 'NameError: actual_missing_symbol_from_last_round')
    atomic_write(output / 'rounds/0010/changes.patch', '-old_setting\n+actual_previous_change\n')
    folder = output / 'rounds/0011'
    atomic_write(folder / 'observations/summary.md', 'Actual profiler summary')
    atomic_write(folder / 'engine/engine.py', '\n'.join(['# source'] * 60) + '\nlast_line_of_actual_source\n')
    atomic_write(folder / 'engine/engine.json', '{"microbatch_wait_ms": 4}')
    pipeline = Pipeline(load(ROOT / 'configs/toy.json'), output)
    import shutil
    shutil.copytree(folder / 'engine', output / 'initial/engine')
    pipeline.state = {'stable': 'initial/engine', 'stable_hash': source_hash(output / 'initial/engine')}
    health = {'runtime': {'request_scheduler': 'serialized_lock', 'batching_mode': 'none'}}
    context = pipeline.context(11, folder, folder / 'engine',
                               SimpleNamespace(url='http://127.0.0.1:1', health=lambda: health))
    assert 'reason-8' in context and 'reason-9' in context and 'reason-10' in context
    assert 'reason-1 |' not in context
    assert 'NameError: actual_missing_symbol_from_last_round' in context
    assert 'actual_previous_change' in context
    assert (folder / 'engine/engine.py').read_text() not in context
    assert str(output / 'current/engine') in context
    assert context.count('<CURRENT_STABLE_ENGINE>') == 1
    assert '<HISTORICAL_CANDIDATES_NOT_CURRENT>' in context
    assert (folder / 'engine/engine.json').read_text() in context
    assert '"request_scheduler": "serialized_lock"' in context
    assert read_json(folder / 'observations/runtime.json') == health
    manifest = read_json(folder / 'agent/context-files.json')
    assert manifest['previous_outcome'] == str(output / 'rounds/0010/outcome.md')
    assert manifest['previous_changes'] == str(output / 'rounds/0010/changes.patch')
    import hashlib
    for path, identity in manifest['included_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == identity


def test_config_freezes_original_protocol():
    c = load(ROOT / 'configs/qwen3-32b-a800.json')
    assert c['evaluation']['repeats'] == 5
    assert c['evaluation']['reference_repeats'] == 3
    assert c['evaluation']['quality']['repeats'] == 1  # README: one pass of the 50-task suite
    assert c['evaluation']['min_gain'] == .02
    assert c['evaluation']['concurrency'] == 2
    assert c['probe']['repeats'] == 1
    assert c['profiling']['coverage'] == 'all'

