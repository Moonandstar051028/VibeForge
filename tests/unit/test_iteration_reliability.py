"""Regression cases for interrupted exploration and comparable measurement."""

from types import SimpleNamespace
import shutil
import pytest

from vibeforge.agent.candidate import candidate_gate, complete_experiment_plan, semantic_fingerprint
from vibeforge.agent.context import create_context
from vibeforge.controller.admission import decide
from vibeforge.engine.observation import observe_call
from vibeforge.evaluation.performance import measurement_mode
from vibeforge.storage.atomic import write_json
from vibeforge.storage.database import Database


def test_interrupted_candidate_remains_retryable_in_same_campaign(paths, toy_config, tmp_path):
    parent, workspace = paths.release('s1'), tmp_path / 'candidate'
    parent.mkdir()
    (parent / 'engine.py').write_text('value = 1\n')
    write_json(parent / 'engine.json', {'precision': {}})
    shutil.copytree(parent, workspace)
    (workspace / 'engine.py').write_text('value = 2\n')
    write_json(paths.data / 'profiles/p1.json', {'model': {}, 'hardware': {}, 'environment': {}})
    database = Database(paths.data / 'state/retry.sqlite3')
    try:
        database.put('retry', {'campaign_id': 'retry', 'stable_ref': 's1', 'profile_hash': 'p1'})
        row = {'parent': 's1', 'candidate': 'c1', 'complete': True,
               'candidate_gate': {'valid': True, 'fingerprint': semantic_fingerprint(workspace)},
               'decision': {'outcome': 'reject', 'reason': 'interrupted_iteration'}}
        database.iteration('retry', 1, row)
        context = create_context(database, paths, 'retry', toy_config, workspace, 2)
        assert context['candidate_contract']['tested_candidates'] == []
        database.iteration('retry', 1, {'decision': {'outcome': 'reject', 'reason': 'correctness_failed'}})
        write_json(paths.campaign('retry') / 'iterations/0001/correctness/report.json', {
            'passed': False, 'checks': [
                {'case': 'long', 'passed': True, 'actual_tokens': [1] * 1000},
                {'case': 'concurrent', 'passed': False, 'generation_passed': False,
                 'actual_tokens': [2] * 1000, 'numerical': {'passed': True}}]})
        context = create_context(database, paths, 'retry', toy_config, workspace, 2)
        assert candidate_gate(workspace, context, {})['reason'] == 'duplicate_change'
        summary = context['previous']['validation_summary']
        assert summary['failed_check_count'] == 1
        assert summary['failed_checks'] == [{'case': 'concurrent', 'passed': False,
                                            'generation_passed': False, 'numerical': {'passed': True}}]
    finally:
        database.close()


def test_explicit_unknown_metric_is_not_silently_replaced(tmp_path):
    write_json(tmp_path / 'observation.json', {'measurements': {
        'normal.phases.decode_forward_seconds': {'value': 1., 'unit': 'seconds'}}})
    context = {'evidence_root': str(tmp_path), 'candidate_contract': {
        'version': 3, 'observation_paths': ['observation.json']}}
    with pytest.raises(ValueError, match='Unknown bottleneck_metric'):
        complete_experiment_plan({'bottleneck_metric': 'nonexistent'}, context)
    plan = complete_experiment_plan({}, context)
    assert plan['bottleneck_reference']['metric'] == 'normal.phases.decode_forward_seconds'


def test_tail_objective_cannot_promote_too_few_requests():
    def rows(latency, count):
        return [{'valid': True, 'metrics': {'latency_p95_seconds': latency},
                 'requests': [{} for _ in range(count)]} for _ in range(3)]
    protocol = {'objective': 'latency_p95_seconds', 'min_gain': .02, 'min_tail_requests': 100}
    assert decide({'passed': True}, rows(2, 1), rows(1, 1), protocol).reason == 'insufficient_tail_samples'
    assert decide({'passed': True}, rows(2, 100), rows(1, 100), protocol).outcome == 'accept'


def test_wall_measurement_uses_same_uninstrumented_backend_for_both_schedulers():
    tokenizer = object()
    calls = []
    def generate():
        assert backend.TOKENIZER is tokenizer
        calls.append(True)
        return {'input_ids': [1], 'token_ids': [2]}
    backend = SimpleNamespace(TOKENIZER=tokenizer, generate=generate)
    for scheduler in ('serialized_lock', 'concurrent_requests', 'microbatch'):
        _, timing = observe_call(backend, {'device': 'cpu', 'request_scheduler': scheduler},
                                 'generate', {}, detailed=False)
        assert timing['instrumentation'] == 'host wall clock only'
        assert timing['ttft_seconds'] is None
    assert len(calls) == 3
    assert measurement_mode({'objective': 'throughput'}) == 'wall'
    assert measurement_mode({'objective': 'throughput', 'limits': {'tpot_seconds_p95': 1}}) == 'detailed'


@pytest.mark.parametrize('eventual_success', [True, False])
def test_analysis_retries_are_bounded_and_keep_unreviewed_evidence(paths, toy_config, eventual_success):
    from vibeforge.controller.supervisor import Supervisor
    controller = Supervisor(paths, 'analysis-retry')
    controller.config = toy_config
    controller.client = SimpleNamespace(health=lambda: {'ready': True})
    controller.database.put('analysis-retry', {'campaign_id': 'analysis-retry', 'state': 'READY',
        'active_ref': 's1', 'stable_ref': 's1', 'completed_iterations': 1, 'analysis_pending': True})
    controller.database.iteration('analysis-retry', 1, {'analysis_pending': True})
    calls = []
    def session(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1 or not eventual_success:
            raise RuntimeError('transient response error')
        controller.database.iteration('analysis-retry', 1, {'analysis_pending': False})
    controller.agent_session = session
    try:
        if eventual_success:
            controller.complete_pending_analyses()
        else:
            with pytest.raises(RuntimeError, match='transient'):
                controller.complete_pending_analyses()
        assert len(calls) == 2
        assert calls[0]['session_folder'] is None
        assert str(calls[1]['session_folder']).endswith('0001/retry-1')
        assert controller.database.iteration('analysis-retry', 1)['analysis_pending'] is not eventual_success
    finally:
        controller.close()


def test_loaded_implementation_access_is_bounded_and_cannot_read_arbitrary_paths(project):
    import importlib.util
    from vibeforge.engine.implementation import read_implementation
    spec = importlib.util.spec_from_file_location('source_fixture', project / 'templates/engine/toy/engine.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    index = read_implementation(backend)
    assert 'backend.generate' in index['symbols']
    first = read_implementation(backend, 'backend.generate', limit=40)
    second = read_implementation(backend, 'backend.generate', offset=40, limit=40)
    assert first['sha256'] == second['sha256'] and first['next_offset'] == 40
    assert len(first['source']) == 40
    with pytest.raises(ValueError, match='Select a symbol'):
        read_implementation(backend, '/etc/passwd')


@pytest.mark.parametrize('name', ['engine.json', 'helper.py', 'engine.py'])
def test_finalization_detects_config_helpers_and_edits_beyond_source_preview(tmp_path, name):
    from vibeforge.agent.runner import candidate_edit_summary
    parent, candidate = tmp_path / 'parent', tmp_path / 'candidate'
    parent.mkdir()
    for n, data in {'engine.py': '# padding\n' * 2000 + 'value = 1\n', 'engine.json': '{}\n'}.items():
        (parent / n).write_text(data)
    shutil.copytree(parent, candidate)
    with (candidate / name).open('a') as stream:
        stream.write('changed = True\n' if name.endswith('.py') else ' ')
    summary = candidate_edit_summary(parent, candidate)
    assert summary['workspace_changed'] and name in summary['changed_files']
    assert 'candidate/' + name in summary['candidate_diff']


def test_planning_compaction_keeps_actionable_knowledge():
    from qwen_agent.utils.tokenization_qwen import tokenizer
    from vibeforge.agent.runner import _fit_model_context
    rule = {'rule_id': 'component', 'implementation': {'recipe': 'Compile the measured helper'},
            'validation': {'scope': 'component only'}, 'evidence': [{'path': 'knowledge/result.json'}]}
    context = {'optimization_brief': {'decision_rule': 'Use evidence'}, 'knowledge': [rule],
               'candidate_contract': {'observation_paths': ['observation.json']},
               'instructions': 'retain essential constraint ' * 1000}
    value, steps, _ = _fit_model_context(context, 'Plan', 8192, 1024, tokenizer,
                                          max_static_tokens=100)
    assert 'control_metadata_retrievable' not in steps or value['knowledge'] == [rule]
    assert 'planning_brief_only' in steps
    assert value['knowledge'] == [rule]


def test_failed_probe_precheck_reaches_refinement_without_more_broken_requests(paths, toy_config, monkeypatch):
    import vibeforge.controller.supervisor as module
    controller = module.Supervisor(paths, 'failed-probe')
    controller.config = toy_config
    controller.config['probe']['warmup'] = 1
    controller.database.put('failed-probe', {'stable_ref': 'parent'})
    for name in ('parent', 'candidate'):
        write_json(paths.release(name) / 'engine.json', {})
    write_json(controller.directory / 'observation.json', {})
    failure = {'passed': False, 'checks': [{'passed': False, 'error': 'FailOnRecompileLimitHit'}]}
    monkeypatch.setattr(module, 'load_reference', lambda _: {})
    monkeypatch.setattr(module, 'evaluate_candidate_precheck', lambda *args: failure)
    def unexpected(**kwargs):
        pytest.fail('Broken candidate must not be warmed or profiled after precheck failure')
    client = SimpleNamespace(generate=unexpected, profile=unexpected)
    client.bounded = lambda *args: client
    try:
        report = controller.quick_candidate_probe(client, 'candidate',
            {'bottleneck_reference': {'evidence_path': 'observation.json'}}, controller.directory / 'probe')
        assert report['needs_revision'] and report['feedback_state'] == 'revise'
        assert report['correctness_precheck'] == failure
        assert report['admission'] is False and report['warmup_runs'] == 0
    finally:
        controller.close()


def test_current_campaign_knowledge_excludes_external_history(paths, toy_config, tmp_path, monkeypatch):
    from vibeforge.agent import context as module
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('value = 1\n')
    write_json(workspace / 'engine.json', {'precision': {}})
    write_json(paths.data / 'profiles/p1.json', {'model': {}, 'hardware': {}, 'environment': {}})
    toy_config['knowledge'] = {'scope': 'current_campaign'}
    database = Database(paths.data / 'state/isolated.sqlite3')
    try:
        database.put('isolated', {'campaign_id': 'isolated', 'stable_ref': 's1', 'profile_hash': 'p1'})
        database.iteration('isolated', 1, {'parent': 's1', 'candidate': 'c1', 'complete': True,
            'hypothesis': 'Own measured experiment',
            'decision': {'outcome': 'reject', 'reason': 'correctness_failed'}})
        # Even an archive containing invalid paths must not be consulted.
        write_json(paths.campaign('isolated') / 'memory/archived-campaigns.json',
                   {'entries': [{'memory_index': '../external.json'}]})
        def forbidden(*args, **kwargs):
            raise AssertionError('external history was accessed')
        monkeypatch.setattr(database, 'list_campaigns', forbidden)
        monkeypatch.setattr(module, '_database_candidates', forbidden)
        context = create_context(database, paths, 'isolated', toy_config, workspace, 2)
        assert context['knowledge_scope'] == 'current_campaign'
        assert context['transfer_hints'] == []
        assert 'archived_campaign_memory' not in context
        assert context['history'][0]['hypothesis'] == 'Own measured experiment'
        assert context['candidate_contract']['history_refs'] == ['isolated:1']
    finally:
        database.close()
