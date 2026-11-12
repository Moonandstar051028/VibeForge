import json

import pytest

from vibeforge.agent.brief import build_optimization_brief
from vibeforge.agent.codex import _brief
from vibeforge.agent.navigation import candidate_index, search_source
from vibeforge.agent.tools import AgentTools


def test_source_index_offsets_include_decorators_and_unicode(tmp_path):
    source = '# 非 ASCII 注释\n@decorator\ndef generate(x):\n    return x\n\nclass Worker:\n    def run(self):\n        return 1\n'
    (tmp_path / 'engine.py').write_text(source)
    (tmp_path / 'engine.json').write_text('{}')
    hidden = tmp_path / '.vibeforge'
    hidden.mkdir()
    (hidden / 'context.json').write_text('{}')
    result = candidate_index(tmp_path)
    assert result['files'] == ['engine.json', 'engine.py']
    functions = {row['symbol']: row for row in result['definitions']}
    row = functions['generate']
    assert source[row['offset']:row['end_offset']] == '@decorator\ndef generate(x):\n    return x\n'
    assert source[functions['Worker.run']['offset']:].startswith('    def run(self):')


def test_index_does_not_follow_links_or_execute_code(tmp_path):
    root = tmp_path / 'work'
    root.mkdir()
    outside = tmp_path / 'outside.py'
    outside.write_text('def secret(): pass\n')
    (root / 'linked.py').symlink_to(outside)
    (root / 'engine.py').write_text("raise RuntimeError('must not execute')\ndef load(): pass\n")
    (root / 'broken.py').write_text('def broken(')
    result = candidate_index(root)
    assert 'linked.py' not in result['files']
    assert any(row['symbol'] == 'load' for row in result['definitions'])
    assert result['syntax_errors'][0]['path'] == 'broken.py'


def test_search_uses_character_offsets_and_paginates(tmp_path):
    path = tmp_path / 'engine.py'
    source = '# 字符\nx = target\ny = target\nz = target\n'
    path.write_text(source)
    first = search_source(path, 'target', limit=2)
    assert [r['line'] for r in first['matches']] == [2, 3]
    assert all(source[r['offset']:].startswith('target') for r in first['matches'])
    second = search_source(path, 'target', offset=first['next_offset'])
    assert second['matches'][0]['line'] == 4
    assert second['next_offset'] is None
    assert search_source(path, 'TARGET')['matches'] == []


def test_search_is_bounded_confined_and_audited(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'engine.py').write_text('needle\n' * 100)
    trace = tmp_path / 'commands.jsonl'
    tools = AgentTools(work, trace)
    result = tools.execute('search_source', {'path': 'engine.py', 'query': 'needle', 'limit': 3})
    assert len(result['matches']) == 3
    assert result['next_offset'] is not None
    with pytest.raises(ValueError):
        tools.execute('search_source', {'path': '../outside.py', 'query': 'needle'})
    with pytest.raises(ValueError):
        tools.execute('search_source', {'path': 'engine.py', 'query': ''})
    assert len(trace.read_text().splitlines()) == 3


def test_python_and_json_checks_validate_without_execution(tmp_path):
    tools = AgentTools(tmp_path, tmp_path / 'trace.jsonl')
    (tmp_path / 'engine.json').write_text('{"enabled": true, "optional": null}')
    assert tools.execute('check_source', {'path': 'engine.json'})['language'] == 'json'
    (tmp_path / 'engine.json').write_text('{"enabled": True}')
    with pytest.raises(json.JSONDecodeError):
        tools.execute('check_source', {'path': 'engine.json'})
    (tmp_path / 'engine.py').write_text("raise RuntimeError('do not execute')")
    assert tools.execute('check_source', {'path': 'engine.py'})['language'] == 'python'
    (tmp_path / 'engine.py').write_text('return 1')
    with pytest.raises(SyntaxError):
        tools.execute('check_source', {'path': 'engine.py'})


def test_codex_brief_contains_actual_source_navigation(tmp_path):
    (tmp_path / 'engine.py').write_text('def load():\n    pass\n')
    (tmp_path / 'helper.py').write_text('def helper():\n    pass\n')
    brief = _brief({'workspace': str(tmp_path)})
    assert 'helper.py' in brief['candidate_files']
    assert {r['symbol'] for r in brief['candidate_source_index']['definitions']} == {'load', 'helper'}


def test_device_rankings_cannot_hide_cpu_bottlenecks():
    device = [{'name': f'kernel_{i}', 'calls': 1, 'self_device_seconds': 10-i} for i in range(8)]
    cpu = [{'name': 'host_allocation', 'calls': 90, 'self_cpu_seconds': 0.7},
           {'name': 'host_sync', 'calls': 20, 'self_cpu_seconds': 1.5}]
    brief = build_optimization_brief({'performance_observation': {
        'top_device_ops': device, 'top_device_operator_ops': device, 'top_cpu_ops': cpu}})
    evidence = brief['operator_evidence']
    assert evidence['top_cpu_ops'][0]['name'] == 'host_sync'
    assert evidence['top_cpu_ops'][0]['seconds'] == 1.5
    assert len(evidence['top_device_ops']) == 5
    assert len(evidence['top_device_operator_ops']) == 5
    assert 'overlap' in evidence['scope']


def test_mcp_exposes_search_and_validates_json(tmp_path, monkeypatch, capsys):
    import io
    from vibeforge.agent.codex_submit import serve

    work = tmp_path / 'work'
    work.mkdir()
    (work / 'engine.py').write_text('def generate(): return 1\n')
    (work / 'engine.json').write_text('{"enabled": true}')
    schema = tmp_path / 'schema.json'
    schema.write_text('{"type": "object"}')
    requests = [{'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}]
    for name, args in [('search_candidate', {'path': 'engine.py', 'query': 'generate'}),
                       ('search_candidate', {'path': '../schema.json', 'query': 'type'}),
                       ('check_candidate', {'path': 'engine.json'})]:
        requests.append({'jsonrpc': '2.0', 'id': len(requests)+1, 'method': 'tools/call',
                         'params': {'name': name, 'arguments': args}})
    monkeypatch.setattr('sys.stdin', io.StringIO('\n'.join(json.dumps(r) for r in requests)))
    serve(schema, tmp_path / 'result.json', workspace=work, trace=tmp_path / 'trace.jsonl')
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert 'search_candidate' in {t['name'] for t in replies[0]['result']['tools']}
    result = json.loads(replies[1]['result']['content'][0]['text'])
    assert result['matches'][0]['offset'] == 4
    assert replies[2]['result']['isError'] is True
    assert replies[3]['result']['isError'] is False
    assert json.loads(replies[3]['result']['content'][0]['text'])['language'] == 'json'


def test_missing_paths_explain_evidence_and_candidate_namespaces(tmp_path):
    work, evidence = tmp_path / 'work', tmp_path / 'evidence'
    work.mkdir()
    (evidence / 'iterations').mkdir(parents=True)
    (evidence / 'iterations' / 'result.json').write_text('{}')
    (work / 'engine.json').write_text('{}')
    tools = AgentTools(work, tmp_path / 'trace.jsonl', evidence)
    with pytest.raises(ValueError, match='Use read_evidence'):
        tools.execute('read_file', {'path': 'iterations/result.json'})
    with pytest.raises(ValueError, match='Use read_evidence'):
        tools.execute('list_evidence', {'path': 'iterations/result.json'})
    with pytest.raises(ValueError, match="Matching workspace-relative file names: \\['engine.json'\\]"):
        tools.execute('read_file', {'path': 'configs/engine.json'})


def test_revision_brief_preserves_input_lengths():
    brief = _brief({'candidate_feedback': {'correctness_precheck': {
        'passed': False, 'checks': [{'case': 'concurrent_request_isolation',
            'passed': False, 'input_lengths': [8, 128], 'observed_batch_sizes': [2, 2]}]}}})
    check = brief['candidate_feedback']['correctness_precheck']['checks'][0]
    assert check['input_lengths'] == [8, 128]


def test_brief_exposes_loaded_symbols_without_recommending_changes():
    index = {'symbols': ['Model.forward', 'backend.generate'], 'scope': 'Loaded definitions'}
    brief = build_optimization_brief({'performance_observation': {'implementation_index': index}})
    assert brief['implementation_index'] == index


def test_evidence_json_pointer_returns_only_selected_facts(tmp_path):
    root = tmp_path / 'evidence'
    root.mkdir()
    record = {'large_dump': [1] * 10000, 'checks': [{'passed': False, 'metric': 1.25}],
              'a/b': {'~key': True}, '': 'empty key'}
    (root / 'report.json').write_text(json.dumps(record))
    tools = AgentTools(tmp_path / 'work', tmp_path / 'trace.jsonl', root)
    result = tools.execute('read_evidence', {'path': 'report.json', 'pointer': '/checks/0'})
    assert json.loads(result) == record['checks'][0]
    assert len(result) < 100
    assert json.loads(tools.execute('read_evidence', {'path': 'report.json', 'pointer': '/a~1b/~0key'})) is True
    assert json.loads(tools.execute('read_evidence', {'path': 'report.json', 'pointer': '/'})) == 'empty key'
    for pointer in ['/checks/01', '/checks/-1', '/checks/2', '/bad~2escape', 'checks', '/missing']:
        with pytest.raises(ValueError):
            tools.execute('read_evidence', {'path': 'report.json', 'pointer': pointer})


def test_context_pointer_is_read_only_and_stays_bounded(tmp_path):
    context = tmp_path / '.vibeforge'
    context.mkdir()
    content = json.dumps({'model': {'name': 'test'}, 'large': 'x' * 9000})
    (context / 'context.json').write_text(content)
    tools = AgentTools(tmp_path, tmp_path / 'trace.jsonl')
    assert json.loads(tools.execute('read_file', {'path': '.vibeforge/context.json', 'pointer': '/model'})) == {'name': 'test'}
    result = tools.execute('read_file', {'path': '.vibeforge/context.json', 'pointer': '/large', 'limit': 100})
    assert len(result) < 250 and 'Next page: offset=100' in result
    assert (context / 'context.json').read_text() == content


def test_refinement_brief_keeps_current_candidate_identity_and_patch():
    target = {'candidate': 'candidate-v2', 'stable_parent': 'baseline-v1',
              'hypothesis': 'Original experiment', 'source_patch': '+changed_line'}
    brief = _brief({'refinement_target': target, 'candidate_engine_config': {'use_cache': False}})
    assert brief['mode'] == 'refinement'
    assert brief['refinement_target'] == target
    assert brief['candidate_engine_config'] == {'use_cache': False}


def test_refinement_context_does_not_label_candidate_as_stable(paths, toy_config, tmp_path):
    from vibeforge.agent.context import create_context
    from vibeforge.storage.atomic import write_json
    from vibeforge.storage.database import Database
    parent, candidate = paths.release('baseline'), tmp_path / 'candidate'
    parent.mkdir()
    candidate.mkdir()
    for root in [parent, candidate]:
        (root / 'engine.py').write_text('value = 1\n')
    write_json(parent / 'engine.json', {'request_scheduler': 'serialized_lock', 'use_cache': True})
    write_json(candidate / 'engine.json', {'request_scheduler': 'microbatch', 'use_cache': False})
    write_json(paths.data / 'profiles/profile.json', {'model': {}, 'hardware': {}, 'environment': {}})
    database = Database(paths.data / 'state/context-test.sqlite3')
    try:
        database.put('refine', {'campaign_id': 'refine', 'stable_ref': 'baseline', 'profile_hash': 'profile'})
        context = create_context(database, paths, 'refine', toy_config, candidate, 1)
        assert context['stable_engine_config']['request_scheduler'] == 'serialized_lock'
        assert context['stable_engine_config']['use_cache'] is True
        assert context['candidate_engine_config']['request_scheduler'] == 'microbatch'
        assert context['candidate_engine_config']['use_cache'] is False
    finally:
        database.close()


def test_invalid_config_write_fails_before_mutation_and_can_be_repaired(tmp_path):
    config = {'backend': 'transformers', 'model_path': '/models/frozen', 'device': 'cuda:0',
              'dtype': 'bfloat16', 'max_context': 4096, 'use_cache': True}
    path = tmp_path / 'engine.json'
    original = json.dumps(config)
    path.write_text(original)
    tools = AgentTools(tmp_path, tmp_path / 'trace.jsonl', runtime_policy={
        'optimization': {'allow_precision_changes': False, 'allowed_surfaces': ['cache_policy']}})
    with pytest.raises(ValueError, match='file is unchanged.*model_path'):
        tools.execute('write_file', {'path': 'engine.json', 'content': '{"backend":"transformers"}'})
    assert path.read_text() == original
    with pytest.raises(ValueError, match='file is unchanged'):
        tools.execute('replace_text', {'path': 'engine.json', 'old': 'bfloat16', 'new': 'float32'})
    assert path.read_text() == original
    with pytest.raises(ValueError, match='file is unchanged'):
        tools.execute('write_file', {'path': 'engine.json', 'content': '{broken'})
    assert path.read_text() == original
    tools.execute('replace_text', {'path': 'engine.json', 'old': '"use_cache": true',
                                  'new': '"use_cache": true, "cache_implementation": "static"'})
    assert json.loads(path.read_text()) == {**config, 'cache_implementation': 'static'}


def test_config_write_guard_honors_authorized_precision_search(tmp_path):
    from vibeforge.contracts.precision import normalize_precision, precision_fields
    original = {'backend': 'toy', 'model_path': '/model', **precision_fields(normalize_precision({'compute_dtype': 'float32'}))}
    target = normalize_precision({'compute_dtype': 'bfloat16'})
    (tmp_path / 'engine.json').write_text(json.dumps(original))
    tools = AgentTools(tmp_path, tmp_path / 'trace.jsonl', runtime_policy={
        'optimization': {'allow_precision_changes': True, 'allowed_precisions': [target]}})
    candidate = {**original, **precision_fields(target)}
    tools.execute('write_file', {'path': 'engine.json', 'content': json.dumps(candidate)})
    assert json.loads((tmp_path / 'engine.json').read_text()) == candidate


def test_analysis_prompt_cannot_inherit_optimization_instructions():
    from vibeforge.agent.codex import _prompt
    prompt = _prompt('Choose an optimization and edit engine.py.', analysis_only=True)
    assert 'MODE: analysis-only' in prompt
    assert 'Choose an optimization and edit engine.py.' not in prompt
    assert 'replace_candidate' not in prompt
    assert 'No edit is required' in prompt
    assert 'record_result' in prompt
